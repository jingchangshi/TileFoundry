"""Bounded subprocesses for compiler/search workers; never fork a live NPU runtime."""
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import uuid


@dataclass(frozen=True)
class ProcessResult:
    command: tuple
    returncode: int
    timed_out: bool
    elapsed_seconds: float
    log_path: str
    stage: str

    @property
    def status(self):
        if self.timed_out:
            return "COMPILE_TIMEOUT" if self.stage == "compile" else "RUN_TIMEOUT"
        if self.returncode:
            return "COMPILE_FAILED" if self.stage == "compile" else "RUN_ERROR"
        return "OK"


@dataclass(frozen=True)
class WorkerCommand:
    """An exec worker, never a Python callback or a fork of the caller runtime.

    The worker writes a fresh JSON object to result_path. Check stages return
    {"failure": null|string}; profile returns samples_us, discarded_warmups,
    measurement_source and (for device samples) a nonempty sync_note. Commands
    share artifacts via explicit paths in argv, not live Python/NPU objects.
    Every result must echo run_id, stage and identity_key supplied through the
    TF_SEARCH_RUN_ID/STAGE/IDENTITY_KEY environment; TF_SEARCH_RESULT_PATH is the
    absolute output path. No implicit compile cache is used by this protocol.
    Logs and JSON results must use different paths. The caller owns device locks.
    """
    argv: tuple
    log_path: str
    result_path: str
    cwd: str = None
    env: dict = None

    def execute(self, *, stage, timeout_s, deadline, settings):
        path = Path(self.result_path)
        if path.resolve() == Path(self.log_path).resolve():
            raise ValueError("worker log and result paths must differ")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.unlink(missing_ok=True)  # Never accept a previous invocation's receipt.
        env = dict(os.environ if self.env is None else self.env)
        env.update(settings)
        run_id = uuid.uuid4().hex
        env["TF_SEARCH_RUN_ID"] = run_id
        env["TF_SEARCH_STAGE"] = stage
        env["TF_SEARCH_RESULT_PATH"] = str(path.resolve())
        result = run_process(self.argv, stage=stage, timeout_s=timeout_s,
                             log_path=self.log_path, deadline=deadline,
                             cwd=self.cwd, env=env)
        if result.status != "OK":
            return result, None
        with path.open() as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict):
            raise ValueError("worker result must be a JSON object")
        if (payload.get("run_id") != run_id or payload.get("stage") != stage
                or payload.get("identity_key") != settings["TF_SEARCH_IDENTITY_KEY"]):
            raise ValueError("worker receipt does not match this invocation and identity")
        return result, payload


class Deadline:
    """One monotonic budget shared by all work within a case/search round."""
    def __init__(self, seconds, *, clock=time.monotonic):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("deadline must be finite and positive")
        self.clock = clock
        self.expires = clock() + seconds

    def remaining(self, limit=None):
        remaining = max(0.0, self.expires - self.clock())
        return remaining if limit is None else min(remaining, limit)


def run_process(command, *, stage, timeout_s, log_path, deadline=None, cwd=None, env=None):
    """Run an argv worker with an enforced wall deadline and retained native logs.

    Each worker owns a new process group. Timeout terminates only that group and reaps
    the worker, including its compiler subprocesses. The caller must hold the device
    flock for device workers; a suspected device hang must stop the search. A caller
    must never use this helper to signal an existing/unrelated process.
    """
    if stage not in ("compile", "candidate", "oracle", "numeric", "launch", "profile"):
        raise ValueError("unknown worker stage")
    if not command or isinstance(command, (str, bytes)):
        raise ValueError("command must be a nonempty argv sequence")
    if not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("timeout must be finite and positive")
    limit = timeout_s if deadline is None else deadline.remaining(timeout_s)
    started = time.monotonic()
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    timed_out = False
    with path.open("wb") as log:
        if limit <= 0:
            log.write(b"Shared deadline exhausted before starting worker\n")
            return ProcessResult(tuple(command), -1, True, 0.0, str(path), stage)
        process = subprocess.Popen(command, cwd=cwd, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        try:
            process.wait(timeout=limit)
        except subprocess.TimeoutExpired:
            timed_out = True
            # The process group is created above for this invocation only.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                pass
            finally:
                # The leader may have exited while a compiler child ignored TERM.
                # Kill the owned group even when wait() already reaped the leader.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
    return ProcessResult(tuple(command), process.returncode, timed_out,
                         time.monotonic() - started, str(path), stage)
