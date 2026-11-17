"""Stateless agy CLI backend with streaming liveness watchdog and mock backend."""

from __future__ import annotations

import asyncio
import json
import math
import os
import re
import shlex
import shutil
import signal
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

RE_503_CAPACITY = re.compile(
    r"UNAVAILABLE\s*\(code\s*503\)(?::\s*No capacity available)?",
    re.IGNORECASE,
)
RE_STREAM_IDLE_LOG = re.compile(
    r"(?:run\.go:\d+\]\s*Run:\s*attempt\s+\d+\s+failed.*stream idle timeout|stream idle timeout)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class StreamWatchdogConfig:
    """Dual-timer streaming liveness and log-tailing watchdog settings for agy workers."""

    startup_timeout_sec: float = 60.0
    idle_stall_timeout_sec: float = 60.0
    active_ceiling_timeout_sec: float = 360.0
    max_503_hits: int = 2
    max_sandbox_tool_calls: int = 2
    max_retries: int = 1
    retry_backoff_sec: float = 1.0


@dataclass(frozen=True)
class GenerationRequest:
    """Single stateless LLM generation request routed by worker role."""

    role: str
    system_prompt: str
    user_prompt: str
    json_schema: dict[str, object] | None = None
    effort: str | None = None


class KernelLLMBackend(Protocol):
    """Protocol implemented by both live AgyCliLLMBackend and MockLLMBackend."""

    async def generate(self, req: GenerationRequest) -> str:
        """Generate response text or JSON string for a GenerationRequest."""


_CODE_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)

CREATIVE_RESPONSE_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {
        "physical_hypothesis": {"type": "string"},
        "target_local_metric": {"type": "string"},
        "dsl_source": {"type": "string"},
    },
    "required": ["physical_hypothesis", "target_local_metric", "dsl_source"],
    "additionalProperties": False,
}


def extract_python_dsl(raw_response: str) -> str:
    """Extract clean Python DSL source from markdown fences, JSON envelope, or raw text."""
    stripped = raw_response.strip()
    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                if isinstance(payload.get("structured_output"), dict):
                    dsl = payload["structured_output"].get("dsl_source")
                    if isinstance(dsl, str) and dsl.strip():
                        return dsl.strip()
                if isinstance(payload.get("dsl_source"), str) and payload["dsl_source"].strip():
                    return payload["dsl_source"].strip()
                if isinstance(payload.get("response"), str) and payload["response"].strip():
                    stripped = payload["response"].strip()
        except json.JSONDecodeError:
            pass
    matches = _CODE_FENCE_RE.findall(stripped)
    if matches:
        return max(matches, key=len).strip()
    return stripped


def extract_creative_payload(raw_response: str) -> dict[str, str]:
    """Parse structured JSON response or markdown fallback from creative explore worker."""
    stripped = raw_response.strip()
    if stripped.startswith("{"):
        try:
            obj = json.loads(stripped)
            if isinstance(obj, dict):
                if isinstance(obj.get("structured_output"), dict):
                    obj = obj["structured_output"]
                elif isinstance(obj.get("response"), str) and obj["response"].strip().startswith("{"):
                    inner = json.loads(obj["response"])
                    if isinstance(inner, dict):
                        obj = inner
                if isinstance(obj.get("dsl_source"), str):
                    return {
                        "physical_hypothesis": str(obj.get("physical_hypothesis", "")),
                        "target_local_metric": str(obj.get("target_local_metric", "")),
                        "dsl_source": extract_python_dsl(str(obj["dsl_source"])),
                    }
        except json.JSONDecodeError:
            pass
    return {
        "physical_hypothesis": "unstructured_fallback",
        "target_local_metric": "latency_us",
        "dsl_source": extract_python_dsl(raw_response),
    }


def _resolve_agy_binary(explicit_bin: str | None = None) -> str:
    candidate = (explicit_bin or os.environ.get("TILEFOUNDRY_NPU_AGY_BIN", "")).strip()
    if candidate:
        resolved = shutil.which(candidate) or (candidate if Path(candidate).is_file() else None)
        if resolved:
            return resolved
    on_path = shutil.which("agy")
    if on_path:
        return on_path
    user_local = Path.home() / ".local" / "bin" / "agy"
    if user_local.is_file():
        return str(user_local)
    raise FileNotFoundError(
        "Cannot locate 'agy' binary. Set TILEFOUNDRY_NPU_AGY_BIN in tilefoundry-npu env.sh."
    )


def _normalize_agy_common_args(raw_common: str) -> list[str]:
    """Strip conflicting print, format, log, effort, and workspace flags safely."""
    tokens = shlex.split(raw_common)
    cleaned: list[str] = []
    skip_next = False
    value_flags = (
        "--output-format",
        "--json-schema",
        "--model",
        "--add-dir",
        "--agent",
        "--log-file",
        "--effort",
        "--print-timeout",
    )
    eq_prefixes = tuple(f"{flag}=" for flag in ("--print", "--prompt", *value_flags))
    for tok in tokens:
        if skip_next:
            skip_next = False
            continue
        if tok in ("--print", "-p", "--prompt"):
            continue
        if tok.startswith(eq_prefixes):
            continue
        if tok in value_flags:
            skip_next = True
            continue
        cleaned.append(tok)
    return cleaned


class _AgyLogTailer:
    """Incremental log tailer detecting upstream 503 capacity and stream idle failures."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._offset = 0
        self._partial = ""
        self.count_503 = 0
        self.count_idle_timeout = 0

    def poll(self) -> list[str]:
        """Read newly written log lines and return matched error event tags."""
        if not self.path.exists():
            return []
        try:
            with self.path.open("r", encoding="utf-8", errors="replace") as fh:
                fh.seek(self._offset)
                chunk = fh.read()
                self._offset = fh.tell()
        except OSError:
            return []

        if not chunk:
            return []

        text = self._partial + chunk
        lines = text.splitlines(keepends=True)
        if lines and not lines[-1].endswith(("\n", "\r")):
            self._partial = lines.pop()
        else:
            self._partial = ""

        events: list[str] = []
        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue
            if RE_503_CAPACITY.search(line):
                self.count_503 += 1
                events.append("503_no_capacity")
            if RE_STREAM_IDLE_LOG.search(line):
                self.count_idle_timeout += 1
                events.append("stream_idle_timeout")
        return events


class _AttemptFailure(Exception):
    """Internal wrapper pairing a worker failure with retry eligibility."""

    def __init__(self, exc: Exception, *, retriable: bool) -> None:
        super().__init__(str(exc))
        self.exc = exc
        self.retriable = retriable


async def _terminate_process_group(proc: asyncio.subprocess.Process) -> None:
    """Terminate an agy subprocess and its entire session process group cleanly."""
    try:
        pgid: int | None = os.getpgid(proc.pid)
    except (ProcessLookupError, PermissionError, OSError):
        pgid = proc.pid

    if proc.returncode is None:
        try:
            if pgid is not None:
                os.killpg(pgid, signal.SIGTERM)
            else:
                proc.terminate()
        except (ProcessLookupError, PermissionError, OSError):
            pass

        try:
            await asyncio.wait_for(proc.wait(), timeout=1.0)
        except asyncio.TimeoutError:
            try:
                if pgid is not None:
                    os.killpg(pgid, signal.SIGKILL)
                else:
                    proc.kill()
            except (ProcessLookupError, PermissionError, OSError):
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=1.0)
            except (asyncio.TimeoutError, ProcessLookupError):
                pass

    if pgid is not None:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass


class AgyCliLLMBackend:
    """Stateless headless wrapper around local agy CLI inside an empty sandbox directory."""

    DEFAULT_COMMON_ARGS = (
        "--input-format text --disable-slash-commands "
        "--sandbox --mode plan --dangerously-skip-permissions"
    )
    DEFAULT_EXPLOIT_MODEL = "gemini-4-hydrogen-medium"
    DEFAULT_EXPLORE_MODEL = "gemini-4-hydrogen-high"
    DEFAULT_ESCALATE_MODEL = "gemini-4-hydrogen-xhigh"

    def __init__(
        self,
        bin_path: str | None = None,
        exploit_model: str | None = None,
        explore_model: str | None = None,
        common_args: str | None = None,
        max_concurrency: int | None = None,
        hard_timeout_sec: float | None = None,
        watchdog: StreamWatchdogConfig | None = None,
        escalate_model: str | None = None,
    ) -> None:
        self._bin = _resolve_agy_binary(bin_path)
        self._exploit_model = (
            exploit_model
            if exploit_model is not None
            else os.environ.get("TILEFOUNDRY_NPU_AGY_EXPLOIT_MODEL", "")
        ).strip() or self.DEFAULT_EXPLOIT_MODEL
        self._explore_model = (
            explore_model
            if explore_model is not None
            else os.environ.get("TILEFOUNDRY_NPU_AGY_EXPLORE_MODEL", "")
        ).strip() or self.DEFAULT_EXPLORE_MODEL
        self._escalate_model = (
            escalate_model
            if escalate_model is not None
            else os.environ.get("TILEFOUNDRY_NPU_AGY_ESCALATE_MODEL", "")
        ).strip() or self.DEFAULT_ESCALATE_MODEL

        raw_common = (
            common_args
            if common_args is not None
            else os.environ.get("TILEFOUNDRY_NPU_AGY_COMMON_ARGS", "")
        ).strip() or self.DEFAULT_COMMON_ARGS
        self._common_args = _normalize_agy_common_args(raw_common)

        raw_conc = os.environ.get("TILEFOUNDRY_NPU_AGY_MAX_CONCURRENCY", "").strip()
        conc = max_concurrency or (int(raw_conc) if raw_conc else 3)
        self._sem = asyncio.Semaphore(max(1, conc))

        self._explicit_watchdog = watchdog is not None or hard_timeout_sec is not None
        if watchdog is not None:
            self._watchdog = watchdog
        elif hard_timeout_sec is not None:
            ceiling = float(hard_timeout_sec)
            self._watchdog = StreamWatchdogConfig(
                startup_timeout_sec=min(60.0, ceiling),
                idle_stall_timeout_sec=min(60.0, ceiling),
                active_ceiling_timeout_sec=ceiling,
            )
        else:
            self._watchdog = self.watchdog_for_model(self._exploit_model)
        self._hard_timeout = self._watchdog.active_ceiling_timeout_sec

    @property
    def watchdog(self) -> StreamWatchdogConfig:
        """Return active StreamWatchdogConfig."""
        return self._watchdog

    def select_model(self, role: str, *, effort: str | None = None) -> str:
        """Return target model ID for role and optional effort escalation."""
        if role == "explore_creative":
            if (effort or "").strip().lower() == "xhigh":
                return self._escalate_model
            return self._explore_model
        return self._exploit_model

    def watchdog_for_model(self, model_id: str) -> StreamWatchdogConfig:
        """Return explicit or tier-adaptive StreamWatchdogConfig for model_id."""
        if self._explicit_watchdog:
            return self._watchdog
        lower = model_id.strip().lower()
        if lower.endswith("-xhigh") or "xhigh" in lower:
            return StreamWatchdogConfig(
                startup_timeout_sec=60.0,
                idle_stall_timeout_sec=300.0,
                active_ceiling_timeout_sec=540.0,
            )
        if lower.endswith("-high") or "high" in lower:
            return StreamWatchdogConfig(
                startup_timeout_sec=60.0,
                idle_stall_timeout_sec=180.0,
                active_ceiling_timeout_sec=360.0,
            )
        return StreamWatchdogConfig(
            startup_timeout_sec=60.0,
            idle_stall_timeout_sec=120.0,
            active_ceiling_timeout_sec=240.0,
        )

    def select_effort(
        self,
        req: GenerationRequest,
        effort_override: str | None = None,
    ) -> str:
        """Select reasoning effort from override, request field, or worker role."""
        if effort_override is not None and effort_override.strip():
            return effort_override.strip()
        if req.effort is not None and req.effort.strip():
            return req.effort.strip()
        return "high" if req.role == "explore_creative" else "medium"

    @staticmethod
    def _sanitized_env(sandbox_dir: str) -> dict[str, str]:
        stripped_prefixes = ("AGY_CONVERSATION", "AGY_PROJECT", "CLAUDECODE", "MCP_")
        env = {
            k: v
            for k, v in os.environ.items()
            if not any(k.startswith(p) for p in stripped_prefixes)
        }
        env["PWD"] = sandbox_dir
        return env

    def build_command(
        self,
        req: GenerationRequest,
        full_prompt: str,
        *,
        log_file: str | Path | None = None,
        effort_override: str | None = None,
    ) -> list[str]:
        """Build isolated agy stream-json CLI argument list without workspace mounts."""
        effort = self.select_effort(req, effort_override=effort_override)
        model_id = self.select_model(req.role, effort=effort)
        cfg = self.watchdog_for_model(model_id)
        resolved_log = (
            str(log_file)
            if log_file is not None
            else str(Path(tempfile.gettempdir()) / "tf_agy_worker.log")
        )
        print_timeout_sec = max(1, int(math.ceil(cfg.active_ceiling_timeout_sec)))
        cmd = [
            self._bin,
            "--model",
            model_id,
            *self._common_args,
            "--output-format",
            "stream-json",
            f"--log-file={resolved_log}",
            f"--effort={effort}",
            f"--print-timeout={print_timeout_sec}s",
        ]
        if req.json_schema is not None:
            cmd.extend(["--json-schema", json.dumps(req.json_schema)])
        cmd.append(f"--print={full_prompt}")
        return cmd

    def _extract_stream_output(
        self,
        *,
        req: GenerationRequest,
        model_id: str,
        exit_code: int | None,
        result_payload: dict[str, object] | None,
        text_deltas: list[str],
        stderr_lines: list[str],
        tailer: _AgyLogTailer,
    ) -> str:
        stderr_text = "\n".join(stderr_lines).strip()
        if result_payload is not None:
            status = str(result_payload.get("status", "SUCCESS")).upper()
            if status == "SUCCESS" and exit_code == 0:
                structured = result_payload.get("structured_output")
                if req.json_schema is not None and structured is not None:
                    if isinstance(structured, (dict, list)):
                        return json.dumps(structured)
                    if isinstance(structured, str) and structured.strip():
                        return structured.strip()
                resp = result_payload.get("response")
                if isinstance(resp, str) and resp.strip():
                    return resp.strip()
                delta_text = "".join(text_deltas).strip()
                return delta_text

            err_detail = str(result_payload.get("error") or status or f"exit_{exit_code}")
            if stderr_text:
                err_detail = f"{err_detail}: {stderr_text}"
            is_retriable = (
                tailer.count_503 > 0
                or tailer.count_idle_timeout > 0
                or bool(RE_503_CAPACITY.search(err_detail))
                or bool(RE_STREAM_IDLE_LOG.search(err_detail))
            )
            raise _AttemptFailure(
                RuntimeError(f"agy failed (model={model_id}): {err_detail}"),
                retriable=is_retriable,
            )

        if exit_code == 0 and text_deltas:
            return "".join(text_deltas).strip()

        is_retriable = (
            tailer.count_503 > 0
            or tailer.count_idle_timeout > 0
            or bool(RE_503_CAPACITY.search(stderr_text))
            or bool(RE_STREAM_IDLE_LOG.search(stderr_text))
        )
        raise _AttemptFailure(
            RuntimeError(f"agy exited with {exit_code} (model={model_id}): {stderr_text}"),
            retriable=is_retriable,
        )

    async def _run_single_attempt(
        self,
        req: GenerationRequest,
        full_prompt: str,
        *,
        effort_override: str | None = None,
    ) -> str:
        effort = self.select_effort(req, effort_override=effort_override)
        model_id = self.select_model(req.role, effort=effort)
        cfg = self.watchdog_for_model(model_id)
        with (
            tempfile.TemporaryDirectory(prefix="tf_agy_worker_") as sandbox_dir,
            tempfile.TemporaryDirectory(prefix="tf_agy_log_") as log_dir,
        ):
            log_file = Path(log_dir) / "agy_worker.log"
            cmd = self.build_command(
                req,
                full_prompt,
                log_file=log_file,
                effort_override=effort_override,
            )
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=sandbox_dir,
                env=self._sanitized_env(sandbox_dir),
                start_new_session=True,
                limit=16 * 1024 * 1024,
            )
            if proc.stdout is None or proc.stderr is None:
                await _terminate_process_group(proc)
                raise _AttemptFailure(
                    RuntimeError("agy subprocess pipes unavailable"),
                    retriable=False,
                )

            queue: asyncio.Queue[tuple[str, str | None]] = asyncio.Queue()

            async def _pump(stream: asyncio.StreamReader, tag: str) -> None:
                try:
                    while True:
                        raw = await stream.readline()
                        if not raw:
                            break
                        await queue.put((tag, raw.decode("utf-8", errors="replace")))
                except asyncio.CancelledError:
                    raise
                finally:
                    await queue.put((tag, None))

            pump_out = asyncio.create_task(_pump(proc.stdout, "stdout"))
            pump_err = asyncio.create_task(_pump(proc.stderr, "stderr"))

            tailer = _AgyLogTailer(log_file)
            start_ts = time.monotonic()
            last_activity_ts = start_ts
            got_init = False
            active_tool_steps: set[int] = set()
            anon_tool_calls = 0
            text_deltas: list[str] = []
            result_payload: dict[str, object] | None = None
            stderr_lines: list[str] = []
            stdout_eof = False
            stderr_eof = False
            poll_interval = max(
                0.01,
                min(
                    0.05,
                    cfg.startup_timeout_sec / 4.0,
                    cfg.idle_stall_timeout_sec / 4.0,
                    cfg.active_ceiling_timeout_sec / 4.0,
                ),
            )

            try:
                while not (stdout_eof and stderr_eof):
                    tag = ""
                    line: str | None = None
                    try:
                        tag, line = await asyncio.wait_for(queue.get(), timeout=poll_interval)
                    except asyncio.TimeoutError:
                        pass

                    if tag == "stdout":
                        if line is None:
                            stdout_eof = True
                        else:
                            stripped = line.strip()
                            if stripped:
                                try:
                                    ev = json.loads(stripped)
                                except json.JSONDecodeError:
                                    ev = None
                                if isinstance(ev, dict):
                                    ev_type = ev.get("event")
                                    if ev_type == "init":
                                        got_init = True
                                        last_activity_ts = time.monotonic()
                                    elif ev_type == "step_update":
                                        raw_su = ev.get("step_update")
                                        su = raw_su if isinstance(raw_su, dict) else ev
                                        got_init = True
                                        last_activity_ts = time.monotonic()
                                        if su.get("step_type") == "tool":
                                            step_idx = su.get("step_index")
                                            if isinstance(step_idx, int) and not isinstance(
                                                step_idx, bool
                                            ):
                                                active_tool_steps.add(step_idx)
                                            elif su.get("state") != "DONE":
                                                anon_tool_calls += 1
                                            if (
                                                len(active_tool_steps) + anon_tool_calls
                                            ) > cfg.max_sandbox_tool_calls:
                                                await _terminate_process_group(proc)
                                                raise _AttemptFailure(
                                                    RuntimeError("sandbox_tool_loop_aborted"),
                                                    retriable=True,
                                                )
                                        delta = su.get("text_delta")
                                        if isinstance(delta, str) and delta:
                                            text_deltas.append(delta)
                                    elif ev_type == "result":
                                        raw_res = ev.get("result")
                                        result_payload = raw_res if isinstance(raw_res, dict) else ev
                                        got_init = True
                                        last_activity_ts = time.monotonic()
                    elif tag == "stderr":
                        if line is None:
                            stderr_eof = True
                        else:
                            cleaned = line.rstrip("\r\n")
                            if cleaned:
                                stderr_lines.append(cleaned)

                    log_events = tailer.poll()
                    if "stream_idle_timeout" in log_events:
                        await _terminate_process_group(proc)
                        raise _AttemptFailure(
                            RuntimeError("agy_stream_idle_timeout"),
                            retriable=True,
                        )
                    if tailer.count_503 >= cfg.max_503_hits:
                        await _terminate_process_group(proc)
                        raise _AttemptFailure(
                            RuntimeError(f"agy_503_no_capacity_x{tailer.count_503}"),
                            retriable=True,
                        )

                    if proc.returncode is None and result_payload is None:
                        now = time.monotonic()
                        if (now - start_ts) > cfg.active_ceiling_timeout_sec:
                            await _terminate_process_group(proc)
                            raise _AttemptFailure(
                                TimeoutError(
                                    f"agy active_ceiling_timeout ({cfg.active_ceiling_timeout_sec}s) "
                                    f"exceeded (role={req.role}, model={model_id})"
                                ),
                                retriable=False,
                            )
                        if not got_init and (now - start_ts) > cfg.startup_timeout_sec:
                            await _terminate_process_group(proc)
                            raise _AttemptFailure(
                                TimeoutError(
                                    f"agy startup_timeout ({cfg.startup_timeout_sec}s) "
                                    f"exceeded (role={req.role}, model={model_id})"
                                ),
                                retriable=True,
                            )
                        if (now - last_activity_ts) > cfg.idle_stall_timeout_sec:
                            await _terminate_process_group(proc)
                            raise _AttemptFailure(
                                TimeoutError(
                                    f"agy idle_stall_timeout ({cfg.idle_stall_timeout_sec}s) "
                                    f"exceeded (role={req.role}, model={model_id})"
                                ),
                                retriable=True,
                            )

                await proc.wait()
                tailer.poll()
                return self._extract_stream_output(
                    req=req,
                    model_id=model_id,
                    exit_code=proc.returncode,
                    result_payload=result_payload,
                    text_deltas=text_deltas,
                    stderr_lines=stderr_lines,
                    tailer=tailer,
                )
            finally:
                for task in (pump_out, pump_err):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(pump_out, pump_err, return_exceptions=True)
                await _terminate_process_group(proc)

    async def generate(
        self,
        req: GenerationRequest,
        *,
        effort_override: str | None = None,
    ) -> str:
        """Execute agy in an empty sandbox with streaming liveness watchdog and retries."""
        full_prompt = (
            "[STATELESS GENERATION CONTRACT: Do NOT read/write files or run shell/MCP tools. "
            "Respond strictly based on the prompt below.]\n\n"
            f"### SYSTEM INSTRUCTIONS\n{req.system_prompt}\n\n"
            f"### USER INPUT\n{req.user_prompt}\n"
        )
        effort = self.select_effort(req, effort_override=effort_override)
        model_id = self.select_model(req.role, effort=effort)
        cfg = self.watchdog_for_model(model_id)
        max_retries = max(0, cfg.max_retries)
        async with self._sem:
            for attempt_idx in range(max_retries + 1):
                try:
                    return await self._run_single_attempt(
                        req,
                        full_prompt,
                        effort_override=effort_override,
                    )
                except _AttemptFailure as failure:
                    if failure.retriable and attempt_idx < max_retries:
                        if cfg.retry_backoff_sec > 0.0:
                            await asyncio.sleep(cfg.retry_backoff_sec)
                        continue
                    raise failure.exc from None
        raise RuntimeError("unreachable retry termination")


@dataclass
class MockLLMBackend:
    """Deterministic scriptable LLM backend for offline orchestrator testing."""

    responses: dict[str, list[str]] = field(default_factory=dict)
    default_response: str = ""
    calls: list[GenerationRequest] = field(default_factory=list)
    routed_models: list[str] = field(default_factory=list)
    routed_efforts: list[str] = field(default_factory=list)

    async def generate(
        self,
        req: GenerationRequest,
        *,
        effort_override: str | None = None,
    ) -> str:
        """Record request and return next queued response for req.role."""
        self.calls.append(req)
        if effort_override and effort_override.strip():
            effort = effort_override.strip()
        elif req.effort and req.effort.strip():
            effort = req.effort.strip()
        else:
            effort = "high" if req.role == "explore_creative" else "medium"
        if req.role == "explore_creative":
            model_id = (
                AgyCliLLMBackend.DEFAULT_ESCALATE_MODEL
                if effort.lower() == "xhigh"
                else AgyCliLLMBackend.DEFAULT_EXPLORE_MODEL
            )
        else:
            model_id = AgyCliLLMBackend.DEFAULT_EXPLOIT_MODEL
        self.routed_models.append(model_id)
        self.routed_efforts.append(effort)
        queue = self.responses.get(req.role)
        if queue:
            return queue.pop(0)
        return self.default_response


def create_llm_backend_from_env() -> KernelLLMBackend:
    """Instantiate configured LLM backend from TILEFOUNDRY_NPU_LLM_BACKEND."""
    kind = os.environ.get("TILEFOUNDRY_NPU_LLM_BACKEND", "agy_cli").strip().lower()
    if kind in ("agy_cli", "agy", ""):
        return AgyCliLLMBackend()
    if kind == "mock":
        return MockLLMBackend()
    raise ValueError(f"unsupported TILEFOUNDRY_NPU_LLM_BACKEND={kind!r}")


__all__ = [
    "AgyCliLLMBackend",
    "CREATIVE_RESPONSE_SCHEMA",
    "GenerationRequest",
    "KernelLLMBackend",
    "MockLLMBackend",
    "RE_503_CAPACITY",
    "RE_STREAM_IDLE_LOG",
    "StreamWatchdogConfig",
    "create_llm_backend_from_env",
    "extract_creative_payload",
    "extract_python_dsl",
]
