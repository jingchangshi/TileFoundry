"""Stateless LLM providers for Ascend kernel evolution."""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import math
import os
import random
import re
import shlex
import shutil
import signal
import tempfile
import time
from collections.abc import Awaitable, Callable, Iterable, Iterator, Mapping
from contextlib import suppress
from dataclasses import dataclass, field, replace
from http.client import HTTPException, HTTPResponse
from pathlib import Path
from string import Formatter
from threading import Event
from typing import Literal, Protocol, TypeAlias
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

RE_503_CAPACITY = re.compile(r"UNAVAILABLE\s*\(code\s*503\)(?::\s*No capacity available)?", re.I)
RE_STREAM_IDLE_LOG = re.compile(r"(?:run\.go:\d+\]\s*Run:\s*attempt\s+\d+\s+failed.*stream idle timeout|stream idle timeout)", re.I)
RE_SSE_DISPATCH = re.compile(r"streamGenerateContent\?alt=sse", re.I)
logger = logging.getLogger(__name__)
_LLM_ENV_PREFIX, _AGY_ENV_PREFIX = "TILEFOUNDRY_NPU_LLM_", "TILEFOUNDRY_NPU_AGY_"
_REASONING_ROLES = frozenset({"explore_creative", "escalate_trackb"})
PURE_SYNTHESIS_ROLES = frozenset({"explore_creative", "exploit_rule", "syntax_repair", "escalate_trackb"})
_EFFORT_LEVELS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
_TimeoutStage: TypeAlias = Literal["startup", "idle_stall", "active_ceiling", "after_eof"]
_CREATIVE_DEFAULT_CEILING_SEC, _PARADIGM_SHIFT_CEILING_SEC = 480.0, 720.0
_REGIME_HEADER_RE = re.compile(r"^###\s*TARGET REGIME\s*\n\s*([A-Za-z0-9_.-]+)\s*:", re.M)
_CODE_FENCE_RE = re.compile(r"```(?:python|cpp|c\+\+|cce|diff)?\s*\n(.*?)```", re.DOTALL | re.I)
_EXECUTABLE_AST_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Assign, ast.AnnAssign, ast.Import, ast.ImportFrom)
_CPP_CODE_MARKERS = ("__aicore__", "__global__", "__simd_vf__", "__simt_vf__", "__simd_callee__", "__simt_callee__", "#include", "constexpr")
_BROKEN_PYTHON_HEAD_RE = re.compile(
    r"^\s*(?:(?:async\s+)?def\s+[A-Za-z_]\w*\s*\(|class\s+[A-Za-z_]\w*\s*[:(]|import\s+[A-Za-z_][\w.]*(?:\s*,|\s+as|\s*$)"
    r"|from\s+[A-Za-z_.][\w.]*\s+import\s+|@[A-Za-z_][\w.]*(?:\s*\(|\s*$))", re.M,
)
_CREATIVE_HEADER_KEY_RE = re.compile(r"^\s*#\s*(target_bottleneck|structural_hypothesis|physical_hypothesis|target_local_metric)\s*:\s*(.+?)\s*$", re.I)
_STATELESS_INSTRUCTION = "[STATELESS GENERATION CONTRACT: Do NOT invoke ANY tool calls. Answer directly in a single turn using ONLY the prompt context below.]"
_HARD_NO_TOOL_DIRECTIVE = (
    "[CRITICAL HARD DIRECTIVE: ZERO TOOL CALLS ALLOWED. Your previous attempt was terminated immediately "
    "for attempting a tool call. Do NOT invoke ANY tool (forbid read_file, view_file, write_file, "
    "write_to_file, run_command, grep_search, list_dir, and invoke_subagent). Output your final answer directly in plain text or JSON right now.]"
)
_CLI_PROXY_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")
_CLI_TELEMETRY_FORMAT = (
    "LLM_CALL_TELEMETRY backend=%s role=%s model=%s ttft_sec=%.3f elapsed_sec=%.3f "
    "status=%s proxy_configured=%s timeout_stage=%s since_last_activity_sec=%.3f"
)
_SSE_LINE_END = re.compile(rb"\r\n|\r|\n")
_CODEX_DISABLED_FEATURES = ("shell_tool", "unified_exec", "multi_agent", "multi_agent_v2", "apps", "plugins", "hooks", "browser_use", "image_generation", "view_image")


@dataclass(frozen=True)
class StreamWatchdogConfig:
    """Dual-timer streaming liveness and log-tailing watchdog settings for CLI workers."""

    startup_timeout_sec: float = 60.0
    idle_stall_timeout_sec: float = 60.0
    active_ceiling_timeout_sec: float = 360.0
    max_503_hits: int = 2
    max_sandbox_tool_calls: int = 0
    max_retries: int = 1
    retry_backoff_sec: float = 1.0
    capacity_503_max_retries: int | None = None
    capacity_503_backoff_sec: tuple[float, ...] = ()
    capacity_503_jitter_sec: float = 0.0


def _config_value(overrides: Mapping[str, str] | None, *keys: str, default: str = "") -> str:
    for source in (overrides or {}, os.environ):
        for key in keys:
            if value := source.get(key, "").strip():
                return value
    return default


def _capacity_503_policy_from_env(overrides: Mapping[str, str] | None = None) -> tuple[int, tuple[float, ...], float]:
    """Read 503 capacity retry count, backoff schedule, and jitter from environment."""
    try:
        max_503_retries = int(_config_value(overrides, f"{_LLM_ENV_PREFIX}MAX_503_RETRIES", default="4"))
    except ValueError:
        raise ValueError("TILEFOUNDRY_NPU_LLM_MAX_503_RETRIES must be an integer") from None
    if max_503_retries < 0:
        raise ValueError("TILEFOUNDRY_NPU_LLM_MAX_503_RETRIES must be non-negative")
    backoff_vals: list[float] = []
    for part in _config_value(overrides, f"{_LLM_ENV_PREFIX}503_BACKOFF_SEC", default="5.0,15.0,35.0,60.0").split(","):
        if cleaned := part.strip():
            try:
                val = float(cleaned)
            except ValueError:
                raise ValueError("TILEFOUNDRY_NPU_LLM_503_BACKOFF_SEC entries must be numeric") from None
            if not math.isfinite(val) or val < 0.0:
                raise ValueError("TILEFOUNDRY_NPU_LLM_503_BACKOFF_SEC entries must be non-negative")
            backoff_vals.append(val)
    return max_503_retries, tuple(backoff_vals or (5.0, 15.0, 35.0, 60.0)), 3.0


def _with_default_503_policy(watchdog: StreamWatchdogConfig, overrides: Mapping[str, str] | None) -> StreamWatchdogConfig:
    if watchdog.capacity_503_max_retries is None and watchdog.max_retries > 0:
        max_503, backoff_503, jitter_503 = _capacity_503_policy_from_env(overrides)
        return replace(watchdog, capacity_503_max_retries=max_503, capacity_503_backoff_sec=backoff_503, capacity_503_jitter_sec=jitter_503)
    return watchdog


def sanitize_effort(effort: str | None, *, default: str = "medium") -> str:
    """Return the requested effort label, using the default for empty input."""
    return (effort or "").strip() or default


@dataclass(frozen=True)
class GenerationRequest:
    """Single stateless LLM generation request routed by worker role."""

    role: str
    system_prompt: str
    user_prompt: str
    effort: str | None = None
    active_timeout_sec: float | None = None
    paradigm_shift: bool = False
    role_hint: str = ""
    session_scope: str = ""
    regime_tag: str = ""


def _is_creative_request(req: GenerationRequest) -> bool:
    return req.role == "explore_creative" or "explore_creative" in req.role_hint.lower() or "creative" in req.role_hint.lower()


def _is_paradigm_shift_request(req: GenerationRequest) -> bool:
    return req.paradigm_shift or any(t in f"{req.role_hint}\n{req.system_prompt}\n{req.user_prompt}".lower() for t in ("paradigm_shift", "paradigmshift", "full_slice_rewrite"))


def _request_session_scope(req: GenerationRequest) -> str:
    if req.session_scope.strip():
        return req.session_scope.strip()
    if req.regime_tag.strip():
        return req.regime_tag.strip()
    return m.group(1).strip() if (m := _REGIME_HEADER_RE.search(req.user_prompt)) is not None else "default"


@dataclass(frozen=True, init=False)
class RoleRoutingConfig:
    """Select a model and timeout from the worker role and cap effort explicitly."""

    exploit_model: str
    explore_model: str
    escalate_model: str
    repair_model: str
    max_effort: str = "xhigh"
    exploit_timeout_sec: float = 360.0
    reasoning_timeout_sec: float = 420.0
    repair_timeout_sec: float = 240.0

    def __init__(
        self, exploit_model: str, explore_model: str, escalate_model: str, repair_model: str,
        max_effort: str = "xhigh", exploit_timeout_sec: float | None = None,
        reasoning_timeout_sec: float = 420.0, repair_timeout_sec: float | None = None,
    ) -> None:
        """Bound repair by an explicit exploit budget, else keep its own default."""
        eff_exploit = RoleRoutingConfig.exploit_timeout_sec if exploit_timeout_sec is None else exploit_timeout_sec
        eff_repair = (RoleRoutingConfig.repair_timeout_sec if exploit_timeout_sec is None else eff_exploit) if repair_timeout_sec is None else repair_timeout_sec
        if max_effort.lower() not in _EFFORT_LEVELS:
            raise ValueError("LLM max effort must be a supported effort level")
        if any(not math.isfinite(t) or t <= 0 for t in (eff_exploit, reasoning_timeout_sec, eff_repair)):
            raise ValueError("LLM role timeouts must be finite and positive")
        for k, v in (
            ("exploit_model", exploit_model), ("explore_model", explore_model), ("escalate_model", escalate_model),
            ("repair_model", repair_model), ("max_effort", max_effort), ("exploit_timeout_sec", eff_exploit),
            ("reasoning_timeout_sec", reasoning_timeout_sec), ("repair_timeout_sec", eff_repair),
        ):
            object.__setattr__(self, k, v)

    def model_for_role(self, role: str) -> str:
        """Return the model configured for this worker role."""
        return {"explore_creative": self.explore_model, "escalate_trackb": self.escalate_model, "syntax_repair": self.repair_model}.get(role, self.exploit_model)

    def timeout_for_role(self, role: str) -> float:
        """Return the time budget for this worker role in seconds."""
        return self.reasoning_timeout_sec if role in _REASONING_ROLES else (self.repair_timeout_sec if role == "syntax_repair" else self.exploit_timeout_sec)

    def effort_for_request(self, req: GenerationRequest, override: str | None = None, *, default: str = "") -> str:
        """Apply the ceiling to recognized levels and preserve provider-specific labels."""
        effort = sanitize_effort(override, default=sanitize_effort(req.effort, default=default))
        ceiling = self.max_effort.lower()
        return ceiling if (effort.lower() in _EFFORT_LEVELS and _EFFORT_LEVELS.index(effort.lower()) > _EFFORT_LEVELS.index(ceiling)) else effort

    @classmethod
    def from_env(cls, overrides: Mapping[str, str] | None = None, *, backend: str | None = None) -> RoleRoutingConfig:
        """Load role settings with explicit values taking precedence over the environment."""
        kind = (backend or _config_value(overrides, f"{_LLM_ENV_PREFIX}BACKEND", default="agy_cli")).lower()
        is_agy = kind in ("agy_cli", "agy")

        def pick(key: str, default: str, *, agy_alias: bool = False) -> str:
            keys = (f"{_LLM_ENV_PREFIX}{key}", f"{_AGY_ENV_PREFIX}{key}") if (agy_alias and is_agy) else (f"{_LLM_ENV_PREFIX}{key}",)
            return _config_value(overrides, *keys, default=default)

        exploit = pick("EXPLOIT_MODEL", AgyCliLLMBackend.DEFAULT_EXPLOIT_MODEL if is_agy else ("mock" if kind == "mock" else ""), agy_alias=True)
        explore = pick("EXPLORE_MODEL", AgyCliLLMBackend.DEFAULT_EXPLORE_MODEL if is_agy else exploit, agy_alias=True)
        raw_t = pick("EXPLOIT_TIMEOUT_SEC", "")
        exploit_t = float(raw_t) if raw_t else cls.exploit_timeout_sec
        return cls(
            exploit_model=exploit, explore_model=explore, escalate_model=pick("ESCALATE_MODEL", explore, agy_alias=True),
            repair_model=pick("REPAIR_MODEL", exploit),
            max_effort=_config_value(overrides, f"{_LLM_ENV_PREFIX}MAX_EFFORT", f"{_LLM_ENV_PREFIX}EFFORT_CAP", default="high" if is_agy else "xhigh"),
            exploit_timeout_sec=exploit_t, reasoning_timeout_sec=float(pick("REASONING_TIMEOUT_SEC", "360" if is_agy else "420")),
            repair_timeout_sec=exploit_t if raw_t else cls.repair_timeout_sec,
        )


class KernelLLMBackend(Protocol):
    """Stateless generation interface shared by remote, subprocess, and offline providers."""

    async def generate(self, req: GenerationRequest) -> str: ...


class LLMGenerationError(RuntimeError):
    """Structured LLM generation error carrying fault plane, code, and retriability."""

    def __init__(self, message: str, *, fault_plane: str = "LLM_TRANSPORT", fault_code: str = "LLM_TIMEOUT_OR_503", retriable: bool = False) -> None:
        super().__init__(message)
        self.fault_plane, self.fault_code, self.retriable = fault_plane, fault_code, retriable


class LLMTimeoutError(TimeoutError, LLMGenerationError):
    """Structured LLM timeout error preserving TimeoutError and LLMGenerationError identity."""

    def __init__(self, message: str, *, fault_plane: str = "LLM_TRANSPORT", fault_code: str = "LLM_TIMEOUT_OR_503", retriable: bool = False) -> None:
        TimeoutError.__init__(self, message)
        self.fault_plane, self.fault_code, self.retriable = fault_plane, fault_code, retriable


class _ProseRefusalMessage(str):
    fault_plane, fault_code, retriable = "SYNTAX_AST", "LLM_PROSE_NO_CODE", False


_PROSE_REFUSAL_ERROR = _ProseRefusalMessage(
    "llm_prose_refusal_no_code: response contains prose without executable DSL or SEARCH/REPLACE hunks"
)


class ProseRefusalError(ValueError, LLMGenerationError):
    """Raised when an LLM reply contains prose without executable DSL or patch hunks."""

    def __init__(self, message: str = _PROSE_REFUSAL_ERROR) -> None:
        ValueError.__init__(self, message)
        self.fault_plane, self.fault_code, self.retriable = "SYNTAX_AST", "LLM_PROSE_NO_CODE", False


def _has_code_or_patch_structure(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if re.search(r"<{3,7}\s*SEARCH\b", stripped) or any(m in stripped for m in _CPP_CODE_MARKERS):
        return True
    try:
        tree = ast.parse(stripped)
    except SyntaxError:
        return bool(_BROKEN_PYTHON_HEAD_RE.search(stripped))
    return any(isinstance(node, _EXECUTABLE_AST_NODES) for node in ast.walk(tree))


def extract_python_dsl(raw_response: str) -> str:
    """Extract clean DSL or Track-B patch text, rejecting pure natural-language refusals."""
    stripped = raw_response.strip()
    if matches := _CODE_FENCE_RE.findall(stripped):
        if valid := [m.strip() for m in matches if _has_code_or_patch_structure(m)]:
            return max(valid, key=len)
        raise ProseRefusalError(_PROSE_REFUSAL_ERROR)
    if not _has_code_or_patch_structure(stripped):
        raise ProseRefusalError(_PROSE_REFUSAL_ERROR)
    return stripped


def extract_creative_payload(raw_response: str) -> dict[str, str]:
    """Parse leading metadata comment lines and fenced code from a creative worker."""
    first_fence = re.search(r"```", raw_response)
    if first_fence is not None:
        header_lines = raw_response[: first_fence.start()].splitlines()
    else:
        header_lines = []
        for raw_line in raw_response.splitlines():
            if not (s := raw_line.strip()):
                continue
            if not s.startswith("#"):
                break
            header_lines.append(raw_line)
    phys = metric = ""
    for line in header_lines:
        if (m := _CREATIVE_HEADER_KEY_RE.match(line)) is None or not (val := m.group(2).strip()):
            continue
        if m.group(1).lower() in ("structural_hypothesis", "physical_hypothesis"):
            phys = val
        elif m.group(1).lower() in ("target_bottleneck", "target_local_metric"):
            metric = val
    return {"physical_hypothesis": phys or "unstructured_fallback", "target_local_metric": metric or "latency_us", "dsl_source": extract_python_dsl(raw_response)}


def _resolve_agy_binary(explicit_bin: str | None = None) -> str:
    for cand in ((explicit_bin or os.environ.get("TILEFOUNDRY_NPU_AGY_BIN", "")).strip(), "agy"):
        if cand and (resolved := shutil.which(cand) or (cand if Path(cand).is_file() else None)):
            return os.path.abspath(resolved)
    if (user_local := Path.home() / ".local" / "bin" / "agy").is_file():
        return str(user_local)
    raise FileNotFoundError("Cannot locate 'agy' binary. Set TILEFOUNDRY_NPU_AGY_BIN in tilefoundry-npu env.sh.")


def _normalize_agy_common_args(raw_common: str) -> list[str]:
    """Strip conflicting print, format, log, effort, and workspace flags safely."""
    val_flags = ("--output-format", "--json-schema", "--model", "--add-dir", "--agent", "--log-file", "--effort", "--print-timeout")
    eq_prefixes = tuple(f"{f}=" for f in ("--print", "--prompt", *val_flags))
    cleaned: list[str] = []
    skip = False
    for tok in shlex.split(raw_common):
        if skip:
            skip = False
        elif tok in ("--print", "-p", "--prompt") or tok.startswith(eq_prefixes):
            continue
        elif tok in val_flags:
            skip = True
        else:
            cleaned.append(tok)
    return cleaned


class _AgyLogTailer:
    """Incremental log tailer detecting upstream 503 capacity and stream idle failures."""

    def __init__(self, path: Path) -> None:
        self.path, self._offset, self._partial = path, 0, ""
        self.count_503 = self.count_idle_timeout = 0
        self.saw_progress = self.sse_in_flight = False

    def poll(self) -> list[str]:
        """Read newly written log lines and return matched error event tags."""
        self.saw_progress = False
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
        lines = (self._partial + chunk).splitlines(keepends=True)
        self._partial = lines.pop() if lines and not lines[-1].endswith(("\n", "\r")) else ""
        events: list[str] = []
        for raw_line in lines:
            if not (line := raw_line.strip()):
                continue
            is_err = False
            if RE_503_CAPACITY.search(line):
                self.count_503 += 1
                events.append("503_no_capacity")
                is_err = True
            if RE_STREAM_IDLE_LOG.search(line):
                self.count_idle_timeout += 1
                events.append("stream_idle_timeout")
                is_err = True
            if not is_err:
                self.saw_progress = True
            if RE_SSE_DISPATCH.search(line):
                self.sse_in_flight = True
        return events


class _AttemptFailure(Exception):
    def __init__(
        self, exc: Exception, *, retriable: bool, ttft_sec: float | None = None,
        timeout_stage: _TimeoutStage | None = None, since_last_activity_sec: float | None = None,
        is_503: bool = False, is_tool_loop: bool = False,
    ) -> None:
        super().__init__(str(exc))
        self.exc, self.retriable, self.ttft_sec = exc, retriable, ttft_sec
        self.timeout_stage, self.since_last_activity_sec = timeout_stage, since_last_activity_sec
        self.is_503, self.is_tool_loop = is_503, is_tool_loop


async def _terminate_process_group(proc: asyncio.subprocess.Process) -> None:
    if proc.returncode is None:
        with suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGTERM)
        with suppress(TimeoutError):
            await asyncio.wait_for(proc.wait(), timeout=1.0)
    with suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGKILL)
    await proc.wait()


def _finalize_exhausted_failure(failure: _AttemptFailure) -> Exception:
    exc = failure.exc
    if isinstance(exc, LLMGenerationError):
        exc.retriable = False
        return exc
    if isinstance(exc, TimeoutError):
        return LLMTimeoutError(str(exc), fault_plane="LLM_TRANSPORT", fault_code="LLM_TIMEOUT_OR_503", retriable=False)
    detail = str(exc)
    is_t_or_503 = failure.is_503 or failure.timeout_stage is not None or "stream_idle_timeout" in detail or bool(RE_STREAM_IDLE_LOG.search(detail) or RE_503_CAPACITY.search(detail))
    return LLMGenerationError(detail, fault_plane="LLM_TRANSPORT", fault_code="LLM_TIMEOUT_OR_503" if is_t_or_503 else "LLM_CLI_ERROR", retriable=False)


async def _run_watched_subprocess(
    argv: list[str], *, cwd: str, env: Mapping[str, str], stdin: bytes | None, cfg: StreamWatchdogConfig,
    on_stdout_line: Callable[[str], None], on_poll: Callable[[asyncio.subprocess.Process], Awaitable[float | None]] | None = None,
    has_started: Callable[[], bool], idle_limit_fn: Callable[[], float] | None = None,
    timeout_msg_fn: Callable[[str, float], str], enforce_after_eof: bool = False, max_stderr_lines: int | None = None,
) -> tuple[int | None, list[str], float | None]:
    """Run a subprocess in its own process group with startup, idle, and ceiling watchdogs."""
    spawn = asyncio.create_task(asyncio.create_subprocess_exec(
        *argv, stdin=asyncio.subprocess.DEVNULL if stdin is None else asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, cwd=cwd, env=dict(env), start_new_session=True, limit=16 * 1024 * 1024,
    ))
    try:
        proc = await asyncio.shield(spawn)
    except asyncio.CancelledError:
        with suppress(OSError, ValueError):
            proc = await spawn
            await asyncio.gather(_terminate_process_group(proc), proc.communicate())
        raise
    except (OSError, ValueError):
        raise RuntimeError("Could not start the configured LLM CLI") from None
    if proc.stdout is None or proc.stderr is None:
        await _terminate_process_group(proc)
        raise _AttemptFailure(RuntimeError("subprocess pipes unavailable"), retriable=False)

    queue: asyncio.Queue[tuple[str, str | None]] = asyncio.Queue()

    async def _pump(stream: asyncio.StreamReader, tag: str) -> None:
        try:
            while raw := await stream.readline():
                await queue.put((tag, raw.decode("utf-8", errors="replace")))
        except asyncio.CancelledError:
            raise
        finally:
            await queue.put((tag, None))

    async def _feed_stdin() -> None:
        assert stdin is not None and proc.stdin is not None
        with suppress(BrokenPipeError, ConnectionResetError):
            proc.stdin.write(stdin); await proc.stdin.drain(); proc.stdin.close()

    tasks = [asyncio.create_task(_pump(proc.stdout, "stdout")), asyncio.create_task(_pump(proc.stderr, "stderr"))]
    if stdin is not None:
        tasks.append(asyncio.create_task(_feed_stdin()))
    start_ts = last_activity_ts = time.monotonic()
    ttft_sec: float | None = None
    stderr_lines: list[str] = []
    stdout_eof = stderr_eof = False
    poll_interval = max(0.01, min(0.05, cfg.startup_timeout_sec / 4.0, cfg.idle_stall_timeout_sec / 4.0, cfg.active_ceiling_timeout_sec / 4.0))

    try:
        while not (stdout_eof and stderr_eof):
            tag, line = "", None
            with suppress(asyncio.TimeoutError):
                tag, line = await asyncio.wait_for(queue.get(), timeout=poll_interval)
            if tag == "stdout":
                if line is None:
                    stdout_eof = True
                else:
                    if line.strip():
                        now_line = time.monotonic()
                        ttft_sec = (now_line - start_ts) if ttft_sec is None else ttft_sec
                        last_activity_ts = now_line
                    on_stdout_line(line)
            elif tag == "stderr":
                if line is None:
                    stderr_eof = True
                elif cleaned := line.rstrip("\r\n"):
                    stderr_lines.append(cleaned)
                    if max_stderr_lines is not None and len(stderr_lines) > max_stderr_lines:
                        stderr_lines = stderr_lines[-max_stderr_lines:]
            if on_poll is not None and (bumped := await on_poll(proc)) is not None:
                last_activity_ts = bumped
            now = time.monotonic()
            eff_idle = idle_limit_fn() if idle_limit_fn is not None else cfg.idle_stall_timeout_sec
            for triggered, stage, lim in (
                ((now - start_ts) > cfg.active_ceiling_timeout_sec, "active_ceiling", cfg.active_ceiling_timeout_sec),
                (not has_started() and (now - start_ts) > cfg.startup_timeout_sec, "startup", cfg.startup_timeout_sec),
                ((now - last_activity_ts) > eff_idle, "idle_stall", eff_idle),
            ):
                if triggered and (stage == "active_ceiling" or proc.returncode is None):
                    await _terminate_process_group(proc)
                    raise _AttemptFailure(TimeoutError(timeout_msg_fn(stage, lim)), retriable=True, ttft_sec=ttft_sec, timeout_stage=stage, since_last_activity_sec=now - last_activity_ts)
        if enforce_after_eof:
            try:
                await asyncio.wait_for(proc.wait(), timeout=max(0.0, cfg.active_ceiling_timeout_sec - (time.monotonic() - start_ts)))
            except asyncio.TimeoutError:
                t_at = time.monotonic()
                await _terminate_process_group(proc)
                raise _AttemptFailure(TimeoutError(timeout_msg_fn("after_eof", cfg.active_ceiling_timeout_sec)), retriable=True, ttft_sec=ttft_sec, timeout_stage="after_eof", since_last_activity_sec=t_at - last_activity_ts) from None
        else:
            await proc.wait()
        return proc.returncode, stderr_lines, ttft_sec
    finally:
        await _terminate_process_group(proc)
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


async def _run_with_503_retry(
    *, sem: asyncio.Semaphore, cfg: StreamWatchdogConfig, fallback_model: str, attempt_fn: Callable[[str | None], Awaitable[str]]
) -> str:
    max_retries = max(0, cfg.max_retries)
    max_503 = max_retries if cfg.capacity_503_max_retries is None else max(0, cfg.capacity_503_max_retries)
    retries_503 = retries_other = 0
    next_model: str | None = None
    async with sem:
        while True:
            try:
                return await attempt_fn(next_model)
            except _AttemptFailure as failure:
                if not failure.retriable:
                    raise _finalize_exhausted_failure(failure) from None
                if failure.is_503 and cfg.capacity_503_max_retries is not None:
                    if retries_503 >= max_503:
                        raise _finalize_exhausted_failure(failure) from None
                    sched = cfg.capacity_503_backoff_sec
                    wait_s = sched[min(retries_503, len(sched) - 1)] if sched else cfg.retry_backoff_sec
                    wait_s += random.uniform(0.0, cfg.capacity_503_jitter_sec) if cfg.capacity_503_jitter_sec > 0.0 else 0.0
                    retries_503 += 1
                    if retries_503 >= min(2, max_503) and fallback_model:
                        next_model = fallback_model
                    if wait_s > 0.0:
                        await asyncio.sleep(wait_s)
                    continue
                if retries_other < max_retries:
                    retries_other += 1
                    if cfg.retry_backoff_sec > 0.0:
                        await asyncio.sleep(cfg.retry_backoff_sec)
                    continue
                raise _finalize_exhausted_failure(failure) from None


def _raise_tool_loop_abort(*, retriable: bool) -> None:
    raise _AttemptFailure(
        LLMGenerationError("sandbox_tool_loop_aborted", fault_plane="LLM_TRANSPORT", fault_code="SANDBOX_TOOL_LOOP_ABORTED", retriable=False),
        retriable=retriable, is_tool_loop=True,
    )


class AgyCliLLMBackend:
    """Stateless headless wrapper around local agy CLI inside an empty sandbox directory."""

    DEFAULT_COMMON_ARGS = "--input-format text --disable-slash-commands --sandbox --mode plan --dangerously-skip-permissions"
    DEFAULT_EXPLOIT_MODEL = "gemini-4-hydrogen-low"
    DEFAULT_EXPLORE_MODEL = "gemini-4-hydrogen-high"
    DEFAULT_ESCALATE_MODEL = "gemini-4-hydrogen-high"

    def __init__(
        self, bin_path: str | None = None, exploit_model: str | None = None, explore_model: str | None = None,
        common_args: str | None = None, max_concurrency: int | None = None, hard_timeout_sec: float | None = None,
        watchdog: StreamWatchdogConfig | None = None, escalate_model: str | None = None, repair_model: str | None = None,
        *, routing: RoleRoutingConfig | None = None, fallback_model: str | None = None, overrides: Mapping[str, str] | None = None,
    ) -> None:
        self._overrides = dict(overrides) if overrides else None
        self._bin = _resolve_agy_binary(bin_path)
        model_ov = {
            f"{_LLM_ENV_PREFIX}{r}_MODEL": m
            for r, m in (("EXPLOIT", exploit_model), ("EXPLORE", explore_model), ("ESCALATE", escalate_model), ("REPAIR", repair_model))
            if m is not None
        }
        self._routing = routing or RoleRoutingConfig.from_env(model_ov, backend="agy_cli")
        raw_fb = (fallback_model if fallback_model is not None else _config_value(self._overrides, f"{_LLM_ENV_PREFIX}FALLBACK_MODEL")).strip()
        self._fallback_model = self._agy_model_id(raw_fb) if raw_fb else ""
        raw_common = (common_args if common_args is not None else os.environ.get("TILEFOUNDRY_NPU_AGY_COMMON_ARGS", "")).strip() or self.DEFAULT_COMMON_ARGS
        self._common_args = _normalize_agy_common_args(raw_common)
        raw_conc = os.environ.get("TILEFOUNDRY_NPU_AGY_MAX_CONCURRENCY", "").strip()
        self._sem = asyncio.Semaphore(max(1, max_concurrency or (int(raw_conc) if raw_conc else 3)))
        self._creative_sem = asyncio.Semaphore(1)
        self._consecutive_active_ceiling_timeouts: dict[tuple[str, str], int] = {}
        self._active_ceiling_cooldown: set[tuple[str, str]] = set()
        self._explicit_watchdog = watchdog is not None or hard_timeout_sec is not None
        if watchdog is not None:
            self._watchdog = _with_default_503_policy(watchdog, self._overrides)
        elif hard_timeout_sec is not None:
            self._watchdog = self._make_watchdog(float(hard_timeout_sec), idle_cap=60.0)
        else:
            self._watchdog = self.watchdog_for_role("exploit_rule")

    def _make_watchdog(self, ceiling: float, *, idle_cap: float) -> StreamWatchdogConfig:
        max_503, backoff_503, jitter_503 = _capacity_503_policy_from_env(self._overrides)
        return StreamWatchdogConfig(
            startup_timeout_sec=min(60.0, ceiling), idle_stall_timeout_sec=min(idle_cap, ceiling),
            active_ceiling_timeout_sec=ceiling, capacity_503_max_retries=max_503,
            capacity_503_backoff_sec=backoff_503, capacity_503_jitter_sec=jitter_503,
        )

    @property
    def watchdog(self) -> StreamWatchdogConfig:
        return self._watchdog

    @staticmethod
    def _agy_model_id(model_id: str) -> str:
        return re.sub(r"-xhigh$", "-high", model_id.strip(), flags=re.IGNORECASE)

    def select_model(self, role: str, *, effort: str | None = None) -> str:
        _ = effort
        return self._agy_model_id(self._routing.model_for_role(role))

    def watchdog_for_model(self, model_id: str) -> StreamWatchdogConfig:
        return self._watchdog if self._explicit_watchdog else self.watchdog_for_role("explore_creative" if "high" in self._agy_model_id(model_id).lower() else "exploit_rule")

    def watchdog_for_role(self, role: str) -> StreamWatchdogConfig:
        if self._explicit_watchdog:
            return self._watchdog
        ceiling = max(_CREATIVE_DEFAULT_CEILING_SEC, self._routing.timeout_for_role(role)) if role == "explore_creative" else self._routing.timeout_for_role(role)
        return self._make_watchdog(ceiling, idle_cap=180.0 if role in _REASONING_ROLES else 120.0)

    def watchdog_for_request(self, req: GenerationRequest) -> StreamWatchdogConfig:
        if self._explicit_watchdog:
            return self._watchdog
        eff_role = "explore_creative" if _is_creative_request(req) else req.role
        cfg = self.watchdog_for_role(eff_role)
        ceiling = max(_PARADIGM_SHIFT_CEILING_SEC, cfg.active_ceiling_timeout_sec) if _is_paradigm_shift_request(req) else cfg.active_ceiling_timeout_sec
        if req.active_timeout_sec is not None and math.isfinite(req.active_timeout_sec) and req.active_timeout_sec > 0.0:
            ceiling = float(req.active_timeout_sec)
        if ceiling == cfg.active_ceiling_timeout_sec:
            return cfg
        return replace(cfg, startup_timeout_sec=min(60.0, ceiling), idle_stall_timeout_sec=min(180.0 if eff_role in _REASONING_ROLES else 120.0, ceiling), active_ceiling_timeout_sec=ceiling)

    def select_effort(self, req: GenerationRequest, effort_override: str | None = None) -> str:
        model = self.select_model(req.role).lower()
        default = "high" if (req.role in _REASONING_ROLES or "carbon" in model or "gemini-pro-orchestrator" in model) else "medium"
        effort = self._routing.effort_for_request(req, effort_override, default=default)
        return "high" if effort.lower() in ("xhigh", "max") else effort

    @staticmethod
    def _sanitized_env(sandbox_dir: str) -> dict[str, str]:
        return {**{k: v for k, v in os.environ.items() if not k.startswith(("AGY_CONVERSATION", "AGY_PROJECT", "CLAUDECODE", "MCP_"))}, "PWD": sandbox_dir}

    def _common_args_for_role(self, role: str) -> list[str]:
        if role not in PURE_SYNTHESIS_ROLES:
            return list(self._common_args)
        filtered, skip = [], False
        for idx, tok in enumerate(self._common_args):
            if skip:
                skip = False
            elif tok == "--mode":
                skip = idx + 1 < len(self._common_args) and not self._common_args[idx + 1].startswith("-")
            elif tok != "--dangerously-skip-permissions" and not tok.startswith("--mode="):
                filtered.append(tok)
        return filtered

    def build_command(
        self, req: GenerationRequest, full_prompt: str, *, log_file: str | Path | None = None,
        effort_override: str | None = None, model_override: str | None = None,
    ) -> list[str]:
        """Build isolated agy stream-json CLI argument list without workspace mounts."""
        effort = self.select_effort(req, effort_override=effort_override)
        model_id = self._agy_model_id(model_override) if model_override else self.select_model(req.role, effort=effort)
        for suffix in ("-low", "-medium", "-high"):
            if model_id.endswith(suffix):
                effort = suffix[1:]
                break
        cfg = self.watchdog_for_request(req)
        resolved_log = str(log_file) if log_file is not None else str(Path(tempfile.gettempdir()) / "tf_agy_worker.log")
        return [
            self._bin, "--model", model_id, *self._common_args_for_role(req.role), "--output-format", "stream-json",
            f"--log-file={resolved_log}", f"--effort={effort}", f"--print-timeout={max(1, int(math.ceil(cfg.active_ceiling_timeout_sec)))}s", f"--print={full_prompt}",
        ]

    @staticmethod
    def _write_no_tool_sandbox_files(sandbox_dir: str) -> None:
        agents_dir = Path(sandbox_dir) / ".agents"
        (agents_dir / "rules").mkdir(parents=True, exist_ok=True)
        (agents_dir / "rules" / "no_tools.md").write_text("---\ntrigger: always_on\n---\nZero tool calls allowed in this stateless synthesis session. Output ONLY the requested header comments and fenced code block directly.\n", encoding="utf-8")
        (agents_dir / "hooks.json").write_text(json.dumps({"no-tools": {"PreToolUse": [{"matcher": "*", "hooks": [{"type": "command", "command": 'echo \'{"decision":"deny","reason":"Tools disabled in stateless synthesis mode"}\''}]}]}}, indent=2) + "\n", encoding="utf-8")

    async def _run_single_attempt(
        self, req: GenerationRequest, full_prompt: str, *, effort_override: str | None = None, model_override: str | None = None
    ) -> str:
        effort = self.select_effort(req, effort_override=effort_override)
        model_id = self._agy_model_id(model_override) if model_override else self.select_model(req.role, effort=effort)
        cfg = self.watchdog_for_request(req)
        with tempfile.TemporaryDirectory(prefix="tf_agy_worker_") as sandbox_dir, tempfile.TemporaryDirectory(prefix="tf_agy_log_") as log_dir:
            if req.role in PURE_SYNTHESIS_ROLES:
                self._write_no_tool_sandbox_files(sandbox_dir)
            log_file = Path(log_dir) / "agy_worker.log"
            cmd = self.build_command(req, full_prompt, log_file=log_file, effort_override=effort_override, model_override=model_override)
            tailer = _AgyLogTailer(log_file)
            got_init = pending_tool_abort = False
            active_tool_steps: set[int] = set()
            anon_tools = denied_tools = 0
            text_deltas: list[str] = []
            result_payload: dict[str, object] | None = None

            def _on_stdout(line: str) -> None:
                nonlocal got_init, anon_tools, denied_tools, result_payload, pending_tool_abort
                if not (stripped := line.strip()):
                    return
                try:
                    ev = json.loads(stripped)
                except json.JSONDecodeError:
                    return
                if not isinstance(ev, dict):
                    return
                ev_type = ev.get("event")
                if ev_type == "init":
                    got_init = True
                elif ev_type == "step_update":
                    su = ev["step_update"] if isinstance(ev.get("step_update"), dict) else ev
                    got_init = True
                    if su.get("step_type") == "tool":
                        err_obj = su.get("error")
                        hook_msg = (str(err_obj.get("message", "")) if isinstance(err_obj, Mapping) else str(err_obj or "")) or str(su.get("tool_info", "") or "")
                        if su.get("state") == "ERROR" and "denied by pre-tool hook" in hook_msg.lower():
                            denied_tools += 1
                            pending_tool_abort = pending_tool_abort or denied_tools >= 2
                        else:
                            step_idx = su.get("step_index")
                            if isinstance(step_idx, int) and not isinstance(step_idx, bool):
                                active_tool_steps.add(step_idx)
                            elif su.get("state") != "DONE" or cfg.max_sandbox_tool_calls == 0:
                                anon_tools += 1
                            pending_tool_abort = pending_tool_abort or (len(active_tool_steps) + anon_tools) > cfg.max_sandbox_tool_calls
                    if isinstance(delta := su.get("text_delta"), str) and delta:
                        text_deltas.append(delta)
                elif ev_type == "result":
                    result_payload = ev["result"] if isinstance(ev.get("result"), dict) else ev
                    got_init = True

            async def _on_poll(proc: asyncio.subprocess.Process) -> float | None:
                if pending_tool_abort:
                    await _terminate_process_group(proc)
                    _raise_tool_loop_abort(retriable=True)
                log_events = tailer.poll()
                if "stream_idle_timeout" in log_events:
                    await _terminate_process_group(proc)
                    raise _AttemptFailure(RuntimeError("agy_stream_idle_timeout"), retriable=True)
                if tailer.count_503 >= cfg.max_503_hits:
                    await _terminate_process_group(proc)
                    raise _AttemptFailure(RuntimeError(f"agy_503_no_capacity_x{tailer.count_503}"), retriable=True, is_503=True)
                return time.monotonic() if (tailer.saw_progress and not log_events) else None

            rc, stderr_lines, _ = await _run_watched_subprocess(
                cmd, cwd=sandbox_dir, env=self._sanitized_env(sandbox_dir), stdin=None, cfg=cfg, on_stdout_line=_on_stdout, on_poll=_on_poll,
                has_started=lambda: got_init or result_payload is not None,
                idle_limit_fn=lambda: max(cfg.idle_stall_timeout_sec, cfg.active_ceiling_timeout_sec) if tailer.sse_in_flight else cfg.idle_stall_timeout_sec,
                timeout_msg_fn=lambda stage, lim: f"agy {stage}_timeout ({lim}s) exceeded (role={req.role}, model={model_id})",
            )
            tailer.poll()
            err_prefix = ""
            if result_payload is not None:
                status = str(result_payload.get("status", "SUCCESS")).upper()
                if status == "SUCCESS" and rc == 0:
                    resp = result_payload.get("response")
                    return resp.strip() if isinstance(resp, str) and resp.strip() else "".join(text_deltas).strip()
                err_prefix = str(result_payload.get("error") or status or f"exit_{rc}")
            elif rc == 0 and text_deltas:
                return "".join(text_deltas).strip()
            stderr_text = "\n".join(stderr_lines).strip()
            combined = f"{err_prefix}: {stderr_text}" if (err_prefix and stderr_text) else (err_prefix or stderr_text)
            combined = f"{combined}\n{''.join(text_deltas)}"
            is_503 = tailer.count_503 > 0 or bool(RE_503_CAPACITY.search(combined))
            raise _AttemptFailure(RuntimeError(f"Agy generation failed with exit status {rc}"), retriable=is_503 or tailer.count_idle_timeout > 0 or bool(RE_STREAM_IDLE_LOG.search(combined)), is_503=is_503)

    async def generate(self, req: GenerationRequest, *, effort_override: str | None = None) -> str:
        """Execute agy in an empty sandbox with streaming liveness watchdog and retries."""
        full_prompt = (
            "[STATELESS GENERATION CONTRACT: Do NOT invoke ANY tool calls (forbid run_command, view_file, "
            "list_dir, grep_search, write_to_file, and invoke_subagent). Answer directly in a single turn "
            f"using ONLY the prompt context below.]\n\n### SYSTEM INSTRUCTIONS\n{req.system_prompt}\n\n### USER INPUT\n{req.user_prompt}\n"
        )
        cfg = self.watchdog_for_request(req)
        scope_key = (_request_session_scope(req), req.role)
        used_cooldown = scope_key in self._active_ceiling_cooldown
        self._active_ceiling_cooldown.discard(scope_key)
        cooldown_model, eff_effort = ("gemini-4-hydrogen-medium", "medium") if used_cooldown else (None, effort_override)
        base_model, tool_retry_used = self.select_model(req.role), False

        async def _attempt(retry_model_override: str | None) -> str:
            nonlocal tool_retry_used
            active_ov = retry_model_override or cooldown_model
            active_model = self._agy_model_id(active_ov) if active_ov else base_model
            prompt_for_attempt = full_prompt
            while True:
                try:
                    result = await self._run_single_attempt(req, prompt_for_attempt, effort_override=eff_effort, model_override=active_ov)
                    break
                except _AttemptFailure as failure:
                    if failure.is_tool_loop:
                        if not tool_retry_used and cfg.max_retries > 0:
                            tool_retry_used, prompt_for_attempt = True, f"{_HARD_NO_TOOL_DIRECTIVE}\n\n{full_prompt}"
                            continue
                        _raise_tool_loop_abort(retriable=False)
                    if failure.timeout_stage == "active_ceiling" and not used_cooldown and _is_creative_request(req) and active_model.lower() == "gemini-4-hydrogen-high":
                        cnt = self._consecutive_active_ceiling_timeouts.get(scope_key, 0) + 1
                        self._consecutive_active_ceiling_timeouts[scope_key] = 0 if cnt >= 2 else cnt
                        if cnt >= 2:
                            self._active_ceiling_cooldown.add(scope_key)
                    raise
            self._consecutive_active_ceiling_timeouts[scope_key] = 0
            return result

        return await _run_with_503_retry(sem=self._creative_sem if req.role in _REASONING_ROLES else self._sem, cfg=cfg, fallback_model=self._fallback_model, attempt_fn=_attempt)


class _NoHTTPRedirects(HTTPRedirectHandler):
    def redirect_request(
        self, req: object, fp: object, code: object, msg: object, headers: object, newurl: object
    ) -> None:
        return None


def _iter_sse_data(chunks: Iterable[bytes]) -> Iterator[bytes]:
    pending, data, first_line, prev_cr = b"", [], True, False
    for chunk in chunks:
        if prev_cr:
            chunk = chunk.removeprefix(b"\n")
        pending += chunk
        prev_cr = pending.endswith(b"\r")
        start = 0
        for match in _SSE_LINE_END.finditer(pending):
            line = pending[start : match.start()]
            start = match.end()
            if first_line:
                line = line.removeprefix(b"\xef\xbb\xbf")
                first_line = False
            if not line and data:
                yield b"\n".join(data)
                data.clear()
            elif line == b"data" or line.startswith(b"data:"):
                data.append(line.partition(b":")[2].removeprefix(b" "))
        pending = pending[start:]


def _reject_http_tools(message: Mapping[str, object], finish_reason: object = None) -> None:
    if message.get("tool_calls") or message.get("function_call") or finish_reason in ("tool_calls", "function_call"):
        raise RuntimeError("LLM HTTP response attempted a tool call")


def _read_sse_completion(chunks: Iterable[bytes]) -> str:
    text: list[str] = []
    for data in _iter_sse_data(chunks):
        if data.strip() == b"[DONE]":
            return "".join(text)
        if not data.strip():
            continue
        event = json.loads(data)
        if not isinstance(event, dict):
            raise ValueError
        if event.get("error"):
            raise RuntimeError("LLM HTTP stream reported an error")
        for idx, choice in enumerate(event.get("choices", ())):
            delta, finish = choice.get("delta") or {}, choice.get("finish_reason")
            _reject_http_tools(delta, finish)
            if choice.get("index", idx) != 0:
                continue
            if isinstance(content := delta.get("content"), str):
                text.append(content)
            if finish:
                return "".join(text)
    raise RuntimeError("LLM HTTP stream ended before completion")


def _read_http_chunks(response: HTTPResponse, deadline: float, cancelled: Event) -> Iterator[bytes]:
    while not cancelled.is_set() and time.monotonic() < deadline:
        chunk = response.read1(64 * 1024)
        if cancelled.is_set() or time.monotonic() >= deadline:
            break
        if not chunk:
            return
        yield chunk
    raise TimeoutError


class OpenAICompatHTTPBackend:
    """Send stateless chat completions using standard-library JSON and SSE decoding."""

    def __init__(
        self, routing: RoleRoutingConfig, *, api_base: str | None = None, api_key_env: str | None = None,
        max_concurrency: int = 4, pass_reasoning_effort: bool | None = None, stream: bool | None = None,
    ) -> None:
        self._routing = routing
        base = api_base or _config_value(None, f"{_LLM_ENV_PREFIX}API_BASE", default="http://127.0.0.1:8000/v1")
        try:
            url = urlsplit(base.strip())
            if url.scheme not in ("http", "https") or not url.hostname or url.username is not None or url.query or url.fragment:
                raise ValueError
        except ValueError:
            raise ValueError("LLM API base must be an HTTP(S) URL without credentials") from None
        path = url.path.rstrip("/") or "/v1"
        self._endpoint = url._replace(path=path if path.endswith("/chat/completions") else f"{path}/chat/completions").geturl()
        default_key = "OPENAI_API_KEY" if url.scheme == "https" and url.hostname == "api.openai.com" else ""
        self._key_var = api_key_env or _config_value(None, f"{_LLM_ENV_PREFIX}API_KEY_ENV", default=default_key)
        if self._key_var and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self._key_var):
            raise ValueError("LLM API key configuration must be an environment variable name")
        self._pass_effort = pass_reasoning_effort if pass_reasoning_effort is not None else _config_value(None, f"{_LLM_ENV_PREFIX}PASS_REASONING_EFFORT") == "1"
        self._stream = stream if stream is not None else _config_value(None, f"{_LLM_ENV_PREFIX}STREAM") == "1"
        self._sem = asyncio.Semaphore(max(1, max_concurrency))

    def _sync_post(self, body: dict[str, object], timeout_sec: float, cancelled: Event) -> str:
        deadline = time.monotonic() + timeout_sec
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream, application/json" if self._stream else "application/json"}
        if key := os.environ.get(self._key_var, "").strip():
            headers["Authorization"] = f"Bearer {key}"
        request = Request(self._endpoint, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
        try:
            with build_opener(_NoHTTPRedirects()).open(request, timeout=timeout_sec) as response:
                if cancelled.is_set():
                    raise TimeoutError
                chunks = _read_http_chunks(response, deadline, cancelled)
                if response.getheader("Content-Type", "").partition(";")[0].strip().lower() == "text/event-stream":
                    content = _read_sse_completion(chunks)
                else:
                    choice = json.loads(b"".join(chunks))["choices"][0]
                    _reject_http_tools(choice["message"], choice.get("finish_reason"))
                    content = choice["message"].get("content")
        except HTTPError as exc:
            exc.close()
            raise RuntimeError(f"LLM HTTP request failed with status {exc.code}") from None
        except URLError as exc:
            raise (TimeoutError("LLM HTTP request timed out") if isinstance(exc.reason, TimeoutError) else RuntimeError("LLM HTTP transport failed")) from None
        except TimeoutError:
            raise TimeoutError("LLM HTTP request timed out") from None
        except (OSError, HTTPException):
            raise RuntimeError("LLM HTTP transport failed") from None
        except (ValueError, KeyError, IndexError, TypeError, AttributeError):
            raise RuntimeError("LLM HTTP response has an invalid completion") from None
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("LLM HTTP response has no text completion")
        return content.strip()

    async def generate(self, req: GenerationRequest, *, effort_override: str | None = None) -> str:
        """Generate a text completion for a role without exposing any tools."""
        if not (model := self._routing.model_for_role(req.role)):
            raise ValueError("Configure an LLM model for the requested HTTP generation role")
        timeout = self._routing.timeout_for_role(req.role)
        body: dict[str, object] = {"model": model, "messages": [{"role": "system", "content": req.system_prompt}, {"role": "user", "content": req.user_prompt}]}
        if self._stream:
            body["stream"] = True
        if self._pass_effort and (effort := self._routing.effort_for_request(req, effort_override)):
            body["reasoning_effort"] = effort
        await self._sem.acquire()
        cancelled = Event()
        try:
            pending = asyncio.get_running_loop().run_in_executor(None, self._sync_post, body, timeout, cancelled)
        except RuntimeError:
            self._sem.release()
            raise
        pending.add_done_callback(lambda fut: (self._sem.release(), not fut.cancelled() and fut.exception()))
        try:
            return await asyncio.wait_for(asyncio.shield(pending), timeout)
        except TimeoutError:
            raise TimeoutError("LLM HTTP request timed out") from None
        finally:
            cancelled.set()


def _proxy_env(environ: Mapping[str, str]) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for key in _CLI_PROXY_KEYS:
        if val := (environ.get(f"{_LLM_ENV_PREFIX}{key}", "").strip() or environ.get(key, "").strip() or environ.get(key.lower(), "").strip()):
            resolved[key] = resolved[key.lower()] = val
    return resolved


def _redact_secrets(text: str) -> str:
    redacted = re.sub(r"(?i)\b((?:proxy-)?authorization\s*[=:]\s*)(?:bearer|basic|digest|token)\s+\S+", r"\1[redacted]", text)
    redacted = re.sub(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/=-]+", r"\1[redacted]", redacted)
    redacted = re.sub(r"(?i)\b((?:api[_-]?key|token|secret|password|authorization)\s*[=:]\s*)\S+", r"\1[redacted]", redacted)
    redacted = re.sub(r"(?i)\b(sk-[A-Za-z0-9_\-]{6,})\b", "[redacted]", redacted)
    return re.sub(r"(https?://[^:/\s]+:)[^@\s]+@", r"\1[redacted]@", redacted)


def _cli_watchdog_from_env(overrides: Mapping[str, str] | None = None) -> StreamWatchdogConfig:
    try:
        startup_sec = float(_config_value(overrides, f"{_LLM_ENV_PREFIX}STARTUP_TIMEOUT_SEC", default="90"))
        idle_sec = float(_config_value(overrides, f"{_LLM_ENV_PREFIX}IDLE_STALL_TIMEOUT_SEC", default="120"))
    except ValueError:
        raise ValueError("LLM CLI watchdog timeouts must be numeric") from None
    if not all(math.isfinite(v) and v > 0 for v in (startup_sec, idle_sec)):
        raise ValueError("LLM CLI watchdog timeouts must be finite and positive")
    max_503, backoff_503, jitter_503 = _capacity_503_policy_from_env(overrides)
    return StreamWatchdogConfig(
        startup_timeout_sec=startup_sec, idle_stall_timeout_sec=idle_sec,
        capacity_503_max_retries=max_503, capacity_503_backoff_sec=backoff_503, capacity_503_jitter_sec=jitter_503,
    )


def _claude_stream_text(event: Mapping[str, object], assistant_text: list[str]) -> str | None:
    ev_type = event.get("type")
    if ev_type == "assistant":
        msg = event.get("message")
        for block in (msg.get("content") if isinstance(msg, Mapping) else ()) or ():
            if isinstance(block, Mapping) and block.get("type") == "text" and isinstance(block.get("text"), str):
                assistant_text.append(block["text"])
        return None
    return event.get("result") if ev_type == "result" and isinstance(event.get("result"), str) else None


class SubprocessCliTemplateBackend:
    """Run a single generation in an empty working directory using an argument vector."""

    def __init__(
        self, kind: str, routing: RoleRoutingConfig, *, bin_path: str | None = None,
        template: str | None = None, max_concurrency: int = 2, watchdog: StreamWatchdogConfig | None = None,
        fallback_model: str | None = None, overrides: Mapping[str, str] | None = None,
    ) -> None:
        if kind not in ("claude_cli", "codex_cli", "custom_cli"):
            raise ValueError("Unknown LLM CLI profile")
        self._kind, self._routing = kind, routing
        self._overrides = dict(overrides) if overrides else None
        self._fallback_model = (fallback_model if fallback_model is not None else _config_value(self._overrides, f"{_LLM_ENV_PREFIX}FALLBACK_MODEL")).strip()
        self._watchdog = _with_default_503_policy(watchdog, self._overrides) if watchdog is not None else None
        self._template: list[str] = []
        self._template_fields: set[str] = set()
        configured_bin = bin_path or _config_value(self._overrides, f"{_LLM_ENV_PREFIX}CLI_BIN")
        if kind == "custom_cli":
            raw_template = template or _config_value(self._overrides, f"{_LLM_ENV_PREFIX}CLI_TEMPLATE")
            try:
                self._template = shlex.split(raw_template)
                for arg in self._template:
                    for _, name, spec, conv in Formatter().parse(arg):
                        if name is not None:
                            if name not in {"bin", "model", "prompt", "effort"} or spec or conv:
                                raise ValueError
                            self._template_fields.add(name)
                if not self._template:
                    raise ValueError
                executable = configured_bin if self._template[0] == "{bin}" else self._template[0]
                if self._template[0] != "{bin}" and ("{" in executable or "}" in executable):
                    raise ValueError
            except ValueError:
                raise ValueError("Invalid LLM CLI template") from None
            if not executable:
                raise ValueError("Configure an executable for the custom LLM CLI template")
        else:
            executable = configured_bin or kind.removesuffix("_cli")
        resolved = shutil.which(executable)
        if resolved is None:
            raise FileNotFoundError("Cannot locate the configured LLM CLI executable")
        self._bin = os.path.abspath(resolved)
        self._sem = asyncio.Semaphore(max(1, max_concurrency))

    def _command(self, model: str, effort: str, prompt: str) -> tuple[list[str], bytes | None]:
        if self._kind == "custom_cli":
            vals = {"bin": self._bin, "model": model, "effort": effort, "prompt": prompt}
            return [self._bin, *(arg.format_map(vals) for arg in self._template[1:])], (None if "prompt" in self._template_fields else prompt.encode("utf-8"))
        if self._kind == "claude_cli":
            argv = [
                self._bin, "--print", "--output-format", "stream-json", "--verbose", "--include-partial-messages",
                "--tools", "", "--strict-mcp-config", "--disable-slash-commands", "--no-session-persistence",
                "--setting-sources", "", "--permission-mode", "dontAsk", "--no-chrome",
                *(["--effort=" + effort] if effort else []),
            ]
        else:
            argv = [
                self._bin, "--no-daemon", "--ask-for-approval", "never", "exec", "--sandbox", "read-only",
                "--skip-git-repo-check", "--ephemeral", "--ignore-user-config", "--ignore-rules", "--json",
                "-c", 'web_search="disabled"', "-c", "mcp_servers={}",
                *(arg for feat in _CODEX_DISABLED_FEATURES for arg in ("--disable", feat)),
                *(("-c", f"model_reasoning_effort={json.dumps(effort)}") if effort else ()),
            ]
        if model:
            argv.append(f"--model={model}")
        if self._kind == "codex_cli":
            argv.append("-")
        return argv, prompt.encode("utf-8")

    @staticmethod
    def _codex_text(output: str) -> str:
        text = ""
        for line in output.splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                raise RuntimeError("Codex CLI returned an invalid generation event") from None
            if not isinstance(event, dict):
                raise RuntimeError("Codex CLI returned an invalid generation event")
            ev_type, item = event.get("type"), event.get("item")
            if ev_type in ("error", "turn.failed"):
                raise RuntimeError("Codex CLI reported a generation failure")
            if isinstance(item, dict):
                if item.get("type") not in ("agent_message", "reasoning"):
                    raise RuntimeError("Codex CLI attempted to use a tool during generation")
                if ev_type == "item.completed" and item.get("type") == "agent_message" and isinstance(item.get("text"), str):
                    text = item["text"]
        return text.strip()

    def watchdog_for_role(self, role: str) -> StreamWatchdogConfig:
        base, ceiling = self._watchdog or _cli_watchdog_from_env(self._overrides), self._routing.timeout_for_role(role)
        return replace(base, startup_timeout_sec=min(base.startup_timeout_sec, ceiling), idle_stall_timeout_sec=min(base.idle_stall_timeout_sec, ceiling), active_ceiling_timeout_sec=ceiling)

    async def _run_streaming(
        self, argv: list[str], stdin: bytes | None, cfg: StreamWatchdogConfig, *, role: str, model: str
    ) -> tuple[str, float | None]:
        with tempfile.TemporaryDirectory(prefix="tf_llm_worker_") as sandbox_dir:
            env = {**os.environ, "PWD": sandbox_dir, **_proxy_env(os.environ)}
            env.pop("CLAUDECODE", None)
            stdout_chunks, assistant_text = [], []
            result_text = stream_error = None
            stream_events_seen = corrupt_stream = saw_first_line = False

            def _on_stdout(line: str) -> None:
                nonlocal result_text, stream_error, stream_events_seen, corrupt_stream, saw_first_line
                if not (stripped := line.strip()):
                    return
                saw_first_line = True
                stdout_chunks.append(line)
                if self._kind != "claude_cli":
                    return
                try:
                    event: object = json.loads(stripped)
                except json.JSONDecodeError:
                    corrupt_stream = corrupt_stream or stripped.startswith(("{", "["))
                    return
                if isinstance(event, Mapping) and isinstance(event.get("type"), str):
                    stream_events_seen = True
                    if (extracted := _claude_stream_text(event, assistant_text)) is not None:
                        result_text = extracted
                    if event.get("type") == "result" and event.get("is_error") is True:
                        stream_error = str(event.get("result") or event.get("subtype") or "error")

            rc, stderr_lines, ttft_sec = await _run_watched_subprocess(
                argv, cwd=sandbox_dir, env=env, stdin=stdin, cfg=cfg, on_stdout_line=_on_stdout, has_started=lambda: saw_first_line,
                timeout_msg_fn=lambda stage, lim: f"LLM CLI generation timed out: {'active_ceiling' if stage == 'after_eof' else stage}_timeout ({lim}s) exceeded{' after stream EOF' if stage == 'after_eof' else ''} (role={role}, model={model})",
                enforce_after_eof=True, max_stderr_lines=40,
            )
            stderr_tail = _redact_secrets("\n".join(stderr_lines))[-2000:]
            raw_output = "".join(stdout_chunks)
            if rc or stream_error is not None:
                prefix = f"LLM CLI generation failed with exit status {rc}" if rc else f"LLM CLI generation failed: {_redact_secrets(stream_error or '')[:500]}"
                detail = f"{prefix}: {stderr_tail}" if stderr_tail else prefix
                blob = f"{stream_error or ''}\n{stderr_tail}\n{raw_output}"
                is_503 = bool(RE_503_CAPACITY.search(blob))
                raise _AttemptFailure(RuntimeError(detail), retriable=is_503 or bool(RE_STREAM_IDLE_LOG.search(blob)), is_503=is_503, ttft_sec=ttft_sec)
            if self._kind != "claude_cli":
                return raw_output.strip(), ttft_sec
            if output := (result_text or "".join(assistant_text)).strip():
                return output, ttft_sec
            if stream_events_seen or corrupt_stream:
                raise _AttemptFailure(RuntimeError(f"LLM CLI returned no assistant text: {stderr_tail}" if stderr_tail else "LLM CLI returned no assistant text"), retriable=True, ttft_sec=ttft_sec)
            return raw_output.strip(), ttft_sec

    async def generate(self, req: GenerationRequest, *, effort_override: str | None = None) -> str:
        """Route a generation request to the CLI and return its final response text."""
        model, effort = self._routing.model_for_role(req.role), self._routing.effort_for_request(req, effort_override)
        prompt = f"{_STATELESS_INSTRUCTION}\n\n### SYSTEM INSTRUCTIONS\n{req.system_prompt}\n\n### USER INPUT\n{req.user_prompt}\n"
        cfg, p_env = self.watchdog_for_role(req.role), _proxy_env(os.environ)
        proxy_cfg = "true" if any(k in p_env for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY")) else "false"

        async def _attempt(model_override: str | None) -> str:
            active_model = model_override or model
            argv, stdin = self._command(active_model, effort, prompt)
            start_ts = time.monotonic()
            try:
                output, ttft_sec = await self._run_streaming(argv, stdin, cfg, role=req.role, model=active_model)
                text = self._codex_text(output.strip()) if self._kind == "codex_cli" else output.strip()
                if not text:
                    raise _AttemptFailure(RuntimeError("LLM CLI returned no text completion"), retriable=False, ttft_sec=ttft_sec)
            except _AttemptFailure as failure:
                logger.info(
                    _CLI_TELEMETRY_FORMAT, self._kind, req.role, active_model, failure.ttft_sec if failure.ttft_sec is not None else -1.0,
                    time.monotonic() - start_ts, "timeout" if isinstance(failure.exc, TimeoutError) else "error", proxy_cfg,
                    failure.timeout_stage or "none", -1.0 if failure.since_last_activity_sec is None else failure.since_last_activity_sec,
                )
                raise
            elapsed = time.monotonic() - start_ts
            logger.info(_CLI_TELEMETRY_FORMAT, self._kind, req.role, active_model, ttft_sec if ttft_sec is not None else elapsed, elapsed, "ok", proxy_cfg, "none", -1.0)
            return text

        return await _run_with_503_retry(sem=self._sem, cfg=cfg, fallback_model=self._fallback_model, attempt_fn=_attempt)


@dataclass
class MockLLMBackend:
    """Deterministic scriptable LLM backend for offline orchestrator testing."""

    responses: dict[str, list[str]] = field(default_factory=dict)
    default_response: str = ""
    calls: list[GenerationRequest] = field(default_factory=list)
    routed_models: list[str] = field(default_factory=list)
    routed_efforts: list[str] = field(default_factory=list)
    routing: RoleRoutingConfig = field(
        default_factory=lambda: RoleRoutingConfig("mock-exploit", "mock-explore", "mock-escalate", "mock-repair")
    )

    async def generate(self, req: GenerationRequest, *, effort_override: str | None = None) -> str:
        self.calls.append(req)
        default = "high" if req.role in _REASONING_ROLES else "medium"
        self.routed_models.append(self.routing.model_for_role(req.role))
        self.routed_efforts.append(self.routing.effort_for_request(req, effort_override, default=default))
        queue = self.responses.get(req.role)
        if queue is None and req.role == "escalate_trackb":
            queue = self.responses.get("explore_creative")
        return queue.pop(0) if queue else self.default_response


def create_llm_backend_from_env(overrides: Mapping[str, str] | None = None) -> KernelLLMBackend:
    """Create a generation provider using explicit overrides and environment settings."""

    def pick(key: str, default: str = "", *, agy_key: str = "") -> str:
        keys = (f"{_LLM_ENV_PREFIX}{key}", f"{_AGY_ENV_PREFIX}{agy_key}") if agy_key else (f"{_LLM_ENV_PREFIX}{key}",)
        return _config_value(overrides, *keys, default=default)

    kind = pick("BACKEND", "agy_cli").lower()
    routing = RoleRoutingConfig.from_env(overrides, backend=kind)
    if kind in ("agy_cli", "agy"):
        return AgyCliLLMBackend(
            bin_path=pick("CLI_BIN", agy_key="BIN") or None, routing=routing,
            common_args=_config_value(overrides, f"{_AGY_ENV_PREFIX}COMMON_ARGS") or None,
            max_concurrency=int(pick("MAX_CONCURRENCY", "3", agy_key="MAX_CONCURRENCY")),
            fallback_model=pick("FALLBACK_MODEL") or None, overrides=overrides,
        )
    if kind == "mock":
        return MockLLMBackend(routing=routing)
    if kind in ("openai_compat", "openai", "vllm", "sglang", "http"):
        return OpenAICompatHTTPBackend(
            routing, api_base=pick("API_BASE") or None, api_key_env=pick("API_KEY_ENV") or None,
            max_concurrency=int(pick("MAX_CONCURRENCY", "4")),
            pass_reasoning_effort=pick("PASS_REASONING_EFFORT") == "1", stream=pick("STREAM") == "1",
        )
    if kind in ("claude_cli", "codex_cli", "custom_cli"):
        return SubprocessCliTemplateBackend(
            kind, routing, bin_path=pick("CLI_BIN") or None, template=pick("CLI_TEMPLATE") or None,
            max_concurrency=int(pick("MAX_CONCURRENCY", "2")), fallback_model=pick("FALLBACK_MODEL") or None, overrides=overrides,
        )
    raise ValueError("Unsupported TILEFOUNDRY_NPU_LLM_BACKEND value")


__all__ = [
    "AgyCliLLMBackend", "GenerationRequest", "KernelLLMBackend", "LLMGenerationError", "LLMTimeoutError",
    "MockLLMBackend", "OpenAICompatHTTPBackend", "PURE_SYNTHESIS_ROLES", "ProseRefusalError", "RE_503_CAPACITY",
    "RE_STREAM_IDLE_LOG", "RoleRoutingConfig", "StreamWatchdogConfig", "SubprocessCliTemplateBackend",
    "create_llm_backend_from_env", "extract_creative_payload", "extract_python_dsl", "sanitize_effort",
]
