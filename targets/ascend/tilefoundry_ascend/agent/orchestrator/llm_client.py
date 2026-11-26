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

RE_503_CAPACITY = re.compile(
    r"UNAVAILABLE\s*\(code\s*503\)(?::\s*No capacity available)?",
    re.IGNORECASE,
)
RE_STREAM_IDLE_LOG = re.compile(
    r"(?:run\.go:\d+\]\s*Run:\s*attempt\s+\d+\s+failed.*stream idle timeout|stream idle timeout)",
    re.IGNORECASE,
)
RE_SSE_DISPATCH = re.compile(r"streamGenerateContent\?alt=sse", re.IGNORECASE)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StreamWatchdogConfig:
    """Dual-timer streaming liveness and log-tailing watchdog settings for agy workers."""

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


def _capacity_503_policy_from_env(
    overrides: Mapping[str, str] | None = None,
) -> tuple[int, tuple[float, ...], float]:
    """Read 503 capacity retry count, backoff schedule, and jitter from environment."""
    raw_retries = _config_value(overrides, f"{_LLM_ENV_PREFIX}MAX_503_RETRIES", default="4")
    try:
        max_503_retries = int(raw_retries)
    except ValueError:
        raise ValueError("TILEFOUNDRY_NPU_LLM_MAX_503_RETRIES must be an integer") from None
    if max_503_retries < 0:
        raise ValueError("TILEFOUNDRY_NPU_LLM_MAX_503_RETRIES must be non-negative")
    raw_sched = _config_value(
        overrides,
        f"{_LLM_ENV_PREFIX}503_BACKOFF_SEC",
        default="5.0,15.0,35.0,60.0",
    )
    backoff_vals: list[float] = []
    for part in raw_sched.split(","):
        cleaned = part.strip()
        if not cleaned:
            continue
        try:
            val = float(cleaned)
        except ValueError:
            raise ValueError(
                "TILEFOUNDRY_NPU_LLM_503_BACKOFF_SEC entries must be numeric"
            ) from None
        if not math.isfinite(val) or val < 0.0:
            raise ValueError("TILEFOUNDRY_NPU_LLM_503_BACKOFF_SEC entries must be non-negative")
        backoff_vals.append(val)
    if not backoff_vals:
        backoff_vals = [5.0, 15.0, 35.0, 60.0]
    return max_503_retries, tuple(backoff_vals), 3.0


_LLM_ENV_PREFIX = "TILEFOUNDRY_NPU_LLM_"
_AGY_ENV_PREFIX = "TILEFOUNDRY_NPU_AGY_"
_REASONING_ROLES = frozenset({"explore_creative", "escalate_trackb"})
_EFFORT_LEVELS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
_TimeoutStage: TypeAlias = Literal["startup", "idle_stall", "active_ceiling", "after_eof"]


_CREATIVE_DEFAULT_CEILING_SEC = 480.0
_PARADIGM_SHIFT_CEILING_SEC = 720.0
_REGIME_HEADER_RE = re.compile(
    r"^###\s*TARGET REGIME\s*\n\s*([A-Za-z0-9_.-]+)\s*:",
    re.MULTILINE,
)


def _config_value(overrides: Mapping[str, str] | None, *keys: str, default: str = "") -> str:
    for source in (overrides or {}, os.environ):
        for key in keys:
            value = source.get(key, "").strip()
            if value:
                return value
    return default


def sanitize_effort(effort: str | None, *, default: str = "medium") -> str:
    """Return the requested effort label, using the default for empty input."""
    return (effort or "").strip() or default


@dataclass(frozen=True)
class GenerationRequest:
    """Single stateless LLM generation request routed by worker role."""

    role: str
    system_prompt: str
    user_prompt: str
    json_schema: dict[str, object] | None = None
    effort: str | None = None
    active_timeout_sec: float | None = None
    paradigm_shift: bool = False
    role_hint: str = ""
    session_scope: str = ""
    regime_tag: str = ""


def _is_creative_request(req: GenerationRequest) -> bool:
    hint = req.role_hint.lower()
    return req.role == "explore_creative" or "explore_creative" in hint or "creative" in hint


def _is_paradigm_shift_request(req: GenerationRequest) -> bool:
    if req.paradigm_shift:
        return True
    haystack = f"{req.role_hint}\n{req.system_prompt}\n{req.user_prompt}".lower()
    return (
        "paradigm_shift" in haystack
        or "paradigmshift" in haystack
        or "full_slice_rewrite" in haystack
    )


def _request_session_scope(req: GenerationRequest) -> str:
    if req.session_scope.strip():
        return req.session_scope.strip()
    if req.regime_tag.strip():
        return req.regime_tag.strip()
    match = _REGIME_HEADER_RE.search(req.user_prompt)
    if match is not None:
        return match.group(1).strip()
    return "default"


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
        self,
        exploit_model: str,
        explore_model: str,
        escalate_model: str,
        repair_model: str,
        max_effort: str = "xhigh",
        exploit_timeout_sec: float | None = None,
        reasoning_timeout_sec: float = 420.0,
        repair_timeout_sec: float | None = None,
    ) -> None:
        """Bound repair by an explicit exploit budget, else keep its own default."""
        if exploit_timeout_sec is None:
            effective_exploit = RoleRoutingConfig.exploit_timeout_sec
            effective_repair = (
                RoleRoutingConfig.repair_timeout_sec
                if repair_timeout_sec is None
                else repair_timeout_sec
            )
        else:
            effective_exploit = exploit_timeout_sec
            effective_repair = (
                exploit_timeout_sec if repair_timeout_sec is None else repair_timeout_sec
            )
        if max_effort.lower() not in _EFFORT_LEVELS:
            raise ValueError("LLM max effort must be a supported effort level")
        if any(
            not math.isfinite(timeout) or timeout <= 0
            for timeout in (effective_exploit, reasoning_timeout_sec, effective_repair)
        ):
            raise ValueError("LLM role timeouts must be finite and positive")
        object.__setattr__(self, "exploit_model", exploit_model)
        object.__setattr__(self, "explore_model", explore_model)
        object.__setattr__(self, "escalate_model", escalate_model)
        object.__setattr__(self, "repair_model", repair_model)
        object.__setattr__(self, "max_effort", max_effort)
        object.__setattr__(self, "exploit_timeout_sec", effective_exploit)
        object.__setattr__(self, "reasoning_timeout_sec", reasoning_timeout_sec)
        object.__setattr__(self, "repair_timeout_sec", effective_repair)

    def model_for_role(self, role: str) -> str:
        """Return the model configured for this worker role."""
        if role == "explore_creative":
            return self.explore_model
        if role == "escalate_trackb":
            return self.escalate_model
        if role == "syntax_repair":
            return self.repair_model
        return self.exploit_model

    def timeout_for_role(self, role: str) -> float:
        """Return the time budget for this worker role in seconds."""
        if role in _REASONING_ROLES:
            return self.reasoning_timeout_sec
        if role == "syntax_repair":
            return self.repair_timeout_sec
        return self.exploit_timeout_sec

    def effort_for_request(
        self, req: GenerationRequest, override: str | None = None, *, default: str = ""
    ) -> str:
        """Apply the ceiling to recognized levels and preserve provider-specific labels."""
        effort = sanitize_effort(override, default=sanitize_effort(req.effort, default=default))
        ceiling = self.max_effort.lower()
        if effort.lower() in _EFFORT_LEVELS:
            if _EFFORT_LEVELS.index(effort.lower()) > _EFFORT_LEVELS.index(ceiling):
                return ceiling
        return effort

    @classmethod
    def from_env(
        cls, overrides: Mapping[str, str] | None = None, *, backend: str | None = None
    ) -> RoleRoutingConfig:
        """Load role settings with explicit values taking precedence over the environment."""
        kind = (backend or _config_value(overrides, f"{_LLM_ENV_PREFIX}BACKEND", default="agy_cli")).lower()
        is_agy = kind in ("agy_cli", "agy")

        def pick(key: str, default: str, *, legacy: bool = False) -> str:
            keys = [f"{_LLM_ENV_PREFIX}{key}"]
            if legacy and is_agy:
                keys.append(f"{_AGY_ENV_PREFIX}{key}")
            return _config_value(overrides, *keys, default=default)

        if is_agy:
            default_exploit = AgyCliLLMBackend.DEFAULT_EXPLOIT_MODEL
        else:
            default_exploit = "mock" if kind == "mock" else ""
        exploit = pick("EXPLOIT_MODEL", default_exploit, legacy=True)
        default_explore = AgyCliLLMBackend.DEFAULT_EXPLORE_MODEL if is_agy else exploit
        explore = pick("EXPLORE_MODEL", default_explore, legacy=True)
        exploit_timeout_raw = pick("EXPLOIT_TIMEOUT_SEC", "")
        exploit_timeout_sec = (
            float(exploit_timeout_raw) if exploit_timeout_raw else cls.exploit_timeout_sec
        )
        return cls(
            exploit_model=exploit,
            explore_model=explore,
            escalate_model=pick("ESCALATE_MODEL", explore, legacy=True),
            repair_model=pick("REPAIR_MODEL", exploit),
            max_effort=_config_value(
                overrides,
                f"{_LLM_ENV_PREFIX}MAX_EFFORT",
                f"{_LLM_ENV_PREFIX}EFFORT_CAP",
                default="high" if is_agy else "xhigh",
            ),
            exploit_timeout_sec=exploit_timeout_sec,
            reasoning_timeout_sec=float(pick("REASONING_TIMEOUT_SEC", "360" if is_agy else "420")),
            repair_timeout_sec=(
                exploit_timeout_sec if exploit_timeout_raw else cls.repair_timeout_sec
            ),
        )


class KernelLLMBackend(Protocol):
    """Stateless generation interface shared by remote, subprocess, and offline providers."""

    async def generate(self, req: GenerationRequest) -> str:
        """Generate response text or a JSON string for a worker request."""
        ...


_CODE_FENCE_RE = re.compile(
    r"```(?:python|cpp|c\+\+|cce|diff)?\s*\n(.*?)```",
    re.DOTALL | re.IGNORECASE,
)

_EXECUTABLE_AST_NODES = (
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
    ast.Assign,
    ast.AnnAssign,
    ast.Import,
    ast.ImportFrom,
)

_CPP_CODE_MARKERS: tuple[str, ...] = (
    "__aicore__",
    "__global__",
    "__simd_vf__",
    "__simt_vf__",
    "__simd_callee__",
    "__simt_callee__",
    "#include",
    "constexpr",
)

_BROKEN_PYTHON_HEAD_RE = re.compile(
    r"^\s*(?:"
    r"(?:async\s+)?def\s+[A-Za-z_][A-Za-z0-9_]*\s*\("
    r"|class\s+[A-Za-z_][A-Za-z0-9_]*\s*[:(]"
    r"|import\s+[A-Za-z_][A-Za-z0-9_.]*(?:\s*,|\s+as|\s*$)"
    r"|from\s+[A-Za-z_.][A-Za-z0-9_.]*\s+import\s+"
    r"|@[A-Za-z_][A-Za-z0-9_.]*(?:\s*\(|\s*$)"
    r")",
    re.MULTILINE,
)

class LLMGenerationError(RuntimeError):
    """Structured LLM generation error carrying fault plane, code, and retriability."""

    def __init__(
        self,
        message: str,
        *,
        fault_plane: str = "LLM_TRANSPORT",
        fault_code: str = "LLM_TIMEOUT_OR_503",
        retriable: bool = False,
    ) -> None:
        super().__init__(message)
        self.fault_plane = fault_plane
        self.fault_code = fault_code
        self.retriable = retriable


class LLMTimeoutError(TimeoutError, LLMGenerationError):
    """Structured LLM timeout error preserving TimeoutError and LLMGenerationError identity."""

    def __init__(
        self,
        message: str,
        *,
        fault_plane: str = "LLM_TRANSPORT",
        fault_code: str = "LLM_TIMEOUT_OR_503",
        retriable: bool = False,
    ) -> None:
        TimeoutError.__init__(self, message)
        self.fault_plane = fault_plane
        self.fault_code = fault_code
        self.retriable = retriable


class _ProseRefusalMessage(str):
    """String message carrying structured prose-refusal fault attributes."""

    fault_plane: str = "SYNTAX_AST"
    fault_code: str = "LLM_PROSE_NO_CODE"
    retriable: bool = False


_PROSE_REFUSAL_ERROR = _ProseRefusalMessage(
    "llm_prose_refusal_no_code: "
    "response contains prose without executable DSL or SEARCH/REPLACE hunks"
)


class ProseRefusalError(ValueError, LLMGenerationError):
    """Raised when an LLM reply contains prose without executable DSL or patch hunks."""

    def __init__(self, message: str = _PROSE_REFUSAL_ERROR) -> None:
        ValueError.__init__(self, message)
        self.fault_plane = "SYNTAX_AST"
        self.fault_code = "LLM_PROSE_NO_CODE"
        self.retriable = False

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


def _has_code_or_patch_structure(text: str) -> bool:
    """Return True if *text* has executable Python, SEARCH/REPLACE hunks, or C++ markers."""
    stripped = text.strip()
    if not stripped:
        return False
    if "<<<<<<< SEARCH" in stripped or "<<< SEARCH" in stripped:
        return True
    if any(marker in stripped for marker in _CPP_CODE_MARKERS):
        return True
    try:
        tree = ast.parse(stripped)
    except SyntaxError:
        return bool(_BROKEN_PYTHON_HEAD_RE.search(stripped))
    return any(isinstance(node, _EXECUTABLE_AST_NODES) for node in ast.walk(tree))


def extract_python_dsl(raw_response: str) -> str:
    """Extract clean DSL or Track-B patch text, rejecting pure natural-language refusals."""
    stripped = raw_response.strip()
    if stripped.startswith("{"):
        try:
            payload = json.loads(stripped)
            if isinstance(payload, dict):
                if isinstance(payload.get("structured_output"), dict):
                    dsl = payload["structured_output"].get("dsl_source")
                    if isinstance(dsl, str):
                        inner = dsl.strip()
                        if inner and inner != stripped:
                            return extract_python_dsl(inner)
                        raise ProseRefusalError(_PROSE_REFUSAL_ERROR)
                if isinstance(payload.get("dsl_source"), str):
                    inner = payload["dsl_source"].strip()
                    if inner and inner != stripped:
                        return extract_python_dsl(inner)
                    raise ProseRefusalError(_PROSE_REFUSAL_ERROR)
                if isinstance(payload.get("response"), str) and payload["response"].strip():
                    stripped = payload["response"].strip()
        except json.JSONDecodeError:
            pass
    matches = _CODE_FENCE_RE.findall(stripped)
    if matches:
        valid_matches = [m.strip() for m in matches if _has_code_or_patch_structure(m)]
        if valid_matches:
            return max(valid_matches, key=len)
        raise ProseRefusalError(_PROSE_REFUSAL_ERROR)
    if not _has_code_or_patch_structure(stripped):
        raise ProseRefusalError(_PROSE_REFUSAL_ERROR)
    return stripped


def extract_creative_payload(raw_response: str) -> dict[str, str]:
    """Parse structured JSON response or markdown fallback from creative explore worker."""
    stripped = raw_response.strip()
    if not stripped.startswith("{") and "```" in stripped:
        for fence_match in re.finditer(r"```[a-zA-Z0-9_-]*\s*\n(.*?)```", stripped, re.DOTALL):
            inner = fence_match.group(1).strip()
            if inner.startswith("{"):
                stripped = inner
                break
    if stripped.startswith("{"):
        try:
            obj = json.loads(stripped)
            if isinstance(obj, dict):
                if isinstance(obj.get("structured_output"), dict):
                    obj = obj["structured_output"]
                elif isinstance(obj.get("response"), str) and obj["response"].strip().startswith(
                    "{"
                ):
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
            return os.path.abspath(resolved)
    on_path = shutil.which("agy")
    if on_path:
        return os.path.abspath(on_path)
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
        self.saw_progress = False
        self.sse_in_flight = False

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
            is_err_line = False
            if RE_503_CAPACITY.search(line):
                self.count_503 += 1
                events.append("503_no_capacity")
                is_err_line = True
            if RE_STREAM_IDLE_LOG.search(line):
                self.count_idle_timeout += 1
                events.append("stream_idle_timeout")
                is_err_line = True
            if not is_err_line:
                self.saw_progress = True
            if RE_SSE_DISPATCH.search(line):
                self.sse_in_flight = True
        return events


class _AttemptFailure(Exception):
    """Internal wrapper pairing a worker failure with retry eligibility."""

    timeout_stage: _TimeoutStage | None

    def __init__(
        self,
        exc: Exception,
        *,
        retriable: bool,
        ttft_sec: float | None = None,
        timeout_stage: _TimeoutStage | None = None,
        since_last_activity_sec: float | None = None,
        is_503: bool = False,
        is_tool_loop: bool = False,
    ) -> None:
        super().__init__(str(exc))
        self.exc = exc
        self.retriable = retriable
        self.ttft_sec = ttft_sec
        self.timeout_stage = timeout_stage
        self.since_last_activity_sec = since_last_activity_sec
        self.is_503 = is_503
        self.is_tool_loop = is_tool_loop


async def _terminate_process_group(proc: asyncio.subprocess.Process) -> None:
    """Terminate the subprocess session and reap the process."""
    if proc.returncode is None:
        with suppress(ProcessLookupError):
            os.killpg(proc.pid, signal.SIGTERM)
        with suppress(TimeoutError):
            await asyncio.wait_for(proc.wait(), timeout=1.0)

    with suppress(ProcessLookupError):
        os.killpg(proc.pid, signal.SIGKILL)
    await proc.wait()


_HARD_NO_TOOL_DIRECTIVE = (
    "[CRITICAL HARD DIRECTIVE: ZERO TOOL CALLS ALLOWED. Your previous attempt was "
    "terminated immediately for attempting a tool call. Do NOT invoke ANY tool "
    "(forbid read_file, view_file, write_file, write_to_file, run_command, "
    "grep_search, list_dir, and invoke_subagent). Output your final answer directly "
    "in plain text or JSON right now.]"
)


def _finalize_exhausted_failure(failure: _AttemptFailure) -> Exception:
    """Attach non-retriable structured fault attributes onto exhausted transport failures."""
    exc = failure.exc
    if isinstance(exc, LLMGenerationError):
        exc.retriable = False
        return exc
    if isinstance(exc, TimeoutError):
        return LLMTimeoutError(
            str(exc),
            fault_plane="LLM_TRANSPORT",
            fault_code="LLM_TIMEOUT_OR_503",
            retriable=False,
        )
    detail = str(exc)
    is_timeout_or_503 = (
        failure.is_503
        or failure.timeout_stage is not None
        or "stream_idle_timeout" in detail
        or bool(RE_STREAM_IDLE_LOG.search(detail))
        or bool(RE_503_CAPACITY.search(detail))
    )
    fault_code = "LLM_TIMEOUT_OR_503" if is_timeout_or_503 else "LLM_CLI_ERROR"
    return LLMGenerationError(
        detail,
        fault_plane="LLM_TRANSPORT",
        fault_code=fault_code,
        retriable=False,
    )


class AgyCliLLMBackend:
    """Stateless headless wrapper around local agy CLI inside an empty sandbox directory."""

    DEFAULT_COMMON_ARGS = (
        "--input-format text --disable-slash-commands "
        "--sandbox --mode plan --dangerously-skip-permissions"
    )
    DEFAULT_EXPLOIT_MODEL = "gemini-4-hydrogen-medium"
    DEFAULT_EXPLORE_MODEL = "gemini-4-hydrogen-high"
    DEFAULT_ESCALATE_MODEL = "gemini-4-hydrogen-high"

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
        repair_model: str | None = None,
        *,
        routing: RoleRoutingConfig | None = None,
        fallback_model: str | None = None,
        overrides: Mapping[str, str] | None = None,
    ) -> None:
        self._overrides = dict(overrides) if overrides else None
        self._bin = _resolve_agy_binary(bin_path)
        model_overrides = {
            f"{_LLM_ENV_PREFIX}{role}_MODEL": model
            for role, model in (
                ("EXPLOIT", exploit_model),
                ("EXPLORE", explore_model),
                ("ESCALATE", escalate_model),
                ("REPAIR", repair_model),
            )
            if model is not None
        }
        self._routing = routing or RoleRoutingConfig.from_env(model_overrides, backend="agy_cli")
        raw_fallback = (
            fallback_model
            if fallback_model is not None
            else _config_value(self._overrides, f"{_LLM_ENV_PREFIX}FALLBACK_MODEL")
        ).strip()
        self._fallback_model = self._agy_model_id(raw_fallback) if raw_fallback else ""

        raw_common = (
            common_args
            if common_args is not None
            else os.environ.get("TILEFOUNDRY_NPU_AGY_COMMON_ARGS", "")
        ).strip() or self.DEFAULT_COMMON_ARGS
        self._common_args = _normalize_agy_common_args(raw_common)

        raw_conc = os.environ.get("TILEFOUNDRY_NPU_AGY_MAX_CONCURRENCY", "").strip()
        conc = max_concurrency or (int(raw_conc) if raw_conc else 3)
        self._sem = asyncio.Semaphore(max(1, conc))
        self._consecutive_active_ceiling_timeouts: dict[tuple[str, str], int] = {}
        self._active_ceiling_cooldown: set[tuple[str, str]] = set()

        self._explicit_watchdog = watchdog is not None or hard_timeout_sec is not None
        if watchdog is not None:
            if watchdog.capacity_503_max_retries is None and watchdog.max_retries > 0:
                max_503, backoff_503, jitter_503 = _capacity_503_policy_from_env(self._overrides)
                self._watchdog = replace(
                    watchdog,
                    capacity_503_max_retries=max_503,
                    capacity_503_backoff_sec=backoff_503,
                    capacity_503_jitter_sec=jitter_503,
                )
            else:
                self._watchdog = watchdog
        elif hard_timeout_sec is not None:
            ceiling = float(hard_timeout_sec)
            max_503, backoff_503, jitter_503 = _capacity_503_policy_from_env(self._overrides)
            self._watchdog = StreamWatchdogConfig(
                startup_timeout_sec=min(60.0, ceiling),
                idle_stall_timeout_sec=min(60.0, ceiling),
                active_ceiling_timeout_sec=ceiling,
                capacity_503_max_retries=max_503,
                capacity_503_backoff_sec=backoff_503,
                capacity_503_jitter_sec=jitter_503,
            )
        else:
            self._watchdog = self.watchdog_for_role("exploit_rule")

    @property
    def watchdog(self) -> StreamWatchdogConfig:
        """Return the watchdog configuration for default exploit workers."""
        return self._watchdog

    @staticmethod
    def _agy_model_id(model_id: str) -> str:
        return re.sub(r"-xhigh$", "-high", model_id.strip(), flags=re.IGNORECASE)

    def select_model(self, role: str, *, effort: str | None = None) -> str:
        """Return the worker's model ID using legacy Agy tier names."""
        return self._agy_model_id(self._routing.model_for_role(role))

    def watchdog_for_model(self, model_id: str) -> StreamWatchdogConfig:
        """Return the legacy Agy watchdog settings for a model tier."""
        if self._explicit_watchdog:
            return self._watchdog
        role = (
            "explore_creative" if "high" in self._agy_model_id(model_id).lower() else "exploit_rule"
        )
        return self.watchdog_for_role(role)

    def watchdog_for_role(self, role: str) -> StreamWatchdogConfig:
        """Return the explicit or role-based Agy watchdog settings."""
        if self._explicit_watchdog:
            return self._watchdog
        ceiling = self._routing.timeout_for_role(role)
        if role == "explore_creative":
            ceiling = max(_CREATIVE_DEFAULT_CEILING_SEC, ceiling)
        max_503_retries, backoff_503, jitter_503 = _capacity_503_policy_from_env(self._overrides)
        return StreamWatchdogConfig(
            startup_timeout_sec=min(60.0, ceiling),
            idle_stall_timeout_sec=min(180.0 if role in _REASONING_ROLES else 120.0, ceiling),
            active_ceiling_timeout_sec=ceiling,
            capacity_503_max_retries=max_503_retries,
            capacity_503_backoff_sec=backoff_503,
            capacity_503_jitter_sec=jitter_503,
        )

    def watchdog_for_request(self, req: GenerationRequest) -> StreamWatchdogConfig:
        """Resolve request-aware watchdog ceiling, scaling creative and ParadigmShift tiers."""
        if self._explicit_watchdog:
            return self._watchdog
        eff_role = "explore_creative" if _is_creative_request(req) else req.role
        cfg = self.watchdog_for_role(eff_role)
        ceiling = cfg.active_ceiling_timeout_sec
        if _is_paradigm_shift_request(req):
            ceiling = max(_PARADIGM_SHIFT_CEILING_SEC, ceiling)
        if (
            req.active_timeout_sec is not None
            and math.isfinite(req.active_timeout_sec)
            and req.active_timeout_sec > 0.0
        ):
            ceiling = float(req.active_timeout_sec)
        if ceiling == cfg.active_ceiling_timeout_sec:
            return cfg
        return replace(
            cfg,
            startup_timeout_sec=min(60.0, ceiling),
            idle_stall_timeout_sec=min(
                180.0 if eff_role in _REASONING_ROLES else 120.0, ceiling
            ),
            active_ceiling_timeout_sec=ceiling,
        )

    def select_effort(
        self,
        req: GenerationRequest,
        effort_override: str | None = None,
    ) -> str:
        """Select effort for Agy, including compatibility with its legacy tiers."""
        model = self.select_model(req.role).lower()
        default = (
            "high"
            if req.role in _REASONING_ROLES
            or "carbon" in model
            or "gemini-pro-orchestrator" in model
            else "medium"
        )
        effort = self._routing.effort_for_request(req, effort_override, default=default)
        return "high" if effort.lower() in ("xhigh", "max") else effort

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
        model_override: str | None = None,
    ) -> list[str]:
        """Build isolated agy stream-json CLI argument list without workspace mounts."""
        effort = self.select_effort(req, effort_override=effort_override)
        model_id = (
            self._agy_model_id(model_override)
            if model_override
            else self.select_model(req.role, effort=effort)
        )
        for tier_suffix in ("-low", "-medium", "-high"):
            if model_id.endswith(tier_suffix):
                effort = tier_suffix[1:]
                break
        cfg = self.watchdog_for_request(req)
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
            delta_joined = "".join(text_deltas)
            combined_out = f"{err_detail}\n{delta_joined}"
            is_503_err = tailer.count_503 > 0 or bool(RE_503_CAPACITY.search(combined_out))
            is_retriable = (
                is_503_err
                or tailer.count_idle_timeout > 0
                or bool(RE_STREAM_IDLE_LOG.search(combined_out))
            )
            raise _AttemptFailure(
                RuntimeError(f"Agy generation failed with exit status {exit_code}"),
                retriable=is_retriable,
                is_503=is_503_err,
            )

        if exit_code == 0 and text_deltas:
            return "".join(text_deltas).strip()

        combined_out = f"{stderr_text}\n{''.join(text_deltas)}"
        is_503_err = tailer.count_503 > 0 or bool(RE_503_CAPACITY.search(combined_out))
        is_retriable = (
            is_503_err
            or tailer.count_idle_timeout > 0
            or bool(RE_STREAM_IDLE_LOG.search(combined_out))
        )
        raise _AttemptFailure(
            RuntimeError(f"Agy generation failed with exit status {exit_code}"),
            retriable=is_retriable,
            is_503=is_503_err,
        )

    async def _run_single_attempt(
        self,
        req: GenerationRequest,
        full_prompt: str,
        *,
        effort_override: str | None = None,
        model_override: str | None = None,
    ) -> str:
        effort = self.select_effort(req, effort_override=effort_override)
        model_id = (
            self._agy_model_id(model_override)
            if model_override
            else self.select_model(req.role, effort=effort)
        )
        cfg = self.watchdog_for_request(req)
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
                model_override=model_override,
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
                                            elif (
                                                su.get("state") != "DONE"
                                                or cfg.max_sandbox_tool_calls == 0
                                            ):
                                                anon_tool_calls += 1
                                            if (
                                                len(active_tool_steps) + anon_tool_calls
                                            ) > cfg.max_sandbox_tool_calls:
                                                await _terminate_process_group(proc)
                                                raise _AttemptFailure(
                                                    LLMGenerationError(
                                                        "sandbox_tool_loop_aborted",
                                                        fault_plane="LLM_TRANSPORT",
                                                        fault_code="SANDBOX_TOOL_LOOP_ABORTED",
                                                        retriable=False,
                                                    ),
                                                    retriable=True,
                                                    is_tool_loop=True,
                                                )
                                        delta = su.get("text_delta")
                                        if isinstance(delta, str) and delta:
                                            text_deltas.append(delta)
                                    elif ev_type == "result":
                                        raw_res = ev.get("result")
                                        result_payload = (
                                            raw_res if isinstance(raw_res, dict) else ev
                                        )
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
                    if tailer.saw_progress and not log_events:
                        last_activity_ts = time.monotonic()
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
                            is_503=True,
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
                                retriable=True,
                                timeout_stage="active_ceiling",
                            )
                        if not got_init and (now - start_ts) > cfg.startup_timeout_sec:
                            await _terminate_process_group(proc)
                            raise _AttemptFailure(
                                TimeoutError(
                                    f"agy startup_timeout ({cfg.startup_timeout_sec}s) "
                                    f"exceeded (role={req.role}, model={model_id})"
                                ),
                                retriable=True,
                                timeout_stage="startup",
                            )
                        effective_idle_sec = (
                            max(cfg.idle_stall_timeout_sec, cfg.active_ceiling_timeout_sec)
                            if tailer.sse_in_flight
                            else cfg.idle_stall_timeout_sec
                        )
                        if (now - last_activity_ts) > effective_idle_sec:
                            await _terminate_process_group(proc)
                            raise _AttemptFailure(
                                TimeoutError(
                                    f"agy idle_stall_timeout ({effective_idle_sec}s) "
                                    f"exceeded (role={req.role}, model={model_id})"
                                ),
                                retriable=True,
                                timeout_stage="idle_stall",
                            )

                await proc.wait()
                tailer.poll()
                return self._extract_stream_output(
                    req=req,
                    exit_code=proc.returncode,
                    result_payload=result_payload,
                    text_deltas=text_deltas,
                    stderr_lines=stderr_lines,
                    tailer=tailer,
                )
            finally:
                await _terminate_process_group(proc)
                for task in (pump_out, pump_err):
                    if not task.done():
                        task.cancel()
                await asyncio.gather(pump_out, pump_err, return_exceptions=True)

    async def generate(
        self,
        req: GenerationRequest,
        *,
        effort_override: str | None = None,
    ) -> str:
        """Execute agy in an empty sandbox with streaming liveness watchdog and retries."""
        full_prompt = (
            "[STATELESS GENERATION CONTRACT: Do NOT invoke ANY tool calls "
            "(forbid run_command, view_file, list_dir, grep_search, write_to_file, "
            "and invoke_subagent). Answer directly in a single turn using ONLY the "
            "prompt context below.]\n\n"
            f"### SYSTEM INSTRUCTIONS\n{req.system_prompt}\n\n"
            f"### USER INPUT\n{req.user_prompt}\n"
        )
        cfg = self.watchdog_for_request(req)
        scope_key = (_request_session_scope(req), req.role)
        used_cooldown = scope_key in self._active_ceiling_cooldown
        if used_cooldown:
            self._active_ceiling_cooldown.discard(scope_key)
        cooldown_model = "gemini-4-hydrogen-medium" if used_cooldown else None
        eff_effort_override = "medium" if used_cooldown else effort_override
        base_model = self.select_model(req.role)

        tool_retry_used = False

        async def _attempt(retry_model_override: str | None) -> str:
            nonlocal tool_retry_used
            active_override = retry_model_override or cooldown_model
            active_model = (
                self._agy_model_id(active_override) if active_override else base_model
            )
            prompt_for_attempt = full_prompt
            while True:
                try:
                    result = await self._run_single_attempt(
                        req,
                        prompt_for_attempt,
                        effort_override=eff_effort_override,
                        model_override=active_override,
                    )
                    break
                except _AttemptFailure as failure:
                    if failure.is_tool_loop:
                        if not tool_retry_used and cfg.max_retries > 0:
                            tool_retry_used = True
                            prompt_for_attempt = f"{_HARD_NO_TOOL_DIRECTIVE}\n\n{full_prompt}"
                            continue
                        raise _AttemptFailure(
                            LLMGenerationError(
                                str(failure.exc),
                                fault_plane="LLM_TRANSPORT",
                                fault_code="SANDBOX_TOOL_LOOP_ABORTED",
                                retriable=False,
                            ),
                            retriable=False,
                            is_tool_loop=True,
                        ) from None
                    if (
                        failure.timeout_stage == "active_ceiling"
                        and not used_cooldown
                        and _is_creative_request(req)
                        and active_model.lower() == "gemini-4-hydrogen-high"
                    ):
                        count = self._consecutive_active_ceiling_timeouts.get(scope_key, 0) + 1
                        if count >= 2:
                            self._active_ceiling_cooldown.add(scope_key)
                            self._consecutive_active_ceiling_timeouts[scope_key] = 0
                        else:
                            self._consecutive_active_ceiling_timeouts[scope_key] = count
                    raise
            self._consecutive_active_ceiling_timeouts[scope_key] = 0
            return result

        return await _run_with_503_retry(
            sem=self._sem,
            cfg=cfg,
            fallback_model=self._fallback_model,
            attempt_fn=_attempt,
        )


async def _run_with_503_retry(
    *,
    sem: asyncio.Semaphore,
    cfg: StreamWatchdogConfig,
    fallback_model: str,
    attempt_fn: Callable[[str | None], Awaitable[str]],
) -> str:
    """Execute *attempt_fn* with 503 exponential backoff, early fallback, and standard retries."""
    max_retries = max(0, cfg.max_retries)
    max_503_retries = (
        max_retries
        if cfg.capacity_503_max_retries is None
        else max(0, cfg.capacity_503_max_retries)
    )
    retries_503 = 0
    retries_other = 0
    next_model_override: str | None = None
    async with sem:
        while True:
            try:
                return await attempt_fn(next_model_override)
            except _AttemptFailure as failure:
                if not failure.retriable:
                    raise _finalize_exhausted_failure(failure) from None
                if failure.is_503 and cfg.capacity_503_max_retries is not None:
                    if retries_503 >= max_503_retries:
                        raise _finalize_exhausted_failure(failure) from None
                    sched = cfg.capacity_503_backoff_sec
                    base_wait = (
                        sched[min(retries_503, len(sched) - 1)]
                        if sched
                        else cfg.retry_backoff_sec
                    )
                    jitter = (
                        random.uniform(0.0, cfg.capacity_503_jitter_sec)
                        if cfg.capacity_503_jitter_sec > 0.0
                        else 0.0
                    )
                    retries_503 += 1
                    if retries_503 >= min(2, max_503_retries) and fallback_model:
                        next_model_override = fallback_model
                    if base_wait + jitter > 0.0:
                        await asyncio.sleep(base_wait + jitter)
                    continue
                if retries_other < max_retries:
                    retries_other += 1
                    if cfg.retry_backoff_sec > 0.0:
                        await asyncio.sleep(cfg.retry_backoff_sec)
                    continue
                raise _finalize_exhausted_failure(failure) from None


_STATELESS_INSTRUCTION = (
    "[STATELESS GENERATION CONTRACT: Do NOT invoke ANY tool calls. "
    "Answer directly in a single turn using ONLY the prompt context below.]"
)


def _system_prompt(req: GenerationRequest) -> str:
    if req.json_schema is None:
        return req.system_prompt
    return f"{req.system_prompt}\n\nReturn only JSON matching this schema:\n{json.dumps(req.json_schema)}"


class _NoHTTPRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_SSE_LINE_END = re.compile(rb"\r\n|\r|\n")


def _iter_sse_data(chunks: Iterable[bytes]) -> Iterator[bytes]:
    pending_line = b""
    data: list[bytes] = []
    first_line = True
    previous_cr = False
    for chunk in chunks:
        if previous_cr:
            chunk = chunk.removeprefix(b"\n")
        pending_line += chunk
        previous_cr = pending_line.endswith(b"\r")
        start = 0
        for match in _SSE_LINE_END.finditer(pending_line):
            line = pending_line[start : match.start()]
            start = match.end()
            if first_line:
                line = line.removeprefix(b"\xef\xbb\xbf")
                first_line = False
            if not line and data:
                yield b"\n".join(data)
                data.clear()
            elif line == b"data" or line.startswith(b"data:"):
                data.append(line.partition(b":")[2].removeprefix(b" "))
        pending_line = pending_line[start:]


def _reject_http_tools(message: Mapping[str, object], finish_reason: object = None) -> None:
    if (
        message.get("tool_calls")
        or message.get("function_call")
        or finish_reason in ("tool_calls", "function_call")
    ):
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
        for index, choice in enumerate(event.get("choices", ())):
            delta = choice.get("delta") or {}
            finish = choice.get("finish_reason")
            _reject_http_tools(delta, finish)
            if choice.get("index", index) != 0:
                continue
            content = delta.get("content")
            if isinstance(content, str):
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
        self,
        routing: RoleRoutingConfig,
        *,
        api_base: str | None = None,
        api_key_env: str | None = None,
        max_concurrency: int = 4,
        pass_reasoning_effort: bool | None = None,
        stream: bool | None = None,
    ) -> None:
        self._routing = routing
        base = api_base or _config_value(
            None, f"{_LLM_ENV_PREFIX}API_BASE", default="http://127.0.0.1:8000/v1"
        )
        try:
            url = urlsplit(base.strip())
            if (
                url.scheme not in ("http", "https")
                or not url.hostname
                or url.username is not None
                or url.query
                or url.fragment
            ):
                raise ValueError
        except ValueError:
            raise ValueError("LLM API base must be an HTTP(S) URL without credentials") from None
        path = url.path.rstrip("/") or "/v1"
        if not path.endswith("/chat/completions"):
            path += "/chat/completions"
        self._endpoint = url._replace(path=path).geturl()
        default_key_var = (
            "OPENAI_API_KEY" if url.scheme == "https" and url.hostname == "api.openai.com" else ""
        )
        self._key_var = api_key_env or _config_value(
            None, f"{_LLM_ENV_PREFIX}API_KEY_ENV", default=default_key_var
        )
        if self._key_var and not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self._key_var):
            raise ValueError("LLM API key configuration must be an environment variable name")
        self._pass_effort = (
            pass_reasoning_effort
            if pass_reasoning_effort is not None
            else _config_value(None, f"{_LLM_ENV_PREFIX}PASS_REASONING_EFFORT") == "1"
        )
        self._stream = (
            stream if stream is not None else _config_value(None, f"{_LLM_ENV_PREFIX}STREAM") == "1"
        )
        self._sem = asyncio.Semaphore(max(1, max_concurrency))

    def _sync_post(
        self, body: dict[str, object], timeout_sec: float, cancelled: Event
    ) -> str:
        deadline = time.monotonic() + timeout_sec
        accept = "text/event-stream, application/json" if self._stream else "application/json"
        headers = {"Content-Type": "application/json", "Accept": accept}
        key = os.environ.get(self._key_var, "").strip()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        request = Request(
            self._endpoint, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST"
        )
        try:
            if cancelled.is_set():
                raise TimeoutError
            with build_opener(_NoHTTPRedirects()).open(request, timeout=timeout_sec) as response:
                chunks = _read_http_chunks(response, deadline, cancelled)
                media_type = response.getheader("Content-Type", "").partition(";")[0].strip().lower()
                if media_type == "text/event-stream":
                    content = _read_sse_completion(chunks)
                else:
                    payload = json.loads(b"".join(chunks))
                    choice = payload["choices"][0]
                    message = choice["message"]
                    _reject_http_tools(message, choice.get("finish_reason"))
                    content = message.get("content")
        except HTTPError as exc:
            exc.close()
            raise RuntimeError(f"LLM HTTP request failed with status {exc.code}") from None
        except URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise TimeoutError("LLM HTTP request timed out") from None
            raise RuntimeError("LLM HTTP transport failed") from None
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
        model = self._routing.model_for_role(req.role)
        if not model:
            raise ValueError("Configure an LLM model for the requested HTTP generation role")
        timeout = self._routing.timeout_for_role(req.role)
        body: dict[str, object] = {
            "model": model,
            "messages": [
                {"role": "system", "content": _system_prompt(req)},
                {"role": "user", "content": req.user_prompt},
            ],
        }
        if self._stream:
            body["stream"] = True
        effort = self._routing.effort_for_request(req, effort_override)
        if self._pass_effort and effort:
            body["reasoning_effort"] = effort

        await self._sem.acquire()
        cancelled = Event()
        try:
            pending = asyncio.get_running_loop().run_in_executor(
                None, self._sync_post, body, timeout, cancelled
            )
        except RuntimeError:
            self._sem.release()
            raise

        def release_slot(completed: asyncio.Future[str]) -> None:
            self._sem.release()
            if not completed.cancelled():
                completed.exception()

        pending.add_done_callback(release_slot)
        try:
            return await asyncio.wait_for(asyncio.shield(pending), timeout)
        except TimeoutError:
            raise TimeoutError("LLM HTTP request timed out") from None
        finally:
            cancelled.set()


_CLI_PROXY_KEYS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")
_CLI_TELEMETRY_FORMAT = (
    "LLM_CALL_TELEMETRY backend=%s role=%s model=%s ttft_sec=%.3f "
    "elapsed_sec=%.3f status=%s proxy_configured=%s timeout_stage=%s "
    "since_last_activity_sec=%.3f"
)


def _log_cli_telemetry(
    *,
    backend: str,
    role: str,
    model: str,
    ttft_sec: float,
    elapsed_sec: float,
    status: str,
    proxy_configured: str,
    timeout_stage: _TimeoutStage | None,
    since_last_activity_sec: float | None,
) -> None:
    """Emit one non-sensitive telemetry line for a CLI generation attempt."""
    logger.info(
        _CLI_TELEMETRY_FORMAT,
        backend,
        role,
        model,
        ttft_sec,
        elapsed_sec,
        status,
        proxy_configured,
        timeout_stage or "none",
        -1.0 if since_last_activity_sec is None else since_last_activity_sec,
    )


def _proxy_env(environ: Mapping[str, str]) -> dict[str, str]:
    """Resolve explicit proxy settings into uppercase and lowercase variables."""
    resolved: dict[str, str] = {}
    for key in _CLI_PROXY_KEYS:
        scoped = environ.get(f"{_LLM_ENV_PREFIX}{key}", "").strip()
        plain = environ.get(key, "").strip() or environ.get(key.lower(), "").strip()
        value = scoped or plain
        if value:
            resolved[key] = value
            resolved[key.lower()] = value
    return resolved


def _redact_secrets(text: str) -> str:
    """Mask credential-shaped substrings in captured diagnostic text."""
    redacted = re.sub(
        r"(?i)\b((?:proxy-)?authorization\s*[=:]\s*)"
        r"(?:bearer|basic|digest|token)\s+\S+",
        r"\1[redacted]",
        text,
    )
    redacted = re.sub(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/=-]+", r"\1[redacted]", redacted)
    redacted = re.sub(
        r"(?i)\b((?:api[_-]?key|token|secret|password|authorization)\s*[=:]\s*)\S+",
        r"\1[redacted]",
        redacted,
    )
    redacted = re.sub(r"(?i)\b(sk-[A-Za-z0-9_\-]{6,})\b", "[redacted]", redacted)
    return re.sub(r"(https?://[^:/\s]+:)[^@\s]+@", r"\1[redacted]@", redacted)


def _cli_watchdog_from_env(
    overrides: Mapping[str, str] | None = None,
) -> StreamWatchdogConfig:
    """Build CLI streaming watchdog settings from explicit environment overrides."""
    startup_raw = _config_value(overrides, f"{_LLM_ENV_PREFIX}STARTUP_TIMEOUT_SEC", default="90")
    idle_raw = _config_value(overrides, f"{_LLM_ENV_PREFIX}IDLE_STALL_TIMEOUT_SEC", default="120")
    try:
        startup_sec = float(startup_raw)
        idle_sec = float(idle_raw)
    except ValueError:
        raise ValueError("LLM CLI watchdog timeouts must be numeric") from None
    if not all(math.isfinite(value) and value > 0 for value in (startup_sec, idle_sec)):
        raise ValueError("LLM CLI watchdog timeouts must be finite and positive")
    max_503_retries, backoff_503, jitter_503 = _capacity_503_policy_from_env(overrides)
    return StreamWatchdogConfig(
        startup_timeout_sec=startup_sec,
        idle_stall_timeout_sec=idle_sec,
        capacity_503_max_retries=max_503_retries,
        capacity_503_backoff_sec=backoff_503,
        capacity_503_jitter_sec=jitter_503,
    )


def _claude_stream_text(
    event: Mapping[str, object],
    assistant_text: list[str],
) -> str | None:
    """Accumulate assistant text from a claude stream-json event."""
    event_type = event.get("type")
    if event_type == "assistant":
        message = event.get("message")
        if isinstance(message, Mapping):
            content = message.get("content")
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, Mapping) and block.get("type") == "text":
                        text = block.get("text")
                        if isinstance(text, str):
                            assistant_text.append(text)
        return None
    if event_type == "result":
        result = event.get("result")
        return result if isinstance(result, str) else None
    return None


class SubprocessCliTemplateBackend:
    """Run a single generation in an empty working directory using an argument vector."""

    def __init__(
        self,
        kind: str,
        routing: RoleRoutingConfig,
        *,
        bin_path: str | None = None,
        template: str | None = None,
        max_concurrency: int = 2,
        watchdog: StreamWatchdogConfig | None = None,
        fallback_model: str | None = None,
        overrides: Mapping[str, str] | None = None,
    ) -> None:
        if kind not in ("claude_cli", "codex_cli", "custom_cli"):
            raise ValueError("Unknown LLM CLI profile")
        self._kind = kind
        self._routing = routing
        self._overrides = dict(overrides) if overrides else None
        raw_fallback = (
            fallback_model
            if fallback_model is not None
            else _config_value(self._overrides, f"{_LLM_ENV_PREFIX}FALLBACK_MODEL")
        ).strip()
        self._fallback_model = raw_fallback
        if (
            watchdog is not None
            and watchdog.capacity_503_max_retries is None
            and watchdog.max_retries > 0
        ):
            max_503, backoff_503, jitter_503 = _capacity_503_policy_from_env(self._overrides)
            self._watchdog: StreamWatchdogConfig | None = replace(
                watchdog,
                capacity_503_max_retries=max_503,
                capacity_503_backoff_sec=backoff_503,
                capacity_503_jitter_sec=jitter_503,
            )
        else:
            self._watchdog = watchdog
        self._template: list[str] = []
        self._template_fields: set[str] = set()
        configured_bin = bin_path or _config_value(self._overrides, f"{_LLM_ENV_PREFIX}CLI_BIN")
        if kind == "custom_cli":
            raw_template = template or _config_value(
                self._overrides, f"{_LLM_ENV_PREFIX}CLI_TEMPLATE"
            )
            try:
                self._template = shlex.split(raw_template)
                fields = {"bin", "model", "prompt", "effort"}
                for arg in self._template:
                    for _, name, spec, conversion in Formatter().parse(arg):
                        if name is None:
                            continue
                        if name not in fields or spec or conversion:
                            raise ValueError
                        self._template_fields.add(name)
                if not self._template:
                    raise ValueError
                executable = self._template[0]
                if executable == "{bin}":
                    executable = configured_bin
                elif "{" in executable or "}" in executable:
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
            values = {"bin": self._bin, "model": model, "effort": effort, "prompt": prompt}
            argv = [self._bin, *(arg.format_map(values) for arg in self._template[1:])]
            stdin = None if "prompt" in self._template_fields else prompt.encode("utf-8")
            return argv, stdin

        if self._kind == "claude_cli":
            argv = [
                self._bin,
                "--print",
                "--output-format",
                "stream-json",
                "--verbose",
                "--include-partial-messages",
                "--tools",
                "",
                "--strict-mcp-config",
                "--disable-slash-commands",
                "--no-session-persistence",
                "--setting-sources",
                "",
                "--permission-mode",
                "dontAsk",
                "--no-chrome",
            ]
            if effort:
                argv.append(f"--effort={effort}")
        else:
            argv = [
                self._bin,
                "--no-daemon",
                "--ask-for-approval",
                "never",
                "exec",
                "--sandbox",
                "read-only",
                "--skip-git-repo-check",
                "--ephemeral",
                "--ignore-user-config",
                "--ignore-rules",
                "--json",
                "-c",
                'web_search="disabled"',
                "-c",
                "mcp_servers={}",
            ]
            for feature in (
                "shell_tool",
                "unified_exec",
                "multi_agent",
                "multi_agent_v2",
                "apps",
                "plugins",
                "hooks",
                "browser_use",
                "image_generation",
                "view_image",
            ):
                argv.extend(("--disable", feature))
            if effort:
                argv.extend(("-c", f"model_reasoning_effort={json.dumps(effort)}"))
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
            event_type = event.get("type")
            item = event.get("item")
            if event_type in ("error", "turn.failed"):
                raise RuntimeError("Codex CLI reported a generation failure")
            if isinstance(item, dict):
                if item.get("type") not in ("agent_message", "reasoning"):
                    raise RuntimeError("Codex CLI attempted to use a tool during generation")
                if event_type == "item.completed" and item.get("type") == "agent_message":
                    value = item.get("text")
                    if isinstance(value, str):
                        text = value
        return text.strip()

    def watchdog_for_role(self, role: str) -> StreamWatchdogConfig:
        """Return streaming watchdog settings bounded by the role timeout."""
        base = self._watchdog or _cli_watchdog_from_env(self._overrides)
        ceiling = self._routing.timeout_for_role(role)
        return StreamWatchdogConfig(
            startup_timeout_sec=min(base.startup_timeout_sec, ceiling),
            idle_stall_timeout_sec=min(base.idle_stall_timeout_sec, ceiling),
            active_ceiling_timeout_sec=ceiling,
            max_503_hits=base.max_503_hits,
            max_sandbox_tool_calls=base.max_sandbox_tool_calls,
            max_retries=base.max_retries,
            retry_backoff_sec=base.retry_backoff_sec,
            capacity_503_max_retries=base.capacity_503_max_retries,
            capacity_503_backoff_sec=base.capacity_503_backoff_sec,
            capacity_503_jitter_sec=base.capacity_503_jitter_sec,
        )

    async def _run_streaming(
        self,
        argv: list[str],
        stdin: bytes | None,
        cfg: StreamWatchdogConfig,
        *,
        role: str,
        model: str,
    ) -> tuple[str, float | None]:
        """Run the CLI with dual-pipe heartbeat watchdogs and redacted diagnostics."""
        with tempfile.TemporaryDirectory(prefix="tf_llm_worker_") as sandbox_dir:
            env = {**os.environ, "PWD": sandbox_dir}
            env.pop("CLAUDECODE", None)
            env.update(_proxy_env(os.environ))
            spawn = asyncio.create_task(
                asyncio.create_subprocess_exec(
                    *argv,
                    stdin=asyncio.subprocess.DEVNULL if stdin is None else asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=sandbox_dir,
                    env=env,
                    start_new_session=True,
                    limit=16 * 1024 * 1024,
                )
            )
            try:
                proc = await asyncio.shield(spawn)
            except asyncio.CancelledError:
                with suppress(OSError, ValueError):
                    proc = await spawn
                    await asyncio.gather(_terminate_process_group(proc), proc.communicate())
                raise
            except (OSError, ValueError):
                raise RuntimeError("Could not start the configured LLM CLI") from None
            assert proc.stdout is not None
            assert proc.stderr is not None

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

            async def _feed_stdin() -> None:
                assert stdin is not None
                assert proc.stdin is not None
                with suppress(BrokenPipeError, ConnectionResetError):
                    proc.stdin.write(stdin)
                    await proc.stdin.drain()
                    proc.stdin.close()

            pump_out = asyncio.create_task(_pump(proc.stdout, "stdout"))
            pump_err = asyncio.create_task(_pump(proc.stderr, "stderr"))
            stdin_task = asyncio.create_task(_feed_stdin()) if stdin is not None else None

            start_ts = time.monotonic()
            last_activity_ts = start_ts
            ttft_sec: float | None = None
            stdout_chunks: list[str] = []
            assistant_text: list[str] = []
            result_text: str | None = None
            stream_error: str | None = None
            stderr_lines: list[str] = []
            stdout_eof = False
            stderr_eof = False
            stream_events_seen = False
            corrupt_stream = False
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
                            if line.strip():
                                now_line = time.monotonic()
                                if ttft_sec is None:
                                    ttft_sec = now_line - start_ts
                                last_activity_ts = now_line
                                stdout_chunks.append(line)
                                if self._kind == "claude_cli":
                                    stripped = line.strip()
                                    try:
                                        event: object = json.loads(stripped)
                                    except json.JSONDecodeError:
                                        if stripped.startswith(("{", "[")):
                                            corrupt_stream = True
                                    else:
                                        if isinstance(event, Mapping) and isinstance(
                                            event.get("type"), str
                                        ):
                                            stream_events_seen = True
                                            extracted = _claude_stream_text(
                                                event, assistant_text
                                            )
                                            if extracted is not None:
                                                result_text = extracted
                                            if (
                                                event.get("type") == "result"
                                                and event.get("is_error") is True
                                            ):
                                                stream_error = str(
                                                    event.get("result")
                                                    or event.get("subtype")
                                                    or "error"
                                                )
                    elif tag == "stderr":
                        if line is None:
                            stderr_eof = True
                        else:
                            cleaned = line.rstrip("\r\n")
                            if cleaned:
                                stderr_lines = [*stderr_lines, cleaned][-40:]

                    now = time.monotonic()
                    if (now - start_ts) > cfg.active_ceiling_timeout_sec:
                        await _terminate_process_group(proc)
                        raise _AttemptFailure(
                            TimeoutError(
                                "LLM CLI generation timed out: active_ceiling_timeout "
                                f"({cfg.active_ceiling_timeout_sec}s) exceeded "
                                f"(role={role}, model={model})"
                            ),
                            retriable=True,
                            ttft_sec=ttft_sec,
                            timeout_stage="active_ceiling",
                            since_last_activity_sec=now - last_activity_ts,
                        )
                    if (
                        proc.returncode is None
                        and ttft_sec is None
                        and (now - start_ts) > cfg.startup_timeout_sec
                    ):
                        await _terminate_process_group(proc)
                        raise _AttemptFailure(
                            TimeoutError(
                                "LLM CLI generation timed out: startup_timeout "
                                f"({cfg.startup_timeout_sec}s) exceeded "
                                f"(role={role}, model={model})"
                            ),
                            retriable=True,
                            ttft_sec=ttft_sec,
                            timeout_stage="startup",
                            since_last_activity_sec=now - last_activity_ts,
                        )
                    if (
                        proc.returncode is None
                        and (now - last_activity_ts) > cfg.idle_stall_timeout_sec
                    ):
                        await _terminate_process_group(proc)
                        raise _AttemptFailure(
                            TimeoutError(
                                "LLM CLI generation timed out: idle_stall_timeout "
                                f"({cfg.idle_stall_timeout_sec}s) exceeded "
                                f"(role={role}, model={model})"
                            ),
                            retriable=True,
                            ttft_sec=ttft_sec,
                            timeout_stage="idle_stall",
                            since_last_activity_sec=now - last_activity_ts,
                        )

                remaining_budget = cfg.active_ceiling_timeout_sec - (
                    time.monotonic() - start_ts
                )
                try:
                    await asyncio.wait_for(proc.wait(), timeout=max(0.0, remaining_budget))
                except asyncio.TimeoutError:
                    timed_out_at = time.monotonic()
                    await _terminate_process_group(proc)
                    raise _AttemptFailure(
                        TimeoutError(
                            "LLM CLI generation timed out: active_ceiling_timeout "
                            f"({cfg.active_ceiling_timeout_sec}s) exceeded after "
                            f"stream EOF (role={role}, model={model})"
                        ),
                        retriable=True,
                        ttft_sec=ttft_sec,
                        timeout_stage="after_eof",
                        since_last_activity_sec=timed_out_at - last_activity_ts,
                    ) from None
                stderr_tail = _redact_secrets("\n".join(stderr_lines))[-2000:]
                raw_output = "".join(stdout_chunks)
                combined_diag = f"{stderr_tail}\n{raw_output}"
                if proc.returncode:
                    detail = f"LLM CLI generation failed with exit status {proc.returncode}"
                    if stderr_tail:
                        detail = f"{detail}: {stderr_tail}"
                    is_503_err = bool(RE_503_CAPACITY.search(combined_diag))
                    is_retriable = is_503_err or bool(RE_STREAM_IDLE_LOG.search(combined_diag))
                    raise _AttemptFailure(
                        RuntimeError(detail),
                        retriable=is_retriable,
                        is_503=is_503_err,
                        ttft_sec=ttft_sec,
                    )
                if stream_error is not None:
                    detail = f"LLM CLI generation failed: {_redact_secrets(stream_error)[:500]}"
                    if stderr_tail:
                        detail = f"{detail}: {stderr_tail}"
                    err_blob = f"{stream_error}\n{combined_diag}"
                    is_503_err = bool(RE_503_CAPACITY.search(err_blob))
                    is_retriable = is_503_err or bool(RE_STREAM_IDLE_LOG.search(err_blob))
                    raise _AttemptFailure(
                        RuntimeError(detail),
                        retriable=is_retriable,
                        is_503=is_503_err,
                        ttft_sec=ttft_sec,
                    )
                if self._kind != "claude_cli":
                    return raw_output.strip(), ttft_sec
                output = (result_text or "".join(assistant_text)).strip()
                if output:
                    return output, ttft_sec
                if stream_events_seen or corrupt_stream:
                    detail = "LLM CLI returned no assistant text"
                    if stderr_tail:
                        detail = f"{detail}: {stderr_tail}"
                    raise _AttemptFailure(
                        RuntimeError(detail),
                        retriable=True,
                        ttft_sec=ttft_sec,
                    )
                return raw_output.strip(), ttft_sec
            finally:
                await _terminate_process_group(proc)
                pending = [pump_out, pump_err]
                if stdin_task is not None:
                    pending.append(stdin_task)
                for task in pending:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*pending, return_exceptions=True)

    async def generate(self, req: GenerationRequest, *, effort_override: str | None = None) -> str:
        """Route a generation request to the CLI and return its final response text."""
        model = self._routing.model_for_role(req.role)
        effort = self._routing.effort_for_request(req, effort_override)
        prompt = (
            f"{_STATELESS_INSTRUCTION}\n\n### SYSTEM INSTRUCTIONS\n{_system_prompt(req)}\n\n"
            f"### USER INPUT\n{req.user_prompt}\n"
        )
        cfg = self.watchdog_for_role(req.role)
        proxy_env = _proxy_env(os.environ)
        proxy_configured = (
            "true"
            if any(k in proxy_env for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"))
            else "false"
        )

        async def _attempt(model_override: str | None) -> str:
            active_model = model_override or model
            argv, stdin = self._command(active_model, effort, prompt)
            start_ts = time.monotonic()
            try:
                output, ttft_sec = await self._run_streaming(
                    argv,
                    stdin,
                    cfg,
                    role=req.role,
                    model=active_model,
                )
                text = output.strip()
                if self._kind == "codex_cli":
                    text = self._codex_text(text)
                if not text:
                    raise _AttemptFailure(
                        RuntimeError("LLM CLI returned no text completion"),
                        retriable=False,
                        ttft_sec=ttft_sec,
                    )
            except _AttemptFailure as failure:
                elapsed = time.monotonic() - start_ts
                status = "timeout" if isinstance(failure.exc, TimeoutError) else "error"
                ttft = failure.ttft_sec if failure.ttft_sec is not None else -1.0
                _log_cli_telemetry(
                    backend=self._kind,
                    role=req.role,
                    model=active_model,
                    ttft_sec=ttft,
                    elapsed_sec=elapsed,
                    status=status,
                    proxy_configured=proxy_configured,
                    timeout_stage=failure.timeout_stage,
                    since_last_activity_sec=failure.since_last_activity_sec,
                )
                raise
            elapsed = time.monotonic() - start_ts
            ttft = ttft_sec if ttft_sec is not None else elapsed
            _log_cli_telemetry(
                backend=self._kind,
                role=req.role,
                model=active_model,
                ttft_sec=ttft,
                elapsed_sec=elapsed,
                status="ok",
                proxy_configured=proxy_configured,
                timeout_stage=None,
                since_last_activity_sec=None,
            )
            return text

        return await _run_with_503_retry(
            sem=self._sem,
            cfg=cfg,
            fallback_model=self._fallback_model,
            attempt_fn=_attempt,
        )


@dataclass
class MockLLMBackend:
    """Deterministic scriptable LLM backend for offline orchestrator testing."""

    responses: dict[str, list[str]] = field(default_factory=dict)
    default_response: str = ""
    calls: list[GenerationRequest] = field(default_factory=list)
    routed_models: list[str] = field(default_factory=list)
    routed_efforts: list[str] = field(default_factory=list)
    routing: RoleRoutingConfig = field(
        default_factory=lambda: RoleRoutingConfig(
            "mock-exploit", "mock-explore", "mock-escalate", "mock-repair"
        )
    )

    async def generate(
        self,
        req: GenerationRequest,
        *,
        effort_override: str | None = None,
    ) -> str:
        """Record a request and return the next queued response for its role."""
        self.calls.append(req)
        default = "high" if req.role in _REASONING_ROLES else "medium"
        self.routed_models.append(self.routing.model_for_role(req.role))
        self.routed_efforts.append(
            self.routing.effort_for_request(req, effort_override, default=default)
        )
        queue = self.responses.get(req.role)
        if queue is None and req.role == "escalate_trackb":
            queue = self.responses.get("explore_creative")
        if queue:
            return queue.pop(0)
        return self.default_response


def create_llm_backend_from_env(overrides: Mapping[str, str] | None = None) -> KernelLLMBackend:
    """Create a generation provider using explicit overrides and environment settings."""

    def pick(key: str, default: str = "", *, legacy: str = "") -> str:
        keys = [f"{_LLM_ENV_PREFIX}{key}"]
        if legacy:
            keys.append(f"{_AGY_ENV_PREFIX}{legacy}")
        return _config_value(overrides, *keys, default=default)

    kind = pick("BACKEND", "agy_cli").lower()
    routing = RoleRoutingConfig.from_env(overrides, backend=kind)
    if kind in ("agy_cli", "agy"):
        return AgyCliLLMBackend(
            bin_path=pick("CLI_BIN", legacy="BIN") or None,
            routing=routing,
            common_args=_config_value(overrides, f"{_AGY_ENV_PREFIX}COMMON_ARGS") or None,
            max_concurrency=int(pick("MAX_CONCURRENCY", "3", legacy="MAX_CONCURRENCY")),
            fallback_model=pick("FALLBACK_MODEL") or None,
            overrides=overrides,
        )
    if kind == "mock":
        return MockLLMBackend(routing=routing)
    if kind in ("openai_compat", "openai", "vllm", "sglang", "http"):
        return OpenAICompatHTTPBackend(
            routing,
            api_base=pick("API_BASE") or None,
            api_key_env=pick("API_KEY_ENV") or None,
            max_concurrency=int(pick("MAX_CONCURRENCY", "4")),
            pass_reasoning_effort=pick("PASS_REASONING_EFFORT") == "1",
            stream=pick("STREAM") == "1",
        )
    if kind in ("claude_cli", "codex_cli", "custom_cli"):
        return SubprocessCliTemplateBackend(
            kind,
            routing,
            bin_path=pick("CLI_BIN") or None,
            template=pick("CLI_TEMPLATE") or None,
            max_concurrency=int(pick("MAX_CONCURRENCY", "2")),
            fallback_model=pick("FALLBACK_MODEL") or None,
            overrides=overrides,
        )
    raise ValueError("Unsupported TILEFOUNDRY_NPU_LLM_BACKEND value")


__all__ = [
    "AgyCliLLMBackend",
    "CREATIVE_RESPONSE_SCHEMA",
    "GenerationRequest",
    "KernelLLMBackend",
    "LLMGenerationError",
    "LLMTimeoutError",
    "MockLLMBackend",
    "OpenAICompatHTTPBackend",
    "ProseRefusalError",
    "RE_503_CAPACITY",
    "RE_STREAM_IDLE_LOG",
    "RoleRoutingConfig",
    "StreamWatchdogConfig",
    "SubprocessCliTemplateBackend",
    "create_llm_backend_from_env",
    "extract_creative_payload",
    "extract_python_dsl",
    "sanitize_effort",
]
