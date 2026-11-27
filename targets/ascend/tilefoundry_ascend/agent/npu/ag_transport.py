"""AgentGates CLI transport, worktree source passthrough, and device pool management."""

from __future__ import annotations

import fcntl
import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import subprocess
import time
import tomllib
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from tilefoundry_ascend.agent.specs.anchor_specs import parse_simple_env_file

logger = logging.getLogger(__name__)

DEFAULT_AGENT_GATES_MAX_ATTEMPTS = 5
DEFAULT_AGENT_GATES_BACKOFF_SEC = 3.0
AGENT_GATES_ADAPTER = "tilefoundry-npu"
AGENT_GATES_DEVICE_FAST_GATE = "tilefoundry-npu-device-fast"
AGENT_GATES_L3_PROMOTION_GATE = "tilefoundry-npu-l3-promotion"
AGENT_GATES_L4_CANNBENCH_GATE = "tilefoundry-npu-cannbench"
AGENT_GATES_L4_CANNBENCH_PREP_GATE = "tilefoundry-npu-cannbench-prep"
FAST_REQUEST_SCHEMA = "tilefoundry-fast-request/v1"
FAST_RESPONSE_SCHEMA = "tilefoundry-fast-response/v1"
FAST_REQUEST_ENV = "TF_FAST_REQUEST"
FAST_SOURCE_ROOT_ENV = "TF_FAST_SOURCE_ROOT"
FAST_STATUS_OK = "ok"
FAST_STATUS_CANDIDATE_FAILURE = "candidate_failure"
FAST_STATUS_REQUEST_ERROR = "request_error"
FAST_STATUS_INFRASTRUCTURE_ERROR = "infrastructure_error"
DEFAULT_L3_PROMOTION_WAIT_SEC = 1920
DEFAULT_L4_CANNBENCH_WAIT_SEC = 5400
FAILURE_KIND_CANDIDATE = "candidate"
FAILURE_KIND_INFRASTRUCTURE = "infrastructure"

_V2_WAIT_SLICE_SEC = 60.0
_V2_SUBMIT_TIMEOUT_SEC = 60.0
_V2_POST_WAIT_TIMEOUT_SEC = 60.0
_V2_STRIPPED_PYTHON_ENV_KEYS = frozenset({"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE"})
_AGENT_GATES_CONFIG_ENV = "AGENT_GATES_CONFIG"
_FAST_ID_SANITIZER = re.compile(r"[^A-Za-z0-9_.-]+")
_LLM_API_KEY_ENV = "TILEFOUNDRY_NPU_LLM_API_KEY_ENV"
_NON_AGENT_GATES_ENV_KEYS = frozenset({"TILEFOUNDRY_NPU_AGENT_GATES_BIN", "TILEFOUNDRY_NPU_LLM_API_KEY"})
_REF_CACHE_THREAD_ENV_KEYS = frozenset(
    {"OMP_NUM_THREADS", "MKL_NUM_THREADS", "ATEN_CPU_CAPABILITY", "TORCH_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"}
)
_STRIPPED_HOST_CPU_ENV_KEYS = frozenset({"ASCEND_RT_VISIBLE_DEVICES", "TILELANG_DISABLE_CACHE", "TORCH_DEVICE_BACKEND_AUTOLOAD", "USE_CUDA"})


class AgentGatesDeviceFaultError(RuntimeError):
    """Structured error raised when an NPU device reports device_fault or rejection."""

    def __init__(
        self, message: str, *, fault_plane: str = "INFRA_DEVICE_FAULT", fault_code: str = "NPU_DEVICE_TSD_FAULT",
        failed_tier: str = "l2_npu", device_id: str = "", rejected_cards: Collection[str] = (),
    ) -> None:
        super().__init__(message)
        self.fault_plane, self.fault_code, self.failed_tier, self.device_id = fault_plane, fault_code, failed_tier, device_id
        self.rejected_cards = tuple(sorted(str(c) for c in rejected_cards))


class OccupancyExhaustedError(RuntimeError):
    """Raised when NPU host-gate contention or BLOCKED_ENV (rc=75) exhausts retries."""

    def __init__(self, attempts: int, receipt_reasons: tuple[str, ...]) -> None:
        self.fault_plane, self.exit_code, self.attempts, self.receipt_reasons = "INFRA_OCCUPANCY", 75, attempts, receipt_reasons
        super().__init__(f"NPU occupancy/env blocked after {attempts} attempts (rc=75): [{', '.join(receipt_reasons)}]")


@dataclass(frozen=True)
class _V2GateExecution:
    """Outcome of one AgentGates submit/wait/report/verify sequence."""

    execution_id: str = ""
    run_id: str = ""
    receipt_path: str = ""
    report: Mapping[str, object] = field(default_factory=dict)
    verification: Mapping[str, object] = field(default_factory=dict)
    transport_error: str = ""
    elapsed_sec: float = 0.0
    device_id: str = ""
    state: str = ""
    execution_terminal: bool = True
    device_rejected: bool = False

    @property
    def execution_active(self) -> bool:
        """Return whether an execution may still be running and must be resumed."""
        return bool(self.execution_id) and not self.execution_terminal

    @property
    def verified(self) -> bool:
        """Return whether verify-requirements proved this run against current sources."""
        return not self.transport_error and self.execution_id != "" and self.verification.get("satisfied") is True


def wait_for_npu_quiescent(max_retries: int = 10) -> None:
    """Wait for NPU devices (davinci) to be closed in /proc/self/fd/."""
    for _ in range(max_retries):
        found = False
        try:
            for fd in Path("/proc/self/fd").iterdir():
                try:
                    if "davinci" in str(fd.readlink()):
                        found = True; break
                except OSError:
                    continue
        except OSError:
            pass
        if not found:
            return
        time.sleep(1.0)
    logger.warning("NPU still busy after waiting for quiescence.")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_py_exec_argv(python_bin: str, script_path: Path, cann_setup: str = "") -> list[str]:
    if cann_setup and Path(cann_setup).is_file() and shutil.which("bisheng") is None:
        return ["/bin/bash", "-c", f"set +u && source {shlex.quote(cann_setup)} >/dev/null 2>&1 && exec {shlex.quote(python_bin)} {shlex.quote(str(script_path))}"]
    return [python_bin, str(script_path)]


def _git_common_dir(path: Path) -> Path | None:
    """Return the resolved git common directory for a checkout or worktree."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env["GIT_OPTIONAL_LOCKS"] = "0"
    try:
        proc = subprocess.run(["git", "-C", str(path), "rev-parse", "--path-format=absolute", "--git-common-dir"], capture_output=True, text=True, check=False, env=env, timeout=10.0)
    except (OSError, subprocess.SubprocessError):
        return None
    return Path(raw).resolve() if proc.returncode == 0 and (raw := proc.stdout.strip()) else None


def _resolve_agent_gates_bin(source_root: Path | None = None) -> str:
    """Resolve the AgentGates CLI from explicit configuration, PATH, or sibling venv."""
    if (env_bin := os.environ.get("TILEFOUNDRY_NPU_AGENT_GATES_BIN", "").strip()) and Path(env_bin).is_file():
        return env_bin
    if on_path := shutil.which("agent-gates"):
        return on_path
    candidates: list[Path] = []
    if env_raw := os.environ.get("TILEFOUNDRY_NPU_ENV_FILE", "").strip():
        env_path = Path(env_raw).expanduser().resolve()
        candidates.extend(p / ".venv" / "bin" / "agent-gates" for p in env_path.parents)
        if env_path.is_file() and (cbin := parse_simple_env_file(env_path).get("TILEFOUNDRY_NPU_AGENT_GATES_BIN", "").strip()):
            candidates.insert(0, Path(cbin).expanduser())
    if source_root is not None:
        rsrc = Path(source_root).expanduser().resolve()
        candidates.append(rsrc.parent / "AgentGates" / ".venv" / "bin" / "agent-gates")
        if (cdir := _git_common_dir(rsrc)) is not None:
            candidates.append(cdir.parent.parent / "AgentGates" / ".venv" / "bin" / "agent-gates")
    candidates.append(Path(__file__).resolve().parents[5].parent / "AgentGates" / ".venv" / "bin" / "agent-gates")
    for cand in candidates:
        if cand.is_file():
            return str(cand)
    raise FileNotFoundError("Cannot locate 'agent-gates' binary. Set TILEFOUNDRY_NPU_AGENT_GATES_BIN or PATH.")


def _resolve_env_file_path(env_file: str | Path | None = None) -> Path:
    raw = str(env_file) if env_file is not None else os.environ.get("TILEFOUNDRY_NPU_ENV_FILE", "").strip()
    if raw and Path(raw).is_file():
        return Path(raw).resolve()
    raise FileNotFoundError("Valid tilefoundry-npu env.sh file is required for AgentGates runs.")


def _resolve_env_values(env_file: str | Path | None = None) -> dict[str, str]:
    raw = str(env_file) if env_file is not None else os.environ.get("TILEFOUNDRY_NPU_ENV_FILE", "").strip()
    return parse_simple_env_file(raw) if raw else {}


def _validate_llm_api_key_env(name: str) -> None:
    if name and (not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) or name.startswith("TILEFOUNDRY_NPU_")):
        raise ValueError(f"{_LLM_API_KEY_ENV} must name a separate environment variable")


def _format_env_line(key: str, value: str) -> str:
    if any(ch in value for ch in "\n\r\x00"):
        raise ValueError(f"{key} must be single-line")
    return f"{key}={shlex.quote(value)}"


def _write_temp_env_file(base_env: Path, dest_path: Path, overrides: Mapping[str, str]) -> Path:
    values = {**parse_simple_env_file(base_env), **overrides}
    _validate_llm_api_key_env(values.get(_LLM_API_KEY_ENV, "").strip())
    stripped = set(overrides) | _NON_AGENT_GATES_ENV_KEYS
    lines = [ln for ln in base_env.read_text(encoding="utf-8").splitlines() if ln.partition("=")[0].strip().removeprefix("export ").strip() not in stripped]
    lines.extend(_format_env_line(k, v) for k, v in overrides.items() if k not in _NON_AGENT_GATES_ENV_KEYS)
    dest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dest_path


def _clean_agent_gates_subprocess_env(extra_env: Mapping[str, str]) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("TILEFOUNDRY_NPU_") and k not in _STRIPPED_HOST_CPU_ENV_KEYS and k not in _REF_CACHE_THREAD_ENV_KEYS}
    env.setdefault("PYTHONNOUSERSITE", "1")
    env.update(extra_env)
    return env


def _agent_gates_cli_env() -> dict[str, str]:
    """Build the CLI environment without project Python startup injection."""
    env = _clean_agent_gates_subprocess_env({})
    for key in _V2_STRIPPED_PYTHON_ENV_KEYS:
        env.pop(key, None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def _parse_agent_gates_json(stdout: str) -> dict[str, object]:
    text = (stdout or "").strip()
    for line in reversed(text.splitlines()):
        if (cand := line.strip()).startswith("{") and cand.endswith("}"):
            try:
                if isinstance((parsed := json.loads(cand)), dict):
                    return parsed
            except json.JSONDecodeError:
                continue
    start = (idx + 1) if (idx := text.rfind("\n{")) >= 0 else text.find("{")
    if start < 0:
        return {}
    try:
        return parsed if isinstance((parsed := json.loads(text[start:])), dict) else {}
    except json.JSONDecodeError:
        return {}


def _agent_gates_config() -> Mapping[str, object]:
    cfg = os.environ.get(_AGENT_GATES_CONFIG_ENV, "").strip()
    return tomllib.loads((Path(cfg).expanduser() if cfg else Path.home() / ".config/agent-gates/config.toml").read_text(encoding="utf-8"))


def _agent_gates_evidence_root() -> Path:
    """Read the configured AgentGates evidence root used by the managed worker."""
    er = _agent_gates_config().get("evidence_root")
    if not isinstance(er, str) or not er.strip():
        raise ValueError("AgentGates evidence_root must be configured")
    return Path(er).expanduser().resolve()


def _agent_gates_workspace_name(source_root: Path) -> str:
    """Resolve the configured AgentGates workspace whose source root is ``source_root``."""
    workspaces = _agent_gates_config().get("workspaces")
    if not isinstance(workspaces, Mapping):
        raise ValueError("AgentGates config must define [workspaces]")
    wanted = source_root.resolve()
    for name, entry in workspaces.items():
        if isinstance(entry, Mapping) and isinstance(cfg := entry.get("source_root"), str) and Path(cfg).expanduser().resolve() == wanted:
            return str(name)
    if (wanted_common := _git_common_dir(wanted)) is not None:
        for name, entry in workspaces.items():
            if isinstance(entry, Mapping) and entry.get("allow_worktrees") is not False and isinstance(cfg := entry.get("source_root"), str) and cfg.strip() and _git_common_dir(Path(cfg).expanduser().resolve()) == wanted_common:
                return str(name)
    raise ValueError(f"No AgentGates workspace is configured for source root {wanted}")


def _agent_gates_source_override(workspace: str, root: Path) -> str | None:
    """Return resolved root when caller checkout differs from workspace source_root."""
    try:
        workspaces = _agent_gates_config().get("workspaces")
    except (OSError, ValueError):
        return None
    entry = workspaces.get(workspace) if isinstance(workspaces, Mapping) else None
    cfg = entry.get("source_root") if isinstance(entry, Mapping) else None
    if not isinstance(cfg, str) or not cfg.strip():
        return None
    return str(rroot) if Path(cfg).expanduser().resolve() != (rroot := root.resolve()) else None


def _workspace_relative_candidate(source_root: Path, candidate_path: Path) -> str | None:
    try:
        return candidate_path.resolve().relative_to(source_root.resolve()).as_posix()
    except ValueError:
        return None


def _source_identity_matches(verification: Mapping[str, object], root: Path) -> bool:
    identity = verification.get("current_source_identity")
    ir = identity.get("root") if isinstance(identity, Mapping) else None
    return bool(ir) and Path(str(ir)).resolve() == root


def _inspect_v2_device_fault(report: Mapping[str, object] | None, verification: Mapping[str, object] | None = None) -> tuple[bool, bool, str]:
    """Inspect AgentGates managed receipts for device_fault or deviceRejected flags."""
    is_df, is_dr, category = False, False, ""

    def _scan(obj: Mapping[str, object] | None) -> None:
        nonlocal is_df, is_dr, category
        if not isinstance(obj, Mapping):
            return
        if obj.get("deviceRejected") is True or obj.get("device_rejected") is True:
            is_dr = True
        for key in ("errorCategory", "failureCategory"):
            if raw_cat := str(obj.get(key) or "").strip().lower():
                category = category or raw_cat
                if raw_cat == "device_fault":
                    is_df = True

    _scan(report); _scan(verification)
    if isinstance(report, Mapping) and isinstance(gates := report.get("gates"), Sequence) and not isinstance(gates, (str, bytes)):
        for row in gates:
            if isinstance(row, Mapping):
                _scan(row)
                if isinstance(obs := row.get("observations"), Mapping):
                    _scan(obs)
    return is_df, is_dr, category


def _is_transient_blocked_env(proc: subprocess.CompletedProcess[str], parsed: Mapping[str, object]) -> bool:
    if proc.returncode == 0 or any(_inspect_v2_device_fault(parsed)[:2]):
        return False
    if parsed.get("status") == "BLOCKED_ENV":
        return True
    if parsed.get("status") == "FAIL" and isinstance(gates := parsed.get("gates"), Sequence) and not isinstance(gates, (str, bytes)):
        for g in gates:
            if isinstance(g, Mapping) and g.get("gateId") == AGENT_GATES_DEVICE_FAST_GATE and g.get("status") == "NOT_RUN":
                if isinstance(f := g.get("failures"), Sequence) and not isinstance(f, (str, bytes)) and "DEPENDENCY_NOT_PASSED" in f:
                    return True
    return False


def _run_agent_gates_cmd(
    cmd: Sequence[str], *, cwd: str, env: Mapping[str, str], timeout_sec: int,
    max_attempts: int = DEFAULT_AGENT_GATES_MAX_ATTEMPTS, backoff_sec: float = DEFAULT_AGENT_GATES_BACKOFF_SEC,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
    wait_for_npu_quiescent()
    attempts, cmd_list, env_dict = max(1, int(max_attempts)), [str(p) for p in cmd], dict(env)
    last_proc: subprocess.CompletedProcess[str] | None = None
    last_parsed: dict[str, object] = {}
    for attempt in range(1, attempts + 1):
        last_proc = subprocess.run(cmd_list, text=True, capture_output=True, cwd=cwd, env=env_dict, timeout=timeout_sec)
        last_parsed = _parse_agent_gates_json(last_proc.stdout)
        if not (_is_transient_blocked_env(last_proc, last_parsed) and attempt < attempts):
            return last_proc, last_parsed
        delay = min(20.0, max(0.0, float(backoff_sec)) * (2 ** (attempt - 1)))
        logger.warning("AgentGates transient BLOCKED_ENV (attempt %d/%d); retrying in %.1fs", attempt, attempts, delay)
        if delay > 0.0:
            sleep_fn(delay)
    assert last_proc is not None
    return last_proc, last_parsed


def _run_agent_gates_v2_cmd(cmd: Sequence[str], *, root: Path, timeout_sec: float) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
    """Run one managed AgentGates command against the configured worker evidence."""
    proc = subprocess.run([str(p) for p in cmd], text=True, capture_output=True, cwd=str(root), env=_agent_gates_cli_env(), timeout=timeout_sec)
    return proc, _parse_agent_gates_json(proc.stdout)


def _normalize_device_pool(device_id: str, device_pool: Sequence[str] = (), excluded: Collection[str] = ()) -> tuple[str, ...]:
    """Return canonical deduplicated decimal card IDs excluding rejected cards."""
    ordered: list[str] = []
    for raw in (device_id, *device_pool):
        if (txt := str(raw or "").strip()).isdigit() and (canon := str(int(txt))) not in ordered and canon not in excluded:
            ordered.append(canon)
    return tuple(ordered)


def _run_npu_smi_query(smi: str, args: Sequence[str]) -> str | None:
    """Run read-only npu-smi query with short backoff retries on transient exit 215."""
    for attempt, delay in enumerate((0.15, 0.35, 0.0)):
        try:
            proc = subprocess.run([smi, *args], capture_output=True, text=True, timeout=10.0, check=False)
        except (OSError, subprocess.SubprocessError, TypeError):
            return None
        if proc.returncode == 0:
            return proc.stdout or ""
        if proc.returncode == 215 and attempt < 2:
            time.sleep(delay); continue
        return None
    return None


def _probe_host_card_load(device_id: str, *, max_quiet_hbm_mb: int | None = None, env_vals: Mapping[str, str] | None = None) -> tuple[int, int, int] | None:
    """Probe card lock/quarantine/process load as (tier, total_mem_mb, proc_count)."""
    if not (txt := str(device_id or "").strip()).isdigit():
        return None
    dev_idx = int(txt)
    if max_quiet_hbm_mb is not None:
        resolved_max_hbm = max(256, int(max_quiet_hbm_mb))
    else:
        raw_hbm = ((env_vals.get("TILEFOUNDRY_NPU_SHARED_MAX_HBM_MB", "") if env_vals else "") or os.environ.get("TILEFOUNDRY_NPU_SHARED_MAX_HBM_MB", "")).strip()
        try:
            resolved_max_hbm = max(256, int(raw_hbm)) if raw_hbm else 16384
        except ValueError:
            resolved_max_hbm = 16384
    lock_raw = ((env_vals.get("AGENT_GATES_DEVICE_LOCK_ROOT", "") if env_vals else "") or os.environ.get("AGENT_GATES_DEVICE_LOCK_ROOT", "")).strip()
    lock_root = Path(lock_raw).expanduser() if lock_raw else Path.home() / ".local/state/agent-gates/locks"
    lease_file = lock_root / f"{hashlib.sha256(f'npu:{dev_idx}'.encode()).hexdigest()}.lock"
    if Path(f"{lease_file}.quarantine.json").is_file():
        return (3, 999999, 999999)
    if lease_file.is_file():
        try:
            fd = os.open(lease_file, os.O_RDONLY)
        except OSError:
            return (3, 999999, 999999)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB); fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            return (3, 999999, 999999)
        finally:
            os.close(fd)
    smi_cmd = ((env_vals.get("AGENT_GATES_SMI_COMMAND", "") if env_vals else "").strip() or os.environ.get("AGENT_GATES_SMI_COMMAND") or "npu-smi")
    if (smi := shutil.which(smi_cmd)) is None or (map_out := _run_npu_smi_query(smi, ("info", "-m"))) is None:
        return None
    npu_ids: list[int] = []
    for line in map_out.splitlines():
        if len(f := line.split()) >= 4 and all(x.isdigit() for x in f[:4]):
            if int(f[3]) == dev_idx:
                npu_ids.append(int(f[0]))
        elif len(f) >= 3 and all(x.isdigit() for x in f[:3]) and int(f[2]) == dev_idx:
            npu_ids.append(int(f[0]))
    if len(unique_ids := sorted(set(npu_ids))) != 1:
        return None
    npu_id = unique_ids[0]
    if (proc_out := _run_npu_smi_query(smi, ("info", "-t", "proc-mem", "-i", str(npu_id)))) is None:
        return None
    if not re.search(rf"^\s*NPU ID\s*:\s*{npu_id}\s*$", proc_out, re.MULTILINE):
        return None
    if proc_ids := re.findall(r"Process id\s*:\s*\d+", proc_out):
        total_mem_mb = sum(int(v) for v in re.findall(r"Process memory\(MB\)\s*:\s*(\d+)", proc_out))
        return (1 if total_mem_mb <= resolved_max_hbm else 2, total_mem_mb, len(proc_ids))
    return (0, 0, 0) if re.search(r"^\s*No process in device\.?\s*$", proc_out, re.MULTILINE) else None


def _probe_host_card_idle(device_id: str) -> bool | None:
    """Return True if card has 0 compute processes, False if busy/locked, None if unknown."""
    load = _probe_host_card_load(device_id)
    return None if load is None else (load[0] == 0)


def order_candidates_by_idle_preflight(
    device_id: str, device_pool: Sequence[str] = (), excluded: Collection[str] = (), env_vals: Mapping[str, str] | None = None
) -> tuple[str, ...]:
    """Order candidate cards: free, unknown, quiet shared, heavy, locked."""
    candidates = _normalize_device_pool(device_id, device_pool, excluded=excluded)
    if len(candidates) <= 1:
        return candidates
    tier0, unknown, tier1, tier2, tier3 = [], [], [], [], []
    for idx, card in enumerate(candidates):
        if (load := _probe_host_card_load(card, env_vals=env_vals)) is None:
            unknown.append(card)
        elif load[0] == 0:
            tier0.append(card)
        elif load[0] in (1, 2):
            (tier1 if load[0] == 1 else tier2).append((load[1], load[2], idx, card))
        else:
            tier3.append(card)
    tier1.sort(); tier2.sort()
    return (*tier0, *unknown, *(r[3] for r in tier1), *(r[3] for r in tier2), *tier3)


def _call_order_candidates_by_idle_preflight(
    device_id: str, device_pool: Sequence[str], *, excluded: Collection[str], env_vals: Mapping[str, str] | None
) -> tuple[str, ...]:
    """Forward env_vals to order_candidates_by_idle_preflight when supported."""
    if env_vals is not None:
        try:
            return order_candidates_by_idle_preflight(device_id, device_pool, excluded=excluded, env_vals=env_vals)
        except TypeError:
            pass
    return order_candidates_by_idle_preflight(device_id, device_pool, excluded=excluded)


def select_idle_campaign_device(
    device_id: str, device_pool: Sequence[str] = (), excluded: Collection[str] = (), env_vals: Mapping[str, str] | None = None
) -> str:
    """Select the highest-priority idle or fallback device from the candidate pool."""
    ordered = _call_order_candidates_by_idle_preflight(device_id, device_pool, excluded=excluded, env_vals=env_vals)
    return ordered[0] if ordered else str(device_id or "").strip()


def _gate_row(report: Mapping[str, object], gate_id: str) -> Mapping[str, object] | None:
    return next((r for r in report.get("gates", ()) if isinstance(r, Mapping) and r.get("gateId") == gate_id), None)


def _gate_observation(report: Mapping[str, object], gate_id: str) -> tuple[Mapping[str, object] | None, Mapping[str, object]]:
    row = _gate_row(report, gate_id)
    if row is None:
        return None, {}
    obs = row.get("observations")
    return row, dict(obs) if isinstance(obs, Mapping) else {}


def _gate_recorded_status(row: Mapping[str, object] | None) -> str:
    return "" if row is None else str(row.get("recordedStatus") or row.get("status") or "")


def _gate_failure_text(row: Mapping[str, object]) -> str:
    raw = row.get("failures")
    return " ".join(str(x) for x in raw) if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) else str(raw or "")


def _blocked_env_category(
    report: Mapping[str, object],
    gate_ids: Sequence[str] = (AGENT_GATES_DEVICE_FAST_GATE, AGENT_GATES_L3_PROMOTION_GATE, AGENT_GATES_L4_CANNBENCH_GATE, AGENT_GATES_L4_CANNBENCH_PREP_GATE),
) -> str:
    """Return the signed BLOCKED_ENV category across device-fast, L3, and L4 gate rows."""
    for gid in gate_ids:
        row, obs = _gate_observation(report, gid)
        if row is not None and (_gate_recorded_status(row).upper() == "BLOCKED_ENV" or "BLOCKED_EXIT_CODE: 75" in _gate_failure_text(row)):
            if cat := str(obs.get("errorCategory") or obs.get("failureCategory") or "").strip().lower():
                return cat
    return ""


def _submit_device_rejected(proc: subprocess.CompletedProcess[str], parsed: Mapping[str, object]) -> bool:
    """Return True when AgentGates refused the named card as unauthorized."""
    return str(parsed.get("code", "")) == "DEVICE_NOT_AUTHORIZED" or "DEVICE_NOT_AUTHORIZED" in f"{proc.stdout}\n{proc.stderr}"


def _as_positive_int(value: object) -> int:
    return 0 if isinstance(value, bool) or not isinstance(value, (int, float)) else max(0, int(value))


def _run_v2_gate(
    *, agent_gates_bin: str, root: Path, workspace: str, gate_id: str, target: str,
    candidate_path: str | None = None, timeout_sec: float, device_id: str = "",
    resume_execution_id: str = "", resume_run_id: str = "",
    on_execution_started: Callable[[str, str], None] | None = None,
) -> _V2GateExecution:
    """Run one managed gate with a single monotonic deadline and one execution identity."""
    started = time.monotonic()
    deadline, requested_device = started + max(1.0, float(timeout_sec)), str(device_id or "").strip()
    source_override = _agent_gates_source_override(workspace, root)
    run_cmd = _run_agent_gates_v2_cmd

    def _failed(eid: str, msg: str, *, run_id: str = "", receipt: str = "", dev: str = "", state: str = "", terminal: bool = True, rejected: bool = False) -> _V2GateExecution:
        return _V2GateExecution(
            execution_id=eid, run_id=run_id, receipt_path=receipt, transport_error=msg[-1000:],
            elapsed_sec=time.monotonic() - started, device_id=dev or requested_device, state=state, execution_terminal=terminal, device_rejected=rejected,
        )

    def _step(cmd: list[str], eid: str, label: str, *, timeout: float, allow_fnf: bool = False, **fkw: object):
        try:
            return run_cmd(cmd, root=root, timeout_sec=timeout), None
        except FileNotFoundError:
            if allow_fnf:
                raise
            return None, _failed(eid, f"{label} failed: agent-gates missing", **fkw)
        except (OSError, subprocess.TimeoutExpired) as exc:
            return None, _failed(eid, f"{label} failed: {exc}", **fkw)

    if resume_execution_id:
        execution_id, resolved_device = resume_execution_id, requested_device
    else:
        c_args = ["--candidate", candidate_path] if candidate_path is not None else []
        d_args = ["--device-id", requested_device] if requested_device else []
        s_args = ["--source", source_override] if source_override is not None else []
        res, err = _step([agent_gates_bin, "doctor", "--adapter", AGENT_GATES_ADAPTER, "--source", str(root), "--gate", gate_id, "--target", target, *c_args, *d_args, "--json"], "", f"{gate_id} doctor", timeout=min(_V2_SUBMIT_TIMEOUT_SEC, max(1.0, deadline - started)), allow_fnf=True)
        if err is not None:
            return err
        dproc, doc = res
        if dproc.returncode != 0 or doc.get("status") != "PASS":
            rf = doc.get("failures")
            return _failed("", ("; ".join(str(x) for x in rf) if isinstance(rf, list) else "") or dproc.stderr or dproc.stdout or f"{gate_id} doctor did not pass for target {target}")
        res, err = _step([agent_gates_bin, "submit", "--workspace", workspace, "--adapter", AGENT_GATES_ADAPTER, "--gate", gate_id, "--target", target, *c_args, *d_args, *s_args, "--json"], "", f"{gate_id} submit", timeout=min(_V2_SUBMIT_TIMEOUT_SEC, max(1.0, deadline - started)), allow_fnf=True)
        if err is not None:
            return err
        proc, parsed = res
        if proc.returncode != 0:
            return _failed("", proc.stderr or proc.stdout or f"{gate_id} submit failed", rejected=_submit_device_rejected(proc, parsed))
        if not (execution_id := str(parsed.get("execution_id", ""))):
            return _failed("", f"{gate_id} submit returned no execution_id")
        resolved_device = str(parsed.get("device_id") or parsed.get("deviceId") or requested_device or "")
        if on_execution_started is not None:
            on_execution_started(execution_id, resolved_device)

    run_id, last_state = resume_run_id, ""
    while not run_id:
        if (rem := deadline - time.monotonic()) <= 0.0:
            return _failed(execution_id, f"{gate_id} execution {execution_id} wait timed out", dev=resolved_device, state=last_state or "RUNNING", terminal=last_state in {"FINISHED", "CANCELLED", "INTERRUPTED"})
        ws = min(_V2_WAIT_SLICE_SEC, rem)
        try:
            proc, parsed = run_cmd([agent_gates_bin, "wait", "--execution-id", execution_id, "--timeout", f"{ws:g}", "--json"], root=root, timeout_sec=ws + 5.0)
        except subprocess.TimeoutExpired:
            continue
        except OSError as exc:
            return _failed(execution_id, f"{gate_id} execution {execution_id} wait failed: {exc}", dev=resolved_device, state=last_state, terminal=False)
        if proc.returncode != 0:
            return _failed(execution_id, proc.stderr or proc.stdout or f"{gate_id} execution {execution_id} wait failed", dev=resolved_device, state=last_state, terminal=False)
        status = parsed.get("status")
        if state := (str(status.get("state") or "") if isinstance(status, Mapping) else ""):
            last_state = state
        if state == "FINISHED":
            run_id = str(status.get("run_id") or "")
            break
        if state in {"CANCELLED", "INTERRUPTED"}:
            return _failed(execution_id, f"{gate_id} execution {execution_id} ended in state {state!r}", dev=resolved_device, state=state)
        if parsed.get("timed_out") is True:
            continue
        return _failed(execution_id, f"{gate_id} execution {execution_id} wait returned no status" if not state else f"{gate_id} execution {execution_id} ended in state {state!r}", dev=resolved_device, state=state or last_state, terminal=bool(state))

    if not run_id:
        return _failed(execution_id, f"{gate_id} execution {execution_id} finished without run_id", dev=resolved_device, state=last_state or "FINISHED")
    try:
        evidence_root = _agent_gates_evidence_root()
    except (OSError, ValueError) as exc:
        return _failed(execution_id, f"{gate_id} evidence root unavailable: {exc}", run_id=run_id, dev=resolved_device, state=last_state or "FINISHED")
    res, err = _step([agent_gates_bin, "report", "--run-id", run_id, "--evidence-root", str(evidence_root), "--json"], execution_id, f"{gate_id} report", timeout=min(_V2_POST_WAIT_TIMEOUT_SEC, max(1.0, deadline - time.monotonic())), run_id=run_id, dev=resolved_device, state=last_state or "FINISHED")
    if err is not None:
        return err
    proc, report = res
    if not report or "receiptPath" not in report:
        return _failed(execution_id, proc.stderr or proc.stdout or f"{gate_id} report returned no receipt", run_id=run_id, dev=resolved_device, state=last_state or "FINISHED")
    receipt_path = str(report.get("receiptPath", ""))
    v_cmd = [agent_gates_bin, "verify-requirements", "--workspace", workspace, "--adapter", AGENT_GATES_ADAPTER, "--gate", gate_id, "--run-id", run_id, "--json"]
    if candidate_path is not None:
        v_cmd.extend(["--target", target, "--candidate", candidate_path])
    if source_override is not None:
        v_cmd.extend(["--source", source_override])
    res, err = _step(v_cmd, execution_id, f"{gate_id} requirement verification", timeout=min(_V2_POST_WAIT_TIMEOUT_SEC, max(1.0, deadline - time.monotonic())), run_id=run_id, receipt=receipt_path, dev=resolved_device, state=last_state or "FINISHED")
    if err is not None:
        return err
    proc, verification = res
    if not verification or "satisfied" not in verification:
        return _failed(execution_id, proc.stderr or proc.stdout or f"{gate_id} requirement verification returned no verdict", run_id=run_id, receipt=receipt_path, dev=resolved_device, state=last_state or "FINISHED")
    is_df, is_dr, _ = _inspect_v2_device_fault(report, verification)
    return _V2GateExecution(execution_id, run_id, receipt_path, report, verification, "", time.monotonic() - started, resolved_device, last_state or "FINISHED", True, bool(is_dr or is_df))


def _fast_slug(value: str, *, fallback: str) -> str:
    return _FAST_ID_SANITIZER.sub("_", value).strip("._") or fallback


def _fast_wire_operator_name(operator_name: str) -> str:
    """Return leaf operator name matching AgentGates FAST_NAME_RE without slashes."""
    return _fast_slug(str(operator_name or "").strip().rsplit("/", 1)[-1], fallback="operator")


def _fast_request_root(root: Path, env_vals: Mapping[str, str]) -> Path:
    """Resolve the request staging root under the AgentGates lease-managed build directory."""
    bsub = env_vals.get("TILEFOUNDRY_NPU_BUILD_SUBDIR", "").strip() or "build"
    if bsub in {".", ".."} or Path(bsub).name != bsub:
        raise ValueError(f"TILEFOUNDRY_NPU_BUILD_SUBDIR must be one relative directory name, got {bsub!r}")
    return root / bsub / "agent_gates" / "fast"


def _fast_cache_roots(root: Path, env_vals: Mapping[str, str]) -> tuple[Path, Path]:
    """Resolve the reference and TileLang caches the device-fast child also uses."""
    if not (raw := env_vals.get("TILEFOUNDRY_NPU_WORK_ROOT", "").strip()):
        raise ValueError("AgentGates machine env has no TILEFOUNDRY_NPU_WORK_ROOT for device-fast")
    if not (wr := Path(raw).expanduser()).is_absolute():
        raise ValueError(f"TILEFOUNDRY_NPU_WORK_ROOT must be an absolute path, got {raw!r}")
    if (res := wr.resolve()) == root or res.is_relative_to(root):
        raise ValueError(f"TILEFOUNDRY_NPU_WORK_ROOT must be outside the source root, got {res}")
    return res / "surrogate" / "reference", res / "surrogate" / "tilelang"


def _stage_fast_candidate(request_dir: Path, root: Path, operator_name: str, index: int, candidate_id: str, module_path: Path) -> tuple[str, str]:
    """Copy a candidate into the request staging tree, returning (relative path, sha256)."""
    src = module_path.resolve()
    if not src.is_file():
        raise FileNotFoundError(f"candidate module {src} does not exist")
    if src.suffix != ".py":
        raise ValueError(f"candidate module {src} must be a Python file")
    op_dir = request_dir / "candidates" / _fast_slug(operator_name, fallback="operator")
    op_dir.mkdir(parents=True, exist_ok=True)
    staged = op_dir / f"{index:03d}_{_fast_slug(candidate_id, fallback='candidate')}.py"
    if staged.exists():
        raise FileExistsError(f"staged candidate {staged} already exists")
    shutil.copyfile(src, staged)
    staged.chmod(0o444)
    return staged.relative_to(root).as_posix(), _sha256_file(staged)


def _write_fast_request(path: Path, payload: Mapping[str, object]) -> str:
    """Write the request once as an immutable descriptor and return its sha256."""
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
        stream.write(data); stream.flush(); os.fchmod(stream.fileno(), 0o444)
    return hashlib.sha256(data).hexdigest()


def _fast_response_mapping(raw: Mapping[str, object], key: str) -> Mapping[str, object]:
    if not isinstance(value := raw.get(key), Mapping):
        raise RuntimeError(f"fast response field {key!r} must be a mapping")
    return value


def _require_fast_envelope(raw: Mapping[str, object], *, request_id: str, request_sha256: str, mode: str) -> str:
    """Validate the response envelope against the submitted request and return its status."""
    if raw.get("schema") != FAST_RESPONSE_SCHEMA:
        raise RuntimeError(f"fast response schema {raw.get('schema')!r} != {FAST_RESPONSE_SCHEMA!r}")
    if raw.get("request_id") != request_id:
        raise RuntimeError(f"fast response request_id {raw.get('request_id')!r} != {request_id!r}")
    if raw.get("request_sha256") != request_sha256:
        raise RuntimeError("fast response request_sha256 does not match the submitted request")
    if raw.get("mode") != mode:
        raise RuntimeError(f"fast response mode {raw.get('mode')!r} != {mode!r}")
    if (st := raw.get("status")) not in (FAST_STATUS_OK, FAST_STATUS_CANDIDATE_FAILURE, FAST_STATUS_REQUEST_ERROR, FAST_STATUS_INFRASTRUCTURE_ERROR):
        raise RuntimeError(f"fast response status {st!r} is not recognized")
    return str(st)


def _raise_for_fast_status(status: str, raw: Mapping[str, object]) -> None:
    if status in (FAST_STATUS_REQUEST_ERROR, FAST_STATUS_INFRASTRUCTURE_ERROR):
        raise RuntimeError(f"fast gate {status}: {str(raw.get('error', '')).strip() or 'no detail'}")


def _verify_fast_bindings(raw: Mapping[str, object], expected: Mapping[str, Mapping[str, Mapping[str, str]]]) -> None:
    """Prove every evaluated candidate matches its staged path and content digest."""
    bindings = _fast_response_mapping(raw, "candidates")
    for op_name, candidates in expected.items():
        if not isinstance(op_b := bindings.get(op_name), Mapping):
            raise RuntimeError(f"fast response has no candidate bindings for operator {op_name!r}")
        if set(op_b) != set(candidates):
            raise RuntimeError(f"fast response candidate set for operator {op_name!r} does not match the request")
        for cid, b in candidates.items():
            if not isinstance(act := op_b.get(cid), Mapping):
                raise RuntimeError(f"fast response binding for {op_name}/{cid} is malformed")
            if act.get("module_path") != b["module_path"]:
                raise RuntimeError(f"fast response module_path for {op_name}/{cid} does not match the staged candidate")
            if act.get("sha256") != b["sha256"]:
                raise RuntimeError(f"fast response sha256 for {op_name}/{cid} does not match the staged candidate")
            if act.get("post_sha256") != b["sha256"]:
                raise RuntimeError(f"fast gate {FAST_STATUS_INFRASTRUCTURE_ERROR}: candidate {op_name}/{cid} changed during evaluation")


def _verify_fast_coverage(raw: Mapping[str, object], *, expected_candidates: int, expected_msprof_targets: int) -> None:
    cov = _fast_response_mapping(raw, "coverage")
    es, ev, te, to = cov.get("expected_candidates"), cov.get("evaluated_candidates"), cov.get("expected_msprof_targets"), cov.get("observed_msprof_targets")
    if es != expected_candidates or ev != expected_candidates:
        raise RuntimeError(f"fast response candidate coverage expected={es!r} evaluated={ev!r} request={expected_candidates}")
    if te != expected_msprof_targets or to != expected_msprof_targets:
        raise RuntimeError(f"fast response msprof coverage expected={te!r} observed={to!r} request={expected_msprof_targets}")


def _is_fast_transient_build_lock(execution: _V2GateExecution) -> bool:
    if execution.transport_error:
        low = execution.transport_error.lower()
        return "resource busy" in low and "build:" in low
    if execution.verified:
        return False
    row, _ = _gate_observation(execution.report, AGENT_GATES_DEVICE_FAST_GATE)
    if str((row or {}).get("recordedStatus") or (row or {}).get("status") or execution.report.get("status") or "").upper() != "BLOCKED_ENV":
        return False
    txt = f"{_gate_failure_text(row) if row is not None else ''} {execution.verification.get('failures') or ''}".lower()
    return "resource busy" in txt and "build:" in txt


def _is_fast_occupancy_blocked(execution: _V2GateExecution, blocked_cat: str, receipt_cat: str, eff_dev_fault: bool) -> tuple[bool, str]:
    if eff_dev_fault:
        return False, ""
    row, _ = _gate_observation(execution.report, AGENT_GATES_DEVICE_FAST_GATE)
    row_st, rep_st = _gate_recorded_status(row).upper(), str(execution.report.get("status") or "").upper()
    fail_txt, ver_fail, transport = (_gate_failure_text(row) if row is not None else ""), str(execution.verification.get("failures") or ""), str(execution.transport_error or "")
    if not (
        blocked_cat == "occupancy" or receipt_cat == "occupancy" or row_st == "BLOCKED_ENV" or rep_st == "BLOCKED_ENV"
        or "BLOCKED_EXIT_CODE: 75" in fail_txt or "BLOCKED_EXIT_CODE: 75" in ver_fail or "BLOCKED_EXIT_CODE: 75" in transport or "BLOCKED_ENV" in transport
    ):
        return False, ""
    return True, (blocked_cat or receipt_cat or fail_txt.strip() or ver_fail.strip() or transport.strip() or rep_st or "BLOCKED_ENV(75)")[:200]


def _fast_held_seconds(report: Mapping[str, object]) -> float:
    _row, obs = _gate_observation(report, AGENT_GATES_DEVICE_FAST_GATE)
    held = obs.get("deviceWindow", {}).get("heldSeconds") if isinstance(obs.get("deviceWindow"), Mapping) else None
    return 0.0 if isinstance(held, bool) or not isinstance(held, (int, float)) else float(held)


def _submit_fast_gate(
    *, agent_gates_bin: str, root: Path, workspace: str, request_id: str, request_relative_path: str,
    request_path: Path, response_path: Path, request_sha256: str, mode: str, timeout_sec: float,
    device_id: str, device_pool: Sequence[str] = (), env_vals: Mapping[str, str] | None = None,
) -> tuple[Mapping[str, object], _V2GateExecution]:
    """Submit the device-fast target with idle card selection and contamination retest."""
    if _sha256_file(request_path) != request_sha256:
        raise RuntimeError("fast request file changed after it was written")
    backoff, occ_backoff = (8.0, 12.0, 16.0, 20.0, 25.0, 30.0), (4.0, 6.0, 8.0, 10.0, 15.0)
    occ_budget, total_occ_sleep = min(max(float(timeout_sec), 10.0), 240.0), 0.0
    rejected_cards, selected_card, saw_df, occ_attempts, occ_reasons = set(), "", False, 0, []
    execution: _V2GateExecution | None = None
    for pass_idx in range(16):
        candidates = _call_order_candidates_by_idle_preflight(device_id, device_pool, excluded=rejected_cards, env_vals=env_vals)
        if not candidates:
            if not rejected_cards and str(device_id or "").strip():
                candidates = (str(device_id).strip(),)
            else:
                break
        contaminated = False
        for idx, card in enumerate(candidates):
            attempt_exec: _V2GateExecution | None = None
            for retry_idx in range(15):
                attempt_exec = _run_v2_gate(
                    agent_gates_bin=agent_gates_bin, root=root, workspace=workspace, gate_id=AGENT_GATES_DEVICE_FAST_GATE,
                    target=f"fast/{request_id}", candidate_path=request_relative_path, timeout_sec=timeout_sec, device_id=card,
                )
                if not _is_fast_transient_build_lock(attempt_exec) or retry_idx >= 14:
                    break
                ws = backoff[min(retry_idx, len(backoff) - 1)]
                logger.info("FAST_GATE_RETRY_BUILD_LOCK request=%s retry=%d/14 wait=%.1fs", request_id, retry_idx + 1, ws)
                time.sleep(ws)
            assert attempt_exec is not None
            execution = attempt_exec
            bcat = _blocked_env_category(attempt_exec.report, (AGENT_GATES_DEVICE_FAST_GATE,))
            is_df, is_dr, rcat = _inspect_v2_device_fault(attempt_exec.report, attempt_exec.verification)
            eff_df = bool(is_df or bcat == "device_fault" or rcat == "device_fault")
            eff_rej = bool(attempt_exec.device_rejected or is_dr or eff_df)
            is_occ, occ_reason = _is_fast_occupancy_blocked(attempt_exec, bcat, rcat, eff_df)
            if eff_rej or is_occ:
                response_path.unlink(missing_ok=True)
                if eff_rej:
                    rejected_cards.add(card)
                if eff_df:
                    saw_df = True
                if is_occ and not eff_rej:
                    occ_attempts += 1
                    if occ_reason and occ_reason not in occ_reasons:
                        occ_reasons.append(occ_reason)
                reason = "device_fault" if eff_df else ("device_rejected" if eff_rej else "occupancy")
                logger.warning("FAST_GATE_CONTAMINATION_RETEST request=%s tainted_device=%s reason=%s next_card=%s", request_id, card, reason, candidates[idx + 1] if idx + 1 < len(candidates) else "backoff")
                contaminated = True
                continue
            selected_card = card
            break
        else:
            if contaminated and pass_idx + 1 < 16 and total_occ_sleep < occ_budget and _normalize_device_pool(device_id, device_pool, excluded=rejected_cards):
                ws = occ_backoff[min(pass_idx, len(occ_backoff) - 1)]
                time.sleep(ws); total_occ_sleep += ws
                continue
        break
    assert execution is not None
    if not selected_card and occ_attempts > 0 and not saw_df:
        raise OccupancyExhaustedError(occ_attempts, tuple(occ_reasons))
    if not selected_card and (saw_df or rejected_cards):
        raise AgentGatesDeviceFaultError(
            f"AgentGates device-fast rejected all candidate NPU cards (rejected={sorted(rejected_cards)}, detail={execution.transport_error or 'device_fault'})",
            fault_plane="INFRA_DEVICE_FAULT", fault_code="NPU_DEVICE_TSD_FAULT", failed_tier="l2_npu",
            device_id=execution.device_id or str(device_id or "").strip(), rejected_cards=rejected_cards,
        )
    if execution.transport_error:
        raise RuntimeError(f"AgentGates device-fast batch failed: {execution.transport_error}")
    if not execution.verified:
        resp_detail = ""
        if response_path.is_file():
            try:
                fraw = json.loads(response_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                fraw = None
            if isinstance(fraw, Mapping) and (st := str(fraw.get("status", "")).strip()):
                resp_detail = f"; response status {st}: {str(fraw.get('error', '')).strip() or 'no detail'}"
        raise RuntimeError(f"AgentGates device-fast receipt did not verify: {execution.verification.get('failures') or 'unsatisfied'}{resp_detail}")
    if not _source_identity_matches(execution.verification, root):
        raise RuntimeError("AgentGates device-fast verified source identity does not match requested source")
    if not response_path.is_file():
        row, _ = _gate_observation(execution.report, AGENT_GATES_DEVICE_FAST_GATE)
        raise RuntimeError(f"AgentGates device-fast batch produced no response: {(row.get('failures') if row is not None else None) or execution.report.get('status') or 'missing response'}")
    raw = json.loads(response_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError("AgentGates device-fast response must be a JSON object")
    _raise_for_fast_status(_require_fast_envelope(raw, request_id=request_id, request_sha256=request_sha256, mode=mode), raw)
    probe_card = selected_card or execution.device_id or str(device_id or "").strip()
    load = _probe_host_card_load(probe_card, env_vals=env_vals) if probe_card else None
    if load is not None and load[0] < 3 and int(load[1]) > 0:
        raw.setdefault("_selected_card_ext_hbm_mb", int(load[1]))
    return raw, execution
