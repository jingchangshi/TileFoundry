"""Standalone zero-config score-sprint marathon runner and event watcher."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Callable, Mapping, Sequence
import ctypes
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import select
import shlex
import shutil
import subprocess
import sys
import tempfile
import time

from tilefoundry_ascend.agent.campaign.anchor_specs import (
    AnchorCaseSpec,
    CannbotOpsRegistry,
    PerfTargetSpec,
    RegimeTargetSpec,
    default_source_root,
    get_default_anchors,
    get_default_regime_specs,
    get_operator_module_relpath,
    get_operator_task,
    parse_simple_env_file,
    resolve_cannbench_repo,
    resolve_cannbench_tasks_root,
)
from tilefoundry_ascend.agent.campaign.campaign_runner import (
    CampaignConfig,
    PerfTargetCampaignRunner,
    patch_regime_into_module,
)
from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (
    CandidateSurrogateResult,
    L3AcceptanceResult,
    _clean_agent_gates_subprocess_env,
    _default_surrogate_cache_dir,
    _default_surrogate_tl_cache_dir,
    _resolve_agent_gates_bin,
    _resolve_env_file_path,
    _run_agent_gates_cmd,
    _write_temp_env_file,
    collect_bottleneck_msprof_via_agent_gates,
    ensure_npu_compare_shim_dir,
    prewarm_anchor_reference_cache,
    run_l4_official_cannbench_via_agent_gates,
    run_micro_probe_via_agent_gates,
    run_surrogate_batch_via_agent_gates,
)
from tilefoundry_ascend.agent.flywheel.ablation_distiller import FlywheelDistiller
from tilefoundry_ascend.agent.kb.bayesian_tracker import BayesianRuleTracker
from tilefoundry_ascend.agent.orchestrator.gate_funnel import GateFunnel
from tilefoundry_ascend.agent.orchestrator.llm_client import (
    AgyCliLLMBackend,
    GenerationRequest,
)

logger = logging.getLogger("tilefoundry_ascend.marathon")

_IN_CLOSE_WRITE = 0x00000008
_IN_MOVED_TO = 0x00000080
_WATCH_MASK = _IN_CLOSE_WRITE | _IN_MOVED_TO
_DEFAULT_AGY_COMMON_ARGS = (
    "--print --input-format text --disable-slash-commands "
    "--sandbox --mode plan --dangerously-skip-permissions --print-timeout 240s"
)
_SUPPORTED_OPERATORS: tuple[str, ...] = ()
_AGENT_GATES_ALLOWED_ENV_KEYS = frozenset(
    {
        "TILEFOUNDRY_NPU_CANN_SETUP",
        "TILEFOUNDRY_NPU_CONDA_SH",
        "TILEFOUNDRY_NPU_CONDA_ENV",
        "TILEFOUNDRY_NPU_CONDA_ENVS_PATH",
        "TILEFOUNDRY_NPU_PYTHON",
        "TILEFOUNDRY_NPU_TILELANG_SOURCE",
        "TILEFOUNDRY_NPU_TILELANG_BUILD",
        "TILEFOUNDRY_NPU_CPU_TESTS",
        "TILEFOUNDRY_NPU_DEVICE_TESTS",
        "TILEFOUNDRY_NPU_BUILD_SUBDIR",
        "TILEFOUNDRY_NPU_CANNBENCH_REPO",
        "TILEFOUNDRY_NPU_WORK_ROOT",
        "TILEFOUNDRY_NPU_CANNBENCH_ARCH",
        "TILEFOUNDRY_NPU_CANNBENCH_TARGETS_FILE",
        "TILEFOUNDRY_NPU_CANNBENCH_TARGETS",
        "TILEFOUNDRY_NPU_CANNBENCH_MCP_URL",
        "TILEFOUNDRY_NPU_CANNBENCH_MCP_TOKEN",
        "TILEFOUNDRY_NPU_CANNBENCH_MCP_TOKEN_FILE",
        "TILEFOUNDRY_NPU_CANNBENCH_BENCHMARK_VERSION",
        "TILEFOUNDRY_NPU_CANNBENCH_POOL_ID",
        "TILEFOUNDRY_NPU_CANNBENCH_AGGREGATION_TOKEN",
        "TILEFOUNDRY_NPU_CANNBENCH_INDEPENDENT_ENV_CONSENT",
        "TILEFOUNDRY_NPU_LLM_BACKEND",
        "TILEFOUNDRY_NPU_AGY_BIN",
        "TILEFOUNDRY_NPU_AGY_EXPLOIT_MODEL",
        "TILEFOUNDRY_NPU_AGY_EXPLORE_MODEL",
        "TILEFOUNDRY_NPU_AGY_COMMON_ARGS",
        "TILEFOUNDRY_NPU_AGY_MAX_CONCURRENCY",
        "TILEFOUNDRY_NPU_TILELANG_REPO",
        "TILEFOUNDRY_NPU_CANNBOT_DSL_REPO",
        "TILEFOUNDRY_NPU_OPS_TRANSFORMER_REPO",
        "TILEFOUNDRY_NPU_ASC_DEVKIT_REPO",
    }
)


def _is_agent_gates_allowed_env_key(key: str) -> bool:
    """Return True if *key* is accepted by the AgentGates tilefoundry-npu adapter."""
    return key in _AGENT_GATES_ALLOWED_ENV_KEYS or key.startswith("TILEFOUNDRY_NPU_CANNBENCH_")


@dataclass(frozen=True)
class MarathonConfig:
    """Configuration for a zero-config score-sprint marathon campaign."""

    operator_name: str
    device_id: str = "1"
    source_root: Path | None = None
    worktree: Path | None = None
    isolate_worktree: bool = False
    state_dir: Path | None = None
    target_module_path: Path | None = None
    env_file: Path | None = None
    checkpoint_path: Path | None = None
    report_path: Path | None = None
    campaign_kb_path: Path | None = None
    log_path: Path | None = None
    max_rounds: int = 80
    max_wall_clock_min: float = 720.0
    k_rule_workers: int = 2
    max_syntax_repairs: int = 1
    min_l2_gain_for_l3: float = 0.005
    min_surrogate_gain: float = 0.08
    min_flywheel_gain: float = 0.03
    stagnation_escalate_rounds: int = 1
    stagnation_freeze_rounds: int = 12
    stagnation_discount: float = 0.45
    target_avg_speedup: float = 2.985
    target_min_case_speedup: float = 2.95
    target_roofline_eta: float = 0.995
    target_surrogate_score: float = 99.8
    min_optimization_rounds: int = 15
    disable_early_goal_stop: bool = False
    run_final_l4: bool = True
    l4_regression_ratio: float = 1.0
    warmup: int = 3
    repeat: int = 5
    enable_tilesize_prescreen: bool = False
    prewarm_l3: bool = True
    salvage_beam_candidates: bool = True
    resume: bool | None = None
    escalate_model: str = "gemini-4-hydrogen-high"
    exploit_effort: str = "high"
    auto_bootstrap_seed: bool = True


def short_operator_tag(operator_name: str) -> str:
    """Return short filesystem tag for an operator name or task id."""
    norm = operator_name.rsplit("/", 1)[-1] if "/" in operator_name else operator_name
    if norm == "attention_residual":
        return "ar"
    return norm.replace("_", "-")


def atomic_write_text(path: Path, text: str) -> None:
    """Atomically write UTF-8 text via mkstemp, fsync, and replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def ensure_operator_seed_module(
    operator_name: str,
    target_module_path: Path,
    *,
    cannbench_repo: Path | None = None,
) -> bool:
    """Bootstrap a Stage-A-valid TileLang seed module + golden fallback for any CANNBench op."""
    target_path = Path(target_module_path).resolve()
    if target_path.is_file():
        return False

    ops_root: Path | None = None
    if cannbench_repo is not None:
        cand = Path(cannbench_repo).expanduser().resolve()
        try:
            ops_root = resolve_cannbench_tasks_root(cand)
        except FileNotFoundError:
            ops_root = cand
    try:
        registry = (
            CannbotOpsRegistry(operators_root=ops_root)
            if ops_root is not None
            else CannbotOpsRegistry.default()
        )
    except FileNotFoundError:
        return False
    if not registry.has_operator(operator_name):
        return False

    spec = registry.resolve(operator_name)
    op_id = spec.op_id
    entrypoint = spec.entrypoint
    golden_path = (spec.op_dir / "golden.py").resolve()
    default_dtype = "float16"
    for case in spec.cases:
        raw_dt = case.get("dtype")
        if isinstance(raw_dt, (list, tuple)) and raw_dt:
            cand_dt = str(raw_dt[0]).strip()
        elif isinstance(raw_dt, str):
            cand_dt = raw_dt.strip()
        else:
            cand_dt = ""
        if cand_dt in ("float16", "bfloat16", "float32"):
            default_dtype = cand_dt
            break

    target_path.parent.mkdir(parents=True, exist_ok=True)
    init_path = target_path.parent / "__init__.py"
    if not init_path.exists():
        init_src = (
            f'"""Auto-seeded Ascend 950PR operator package for {op_id}."""\n\n'
            f"from .{target_path.stem} import {entrypoint}, kernel\n\n"
            f'__all__ = [{entrypoint!r}, "kernel"]\n'
        )
        atomic_write_text(init_path, init_src)

    seed_src = (
        f'"""Auto-seeded TileLang Ascend950PR scaffold and golden fallback for {op_id}."""\n\n'
        "import contextlib\n"
        "import importlib.util\n"
        "import os\n"
        "from pathlib import Path\n\n"
        "import tilelang\n"
        "import tilelang.ascend.language as T\n"
        "import torch\n\n"
        "_KERNEL_CACHE: dict[tuple[object, ...], object] = {}\n"
        "_GOLDEN_FN = None\n"
        f"_GOLDEN_TASK = {spec.task!r}\n"
        f"_DEFAULT_GOLDEN_PATH = Path({str(golden_path)!r})\n\n\n"
        "def _ascend_target_ctx():\n"
        "    try:\n"
        "        from tilelang import tvm\n"
        "        from tilelang.backend.module import create_backend_context\n\n"
        "        if tvm.target.Target.current(allow_none=True) is None:\n"
        '            return create_backend_context("ascend").target\n'
        "    except Exception:\n"
        "        pass\n"
        "    return contextlib.nullcontext()\n\n\n"
        "def _load_golden_fn():\n"
        "    global _GOLDEN_FN\n"
        "    if _GOLDEN_FN is not None:\n"
        "        return _GOLDEN_FN\n"
        "    candidates = [_DEFAULT_GOLDEN_PATH]\n"
        '    env_repo = os.environ.get("TILEFOUNDRY_NPU_CANNBENCH_REPO", "").strip()\n'
        "    if env_repo:\n"
        '        candidates.insert(0, Path(env_repo).expanduser() / "operators" / _GOLDEN_TASK / "golden.py")\n'
        '        candidates.insert(0, Path(env_repo).expanduser() / "tasks" / _GOLDEN_TASK / "golden.py")\n'
        "    for cand in candidates:\n"
        "        if cand.is_file():\n"
        '            mod_name = "_tf_golden_" + _GOLDEN_TASK.replace("/", "_")\n'
        "            mod_spec = importlib.util.spec_from_file_location(mod_name, str(cand))\n"
        "            if mod_spec is not None and mod_spec.loader is not None:\n"
        "                mod = importlib.util.module_from_spec(mod_spec)\n"
        "                mod_spec.loader.exec_module(mod)\n"
        f"                fn = getattr(mod, {entrypoint!r}, None)\n"
        "                if callable(fn):\n"
        "                    _GOLDEN_FN = fn\n"
        "                    return fn\n"
        '    raise FileNotFoundError(f"Cannot locate golden.py for {_GOLDEN_TASK!r}")\n\n\n'
        f"def _select_{op_id}_config(**kwargs) -> dict[str, object]:\n"
        f'    """Select initial TileLang config for {op_id}."""\n'
        f'    raw_dt = str(kwargs.get("tl_dtype", {default_dtype!r})).strip()\n'
        '    tl_dtype = raw_dt if raw_dt in ("float16", "bfloat16", "float32") else "float16"\n'
        "    return {\n"
        '        "tile_m": 64,\n'
        '        "tile_n": 64,\n'
        '        "num_stages": 1,\n'
        '        "tl_dtype": tl_dtype,\n'
        '        "pass_configs": {},\n'
        '        "compile_flags": [],\n'
        "    }\n\n\n"
        f"def _build_{op_id}_prim_func(\n"
        "    tile_m: int = 64,\n"
        "    tile_n: int = 64,\n"
        "    num_stages: int = 1,\n"
        f"    tl_dtype: str = {default_dtype!r},\n"
        "    **_extra_kwargs,\n"
        "):\n"
        f'    """Build Stage-A-valid TileLang @T.prim_func scaffold for {op_id}."""\n'
        "    _ = num_stages\n"
        "    m_blk = max(16, min(64, int(tile_m)))\n"
        "    n_blk = max(16, min(64, int(tile_n)))\n"
        '    dt = tl_dtype if tl_dtype in ("float16", "bfloat16", "float32") else "float16"\n'
        "    with _ascend_target_ctx():\n\n"
        "        @T.prim_func\n"
        "        def main(\n"
        "            X: T.Tensor((m_blk, n_blk), dt),\n"
        "            Y: T.Tensor((m_blk, n_blk), dt),\n"
        "        ):\n"
        "            with T.Kernel(1):\n"
        "                x_ub = T.alloc_shared((m_blk, n_blk), dt)\n"
        "                T.copy(X[0:m_blk, 0:n_blk], x_ub)\n"
        "                T.copy(x_ub, Y[0:m_blk, 0:n_blk])\n\n"
        "    return main\n\n\n"
        "def _get_kernel(**kwargs):\n"
        f"    cfg = _select_{op_id}_config(**kwargs)\n"
        '    key = (int(cfg["tile_m"]), int(cfg["tile_n"]), str(cfg["tl_dtype"]))\n'
        "    compiled = _KERNEL_CACHE.get(key)\n"
        "    if compiled is None:\n"
        "        with _ascend_target_ctx():\n"
        f"            prog = _build_{op_id}_prim_func(\n"
        '                tile_m=int(cfg["tile_m"]),\n'
        '                tile_n=int(cfg["tile_n"]),\n'
        '                num_stages=int(cfg["num_stages"]),\n'
        '                tl_dtype=str(cfg["tl_dtype"]),\n'
        "            )\n"
        '        compiled = tilelang.compile(prog, target="ascend", out_idx=[1])\n'
        "        _KERNEL_CACHE[key] = compiled\n"
        "    return compiled\n\n\n"
        f"def {entrypoint}(*args, **kwargs):\n"
        f'    """Reference-backed entrypoint for {op_id} until promoted TileLang slices take over."""\n'
        "    return _load_golden_fn()(*args, **kwargs)\n\n\n"
        f"kernel = {entrypoint}\n\n"
        f"__all__ = [\n"
        f'    "_build_{op_id}_prim_func",\n'
        f'    "_select_{op_id}_config",\n'
        f'    {entrypoint!r},\n'
        '    "kernel",\n'
        "]\n"
    )
    atomic_write_text(target_path, seed_src)
    return True


def _ensure_sys_paths(worktree: Path) -> None:
    for entry in (
        str(worktree / "targets" / "ascend"),
        str(worktree / "src"),
        str(worktree),
    ):
        if entry not in sys.path:
            sys.path.insert(0, entry)


def _find_cann_setup_script(seeded: Mapping[str, str]) -> str:
    direct = seeded.get("TILEFOUNDRY_NPU_CANN_SETUP", "").strip()
    if direct and Path(direct).is_file():
        return str(Path(direct).resolve())
    for env_key in ("ASCEND_TOOLKIT_HOME", "ASCEND_HOME_PATH"):
        raw = os.environ.get(env_key, "").strip()
        if not raw:
            continue
        base = Path(raw).expanduser()
        for cand in (base / "set_env.sh", base.parent / "set_env.sh"):
            if cand.is_file():
                return str(cand.resolve())
    bisheng = shutil.which("bisheng")
    if bisheng:
        b_path = Path(bisheng).resolve()
        for parent in b_path.parents[:5]:
            cand = parent / "set_env.sh"
            if cand.is_file():
                return str(cand)
    return ""


def _infer_conda_settings(python_bin: str, seeded: Mapping[str, str]) -> dict[str, str]:
    out: dict[str, str] = {}
    py_path = Path(python_bin).expanduser()
    resolved = py_path.resolve() if py_path.exists() else py_path
    parts = resolved.parents
    if len(parts) >= 4 and parts[2].name == "envs":
        out["TILEFOUNDRY_NPU_CONDA_ENV"] = parts[1].name
        out["TILEFOUNDRY_NPU_CONDA_ENVS_PATH"] = str(parts[2])
        conda_sh = parts[3] / "etc" / "profile.d" / "conda.sh"
        if conda_sh.is_file():
            out["TILEFOUNDRY_NPU_CONDA_SH"] = str(conda_sh)

    conda_prefix = os.environ.get("CONDA_PREFIX", "").strip()
    if conda_prefix and "TILEFOUNDRY_NPU_CONDA_ENV" not in out:
        cp = Path(conda_prefix)
        out["TILEFOUNDRY_NPU_CONDA_ENV"] = cp.name
        if cp.parent.name == "envs":
            out["TILEFOUNDRY_NPU_CONDA_ENVS_PATH"] = str(cp.parent)
            cand_sh = cp.parent.parent / "etc" / "profile.d" / "conda.sh"
            if cand_sh.is_file():
                out["TILEFOUNDRY_NPU_CONDA_SH"] = str(cand_sh)

    conda_exe = os.environ.get("CONDA_EXE", "").strip()
    if conda_exe and "TILEFOUNDRY_NPU_CONDA_SH" not in out:
        ce = Path(conda_exe).resolve()
        if len(ce.parents) >= 2:
            cand_sh = ce.parents[1] / "etc" / "profile.d" / "conda.sh"
            if cand_sh.is_file():
                out["TILEFOUNDRY_NPU_CONDA_SH"] = str(cand_sh)

    for key in (
        "TILEFOUNDRY_NPU_CONDA_SH",
        "TILEFOUNDRY_NPU_CONDA_ENV",
        "TILEFOUNDRY_NPU_CONDA_ENVS_PATH",
    ):
        val = seeded.get(key, "").strip() or os.environ.get(key, "").strip()
        if val:
            out[key] = val
    return out


def discover_marathon_environment(
    *,
    operator_name: str,
    worktree: Path,
    state_dir: Path,
    base_env_file: str | Path | None = None,
) -> dict[str, str]:
    """Discover runtime toolchain and workspace paths with no hardcoded machine paths."""
    seeded: dict[str, str] = {}
    env_hint = base_env_file or os.environ.get("TILEFOUNDRY_NPU_ENV_FILE", "").strip()
    if env_hint and Path(env_hint).is_file():
        seeded.update(parse_simple_env_file(env_hint))

    for key, value in os.environ.items():
        if key.startswith("TILEFOUNDRY_NPU_") and value.strip() and key not in seeded:
            seeded[key] = value.strip()

    ws_parent = worktree.resolve().parent
    python_bin = (
        seeded.get("TILEFOUNDRY_NPU_PYTHON")
        or os.environ.get("TILEFOUNDRY_NPU_PYTHON", "").strip()
        or sys.executable
    )
    values: dict[str, str] = dict(seeded)
    values["TILEFOUNDRY_NPU_PYTHON"] = python_bin
    values.update(_infer_conda_settings(python_bin, seeded))

    cann_setup = _find_cann_setup_script(seeded)
    if cann_setup:
        values["TILEFOUNDRY_NPU_CANN_SETUP"] = cann_setup

    tl_dir = ws_parent / "tilelang"
    if "TILEFOUNDRY_NPU_TILELANG_SOURCE" not in values and tl_dir.is_dir():
        values["TILEFOUNDRY_NPU_TILELANG_SOURCE"] = str(tl_dir)
    if "TILEFOUNDRY_NPU_TILELANG_REPO" not in values and tl_dir.is_dir():
        values["TILEFOUNDRY_NPU_TILELANG_REPO"] = str(tl_dir)
    tl_src = Path(values["TILEFOUNDRY_NPU_TILELANG_SOURCE"]) if values.get("TILEFOUNDRY_NPU_TILELANG_SOURCE") else tl_dir
    if "TILEFOUNDRY_NPU_TILELANG_BUILD" not in values and (tl_src / "build").is_dir():
        values["TILEFOUNDRY_NPU_TILELANG_BUILD"] = str(tl_src / "build")

    if not values.get("TILEFOUNDRY_NPU_CANNBENCH_REPO"):
        for folder_name in ("cann-bench", "cannbot-ops"):
            cand = ws_parent / folder_name
            if cand.is_dir() and ((cand / "tasks").is_dir() or (cand / "operators").is_dir()):
                values["TILEFOUNDRY_NPU_CANNBENCH_REPO"] = str(cand)
                break

    sibling_keys = (
        ("TILEFOUNDRY_NPU_CANNBOT_DSL_REPO", "cannbot-dsl"),
        ("TILEFOUNDRY_NPU_OPS_TRANSFORMER_REPO", "ops-transformer"),
        ("TILEFOUNDRY_NPU_ASC_DEVKIT_REPO", "asc-devkit"),
    )
    for key, folder_name in sibling_keys:
        if not values.get(key):
            cand = ws_parent / folder_name
            if cand.is_dir():
                values[key] = str(cand)

    if not values.get("TILEFOUNDRY_NPU_AGENT_GATES_BIN"):
        ag_cand = ws_parent / "AgentGates" / ".venv" / "bin" / "agent-gates"
        if ag_cand.is_file():
            values["TILEFOUNDRY_NPU_AGENT_GATES_BIN"] = str(ag_cand)
        else:
            ag_which = shutil.which("agent-gates")
            if ag_which:
                values["TILEFOUNDRY_NPU_AGENT_GATES_BIN"] = ag_which

    if not values.get("TILEFOUNDRY_NPU_AGY_BIN"):
        agy_which = shutil.which("agy")
        local_agy = Path.home() / ".local" / "bin" / "agy"
        if agy_which:
            values["TILEFOUNDRY_NPU_AGY_BIN"] = agy_which
        elif local_agy.is_file():
            values["TILEFOUNDRY_NPU_AGY_BIN"] = str(local_agy)

    values.setdefault("TILEFOUNDRY_NPU_CPU_TESTS", "")
    values.setdefault("TILEFOUNDRY_NPU_DEVICE_TESTS", "")
    values.setdefault("TILEFOUNDRY_NPU_BUILD_SUBDIR", "build")
    values["TILEFOUNDRY_NPU_WORK_ROOT"] = str((state_dir / "cannbench-work").resolve())
    values.setdefault("TILEFOUNDRY_NPU_CANNBENCH_ARCH", "ascend950")
    values["TILEFOUNDRY_NPU_CANNBENCH_TARGETS"] = (
        f"{get_operator_task(operator_name)}:{get_operator_module_relpath(operator_name)}"
    )
    values.setdefault("TILEFOUNDRY_NPU_LLM_BACKEND", "agy_cli")
    values.setdefault("TILEFOUNDRY_NPU_AGY_EXPLOIT_MODEL", "gemini-4-hydrogen-medium")
    values.setdefault("TILEFOUNDRY_NPU_AGY_EXPLORE_MODEL", "gemini-4-hydrogen-high")
    values.setdefault("TILEFOUNDRY_NPU_AGY_COMMON_ARGS", _DEFAULT_AGY_COMMON_ARGS)
    values.setdefault("TILEFOUNDRY_NPU_AGY_MAX_CONCURRENCY", "2")
    return values


def _format_env_line(key: str, value: str) -> str:
    if any(ch.isspace() for ch in value):
        escaped = value.replace('"', '\\"')
        return f'{key}="{escaped}"'
    return f"{key}={value}"


def ensure_marathon_env_file(
    *,
    operator_name: str,
    worktree: Path,
    state_dir: Path,
    env_file: str | Path | None = None,
) -> Path:
    """Reuse an existing env.sh or materialize a discovered zero-config env.sh."""
    target = Path(env_file).expanduser() if env_file is not None else (state_dir / "env.sh")
    target = target.resolve()
    discovered = discover_marathon_environment(
        operator_name=operator_name,
        worktree=worktree,
        state_dir=state_dir,
        base_env_file=target if target.is_file() else None,
    )
    allowed_discovered = {
        key: val for key, val in discovered.items() if _is_agent_gates_allowed_env_key(key)
    }
    if target.is_file():
        existing = parse_simple_env_file(target)
        disallowed_existing = [
            key for key in existing if not _is_agent_gates_allowed_env_key(key)
        ]
        missing = [
            key
            for key, val in allowed_discovered.items()
            if val and not existing.get(key)
        ]
        if not missing and not disallowed_existing:
            return target
        merged = dict(allowed_discovered)
        merged.update(
            {
                key: val
                for key, val in existing.items()
                if val and _is_agent_gates_allowed_env_key(key)
            }
        )
    else:
        merged = allowed_discovered

    lines = [
        f"# Isolated marathon environment for {operator_name}",
        "# Plain KEY=VALUE assignments; safe for env parsers and shell sourcing.",
        "",
    ]
    for key in sorted(merged):
        lines.append(_format_env_line(key, merged[key]))
    lines.append("")
    atomic_write_text(target, "\n".join(lines))
    return target


def configure_marathon_process_env(
    *,
    operator_name: str,
    env_file: Path,
    worktree: Path | None = None,
) -> dict[str, str]:
    """Configure coordinator process env vars cleanly for isolated worker subprocesses."""
    short_op = short_operator_tag(operator_name)
    tmp_dir = Path(tempfile.gettempdir()) / f"tf_marathon_{short_op}"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = str(tmp_dir)
    os.environ["TEMP"] = str(tmp_dir)
    os.environ["TMP"] = str(tmp_dir)
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    os.environ["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    os.environ["USE_CUDA"] = "0"
    os.environ.pop("ASCEND_RT_VISIBLE_DEVICES", None)
    os.environ["TILEFOUNDRY_NPU_ENV_FILE"] = str(env_file)

    parsed = parse_simple_env_file(env_file)
    for key in (
        "TILEFOUNDRY_NPU_AGENT_GATES_BIN",
        "TILEFOUNDRY_NPU_LLM_BACKEND",
        "TILEFOUNDRY_NPU_AGY_BIN",
        "TILEFOUNDRY_NPU_AGY_EXPLOIT_MODEL",
        "TILEFOUNDRY_NPU_AGY_EXPLORE_MODEL",
        "TILEFOUNDRY_NPU_AGY_COMMON_ARGS",
        "TILEFOUNDRY_NPU_AGY_MAX_CONCURRENCY",
        "TILEFOUNDRY_NPU_CANNBENCH_REPO",
        "TILEFOUNDRY_NPU_TILELANG_SOURCE",
        "TILEFOUNDRY_NPU_TILELANG_BUILD",
    ):
        val = parsed.get(key, "").strip()
        if val and not os.environ.get(key):
            os.environ[key] = val

    if not os.environ.get("TILEFOUNDRY_NPU_AGENT_GATES_BIN", "").strip():
        candidates: list[Path] = []
        if worktree is not None:
            candidates.append(worktree.parent / "AgentGates" / ".venv" / "bin" / "agent-gates")
        else:
            src_root = Path(parsed.get("TILEFOUNDRY_NPU_CANNBENCH_REPO", "")).parent
            if str(src_root):
                candidates.append(src_root / "AgentGates" / ".venv" / "bin" / "agent-gates")
        for cand in candidates:
            if cand.is_file():
                os.environ["TILEFOUNDRY_NPU_AGENT_GATES_BIN"] = str(cand)
                break
        else:
            ag_which = shutil.which("agent-gates")
            if ag_which:
                os.environ["TILEFOUNDRY_NPU_AGENT_GATES_BIN"] = ag_which
    return parsed


def _resolve_worktree(cfg: MarathonConfig) -> tuple[Path, Path]:
    source_root = (
        Path(cfg.source_root).expanduser().resolve()
        if cfg.source_root is not None
        else default_source_root()
    )
    if cfg.worktree is not None:
        return source_root, Path(cfg.worktree).expanduser().resolve()
    if not cfg.isolate_worktree:
        return source_root, source_root

    short_op = short_operator_tag(cfg.operator_name)
    cand = source_root.parent / f"tilefoundry-marathon-{short_op}"
    if cand.is_dir() and (cand / ".git").exists():
        return source_root, cand.resolve()
    subprocess.run(
        ["git", "-C", str(source_root), "worktree", "add", "--detach", str(cand), "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    )
    return source_root, cand.resolve()


class FastCreativeAgyBackend(AgyCliLLMBackend):
    """Route escalated creative requests to stable high-tier exploration model."""

    def __init__(self, escalate_model: str = "gemini-4-hydrogen-high") -> None:
        super().__init__(escalate_model=escalate_model)

    def select_effort(
        self,
        req: GenerationRequest,
        effort_override: str | None = None,
    ) -> str:
        """Clamp unsupported xhigh effort requests down to high."""
        effort = super().select_effort(req, effort_override=effort_override)
        if effort.strip().lower() == "xhigh":
            return "high"
        return effort


class ExtremeCampaignRunner(PerfTargetCampaignRunner):
    """Runner with stagnation-discounted regime rotation and per-round telemetry."""

    def __init__(
        self,
        *args,
        stagnation_discount: float = 0.45,
        op_logger: logging.Logger | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._live_stag_by_regime: dict[str, int] = {}
        self._stagnation_discount = max(0.0, float(stagnation_discount))
        self._op_logger = op_logger or logger

    def _restore_from_checkpoint(self):  # type: ignore[override]
        restored = super()._restore_from_checkpoint()
        self._live_stag_by_regime = dict(restored[12])
        return restored

    def _select_active_regime(
        self,
        current_regime: str | None,
        frozen_regimes: set[str],
        best_result: CandidateSurrogateResult,
        active_anchors: Sequence[AnchorCaseSpec],
        *,
        allow_satisfied_fallback: bool = False,
        exclude_regime: str | None = None,
    ) -> RegimeTargetSpec | None:
        by_tag = {r.regime_tag: r for r in self._regime_specs}
        if current_regime and current_regime not in frozen_regimes and current_regime in by_tag:
            spec = by_tag[current_regime]
            if not self._is_regime_satisfied(spec, best_result, active_anchors):
                return spec

        candidates = [
            spec
            for spec in self._regime_specs
            if spec.regime_tag not in frozen_regimes
            and not self._is_regime_satisfied(spec, best_result, active_anchors)
        ]
        if not candidates and allow_satisfied_fallback:
            if (
                current_regime
                and current_regime not in frozen_regimes
                and current_regime in by_tag
            ):
                return by_tag[current_regime]
            candidates = [
                spec for spec in self._regime_specs if spec.regime_tag not in frozen_regimes
            ]
        if not candidates:
            return None
        if exclude_regime and any(spec.regime_tag != exclude_regime for spec in candidates):
            candidates = [spec for spec in candidates if spec.regime_tag != exclude_regime]
        candidates.sort(
            key=lambda s: best_result.regime_headroom(s.regime_tag)
            / (1.0 + self._stagnation_discount * self._live_stag_by_regime.get(s.regime_tag, 0)),
            reverse=True,
        )
        return candidates[0]

    def _save_checkpoint(self, **kwargs) -> None:  # type: ignore[override]
        self._live_stag_by_regime = {
            str(k): int(v) for k, v in (kwargs.get("stagnation_by_regime") or {}).items()
        }
        super()._save_checkpoint(**kwargs)
        rounds = kwargs.get("rounds") or ()
        best_result = kwargs.get("best_result")
        frozen_regimes = sorted(str(x) for x in (kwargs.get("frozen_regimes") or ()))
        stop_reason = str(kwargs.get("stop_reason", ""))
        op_name = self._config.operator_name
        if rounds:
            last = rounds[-1]
            self._op_logger.info(
                "ROUND_TELEMETRY op=%s round=%d regime=%s promoted=%s "
                "best_score=%.6f best_speedup=%.6f stag=%s frozen=%s stop=%s",
                op_name,
                last.round_idx,
                last.regime_tag,
                last.promoted,
                best_result.surrogate_score if best_result else 0.0,
                best_result.avg_speedup if best_result else 0.0,
                self._live_stag_by_regime,
                frozen_regimes,
                stop_reason,
            )
        else:
            self._op_logger.info(
                "BASELINE_TELEMETRY op=%s best_score=%.6f best_speedup=%.6f stop=%s",
                op_name,
                best_result.surrogate_score if best_result else 0.0,
                best_result.avg_speedup if best_result else 0.0,
                stop_reason,
            )


def _prewarm_l3_candidate_kernels(
    *,
    operator_name: str,
    candidate_module_path: Path,
    worktree: Path,
    env_file: Path,
    tl_cache_dir: Path,
    op_logger: logging.Logger,
) -> None:
    resolved_env = _resolve_env_file_path(env_file)
    env_vals = parse_simple_env_file(resolved_env)
    python_bin = (
        env_vals.get("TILEFOUNDRY_NPU_PYTHON")
        or os.environ.get("TILEFOUNDRY_NPU_PYTHON")
        or sys.executable
    )
    cann_setup = (env_vals.get("TILEFOUNDRY_NPU_CANN_SETUP") or "").strip()
    tl_source = (env_vals.get("TILEFOUNDRY_NPU_TILELANG_SOURCE") or "").strip()

    tl_cache_dir.mkdir(parents=True, exist_ok=True)
    run_env = dict(os.environ)
    run_env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    run_env["ASCEND_RT_VISIBLE_DEVICES"] = ""
    run_env["USE_CUDA"] = "0"
    run_env["TILELANG_CACHE_DIR"] = str(tl_cache_dir)
    run_env["TILELANG_KERNEL_CACHE_USE_LIB_STAMP"] = "1"
    run_env["TILELANG_DISABLE_CACHE"] = "0"
    run_env["PYTHONDONTWRITEBYTECODE"] = "1"
    pp_parts = [
        str(worktree),
        str(worktree / "src"),
        str(worktree / "targets" / "ascend"),
    ]
    if tl_source:
        pp_parts.insert(0, tl_source)
    run_env["PYTHONPATH"] = os.pathsep.join(pp_parts)

    task_name = get_operator_task(operator_name)
    short_entry = task_name.rsplit("/", 1)[-1]
    cann_repo_hint = (env_vals.get("TILEFOUNDRY_NPU_CANNBENCH_REPO") or "").strip()

    prewarm_script = f"""
import importlib, importlib.util, json, os, sys
from pathlib import Path
import torch

os.environ["TILELANG_CACHE_DIR"] = {str(tl_cache_dir)!r}
os.environ["TILELANG_KERNEL_CACHE_USE_LIB_STAMP"] = "1"
os.environ["TILELANG_DISABLE_CACHE"] = "0"

tl_env_mod = importlib.import_module("tilelang.env")
tl_kc = importlib.import_module("tilelang.cache.kernel_cache")
tl_env_mod.env.enable_cache()
tl_kc.KernelCache._get_cache_namespace.cache_clear()
tl_kc.KernelCache._get_tilelang_lib_stamp.cache_clear()
tl_kc.KernelCache._create_dirs()

worktree = Path({str(worktree)!r})
bench_dir = worktree / "tests" / "ascend" / "benchmarks"
if str(bench_dir) not in sys.path:
    sys.path.insert(0, str(bench_dir))
import cannbench_contract

op_name = {short_entry!r}
task_name = {task_name!r}
cann_repo_hint = {cann_repo_hint!r}
mod_path = Path({str(candidate_module_path.resolve())!r})
spec = importlib.util.spec_from_file_location("_tf_l3_prewarm_cand", str(mod_path))
mod = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = mod
spec.loader.exec_module(mod)

orig_get_kernel = getattr(mod, "_get_kernel", None)
orig_combine = getattr(mod, "_get_combine_kernel", None)

if callable(orig_get_kernel):
    def _stub_get_kernel(*args, **kwargs):
        _ = orig_get_kernel(*args, **kwargs)
        splits = int(kwargs.get("num_kv_splits", 1) or 1)
        if op_name == "attention_residual":
            return lambda v_f, x_f, *rest: torch.empty_like(x_f)
        if splits <= 1:
            return lambda *a, **kw: torch.empty_like(a[0]) if a else None
        return lambda *rest: None

    mod._get_kernel = _stub_get_kernel

if callable(orig_combine):
    def _stub_combine(*args, **kwargs):
        _ = orig_combine(*args, **kwargs)
        return lambda o_p, l_p, *rest: torch.empty(
            (o_p.shape[1], o_p.shape[2]), dtype=torch.bfloat16, device=o_p.device
        )
    mod._get_combine_kernel = _stub_combine

def _empty_tensors(shapes, dtypes):
    return [
        torch.empty(tuple(sh), dtype=cannbench_contract.str_to_torch_dtype(dt), device="cpu")
        for sh, dt in zip(shapes, dtypes)
        if sh is not None and dt is not None and str(dt).lower() != "none"
    ]

task_dir = bench_dir / op_name
if not task_dir.is_dir() and cann_repo_hint:
    for _subdir in ("tasks", "operators"):
        cand_dir = Path(cann_repo_hint) / _subdir / task_name
        if cand_dir.is_dir():
            task_dir = cand_dir
            break
harness = cannbench_contract.Harness(task_name, task_dir)
entrypoint = getattr(mod, op_name, getattr(mod, "kernel", None))
compiled = 0
warnings = 0

def _try_call(tag, fn, params):
    global compiled, warnings
    if not callable(fn):
        return
    try:
        fn(**params)
        compiled += 1
    except Exception as exc:
        warnings += 1
        print(f"PREWARM_WARN {{tag}}: {{type(exc).__name__}}: {{exc}}", file=sys.stderr)

for case in harness.cases:
    raw = _empty_tensors(case.input_shapes, case.dtypes)
    if harness.get_input:
        in_p = {{
            item.name: raw[i] if i < len(raw) else None
            for i, item in enumerate(harness.info.inputs)
        }}
        in_p.update({{k: v for k, v in case.attrs.items() if k not in in_p}})
        raw = harness.get_input(**in_p)
    params = harness.params(case, raw)
    _try_call(f"case_{{case.case_num}}_tuned", entrypoint, params)
    if op_name == "attention_residual" and case.case_num in (1, 16, 20):
        p_base = dict(params)
        p_base["schedule_mode"] = "baseline"
        _try_call(f"case_{{case.case_num}}_baseline", entrypoint, p_base)

if op_name == "mla":
    supp_path = bench_dir / "cases" / "supplemental_mla.json"
    if supp_path.is_file():
        supp = json.loads(supp_path.read_text(encoding="utf-8"))
        for row in supp.get("cases", ()):
            raw = _empty_tensors(row["input_shapes"], row["dtypes"])
            case = cannbench_contract.Case(
                case_num=row["case_id"],
                label=row["key"],
                input_shapes=row["input_shapes"],
                dtypes=row["dtypes"],
                attrs=row["attrs"],
                value_ranges=row["value_range"],
            )
            in_p = {{
                item.name: raw[i] if i < len(raw) else None
                for i, item in enumerate(harness.info.inputs)
            }}
            in_p.update({{k: v for k, v in case.attrs.items() if k not in in_p}})
            raw = harness.get_input(**in_p)
            params = harness.params(case, raw)
            for mode in ("tuned", "baseline"):
                pm = dict(params)
                pm["schedule_mode"] = mode
                _try_call(f"supp_{{row['key']}}_{{mode}}", entrypoint, pm)
    _try_call(
        "mla_dynamic_shape",
        entrypoint,
        {{
            "q_nope": torch.empty((2, 64, 16, 512), dtype=torch.bfloat16),
            "q_rope": torch.empty((2, 64, 16, 64), dtype=torch.bfloat16),
            "k_nope": torch.empty((2, 128, 1, 512), dtype=torch.bfloat16),
            "k_rope": torch.empty((2, 128, 1, 64), dtype=torch.bfloat16),
            "v": torch.empty((2, 128, 1, 512), dtype=torch.bfloat16),
            "numKVHeads": 1,
            "scaleValue": -1.0,
            "inputLayout": "BSND",
            "is_causal": True,
        }},
    )
elif op_name == "attention_residual":
    _try_call(
        "ar_dynamic_shape",
        entrypoint,
        {{
            "v_stack": torch.empty((4, 1, 16, 256), dtype=torch.bfloat16),
            "x": torch.empty((1, 16, 256), dtype=torch.bfloat16),
            "w": torch.empty((256,), dtype=torch.bfloat16),
            "gamma": torch.empty((256,), dtype=torch.bfloat16),
            "alpha": 0.5,
            "epsilon": 1e-6,
        }},
    )

print(json.dumps({{"compiled": compiled, "warnings": warnings}}))
"""
    with tempfile.TemporaryDirectory(prefix="tf_l3_prewarm_") as tmp_dir:
        driver_path = Path(tmp_dir) / "prewarm_l3_driver.py"
        driver_path.write_text(prewarm_script, encoding="utf-8")
        if cann_setup and Path(cann_setup).is_file() and shutil.which("bisheng") is None:
            bash_cmd = (
                f"set +u && source {shlex.quote(cann_setup)} >/dev/null 2>&1 "
                f"&& exec {shlex.quote(python_bin)} {shlex.quote(str(driver_path))}"
            )
            argv = ["/bin/bash", "-c", bash_cmd]
        else:
            argv = [python_bin, str(driver_path)]
        proc = subprocess.run(
            argv,
            text=True,
            capture_output=True,
            cwd=str(worktree),
            env=run_env,
            timeout=900,
        )
    if proc.returncode != 0:
        op_logger.warning(
            "L3 CPU prewarm exited rc=%d: %s",
            proc.returncode,
            (proc.stderr or proc.stdout or "")[-600:],
        )
    else:
        last_line = (proc.stdout or "").strip().splitlines()[-1:] or ["{}"]
        op_logger.info("L3_CPU_PREWARM_OK op=%s summary=%s", operator_name, last_line[0])


def build_prewarmed_l3_runner(
    *,
    operator_name: str,
    worktree: Path,
    env_file: Path,
    device_id: str,
    prewarm_l3: bool = True,
    op_logger: logging.Logger | None = None,
) -> Callable[[Path], L3AcceptanceResult]:
    """Build an L3 contract verifier with one-time reference and CPU kernel prewarming."""
    log = op_logger or logger
    ref_prewarmed = False

    def _ensure_ref_cache() -> None:
        nonlocal ref_prewarmed
        if ref_prewarmed:
            return
        resolved_env = _resolve_env_file_path(env_file)
        env_vals = parse_simple_env_file(resolved_env)
        cann_repo = resolve_cannbench_repo(
            cannbench_repo=env_vals.get("TILEFOUNDRY_NPU_CANNBENCH_REPO"),
            env_file=resolved_env,
        )
        cache_dir = _default_surrogate_cache_dir()
        task = get_operator_task(operator_name)
        all_specs = [AnchorCaseSpec(task, i, "", 1.0, 1.0) for i in range(1, 21)]
        for start in range(0, len(all_specs), 5):
            chunk = all_specs[start : start + 5]
            prewarm_anchor_reference_cache(
                operator_name,
                chunk,
                cannbench_repo=cann_repo,
                env_file=resolved_env,
                cache_dir=cache_dir,
                source_root=worktree,
            )
        ref_prewarmed = True
        log.info("L3_REF_CACHE_READY op=%s cases=1..20 cache=%s", operator_name, cache_dir)

    shim_cache_hook = (
        "\nimport importlib as _tf_importlib\n"
        "import os as _tf_os\n"
        "_tf_tl_dir = _tf_os.environ.get('TF_SURROGATE_TL_CACHE_DIR', '').strip()\n"
        "if _tf_tl_dir:\n"
        "    _tf_os.makedirs(_tf_tl_dir, exist_ok=True)\n"
        "    _tf_os.environ['TILELANG_CACHE_DIR'] = _tf_tl_dir\n"
        "    _tf_os.environ['TILELANG_KERNEL_CACHE_USE_LIB_STAMP'] = '1'\n"
        "    _tf_os.environ['TILELANG_DISABLE_CACHE'] = '0'\n"
        "    try:\n"
        "        _tl_env_mod = _tf_importlib.import_module('tilelang.env')\n"
        "        _tl_kc = _tf_importlib.import_module('tilelang.cache.kernel_cache')\n"
        "        _tl_env_mod.env.enable_cache()\n"
        "        _tl_kc.KernelCache._get_cache_namespace.cache_clear()\n"
        "        _tl_kc.KernelCache._get_tilelang_lib_stamp.cache_clear()\n"
        "        _tl_kc.KernelCache._create_dirs()\n"
        "    except Exception:\n"
        "        pass\n"
    )

    def _run_l3(module_path: Path) -> L3AcceptanceResult:
        t0 = time.monotonic()
        resolved_env = _resolve_env_file_path(env_file)
        ag_bin = _resolve_agent_gates_bin(worktree)
        env_vals = parse_simple_env_file(resolved_env)
        cann_repo = resolve_cannbench_repo(
            cannbench_repo=env_vals.get("TILEFOUNDRY_NPU_CANNBENCH_REPO"),
            env_file=resolved_env,
        )
        cache_dir = _default_surrogate_cache_dir()
        tl_cache_dir = _default_surrogate_tl_cache_dir()

        if prewarm_l3:
            _ensure_ref_cache()
            _prewarm_l3_candidate_kernels(
                operator_name=operator_name,
                candidate_module_path=module_path,
                worktree=worktree,
                env_file=resolved_env,
                tl_cache_dir=tl_cache_dir,
                op_logger=log,
            )

        with tempfile.TemporaryDirectory(prefix="tf_l3_gate_") as tmp_dir:
            tmp = Path(tmp_dir)
            shim_dir = ensure_npu_compare_shim_dir(tmp / "shim", cannbench_repo=cann_repo)
            shim_py = shim_dir / "npu_compare.py"
            shim_py.write_text(
                shim_py.read_text(encoding="utf-8") + shim_cache_hook,
                encoding="utf-8",
            )
            tmp_env = _write_temp_env_file(
                resolved_env,
                tmp / "env.sh",
                {
                    "TILEFOUNDRY_NPU_CPU_TESTS": "tests/ascend/agent/test_llm_watchdog.py",
                    "TILEFOUNDRY_NPU_DEVICE_TESTS": (
                        "tests/ascend/agent/test_surrogate_batch_npu.py "
                        "tests/ascend/benchmarks/test_npu_acceptance.py"
                    ),
                },
            )
            cmd = [
                ag_bin,
                "run",
                "--adapter",
                "tilefoundry-npu",
                "--source",
                str(worktree),
                "--env-file",
                str(tmp_env),
                "--device-id",
                str(device_id),
                "--gate",
                "tilefoundry-npu-device",
                "--json",
            ]
            l3_env = {
                "CANNBENCH_CACHE_DIR": str(cache_dir),
                "TF_SURROGATE_TL_CACHE_DIR": str(tl_cache_dir),
                "TF_SURROGATE_SHIM_ONLY": "1",
                "TF_SURROGATE_SHIM_DIR": str(shim_dir),
                "TF_L3_TARGET_OPERATOR": operator_name,
                "TF_L3_TARGET_MODULE_PATH": str(Path(module_path).resolve()),
            }
            sub_env = _clean_agent_gates_subprocess_env(l3_env)
            proc, parsed = _run_agent_gates_cmd(
                cmd,
                cwd=str(worktree),
                env=sub_env,
                timeout_sec=900,
            )
        ok = proc.returncode == 0 and parsed.get("status") == "PASS"
        err = "" if ok else (proc.stderr or proc.stdout or "L3 contract failed")[-1000:]
        log.info(
            "L3_CONTRACT_DONE op=%s passed=%s elapsed=%.1fs run_id=%s",
            operator_name,
            ok,
            time.monotonic() - t0,
            parsed.get("runId", ""),
        )
        return L3AcceptanceResult(
            passed=ok,
            run_id=str(parsed.get("runId", "")),
            receipt_path=str(parsed.get("receiptPath", "")),
            elapsed_sec=time.monotonic() - t0,
            error_message=err,
        )

    return _run_l3


def build_resilient_l2_evaluator(
    *,
    operator_name: str,
    worktree: Path,
    env_file: Path,
    device_id: str,
    warmup: int = 3,
    repeat: int = 5,
    batch_fn: Callable[..., tuple[dict[str, CandidateSurrogateResult], str, str]] | None = None,
    op_logger: logging.Logger | None = None,
) -> Callable[[Sequence[tuple[str, Path]], Sequence[AnchorCaseSpec]], dict[str, CandidateSurrogateResult]]:
    """Build batched L2 evaluator with single-candidate isolation fallback on failures."""
    log = op_logger or logger
    eval_fn = batch_fn or run_surrogate_batch_via_agent_gates

    def _eval(
        cands: Sequence[tuple[str, Path]],
        anc_seq: Sequence[AnchorCaseSpec],
    ) -> dict[str, CandidateSurrogateResult]:
        cand_list = list(cands)
        try:
            res, _rid, _rcp = eval_fn(
                operator_name,
                cand_list,
                anc_seq,
                source_root=worktree,
                env_file=env_file,
                device_id=device_id,
                warmup=warmup,
                repeat=repeat,
                require_all_passed=False,
                timeout_sec=660,
            )
            return res
        except Exception as exc:
            if len(cand_list) <= 1:
                raise
            log.warning(
                "Batched L2 failed (%s: %s); falling back to single-candidate evaluation for %s",
                type(exc).__name__,
                exc,
                [cid for cid, _ in cand_list],
            )
            merged: dict[str, CandidateSurrogateResult] = {}
            last_exc: Exception | None = None
            for cid, mod_path in cand_list:
                try:
                    one_res, _rid, _rcp = eval_fn(
                        operator_name,
                        [(cid, mod_path)],
                        anc_seq,
                        source_root=worktree,
                        env_file=env_file,
                        device_id=device_id,
                        warmup=warmup,
                        repeat=repeat,
                        require_all_passed=False,
                        timeout_sec=360,
                    )
                    merged.update(one_res)
                except Exception as sub_exc:
                    last_exc = sub_exc
                    log.warning(
                        "Single-candidate L2 fallback failed for %s: %s: %s",
                        cid,
                        type(sub_exc).__name__,
                        sub_exc,
                    )
            if merged:
                return merged
            assert last_exc is not None
            raise last_exc

    return _eval


def salvage_unpromoted_beam_winner(
    *,
    operator_name: str,
    checkpoint_path: Path,
    target_module_path: Path,
    worktree: Path,
    env_file: Path,
    device_id: str,
    min_surrogate_gain: float,
    l2_evaluator: Callable[
        [Sequence[tuple[str, Path]], Sequence[AnchorCaseSpec]],
        dict[str, CandidateSurrogateResult],
    ],
    l3_runner: Callable[[Path], L3AcceptanceResult],
    op_logger: logging.Logger | None = None,
) -> bool:
    """Re-verify and promote high-scoring unpromoted beam entries across restarts."""
    if not checkpoint_path.is_file():
        return False
    log = op_logger or logger
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    best_dict = payload.get("best_result") or {}
    best_res = CandidateSurrogateResult.from_dict(best_dict)
    if not payload.get("incumbent_source"):
        return False
    incumbent_src = str(payload["incumbent_source"])
    promoted_set = {int(x) for x in (payload.get("promoted_rounds") or ())}
    tried_cache_path = checkpoint_path.parent / ".salvage_tried.json"
    tried_keys: set[str] = set()
    if tried_cache_path.is_file():
        try:
            tried_keys = set(json.loads(tried_cache_path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            tried_keys = set()

    beam_states = payload.get("beam_states") or {}
    specs_by_tag = {s.regime_tag: s for s in get_default_regime_specs(operator_name)}
    candidates: list[tuple[float, str, dict[str, object], RegimeTargetSpec]] = []
    for rtag, bmap in beam_states.items():
        spec = specs_by_tag.get(str(rtag))
        if spec is None or not isinstance(bmap, Mapping):
            continue
        for row in bmap.get("exploit_beam", ()):
            if not isinstance(row, Mapping):
                continue
            r_idx = int(row.get("round_idx", -1))
            if r_idx in promoted_set or not row.get("dsl_source"):
                continue
            if not row.get("accuracy_passed", False):
                continue
            obj = float(row.get("objective_score", 0.0) or 0.0)
            candidates.append((obj, str(rtag), dict(row), spec))

    if not candidates:
        return False
    candidates.sort(key=lambda item: item[0], reverse=True)

    active_anchors = [
        AnchorCaseSpec.from_dict(item) for item in (payload.get("active_anchors") or ())
    ] or list(get_default_anchors(operator_name))

    for _obj, rtag, row, regime_spec in candidates[:2]:
        cid = str(row.get("candidate_id", ""))
        r_idx = int(row.get("round_idx", -1))
        try:
            patched_src = patch_regime_into_module(
                incumbent_src,
                regime_spec,
                str(row["dsl_source"]),
            )
        except Exception as exc:
            log.warning("SALVAGE_SKIPPED %s patch failed: %s", cid, exc)
            continue

        sha = hashlib.sha256(patched_src.encode("utf-8")).hexdigest()
        cache_key = f"{cid}:{sha}"
        if cache_key in tried_keys:
            continue

        with tempfile.TemporaryDirectory(prefix="tf_marathon_salvage_") as tmp_dir:
            cand_file = Path(tmp_dir) / f"{cid or 'cand'}.py"
            cand_file.write_text(patched_src, encoding="utf-8")
            try:
                res_map = l2_evaluator([(cid, cand_file)], active_anchors)
            except Exception as exc:
                log.warning("SALVAGE_SKIPPED %s L2 eval failed: %s", cid, exc)
                continue
        cand_res = res_map.get(cid)
        if (
            cand_res is None
            or not cand_res.all_passed
            or cand_res.surrogate_score <= best_res.surrogate_score + min_surrogate_gain
        ):
            tried_keys.add(cache_key)
            atomic_write_text(tried_cache_path, json.dumps(sorted(tried_keys)) + "\n")
            continue

        atomic_write_text(target_module_path, patched_src)
        l3_res = l3_runner(target_module_path)
        if not l3_res.passed:
            atomic_write_text(target_module_path, incumbent_src)
            tried_keys.add(cache_key)
            atomic_write_text(tried_cache_path, json.dumps(sorted(tried_keys)) + "\n")
            log.warning(
                "SALVAGE_SKIPPED %s L3 failed: %s",
                cid,
                l3_res.error_message[-300:],
            )
            continue

        try:
            msprof_map = collect_bottleneck_msprof_via_agent_gates(
                operator_name,
                target_module_path,
                [a for a in active_anchors if a.regime_tag == rtag],
                source_root=worktree,
                env_file=env_file,
                device_id=device_id,
                timeout_sec=600,
            )
        except Exception as exc:
            log.warning("SALVAGE msprof refresh failed (non-fatal): %s", exc)
            msprof_map = {}

        msprof_cache = dict(payload.get("msprof_cache") or {})
        if rtag in msprof_map:
            msprof_cache[f"{sha}:{rtag}"] = msprof_map[rtag]

        rounds_raw = list(payload.get("rounds") or [])
        for r in rounds_raw:
            cur_idx = int(r.get("round_idx", -1))
            if cur_idx == r_idx:
                r["promoted"] = True
                r["rolled_back_l3"] = False
                r["stagnation_rounds_after"] = 0
                r["best_round_surrogate_score"] = round(cand_res.surrogate_score, 6)
                r["best_so_far_surrogate_score"] = round(cand_res.surrogate_score, 6)
            if (
                cur_idx > r_idx
                and float(r.get("best_so_far_surrogate_score", 0.0)) < cand_res.surrogate_score
            ):
                r["best_so_far_surrogate_score"] = round(cand_res.surrogate_score, 6)
                r["best_so_far_speedup"] = round(cand_res.avg_speedup, 6)

        prom_rounds = sorted(promoted_set | {r_idx}) if r_idx >= 0 else sorted(promoted_set)
        rb_rounds = [int(x) for x in payload.get("rolled_back_rounds", ()) if int(x) != r_idx]
        stag_map = {str(k): int(v) for k, v in (payload.get("stagnation_by_regime") or {}).items()}
        stag_map[rtag] = 0

        payload["incumbent_source"] = patched_src
        payload["best_result"] = cand_res.to_dict()
        payload["promoted_rounds"] = prom_rounds
        payload["rolled_back_rounds"] = rb_rounds
        payload["stagnation_by_regime"] = stag_map
        payload["stagnation_rounds"] = 0
        payload["current_regime"] = rtag
        payload["last_unpromoted_regime"] = None
        payload["rounds"] = rounds_raw
        payload["msprof_cache"] = msprof_cache
        atomic_write_text(checkpoint_path, json.dumps(payload, indent=2) + "\n")
        log.info(
            "SALVAGE_PROMOTED op=%s %s score=%.6f -> %.6f speedup=%.6f run_id=%s",
            operator_name,
            cid,
            best_res.surrogate_score,
            cand_res.surrogate_score,
            cand_res.avg_speedup,
            l3_res.run_id,
        )
        return True
    return False


def _configure_marathon_logging(log_path: Path | None) -> None:
    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            stream=sys.stderr,
        )
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        resolved = str(log_path.resolve())
        already = any(
            isinstance(h, logging.FileHandler) and h.baseFilename == resolved
            for h in root_logger.handlers
        )
        if not already:
            fh = logging.FileHandler(resolved, encoding="utf-8")
            fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
            root_logger.addHandler(fh)


def run_operator_marathon(
    config: MarathonConfig | None = None,
    **overrides: object,
) -> int:
    """Configure workspace, environment, and run ExtremeCampaignRunner."""
    if config is None:
        cfg = MarathonConfig(**overrides)  # type: ignore[arg-type]
    elif overrides:
        data = {k: getattr(config, k) for k in config.__dataclass_fields__}
        data.update(overrides)
        cfg = MarathonConfig(**data)  # type: ignore[arg-type]
    else:
        cfg = config

    _source_root, worktree = _resolve_worktree(cfg)
    _ensure_sys_paths(worktree)

    short_op = short_operator_tag(cfg.operator_name)
    state_dir = (
        Path(cfg.state_dir).expanduser().resolve()
        if cfg.state_dir is not None
        else (worktree / ".tilefoundry-marathon" / cfg.operator_name).resolve()
    )
    state_dir.mkdir(parents=True, exist_ok=True)

    target_module = (
        Path(cfg.target_module_path).expanduser().resolve()
        if cfg.target_module_path is not None
        else (worktree / get_operator_module_relpath(cfg.operator_name)).resolve()
    )
    checkpoint_path = (
        Path(cfg.checkpoint_path).expanduser().resolve()
        if cfg.checkpoint_path is not None
        else (state_dir / "checkpoint.json")
    )
    report_path = (
        Path(cfg.report_path).expanduser().resolve()
        if cfg.report_path is not None
        else (state_dir / "report.json")
    )
    campaign_kb_path = (
        Path(cfg.campaign_kb_path).expanduser().resolve()
        if cfg.campaign_kb_path is not None
        else (state_dir / "campaign_kb.yaml")
    )

    _configure_marathon_logging(cfg.log_path)
    op_logger = logging.getLogger(f"tilefoundry_ascend.marathon.{short_op}")

    env_file = ensure_marathon_env_file(
        operator_name=cfg.operator_name,
        worktree=worktree,
        state_dir=state_dir,
        env_file=cfg.env_file,
    )
    configure_marathon_process_env(
        operator_name=cfg.operator_name,
        env_file=env_file,
        worktree=worktree,
    )
    op_logger.info(
        "MARATHON_BOOT op=%s device=%s worktree=%s env_file=%s",
        cfg.operator_name,
        cfg.device_id,
        worktree,
        env_file,
    )
    if cfg.auto_bootstrap_seed and not target_module.is_file():
        raw_cb = (
            parse_simple_env_file(env_file).get("TILEFOUNDRY_NPU_CANNBENCH_REPO", "").strip()
            or os.environ.get("TILEFOUNDRY_NPU_CANNBENCH_REPO", "").strip()
        )
        if ensure_operator_seed_module(
            cfg.operator_name,
            target_module,
            cannbench_repo=Path(raw_cb) if raw_cb else None,
        ):
            op_logger.info(
                "AUTO_BOOTSTRAP_SEED op=%s target=%s",
                cfg.operator_name,
                target_module,
            )

    l2_evaluator = build_resilient_l2_evaluator(
        operator_name=cfg.operator_name,
        worktree=worktree,
        env_file=env_file,
        device_id=str(cfg.device_id),
        warmup=cfg.warmup,
        repeat=cfg.repeat,
        op_logger=op_logger,
    )
    l3_runner = build_prewarmed_l3_runner(
        operator_name=cfg.operator_name,
        worktree=worktree,
        env_file=env_file,
        device_id=str(cfg.device_id),
        prewarm_l3=cfg.prewarm_l3,
        op_logger=op_logger,
    )

    if cfg.salvage_beam_candidates:
        salvage_unpromoted_beam_winner(
            operator_name=cfg.operator_name,
            checkpoint_path=checkpoint_path,
            target_module_path=target_module,
            worktree=worktree,
            env_file=env_file,
            device_id=str(cfg.device_id),
            min_surrogate_gain=cfg.min_surrogate_gain,
            l2_evaluator=l2_evaluator,
            l3_runner=l3_runner,
            op_logger=op_logger,
        )

    perf_target = PerfTargetSpec(
        operator_name=cfg.operator_name,
        target_avg_speedup=cfg.target_avg_speedup,
        target_min_case_speedup=cfg.target_min_case_speedup,
        target_roofline_eta=cfg.target_roofline_eta,
        target_surrogate_score=cfg.target_surrogate_score,
        min_optimization_rounds=cfg.min_optimization_rounds,
        per_anchor_max_us={},
    )
    campaign_cfg = CampaignConfig(
        operator_name=cfg.operator_name,
        max_rounds=cfg.max_rounds,
        max_wall_clock_min=cfg.max_wall_clock_min,
        k_rule_workers=cfg.k_rule_workers,
        max_syntax_repairs=cfg.max_syntax_repairs,
        min_l2_gain_for_l3=cfg.min_l2_gain_for_l3,
        min_surrogate_gain=cfg.min_surrogate_gain,
        min_flywheel_gain=cfg.min_flywheel_gain,
        stagnation_escalate_rounds=cfg.stagnation_escalate_rounds,
        stagnation_freeze_rounds=cfg.stagnation_freeze_rounds,
        disable_early_goal_stop=cfg.disable_early_goal_stop,
        run_final_l4=cfg.run_final_l4,
        l4_regression_ratio=cfg.l4_regression_ratio,
        device_id=str(cfg.device_id),
        warmup=cfg.warmup,
        repeat=cfg.repeat,
        enable_tilesize_prescreen=cfg.enable_tilesize_prescreen,
        exploit_effort=cfg.exploit_effort,
    )
    tracker = BayesianRuleTracker.from_seed_yaml()
    flywheel = FlywheelDistiller(
        funnel=GateFunnel(skip_stage_b=True),
        tracker=tracker,
        probe_runner=lambda spec: run_micro_probe_via_agent_gates(
            spec,
            source_root=worktree,
            env_file=env_file,
            device_id=str(cfg.device_id),
            min_speedup=1.08,
            timeout_sec=360,
        ),
    )
    runner = ExtremeCampaignRunner(
        config=campaign_cfg,
        target_module_path=target_module,
        perf_target=perf_target,
        llm_backend=FastCreativeAgyBackend(escalate_model=cfg.escalate_model),
        tracker=tracker,
        flywheel_distiller=flywheel,
        l2_batch_evaluator=l2_evaluator,
        msprof_collector=lambda op_name, cand_path, anc_seq: collect_bottleneck_msprof_via_agent_gates(
            op_name,
            cand_path,
            anc_seq,
            source_root=worktree,
            env_file=env_file,
            device_id=str(cfg.device_id),
            timeout_sec=600,
        ),
        l3_contract_runner=l3_runner,
        l4_cannbench_runner=lambda _mod_path: run_l4_official_cannbench_via_agent_gates(
            operator_name=cfg.operator_name,
            source_root=worktree,
            env_file=env_file,
            device_id=str(cfg.device_id),
            timeout_sec=3600,
        ),
        checkpoint_path=checkpoint_path,
        campaign_kb_path=campaign_kb_path,
        source_root=worktree,
        env_file=env_file,
        stagnation_discount=cfg.stagnation_discount,
        op_logger=op_logger,
    )
    resume_flag = checkpoint_path.is_file() if cfg.resume is None else bool(cfg.resume)
    op_logger.info(
        "Launching ExtremeCampaignRunner op=%s device=%s resume=%s max_rounds=%d "
        "max_wall_min=%.1f target_score=%.1f target_eta=%.3f worktree=%s",
        cfg.operator_name,
        cfg.device_id,
        resume_flag,
        campaign_cfg.max_rounds,
        campaign_cfg.max_wall_clock_min,
        perf_target.target_surrogate_score,
        perf_target.target_roofline_eta,
        worktree,
    )
    report = asyncio.run(runner.run(resume=resume_flag))
    rendered = json.dumps(report.to_dict(), indent=2)
    atomic_write_text(report_path, rendered + "\n")
    print(rendered)
    return 0


def _atomic_write_json(path: Path, payload: object) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2) + "\n")


def _is_runner_alive(op: str, op_dir: Path, operator_name: str = "") -> bool:
    my_pid = os.getpid()
    dir_str = str(op_dir)
    needles = [f"/{op}/run_extreme.py", dir_str]
    op_tokens = [f"--operator {op}", f"--operator={op}"]
    if operator_name:
        op_tokens.extend([f"--operator {operator_name}", f"--operator={operator_name}"])
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid in (1, my_pid):
            continue
        try:
            cmd = (
                Path(f"/proc/{pid}/cmdline")
                .read_bytes()
                .replace(b"\x00", b" ")
                .decode("utf-8", "replace")
            )
        except OSError:
            continue
        if "wait_next_round" in cmd or "watch" in cmd:
            continue
        if any(needle in cmd for needle in needles):
            return True
        is_cli = "tilefoundry_ascend.agent.campaign" in cmd or "tilefoundry-ascend-marathon" in cmd
        if is_cli and any(tok in cmd for tok in op_tokens):
            return True
    return False


def _anchor_comparison(ckpt: Mapping[str, object]) -> list[dict[str, object]]:
    base_anc = {
        int(a["case_num"]): a
        for a in (ckpt.get("baseline_result") or {}).get("anchors", ())
        if isinstance(a, dict) and "case_num" in a
    }
    rows: list[dict[str, object]] = []
    for a in (ckpt.get("best_result") or {}).get("anchors", ()):
        if not isinstance(a, dict) or "case_num" not in a:
            continue
        cnum = int(a["case_num"])
        ba = base_anc.get(cnum, {})
        rows.append(
            {
                "case_num": cnum,
                "regime_tag": a.get("regime_tag", ""),
                "base_us": ba.get("cand_us"),
                "best_us": a.get("cand_us"),
                "base_sp": ba.get("speedup"),
                "best_sp": a.get("speedup"),
                "best_eta": a.get("roofline_eta"),
                "best_eq3": a.get("eq3_score"),
            }
        )
    return rows


def _read_watch_state(op: str, op_dir: Path) -> dict[str, object] | None:
    ckpt_path = op_dir / "checkpoint.json"
    rep_path = op_dir / "report.json"
    if not ckpt_path.is_file():
        return None
    try:
        ckpt = json.loads(ckpt_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    rep = None
    report_done = False
    if rep_path.is_file():
        try:
            rep = json.loads(rep_path.read_text(encoding="utf-8"))
            report_done = isinstance(rep, dict) and "stop_reason" in rep
        except (OSError, ValueError):
            report_done = False

    best = ckpt.get("best_result") or {}
    base = ckpt.get("baseline_result") or {}
    rounds = list(ckpt.get("rounds") or ())
    operator_name = str(ckpt.get("operator_name", op))
    return {
        "op": op,
        "operator_name": operator_name,
        "rounds_len": len(rounds),
        "rounds": rounds,
        "promoted_rounds": [int(x) for x in (ckpt.get("promoted_rounds") or ())],
        "rolled_back_rounds": [int(x) for x in (ckpt.get("rolled_back_rounds") or ())],
        "base_score": round(float(base.get("surrogate_score", 0.0)), 6),
        "best_score": round(float(best.get("surrogate_score", 0.0)), 6),
        "best_speedup": round(float(best.get("avg_speedup", 0.0)), 6),
        "min_case_speedup": round(float(best.get("min_case_speedup", 0.0)), 6),
        "avg_roofline_eta": round(float(best.get("avg_roofline_eta", 0.0)), 6),
        "stagnation_by_regime": dict(ckpt.get("stagnation_by_regime") or {}),
        "frozen_regimes": list(ckpt.get("frozen_regimes") or ()),
        "stop_reason": str(ckpt.get("stop_reason", "in_progress")),
        "report_done": report_done,
        "l4_result": (rep.get("l4_result") if isinstance(rep, dict) else None)
        or ckpt.get("l4_result"),
        "anchors": _anchor_comparison(ckpt),
        "runner_alive": _is_runner_alive(op, op_dir, operator_name=operator_name),
    }


def _snapshot_key(state: Mapping[str, object]) -> tuple:
    return (
        state["rounds_len"],
        tuple(state["promoted_rounds"]),
        state["best_score"],
        state["stop_reason"],
        state["report_done"],
    )


def _prev_key(prev: Mapping[str, object]) -> tuple:
    return (
        int(prev.get("rounds_len", 0)),
        tuple(int(x) for x in (prev.get("promoted_rounds") or ())),
        round(float(prev.get("best_score", 0.0)), 6),
        str(prev.get("stop_reason", "in_progress")),
        bool(prev.get("report_done", False)),
    )


def _tail_log(op_dir: Path, lines: int = 35) -> str:
    log_path = op_dir / "campaign.log"
    if not log_path.is_file():
        return ""
    try:
        all_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(all_lines[-lines:])


def _detect_watch_changes(
    state_dirs: Mapping[str, Path],
    last_reported: Mapping[str, object],
) -> tuple[list[dict[str, object]], dict[str, dict[str, object]]]:
    states: dict[str, dict[str, object]] = {}
    events: list[dict[str, object]] = []
    for op, op_dir in state_dirs.items():
        st = _read_watch_state(op, op_dir)
        if st is None:
            continue
        states[op] = st
        prev = last_reported.get(op) or {}
        if not isinstance(prev, Mapping):
            prev = {}
        if _snapshot_key(st) == _prev_key(prev):
            continue

        prev_len = int(prev.get("rounds_len", 0))
        rounds_list = st["rounds"]
        assert isinstance(rounds_list, list)
        if int(st["rounds_len"]) > prev_len:
            for r in rounds_list[prev_len:]:
                ev: dict[str, object] = {
                    "event_type": "round_completed",
                    "op": op,
                    "operator_name": st["operator_name"],
                    "round_idx": r.get("round_idx"),
                    "regime_tag": r.get("regime_tag"),
                    "parent_lane": r.get("parent_lane"),
                    "creative_effort": r.get("creative_effort"),
                    "l0_l1_passed_ids": r.get("l0_l1_passed_ids", []),
                    "l0_l1_failed": r.get("l0_l1_failed", []),
                    "best_round_surrogate_score": r.get("best_round_surrogate_score"),
                    "best_so_far_surrogate_score": r.get("best_so_far_surrogate_score"),
                    "best_round_speedup": r.get("best_round_speedup"),
                    "best_so_far_speedup": r.get("best_so_far_speedup"),
                    "promoted": bool(r.get("promoted", False)),
                    "rolled_back_l3": bool(r.get("rolled_back_l3", False)),
                    "stagnation_by_regime": st["stagnation_by_regime"],
                    "frozen_regimes": st["frozen_regimes"],
                }
                if ev["promoted"]:
                    ev["anchors"] = st["anchors"]
                events.append(ev)
        elif float(st["best_score"]) > round(float(prev.get("best_score", 0.0)), 6) or (
            st["promoted_rounds"] != [int(x) for x in (prev.get("promoted_rounds") or ())]
        ):
            events.append(
                {
                    "event_type": "salvage_promoted",
                    "op": op,
                    "operator_name": st["operator_name"],
                    "best_score": st["best_score"],
                    "best_speedup": st["best_speedup"],
                    "min_case_speedup": st["min_case_speedup"],
                    "avg_roofline_eta": st["avg_roofline_eta"],
                    "promoted_rounds": st["promoted_rounds"],
                    "stagnation_by_regime": st["stagnation_by_regime"],
                    "anchors": st["anchors"],
                }
            )

        if (bool(st["report_done"]) and not bool(prev.get("report_done", False))) or (
            str(st["stop_reason"]) != "in_progress"
            and str(prev.get("stop_reason", "in_progress")) == "in_progress"
        ):
            events.append(
                {
                    "event_type": "campaign_finished",
                    "op": op,
                    "operator_name": st["operator_name"],
                    "stop_reason": st["stop_reason"],
                    "report_done": st["report_done"],
                    "total_rounds": st["rounds_len"],
                    "base_score": st["base_score"],
                    "best_score": st["best_score"],
                    "best_speedup": st["best_speedup"],
                    "min_case_speedup": st["min_case_speedup"],
                    "avg_roofline_eta": st["avg_roofline_eta"],
                    "promoted_rounds": st["promoted_rounds"],
                    "l4_result": st["l4_result"],
                    "anchors": st["anchors"],
                }
            )
    return events, states


def _build_watch_output(
    state_dirs: Mapping[str, Path],
    events: list[dict[str, object]],
    states: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    scoreboard = {}
    for op, st in states.items():
        scoreboard[op] = {
            "operator_name": st["operator_name"],
            "rounds_len": st["rounds_len"],
            "promoted_rounds": st["promoted_rounds"],
            "base_score": st["base_score"],
            "best_score": st["best_score"],
            "best_speedup": st["best_speedup"],
            "min_case_speedup": st["min_case_speedup"],
            "avg_roofline_eta": st["avg_roofline_eta"],
            "stagnation_by_regime": st["stagnation_by_regime"],
            "frozen_regimes": st["frozen_regimes"],
            "stop_reason": st["stop_reason"],
            "report_done": st["report_done"],
            "runner_alive": st["runner_alive"],
        }
    both_finished = len(states) == len(state_dirs) and len(states) > 0 and all(
        bool(st["report_done"]) and str(st["stop_reason"]) != "in_progress"
        for st in states.values()
    )
    return {
        "events": events,
        "scoreboard": scoreboard,
        "both_finished": both_finished,
    }


def _persist_watch_states(state_path: Path, states: Mapping[str, Mapping[str, object]]) -> None:
    saved = {
        op: {
            "rounds_len": st["rounds_len"],
            "promoted_rounds": st["promoted_rounds"],
            "best_score": st["best_score"],
            "stop_reason": st["stop_reason"],
            "report_done": st["report_done"],
        }
        for op, st in states.items()
    }
    _atomic_write_json(state_path, saved)


def _setup_inotify(state_dirs: Mapping[str, Path]) -> int | None:
    try:
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.inotify_init1.argtypes = [ctypes.c_int]
        libc.inotify_init1.restype = ctypes.c_int
        libc.inotify_add_watch.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_uint32]
        libc.inotify_add_watch.restype = ctypes.c_int
        fd = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
        if fd < 0:
            return None
        for op_dir in state_dirs.values():
            if not op_dir.is_dir():
                continue
            wd = libc.inotify_add_watch(fd, str(op_dir).encode("utf-8"), _WATCH_MASK)
            if wd < 0:
                os.close(fd)
                return None
        return fd
    except OSError:
        return None


def watch_marathon_rounds(
    *,
    state_dirs: Mapping[str, Path],
    state_file: Path,
    poll_interval_sec: float = 30.0,
    once: bool = False,
) -> dict[str, object]:
    """Watch checkpoint directories for round completions or unexpected exits."""
    last_reported: dict[str, object] = {}
    if state_file.is_file():
        try:
            loaded = json.loads(state_file.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                last_reported = loaded
        except (OSError, ValueError):
            last_reported = {}

    events, states = _detect_watch_changes(state_dirs, last_reported)
    if events or once:
        _persist_watch_states(state_file, states)
        return _build_watch_output(state_dirs, events, states)

    fd = _setup_inotify(state_dirs)
    dead_streak = {op: 0 for op in state_dirs}
    try:
        while True:
            if fd is not None:
                readable, _, _ = select.select([fd], [], [], poll_interval_sec)
                if readable:
                    while True:
                        try:
                            chunk = os.read(fd, 4096)
                            if not chunk:
                                break
                        except BlockingIOError:
                            break
            else:
                time.sleep(poll_interval_sec)

            events, states = _detect_watch_changes(state_dirs, last_reported)
            if events:
                _persist_watch_states(state_file, states)
                return _build_watch_output(state_dirs, events, states)

            unexpected: list[dict[str, object]] = []
            for op, st in states.items():
                still_running = st["stop_reason"] == "in_progress" and not st["report_done"]
                if still_running and not st["runner_alive"]:
                    dead_streak[op] += 1
                    if dead_streak[op] >= 2:
                        unexpected.append(
                            {
                                "event_type": "runner_exited_unexpectedly",
                                "op": op,
                                "operator_name": st["operator_name"],
                                "rounds_len": st["rounds_len"],
                                "best_score": st["best_score"],
                                "log_tail": _tail_log(state_dirs[op], 40),
                            }
                        )
                else:
                    dead_streak[op] = 0
            if unexpected:
                return _build_watch_output(state_dirs, unexpected, states)
    finally:
        if fd is not None:
            os.close(fd)


def _discover_watch_state_dirs(
    base_dir: Path | None,
    explicit_pairs: Sequence[str],
) -> tuple[dict[str, Path], Path]:
    if explicit_pairs:
        resolved: dict[str, Path] = {}
        first_parent: Path | None = None
        for item in explicit_pairs:
            if "=" in item:
                label, raw_dir = item.split("=", 1)
            else:
                p = Path(item).expanduser().resolve()
                label, raw_dir = p.name, str(p)
            p = Path(raw_dir).expanduser().resolve()
            resolved[label.strip()] = p
            if first_parent is None:
                first_parent = p.parent
        assert first_parent is not None
        anchor = Path(base_dir).expanduser().resolve() if base_dir else first_parent
        return resolved, anchor / "last_reported.json"

    anchor = (
        Path(base_dir).expanduser().resolve()
        if base_dir is not None
        else (default_source_root() / ".tilefoundry-marathon").resolve()
    )
    resolved = {}
    if anchor.is_dir():
        for child in sorted(anchor.iterdir()):
            if child.is_dir() and (child / "checkpoint.json").is_file():
                resolved[child.name] = child
    if not resolved and anchor.is_dir():
        for child in sorted(anchor.iterdir()):
            if child.is_dir():
                resolved[child.name] = child
    return resolved, anchor / "last_reported.json"


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--operator",
        required=True,
        help="Target operator to optimize (short name e.g. rms_norm or task id e.g. level2/rms_norm).",
    )
    parser.add_argument("--device-id", default="1", help="Authorized NPU device ID.")
    parser.add_argument("--source-root", default=None, help="TileFoundry repository root.")
    parser.add_argument("--worktree", default=None, help="Optional isolated worktree root.")
    parser.add_argument(
        "--isolate-worktree",
        action="store_true",
        help="Create or reuse an isolated sibling git worktree automatically.",
    )
    parser.add_argument("--state-dir", default=None, help="Directory for checkpoint/logs/env.")
    parser.add_argument("--target-file", default=None, help="Override target kernels.py path.")
    parser.add_argument("--env-file", default=None, help="Optional pre-existing env.sh path.")
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint JSON path.")
    parser.add_argument("--report-json", default=None, help="Override final report JSON path.")
    parser.add_argument("--campaign-kb", default=None, help="Override distilled KB YAML path.")
    parser.add_argument("--log-file", default=None, help="Optional file path to tee runner logs.")
    parser.add_argument("--max-rounds", type=int, default=80)
    parser.add_argument("--max-wall-clock-min", type=float, default=720.0)
    parser.add_argument("--k-rule-workers", type=int, default=2)
    parser.add_argument("--no-final-l4", action="store_true", help="Skip final L4 CANNBench gate.")
    parser.add_argument("--fresh", action="store_true", help="Ignore prior checkpoint and restart.")
    parser.add_argument("--no-salvage", action="store_true", help="Disable beam salvage check.")


def build_marathon_arg_parser() -> argparse.ArgumentParser:
    """Construct standalone CLI parser for zero-config marathon runs and watching."""
    parser = argparse.ArgumentParser(
        prog="tilefoundry-ascend-marathon",
        description="Zero-config extreme score-sprint runner and round watcher.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run_parser = sub.add_parser("run", help="Run or auto-resume a score-sprint campaign.")
    _add_run_arguments(run_parser)

    watch_parser = sub.add_parser("watch", help="Watch marathon rounds and emit JSON events.")
    watch_parser.add_argument("--base-dir", default=None, help="Base directory containing runs.")
    watch_parser.add_argument(
        "--state-dir",
        action="append",
        default=[],
        help="Explicit label=path pair to watch (repeatable).",
    )
    watch_parser.add_argument("--state-file", default=None, help="Override last_reported JSON.")
    watch_parser.add_argument("--poll-interval-sec", type=float, default=30.0)
    watch_parser.add_argument("--once", action="store_true", help="Snapshot once and exit.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute marathon CLI subcommands (defaults to 'run' when --operator is given)."""
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if raw_argv and raw_argv[0] not in ("run", "watch", "-h", "--help"):
        raw_argv = ["run", *raw_argv]
    args = build_marathon_arg_parser().parse_args(raw_argv)

    if args.command == "watch":
        state_dirs, default_state_file = _discover_watch_state_dirs(
            Path(args.base_dir) if args.base_dir else None,
            args.state_dir,
        )
        state_file = (
            Path(args.state_file).expanduser().resolve()
            if args.state_file
            else default_state_file
        )
        output = watch_marathon_rounds(
            state_dirs=state_dirs,
            state_file=state_file,
            poll_interval_sec=float(args.poll_interval_sec),
            once=bool(args.once),
        )
        print(json.dumps(output, indent=2), flush=True)
        return 0

    cfg = MarathonConfig(
        operator_name=str(args.operator),
        device_id=str(args.device_id),
        source_root=Path(args.source_root) if args.source_root else None,
        worktree=Path(args.worktree) if args.worktree else None,
        isolate_worktree=bool(args.isolate_worktree),
        state_dir=Path(args.state_dir) if args.state_dir else None,
        target_module_path=Path(args.target_file) if args.target_file else None,
        env_file=Path(args.env_file) if args.env_file else None,
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
        report_path=Path(args.report_json) if args.report_json else None,
        campaign_kb_path=Path(args.campaign_kb) if args.campaign_kb else None,
        log_path=Path(args.log_file) if args.log_file else None,
        max_rounds=int(args.max_rounds),
        max_wall_clock_min=float(args.max_wall_clock_min),
        k_rule_workers=int(args.k_rule_workers),
        run_final_l4=not bool(args.no_final_l4),
        resume=False if args.fresh else None,
        salvage_beam_candidates=not bool(args.no_salvage),
    )
    return run_operator_marathon(cfg)


__all__ = [
    "ExtremeCampaignRunner",
    "FastCreativeAgyBackend",
    "MarathonConfig",
    "atomic_write_text",
    "build_marathon_arg_parser",
    "build_prewarmed_l3_runner",
    "build_resilient_l2_evaluator",
    "configure_marathon_process_env",
    "discover_marathon_environment",
    "ensure_marathon_env_file",
    "ensure_operator_seed_module",
    "main",
    "run_operator_marathon",
    "salvage_unpromoted_beam_winner",
    "short_operator_tag",
    "watch_marathon_rounds",
]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
