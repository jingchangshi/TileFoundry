"""Lease-locked AgentGates L2 surrogate CANNBench batch evaluator on NPU."""

from __future__ import annotations

import gc
import hashlib
import importlib
import importlib.util
import json
import math
import os
import re
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tilefoundry_ascend.agent.campaign.anchor_specs import (
    AnchorCaseSpec,
    default_source_root,
    get_default_anchors,
    get_operator_entrypoint,
    get_operator_module_relpath,
    load_anchor_harness_and_cases,
    load_cannbench_contract_module,
    resolve_cannbench_repo,
)
from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (
    FAST_REQUEST_ENV,
    FAST_REQUEST_SCHEMA,
    FAST_RESPONSE_SCHEMA,
    FAST_SOURCE_ROOT_ENV,
    FAST_STATUS_CANDIDATE_FAILURE,
    FAST_STATUS_INFRASTRUCTURE_ERROR,
    FAST_STATUS_OK,
    FAST_STATUS_REQUEST_ERROR,
    AnchorEvalMetrics,
    CandidateSurrogateResult,
    aggregate_surrogate_result,
    build_anchor_metrics,
    ensure_npu_compare_shim_dir,
    measure_npu_ring_burst_stats,
    measure_npu_ring_burst_us,
    measure_paired_interleaved_micro_burst,
)
from tilefoundry_ascend.agent.flywheel.ablation_distiller import (
    MicroProbeSpec,
    verify_micro_probe_hardware,
)
from tilefoundry_ascend.ascend.cost_model.msprof_report_parser import (
    AggregatedMsprofReport,
    P2InstructionDiveSummary,
    extract_p2_from_objdump,
    find_p2_trace_files,
    parse_msprof_op_report,
    parse_p2_instruction_trace,
)
from tilefoundry_ascend.ascend.passes.physical_ledger import build_objective_physical_ledger

_ENV_SHIM_DIR = os.environ.get("TF_SURROGATE_SHIM_DIR", "").strip()
if (
    _ENV_SHIM_DIR
    and (_ENV_SHIM_DIR not in sys.path)
    and (Path(_ENV_SHIM_DIR) / "npu_compare.py").is_file()
):
    sys.path.insert(0, _ENV_SHIM_DIR)


def _require_npu_env() -> None:
    if not hasattr(torch, "npu"):
        try:
            importlib.import_module("torch_npu")
        except ImportError:
            pass
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        if os.environ.get(FAST_REQUEST_ENV, "").strip() or os.environ.get(
            "TF_SURROGATE_BATCH_REQUEST", ""
        ).strip():
            raise RuntimeError("Ascend NPU device is unavailable for surrogate batch request.")
        pytest.skip("Ascend NPU device is unavailable on this host.")


def _resolve_batch_cannbench_repo(source_root: Path) -> Path:
    explicit = os.environ.get("TF_SURROGATE_CANNBENCH_REPO", "").strip()
    if explicit and Path(explicit).is_dir():
        return Path(explicit).resolve()
    try:
        return resolve_cannbench_repo()
    except FileNotFoundError:
        sibling = source_root.parent / "cann-bench"
        if (sibling / "tasks").is_dir():
            return sibling.resolve()
        raise


def _ensure_npu_compare(cannbench_repo: Path, shim_dir: Path) -> None:
    if importlib.util.find_spec("npu_compare") is not None:
        return
    created = ensure_npu_compare_shim_dir(shim_dir, cannbench_repo=cannbench_repo)
    if str(created) not in sys.path:
        sys.path.insert(0, str(created))


def _load_candidate_entrypoint(module_path: Path, entrypoint: str, unique_tag: str):
    mod_name = f"_tf_surrogate_cand_{unique_tag}"
    spec = importlib.util.spec_from_file_location(mod_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load candidate module {module_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, entrypoint):
        raise AttributeError(f"{module_path} does not define {entrypoint!r}")
    return getattr(mod, entrypoint), mod


def _measure_npu_latency_us(fn_call, *, warmup: int, repeat: int) -> float:
    try:
        profiler = importlib.import_module("tilelang.profiler")
        lat_ms = profiler.do_bench(
            fn_call,
            _n_warmup=warmup,
            _n_repeat=repeat,
            backend="event",
            return_mode="median",
            device=torch.device("npu:0"),
        )
        return float(lat_ms) * 1000.0
    except Exception:
        for _ in range(max(1, warmup)):
            fn_call()
        torch.npu.synchronize()
        samples_ms: list[float] = []
        for _ in range(max(1, repeat)):
            start_ev = torch.npu.Event(enable_timing=True)
            end_ev = torch.npu.Event(enable_timing=True)
            start_ev.record()
            fn_call()
            end_ev.record()
            torch.npu.synchronize()
            samples_ms.append(float(start_ev.elapsed_time(end_ev)))
        return float(statistics.median(samples_ms)) * 1000.0


def _measure_npu_latency_stats_us(
    fn_call,
    *,
    warmup: int,
    repeat: int,
) -> tuple[float, float]:
    """Collect NPU Event latency samples and return (median_us, batch_cv)."""
    if not hasattr(getattr(torch, "npu", None), "Event"):
        return float(_measure_npu_latency_us(fn_call, warmup=warmup, repeat=repeat)), 0.0
    for _ in range(max(6, int(warmup))):
        fn_call()
    torch.npu.synchronize()
    samples_us: list[float] = []
    for _ in range(max(1, repeat)):
        start_ev = torch.npu.Event(enable_timing=True)
        end_ev = torch.npu.Event(enable_timing=True)
        start_ev.record()
        fn_call()
        end_ev.record()
        torch.npu.synchronize()
        samples_us.append(float(start_ev.elapsed_time(end_ev)) * 1000.0)
    med = float(statistics.median(samples_us))
    cv = float(statistics.pstdev(samples_us)) / max(med, 1e-9) if len(samples_us) >= 2 else 0.0
    return med, cv


def _measure_null_launch_us(warmup: int = 5, repeat: int = 20) -> float | None:
    """Measure median minimal no-op NPU kernel launch latency in microseconds."""
    npu_mod = getattr(torch, "npu", None)
    if npu_mod is None or not hasattr(npu_mod, "Event") or not hasattr(npu_mod, "synchronize"):
        return None
    is_avail = getattr(npu_mod, "is_available", None)
    if callable(is_avail) and not is_avail():
        return None
    try:
        buf = torch.empty((1,), dtype=torch.float32, device="npu:0")
        for _ in range(max(1, int(warmup))):
            buf.zero_()
        npu_mod.synchronize()
        samples_us: list[float] = []
        for _ in range(max(1, int(repeat))):
            start_ev = npu_mod.Event(enable_timing=True)
            end_ev = npu_mod.Event(enable_timing=True)
            start_ev.record()
            buf.zero_()
            end_ev.record()
            npu_mod.synchronize()
            elapsed_us = float(start_ev.elapsed_time(end_ev)) * 1000.0
            if math.isfinite(elapsed_us) and elapsed_us > 0.0:
                samples_us.append(elapsed_us)
        if not samples_us:
            return None
        med = float(statistics.median(samples_us))
        return med if math.isfinite(med) and med > 0.0 else None
    except Exception:
        return None


def _normalize_pmu_ratio(raw_val: int | float) -> float:
    """Normalize a PMU ratio in [0, 1] or percentage in [0, 100] onto [0.0, 1.0]."""
    val = float(raw_val)
    if val > 1.05:
        val /= 100.0
    return max(0.0, min(1.0, val))


def _read_prof_number(prof: object, *names: str) -> int | float | None:
    """Return the first present finite non-boolean numeric counter on a profiler object."""
    for name in names:
        raw: object | None
        if isinstance(prof, Mapping):
            raw = prof.get(name)
        else:
            raw = getattr(prof, name, None)
        if raw is None or isinstance(raw, bool):
            continue
        if isinstance(raw, (int, float)) and math.isfinite(float(raw)):
            return raw
    return None


def _extract_msprof_metrics(prof: object) -> dict[str, float | int]:
    """Extract only genuinely measured non-null PMU counters from a profiler result."""
    metrics: dict[str, float | int] = {}

    aic_mad_raw = _read_prof_number(prof, "aic_mad", "aic_mac_ratio")
    if aic_mad_raw is not None:
        aic_mad = _normalize_pmu_ratio(aic_mad_raw)
        metrics["aic_mac_ratio"] = aic_mad
        metrics["aic_mad"] = aic_mad

    for key in ("aic_scalar", "aic_mte1", "aic_mte3", "aic_fixpipe", "aiv_scalar", "aiv_mte3"):
        raw = _read_prof_number(prof, key)
        if raw is not None:
            metrics[key] = _normalize_pmu_ratio(raw)

    aiv_vec_raw = _read_prof_number(prof, "aiv_vec", "aiv_vec_ratio")
    if aiv_vec_raw is not None:
        aiv_vec = _normalize_pmu_ratio(aiv_vec_raw)
        metrics["aiv_vec_ratio"] = aiv_vec
        metrics["aiv_vec"] = aiv_vec

    aiv0_raw = _read_prof_number(prof, "aiv0_vec_ratio", "aiv0_vec")
    if aiv0_raw is not None:
        metrics["aiv0_vec_ratio"] = _normalize_pmu_ratio(aiv0_raw)
    aiv1_raw = _read_prof_number(prof, "aiv1_vec_ratio", "aiv1_vec")
    if aiv1_raw is not None:
        metrics["aiv1_vec_ratio"] = _normalize_pmu_ratio(aiv1_raw)

    aic_mte2_raw = _read_prof_number(prof, "aic_mte2")
    norm_aic_mte2: float | None = None
    if aic_mte2_raw is not None:
        norm_aic_mte2 = _normalize_pmu_ratio(aic_mte2_raw)
        metrics["aic_mte2"] = norm_aic_mte2

    aiv_mte2_raw = _read_prof_number(prof, "aiv_mte2")
    norm_aiv_mte2: float | None = None
    if aiv_mte2_raw is not None:
        norm_aiv_mte2 = _normalize_pmu_ratio(aiv_mte2_raw)
        metrics["aiv_mte2"] = norm_aiv_mte2

    explicit_mte2_wait = _read_prof_number(prof, "mte2_wait_ratio")
    if explicit_mte2_wait is not None:
        metrics["mte2_wait_ratio"] = _normalize_pmu_ratio(explicit_mte2_wait)
    elif norm_aiv_mte2 is not None or norm_aic_mte2 is not None:
        metrics["mte2_wait_ratio"] = max(
            val for val in (norm_aiv_mte2, norm_aic_mte2) if val is not None
        )

    explicit_sync = _read_prof_number(prof, "sync_stall_ratio")
    if explicit_sync is not None:
        metrics["sync_stall_ratio"] = _normalize_pmu_ratio(explicit_sync)

    l2_hit_raw = _read_prof_number(prof, "l2_hit_rate", "l2_cache_hit_rate")
    if l2_hit_raw is not None:
        metrics["l2_hit_rate"] = _normalize_pmu_ratio(l2_hit_raw)

    bank_conflict_raw = _read_prof_number(
        prof, "bank_conflict_ratio", "ub_bank_conflict_ratio", "resource_conflict_ratio"
    )
    if bank_conflict_raw is not None:
        metrics["bank_conflict_ratio"] = _normalize_pmu_ratio(bank_conflict_raw)

    for int_key in ("aic_total_cycles", "aiv_total_cycles"):
        raw_int = _read_prof_number(prof, int_key)
        if raw_int is not None:
            metrics[int_key] = int(raw_int)

    hbm_read_raw = _read_prof_number(prof, "hbm_read_bytes")
    if hbm_read_raw is not None:
        metrics["hbm_read_bytes"] = int(hbm_read_raw)
    hbm_write_raw = _read_prof_number(prof, "hbm_write_bytes")
    if hbm_write_raw is not None:
        metrics["hbm_write_bytes"] = int(hbm_write_raw)
    actual_gm_raw = _read_prof_number(prof, "actual_gm_bytes")
    if actual_gm_raw is not None:
        metrics["actual_gm_bytes"] = int(actual_gm_raw)
    elif hbm_read_raw is not None or hbm_write_raw is not None:
        metrics["actual_gm_bytes"] = int(hbm_read_raw or 0) + int(hbm_write_raw or 0)

    raw_dur_us = _read_prof_number(prof, "dur_us", "measured_latency_us")
    if raw_dur_us is not None:
        metrics["dur_us"] = float(raw_dur_us)
    else:
        raw_dur_ns = _read_prof_number(prof, "dur_ns")
        if raw_dur_ns is not None:
            metrics["dur_us"] = float(raw_dur_ns) / 1000.0

    return metrics


_P1_AIC_METRICS = "PipeUtilization,Memory,MemoryL0,L2Cache,ResourceConflictRatio"
_P2_AIC_METRICS = (
    "PipeUtilization,Memory,MemoryL0,L2Cache,ResourceConflictRatio,PipeTimeline"
)


def _locate_opprof_dirs(output_dir: Path) -> list[Path]:
    """Return OPPROF_* directories under output_dir or output_dir itself if CSVs exist."""
    opprof_dirs = sorted(
        path for path in output_dir.rglob("OPPROF_*") if path.is_dir()
    )
    if opprof_dirs:
        return opprof_dirs
    if any(output_dir.rglob("*.csv")):
        return [output_dir]
    return []


def _aggregated_report_to_telemetry(report: AggregatedMsprofReport) -> dict[str, object]:
    """Convert an AggregatedMsprofReport into JSON-safe telemetry without fabricating counters."""
    telemetry: dict[str, object] = dict(report.to_telemetry_dict())
    if "bank_conflict_ratio" in telemetry and "ub_bank_conflict_ratio" not in telemetry:
        telemetry["ub_bank_conflict_ratio"] = telemetry["bank_conflict_ratio"]
    if "aic_mac_ratio" in telemetry and "aic_mad" not in telemetry:
        telemetry["aic_mad"] = telemetry["aic_mac_ratio"]
    if "aiv_vec_ratio" in telemetry and "aiv_vec" not in telemetry:
        telemetry["aiv_vec"] = telemetry["aiv_vec_ratio"]
    if "mte2_wait_ratio" in telemetry and "aiv_mte2" not in telemetry:
        telemetry["aiv_mte2"] = telemetry["mte2_wait_ratio"]
    if telemetry:
        telemetry["msprof_source"] = "msprof_op"
    return telemetry


def _format_p2_summary_fields(
    summary: P2InstructionDiveSummary,
    *,
    p2_source: str,
) -> dict[str, object]:
    """Serialize P2InstructionDiveSummary into compact JSON-safe deep-dive telemetry fields."""
    hotspots = [spot.to_dict() for spot in summary.source_line_hotspots]
    issue_intervals = {
        pipe: stats.to_dict() for pipe, stats in summary.issue_interval_us.items()
    }
    summary_lines = [
        (
            f"{spot.source_location} ({spot.instruction_count} instrs, "
            f"{spot.instruction_share:.1%}, pipes={','.join(spot.pipes)})"
        )
        for spot in summary.source_line_hotspots[:5]
    ]
    if summary.mte_waitflag_count > 0:
        summary_lines.append(
            f"mte_waitflag: {summary.mte_waitflag_count}/{summary.total_instructions} "
            f"({summary.mte_waitflag_share:.1%})"
        )
    return {
        "p2_source": p2_source,
        "p2_total_instructions": summary.total_instructions,
        "p2_mte_waitflag_count": summary.mte_waitflag_count,
        "p2_mte_waitflag_share": summary.mte_waitflag_share,
        "p2_attributed_instruction_share": summary.attributed_instruction_share,
        "p2_instruction_mix": dict(summary.instruction_mix),
        "p2_issue_interval_us": issue_intervals,
        "p2_hotspots": hotspots,
        "p2_summary_lines": summary_lines,
    }


def _discover_compiled_objects(
    mod_obj: object | None,
    module_path: Path | None,
    output_dir: Path | None = None,
) -> list[Path]:
    """Collect candidate compiled object/ELF paths for static DWARF fallback analysis."""
    found: list[Path] = []
    seen: set[Path] = set()

    def _add(raw_path: object) -> None:
        if raw_path is None:
            return
        candidate = Path(str(raw_path))
        if candidate.is_file() and candidate not in seen:
            seen.add(candidate)
            found.append(candidate)

    if mod_obj is not None:
        for attr in ("_COMPILED_OBJ_PATH", "_COMPILED_OBJ_PATHS", "compiled_obj_paths"):
            val = getattr(mod_obj, attr, None)
            if isinstance(val, (str, Path)):
                _add(val)
            elif isinstance(val, Sequence):
                for item in val:
                    _add(item)

    search_roots: list[Path] = []
    if output_dir is not None and output_dir.is_dir():
        search_roots.append(output_dir)
    if module_path is not None and module_path.parent.is_dir():
        search_roots.append(module_path.parent)
    tl_cache_raw = os.environ.get("TILELANG_CACHE_DIR", "").strip()
    if tl_cache_raw and Path(tl_cache_raw).is_dir():
        search_roots.append(Path(tl_cache_raw))
    for root in search_roots:
        for suffix in ("*.o", "*.so", "*.out", "*.elf"):
            for obj in sorted(root.rglob(suffix))[:8]:
                _add(obj)
    return found


def _build_msprof_op_case_driver_script(
    *,
    operator_name: str,
    module_path: Path,
    entrypoint: str,
    case_num: int,
    source_root: Path,
    cannbench_repo: Path,
) -> str:
    """Build an isolated Python driver script invoking one operator case for msprof op."""
    sys_paths = [
        str((source_root / "src").resolve()),
        str((source_root / "targets" / "ascend").resolve()),
        str(source_root.resolve()),
    ]
    npu_cmp_mod = sys.modules.get("npu_compare")
    npu_cmp_file = getattr(npu_cmp_mod, "__file__", None) if npu_cmp_mod is not None else None
    if npu_cmp_file:
        sys_paths.append(str(Path(str(npu_cmp_file)).resolve().parent))
    config = {
        "operator_name": operator_name,
        "module_path": str(module_path.resolve()),
        "entrypoint": entrypoint,
        "case_num": int(case_num),
        "source_root": str(source_root.resolve()),
        "cannbench_repo": str(cannbench_repo.resolve()),
        "sys_paths": sys_paths,
        "shim_dir": os.environ.get("TF_SURROGATE_SHIM_DIR", "").strip(),
    }
    encoded_cfg = json.dumps(config)
    return (
        "import importlib\n"
        "import importlib.util\n"
        "import json\n"
        "import sys\n"
        "from pathlib import Path\n"
        "import torch\n"
        f"CFG = json.loads({encoded_cfg!r})\n"
        "for path_item in CFG['sys_paths']:\n"
        "    if path_item and path_item not in sys.path:\n"
        "        sys.path.insert(0, path_item)\n"
        "shim_dir = CFG['shim_dir']\n"
        "if shim_dir and shim_dir not in sys.path and (Path(shim_dir) / 'npu_compare.py').is_file():\n"
        "    sys.path.insert(0, shim_dir)\n"
        "if not hasattr(torch, 'npu'):\n"
        "    try:\n"
        "        importlib.import_module('torch_npu')\n"
        "    except ImportError:\n"
        "        pass\n"
        "torch.npu.set_device(0)\n"
        "from tilefoundry_ascend.agent.campaign.anchor_specs import (\n"
        "    AnchorCaseSpec,\n"
        "    load_anchor_harness_and_cases,\n"
        "    load_cannbench_contract_module,\n"
        ")\n"
        "source_root = Path(CFG['source_root'])\n"
        "cannbench_repo = Path(CFG['cannbench_repo'])\n"
        "contract_mod = load_cannbench_contract_module(source_root=source_root)\n"
        "harness, cases = load_anchor_harness_and_cases(\n"
        "    CFG['operator_name'],\n"
        "    [AnchorCaseSpec('', int(CFG['case_num']), '', 1.0, 1.0)],\n"
        "    cannbench_repo=cannbench_repo,\n"
        "    source_root=source_root,\n"
        ")\n"
        "case = next(c for c in cases if int(c.case_num) == int(CFG['case_num']))\n"
        "spec = importlib.util.spec_from_file_location('_tf_msprof_cand', CFG['module_path'])\n"
        "if spec is None or spec.loader is None:\n"
        "    raise RuntimeError(f\"failed to load {CFG['module_path']}\")\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "sys.modules['_tf_msprof_cand'] = mod\n"
        "spec.loader.exec_module(mod)\n"
        "fn = getattr(mod, CFG['entrypoint'])\n"
        "inputs = harness.prepare(case)\n"
        "npu_inputs = contract_mod.map_tensors(inputs, lambda x: x.npu())\n"
        "params = harness.params(case, npu_inputs)\n"
        "fn(**params)\n"
        "torch.npu.synchronize()\n"
    )


def _run_msprof_op_subprocess(
    *,
    msprof_bin: str | None,
    output_dir: Path,
    aic_metrics: str,
    warmup: int,
    repeat: int,
    app_argv: Sequence[str],
    msprof_op_command: Sequence[str] | None = None,
    timeout_sec: int = 180,
) -> subprocess.CompletedProcess[str]:
    """Execute msprof op (or an explicit custom wrapper command) into output_dir."""
    if msprof_op_command is not None:
        cmd = [
            token.format(output_dir=str(output_dir), aic_metrics=aic_metrics)
            if ("{output_dir}" in token or "{aic_metrics}" in token)
            else token
            for token in msprof_op_command
        ]
    else:
        if not msprof_bin:
            raise FileNotFoundError("msprof binary not found")
        cmd = [
            msprof_bin,
            "op",
            f"--output={output_dir}",
            f"--aic-metrics={aic_metrics}",
            f"--warm-up={max(0, int(warmup))}",
            f"--launch-count={max(1, int(repeat))}",
            *app_argv,
        ]
    return subprocess.run(
        cmd,
        text=True,
        capture_output=True,
        timeout=max(10, int(timeout_sec)),
        check=False,
    )


def _parse_p2_from_report_dir(
    report_dir: Path,
    *,
    compiled_obj_paths: Sequence[Path | str] = (),
) -> dict[str, object]:
    """Parse P2 instruction traces from report_dir, falling back to static DWARF objdump."""
    trace_files = find_p2_trace_files(report_dir)
    pc_start_candidate = next(iter(sorted(report_dir.rglob("pc_start_addr.txt"))), None)
    for trace_file in trace_files:
        summary = parse_p2_instruction_trace(
            trace_file,
            pc_start_addr_source=pc_start_candidate,
        )
        if summary.total_instructions > 0:
            return _format_p2_summary_fields(summary, p2_source="msprof_p2_trace")
    if compiled_obj_paths:
        summary = extract_p2_from_objdump(compiled_obj_paths)
        if summary.total_instructions > 0:
            return _format_p2_summary_fields(summary, p2_source="llvm_objdump_dwarf")
    return {}


def _measure_msprof_op_metrics(
    *,
    operator_name: str = "",
    candidate_id: str = "",
    module_path: Path | str | None = None,
    entrypoint: str = "",
    case_num: int | None = None,
    source_root: Path | None = None,
    cannbench_repo: Path | None = None,
    warmup: int = 1,
    repeat: int = 3,
    msprof_op_command: Sequence[str] | None = None,
    case_runner_argv: Sequence[str] | None = None,
    compiled_obj_paths: Sequence[Path | str] = (),
) -> dict[str, object] | None:
    """Run msprof op on one target case and return parsed CSV and optional P2 telemetry."""
    if os.environ.get("TF_DISABLE_MSPROF_OP", "").strip() == "1":
        return None
    msprof_bin = shutil.which("msprof")
    if msprof_op_command is None and not msprof_bin:
        return None

    is_p2_deep_dive = str(candidate_id).startswith("bottleneck_p2_")
    has_runner = case_runner_argv is not None or (
        module_path is not None and bool(operator_name) and case_num is not None
    )
    if msprof_op_command is None and not has_runner:
        return None

    with tempfile.TemporaryDirectory(prefix="tf_msprof_op_") as tmp_dir_raw:
        tmp_dir = Path(tmp_dir_raw)
        output_dir = tmp_dir / "out"
        output_dir.mkdir(parents=True, exist_ok=True)
        if case_runner_argv is not None:
            app_argv = list(case_runner_argv)
        elif has_runner and module_path is not None and case_num is not None:
            eff_source_root = source_root or default_source_root()
            eff_cann_repo = cannbench_repo or _resolve_batch_cannbench_repo(eff_source_root)
            eff_entrypoint = entrypoint or get_operator_entrypoint(operator_name)
            driver_path = tmp_dir / "msprof_case_driver.py"
            driver_path.write_text(
                _build_msprof_op_case_driver_script(
                    operator_name=operator_name,
                    module_path=Path(module_path),
                    entrypoint=eff_entrypoint,
                    case_num=int(case_num),
                    source_root=eff_source_root,
                    cannbench_repo=eff_cann_repo,
                ),
                encoding="utf-8",
            )
            app_argv = [sys.executable, str(driver_path)]
        else:
            app_argv = []

        requested_metrics = _P2_AIC_METRICS if is_p2_deep_dive else _P1_AIC_METRICS
        try:
            proc = _run_msprof_op_subprocess(
                msprof_bin=msprof_bin,
                output_dir=output_dir,
                aic_metrics=requested_metrics,
                warmup=warmup,
                repeat=repeat,
                app_argv=app_argv,
                msprof_op_command=msprof_op_command,
                timeout_sec=120 if is_p2_deep_dive else 180,
            )
        except subprocess.TimeoutExpired:
            proc = subprocess.CompletedProcess(
                args=[],
                returncode=124,
                stdout="",
                stderr="msprof_op_timeout",
            )
        opprof_dirs = _locate_opprof_dirs(output_dir)
        if is_p2_deep_dive and (proc.returncode != 0 or not opprof_dirs):
            try:
                proc = _run_msprof_op_subprocess(
                    msprof_bin=msprof_bin,
                    output_dir=output_dir,
                    aic_metrics=_P1_AIC_METRICS,
                    warmup=warmup,
                    repeat=repeat,
                    app_argv=app_argv,
                    msprof_op_command=msprof_op_command,
                    timeout_sec=180,
                )
            except subprocess.TimeoutExpired:
                proc = subprocess.CompletedProcess(
                    args=[],
                    returncode=124,
                    stdout="",
                    stderr="msprof_op_timeout",
                )
            opprof_dirs = _locate_opprof_dirs(output_dir)

        merged_obj_paths: list[Path | str] = list(compiled_obj_paths)
        if is_p2_deep_dive and opprof_dirs:
            seen_objs = {str(Path(p)) for p in merged_obj_paths}
            for discovered in _discover_compiled_objects(None, None, opprof_dirs[-1]):
                if str(discovered) not in seen_objs:
                    seen_objs.add(str(discovered))
                    merged_obj_paths.append(discovered)

        telemetry: dict[str, object] = {}
        if opprof_dirs:
            try:
                report = parse_msprof_op_report(opprof_dirs[-1])
                if report.subkernels or report.parsed_tables:
                    telemetry.update(_aggregated_report_to_telemetry(report))
            except (FileNotFoundError, OSError, ValueError):
                pass
            if is_p2_deep_dive:
                telemetry.update(
                    _parse_p2_from_report_dir(
                        opprof_dirs[-1],
                        compiled_obj_paths=merged_obj_paths,
                    )
                )
        elif is_p2_deep_dive and merged_obj_paths:
            summary = extract_p2_from_objdump(merged_obj_paths)
            if summary.total_instructions > 0:
                telemetry.update(
                    _format_p2_summary_fields(summary, p2_source="llvm_objdump_dwarf")
                )

        return telemetry or None


def _measure_msprof_detail_metrics(
    fn_call=None,
    *,
    warmup: int = 1,
    repeat: int = 3,
    operator_name: str = "",
    candidate_id: str = "",
    module_path: Path | str | None = None,
    entrypoint: str = "",
    case_num: int | None = None,
    source_root: Path | None = None,
    cannbench_repo: Path | None = None,
    msprof_op_command: Sequence[str] | None = None,
    case_runner_argv: Sequence[str] | None = None,
    compiled_obj_paths: Sequence[Path | str] = (),
    enable_msprof_op: bool = True,
) -> dict[str, object]:
    """Measure NPU telemetry via msprof op CSVs first, falling back to tilelang.profiler."""
    is_p2_deep_dive = str(candidate_id).startswith("bottleneck_p2_")
    if enable_msprof_op:
        op_metrics = _measure_msprof_op_metrics(
            operator_name=operator_name,
            candidate_id=candidate_id,
            module_path=module_path,
            entrypoint=entrypoint,
            case_num=case_num,
            source_root=source_root,
            cannbench_repo=cannbench_repo,
            warmup=warmup,
            repeat=repeat,
            msprof_op_command=msprof_op_command,
            case_runner_argv=case_runner_argv,
            compiled_obj_paths=compiled_obj_paths,
        )
        if op_metrics:
            return op_metrics

    if fn_call is None:
        if is_p2_deep_dive and compiled_obj_paths:
            summary = extract_p2_from_objdump(compiled_obj_paths)
            if summary.total_instructions > 0:
                return _format_p2_summary_fields(summary, p2_source="llvm_objdump_dwarf")
        return {"error": "msprof_detail_failed: no_msprof_op_and_no_callable"}

    try:
        profiler = importlib.import_module("tilelang.profiler")
        prof = profiler.do_bench(
            fn_call,
            _n_warmup=max(1, int(warmup)),
            _n_repeat=max(1, int(repeat)),
            backend="msprof_detail",
            device=torch.device("npu:0"),
        )
        metrics: dict[str, object] = dict(_extract_msprof_metrics(prof))
        if is_p2_deep_dive and compiled_obj_paths:
            summary = extract_p2_from_objdump(compiled_obj_paths)
            if summary.total_instructions > 0:
                metrics.update(
                    _format_p2_summary_fields(summary, p2_source="llvm_objdump_dwarf")
                )
        return metrics
    except Exception as exc:
        if is_p2_deep_dive and compiled_obj_paths:
            summary = extract_p2_from_objdump(compiled_obj_paths)
            if summary.total_instructions > 0:
                return _format_p2_summary_fields(summary, p2_source="llvm_objdump_dwarf")
        return {"error": f"msprof_detail_failed: {type(exc).__name__}: {exc}"[:300]}


def _default_batch_request(source_root: Path, cannbench_repo: Path) -> dict[str, object]:
    batches = []
    for op_name in ("attention_residual", "mla"):
        mod_path = (source_root / get_operator_module_relpath(op_name)).resolve()
        batches.append(
            {
                "operator_name": op_name,
                "entrypoint": get_operator_entrypoint(op_name),
                "candidates": [
                    {"candidate_id": f"{op_name}_baseline", "module_path": str(mod_path)}
                ],
                "anchors": [a.to_dict() for a in get_default_anchors(op_name)],
            }
        )
    return {
        "source_root": str(source_root),
        "cannbench_repo": str(cannbench_repo),
        "warmup": 3,
        "repeat": 5,
        "require_all_passed": True,
        "response_path": "",
        "batches": batches,
    }


_INCUMBENT_CANDIDATE_IDS = frozenset({"incumbent_slot", "__incumbent__", "incumbent_baseline"})
_POISON_PILL_SKIP_INCUMBENT_IDS = frozenset({"incumbent_slot", "__incumbent__"})


_POISON_PILL_WORKER_SCRIPT = """import importlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import torch

from tilefoundry_ascend.agent.campaign.anchor_specs import (
    AnchorCaseSpec,
    load_anchor_harness_and_cases,
    load_cannbench_contract_module,
)


def _configure_tilelang_cache(tl_cache_dir: str) -> None:
    resolved = Path(tl_cache_dir).resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    os.environ["TILELANG_CACHE_DIR"] = str(resolved)
    os.environ["TILELANG_KERNEL_CACHE_USE_LIB_STAMP"] = "1"
    os.environ["TILELANG_DISABLE_CACHE"] = "0"
    tl_env_mod = importlib.import_module("tilelang.env")
    tl_env_mod.env.enable_cache()
    kc_mod = importlib.import_module("tilelang.cache.kernel_cache")
    kc_mod.KernelCache._get_cache_namespace.cache_clear()
    kc_mod.KernelCache._get_tilelang_lib_stamp.cache_clear()
    kc_mod.KernelCache._create_dirs()


def _load_candidate_entrypoint(module_path: Path, entrypoint: str, unique_tag: str):
    mod_name = f"_tf_surrogate_cand_{unique_tag}"
    spec = importlib.util.spec_from_file_location(mod_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load candidate module {module_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, entrypoint):
        raise AttributeError(f"{module_path} does not define {entrypoint!r}")
    return getattr(mod, entrypoint), mod

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
out_path = Path(sys.argv[2])
op_name = str(payload["operator_name"])
entrypoint = str(payload["entrypoint"])
cid = str(payload["candidate_id"])
mod_path = Path(str(payload["module_path"]))
source_root = Path(str(payload["source_root"]))
cannbench_repo = Path(str(payload["cannbench_repo"]))
anchors = [AnchorCaseSpec.from_dict(item) for item in payload["anchors"]]

if not hasattr(torch, "npu"):
    try:
        importlib.import_module("torch_npu")
    except ImportError:
        pass
if hasattr(torch, "npu") and torch.npu.is_available():
    torch.npu.set_device(0)

tl_cache_env = os.environ.get("TILELANG_CACHE_DIR", "").strip()
if tl_cache_env:
    _configure_tilelang_cache(tl_cache_env)

contract_mod = load_cannbench_contract_module(source_root=source_root)
harness, cases = load_anchor_harness_and_cases(
    op_name,
    anchors,
    cannbench_repo=cannbench_repo,
    source_root=source_root,
)
case_by_num = {int(c.case_num): c for c in cases}
fn, mod_obj = _load_candidate_entrypoint(
    mod_path, entrypoint, f"pill_{op_name}_{int(time.monotonic_ns())}"
)

case_results = {}
all_passed = True
first_error = ""

with contract_mod.ReferenceCache(harness) as cache:
    for spec in anchors:
        cnum = int(spec.case_num)
        case = case_by_num[cnum]

        def _prep(c=case, h=harness):
            ts = h.prepare(c)
            return {"inputs": ts, "oracle": h.reference_output(c, ts)}

        bundle = cache.get(case, "oracle", _prep)
        tensors = bundle["inputs"]
        oracle = bundle["oracle"]

        def _native_fn(c=case, ts=tensors, h=harness, ch=cache):
            return ch.get(c, "native", lambda: h.reference_output(c, ts, native=True))

        npu_tensors = None
        params = None
        actual = None
        try:
            npu_tensors = contract_mod.map_tensors(tensors, lambda x: x.npu())
            params = harness.params(case, npu_tensors)
            actual = fn(**params)
            if hasattr(torch, "npu") and hasattr(torch.npu, "synchronize"):
                torch.npu.synchronize()
            passed, details = harness.compare(case, actual, oracle, tensors, _native_fn)
            if passed:
                case_results[str(cnum)] = {"passed": True, "error": ""}
            else:
                err_msg = json.dumps(details)[:300]
                case_results[str(cnum)] = {"passed": False, "error": err_msg}
                all_passed = False
                if not first_error:
                    first_error = err_msg
        except Exception as exc:
            err_msg = f"{type(exc).__name__}: {exc}"[:300]
            case_results[str(cnum)] = {"passed": False, "error": err_msg}
            all_passed = False
            if not first_error:
                first_error = err_msg
        finally:
            actual = None
            params = None
            npu_tensors = None
            if hasattr(torch, "npu") and hasattr(torch.npu, "empty_cache"):
                torch.npu.empty_cache()

cache_obj = getattr(mod_obj, "_KERNEL_CACHE", None)
if isinstance(cache_obj, dict):
    cache_obj.clear()

tmp_out = out_path.with_suffix(".tmp")
tmp_out.write_text(
    json.dumps({"all_passed": all_passed, "error": first_error, "cases": case_results}),
    encoding="utf-8",
)
os.replace(tmp_out, out_path)
"""


_POISON_PILL_BATCH_WORKER_SCRIPT = """import importlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import torch

from tilefoundry_ascend.agent.campaign.anchor_specs import (
    AnchorCaseSpec,
    load_anchor_harness_and_cases,
    load_cannbench_contract_module,
)


def _configure_tilelang_cache(tl_cache_dir: str) -> None:
    resolved = Path(tl_cache_dir).resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    os.environ["TILELANG_CACHE_DIR"] = str(resolved)
    os.environ["TILELANG_KERNEL_CACHE_USE_LIB_STAMP"] = "1"
    os.environ["TILELANG_DISABLE_CACHE"] = "0"
    tl_env_mod = importlib.import_module("tilelang.env")
    tl_env_mod.env.enable_cache()
    kc_mod = importlib.import_module("tilelang.cache.kernel_cache")
    kc_mod.KernelCache._get_cache_namespace.cache_clear()
    kc_mod.KernelCache._get_tilelang_lib_stamp.cache_clear()
    kc_mod.KernelCache._create_dirs()


def _load_candidate_entrypoint(module_path: Path, entrypoint: str, unique_tag: str):
    mod_name = f"_tf_surrogate_cand_{unique_tag}"
    spec = importlib.util.spec_from_file_location(mod_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load candidate module {module_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, entrypoint):
        raise AttributeError(f"{module_path} does not define {entrypoint!r}")
    return getattr(mod, entrypoint), mod


payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
out_path = Path(sys.argv[2])
op_name = str(payload["operator_name"])
entrypoint = str(payload["entrypoint"])
candidates_raw = list(payload["candidates"])
source_root = Path(str(payload["source_root"]))
cannbench_repo = Path(str(payload["cannbench_repo"]))
anchors = [AnchorCaseSpec.from_dict(item) for item in payload["anchors"]]

if not hasattr(torch, "npu"):
    try:
        importlib.import_module("torch_npu")
    except ImportError:
        pass
if hasattr(torch, "npu") and torch.npu.is_available():
    torch.npu.set_device(0)

tl_cache_env = os.environ.get("TILELANG_CACHE_DIR", "").strip()
if tl_cache_env:
    _configure_tilelang_cache(tl_cache_env)

contract_mod = load_cannbench_contract_module(source_root=source_root)
harness, cases = load_anchor_harness_and_cases(
    op_name,
    anchors,
    cannbench_repo=cannbench_repo,
    source_root=source_root,
)
case_by_num = {int(c.case_num): c for c in cases}

results = {}
loaded_cands = []
for idx, item in enumerate(candidates_raw):
    cid = str(item["candidate_id"])
    mod_path = Path(str(item["module_path"]))
    results[cid] = {"all_passed": True, "error": "", "cases": {}}
    try:
        fn, mod_obj = _load_candidate_entrypoint(
            mod_path,
            entrypoint,
            f"bpill_{op_name}_{idx}_{int(time.monotonic_ns())}",
        )
        loaded_cands.append((cid, fn, mod_obj))
    except Exception as exc:
        err_msg = f"{type(exc).__name__}: {exc}"[:300]
        results[cid]["all_passed"] = False
        results[cid]["error"] = err_msg
        for spec in anchors:
            results[cid]["cases"][str(int(spec.case_num))] = {
                "passed": False,
                "error": err_msg,
            }

try:
    with contract_mod.ReferenceCache(harness) as cache:
        for spec in anchors:
            cnum = int(spec.case_num)
            case = case_by_num[cnum]

            def _prep(c=case, h=harness):
                ts = h.prepare(c)
                return {"inputs": ts, "oracle": h.reference_output(c, ts)}

            bundle = cache.get(case, "oracle", _prep)
            tensors = bundle["inputs"]
            oracle = bundle["oracle"]

            def _native_fn(c=case, ts=tensors, h=harness, ch=cache):
                return ch.get(c, "native", lambda: h.reference_output(c, ts, native=True))

            npu_tensors = None
            try:
                npu_tensors = contract_mod.map_tensors(tensors, lambda x: x.npu())
                for cid, fn, _mod_obj in loaded_cands:
                    params = None
                    actual = None
                    try:
                        params = harness.params(case, npu_tensors)
                        actual = fn(**params)
                        if hasattr(torch, "npu") and hasattr(torch.npu, "synchronize"):
                            torch.npu.synchronize()
                        passed, details = harness.compare(
                            case, actual, oracle, tensors, _native_fn
                        )
                        if passed:
                            results[cid]["cases"][str(cnum)] = {"passed": True, "error": ""}
                        else:
                            err_msg = json.dumps(details)[:300]
                            results[cid]["cases"][str(cnum)] = {
                                "passed": False,
                                "error": err_msg,
                            }
                            results[cid]["all_passed"] = False
                            if not results[cid]["error"]:
                                results[cid]["error"] = err_msg
                    except Exception as exc:
                        err_msg = f"{type(exc).__name__}: {exc}"[:300]
                        results[cid]["cases"][str(cnum)] = {
                            "passed": False,
                            "error": err_msg,
                        }
                        results[cid]["all_passed"] = False
                        if not results[cid]["error"]:
                            results[cid]["error"] = err_msg
                    finally:
                        actual = None
                        params = None
                        if hasattr(torch, "npu") and hasattr(torch.npu, "empty_cache"):
                            torch.npu.empty_cache()
            finally:
                npu_tensors = None
                if hasattr(torch, "npu") and hasattr(torch.npu, "empty_cache"):
                    torch.npu.empty_cache()
finally:
    for _cid, _fn, mod_obj in loaded_cands:
        cache_obj = getattr(mod_obj, "_KERNEL_CACHE", None)
        if isinstance(cache_obj, dict):
            cache_obj.clear()
    if hasattr(torch, "npu") and hasattr(torch.npu, "empty_cache"):
        torch.npu.empty_cache()

tmp_out = out_path.with_suffix(".tmp")
tmp_out.write_text(json.dumps({"candidates": results}), encoding="utf-8")
os.replace(tmp_out, out_path)
"""


def _build_poison_pill_worker_env(source_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    pp_entries: list[str] = []
    for candidate_dir in (
        str(source_root),
        str(source_root / "src"),
        str(source_root / "targets" / "ascend"),
    ):
        if candidate_dir not in pp_entries:
            pp_entries.append(candidate_dir)
    for entry in sys.path:
        if entry and (Path(entry) / "npu_compare.py").is_file() and entry not in pp_entries:
            pp_entries.insert(0, entry)
    existing_pp = env.get("PYTHONPATH", "").strip()
    if existing_pp:
        for item in existing_pp.split(os.pathsep):
            if item and item not in pp_entries:
                pp_entries.append(item)
    env["PYTHONPATH"] = os.pathsep.join(pp_entries)
    return env


def _screen_candidate_isolated(
    *,
    operator_name: str,
    entrypoint: str,
    candidate_id: str,
    module_path: Path,
    anchors: Sequence[AnchorCaseSpec],
    source_root: Path,
    cannbench_repo: Path,
    timeout_sec: float = 30.0,
) -> tuple[bool, str, dict[int, tuple[bool, str]]]:
    """Run one-shot accuracy verification for a candidate in an isolated subprocess."""
    eff_timeout = max(0.05, float(timeout_sec))
    worker_env = _build_poison_pill_worker_env(source_root)
    with tempfile.TemporaryDirectory(prefix="tf_poison_pill_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        req_path = tmp_root / "req.json"
        out_path = tmp_root / "out.json"
        script_path = tmp_root / "worker.py"
        req_path.write_text(
            json.dumps(
                {
                    "operator_name": operator_name,
                    "entrypoint": entrypoint,
                    "candidate_id": candidate_id,
                    "module_path": str(module_path),
                    "source_root": str(source_root),
                    "cannbench_repo": str(cannbench_repo),
                    "anchors": [a.to_dict() for a in anchors],
                }
            ),
            encoding="utf-8",
        )
        script_path.write_text(_POISON_PILL_WORKER_SCRIPT, encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, str(script_path), str(req_path), str(out_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(source_root),
            env=worker_env,
            start_new_session=True,
        )
        try:
            stdout, stderr = proc.communicate(timeout=eff_timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                proc.kill()
            proc.communicate()
            return (
                False,
                f"poison_pill_timeout_or_crash: timeout after {eff_timeout:g}s",
                {},
            )

        if proc.returncode != 0:
            detail = (stderr or stdout or "").strip()[-300:]
            msg = f"poison_pill_timeout_or_crash: exit_code={proc.returncode}"
            if detail:
                msg = f"{msg} ({detail})"
            return False, msg[:300], {}

        if not out_path.is_file():
            return False, "poison_pill_timeout_or_crash: missing worker output", {}
        try:
            raw = json.loads(out_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            return False, f"poison_pill_timeout_or_crash: invalid worker json: {exc}"[:300], {}

        per_case = _parse_poison_pill_cases(raw.get("cases"))
        all_passed = bool(raw.get("all_passed", False))
        err_msg = str(raw.get("error", ""))
        return all_passed, err_msg[:300], per_case


def _parse_poison_pill_cases(raw_cases: object) -> dict[int, tuple[bool, str]]:
    per_case: dict[int, tuple[bool, str]] = {}
    if isinstance(raw_cases, Mapping):
        for k, row in raw_cases.items():
            if isinstance(row, Mapping):
                per_case[int(k)] = (bool(row.get("passed", False)), str(row.get("error", "")))
    return per_case


def _screen_candidates_isolated_batch(
    *,
    operator_name: str,
    entrypoint: str,
    candidates: Sequence[tuple[str, Path]],
    anchors: Sequence[AnchorCaseSpec],
    source_root: Path,
    cannbench_repo: Path,
    timeout_sec: float = 30.0,
) -> tuple[bool, dict[str, tuple[bool, str, dict[int, tuple[bool, str]]]]]:
    """Run one-shot accuracy verification for multiple candidates in one isolated subprocess."""
    if not candidates:
        return True, {}
    eff_timeout = max(0.05, float(timeout_sec))
    total_timeout = max(eff_timeout, min(eff_timeout * max(1, len(candidates)), 120.0))
    worker_env = _build_poison_pill_worker_env(source_root)
    with tempfile.TemporaryDirectory(prefix="tf_poison_pill_batch_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        req_path = tmp_root / "req.json"
        out_path = tmp_root / "out.json"
        script_path = tmp_root / "worker.py"
        req_path.write_text(
            json.dumps(
                {
                    "operator_name": operator_name,
                    "entrypoint": entrypoint,
                    "candidates": [
                        {"candidate_id": cid, "module_path": str(mod_path)}
                        for cid, mod_path in candidates
                    ],
                    "source_root": str(source_root),
                    "cannbench_repo": str(cannbench_repo),
                    "anchors": [a.to_dict() for a in anchors],
                }
            ),
            encoding="utf-8",
        )
        script_path.write_text(_POISON_PILL_BATCH_WORKER_SCRIPT, encoding="utf-8")
        proc = subprocess.Popen(
            [sys.executable, str(script_path), str(req_path), str(out_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=str(source_root),
            env=worker_env,
            start_new_session=True,
        )
        try:
            proc.communicate(timeout=total_timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except OSError:
                proc.kill()
            proc.communicate()
            return False, {}

        if proc.returncode != 0 or not out_path.is_file():
            return False, {}
        try:
            raw = json.loads(out_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return False, {}

        if not isinstance(raw, Mapping):
            return False, {}
        raw_cands = raw.get("candidates")
        if not isinstance(raw_cands, Mapping):
            return False, {}

        parsed_map: dict[str, tuple[bool, str, dict[int, tuple[bool, str]]]] = {}
        for cid, _mod_path in candidates:
            entry = raw_cands.get(cid)
            if not isinstance(entry, Mapping):
                return False, {}
            try:
                per_case = _parse_poison_pill_cases(entry.get("cases"))
            except (TypeError, ValueError):
                return False, {}
            all_passed = bool(entry.get("all_passed", False))
            err_msg = str(entry.get("error", ""))[:300]
            parsed_map[cid] = (all_passed, err_msg, per_case)
        return True, parsed_map


_ORIG_LOAD_CANDIDATE_ENTRYPOINT = _load_candidate_entrypoint
_ORIG_LOAD_HARNESS_AND_CASES = load_anchor_harness_and_cases
_ORIG_SCREEN_CANDIDATE_ISOLATED = _screen_candidate_isolated


def _evaluate_operator_batch(
    batch: dict[str, object],
    *,
    source_root: Path,
    cannbench_repo: Path,
    warmup: int,
    repeat: int,
    collect_msprof_detail: bool = False,
    msprof_out: dict[str, dict[str, dict[str, object]]] | None = None,
    poison_pill_isolation: bool | None = None,
    poison_pill_timeout_sec: float = 30.0,
) -> dict[str, CandidateSurrogateResult]:
    op_name = str(batch["operator_name"])
    entrypoint = str(batch.get("entrypoint") or get_operator_entrypoint(op_name))
    anchors = [AnchorCaseSpec.from_dict(item) for item in batch["anchors"]]
    candidates = list(batch["candidates"])
    raw_msprof_cases = batch.get("msprof_case_nums")
    if raw_msprof_cases is not None:
        msprof_case_set = {int(c) for c in raw_msprof_cases}
    elif collect_msprof_detail:
        msprof_case_set = {int(a.case_num) for a in anchors}
    else:
        msprof_case_set = set()
    msprof_warmup = int(batch.get("msprof_warmup", 1))
    msprof_repeat = int(batch.get("msprof_repeat", 3))
    eff_pill_timeout = float(batch.get("poison_pill_timeout_sec", poison_pill_timeout_sec))

    contract_mod = load_cannbench_contract_module(source_root=source_root)
    harness, cases = load_anchor_harness_and_cases(
        op_name,
        anchors,
        cannbench_repo=cannbench_repo,
        source_root=source_root,
    )
    case_by_num = {int(c.case_num): c for c in cases}

    has_npu_event = hasattr(getattr(torch, "npu", None), "Event")
    non_incumbents = [
        cand for cand in candidates if str(cand["candidate_id"]) not in _INCUMBENT_CANDIDATE_IDS
    ]
    if poison_pill_isolation is not None:
        use_poison_pill = bool(poison_pill_isolation) and bool(non_incumbents)
    else:
        use_poison_pill = (
            len(candidates) >= 2
            and bool(non_incumbents)
            and (
                (_screen_candidate_isolated is not _ORIG_SCREEN_CANDIDATE_ISOLATED)
                or (
                    has_npu_event
                    and _load_candidate_entrypoint is _ORIG_LOAD_CANDIDATE_ENTRYPOINT
                    and load_anchor_harness_and_cases is _ORIG_LOAD_HARNESS_AND_CASES
                )
            )
        )

    per_cand_metrics: dict[str, list[AnchorEvalMetrics]] = {
        str(cand["candidate_id"]): [] for cand in candidates
    }
    phase_a_passed_ids: set[str] = set()
    loaded_cands: list[tuple[str, object, object]] = []
    cand_paths: dict[str, Path] = {}

    if use_poison_pill:
        with contract_mod.ReferenceCache(harness) as warm_cache:
            cache_folder = getattr(warm_cache, "folder", None)
            for spec in anchors:
                case = case_by_num[int(spec.case_num)]
                oracle_file: Path | None = None
                if cache_folder is not None:
                    get_case_id = getattr(case, "get_case_id_str", None)
                    case_id_str = get_case_id() if callable(get_case_id) else None
                    if isinstance(case_id_str, str):
                        digest = hashlib.sha256(case_id_str.encode()).hexdigest()
                        oracle_file = Path(cache_folder) / f"{digest}.oracle.pt"
                if (
                    oracle_file is None
                    or not oracle_file.is_file()
                    or oracle_file.stat().st_size <= 0
                ):
                    def _warm_prep(c=case, h=harness):
                        ts = h.prepare(c)
                        return {"inputs": ts, "oracle": h.reference_output(c, ts)}

                    warm_cache.get(case, "oracle", _warm_prep)

        non_incumbent_pairs = [
            (str(cand["candidate_id"]), Path(str(cand["module_path"])))
            for cand in candidates
            if str(cand["candidate_id"]) not in _INCUMBENT_CANDIDATE_IDS
        ]
        batch_ok = False
        batch_results: dict[str, tuple[bool, str, dict[int, tuple[bool, str]]]] = {}
        if (
            len(non_incumbent_pairs) >= 2
            and _screen_candidate_isolated is _ORIG_SCREEN_CANDIDATE_ISOLATED
        ):
            batch_ok, batch_results = _screen_candidates_isolated_batch(
                operator_name=op_name,
                entrypoint=entrypoint,
                candidates=non_incumbent_pairs,
                anchors=anchors,
                source_root=source_root,
                cannbench_repo=cannbench_repo,
                timeout_sec=eff_pill_timeout,
            )

        for cid, mod_path in non_incumbent_pairs:
            if batch_ok:
                passed_iso, batch_err, per_case = batch_results[cid]
            else:
                passed_iso, batch_err, per_case = _screen_candidate_isolated(
                    operator_name=op_name,
                    entrypoint=entrypoint,
                    candidate_id=cid,
                    module_path=mod_path,
                    anchors=anchors,
                    source_root=source_root,
                    cannbench_repo=cannbench_repo,
                    timeout_sec=eff_pill_timeout,
                )
            if passed_iso:
                phase_a_passed_ids.add(cid)
            else:
                fallback_err = batch_err or "poison_pill_timeout_or_crash"
                for spec in anchors:
                    case_entry = per_case.get(int(spec.case_num))
                    case_err = (
                        case_entry[1]
                        if case_entry is not None and not case_entry[0] and case_entry[1]
                        else fallback_err
                    )
                    per_cand_metrics[cid].append(
                        build_anchor_metrics(
                            spec,
                            accuracy_passed=False,
                            cand_us=0.0,
                            error_message=case_err[:300],
                        )
                    )

        for idx, cand in enumerate(candidates):
            cid = str(cand["candidate_id"])
            if cid not in _INCUMBENT_CANDIDATE_IDS and cid not in phase_a_passed_ids:
                continue
            mod_path = Path(str(cand["module_path"]))
            tag = f"{op_name}_{idx}_{int(time.monotonic_ns())}"
            fn, mod_obj = _load_candidate_entrypoint(mod_path, entrypoint, tag)
            loaded_cands.append((cid, fn, mod_obj))
            cand_paths[cid] = mod_path
    else:
        for idx, cand in enumerate(candidates):
            cid = str(cand["candidate_id"])
            mod_path = Path(str(cand["module_path"]))
            tag = f"{op_name}_{idx}_{int(time.monotonic_ns())}"
            fn, mod_obj = _load_candidate_entrypoint(mod_path, entrypoint, tag)
            loaded_cands.append((cid, fn, mod_obj))
            cand_paths[cid] = mod_path

    with contract_mod.ReferenceCache(harness) as cache:
        for spec in anchors:
            case = case_by_num[int(spec.case_num)]

            def _prep(c=case, h=harness):
                ts = h.prepare(c)
                return {"inputs": ts, "oracle": h.reference_output(c, ts)}

            bundle = cache.get(case, "oracle", _prep)
            tensors = bundle["inputs"]
            oracle = bundle["oracle"]

            def _native_fn(c=case, ts=tensors, h=harness, ch=cache):
                return ch.get(c, "native", lambda: h.reference_output(c, ts, native=True))

            paired_interleaved = bool(batch.get("paired_interleaved", True))
            passed_cands: list[tuple[int, str, object, object]] = []
            metrics_by_cid: dict[str, AnchorEvalMetrics] = {}

            for orig_idx, (cid, fn, mod_obj) in enumerate(loaded_cands):
                if use_poison_pill and (
                    cid in phase_a_passed_ids or cid in _POISON_PILL_SKIP_INCUMBENT_IDS
                ):
                    passed_cands.append((orig_idx, cid, fn, mod_obj))
                    continue
                npu_tensors = None
                params = None
                actual = None
                try:
                    npu_tensors = contract_mod.map_tensors(tensors, lambda x: x.npu())
                    params = harness.params(case, npu_tensors)
                    actual = fn(**params)
                    torch.npu.synchronize()
                    passed, details = harness.compare(
                        case, actual, oracle, tensors, _native_fn
                    )
                    if passed:
                        passed_cands.append((orig_idx, cid, fn, mod_obj))
                    else:
                        metrics_by_cid[cid] = build_anchor_metrics(
                            spec,
                            accuracy_passed=False,
                            cand_us=0.0,
                            error_message=json.dumps(details)[:300],
                        )
                except Exception as exc:
                    metrics_by_cid[cid] = build_anchor_metrics(
                        spec,
                        accuracy_passed=False,
                        cand_us=0.0,
                        error_message=f"{type(exc).__name__}: {exc}"[:300],
                    )
                finally:
                    actual = None
                    params = None
                    npu_tensors = None
                    torch.npu.empty_cache()

            if passed_cands:
                base_npu_tensors = None
                base_params = None
                try:
                    base_npu_tensors = contract_mod.map_tensors(tensors, lambda x: x.npu())
                    base_params = harness.params(case, base_npu_tensors)
                    incumbent_passed = bool(
                        loaded_cands
                        and loaded_cands[0][0] in {"incumbent_slot", "__incumbent__"}
                        and passed_cands[0][0] == 0
                    )
                    if len(passed_cands) >= 2 and has_npu_event and paired_interleaved:
                        paired_stats = measure_paired_interleaved_micro_burst(
                            [(cid, fn) for _orig_idx, cid, fn, _mod in passed_cands],
                            base_params,
                            warmup=warmup,
                            repeat=repeat,
                            t_hw_us=spec.t_hw_us,
                            max_vram_bytes=512 * 1024 * 1024,
                        )
                        for pos, (_orig_idx, cid, _fn, _mod) in enumerate(passed_cands):
                            cand_us, ring_burst_k, raw_cv, med_r, rel_mad = paired_stats[cid]
                            if incumbent_passed:
                                if pos == 0:
                                    paired_speedup = 1.0
                                    cand_rel_mad = 0.0
                                    burst_cv = raw_cv
                                else:
                                    paired_speedup = med_r
                                    cand_rel_mad = rel_mad
                                    burst_cv = rel_mad
                            else:
                                paired_speedup = 0.0
                                cand_rel_mad = rel_mad if pos > 0 else 0.0
                                burst_cv = raw_cv
                            metrics_by_cid[cid] = build_anchor_metrics(
                                spec,
                                accuracy_passed=True,
                                cand_us=cand_us,
                                ring_burst_k=ring_burst_k,
                                burst_batch_cv=burst_cv,
                                spike_retries=0,
                                paired_speedup_vs_incumbent=paired_speedup,
                                paired_rel_mad=cand_rel_mad,
                            )
                    else:
                        for _orig_idx, cid, fn, _mod in passed_cands:
                            ring_burst_k = 1
                            burst_batch_cv = 0.0
                            spike_retries = 0
                            if has_npu_event and 0.0 < float(spec.t_hw_us) < 5.0:
                                cand_us, ring_burst_k, burst_batch_cv, spike_retries = (
                                    measure_npu_ring_burst_stats(
                                        fn,
                                        base_params,
                                        warmup=warmup,
                                        repeat=repeat,
                                        t_hw_us=spec.t_hw_us,
                                        max_vram_bytes=512 * 1024 * 1024,
                                    )
                                )
                            elif has_npu_event:
                                cand_us, burst_batch_cv = _measure_npu_latency_stats_us(
                                    lambda f=fn, p=base_params: f(**p),
                                    warmup=warmup,
                                    repeat=repeat,
                                )
                                if 0.0 < cand_us < 8.0:
                                    cand_us, ring_burst_k, burst_batch_cv, spike_retries = (
                                        measure_npu_ring_burst_stats(
                                            fn,
                                            base_params,
                                            warmup=warmup,
                                            repeat=repeat,
                                            t_hw_us=spec.t_hw_us,
                                            first_probe_us=cand_us,
                                            max_vram_bytes=512 * 1024 * 1024,
                                        )
                                    )
                            else:
                                cand_us = _measure_npu_latency_us(
                                    lambda f=fn, p=base_params: f(**p),
                                    warmup=warmup,
                                    repeat=repeat,
                                )
                            metrics_by_cid[cid] = build_anchor_metrics(
                                spec,
                                accuracy_passed=True,
                                cand_us=cand_us,
                                ring_burst_k=ring_burst_k,
                                burst_batch_cv=burst_batch_cv,
                                spike_retries=spike_retries,
                            )

                    if msprof_out is not None and int(spec.case_num) in msprof_case_set:
                        for _orig_idx, cid, fn, mod_obj in passed_cands:
                            msprof_out.setdefault(cid, {})[str(int(spec.case_num))] = (
                                _measure_msprof_detail_metrics(
                                    lambda f=fn, p=base_params: f(**p),
                                    warmup=msprof_warmup,
                                    repeat=msprof_repeat,
                                    operator_name=op_name,
                                    candidate_id=cid,
                                    module_path=cand_paths[cid],
                                    entrypoint=entrypoint,
                                    case_num=int(spec.case_num),
                                    source_root=source_root,
                                    cannbench_repo=cannbench_repo,
                                    compiled_obj_paths=_discover_compiled_objects(
                                        mod_obj, cand_paths[cid]
                                    ),
                                )
                            )
                except Exception as exc:
                    for _orig_idx, cid, _fn, _mod in passed_cands:
                        if cid not in metrics_by_cid:
                            metrics_by_cid[cid] = build_anchor_metrics(
                                spec,
                                accuracy_passed=False,
                                cand_us=0.0,
                                error_message=f"{type(exc).__name__}: {exc}"[:300],
                            )
                finally:
                    base_params = None
                    base_npu_tensors = None
                    torch.npu.empty_cache()

            for cid, _fn, _mod in loaded_cands:
                per_cand_metrics[cid].append(metrics_by_cid[cid])

    for _cid, _fn, mod_obj in loaded_cands:
        cache_obj = getattr(mod_obj, "_KERNEL_CACHE", None)
        if isinstance(cache_obj, dict):
            cache_obj.clear()
    gc.collect()
    torch.npu.empty_cache()

    return {
        str(cand["candidate_id"]): aggregate_surrogate_result(
            str(cand["candidate_id"]),
            op_name,
            per_cand_metrics[str(cand["candidate_id"])],
        )
        for cand in candidates
    }


def _collect_msprof_targets(
    targets: list[Mapping[str, object]],
    *,
    source_root: Path,
    cannbench_repo: Path,
) -> dict[str, dict[str, dict[str, dict[str, object]]]]:
    by_op: dict[str, list[Mapping[str, object]]] = {}
    for row in targets:
        if isinstance(row, Mapping):
            op_name = str(row["operator_name"])
            by_op.setdefault(op_name, []).append(row)

    telemetry_out: dict[str, dict[str, dict[str, dict[str, object]]]] = {}
    contract_mod = load_cannbench_contract_module(source_root=source_root)
    for op_name, rows in by_op.items():
        entrypoint = get_operator_entrypoint(op_name)
        case_nums = sorted({int(r["case_num"]) for r in rows})
        specs = [AnchorCaseSpec("", cnum, "", 1.0, 1.0) for cnum in case_nums]
        harness, cases = load_anchor_harness_and_cases(
            op_name,
            specs,
            cannbench_repo=cannbench_repo,
            source_root=source_root,
        )
        case_by_num = {int(c.case_num): c for c in cases}

        loaded_mods: dict[tuple[str, str], tuple[object, object]] = {}
        for idx, row in enumerate(rows):
            cid = str(row["candidate_id"])
            mod_path = str(Path(str(row["module_path"])).resolve())
            key = (cid, mod_path)
            if key not in loaded_mods:
                ep = str(row.get("entrypoint") or entrypoint)
                tag = f"msprof_{op_name}_{idx}_{int(time.monotonic_ns())}"
                loaded_mods[key] = _load_candidate_entrypoint(Path(mod_path), ep, tag)

        with contract_mod.ReferenceCache(harness) as cache:
            for row in rows:
                cid = str(row["candidate_id"])
                mod_path = str(Path(str(row["module_path"])).resolve())
                ep = str(row.get("entrypoint") or entrypoint)
                cnum = int(row["case_num"])
                fn, mod_obj = loaded_mods[(cid, mod_path)]
                case = case_by_num[cnum]

                def _prep(c=case, h=harness):
                    ts = h.prepare(c)
                    return {"inputs": ts, "oracle": h.reference_output(c, ts)}

                bundle = cache.get(case, "oracle", _prep)
                tensors = bundle["inputs"]
                npu_tensors = None
                params = None
                try:
                    npu_tensors = contract_mod.map_tensors(tensors, lambda x: x.npu())
                    params = harness.params(case, npu_tensors)
                    metrics = _measure_msprof_detail_metrics(
                        lambda f=fn, p=params: f(**p),
                        warmup=int(row.get("warmup", 1)),
                        repeat=int(row.get("repeat", 3)),
                        operator_name=op_name,
                        candidate_id=cid,
                        module_path=Path(mod_path),
                        entrypoint=ep,
                        case_num=cnum,
                        source_root=source_root,
                        cannbench_repo=cannbench_repo,
                        compiled_obj_paths=_discover_compiled_objects(
                            mod_obj, Path(mod_path)
                        ),
                    )
                except Exception as exc:
                    metrics = {"error": f"msprof_target_failed: {type(exc).__name__}: {exc}"[:300]}
                finally:
                    params = None
                    npu_tensors = None
                    torch.npu.empty_cache()
                telemetry_out.setdefault(op_name, {}).setdefault(cid, {})[str(cnum)] = metrics

        for _fn, mod_obj in loaded_mods.values():
            cache_obj = getattr(mod_obj, "_KERNEL_CACHE", None)
            if isinstance(cache_obj, dict):
                cache_obj.clear()
        gc.collect()
        torch.npu.empty_cache()
    return telemetry_out


_FAST_SHA256 = re.compile(r"[0-9a-f]{64}")


class _FastRequestError(RuntimeError):
    """A fast request contract violation reported as a request_error response."""


@dataclass(frozen=True)
class _FastCandidate:
    operator_name: str
    candidate_id: str
    module_path: Path
    relative_path: str
    sha256: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fast_int(value: object, field: str, *, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise _FastRequestError(f"fast request field {field!r} must be a positive integer")
    return value


def _fast_source_root(source_root_raw: str) -> Path:
    raw = source_root_raw.strip()
    if not raw:
        raise RuntimeError(
            f"fast request requires {FAST_SOURCE_ROOT_ENV} from the AgentGates adapter"
        )
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise RuntimeError(f"fast request environment {FAST_SOURCE_ROOT_ENV} must be absolute")
    resolved = path.resolve()
    if not resolved.is_dir():
        raise RuntimeError(
            f"fast request environment {FAST_SOURCE_ROOT_ENV} must be a directory: {resolved}"
        )
    return resolved


def _fast_env_dir(name: str, *, create: bool) -> Path:
    raw = os.environ.get(name, "").strip()
    if not raw:
        raise RuntimeError(f"fast request requires {name} from the AgentGates adapter")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise RuntimeError(f"fast request environment {name} must be absolute")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    elif not path.is_dir():
        raise RuntimeError(f"fast request environment {name} must be a directory: {path}")
    return path.resolve()


def _fast_path(source_root: Path, value: object, field: str) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise _FastRequestError(f"fast request field {field!r} is required")
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise _FastRequestError(f"fast request field {field!r} must be workspace-relative")
    unresolved = source_root / relative
    if unresolved.is_symlink():
        raise _FastRequestError(f"fast request field {field!r} must not be a symlink")
    try:
        resolved = unresolved.resolve()
        resolved.relative_to(source_root)
    except ValueError:
        raise _FastRequestError(
            f"fast request field {field!r} escapes the source root"
        ) from None
    return resolved


def _fast_candidate(
    source_root: Path,
    operator_name: str,
    row: Mapping[str, object],
    used: set[tuple[str, str]],
) -> _FastCandidate:
    candidate_id = str(row.get("candidate_id", "")).strip()
    if not candidate_id:
        raise _FastRequestError(
            f"fast request candidate under {operator_name!r} has no candidate_id"
        )
    key = (operator_name, candidate_id)
    if key in used:
        raise _FastRequestError(
            f"duplicate fast request candidate {operator_name}/{candidate_id}"
        )
    used.add(key)
    relative_path = str(row.get("module_path", "")).strip()
    module_path = _fast_path(source_root, relative_path, "module_path")
    if not module_path.is_file():
        raise _FastRequestError(
            f"fast request module {relative_path!r} is not a regular file"
        )
    sha256 = str(row.get("sha256", "")).strip().lower()
    if not _FAST_SHA256.fullmatch(sha256):
        raise _FastRequestError(
            f"fast request module {relative_path!r} has no valid sha256"
        )
    if _sha256_file(module_path) != sha256:
        raise _FastRequestError(
            f"fast request module {relative_path!r} does not match its declared sha256"
        )
    return _FastCandidate(
        operator_name=operator_name,
        candidate_id=candidate_id,
        module_path=module_path,
        relative_path=relative_path,
        sha256=sha256,
    )


def _fast_bindings(candidates: list[_FastCandidate]) -> dict[str, dict[str, dict[str, str]]]:
    bindings: dict[str, dict[str, dict[str, str]]] = {}
    for candidate in candidates:
        post_sha256 = _sha256_file(candidate.module_path)
        if post_sha256 != candidate.sha256:
            raise RuntimeError(
                f"fast gate {FAST_STATUS_INFRASTRUCTURE_ERROR}: candidate "
                f"{candidate.operator_name}/{candidate.candidate_id} changed during evaluation"
            )
        bindings.setdefault(candidate.operator_name, {})[candidate.candidate_id] = {
            "module_path": candidate.relative_path,
            "sha256": candidate.sha256,
            "post_sha256": post_sha256,
        }
    return bindings


def _fast_response_payload(
    *,
    request_id: str,
    request_sha256: str,
    mode: str,
    status: str,
    error: str = "",
    operators: Mapping[str, object] | None = None,
    msprof_telemetry: Mapping[str, object] | None = None,
    candidates: Mapping[str, object] | None = None,
    coverage: Mapping[str, int] | None = None,
    micro_probe: Mapping[str, object] | None = None,
    null_launch_us: float | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": FAST_RESPONSE_SCHEMA,
        "request_id": request_id,
        "request_sha256": request_sha256,
        "mode": mode,
        "status": status,
        "error": error[-1000:],
        "operators": dict(operators or {}),
        "msprof_telemetry": dict(msprof_telemetry or {}),
        "candidates": dict(candidates or {}),
        "coverage": dict(
            coverage
            or {
                "expected_candidates": 0,
                "evaluated_candidates": 0,
                "expected_msprof_targets": 0,
                "observed_msprof_targets": 0,
            }
        ),
    }
    if micro_probe is not None:
        payload["micro_probe"] = dict(micro_probe)
    if mode == "batch" and null_launch_us is not None:
        null_val = float(null_launch_us)
        if math.isfinite(null_val) and null_val > 0.0:
            payload["null_launch_us"] = null_val
    return payload


def _run_fast_batch(
    req: Mapping[str, object],
    *,
    source_root: Path,
    cannbench_repo: Path,
    response_path: Path,
    request_id: str,
    request_sha256: str,
) -> None:
    raw_batches = req.get("batches")
    if not isinstance(raw_batches, list) or not raw_batches:
        raise _FastRequestError("fast batch request requires a non-empty batches list")
    if req.get("msprof_targets") not in (None, []):
        raise _FastRequestError("fast batch request must not declare msprof_targets")
    warmup = _fast_int(req.get("warmup"), "warmup", default=3)
    repeat = _fast_int(req.get("repeat"), "repeat", default=5)

    used_candidates: set[tuple[str, str]] = set()
    operators: set[str] = set()
    candidates: list[_FastCandidate] = []
    normalized_batches: list[dict[str, object]] = []
    for row in raw_batches:
        if not isinstance(row, Mapping):
            raise _FastRequestError("fast batch entries must be objects")
        operator_name = str(row.get("operator_name", "")).strip()
        if not operator_name:
            raise _FastRequestError("fast batch entry has no operator_name")
        if operator_name in operators:
            raise _FastRequestError(f"duplicate fast batch operator {operator_name!r}")
        operators.add(operator_name)
        raw_candidates = row.get("candidates")
        if not isinstance(raw_candidates, list) or not raw_candidates:
            raise _FastRequestError(
                f"fast batch operator {operator_name!r} has no candidates"
            )
        normalized_candidates: list[dict[str, str]] = []
        for cand_row in raw_candidates:
            if not isinstance(cand_row, Mapping):
                raise _FastRequestError("fast batch candidates must be objects")
            candidate = _fast_candidate(source_root, operator_name, cand_row, used_candidates)
            candidates.append(candidate)
            normalized_candidates.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "module_path": str(candidate.module_path),
                }
            )
        raw_anchors = row.get("anchors")
        if not isinstance(raw_anchors, list) or not raw_anchors:
            raise _FastRequestError(
                f"fast batch operator {operator_name!r} has no anchors"
            )
        normalized_batches.append(
            {
                "operator_name": operator_name,
                "entrypoint": str(row.get("entrypoint") or get_operator_entrypoint(operator_name)),
                "candidates": normalized_candidates,
                "anchors": raw_anchors,
            }
        )

    null_launch_us = _measure_null_launch_us()
    results_by_op: dict[str, dict[str, dict[str, object]]] = {}
    failures: list[str] = []
    for batch in normalized_batches:
        op_name = str(batch["operator_name"])
        op_res = _evaluate_operator_batch(
            batch,
            source_root=source_root,
            cannbench_repo=cannbench_repo,
            warmup=warmup,
            repeat=repeat,
        )
        results_by_op[op_name] = {cid: res.to_dict() for cid, res in op_res.items()}
        failures.extend(
            f"{op_name}/{cid}" for cid, res in op_res.items() if not res.all_passed
        )
    payload = _fast_response_payload(
        request_id=request_id,
        request_sha256=request_sha256,
        mode="batch",
        status=FAST_STATUS_OK if not failures else FAST_STATUS_CANDIDATE_FAILURE,
        operators=results_by_op,
        candidates=_fast_bindings(candidates),
        coverage={
            "expected_candidates": len(candidates),
            "evaluated_candidates": sum(len(cands) for cands in results_by_op.values()),
            "expected_msprof_targets": 0,
            "observed_msprof_targets": 0,
        },
        null_launch_us=null_launch_us,
    )
    _atomic_write_json(response_path, payload)
    print(json.dumps(payload, sort_keys=True), flush=True)


def _run_fast_profile(
    req: Mapping[str, object],
    *,
    source_root: Path,
    cannbench_repo: Path,
    response_path: Path,
    request_id: str,
    request_sha256: str,
) -> None:
    raw_targets = req.get("msprof_targets")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise _FastRequestError("fast profile request requires a non-empty msprof_targets list")
    if req.get("batches") not in (None, []):
        raise _FastRequestError("fast profile request must not declare batches")
    default_warmup = _fast_int(req.get("warmup"), "warmup", default=1)
    default_repeat = _fast_int(req.get("repeat"), "repeat", default=3)

    used_candidates: set[tuple[str, str]] = set()
    candidate_by_key: dict[tuple[str, str], _FastCandidate] = {}
    seen_targets: set[tuple[str, str, int]] = set()
    candidates: list[_FastCandidate] = []
    normalized_rows: list[dict[str, object]] = []
    for row in raw_targets:
        if not isinstance(row, Mapping):
            raise _FastRequestError("fast profile targets must be objects")
        operator_name = str(row.get("operator_name", "")).strip()
        if not operator_name:
            raise _FastRequestError("fast profile target has no operator_name")
        declared_candidate_id = str(row.get("candidate_id", "")).strip()
        if not declared_candidate_id:
            raise _FastRequestError(
                f"fast request candidate under {operator_name!r} has no candidate_id"
            )
        key = (operator_name, declared_candidate_id)
        candidate = candidate_by_key.get(key)
        if candidate is None:
            candidate = _fast_candidate(source_root, operator_name, row, used_candidates)
            candidate_by_key[key] = candidate
            candidates.append(candidate)
        elif (
            str(row.get("module_path", "")).strip() != candidate.relative_path
            or str(row.get("sha256", "")).strip().lower() != candidate.sha256
        ):
            raise _FastRequestError(
                f"fast profile candidate {operator_name}/{candidate.candidate_id} "
                "changes module_path or sha256 across targets"
            )
        case_num = row.get("case_num")
        if isinstance(case_num, bool) or not isinstance(case_num, int) or case_num <= 0:
            raise _FastRequestError(
                "fast profile target case_num must be a positive integer"
            )
        target_key = (operator_name, candidate.candidate_id, case_num)
        if target_key in seen_targets:
            raise _FastRequestError(
                f"duplicate fast profile target {operator_name}/"
                f"{candidate.candidate_id}#{case_num}"
            )
        seen_targets.add(target_key)
        normalized_rows.append(
            {
                "operator_name": operator_name,
                "candidate_id": candidate.candidate_id,
                "module_path": str(candidate.module_path),
                "case_num": case_num,
                "warmup": _fast_int(row.get("warmup"), "warmup", default=default_warmup),
                "repeat": _fast_int(row.get("repeat"), "repeat", default=default_repeat),
            }
        )

    telemetry = _collect_msprof_targets(
        normalized_rows,
        source_root=source_root,
        cannbench_repo=cannbench_repo,
    )
    observed_targets = sum(
        len(cases) for cands in telemetry.values() for cases in cands.values()
    )
    payload = _fast_response_payload(
        request_id=request_id,
        request_sha256=request_sha256,
        mode="profile",
        status=FAST_STATUS_OK,
        msprof_telemetry=telemetry,
        candidates=_fast_bindings(candidates),
        coverage={
            "expected_candidates": len(candidates),
            "evaluated_candidates": len(candidates),
            "expected_msprof_targets": len(raw_targets),
            "observed_msprof_targets": observed_targets,
        },
    )
    _atomic_write_json(response_path, payload)
    print(json.dumps(payload, sort_keys=True), flush=True)


def _run_fast_micro_probe(
    req: Mapping[str, object],
    *,
    source_root: Path,
    response_path: Path,
    request_id: str,
    request_sha256: str,
) -> None:
    raw_probe = req.get("probe")
    if not isinstance(raw_probe, Mapping):
        raise _FastRequestError("fast micro_probe request requires a probe object")
    probe_id = str(raw_probe.get("probe_id", "")).strip()
    probe_kind = str(raw_probe.get("probe_kind", "")).strip()
    factory_name = str(raw_probe.get("factory_name", "")).strip()
    tensor_dtype = str(raw_probe.get("tensor_dtype", "")).strip()
    if not probe_id or not probe_kind or not factory_name or not tensor_dtype:
        raise _FastRequestError(
            "fast micro_probe request probe requires probe_id, probe_kind, "
            "factory_name, and tensor_dtype"
        )
    line_counts: dict[str, int] = {}
    for field in ("line_count_baseline", "line_count_optimized"):
        value = raw_probe.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise _FastRequestError(
                f"fast request field 'probe.{field}' must be a non-negative integer"
            )
        line_counts[field] = value
    raw_shape = raw_probe.get("tensor_shape")
    if (
        not isinstance(raw_shape, list)
        or not 1 <= len(raw_shape) <= 8
        or any(
            isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0
            for dim in raw_shape
        )
    ):
        raise _FastRequestError(
            "fast request field 'probe.tensor_shape' must list 1 to 8 positive integers"
        )
    min_speedup = raw_probe.get("min_speedup")
    if (
        isinstance(min_speedup, bool)
        or not isinstance(min_speedup, (int, float))
        or not math.isfinite(min_speedup)
        or min_speedup <= 0.0
    ):
        raise _FastRequestError(
            "fast request field 'probe.min_speedup' must be a finite positive number"
        )

    raw_candidates = req.get("candidates")
    if not isinstance(raw_candidates, list) or len(raw_candidates) != 2:
        raise _FastRequestError(
            "fast micro_probe request requires baseline and optimized candidates"
        )
    used_candidates: set[tuple[str, str]] = set()
    candidates = [
        _fast_candidate(source_root, "micro_probe", row, used_candidates)
        for row in raw_candidates
        if isinstance(row, Mapping)
    ]
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    if len(candidates) != 2 or set(by_id) != {"baseline", "optimized"}:
        raise _FastRequestError(
            "fast micro_probe request requires baseline and optimized candidates"
        )
    spec = MicroProbeSpec(
        probe_id=probe_id,
        probe_kind=probe_kind,
        factory_name=factory_name,
        baseline_probe_source=by_id["baseline"].module_path.read_text(encoding="utf-8"),
        optimized_probe_source=by_id["optimized"].module_path.read_text(encoding="utf-8"),
        line_count_baseline=line_counts["line_count_baseline"],
        line_count_optimized=line_counts["line_count_optimized"],
        tensor_shape=tuple(raw_shape),
        tensor_dtype=tensor_dtype,
    )
    passed, metrics = verify_micro_probe_hardware(
        spec,
        min_speedup=float(min_speedup),
    )
    payload = _fast_response_payload(
        request_id=request_id,
        request_sha256=request_sha256,
        mode="micro_probe",
        status=FAST_STATUS_OK if passed else FAST_STATUS_CANDIDATE_FAILURE,
        candidates=_fast_bindings(candidates),
        coverage={
            "expected_candidates": len(candidates),
            "evaluated_candidates": len(candidates),
            "expected_msprof_targets": 0,
            "observed_msprof_targets": 0,
        },
        micro_probe={
            "probe_id": probe_id,
            "metrics": {str(key): float(value) for key, value in metrics.items()},
        },
    )
    _atomic_write_json(response_path, payload)
    print(json.dumps(payload, sort_keys=True), flush=True)


def _handle_fast_request(request_path_raw: str, source_root_raw: str, shim_dir: Path) -> None:
    request_path = Path(request_path_raw)
    if not request_path.is_file():
        raise RuntimeError(f"fast request {request_path} does not exist")
    request_bytes = request_path.read_bytes()
    request_sha256 = hashlib.sha256(request_bytes).hexdigest()
    try:
        req = json.loads(request_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"fast request {request_path} is not valid JSON: {exc}") from None
    if not isinstance(req, dict):
        raise RuntimeError(f"fast request {request_path} must be a JSON object")
    source_root = _fast_source_root(source_root_raw)
    response_path = _fast_path(source_root, req.get("response_path"), "response_path")
    request_id = str(req.get("request_id", "")).strip()
    mode = str(req.get("mode", "")).strip()

    def _fail(status: str, message: str) -> None:
        _atomic_write_json(
            response_path,
            _fast_response_payload(
                request_id=request_id,
                request_sha256=request_sha256,
                mode=mode,
                status=status,
                error=message,
            ),
        )

    try:
        if req.get("schema") != FAST_REQUEST_SCHEMA:
            raise _FastRequestError(
                f"fast request schema {req.get('schema')!r} != {FAST_REQUEST_SCHEMA!r}"
            )
        if not request_id:
            raise _FastRequestError("fast request has no request_id")
        if mode not in ("batch", "profile", "micro_probe"):
            raise _FastRequestError(f"fast request mode {mode!r} is not recognized")
        if mode == "micro_probe":
            tl_cache_raw = os.environ.get("TILELANG_CACHE_DIR", "").strip()
            if tl_cache_raw:
                _configure_tilelang_cache(tl_cache_raw)
            _run_fast_micro_probe(
                req,
                source_root=source_root,
                response_path=response_path,
                request_id=request_id,
                request_sha256=request_sha256,
            )
            return
        cannbench_repo = _fast_env_dir(
            "TILEFOUNDRY_NPU_CANNBENCH_REPO", create=False
        )
        tl_cache_dir = _fast_env_dir("TILELANG_CACHE_DIR", create=True)
        reference_cache_dir = _fast_env_dir("CANNBENCH_CACHE_DIR", create=True)
        os.environ["CANNBENCH_CACHE_DIR"] = str(reference_cache_dir)
        _ensure_npu_compare(cannbench_repo, shim_dir)
        _configure_tilelang_cache(str(tl_cache_dir))
        if mode == "batch":
            _run_fast_batch(
                req,
                source_root=source_root,
                cannbench_repo=cannbench_repo,
                response_path=response_path,
                request_id=request_id,
                request_sha256=request_sha256,
            )
        else:
            _run_fast_profile(
                req,
                source_root=source_root,
                cannbench_repo=cannbench_repo,
                response_path=response_path,
                request_id=request_id,
                request_sha256=request_sha256,
            )
    except _FastRequestError as exc:
        _fail(FAST_STATUS_REQUEST_ERROR, str(exc))
    except Exception as exc:
        _fail(FAST_STATUS_INFRASTRUCTURE_ERROR, f"{type(exc).__name__}: {exc}")
        raise


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


def _configure_tilelang_cache(tl_cache_dir: str) -> None:
    resolved = Path(tl_cache_dir).resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    os.environ["TILELANG_CACHE_DIR"] = str(resolved)
    os.environ["TILELANG_KERNEL_CACHE_USE_LIB_STAMP"] = "1"
    os.environ["TILELANG_DISABLE_CACHE"] = "0"
    tl_env_mod = importlib.import_module("tilelang.env")
    tl_env_mod.env.enable_cache()
    kc_mod = importlib.import_module("tilelang.cache.kernel_cache")
    kc_mod.KernelCache._get_cache_namespace.cache_clear()
    kc_mod.KernelCache._get_tilelang_lib_stamp.cache_clear()
    kc_mod.KernelCache._create_dirs()


def test_extract_msprof_metrics_omits_unmeasured_counters() -> None:
    """Profiler extraction never fabricates sync_stall_ratio or absent PMU keys."""
    prof = SimpleNamespace(
        aic_mad=85.0,
        aiv_vec=25.0,
        aiv_mte2=40.0,
        aic_total_cycles=2000,
        aiv_total_cycles=4000,
        hbm_read_bytes=1048576,
        hbm_write_bytes=524288,
        dur_us=18.5,
    )
    metrics = _extract_msprof_metrics(prof)
    assert "sync_stall_ratio" not in metrics
    assert "l2_hit_rate" not in metrics
    assert "bank_conflict_ratio" not in metrics
    assert "aic_fixpipe" not in metrics
    assert metrics["aic_mac_ratio"] == pytest.approx(0.85)
    assert metrics["aiv_vec_ratio"] == pytest.approx(0.25)
    assert metrics["mte2_wait_ratio"] == pytest.approx(0.40)
    assert metrics["hbm_read_bytes"] == 1048576
    assert metrics["hbm_write_bytes"] == 524288
    assert metrics["actual_gm_bytes"] == 1572864
    assert metrics["dur_us"] == pytest.approx(18.5)

    explicit_prof = SimpleNamespace(
        aiv_vec=0.60,
        sync_stall_ratio=22.0,
        l2_hit_rate=78.5,
        dur_ns=12000.0,
    )
    explicit_metrics = _extract_msprof_metrics(explicit_prof)
    assert explicit_metrics["sync_stall_ratio"] == pytest.approx(0.22)
    assert explicit_metrics["l2_hit_rate"] == pytest.approx(0.785)
    assert explicit_metrics["dur_us"] == pytest.approx(12.0)
    assert "aic_mac_ratio" not in explicit_metrics
    assert "mte2_wait_ratio" not in explicit_metrics

    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=None,
        pipe_cycles={"AIV0.PIPE_V": 100},
        waterlines={},
    )
    obs = ledger.to_observation_5d(telemetry=metrics)
    assert "l2_sync_stall_ratio" not in obs.measured_l2_fields
    assert "l2_cache_hit_rate" not in obs.measured_l2_fields
    assert "l2_fixpipe_ratio" not in obs.measured_l2_fields
    assert obs.measured_l2_fields == (
        "l2_measured_latency_us",
        "l2_aic_mac_ratio",
        "l2_aiv_vec_ratio",
        "l2_mte2_wait_ratio",
    )


def test_fast_bindings_treat_candidate_change_as_infrastructure(tmp_path: Path) -> None:
    """A candidate rewritten after evaluation invalidates the run without blaming the request."""
    module = tmp_path / "kernel.py"
    module.write_text("x = 1\n", encoding="utf-8")
    candidate = _FastCandidate(
        operator_name="softmax",
        candidate_id="cand-1",
        module_path=module,
        relative_path="kernel.py",
        sha256=_sha256_file(module),
    )
    module.write_text("x = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError) as failure:
        _fast_bindings([candidate])
    assert not isinstance(failure.value, _FastRequestError)
    assert FAST_STATUS_INFRASTRUCTURE_ERROR in str(failure.value)
    assert "changed during evaluation" in str(failure.value)


def test_measure_npu_ring_burst_resamples_once_on_high_cv() -> None:
    """Verify ring-burst discards first noisy burst when CV > 0.25 and resamples once."""

    @dataclass(frozen=True)
    class _TensorStub:
        nbytes: int

    class _EventStub:
        def __init__(self, clock: list[float]) -> None:
            self._clock = clock
            self._ts = 0.0

        def record(self) -> None:
            self._ts = self._clock[0]

        def elapsed_time(self, end_event: _EventStub) -> float:
            return end_event._ts - self._ts

    burst_call = 0
    clock = [1.0]
    first_round_ms = [0.002 * 24, 0.020 * 24, 0.002 * 24]
    second_round_ms = [0.003 * 24, 0.003 * 24, 0.003 * 24]
    recorded_durations = iter(first_round_ms + second_round_ms)
    invoke_count = 0

    def _fn(x: _TensorStub) -> None:
        nonlocal invoke_count, burst_call
        _ = x
        invoke_count += 1
        if invoke_count > 2 and (invoke_count - 2) % 24 == 1:
            burst_call += 1
            clock[0] += next(recorded_durations, 0.003 * 24)

    avg_us, burst_k = measure_npu_ring_burst_us(
        _fn,
        (_TensorStub(4 * 1024 * 1024),),
        warmup=1,
        repeat=3,
        t_hw_us=1.5,
        first_probe_us=1.5,
        event_factory=lambda: _EventStub(clock),
        synchronize_fn=lambda: None,
        clone_fn=lambda obj: obj,
    )
    assert burst_k == 24
    assert burst_call == 6
    assert avg_us == pytest.approx(3.0)


def test_surrogate_anchor_batch_npu() -> None:
    """Evaluate configured or baseline 10-anchor surrogate batch inside AgentGates lease."""
    _require_npu_env()
    torch.npu.set_device(0)

    fast_req_env = os.environ.get(FAST_REQUEST_ENV, "").strip()
    req_env = os.environ.get("TF_SURROGATE_BATCH_REQUEST", "").strip()
    if fast_req_env and req_env:
        raise RuntimeError(
            f"{FAST_REQUEST_ENV} and TF_SURROGATE_BATCH_REQUEST must not both be set"
        )
    shim_only = os.environ.get("TF_SURROGATE_SHIM_ONLY", "").strip() == "1"
    shim_env_dir = os.environ.get("TF_SURROGATE_SHIM_DIR", "").strip()
    tl_cache_env = os.environ.get("TF_SURROGATE_TL_CACHE_DIR", "").strip()

    if shim_only:
        if tl_cache_env:
            _configure_tilelang_cache(tl_cache_env)
        if shim_env_dir and (Path(shim_env_dir) / "npu_compare.py").is_file():
            if shim_env_dir not in sys.path:
                sys.path.insert(0, shim_env_dir)
            importlib.import_module("npu_compare")
            return
        target_shim = (
            Path(shim_env_dir)
            if shim_env_dir
            else Path(tempfile.mkdtemp(prefix="tf_surrogate_shim_persist_"))
        )
        cann_repo = _resolve_batch_cannbench_repo(default_source_root())
        _ensure_npu_compare(cann_repo, target_shim)
        importlib.import_module("npu_compare")
        return

    with tempfile.TemporaryDirectory(prefix="tf_surrogate_shim_") as shim_dir:
        if fast_req_env:
            if not Path(fast_req_env).is_file():
                raise RuntimeError(f"{FAST_REQUEST_ENV} does not exist: {fast_req_env}")
            _handle_fast_request(
                fast_req_env,
                os.environ.get(FAST_SOURCE_ROOT_ENV, "").strip(),
                Path(shim_dir),
            )
            return
        if req_env and Path(req_env).is_file():
            req = json.loads(Path(req_env).read_text(encoding="utf-8"))
            source_root = Path(str(req["source_root"])).resolve()
            cann_repo = Path(str(req["cannbench_repo"])).resolve()
        else:
            source_root = default_source_root()
            cann_repo = _resolve_batch_cannbench_repo(source_root)
            req = _default_batch_request(source_root, cann_repo)

        _ensure_npu_compare(cann_repo, Path(shim_dir))

        if not os.environ.get("CANNBENCH_CACHE_DIR"):
            uid = os.getuid() if hasattr(os, "getuid") else 0
            cache_dir = Path(tempfile.gettempdir()) / f"tf_surrogate_ref_cache_{uid}"
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ["CANNBENCH_CACHE_DIR"] = str(cache_dir)

        uid = os.getuid() if hasattr(os, "getuid") else 0
        default_tl_cache = str(Path(tempfile.gettempdir()) / f"tf_surrogate_tl_cache_{uid}")
        tl_cache_dir = (
            str(req.get("tilelang_cache_dir", "")).strip()
            or os.environ.get("TF_SURROGATE_TL_CACHE_DIR", "").strip()
            or default_tl_cache
        )
        _configure_tilelang_cache(tl_cache_dir)

        warmup = int(req.get("warmup", 3))
        repeat = int(req.get("repeat", 5))
        require_all_passed = bool(req.get("require_all_passed", False))
        collect_msprof_detail = bool(req.get("collect_msprof_detail", False))
        resp_path_raw = str(req.get("response_path", "")).strip()

        results_by_op: dict[str, dict[str, dict[str, object]]] = {}
        parsed_by_op: dict[str, dict[str, CandidateSurrogateResult]] = {}
        msprof_telemetry: dict[str, dict[str, dict[str, dict[str, object]]]] = {}
        for batch in req.get("batches", ()):
            op_msprof: dict[str, dict[str, dict[str, object]]] = {}
            op_res = _evaluate_operator_batch(
                batch,
                source_root=source_root,
                cannbench_repo=cann_repo,
                warmup=warmup,
                repeat=repeat,
                collect_msprof_detail=collect_msprof_detail,
                msprof_out=op_msprof,
            )
            op_name = str(batch["operator_name"])
            parsed_by_op[op_name] = op_res
            results_by_op[op_name] = {cid: res.to_dict() for cid, res in op_res.items()}
            if op_msprof:
                msprof_telemetry[op_name] = op_msprof

        raw_msprof_targets = req.get("msprof_targets") or []
        if raw_msprof_targets:
            target_tel = _collect_msprof_targets(
                list(raw_msprof_targets),
                source_root=source_root,
                cannbench_repo=cann_repo,
            )
            for op_name, cands_map in target_tel.items():
                op_bucket = msprof_telemetry.setdefault(op_name, {})
                for cid, cases_map in cands_map.items():
                    op_bucket.setdefault(cid, {}).update(cases_map)

        summary_payload: dict[str, object] = {"operators": results_by_op}
        if msprof_telemetry or collect_msprof_detail or raw_msprof_targets:
            summary_payload["msprof_telemetry"] = msprof_telemetry
        if resp_path_raw:
            _atomic_write_json(Path(resp_path_raw), summary_payload)
        print(json.dumps(summary_payload, sort_keys=True), flush=True)

        if require_all_passed:
            for op_name, op_res in parsed_by_op.items():
                for cid, res in op_res.items():
                    assert res.all_passed, f"{op_name}/{cid} failed: {res.to_dict()}"
                    for anchor in res.anchors:
                        assert anchor.cand_us > 0.0, f"{op_name}/{cid} #{anchor.case_num} invalid time"
