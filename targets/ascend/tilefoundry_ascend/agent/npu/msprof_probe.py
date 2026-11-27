"""On-device msprof bottleneck profiling, micro-probes, and LocalHardwareProbePort."""

from __future__ import annotations

import json
import logging
import math
import os
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Protocol

from tilefoundry_ascend.agent.npu.ag_transport import (
    FAST_REQUEST_SCHEMA, _agent_gates_workspace_name, _build_py_exec_argv, _fast_cache_roots,
    _fast_request_root, _fast_response_mapping, _fast_wire_operator_name, _hook,
    _resolve_agent_gates_bin, _resolve_env_file_path, _sha256_file, _stage_fast_candidate,
    _submit_fast_gate, _verify_fast_bindings, _verify_fast_coverage, _write_fast_request,
)
from tilefoundry_ascend.agent.npu.l2_paired_eval import (
    AnchorEvalMetrics, CandidateSurrogateResult, prewarm_anchor_reference_cache,
    prewarm_surrogate_tilelang_cache,
)
from tilefoundry_ascend.agent.probes._cann_env import (
    LocalHardwareProbePort, ensure_cann_include_cpath, run_msprof_detail,
)
from tilefoundry_ascend.agent.specs.anchor_specs import (
    AnchorCaseSpec, default_source_root, parse_simple_env_file, resolve_cannbench_repo,
)

logger = logging.getLogger(__name__)


class MicroProbeSources(Protocol):
    """Metadata and the two probe source texts a hardware request carries."""

    probe_id: str
    probe_kind: str
    factory_name: str
    baseline_probe_source: str
    optimized_probe_source: str
    line_count_baseline: int
    line_count_optimized: int
    tensor_shape: tuple[int, ...]
    tensor_dtype: str


def bottleneck_anchor_for_regime(result: CandidateSurrogateResult, regime_tag: str) -> AnchorEvalMetrics | None:
    """Return the lowest-Eq.3-score evaluated anchor matching regime_tag."""
    matched = [a for a in result.anchors if a.regime_tag == regime_tag]
    return min(matched, key=lambda a: (a.eq3_score, -a.cand_us, a.case_num)) if matched else None


def _collect_profile_via_agent_gates(
    operator_name: str, candidate_path: str | Path,
    anchors_by_regime: Sequence[AnchorCaseSpec] | Mapping[str, AnchorCaseSpec], *,
    candidate_id: str, source_root: str | Path | None = None, env_file: str | Path | None = None,
    device_id: str = "1", device_pool: Sequence[str] = (), warmup: int = 1, repeat: int = 3,
    timeout_sec: int = 300,
) -> dict[str, dict[str, object]]:
    """Collect mode='profile' msprof telemetry inside an AgentGates lease."""
    regime_items = (
        [(str(t), s) for t, s in anchors_by_regime.items()]
        if isinstance(anchors_by_regime, Mapping)
        else [(s.regime_tag, s) for s in anchors_by_regime]
    )
    if not regime_items:
        return {}
    unique_cnums = list(dict.fromkeys(int(s.case_num) for _t, s in regime_items))
    root = Path(source_root).resolve() if source_root is not None else default_source_root()
    resolved_env = _hook("_resolve_env_file_path", _resolve_env_file_path)(env_file)
    ag_bin = _hook("_resolve_agent_gates_bin", _resolve_agent_gates_bin)(root)
    env_vals = parse_simple_env_file(resolved_env)
    cann_repo = _hook("resolve_cannbench_repo", resolve_cannbench_repo)(
        cannbench_repo=env_vals.get("TILEFOUNDRY_NPU_CANNBENCH_REPO"), env_file=resolved_env,
    )
    cache_dir, tl_cache_dir = _fast_cache_roots(root, env_vals)
    cand_resolved = Path(candidate_path).resolve()
    unique_specs = [next(s for _t, s in regime_items if int(s.case_num) == cn) for cn in unique_cnums]
    _hook("prewarm_anchor_reference_cache", prewarm_anchor_reference_cache)(
        operator_name, unique_specs, cannbench_repo=cann_repo, env_file=resolved_env, cache_dir=cache_dir, source_root=root,
    )
    _hook("prewarm_surrogate_tilelang_cache", prewarm_surrogate_tilelang_cache)(
        operator_name, [(candidate_id, cand_resolved)], unique_specs, cannbench_repo=cann_repo,
        env_file=resolved_env, cache_dir=cache_dir, tl_cache_dir=tl_cache_dir, source_root=root,
    )
    req_root, req_id = _fast_request_root(root, env_vals), uuid.uuid4().hex
    req_dir, req_path, resp_path = req_root / req_id, req_root / f"{req_id}.request.json", req_root / f"{req_id}.response.json"
    wire_op = _fast_wire_operator_name(operator_name)
    relative, digest = _stage_fast_candidate(req_dir, root, wire_op, 0, candidate_id, cand_resolved)
    payload: dict[str, object] = {
        "schema": FAST_REQUEST_SCHEMA, "request_id": req_id, "mode": "profile",
        "warmup": int(warmup), "repeat": int(repeat), "require_all_passed": False,
        "response_path": resp_path.relative_to(root).as_posix(), "batches": [],
        "msprof_targets": [
            {"operator_name": wire_op, "candidate_id": candidate_id, "module_path": relative, "sha256": digest, "case_num": c, "warmup": int(warmup), "repeat": int(repeat)}
            for c in unique_cnums
        ],
    }
    raw_resp, _ = _hook("_submit_fast_gate", _submit_fast_gate)(
        agent_gates_bin=ag_bin, root=root, workspace=_hook("_agent_gates_workspace_name", _agent_gates_workspace_name)(root),
        request_id=req_id, request_relative_path=req_path.relative_to(root).as_posix(), request_path=req_path,
        response_path=resp_path, request_sha256=_write_fast_request(req_path, payload), mode="profile",
        timeout_sec=float(timeout_sec), device_id=str(device_id), device_pool=device_pool, env_vals=env_vals,
    )
    _verify_fast_bindings(raw_resp, {wire_op: {candidate_id: {"module_path": relative, "sha256": digest}}})
    _verify_fast_coverage(raw_resp, expected_candidates=1, expected_msprof_targets=len(unique_cnums))
    t_all = _fast_response_mapping(raw_resp, "msprof_telemetry")
    t_raw = t_all.get(wire_op) if wire_op in t_all else t_all.get(operator_name)
    by_case = t_raw.get(candidate_id) if isinstance(t_raw, Mapping) else None
    if not isinstance(by_case, Mapping):
        raise RuntimeError(f"fast profile response has no msprof telemetry for {operator_name!r}")
    return {
        tag: {str(k): v for k, v in raw_m.items()}
        for tag, spec in regime_items
        if isinstance(raw_m := by_case.get(str(int(spec.case_num))), Mapping) and "error" not in raw_m
    }


def collect_bottleneck_msprof_via_agent_gates(
    operator_name: str, candidate_path: str | Path,
    anchors_by_regime: Sequence[AnchorCaseSpec] | Mapping[str, AnchorCaseSpec], *,
    source_root: str | Path | None = None, env_file: str | Path | None = None,
    device_id: str = "1", device_pool: Sequence[str] = (), warmup: int = 1, repeat: int = 3, timeout_sec: int = 300,
) -> dict[str, dict[str, float]]:
    """Collect numeric msprof telemetry for bottleneck anchors inside an AgentGates lease."""
    raw = _collect_profile_via_agent_gates(
        operator_name, candidate_path, anchors_by_regime, candidate_id="bottleneck_probe",
        source_root=source_root, env_file=env_file, device_id=device_id, device_pool=device_pool,
        warmup=warmup, repeat=repeat, timeout_sec=timeout_sec,
    )
    return {
        tag: {str(k): float(v) for k, v in metrics.items() if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))}
        for tag, metrics in raw.items()
    }


def collect_p2_deep_dive_via_agent_gates(
    operator_name: str, candidate_path: str | Path,
    anchors_by_regime: Sequence[AnchorCaseSpec] | Mapping[str, AnchorCaseSpec], *,
    source_root: str | Path | None = None, env_file: str | Path | None = None,
    device_id: str = "1", device_pool: Sequence[str] = (), warmup: int = 1, repeat: int = 3, timeout_sec: int = 300,
) -> dict[str, object]:
    """Collect instruction-level deep-dive telemetry and DWARF hotspots via AgentGates."""
    return dict(_collect_profile_via_agent_gates(
        operator_name, candidate_path, anchors_by_regime, candidate_id="bottleneck_p2_deep_dive",
        source_root=source_root, env_file=env_file, device_id=device_id, device_pool=device_pool,
        warmup=warmup, repeat=repeat, timeout_sec=timeout_sec,
    ))


def _prewarm_micro_probe_cache(spec: MicroProbeSources, *, source_root: Path, env_file: Path, tl_cache_dir: Path) -> None:
    env_vals = parse_simple_env_file(env_file)
    python_bin = env_vals.get("TILEFOUNDRY_NPU_PYTHON") or os.environ.get("TILEFOUNDRY_NPU_PYTHON") or sys.executable
    cann_setup = (env_vals.get("TILEFOUNDRY_NPU_CANN_SETUP") or "").strip()
    tl_source = (env_vals.get("TILEFOUNDRY_NPU_TILELANG_SOURCE") or "").strip()
    tl_cache_dir.mkdir(parents=True, exist_ok=True)
    pp_parts = [*( [tl_source] if tl_source else [] ), str(source_root), str(source_root / "src"), str(source_root / "targets" / "ascend")]
    run_env = {
        **os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0", "ASCEND_RT_VISIBLE_DEVICES": "", "USE_CUDA": "0",
        "TILELANG_CACHE_DIR": str(tl_cache_dir), "TILELANG_KERNEL_CACHE_USE_LIB_STAMP": "1",
        "TILELANG_DISABLE_CACHE": "0", "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": os.pathsep.join(pp_parts),
    }
    sources = (("base", spec.baseline_probe_source), ("opt", spec.optimized_probe_source))
    script = (
        "import importlib, importlib.util, os, tempfile\nfrom pathlib import Path\n"
        f"os.environ['TILELANG_CACHE_DIR'] = {str(tl_cache_dir)!r}\n"
        "os.environ['TILELANG_KERNEL_CACHE_USE_LIB_STAMP'] = '1'\nos.environ['TILELANG_DISABLE_CACHE'] = '0'\n"
        "import tilelang\nimportlib.import_module('tilelang.env').env.enable_cache()\n"
        "tl_kc = importlib.import_module('tilelang.cache.kernel_cache')\n"
        "tl_kc.KernelCache._get_cache_namespace.cache_clear()\ntl_kc.KernelCache._get_tilelang_lib_stamp.cache_clear()\ntl_kc.KernelCache._create_dirs()\n"
        f"sources, factory = {json.dumps(sources)}, {spec.factory_name!r}\n"
        "with tempfile.TemporaryDirectory(prefix='tf_probe_pre_') as td:\n"
        "    for tag, src in sources:\n        p = Path(td) / f'{tag}.py'\n        p.write_text(src, encoding='utf-8')\n"
        "        sp = importlib.util.spec_from_file_location(f'_tf_p_{tag}', str(p))\n"
        "        mod = importlib.util.module_from_spec(sp)\n        sp.loader.exec_module(mod)\n"
        "        tilelang.compile(getattr(mod, factory)(), target='ascend', out_idx=-1)\n"
    )
    with tempfile.TemporaryDirectory(prefix="tf_probe_cpu_") as tmp_dir:
        drv = Path(tmp_dir) / "prewarm_probe.py"
        drv.write_text(script, encoding="utf-8")
        try:
            subprocess.run(_build_py_exec_argv(python_bin, drv, cann_setup), text=True, capture_output=True, cwd=str(source_root), env=run_env, timeout=180)
        except Exception as exc:
            logger.debug("Micro-probe CPU prewarm skipped: %s", exc)


def run_micro_probe_via_agent_gates(
    spec: MicroProbeSources, *, source_root: str | Path | None = None, env_file: str | Path | None = None,
    device_id: str = "1", device_pool: Sequence[str] = (), min_speedup: float = 1.08,
    timeout_sec: int = 360, prewarm_cache: bool = True,
) -> dict[str, float]:
    """Run hardware micro-probe verification through a typed AgentGates fast request."""
    root = Path(source_root).resolve() if source_root is not None else default_source_root()
    resolved_env = _hook("_resolve_env_file_path", _resolve_env_file_path)(env_file)
    ag_bin = _hook("_resolve_agent_gates_bin", _resolve_agent_gates_bin)(root)
    env_vals = parse_simple_env_file(resolved_env)
    _ref_cache_dir, tl_cache_dir = _fast_cache_roots(root, env_vals)
    if prewarm_cache:
        _hook("_prewarm_micro_probe_cache", _prewarm_micro_probe_cache)(spec, source_root=root, env_file=resolved_env, tl_cache_dir=tl_cache_dir)
    req_root, req_id = _fast_request_root(root, env_vals), uuid.uuid4().hex
    req_dir, req_path, resp_path = req_root / req_id, req_root / f"{req_id}.request.json", req_root / f"{req_id}.response.json"
    staged: dict[str, dict[str, dict[str, str]]] = {"micro_probe": {}}
    entries: list[dict[str, str]] = []
    for idx, (cid, src) in enumerate((("baseline", spec.baseline_probe_source), ("optimized", spec.optimized_probe_source))):
        sp = req_dir / "candidates" / "micro_probe" / f"{idx:03d}_{cid}.py"
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(src, encoding="utf-8")
        sp.chmod(0o444)
        rel, dig = sp.relative_to(root).as_posix(), _sha256_file(sp)
        staged["micro_probe"][cid] = {"module_path": rel, "sha256": dig}
        entries.append({"candidate_id": cid, "module_path": rel, "sha256": dig})
    payload: dict[str, object] = {
        "schema": FAST_REQUEST_SCHEMA, "request_id": req_id, "mode": "micro_probe", "warmup": 1, "repeat": 1,
        "require_all_passed": False, "response_path": resp_path.relative_to(root).as_posix(), "candidates": entries,
        "probe": {
            "probe_id": spec.probe_id, "probe_kind": spec.probe_kind, "factory_name": spec.factory_name,
            "line_count_baseline": int(spec.line_count_baseline), "line_count_optimized": int(spec.line_count_optimized),
            "tensor_shape": [int(d) for d in spec.tensor_shape], "tensor_dtype": spec.tensor_dtype, "min_speedup": float(min_speedup),
        },
    }
    raw_resp, _ = _hook("_submit_fast_gate", _submit_fast_gate)(
        agent_gates_bin=ag_bin, root=root, workspace=_hook("_agent_gates_workspace_name", _agent_gates_workspace_name)(root),
        request_id=req_id, request_relative_path=req_path.relative_to(root).as_posix(), request_path=req_path,
        response_path=resp_path, request_sha256=_write_fast_request(req_path, payload), mode="micro_probe",
        timeout_sec=float(timeout_sec), device_id=str(device_id), device_pool=device_pool, env_vals=env_vals,
    )
    _verify_fast_bindings(raw_resp, staged)
    _verify_fast_coverage(raw_resp, expected_candidates=2, expected_msprof_targets=0)
    probe_row = _fast_response_mapping(raw_resp, "micro_probe")
    if probe_row.get("probe_id") != spec.probe_id:
        raise RuntimeError(f"fast micro-probe response probe_id {probe_row.get('probe_id')!r} != {spec.probe_id!r}")
    metrics_raw = probe_row.get("metrics")
    if not isinstance(metrics_raw, Mapping):
        raise RuntimeError("fast micro-probe response metrics must be a mapping")
    metrics = {str(k): float(v) for k, v in metrics_raw.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    if not metrics:
        raise RuntimeError("fast micro-probe response carried no numeric metrics")
    return metrics


__all__ = [
    "LocalHardwareProbePort", "MicroProbeSources", "_prewarm_micro_probe_cache",
    "bottleneck_anchor_for_regime", "collect_bottleneck_msprof_via_agent_gates",
    "collect_p2_deep_dive_via_agent_gates", "ensure_cann_include_cpath",
    "run_micro_probe_via_agent_gates", "run_msprof_detail",
]
