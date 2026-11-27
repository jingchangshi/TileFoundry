"""Consolidated L5 campaign helpers eliminating duplicated coordinator utilities."""

from __future__ import annotations

import hashlib, json, math
from collections.abc import Mapping, Sequence
from dataclasses import fields
from pathlib import Path

from tilefoundry_ascend.agent.campaign.regime_slice_patcher import (
    _active_trackb_kernel, _ordered_top_level_trackb_kernels,
    _render_trackb_tu_from_module, _trackb_kernel_names, resolve_regime_config,
)
from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (
    AnchorEvalMetrics, CandidateSurrogateResult, extract_ordered_tensor_shapes,
)
from tilefoundry_ascend.agent.orchestrator.cpp_slice_protocol import select_active_trackb_kernel
from tilefoundry_ascend.agent.orchestrator.gate_funnel import StageAResult, evaluate_ascendc_candidate, run_stage_a_static
from tilefoundry_ascend.agent.specs.anchor_specs import (
    AnchorCaseSpec, CannbenchOpsRegistry, PerfTargetSpec, RegimeTargetSpec,
    _has_flat_operator_layout, build_anchor_var_bindings, extract_trackb_cpp_source,
    get_operator_entrypoint, lookup_case_dims, parse_simple_env_file, resolve_cannbench_repo,
    resolve_cannbench_tasks_root as _resolve_tasks_from_repo,
)
from tilefoundry_ascend.agent.specs.latency_algebra import AnchorConservationPoint, is_regime_saturated, regime_bandit_ucb_us
from tilefoundry_ascend.ascend.json_record import atomic_write_text
from tilefoundry_ascend.ascend.passes.physical_ledger import ObjectivePhysicalLedger
from tilefoundry_ascend.ascend.sil.op_analyzer import resolve_canonical_topology_axes
from tilefoundry_ascend.ascend.sil.pytorch_frontend import PyTorchOpSemantics, PyTorchSemanticAnalyzer
from tilefoundry_ascend.ascend.sil.schema import ConstraintRelaxationMask, TopologyAxes
from tilefoundry_ascend.ascend.sil.verifier import verify_source_against_facets
from tilefoundry_ascend.ascend.target import AscendTarget


def source_sha256(source: str) -> str:
    """Return the hex SHA-256 digest of UTF-8 source text."""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def write_source_and_verify(path: Path, source: str, *, expected_sha256: str | None = None) -> str:
    """Atomically write *source* to *path* and verify SHA-256 matches."""
    expected = expected_sha256 or source_sha256(source)
    atomic_write_text(path, source)
    try:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        return f"restored module {path} cannot be read back: {exc}"
    return "" if actual == expected else f"restored module {path} digest {actual} != {expected}"


def ensure_candidate_bytes(*, path: Path, source: str, expected_sha256: str) -> str:
    """Restore *source* to *path* if missing or if its digest drifted."""
    expected = expected_sha256 or source_sha256(source)
    try:
        current = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        current = ""
    return "" if current == expected else write_source_and_verify(path, source, expected_sha256=expected)


def persist_runtime_state(checkpoint_path: Path | None, key: str, value: Mapping[str, object] | None) -> None:
    """Upsert a recoverable execution record into *checkpoint_path* JSON."""
    if checkpoint_path is None:
        return
    payload: dict[str, object] = {}
    if checkpoint_path.is_file():
        try:
            loaded = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            payload = dict(loaded) if isinstance(loaded, Mapping) else {}
        except (OSError, ValueError):
            payload = {}
    payload[key] = dict(value) if value is not None else None
    atomic_write_text(checkpoint_path, json.dumps(payload, indent=2) + "\n")


def extract_valid_non_negative_float(value: object) -> float | None:
    """Return finite non-negative float from *value*, else None."""
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)) and float(value) >= 0.0 else None


def extract_cost_pred_us(pred: object) -> float:
    """Extract scalar microsecond prediction from a float or cost prediction packet."""
    if isinstance(pred, (int, float)) and not isinstance(pred, bool):
        return float(pred)
    if pred is None:
        return 0.0
    cal = float(getattr(pred, "pred_us", 0.0) or 0.0)
    return cal if cal > 0.0 else float(getattr(pred, "tier0_physics_us", 0.0) or 0.0) + float(getattr(pred, "tier1_penalty_us", 0.0) or 0.0)


def surrogate_rank_key(res: CandidateSurrogateResult) -> tuple[float, float, float]:
    """Return lexicographic ranking key for passing surrogate results."""
    return (res.surrogate_score, res.min_case_speedup, res.avg_speedup)


def resolve_cann_root_from_env_file(env_file: str | Path | None) -> Path | None:
    """Return CANN install root derived from env.sh setup script path."""
    raw = parse_simple_env_file(Path(env_file)).get("TILEFOUNDRY_NPU_CANN_SETUP", "").strip() if (env_file and Path(env_file).is_file()) else ""
    parent = Path(raw).expanduser().parent if raw else None
    return parent if (parent is not None and parent.is_dir()) else None


def resolve_operator_entrypoint_safe(operator_name: str) -> str:
    """Resolve operator entrypoint name, returning empty string when unknown."""
    try:
        return get_operator_entrypoint(operator_name)
    except (KeyError, FileNotFoundError):
        return ""


def resolve_cannbench_tasks_root(*, cannbench_repo: Path | None = None, env_file: Path | None = None, source_root: Path | None = None) -> Path | None:
    """Resolve cann-bench tasks directory, falling back to in-repo benchmark tasks."""
    try:
        return _resolve_tasks_from_repo(resolve_cannbench_repo(cannbench_repo=cannbench_repo, env_file=env_file))
    except FileNotFoundError:
        in_repo = source_root / "tests" / "ascend" / "benchmarks" if source_root is not None else None
        return in_repo if (in_repo is not None and _has_flat_operator_layout(in_repo)) else None


def lookup_case_dims_safe(
    operator_name: str, case_num: int, *, regime_tag: str = "", regime_specs: Sequence[RegimeTargetSpec] = (),
    cache: dict[int, dict[str, object]] | None = None, cannbench_repo: Path | None = None, env_file: Path | None = None,
) -> dict[str, object]:
    """Look up case dimensions with cache and representative-regime fallback."""
    cnum = int(case_num)
    if cache is not None and cnum in cache:
        return dict(cache[cnum])
    dims: dict[str, object] = {}
    try:
        dims = dict(lookup_case_dims(operator_name, cnum, cannbench_repo=cannbench_repo, env_file=env_file))
    except (FileNotFoundError, OSError, KeyError, ValueError, TypeError):
        if (hit := next((s for s in regime_specs if regime_tag and s.regime_tag == regime_tag), regime_specs[0] if regime_specs else None)) is not None:
            dims = dict(hit.representative_anchor_dims or hit.representative_hir_dims)
    if cache is not None:
        cache[cnum] = dict(dims)
    return dims


def resolve_operator_semantics(
    operator_name: str, *, source_root: Path, pure_golden: bool = False, tasks_root: Path | None = None, cannbench_repo: Path | None = None, env_file: Path | None = None,
) -> tuple[PyTorchOpSemantics | None, Mapping[str, object] | None]:
    """Load PyTorch golden semantics and proto metadata for *operator_name*."""
    if (eff_root := tasks_root or resolve_cannbench_tasks_root(cannbench_repo=cannbench_repo, env_file=env_file, source_root=source_root)) is None:
        return None, None
    try:
        spec = CannbenchOpsRegistry(tasks_root=eff_root, source_root=source_root, pure_golden=bool(pure_golden)).resolve(operator_name)
    except (FileNotFoundError, KeyError, ValueError):
        return None, None
    proto = spec.proto if isinstance(spec.proto, Mapping) else None
    try:
        return (PyTorchSemanticAnalyzer().analyze_source(spec.golden_path.read_text(encoding="utf-8"), entry_name=spec.entrypoint, proto=spec.proto) if spec.golden_path.is_file() else None), proto
    except (OSError, SyntaxError, ValueError):
        return None, proto


def derive_topology_axes(
    operator_name: str, ledger: ObjectivePhysicalLedger | None, *, topology_axes: TopologyAxes | None = None,
    semantic_topology_axes: TopologyAxes | None = None, sil_advisory: object | None = None, features: object | None = None, route: object | None = None,
) -> TopologyAxes:
    """Resolve canonical TopologyAxes from explicit overrides, route, or ledger."""
    if topology_axes is not None or semantic_topology_axes is not None:
        return topology_axes or semantic_topology_axes  # type: ignore[return-value]
    if sil_advisory is not None:
        return resolve_canonical_topology_axes(sil_advisory, features=features)
    if route is not None:
        ra = route.get("topology_axes") if isinstance(route, Mapping) else getattr(route, "topology_axes", None)
        if isinstance(ra, TopologyAxes):
            return ra
        if isinstance(ra, Mapping) and {"compute_engine", "memory_ring", "iteration_regularity"} <= set(ra):
            return TopologyAxes.from_dict(ra)
    uses_cube = bool(ledger is not None and (ledger.pipe_cycles.get("AIC.PIPE_M", 0) > 0 or ledger.pipe_cycles.get("AIC.CUBE", 0) > 0))
    return resolve_canonical_topology_axes({"op_name": operator_name, "uses_cube": uses_cube}, features=features)


def resolve_active_trackb_kernel(module_source: str, regime_spec: RegimeTargetSpec, *, cache: dict[tuple[str, str], str | None] | None = None) -> str | None:
    """Resolve launched Track-B kernel name from instantiation truth or raw TU."""
    if regime_spec.runtime_backend not in ("ascendc_trackb", "hybrid_trackb") or (ext := extract_trackb_cpp_source(module_source, regime_spec.trackb_symbols)) is None:
        return None
    ckey = (source_sha256(module_source), regime_spec.regime_tag)
    if cache is not None and ckey in cache:
        return cache[ckey]
    rendered = _render_trackb_tu_from_module(module_source, ext[1], regime_spec, require_renderer=False)
    if rendered == ext[1]:
        if (res := select_active_trackb_kernel(ext[1], regime_tag=regime_spec.regime_tag, module_source=module_source)) is None:
            raise ValueError(f"Track-B source for regime {regime_spec.regime_tag!r} contains no __global__ kernel")
    else:
        try:
            res = _active_trackb_kernel(rendered, ext[1], regime_tag=regime_spec.regime_tag, module_source=module_source)
        except ValueError as exc:
            if "no unique top-level launch target" not in str(exc) or not (ordered := _ordered_top_level_trackb_kernels(rendered, ext[1], _trackb_kernel_names(ext[1]))):
                raise
            tagged = select_active_trackb_kernel(ext[1], regime_tag=regime_spec.regime_tag, module_source=module_source)
            res = tagged if tagged in ordered else ordered[0]
    if cache is not None:
        cache[ckey] = res
    return res


def resolve_active_trackb_kernels(module_source: str, regime_spec: RegimeTargetSpec) -> tuple[str, ...]:
    """Resolve ordered active Track-B ``__global__`` kernels for single or multi-stage TUs."""
    if regime_spec.runtime_backend not in ("ascendc_trackb", "hybrid_trackb") or (ext := extract_trackb_cpp_source(module_source, regime_spec.trackb_symbols)) is None:
        return ()
    ordered = _ordered_top_level_trackb_kernels(_render_trackb_tu_from_module(module_source, ext[1], regime_spec, require_renderer=False), ext[1], _trackb_kernel_names(ext[1]))
    if ordered:
        return ordered
    single = resolve_active_trackb_kernel(module_source, regime_spec)
    return (single,) if single else ()


def resolve_trackb_freedom_mask(
    regime_spec: RegimeTargetSpec, *, freedom_masks: Mapping[str, ConstraintRelaxationMask] | None = None,
    baseline_source: str = "", target_module_path: Path | None = None,
) -> ConstraintRelaxationMask:
    """Return active 7-facet mask for *regime_spec*, seeding Track-B baseline facets once."""
    if freedom_masks and regime_spec.regime_tag in freedom_masks:
        return freedom_masks[regime_spec.regime_tag]
    bsrc = baseline_source or (target_module_path.read_text(encoding="utf-8") if target_module_path and target_module_path.is_file() else "")
    btb = extract_trackb_cpp_source(bsrc, regime_spec.trackb_symbols) if (bsrc and regime_spec.runtime_backend in ("ascendc_trackb", "hybrid_trackb")) else None
    if btb is not None and not (rep := verify_source_against_facets(_render_trackb_tu_from_module(bsrc, btb[1], regime_spec, require_renderer=False), ConstraintRelaxationMask.bottom())).passed and rep.suggested_fixes:
        f = rep.suggested_fixes
        return ConstraintRelaxationMask(
            alg_compute_and_monoid=True, tile_and_stage_knobs=True,
            exec_mode_and_spr=bool(f.get("unlock_exec_mode_and_spr", False)), sram_layout_and_aliasing=bool(f.get("unlock_sram_layout_and_aliasing", False)),
            dma_and_cache_hierarchy=bool(f.get("unlock_dma_and_cache_hierarchy", False)), pipeline_and_sync_graph=bool(f.get("unlock_pipeline_and_sync_graph", False)),
            grid_and_workspace_topology=bool(f.get("unlock_grid_and_workspace_topology", False)),
        )
    return ConstraintRelaxationMask.bottom()


def run_l0_gate(
    module_source: str, regime_spec: RegimeTargetSpec, *, target: AscendTarget,
    enable_track_router: bool = False, freedom_masks: Mapping[str, ConstraintRelaxationMask] | None = None,
    active_incumbent_source: str = "", target_module_path: Path | None = None,
    anchor_dims_override: Mapping[str, object] | None = None, op_semantics: PyTorchOpSemantics | None = None,
    op_proto: Mapping[str, object] | None = None, baseline_tb_tu: str | None = None,
) -> StageAResult:
    """Run unified Stage-A L0 static/waterline verification for Track-A, Track-B, or hybrid."""
    req_dma, req_cube = bool(op_semantics is not None and not op_semantics.has_indirect_index), bool(op_semantics is not None and op_semantics.has_gemm_contraction)
    is_hybrid = regime_spec.runtime_backend == "hybrid_trackb"
    tb_ext = extract_trackb_cpp_source(module_source, regime_spec.trackb_symbols) if regime_spec.runtime_backend in ("ascendc_trackb", "hybrid_trackb") else None
    mask = resolve_trackb_freedom_mask(regime_spec, freedom_masks=freedom_masks, baseline_source=active_incumbent_source, target_module_path=target_module_path)

    def _eval_tb() -> StageAResult:
        assert tb_ext is not None
        tu = _render_trackb_tu_from_module(module_source, tb_ext[1], regime_spec, require_renderer=False, anchor_dims_override=anchor_dims_override)
        return evaluate_ascendc_candidate(tu, freedom_mask=mask, target=target, enable_track_router=enable_track_router, case_ratios=regime_spec.representative_case_ratios, baseline_cpp_source=baseline_tb_tu, require_hardware_dma=req_dma, require_cube_engine=req_cube, module_source=module_source, regime_tag=regime_spec.regime_tag)
    if regime_spec.runtime_backend == "ascendc_trackb" and tb_ext is not None:
        return _eval_tb()
    dims = dict(anchor_dims_override or regime_spec.representative_anchor_dims or regime_spec.representative_hir_dims)
    try:
        resolved = resolve_regime_config(module_source, regime_spec, dims=dims)
    except (SyntaxError, AttributeError, ImportError, KeyError, NameError, RuntimeError, TypeError, ValueError) as exc:
        return StageAResult(passed=False, prim_func=None, ledger=None, error_message=f"stage_a_config_resolve_error: {type(exc).__name__}: {exc}")
    primary = run_stage_a_static(
        module_source, factory_name=resolved.builder_name, factory_kwargs=resolved.builder_kwargs,
        pass_configs=resolved.pass_configs or None, compile_flags=resolved.compile_flags or None,
        target=target, var_bindings=build_anchor_var_bindings(dims, kwargs=resolved.builder_kwargs) or None,
        enable_track_router=enable_track_router, case_ratios=regime_spec.representative_case_ratios,
        require_hardware_dma=req_dma, require_cube_engine=bool(req_cube and not (is_hybrid and tb_ext is not None)),
        ordered_tensor_shapes=extract_ordered_tensor_shapes(case_entry=regime_spec.representative_case_entry, dims=dims, proto=op_proto),
    )
    return _eval_tb() if (not primary.passed and is_hybrid and tb_ext is not None) else primary


def slice_contract_instruction(func_name: str, regime_spec: RegimeTargetSpec, *, operator_entrypoint: str = "") -> str:
    """Format slice scope instructions for *func_name* and *regime_spec*."""
    warn = " CRITICAL: the public operator entrypoint is named 'kernel', so you MUST NOT define a top-level '@T.prim_func def kernel' or 'def kernel'; name any inner prim_func differently (e.g., '_ar_kernel_impl') and return it from the builder." if operator_entrypoint == "kernel" else ""
    if not regime_spec.editable_symbols:
        return f"function '{func_name}'{warn}"
    hnote = f" or '{regime_spec.allowed_helper_prefix}*' helpers" if regime_spec.allowed_helper_prefix else ""
    return (
        f"authorized slice symbols ({', '.join(regime_spec.editable_symbols)}){hnote} for entrypoint '{func_name}'. "
        "You may output ONLY the subset of functions or prefixed macros you modified (e.g., just _select_*_config, a single helper macro, "
        f"or one builder); omitted sibling symbols in this regime automatically inherit their parent implementation. Preserve exact public entrypoint '{func_name}' and `kernel = {func_name}` without renaming or modifying symbols from other regimes{warn}"
    )


def to_conservation_point(cand: AnchorEvalMetrics, inc: AnchorEvalMetrics | None) -> AnchorConservationPoint:
    """Convert candidate and incumbent anchor metrics into an AnchorConservationPoint."""
    inc_us = float(inc.cand_us) if (inc is not None and inc.cand_us > 0.0) else float(cand.cand_us)
    is_paired = float(getattr(cand, "paired_speedup_vs_incumbent", 0.0) or 0.0) > 0.0
    cmad = float(getattr(cand, "paired_rel_mad", 0.0) or 0.0)
    kw: dict[str, object] = {
        "case_num": int(cand.case_num), "regime_tag": str(cand.regime_tag), "ref_us": float(cand.ref_perf_us),
        "inc_us": inc_us, "cand_us": float(cand.cand_us), "t_roof_us": float(cand.t_hw_us),
        "paired_rel_mad": cmad if is_paired else max(cmad, float(getattr(inc, "paired_rel_mad", 0.0) or 0.0) if inc else 0.0),
        "accuracy_passed": bool(cand.accuracy_passed),
    }
    if {"cand_burst_cv", "ref_burst_cv", "burst_count"} <= {f.name for f in fields(AnchorConservationPoint)}:
        kw.update({"cand_burst_cv": float(cand.burst_batch_cv), "ref_burst_cv": (0.0 if is_paired else (float(inc.burst_batch_cv) if inc else 0.0)), "burst_count": max(1, int(cand.ring_burst_k))})
    return AnchorConservationPoint(**kw)  # type: ignore[arg-type]


def is_regime_satisfied(perf_target: PerfTargetSpec | None, regime_spec: RegimeTargetSpec | str, best_result: CandidateSurrogateResult, active_anchors: Sequence[AnchorCaseSpec]) -> bool:
    """Return True when all active anchors in *regime_spec* meet *perf_target*."""
    if perf_target is None:
        return False
    tag = regime_spec if isinstance(regime_spec, str) else regime_spec.regime_tag
    nums = {a.case_num for a in active_anchors if a.regime_tag == tag}
    matched = [a for a in best_result.anchors if a.case_num in nums]
    return bool(matched) and all(m.accuracy_passed and m.roofline_eta >= perf_target.target_roofline_eta and m.speedup >= perf_target.target_min_case_speedup and (perf_target.per_anchor_max_us.get(m.case_num) is None or m.cand_us <= perf_target.per_anchor_max_us[m.case_num]) for m in matched)


def regime_mean_recoverable_excess_us(best_result: CandidateSurrogateResult, regime_spec: RegimeTargetSpec | str, *, dispatch_floor_us: float = 0.0) -> float | None:
    """Compute mean recoverable silicon excess (us) above *dispatch_floor_us*."""
    tag = regime_spec if isinstance(regime_spec, str) else regime_spec.regime_tag
    matched = [a for a in best_result.anchors if a.regime_tag == tag and a.cand_us > 0.0]
    return (sum(to_conservation_point(a, a).recoverable_sched_us(float(dispatch_floor_us or 0.0)) for a in matched) / len(matched)) if matched else None


def select_active_regime(
    regime_specs: Sequence[RegimeTargetSpec], perf_target: PerfTargetSpec | None, current_regime: str | None,
    frozen_regimes: set[str], best_result: CandidateSurrogateResult, active_anchors: Sequence[AnchorCaseSpec], *,
    allow_satisfied_fallback: bool = False, exclude_regime: str | None = None,
    stagnation_by_regime: Mapping[str, int] | None = None, compile_wall_streak_by_regime: Mapping[str, int] | None = None,
    last_round_promoted_by_regime: Mapping[str, bool] | None = None, last_round_promoted: bool = True, dispatch_floor_us: float = 0.0,
) -> RegimeTargetSpec | None:
    """Select the highest-urgency unfrozen regime using bandit UCB and headroom."""
    specs, by_tag, wall_map, stag_map = tuple(regime_specs), {r.regime_tag: r for r in regime_specs}, compile_wall_streak_by_regime or {}, stagnation_by_regime or {}
    unfrozen_n = sum(1 for s in specs if s.regime_tag not in frozen_regimes)
    elig = [s for s in specs if s.regime_tag not in frozen_regimes and (unfrozen_n <= 1 or int(wall_map.get(s.regime_tag, 0) or 0) < 3)]
    cands = [s for s in elig if not is_regime_satisfied(perf_target, s, best_result, active_anchors)] or (elig if allow_satisfied_fallback else [])
    if not cands:
        return None
    if exclude_regime and any(s.regime_tag != exclude_regime for s in cands):
        cands = [s for s in cands if s.regime_tag != exclude_regime]
    floor = float(dispatch_floor_us or 0.0)
    pts_map = {sp.regime_tag: [to_conservation_point(a, a) for a in best_result.anchors if ((int(a.case_num) in {int(x.case_num) for x in active_anchors if x.regime_tag == sp.regime_tag}) if any(x.regime_tag == sp.regime_tag for x in active_anchors) else (a.regime_tag == sp.regime_tag)) and a.cand_us > 0.0] for sp in cands}
    sat_map = {t: is_regime_saturated(pts, floor, sigma_mult=2.0) for t, pts in pts_map.items()}
    unsat_map = {t: bool(pts) and not sat_map[t] for t, pts in pts_map.items()}
    _dep = lambda sp: 1 if sat_map.get(sp.regime_tag, False) and any(t != sp.regime_tag and u for t, u in unsat_map.items()) else 0

    def _key(spec: RegimeTargetSpec) -> tuple[int, int, int, float, float, float, str]:
        pts, stag = pts_map.get(spec.regime_tag, ()), max(0, int(stag_map.get(spec.regime_tag, 0) or 0))
        rec, headroom = regime_mean_recoverable_excess_us(best_result, spec, dispatch_floor_us=floor), max(0.0, float(best_result.regime_headroom(spec.regime_tag)))
        ucb = regime_bandit_ucb_us(pts, floor, stag) if pts else (headroom / math.sqrt(1.0 + stag))
        return (1 if int(wall_map.get(spec.regime_tag, 0) or 0) >= 2 else 0, _dep(spec), -(1 if rec is None or rec > 0.0 else 0), -ucb, -headroom, -(rec or 0.0), spec.regime_tag)

    cands.sort(key=_key)
    top_spec = cands[0]
    was_promoted = (last_round_promoted_by_regime or {}).get(current_regime, last_round_promoted) if current_regime else last_round_promoted
    if current_regime and current_regime in {s.regime_tag for s in cands} and current_regime in by_tag and int(wall_map.get(current_regime, 0) or 0) < 2 and was_promoted and _dep(by_tag[current_regime]) == 0:
        tpts, cpts = pts_map.get(top_spec.regime_tag, ()), pts_map.get(current_regime, ())
        if tpts and cpts and regime_bandit_ucb_us(tpts, floor, 0) - regime_bandit_ucb_us(cpts, floor, 0) <= 2.0 * math.sqrt(sum(p.sigma_paired_us() ** 2 for p in cpts)):
            return by_tag[current_regime]
    return top_spec
