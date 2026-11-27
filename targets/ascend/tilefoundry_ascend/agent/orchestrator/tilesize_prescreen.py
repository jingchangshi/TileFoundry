"""Concrete (BM, BN, BK) tile-size dry-run batch prescreener without bisheng."""

from __future__ import annotations

import ast
import math
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from itertools import product

from tilefoundry_ascend.agent.gates.gate_funnel import StageAResult
from tilefoundry_ascend.ascend.cost_model.cost_gate_policy import is_paradigm_shift_exempt
from tilefoundry_ascend.ascend.cost_model.hybrid_cost_model import (
    CostFeatures, CostPrediction, HybridAscendCostModel, build_cost_features_from_ledger,
)
from tilefoundry_ascend.ascend.sil.schema import TopologyAxes
from tilefoundry_ascend.ascend.target import AscendSpaceFacts, AscendTarget
from tilefoundry_ascend.ascend.target.facts import (
    ascend_space_facts, cube_macs_per_cycle as _default_cube_macs_per_cycle,
)


def _align512(nbytes: int) -> int:
    return ((nbytes + 511) // 512) * 512 if nbytes > 0 else 0


@dataclass(frozen=True)
class TileConfigCandidate:
    """Candidate tile shape and buffering schedule before lowering."""

    bm: int
    bn: int
    bk: int
    stages: int = 2
    dtype_bytes: int = 2
    accum_bytes: int = 4
    aiv_split: int = 2
    d_out: int = 0
    dual_accumulator: bool = False
    dtype: str = ""
    hf32: bool = False
    l1_half_bank_split: bool = True
    stack_bytes: int = 0


@dataclass(frozen=True)
class PrescreenScore:
    """Hardware feasibility and zero-parameter Roofline score for a tile config."""

    candidate: TileConfigCandidate
    is_feasible: bool
    l1_bytes: int
    l0a_bytes: int
    l0b_bytes: int
    l0c_bytes: int
    ub_bytes_per_aiv: int
    arithmetic_intensity: float
    fixpipe_to_cube_ratio: float
    score: float
    rejection_reason: str | None = None
    cube_macs_per_cycle: int = 4096
    half_l1_penalty: float = 0.0
    t_roof_us: float = 0.0
    tail_waste_ratio: float = 0.0


@dataclass(frozen=True)
class CostRankedCandidate:
    """Stage-A feasible candidate scored by unified HybridAscendCostModel."""

    candidate_id: str
    pred_us: float
    tier0_physics_us: float
    tier1_penalty_us: float
    dominant_bound: str
    topology_bucket: str
    ledger: object | None = None
    parent_relative_ratio: float = 1.0
    coarse_reject_exempt: bool = False


@dataclass(frozen=True)
class LatticeSweepCandidate:
    """Zero-LLM Stage-A evaluated tile lattice candidate and predicted cost."""

    knob_overrides: dict[str, int]
    patched_source: str
    pred_us: float
    dominant_bound: str
    ub_bytes: int
    l1_bytes: int
    t_roof_us: float = 0.0
    tail_waste_ratio: float = 0.0


@dataclass(frozen=True)
class _LiveIrRecord:
    is_ok: bool
    sort_key: tuple[float, ...]
    cand: TileConfigCandidate
    status: str
    scope_bytes: dict[str, int]
    degraded_cnt: int
    ii_val: int | None
    burst_eff: float
    est_waves: int
    wave_burst_score: float
    cost_pred: CostPrediction | None
    cost_score: float


def _resolve_candidate_dtype(cand: TileConfigCandidate, explicit_dtype: str | None) -> str:
    if explicit_dtype and explicit_dtype.strip():
        return explicit_dtype.strip().lower()
    if cand.dtype and cand.dtype.strip():
        return cand.dtype.strip().lower()
    return "fp8e4m3" if cand.dtype_bytes == 1 else ("f32" if cand.dtype_bytes == 4 else "f16")


def _infer_dtype_bytes(dtype_str: str) -> int:
    low = dtype_str.lower()
    return 4 if ("32" in low and "bf" not in low and "hf" not in low) else 2


def _resolve_cube_macs(facts: AscendSpaceFacts, dtype_name: str, *, hf32: bool) -> int:
    if hasattr(facts, "cube_macs_per_cycle"):
        try:
            return int(facts.cube_macs_per_cycle(dtype_name, hf32=hf32))
        except Exception:
            pass
    return int(_default_cube_macs_per_cycle(dtype_name, hf32=hf32))


def _compute_l0c_bytes(cand: TileConfigCandidate) -> int:
    if cand.d_out > 0 or cand.dual_accumulator:
        return _align512(cand.bm * (cand.d_out or cand.bn) * cand.accum_bytes) + _align512(
            cand.bm * cand.bn * cand.accum_bytes
        )
    return cand.bm * cand.bn * cand.accum_bytes


def _compute_tile_roofline_and_waste(
    bm: int, bn: int, bk: int, *, dtype_bytes: int, accum_bytes: int, l0c_bytes: int,
    facts: AscendSpaceFacts, macs_per_cycle: int, m_dim: int = 0, n_dim: int = 0, k_dim: int = 0,
) -> tuple[float, float]:
    _ = accum_bytes
    l0a_slot, l0b_slot = bm * bk * dtype_bytes, bk * bn * dtype_bytes
    clock_ghz = float(getattr(facts, "clock_ghz", 1.65) or 1.65)
    hbm_bps = float(getattr(facts, "hbm_peak_bandwidth_bps", 0) or 0)
    fixpipe_bpc = max(1, int(getattr(facts, "fixpipe_bytes_per_cycle", 128) or 128))
    num_aic = max(1, int(getattr(facts, "num_aic_cores", 28) or 28))

    if m_dim > 0 and n_dim > 0 and k_dim > 0:
        tm, tn, tk = math.ceil(m_dim / bm), math.ceil(n_dim / bn), math.ceil(k_dim / bk)
        padded_vol = tm * bm * tn * bn * tk * bk
        tail_waste = max(0.0, float(padded_vol - m_dim * n_dim * k_dim) / float(max(1, padded_vol)))
        padded_mem = float(tm * tn * tk * (l0a_slot + l0b_slot) + tm * tn * l0c_bytes)
        t_comp = max(float(2 * padded_vol) / float(num_aic * max(1, macs_per_cycle)), float(tm * tn * l0c_bytes) / float(num_aic * fixpipe_bpc)) / (clock_ghz * 1e3)
        t_mem = (padded_mem / hbm_bps) * 1e6 if hbm_bps > 0.0 else 0.0
        return max(t_comp, t_mem), tail_waste

    flops = float(2 * bm * bn * bk)
    t_comp = max(flops / float(max(1, macs_per_cycle)), float(l0c_bytes) / float(fixpipe_bpc)) / (clock_ghz * 1e3)
    per_core_bps = hbm_bps / float(num_aic)
    t_mem = (float(l0a_slot + l0b_slot + l0c_bytes) / per_core_bps) * 1e6 if per_core_bps > 0.0 else 0.0
    return (max(t_comp, t_mem) / max(1.0, flops)) * 1e6, 0.0


def _reject_score(cand: TileConfigCandidate, reason: str, macs_per_cycle: int) -> PrescreenScore:
    return PrescreenScore(
        candidate=cand, is_feasible=False, l1_bytes=0, l0a_bytes=0, l0b_bytes=0, l0c_bytes=0,
        ub_bytes_per_aiv=0, arithmetic_intensity=0.0, fixpipe_to_cube_ratio=0.0, score=0.0,
        rejection_reason=reason, cube_macs_per_cycle=macs_per_cycle, half_l1_penalty=0.0,
    )


def evaluate_tile_candidate(
    cand: TileConfigCandidate, facts: AscendSpaceFacts, *, simtvf: bool = False,
    dtype: str | None = None, hf32: bool | None = None, l1_half_bank_split: bool | None = None,
    m_dim: int = 0, n_dim: int = 0, k_dim: int = 0, stack_bytes: int | None = None,
) -> PrescreenScore:
    """Evaluate hard physical feasibility and zero-parameter Roofline lower bound."""
    _ = l1_half_bank_split
    eff_dtype = _resolve_candidate_dtype(cand, dtype)
    eff_hf32 = cand.hf32 if hf32 is None else bool(hf32)
    macs = max(1, _resolve_cube_macs(facts, eff_dtype, hf32=eff_hf32))
    eff_stack = int(cand.stack_bytes if stack_bytes is None else stack_bytes)
    if cand.bm <= 0 or cand.bn <= 0 or cand.bk <= 0 or cand.stages < 1 or cand.aiv_split < 1 or cand.dtype_bytes <= 0 or cand.accum_bytes <= 0:
        return _reject_score(cand, "non_positive_tile_dimension", macs)
    if cand.bm % 16 != 0 or cand.bn % 16 != 0 or cand.bk % 16 != 0:
        return _reject_score(cand, "unaligned_mmad_16x16x16_atom", macs)

    l0a_slot, l0b_slot = cand.bm * cand.bk * cand.dtype_bytes, cand.bk * cand.bn * cand.dtype_bytes
    l0a_bytes, l0b_bytes = cand.stages * l0a_slot, cand.stages * l0b_slot
    l1_bytes = l0a_bytes + l0b_bytes
    l0c_bytes = _compute_l0c_bytes(cand)
    ub_width = max(cand.bn, cand.d_out) if (cand.d_out > 0 or cand.dual_accumulator) else cand.bn
    ub_bytes = cand.stages * max(1, (cand.bm + cand.aiv_split - 1) // cand.aiv_split) * ub_width * cand.accum_bytes
    ub_limit = facts.ub_limit_bytes("simtvf" if simtvf else "simdvf")
    reason = next((
        f"{tag}:{val}>{lim}"
        for cond, tag, val, lim in (
            (eff_stack > 0, "stack_overflow", eff_stack, 0),
            (l1_bytes > facts.l1_bytes, "L1_overflow", l1_bytes, facts.l1_bytes),
            (l0a_bytes > facts.l0a_bytes, "L0A_overflow", l0a_bytes, facts.l0a_bytes),
            (l0b_bytes > facts.l0b_bytes, "L0B_overflow", l0b_bytes, facts.l0b_bytes),
            (l0c_bytes > facts.l0c_bytes, "L0C_overflow", l0c_bytes, facts.l0c_bytes),
            (ub_bytes > ub_limit, "UB_overflow", ub_bytes, ub_limit),
        ) if cond
    ), None)
    flops, mem = float(2 * cand.bm * cand.bn * cand.bk), float(l0a_slot + l0b_slot + l0c_bytes)
    fp_ratio = (float(l0c_bytes) / float(max(1, facts.fixpipe_bytes_per_cycle))) / max(1.0, flops / float(macs))
    t_roof_us, tail_waste = _compute_tile_roofline_and_waste(
        cand.bm, cand.bn, cand.bk, dtype_bytes=cand.dtype_bytes, accum_bytes=cand.accum_bytes,
        l0c_bytes=l0c_bytes, facts=facts, macs_per_cycle=macs, m_dim=m_dim, n_dim=n_dim, k_dim=k_dim,
    )
    score = (1.0 / max(1e-12, t_roof_us)) * max(0.0, 1.0 - tail_waste)
    return PrescreenScore(
        candidate=cand, is_feasible=(reason is None), l1_bytes=l1_bytes, l0a_bytes=l0a_bytes,
        l0b_bytes=l0b_bytes, l0c_bytes=l0c_bytes, ub_bytes_per_aiv=ub_bytes,
        arithmetic_intensity=(flops / mem if mem > 0.0 else 0.0), fixpipe_to_cube_ratio=fp_ratio,
        score=score if reason is None else 0.0, rejection_reason=reason, cube_macs_per_cycle=macs,
        t_roof_us=t_roof_us, tail_waste_ratio=tail_waste,
    )


def prescreen_tile_configs(
    candidates: Iterable[TileConfigCandidate], *, target: AscendTarget | None = None,
    simtvf: bool = False, top_k: int | None = None, dtype: str | None = None,
    hf32: bool | None = None, l1_half_bank_split: bool | None = None,
    m_dim: int = 0, n_dim: int = 0, k_dim: int = 0, cost_model_mode: str = "off",
) -> list[PrescreenScore]:
    """Filter tile shapes by hard physical feasibility and rank by Roofline and tail waste."""
    _ = cost_model_mode
    facts = (target or AscendTarget("ascend.950pr")).get_facts(AscendSpaceFacts)
    feasible = [
        item for c in candidates
        if (item := evaluate_tile_candidate(
            c, facts, simtvf=simtvf, dtype=dtype, hf32=hf32,
            l1_half_bank_split=l1_half_bank_split, m_dim=m_dim, n_dim=n_dim, k_dim=k_dim,
        )).is_feasible
    ]
    feasible.sort(key=lambda it: (round(it.t_roof_us, 12), round(it.tail_waste_ratio, 12), it.candidate.bm, it.candidate.bn, it.candidate.bk, it.candidate.stages))
    return feasible[:top_k] if (top_k is not None and top_k >= 0) else feasible


def _aligned_axis_values(
    value: int, step: int, *, max_val: int | None = None, require_even_fractals: bool = False
) -> tuple[int, ...]:
    raw = (value - step, value, value + step) if value % step == 0 else ((value // step) * step, (value // step) * step + step)
    return tuple(dict.fromkeys(
        c for c in raw if c > 0 and c % 16 == 0 and (not require_even_fractals or (c // 16) % 2 == 0) and (max_val is None or c <= max_val)
    ))


def generate_aligned_neighborhood(
    center: TileConfigCandidate, *, m_max: int | None = None, n_max: int | None = None, k_max: int | None = None
) -> tuple[TileConfigCandidate, ...]:
    """Expand +/-1 fractal-aligned (bm, bn, bk, stages) neighbors around a tile config."""
    bm_vals = _aligned_axis_values(center.bm, 32 if center.aiv_split == 2 else 16, max_val=m_max, require_even_fractals=(center.aiv_split == 2))
    bn_vals = _aligned_axis_values(center.bn, 16, max_val=n_max)
    bk_vals = _aligned_axis_values(center.bk, 16, max_val=k_max)
    stage_vals = tuple(dict.fromkeys(st for st in (center.stages - 1, center.stages, center.stages + 1) if st >= 1))
    return tuple(replace(center, bm=bm, bn=bn, bk=bk, stages=st) for bm, bn, bk, st in product(bm_vals, bn_vals, bk_vals, stage_vals))


def count_dsl_gemm_calls(dsl_source: str) -> int:
    """Count AST gemm call sites in a Python TileLang DSL snippet."""
    try:
        tree = ast.parse(dsl_source)
    except SyntaxError:
        return 0
    return sum(
        1 for n in ast.walk(tree) if isinstance(n, ast.Call)
        and ((isinstance(n.func, ast.Attribute) and n.func.attr == "gemm") or (isinstance(n.func, ast.Name) and n.func.id == "gemm"))
    )


def inspect_builder_tile_params(dsl_source: str, factory_name: str | None) -> tuple[set[str], bool]:
    """Return top-level builder parameter names and whether it accepts **kwargs."""
    if not factory_name:
        return set(), False
    try:
        tree = ast.parse(dsl_source)
    except SyntaxError:
        return set(), False
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == factory_name:
            return {a.arg for a in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)}, node.args.kwarg is not None
    return set(), False


def _bounded_live_ir_variants(
    center: TileConfigCandidate, *, tune_m: bool, tune_n: bool, tune_k: bool,
    tune_stages: bool, max_variants: int, pure_vector_row_tile: bool = False,
) -> list[TileConfigCandidate]:
    variants = [center]
    seen = {(center.bm, center.bn, center.bk, center.stages)}

    def _add(cand: TileConfigCandidate) -> None:
        if len(variants) < max_variants and (k := (cand.bm, cand.bn, cand.bk, cand.stages)) not in seen:
            seen.add(k)
            variants.append(cand)

    if pure_vector_row_tile:
        pool = (2, 4, 6, 8, 16, 1, 32) if center.bm <= 4 else (8, 16, 24, 32, 4, 12, 2, 1)
        bm_vals = (center.bm, *[m for m in pool if m != center.bm])
        stage_vals = (center.stages, *[s for s in (1, 2, 3) if tune_stages and s != center.stages]) if tune_stages else (center.stages,)
        for st, bm_c in product(stage_vals, bm_vals):
            _add(replace(center, bm=bm_c, stages=st))
        return variants

    if tune_m and center.bm >= 64:
        step = 16 * max(1, center.aiv_split)
        if (half_bm := ((center.bm // 2) // step) * step) >= 16:
            _add(replace(center, bm=half_bm))
    if tune_m and center.bm in (32, 64):
        _add(replace(center, bm=center.bm * 2))

    ordered = sorted(
        generate_aligned_neighborhood(center),
        key=lambda c: (abs(c.bm - center.bm) + abs(c.bn - center.bn) + abs(c.bk - center.bk) + 32 * abs(c.stages - center.stages), c.bm, c.bn),
    )
    for cand in ordered:
        _add(replace(
            center, bm=cand.bm if tune_m else center.bm, bn=cand.bn if tune_n else center.bn,
            bk=cand.bk if tune_k else center.bk, stages=cand.stages if tune_stages else center.stages,
        ))
    return variants


def _apply_tile_candidate_to_kwargs(
    base_kwargs: Mapping[str, object] | None, cand: TileConfigCandidate, *, params: set[str], has_varkw: bool
) -> dict[str, object]:
    kw = dict(base_kwargs or {})
    for keys, val, fb in (
        (("bm", "tile_m", "block_m", "tile_rows", "row_tile"), cand.bm, "bm"),
        (("bn", "tile_n", "block_n"), cand.bn, "bn"), (("bk", "tile_k", "block_k"), cand.bk, None), (("num_stages", "stages"), cand.stages, None),
    ):
        if hit := next((k for k in keys if k in params), None):
            kw[hit] = val
        elif has_varkw and fb is not None:
            kw[fb] = val
    return kw


def rank_stage_a_candidates_with_cost_model(
    candidates: Sequence[tuple[str, str, StageAResult]], *, cost_model: HybridAscendCostModel,
    live_plan: object | None = None, axes: TopologyAxes | None = None, blueprint: object | None = None,
    sil_advisory: object | None = None, parent_ledger: object | None = None,
    parent_features: CostFeatures | None = None, creative_candidate_ids: Collection[str] | None = None,
    parent_workspace_stages: int = 0,
) -> list[CostRankedCandidate]:
    """Rank Stage-A passers by parent-relative delta cost, ascending, with stable ties."""
    parent_cf = parent_features or (build_cost_features_from_ledger(parent_ledger, live_plan=live_plan, axes=axes, blueprint=blueprint, sil_advisory=sil_advisory, target=cost_model.target) if parent_ledger is not None else None)
    parent_pred = cost_model.predict(parent_cf) if parent_cf is not None else None
    raw_p_axes = axes if isinstance(axes, TopologyAxes) else getattr(parent_ledger, "topology_axes", None)
    eff_p_axes = raw_p_axes if isinstance(raw_p_axes, TopologyAxes) else None
    creative_ids = frozenset(str(cid) for cid in (creative_candidate_ids or ()))
    scored: list[tuple[float, int, CostRankedCandidate]] = []
    for idx, (cand_id, _src, stage_a) in enumerate(candidates):
        ledger = getattr(stage_a, "ledger", None)
        raw_c_axes = getattr(ledger, "topology_axes", None)
        cand_axes = raw_c_axes if isinstance(raw_c_axes, TopologyAxes) else eff_p_axes
        features = build_cost_features_from_ledger(ledger, live_plan=live_plan, axes=cand_axes, blueprint=blueprint, sil_advisory=sil_advisory, target=cost_model.target)
        pred = cost_model.predict(features)
        rel_ratio = float(cost_model.predict_delta_ratio(features, parent_cf)) if (parent_pred is not None and parent_cf is not None) else 1.0
        score = float(parent_pred.pred_us) * rel_ratio if (parent_pred is not None and parent_cf is not None) else float(pred.pred_us)
        exempt = is_paradigm_shift_exempt(
            cand_axes, eff_p_axes, cand_workspace_stages=int(getattr(ledger, "workspace_stages", parent_workspace_stages) or 0),
            parent_workspace_stages=int(parent_workspace_stages), is_creative=str(cand_id) in creative_ids,
        )
        item = CostRankedCandidate(
            candidate_id=str(cand_id), pred_us=score, tier0_physics_us=float(pred.tier0_physics_us),
            tier1_penalty_us=float(pred.tier1_penalty_us), dominant_bound=str(pred.dominant_bound),
            topology_bucket=str(pred.topology_bucket), ledger=ledger, parent_relative_ratio=rel_ratio, coarse_reject_exempt=exempt,
        )
        scored.append((item.pred_us, idx, item))
    scored.sort(key=lambda e: (e[0], e[1]))
    return [e[2] for e in scored]


def _extract_problem_mnk(var_bindings: Mapping[str, int] | None, dims: Mapping[str, object] | None) -> tuple[int, int, int]:
    found = {"m": 0, "n": 0, "k": 0}
    for src in (var_bindings, dims):
        if not isinstance(src, Mapping):
            continue
        for key, tgt in (("M", "m"), ("m_dim", "m"), ("m_rows", "m"), ("total_rows", "m"), ("N", "n"), ("n_dim", "n"), ("K", "k"), ("k_dim", "k")):
            if found[tgt] == 0 and isinstance(raw := src.get(key), int) and not isinstance(raw, bool) and raw > 0:
                found[tgt] = int(raw)
    return found["m"], found["n"], found["k"]


def _reject_live_ir(idx: int, cand: TileConfigCandidate, msg: str) -> _LiveIrRecord:
    return _LiveIrRecord(False, (float("inf"), 1.0, float(idx)), cand, f"REJECT({msg})", {}, 0, None, 1.0, 0, 0.0, None, 0.0)


def _compute_vector_roofline_us(ledger: object, *, clock_ghz: float, hbm_bps: float, cycles: int) -> float:
    t_comp = float(cycles) / (clock_ghz * 1e3)
    gm = float(ledger.total_gm_bytes() if hasattr(ledger, "total_gm_bytes") else getattr(ledger, "gm_roundtrip_bytes", 0))
    t_mem = (gm / hbm_bps) * 1e6 if hbm_bps > 0.0 else 0.0
    return max(t_comp, t_mem)


def _ledger_overflow_or_illegal(ledger: object, stage_a: object) -> str | None:
    waterlines = getattr(ledger, "waterlines", None) or {}
    if hit := next(((s, wl.peak_allocated_bytes, wl.capacity_bytes) for s in ("L0C", "UB", "L1", "L0A", "L0B", "SSBUF", *sorted(waterlines)) if (wl := waterlines.get(s)) is not None and wl.overflow_bytes > 0), None):
        return f"{hit[0]}_overflow:{hit[1]}>{hit[2]}"
    if int(getattr(ledger, "max_overflow_bytes", 0) or 0) > 0:
        return "ledger_overflow"
    if (sb := max(int(getattr(ledger, "stack_bytes", 0) or 0), int(getattr(stage_a, "stack_bytes", 0) or 0))) > 0:
        return f"stack_overflow:{sb}>0"
    if deg := [b for wl in waterlines.values() for b in wl.degraded_single_buffers]:
        return f"degraded_single_buffers:{','.join(deg)}"
    if not getattr(stage_a, "passed", True) or not getattr(ledger, "is_legal", True):
        return str(getattr(stage_a, "error_message", None) or "illegal_schedule")
    return None


def prescreen_live_ir_neighborhood(
    dsl_source: str, *, factory_name: str, center: TileConfigCandidate,
    base_kwargs: Mapping[str, object] | None = None, var_bindings: Mapping[str, int] | None = None,
    config_selector_name: str | None = None, pass_configs: Mapping[str, object] | None = None,
    compile_flags: Sequence[str] | str | None = None, dims: Mapping[str, object] | None = None,
    target: AscendTarget | None = None, max_variants: int = 8, cost_model: HybridAscendCostModel | None = None,
    cost_model_mode: str | None = None, axes: TopologyAxes | None = None, live_plan: object | None = None,
    blueprint: object | None = None, sil_advisory: object | None = None,
    ordered_tensor_shapes: Sequence[Sequence[int]] | None = None,
) -> str:
    """Run bounded Stage A live-IR dry-runs across aligned tile variants around center."""
    eff_cm = None if (cost_model_mode is not None and str(cost_model_mode).strip().lower() == "off") else cost_model
    params, has_varkw = inspect_builder_tile_params(dsl_source, factory_name)
    tune_m = has_varkw or any(k in params for k in ("bm", "tile_m", "block_m", "tile_rows", "row_tile"))
    tune_n = has_varkw or any(k in params for k in ("bn", "tile_n", "block_n"))
    tune_k = has_varkw or any(k in params for k in ("bk", "tile_k", "block_k"))
    tune_stages = has_varkw or "stages" in params or "num_stages" in params
    if not (tune_m or tune_n or tune_k or tune_stages):
        return ""
    pure_vec = count_dsl_gemm_calls(dsl_source) == 0 and any(k in params for k in ("tile_m", "tile_rows", "row_tile")) and "bm" not in params and "block_m" not in params

    from tilefoundry_ascend.agent.gates.gate_funnel import (  # noqa: PLC0415
        _merge_pass_configs_and_flags, _resolve_prim_func_and_configs_from_dsl, run_stage_a_static,
    )

    sel_pc: dict[str, object] = {}
    sel_cf: list[str] = []
    if config_selector_name and pass_configs is None and compile_flags is None:
        try:
            _pf, sel_pc, sel_cf = _resolve_prim_func_and_configs_from_dsl(
                dsl_source, factory_name=factory_name, factory_kwargs=base_kwargs, config_selector_name=config_selector_name, dims=dims
            )
        except Exception:
            sel_pc, sel_cf = {}, []
    merged_pc, merged_cf = _merge_pass_configs_and_flags(sel_pc, sel_cf, pass_configs, compile_flags)
    variants = _bounded_live_ir_variants(
        center, tune_m=tune_m, tune_n=tune_n, tune_k=tune_k, tune_stages=tune_stages,
        max_variants=max(1, min(8, int(max_variants))), pure_vector_row_tile=pure_vec,
    )
    total_rows = 0
    for src in (var_bindings, dims):
        if isinstance(src, Mapping):
            if isinstance(rm := src.get("m_rows", src.get("total_rows", src.get("M", src.get("m_dim")))), int) and not isinstance(rm, bool) and rm > 0:
                total_rows = int(rm)
                break
            bv, sv = src.get("B", src.get("b_dim", src.get("batch"))), src.get("S", src.get("s_dim", src.get("S_q", src.get("seq_len"))))
            if isinstance(bv, int) and isinstance(sv, int) and not isinstance(bv, bool) and not isinstance(sv, bool) and bv > 0 and sv > 0:
                total_rows = int(bv) * int(sv)
                break
    facts = ascend_space_facts(target or AscendTarget())
    num_aiv = max(1, int(facts.num_aiv_cores))
    m_dim, n_dim, k_dim = _extract_problem_mnk(var_bindings, dims)
    clock_ghz, hbm_bps = float(getattr(facts, "clock_ghz", 1.65) or 1.65), float(getattr(facts, "hbm_peak_bandwidth_bps", 0) or 0)
    records: list[_LiveIrRecord] = []
    any_ledger, parent_features, parent_pred = False, None, None

    for idx, cand in enumerate(variants):
        if cand.bm <= 0 or cand.bn <= 0 or cand.bk <= 0 or cand.stages < 1 or cand.aiv_split < 1 or cand.dtype_bytes <= 0 or cand.accum_bytes <= 0:
            records.append(_reject_live_ir(idx, cand, "non_positive_tile_dimension"))
            continue
        if (not pure_vec) and (cand.bm % 16 != 0 or cand.bn % 16 != 0 or cand.bk % 16 != 0):
            records.append(_reject_live_ir(idx, cand, "unaligned_mmad_16x16x16_atom"))
            continue
        res = run_stage_a_static(
            dsl_source, factory_name=factory_name,
            factory_kwargs=_apply_tile_candidate_to_kwargs(base_kwargs, cand, params=params, has_varkw=has_varkw),
            target=target, pass_configs=merged_pc or None, compile_flags=merged_cf or None,
            dims=dims, var_bindings=var_bindings, ordered_tensor_shapes=ordered_tensor_shapes,
        )
        if (ledger := res.ledger) is None:
            records.append(_reject_live_ir(idx, cand, res.error_message or "stage_a_failed"))
            continue
        any_ledger = True
        scope_bytes = {s: ledger.waterlines[s].peak_allocated_bytes for s in ("L1", "L0A", "L0B", "L0C", "UB", "SSBUF") if s in ledger.waterlines}
        degraded_cnt = sum(len(wl.degraded_single_buffers) for wl in ledger.waterlines.values())
        rej = _ledger_overflow_or_illegal(ledger, res)
        status, is_ok = ("OK", True) if rej is None else (f"REJECT({rej})", False)
        dom_cycles = ledger.pipe_cycles.get(ledger.dominant_pipe, max((v for v in ledger.pipe_cycles.values() if v > 0), default=0))
        burst_eff = float(getattr(ledger, "dma_burst_efficiency_ratio", 1.0) or 1.0)
        est_waves = math.ceil(math.ceil(total_rows / cand.bm) / float(num_aiv)) if (pure_vec and total_rows > 0 and cand.bm > 0) else 0
        if pure_vec:
            padded_rows = math.ceil(total_rows / cand.bm) * cand.bm if (total_rows > 0 and cand.bm > 0) else 0
            tail_waste = max(0.0, float(padded_rows - total_rows) / float(max(1, padded_rows))) if padded_rows > 0 else 0.0
            t_roof_us = _compute_vector_roofline_us(ledger, clock_ghz=clock_ghz, hbm_bps=hbm_bps, cycles=max(1, est_waves) * dom_cycles)
            sort_key: tuple[float, ...] = (t_roof_us, tail_waste, float(dom_cycles), float(idx))
        else:
            macs = max(1, _resolve_cube_macs(facts, _resolve_candidate_dtype(cand, None), hf32=cand.hf32))
            t_roof_us, tail_waste = _compute_tile_roofline_and_waste(
                cand.bm, cand.bn, cand.bk, dtype_bytes=cand.dtype_bytes, accum_bytes=cand.accum_bytes,
                l0c_bytes=_compute_l0c_bytes(cand), facts=facts, macs_per_cycle=macs, m_dim=m_dim, n_dim=n_dim, k_dim=k_dim,
            )
            sort_key = (t_roof_us, tail_waste, float(dom_cycles) / float(max(1, cand.bm * cand.bn * cand.bk)), float(idx))
        cost_pred, cost_score = None, 0.0
        if eff_cm is not None:
            cf = build_cost_features_from_ledger(ledger, live_plan=live_plan, axes=axes, blueprint=blueprint, sil_advisory=sil_advisory, target=eff_cm.target)
            cost_pred = eff_cm.predict(cf)
            cost_score = float(parent_pred.pred_us) * float(eff_cm.predict_delta_ratio(cf, parent_features)) if (parent_pred is not None and parent_features is not None) else float(cost_pred.pred_us)
            sort_key = (cost_score, *sort_key)
            if idx == 0:
                parent_features, parent_pred = cf, cost_pred
        records.append(_LiveIrRecord(is_ok, sort_key, cand, status, scope_bytes, degraded_cnt, ledger.steady_state_ii, burst_eff, est_waves, t_roof_us, cost_pred, cost_score))

    if not any_ledger:
        return ""
    ordered = [*sorted((r for r in records if r.is_ok), key=lambda r: r.sort_key)[:4], *[r for r in records if not r.is_ok][:4]]
    if not ordered:
        return ""
    lines = ["Live-IR Tile Prescreen Summary:"]
    for rank_idx, rec in enumerate(ordered, start=1):
        extra = (f", ii={rec.ii_val}" if rec.ii_val is not None else "") + (f", degraded={rec.degraded_cnt}" if rec.degraded_cnt > 0 else "")
        cost_sfx = f", pred_us={rec.cost_score:.2f}us ({rec.cost_pred.dominant_bound})" if rec.cost_pred is not None else ""
        if pure_vec:
            wls = ", ".join(f"{s.lower()}={rec.scope_bytes.get(s, 0)}B" for s in ("UB", "SSBUF"))
            v_ext = f", dma_burst_eff={rec.burst_eff:.3f}" + (f", est_waves={rec.est_waves}, wave_burst_score={rec.wave_burst_score:.2f}" if rec.est_waves > 0 else "")
            lines.append(f"  {rank_idx}. tile_m={rec.cand.bm}, stages={rec.cand.stages} | {rec.status} | {wls}{extra}{v_ext}{cost_sfx}")
        else:
            wls = ", ".join(f"{s.lower()}={rec.scope_bytes.get(s, 0)}B" for s in ("L1", "L0A", "L0B", "L0C", "UB", "SSBUF"))
            lines.append(f"  {rank_idx}. bm={rec.cand.bm}, bn={rec.cand.bn}, bk={rec.cand.bk}, stages={rec.cand.stages} | {rec.status} | {wls}{extra}{cost_sfx}")
    return "\n".join(lines)


def format_prescreen_summary(scores: Sequence[PrescreenScore]) -> str:
    """Format ranked PrescreenScore entries into a concise prompt-ready summary."""
    if not scores:
        return "No feasible tile configurations found."
    lines = ["Tile Prescreen Summary:"]
    for idx, item in enumerate(scores, start=1):
        c = item.candidate
        status = "OK" if item.is_feasible else f"REJECT({item.rejection_reason or 'invalid'})"
        d_out_part = f", d_out={c.d_out}" if c.d_out > 0 else ""
        lines.append(
            f"  {idx}. bm={c.bm}, bn={c.bn}, bk={c.bk}, stages={c.stages}, aiv_split={c.aiv_split}{d_out_part} | {status} | "
            f"l1={item.l1_bytes}B, l0c={item.l0c_bytes}B, ub_aiv={item.ub_bytes_per_aiv}B | "
            f"ai={item.arithmetic_intensity:.2f}, fp_ratio={item.fixpipe_to_cube_ratio:.2f}, score={item.score:.3f}"
        )
    return "\n".join(lines)


def _extract_selector_base_config(
    module_source: str, selector_fn_name: str, sample_builder_kwargs: Mapping[str, object], dims: Mapping[str, int] | None
) -> dict[str, object]:
    tree = ast.parse(module_source)
    from tilefoundry_ascend.agent.gates.gate_funnel import (  # noqa: PLC0415
        _filter_kwargs_for_callable, _merge_dims_into_kwargs,
    )
    merged_kwargs = _merge_dims_into_kwargs(sample_builder_kwargs, dims)
    ns: dict[str, object] = {}
    try:
        exec(compile(tree, "<lattice_selector_probe>", "exec"), ns)
        if callable(sel_fn := ns.get(selector_fn_name)) and isinstance(sel_out := sel_fn(**_filter_kwargs_for_callable(sel_fn, merged_kwargs)), Mapping):
            return {str(k): v for k, v in sel_out.items()}
    except Exception:
        pass
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name != selector_fn_name:
            continue
        local_ints = {k: int(v) for k, v in merged_kwargs.items() if isinstance(v, int) and not isinstance(v, bool)}
        for stmt in node.body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name) and isinstance(stmt.value, ast.Constant) and isinstance(stmt.value.value, int) and not isinstance(stmt.value.value, bool):
                local_ints[stmt.targets[0].id] = int(stmt.value.value)
        for inner in ast.walk(node):
            if isinstance(inner, ast.Return) and isinstance(inner.value, ast.Dict):
                parsed = {
                    kn.value: (vn.value if isinstance(vn, ast.Constant) else local_ints[vn.id])
                    for kn, vn in zip(inner.value.keys, inner.value.values)
                    if isinstance(kn, ast.Constant) and isinstance(kn.value, str) and (isinstance(vn, ast.Constant) or (isinstance(vn, ast.Name) and vn.id in local_ints))
                }
                if parsed:
                    return parsed
    return {}


def _generate_lattice_override_grid(base_cfg: Mapping[str, object], facts: AscendSpaceFacts, max_points: int) -> list[dict[str, int]]:
    cap, has_stages = max(1, int(max_points)), "num_stages" in base_cfg
    if has_stages:
        st_b = int(base_cfg["num_stages"]) if isinstance(base_cfg["num_stages"], int) else 1
        st_opts: tuple[int | None, ...] = tuple(dict.fromkeys((st_b, 1, 2)))
    else:
        st_b, st_opts = 1, (None,)
    if all(k in base_cfg for k in ("bm", "bn", "bk")):
        dtype_str = str(base_cfg.get("dtype") or base_cfg.get("tl_dtype") or "f16").lower()
        dtype_b = _infer_dtype_bytes(dtype_str)
        bm_b, bn_b, bk_b = (int(base_cfg[k]) if isinstance(base_cfg[k], int) else d for k, d in (("bm", 64), ("bn", 128), ("bk", 64)))
        scored: list[tuple[int, float, dict[str, int]]] = []
        for bm, bn, bk, st in product(
            tuple(dict.fromkeys((bm_b, 32, 64, 128))), tuple(dict.fromkeys((bn_b, 64, 128, 256))),
            tuple(dict.fromkeys((bk_b, 32, 64, 128))), st_opts,
        ):
            if bm <= 0 or bn <= 0 or bk <= 0 or bm % 16 or bn % 16 or bk % 16:
                continue
            ev = evaluate_tile_candidate(TileConfigCandidate(bm=bm, bn=bn, bk=bk, stages=st or 1, dtype_bytes=dtype_b, accum_bytes=4), facts, dtype=dtype_str)
            if ev.is_feasible and ev.l1_bytes <= facts.l1_bytes and ev.l0c_bytes <= facts.l0c_bytes:
                ov = {"bm": int(bm), "bn": int(bn), "bk": int(bk), **({"num_stages": int(st)} if st is not None else {})}
                scored.append((0 if (bm, bn, bk, st) == (bm_b, bn_b, bk_b, st_b if has_stages else None) else 1, -ev.score, ov))
        scored.sort(key=lambda item: (item[0], item[1]))
        return [item[2] for item in scored[:cap]]
    if "chunk_t" in base_cfg:
        ct_b = int(base_cfg["chunk_t"]) if isinstance(base_cfg["chunk_t"], int) else 32
        dv_b = int(base_cfg["dv_tile"]) if ("dv_tile" in base_cfg and isinstance(base_cfg["dv_tile"], int)) else 32
        dv_opts = tuple(dict.fromkeys((dv_b, 32, 64))) if "dv_tile" in base_cfg else (None,)
        return [{"chunk_t": int(ct), **({"dv_tile": int(dv)} if dv is not None else {}), **({"num_stages": int(st)} if st is not None else {})}
                for ct, dv, st in product(tuple(dict.fromkeys((ct_b, 16, 32, 64))), dv_opts, st_opts)][:cap]
    if "tile_m" in base_cfg:
        tm_b = int(base_cfg["tile_m"]) if isinstance(base_cfg["tile_m"], int) else 4
        return [{"tile_m": int(tm), **({"num_stages": int(st)} if st is not None else {})}
                for tm, st in product(tuple(dict.fromkeys((tm_b, 1, 2, 4, 8, 16, 32))), st_opts)][:cap]
    return []


def _patch_selector_source(module_source: str, selector_fn_name: str, overrides: Mapping[str, int]) -> str:
    target_node = next((n for n in ast.parse(module_source).body if isinstance(n, ast.FunctionDef) and n.name == selector_fn_name), None)
    if target_node is None or target_node.end_lineno is None:
        raise ValueError(f"Selector function {selector_fn_name!r} not found at module top level")
    lines = module_source.splitlines(keepends=True)
    insert_at, alias = int(target_node.end_lineno), f"_tf_sweep_orig_{selector_fn_name}"
    wrapper = (
        f"\nimport functools as _tf_sweep_functools\n{alias} = {selector_fn_name}\n"
        f"@_tf_sweep_functools.wraps({alias})\ndef {selector_fn_name}(*args, **kwargs):\n"
        f"    _c = dict({alias}(*args, **kwargs)); _c.update({repr({str(k): int(v) for k, v in overrides.items()})}); return _c\n"
    )
    if insert_at > 0 and not lines[insert_at - 1].endswith("\n"):
        lines[insert_at - 1] += "\n"
    patched = "".join([*lines[:insert_at], wrapper, *lines[insert_at:]])
    ast.parse(patched)
    return patched


def _static_dominant_bound_from_ledger(ledger: object) -> str:
    dom_pipe = str(getattr(ledger, "dominant_pipe", "") or "").upper()
    if any(t in dom_pipe for t in ("MTE2", "MTE3", "DMA")):
        return "memory_bound"
    if any(t in dom_pipe for t in ("PIPE_M", "CUBE", "PIPE_V", "VEC", "FIX")):
        return "compute_bound"
    if isinstance(pipes := getattr(ledger, "pipe_cycles", None), Mapping):
        mte_c = sum(int(v or 0) for k, v in pipes.items() if int(v or 0) > 0 and any(t in str(k).upper() for t in ("MTE2", "MTE3", "DMA")))
        comp_c = sum(int(v or 0) for k, v in pipes.items() if int(v or 0) > 0 and any(t in str(k).upper() for t in ("PIPE_M", "CUBE", "PIPE_V", "VEC", "FIX")))
        return "memory_bound" if mte_c > comp_c else ("compute_bound" if comp_c > 0 else "balanced")
    return "balanced"


def sweep_executable_lattice_zero_llm(
    module_source: str, *, selector_fn_name: str, builder_fn_name: str,
    sample_builder_kwargs: Mapping[str, object], dims: Mapping[str, int] | None,
    cost_model: HybridAscendCostModel | None = None, cost_model_mode: str | None = None,
    target: AscendTarget | None = None, max_points: int = 24, top_k: int = 3,
) -> list[LatticeSweepCandidate]:
    """Sweep bounded tile lattice with Stage-A feasibility and Roofline ranking."""
    if top_k <= 0 or max_points <= 0:
        return []
    eff_cm = None if (cost_model_mode is not None and str(cost_model_mode).strip().lower() == "off") else cost_model
    eff_target = target or (eff_cm.target if eff_cm is not None else AscendTarget("ascend.950pr"))
    eff_facts = eff_cm.facts if (eff_cm is not None and target is None) else eff_target.get_facts(AscendSpaceFacts)
    if not (base_cfg := _extract_selector_base_config(module_source, selector_fn_name, sample_builder_kwargs, dims)):
        return []
    if not (override_grid := _generate_lattice_override_grid(base_cfg, eff_facts, max_points)):
        return []

    from tilefoundry_ascend.agent.gates.gate_funnel import (  # noqa: PLC0415
        run_stage_a_static,
    )

    m_dim, n_dim, k_dim = _extract_problem_mnk(None, dims)
    clock_ghz, hbm_bps = float(getattr(eff_facts, "clock_ghz", 1.65) or 1.65), float(getattr(eff_facts, "hbm_peak_bandwidth_bps", 0) or 0)
    ranked: list[tuple[tuple[float, ...], LatticeSweepCandidate]] = []
    for idx, ov in enumerate(override_grid):
        if any(int(v) <= 0 for v in ov.values()) or (all(k in ov for k in ("bm", "bn", "bk")) and (ov["bm"] % 16 != 0 or ov["bn"] % 16 != 0 or ov["bk"] % 16 != 0)):
            continue
        try:
            patched_source = _patch_selector_source(module_source, selector_fn_name, ov)
        except (SyntaxError, ValueError):
            continue
        stage_a = run_stage_a_static(
            patched_source, factory_name=builder_fn_name, factory_kwargs={**sample_builder_kwargs, **ov},
            target=eff_target, config_selector_name=selector_fn_name, dims=dims,
        )
        ledger = stage_a.ledger
        if ledger is None or _ledger_overflow_or_illegal(ledger, stage_a) is not None:
            continue
        waterlines = getattr(ledger, "waterlines", None) or {}
        ub_bytes = int(getattr(waterlines.get("UB"), "peak_allocated_bytes", 0) or 0)
        l1_bytes = int(getattr(waterlines.get("L1"), "peak_allocated_bytes", 0) or 0)
        pipes = getattr(ledger, "pipe_cycles", None) or {}
        bottleneck_cycles = max((int(val or 0) for val in pipes.values()), default=0) if isinstance(pipes, Mapping) else 0
        t_roof_us = tail_waste = 0.0
        if all(k in ov for k in ("bm", "bn", "bk")):
            dtype_str = str(base_cfg.get("dtype") or base_cfg.get("tl_dtype") or "f16").lower()
            macs = max(1, _resolve_cube_macs(eff_facts, dtype_str, hf32=False))
            t_roof_us, tail_waste = _compute_tile_roofline_and_waste(
                int(ov["bm"]), int(ov["bn"]), int(ov["bk"]), dtype_bytes=_infer_dtype_bytes(dtype_str),
                accum_bytes=4, l0c_bytes=int(ov["bm"]) * int(ov["bn"]) * 4, facts=eff_facts,
                macs_per_cycle=macs, m_dim=m_dim, n_dim=n_dim, k_dim=k_dim,
            )
        else:
            t_roof_us = _compute_vector_roofline_us(ledger, clock_ghz=clock_ghz, hbm_bps=hbm_bps, cycles=bottleneck_cycles)
            if m_dim > 0 and (tile_m := ov.get("tile_m") or ov.get("chunk_t") or 0) > 0:
                padded_m = math.ceil(m_dim / int(tile_m)) * int(tile_m)
                tail_waste = max(0.0, float(padded_m - m_dim) / float(max(1, padded_m)))
        if eff_cm is not None:
            policy_hint: dict[str, object] = {**base_cfg, **ov}
            if all(k in ov for k in ("bm", "bn", "bk")):
                policy_hint["tile_mnk"] = [ov["bm"], ov["bn"], ov["bk"]]
            if "num_stages" in ov:
                policy_hint["l1_stages"] = ov["num_stages"]
            pred = eff_cm.predict(build_cost_features_from_ledger(ledger, live_plan={"policy": policy_hint}, target=eff_target))
            pred_us, dom_bound = float(pred.pred_us), str(pred.dominant_bound)
            rank_key: tuple[float, ...] = (pred_us, float(ub_bytes), float(idx))
        else:
            pred_us, dom_bound = 0.0, _static_dominant_bound_from_ledger(ledger)
            rank_key = (round(t_roof_us, 12), round(tail_waste, 12), float(bottleneck_cycles), float(idx))
        ranked.append((rank_key, LatticeSweepCandidate(
            knob_overrides=dict(ov), patched_source=patched_source, pred_us=pred_us, dominant_bound=dom_bound,
            ub_bytes=ub_bytes, l1_bytes=l1_bytes, t_roof_us=t_roof_us, tail_waste_ratio=tail_waste,
        )))
    return [e[1] for e in sorted(ranked, key=lambda e: e[0])[: int(top_k)]]


__all__ = [
    "CostRankedCandidate", "LatticeSweepCandidate", "PrescreenScore", "TileConfigCandidate",
    "count_dsl_gemm_calls", "evaluate_tile_candidate", "format_prescreen_summary",
    "generate_aligned_neighborhood", "inspect_builder_tile_params", "prescreen_live_ir_neighborhood",
    "prescreen_tile_configs", "rank_stage_a_candidates_with_cost_model", "sweep_executable_lattice_zero_llm",
]
