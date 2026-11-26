"""Concrete (BM, BN, BK) tile-size dry-run batch prescreener without bisheng."""

from __future__ import annotations

import ast
import math
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product

from tilefoundry_ascend.ascend.cost_model.calibration_backtest import (
    PrescreenAblationReport,
    PrescreenAblationSample,
    evaluate_prescreen_ablation,
)
from tilefoundry_ascend.ascend.cost_model.cost_gate_policy import is_paradigm_shift_exempt
from tilefoundry_ascend.ascend.cost_model.hybrid_cost_model import (
    CostFeatures,
    CostPrediction,
    HybridAscendCostModel,
    build_cost_features_from_ledger,
)
from tilefoundry_ascend.ascend.sil.schema import TopologyAxes
from tilefoundry_ascend.ascend.target import AscendSpaceFacts, AscendTarget
from tilefoundry_ascend.ascend.target.facts import ascend_space_facts
from tilefoundry_ascend.ascend.target.facts import (
    cube_macs_per_cycle as _default_cube_macs_per_cycle,
)


def _align512(nbytes: int) -> int:
    if nbytes <= 0:
        return 0
    return ((nbytes + 511) // 512) * 512


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


def _resolve_candidate_dtype(cand: TileConfigCandidate, explicit_dtype: str | None) -> str:
    if explicit_dtype and explicit_dtype.strip():
        return explicit_dtype.strip().lower()
    if cand.dtype and cand.dtype.strip():
        return cand.dtype.strip().lower()
    if cand.dtype_bytes == 1:
        return "fp8e4m3"
    if cand.dtype_bytes == 4:
        return "f32"
    return "f16"


def _resolve_cube_macs(facts: AscendSpaceFacts, dtype_name: str, *, hf32: bool) -> int:
    if hasattr(facts, "cube_macs_per_cycle"):
        try:
            return int(facts.cube_macs_per_cycle(dtype_name, hf32=hf32))
        except Exception:
            pass
    return int(_default_cube_macs_per_cycle(dtype_name, hf32=hf32))


def _compute_tile_roofline_and_waste(
    bm: int,
    bn: int,
    bk: int,
    *,
    dtype_bytes: int,
    accum_bytes: int,
    l0c_bytes: int,
    facts: AscendSpaceFacts,
    macs_per_cycle: int,
    m_dim: int = 0,
    n_dim: int = 0,
    k_dim: int = 0,
) -> tuple[float, float]:
    """Compute theoretical Roofline lower bound (us) and tail-tile waste ratio."""
    l0a_slot = bm * bk * dtype_bytes
    l0b_slot = bk * bn * dtype_bytes
    flops = float(2 * bm * bn * bk)
    mem_traffic = float(l0a_slot + l0b_slot + l0c_bytes)
    clock_ghz = float(getattr(facts, "clock_ghz", 1.65) or 1.65)
    hbm_bps = float(getattr(facts, "hbm_peak_bandwidth_bps", 0) or 0)
    fixpipe_bpc = max(1, int(getattr(facts, "fixpipe_bytes_per_cycle", 128) or 128))
    num_aic = max(1, int(getattr(facts, "num_aic_cores", 28) or 28))

    if m_dim > 0 and n_dim > 0 and k_dim > 0:
        tiles_m = math.ceil(m_dim / bm)
        tiles_n = math.ceil(n_dim / bn)
        tiles_k = math.ceil(k_dim / bk)
        padded_vol = tiles_m * bm * tiles_n * bn * tiles_k * bk
        actual_vol = m_dim * n_dim * k_dim
        tail_waste_ratio = max(0.0, float(padded_vol - actual_vol) / float(max(1, padded_vol)))
        padded_flops = float(2 * padded_vol)
        padded_mem = float(
            tiles_m * tiles_n * tiles_k * (l0a_slot + l0b_slot)
            + tiles_m * tiles_n * l0c_bytes
        )
        cube_cycles = padded_flops / float(num_aic * max(1, macs_per_cycle))
        fixpipe_cycles = float(tiles_m * tiles_n * l0c_bytes) / float(num_aic * fixpipe_bpc)
        t_comp_us = max(cube_cycles, fixpipe_cycles) / (clock_ghz * 1e3)
        t_mem_us = (padded_mem / hbm_bps) * 1e6 if hbm_bps > 0.0 else 0.0
        return max(t_comp_us, t_mem_us), tail_waste_ratio

    _ = accum_bytes
    per_core_hbm_bps = hbm_bps / float(num_aic)
    cube_cycles = flops / float(max(1, macs_per_cycle))
    fixpipe_cycles = float(l0c_bytes) / float(fixpipe_bpc)
    t_tile_comp_us = max(cube_cycles, fixpipe_cycles) / (clock_ghz * 1e3)
    t_tile_mem_us = (mem_traffic / per_core_hbm_bps) * 1e6 if per_core_hbm_bps > 0.0 else 0.0
    t_tile_roof_us = max(t_tile_comp_us, t_tile_mem_us)
    t_roof_per_mflop_us = (t_tile_roof_us / max(1.0, flops)) * 1e6
    return t_roof_per_mflop_us, 0.0


def evaluate_tile_candidate(
    cand: TileConfigCandidate,
    facts: AscendSpaceFacts,
    *,
    simtvf: bool = False,
    dtype: str | None = None,
    hf32: bool | None = None,
    l1_half_bank_split: bool | None = None,
    m_dim: int = 0,
    n_dim: int = 0,
    k_dim: int = 0,
    stack_bytes: int | None = None,
) -> PrescreenScore:
    """Evaluate hard physical feasibility and zero-parameter Roofline lower bound."""
    _ = l1_half_bank_split
    eff_dtype = _resolve_candidate_dtype(cand, dtype)
    eff_hf32 = cand.hf32 if hf32 is None else bool(hf32)
    macs_per_cycle = max(1, _resolve_cube_macs(facts, eff_dtype, hf32=eff_hf32))
    eff_stack = int(cand.stack_bytes if stack_bytes is None else stack_bytes)

    if (
        cand.bm <= 0
        or cand.bn <= 0
        or cand.bk <= 0
        or cand.stages < 1
        or cand.aiv_split < 1
        or cand.dtype_bytes <= 0
        or cand.accum_bytes <= 0
    ):
        return PrescreenScore(
            candidate=cand,
            is_feasible=False,
            l1_bytes=0,
            l0a_bytes=0,
            l0b_bytes=0,
            l0c_bytes=0,
            ub_bytes_per_aiv=0,
            arithmetic_intensity=0.0,
            fixpipe_to_cube_ratio=0.0,
            score=0.0,
            rejection_reason="non_positive_tile_dimension",
            cube_macs_per_cycle=macs_per_cycle,
            half_l1_penalty=0.0,
        )
    if cand.bm % 16 != 0 or cand.bn % 16 != 0 or cand.bk % 16 != 0:
        return PrescreenScore(
            candidate=cand,
            is_feasible=False,
            l1_bytes=0,
            l0a_bytes=0,
            l0b_bytes=0,
            l0c_bytes=0,
            ub_bytes_per_aiv=0,
            arithmetic_intensity=0.0,
            fixpipe_to_cube_ratio=0.0,
            score=0.0,
            rejection_reason="unaligned_mmad_16x16x16_atom",
            cube_macs_per_cycle=macs_per_cycle,
            half_l1_penalty=0.0,
        )

    l0a_slot = cand.bm * cand.bk * cand.dtype_bytes
    l0b_slot = cand.bk * cand.bn * cand.dtype_bytes
    l1_bytes = cand.stages * (l0a_slot + l0b_slot)
    l0a_bytes = cand.stages * l0a_slot
    l0b_bytes = cand.stages * l0b_slot
    if cand.d_out > 0 or cand.dual_accumulator:
        out_width = cand.d_out or cand.bn
        l0c_bytes = _align512(cand.bm * out_width * cand.accum_bytes) + _align512(
            cand.bm * cand.bn * cand.accum_bytes
        )
        ub_width = max(cand.bn, cand.d_out)
    else:
        l0c_bytes = cand.bm * cand.bn * cand.accum_bytes
        ub_width = cand.bn
    ub_rows = max(1, (cand.bm + cand.aiv_split - 1) // cand.aiv_split)
    ub_bytes = cand.stages * ub_rows * ub_width * cand.accum_bytes

    ub_limit = facts.ub_limit_bytes("simtvf" if simtvf else "simdvf")
    if eff_stack > 0:
        reason: str | None = f"stack_overflow:{eff_stack}>0"
    elif l1_bytes > facts.l1_bytes:
        reason = f"L1_overflow:{l1_bytes}>{facts.l1_bytes}"
    elif l0a_bytes > facts.l0a_bytes:
        reason = f"L0A_overflow:{l0a_bytes}>{facts.l0a_bytes}"
    elif l0b_bytes > facts.l0b_bytes:
        reason = f"L0B_overflow:{l0b_bytes}>{facts.l0b_bytes}"
    elif l0c_bytes > facts.l0c_bytes:
        reason = f"L0C_overflow:{l0c_bytes}>{facts.l0c_bytes}"
    elif ub_bytes > ub_limit:
        reason = f"UB_overflow:{ub_bytes}>{ub_limit}"
    else:
        reason = None

    flops = float(2 * cand.bm * cand.bn * cand.bk)
    mem_traffic = float(l0a_slot + l0b_slot + l0c_bytes)
    intensity = flops / mem_traffic if mem_traffic > 0.0 else 0.0
    cube_cycles = max(1.0, flops / float(macs_per_cycle))
    fixpipe_cycles = float(l0c_bytes) / float(max(1, facts.fixpipe_bytes_per_cycle))
    fp_ratio = fixpipe_cycles / cube_cycles
    t_roof_us, tail_waste = _compute_tile_roofline_and_waste(
        cand.bm,
        cand.bn,
        cand.bk,
        dtype_bytes=cand.dtype_bytes,
        accum_bytes=cand.accum_bytes,
        l0c_bytes=l0c_bytes,
        facts=facts,
        macs_per_cycle=macs_per_cycle,
        m_dim=m_dim,
        n_dim=n_dim,
        k_dim=k_dim,
    )
    score = (1.0 / max(1e-12, t_roof_us)) * max(0.0, 1.0 - tail_waste)

    return PrescreenScore(
        candidate=cand,
        is_feasible=(reason is None),
        l1_bytes=l1_bytes,
        l0a_bytes=l0a_bytes,
        l0b_bytes=l0b_bytes,
        l0c_bytes=l0c_bytes,
        ub_bytes_per_aiv=ub_bytes,
        arithmetic_intensity=intensity,
        fixpipe_to_cube_ratio=fp_ratio,
        score=score if reason is None else 0.0,
        rejection_reason=reason,
        cube_macs_per_cycle=macs_per_cycle,
        half_l1_penalty=0.0,
        t_roof_us=t_roof_us,
        tail_waste_ratio=tail_waste,
    )


def prescreen_tile_configs(
    candidates: Iterable[TileConfigCandidate],
    *,
    target: AscendTarget | None = None,
    simtvf: bool = False,
    top_k: int | None = None,
    dtype: str | None = None,
    hf32: bool | None = None,
    l1_half_bank_split: bool | None = None,
    m_dim: int = 0,
    n_dim: int = 0,
    k_dim: int = 0,
    cost_model_mode: str = "off",
) -> list[PrescreenScore]:
    """Filter tile shapes by hard physical feasibility and rank by Roofline and tail waste."""
    _ = cost_model_mode
    facts = (target or AscendTarget("ascend.950pr")).get_facts(AscendSpaceFacts)
    scored = [
        evaluate_tile_candidate(
            c,
            facts,
            simtvf=simtvf,
            dtype=dtype,
            hf32=hf32,
            l1_half_bank_split=l1_half_bank_split,
            m_dim=m_dim,
            n_dim=n_dim,
            k_dim=k_dim,
        )
        for c in candidates
    ]
    feasible = [item for item in scored if item.is_feasible]
    feasible.sort(
        key=lambda item: (
            round(item.t_roof_us, 12),
            round(item.tail_waste_ratio, 12),
            item.candidate.bm,
            item.candidate.bn,
            item.candidate.bk,
            item.candidate.stages,
        )
    )
    if top_k is not None and top_k >= 0:
        return feasible[:top_k]
    return feasible


def _aligned_axis_values(
    value: int,
    step: int,
    *,
    max_val: int | None = None,
    require_even_fractals: bool = False,
) -> tuple[int, ...]:
    if value % step == 0:
        raw = (value - step, value, value + step)
    else:
        base = (value // step) * step
        raw = (base, base + step)
    valid: list[int] = []
    for cand in raw:
        if cand <= 0 or cand % 16 != 0:
            continue
        if require_even_fractals and (cand // 16) % 2 != 0:
            continue
        if max_val is not None and cand > max_val:
            continue
        if cand not in valid:
            valid.append(cand)
    return tuple(valid)


def generate_aligned_neighborhood(
    center: TileConfigCandidate,
    *,
    m_max: int | None = None,
    n_max: int | None = None,
    k_max: int | None = None,
) -> tuple[TileConfigCandidate, ...]:
    """Expand +/-1 fractal-aligned (bm, bn, bk, stages) neighbors around a tile config."""
    m_step = 32 if center.aiv_split == 2 else 16
    bm_vals = _aligned_axis_values(
        center.bm,
        m_step,
        max_val=m_max,
        require_even_fractals=(center.aiv_split == 2),
    )
    bn_vals = _aligned_axis_values(center.bn, 16, max_val=n_max)
    bk_vals = _aligned_axis_values(center.bk, 16, max_val=k_max)
    stage_vals = tuple(
        dict.fromkeys(st for st in (center.stages - 1, center.stages, center.stages + 1) if st >= 1)
    )
    neighbors: list[TileConfigCandidate] = []
    for bm, bn, bk, stages in product(bm_vals, bn_vals, bk_vals, stage_vals):
        neighbors.append(
            TileConfigCandidate(
                bm=bm,
                bn=bn,
                bk=bk,
                stages=stages,
                dtype_bytes=center.dtype_bytes,
                accum_bytes=center.accum_bytes,
                aiv_split=center.aiv_split,
                d_out=center.d_out,
                dual_accumulator=center.dual_accumulator,
            )
        )
    return tuple(neighbors)


def count_dsl_gemm_calls(dsl_source: str) -> int:
    """Count AST gemm call sites in a Python TileLang DSL snippet."""
    try:
        tree = ast.parse(dsl_source)
    except SyntaxError:
        return 0
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Attribute) and fn.attr == "gemm":
                count += 1
            elif isinstance(fn, ast.Name) and fn.id == "gemm":
                count += 1
    return count


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
            names = {arg.arg for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs)}
            return names, node.args.kwarg is not None
    return set(), False


def _bounded_live_ir_variants(
    center: TileConfigCandidate,
    *,
    tune_m: bool,
    tune_n: bool,
    tune_k: bool,
    tune_stages: bool,
    max_variants: int,
    pure_vector_row_tile: bool = False,
) -> list[TileConfigCandidate]:
    variants: list[TileConfigCandidate] = [center]
    seen: set[tuple[int, int, int, int]] = {(center.bm, center.bn, center.bk, center.stages)}

    def _add(cand: TileConfigCandidate) -> None:
        if len(variants) >= max_variants:
            return
        key = (cand.bm, cand.bn, cand.bk, cand.stages)
        if key not in seen:
            seen.add(key)
            variants.append(cand)

    if pure_vector_row_tile:
        pool = (2, 4, 6, 8, 16, 1, 32) if center.bm <= 4 else (8, 16, 24, 32, 4, 12, 2, 1)
        bm_vals = (center.bm, *[m for m in pool if m != center.bm])
        stage_vals = (
            (center.stages, *[s for s in (1, 2, 3) if tune_stages and s != center.stages])
            if tune_stages
            else (center.stages,)
        )
        for st in stage_vals:
            for bm_cand in bm_vals:
                _add(
                    TileConfigCandidate(
                        bm=bm_cand,
                        bn=center.bn,
                        bk=center.bk,
                        stages=st,
                        dtype_bytes=center.dtype_bytes,
                        accum_bytes=center.accum_bytes,
                        aiv_split=center.aiv_split,
                        d_out=center.d_out,
                        dual_accumulator=center.dual_accumulator,
                    )
                )
                if len(variants) >= max_variants:
                    return variants
        return variants

    if tune_m and center.bm >= 64:
        step = 16 * max(1, center.aiv_split)
        half_bm = ((center.bm // 2) // step) * step
        if half_bm >= 16:
            _add(
                TileConfigCandidate(
                    bm=half_bm,
                    bn=center.bn,
                    bk=center.bk,
                    stages=center.stages,
                    dtype_bytes=center.dtype_bytes,
                    accum_bytes=center.accum_bytes,
                    aiv_split=center.aiv_split,
                    d_out=center.d_out,
                    dual_accumulator=center.dual_accumulator,
                )
            )
    if tune_m and center.bm in (32, 64):
        _add(
            TileConfigCandidate(
                bm=center.bm * 2,
                bn=center.bn,
                bk=center.bk,
                stages=center.stages,
                dtype_bytes=center.dtype_bytes,
                accum_bytes=center.accum_bytes,
                aiv_split=center.aiv_split,
                d_out=center.d_out,
                dual_accumulator=center.dual_accumulator,
            )
        )

    neighbors = generate_aligned_neighborhood(center)
    ordered = sorted(
        neighbors,
        key=lambda c: (
            abs(c.bm - center.bm)
            + abs(c.bn - center.bn)
            + abs(c.bk - center.bk)
            + 32 * abs(c.stages - center.stages),
            c.bm,
            c.bn,
        ),
    )
    for cand in ordered:
        mapped = TileConfigCandidate(
            bm=cand.bm if tune_m else center.bm,
            bn=cand.bn if tune_n else center.bn,
            bk=cand.bk if tune_k else center.bk,
            stages=cand.stages if tune_stages else center.stages,
            dtype_bytes=center.dtype_bytes,
            accum_bytes=center.accum_bytes,
            aiv_split=center.aiv_split,
            d_out=center.d_out,
            dual_accumulator=center.dual_accumulator,
        )
        _add(mapped)
        if len(variants) >= max_variants:
            break
    return variants


def _apply_tile_candidate_to_kwargs(
    base_kwargs: Mapping[str, object] | None,
    cand: TileConfigCandidate,
    *,
    params: set[str],
    has_varkw: bool,
) -> dict[str, object]:
    kw = dict(base_kwargs or {})
    if "bm" in params:
        kw["bm"] = cand.bm
    elif "tile_m" in params:
        kw["tile_m"] = cand.bm
    elif "block_m" in params:
        kw["block_m"] = cand.bm
    elif "tile_rows" in params:
        kw["tile_rows"] = cand.bm
    elif "row_tile" in params:
        kw["row_tile"] = cand.bm
    elif has_varkw:
        kw["bm"] = cand.bm

    if "bn" in params:
        kw["bn"] = cand.bn
    elif "tile_n" in params:
        kw["tile_n"] = cand.bn
    elif "block_n" in params:
        kw["block_n"] = cand.bn
    elif has_varkw:
        kw["bn"] = cand.bn

    if "bk" in params:
        kw["bk"] = cand.bk
    elif "tile_k" in params:
        kw["tile_k"] = cand.bk
    elif "block_k" in params:
        kw["block_k"] = cand.bk

    if "num_stages" in params:
        kw["num_stages"] = cand.stages
    elif "stages" in params:
        kw["stages"] = cand.stages
    return kw


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


def rank_stage_a_candidates_with_cost_model(
    candidates: Sequence[tuple[str, str, StageAResult]],
    *,
    cost_model: HybridAscendCostModel,
    live_plan: object | None = None,
    axes: TopologyAxes | None = None,
    blueprint: object | None = None,
    sil_advisory: object | None = None,
    parent_ledger: object | None = None,
    parent_features: CostFeatures | None = None,
    creative_candidate_ids: Collection[str] | None = None,
    parent_workspace_stages: int = 0,
) -> list[CostRankedCandidate]:
    """Rank Stage-A passers by parent-relative delta cost, ascending, with stable ties."""
    parent_cf = parent_features
    if parent_cf is None and parent_ledger is not None:
        parent_cf = build_cost_features_from_ledger(
            parent_ledger,
            live_plan=live_plan,
            axes=axes,
            blueprint=blueprint,
            sil_advisory=sil_advisory,
            target=cost_model.target,
        )
    parent_pred = cost_model.predict(parent_cf) if parent_cf is not None else None
    raw_parent_axes = (
        axes
        if isinstance(axes, TopologyAxes)
        else getattr(parent_ledger, "topology_axes", None)
    )
    eff_parent_axes = raw_parent_axes if isinstance(raw_parent_axes, TopologyAxes) else None
    creative_ids = frozenset(str(cid) for cid in (creative_candidate_ids or ()))
    scored: list[tuple[float, int, CostRankedCandidate]] = []
    for idx, (cand_id, _patched_src, stage_a) in enumerate(candidates):
        ledger = getattr(stage_a, "ledger", None)
        raw_cand_axes = getattr(ledger, "topology_axes", None)
        cand_axes = (
            raw_cand_axes if isinstance(raw_cand_axes, TopologyAxes) else eff_parent_axes
        )
        features = build_cost_features_from_ledger(
            ledger,
            live_plan=live_plan,
            axes=cand_axes,
            blueprint=blueprint,
            sil_advisory=sil_advisory,
            target=cost_model.target,
        )
        pred = cost_model.predict(features)
        score = float(pred.pred_us)
        rel_ratio = 1.0
        if parent_pred is not None and parent_cf is not None:
            ratio = float(cost_model.predict_delta_ratio(features, parent_cf))
            score = float(parent_pred.pred_us) * ratio
            rel_ratio = ratio
        cand_ws_stages = int(getattr(ledger, "workspace_stages", parent_workspace_stages) or 0)
        exempt = is_paradigm_shift_exempt(
            cand_axes,
            eff_parent_axes,
            cand_workspace_stages=cand_ws_stages,
            parent_workspace_stages=int(parent_workspace_stages),
            is_creative=str(cand_id) in creative_ids,
        )
        ranked_item = CostRankedCandidate(
            candidate_id=str(cand_id),
            pred_us=score,
            tier0_physics_us=float(pred.tier0_physics_us),
            tier1_penalty_us=float(pred.tier1_penalty_us),
            dominant_bound=str(pred.dominant_bound),
            topology_bucket=str(pred.topology_bucket),
            ledger=ledger,
            parent_relative_ratio=rel_ratio,
            coarse_reject_exempt=exempt,
        )
        scored.append((ranked_item.pred_us, idx, ranked_item))
    scored.sort(key=lambda entry: (entry[0], entry[1]))
    return [entry[2] for entry in scored]


def _extract_problem_mnk(
    var_bindings: Mapping[str, int] | None,
    dims: Mapping[str, object] | None,
) -> tuple[int, int, int]:
    m_val = 0
    n_val = 0
    k_val = 0
    for src in (var_bindings, dims):
        if not isinstance(src, Mapping):
            continue
        for key, target_name in (
            ("M", "m"),
            ("m_dim", "m"),
            ("m_rows", "m"),
            ("total_rows", "m"),
            ("N", "n"),
            ("n_dim", "n"),
            ("K", "k"),
            ("k_dim", "k"),
        ):
            raw = src.get(key)
            if isinstance(raw, int) and not isinstance(raw, bool) and raw > 0:
                if target_name == "m" and m_val == 0:
                    m_val = int(raw)
                elif target_name == "n" and n_val == 0:
                    n_val = int(raw)
                elif target_name == "k" and k_val == 0:
                    k_val = int(raw)
    return m_val, n_val, k_val


def prescreen_live_ir_neighborhood(
    dsl_source: str,
    *,
    factory_name: str,
    center: TileConfigCandidate,
    base_kwargs: Mapping[str, object] | None = None,
    var_bindings: Mapping[str, int] | None = None,
    config_selector_name: str | None = None,
    pass_configs: Mapping[str, object] | None = None,
    compile_flags: Sequence[str] | str | None = None,
    dims: Mapping[str, object] | None = None,
    target: AscendTarget | None = None,
    max_variants: int = 8,
    cost_model: HybridAscendCostModel | None = None,
    cost_model_mode: str | None = None,
    axes: TopologyAxes | None = None,
    live_plan: object | None = None,
    blueprint: object | None = None,
    sil_advisory: object | None = None,
    ordered_tensor_shapes: Sequence[Sequence[int]] | None = None,
) -> str:
    """Run bounded Stage A live-IR dry-runs across aligned tile variants around center."""
    eff_cost_model = (
        None
        if (cost_model_mode is not None and str(cost_model_mode).strip().lower() == "off")
        else cost_model
    )
    params, has_varkw = inspect_builder_tile_params(dsl_source, factory_name)
    m_keys = ("bm", "tile_m", "block_m", "tile_rows", "row_tile")
    n_keys = ("bn", "tile_n", "block_n")
    k_keys = ("bk", "tile_k", "block_k")
    tune_m = has_varkw or any(k in params for k in m_keys)
    tune_n = has_varkw or any(k in params for k in n_keys)
    tune_k = has_varkw or any(k in params for k in k_keys)
    tune_stages = has_varkw or "stages" in params or "num_stages" in params
    if not (tune_m or tune_n or tune_k or tune_stages):
        return ""
    pure_vector_row_tile = (
        count_dsl_gemm_calls(dsl_source) == 0
        and any(k in params for k in ("tile_m", "tile_rows", "row_tile"))
        and "bm" not in params
        and "block_m" not in params
    )

    from tilefoundry_ascend.agent.orchestrator.gate_funnel import (  # noqa: PLC0415
        _merge_pass_configs_and_flags,
        _resolve_prim_func_and_configs_from_dsl,
        run_stage_a_static,
    )

    sel_pc: dict[str, object] = {}
    sel_cf: list[str] = []
    if config_selector_name and pass_configs is None and compile_flags is None:
        try:
            _pf, sel_pc, sel_cf = _resolve_prim_func_and_configs_from_dsl(
                dsl_source,
                factory_name=factory_name,
                factory_kwargs=base_kwargs,
                config_selector_name=config_selector_name,
                dims=dims,
            )
        except Exception:
            sel_pc, sel_cf = {}, []
    merged_pc, merged_cf = _merge_pass_configs_and_flags(
        sel_pc,
        sel_cf,
        pass_configs,
        compile_flags,
    )

    variants = _bounded_live_ir_variants(
        center,
        tune_m=tune_m,
        tune_n=tune_n,
        tune_k=tune_k,
        tune_stages=tune_stages,
        max_variants=max(1, min(8, int(max_variants))),
        pure_vector_row_tile=pure_vector_row_tile,
    )

    total_rows = 0
    for src in (var_bindings, dims):
        if isinstance(src, Mapping):
            raw_m = src.get("m_rows", src.get("total_rows", src.get("M", src.get("m_dim"))))
            if isinstance(raw_m, int) and not isinstance(raw_m, bool) and raw_m > 0:
                total_rows = int(raw_m)
                break
            b_val = src.get("B", src.get("b_dim", src.get("batch")))
            s_val = src.get("S", src.get("s_dim", src.get("S_q", src.get("seq_len"))))
            if (
                isinstance(b_val, int)
                and isinstance(s_val, int)
                and not isinstance(b_val, bool)
                and not isinstance(s_val, bool)
                and b_val > 0
                and s_val > 0
            ):
                total_rows = int(b_val) * int(s_val)
                break
    eff_target = target or AscendTarget()
    facts = ascend_space_facts(eff_target)
    num_aiv_cores = max(1, int(facts.num_aiv_cores))
    m_dim, n_dim, k_dim = _extract_problem_mnk(var_bindings, dims)
    clock_ghz = float(getattr(facts, "clock_ghz", 1.65) or 1.65)
    hbm_bps = float(getattr(facts, "hbm_peak_bandwidth_bps", 0) or 0)

    records: list[
        tuple[
            bool,
            tuple[float, ...],
            TileConfigCandidate,
            str,
            dict[str, int],
            int,
            int | None,
            float,
            int,
            float,
            CostPrediction | None,
            float,
        ]
    ] = []
    any_ledger = False
    parent_features: CostFeatures | None = None
    parent_pred: CostPrediction | None = None
    for idx, cand in enumerate(variants):
        if (
            cand.bm <= 0
            or cand.bn <= 0
            or cand.bk <= 0
            or cand.stages < 1
            or cand.aiv_split < 1
            or cand.dtype_bytes <= 0
            or cand.accum_bytes <= 0
        ):
            records.append(
                (
                    False,
                    (float("inf"), 1.0, float(idx)),
                    cand,
                    "REJECT(non_positive_tile_dimension)",
                    {},
                    0,
                    None,
                    1.0,
                    0,
                    0.0,
                    None,
                    0.0,
                )
            )
            continue
        if (not pure_vector_row_tile) and (
            cand.bm % 16 != 0 or cand.bn % 16 != 0 or cand.bk % 16 != 0
        ):
            records.append(
                (
                    False,
                    (float("inf"), 1.0, float(idx)),
                    cand,
                    "REJECT(unaligned_mmad_16x16x16_atom)",
                    {},
                    0,
                    None,
                    1.0,
                    0,
                    0.0,
                    None,
                    0.0,
                )
            )
            continue
        cand_kwargs = _apply_tile_candidate_to_kwargs(
            base_kwargs,
            cand,
            params=params,
            has_varkw=has_varkw,
        )
        res = run_stage_a_static(
            dsl_source,
            factory_name=factory_name,
            factory_kwargs=cand_kwargs,
            target=target,
            pass_configs=merged_pc or None,
            compile_flags=merged_cf or None,
            dims=dims,
            var_bindings=var_bindings,
            ordered_tensor_shapes=ordered_tensor_shapes,
        )
        ledger = res.ledger
        if ledger is None:
            records.append(
                (
                    False,
                    (float("inf"), 1.0, float(idx)),
                    cand,
                    f"REJECT({res.error_message or 'stage_a_failed'})",
                    {},
                    0,
                    None,
                    1.0,
                    0,
                    0.0,
                    None,
                    0.0,
                )
            )
            continue

        any_ledger = True
        scope_bytes = {
            scope: ledger.waterlines[scope].peak_allocated_bytes
            for scope in ("L1", "L0A", "L0B", "L0C", "UB", "SSBUF")
            if scope in ledger.waterlines
        }
        degraded = [
            buf
            for wl in ledger.waterlines.values()
            for buf in wl.degraded_single_buffers
        ]
        overflow_item = None
        for scope in ("L0C", "UB", "L1", "L0A", "L0B", "SSBUF", *sorted(ledger.waterlines)):
            wl = ledger.waterlines.get(scope)
            if wl is not None and wl.overflow_bytes > 0:
                overflow_item = (scope, wl.peak_allocated_bytes, wl.capacity_bytes)
                break

        stack_bytes = max(
            int(getattr(ledger, "stack_bytes", 0) or 0),
            int(getattr(res, "stack_bytes", 0) or 0),
        )
        if overflow_item is not None:
            sc, pk, cap = overflow_item
            status = f"REJECT({sc}_overflow:{pk}>{cap})"
            is_ok = False
        elif stack_bytes > 0:
            status = f"REJECT(stack_overflow:{stack_bytes}>0)"
            is_ok = False
        elif degraded:
            status = f"REJECT(degraded_single_buffers:{','.join(degraded)})"
            is_ok = False
        elif not res.passed or not ledger.is_legal:
            status = f"REJECT({res.error_message or 'illegal_schedule'})"
            is_ok = False
        else:
            status = "OK"
            is_ok = True

        dom_cycles = ledger.pipe_cycles.get(
            ledger.dominant_pipe,
            max((v for v in ledger.pipe_cycles.values() if v > 0), default=0),
        )
        burst_eff = float(getattr(ledger, "dma_burst_efficiency_ratio", 1.0) or 1.0)
        est_waves = (
            math.ceil(math.ceil(total_rows / cand.bm) / float(num_aiv_cores))
            if (pure_vector_row_tile and total_rows > 0 and cand.bm > 0)
            else 0
        )
        if pure_vector_row_tile:
            padded_rows = (
                math.ceil(total_rows / cand.bm) * cand.bm
                if (total_rows > 0 and cand.bm > 0)
                else 0
            )
            tail_waste = (
                max(0.0, float(padded_rows - total_rows) / float(max(1, padded_rows)))
                if padded_rows > 0
                else 0.0
            )
            waves_mult = max(1, est_waves)
            t_comp_us = (float(waves_mult) * float(dom_cycles)) / (clock_ghz * 1e3)
            gm_bytes = float(
                ledger.total_gm_bytes()
                if hasattr(ledger, "total_gm_bytes")
                else getattr(ledger, "gm_roundtrip_bytes", 0)
            )
            t_mem_us = (gm_bytes / hbm_bps) * 1e6 if hbm_bps > 0.0 else 0.0
            t_roof_us = max(t_comp_us, t_mem_us)
            wave_burst_score = t_roof_us
            sort_key: tuple[float, ...] = (
                t_roof_us,
                tail_waste,
                float(dom_cycles),
                float(idx),
            )
        else:
            eff_dtype = _resolve_candidate_dtype(cand, None)
            macs_per_cycle = max(1, _resolve_cube_macs(facts, eff_dtype, hf32=cand.hf32))
            l0c_bytes = (
                _align512(cand.bm * (cand.d_out or cand.bn) * cand.accum_bytes)
                + _align512(cand.bm * cand.bn * cand.accum_bytes)
                if (cand.d_out > 0 or cand.dual_accumulator)
                else cand.bm * cand.bn * cand.accum_bytes
            )
            t_roof_us, tail_waste = _compute_tile_roofline_and_waste(
                cand.bm,
                cand.bn,
                cand.bk,
                dtype_bytes=cand.dtype_bytes,
                accum_bytes=cand.accum_bytes,
                l0c_bytes=l0c_bytes,
                facts=facts,
                macs_per_cycle=macs_per_cycle,
                m_dim=m_dim,
                n_dim=n_dim,
                k_dim=k_dim,
            )
            wave_burst_score = t_roof_us
            norm_dom_cycles = float(dom_cycles) / float(max(1, cand.bm * cand.bn * cand.bk))
            sort_key = (t_roof_us, tail_waste, norm_dom_cycles, float(idx))

        cost_pred: CostPrediction | None = None
        cost_score = 0.0
        if eff_cost_model is not None:
            cf = build_cost_features_from_ledger(
                ledger,
                live_plan=live_plan,
                axes=axes,
                blueprint=blueprint,
                sil_advisory=sil_advisory,
                target=eff_cost_model.target,
            )
            cost_pred = eff_cost_model.predict(cf)
            cost_score = float(cost_pred.pred_us)
            if parent_pred is not None and parent_features is not None:
                ratio = eff_cost_model.predict_delta_ratio(cf, parent_features)
                cost_score = float(parent_pred.pred_us) * float(ratio)
            sort_key = (cost_score, *sort_key)
            if idx == 0:
                parent_features = cf
                parent_pred = cost_pred
        records.append(
            (
                is_ok,
                sort_key,
                cand,
                status,
                scope_bytes,
                len(degraded),
                ledger.steady_state_ii,
                burst_eff,
                est_waves,
                wave_burst_score,
                cost_pred,
                cost_score,
            )
        )

    if not any_ledger:
        return ""

    feasible = [item for item in records if item[0]]
    rejected = [item for item in records if not item[0]]
    feasible.sort(key=lambda item: item[1])
    ordered = [*feasible[:4], *rejected[:4]]
    if not ordered:
        return ""

    lines = ["Live-IR Tile Prescreen Summary:"]
    for rank_idx, (
        _ok,
        _rk,
        cand,
        status,
        sbytes,
        deg_cnt,
        ii_val,
        burst_eff,
        est_waves,
        wave_burst_score,
        cost_pred,
        cost_score,
    ) in enumerate(ordered, start=1):
        extra = f", ii={ii_val}" if ii_val is not None else ""
        if deg_cnt > 0:
            extra += f", degraded={deg_cnt}"
        cost_suffix = (
            f", pred_us={cost_score:.2f}us ({cost_pred.dominant_bound})"
            if cost_pred is not None
            else ""
        )
        if pure_vector_row_tile:
            wl_parts = [
                f"{scope.lower()}={sbytes.get(scope, 0)}B"
                for scope in ("UB", "SSBUF")
            ]
            vec_extra = f", dma_burst_eff={burst_eff:.3f}"
            if est_waves > 0:
                vec_extra += f", est_waves={est_waves}, wave_burst_score={wave_burst_score:.2f}"
            lines.append(
                f"  {rank_idx}. tile_m={cand.bm}, stages={cand.stages} "
                f"| {status} | {', '.join(wl_parts)}{extra}{vec_extra}{cost_suffix}"
            )
        else:
            wl_parts = [
                f"{scope.lower()}={sbytes.get(scope, 0)}B"
                for scope in ("L1", "L0A", "L0B", "L0C", "UB", "SSBUF")
            ]
            lines.append(
                f"  {rank_idx}. bm={cand.bm}, bn={cand.bn}, bk={cand.bk}, stages={cand.stages} "
                f"| {status} | {', '.join(wl_parts)}{extra}{cost_suffix}"
            )
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
            f"  {idx}. bm={c.bm}, bn={c.bn}, bk={c.bk}, stages={c.stages}, "
            f"aiv_split={c.aiv_split}{d_out_part} | {status} | l1={item.l1_bytes}B, "
            f"l0c={item.l0c_bytes}B, ub_aiv={item.ub_bytes_per_aiv}B | "
            f"ai={item.arithmetic_intensity:.2f}, fp_ratio={item.fixpipe_to_cube_ratio:.2f}, "
            f"score={item.score:.3f}"
        )
    return "\n".join(lines)


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


def _extract_selector_base_config(
    module_source: str,
    selector_fn_name: str,
    sample_builder_kwargs: Mapping[str, object],
    dims: Mapping[str, int] | None,
) -> dict[str, object]:
    """Extract baseline selector config via direct module call with AST dict fallback."""
    tree = ast.parse(module_source)
    from tilefoundry_ascend.agent.orchestrator.gate_funnel import (  # noqa: PLC0415
        _filter_kwargs_for_callable,
        _merge_dims_into_kwargs,
    )

    merged_kwargs = _merge_dims_into_kwargs(sample_builder_kwargs, dims)
    ns: dict[str, object] = {}
    try:
        exec(compile(tree, "<lattice_selector_probe>", "exec"), ns)
        sel_fn = ns.get(selector_fn_name)
        if callable(sel_fn):
            call_kw = _filter_kwargs_for_callable(sel_fn, merged_kwargs)
            sel_out = sel_fn(**call_kw)
            if isinstance(sel_out, Mapping):
                return {str(k): v for k, v in sel_out.items()}
    except Exception:
        pass

    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name != selector_fn_name:
            continue
        local_ints: dict[str, int] = {}
        for arg_name, arg_val in merged_kwargs.items():
            if isinstance(arg_val, int) and not isinstance(arg_val, bool):
                local_ints[arg_name] = int(arg_val)
        for stmt in node.body:
            if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
                tgt = stmt.targets[0]
                if (
                    isinstance(tgt, ast.Name)
                    and isinstance(stmt.value, ast.Constant)
                    and isinstance(stmt.value.value, int)
                    and not isinstance(stmt.value.value, bool)
                ):
                    local_ints[tgt.id] = int(stmt.value.value)
        for inner in ast.walk(node):
            if isinstance(inner, ast.Return) and isinstance(inner.value, ast.Dict):
                parsed: dict[str, object] = {}
                for k_node, v_node in zip(inner.value.keys, inner.value.values):
                    if not (isinstance(k_node, ast.Constant) and isinstance(k_node.value, str)):
                        continue
                    if isinstance(v_node, ast.Constant):
                        parsed[k_node.value] = v_node.value
                    elif isinstance(v_node, ast.Name) and v_node.id in local_ints:
                        parsed[k_node.value] = local_ints[v_node.id]
                if parsed:
                    return parsed
    return {}


def _generate_lattice_override_grid(
    base_cfg: Mapping[str, object],
    facts: AscendSpaceFacts,
    max_points: int,
) -> list[dict[str, int]]:
    """Generate bounded hardware-feasible knob override points for Cube, Vector, or State."""
    cap = max(1, int(max_points))
    has_stages = "num_stages" in base_cfg
    stage_vals = (1, 2) if has_stages else (None,)

    if all(k in base_cfg for k in ("bm", "bn", "bk")):
        dtype_str = str(base_cfg.get("dtype") or base_cfg.get("tl_dtype") or "f16").lower()
        dtype_bytes = 4 if ("32" in dtype_str and "bf" not in dtype_str and "hf" not in dtype_str) else 2
        bm_base = int(base_cfg["bm"]) if isinstance(base_cfg["bm"], int) else 64
        bn_base = int(base_cfg["bn"]) if isinstance(base_cfg["bn"], int) else 128
        bk_base = int(base_cfg["bk"]) if isinstance(base_cfg["bk"], int) else 64
        st_base = (
            int(base_cfg["num_stages"])
            if (has_stages and isinstance(base_cfg["num_stages"], int))
            else 1
        )
        bm_opts = tuple(dict.fromkeys((bm_base, 32, 64, 128)))
        bn_opts = tuple(dict.fromkeys((bn_base, 64, 128, 256)))
        bk_opts = tuple(dict.fromkeys((bk_base, 32, 64, 128)))
        st_opts = tuple(dict.fromkeys((st_base, 1, 2))) if has_stages else (None,)

        scored_points: list[tuple[int, float, dict[str, int]]] = []
        seen_cube: set[tuple[int, int, int, int | None]] = set()
        for bm in bm_opts:
            for bn in bn_opts:
                for bk in bk_opts:
                    if bm <= 0 or bn <= 0 or bk <= 0 or bm % 16 != 0 or bn % 16 != 0 or bk % 16 != 0:
                        continue
                    for st in st_opts:
                        key = (bm, bn, bk, st)
                        if key in seen_cube:
                            continue
                        seen_cube.add(key)
                        eff_st = st if st is not None else 1
                        ev = evaluate_tile_candidate(
                            TileConfigCandidate(
                                bm=bm,
                                bn=bn,
                                bk=bk,
                                stages=eff_st,
                                dtype_bytes=dtype_bytes,
                                accum_bytes=4,
                            ),
                            facts,
                            dtype=dtype_str,
                        )
                        if ev.l1_bytes > facts.l1_bytes or ev.l0c_bytes > facts.l0c_bytes:
                            continue
                        if not ev.is_feasible:
                            continue
                        ov: dict[str, int] = {"bm": int(bm), "bn": int(bn), "bk": int(bk)}
                        if has_stages and st is not None:
                            ov["num_stages"] = int(st)
                        is_base = 0 if (bm == bm_base and bn == bn_base and bk == bk_base and st == (st_base if has_stages else None)) else 1
                        scored_points.append((is_base, -ev.score, ov))
        scored_points.sort(key=lambda item: (item[0], item[1]))
        return [item[2] for item in scored_points[:cap]]

    if "chunk_t" in base_cfg:
        ct_base = int(base_cfg["chunk_t"]) if isinstance(base_cfg["chunk_t"], int) else 32
        ct_opts = tuple(dict.fromkeys((ct_base, 16, 32, 64)))
        has_dv = "dv_tile" in base_cfg
        dv_base = (
            int(base_cfg["dv_tile"]) if (has_dv and isinstance(base_cfg["dv_tile"], int)) else 32
        )
        dv_opts = tuple(dict.fromkeys((dv_base, 32, 64))) if has_dv else (None,)
        st_base = (
            int(base_cfg["num_stages"])
            if (has_stages and isinstance(base_cfg["num_stages"], int))
            else 1
        )
        st_opts = tuple(dict.fromkeys((st_base, 1, 2))) if has_stages else (None,)
        grid: list[dict[str, int]] = []
        for ct in ct_opts:
            for dv in dv_opts:
                for st in st_opts:
                    ov = {"chunk_t": int(ct)}
                    if has_dv and dv is not None:
                        ov["dv_tile"] = int(dv)
                    if has_stages and st is not None:
                        ov["num_stages"] = int(st)
                    grid.append(ov)
                    if len(grid) >= cap:
                        return grid
        return grid

    if "tile_m" in base_cfg:
        tm_base = int(base_cfg["tile_m"]) if isinstance(base_cfg["tile_m"], int) else 4
        tm_opts = tuple(dict.fromkeys((tm_base, 1, 2, 4, 8, 16, 32)))
        st_base = (
            int(base_cfg["num_stages"])
            if (has_stages and isinstance(base_cfg["num_stages"], int))
            else 1
        )
        st_opts = tuple(dict.fromkeys((st_base, 1, 2))) if has_stages else stage_vals
        grid = []
        for tm in tm_opts:
            for st in st_opts:
                ov = {"tile_m": int(tm)}
                if has_stages and st is not None:
                    ov["num_stages"] = int(st)
                grid.append(ov)
                if len(grid) >= cap:
                    return grid
        return grid

    return []


def _patch_selector_source(
    module_source: str,
    selector_fn_name: str,
    overrides: Mapping[str, int],
) -> str:
    """Wrap selector_fn_name minimally via AST line insertion to apply *overrides*."""
    tree = ast.parse(module_source)
    target_node: ast.FunctionDef | None = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == selector_fn_name:
            target_node = node
            break
    if target_node is None or target_node.end_lineno is None:
        raise ValueError(f"Selector function {selector_fn_name!r} not found at module top level")
    lines = module_source.splitlines(keepends=True)
    insert_at = int(target_node.end_lineno)
    ov_literal = repr({str(k): int(v) for k, v in overrides.items()})
    orig_alias = f"_tf_sweep_orig_{selector_fn_name}"
    wrapper_block = (
        "\n"
        "import functools as _tf_sweep_functools\n"
        f"{orig_alias} = {selector_fn_name}\n"
        f"@_tf_sweep_functools.wraps({orig_alias})\n"
        f"def {selector_fn_name}(*args, **kwargs):\n"
        f"    _tf_sweep_cfg = dict({orig_alias}(*args, **kwargs))\n"
        f"    _tf_sweep_cfg.update({ov_literal})\n"
        "    return _tf_sweep_cfg\n"
    )
    if insert_at > 0 and not lines[insert_at - 1].endswith("\n"):
        lines[insert_at - 1] = lines[insert_at - 1] + "\n"
    patched = "".join([*lines[:insert_at], wrapper_block, *lines[insert_at:]])
    ast.parse(patched)
    return patched


def _static_dominant_bound_from_ledger(ledger: object) -> str:
    """Derive dominant bound from Stage-A ledger pipe cycles without CostModel."""
    dom_pipe = str(getattr(ledger, "dominant_pipe", "") or "").upper()
    if any(tok in dom_pipe for tok in ("MTE2", "MTE3", "DMA")):
        return "memory_bound"
    if any(tok in dom_pipe for tok in ("PIPE_M", "CUBE", "PIPE_V", "VEC", "FIX")):
        return "compute_bound"
    pipes = getattr(ledger, "pipe_cycles", None)
    if isinstance(pipes, Mapping):
        mte_c = 0
        comp_c = 0
        for pname, pval in pipes.items():
            val = int(pval or 0)
            if val <= 0:
                continue
            up = str(pname).upper()
            if "MTE2" in up or "MTE3" in up or "DMA" in up:
                mte_c += val
            elif any(tok in up for tok in ("PIPE_M", "CUBE", "PIPE_V", "VEC", "FIX")):
                comp_c += val
        if mte_c > comp_c:
            return "memory_bound"
        if comp_c > 0:
            return "compute_bound"
    return "balanced"


def sweep_executable_lattice_zero_llm(
    module_source: str,
    *,
    selector_fn_name: str,
    builder_fn_name: str,
    sample_builder_kwargs: Mapping[str, object],
    dims: Mapping[str, int] | None,
    cost_model: HybridAscendCostModel | None = None,
    cost_model_mode: str | None = None,
    target: AscendTarget | None = None,
    max_points: int = 24,
    top_k: int = 3,
) -> list[LatticeSweepCandidate]:
    """Sweep bounded tile lattice with Stage-A feasibility and Roofline ranking."""
    if top_k <= 0 or max_points <= 0:
        return []
    eff_cost_model = (
        None
        if (cost_model_mode is not None and str(cost_model_mode).strip().lower() == "off")
        else cost_model
    )
    eff_target = target or (
        eff_cost_model.target if eff_cost_model is not None else AscendTarget("ascend.950pr")
    )
    eff_facts = (
        eff_cost_model.facts
        if (eff_cost_model is not None and target is None)
        else eff_target.get_facts(AscendSpaceFacts)
    )
    base_cfg = _extract_selector_base_config(
        module_source,
        selector_fn_name,
        sample_builder_kwargs,
        dims,
    )
    if not base_cfg:
        return []
    override_grid = _generate_lattice_override_grid(base_cfg, eff_facts, max_points)
    if not override_grid:
        return []

    from tilefoundry_ascend.agent.orchestrator.gate_funnel import (  # noqa: PLC0415
        run_stage_a_static,
    )

    m_dim, n_dim, k_dim = _extract_problem_mnk(None, dims)
    clock_ghz = float(getattr(eff_facts, "clock_ghz", 1.65) or 1.65)
    hbm_bps = float(getattr(eff_facts, "hbm_peak_bandwidth_bps", 0) or 0)
    ranked_candidates: list[tuple[tuple[float, ...], LatticeSweepCandidate]] = []
    for idx, ov in enumerate(override_grid):
        if any(int(v) <= 0 for v in ov.values()):
            continue
        if all(k in ov for k in ("bm", "bn", "bk")) and (
            ov["bm"] % 16 != 0 or ov["bn"] % 16 != 0 or ov["bk"] % 16 != 0
        ):
            continue
        try:
            patched_source = _patch_selector_source(module_source, selector_fn_name, ov)
        except (SyntaxError, ValueError):
            continue
        stage_a = run_stage_a_static(
            patched_source,
            factory_name=builder_fn_name,
            factory_kwargs={**sample_builder_kwargs, **ov},
            target=eff_target,
            config_selector_name=selector_fn_name,
            dims=dims,
        )
        ledger = stage_a.ledger
        if not stage_a.passed or ledger is None:
            continue
        if not getattr(ledger, "is_legal", True):
            continue
        if int(getattr(ledger, "max_overflow_bytes", 0) or 0) > 0:
            continue
        if int(getattr(ledger, "stack_bytes", 0) or 0) > 0:
            continue
        if int(getattr(stage_a, "stack_bytes", 0) or 0) > 0:
            continue
        waterlines = getattr(ledger, "waterlines", None) or {}
        if any(int(getattr(wl, "overflow_bytes", 0) or 0) > 0 for wl in waterlines.values()):
            continue
        ub_wl = waterlines.get("UB")
        l1_wl = waterlines.get("L1")
        ub_bytes = int(getattr(ub_wl, "peak_allocated_bytes", 0) or 0)
        l1_bytes = int(getattr(l1_wl, "peak_allocated_bytes", 0) or 0)

        t_roof_us = 0.0
        tail_waste = 0.0
        if all(k in ov for k in ("bm", "bn", "bk")):
            dtype_str = str(base_cfg.get("dtype") or base_cfg.get("tl_dtype") or "f16").lower()
            dtype_bytes = (
                4 if ("32" in dtype_str and "bf" not in dtype_str and "hf" not in dtype_str) else 2
            )
            macs = max(1, _resolve_cube_macs(eff_facts, dtype_str, hf32=False))
            t_roof_us, tail_waste = _compute_tile_roofline_and_waste(
                int(ov["bm"]),
                int(ov["bn"]),
                int(ov["bk"]),
                dtype_bytes=dtype_bytes,
                accum_bytes=4,
                l0c_bytes=int(ov["bm"]) * int(ov["bn"]) * 4,
                facts=eff_facts,
                macs_per_cycle=macs,
                m_dim=m_dim,
                n_dim=n_dim,
                k_dim=k_dim,
            )
        else:
            pipes = getattr(ledger, "pipe_cycles", None) or {}
            bottleneck_cycles = (
                max((int(val or 0) for val in pipes.values()), default=0)
                if isinstance(pipes, Mapping)
                else 0
            )
            t_comp_us = float(bottleneck_cycles) / (clock_ghz * 1e3)
            gm_bytes = float(
                ledger.total_gm_bytes()
                if hasattr(ledger, "total_gm_bytes")
                else getattr(ledger, "gm_roundtrip_bytes", 0)
            )
            t_mem_us = (gm_bytes / hbm_bps) * 1e6 if hbm_bps > 0.0 else 0.0
            t_roof_us = max(t_comp_us, t_mem_us)
            tile_m = ov.get("tile_m") or ov.get("chunk_t") or 0
            if m_dim > 0 and tile_m > 0:
                padded_m = math.ceil(m_dim / int(tile_m)) * int(tile_m)
                tail_waste = max(0.0, float(padded_m - m_dim) / float(max(1, padded_m)))

        if eff_cost_model is not None:
            policy_hint: dict[str, object] = {**base_cfg, **ov}
            if all(k in ov for k in ("bm", "bn", "bk")):
                policy_hint["tile_mnk"] = [ov["bm"], ov["bn"], ov["bk"]]
            if "num_stages" in ov:
                policy_hint["l1_stages"] = ov["num_stages"]
            cf = build_cost_features_from_ledger(
                ledger,
                live_plan={"policy": policy_hint},
                target=eff_target,
            )
            pred = eff_cost_model.predict(cf)
            pred_us = float(pred.pred_us)
            dom_bound = str(pred.dominant_bound)
            rank_key: tuple[float, ...] = (pred_us, float(ub_bytes), float(idx))
        else:
            pipes = getattr(ledger, "pipe_cycles", None) or {}
            bottleneck_cycles = (
                max((int(val or 0) for val in pipes.values()), default=0)
                if isinstance(pipes, Mapping)
                else 0
            )
            pred_us = 0.0
            dom_bound = _static_dominant_bound_from_ledger(ledger)
            rank_key = (
                round(t_roof_us, 12),
                round(tail_waste, 12),
                float(bottleneck_cycles),
                float(idx),
            )

        cand = LatticeSweepCandidate(
            knob_overrides=dict(ov),
            patched_source=patched_source,
            pred_us=pred_us,
            dominant_bound=dom_bound,
            ub_bytes=ub_bytes,
            l1_bytes=l1_bytes,
            t_roof_us=t_roof_us,
            tail_waste_ratio=tail_waste,
        )
        ranked_candidates.append((rank_key, cand))

    ranked_candidates.sort(key=lambda entry: entry[0])
    return [entry[1] for entry in ranked_candidates[: int(top_k)]]


__all__ = [
    "CostRankedCandidate",
    "LatticeSweepCandidate",
    "PrescreenAblationReport",
    "PrescreenAblationSample",
    "PrescreenScore",
    "TileConfigCandidate",
    "count_dsl_gemm_calls",
    "evaluate_prescreen_ablation",
    "evaluate_tile_candidate",
    "format_prescreen_summary",
    "generate_aligned_neighborhood",
    "inspect_builder_tile_params",
    "prescreen_live_ir_neighborhood",
    "prescreen_tile_configs",
    "rank_stage_a_candidates_with_cost_model",
    "sweep_executable_lattice_zero_llm",
]
