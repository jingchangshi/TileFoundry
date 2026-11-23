"""Concrete (BM, BN, BK) tile-size dry-run batch prescreener without bisheng."""

from __future__ import annotations

import ast
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product

from tilefoundry_ascend.ascend.cost_model.calibration_backtest import (
    PrescreenAblationReport,
    PrescreenAblationSample,
    evaluate_prescreen_ablation,
)
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


@dataclass(frozen=True)
class PrescreenScore:
    """Hardware feasibility and arithmetic-intensity score for a tile config."""

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


def evaluate_tile_candidate(
    cand: TileConfigCandidate,
    facts: AscendSpaceFacts,
    *,
    simtvf: bool = False,
    dtype: str | None = None,
    hf32: bool | None = None,
    l1_half_bank_split: bool | None = None,
) -> PrescreenScore:
    """Compute exact scratchpad occupancy, per-dtype Cube balance, and Half-L1 conflict score."""
    eff_dtype = _resolve_candidate_dtype(cand, dtype)
    eff_hf32 = cand.hf32 if hf32 is None else bool(hf32)
    macs_per_cycle = max(1, _resolve_cube_macs(facts, eff_dtype, hf32=eff_hf32))

    if cand.bm <= 0 or cand.bn <= 0 or cand.bk <= 0 or cand.stages < 1 or cand.aiv_split < 1:
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
    if l1_bytes > facts.l1_bytes:
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

    eff_half_split = cand.l1_half_bank_split if l1_half_bank_split is None else bool(l1_half_bank_split)
    half_l1_limit = int(getattr(facts, "l1_bank_bytes", facts.l1_bytes // 2) or (facts.l1_bytes // 2))
    half_l1_penalty = 0.0
    if cand.stages >= 2:
        if not eff_half_split:
            half_l1_penalty = 0.25
        else:
            stages_per_bank = math.ceil(cand.stages / 2)
            half_bank_bytes = stages_per_bank * (l0a_slot + l0b_slot)
            if half_bank_bytes > half_l1_limit:
                overflow_ratio = (half_bank_bytes - half_l1_limit) / float(max(1, half_l1_limit))
                half_l1_penalty = min(0.75, 0.25 + 0.50 * overflow_ratio)

    flops = float(2 * cand.bm * cand.bn * cand.bk)
    mem_traffic = float(l0a_slot + l0b_slot + l0c_bytes)
    intensity = flops / mem_traffic if mem_traffic > 0.0 else 0.0
    cube_cycles = max(1.0, flops / float(macs_per_cycle))
    fixpipe_cycles = float(l0c_bytes) / float(max(1, facts.fixpipe_bytes_per_cycle))
    fp_ratio = fixpipe_cycles / cube_cycles
    stage_bonus = 1.25 if cand.stages >= 2 else 1.0
    score = (intensity * stage_bonus) / (
        (1.0 + max(0.0, fp_ratio - 1.0)) * (1.0 + half_l1_penalty)
    )

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
        half_l1_penalty=round(half_l1_penalty, 6),
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
) -> list[PrescreenScore]:
    """Batch-filter and rank candidate tile shapes by scratchpad legality and balance."""
    facts = (target or AscendTarget("ascend.950pr")).get_facts(AscendSpaceFacts)
    scored = [
        evaluate_tile_candidate(
            c,
            facts,
            simtvf=simtvf,
            dtype=dtype,
            hf32=hf32,
            l1_half_bank_split=l1_half_bank_split,
        )
        for c in candidates
    ]
    feasible = [item for item in scored if item.is_feasible]
    feasible.sort(key=lambda item: item.score, reverse=True)
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
    scored: list[tuple[float, int, CostRankedCandidate]] = []
    for idx, (cand_id, _patched_src, stage_a) in enumerate(candidates):
        ledger = getattr(stage_a, "ledger", None)
        features = build_cost_features_from_ledger(
            ledger,
            live_plan=live_plan,
            axes=axes,
            blueprint=blueprint,
            sil_advisory=sil_advisory,
            target=cost_model.target,
        )
        pred = cost_model.predict(features)
        score = float(pred.pred_us)
        if parent_pred is not None and parent_cf is not None:
            ratio = cost_model.predict_delta_ratio(features, parent_cf)
            score = float(parent_pred.pred_us) * float(ratio)
        ranked_item = CostRankedCandidate(
            candidate_id=str(cand_id),
            pred_us=score,
            tier0_physics_us=float(pred.tier0_physics_us),
            tier1_penalty_us=float(pred.tier1_penalty_us),
            dominant_bound=str(pred.dominant_bound),
            topology_bucket=str(pred.topology_bucket),
            ledger=ledger,
        )
        scored.append((ranked_item.pred_us, idx, ranked_item))
    scored.sort(key=lambda entry: (entry[0], entry[1]))
    return [entry[2] for entry in scored]


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
    axes: TopologyAxes | None = None,
    live_plan: object | None = None,
    blueprint: object | None = None,
    sil_advisory: object | None = None,
    ordered_tensor_shapes: Sequence[tuple[int, ...]] | None = None,
) -> str:
    """Run bounded Stage A live-IR dry-runs across aligned tile variants around center."""
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
    num_aiv_cores = max(1, int(ascend_space_facts(target or AscendTarget()).num_aiv_cores))

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
                    (float("inf"), 10**9, idx),
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

        if overflow_item is not None:
            sc, pk, cap = overflow_item
            status = f"REJECT({sc}_overflow:{pk}>{cap})"
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

        ii_rank = (
            float(ledger.steady_state_ii)
            if ledger.steady_state_ii is not None and ledger.steady_state_ii > 0
            else float("inf")
        )
        dom_cycles = ledger.pipe_cycles.get(
            ledger.dominant_pipe,
            max((v for v in ledger.pipe_cycles.values() if v > 0), default=0),
        )
        min_dma = int(getattr(ledger, "min_gm_dma_bytes", 0))
        raw_eff = float(getattr(ledger, "dma_burst_efficiency_ratio", 1.0))
        if 0 < min_dma < 4096:
            raw_eff = raw_eff * max(0.15, float(min_dma) / 4096.0)
        burst_eff = max(raw_eff, 0.05)
        est_waves = (
            math.ceil(math.ceil(total_rows / cand.bm) / float(num_aiv_cores))
            if (pure_vector_row_tile and total_rows > 0 and cand.bm > 0)
            else 0
        )
        wave_burst_score = (
            (float(est_waves) * float(cand.bm)) / burst_eff
            if est_waves > 0
            else (1.0 / burst_eff if pure_vector_row_tile else 0.0)
        )
        if pure_vector_row_tile and est_waves > 0:
            sort_key: tuple[float, ...] = (wave_burst_score, ii_rank, float(dom_cycles), float(idx))
        elif pure_vector_row_tile:
            sort_key = (ii_rank, -burst_eff, float(dom_cycles), float(idx))
        else:
            area = max(1.0, float(cand.bm * cand.bn))
            norm_ii = (ii_rank / area) if ii_rank != float("inf") else float("inf")
            norm_cycles = float(dom_cycles) / area
            sort_key = (norm_ii, norm_cycles, -area, float(idx))
        cost_pred: CostPrediction | None = None
        cost_score = 0.0
        if cost_model is not None:
            cf = build_cost_features_from_ledger(
                ledger,
                live_plan=live_plan,
                axes=axes,
                blueprint=blueprint,
                sil_advisory=sil_advisory,
                target=cost_model.target,
            )
            cost_pred = cost_model.predict(cf)
            cost_score = float(cost_pred.pred_us)
            if parent_pred is not None and parent_features is not None:
                ratio = cost_model.predict_delta_ratio(cf, parent_features)
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


__all__ = [
    "CostRankedCandidate",
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
]
