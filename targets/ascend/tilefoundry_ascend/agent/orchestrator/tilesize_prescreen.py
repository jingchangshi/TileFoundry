"""Concrete (BM, BN, BK) tile-size dry-run batch prescreener without bisheng."""

from __future__ import annotations

import ast
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import product

from tilefoundry_ascend.ascend.target import AscendSpaceFacts, AscendTarget


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


def evaluate_tile_candidate(
    cand: TileConfigCandidate,
    facts: AscendSpaceFacts,
    *,
    simtvf: bool = False,
) -> PrescreenScore:
    """Compute exact scratchpad occupancy and hardware balance for one tile tuple."""
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

    flops = float(2 * cand.bm * cand.bn * cand.bk)
    mem_traffic = float(l0a_slot + l0b_slot + l0c_bytes)
    intensity = flops / mem_traffic if mem_traffic > 0.0 else 0.0
    cube_cycles = max(1.0, flops / 4096.0)
    fixpipe_cycles = float(l0c_bytes) / float(max(1, facts.fixpipe_bytes_per_cycle))
    fp_ratio = fixpipe_cycles / cube_cycles
    stage_bonus = 1.25 if cand.stages >= 2 else 1.0
    score = (intensity * stage_bonus) / (1.0 + max(0.0, fp_ratio - 1.0))

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
    )


def prescreen_tile_configs(
    candidates: Iterable[TileConfigCandidate],
    *,
    target: AscendTarget | None = None,
    simtvf: bool = False,
    top_k: int | None = None,
) -> list[PrescreenScore]:
    """Batch-filter and rank candidate tile shapes by scratchpad legality and balance."""
    facts = (target or AscendTarget("ascend.950pr")).get_facts(AscendSpaceFacts)
    scored = [evaluate_tile_candidate(c, facts, simtvf=simtvf) for c in candidates]
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
    elif has_varkw:
        kw["bm"] = cand.bm

    if "bn" in params:
        kw["bn"] = cand.bn
    elif "tile_n" in params:
        kw["tile_n"] = cand.bn
    elif has_varkw:
        kw["bn"] = cand.bn

    if "bk" in params:
        kw["bk"] = cand.bk
    elif "tile_k" in params:
        kw["tile_k"] = cand.bk

    if "num_stages" in params:
        kw["num_stages"] = cand.stages
    elif "stages" in params:
        kw["stages"] = cand.stages
    return kw


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
) -> str:
    """Run bounded Stage A live-IR dry-runs across aligned tile variants around center."""
    params, has_varkw = inspect_builder_tile_params(dsl_source, factory_name)
    tune_m = has_varkw or "bm" in params or "tile_m" in params
    tune_n = has_varkw or "bn" in params or "tile_n" in params
    if not (tune_m and tune_n):
        return ""

    from tilefoundry_ascend.agent.orchestrator.gate_funnel import (
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

    tune_k = has_varkw or "bk" in params or "tile_k" in params
    tune_stages = has_varkw or "stages" in params or "num_stages" in params
    variants = _bounded_live_ir_variants(
        center,
        tune_m=tune_m,
        tune_n=tune_n,
        tune_k=tune_k,
        tune_stages=tune_stages,
        max_variants=max(1, min(8, int(max_variants))),
    )

    records: list[tuple[bool, tuple[float, int, int], TileConfigCandidate, str, dict[str, int], int, int | None]] = []
    any_ledger = False
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
        records.append(
            (
                is_ok,
                (ii_rank, dom_cycles, idx),
                cand,
                status,
                scope_bytes,
                len(degraded),
                ledger.steady_state_ii,
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
    for rank_idx, (_ok, _rk, cand, status, sbytes, deg_cnt, ii_val) in enumerate(ordered, start=1):
        wl_parts = [
            f"{scope.lower()}={sbytes.get(scope, 0)}B"
            for scope in ("L1", "L0A", "L0B", "L0C", "UB", "SSBUF")
        ]
        extra = f", ii={ii_val}" if ii_val is not None else ""
        if deg_cnt > 0:
            extra += f", degraded={deg_cnt}"
        lines.append(
            f"  {rank_idx}. bm={cand.bm}, bn={cand.bn}, bk={cand.bk}, stages={cand.stages} "
            f"| {status} | {', '.join(wl_parts)}{extra}"
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
    "PrescreenScore",
    "TileConfigCandidate",
    "count_dsl_gemm_calls",
    "evaluate_tile_candidate",
    "format_prescreen_summary",
    "generate_aligned_neighborhood",
    "inspect_builder_tile_params",
    "prescreen_live_ir_neighborhood",
    "prescreen_tile_configs",
]
