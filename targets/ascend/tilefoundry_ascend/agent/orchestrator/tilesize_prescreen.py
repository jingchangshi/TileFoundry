"""Concrete (BM, BN, BK) tile-size dry-run batch prescreener without bisheng."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import product

from tilefoundry_ascend.ascend.target import AscendSpaceFacts, AscendTarget


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
    l0c_bytes = cand.bm * cand.bn * cand.accum_bytes
    ub_rows = max(1, (cand.bm + cand.aiv_split - 1) // cand.aiv_split)
    ub_bytes = cand.stages * ub_rows * cand.bn * cand.accum_bytes

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
            )
        )
    return tuple(neighbors)


def format_prescreen_summary(scores: Sequence[PrescreenScore]) -> str:
    """Format ranked PrescreenScore entries into a concise prompt-ready summary."""
    if not scores:
        return "No feasible tile configurations found."
    lines = ["Tile Prescreen Summary:"]
    for idx, item in enumerate(scores, start=1):
        c = item.candidate
        status = "OK" if item.is_feasible else f"REJECT({item.rejection_reason or 'invalid'})"
        lines.append(
            f"  {idx}. bm={c.bm}, bn={c.bn}, bk={c.bk}, stages={c.stages}, "
            f"aiv_split={c.aiv_split} | {status} | l1={item.l1_bytes}B, "
            f"l0c={item.l0c_bytes}B, ub_aiv={item.ub_bytes_per_aiv}B | "
            f"ai={item.arithmetic_intensity:.2f}, fp_ratio={item.fixpipe_to_cube_ratio:.2f}, "
            f"score={item.score:.3f}"
        )
    return "\n".join(lines)


__all__ = [
    "PrescreenScore",
    "TileConfigCandidate",
    "evaluate_tile_candidate",
    "format_prescreen_summary",
    "generate_aligned_neighborhood",
    "prescreen_tile_configs",
]
