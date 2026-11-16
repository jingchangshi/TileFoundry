"""Concrete (BM, BN, BK) tile-size dry-run batch prescreener without bisheng."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

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


__all__ = [
    "PrescreenScore",
    "TileConfigCandidate",
    "evaluate_tile_candidate",
    "prescreen_tile_configs",
]
