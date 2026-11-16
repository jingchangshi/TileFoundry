"""Static-dynamic residual modeling and two-lane (Exploit + Stepping-Stone) beam search."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from tilefoundry_ascend.ascend.passes.physical_ledger import ObjectivePhysicalLedger

_NOVELTY_TOKENS = (
    "T.alloc_ss",
    "T.dual_copy",
    "T.Persistent",
    "T.SimtVF",
    "T.Scope",
    "unit_flag_ctrl",
    "vgather2",
)


@dataclass(frozen=True)
class StaticDynamicResidual:
    """Gap between predicted ledger utilization and measured NPU msprof counters."""

    measured_latency_us: float
    predicted_bottleneck_util: float
    measured_aic_mac_ratio: float
    measured_aiv_vec_ratio: float
    measured_mte2_wait_ratio: float
    measured_sync_stall_ratio: float
    unexplained_gap_ratio: float

    def to_dict(self) -> dict[str, float]:
        """Return float dictionary representation."""
        return {
            "measured_latency_us": self.measured_latency_us,
            "predicted_bottleneck_util": self.predicted_bottleneck_util,
            "measured_aic_mac_ratio": self.measured_aic_mac_ratio,
            "measured_aiv_vec_ratio": self.measured_aiv_vec_ratio,
            "measured_mte2_wait_ratio": self.measured_mte2_wait_ratio,
            "measured_sync_stall_ratio": self.measured_sync_stall_ratio,
            "unexplained_gap_ratio": self.unexplained_gap_ratio,
        }


def compute_static_dynamic_residual(
    ledger: ObjectivePhysicalLedger,
    telemetry: Mapping[str, object],
) -> StaticDynamicResidual:
    """Compute StaticDynamicResidual from static ledger and msprof telemetry counters."""
    measured_us = float(telemetry.get("measured_latency_us", telemetry.get("dur_us", 0.0)))
    aic_mac = float(telemetry.get("aic_mac_ratio", telemetry.get("aic_mad", 0.0)))
    aiv_vec = float(telemetry.get("aiv_vec_ratio", telemetry.get("aiv_vec", 0.0)))
    mte2_wait = float(telemetry.get("mte2_wait_ratio", telemetry.get("aiv_mte2", 0.0)))
    sync_stall = float(telemetry.get("sync_stall_ratio", 0.0))

    positive_cycles = [v for v in ledger.pipe_cycles.values() if v > 0]
    if positive_cycles and ledger.steady_state_ii and ledger.steady_state_ii > 0:
        predicted_util = min(1.0, max(positive_cycles) / float(max(1, sum(positive_cycles))))
    elif positive_cycles:
        dom = max(positive_cycles)
        total = max(1, sum(positive_cycles))
        predicted_util = min(1.0, max(0.5, float(dom) / float(total) + 0.35))
    else:
        predicted_util = 0.0

    observed_util = max(aic_mac, aiv_vec, mte2_wait)
    raw_gap = max(0.0, predicted_util - observed_util)
    unexplained = min(1.0, max(raw_gap, sync_stall * 0.5))
    return StaticDynamicResidual(
        measured_latency_us=measured_us,
        predicted_bottleneck_util=round(predicted_util, 6),
        measured_aic_mac_ratio=aic_mac,
        measured_aiv_vec_ratio=aiv_vec,
        measured_mte2_wait_ratio=mte2_wait,
        measured_sync_stall_ratio=sync_stall,
        unexplained_gap_ratio=round(unexplained, 6),
    )


def extract_structural_signature(dsl_source: str, track: str = "") -> tuple[str, ...]:
    """Extract sorted architectural primitive signature for stepping-stone diversity."""
    tags: list[str] = []
    if track:
        tags.append(f"track:{track}")
    for token in _NOVELTY_TOKENS:
        if token in dsl_source:
            tags.append(token)
    return tuple(sorted(set(tags)))


@dataclass(frozen=True)
class BeamCandidate:
    """Single evaluated kernel candidate tracked by the two-lane beam search."""

    candidate_id: str
    round_idx: int
    worker_kind: str
    rule_id: str | None
    dsl_source: str
    track: str
    ledger: ObjectivePhysicalLedger
    residual: StaticDynamicResidual
    latency_us: float
    accuracy_passed: bool
    structural_signature: tuple[str, ...] = ()
    hypothesis: str = ""
    parent_id: str | None = None


@dataclass
class TwoLaneBeamSearch:
    """Maintains greedy Exploit beam and diversity-preserving Stepping-Stone beam."""

    exploit_width: int = 2
    stepping_stone_width: int = 2
    exploit_beam: list[BeamCandidate] = field(default_factory=list)
    stepping_stone_beam: list[BeamCandidate] = field(default_factory=list)
    seen_signatures: set[tuple[str, ...]] = field(default_factory=set)

    def update(self, new_candidates: Sequence[BeamCandidate]) -> tuple[list[BeamCandidate], list[BeamCandidate]]:
        """Update Exploit and Stepping-Stone lanes from legal, accurate candidates."""
        valid = [
            c
            for c in new_candidates
            if c.accuracy_passed and c.ledger.is_legal and c.latency_us > 0.0
        ]
        if not valid:
            return list(self.exploit_beam), list(self.stepping_stone_beam)

        exploit_pool = {c.candidate_id: c for c in self.exploit_beam}
        for cand in valid:
            exploit_pool[cand.candidate_id] = cand
        sorted_exploit = sorted(exploit_pool.values(), key=lambda c: (c.latency_us, c.candidate_id))
        self.exploit_beam = sorted_exploit[: self.exploit_width]
        exploit_ids = {c.candidate_id for c in self.exploit_beam}
        for c in self.exploit_beam:
            self.seen_signatures.add(c.structural_signature)

        stone_pool = [
            c
            for c in (*self.stepping_stone_beam, *valid)
            if c.candidate_id not in exploit_ids
        ]
        deduped_stones: dict[tuple[str, ...], BeamCandidate] = {}
        for cand in stone_pool:
            sig = cand.structural_signature or (cand.candidate_id,)
            prev = deduped_stones.get(sig)
            if prev is None or cand.latency_us < prev.latency_us:
                deduped_stones[sig] = cand

        novel_ranked = sorted(
            deduped_stones.values(),
            key=lambda c: (
                0 if c.structural_signature not in self.seen_signatures else 1,
                -len(c.structural_signature),
                c.latency_us,
                c.candidate_id,
            ),
        )
        self.stepping_stone_beam = novel_ranked[: self.stepping_stone_width]
        for c in self.stepping_stone_beam:
            self.seen_signatures.add(c.structural_signature)
        return list(self.exploit_beam), list(self.stepping_stone_beam)

    def parent_pool(self) -> list[BeamCandidate]:
        """Return combined unique parent candidates across both beams."""
        merged: dict[str, BeamCandidate] = {}
        for c in (*self.exploit_beam, *self.stepping_stone_beam):
            merged[c.candidate_id] = c
        return list(merged.values())


__all__ = [
    "BeamCandidate",
    "StaticDynamicResidual",
    "TwoLaneBeamSearch",
    "compute_static_dynamic_residual",
    "extract_structural_signature",
]
