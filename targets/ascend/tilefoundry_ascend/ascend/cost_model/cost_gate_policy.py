"""Adaptive CostModel lifecycle policy and paradigm-shift exemption state machine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from tilefoundry_ascend.ascend.sil.schema import TopologyAxes


class CostModelMode(StrEnum):
    """Execution mode for the stage-A cost-model prescreen gate."""

    OFF = "off"
    SHADOW = "shadow"
    AUTO = "auto"
    COARSE_ONLY = "coarse_only"


@dataclass(frozen=True, slots=True)
class CostGateDecision:
    """Immutable lifecycle gate decision governing prescreen pruning and prompt guidance."""

    mode: CostModelMode
    engaged: bool
    reason: str
    coarse_reject_ratio: float
    allow_prompt_guidance: bool
    allow_candidate_pruning: bool


def is_paradigm_shift_exempt(
    cand_axes: TopologyAxes | None,
    parent_axes: TopologyAxes | None,
    *,
    cand_workspace_stages: int = 0,
    parent_workspace_stages: int = 0,
    is_creative: bool = False,
) -> bool:
    """Return True when a candidate crosses topology/workspace paradigms or is creative."""
    if is_creative:
        return True
    if cand_workspace_stages != parent_workspace_stages:
        return True
    if cand_axes is not None and parent_axes is not None and cand_axes != parent_axes:
        return True
    return False


def evaluate_cost_gate_policy(
    *,
    mode: CostModelMode | str = CostModelMode.OFF,
    enable_prescreen_flag: bool = True,
    round_idx: int = 1,
    stagnation_rounds: int = 0,
    incumbent_roofline_eta: float = 0.0,
    target_roofline_eta: float = 0.85,
    prev_local_rel_gain: float | None = None,
    mean_recoverable_excess_us: float | None = None,
    disengage_roofline_eta: float = 0.65,
    slow_gain_threshold: float = 0.03,
    min_recoverable_excess_us: float = 1.0,
    coarse_reject_ratio: float = 3.0,
) -> CostGateDecision:
    """Evaluate whether the adaptive cost-model gate should engage for the current round."""
    resolved_mode = (
        CostModelMode.OFF
        if not enable_prescreen_flag
        else CostModelMode(str(mode).strip().lower())
    )
    if resolved_mode == CostModelMode.OFF:
        return CostGateDecision(
            mode=resolved_mode,
            engaged=False,
            reason="disabled_by_config",
            coarse_reject_ratio=float("inf"),
            allow_prompt_guidance=False,
            allow_candidate_pruning=False,
        )
    if resolved_mode == CostModelMode.SHADOW:
        return CostGateDecision(
            mode=resolved_mode,
            engaged=False,
            reason="shadow_observation_only",
            coarse_reject_ratio=float("inf"),
            allow_prompt_guidance=False,
            allow_candidate_pruning=False,
        )
    if resolved_mode == CostModelMode.COARSE_ONLY:
        return CostGateDecision(
            mode=resolved_mode,
            engaged=True,
            reason="forced_coarse_only",
            coarse_reject_ratio=float(coarse_reject_ratio),
            allow_prompt_guidance=True,
            allow_candidate_pruning=True,
        )

    effective_eta_cap = min(
        float(disengage_roofline_eta),
        max(0.50, float(target_roofline_eta) - 0.15),
    )
    if float(incumbent_roofline_eta) >= effective_eta_cap:
        return CostGateDecision(
            mode=resolved_mode,
            engaged=False,
            reason="near_theoretical_limit",
            coarse_reject_ratio=float("inf"),
            allow_prompt_guidance=False,
            allow_candidate_pruning=False,
        )
    if (
        mean_recoverable_excess_us is not None
        and float(mean_recoverable_excess_us) <= float(min_recoverable_excess_us)
    ):
        return CostGateDecision(
            mode=resolved_mode,
            engaged=False,
            reason="near_launch_floor",
            coarse_reject_ratio=float("inf"),
            allow_prompt_guidance=False,
            allow_candidate_pruning=False,
        )
    if int(stagnation_rounds) >= 1:
        return CostGateDecision(
            mode=resolved_mode,
            engaged=False,
            reason="stagnation_detected",
            coarse_reject_ratio=float("inf"),
            allow_prompt_guidance=False,
            allow_candidate_pruning=False,
        )
    if (
        int(round_idx) >= 2
        and prev_local_rel_gain is not None
        and float(prev_local_rel_gain) < float(slow_gain_threshold)
    ):
        return CostGateDecision(
            mode=resolved_mode,
            engaged=False,
            reason="slow_iteration_velocity",
            coarse_reject_ratio=float("inf"),
            allow_prompt_guidance=False,
            allow_candidate_pruning=False,
        )
    return CostGateDecision(
        mode=resolved_mode,
        engaged=True,
        reason="early_coarse_prescreen",
        coarse_reject_ratio=float(coarse_reject_ratio),
        allow_prompt_guidance=True,
        allow_candidate_pruning=True,
    )


__all__ = [
    "CostGateDecision",
    "CostModelMode",
    "evaluate_cost_gate_policy",
    "is_paradigm_shift_exempt",
]
