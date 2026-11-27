"""Statistically gated CostModel lifecycle policy and paradigm-shift exemption."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from tilefoundry_ascend.ascend.sil.schema import TopologyAxes


class CostModelMode(StrEnum):
    """Execution mode for the stage-A cost-model advisory gate."""

    OFF = "off"
    SHADOW = "shadow"
    ADVISORY = "advisory"


@dataclass(frozen=True, slots=True)
class CostGateDecision:
    """Immutable lifecycle gate decision governing advisory prompt guidance."""

    mode: CostModelMode
    engaged: bool
    reason: str
    allow_prompt_guidance: bool
    allow_candidate_pruning: bool = False


def is_paradigm_shift_exempt(
    cand_axes: TopologyAxes | None,
    parent_axes: TopologyAxes | None,
    *,
    cand_workspace_stages: int = 0,
    parent_workspace_stages: int = 0,
    is_creative: bool = False,
) -> bool:
    """Return True when a candidate crosses topology/workspace paradigms or is creative."""
    if is_creative or cand_workspace_stages != parent_workspace_stages:
        return True
    return cand_axes is not None and parent_axes is not None and cand_axes != parent_axes


def _gate(mode: CostModelMode, engaged: bool, reason: str) -> CostGateDecision:
    return CostGateDecision(
        mode=mode,
        engaged=engaged,
        reason=reason,
        allow_prompt_guidance=engaged,
        allow_candidate_pruning=False,
    )


def evaluate_cost_gate_policy(
    *,
    mode: CostModelMode | str = CostModelMode.OFF,
    enable_prescreen_flag: bool = True,
    bucket_clean_samples: int = 0,
    out_of_sample_spearman_rho: float | None = None,
) -> CostGateDecision:
    """Evaluate whether advisory cost-model guidance should engage for the current round."""
    resolved = (
        CostModelMode.OFF
        if not enable_prescreen_flag
        else CostModelMode(str(mode).strip().lower())
    )
    if resolved == CostModelMode.OFF:
        return _gate(resolved, False, "disabled_by_config")
    if resolved == CostModelMode.SHADOW:
        return _gate(resolved, False, "shadow_observation_only")

    samples = int(bucket_clean_samples)
    if samples < 8:
        return _gate(resolved, False, "insufficient_bucket_samples")
    if out_of_sample_spearman_rho is None or not math.isfinite(float(out_of_sample_spearman_rho)):
        return _gate(resolved, False, "missing_spearman_rho")
    rho = float(out_of_sample_spearman_rho)
    if rho >= 1.0:
        return _gate(resolved, True, "advisory_t_stat_validated")
    if rho <= 0.0:
        return _gate(resolved, False, "spearman_t_stat_below_threshold")
    t_stat = rho * math.sqrt(float(samples - 2) / (1.0 - rho * rho))
    if t_stat > 2.0:
        return _gate(resolved, True, "advisory_t_stat_validated")
    return _gate(resolved, False, "spearman_t_stat_below_threshold")


__all__ = [
    "CostGateDecision",
    "CostModelMode",
    "evaluate_cost_gate_policy",
    "is_paradigm_shift_exempt",
]
