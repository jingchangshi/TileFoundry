"""CEGAR guardrail promotion gate, L3 contract coordination, and atomic rollback."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from tilefoundry_ascend.agent.campaign.funnel_telemetry import MAX_CLEAN_CALIBRATION_BURST_CV
from tilefoundry_ascend.agent.campaign.helpers import to_conservation_point
from tilefoundry_ascend.agent.npu import (
    FAILURE_KIND_CANDIDATE,
    AnchorEvalMetrics,
    CandidateSurrogateResult,
    L3AcceptanceResult,
    aggregate_surrogate_result,
)
from tilefoundry_ascend.agent.specs.anchor_specs import AnchorCaseSpec, GuardrailCaseSpec, RegimeTargetSpec
from tilefoundry_ascend.agent.specs.fault_types import L3Resolution
from tilefoundry_ascend.agent.specs.latency_algebra import (
    AnchorConservationPoint,
    check_family_and_sibling_non_regression,
    check_guardrail_non_regression,
    check_regime_net_improvement,
)
from tilefoundry_ascend.ascend.json_record import JsonRecordMixin

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PendingPromotion(JsonRecordMixin):
    """Candidate held for promotion after an infrastructure L3 deferral."""

    candidate_source: str
    candidate_sha256: str = ""
    target_path: str = ""
    round_idx: int = 0
    regime_tag: str = ""
    candidate_result: CandidateSurrogateResult | None = None
    candidate_id: str = ""
    builder_name: str = ""
    parent_lane: str = ""
    creative_effort: str = ""
    execution_id: str = ""
    run_id: str = ""
    receipt_path: str = ""
    error_message: str = ""
    attempts: tuple[dict[str, str], ...] = ()
    device_id: str = ""
    execution_terminal: bool = True
    retry_used: bool = False

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> PendingPromotion:
        """Reconstruct a pending promotion record from a checkpoint mapping."""
        if not isinstance(payload.get("candidate_result"), Mapping):
            raise ValueError("pending_promotion.candidate_result must be a mapping")
        return super().from_dict(payload)


def classify_l3_resolution(
    l3_res: L3AcceptanceResult, *, execution_terminal: bool | None = None
) -> L3Resolution:
    """Classify an L3 acceptance outcome into orthogonal promotion resolution states."""
    if l3_res.passed:
        return L3Resolution.PROMOTED
    is_terminal = bool(l3_res.execution_terminal) if execution_terminal is None else bool(execution_terminal)
    if l3_res.failure_kind == FAILURE_KIND_CANDIDATE and is_terminal:
        return L3Resolution.CANDIDATE_REJECTED
    return L3Resolution.INFRA_DEFERRED


def _guardrail_point(g_metric: AnchorEvalMetrics, *, inc_us: float, paired: bool) -> AnchorConservationPoint:
    return AnchorConservationPoint(
        case_num=int(g_metric.case_num), regime_tag=str(g_metric.regime_tag),
        ref_us=float(g_metric.ref_perf_us), inc_us=inc_us, cand_us=float(g_metric.cand_us),
        t_roof_us=float(g_metric.t_hw_us),
        paired_rel_mad=float(getattr(g_metric, "paired_rel_mad", 0.0) or 0.0),
        accuracy_passed=bool(g_metric.accuracy_passed),
        cand_burst_cv=float(g_metric.burst_batch_cv) if paired else 0.0,
        ref_burst_cv=0.0,
        burst_count=max(1, int(g_metric.ring_burst_k)) if paired else 1,
    )


def _maybe_retest_noisy(
    anchors: Sequence[AnchorEvalMetrics],
    cur_res: CandidateSurrogateResult,
    retest_used: bool,
    retest_fn: Callable[[Sequence[AnchorEvalMetrics], CandidateSurrogateResult], CandidateSurrogateResult | None] | None,
) -> tuple[bool, CandidateSurrogateResult | None]:
    noisy = any(float(getattr(a, "burst_batch_cv", 0.0) or 0.0) > MAX_CLEAN_CALIBRATION_BURST_CV for a in anchors)
    if retest_used or not noisy or retest_fn is None:
        return False, None
    return True, retest_fn(anchors, cur_res)


def evaluate_promotion_gate(
    *,
    candidate_id: str,
    cur_res: CandidateSurrogateResult,
    best_result: CandidateSurrogateResult,
    regime_spec: RegimeTargetSpec,
    guardrail_cases: Sequence[GuardrailCaseSpec],
    cand_guardrail_metrics: Mapping[int, AnchorEvalMetrics],
    unchanged_siblings: set[str],
    active_anchors: Sequence[AnchorCaseSpec],
    retest_noisy_fn: Callable[[Sequence[AnchorEvalMetrics], CandidateSurrogateResult], CandidateSurrogateResult | None] | None = None,
    recheck_single_sibling_fn: Callable[[AnchorCaseSpec], CandidateSurrogateResult | None] | None = None,
) -> tuple[bool, CandidateSurrogateResult, str]:
    """Enforce unbounded CEGAR guardrails, 3-sigma non-regression, and 2-sigma gain."""
    _ = candidate_id
    cur_by_case = {int(a.case_num): a for a in cur_res.anchors}
    for guardrail in guardrail_cases:
        g_m = cand_guardrail_metrics.get(int(guardrail.case_num)) or cur_by_case.get(int(guardrail.case_num))
        if g_m is None:
            continue
        if not g_m.accuracy_passed:
            return False, cur_res, "guardrail_accuracy_failed"
        paired_sp = float(getattr(g_m, "paired_speedup_vs_incumbent", 0.0) or 0.0)
        if paired_sp > 0.0 and float(g_m.cand_us) > 0.0:
            if not check_guardrail_non_regression(_guardrail_point(g_m, inc_us=float(g_m.cand_us) * paired_sp, paired=True), sigma_mult=3.0):
                return False, cur_res, "guardrail_latency_regression"
        elif float(guardrail.incumbent_ref_us) > 0.0:
            if not check_guardrail_non_regression(_guardrail_point(g_m, inc_us=float(guardrail.incumbent_ref_us), paired=False), sigma_mult=3.0):
                return False, cur_res, "guardrail_latency_regression"

    best_by_case = {int(a.case_num): a for a in best_result.anchors}
    retest_used = False
    for _attempt in range(2):
        active_cand = [a for a in cur_res.anchors if a.regime_tag == regime_spec.regime_tag] or list(cur_res.anchors)
        active_ok, active_failed = check_family_and_sibling_non_regression(
            [to_conservation_point(a, best_by_case.get(int(a.case_num))) for a in active_cand], (), sigma_mult=3.0
        )
        if not active_ok:
            regressed = [a for a in active_cand if int(a.case_num) in set(active_failed)]
            did_try, retested = _maybe_retest_noisy(regressed, cur_res, retest_used, retest_noisy_fn)
            if did_try and retested is not None:
                if not retested.all_passed:
                    return False, cur_res, "recheck_failed"
                cur_res, retest_used = retested, True
                continue
            return False, cur_res, "active_tail_regression"

        sibling_cand = [
            a for a in cur_res.anchors
            if a.regime_tag != regime_spec.regime_tag and a.regime_tag not in unchanged_siblings and int(a.case_num) in best_by_case
        ]
        sib_ok, sib_failed = check_family_and_sibling_non_regression(
            (), [to_conservation_point(a, best_by_case.get(int(a.case_num))) for a in sibling_cand], sigma_mult=3.0
        )
        if not sib_ok:
            dipped = [a for a in sibling_cand if int(a.case_num) in set(sib_failed)]
            if len(dipped) >= 2:
                did_try, retested = _maybe_retest_noisy(dipped, cur_res, retest_used, retest_noisy_fn)
                if did_try and retested is not None:
                    if not retested.all_passed:
                        return False, cur_res, "recheck_failed"
                    cur_res, retest_used = retested, True
                    continue
                return False, cur_res, "multi_sibling_regression"
            if len(dipped) == 1:
                if recheck_single_sibling_fn is None:
                    return False, cur_res, "sibling_regression"
                d_anc = dipped[0]
                d_spec = {int(a.case_num): a for a in active_anchors}.get(int(d_anc.case_num)) or AnchorCaseSpec(
                    task=f"operator/{cur_res.operator_name}", case_num=int(d_anc.case_num),
                    regime_tag=d_anc.regime_tag, ref_perf_us=d_anc.ref_perf_us, t_hw_us=d_anc.t_hw_us,
                )
                recheck_res = recheck_single_sibling_fn(d_spec)
                recheck_anc = (
                    next((a for a in recheck_res.anchors if int(a.case_num) == int(d_anc.case_num)), None)
                    if recheck_res is not None and recheck_res.all_passed else None
                )
                if recheck_anc is None or not check_guardrail_non_regression(
                    to_conservation_point(recheck_anc, best_by_case.get(int(d_anc.case_num))), sigma_mult=3.0
                ):
                    return False, cur_res, "recheck_failed"
                merged = tuple(recheck_anc if int(a.case_num) == int(d_anc.case_num) else a for a in cur_res.anchors)
                cur_res = aggregate_surrogate_result(
                    cur_res.candidate_id, cur_res.operator_name, merged, error_message=cur_res.error_message
                )
        break

    final_best = {int(a.case_num): a for a in best_result.anchors}
    final_active = [a for a in cur_res.anchors if a.regime_tag == regime_spec.regime_tag] or list(cur_res.anchors)
    gain_pts = [to_conservation_point(a, final_best.get(int(a.case_num))) for a in final_active if int(a.case_num) in final_best]
    if gain_pts:
        gain_ok, _, _ = check_regime_net_improvement(gain_pts, sigma_mult=2.0)
        if not gain_ok:
            all_pts = [to_conservation_point(a, final_best.get(int(a.case_num))) for a in cur_res.anchors if int(a.case_num) in final_best]
            global_ok, _, _ = check_regime_net_improvement(all_pts, sigma_mult=2.0)
            if not global_ok:
                return False, cur_res, "insufficient_gain"
    elif cur_res.surrogate_score <= best_result.surrogate_score:
        return False, cur_res, "insufficient_gain"
    return True, cur_res, ""


@dataclass(frozen=True)
class RankedL2Candidate:
    """Single L2-passing candidate eligible for ranked L3 promotion."""

    candidate_id: str
    candidate_source: str
    candidate_result: CandidateSurrogateResult
    candidate_sha256: str = ""
    builder_name: str = ""
    parent_lane: str = ""
    creative_effort: str = ""
    rank: int = 0
    speedup: float = 0.0


@dataclass(frozen=True)
class RankedL2Survivors:
    """Top-2 L2 Pareto survivors ordered by descending speedup."""

    survivors: tuple[RankedL2Candidate, ...] = ()
    rejected: tuple[tuple[str, str], ...] = ()

    @property
    def rank_1(self) -> RankedL2Candidate | None:
        """Return the first-ranked L2 survivor if present."""
        return self.survivors[0] if self.survivors else None

    @property
    def rank_2(self) -> RankedL2Candidate | None:
        """Return the second-ranked L2 survivor if present."""
        return self.survivors[1] if len(self.survivors) > 1 else None


@dataclass(frozen=True)
class PromotionDecision:
    """Terminal decision from Top-2 L3 promotion coordination."""

    resolution: L3Resolution
    promoted_candidate: RankedL2Candidate | None = None
    l3_result: L3AcceptanceResult | None = None
    pending_promotion: PendingPromotion | None = None
    attempts: tuple[tuple[str, L3Resolution, L3AcceptanceResult], ...] = ()
    new_guardrail_cases: tuple[int, ...] = ()
    fallback_used: bool = False


def select_top_two_l2_survivors(
    candidates: Sequence[RankedL2Candidate],
    *,
    best_result: CandidateSurrogateResult,
    regime_spec: RegimeTargetSpec,
    guardrail_cases: Sequence[GuardrailCaseSpec] = (),
    guardrail_metrics_by_cid: Mapping[str, Mapping[int, AnchorEvalMetrics]] | None = None,
    unchanged_siblings_by_cid: Mapping[str, set[str]] | None = None,
    active_anchors: Sequence[AnchorCaseSpec] = (),
    retest_noisy_fn: Callable[[Sequence[AnchorEvalMetrics], CandidateSurrogateResult], CandidateSurrogateResult | None] | None = None,
    recheck_single_sibling_fn: Callable[[AnchorCaseSpec], CandidateSurrogateResult | None] | None = None,
) -> RankedL2Survivors:
    """Filter L2 candidates via promotion gate and rank the top 2 by descending speedup."""
    qualified, rejected = [], []
    g_map, u_map = guardrail_metrics_by_cid or {}, unchanged_siblings_by_cid or {}
    for cand in candidates:
        if not cand.candidate_result.all_passed:
            rejected.append((cand.candidate_id, "l2_not_passed"))
            continue
        ok, updated_res, reason = evaluate_promotion_gate(
            candidate_id=cand.candidate_id, cur_res=cand.candidate_result, best_result=best_result,
            regime_spec=regime_spec, guardrail_cases=guardrail_cases,
            cand_guardrail_metrics=g_map.get(cand.candidate_id, {}),
            unchanged_siblings=u_map.get(cand.candidate_id, set()), active_anchors=active_anchors,
            retest_noisy_fn=retest_noisy_fn, recheck_single_sibling_fn=recheck_single_sibling_fn,
        )
        if not ok:
            rejected.append((cand.candidate_id, reason or "gate_rejected"))
        else:
            qualified.append(replace(cand, candidate_result=updated_res, speedup=float(updated_res.avg_speedup)))
    qualified.sort(key=lambda it: (-float(it.speedup), -float(it.candidate_result.surrogate_score), it.candidate_id))
    return RankedL2Survivors(
        survivors=tuple(replace(item, rank=idx) for idx, item in enumerate(qualified[:2], start=1)),
        rejected=tuple(rejected),
    )


class PromotionCoordinator:
    """Coordinate CEGAR guardrail updates and L3 candidate commit/rollback."""

    def __init__(self, *, target_module_path: Path, write_and_verify_fn: Callable[[Path, str], str]) -> None:
        self._target_module_path = target_module_path
        self._write_and_verify = write_and_verify_fn

    def rollback_to_incumbent(self, incumbent_source: str) -> None:
        """Atomically restore incumbent source and verify written bytes."""
        err = self._write_and_verify(self._target_module_path, incumbent_source)
        if err:
            raise RuntimeError(err)

    def coordinate_top_two_l3_promotion(
        self,
        survivors: RankedL2Survivors,
        *,
        incumbent_source: str,
        round_idx: int,
        regime_tag: str,
        run_l3_fn: Callable[[RankedL2Candidate], L3AcceptanceResult],
        target_path: str = "",
    ) -> PromotionDecision:
        """Run L3 acceptance for rank-1 survivor with same-round rank-2 fallback."""
        if not survivors.survivors:
            return PromotionDecision(resolution=L3Resolution.NOT_RUN)
        resolved_target = target_path or str(self._target_module_path)
        attempts: list[tuple[str, L3Resolution, L3AcceptanceResult]] = []
        failed_cases: list[int] = []
        for idx, cand in enumerate(survivors.survivors[:2]):
            fallback_used = idx > 0
            err = self._write_and_verify(self._target_module_path, cand.candidate_source)
            if err:
                raise RuntimeError(err)
            l3_res = run_l3_fn(cand)
            resolution = classify_l3_resolution(l3_res)
            attempts.append((cand.candidate_id, resolution, l3_res))
            for cnum in l3_res.failed_case_nums:
                if int(cnum) not in failed_cases:
                    failed_cases.append(int(cnum))
            if resolution is L3Resolution.PROMOTED:
                return PromotionDecision(
                    resolution=L3Resolution.PROMOTED, promoted_candidate=cand, l3_result=l3_res,
                    attempts=tuple(attempts), new_guardrail_cases=tuple(failed_cases), fallback_used=fallback_used,
                )
            if resolution is L3Resolution.INFRA_DEFERRED:
                self.rollback_to_incumbent(incumbent_source)
                pending = PendingPromotion(
                    candidate_source=cand.candidate_source,
                    candidate_sha256=l3_res.candidate_sha256 or cand.candidate_sha256,
                    target_path=l3_res.target_path or resolved_target, round_idx=round_idx,
                    regime_tag=regime_tag, candidate_result=cand.candidate_result,
                    candidate_id=cand.candidate_id, builder_name=cand.builder_name,
                    parent_lane=cand.parent_lane, creative_effort=cand.creative_effort,
                    execution_id=l3_res.execution_id, run_id=l3_res.run_id,
                    receipt_path=l3_res.receipt_path, error_message=l3_res.error_message,
                    device_id=l3_res.device_id, execution_terminal=l3_res.execution_terminal,
                )
                return PromotionDecision(
                    resolution=L3Resolution.INFRA_DEFERRED, l3_result=l3_res, pending_promotion=pending,
                    attempts=tuple(attempts), new_guardrail_cases=tuple(failed_cases), fallback_used=fallback_used,
                )
        self.rollback_to_incumbent(incumbent_source)
        return PromotionDecision(
            resolution=L3Resolution.CANDIDATE_REJECTED, l3_result=attempts[-1][2],
            attempts=tuple(attempts), new_guardrail_cases=tuple(failed_cases), fallback_used=len(attempts) > 1,
        )


__all__ = [
    "PendingPromotion",
    "PromotionCoordinator",
    "PromotionDecision",
    "RankedL2Candidate",
    "RankedL2Survivors",
    "classify_l3_resolution",
    "evaluate_promotion_gate",
    "select_top_two_l2_survivors",
    "to_conservation_point",
]
