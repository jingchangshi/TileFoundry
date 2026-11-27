"""CEGAR guardrail promotion gate, L3 contract coordination, and atomic rollback."""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path

from tilefoundry_ascend.agent.specs.anchor_specs import (
    AnchorCaseSpec,
    GuardrailCaseSpec,
    RegimeTargetSpec,
)
from tilefoundry_ascend.agent.specs.fault_types import L3Resolution
from tilefoundry_ascend.agent.campaign.funnel_telemetry import MAX_CLEAN_CALIBRATION_BURST_CV
from tilefoundry_ascend.agent.specs.latency_algebra import (
    AnchorConservationPoint,
    check_family_and_sibling_non_regression,
    check_guardrail_non_regression,
    check_regime_net_improvement,
)
from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (
    FAILURE_KIND_CANDIDATE,
    AnchorEvalMetrics,
    CandidateSurrogateResult,
    L3AcceptanceResult,
    aggregate_surrogate_result,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PendingPromotion:
    """Candidate held for promotion after an infrastructure L3 deferral."""

    candidate_source: str
    candidate_sha256: str
    target_path: str
    round_idx: int
    regime_tag: str
    candidate_result: CandidateSurrogateResult
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

    def to_dict(self) -> dict[str, object]:
        """Serialize the pending promotion record for checkpoints and reports."""
        return {
            "candidate_source": self.candidate_source,
            "candidate_sha256": self.candidate_sha256,
            "target_path": self.target_path,
            "round_idx": self.round_idx,
            "regime_tag": self.regime_tag,
            "candidate_result": self.candidate_result.to_dict(),
            "candidate_id": self.candidate_id,
            "builder_name": self.builder_name,
            "parent_lane": self.parent_lane,
            "creative_effort": self.creative_effort,
            "execution_id": self.execution_id,
            "run_id": self.run_id,
            "receipt_path": self.receipt_path,
            "error_message": self.error_message,
            "attempts": [dict(item) for item in self.attempts],
            "device_id": self.device_id,
            "execution_terminal": self.execution_terminal,
            "retry_used": self.retry_used,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> PendingPromotion:
        """Reconstruct a pending promotion record from a checkpoint mapping."""
        raw_result = payload.get("candidate_result")
        if not isinstance(raw_result, Mapping):
            raise ValueError("pending_promotion.candidate_result must be a mapping")
        raw_attempts = payload.get("attempts", ())
        attempts = tuple(
            {str(k): str(v) for k, v in item.items()}
            for item in raw_attempts
            if isinstance(item, Mapping)
        )
        return cls(
            candidate_source=str(payload["candidate_source"]),
            candidate_sha256=str(payload.get("candidate_sha256", "")),
            target_path=str(payload.get("target_path", "")),
            round_idx=int(payload.get("round_idx", 0)),
            regime_tag=str(payload.get("regime_tag", "")),
            candidate_result=CandidateSurrogateResult.from_dict(raw_result),
            candidate_id=str(payload.get("candidate_id", "")),
            builder_name=str(payload.get("builder_name", "")),
            parent_lane=str(payload.get("parent_lane", "")),
            creative_effort=str(payload.get("creative_effort", "")),
            execution_id=str(payload.get("execution_id", "")),
            run_id=str(payload.get("run_id", "")),
            receipt_path=str(payload.get("receipt_path", "")),
            error_message=str(payload.get("error_message", "")),
            attempts=attempts,
            device_id=str(payload.get("device_id", "")),
            execution_terminal=bool(payload.get("execution_terminal", True)),
            retry_used=bool(payload.get("retry_used", False)),
        )


def classify_l3_resolution(
    l3_res: L3AcceptanceResult,
    *,
    execution_terminal: bool | None = None,
) -> L3Resolution:
    """Classify an L3 acceptance outcome into orthogonal promotion resolution states."""
    if l3_res.passed:
        return L3Resolution.PROMOTED
    is_terminal = (
        bool(l3_res.execution_terminal) if execution_terminal is None else bool(execution_terminal)
    )
    if l3_res.failure_kind == FAILURE_KIND_CANDIDATE and is_terminal:
        return L3Resolution.CANDIDATE_REJECTED
    return L3Resolution.INFRA_DEFERRED


def to_conservation_point(
    cand: AnchorEvalMetrics,
    inc: AnchorEvalMetrics | None,
) -> AnchorConservationPoint:
    """Convert candidate and incumbent anchor metrics into a conservation point."""
    inc_us = float(inc.cand_us) if (inc is not None and inc.cand_us > 0.0) else float(cand.cand_us)
    is_paired = float(getattr(cand, "paired_speedup_vs_incumbent", 0.0) or 0.0) > 0.0
    if is_paired:
        mad = float(getattr(cand, "paired_rel_mad", 0.0) or 0.0)
        ref_burst_cv = 0.0
    else:
        mad = max(
            float(getattr(cand, "paired_rel_mad", 0.0) or 0.0),
            float(getattr(inc, "paired_rel_mad", 0.0) or 0.0) if inc is not None else 0.0,
        )
        ref_burst_cv = float(inc.burst_batch_cv) if inc is not None else 0.0
    point_fields = {f.name for f in fields(AnchorConservationPoint)}
    if {"cand_burst_cv", "ref_burst_cv", "burst_count"} <= point_fields:
        return AnchorConservationPoint(
            case_num=int(cand.case_num),
            regime_tag=str(cand.regime_tag),
            ref_us=float(cand.ref_perf_us),
            inc_us=inc_us,
            cand_us=float(cand.cand_us),
            t_roof_us=float(cand.t_hw_us),
            paired_rel_mad=mad,
            accuracy_passed=bool(cand.accuracy_passed),
            cand_burst_cv=float(cand.burst_batch_cv),
            ref_burst_cv=ref_burst_cv,
            burst_count=max(1, int(cand.ring_burst_k)),
        )
    return AnchorConservationPoint(
        case_num=int(cand.case_num),
        regime_tag=str(cand.regime_tag),
        ref_us=float(cand.ref_perf_us),
        inc_us=inc_us,
        cand_us=float(cand.cand_us),
        t_roof_us=float(cand.t_hw_us),
        paired_rel_mad=mad,
        accuracy_passed=bool(cand.accuracy_passed),
    )


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
        g_metric = cand_guardrail_metrics.get(int(guardrail.case_num))
        if g_metric is None:
            g_metric = cur_by_case.get(int(guardrail.case_num))
        if g_metric is None:
            continue
        if not g_metric.accuracy_passed:
            return False, cur_res, "guardrail_accuracy_failed"
        inc_ref = float(guardrail.incumbent_ref_us)
        paired_sp = float(getattr(g_metric, "paired_speedup_vs_incumbent", 0.0) or 0.0)
        if paired_sp > 0.0 and float(g_metric.cand_us) > 0.0:
            eff_cand_us = float(g_metric.cand_us)
            eff_inc_us = eff_cand_us * paired_sp
            point_fields = {f.name for f in fields(AnchorConservationPoint)}
            if {"cand_burst_cv", "ref_burst_cv", "burst_count"} <= point_fields:
                g_point = AnchorConservationPoint(
                    case_num=int(g_metric.case_num),
                    regime_tag=str(g_metric.regime_tag),
                    ref_us=float(g_metric.ref_perf_us),
                    inc_us=eff_inc_us,
                    cand_us=eff_cand_us,
                    t_roof_us=float(g_metric.t_hw_us),
                    paired_rel_mad=float(getattr(g_metric, "paired_rel_mad", 0.0) or 0.0),
                    accuracy_passed=bool(g_metric.accuracy_passed),
                    cand_burst_cv=float(g_metric.burst_batch_cv),
                    ref_burst_cv=0.0,
                    burst_count=max(1, int(g_metric.ring_burst_k)),
                )
            else:
                g_point = AnchorConservationPoint(
                    case_num=int(g_metric.case_num),
                    regime_tag=str(g_metric.regime_tag),
                    ref_us=float(g_metric.ref_perf_us),
                    inc_us=eff_inc_us,
                    cand_us=eff_cand_us,
                    t_roof_us=float(g_metric.t_hw_us),
                    paired_rel_mad=float(getattr(g_metric, "paired_rel_mad", 0.0) or 0.0),
                    accuracy_passed=bool(g_metric.accuracy_passed),
                )
            if not check_guardrail_non_regression(g_point, sigma_mult=3.0):
                return False, cur_res, "guardrail_latency_regression"
        elif inc_ref > 0.0:
            g_point = AnchorConservationPoint(
                case_num=int(g_metric.case_num),
                regime_tag=str(g_metric.regime_tag),
                ref_us=float(g_metric.ref_perf_us),
                inc_us=inc_ref,
                cand_us=float(g_metric.cand_us),
                t_roof_us=float(g_metric.t_hw_us),
                paired_rel_mad=float(getattr(g_metric, "paired_rel_mad", 0.0) or 0.0),
                accuracy_passed=bool(g_metric.accuracy_passed),
            )
            if not check_guardrail_non_regression(g_point, sigma_mult=3.0):
                return False, cur_res, "guardrail_latency_regression"

    best_by_case = {int(a.case_num): a for a in best_result.anchors}
    retest_used = False
    for _attempt in range(2):
        active_anchors_cand = [
            a for a in cur_res.anchors if a.regime_tag == regime_spec.regime_tag
        ]
        if not active_anchors_cand:
            active_anchors_cand = list(cur_res.anchors)
        active_pts = [
            to_conservation_point(a, best_by_case.get(int(a.case_num)))
            for a in active_anchors_cand
        ]
        active_ok, active_failed = check_family_and_sibling_non_regression(
            active_pts, (), sigma_mult=3.0
        )
        if not active_ok:
            failed_set = set(active_failed)
            regressed_active = [a for a in active_anchors_cand if int(a.case_num) in failed_set]
            noisy_active = any(
                float(getattr(a, "burst_batch_cv", 0.0) or 0.0) > MAX_CLEAN_CALIBRATION_BURST_CV
                for a in regressed_active
            )
            if not retest_used and noisy_active and retest_noisy_fn is not None:
                retested = retest_noisy_fn(regressed_active, cur_res)
                if retested is not None:
                    if not retested.all_passed:
                        return False, cur_res, "recheck_failed"
                    cur_res = retested
                    retest_used = True
                    continue
            return False, cur_res, "active_tail_regression"

        sibling_cand = [
            a
            for a in cur_res.anchors
            if a.regime_tag != regime_spec.regime_tag
            and a.regime_tag not in unchanged_siblings
            and int(a.case_num) in best_by_case
        ]
        sibling_pts = [
            to_conservation_point(a, best_by_case.get(int(a.case_num)))
            for a in sibling_cand
        ]
        sib_ok, sib_failed = check_family_and_sibling_non_regression(
            (), sibling_pts, sigma_mult=3.0
        )
        if not sib_ok:
            sib_set = set(sib_failed)
            dipped_siblings = [a for a in sibling_cand if int(a.case_num) in sib_set]
            if len(dipped_siblings) >= 2:
                noisy_sib = any(
                    float(getattr(a, "burst_batch_cv", 0.0) or 0.0)
                    > MAX_CLEAN_CALIBRATION_BURST_CV
                    for a in dipped_siblings
                )
                if not retest_used and noisy_sib and retest_noisy_fn is not None:
                    retested = retest_noisy_fn(dipped_siblings, cur_res)
                    if retested is not None:
                        if not retested.all_passed:
                            return False, cur_res, "recheck_failed"
                        cur_res = retested
                        retest_used = True
                        continue
                return False, cur_res, "multi_sibling_regression"
            if len(dipped_siblings) == 1:
                dipped = dipped_siblings[0]
                if recheck_single_sibling_fn is not None:
                    anchor_spec_by_case = {int(a.case_num): a for a in active_anchors}
                    dipped_spec = anchor_spec_by_case.get(int(dipped.case_num))
                    if dipped_spec is None:
                        dipped_spec = AnchorCaseSpec(
                            task=f"operator/{cur_res.operator_name}",
                            case_num=int(dipped.case_num),
                            regime_tag=dipped.regime_tag,
                            ref_perf_us=dipped.ref_perf_us,
                            t_hw_us=dipped.t_hw_us,
                        )
                    recheck_res = recheck_single_sibling_fn(dipped_spec)
                    if recheck_res is None or not recheck_res.all_passed:
                        return False, cur_res, "recheck_failed"
                    recheck_anc = next(
                        (a for a in recheck_res.anchors if int(a.case_num) == int(dipped.case_num)),
                        None,
                    )
                    if recheck_anc is None:
                        return False, cur_res, "recheck_failed"
                    best_anc = best_by_case.get(int(dipped.case_num))
                    if not check_guardrail_non_regression(
                        to_conservation_point(recheck_anc, best_anc), sigma_mult=3.0
                    ):
                        return False, cur_res, "recheck_failed"
                    merged = tuple(
                        recheck_anc if int(a.case_num) == int(dipped.case_num) else a
                        for a in cur_res.anchors
                    )
                    cur_res = aggregate_surrogate_result(
                        cur_res.candidate_id,
                        cur_res.operator_name,
                        merged,
                        error_message=cur_res.error_message,
                    )
                else:
                    return False, cur_res, "sibling_regression"
        break

    final_best_by_case = {int(a.case_num): a for a in best_result.anchors}
    final_active = [a for a in cur_res.anchors if a.regime_tag == regime_spec.regime_tag]
    if not final_active:
        final_active = list(cur_res.anchors)
    gain_pts = [
        to_conservation_point(a, final_best_by_case.get(int(a.case_num)))
        for a in final_active
        if int(a.case_num) in final_best_by_case
    ]
    if gain_pts:
        gain_ok, _net_gain, _thresh = check_regime_net_improvement(gain_pts, sigma_mult=2.0)
        if not gain_ok:
            all_pts = [
                to_conservation_point(a, final_best_by_case.get(int(a.case_num)))
                for a in cur_res.anchors
                if int(a.case_num) in final_best_by_case
            ]
            global_ok, _gnet, _gth = check_regime_net_improvement(all_pts, sigma_mult=2.0)
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
    retest_noisy_fn: Callable[
        [Sequence[AnchorEvalMetrics], CandidateSurrogateResult],
        CandidateSurrogateResult | None,
    ]
    | None = None,
    recheck_single_sibling_fn: Callable[
        [AnchorCaseSpec], CandidateSurrogateResult | None
    ]
    | None = None,
) -> RankedL2Survivors:
    """Filter L2 candidates via promotion gate and rank the top 2 by descending speedup."""
    qualified: list[RankedL2Candidate] = []
    rejected: list[tuple[str, str]] = []
    g_map = guardrail_metrics_by_cid or {}
    u_map = unchanged_siblings_by_cid or {}

    for cand in candidates:
        if not cand.candidate_result.all_passed:
            rejected.append((cand.candidate_id, "l2_not_passed"))
            continue
        ok, updated_res, reason = evaluate_promotion_gate(
            candidate_id=cand.candidate_id,
            cur_res=cand.candidate_result,
            best_result=best_result,
            regime_spec=regime_spec,
            guardrail_cases=guardrail_cases,
            cand_guardrail_metrics=g_map.get(cand.candidate_id, {}),
            unchanged_siblings=u_map.get(cand.candidate_id, set()),
            active_anchors=active_anchors,
            retest_noisy_fn=retest_noisy_fn,
            recheck_single_sibling_fn=recheck_single_sibling_fn,
        )
        if not ok:
            rejected.append((cand.candidate_id, reason or "gate_rejected"))
            continue
        qualified.append(
            replace(
                cand,
                candidate_result=updated_res,
                speedup=float(updated_res.avg_speedup),
            )
        )

    qualified.sort(
        key=lambda item: (
            -float(item.speedup),
            -float(item.candidate_result.surrogate_score),
            item.candidate_id,
        )
    )
    top_two = tuple(
        replace(item, rank=idx) for idx, item in enumerate(qualified[:2], start=1)
    )
    return RankedL2Survivors(survivors=top_two, rejected=tuple(rejected))


def _build_pending_promotion(
    cand: RankedL2Candidate,
    l3_res: L3AcceptanceResult,
    *,
    target_path: str,
    round_idx: int,
    regime_tag: str,
) -> PendingPromotion:
    return PendingPromotion(
        candidate_source=cand.candidate_source,
        candidate_sha256=l3_res.candidate_sha256 or cand.candidate_sha256,
        target_path=l3_res.target_path or target_path,
        round_idx=round_idx,
        regime_tag=regime_tag,
        candidate_result=cand.candidate_result,
        candidate_id=cand.candidate_id,
        builder_name=cand.builder_name,
        parent_lane=cand.parent_lane,
        creative_effort=cand.creative_effort,
        execution_id=l3_res.execution_id,
        run_id=l3_res.run_id,
        receipt_path=l3_res.receipt_path,
        error_message=l3_res.error_message,
        device_id=l3_res.device_id,
        execution_terminal=l3_res.execution_terminal,
    )


class PromotionCoordinator:
    """Coordinate CEGAR guardrail updates and L3 candidate commit/rollback."""

    def __init__(
        self,
        *,
        target_module_path: Path,
        write_and_verify_fn: Callable[[Path, str], str],
    ) -> None:
        self._target_module_path = target_module_path
        self._write_and_verify = write_and_verify_fn

    def rollback_to_incumbent(self, incumbent_source: str) -> None:
        """Atomically restore incumbent source and verify written bytes."""
        err = self._write_and_verify(self._target_module_path, incumbent_source)
        if err:
            raise RuntimeError(err)

    def _stage_candidate(self, source: str) -> None:
        err = self._write_and_verify(self._target_module_path, source)
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
            self._stage_candidate(cand.candidate_source)
            l3_res = run_l3_fn(cand)
            resolution = classify_l3_resolution(l3_res)
            attempts.append((cand.candidate_id, resolution, l3_res))
            for case_num in l3_res.failed_case_nums:
                if int(case_num) not in failed_cases:
                    failed_cases.append(int(case_num))

            if resolution is L3Resolution.PROMOTED:
                return PromotionDecision(
                    resolution=L3Resolution.PROMOTED,
                    promoted_candidate=cand,
                    l3_result=l3_res,
                    attempts=tuple(attempts),
                    new_guardrail_cases=tuple(failed_cases),
                    fallback_used=fallback_used,
                )

            if resolution is L3Resolution.INFRA_DEFERRED:
                self.rollback_to_incumbent(incumbent_source)
                pending = _build_pending_promotion(
                    cand,
                    l3_res,
                    target_path=resolved_target,
                    round_idx=round_idx,
                    regime_tag=regime_tag,
                )
                return PromotionDecision(
                    resolution=L3Resolution.INFRA_DEFERRED,
                    l3_result=l3_res,
                    pending_promotion=pending,
                    attempts=tuple(attempts),
                    new_guardrail_cases=tuple(failed_cases),
                    fallback_used=fallback_used,
                )

        self.rollback_to_incumbent(incumbent_source)
        last_l3 = attempts[-1][2]
        return PromotionDecision(
            resolution=L3Resolution.CANDIDATE_REJECTED,
            l3_result=last_l3,
            attempts=tuple(attempts),
            new_guardrail_cases=tuple(failed_cases),
            fallback_used=len(attempts) > 1,
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
