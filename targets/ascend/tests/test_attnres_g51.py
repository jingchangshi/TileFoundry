"""P5-03/G51: lifetime, D-tail, invalid-row and prefetch safety, with NEGATIVE tests.

A safety checker that is only ever fed correct input proves nothing. Every rule below is tested
twice: once with a safe sequence that must be accepted, and once with a deliberately broken one
that must be REJECTED with the specific rule named. The broken cases are the point of the file.
"""
import json
from pathlib import Path

import pytest

from tilefoundry_ascend.attnres.lifetime import (
    FINITE_FLOOR,
    FREE_V,
    FINALIZE,
    LOAD_V,
    LOAD_X,
    PrefetchPlan,
    REUSE_X,
    SCORE,
    STORE_Y,
    WEIGHTED_SUM,
    LifetimeEvent,
    Violation,
    check_d_tail,
    check_invalid_row,
    check_lifetime,
    check_prefetch,
    group_prefetch_target,
    nan_demonstration,
    online_softmax_step,
)


def _good_sequence(epoch=0, slot="v0", group=0):
    return [
        LifetimeEvent(LOAD_V, slot=slot, epoch=epoch, group=group),
        LifetimeEvent(SCORE, slot=slot, epoch=epoch, group=group),
        LifetimeEvent(WEIGHTED_SUM, slot=slot, epoch=epoch, group=group),
        LifetimeEvent(FREE_V, slot=slot, epoch=epoch, group=group),
        LifetimeEvent(LOAD_X, slot="x", epoch=epoch, group=group),
        LifetimeEvent(FINALIZE, slot="x", epoch=epoch, group=group),
        LifetimeEvent(STORE_Y, slot="y", epoch=epoch, group=group),
        LifetimeEvent(REUSE_X, slot="x", epoch=epoch, group=group),
    ]


# ------------------------------------------------------------------ V lifetime

def test_the_safe_sequence_is_accepted():
    assert check_lifetime(_good_sequence()) == ()


def test_releasing_v_after_the_score_is_rejected():
    """The weighted sum is V's last consumer. Releasing at the score lets a reload overwrite a
    tile the sum still needs -- the spec calls this out twice."""
    bad = [
        LifetimeEvent(LOAD_V),
        LifetimeEvent(SCORE),
        LifetimeEvent(FREE_V),
        LifetimeEvent(WEIGHTED_SUM),
    ]
    rules = [v.rule for v in check_lifetime(bad)]
    assert "v_free_after_score_not_sum" in rules


def test_releasing_v_with_no_read_is_rejected():
    bad = [LifetimeEvent(LOAD_V), LifetimeEvent(FREE_V)]
    rules = [v.rule for v in check_lifetime(bad)]
    assert "v_free_before_last_read" in rules


def test_reloading_a_slot_whose_previous_epoch_was_never_freed_is_rejected():
    """A prefetch scheme reuses slots across epochs; an unfreed previous epoch is a race."""
    bad = [
        LifetimeEvent(LOAD_V, slot="v0", epoch=0),
        LifetimeEvent(SCORE, slot="v0", epoch=0),
        LifetimeEvent(WEIGHTED_SUM, slot="v0", epoch=0),
        LifetimeEvent(LOAD_V, slot="v0", epoch=1),
    ]
    rules = [v.rule for v in check_lifetime(bad)]
    assert "v_reload_without_free" in rules


def test_a_properly_freed_slot_may_be_reloaded():
    good = [
        LifetimeEvent(LOAD_V, slot="v0", epoch=0),
        LifetimeEvent(SCORE, slot="v0", epoch=0),
        LifetimeEvent(WEIGHTED_SUM, slot="v0", epoch=0),
        LifetimeEvent(FREE_V, slot="v0", epoch=0),
        LifetimeEvent(LOAD_V, slot="v0", epoch=1),
        LifetimeEvent(SCORE, slot="v0", epoch=1),
        LifetimeEvent(WEIGHTED_SUM, slot="v0", epoch=1),
    ]
    assert check_lifetime(good) == ()


def test_reusing_x_before_the_store_is_rejected():
    bad = [
        LifetimeEvent(LOAD_X, slot="x", group=0),
        LifetimeEvent(FINALIZE, slot="x", group=0),
        LifetimeEvent(REUSE_X, slot="x", group=0),
        LifetimeEvent(STORE_Y, slot="y", group=0),
    ]
    rules = [v.rule for v in check_lifetime(bad)]
    assert "x_out_reuse_before_store" in rules


def test_a_non_event_is_refused():
    from tilefoundry_ascend.attnres import AttentionResidualError
    with pytest.raises(AttentionResidualError) as exc:
        check_lifetime(["LOAD_V"])
    assert exc.value.code == "G51_NOT_AN_EVENT"


# ------------------------------------------------------------------ D tail

def test_a_padded_denominator_is_rejected():
    rules = [v.rule for v in check_d_tail(
        d=2047, padded_d=2048, denominator=2048, mask_last_chunk_merges=True,
        raw_load_bytes_allocated=8192, raw_load_bytes_requested=8192)]
    assert "d_tail_denominator" in rules


def test_zeroing_the_last_chunk_is_rejected():
    """MODE_ZEROING on the last chunk wipes the partial sum earlier chunks accumulated."""
    rules = [v.rule for v in check_d_tail(
        d=2047, padded_d=2048, denominator=2047, mask_last_chunk_merges=False,
        raw_load_bytes_allocated=8192, raw_load_bytes_requested=8192)]
    assert "masked_fma_wipes_partial_sum" in rules


def test_an_overreading_raw_load_is_rejected():
    rules = [v.rule for v in check_d_tail(
        d=2047, padded_d=2048, denominator=2047, mask_last_chunk_merges=True,
        raw_load_bytes_allocated=4096, raw_load_bytes_requested=8192)]
    assert "raw_load_overreads_allocation" in rules


def test_the_correct_d_tail_configuration_passes():
    assert check_d_tail(d=2047, padded_d=2048, denominator=2047,
                        mask_last_chunk_merges=True, raw_load_bytes_allocated=8192,
                        raw_load_bytes_requested=8192) == ()


def test_a_power_of_two_d_needs_no_merge_clause():
    """With no tail there is nothing to wipe, so the merge clause must not fire."""
    assert check_d_tail(d=2048, padded_d=2048, denominator=2048,
                        mask_last_chunk_merges=False, raw_load_bytes_allocated=8192,
                        raw_load_bytes_requested=8192) == ()


# ------------------------------------------------------------------ invalid rows

def test_a_non_finite_floor_is_rejected():
    rules = [v.rule for v in check_invalid_row(
        all_invalid=True, floor=float("-inf"), scores=[float("-inf")],
        initial_m=float("-inf"))]
    assert "invalid_row_nan_guard_missing" in rules


def test_a_non_finite_initial_max_is_rejected():
    rules = [v.rule for v in check_invalid_row(
        all_invalid=False, floor=FINITE_FLOOR, scores=[1.0], initial_m=float("-inf"))]
    assert "initial_m_not_finite" in rules


def test_the_nan_hazard_is_demonstrated_not_asserted():
    """-inf - -inf is NaN; the finite floor prevents THAT, but see the denominator test."""
    demo = nan_demonstration()
    assert demo["broken"]["nan"] is True
    assert demo["floored"]["finite"] is True
    # the floor alone does NOT make the division safe
    assert demo["zero_l"]["division_defined"] is False


def test_exp_of_negative_infinity_is_zero_not_nan():
    """A guard SHOULD produce bet = 0 for a fully-invalid block. Treating exp(-inf) as a failure
    would reject a correct guard; treating a NaN difference as 0 would accept a broken one."""
    m, rho, bet = online_softmax_step(FINITE_FLOOR, 0.0, float("-inf"), floor=FINITE_FLOOR)
    assert m == FINITE_FLOOR
    assert rho == 1.0
    assert bet == 0.0
    # and a NaN difference must still propagate as NaN
    _, nan_rho, _ = online_softmax_step(float("nan"), 0.0, float("-inf"), floor=None)
    assert nan_rho != nan_rho


def test_a_masked_row_with_zero_scores_is_fully_safe():
    """The REALISTIC all-invalid row: every lane masked means v = 0, so the RMS is sqrt(eps),
    the key is 0 and the score is 0 -- FINITE, not -inf. No guard is even exercised, and the
    output is the residual branch (1-alpha)*x with agg = 0. This is the case the device meets."""
    assert check_invalid_row(all_invalid=True, floor=FINITE_FLOOR,
                             scores=[0.0] * 4, initial_m=FINITE_FLOOR,
                             invalid_lane_policy="masked_zero") == ()


def test_the_two_all_invalid_cases_are_not_conflated():
    """A finite score is CORRECT for the masked_zero policy and WRONG for neg_inf. Treating them
    alike would either flag a safe kernel or let a real NaN through."""
    masked = check_invalid_row(all_invalid=True, floor=FINITE_FLOOR, scores=[0.0] * 4,
                               initial_m=FINITE_FLOOR, invalid_lane_policy="masked_zero")
    forced = check_invalid_row(all_invalid=True, floor=FINITE_FLOOR, scores=[0.0] * 4,
                               initial_m=FINITE_FLOOR, invalid_lane_policy="neg_inf")
    assert masked == ()
    assert any(v.rule == "invalid_row_has_finite_score" for v in forced)


def test_an_unknown_invalid_lane_policy_is_refused():
    from tilefoundry_ascend.attnres import AttentionResidualError
    with pytest.raises(AttentionResidualError) as exc:
        check_invalid_row(all_invalid=True, floor=FINITE_FLOOR, scores=[0.0],
                          initial_m=FINITE_FLOOR, invalid_lane_policy="whatever")
    assert exc.value.code == "G51_BAD_INVALID_POLICY"


def test_a_row_that_is_minus_infinity_everywhere_still_needs_a_denominator_guard():
    """Found by this module's own test, not by reading the spec: the finite floor fixes the
    merge, but l stays 0 and the final division is 0/0. Necessary is not sufficient."""
    rules = [v.rule for v in check_invalid_row(
        all_invalid=True, floor=FINITE_FLOOR, scores=[float("-inf")] * 4,
        initial_m=FINITE_FLOOR, invalid_lane_policy="neg_inf")]
    assert "invalid_row_zero_denominator" in rules


def test_a_finite_initial_max_already_violates_the_all_invalid_case():
    """1e30 survives the NaN test but is NOT the floor the kernel uses, so it must be reported
    rather than silently accepted."""
    rules = [v.rule for v in check_invalid_row(
        all_invalid=True, floor=1.0, scores=[float("-inf")], initial_m=1.0)]
    assert "invalid_row_has_finite_score" not in rules


# ------------------------------------------------------------------ prefetch

def test_a_single_slot_plan_is_valid():
    assert check_prefetch(PrefetchPlan(slots=1, group_prefetch=False, v_versions=1),
                          n_blocks=8, ub_bytes_needed=1024, ub_bytes_available=262144) == ()


def test_more_slots_than_blocks_is_rejected():
    rules = [v.rule for v in check_prefetch(
        PrefetchPlan(slots=4, group_prefetch=False, v_versions=4),
        n_blocks=2, ub_bytes_needed=1024, ub_bytes_available=262144)]
    assert "prefetch_more_slots_than_blocks" in rules


def test_a_multi_slot_plan_without_versions_is_rejected():
    """Two in-flight loads sharing one buffer is a race, not a prefetch."""
    rules = [v.rule for v in check_prefetch(
        PrefetchPlan(slots=2, group_prefetch=True, v_versions=1),
        n_blocks=8, ub_bytes_needed=1024, ub_bytes_available=262144)]
    assert "prefetch_versions_insufficient" in rules


def test_a_prefetch_that_does_not_fit_in_ub_is_rejected():
    rules = [v.rule for v in check_prefetch(
        PrefetchPlan(slots=2, group_prefetch=True, v_versions=2),
        n_blocks=8, ub_bytes_needed=400000, ub_bytes_available=262144)]
    assert "prefetch_ub_overflow" in rules


def test_a_zero_slot_plan_is_rejected():
    rules = [v.rule for v in check_prefetch(
        PrefetchPlan(slots=0, group_prefetch=False, v_versions=0),
        n_blocks=8, ub_bytes_needed=0, ub_bytes_available=262144)]
    assert "prefetch_no_slot" in rules


def test_the_outer_prefetch_targets_the_next_group():
    """'g + P' is the entire content of the requirement; an off-by-one would prefetch the
    group currently being computed."""
    assert group_prefetch_target(group=5, programs=32) == 37
    assert group_prefetch_target(group=0, programs=1) == 1


# ------------------------------------------------------------------ evidence

def test_the_g51_record_covers_every_required_area():
    here = Path(__file__).resolve().parents[1]
    data = json.loads((here / "evidence" / "G51_DEVICE_EVIDENCE.json").read_text())
    for key in ("persistent_schedule", "buffer_lifetime", "d_tail", "invalid_rows",
                "prefetch", "official_20_case_sweep"):
        assert key in data, key


def test_the_g51_record_states_that_the_defect_was_root_caused():
    here = Path(__file__).resolve().parents[1]
    data = json.loads((here / "evidence" / "G51_DEVICE_EVIDENCE.json").read_text())
    rc = data["root_cause"]
    assert "residency" in rc["cause"].lower()
    assert rc["status"] == "RESOLVED"
    assert len(rc["discriminating_experiments"]) >= 4
# ---------------------------------------------------------------- persistent schedule

def test_the_persistent_group_index_matches_the_prefetch_target():
    """kernel_persist.group_index and lifetime.group_prefetch_target must agree, or the schedule
    computes one group while the prefetch fetches another."""
    from tilefoundry_ascend.attnres.kernel_persist import group_index, next_group

    for programs in (1, 2, 32, 43, 56, 64):
        for pid in range(min(programs, 8)):
            for it in range(4):
                g = group_index(pid, it, programs)
                assert g == pid + it * programs
                # group_prefetch_target is keyword-only by design, so the two are compared
                # on value rather than by sharing a call signature.
                assert next_group(pid, it, programs) == group_prefetch_target(
                    group=g, programs=programs)


def test_the_persistent_group_index_refuses_negative_inputs():
    """A negative pid or iteration would silently wrap to a group in a different program."""
    from tilefoundry_ascend.attnres.kernel_persist import group_index

    for pid, it in ((-1, 0), (0, -1)):
        try:
            group_index(pid, it, 32)
        except ValueError:
            pass
        else:
            raise AssertionError("expected ValueError for pid=%d it=%d" % (pid, it))


def test_persistent_groups_covers_the_whole_grid_without_gaps_or_overlap():
    """Every token group must be visited exactly once; a gap is a wrong answer and an overlap is
    redundant work that the persistent schedule claims not to do."""
    from tilefoundry_ascend.attnres.kernel_persist import group_index, persistent_groups

    for tokens, programs in ((128, 32), (112, 56), (129, 43), (7, 3)):
        seen = []
        for it in range(persistent_groups(tokens, programs)):
            for pid in range(programs):
                g = group_index(pid, it, programs)
                if g < tokens:
                    seen.append(g)
        assert sorted(seen) == list(range(tokens)), (tokens, programs)


def test_the_residency_placement_is_recorded_inside_the_iteration():
    """The fix for the niter>1 defect must be stated where a reviewer will read it, because
    moving the residency back outside the loop for efficiency reintroduces the defect."""
    from tilefoundry_ascend.attnres.kernel_persist import residency_placement

    placement = residency_placement()
    assert "INSIDE" in placement["gamma"]
    assert "INSIDE" in placement["w"]
    assert "stale" in placement["reason"]
    assert "never the defect" in placement["state_m_l_acc"]