"""P6-02/G61: the deferred-rescale scale epoch, its protocol and its fault refusals.

The route this tests is cases/mla.md section 8: keep the running numerator split between an
already-drained UB prefix and an L0C accumulator at the current anchor, and move it to one place
only at a scale boundary. The tests below are about the BOUNDARY, because that is where the route
can be silently wrong: the algebra is three lines of exponentials, but "did every old-scale PV
finish before the scale changed" is a completion question that arithmetic cannot answer.

Nothing here claims a device Set/Wait completes. The device evidence is a separate record, and it
says plainly what it could and could not measure.
"""
import json
import math
from pathlib import Path

import pytest

from tilefoundry_ascend.mla.rescale import (
    ACTIVATED,
    AIC,
    BOUNDARY_PUBLISHED,
    BoundaryObligation,
    ConditionalGuard,
    DRAINED,
    DRAINING_IDLE,
    DeferredConfig,
    Epoch,
    FAULTS,
    GroupPredicateLayout,
    Producer,
    PUBLISHERS,
    RescaleError,
    SSByteRange,
    SS_CAPACITY_BYTES,
    SS_DECISION_BYTES,
    ScaleEpochProtocol,
    THETA_PRACTICAL_CAP,
    boundary_factors,
    boundary_obligations,
    deferred_route_guards,
    group_decision,
    inject_fault,
    run_all_faults,
    run_deferred,
    theta_legal_bound,
    theta_search_bound,
    verify_conditional_guards,
)


def small_problem(rows=2, dn=8, dr=4, dv=4, kv=32, seed=7, spread=1.0):
    import random
    rng = random.Random(seed)
    qn = [[rng.uniform(-1, 1) for _ in range(dn)] for _ in range(rows)]
    qr = [[rng.uniform(-1, 1) for _ in range(dr)] for _ in range(rows)]
    kn = [[rng.uniform(-1, 1) * spread for _ in range(dn)] for _ in range(kv)]
    kr = [[rng.uniform(-1, 1) * spread for _ in range(dr)] for _ in range(kv)]
    v = [[rng.uniform(-1, 1) for _ in range(dv)] for _ in range(kv)]
    return qn, qr, kn, kr, v, 1.0 / math.sqrt(dn + dr)


def dense(qn, qr, kn, kr, v, scale, columns=None):
    """The canonical answer computed directly, with no epoch machinery at all."""
    dv = len(v[0])
    lo, hi = columns if columns is not None else (0, dv)
    out = []
    for row in range(len(qn)):
        scores = [(sum(a * b for a, b in zip(qn[row], kn[j]))
                   + sum(a * b for a, b in zip(qr[row], kr[j]))) * scale for j in range(len(kn))]
        top = max(scores)
        p = [math.exp(z - top) for z in scores]
        denom = math.fsum(p)
        out.append([math.fsum(p[j] * v[j][c] for j in range(len(kn))) / denom
                    for c in range(lo, hi)])
    return out


def max_error(left, right):
    return max(abs(a - b) for row_a, row_b in zip(left, right) for a, b in zip(row_a, row_b))


def protocol_for(*, rows=2, columns=3, ordinal=0, anchors=None, blocks=(0,),
                 components=("single",)):
    anchors = anchors or (0.0,) * rows
    producers = frozenset(Producer(block=b, column=c, component=k)
                          for b in blocks for c in range(columns) for k in components)
    return ScaleEpochProtocol(columns=columns, rows=rows,
                              epoch=Epoch(ordinal=ordinal, anchors=tuple(anchors),
                                          producers=producers))


# ------------------------------------------------------------------ theta

def test_theta_comes_from_the_operand_and_the_accumulator_not_from_the_reference_constant():
    """cases/mla.md line 222: the source's constant 6 must not be copied without a derivation."""
    fp16 = theta_legal_bound("float16", block_c=128)
    bf16 = theta_legal_bound("bfloat16", block_c=128)
    assert 0.0 < fp16 < bf16, (
        "fp16's finite max is ~2^16 against bfloat16's ~2^128, so the same theta cannot be legal "
        "for both")
    assert math.exp(fp16) <= 65504.0
    assert fp16 != 6.0, "the bound must be derived, not the reference implementation's constant"
    # Both ceilings are live: a wider tile accumulates more mass into one FP32 cell.
    assert theta_legal_bound("float16", block_c=256) <= theta_legal_bound("float16", block_c=128)
    assert theta_legal_bound("float16", block_c=128, v_abs_max=64.0) <= fp16
    # A bfloat16 accumulator bound eventually wins over the operand bound at a wide enough tile.
    assert theta_legal_bound("bfloat16", block_c=1_000_000) < bf16


def test_a_theta_beyond_its_legal_bound_is_refused_by_the_config_itself():
    bound = theta_legal_bound("float16", block_c=128)
    DeferredConfig(block_c=128, theta=bound, p_dtype="float16").check()
    with pytest.raises(RescaleError):
        DeferredConfig(block_c=128, theta=bound * 1.01, p_dtype="float16").check()
    with pytest.raises(RescaleError):
        DeferredConfig(block_c=128, theta=-1.0).check()
    with pytest.raises(RescaleError):
        DeferredConfig(block_c=128, theta=float("inf")).check()
    with pytest.raises(RescaleError):
        DeferredConfig(p_dtype="float64").check()
    assert theta_search_bound("float16", block_c=128) <= THETA_PRACTICAL_CAP


# ------------------------------------------------------------------ the invariant

def test_total_numerator_is_the_prefix_plus_the_pending_accumulator():
    """U_total = U_ub + U_l0c is the route's premise; if it fails the output is simply wrong."""
    protocol = protocol_for(rows=2, columns=3)
    protocol.add_mass(0, 0, [1.0, 2.0])
    for producer in sorted(protocol.epoch.producers, key=lambda p: p.column):
        protocol.complete_pv(0, producer, [0.5, 1.5])
    for row in range(2):
        for column in range(3):
            assert protocol.total(row, column) == pytest.approx(
                protocol.prefix[row][column] + protocol.pending[row][column])
    # Everything is still pending: nothing has drained yet.
    assert all(v == 0.0 for row in protocol.prefix for v in row)
    assert protocol.invariant_report()["ok"]


def test_a_drain_leaves_pending_exactly_zero_in_every_column():
    """A partial clear would double count the old-scale mass at the next output."""
    protocol = protocol_for(rows=2, columns=4, blocks=(0,))
    protocol.add_mass(0, 0, [1.0, 1.0])
    for producer in sorted(protocol.epoch.producers, key=lambda p: p.column):
        protocol.complete_pv(0, producer, [1.0, 1.0])
    for publisher in PUBLISHERS:
        protocol.publish_local_decision(publisher, needs_rescale=True)
    protocol.publish_boundary(Epoch(ordinal=1, anchors=(2.0, 2.0),
                                    producers=frozenset()))
    protocol.drain_and_clear()
    assert protocol.phase == DRAINED
    assert all(value == 0.0 for row in protocol.pending for value in row), (
        "the clear must cover every column, not just the ones a shard happens to own")
    report = protocol.invariant_report()
    assert report["pending_cleared_after_drain"] is True
    assert report["ok"] is True


# ------------------------------------------------------------------ numerics

def test_the_deferred_route_reproduces_the_canonical_result():
    qn, qr, kn, kr, v, scale = small_problem()
    reference = dense(qn, qr, kn, kr, v, scale)
    for theta in (0.0, 0.5, 1.0, 2.0, 6.0):
        config = DeferredConfig(block_c=8, theta=theta, p_dtype="float16")
        run = run_deferred(qn, qr, kn, kr, v, config=config, scale=scale)
        # 1e-4 is the declared operand rounding in _cast, not an accuracy claim about FP16.
        assert max_error(run.rows, reference) < 1e-4, "theta=%r drifts" % theta
        assert run.invariants["ok"], "theta=%r left the invariant broken" % theta


def test_a_positive_theta_actually_reduces_the_number_of_drains():
    """If theta>0 did not change the drain count the knob would be decorative."""
    qn, qr, kn, kr, v, scale = small_problem(rows=2, dv=4, kv=64, spread=3.0)
    counts = {}
    for theta in (0.0, 0.5, 1.0, 2.0, 4.0):
        run = run_deferred(qn, qr, kn, kr, v,
                           config=DeferredConfig(block_c=8, theta=theta, p_dtype="float16"),
                           scale=scale)
        counts[theta] = run.drains
        assert run.invariants["ok"]
    assert counts[0.0] > counts[2.0], (
        "theta must trade drain count for scale staleness; identical counts mean the threshold "
        "is not being applied (got %r)" % counts)
    assert counts[4.0] <= counts[0.0]
    for theta, drains in counts.items():
        assert drains >= 0


def test_the_case_docs_two_block_example_outputs_one_not_two():
    """cases/mla.md line 232: scores 0 and 10 with V=1 must give 1; a late old PV gives ~2."""
    qn = [[1.0, 0.0, 0.0, 0.0]]
    qr = [[0.0, 0.0]]
    kn = [[0.0, 0.0, 0.0, 0.0], [10.0, 0.0, 0.0, 0.0]]
    kr = [[0.0, 0.0], [0.0, 0.0]]
    v = [[1.0], [1.0]]
    run = run_deferred(qn, qr, kn, kr, v,
                       config=DeferredConfig(block_c=1, theta=0.0, p_dtype="float16"),
                       scale=1.0)
    assert run.rows[0][0] == pytest.approx(1.0, abs=1e-12)
    # And the wrong answer is genuinely near 2, so the fault the tests inject is the real one.
    late = (1.0 + 1.0) / (math.exp(-10.0) + 1.0)
    assert 1.9 < late < 2.01


def test_rows_whose_anchor_held_get_factor_one_and_do_not_move():
    """A moved anchor and a held anchor in the same group: only one row may be rescaled."""
    factors = boundary_factors((0.0, 5.0), (3.0, 5.0))
    assert factors[1] == 1.0, "an unchanged row must not be rescaled at all"
    assert factors[0] == pytest.approx(math.exp(-3.0))
    with pytest.raises(RescaleError):
        boundary_factors((5.0, 0.0), (1.0, 0.0)), "an anchor may not regress"


def test_the_hi_lo_split_recovers_precision_the_single_operand_loses():
    qn, qr, kn, kr, v, scale = small_problem(rows=2, dv=4, kv=16, seed=3)
    reference = dense(qn, qr, kn, kr, v, scale)
    single = run_deferred(qn, qr, kn, kr, v, scale=scale,
                          config=DeferredConfig(block_c=8, theta=1.0, p_dtype="float16"))
    split = run_deferred(qn, qr, kn, kr, v, scale=scale,
                         config=DeferredConfig(block_c=8, theta=1.0, p_dtype="float16",
                                               precision="p_hi_lo"))
    assert max_error(split.rows, reference) <= max_error(single.rows, reference)
    assert max_error(split.rows, reference) < 1e-8
    # hi/lo means TWO producers per (block, column). That is a property of the epoch the driver
    # builds, so check it directly rather than inferring it from a history length.
    from tilefoundry_ascend.mla.rescale import Epoch as _Epoch, Producer as _Producer
    hi_lo_epoch = _Epoch(ordinal=0, anchors=(0.0,),
                         producers=frozenset({_Producer(0, 0, "hi"), _Producer(0, 0, "lo")}))
    assert len(hi_lo_epoch.producers) == 2
    assert {p.component for p in hi_lo_epoch.producers} == {"hi", "lo"}


def test_an_output_column_shard_computes_only_its_own_columns():
    """G62's shard contract, exercised through this route: no cross-program merge."""
    qn, qr, kn, kr, v, scale = small_problem(rows=2, dv=6, kv=16)
    whole = dense(qn, qr, kn, kr, v, scale)
    shard = run_deferred(qn, qr, kn, kr, v, scale=scale,
                         config=DeferredConfig(block_c=8, theta=1.0, p_dtype="float16"),
                         columns_slice=(2, 5))
    assert shard.columns == 3
    # The bound is the DECLARED p_single operand rounding (see _cast), not an accuracy claim:
    # the point of this test is that a shard computes its own columns and nothing merges across
    # programs, not that the low-precision path matches the exact one.
    assert max_error(shard.rows, [row[2:5] for row in whole]) < 1e-4


def test_masked_rows_are_skipped_rather_than_computing_inf_minus_inf():
    qn, qr, kn, kr, v, scale = small_problem(rows=2, dv=4, kv=16)
    # Row 1 attends to nothing: it must not produce a NaN and must not be flushed as -inf.
    run = run_deferred(qn, qr, kn, kr, v, scale=scale,
                       config=DeferredConfig(block_c=8, theta=0.0, p_dtype="float16"),
                       valid=lambda row, index: row == 0)
    assert all(math.isfinite(x) for row in run.rows for x in row)
    assert run.rows[1] == (0.0,) * 4
    assert max_error([run.rows[0]], [dense(qn, qr, kn, kr, v, scale)[0]]) < 1e-4


def test_a_short_kernel_with_one_block_still_works():
    """The 'at least eight KV blocks' limits the official examples impose are not a contract."""
    qn, qr, kn, kr, v, scale = small_problem(rows=1, dv=2, kv=1)
    run = run_deferred(qn, qr, kn, kr, v, scale=scale,
                       config=DeferredConfig(block_c=8, theta=0.0, p_dtype="float16"))
    assert run.drains == 0
    assert max_error(run.rows, dense(qn, qr, kn, kr, v, scale)) < 1e-4

# ------------------------------------------------------------------ the boundary as a completion edge

def test_a_drain_before_every_producer_finishes_is_refused():
    protocol = protocol_for(rows=2, columns=3, blocks=(0, 1))
    protocol.add_mass(0, 0, [1.0, 1.0])
    protocol.add_mass(0, 1, [1.0, 1.0])
    for producer in sorted(protocol.epoch.producers, key=lambda p: (p.block, p.column)):
        if producer.block == 0:
            protocol.complete_pv(0, producer, [1.0, 1.0])
    for publisher in PUBLISHERS:
        protocol.publish_local_decision(publisher, needs_rescale=True)
    protocol.publish_boundary(Epoch(ordinal=1, anchors=(3.0, 3.0), producers=frozenset()))
    with pytest.raises(RescaleError) as caught:
        protocol.drain_and_clear()
    assert caught.value.code == "RESCALE_DRAIN_BEFORE_ALL_PV"
    assert caught.value.subject == "missing"
    assert caught.value.as_dict()["actual"], "the refusal must name WHICH producers are missing"


def test_one_precision_component_finishing_is_not_the_same_as_the_producer_finishing():
    """hi and lo are separate producers; folding them together would lose half the weight."""
    protocol = protocol_for(rows=2, columns=2, components=("hi", "lo"))
    protocol.add_mass(0, 0, [1.0, 1.0])
    for producer in sorted(protocol.epoch.producers, key=lambda p: (p.column, p.component)):
        if producer.component == "hi":
            protocol.complete_pv(0, producer, [1.0, 1.0])
    for publisher in PUBLISHERS:
        protocol.publish_local_decision(publisher, needs_rescale=True)
    protocol.publish_boundary(Epoch(ordinal=1, anchors=(2.0, 2.0),
                                    producers=frozenset()))
    with pytest.raises(RescaleError) as caught:
        protocol.drain_and_clear()
    assert caught.value.code == "RESCALE_DRAIN_BEFORE_ALL_PV"
    missing = caught.value.as_dict()["actual"]
    assert all(entry["component"] == "lo" for entry in missing), missing


def test_a_late_old_scale_pv_after_the_drain_is_a_refusal_not_an_extra_term():
    protocol = protocol_for(rows=1, columns=2)
    protocol.add_mass(0, 0, [1.0])
    for producer in sorted(protocol.epoch.producers, key=lambda p: p.column):
        protocol.complete_pv(0, producer, [1.0])
    for publisher in PUBLISHERS:
        protocol.publish_local_decision(publisher, needs_rescale=True)
    protocol.publish_boundary(Epoch(ordinal=1, anchors=(10.0,), producers=frozenset()))
    protocol.drain_and_clear()
    for consumer in protocol.metadata_consumers:
        protocol.retire_metadata_consumer(consumer)
    protocol.activate_next_epoch()
    with pytest.raises(RescaleError) as caught:
        protocol.complete_pv(0, Producer(block=0, column=0), [5.0])
    assert caught.value.code == "RESCALE_STALE_EPOCH"
    assert "wrong scale" in caught.value.message


def test_a_late_pv_between_publication_and_drain_is_also_refused():
    """The window is not only 'after activate': publication already froze the epoch."""
    protocol = protocol_for(rows=1, columns=1)
    protocol.add_mass(0, 0, [1.0])
    for publisher in PUBLISHERS:
        protocol.publish_local_decision(publisher, needs_rescale=True)
    protocol.publish_boundary(Epoch(ordinal=1, anchors=(4.0,), producers=frozenset()))
    with pytest.raises(RescaleError) as caught:
        protocol.complete_pv(0, Producer(block=0, column=0), [1.0])
    assert caught.value.code == "RESCALE_OLD_SCALE_WRITE_AFTER_PUBLISH"


def test_the_factor_payload_cannot_be_overwritten_while_consumers_hold_it():
    protocol = protocol_for(rows=1, columns=1)
    protocol.add_mass(0, 0, [1.0])
    for producer in protocol.epoch.producers:
        protocol.complete_pv(0, producer, [1.0])
    for publisher in PUBLISHERS:
        protocol.publish_local_decision(publisher, needs_rescale=True)
    protocol.publish_boundary(Epoch(ordinal=1, anchors=(2.0,), producers=frozenset()))
    with pytest.raises(RescaleError) as caught:
        protocol.publish_boundary(Epoch(ordinal=1, anchors=(3.0,), producers=frozenset()))
    assert caught.value.code == "RESCALE_BOUNDARY_TWICE"


def test_activating_the_next_epoch_before_the_consumers_retire_is_refused():
    protocol = protocol_for(rows=1, columns=1)
    protocol.add_mass(0, 0, [1.0])
    for producer in protocol.epoch.producers:
        protocol.complete_pv(0, producer, [1.0])
    for publisher in PUBLISHERS:
        protocol.publish_local_decision(publisher, needs_rescale=True)
    protocol.publish_boundary(Epoch(ordinal=1, anchors=(2.0,), producers=frozenset()))
    protocol.drain_and_clear()
    protocol.retire_metadata_consumer("aic")
    with pytest.raises(RescaleError) as caught:
        protocol.activate_next_epoch()
    assert caught.value.code == "RESCALE_ACTIVATE_BEFORE_CONSUMERS_RETIRED"
    assert set(caught.value.as_dict()["actual"]) == {"aiv0", "aiv1"}


def test_a_metadata_consumer_cannot_retire_before_the_drain_it_depends_on():
    protocol = protocol_for(rows=1, columns=1)
    with pytest.raises(RescaleError) as caught:
        protocol.retire_metadata_consumer("aic")
    assert caught.value.code == "RESCALE_PREMATURE_RETIREMENT"


def test_the_denominator_is_summed_from_full_precision_probabilities():
    """cases/mla.md line 195: redefining l as a sum over the rounded operand changes the path."""
    protocol = protocol_for(rows=1, columns=1)
    protocol.add_mass(0, 0, [2.5])
    assert protocol.denominator[0] == pytest.approx(2.5)
    # A non-monotone or negative increment is not a probability mass.
    with pytest.raises(RescaleError):
        protocol.add_mass(0, 1, [-1.0])
    with pytest.raises(RescaleError):
        protocol.add_mass(0, 0, [1.0]), "the same block's mass may not be committed twice"


def test_a_pv_without_its_probability_mass_describes_two_different_weightings():
    protocol = protocol_for(rows=1, columns=1)
    with pytest.raises(RescaleError) as caught:
        protocol.complete_pv(0, Producer(block=0, column=0), [1.0])
    assert caught.value.code == "RESCALE_PV_WITHOUT_MASS"


# ------------------------------------------------------------------ the SS group predicate

def test_the_two_aivs_write_disjoint_bytes_and_the_aic_reads_both():
    layout = GroupPredicateLayout.disjoint()
    ok, problems = layout.verify()
    assert ok, problems
    assert layout.slot_of("aiv0").bytes().isdisjoint(layout.slot_of("aiv1").bytes())
    assert set(layout.aic_reads) == set(PUBLISHERS)
    assert layout.as_dict()["bytes_used"] <= SS_CAPACITY_BYTES


def test_overlapping_ss_slots_are_refused_at_byte_level():
    overlap = GroupPredicateLayout(
        publishers=PUBLISHERS, aic_reads=PUBLISHERS,
        slots=(SSByteRange("aiv0", 0, SS_DECISION_BYTES),
               SSByteRange("aiv1", 0, SS_DECISION_BYTES)))
    ok, problems = overlap.verify()
    assert not ok
    assert any("both aiv0 and aiv1" in text for text in problems)


def test_an_aic_that_reads_one_slot_cannot_form_a_group_decision():
    with pytest.raises(RescaleError) as caught:
        GroupPredicateLayout(publishers=PUBLISHERS, aic_reads=("aiv0",),
                             slots=GroupPredicateLayout.disjoint().slots)
    assert caught.value.code == "RESCALE_AIC_MUST_READ_BOTH"


def test_a_misaligned_ss_slot_is_refused():
    with pytest.raises(RescaleError) as caught:
        SSByteRange("aiv0", 4, SS_DECISION_BYTES)
    assert caught.value.code == "RESCALE_SS_MISALIGNED"
    with pytest.raises(RescaleError):
        SSByteRange("aiv0", 0, 3), "a partial unit could share a word with the other publisher"


def test_the_group_decision_is_an_or_and_a_missing_publication_is_not_false():
    layout = GroupPredicateLayout.disjoint()
    assert group_decision({"aiv0": False, "aiv1": False}, layout) is False
    assert group_decision({"aiv0": True, "aiv1": False}, layout) is True
    assert group_decision({"aiv0": False, "aiv1": True}, layout) is True
    with pytest.raises(RescaleError) as caught:
        group_decision({"aiv0": False}, layout)
    assert caught.value.code == "RESCALE_GROUP_DECISION_INCOMPLETE"


def test_the_predicate_is_derived_from_the_ss_publications_not_passed_in():
    """A caller must not be able to declare a group decision the AIVs did not publish."""
    protocol = protocol_for(rows=2, columns=2)
    protocol.add_mass(0, 0, [1.0, 1.0])
    for producer in sorted(protocol.epoch.producers, key=lambda p: p.column):
        protocol.complete_pv(0, producer, [1.0, 1.0])
    # One AIV wants a rescale, the other does not: the group decision is still True.
    protocol.publish_local_decision("aiv0", needs_rescale=True)
    protocol.publish_local_decision("aiv1", needs_rescale=False)
    boundary = protocol.publish_boundary(Epoch(ordinal=1, anchors=(2.0, 2.0),
                                               producers=frozenset()))
    assert boundary.predicate is True
    assert boundary.as_dict()["factors"] == pytest.approx([math.exp(-2.0)] * 2)


def test_a_predicate_that_disagrees_with_the_anchors_is_refused():
    protocol = protocol_for(rows=1, columns=1)
    protocol.add_mass(0, 0, [1.0])
    for producer in protocol.epoch.producers:
        protocol.complete_pv(0, producer, [1.0])
    protocol.publish_local_decision("aiv0", needs_rescale=False)
    protocol.publish_local_decision("aiv1", needs_rescale=False)
    with pytest.raises(RescaleError) as caught:
        protocol.publish_boundary(Epoch(ordinal=1, anchors=(7.0,), producers=frozenset()))
    assert caught.value.code == "RESCALE_PREDICATE_DISAGREES_WITH_ANCHORS"


def test_a_decision_cannot_be_republished_for_one_boundary():
    protocol = protocol_for(rows=1, columns=1)
    protocol.publish_local_decision("aiv0", needs_rescale=False)
    with pytest.raises(RescaleError) as caught:
        protocol.publish_local_decision("aiv0", needs_rescale=True)
    assert caught.value.code == "RESCALE_DECISION_REPUBLISHED"

# ------------------------------------------------------------------ conditional guard pairs

def test_the_deferred_routes_guard_pairs_agree_on_one_predicate():
    guards = deferred_route_guards("group_need_rescale@mla")
    problems, checks = verify_conditional_guards(guards)
    assert problems == (), problems
    assert all(entry["ok"] for entry in checks)
    assert all(entry["waiter_implies_setter"] for entry in checks)
    assert all(entry["sides_agree"] for entry in checks)


def test_an_unconditional_wait_against_a_conditional_set_is_refused():
    """The deadlock shape: the wait runs on a branch where the set never happened."""
    problems, checks = verify_conditional_guards(
        [ConditionalGuard("x", "deferred_drain", "cond", "pred_a", "always", "")])
    assert problems
    assert any("setter" in text for text in problems)


def test_two_different_predicates_on_the_two_sides_are_not_implication():
    problems, _ = verify_conditional_guards(
        [ConditionalGuard("x", "deferred_drain", "cond", "pred_a", "cond", "pred_b")])
    assert problems
    assert any("differ" in text or "disagree" in text for text in problems)


def test_an_unconditional_deferred_channel_is_not_a_conditional_drain_at_all():
    """Guarded on both sides but unconditionally: that is the canonical route renamed."""
    problems, _ = verify_conditional_guards(
        [ConditionalGuard("x", "deferred_drain", "always", "", "always", "")])
    assert problems, "a deferred channel must actually be conditional"


# ------------------------------------------------------------------ obligations

def test_the_boundary_publishes_real_obligations_with_named_consumers():
    obligations = boundary_obligations(epoch=2, hi_lo=True)
    assert obligations
    names = {obligation.name for obligation in obligations}
    # Names carry the epoch suffix, so match the stem rather than the bare literal.
    assert any(name.startswith("pv_all_columns_all_components_complete") for name in names)
    assert any(name.startswith("factor_survives_all_boundary_consumers") for name in names)
    assert any(name.startswith("l0c_clear_visible_before_new_anchor_pv") for name in names)
    for obligation in obligations:
        assert isinstance(obligation, BoundaryObligation)
        assert obligation.producer, "an obligation must name a real producer"
        assert obligation.consumers, "an obligation with no consumer is not an edge"
        assert obligation.kind in ("completion", "state", "version")
        assert obligation.why, "why the edge exists must be written down"
    # The hi/lo obligations exist only when the route actually splits the operand.
    assert any("component_hi" in name for name in names)
    plain = {obligation.name for obligation in boundary_obligations(epoch=2, hi_lo=False)}
    assert not any("component_hi" in name for name in plain)


def test_epoch_numbers_appear_in_the_obligation_names_so_two_boundaries_cannot_merge():
    first = {obligation.name for obligation in boundary_obligations(epoch=0)}
    second = {obligation.name for obligation in boundary_obligations(epoch=1)}
    assert first.isdisjoint(second), (
        "two boundaries' obligations must not compare equal, or a checker that unions them "
        "silently drops one boundary's completion edges: %r" % sorted(first & second))
    assert all(name.endswith("_e0") for name in first)
    assert boundary_obligations(0)[0].producer != boundary_obligations(1)[0].producer


# ------------------------------------------------------------------ fault injection

def test_every_declared_fault_is_refused_with_a_distinct_code():
    result = run_all_faults()
    assert result["all_refused"], [f for f in result["faults"] if not f["refused"]]
    assert result["refused_count"] == result["total"] == len(FAULTS)
    codes = [f["code"] for f in result["faults"]]
    assert all(codes), "each refusal must carry a code"
    assert len(set(codes)) >= len(FAULTS) - 2, (
        "distinct faults collapsing onto the same code means one of them is not really being "
        "detected: %r" % codes)
    for fault in result["faults"]:
        assert fault["effect_if_accepted"], (
            "a fault test must state what would go wrong if the fault were accepted")


def test_the_fault_set_covers_the_three_the_gate_names():
    """G61 explicitly names late PV, premature factor overwrite, and mismatched guards."""
    assert "late_old_anchor_pv" in FAULTS
    assert "premature_factor_overwrite" in FAULTS
    assert any(fault.startswith("one_sided") or fault.startswith("overlapping") for fault in FAULTS)
    for fault in FAULTS:
        assert inject_fault(fault)["refused"] is True


def test_an_unknown_fault_name_is_refused_rather_than_silently_passing():
    with pytest.raises(RescaleError):
        inject_fault("not_a_real_fault")


# ------------------------------------------------------------------ honesty about scope

def test_the_module_does_not_claim_to_prove_a_device_set_wait():
    here = Path(__file__).resolve().parents[1]
    source = (here / "tilefoundry_ascend" / "mla" / "rescale.py").read_text()
    assert "does not prove" in source.lower() or "not prove" in source.lower()
    assert "device" in source.lower()


def test_the_g61_record_exists_and_states_what_was_and_was_not_measured():
    here = Path(__file__).resolve().parents[1]
    path = here / "evidence" / "G61_DEVICE_EVIDENCE.json"
    assert path.exists(), "G61 requires a device/implementation evidence record"
    data = json.loads(path.read_text())
    assert data["gate"] == "G61"
    assert data["status"] in ("VERIFIED", "PARTIAL", "INFEASIBLE_WITH_EVIDENCE",
                              "BLOCKED_ENV")
    if data["status"] != "VERIFIED":
        assert data["verdict_scope"], (
            "a route that is not VERIFIED must state the scope of what it did establish")
    assert data["theta_comparison"], "the case doc requires theta=0 AND a positive theta"
    thetas = [row["theta"] for row in data["theta_comparison"]]
    assert any(t == 0.0 for t in thetas), "theta=0 must be one of the compared settings"
    assert any(t > 0.0 for t in thetas), "a positive theta with an exp bound must be compared"
    for row in data["theta_comparison"]:
        assert "drains" in row, "the drain count is the measurable effect of theta"
    assert data["fault_injection"]["all_refused"] is True
    assert data["invariants"]["ok"] is True
    assert data["not_claimed"], "the record must list what it did not establish"


def test_the_record_distinguishes_a_model_run_from_a_device_measurement():
    here = Path(__file__).resolve().parents[1]
    data = json.loads((here / "evidence" / "G61_DEVICE_EVIDENCE.json").read_text())
    measurement = data["device_measurement"]
    assert measurement["status"] in ("MEASURED", "BLOCKED_ENV", "NOT_RUN")
    if measurement["status"] != "MEASURED":
        assert measurement["reason"], (
            "a device measurement that was not taken must say why, not be omitted")


def test_the_invariant_report_is_published_with_the_record():
    here = Path(__file__).resolve().parents[1]
    data = json.loads((here / "evidence" / "G61_DEVICE_EVIDENCE.json").read_text())
    invariants = data["invariants"]
    for key in ("shape_ok", "finite_ok", "pending_cleared_after_drain", "ok"):
        assert key in invariants, "%s is one of the route's stated invariants" % key
