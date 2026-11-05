"""P (predicate) module controls: independent bound-derived coverage.

Every check here derives the required active set from the BOUNDS, never from an
authored active count, and compares it with a DISTINCT actual side. The four
decisive bad controls each fail by their own code, each paired with a positive
control on the same harness so a rule that refuses everything scores zero.

Exit codes: 0 PASS, 1 FAIL, 2 ERROR.
"""
from __future__ import annotations

import tilefoundry_ascend  # noqa: F401  (registers the target / passes)
from tilefoundry.ir.hir.pipeline import PipelineError
from tilefoundry.passes.pass_base import ModulePass
from tilefoundry.passes.pass_manager import _TARGET_PASSES, PassPoint
from tilefoundry_ascend.passes import pipeline as d0
from tilefoundry_ascend.passes.pipeline import (
    PIPELINE_MANIFEST_KEY,
    PipelineManifest,
    PredicateRequirement,
    TailPolicy,
)
from tilefoundry_ascend.tf_target import Ascend950PRTarget

FULL = frozenset(range(64))
TAIL6 = frozenset(range(6))
N70 = 70
BASE = 64


class Missing(AssertionError):
    pass


def _expect_refusal(call, code, what):
    try:
        call()
    except PipelineError as error:
        if getattr(error, "code", None) != code:
            raise AssertionError(
                "%s refused as %r, expected %r" % (what, getattr(error, "code", None), code)
            ) from error
        return
    raise AssertionError("%s was ACCEPTED, expected refusal %r" % (what, code))


def _expect_accept(call, what):
    try:
        return call()
    except PipelineError as error:
        raise AssertionError(
            "%s was REFUSED as %r, but it is the positive control and must pass"
            % (what, getattr(error, "code", None))
        ) from error


# --- normalize: per-lane window, logical base, UNKNOWN/N=0 refused -------------


def test_normalize_n70_last_issue_needs_lanes_0_to_5():
    req = d0.normalize_tail_requirement(N70, BASE, TailPolicy.MASK_ACTIVE)
    assert req.status == "NORMALIZED"
    assert req.required_active == TAIL6, req.required_active
    assert req.granularity_bits == 32
    evidence = dict(req.bounds_evidence)
    assert evidence == {"logical_len": 70, "lane_count": 64, "origin": 64}, evidence


def test_normalize_full_issue_is_all_lanes():
    req = d0.normalize_tail_requirement(BASE, BASE, TailPolicy.MASK_ACTIVE)
    assert req.required_active == FULL, req.required_active


def test_normalize_full_tile_on_full_issue_is_accepted():
    _expect_accept(
        lambda: d0.normalize_tail_requirement(BASE, BASE, TailPolicy.FULL_TILE),
        "FULL_TILE on a real full issue",
    )


def test_normalize_full_tile_on_n70_is_unproven():
    _expect_refusal(
        lambda: d0.normalize_tail_requirement(N70, BASE, TailPolicy.FULL_TILE),
        "UNPROVEN_FULL_TILE",
        "FULL_TILE claimed for N70/lane64 whose last issue needs six lanes",
    )


def test_normalize_logical_length_zero_is_not_a_full_tile():
    _expect_refusal(
        lambda: d0.normalize_tail_requirement(0, BASE, TailPolicy.MASK_ACTIVE),
        "PREDICATE_PROOF_UNAVAILABLE",
        "logical length 0 (0 % base == 0 must not be read as full)",
    )


def test_normalize_unknown_bound_is_refused_not_defaulted():
    _expect_refusal(
        lambda: d0.normalize_tail_requirement(d0.UNKNOWN, BASE, TailPolicy.MASK_ACTIVE),
        "PIPELINE_BOUND_UNKNOWN",
        "UNKNOWN logical length",
    )


# --- coverage: independent recompute, four decisive bad controls --------------


def test_coverage_exact_six_lanes_passes():
    _expect_accept(
        lambda: d0.verify_issue_predicate_coverage(N70, BASE, TAIL6),
        "the exact {0..5} emitted for N70/lane64",
    )


def test_bad_coverage_full_predicate_on_partial_tail():
    _expect_refusal(
        lambda: d0.verify_issue_predicate_coverage(N70, BASE, FULL),
        "MASK_COVERAGE_MISMATCH",
        "a full 64-lane predicate where the bounds need six lanes",
    )


def test_bad_coverage_five_lanes():
    _expect_refusal(
        lambda: d0.verify_issue_predicate_coverage(N70, BASE, frozenset(range(5))),
        "MASK_COVERAGE_MISMATCH",
        "a 5-lane predicate where six are required",
    )


def test_bad_coverage_seven_lanes():
    _expect_refusal(
        lambda: d0.verify_issue_predicate_coverage(N70, BASE, frozenset(range(7))),
        "MASK_COVERAGE_MISMATCH",
        "a 7-lane predicate where six are required",
    )


def test_bad_granularity_32_vs_16():
    _expect_refusal(
        lambda: d0.verify_issue_predicate_coverage(
            N70, BASE, TAIL6, granularity_bits=32, actual_granularity_bits=16),
        "MASK_GRANULARITY_MISMATCH",
        "a 16-bit conversion mask used where 32-bit arithmetic needs 32 bits",
    )


def test_coverage_granularity_match_passes():
    _expect_accept(
        lambda: d0.verify_issue_predicate_coverage(
            N70, BASE, TAIL6, granularity_bits=32, actual_granularity_bits=32),
        "32-bit mask for a 32-bit consumer",
    )


def test_bad_scope_cross_region():
    _expect_refusal(
        lambda: d0.verify_issue_predicate_coverage(
            N70, BASE, TAIL6, scope="root", actual_scope="conversion"),
        "PREDICATE_SCOPE_MISMATCH",
        "a producer in a different scope from the consumer",
    )


def test_coverage_same_scope_passes():
    _expect_accept(
        lambda: d0.verify_issue_predicate_coverage(
            N70, BASE, TAIL6, scope="root", actual_scope="root"),
        "producer and consumer in the same scope",
    )


def test_bad_unknown_bound_cannot_be_proved():
    _expect_refusal(
        lambda: d0.verify_issue_predicate_coverage(d0.UNKNOWN, BASE, TAIL6),
        "PREDICATE_PROOF_UNAVAILABLE",
        "an UNKNOWN bound, which must never be judged safe",
    )


# --- part: 0 only through a unique projection; raw part1 refused --------------


class _UniqueProjection:
    def unique_part(self):
        return 0


class _PartOneRequest:
    def unique_part(self):
        return 1


def test_part_unique_projection_is_zero():
    assert d0.normalize_part_selection(_UniqueProjection()) == 0


def test_bad_part_one_request_is_unsupported():
    _expect_refusal(
        lambda: d0.select_part(_PartOneRequest()), "PART_SELECTION_UNSUPPORTED",
        "a projection requesting part1",
    )
    _expect_refusal(
        lambda: d0.normalize_part_selection(_PartOneRequest()), "PART_SELECTION_UNSUPPORTED",
        "normalize writing part from a part1 projection",
    )


def test_bad_missing_projection_is_unsupported():
    _expect_refusal(
        lambda: d0.select_part(None), "PART_SELECTION_UNSUPPORTED",
        "no projection at all (part must not default to 0)",
    )


def test_bad_tir_raw_part_one_is_unsupported():
    _expect_refusal(
        lambda: d0.verify_tir_part(1), "PART_SELECTION_UNSUPPORTED",
        "a directly constructed TIR part1",
    )
    d0.verify_tir_part(0)
    d0.verify_tir_part(None)


# --- wiring: registered, non-fabricating, actual-vs-required ------------------


class _FakeModule:
    def __init__(self, manifest):
        self.functions = ()
        self.metadata = {PIPELINE_MANIFEST_KEY: manifest}


def _manifest(requirements=()):
    return PipelineManifest(
        schema="ascend.pipeline_manifest/1", loops=(),
        buffer_declarations=(), requirements=tuple(requirements),
    )


def _normalized_requirement():
    return PredicateRequirement(
        issue_source_id="loop[0]:i", scope="root", required_active=TAIL6,
        granularity_bits=32, producer_kind="VF",
        bounds_evidence=(("logical_len", 70), ("lane_count", 64), ("origin", 64)),
        policy=TailPolicy.MASK_ACTIVE, part_selection=0, status="NORMALIZED",
    )


def test_both_p_passes_are_registered_on_the_public_path():
    before = {t.__name__ for t in _TARGET_PASSES.get((Ascend950PRTarget, PassPoint.BEFORE_LOWER), ())}
    after = {t.__name__ for t in _TARGET_PASSES.get((Ascend950PRTarget, PassPoint.AFTER_LOWER), ())}
    assert "NormalizeSchedulePredicates" in before, sorted(before)
    assert "VerifyPredicateCoverage" in after, sorted(after)
    assert issubclass(d0.NormalizeSchedulePredicates, ModulePass)
    assert issubclass(d0.VerifyPredicateCoverage, ModulePass)


def test_normalize_does_not_fabricate_unknown_requirements():
    module = _FakeModule(_manifest())
    d0.NormalizeSchedulePredicates().run(module)
    manifest = module.metadata[PIPELINE_MANIFEST_KEY]
    assert manifest.requirements == (), manifest.requirements


def test_verify_compares_actual_side_not_required_against_itself():
    module = _FakeModule(_manifest((_normalized_requirement(),)))
    _expect_refusal(
        lambda: d0.VerifyPredicateCoverage().run(module),
        "MISSING_TAIL_PREDICATE",
        "a NORMALIZED requirement with no emitted predicate (old code passed by comparing required to itself)",
    )


def test_verify_refuses_unknown_requirement_instead_of_skipping():
    unknown = PredicateRequirement(
        issue_source_id="loop[0]:i", scope="root", required_active=frozenset(),
        granularity_bits=32, producer_kind="VF", bounds_evidence=(),
        policy=TailPolicy.MASK_ACTIVE, part_selection=0, status="UNKNOWN",
    )
    module = _FakeModule(_manifest((unknown,)))
    _expect_refusal(
        lambda: d0.VerifyPredicateCoverage().run(module),
        "PREDICATE_PROOF_UNAVAILABLE",
        "an UNKNOWN requirement (old code silently skipped it)",
    )


TESTS = (
    test_normalize_n70_last_issue_needs_lanes_0_to_5,
    test_normalize_full_issue_is_all_lanes,
    test_normalize_full_tile_on_full_issue_is_accepted,
    test_normalize_full_tile_on_n70_is_unproven,
    test_normalize_logical_length_zero_is_not_a_full_tile,
    test_normalize_unknown_bound_is_refused_not_defaulted,
    test_coverage_exact_six_lanes_passes,
    test_bad_coverage_full_predicate_on_partial_tail,
    test_bad_coverage_five_lanes,
    test_bad_coverage_seven_lanes,
    test_bad_granularity_32_vs_16,
    test_coverage_granularity_match_passes,
    test_bad_scope_cross_region,
    test_coverage_same_scope_passes,
    test_bad_unknown_bound_cannot_be_proved,
    test_part_unique_projection_is_zero,
    test_bad_part_one_request_is_unsupported,
    test_bad_missing_projection_is_unsupported,
    test_bad_tir_raw_part_one_is_unsupported,
    test_both_p_passes_are_registered_on_the_public_path,
    test_normalize_does_not_fabricate_unknown_requirements,
    test_verify_compares_actual_side_not_required_against_itself,
    test_verify_refuses_unknown_requirement_instead_of_skipping,
)


def main() -> int:
    failed = 0
    for case in TESTS:
        try:
            case()
        except Exception as error:  # noqa: BLE001
            failed += 1
            print("FAIL %s: %s: %s" % (case.__name__, type(error).__name__, error))
        else:
            print("PASS %s" % case.__name__)
    print("%d/%d passed" % (len(TESTS) - failed, len(TESTS)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
