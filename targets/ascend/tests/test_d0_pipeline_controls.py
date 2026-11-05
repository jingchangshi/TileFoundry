"""D0 TDD baseline: the four decisive bad controls and the predicate refusal codes.

RED BY DESIGN, RIGHT NOW. D0-Target (task-63) does not exist yet, so every case
below fails with a NAMED failure that says which contract element is missing and
which refusal code is expected next. That is the point: a baseline that fails with
"module has no attribute ..." or an ImportError traceback measures nothing, and a
baseline that cannot go green later is not a test at all.

WHY FUNCTION LEVEL, RATHER THAN THROUGH A WHOLE FINALIZE RUN.
The design (architect_d0_pipeline_surface_design.md section 6) makes each bad
control a SINGLE injected defect that exactly one checker must refuse by name. That
predicate is what has to be falsifiable, and it can be exercised directly and in
milliseconds. Driving the full parse -> analyze -> finalize path would make every
case depend on unrelated machinery and would measure a tree walk as much as a rule.
The PASS LEVEL wiring is covered separately, by the two wiring cases at the bottom,
so this file cannot pass by testing predicates that nothing ever calls.

EVERY bad case is paired with a CONTROL GROUP on the same harness: the normal input
must be ACCEPTED. Without that pairing, a checker that refuses EVERYTHING would
score a perfect baseline, which is exactly the failure mode the pairing forbids.

Exit codes (repo convention): 0 PASS, 1 FAIL, 2 ERROR.

CONTRACT this file requires of D0-Target. All names exported from ONE of
PASS_MODULE_CANDIDATES, ideally tilefoundry_ascend.passes.pipeline:

    TAIL_FULL_TILE, TAIL_MASK_ACTIVE, UNKNOWN      typed carriers, not free strings
    validate_pipeline_declarations(spec, buffer_versions)
    normalize_tail_requirement(logical_len, base, policy)
    select_part(projection)
    verify_buffer_versions(declared, actual)
    verify_predicate_coverage(required, actual, ...)
    verify_pipeline_preservation(expected, actual)
    ValidatePipelineDeclarations, NormalizeSchedulePredicates,
    VerifyPredicateCoverage, VerifyPipelinePreservation   ModulePass subclasses

If an implementer lands a symbol under a different module, ADD that module to
PASS_MODULE_CANDIDATES. Do not weaken an assertion or delete a control to make this
file green: the code names are the contract, and they come from the design.
"""
from __future__ import annotations

import dataclasses
import importlib
import os
import sys
import traceback

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "targets", "ascend")]

#: Where D0-Target may legitimately put the pipeline rules. passes/pipeline.py is
#: first because the sibling pass module is passes/completion.py.
PASS_MODULE_CANDIDATES = (
    "tilefoundry_ascend.passes.pipeline",
    "tilefoundry_ascend.passes.d0_pipeline",
    "tilefoundry_ascend.passes.schedule",
)

REQUIRED_SYMBOLS = (
    "TAIL_FULL_TILE",
    "TAIL_MASK_ACTIVE",
    "UNKNOWN",
    "validate_pipeline_declarations",
    "normalize_tail_requirement",
    "select_part",
    "verify_buffer_versions",
    "verify_predicate_coverage",
    "verify_pipeline_preservation",
)

REQUIRED_PASSES = (
    "ValidatePipelineDeclarations",
    "NormalizeSchedulePredicates",
    "VerifyPredicateCoverage",
    "VerifyPipelinePreservation",
)

#: Every code this baseline expects an implementation to be able to produce.
EXPECTED_CODES = (
    "PIPELINE_DECLARATION_LOST",
    "BUFFER_VERSION_DECLARATION_CHANGED",
    "BUFFER_VERSION_EXCEEDS_PIPELINE_BUDGET",
    "UNPROVEN_FULL_TILE",
    "MASK_COVERAGE_MISMATCH",
    "PART_SELECTION_UNSUPPORTED",
    "PREDICATE_STALE_OR_OVERWRITTEN",
    "DMA_TAIL_ACCESS_UNSUPPORTED",
    "PIPELINE_BOUND_UNKNOWN",
    "DEFAULTED_UNKNOWN_VALUE",
)


class ContractMissing(AssertionError):
    """A D0 contract element does not exist YET.

    Deliberately an AssertionError rather than an import failure: this has to read
    as a NAMED test failure naming the missing piece and the refusal code it owes,
    not as a broken test file. When D0-Target lands the same case turns green with
    no edit here.
    """


def _d0():
    """Resolve the D0 rule surface, or fail by name listing everything absent."""
    tried = []
    for name in PASS_MODULE_CANDIDATES:
        try:
            module = importlib.import_module(name)
        except ImportError as error:
            tried.append("%s (import failed: %s)" % (name, error))
            continue
        missing = [s for s in REQUIRED_SYMBOLS if not hasattr(module, s)]
        if not missing:
            return module
        tried.append("%s (missing %s)" % (name, ", ".join(missing)))
    raise ContractMissing(
        "D0-Target rules do not exist yet, so no refusal code can be produced. "
        "Looked for %s in: %s. Expected codes this baseline pins: %s."
        % (", ".join(REQUIRED_SYMBOLS), "; ".join(tried) or "no candidate module",
           ", ".join(EXPECTED_CODES))
    )


def _code_of(error):
    """The stable refusal code of a raised refusal, or None.

    The design pins the CODE, not the transport, so a code attribute is preferred
    and a message is accepted only if it names one of the known codes. A refusal
    with no code is NOT a pass: "something threw" is not a named refusal.
    """
    code = getattr(error, "code", None)
    if isinstance(code, str) and code:
        return code
    text = str(error)
    for candidate in EXPECTED_CODES:
        if candidate in text:
            return candidate
    return None


def _expect_refusal(call, expected, what):
    """Assert that call refuses with EXACTLY expected, and say what happened."""
    try:
        result = call()
    except ContractMissing:
        raise
    except Exception as error:  # a refusal may be any typed exception
        actual = _code_of(error)
        if actual is None:
            raise AssertionError(
                "%s was refused WITHOUT a named code: %s: %s. The design requires a "
                "stable code (%s); an untyped exception cannot be asserted on and "
                "cannot be told apart from a crash."
                % (what, type(error).__name__, str(error)[:300], expected)
            ) from error
        assert actual == expected, (
            "%s was refused as %r, but this control isolates %r. A different code "
            "means the wrong rule fired, so this case does not prove the rule it "
            "claims to." % (what, actual, expected)
        )
        return
    raise AssertionError(
        "%s was ACCEPTED (%r), but it must be refused with %r. That acceptance is "
        "the defect this case exists to catch, not a harness problem."
        % (what, result, expected)
    )


def _expect_accept(call, what):
    """Assert that call is ACCEPTED. The control group every bad case needs.

    Paired with _expect_refusal on the SAME harness: it proves the harness is not
    simply refusing everything, which would otherwise score a perfect baseline.
    """
    try:
        return call()
    except ContractMissing:
        raise
    except Exception as error:
        actual = _code_of(error)
        raise AssertionError(
            "%s was REFUSED%s, but this is the NORMAL-PATH CONTROL and it must be "
            "accepted. Either the rule is too strict, or the control group is not a "
            "control at all and every refusal next to it proves nothing."
            % (what, (" as %s" % actual) if actual else " (%s)" % type(error).__name__)
        ) from error


# ---------------------------------------------------------------------------
# 1) A lost pipeline declaration must be refused on BOTH paths the design names:
#    a clone that drops the field, and a lowering that builds the TIR loop without
#    it. The authored source is untouched in either case, which is what makes the
#    loss invisible to anyone reading only the source.
# ---------------------------------------------------------------------------

def _pipeline_spec_type():
    """The typed spec, or a NAMED failure.

    D0-CORE owns this symbol. Importing it lazily through here keeps a missing
    D0-CORE a named assertion instead of an ImportError, which would be counted as
    an ERROR (exit 2) and read as a broken test file rather than an absent contract.
    """
    try:
        from tilefoundry.ir.hir.pipeline import PipelineSpec
    except ImportError as error:
        raise ContractMissing(
            "D0-CORE has not landed tilefoundry.ir.hir.pipeline.PipelineSpec (%s), so "
            "no pipeline declaration can be built for these controls" % (error,)
        ) from error
    return PipelineSpec


def _spec(num_stages=2):
    return _pipeline_spec_type()(num_stages=num_stages)


def _descriptor(pipeline):
    """The typed (loop, pipeline) pair the preservation check compares.

    Real lowering produces this pair for the authored loop and again for the TIR
    loop; the rule is a comparison of the two, so it can be exercised from the pair
    alone. Each side is built independently, so a case cannot mutate expected and
    actual from one object and satisfy itself.
    """
    return {"loop": "root.for#0", "pipeline": pipeline}


def test_control_a_clone_keeps_the_pipeline_declaration():
    """CONTROL: a reconstruct (what clone/specialize do) keeps the declaration."""
    _d0()
    spec_type = _pipeline_spec_type()
    spec = _spec()
    # PipelineSpec is a frozen dataclass, so this IS what a clone/specialize path
    # does to it: reconstruct the same fields. The descriptor is then built from the
    # reconstructed spec, independently of the original.
    clone = _descriptor(dataclasses.replace(spec))
    assert clone["pipeline"] is not None and clone["pipeline"].num_stages == 2, (
        "reconstructing the descriptor dropped the pipeline spec, so the clone path "
        "loses the declaration before any checker runs"
    )
    assert isinstance(clone["pipeline"], spec_type), (
        "the reconstructed descriptor carries %r, not a PipelineSpec"
        % (type(clone["pipeline"]).__name__,)
    )


def test_bad_a1_clone_drops_the_pipeline_declaration():
    """BAD: clone dropped it -> PIPELINE_DECLARATION_LOST."""
    d0 = _d0()
    expected = _descriptor(_spec())
    lost_by_clone = _descriptor(None)
    _expect_refusal(
        lambda: d0.verify_pipeline_preservation(expected, lost_by_clone),
        "PIPELINE_DECLARATION_LOST",
        "a clone that dropped LoopRegion.pipeline while the authored source kept it",
    )


def test_bad_a2_convert_drops_the_pipeline_on_the_tir_loop():
    """BAD: the TIR For was built without it -> same code, other path."""
    d0 = _d0()
    expected = _descriptor(_spec())
    lost_by_convert = _descriptor(None)
    _expect_refusal(
        lambda: d0.verify_pipeline_preservation(expected, lost_by_convert),
        "PIPELINE_DECLARATION_LOST",
        "a ConvertHIRToTIR result whose For.pipeline is None while the authored loop "
        "declared one",
    )


# ---------------------------------------------------------------------------
# 2) A stage budget is a LOOP budget, never a per-buffer version count. Two
#    distinct faults: the declaration changed between authoring and lowering, and
#    an explicit version count that exceeds the budget.
# ---------------------------------------------------------------------------

def test_control_b_budget_two_with_one_version_each_is_accepted():
    """CONTROL: budget2 with x/y fixed at 1 must be accepted."""
    d0 = _d0()
    _expect_accept(
        lambda: d0.validate_pipeline_declarations(_spec(2), {"x": 1, "y": 1}),
        "budget 2 with x=1 and y=1 (the budget never rewrites buffer versions)",
    )


def test_control_b_budget_four_with_four_versions_is_accepted():
    """CONTROL: x=4/y=2 under budget4 is legal (versions <= budget)."""
    d0 = _d0()
    _expect_accept(
        lambda: d0.validate_pipeline_declarations(_spec(4), {"x": 4, "y": 2}),
        "budget 4 with x=4 and y=2",
    )


def test_bad_b1_declaration_says_one_but_lowering_claims_two():
    """BAD: decl x=1 but lowering claims 2 -> CHANGED, not EXCEEDS."""
    d0 = _d0()
    _expect_refusal(
        lambda: d0.verify_buffer_versions({"x": 1, "y": 1}, {"x": 2, "y": 1}),
        "BUFFER_VERSION_DECLARATION_CHANGED",
        "a lowering whose buffer version (x=2) differs from the authored "
        "declaration (x=1)",
    )


def test_bad_b2_four_versions_under_a_budget_of_three():
    """BAD: x=4 under budget3 must EXCEED, and must not be silently capped."""
    d0 = _d0()
    _expect_refusal(
        lambda: d0.validate_pipeline_declarations(_spec(3), {"x": 4, "y": 1}),
        "BUFFER_VERSION_EXCEEDS_PIPELINE_BUDGET",
        "x=4 declared under a stage budget of 3",
    )


# ---------------------------------------------------------------------------
# 3) N70 over a base-64 issue: the required active set of the tail issue is six
#    lanes, so FULL_TILE is unprovable. Two faults: the strategy claims FULL_TILE,
#    and the strategy is unchanged but the lowering emits a full predicate.
# ---------------------------------------------------------------------------

N70_LOGICAL = 70
BASE = 64
TAIL_SET = {0, 1, 2, 3, 4, 5}


def test_control_c_a_true_tail_requirement_is_accepted():
    """CONTROL: N70/base64 with the honest MASK_ACTIVE tail is accepted."""
    d0 = _d0()
    _expect_accept(
        lambda: d0.normalize_tail_requirement(N70_LOGICAL, BASE, d0.TAIL_MASK_ACTIVE),
        "N70 over base64 with the honest MASK_ACTIVE tail policy",
    )


def test_bad_c1_full_tile_claimed_for_an_unproven_tail():
    """BAD: FULL_TILE on N70/base64 -> UNPROVEN_FULL_TILE."""
    d0 = _d0()
    _expect_refusal(
        lambda: d0.normalize_tail_requirement(N70_LOGICAL, BASE, d0.TAIL_FULL_TILE),
        "UNPROVEN_FULL_TILE",
        "FULL_TILE claimed for N70 over base64, where the tail issue is six lanes",
    )


def test_bad_c2_lowering_emits_a_full_predicate_for_a_partial_tail():
    """BAD: policy unchanged, lowering emits full -> MASK_COVERAGE_MISMATCH."""
    d0 = _d0()
    _expect_refusal(
        lambda: d0.verify_predicate_coverage(TAIL_SET, set(range(BASE))),
        "MASK_COVERAGE_MISMATCH",
        "a lowering that emitted a full 64-lane predicate where the bounds require "
        "the six-lane set {0..5}",
    )


def test_control_c3_exact_coverage_is_accepted():
    """CONTROL: the exact required six-lane set is accepted by the same checker."""
    d0 = _d0()
    _expect_accept(
        lambda: d0.verify_predicate_coverage(TAIL_SET, set(TAIL_SET)),
        "actual predicate coverage equal to the re-derived required set",
    )


# ---------------------------------------------------------------------------
# 4) UNKNOWN must never be read as 0. Two faults: an UNKNOWN extent normalised to
#    zero, and a normalised offset whose status is UNKNOWN but whose value is 0.
# ---------------------------------------------------------------------------

def test_control_d_a_known_extent_is_accepted():
    """CONTROL: a statically known extent is accepted."""
    d0 = _d0()
    _expect_accept(
        lambda: d0.normalize_tail_requirement(BASE, BASE, d0.TAIL_MASK_ACTIVE),
        "a statically known extent of 64 over base 64",
    )


def test_bad_d1_unknown_extent_is_refused_rather_than_defaulted():
    """BAD: UNKNOWN extent -> PIPELINE_BOUND_UNKNOWN, never 0."""
    d0 = _d0()
    _expect_refusal(
        lambda: d0.normalize_tail_requirement(d0.UNKNOWN, BASE, d0.TAIL_MASK_ACTIVE),
        "PIPELINE_BOUND_UNKNOWN",
        "an UNKNOWN tail extent passed into normalisation",
    )


def test_bad_d2_unknown_status_with_a_zero_value_is_refused():
    """BAD: status UNKNOWN paired with value 0 -> DEFAULTED_UNKNOWN_VALUE."""
    d0 = _d0()
    _expect_refusal(
        lambda: d0.verify_predicate_coverage(
            TAIL_SET, set(TAIL_SET), offset_value=0, offset_status=d0.UNKNOWN,
        ),
        "DEFAULTED_UNKNOWN_VALUE",
        "an UNKNOWN offset status recorded with the value 0, i.e. UNKNOWN silently "
        "defaulted to zero",
    )


# ---------------------------------------------------------------------------
# 5) part is written as 0 only when the projection is UNIQUE; a part1 request or an
#    unknown projection is refused rather than guessed.
# ---------------------------------------------------------------------------

class _UniqueProjection:
    """A projection the target reports as uniquely part0 (carrier/transport/dtype)."""

    def unique_part(self):
        return 0

    def __repr__(self):
        return "<unique projection: part0>"


class _PartOneRequest:
    def unique_part(self):
        return 1

    def __repr__(self):
        return "<projection requests part1>"


def test_control_e_a_unique_projection_normalises_to_part_zero():
    """CONTROL: the unique projection normalises to part 0."""
    d0 = _d0()
    part = _expect_accept(
        lambda: d0.select_part(_UniqueProjection()),
        "a projection that uniquely determines part0",
    )
    assert part == 0, (
        "the unique projection normalised to %r instead of 0; D0 requires part to be "
        "WRITTEN as 0 only when the projection is unique, and never guessed" % (part,)
    )


def test_bad_e1_part_one_request_is_unsupported():
    """BAD: a part1 request -> PART_SELECTION_UNSUPPORTED."""
    d0 = _d0()
    _expect_refusal(
        lambda: d0.select_part(_PartOneRequest()),
        "PART_SELECTION_UNSUPPORTED",
        "a target requesting part1, which D0 cannot select",
    )


# ---------------------------------------------------------------------------
# 6) A predicate produced in an older region (or overwritten) must not satisfy a
#    later consumer.
# ---------------------------------------------------------------------------

def test_control_f_a_fresh_same_region_producer_is_accepted():
    """CONTROL: a fresh producer in the consumer own region is accepted."""
    d0 = _d0()
    _expect_accept(
        lambda: d0.verify_predicate_coverage(
            TAIL_SET, set(TAIL_SET), producer_is_fresh=True),
        "a fresh predicate producer in the same execution region",
    )


def test_bad_f1_stale_producer_from_another_region_is_refused():
    """BAD: stale/overwritten producer -> PREDICATE_STALE_OR_OVERWRITTEN."""
    d0 = _d0()
    _expect_refusal(
        lambda: d0.verify_predicate_coverage(
            TAIL_SET, set(TAIL_SET), producer_is_fresh=False),
        "PREDICATE_STALE_OR_OVERWRITTEN",
        "a consumer whose producer was written in an older region and not re-issued",
    )


# ---------------------------------------------------------------------------
# 7) A VF predicate can be legal while the DMA window still cannot be expressed.
#    The VF mask must not be allowed to stand in for the DMA access.
# ---------------------------------------------------------------------------

def test_control_g_a_dma_expressible_tail_is_accepted():
    """CONTROL: a VF-legal tail whose DMA window IS expressible is accepted."""
    d0 = _d0()
    _expect_accept(
        lambda: d0.verify_predicate_coverage(
            TAIL_SET, set(TAIL_SET), dma_window=True),
        "a six-lane tail whose DMA access window is expressible",
    )


def test_bad_g1_vf_legal_but_dma_window_inexpressible():
    """BAD: VF legal, DMA window not expressible -> DMA_TAIL_ACCESS_UNSUPPORTED."""
    d0 = _d0()
    _expect_refusal(
        lambda: d0.verify_predicate_coverage(
            TAIL_SET, set(TAIL_SET), dma_window=False),
        "DMA_TAIL_ACCESS_UNSUPPORTED",
        "a tail legal for the VF predicate whose DMA access window cannot be "
        "expressed -- the VF mask must not excuse the DMA access",
    )


# ---------------------------------------------------------------------------
# Pass-level wiring. Without these the file could go green on predicate functions
# that no finalize path ever calls, which is how a verified rule ends up unbound.
# ---------------------------------------------------------------------------

def test_control_h_the_four_checkers_are_real_module_passes():
    """CONTROL: the four named checkers exist and are ModulePass subclasses."""
    d0 = _d0()
    from tilefoundry.passes.pass_base import ModulePass
    for name in REQUIRED_PASSES:
        assert hasattr(d0, name), (
            "%s is not exported, so nothing can register it at a pass point" % (name,)
        )
        pass_type = getattr(d0, name)
        assert isinstance(pass_type, type) and issubclass(pass_type, ModulePass), (
            "%s is not a ModulePass subclass, so the finalize pipeline cannot run it"
            % (name,)
        )


def test_bad_h1_the_after_checkers_are_registered_at_after_lower():
    """BAD: a checker that is never registered must be caught here.

    The design is explicit that checks invoked only from a script do not count as
    wired. Registration is observable in the pass registry, so a missing
    registration fails by name instead of being found by reading the code.
    """
    d0 = _d0()
    from tilefoundry.passes.pass_manager import PassPoint, _TARGET_PASSES
    registered = {pass_type.__name__
                  for (_target, point), pass_types in _TARGET_PASSES.items()
                  if point is PassPoint.AFTER_LOWER
                  for pass_type in pass_types}
    for name in ("VerifyPredicateCoverage", "VerifyPipelinePreservation"):
        assert name in registered, (
            "%s is not registered at AFTER_LOWER (registered AFTER passes: %s), so "
            "the public finalize path never runs it and a broken lowering would pass"
            % (name, sorted(registered) or "none")
        )


TESTS = (
    test_control_a_clone_keeps_the_pipeline_declaration,
    test_bad_a1_clone_drops_the_pipeline_declaration,
    test_bad_a2_convert_drops_the_pipeline_on_the_tir_loop,
    test_control_b_budget_two_with_one_version_each_is_accepted,
    test_control_b_budget_four_with_four_versions_is_accepted,
    test_bad_b1_declaration_says_one_but_lowering_claims_two,
    test_bad_b2_four_versions_under_a_budget_of_three,
    test_control_c_a_true_tail_requirement_is_accepted,
    test_bad_c1_full_tile_claimed_for_an_unproven_tail,
    test_bad_c2_lowering_emits_a_full_predicate_for_a_partial_tail,
    test_control_c3_exact_coverage_is_accepted,
    test_control_d_a_known_extent_is_accepted,
    test_bad_d1_unknown_extent_is_refused_rather_than_defaulted,
    test_bad_d2_unknown_status_with_a_zero_value_is_refused,
    test_control_e_a_unique_projection_normalises_to_part_zero,
    test_bad_e1_part_one_request_is_unsupported,
    test_control_f_a_fresh_same_region_producer_is_accepted,
    test_bad_f1_stale_producer_from_another_region_is_refused,
    test_control_g_a_dma_expressible_tail_is_accepted,
    test_bad_g1_vf_legal_but_dma_window_inexpressible,
    test_control_h_the_four_checkers_are_real_module_passes,
    test_bad_h1_the_after_checkers_are_registered_at_after_lower,
)


def main() -> int:
    print("D0 TDD baseline: four decisive bad controls + predicate refusal codes")
    print("  rule module candidates: %s" % ", ".join(PASS_MODULE_CANDIDATES))
    passed, failed, errored = [], [], []
    for case in TESTS:
        try:
            case()
        except AssertionError as error:
            failed.append(case.__name__)
            print("  FAIL  %s" % case.__name__)
            print("        %s" % str(error)[:600])
        except Exception as error:  # a real crash, not a named refusal
            errored.append(case.__name__)
            print("  ERROR %s: %s: %s" % (case.__name__, type(error).__name__, error))
            traceback.print_exc()
        else:
            passed.append(case.__name__)
            print("  PASS  %s" % case.__name__)
    total = len(TESTS)
    print()
    print("%d/%d passed, %d failed, %d errored"
          % (len(passed), total, len(failed), len(errored)))
    if errored:
        return 2
    if failed:
        print()
        print("RED BY DESIGN until D0-Target lands. Each line names the rule and the")
        print("refusal code it owes: a failure here is the contract, not a broken test.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())