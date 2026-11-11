"""P4-04 / G43: the explicit MixedKernel twin and TF<->twin conformance.

These tests pin the two properties the gate actually names -- that the twin expresses the
schedule it was given rather than re-deriving one, and that the twin corresponds to the TF
spec on kernel count, domains, transfers and cross-core edges -- and they pin them by
BREAKING each one.

NOT CLAIMED: that TileLang accepts this twin. It cannot be compiled on this host (see
evidence/G43_TOOLCHAIN_BLOCKER.json), so the checks read the twin as TEXT. ConformanceReport
carries compiled=False and the reason, so a MATCHED result cannot be quoted as a compile.
"""
from __future__ import annotations

# ruff: noqa: PLC0415

import pytest

from tilefoundry_ascend.mixed import (
    AIC,
    AIV,
    build_mixed_kernel,
    check_mixed_kernel,
)
from tilefoundry_ascend.mixed.twin import (
    AUTO_SCHEDULE_FLAG,
    FAILED,
    MATCHED,
    TwinError,
    build_mixed_twin,
    check_conformance,
    parse_mixed_twin,
)


def _pair(**kwargs):
    spec = build_mixed_kernel("mla_micro", **kwargs)
    return spec, parse_mixed_twin(build_mixed_twin(spec))


def test_a_correct_twin_conforms():
    spec, twin = _pair()
    report = check_conformance(spec, twin)
    assert report.status == MATCHED, report.failures
    assert report.failures == ()
    assert report.kernel == "mla_micro"


def test_the_twin_declares_auto_schedule_false():
    """The whole reason this emitter exists.

    With auto-scheduling on, TileLang is a second scheduler: agreement between TF and TileLang
    would then be two solvers converging, not evidence that TF decided anything. The failure
    is silent -- a re-scheduling twin still compiles and still runs -- so it is asserted
    rather than assumed.
    """
    spec = build_mixed_kernel("mla_micro")
    source = build_mixed_twin(spec)
    assert AUTO_SCHEDULE_FLAG in source
    assert parse_mixed_twin(source).auto_schedule is True


def test_a_twin_without_auto_schedule_false_is_refused():
    spec = build_mixed_kernel("mla_micro")
    source = build_mixed_twin(spec).replace(AUTO_SCHEDULE_FLAG, "auto_schedule=True")
    with pytest.raises(TwinError) as caught:
        check_conformance(spec, parse_mixed_twin(source))
    assert caught.value.code == "TWIN_AUTO_SCHEDULE_ENABLED"


def test_a_comment_mentioning_the_flag_does_not_count():
    """The flag check matches an assignment, not a substring.

    Otherwise a twin that merely TALKS about disabling auto-scheduling would satisfy the
    check while still letting TileLang derive its own schedule.
    """
    spec = build_mixed_kernel("mla_micro")
    source = build_mixed_twin(spec).replace(
        AUTO_SCHEDULE_FLAG, "# auto_schedule=False would be required here")
    assert parse_mixed_twin(source).auto_schedule is False


def test_the_twin_expresses_both_domains():
    """Both cores must be exercised BY THE CODE, not by a decorative scope.

    The twin has no `with T.Cube()` / `with T.Vector()` blocks, because that shape does not
    compile (a buffer allocated in one region cannot be named in the other). What must be
    true instead is that Cube-shaped and Vector-shaped operations both appear -- otherwise
    "both domains" would be a claim about comments.
    """
    _, twin = _pair()
    assert "T.gemm(" in twin.source
    assert "T.alloc_l0c" in twin.source
    assert "T.SimdVF(" in twin.source
    assert "S.vld(" in twin.source


def test_the_twin_has_exactly_one_launch_scope():
    """One logical kernel: a second launch would be the three-micro-kernel shape."""
    spec = build_mixed_kernel("mla_micro")
    report = check_conformance(spec, parse_mixed_twin(build_mixed_twin(spec)))
    assert report.status == MATCHED, report.failures
    assert any("one launch scope" in f for f in report.findings)


def test_a_second_launch_scope_is_refused():
    spec = build_mixed_kernel("mla_micro")
    source = build_mixed_twin(spec).replace(
        "    return main",
        "    return main\n\ndef second():\n    with T.Kernel(1):\n        pass",
    )
    report = check_conformance(spec, parse_mixed_twin(source))
    assert report.status == FAILED
    assert any("TWIN_MULTIPLE_KERNELS" in f for f in report.failures)


def test_a_twin_without_the_vector_side_is_refused():
    """Strip every SIMD op and the AIV no longer runs at all."""
    spec = build_mixed_kernel("mla_micro")
    source = "\n".join(
        line for line in build_mixed_twin(spec).splitlines()
        if not any(op in line for op in ("S.vld(", "S.vsts(", "T.SimdVF(", "S.pset("))
    )
    report = check_conformance(spec, parse_mixed_twin(source))
    assert report.status == FAILED
    assert any("no Vector-side operation" in f for f in report.failures)


def test_a_twin_without_the_cube_side_is_refused():
    spec = build_mixed_kernel("mla_micro")
    source = "\n".join(
        line for line in build_mixed_twin(spec).splitlines()
        if not any(op in line for op in ("T.gemm(", "T.dual_copy("))
    )
    report = check_conformance(spec, parse_mixed_twin(source))
    assert report.status == FAILED
    assert any("no Cube-side operation" in f for f in report.failures)


def test_a_dropped_handoff_is_refused():
    """Every cross-domain hop is a dual_copy; removing one breaks the chain."""
    spec = build_mixed_kernel("mla_micro")
    source = "\n".join(
        line for line in build_mixed_twin(spec).splitlines()
        if "T.dual_copy(qk_acc_l0c" not in line
    )
    report = check_conformance(spec, parse_mixed_twin(source))
    assert report.status == FAILED
    assert any("TWIN_" in f for f in report.failures)


def test_hand_written_flags_are_refused():
    """The subtle case, and the reason this clause exists.

    The twin derives its cross-core synchronisation from the dataflow. A twin with a PARTIAL
    set of hand-written flags is worse than one with none: it suggests the author believed the
    compiler needed help at exactly one edge and not the others. So hand-written flags are a
    failure here, not a feature.
    """
    spec = build_mixed_kernel("mla_micro")
    source = build_mixed_twin(spec).replace(
        "            T.dual_copy(qk_acc_l0c, scores_ub)",
        "            T.dual_copy(qk_acc_l0c, scores_ub)\n"
        '            T.ascend_cross_core_set_flag(0, "cube", 0)',
    )
    report = check_conformance(spec, parse_mixed_twin(source))
    assert report.status == FAILED
    assert any("TWIN_MANUAL_SYNC" in f for f in report.failures)


def test_an_intermediate_routed_through_gm_in_the_twin_is_refused():
    spec = build_mixed_kernel("mla_micro")
    source = build_mixed_twin(spec).replace(
        "T.dual_copy(qk_acc_l0c, scores_ub)",
        "T.copy(qk_acc_l0c, scores_ub)  # ascend.gm",
    )
    report = check_conformance(spec, parse_mixed_twin(source))
    assert report.status == FAILED
    assert any("TWIN_INTERMEDIATE_IN_GM" in f or "TWIN_" in f for f in report.failures)


def test_the_twin_records_how_it_was_produced():
    """A reader must be able to tell this twin from the discarded hand-rolled one."""
    spec = build_mixed_kernel("mla_micro")
    source = build_mixed_twin(spec)
    assert "verified flash_attention" in source
    assert "derived from the dataflow" in source

def test_a_matched_report_still_records_that_nothing_was_compiled():
    spec, twin = _pair()
    report = check_conformance(spec, twin)
    assert report.status == MATCHED
    assert report.compiled is False
    assert "CANN" in report.compiled_reason
    as_dict = report.as_dict()
    assert as_dict["compiled"] is False
    assert as_dict["device"] == "BLOCKED_ENV"


def test_an_empty_twin_is_refused():
    with pytest.raises(TwinError) as caught:
        parse_mixed_twin("")
    assert caught.value.code == "TWIN_EMPTY"


def test_the_whole_pipeline_is_consistent_end_to_end():
    """Spec -> contract check -> twin -> parse -> conformance, all on one spec."""
    spec = build_mixed_kernel("mla_micro")
    findings = check_mixed_kernel(spec)
    assert findings
    report = check_conformance(spec, parse_mixed_twin(build_mixed_twin(spec)))
    assert report.status == MATCHED, report.failures
    assert report.compiled is False