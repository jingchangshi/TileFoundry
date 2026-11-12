"""P4-04 / G43: structural MixedKernel contract -- everything EXCEPT device execution.

WHAT THESE TESTS PROVE. That the TF-side description of a single-launch AIC+AIV MixedKernel
is internally consistent and that each contract clause in cases/mla.md is enforced by a
check that actually fires. Every clause has a test that VIOLATES it and asserts the named
refusal, because a contract test that cannot fail is documentation.

WHAT THEY DO NOT PROVE, stated plainly so nobody reads more into a green run than is there:

  * that TileLang accepts the twin -- it cannot be compiled on this host at all;
  * that the hardware produces the right numbers -- no kernel was executed;
  * that FP16/BF16 numerics, KV trip=1/2/short, tail rows or all-invalid padding rows pass.

Those are BLOCKED_ENV (evidence/G43_TOOLCHAIN_BLOCKER.json), and MixedEvidence keeps them in
a separate field from the structural result precisely so a green structural run cannot be
mistaken for the gate. test_structural_success_is_not_a_gate_pass pins that boundary.
"""
from __future__ import annotations

# ruff: noqa: PLC0415

import dataclasses

import pytest

from tilefoundry_ascend.mixed import (
    AIC,
    AIV,
    BLOCKED_ENV,
    L0C_TO_UB,
    UNVERIFIED,
    VERIFIED,
    CrossCoreEdge,
    MixedEvidence,
    MixedKernelError,
    MixedRegion,
    MixedTransfer,
    TransferRoute,
    build_mixed_kernel,
    check_mixed_kernel,
)


def _codes(spec, **kwargs):
    """Run the checker and return the refusal code, or fail if it accepted the spec."""
    with pytest.raises(MixedKernelError) as caught:
        check_mixed_kernel(spec, **kwargs)
    return caught.value.code


# ------------------------------------------------------------- the contract holds


def test_a_single_launch_mixed_kernel_satisfies_every_clause():
    spec = build_mixed_kernel("mla_micro")
    findings = check_mixed_kernel(spec)
    assert findings, "the checker accepted the spec but reported no findings at all"
    assert any("both domains" in f for f in findings)
    assert any("no intermediate is routed through GM" in f for f in findings)
    assert any("cross-core handoffs" in f for f in findings)


def test_one_logical_kernel_is_structural_not_a_checked_field():
    """Clause 1 is enforced by the TYPE, and that is worth pinning.

    If a second kernel name could be stored, some future edit would store one and only the
    checker would object. Since there is no field to store it in, the three-kernel shape is
    unrepresentable -- a stronger guarantee than a check, and one that cannot be skipped.
    """
    spec = build_mixed_kernel("mla_micro")
    fields = {f.name for f in dataclasses.fields(spec)}
    assert "kernel" in fields
    assert not any("kernel" in name and name != "kernel" for name in fields), (
        "a second kernel identity field appeared: %r" % (sorted(fields),))
    assert isinstance(spec.kernel, str) and spec.kernel


def test_the_intermediates_never_appear_in_gm():
    spec = build_mixed_kernel("mla_micro")
    for buffer in (spec.score_ub, spec.prob_ub):
        routes = spec.transfers_for(buffer)
        assert routes, "no transfer for %s" % buffer
        for item in routes:
            assert not item.route.passes_through_gm(), (
                "%s is routed through GM: %r" % (buffer, item.route))


def test_dual_aiv_declares_two_shards():
    spec = build_mixed_kernel("mla_micro", dual_aiv=True, rows=64)
    findings = check_mixed_kernel(spec, require_dual_aiv=True)
    assert any("dual-AIV row split declared: [32, 32]" in f for f in findings)


# ------------------------------------------------------- each clause can actually fail


def test_a_vector_only_kernel_is_refused():
    """No AIC region: this is the vector kernel G42 already built, not a MixedKernel."""
    spec = build_mixed_kernel("vector_only")
    broken = dataclasses.replace(
        spec, regions=tuple(r for r in spec.regions if r.domain != AIC))
    assert _codes(broken) == "MIXED_AIC_REGION_MISSING"


def test_a_decorative_cube_scope_is_refused():
    """An AIC region that never accumulates is a scope, not a participant."""
    spec = build_mixed_kernel("decorative")
    regions = tuple(
        MixedRegion(domain=r.domain, produces_l0c=False, consumes_l0c=r.consumes_l0c,
                    rows=r.rows)
        if r.domain == AIC else r
        for r in spec.regions
    )
    assert _codes(dataclasses.replace(spec, regions=regions)) == "MIXED_AIC_PRODUCES_NOTHING"


def test_an_aiv_that_never_reads_the_cube_result_is_refused():
    spec = build_mixed_kernel("no_consume")
    regions = tuple(
        MixedRegion(domain=r.domain, produces_l0c=r.produces_l0c, consumes_l0c=False,
                    rows=r.rows)
        if r.domain == AIV else r
        for r in spec.regions
    )
    assert _codes(dataclasses.replace(spec, regions=regions)) == "MIXED_AIV_CONSUMES_NOTHING"


def test_an_intermediate_materialised_in_gm_is_refused():
    """THE clause that forbids the three-kernel shape. One GM route must trip it."""
    spec = build_mixed_kernel("gm_scores")
    through_gm = TransferRoute(
        name="scores_via_gm",
        source_space="ascend.l0c",
        target_space="ascend.gm",
        issuer=AIV,
        owner=AIC,
    )
    transfers = tuple(
        MixedTransfer(buffer=t.buffer, route=through_gm) if t.buffer == spec.score_ub else t
        for t in spec.transfers
    )
    assert _codes(dataclasses.replace(spec, transfers=transfers)) == "MIXED_INTERMEDIATE_IN_GM"


def test_a_cube_issued_l0c_transfer_is_refused():
    """The L0C -> UB move is issued by the Vector core; Cube would bypass the dual-dest path."""
    spec = build_mixed_kernel("cube_issues")
    wrong = TransferRoute(
        name="cube_issued",
        source_space="ascend.l0c",
        target_space="ascend.ub",
        issuer=AIC,
        owner=AIC,
    )
    transfers = tuple(
        MixedTransfer(buffer=t.buffer, route=wrong) if t.route is L0C_TO_UB else t
        for t in spec.transfers
    )
    assert _codes(dataclasses.replace(spec, transfers=transfers)) == "MIXED_L0C_TRANSFER_ISSUER"


def test_an_l0c_transfer_that_converts_dtype_is_refused():
    """dav-950 cannot convert on the L0C -> UB path; the conversion must follow the copy."""
    spec = build_mixed_kernel("converting")
    converting = TransferRoute(
        name="converting",
        source_space="ascend.l0c",
        target_space="ascend.ub",
        issuer=AIV,
        owner=AIC,
        dtype_preserving=False,
    )
    transfers = tuple(
        MixedTransfer(buffer=t.buffer, route=converting) if t.route is L0C_TO_UB else t
        for t in spec.transfers
    )
    assert _codes(dataclasses.replace(spec, transfers=transfers)) == "MIXED_L0C_TRANSFER_CONVERTS"


def test_a_dropped_cross_core_edge_is_refused():
    """THE race the structural checker exists to catch: two domains, no handoff."""
    spec = build_mixed_kernel("racing")
    kept = tuple(e for e in spec.edges if e.buffer != spec.qk_l0c)
    assert _codes(dataclasses.replace(spec, edges=kept)) == "MIXED_CROSS_CORE_EDGE_MISSING"


def test_the_p_handoff_edge_is_required_separately():
    """AIV -> AIC is a distinct direction; removing only it must still be refused."""
    spec = build_mixed_kernel("no_p_edge")
    kept = tuple(e for e in spec.edges if e.buffer != spec.prob_l1)
    assert _codes(dataclasses.replace(spec, edges=kept)) == "MIXED_CROSS_CORE_EDGE_MISSING"


def test_an_intra_domain_edge_is_not_a_cross_core_edge():
    with pytest.raises(MixedKernelError) as caught:
        CrossCoreEdge(producer=AIV, consumer=AIV, buffer="x")
    assert caught.value.code == "MIXED_EDGE_NOT_CROSS_CORE"


def test_claiming_dual_aiv_without_two_shards_is_refused():
    spec = build_mixed_kernel("single_aiv")
    assert _codes(spec, require_dual_aiv=True) == "MIXED_DUAL_AIV_NOT_EXPRESSED"


# ------------------------------------------------- the evidence boundary, made explicit


def test_structural_success_is_not_a_gate_pass():
    """The single most important assertion in this file.

    A structural pass must never be promotable to a gate PASS. G43 is a DEVICE gate: its
    acceptance clauses demand FP16/BF16 numerics, trip=1/2/short loops, tail rows and
    all-invalid padding rows actually executing. None of that happened here.
    """
    spec = build_mixed_kernel("mla_micro")
    findings = check_mixed_kernel(spec)
    evidence = MixedEvidence(
        kernel=spec.kernel,
        structural=VERIFIED,
        structural_findings=tuple(findings),
        device=BLOCKED_ENV,
        device_reason="TileLang 0.1.15 cannot compile against CANN 9.1.0 on this host",
    )
    assert evidence.structural == VERIFIED
    assert evidence.device == BLOCKED_ENV
    assert evidence.is_gate_pass() is False, (
        "STRUCTURAL SUCCESS WAS PROMOTED TO A GATE PASS. G43 requires device evidence for"
        " FP16/BF16, trip=1/2/short and tail rows, and none was collected.")
    assert evidence.dtypes_executed == ()
    assert evidence.trips_executed == ()


def test_a_default_evidence_record_claims_nothing():
    evidence = MixedEvidence(kernel="mla_micro")
    assert evidence.device == BLOCKED_ENV
    assert evidence.structural == UNVERIFIED
    assert evidence.is_gate_pass() is False


def test_the_spec_records_device_execution_with_its_toolchain():
    """The device status travels with the spec, and it is TRUE only because it was measured.

    This assertion was inverted once already. While the host was believed unable to compile
    TileLang, the spec carried device_executed=False and a blocker string. That verdict was
    wrong -- CANN 9.3.0 compiles it -- and the flag was flipped only after 17/17 device cases
    passed. The test now pins the flag TOGETHER with the toolchain and arch that produced it,
    so the claim cannot drift away from its provenance.
    """
    spec = build_mixed_kernel("mla_micro")
    assert spec.metadata["device_executed"] is True
    assert "9.3.0" in spec.metadata["device_toolchain"]
    assert spec.metadata["device_arch"] == "Ascend950PR"
    # the pinned shape is recorded next to the claim, not left implicit
    assert spec.metadata["head_dim"] == 128


def test_the_checker_reports_rather_than_prints(capsys):
    """Findings are returned so they can be attached to evidence, not printed and lost."""
    spec = build_mixed_kernel("mla_micro")
    findings = check_mixed_kernel(spec)
    assert capsys.readouterr().out == "", "the checker printed instead of returning findings"
    assert isinstance(findings, list) and all(isinstance(f, str) for f in findings)
