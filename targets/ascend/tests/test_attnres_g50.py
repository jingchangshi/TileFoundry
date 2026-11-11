"""P5-01/G50: attention_residual model, schedule, kernel-emission and evidence tests.

These tests are about CONTRACTS, not numbers. The numbers come from device execution and are
recorded in evidence/G50_DEVICE_EVIDENCE.json; a test here asserting a device value would be a
second, drifting copy of that record. What is tested here is that the description cannot silently
become a different operator.
"""
import json
import re
from pathlib import Path

import pytest

from tilefoundry_ascend.attnres import (
    AttentionResidualError,
    BLOCKED_ONLINE,
    BLOCK_COMPUTE,
    FACTORED_GW,
    FAMILIES,
    FAILED,
    LITERAL_FP32,
    MATCHED,
    ONLINE,
    RESIDENT_N,
    build_attention_residual,
    build_attnres_twin,
    build_scheduled_hir,
    check_attnres_conformance,
    check_scheduled_hir,
    measured_attention_evidence,
    parse_attnres_twin,
)


def _spec(**kw):
    base = dict(n=4, tokens=32, d=512)
    base.update(kw)
    return build_attention_residual(**base)


# ------------------------------------------------------------------ operator domain

def test_n_below_two_is_outside_the_domain():
    """N == 1 makes the depth softmax identically 1, so the score path cannot affect the
    output and the operator is no longer the one being specified."""
    with pytest.raises(AttentionResidualError) as exc:
        build_attention_residual(n=1, tokens=8, d=64)
    assert exc.value.code == "ATTNRES_N_BELOW_DOMAIN"


def test_only_the_declared_storage_dtypes_are_accepted():
    with pytest.raises(AttentionResidualError) as exc:
        build_attention_residual(n=2, tokens=8, d=64, dtype="float32")
    assert exc.value.code == "ATTNRES_BAD_DTYPE"


def test_unknown_family_and_score_mode_are_refused():
    with pytest.raises(AttentionResidualError) as exc:
        build_attention_residual(n=4, tokens=8, d=64, family="not_a_family")
    assert exc.value.code == "ATTNRES_UNKNOWN_FAMILY"
    with pytest.raises(AttentionResidualError) as exc:
        build_attention_residual(n=4, tokens=8, d=64, score_mode="not_a_mode")
    assert exc.value.code == "ATTNRES_UNKNOWN_SCORE_MODE"


# ------------------------------------------------------------------ the three families

def test_online_is_cn_one_and_pays_one_update_per_block():
    spec = _spec(family=ONLINE, n=8)
    assert spec.block_n == 1
    assert spec.state_updates() == 8


def test_resident_n_is_cn_n_and_pays_a_single_update():
    spec = _spec(family=RESIDENT_N, n=8)
    assert spec.block_n == 8
    assert spec.state_updates() == 1


def test_blocked_online_is_a_strict_partial_block():
    spec = _spec(family=BLOCKED_ONLINE, n=8)
    assert 1 < spec.block_n < 8
    assert spec.state_updates() == 2


def test_the_families_are_ordered_by_state_update_count():
    """The whole point of CN is fewer merges. If the counts did not fall in this order, two
    family names would be describing the same schedule."""
    counts = {}
    for family in FAMILIES:
        counts[family] = _spec(family=family, n=16).state_updates()
    assert counts[ONLINE] > counts[BLOCKED_ONLINE] > counts[RESIDENT_N]


def test_a_family_whose_cn_contradicts_it_is_refused():
    """Otherwise 'online' with CN=N would be resident_n under a different name."""
    with pytest.raises(AttentionResidualError) as exc:
        build_attention_residual(n=8, tokens=8, d=64, family=ONLINE, block_n=4)
    assert exc.value.code == "ATTNRES_CN_CONTRADICTS_FAMILY"
    with pytest.raises(AttentionResidualError) as exc:
        build_attention_residual(n=8, tokens=8, d=64, family=RESIDENT_N, block_n=2)
    assert exc.value.code == "ATTNRES_CN_CONTRADICTS_FAMILY"
    with pytest.raises(AttentionResidualError) as exc:
        build_attention_residual(n=8, tokens=8, d=64, family=BLOCKED_ONLINE, block_n=8)
    assert exc.value.code == "ATTNRES_CN_CONTRADICTS_FAMILY"


# ------------------------------------------------------------------ axes and D handling

def test_padded_d_is_a_power_of_two_and_only_an_allocation_detail():
    for d in (2047, 1000, 257, 100, 512, 7168):
        spec = _spec(d=d)
        pd = spec.padded_d()
        assert pd >= d
        assert (pd & (pd - 1)) == 0, "padded width must be a power of two"
        assert spec.d == d, "the real D must not be rewritten by padding"


def test_d_tail_is_recognised_for_non_power_of_two_widths():
    assert _spec(d=2048).denormalises_d() is True
    assert _spec(d=2047).denormalises_d() is False


def test_the_rms_denominator_is_the_real_d():
    """A padded denominator would rescale every score, so the spec must record the real D."""
    spec = _spec(d=2047)
    hir = build_scheduled_hir(spec)
    assert hir.address_equations["D"] == "2047"


# ------------------------------------------------------------------ scheduled HIR

def test_the_hir_has_one_launch_and_all_pipes():
    hir = build_scheduled_hir(_spec())
    findings = check_scheduled_hir(hir)
    assert any("one logical launch" in f for f in findings)
    assert any("pipe(s)" in f for f in findings)


def test_the_hir_names_every_task_the_spec_lists():
    hir = build_scheduled_hir(_spec())
    names = set(hir.task_names())
    assert names == {"LOAD_V", "BLOCK_COMPUTE", "LOAD_X", "FINALIZE", "STORE_Y"}


def test_v_release_belongs_to_the_weighted_sum_task():
    """The spec calls this out twice: attaching the V release to the score alone would let a
    reload overwrite a tile the weighted sum still needs."""
    hir = build_scheduled_hir(_spec())
    assert hir.metadata["v_last_consumer"] == BLOCK_COMPUTE
    check_scheduled_hir(hir)  # must not raise


def test_the_softmax_axis_is_recorded_as_n():
    hir = build_scheduled_hir(_spec())
    assert hir.metadata["softmax_axis"] == "N"


def test_every_edge_endpoint_exists():
    hir = build_scheduled_hir(_spec())
    names = set(hir.task_names())
    for edge in hir.edges:
        assert edge.producer in names
        assert edge.consumer in names


def test_a_non_fp32_state_is_refused():
    from dataclasses import replace
    from tilefoundry_ascend.attnres import FP32State
    spec = _spec()
    broken = replace(spec, state=FP32State(m="float16"))
    with pytest.raises(AttentionResidualError) as exc:
        check_scheduled_hir(build_scheduled_hir(broken))
    assert exc.value.code == "ATTNRES_STATE_NOT_FP32"


def test_a_second_launch_is_refused():
    from dataclasses import replace
    spec = replace(_spec(), launches=2)
    with pytest.raises(AttentionResidualError) as exc:
        build_scheduled_hir(spec)
    assert exc.value.code == "ATTNRES_MULTIPLE_LAUNCHES"


# ------------------------------------------------------------------ emitted kernel

def test_the_emitted_kernel_has_one_launch_scope():
    spec = _spec()
    src = build_attnres_twin(spec)
    assert len(re.findall(r"with T\.(?:Kernel|MixedKernel)\(", src)) == 1


def test_the_emitted_kernel_aggregates_the_original_v():
    """Using the normalised key instead of the original V keeps the shapes and the ballpark
    magnitude, and is simply a different operator."""
    src = build_attnres_twin(_spec())
    assert "acc_ub[i] = st[3] * acc_ub[i] + st[4] * af[i]" in src
    assert 'af[i] = T.if_then_else(i < D, T.cast(v_ub[i], "float32")' in src


def test_the_emitted_kernel_declares_fp32_state():
    src = build_attnres_twin(_spec())
    acc = [l for l in src.splitlines() if "acc_ub = T.alloc_shared" in l][0]
    st = [l for l in src.splitlines() if "st = T.alloc_shared" in l][0]
    assert '"float32"' in acc
    assert '"float32"' in st


def test_literal_fp32_and_factored_gw_emit_different_score_paths():
    """The two modes are not interchangeable: hoisting the division changes the float order."""
    lit = build_attnres_twin(_spec(score_mode=LITERAL_FP32))
    fac = build_attnres_twin(_spec(score_mode=FACTORED_GW))
    assert "(vf[i] / rms[0]) * g_ub[i] * w_ub[i]" in lit
    assert "vf[i] * (g_ub[i] * w_ub[i])" in fac
    assert lit != fac


def test_both_score_modes_conform():
    for mode in (LITERAL_FP32, FACTORED_GW):
        spec = _spec(score_mode=mode)
        report = check_attnres_conformance(spec, parse_attnres_twin(build_attnres_twin(spec)))
        assert report.status == MATCHED, report.failures


# ------------------------------------------------------------------ refusals

def test_a_second_kernel_is_refused():
    spec = _spec()
    src = build_attnres_twin(spec) + '\n\ndef second():\n    with T.Kernel(1):\n        pass\n'
    report = check_attnres_conformance(spec, parse_attnres_twin(src))
    assert report.status == FAILED
    assert any("TWIN_MULTIPLE_KERNELS" in f for f in report.failures)


def test_a_padded_denominator_is_refused():
    spec = _spec(d=2047)
    src = build_attnres_twin(spec).replace("qs[0] / T.float32(D)", "qs[0] / T.float32(PD)")
    report = check_attnres_conformance(spec, parse_attnres_twin(src))
    assert report.status == FAILED
    assert any("TWIN_PADDED_DENOMINATOR" in f for f in report.failures)


def test_aggregating_the_normalised_key_is_refused():
    spec = _spec()
    src = build_attnres_twin(spec).replace(
        'af[i] = T.if_then_else(i < D, T.cast(v_ub[i], "float32"), T.float32(0))',
        'af[i] = T.if_then_else(i < D, T.cast(v_ub[i] / (qs[0] + eps), "float32"), T.float32(0))')
    report = check_attnres_conformance(spec, parse_attnres_twin(src))
    assert report.status == FAILED
    assert any("TWIN_AGGREGATION_NOT_ORIGINAL_V" in f for f in report.failures)


def test_a_missing_residual_path_is_refused():
    spec = _spec()
    src = build_attnres_twin(spec).replace("st[6] = T.float32(1) - alpha", "st[6] = T.float32(0)")
    report = check_attnres_conformance(spec, parse_attnres_twin(src))
    assert report.status == FAILED
    assert any("TWIN_MISSING_RESIDUAL" in f for f in report.failures)


def test_hand_written_cross_core_flags_are_refused():
    """This kernel has ONE core, so a hand-written flag implies an obligation that does not
    exist -- and a partial set would be worse than none."""
    spec = _spec()
    src = build_attnres_twin(spec).replace(
        '                st[1] = T.float32(0)',
        '                st[1] = T.float32(0)\n                T.ascend_cross_core_set_flag(0, "cube", 0)')
    report = check_attnres_conformance(spec, parse_attnres_twin(src))
    assert report.status == FAILED
    assert any("TWIN_MANUAL_SYNC" in f for f in report.failures)


def test_a_declared_score_mode_that_is_not_emitted_is_refused():
    spec = _spec(score_mode=LITERAL_FP32)
    src = build_attnres_twin(_spec(score_mode=FACTORED_GW))
    report = check_attnres_conformance(spec, parse_attnres_twin(src))
    assert report.status == FAILED
    assert any("TWIN_SCORE_MODE_MISMATCH" in f for f in report.failures)


# ------------------------------------------------------------------ evidence

def test_the_evidence_requires_both_structural_and_device():
    """A structural pass must never be readable as a device pass."""
    from dataclasses import replace
    from tilefoundry_ascend.attnres import AttentionEvidence, BLOCKED_ENV, VERIFIED
    e = AttentionEvidence(kernel="k", structural=VERIFIED, device=BLOCKED_ENV)
    assert e.is_gate_pass() is False


def test_the_gate_passes_on_the_measured_record():
    e = measured_attention_evidence(_spec())
    assert e.is_gate_pass() is True
    assert e.cases_passed == 20
    assert e.cases_total == 20
    assert "Ascend950PR" in e.device_reason


def test_the_evidence_record_is_the_only_source_of_the_device_claim():
    """The numbers are read from the committed record, not duplicated in code."""
    here = Path(__file__).resolve().parents[1]
    data = json.loads((here / "evidence" / "G50_DEVICE_EVIDENCE.json").read_text())
    assert data["official_cases"]["total"] == 20
    assert data["official_cases"]["passed"] == 20
    assert "NOT edited" in data["official_cases"]["thresholds"]


def test_the_record_states_what_is_not_claimed():
    """A record that only lists successes invites the reader to assume the rest."""
    here = Path(__file__).resolve().parents[1]
    data = json.loads((here / "evidence" / "G50_DEVICE_EVIDENCE.json").read_text())
    not_claimed = " ".join(data["not_claimed"])
    assert "persistent" in not_claimed.lower()
    assert "G51" in not_claimed or "G52" in not_claimed
