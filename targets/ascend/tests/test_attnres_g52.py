"""P5-02/G52: blocked_online / resident_n, the two score policies, and resource pruning.

The device numbers live in evidence/G52_DEVICE_EVIDENCE.json. These tests check the properties
that make the families genuinely distinct and the pruning genuinely binding, because those are
the claims a reader cannot verify from a pass count alone.
"""
import json
from pathlib import Path

import pytest

from tilefoundry_ascend.attnres import (
    AttentionResidualError,
    BLOCKED_ONLINE,
    FACTORED_GW,
    LITERAL_FP32,
    ONLINE,
    RESIDENT_N,
    build_attention_residual,
)


# ------------------------------------------------------------------ block-state algebra

def test_state_updates_are_ceil_n_over_cn():
    from tilefoundry_ascend.attnres.kernel_cn import n_blocks
    for n, cn, expected in ((8, 1, 8), (8, 2, 4), (8, 4, 2), (8, 8, 1), (7, 4, 2), (5, 2, 3)):
        assert n_blocks(n, cn) == expected
        spec = build_attention_residual(n=n, tokens=8, d=64,
                                        family=(ONLINE if cn == 1 else
                                                RESIDENT_N if cn == n else BLOCKED_ONLINE),
                                        block_n=cn)
        assert spec.state_updates() == expected


def test_the_family_name_is_derived_from_cn_not_stored():
    """A report must not be able to label a schedule it did not run."""
    from tilefoundry_ascend.attnres.kernel_cn import kernel_summary
    assert kernel_summary(n=8, block_n=1, d=64)["family"] == ONLINE
    assert kernel_summary(n=8, block_n=4, d=64)["family"] == BLOCKED_ONLINE
    assert kernel_summary(n=8, block_n=8, d=64)["family"] == RESIDENT_N


def test_the_merge_algebra_is_stated_once():
    """The kernel and the oracle must be checked against the same statement of the rule."""
    from tilefoundry_ascend.attnres.kernel_cn import merge_algebra
    a = merge_algebra()
    assert "rho * l + bet * l_b" in a["l_update"]
    assert "rho * acc + bet * u_b" in a["acc_update"]
    assert "(1 - alpha)" in a["final"]


def test_the_two_pass_form_admits_its_second_v_read():
    """The blocked form re-reads V; recording it is the difference between a cost and a bug."""
    from tilefoundry_ascend.attnres.kernel_cn import kernel_summary
    s = kernel_summary(n=8, block_n=4, d=64)
    assert s["v_reads_per_block"] == 2
    assert s["v_reads_total"] == 16


def test_cn_outside_the_measured_domain_is_refused_by_the_kernel():
    from tilefoundry_ascend.attnres.kernel_cn import check_config
    with pytest.raises(ValueError):
        check_config(n=8, block_n=0, score_mode=LITERAL_FP32)
    with pytest.raises(ValueError):
        check_config(n=8, block_n=9, score_mode=LITERAL_FP32)
    with pytest.raises(ValueError):
        check_config(n=8, block_n=2, score_mode="nonsense")
    with pytest.raises(ValueError):
        check_config(n=1, block_n=1, score_mode=LITERAL_FP32)


def test_only_measured_cn_values_are_flagged_measured():
    from tilefoundry_ascend.attnres.kernel_cn import kernel_summary, MEASURED_CN
    assert 16 not in MEASURED_CN
    assert kernel_summary(n=32, block_n=16, d=64)["measured"] is False


# ------------------------------------------------------------------ resident_n feasibility

def test_resident_n_gets_a_hard_feasibility_wall():
    """Holding all N blocks is the family's definition, so it must be a wall, not a cost."""
    from tilefoundry_ascend.attnres.resources import (
        generate_candidates, INFEASIBLE, RESIDENT_N as RN)
    cands = generate_candidates(n=64, tokens=64, d=4096)
    resident = [c for c in cands if c.family == RN]
    assert resident, "resident_n must appear in the generated space"
    assert all(not c.feasible for c in resident), (
        "all 64 blocks of 4096 elements cannot fit in UB, so every resident_n candidate must "
        "be rejected")


# ------------------------------------------------------------------ resource pruning

def test_pruning_rejects_on_real_reasons_not_vibes():
    from tilefoundry_ascend.attnres.resources import generate_candidates, summarise, UB_BYTES
    s = summarise(generate_candidates(n=8, tokens=64, d=7168))
    assert s["infeasible"] > 0
    joined = " ".join(s["rejections_by_reason"])
    assert "UB overflow" in joined or "VF register" in joined
    assert UB_BYTES == 262144


def test_a_larger_d_prunes_at_least_as_hard():
    """Monotonicity: the feasible set cannot grow as the binding resource grows."""
    from tilefoundry_ascend.attnres.resources import generate_candidates, summarise
    small = summarise(generate_candidates(n=4, tokens=64, d=1024))
    big = summarise(generate_candidates(n=4, tokens=64, d=7168))
    assert big["feasible"] <= small["feasible"]
    assert big["pruned_fraction"] >= small["pruned_fraction"]


def test_resident_n_prunes_more_than_online_at_the_same_shape():
    from tilefoundry_ascend.attnres.resources import generate_candidates
    cands = generate_candidates(n=32, tokens=64, d=4096)
    on = [c for c in cands if c.family == ONLINE]
    rn = [c for c in cands if c.family == RESIDENT_N]
    assert sum(c.feasible for c in rn) < sum(c.feasible for c in on)


def test_unmeasured_candidates_are_labelled_unmeasured():
    """A feasible candidate is not a result; conflating them is how a report overstates."""
    from tilefoundry_ascend.attnres.resources import (
        generate_candidates, MEASURED, UNMEASURED)
    cands = generate_candidates(n=4, tokens=32, d=512)
    assert any(c.evidence == MEASURED for c in cands)
    assert any(c.evidence == UNMEASURED for c in cands)
    for c in cands:
        if c.token_rows > 1 or c.v_versions > 1:
            assert c.evidence == UNMEASURED


def test_ub_accounting_is_itemised():
    """A rejection has to name its cause, so the accounting cannot be a single opaque number."""
    from tilefoundry_ascend.attnres.resources import measure_resources
    spec = build_attention_residual(n=8, tokens=64, d=2047)
    use = measure_resources(spec)
    assert use.ub_bytes == sum(use.items.values())
    assert "g_ub" in use.items and "acc_ub" in use.items
    assert sum(1 for v in use.items.values() if v > 0) >= 6


# ------------------------------------------------------------------ evidence

def test_the_g52_record_covers_all_three_families_and_both_modes():
    here = Path(__file__).resolve().parents[1]
    data = json.loads((here / "evidence" / "G52_DEVICE_EVIDENCE.json").read_text())
    configs = " ".join(r["config"] for r in data["official_20_case_sweep"]["results"])
    assert "online" in configs
    assert "blocked_online" in configs
    assert "resident_n" in configs
    assert "literal_fp32" in configs
    assert "factored_gw" in configs
    for r in data["official_20_case_sweep"]["results"]:
        assert r["passed"] == r["total"] == 20


def test_the_g52_record_states_the_open_niter_issue():
    """An open defect must be recorded where a reader of the gate will see it."""
    here = Path(__file__).resolve().parents[1]
    data = json.loads((here / "evidence" / "G52_DEVICE_EVIDENCE.json").read_text())
    issue = data["niter_gt_1_investigation"]
    assert issue["status"].startswith("OPEN")
    assert len(issue["hypotheses_tested_and_REJECTED"]) >= 5
    assert "second reducer" in issue["established_by_bisection"].lower()


def test_the_g52_record_does_not_claim_performance():
    here = Path(__file__).resolve().parents[1]
    data = json.loads((here / "evidence" / "G52_DEVICE_EVIDENCE.json").read_text())
    joined = " ".join(data["not_claimed"]).lower()
    assert "performance" in joined
    assert "persistent" in joined


def test_score_modes_are_recorded_as_non_identical():
    here = Path(__file__).resolve().parents[1]
    data = json.loads((here / "evidence" / "G52_DEVICE_EVIDENCE.json").read_text())
    assert "NOT bit-identical" in data["score_modes"]["distinction"]

# ------------------------------------------------------------------ CN>1 emitted kernel

def test_the_blocked_twin_matches_for_both_families():
    from tilefoundry_ascend.attnres.twin import (
        build_attnres_twin_cn, check_attnres_conformance_cn, parse_attnres_twin, MATCHED)
    for family in (BLOCKED_ONLINE, RESIDENT_N):
        spec = build_attention_residual(n=8, tokens=8, d=512, family=family)
        report = check_attnres_conformance_cn(
            spec, parse_attnres_twin(build_attnres_twin_cn(spec)))
        assert report.status == MATCHED, (family, report.failures)


def test_the_blocked_twin_parses_as_python():
    import ast
    from tilefoundry_ascend.attnres.twin import build_attnres_twin_cn
    spec = build_attention_residual(n=8, tokens=8, d=512, family=BLOCKED_ONLINE)
    ast.parse(build_attnres_twin_cn(spec))


def test_the_blocked_twin_keeps_block_state_separate():
    """Folding the block into the running acc would make the merge meaningless."""
    from tilefoundry_ascend.attnres.twin import build_attnres_twin_cn
    spec = build_attention_residual(n=8, tokens=8, d=512, family=BLOCKED_ONLINE)
    src = build_attnres_twin_cn(spec)
    assert "u_ub" in src
    assert "st[7]" in src and "st[8]" in src


def test_a_blocked_twin_without_the_second_pass_is_refused():
    """Re-reading V is what the blocked form costs; dropping it silently means holding the
    block, which is resident_n's cost and not blocked_online's."""
    from tilefoundry_ascend.attnres.twin import (
        build_attnres_twin_cn, check_attnres_conformance_cn, parse_attnres_twin, FAILED)
    spec = build_attention_residual(n=8, tokens=8, d=512, family=BLOCKED_ONLINE)
    src = build_attnres_twin_cn(spec).replace("for j in range(CN):", "for j in range(1):")
    report = check_attnres_conformance_cn(spec, parse_attnres_twin(src))
    assert report.status == FAILED
    assert any("TWIN_MISSING_SECOND_PASS" in f for f in report.failures)


def test_the_cn_one_twin_still_matches_untouched():
    """The generalisation must not perturb the kernel that G50 actually measured."""
    from tilefoundry_ascend.attnres.twin import (
        build_attnres_twin, check_attnres_conformance, parse_attnres_twin, MATCHED)
    spec = build_attention_residual(n=4, tokens=32, d=512)
    report = check_attnres_conformance(spec, parse_attnres_twin(build_attnres_twin(spec)))
    assert report.status == MATCHED, report.failures


def test_the_hir_models_one_task_pair_per_v_block_not_per_cn_group():
    """Modelling one task per CN group would hide the second V read the blocked form performs."""
    from tilefoundry_ascend.attnres import build_scheduled_hir
    for family, n in ((BLOCKED_ONLINE, 8), (RESIDENT_N, 8)):
        spec = build_attention_residual(n=n, tokens=4, d=64, family=family)
        hir = build_scheduled_hir(spec)
        loads = sum(1 for t in hir.tasks if t.name == "LOAD_V")
        assert loads == spec.n * 4, (family, loads)
        assert hir.metadata["block_n"] == spec.block_n
        assert hir.metadata["v_reads_per_element"] == 2
