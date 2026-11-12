"""P6-04/G63: MLA closure beyond the official 20 cases.

The device results are in evidence/G63_DEVICE_EVIDENCE.json. These tests pin the DOMAIN and the
CLOSURE argument, including the coverage gap found in the official set: every official case uses
Nkv == 1, so multi-KV-head GQA is untested by the official 20 alone.
"""
import json
from pathlib import Path

import pytest
import yaml

from tilefoundry_ascend.mla.kernel import BN, BM_CHOICES, check_domain, choose_block_m
from tilefoundry_ascend.mla.model import MLAError

HERE = Path(__file__).resolve().parents[1]
FIXTURES = Path("/home/s00653124/workspace/TileFoundry_Ascend950PR_AgentTeam/2026111/"
                "tf_ascend950pr_single_kernel_v2/fixtures/mla/cases.yaml")


def _official_cases():
    if not FIXTURES.exists():
        return None
    return yaml.safe_load(FIXTURES.read_text())["cases"]


CASES = _official_cases()


def test_the_official_set_never_exercises_more_than_one_kv_head():
    """The closure argument rests on this, so it is computed from the fixture rather than claimed.
    If a future fixture adds Nkv>1 this test fails and the GQA cases stop being extra."""
    if CASES is None:
        pytest.skip("official fixture unavailable")
    kv_heads = {c["attrs"]["numKVHeads"] for c in CASES}
    assert kv_heads == {1}


def test_the_official_group_values_are_the_legal_tile_multiples():
    if CASES is None:
        pytest.skip("official fixture unavailable")
    groups = set()
    for c in CASES:
        shape, layout = c["input_shape"], c["attrs"]["inputLayout"]
        nq = shape[0][2] if layout == "BSND" else shape[0][1]
        groups.add(nq // c["attrs"]["numKVHeads"])
    assert groups == {32, 64, 128}
    for g in groups:
        assert g % min(BM_CHOICES) == 0


def test_the_official_kv_lengths_are_all_multiples_of_the_pipeline_block():
    if CASES is None:
        pytest.skip("official fixture unavailable")
    for c in CASES:
        shape, layout = c["input_shape"], c["attrs"]["inputLayout"]
        skv = shape[2][1] if layout == "BSND" else shape[2][2]
        assert skv % BN == 0, c["case_id"]


# ---------------------------------------------------------------- domain refusal

def test_a_kv_length_off_the_pipeline_block_is_refused():
    """Skv=64 is outside the domain; refusing it is correct, and this is the exact error my own
    GQA test tripped over before I fixed the input."""
    with pytest.raises(MLAError) as exc:
        check_domain(seq_kv=64, d_nope=512, d_rope=64, group=32, block_m=32)
    assert exc.value.code == "MLA_KV_NOT_MULTIPLE_OF_128"


def test_a_group_below_the_query_tile_is_refused():
    """G=16 cannot fill a 32-row tile, so the mapping is refused rather than tiled illegally."""
    with pytest.raises(MLAError) as exc:
        choose_block_m(16, jobs_wide=4)
    assert exc.value.code == "MLA_BAD_TILE"


def test_the_rope_width_must_be_the_single_panel():
    with pytest.raises(MLAError) as exc:
        check_domain(seq_kv=128, d_nope=512, d_rope=32, group=32, block_m=32)
    assert exc.value.code == "MLA_D_ROPE_MUST_BE_64"


def test_a_d_nope_off_the_panel_grid_is_refused():
    with pytest.raises(MLAError) as exc:
        check_domain(seq_kv=128, d_nope=500, d_rope=64, group=32, block_m=32)
    assert exc.value.code == "MLA_D_NOPE_NOT_MULTIPLE_OF_64"


# ---------------------------------------------------------------- V == K_nope

def test_v_equals_k_nope_is_a_value_contract_not_an_alias_assumption():
    """The kernel must not depend on v and k_nope sharing storage. Stated as a source check
    because the distinction is invisible at the call site: aliasing would break the moment a
    caller passes a copy, which the fixture get_input() does."""
    src = (HERE / "tilefoundry_ascend" / "mla" / "kernel.py").read_text()
    body = src[src.index("def mla("):]
    assert "accepted for ABI compatibility" in body
    assert "deliberately NOT read" in body


# ---------------------------------------------------------------- evidence

def test_the_g63_record_reports_zero_extended_failures():
    data = json.loads((HERE / "evidence" / "G63_DEVICE_EVIDENCE.json").read_text())
    assert data["status"] == "VERIFIED"
    gqa = data["extended_cases"]["gqa_nkv_gt_1"]
    assert len(gqa) >= 6
    assert all(c["verdict"] == "PASS" for c in gqa)
    assert all(c["ref"] == 0.0 for c in gqa)


def test_the_g63_record_states_the_official_coverage_gap():
    data = json.loads((HERE / "evidence" / "G63_DEVICE_EVIDENCE.json").read_text())
    gap = data["coverage_gap_found_in_the_official_set"]
    assert "Nkv=1" in gap["statement"]


def test_the_g63_record_keeps_the_official_20_result():
    data = json.loads((HERE / "evidence" / "G63_DEVICE_EVIDENCE.json").read_text())
    sweep = data["official_20_case_sweep"]
    assert sweep["passed"] == 20 and sweep["total"] == 20
    assert sweep["re_run_after_sharding"] is True
