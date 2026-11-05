#!/usr/bin/env python3
"""CPU-only negative controls for the vcvt probe harness data path.

Every defect this file pins down was REAL: a KeyError that killed a narrowing case
after it had already compiled, and a readback that compared 16-bit codewords against
32-bit float bits. None of these tests touches a device; they fail on the data path.

Run: python -B tools/p3/native_probe/test_case_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import probe_base  # noqa: E402
from probe_base import CaseDataError  # noqa: E402

PASS: list = []
FAIL: list = []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
    except Exception as exc:  # noqa: BLE001
        FAIL.append("%s: %s: %s" % (name, type(exc).__name__, exc))


def want_raises(fn, needle):
    try:
        fn()
    except CaseDataError as exc:
        assert needle in str(exc), "wrong message: %s" % exc
        return
    raise AssertionError("expected CaseDataError containing %r, none raised" % needle)


def case(**kw):
    base = {"name": "synthetic"}
    base.update(kw)
    return base


# --- the actual regression: a narrowing case must not read the widening key -----

def test_narrowing_case_selects_float32():
    c = probe_base.load_case("vcvt_f32_to_bf16_rne_ftz")
    assert probe_base.source_dtype(c) == "float32", probe_base.source_dtype(c)
    assert probe_base.dest_dtype(c) == "bfloat16"
    assert len(probe_base.input_bits_of(c)) == 64


def test_widening_case_selects_bfloat16():
    c = probe_base.load_case("vcvt_bf16_to_f32_part0")
    assert probe_base.source_dtype(c) == "bfloat16"
    assert probe_base.dest_dtype(c) == "float32"


def test_every_vcvt_case_resolves_its_own_source():
    for name in ("vcvt_bf16_to_f32_part0", "vcvt_f32_to_bf16_rne_ftz",
                 "vcvt_inactive_lane_preservation"):
        c = probe_base.load_case(name)
        bits = probe_base.input_bits_of(c)
        width = 4 if probe_base.source_dtype(c) == "bfloat16" else 8
        assert len(bits) == 64, name
        assert all(len(b) == 2 + width for b in bits), name


# --- NEGATIVE: missing key must be NAMED, not a bare KeyError -------------------

def test_missing_source_key_is_named():
    c = case(source_dtype="bfloat16", input_bits=["0x00000000"] * 64)
    want_raises(lambda: probe_base.input_bits_of(c), "must carry")


def test_missing_source_key_names_the_field():
    c = case(source_dtype="float32")
    try:
        probe_base.input_bits_of(c)
    except CaseDataError as exc:
        assert "input_bits" in str(exc), exc
        return
    raise AssertionError("no CaseDataError")


# --- NEGATIVE: the OLD behaviour, reproduced, must now be refused ---------------

def test_old_widening_branch_would_have_raised_keyerror():
    """The exact defect: reading input_bf16_bits on a narrowing case."""
    c = probe_base.load_case("vcvt_f32_to_bf16_rne_ftz")
    try:
        c["input_bf16_bits"]
    except KeyError:
        pass  # this is what used to happen, after compile, on the device
    else:
        raise AssertionError("the narrowing case unexpectedly carries the widening key")
    # and the new path does NOT touch that key:
    assert probe_base.input_bits_of(c)


# --- NEGATIVE: wrong bit WIDTH must be refused, not reinterpreted --------------

def test_wrong_width_source_is_refused():
    c = case(source_dtype="float32", input_bits=["0x3F80"] * 64)   # bf16 width, f32 claimed
    want_raises(lambda: probe_base.input_bits_of(c), "not a 0x08-wide")


def test_wrong_width_source_is_refused_bf16():
    c = case(source_dtype="bfloat16", input_bf16_bits=["0x3F800000"] * 64)
    want_raises(lambda: probe_base.input_bits_of(c), "not a 0x04-wide")


def test_wrong_width_prefill_is_refused():
    c = case(source_dtype="float32", dest_dtype="bfloat16",
             input_bits=["0x3F800000"] * 64,
             dst_prefill_bits=["0x00000000"] * 64)          # f32 width, bf16 claimed
    want_raises(lambda: probe_base.prefilled_bits_of(c), "not a 0x04-wide")


def test_non_hex_is_refused():
    c = case(source_dtype="float32", input_bits=["3F800000"] * 64)
    want_raises(lambda: probe_base.input_bits_of(c), "not a 0x08-wide")


# --- NEGATIVE: unknown dtype must be refused, not defaulted --------------------

def test_unknown_source_dtype_is_refused():
    c = case(source_dtype="float16", input_bits=["0x0000"] * 64)
    want_raises(lambda: probe_base.source_dtype(c), "unsupported source_dtype")


# --- the bf16 readback defect, pinned ------------------------------------------

def test_observed_bits_width_follows_the_tensor():
    src = Path(probe_base.__file__).read_text()
    assert "view(np.uint16)" in src, "bf16 output must be read as 16-bit codewords"
    assert "to(torch.float32).numpy()).view(np.uint32)" not in src, \
        "the old blanket float32 cast is still present"


def test_observed_bits_refuses_unknown_dtype():
    src = Path(probe_base.__file__).read_text()
    assert "refusing to reinterpret it" in src


# --- the predicted layout must match the OBSERVED part=0 slot rule -------------

def test_predicted_rne_uses_even_slot_layout():
    c = probe_base.load_case("vcvt_f32_to_bf16_rne_ftz")
    pred = c["predicted"]["rne_bf16_even_slots"]
    assert len(pred) == 64
    for slot in range(1, 64, 2):
        assert pred[slot] == "0x0000", "odd slot %d must be the preserved prefill" % slot
    assert any(pred[s] != "0x0000" for s in range(0, 64, 2)), "no even slot written"


def test_predicted_layout_declares_odd_slot_rule():
    c = probe_base.load_case("vcvt_f32_to_bf16_rne_ftz")
    assert c["layout"]["odd_slots"].startswith("keep the prefill")
    assert "OBSERVED" in c["layout"]["basis"]


def test_contiguous_prediction_is_kept_as_reference_only():
    c = probe_base.load_case("vcvt_f32_to_bf16_rne_ftz")
    assert "rne_bf16_contiguous_reference" in c["predicted"]
    assert "rne_bf16_even_slots" in c["predicted"]
    assert c["predicted"]["rne_bf16_even_slots"] != c["predicted"]["rne_bf16_contiguous_reference"]


# --- the FTZ conclusion must be an observation, never an inference -------------

def test_ftz_is_not_asserted_from_the_case_name():
    c = probe_base.load_case("vcvt_f32_to_bf16_rne_ftz")
    note = c["predicted"]["note"]
    assert "never assumed" in note, note
    assert "NAME says ftz and is not evidence" in note, note


# --- the narrowing kernel must exist and be selected by the case ---------------

def test_narrowing_kernel_exists():
    import vcvt_probe
    assert hasattr(vcvt_probe, "build_kernel_narrowing")
    src = Path(vcvt_probe.__file__).read_text()
    assert '"bfloat16", mask16, part=0' in src, "narrowing kernel must convert TO bfloat16"


def test_kernel_selected_by_case_direction():
    import inspect
    import vcvt_probe
    sig = inspect.signature(vcvt_probe.build_kernel)
    assert "case" in sig.parameters, "build_kernel must accept the case"
    src = Path(vcvt_probe.__file__).read_text()
    assert "source_dtype" in src, "kernel choice must key off source_dtype"


def test_run_probe_passes_the_case_to_multi_arg_builders():
    src = Path(probe_base.__file__).read_text()
    assert "build_from_case" in src
    assert "len(params) >= 2" in src, "single-arg builders (vcadd/vmula) must keep working"


# --- comparison must be a real per-lane verdict, not a description -------------

def test_compare_returns_per_lane_verdict_for_16bit_predictions():
    """"compare via the bf16 codeword" is not a comparison. This pins that the
    narrowing prediction is actually compared lane by lane."""
    c = probe_base.load_case("vcvt_f32_to_bf16_rne_ftz")
    want = c["predicted"]["rne_bf16_even_slots"]
    verdict = probe_base.compare(c, list(want))
    vm = verdict["model_matches"]["rne_bf16_even_slots"]
    assert vm.get("comparable") is True, vm
    assert vm["exact_all_lanes"] is True, vm
    assert vm["lanes_compared"] == 64
    assert vm["mismatched_lanes"] == []


def test_compare_reports_mismatches_not_just_a_note():
    c = probe_base.load_case("vcvt_f32_to_bf16_rne_ftz")
    want = list(c["predicted"]["rne_bf16_even_slots"])
    obs = list(want)
    obs[0] = "0xDEAD"                      # corrupt exactly one lane
    vm = probe_base.compare(c, obs)["model_matches"]["rne_bf16_even_slots"]
    assert vm["exact_all_lanes"] is False
    assert vm["mismatched_lanes"] == [0]
    assert vm["mismatched_predicted"] == [want[0]]
    assert vm["lanes_matching"] == 63


def test_compare_refuses_width_mismatch_instead_of_pretending():
    c = probe_base.load_case("vcvt_f32_to_bf16_rne_ftz")
    obs32 = ["0x00000000"] * 64         # 32-bit words against a 16-bit prediction
    vm = probe_base.compare(c, obs32)["model_matches"]["rne_bf16_even_slots"]
    assert vm["comparable"] is False
    assert "readback width does not match" in vm["why"]


def test_even_odd_slot_check_is_computed():
    c = probe_base.load_case("vcvt_f32_to_bf16_rne_ftz")
    want = c["predicted"]["rne_bf16_even_slots"]
    chk = probe_base.compare(c, list(want))["even_odd_slot_check"]
    assert chk["odd_slots_all_zero"] is True
    assert chk["consistent_with_part0"] is True
    assert chk["even_slots_written"] == 28, chk["even_slots_written"]


def test_even_odd_slot_check_detects_a_written_odd_slot():
    c = probe_base.load_case("vcvt_f32_to_bf16_rne_ftz")
    obs = list(c["predicted"]["rne_bf16_even_slots"])
    obs[1] = "0x3F80"                    # part=0 must never write an odd slot
    chk = probe_base.compare(c, obs)["even_odd_slot_check"]
    assert chk["odd_slots_all_zero"] is False
    assert chk["consistent_with_part0"] is False


def test_ftz_verdict_is_computed_from_observation():
    c = probe_base.load_case("vcvt_f32_to_bf16_rne_ftz")
    want = list(c["predicted"]["rne_bf16_even_slots"])
    ftz = probe_base.compare(c, want)["ftz_daz_observation"]
    assert ftz["all_flushed"] is False, ftz
    assert ftz["none_flushed"] is True
    assert "FTZ/DAZ INACTIVE" in ftz["verdict"]
    assert "NOT a precision-protocol failure" in ftz["recorded_as"]
    assert "is not evidence" in ftz["not_derived_from"]


def test_ftz_verdict_would_flip_if_the_subnormal_were_flushed():
    c = probe_base.load_case("vcvt_f32_to_bf16_rne_ftz")
    obs = list(c["predicted"]["rne_bf16_even_slots"])
    obs[8] = "0x0000"                    # the decisive subnormal, flushed
    ftz = probe_base.compare(c, obs)["ftz_daz_observation"]
    assert ftz["all_flushed"] is True
    assert "FTZ/DAZ ACTIVE" in ftz["verdict"]


def test_subnormal_probe_uses_a_decisive_input_only():
    c = probe_base.load_case("vcvt_f32_to_bf16_rne_ftz")
    probes = c["subnormal_probe"]
    assert [p["src_f32_bits"] for p in probes] == ["0x00400000"], (
        "0x00000001 is below the smallest bf16 subnormal and rounds to 0x0000 under "
        "plain RNE, so it cannot distinguish FTZ from RNE")
    assert probes[0]["if_not_flushed"] == "0x0040"


def test_mask_gap_is_declared_not_hidden():
    """vcvt_inactive_lane_preservation declares a partial mask the kernel ignores.

    Until the mask is plumbed in, that case cannot support any inactive-lane claim.
    The gap is pinned so it cannot be quietly forgotten or cited as evidence.
    """
    import vcvt_probe
    assert hasattr(vcvt_probe, "MASK_NOT_PLUMBED")
    assert "vcvt_inactive_lane_preservation" in vcvt_probe.MASK_NOT_PLUMBED
    src = Path(vcvt_probe.__file__).read_text()
    assert "KNOWN OPEN GAP" in src
    print("    NOTE: vcvt_inactive_lane_preservation mask is NOT plumbed in; "
          "its inactive-lane claim is unexercised")


def test_case1_nan_payload_mismatch_is_expected_not_silent():
    """The one real case-1 mismatch is the NaN payload, and it must stay visible."""
    c = probe_base.load_case("vcvt_bf16_to_f32_part0")
    want = list(c["predicted"]["part0_widen"])
    obs = list(want)
    obs[5] = "0x7FFFFFFF"          # what device 1 actually returned for bf16 NaN 0x7FC0
    vm = probe_base.compare(c, obs)["model_matches"]["part0_widen"]
    assert vm["mismatched_lanes"] == [5]
    assert vm["mismatched_predicted"] == ["0x7FC00000"]
    assert vm["lanes_matching"] == 63


def test_nan_observation_is_ruled_an_observation_not_a_protocol_failure():
    """Ruling: bf16 NaN payload is not preserved. Recorded, NOT judged a failure.

    Kept in the artifact so the distinction survives context loss: 63/64 is NOT a
    pass, and it is NOT a frozen-protocol failure either. Any tightening needs an
    architect ruling.
    """
    c = probe_base.load_case("vcvt_bf16_to_f32_part0")
    obs = c.get("native_observations", {})
    nan = obs.get("nan_payload_not_preserved")
    assert nan is not None, "the NaN observation must be recorded in the case artifact"
    assert nan["device_observed"] == "0x7FFFFFFF"
    assert nan["model_predicted"] == "0x7FC00000"
    assert nan["counts_as_pass"] is False
    assert nan["counts_as_protocol_failure"] is False
    assert "architect" in nan["ruling"].lower()


def test_expected_bits_sha_matches_readme():
    """The README pins the artifact hash; a regenerated artifact must not leave it stale."""
    import hashlib
    here = Path(probe_base.__file__).resolve().parent
    actual = hashlib.sha256((here / "expected_bits.json").read_bytes()).hexdigest()
    readme = (here / "README.md").read_text()
    assert actual in readme, (
        "tools/p3/native_probe/README.md does not pin the current expected_bits.json "
        "sha256 %s" % actual)


TESTS = [(k, v) for k, v in sorted(globals().items())
         if k.startswith("test_") and callable(v)]


def main() -> int:
    for name, fn in TESTS:
        check(name, fn)
    print("checks:", len(TESTS), "passed:", len(PASS), "failed:", len(FAIL))
    for line in FAIL:
        print("  FAIL", line)
    print("RESULT:", "PASS" if not FAIL else "FAIL")
    return 0 if not FAIL else 1


if __name__ == "__main__":
    raise SystemExit(main())