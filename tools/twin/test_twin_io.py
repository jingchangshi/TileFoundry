"""T0-T4 CPU tests: twin-io/1 ABI, three roles, positives and DECISIVE bad controls.

Run with the CPU env (torch must not autoload the NPU backend):
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 python tools/twin/test_twin_io.py
or with pytest in an interpreter that has numpy+torch+pytest.
"""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

import numpy as np

try:
    import pytest
except ImportError:
    class _MiniPytest:
        @staticmethod
        def importorskip(name):
            import importlib
            return importlib.import_module(name)
    pytest = _MiniPytest()

import reference as R
import twin_io as T

HERE = Path(__file__).resolve().parent
BUNDLES = json.loads((HERE / "fixtures" / "bundles.json").read_text())
STORAGE = HERE / "fixtures" / "actual"
EPS_BITS = T.eps_f32_bits(R.EPS)


def bundle(fixture_id):
    return copy.deepcopy(BUNDLES["bundles"][fixture_id])


def load(fixture_id, expected_eps_bits=EPS_BITS):
    item = bundle(fixture_id)
    return T.load_actual_inputs(
        item, {"case_id": item["case_id"], "input_id": item["input_id"],
               "eps_bits": expected_eps_bits},
        base_dir=STORAGE / item["storage_dir"])


def positive(fixture_id):
    loaded = load(fixture_id)
    operation = "vadd_one" if "VADD" in fixture_id else "rmsnorm"
    ref = R.compute_reference(loaded, operation)
    algo = R.replay_logical({"entry": "t." + operation, "operation": operation}, loaded)
    return loaded, ref, algo, T.compare_outputs(
        algo["values"], ref["values"], R.PROTOCOL, dtype=ref["dtype"])


# ------------------------------------------------------------- positive cases
def test_f32_vadd_one_positive_passes():
    loaded, ref, algo, report = positive("T-FIX-VADD-001")
    assert report["status"] == "PASS", report
    assert report["mismatch_count"] == 0 and report["n_elements"] == R.VADD_N
    assert ref["dtype"] == "float32"
    expected = (np.arange(R.VADD_N, dtype=np.float32) + np.float32(0.5) + np.float32(1.0))
    assert np.array_equal(ref["values"], expected)


def test_single_row_rmsnorm_all_patterns_positive_pass():
    for fixture in BUNDLES["bundles"]:
        if "RMS" not in fixture:
            continue
        loaded, ref, algo, report = positive(fixture)
        assert ref["dtype"] == "bfloat16"
        assert report["status"] == "PASS", (fixture, report)
        assert report["mismatch_count"] == 0
        # The independent reference is FP64 -> ONE bf16 rounding.
        exact, bits = R.reference_rmsnorm_bf16_once(
            loaded["arrays"]["x"], loaded["arrays"]["w"], loaded["eps_f32"])
        assert np.array_equal(bits, ref["values"])
        assert not R.reference_quantization_mismatch(exact, bits)


def test_actual_input_roundtrip_is_byte_exact_and_eps_is_f32_bits():
    item = bundle("T-FIX-RMS-ROW-001")
    assert item["eps"]["bits"] == EPS_BITS
    assert item["eps"]["hex"] == "0x%08x" % EPS_BITS
    loaded = load("T-FIX-RMS-ROW-001")
    record = item["tensors"][0]
    raw = (STORAGE / item["storage_dir"] / record["storage_path"]).read_bytes()
    assert T.sha256_bytes(raw) == record["sha256"]
    assert loaded["arrays"]["x"].dtype == np.uint16  # real bf16 bits, not f32


# ------------------------------------------------- decisive bad controls (1-4)
def test_bad_control_wrong_one_lane_is_numeric_mismatch():
    loaded = load("T-FIX-RMS-ROW-001")   # zero pattern: reference is all zero
    _, ref = R.reference_rmsnorm_bf16_once(
        loaded["arrays"]["x"], loaded["arrays"]["w"], loaded["eps_f32"])
    got = ref.copy()
    got[0, 0] = R.bf16_bits_from_f32_cast(np.array([1.0], np.float32))[0]  # one lane off
    report = T.compare_outputs(got, ref, R.PROTOCOL, dtype="bfloat16")
    assert report["status"] == "FAIL"
    assert report["code"] == "NUMERIC_MISMATCH"
    assert report["mismatch_count"] == 1
    assert report["mismatch_coords"] == [[0, 0]]


def test_bad_control_input_hash_mismatch_is_rejected():
    item = bundle("T-FIX-RMS-ROW-001")
    item["tensors"][0]["sha256"] = "0" * 64
    try:
        T.load_actual_inputs(item, {"eps_bits": EPS_BITS},
                             base_dir=STORAGE / item["storage_dir"])
    except T.TwinError as error:
        assert error.code == "INPUT_DIGEST_MISMATCH", error.as_dict()
    else:
        raise AssertionError("a corrupted input hash was accepted")


def test_bad_control_eps_bits_mismatch_is_rejected():
    item = bundle("T-FIX-RMS-ROW-001")
    try:
        T.load_actual_inputs(item, {"eps_bits": EPS_BITS ^ 1},
                             base_dir=STORAGE / item["storage_dir"])
    except T.TwinError as error:
        assert error.code == "EPS_BITS_MISMATCH", error.as_dict()
    else:
        raise AssertionError("a wrong eps f32 bit pattern was accepted")


def test_bad_control_double_rounding_reference_is_detected():
    # 1 + 2^-8 + 2^-30 sits just above a bf16 tie; f64->f32 rounds it DOWN to the
    # tie, and a second (bf16) rounding then picks the even value 1.0. One f64->bf16
    # rounding picks 1+2^-7. The two differ, and the double-rounded reference must
    # be refused.
    exact = np.float64(1.0) + 2.0 ** -8 + 2.0 ** -30
    single = R.bf16_bits_from_f64_once(np.array([exact]))[0]
    double = R.bf16_bits_from_f32_cast(np.array([np.float32(exact)], np.float32))[0]
    assert single != double, "test value no longer straddles the double-rounding boundary"
    assert R.reference_quantization_mismatch(np.array([exact]), np.array([double]))
    try:
        R.assert_single_round_reference(np.array([exact]), np.array([double]))
    except T.TwinError as error:
        assert error.code == "REFERENCE_QUANTIZATION_MISMATCH", error.as_dict()
    else:
        raise AssertionError("a double-rounded reference was accepted")


# --------------------------------------------- comparator ABI / nonfinite guards
def test_comparator_rejects_shape_mismatch():
    ref = np.zeros((1, 8), dtype=np.uint16)
    got = np.zeros((1, 7), dtype=np.uint16)
    report = T.compare_outputs(got, ref, R.PROTOCOL, dtype="bfloat16", shape=[1, 8])
    assert report["status"] == "FAIL" and report["code"] == "OUTPUT_ABI_MISMATCH"


def test_comparator_rejects_nonfinite():
    ref = np.array([[0x3F80, 0x0000]], dtype=np.uint16)   # 1.0, 0.0
    got = np.array([[0x7FC0, 0x0000]], dtype=np.uint16)   # NaN, 0.0
    report = T.compare_outputs(got, ref, R.PROTOCOL, dtype="bfloat16", shape=[1, 2])
    assert report["status"] == "FAIL" and report["code"] == "NONFINITE_OUTPUT"


# --------------------------------------------------------------- independence
def test_reference_module_is_independent_of_compiler_and_tir():
    source = (HERE / "reference.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            imported.add(node.module.split(".")[0])
    forbidden = {"tilefoundry", "tilelang", "tilefoundry_ascend", "evalref"}
    assert not (imported & forbidden), imported & forbidden
    lowered = source.lower()
    for token in ("identity_evaluator", "as_script", "@prim_func", "evalref"):
        assert token not in lowered, token


def test_twin_io_never_regenerates_an_input_from_seed():
    # A consumer must reject a bundle that lacks real bytes (no silent seed replay).
    item = bundle("T-FIX-RMS-ROW-001")
    item["tensors"][0]["storage_path"] = "does-not-exist.bin"
    try:
        T.load_actual_inputs(item, {"eps_bits": EPS_BITS},
                             base_dir=STORAGE / item["storage_dir"])
    except T.TwinError as error:
        assert error.code in ("INPUT_MISSING", "INPUT_DIGEST_MISMATCH"), error.as_dict()
    else:
        raise AssertionError("a bundle with no recorded bytes was accepted")


if __name__ == "__main__":
    failed = 0
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn()
                print("PASS", _name)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print("FAIL", _name, type(exc).__name__, exc)
    raise SystemExit(1 if failed else 0)
