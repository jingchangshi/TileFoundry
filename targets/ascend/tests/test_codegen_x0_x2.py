"""X0-X2 tests: mode/Plan preservation, source/ABI identity split, UNIMPLEMENTED.

CPU only; no device, no real TL compile.
Run: PYTHONPATH=src:targets/ascend python targets/ascend/tests/test_codegen_x0_x2.py
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "targets" / "ascend"))
sys.path.insert(0, str(REPO / "targets" / "ascend" / "tests"))

from tilefoundry_ascend.codegen import x_runtime as X
from tilefoundry_ascend import sync as C
from tilefoundry_ascend.plan import resolved_plan as RP
import test_plan_r4_r5 as R4


def applied_plan():
    plan = R4.build_stage()
    return dataclasses.replace(plan, state=dataclasses.replace(
        plan.state, application_status=RP.STATE_APPLIED))


def verified_c():
    return C.CoverageReport("VERIFIED", None, 1, 1, 1, (), 7)


def source_map(preserved=True, with_artifacts=False):
    result = {"statements": {"s0": "copy"}, "plan_preserved": preserved}
    if with_artifacts:
        result["cce"] = {"status": "OBSERVED", "sha256": "c" * 64}
        result["binary"] = {"status": "OBSERVED", "sha256": "b" * 64}
        result["load"] = {"status": "OBSERVED", "handle": "callable"}
    return result


def abi():
    return {"args": [{"name": "x", "kind": "in", "dtype": "f32", "shape": [256], "nbytes": 1024},
                     {"name": "out", "kind": "out", "dtype": "f32", "shape": [256], "nbytes": 1024}],
            "output_index": 1, "eps_f32_bits": None}


def request(mode=X.LOGICAL_TL_RECOMPILE, preserved=True):
    return X.validate_adapter_request(object(), applied_plan(), verified_c(), {"env": "v2"},
                                      mode, source_map=source_map(preserved), abi=abi())


def expect_code(code, fn):
    try:
        fn()
    except Exception as error:  # noqa: BLE001
        got = getattr(error, "code", None)
        if got != code:
            raise AssertionError("expected %s, got %r: %s" % (code, got, error))
        return error
    raise AssertionError("expected %s, but nothing was refused" % code)


def test_x0_modes_and_verified_inputs_are_explicit():
    req = request(X.LOGICAL_TL_RECOMPILE)
    assert req.mode == "LOGICAL_TL_RECOMPILE" and req.source_identity.tir_digest
    preserve = request(X.RESOLVED_PLAN_PRESERVING)
    assert preserve.mode == "RESOLVED_PLAN_PRESERVING"
    expect_code("X_MODE_INVALID", lambda: X.validate_adapter_request(
        object(), applied_plan(), verified_c(), {"env": "v2"}, "UNKNOWN"))
    expect_code("X_PLAN_NOT_READY", lambda: X.validate_adapter_request(
        object(), R4.build_stage(), verified_c(), {"env": "v2"}, X.LOGICAL_TL_RECOMPILE))
    expect_code("COMPLETION_NOT_VERIFIED", lambda: X.validate_adapter_request(
        object(), applied_plan(), C.CoverageReport("FAILED", "x", 1, 1, 0, ("ob",), 0),
        {"env": "v2"}, X.LOGICAL_TL_RECOMPILE))


def test_x1_source_map_and_abi_are_required():
    expect_code("X_SOURCE_COVERAGE_MISMATCH", lambda: X.validate_adapter_request(
        object(), applied_plan(), verified_c(), {"env": "v2"}, X.LOGICAL_TL_RECOMPILE,
        source_map={}, abi=abi()))
    req = request()
    expect_code("X_SOURCE_COVERAGE_MISMATCH", lambda: X.lower_to_tilelang(req, object(), {}))
    module = X.lower_to_tilelang(req, object(), source_map())
    assert module.source_status == "SOURCE_MAPPED" and module.plan_preserved is True


def test_x1_plan_preservation_mismatch_is_refused():
    req = request(X.RESOLVED_PLAN_PRESERVING, preserved=True)
    expect_code("X_PLAN_PRESERVATION_MISMATCH",
                lambda: X.lower_to_tilelang(req, object(), source_map(preserved=False)))
    module = X.lower_to_tilelang(req, object(), source_map(preserved=True))
    expect_code("X_PLAN_PRESERVATION_MISMATCH",
                lambda: X.verify_adapter_correspondence(
                    req, dataclasses.replace(module, plan_preserved=False), source_map(), X.KernelABI(abi()["args"], 1, None)))


def test_x2_correspondence_reports_source_mapped_but_not_pass_without_identities():
    req = request()
    module = X.lower_to_tilelang(req, object(), source_map())
    result = X.verify_adapter_correspondence(req, module, source_map(), X.KernelABI(abi()["args"], 1, None))
    assert result["status"] == "SOURCE_MAPPED"
    assert result["identities"]["source"] == "MAPPED"
    assert set(result["missing_execution_identities"]) == {"cce", "binary", "load"}
    expect_code("X_ARTIFACT_IDENTITY_MISSING",
                lambda: X.verify_adapter_correspondence(
                    req, module, source_map(), X.KernelABI(abi()["args"], 1, None), claim_pass=True))
    good = X.verify_adapter_correspondence(req, module, source_map(with_artifacts=True),
                                           X.KernelABI(abi()["args"], 1, None), claim_pass=True)
    assert good["status"] == "SOURCE_MAPPED" and not good["missing_execution_identities"]


def test_x2_unimplemented_compile_bind_invoke_are_named():
    expect_code("CODEGEN_CONTRACT_UNIMPLEMENTED", lambda: X.compile_tilelang(None, None, None, None))
    expect_code("CODEGEN_CONTRACT_UNIMPLEMENTED", lambda: X.bind_kernel(None, None, None, None))
    expect_code("CODEGEN_CONTRACT_UNIMPLEMENTED", lambda: X.invoke_kernel(None, None))
    expect_code("CODEGEN_CONTRACT_UNIMPLEMENTED", lambda: X.synchronize_and_readback(None, None))


def test_x_emit_boundary_requires_r5_verified_and_stays_unimplemented():
    plan = applied_plan()
    expect_code("COMPLETION_NOT_VERIFIED", lambda: X.emit_for_codegen(plan))
    attached = RP.attach_completion(plan, verified_c())
    expect_code("CODEGEN_CONTRACT_UNIMPLEMENTED", lambda: X.emit_for_codegen(attached))


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
