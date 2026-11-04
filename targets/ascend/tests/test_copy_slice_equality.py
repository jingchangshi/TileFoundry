"""M2c: CPU semantic equality for the authored CopySlice HIR.

Runs the *authored* HIR (examples/rmsnorm_950pr/copy_slice.py:CopySlice) through
the real TF HIR evaluator (tilefoundry.evaluator.evaluate) on non-trivial CPU f32
data and asserts the region's GM->UB->GM Copy chain is value-transparent.

The check is made discriminating on purpose: a second, deliberately non-identity
authored program (examples/rmsnorm_950pr/mini_slice.py:MiniSlice, y = x + t with
t = x, i.e. y = 2x) is run through the *same* comparison and must be reported as
NOT equal.  A comparison that cannot fail is not evidence, so a non-discriminating
comparison is itself a FAIL.

What this proves: the authored program, executed by the TF reference
interpreter on the host, returns the input unchanged for several distinct
inputs, and does so without degenerating to a constant.

What this does NOT prove (stated so the record is not over-read):
  * physical output non-aliasing: the generic Copy value rule returns the source
    tensor, so a shared storage identity here is expected and is NOT a failure;
  * device execution: no NPU, no TIR compile;
  * register/mask semantics: CopySlice contains no SIMD instruction.

Exit codes are part of the evidence contract:
  0  PASS      equality holds for every probe input, and the check discriminates
  1  FAIL      a checked property is false
  2  NOT_RUN   torch (or the IR/evaluator) is unavailable; nothing was measured.
               This is not a pass: a caller must not read 2 as success.

Run (plain python, no pytest needed):
  PYTHONPATH=src:targets/ascend /path/to/python \
      targets/ascend/tests/test_copy_slice_equality.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

EXIT_PASS, EXIT_FAIL, EXIT_NOT_RUN = 0, 1, 2

REPO = Path(__file__).resolve().parents[3]
AUTHORED = REPO / "examples" / "rmsnorm_950pr" / "copy_slice.py"
AUTHORED_ENTRY = "CopySlice"
# Deliberately non-identity control program used to prove the check can fail.
CONTROL = REPO / "examples" / "rmsnorm_950pr" / "mini_slice.py"
CONTROL_ENTRY = "MiniSlice"
ELEMENT_COUNT = 64


def _probe_inputs():
    """Deterministic, non-degenerate f32 probes (distinct values, not all equal)."""
    import torch

    generator = torch.Generator().manual_seed(20261104)
    return {
        "randn_seeded": torch.randn(ELEMENT_COUNT, generator=generator, dtype=torch.float32),
        "arange_ramp": torch.arange(ELEMENT_COUNT, dtype=torch.float32),
        "signed_pattern": torch.linspace(-8.0, 8.0, ELEMENT_COUNT, dtype=torch.float32),
    }


def _load(path: Path, entry: str):
    from tilefoundry.cli.source import load_authored_ir

    return load_authored_ir(f"{path}:{entry}").entry_function()


def _compare(torch, evaluate, fn, inputs, expected) -> dict:
    """Evaluate fn on inputs and compare the result against expected."""
    try:
        result = evaluate(fn, *inputs)
    except Exception as exc:  # noqa: BLE001 - reported verbatim
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    data = result.data if hasattr(result, "data") and not isinstance(result, torch.Tensor) else result
    reference = expected
    detail = {
        "result_type": type(data).__name__,
        "result_shape": list(getattr(data, "shape", ())),
        "result_dtype": str(getattr(data, "dtype", None)),
        "equal_to_expected": bool(
            torch.is_tensor(data) and data.shape == reference.shape and torch.equal(data, reference)
        ),
        "degenerate_input": int(torch.unique(reference).numel()) < 2,
        "result_is_constant": bool(
            torch.is_tensor(data) and data.numel() > 1 and torch.unique(data).numel() == 1
        ),
        # Observation only, explicitly not a criterion: the generic Copy value
        # rule returns the source tensor, so a shared buffer is expected here.
        "shares_storage_with_expected": bool(
            torch.is_tensor(data) and data.data_ptr() == reference.data_ptr()
        ),
    }
    detail["ok"] = bool(
        detail["equal_to_expected"]
        and not detail["degenerate_input"]
        and not detail["result_is_constant"]
    )
    return detail


def run() -> dict:
    """Evaluate the authored program; return a JSON-able record."""
    record = {
        "module": "M2c",
        "artifact": str(AUTHORED.relative_to(REPO)),
        "entry": AUTHORED_ENTRY,
        "evaluator": "tilefoundry.evaluator.evaluate",
    }
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - reported verbatim
        record.update(
            status="NOT_RUN",
            exit_code=EXIT_NOT_RUN,
            reason=f"torch unavailable: {type(exc).__name__}: {exc}",
        )
        return record

    try:
        from tilefoundry.evaluator import evaluate

        fn = _load(AUTHORED, AUTHORED_ENTRY)
    except Exception as exc:  # noqa: BLE001 - reported verbatim
        record.update(
            status="NOT_RUN",
            exit_code=EXIT_NOT_RUN,
            reason=f"authored IR or evaluator unavailable: {type(exc).__name__}: {exc}",
        )
        return record

    record["torch_version"] = torch.__version__
    record["params"] = [p.name for p in fn.params]
    failures = []

    record["probes"] = {}
    for name, x in _probe_inputs().items():
        detail = _compare(torch, evaluate, fn, (x,), x)
        record["probes"][name] = detail
        if not detail.get("ok"):
            record["probes"][name]["failed"] = True
            failures.append(f"{name}: {json.dumps(detail, sort_keys=True)}")

    # Discrimination proof: the same comparison must reject a non-identity program.
    try:
        control = _load(CONTROL, CONTROL_ENTRY)
        x = torch.arange(ELEMENT_COUNT, dtype=torch.float32)
        detail = _compare(torch, evaluate, control, (x, x), x)
        detail["expectation"] = "must be reported NOT equal (control computes 2x)"
        detail["discriminates"] = not bool(detail.get("equal_to_expected"))
        record["control"] = detail
        if not detail["discriminates"]:
            failures.append(
                "discrimination: control program (y = x + t, t = x) compared equal to x, "
                "so the equality check does not discriminate"
            )
    except Exception as exc:  # noqa: BLE001 - reported verbatim
        record["control"] = {"discriminates": False, "error": f"{type(exc).__name__}: {exc}"}
        failures.append(f"discrimination control unavailable: {type(exc).__name__}: {exc}")

    record["failures"] = failures
    record["exit_code"] = EXIT_FAIL if failures else EXIT_PASS
    record["status"] = "FAIL" if failures else "PASS"
    return record


# --- pytest-compatible entry points (no pytest required to run the module) ---


def test_authored_copy_slice_cpu_equality() -> None:
    import unittest

    record = run()
    if record["status"] == "NOT_RUN":
        raise unittest.SkipTest(record["reason"])
    assert record["status"] == "PASS", json.dumps(record, indent=2)


def test_probe_inputs_are_nontrivial() -> None:
    """A degenerate probe set would make the equality assertion vacuous."""
    for name, x in _probe_inputs().items():
        assert int(x.unique().numel()) >= 2, name


def main() -> int:
    record = run()
    print(json.dumps(record, indent=2, sort_keys=True))
    return int(record["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
