"""Mask granularity under CONTROL FLOW: a branch/loop may clobber the producer record.

Symptom this file pins: 'pset16(mask); if True: pset32(mask); vcvt16(mask)' was ACCEPTED,
because the If/For save/restore in verify_predicate_producers brought the pre-construct
16-bit record back and hid the real overwrite. The rule is now conservative: after a
construct in which a mask Var was written ANYWHERE, that mask's granularity is UNKNOWN and
a consumer that requires a specific width is REFUSED BY NAME. Uses of a mask produced and
read INSIDE the same construct are still checked there.

Fixtures live in tests/fixtures/p3_carrier/mask_closure/ (authored TIR, same shape as the
public schedule-finalize artifact). Each case is loaded with load_authored_ir and run
through verify_module + verify_prim_function.

Exit codes: 0 PASS, 1 FAIL, 2 ERROR.
"""
from __future__ import annotations

import glob
import os
import sys

import tilefoundry_ascend  # noqa: F401

from tilefoundry.cli.source import load_authored_ir
from tilefoundry.visitor_registry.verify import verify_module, verify_prim_function

if "torch_npu" in sys.modules:
    raise SystemExit("REFUSED: torch_npu is loaded; this check is CPU-only")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DIR = os.path.join(ROOT, "targets", "ascend", "tests", "fixtures", "p3_carrier", "mask_closure")
FRAGMENT = "was written inside a control-flow construct"


def _verify(name):
    module = load_authored_ir(os.path.join(DIR, f"{name}.py") + ":Bf16VcvtChain")
    verify_module(module)
    verify_prim_function(module.entry_function())


def _refusal(name):
    try:
        _verify(name)
    except Exception as error:  # noqa: BLE001 -- the message is the verdict
        return str(error)
    return None


def _assert_clobber_refused(name):
    message = _refusal(name)
    if message is None:
        raise AssertionError(f"{name}: ACCEPTED; the control-flow clobber was not caught")
    if FRAGMENT not in message:
        raise AssertionError(f"{name}: refused by the wrong layer: {message[:200]}")
    print(f"       {name}: {message.splitlines()[0][:150]}")


CHECKS = [
    ("positive_branch_inside_produce_and_use", lambda: _verify("branch_inside_produce_and_use")),
    ("branch_clobber_refused", lambda: _assert_clobber_refused("branch_clobber")),
    ("loop_clobber_refused", lambda: _assert_clobber_refused("loop_clobber")),
    ("nested_clobber_refused", lambda: _assert_clobber_refused("nested_clobber")),
    ("if_else_clobber_refused", lambda: _assert_clobber_refused("if_else_clobber")),
]


def main() -> int:
    failed = 0
    for name, check in CHECKS:
        try:
            check()
        except AssertionError as error:
            failed += 1
            print(f"FAIL {name}")
            print(f"     {error}")
        except Exception as error:  # noqa: BLE001
            failed += 1
            print(f"ERROR {name}: {error!r}")
        else:
            print(f"ok   {name}")
    print(f"{len(CHECKS) - failed}/{len(CHECKS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
