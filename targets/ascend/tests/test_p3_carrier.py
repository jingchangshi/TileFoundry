"""P3 V02: the vld carrier signature, pinned by typed admission and a real program.

WHAT THIS FILE IS FOR. `ascend_vld` has two exact signatures and nothing in between:

    src bf16 smem + dist="UNPK_B16" -> dst bf16 CARRIER   (an unpacking transport load)
    src f32  smem + no dist         -> dst f32  RMEM      (a plain element load)

The carrier class is not an f32 element register: only a conversion may consume it, and a
bf16 load that lands in an f32 element register would be a numeric cast fused into vld.

WHY IT EXISTS AT ALL. At e73f0f2 the module carried TWO definitions of `_VldDist`
(ops.py:86 and ops.py:117); the second silently replaced the first, so the rule that was
in force checked only "src dtype says UNPK_B16" and never looked at the destination. A
bf16 UNPK_B16 load into an f32 element register was ACCEPTED and finalized. This file
pins the single definition and both refusals, and the evidence bundle for it records the
mutation run that deletes the rule and watches these negatives turn positive.

HOW THE REFUSAL IS ASSERTED. By the message the layer that STATES the rule produces, not
by "an exception happened". A refusal that arrives from some other check (a shape
mismatch, a missing producer) is not evidence for this rule, so each negative asserts the
rule's own text.

TWO LEVELS, because they are different claims:
  * typed admission -- `check_call` on real TensorTypes with the declared instruction:
    the exact signature union, including the two positives. This is the same checker the
    HIR type inferencer and the TIR walker both call, so it is not a re-implementation.
  * a real program through the public CLI -- the live negative (bf16 UNPK_B16 into an
    f32 register, consumed by vsts, so it cannot be dead-code eliminated) must fail
    analyze by name, and the f32 positive must still finalize.

Exit codes: 0 PASS, 1 FAIL, 2 ERROR.
"""
from __future__ import annotations

import os
import subprocess
import sys

import tilefoundry_ascend  # noqa: F401  (registers SimdVF and its instructions)

from tilefoundry.ir.hir.schedule import ScheduleRefusal
from tilefoundry.ir.pattern import between_rules
from tilefoundry.ir.types import DType, Mesh, StorageKind, TensorType, Topology
from tilefoundry.visitor_registry.instruction import check_call
from tilefoundry_ascend.ir.ops import Vld, _VldDist

if "torch_npu" in sys.modules:
    raise SystemExit(
        "REFUSED: torch_npu is loaded in this process. This check is CPU-only and must"
        " never touch the device stack."
    )

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
FIX = os.path.join(ROOT, "targets", "ascend", "tests", "fixtures", "p3_carrier")
LANES = 64
AIV = Mesh((Topology("aiv", 1),), (1,), ("core",))

#: The rule's own wording. A refusal that does not carry it came from somewhere else.
RULE_TEXT = "bf16 loads state dist=UNPK_B16 and land in the carrier class"


def tensor(dtype, storage):
    return TensorType((LANES,), DType.from_name(dtype), None, StorageKind[storage.upper()])


def cli(*args):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([os.path.join(ROOT, "src"), os.path.join(ROOT, "targets", "ascend")])
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run([sys.executable, "-m", "tilefoundry.cli", *args], cwd=ROOT, env=env, capture_output=True, text=True)


def _admission(src_dtype, src_storage, dst_dtype, dst_storage, dist):
    """Admit one vld through the shared checker; return None or the refusal text."""
    op = Vld(dist=dist)
    try:
        check_call(op, (tensor(src_dtype, src_storage), tensor(dst_dtype, dst_storage)), AIV, frame=None)
    except ScheduleRefusal as error:
        return str(error)
    return None


def _rule_line(message):
    """The line of a refusal that carries the rule's own wording."""
    for line in message.splitlines():
        if RULE_TEXT in line:
            return line.strip()[:200]
    return message.splitlines()[0][:200]


def _assert_rule_refusal(message, what):
    if message is None:
        raise AssertionError(f"{what}: ACCEPTED; the signature rule did not fire")
    if RULE_TEXT not in message:
        raise AssertionError(f"{what}: refused, but not by the vld signature rule: {message}")
    print("       refused: " + _rule_line(message))


def check_one_rule_definition() -> None:
    """The declaration carries exactly ONE _VldDist rule. Two would be two truths."""
    rules = [rule for rule in between_rules(Vld) if isinstance(rule, _VldDist)]
    if len(rules) != 1:
        raise AssertionError(f"Vld carries {len(rules)} _VldDist rules; exactly one must be in force")


def check_f32_no_dist_to_rmem_accepted() -> None:
    """The plain f32 element load is still admitted: the rule must not over-refuse."""
    message = _admission("f32", "smem", "f32", "rmem", None)
    if message is not None:
        raise AssertionError(f"f32 load with no dist was refused: {message}")


def check_bf16_unpk_to_carrier_accepted() -> None:
    """The unpacking transport load into the carrier class is admitted."""
    message = _admission("bf16", "smem", "bf16", "carrier", "UNPK_B16")
    if message is not None:
        raise AssertionError(f"bf16 UNPK_B16 into carrier was refused: {message}")


def check_bf16_unpk_to_f32_rmem_refused() -> None:
    """bf16 + UNPK_B16 into an f32 ELEMENT register: a cast fused into vld. Refused."""
    _assert_rule_refusal(_admission("bf16", "smem", "f32", "rmem", "UNPK_B16"), "bf16->f32 rmem")


def check_f32_no_dist_to_carrier_refused() -> None:
    """f32 with no dist into the carrier class: the carrier is the UNPK destination only."""
    _assert_rule_refusal(_admission("f32", "smem", "bf16", "carrier", None), "f32->carrier")


def check_bf16_without_dist_refused() -> None:
    """A packed bf16 load that states no dist is refused rather than assumed."""
    _assert_rule_refusal(_admission("bf16", "smem", "bf16", "carrier", None), "bf16 no dist")


def check_f32_with_dist_refused() -> None:
    """An f32 load that states UNPK_B16 is refused: the dist is part of the signature."""
    _assert_rule_refusal(_admission("f32", "smem", "f32", "rmem", "UNPK_B16"), "f32 with dist")


def check_live_bf16_to_f32_program_refused() -> None:
    """The same defect as a LIVE program: the load feeds vsts, so it cannot be dropped."""
    result = cli("analyze", os.path.join(FIX, "vld_bf16_to_f32.py") + ":VldBf16ToF32")
    if result.returncode == 0:
        raise AssertionError("live bf16->f32 vld program was ACCEPTED by analyze")
    if RULE_TEXT not in result.stderr:
        raise AssertionError(f"live program refused, but not by the vld signature rule: {result.stderr.strip()[-300:]}")
    print(f"       refused (analyze exit {result.returncode}): {_rule_line(result.stderr)}")


def check_f32_positive_program_still_finalizes() -> None:
    """The f32 no-dist positive program still finalizes: no regression from the rule."""
    out = os.path.join(os.environ.get("TMPDIR", "/tmp"), "p3_carrier_f32_vadd.py")
    result = cli("schedule", "finalize", os.path.join(ROOT, "examples", "rmsnorm_950pr", "vf_vadd.py") + ":VfVadd", out)
    if result.returncode != 0:
        raise AssertionError(f"f32 vadd finalize failed: {result.stderr.strip()[-300:]}")


CHECKS = [
    ("one_rule_definition", check_one_rule_definition),
    ("f32_no_dist_to_rmem_accepted", check_f32_no_dist_to_rmem_accepted),
    ("bf16_unpk_to_carrier_accepted", check_bf16_unpk_to_carrier_accepted),
    ("bf16_unpk_to_f32_rmem_refused", check_bf16_unpk_to_f32_rmem_refused),
    ("f32_no_dist_to_carrier_refused", check_f32_no_dist_to_carrier_refused),
    ("bf16_without_dist_refused", check_bf16_without_dist_refused),
    ("f32_with_dist_refused", check_f32_with_dist_refused),
    ("live_bf16_to_f32_program_refused", check_live_bf16_to_f32_program_refused),
    ("f32_positive_program_still_finalizes", check_f32_positive_program_still_finalizes),
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
        except Exception as error:  # noqa: BLE001 -- any raise is an ERROR, not a verdict
            failed += 1
            print(f"ERROR {name}: {error!r}")
        else:
            print(f"ok   {name}")
    if failed:
        print(f"{failed}/{len(CHECKS)} checks FAILED")
        return 1
    print(f"{len(CHECKS)}/{len(CHECKS)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
