"""M5: target-aware automatic candidate selection for a generic copy/reshard.

A generic, unscheduled storage-changing Reshard has to become the RIGHT Ascend move for
its direction, chosen by PLACEMENT and declared operand storages, without changing what
a non-Ascend target sees. Three claims are checked here and one is refused:

  1. the GLOBAL registry is untouched and CUDA-visible sole_candidate is still the single
     Copy carrier -- appending an Ascend op globally would make sole_candidate return None
     for every target, which is exactly the regression this guards;
  2. outside a mode, gm->smem picks ascend_gm_to_ub and smem->gmem picks ascend_ub_to_gm;
  3. inside a SimdVF, smem->rmem picks ascend_vld and rmem->smem picks ascend_vsts, and an
     outside-shaped move is NOT silently given an outside op inside a mode;
  4. when more than one declared candidate fits, selection REFUSES as ambiguous rather
     than picking one.

The operand mapping is the registered Reshard callback (src type, result type), so the
selection and the emitted operands use the same source of truth.

WHAT THIS DOES NOT COVER (stated, not faked): a generic 2-arg HIR Binary add has NO
honest Ascend mapping. ascend_vadd is predicated and needs a mask operand that a generic
add cannot source; inventing one would violate the producer/dominance rule, and Ascend
has no outside-only arithmetic op. That case is OUT OF SCOPE for M5 and keeps its
existing behaviour.

Run with the interpreter of record, TORCH_DEVICE_BACKEND_AUTOLOAD=0 (device untouched):
    PYTHONPATH=<repo>/src:<repo>/targets/ascend \
      python targets/ascend/tests/test_candidate_selection.py
Exit codes: 0 PASS, 1 FAIL, 2 ERROR.
"""
from __future__ import annotations

import sys

import tilefoundry_ascend  # noqa: F401  (registers SimdVF and the Ascend instructions)

from tilefoundry.ir.core import Call, Op, Var
from tilefoundry.ir.core.execution import RegionRequirement
from tilefoundry.ir.core.param_def import MemoryEffect, ParamDef
from tilefoundry.ir.hir.math.binary import Binary as HirBinary
from tilefoundry.ir.hir.schedule import ScheduleRefusal
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.pattern import TensorPattern
from tilefoundry.ir.tir.arith import Binary as TirBinary
from tilefoundry.ir.tir.memory.copy import Copy
from tilefoundry.ir.types import DType, StorageKind, make_tensor_type
from tilefoundry.target import CpuTarget
from tilefoundry.visitor_registry.candidates import (
    candidate_ops,
    register_target_candidates,
    select_instruction,
    sole_candidate,
    target_candidate_ops,
)
from tilefoundry_ascend import Ascend950PRTarget
from tilefoundry_ascend.ir.ops import GmToUb, UbToGm, Vld, Vsts

if "torch_npu" in sys.modules:
    raise SystemExit(
        "REFUSED: torch_npu is loaded in this process. This check is CPU-only and must"
        " never touch the device stack."
    )

SHAPE = (64,)


def _tensor(storage):
    return make_tensor_type(SHAPE, dtype=DType.f32, storage=storage)


def _reshard_call(src_storage, dst_storage):
    """A generic HIR Reshard from one storage to another."""
    source = Var(name="x", type=_tensor(src_storage))
    return Call(target=Reshard(), args=(source,), type=_tensor(dst_storage))


def _selected(src_storage, dst_storage, *, inside, target):
    op = select_instruction(
        _reshard_call(src_storage, dst_storage),
        target=target,
        inside=inside,
        active_kind="ascend.simdvf" if inside else None,
    )
    assert op is not None, "expected a selection, got None"
    return type(op)


def _expect_refusal(src_storage, dst_storage, *, inside, target, needle):
    try:
        select_instruction(
            _reshard_call(src_storage, dst_storage),
            target=target,
            inside=inside,
            active_kind="ascend.simdvf" if inside else None,
        )
    except ScheduleRefusal as error:
        text = str(error)
        assert needle in text, f"refused for the wrong reason: {text}"
        return text
    raise AssertionError("expected a refusal; the selection was accepted")


def check_global_registry_untouched() -> None:
    """No Ascend op may be appended globally; CUDA sole_candidate stays unambiguous."""
    assert candidate_ops(Reshard) == (Copy,), (
        f"global Reshard candidates changed: {candidate_ops(Reshard)}"
    )
    assert candidate_ops(HirBinary) == (TirBinary,), (
        f"global Binary candidates changed: {candidate_ops(HirBinary)}"
    )
    # A non-Ascend target scopes nothing, so sole_candidate is the historical global one.
    assert target_candidate_ops(CpuTarget(), Reshard) == (), "Cpu target scoped candidates"
    legacy = sole_candidate(_reshard_call("gmem", "smem"))
    assert isinstance(legacy, Copy), f"legacy sole_candidate is {type(legacy).__name__}, not Copy"


def check_ascend_outside_selection() -> None:
    target = Ascend950PRTarget()
    assert _selected("gmem", "smem", inside=False, target=target) is GmToUb
    assert _selected("smem", "gmem", inside=False, target=target) is UbToGm


def check_ascend_inside_selection() -> None:
    target = Ascend950PRTarget()
    assert _selected("smem", "rmem", inside=True, target=target) is Vld
    assert _selected("rmem", "smem", inside=True, target=target) is Vsts


def check_inside_never_uses_an_outside_op() -> None:
    """Placement is the first filter: an outside-shaped move inside a mode must refuse."""
    target = Ascend950PRTarget()
    _expect_refusal(
        "gmem", "smem", inside=True, target=target, needle="none of the declared instructions"
    )


# Two structurally identical test-only ops: same placement, same declared storages, so
# a target that offers both cannot disambiguate and MUST refuse. They are not registered
# anywhere global, so they cannot leak into other targets.
class _AmbiguousA(Op):
    execution_region = RegionRequirement(kind="ascend.simdvf", placement="outside")
    src = ParamDef(kind="input", pattern=TensorPattern(storage=StorageKind.GMEM), effect=MemoryEffect.READ)
    dst = ParamDef(kind="input", pattern=TensorPattern(storage=StorageKind.SMEM), effect=MemoryEffect.WRITE)


class _AmbiguousB(_AmbiguousA):
    pass


class _FakeTarget:
    name = "m5-ambiguity-test"


def check_ambiguity_refuses() -> None:
    register_target_candidates("m5-ambiguity-test", Reshard, (_AmbiguousA, _AmbiguousB))
    text = _expect_refusal(
        "gmem", "smem", inside=False, target=_FakeTarget(), needle="ambiguous"
    )
    assert "_AmbiguousA" in text and "_AmbiguousB" in text, f"ambiguity did not name both: {text}"


CHECKS = [
    ("global_registry_untouched", check_global_registry_untouched),
    ("ascend_outside_selection", check_ascend_outside_selection),
    ("ascend_inside_selection", check_ascend_inside_selection),
    ("inside_never_uses_an_outside_op", check_inside_never_uses_an_outside_op),
    ("ambiguity_refuses", check_ambiguity_refuses),
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
    if failed:
        print(f"{failed}/{len(CHECKS)} checks FAILED")
        return 1
    print(f"{len(CHECKS)}/{len(CHECKS)} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
