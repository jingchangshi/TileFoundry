"""task-76 B: operand-WRITE end-to-end deletion is observable.

A loop whose ONLY effect is an instruction writing a private operand (gm_to_ub writing a UB
window) with its produced value unused must be DCE'd (LoopRegion=0); the SAME loop carrying
that value back into GM via insert_slice / return must be retained (LoopRegion>=1). This is
the end-to-end complement to the classifier unit test.
"""
from __future__ import annotations

# ruff: noqa: PLC0415, I001  -- local imports keep each check's dependency surface explicit

import os
import pathlib
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "targets", "ascend")]

HEADER = '''from __future__ import annotations
from tilefoundry.dsl import *
from tilefoundry_ascend import Ascend950PRTarget

L = 64

@module(entry="k", target=Ascend950PRTarget(), topologies=(Topology("aiv", 1),))
class M:
    @func
    def k(x: Tensor[(256,), "f32"]) -> Tensor[(256,), "f32"]:
        with Mesh(("aiv",), layout=(1,), names=("core",)) as aiv:
            y = tf.empty(Tensor[(256,), "f32", (256,), "gmem"])
            ub = tf.empty(Tensor[(L,), "f32", (L,), "smem"])
            for i in range(4):
                s = tf.schedule((x[i * L:i * L + L], ub), op=T.ascend_gm_to_ub())
'''

OPERAND_ONLY = HEADER + "            return y\n"
GM_RETURN = HEADER + "                y = tf.insert_slice(y, s, (i * L,))\n            return y\n"


def _load(source):
    from tilefoundry.cli.source import load_authored_ir
    from tilefoundry.ir.visitor import collect_exprs
    from tilefoundry.ir.hir.loop_region import LoopRegion

    directory = pathlib.Path(tempfile.mkdtemp(prefix="tf_t76_"))
    path = directory / "prog.py"
    path.write_text(source)
    module = load_authored_ir("%s:M" % path)
    function = module.entry_function()
    body = function.body
    return [e for e in collect_exprs(body) if isinstance(e, LoopRegion)] if body is not None else []


def test_operand_write_only_loop_is_deleted():
    assert len(_load(OPERAND_ONLY)) == 0, "an unconsumed private operand-write loop must be DCE'd"


def test_gm_return_operand_loop_is_retained():
    loops = _load(GM_RETURN)
    assert len(loops) == 1, "the same loop carried into GM/return must be retained"


def main() -> int:
    tests = (test_operand_write_only_loop_is_deleted, test_gm_return_operand_loop_is_retained)
    failed = 0
    for case in tests:
        try:
            case()
        except Exception as error:  # noqa: BLE001
            failed += 1
            print("FAIL %s: %s: %s" % (case.__name__, type(error).__name__, error))
        else:
            print("PASS %s" % case.__name__)
    print("%d/%d passed" % (len(tests) - failed, len(tests)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
