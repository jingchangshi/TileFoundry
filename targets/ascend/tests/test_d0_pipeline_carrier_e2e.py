"""D0 end-to-end carrier: the pipelined loop must survive to a real TIR For.

This is the program F0 exists for. Its loop DOES leave a value the function returns
(`y = tf.insert_slice(y, back, ...)` then `return y`), so it survives parse-time
DCE -- but before the reader fix it still could not finalize: the AFTER reader
returned an empty actual and falsely reported PIPELINE_DECLARATION_LOST.

Asserts the full chain, not a count:
  checked -> analyze -> finalize all succeed;
  HIR has >=1 LoopRegion and its .pipeline is typed;
  the lowered TIR has a For whose .pipeline is not None and the reader finds it;
  as_script preserves the authored `tf.pipelined(` surface.
"""
from __future__ import annotations

import pathlib
import tempfile

import tilefoundry_ascend  # noqa: F401
from tilefoundry.analysis import analyze
from tilefoundry.analysis.check import check_program
from tilefoundry.cli.source import load_authored_ir
from tilefoundry.inspection import as_script
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.tir.stmts import ExecutionScope, For, MeshScope, Sequential
from tilefoundry.ir.visitor import collect_exprs
from tilefoundry.schedule import finalize
from tilefoundry_ascend.passes.pipeline import collect_tir_pipeline_declarations

CARRIER = '''"""F0 acceptance carrier: pipelined VF vadd whose SMEM result feeds the return."""
from __future__ import annotations
from tilefoundry.dsl import *
from tilefoundry_ascend import Ascend950PRTarget

L = 64

@module(entry="k", target=Ascend950PRTarget(), topologies=(Topology("aiv", 1),))
class PipeVaddCarrier:
    @func
    def k(x: Tensor[(256,), "f32"], z: Tensor[(256,), "f32"]) -> Tensor[(256,), "f32"]:
        with Mesh(("aiv",), layout=(1,), names=("core",)) as aiv:
            y = tf.empty(Tensor[(256,), "f32", (256,), "gmem"])
            for i in tf.pipelined(4, num_stages=2, annotations={"enable_offset": False}):
                ux = tf.empty(Tensor[(L,), "f32", (L,), "smem"])
                sx = tf.schedule((x[i * L:i * L + L], ux), op=T.ascend_gm_to_ub())
                uz = tf.empty(Tensor[(L,), "f32", (L,), "smem"])
                sz = tf.schedule((z[i * L:i * L + L], uz), op=T.ascend_gm_to_ub())
                with T.SimdVF():
                    va = tf.empty(Tensor[(L,), "f32", (L,), "rmem"])
                    vb = tf.empty(Tensor[(L,), "f32", (L,), "rmem"])
                    mask = tf.empty(Tensor[(L,), "bool", (L,), "rmem"])
                    la = tf.schedule((sx, va), op=T.ascend_vld())
                    lb = tf.schedule((sz, vb), op=T.ascend_vld())
                    pred = tf.schedule((mask,), op=T.ascend_pset(elem_width=32))
                    acc = tf.empty(Tensor[(L,), "f32", (L,), "rmem"])
                    total = tf.schedule((la, lb, pred, acc), op=T.ascend_vadd())
                    stored = tf.empty(Tensor[(L,), "f32", (L,), "smem"])
                    back = tf.schedule((total, stored), op=T.ascend_vsts())
                y = tf.insert_slice(y, back, (i * L,))
            return y
'''


def _walk_for(node, found):
    if isinstance(node, (list, tuple)):
        for item in node:
            _walk_for(item, found)
    elif isinstance(node, For):
        found.append(node)
        _walk_for(node.body, found)
    elif isinstance(node, (Sequential, MeshScope, ExecutionScope)):
        _walk_for(node.body, found)


def test_carrier_survives_to_a_typed_tir_pipeline():
    directory = pathlib.Path(tempfile.mkdtemp(prefix="tf_f0_carrier_"))
    path = directory / "carrier.py"
    path.write_text(CARRIER)
    module = load_authored_ir("%s:PipeVaddCarrier" % path)
    function = module.entry_function()

    loops = [e for e in collect_exprs(function.body) if isinstance(e, LoopRegion)]
    assert len(loops) >= 1, "the authored loop was dead-code eliminated"
    assert loops[0].pipeline is not None, "HIR LoopRegion lost its pipeline spec"
    assert loops[0].pipeline.num_stages == 2

    checked, entry = check_program(module, function)
    result = analyze(checked, entry, analysis=("memory",))
    lowered = finalize(result, authored=function)

    tir_fors = []
    for item in lowered.functions:
        _walk_for(getattr(item, "body", None), tir_fors)
    assert len(tir_fors) >= 1, "the lowered TIR has no For -- the loop vanished"
    assert tir_fors[0].pipeline is not None, "TIR For.pipeline is None"
    assert tir_fors[0].pipeline.num_stages == 2
    assert collect_tir_pipeline_declarations(lowered), (
        "the AFTER reader still finds no pipeline in a lowered program that has one"
    )

    printed = as_script(lowered)
    assert "tf.pipelined(" in printed, printed
