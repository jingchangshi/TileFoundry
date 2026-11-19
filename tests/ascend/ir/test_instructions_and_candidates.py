"""Verify Ascend execution modes, 19 canonical pipe TIR ops, and 8 core HIR bindings."""

from __future__ import annotations

import pathlib

import pytest

from tilefoundry.analysis import analyze, check_program
from tilefoundry.cli.source import load_authored_ir
from tilefoundry.evaluator import evaluate
from tilefoundry.ir.core import supported_op_capabilities
from tilefoundry.ir.core.execution import declared_kinds, execution_kind_for_surface
from tilefoundry.ir.hir.math.binary import Binary as HirBinary
from tilefoundry.ir.hir.math.clamp import Clamp as HirClamp
from tilefoundry.ir.hir.math.unary import Unary as HirUnary
from tilefoundry.ir.hir.nn.matmul import MatMul
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.hir.tensor.cast import Cast
from tilefoundry.ir.hir.tensor.reduce import Reduce as HirReduce
from tilefoundry.ir.hir.tensor.where import Where as HirWhere
from tilefoundry.passes.transforms.convert_hir_to_tir import LoweringError
from tilefoundry.schedule.finalize import finalize
from tilefoundry.visitor_registry import typeinfer_registry
from tilefoundry.visitor_registry.access_relation import access_relation_registry
from tilefoundry.visitor_registry.candidates import target_candidate_ops
from tilefoundry.visitor_registry.verify import VerifyError, verify_prim_function
from tilefoundry_ascend import AscendTarget
from tilefoundry_ascend.ascend.ir.execution import (
    SIMDVF_KIND,
    SIMTVF_KIND,
    SUPPORTED_VECTOR_DTYPES,
    SimdVF,
    SimtVF,
)
from tilefoundry_ascend.ascend.ir.tir.instructions import (
    ASCEND_OP_SPECS,
    AscendCrossCoreSync,
    AscendDualCopy,
    AscendDualCopyL0cToUb,
    AscendDualCopyUbToL1,
    AscendGmToL1,
    AscendGmToUb,
    AscendL0cToGm,
    AscendL1ToL0A,
    AscendL1ToL0B,
    AscendMmad,
    AscendRegToUb,
    AscendUbToGm,
    AscendUbToReg,
    AscendVbinary,
    AscendVcast,
    AscendVclamp,
    AscendVreduce,
    AscendVunary,
    AscendVwhere,
)
from tilefoundry_ascend.ascend.target import AscendSpaceFacts
import torch

HEADER = (
    "from __future__ import annotations\n\n"
    "from tilefoundry.dsl import *\n"
    "from tilefoundry_ascend import AscendTarget\n"
    "TOPOS = (Topology('block', 1), Topology('aic', 1), "
    "Topology('aiv', 2), Topology('lane', 64))\n"
)


def _load_module(tmp_path: pathlib.Path, body: str, name: str = "prog.py"):
    """Write *body* to *tmp_path* and parse authored IR module M."""
    path = tmp_path / name
    path.write_text(HEADER + "\n" + body, encoding="utf-8")
    return load_authored_ir(f"{path}:M")


def _lower(mod):
    """Run check, memory analysis, and HIR-to-TIR finalization on *mod*."""
    authored = mod.entry_function()
    checked_mod, checked_fn = check_program(mod, authored)
    analyzed = analyze(checked_mod, checked_fn, analysis="memory")
    return finalize(analyzed, authored=authored)


def test_simdvf_and_simtvf_execution_kind_registration() -> None:
    """Verify SimdVF and SimtVF execution domains and UB budget projections."""
    assert SIMDVF_KIND.name == "ascend.simdvf"
    assert SIMTVF_KIND.name == "ascend.simtvf"
    assert SimdVF.execution_kind == SIMDVF_KIND
    assert SimtVF.execution_kind == SIMTVF_KIND
    assert execution_kind_for_surface("SimdVF", dialect="T") == SIMDVF_KIND
    assert execution_kind_for_surface("SimtVF", dialect="T") == SIMTVF_KIND
    assert SIMDVF_KIND in declared_kinds()
    assert SIMTVF_KIND in declared_kinds()
    assert len(SUPPORTED_VECTOR_DTYPES) >= 16

    target = AscendTarget("ascend.950pr")
    spaces = target.get_facts(AscendSpaceFacts)
    assert spaces.ub_limit_bytes("simtvf") == 216 * 1024
    assert spaces.ub_limit_bytes("simdvf") == 248 * 1024
    assert spaces.ub_limit_bytes("simdvf", unreserved_ubuf=True) == 256 * 1024


def test_canonical_pipe_tir_primitives_inventory_and_registries() -> None:
    """Verify the 19 canonical pipe-defining TIR ops register capabilities and semantics."""
    assert len(ASCEND_OP_SPECS) == 19
    target = AscendTarget("ascend.950pr")
    supported = {op_cls for op_cls, _ in supported_op_capabilities(target)}
    for spec in ASCEND_OP_SPECS:
        assert spec.op_cls in supported
        assert typeinfer_registry.lookup(spec.op_cls) is not None
        assert access_relation_registry.lookup(spec.op_cls) is not None
        if spec.capability == "ascend.vector":
            assert spec.placement == "inside"
        else:
            assert spec.placement == "outside"


def test_canonical_core_hir_candidate_bindings() -> None:
    """Verify the 8 core HIR algebraic and sharding candidate bindings."""
    target = AscendTarget("ascend.950pr")
    expected_singletons = (
        (MatMul, AscendMmad),
        (Cast, AscendVcast),
        (HirBinary, AscendVbinary),
        (HirUnary, AscendVunary),
        (HirReduce, AscendVreduce),
        (HirWhere, AscendVwhere),
        (HirClamp, AscendVclamp),
    )
    for hir_cls, expected_tir in expected_singletons:
        cands = target_candidate_ops(target, hir_cls)
        assert cands == (expected_tir,)

    reshard_cands = set(target_candidate_ops(target, Reshard))
    expected_reshard = {
        AscendGmToUb,
        AscendUbToGm,
        AscendGmToL1,
        AscendL1ToL0A,
        AscendL1ToL0B,
        AscendL0cToGm,
        AscendDualCopy,
        AscendDualCopyL0cToUb,
        AscendDualCopyUbToL1,
        AscendUbToReg,
        AscendRegToUb,
    }
    assert reshard_cands == expected_reshard


def test_parametric_vector_lowering_and_cpu_eval(tmp_path: pathlib.Path) -> None:
    """Verify parametric vbinary, vunary, and vreduce lower, verify, and evaluate."""
    body = """
@module(entry="vec_pipe", target=AscendTarget("ascend.950pr"), topologies=TOPOS)
class M:
    @func
    def vec_pipe(a: Tensor[(4, 64), "f32"], b: Tensor[(4, 64), "f32"]) -> Tensor[(4, 64), "f32"]:
        with Mesh(("block", "aiv"), (1, 2), names=("b", "v")) as _bv:
            a_ub = tf.schedule((a, tf.empty(Tensor[(4, 64), "f32", "smem"])), op=T.ascend_gm_to_ub())
            b_ub = tf.schedule((b, tf.empty(Tensor[(4, 64), "f32", "smem"])), op=T.ascend_gm_to_ub())
            with T.SimdVF():
                ar = tf.schedule((a_ub, tf.empty(Tensor[(4, 64), "f32", "rmem"])), op=T.ascend_ub_to_reg())
                br = tf.schedule((b_ub, tf.empty(Tensor[(4, 64), "f32", "rmem"])), op=T.ascend_ub_to_reg())
                sr = tf.schedule((ar, br, tf.empty(Tensor[(4, 64), "f32", "rmem"])), op=T.ascend_vbinary(kind="add"))
                er = tf.schedule((sr, tf.empty(Tensor[(4, 64), "f32", "rmem"])), op=T.ascend_vunary(kind="exp"))
                c_ub = tf.schedule((er, tf.empty(Tensor[(4, 64), "f32", "smem"])), op=T.ascend_reg_to_ub())
            return tf.schedule((c_ub, tf.empty(Tensor[(4, 64), "f32", "gmem"])), op=T.ascend_ub_to_gm())
"""
    mod = _load_module(tmp_path, body)
    lowered = _lower(mod)
    verify_prim_function(lowered.entry_function(), module_fns=lowered)

    x = torch.ones((4, 64), dtype=torch.float32)
    y = torch.ones((4, 64), dtype=torch.float32)
    out = evaluate(mod.entry_function(), x, y).data
    torch.testing.assert_close(out, torch.exp(x + y))


def test_placement_and_crosscore_sync_bounds(tmp_path: pathlib.Path) -> None:
    """Verify DMA inside SimdVF is rejected and CrossCoreSync enforces depth bounds."""
    bad_dma = """
@module(entry="bad_dma", target=AscendTarget("ascend.950pr"), topologies=TOPOS)
class M:
    @func
    def bad_dma(a: Tensor[(64,), "f32"]) -> Tensor[(64,), "f32"]:
        with Mesh(("block", "aiv"), (1, 2), names=("b", "v")) as _bv:
            with T.SimdVF():
                a_ub = tf.schedule((a, tf.empty(Tensor[(64,), "f32", "smem"])), op=T.ascend_gm_to_ub())
            return tf.schedule((a_ub, tf.empty(Tensor[(64,), "f32", "gmem"])), op=T.ascend_ub_to_gm())
"""
    with pytest.raises((LoweringError, VerifyError)):
        lowered = _lower(_load_module(tmp_path, bad_dma, "bad_dma.py"))
        verify_prim_function(lowered.entry_function(), module_fns=lowered)

    bad_sync = """
@module(entry="bad_sync", target=AscendTarget("ascend.950pr"), topologies=TOPOS)
class M:
    @func
    def bad_sync(flag: Tensor[(1,), "i32"]) -> Tensor[(1,), "i32"]:
        with Mesh(("block",), (1,), names=("b",)) as _b:
            return tf.schedule((flag,), op=T.ascend_cross_core_sync(mode="mode4", flag_id=19))
"""
    with pytest.raises((LoweringError, VerifyError)):
        lowered_sync = _lower(_load_module(tmp_path, bad_sync, "bad_sync.py"))
        verify_prim_function(lowered_sync.entry_function(), module_fns=lowered_sync)
    assert AscendCrossCoreSync is not None
