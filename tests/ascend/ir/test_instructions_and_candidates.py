"""Verify Ascend execution modes, TIR instructions, candidates, and placement checks."""

from __future__ import annotations

import pathlib

import pytest
import torch

from tilefoundry.analysis import analyze, check_program
from tilefoundry.cli.source import load_authored_ir
from tilefoundry.evaluator import evaluate
from tilefoundry.ir.core import Call, Var, supported_op_capabilities
from tilefoundry.ir.core.execution import declared_kinds, execution_kind_for_surface
from tilefoundry.ir.core.kinds import BinaryKind, ReduceKind, UnaryKind
from tilefoundry.ir.hir.math.binary import Binary as HirBinary
from tilefoundry.ir.hir.math.clamp import Clamp as HirClamp
from tilefoundry.ir.hir.math.unary import Unary as HirUnary
from tilefoundry.ir.hir.nn.matmul import MatMul
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.hir.tensor.cast import Cast
from tilefoundry.ir.hir.tensor.reduce import Reduce as HirReduce
from tilefoundry.ir.hir.tensor.where import Where as HirWhere
from tilefoundry.ir.types import DType, Layout, StorageKind, TensorType
from tilefoundry.passes.transforms.convert_hir_to_tir import LoweringError
from tilefoundry.schedule.candidates import candidates as report_candidates
from tilefoundry.schedule.facts import listing as facts_listing
from tilefoundry.schedule.facts import one as facts_one
from tilefoundry.schedule.finalize import finalize
from tilefoundry.visitor_registry import typeinfer_registry
from tilefoundry.visitor_registry.access_relation import access_relation_registry
from tilefoundry.visitor_registry.candidates import select_instruction, sole_candidate
from tilefoundry.visitor_registry.verify import VerifyError, verify_prim_function
from tilefoundry_ascend import AscendTarget
from tilefoundry_ascend.ascend.ir.execution import SIMDVF_KIND, SIMTVF_KIND, SimdVF, SimtVF
from tilefoundry_ascend.ascend.ir.tir.instructions import (
    ASCEND_OP_SPECS,
    AscendDualCopyL0cToUb,
    AscendDualCopyUbToL1,
    AscendGmToL1,
    AscendGmToUb,
    AscendL0cToGm,
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

HEADER = (
    "from __future__ import annotations\n\n"
    "from tilefoundry.dsl import *\n"
    "from tilefoundry_ascend import AscendTarget\n"
    "TOPOS = (Topology('block', 1), Topology('aic', 1), "
    "Topology('aiv', 2), Topology('lane', 64))\n"
)


def _load_module(tmp_path: pathlib.Path, body: str, name: str = "prog.py"):
    path = tmp_path / name
    path.write_text(HEADER + "\n" + body, encoding="utf-8")
    return load_authored_ir(f"{path}:M")


def _lower(mod):
    authored = mod.entry_function()
    checked_mod, checked_fn = check_program(mod, authored)
    analyzed = analyze(checked_mod, checked_fn, analysis="memory")
    return finalize(analyzed, authored=authored)


def test_simdvf_and_simtvf_execution_kind_registration():
    assert SIMDVF_KIND.name == "ascend.simdvf"
    assert SIMDVF_KIND.surface == "SimdVF"
    assert SIMDVF_KIND.required_topologies == ("aiv",)
    assert SIMDVF_KIND.mask_width_bits == 32
    assert SIMDVF_KIND.conversion_mask_width_bits == 16
    assert SIMDVF_KIND.lane_count is None
    assert SIMDVF_KIND.register_storages == (StorageKind.RMEM, StorageKind.CARRIER)
    assert SIMDVF_KIND.element_dtypes == (DType.f32,)
    assert SimdVF.execution_kind == SIMDVF_KIND
    assert execution_kind_for_surface("SimdVF", dialect="T") == SIMDVF_KIND
    assert SIMDVF_KIND in declared_kinds()

    assert SIMTVF_KIND.name == "ascend.simtvf"
    assert SIMTVF_KIND.surface == "SimtVF"
    assert SimtVF.execution_kind == SIMTVF_KIND
    assert execution_kind_for_surface("SimtVF", dialect="T") == SIMTVF_KIND
    assert SIMTVF_KIND in declared_kinds()


def test_ascend_ops_registered_with_target_and_semantics():
    target = AscendTarget("ascend.950pr")
    supported = {op_cls for op_cls, _ in supported_op_capabilities(target)}
    assert len(ASCEND_OP_SPECS) == 50
    for spec in ASCEND_OP_SPECS:
        assert spec.op_cls in supported
        assert typeinfer_registry.lookup(spec.op_cls) is not None
        assert access_relation_registry.lookup(spec.op_cls) is not None
    facts = target.get_facts(AscendSpaceFacts)
    assert facts.ub_usable_bytes == 221184


def test_target_scoped_candidates_and_global_isolation():
    target = AscendTarget("ascend.950pr")
    shape = (16, 16)
    layout = Layout(shape)
    gm = TensorType(shape, DType.f16, layout, StorageKind.GMEM)
    smem = TensorType(shape, DType.f16, layout, StorageKind.SMEM)
    tmem = TensorType(shape, DType.f16, layout, StorageKind.TMEM)
    rmem_f16 = TensorType(shape, DType.f16, layout, StorageKind.RMEM)
    rmem_f32 = TensorType(shape, DType.f32, layout, StorageKind.RMEM)

    assert type(sole_candidate(Reshard())) is not AscendGmToUb
    gm_call = Call(Reshard(), (Var("x", type=gm),), type=smem)
    assert type(select_instruction(gm_call, target=target, inside=False)) is AscendGmToUb
    ug_call = Call(Reshard(), (Var("x", type=smem),), type=gm)
    assert type(select_instruction(ug_call, target=target, inside=False)) is AscendUbToGm
    gl_call = Call(Reshard(), (Var("x", type=gm),), type=tmem)
    assert type(select_instruction(gl_call, target=target, inside=False)) is AscendGmToL1
    lg_call = Call(Reshard(), (Var("x", type=tmem),), type=gm)
    assert type(select_instruction(lg_call, target=target, inside=False)) is AscendL0cToGm
    lu_call = Call(Reshard(), (Var("x", type=tmem),), type=smem)
    assert type(select_instruction(lu_call, target=target, inside=False)) is AscendDualCopyL0cToUb
    ul_call = Call(Reshard(), (Var("x", type=smem),), type=tmem)
    assert type(select_instruction(ul_call, target=target, inside=False)) is AscendDualCopyUbToL1

    ur_call = Call(Reshard(), (Var("x", type=smem),), type=rmem_f16)
    selected_ur = select_instruction(
        ur_call, target=target, inside=True, active_kind="ascend.simdvf"
    )
    assert type(selected_ur) is AscendUbToReg
    ru_call = Call(Reshard(), (Var("x", type=rmem_f16),), type=smem)
    selected_ru = select_instruction(
        ru_call, target=target, inside=True, active_kind="ascend.simdvf"
    )
    assert type(selected_ru) is AscendRegToUb

    cast_call = Call(Cast(dtype=DType.f32), (Var("x", type=rmem_f16),), type=rmem_f32)
    selected_cast = select_instruction(
        cast_call, target=target, inside=True, active_kind="ascend.simdvf"
    )
    assert type(selected_cast) is AscendVcast

    mm_call = Call(MatMul(), (Var("a", type=tmem), Var("b", type=tmem)), type=tmem)
    assert type(select_instruction(mm_call, target=target, inside=False)) is AscendMmad

    for bkind in (BinaryKind.ADD, BinaryKind.SUB, BinaryKind.MUL, BinaryKind.DIV, BinaryKind.MAX, BinaryKind.LT):
        b_out = TensorType(shape, DType.bool if bkind is BinaryKind.LT else DType.f32, layout, StorageKind.RMEM)
        b_call = Call(
            HirBinary(kind=bkind),
            (Var("a", type=rmem_f32), Var("b", type=rmem_f32)),
            type=b_out,
        )
        sel = select_instruction(b_call, target=target, inside=True, active_kind="ascend.simdvf")
        assert type(sel) is AscendVbinary and sel.kind is bkind

    u_call = Call(HirUnary(kind=UnaryKind.EXP), (Var("a", type=rmem_f32),), type=rmem_f32)
    assert type(select_instruction(u_call, target=target, inside=True, active_kind="ascend.simdvf")) is AscendVunary
    r_call = Call(
        HirReduce(axes=(-1,), keepdim=True, kind=ReduceKind.SUM),
        (Var("a", type=rmem_f32),),
        type=TensorType((16, 1), DType.f32, Layout((16, 1)), StorageKind.RMEM),
    )
    assert type(select_instruction(r_call, target=target, inside=True, active_kind="ascend.simdvf")) is AscendVreduce
    rmem_bool = TensorType(shape, DType.bool, layout, StorageKind.RMEM)
    w_call = Call(
        HirWhere(),
        (Var("c", type=rmem_bool), Var("a", type=rmem_f32), Var("b", type=rmem_f32)),
        type=rmem_f32,
    )
    assert type(select_instruction(w_call, target=target, inside=True, active_kind="ascend.simdvf")) is AscendVwhere
    cl_call = Call(
        HirClamp(min_val=-1.0, max_val=1.0),
        (Var("a", type=rmem_f32),),
        type=rmem_f32,
    )
    assert type(select_instruction(cl_call, target=target, inside=True, active_kind="ascend.simdvf")) is AscendVclamp


def test_cube_flow_eval_analyze_finalize_and_verify(tmp_path: pathlib.Path):
    body = """
@module(entry="k", target=AscendTarget("ascend.950pr"), topologies=TOPOS)
class M:
    @func
    def k(a: Tensor[(16, 16), "f16"], b: Tensor[(16, 16), "f16"]) -> Tensor[(16, 16), "f32"]:
        with Mesh(("aic",), (1,), names=("c",)) as _c:
            a_l1 = tf.schedule((a, tf.empty(Tensor[(16, 16), "f16", "tmem"])), op=T.ascend_gm_to_l1())
            b_l1 = tf.schedule((b, tf.empty(Tensor[(16, 16), "f16", "tmem"])), op=T.ascend_gm_to_l1())
            a_l0 = tf.schedule((a_l1, tf.empty(Tensor[(16, 16), "f16", "tmem"])), op=T.ascend_l1_to_l0a())
            b_l0 = tf.schedule((b_l1, tf.empty(Tensor[(16, 16), "f16", "tmem"])), op=T.ascend_l1_to_l0b())
            acc0 = tf.zeros(Tensor[(16, 16), "f32", "tmem"])
            acc1 = tf.schedule((acc0, a_l0, b_l0), op=T.ascend_mmad())
            return tf.schedule((acc1, tf.empty(Tensor[(16, 16), "f32", "gmem"])), op=T.ascend_l0c_to_gm())
"""
    mod = _load_module(tmp_path, body, "cube.py")
    a = torch.randn(16, 16, dtype=torch.float16)
    b = torch.randn(16, 16, dtype=torch.float16)
    out = evaluate(mod.entry_function(), a, b).data
    ref = torch.matmul(a.float(), b.float())
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)

    lowered = _lower(mod)
    verify_prim_function(lowered.entry_function())


def test_vector_simtvf_flow_eval_analyze_finalize_and_verify(tmp_path: pathlib.Path):
    body = """
@module(entry="k", target=AscendTarget("ascend.950pr"), topologies=TOPOS)
class M:
    @func
    def k(x: Tensor[(8, 16), "f16"], w: Tensor[(8, 16), "f16"]) -> Tensor[(8, 16), "f16"]:
        with Mesh(("aiv",), (1,), names=("v",)) as _v:
            x_ub = tf.schedule((x, tf.empty(Tensor[(8, 16), "f16", "smem"])), op=T.ascend_gm_to_ub())
            w_ub = tf.schedule((w, tf.empty(Tensor[(8, 16), "f16", "smem"])), op=T.ascend_gm_to_ub())
            with T.SimdVF():
                xr = tf.schedule((x_ub, tf.empty(Tensor[(8, 16), "f16", "rmem"])), op=T.ascend_ub_to_reg())
                wr = tf.schedule((w_ub, tf.empty(Tensor[(8, 16), "f16", "rmem"])), op=T.ascend_ub_to_reg())
                xf = tf.schedule((xr, tf.empty(Tensor[(8, 16), "f32", "rmem"])), op=T.ascend_vcast(dtype="f32"))
                wf = tf.schedule((wr, tf.empty(Tensor[(8, 16), "f32", "rmem"])), op=T.ascend_vcast(dtype="f32"))
                prod = tf.schedule((xf, wf, tf.empty(Tensor[(8, 16), "f32", "rmem"])), op=T.ascend_vmul())
                ex = tf.schedule((prod, tf.empty(Tensor[(8, 16), "f32", "rmem"])), op=T.ascend_vexp())
                row_sum = tf.schedule((ex, tf.empty(Tensor[(8, 1), "f32", "rmem"])), op=T.ascend_vreduce_sum(axes=(-1,), keepdim=True))
                normed = tf.schedule((ex, row_sum, tf.empty(Tensor[(8, 16), "f32", "rmem"])), op=T.ascend_vdiv())
                y16 = tf.schedule((normed, tf.empty(Tensor[(8, 16), "f16", "rmem"])), op=T.ascend_vcast(dtype="f16"))
                y_ub = tf.schedule((y16, tf.empty(Tensor[(8, 16), "f16", "smem"])), op=T.ascend_reg_to_ub())
            return tf.schedule((y_ub, tf.empty(Tensor[(8, 16), "f16", "gmem"])), op=T.ascend_ub_to_gm())
"""
    mod = _load_module(tmp_path, body, "vec.py")
    x = torch.randn(8, 16, dtype=torch.float16)
    w = torch.randn(8, 16, dtype=torch.float16)
    out = evaluate(mod.entry_function(), x, w).data
    ref = torch.softmax(x.float() * w.float(), dim=-1).to(torch.float16)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)

    lowered = _lower(mod)
    verify_prim_function(lowered.entry_function())


def test_placement_and_escape_refusals(tmp_path: pathlib.Path):
    outside_vec = """
@module(entry="k", target=AscendTarget("ascend.950pr"), topologies=TOPOS)
class M:
    @func
    def k(x: Tensor[(8, 16), "f32"]) -> Tensor[(8, 16), "f32"]:
        with Mesh(("aiv",), (1,), names=("v",)) as _v:
            x_ub = tf.schedule((x, tf.empty(Tensor[(8, 16), "f32", "smem"])), op=T.ascend_gm_to_ub())
            xr = tf.schedule((x_ub, tf.empty(Tensor[(8, 16), "f32", "rmem"])), op=T.ascend_ub_to_reg())
            yr = tf.schedule((xr, xr, tf.empty(Tensor[(8, 16), "f32", "rmem"])), op=T.ascend_vadd())
            y_ub = tf.schedule((yr, tf.empty(Tensor[(8, 16), "f32", "smem"])), op=T.ascend_reg_to_ub())
            return tf.schedule((y_ub, tf.empty(Tensor[(8, 16), "f32", "gmem"])), op=T.ascend_ub_to_gm())
"""
    m1 = _load_module(tmp_path, outside_vec, "outside_vec.py")
    with pytest.raises(VerifyError):
        _lower(m1)

    dma_in_vf = """
@module(entry="k", target=AscendTarget("ascend.950pr"), topologies=TOPOS)
class M:
    @func
    def k(x: Tensor[(8, 16), "f32"]) -> Tensor[(8, 16), "f32"]:
        with Mesh(("aiv",), (1,), names=("v",)) as _v:
            with T.SimdVF():
                x_ub = tf.schedule((x, tf.empty(Tensor[(8, 16), "f32", "smem"])), op=T.ascend_gm_to_ub())
            return tf.schedule((x_ub, tf.empty(Tensor[(8, 16), "f32", "gmem"])), op=T.ascend_ub_to_gm())
"""
    with pytest.raises((VerifyError, LoweringError)):
        _lower(_load_module(tmp_path, dma_in_vf, "dma_in_vf.py"))

    wrong_mesh = """
@module(entry="k", target=AscendTarget("ascend.950pr"), topologies=TOPOS)
class M:
    @func
    def k(x: Tensor[(8, 16), "f32"]) -> Tensor[(8, 16), "f32"]:
        with Mesh(("aic",), (1,), names=("c",)) as _c:
            x_ub = tf.schedule((x, tf.empty(Tensor[(8, 16), "f32", "smem"])), op=T.ascend_gm_to_ub())
            with T.SimdVF():
                xr = tf.schedule((x_ub, tf.empty(Tensor[(8, 16), "f32", "rmem"])), op=T.ascend_ub_to_reg())
                y_ub = tf.schedule((xr, tf.empty(Tensor[(8, 16), "f32", "smem"])), op=T.ascend_reg_to_ub())
            return tf.schedule((y_ub, tf.empty(Tensor[(8, 16), "f32", "gmem"])), op=T.ascend_ub_to_gm())
"""
    with pytest.raises(VerifyError):
        _lower(_load_module(tmp_path, wrong_mesh, "wrong_mesh.py"))

    rmem_escape = """
@module(entry="k", target=AscendTarget("ascend.950pr"), topologies=TOPOS)
class M:
    @func
    def k(x: Tensor[(8, 16), "f32"]) -> Tensor[(8, 16), "f32"]:
        with Mesh(("aiv",), (1,), names=("v",)) as _v:
            x_ub = tf.schedule((x, tf.empty(Tensor[(8, 16), "f32", "smem"])), op=T.ascend_gm_to_ub())
            with T.SimdVF():
                xr = tf.schedule((x_ub, tf.empty(Tensor[(8, 16), "f32", "rmem"])), op=T.ascend_ub_to_reg())
                xr
            y_ub = tf.schedule((xr, tf.empty(Tensor[(8, 16), "f32", "smem"])), op=T.ascend_reg_to_ub())
            return tf.schedule((y_ub, tf.empty(Tensor[(8, 16), "f32", "gmem"])), op=T.ascend_ub_to_gm())
"""
    with pytest.raises(VerifyError):
        _lower(_load_module(tmp_path, rmem_escape, "rmem_escape.py"))


def test_schedule_facts_and_candidates_reports(tmp_path: pathlib.Path):
    unscheduled = """
@module(entry="k", target=AscendTarget("ascend.950pr"), topologies=TOPOS)
class M:
    @func
    def k(x: Tensor[(8, 16), "f16"]) -> Tensor[(8, 16), "f16"]:
        with Mesh(("aiv",), (1,), names=("v",)) as _v:
            x_ub = tf.reshard(x, layout=(8, 16), storage="smem")
            with T.SimdVF():
                xr = tf.reshard(x_ub, layout=(8, 16), storage="rmem")
                xf = tf.cast(xr, dtype="f32")
                xh = tf.cast(xf, dtype="f16")
                y_ub = tf.reshard(xh, layout=(8, 16), storage="smem")
            return tf.reshard(y_ub, layout=(8, 16), storage="gmem")
"""
    mod = _load_module(tmp_path, unscheduled, "unscheduled.py")
    target = mod.resolve_target()
    facts_data = facts_listing(target)
    assert "ascend.950pr" in str(facts_data["target"])
    fact_ids = {row["id"] for row in facts_data["instructions"]}
    assert {f"T.{spec.name}" for spec in ASCEND_OP_SPECS} <= fact_ids
    mmad_fact = facts_one(target, "T.ascend_mmad")
    assert mmad_fact["id"] == "T.ascend_mmad"

    checked_mod, checked_fn = check_program(mod, mod.entry_function())
    cand_data = report_candidates(checked_mod, checked_fn)
    defaults = {
        candidate["id"]
        for line in cand_data["lines"]
        for candidate in line["candidates"]
        if candidate.get("default")
    }
    assert "T.ascend_gm_to_ub" in defaults
    assert "T.ascend_ub_to_gm" in defaults
    assert "T.ascend_ub_to_reg" in defaults
    assert "T.ascend_reg_to_ub" in defaults
    assert "T.ascend_vcast" in defaults

    usable_ids = {
        candidate["id"]
        for line in cand_data["lines"]
        for candidate in line["candidates"]
    }
    assert "T.ascend_gm_to_ub_pad" in usable_ids
    assert "T.ascend_ub_to_gm_atomic" in usable_ids
    assert "T.ascend_vld_dist" in usable_ids
    assert "T.ascend_vsts_dist" in usable_ids


def test_unified_hir_lowering_inside_simdvf_without_schedule(tmp_path: pathlib.Path):
    """Verify standard HIR math/reduce/where/clamp ops lower automatically inside SimdVF."""
    body = """
@module(entry="k", target=AscendTarget("ascend.950pr"), topologies=TOPOS)
class M:
    @func
    def k(x: Tensor[(8, 16), "f32"], y: Tensor[(8, 16), "f32"]) -> Tensor[(8, 16), "f32"]:
        with Mesh(("aiv",), (1,), names=("v",)) as _v:
            x_ub = tf.reshard(x, layout=(8, 16), storage="smem")
            y_ub = tf.reshard(y, layout=(8, 16), storage="smem")
            two_ub = tf.full_like(y_ub, 2.0)
            with T.SimdVF():
                xr = tf.reshard(x_ub, layout=(8, 16), storage="rmem")
                yr = tf.reshard(y_ub, layout=(8, 16), storage="rmem")
                two_r = tf.reshard(two_ub, layout=(8, 16), storage="rmem")
                diff = xr - yr
                clamped = tf.clamp(diff + yr * two_r, min_val=-4.0, max_val=4.0)
                ex = tf.exp(clamped)
                row_sum = tf.reduce(ex, axes=(-1,), keepdim=True, kind=ReduceKind.SUM)
                normed = ex / row_sum
                mask = xr < yr
                out_r = tf.where(mask, normed, clamped)
                out_ub = tf.reshard(out_r, layout=(8, 16), storage="smem")
            return tf.reshard(out_ub, layout=(8, 16), storage="gmem")
"""
    mod = _load_module(tmp_path, body, "unified_hir.py")
    x = torch.randn(8, 16, dtype=torch.float32)
    y = torch.randn(8, 16, dtype=torch.float32)
    out = evaluate(mod.entry_function(), x, y).data
    clamped_ref = torch.clamp((x - y) + y * 2.0, min=-4.0, max=4.0)
    ex_ref = torch.exp(clamped_ref)
    normed_ref = ex_ref / ex_ref.sum(dim=-1, keepdim=True)
    ref = torch.where(x < y, normed_ref, clamped_ref)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)

    lowered = _lower(mod)
    verify_prim_function(lowered.entry_function())


def test_predicate_scalar_vector_and_granularity_enforcement(tmp_path: pathlib.Path):
    """Verify predicate, scalar-vector, lane-index, FMA ops and 32-bit predicate checks."""
    valid_body = """
@module(entry="k", target=AscendTarget("ascend.950pr"), topologies=TOPOS)
class M:
    @func
    def k(x: Tensor[(4, 16), "f32"]) -> Tensor[(4, 16), "f32"]:
        with Mesh(("aiv",), (1,), names=("v",)) as _v:
            x_ub = tf.schedule((x, tf.empty(Tensor[(4, 16), "f32", "smem"])), op=T.ascend_gm_to_ub())
            with T.SimdVF():
                xr = tf.schedule((x_ub, tf.empty(Tensor[(4, 16), "f32", "rmem"])), op=T.ascend_ub_to_reg())
                bias = tf.schedule((tf.empty(Tensor[(4, 16), "f32", "rmem"]),), op=T.ascend_vdup(value=1.0))
                idx = tf.schedule((tf.empty(Tensor[(4, 16), "f32", "rmem"]),), op=T.ascend_vci(start=0, order="INC_ORDER"))
                acc = tf.schedule((bias, xr, idx), op=T.ascend_vmula())
                scaled = tf.schedule((acc, tf.empty(Tensor[(4, 16), "f32", "rmem"])), op=T.ascend_vmuls(scalar=0.5))
                shifted = tf.schedule((scaled, tf.empty(Tensor[(4, 16), "f32", "rmem"])), op=T.ascend_vadds(scalar=2.0))
                tail_mask = tf.schedule((tf.empty(Tensor[(4, 16), "bool", "rmem"]),), op=T.ascend_update_mask(count=8, elem_width=32))
                cmp_mask = tf.schedule((xr, tf.empty(Tensor[(4, 16), "bool", "rmem"])), op=T.ascend_vcmps(scalar=0.0, cmp_op="gt", elem_width=32))
                comb = tf.schedule((tail_mask, cmp_mask, tf.empty(Tensor[(4, 16), "bool", "rmem"])), op=T.ascend_pand(elem_width=32))
                sel = tf.schedule((comb, shifted, bias, tf.empty(Tensor[(4, 16), "f32", "rmem"])), op=T.ascend_vsel())
                out_ub = tf.schedule((sel, tf.empty(Tensor[(4, 16), "f32", "smem"])), op=T.ascend_reg_to_ub())
            return tf.schedule((out_ub, tf.empty(Tensor[(4, 16), "f32", "gmem"])), op=T.ascend_ub_to_gm())
"""
    mod = _load_module(tmp_path, valid_body, "pred_valid.py")
    x = torch.randn(4, 16, dtype=torch.float32)
    out = evaluate(mod.entry_function(), x).data
    idx_ref = torch.arange(16, dtype=torch.float32).unsqueeze(0).expand(4, 16)
    shifted_ref = (1.0 + x * idx_ref) * 0.5 + 2.0
    tail_ref = (torch.arange(16) < 8).unsqueeze(0).expand(4, 16)
    comb_ref = tail_ref & (x > 0.0)
    ref = torch.where(comb_ref, shifted_ref, torch.ones_like(x))
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
    lowered = _lower(mod)
    verify_prim_function(lowered.entry_function())

    bad_width = """
@module(entry="k", target=AscendTarget("ascend.950pr"), topologies=TOPOS)
class M:
    @func
    def k(x: Tensor[(4, 16), "f32"]) -> Tensor[(4, 16), "f32"]:
        with Mesh(("aiv",), (1,), names=("v",)) as _v:
            x_ub = tf.schedule((x, tf.empty(Tensor[(4, 16), "f32", "smem"])), op=T.ascend_gm_to_ub())
            with T.SimdVF():
                xr = tf.schedule((x_ub, tf.empty(Tensor[(4, 16), "f32", "rmem"])), op=T.ascend_ub_to_reg())
                m16 = tf.schedule((tf.empty(Tensor[(4, 16), "bool", "rmem"]),), op=T.ascend_pset(elem_width=16, pat="PAT_ALL"))
                sel = tf.schedule((m16, xr, xr, tf.empty(Tensor[(4, 16), "f32", "rmem"])), op=T.ascend_vsel())
                out_ub = tf.schedule((sel, tf.empty(Tensor[(4, 16), "f32", "smem"])), op=T.ascend_reg_to_ub())
            return tf.schedule((out_ub, tf.empty(Tensor[(4, 16), "f32", "gmem"])), op=T.ascend_ub_to_gm())
"""
    with pytest.raises(VerifyError, match="element width 16"):
        _lower(_load_module(tmp_path, bad_width, "pred_bad.py"))


def test_reshard_four_extension_categories_eval_and_verify(tmp_path: pathlib.Path):
    """Verify dual_copy, layout/pad DMA, SimdVF dist, and atomic writeback Reshard ops."""
    body = """
@module(entry="k", target=AscendTarget("ascend.950pr"), topologies=TOPOS)
class M:
    @func
    def k(
        x: Tensor[(8, 12), "f32"],
        w: Tensor[(8, 16), "f32"],
        acc_gm: Tensor[(8, 16), "f32"],
    ) -> Tensor[(8, 16), "f32"]:
        with Mesh(("aiv",), (1,), names=("v",)) as _v:
            x_pad = tf.schedule(
                (x, tf.empty(Tensor[(8, 16), "f32", "smem"])),
                op=T.ascend_gm_to_ub_pad(pad_value=2.0),
            )
            w_l1 = tf.schedule(
                (w, tf.empty(Tensor[(8, 16), "f32", "tmem"])),
                op=T.ascend_gm_to_l1_nz(nz_mode="nd2nz", transpose=False),
            )
            w_ub = tf.schedule(
                (w_l1, tf.empty(Tensor[(8, 16), "f32", "smem"])),
                op=T.ascend_dual_copy_l0c_to_ub(split_mode="m_split"),
            )
            with T.SimdVF():
                xr = tf.schedule(
                    (x_pad, tf.empty(Tensor[(8, 16), "f32", "rmem"])),
                    op=T.ascend_vld_dist(dist="NORM_B32"),
                )
                wr = tf.schedule(
                    (w_ub, tf.empty(Tensor[(8, 16), "f32", "rmem"])),
                    op=T.ascend_vld_dist(dist="NORM_B32"),
                )
                one_ub = tf.schedule(
                    (xr, tf.empty(Tensor[(8, 1), "f32", "smem"])),
                    op=T.ascend_vsts_dist(dist="ONEPT_B32"),
                )
                one_brc = tf.schedule(
                    (one_ub, tf.empty(Tensor[(8, 16), "f32", "rmem"])),
                    op=T.ascend_vld_dist(dist="BRC_B32"),
                )
                y_r = (xr + wr) + one_brc
                y_ub = tf.schedule(
                    (y_r, tf.empty(Tensor[(8, 16), "f32", "smem"])),
                    op=T.ascend_vsts_dist(dist="NORM_B32"),
                )
            y_l1 = tf.schedule(
                (y_ub, tf.empty(Tensor[(8, 16), "f32", "tmem"])),
                op=T.ascend_dual_copy_ub_to_l1(nd2nz=True),
            )
            y_round = tf.schedule(
                (y_l1, tf.empty(Tensor[(8, 16), "f32", "smem"])),
                op=T.ascend_dual_copy_l0c_to_ub(split_mode="m_split"),
            )
            return tf.schedule(
                (y_round, acc_gm),
                op=T.ascend_ub_to_gm_atomic(atomic_op="add"),
            )
"""
    mod = _load_module(tmp_path, body, "reshard_ext.py")
    x = torch.randn(8, 12, dtype=torch.float32)
    w = torch.randn(8, 16, dtype=torch.float32)
    acc = torch.randn(8, 16, dtype=torch.float32)
    out = evaluate(mod.entry_function(), x, w, acc).data
    x_padded = torch.full((8, 16), 2.0, dtype=torch.float32)
    x_padded[:, :12] = x
    ref = acc + x_padded + w + x_padded[:, :1].expand(8, 16)
    torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
    lowered = _lower(mod)
    verify_prim_function(lowered.entry_function())

    transpose_and_l0c_atomic = """
@module(entry="k", target=AscendTarget("ascend.950pr"), topologies=TOPOS)
class M:
    @func
    def k(
        k_gm: Tensor[(8, 16), "f32"],
        acc_f16: Tensor[(16, 8), "f16"],
    ) -> Tensor[(16, 8), "f16"]:
        with Mesh(("aic",), (1,), names=("c",)) as _c:
            k_t = tf.schedule(
                (k_gm, tf.empty(Tensor[(16, 8), "f32", "tmem"])),
                op=T.ascend_gm_to_l1_nz(nz_mode="dn2nz", transpose=True),
            )
            return tf.schedule(
                (k_t, acc_f16),
                op=T.ascend_l0c_to_gm_atomic(atomic_op="add"),
            )
"""
    mod2 = _load_module(tmp_path, transpose_and_l0c_atomic, "reshard_l0c_atomic.py")
    k_gm = torch.randn(8, 16, dtype=torch.float32)
    acc_f16 = torch.randn(16, 8, dtype=torch.float16)
    out2 = evaluate(mod2.entry_function(), k_gm, acc_f16).data
    ref2 = acc_f16 + k_gm.transpose(0, 1).to(torch.float16)
    torch.testing.assert_close(out2, ref2, rtol=1e-3, atol=1e-3)
    lowered2 = _lower(mod2)
    verify_prim_function(lowered2.entry_function())
