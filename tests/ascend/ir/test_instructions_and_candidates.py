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
from tilefoundry.ir.hir.nn.matmul import MatMul
from tilefoundry.ir.hir.sharding.reshard import Reshard
from tilefoundry.ir.hir.tensor.cast import Cast
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
from tilefoundry_ascend.ascend.ir.execution import SIMTVF_KIND, SimtVF
from tilefoundry_ascend.ascend.ir.tir.instructions import (
    ASCEND_OP_SPECS,
    AscendGmToL1,
    AscendGmToUb,
    AscendL0cToGm,
    AscendMmad,
    AscendRegToUb,
    AscendUbToGm,
    AscendUbToReg,
    AscendVcast,
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


def test_simtvf_execution_kind_registration():
    assert SIMTVF_KIND.name == "ascend.simtvf"
    assert SIMTVF_KIND.surface == "SimtVF"
    assert SIMTVF_KIND.required_topologies == ("aiv",)
    assert SIMTVF_KIND.mask_width_bits == 32
    assert SIMTVF_KIND.conversion_mask_width_bits == 16
    assert SIMTVF_KIND.lane_count is None
    assert SIMTVF_KIND.register_storages == (StorageKind.RMEM, StorageKind.CARRIER)
    assert SIMTVF_KIND.element_dtypes == (DType.f32,)
    assert SimtVF.execution_kind == SIMTVF_KIND
    assert execution_kind_for_surface("SimtVF", dialect="T") == SIMTVF_KIND
    assert SIMTVF_KIND in declared_kinds()


def test_ascend_ops_registered_with_target_and_semantics():
    target = AscendTarget("ascend.950pr")
    supported = {op_cls for op_cls, _ in supported_op_capabilities(target)}
    assert len(ASCEND_OP_SPECS) == 20
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

    ur_call = Call(Reshard(), (Var("x", type=smem),), type=rmem_f16)
    selected_ur = select_instruction(
        ur_call, target=target, inside=True, active_kind="ascend.simtvf"
    )
    assert type(selected_ur) is AscendUbToReg
    ru_call = Call(Reshard(), (Var("x", type=rmem_f16),), type=smem)
    selected_ru = select_instruction(
        ru_call, target=target, inside=True, active_kind="ascend.simtvf"
    )
    assert type(selected_ru) is AscendRegToUb

    cast_call = Call(Cast(dtype=DType.f32), (Var("x", type=rmem_f16),), type=rmem_f32)
    selected_cast = select_instruction(
        cast_call, target=target, inside=True, active_kind="ascend.simtvf"
    )
    assert type(selected_cast) is AscendVcast

    mm_call = Call(MatMul(), (Var("a", type=tmem), Var("b", type=tmem)), type=tmem)
    assert type(select_instruction(mm_call, target=target, inside=False)) is AscendMmad


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
            with T.SimtVF():
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
            with T.SimtVF():
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
            with T.SimtVF():
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
            with T.SimtVF():
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
            with T.SimtVF():
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
