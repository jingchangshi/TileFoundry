"""Tests for Ascend950PR Z3 modulo pipeline solving and passes."""

from __future__ import annotations

import pathlib

import pytest

from tilefoundry.analysis import analyze, check_program
from tilefoundry.cli.source import load_authored_ir
from tilefoundry.inspection import as_script
from tilefoundry.ir.hir.pipeline import BufferSchedule, PipelineError, PipelineSpec
from tilefoundry.schedule.finalize import finalize
from tilefoundry.vendor.tilelang_z3 import (
    Buffer,
    Capacity,
    Edge,
    Problem,
    Resource,
    Reuse,
    Task,
    verify_assignment,
)
from tilefoundry_ascend.ascend.schedule import (
    extract_tir_loop_problem,
    solve_ascend_pipeline,
    validate_pipeline_spec,
)
from tilefoundry_ascend.ascend.schedule.pipeline_solver import build_resolved_spec
from tilefoundry_ascend.ascend.target import AscendSpaceFacts, AscendTarget
from tilefoundry.ir.tir.stmts import For
from tilefoundry.ir.visitor import StmtVisitor, walk_prim_function


def _collect_pipelined_loops(fn) -> list[For]:
    loops: list[For] = []

    class _Visitor(StmtVisitor):
        def visit_For(self, stmt: For) -> None:
            if stmt.pipeline is not None:
                loops.append(stmt)
            self.generic_visit(stmt)

    walk_prim_function(_Visitor(), fn)
    return loops

HEADER = (
    "from __future__ import annotations\n\n"
    "from tilefoundry.dsl import *\n"
    "from tilefoundry.ir.core.kinds import BinaryKind\n"
    "from tilefoundry_ascend import AscendTarget\n"
    "TOPOS = (Topology('block', 1), Topology('aic', 1), "
    "Topology('aiv', 2), Topology('lane', 64))\n"
)


def _target() -> AscendTarget:
    return AscendTarget("ascend.950pr")


def _vector_resources() -> tuple[Resource, ...]:
    return (
        Resource(owner="aiv", core="aiv0", pipe="mte2", capacity=1),
        Resource(owner="aiv", core="aiv0", pipe="vector", capacity=1),
        Resource(owner="aiv", core="aiv0", pipe="mte3", capacity=1),
    )


def _mix_resources() -> tuple[Resource, ...]:
    return (
        *_vector_resources(),
        Resource(owner="aic", core="aic0", pipe="mte2", capacity=1),
        Resource(owner="aic", core="aic0", pipe="mte1", capacity=1),
        Resource(owner="aic", core="aic0", pipe="cube", capacity=1),
        Resource(owner="aic", core="aic0", pipe="fixpipe", capacity=1),
    )


def test_vector_ub_double_buffer_solves_within_capacity():
    facts = _target().get_facts(AscendSpaceFacts)
    problem = Problem(
        tasks=(
            Task(name="load", resources=(("aiv", "aiv0", "mte2", 1),), latency=1, stage=0),
            Task(name="vec", resources=(("aiv", "aiv0", "vector", 1),), latency=1, stage=1),
            Task(name="store", resources=(("aiv", "aiv0", "mte3", 1),), latency=1, stage=2),
        ),
        buffers=(
            Buffer(
                name="ub_in",
                slot_bytes=64 * 1024,
                space="UB",
                owner="aiv",
                auto_min=1,
                auto_max=2,
            ),
            Buffer(
                name="ub_out",
                slot_bytes=32 * 1024,
                space="UB",
                owner="aiv",
                auto_min=1,
                auto_max=2,
            ),
        ),
        edges=(
            Edge(src="load", dst="vec", distance=0, latency=1),
            Edge(src="vec", dst="store", distance=0, latency=1),
        ),
        reuses=(
            Reuse(buffer="ub_in", last_reader="vec", next_writer="load", delta=0, read_latency=1),
            Reuse(
                buffer="ub_out",
                last_reader="store",
                next_writer="vec",
                delta=0,
                read_latency=1,
            ),
        ),
        resources=_vector_resources(),
        capacities=(Capacity(space="UB", owner="aiv", bytes=facts.ub_usable_bytes),),
        max_stage=2,
        manual=True,
    )
    result = solve_ascend_pipeline(problem)
    assert verify_assignment(problem, result.assignment) == []
    assert result.buffer_versions["ub_in"] == 2
    assert result.buffer_versions["ub_out"] == 2
    assert result.task_stages == {"load": 0, "vec": 1, "store": 2}
    resolved = build_resolved_spec(
        PipelineSpec(max_stage=2, max_buffer_versions=2, enable_offset=True),
        result,
    )
    assert resolved.fixed_ii == result.assignment.ii


def test_cube_l1_l0_pipeline_solves():
    facts = _target().get_facts(AscendSpaceFacts)
    problem = Problem(
        tasks=(
            Task(name="gm_l1_a", resources=(("aic", "aic0", "mte2", 1),), latency=1, stage=0),
            Task(name="l1_l0a", resources=(("aic", "aic0", "mte1", 1),), latency=1, stage=1),
            Task(name="l1_l0b", resources=(("aic", "aic0", "mte1", 1),), latency=1, stage=1),
            Task(name="mmad", resources=(("aic", "aic0", "cube", 1),), latency=1, stage=2),
            Task(name="fix", resources=(("aic", "aic0", "fixpipe", 1),), latency=1, stage=3),
        ),
        buffers=(
            Buffer(
                name="l1_a",
                slot_bytes=64 * 1024,
                space="L1",
                owner="aic",
                auto_min=1,
                auto_max=2,
            ),
            Buffer(
                name="l0a_buf",
                slot_bytes=16 * 1024,
                space="L0A",
                owner="aic",
                auto_min=1,
                auto_max=2,
            ),
            Buffer(
                name="l0b_buf",
                slot_bytes=16 * 1024,
                space="L0B",
                owner="aic",
                auto_min=1,
                auto_max=2,
            ),
            Buffer(
                name="l0c_buf",
                slot_bytes=64 * 1024,
                space="L0C",
                owner="aic",
                fixed=1,
            ),
        ),
        edges=(
            Edge(src="gm_l1_a", dst="l1_l0a", distance=0, latency=1),
            Edge(src="l1_l0a", dst="mmad", distance=0, latency=1),
            Edge(src="l1_l0b", dst="mmad", distance=0, latency=1),
            Edge(src="mmad", dst="fix", distance=0, latency=1),
        ),
        reuses=(
            Reuse(
                buffer="l1_a",
                last_reader="l1_l0a",
                next_writer="gm_l1_a",
                delta=0,
                read_latency=1,
            ),
            Reuse(
                buffer="l0a_buf",
                last_reader="mmad",
                next_writer="l1_l0a",
                delta=0,
                read_latency=1,
            ),
            Reuse(
                buffer="l0b_buf",
                last_reader="mmad",
                next_writer="l1_l0b",
                delta=0,
                read_latency=1,
            ),
        ),
        resources=_mix_resources(),
        capacities=(
            Capacity(space="L1", owner="aic", bytes=facts.l1_bytes),
            Capacity(space="L0A", owner="aic", bytes=facts.l0a_bytes),
            Capacity(space="L0B", owner="aic", bytes=facts.l0b_bytes),
            Capacity(space="L0C", owner="aic", bytes=facts.l0c_bytes),
        ),
        max_stage=3,
        manual=True,
    )
    result = solve_ascend_pipeline(problem)
    assert verify_assignment(problem, result.assignment) == []
    assert 1 <= result.buffer_versions["l1_a"] <= 2
    assert 1 <= result.buffer_versions["l0a_buf"] <= 2
    assert result.buffer_versions["l0c_buf"] == 1


def test_mix_flashmla_pipeline_with_dual_copy():
    facts = _target().get_facts(AscendSpaceFacts)
    dual_res = (("aic", "aic0", "fixpipe", 1), ("aiv", "aiv0", "mte2", 1))
    problem = Problem(
        tasks=(
            Task(name="load_kv", resources=(("aic", "aic0", "mte2", 1),), latency=1, stage=0),
            Task(name="mmad_qk", resources=(("aic", "aic0", "cube", 1),), latency=1, stage=1),
            Task(name="dual_s", resources=dual_res, latency=1, stage=1),
            Task(name="softmax", resources=(("aiv", "aiv0", "vector", 1),), latency=1, stage=2),
            Task(name="dual_p", resources=dual_res, latency=1, stage=2),
            Task(name="mmad_pv", resources=(("aic", "aic0", "cube", 1),), latency=1, stage=3),
        ),
        buffers=(
            Buffer(
                name="l1_kv",
                slot_bytes=64 * 1024,
                space="L1",
                owner="aic",
                auto_min=1,
                auto_max=2,
            ),
            Buffer(
                name="l0c_s",
                slot_bytes=32 * 1024,
                space="L0C",
                owner="aic",
                auto_min=1,
                auto_max=2,
            ),
            Buffer(
                name="ub_s",
                slot_bytes=32 * 1024,
                space="UB",
                owner="aiv",
                auto_min=1,
                auto_max=2,
            ),
            Buffer(
                name="l1_p",
                slot_bytes=32 * 1024,
                space="L1",
                owner="aic",
                auto_min=1,
                auto_max=2,
            ),
        ),
        edges=(
            Edge(src="load_kv", dst="mmad_qk", distance=0, latency=1),
            Edge(src="mmad_qk", dst="dual_s", distance=0, latency=1),
            Edge(src="dual_s", dst="softmax", distance=0, latency=1),
            Edge(src="softmax", dst="dual_p", distance=0, latency=1),
            Edge(src="dual_p", dst="mmad_pv", distance=0, latency=1),
        ),
        reuses=(
            Reuse(buffer="ub_s", last_reader="softmax", next_writer="dual_s", delta=0, read_latency=1),
            Reuse(buffer="l1_p", last_reader="mmad_pv", next_writer="dual_p", delta=0, read_latency=1),
        ),
        resources=_mix_resources(),
        capacities=(
            Capacity(space="L1", owner="aic", bytes=facts.l1_bytes),
            Capacity(space="L0C", owner="aic", bytes=facts.l0c_bytes),
            Capacity(space="UB", owner="aiv", bytes=facts.ub_usable_bytes),
        ),
        max_stage=3,
        manual=True,
    )
    result = solve_ascend_pipeline(problem)
    assert verify_assignment(problem, result.assignment) == []
    assert result.assignment.ii >= 2
    assert 1 <= result.buffer_versions["ub_s"] <= 2
    assert 1 <= result.buffer_versions["l1_p"] <= 2


def test_over_capacity_ub_fixed_two_rejected():
    facts = _target().get_facts(AscendSpaceFacts)
    problem = Problem(
        tasks=(
            Task(name="load", resources=(("aiv", "aiv0", "mte2", 1),), latency=1, stage=0),
            Task(name="vec", resources=(("aiv", "aiv0", "vector", 1),), latency=1, stage=1),
        ),
        buffers=(
            Buffer(
                name="huge_ub",
                slot_bytes=128 * 1024,
                space="UB",
                owner="aiv",
                fixed=2,
            ),
        ),
        edges=(Edge(src="load", dst="vec", distance=0, latency=1),),
        reuses=(),
        resources=_vector_resources(),
        capacities=(Capacity(space="UB", owner="aiv", bytes=facts.ub_usable_bytes),),
        max_stage=1,
        manual=True,
    )
    with pytest.raises(PipelineError) as caught:
        solve_ascend_pipeline(problem)
    assert caught.value.code == "ASCEND_PIPELINE_UNSAT"


def test_validate_pipeline_spec_rejects_bad_space():
    bad = PipelineSpec(
        max_stage=2,
        max_buffer_versions=2,
        enable_offset=True,
        buffer_schedules=(BufferSchedule(root="x", versions=("fixed", 2), space="HBM"),),
    )
    with pytest.raises(PipelineError) as caught:
        validate_pipeline_spec(bad, _target())
    assert caught.value.code == "ASCEND_UNKNOWN_BUFFER_SPACE"


def test_end_to_end_hir_to_tir_pipeline_resolution(tmp_path: pathlib.Path):
    body = """
@module(entry="piped_add", target=AscendTarget("ascend.950pr"), topologies=TOPOS)
class M:
    @func
    def piped_add(
        a: Tensor[(4, 64), "f32"],
        b: Tensor[(4, 64), "f32"],
    ) -> Tensor[(4, 64), "f32"]:
        with Mesh(("aiv",), (1,), names=("v",)) as _v:
            c = tf.empty(Tensor[(4, 64), "f32", "gmem"])
            for k in tf.pipelined(4, max_stage=2, max_buffer_versions=2, enable_offset=True):
                a_ub = tf.schedule((a[k, :], tf.empty(Tensor[(64,), "f32", "smem"])), op=T.ascend_gm_to_ub())
                b_ub = tf.schedule((b[k, :], tf.empty(Tensor[(64,), "f32", "smem"])), op=T.ascend_gm_to_ub())
                with T.SimdVF():
                    ar = tf.schedule((a_ub, tf.empty(Tensor[(64,), "f32", "rmem"])), op=T.ascend_ub_to_reg())
                    br = tf.schedule((b_ub, tf.empty(Tensor[(64,), "f32", "rmem"])), op=T.ascend_ub_to_reg())
                    cr = tf.schedule((ar, br, tf.empty(Tensor[(64,), "f32", "rmem"])), op=T.ascend_vbinary(kind="add"))
                    c_ub = tf.schedule((cr, tf.empty(Tensor[(64,), "f32", "smem"])), op=T.ascend_reg_to_ub())
                c_row = tf.schedule((c_ub, tf.empty(Tensor[(64,), "f32", "gmem"])), op=T.ascend_ub_to_gm())
                c = tf.insert_slice(c, tf.reshape(c_row, new_shape=(1, 64)), (k, 0))
            return c
"""
    path = tmp_path / "piped_add.py"
    path.write_text(HEADER + "\n" + body, encoding="utf-8")
    mod = load_authored_ir(f"{path}:M")
    authored = mod.entry_function()
    checked_mod, checked_fn = check_program(mod, authored)
    analyzed = analyze(checked_mod, checked_fn, analysis="memory")
    lowered = finalize(analyzed, authored=authored)

    pipelines = lowered.metadata.get("ascend.pipelines")
    assert isinstance(pipelines, dict) and "piped_add:k" in pipelines
    record = pipelines["piped_add:k"]
    assert record["ii"] >= 1
    assert any(v == 2 for v in record["versions"].values())
    printed = as_script(lowered)
    assert "tf.pipelined(" in printed
    assert f"fixed_ii={record['ii']}" in printed
    assert any(edge_task.startswith("t") and "simdvf" in edge_task for edge_task in record["stages"])


def test_loop_carried_accumulator_emits_distance_one_and_pins_version(tmp_path: pathlib.Path):
    body = """
@module(entry="accum_loop", target=AscendTarget("ascend.950pr"), topologies=TOPOS)
class M:
    @func
    def accum_loop(v: Tensor[(4, 64, 128), "f32"]) -> Tensor[(64, 128), "f32"]:
        with Mesh(("aiv",), (1,), names=("v",)) as _v:
            acc_ub = tf.zeros(Tensor[(64, 128), "f32", "smem"])
            for k in tf.pipelined(4, max_stage=2, max_buffer_versions=2, enable_offset=True):
                v_tile = tf.reshape(v[k, :, :], new_shape=(64, 128))
                v_ub = tf.schedule((v_tile, tf.empty(Tensor[(64, 128), "f32", "smem"])), op=T.ascend_gm_to_ub())
                with T.SimdVF():
                    vr = tf.schedule((v_ub, tf.empty(Tensor[(64, 128), "f32", "rmem"])), op=T.ascend_ub_to_reg())
                    ar = tf.schedule((acc_ub, tf.empty(Tensor[(64, 128), "f32", "rmem"])), op=T.ascend_ub_to_reg())
                    sr = tf.schedule((ar, vr, tf.empty(Tensor[(64, 128), "f32", "rmem"])), op=T.ascend_vbinary(kind="add"))
                    acc_ub = tf.schedule((sr, tf.empty(Tensor[(64, 128), "f32", "smem"])), op=T.ascend_reg_to_ub())
            return tf.schedule((acc_ub, tf.empty(Tensor[(64, 128), "f32", "gmem"])), op=T.ascend_ub_to_gm())
"""
    path = tmp_path / "accum_loop.py"
    path.write_text(HEADER + "\n" + body, encoding="utf-8")
    mod = load_authored_ir(f"{path}:M")
    authored = mod.entry_function()
    checked_mod, checked_fn = check_program(mod, authored)
    analyzed = analyze(checked_mod, checked_fn, analysis="memory")
    lowered = finalize(analyzed, authored=authored)

    fn = lowered.entry_function()
    loops = _collect_pipelined_loops(fn)
    assert len(loops) == 1
    problem, _ = extract_tir_loop_problem(loops[0], fn_stmt=fn.body)
    d1_edges = [e for e in problem.edges if e.distance == 1]
    assert d1_edges, "expected loop-carried distance=1 edge on acc_ub"
    simdvf_task = next(t for t in problem.tasks if "simdvf" in t.name)
    assert simdvf_task.latency > 1
    assert any(e.src == simdvf_task.name and e.dst == simdvf_task.name and e.latency == simdvf_task.latency for e in d1_edges)

    pipelines = lowered.metadata["ascend.pipelines"]
    rec = pipelines["accum_loop:k"]
    assert rec["ii"] >= simdvf_task.latency
    assert rec["versions"]["acc_ub"] == 1
    scratch_versions = [v for k, v in rec["versions"].items() if k != "acc_ub"]
    assert any(v == 2 for v in scratch_versions)


def test_aiv2_mesh_splits_ub_footprint_and_cost_model_scales(tmp_path: pathlib.Path):
    """Verify 2-AIV mesh splits UB footprint (153600 B / 2 = 76800 B) and scales latency."""
    body = """
@module(entry="aiv2_add", target=AscendTarget("ascend.950pr"), topologies=TOPOS)
class M:
    @func
    def aiv2_add(x: Tensor[(4, 128, 300), "f32"]) -> Tensor[(4, 128, 300), "f32"]:
        with Mesh(("aiv",), (2,), names=("v",)) as _v:
            out = tf.empty(Tensor[(4, 128, 300), "f32", "gmem"])
            for k in tf.pipelined(4, max_stage=2, max_buffer_versions=2, enable_offset=True):
                xt = tf.reshape(x[k, :, :], new_shape=(128, 300))
                x_ub = tf.schedule((xt, tf.empty(Tensor[(128, 300), "f32", "smem"])), op=T.ascend_gm_to_ub())
                with T.SimdVF():
                    xr = tf.schedule((x_ub, tf.empty(Tensor[(128, 300), "f32", "rmem"])), op=T.ascend_ub_to_reg())
                    yr = tf.schedule((xr, xr, tf.empty(Tensor[(128, 300), "f32", "rmem"])), op=T.ascend_vbinary(kind="add"))
                    y_ub = tf.schedule((yr, tf.empty(Tensor[(128, 300), "f32", "smem"])), op=T.ascend_reg_to_ub())
                row_gm = tf.schedule((y_ub, tf.empty(Tensor[(128, 300), "f32", "gmem"])), op=T.ascend_ub_to_gm())
                out = tf.insert_slice(out, tf.reshape(row_gm, new_shape=(1, 128, 300)), (k, 0, 0))
            return out
"""
    path = tmp_path / "aiv2_add.py"
    path.write_text(HEADER + "\n" + body, encoding="utf-8")
    mod = load_authored_ir(f"{path}:M")
    authored = mod.entry_function()
    checked_mod, checked_fn = check_program(mod, authored)
    analyzed = analyze(checked_mod, checked_fn, analysis="memory")
    lowered = finalize(analyzed, authored=authored)

    fn = lowered.entry_function()
    loops = _collect_pipelined_loops(fn)
    problem, _ = extract_tir_loop_problem(loops[0], fn_stmt=fn.body)
    ub_bufs = [b for b in problem.buffers if b.space == "UB"]
    assert ub_bufs and all(b.slot_bytes == 76800 for b in ub_bufs)
    dma_in = next(t for t in problem.tasks if "gm_to_ub" in t.name)
    assert dma_in.latency == 12


def test_mode_aware_ub_budget_simdvf_vs_simtvf_and_unreserved(tmp_path: pathlib.Path):
    """Verify pure SimdVF (248/256 KiB) admits 240 KiB schedules that SimtVF rejects."""
    pure_simdvf_240k = """
@module(entry="pure_simdvf", target=AscendTarget("ascend.950pr"), topologies=TOPOS)
class M:
    @func
    def pure_simdvf(x: Tensor[(4, 64, 320), "f32"]) -> Tensor[(4, 64, 320), "f32"]:
        with Mesh(("aiv",), (1,), names=("v",)) as _v:
            out = tf.empty(Tensor[(4, 64, 320), "f32", "gmem"])
            for k in tf.pipelined(4, max_stage=2, max_buffer_versions=2, enable_offset=True):
                xt = tf.reshape(x[k, :, :], new_shape=(64, 320))
                x_ub = tf.schedule((xt, tf.empty(Tensor[(64, 320), "f32", "smem"])), op=T.ascend_gm_to_ub())
                with T.SimdVF():
                    xr = tf.schedule((x_ub, tf.empty(Tensor[(64, 320), "f32", "rmem"])), op=T.ascend_ub_to_reg())
                    y_ub = tf.schedule((xr, tf.empty(Tensor[(64, 320), "f32", "smem"])), op=T.ascend_reg_to_ub())
                row_gm = tf.schedule((y_ub, tf.empty(Tensor[(64, 320), "f32", "gmem"])), op=T.ascend_ub_to_gm())
                out = tf.insert_slice(out, tf.reshape(row_gm, new_shape=(1, 64, 320)), (k, 0, 0))
            return out
"""
    p1 = tmp_path / "pure_simdvf.py"
    p1.write_text(HEADER + "\n" + pure_simdvf_240k, encoding="utf-8")
    mod1 = load_authored_ir(f"{p1}:M")
    c_mod1, c_fn1 = check_program(mod1, mod1.entry_function())
    low1 = finalize(analyze(c_mod1, c_fn1, analysis="memory"), authored=mod1.entry_function())
    fn1 = low1.entry_function()
    prob1, _ = extract_tir_loop_problem(_collect_pipelined_loops(fn1)[0], fn_stmt=fn1.body)
    assert {(c.space, c.owner): c.bytes for c in prob1.capacities}[("UB", "aiv")] == 248 * 1024
    versions1 = list(low1.metadata["ascend.pipelines"]["pure_simdvf:k"]["versions"].values())
    assert sorted(versions1) == [1, 2]

    import dataclasses  # noqa: PLC0415
    from tilefoundry.ir.tir.stmts import ExecutionScope, Sequential  # noqa: PLC0415

    def _with_simtvf(node):
        if isinstance(node, ExecutionScope):
            return dataclasses.replace(node, kind="ascend.simtvf", body=_with_simtvf(node.body))
        if isinstance(node, Sequential):
            return dataclasses.replace(node, body=tuple(_with_simtvf(s) for s in node.body))
        if dataclasses.is_dataclass(node) and hasattr(node, "body"):
            return dataclasses.replace(node, body=_with_simtvf(node.body))
        return node

    loop1 = _collect_pipelined_loops(fn1)[0]
    simtvf_loop = _with_simtvf(loop1)
    simtvf_fn_body = _with_simtvf(fn1.body)
    prob2, _ = extract_tir_loop_problem(simtvf_loop, fn_stmt=simtvf_fn_body)
    assert {(c.space, c.owner): c.bytes for c in prob2.capacities}[("UB", "aiv")] == 216 * 1024
    with pytest.raises(PipelineError, match="ASCEND_PIPELINE_UNSAT"):
        solve_ascend_pipeline(prob2)

    prob_unres, _ = extract_tir_loop_problem(
        _collect_pipelined_loops(fn1)[0], fn_stmt=fn1.body, unreserved_ubuf=True
    )
    assert {(c.space, c.owner): c.bytes for c in prob_unres.capacities}[("UB", "aiv")] == 256 * 1024


def test_sil_policy_override_shadow_dump_and_automatic_fallback(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify SIL policy overrides (L1/L0C/UB-256KB), shadow_dump_path, and automatic fallback."""
    import json  # noqa: PLC0415
    from tilefoundry_ascend.ascend.schedule.pipeline_solver import (  # noqa: PLC0415
        _op_resource_mask,
    )
    from tilefoundry_ascend.ascend.ir.tir.instructions import AscendDualCopyL0cToUb  # noqa: PLC0415
    from tilefoundry_ascend.ascend.target import NodeResourceMask  # noqa: PLC0415

    monkeypatch.chdir(tmp_path)

    mask = _op_resource_mask(AscendDualCopyL0cToUb(), (None, None))
    assert mask == NodeResourceMask.for_aic("FIXPIPE")

    facts = _target().get_facts(AscendSpaceFacts)
    problem = Problem(
        tasks=(
            Task(name="load", resources=(("aiv", "aiv0", "mte2", 1),), latency=1, stage=0),
            Task(name="vec", resources=(("aiv", "aiv0", "vector", 1),), latency=1, stage=1),
        ),
        buffers=(
            Buffer(name="ub_big", slot_bytes=252 * 1024, space="UB", owner="aiv", fixed=1),
            Buffer(name="l1_buf", slot_bytes=64 * 1024, space="L1", owner="aic", auto_min=1, auto_max=2),
            Buffer(name="l0c_buf", slot_bytes=32 * 1024, space="L0C", owner="aic", auto_min=1, auto_max=2),
        ),
        edges=(Edge(src="load", dst="vec", distance=0, latency=1),),
        reuses=(
            Reuse(buffer="ub_big", last_reader="vec", next_writer="load", delta=0, read_latency=1),
            Reuse(buffer="l1_buf", last_reader="vec", next_writer="load", delta=0, read_latency=1),
        ),
        resources=_vector_resources(),
        capacities=(
            Capacity(space="UB", owner="aiv", bytes=facts.ub_simdvf_usable_bytes),
            Capacity(space="L1", owner="aic", bytes=facts.l1_bytes),
            Capacity(space="L0C", owner="aic", bytes=facts.l0c_bytes),
        ),
        max_stage=1,
        manual=True,
    )
    shadow_file = tmp_path / "shadow" / "solved.json"
    res = solve_ascend_pipeline(
        problem,
        sil_policy={"ub_256kb_unlock": True, "l1_stages": 2, "l0c_buffers": 2},
        shadow_dump_path=shadow_file,
    )
    assert res.fallback_used is False
    assert res.buffer_versions["l1_buf"] == 2
    assert res.buffer_versions["l0c_buf"] == 2
    assert shadow_file.is_file()
    shadow_data = json.loads(shadow_file.read_text(encoding="utf-8"))
    assert shadow_data["ii"] == res.assignment.ii
    assert shadow_data["fallback_used"] is False
    assert not pathlib.Path("envelope.json").exists()

    fallback_prob = Problem(
        tasks=(
            Task(name="load", resources=(("aiv", "aiv0", "mte2", 1),), latency=1, stage=0),
            Task(name="vec", resources=(("aiv", "aiv0", "vector", 1),), latency=1, stage=1),
        ),
        buffers=(
            Buffer(name="l1_buf", slot_bytes=128 * 1024, space="L1", owner="aic", auto_min=1, auto_max=2),
        ),
        edges=(Edge(src="load", dst="vec", distance=0, latency=1),),
        reuses=(Reuse(buffer="l1_buf", last_reader="vec", next_writer="load", delta=0, read_latency=1),),
        resources=_vector_resources(),
        capacities=(Capacity(space="L1", owner="aic", bytes=facts.l1_bytes),),
        max_stage=1,
        manual=True,
    )
    fb_res = solve_ascend_pipeline(fallback_prob, sil_policy={"l1_stages": 8})
    assert fb_res.fallback_used is True
    assert "UNSAT" in fb_res.fallback_reason or "CAPACITY" in fb_res.fallback_reason
    assert 1 <= fb_res.buffer_versions["l1_buf"] <= 2


def test_sil_l1_slot_aliasing_saves_capacity_and_rejects_overlap():
    """Verify MLA K_rope -> P L1 slot aliasing rescues capacity and falls back on overlap."""
    facts = _target().get_facts(AscendSpaceFacts)
    problem = Problem(
        tasks=(
            Task(name="load_k_rope", resources=(("aic", "aic0", "mte2", 1),), latency=1, stage=0),
            Task(name="mmad_qk", resources=(("aic", "aic0", "cube", 1),), latency=1, stage=0),
            Task(name="dual_p", resources=(("aic", "aic0", "fixpipe", 1),), latency=1, stage=0),
            Task(name="mmad_pv", resources=(("aic", "aic0", "cube", 1),), latency=1, stage=0),
        ),
        buffers=(
            Buffer(name="l1_k_rope", slot_bytes=256 * 1024, space="L1", owner="aic", fixed=1),
            Buffer(name="l1_p", slot_bytes=256 * 1024, space="L1", owner="aic", fixed=1),
        ),
        edges=(
            Edge(src="load_k_rope", dst="mmad_qk", distance=0, latency=1),
            Edge(src="mmad_qk", dst="dual_p", distance=0, latency=1),
            Edge(src="dual_p", dst="mmad_pv", distance=0, latency=1),
        ),
        reuses=(
            Reuse(buffer="l1_k_rope", last_reader="mmad_qk", next_writer="load_k_rope", delta=0, read_latency=1),
            Reuse(buffer="l1_p", last_reader="mmad_pv", next_writer="dual_p", delta=0, read_latency=1),
        ),
        resources=_mix_resources(),
        capacities=(Capacity(space="L1", owner="aic", bytes=300 * 1024),),
        max_stage=0,
        manual=False,
    )
    with pytest.raises(PipelineError, match="ASCEND_PIPELINE_UNSAT"):
        solve_ascend_pipeline(problem)

    res = solve_ascend_pipeline(problem, sil_policy={"l1_slot_aliasing": {"l1_p": "l1_k_rope"}})
    assert res.fallback_used is False
    assert verify_assignment(res.problem, res.assignment) == []
    slot_map = {b.name: b.slot_bytes for b in res.problem.buffers}
    assert slot_map["l1_k_rope"] == 256 * 1024
    assert slot_map["l1_p"] == 0

    bad_prob = Problem(
        tasks=(
            Task(name="load_k_rope", resources=(("aic", "aic0", "mte2", 1),), latency=1, stage=0),
            Task(name="dual_p", resources=(("aic", "aic0", "fixpipe", 1),), latency=1, stage=0),
            Task(name="mmad_qk", resources=(("aic", "aic0", "cube", 1),), latency=1, stage=2),
            Task(name="mmad_pv", resources=(("aic", "aic0", "cube", 1),), latency=1, stage=3),
        ),
        buffers=(
            Buffer(name="l1_k_rope", slot_bytes=64 * 1024, space="L1", owner="aic", auto_min=1, auto_max=4),
            Buffer(name="l1_p", slot_bytes=64 * 1024, space="L1", owner="aic", auto_min=1, auto_max=4),
        ),
        edges=(
            Edge(src="load_k_rope", dst="mmad_qk", distance=0, latency=1),
            Edge(src="dual_p", dst="mmad_pv", distance=0, latency=1),
        ),
        reuses=(
            Reuse(buffer="l1_k_rope", last_reader="mmad_qk", next_writer="load_k_rope", delta=0, read_latency=1),
            Reuse(buffer="l1_p", last_reader="mmad_pv", next_writer="dual_p", delta=0, read_latency=1),
        ),
        resources=_mix_resources(),
        capacities=(Capacity(space="L1", owner="aic", bytes=facts.l1_bytes),),
        max_stage=3,
        manual=True,
    )
    bad_res = solve_ascend_pipeline(bad_prob, sil_policy={"l1_slot_aliasing": {"l1_p": "l1_k_rope"}})
    assert bad_res.fallback_used is True


def test_extract_tir_loop_problem_dynamic_fallback_with_dim_hints():
    """Verify allow_dynamic_fallback + dim_hints evaluates symbolic DimVar buffer shapes."""
    from tilefoundry.ir.core import Call, Constant, Var  # noqa: PLC0415
    from tilefoundry.ir.tir.memory import AllocTensor  # noqa: PLC0415
    from tilefoundry.ir.tir.stmts import Evaluate, For, LetStmt, Sequential  # noqa: PLC0415
    from tilefoundry.ir.types import DType, StorageKind, TensorType  # noqa: PLC0415
    from tilefoundry.ir.types.dim import DimVar  # noqa: PLC0415
    from tilefoundry_ascend.ascend.ir.tir.instructions import (  # noqa: PLC0415
        AscendGmToUb,
        AscendUbToGm,
    )

    d_dyn = DimVar("N_dyn", 16, 512)
    gm_ty = TensorType((4, d_dyn), DType.f16, None, StorageKind.GMEM)
    ub_ty = TensorType((d_dyn,), DType.f16, None, StorageKind.SMEM)
    sc = TensorType.umat_scalar()
    k = Var("k", type=sc)
    gm_in = Var("gm_in", type=gm_ty)
    gm_out = Var("gm_out", type=gm_ty)
    ub_var = Var("ub_buf", type=ub_ty)

    alloc = LetStmt(ub_var, Call(AllocTensor(tensor_type=ub_ty), (), type=ub_ty))
    load = Evaluate(AscendGmToUb(), (gm_in, ub_var))
    store = Evaluate(AscendUbToGm(), (ub_var, gm_out))
    loop = For(
        k,
        Constant(0, type=sc),
        Constant(4, type=sc),
        Constant(1, type=sc),
        Sequential((alloc, load, store)),
        pipeline=PipelineSpec(max_stage=1, max_buffer_versions=2, enable_offset=True),
    )

    with pytest.raises(PipelineError, match="ASCEND_DYNAMIC_PIPELINE_BUFFER"):
        extract_tir_loop_problem(loop)

    problem, _ = extract_tir_loop_problem(
        loop,
        dim_hints={"N_dyn": 128},
        allow_dynamic_fallback=True,
    )
    buf = next(b for b in problem.buffers if b.name == "ub_buf")
    assert buf.slot_bytes == 256
    res = solve_ascend_pipeline(problem)
    assert res.assignment.ii >= 1
