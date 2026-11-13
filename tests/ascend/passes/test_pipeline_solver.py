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
    solve_ascend_pipeline,
    validate_pipeline_spec,
)
from tilefoundry_ascend.ascend.schedule.pipeline_solver import build_resolved_spec
from tilefoundry_ascend.ascend.target import AscendSpaceFacts, AscendTarget

HEADER = (
    "from __future__ import annotations\n\n"
    "from tilefoundry.dsl import *\n"
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
                with T.SimtVF():
                    ar = tf.schedule((a_ub, tf.empty(Tensor[(64,), "f32", "rmem"])), op=T.ascend_ub_to_reg())
                    br = tf.schedule((b_ub, tf.empty(Tensor[(64,), "f32", "rmem"])), op=T.ascend_ub_to_reg())
                    cr = tf.schedule((ar, br, tf.empty(Tensor[(64,), "f32", "rmem"])), op=T.ascend_vadd())
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
