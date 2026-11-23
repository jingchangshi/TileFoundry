"""Tests for Ascend950PR Z3 modulo pipeline solving."""

from __future__ import annotations

import concurrent.futures
import json
import pathlib

import pytest
import tilelang.ascend.language as T

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
    PipelineError,
    inspect_tilelang_primfunc,
    read_from_sil,
    solve_ascend_pipeline,
    solve_pipeline_with_monotone_backoff,
    task_resource_mask,
)
from tilefoundry_ascend.ascend.target import AscendSpaceFacts, AscendTarget, NodeResourceMask


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


def test_loop_carried_accumulator_distance_one_pins_version():
    facts = _target().get_facts(AscendSpaceFacts)
    problem = Problem(
        tasks=(
            Task(name="load_v", resources=(("aiv", "aiv0", "mte2", 1),), latency=2, stage=0),
            Task(name="simdvf_add", resources=(("aiv", "aiv0", "vector", 1),), latency=3, stage=1),
        ),
        buffers=(
            Buffer(
                name="v_ub",
                slot_bytes=16 * 1024,
                space="UB",
                owner="aiv",
                auto_min=1,
                auto_max=2,
            ),
            Buffer(
                name="acc_ub",
                slot_bytes=16 * 1024,
                space="UB",
                owner="aiv",
                fixed=1,
            ),
        ),
        edges=(
            Edge(src="load_v", dst="simdvf_add", distance=0, latency=2),
            Edge(src="simdvf_add", dst="simdvf_add", distance=1, latency=3),
        ),
        reuses=(
            Reuse(
                buffer="v_ub",
                last_reader="simdvf_add",
                next_writer="load_v",
                delta=0,
                read_latency=1,
            ),
        ),
        resources=_vector_resources(),
        capacities=(Capacity(space="UB", owner="aiv", bytes=facts.ub_simdvf_usable_bytes),),
        max_stage=1,
        manual=True,
    )
    res = solve_ascend_pipeline(problem)
    assert verify_assignment(res.problem, res.assignment) == []
    assert res.assignment.ii >= 3
    assert res.buffer_versions["acc_ub"] == 1
    assert res.buffer_versions["v_ub"] == 2


def test_tilelang_pipelined_and_mode_aware_ub_budgets():
    @T.prim_func
    def piped_add(
        a: T.Tensor((4, 64), "float32"),
        b: T.Tensor((4, 64), "float32"),
        c: T.Tensor((4, 64), "float32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((1, 64), "float32")
            b_ub = T.alloc_shared((1, 64), "float32")
            c_ub = T.alloc_shared((1, 64), "float32")
            for k in T.Pipelined(4, num_stages=2):
                T.copy(a[k : k + 1, 0:64], a_ub)
                T.copy(b[k : k + 1, 0:64], b_ub)
                with T.SimdVF():
                    m32 = T.simd.pset(32, "PAT_ALL")
                    va = T.simd.vld(a_ub[0, 0], dist="NORM_B32")
                    vb = T.simd.vld(b_ub[0, 0], dist="NORM_B32")
                    T.simd.vsts(c_ub[0, 0], T.simd.vadd(va, vb, mask=m32), mask=m32, dist="NORM_B32")
                T.copy(c_ub, c[k : k + 1, 0:64])

    insp = inspect_tilelang_primfunc(piped_add)
    assert insp.ledger.analysis_mode == "exact_modulo_z3"
    assert insp.ledger.steady_state_ii is not None and insp.ledger.steady_state_ii >= 1
    assert insp.ledger.waterlines["UB"].overflow_bytes == 0

    facts = _target().get_facts(AscendSpaceFacts)
    tasks = (
        Task(name="load", resources=(("aiv", "aiv0", "mte2", 1),), latency=1, stage=0),
        Task(name="vec", resources=(("aiv", "aiv0", "vector", 1),), latency=1, stage=1),
    )
    buffers = (Buffer(name="x_ub", slot_bytes=240 * 1024, space="UB", owner="aiv", fixed=1),)
    edges = (Edge(src="load", dst="vec", distance=0, latency=1),)
    reuses = (Reuse(buffer="x_ub", last_reader="vec", next_writer="load", delta=0, read_latency=1),)

    simdvf_prob = Problem(
        tasks=tasks,
        buffers=buffers,
        edges=edges,
        reuses=reuses,
        resources=_vector_resources(),
        capacities=(Capacity(space="UB", owner="aiv", bytes=facts.ub_limit_bytes("simdvf")),),
        max_stage=1,
        manual=True,
    )
    assert solve_ascend_pipeline(simdvf_prob).assignment.ii >= 1

    simtvf_prob = Problem(
        tasks=tasks,
        buffers=buffers,
        edges=edges,
        reuses=reuses,
        resources=_vector_resources(),
        capacities=(Capacity(space="UB", owner="aiv", bytes=facts.ub_limit_bytes("simtvf")),),
        max_stage=1,
        manual=True,
    )
    with pytest.raises(PipelineError, match="ASCEND_PIPELINE_UNSAT"):
        solve_ascend_pipeline(simtvf_prob)


def test_sil_policy_override_shadow_dump_and_automatic_fallback(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Verify SIL policy overrides (L1/L0C/UB-256KB), shadow_dump_path, and automatic fallback."""
    monkeypatch.chdir(tmp_path)

    dual_task = Task(
        name="dual_l0c_to_ub",
        resources=(("aic", "aic", "fixpipe", 1, 0), ("aiv", "aiv0", "mte2", 1, 0)),
        latency=1,
        stage=0,
    )
    mask = task_resource_mask(dual_task)
    assert mask == (NodeResourceMask.for_aic("FIXPIPE") | NodeResourceMask.for_aiv0("MTE2"))

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


def test_solve_pipeline_with_monotone_backoff_and_strict_read_from_sil(
    tmp_path: pathlib.Path,
) -> None:
    """Verify monotone backoff converges or raises PipelineError and SIL read returns None."""
    assert read_from_sil(tmp_path / "missing_sil.json") is None
    bad_json = tmp_path / "empty_sil.json"
    bad_json.write_text(json.dumps({"roofline": {}}), encoding="utf-8")
    assert read_from_sil(bad_json) is None
    valid_json = tmp_path / "valid_sil.json"
    valid_json.write_text(json.dumps({"roofline": {"compute_flops": 1024}}), encoding="utf-8")
    assert read_from_sil(valid_json) == {"roofline": {"compute_flops": 1024}}

    facts = AscendTarget("ascend.950pr").get_facts(AscendSpaceFacts)
    tasks = (
        Task(name="load", resources=(("aiv", "aiv0", "mte2", 1),), latency=1, stage=0),
        Task(name="vec", resources=(("aiv", "aiv0", "vector", 1),), latency=1, stage=1),
    )
    edges = (Edge(src="load", dst="vec", distance=0, latency=1),)
    reuses = (Reuse(buffer="b0", last_reader="vec", next_writer="load", delta=0, read_latency=1),)
    resources = _vector_resources()

    tight_problem = Problem(
        tasks=tasks,
        edges=edges,
        buffers=(Buffer(name="b0", slot_bytes=262144, space="UB", owner="aiv", fixed=1),),
        reuses=reuses,
        resources=resources,
        capacities=(Capacity(space="UB", owner="aiv", bytes=facts.ub_simdvf_usable_bytes),),
        max_stage=1,
        manual=True,
    )
    with pytest.raises(PipelineError):
        solve_ascend_pipeline(tight_problem)

    backed_off = solve_pipeline_with_monotone_backoff(tight_problem, max_retries=4)
    assert backed_off.fallback_used is False
    assert verify_assignment(backed_off.problem, backed_off.assignment) == []
    assert backed_off.problem.buffers[0].slot_bytes <= facts.ub_simdvf_usable_bytes

    impossible_problem = Problem(
        tasks=tasks,
        edges=edges,
        buffers=(Buffer(name="b0", slot_bytes=32, space="UB", owner="aiv", fixed=1),),
        reuses=reuses,
        resources=resources,
        capacities=(Capacity(space="UB", owner="aiv", bytes=16),),
        max_stage=1,
        manual=True,
    )
    with pytest.raises(PipelineError):
        solve_pipeline_with_monotone_backoff(impossible_problem, max_retries=4)


def test_solve_ascend_pipeline_thread_safe_concurrent_workers() -> None:
    """Verify concurrent thread-pool pipeline solves isolate Z3 contexts without crashing."""
    facts = AscendTarget("ascend.950pr").get_facts(AscendSpaceFacts)
    problem = Problem(
        tasks=(
            Task(name="load", resources=(("aiv", "aiv0", "mte2", 1),), latency=1, stage=0),
            Task(name="vec", resources=(("aiv", "aiv0", "vector", 1),), latency=2, stage=1),
            Task(name="store", resources=(("aiv", "aiv0", "mte3", 1),), latency=1, stage=2),
        ),
        edges=(
            Edge(src="load", dst="vec", distance=0, latency=1),
            Edge(src="vec", dst="store", distance=0, latency=2),
        ),
        buffers=(
            Buffer(name="b_in", slot_bytes=4096, space="UB", owner="aiv", auto_min=1, auto_max=2),
            Buffer(name="b_out", slot_bytes=4096, space="UB", owner="aiv", auto_min=1, auto_max=2),
        ),
        reuses=(
            Reuse(buffer="b_in", last_reader="vec", next_writer="load", delta=0, read_latency=1),
            Reuse(buffer="b_out", last_reader="store", next_writer="vec", delta=0, read_latency=1),
        ),
        resources=_vector_resources(),
        capacities=(Capacity(space="UB", owner="aiv", bytes=facts.ub_simdvf_usable_bytes),),
        max_stage=2,
        manual=True,
    )

    def _run_once(_idx: int) -> tuple[int, list[str]]:
        res = solve_ascend_pipeline(problem)
        return res.assignment.ii, verify_assignment(res.problem, res.assignment)

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        outcomes = list(executor.map(_run_once, range(16)))
    assert len(outcomes) == 16
    assert all(ii >= 2 and probs == [] for ii, probs in outcomes)
