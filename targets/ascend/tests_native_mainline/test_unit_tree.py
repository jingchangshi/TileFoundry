"""UnitTree and NormalizedProblem over the REAL lowered S0 author program.

Every record asserted here is read from a module produced by the normal entry
(checked_program -> analyze -> finalize) over the authored fixture. No TIR is
hand-built, no source is skipped, and the expected coverage sets are enumerated
independently of the units they are compared against.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tilefoundry.analysis import analyze
from tilefoundry.cli.source import checked_program
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate, ExecutionScope, For
from tilefoundry.ir.types.utils import static_dim_value
from tilefoundry.schedule import finalize
from tilefoundry_ascend.ir.execution import SIMD_VF_KIND
from tilefoundry_ascend.plan.problem import (
    CoverageVerdict,
    NormalizedProblem,
    RegionProblem,
    analyze_native_problem,
)
from tilefoundry_ascend.plan.units import (
    DMA,
    VF,
    Region,
    SourceUnit,
    UnitTree,
    UnitTreeError,
    build_native_unit_tree,
    unit_accesses,
)

AUTHORS = Path(__file__).resolve().parent / "fixtures" / "authors"
FIXTURE = AUTHORS / "minimal_aiv.py"
SOURCE = "%s:MinimalAivPipeline" % (FIXTURE,)
ENTRY = "vsum"
STEPS = 4
LANES = 64


def _run_chain(source):
    program = checked_program(source, None, command="test unit tree")
    analyzed = analyze(program.module, program.entry, analysis=("compute-cost", "memory"))
    module = finalize(analyzed, authored=program.authored)
    tree = build_native_unit_tree(module, authored=program.authored)
    target = module.entry_function().target
    problem = analyze_native_problem(tree, None, None, target)
    return program, module, tree, problem


def _variant(tmp_path, old, new, name="variant.py"):
    text = FIXTURE.read_text(encoding="utf-8")
    assert old in text, "the fixture no longer contains %r" % (old,)
    path = tmp_path / name
    path.write_text(text.replace(old, new), encoding="utf-8")
    return "%s:MinimalAivPipeline" % (path,)


def _units_of(tree, region):
    return [child for child in region.direct_children if isinstance(child, SourceUnit)]


def test_unit_tree_groups_the_whole_vf_into_one_unit():
    _, _, tree, _ = _run_chain(SOURCE)
    assert isinstance(tree, UnitTree)
    assert isinstance(tree.entry_ref, PrimFunction)

    assert len(tree.roots) == 1
    root = tree.roots[0]
    assert isinstance(root, Region)
    assert root.parent_id is None and isinstance(root.node_ref, PrimFunction)
    loop_regions = [region for region in tree.regions if region.parent_id is not None]
    assert len(loop_regions) == 1
    loop = loop_regions[0]
    assert isinstance(loop.node_ref, For)
    assert loop.parent_id == root.region_id

    # Outside the loop: two DMA units (the initial load and the final store) plus
    # the loop region itself. Inside the loop: one DMA unit and one whole-VF unit.
    assert [child.kind if isinstance(child, SourceUnit) else "REGION"
            for child in root.direct_children] == [DMA, "REGION", DMA]
    inside = _units_of(tree, loop)
    assert [unit.kind for unit in inside] == [DMA, VF]
    assert len(tree.units) == 4
    assert [unit.kind for unit in tree.units] == [DMA, DMA, VF, DMA]

    vf_units = [unit for unit in tree.units if unit.kind == VF]
    assert len(vf_units) == 1
    vf = vf_units[0]
    assert isinstance(vf.node_ref, ExecutionScope)
    assert vf.node_ref.kind == SIMD_VF_KIND
    assert vf.whole_body is vf.node_ref
    # The five in-region instructions are ONE unit, in real source order.
    members = [type(effect.source_ref.callable).__name__ for effect in vf.effects]
    assert members == ["Vld", "Vld", "Pset", "Vadd", "Vsts"]
    assert len(vf.effects) == 5
    assert vf.resources, "the whole-VF unit must state its target resources"
    assert {use.resource.pipe for use in vf.resources} == {"vector"}
    assert all(access.task_id == vf.unit_id for access in unit_accesses(vf))

    # The loop Region carries a typed summary of the REAL lowering binding record.
    summary = loop.binding_summary
    assert summary is not None
    assert summary.actual_loop is loop.node_ref
    assert summary.actual_induction_var is loop.node_ref.induction_var
    assert summary.induction_var is summary.source_region.induction_var
    assert isinstance(summary.source_region, LoopRegion)
    assert summary.actual_bounds == (loop.node_ref.start, loop.node_ref.stop, loop.node_ref.step)
    assert (static_dim_value(summary.start), static_dim_value(summary.stop),
            static_dim_value(summary.step)) == (0, STEPS, 1)
    assert len(summary.carried_params) == len(summary.carried_initials)
    assert len(summary.carried_params) == len(summary.yield_values) == 1
    assert summary.captures == summary.source_region.captures()
    assert summary.binding_map[0][0] is summary.source_region.induction_var
    assert summary.binding_map[0][1] is loop.node_ref.induction_var
    assert summary.binding_map[1][0] is summary.carried_params[0]


def test_effects_and_accesses_come_from_real_param_defs():
    _, _, tree, problem = _run_chain(SOURCE)
    loop_region = [region for region in problem.regions
                   if isinstance(region.region_ref.node_ref, For)][0]
    vf_task = [task for task in loop_region.tasks if task.unit_ref.kind == VF][0]
    vf = vf_task.unit_ref
    accesses = unit_accesses(vf)
    modes = [(access.operand_index, access.mode) for access in accesses]
    assert modes == [
        (0, "READ"), (1, "WRITE"),      # vld
        (0, "READ"), (1, "WRITE"),      # vld
        (0, "WRITE"),                   # pset
        (0, "READ"), (1, "READ"), (2, "READ"), (3, "READ_WRITE"),  # vadd
        (0, "READ"), (1, "WRITE"),      # vsts
    ]
    assert all(access.root_ref is not None and access.view_ref is not None
               for access in accesses)
    assert all(access.guard is not None and access.generation is not None
               for access in accesses)
    # Every effect is a declared MEMORY effect from the instruction schema and its
    # completion comes from the target instruction contract.
    assert all(effect.kind == "MEMORY" for effect in vf.effects)
    assert [effect.completion for effect in vf.effects] == [
        "READ_COMPLETE", "READ_COMPLETE", "ISSUE", "ISSUE", "WRITE_VISIBLE",
    ]
    assert all(effect.pipe == "vector" for effect in vf.effects)
    # Dependences are write-involving, so a read/read pair never becomes a data edge.
    for region in problem.regions:
        for edge in region.edges:
            assert edge.kind in {"RAW", "WAR", "WAW"}
            assert edge.kind in {"RAW", "WAR", "WAW"}
            assert edge.effect_ids
            assert all(effect_id.startswith("effect:") for effect_id in edge.effect_ids)
    kinds = [(edge.kind, edge.producer, edge.consumer)
             for region in problem.regions for edge in region.edges]
    # The loop region states THREE edges, not one: the intra-iteration producer
    # ->consumer pair, and the CARRIED self edge (unit:2 -> unit:2) that says
    # iteration k+1's read of the loop-carried slice is served by iteration k's
    # write. That pair used to be dropped silently because both accesses belong
    # to the same task; a distance-0 self edge would be unsatisfiable, so it is
    # stated at distance 1 and that is asserted here, not just its presence.
    assert kinds == [("RAW", "unit:0", "unit:3"),
                     ("RAW", "unit:1", "unit:2"),
                     ("RAW", "unit:2", "unit:2")]
    carried = [edge for region in problem.regions for edge in region.edges
               if edge.producer == edge.consumer]
    assert len(carried) == 1, carried
    assert int(carried[0].distance) == 1, "a carried pair is one iteration apart"
    assert carried[0].delay_fact.state != "PROVEN_ZERO", \
        "a carried edge can never be in the proven-zero family"


def test_problem_has_the_complete_finite_sets_and_stage_order():
    _, _, _, problem = _run_chain(SOURCE)
    assert isinstance(problem, NormalizedProblem)
    assert problem.source_snapshot.identity == problem.unit_tree.source_snapshot.identity
    assert problem.target_identity.startswith("sha256:")
    assert problem.model_version.startswith("sha256:")
    assert len(problem.regions) == 2
    root_problem, loop_problem = problem.regions
    assert isinstance(root_problem, RegionProblem)
    assert [task.task_id for task in root_problem.tasks] == ["unit:0", "unit:3"]
    assert [task.task_id for task in loop_problem.tasks] == ["unit:1", "unit:2"]
    assert [task.source_order for task in root_problem.tasks] == [0, 2]
    assert [task.source_order for task in loop_problem.tasks] == [0, 1]
    assert root_problem.pipes == ("mte2", "mte3")
    assert loop_problem.pipes == ("mte2", "vector")
    assert all(region.accesses and region.effects and region.resources
               and region.owners for region in problem.regions)
    # Stage facts come from the authored PipelineSpec; no offset plan is invented.
    assert loop_problem.enable_offset is False
    assert loop_problem.max_stage == 2
    assert loop_problem.fixed_ii is None
    assert loop_problem.ii_min == 1 and loop_problem.ii_max >= 1
    # A straight-line region runs once: its interval is its serial span.
    serial = sum(task.latency for task in root_problem.tasks)
    assert root_problem.max_stage == 0
    assert root_problem.ii_min == root_problem.ii_max == serial

    loop_buffers = {buffer.buffer_id: buffer for buffer in loop_problem.buffers}
    smem = [buffer for buffer in loop_buffers.values() if buffer.space == "smem"]
    assert smem and all(buffer.slot_bytes == LANES * 4 for buffer in smem)
    carried = [buffer for buffer in smem if buffer.initial_values]
    assert len(carried) == 1 and len(carried[0].initial_values) == 1
    assert loop_problem.capacities and loop_problem.capacities[0].usable_bytes == 256 * 1024


def test_source_coverage_is_complete_and_independently_enumerated():
    program, module, tree, problem = _run_chain(SOURCE)
    coverage = problem.coverage
    assert isinstance(coverage, CoverageVerdict)
    assert coverage.complete, coverage.refusals
    assert coverage.refusals == ()
    assert coverage.expected_tasks == coverage.covered_tasks
    assert coverage.expected_accesses == coverage.covered_accesses
    assert coverage.expected_effects == coverage.covered_effects
    # 8 Evaluate statements in the lowered body, 17 effect-bearing operands,
    # one declared effect per instruction.
    assert len(coverage.expected_tasks) == 8
    assert len(coverage.expected_accesses) == 17
    assert len(coverage.expected_effects) == 8
    assert coverage.source_identity == tree.source_snapshot.identity
    assert coverage.problem_identity == problem.coverage.problem_identity


def test_problem_shape_does_not_depend_on_trip_count(tmp_path):
    _, _, small_tree, small = _run_chain(SOURCE)
    variant = _variant(tmp_path, "tf.pipelined(4,", "tf.pipelined(8,")
    _, _, large_tree, large = _run_chain(variant)

    assert [unit.kind for unit in large_tree.units] == [unit.kind for unit in small_tree.units]
    assert len(large_tree.units) == len(small_tree.units) == 4
    assert len(large.regions) == len(small.regions) == 2
    assert [task.task_id for region in large.regions for task in region.tasks] == [
        task.task_id for region in small.regions for task in region.tasks]
    assert large.coverage.expected_tasks == small.coverage.expected_tasks
    assert large.coverage.expected_accesses == small.coverage.expected_accesses
    assert large.coverage.expected_effects == small.coverage.expected_effects
    assert large.coverage.complete and small.coverage.complete

    loop_region = [region for region in large.regions
                   if isinstance(region.region_ref.node_ref, For)][0]
    assert static_dim_value(loop_region.region_ref.stop_expr) == 8
    small_loop = [region for region in small.regions
                  if isinstance(region.region_ref.node_ref, For)][0]
    assert static_dim_value(small_loop.region_ref.stop_expr) == STEPS


def test_unsupported_instruction_is_a_named_source_located_refusal(tmp_path):
    # A real author variant whose transfer is a core Copy: the normal entry still
    # finalizes, but the instruction has no contract in the Ascend instruction
    # table, so the unit boundary refuses by name and keeps the source node.
    source = _variant(tmp_path, "op=T.ascend_gm_to_ub()", "op=T.copy()", name="copy.py")
    program = checked_program(source, None, command="test unit tree")
    analyzed = analyze(program.module, program.entry, analysis=("compute-cost", "memory"))
    module = finalize(analyzed, authored=program.authored)
    with pytest.raises(UnitTreeError) as caught:
        build_native_unit_tree(module, authored=program.authored)
    error = caught.value
    assert error.code == "UNIT_TREE_INSTRUCTION_UNREGISTERED"
    assert isinstance(error.source_ref, Evaluate)
    assert type(error.source_ref.callable).__name__ == "Copy"
    assert error.detail
