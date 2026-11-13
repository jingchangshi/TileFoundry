"""StoragePlan: post-version placement, real capacity facts and named refusals.

The S0 cases run on the REAL plan the chain produces (normal entry -> analyze ->
analyze_native_problem -> resolve_schedule). The problem-to-core adapter is not
landed yet, so the test supplies the SAT-shaped assignment resolve_schedule
verifies; every buffer type, version count and capacity still comes from the real
problem/plan, not from a hand-written TIR or a made-up footprint.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from tilefoundry.analysis import analyze
from tilefoundry.cli.source import checked_program
from tilefoundry.ir.core import Var
from tilefoundry.ir.types import DType
from tilefoundry.ir.types.utils import make_tensor_type
from tilefoundry.schedule import finalize
from tilefoundry_ascend.plan.numeric import to_schedule_problem
from tilefoundry_ascend.plan.problem import analyze_native_problem
from tilefoundry_ascend.plan.resolved import BufferPlan, RegionPlan, ResolvedPlan, resolve_schedule
from tilefoundry_ascend.plan.units import build_native_unit_tree
from tilefoundry_ascend.solver import solve_region
from tilefoundry_ascend.storage import (
    ScratchAllocation,
    StorageError,
    StorageLowering,
    place_resolved_versions,
    require_storage_verified,
    verify_storage_current,
)
from tilefoundry_ascend.tf_target import Ascend950PRTarget

AUTHOR = "targets/ascend/tests_native_mainline/fixtures/authors/minimal_aiv.py:MinimalAivPipeline"
UB_BYTES = 256 * 1024


def _real_root_problem():
    program = checked_program(AUTHOR, None, command="test storage")
    analyzed = analyze(program.module, program.entry, analysis=("compute-cost", "memory"))
    module = finalize(analyzed, authored=program.authored)
    tree = build_native_unit_tree(module, authored=program.authored)
    problem = analyze_native_problem(tree, None, None, module.entry_function().target)
    root = problem.regions[0]
    root_problem = replace(problem, regions=(root,))
    # The REAL numeric projection and the shared solver decide this region.
    assignment = solve_region(to_schedule_problem(root), None)
    assert assignment.status == "SAT", assignment.detail
    plan = resolve_schedule(root_problem, assignment)
    return plan, module.entry_function().target


def _var(name, *, shape=(64,), storage="smem", dtype=DType.f32):
    return Var(name, type=make_tensor_type(tuple(shape), dtype, storage))


def _plan(buffers, *, plan_identity="sha256:test-plan", target_identity=None):
    from tilefoundry_ascend.plan.problem import _target_identity

    if target_identity is None:
        target_identity = _target_identity(Ascend950PRTarget())
    region = RegionPlan(
        region_ref=SimpleNamespace(region_id="region:0"),
        lower_expr=None, stop_expr=None, step_expr=None,
        ii=1, origin=0, max_shift=0, tasks=(), buffers=tuple(buffers),
        identity="sha256:test-region",
    )
    return ResolvedPlan(
        source_identity="sha256:test-source",
        target_identity=target_identity,
        model_version="sha256:test-model",
        regions=(region,),
        task_to_native_source={},
        feasibility="SAT",
        optimality=None,
        plan_identity=plan_identity,
    )


def _buffer(buffer_id, root, versions):
    return BufferPlan(buffer_ref=root, buffer_id=buffer_id, domain=None,
                      resolved_versions=versions)


def test_synthetic_ub_layout_is_placed_but_not_verified():
    # Synthetic input exercises layout, not plan authentication.
    plan = _plan((_buffer("buffer:0", _var("tile"), 1),))
    target = Ascend950PRTarget()
    storage = place_resolved_versions(plan, None, target, None)

    smem = [entry for entry in storage.entries if entry.space == "smem"]
    assert len(smem) == 1
    placed = smem[0]
    assert [requirement.space for requirement in storage.requirements] == ["smem"]
    assert placed.slot_count == 1
    assert placed.base_offset_bytes == 0
    assert placed.version_stride_bytes == placed.allocation_bytes
    assert placed.allocation_bytes == 64 * 4
    assert placed.allocation_bytes <= UB_BYTES
    assert [capacity.usable_bytes for capacity in storage.capacities] == [UB_BYTES]
    assert storage.peak_bytes[0][1] == placed.allocation_bytes
    assert storage.identity.startswith("sha256:")

    assert storage.verified is False
    with pytest.raises(StorageError) as caught:
        require_storage_verified(storage)
    assert caught.value.code == "STORAGE_VERIFICATION_INPUTS_MISSING"

    with pytest.raises(StorageError, match="STORAGE_VERIFICATION_INPUTS_MISSING"):
        verify_storage_current(plan, storage, target)


def test_two_versions_are_contiguous_and_roots_do_not_overlap():
    first = _var("tile_a")
    second = _var("tile_b")
    plan = _plan((_buffer("buffer:0", first, 2), _buffer("buffer:1", second, 1)))
    target = Ascend950PRTarget()
    storage = place_resolved_versions(plan, None, target, None)
    by_buffer = {str(entry.root.name): entry for entry in storage.entries}

    tile_a = by_buffer["tile_a"]
    tile_b = by_buffer["tile_b"]
    assert tile_a.slot_count == 2
    assert tile_a.version_stride_bytes == tile_a.allocation_bytes // 2
    assert tile_a.version_stride_bytes == 64 * 4
    assert tile_b.base_offset_bytes == tile_a.base_offset_bytes + tile_a.allocation_bytes
    assert tile_a.base_offset_bytes + tile_a.allocation_bytes <= tile_b.base_offset_bytes
    with pytest.raises(StorageError, match="STORAGE_VERIFICATION_INPUTS_MISSING"):
        verify_storage_current(plan, storage, target)


def test_requirement_above_ub_capacity_is_refused_not_clipped():
    huge = _var("huge", shape=(131072,))
    plan = _plan((_buffer("buffer:0", huge, 1),))
    with pytest.raises(StorageError) as caught:
        place_resolved_versions(plan, None, Ascend950PRTarget(), None)
    assert caught.value.code == "STORAGE_CAPACITY_EXCEEDED"
    assert "not clipped" in caught.value.detail


def test_a_register_space_is_refused_by_the_register_gate_not_by_byte_capacity():
    """A counted register space gets the MORE PRECISE refusal, not the old one.

    This used to assert UNKNOWN_CAPACITY, which said "the address space has no
    capacity". For rmem/carrier that was the wrong layer: the resource is a COUNT
    of registers, and what is missing is a declared per-domain register budget. The
    code is now STORAGE_REGISTER_UNSOURCED, which names the side that is unsourced.
    Asserting the exact code (not "any StorageError") is the point: the newer code is
    strictly more informative and this test exists to keep that distinction.
    """
    for space in ("rmem", "carrier"):
        buffer = _plan((_buffer("buffer:0", _var("reg", storage=space), 1),))
        with pytest.raises(StorageError) as caught:
            place_resolved_versions(buffer, None, Ascend950PRTarget(), None)
        assert caught.value.code == "STORAGE_REGISTER_UNSOURCED", (
            space, caught.value.code)
        assert space in str(caught.value), str(caught.value)[:200]


def test_tampered_placement_is_detected():
    # Synthetic input exercises layout, not plan authentication.
    plan = _plan((_buffer("buffer:0", _var("tile"), 1),))
    target = Ascend950PRTarget()
    storage = place_resolved_versions(plan, None, target, None)
    with pytest.raises(StorageError, match="STORAGE_VERIFICATION_INPUTS_MISSING"):
        verify_storage_current(plan, storage, target)

    entry = storage.entries[0]
    offset = replace(entry, base_offset_bytes=entry.base_offset_bytes + 64)
    with pytest.raises(StorageError) as caught:
        verify_storage_current(plan, replace(storage, entries=(offset,)), target)
    assert caught.value.code == "STORAGE_PLACEMENT_MISMATCH"

    bytes_ = replace(entry, allocation_bytes=entry.allocation_bytes + 256)
    with pytest.raises(StorageError) as caught:
        verify_storage_current(plan, replace(storage, entries=(bytes_,)), target)
    assert caught.value.code == "STORAGE_BYTES_MISMATCH"

    other_plan = replace(plan, plan_identity="sha256:someone-else")
    with pytest.raises(StorageError) as caught:
        verify_storage_current(other_plan, storage, target)
    assert caught.value.code == "STORAGE_PLAN_IDENTITY_MISMATCH"


def test_scratch_is_charged_once_per_allocation():
    tile = _var("tile")
    plan = _plan((_buffer("buffer:0", tile, 2),))
    scratch = ScratchAllocation(
        scratch_id="scratch:0",
        allocation_ref=_var("scratch_window", storage="smem"),
        space="smem",
        root_buffer_id="buffer:0",
        dtype="f32",
        lanes=64,
        copies=2,
        source_unit_id="unit:0",
    )
    lowering = StorageLowering(scratch_allocations=(scratch,))
    target = Ascend950PRTarget()
    storage = place_resolved_versions(plan, None, target, lowering)

    assert len(storage.requirements[0].extra_scratch) == 1
    assert len(storage.scratch_entries) == 1
    scratch_entry = storage.scratch_entries[0]
    assert scratch_entry.copies == 2
    assert scratch_entry.copy_stride_bytes == 64 * 4
    assert scratch_entry.allocation_bytes == 2 * 64 * 4
    assert scratch_entry.base_offset_bytes == 2 * 64 * 4

    # One buffer version ring + ONE scratch allocation, not one scratch per version.
    expected_peak = 2 * 64 * 4 + 2 * 64 * 4
    assert storage.peak_bytes[0][1] == expected_peak
    assert storage.peak_bytes[0][1] != 2 * (64 * 4 + 2 * 64 * 4)
    with pytest.raises(StorageError, match="STORAGE_VERIFICATION_INPUTS_MISSING"):
        verify_storage_current(plan, storage, target, lowering=lowering)

def test_forged_verification_never_authorizes_tampered_storage():
    from tilefoundry_ascend.storage import StorageVerification

    # Synthetic input exercises the verifier refusal boundary.
    plan = _plan((_buffer("buffer:0", _var("tile"), 1),))
    target = Ascend950PRTarget()
    lowering = StorageLowering()
    storage = place_resolved_versions(plan, None, target, lowering)
    forged = StorageVerification(True, plan.plan_identity, storage.identity,
                                 "invented", storage.target_identity)
    bad_entry = replace(storage.entries[0], base_offset_bytes=64)
    bad = replace(storage, entries=(bad_entry,), verification=forged)
    with pytest.raises(StorageError, match="STORAGE_VERIFICATION_INPUTS_MISSING"):
        require_storage_verified(bad)
    with pytest.raises(StorageError, match="STORAGE_PLACEMENT_MISMATCH"):
        require_storage_verified(bad, plan=plan, target=target,
                                 completion=object(), lowering=lowering)


def test_storage_verification_needs_a_rebindable_completion():
    from tilefoundry_ascend.completion import CompletionError

    # Synthetic input exercises the verifier refusal boundary. A hand-built plan
    # carries no problem_ref, so the completion verifier cannot re-bind the
    # effect table and refuses with the CONCRETE reason, naming the region.
    plan = _plan((_buffer("buffer:0", _var("tile"), 1),))
    target = Ascend950PRTarget()
    lowering = StorageLowering()
    completion = object()
    storage = place_resolved_versions(plan, completion, target, lowering)
    with pytest.raises(CompletionError) as caught:
        require_storage_verified(storage, plan=plan, completion=completion,
                                 target=target, lowering=lowering)
    assert caught.value.code == "COMPLETION_PROBLEM_BINDING_UNAVAILABLE"
    # the refusal locates the missing fact rather than reporting a wall
    assert "problem_ref" in caught.value.message and "region:0" in caught.value.message
    assert "region:0" in str(caught.value)

def test_changed_actual_target_refuses_old_plan():
    # Synthetic input exercises layout, not plan authentication.
    plan = _plan((_buffer("buffer:0", _var("tile"), 1),))
    target = Ascend950PRTarget()
    storage = place_resolved_versions(plan, None, target, StorageLowering())
    changed = Ascend950PRTarget(aiv_count=1, aic_count=1)
    with pytest.raises(StorageError, match="STORAGE_TARGET_IDENTITY_MISMATCH"):
        place_resolved_versions(plan, None, changed, StorageLowering())
    with pytest.raises(StorageError, match="STORAGE_TARGET_IDENTITY_MISMATCH"):
        verify_storage_current(plan, storage, changed)

COPY_ONLY = ("targets/ascend/tests_native_mainline/fixtures/authors/"
             "copy_only.py:CopyOnly")
INDEPENDENT = ("targets/ascend/tests_native_mainline/fixtures/authors/"
               "independent_ub.py:IndependentUB")


def _chain(path):
    program = checked_program(path, None, command="test storage chain")
    analyzed = analyze(program.module, program.entry,
                       analysis=("compute-cost", "memory"))
    module = finalize(analyzed, authored=program.authored)
    tree = build_native_unit_tree(module, authored=program.authored)
    problem = analyze_native_problem(tree, None, None, module.entry_function().target)
    answers = {region.region_ref.region_id:
               solve_region(to_schedule_problem(region)) for region in problem.regions}
    assert all(answer.status == "SAT" for answer in answers.values()), answers
    plan = resolve_schedule(problem, answers)
    target = module.entry_function().target
    effects = tuple(effect for region in problem.regions for effect in region.effects)
    return plan, target, effects


def test_real_inputs_verify_storage_and_tampering_still_refuses():
    """Positive acceptance with ALL real inputs: plan+completion+target+lowering.

    This uses the GM/UB-only S0 slice: a plan that places bytes in a space the
    target gives no capacity for still refuses with UNKNOWN_CAPACITY (the next
    test), so a verified placement exists exactly where the capacity facts do.
    """
    from tilefoundry_ascend.completion import (build_symbolic_completion,
                                               verify_completion, CompletionError)

    plan, target, effects = _chain(COPY_ONLY)
    completion = verify_completion(
        plan, build_symbolic_completion(plan, target, effects), target)
    lowering = StorageLowering()
    storage = place_resolved_versions(plan, completion, target, lowering)
    verified = require_storage_verified(storage, plan=plan, completion=completion,
                                        target=target, lowering=lowering)
    assert verified.verification is not None
    assert verified.verification.verified is True
    assert verified.verification.plan_identity == plan.plan_identity
    assert verified.verification.storage_identity == storage.identity
    assert [entry.space for entry in verified.entries] == ["smem"]
    assert verified.peak_bytes[0][1] == 64 * 4

    # tampering with an offset is caught WITH every real input present
    bad = replace(storage, entries=(replace(storage.entries[0],
                                            base_offset_bytes=9999),))
    with pytest.raises(StorageError) as caught:
        require_storage_verified(bad, plan=plan, completion=completion,
                                 target=target, lowering=lowering)
    assert caught.value.code == "STORAGE_PLACEMENT_MISMATCH"

    # An object that is not a real CompletionPlan never authorizes anything. It
    # is caught here by the recomputed completion identity, and a copy of that
    # identity string is caught by the completion verifier itself: neither path
    # reaches a verified record.
    with pytest.raises(StorageError) as caught:
        require_storage_verified(storage, plan=plan, completion=object(),
                                 target=target, lowering=lowering)
    assert caught.value.code == "STORAGE_PLACEMENT_MISMATCH"
    assert "completion_identity" in caught.value.detail

    imposter = SimpleNamespace(identity=completion.identity)
    with pytest.raises(CompletionError) as caught:
        require_storage_verified(storage, plan=plan, completion=imposter,
                                 target=target, lowering=lowering)
    assert caught.value.code == "COMPLETION_INVALID"

    # the detached property still proves nothing on its own
    assert storage.verified is False


def test_a_real_register_plan_still_refuses_an_undeclared_register_budget():
    """The iron rule survives a verified completion: no budget, no placement.

    independent_ub solves and completes (its buffers include RMEM registers), but
    the target declares no per-domain register COUNT and the lowering reports no
    occupancy. Placement must refuse rather than backfill, even now that completion
    is real - and it must refuse in the REGISTER vocabulary, because the missing
    fact is a register budget, not an address-space capacity.
    """
    from tilefoundry_ascend.completion import (build_symbolic_completion,
                                               verify_completion)

    plan, target, effects = _chain(INDEPENDENT)
    completion = verify_completion(
        plan, build_symbolic_completion(plan, target, effects), target)
    assert completion.verification.status == "VERIFIED"
    spaces = sorted({str(buffer.buffer_ref.type.storage.name).lower()
                     for region in plan.regions for buffer in region.buffers})
    assert "rmem" in spaces, spaces
    with pytest.raises(StorageError) as caught:
        place_resolved_versions(plan, completion, target, StorageLowering())
    assert caught.value.code == "STORAGE_REGISTER_UNSOURCED", caught.value.code
    assert "rmem" in str(caught.value), str(caught.value)[:200]

