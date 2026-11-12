"""Fail-closed checks for the private S0 candidate."""
import pytest
from dataclasses import replace
from tilefoundry.analysis import analyze
from tilefoundry.cli.source import checked_program
from tilefoundry.schedule import finalize
from tilefoundry_ascend.plan.units import build_native_unit_tree
from tilefoundry_ascend.plan.problem import analyze_native_problem
from tilefoundry_ascend.plan.numeric import to_schedule_problem
from tilefoundry_ascend.plan.resolved import resolve_schedule
from tilefoundry_ascend.solver import solve_region
from tilefoundry_ascend.plan.mainline import apply_symbolic_pipeline, verify_actual_program, install_entry, PlanError
from tilefoundry_ascend.plan.verify import NativeVerdict
from tilefoundry_ascend.plan.identity import function_identity

SOURCE = "targets/ascend/tests_native_mainline/fixtures/authors/minimal_aiv.py:MinimalAivPipeline"


def _candidate():
    program = checked_program(SOURCE, None, command="test apply")
    result = analyze(program.module, program.entry, analysis=("compute-cost", "memory"))
    module = finalize(result, authored=program.authored)
    tree = build_native_unit_tree(module, authored=program.authored)
    problem = analyze_native_problem(tree, None, None, module.entry_function().target)
    assignments = {r.region_ref.region_id: solve_region(to_schedule_problem(r), None) for r in problem.regions}
    assert all(a.status == "SAT" for a in assignments.values())
    from tilefoundry_ascend.plan.resolved import ResolveError
    with pytest.raises(ResolveError, match="PLAN_DEPENDENCE_PROOF_UNAVAILABLE"):
        resolve_schedule(problem, assignments)
    # An untrusted clone exercises installation refusal without fabricating a plan.
    entry = replace(module.entry_function())
    candidate = replace(module, functions=(entry,), metadata=dict(module.metadata))
    return module, candidate, None, None


def test_missing_completion_and_storage_refuse_verification():
    module, candidate, plan, mapping = _candidate()
    original = module.entry_function()
    verdict = verify_actual_program(module, candidate, plan, None, None, source_map=mapping)
    assert verdict.status == "REFUSED"
    assert verdict.code == "VERIFY_PLAN_MISSING"
    assert verdict.receipt is None
    with pytest.raises(PlanError):
        install_entry(module, candidate.entry_function(), conformance=verdict)
    assert module.entry_function() is original


def test_manually_constructed_verified_verdict_cannot_install():
    module, candidate, plan, mapping = _candidate()
    original = module.entry_function()
    entry = candidate.entry_function()
    forged = NativeVerdict(status="VERIFIED", source_identity=function_identity(original),
        plan_identity="unverified", candidate_identity=function_identity(entry),
        original_entry_identity=function_identity(original), candidate_entry_identity=function_identity(entry),
        entry_identity=function_identity(entry), original_module_ref=module,
        candidate_module_ref=candidate, original_entry_ref=original, candidate_entry_ref=entry)
    with pytest.raises(PlanError) as caught:
        install_entry(module, entry, conformance=forged)
    assert caught.value.code == "INSTALL_CANDIDATE_UNVERIFIED"
    assert module.entry_function() is original

# ------------------------------------------------- kernel frame support domain
def _fixture_entry(path="targets/ascend/tests_native_mainline/fixtures/authors/"
                       "independent_ub.py:IndependentUB"):
    program = checked_program(path, None, command="test kernel frame")
    analyzed = analyze(program.module, program.entry,
                       analysis=("compute-cost", "memory"))
    module = finalize(analyzed, authored=program.authored)
    return module, module.entry_function()


def test_a_candidate_without_a_kernel_frame_is_refused_by_the_emitters_code():
    from tilefoundry.ir.tir.stmts import Sequential
    from tilefoundry_ascend.plan.apply import SymbolicApplyError, require_kernel_frame

    _, entry = _fixture_entry()
    mesh = entry.body.body[0]
    stripped = replace(entry, body=Sequential(tuple(mesh.body.body)))
    with pytest.raises(SymbolicApplyError) as caught:
        require_kernel_frame(stripped)
    assert caught.value.code == "TL_MISSING_MESH_SCOPE"
    with pytest.raises(SymbolicApplyError) as caught:
        require_kernel_frame(replace(entry, body=Sequential((mesh, mesh))))
    assert caught.value.code == "TL_MULTIPLE_MESH_SCOPES"


def test_the_frame_rejects_a_mesh_level_the_backend_cannot_launch():
    from tilefoundry.ir.types import Mesh, Topology
    from tilefoundry.ir.tir.stmts import Sequential
    from tilefoundry_ascend.plan.apply import SymbolicApplyError, require_kernel_frame

    _, entry = _fixture_entry()
    mesh = entry.body.body[0]
    aic = replace(mesh, mesh=Mesh(topologies=(Topology(name="aic", size=1),),
                                  layout=((1,),), names=("core",)))
    with pytest.raises(SymbolicApplyError) as caught:
        require_kernel_frame(replace(entry, body=Sequential((aic,))))
    assert caught.value.code == "TL_UNSUPPORTED_MESH_LEVEL"
    two = replace(mesh, mesh=Mesh(topologies=(Topology(name="aiv", size=1),
                                              Topology(name="aic", size=1)),
                                  layout=((1,), (1,)), names=("core", "core")))
    with pytest.raises(SymbolicApplyError) as caught:
        require_kernel_frame(replace(entry, body=Sequential((two,))))
    assert caught.value.code == "TL_UNSUPPORTED_MESH"


def test_an_allocation_hoisted_out_of_the_kernel_frame_is_refused():
    # The packed API refuses a dynamic shared buffer no kernel scope owns
    # (make_packed_api.cc: "variables (buf_dyn_shmem,) are used, but not passed
    # in as API arguments"). A candidate that lowers that way must be refused
    # here, not exported to the backend as a crash.
    from tilefoundry.ir.core import Call
    from tilefoundry.ir.tir.memory.alloc_tensor import AllocTensor
    from tilefoundry.ir.tir.stmts import ExecutionScope, LetStmt, Sequential
    from tilefoundry_ascend.plan.apply import (SymbolicApplyError, alloc_nodes,
                                               require_kernel_frame)

    _, entry = _fixture_entry()
    mesh = entry.body.body[0]
    vector = next(item for item in mesh.body.body if isinstance(item, ExecutionScope))
    real_allocs = [item for item in vector.body.body
                   if isinstance(item, LetStmt) and isinstance(item.value, Call)
                   and isinstance(item.value.target, AllocTensor)]
    assert real_allocs, "the lowered program allocates its vector lanes inside the frame"
    hoisted_stmt = real_allocs[0]
    rebuilt_vector = replace(vector, body=Sequential(tuple(
        item for item in vector.body.body if item is not hoisted_stmt)))
    rebuilt_mesh = replace(mesh, body=Sequential(tuple(
        item if not isinstance(item, ExecutionScope) else rebuilt_vector
        for item in mesh.body.body)))
    candidate = replace(entry, body=Sequential((hoisted_stmt, rebuilt_mesh)))
    outside = [node for is_inside, node in alloc_nodes(candidate) if not is_inside]
    assert len(outside) == 1, "the hoisted allocation is now outside every frame"
    # the untouched program keeps every allocation inside the frame
    assert all(is_inside for is_inside, _ in alloc_nodes(entry))
    with pytest.raises(SymbolicApplyError) as caught:
        require_kernel_frame(candidate)
    assert caught.value.code == "TL_ALLOC_OUTSIDE_KERNEL_FRAME"
    # and the untouched real candidate passes, so the check is not unconditional
    assert require_kernel_frame(entry) is mesh


def test_verify_refuses_a_frameless_candidate_before_reading_any_plan():
    from tilefoundry.ir.tir.stmts import Sequential

    module, entry = _fixture_entry()
    mesh = entry.body.body[0]
    frameless = replace(entry, body=Sequential(tuple(mesh.body.body)))
    candidate = replace(module, name=module.name + "#candidate",
                        functions=(frameless,), entry=frameless.name)
    assert candidate is not module
    verdict = verify_actual_program(module, candidate, None, None, None)
    assert verdict.status == "REFUSED"
    assert verdict.code == "TL_MISSING_MESH_SCOPE"
    assert module.entry_function() is entry
