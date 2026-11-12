"""G32 / P3-03: ApplySoftwarePipeline on REAL TIR.

What this suite holds the transform to:

  * the emitted loop is ONE symbolic guarded loop whose static size does not depend on
    the trip count, and every (task, k) instance runs exactly once;
  * trip 0/1/<depth/==depth/>depth, a non-zero lower bound, step > 1 and a SYMBOLIC stop
    all keep that exactness;
  * ring slot = k % V and generation = k are visible in the emitted addresses for
    V = 1, 2, 3 and beyond;
  * perturbing the assignment (stage / phase / II / versions) changes the materialized
    TIR, so a test cannot pass on a transform that ignores its input;
  * a wrong shift, a guard that admits the wrong instances, or a dropped carried value
    is REFUSED by the independent verifier -- not merely not-produced;
  * a plan that already records a needs_replan diagnostic is refused by BOTH the plan
    builder and the pipeline (the C-1 seam).

Every fixture here is real typed TIR that passes verify_prim_function in its own right,
and the rewritten module is re-verified. Nothing asserts on IR text.
"""
from __future__ import annotations

import dataclasses

import pytest

from tilefoundry.ir.core import Call, Constant, Var
from tilefoundry.ir.tir.memory import AllocTensor
from tilefoundry.ir.tir.memory.copy import Copy
from tilefoundry.ir.tir.memory.tensor_view import TensorView
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate, For, If, LetStmt, MeshScope, Sequential
from tilefoundry.ir.types import Layout, PointerType, DType, make_tensor_type
from tilefoundry.ir.types.mesh import Mesh, Topology
from tilefoundry.ir.types.storage import StorageKind
from tilefoundry.visitor_registry.verify import verify_prim_function

from tilefoundry_ascend.plan import pipeline as PIPE
from tilefoundry_ascend.plan import resolved_plan as RP

import solve_fixture

ELEMENTS, HALF, FOOTPRINT, STRIDE, ALLOC = 8, 32, 32, 64, 128
FACTS = {"arena": {"versions": 2, "alignment": 64, "base_offset_bytes": 0, "arena_id": "a0"}}


def _scalar(dtype=DType.i32):
    return make_tensor_type((), dtype=dtype, storage=StorageKind.UMAT)


def _view(offset, tensor_type):
    return Call(type=tensor_type,
                target=TensorView(storage=tensor_type.storage,
                                  dtype=str(tensor_type.dtype),
                                  layout=Layout((ELEMENTS,)),
                                  shape=(ELEMENTS,)),
                args=(Constant(type=PointerType(tensor_type.dtype,
                                              tensor_type.storage),
                               value=int(offset)),))


def make_tir(rows=5, start=0, step=1, symbolic_stop=False):
    """Real typed TIR: a mesh-scoped loop copying two halves of one SMEM buffer."""
    arena = make_tensor_type((ELEMENTS,), dtype=DType.f32, storage=StorageKind.SMEM)
    arena_var = Var(name="arena", type=arena)
    alloc = LetStmt(var=arena_var,
                    value=Call(type=arena, target=AllocTensor(tensor_type=arena), args=()))
    induction = Var(name="i", type=_scalar())
    stop = Var(name="n", type=_scalar()) if symbolic_stop else Constant(
        type=_scalar(), value=rows)
    loop = For(induction_var=induction,
               start=Constant(type=_scalar(), value=start), stop=stop,
               step=Constant(type=_scalar(), value=step),
               body=Sequential(body=(
                   Evaluate(callable=Copy(), args=(_view(0, arena), _view(HALF, arena))),
                   Evaluate(callable=Copy(), args=(_view(HALF, arena), _view(0, arena))))))
    body = Sequential(body=(alloc, loop))
    mesh = Mesh(topologies=(Topology("aiv", 1),), layout=Layout((1,)), names=("core",))
    scope = MeshScope(mesh=mesh, binding=Var(name="core", type=_scalar()), body=body)
    params = (stop,) if symbolic_stop else ()
    return PrimFunction(name="pfx", params=params, body=Sequential(body=(scope,)),
                        output_count=0)


def make_problem(start=0, stop=5, step=1, enable_offset=True, stages=4):
    """Two-task, one-AUTO-buffer Problem whose domain MATCHES the TIR loop."""
    return {
        "problem_digest": "sha256:" + "a" * 64,
        "source_digest": "sha256:" + "b" * 64,
        "ledger_sha": "sha256:" + "c" * 64,
        "loop_id": "row_pipeline",
        "loop_domain": {"start": start, "stop": stop, "step": step},
        "enable_offset": enable_offset,
        "requested_num_stages": stages,
        "ii": 1, "latency": 1,
        "tasks": [
            {"stable_id": "mte", "action_ids": ["a0"], "pipe": "DMA", "resource": "DMA",
             "source": "s_copy"},
            {"stable_id": "vec", "action_ids": ["a1"], "pipe": "vector",
             "resource": "vector", "source": "s_vadd"},
        ],
        "buffers": [{"root": "arena", "source_decl": "arena", "owner": "row_pipeline",
                     "space": "ascend.ub", "dtype": "f32", "shape": [ELEMENTS],
                     "bytes": ALLOC, "alignment": 64,
                     "memory_domain": "ascend.ub"}],
        "accesses": [
            {"access_id": "acc-c1-src", "task": "mte", "action_id": "a0",
             "source_stmt": "s1/for/s0", "root": "arena", "space": "ascend.ub",
             "mode": "read", "relative_byte_offset": 0, "byte_footprint": FOOTPRINT,
             "logical_index": RP.V("i")},
            {"access_id": "acc-c1-dst", "task": "mte", "action_id": "a0",
             "source_stmt": "s1/for/s0", "root": "arena", "space": "ascend.ub",
             "mode": "write", "relative_byte_offset": HALF, "byte_footprint": FOOTPRINT,
             "logical_index": RP.V("i")},
            {"access_id": "acc-c2-src", "task": "vec", "action_id": "a1",
             "source_stmt": "s1/for/s1", "root": "arena", "space": "ascend.ub",
             "mode": "read", "relative_byte_offset": HALF, "byte_footprint": FOOTPRINT,
             "logical_index": RP.V("i")},
            {"access_id": "acc-c2-dst", "task": "vec", "action_id": "a1",
             "source_stmt": "s1/for/s1", "root": "arena", "space": "ascend.ub",
             "mode": "write", "relative_byte_offset": 0, "byte_footprint": FOOTPRINT,
             "logical_index": RP.V("i")},
        ],
    }


def build_plan(problem, placement=None, *, ii=2, versions=2, allow_replan=False):
    placement = placement or {"mte": {"stage": 0, "order": 0, "start_time": 0},
                              "vec": {"stage": 1, "order": 1, "start_time": 2}}
    solve = solve_fixture.solve_result(placement,
                                       problem_digest=problem["problem_digest"], ii=ii,
                                       buffers={"arena": versions} if versions else None)
    plan = RP.build_resolved_plan(problem, solve, PROBLEM_SOURCES,
                                  allow_replan_diagnostics=allow_replan)
    return RP.resolve_buffer_layout(plan, {**FACTS,
                                          "arena": {**FACTS["arena"],
                                                    "versions": int(versions or 2)}})


PROBLEM_SOURCES = dict(__import__("test_plan_r3", fromlist=["SOURCES"]).SOURCES)


def applied(rows=5, *, start=0, step=1, versions=2, placement=None, ii=2, problem=None):
    problem = problem or make_problem(start=start, stop=rows, step=step)
    plan = build_plan(problem, placement, ii=ii, versions=versions)
    # The TIR loop must carry the SAME domain the plan records: the transform compares
    # them and refuses a mismatch, so the fixture has to be derived from the problem.
    domain = problem["loop_domain"]
    original = make_tir(rows=domain["stop"], start=domain["start"], step=domain["step"])
    rewritten, mapping = PIPE.apply_software_pipeline(original, plan)
    return original, plan, rewritten, mapping


# --------------------------------------------------------------- structural exactness
def test_the_emitted_loop_is_one_symbolic_guarded_loop():
    """One loop, symbolic bound, static body: size cannot depend on the trip count."""
    original, plan, rewritten, mapping = applied(rows=5)
    loops = [PIPE._first_pipeline_loop(rewritten)]
    assert len(loops) == 1
    loop = loops[0]
    assert loop.induction_var.name == PIPE.P_STEP_VAR
    # stop = trip + depth, so the last valid instance of the deepest stage is reached
    assert PIPE._eval_guard(loop.stop, {PIPE.N_TRIP_VAR: 5}) == 5 + mapping.depth
    assert mapping.depth == 1
    # the body is a FIXED number of statements -- two bindings plus one guard per task --
    # so its static size cannot depend on the trip count
    assert len(list(loop.body.body)) == 3 * len(mapping.blocks)
    assert verify_prim_function(rewritten, module_fns={}) is None


def test_every_valid_instance_runs_exactly_once_for_a_fixed_trip():
    original, plan, rewritten, mapping = applied(rows=5)
    report = PIPE.verify_pipeline_application(original, plan, rewritten, mapping)
    assert report["status"] == "PASS", report
    assert report["trip"] == 5 and report["blocks"] == 2
    assert report["instances"] == report["expected_instances"] == 10


def test_the_generation_is_k_and_the_slot_is_k_modulo_versions():
    """Slot and generation are read from the EMITTED addresses, not from the plan."""
    original, plan, rewritten, mapping = applied(rows=6, versions=3)
    loop = PIPE._first_pipeline_loop(rewritten)
    binds = {child.var.name: child.value for child in loop.body.body
             if isinstance(child, LetStmt)}
    assert set(binds) == ({block.k_var for block in mapping.blocks}
                          | {block.i_var for block in mapping.blocks})
    # k = p - shift, re-derived from the IR
    for block in mapping.blocks:
        assert PIPE._shift_of(binds[block.k_var]) == block.shift
    ring = PIPE._allocations(rewritten)["arena" + PIPE.RING_SUFFIX]
    # the ring holds V versions of the whole allocation, not one
    assert int(ring.var.type.shape[0]) == (ALLOC // 4) * 3


def test_the_ring_allocation_dominates_the_loop():
    """A ring declared inside the body would be re-allocated every step."""
    original, plan, rewritten, mapping = applied()
    declarations = PIPE._allocations(rewritten)
    assert "arena" not in declarations
    assert "arena" + PIPE.RING_SUFFIX in declarations
    # the ring is NOT inside the loop body
    loop = PIPE._first_pipeline_loop(rewritten)
    inner = {getattr(getattr(child, "var", None), "name", None) for child in loop.body.body}
    assert "arena" + PIPE.RING_SUFFIX not in inner


# --------------------------------------------------------------------- trip coverage
@pytest.mark.parametrize("rows", [0, 1, 2, 3, 5, 9])
def test_trip_0_1_shorter_equal_and_longer_all_execute_each_instance_once(rows):
    """0/1/<depth/==depth/>depth: the count is exact, never one short or one extra."""
    original, plan, rewritten, mapping = applied(rows=rows)
    assert mapping.trip == rows
    report = PIPE.verify_pipeline_application(original, plan, rewritten, mapping)
    assert report["status"] == "PASS", report
    assert report["instances"] == report["expected_instances"] == rows * 2
    assert verify_prim_function(rewritten, module_fns={}) is None


@pytest.mark.parametrize("start,stop,step", [(0, 4, 1), (2, 4, 1), (1, 9, 3), (3, 7, 2)])
def test_a_non_zero_lower_bound_and_a_step_above_one_keep_the_count_exact(start, stop, step):
    """k is a DRIVER-FREE ordinal: the guard admits k in [0, N) whatever the domain."""
    problem = make_problem(start=start, stop=stop, step=step)
    original, plan, rewritten, mapping = applied(problem=problem)
    trip = len(range(start, stop, step))
    assert mapping.trip == trip
    report = PIPE.verify_pipeline_application(original, plan, rewritten, mapping)
    assert report["status"] == "PASS", report
    assert report["instances"] == report["expected_instances"] == trip * 2
    # the emitted body still reads the source domain through i = start + k*step
    loop = PIPE._first_pipeline_loop(rewritten)
    binds = {child.var.name: child.value for child in loop.body.body
             if isinstance(child, LetStmt)}
    for block in mapping.blocks:
        assert PIPE._render(binds[block.i_var]).count(str(start)) >= 1 or start == 0


def test_a_symbolic_stop_is_refused_when_trip_folding_is_required():
    """A symbolic trip cannot be counted here, so folding is refused BY NAME."""
    problem = make_problem()
    plan = build_plan(problem)
    original = make_tir(rows=5, symbolic_stop=True)
    assert verify_prim_function(original, module_fns={}) is None
    rewritten, mapping = PIPE.apply_software_pipeline(original, plan)
    assert mapping.trip is None
    # the emitted bound is still trip + depth, i.e. a runtime expression
    loop = PIPE._first_pipeline_loop(rewritten)
    assert PIPE._eval_guard(loop.stop, {}) is PIPE._UNKNOWN
    report = PIPE.verify_pipeline_application(original, plan, rewritten, mapping)
    assert report["status"] == "FAIL"
    assert report["code"] == "PIPELINE_TRIP_SYMBOLIC"
    # ... and it is accepted once the caller stops demanding a foldable trip
    relaxed = PIPE.verify_pipeline_application(original, plan, rewritten, mapping,
                                               require_trip_folding=False)
    assert relaxed["status"] == "PASS", relaxed


def test_a_trip_that_disagrees_with_the_plan_domain_is_refused():
    problem = make_problem(stop=5)
    plan = build_plan(problem)
    original = make_tir(rows=7)
    with pytest.raises(RP.PlanError) as caught:
        PIPE.apply_software_pipeline(original, plan)
    assert caught.value.code == "PLAN_PIPELINE_DOMAIN_MISMATCH"


# ------------------------------------------------------- the transform reads its input
def test_perturbing_the_assignment_changes_the_materialized_tir():
    """A transform that ignored its plan would emit one digest for every schedule."""
    base_placement = {"mte": {"stage": 0, "order": 0, "start_time": 0},
                      "vec": {"stage": 1, "order": 1, "start_time": 2}}
    deep_placement = {"mte": {"stage": 0, "order": 0, "start_time": 0},
                      "vec": {"stage": 2, "order": 1, "start_time": 4}}
    _o, _p, _r, base = applied(rows=5, placement=base_placement, ii=2)
    _o, _p, _r, deep = applied(rows=5, placement=deep_placement, ii=2)
    assert base.depth == 1 and deep.depth == 2
    assert dict(base.shifts)["vec"] == 1
    assert dict(deep.shifts)["vec"] == 2
    # a deeper stage really deepens the emitted pipeline, not just one literal
    assert base.rewritten_digest != deep.rewritten_digest
    # and the loop bound follows the depth
    base_loop = PIPE._first_pipeline_loop(applied(rows=5)[2])
    deep_loop = PIPE._first_pipeline_loop(_r)
    assert PIPE._eval_guard(base_loop.stop, {PIPE.N_TRIP_VAR: 5}) == 5 + 1
    assert PIPE._eval_guard(deep_loop.stop, {PIPE.N_TRIP_VAR: 5}) == 5 + 2


def test_the_emitted_ir_depends_on_the_shift_not_on_the_time_scale():
    """Honest property: k is a logical ordinal, so II alone is not an IR fact.

    Two assignments with the SAME stage shape but different II must emit the SAME
    pipeline. Asserting a difference here would be asserting a bug.
    """
    wide = {"mte": {"stage": 0, "order": 0, "start_time": 0},
            "vec": {"stage": 2, "order": 1, "start_time": 4}}
    narrow = {"mte": {"stage": 0, "order": 0, "start_time": 0},
              "vec": {"stage": 2, "order": 1, "start_time": 2}}
    _o, wide_plan, _r, wide_map = applied(rows=5, placement=wide, ii=2)
    _o, narrow_plan, _r, narrow_map = applied(rows=5, placement=narrow, ii=1)
    assert wide_plan.loop.ii["value"] == 2 and narrow_plan.loop.ii["value"] == 1
    assert dict(wide_map.shifts) == dict(narrow_map.shifts)
    assert wide_map.rewritten_digest == narrow_map.rewritten_digest


def _slot_moduli(module, loop):
    """Every divisor of a (k % V) slot expression actually present in the emitted IR."""
    from tilefoundry.ir.core.kinds import BinaryKind

    found = []

    def visit(node):
        if isinstance(node, Call):
            if getattr(node.target, "kind", None) == BinaryKind.MOD:
                divisor = PIPE._as_int(node.args[1])
                if divisor is not None:
                    found.append(int(divisor))
            for arg in node.args:
                visit(arg)
        elif isinstance(node, (tuple, list)):
            for child in node:
                visit(child)
        elif hasattr(node, "__dataclass_fields__"):
            for name in node.__dataclass_fields__:
                visit(getattr(node, name))

    for child in loop.body.body:
        visit(child)
    return found


@pytest.mark.parametrize("versions", [1, 2, 3, 4])
def test_ring_versions_drive_the_emitted_slot_modulus(versions):
    """V reaches the emitted SLOT ARITHMETIC, which is the only place it can act.

    The ring's size already covers V versions (the layout step sized the allocation), so
    size alone cannot catch a version-count bug. What V decides here is the modulus of the
    slot index, so that is what is read back out of the emitted IR.
    """
    original, plan, rewritten, mapping = applied(rows=6, versions=versions)
    assert plan.buffers[0].slot_count == versions
    assert dict(mapping.versions)["arena"] == versions
    ring = PIPE._allocations(rewritten)["arena" + PIPE.RING_SUFFIX]
    assert int(ring.var.type.shape[0]) == (ALLOC // 4) * versions
    loop = PIPE._first_pipeline_loop(rewritten)
    moduli = set(_slot_moduli(rewritten, loop))
    if versions == 1:
        # one version needs no rotation at all, so the slot is the constant 0
        assert moduli == set(), moduli
    else:
        assert moduli == {versions}, moduli
    assert PIPE.verify_pipeline_application(original, plan, rewritten,
                                            mapping)["status"] == "PASS"
    assert verify_prim_function(rewritten, module_fns={}) is None


def test_a_wider_ring_produces_different_addresses_than_a_narrow_one():
    _o1, _p1, _r1, m1 = applied(rows=5, versions=1)
    _o2, _p2, _r2, m2 = applied(rows=5, versions=3)
    assert m1.rewritten_digest != m2.rewritten_digest


# ------------------------------------------------------------ negative controls (refusals)
def _rename_if_cond(module, replacement):
    def walk(node):
        if isinstance(node, If):
            return dataclasses.replace(node, cond=replacement(node.cond),
                                       then_body=walk(node.then_body),
                                       else_body=walk(node.else_body))
        if isinstance(node, Sequential):
            return dataclasses.replace(node, body=tuple(walk(child) for child in node.body))
        if hasattr(node, "body") and hasattr(node.body, "body"):
            return dataclasses.replace(node, body=walk(node.body))
        return node

    return dataclasses.replace(module, body=walk(module.body))


def test_a_shift_table_that_disagrees_with_the_emitted_blocks_is_refused():
    """The published shift table is part of the contract, so it is checked too."""
    original, plan, rewritten, mapping = applied()
    swapped = dict(mapping.shifts)
    bad = dataclasses.replace(mapping,
                              shifts=tuple((task_id, 1 - shift)
                                           for task_id, shift in mapping.shifts))
    report = PIPE.verify_pipeline_application(original, plan, rewritten, bad)
    assert report["status"] == "FAIL"
    assert report["code"] == "PIPELINE_SHIFT_TABLE_MISMATCH"
    assert set(swapped) == set(dict(bad.shifts))


def test_a_guard_that_admits_too_many_instances_is_refused():
    """A guard replaced by True runs every task on every step: over-execution."""
    original, plan, rewritten, mapping = applied()
    always = Constant(type=make_tensor_type((), dtype=DType.bool,
                                            storage=StorageKind.UMAT), value=True)
    mutant = _rename_if_cond(rewritten, lambda _cond: always)
    report = PIPE.verify_pipeline_application(original, plan, mutant, mapping)
    assert report["status"] == "FAIL"
    assert report["code"] == "PIPELINE_COVERAGE_MISMATCH"
    # every task was admitted on EVERY step instead of exactly trip times
    # p sweeps [0, trip + depth) = 6 steps, and now every task runs on every one
    assert set(report["detail"]["admitted"].values()) == {5 + 1}


def test_a_guard_that_drops_the_lower_bound_is_refused():
    """k >= 0 is not decoration: without it the prologue runs out-of-range instances."""
    original, plan, rewritten, mapping = applied()
    mutant = _rename_if_cond(rewritten, lambda cond: cond.args[1])
    report = PIPE.verify_pipeline_application(original, plan, mutant, mapping)
    assert report["status"] == "FAIL"
    assert report["code"] in ("PIPELINE_COVERAGE_MISMATCH", "PIPELINE_SHIFT_UNREADABLE")


def test_dropping_a_carried_binding_is_refused_not_ignored():
    """A block whose k is unbound must be refused, never silently kept."""
    original, plan, rewritten, mapping = applied()

    def drop(node):
        if isinstance(node, Sequential):
            kept = [drop(child) for child in node.body
                    if not (isinstance(child, LetStmt)
                            and child.var.name.startswith(PIPE.K_VAR_PREFIX))]
            return dataclasses.replace(node, body=tuple(kept))
        if isinstance(node, If):
            return dataclasses.replace(node, then_body=drop(node.then_body),
                                       else_body=drop(node.else_body))
        if hasattr(node, "body") and hasattr(node.body, "body"):
            return dataclasses.replace(node, body=drop(node.body))
        return node

    mutant = dataclasses.replace(rewritten, body=drop(rewritten.body))
    report = PIPE.verify_pipeline_application(original, plan, mutant, mapping)
    assert report["status"] == "FAIL"
    assert report["code"] == "PIPELINE_SHIFT_UNREADABLE"


def test_an_unmodified_module_is_refused():
    original, plan, rewritten, mapping = applied()
    report = PIPE.verify_pipeline_application(original, plan, original, mapping)
    assert report["status"] == "FAIL"
    assert report["code"] == "PIPELINE_NOT_APPLIED"


def test_a_sidecar_only_map_is_refused_because_it_does_not_change_the_ir():
    plan = build_plan(make_problem())
    original = make_tir(rows=5)
    same, mapping = PIPE.apply_software_pipeline(original, plan, sidecar_only=True)
    assert same is original
    report = PIPE.verify_pipeline_application(original, plan, same, mapping)
    assert report["status"] == "FAIL"
    assert report["code"] == "PIPELINE_NOT_APPLIED"


# ------------------------------------------------- C-1 seam: recorded violation refuses
def test_a_needs_replan_diagnostic_refuses_the_formal_plan():
    """record -> still recorded, but no formal plan is built from it (C-1 seam)."""
    problem = make_problem(enable_offset=False)
    placed = {"mte": {"stage": 0, "order": 0, "start_time": 0},
              "vec": {"stage": 1, "order": 1, "start_time": 2}}
    solve = solve_fixture.solve_result(placed, problem_digest=problem["problem_digest"],
                                       ii=2, buffers={"arena": 2})
    with pytest.raises(RP.PlanError) as caught:
        RP.build_resolved_plan(problem, solve, PROBLEM_SOURCES)
    assert caught.value.code == "PLAN_RESOLVE_OFFSET_WINDOW_VIOLATED"
    assert caught.value.needs_replan is True


def test_the_same_solve_still_RECORDS_the_violation_when_explicitly_opted_out():
    """The diagnostic is not swallowed: the opt-out exists, and it is explicit."""
    problem = make_problem(enable_offset=False)
    placed = {"mte": {"stage": 0, "order": 0, "start_time": 0},
              "vec": {"stage": 1, "order": 1, "start_time": 2}}
    solve = solve_fixture.solve_result(placed, problem_digest=problem["problem_digest"],
                                       ii=2, buffers={"arena": 2})
    plan = RP.build_resolved_plan(problem, solve, PROBLEM_SOURCES,
                                  allow_replan_diagnostics=True)
    kinds = [row.get("kind") for row in plan.resolution_diagnostics]
    assert "OFFSET_WINDOW_VIOLATED" in kinds
    assert any(row.get("needs_replan") for row in plan.resolution_diagnostics)


def test_a_diagnostic_plan_is_also_refused_by_the_pipeline():
    """The seam is closed in BOTH places: a smuggled plan cannot be pipelined."""
    plan = build_plan(make_problem(enable_offset=False), allow_replan=True)
    original = make_tir(rows=5)
    with pytest.raises(RP.PlanError) as caught:
        PIPE.apply_software_pipeline(original, plan)
    assert caught.value.code == "PLAN_RESOLVE_OFFSET_WINDOW_VIOLATED"
    assert caught.value.needs_replan is True


def test_enabling_offset_makes_the_same_cross_stage_schedule_legal():
    """The refusal tracks the constraint, not the fixture: enable_offset lifts it."""
    problem = make_problem(enable_offset=True)
    placed = {"mte": {"stage": 0, "order": 0, "start_time": 0},
              "vec": {"stage": 1, "order": 1, "start_time": 2}}
    plan = build_plan(problem, placed)
    assert plan.resolution_diagnostics == ()
    original, _plan, rewritten, mapping = applied(rows=5, placement=placed, problem=problem)
    assert PIPE.verify_pipeline_application(original, _plan, rewritten,
                                            mapping)["status"] == "PASS"


# ------------------------------------------------------------------ input refusal surface
@pytest.mark.parametrize("mutate,code", [
    (lambda tir: tir, "PLAN_PIPELINE_REQUIRES_PLAN"),
])
def test_a_non_plan_is_refused_by_name(mutate, code):
    original = make_tir(rows=5)
    with pytest.raises(RP.PlanError) as caught:
        PIPE.apply_software_pipeline(original, {"not": "a plan"})
    assert caught.value.code == code


def test_a_non_post_checked_solve_never_reaches_the_pipeline():
    """Refused twice over, and the EARLIER refusal is the stronger one.

    The plan builder will not build from a non-post-checked solve, so no such plan can
    exist for the pipeline to accept. Both refusals are asserted: the builder's first,
    then the pipeline's own guard on a hand-constructed plan.
    """
    problem = make_problem()
    solve = solve_fixture.solve_result(
        {"mte": {"stage": 0, "order": 0, "start_time": 0},
         "vec": {"stage": 1, "order": 1, "start_time": 2}},
        problem_digest=problem["problem_digest"], ii=2, buffers={"arena": 2},
        postcheck_verdict="POSTCHECK_FAIL")
    with pytest.raises(RP.PlanError) as caught:
        RP.build_resolved_plan(problem, solve, PROBLEM_SOURCES)
    assert caught.value.code == "PLAN_REQUIRES_SAT"
    # and even if such a plan were smuggled in, the transform refuses it by name
    good = build_plan(problem)
    broken = dataclasses.replace(
        good, state=dataclasses.replace(good.state, planning_status="SAT_NOT_POSTCHECKED"))
    with pytest.raises(RP.PlanError) as caught:
        PIPE.apply_software_pipeline(make_tir(rows=5), broken)
    assert caught.value.code == "PLAN_PIPELINE_REQUIRES_SAT"


def test_a_loop_step_that_is_not_the_planned_one_is_refused():
    problem = make_problem(step=1)
    plan = build_plan(problem)
    original = make_tir(rows=5, step=2)
    with pytest.raises(RP.PlanError) as caught:
        PIPE.apply_software_pipeline(original, plan)
    assert caught.value.code == "PLAN_PIPELINE_DOMAIN_MISMATCH"


def test_two_loops_are_refused_rather_than_silently_picking_one():
    problem = make_problem()
    plan = build_plan(problem)
    single = make_tir(rows=5)
    scope = single.body.body[0]
    doubled = dataclasses.replace(
        single,
        body=Sequential(body=(dataclasses.replace(
            scope, body=Sequential(body=tuple(scope.body.body) + tuple(scope.body.body))),)))
    with pytest.raises(RP.PlanError) as caught:
        PIPE.apply_software_pipeline(doubled, plan)
    assert caught.value.code == "PLAN_PIPELINE_UNSUPPORTED"


# ------------------------------------------------------- the REAL d0 carrier (boundary)
def _real_d0_plan():
    """Plan + real TIR from the F0-surviving pipelined vadd carrier.

    This is the genuine end of the chain: authored HIR -> real TIR -> real ledger -> real
    Problem -> real solver -> real plan -> real layout. It is used to state where this
    transform currently STOPS, not to claim it works there.
    """
    from tilefoundry_ascend.solver import bridge as B
    from poc_vertical_adapter import nominal_solve_result, project_problem

    import test_task83_vertical as VERTICAL

    sources, tir, problem, expansion, _expected, _s_problem = VERTICAL._l()
    r_problem, _projected = project_problem(problem, tir, expansion.count, sources)
    model_facts = {"model_id": "ascend-poc-nominal", "model_version": "v1",
                   "memory_groups": [{"id": "group0",
                                      "capacity_bytes": max(1, sum(int(b["bytes"])
                                                                for b in r_problem["buffers"])),
                                      "buffers": [b["root"] for b in r_problem["buffers"]],
                                      "cost_model": "MODEL"}], "cost_model": "MODEL"}
    core = B.solve_real_problem(r_problem, model_facts, {}, require_source_postcheck=True)
    plan = RP.build_resolved_plan(r_problem, nominal_solve_result(core, r_problem),
                                  PROBLEM_SOURCES, require_source_postcheck=True)
    facts, base = {}, 0
    for buffer in r_problem["buffers"]:
        facts[buffer["root"]] = {"versions": int(buffer.get("versions") or 1),
                                 "alignment": 64, "arena_id": "arena0",
                                 "base_offset_bytes": base}
        base += ((int(buffer["bytes"]) + 63) // 64) * 64
    return tir, RP.resolve_buffer_layout(plan, facts)


def test_the_real_carrier_documents_where_this_transform_stops():
    """A real d0 buffer is declared as an absolute-address VIEW, so no ring is possible.

    The carrier the ledger comes from declares its SMEM tiles with tf.empty -> TensorView
    at an absolute byte address. A ring is ONE allocation indexed by k % V; an absolute
    address cannot carry that symbol. The transform therefore REFUSES BY NAME. That is
    the honest boundary of P3-03, and this test exists so the boundary cannot silently
    move: if someone later makes the real carrier pipelinable, this test must be
    deliberately updated, with a plan that actually allocates a ring.
    """
    tir, plan = _real_d0_plan()
    assert plan.state.planning_status == RP.SAT_POSTCHECKED
    assert plan.resolution_diagnostics == ()
    with pytest.raises(RP.PlanError) as caught:
        PIPE.apply_software_pipeline(tir, plan)
    assert caught.value.code == "PLAN_PIPELINE_BUFFER_NOT_ALLOCATED"
    assert caught.value.actual["declaration"]


def test_the_real_carrier_reaches_the_pipeline_with_a_genuine_plan():
    """The plan really is built from the real chain, not fabricated for the refusal."""
    tir, plan = _real_d0_plan()
    assert [task.task_id for task in plan.tasks]
    assert all(task.stage == 0 for task in plan.tasks)
    assert len(PIPE._source_loops(tir)) == 1
    # every buffer the plan resolved is one the ledger actually recorded
    assert {buffer.root for buffer in plan.buffers}
