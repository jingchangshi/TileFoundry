"""R4/R5 tests: phase domains + stage-shifted access mapping + C attach/X boundary.

Run: PYTHONPATH=src:targets/ascend python targets/ascend/tests/test_plan_r4_r5.py
Static CPU only; no device.
"""
from __future__ import annotations

import dataclasses
import sys
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "targets" / "ascend"))
sys.path.insert(0, str(REPO / "targets" / "ascend" / "tests"))

from tilefoundry.ir.tir.flag import Drain
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate, Sequential

from tilefoundry_ascend.plan import materialize as MAT
from tilefoundry_ascend.plan import resolved_plan as RP
from tilefoundry_ascend import sync as C
import test_plan_r3 as R3


def stage_problem():
    problem = R3.make_problem(rows=5)
    problem["stage_depth"] = 2
    # The consumer view is stage-shifted: its ordinal is i+1, not current i/k.
    problem["accesses"][1]["execution_ordinal"] = RP.Add(RP.V("i"), RP.C(1))
    return problem


def build_stage():
    plan = RP.build_resolved_plan(stage_problem(), R3.make_solve(rows=5), R3.SOURCES)
    return RP.resolve_buffer_layout(plan, R3.FACTS)


def expect_code(code, fn):
    try:
        fn()
    except Exception as error:  # noqa: BLE001
        got = getattr(error, "code", None)
        if got != code:
            raise AssertionError("expected %s, got %r: %s" % (code, got, error))
        return error
    raise AssertionError("expected %s, but nothing was refused" % code)


def test_r4_phase_domains_and_stage_shift_are_explicit_and_disjoint():
    plan = build_stage()
    loop = plan.loop
    assert loop.stage_depth == 2 and loop.stage_shift == 1
    assert loop.prologue_domain == {"start": 0, "stop": 1, "step": 1}
    assert loop.steady_domain == {"start": 1, "stop": 4, "step": 1}
    assert loop.drain_domain == {"start": 4, "stop": 5, "step": 1}
    assert RP.validate_phase_domains(loop)["valid"] is True
    # non-overlap and in-loop bounds
    assert loop.prologue_domain["stop"] <= loop.steady_domain["start"]
    assert loop.steady_domain["stop"] <= loop.drain_domain["start"]
    assert loop.drain_domain["stop"] <= loop.domain["stop"]


def test_r4_access_k_comes_from_accessplan_not_current_loop_i():
    plan = build_stage()
    rows = MAT.replay_plan_addresses(plan)
    # c1-dst is shifted by +1: at loop k=0 it consumes ordinal 1 / slot 1 / 96..128,
    # NOT the current loop slot 0 / 32..64.
    k0 = [row for row in rows if row["access_id"] == "acc-c1-dst" and row["k"] == 0][0]
    assert k0["generation"] == 1 and k0["slot"] == 1
    assert (k0["start"], k0["end"]) == (96, 128)
    k2 = [row for row in rows if row["access_id"] == "acc-c1-dst" and row["k"] == 2][0]
    assert k2["generation"] == 3 and k2["slot"] == 1


def test_r4_materialization_has_phase_controls_and_terminal_drain():
    plan = build_stage()
    original = R3.make_tir_fixture()
    rewritten, materialization = MAT.materialize_plan(original, plan)
    report = MAT.verify_plan_application(original, plan, rewritten, materialization)
    assert report["status"] == "PASS", report
    drains = [stmt for stmt in MAT.index_statements(rewritten).values()
              if isinstance(stmt, Evaluate) and isinstance(stmt.callable, Drain)]
    assert len(drains) == 1
    # top-level phase control nodes: prologue, steady, drain
    top = [stmt for stmt in rewritten.body.body if hasattr(stmt, "then_body")]
    assert len(top) >= 3


def test_r4_missing_drain_is_plan_drain_missing():
    plan = build_stage()
    original = R3.make_tir_fixture()
    rewritten, materialization = MAT.materialize_plan(original, plan)
    kept = [stmt for stmt in rewritten.body.body
            if not (isinstance(stmt, Evaluate) and isinstance(stmt.callable, Drain))]
    broken = PrimFunction(name=rewritten.name, params=rewritten.params,
                          body=Sequential(body=tuple(kept)), output_count=rewritten.output_count,
                          target=rewritten.target)
    report = MAT.verify_plan_application(original, plan, broken, materialization)
    assert report["status"] == "FAIL" and report["code"] == "PLAN_DRAIN_MISSING", report


def test_r4_bad_stage_mapping_and_overlap_are_named():
    plan = build_stage()
    bad_access = dataclasses.replace(plan.accesses[1], execution_ordinal=None)
    bad_plan = dataclasses.replace(plan, accesses=(plan.accesses[0], bad_access) + plan.accesses[2:])
    expect_code("PLAN_STAGE_MAPPING_INVALID",
                lambda: MAT.materialize_plan(R3.make_tir_fixture(), bad_plan))
    overlap = dataclasses.replace(plan.loop,
                                  prologue_domain={"start": 0, "stop": 3, "step": 1},
                                  steady_domain={"start": 2, "stop": 4, "step": 1})
    expect_code("PLAN_PHASE_DOMAIN_OVERLAP", lambda: RP.validate_phase_domains(overlap))


def test_r5_attach_requires_real_verified_C_coverage():
    plan = build_stage()
    failed = C.CoverageReport("FAILED", "COMPLETION_NOT_VERIFIED", 1, 1, 0, ("ob-1",), 0)
    expect_code("COMPLETION_NOT_VERIFIED", lambda: RP.attach_completion(plan, failed))
    expect_code("COMPLETION_NOT_VERIFIED", lambda: RP.attach_completion(plan, None))
    verified = C.CoverageReport("VERIFIED", None, 1, 1, 1, (), 7)
    attached = RP.attach_completion(plan, verified)
    assert attached.state.completion_status == RP.COMPLETION_VERIFIED
    assert all(sync.status == RP.COMPLETION_VERIFIED for sync in attached.syncs)


def test_r5_x_interface_is_named_unimplemented_not_fake_success():
    plan = build_stage()
    expect_code("COMPLETION_NOT_VERIFIED", lambda: RP.emit_for_codegen(plan))
    verified = C.CoverageReport("VERIFIED", None, 1, 1, 1, (), 7)
    attached = RP.attach_completion(plan, verified)
    expect_code("CODEGEN_CONTRACT_UNIMPLEMENTED", lambda: RP.emit_for_codegen(attached))
    expect_code("CODEGEN_CONTRACT_UNIMPLEMENTED",
                lambda: RP.emit_for_codegen(attached, {"implemented": True}))


if __name__ == "__main__":
    failed = 0
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn()
                print("PASS", _name)
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print("FAIL", _name, type(exc).__name__, exc)
    raise SystemExit(1 if failed else 0)


# --------------------------------------------------------------------------------------
# REACHABILITY controls: a rewritten WRITE that sits in a statically dead branch is not an
# applied write.
#
# The defect these pin down: index_statements_ledger() walked If(False).then AND
# If(True).else unconditionally, and verify_plan_application() judged "applied" purely by
# `new_fingerprint in rewritten_fingerprints`. So a REAL rewritten write wrapped in If(False)
# still verified PASS with applied_accesses unchanged at 12. Same family as C's Set/Wait
# reachability: one predicate, granted to Drain but withheld from Set/Wait, and here to
# nothing at all.
# --------------------------------------------------------------------------------------

def _wrap_in_dead_if(module, predicate, branch="then", constant=False):
    """Wrap EVERY statement matching `predicate` in a statically dead If arm.

    All matches are wrapped in ONE pass, keyed by object identity. Two reasons:
      * the fixture contains DUPLICATE statements with identical fingerprints (measured: the
        chosen write's fingerprint occurs 3x), so wrapping only the first leaves another copy
        reachable and the control passes for the wrong reason;
      * a repeated pass would re-wrap its own output forever (the original statement is still
        inside the wrapper, so it keeps matching).
    """
    from tilefoundry.ir.core import Constant
    from tilefoundry.ir.tir.stmts import If
    from tilefoundry.ir.types import make_tensor_type

    cond_type = make_tensor_type((), dtype="i1", storage="umat")
    cond = Constant(value=constant, type=cond_type)
    sealed_ids = set()

    def wrap_once(item):
        body = Sequential(body=(item,))
        empty = Sequential(body=())
        inner = (If(cond=cond, then_body=body, else_body=empty) if branch == "then"
                 else If(cond=cond, then_body=empty, else_body=body))
        sealed_ids.add(id(item))
        return Sequential(body=(inner,))

    def rewrite(node):
        if not dataclasses.is_dataclass(node):
            return None
        for field in dataclasses.fields(node):
            value = getattr(node, field.name, None)
            if isinstance(value, (list, tuple)):
                changed = False
                body = list(value)
                for index, item in enumerate(body):
                    if predicate(item) and id(item) not in sealed_ids:
                        body[index] = wrap_once(item)
                        changed = True
                        continue
                    found = rewrite(item)
                    if found is not None:
                        body[index] = found
                        changed = True
                if changed:
                    return dataclasses.replace(node, **{field.name: tuple(body)})
            elif dataclasses.is_dataclass(value):
                if predicate(value) and id(value) not in sealed_ids:
                    return dataclasses.replace(node, **{field.name: wrap_once(value)})
                found = rewrite(value)
                if found is not None:
                    return dataclasses.replace(node, **{field.name: found})
        return None

    mutant = rewrite(module)
    assert mutant is not None, "could not place the statement in an If"
    assert sealed_ids, "predicate matched nothing"
    return mutant, len(sealed_ids)


def test_r4_rewritten_write_in_dead_branch_is_refused_by_name():
    """A real rewritten WRITE sealed in If(False) must be refused, not counted as applied."""
    plan = build_stage()
    original = R3.make_tir_fixture()
    rewritten, materialization = MAT.materialize_plan(original, plan)
    baseline = MAT.verify_plan_application(original, plan, rewritten, materialization)
    assert baseline["status"] == "PASS", baseline
    # Baseline is asserted against the real count for THIS fixture (10 writes x 1 trip).
    # The 12/12 figure belongs to the vertical flow's kernel, not this fixture: hardcoding 12
    # here would make the control fail for a reason unrelated to reachability.
    assert baseline["applied_accesses"] == baseline["expected_applications"], baseline
    assert baseline["applied_accesses"] > 0, baseline

    # Choose a write the map actually recorded, so the mutation moves a REAL application.
    recorded = {entry["new_fingerprint"] for entry in materialization.entries}
    target_path, target = None, None
    for path, stmt in MAT.index_statements_ledger(rewritten).items():
        if MAT.fingerprint(stmt) in recorded and isinstance(stmt, Evaluate):
            target_path, target = path, stmt
            break
    assert target is not None, "no recorded statement found in the rewritten tree"

    def _match(item):
        return isinstance(item, Evaluate) and MAT.fingerprint(item) == MAT.fingerprint(target)

    mutant, _count = _wrap_in_dead_if(rewritten, _match, "then", constant=False)
    report = MAT.verify_plan_application(original, plan, mutant, materialization)
    assert report["status"] == "FAIL", report
    assert report["code"] == "PLAN_WRITE_UNREACHABLE", report
    assert report["applied_accesses"] == 0, report


def test_r4_rewritten_write_in_dead_else_of_if_true_is_refused():
    """If(True) kills the ELSE arm; that direction must be refused too."""
    plan = build_stage()
    original = R3.make_tir_fixture()
    rewritten, materialization = MAT.materialize_plan(original, plan)
    recorded = {entry["new_fingerprint"] for entry in materialization.entries}
    target = next(stmt for stmt in MAT.index_statements_ledger(rewritten).values()
                  if MAT.fingerprint(stmt) in recorded and isinstance(stmt, Evaluate))

    def _match(item):
        return isinstance(item, Evaluate) and MAT.fingerprint(item) == MAT.fingerprint(target)

    mutant, _count = _wrap_in_dead_if(rewritten, _match, "else", constant=True)
    report = MAT.verify_plan_application(original, plan, mutant, materialization)
    assert report["status"] == "FAIL", report
    assert report["code"] == "PLAN_WRITE_UNREACHABLE", report


def test_r4_reachable_if_branches_are_not_wrongly_dropped():
    """POSITIVE control: If(True).then and a run-time If must keep their statements counted.

    Without this, "ignore everything inside any If" would satisfy both negatives above while
    silently discarding real applications.
    """
    from tilefoundry.ir.core import Constant, Var
    from tilefoundry.ir.types import make_tensor_type

    plan = build_stage()
    original = R3.make_tir_fixture()
    rewritten, materialization = MAT.materialize_plan(original, plan)
    recorded = {entry["new_fingerprint"] for entry in materialization.entries}
    target = next(stmt for stmt in MAT.index_statements_ledger(rewritten).values()
                  if MAT.fingerprint(stmt) in recorded and isinstance(stmt, Evaluate))
    cond_type = make_tensor_type((), dtype="i1", storage="umat")

    def _match(item):
        return isinstance(item, Evaluate) and MAT.fingerprint(item) == MAT.fingerprint(target)

    # If(True) -> the then arm executes, so the write still counts and R still PASSes.
    live, _ = _wrap_in_dead_if(rewritten, _match, "then", constant=True)
    report = MAT.verify_plan_application(original, plan, live, materialization)
    assert report["status"] == "PASS", report
    baseline_applied = MAT.verify_plan_application(
        original, plan, rewritten, materialization)["applied_accesses"]
    assert report["applied_accesses"] == baseline_applied, report

    # A run-time condition decides nothing statically -> both arms reachable -> still PASS.
    from tilefoundry.ir.tir.stmts import If
    runtime = dataclasses.replace(
        rewritten,
        body=Sequential(body=(If(cond=Var(name="runtime_flag", type=cond_type),
                                  then_body=rewritten.body,
                                  else_body=Sequential(body=())),)))
    report2 = MAT.verify_plan_application(original, plan, runtime, materialization)
    assert report2["status"] == "PASS", report2
    assert report2["applied_accesses"] == baseline_applied, report2


# --------------------------------------------------------------------------------------
# LOOP reachability for R: a write inside a loop that provably never runs is not applied.
#
# Same family as the If(False) repair above, one statement kind over: the census learned
# about If arms but still descended into every loop body unconditionally. The architect
# reported (and I reproduced) that a GmToUb write moved into a zero-trip For still PASSes
# with applied 12 after the digest is backfilled; While(False) refused, but under the
# inaccurate MATERIALIZATION_MAP_MISMATCH.
# --------------------------------------------------------------------------------------

def _wrap_in_loop(module, predicate, kind, **bounds):
    """Move every statement matching `predicate` into a loop, in ONE identity-keyed pass.

    Keyed by id() so a statement is wrapped exactly once: the fixture contains duplicate
    statements with identical fingerprints, and a second pass would re-wrap its own output.
    """
    from tilefoundry.ir.core import Constant, Var
    from tilefoundry.ir.tir.stmts import For, While
    from tilefoundry.ir.types import make_tensor_type

    def int_const(value):
        return Constant(value=value, type=make_tensor_type((), dtype="i32", storage="umat"))

    sealed_ids = set()

    def wrap_once(item):
        sequence = Sequential(body=(item,))
        if kind == "while":
            cond = Constant(value=bounds["cond"],
                            type=make_tensor_type((), dtype="i1", storage="umat"))
            loop = While(cond=cond, body=sequence)
        else:
            loop = For(induction_var=Var(name="i", type=make_tensor_type((), dtype="i32",
                                                                       storage="umat")),
                       start=int_const(bounds["start"]), stop=int_const(bounds["stop"]),
                       step=int_const(bounds["step"]), body=sequence)
        sealed_ids.add(id(item))
        return Sequential(body=(loop,))

    def rewrite(node):
        if not dataclasses.is_dataclass(node):
            return None
        for field in dataclasses.fields(node):
            value = getattr(node, field.name, None)
            if isinstance(value, (list, tuple)):
                changed = False
                body = list(value)
                for index, item in enumerate(body):
                    if predicate(item) and id(item) not in sealed_ids:
                        body[index] = wrap_once(item)
                        changed = True
                        continue
                    found = rewrite(item)
                    if found is not None:
                        body[index] = found
                        changed = True
                if changed:
                    return dataclasses.replace(node, **{field.name: tuple(body)})
            elif dataclasses.is_dataclass(value):
                if predicate(value) and id(value) not in sealed_ids:
                    return dataclasses.replace(node, **{field.name: wrap_once(value)})
                found = rewrite(value)
                if found is not None:
                    return dataclasses.replace(node, **{field.name: found})
        return None

    mutant = rewrite(module)
    assert mutant is not None, "could not place the statement in a loop"
    assert sealed_ids, "predicate matched nothing"
    return mutant, len(sealed_ids)


def _r4_loop_case(kind, **bounds):
    """Shared setup: move a recorded write into a loop and verify the result."""
    plan = build_stage()
    original = R3.make_tir_fixture()
    rewritten, materialization = MAT.materialize_plan(original, plan)
    baseline = MAT.verify_plan_application(original, plan, rewritten, materialization)
    assert baseline["status"] == "PASS", baseline
    baseline_applied = baseline["applied_accesses"]
    assert baseline_applied > 0, baseline

    recorded = {entry["new_fingerprint"] for entry in materialization.entries}
    target = None
    for _path, stmt in MAT.index_statements_ledger(rewritten).items():
        if MAT.fingerprint(stmt) in recorded and isinstance(stmt, Evaluate):
            target = stmt
            break
    assert target is not None, "no recorded statement found in the rewritten tree"

    def _match(item):
        return isinstance(item, Evaluate) and MAT.fingerprint(item) == MAT.fingerprint(target)

    mutant, count = _wrap_in_loop(rewritten, _match, kind, **bounds)
    assert count == 3, "expected the 3 duplicate copies to be sealed, got %d" % count
    return plan, original, materialization, mutant, baseline_applied


def test_r4_rewritten_write_in_while_false_is_refused_by_name():
    """While(False) must refuse with the SAME named code as a dead If arm."""
    plan, original, materialization, mutant, baseline_applied = _r4_loop_case("while", cond=False)
    report = MAT.verify_plan_application(original, plan, mutant, materialization)
    assert report["status"] == "FAIL", report
    assert report["code"] == "PLAN_WRITE_UNREACHABLE", report
    assert report["applied_accesses"] == 0, report
    assert baseline_applied > 0


def test_r4_rewritten_write_in_zero_trip_for_is_refused_by_name():
    """The form the If-only repair missed: a write in a zero-trip counted loop."""
    plan, original, materialization, mutant, baseline_applied = _r4_loop_case(
        "for", start=0, stop=0, step=1)
    report = MAT.verify_plan_application(original, plan, mutant, materialization)
    assert report["status"] == "FAIL", report
    # Named code, not the vague MATERIALIZATION_MAP_MISMATCH the architect observed.
    assert report["code"] == "PLAN_WRITE_UNREACHABLE", report
    assert report["applied_accesses"] == 0, report
    assert baseline_applied > 0


def test_r4_rewritten_write_in_reachable_loops_is_not_wrongly_dropped():
    """POSITIVE control: While(True) and a running For must still count the write.

    An unprovable trip count must never be assumed zero -- doing so would discard real
    applied writes, the opposite failure and just as wrong.
    """
    plan = build_stage()
    original = R3.make_tir_fixture()
    rewritten, materialization = MAT.materialize_plan(original, plan)
    baseline = MAT.verify_plan_application(original, plan, rewritten, materialization)
    assert baseline["status"] == "PASS", baseline
    baseline_applied = baseline["applied_accesses"]

    recorded = {entry["new_fingerprint"] for entry in materialization.entries}
    target = None
    for _path, stmt in MAT.index_statements_ledger(rewritten).items():
        if MAT.fingerprint(stmt) in recorded and isinstance(stmt, Evaluate):
            target = stmt
            break
    assert target is not None

    def _match(item):
        return isinstance(item, Evaluate) and MAT.fingerprint(item) == MAT.fingerprint(target)

    for kind, bounds in (
        ("while", {"cond": True}),          # While(True): body always runs
        ("for", {"start": 0, "stop": 4, "step": 1}),   # ascending, runs
        ("for", {"start": 4, "stop": 0, "step": -1}),  # descending, runs
        ("for", {"start": 0, "stop": 1, "step": 1}),   # single iteration
    ):
        mutant, _count = _wrap_in_loop(rewritten, _match, kind, **bounds)
        report = MAT.verify_plan_application(original, plan, mutant, materialization)
        assert report["status"] == "PASS", (kind, bounds, report)
        assert report["applied_accesses"] == baseline_applied, (kind, bounds, report)


def test_r4_loop_body_can_execute_matches_direct_measurement():
    """The loop predicate itself, on the boundary cases a naive rule gets wrong."""
    from tilefoundry.ir.core import Constant, Var
    from tilefoundry.ir.tir.stmts import For, While
    from tilefoundry.ir.types import make_tensor_type

    int_type = make_tensor_type((), dtype="i32", storage="umat")
    bool_type = make_tensor_type((), dtype="i1", storage="umat")

    def k(value):
        return Constant(value=value, type=int_type)

    def alloy(start, stop, step):
        return For(induction_var=Var(name="i", type=int_type), start=k(start), stop=k(stop),
                   step=k(step), body=Sequential(body=()))

    assert MAT._loop_body_can_execute(While(
        cond=Constant(value=False, type=bool_type), body=Sequential(body=()))) is False
    assert MAT._loop_body_can_execute(While(
        cond=Constant(value=True, type=bool_type), body=Sequential(body=()))) is True
    assert MAT._loop_body_can_execute(While(
        cond=Var(name="rt", type=bool_type), body=Sequential(body=()))) is True

    assert MAT._loop_body_can_execute(alloy(0, 4, 1)) is True
    assert MAT._loop_body_can_execute(alloy(4, 0, -1)) is True     # descending runs
    assert MAT._loop_body_can_execute(alloy(0, 1, 1)) is True      # one iteration
    assert MAT._loop_body_can_execute(alloy(0, 0, 1)) is False     # empty
    assert MAT._loop_body_can_execute(alloy(5, 2, 1)) is False     # empty ascending
    assert MAT._loop_body_can_execute(alloy(2, 5, -1)) is False    # empty descending
    assert MAT._loop_body_can_execute(alloy(0, 4, 0)) is False     # non-progressing

    # A run-time bound cannot be shown empty, so the body stays reachable.
    assert MAT._loop_body_can_execute(For(
        induction_var=Var(name="i", type=int_type), start=k(0),
        stop=Var(name="n", type=int_type), step=k(1),
        body=Sequential(body=()))) is True
