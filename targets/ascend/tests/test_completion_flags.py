"""Task-83 C anchor-aware reuse tests (targeted, CPU only)."""
from __future__ import annotations
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "targets" / "ascend"))
sys.path.insert(0, str(REPO / "targets" / "ascend" / "tests"))
from tilefoundry_ascend import sync as C
from tilefoundry_ascend.schedule.ledger import extract_ledger
import test_task83_vertical as T


def _materialised_carrier(tir, problem, sources):
    """The carrier the REAL C flow injects into: R's materialised, unrolled pipeline.

    Injecting into the un-lowered For body is not the flow under test. That body has ONE
    statement per action and no per-iteration instances, so a generation-1 Wait cannot be
    placed after a generation-0 Set without a prologue the static POC does not emit; C now
    refuses that shape by name (COMPLETION_STARTUP_UNSATISFIED, see
    test_completion_ordering.py) instead of counting it as coverage.
    """
    from poc_vertical_adapter import nominal_solve_result, project_problem
    from tilefoundry_ascend.plan import materialize as MAT
    from tilefoundry_ascend.plan import resolved_plan as RP
    from tilefoundry_ascend.solver import bridge as B

    r_problem, _ = project_problem(problem, tir, 4, sources)
    model_facts = {
        "model_id": "m", "model_version": "v1",
        "memory_groups": [{
            "id": "g0",
            "capacity_bytes": max(1, sum(int(b["bytes"]) for b in r_problem["buffers"])),
            "buffers": [b["root"] for b in r_problem["buffers"]],
            "cost_model": "MODEL",
        }],
        "cost_model": "MODEL",
    }
    # P3-02: run the REAL source postcheck and let the plan builder consume the REAL
    # assignment. The _solve_status patch that used to stand in for this is gone: it
    # labelled a fabricated schedule as SAT_POSTCHECKED, so this test exercised completion
    # behaviour for a plan the production path could never actually build.
    core = B.solve_real_problem(r_problem, model_facts, {}, require_source_postcheck=True)
    assert core.state == "SAT", core.state
    assert (core.postcheck or {}).get("verdict") == "POSTCHECK_PASS", core.postcheck
    nominal = nominal_solve_result(core, r_problem)
    plan = RP.build_resolved_plan(r_problem, nominal,
                                  {"L": "l", "S": "s", "C": "c", "F": "f", "P": "p"},
                                  require_source_postcheck=True)
    base, facts = 0, {}
    for buf in r_problem["buffers"]:
        facts[buf["root"]] = {"versions": 1, "alignment": 64, "arena_id": "arena0",
                              "base_offset_bytes": base}
        base += ((buf["bytes"] + 63) // 64) * 64
    plan = RP.resolve_buffer_layout(plan, facts)
    rewritten, matmap = MAT.materialize_plan(tir, plan)
    return rewritten, matmap


def setup():
    sources, tir, problem, expansion, expected, _sp = T._l()
    ledger = extract_ledger(tir)
    reuse = C.derive_reuse_obligations_from_ledger(ledger, expansion)
    required = C.derive_completion_obligations_from_ledger(ledger, expansion, problem=problem,
                                                           expected=expected)
    caps = {"supported_pairs": [["vector", "DMA"], ["DMA", "vector"]],
            "same_pipe_pipe_order": False, "flag_pool_size": 64}
    actionable = tuple(o for o in required if o.kind in ("RAW", "WAR", "FREE"))
    channels, _ = C.choose_completion_mechanisms(actionable, {}, caps)
    allocation = C.allocate_completion_resources(channels, {}, caps)
    rewritten, matmap = _materialised_carrier(tir, problem, sources)
    return (rewritten, ledger, problem, expansion, expected, reuse, required, caps, channels,
            allocation, matmap)


def ctx_rewritten():
    """The injected carrier BEFORE injection: already completion-bearing since P4-03."""
    return setup()[0]


def test_reuse_is_18_and_requires_both_actions():
    *_, reuse, required, _caps, _channels, _allocation, _matmap = setup()
    assert len(reuse) == 18
    assert len(required) == 47
    assert {o.kind for o in reuse} == {"WAR", "FREE"}


def test_anchored_injection_covers_47():
    (tir, ledger, problem, expansion, expected, reuse, required, caps, channels, allocation,
     matmap) = setup()
    actionable = tuple(o for o in required if o.kind in ("RAW", "WAR", "FREE"))
    actual, placement = C.inject_completion_anchored(tir, None, actionable, allocation,
                                                     channels=channels, ledger=ledger,
                                                     materialization=matmap)
    nodes = C._actual_nodes(actual)
    # THE CARRIER ALREADY CARRIES ITS COMPLETION BLOCK. Since P4-03 the injected program is
    # R3's materialised program WITH completion emitted into it (16 Sets, 14 Waits, 1 Drain),
    # and inject_completion_anchored adds the 30/30 the obligations require on top. Asserting
    # the bare 30/30 therefore asserts something false about a correct program; what matters is
    # that the required edges were ADDED and there is still exactly one region exit.
    base = C._actual_nodes(ctx_rewritten())
    assert len(nodes["set"]) == len(base["set"]) + 30
    assert len(nodes["wait"]) == len(base["wait"]) + 30
    assert len(nodes["drain"]) == 1
    report = C.verify_completion_coverage_from_ledger(
        ledger, expansion, problem=problem, actual_tir=actual, placement_map=placement,
        capabilities=caps, expected=expected)
    assert report.status == "VERIFIED", report.as_dict()
    assert report.required == 47 and report.covered == 47
    assert report.covered_by_kind == {"RAW": 12, "SSA": 16, "WAR_FREE": 18, "FINAL": 1}
    assert len(report.edge_table) == 18


def test_missing_wait_is_not_counted():
    (tir, ledger, problem, expansion, expected, reuse, required, caps, channels, allocation,
     matmap) = setup()
    actionable = tuple(o for o in required if o.kind in ("RAW", "WAR", "FREE"))
    actual, placement = C.inject_completion_anchored(
        tir, None, actionable, allocation, channels=channels, ledger=ledger,
        materialization=matmap, drop_wait_ids=(reuse[0].obligation_id,))
    report = C.verify_completion_coverage_from_ledger(
        ledger, expansion, problem=problem, actual_tir=actual, placement_map=placement,
        capabilities=caps, expected=expected)
    assert report.status == "FAILED" and report.code == "COMPLETION_NOT_VERIFIED", report.as_dict()
    assert report.covered_by_kind["WAR_FREE"] == 17
    assert report.covered == 46


def _single_actual(reuse, channel, allocation, *, reverse=False, wrong_pipe=False):
    import dataclasses
    from tilefoundry.ir.tir.flag import Drain, FlagSet, FlagWait
    from tilefoundry.ir.tir.prim_function import PrimFunction
    from tilefoundry.ir.tir.stmts import Evaluate, Sequential
    ob = reuse[0]
    set_call = C._flag_node(FlagSet, ob, channel, allocation, int(ob.generation_domain[0]))
    wait_call = C._flag_node(FlagWait, ob, channel, allocation, int(ob.generation_domain[0]))
    if wrong_pipe:
        attrs = dict(vars(set_call))
        attrs["producer_pipe"] = "DMA"
        set_call = FlagSet(**attrs)
    nodes = [Evaluate(callable=wait_call, args=()), Evaluate(callable=set_call, args=())] if reverse else [Evaluate(callable=set_call, args=()), Evaluate(callable=wait_call, args=())]
    body = Sequential(body=tuple(nodes) + (Evaluate(callable=Drain(), args=()),))
    return PrimFunction(name="c_probe", params=(), body=body, output_count=0)


def test_anchor_missing_is_named_not_counted():
    import dataclasses
    (tir, ledger, problem, expansion, expected, reuse, required, caps, channels, allocation,
     matmap) = setup()
    bad = dataclasses.replace(reuse[0], producer_action="missing-action")
    try:
        C.inject_completion_anchored(tir, None, (bad,), allocation,
                                     channels=channels, ledger=ledger,
                                     materialization=matmap)
    except Exception as error:  # noqa: BLE001
        assert getattr(error, "code", None) == "COMPLETION_ANCHOR_NOT_FOUND", error
    else:
        raise AssertionError("missing anchor was accepted")


def test_cross_pipe_same_order_lie_is_named():
    *_, reuse, _required, _caps, _channels, _allocation, _matmap = setup()
    try:
        C.choose_completion_mechanisms(reuse, {}, {"supported_pairs": [["vector", "DMA"]],
                                                   "same_pipe_pipe_order": True})
    except Exception as error:  # noqa: BLE001
        assert getattr(error, "code", None) == "COMPLETION_MECHANISM_UNPROVEN", error
    else:
        raise AssertionError("cross-pipe SAME_PIPE_ORDER lie was accepted")


def test_pool_exhausted_is_named():
    *_, reuse, _required, caps, channels, _allocation, _matmap = setup()
    try:
        C.allocate_completion_resources(channels, {}, {"flag_pool_size": 5})
    except Exception as error:  # noqa: BLE001
        assert getattr(error, "code", None) == "FLAG_POOL_EXHAUSTED", error
    else:
        raise AssertionError("pool exhaustion was not refused")


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn(); print("PASS", name)
            except Exception as exc:  # noqa: BLE001
                failed += 1; print("FAIL", name, type(exc).__name__, exc)
    raise SystemExit(1 if failed else 0)
