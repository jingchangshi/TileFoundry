"""C0-C2 tests: source obligations, static actual Set/Wait/Drain, independent coverage.

Run: PYTHONPATH=src:targets/ascend python targets/ascend/tests/test_completion_c0_c2.py
No device; a fake typed-TIR module and simple R-Plan-shaped fixture only.
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

from tilefoundry.ir.tir.memory.copy import Copy
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate, Sequential

from tilefoundry_ascend import sync as C

CAPS = {"supported_pairs": [["mte2", "vector"]], "same_pipe_pipe_order": False,
        "flag_pool_size": 4}


def make_plan(trip=3):
    return SimpleNamespace(
        loop=SimpleNamespace(domain={"start": 0, "stop": trip, "step": 1},
                             source_loop_id="main"),
        tasks=[SimpleNamespace(task_id="t0", pipe="mte2"),
               SimpleNamespace(task_id="t1", pipe="vector")],
        buffers=[SimpleNamespace(root="x", slot_count=2, version_stride_bytes=64,
                                 base_offset_bytes=0, allocation_bytes=128, owner="main")],
        accesses=[SimpleNamespace(access_id="a0", task_id="t0", buffer_root="x", mode="write"),
                  SimpleNamespace(access_id="a1", task_id="t1", buffer_root="x", mode="read")],
    )


def make_problem(trip=3, deps=None):
    return {"dependencies": deps if deps is not None else [{
        "source_id": "dep-raw", "producer": "t0", "consumer": "t1", "kind": "RAW",
        "distance": 0, "shared_root": "x"}]}


def make_tir():
    # one real typed effect statement; the completion nodes are added by inject
    return PrimFunction(name="c_fixture", params=(),
                        body=Sequential(body=(Evaluate(callable=Copy(), args=()),)),
                        output_count=0)


def derive_and_choose(problem=None, plan=None, caps=None):
    problem = problem or make_problem()
    plan = plan or make_plan()
    caps = caps or CAPS
    obligations = C.derive_completion_obligations(problem, plan, completion_capabilities=caps)
    channels, discharges = C.choose_completion_mechanisms(obligations, {}, caps)
    allocation = C.allocate_completion_resources(channels, {}, caps)
    return problem, plan, caps, obligations, channels, discharges, allocation


def expect_code(code, fn):
    try:
        fn()
    except Exception as error:  # noqa: BLE001
        got = getattr(error, "code", None)
        if got != code:
            raise AssertionError("expected %s, got %r: %s" % (code, got, error))
        return error
    raise AssertionError("expected %s, but nothing was refused" % code)


def test_positive_1_raw_v2_k0_to_2_verified():
    problem, plan, caps, obligations, channels, discharges, allocation = derive_and_choose()
    assert len(obligations) == 1, obligations
    assert obligations[0].kind == "READY" and obligations[0].versions == 2
    assert len(channels) == 1 and len(discharges) == 1
    actual, placement = C.inject_completion(make_tir(), plan, obligations, allocation,
                                             channels=channels)
    report = C.verify_completion_coverage(problem, plan, actual, placement, caps,
                                           obligations=obligations)
    assert report.status == "VERIFIED", report.as_dict()
    assert report.required == 1 and report.covered == 1
    assert len(placement["placements"]) == 7  # 3 SET + 3 WAIT + 1 DRAIN
    assert actual is not make_tir()


def test_positive_channels_are_derived_not_fixed_six():
    obligations = C.derive_completion_obligations(make_problem(), make_plan(),
                                                   completion_capabilities=CAPS)
    channels, _ = C.choose_completion_mechanisms(obligations, {}, CAPS)
    assert len(channels) == len(obligations) == 1, channels
    assert channels[0].covered_obligation_ids == (obligations[0].obligation_id,)


def test_missing_source_access_is_refused():
    plan = make_plan()
    plan.accesses = plan.accesses[:1]  # no consumer access for t1
    expect_code("COMPLETION_SOURCE_COVERAGE_MISSING",
                lambda: C.derive_completion_obligations(make_problem(), plan,
                                                         completion_capabilities=CAPS))


def test_capability_unknown_and_pool_exhausted_are_named():
    obligations = C.derive_completion_obligations(make_problem(), make_plan(),
                                                   completion_capabilities=CAPS)
    expect_code("COMPLETION_CAPABILITY_UNKNOWN",
                lambda: C.choose_completion_mechanisms(
                    obligations, {}, {"supported_pairs": [["cube", "vector"]],
                                      "same_pipe_pipe_order": False}))
    expect_code("FLAG_POOL_EXHAUSTED",
                lambda: C.allocate_completion_resources(
                    C.choose_completion_mechanisms(obligations, {}, CAPS)[0], {},
                    {"flag_pool_size": 0}))


def test_delete_one_necessary_wait_is_completion_not_verified():
    problem, plan, caps, obligations, channels, _d, allocation = derive_and_choose()
    actual, placement = C.inject_completion(make_tir(), plan, obligations, allocation,
                                             channels=channels)
    # Remove one actual Wait node, leaving the map claiming it still exists.
    kept = []
    removed = False
    for stmt in actual.body.body:
        if (not removed and isinstance(stmt, Evaluate)
                and type(stmt.callable).__name__ == "FlagWait"):
            removed = True
            continue
        kept.append(stmt)
    broken = PrimFunction(name=actual.name, params=actual.params,
                          body=Sequential(body=tuple(kept)), output_count=actual.output_count,
                          target=actual.target)
    report = C.verify_completion_coverage(problem, plan, broken, placement, caps,
                                           obligations=obligations)
    assert report.status == "FAILED"
    assert report.code == "COMPLETION_NOT_VERIFIED", report.as_dict()


def test_forged_set_wait_metadata_without_nodes_is_rejected():
    problem, plan, caps, obligations, channels, _d, allocation = derive_and_choose()
    _actual, placement = C.inject_completion(make_tir(), plan, obligations, allocation,
                                             channels=channels)
    # Claim a complete plan, but hand in the original module with no actual Set/Wait/Drain.
    forged = C.verify_completion_coverage(problem, plan, make_tir(), placement, caps,
                                          obligations=obligations)
    assert forged.status == "FAILED" and forged.code in (
        "COMPLETION_NOT_VERIFIED", "COMPLETION_MAP_MISMATCH"), forged.as_dict()


def test_missing_obligation_is_completion_not_verified():
    problem, plan, caps, _obligations, _ch, _d, _a = derive_and_choose()
    # A verifier that omits the one source-derived obligation must fail, even with
    # an otherwise valid actual placement.
    report = C.verify_completion_coverage(problem, plan, make_tir(), {"placements": ()},
                                          caps, obligations=())
    assert report.status == "FAILED" and report.code == "COMPLETION_NOT_VERIFIED"
    assert report.missing_obligation_ids


def test_no_actual_nodes_is_not_verified_and_no_drain_is_named():
    problem, plan, caps, obligations, _ch, _d, _a = derive_and_choose()
    report = C.verify_completion_coverage(problem, plan, make_tir(), {"placements": ()},
                                          caps, obligations=obligations)
    assert report.status == "FAILED"
    assert report.code in ("COMPLETION_NOT_VERIFIED", "FINAL_DRAIN_MISSING")


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
