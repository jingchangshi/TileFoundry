"""task-82: mechanical POC instance expansion (i in [0,4)) + 12 RAW obligations.

Expansion is derived from the typed loop domain and offset structure (never a hard-coded
window table); wrong count/stride and non-affine offsets are refused by name.
"""
from __future__ import annotations

# ruff: noqa: PLC0415, I001

import dataclasses
import functools
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "targets", "ascend")]
FIX = os.path.join(ROOT, "targets", "ascend", "tests", "fixtures", "d0_pipeline", "pipe_vadd_carrier.py")


@functools.lru_cache(maxsize=1)
def _built():
    from tilefoundry.cli.source import checked_program
    from tilefoundry.analysis.api import analyze
    from tilefoundry.schedule.finalize import finalize
    from tilefoundry_ascend.schedule.ledger import build_problem, extract_ledger

    prog = checked_program(FIX + ":PipeVaddCarrier", None, command="task82-expand")
    analyzed = analyze(prog.module, prog.entry, analysis=("memory",))
    module = finalize(analyzed, authored=prog.authored)
    ledger = extract_ledger(module.entry_function())
    return ledger, build_problem(ledger)


def _refusal(call):
    from tilefoundry_ascend.schedule.ledger import LedgerError

    try:
        call()
    except LedgerError as error:
        return str(error).split(":", 1)[0]
    return None


TESTS = {}


def _test(fn):
    TESTS[fn.__name__] = fn
    return fn


@_test
def test_expansion_is_derived_and_four_instances():
    from tilefoundry_ascend.schedule.poc_expansion import expand_loop_instances

    ledger, _problem = _built()
    expansion = expand_loop_instances(ledger)
    assert expansion.count == 4 and expansion.stop == 4 and expansion.step == 1, expansion
    assert expansion.stride_bytes == 256, expansion.stride_bytes
    provenance = expansion.to_provenance()
    assert provenance["poc_instance_expansion"] is True and provenance["count"] == 4
    assert len(provenance["instances"]) == 4


@_test
def test_gm_windows_follow_loop_domain_not_a_constant_table():
    from tilefoundry_ascend.schedule.poc_expansion import expand_loop_instances

    ledger, _problem = _built()
    expansion = expand_loop_instances(ledger)
    root_of = {e.action_id: e.root for e in ledger.entries}
    space_of = {e.action_id: e.space for e in ledger.entries}
    for inst in expansion.instances:
        i = inst["i"]
        for task, window in inst["windows"].items():
            if space_of[task] == "GMEM":
                assert window == (i * 256, i * 256 + 256), (i, root_of[task], window)
            elif space_of[task] == "SMEM":
                assert window == (0, 256), (i, root_of[task], window)


@_test
def test_twelve_raw_obligations_present():
    from tilefoundry_ascend.schedule.poc_expansion import expand_loop_instances

    ledger, problem = _built()
    expansion = expand_loop_instances(ledger)
    ops = {e.action_id: e.op for e in ledger.entries}
    spaces = {e.action_id: e.space for e in ledger.entries}
    load_edges = 0
    store_edges = 0
    for edge in problem.dependencies:
        if edge["kind"] != "RAW" or edge["status"] != "BYTES":
            continue
        p, c = edge["producer"], edge["consumer"]
        if spaces.get(p) == "SMEM" and spaces.get(c) == "SMEM":
            if ops.get(p) == "GmToUb" and ops.get(c) == "Vld":
                load_edges += 1
            if ops.get(p) == "Vsts" and ops.get(c) == "copy":
                store_edges += 1
    assert load_edges >= 2, ("need X-in and Z-in load->VF RAW", load_edges)
    assert store_edges >= 1, ("need Y-out VF->store RAW", store_edges)
    total_instances = (load_edges + store_edges) * expansion.count
    assert total_instances == 12, total_instances


@_test
def test_bad_expansion_count_is_refused():
    from tilefoundry_ascend.schedule.poc_expansion import expand_loop_instances, verify_expansion

    ledger, _problem = _built()
    expansion = expand_loop_instances(ledger)
    for wrong in (3, 5):
        mutant = dataclasses.replace(expansion, count=wrong)
        assert _refusal(lambda: verify_expansion(mutant, expected_count=4, expected_stride_bytes=256)) == "POC_EXPANSION_COUNT_MISMATCH"


@_test
def test_bad_stride_is_refused():
    from tilefoundry_ascend.schedule.poc_expansion import expand_loop_instances, verify_expansion

    ledger, _problem = _built()
    expansion = expand_loop_instances(ledger)
    mutant = dataclasses.replace(expansion, stride_bytes=64)  # i*64 vs i*256 mixed
    assert _refusal(lambda: verify_expansion(mutant, expected_count=4, expected_stride_bytes=256)) == "POC_EXPANSION_STRIDE_MISMATCH"


@_test
def test_non_affine_offset_is_unsupported():
    from tilefoundry_ascend.schedule.poc_expansion import _offset_bytes

    fake = types.SimpleNamespace(offset_structure=(("call", "DimMul", [("var", "i"), ("var", "i")], []),))
    assert _refusal(lambda: _offset_bytes(fake, {"i"})) == "POC_EXPANSION_UNSUPPORTED"


def main() -> int:
    passed = failed = errored = 0
    for name, case in TESTS.items():
        try:
            case()
        except AssertionError as error:
            failed += 1
            print("FAIL %s: %s" % (name, error))
        except Exception as error:  # noqa: BLE001
            errored += 1
            print("ERROR %s: %s: %s" % (name, type(error).__name__, error))
        else:
            passed += 1
            print("PASS %s" % name)
    print("%d/%d passed, %d failed, %d errored" % (passed, len(TESTS), failed, errored))
    if errored:
        return 2
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
