"""L6: real BF16 vcvt/quantisation chain + N70/loop mutation capture.

Part A proves the BF16 example ACTUALLY runs a conversion chain: the ledger carries Vcvt
actions with elem_width=16 / part=0 / direction, and Vsts with pack=PK_B32 -- not merely a
dtype label. Escape edges are checked with the full escape-edge/2 payload.

Part B proves budget / versions / dependency-window changes are CAPTURED: mutating the loop
span changes an access window and therefore at least one dependency's window (and the problem
digest); changing budget or buffer versions changes the policy and the digest.

Exit codes: 0 PASS, 1 FAIL, 2 ERROR, 3 NOT_RUN.
"""
from __future__ import annotations

# ruff: noqa: PLC0415, I001  -- local imports keep each check's dependency surface explicit

import functools
import os
import sys
import types

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "targets", "ascend")]
FIX = os.path.join(ROOT, "targets", "ascend", "tests", "fixtures", "rmsnorm", "rmsnorm_row.py")


@functools.lru_cache(maxsize=1)
def _bf16_ledger():
    from tilefoundry.cli.source import checked_program
    from tilefoundry.analysis.api import analyze
    from tilefoundry.schedule.finalize import finalize
    from tilefoundry_ascend.schedule.ledger import extract_ledger

    prog = checked_program(FIX + ":RMSNormRow", None, command="l6-bf16")
    analyzed = analyze(prog.module, prog.entry, analysis=("memory",))
    module = finalize(analyzed, authored=prog.authored)
    return extract_ledger(
        module.entry_function(),
        escape_records=module.metadata.get("escape_records"),
        declaration_sites=module.metadata.get("escape_declaration_sites"),
    )


def _acc(task, root, mode, off, footprint):
    from tilefoundry_ascend.schedule.ledger import ProblemAccess

    return ProblemAccess(
        task=task, instruction=task, root=root, root_kind="real", space="GMEM", mode=mode,
        byte_footprint=footprint, byte_offset=off, source_offset="[]", attrs=(),
        offset_structure=(("literal", off),), guard=(), normalized_offset=off,
        offset_status=("BYTES" if off is not None else "OFFSET_UNKNOWN"),
        logical_iteration="t", iteration_path="i", iteration_extent="1", iteration_domains=(("i", "4"),),
    )


def _fake_problem(dependencies, policy):
    return types.SimpleNamespace(
        schema="real-problem/1",
        problem_digest="0" * 16,
        ledger_sha="ledger",
        source_digest="source",
        policy=policy,
        dependencies=list(dependencies),
        as_dict=lambda: {"dependencies": list(dependencies), "policy": policy},
    )


def _windows(dependencies):
    return sorted(
        (d["kind"], d["status"], d["producer_window"], d["consumer_window"]) for d in dependencies
    )


def _digest(document):
    from tilefoundry_ascend.schedule.ledger import _digest_of_problem_dict

    return _digest_of_problem_dict(document["problem"])


TESTS = {}


def _test(fn):
    TESTS[fn.__name__] = fn
    return fn


@_test
def test_bf16_really_runs_vcvt_and_packed_store():
    ledger = _bf16_ledger()
    vcvt = [e for e in ledger.entries if e.op == "Vcvt"]
    assert vcvt, "no Vcvt action in the BF16 example"
    quantised = [e for e in vcvt if ("elem_width", 16) in e.attrs and ("part", 0) in e.attrs]
    assert quantised, "Vcvt actions do not carry elem_width=16 / part=0"
    assert any(("direction", "forward") in e.attrs or ("direction", "reverse") in e.attrs for e in vcvt)
    packed = [e for e in ledger.entries if e.op == "Vsts" and ("pack", "PK_B32") in e.attrs]
    assert packed, "no PK_B32 packed store: the store is not quantised"


@_test
def test_bf16_escape_edges_full_payload():
    from tilefoundry_ascend.schedule.ledger import verify_escape_endpoints

    ledger = _bf16_ledger()
    assert ledger.escape_edges, "no escape edge in the BF16 example"
    report = verify_escape_endpoints(ledger)
    assert report["complete"] is True
    for edge in ledger.escape_edges:
        assert edge.schema_version == "escape-edge/2", edge.schema_version
        assert edge.domain, "escape edge lacks its domain"
        assert edge.inner_type and edge.outer_type, "escape edge lacks typed endpoints"


@_test
def test_n70_span_mutation_changes_a_dependency_window():
    from tilefoundry_ascend.schedule.ledger import derive_dependencies

    baseline = derive_dependencies([
        _acc("w0", "buf", "write", 0, 64), _acc("r0", "buf", "read", 0, 64),
        _acc("w1", "buf", "write", 64, 6), _acc("r1", "buf", "read", 64, 6),  # N70 tail: 6 lanes
    ])
    assert len(baseline) == 2 and all(d["kind"] == "RAW" and d["status"] == "BYTES" for d in baseline)
    assert (64, 70) in {d["producer_window"] for d in baseline}, _windows(baseline)

    # Mutate the span of the tail issue: 6 -> 64 bytes.
    mutated = derive_dependencies([
        _acc("w0", "buf", "write", 0, 64), _acc("r0", "buf", "read", 0, 64),
        _acc("w1", "buf", "write", 64, 64), _acc("r1", "buf", "read", 64, 64),
    ])
    assert _windows(baseline) != _windows(mutated), "span change did not change a dependency window"


@_test
def test_n70_span_mutation_changes_the_digest():
    from tilefoundry_ascend.schedule.ledger import build_s_problem_document, derive_dependencies

    policy = {"status": "WIRED", "requested_budget": {"loop[0]:i": 2}, "buffer_versions": {"b": 1}}
    baseline = derive_dependencies([
        _acc("w0", "buf", "write", 0, 64), _acc("r0", "buf", "read", 0, 64),
        _acc("w1", "buf", "write", 64, 6), _acc("r1", "buf", "read", 64, 6),
    ])
    mutated = derive_dependencies([
        _acc("w0", "buf", "write", 0, 64), _acc("r0", "buf", "read", 0, 64),
        _acc("w1", "buf", "write", 64, 64), _acc("r1", "buf", "read", 64, 64),
    ])
    before = build_s_problem_document(_fake_problem(baseline, policy))
    after = build_s_problem_document(_fake_problem(mutated, policy))
    assert _digest(before) != _digest(after), "span mutation did not change the problem digest"


@_test
def test_budget_and_versions_mutations_change_the_digest():
    from tilefoundry_ascend.schedule.ledger import build_s_problem_document, derive_dependencies

    deps = derive_dependencies([
        _acc("w0", "buf", "write", 0, 64), _acc("r0", "buf", "read", 0, 64),
    ])
    base_policy = {"status": "WIRED", "requested_budget": {"loop[0]:i": 2}, "buffer_versions": {"b": 1}}
    budget_mutated = {"status": "WIRED", "requested_budget": {"loop[0]:i": 3}, "buffer_versions": {"b": 1}}
    versions_mutated = {"status": "WIRED", "requested_budget": {"loop[0]:i": 2}, "buffer_versions": {"b": 2}}
    base = _digest(build_s_problem_document(_fake_problem(deps, base_policy)))
    assert base != _digest(build_s_problem_document(_fake_problem(deps, budget_mutated))), "budget mutation not captured"
    assert base != _digest(build_s_problem_document(_fake_problem(deps, versions_mutated))), "versions mutation not captured"


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
