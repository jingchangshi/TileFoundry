"""L5: independent S-facing Problem serialization + UNKNOWN-edge contract + coverage gap.

Proves canonical text is deterministic and round-trips, that a mutated field / dropped policy
changes the digest, that kind x status counts and unknown_ratio are reported, that UNKNOWN
edges are excluded from S hard constraints, and that a high unknown ratio surfaces the NAMED
gap WINDOW_COVERAGE_LOW instead of handing S a huge mostly-unknown graph.

Exit codes: 0 PASS, 1 FAIL, 2 ERROR, 3 NOT_RUN.
"""
from __future__ import annotations

# ruff: noqa: PLC0415, I001  -- local imports keep each check's dependency surface explicit

import copy
import functools
import os
import pathlib
import sys
import tempfile
import types

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "targets", "ascend")]
FIX = os.path.join(ROOT, "targets", "ascend", "tests", "fixtures", "d0_pipeline", "pipe_vadd_carrier.py")


@functools.lru_cache(maxsize=1)
def _built():
    from tilefoundry.cli.source import checked_program
    from tilefoundry.analysis.api import analyze
    from tilefoundry.schedule.finalize import finalize
    from tilefoundry_ascend.schedule.ledger import (
        build_problem, capture_problem_sources, extract_ledger,
    )

    prog = checked_program(FIX + ":PipeVaddCarrier", None, command="l5-serialize")
    sources = capture_problem_sources(prog.module)
    analyzed = analyze(prog.module, prog.entry, analysis=("memory",))
    module = finalize(analyzed, authored=prog.authored)
    ledger = extract_ledger(module.entry_function())
    return build_problem(ledger, sources)


def _refusal(call):
    from tilefoundry_ascend.schedule.ledger import LedgerError

    try:
        call()
    except LedgerError as error:
        return str(error).split(":", 1)[0]
    return None


def _access(task, root, mode, off, footprint):
    from tilefoundry_ascend.schedule.ledger import ProblemAccess

    return ProblemAccess(
        task=task, instruction=task, root=root, root_kind="real", space="GMEM", mode=mode,
        byte_footprint=footprint, byte_offset=off, source_offset="[]", attrs=(),
        offset_structure=(("literal", off),), guard=(), normalized_offset=off,
        offset_status=("BYTES" if off is not None else "OFFSET_UNKNOWN"),
        logical_iteration="outside", iteration_path="outside", iteration_extent="1",
        iteration_domains=(),
    )


def _fake_problem(dependencies):
    return types.SimpleNamespace(
        schema="real-problem/1",
        problem_digest="0" * 16,
        ledger_sha="ledger",
        source_digest="source",
        dependencies=list(dependencies),
        as_dict=lambda: {"dependencies": list(dependencies), "policy": {}},
    )


TESTS = {}


def _test(fn):
    TESTS[fn.__name__] = fn
    return fn


@_test
def test_document_is_deterministic_and_round_trips():
    from tilefoundry_ascend.schedule.ledger import (
        S_PROBLEM_SCHEMA, read_s_problem, serialize_s_problem, verify_s_problem_document,
        write_s_problem,
    )

    problem = _built()
    first = serialize_s_problem(problem)
    second = serialize_s_problem(problem)
    assert first == second, "canonical serialization is not deterministic"
    directory = pathlib.Path(tempfile.mkdtemp(prefix="tf_l5_"))
    path = directory / "problem.json"
    write_s_problem(path, problem)
    readback = read_s_problem(path)
    assert readback["schema"] == S_PROBLEM_SCHEMA
    report = verify_s_problem_document(readback, problem=problem)
    assert report["problem_digest"] == problem.problem_digest
    assert readback["s_contract"]["unknown_not_a_hard_constraint"] is True


@_test
def test_counts_and_unknown_ratio_and_named_gap():
    from tilefoundry_ascend.schedule.ledger import build_s_problem_document

    problem = _built()
    document = build_s_problem_document(problem)
    deps = document["dependencies"]
    counts = deps["counts"]
    assert set(counts) == {"RAW_PROVEN", "RAW_UNKNOWN", "WAR_PROVEN", "WAR_UNKNOWN", "WAW_PROVEN", "WAW_UNKNOWN", "REUSE_PROVEN", "REUSE_UNKNOWN"}, counts
    assert sum(counts.values()) == deps["total"]
    # The carrier has UNKNOWN windows -> high ratio -> the NAMED gap must be present.
    assert deps["unknown_ratio"] >= 0.5, deps
    assert "WINDOW_COVERAGE_LOW" in document["coverage_gaps"], document["coverage_gaps"]


@_test
def test_unknown_edges_are_not_hard_constraints():
    from tilefoundry_ascend.schedule.ledger import (
        dependency_kind_status_counts, s_hard_constraint_dependencies, s_unknown_dependencies,
    )

    # One PROVEN RAW (overlapping known windows) and one UNKNOWN (missing offset).
    deps = derive_probe()
    problem = _fake_problem(deps)
    hard = s_hard_constraint_dependencies(problem)
    unknown = s_unknown_dependencies(problem)
    assert hard and all(edge["status"] == "BYTES" for edge in hard)
    assert unknown and all(edge["status"] != "BYTES" for edge in unknown)
    counts = dependency_kind_status_counts(deps)
    assert counts["RAW_PROVEN"] >= 1 and counts["RAW_UNKNOWN"] >= 1, counts


def derive_probe():
    from tilefoundry_ascend.schedule.ledger import derive_dependencies

    proven = derive_dependencies([
        _access("w", "buf", "write", 0, 64),
        _access("r", "buf", "read", 32, 64),
    ])
    unknown = derive_dependencies([
        _access("w2", "buf2", "write", None, 64),
        _access("r2", "buf2", "read", 0, 64),
    ])
    return proven + unknown


@_test
def test_proven_only_problem_has_no_gap():
    from tilefoundry_ascend.schedule.ledger import build_s_problem_document

    deps = derive_probe()
    proven = [edge for edge in deps if edge["status"] == "BYTES"]
    document = build_s_problem_document(_fake_problem(proven))
    assert document["coverage_gaps"] == [], document["coverage_gaps"]
    assert document["dependencies"]["counts"]["RAW_PROVEN"] >= 1


@_test
def test_bad_mutated_field_changes_digest():
    from tilefoundry_ascend.schedule.ledger import build_s_problem_document, verify_s_problem_document

    problem = _built()
    document = copy.deepcopy(build_s_problem_document(problem))
    document["problem"]["policy"] = {}  # drop the whole policy
    assert _refusal(lambda: verify_s_problem_document(document, problem=problem)) == "PROBLEM_DIGEST_MISMATCH"


@_test
def test_bad_unknown_schema_is_refused():
    from tilefoundry_ascend.schedule.ledger import build_s_problem_document, verify_s_problem_document

    problem = _built()
    document = copy.deepcopy(build_s_problem_document(problem))
    document["schema"] = "s-problem/0"
    assert _refusal(lambda: verify_s_problem_document(document, problem=problem)) == "PROBLEM_SCHEMA_UNSUPPORTED"


@_test
def test_bad_missing_s_contract_is_refused():
    from tilefoundry_ascend.schedule.ledger import build_s_problem_document, verify_s_problem_document

    problem = _built()
    document = copy.deepcopy(build_s_problem_document(problem))
    document["s_contract"] = {}
    assert _refusal(lambda: verify_s_problem_document(document, problem=problem)) == "S_CONTRACT_MISSING"


@_test
def test_complexity_stats_prefilter():
    from tilefoundry_ascend.schedule.ledger import derive_dependencies_with_stats

    problem = _built()
    _deps, stats = derive_dependencies_with_stats(problem.accesses)
    for key in ("accesses", "pairs_total_naive", "pairs_candidate_root_space", "elapsed_seconds"):
        assert key in stats, stats
    assert stats["pairs_candidate_root_space"] <= stats["pairs_total_naive"], stats


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
