"""L4: dependencies from typed byte-WINDOW intersection + loop DOMAIN, not names/order.

The decisive pair: two accesses whose task names look alike but whose byte windows are
DISJOINT must have NO dependency; two accesses whose names differ but which alias the SAME
root and whose windows INTERSECT must HAVE one. A name/statement-order heuristic gives the
opposite answer on exactly these two cases, so they falsify it. UNKNOWN windows are recorded
as status UNKNOWN, never dropped and never defaulted to 0.

Exit codes: 0 PASS, 1 FAIL, 2 ERROR, 3 NOT_RUN.
"""
from __future__ import annotations

# ruff: noqa: PLC0415, I001  -- local imports keep each check's dependency surface explicit

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "targets", "ascend")]

TESTS = {}


def _test(fn):
    TESTS[fn.__name__] = fn
    return fn


def _access(task, root, mode, off, footprint, domains=()):
    from tilefoundry_ascend.schedule.ledger import ProblemAccess

    return ProblemAccess(
        task=task,
        instruction=task,
        root=root,
        root_kind="real",
        space="GMEM",
        mode=mode,
        byte_footprint=footprint,
        byte_offset=off,
        source_offset="[]",
        attrs=(),
        offset_structure=(("literal", off),),
        guard=(),
        normalized_offset=off,
        offset_status=("BYTES" if off is not None else "OFFSET_UNKNOWN"),
        logical_iteration=("t" if domains else "outside"),
        iteration_path=("/".join(name for name, _ in domains) or "outside"),
        iteration_extent="1",
        iteration_domains=tuple(domains),
    )


def _kinds(deps):
    return sorted(edge["kind"] for edge in deps)


@_test
def test_similar_names_disjoint_windows_have_no_dependency():
    from tilefoundry_ascend.schedule.ledger import derive_dependencies

    # Same root, look-alike task names, but the byte windows do not intersect.
    a = _access("xa", "buf", "write", 0, 64)
    b = _access("xb", "buf", "read", 64, 64)
    assert derive_dependencies([a, b]) == [], "disjoint byte windows must not create a dependency"


@_test
def test_different_names_same_root_overlapping_windows_have_dependency():
    from tilefoundry_ascend.schedule.ledger import derive_dependencies

    writer = _access("producer_left", "buf", "write", 0, 64)
    reader = _access("consumer_right", "buf", "read", 32, 64)
    deps = derive_dependencies([writer, reader])
    assert len(deps) == 1, deps
    edge = deps[0]
    assert edge["kind"] == "RAW" and edge["status"] == "BYTES", edge
    assert edge["producer"] == "producer_left" and edge["consumer"] == "consumer_right"
    assert edge["producer_window"] == (0, 64) and edge["consumer_window"] == (32, 96)


@_test
def test_kind_comes_from_mode_not_statement_order():
    from tilefoundry_ascend.schedule.ledger import derive_dependencies

    writer = _access("w", "buf", "write", 0, 64)
    reader = _access("r", "buf", "read", 0, 64)
    # Read listed FIRST: an order-based heuristic would report WAR with producer=read.
    forward = derive_dependencies([reader, writer])
    backward = derive_dependencies([writer, reader])
    assert _kinds(forward) == ["RAW"] and _kinds(backward) == ["RAW"], (forward, backward)
    assert {e["producer"] for e in forward + backward} == {"w"}, (forward, backward)


@_test
def test_dependency_kinds_waw_and_reuse():
    from tilefoundry_ascend.schedule.ledger import derive_dependencies

    assert _kinds(derive_dependencies([_access("w1", "buf", "write", 0, 64), _access("w2", "buf", "write", 32, 64)])) == ["WAW"]
    assert _kinds(derive_dependencies([_access("r1", "buf", "read", 0, 64), _access("r2", "buf", "read", 32, 64)])) == ["REUSE"]


@_test
def test_unknown_window_is_recorded_not_dropped():
    from tilefoundry_ascend.schedule.ledger import derive_dependencies

    unknown = _access("w", "buf", "write", None, 64)
    reader = _access("r", "buf", "read", 0, 64)
    deps = derive_dependencies([unknown, reader])
    assert len(deps) == 1, "an UNKNOWN window must not be treated as independent"
    assert deps[0]["status"] == "UNKNOWN", deps[0]
    assert deps[0]["kind"] == "RAW"


@_test
def test_distance_uses_iteration_domain():
    from tilefoundry_ascend.schedule.ledger import derive_dependencies

    same = derive_dependencies([
        _access("w", "buf", "write", 0, 64, domains=(("i", "4"),)),
        _access("r", "buf", "read", 0, 64, domains=(("i", "4"),)),
    ])
    other = derive_dependencies([
        _access("w", "buf", "write", 0, 64, domains=(("i", "4"),)),
        _access("r", "buf", "read", 0, 64, domains=(("j", "4"),)),
    ])
    assert same[0]["distance"] == 0 and other[0]["distance"] == 1


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
