"""task-77: part-selection wiring proof.

A directly constructed part1 (HIR Vcvt op mutated part=0 -> part=1 after analysis, then the
PUBLIC finalize path) must be refused by the P pass chain with PART_SELECTION_UNSUPPORTED.
Patching verify_tir_part to a no-op removes that named refusal -- proving the wiring enforces it.

Exit codes: 0 PASS, 1 FAIL, 2 ERROR, 3 NOT_RUN.
"""
from __future__ import annotations

# ruff: noqa: PLC0415, I001

import os
import sys
import unittest.mock

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "targets", "ascend")]
FIX = os.path.join(ROOT, "targets", "ascend", "tests", "fixtures", "rmsnorm", "rmsnorm_row.py")


def _prepare():
    from tilefoundry.cli.source import checked_program
    from tilefoundry.analysis.api import analyze

    prog = checked_program(FIX + ":RMSNormRow", None, command="task77")
    analyzed = analyze(prog.module, prog.entry, analysis=("memory",))
    return prog, analyzed


def _first_vcvt_op(function):
    from tilefoundry.ir.core import Call

    found = [None]

    def visit(node):
        if found[0] is not None or node is None:
            return
        if isinstance(node, (tuple, list)):
            for child in node:
                visit(child)
            return
        if isinstance(node, Call):
            if type(getattr(node.target, "op", None)).__name__ == "Vcvt":
                found[0] = node.target.op
                return
            for arg in node.args:
                visit(arg)
            return
        for attr in ("body", "then_body", "else_body", "yield_values", "args", "elements"):
            visit(getattr(node, attr, None))

    visit(function.body)
    return found[0]


def _outcome(analyzed, authored):
    from tilefoundry.schedule.finalize import finalize
    from tilefoundry.ir.hir.pipeline import PipelineError

    try:
        finalize(analyzed, authored=authored)
    except PipelineError as error:
        return ("Pipeline", error.code)
    except Exception as error:  # noqa: BLE001
        return ("Other", type(error).__name__)
    return ("OK", None)


def test_part1_refused_by_public_finalize():
    prog, analyzed = _prepare()
    op = _first_vcvt_op(prog.module.entry_function())
    assert op is not None and getattr(op, "part", None) == 0
    setattr(op, "part", 1)
    assert _outcome(analyzed, prog.authored) == ("Pipeline", "PART_SELECTION_UNSUPPORTED")


def test_baseline_fixture_still_finalizes():
    prog, analyzed = _prepare()
    assert _outcome(analyzed, prog.authored) == ("OK", None)


def test_removing_the_wiring_drops_the_named_refusal():
    prog, analyzed = _prepare()
    setattr(_first_vcvt_op(prog.module.entry_function()), "part", 1)
    with unittest.mock.patch("tilefoundry_ascend.passes.pipeline.verify_tir_part", lambda part: None):
        outcome = _outcome(analyzed, prog.authored)
    assert outcome != ("Pipeline", "PART_SELECTION_UNSUPPORTED"), outcome


def main() -> int:
    tests = (test_part1_refused_by_public_finalize, test_baseline_fixture_still_finalizes, test_removing_the_wiring_drops_the_named_refusal)
    failed = 0
    for case in tests:
        try:
            case()
        except Exception as error:  # noqa: BLE001
            failed += 1
            print("FAIL %s: %s: %s" % (case.__name__, type(error).__name__, error))
        else:
            print("PASS %s" % case.__name__)
    print("%d/%d passed" % (len(tests) - failed, len(tests)))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
