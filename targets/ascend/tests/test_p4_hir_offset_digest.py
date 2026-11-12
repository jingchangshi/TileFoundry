"""G-P4-6 real HIR mutation: the DUS offset 0 -> 64+idx*L must change the digests.

Two REAL HIR variants are written to a temp dir from the same fixture, each compiled through the
public path (checked_program -> analyze(memory) -> finalize -> extract_ledger -> build_problem).
Both compile; their offset STRUCTURE (op kinds + operands + constants + vars) differs, so
source_digest and problem_digest MUST differ. The same HIR compiled twice must be byte-identical in
digest (determinism). The normalized offset stays UNKNOWN in both.

Exit codes: 0 PASS, 1 FAIL, 2 ERROR, 3 NOT_RUN.
"""
from __future__ import annotations
import os, sys, tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "targets", "ascend")]
FIX = os.path.join(ROOT, "targets", "ascend", "tests", "fixtures", "rmsnorm", "aiv56_ragged.py")
OLD = "y = tf.insert_slice(y, wv, (idx * L,))"
NEW = "y = tf.insert_slice(y, wv, (64 + idx * L,))"


def compile_text(text, directory, name):
    from tilefoundry.cli.source import checked_program
    from tilefoundry.analysis.api import analyze
    from tilefoundry.schedule.finalize import finalize
    from tilefoundry_ascend.schedule.ledger import build_problem, extract_ledger
    path = os.path.join(directory, name + ".py")
    open(path, "w", encoding="utf-8").write(text)
    prog = checked_program(path + ":Own56R", None, command="hir-mutation")
    analyzed = analyze(prog.module, prog.entry, analysis=("memory",))
    module = finalize(analyzed, authored=prog.authored)
    ledger = extract_ledger(module.entry_function())
    return ledger, build_problem(ledger)


def main():
    checks = {}
    base = open(FIX, encoding="utf-8").read()
    assert OLD in base, "fixture no longer contains the expected DUS offset line"
    directory = tempfile.mkdtemp()
    zero_ledger, zero = compile_text(base, directory, "zero")
    plus_ledger, plus = compile_text(base.replace(OLD, NEW), directory, "plus64")
    again_ledger, again = compile_text(base, directory, "zero_again")

    zero_out = [e for e in zero_ledger.entries if e.root == "out"]
    plus_out = [e for e in plus_ledger.entries if e.root == "out"]
    checks["both_hir_compile"] = bool(zero_out) and bool(plus_out)
    checks["offset_structure_differs"] = zero_out[0].offset_structure != plus_out[0].offset_structure
    checks["source_digest_differs"] = zero.source_digest != plus.source_digest
    checks["problem_digest_differs"] = zero.problem_digest != plus.problem_digest
    checks["normalized_offset_unknown_both"] = zero_out[0].byte_offset is None and plus_out[0].byte_offset is None
    checks["deterministic_same_hir"] = zero.source_digest == again.source_digest and zero.problem_digest == again.problem_digest
    checks["guard_provenance_recorded"] = all(isinstance(e.guard, tuple) for e in zero_ledger.entries)

    for key, value in checks.items():
        print(("PASS     " if value else "FAIL     ") + key)
    print(f"zero source_digest={zero.source_digest} problem_digest={zero.problem_digest}")
    print(f"plus64 source_digest={plus.source_digest} problem_digest={plus.problem_digest}")
    if not all(checks.values()):
        print(f"{sum(not v for v in checks.values())}/{len(checks)} FAILED"); return 1
    print(f"{len(checks)}/{len(checks)} checks passed"); return 0


if __name__ == "__main__":
    try: raise SystemExit(main())
    except SystemExit: raise
    except Exception as error:
        print(f"ERROR {type(error).__name__}: {error}"); raise SystemExit(2)
