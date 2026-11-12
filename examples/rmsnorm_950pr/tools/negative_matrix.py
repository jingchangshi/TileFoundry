#!/usr/bin/env python3
"""M6 negative matrix -- TRACE MODEL ONLY, NOT formal P5 TF TIR.

SCOPE LIMIT (read before citing this): the operator sequence exercised here comes from
the custom text emitted by lowering/materialize.py ("# Resolved TF TIR", def/for plus
"mte2 ..." and "simdvf ..." comment lines), NOT from formal TF nodes produced by the
public finalize path. The credit this matrix earns is therefore FINITE TRACE-MODEL
credit. It is NOT evidence of P5 formal TIR, and must not be cited as such. Emitting real
TF nodes through the normal finalize path is a separate, unscoped fix.

Second limit: the layout rule inspects a NEW placement built from buffer sizes/alignment;
it IGNORES the plan, so it does NOT check the addresses that will actually be placed and
cannot catch hazards specific to the real placement.

One pinned message per sync/layout rule, per-rule exit code.

Each RULE owns ONE violating artifact and ONE pinned message substring. A rule PASSES
(exit 0 for that rule) only if building its violating artifact makes the verifier emit an
error containing that substring. Every rule is reported with its own exit code, so a rule
that silently stops firing is visible rather than averaged into an overall pass.

CPU only. Imports the ascend compiler (needs PYTHONPATH both roots). Exit 0 iff all rules
pass, else 1.
"""
from __future__ import annotations
import copy, re, sys
from types import SimpleNamespace

from tilefoundry_ascend.schedule.program import build_program, case_from_dict
from tilefoundry_ascend.schedule.problem import extract_problem
from tilefoundry_ascend.schedule.solver import solve
from tilefoundry_ascend.schedule.sync import build_plan
from tilefoundry_ascend.lowering.materialize import materialize_tir
from tilefoundry_ascend.lowering.verify import (verify, parse_events, mutate_drop_wait,
                                                mutate_early_free, mutate_wrong_slot,
                                                verify_text_events, verify_pipes)
from tilefoundry_ascend.lowering.layout_check import verify_layout

CASE = {"case_id": "m6matrix", "shape": [64, 4096], "loop_mode": "dyn",
        "requested_num_stages": 2, "x_versions": 2, "y_versions": 1}


def _errs(result):
    return result.get("errors", [])


def _drop_drain(text):
    return "\n".join(l for l in text.splitlines() if "drain()" not in l)


def _flag_mismatch(text):
    out, done = [], False
    for l in text.splitlines():
        if not done:
            m = re.search(r"^(\s*set_flag\s+\w+\[\d+\]\s+gen=\d+\s+flag=)(\d+)", l)
            if m:
                l = m.group(1) + "999" + l[m.end():]
                done = True
        out.append(l)
    return "\n".join(out)


def _bad_pipes(plan):
    p = copy.deepcopy(plan)
    for a in p.get("sync_actions", []):
        if a.get("channel") == "x_ready":
            a["producer_pipe"] = "vector"; a["consumer_pipe"] = "vector"
            break
    return p


def build_rules(problem, plan, text, parsed):
    # ProblemBuffer is frozen; build a lightweight view whose first UB buffer is
    # oversized so the physical footprint exceeds capacity (a post-layout hazard).
    bufs, done = [], False
    for b in problem.buffers:
        by = b.bytes_
        if not done and b.space == "ascend.ub":
            by = 10 ** 9
            done = True
        bufs.append(SimpleNamespace(root=b.root, space=b.space, bytes_=by,
                                    alignment=b.alignment, versions=b.versions))
    huge = SimpleNamespace(buffers=bufs)
    return [
        ("DROP_X_READY_WAIT", "x_ready",
         lambda: _errs(verify_text_events(problem, mutate_drop_wait(parsed, "x_ready")))),
        ("DROP_Y_READY_WAIT", "y_ready",
         lambda: _errs(verify_text_events(problem, mutate_drop_wait(parsed, "y_ready")))),
        ("EARLY_X_FREE", "early free",
         lambda: _errs(verify_text_events(problem, mutate_early_free(parsed)))),
        ("WRONG_X_SLOT", "mte2 writes x slot",
         lambda: _errs(verify_text_events(problem, mutate_wrong_slot(parsed)))),
        ("EMPTY_TIR", "[M6-EMPTY-TIR]", lambda: _errs(verify(problem, ""))),
        ("UNKNOWN_STATEMENT", "[M6-UNKNOWN-STATEMENT]",
         lambda: _errs(verify(problem, text + "\n    frobnicate zzz = 1"))),
        ("MISSING_DRAIN", "[M6-MISSING-DRAIN]", lambda: _errs(verify(problem, _drop_drain(text)))),
        ("DRAIN_NOT_LAST", "[M6-DRAIN-NOT-LAST]",
         lambda: _errs(verify(problem, text + "\n    mte2 x_load slot=0 t=0"))),
        ("FLAG_MISMATCH", "[M6-FLAG-MISMATCH]",
         lambda: _errs(verify(problem, _flag_mismatch(text)))),
        ("MISSING_X_FREE_WAIT", "[M6-MISSING-X-FREE-WAIT]",
         lambda: _errs(verify_text_events(problem, mutate_drop_wait(parsed, "x_free")))),
        ("OUTSTANDING_GENERATION", "[M6-OUTSTANDING-GENERATION]",
         lambda: _errs(verify(problem, "mte2 x_load slot=0 t=0\n    set_flag x_ready[0] gen=0 flag=0  # t=0\n  drain()"))),
        ("DRAIN_TRAILING_GARBAGE", "[M6-UNKNOWN-STATEMENT]",
         lambda: _errs(verify(problem, text.replace("drain()", "drain(); hidden_gm_write()")))),
        ("UNKNOWN_CHANNEL", "[M6-UNKNOWN-CHANNEL]",
         lambda: _errs(verify(problem, text.replace("drain()", "wait_flag bogus_chan[0] gen=0 flag=0  # t=0\n  drain()")))),
        ("PIPE_MISMATCH", "[M6-PIPE-MISMATCH]",
         lambda: _errs(verify_pipes(_bad_pipes(plan)))),
        ("POST_LAYOUT_HAZARD", "[M6-POST-LAYOUT-HAZARD]",
         lambda: _errs(verify_layout(huge, plan))),
    ]


def main():
    cfg = case_from_dict(CASE)
    program = build_program(cfg)
    problem = extract_problem(program)
    result = solve(problem)
    if result.status != "solved":
        print("cannot solve base case:", result.status, file=sys.stderr)
        return 2
    plan = build_plan(problem, result)
    text, events, hdr = materialize_tir(problem, plan, 4)
    # Baseline: the unmutated artifact must verify clean (no false positives).
    base = verify(problem, text)
    layout = verify_layout(problem, plan)
    parsed = parse_events(text)
    rules = build_rules(problem, plan, text, parsed)
    print("SCOPE: trace model only (custom text, not formal P5 TF TIR); "
          "layout checks a NEW placement, not the addresses actually placed.")
    print("baseline verify ok=%s events=%d ; layout ok=%s" % (base["ok"], base["events"], layout["ok"]))
    print("%-26s | %-3s | %s" % ("rule", "rc", "pinned message"))
    print("-" * 90)
    all_ok = base["ok"] and layout["ok"]
    for name, pinned, build in rules:
        try:
            errs = build()
        except Exception as e:  # noqa: BLE001
            errs = ["<exception %s: %s>" % (type(e).__name__, e)]
        hit = any(pinned in (e if isinstance(e, str) else str(e)) for e in errs)
        rc = 0 if hit else 1
        all_ok = all_ok and hit
        print("%-26s | %-3d | %s" % (name, rc, pinned))
        if not hit:
            print("      ERRORS: %s" % [str(e)[:120] for e in errs[:3]])
    print("-" * 90)
    print("RESULT:", "ALL RULES PASS" if all_ok else "FAILURE")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
