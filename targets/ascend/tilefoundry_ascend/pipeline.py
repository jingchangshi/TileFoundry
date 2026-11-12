"""End-to-end Ascend compile driver: HIR -> Problem -> Solve -> Plan -> TIR -> Verify."""
from __future__ import annotations

import hashlib, json

from .schedule.program import build_program, case_from_dict, CaseConfig
from .schedule.problem import extract_problem
from .schedule.solver import solve
from .schedule.sync import build_plan
from .lowering.materialize import materialize_tir
from .lowering.verify import verify, parse_events, mutate_drop_wait, mutate_early_free, mutate_wrong_slot, verify_text_events
from .lowering.layout_check import verify_layout
from .memory import SPACE_FACTS


def capacity_check(program):
    ub_nominal = SPACE_FACTS["ascend.ub"].nominal_bytes or (256 * 1024)
    footprint = program.buffer_bytes_total()
    ok = footprint <= ub_nominal
    return {"ub_nominal_bytes": ub_nominal, "ub_footprint_bytes": footprint,
            "ok": ok, "reason": "" if ok else f"UB footprint {footprint} > nominal {ub_nominal}"}


def _steps_for(cfg: CaseConfig):
    return max(1, cfg.nbase + (1 if cfg.rem else 0))


def compile_case(case_dict: dict, steps: int | None = None):
    cfg = case_from_dict(case_dict)
    program = build_program(cfg)
    cap = capacity_check(program)
    problem = extract_problem(program)
    result = solve(problem)
    plan = build_plan(problem, result)
    st = _steps_for(cfg) if steps is None else steps
    out = {"case_id": cfg.case_id, "case_config": cfg.as_dict(), "program": program.as_dict(),
           "capacity": cap, "problem": problem.as_dict(), "solve": result.as_dict(),
           "plan": plan, "verify_steps": st}
    if result.status != "solved":
        out.update(tir_text="", tir_header={}, verify={"ok": False, "errors": ["not solved"], "events": 0},
                   negative_tests={}, all_negative_rejected=False, plan_digest=None)
        return out
    text, events, hdr = materialize_tir(problem, plan, st)
    ver = verify(problem, text, plan)
    lay = verify_layout(problem, plan)
    parsed = parse_events(text)
    # The negative probes are named after the ROLE they break, and the role is resolved to the
    # derived channel the plan actually produced. A literal channel name here would mutate a
    # channel that no longer exists, and a mutation that changes nothing is a negative test that
    # cannot fail -- the worst possible kind.
    from .lowering.materialize import channel_names
    roles = channel_names(plan)
    neg = {}
    for name, mut in (
            ("drop_y_ready_wait",
             lambda e: mutate_drop_wait(e, roles["y_ready"])),
            ("drop_x_ready_wait",
             lambda e: mutate_drop_wait(e, roles["x_ready"])),
            ("early_x_free",
             lambda e: mutate_early_free(e, free_channel=roles["x_free"],
                                         ready_channel=roles["x_ready"])),
            ("wrong_x_slot",
             lambda e: mutate_wrong_slot(e, versions=problem.policy["x_versions"]))):
        # A mutation that cannot change the program must not be reported as "rejected": that
        # would make all_negative_rejected true on the strength of a no-op. It is recorded as
        # NOT_APPLICABLE with the reason, and the caller sees the gap.
        try:
            mutated = mut(parsed)
        except ValueError as error:
            neg[name] = {"rejected": False, "not_applicable": str(error), "errors": []}
            continue
        r = verify_text_events(problem, mutated, plan)
        neg[name] = {"rejected": not r["ok"], "errors": r["errors"][:4]}
    out.update(tir_text=text, tir_header=hdr, verify=ver, layout=lay, negative_tests=neg,
               all_negative_rejected=all(v["rejected"] for v in neg.values()),
               plan_digest=hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()[:16])
    return out


def trip_count_tests(base_case: dict, trips=(0, 1, 2, 3, 4, 5, 9)):
    out = {}
    for n in trips:
        r = compile_case(base_case, steps=n)
        out[str(n)] = {"verify_ok": r["verify"]["ok"], "errors": r["verify"]["errors"][:3],
                       "negatives_rejected": r["all_negative_rejected"]}
    return out


__all__ = ["compile_case", "capacity_check", "trip_count_tests"]
