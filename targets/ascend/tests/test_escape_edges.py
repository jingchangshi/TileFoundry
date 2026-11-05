"""04-05 follow-up: escape/value-binding edges from TYPED records, not emission order (CPU).

Every endpoint must come from the Core escape site records through their declaration anchors:
a field names its producer by (record scope, ssa_group), the anchor turns that group into a
structural TIR path, and that path names exactly one SMEM root. The checks below first establish
the UNMUTATED baseline passes, then require each mutation to be refused BY NAME (a generic
exception is not a pass), including the decisive one: corrupting an anchor path or a recorded
type must change or refuse the result instead of silently reproducing the same 128 edges.

Exit codes: 0 PASS, 1 FAIL, 2 ERROR, 3 NOT_RUN.
"""
from __future__ import annotations
import copy, dataclasses, json, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "targets", "ascend")]
FIX = os.path.join(ROOT, "targets", "ascend", "tests", "fixtures", "rmsnorm", "rmsnorm_main.py")


def build():
    from tilefoundry.cli.source import checked_program
    from tilefoundry.analysis.api import analyze
    from tilefoundry.schedule.finalize import finalize
    prog = checked_program(FIX + ":RMSNormD", None, command="escape-edges-test")
    analyzed = analyze(prog.module, prog.entry, analysis=("memory",))
    return finalize(analyzed, authored=prog.authored)


def main():
    from tilefoundry_ascend.schedule.ledger import (
        LedgerError, build_escape_edges, extract_ledger, validate_escape_edges,
    )
    module = build()
    records = module.metadata["escape_records"]
    sites = module.metadata["escape_declaration_sites"]
    function = module.entry_function()

    sentinel = object()

    def ledger_with(rec=sentinel, anc=sentinel):
        """Build a ledger from mutated records/anchors; the sentinel means 'use the real ones'."""
        return extract_ledger(
            function,
            escape_records=copy.deepcopy(records) if rec is sentinel else rec,
            declaration_sites=copy.deepcopy(sites) if anc is sentinel else anc,
        )

    checks = {}
    baseline = ledger_with()
    report = validate_escape_edges(baseline)
    checks["unmutated_baseline_accepted"] = report["complete"] is True
    checks["baseline_edge_count_128"] = len(baseline.escape_edges) == 128
    checks["weight_and_main_domains_covered"] = report["domains"] == ["main", "weight"]
    checks["dropped_types_all_bf16_smem"] = report["dropped_types"] == {"bf16 SMEM": 128}
    checks["no_object_identity_in_edges"] = all(
        "0x" not in edge.inner_action_id and "0x" not in edge.outer_action_id
        for edge in baseline.escape_edges
    )

    # DETERMINISM: two independent lowerings produce byte-identical anchors and edges.
    other = build()
    checks["anchors_deterministic"] = json.dumps(
        other.metadata["escape_declaration_sites"], sort_keys=True, default=str
    ) == json.dumps(sites, sort_keys=True, default=str)
    checks["anchor_payload_has_no_memory_address"] = "0x" not in json.dumps(sites, default=str)

    # The typed records must be the PREMISE: corrupting an anchor a real field depends on has to
    # change or refuse the result, never reproduce the same edges.
    forged = copy.deepcopy(sites)
    group = baseline.escape_edges[0].ssa_group
    forged[group] = [dict(forged[group][0], path="s9/FAKE")]
    try:
        build_escape_edges(baseline, copy.deepcopy(records), forged)
        checks["forged_anchor_changes_or_refuses"] = False
    except LedgerError as error:
        checks["forged_anchor_changes_or_refuses"] = "ESCAPE_" in str(error)

    def refusal(label, call):
        """Named refusal only: the target error must be a LedgerError carrying a code we expect."""
        try:
            call()
        except LedgerError as error:
            checks[label] = "ESCAPE_" in str(error)
            return
        checks[label] = False

    refusal("neg_delete_all_edges", lambda: validate_escape_edges(
        dataclasses.replace(ledger_with(), escape_edges=[])
    ))

    def only_main():
        led = ledger_with()
        led.escape_edges = [edge for edge in led.escape_edges if edge.domain == "main"]
        validate_escape_edges(led)

    refusal("neg_only_main_domain_refused", only_main)

    def drop_one():
        led = ledger_with()
        led.escape_edges = led.escape_edges[1:]
        validate_escape_edges(led)

    refusal("neg_drop_single_edge_refused", drop_one)

    def fake_ids():
        led = ledger_with()
        led.escape_edges[0] = dataclasses.replace(
            led.escape_edges[0], inner_action_id="NONEXISTENT", outer_action_id="NONEXISTENT"
        )
        validate_escape_edges(led)

    refusal("neg_fake_action_ids_refused", fake_ids)

    def wrong_guard():
        led = ledger_with()
        led.escape_edges[0] = dataclasses.replace(
            led.escape_edges[0], guard=(("then", ("Constant", False)),)
        )
        validate_escape_edges(led)

    refusal("neg_wrong_guard_refused", wrong_guard)

    def wrong_iteration():
        led = ledger_with()
        led.escape_edges[0] = dataclasses.replace(led.escape_edges[0], iteration_path="wrong_loop")
        validate_escape_edges(led)

    refusal("neg_wrong_iteration_refused", wrong_iteration)

    def swap_fields():
        led = ledger_with()
        index = [i for i, edge in enumerate(led.escape_edges) if edge.domain == "main"]
        first, second = led.escape_edges[index[0]], led.escape_edges[index[1]]
        led.escape_edges[index[0]] = dataclasses.replace(first, field_path=second.field_path)
        led.escape_edges[index[1]] = dataclasses.replace(second, field_path=first.field_path)
        validate_escape_edges(led)

    refusal("neg_field_permutation_refused", swap_fields)

    def wrong_domain():
        led = ledger_with()
        led.escape_edges[0] = dataclasses.replace(led.escape_edges[0], domain="main")
        validate_escape_edges(led)

    refusal("neg_wrong_domain_refused", wrong_domain)

    refusal("neg_empty_anchors_refused", lambda: validate_escape_edges(
        ledger_with(anc={})
    ))
    refusal("neg_missing_anchors_refused", lambda: validate_escape_edges(
        ledger_with(anc=None)
    ))

    def forged_group():
        bad = copy.deepcopy(records)
        highest = max(field["ssa_group"] for record in bad for field in record["fields"])
        for record in bad:
            for field in record["fields"]:
                field["ssa_group"] = highest
        validate_escape_edges(ledger_with(rec=bad))

    refusal("neg_forged_ssa_group_refused", forged_group)

    def forged_type():
        bad = copy.deepcopy(records)
        for record in bad:
            for field in record["fields"]:
                field["inner_type"] = field["inner_type"].replace("f32", "bf16")
        validate_escape_edges(ledger_with(rec=bad))

    refusal("neg_forged_inner_type_refused", forged_type)

    for key, value in checks.items():
        print(("ok   " if value else "FAIL ") + key)
    print(
        f"edges={len(baseline.escape_edges)} scopes={report['scopes']} "
        f"domains={report['domains']} dropped={report['dropped_types']}"
    )
    if not all(checks.values()):
        print(f"{sum(not value for value in checks.values())}/{len(checks)} FAILED")
        return 1
    print(f"{len(checks)}/{len(checks)} checks passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as error:
        print(f"ERROR {type(error).__name__}: {error}")
        raise SystemExit(2)