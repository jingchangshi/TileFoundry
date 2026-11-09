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

    def forged_ordinals():
        """The decisive premise test: corrupt EVERY ordinal the records name.

        Before the anchor existed this reproduced the same 128 edges, which is exactly the defect
        the audit named. The site now knows which ordinals it emitted, so this must refuse.
        """
        bad = copy.deepcopy(records)
        for record in bad:
            for field in record["fields"]:
                field["declaration_ordinal"] = 999999
                if field.get("outer_declaration_ordinal") is not None:
                    field["outer_declaration_ordinal"] = 999998
        validate_escape_edges(ledger_with(rec=bad))

    refusal("neg_forged_ordinals_refused", forged_ordinals)

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

    # ------------------------------------------------------------------------------------
    # FIELD-LEVEL payload negatives: one per EscapeEdge field, each a plausible-but-wrong
    # value (never an obviously malformed one), each required to be refused BY NAME.
    #
    # Every case mutates ONLY the edge payload after extract_ledger() has produced a valid
    # ledger, so the validator cannot be rescued by rebuilding from the typed records: the
    # expectation it compares against is rebuilt independently from escape_records +
    # declaration_sites inside validate_escape_edges().
    # ------------------------------------------------------------------------------------
    def field_negatives():
        """Return {field: (accepted, code)} by mutating one payload field at a time."""
        results = {}

        def probe(name, mutate):
            led = ledger_with()
            try:
                mutate(led)
            except Exception as error:  # noqa: BLE001 - an unbuildable mutant is a test defect
                results[name] = ("MUTANT-UNBUILDABLE", type(error).__name__)
                return
            try:
                validate_escape_edges(led)
            except LedgerError as error:
                results[name] = (False, str(error).split(":", 1)[0])
            except Exception as error:  # noqa: BLE001
                results[name] = ("NON-NAMED", type(error).__name__)
            else:
                results[name] = (True, None)

        def pick(led, predicate):
            for index, edge in enumerate(led.escape_edges):
                if predicate(edge):
                    return index, edge
            raise LookupError("no edge matches")

        def set_field(index, led, **changes):
            led.escape_edges[index] = dataclasses.replace(led.escape_edges[index], **changes)

        def another(pool, current):
            return next(value for value in pool if value != current)

        # --- roots: swap in another LEGAL SMEM root (a value that exists, but not this edge's)
        probe("inner_root", lambda led: set_field(0, led, inner_root=next(
            edge.inner_root for edge in led.escape_edges
            if edge.inner_root != led.escape_edges[0].inner_root)))
        probe("outer_root", lambda led: set_field(0, led, outer_root=next(
            edge.outer_root for edge in led.escape_edges
            if edge.outer_root != led.escape_edges[0].outer_root)))
        probe("inner_outer_root_swapped", lambda led: set_field(
            0, led, inner_root=led.escape_edges[0].outer_root,
            outer_root=led.escape_edges[0].inner_root))

        # --- action ids: another action that REALLY exists in the ledger (wrong binding)
        probe("inner_action_id", lambda led: set_field(0, led, inner_action_id=another(
            [edge.inner_action_id for edge in led.escape_edges], led.escape_edges[0].inner_action_id)))
        probe("outer_action_id", lambda led: set_field(0, led, outer_action_id=another(
            [edge.outer_action_id for edge in led.escape_edges], led.escape_edges[0].outer_action_id)))
        probe("inner_outer_action_swapped", lambda led: set_field(
            0, led, inner_action_id=led.escape_edges[0].outer_action_id,
            outer_action_id=led.escape_edges[0].inner_action_id))

        # --- recorded types: bf16 -> f32 keeps the byte footprint identical, so only a real
        # dtype check can refuse it (this is the decisive one for "type is just a label").
        # The mutation must actually CHANGE the text: the carrier has both bf16 and f32 edges,
        # and rewriting an already-f32 edge would be a no-op that proves nothing.
        def retype(led, attribute):
            index, edge = pick(led, lambda e: "bf16" in str(getattr(e, attribute)))
            set_field(index, led, **{attribute: str(getattr(edge, attribute)).replace("bf16", "f32")})

        probe("inner_type", lambda led: retype(led, "inner_type"))
        probe("outer_type", lambda led: retype(led, "outer_type"))

        # --- naming: which outer scope/binding the value binds to
        probe("outer_binding", lambda led: set_field(0, led, outer_binding="FORGED_BINDING"))
        probe("outer_scope", lambda led: set_field(0, led, outer_scope=next(
            edge.outer_scope for edge in led.escape_edges
            if edge.outer_scope != led.escape_edges[0].outer_scope)))

        # --- loop structure an edge claims to live in
        probe("iteration_path", lambda led: set_field(0, led, iteration_path="wrong_loop"))
        probe("iteration_domains_emptied", lambda led: set_field(
            pick(led, lambda edge: edge.iteration_domains)[0], led, iteration_domains=()))
        probe("iteration_domains_wrong", lambda led: set_field(
            pick(led, lambda edge: edge.iteration_domains)[0], led,
            iteration_domains=(("zzz", "1"),)))
        probe("iteration_domains_extra", lambda led: set_field(
            pick(led, lambda edge: edge.iteration_domains)[0], led,
            iteration_domains=tuple(pick(led, lambda edge: edge.iteration_domains)[1].iteration_domains)
            + (("extra", "2"),)))

        # --- guard: the real producers here are UNGUARDED, so ADDING a guard is the forgery
        probe("guard_added", lambda led: set_field(
            0, led, guard=(("then", ("Constant", False)),)))

        # --- identity of the binding itself
        probe("ssa_group", lambda led: set_field(0, led, ssa_group=another(
            [edge.ssa_group for edge in led.escape_edges], led.escape_edges[0].ssa_group)))
        probe("domain", lambda led: set_field(
            0, led, domain=("main" if led.escape_edges[0].domain == "weight" else "weight")))
        probe("edge_kind", lambda led: set_field(0, led, edge_kind="ESCAPE_ORDINARY"))
        probe("schema_version", lambda led: set_field(0, led, schema_version="escape-edge/1"))
        probe("field_path", lambda led: set_field(0, led, field_path="FORGED.field"))
        probe("scope_path", lambda led: set_field(0, led, scope_path="FORGED_SCOPE"))
        return results

    field_results = field_negatives()

    #: Fields that must be refused with the generic payload-mismatch code (their value differs
    #: from the independently rebuilt expectation) rather than a structural code.
    _MISMATCH_FIELDS = (
        "inner_root", "outer_root", "inner_outer_root_swapped",
        "inner_action_id", "outer_action_id", "inner_outer_action_swapped",
        "inner_type", "outer_type", "outer_binding", "outer_scope",
        "iteration_path", "iteration_domains_emptied", "iteration_domains_wrong",
        "iteration_domains_extra", "guard_added", "ssa_group", "domain", "schema_version",
    )
    _CODE_BY_FIELD = {
        "edge_kind": "ESCAPE_EDGE_KIND",
        "field_path": "ESCAPE_EDGE_MISSING",
        "scope_path": "ESCAPE_EDGE_MISSING",
    }
    for field in _MISMATCH_FIELDS:
        accepted, code = field_results[field]
        checks["neg_field_" + field] = (accepted is False) and code == "ESCAPE_EDGE_MISMATCH"
    for field, expected_code in _CODE_BY_FIELD.items():
        accepted, code = field_results[field]
        checks["neg_field_" + field] = (accepted is False) and code == expected_code

    # COMPLETENESS: every declared EscapeEdge field must appear in at least one negative, and
    # every negative must be NAMED. A field with no negative is an uncovered field, not a pass.
    covered_fields = set()
    for field in _MISMATCH_FIELDS:
        covered_fields.add(field.split("_swapped")[0].split("_emptied")[0]
                           .split("_wrong")[0].split("_extra")[0].split("_added")[0])
    covered_fields |= set(_CODE_BY_FIELD)
    declared_fields = {f.name for f in dataclasses.fields(baseline.escape_edges[0])}
    unchecked = sorted(declared_fields - covered_fields)
    checks["every_escape_edge_field_has_a_negative"] = not unchecked
    if unchecked:
        print("UNCHECKED EscapeEdge fields: " + repr(unchecked))
    checks["every_field_negative_is_named"] = all(
        code is not None and code != "NON-NAMED" and not str(code).startswith("MUTANT")
        for accepted, code in field_results.values())
    checks["no_field_negative_was_accepted"] = not any(
        accepted is True for accepted, _code in field_results.values())


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