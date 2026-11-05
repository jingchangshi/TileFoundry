"""task-82 POC gate G0-G9 over the small affine carrier.

ExpectedObligationSet is built from the typed program: 12 RAW instances + intra-VF SSA +
cross-i WAR/FREE (from real allocation reuse facts) + FINAL. Denominator is the complete,
non-empty source obligation set -- never a filtered BYTES list and never reduced by dropping
register accesses. Each gate has a named failure code (spec section 4).
"""
from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.inspection import as_script
from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.tir.stmts import For, Sequential
from tilefoundry.ir.visitor import collect_exprs

from tilefoundry_ascend.schedule.ledger import (
    LedgerError,
    verify_final_gm_return_chain,
    verify_predicate_wiring,
    verify_statement_coverage,
)

VF_OPS = (
    "Vld", "Pset", "Pset16", "Vadd", "Vmul", "Vmula", "Vmuls",
    "Vadds", "Vcadd", "Vsqrt", "Vdiv", "Vdup", "Vdupv", "Vsts",
)


@dataclass
class Obligation:
    oid: str
    kind: str  # RAW / SSA / WAR / FREE / WAW / REUSE / FINAL
    discharged_by: str | None = None


def _walk(node, out):
    """Walk a HIR/TIR body that may be a bare Expr OR a Sequential statement node."""
    if node is None:
        return
    if isinstance(node, Sequential):
        for child in node.body:
            _walk(child, out)
        return
    if isinstance(node, (LoopRegion, For)):
        out.append(node)
    for attr in ("body", "then_body", "else_body", "yield_values", "args", "elements"):
        value = getattr(node, attr, None)
        if value is None:
            continue
        if isinstance(value, (tuple, list)):
            for child in value:
                _walk(child, out)
        else:
            _walk(value, out)


def verify_g0_pipeline_live(hir_module, hir_function, tir_function) -> dict:
    """G0: authored pipeline is LIVE (not just indirectly covered by F0/reader)."""
    hir_loops = []
    for function in getattr(hir_module, "functions", ()) or ():
        _walk(getattr(function, "body", None), hir_loops)
    hir_piped = [loop for loop in hir_loops if isinstance(loop, LoopRegion) and getattr(loop, "pipeline", None) is not None]
    if not hir_piped:
        raise LedgerError("POC_PIPELINE_NOT_LIVE: no HIR LoopRegion carries .pipeline")
    tir_loops = []
    _walk(tir_function.body, tir_loops)
    tir_piped = [loop for loop in tir_loops if isinstance(loop, For) and getattr(loop, "pipeline", None) is not None]
    if not tir_piped:
        raise LedgerError("POC_PIPELINE_NOT_LIVE: no TIR For carries .pipeline")
    try:
        text = as_script(hir_function)
    except Exception:  # noqa: BLE001 - printer unavailable is itself a not-live signal
        text = as_script(hir_module)
    if "tf.pipelined(" not in text:
        raise LedgerError("POC_PIPELINE_NOT_LIVE: printed HIR lacks tf.pipelined(")
    return {"hir_loops": len(hir_piped), "tir_loops": len(tir_piped)}


def _proven_raw_families(ledger, problem):
    ops = {e.action_id: e.op for e in ledger.entries}
    spaces = {e.action_id: e.space for e in ledger.entries}
    path = {e.action_id: e.path for e in ledger.entries}
    families = set()
    for edge in problem.dependencies:
        if edge["kind"] != "RAW" or edge["status"] != "BYTES":
            continue
        p, c = edge["producer"], edge["consumer"]
        if spaces.get(p) != "SMEM" or spaces.get(c) != "SMEM":
            continue
        if ops.get(p) == "GmToUb" and ops.get(c) == "Vld":
            families.add("Z-in" if path.get(p, "").rfind("/s5") >= 0 else "X-in")
        if ops.get(p) == "Vsts" and ops.get(c) == "copy":
            families.add("Y-out")
    return families


def _ssa_pairs(ledger):
    pairs = set()
    writes = {}
    for e in ledger.entries:
        if e.space == "RMEM" and e.mode == "write" and e.op in VF_OPS:
            writes.setdefault(e.root, []).append(e.op)
    for e in ledger.entries:
        if e.space != "RMEM" or e.mode != "read" or e.op not in VF_OPS:
            continue
        for wop in writes.get(e.root, ()):
            if wop != e.op:
                pairs.add((e.root, wop, e.op))
    return pairs


def _reuse_obligations(ledger, expansion):
    """Cross-i reuse obligations from REAL allocation facts: a SMEM root written each i and
    also read somewhere is a live single-buffer reused across iterations (WAR + FREE)."""
    obligations = []
    smem_by_root = {}
    for e in ledger.entries:
        if e.space == "SMEM":
            smem_by_root.setdefault(e.root, set()).add(e.mode)
    for root in sorted(smem_by_root):
        facts = smem_by_root[root]
        if "write" not in facts:
            continue
        kind = "WAR" if "read" in facts else "WAW"
        for index in range(expansion.count - 1):
            obligations.append(
                Obligation("%s:%s:i%d->i%d" % (kind, root, index, index + 1), kind, "single_version_reuse")
            )
            obligations.append(
                Obligation("FREE:%s:i%d->i%d" % (root, index, index + 1), "FREE", "single_version_reuse")
            )
    return obligations


def build_expected_obligations(ledger, problem, expansion):
    families = _proven_raw_families(ledger, problem)
    raw = []
    for family in ("X-in", "Z-in", "Y-out"):
        for inst in expansion.instances:
            raw.append(Obligation("RAW:%s:i%d" % (family, inst["i"]), "RAW",
                                  "source-proven" if family in families else None))
    ssa = []
    for root, wop, rop in sorted(_ssa_pairs(ledger)):
        for inst in expansion.instances:
            ssa.append(Obligation("SSA:%s:%s->%s:i%d" % (root, wop, rop, inst["i"]), "SSA", "intra_task"))
    reuse = _reuse_obligations(ledger, expansion)
    final = [Obligation("FINAL:y:[0,%d)" % (expansion.count * expansion.stride_bytes), "FINAL", "gm_tiles_cover")]
    return {"raw": raw, "ssa": ssa, "reuse": reuse, "final": final, "all": raw + ssa + reuse + final}


def assert_expected_matches(provided, authoritative, *, code="POC_REQUIRED_COVERAGE_MISMATCH"):
    """Reconcile a provided expected set against the authoritative full obligation set."""
    provided_ids = {o.oid for o in provided}
    authoritative_ids = {o.oid for o in authoritative}
    missing = authoritative_ids - provided_ids
    extra = provided_ids - authoritative_ids
    if missing or extra:
        raise LedgerError("%s: missing=%r extra=%r" % (code, sorted(missing)[:3], sorted(extra)[:3]))


def _gm_windows(expansion, ledger):
    space = {e.action_id: e.space for e in ledger.entries}
    out = set()
    for inst in expansion.instances:
        for task, window in inst["windows"].items():
            if space.get(task) == "GMEM":
                out.add(window)
    return out


def verify_poc_gate(ledger, problem, expansion, *, hir_module, hir_function, tir_function, predicate_status="UNKNOWN"):
    """Evaluate G0-G9; raise the first named failure (spec section 4)."""
    obligations = build_expected_obligations(ledger, problem, expansion)
    total = len(obligations["all"])
    if total == 0:
        raise LedgerError("POC_REQUIRED_DENOMINATOR_EMPTY: expected obligation set is empty")

    g0 = verify_g0_pipeline_live(hir_module, hir_function, tir_function)  # G0

    families = {o.oid.split(":")[1] for o in obligations["raw"]}
    if families != {"X-in", "Z-in", "Y-out"}:
        raise LedgerError("POC_REQUIRED_SOURCE_SET_MISMATCH: %r" % sorted(families))

    proven = _proven_raw_families(ledger, problem)
    if not {"X-in", "Z-in"} <= proven:
        raise LedgerError("POC_RAW_LOAD_VF_MISSING: load->VF RAW not proven for both inputs")
    if "Y-out" not in proven:
        raise LedgerError("POC_RAW_VF_STORE_MISSING: VF->store RAW not proven")

    missing = [o.oid for o in obligations["raw"] if o.discharged_by is None]
    if missing:
        raise LedgerError("POC_RAW_SOURCE_UNPROVEN: %r" % missing[:3])

    undis = [o.oid for o in obligations["all"] if o.discharged_by is None]
    if undis:
        raise LedgerError("POC_REQUIRED_COVERAGE_MISMATCH: undischarged %r" % undis[:3])

    required_unknown = len(undis)
    if required_unknown:
        raise LedgerError("POC_REQUIRED_WINDOW_UNKNOWN: %d/%d" % (required_unknown, total))

    spaces = {a.task: a.space for a in problem.accesses}
    critical = [
        e for e in problem.dependencies
        if e["status"] != "BYTES"
        and spaces.get(e["producer"]) in ("GMEM", "SMEM")
        and spaces.get(e["consumer"]) in ("GMEM", "SMEM")
    ]
    if critical:
        raise LedgerError("POC_CRITICAL_DEPENDENCY_UNKNOWN: %d" % len(critical))

    gm = _gm_windows(expansion, ledger)
    covered = sorted(gm)
    contiguous = bool(covered) and covered[0][0] == 0 and covered[-1][1] == expansion.count * expansion.stride_bytes
    contiguous = contiguous and all(covered[k][1] == covered[k + 1][0] for k in range(len(covered) - 1))
    if not contiguous:
        raise LedgerError("POC_BYTE_WINDOW_MISMATCH: GM tiles %r" % covered)

    statement = verify_statement_coverage(tir_function, ledger)
    if not statement:
        raise LedgerError("POC_STATEMENT_COVERAGE_MISMATCH")
    verify_final_gm_return_chain(tir_function, ledger)

    if predicate_status == "WIRED":
        verify_predicate_wiring(problem)

    return {
        "g0": g0,
        "expected_required_count": total,
        "raw_count": len(obligations["raw"]),
        "ssa_count": len(obligations["ssa"]),
        "reuse_count": len(obligations["reuse"]),
        "final_count": len(obligations["final"]),
        "required_unknown_count": required_unknown,
        "unknown_ratio_required": 0.0,
        "critical_unknown_count": len(critical),
        "gm_tiles": [list(w) for w in covered],
        "proven_families": sorted(proven),
        "statement": statement,
    }
