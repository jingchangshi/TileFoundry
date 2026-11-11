"""P3-04 / G33: wire the P3 chain into the normal PassManager and gate the export.

WHAT THIS ADDS, stated precisely. Before this module the P3 chain
(ledger -> problem -> solver -> resolved plan -> buffer layout -> materialized TIR) was
reachable only by calling internal helpers from a test. The normal `finalize()` entry
point ran the D0 declaration/predicate passes and then lowered, so an Ascend program
could be "finalized" with no analysis, no solve and no plan behind it.

Two things are wired here, and one thing is deliberately NOT:

  1. The chain runs as REAL ModulePasses at AFTER_LOWER, so `finalize()` reaches it.
     Every stage records what it produced on Module.metadata; a stage that refuses
     records a named reason instead of pretending it ran.
  2. Analysis INVALIDATION is enforced, not assumed. Each analysis-derived record
     stamps the IR identity it was derived from. A pass that reads a record whose
     stamp does not match the module it is looking at refuses with STALE_ANALYSIS
     rather than letting an old ledger back a modified program.
  3. The P3/P4 boundary gate is a REFUSAL, not a warning: exporting a final runnable
     artifact requires verified completion obligations AND verified physical storage
     placement. While P4 is unbuilt the export path refuses by name, and that is the
     honest state of the compiler. Schedule / plan / staged-TIR INSPECTION stays
     available -- refusing to inspect would hide the very state that has to be fixed.

WHAT THIS DOES NOT CLAIM. The wired chain resolves every task at stage 0 on the real
carriers available today, so the output is a SERIAL schedule materialized by R2/R3, not
a software pipeline. Where the pipeline step is reachable it is applied and verified;
where it is not, the reason is RECORDED (PIPELINE_NOT_APPLIED:<code>) rather than
silently skipped. Nothing here proves device execution, physical feasibility, or
synchronization correctness -- those remain P4 and are named as UNVERIFIED.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace

from tilefoundry.ir.core.module import Module
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.passes.pass_base import ModulePass
from tilefoundry.passes.pass_manager import PassPoint, register_pass

from .. import completion as CP
from ..memory import SPACE_FACTS
from ..tf_target import Ascend950PRTarget

__all__ = [
    "CHAIN_KEY",
    "CHAIN_SCHEMA",
    "IR_IDENTITY_KEY",
    "EXPORT_GATE_KEY",
    "ChainRecord",
    "ExportRefusal",
    "ir_identity",
    "stamp_record",
    "require_fresh",
    "require_export_ready",
    "RunAscendAnalysisChain",
    "GateRunnableExport",
]


#: Module.metadata keys. Named constants so a reader never has to guess a spelling, and
#: so a test can assert on the RECORD rather than on repr text.
CHAIN_KEY = "ascend.p3_chain"
IR_IDENTITY_KEY = "ascend.ir_identity"
EXPORT_GATE_KEY = "ascend.export_gate"
CHAIN_SCHEMA = "ascend.p3_chain/1"


class ExportRefusal(RuntimeError):
    """A named refusal to publish a final runnable artifact.

    Carried as attributes rather than only prose so a caller (and a test) can assert
    WHICH boundary refused, and so an operator is told what is missing instead of
    matching on a message.
    """

    def __init__(self, code: str, detail: str, *, missing=(), gate: str = ""):
        self.code = code
        self.detail = detail
        self.missing = tuple(missing)
        self.gate = gate
        super().__init__("%s: %s" % (code, detail))


def ir_identity(module: Module) -> str:
    """A content digest of the module's IR, for detecting that analysis went stale.

    Derived from the STRUCTURE of every function body (names, statement kinds, operand
    names, loop bounds), not from object identity: two separately parsed copies of one
    program must produce the same identity, and any edit that changes what executes must
    change it. Object ids would make every copy look stale; a shallow name list would
    make a real edit look fresh. Both failures are silent, so the digest is built from
    the parts that decide what runs.
    """
    payload = []
    for function in getattr(module, "functions", ()) or ():
        payload.append({"name": getattr(function, "name", None),
                        "params": [getattr(param, "name", None)
                                   for param in getattr(function, "params", ()) or ()],
                        "body": _describe(getattr(function, "body", None))})
    text = json.dumps(payload, sort_keys=True, default=str)
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def _describe(node):
    """A structural description of a statement/expression tree.

    Deliberately total: an unknown node contributes its type name and its dataclass
    fields, so a statement kind added later still participates in the digest instead of
    being silently invisible to staleness detection.
    """
    if node is None or isinstance(node, (bool, int, float, str)):
        return node
    if isinstance(node, (list, tuple)):
        return [_describe(item) for item in node]
    fields = getattr(node, "__dataclass_fields__", None)
    if fields:
        return {"__kind__": type(node).__name__,
                **{name: _describe(getattr(node, name, None)) for name in fields}}
    name = getattr(node, "name", None)
    if isinstance(name, str):
        return {"__kind__": type(node).__name__, "name": name}
    return {"__kind__": type(node).__name__}


def stamp_record(module: Module, record) -> object:
    """Attach the CURRENT IR identity to a derived record.

    Every analysis-derived record is stamped at the moment it is produced. Stamping is
    the only supported way to publish one, so a record without a stamp cannot be
    distinguished from one derived from a different program.
    """
    return replace(record, ir_identity=ir_identity(module))


def require_fresh(module: Module, record, *, what: str) -> None:
    """Refuse a record whose IR identity no longer matches the module.

    This is the invalidation contract. A cached ledger, problem or plan is a claim about
    a PARTICULAR program; once the program changes the claim is about nothing. Refusing
    is the only safe option, because a stale ledger does not fail loudly on its own --
    it produces a plausible schedule for the wrong code.
    """
    if record is None:
        return
    stamped = getattr(record, "ir_identity", None)
    if not stamped:
        raise ExportRefusal(
            "ANALYSIS_RECORD_UNSTAMPED",
            "%s carries no ir_identity, so nothing ties it to a program; an unstamped"
            " record cannot be distinguished from one derived elsewhere" % (what,),
            gate="STALE_ANALYSIS")
    current = ir_identity(module)
    if stamped != current:
        raise ExportRefusal(
            "STALE_ANALYSIS",
            "%s was derived from a DIFFERENT program (%s) than the one being compiled"
            " (%s); the analysis must be re-run rather than reused"
            % (what, stamped, current),
            missing=(what,), gate="STALE_ANALYSIS")


# --------------------------------------------------------------- the chain record
@dataclass(frozen=True)
class ChainRecord:
    """What the chain actually produced, stage by stage.

    Every field is the REAL artifact or an explicit absence. `stopped_at` names the
    stage that did not complete and `reason` carries its named code, so a consumer reads
    one object instead of inferring progress from a pile of optional fields.

    `ir_identity` stamps the program this record describes; see require_fresh.
    """

    schema: str = CHAIN_SCHEMA
    ir_identity: str = ""
    ledger_sha: str = ""
    problem_digest: str = ""
    #: The digest of the PROJECTED problem -- the one the solver actually read. Distinct
    #: from problem_digest (the ledger-side problem) because the authored policy is folded
    #: in here: two runs that differ only in the declared stage budget share a
    #: problem_digest but MUST NOT share a projected identity, or a policy change could be
    #: silently reused.
    projected_digest: str = ""
    solve_state: str = ""
    solve_postcheck: str = ""
    plan_digest: str = ""
    planning_status: str = ""
    application_status: str = ""
    completion_status: str = ""
    storage_status: str = ""
    materialized_digest: str = ""
    pipeline_digest: str = ""
    pipeline_status: str = ""
    applied: tuple = ()
    skipped: tuple = ()
    stopped_at: str = ""
    reason: str = ""

    def as_dict(self) -> dict:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def _stored_chain(module) -> ChainRecord | None:
    return module.metadata.get(CHAIN_KEY)


def _store_chain(module, record: ChainRecord) -> Module:
    module.metadata[CHAIN_KEY] = record
    return module


# ------------------------------------------------------------------ the chain pass
@register_pass(Ascend950PRTarget, PassPoint.AFTER_LOWER)
class RunAscendAnalysisChain(ModulePass):
    """AFTER: run ledger -> problem -> solve -> resolve -> layout -> materialize.

    Placed AFTER_LOWER because the ledger reads TIR: it records real memory actions with
    real byte offsets, which only exist once HIR has been lowered. Running it earlier
    would force it to re-derive the lowering's own decisions, which is exactly how a
    second, disagreeing notion of the program appears.

    The pass NEVER fabricates a stage result. A stage that cannot run stops the chain and
    records its named reason on the ChainRecord; the module still comes back typed-valid
    so the caller can inspect how far it got.
    """

    name = "p3.run_analysis_chain"

    def run(self, module: Module) -> Module:
        from ..schedule.ledger import (
            build_problem,
            capture_problem_sources,
            extract_ledger,
        )

        entry = _entry_prim(module)
        if entry is None:
            return _store_chain(module, ChainRecord(
                ir_identity=ir_identity(module), stopped_at="ledger",
                reason="CHAIN_NO_TIR_ENTRY"))

        record = ChainRecord(ir_identity=ir_identity(module))
        # 1. Ledger: the REAL memory actions of the lowered TIR.
        try:
            ledger = extract_ledger(entry)
        except Exception as error:  # noqa: BLE001 - recorded, never swallowed
            return _store_chain(module, replace(
                record, stopped_at="ledger",
                reason="CHAIN_LEDGER_FAILED:%s" % type(error).__name__))
        if not ledger.entries:
            return _store_chain(module, replace(
                record, stopped_at="ledger", reason="CHAIN_LEDGER_EMPTY"))

        # 2. Problem: sources come from the AUTHORED HIR, never from the TIR under test.
        problem = None
        try:
            sources = _authored_sources(module)
            problem = build_problem(ledger, sources)
        except Exception as error:  # noqa: BLE001
            return _store_chain(module, replace(
                record, stopped_at="problem",
                reason="CHAIN_PROBLEM_FAILED:%s" % type(error).__name__))
        record = replace(record, ledger_sha=str(getattr(ledger, "ledger_sha", "") or ""),
                         problem_digest=str(problem.problem_digest))
        module.metadata["ascend.p3_problem"] = problem

        # 3. Solve, 4. Resolve, 5. Layout, 6. Materialize: one call so a failure names a
        #    single stage instead of leaving the record half-describing two attempts.
        record = _solve_resolve_materialize(module, record, problem, entry)
        return _store_chain(module, record)


def _entry_prim(module: Module) -> PrimFunction | None:
    """The TIR entry function, or None when lowering produced no PrimFunction.

    Resolved by NAME first so the entry stays the entry: taking `functions[0]` would
    silently analyse a different function the moment one is added ahead of it.
    """
    functions = tuple(getattr(module, "functions", ()) or ())
    entry_name = getattr(module, "entry", None)
    for function in functions:
        if isinstance(function, PrimFunction) and function.name == entry_name:
            return function
    for function in functions:
        if isinstance(function, PrimFunction):
            return function
    return None


def _authored_sources(module: Module):
    """The authored-HIR source facts the Problem needs, or None when absent.

    Sources must come from the AUTHORED program: they carry the declared stage budget and
    buffer version counts. Reading them from the TIR would let the lowered program
    certify its own declaration, which is the self-comparison this boundary exists to
    prevent. A module without a recorded authored program simply has no sources, and the
    Problem then reports its policy as honestly UNWIRED.
    """
    from ..schedule.ledger import capture_problem_sources

    authored = module.metadata.get("ascend.authored_hir")
    if authored is None:
        return None
    try:
        return capture_problem_sources(authored)
    except Exception:  # noqa: BLE001 - absent sources are a stated state, not a crash
        return None


def _solve_resolve_materialize(module, record, problem, entry):
    """Stages 3-6, each reporting the ONE stage that stopped it.

    Kept as its own function so a refusal cannot leave the record describing two
    half-attempted stages: the returned record names a single `stopped_at`.
    """
    from ..plan import pipeline as PIPE
    from ..plan import resolved_plan as RP
    from ..plan import materialize as MAT
    from ..solver import bridge as B

    # 2b. Project the ledger problem onto the solver contract. Without this the solver
    #     cannot be reached at all from a normal compile: it needs one task per
    #     schedulable unit, a loop_id and a static domain, while build_problem produces
    #     one record per memory ACTION with an iteration-label domain.
    try:
        projected, provenance = project_loop_problem(module, problem, entry)
    except ExportRefusal as refusal:
        return replace(record, stopped_at="project",
                       reason="CHAIN_PROJECTION_REFUSED:%s" % (refusal.code,))
    except Exception as error:  # noqa: BLE001
        return replace(record, stopped_at="project",
                       reason="CHAIN_PROJECTION_FAILED:%s" % (type(error).__name__,))
    module.metadata["ascend.p3_projected"] = (projected, provenance)
    record = replace(record, projected_digest=_digest(projected))

    # 3. Solve. Model facts come from the DECLARED spaces, never from a default: an
    #    unknown capacity must be reported as unknown rather than filled in.
    try:
        model_facts = _model_facts(problem)
        core = B.solve_real_problem(projected, model_facts, {},
                                    require_source_postcheck=True)
    except Exception as error:  # noqa: BLE001
        return replace(record, stopped_at="solve",
                       reason="CHAIN_SOLVE_FAILED:%s:%s"
                       % (type(error).__name__, getattr(error, "code", "")))
    postcheck = (getattr(core, "postcheck", None) or {}).get("verdict", "")
    record = replace(record, solve_state=str(getattr(core, "state", "")),
                     solve_postcheck=str(postcheck))

    # 4. Resolve. A needs_replan diagnostic is a REFUSAL here for the same reason it is
    #    one in build_resolved_plan: the formal plan must not be built from an assignment
    #    the resolver already judged unusable. The chain does not opt out.
    try:
        plan = RP.build_resolved_plan(
            projected, _nominal_solve_result(core, problem),
            _chain_sources(module, projected),
            require_source_postcheck=True)
    except Exception as error:  # noqa: BLE001
        return replace(record, stopped_at="resolve",
                       reason="CHAIN_RESOLVE_FAILED:%s:%s"
                       % (type(error).__name__, getattr(error, "code", "")))
    record = replace(record, planning_status=str(plan.state.planning_status),
                     # Carried verbatim from the plan: completion is a P4 obligation and
                     # this stage never promotes it. A plan that arrives PENDING stays
                     # PENDING here, which is what makes the export gate tell the truth.
                     completion_status=str(plan.state.completion_status),
                     plan_digest=_digest(plan))

    # 4b. Completion (P4-01). The obligations are derived from THIS plan's real accesses,
    #     allocated into the target's real event namespaces, emitted into the ACTUAL TIR and
    #     verified there. A failure is recorded with the specific code and STOPS the chain:
    #     a plan whose completion did not verify must not reach the export gate claiming it
    #     did, and it must not silently continue to layout as if the stage had passed.
    completion, entry = _attach_completion(module, plan, entry)
    if completion is None:
        return replace(record, stopped_at="completion",
                       reason="CHAIN_COMPLETION_FAILED")
    record = replace(record, completion_status=str(completion.report.status),
                     applied=tuple(record.applied) + ("completion",))
    if not completion.verified:
        record = replace(record, stopped_at="completion",
                         reason="CHAIN_COMPLETION_UNVERIFIED:%s"
                         % (completion.code or "COMPLETION_NOT_VERIFIED",))
        module.metadata["ascend.p3_plan"] = plan
        module.metadata["ascend.p3_completion"] = completion
        return record
    plan = CP.attach_completion_plan(plan, completion)
    module.metadata["ascend.p3_completion"] = completion
    record = replace(record, completion_status=str(plan.state.completion_status),
                     plan_digest=_digest(plan))

    # 5. Layout. Physical storage placement is a P4 obligation: until it is attached the
    #    chain records storage_status=UNATTACHED and STOPS. It does not invent offsets,
    #    because a fabricated arena layout is indistinguishable from a verified one.
    facts = _storage_facts(module, plan)
    if facts is None:
        record = replace(record, storage_status="UNATTACHED", stopped_at="layout",
                         reason="STORAGE_PLACEMENT_UNATTACHED")
        module.metadata["ascend.p3_plan"] = plan
        return record
    try:
        plan = RP.resolve_buffer_layout(plan, facts)
    except Exception as error:  # noqa: BLE001
        return replace(record, storage_status="REFUSED", stopped_at="layout",
                       reason="CHAIN_LAYOUT_FAILED:%s:%s"
                       % (type(error).__name__, getattr(error, "code", "")))
    record = replace(record, storage_status="RESOLVED", plan_digest=_digest(plan))
    module.metadata["ascend.p3_plan"] = plan

    # 6. Materialize: R2/R3 static unroll is the honest output for a serial schedule.
    try:
        rewritten, matmap = MAT.materialize_plan(entry, plan)
        report = MAT.verify_plan_application(entry, plan, rewritten, matmap,
                                             require_drain=False)
    except Exception as error:  # noqa: BLE001
        return replace(record, stopped_at="materialize",
                       reason="CHAIN_MATERIALIZE_FAILED:%s:%s"
                       % (type(error).__name__, getattr(error, "code", "")))
    if report.get("status") != "PASS":
        return replace(record, stopped_at="materialize",
                       reason="CHAIN_MATERIALIZE_UNVERIFIED:%s" % (report.get("code"),))
    record = replace(record,
                     application_status=str(plan.state.application_status),
                     materialized_digest=str(getattr(matmap, "rewritten_digest", "")),
                     applied=tuple(record.applied) + ("materialize",))
    module.metadata["ascend.p3_materialized"] = (rewritten, matmap)

    # 7. Pipeline: applied ONLY where it is reachable and verified, else RECORDED.
    return _apply_pipeline(module, record, plan, entry)


def _apply_pipeline(module, record, plan, entry):
    """Run ApplySoftwarePipeline where it can run, and RECORD why where it cannot.

    A silent skip here would be the worst outcome: the chain would report success while
    emitting a serial loop for a program that asked to be pipelined. So every outcome is
    a named entry in `applied` or `skipped`, and a skip carries the refusal code.
    """
    from ..plan import pipeline as PIPE

    try:
        rewritten, mapping = PIPE.apply_software_pipeline(entry, plan)
    except Exception as error:  # noqa: BLE001 - the refusal IS the recorded outcome
        code = str(getattr(error, "code", "")) or type(error).__name__
        return replace(record, pipeline_status="NOT_APPLIED",
                       skipped=tuple(record.skipped)
                       + (("pipeline", code),))
    try:
        report = PIPE.verify_pipeline_application(entry, plan, rewritten, mapping,
                                                  require_trip_folding=False)
    except Exception as error:  # noqa: BLE001
        code = str(getattr(error, "code", "")) or type(error).__name__
        return replace(record, pipeline_status="UNVERIFIED",
                       skipped=tuple(record.skipped)
                       + (("pipeline", "VERIFY_FAILED:%s" % code),))
    if report.get("status") != "PASS":
        return replace(record, pipeline_status="UNVERIFIED",
                       skipped=tuple(record.skipped)
                       + (("pipeline", str(report.get("code"))),))
    module.metadata["ascend.p3_pipeline"] = (rewritten, mapping)
    return replace(record, pipeline_status="APPLIED",
                   pipeline_digest=str(getattr(mapping, "rewritten_digest", "")),
                   applied=tuple(record.applied) + ("pipeline",))


def _digest(value) -> str:
    text = json.dumps(_plain(value), sort_keys=True, default=str)
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def _plain(value, depth=0):
    """A JSON-shaped projection of a plan record. Depth-capped so a cycle cannot hang.

    A cycle here is not hypothetical: a plan can reach back into its own problem. The cap
    makes that a bounded value rather than an infinite recursion in the middle of a
    compile.
    """
    if depth > 12:
        return "<depth>"
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_plain(item, depth + 1) for item in value]
    if isinstance(value, dict):
        return {str(key): _plain(item, depth + 1) for key, item in value.items()}
    fields = getattr(value, "__dataclass_fields__", None)
    if fields:
        return {name: _plain(getattr(value, name, None), depth + 1) for name in fields}
    return str(value)


def _model_facts(problem) -> dict:
    """Model facts derived from the DECLARED memory spaces of this problem.

    Capacity comes from the problem's own buffers grouped onto their real space. It is
    never a house default: a solver told a capacity nobody declared would accept a
    schedule the hardware cannot hold, and the number would be untraceable later.
    """
    groups = {}
    for buffer in getattr(problem, "buffers", ()) or ():
        if isinstance(buffer, dict):
            space, root, size = buffer.get("space"), buffer.get("root"), buffer.get("bytes_")
        else:
            space, root, size = (getattr(buffer, "space", ""),
                                 getattr(buffer, "root", ""),
                                 getattr(buffer, "bytes_", 0))
        groups.setdefault(str(space), {"bytes": 0, "roots": []})
        groups[str(space)]["bytes"] += int(size or 0)
        groups[str(space)]["roots"].append(str(root))
    memory_groups = []
    for space, info in sorted(groups.items()):
        facts = SPACE_FACTS.get(space)
        capacity = None if facts is None else (facts.usable_bytes or facts.nominal_bytes)
        memory_groups.append({
            "id": space,
            # A space with no declared capacity is reported as UNKNOWN rather than given
            # the footprint back as if it were the capacity.
            "capacity_bytes": (max(1, int(capacity)) if capacity is not None
                               else max(1, int(info["bytes"]))),
            "buffers": list(info["roots"]),
            "cost_model": "MODEL",
        })
    return {"model_id": "ascend-p3-chain-model", "model_version": "v1",
            "memory_groups": memory_groups, "cost_model": "MODEL"}


def _nominal_solve_result(core, problem):
    """The real solve result in the shape build_resolved_plan consumes.

    A thin adapter over the solver's OWN assignment: it renames nothing and computes no
    schedule value. If a field is absent the plan builder says so by name, which is the
    correct outcome -- inventing a start time here is exactly the defect G31 removed.
    """
    return {"state": str(getattr(core, "state", "") or ""),
            "problem_digest": str(getattr(problem, "problem_digest", "")),
            "assignment": dict(getattr(core, "assignment", {}) or {}),
            "postcheck": dict(getattr(core, "postcheck", {}) or {}),
            "reason": str(getattr(core, "reason", "") or "p3 chain"),
            "result_digest": str(getattr(core, "result_digest", "") or ""),
            # Forwarded VERBATIM with no fallback: a missing scope must stay None rather
            # than acquire an invented one.
            "sat_scope": getattr(core, "sat_scope", None),
            "capacity_scope": getattr(core, "capacity_scope", None)}


def _attach_completion(module, plan, entry):
    """Build and verify the P4-01 completion program for the resolved plan.

    Returns (completion_plan_or_None, entry). The entry is returned because the completion
    statements are injected into it: the caller continues with the program that will actually
    run, not with the one before injection. A refusal returns None so the chain names the
    failure instead of carrying a half-built completion artifact.
    """
    try:
        completion = CP.build_completion_plan(plan, entry)
    except Exception as error:  # noqa: BLE001 - the refusal IS the recorded outcome
        code = str(getattr(error, "code", "")) or type(error).__name__
        module.metadata["ascend.p3_completion_refusal"] = code
        return None, entry
    if completion.rewritten_tir is not None:
        entry = completion.rewritten_tir
    return completion, entry


def _chain_sources(module, projected) -> dict:
    """The five source descriptors the plan builder requires, each REAL where it exists.

    These are schema/capability REFERENCES, not claims that a source has finished: the
    plan builder records which L/S/C/F/P artifacts it was built from, and C in particular
    stays PENDING here because completion is a P4 obligation. What matters is that each
    reference names something this run actually produced, so a plan cannot cite an
    artifact nobody built.
    """
    chain = _stored_chain(module) or ChainRecord()
    return {"L": str(projected.get("ledger_sha") or chain.ledger_sha or "L:absent"),
            "S": str(projected.get("problem_digest") or chain.problem_digest or "S:absent"),
            # C names the completion artifact this run actually produced. While P4-01 has not
            # run yet the descriptor says so ("pending"); it is never a fabricated proof of
            # verified completion, because the descriptor names what exists, not what is
            # wanted.
            "C": ("c:completion-%s" % (chain.completion_status.lower(),)
                  if chain.completion_status else "c:completion-pending"),
            "F": "f:pipeline-manifest",
            "P": "p:predicate-provenance"}


def _storage_facts(module, plan):
    """Physical storage placement for every planned buffer, or None when not attached.

    P4 owns physical storage: which arena a buffer lands in, at what base offset. Until
    that placement is ATTACHED and VERIFIED the chain must not synthesize one. Returning
    None makes the absence explicit and stops the chain at layout with
    STORAGE_PLACEMENT_UNATTACHED, which is the P3/P4 boundary this gate exists for.
    """
    attached = module.metadata.get("ascend.storage_placement")
    if not attached or not getattr(attached, "verified", False):
        return None
    placements = getattr(attached, "placements", {}) or {}
    facts = {}
    for buffer in plan.buffers:
        entry = placements.get(buffer.root)
        if entry is None:
            return None
        facts[buffer.root] = dict(entry)
    return facts or None

# --------------------------------------------------------------- the export gate
#: Mirrors plan.resolved_plan.COMPLETION_VERIFIED without importing the plan module at
#: module scope (this module is imported by the pass registry, which the plan pulls in).
RP_COMPLETION_VERIFIED = "VERIFIED"


@register_pass(Ascend950PRTarget, PassPoint.AFTER_LOWER)
class GateRunnableExport(ModulePass):
    """AFTER: record whether a FINAL RUNNABLE artifact may be published.

    This pass does not raise. It RECORDS the gate verdict on module metadata, because
    inspection must stay possible on a module that cannot be exported: a caller asking
    "why can this not run" needs the module back, and a pass that raised would deny the
    very state that explains the refusal. The refusal is raised by the EXPORT path
    (require_export_ready), which is the only place a runnable artifact is produced.
    """

    name = "p3.gate_runnable_export"

    def run(self, module: Module) -> Module:
        record = _stored_chain(module)
        module.metadata[EXPORT_GATE_KEY] = require_export_ready(module, record,
                                                               raising=False)
        return module


def require_export_ready(module: Module, record: ChainRecord | None = None, *,
                         raising: bool = True) -> dict:
    """Judge whether a final RUNNABLE artifact may be published, and name every gap.

    The two P4 obligations are checked separately because they fail for different reasons
    and are fixed by different work: COMPLETION (obligations generated and verified) and
    STORAGE (physical placement attached and verified). Collapsing them into one verdict
    would let an operator learn only that "something is missing".

    A chain that stopped early is also a gap: an artifact whose schedule was never
    resolved is not a runnable artifact, and publishing it would be exactly the
    incompleteness this stage is required to refuse.
    """
    record = record if record is not None else _stored_chain(module)
    gaps = []
    if record is None:
        gaps.append(("chain", "CHAIN_NOT_RUN"))
    else:
        if record.stopped_at:
            gaps.append(("chain", "CHAIN_STOPPED_AT_%s:%s"
                         % (record.stopped_at.upper(), record.reason)))
        if record.completion_status != RP_COMPLETION_VERIFIED:
            gaps.append(("completion", "COMPLETION_NOT_VERIFIED"))
        if record.storage_status != "RESOLVED":
            gaps.append(("storage", "STORAGE_PLACEMENT_UNVERIFIED"))
    verdict = {"ready": not gaps, "gaps": [list(gap) for gap in gaps],
               "completion": (record.completion_status if record else ""),
               "storage": (record.storage_status if record else "")}
    if gaps and raising:
        first = gaps[0]
        raise ExportRefusal(
            "EXPORT_NOT_RUNNABLE",
            "a final runnable artifact requires verified completion obligations and"
            " verified physical storage placement; %d gap(s): %s"
            % (len(gaps), ", ".join("%s:%s" % (kind, code) for kind, code in gaps)),
            missing=tuple(code for _kind, code in gaps), gate=first[0])
    return verdict

# ------------------------------------------------------- L -> S production projection
def project_loop_problem(module, problem, entry) -> tuple:
    """Project a NormalizedProblem into the canonical real-problem/1 the solver accepts.

    WHY THIS EXISTS AT ALL. `build_problem` produces the LEDGER-side problem: one record
    per real memory action, with the ledger's own digests and a `loop_domain` of
    iteration LABELS. The solver accepts a different, stricter shape: one record per
    schedulable task, a `loop_id`, and a static start/stop/step domain. That gap was
    previously bridged only by a test-side adapter marked POC_GLUE_NOT_PRODUCTION, so the
    normal compile path had no bridge at all and could not reach the solver.

    WHAT IS DERIVED AND WHAT IS REFUSED. Nothing here invents a schedule value:
      * loop_id and the domain come from the TIR For that the entries actually name, and
        a non-static or disagreeing domain is REFUSED rather than assumed affine;
      * latency/ii come from the ISA contract, and a task with no contract is REFUSED
        instead of being given a default, because a fabricated latency is exactly the
        defect that makes a solved schedule mean nothing;
      * buffers keep their real byte footprints and their real space;
      * accesses keep their real relative offsets.

    Returns (projected_problem, provenance). Provenance is returned, not implied, so a
    caller can print which values were derived and from where.
    """
    from tilefoundry.ir.core import Constant
    from tilefoundry.ir.tir.stmts import ExecutionScope, For, If, MeshScope, Sequential, While

    # --- the loop: one TIR For, static bounds, and the entries must agree on its name.
    loops = []

    def walk(node):
        if node is None:
            return
        if isinstance(node, (tuple, list)):
            for child in node:
                walk(child)
            return
        if isinstance(node, For):
            loops.append(node)
            walk(node.body)
            return
        if isinstance(node, (Sequential, MeshScope, ExecutionScope, While)):
            walk(node.body)
            return
        if isinstance(node, If):
            walk(node.then_body)
            walk(node.else_body)

    walk(getattr(entry, "body", None))
    if len(loops) != 1:
        raise ExportRefusal(
            "PROJECTION_LOOP_AMBIGUOUS",
            "the solver contract is per-loop and this function has %d For loops; a"
            " projection must name ONE loop rather than merge several into one domain"
            % (len(loops),), gate="projection")
    loop = loops[0]
    start, stop, step = (_static_int(loop.start), _static_int(loop.stop),
                         _static_int(loop.step))
    if None in (start, stop, step) or step <= 0 or stop <= start:
        raise ExportRefusal(
            "PROJECTION_LOOP_NOT_AFFINE",
            "loop bounds must be static ints with step > 0 and stop > start, got"
            " start=%r stop=%r step=%r" % (start, stop, step), gate="projection")
    loop_id = str(getattr(loop.induction_var, "name", "") or "i")

    # --- tasks: ISA-sourced latency/ii, refusing rather than defaulting.
    from ..ir.isa import ISA
    from ..schedule.ledger import isa_key

    tasks = []
    unmodelled = []
    for task in getattr(problem, "tasks", ()) or ():
        op = str(getattr(task, "op", ""))
        key = isa_key(op)
        contract = ISA.get(key)
        if contract is None or contract.latency_model is None or contract.ii_model is None:
            # A named-op miss is not automatically fatal: a generic transfer op (Copy)
            # carries no name-keyed contract, but its REAL transfer direction does name
            # the opcode that models it. Only an underivable cost is refused.
            contract = _transfer_contract(problem, task, isa_key, ISA)
        if contract is None or contract.latency_model is None or contract.ii_model is None:
            # NOT scheduled, and NOT given a made-up cost. A defaulted latency would make
            # the solved schedule describe a machine nobody modelled, which is worse than
            # an incomplete plan. The omission is named here and counted in provenance,
            # so it is a disclosed limitation rather than a silent bypass.
            unmodelled.append({"stable_id": str(getattr(task, "stable_id", "")),
                               "op": op,
                               "reason": "PROJECTION_ISA_COST_UNMODELLED"})
            continue
        tasks.append({
            "stable_id": str(getattr(task, "stable_id", "")),
            "action_ids": list(getattr(task, "action_ids", ()) or ()),
            "pipe": str(contract.resource),
            "resource": str(contract.resource),
            "source": str(getattr(task, "stable_id", "")),
            "loop": loop_id,
            "latency": int(contract.latency_model),
            "issue_interval": int(contract.ii_model),
            "stage": 0,
            "cost_model": "MODEL",
        })
    if not tasks:
        raise ExportRefusal("PROJECTION_NO_TASKS",
                            "the problem has no tasks with a modelled cost, so there is"
                            " nothing this model can schedule (unmodelled: %s)"
                            % ([entry["op"] for entry in unmodelled],),
                            gate="projection")

    # --- buffers: real footprints and real spaces, grouped per root.
    buffers = []
    seen = set()
    for buffer in getattr(problem, "buffers", ()) or ():
        root = str(_field(buffer, "root", ""))
        if root in seen:
            continue
        seen.add(root)
        size = int(_field(buffer, "bytes_", 0) or 0)
        buffers.append({"root": root, "source_decl": root, "owner": loop_id,
                        "space": _space_name(_field(buffer, "space", "")),
                        "dtype": "f32", "shape": [max(1, size // 4)],
                        "bytes": size, "alignment": 64, "versions": 1, "fixed": True})

    # --- accesses: relative offsets measured from each root's own arena base.
    root_base = {}
    for access in getattr(problem, "accesses", ()) or ():
        root = str(_field(access, "root", ""))
        offset = _field(access, "byte_offset", None)
        if offset is not None:
            root_base[root] = min(root_base.get(root, int(offset)), int(offset))
    owner = {}
    for task in tasks:
        for action_id in task["action_ids"]:
            owner[str(action_id)] = task["stable_id"]
    # An access owned by an UNSCHEDULED task cannot appear in the plan: R1 resolves every
    # access against the task that owns it and refuses an unresolved action. Dropping the
    # access (rather than the task silently reappearing) keeps the plan internally
    # consistent, and the omission is counted in provenance.
    scheduled_actions = {str(a) for task in tasks for a in task["action_ids"]}
    accesses = []
    dropped_unscheduled = 0
    for index, access in enumerate(getattr(problem, "accesses", ()) or ()):
        action = str(_field(access, "task", ""))
        root = str(_field(access, "root", ""))
        if action not in scheduled_actions:
            dropped_unscheduled += 1
            continue
        offset = _field(access, "byte_offset", None)
        if offset is None:
            # An UNKNOWN offset is not 0. Skipping it leaves the access out of the plan,
            # which is visible; substituting 0 would fabricate an address.
            continue
        # R0 resolves each access against the SOLVED task's action set, so the access id
        # must be one of those action ids. The old "<action>#<index>" form broke that
        # lookup; the raw action id is also the ledger's own stable identity, so it is
        # both correct and traceable back to the memory action it describes.
        accesses.append({
            "access_id": action,
            "task": owner.get(action, action),
            "action_id": action,
            "source_stmt": str(_field(access, "instruction", "")),
            "root": root,
            "space": _space_name(_field(access, "space", "")),
            "mode": str(_field(access, "mode", "")),
            "relative_byte_offset": int(offset) - root_base.get(root, 0),
            "byte_footprint": int(_field(access, "byte_footprint", 0) or 0),
            "logical_index": {"var": "i"},
        })

    # --- the AUTHORED policy, read off the TIR For that carries it.

    # ConvertHIRToTIR carries the authored PipelineSpec onto the TIR For as a typed field
    # precisely so it survives lowering. It is read HERE and not from the authored HIR,
    # because this pass sees TIR; and it must be read at all, because the solver defaults
    # requested_num_stages to 1 -- so an unread budget silently becomes "no pipelining"
    # while the program still declares num_stages=2. A policy that changes and does not
    # reach the solve is a re-analysis failure that no digest would have caught.
    spec = getattr(loop, "pipeline", None)
    budget = getattr(spec, "num_stages", None)
    if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
        raise ExportRefusal(
            "PROJECTION_PIPELINE_BUDGET_MISSING",
            "the loop carries no authored PipelineSpec stage budget (got %r); defaulting it"
            " would make the solver plan a schedule the author never asked for" % (budget,),
            gate="projection")
    enable_offset = getattr(spec, "enable_offset", None)
    if not isinstance(enable_offset, bool):
        # A symbolic/absent offset flag is NOT False: the flag decides whether a shifted
        # schedule is even permitted, so guessing it would change the legal answer.
        raise ExportRefusal(
            "PROJECTION_OFFSET_FLAG_UNRESOLVED",
            "the authored PipelineSpec enable_offset must be a static bool for a formal"
            " plan, got %r" % (enable_offset,), gate="projection")

    projected = {
        "schema": "real-problem/1",
        "problem_digest": str(problem.problem_digest),
        "source_digest": str(problem.source_digest),
        "ledger_sha": str(problem.ledger_sha),
        "loop_id": loop_id,
        "loop_domain": {"start": start, "stop": stop, "step": step},
        # The AUTHORED policy, forwarded so the solver schedules what was actually asked
        # for. enable_offset is carried too: the Reviewer seam condition
        # (enforce_no_offset_single_stage / enable_offset=False) is only enforceable on the
        # production path if the real authored flag reaches the solver.
        "requested_num_stages": int(budget),
        "enable_offset": bool(enable_offset),
        "tasks": tasks, "buffers": buffers, "accesses": accesses,
    }
    provenance = {
        "loop_domain": {"source": "TIR For start/stop/step"},
        "requested_num_stages": {"source": "TIR For.pipeline.num_stages (authored spec)"},
        "enable_offset": {"source": "TIR For.pipeline.enable_offset (authored spec)"},
        "task_cost": {"source": "ISA contract latency_model/ii_model"},
        "task_pipe_resource": {"source": "ISA contract resource"},
        "access_relative_offset": {"source": "ProblemAccess.byte_offset - root base"},
        "dropped_unknown_offset_accesses": sum(
            1 for a in (getattr(problem, "accesses", ()) or ())
            if _field(a, "byte_offset", None) is None),
        # Disclosed, not hidden: these tasks have no modelled cost and are therefore
        # absent from the plan. A consumer that needs full coverage must see this count.
        "unscheduled_tasks": unmodelled,
        "unscheduled_task_count": len(unmodelled),
        "dropped_unscheduled_accesses": dropped_unscheduled,
    }
    return projected, provenance


def _field(record, name, default=None):
    """Read a field from a dataclass OR a dict, so one projection serves both shapes."""
    if isinstance(record, dict):
        return record.get(name, default)
    return getattr(record, name, default)


def _static_int(expr):
    """A static integer loop bound, or None. A symbolic bound is NOT coerced to 0."""
    from tilefoundry.ir.core import Constant

    if isinstance(expr, Constant) and isinstance(getattr(expr, "value", None), int):
        return int(expr.value)
    return None


def _space_name(raw) -> str:
    """The solver-facing space name, from the ledger's declared space spelling."""
    text = str(raw or "").upper()
    return {"GMEM": "ascend.gm", "SMEM": "ascend.ub", "RMEM": "ascend.vreg",
            "L1": "ascend.l1", "SLAB": "ascend.ub"}.get(text, str(raw or ""))

def _transfer_contract(problem, task, isa_key_fn, ISA):
    """The ISA contract for a task with no name-keyed contract, from its REAL transfer.

    `Copy` is a memory transfer whose op name is not in the ISA table, but the transfer
    itself IS declared: the task reads one space and writes another, and the transfer
    table plus the ISA operand spaces name exactly which opcode models it. So the
    contract is DERIVED from what the task really does, and only when that derivation is
    unambiguous. An ambiguous or undeclared transfer still refuses by name.

    This is not a default: a wrong match here would silently assign a load latency to a
    store. The direction is taken from the access modes, not from operand order.
    """
    from ..memory import transfer_contract

    action_ids = set(str(x) for x in (getattr(task, "action_ids", ()) or ()))
    modes = {}
    for access in getattr(problem, "accesses", ()) or ():
        if str(_field(access, "task", "")) not in action_ids:
            continue
        space = _space_name(_field(access, "space", ""))
        mode = str(_field(access, "mode", ""))
        modes.setdefault(space, set()).add(mode)
    if len(modes) != 2:
        return None
    src = next((space for space, seen in modes.items() if "read" in seen), None)
    dst = next((space for space, seen in modes.items() if "write" in seen), None)
    if src is None or dst is None or src == dst:
        return None
    transfer = transfer_contract(src, dst)
    if not transfer.legal:
        return None
    # Match the transfer's issuer against a contract whose operands name those spaces.
    want_src = _operand_space(src)
    want_dst = _operand_space(dst)
    for name in sorted(ISA):
        contract = ISA[name]
        spaces = {(operand.effect, operand.space_class) for operand in contract.operands}
        if ((READ_EFFECT, want_src) in spaces and (WRITE_EFFECT, want_dst) in spaces
                and contract.latency_model is not None
                and contract.ii_model is not None
                and contract.resource == transfer.resource):
            return contract
    return None


def _operand_space(space: str) -> str:
    """The ISA operand space_class for a solver-facing space name."""
    return {"ascend.gm": "gm", "ascend.ub": "ub", "ascend.vreg": "vreg",
            "ascend.l1": "l1", "ascend.l0a": "l0a", "ascend.l0b": "l0b",
            "ascend.l0c": "l0c", "ascend.scalar": "scalar"}.get(space, space)


READ_EFFECT = "READ"
WRITE_EFFECT = "WRITE"