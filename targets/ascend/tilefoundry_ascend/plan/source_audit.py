"""Independent checks of instruction effects against actual TIR operations.

This is a building block, not a source-coverage verdict: aliasing, dependence
and generation proofs must also succeed before a problem can be certified.
"""
import json
from dataclasses import dataclass

from tilefoundry.ir.core.param_def import MemoryEffect
from tilefoundry.ir.tir.stmts import Evaluate, ExecutionScope, Sequential
from tilefoundry_ascend.ir.isa import ISA


@dataclass(frozen=True)
class ReadEvidence:
    """One operand read and the write (or ENTRY boundary) that reaches it."""

    access_id: str
    task_id: str
    operand_index: int
    buffer_root: str
    producer: str
    producer_effect: str
    effect_id: str
    from_entry: bool

    def canonical(self) -> dict:
        return {"access_id": self.access_id, "task_id": self.task_id,
                "operand_index": int(self.operand_index),
                "buffer_root": self.buffer_root, "producer": self.producer,
                "producer_effect": self.producer_effect,
                "effect_id": self.effect_id,
                "from_entry": bool(self.from_entry)}


@dataclass(frozen=True)
class WriteEvidence:
    """One covering write of one root, with the effect that declared it."""

    access_id: str
    task_id: str
    buffer_root: str
    effect_id: str

    def canonical(self) -> dict:
        return {"access_id": self.access_id, "task_id": self.task_id,
                "buffer_root": self.buffer_root, "effect_id": self.effect_id}


class SourceAuditError(ValueError):
    def __init__(self, code, detail):
        self.code = code
        super().__init__(f"{code}: {detail}")


def _instructions(node):
    if isinstance(node, Evaluate):
        yield node
    elif isinstance(node, (ExecutionScope, Sequential)):
        for child in (node.body.body if isinstance(node, ExecutionScope) else node.body):
            yield from _instructions(child)


def audit_instruction_effects(region):
    """Check actual op schemas, operand identities and ISA effect declarations.

    This does not establish roots, access relations, guards, cross-region edges,
    loop-carried dependencies, or completion latency. Callers must establish
    these independently before accepting a schedule.
    """
    statements = {}
    for task in region.tasks:
        unit = task.unit_ref
        for statement in _instructions(unit.node_ref):
            key = id(statement)
            if key in statements:
                raise SourceAuditError("SOURCE_INSTRUCTION_DUPLICATE", "instruction occurs in two tasks")
            statements[key] = (task.task_id, statement)
    effects = {}
    for effect in region.effects:
        key = id(effect.source_ref)
        if key in effects or key not in statements:
            raise SourceAuditError("SOURCE_EFFECT_COVERAGE", "duplicate or foreign instruction effect")
        effects[key] = effect
    if set(effects) != set(statements):
        raise SourceAuditError("SOURCE_EFFECT_COVERAGE", "instruction effect missing")
    accesses = {}
    for access in region.accesses:
        key = (id(access.source_ref), access.operand_index)
        if key in accesses or key[0] not in statements:
            raise SourceAuditError("SOURCE_ACCESS_COVERAGE", "duplicate or foreign operand access")
        accesses[key] = access
    consumed = set()
    for key, (task_id, statement) in statements.items():
        schema = getattr(type(statement.callable), "_op_schema", None)
        if schema is None or not schema.name.startswith("ascend_"):
            raise SourceAuditError("SOURCE_ISA_UNSUPPORTED", "instruction has no Ascend ISA contract")
        contract = ISA.get(schema.name[len("ascend_"):])
        if contract is None:
            raise SourceAuditError("SOURCE_ISA_UNSUPPORTED", "instruction has no Ascend ISA contract")
        params = tuple(p for p in schema.signature if p.kind == "input")
        if len(params) != len(statement.args):
            raise SourceAuditError("SOURCE_ACCESS_UNSUPPORTED", "instruction arity differs from schema")
        access_ids = []
        for index, (param, operand) in enumerate(zip(params, statement.args, strict=True)):
            access = accesses.get((key, index))
            if param.effect is None or access is None:
                raise SourceAuditError("SOURCE_ACCESS_COVERAGE", "operand effect is undeclared or missing")
            modes = {MemoryEffect.READ: "READ", MemoryEffect.WRITE: "WRITE",
                     MemoryEffect.READ | MemoryEffect.WRITE: "READ_WRITE"}
            if (access.task_id != task_id or access.view_ref is not operand
                    or access.mode != modes.get(param.effect)):
                raise SourceAuditError("SOURCE_ACCESS_MISMATCH", "operand or declared access mode changed")
            access_ids.append(access.access_id)
            consumed.add((key, index))
        completion = {"issue": "ISSUE", "issue+visible": "WRITE_VISIBLE",
                      "issue+write_visible": "WRITE_VISIBLE", "issue+data_ready": "READ_COMPLETE",
                      "issue+state_complete": "STATE_COMPLETE"}.get(contract.completion)
        if completion is None:
            raise SourceAuditError("SOURCE_COMPLETION_UNSUPPORTED", "ISA completion has no supported mapping")
        effect = effects[key]
        if (effect.task_id != task_id or effect.kind != "MEMORY" or effect.state_ref is not None
                or effect.pipe != contract.resource or effect.completion != completion
                or getattr(effect, "instruction", None) != contract.name
                or effect.access_ids != tuple(access_ids)):
            raise SourceAuditError("SOURCE_EFFECT_MISMATCH", "problem effect differs from actual instruction")
    if consumed != set(accesses):
        raise SourceAuditError("SOURCE_ACCESS_COVERAGE", "extra operand access")


def verify_model_source(source) -> bool:
    """True when the locator names a row the delay table actually holds.

    An EdgeProof may carry any structurally valid locator, including one whose
    symbol or field no registered row matches - a pointer to nothing is not a
    source. The row looked up must carry the same field value the locator
    states, so a stale or invented locator cannot stand in for evidence.
    """
    from .delays import _MODELLED_GAPS, _OCCUPANCY

    if getattr(source, "module", "") == "plan/delays.py":
        return True  # predicate/profile rows are their own source
    rows = dict(_OCCUPANCY)
    rows.update(_MODELLED_GAPS)
    row = rows.get(getattr(source, "symbol", ""))
    if row is None:
        return False
    fact = row.source
    return (fact.module == source.module and fact.field == source.field
            and getattr(fact.value, "value", fact.value) ==
            getattr(source.value, "value", source.value))


def audit_plan_buffers(plan, problem):
    """Compare every actual problem access root with region and plan buffers.

    This closes omission and substitution gaps. It is not dependence proof:
    callers must independently validate access roots/relations and edges.
    """
    regions = tuple(problem.regions)
    plans = tuple(plan.regions)
    if len(regions) != len(plans):
        raise SourceAuditError("SOURCE_REGION_COVERAGE", "plan and problem region counts differ")
    matched = set()
    for region in regions:
        matches = [p for p in plans if p.region_ref is region.region_ref]
        if len(matches) != 1 or id(matches[0]) in matched:
            raise SourceAuditError("SOURCE_REGION_COVERAGE", "region is missing or duplicated")
        current = matches[0]
        matched.add(id(current))
        audit_instruction_effects(region)
        roots = {id(a.root_ref): a.root_ref for a in region.accesses}
        buffers = {id(b.root_ref): b for b in region.buffers}
        if len(buffers) != len(region.buffers) or set(roots) != set(buffers):
            raise SourceAuditError("SOURCE_BUFFER_COVERAGE", "problem buffers differ from actual access roots")
        planned = {id(b.buffer_ref): b for b in current.buffers}
        if len(planned) != len(current.buffers) or set(planned) != set(buffers):
            raise SourceAuditError("SOURCE_BUFFER_COVERAGE", "planned buffers differ from problem access roots")
        for key, buffer in buffers.items():
            if roots[key] is not buffer.root_ref or planned[key].buffer_ref is not buffer.root_ref:
                raise SourceAuditError("SOURCE_BUFFER_IDENTITY", "buffer root is not the same object")
            if planned[key].buffer_id != buffer.buffer_id or planned[key].domain != buffer.domain:
                raise SourceAuditError("SOURCE_BUFFER_MISMATCH", "buffer identity or version domain changed")
            version = planned[key].resolved_versions
            domain = buffer.domain
            if (getattr(domain, "value", version) != version
                    or version < getattr(domain, "minimum", version)
                    or version > getattr(domain, "maximum", version)):
                raise SourceAuditError("SOURCE_BUFFER_VERSION", "resolved versions violate the source domain")

def read_static_ub_windows(entry):
    """Read static f32 UB byte windows from real TIR; no dependence certification.

    Exact repeated windows alias; distinct overlapping windows are unsupported.
    Unknown address, shape, type or layout is refused rather than guessed.
    """
    from tilefoundry.ir.core import Call, Constant
    from tilefoundry.ir.tir.memory.tensor_view import TensorView
    from tilefoundry.ir.tir.stmts import LetStmt, MeshScope
    from tilefoundry.ir.types.utils import static_dim_value

    windows = []
    def visit(node):
        if isinstance(node, Sequential):
            for item in node.body:
                visit(item)
        elif isinstance(node, (MeshScope, ExecutionScope)):
            visit(node.body)
        elif isinstance(node, LetStmt):
            value = node.value
            if not isinstance(value, Call) or not isinstance(value.target, TensorView):
                return
            ty = node.var.type
            if ty.storage.name != "SMEM":
                return
            base = value.args[0] if len(value.args) == 1 else None
            shape = tuple(static_dim_value(d) for d in ty.shape)
            layout = ty.layout
            if (not isinstance(base, Constant) or type(base.value) is not int
                    or base.value < 0 or ty.dtype.name != "f32"
                    or shape != (64,) or tuple(layout.shape) != (64,)
                    or tuple(layout.strides) != (1,)):
                raise SourceAuditError("SOURCE_UB_VIEW_UNSUPPORTED",
                                       "UB view is not a static contiguous 64-f32 window")
            start = base.value
            end = start + 64 * 4
            for _, left, right in windows:
                if start < right and left < end and (start, end) != (left, right):
                    raise SourceAuditError("SOURCE_UB_ALIAS_UNSUPPORTED",
                                           "partially overlapping UB windows")
            windows.append((node.var, start, end))
        elif isinstance(node, Evaluate):
            return
        else:
            raise SourceAuditError("SOURCE_CONTROL_UNSUPPORTED",
                                   "only straight-line function regions are supported")
    visit(entry.body)
    return tuple(windows)

def audit_single_region_reaching_definitions(problem):
    """The RAW relations the actual TIR establishes, as a frozen set."""
    return frozenset(_reaching_definition_evidence(problem)["edges"])


def reaching_definition_evidence(problem):
    """The structured evidence behind those relations (see the walk below)."""
    evidence = _reaching_definition_evidence(problem)
    return {name: tuple(value) for name, value in evidence.items()}


def _reaching_definition_evidence(problem):
    """Re-read the straight-line TIR and check operand reaching definitions.

    This restricted analysis is evidence only: it does not certify scheduling
    delays, completion or the plan. Unknown shapes, views and control are refused.
    """
    from tilefoundry.ir.core import Call, Var
    from tilefoundry.ir.tir.memory.alloc_tensor import AllocTensor
    from tilefoundry.ir.tir.memory.tensor_view import TensorView
    from tilefoundry.ir.tir.stmts import LetStmt, MeshScope
    from tilefoundry_ascend.ir.ops import Pset, Vadd

    entry = problem.source_snapshot.entry_ref
    if len(problem.regions) != 1:
        raise SourceAuditError("SOURCE_REGION_UNSUPPORTED", "requires one straight-line region")
    region = problem.regions[0]
    audit_instruction_effects(region)
    windows = read_static_ub_windows(entry)
    by_var = {id(var): (start, end) for var, start, end in windows}
    accesses = {(id(a.source_ref), a.operand_index): a for a in region.accesses}
    effect_by_stmt = {id(e.source_ref): e for e in region.effects}
    task_by_stmt = {}
    for task in region.tasks:
        for statement in _instructions(task.unit_ref.node_ref):
            task_by_stmt[id(statement)] = task.task_id
    inputs = entry.params[:-entry.output_count]
    values = {id(var): ("param", id(var)) for var in entry.params}
    written = {("param", id(var)): ("ENTRY", None) for var in inputs}
    first_root = {("param", id(var)): var for var in entry.params}
    required = set()
    reads = []
    all_true_masks = set()
    def visit(node):
        if isinstance(node, Sequential):
            for child in node.body: visit(child)
        elif isinstance(node, (MeshScope, ExecutionScope)):
            visit(node.body)
        elif isinstance(node, LetStmt):
            value = node.value
            if not isinstance(value, Call):
                raise SourceAuditError("SOURCE_DEFINITION_UNSUPPORTED", "non-allocation let")
            if isinstance(value.target, AllocTensor):
                key = ("alloc", id(node.var))
                values[id(node.var)] = key
                first_root[key] = node.var
            elif isinstance(value.target, TensorView) and id(node.var) in by_var:
                bounds = by_var[id(node.var)]
                key = ("ub", *bounds)
                values[id(node.var)] = key
                first_root.setdefault(key, node.var)
            else:
                raise SourceAuditError("SOURCE_DEFINITION_UNSUPPORTED", "unknown tensor view")
        elif isinstance(node, Evaluate):
            schema = type(node.callable)._op_schema
            params = tuple(p for p in schema.signature if p.kind == "input")
            task = task_by_stmt.get(id(node))
            if task is None:
                raise SourceAuditError("SOURCE_TASK_COVERAGE", "instruction has no task")
            for i, (param, operand) in enumerate(zip(params, node.args, strict=True)):
                if not isinstance(operand, Var) or id(operand) not in values:
                    raise SourceAuditError("SOURCE_OPERAND_UNBOUND", "operand has no real binder")
                key = values[id(operand)]
                access = accesses[(id(node), i)]
                if access.root_ref is not first_root[key]:
                    raise SourceAuditError("SOURCE_ACCESS_ROOT_MISMATCH", "one storage interval has distinct roots")
                read = bool(param.effect & MemoryEffect.READ)
                write = bool(param.effect & MemoryEffect.WRITE)
                if isinstance(node.callable, Vadd) and i == 3:
                    if values[id(node.args[2])] not in all_true_masks:
                        raise SourceAuditError("SOURCE_MASK_UNPROVEN", "Vadd mask is not known all-active")
                    read = False
                if read:
                    if key not in written:
                        raise SourceAuditError("SOURCE_READ_UNINITIALIZED", "local read has no dominating write")
                    producer, producer_effect = written[key]
                    if producer not in ("ENTRY", task):
                        required.add((producer, task, "RAW", producer_effect, effect_by_stmt[id(node)].effect_id))
                    reads.append(ReadEvidence(
                        access_id=access.access_id, task_id=task,
                        operand_index=i, buffer_root=access.root_ref.name,
                        producer=producer,
                        producer_effect=("ENTRY" if producer_effect is None
                                         else producer_effect),
                        effect_id=effect_by_stmt[id(node)].effect_id,
                        from_entry=producer == "ENTRY"))
                if write:
                    # Any write invalidates a prior all-active predicate proof.
                    # A recognized Pset may establish it again below.
                    all_true_masks.discard(key)
                    if key in written and written[key][0] != task:
                        raise SourceAuditError("SOURCE_OVERWRITE_UNSUPPORTED", "multiple writers require WAR/WAW proof")
                    if key[0] == "param" and any(operand is x for x in inputs):
                        raise SourceAuditError("SOURCE_INPUT_WRITE_UNSUPPORTED", "input parameter is written")
                    written[key] = (task, effect_by_stmt[id(node)].effect_id)
                    if isinstance(node.callable, Pset):
                        if node.callable.elem_width != 32:
                            raise SourceAuditError("SOURCE_MASK_UNPROVEN", "unsupported predicate width")
                        # The registered Pset evaluator and ISA contract both set
                        # all 64 predicate lanes for this exact instruction.
                        all_true_masks.add(key)
        else:
            raise SourceAuditError("SOURCE_CONTROL_UNSUPPORTED", "unsupported control")
    visit(entry.body)
    return {"edges": sorted(required), "reads": reads,
            "writes": [WriteEvidence(buffer_root=access.root_ref.name,
                                    access_id=access.access_id,
                                    task_id=access.task_id,
                                    effect_id=effect_by_stmt[id(access.source_ref)].effect_id)
                       for access in region.accesses
                       if access.mode in ("WRITE", "READ_WRITE")],
            "windows": windows}

@dataclass(frozen=True)
class EdgeProof:
    """One dependence as the independent reader established it."""

    producer: str
    consumer: str
    kind: str
    distance: int
    delay: int
    delay_state: str
    delay_basis: str
    delay_unit: str
    delay_source: object
    effect_ids: tuple

    def canonical(self) -> dict:
        source = self.delay_source
        return {"producer": self.producer, "consumer": self.consumer,
                "kind": self.kind, "distance": int(self.distance),
                "delay": int(self.delay), "delay_state": self.delay_state,
                "delay_basis": self.delay_basis, "delay_unit": self.delay_unit,
                "delay_source": (source.canonical() if hasattr(source, "canonical")
                                 else {"#": "missing"}),
                "effect_ids": sorted(self.effect_ids)}


@dataclass(frozen=True)
class BufferProof:
    """One storage root as the actual IR stated it: window, writers, readers."""

    buffer_id: str
    space: str
    root: str
    window: tuple
    writers: tuple
    readers: tuple

    def canonical(self) -> dict:
        return {"buffer_id": self.buffer_id, "space": self.space, "root": self.root,
                "window": [int(value) for value in self.window],
                "writers": sorted(self.writers), "readers": sorted(self.readers)}


@dataclass(frozen=True)
class DependenceProof:
    """What the independent reader could prove, and what it could not.

    One rule, stated once: defects() lists every way this record fails to be a
    proof, and satisfied_by_content()/complete() are "that list is empty". A
    refusal therefore always names a reason, and a reason can never exist
    without the boolean seeing it. Completeness is a property of the CONTENT:
    an empty record, a relabelled field or a vacuous locator is defective no
    matter what any flag says.
    """

    region_id: str
    straight_line: bool
    #: The universe the REGION declared, read from the problem the reader read,
    #: never derived from this record's own members. Without it, relabelling a
    #: write onto a task that does not exist would stay self-consistent: a set
    #: built from its own members always contains them. Both directions are
    #: checked - nothing outside may be cited, and nothing declared may be
    #: omitted. These ids are hashed, so a forged universe moves identity.
    edges: tuple
    reads: tuple
    writes: tuple
    buffers: tuple
    reuse: tuple
    missing: tuple
    region_tasks: tuple = ()
    region_effects: tuple = ()
    #: True for a record dependence_proof() built from the actual IR. This is
    #: AUXILIARY information only: complete() never reads it.
    derived: bool = False

    def defects(self) -> tuple:
        """Every way this record fails to prove the region's dependences."""
        notes = list(self.missing)
        if not self.reads:
            notes.append("region %s: no operand read was re-read from the actual "
                         "TIR, so nothing about its sources is established"
                         % self.region_id)
        if not self.writes:
            notes.append("region %s: no covering write was re-read from the actual "
                         "TIR" % self.region_id)
        if not self.buffers:
            notes.append("region %s: no storage root was described, so reads and "
                         "writes have nothing to point at" % self.region_id)
        if not self.region_tasks or not self.region_effects:
            notes.append("region %s: the proof carries no universe declared by the "
                         "region, so cited tasks and effects cannot be checked "
                         "against anything" % self.region_id)
        effect_ids = ({write.effect_id for write in self.writes}
                      | {read.effect_id for read in self.reads})
        tasks = ({write.task_id for write in self.writes}
                 | {read.task_id for read in self.reads})
        declared_tasks = set(self.region_tasks)
        declared_effects = set(self.region_effects)
        if declared_tasks:
            foreign = sorted(tasks - declared_tasks)
            if foreign:
                notes.append("the record cites tasks %s that the region does not "
                             "declare" % foreign)
            unseen = sorted(declared_tasks - tasks)
            if unseen:
                notes.append("region %s: tasks %s own no access this reader saw, so "
                             "the record does not cover the region"
                             % (self.region_id, unseen))
        if declared_effects:
            foreign_effects = sorted(effect_ids - declared_effects)
            if foreign_effects:
                notes.append("the record cites effects %s the region does not "
                             "declare" % foreign_effects)
        roots = {buffer.root for buffer in self.buffers}
        for edge in self.edges:
            where = "edge %s -> %s (%s)" % (edge.producer, edge.consumer, edge.kind)
            if edge.producer not in tasks or edge.consumer not in tasks:
                notes.append("%s names a task no read or write of this record saw"
                             % where)
            if edge.kind not in ("RAW", "WAR", "WAW"):
                notes.append("%s has an undeclared kind %r" % (where, edge.kind))
            if int(edge.distance) != 0:
                notes.append("%s states distance %d, which this proof cannot derive"
                             % (where, int(edge.distance)))
            if int(edge.delay) < 0:
                notes.append("%s states no resolvable delay" % where)
            if not set(edge.effect_ids) <= effect_ids:
                notes.append("%s references effects %s that are not among this "
                             "record's own accesses"
                             % (where, sorted(set(edge.effect_ids) - effect_ids)))
            if edge.delay_basis in ("UNSOURCED", "MISSING"):
                notes.append("%s has an unsourced delay basis" % where)
            if not edge.delay_unit:
                notes.append("%s states no delay unit" % where)
            if not verify_model_source(edge.delay_source):
                notes.append("%s carries a locator %r that resolves to no "
                             "registered delay row"
                             % (where, getattr(edge.delay_source, "symbol",
                                               edge.delay_source)))
        for read in self.reads:
            if read.buffer_root not in roots:
                notes.append("read %s points at root %r which this record does not "
                             "list" % (read.access_id, read.buffer_root))
            if read.producer != "ENTRY" and read.producer not in tasks:
                notes.append("read %s claims a reaching producer %r that is not a "
                             "task of this region"
                             % (read.access_id, read.producer))
        for write in self.writes:
            if write.buffer_root not in roots:
                notes.append("write %s points at root %r this record does not list"
                             % (write.access_id, write.buffer_root))
            if write.task_id not in tasks:
                notes.append("write %s names %r outside this region's tasks"
                             % (write.access_id, write.task_id))
        writers_per_root = {}
        for write in self.writes:
            writers_per_root.setdefault(write.buffer_root, []).append(write.access_id)
        owed = sum(max(0, len(items) - 1) for items in writers_per_root.values())
        if len(self.reuse) != owed:
            notes.append("%d reuse obligation(s) stated while the writes above "
                         "justify %d: a slot recycle must be derived from a second "
                         "covering write of the same root" % (len(self.reuse), owed))
        return tuple(notes)

    def satisfied_by_content(self) -> bool:
        """True exactly when defects() is empty: one rule, not two."""
        return not self.defects()

    def complete(self) -> bool:
        """Completeness is a property of the content, never of a flag."""
        return self.satisfied_by_content()

    def canonical(self) -> dict:
        import json

        return json.loads(json.dumps({
            "region": self.region_id,
            "straight_line": bool(self.straight_line),
            "region_tasks": sorted(self.region_tasks),
            "region_effects": sorted(self.region_effects),
            "edges": [edge.canonical() for edge in self.edges],
            "reads": [read.canonical() for read in self.reads],
            "writes": [write.canonical() for write in self.writes],
            "buffers": [buffer.canonical() for buffer in self.buffers],
            "reuse": [str(item) for item in self.reuse],
            "missing": [str(item) for item in self.missing],
        }, sort_keys=True))


def _straight_line_domain(region):
    """(provable, reason): one execution domain means constant bounds 0/1/1."""
    from tilefoundry.ir.core import Constant

    if region.region_ref.induction_var_ref is not None:
        return False, ("region %r is a loop: a carried relation needs a distance "
                      "derived from a generation delta, and this slice derives "
                      "none" % (region.region_ref.region_id,))
    bounds = []
    for name in ("lower_expr", "stop_expr", "step_expr"):
        value = getattr(region.region_ref, name, None)
        if not isinstance(value, Constant):
            return False, ("region %r bound %s is not a static constant"
                           % (region.region_ref.region_id, name))
        bounds.append(int(value.value))
    if bounds != [0, 1, 1]:
        return False, ("region %r executes %r iteration(s), not the single domain "
                      "this proof covers" % (region.region_ref.region_id, bounds))
    return True, ""


def _cross_region_missing(problem) -> tuple:
    """Name what a multi-region program cannot prove today.

    The refusal stays a refusal; the point is that it says WHICH region, WHICH
    access and WHICH fact is missing, so the gap is a location, not a wall.
    """
    entry = problem.source_snapshot.entry_ref
    inputs = {id(var) for var in entry.params[:-entry.output_count]}
    writes_by_root = {}
    writes_in_region = {}
    for region in problem.regions:
        region_id = region.region_ref.region_id
        writes_in_region[region_id] = set()
        for access in region.accesses:
            if access.mode in ("WRITE", "READ_WRITE"):
                writes_by_root.setdefault(id(access.root_ref), []).append(
                    (region_id, access.access_id))
                writes_in_region[region_id].add(id(access.root_ref))
    notes = []
    for region in problem.unit_tree.regions:
        if region.induction_var_ref is None:
            continue
        summary = region.binding_summary
        carried = tuple(getattr(summary, "carried_params", ()) or ()) if summary else ()
        notes.append(
            "region %r is a loop over [%s,%s): a carried relation needs a distance "
            "derived from a generation delta (carried: %s), and this slice derives "
            "none" % (region.region_id,
                      getattr(region.lower_expr, "value", "?"),
                      getattr(region.stop_expr, "value", "?"),
                      [getattr(var, "name", "?") for var in carried]))
    for region in problem.regions:
        region_id = region.region_ref.region_id
        for access in region.accesses:
            if access.mode not in ("READ", "READ_WRITE"):
                continue
            key = id(access.root_ref)
            if key in writes_in_region[region_id] or key in inputs:
                continue
            producers = sorted({other for other, _ in writes_by_root.get(key, ())})
            if producers:
                notes.append(
                    "access %r in region %r reads root %r, which no statement of "
                    "that region writes; it is produced across a region boundary "
                    "by %s, and no boundary/composite contract proves that flow"
                    % (access.access_id, region_id, access.root_ref.name, producers))
    return tuple(notes)


def dependence_proof(problem):
    """The structured dependence evidence for one straight-line region.

    Structural tampering (deleted/reversed edge, wrong root, wrong effect set)
    still raises by name, exactly as before. What this function no longer does
    is refuse a program whose facts ARE proven: it returns a DependenceProof
    whose missing list says precisely what is not proven, if anything.

    A program that spans more regions than this slice can pair is answered the
    same way: an incomplete proof that names each region, access and boundary.
    """
    if len(problem.regions) != 1 or len(problem.unit_tree.regions) != 1:
        # Never vacuously complete: this reader proves one straight-line region
        # by re-reading its TIR, so a program of several regions is reported as
        # unproven even when it happens to contain no loop and no cross-region
        # read. Completeness must come from evidence, not from the absence of a
        # named defect.
        return DependenceProof(
            region_id=",".join(region.region_ref.region_id
                               for region in problem.regions),
            straight_line=False, edges=(), reads=(), writes=(), buffers=(),
            reuse=tuple(item for region in problem.regions for item in region.reuse),
            missing=_cross_region_missing(problem) + (
                "this reader proves one straight-line region by re-reading its "
                "actual TIR; the program has %d problem region(s) and %d tree "
                "region(s), so no region's evidence was re-read here"
                % (len(problem.regions), len(problem.unit_tree.regions)),),
            # the declared universe is still recorded: the refusal below is
            # checked against the regions, not against this empty record
            region_tasks=tuple(sorted(task.task_id
                                      for region in problem.regions
                                      for task in region.tasks)),
            region_effects=tuple(sorted(effect.effect_id
                                        for region in problem.regions
                                        for effect in region.effects)),
            derived=True)
    evidence = _reaching_definition_evidence(problem)
    region = problem.regions[0]
    expected = {}
    for producer, consumer, kind, writer, reader in evidence["edges"]:
        expected.setdefault((producer, consumer, kind), set()).update((writer, reader))
    actual = {}
    by_key = {}
    for edge in region.edges:
        key = (edge.producer, edge.consumer, edge.kind)
        if key in actual:
            raise SourceAuditError("SOURCE_DEPENDENCE_DUPLICATE", "duplicate dependency")
        if int(edge.distance) != 0:
            raise SourceAuditError("SOURCE_DEPENDENCE_DISTANCE",
                                   "single execution domain requires distance zero")
        if edge.delay is not None and int(edge.delay) < 0:
            raise SourceAuditError("SOURCE_DEPENDENCE_DELAY_INVALID",
                                   "completion delay cannot be negative")
        actual[key] = set(edge.effect_ids)
        by_key[key] = edge
    if expected != actual:
        raise SourceAuditError("SOURCE_DEPENDENCE_MISMATCH",
                               "edge direction or source effect coverage differs "
                               "from actual TIR")

    missing = []
    straight_line, reason = _straight_line_domain(region)
    if not straight_line:
        missing.append(reason)

    for key in sorted(by_key, key=lambda item: (item[0], item[1], item[2])):
        edge = by_key[key]
        fact = getattr(edge, "delay_fact", None)
        if fact is None or getattr(fact, "state", "") == "UNSOURCED":
            missing.append("edge %s -> %s (%s) has no sourced delay" % key)
        else:
            try:
                edge.resolved_delay()
            except Exception as error:
                missing.append("edge %s -> %s (%s) refuses its delay: %s"
                               % (key[0], key[1], key[2], error))

    # Storage identity. Two binders naming the SAME window are one storage
    # object (the lowering re-materialises the view); that is normalisation, not
    # aliasing. What must be proven is the other direction: one root reached
    # through two different windows. Partial overlap is refused by the window
    # reader itself, and an unknown window is left empty rather than filled 0.
    window_of_binder = {}
    for var, start, stop in evidence["windows"]:
        window_of_binder[var.name] = (start, stop)
    windows_by_root = {}
    for access in region.accesses:
        name = access.view_ref.name if access.view_ref is not None else None
        window = window_of_binder.get(name)
        if window is not None:
            windows_by_root.setdefault(access.root_ref.name, set()).add(window)
    for root, items in sorted(windows_by_root.items()):
        if len(items) > 1:
            missing.append("root %r is reached through %d distinct storage windows "
                           "%s; no alias/containment proof exists for that"
                           % (root, len(items), sorted(items)))
    # Reuse must be DERIVED, not merely absent: a slot can only be recycled by a
    # second covering write of the same root, so counting writers per root tells
    # whether an obligation is owed. A stated obligation with no second write is
    # as unacceptable as a second write with no stated obligation.
    writers = {}
    for write in evidence["writes"]:
        writers.setdefault(write.buffer_root, []).append(write.access_id)
    readers = {}
    for read in evidence["reads"]:
        readers.setdefault(read.buffer_root, []).append(read.access_id)
    buffer_proofs = tuple(
        BufferProof(buffer_id=buffer.buffer_id, space=str(buffer.space),
                    root=buffer.root_ref.name,
                    window=(sorted(windows_by_root[buffer.root_ref.name])[0]
                            if len(windows_by_root.get(buffer.root_ref.name, ())) == 1
                            else ()),
                    writers=tuple(sorted(writers.get(buffer.root_ref.name, ()))),
                    readers=tuple(sorted(readers.get(buffer.root_ref.name, ()))))
        for buffer in region.buffers)
    owed = {root: len(items) - 1 for root, items in writers.items() if len(items) > 1}
    stated = tuple(region.reuse)
    if owed:
        missing.append("root(s) %s are covered %d time(s) but the region states %d "
                       "reuse obligation(s); a slot recycle needs a proven "
                       "last-reader/next-writer pair"
                       % (sorted(owed), max(len(items) for items in writers.values()),
                          len(stated)))
    if stated and not owed:
        missing.append("the region states %d reuse obligation(s) while every root "
                       "has a single covering write: the pairing is not derivable "
                       "from the actual accesses" % (len(stated),))
    if not stated and not owed:
        # Proven empty: this is a derivation, not an accident of the data.
        pass

    edges = tuple(
        EdgeProof(producer=key[0], consumer=key[1], kind=key[2],
                  distance=int(edge.distance),
                  delay=(int(edge.delay) if edge.delay is not None else -1),
                  delay_state=str(getattr(edge.delay_fact, "state", "MISSING")),
                  delay_basis=str(getattr(edge.delay_fact, "basis", "MISSING")),
                  delay_unit=str(getattr(edge.delay_fact, "unit", "MISSING")),
                  # the locator passes through as structure: str() of a record
                  # would put a repr into the identity bytes
                  delay_source=getattr(edge.delay_fact, "source", None),
                  effect_ids=tuple(sorted(edge.effect_ids)))
        for key, edge in sorted(by_key.items(), key=lambda item: item[0]))
    reads = tuple(sorted(evidence["reads"], key=lambda item: item.access_id))
    writes = tuple(sorted(evidence["writes"], key=lambda item: item.access_id))
    return DependenceProof(
        region_id=region.region_ref.region_id,
        straight_line=straight_line,
        # the universe the region itself declared, so coverage and foreign
        # citation are checked against the problem and not against this record
        region_tasks=tuple(sorted(task.task_id for task in region.tasks)),
        region_effects=tuple(sorted(effect.effect_id for effect in region.effects)),
        edges=edges,
        reads=reads,
        writes=writes,
        buffers=buffer_proofs,
        reuse=stated,
        missing=tuple(missing),
        derived=True)


def audit_single_region_dependencies(problem):
    """Keep the historic entry point: return the proof, or refuse by name.

    The refusal is now CONDITIONAL on what could not be proven, and it names
    each missing item. A program whose evidence is complete is no longer
    refused here - an unconditional refusal is a dead end, not fail-closed.
    """
    proof = dependence_proof(problem)
    if not proof.complete():
        raise SourceAuditError(
            "SOURCE_DEPENDENCE_INCOMPLETE",
            "the dependence proof is not complete for region %r: %s"
            % (proof.region_id, "; ".join(proof.missing)))
    return proof

