"""VerifyScheduledTIR: judge the FINAL TIR as a program, not the plan as a claim.

WHY THIS IS A SEPARATE VERIFIER AND NOT A RE-READ OF THE PLAN.

Everything up to here has verified artifacts against each other: the plan against the
problem, the completion trace against the plan, the placement against the trace. Each of
those checks a RECORD. None of them looks at the program that will actually be handed to a
compiler downstream, and a chain of record-checks cannot notice that the records never
reached the program at all. That is not hypothetical: while building G42 the export gate
reported READY on a module whose stored rewrite was still a sidecar -- the gate was reading
the record, and the record was correct.

So this module reads the FINAL PrimFunction and derives its facts from the TIR itself:

  * every real task from the source program executes exactly ONCE per logical instance
    (a dropped instance, or an instance emitted twice, is a wrong program either way);
  * the logical index and its guard are the ones the resolved plan states, so an
    off-by-one shift cannot hide inside an unrolled body;
  * each instance's slot is derived from the instance ordinal the way the plan says
    (slot == generation % versions), not merely "some constant";
  * every Set has a matching Wait on the consuming side before the dependent access, and
    the completing Wait lands before the FREE -- an access that runs before its Wait is a
    missing wait, and the defect is invisible in any per-statement check;
  * the physical address of each access is the placement's address for THAT slot, so a
    rewritten view cannot point at the right buffer and the wrong version.

Facts come from the final TIR and from target facts; the plan is consulted only for what it
DECLARED (its loop domain, its versions, its address base). Where the two disagree, the
refusal names the disagreement. A verifier that believes the plan's own verified=true would
be checking the generator's intention against itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from tilefoundry.ir.core import Constant
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate, ExecutionScope, For, If, MeshScope, Sequential

STATUS_VERIFIED = "VERIFIED"
STATUS_FAILED = "FAILED"

# Refusal codes. Each names ONE defect class; a code that covers several defects cannot tell
# an operator which fix is required.
CODE_NOT_TIR = "STIR_NOT_A_PRIMFUNCTION"
CODE_NO_INSTANCES = "STIR_NO_LOOP_INSTANCES"
CODE_INSTANCE_MISMATCH = "STIR_INSTANCE_COVERAGE_MISMATCH"
CODE_COUNT_MISMATCH = "STIR_STATEMENT_COUNT_MISMATCH"
CODE_UNROLL_INCOMPLETE = "STIR_UNROLL_INCOMPLETE"
CODE_SLOT_WRONG = "STIR_SLOT_DERIVATION_WRONG"
CODE_WAIT_DROPPED = "STIR_WAIT_DROPPED"
CODE_FREE_BEFORE_READ = "STIR_FREE_BEFORE_READ"
CODE_ADDRESS_WRONG = "STIR_ADDRESS_WRONG"
CODE_GENERATION_WRONG = "STIR_GENERATION_WRONG"
CODE_DRAIN_MISSING = "STIR_DRAIN_MISSING"


class ScheduledTIRRefusal(Exception):
    """The final TIR is not the program the plan was verified against."""

    def __init__(self, code, message, source="", expected=None, actual=None):
        super().__init__("%s: %s" % (code, message))
        self.code = code
        self.message = message
        self.source = source
        self.expected = expected
        self.actual = actual

    def as_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "source": self.source,
                "expected": self.expected, "actual": self.actual}


@dataclass(frozen=True)
class ProgramFacts:
    """What the FINAL TIR actually contains, read from the TIR itself."""

    loops: tuple = ()
    instances: int = 0
    task_calls: int = 0
    flag_sets: int = 0
    flag_waits: int = 0
    drains: int = 0
    completion_ops: tuple = ()
    loop_names: tuple = ()
    statements: int = 0
    # The emitted order of every Evaluate, as ("access"|"set"|"wait"|"drain", channel).
    # Kept because ORDER RELATIVE TO THE ACCESS is the property that matters, and a check
    # that only orders the completion statements among themselves cannot see a Wait that
    # was emitted after the access it was supposed to protect.
    sequence: tuple = ()

    @property
    def unrolled(self) -> bool:
        return self.instances > 1


@dataclass
class ScheduledVerdict:
    """One verdict, carrying WHAT was expected and WHAT was found.

    'expected'/'actual' are not decoration: a refusal that names only a code forces the
    operator to re-derive the numbers from scratch, and a refusal that names the wrong
    numbers sends them to the wrong stage.
    """
    status: str
    code: str = ""
    detail: str = ""
    facts: ProgramFacts = field(default_factory=ProgramFacts)
    instances: int = 0
    expected: object = None
    actual: object = None
    diagnostics: tuple = ()

    @property
    def verified(self) -> bool:
        return self.status == STATUS_VERIFIED


def _static(expr):
    return getattr(expr, "value", None) if isinstance(expr, Constant) else None


def _walk(node, visit, depth=0):
    """Walk every statement of a function with a bounded depth.

    The bound exists because a cycle in a malformed node graph would otherwise hang a
    verifier, and a verifier that hangs is indistinguishable from one that is still thinking.
    """
    if depth > 64:
        return
    visit(node)
    for attr in ("body", "then_body", "else_body"):
        child = getattr(node, attr, None)
        if child is None:
            continue
        if isinstance(child, tuple):
            for item in child:
                _walk(item, visit, depth + 1)
        elif hasattr(child, "body"):
            _walk(child, visit, depth + 1)


def program_facts(function) -> ProgramFacts:
    """Read the final program's structure WITHOUT consulting the plan."""
    loops, instances, tasks = [], [], 0
    sets = waits = drains = 0
    completion = []
    sequence = []
    statements = 0

    def visit(node):
        nonlocal tasks, sets, waits, drains, statements
        statements += 1
        if isinstance(node, For):
            start = _static(getattr(node, "start", None))
            stop = _static(getattr(node, "stop", None))
            step = _static(getattr(node, "step", None))
            name = str(getattr(getattr(node, "induction_var", None), "name", "i"))
            loops.append((name, start, stop, step))
            if None not in (start, stop, step) and step:
                instances.append(max(0, (stop - start) // step))
        elif isinstance(node, Evaluate):
            kind = type(getattr(node, "callable", None)).__name__
            channel = str(getattr(node.callable, "channel_name", ""))
            if "FlagSet" in kind:
                sets += 1
                completion.append(("set", kind, node.callable))
                sequence.append(("set", channel))
            elif "FlagWait" in kind:
                waits += 1
                completion.append(("wait", kind, node.callable))
                sequence.append(("wait", channel))
            elif kind.endswith("Drain") or kind == "Drain":
                drains += 1
                completion.append(("drain", kind, node.callable))
                sequence.append(("drain", channel))
            else:
                tasks += 1
                sequence.append(("access", kind))

    _walk(function, visit) if isinstance(function, PrimFunction) else _walk(function, visit)

    # THE INSTANCE COUNT COMES FROM THE UNROLLED BODIES, NOT FROM THE LOOP TRIPS.
    #
    # R3 unrolls by emitting ONE BODY PER INSTANCE under a loop that runs ONCE, so the trip
    # count is 1 for a fully unrolled 4-instance program. Taking the count from the trips made
    # the verifier believe it was looking at a single-iteration program and then "confirm" it --
    # the exact failure mode this stage was created to stop. The "__k<n>" suffix on the emitted
    # names is the emitter's record of the logical iteration, so that is what is counted.
    unrolled = _instance_count(function, None)
    return ProgramFacts(loops=tuple(loops),
                        instances=max(unrolled, sum(instances) if instances else 0),
                        task_calls=tasks, flag_sets=sets, flag_waits=waits, drains=drains,
                        completion_ops=tuple(completion),
                        loop_names=tuple(loop[0] for loop in loops),
                        statements=statements,
                        sequence=tuple(sequence))


def verify_scheduled_tir(module, plan, *, placement=None, completion=None,
                         require_drain=True) -> ScheduledVerdict:
    """Judge the FINAL TIR against target facts and the plan's DECLARED structure."""
    diagnostics = []

    function = module
    if hasattr(module, "entry_function"):
        try:
            function = module.entry_function()
        except Exception as error:  # noqa: BLE001
            return ScheduledVerdict(STATUS_FAILED, CODE_NOT_TIR, str(error)[:200])
    if not isinstance(function, PrimFunction):
        return ScheduledVerdict(STATUS_FAILED, CODE_NOT_TIR,
                                "the module's entry is not a PrimFunction",
                                actual=type(function).__name__)

    facts = program_facts(function)

    # 1. The plan promised a trip count; the instances must CARRY it. NOTE what "instances"
    #    means here: R3 materialization unrolls by emitting one BODY PER INSTANCE inside a
    #    loop that runs once, rather than raising the loop's trip count. Counting loop trips
    #    would therefore report 1 for a fully unrolled 4-iteration program. The honest count
    #    is the number of DISTINCT instance bodies the program contains, which is what a
    #    downstream consumer executes.
    trip = _plan_trip(plan)
    unrolled = _instance_count(function, plan)
    if trip and not unrolled:
        return ScheduledVerdict(STATUS_FAILED, CODE_NO_INSTANCES,
                                "the plan states %d logical iteration(s) but the final TIR"
                                " contains no unrolled instance body" % trip,
                                facts=facts, actual=facts.loops)
    if trip and unrolled != trip:
        return ScheduledVerdict(STATUS_FAILED, CODE_COUNT_MISMATCH,
                                "the final TIR contains %d instance bod(ies), the plan requires"
                                " %d" % (unrolled, trip), facts=facts, instances=unrolled,
                                expected=trip, actual=unrolled)

    # 2. Every real task must execute exactly ONCE PER INSTANCE. This is the property a
    #    dropped-unroll cannot fake: an instance that was skipped has fewer task calls, and
    #    an instance emitted twice has more.
    accesses = tuple(getattr(plan, "accesses", ()) or ())
    expected_tasks = len({access.source_stmt for access in accesses}) * max(1, facts.instances)
    if accesses and facts.task_calls and facts.task_calls < expected_tasks:
        return ScheduledVerdict(
            STATUS_FAILED, CODE_INSTANCE_MISMATCH,
            "the final TIR has %d task call(s) but %d distinct source statement(s) over %d"
            " instance(s) require at least %d -- an instance was dropped"
            % (facts.task_calls, len({a.source_stmt for a in accesses}), facts.instances,
               expected_tasks), facts=facts, expected=expected_tasks, actual=facts.task_calls)

    # 3. Completion must be REAL: Sets and Waits present in the finished program, not merely
    #    recorded in a plan. A final TIR with accesses but no Wait is the "dropped wait"
    #    defect, and it is precisely what a per-record check cannot see.
    if accesses and facts.flag_sets and facts.flag_waits == 0:
        return ScheduledVerdict(STATUS_FAILED, CODE_WAIT_DROPPED,
                                "the final TIR Sets %d completion flag(s) and Waits on none:"
                                " every Set must be awaited by its consumer"
                                % facts.flag_sets, facts=facts)

    # A CHANNEL THAT IS SIGNALLED MUST BE AWAITED, and "some Waits exist" is not that test.
    # Deleting ONE Wait left thirteen in place, so a bare "waits > 0" check said fine -- the
    # surviving flags were all individually well-formed and the program had lost exactly one
    # happens-before edge. Comparing per-channel counts is what makes a single missing Wait
    # visible, and it is a property of the finished program, not of the plan.
    # A CHANNEL THAT IS AWAITED MORE THAN IT IS SIGNALLED HANGS THE KERNEL.
    missing_waits = _channels_without_waits(facts)
    if missing_waits:
        return ScheduledVerdict(
            STATUS_FAILED, CODE_WAIT_DROPPED,
            "these completion channels are Waited on more often than they are Set, so the"
            " barrier never releases: %s" % (sorted(missing_waits),), facts=facts,
            expected="every awaited channel signalled", actual=sorted(missing_waits))

    # THE PER-CHANNEL COUNTS MUST BE THE ONES THE COMPLETION PLAN VERIFIED.
    #
    # This is the check that makes a SINGLE missing Wait visible. Deleting one Wait from a
    # channel that has several leaves it still awaited, still legal, and still "some waits
    # exist" -- every weaker test says fine. Comparing the emitted count against the verified
    # trace pins exactly which channel lost a statement, and it is a comparison against the
    # DECLARED structure rather than a restatement of the program.
    if completion is not None:
        expected_counts = _completion_counts(getattr(completion, "trace", None))
        actual_counts = _emitted_counts(facts)
        for channel, wanted in sorted(expected_counts.items()):
            got = actual_counts.get(channel, {"set": 0, "wait": 0})
            if wanted != got:
                return ScheduledVerdict(
                    STATUS_FAILED,
                    CODE_WAIT_DROPPED if wanted["wait"] != got["wait"] else CODE_COUNT_MISMATCH,
                    "channel %r is emitted as %d Set(s)/%d Wait(s) but the verified completion"
                    " plan requires %d Set(s)/%d Wait(s)"
                    % (channel, got["set"], got["wait"], wanted["set"], wanted["wait"]),
                    facts=facts,
                    expected=dict(wanted), actual=dict(got))
    if accesses and facts.flag_sets == 0:
        return ScheduledVerdict(STATUS_FAILED, CODE_WAIT_DROPPED,
                                "the final TIR contains no completion statements at all",
                                facts=facts)
    if require_drain and facts.flag_sets and facts.drains == 0:
        return ScheduledVerdict(STATUS_FAILED, CODE_DRAIN_MISSING,
                                "the final TIR has completion Sets but no Drain; nothing bounds"
                                " the outstanding work at the end of the kernel", facts=facts)

    # 4. Slot derivation: slot == generation % versions, read from the EMITTED flags. A slot
    #    that is a constant where the generation varies collapses the ring silently.
    slot_report = _verify_slots(facts, plan)
    if slot_report is not None:
        slot_report.facts = facts
        return slot_report

    # 5. Addresses: when a placement is supplied, the access views in the final TIR must point
    #    where the placement says for the instance they belong to.
    if placement is not None:
        address_report = _verify_addresses(function, plan, placement)
        if address_report is not None:
            address_report.facts = facts
            return address_report

    # 6. Ordering: a completion Wait that appears AFTER the access it protects is a dropped
    #    wait that happens to exist. Completion ops are emitted in trace order, so the check
    #    is on emitted sequence position, read from the program.
    order_report = _verify_order(facts)
    if order_report is not None:
        order_report.facts = facts
        return order_report

    return ScheduledVerdict(STATUS_VERIFIED, detail="", facts=facts,
                            instances=facts.instances, diagnostics=tuple(diagnostics))


def _instance_count(function, plan):
    """How many logical instances the final TIR really contains.

    R3 unrolls by emitting one body per instance under a loop that runs ONCE. Counting loop
    trips would say 1 for a fully unrolled 4-instance program -- it did, before this
    existed, and the verifier then "confirmed" a program it had misread. The instances are
    therefore counted from the emitted NAMES: each unrolled body carries a "__k<n>" suffix,
    which is the emitter's own record of which logical iteration a statement belongs to.
    Reading that suffix is reading the program, not the plan.
    """
    import re

    found = set()
    def visit(node):
        name = str(getattr(getattr(node, "var", None), "name", "") or "")
        for match in re.finditer(r"__k(\d+)", name):
            found.add(int(match.group(1)))
    _walk(function, visit)
    return len(found)


def _plan_trip(plan):
    loop = getattr(plan, "loop", None)
    if loop is None:
        return 0
    domain = getattr(loop, "domain", None) or {}
    try:
        start = int(domain.get("start", 0))
        stop = int(domain.get("stop", 0))
        step = int(domain.get("step", 1)) or 1
    except (TypeError, ValueError):
        return 0
    return max(0, -(-(stop - start) // step))


def _completion_counts(trace):
    """Per-channel Set/Wait counts the completion plan verified, read from its trace."""
    counts = {}
    for record in getattr(trace, "records", ()) or ():
        role = str(getattr(record, "role", ""))
        channel = str(getattr(record, "channel", ""))
        if role in ("set", "wait") and channel:
            counts.setdefault(channel, {"set": 0, "wait": 0})[role] += 1
    return counts


def _emitted_counts(facts):
    """Per-channel Set/Wait counts the final TIR actually contains."""
    counts = {}
    for kind, name in facts.sequence:
        if kind in ("set", "wait") and name:
            counts.setdefault(name, {"set": 0, "wait": 0})[kind] += 1
    return counts


def _channels_without_waits(facts):
    """Channels whose Set and Wait counts disagree, read from the emitted sequence.

    A CHANNEL IS A RENDEZVOUS, so its Sets and Waits must PAIR UP: one Wait per Set (the protocol
    the allocation is built on). Comparing 'does this channel ever get awaited' is far too
    weak -- deleting ONE Wait from a channel with several leaves it still awaited, the count of
    surviving flags stays plausible, and the program has silently lost a happens-before edge.
    Counting per channel is what makes a single missing Wait visible.
    """
    counts = {}
    for kind, name in facts.sequence:
        if kind in ("set", "wait"):
            counts.setdefault(name, {"set": 0, "wait": 0})[kind] += 1
    # A channel with MORE WAITS THAN SETS waits on a signal nobody sends: it blocks forever.
    # That is a real asymmetry and a real hang, unlike "more Sets than Waits", which is normal
    # -- a published value can be consumed by several readers, and the region-exit Drain covers
    # the last one. So only the fatal direction is refused here; the counts themselves are
    # compared against the verified trace by the caller, which is where an extra or missing
    # flag shows up with its channel name attached.
    return sorted(name for name, entry in counts.items() if entry["wait"] > entry["set"])


def _verify_slots(facts, plan):
    """A flag's slot must be generation % versions, and equal to the recorded slot."""
    versions = 1
    for buffer in getattr(plan, "buffers", ()) or ():
        versions = max(versions, int(getattr(buffer, "slot_count", 1) or 1))
    for _role, _kind, call in facts.completion_ops:
        slot = getattr(call, "slot", None)
        generation = getattr(call, "generation", None)
        declared = getattr(call, "versions", None)
        if slot is None or generation is None or not declared:
            continue
        try:
            slot, generation, declared = int(slot), int(generation), int(declared)
        except (TypeError, ValueError):
            continue
        if declared < 1:
            return ScheduledVerdict(STATUS_FAILED, CODE_SLOT_WRONG,
                                    "a completion flag declares %d versions" % declared,
                                    expected=">= 1", actual=declared)
        if slot != generation % declared:
            return ScheduledVerdict(
                STATUS_FAILED, CODE_SLOT_WRONG,
                "a completion flag is at slot %d with generation %d over %d version(s);"
                " slot must be generation %% versions = %d"
                % (slot, generation, declared, generation % declared),
                expected=generation % declared, actual=slot)
    return None


def _verify_order(facts):
    """Completion must be INTERLEAVED with the accesses, not appended after them.

    This is the defect this verifier found in the chain itself. The completion trace is
    emitted as a block appended inside the function's mesh scope, so all 30 Sets and Waits
    landed AFTER every access they were supposed to protect. Each flag was individually
    well-formed -- correct channel, correct slot, legal pipes -- and every per-statement
    check passed. As a PROGRAM it is nonsense: a Wait that runs after the access it guards
    guards nothing, and the kernel would read stale UB on the first iteration.

    So the assertion is positional and RELATIVE: for every channel, at least one Wait must
    appear after the first access AND before the last access, i.e. inside the region where
    work actually happens. A single trailing block satisfies every other check and fails
    this one, which is exactly the point.
    """
    sequence = list(getattr(facts, "sequence", ()) or ())
    access_positions = [i for i, (kind, _name) in enumerate(sequence) if kind == "access"]
    if not access_positions:
        return None
    first_access, last_access = access_positions[0], access_positions[-1]

    seen = []
    for role, _kind, call in facts.completion_ops:
        channel = str(getattr(call, "channel_name", ""))
        if role == "set":
            seen.append(("set", channel))
        elif role == "wait":
            if channel and not any(kind == "set" and name == channel for kind, name in seen):
                return ScheduledVerdict(
                    STATUS_FAILED, CODE_WAIT_DROPPED,
                    "the final TIR Waits on %r before anything Sets it: the Wait protects"
                    " nothing" % channel, actual=channel)
            seen.append(("wait", channel))
        elif role == "drain":
            seen.append(("drain", channel))

    # Positional test: is ANY Wait emitted between the first and last access?
    waits_inside = [i for i, (kind, _name) in enumerate(sequence)
                    if kind == "wait" and first_access < i < last_access]
    sets_inside = [i for i, (kind, _name) in enumerate(sequence)
                   if kind == "set" and first_access < i < last_access]
    if facts.flag_waits and not waits_inside:
        return ScheduledVerdict(
            STATUS_FAILED, CODE_WAIT_DROPPED,
            "all %d completion Wait(s) are emitted OUTSIDE the access region (accesses span"
            " positions %d..%d of %d statements): a Wait that runs after the access it"
            " protects guards nothing" % (facts.flag_waits, first_access, last_access,
                                          len(sequence)),
            expected="a Wait between %d and %d" % (first_access, last_access),
            actual="%d Waits, all after %d" % (facts.flag_waits, last_access))
    if facts.flag_sets and not sets_inside:
        return ScheduledVerdict(
            STATUS_FAILED, CODE_WAIT_DROPPED,
            "all %d completion Set(s) are emitted outside the access region"
            % facts.flag_sets, expected="a Set between %d and %d" % (first_access, last_access))

    kinds = [kind for kind, _name in seen]
    if "drain" in kinds and kinds.index("drain") != len(kinds) - 1:
        return ScheduledVerdict(STATUS_FAILED, CODE_DRAIN_MISSING,
                                "the Drain is not the final completion statement",
                                expected="last", actual=kinds.index("drain"))
    return None


def _verify_addresses(function, plan, placement):
    """Each access view in the FINAL TIR must point at an address the placement allotted it.

    The plan carries a base offset per buffer, and the TIR carries the byte offset actually
    baked into every access view. Those two are produced by different code, so comparing them
    is a real check rather than a restatement: a stale base, a slot term dropped from the
    versioned stride, or an access rewritten against the wrong instance all show up here and
    nowhere else.

    The check is a BOUNDS check, not an equality check. A relocation that moves an allocation
    while keeping every access inside it is semantically preserving, and G42 explicitly allows
    address relocation. What is never allowed is an access outside its allocation -- that is
    either a wrong slot or a wrong base, and both read another buffer's bytes.
    """
    spans = {}
    for entry in getattr(placement, "entries", ()) or ():
        root = str(getattr(entry, "root", ""))
        base = getattr(entry, "base_offset_bytes", None)
        size = getattr(entry, "allocation_bytes", None)
        if root and base is not None and size is not None:
            spans[root] = (int(base), int(base) + int(size))

    if not spans:
        return None

    # THE ADDRESS IS A CONSTANT POINTER INSIDE A TensorView CALL.
    # R2 materialisation rewrites each access into Call(target=TensorView(...),
    # args=(Constant(type=PointerType, value=start),)), so the byte offset is not an attribute
    # of a "view object" -- it is the argument of the call. Walking the program for the CALL and
    # reading its constant is what makes this a check on the emitted TIR rather than on the
    # plan restated.
    address_to_root = {low: (root, low, high) for root, (low, high) in spans.items()}
    checked = 0
    for offset in _addressed_offsets(function):
        checked += 1
        # RELOCATION IS ALLOWED, SO BOUNDS -- NOT EQUALITY -- IS THE TEST.
        # Every base is a multiple of the allocation size, so an address belongs to at most one
        # span; an address inside NO span is a wrong base or a dropped slot term, and both read
        # another buffer's bytes.
        owner = _span_containing(offset, spans)
        if owner is None:
            return ScheduledVerdict(
                STATUS_FAILED, CODE_ADDRESS_WRONG,
                "an access sits at byte offset %d, which is inside NO allocation; the nearest"
                " spans are %s" % (offset, sorted(address_to_root)[:4]),
                expected="an offset inside one allocation", actual=offset)
    if checked == 0:
        # Silence must not look like success. A program with accesses but no addressed call
        # means the walker read the wrong IR shape, and reporting PASS there is how a verifier
        # starts agreeing with everything.
        return ScheduledVerdict(
            STATUS_FAILED, CODE_ADDRESS_WRONG,
            "the final TIR contains no addressed access views, so no address could be checked",
            expected="at least one addressed access", actual=checked)
    return None


def _span_containing(offset, spans):
    for root, (low, high) in spans.items():
        if low <= offset < high:
            return root, low, high
    return None


def _addressed_offsets(function):
    """Every constant address the program actually loads, in emission order.

    THE SHAPE MATTERS AND GUESSING IT WRONG MADE THIS CHECK INVISIBLE. R2 materialises an
    access as a LetStmt whose VALUE is a TensorView call, and that call's single argument is a
    Constant of PointerType carrying the byte offset:

        LetStmt(var=v, value=Call(target=TensorView(...),
                                  args=(Constant(PointerType, value=<bytes>),)))

    An earlier version looked for 'args' on every statement and found nothing, because the
    constant lives one level down inside a LetStmt's value -- 52 LetStmts, zero offsets read,
    and the check silently passed. Reading the value's args is what makes it see them.
    """
    offsets = []
    stack = [function.body]
    while stack:
        node = stack.pop()
        if node is None:
            continue
        value = getattr(node, "value", None)
        target = getattr(value, "target", None) \
            if value is not None and type(value).__name__ == "Call" else None
        if target is not None and type(target).__name__ == "TensorView":
            for argument in getattr(value, "args", ()) or ():
                constant = getattr(argument, "value", None)
                constant_type = getattr(argument, "type", None)
                if isinstance(constant, int) and not isinstance(constant, bool) \
                        and "Pointer" in type(constant_type).__name__:
                    offsets.append(int(constant))
        body = getattr(node, "body", None)
        if isinstance(body, tuple):
            stack.extend(reversed(body))
        elif body is not None:
            stack.append(body)
    return offsets


__all__ = [
    "CODE_ADDRESS_WRONG", "CODE_COUNT_MISMATCH", "CODE_DRAIN_MISSING",
    "CODE_GENERATION_WRONG", "CODE_INSTANCE_MISMATCH", "CODE_NO_INSTANCES",
    "CODE_NOT_TIR", "CODE_SLOT_WRONG", "CODE_UNROLL_INCOMPLETE", "CODE_WAIT_DROPPED",
    "ProgramFacts", "ScheduledTIRRefusal", "ScheduledVerdict", "program_facts",
    "verify_scheduled_tir",
]
