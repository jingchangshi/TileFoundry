"""Emit the completion trace as REAL typed TIR, and read it back.

WHY EMIT AT ALL. A completion plan that exists only as a Python list cannot be verified
against the program that will run: the verifier would be reading the generator's own
intention. This module materialises FlagSet / FlagWait / Drain statements -- the CORE's typed
ops, so the standard verifier re-checks slot/generation/pipe legality -- and reads them back
out of the finished TIR. Verification therefore reads the ACTUAL program, and the structural
checks (slot == generation % versions, non-empty channel, legal pipe pair) run twice: once in
the core verifier, once here.

THE EMITTER PRESERVES ORDER, and that is the point. Statements are appended in trace order
inside the function's own mesh scope, so a Set that must precede its Wait still does, and a
FREE Set still lands after the last consumer's read_end. Records that carry a CONDITIONAL
guard are wrapped in an If around the single statement, so a guarded Set and a guarded Wait
become real control flow rather than an attribute the checker has to trust.

The real ACCESS statements of the program are NOT emitted here: they already exist in the
module. The trace's access roles are the references the checker uses to decide "did the FREE
happen after the last read", and they are resolved to the module's own statement order.
"""
from __future__ import annotations

from dataclasses import replace

from tilefoundry.ir.core import Constant, Var
from tilefoundry.ir.tir.flag import Drain, FlagSet, FlagWait
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import Evaluate, ExecutionScope, For, If, MeshScope, Sequential
from tilefoundry.ir.types import DType, make_tensor_type

from .guards import ALWAYS, COND, NEVER, normalize_guard
from .model import CompletionError
from .schedule import ROLE_DRAIN, ROLE_SET, ROLE_WAIT, TraceRecord

_SYNC_KEY = "ascend.completion_sync"


def _bool_const(value: bool) -> Constant:
    return Constant(type=make_tensor_type((), dtype=DType.bool), value=bool(value))


def _flag_kind(kind) -> str:
    """The CORE completion kind for one obligation kind.

    The core knows ready/free/once. STATE and EXIT edges are ordering edges between two
    writes (or the region exit), which is 'once' in the core's vocabulary, so they are not
    mislabelled as a data-ready edge.
    """
    if str(kind).lower() in ("ready",):
        return "ready"
    if str(kind).lower() in ("free",):
        return "free"
    return "once"


def _flag_node(cls, record):
    return cls(channel_name=record.channel, kind=_flag_kind(record.kind),
               slot=int(record.slot), generation=int(record.generation),
               versions=max(1, int(record.versions)), flag_id=int(record.event_id),
               producer_pipe=str(record.producer_pipe or "device"),
               consumer_pipe=str(record.consumer_pipe or "device"))


def _wrap(statement, guard, induction_name="i"):
    """Wrap one statement in its guard, so conditional completion is REAL control flow."""
    normalized = normalize_guard(guard)
    if normalized.kind == ALWAYS:
        return statement
    if normalized.kind == NEVER:
        return If(cond=_bool_const(False), then_body=Sequential(body=()),
                  else_body=Sequential(body=(statement,)))
    source = normalized.source
    if isinstance(source, For):
        raise CompletionError("COMPLETION_GUARD_INVALID",
                              "a For cannot be a statement guard", "", "expression", "For")
    cond = source if source is not None else Var(name=induction_name,
                                                type=make_tensor_type((), dtype=DType.bool))
    return If(cond=cond, then_body=Sequential(body=(statement,)),
              else_body=Sequential(body=()))


def completion_ops(target=None):
    """The completion op classes THIS target declares, falling back to the core ones.

    A target may bind its own completion ops (Ascend declares AivFlagSet/AivFlagWait/AivDrain,
    each with execution_mesh = memory_execution_mesh()). Emitting the CORE FlagSet into a
    mesh-scoped Ascend kernel is not a cosmetic mismatch: the core op declares
    thread_execution_mesh(), the kernel's mesh is ("aiv",), so the standard verifier rejects
    the finished program with "does not match Mesh(('thread',)...)". The completion statements
    were correct in content and illegal in scope, which is why the failure showed up only at
    the FINAL typed verification rather than here.
    """
    if target is not None:
        factory = getattr(target, "completion_ops", None)
        classes = factory() if callable(factory) else None
        if classes:
            return classes
    return {"set": FlagSet, "wait": FlagWait, "drain": Drain}


def trace_nodes(trace, ops=None) -> list:
    """The typed TIR statements for one completion trace, in execution order."""
    return [node for _record, node in trace_node_pairs(trace, ops)]


def trace_node_pairs(trace, ops=None) -> list:
    """[(record, node)] for one completion trace, so placement never guesses the pairing.

    Returning nodes alone forced the caller to re-derive which record a node came from, and an
    off-by-one there puts a Wait next to the WRONG access -- a defect that looks like a working
    program. Pairing them at construction makes that impossible.
    """
    ops = ops or completion_ops()
    set_cls, wait_cls, drain_cls = ops["set"], ops["wait"], ops["drain"]
    pairs = []
    for record in trace.records:
        if record.role == ROLE_SET:
            node = Evaluate(callable=_flag_node(set_cls, record), args=())
        elif record.role == ROLE_WAIT:
            node = Evaluate(callable=_flag_node(wait_cls, record), args=())
        elif record.role == ROLE_DRAIN:
            node = Evaluate(callable=drain_cls(), args=())
        else:
            # An access role is NOT a completion statement: the real access already exists in
            # the module, and re-emitting a copy would double the program.
            continue
        pairs.append((record, _wrap(node, record.guard)))
    return pairs


def _mesh_scopes(function):
    """The innermost mesh/execution scope path, so a drain lands where threads execute.

    THE PATH INCLUDES THE SCOPE SEGMENT ITSELF. It used to return the path only up to the scope
    and append the scope name separately, so a caller that descended by segment reached the
    scope's PARENT and then reported "no instance groups" -- a lookup that misses and is read
    as "this program is not unrolled". Returning the full path ('s0/MeshScope/s0') is what
    makes one descent convention work for every caller.
    """
    def find(node, path):
        if isinstance(node, Sequential):
            for index, child in enumerate(node.body):
                found = find(child, path + ("s%d" % index,))
                if found is not None:
                    return found
            return None
        if isinstance(node, (MeshScope, ExecutionScope)):
            here = path + (type(node).__name__,)
            deeper = find(node.body, here)
            # The innermost scope is the LAST one on the chain; its path is the full one.
            return deeper if deeper is not None else "/".join(here)
        return None

    found = find(function.body, ())
    return None if found is None else (found, "")


def _instance_of_record(record):
    """The instance ordinal a completion record belongs to, or None when unanchored.

    The trace enumerates (sweep, issue_order) instances; R3 emits one body per logical
    iteration numbered by the "__k<n>" suffix. On a single-sweep unrolled program the sweep IS
    the instance ordinal. Returning None for anything else keeps an ambiguous record from
    being placed by luck -- a moved Wait is a barrier the plan never asked for.
    """
    sweep = getattr(record, "sweep", None)
    order = getattr(record, "issue_order", None)
    if sweep is None:
        return None
    try:
        sweep = int(sweep)
    except (TypeError, ValueError):
        return None
    # A multi-order sweep would need the per-task mapping; a single issue order per sweep is
    # the shape R3 materializes, and anything else is left unplaced and reported.
    return sweep if order is not None else None


def _replace_body(node, old_body, new_body):
    """Replace 'old_body' with 'new_body' inside this node (a wrapper Sequential or a loop).

    The rebuild must modify the SAME node the group indices were measured against. Walking
    down explicitly, and failing loudly when the body is not found, is what keeps a placement
    that silently did nothing from being mistaken for a program with no completion statements.
    """
    if getattr(node, "body", None) is old_body:
        return replace(node, body=new_body)
    inner = getattr(node, "body", None)
    if isinstance(inner, Sequential):
        children = list(inner.body)
        for index, child in enumerate(children):
            if _contains_body(child, old_body):
                children[index] = _replace_body(child, old_body, new_body)
                return replace(node, body=Sequential(body=tuple(children)))
    raise CompletionError(
        "COMPLETION_ANCHOR_NOT_FOUND",
        "the interleaved body could not be written back into the program",
        "unrolled body", "a node containing the instance groups", None)


def _contains_body(node, target, depth=0):
    if node is target:
        return True
    if depth > 8:
        return False
    return any(_contains_body(child, target, depth + 1)
               for child in _instance_children(node))


def _insert_before(node, line, new_node, depth=0):
    """Insert 'new_node' immediately before the statement at 'line', inside 'node'.

    The path walks ONLY statement levels (sN and scope segments), so the insertion lands
    between two real statements rather than before the whole region that contains them. That
    distinction is the whole defect: a Wait placed before the region is a Wait that runs before
    the access it must follow.
    """
    if depth > 12:
        return node
    segments = list(line)
    if not segments:
        return node
    segment = segments[0]
    rest = segments[1:]

    # A Sequential's body is a BARE TUPLE, not a nested Sequential. Every helper here that
    # treated it as a node returned the input unchanged, so the placement reported success
    # while inserting nothing -- the emission then looked like a program with no completion
    # statements at all.
    def children_of(node):
        """(children, owner) for the level an 'sN' selects, or (None, None).

        Three shapes all hold statements at this level, and all three must be indexed the same
        way: a Sequential holds a tuple, a scope (MeshScope/ExecutionScope) holds a Sequential,
        and a For holds a Sequential. Handling only the Sequential case made every path that
        crossed a scope a silent no-op -- exactly the 'ExecutionScope/s11' paths that never
        placed.
        """
        if isinstance(node, Sequential):
            return list(node.body), node
        body = getattr(node, "body", None)
        if isinstance(body, Sequential):
            return list(body.body), body
        return None, None

    if segment.startswith("s") and segment[1:].isdigit():
        children, owner = children_of(node)
        if children is None:
            return node
        index = int(segment[1:])
        if index >= len(children):
            return node
        if rest:
            children[index] = _insert_before(children[index], rest, new_node, depth + 1)
        else:
            children.insert(index, new_node)
        # Write the tuple back into the node that OWNS the children: a Sequential takes a bare
        # tuple, a scope/For takes the wrapped Sequential.
        if owner is node:
            return replace(node, body=tuple(children))
        return replace(node, body=replace(owner, body=tuple(children)))

    # A SCOPE segment names this node: descend into its body. Passing 'segments' again instead
    # of 'rest' re-visited the same segment forever, so the descent burned its depth limit and
    # returned the node unchanged.
    inner = getattr(node, "body", None)
    if inner is None:
        return node
    return replace(node, body=_insert_before(inner, rest, new_node, depth + 1))


def _advance(path):
    """The position AFTER 'path': the same segments with the last index incremented.

    'after statement N' and 'before statement N+1' are the same place. Expressing a Set that
    way lets BOTH roles use one anchor convention, so a single ordered insertion list can
    reproduce the trace's interleaving exactly.
    """
    segments = list(path)
    for index in range(len(segments) - 1, -1, -1):
        segment = segments[index]
        if segment.startswith("s") and segment[1:].isdigit():
            segments[index] = "s%d" % (int(segment[1:]) + 1)
            return tuple(segments)
    return tuple(list(path) + ["end"])


def _segment_value(segment):
    """Numeric value of an 'sN' segment; scopes and 'end' sort after their siblings."""
    text = str(segment)
    if text.startswith("s") and text[1:].isdigit():
        return int(text[1:])
    return 1 << 20


def _placement_position(record, anchor, instance):
    """Where this record will be INSERTED, expressed in the same units as the anchor map.

    A Set lands after its producer, so its position is the producer's position advanced by one;
    a Wait lands before its consumer, so its position is the consumer's position. Both are
    derived from the neighbouring ACCESS, which is what makes the resulting order the real
    happens-before order rather than a guess from the record's role.
    """
    positions = sorted((int(position), tuple(path))
                       for position, path in anchor.get(instance, {}).items() if path)
    if not positions:
        return None
    here = int(getattr(record, "index", 0))
    if str(getattr(record, "role", "")) == ROLE_WAIT:
        for access_position, path in positions:
            if access_position > here:
                return path
        return None
    producer = None
    for access_position, path in positions:
        if access_position < here:
            producer = path
        else:
            break
    if producer is None:
        return ("s0",)
    return _advance(producer)


def _canonical_trace(trace, anchor):
    """The trace reordered so each instance's completion statements follow their anchor.

    Records are grouped by instance, ordered by the anchor position they attach to, and kept in
    their original relative order within one anchor (a stable sort). Non-completion records
    (the accesses) keep their slots because the anchors are expressed in terms of their trace
    indices, and the emitter resolves positions against them afterwards.
    """
    def sort_key(record):
        # ORDER BY THE PLACEMENT POSITION, THEN BY TRACE INDEX -- never by role.
        #
        # A SET is placed AFTER the access that produced the value; a WAIT is placed BEFORE the
        # access it guards. That pair of rules already encodes the happens-before relation, so
        # once each record has a placement position the ordering falls out of it.
        #
        # Sorting by role as well was wrong: 'set ready.s4' and 'wait ready.s4' can share a
        # position (both sit at the boundary of the same region) while the set must still come
        # first. Phase-first inverted exactly those pairs. Position-then-index keeps the set
        # ahead of its own wait and still puts the region's entry waits before its exit sets.
        instance = _instance_of_record(record)
        path = _placement_position(record, anchor, instance)
        if path is None:
            return (instance if instance is not None else -1, (), 1,
                    int(getattr(record, "index", 0)))
        key = tuple(_segment_value(part) for part in path)
        return (instance if instance is not None else -1, key, 0,
                int(getattr(record, "index", 0)))

    # ONLY THE COMPLETION RECORDS ARE REORDERED, IN THEIR EXISTING SLOTS.
    #
    # The access records must stay exactly where they are: they are what gives each slot a
    # generation during the protocol simulation, so dropping them left every slot's generation
    # None and the verifier reported COMPLETION_STALE_GENERATION. Permuting the completion
    # records among the slots they already occupy fixes the ORDER without touching the model
    # the order is checked against.
    slots = [index for index, record in enumerate(trace.records)
             if record.role in (ROLE_SET, ROLE_WAIT)]
    movable = sorted((trace.records[index] for index in slots), key=sort_key)
    records = list(trace.records)
    for slot, record in zip(slots, movable):
        records[slot] = record

    # RENUMBER. The placement keys every decision on 'record.index', and the trace's records
    # are then looked up in the anchor map by that same number. Permuting the records WITHOUT
    # renumbering left each one carrying its OLD index, so placement resolved it to its old
    # anchor and silently undid the canonicalisation -- emitted order came back as the original
    # order while 'expected' was the canonical one.
    renumbered = []
    remap = {}
    for position, record in enumerate(records):
        remap[int(getattr(record, "index", position))] = position
        renumbered.append(replace(record, index=position))
    return replace(trace, records=tuple(renumbered)), remap


def _place_many(node, entries, depth=0):
    """Insert every (path, new_node) in 'entries' into 'node', in ONE rebuild.

    Inserting them one at a time shifted sibling indices after each insert, so a statement
    targeted at child 5 landed before the statement already inserted at child 5 and the trace
    order came out reversed -- the second half of every instance silently misplaced. Grouping
    by target and rebuilding once is the only way to keep both the POSITION and the ORDER.

    'entries' is ordered by trace index; nodes targeting the same position keep that order.
    """
    if depth > 12 or not entries:
        return node
    # 'end' is the caller's marker for "after the last statement of this level", which is what
    # a completion statement with no following access in the trace needs.
    end_here = [new_node for segments, new_node in entries if segments == ("end",)]
    entries = [(segments, new_node) for segments, new_node in entries
               if segments != ("end",)]
    # 'entries' arrives in TRACE order and that order must survive the grouping. Sorting the
    # keys would reorder the completion program, and the verifier compares it statement by
    # statement -- a correctly PLACED statement in the wrong ORDER is still a mismatch.
    grouped = {}
    for segments, new_node in entries:
        if not segments:
            continue
        grouped.setdefault(segments[0], []).append((segments[1:], new_node))

    children, owner = _statement_children(node)
    if children is None:
        return node

    # Deeper targets first, so the child arrays are final before this level is assembled.
    for segment, items in grouped.items():
        if not segment.startswith("s") or not segment[1:].isdigit():
            inner = getattr(node, "body", None)
            if inner is None:
                continue
            return replace(node, body=_place_many(inner, items, depth + 1))
        index = int(segment[1:])
        deeper = [(rest, new_node) for rest, new_node in items if rest]
        if deeper and index < len(children):
            children[index] = _place_many(children[index], deeper, depth + 1)

    # Group by TARGET POSITION, keeping the TRACE order inside each group.
    #
    # Both halves are required and they pull in opposite directions:
    #   * position decides where the statement goes (a Wait must follow the access it orders);
    #   * trace order decides the order among statements sharing a position (the trace emits
    #     set(ready), set(ready), wait(ready) for the same region, and the verifier compares the
    #     emitted program element by element against that).
    # A running offset applied to every entry conflated the two and reordered the pairs.
    insertions = {}
    for segments, new_node in entries:
        if not segments or segments[0] == "end":
            continue
        if segments[1:]:
            continue
        index = int(segments[0][1:])
        insertions.setdefault(index, []).append(new_node)
    trailing = []

    # TRACE ORDER IS THE OUTPUT ORDER. The trace interleaves: set(ready.s1), set(ready.s4),
    # wait(ready.s1), set(free.s1), wait(ready.s4), set(free.s4) -- two statements can target
    # the SAME position one after another. Emitting "everything at position P, then position
    # Q" reorders exactly those pairs, and the verifier compares element by element.
    #
    # So place by SEQUENCE: walk the entries in trace order, insert each node before its
    # anchor's CURRENT index, and advance the running offset so later insertions at or after
    # that anchor land after it.
    rebuilt = []
    for index, child in enumerate(children):
        rebuilt.extend(insertions.pop(index, ()))
        rebuilt.append(child)
    for index in sorted(insertions):
        rebuilt.extend(insertions[index])
    rebuilt.extend(trailing)
    rebuilt.extend(end_here)
    if owner is node:
        return replace(node, body=tuple(rebuilt))
    return replace(node, body=replace(owner, body=tuple(rebuilt)))


def _statement_children(node):
    """(children list, owning node) for the level 'sN' indexes, or (None, None)."""
    if isinstance(node, Sequential):
        return list(node.body), node
    body = getattr(node, "body", None)
    if isinstance(body, Sequential):
        return list(body.body), body
    return None, None


def _append_into(node, new_node):
    """Append 'new_node' after the last statement of 'node' (its own level).

    Sequential.body is a tuple, so appending means rebuilding the tuple -- not wrapping it in
    another Sequential, which would add a nesting level the plan never described.
    """
    if isinstance(node, Sequential):
        return replace(node, body=tuple(node.body) + (new_node,))
    return node


def _instance_children(statement):
    """The top-level children of one instance body, or the statement itself."""
    body = getattr(statement, "body", None)
    if isinstance(statement, Sequential):
        return tuple(statement.body)
    if isinstance(body, Sequential):
        return tuple(body.body)
    return ()


def _access_anchors(trace, materialization=None):
    """{instance ordinal: {trace index: top-level child index}} for every ACCESS record.

    THE BUFFER NAME IN THE RECORD IS NOT AN ADDRESS, and treating it as one is what made the
    first attempt place waits at the wrong spot. A channel names a BUFFER ("smem@.../for:i/s1"),
    and that buffer is written by one statement and read by a completely different one --
    usually inside a nested ExecutionScope. Resolving the channel alone therefore gives the
    READ the WRITER's position, so a wait meant to precede the read was emitted before the
    write: COMPLETION_MAP_MISMATCH, expected set got wait.

    Two facts disambiguate it, and the record carries both:
      * 'task' -- the task the access belongs to. A read of s1 lives in the consuming task, not
        in the task that wrote s1, so the two get different positions.
      * the ROLE -- read_begin/read_end versus write_begin, which selects the map's 'mode'.
    The materialization map then supplies the real, rewritten position for that (task, mode, k).
    Reading it is not re-deriving the plan: it is reading the unroller's own record of where
    each access actually landed, which is the only thing that distinguishes a read's position
    from a write's.
    """
    positions = _materialized_positions(materialization)
    anchors = {}
    for record in trace.records:
        channel = str(getattr(record, "channel", ""))
        if not channel.startswith("access:"):
            continue
        instance = _instance_of_record(record)
        if instance is None:
            continue
        role = str(getattr(record, "role", ""))
        mode = "read" if role in ("read_begin", "read_end") else "write"
        task = str(getattr(record, "task", "") or "")
        path = _strip_buffer(channel[len("access:"):])
        child = _position_for(positions, task, path, instance, mode)
        anchors.setdefault(instance, {})[int(getattr(record, "index", 0))] = child
    return anchors


def _strip_buffer(path):
    """Drop a leading space qualifier ('smem@...') so the path is a real statement path."""
    text = str(path)
    if "@" not in text:
        return text
    qualifier, _, rest = text.partition("@")
    return rest if qualifier in ("smem", "ub", "gm", "l1", "l0a", "l0b", "l0c", "vreg", "preg",
                                 "scalar", "bias", "l2") else text


def _materialized_positions(materialization):
    """{(task path, mode, k): top-level child index} from the unroller's own map."""
    positions = {}
    for entry in getattr(materialization, "entries", ()) or ():
        task = str(_field(entry, "task_id", "") or "")
        mode = str(_field(entry, "mode", "") or "")
        try:
            k = int(_field(entry, "k", 0) or 0)
        except (TypeError, ValueError):
            continue
        new_stmt = str(_field(entry, "new_stmt", "") or "")
        if not task or not new_stmt:
            continue
        line = _relative_path(_strip_unrolled(new_stmt))
        if line:
            positions[(task, mode, k)] = _hoist_out_of_execution_scope(line)
    return positions


def _hoist_out_of_execution_scope(path):
    """Drop everything from 'ExecutionScope' onward.

    THE TARGET SAYS THIS AND THE IR ENFORCES IT: "Flag ops are mesh-scoped, not SIMD-VF mode
    ops; placing them inside ExecutionScope('ascend.simdvf') is invalid." AivFlagWait declares
    no execution placement, so the shared verifier rejects a flag emitted inside a mode scope
    with "this operation declares no execution placement and may not run inside ExecutionScope".

    P4-01 already established the rule (sync._anchor_path); this keeps the interleaved emitter
    on the SAME convention instead of inventing a second one. The enclosing AIV MeshScope is
    the correct domain for a flag, and the statement that owns the ExecutionScope is the right
    anchor: the flag then runs before (or after) that region as a whole, which is exactly the
    order the plan describes.
    """
    segments = list(path)
    for index, segment in enumerate(segments):
        if segment == "ExecutionScope":
            return tuple(segments[:index])
    return tuple(segments)


def _relative_path(task_path):
    """('s6', 'ExecutionScope', 's2') for '.../for:i/s6/ExecutionScope/s2/Vld:0'.

    The path RELATIVE TO THE INSTANCE BODY, truncated at the last statement segment. Keeping
    only the top-level child collapsed a write and the read it feeds into the same position,
    which is why a Wait could not be placed between them: it is the deeper segments that
    separate them.
    """
    parts = [part for part in str(task_path).split("/") if part]
    for index, part in enumerate(parts):
        if part.startswith("for:"):
            tail = parts[index + 1:]
            # Drop the trailing callable segment ('Vld:0', 'GmToUb:1'): it names what runs, not
            # where it sits.
            kept = []
            for segment in tail:
                if segment.startswith("s") and segment[1:].isdigit():
                    kept.append(segment)
                elif segment in ("MeshScope", "ExecutionScope"):
                    kept.append(segment)
                else:
                    break
            return tuple(kept)
    return ()


def _strip_unrolled(path):
    """'unrolled:k0:k/s0/...' -> 'k/s0/...' (the unroller's prefix is not part of the tree)."""
    text = str(path)
    if not text.startswith("unrolled:"):
        return text
    rest = text[len("unrolled:"):]
    cut = rest.find(":")
    return rest[cut + 1:] if cut >= 0 else rest


def _field(entry, name, default=None):
    if isinstance(entry, dict):
        return entry.get(name, default)
    return getattr(entry, name, default)


def _position_for(positions, task, path, instance, mode):
    """The child index of one access, resolved through the materialization map.

    Returns None when the map does not place it. None makes the caller leave the statement
    alone, which is the honest outcome: a completion statement at a position chosen by
    coincidence is worse than one reported as unplaceable.
    """
    direct = positions.get((task, mode, instance))
    if direct is not None:
        return direct
    for (entry_task, entry_mode, k), child in positions.items():
        if entry_mode != mode or k != instance:
            continue
        if entry_task == task or entry_task.startswith(task) or task.startswith(entry_task):
            return child
    return None


def _child_index(task_path):
    """The TOP-LEVEL child index of an instance body that a task path names.

    Paths look like 'k/s0/MeshScope/s0/for:i/s2/ExecutionScope/s10'. Inside one instance body
    the only segment that matters is the first one after the loop, because everything below it
    is nested INSIDE that child. Returning None keeps an unparseable path from being placed by
    coincidence.
    """
    parts = [part for part in str(task_path).split("/") if part]
    for index, part in enumerate(parts):
        if part.startswith("for:"):
            following = parts[index + 1] if index + 1 < len(parts) else ""
            if following.startswith("s") and following[1:].isdigit():
                return int(following[1:])
            return None
    return None


def _ordinal_at(groups, index):
    """The instance ordinal whose group contains the statement at 'index', or None."""
    for ordinal, members in groups.items():
        for member_index, _statement in members:
            if member_index == index:
                return ordinal
    return None


def _instance_runs(container):
    """Yield (ordinal_or_None, statement) for each top-level child of an unrolled body.

    Grouping is by the emitted "__k<n>" name suffix, which is the emitter's own record of the
    logical iteration a statement belongs to. Reading it is reading the PROGRAM. Statements
    with no suffix (a region-level prologue) are yielded with None so the caller keeps them in
    place instead of dropping them.
    """
    import re

    body = getattr(container, "body", None)
    if not isinstance(body, Sequential):
        return
    for statement in body.body:
        name = str(getattr(getattr(statement, "var", None), "name", "") or "")
        match = re.search(r"__k(\d+)", name)
        yield (int(match.group(1)) if match else None), statement


def _rebuild_at(function, path, new_node):
    """Replace the node at a structural segment path ('s0/MeshScope/s0') with 'new_node'.

    THE LAST SEGMENT REPLACES, EVERY EARLIER ONE DESCENDS. 'sN' means "child N of the body";
    a scope segment means "this node's body". A path ending in 'sN' therefore replaces that
    CHILD, and a path ending in a scope name replaces that scope's body. Note the asymmetry
    against the descent: while walking, 's0' must first descend INTO child 0 before the next
    segment applies.
    """
    segments = [part for part in str(path).split("/") if part]
    if not segments:
        return replace(function, body=new_node)

    def walk(node, remaining):
        if not remaining:
            return new_node
        segment = remaining[0]
        rest = remaining[1:]
        if segment.startswith("s") and segment[1:].isdigit():
            # A Sequential's body is a BARE TUPLE, so 'sN' indexes the NODE ITSELF when the
            # node is a Sequential. Checking isinstance(body, Sequential) here returned the
            # node untouched for exactly that case -- the write-back silently did nothing and
            # the interleaved loops were discarded.
            if isinstance(node, Sequential):
                children = list(node.body)
            else:
                body = getattr(node, "body", None)
                if not isinstance(body, Sequential):
                    return node
                children = list(body.body)
            index = int(segment[1:])
            if index >= len(children):
                return node
            children[index] = new_node if not rest else walk(children[index], rest)
            # A Sequential's body is ALWAYS a tuple; rebuilding it with a bare node produced
            # Sequential(body=<For>), which the shared IR verifier rejects three passes later.
            if isinstance(node, Sequential):
                return replace(node, body=tuple(children))
            return replace(node, body=Sequential(body=tuple(children)))
        # A SCOPE SEGMENT IS A STEP, NOT A REPLACEMENT -- even when it is last. 's0/MeshScope'
        # names the MeshScope ITSELF; replacing its body with the new loop would put the loop
        # where the scope's children belong and drop everything else in the scope. When the
        # scope IS the final target the caller must pass the path to the node INSIDE it.
        inner = getattr(node, "body", None)
        if inner is None:
            return node
        if not rest:
            # Ending on a scope labels the node we are standing in; there is nothing left to
            # select, so return it unchanged and let the caller notice.
            return node
        return replace(node, body=walk(inner, rest))

    return walk(function, segments)


def _instance_groups(function):
    """Split an unrolled body into its per-instance groups, keyed by instance ordinal.

    R3 materialization emits ONE BODY PER LOGICAL ITERATION under a loop that runs once, and
    suffixes every declared name with "__k<n>". That suffix is the emitter's own record of
    which instance a statement belongs to, so it is what can be used to put a Set/Wait next to
    the access it orders. Reading it is reading the PROGRAM, not the plan.

    Returns {ordinal: (prefix_nodes, suffix_nodes)} together with the container so the caller
    can rebuild it. A body without instance suffixes yields {} and the caller falls back to
    appending, which is the correct behaviour for a program that is not unrolled.
    """
    import re

    if not isinstance(function, PrimFunction):
        return {}, None, [], None

    # DESCEND BY SEGMENT, not by "the sN parts". The mesh scope path is ('s0', 'MeshScope'),
    # and filtering for parts that start with "s" silently DROPS the MeshScope segment -- so
    # the descent stopped one level too high, found no instance groups, and fell back to
    # appending. That is the same class of bug as the old index_statements walk: a traversal
    # that only understands the segments it was written against.
    scope = _mesh_scopes(function)
    if scope is None:
        return {}, None, [], None
    path, _kind = scope

    container = function
    for segment in [part for part in str(path).split("/") if part]:
        body = getattr(container, "body", None)
        if segment.startswith("s") and segment[1:].isdigit():
            if not isinstance(body, Sequential) or int(segment[1:]) >= len(body.body):
                return {}, None, [], None
            container = body.body[int(segment[1:])]
        elif segment in ("MeshScope", "ExecutionScope"):
            # The scope segment is the node we JUST descended into: 'sN' already selected it.
            # Assigning its BODY here skipped one level, so the loop search then ran on the
            # wrong node and reported "not unrolled". The segment names the container, it does
            # not tell you to leave it.
            if type(container).__name__ != segment:
                return {}, None, [], None
            continue
        else:
            return {}, None, [], None

    import re

    def ordinal_of(node):
        """The instance ordinal of a statement, read from ANY name it declares.

        The unrolled body groups its statements under a Sequential PER INSTANCE, so the
        instance's ordinal lives on the statements INSIDE that group, not on the group itself
        (a Sequential declares no name). Scanning the whole subtree is what lets the group be
        identified without assuming which statement kind carries the suffix.
        """
        best = None

        def visit(inner, depth=0):
            nonlocal best
            if depth > 32 or best is not None:
                return
            name = str(getattr(getattr(inner, "var", None), "name", "") or "")
            match = re.search(r"__k(\d+)", name)
            if match:
                best = int(match.group(1))
                return
            for attr in ("body", "then_body", "else_body"):
                child = getattr(inner, attr, None)
                if child is None:
                    continue
                if isinstance(child, tuple):
                    for item in child:
                        visit(item, depth + 1)
                elif hasattr(child, "body"):
                    visit(child, depth + 1)

        visit(node)
        return best

    # The instanced bodies live under the loop, so find the loop between here and them. It is
    # the REGION whose body is replicated per instance. Two shapes are both real in this tree:
    #   MeshScope -> Seq -> For            (the carrier)
    #   MeshScope -> Seq -> For -> Seq*    (a deeper wrapper)
    # so the search descends through transparent Sequentials instead of assuming one depth.
    def find_loop(node, depth=0):
        """(loop, index within its parent Sequential), or (None, None)."""
        if depth > 8:
            return None, None
        if isinstance(node, Sequential):
            for index, child in enumerate(node.body):
                if isinstance(child, For):
                    return child, index
                found, _ = find_loop(child, depth + 1)
                if found is not None:
                    return found, None
        return None, None

    body = getattr(container, "body", None)
    loop, loop_index = find_loop(body) if body is not None else (None, None)
    if loop is None:
        return {}, None, [], None
    container = loop
    body = getattr(container, "body", None)
    if not isinstance(body, Sequential):
        return {}, None, [], None
    # Reach the level whose children ARE the per-instance groups: descend through any single
    # wrapper Sequential so the group scan sees the instance bodies rather than one wrapper.
    # 'container' MUST stay the node that owns 'body'; rebuilding it via replace() here made it
    # a DIFFERENT object from the one in the tree, so the later rebuild mutated a copy and left
    # Sequential(body=<For>) -- a corrupt structure the shared IR verifier rejected.
    while len(body.body) == 1 and isinstance(body.body[0], Sequential):
        body = body.body[0]
        # NOTE: 'container' is deliberately NOT rebuilt here. It stays the node reachable by
        # 'path', and the caller re-wraps 'body' explicitly; rebuilding a copy here produced a
        # replacement that no longer matched the tree, so its edits were discarded.
    # The path must POINT AT THE LOOP NODE, not at its body: the rebuild replaces that node
    # with a new loop carrying the interleaved body. Ending the path on a 'for:' label made the
    # last segment resolve against the loop's BODY, so the rebuild wrote a Sequential where a
    # For belonged and the shared IR verifier rejected the module.
    if loop_index is None:
        return {}, None, [], None
    path = path + "/s%d" % loop_index

    groups = {}
    for index, statement in enumerate(body.body):
        ordinal = ordinal_of(statement)
        if ordinal is None:
            continue
        groups.setdefault(ordinal, []).append((index, statement))
    if not groups:
        return {}, None, [], None
    # 'container' is the node NAMED BY 'path' (where the rebuild writes back); 'body' is the
    # level the group indices were measured at. They differ when a wrapper Sequential was
    # descended through, and conflating them is what made the placement silently no-op.
    return groups, container, path, body


def emit_trace_tir(function, trace, *, events=None, target=None, anchor=None,
                   materialization=None) -> PrimFunction:
    """Return 'function' with the completion statements placed AT their instance positions.

    THE DEFECT THIS REPLACES. The first version appended the whole completion program as one
    block inside the mesh scope, after every access. Every flag was individually legal --
    correct channel, correct slot, legal pipe pair -- and the resulting program still could not
    work: a Wait emitted after the access it guards guards nothing, so the kernel would read
    stale UB on the very first iteration. A per-statement check cannot see this, and it is why
    VerifyScheduledTIR tests ORDER RELATIVE TO THE ACCESS rather than just counting flags.

    The placement rule is the trace's own, applied to real statements: FREE/STATE/READY waits
    that protect a write or a read go BEFORE their instance body; READY sets that publish a
    write's completion and the region-exit Drain go AFTER it. Records whose instance cannot be
    located are appended with a diagnostic rather than silently dropped (a lost Set loses a
    synchronization) or silently moved (a moved Wait puts a barrier where the plan never said
    there was one).

    'events' is the EventAssignment; when supplied, each record's flag id is taken from it so
    the emitted program and the allocation cannot silently disagree.
    """
    allocation = None
    if events is not None:
        allocation = {item.channel: item.event_id for item in events.allocations}
    records = []
    for record in trace.records:
        if allocation and record.channel in allocation:
            record = record.with_event(allocation[record.channel])
        elif allocation and record.role in (ROLE_SET, ROLE_WAIT):
            # A traced channel with no allocation is a real gap: inventing an id here would
            # hide an exhausted/unallocated namespace.
            keyed = [name for name in allocation
                     if name.split("#")[0] == record.channel]
            if not keyed:
                raise CompletionError(
                    "COMPLETION_CHANNEL_UNBOUND",
                    "the completion trace contains a channel with no physical event"
                    " allocation, so it cannot be emitted",
                    record.channel, "an allocated event id", None)
            record = record.with_event(allocation["%s#s%d" % (record.channel, record.slot)])
        records.append(record)
    resolved = replace(trace, records=tuple(records))
    ops = completion_ops(target)

    # CANONICALISE BEFORE BUILDING THE NODES.
    #
    # The placement walks 'owning', which pairs each record with its node. Building that list
    # from the ORIGINAL trace and canonicalising 'resolved' afterwards left the two out of
    # step: the program was emitted in the old order while verification compared it against the
    # canonical one, so the reordering never actually reached the program. Canonicalisation has
    # to come first, and 'owning' must be built from its result.
    anchor = anchor if anchor is not None else _access_anchors(resolved, materialization)
    groups, container, indices, _group_body = _instance_groups(function)
    if groups and anchor:
        resolved, remap = _canonical_trace(resolved, anchor)
        anchor = {instance: {remap.get(index, index): path
                             for index, path in by_index.items()}
                  for instance, by_index in anchor.items()}

    owning = trace_node_pairs(resolved, ops)
    nodes = [node for _record, node in owning]
    if not nodes:
        return function, resolved

    # INTERLEAVED EMISSION. Nodes and records are built by ONE loop, so the pairing below is
    # exact rather than positional guesswork: a node is emitted only for a Set/Wait/Drain
    # record, and 'owning' carries that record alongside it.
    if groups and anchor:
        # INTERLEAVE IN TRACE ORDER, POSITIONED BY THE ACCESS ANCHORS.
        #
        # The emitted program must satisfy TWO things at once, and each alone is easy:
        #   1. the completion statements appear in the TRACE's order (the verifier compares
        #      them statement by statement against the program it verified);
        #   2. each one sits next to the access it orders (a Wait after its access guards
        #      nothing).
        # Binning by role -- all waits before the instance, all sets after -- satisfies (2) for
        # some records and BREAKS (1), because the trace interleaves them. That is exactly what
        # COMPLETION_MAP_MISMATCH reported: expected set, emitted wait.
        #
        # So the placement is driven by the trace's own sequence: each completion statement is
        # flushed just before the first access that follows it in the trace, using the access
        # anchors to know which real statement that is.
        region_exit = []
        by_instance = {}
        for record, node in owning:
            if record.role == ROLE_DRAIN:
                # A Drain is a REGION-EXIT barrier: ONCE, after every instance, never inside
                # one. Appending it per slot emitted N drains for one trace record, which the
                # verifier saw as "3 more completion statements than required".
                region_exit.append(node)
                continue
            by_instance.setdefault(_instance_of_record(record), []).append((record, node))

        rebuilt = []
        placed_any = False
        for index, statement in enumerate(container.body.body):
            ordinal = _ordinal_at(groups, index)
            if ordinal is None:
                rebuilt.append(statement)
                continue
            pending = list(by_instance.get(ordinal, []))
            anchors = anchor.get(ordinal, {})
            # Each completion statement goes immediately BEFORE the next access in trace order.
            # Anchoring at the FULL nested path is required: a write and the read it feeds can
            # live in the SAME child (both are inside one ExecutionScope), so a child-level
            # anchor cannot put a Wait between them -- it can only put it before or after both.
            ordered = sorted(
                ((position, tuple(path)) for position, path in anchors.items() if path),
                key=lambda item: item[0])
            # Build the placement list in TRACE ORDER, then rebuild the instance body ONCE.
            #
            # A WAIT anchors to the access it GUARDS (the next access) and goes BEFORE it.
            # A SET anchors to the access that PRODUCED the value (the previous access) and
            # goes AFTER it -- which is the same as "before the following statement".
            #
            # Anchoring a Set to the NEXT access instead put every Set in front of the region
            # that consumed its buffer rather than just after the write that published it, so
            # the emitted sequence came out as set,wait,set,... instead of set,set,wait,...
            # and the verifier (rightly) refused it. Normalising both roles to a single
            # 'insert before index N' makes ONE ordering possible, which is what the trace says.
            # ENTRY WAITS GO BEFORE THE REGION, EXIT SETS GO AFTER IT.
            # Both are anchored to the instance's access range rather than to a per-record
            # neighbour, because after hoisting every record in one region shares that anchor.
            entries = []
            for record, line in pending:
                position = _placement_position(record, anchor, ordinal) or ("end",)
                entries.append((position, line))
            new_statement = _place_many(statement, entries)
            placed_any = True
            rebuilt.append(new_statement)
        if placed_any:
            rebuilt.extend(region_exit)
            new_body = Sequential(body=tuple(rebuilt))
            new_container = replace(container, body=new_body)
            return _rebuild_at(function, indices, new_container), resolved

    scope = _mesh_scopes(function)
    if scope is None:
        # No mesh scope: appending at function top level is the only structurally valid
        # placement left, and it is recorded by being what it is (the drain is function-level).
        body = Sequential(body=tuple(function.body.body) + tuple(nodes))
        return replace(function, body=body), resolved
    path, _kind = scope
    indices = [int(part[1:]) for part in path.split("/") if part.startswith("s")]
    if not indices:
        body = Sequential(body=tuple(function.body.body) + tuple(nodes))
        return replace(function, body=body), resolved

    def insert(node, remaining):
        if not remaining:
            if isinstance(node, Sequential):
                return Sequential(body=tuple(node.body) + tuple(nodes))
            body = getattr(node, "body", None)
            if isinstance(body, Sequential):
                return replace(node, body=Sequential(body=tuple(body.body) + tuple(nodes)))
            return node
        index = remaining[0]
        if isinstance(node, Sequential):
            children = list(node.body)
            children[index] = insert(children[index], remaining[1:])
            return Sequential(body=tuple(children))
        body = getattr(node, "body", None)
        if isinstance(body, Sequential):
            return replace(node, body=insert(body, remaining))
        return node

    return replace(function, body=insert(function.body, indices)), resolved


def read_trace_from_tir(function) -> list:
    """Read the completion statements BACK out of a finished TIR, in tree order.

    The reader is deliberately independent of the trace that produced them: it walks the real
    tree, honours reachability (a Set inside a statically false arm cannot execute and is not
    coverage), and returns records with the ATTRIBUTES THE PROGRAM ACTUALLY CARRIES. A verifier
    built on this cannot be satisfied by a map or a flag.
    """
    from .guards import ALWAYS, NEVER

    records = []
    counter = {"n": 0}

    def walk(node, reachable=True, guard=None):
        if node is None:
            return
        if isinstance(node, If):
            cond = getattr(node, "cond", None)
            static = _static_bool(cond)
            then_guard = _guard_of(cond, guard)
            if static is False:
                walk(node.else_body, reachable, guard)
            elif static is True:
                walk(node.then_body, reachable, guard)
            else:
                walk(node.then_body, reachable, then_guard)
                walk(node.else_body, reachable, guard)
            return
        if isinstance(node, Sequential):
            # A Sequential's body is a tuple; anything else is a single node. Iterating a
            # single node raised 'For object is not iterable' -- and it only showed up once the
            # reader was pointed at an UNROLLED body, which contains loops. The reader is
            # supposed to be independent of the emitter, so it must accept every statement
            # shape the real IR contains, not just the ones the emitter happened to produce.
            children = node.body if isinstance(node.body, tuple) else (node.body,)
            for child in children:
                walk(child, reachable, guard)
            return
        if isinstance(node, (MeshScope, ExecutionScope, For)):
            body = getattr(node, "body", None)
            if isinstance(node, For) and _trip(node) == 0:
                walk(body, False, guard)
            else:
                walk(body, reachable, guard)
            return
        if isinstance(node, Evaluate):
            target = node.callable
            if isinstance(target, FlagSet):
                records.append(_record(counter, "set", target, reachable, guard))
            elif isinstance(target, FlagWait):
                records.append(_record(counter, "wait", target, reachable, guard))
            elif isinstance(target, Drain):
                records.append(_record(counter, "drain", target, reachable, guard))
            return

    walk(function.body)
    return records


def _record(counter, role, target, reachable, guard):
    record = TraceRecord(
        index=counter["n"], role=role,
        channel=str(getattr(target, "channel_name", "")),
        obligation_id="", kind=str(getattr(target, "kind", "")),
        generation=int(getattr(target, "generation", -1)),
        slot=int(getattr(target, "slot", -1)),
        versions=int(getattr(target, "versions", 1)),
        task="", core="", pipe="",
        producer_pipe=str(getattr(target, "producer_pipe", "")),
        consumer_pipe=str(getattr(target, "consumer_pipe", "")),
        guard=guard, event_id=int(getattr(target, "flag_id", -1)),
        sweep=0, issue_order=counter["n"])
    counter["n"] += 1
    return _Annotated(record, reachable)


class _Annotated:
    """A read-back record plus the two facts only the READER can know."""

    __slots__ = ("record", "reachable")

    def __init__(self, record, reachable):
        self.record = record
        self.reachable = reachable

    def __getattr__(self, name):
        return getattr(self.record, name)

    def as_dict(self):
        return {**self.record.as_dict(), "reachable": self.reachable}


def _static_bool(cond):
    value = getattr(cond, "value", None)
    if isinstance(value, bool):
        return value
    return None


def _guard_of(cond, outer):
    from .guards import Guard

    static = _static_bool(cond)
    if static is True:
        return outer
    if static is False:
        return Guard(NEVER, key="never")
    return Guard(COND, key=_key(cond), source=cond)


def _key(cond):
    from .guards import guard_key

    return guard_key(cond)


def _trip(loop):
    values = []
    for bound in (getattr(loop, "start", None), getattr(loop, "stop", None),
                  getattr(loop, "step", None)):
        value = getattr(bound, "value", None)
        if not isinstance(value, int):
            return None
        values.append(value)
    start, stop, step = values
    if step <= 0:
        return 0
    return max(0, (stop - start + step - 1) // step)


__all__ = ["completion_ops", "trace_node_pairs", "trace_nodes", "emit_trace_tir",
           "read_trace_from_tir"]
