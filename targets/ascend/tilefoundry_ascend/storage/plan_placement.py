"""Assigning real byte ranges: the allocator, and the checks that make it trustworthy.

The allocator is deliberately simple (first-fit by descending size inside one space); its
value is not the heuristic but the INPUTS it is allowed to see. It places a buffer only
with an interval derived from completion evidence, and it reuses bytes only between
buffers whose intervals are disjoint. Every violation is a named refusal.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

from ..memory import SPACE_FACTS
from .model import (
    LAYOUT_FRACTAL_NZ,
    LAYOUT_LINEAR,
    LAYOUT_RESERVED,
    LAYOUT_SCRATCH,
    STORAGE_SCHEMA,
    LayoutRequest,
    StorageFacts,
    StorageRefusal,
    align_up,
    bytes_for,
    facts_for,
)
from .placement import (
    STATUS_PLACED,
    STATUS_REFUSED,
    Interval,
    PlacementEntry,
    PlacementReport,
    SpaceUsage,
    StoragePlacement,
)


def _digest(payload) -> str:
    import hashlib
    import json

    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str)
                          .encode("utf-8")).hexdigest()[:16]


# ------------------------------------------------------------------ layout classification
#: Which layout a root occupies, decided from the SPACE and the plan's own declaration.
#: A matrix operand in L0A/L0B is fractal; a UB staging tile is linear; a UF32 accumulation
#: is scratch. The rule is stated once here so the placement and its verifier cannot differ.
FRACTAL_SPACES = ("ascend.l0a", "ascend.l0b")


def layout_of(buffer, *, scratch_roots=()) -> str:
    """The layout kind of one buffer, from its space and declaration.

    'scratch_roots' is how a CALLER declares that a root is FP32 accumulation scratch. It is
    a parameter, not a guess: inferring scratch from a dtype would silently turn every f32
    buffer into a 4-byte-per-lane reservation even when the plan meant a payload array.
    """
    root = str(getattr(buffer, "root", ""))
    if root in scratch_roots:
        return LAYOUT_SCRATCH
    space = str(getattr(buffer, "space", ""))
    if space in FRACTAL_SPACES:
        return LAYOUT_FRACTAL_NZ
    return LAYOUT_LINEAR


def _request_for(buffer, layout, *, scratch_roots=(), reserved_roots=(), slot_count=None):
    shape = tuple(getattr(buffer, "shape", ()) or ())
    dtype = str(getattr(buffer, "dtype", "f32") or "f32")
    lanes = 0
    scratch_dtype = ""
    if layout == LAYOUT_SCRATCH:
        lanes = 1
        for dim in shape:
            lanes *= int(dim)
        scratch_dtype = dtype
    reserved = 0
    if str(getattr(buffer, "root", "")) in reserved_roots:
        reserved = int(getattr(buffer, "single_version_bytes", 0) or 0)
    return LayoutRequest(
        root=str(getattr(buffer, "root", "")), shape=shape, dtype=dtype, layout=layout,
        version_stride_bytes=getattr(buffer, "version_stride_bytes", None),
        scratch_dtype=scratch_dtype, scratch_lanes=lanes, reserved_bytes=reserved,
        alignment=int(getattr(buffer, "alignment", 64) or 64),
        source=str(getattr(buffer, "source_decl", "") or ""))


# ------------------------------------------------------------------ completion-aware lives
def intervals_from_completion(plan, trace) -> dict:
    """root -> slot -> Interval, with each END taken from a COMPLETION record.

    THE RULE, stated so it can be checked: a slot's live interval ends at the point where
    the last read of that slot has COMPLETED. It does not end when the last read is issued,
    because an issue is not a completion: a store that has been issued is still reading the
    buffer, and handing those bytes to another logical buffer at that moment is precisely
    the "reuse before MTE3 read completes" defect.

    Where no completion record exists for a slot, the interval is closed at the END of the
    trace and the caller is told so in 'diagnostics' -- an unclosed interval is not reusable,
    which is the safe direction, and it is reported rather than silently treated as free.
    """
    starts = {}
    stops = {}
    evidence = {}
    open_roots = set()
    for record in trace.records:
        root = _root_of(record)
        if not root:
            continue
        key = (root, int(record.slot))
        if record.role == "write_begin":
            starts.setdefault(key, record.index)
        elif record.role == "read_end":
            stops[key] = max(stops.get(key, record.index), record.index)
            evidence[key] = "access:%s gen=%d read_end" % (root, record.generation)
        elif record.role == "write_begin" and key not in stops:
            open_roots.add(root)

    intervals = {}
    diagnostics = []
    horizon = (trace.records[-1].index + 1) if trace.records else 0
    for (root, slot), start in sorted(starts.items()):
        stop = stops.get((root, slot))
        if stop is None:
            # A write whose read never completed WITHIN this trace. It is live to the end.
            stop = horizon
            diagnostics.append(
                "slot %d of %r has no completed read in the trace; its interval is closed at"
                " the region end (%d) and is therefore NOT offered for reuse" % (slot, root,
                                                                                 horizon))
            open_roots.add(root)
        intervals.setdefault(root, {})[slot] = Interval(
            start=start, stop=stop,
            evidence=evidence.get((root, slot), "closed at region end"))
    return intervals, tuple(diagnostics)


def _root_of(record) -> str:
    """The buffer root a trace record refers to, from the channel name it carries."""
    channel = str(getattr(record, "channel", "") or "")
    if channel.startswith("access:"):
        return channel.split("access:", 1)[1]
    return ""


def whole_region_interval(trace, root) -> Interval:
    """The interval covering every instance of a root: used when per-slot data is absent.

    A root with NO access record in the completion trace gets the WHOLE REGION, not an empty
    interval. An empty interval overlaps nothing, so every such root would be placed at
    offset 0 and alias every other such root -- and the alias check would pass, because two
    zero-length intervals do not overlap. That was a real defect here: gm buffers, which are
    function parameters and never appear as a completion access, all landed on byte 0 with
    'reused=1024' reported as if it were an achievement.

    "No completion access" means the completion program says nothing about this root, NOT
    that the root is dead. The safe reading of silence is "live throughout".
    """
    indices = [record.index for record in trace.records if _root_of(record) == root]
    if not indices:
        horizon = (trace.records[-1].index + 1) if trace.records else 1
        return Interval(start=0, stop=horizon,
                        evidence="no completion access for %r: live across the whole region"
                                 % root)
    return Interval(start=min(indices), stop=max(indices) + 1,
                    evidence="whole-region span of %r" % root)


# ------------------------------------------------------------------ allocation
def _capacity_of(space, capacities) -> tuple:
    """(capacity_bytes_or_None, known) for one space.

    A REGISTER class is special and the distinction is easy to get wrong: ascend.vreg's
    'nominal_bytes' is the WIDTH OF ONE REGISTER (256 B), not a budget for all live vector
    registers. Treating it as a capacity would either refuse every real program (a 64-lane
    f32 tile is exactly 256 B, so two of them 'exceed' it) or, worse, pass by luck. The
    aggregate register budget needs a register COUNT, which this target does not source --
    so the capacity is reported UNKNOWN, and a caller that needs the bound must say so.
    """
    if capacities and space in capacities:
        return int(capacities[space]), True
    facts = SPACE_FACTS.get(space)
    if facts is None:
        raise StorageRefusal("STORAGE_SPACE_UNKNOWN", "no declared space facts", space,
                             sorted(SPACE_FACTS), space)
    if not facts.allocatable:
        raise StorageRefusal(
            "STORAGE_SPACE_UNALLOCATABLE",
            "this space is not allocatable (it is an implicit cache); placing bytes there"
            " would produce an address no allocation can satisfy",
            space, "allocatable space", facts.category)
    from ..memory import REGISTER

    if facts.category == REGISTER:
        return None, False
    usable = facts.usable_bytes
    if usable is None:
        return facts.nominal_bytes, facts.nominal_bytes is not None
    return int(usable), True


def plan_storage(plan, trace, *, facts=None, capacities=None, scratch_roots=(),
                 reserved_roots=(), profile_id=None, completion_digest="") -> StoragePlacement:
    """Compute a versioned, completion-aware placement. Refuses rather than guesses."""
    facts = facts or facts_for(profile_id)
    entries = []
    diagnostics = []
    try:
        intervals_by_root, notes = intervals_from_completion(plan, trace)
        diagnostics.extend(notes)
        for buffer in plan.buffers:
            root = str(buffer.root)
            layout = layout_of(buffer, scratch_roots=scratch_roots)
            request = _request_for(buffer, layout, scratch_roots=scratch_roots,
                                   reserved_roots=reserved_roots)
            sizes = bytes_for(request, facts)
            slots = max(1, int(getattr(buffer, "slot_count", 1) or 1))
            per_root = intervals_by_root.get(root)
            if per_root:
                # The union carries the UNDERLYING completion facts, not just a count. A
                # reviewer asking "why is this reusable at 52" needs the read that closed it;
                # "union of 1 slot interval(s)" is a summary that hides exactly the evidence
                # the reuse rests on, and a summary cannot be checked against the trace.
                evidence_parts = sorted({i.evidence for i in per_root.values() if i.evidence})
                interval = Interval(
                    start=min(i.start for i in per_root.values()),
                    stop=max(i.stop for i in per_root.values()),
                    evidence=("union of %d slot interval(s): %s"
                              % (len(per_root), "; ".join(evidence_parts))
                              if evidence_parts else
                              "union of %d slot interval(s)" % len(per_root)))
            else:
                interval = whole_region_interval(trace, root)
                diagnostics.append(
                    "%r has no per-slot access in the completion trace; its interval is the"
                    " whole-region span and it will not share bytes" % root)
            entries.append({"buffer": buffer, "layout": layout, "sizes": sizes,
                            "slots": slots, "interval": interval})
    except StorageRefusal as refusal:
        return StoragePlacement(
            schema=STORAGE_SCHEMA, entries=(), facts=facts, input_digest=_digest([]),
            completion_digest=completion_digest,
            report=PlacementReport(STATUS_REFUSED, refusal.code, refusal.message),
            diagnostics=tuple(diagnostics))

    placed = []
    by_space = {}
    for item in entries:
        by_space.setdefault(str(item["buffer"].space), []).append(item)

    for space in sorted(by_space):
        capacity, known = _capacity_of(space, capacities)
        space_facts = SPACE_FACTS[space]
        space_align = int(space_facts.alignment or 1)
        # Largest first is only a deterministic ORDER; the legal placement comes from the
        # interval test below, so a different order changes addresses but not legality.
        items = sorted(by_space[space],
                       key=lambda item: (-(item["slots"] * item["sizes"]["single_version_bytes"]),
                                         str(item["buffer"].root)))
        occupied = []   # (base, end, entry, interval)
        for item in items:
            buffer = item["buffer"]
            sizes = item["sizes"]
            stride = align_up(sizes["version_stride_bytes"], max(space_align,
                                                                int(buffer.alignment or 1)))
            allocation = stride * item["slots"]
            interval = item["interval"]
            base = None
            shared = []
            for _base, _end, other, other_interval in occupied:
                if not interval.overlaps(other_interval):
                    continue  # disjoint lifetimes: those bytes are available again
            candidates = [0] + [end for _base, end, _o, _i in occupied]
            for candidate in sorted(candidates):
                aligned = align_up(candidate, max(space_align, int(buffer.alignment or 1)))
                overlapping = [(base_o, end_o, root_o) for base_o, end_o, root_o, iv_o
                               in occupied
                               if aligned < end_o and base_o < aligned + allocation
                               and iv_o.overlaps(interval)]
                if overlapping:
                    continue
                if any(aligned < end_o and base_o < aligned + allocation
                       and not iv_o.overlaps(interval)
                       for base_o, end_o, root_o, iv_o in occupied):
                    shared = [root_o for base_o, end_o, root_o, iv_o in occupied
                              if aligned < end_o and base_o < aligned + allocation]
                base = aligned
                break
            if base is None:
                base = align_up(max([0] + [end for _b, end, _o, _i in occupied]),
                                max(space_align, int(buffer.alignment or 1)))
            placed.append(PlacementEntry(
                root=str(buffer.root), space=space, owner=str(getattr(buffer, "owner", "")),
                layout=item["layout"], base_offset_bytes=int(base),
                version_stride_bytes=int(stride), slot_count=int(item["slots"]),
                allocation_bytes=int(allocation), interval=interval,
                payload_bytes=sizes["payload_bytes"], padding_bytes=sizes["padding_bytes"],
                scratch_bytes=sizes["scratch_bytes"], staging_bytes=sizes["staging_bytes"],
                reserved_bytes=sizes["reserved_bytes"], shared_with=tuple(sorted(set(shared))),
                slot_source=str(getattr(buffer, "slot_source", "") or "")))
            occupied.append((base, base + allocation, str(buffer.root), interval))

    usage = {}
    peak = 0
    for space in sorted(by_space):
        members = [entry for entry in placed if entry.space == space]
        capacity, known = _capacity_of(space, capacities)
        high = max([0] + [entry.range()[1] for entry in members] +
                   [entry.range()[0] for entry in members])
        # PEAK IS THE MAXIMUM LIVE FOOTPRINT, not the highest address: a hole left by a
        # short-lived buffer is not live, and reporting the highest address would overstate
        # usage for exactly the programs that reuse well.
        high = _peak_live(members)
        reused = sum(entry.allocation_bytes for entry in members if entry.shared_with)
        usage[space] = SpaceUsage(space=space, capacity_bytes=capacity, peak_bytes=high,
                                  entries=tuple(members), capacity_known=known,
                                  reused_bytes=reused)
        peak = max(peak, high)

    report = verify_placement(placed, usage, diagnostics=diagnostics)
    versioning = {entry.root: {"slot_count": entry.slot_count,
                               "version_stride_bytes": entry.version_stride_bytes}
                  for entry in placed}
    input_digest = _digest([{key: value for key, value in entry.as_dict().items()
                             if key not in ("base_offset_bytes", "shared_with")}
                            for entry in placed] + [versioning])
    return StoragePlacement(schema=STORAGE_SCHEMA, entries=tuple(placed), report=report,
                            input_digest=input_digest, facts=facts, versioning=versioning,
                            completion_digest=completion_digest,
                            diagnostics=tuple(diagnostics))


def _peak_live(entries) -> int:
    """The maximum number of bytes simultaneously live, computed from the intervals."""
    if not entries:
        return 0
    events = []
    for entry in entries:
        events.append((entry.interval.start, entry.allocation_bytes))
        events.append((entry.interval.stop, -entry.allocation_bytes))
    events.sort(key=lambda pair: (pair[0], pair[1]))
    live = 0
    high = 0
    for _point, delta in events:
        live += delta
        high = max(high, live)
    return high


def verify_placement(entries, usage, *, diagnostics=()) -> PlacementReport:
    """Re-check a computed placement. Every rule names itself in the refusal code."""
    errors = []
    for space, item in usage.items():
        if not item.fits:
            errors.append(("STORAGE_CAPACITY_EXCEEDED",
                           "%s peak %d bytes exceeds capacity %d bytes"
                           % (space, item.peak_bytes, item.capacity_bytes)))
    for index, left in enumerate(entries):
        for right in entries[index + 1:]:
            if left.space != right.space:
                continue
            base_l, end_l = left.range()
            base_r, end_r = right.range()
            if not (base_l < end_r and base_r < end_l):
                continue
            if left.interval.overlaps(right.interval):
                errors.append((
                    "STORAGE_ALIAS_ILLEGAL",
                    "%r [%d,%d) overlaps %r [%d,%d) in %s while their live intervals"
                    " [%d,%d) and [%d,%d) also overlap"
                    % (left.root, base_l, end_l, right.root, base_r, end_r, left.space,
                       left.interval.start, left.interval.stop,
                       right.interval.start, right.interval.stop)))
            elif left.owner != right.owner and not _owners_may_share(left, right):
                errors.append((
                    "STORAGE_CROSS_OWNER_ALIAS",
                    "%r (%s) and %r (%s) share bytes in %s; different owners must not alias"
                    % (left.root, left.owner, right.root, right.owner, left.space)))
    for entry in entries:
        alignment = int(SPACE_FACTS[entry.space].alignment or 1)
        if entry.base_offset_bytes % alignment:
            errors.append(("STORAGE_ALIGNMENT_INVALID",
                           "%r base %d is not aligned to %d"
                           % (entry.root, entry.base_offset_bytes, alignment)))
        if entry.allocation_bytes <= 0:
            errors.append(("STORAGE_EMPTY_ALLOCATION",
                           "%r occupies no bytes; a zero-size placement means the footprint"
                           " was never computed" % (entry.root,)))
        if entry.version_stride_bytes < entry.payload_bytes:
            errors.append(("STORAGE_STRIDE_TOO_SMALL",
                           "%r stride %d cannot hold payload %d"
                           % (entry.root, entry.version_stride_bytes, entry.payload_bytes)))
    if errors:
        code, detail = errors[0]
        return PlacementReport(STATUS_REFUSED, code,
                               detail + ("" if len(errors) == 1 else
                                         " (+%d more violation(s))" % (len(errors) - 1)),
                               spaces=usage, entry_count=len(entries),
                               peak_bytes=max([0] + [u.peak_bytes for u in usage.values()]))
    return PlacementReport(STATUS_PLACED, "", "", spaces=usage, entry_count=len(entries),
                           peak_bytes=max([0] + [u.peak_bytes for u in usage.values()]))


def _owners_may_share(left, right) -> bool:
    """Whether two owners are the same allocation identity.

    UB is per-AIV-instance, so two different AIV owners at one offset are different storage
    and must not alias. GM is device-global, so its single owner is shared by construction.
    """
    if left.space == "ascend.gm":
        return True
    return left.owner == right.owner


__all__ = ["plan_storage", "verify_placement", "intervals_from_completion", "layout_of",
           "FRACTAL_SPACES", "STATUS_PLACED", "STATUS_REFUSED"]
