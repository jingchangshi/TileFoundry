"""M6/P5 physical address / layout stage check.

Assigns each buffer a physical base in its memory space from its byte size, per-slot
alignment and version count (a ring buffer occupies versions*slot_size bytes), then
rejects POST-LAYOUT HAZARDS:
  * a buffer placed in an unallocatable space (e.g. ascend.l2 is an implicit cache),
  * a base not aligned to the buffer's alignment,
  * two distinct buffers whose physical byte ranges overlap in the same space,
  * a space footprint exceeding its usable/nominal capacity.

LIMIT: this builds a NEW deterministic placement from buffer sizes/alignment/versions and
IGNORES the plan. It therefore does NOT check the addresses that will actually be placed;
it cannot catch a hazard specific to the real placement. It is a structural feasibility
check, not verification of placed addresses. Every error is tagged with the stable
rule name POST-LAYOUT-HAZARD so one pinned message owns the case.
"""
from __future__ import annotations

from ..memory import SPACE_FACTS

RULE = "M6-POST-LAYOUT-HAZARD"


def _align_up(x: int, a: int) -> int:
    if a <= 1:
        return x
    return ((x + a - 1) // a) * a


def verify_layout(problem, plan=None) -> dict:
    errors = []
    layout = {}
    by_space = {}
    for b in problem.buffers:
        by_space.setdefault(b.space, []).append(b)

    for space, bufs in by_space.items():
        facts = SPACE_FACTS.get(space)
        if facts is None:
            errors.append("[%s] unknown memory space %r" % (RULE, space))
            continue
        if not facts.allocatable:
            errors.append("[%s] space %r is not allocatable (implicit cache); %d buffer(s) placed there"
                          % (RULE, space, len(bufs)))
            continue
        salign = facts.alignment or 1
        cursor = 0
        entries = []
        # Deterministic placement: largest first, then root name.
        for b in sorted(bufs, key=lambda z: (-(max(1, z.versions) * max(0, z.bytes_)), z.root)):
            balign = max(b.alignment or 1, salign)
            base = _align_up(cursor, balign)
            if b.alignment and base % b.alignment != 0:
                errors.append("[%s] %s base %d not aligned to %d" % (RULE, b.root, base, b.alignment))
            slot = _align_up(max(0, b.bytes_), salign)
            size = slot * max(1, b.versions)
            if size <= 0:
                errors.append("[%s] %s has nonpositive byte size" % (RULE, b.root))
            entries.append({"root": b.root, "base": base, "size": size,
                            "slot": slot, "versions": max(1, b.versions)})
            cursor = base + size
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                r1, r2 = entries[i], entries[j]
                if r1["base"] < r2["base"] + r2["size"] and r2["base"] < r1["base"] + r1["size"]:
                    errors.append("[%s] physical ranges overlap: %s[%d:%d] and %s[%d:%d]"
                                  % (RULE, r1["root"], r1["base"], r1["base"] + r1["size"],
                                     r2["root"], r2["base"], r2["base"] + r2["size"]))
        cap = facts.usable_bytes or facts.nominal_bytes
        if cap is not None and cursor > cap:
            errors.append("[%s] %s footprint %d > capacity %d" % (RULE, space, cursor, cap))
        layout[space] = {"alignment": salign, "entries": entries,
                         "footprint": cursor, "capacity": cap}
    return {"ok": not errors, "errors": errors, "layout": layout, "rule": RULE}


__all__ = ["verify_layout", "RULE"]
