"""Versioned, completion-aware physical storage placement.

THE TWO QUESTIONS, AND WHY ISSUE ORDER ANSWERS NEITHER.

A placement is a byte range per buffer per slot, plus the right to let two buffers share
bytes. Sharing is only legal when their LIVE INTERVALS are disjoint. The tempting way to
decide that is instruction issue order: the producer writes here, the consumer reads
there, so after the consumer is issued the producer's memory is free. That is wrong on
this target, and wrong in a way that produces a program that mostly works:

  * The consumer's READ is ISSUED, not COMPLETE. An MTE3 store handed a UB pointer is
    still reading that pointer after the instruction retires from the issue queue. Handing
    those bytes to another buffer at issue time corrupts the output -- silently, and only
    when the timing happens to overlap.
  * Conversely, refusing to reuse anything until the whole region ends is the OTHER
    defect: it serializes a pipeline that does not need serializing, and the spec forbids
    it ("legal short-lifetime reuse must be preserved, not hidden by full serialization").

So the interval of a buffer's slot is closed by its COMPLETION evidence, taken from the
P4-01 completion trace: the slot is live from its producing write until the LAST reader's
read has completed (and for a write-only output, until its store's read completes). That
end point exists in the trace as a read_end / drain record; this module reads it and does
not re-derive it from statement order.

INVALIDATION. Storage placement is a function of (versions, layout, owner, alignment,
completion). Any of those changing makes every previously computed offset meaningless, so
the placement carries a digest of exactly those inputs. A caller holding a stale placement
gets STORAGE_PLACEMENT_STALE, not a silently wrong address.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

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

STATUS_PLACED = "PLACED"
STATUS_REFUSED = "REFUSED"


@dataclass(frozen=True)
class Interval:
    """A slot's LIVE interval, in the completion trace's instance order.

    Both ends are indices into the same order, so comparing two intervals is meaningful
    only within one trace. 'start' is the producing write; 'stop' is exclusive and is set
    from the completion evidence (a completed read), never from an issue.

    'evidence' names the trace record that CLOSED the interval. Keeping it means a
    reviewer can ask "why is this reusable at 41" and get an answer instead of a number.
    """

    start: int
    stop: int
    evidence: str = ""

    def overlaps(self, other: "Interval") -> bool:
        return self.start < other.stop and other.start < self.stop

    def as_dict(self):
        return {"start": self.start, "stop": self.stop, "evidence": self.evidence}


@dataclass(frozen=True)
class PlacementEntry:
    """One buffer's placed bytes."""

    root: str
    space: str
    owner: str
    layout: str
    base_offset_bytes: int
    version_stride_bytes: int
    slot_count: int
    allocation_bytes: int
    interval: Interval
    payload_bytes: int = 0
    padding_bytes: int = 0
    scratch_bytes: int = 0
    staging_bytes: int = 0
    reserved_bytes: int = 0
    shared_with: tuple = ()
    slot_source: str = ""

    def range(self) -> tuple:
        return (self.base_offset_bytes, self.base_offset_bytes + self.allocation_bytes)

    def as_dict(self):
        return {"root": self.root, "space": self.space, "owner": self.owner,
                "layout": self.layout, "base_offset_bytes": self.base_offset_bytes,
                "version_stride_bytes": self.version_stride_bytes,
                "slot_count": self.slot_count, "allocation_bytes": self.allocation_bytes,
                "payload_bytes": self.payload_bytes, "padding_bytes": self.padding_bytes,
                "scratch_bytes": self.scratch_bytes, "staging_bytes": self.staging_bytes,
                "reserved_bytes": self.reserved_bytes,
                "interval": self.interval.as_dict(),
                "shared_with": list(self.shared_with), "slot_source": self.slot_source}


@dataclass(frozen=True)
class SpaceUsage:
    """One memory space's peak and per-buffer breakdown."""

    space: str
    capacity_bytes: Optional[int]
    peak_bytes: int
    entries: tuple = ()
    capacity_known: bool = True
    reused_bytes: int = 0

    @property
    def fits(self) -> bool:
        if self.capacity_bytes is None:
            return True
        return self.peak_bytes <= int(self.capacity_bytes)

    def as_dict(self):
        return {"space": self.space, "capacity_bytes": self.capacity_bytes,
                "peak_bytes": self.peak_bytes, "capacity_known": self.capacity_known,
                "fits": self.fits, "reused_bytes": self.reused_bytes,
                "entries": [entry.as_dict() for entry in self.entries]}


@dataclass(frozen=True)
class PlacementReport:
    status: str
    code: str = ""
    detail: str = ""
    spaces: dict = field(default_factory=dict)
    entry_count: int = 0
    peak_bytes: int = 0
    result_digest: str = ""

    @property
    def placed(self) -> bool:
        return self.status == STATUS_PLACED

    def as_dict(self):
        return {"status": self.status, "code": self.code, "detail": self.detail,
                "spaces": {name: usage.as_dict() for name, usage in self.spaces.items()},
                "entry_count": self.entry_count, "peak_bytes": self.peak_bytes,
                "result_digest": self.result_digest}


@dataclass(frozen=True)
class StoragePlacement:
    """The verified physical placement, plus the inputs it is valid for."""

    schema: str
    entries: tuple
    report: PlacementReport
    input_digest: str
    facts: StorageFacts
    versioning: dict = field(default_factory=dict)
    completion_digest: str = ""
    diagnostics: tuple = ()

    @property
    def verified(self) -> bool:
        return self.report.placed

    @property
    def placements(self) -> dict:
        return {entry.root: entry.as_dict() for entry in self.entries}

    def as_dict(self):
        return {"schema": self.schema, "input_digest": self.input_digest,
                "completion_digest": self.completion_digest,
                "facts": self.facts.as_dict(),
                "entries": [entry.as_dict() for entry in self.entries],
                "report": self.report.as_dict(),
                "diagnostics": list(self.diagnostics)}
