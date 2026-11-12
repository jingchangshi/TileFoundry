"""Physical storage planning: what a buffer OCCUPIES, and when it may be reused.

WHY THIS MODULE EXISTS. The pre-P4 layout stage assigned each buffer a base offset from
its byte size and its version count, ordered largest-first. That answers "can these
buffers coexist" and nothing else. It cannot answer the two questions G41 is about:

  * WHAT SIZE a buffer really occupies. A tile in L0A/L0B is stored in a fractal (NZ)
    layout, not row-major, so its footprint is not rows*cols*itemsize. An FP32
    accumulation buffer holds 4 bytes per lane even when the operand dtype is 2. A
    backend that stages a conversion needs scratch beside the result. A region may
    reserve bytes it never exposes as a buffer. Multiplying the bare element bytes and
    calling that the footprint is an estimate, and it is routinely too small.

  * WHEN a buffer's memory becomes reusable. Two buffers may share bytes only if their
    live intervals do not overlap. Those intervals are decided by the COMPLETION
    program, not by instruction issue order: an output staging buffer is not reusable
    when its MTE3 store is ISSUED, only when that store's read of the buffer has
    COMPLETED. Reusing on issue is the defect; serializing everything is the other
    defect, and the spec forbids both.

So this package computes a real, versioned, completion-aware placement and verifies it.
Nothing here invents a hardware number: every byte count comes from the plan's declared
sizes, the dtype registry, or a DECLARED layout fact, and an unknown fact is REFUSED
rather than defaulted.
"""
from .model import (
    ALIGN_RULE,
    LAYOUT_FACTS,
    LAYOUT_FRACTAL_NZ,
    LAYOUT_LINEAR,
    LAYOUT_RESERVED,
    LAYOUT_SCRATCH,
    STORAGE_SCHEMA,
    LayoutRefusal,
    LayoutRequest,
    StorageFacts,
    StorageRefusal,
    align_up,
    bytes_for,
    dtype_bytes,
    facts_for,
    layout_fact,
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
from .plan_placement import (
    FRACTAL_SPACES,
    intervals_from_completion,
    layout_of,
    plan_storage,
    verify_placement,
)
from .plan_storage import (
    STORAGE_PLACEMENT_KEY,
    StoragePlan,
    attach_storage_plan,
    build_storage_plan,
    require_storage_verified,
)
from .verify import (
    STATUS_FAILED,
    STATUS_VERIFIED,
    StorageVerdict,
    invalidated_inputs,
    verify_storage,
)

__all__ = [
    "ALIGN_RULE", "LAYOUT_FACTS", "LAYOUT_FRACTAL_NZ", "LAYOUT_LINEAR", "LAYOUT_RESERVED",
    "LAYOUT_SCRATCH", "STORAGE_SCHEMA", "LayoutRefusal", "LayoutRequest", "StorageFacts",
    "StorageRefusal", "align_up", "bytes_for", "dtype_bytes", "facts_for", "layout_fact",
    "STATUS_PLACED", "STATUS_REFUSED", "Interval", "PlacementEntry", "PlacementReport",
    "SpaceUsage", "StoragePlacement",
    "FRACTAL_SPACES", "intervals_from_completion", "layout_of", "plan_storage",
    "verify_placement",
    "STORAGE_PLACEMENT_KEY", "StoragePlan", "attach_storage_plan", "build_storage_plan",
    "require_storage_verified",
    "STATUS_FAILED", "STATUS_VERIFIED", "StorageVerdict", "invalidated_inputs",
    "verify_storage",
]
