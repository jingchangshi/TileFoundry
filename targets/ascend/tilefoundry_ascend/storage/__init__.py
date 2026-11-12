"""Native post-version physical storage placement (StoragePlan).

Real implementation of the frozen storage surface:

    place_resolved_versions(plan, completion, target, lowering) -> StoragePlan
    verify_storage_current(plan, storage, target) -> StoragePlan (verified)
    require_storage_verified(storage) -> StoragePlan

THE IRON RULE. A capacity is a target FACT, never a footprint echo. When the
target states no capacity for a space the plan places bytes in, placement refuses
with UNKNOWN_CAPACITY. None is never treated as "fits", a requirement is never
clipped to a capacity, and a capacity is never derived from the requirement.

REUSED FACILITIES. Layout/bytes come from storage/model.py (LayoutRequest,
bytes_for, align_up, Breakdown, PhysicalLayout, Requirement); placement records
come from storage/placement.py (PlacementEntry, Interval). This file adds plan
consumption, the per-StorageKey bump allocator, the capacity refusal and the
recomputing verifier. It does not define a second physical answer.

VERSIONS. versions come from BufferPlan.resolved_versions. The V slots of one
root are contiguous with version_stride_bytes == slot_bytes (>= the aligned
single-version requirement); slot k is base + (k mod V) * stride. Different roots
do not overlap in this slice (no alias proof exists yet).

SCRATCH. Scratch allocations come from the lowering requirement argument, one
record per real allocation. An allocation is charged ONCE (copies * aligned
stride added to its space peak once), never once per buffer version. A scratch
allocation names the buffer root it serves; one that names no plan buffer is
refused by name.

STATED GAPS (S0/S1 scope; do not carry past S2):
- completion is not implemented yet. PlacementEntry.interval therefore closes at
  the whole region ("lowering:no-completion") and no cross-root reuse is claimed.
  A lowering-declared LiveWindow is passed through; an empty window is a stated
  gap, not a completion claim.
- OwnerKey.program_group is None: the S0 chain runs one program group and the
  plan carries no group key. Per-program partition is a later-stage fact.
- effects_identity is derived from the real unit effects when the lowering
  provides the unit tree; otherwise it is empty and says so.
- Verification requires the actual completion and lowering and recomputes the
  placement before calling the completion verifier. Completion verification
  currently refuses because the plan lacks source problem and effect facts.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from typing import Optional

from ..plan.units import OwnerKey, StorageKey
from .model import (
    LAYOUT_LINEAR,
    LAYOUT_SCRATCH,
    Breakdown,
    LayoutRequest,
    PhysicalLayout,
    Requirement,
    STATUS_REFERENCE_IMPLEMENTATION,
    align_up,
    bytes_for,
)
from .placement import Interval, PlacementEntry

__all__ = [
    "CAPACITY_UNKNOWN",
    "Capacity",
    "LiveWindow",
    "ScratchAllocation",
    "ScratchPlacementEntry",
    "ScratchRequirement",
    "StorageError",
    "StorageLowering",
    "StoragePlan",
    "StorageVerification",
    "UNKNOWN_CAPACITY",
    "place_resolved_versions",
    "require_storage_verified",
    "verify_storage_current",
]

UNKNOWN_CAPACITY = "UNKNOWN_CAPACITY"
CAPACITY_UNKNOWN = UNKNOWN_CAPACITY
STORAGE_SCHEMA = "native_mainline_storage_plan"
STORAGE_PLAN_SCHEMA = STORAGE_SCHEMA

_TIR_SPACE_TO_TARGET = {
    "gmem": "ascend.gm",
    "smem": "ascend.ub",
    "rmem": "ascend.vreg",
    "carrier": "ascend.preg",
}
_GM_SPACES = ("gmem",)


class StorageError(RuntimeError):
    """A named storage refusal."""

    def __init__(self, code: str, detail: str, subject: str = "") -> None:
        self.code = code
        self.detail = detail
        self.subject = subject
        super().__init__("%s: %s (subject=%r)" % (code, detail, subject))


@dataclass(frozen=True)
class LiveWindow:
    """The completion-backed live window of one scratch allocation."""

    begin_site: str = ""
    end_sites: tuple = ()
    generation: object | None = None
    guard: object | None = None
    happens_before: tuple = ()


@dataclass(frozen=True)
class ScratchRequirement:
    """One real scratch allocation, stated once."""

    scratch_id: str
    storage_key: StorageKey
    layout: PhysicalLayout
    copies: int
    live_window: LiveWindow
    source_unit_id: str = ""
    root_buffer_id: str = ""


@dataclass(frozen=True)
class ScratchAllocation:
    """What the lowering states for one scratch allocation."""

    scratch_id: str
    allocation_ref: object
    space: str
    root_buffer_id: str
    dtype: str = "f32"
    shape: tuple = ()
    lanes: int = 0
    copies: int = 1
    source_unit_id: str = ""
    live_window: Optional[LiveWindow] = None


@dataclass(frozen=True)
class StorageLowering:
    """The lowering-side facts this stage consumes (scratch, optional unit tree)."""

    scratch_allocations: tuple = ()
    unit_tree: object | None = None
    plan_iteration: int = 0


@dataclass(frozen=True)
class ScratchPlacementEntry:
    """One placed scratch allocation (schema section 5)."""

    scratch_id: str
    allocation_ref: object
    storage_key: StorageKey
    base_offset_bytes: int
    copy_stride_bytes: int
    copies: int
    allocation_bytes: int
    layout: PhysicalLayout
    live_window: LiveWindow
    requirement_identity: str


@dataclass(frozen=True)
class Capacity:
    """One known per-StorageKey capacity and the peak placed in it."""

    storage_key: StorageKey
    buffers: tuple
    usable_bytes: int
    reserved_bytes: int
    source: str
    status: str = STATUS_REFERENCE_IMPLEMENTATION


@dataclass(frozen=True)
class StorageVerification:
    """The verifier's own record; never a caller-supplied flag."""

    verified: bool
    plan_identity: str
    storage_identity: str
    requirements_identity: str
    target_identity: str
    detail: str = ""


@dataclass(frozen=True)
class StoragePlan:
    """The complete post-version placement of the plan's buffers and scratch."""

    identity: str = ""
    plan_identity: str = ""
    completion_identity: str = ""
    effects_identity: str = ""
    target_identity: str = ""
    lowering_identity: str = ""
    plan_iteration: int = 0
    requirements: tuple = ()
    entries: tuple = ()
    scratch_entries: tuple = ()
    capacities: tuple = ()
    peak_bytes: tuple = ()
    verification: StorageVerification | None = field(default=None, compare=False)

    @property
    def verified(self) -> bool:
        """A detached status cannot prove current inputs; reverify at the gate."""
        return False

    def as_dict(self):
        return {
            "schema": STORAGE_SCHEMA,
            "identity": self.identity,
            "plan_identity": self.plan_identity,
            "completion_identity": self.completion_identity,
            "effects_identity": self.effects_identity,
            "target_identity": self.target_identity,
            "lowering_identity": self.lowering_identity,
            "plan_iteration": self.plan_iteration,
            "requirements": [
                {
                    "root": _root_text(item.root),
                    "space": item.space,
                    "versions": item.versions,
                    "slot_bytes": item.slot_bytes,
                    "version_stride_bytes": item.version_stride_bytes,
                    "layout": item.layout.as_dict(),
                    "extra_scratch": [scratch.scratch_id for scratch in item.extra_scratch],
                }
                for item in self.requirements
            ],
            "entries": [_entry_identity(item) for item in self.entries],
            "scratch_entries": [_scratch_entry_identity(item)
                                for item in self.scratch_entries],
            "capacities": [
                {
                    "storage_key": _key_text(item.storage_key),
                    "buffers": list(item.buffers),
                    "usable_bytes": item.usable_bytes,
                    "reserved_bytes": item.reserved_bytes,
                    "source": item.source,
                    "status": item.status,
                }
                for item in self.capacities
            ],
            "peak_bytes": [[_key_text(key), value] for key, value in self.peak_bytes],
        }


def _digest(payload) -> str:
    text = json.dumps(payload, sort_keys=True, default=str)
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def _key_text(key: StorageKey) -> str:
    return "%s|%s" % (key.space, getattr(key.owner, "core_kind", ""))


def _root_text(root) -> str:
    return str(getattr(root, "name", root))


def _target_memory(target):
    from tilefoundry.analysis.facts import MemoryHierarchyFacts

    try:
        return target.get_facts(MemoryHierarchyFacts)
    except Exception as error:  # a missing fact is a refusal, never a default
        raise StorageError(
            "STORAGE_TARGET_FACTS_UNKNOWN",
            "the target states no memory hierarchy; capacity cannot be known: %s" % (error,),
            type(target).__name__,
        ) from None


def _level(target, space: str):
    memory = _target_memory(target)
    level = memory.explicit(str(space))
    if level is None:
        raise StorageError(
            "STORAGE_SPACE_UNKNOWN",
            "the target states no memory level named %r, so no capacity or owner is known"
            % (space,),
            str(space),
        )
    return level


def _alignment(target, space: str) -> int:
    facts = target.space_facts().get(_TIR_SPACE_TO_TARGET.get(str(space), str(space)))
    if facts is None or facts.alignment is None:
        raise StorageError(
            "STORAGE_ALIGNMENT_UNKNOWN",
            "the target states no alignment for space %r" % (space,),
            str(space),
        )
    return int(facts.alignment)


def _owner_key(target, space: str) -> OwnerKey:
    owner = str(_level(target, space).owner)
    core_kind = "AIC" if owner == "aic" else ("AIV" if owner == "aiv" else "DEVICE")
    return OwnerKey(program_group=None, core_kind=core_kind, sid=None)


def _buffer_type(buffer_ref):
    from tilefoundry.ir.types import TensorType

    tensor_type = getattr(buffer_ref, "type", None)
    if not isinstance(tensor_type, TensorType):
        raise StorageError(
            "STORAGE_BUFFER_TYPE",
            "a plan buffer does not reference a real tensor-typed root Var",
            type(buffer_ref).__name__,
        )
    return tensor_type


def _static_shape(tensor_type) -> tuple:
    from tilefoundry.ir.types.utils import static_dim_value

    shape = []
    for dim in tuple(getattr(tensor_type, "shape", ()) or ()):
        value = static_dim_value(dim)
        if value is None:
            raise StorageError(
                "STORAGE_SHAPE_UNKNOWN",
                "a buffer shape dimension is not a static integer",
                str(tensor_type),
            )
        shape.append(int(value))
    return tuple(shape)


def _plan_buffers(plan):
    ordered = []
    seen = {}
    for region in getattr(plan, "regions", ()) or ():
        for buffer in getattr(region, "buffers", ()) or ():
            previous = seen.get(id(buffer.buffer_ref))
            if previous is not None:
                if int(previous.resolved_versions) != int(buffer.resolved_versions):
                    raise StorageError(
                        "STORAGE_VERSION_CONFLICT",
                        "one root has different resolved versions across regions",
                        str(buffer.buffer_id),
                    )
                continue
            versions = int(getattr(buffer, "resolved_versions", 0) or 0)
            if versions < 1:
                raise StorageError(
                    "STORAGE_VERSION_INVALID",
                    "a plan buffer has no positive resolved version count",
                    str(buffer.buffer_id),
                )
            seen[id(buffer.buffer_ref)] = buffer
            ordered.append(buffer)
    return tuple(ordered)


def _capacity_fact(target, space: str, *, gmem_exempt: bool):
    level = _level(target, space)
    capacity = level.capacity_bytes
    if capacity is None:
        if gmem_exempt and str(space) in _GM_SPACES:
            return None
        raise StorageError(
            UNKNOWN_CAPACITY,
            "the target states no capacity for space %r; a requirement must not be "
            "backfilled as a capacity" % (space,),
            str(space),
        )
    if int(capacity) <= 0:
        raise StorageError(
            UNKNOWN_CAPACITY,
            "the target states a non-positive capacity for space %r" % (space,),
            str(space),
        )
    return int(capacity)


def _physical_layout(buffer, target) -> tuple:
    tensor_type = _buffer_type(buffer.buffer_ref)
    space = getattr(tensor_type.storage, "name", str(tensor_type.storage)).lower()
    shape = _static_shape(tensor_type)
    dtype_name = getattr(tensor_type.dtype, "name", str(tensor_type.dtype))
    alignment = _alignment(target, space)
    request = LayoutRequest(
        root=str(buffer.buffer_id),
        shape=shape,
        dtype=dtype_name,
        layout=LAYOUT_LINEAR,
        alignment=alignment,
        source="plan.BufferPlan buffer_ref TensorType",
    )
    sized = bytes_for(request)
    single = int(sized["single_version_bytes"])
    slot = align_up(single, alignment)
    payload = int(sized["payload_bytes"])
    breakdown = Breakdown(
        payload_bytes=payload,
        padding_bytes=int(sized["padding_bytes"]) + (slot - single),
        scratch_bytes=0,
        staging_bytes=int(sized["staging_bytes"]),
        reserved_bytes=int(sized["reserved_bytes"]),
    )
    total = (breakdown.payload_bytes + breakdown.padding_bytes
             + breakdown.scratch_bytes + breakdown.staging_bytes
             + breakdown.reserved_bytes)
    if total != slot:
        raise StorageError(
            "STORAGE_BREAKDOWN_MISMATCH",
            "the layout breakdown does not sum to the aligned requirement",
            str(buffer.buffer_id),
        )
    layout = PhysicalLayout(
        layout_kind="ND",
        logical_shape=shape,
        physical_shape=shape,
        dtype=dtype_name,
        item_bytes=int(sized["item_bytes"]),
        pitch_bytes=payload,
        alignment_bytes=alignment,
        required_bytes=slot,
        breakdown=breakdown,
        source="bytes_for(LayoutRequest(layout=linear)); align_up(alignment=%d)" % (alignment,),
        status=STATUS_REFERENCE_IMPLEMENTATION,
    )
    return layout, space


def _target_identity(plan, target) -> str:
    """Bind placement to the actual target, checking the problem target identity."""
    from dataclasses import asdict
    from tilefoundry.target.facts import TopologyFacts

    plan_target = str(getattr(plan, "target_identity", "") or "")
    from ..plan.problem import _target_identity as problem_target_identity
    actual_plan_target = problem_target_identity(target)
    if not plan_target or plan_target != actual_plan_target:
        raise StorageError("STORAGE_TARGET_IDENTITY_MISMATCH",
                           "the actual target does not match the plan target identity")
    facts = target.space_facts()
    return _digest({
        "target": actual_plan_target,
        "memory": asdict(_target_memory(target)),
        "topology": asdict(target.get_facts(TopologyFacts)),
        "spaces": {name: fact.as_dict() for name, fact in sorted(facts.items())},
    })


def _plan_requirements(plan, target):
    """(buffer_id, Requirement) for every unique plan buffer, in plan order."""
    pairs = []
    for buffer in _plan_buffers(plan):
        layout, space = _physical_layout(buffer, target)
        pairs.append((str(buffer.buffer_id), Requirement(
            root=getattr(buffer, "buffer_ref", None),
            owner=_owner_key(target, space),
            space=space,
            versions=int(buffer.resolved_versions),
            layout=layout,
            version_stride_bytes=layout.required_bytes,
            slot_bytes=layout.required_bytes,
            extra_scratch=(),
            capacity_source=("GM ABI (not ring-allocated)"
                             if space in _GM_SPACES
                             else "target.memory_hierarchy.%s.capacity_bytes" % (space,)),
            completion_edge_ids=(),
            reuse_group="",
            invalidation_key="",
            target_profile_digest=_target_identity(plan, target),
        )))
    return tuple(pairs)


def _effects_identity(lowering) -> str:
    unit_tree = getattr(lowering, "unit_tree", None)
    if unit_tree is None:
        return ""
    rows = []
    for unit in getattr(unit_tree, "units", ()) or ():
        for effect in getattr(unit, "effects", ()) or ():
            rows.append((effect.effect_id, effect.kind, effect.pipe,
                         tuple(effect.access_ids), effect.completion))
    return _digest(rows)


def _lowering_identity(lowering) -> str:
    unit_tree = getattr(lowering, "unit_tree", None)
    payload = []
    for allocation in getattr(lowering, "scratch_allocations", ()) or ():
        payload.append({
            "scratch_id": allocation.scratch_id,
            "root_buffer_id": allocation.root_buffer_id,
            "space": allocation.space,
            "dtype": allocation.dtype,
            "shape": tuple(allocation.shape),
            "lanes": int(allocation.lanes),
            "copies": int(allocation.copies),
            "source_unit_id": allocation.source_unit_id,
        })
    payload.sort(key=lambda row: row["scratch_id"])
    return _digest({
        "scratch": payload,
        "unit_source": getattr(getattr(unit_tree, "source_snapshot", None), "identity", ""),
    })


def _requirement_identity(requirement) -> str:
    return _digest({
        "root": _root_text(requirement.root),
        "space": requirement.space,
        "owner": _key_text(StorageKey(requirement.space, requirement.owner)),
        "versions": requirement.versions,
        "layout": requirement.layout.as_dict(),
        "version_stride_bytes": requirement.version_stride_bytes,
        "slot_bytes": requirement.slot_bytes,
        "extra_scratch": [item.scratch_id for item in requirement.extra_scratch],
    })


def _scratch_requirement(allocation, target) -> ScratchRequirement:
    space = str(allocation.space)
    alignment = _alignment(target, space)
    shape = tuple(int(dim) for dim in allocation.shape)
    lanes = int(allocation.lanes)
    copies = int(allocation.copies)
    if not shape and lanes <= 0:
        raise StorageError(
            "STORAGE_SCRATCH_UNSPECIFIED",
            "a scratch allocation states neither a shape nor lanes",
            str(allocation.scratch_id),
        )
    if copies < 1:
        raise StorageError("STORAGE_SCRATCH_UNSPECIFIED",
                           "scratch copies must be positive", str(allocation.scratch_id))
    request = LayoutRequest(
        root=str(allocation.scratch_id),
        shape=shape,
        dtype=str(allocation.dtype),
        layout=LAYOUT_SCRATCH,
        scratch_dtype=str(allocation.dtype),
        scratch_lanes=lanes,
        alignment=alignment,
        source="lowering.scratch_allocations",
    )
    sized = bytes_for(request)
    single = int(sized["single_version_bytes"])
    stride = align_up(single, alignment)
    layout = PhysicalLayout(
        layout_kind="SCRATCH",
        logical_shape=shape,
        physical_shape=shape,
        dtype=str(allocation.dtype),
        item_bytes=int(sized["item_bytes"]),
        pitch_bytes=int(sized["payload_bytes"]),
        alignment_bytes=alignment,
        required_bytes=stride,
        breakdown=Breakdown(
            payload_bytes=int(sized["payload_bytes"]),
            padding_bytes=int(sized["padding_bytes"]) + (stride - single),
            scratch_bytes=int(sized["scratch_bytes"]),
            staging_bytes=int(sized["staging_bytes"]),
            reserved_bytes=int(sized["reserved_bytes"]),
        ),
        source="lowering.scratch allocation; bytes_for(layout=scratch)",
        status=STATUS_REFERENCE_IMPLEMENTATION,
    )
    return ScratchRequirement(
        scratch_id=str(allocation.scratch_id),
        storage_key=StorageKey(space, _owner_key(target, space)),
        layout=layout,
        copies=copies,
        live_window=allocation.live_window or LiveWindow(),
        source_unit_id=str(allocation.source_unit_id),
        root_buffer_id=str(allocation.root_buffer_id),
    )


def _bump(spaces: dict, key: StorageKey, size: int, alignment: int) -> int:
    base = align_up(spaces.get(key, 0), alignment)
    spaces[key] = base + size
    return base


def _scratch_by_root(lowering, plan, target):
    """{buffer_id: (ScratchRequirement,...)} with one record per real allocation."""
    buffer_ids = {str(buffer.buffer_id) for buffer in _plan_buffers(plan)}
    grouped = {}
    for allocation in getattr(lowering, "scratch_allocations", ()) or ():
        root_buffer_id = str(getattr(allocation, "root_buffer_id", ""))
        if not root_buffer_id:
            raise StorageError(
                "STORAGE_SCRATCH_UNSPECIFIED",
                "a scratch allocation does not name the buffer root it serves",
                str(allocation.scratch_id),
            )
        if root_buffer_id not in buffer_ids:
            raise StorageError(
                "STORAGE_SCRATCH_UNKNOWN_ROOT",
                "a scratch allocation names no buffer of this plan",
                "%s -> %s" % (allocation.scratch_id, root_buffer_id),
            )
        grouped.setdefault(root_buffer_id, []).append(_scratch_requirement(allocation, target))
    for rows in grouped.values():
        rows.sort(key=lambda item: item.scratch_id)
    return {key: tuple(rows) for key, rows in grouped.items()}


def _with_scratch(pairs, scratch_by_root):
    attached = []
    for buffer_id, requirement in pairs:
        candidate = replace(requirement,
                            extra_scratch=tuple(scratch_by_root.get(buffer_id, ())))
        attached.append((buffer_id,
                         replace(candidate,
                                 invalidation_key=_requirement_identity(candidate))))
    return tuple(attached)


def _scratch_ref(lowering, scratch_id):
    for allocation in getattr(lowering, "scratch_allocations", ()) or ():
        if allocation.scratch_id == scratch_id:
            return allocation.allocation_ref
    return None


def _scratch_identity(scratch) -> str:
    return _digest({"scratch": scratch.scratch_id,
                    "stride": scratch.layout.required_bytes,
                    "copies": scratch.copies,
                    "space": scratch.storage_key.space})


def _entry_identity(entry) -> tuple:
    return (_root_text(entry.root), entry.base_offset_bytes, entry.version_stride_bytes,
            entry.slot_count, entry.allocation_bytes)


def _scratch_entry_identity(entry) -> tuple:
    return (entry.scratch_id, entry.base_offset_bytes, entry.copy_stride_bytes,
            entry.copies, entry.allocation_bytes)


def place_resolved_versions(plan, completion, target, lowering):
    """Place each buffer's versions and the lowering's scratch; refuse unknown capacity."""
    if plan is None:
        raise StorageError("STORAGE_PLAN_MISSING", "place_resolved_versions needs a plan")
    if target is None:
        raise StorageError("STORAGE_TARGET_MISSING", "place_resolved_versions needs a target")
    lowering = lowering if lowering is not None else StorageLowering()

    pairs = _with_scratch(_plan_requirements(plan, target),
                          _scratch_by_root(lowering, plan, target))
    requirements = tuple(requirement for _buffer_id, requirement in pairs)

    spaces = {}
    entries = []
    buffers_by_key = {}
    for buffer_id, requirement in pairs:
        key = StorageKey(requirement.space, requirement.owner)
        if requirement.space in _GM_SPACES:
            continue
        base = _bump(spaces, key, requirement.version_stride_bytes * requirement.versions,
                     requirement.layout.alignment_bytes)
        entries.append(PlacementEntry(
            root=requirement.root,
            space=requirement.space,
            owner=requirement.owner.core_kind,
            layout="ND",
            base_offset_bytes=base,
            version_stride_bytes=requirement.version_stride_bytes,
            slot_count=requirement.versions,
            allocation_bytes=requirement.version_stride_bytes * requirement.versions,
            interval=Interval(0, max(1, requirement.versions),
                              evidence="lowering:no-completion"),
            payload_bytes=requirement.layout.breakdown.payload_bytes,
            padding_bytes=requirement.layout.breakdown.padding_bytes,
            scratch_bytes=requirement.layout.breakdown.scratch_bytes,
            staging_bytes=requirement.layout.breakdown.staging_bytes,
            reserved_bytes=requirement.layout.breakdown.reserved_bytes,
            layout_descriptor=requirement.layout,
            extra_scratch=requirement.extra_scratch,
            capacity_source=requirement.capacity_source,
            completion_proof_id="",
            owner_key=requirement.owner,
            target_profile_digest=requirement.target_profile_digest,
            plan_iteration=int(getattr(lowering, "plan_iteration", 0) or 0),
            slot_source="plan.BufferPlan.resolved_versions",
        ))
        buffers_by_key.setdefault(key, []).append(buffer_id)

    scratch_entries = []
    all_scratch = sorted(
        (scratch for rows in _scratch_by_root(lowering, plan, target).values()
         for scratch in rows),
        key=lambda item: item.scratch_id,
    )
    for scratch in all_scratch:
        key = scratch.storage_key
        stride = scratch.layout.required_bytes
        base = _bump(spaces, key, stride * scratch.copies,
                     scratch.layout.alignment_bytes)
        scratch_entries.append(ScratchPlacementEntry(
            scratch_id=scratch.scratch_id,
            allocation_ref=_scratch_ref(lowering, scratch.scratch_id),
            storage_key=key,
            base_offset_bytes=base,
            copy_stride_bytes=stride,
            copies=scratch.copies,
            allocation_bytes=stride * scratch.copies,
            layout=scratch.layout,
            live_window=scratch.live_window,
            requirement_identity=_scratch_identity(scratch),
        ))

    capacities = []
    peak_bytes = []
    for key, peak in spaces.items():
        capacity = _capacity_fact(target, key.space, gmem_exempt=False)
        if peak > capacity:
            raise StorageError(
                "STORAGE_CAPACITY_EXCEEDED",
                "the placed peak exceeds the target capacity; the requirement is not clipped",
                "%s peak=%d capacity=%d" % (key.space, peak, capacity),
            )
        capacities.append(Capacity(
            storage_key=key,
            buffers=tuple(buffers_by_key.get(key, ())),
            usable_bytes=capacity,
            reserved_bytes=0,
            source="target.memory_hierarchy.%s.capacity_bytes" % (key.space,),
        ))
        peak_bytes.append((key, peak))

    target_identity = _target_identity(plan, target)
    completion_identity = str(getattr(completion, "identity", "") or "") if completion else ""
    effects_identity = _effects_identity(lowering)
    lowering_identity = _lowering_identity(lowering)
    plan_identity = str(getattr(plan, "plan_identity", ""))
    payload = {
        "plan": plan_identity,
        "completion": completion_identity,
        "effects": effects_identity,
        "target": target_identity,
        "lowering": lowering_identity,
        "requirements": [item.invalidation_key for item in requirements],
        "entries": [_entry_identity(item) for item in entries],
        "scratch": [_scratch_entry_identity(item) for item in scratch_entries],
        "capacities": [(_key_text(item.storage_key), item.usable_bytes) for item in capacities],
        "peak": [(_key_text(key), value) for key, value in peak_bytes],
    }
    return StoragePlan(
        identity=_digest(payload),
        plan_identity=plan_identity,
        completion_identity=completion_identity,
        effects_identity=effects_identity,
        target_identity=target_identity,
        lowering_identity=lowering_identity,
        plan_iteration=int(getattr(lowering, "plan_iteration", 0) or 0),
        requirements=requirements,
        entries=tuple(entries),
        scratch_entries=tuple(scratch_entries),
        capacities=tuple(capacities),
        peak_bytes=tuple(peak_bytes),
        verification=None,
    )


def verify_storage_current(plan, storage, target, *, completion=None, lowering=None):
    """Recompute placement against the current plan, target, completion and lowering."""
    if not isinstance(storage, StoragePlan):
        raise StorageError("STORAGE_PLAN_UNKNOWN",
                           "verify_storage_current needs the placed StoragePlan",
                           type(storage).__name__)
    if plan is None or target is None:
        raise StorageError("STORAGE_VERIFY_INPUTS",
                           "verification needs the plan and the target it was placed for")
    plan_identity = str(getattr(plan, "plan_identity", ""))
    if storage.plan_identity != plan_identity:
        raise StorageError("STORAGE_PLAN_IDENTITY_MISMATCH",
                           "the attached storage was placed for a different plan",
                           storage.plan_identity)
    target_identity = _target_identity(plan, target)
    if storage.target_identity != target_identity:
        raise StorageError("STORAGE_TARGET_IDENTITY_MISMATCH",
                           "the attached storage was placed for a different target",
                           storage.target_identity)

    recomputed = _plan_requirements(plan, target)
    attached = {id(item.root): item for item in storage.requirements}
    if len(recomputed) != len(storage.requirements):
        raise StorageError(
            "STORAGE_REQUIREMENT_MISMATCH",
            "the attached requirements do not match the plan's real buffer set",
            "%d != %d" % (len(recomputed), len(storage.requirements)),
        )
    for buffer_id, requirement in recomputed:
        stored = attached.get(id(requirement.root))
        if stored is None:
            raise StorageError("STORAGE_REQUIREMENT_MISMATCH",
                               "a plan buffer has no attached requirement", buffer_id)
        if stored.owner != requirement.owner:
            raise StorageError("STORAGE_OWNER_MISMATCH",
                               "the attached owner disagrees with the target memory fact",
                               buffer_id)
        if (stored.space, stored.versions, stored.slot_bytes,
                stored.version_stride_bytes) != (
                requirement.space, requirement.versions, requirement.slot_bytes,
                requirement.version_stride_bytes):
            raise StorageError(
                "STORAGE_REQUIREMENT_MISMATCH",
                "the attached requirement disagrees with the recomputed layout/versions",
                buffer_id,
            )
        if stored.layout.as_dict() != requirement.layout.as_dict():
            raise StorageError("STORAGE_LAYOUT_MISMATCH",
                               "the attached physical layout disagrees with the recomputed one",
                               buffer_id)

    attached_scratch = {item.scratch_id: item for item in storage.scratch_entries}
    if len(attached_scratch) != len(storage.scratch_entries):
        raise StorageError("STORAGE_SCRATCH_DUPLICATE",
                           "an attached scratch id appears twice")
    expected_scratch = {}
    for requirement in storage.requirements:
        for scratch in requirement.extra_scratch:
            if scratch.scratch_id in expected_scratch:
                raise StorageError("STORAGE_SCRATCH_DUPLICATE",
                                   "one scratch allocation is attached twice",
                                   scratch.scratch_id)
            expected_scratch[scratch.scratch_id] = scratch
    if set(expected_scratch) != set(attached_scratch):
        raise StorageError(
            "STORAGE_SCRATCH_MISMATCH",
            "the attached scratch entries do not match the attached requirements",
            ",".join(sorted(set(expected_scratch) ^ set(attached_scratch))),
        )
    for scratch_id, scratch in expected_scratch.items():
        entry = attached_scratch[scratch_id]
        if (entry.copy_stride_bytes != scratch.layout.required_bytes
                or entry.copies != scratch.copies
                or entry.allocation_bytes != scratch.layout.required_bytes * scratch.copies
                or entry.storage_key != scratch.storage_key):
            raise StorageError("STORAGE_SCRATCH_MISMATCH",
                               "the attached scratch bytes disagree with its requirement",
                               scratch_id)

    spaces = {}
    expected_entries = {}
    buffers_by_key = {}
    for buffer_id, requirement in recomputed:
        key = StorageKey(requirement.space, requirement.owner)
        if requirement.space in _GM_SPACES:
            continue
        base = _bump(spaces, key, requirement.version_stride_bytes * requirement.versions,
                     requirement.layout.alignment_bytes)
        expected_entries[id(requirement.root)] = (base, requirement, buffer_id)
        buffers_by_key.setdefault(key, []).append(buffer_id)
    expected_scratch_bases = {}
    for scratch in sorted(expected_scratch.values(), key=lambda item: item.scratch_id):
        key = scratch.storage_key
        base = _bump(spaces, key, scratch.layout.required_bytes * scratch.copies,
                     scratch.layout.alignment_bytes)
        expected_scratch_bases[scratch.scratch_id] = base
        buffers_by_key.setdefault(key, [])

    entries = {id(item.root): item for item in storage.entries}
    if set(entries) != set(expected_entries):
        raise StorageError("STORAGE_ENTRY_COUNT_MISMATCH",
                           "the attached placement entry set does not match the plan",
                           "%d != %d" % (len(entries), len(expected_entries)))
    for root_id, (base, requirement, buffer_id) in expected_entries.items():
        entry = entries[root_id]
        subject = "%s base=%s" % (buffer_id, base)
        if entry.base_offset_bytes != base:
            raise StorageError("STORAGE_PLACEMENT_MISMATCH",
                               "the attached base offset is not the recomputed offset",
                               subject)
        if entry.version_stride_bytes != requirement.version_stride_bytes:
            raise StorageError("STORAGE_STRIDE_MISMATCH",
                               "the attached version stride disagrees with the requirement",
                               subject)
        if entry.slot_count != requirement.versions:
            raise StorageError("STORAGE_VERSION_MISMATCH",
                               "the attached slot count is not the plan's resolved versions",
                               subject)
        if entry.allocation_bytes != requirement.version_stride_bytes * requirement.versions:
            raise StorageError("STORAGE_BYTES_MISMATCH",
                               "the attached allocation bytes disagree with versions*stride",
                               subject)
        if entry.base_offset_bytes % requirement.layout.alignment_bytes:
            raise StorageError("STORAGE_ALIGNMENT_MISMATCH",
                               "the attached base offset is not aligned for its layout",
                               subject)
    for scratch_id, base in expected_scratch_bases.items():
        if attached_scratch[scratch_id].base_offset_bytes != base:
            raise StorageError("STORAGE_PLACEMENT_MISMATCH",
                               "the attached scratch base offset is not the recomputed one",
                               "%s base=%s" % (scratch_id, base))

    attached_capacities = {_key_text(item.storage_key): item for item in storage.capacities}
    attached_peaks = {_key_text(key): value for key, value in storage.peak_bytes}
    expected_keys = {_key_text(key) for key in spaces}
    if set(attached_capacities) != expected_keys:
        raise StorageError("STORAGE_CAPACITY_MISMATCH",
                           "the attached capacities do not cover the placed spaces")
    if set(attached_peaks) != expected_keys:
        raise StorageError("STORAGE_CAPACITY_MISMATCH",
                           "the attached peaks do not cover the placed spaces")
    for key, peak in spaces.items():
        capacity = _capacity_fact(target, key.space, gmem_exempt=False)
        if peak > capacity:
            raise StorageError("STORAGE_CAPACITY_EXCEEDED",
                               "the recomputed peak exceeds the target capacity",
                               "%s peak=%d capacity=%d" % (key.space, peak, capacity))
        stored = attached_capacities[_key_text(key)]
        if stored.usable_bytes != capacity:
            raise StorageError("STORAGE_CAPACITY_MISMATCH",
                               "the attached capacity is not the target fact",
                               _key_text(key))
        if stored.buffers != tuple(buffers_by_key.get(key, ())):
            raise StorageError("STORAGE_CAPACITY_MISMATCH",
                               "the attached per-space buffer list disagrees with the plan",
                               _key_text(key))
        if attached_peaks[_key_text(key)] != peak:
            raise StorageError("STORAGE_CAPACITY_MISMATCH",
                               "the attached peak disagrees with the recomputed peak",
                               _key_text(key))

    recomputed_payload = {
        "plan": plan_identity,
        "completion": storage.completion_identity,
        "effects": storage.effects_identity,
        "target": target_identity,
        "lowering": storage.lowering_identity,
        "requirements": [item.invalidation_key for item in storage.requirements],
        "entries": [_entry_identity(item) for item in storage.entries],
        "scratch": [_scratch_entry_identity(item) for item in storage.scratch_entries],
        "capacities": [(_key_text(item.storage_key), item.usable_bytes)
                       for item in storage.capacities],
        "peak": [(_key_text(key), value) for key, value in storage.peak_bytes],
    }
    if _digest(recomputed_payload) != storage.identity:
        raise StorageError(
            "STORAGE_PLACEMENT_MISMATCH",
            "the attached placement does not hash to its recorded identity "
            "(offset, bytes, version, capacity or scratch changed)",
            storage.identity,
        )

    if completion is None or lowering is None:
        raise StorageError("STORAGE_VERIFICATION_INPUTS_MISSING",
                           "completion and lowering are required to independently verify storage")
    expected = place_resolved_versions(plan, completion, target, lowering)
    for name in ("requirements", "entries", "scratch_entries", "capacities", "peak_bytes",
                 "completion_identity", "effects_identity", "lowering_identity", "plan_iteration"):
        if getattr(storage, name) != getattr(expected, name):
            raise StorageError("STORAGE_PLACEMENT_MISMATCH",
                               "current placement differs from recomputed " + name)
    for actual, fresh in zip(storage.requirements, expected.requirements, strict=True):
        if actual.root is not fresh.root:
            raise StorageError("STORAGE_REQUIREMENT_MISMATCH", "buffer root identity changed")
    for actual, fresh in zip(storage.scratch_entries, expected.scratch_entries, strict=True):
        if actual.allocation_ref is not fresh.allocation_ref:
            raise StorageError("STORAGE_SCRATCH_MISMATCH", "scratch allocation identity changed")
    from ..completion import verify_completion
    verify_completion(plan, completion, target)

    verification = StorageVerification(
        verified=True,
        plan_identity=plan_identity,
        storage_identity=storage.identity,
        requirements_identity=_digest([item.invalidation_key
                                       for item in storage.requirements]),
        target_identity=target_identity,
        detail="recomputed from plan+target; every attached offset/byte/version matches",
    )
    return replace(storage, verification=verification)


def require_storage_verified(storage, *, plan=None, completion=None, target=None, lowering=None):
    """Recheck the actual inputs and placement; a caller-provided status is not proof."""
    if not isinstance(storage, StoragePlan):
        raise StorageError("STORAGE_NOT_VERIFIED", "no StoragePlan was produced", type(storage).__name__)
    if plan is None or completion is None or target is None or lowering is None:
        raise StorageError("STORAGE_VERIFICATION_INPUTS_MISSING",
                           "plan, completion, target and lowering are required to reverify storage")
    return verify_storage_current(plan, storage, target, completion=completion, lowering=lowering)
