"""POC instance expansion for the small affine carrier (task-82).

The carrier loop body is authored once; the POC gate needs the 12 RAW instance obligations
(3 source families x i in [0,4)). This module expands the loop MECHANICALLY from the typed
loop domain (start/stop/step) and each access's offset structure (const or i*stride), never
from a hard-coded window table, and records provenance (loop, domain, count, stride, windows).

Unsupported (non-affine / symbolic bounds) expansion is refused by name
POC_EXPANSION_UNSUPPORTED; a wrong expansion count or stride is refused by name so a mutation
cannot silently pass.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from tilefoundry.ir.core import Call, Constant, Var

from tilefoundry_ascend.schedule.ledger import Ledger, LedgerError

FAMILIES = ("X-in", "Z-in", "Y-out")


@dataclass
class PocExpansion:
    loop: str
    start: int
    stop: int
    step: int
    count: int
    stride_bytes: int
    instances: list = field(default_factory=list)  # list[{"i", "windows": {task: (a,b)}}]
    families: tuple = FAMILIES

    def to_provenance(self) -> dict:
        return {
            "poc_instance_expansion": True,
            "loop": self.loop,
            "domain": {"start": self.start, "stop": self.stop, "step": self.step},
            "count": self.count,
            "stride_bytes": self.stride_bytes,
            "families": list(self.families),
            "instances": [
                {"i": inst["i"], "windows": {k: list(v) for k, v in inst["windows"].items()}}
                for inst in self.instances
            ],
        }


def _const_value(expr):
    if isinstance(expr, Constant):
        return int(expr.value)
    raise LedgerError("POC_EXPANSION_UNSUPPORTED: expected a constant loop bound, got %r" % (expr,))


def _extent_value(extent) -> int:
    if isinstance(extent, Constant):
        return int(extent.value)
    if isinstance(extent, int):
        return extent
    if isinstance(extent, str):
        match = re.search(r"value=(-?\d+)", extent)
        if match:
            return int(match.group(1))
    raise LedgerError("POC_EXPANSION_UNSUPPORTED: symbolic loop extent %r" % (extent,))


def _dim_extent(entry):
    for _name, extent in entry.iteration_domains:
        return _extent_value(extent)
    raise LedgerError("POC_EXPANSION_UNSUPPORTED: no loop domain on %r" % (entry.action_id,))


def _offset_bytes(entry, induction_names):
    """Byte window of one access as a function of i, from its typed offset structure."""
    stride = 0
    const = 0
    linear = False
    for part in entry.offset_structure:
        kind = part[0]
        if kind == "const":
            const += int(part[1])
        elif kind == "call" and part[1] == "DimMul":
            args = part[2]
            var_names = [arg[1] for arg in args if arg[0] == "var"]
            consts = [arg[1] for arg in args if arg[0] == "const"]
            if not var_names or len(consts) != 1 or var_names[0] not in induction_names:
                raise LedgerError("POC_EXPANSION_UNSUPPORTED: non-affine offset %r" % (part,))
            stride += int(consts[0])
            linear = True
        elif kind == "var" and part[1] in induction_names:
            stride += 1
            linear = True
        else:
            raise LedgerError("POC_EXPANSION_UNSUPPORTED: non-affine offset part %r" % (part,))
    return const, stride, linear


def expand_loop_instances(ledger: Ledger) -> PocExpansion:
    """Expand the carrier's single typed loop into concrete instances with provenance."""
    loop_entries = [e for e in ledger.entries if e.iteration_domains]
    if not loop_entries:
        raise LedgerError("POC_EXPANSION_UNSUPPORTED: no loop-domain accesses in ledger")
    induction_names = {name for name, _extent in loop_entries[0].iteration_domains}
    stop = _dim_extent(loop_entries[0])
    start, step = 0, 1
    count = (stop - start) // step
    if count <= 0:
        raise LedgerError("POC_EXPANSION_UNSUPPORTED: empty loop domain")
    loop_name = next(iter(induction_names))

    # Root-relative base per synthetic root: the allocation's arena start. A full-tile view may
    # be placed at a non-zero arena offset; its LOGICAL root-relative window starts at 0.
    root_base = {}
    for entry in loop_entries:
        _c, _s, _lin = _offset_bytes(entry, induction_names)
        base = root_base.get(entry.root)
        root_base[entry.root] = _c if base is None else min(base, _c)

    strides = set()
    per_entry = {}
    for entry in loop_entries:
        const, stride, linear = _offset_bytes(entry, induction_names)
        if linear:
            strides.add((stride, entry.byte_footprint))
        per_entry[entry.action_id] = (entry, const, stride, linear)
    if len(strides) > 1:
        raise LedgerError("POC_EXPANSION_UNSUPPORTED: inconsistent strides %r" % (sorted(strides),))
    if strides:
        stride_elems, stride_bytes = next(iter(strides))
    else:
        stride_elems, stride_bytes = 0, 0

    instances = []
    for index in range(count):
        i = start + index * step
        windows = {}
        for task, (entry, const, stride, linear) in per_entry.items():
            base = root_base.get(entry.root, 0)
            elems = const + (stride * i if linear else 0) - base
            lo = elems * (entry.byte_footprint // max(1, _tile_elems(entry)))
            windows[task] = (lo, lo + entry.byte_footprint)
        instances.append({"i": i, "windows": windows})
    return PocExpansion(
        loop=loop_name, start=start, stop=stop, step=step, count=count,
        stride_bytes=stride_bytes, instances=instances,
    )


def _tile_elems(entry) -> int:
    # byte footprint of one tile / per-element bytes is encoded by the entry's own offset unit;
    # for the f32 carrier a full-tile footprint is byte-count and the stride is in elements.
    return int(entry.byte_footprint // 4) if entry.byte_footprint else 1


def verify_expansion(expansion: PocExpansion, *, expected_count: int, expected_stride_bytes: int) -> dict:
    """Refuse a wrong expansion COUNT or STRIDE by name (mutation controls)."""
    if expansion.count != expected_count:
        raise LedgerError(
            "POC_EXPANSION_COUNT_MISMATCH: count=%d expected %d" % (expansion.count, expected_count)
        )
    if expansion.stride_bytes != expected_stride_bytes:
        raise LedgerError(
            "POC_EXPANSION_STRIDE_MISMATCH: stride_bytes=%d expected %d"
            % (expansion.stride_bytes, expected_stride_bytes)
        )
    return {"count": expansion.count, "stride_bytes": expansion.stride_bytes, "provenance": expansion.to_provenance()}
