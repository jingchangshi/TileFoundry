#!/usr/bin/env python3
"""P4 PLAN-LEVEL alias checker (WBS 04-02). SPEC-side tool; not an IR/lowering change.

THE ONLY OVERLAP IMPLEMENTATION IS view_spec.alias_conflicts. This module adds:
  * construction of views/accesses FROM REAL Access objects (tilefoundry_ascend
    .schedule.problem.Access: task, root, space, mode, byte_footprint, logical_iteration);
  * PLAN-layer rules that belong above the target-side validator:
      PLAN_STALE_VIEW       view.generation != allocation generation
      PLAN_EXTENT_OVERFLOW  element count above the representational cap
      ACCESS_MODE_MISSING   an access has no explicit read/write mode (NEVER default read)
      ALIAS_OVERLAP         returned BY alias_conflicts, never recomputed here;
  * LIFETIME PARTITIONING (generation is spec/plan, not target): views are grouped by
    (root, generation) and alias_conflicts is called ONCE PER GROUP, so two views that
    live in different generations are sequenced reuse (governed by sync), not an alias
    conflict.

No byte-range intersection predicate is written in this file. Deleting the
alias_conflicts call must therefore make an illegal overlapping plan ACCEPTED; that
mutation is pinned by tools/p4/test_alias_plan.py.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "p3"))

from view_spec import DTYPE_WIDTH, MAX_ELEMENTS, GmemRoot, OffsetView, alias_conflicts  # noqa: E402

RULES = ("ALIAS_OVERLAP", "PLAN_STALE_VIEW", "PLAN_EXTENT_OVERFLOW", "ACCESS_MODE_MISSING")


class AliasPlanError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__("%s: %s" % (code, message))
        self.code = code


@dataclass(frozen=True)
class PlanRootSpec:
    """Plan-layer identity of one root: dtype widths, byte size, allocation generation."""
    dtype: str
    root_bytes: int
    base_byte: int = 0
    generation: int = 0
    element_cap: int = MAX_ELEMENTS

    @property
    def width(self) -> int:
        return DTYPE_WIDTH[self.dtype]


class AliasPlanChecker:
    """Checks one normalized plan. trace[] records every alias_conflicts call."""

    def __init__(self) -> None:
        self.trace: list[dict] = []

    def views_from_accesses(self, accesses, root_specs: dict):
        """Returns (views, access_modes). The MODE map is built here so omission cannot
        become an implicit read later."""
        """Build views from REAL Access objects.

        Distinct (root, dtype, byte range) groups become one view; the mode is OR
        (write if any access writes). A missing/unknown mode is a named refusal: a
        caller must never be able to make a write look like a read by omission.
        """
        groups: dict[tuple, dict] = {}
        for a in accesses:
            if isinstance(a, str):
                raise AliasPlanError("ACCESS_MODE_MISSING", "access %r is a bare string" % (a,))
            mode = getattr(a, "mode", None)
            if mode not in ("read", "write"):
                raise AliasPlanError("ACCESS_MODE_MISSING",
                                     "access %r on root %r has mode %r (no default read)"
                                     % (getattr(a, "task", "?"), getattr(a, "root", "?"), mode))
            root = getattr(a, "root", None)
            spec = root_specs.get(root)
            if spec is None:
                raise AliasPlanError("UNKNOWN_ROOT", "access root %r has no PlanRootSpec" % (root,))
            width = spec.width
            footprint = int(getattr(a, "byte_footprint", 0))
            if footprint < 0 or footprint % width:
                raise AliasPlanError("ACCESS_FOOTPRINT_INVALID",
                                     "root %r footprint %d is not a non-negative multiple of %d"
                                     % (root, footprint, width))
            offset = int(getattr(a, "byte_offset", 0))
            key = (root, str(spec.dtype), offset, footprint)
            g = groups.setdefault(key, {"mode": mode, "task": getattr(a, "task", "?"),
                                        "iter": getattr(a, "logical_iteration", "t")})
            if mode == "write":
                g["mode"] = "write"
        views, modes = [], {}
        for (root, dtype, offset, footprint), g in sorted(groups.items()):
            name = "%s:%s@%d:%s" % (root, dtype, offset, g["task"])
            views.append(OffsetView(name=name, root=root,
                                    generation=root_specs[root].generation,
                                    byte_offset=offset, dtype=dtype,
                                    count=footprint // DTYPE_WIDTH[dtype]))
            modes[name] = g["mode"]
        return views, modes

    def check(self, views: list[OffsetView], root_specs: dict,
              accesses: dict[str, str]) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {r: [] for r in RULES}
        known: list[OffsetView] = []
        for v in views:
            spec = root_specs.get(v.root)
            if spec is None:
                out["PLAN_STALE_VIEW"].append("%s: unknown root %r" % (v.name, v.root))
                continue
            if v.generation != spec.generation:
                out["PLAN_STALE_VIEW"].append(
                    "%s: view generation %d != allocation generation %d"
                    % (v.name, v.generation, spec.generation))
                continue
            if v.count > spec.element_cap:
                out["PLAN_EXTENT_OVERFLOW"].append(
                    "%s: %d elements > cap %d" % (v.name, v.count, spec.element_cap))
            if accesses.get(v.name) not in ("read", "write"):
                raise AliasPlanError("ACCESS_MODE_MISSING",
                                     "%s: access map has %r (no default read)"
                                     % (v.name, accesses.get(v.name)))
            known.append(v)
        # LIFETIME PARTITION: alias is checked inside one (root, generation) group only.
        groups: dict[tuple, list[OffsetView]] = {}
        for v in known:
            groups.setdefault((v.root, v.generation), []).append(v)
        for key in sorted(groups):
            group = groups[key]
            pairs = alias_conflicts(group, accesses)   # <<< THE ONLY OVERLAP IMPLEMENTATION
            self.trace.append({"call": "view_spec.alias_conflicts",
                               "group": "%s@%d" % key,
                               "views": [v.name for v in group],
                               "pairs": [list(p) for p in pairs]})
            for a, b in pairs:
                out["ALIAS_OVERLAP"].append("%s/%s: intersecting same-root views with a write"
                                            % (a, b))
        return out

    def check_plan(self, *, accesses=None, views=None, root_specs, access_modes=None) -> dict:
        if views is None:
            if accesses is None:
                raise AliasPlanError("NO_INPUTS", "either accesses or views is required")
            views, derived_modes = self.views_from_accesses(accesses, root_specs)
            if access_modes is None:
                access_modes = derived_modes
        if access_modes is None:
            # Views supplied by hand still need explicit modes; an omitted access must
            # NOT default to read (that is exactly the hole this rule closes).
            raise AliasPlanError("ACCESS_MODE_MISSING",
                                 "access_modes map is required for hand-supplied views")
        return self.check(views, root_specs, access_modes)


__all__ = ["AliasPlanChecker", "AliasPlanError", "PlanRootSpec", "RULES", "alias_conflicts"]
