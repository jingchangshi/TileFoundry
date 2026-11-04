"""P1 report: memory-space semantics, capacity model, classification and refusals.

Emits the contract as JSON: the five semantic spaces and their capacity model, a set of
classified accesses, and the refusals. JSON only, so the artifact is reproducible.

    PYTHONPATH=<repo>/src:<repo>/targets/ascend python -m tilefoundry_ascend.placement_report
"""
from __future__ import annotations

import json

from .placement import (
    AccessSpec,
    PlacementRefusal,
    ROLE_AIC_OPERAND,
    ROLE_PARAM,
    ROLE_PRIVATE_TEMP,
    ROLE_SCALAR,
    ROLE_STAGED_TILE,
    ROLE_VERSIONED_SLOT,
    SLAB,
    classify,
    space_table,
)


def _classified(accesses) -> list:
    rows = []
    for access in accesses:
        try:
            rows.append({
                "name": access.name, "role": access.role, "bytes": access.bytes,
                "versions": access.versions, "domain": access.domain,
                "inside_vf": access.inside_vf, "wants": access.wants,
                "space": classify(access),
            })
        except PlacementRefusal as error:
            rows.append({
                "name": access.name, "role": access.role, "bytes": access.bytes,
                "refused": str(error),
            })
    return rows


def main() -> int:
    accesses = [
        AccessSpec("x", ROLE_PARAM, 64 * 64 * 2),
        AccessSpec("out", ROLE_PARAM, 64 * 64 * 2),
        AccessSpec("x_tile", ROLE_STAGED_TILE, 64 * 64 * 2),
        AccessSpec("acc", ROLE_PRIVATE_TEMP, 256, inside_vf=True),
        AccessSpec("eps", ROLE_SCALAR, 4, inside_vf=True),
        AccessSpec("ring_y", ROLE_VERSIONED_SLOT, 64 * 64 * 2, versions=2),
        AccessSpec("w_l1", ROLE_AIC_OPERAND, 64 * 64 * 2, domain="aic"),
        AccessSpec("bad_l2", ROLE_PARAM, 8, wants="ascend.l2"),
        AccessSpec("bad_temp_outside", ROLE_PRIVATE_TEMP, 8, inside_vf=False),
        AccessSpec("bad_role", "no-such-role", 8),
    ]
    report = {
        "stage": "P1",
        "spaces": space_table(),
        "classified": _classified(accesses),
        "slab_note": "slab is a versioned staging ring inside UB; capacity is the parent bound",
        "slab_example": {"space": SLAB, "slot_bytes": 64 * 64 * 2, "versions": 2},
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
