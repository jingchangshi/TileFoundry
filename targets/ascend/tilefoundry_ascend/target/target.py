"""Ascend950PR target registration and device facts.

The 950PR facts below are device records for the local Ascend950PR_9579 SKU,
cross-checked with npu-smi on this host (2 physical devices, 28 AIC / 56 AIV by
the 3510 architecture description). They are *records*, not universal constants:
the AIV grid size of a program is a program parameter; the number of physical
AIVs comes from this profile.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ..memory import MemorySpaceFacts, space_facts_table


@dataclass(frozen=True)
class DeviceProfile:
    profile_id: str
    device_name: str
    sku: str
    arch: str
    aic_count: int
    aiv_count: int
    spaces: dict
    source: str
    notes: str = ""

    def aiv_grid_limit(self) -> int:
        return self.aiv_count

    def space(self, name: str) -> Optional[MemorySpaceFacts]:
        return self.spaces.get(name)

    def as_dict(self) -> dict:
        return {
            "profile_id": self.profile_id,
            "device_name": self.device_name,
            "sku": self.sku,
            "arch": self.arch,
            "aic_count": self.aic_count,
            "aiv_count": self.aiv_count,
            "source": self.source,
            "notes": self.notes,
            "spaces": {k: v.as_dict() for k, v in self.spaces.items()},
        }


def ascend950pr_profile() -> DeviceProfile:
    return DeviceProfile(
        profile_id="ascend950pr_9579",
        device_name="Ascend950PR_9579",
        sku="Ascend950PR",
        arch="npu_arch_3510",
        aic_count=28,
        aiv_count=56,
        spaces=space_facts_table(),
        source="npu-smi 25.7.rc1 on local host + 3510 architecture spec (S15-S19); capacities nominal",
        notes=(
            "UB/L1/L0 capacities are nominal per 3510 docs. Usable values depend on "
            "reservation, compile options and execution mode; not measured here, so "
            "usable_bytes is None where unmeasured. AIC spaces are contract-only in this round."
        ),
    )


__all__ = ["DeviceProfile", "ascend950pr_profile"]