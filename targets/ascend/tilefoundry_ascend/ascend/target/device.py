"""Ascend 950PR device resource model."""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.types import DType
from tilefoundry.target.base import Device


@dataclass(frozen=True)
class Ascend950PR(Device):
    """One Ascend 950PR package: coupled blocks, scratchpads, HBM, and rates."""

    name: str
    sm_count: int
    aic_count: int
    aiv_count: int
    hbm_capacity_bytes: int
    hbm_owner: str
    hbm_bandwidth_bytes_per_second: int
    l1_bytes_per_aic: int
    l0a_bytes_per_aic: int
    l0b_bytes_per_aic: int
    l0c_bytes_per_aic: int
    ub_nominal_bytes_per_aiv: int
    ub_usable_bytes_per_aiv: int
    rmem_bytes_per_lane: int
    _unit_flops: tuple[tuple[str, tuple[tuple[DType, int], ...]], ...]
    aiv_lane_events_per_second: int

    @property
    def unit_flops_per_second(self) -> dict[str, dict[DType, int]]:
        """Return nominal compute throughput map per execution unit."""
        return {unit: dict(entries) for unit, entries in self._unit_flops}


__all__ = ["Ascend950PR"]
