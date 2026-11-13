"""Fact projections for Ascend targets."""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.analysis.facts import (
    ExplicitMemoryLevelFacts,
    MemoryHierarchyFacts,
    PerformanceServiceFacts,
    ThroughputFacts,
)
from tilefoundry.ir.types import DType
from tilefoundry.target.base import UnsupportedCapabilityError


@dataclass(frozen=True)
class AscendSpaceFacts:
    """Explicit Ascend on-chip memory capacities and alignment rules."""

    ub_nominal_bytes: int
    ub_usable_bytes: int
    l1_bytes: int
    l0a_bytes: int
    l0b_bytes: int
    l0c_bytes: int
    ub_align_bytes: int
    l1_align_bytes: int
    l0a_align_bytes: int
    l0b_align_bytes: int
    l0c_align_bytes: int

    def limit_bytes(self, space: str) -> int | None:
        """Return capacity limit in bytes for an Ascend scratchpad space."""
        return {
            "UB": self.ub_usable_bytes,
            "L1": self.l1_bytes,
            "L0A": self.l0a_bytes,
            "L0B": self.l0b_bytes,
            "L0C": self.l0c_bytes,
        }.get(space)


def ascend_space_facts(target) -> AscendSpaceFacts:
    """Project per-core on-chip space limits and alignment constraints."""
    arch = target.architecture
    dev = target.device
    return AscendSpaceFacts(
        ub_nominal_bytes=dev.ub_nominal_bytes_per_aiv,
        ub_usable_bytes=dev.ub_usable_bytes_per_aiv,
        l1_bytes=dev.l1_bytes_per_aic,
        l0a_bytes=dev.l0a_bytes_per_aic,
        l0b_bytes=dev.l0b_bytes_per_aic,
        l0c_bytes=dev.l0c_bytes_per_aic,
        ub_align_bytes=arch.ub_align_bytes,
        l1_align_bytes=arch.l1_align_bytes,
        l0a_align_bytes=arch.l0a_align_bytes,
        l0b_align_bytes=arch.l0b_align_bytes,
        l0c_align_bytes=arch.l0c_align_bytes,
    )


def memory_hierarchy(target, query: object = None) -> MemoryHierarchyFacts:
    """Return the explicit Ascend memory levels owned by target/aic/aiv/lane."""
    del query
    arch = target.architecture
    dev = target.device
    tmem_bytes = (
        dev.l1_bytes_per_aic
        + dev.l0a_bytes_per_aic
        + dev.l0b_bytes_per_aic
        + dev.l0c_bytes_per_aic
    )
    return MemoryHierarchyFacts(
        explicit_levels=(
            ExplicitMemoryLevelFacts(
                name="host",
                capacity_bytes=dev.hbm_capacity_bytes,
                scope="device",
                owner=dev.hbm_owner,
            ),
            ExplicitMemoryLevelFacts(
                name="gmem",
                capacity_bytes=dev.hbm_capacity_bytes,
                scope="device",
                owner=dev.hbm_owner,
            ),
            ExplicitMemoryLevelFacts(
                name="tmem",
                capacity_bytes=tmem_bytes,
                scope="aic",
                owner=arch.tmem_owner,
            ),
            ExplicitMemoryLevelFacts(
                name="smem",
                capacity_bytes=dev.ub_usable_bytes_per_aiv,
                scope="aiv",
                owner=arch.smem_owner,
            ),
            ExplicitMemoryLevelFacts(
                name="rmem",
                capacity_bytes=dev.rmem_bytes_per_lane,
                scope="lane",
                owner=arch.rmem_owner,
            ),
        ),
        implicit_levels=(),
        relations=(),
    )


def throughput(target, query: object = None) -> ThroughputFacts:
    """Return nominal compute and HBM bandwidth ceilings for Ascend roofline."""
    del query
    dev = target.device
    aic_rates = dev.unit_flops_per_second.get("aic", {})
    aiv_rates = dev.unit_flops_per_second.get("aiv", {})
    merged: dict[DType, int] = dict(aic_rates)
    for dtype, rate in aiv_rates.items():
        merged.setdefault(dtype, rate)
    return ThroughputFacts(
        peak_flops_per_second=tuple(
            sorted(merged.items(), key=lambda item: item[0].name)
        ),
        memory_bandwidth_bytes_per_second=dev.hbm_bandwidth_bytes_per_second,
        bandwidth_level="gmem",
    )


def performance_service(target, query: object = None) -> PerformanceServiceFacts:
    """Return per-unit compute and bandwidth rates for one Ascend topology level."""
    if query is None:
        unit = "block"
        active = target.device.sm_count
    elif isinstance(query, tuple):
        unit = str(query[0])
        active = query[1]
    else:
        unit = str(query)
        active = None
    divisors = {
        "block": target.device.sm_count,
        "aic": target.device.aic_count,
        "aiv": target.device.aiv_count,
        "lane": target.device.aiv_count * target.architecture.vector_lanes_f32,
    }
    if unit not in divisors:
        raise UnsupportedCapabilityError(
            f"ascend: no performance service for topology level {unit!r}"
        )
    share = divisors[unit]
    active_units = min(share, active) if active is not None and unit == "block" else 1
    dev = target.device
    aic_rates = dev.unit_flops_per_second.get("aic", {})
    aiv_rates = dev.unit_flops_per_second.get("aiv", {})
    merged: dict[DType, int] = dict(aic_rates)
    for dtype, rate in aiv_rates.items():
        merged.setdefault(dtype, rate)
    unit_flops = tuple(
        sorted(
            ((dtype, rate * active_units // share) for dtype, rate in merged.items()),
            key=lambda item: item[0].name,
        )
    )
    lane_rate = dev.aiv_lane_events_per_second * active_units // share
    unit_ops = tuple(
        (kind, lane_rate)
        for kind in ("integer", "predicate", "select", "local_move", "special")
    )
    return PerformanceServiceFacts(
        unit_flops=unit_flops,
        unit_ops=unit_ops,
        unit_bandwidth=(("gmem", dev.hbm_bandwidth_bytes_per_second * active_units // share),),
        unit=unit,
    )


__all__ = [
    "AscendSpaceFacts",
    "ascend_space_facts",
    "memory_hierarchy",
    "performance_service",
    "throughput",
]
