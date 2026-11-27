"""Fact projections for Ascend targets."""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.analysis.facts import (
    ExplicitMemoryLevelFacts,
    ImplicitMemoryLevelFacts,
    MemoryHierarchyFacts,
    MemoryLevelRelation,
    MemoryRelationKind,
    PerformanceServiceFacts,
    ThroughputFacts,
)
from tilefoundry.ir.types import DType
from tilefoundry.target.base import UnsupportedCapabilityError
from tilefoundry_ascend.ascend.target.architecture import (
    is_dma_path_legal,
    resolve_cube_macs_per_cycle,
)
from tilefoundry_ascend.ascend.target.device import detect_runtime_core_counts

DEFAULT_CUBE_MACS_PER_CYCLE: tuple[tuple[str, int], ...] = (
    ("f4e2m1", 16384),
    ("f4e1m2", 16384),
    ("fp8e4m3", 8192),
    ("fp8e5m2", 8192),
    ("hif8", 8192),
    ("i8", 8192),
    ("u8", 8192),
    ("f16", 4096),
    ("bf16", 4096),
    ("hf32", 2048),
    ("f32", 256),
)


def cube_macs_per_cycle(dtype_name: str, hf32: bool = False) -> int:
    """Return dav-3510 Cube MACs/cycle per AIC for *dtype_name*."""
    return resolve_cube_macs_per_cycle(DEFAULT_CUBE_MACS_PER_CYCLE, dtype_name, hf32=hf32)


_CUBE_DTYPE_ALIASES: dict[str, str] = {
    "float16": "f16",
    "fp16": "f16",
    "bfloat16": "bf16",
    "float32": "f32",
    "fp32": "f32",
    "int8": "i8",
    "uint8": "u8",
    "e4m3": "fp8e4m3",
    "fp8_e4m3": "fp8e4m3",
    "float8_e4m3": "fp8e4m3",
    "float8_e4m3fn": "fp8e4m3",
    "e5m2": "fp8e5m2",
    "fp8_e5m2": "fp8e5m2",
    "float8_e5m2": "fp8e5m2",
    "float4_e2m1": "f4e2m1",
    "fp4_e2m1": "f4e2m1",
    "float4_e1m2": "f4e1m2",
    "fp4_e1m2": "f4e1m2",
}


def resolve_cube_macs_safe(
    facts: AscendSpaceFacts,
    dtype_name: object,
    *,
    hf32: bool = False,
) -> int:
    """Resolve Cube MACs/cycle safely from canonical or full TIR dtype names."""
    raw = str(getattr(dtype_name, "name", dtype_name) or "").strip().lower()
    if not raw:
        raise ValueError(f"Unsupported empty Cube operand dtype {dtype_name!r}")
    canon = _CUBE_DTYPE_ALIASES.get(raw, raw)
    if hf32 and canon in ("f32", "hf32"):
        canon = "hf32"
    try:
        return int(facts.cube_macs_per_cycle(canon, hf32=hf32))
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(
            f"Unsupported Cube operand dtype {dtype_name!r} (canonical={canon!r})"
        ) from exc

PURE_SIMDVF_UBUF_FLAGS: tuple[str, str] = (
    "--cce-disable-vf-stack-reserved-ubuf",
    "--cce-disable-asc-reserved-ubuf",
)


@dataclass(frozen=True)
class AscendSpaceFacts:
    """Explicit Ascend on-chip memory capacities and alignment rules."""

    ub_nominal_bytes: int
    ub_usable_bytes: int
    ub_simdvf_usable_bytes: int
    ub_simtvf_usable_bytes: int
    l1_bytes: int
    l0a_bytes: int
    l0b_bytes: int
    l0c_bytes: int
    ub_align_bytes: int
    l1_align_bytes: int
    l0a_align_bytes: int
    l0b_align_bytes: int
    l0c_align_bytes: int
    ssbuf_bytes: int
    ssbuf_align_bytes: int
    ssbuf_latency_cycles: int
    ssbuf_owner: str
    max_mte2_inflight_per_aiv: int
    fixpipe_bytes_per_cycle: int
    valid_dma_paths: tuple[str, ...]
    forbidden_dma_paths: tuple[str, ...]
    num_aic_cores: int = 28
    num_aiv_cores: int = 56
    hbm_peak_bandwidth_bps: int = 0
    hbm_burst_bandwidth_table: tuple[tuple[int, int], ...] = ()
    clock_ghz: float = 1.65
    cube_macs_per_cycle_table: tuple[tuple[str, int], ...] = DEFAULT_CUBE_MACS_PER_CYCLE
    l1_banks: int = 2
    l1_bank_bytes: int = 262144
    ub_bank_groups: int = 8
    ub_banks_per_group: int = 2
    ub_bank_bytes: int = 16384
    ub_datablock_bytes: int = 32
    bias_table_bytes: int = 4096
    fixbuf_bytes: int = 16384
    l2_capacity_bytes: int = 134217728
    l2_bytes_per_cycle_per_aic: int = 101
    l0a_layout: str = "NZ"
    l0b_layout: str = "ZN"
    l0c_layout: str = "NZ"
    max_crosscore_flag_depth: int = 15
    hardware_mutex_count: int = 28
    max_mmad_mnk: int = 4095
    forbid_cube_int4: bool = True

    def cube_macs_per_cycle(self, dtype_name: str, hf32: bool = False) -> int:
        """Return Cube MACs/cycle per AIC for *dtype_name* (with optional HF32 mode)."""
        return resolve_cube_macs_per_cycle(self.cube_macs_per_cycle_table, dtype_name, hf32=hf32)

    def effective_hbm_bandwidth_bps(self, payload_bytes: int) -> float:
        """Return piecewise-linear HBM bandwidth estimate (byte/s) for *payload_bytes*."""
        if not self.hbm_burst_bandwidth_table:
            return float(self.hbm_peak_bandwidth_bps)
        table = sorted(self.hbm_burst_bandwidth_table, key=lambda item: item[0])
        if payload_bytes <= table[0][0]:
            return float(table[0][1])
        for idx in range(1, len(table)):
            lo_b, lo_r = table[idx - 1]
            hi_b, hi_r = table[idx]
            if payload_bytes <= hi_b:
                span = max(1, hi_b - lo_b)
                frac = float(payload_bytes - lo_b) / float(span)
                return float(lo_r) + (float(hi_r) - float(lo_r)) * frac
        return float(table[-1][1])

    def dma_burst_efficiency(self, payload_bytes: int) -> float:
        """Return normalized [0, 1] burst efficiency against peak HBM burst rate."""
        if payload_bytes <= 0 or not self.hbm_burst_bandwidth_table:
            return 1.0
        ref_peak = float(max(rate for _sz, rate in self.hbm_burst_bandwidth_table))
        if ref_peak <= 0.0:
            return 1.0
        eff = self.effective_hbm_bandwidth_bps(payload_bytes)
        return round(min(1.0, max(0.0, eff / ref_peak)), 6)

    def ub_limit_bytes(self, mode: str = "simdvf", *, unreserved_ubuf: bool = False) -> int:
        """Return mode-specific UB budget in bytes for SimdVF vs SimtVF execution."""
        if mode in ("simtvf", "ascend.simtvf", "mixed"):
            return self.ub_simtvf_usable_bytes
        if mode == "simdvf_unreserved":
            return self.ub_nominal_bytes
        if mode in ("simdvf", "ascend.simdvf"):
            return self.ub_nominal_bytes if unreserved_ubuf else self.ub_simdvf_usable_bytes
        raise ValueError(f"unsupported Ascend UB mode: {mode!r}")

    def limit_bytes(self, space: str) -> int | None:
        """Return capacity limit in bytes for an Ascend scratchpad space."""
        return {
            "UB": self.ub_usable_bytes,
            "L1": self.l1_bytes,
            "L0A": self.l0a_bytes,
            "L0B": self.l0b_bytes,
            "L0C": self.l0c_bytes,
            "SSBUF": self.ssbuf_bytes,
        }.get(space)

    def is_dma_path_legal(self, src: str, dst: str) -> bool:
        """Return whether direct DMA between ``src`` and ``dst`` exists on 3510."""
        return is_dma_path_legal(self.valid_dma_paths, self.forbidden_dma_paths, src, dst)


def _select_ub_bytes(target, query: object = None, *, default_simdvf: bool = False) -> int:
    dev = target.device
    if query is None:
        return dev.ub_simdvf_usable_bytes_per_aiv if default_simdvf else dev.ub_usable_bytes_per_aiv
    mode = str(query)
    if mode in ("simtvf", "ascend.simtvf", "mixed"):
        return dev.ub_usable_bytes_per_aiv
    if mode in ("simdvf", "ascend.simdvf"):
        return dev.ub_simdvf_usable_bytes_per_aiv
    if mode == "simdvf_unreserved":
        return dev.ub_nominal_bytes_per_aiv
    raise UnsupportedCapabilityError(f"ascend: unsupported UB query {query!r}")


def ascend_space_facts(target, query: object = None) -> AscendSpaceFacts:
    """Project per-core on-chip space limits and alignment constraints."""
    arch = target.architecture
    dev = target.device
    return AscendSpaceFacts(
        ub_nominal_bytes=dev.ub_nominal_bytes_per_aiv,
        ub_usable_bytes=_select_ub_bytes(target, query, default_simdvf=False),
        ub_simdvf_usable_bytes=dev.ub_simdvf_usable_bytes_per_aiv,
        ub_simtvf_usable_bytes=dev.ub_usable_bytes_per_aiv,
        l1_bytes=dev.l1_bytes_per_aic,
        l0a_bytes=dev.l0a_bytes_per_aic,
        l0b_bytes=dev.l0b_bytes_per_aic,
        l0c_bytes=dev.l0c_bytes_per_aic,
        ub_align_bytes=arch.ub_align_bytes,
        l1_align_bytes=arch.l1_align_bytes,
        l0a_align_bytes=arch.l0a_align_bytes,
        l0b_align_bytes=arch.l0b_align_bytes,
        l0c_align_bytes=arch.l0c_align_bytes,
        ssbuf_bytes=dev.ssbuf_bytes_per_block,
        ssbuf_align_bytes=arch.ssbuf_align_bytes,
        ssbuf_latency_cycles=dev.ssbuf_latency_cycles,
        ssbuf_owner=arch.ssbuf_owner,
        max_mte2_inflight_per_aiv=arch.max_mte2_inflight_per_aiv,
        fixpipe_bytes_per_cycle=arch.fixpipe_bytes_per_cycle,
        valid_dma_paths=arch.valid_dma_paths,
        forbidden_dma_paths=arch.forbidden_dma_paths,
        num_aic_cores=dev.aic_count,
        num_aiv_cores=dev.aiv_count,
        hbm_peak_bandwidth_bps=dev.hbm_bandwidth_bytes_per_second,
        hbm_burst_bandwidth_table=dev.hbm_burst_bandwidth_table,
        clock_ghz=dev.clock_ghz,
        cube_macs_per_cycle_table=arch.cube_macs_per_cycle_table,
        l1_banks=arch.l1_banks,
        l1_bank_bytes=arch.l1_bank_bytes,
        ub_bank_groups=arch.ub_bank_groups,
        ub_banks_per_group=arch.ub_banks_per_group,
        ub_bank_bytes=arch.ub_bank_bytes,
        ub_datablock_bytes=arch.ub_datablock_bytes,
        bias_table_bytes=arch.bias_table_bytes,
        fixbuf_bytes=arch.fixbuf_bytes,
        l2_capacity_bytes=dev.l2_capacity_bytes,
        l2_bytes_per_cycle_per_aic=arch.l2_bytes_per_cycle_per_aic,
        l0a_layout=arch.l0a_layout,
        l0b_layout=arch.l0b_layout,
        l0c_layout=arch.l0c_layout,
        max_crosscore_flag_depth=arch.max_crosscore_flag_depth,
        hardware_mutex_count=arch.hardware_mutex_count,
        max_mmad_mnk=arch.max_mmad_mnk,
        forbid_cube_int4=arch.forbid_cube_int4,
    )


def memory_hierarchy(target, query: object = None) -> MemoryHierarchyFacts:
    """Return explicit/implicit Ascend memory levels and UB/SIMT-DCache sharing edge."""
    arch = target.architecture
    dev = target.device
    smem_capacity = _select_ub_bytes(target, query, default_simdvf=True)
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
                capacity_bytes=smem_capacity,
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
        implicit_levels=(
            ImplicitMemoryLevelFacts(
                name="simt_dcache",
                capacity_bytes=None,
                scope="aiv",
            ),
        ),
        relations=(
            MemoryLevelRelation(
                kind=MemoryRelationKind.CACHES,
                near="simt_dcache",
                far="gmem",
            ),
            MemoryLevelRelation(
                kind=MemoryRelationKind.SHARES_CAPACITY_WITH,
                near="simt_dcache",
                far="smem",
                shared_capacity_bytes=dev.ub_nominal_bytes_per_aiv,
            ),
        ),
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
    "DEFAULT_CUBE_MACS_PER_CYCLE",
    "PURE_SIMDVF_UBUF_FLAGS",
    "AscendSpaceFacts",
    "ascend_space_facts",
    "cube_macs_per_cycle",
    "detect_runtime_core_counts",
    "is_dma_path_legal",
    "memory_hierarchy",
    "performance_service",
    "resolve_cube_macs_safe",
    "throughput",
]
