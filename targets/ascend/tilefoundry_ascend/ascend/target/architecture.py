"""Ascend DaVinci 950 architecture model."""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.types import DType
from tilefoundry.target.base import Architecture


def resolve_cube_macs_per_cycle(
    table: tuple[tuple[str, int], ...],
    dtype_name: str | DType,
    *,
    hf32: bool = False,
) -> int:
    """Resolve Cube MACs/cycle per AIC from *table* for *dtype_name*."""
    key = dtype_name.name if isinstance(dtype_name, DType) else str(dtype_name).strip().lower()
    if hf32 and key in ("f32", "hf32"):
        key = "hf32"
    mapping = dict(table)
    if key not in mapping:
        raise ValueError(f"unsupported Ascend 950 Cube dtype: {dtype_name!r}")
    return mapping[key]


def is_dma_path_legal(
    valid_dma_paths: tuple[str, ...],
    forbidden_dma_paths: tuple[str, ...],
    src: str,
    dst: str,
) -> bool:
    """Return whether direct DMA between ``src`` and ``dst`` exists on 3510."""
    path = f"{src.strip().upper()}->{dst.strip().upper()}"
    if path in forbidden_dma_paths:
        return False
    return path in valid_dma_paths


@dataclass(frozen=True)
class DaVinci950(Architecture):
    """DaVinci 950 architectural capabilities, geometry, and ownership."""

    name: str
    supported_compute_dtypes: tuple[DType, ...]
    capabilities: tuple[str, ...]
    aic_per_block: int
    aiv_per_block: int
    vector_lanes_f32: int
    max_threads_per_cta: int
    ub_align_bytes: int
    l1_align_bytes: int
    l0a_align_bytes: int
    l0b_align_bytes: int
    l0c_align_bytes: int
    tmem_owner: str
    smem_owner: str
    rmem_owner: str
    cube_mmad_atom: str
    ssbuf_align_bytes: int
    ssbuf_owner: str
    valid_dma_paths: tuple[str, ...]
    forbidden_dma_paths: tuple[str, ...]
    max_mte2_inflight_per_aiv: int
    fixpipe_bytes_per_cycle: int
    cube_macs_per_cycle_table: tuple[tuple[str, int], ...]
    l1_banks: int
    l1_bank_bytes: int
    ub_bank_groups: int
    ub_banks_per_group: int
    ub_bank_bytes: int
    ub_datablock_bytes: int
    bias_table_bytes: int
    fixbuf_bytes: int
    l2_bytes_per_cycle_per_aic: int
    l0a_layout: str
    l0b_layout: str
    l0c_layout: str
    max_crosscore_flag_depth: int
    hardware_mutex_count: int
    max_mmad_mnk: int
    forbidden_cube_dtypes: tuple[str, ...]
    series: str = "ascend_950"
    cce_arch: str = "dav-3510"
    supported_skus: tuple[str, ...] = ("ascend_950pr", "ascend_950dt")
    default_cann_series: str = "cann_9_3"
    supported_cann_series: tuple[str, ...] = ("cann_9_3",)

    @property
    def forbid_cube_int4(self) -> bool:
        """Return whether Cube MMAD forbids int4/uint4 input operands."""
        return "i4" in self.forbidden_cube_dtypes or "u4" in self.forbidden_cube_dtypes

    @property
    def cube_macs_per_cycle_map(self) -> dict[str, int]:
        """Return per-dtype Cube MACs/cycle per AIC as a dictionary."""
        return dict(self.cube_macs_per_cycle_table)

    def cube_macs_per_cycle(self, dtype_name: str, hf32: bool = False) -> int:
        """Return Cube MACs/cycle per AIC for *dtype_name* (with optional HF32 mode)."""
        return resolve_cube_macs_per_cycle(self.cube_macs_per_cycle_table, dtype_name, hf32=hf32)

    def supports_compute_dtype(self, dtype: DType) -> bool:
        """Return whether DaVinci 950 supports compute on ``dtype``."""
        return dtype in self.supported_compute_dtypes

    def is_dma_path_legal(self, src: str, dst: str) -> bool:
        """Return whether direct DMA between ``src`` and ``dst`` exists on 3510."""
        return is_dma_path_legal(self.valid_dma_paths, self.forbidden_dma_paths, src, dst)

    def topology_limit(self, name: str) -> int:
        """Return the structural limit for an Ascend topology level."""
        limits = {
            "aic": self.aic_per_block,
            "aiv": self.aiv_per_block,
            "lane": self.vector_lanes_f32,
        }
        try:
            return limits[name]
        except KeyError:
            raise ValueError(
                f"{self.name}: no architecture limit for topology level {name!r}"
            ) from None


__all__ = ["DaVinci950", "is_dma_path_legal", "resolve_cube_macs_per_cycle"]
