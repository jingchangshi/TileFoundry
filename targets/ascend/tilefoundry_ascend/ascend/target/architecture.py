"""Ascend DaVinci 950 architecture model."""

from __future__ import annotations

from dataclasses import dataclass

from tilefoundry.ir.types import DType
from tilefoundry.target.base import Architecture


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

    def supports_compute_dtype(self, dtype: DType) -> bool:
        """Return whether DaVinci 950 supports compute on ``dtype``."""
        return dtype in self.supported_compute_dtypes

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


__all__ = ["DaVinci950"]
