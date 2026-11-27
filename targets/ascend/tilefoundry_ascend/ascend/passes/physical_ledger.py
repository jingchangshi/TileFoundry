"""Objective physical ledger exports for Ascend schedule and envelope inspection."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import ClassVar, Literal

from tilefoundry_ascend.ascend.json_record import JsonRecordMixin
from tilefoundry_ascend.ascend.passes.observation_space import (
    CompilerSummaryLike,
    ObservationVector5D,
    RooflineSummaryLike,
)


@dataclass(frozen=True)
class ScopeWaterline(JsonRecordMixin):
    """Memory capacity and compiler-bloat accounting for one on-chip scope."""

    scope: str
    capacity_bytes: int
    explicit_bytes: int
    implicit_bloat_bytes: int
    peak_allocated_bytes: int
    overflow_bytes: int
    explicit_buffers: dict[str, int]
    implicit_buffers: dict[str, int]
    degraded_single_buffers: tuple[str, ...] = ()


@dataclass(frozen=True)
class ObjectivePhysicalLedger(JsonRecordMixin):
    """Rule-blind hardware utilization, waterline, and legality telemetry."""

    _json_sort_keys: ClassVar[bool] = True

    analysis_mode: str
    is_legal: bool
    steady_state_ii: int | None
    pipe_cycles: dict[str, int]
    dominant_pipe: str
    waterlines: dict[str, ScopeWaterline]
    mte2_inflight_per_aiv: int
    unaligned_dma_padding_bytes: int
    opaque_dynamic_regions: tuple[str, ...] = ()
    grid_blocks: int = 1
    wave_occupancy: float = 0.0
    gm_roundtrip_bytes: int = 0
    dominant_contig_bytes: int = 0
    effective_cube_m: int = 0
    has_indirect_gather: bool = False
    has_lowbit_quant: bool = False
    has_online_rescale_loop: bool = False
    in_loop_ub_reload_stores: int = 0
    uses_unreserved_ubuf: bool = False
    grid_is_symbolic: bool = False
    min_gm_dma_bytes: int = 0
    dma_burst_efficiency_ratio: float = 1.0
    uses_dynamic_offset_pipelining: bool = False
    has_unlimit_memory_attr: bool = False
    has_coupled_reduce_sfu_accum: bool = False
    active_grid_blocks: int = 0
    in_loop_pipe_all_barriers: int = 0
    ub_bank_conflict_risk: float = 0.0
    l1_bank_conflict_risk: float = 0.0
    ub_scalar_compute_loops: int = 0
    microarch_profile: dict[str, object] = field(default_factory=dict)
    gm_traffic_scope: Literal["per_tile", "full_grid"] = "per_tile"

    def __post_init__(self) -> None:
        if self.analysis_mode not in ("exact_modulo_z3", "readonly_envelope"):
            raise ValueError(
                f"analysis_mode must be 'exact_modulo_z3' or 'readonly_envelope', "
                f"got {self.analysis_mode!r}"
            )
        if self.gm_traffic_scope not in ("per_tile", "full_grid"):
            raise ValueError(
                f"gm_traffic_scope must be 'per_tile' or 'full_grid', "
                f"got {self.gm_traffic_scope!r}"
            )

    @property
    def max_overflow_bytes(self) -> int:
        """Return the maximum byte overflow across all tracked memory scopes."""
        return max((wl.overflow_bytes for wl in self.waterlines.values()), default=0)

    def total_gm_bytes(self, grid_blocks: int | None = None) -> int:
        """Return full-grid GM traffic bytes using explicit per-tile or full-grid scope."""
        base_bytes = max(0, int(self.gm_roundtrip_bytes))
        if self.gm_traffic_scope == "full_grid":
            return base_bytes
        blocks = max(
            1,
            int(
                grid_blocks
                if grid_blocks is not None
                else (self.active_grid_blocks or self.grid_blocks or 1)
            ),
        )
        return base_bytes * blocks

    def to_observation_5d(
        self,
        *,
        compiler_summary: CompilerSummaryLike | None = None,
        telemetry: Mapping[str, object] | None = None,
        roofline: RooflineSummaryLike | Mapping[str, object] | None = None,
    ) -> ObservationVector5D:
        """Project this L0 ledger plus L1/L2 telemetry and Roofline onto the 5-axis space."""
        return ObservationVector5D.from_tiers(
            self,
            compiler_summary=compiler_summary,
            telemetry=telemetry,
            roofline=roofline,
        )

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ObjectivePhysicalLedger:
        """Reconstruct an ObjectivePhysicalLedger from a serialized dictionary."""
        norm = dict(payload)
        if "active_grid_blocks" not in norm:
            norm["active_grid_blocks"] = norm.get("grid_blocks", 1)
        return super().from_dict(norm)


def compute_dominant_pipe(pipe_cycles: dict[str, int]) -> str:
    """Return the pipe key with the highest cycle count, or 'NONE' if empty."""
    positive = {k: v for k, v in pipe_cycles.items() if v > 0}
    if not positive:
        return "NONE"
    return max(positive.items(), key=lambda item: (item[1], item[0]))[0]


def build_scope_waterline(
    *,
    scope: str,
    capacity_bytes: int,
    explicit_buffers: dict[str, int],
    implicit_buffers: dict[str, int],
    peak_allocated_bytes: int | None = None,
    degraded_single_buffers: tuple[str, ...] = (),
) -> ScopeWaterline:
    """Build a ScopeWaterline from explicit, implicit, and peak allocation maps."""
    explicit_total = sum(explicit_buffers.values())
    implicit_total = sum(implicit_buffers.values())
    peak = (
        peak_allocated_bytes
        if peak_allocated_bytes is not None
        else explicit_total + implicit_total
    )
    overflow = max(0, peak - capacity_bytes) if capacity_bytes > 0 else 0
    return ScopeWaterline(
        scope=scope,
        capacity_bytes=capacity_bytes,
        explicit_bytes=explicit_total,
        implicit_bloat_bytes=implicit_total,
        peak_allocated_bytes=peak,
        overflow_bytes=overflow,
        explicit_buffers=dict(explicit_buffers),
        implicit_buffers=dict(implicit_buffers),
        degraded_single_buffers=tuple(degraded_single_buffers),
    )


def build_objective_physical_ledger(
    *,
    analysis_mode: str,
    is_legal: bool,
    steady_state_ii: int | None,
    pipe_cycles: dict[str, int],
    waterlines: dict[str, ScopeWaterline],
    mte2_inflight_per_aiv: int = 0,
    unaligned_dma_padding_bytes: int = 0,
    opaque_dynamic_regions: tuple[str, ...] = (),
    grid_blocks: int = 1,
    wave_occupancy: float = 0.0,
    gm_roundtrip_bytes: int = 0,
    dominant_contig_bytes: int = 0,
    effective_cube_m: int = 0,
    has_indirect_gather: bool = False,
    has_lowbit_quant: bool = False,
    has_online_rescale_loop: bool = False,
    in_loop_ub_reload_stores: int = 0,
    uses_unreserved_ubuf: bool = False,
    grid_is_symbolic: bool = False,
    min_gm_dma_bytes: int = 0,
    dma_burst_efficiency_ratio: float = 1.0,
    uses_dynamic_offset_pipelining: bool = False,
    has_unlimit_memory_attr: bool = False,
    has_coupled_reduce_sfu_accum: bool = False,
    active_grid_blocks: int | None = None,
    in_loop_pipe_all_barriers: int = 0,
    ub_bank_conflict_risk: float = 0.0,
    l1_bank_conflict_risk: float = 0.0,
    ub_scalar_compute_loops: int = 0,
    microarch_profile: Mapping[str, object] | None = None,
    gm_traffic_scope: Literal["per_tile", "full_grid"] = "per_tile",
) -> ObjectivePhysicalLedger:
    """Assemble an ObjectivePhysicalLedger and derive the dominant pipe."""
    cleaned_pipes = dict(sorted(pipe_cycles.items()))
    overflow_free = all(wl.overflow_bytes == 0 for wl in waterlines.values())
    resolved_active = (
        max(0, int(grid_blocks)) if active_grid_blocks is None else int(active_grid_blocks)
    )
    return ObjectivePhysicalLedger(
        analysis_mode=analysis_mode,
        is_legal=bool(is_legal and overflow_free),
        steady_state_ii=steady_state_ii,
        pipe_cycles=cleaned_pipes,
        dominant_pipe=compute_dominant_pipe(cleaned_pipes),
        waterlines=dict(waterlines),
        mte2_inflight_per_aiv=int(mte2_inflight_per_aiv),
        unaligned_dma_padding_bytes=int(unaligned_dma_padding_bytes),
        opaque_dynamic_regions=tuple(opaque_dynamic_regions),
        grid_blocks=int(grid_blocks),
        wave_occupancy=float(wave_occupancy),
        gm_roundtrip_bytes=int(gm_roundtrip_bytes),
        dominant_contig_bytes=int(dominant_contig_bytes),
        effective_cube_m=int(effective_cube_m),
        has_indirect_gather=bool(has_indirect_gather),
        has_lowbit_quant=bool(has_lowbit_quant),
        has_online_rescale_loop=bool(has_online_rescale_loop),
        in_loop_ub_reload_stores=int(in_loop_ub_reload_stores),
        uses_unreserved_ubuf=bool(uses_unreserved_ubuf),
        grid_is_symbolic=bool(grid_is_symbolic),
        min_gm_dma_bytes=int(min_gm_dma_bytes),
        dma_burst_efficiency_ratio=float(dma_burst_efficiency_ratio),
        uses_dynamic_offset_pipelining=bool(uses_dynamic_offset_pipelining),
        has_unlimit_memory_attr=bool(has_unlimit_memory_attr),
        has_coupled_reduce_sfu_accum=bool(has_coupled_reduce_sfu_accum),
        active_grid_blocks=resolved_active,
        in_loop_pipe_all_barriers=int(in_loop_pipe_all_barriers),
        ub_bank_conflict_risk=min(1.0, max(0.0, float(ub_bank_conflict_risk))),
        l1_bank_conflict_risk=min(1.0, max(0.0, float(l1_bank_conflict_risk))),
        ub_scalar_compute_loops=max(0, int(ub_scalar_compute_loops)),
        microarch_profile=dict(microarch_profile or {}),
        gm_traffic_scope=gm_traffic_scope,
    )


__all__ = [
    "ObjectivePhysicalLedger",
    "ScopeWaterline",
    "build_objective_physical_ledger",
    "build_scope_waterline",
    "compute_dominant_pipe",
]
