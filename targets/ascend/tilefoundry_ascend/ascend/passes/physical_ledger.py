"""Objective physical ledger exports for Ascend schedule and envelope inspection."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field

from tilefoundry_ascend.ascend.passes.observation_space import (
    CompilerSummaryLike,
    ObservationVector5D,
)


@dataclass(frozen=True)
class ScopeWaterline:
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

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dictionary representation."""
        return {
            "scope": self.scope,
            "capacity_bytes": self.capacity_bytes,
            "explicit_bytes": self.explicit_bytes,
            "implicit_bloat_bytes": self.implicit_bloat_bytes,
            "peak_allocated_bytes": self.peak_allocated_bytes,
            "overflow_bytes": self.overflow_bytes,
            "explicit_buffers": dict(self.explicit_buffers),
            "implicit_buffers": dict(self.implicit_buffers),
            "degraded_single_buffers": list(self.degraded_single_buffers),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ScopeWaterline:
        """Reconstruct a ScopeWaterline from a serialized dictionary."""
        raw_exp = payload.get("explicit_buffers") or {}
        raw_imp = payload.get("implicit_buffers") or {}
        raw_deg = payload.get("degraded_single_buffers") or ()
        return cls(
            scope=str(payload["scope"]),
            capacity_bytes=int(payload["capacity_bytes"]),
            explicit_bytes=int(payload["explicit_bytes"]),
            implicit_bloat_bytes=int(payload["implicit_bloat_bytes"]),
            peak_allocated_bytes=int(payload["peak_allocated_bytes"]),
            overflow_bytes=int(payload["overflow_bytes"]),
            explicit_buffers={str(k): int(v) for k, v in dict(raw_exp).items()},
            implicit_buffers={str(k): int(v) for k, v in dict(raw_imp).items()},
            degraded_single_buffers=tuple(str(item) for item in raw_deg),
        )


@dataclass(frozen=True)
class ObjectivePhysicalLedger:
    """Rule-blind hardware utilization, waterline, and legality telemetry."""

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
    microarch_profile: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.analysis_mode not in ("exact_modulo_z3", "readonly_envelope"):
            raise ValueError(
                f"analysis_mode must be 'exact_modulo_z3' or 'readonly_envelope', "
                f"got {self.analysis_mode!r}"
            )

    @property
    def max_overflow_bytes(self) -> int:
        """Return the maximum byte overflow across all tracked memory scopes."""
        return max((wl.overflow_bytes for wl in self.waterlines.values()), default=0)

    def to_observation_5d(
        self,
        *,
        compiler_summary: CompilerSummaryLike | None = None,
        telemetry: Mapping[str, object] | None = None,
    ) -> ObservationVector5D:
        """Project this L0 ledger plus L1/L2 telemetry onto the 5-axis observation space."""
        return ObservationVector5D.from_tiers(
            self,
            compiler_summary=compiler_summary,
            telemetry=telemetry,
        )

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable dictionary representation."""
        payload = asdict(self)
        payload["waterlines"] = {
            scope: wl.to_dict() for scope, wl in self.waterlines.items()
        }
        payload["opaque_dynamic_regions"] = list(self.opaque_dynamic_regions)
        return payload

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize the physical ledger into formatted JSON text."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ObjectivePhysicalLedger:
        """Reconstruct an ObjectivePhysicalLedger from a serialized dictionary."""
        raw_ii = payload.get("steady_state_ii")
        raw_pipes = payload.get("pipe_cycles") or {}
        raw_waterlines = payload.get("waterlines") or {}
        raw_opaque = payload.get("opaque_dynamic_regions") or ()
        waterlines = {
            str(scope): (
                wl if isinstance(wl, ScopeWaterline) else ScopeWaterline.from_dict(wl)
            )
            for scope, wl in dict(raw_waterlines).items()
        }
        return cls(
            analysis_mode=str(payload["analysis_mode"]),
            is_legal=bool(payload["is_legal"]),
            steady_state_ii=int(raw_ii) if raw_ii is not None else None,
            pipe_cycles={str(k): int(v) for k, v in dict(raw_pipes).items()},
            dominant_pipe=str(payload["dominant_pipe"]),
            waterlines=waterlines,
            mte2_inflight_per_aiv=int(payload.get("mte2_inflight_per_aiv", 0)),
            unaligned_dma_padding_bytes=int(payload.get("unaligned_dma_padding_bytes", 0)),
            opaque_dynamic_regions=tuple(str(item) for item in raw_opaque),
            grid_blocks=int(payload.get("grid_blocks", 1)),
            wave_occupancy=float(payload.get("wave_occupancy", 0.0)),
            gm_roundtrip_bytes=int(payload.get("gm_roundtrip_bytes", 0)),
            dominant_contig_bytes=int(payload.get("dominant_contig_bytes", 0)),
            effective_cube_m=int(payload.get("effective_cube_m", 0)),
            has_indirect_gather=bool(payload.get("has_indirect_gather", False)),
            has_lowbit_quant=bool(payload.get("has_lowbit_quant", False)),
            has_online_rescale_loop=bool(payload.get("has_online_rescale_loop", False)),
            in_loop_ub_reload_stores=int(payload.get("in_loop_ub_reload_stores", 0)),
            uses_unreserved_ubuf=bool(payload.get("uses_unreserved_ubuf", False)),
            grid_is_symbolic=bool(payload.get("grid_is_symbolic", False)),
            min_gm_dma_bytes=int(payload.get("min_gm_dma_bytes", 0)),
            dma_burst_efficiency_ratio=float(payload.get("dma_burst_efficiency_ratio", 1.0)),
            uses_dynamic_offset_pipelining=bool(
                payload.get("uses_dynamic_offset_pipelining", False)
            ),
            has_unlimit_memory_attr=bool(payload.get("has_unlimit_memory_attr", False)),
            has_coupled_reduce_sfu_accum=bool(payload.get("has_coupled_reduce_sfu_accum", False)),
            active_grid_blocks=int(
                payload.get("active_grid_blocks", payload.get("grid_blocks", 1))
            ),
            in_loop_pipe_all_barriers=int(payload.get("in_loop_pipe_all_barriers", 0)),
            ub_bank_conflict_risk=float(payload.get("ub_bank_conflict_risk", 0.0)),
            l1_bank_conflict_risk=float(payload.get("l1_bank_conflict_risk", 0.0)),
            microarch_profile=dict(payload.get("microarch_profile") or {}),
        )


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
    microarch_profile: Mapping[str, object] | None = None,
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
        microarch_profile=dict(microarch_profile or {}),
    )


__all__ = [
    "ObjectivePhysicalLedger",
    "ScopeWaterline",
    "build_objective_physical_ledger",
    "build_scope_waterline",
    "compute_dominant_pipe",
]
