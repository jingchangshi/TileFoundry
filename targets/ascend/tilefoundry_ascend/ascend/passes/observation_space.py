"""Five-axis orthogonal MECE observation space spanning L0 ledger, L1 compiler, and L2 telemetry."""

from __future__ import annotations

import inspect
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Protocol, runtime_checkable

from tilefoundry_ascend.ascend.sil.schema import ConstraintRelaxationMask

_L2_FIELD_SOURCES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("l2_measured_latency_us", ("measured_latency_us", "dur_us")),
    ("l2_aic_mac_ratio", ("aic_mac_ratio", "aic_mad")),
    ("l2_aiv_vec_ratio", ("aiv_vec_ratio", "aiv_vec")),
    ("l2_roofline_eta", ("roofline_eta",)),
    ("l2_mte2_wait_ratio", ("mte2_wait_ratio", "aiv_mte2", "aic_mte2")),
    ("l2_fixpipe_ratio", ("aic_fixpipe",)),
    ("l2_cache_hit_rate", ("l2_hit_rate", "l2_cache_hit_rate")),
    (
        "l2_bank_conflict_ratio",
        ("bank_conflict_ratio", "ub_bank_conflict_ratio", "resource_conflict_ratio"),
    ),
    ("l2_sync_stall_ratio", ("sync_stall_ratio",)),
    ("l2_pseudo_double_buffer", ("pseudo_double_buffer_serialized",)),
)

_PMU_L2_FIELDS: frozenset[str] = frozenset(
    {
        "l2_aic_mac_ratio",
        "l2_aiv_vec_ratio",
        "l2_mte2_wait_ratio",
        "l2_fixpipe_ratio",
        "l2_cache_hit_rate",
        "l2_bank_conflict_ratio",
        "l2_sync_stall_ratio",
        "l2_pseudo_double_buffer",
    }
)

_VALID_COMPUTE_ENGINES: frozenset[str] = frozenset({"aiv_only", "aic_only", "coupled_1c2v"})


def _normalize_compute_engine(raw: object) -> str:
    """Normalize a compute engine label to a canonical 3D topology tag or empty string."""
    if raw is None:
        return ""
    cleaned = str(raw).strip()
    return cleaned if cleaned in _VALID_COMPUTE_ENGINES else ""


def _resolve_compute_engine(
    *,
    topology_axes: object | None,
    telemetry: Mapping[str, object],
    ledger: PhysicalLedgerLike,
    cube_cycles: int,
    vec_cycles: int,
    l2_aic_mac: float,
    l2_aiv_vec: float,
) -> str:
    """Resolve compute_engine from topology_axes, telemetry, ledger profile, or cycle counts."""
    for source in (topology_axes, telemetry.get("topology_axes")):
        if isinstance(source, Mapping):
            candidate = _normalize_compute_engine(source.get("compute_engine"))
            if candidate:
                return candidate
        elif source is not None:
            candidate = _normalize_compute_engine(getattr(source, "compute_engine", ""))
            if candidate:
                return candidate
    for raw in (
        telemetry.get("compute_engine"),
        ledger.microarch_profile.get("compute_engine"),
        getattr(ledger, "compute_engine", ""),
    ):
        candidate = _normalize_compute_engine(raw)
        if candidate:
            return candidate
    has_cube = cube_cycles > 0 or l2_aic_mac > 0.0
    has_vec = vec_cycles > 0 or l2_aiv_vec > 0.0
    if has_cube and has_vec:
        return "coupled_1c2v"
    if has_cube:
        return "aic_only"
    if has_vec:
        return "aiv_only"
    return ""


def _resolve_num_stages(
    telemetry: Mapping[str, object],
    ledger: PhysicalLedgerLike,
) -> int:
    """Resolve pipeline stage count from telemetry, ledger profile, or default 1."""
    for raw in (
        telemetry.get("num_stages"),
        telemetry.get("pipeline_depth"),
        ledger.microarch_profile.get("num_stages"),
        ledger.microarch_profile.get("pipeline_depth"),
        getattr(ledger, "num_stages", None),
        getattr(ledger, "pipeline_depth", None),
    ):
        val = _as_int(raw, 0)
        if val > 0:
            return val
    return 1


class RooflineSummaryLike(Protocol):
    """Structural view of analytical HIR Roofline bounds consumed by from_tiers."""

    @property
    def min_gm_bytes(self) -> int: ...

    @property
    def compute_floor_us(self) -> float: ...


@runtime_checkable
class _FullGridTrafficLedgerLike(Protocol):
    """Optional ledger extension providing explicit full-grid GM traffic totals."""

    def total_gm_bytes(self, grid_blocks: int | None = None) -> int: ...


@runtime_checkable
class _ScopedTrafficLedgerLike(Protocol):
    """Optional ledger extension declaring single-tile versus full-grid GM traffic scope."""

    @property
    def gm_traffic_scope(self) -> str: ...


@runtime_checkable
class _ActiveGridLedgerLike(Protocol):
    """Optional ledger extension reporting active launched grid blocks."""

    @property
    def active_grid_blocks(self) -> int: ...


class ScopeWaterlineLike(Protocol):
    """Structural view of one on-chip memory waterline consumed by the observation vector."""

    @property
    def peak_allocated_bytes(self) -> int: ...

    @property
    def degraded_single_buffers(self) -> tuple[str, ...]: ...


class PhysicalLedgerLike(Protocol):
    """Structural view of the L0 objective physical ledger fields consumed by from_tiers."""

    @property
    def pipe_cycles(self) -> Mapping[str, int]: ...

    @property
    def waterlines(self) -> Mapping[str, ScopeWaterlineLike]: ...

    @property
    def gm_roundtrip_bytes(self) -> int: ...

    @property
    def dominant_contig_bytes(self) -> int: ...

    @property
    def dma_burst_efficiency_ratio(self) -> float: ...

    @property
    def mte2_inflight_per_aiv(self) -> int: ...

    @property
    def unaligned_dma_padding_bytes(self) -> int: ...

    @property
    def max_overflow_bytes(self) -> int: ...

    @property
    def ub_bank_conflict_risk(self) -> float: ...

    @property
    def l1_bank_conflict_risk(self) -> float: ...

    @property
    def steady_state_ii(self) -> int | None: ...

    @property
    def in_loop_pipe_all_barriers(self) -> int: ...

    @property
    def uses_dynamic_offset_pipelining(self) -> bool: ...

    @property
    def grid_blocks(self) -> int: ...

    @property
    def wave_occupancy(self) -> float: ...

    @property
    def has_indirect_gather(self) -> bool: ...

    @property
    def has_online_rescale_loop(self) -> bool: ...

    @property
    def has_lowbit_quant(self) -> bool: ...

    @property
    def ub_scalar_compute_loops(self) -> int: ...

    @property
    def min_gm_dma_bytes(self) -> int: ...

    @property
    def microarch_profile(self) -> Mapping[str, object]: ...


class CompilerSummaryLike(Protocol):
    """Structural view of the L1 bisheng resource-summary fields consumed by from_tiers."""

    @property
    def vec_regs(self) -> int: ...

    @property
    def scalar_regs(self) -> int: ...

    @property
    def spill_bytes(self) -> int: ...

    @property
    def stack_bytes(self) -> int: ...

    @property
    def vloopv2_loops(self) -> int: ...

    @property
    def software_loops(self) -> int: ...

    @property
    def has_local_membar_in_vf_call(self) -> bool: ...

    @property
    def vf_fusion_breaks(self) -> int: ...

    @property
    def alias_serialized_buffers(self) -> int: ...

    @property
    def dual_issue_ipc_collapse(self) -> float: ...

    @property
    def dcache_dcci_flushes(self) -> int: ...

    @property
    def extra_signals(self) -> Mapping[str, float]: ...


def _as_int(value: object, default: int = 0) -> int:
    """Narrow a deserialized value to a Python int without accepting booleans."""
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return default


def _as_float(value: object, default: float = 0.0) -> float:
    """Narrow a deserialized value to a Python float without accepting booleans."""
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _as_bool(value: object, default: bool = False) -> bool:
    """Narrow a deserialized value to a Python bool."""
    return value if isinstance(value, bool) else default


def _as_str_tuple(value: object) -> tuple[str, ...]:
    """Narrow a deserialized sequence of field names to a tuple of strings."""
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _require_mapping(value: object, field_name: str) -> Mapping[str, object]:
    """Return a serialized axis payload as a mapping or fail with ValueError."""
    if not isinstance(value, Mapping):
        raise ValueError(f"observation field {field_name!r} must be a mapping")
    return value


def _telemetry_float(telemetry: Mapping[str, object], *keys: str, default: float = 0.0) -> float:
    """Read the first present finite numeric telemetry key, otherwise return the default."""
    for key in keys:
        if key not in telemetry:
            continue
        value = telemetry[key]
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return default


def _telemetry_bool(telemetry: Mapping[str, object], key: str, default: bool = False) -> bool:
    """Read a boolean L2 telemetry flag, otherwise return the neutral default."""
    value = telemetry.get(key)
    return value if isinstance(value, bool) else default


def _has_measured_l2_field(
    telemetry: Mapping[str, object], field_name: str, keys: tuple[str, ...]
) -> bool:
    """Return True only when telemetry carries a genuine non-null value for an L2 field."""
    if field_name == "l2_pseudo_double_buffer":
        return any(isinstance(telemetry.get(key), bool) for key in keys)
    for key in keys:
        if key not in telemetry:
            continue
        value = telemetry[key]
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return True
    return False


def _resolve_mte2_wait_ratio(telemetry: Mapping[str, object]) -> float:
    """Resolve MTE2 wait ratio from explicit counter or max of AIV/AIC MTE2 counters."""
    if _has_measured_l2_field(telemetry, "l2_mte2_wait_ratio", ("mte2_wait_ratio",)):
        return _telemetry_float(telemetry, "mte2_wait_ratio")
    candidates: list[float] = []
    for key in ("aiv_mte2", "aic_mte2"):
        if _has_measured_l2_field(telemetry, "l2_mte2_wait_ratio", (key,)):
            candidates.append(_telemetry_float(telemetry, key))
    return max(candidates) if candidates else 0.0


def _resolve_static_ledger_gm_bytes(ledger: PhysicalLedgerLike) -> int:
    """Resolve P0 full-grid GM traffic bytes from the static physical ledger."""
    grid_count = (
        ledger.active_grid_blocks
        if isinstance(ledger, _ActiveGridLedgerLike) and ledger.active_grid_blocks > 0
        else ledger.grid_blocks
    )
    effective_blocks = max(1, int(grid_count))
    if isinstance(ledger, _FullGridTrafficLedgerLike):
        total_fn = ledger.total_gm_bytes
        if callable(total_fn):
            sig = inspect.signature(total_fn)
            positional = [
                param
                for param in sig.parameters.values()
                if param.kind
                in (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                )
            ]
            if positional and positional[0].default is inspect.Parameter.empty:
                return max(0, int(total_fn(effective_blocks)))
            return max(0, int(total_fn()))
        if isinstance(total_fn, int) and not isinstance(total_fn, bool):
            return max(0, total_fn)
    scope = (
        ledger.gm_traffic_scope if isinstance(ledger, _ScopedTrafficLedgerLike) else "per_tile"
    )
    raw_bytes = max(0, int(ledger.gm_roundtrip_bytes))
    if scope == "full_grid":
        return raw_bytes
    return raw_bytes * effective_blocks


def _resolve_roofline_bounds(
    roofline: RooflineSummaryLike | Mapping[str, object] | None,
    telemetry: Mapping[str, object],
) -> tuple[float, float]:
    """Resolve blueprint minimum GM bytes and compute floor microseconds."""
    min_gm = 0.0
    compute_floor_us = 0.0
    if roofline is not None:
        if isinstance(roofline, Mapping):
            min_gm = _as_float(roofline.get("min_gm_bytes"))
            compute_floor_us = _as_float(roofline.get("compute_floor_us"))
        else:
            min_gm = float(roofline.min_gm_bytes)
            compute_floor_us = float(roofline.compute_floor_us)
    if min_gm <= 0.0:
        min_gm = _telemetry_float(telemetry, "blueprint_min_gm_bytes")
    if compute_floor_us <= 0.0:
        compute_floor_us = _telemetry_float(telemetry, "blueprint_compute_floor_us")
    return max(0.0, min_gm), max(0.0, compute_floor_us)


def _resolve_measured_hbm_bytes(telemetry: Mapping[str, object]) -> float:
    """Resolve measured HBM traffic bytes when explicitly present and positive."""
    explicit_actual = _telemetry_float(telemetry, "actual_gm_bytes")
    if explicit_actual > 0.0:
        return explicit_actual
    hbm_read = _telemetry_float(telemetry, "hbm_read_bytes")
    hbm_write = _telemetry_float(telemetry, "hbm_write_bytes")
    if hbm_read + hbm_write > 0.0:
        return hbm_read + hbm_write
    return 0.0


def compute_transfer_inflation_ratio(
    ledger: PhysicalLedgerLike,
    *,
    telemetry: Mapping[str, object] | None = None,
    roofline: RooflineSummaryLike | Mapping[str, object] | None = None,
) -> float:
    """Compute HBM transfer inflation from measured HBM traffic or static roundtrip bytes."""
    tel = telemetry if telemetry is not None else {}
    blueprint_min_gm_bytes, _ = _resolve_roofline_bounds(roofline, tel)
    if blueprint_min_gm_bytes <= 0.0:
        return max(0.0, _telemetry_float(tel, "transfer_inflation_ratio"))
    measured_hbm_bytes = _resolve_measured_hbm_bytes(tel)
    if measured_hbm_bytes > 0.0:
        return measured_hbm_bytes / max(1.0, float(blueprint_min_gm_bytes))
    extra_roundtrip_bytes = float(_resolve_static_ledger_gm_bytes(ledger))
    return (float(blueprint_min_gm_bytes) + extra_roundtrip_bytes) / max(
        1.0, float(blueprint_min_gm_bytes)
    )


@dataclass(frozen=True)
class ComputeIssueAxis:
    """Axis 1: AIC/AIV compute pipe utilization and vector/instruction issue efficiency."""

    l0_cube_cycles: int = 0
    l0_vec_cycles: int = 0
    l0_cube_to_vec_ratio: float = 0.0
    l1_vloopv2_loops: int = 0
    l1_software_loops: int = 0
    l1_vf_fusion_breaks: int = 0
    l1_dual_issue_collapse: float = 0.0
    l2_aic_mac_ratio: float = 0.0
    l2_aiv_vec_ratio: float = 0.0
    l2_roofline_eta: float = 0.0
    effective_compute_eta: float = 0.0
    l0_aiv_subcore_imbalance: bool = False
    l0_ub_scalar_compute_loops: int = 0
    l1_scalar_compute_loop_fallback: float = 0.0


@dataclass(frozen=True)
class OffChipMemoryDmaAxis:
    """Axis 2: HBM/L2 traffic, DMA burst coalescing, MTE2 queue depth, and FixPipe drain."""

    l0_gm_roundtrip_bytes: int = 0
    l0_dominant_contig_bytes: int = 0
    l0_dma_burst_efficiency: float = 1.0
    l0_mte2_inflight_per_aiv: int = 0
    l0_unaligned_padding_bytes: int = 0
    l1_dcache_dcci_flushes: int = 0
    l1_objdump_mte2_bytes: int = 0
    l2_mte2_wait_ratio: float = 0.0
    l2_fixpipe_ratio: float = 0.0
    l2_cache_hit_rate: float = 1.0
    transfer_inflation_ratio: float = 0.0
    l0_min_gm_dma_bytes: int = 0
    l0_fixpipe_cycles: int = 0
    l1_dcache_bypass_narrow_store: float = 0.0


@dataclass(frozen=True)
class OnChipSramRegAxis:
    """Axis 3: L1/L0/UB/SSBUF waterlines, bank conflicts, register pressure, and spills."""

    l0_max_overflow_bytes: int = 0
    l0_ub_peak_bytes: int = 0
    l0_l1_peak_bytes: int = 0
    l0_ub_bank_conflict_risk: float = 0.0
    l0_l1_bank_conflict_risk: float = 0.0
    l0_degraded_single_buffers: int = 0
    l1_vec_regs: int = 0
    l1_scalar_regs: int = 0
    l1_spill_bytes: int = 0
    l1_stack_bytes: int = 0
    l2_bank_conflict_ratio: float = 0.0


@dataclass(frozen=True)
class PipelineSyncAxis:
    """Axis 4: Modulo initiation interval, multi-buffer overlap, and cross-pipe barriers."""

    l0_steady_state_ii: int = 0
    l0_in_loop_pipe_all_barriers: int = 0
    l0_uses_dynamic_offset_pipe: bool = False
    l1_local_membar_in_vf: bool = False
    l1_alias_serialized_buffers: int = 0
    l2_sync_stall_ratio: float = 0.0
    l2_pseudo_double_buffer: bool = False


@dataclass(frozen=True)
class GridIrregularityAxis:
    """Axis 5: Block concurrency, wave quantization, indirect gather, and recurrence."""

    l0_grid_blocks: int = 1
    l0_wave_occupancy: float = 1.0
    l0_has_indirect_gather: bool = False
    l0_has_online_rescale: bool = False
    l0_has_lowbit_quant: bool = False
    l1_subkernel_count: int = 1
    l2_measured_latency_us: float = 0.0


@dataclass(frozen=True)
class AxisResidual5D:
    """Normalized [0, 1] co-axial bottleneck residuals driving 1:1 AST facet relaxation."""

    compute_issue_gap: float
    offchip_dma_gap: float
    onchip_sram_reg_gap: float
    pipeline_sync_gap: float
    grid_wave_gap: float

    @property
    def dominant_axis(self) -> str:
        """Return the name of the highest-loss physical observation axis."""
        pairs = (
            ("compute_issue", self.compute_issue_gap),
            ("offchip_dma", self.offchip_dma_gap),
            ("onchip_sram_reg", self.onchip_sram_reg_gap),
            ("pipeline_sync", self.pipeline_sync_gap),
            ("grid_wave", self.grid_wave_gap),
        )
        return max(pairs, key=lambda item: item[1])[0]

    def unlock_causal_facets(self, current: ConstraintRelaxationMask) -> ConstraintRelaxationMask:
        """Map active axis residuals 1:1 onto minimal MECE 7-facet permission unlocks."""
        to_unlock: list[str] = []
        if self.compute_issue_gap >= 0.45:
            to_unlock.append("alg_compute_and_monoid")
        if self.onchip_sram_reg_gap >= 0.45 or self.grid_wave_gap >= 0.40:
            to_unlock.append("tile_and_stage_knobs")
        if self.compute_issue_gap >= 0.25:
            to_unlock.append("exec_mode_and_spr")
        if self.offchip_dma_gap >= 0.25:
            to_unlock.append("dma_and_cache_hierarchy")
        if self.onchip_sram_reg_gap >= 0.20:
            to_unlock.append("sram_layout_and_aliasing")
        if self.pipeline_sync_gap >= 0.20:
            to_unlock.append("pipeline_and_sync_graph")
        if self.grid_wave_gap >= 0.25:
            to_unlock.append("grid_and_workspace_topology")
        return current.unlock(*to_unlock) if to_unlock else current


@dataclass(frozen=True)
class ObservationVector5D:
    """Complete 5-axis hardware observation vector unifying L0, L1, and L2 telemetry."""

    compute_issue: ComputeIssueAxis
    offchip_dma: OffChipMemoryDmaAxis
    onchip_sram_reg: OnChipSramRegAxis
    pipeline_sync: PipelineSyncAxis
    grid_irregularity: GridIrregularityAxis
    has_l2_telemetry: bool = False
    has_l1_compiler_summary: bool = False
    measured_l2_fields: tuple[str, ...] = ()
    compute_engine: str = ""
    num_stages: int = 1
    has_l2_pmu: bool = False

    def __post_init__(self) -> None:
        normalized_engine = _normalize_compute_engine(self.compute_engine)
        if normalized_engine != self.compute_engine:
            object.__setattr__(self, "compute_engine", normalized_engine)
        normalized_stages = max(1, int(self.num_stages))
        if normalized_stages != self.num_stages:
            object.__setattr__(self, "num_stages", normalized_stages)
        if not self.has_l2_pmu and self.has_l2_telemetry:
            derived_pmu = any(f in _PMU_L2_FIELDS for f in self.measured_l2_fields) or (
                not self.measured_l2_fields
                and (
                    self.compute_issue.l2_aiv_vec_ratio > 0.0
                    or self.compute_issue.l2_aic_mac_ratio > 0.0
                    or self.offchip_dma.l2_mte2_wait_ratio > 0.0
                    or self.pipeline_sync.l2_sync_stall_ratio > 0.0
                )
            )
            if derived_pmu:
                object.__setattr__(self, "has_l2_pmu", True)

    @property
    def transfer_inflation_ratio(self) -> float:
        """Return actual GM bytes divided by the blueprint minimum GM bytes."""
        return self.offchip_dma.transfer_inflation_ratio

    @property
    def effective_compute_eta(self) -> float:
        """Return blueprint compute floor latency divided by measured kernel latency."""
        return self.compute_issue.effective_compute_eta

    @classmethod
    def from_tiers(
        cls,
        ledger: PhysicalLedgerLike,
        *,
        compiler_summary: CompilerSummaryLike | None = None,
        telemetry: Mapping[str, object] | None = None,
        roofline: RooflineSummaryLike | Mapping[str, object] | None = None,
        topology_axes: object | None = None,
    ) -> ObservationVector5D:
        """Assemble the 5-axis vector from L0 ledger, L1 summary, L2 telemetry, and Roofline."""
        tel = telemetry if telemetry is not None else {}
        cube_cycles = _as_int(ledger.pipe_cycles.get("AIC.PIPE_M")) + _as_int(
            ledger.pipe_cycles.get("AIC.CUBE")
        )
        aiv0_v = _as_int(ledger.pipe_cycles.get("AIV0.PIPE_V"))
        aiv1_v = _as_int(ledger.pipe_cycles.get("AIV1.PIPE_V"))
        vec_cycles = aiv0_v + aiv1_v
        fixpipe_cycles = (
            _as_int(ledger.pipe_cycles.get("AIC.PIPE_FIX"))
            if "AIC.PIPE_FIX" in ledger.pipe_cycles
            else _as_int(ledger.pipe_cycles.get("PIPE_FIX"))
        )
        ub_waterline = ledger.waterlines.get("UB")
        l1_waterline = ledger.waterlines.get("L1")
        degraded_count = sum(
            len(waterline.degraded_single_buffers) for waterline in ledger.waterlines.values()
        )

        compiler = compiler_summary
        vec_regs = compiler.vec_regs if compiler is not None else 0
        scalar_regs = compiler.scalar_regs if compiler is not None else 0
        spill_bytes = compiler.spill_bytes if compiler is not None else 0
        stack_bytes = compiler.stack_bytes if compiler is not None else 0
        vloopv2_loops = compiler.vloopv2_loops if compiler is not None else 0
        software_loops = compiler.software_loops if compiler is not None else 0
        local_membar = compiler.has_local_membar_in_vf_call if compiler is not None else False
        vf_fusion_breaks = compiler.vf_fusion_breaks if compiler is not None else 0
        alias_serialized = compiler.alias_serialized_buffers if compiler is not None else 0
        dual_issue_collapse = compiler.dual_issue_ipc_collapse if compiler is not None else 0.0
        dcci_flushes = compiler.dcache_dcci_flushes if compiler is not None else 0
        raw_extra = getattr(compiler, "extra_signals", None) if compiler is not None else None
        extra_signals = raw_extra if isinstance(raw_extra, Mapping) else {}
        ub_loops = max(0, _as_int(getattr(ledger, "ub_scalar_compute_loops", 0)))
        min_dma = max(0, _as_int(getattr(ledger, "min_gm_dma_bytes", 0)))

        subkernel_count = max(1, _as_int(ledger.microarch_profile.get("subkernel_count"), 1))
        mte2_bytes_hint = _as_int(ledger.microarch_profile.get("mte2_bytes_hint"), 0)
        measured_latency_us = _telemetry_float(tel, "measured_latency_us", "dur_us")
        measured_fields = tuple(
            field_name
            for field_name, keys in _L2_FIELD_SOURCES
            if _has_measured_l2_field(tel, field_name, keys)
        )
        has_l2_telemetry = bool(measured_fields) and not _telemetry_bool(
            tel, "is_analytical_proxy"
        )
        measured_l2_fields = measured_fields if has_l2_telemetry else ()

        _blueprint_min_gm_bytes, blueprint_compute_floor_us = _resolve_roofline_bounds(
            roofline, tel
        )
        transfer_inflation_ratio = compute_transfer_inflation_ratio(
            ledger,
            telemetry=tel,
            roofline=roofline,
        )

        if blueprint_compute_floor_us > 0.0 and measured_latency_us > 0.0:
            effective_compute_eta = blueprint_compute_floor_us / max(1e-3, measured_latency_us)
        else:
            effective_compute_eta = max(0.0, _telemetry_float(tel, "effective_compute_eta"))

        scalar_loop_fallback = max(
            float(ub_loops),
            _as_float(extra_signals.get("scalar_compute_loop_fallback")),
            _telemetry_float(tel, "scalar_compute_loop_fallback"),
        )
        narrow_store_bypass = max(
            _as_float(extra_signals.get("dcache_bypass_narrow_store")),
            _telemetry_float(tel, "dcache_bypass_narrow_store"),
            (32.0 / float(min_dma)) if 0 < min_dma < 32 else 0.0,
        )

        l2_aic_mac = _telemetry_float(tel, "aic_mac_ratio", "aic_mad")
        l2_aiv_vec = _telemetry_float(tel, "aiv_vec_ratio", "aiv_vec")
        compute_engine = _resolve_compute_engine(
            topology_axes=topology_axes,
            telemetry=tel,
            ledger=ledger,
            cube_cycles=cube_cycles,
            vec_cycles=vec_cycles,
            l2_aic_mac=l2_aic_mac,
            l2_aiv_vec=l2_aiv_vec,
        )
        num_stages = _resolve_num_stages(tel, ledger)
        has_l2_pmu = bool(has_l2_telemetry) and any(
            field_name in _PMU_L2_FIELDS for field_name in measured_l2_fields
        )

        return cls(
            compute_issue=ComputeIssueAxis(
                l0_cube_cycles=cube_cycles,
                l0_vec_cycles=vec_cycles,
                l0_cube_to_vec_ratio=float(cube_cycles) / float(max(1, vec_cycles)),
                l1_vloopv2_loops=vloopv2_loops,
                l1_software_loops=software_loops,
                l1_vf_fusion_breaks=vf_fusion_breaks,
                l1_dual_issue_collapse=dual_issue_collapse,
                l2_aic_mac_ratio=l2_aic_mac,
                l2_aiv_vec_ratio=l2_aiv_vec,
                l2_roofline_eta=_telemetry_float(tel, "roofline_eta"),
                effective_compute_eta=effective_compute_eta,
                l0_aiv_subcore_imbalance=(aiv0_v > 0 and aiv1_v == 0),
                l0_ub_scalar_compute_loops=ub_loops,
                l1_scalar_compute_loop_fallback=scalar_loop_fallback,
            ),
            offchip_dma=OffChipMemoryDmaAxis(
                l0_gm_roundtrip_bytes=ledger.gm_roundtrip_bytes,
                l0_dominant_contig_bytes=ledger.dominant_contig_bytes,
                l0_dma_burst_efficiency=ledger.dma_burst_efficiency_ratio,
                l0_mte2_inflight_per_aiv=ledger.mte2_inflight_per_aiv,
                l0_unaligned_padding_bytes=ledger.unaligned_dma_padding_bytes,
                l1_dcache_dcci_flushes=dcci_flushes,
                l1_objdump_mte2_bytes=mte2_bytes_hint,
                l2_mte2_wait_ratio=_resolve_mte2_wait_ratio(tel),
                l2_fixpipe_ratio=_telemetry_float(tel, "aic_fixpipe"),
                l2_cache_hit_rate=_telemetry_float(
                    tel, "l2_hit_rate", "l2_cache_hit_rate", default=1.0
                ),
                transfer_inflation_ratio=transfer_inflation_ratio,
                l0_min_gm_dma_bytes=min_dma,
                l0_fixpipe_cycles=fixpipe_cycles,
                l1_dcache_bypass_narrow_store=narrow_store_bypass,
            ),
            onchip_sram_reg=OnChipSramRegAxis(
                l0_max_overflow_bytes=ledger.max_overflow_bytes,
                l0_ub_peak_bytes=(
                    ub_waterline.peak_allocated_bytes if ub_waterline is not None else 0
                ),
                l0_l1_peak_bytes=(
                    l1_waterline.peak_allocated_bytes if l1_waterline is not None else 0
                ),
                l0_ub_bank_conflict_risk=ledger.ub_bank_conflict_risk,
                l0_l1_bank_conflict_risk=ledger.l1_bank_conflict_risk,
                l0_degraded_single_buffers=degraded_count,
                l1_vec_regs=vec_regs,
                l1_scalar_regs=scalar_regs,
                l1_spill_bytes=spill_bytes,
                l1_stack_bytes=stack_bytes,
                l2_bank_conflict_ratio=_telemetry_float(
                    tel,
                    "bank_conflict_ratio",
                    "ub_bank_conflict_ratio",
                    "resource_conflict_ratio",
                ),
            ),
            pipeline_sync=PipelineSyncAxis(
                l0_steady_state_ii=int(ledger.steady_state_ii or 0),
                l0_in_loop_pipe_all_barriers=ledger.in_loop_pipe_all_barriers,
                l0_uses_dynamic_offset_pipe=ledger.uses_dynamic_offset_pipelining,
                l1_local_membar_in_vf=local_membar,
                l1_alias_serialized_buffers=alias_serialized,
                l2_sync_stall_ratio=_telemetry_float(tel, "sync_stall_ratio"),
                l2_pseudo_double_buffer=_telemetry_bool(tel, "pseudo_double_buffer_serialized"),
            ),
            grid_irregularity=GridIrregularityAxis(
                l0_grid_blocks=ledger.grid_blocks,
                l0_wave_occupancy=ledger.wave_occupancy,
                l0_has_indirect_gather=ledger.has_indirect_gather,
                l0_has_online_rescale=ledger.has_online_rescale_loop,
                l0_has_lowbit_quant=ledger.has_lowbit_quant,
                l1_subkernel_count=subkernel_count,
                l2_measured_latency_us=measured_latency_us,
            ),
            has_l2_telemetry=has_l2_telemetry,
            has_l1_compiler_summary=compiler is not None,
            measured_l2_fields=measured_l2_fields,
            compute_engine=compute_engine,
            num_stages=num_stages,
            has_l2_pmu=has_l2_pmu,
        )

    def compute_axis_residuals(self) -> AxisResidual5D:
        """Compute normalized [0, 1] co-axial static-dynamic residuals across all 5 axes."""
        has_sw_loop = (
            self.compute_issue.l1_software_loops > 0
            and self.compute_issue.l1_vloopv2_loops == 0
        )
        sw_loop_penalty = 0.5 if has_sw_loop else 0.0
        compute_utilization = max(
            self.compute_issue.l2_aic_mac_ratio,
            self.compute_issue.l2_aiv_vec_ratio,
        )
        eff_eta = (
            self.compute_issue.l2_roofline_eta
            if self.compute_issue.l2_roofline_eta > 0.0
            else self.compute_issue.effective_compute_eta
        )
        roofline_gap = (
            1.0 - eff_eta
            if compute_utilization >= 0.5 and 0.0 < eff_eta < 0.5
            else 0.0
        )
        scalar_trap = (
            self.compute_issue.l0_cube_cycles == 0
            and self.compute_issue.l0_vec_cycles == 0
        ) or (
            self.has_l1_compiler_summary
            and self.onchip_sram_reg.l1_vec_regs == 0
            and self.compute_issue.l2_aiv_vec_ratio < 0.02
            and self.compute_issue.l2_aic_mac_ratio == 0.0
        )
        scalar_loop_hazard = min(
            1.0,
            0.5
            * max(
                0.0,
                float(self.compute_issue.l0_ub_scalar_compute_loops),
                self.compute_issue.l1_scalar_compute_loop_fallback,
            ),
        )
        compute_gap = min(
            1.0,
            max(
                1.0 if scalar_trap else 0.0,
                sw_loop_penalty,
                scalar_loop_hazard,
                self.compute_issue.l1_dual_issue_collapse,
                0.25 * min(4, self.compute_issue.l1_vf_fusion_breaks),
                roofline_gap,
            ),
        )
        inflation_gap = (
            min(1.0, max(0.0, self.offchip_dma.transfer_inflation_ratio - 1.0))
            if self.offchip_dma.transfer_inflation_ratio > 1.30
            else 0.0
        )
        narrow_store_hazard = min(
            1.0,
            max(0.0, self.offchip_dma.l1_dcache_bypass_narrow_store) / 4.0,
        )
        dma_gap = min(
            1.0,
            max(
                self.offchip_dma.l2_mte2_wait_ratio,
                1.0 - self.offchip_dma.l0_dma_burst_efficiency,
                1.0 - self.offchip_dma.l2_cache_hit_rate,
                0.35 if self.offchip_dma.l1_dcache_dcci_flushes > 0 else 0.0,
                inflation_gap,
                narrow_store_hazard,
            ),
        )
        sram_gap = min(
            1.0,
            max(
                1.0 if self.onchip_sram_reg.l0_max_overflow_bytes > 0 else 0.0,
                min(1.0, float(self.onchip_sram_reg.l1_spill_bytes) / 256.0),
                min(1.0, max(0.0, float(self.onchip_sram_reg.l1_stack_bytes)) / 1024.0),
                self.onchip_sram_reg.l2_bank_conflict_ratio,
                self.onchip_sram_reg.l0_ub_bank_conflict_risk,
            ),
        )
        sync_gap = min(
            1.0,
            max(
                self.pipeline_sync.l2_sync_stall_ratio,
                0.6 if self.pipeline_sync.l1_local_membar_in_vf else 0.0,
                0.5 if self.pipeline_sync.l2_pseudo_double_buffer else 0.0,
                0.25 * min(4, self.pipeline_sync.l1_alias_serialized_buffers),
            ),
        )
        wave_occupancy = self.grid_irregularity.l0_wave_occupancy
        wave_waste = (
            1.0 - wave_occupancy / math.ceil(wave_occupancy)
            if wave_occupancy > 0.0
            else 0.0
        )
        grid_gap = min(1.0, max(0.0, wave_waste))
        return AxisResidual5D(
            compute_issue_gap=round(compute_gap, 4),
            offchip_dma_gap=round(dma_gap, 4),
            onchip_sram_reg_gap=round(sram_gap, 4),
            pipeline_sync_gap=round(sync_gap, 4),
            grid_wave_gap=round(grid_gap, 4),
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize all 5 observation axes, orthogonal ratios, residuals, and provenance."""
        return {
            "compute_issue": asdict(self.compute_issue),
            "offchip_dma": asdict(self.offchip_dma),
            "onchip_sram_reg": asdict(self.onchip_sram_reg),
            "pipeline_sync": asdict(self.pipeline_sync),
            "grid_irregularity": asdict(self.grid_irregularity),
            "transfer_inflation_ratio": self.transfer_inflation_ratio,
            "effective_compute_eta": self.effective_compute_eta,
            "has_l2_telemetry": self.has_l2_telemetry,
            "has_l1_compiler_summary": self.has_l1_compiler_summary,
            "measured_l2_fields": list(self.measured_l2_fields),
            "compute_engine": self.compute_engine,
            "num_stages": self.num_stages,
            "has_l2_pmu": self.has_l2_pmu,
            "residuals": asdict(self.compute_axis_residuals()),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ObservationVector5D:
        """Reconstruct the 5-axis observation vector from its serialized dictionary form."""
        compute_payload = _require_mapping(payload.get("compute_issue"), "compute_issue")
        dma_payload = _require_mapping(payload.get("offchip_dma"), "offchip_dma")
        sram_payload = _require_mapping(payload.get("onchip_sram_reg"), "onchip_sram_reg")
        sync_payload = _require_mapping(payload.get("pipeline_sync"), "pipeline_sync")
        grid_payload = _require_mapping(payload.get("grid_irregularity"), "grid_irregularity")
        raw_eff_eta = compute_payload.get("effective_compute_eta")
        if raw_eff_eta is None:
            raw_eff_eta = payload.get("effective_compute_eta")
        raw_inflation = dma_payload.get("transfer_inflation_ratio")
        if raw_inflation is None:
            raw_inflation = payload.get("transfer_inflation_ratio")
        return cls(
            compute_issue=ComputeIssueAxis(
                l0_cube_cycles=_as_int(compute_payload.get("l0_cube_cycles")),
                l0_vec_cycles=_as_int(compute_payload.get("l0_vec_cycles")),
                l0_cube_to_vec_ratio=_as_float(compute_payload.get("l0_cube_to_vec_ratio")),
                l1_vloopv2_loops=_as_int(compute_payload.get("l1_vloopv2_loops")),
                l1_software_loops=_as_int(compute_payload.get("l1_software_loops")),
                l1_vf_fusion_breaks=_as_int(compute_payload.get("l1_vf_fusion_breaks")),
                l1_dual_issue_collapse=_as_float(compute_payload.get("l1_dual_issue_collapse")),
                l2_aic_mac_ratio=_as_float(compute_payload.get("l2_aic_mac_ratio")),
                l2_aiv_vec_ratio=_as_float(compute_payload.get("l2_aiv_vec_ratio")),
                l2_roofline_eta=_as_float(compute_payload.get("l2_roofline_eta")),
                effective_compute_eta=_as_float(raw_eff_eta),
                l0_aiv_subcore_imbalance=_as_bool(
                    compute_payload.get("l0_aiv_subcore_imbalance")
                ),
                l0_ub_scalar_compute_loops=_as_int(
                    compute_payload.get("l0_ub_scalar_compute_loops")
                ),
                l1_scalar_compute_loop_fallback=_as_float(
                    compute_payload.get("l1_scalar_compute_loop_fallback")
                ),
            ),
            offchip_dma=OffChipMemoryDmaAxis(
                l0_gm_roundtrip_bytes=_as_int(dma_payload.get("l0_gm_roundtrip_bytes")),
                l0_dominant_contig_bytes=_as_int(dma_payload.get("l0_dominant_contig_bytes")),
                l0_dma_burst_efficiency=_as_float(
                    dma_payload.get("l0_dma_burst_efficiency"), 1.0
                ),
                l0_mte2_inflight_per_aiv=_as_int(dma_payload.get("l0_mte2_inflight_per_aiv")),
                l0_unaligned_padding_bytes=_as_int(dma_payload.get("l0_unaligned_padding_bytes")),
                l1_dcache_dcci_flushes=_as_int(dma_payload.get("l1_dcache_dcci_flushes")),
                l1_objdump_mte2_bytes=_as_int(dma_payload.get("l1_objdump_mte2_bytes")),
                l2_mte2_wait_ratio=_as_float(dma_payload.get("l2_mte2_wait_ratio")),
                l2_fixpipe_ratio=_as_float(dma_payload.get("l2_fixpipe_ratio")),
                l2_cache_hit_rate=_as_float(dma_payload.get("l2_cache_hit_rate"), 1.0),
                transfer_inflation_ratio=_as_float(raw_inflation),
                l0_min_gm_dma_bytes=_as_int(dma_payload.get("l0_min_gm_dma_bytes")),
                l0_fixpipe_cycles=_as_int(dma_payload.get("l0_fixpipe_cycles")),
                l1_dcache_bypass_narrow_store=_as_float(
                    dma_payload.get("l1_dcache_bypass_narrow_store")
                ),
            ),
            onchip_sram_reg=OnChipSramRegAxis(
                l0_max_overflow_bytes=_as_int(sram_payload.get("l0_max_overflow_bytes")),
                l0_ub_peak_bytes=_as_int(sram_payload.get("l0_ub_peak_bytes")),
                l0_l1_peak_bytes=_as_int(sram_payload.get("l0_l1_peak_bytes")),
                l0_ub_bank_conflict_risk=_as_float(sram_payload.get("l0_ub_bank_conflict_risk")),
                l0_l1_bank_conflict_risk=_as_float(sram_payload.get("l0_l1_bank_conflict_risk")),
                l0_degraded_single_buffers=_as_int(
                    sram_payload.get("l0_degraded_single_buffers")
                ),
                l1_vec_regs=_as_int(sram_payload.get("l1_vec_regs")),
                l1_scalar_regs=_as_int(sram_payload.get("l1_scalar_regs")),
                l1_spill_bytes=_as_int(sram_payload.get("l1_spill_bytes")),
                l1_stack_bytes=_as_int(sram_payload.get("l1_stack_bytes")),
                l2_bank_conflict_ratio=_as_float(sram_payload.get("l2_bank_conflict_ratio")),
            ),
            pipeline_sync=PipelineSyncAxis(
                l0_steady_state_ii=_as_int(sync_payload.get("l0_steady_state_ii")),
                l0_in_loop_pipe_all_barriers=_as_int(
                    sync_payload.get("l0_in_loop_pipe_all_barriers")
                ),
                l0_uses_dynamic_offset_pipe=_as_bool(
                    sync_payload.get("l0_uses_dynamic_offset_pipe")
                ),
                l1_local_membar_in_vf=_as_bool(sync_payload.get("l1_local_membar_in_vf")),
                l1_alias_serialized_buffers=_as_int(
                    sync_payload.get("l1_alias_serialized_buffers")
                ),
                l2_sync_stall_ratio=_as_float(sync_payload.get("l2_sync_stall_ratio")),
                l2_pseudo_double_buffer=_as_bool(sync_payload.get("l2_pseudo_double_buffer")),
            ),
            grid_irregularity=GridIrregularityAxis(
                l0_grid_blocks=_as_int(grid_payload.get("l0_grid_blocks"), 1),
                l0_wave_occupancy=_as_float(grid_payload.get("l0_wave_occupancy"), 1.0),
                l0_has_indirect_gather=_as_bool(grid_payload.get("l0_has_indirect_gather")),
                l0_has_online_rescale=_as_bool(grid_payload.get("l0_has_online_rescale")),
                l0_has_lowbit_quant=_as_bool(grid_payload.get("l0_has_lowbit_quant")),
                l1_subkernel_count=_as_int(grid_payload.get("l1_subkernel_count"), 1),
                l2_measured_latency_us=_as_float(grid_payload.get("l2_measured_latency_us")),
            ),
            has_l2_telemetry=_as_bool(payload.get("has_l2_telemetry")),
            has_l1_compiler_summary=_as_bool(payload.get("has_l1_compiler_summary")),
            measured_l2_fields=_as_str_tuple(payload.get("measured_l2_fields")),
            compute_engine=_normalize_compute_engine(payload.get("compute_engine")),
            num_stages=max(1, _as_int(payload.get("num_stages"), 1)),
            has_l2_pmu=_as_bool(payload.get("has_l2_pmu")),
        )


__all__ = [
    "AxisResidual5D",
    "CompilerSummaryLike",
    "ComputeIssueAxis",
    "GridIrregularityAxis",
    "ObservationVector5D",
    "OffChipMemoryDmaAxis",
    "OnChipSramRegAxis",
    "PhysicalLedgerLike",
    "PipelineSyncAxis",
    "RooflineSummaryLike",
    "ScopeWaterlineLike",
    "compute_transfer_inflation_ratio",
]
