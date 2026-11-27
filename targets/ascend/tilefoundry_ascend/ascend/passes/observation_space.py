"""Five-axis orthogonal MECE observation space spanning L0 ledger, L1 compiler, and L2 telemetry."""

from __future__ import annotations

import inspect
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from typing import Protocol, runtime_checkable

from tilefoundry_ascend.ascend.json_record import JsonRecordMixin
from tilefoundry_ascend.ascend.sil.schema import ConstraintRelaxationMask

_L2_FIELD_SOURCES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("l2_measured_latency_us", ("measured_latency_us", "dur_us")),
    ("l2_aic_mac_ratio", ("aic_mac_ratio", "aic_mad")),
    ("l2_aiv_vec_ratio", ("aiv_vec_ratio", "aiv_vec")),
    ("l2_roofline_eta", ("roofline_eta",)),
    ("l2_mte2_wait_ratio", ("mte2_wait_ratio", "aiv_mte2", "aic_mte2")),
    ("l2_fixpipe_ratio", ("aic_fixpipe",)),
    ("l2_cache_hit_rate", ("l2_hit_rate", "l2_cache_hit_rate")),
    ("l2_bank_conflict_ratio", ("bank_conflict_ratio", "ub_bank_conflict_ratio", "resource_conflict_ratio")),
    ("l2_sync_stall_ratio", ("sync_stall_ratio",)),
    ("l2_pseudo_double_buffer", ("pseudo_double_buffer_serialized",)),
)

_PMU_L2_FIELDS: frozenset[str] = frozenset(
    name for name, _ in _L2_FIELD_SOURCES if name not in ("l2_measured_latency_us", "l2_roofline_eta")
)

_VALID_COMPUTE_ENGINES: frozenset[str] = frozenset({"aiv_only", "aic_only", "coupled_1c2v"})


def _normalize_compute_engine(raw: object) -> str:
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
    sources = (
        *(
            src.get("compute_engine") if isinstance(src, Mapping) else getattr(src, "compute_engine", "")
            for src in (topology_axes, telemetry.get("topology_axes"))
            if src is not None
        ),
        telemetry.get("compute_engine"),
        ledger.microarch_profile.get("compute_engine"),
        getattr(ledger, "compute_engine", ""),
    )
    for raw in sources:
        if candidate := _normalize_compute_engine(raw):
            return candidate
    has_cube = cube_cycles > 0 or l2_aic_mac > 0.0
    has_vec = vec_cycles > 0 or l2_aiv_vec > 0.0
    if has_cube and has_vec:
        return "coupled_1c2v"
    return "aic_only" if has_cube else ("aiv_only" if has_vec else "")


def _resolve_num_stages(telemetry: Mapping[str, object], ledger: PhysicalLedgerLike) -> int:
    for raw in (
        telemetry.get("num_stages"),
        telemetry.get("pipeline_depth"),
        ledger.microarch_profile.get("num_stages"),
        ledger.microarch_profile.get("pipeline_depth"),
        getattr(ledger, "num_stages", None),
        getattr(ledger, "pipeline_depth", None),
    ):
        if (val := _as_int(raw, 0)) > 0:
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
    if isinstance(value, bool):
        return default
    return int(value) if isinstance(value, (int, float)) else default


def _as_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, bool):
        return default
    return float(value) if isinstance(value, (int, float)) else default


def _as_bool(value: object, default: bool = False) -> bool:
    return value if isinstance(value, bool) else default


def _as_str_tuple(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    return tuple(item for item in value if isinstance(item, str))


def _require_mapping(value: object, field_name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"observation field {field_name!r} must be a mapping")
    return value


def _is_finite_num(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _telemetry_float(telemetry: Mapping[str, object], *keys: str, default: float = 0.0) -> float:
    for key in keys:
        if key in telemetry and _is_finite_num(val := telemetry[key]):
            return float(val)
    return default


def _telemetry_bool(telemetry: Mapping[str, object], key: str) -> bool:
    return val if isinstance(val := telemetry.get(key), bool) else False


def _has_measured_l2_field(
    telemetry: Mapping[str, object], field_name: str, keys: tuple[str, ...]
) -> bool:
    if field_name == "l2_pseudo_double_buffer":
        return any(isinstance(telemetry.get(k), bool) for k in keys)
    return any(k in telemetry and _is_finite_num(telemetry[k]) for k in keys)


def _resolve_mte2_wait_ratio(telemetry: Mapping[str, object]) -> float:
    if _has_measured_l2_field(telemetry, "l2_mte2_wait_ratio", ("mte2_wait_ratio",)):
        return _telemetry_float(telemetry, "mte2_wait_ratio")
    cands = [
        _telemetry_float(telemetry, k)
        for k in ("aiv_mte2", "aic_mte2")
        if _has_measured_l2_field(telemetry, "l2_mte2_wait_ratio", (k,))
    ]
    return max(cands) if cands else 0.0


def _resolve_static_ledger_gm_bytes(ledger: PhysicalLedgerLike) -> int:
    grid_count = (
        ledger.active_grid_blocks
        if isinstance(ledger, _ActiveGridLedgerLike) and ledger.active_grid_blocks > 0
        else ledger.grid_blocks
    )
    effective_blocks = max(1, int(grid_count))
    if isinstance(ledger, _FullGridTrafficLedgerLike):
        total_fn = ledger.total_gm_bytes
        if callable(total_fn):
            positional = [
                p
                for p in inspect.signature(total_fn).parameters.values()
                if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            ]
            val = total_fn(effective_blocks) if positional and positional[0].default is inspect.Parameter.empty else total_fn()
            return max(0, int(val))
        if isinstance(total_fn, int) and not isinstance(total_fn, bool):
            return max(0, total_fn)
    scope = ledger.gm_traffic_scope if isinstance(ledger, _ScopedTrafficLedgerLike) else "per_tile"
    raw_bytes = max(0, int(ledger.gm_roundtrip_bytes))
    return raw_bytes if scope == "full_grid" else raw_bytes * effective_blocks


def _resolve_roofline_bounds(
    roofline: RooflineSummaryLike | Mapping[str, object] | None,
    telemetry: Mapping[str, object],
) -> tuple[float, float]:
    min_gm, floor_us = 0.0, 0.0
    if isinstance(roofline, Mapping):
        min_gm = _as_float(roofline.get("min_gm_bytes"))
        floor_us = _as_float(roofline.get("compute_floor_us"))
    elif roofline is not None:
        min_gm = float(roofline.min_gm_bytes)
        floor_us = float(roofline.compute_floor_us)
    if min_gm <= 0.0:
        min_gm = _telemetry_float(telemetry, "blueprint_min_gm_bytes")
    if floor_us <= 0.0:
        floor_us = _telemetry_float(telemetry, "blueprint_compute_floor_us")
    return max(0.0, min_gm), max(0.0, floor_us)


def _resolve_measured_hbm_bytes(telemetry: Mapping[str, object]) -> float:
    if (explicit := _telemetry_float(telemetry, "actual_gm_bytes")) > 0.0:
        return explicit
    total = _telemetry_float(telemetry, "hbm_read_bytes") + _telemetry_float(telemetry, "hbm_write_bytes")
    return total if total > 0.0 else 0.0


def compute_transfer_inflation_ratio(
    ledger: PhysicalLedgerLike,
    *,
    telemetry: Mapping[str, object] | None = None,
    roofline: RooflineSummaryLike | Mapping[str, object] | None = None,
) -> float:
    """Compute HBM transfer inflation from measured HBM traffic or static roundtrip bytes."""
    tel = telemetry if telemetry is not None else {}
    min_gm, _ = _resolve_roofline_bounds(roofline, tel)
    if min_gm <= 0.0:
        return max(0.0, _telemetry_float(tel, "transfer_inflation_ratio"))
    if (measured := _resolve_measured_hbm_bytes(tel)) > 0.0:
        return measured / max(1.0, float(min_gm))
    return (float(min_gm) + float(_resolve_static_ledger_gm_bytes(ledger))) / max(1.0, float(min_gm))


@dataclass(frozen=True)
class ComputeIssueAxis(JsonRecordMixin):
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
class OffChipMemoryDmaAxis(JsonRecordMixin):
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
class OnChipSramRegAxis(JsonRecordMixin):
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
class PipelineSyncAxis(JsonRecordMixin):
    """Axis 4: Modulo initiation interval, multi-buffer overlap, and cross-pipe barriers."""

    l0_steady_state_ii: int = 0
    l0_in_loop_pipe_all_barriers: int = 0
    l0_uses_dynamic_offset_pipe: bool = False
    l1_local_membar_in_vf: bool = False
    l1_alias_serialized_buffers: int = 0
    l2_sync_stall_ratio: float = 0.0
    l2_pseudo_double_buffer: bool = False


@dataclass(frozen=True)
class GridIrregularityAxis(JsonRecordMixin):
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
        rules = (
            (self.compute_issue_gap >= 0.45, "alg_compute_and_monoid"),
            (self.onchip_sram_reg_gap >= 0.45 or self.grid_wave_gap >= 0.40, "tile_and_stage_knobs"),
            (self.compute_issue_gap >= 0.25, "exec_mode_and_spr"),
            (self.offchip_dma_gap >= 0.25, "dma_and_cache_hierarchy"),
            (self.onchip_sram_reg_gap >= 0.20, "sram_layout_and_aliasing"),
            (self.pipeline_sync_gap >= 0.20, "pipeline_and_sync_graph"),
            (self.grid_wave_gap >= 0.25, "grid_and_workspace_topology"),
        )
        to_unlock = [facet for cond, facet in rules if cond]
        return current.unlock(*to_unlock) if to_unlock else current


def _coerce_axis(
    axis_cls: type,
    payload: Mapping[str, object],
    overrides: Mapping[str, object] | None = None,
):
    kwargs: dict[str, object] = {}
    for f in fields(axis_cls):
        default = f.default
        raw = overrides[f.name] if overrides and f.name in overrides else payload.get(f.name)
        if isinstance(default, bool):
            kwargs[f.name] = _as_bool(raw, default)
        elif isinstance(default, int):
            kwargs[f.name] = _as_int(raw, default)
        else:
            kwargs[f.name] = _as_float(raw, float(default))
    return axis_cls(**kwargs)


@dataclass(frozen=True)
class ObservationVector5D(JsonRecordMixin):
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
        pc = ledger.pipe_cycles
        cube_cycles = _as_int(pc.get("AIC.PIPE_M")) + _as_int(pc.get("AIC.CUBE"))
        aiv0_v, aiv1_v = _as_int(pc.get("AIV0.PIPE_V")), _as_int(pc.get("AIV1.PIPE_V"))
        vec_cycles = aiv0_v + aiv1_v
        fixpipe_cycles = _as_int(pc.get("AIC.PIPE_FIX") if "AIC.PIPE_FIX" in pc else pc.get("PIPE_FIX"))
        ub_wl, l1_wl = ledger.waterlines.get("UB"), ledger.waterlines.get("L1")
        degraded_count = sum(len(wl.degraded_single_buffers) for wl in ledger.waterlines.values())

        c = compiler_summary
        raw_extra = getattr(c, "extra_signals", None) if c is not None else None
        extra = raw_extra if isinstance(raw_extra, Mapping) else {}
        ub_loops = max(0, _as_int(getattr(ledger, "ub_scalar_compute_loops", 0)))
        min_dma = max(0, _as_int(getattr(ledger, "min_gm_dma_bytes", 0)))
        subkernel_count = max(1, _as_int(ledger.microarch_profile.get("subkernel_count"), 1))
        mte2_bytes_hint = _as_int(ledger.microarch_profile.get("mte2_bytes_hint"), 0)

        measured_latency_us = _telemetry_float(tel, "measured_latency_us", "dur_us")
        measured_fields = tuple(
            fn for fn, keys in _L2_FIELD_SOURCES if _has_measured_l2_field(tel, fn, keys)
        )
        has_l2_telemetry = bool(measured_fields) and not _telemetry_bool(tel, "is_analytical_proxy")
        measured_l2_fields = measured_fields if has_l2_telemetry else ()

        _min_gm, blueprint_floor_us = _resolve_roofline_bounds(roofline, tel)
        transfer_inflation_ratio = compute_transfer_inflation_ratio(
            ledger, telemetry=tel, roofline=roofline
        )
        has_case_idx = _is_finite_num(tel.get("anchor_case_idx"))
        case_t_hw_us = _telemetry_float(tel, "case_t_hw_us")
        case_roofline_eta = _telemetry_float(tel, "roofline_eta")
        if has_case_idx and case_t_hw_us > 0.0 and measured_latency_us > 0.0:
            raw_eta = case_t_hw_us / max(1e-3, measured_latency_us)
        elif has_case_idx and case_roofline_eta > 0.0:
            raw_eta = case_roofline_eta
        elif blueprint_floor_us > 0.0 and measured_latency_us > 0.0:
            raw_eta = blueprint_floor_us / max(1e-3, measured_latency_us)
        elif case_t_hw_us > 0.0 and measured_latency_us > 0.0:
            raw_eta = case_t_hw_us / max(1e-3, measured_latency_us)
        else:
            raw_eta = _telemetry_float(tel, "effective_compute_eta")
        effective_compute_eta = min(1.0, max(0.0, raw_eta))

        scalar_loop_fallback = max(
            float(ub_loops),
            _as_float(extra.get("scalar_compute_loop_fallback")),
            _telemetry_float(tel, "scalar_compute_loop_fallback"),
        )
        narrow_store_bypass = max(
            _as_float(extra.get("dcache_bypass_narrow_store")),
            _telemetry_float(tel, "dcache_bypass_narrow_store"),
            (32.0 / float(min_dma)) if 0 < min_dma < 32 else 0.0,
        )
        l2_aic_mac = _telemetry_float(tel, "aic_mac_ratio", "aic_mad")
        l2_aiv_vec = _telemetry_float(tel, "aiv_vec_ratio", "aiv_vec")

        return cls(
            compute_issue=ComputeIssueAxis(
                l0_cube_cycles=cube_cycles,
                l0_vec_cycles=vec_cycles,
                l0_cube_to_vec_ratio=float(cube_cycles) / float(max(1, vec_cycles)),
                l1_vloopv2_loops=c.vloopv2_loops if c else 0,
                l1_software_loops=c.software_loops if c else 0,
                l1_vf_fusion_breaks=c.vf_fusion_breaks if c else 0,
                l1_dual_issue_collapse=c.dual_issue_ipc_collapse if c else 0.0,
                l2_aic_mac_ratio=l2_aic_mac,
                l2_aiv_vec_ratio=l2_aiv_vec,
                l2_roofline_eta=case_roofline_eta,
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
                l1_dcache_dcci_flushes=c.dcache_dcci_flushes if c else 0,
                l1_objdump_mte2_bytes=mte2_bytes_hint,
                l2_mte2_wait_ratio=_resolve_mte2_wait_ratio(tel),
                l2_fixpipe_ratio=_telemetry_float(tel, "aic_fixpipe"),
                l2_cache_hit_rate=_telemetry_float(tel, "l2_hit_rate", "l2_cache_hit_rate", default=1.0),
                transfer_inflation_ratio=transfer_inflation_ratio,
                l0_min_gm_dma_bytes=min_dma,
                l0_fixpipe_cycles=fixpipe_cycles,
                l1_dcache_bypass_narrow_store=narrow_store_bypass,
            ),
            onchip_sram_reg=OnChipSramRegAxis(
                l0_max_overflow_bytes=ledger.max_overflow_bytes,
                l0_ub_peak_bytes=ub_wl.peak_allocated_bytes if ub_wl is not None else 0,
                l0_l1_peak_bytes=l1_wl.peak_allocated_bytes if l1_wl is not None else 0,
                l0_ub_bank_conflict_risk=ledger.ub_bank_conflict_risk,
                l0_l1_bank_conflict_risk=ledger.l1_bank_conflict_risk,
                l0_degraded_single_buffers=degraded_count,
                l1_vec_regs=c.vec_regs if c else 0,
                l1_scalar_regs=c.scalar_regs if c else 0,
                l1_spill_bytes=c.spill_bytes if c else 0,
                l1_stack_bytes=c.stack_bytes if c else 0,
                l2_bank_conflict_ratio=_telemetry_float(
                    tel, "bank_conflict_ratio", "ub_bank_conflict_ratio", "resource_conflict_ratio"
                ),
            ),
            pipeline_sync=PipelineSyncAxis(
                l0_steady_state_ii=int(ledger.steady_state_ii or 0),
                l0_in_loop_pipe_all_barriers=ledger.in_loop_pipe_all_barriers,
                l0_uses_dynamic_offset_pipe=ledger.uses_dynamic_offset_pipelining,
                l1_local_membar_in_vf=c.has_local_membar_in_vf_call if c else False,
                l1_alias_serialized_buffers=c.alias_serialized_buffers if c else 0,
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
            has_l1_compiler_summary=c is not None,
            measured_l2_fields=measured_l2_fields,
            compute_engine=_resolve_compute_engine(
                topology_axes=topology_axes,
                telemetry=tel,
                ledger=ledger,
                cube_cycles=cube_cycles,
                vec_cycles=vec_cycles,
                l2_aic_mac=l2_aic_mac,
                l2_aiv_vec=l2_aiv_vec,
            ),
            num_stages=_resolve_num_stages(tel, ledger),
            has_l2_pmu=bool(has_l2_telemetry) and any(f in _PMU_L2_FIELDS for f in measured_l2_fields),
        )

    def compute_axis_residuals(self) -> AxisResidual5D:
        """Compute normalized [0, 1] co-axial static-dynamic residuals across all 5 axes."""
        ci, dma, sram, sync, grid = (
            self.compute_issue,
            self.offchip_dma,
            self.onchip_sram_reg,
            self.pipeline_sync,
            self.grid_irregularity,
        )
        sw_penalty = 0.5 if (ci.l1_software_loops > 0 and ci.l1_vloopv2_loops == 0) else 0.0
        util = max(ci.l2_aic_mac_ratio, ci.l2_aiv_vec_ratio)
        eff_eta = ci.l2_roofline_eta if ci.l2_roofline_eta > 0.0 else ci.effective_compute_eta
        roofline_gap = (1.0 - eff_eta) if (util >= 0.5 and 0.0 < eff_eta < 0.5) else 0.0
        scalar_trap = (ci.l0_cube_cycles == 0 and ci.l0_vec_cycles == 0) or (
            self.has_l1_compiler_summary
            and sram.l1_vec_regs == 0
            and ci.l2_aiv_vec_ratio < 0.02
            and ci.l2_aic_mac_ratio == 0.0
        )
        scalar_loop = min(
            1.0,
            0.5 * max(0.0, float(ci.l0_ub_scalar_compute_loops), ci.l1_scalar_compute_loop_fallback),
        )
        compute_gap = min(
            1.0,
            max(
                1.0 if scalar_trap else 0.0,
                sw_penalty,
                scalar_loop,
                ci.l1_dual_issue_collapse,
                0.25 * min(4, ci.l1_vf_fusion_breaks),
                roofline_gap,
            ),
        )
        inflation_gap = (
            min(1.0, max(0.0, dma.transfer_inflation_ratio - 1.0))
            if dma.transfer_inflation_ratio > 1.30
            else 0.0
        )
        dma_gap = min(
            1.0,
            max(
                dma.l2_mte2_wait_ratio,
                1.0 - dma.l0_dma_burst_efficiency,
                1.0 - dma.l2_cache_hit_rate,
                0.35 if dma.l1_dcache_dcci_flushes > 0 else 0.0,
                inflation_gap,
                min(1.0, max(0.0, dma.l1_dcache_bypass_narrow_store) / 4.0),
            ),
        )
        sram_gap = min(
            1.0,
            max(
                1.0 if sram.l0_max_overflow_bytes > 0 else 0.0,
                min(1.0, float(sram.l1_spill_bytes) / 256.0),
                min(1.0, max(0.0, float(sram.l1_stack_bytes)) / 1024.0),
                sram.l2_bank_conflict_ratio,
                sram.l0_ub_bank_conflict_risk,
            ),
        )
        sync_gap = min(
            1.0,
            max(
                sync.l2_sync_stall_ratio,
                0.6 if sync.l1_local_membar_in_vf else 0.0,
                0.5 if sync.l2_pseudo_double_buffer else 0.0,
                0.25 * min(4, sync.l1_alias_serialized_buffers),
            ),
        )
        occ = grid.l0_wave_occupancy
        grid_gap = min(1.0, max(0.0, (1.0 - occ / math.ceil(occ)) if occ > 0.0 else 0.0))
        return AxisResidual5D(
            compute_issue_gap=round(compute_gap, 4),
            offchip_dma_gap=round(dma_gap, 4),
            onchip_sram_reg_gap=round(sram_gap, 4),
            pipeline_sync_gap=round(sync_gap, 4),
            grid_wave_gap=round(grid_gap, 4),
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize all 5 observation axes, orthogonal ratios, residuals, and provenance."""
        payload = super().to_dict()
        payload["transfer_inflation_ratio"] = self.transfer_inflation_ratio
        payload["effective_compute_eta"] = self.effective_compute_eta
        payload["residuals"] = asdict(self.compute_axis_residuals())
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ObservationVector5D:
        """Reconstruct the 5-axis observation vector from its serialized dictionary form."""
        compute_p = _require_mapping(payload.get("compute_issue"), "compute_issue")
        dma_p = _require_mapping(payload.get("offchip_dma"), "offchip_dma")
        sram_p = _require_mapping(payload.get("onchip_sram_reg"), "onchip_sram_reg")
        sync_p = _require_mapping(payload.get("pipeline_sync"), "pipeline_sync")
        grid_p = _require_mapping(payload.get("grid_irregularity"), "grid_irregularity")

        eff_eta = compute_p.get("effective_compute_eta")
        if eff_eta is None:
            eff_eta = payload.get("effective_compute_eta")
        inflation = dma_p.get("transfer_inflation_ratio")
        if inflation is None:
            inflation = payload.get("transfer_inflation_ratio")

        return cls(
            compute_issue=_coerce_axis(ComputeIssueAxis, compute_p, {"effective_compute_eta": eff_eta}),
            offchip_dma=_coerce_axis(OffChipMemoryDmaAxis, dma_p, {"transfer_inflation_ratio": inflation}),
            onchip_sram_reg=_coerce_axis(OnChipSramRegAxis, sram_p),
            pipeline_sync=_coerce_axis(PipelineSyncAxis, sync_p),
            grid_irregularity=_coerce_axis(GridIrregularityAxis, grid_p),
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
