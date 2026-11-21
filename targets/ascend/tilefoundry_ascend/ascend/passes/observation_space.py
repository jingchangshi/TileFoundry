"""Five-axis orthogonal MECE observation space spanning L0 ledger, L1 compiler, and L2 telemetry."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Protocol

from tilefoundry_ascend.ascend.sil.schema import ConstraintRelaxationMask

_L2_FIELD_SOURCES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("l2_measured_latency_us", ("measured_latency_us", "dur_us")),
    ("l2_aic_mac_ratio", ("aic_mac_ratio", "aic_mad")),
    ("l2_aiv_vec_ratio", ("aiv_vec_ratio", "aiv_vec")),
    ("l2_roofline_eta", ("roofline_eta",)),
    ("l2_mte2_wait_ratio", ("mte2_wait_ratio", "aiv_mte2")),
    ("l2_fixpipe_ratio", ("aic_fixpipe",)),
    ("l2_cache_hit_rate", ("l2_hit_rate",)),
    ("l2_bank_conflict_ratio", ("bank_conflict_ratio",)),
    ("l2_sync_stall_ratio", ("sync_stall_ratio",)),
    ("l2_pseudo_double_buffer", ("pseudo_double_buffer_serialized",)),
)

_L2_TELEMETRY_KEYS: tuple[str, ...] = tuple(
    dict.fromkeys(key for _field, keys in _L2_FIELD_SOURCES for key in keys)
)


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
    """Read the first present numeric telemetry key, otherwise return the neutral default."""
    for key in keys:
        if key not in telemetry:
            continue
        value = telemetry[key]
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            return float(value)
    return default


def _telemetry_bool(telemetry: Mapping[str, object], key: str, default: bool = False) -> bool:
    """Read a boolean L2 telemetry flag, otherwise return the neutral default."""
    value = telemetry.get(key)
    return value if isinstance(value, bool) else default


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

    @classmethod
    def from_tiers(
        cls,
        ledger: PhysicalLedgerLike,
        *,
        compiler_summary: CompilerSummaryLike | None = None,
        telemetry: Mapping[str, object] | None = None,
    ) -> ObservationVector5D:
        """Assemble the 5-axis vector directly from L0 ledger, L1 summary, and L2 telemetry."""
        tel = telemetry if telemetry is not None else {}
        cube_cycles = _as_int(ledger.pipe_cycles.get("AIC.PIPE_M")) + _as_int(
            ledger.pipe_cycles.get("AIC.CUBE")
        )
        vec_cycles = _as_int(ledger.pipe_cycles.get("AIV0.PIPE_V")) + _as_int(
            ledger.pipe_cycles.get("AIV1.PIPE_V")
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

        subkernel_count = max(1, _as_int(ledger.microarch_profile.get("subkernel_count"), 1))
        mte2_bytes_hint = _as_int(ledger.microarch_profile.get("mte2_bytes_hint"), 0)
        measured_latency_us = _telemetry_float(tel, "measured_latency_us", "dur_us")
        has_l2_telemetry = any(key in tel for key in _L2_TELEMETRY_KEYS) and not _telemetry_bool(
            tel, "is_analytical_proxy"
        )
        measured_l2_fields = (
            tuple(field for field, keys in _L2_FIELD_SOURCES if any(key in tel for key in keys))
            if has_l2_telemetry
            else ()
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
                l2_aic_mac_ratio=_telemetry_float(tel, "aic_mac_ratio", "aic_mad"),
                l2_aiv_vec_ratio=_telemetry_float(tel, "aiv_vec_ratio", "aiv_vec"),
                l2_roofline_eta=_telemetry_float(tel, "roofline_eta"),
            ),
            offchip_dma=OffChipMemoryDmaAxis(
                l0_gm_roundtrip_bytes=ledger.gm_roundtrip_bytes,
                l0_dominant_contig_bytes=ledger.dominant_contig_bytes,
                l0_dma_burst_efficiency=ledger.dma_burst_efficiency_ratio,
                l0_mte2_inflight_per_aiv=ledger.mte2_inflight_per_aiv,
                l0_unaligned_padding_bytes=ledger.unaligned_dma_padding_bytes,
                l1_dcache_dcci_flushes=dcci_flushes,
                l1_objdump_mte2_bytes=mte2_bytes_hint,
                l2_mte2_wait_ratio=_telemetry_float(tel, "mte2_wait_ratio", "aiv_mte2"),
                l2_fixpipe_ratio=_telemetry_float(tel, "aic_fixpipe"),
                l2_cache_hit_rate=_telemetry_float(tel, "l2_hit_rate", default=1.0),
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
                l2_bank_conflict_ratio=_telemetry_float(tel, "bank_conflict_ratio"),
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
        )

    def compute_axis_residuals(self) -> AxisResidual5D:
        """Compute normalized [0, 1] co-axial static-dynamic residuals across all 5 axes."""
        has_sw_loop = (
            self.compute_issue.l1_software_loops > 0
            and self.compute_issue.l1_vloopv2_loops == 0
        )
        sw_loop_penalty = 0.5 if has_sw_loop else 0.0
        compute_gap = min(
            1.0,
            max(
                sw_loop_penalty,
                self.compute_issue.l1_dual_issue_collapse,
                0.25 * min(4, self.compute_issue.l1_vf_fusion_breaks),
            ),
        )
        dma_gap = min(
            1.0,
            max(
                self.offchip_dma.l2_mte2_wait_ratio,
                1.0 - self.offchip_dma.l0_dma_burst_efficiency,
                1.0 - self.offchip_dma.l2_cache_hit_rate,
                0.35 if self.offchip_dma.l1_dcache_dcci_flushes > 0 else 0.0,
            ),
        )
        sram_gap = min(
            1.0,
            max(
                1.0 if self.onchip_sram_reg.l0_max_overflow_bytes > 0 else 0.0,
                min(1.0, float(self.onchip_sram_reg.l1_spill_bytes) / 256.0),
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
        wave_waste = (1.0 - wave_occupancy) if 0.0 < wave_occupancy < 1.0 else 0.0
        grid_gap = min(1.0, max(0.0, wave_waste))
        return AxisResidual5D(
            compute_issue_gap=round(compute_gap, 4),
            offchip_dma_gap=round(dma_gap, 4),
            onchip_sram_reg_gap=round(sram_gap, 4),
            pipeline_sync_gap=round(sync_gap, 4),
            grid_wave_gap=round(grid_gap, 4),
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize all 5 observation axes, residuals, and the L2 telemetry provenance flag."""
        return {
            "compute_issue": asdict(self.compute_issue),
            "offchip_dma": asdict(self.offchip_dma),
            "onchip_sram_reg": asdict(self.onchip_sram_reg),
            "pipeline_sync": asdict(self.pipeline_sync),
            "grid_irregularity": asdict(self.grid_irregularity),
            "has_l2_telemetry": self.has_l2_telemetry,
            "has_l1_compiler_summary": self.has_l1_compiler_summary,
            "measured_l2_fields": list(self.measured_l2_fields),
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
    "ScopeWaterlineLike",
]
