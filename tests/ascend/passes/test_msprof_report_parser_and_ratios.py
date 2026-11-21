"""Tests for multi-subkernel msprof parsing, P2 trace analysis, and orthogonal physical ratios."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

from tests.ascend.agent.test_surrogate_batch_npu import _extract_msprof_metrics
from tilefoundry_ascend.agent.orchestrator.residual_beam import (
    StaticDynamicResidual,
    compute_static_dynamic_residual,
)
from tilefoundry_ascend.ascend.cost_model.msprof_report_parser import (
    AggregatedMsprofReport,
    P2InstructionDiveSummary,
    SubkernelMsprofMetrics,
    aggregate_subkernel_msprof,
    parse_msprof_op_report,
    parse_p2_instruction_trace,
    parse_pc_start_addr,
)
from tilefoundry_ascend.ascend.passes.hir_tir_bridge import HirRooflineSummary
from tilefoundry_ascend.ascend.passes.observation_space import ObservationVector5D
from tilefoundry_ascend.ascend.passes.physical_ledger import (
    ObjectivePhysicalLedger,
    build_objective_physical_ledger,
)


def _sample_roofline(
    *,
    min_gm_bytes: int = 1_000_000,
    compute_floor_us: float = 15.0,
    memory_floor_us: float = 20.0,
) -> HirRooflineSummary:
    return HirRooflineSummary(
        anchor_dims={"M": 1024, "N": 1024},
        compute_flops=2_000_000_000,
        min_gm_bytes=min_gm_bytes,
        arithmetic_intensity=float(2_000_000_000) / float(max(1, min_gm_bytes)),
        compute_floor_us=compute_floor_us,
        memory_floor_us=memory_floor_us,
        ideal_roofline_us=max(compute_floor_us, memory_floor_us),
        dominant_bound="memory" if memory_floor_us >= compute_floor_us else "compute",
    )


def _sample_ledger(
    *,
    gm_roundtrip_bytes: int = 0,
    grid_blocks: int = 16,
    active_grid_blocks: int | None = None,
) -> ObjectivePhysicalLedger:
    return build_objective_physical_ledger(
        analysis_mode="exact_modulo_z3",
        is_legal=True,
        steady_state_ii=2,
        pipe_cycles={"AIV0.PIPE_V": 200, "AIV1.PIPE_V": 200},
        waterlines={},
        grid_blocks=grid_blocks,
        active_grid_blocks=active_grid_blocks,
        gm_roundtrip_bytes=gm_roundtrip_bytes,
    )


@dataclass(frozen=True)
class _FullGridMethodLedger:
    """Stub ledger exposing callable total_gm_bytes() with no required arguments."""

    pipe_cycles: dict[str, int]
    waterlines: dict[str, object]
    gm_roundtrip_bytes: int
    dominant_contig_bytes: int = 64
    dma_burst_efficiency_ratio: float = 1.0
    mte2_inflight_per_aiv: int = 2
    unaligned_dma_padding_bytes: int = 0
    max_overflow_bytes: int = 0
    ub_bank_conflict_risk: float = 0.0
    l1_bank_conflict_risk: float = 0.0
    steady_state_ii: int | None = 2
    in_loop_pipe_all_barriers: int = 0
    uses_dynamic_offset_pipelining: bool = False
    grid_blocks: int = 8
    wave_occupancy: float = 1.0
    has_indirect_gather: bool = False
    has_online_rescale_loop: bool = False
    has_lowbit_quant: bool = False
    microarch_profile: dict[str, object] = field(default_factory=dict)
    gm_traffic_scope: str = "full_grid"
    explicit_total_bytes: int = 1_500_000

    def total_gm_bytes(self) -> int:
        return self.explicit_total_bytes


@dataclass(frozen=True)
class _ScopedFullGridLedger:
    """Stub ledger declaring full_grid scope without a total_gm_bytes() method."""

    pipe_cycles: dict[str, int]
    waterlines: dict[str, object]
    gm_roundtrip_bytes: int
    dominant_contig_bytes: int = 64
    dma_burst_efficiency_ratio: float = 1.0
    mte2_inflight_per_aiv: int = 2
    unaligned_dma_padding_bytes: int = 0
    max_overflow_bytes: int = 0
    ub_bank_conflict_risk: float = 0.0
    l1_bank_conflict_risk: float = 0.0
    steady_state_ii: int | None = 2
    in_loop_pipe_all_barriers: int = 0
    uses_dynamic_offset_pipelining: bool = False
    grid_blocks: int = 16
    wave_occupancy: float = 1.0
    has_indirect_gather: bool = False
    has_online_rescale_loop: bool = False
    has_lowbit_quant: bool = False
    microarch_profile: dict[str, object] = field(default_factory=dict)
    gm_traffic_scope: str = "full_grid"


def test_parse_msprof_multi_subkernel_split_kv_and_combine_duration_weighted(
    tmp_path: Path,
) -> None:
    """Multi-subkernel CSV reports compute exact duration-weighted ratios and sum HBM bytes."""
    dump_dir = tmp_path / "OPPROF_nested" / "dump"
    dump_dir.mkdir(parents=True)

    (dump_dir / "OpBasicInfo.csv").write_text(
        "Op Name,Task Duration(us),Block Dim\n"
        "split_kv_main,80.0,56\n"
        "split_kv_combine,20.0,16\n",
        encoding="utf-8",
    )
    (dump_dir / "PipeUtilization.csv").write_text(
        "Op Name,aic_mac_ratio(%),aiv_vec_ratio(%),aiv_mte2_ratio(%),"
        "aic_fixpipe_ratio(%),aiv0_vec_ratio,aiv1_vec_ratio\n"
        "split_kv_combine,0.0,85.0,55.0,0.0,0.90,0.80\n"
        "split_kv_main,75.0,40.0,30.0,25.0,0.45,0.35\n",
        encoding="utf-8",
    )
    (dump_dir / "Memory.csv").write_text(
        "Op Name,HBM Read(KB),HBM Write(KB)\n"
        "split_kv_main,2000,500\n"
        "split_kv_combine,400,100\n",
        encoding="utf-8",
    )
    (dump_dir / "MemoryL0.csv").write_text(
        "Op Name,l0a_datasize(KB),l0b_datasize(KB),l0c_datasize(KB)\n"
        "split_kv_main,256,256,128\n"
        "split_kv_combine,N/A,N/A,N/A\n",
        encoding="utf-8",
    )
    (dump_dir / "L2Cache.csv").write_text(
        "Op Name,Hit Rate(%)\n"
        "split_kv_main,80.0\n"
        "split_kv_combine,50.0\n",
        encoding="utf-8",
    )

    report = parse_msprof_op_report(tmp_path)
    assert isinstance(report, AggregatedMsprofReport)
    assert len(report.subkernels) == 2
    assert report.total_duration_us == pytest.approx(100.0)
    assert report.dominant_subkernel_name == "split_kv_main"
    assert report.dominant_subkernel_share == pytest.approx(0.80)

    assert report.aic_mac_ratio == pytest.approx((80.0 * 0.75 + 20.0 * 0.0) / 100.0)
    assert report.aiv_vec_ratio == pytest.approx((80.0 * 0.40 + 20.0 * 0.85) / 100.0)
    assert report.mte2_wait_ratio == pytest.approx((80.0 * 0.30 + 20.0 * 0.55) / 100.0)
    assert report.aic_fixpipe == pytest.approx((80.0 * 0.25 + 20.0 * 0.0) / 100.0)
    assert report.l2_hit_rate == pytest.approx((80.0 * 0.80 + 20.0 * 0.50) / 100.0)
    assert report.aiv0_vec_ratio == pytest.approx((80.0 * 0.45 + 20.0 * 0.90) / 100.0)
    assert report.aiv1_vec_ratio == pytest.approx((80.0 * 0.35 + 20.0 * 0.80) / 100.0)

    assert report.hbm_read_bytes == 2400 * 1024
    assert report.hbm_write_bytes == 600 * 1024
    assert report.actual_gm_bytes == 3000 * 1024
    assert report.l0_transfer_bytes == 640 * 1024

    assert report.sync_stall_ratio is None
    assert report.bank_conflict_ratio is None

    telemetry = report.to_telemetry_dict()
    assert "sync_stall_ratio" not in telemetry
    assert "bank_conflict_ratio" not in telemetry
    assert telemetry["actual_gm_bytes"] == 3000 * 1024
    assert telemetry["dominant_subkernel_name"] == "split_kv_main"
    assert telemetry["subkernel_count"] == 2

    obs = ObservationVector5D.from_tiers(_sample_ledger(), telemetry=telemetry)
    assert "l2_sync_stall_ratio" not in obs.measured_l2_fields
    assert "l2_bank_conflict_ratio" not in obs.measured_l2_fields
    assert "l2_aic_mac_ratio" in obs.measured_l2_fields
    assert "l2_cache_hit_rate" in obs.measured_l2_fields


def test_parse_msprof_per_subkernel_subdirs_and_tall_tables(tmp_path: Path) -> None:
    """Separate per-subkernel directories and tall CSV formats aggregate cleanly."""
    sub0 = tmp_path / "op_0"
    sub1 = tmp_path / "op_1"
    sub0.mkdir()
    sub1.mkdir()

    (sub0 / "OpBasicInfo.csv").write_text(
        "Item,Value\n"
        "Op Name,stage_main\n"
        "Task Duration(us),60.0\n",
        encoding="utf-8",
    )
    (sub0 / "PipeUtilization.csv").write_text(
        "Pipe,Ratio(%)\n"
        "aic_mac_ratio,50%\n"
        "aiv_vec_ratio,60%\n"
        "mte2_wait_ratio,70%\n",
        encoding="utf-8",
    )
    (sub0 / "ResourceConflictRatio.csv").write_text(
        "Metric,Ratio\n"
        "vec_bankgroup_cflt_ratio,0.12\n",
        encoding="utf-8",
    )
    (sub0 / "Memory.csv").write_text(
        "Memory,Datasize(B)\n"
        "HBM Read,1000000\n"
        "HBM Write,500000\n",
        encoding="utf-8",
    )

    (sub1 / "OpBasicInfo.csv").write_text(
        "Op Name,Task Duration(ns)\n"
        "stage_reduce,40000\n",
        encoding="utf-8",
    )
    (sub1 / "PipeUtilization.csv").write_text(
        "aic_mac_ratio,aiv_vec_ratio,mte2_wait_ratio\n"
        "0.0,0.85,0.20\n",
        encoding="utf-8",
    )
    (sub1 / "ResourceConflictRatio.csv").write_text(
        "vec_bank_cflt_ratio\n"
        "0.02\n",
        encoding="utf-8",
    )
    (sub1 / "Memory.csv").write_text(
        "hbm_read_bytes,hbm_write_bytes\n"
        "250000,250000\n",
        encoding="utf-8",
    )

    report = parse_msprof_op_report(tmp_path)
    assert report.total_duration_us == pytest.approx(100.0)
    assert report.dominant_subkernel_name == "stage_main"
    assert report.dominant_subkernel_share == pytest.approx(0.60)
    assert report.aic_mac_ratio == pytest.approx(0.30)
    assert report.aiv_vec_ratio == pytest.approx(0.70)
    assert report.mte2_wait_ratio == pytest.approx(0.50)
    assert report.bank_conflict_ratio == pytest.approx(0.08)
    assert report.actual_gm_bytes == 2_000_000
    assert report.sync_stall_ratio is None
    assert report.l2_hit_rate is None
    assert "sync_stall_ratio" not in report.to_telemetry_dict()
    assert "l2_hit_rate" not in report.to_telemetry_dict()


def test_parse_msprof_missing_directory_raises(tmp_path: Path) -> None:
    """Missing msprof directory fails explicitly with FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        parse_msprof_op_report(tmp_path / "does_not_exist")


def test_aggregate_subkernel_msprof_preserves_explicit_sync_stall_only_when_measured() -> None:
    """Explicit sync_stall_ratio is weighted only over subkernels that measured it."""
    subs = (
        SubkernelMsprofMetrics(name="k0", duration_us=75.0, sync_stall_ratio=0.40),
        SubkernelMsprofMetrics(name="k1", duration_us=25.0, sync_stall_ratio=0.10),
    )
    report = aggregate_subkernel_msprof(subs)
    assert report.sync_stall_ratio == pytest.approx(0.325)
    assert report.to_telemetry_dict()["sync_stall_ratio"] == pytest.approx(0.325)


def test_p2_instruction_trace_overlapping_intervals_and_dwarf_attribution(tmp_path: Path) -> None:
    """P2 parser computes exact same-pipe issue intervals without negative bubble subtraction."""
    pc_file = tmp_path / "pc_start_addr.txt"
    pc_file.write_text(
        "PC Start Address\n"
        "0x1000 kernels/tile_op.cpp:42\n"
        "0x1040 kernels/tile_op.cpp:78\n"
        "0x1080 kernels/tile_op.cpp:105\n",
        encoding="utf-8",
    )
    trace_events = [
        {"pipe": "MTE2", "opcode": "copy_gm2ub", "issue_us": 0.0, "dur_us": 6.0, "pc": "0x1004"},
        {"pipe": "MTE2", "opcode": "copy_gm2ub", "issue_us": 1.0, "dur_us": 6.0, "pc": "0x100c"},
        {"pipe": "MTE2", "opcode": "mte_WaitFlag", "issue_us": 3.5, "dur_us": 5.0, "pc": "0x1014"},
        {"pipe": "MTE2", "opcode": "copy_gm2ub", "issue_us": 6.5, "dur_us": 5.0, "pc": "0x101c"},
        {"pipe": "VEC", "opcode": "vadd", "issue_us": 0.5, "dur_us": 4.0, "pc": "0x1044"},
        {"pipe": "VEC", "opcode": "vmul", "issue_us": 2.5, "dur_us": 4.0, "pc": "0x104c"},
        {"pipe": "VEC", "opcode": "mte_WaitFlag", "issue_us": 6.5, "dur_us": 3.0, "pc": "0x1050"},
        {"pipe": "VEC", "opcode": "vexp", "issue_us": 12.5, "dur_us": 4.0, "pc": "0x1088"},
    ]
    trace_file = tmp_path / "sim_trace.json"
    trace_file.write_text(json.dumps({"traceEvents": trace_events}), encoding="utf-8")

    summary = parse_p2_instruction_trace(tmp_path)
    assert isinstance(summary, P2InstructionDiveSummary)
    assert summary.total_instructions == 8
    assert sum(summary.instruction_mix.values()) == pytest.approx(1.0)
    assert summary.instruction_mix["mte2"] == pytest.approx(3.0 / 8.0)
    assert summary.instruction_mix["vector"] == pytest.approx(3.0 / 8.0)
    assert summary.instruction_mix["sync"] == pytest.approx(2.0 / 8.0)
    assert summary.mte_waitflag_count == 2
    assert summary.mte_waitflag_share == pytest.approx(0.25)

    mte2_stats = summary.issue_interval_us["MTE2"]
    assert mte2_stats.instruction_count == 4
    assert mte2_stats.interval_count == 3
    assert mte2_stats.p10_us == pytest.approx(1.3)
    assert mte2_stats.p50_us == pytest.approx(2.5)
    assert mte2_stats.p90_us == pytest.approx(2.9)
    assert mte2_stats.p10_us >= 0.0

    vec_stats = summary.issue_interval_us["VEC"]
    assert vec_stats.instruction_count == 4
    assert vec_stats.interval_count == 3
    assert vec_stats.p10_us == pytest.approx(2.4)
    assert vec_stats.p50_us == pytest.approx(4.0)
    assert vec_stats.p90_us == pytest.approx(5.6)

    assert summary.attributed_instruction_share == pytest.approx(1.0)
    assert len(summary.source_line_hotspots) == 3
    assert summary.source_line_hotspots[0].source_location == "kernels/tile_op.cpp:42"
    assert summary.source_line_hotspots[0].instruction_count == 4
    assert summary.source_line_hotspots[0].instruction_share == pytest.approx(0.50)
    assert summary.source_line_hotspots[0].pipes == ("MTE2",)
    assert summary.source_line_hotspots[1].source_location == "kernels/tile_op.cpp:78"
    assert summary.source_line_hotspots[1].instruction_count == 3
    assert summary.source_line_hotspots[2].source_location == "kernels/tile_op.cpp:105"
    assert summary.source_line_hotspots[2].instruction_count == 1

    table = parse_pc_start_addr(pc_file)
    assert len(table) == 3
    assert summary.to_dict()["mte_waitflag_share"] == pytest.approx(0.25)


def test_orthogonal_hbm_bound_vs_redundant_gm_roundtrip_in_p0_and_p1() -> None:
    """Verify transfer_inflation_ratio separates HBM saturation from redundant GM traffic."""
    roofline = _sample_roofline(min_gm_bytes=1_000_000, compute_floor_us=10.0)

    p0_true_hbm_ledger = _sample_ledger(gm_roundtrip_bytes=65_000, grid_blocks=16)
    p0_true_obs = ObservationVector5D.from_tiers(
        p0_true_hbm_ledger,
        telemetry={"dur_us": 25.0, "mte2_wait_ratio": 0.72, "l2_hit_rate": 0.85},
        roofline=roofline,
    )
    assert p0_true_obs.transfer_inflation_ratio == pytest.approx(1.04)
    assert p0_true_obs.offchip_dma.transfer_inflation_ratio == pytest.approx(1.04)
    assert p0_true_obs.transfer_inflation_ratio <= 1.15
    assert p0_true_obs.offchip_dma.l2_mte2_wait_ratio > 0.65

    p0_redundant_ledger = _sample_ledger(gm_roundtrip_bytes=120_000, grid_blocks=16)
    p0_redundant_obs = ObservationVector5D.from_tiers(
        p0_redundant_ledger,
        telemetry={"dur_us": 45.0, "mte2_wait_ratio": 0.62, "l2_hit_rate": 0.85},
        roofline=roofline,
    )
    assert p0_redundant_obs.transfer_inflation_ratio == pytest.approx(1.92)
    assert p0_redundant_obs.transfer_inflation_ratio > 1.35
    assert p0_redundant_obs.offchip_dma.l2_mte2_wait_ratio > 0.55

    p1_true_res = compute_static_dynamic_residual(
        p0_redundant_ledger,
        {
            "dur_us": 24.0,
            "mte2_wait_ratio": 0.74,
            "l2_hit_rate": 0.82,
            "hbm_read_bytes": 550_000,
            "hbm_write_bytes": 520_000,
        },
        roofline=roofline,
    )
    assert p1_true_res.transfer_inflation_ratio == pytest.approx(1.07)
    assert p1_true_res.observation is not None
    assert p1_true_res.observation.transfer_inflation_ratio == pytest.approx(1.07)

    p1_redundant_res = compute_static_dynamic_residual(
        p0_true_hbm_ledger,
        {
            "dur_us": 52.0,
            "mte2_wait_ratio": 0.64,
            "hbm_read_bytes": 1_200_000,
            "hbm_write_bytes": 650_000,
        },
        roofline=roofline,
    )
    assert p1_redundant_res.transfer_inflation_ratio == pytest.approx(1.85)
    assert p1_redundant_res.transfer_inflation_ratio > 1.35

    method_obs = ObservationVector5D.from_tiers(
        _FullGridMethodLedger(pipe_cycles={"AIV0.PIPE_V": 10}, waterlines={}, gm_roundtrip_bytes=1),
        telemetry={"dur_us": 30.0},
        roofline=roofline,
    )
    assert method_obs.transfer_inflation_ratio == pytest.approx(1.50)

    scoped_obs = ObservationVector5D.from_tiers(
        _ScopedFullGridLedger(
            pipe_cycles={"AIV0.PIPE_V": 10},
            waterlines={},
            gm_roundtrip_bytes=1_100_000,
        ),
        telemetry={"dur_us": 30.0},
        roofline=roofline,
    )
    assert scoped_obs.transfer_inflation_ratio == pytest.approx(1.10)


def test_orthogonal_compute_ceiling_vs_degraded_vector_execution_and_roundtrip() -> None:
    """Verify effective_compute_eta separates compute saturation from degraded execution."""
    roofline = _sample_roofline(min_gm_bytes=500_000, compute_floor_us=18.0, memory_floor_us=8.0)
    ledger = _sample_ledger(gm_roundtrip_bytes=32_000, grid_blocks=16)

    ceiling_res = compute_static_dynamic_residual(
        ledger,
        {
            "dur_us": 24.0,
            "aiv_vec_ratio": 0.78,
            "hbm_read_bytes": 260_000,
            "hbm_write_bytes": 260_000,
        },
        roofline=roofline,
    )
    assert ceiling_res.measured_aiv_vec_ratio > 0.65
    assert ceiling_res.effective_compute_eta == pytest.approx(0.75)
    assert ceiling_res.effective_compute_eta >= 0.60
    assert ceiling_res.observation is not None
    assert ceiling_res.observation.effective_compute_eta == pytest.approx(0.75)
    assert ceiling_res.observation.compute_axis_residuals().compute_issue_gap == 0.0

    degraded_res = compute_static_dynamic_residual(
        ledger,
        {
            "dur_us": 100.0,
            "aiv_vec_ratio": 0.72,
            "hbm_read_bytes": 260_000,
            "hbm_write_bytes": 260_000,
        },
        roofline=roofline,
    )
    assert degraded_res.measured_aiv_vec_ratio > 0.60
    assert degraded_res.effective_compute_eta == pytest.approx(0.18)
    assert degraded_res.effective_compute_eta < 0.25
    assert degraded_res.observation is not None
    assert degraded_res.observation.compute_axis_residuals().compute_issue_gap == pytest.approx(
        0.82
    )

    obs_dict = degraded_res.observation.to_dict()
    assert obs_dict["effective_compute_eta"] == pytest.approx(0.18)
    assert obs_dict["transfer_inflation_ratio"] == pytest.approx(1.04)
    assert ObservationVector5D.from_dict(obs_dict) == degraded_res.observation

    res_dict = degraded_res.to_dict()
    assert res_dict["effective_compute_eta"] == pytest.approx(0.18)
    assert res_dict["transfer_inflation_ratio"] == pytest.approx(1.04)
    assert StaticDynamicResidual.from_dict(res_dict) == degraded_res

    tel_fallback_obs = ObservationVector5D.from_tiers(
        ledger,
        telemetry={
            "dur_us": 90.0,
            "blueprint_min_gm_bytes": 500_000,
            "blueprint_compute_floor_us": 18.0,
            "actual_gm_bytes": 800_000,
        },
    )
    assert tel_fallback_obs.transfer_inflation_ratio == pytest.approx(1.60)
    assert tel_fallback_obs.effective_compute_eta == pytest.approx(0.20)
    assert "l2_sync_stall_ratio" not in tel_fallback_obs.measured_l2_fields

    null_sync_obs = ObservationVector5D.from_tiers(
        ledger,
        telemetry={"dur_us": 20.0, "sync_stall_ratio": None},
    )
    assert "l2_sync_stall_ratio" not in null_sync_obs.measured_l2_fields


def test_surrogate_extract_msprof_honest_telemetry_end_to_end() -> None:
    """Surrogate profiler extraction omits absent sync_stall_ratio into ObservationVector5D."""
    prof = SimpleNamespace(
        aic_mad=0.20,
        aiv_vec=0.20,
        aic_mte2=0.10,
        aiv_mte2=0.10,
        aic_total_cycles=1000,
        aiv_total_cycles=1000,
        dur_us=25.0,
    )
    metrics = _extract_msprof_metrics(prof)
    assert "sync_stall_ratio" not in metrics
    obs = ObservationVector5D.from_tiers(_sample_ledger(), telemetry=metrics)
    assert "l2_sync_stall_ratio" not in obs.measured_l2_fields
    assert obs.pipeline_sync.l2_sync_stall_ratio == 0.0
