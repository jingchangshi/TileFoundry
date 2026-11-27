"""Reject/accept tests for L0/L1 SSOT helpers, unit_kind compile keys, and JsonRecordMixin."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from tilefoundry_ascend.agent.contracts import CompileTaskSpec
from tilefoundry_ascend.agent.specs.fault_types import FaultPlane, TypedFault
from tilefoundry_ascend.agent.specs.latency_algebra import StaticDynamicResidual
from tilefoundry_ascend.agent.specs.ops_registry import GuardrailCaseSpec
from tilefoundry_ascend.ascend.compile_cache import (
    CompileTaskSpecLike,
    UnitKind,
    compute_compile_task_key,
)
from tilefoundry_ascend.ascend.cost_model.hybrid_cost_model import (
    CalibrationSample,
    CostModelEvalReport,
)
from tilefoundry_ascend.ascend.cost_model.pipe_schedule_eval import Ascend950PipeScheduleResult
from tilefoundry_ascend.ascend.target import (
    VALID_CANN_SERIES,
    AscendSpaceFacts,
    AscendTarget,
    DaVinci950,
    resolve_cann_series,
)
from tilefoundry_ascend.ascend.target.architecture import is_dma_path_legal


@dataclass(frozen=True)
class _FakeCompileTaskSpec:
    builder_name: str
    static_kwargs: tuple[tuple[str, object], ...]
    pass_configs: tuple[tuple[str, object], ...]
    compile_flags: tuple[str, ...]
    unit_kind: UnitKind
    unit_source: str


def test_reject_unit_kind_key_collision_and_stale_cann_9_2() -> None:
    """Reject colliding keys across unit_kind variants and reject stale cann_9_2."""
    src = "extern \"C\" void kernel_entry() {}"
    key_tl = compute_compile_task_key(
        normalized_kernel_unit_src=src,
        compile_flags=("-O2",),
        unit_kind="tilelang_dsl",
    )
    key_cpp = compute_compile_task_key(
        normalized_kernel_unit_src=src,
        compile_flags=("-O2",),
        unit_kind="cce_cpp20",
    )
    assert key_tl != key_cpp

    spec_tl: CompileTaskSpecLike = _FakeCompileTaskSpec(
        builder_name="build_fn",
        static_kwargs=(("tile_m", 16),),
        pass_configs=(("tl.ascend_auto_sync", True),),
        compile_flags=("-O2",),
        unit_kind="tilelang_dsl",
        unit_source=src,
    )
    spec_cpp = CompileTaskSpec(
        task_key="",
        builder_name="build_fn",
        static_kwargs=(("tile_m", 16),),
        pass_configs=(("tl.ascend_auto_sync", True),),
        compile_flags=("-O2",),
        unit_kind="cce_cpp20",
        unit_source=src,
    )
    assert compute_compile_task_key(spec_tl) != compute_compile_task_key(spec_cpp)

    invalid_kwargs: dict[str, object] = {
        "normalized_kernel_unit_src": src,
        "unit_kind": "invalid_kind",
    }
    with pytest.raises(ValueError, match="unit_kind"):
        compute_compile_task_key(**invalid_kwargs)

    target = AscendTarget("ascend.950pr")
    arch = target.architecture
    assert isinstance(arch, DaVinci950)
    assert "cann_9_2" not in VALID_CANN_SERIES
    assert arch.default_cann_series == "cann_9_3"
    assert arch.supported_cann_series == ("cann_9_3",)
    with pytest.raises(ValueError, match="Unsupported CANN series"):
        resolve_cann_series("cann_9_2")


def test_accept_ssot_key_dma_legal_and_json_record_roundtrip() -> None:
    """Accept canonical compile keys, SSOT DMA path checks, and JsonRecordMixin roundtrips."""
    spec_a = CompileTaskSpec(
        task_key="",
        builder_name="build_norm",
        static_kwargs=(("d_dim", 2048), ("tile_m", 8)),
        pass_configs=(("b", 2), ("a", 1)),
        compile_flags=("-O2", "-std=c++20"),
        unit_kind="cce_cpp20",
        unit_source="void norm() {}",
    )
    spec_b = _FakeCompileTaskSpec(
        builder_name="build_norm",
        static_kwargs=(("tile_m", 8), ("d_dim", 2048)),
        pass_configs=(("a", 1), ("b", 2)),
        compile_flags=("-std=c++20", "-O2"),
        unit_kind="cce_cpp20",
        unit_source="void norm() {}",
    )
    assert compute_compile_task_key(spec_a) == compute_compile_task_key(spec_b)

    target = AscendTarget("ascend.950pr")
    arch = target.architecture
    facts = target.get_facts(AscendSpaceFacts)
    assert isinstance(arch, DaVinci950)
    assert isinstance(facts, AscendSpaceFacts)
    for src, dst, expected in (
        ("gm", "ub", True),
        ("l0c", "ub", True),
        ("l1", "gm", False),
        ("gm", "l0a", False),
    ):
        ssot = is_dma_path_legal(arch.valid_dma_paths, arch.forbidden_dma_paths, src, dst)
        assert ssot is expected
        assert arch.is_dma_path_legal(src, dst) is expected
        assert facts.is_dma_path_legal(src, dst) is expected

    fault = TypedFault(FaultPlane.COMPILER_LOWERING, "E_STACK", "L2", "overflow", (0, 2), False)
    assert TypedFault.from_json(fault.to_json()) == fault

    residual = StaticDynamicResidual(
        measured_latency_us=12.5,
        predicted_bottleneck_util=0.75,
        measured_aic_mac_ratio=0.1,
        measured_aiv_vec_ratio=0.6,
        measured_mte2_wait_ratio=0.2,
        measured_sync_stall_ratio=0.05,
        unexplained_gap_ratio=0.15,
        stack_bytes=64,
    )
    assert StaticDynamicResidual.from_json(residual.to_json()) == residual

    guard = GuardrailCaseSpec("rms_norm", 4, "wide", 18.0, 6.0)
    assert GuardrailCaseSpec.from_json(guard.to_json()) == guard

    sample = CalibrationSample(
        named_features={"intercept": 1.0, "wave_tail_waste": 0.2},
        measured_us=20.0,
        analytical_us=16.0,
    )
    assert CalibrationSample.from_json(sample.to_json()) == sample

    report = CostModelEvalReport(10, 0.12, 0.09, 0.91, 0.88, 0.9, 0.03)
    assert CostModelEvalReport.from_json(report.to_json()) == report

    pipe_res = Ascend950PipeScheduleResult(
        pipe_ns={"MTE2": 250.0, "VECTOR": 400.0},
        ave_compute_ns=400.0,
        ave_io_ns=250.0,
        steady_state_ii_ns=400.0,
        drain_bubble_ns=30.0,
        loop_total_ns=430.0,
        dominant_pipe="VECTOR",
        total_ns=430.0,
    )
    assert Ascend950PipeScheduleResult.from_json(pipe_res.to_json()) == pipe_res
