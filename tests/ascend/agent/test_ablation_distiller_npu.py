"""Live Ascend 950 NPU verification for WP6 flywheel micro-probe synthesis and admission."""

import os
from pathlib import Path

import pytest
import torch

from tilefoundry_ascend.agent import BayesianRuleTracker, FlywheelDistiller, GateFunnel
from tilefoundry_ascend.agent.orchestrator import StageCResult, compute_static_dynamic_residual

_AR_BASE = """
import tilelang.ascend.language as T

def make_ar_probe_kernel():
    n_depth = 4
    tile = 128

    @T.prim_func
    def main(
        A: T.Tensor((n_depth, 4, tile), "float32"),
        Y: T.Tensor((4, tile), "float32"),
    ):
        with T.Kernel(1) as bx:
            a_ub = T.alloc_shared((tile,), "float32")
            y_ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({a_ub: 1, y_ub: 1})
            for m in T.serial(4):
                with T.SimdVF():
                    m32 = T.simd.pset(32)
                    vz = T.simd.vdup(T.float32(0.0), "float32", mask=m32)
                    for k in range(2):
                        T.simd.vsts(y_ub[k * 64], vz, mask=m32)
                for n in T.Pipelined(n_depth, num_stages=1):
                    T.copy(A[n, m, 0:tile], a_ub)
                    with T.SimdVF():
                        m32 = T.simd.pset(32)
                        for k in range(2):
                            va = T.simd.vld(a_ub[k * 64])
                            vy = T.simd.vld(y_ub[k * 64])
                            T.simd.vsts(y_ub[k * 64], T.simd.vadd(vy, va, m32), m32)
                T.copy(y_ub, Y[m, 0:tile])
                T.ascend_pipe_barrier("PIPE_ALL")
    return main
"""

_AR_WINNER = """
import tilelang.ascend.language as T

def make_ar_probe_kernel():
    n_depth = 4
    tile = 128

    @T.prim_func
    def main(
        A: T.Tensor((n_depth, 4, tile), "float32"),
        Y: T.Tensor((4, tile), "float32"),
    ):
        with T.Kernel(4) as bx:
            a_ub = T.alloc_shared((tile,), "float32")
            y_ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({a_ub: 2, y_ub: 2})
            with T.SimdVF():
                m32 = T.simd.pset(32)
                vz = T.simd.vdup(T.float32(0.0), "float32", mask=m32)
                for k in range(2):
                    T.simd.vsts(y_ub[k * 64], vz, mask=m32)
            for n in T.Pipelined(n_depth, num_stages=2):
                T.copy(A[n, bx, 0:tile], a_ub)
                with T.SimdVF():
                    m32 = T.simd.pset(32)
                    for k in range(2):
                        va = T.simd.vld(a_ub[k * 64])
                        vy = T.simd.vld(y_ub[k * 64])
                        T.simd.vsts(y_ub[k * 64], T.simd.vadd(vy, va, m32), m32)
            T.copy(y_ub, Y[bx, 0:tile])
    return main
"""


def test_live_flywheel_micro_probe_verification_and_kb_admission(tmp_path: Path) -> None:
    """Distill winning DSL diffs into 30-50 line micro-probes, verify on NPU, and admit rules."""
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Ascend NPU device is required for live flywheel micro-probe verification.")

    import tilelang  # noqa: PLC0415

    from tilefoundry_ascend.agent.probes import run_msprof_detail  # noqa: PLC0415

    def _npu_eval(_dsl: str, stage_a) -> StageCResult:
        assert stage_a.prim_func is not None and stage_a.ledger is not None
        try:
            kernel = tilelang.compile(stage_a.prim_func, target="ascend", out_idx=[1])
            a = torch.randn((4, 4, 128), dtype=torch.float32).npu()
            out = kernel(a)
            torch.npu.synchronize()
            torch.testing.assert_close(out.cpu(), a.cpu().sum(dim=0), atol=1e-4, rtol=1e-4)
            prof = run_msprof_detail(lambda: kernel(a), n_warmup=2, n_repeat=5)
            tel = {"dur_us": float(prof.dur_us), "aiv_vec": float(prof.aiv_vec), "aiv_mte2": float(prof.aiv_mte2)}
            return StageCResult(
                passed=True,
                accuracy_passed=True,
                measured_latency_us=float(prof.dur_us),
                residual=compute_static_dynamic_residual(stage_a.ledger, tel),
                telemetry=tel,
            )
        except Exception as exc:
            return StageCResult(
                passed=False,
                accuracy_passed=False,
                measured_latency_us=0.0,
                residual=None,
                telemetry={},
                error_message=str(exc)[-300:],
            )

    tracker = BayesianRuleTracker.from_seed_yaml()
    initial_rules = len(tracker.rules)
    funnel = GateFunnel(device_runner=_npu_eval)

    base_res = funnel.evaluate(_AR_BASE, factory_name="make_ar_probe_kernel")
    win_res = funnel.evaluate(_AR_WINNER, factory_name="make_ar_probe_kernel")
    assert base_res.passed and win_res.passed
    assert base_res.stage_c is not None and win_res.stage_c is not None

    out_yaml = tmp_path / "flywheel_admitted_kb.yaml"
    distiller = FlywheelDistiller(
        funnel=funnel,
        tracker=tracker,
        probe_runner=None,
        min_edit_gain=0.01,
        min_probe_speedup=1.05,
    )
    report = distiller.distill_from_pair(
        baseline_dsl=_AR_BASE,
        winner_dsl=_AR_WINNER,
        baseline_latency_us=base_res.stage_c.measured_latency_us,
        winner_latency_us=win_res.stage_c.measured_latency_us,
        factory_name="make_ar_probe_kernel",
        kb_output_path=out_yaml,
    )

    assert len(report.kept_positive_chunks) >= 1
    assert len(report.admitted_rule_ids) >= 1
    assert len(tracker.rules) > initial_rules
    assert out_yaml.is_file()
    for out in report.distilled_outcomes:
        if out.admitted:
            assert out.hardware_verified is True
            assert out.probe_metrics["speedup"] >= 1.05
            assert 25 <= out.probe.line_count_baseline <= 50
            assert 25 <= out.probe.line_count_optimized <= 50

    report_dir = os.environ.get("TILEFOUNDRY_FLYWHEEL_REPORT_DIR", "").strip()
    if report_dir:
        dest = Path(report_dir)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "wp6_flywheel_report.json").write_text(report.to_json(), encoding="utf-8")
