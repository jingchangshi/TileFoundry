"""Tests for parent-relative Stage-A ranking, staged ledger spans, and receipt verification."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tilefoundry_ascend.agent.orchestrator import gate_funnel as gf
from tilefoundry_ascend.agent.orchestrator import tilesize_prescreen as tp
from tilefoundry_ascend.ascend.cost_model.hybrid_cost_model import CostFeatures, CostPrediction
from tilefoundry_ascend.ascend.target import AscendTarget


class _FakeLedger:
    is_legal = True
    steady_state_ii = 1
    pipe_cycles = {"AIV0.PIPE_V": 1}
    dominant_pipe = "AIV0.PIPE_V"
    waterlines: dict[str, object] = {}
    min_gm_dma_bytes = 0
    dma_burst_efficiency_ratio = 1.0
    grid_blocks = 56


def _prediction(pred_us: float) -> CostPrediction:
    return CostPrediction(
        pred_us=pred_us,
        tier0_physics_us=pred_us,
        tier1_penalty_us=0.0,
        tier2_residual_factor=1.0,
        cube_floor_us=0.0,
        vec_floor_us=0.0,
        hbm_floor_us=pred_us,
        l2_hit_rate=0.0,
        wave_quantization_ratio=1.0,
        bank_conflict_us=0.0,
        effective_hbm_bw_bps=1.0,
        dominant_bound="hbm",
    )


class _StubCostModel:
    def __init__(self, predictions: list[float], ratios: list[float] | None = None) -> None:
        self.target = AscendTarget("ascend.950pr")
        self._predictions = list(predictions)
        self._ratios = list(ratios or [])
        self.ratio_parents: list[CostFeatures] = []

    def predict(self, _features: CostFeatures) -> CostPrediction:
        return _prediction(self._predictions.pop(0))

    def predict_delta_ratio(self, _candidate: CostFeatures, parent: CostFeatures) -> float:
        self.ratio_parents.append(parent)
        return self._ratios.pop(0)


def _stage_a(ledger: object | None = None) -> gf.StageAResult:
    return gf.StageAResult(passed=True, prim_func=None, ledger=ledger)


def test_rank_stage_a_candidates_applies_parent_delta_ratio() -> None:
    model = _StubCostModel([100.0, 10.0, 20.0], [1.5, 0.8])
    candidates = [("alpha", "", _stage_a()), ("beta", "", _stage_a())]

    ranked = tp.rank_stage_a_candidates_with_cost_model(
        candidates,
        cost_model=model,
        parent_ledger=object(),
    )

    assert [item.candidate_id for item in ranked] == ["beta", "alpha"]
    assert [item.pred_us for item in ranked] == [80.0, 150.0]
    assert len(model.ratio_parents) == 2
    assert model.ratio_parents[0] is model.ratio_parents[1]


def test_rank_stage_a_candidates_without_parent_keeps_absolute_cost_and_ties() -> None:
    model = _StubCostModel([7.0, 7.0, 3.0])
    candidates = [("first", "", _stage_a()), ("second", "", _stage_a()), ("third", "", _stage_a())]

    ranked = tp.rank_stage_a_candidates_with_cost_model(candidates, cost_model=model)

    assert [item.candidate_id for item in ranked] == ["third", "first", "second"]
    assert [item.pred_us for item in ranked] == [3.0, 7.0, 7.0]
    assert model.ratio_parents == []


_DSL_SOURCE = """
def build_kernel(bm=64, bn=64, bk=64, stages=2):
    return None
"""


def test_live_prescreen_reuses_center_as_delta_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _StubCostModel([100.0, 100.0], [0.4])

    def fake_stage_a(*_args: object, **_kwargs: object) -> gf.StageAResult:
        return _stage_a(_FakeLedger())

    monkeypatch.setattr(gf, "run_stage_a_static", fake_stage_a)
    summary = tp.prescreen_live_ir_neighborhood(
        _DSL_SOURCE,
        factory_name="build_kernel",
        center=tp.TileConfigCandidate(bm=64, bn=64, bk=64, stages=2),
        max_variants=2,
        cost_model=model,
    )

    assert "1. bm=32" in summary
    assert "2. bm=64" in summary
    assert "pred_us=40.00us" in summary
    assert "pred_us=100.00us" in summary
    assert len(model.ratio_parents) == 1


_STAGED_UB_TU = r"""
constexpr int STAGE_ELEMS = ROWS_BLK * D_UB_STRIDE + 64;
constexpr int STAGE_BYTES = STAGE_ELEMS * sizeof(InT);
template <int ROWS_BLK, int D_UB_STRIDE, int STAGES, typename InT>
__global__ __vector__ void softmax_like(__gm__ InT* src) {
    __ubuf__ InT* x_ub0 = (__ubuf__ InT*)0;
    __ubuf__ InT* x_ub1 = (__ubuf__ InT*)(STAGES == 2 ? STAGE_BYTES : 0);
    __ubuf__ float* exp_ub = (__ubuf__ float*)(STAGES * STAGE_BYTES);
}
softmax_like<16, 64, 2, half>(nullptr);
"""


def test_static_ledger_expands_template_stage_offsets() -> None:
    ledger = gf._build_ascendc_static_ledger(_STAGED_UB_TU)

    assert ledger.waterlines["UB"].explicit_bytes == 3 * 2176


_SINGLE_STAGE_UB_TU = r"""
constexpr int STAGE_BYTES = 16384;
template <int STAGES>
__global__ __vector__ void staged_kernel() {
    __ubuf__ half* x_ub0 = (__ubuf__ half*)0;
    asc_sync_notify(PIPE_MTE2, PIPE_V, (event_t)0);
    asc_sync_wait(PIPE_MTE2, PIPE_V, (event_t)0);
    asc_sync_notify(PIPE_MTE2, PIPE_V, (event_t)1);
    asc_sync_wait(PIPE_MTE2, PIPE_V, (event_t)1);
}
staged_kernel<2>();
"""


def test_static_ledger_expands_single_staged_pointer_from_ping_pong() -> None:
    ledger = gf._build_ascendc_static_ledger(_SINGLE_STAGE_UB_TU)

    assert ledger.waterlines["UB"].explicit_bytes == 32768


def test_static_ledger_explicit_ub_bytes_overrides_inferred_span() -> None:
    source = _SINGLE_STAGE_UB_TU.replace(
        "constexpr int STAGE_BYTES = 16384;",
        "constexpr int ub_bytes = 4096;\nconstexpr int STAGE_BYTES = 16384;",
    )
    ledger = gf._build_ascendc_static_ledger(source)

    assert ledger.waterlines["UB"].explicit_bytes == 4096


_L1_TAIL_TU = r"""
constexpr int L1_STAGES = 2;
constexpr int K_ROPE_STAGE_BYTES = 16384;
__cbuf__ half* k_rope_l1_base = (__cbuf__ half*)0;
void run() {
    asc_mmad();
}
"""


def test_static_ledger_preserves_declared_l1_tail_size() -> None:
    ledger = gf._build_ascendc_static_ledger(_L1_TAIL_TU)

    assert ledger.waterlines["L1"].explicit_bytes == 16384


def test_static_ledger_ignores_other_scope_stage_factor_for_ub() -> None:
    source = r"""
constexpr int L1_STAGES = 2;
constexpr int STAGE_BYTES = 16384;
__ubuf__ half* x_ub = (__ubuf__ half*)0;
"""
    ledger = gf._build_ascendc_static_ledger(source)

    assert ledger.waterlines["UB"].explicit_bytes == 8192


def test_static_ledger_does_not_scale_numbered_tail_stage_bytes() -> None:
    source = r"""
constexpr int STAGES = 2;
constexpr int X_STAGE = 2048;
constexpr int X_STAGE_BYTES = X_STAGE * sizeof(half);
__ubuf__ half* x_ub0 = (__ubuf__ half*)0;
__ubuf__ half* x_ub1 = (__ubuf__ half*)(X_STAGE_BYTES);
"""
    ledger = gf._build_ascendc_static_ledger(source)

    assert ledger.waterlines["UB"].explicit_bytes == 8192


_TRANSFER_SOURCE = r"""
constexpr int BM = 64;
constexpr int BN = 128;
constexpr int BK = 32;
void run() {
    asc_copy_gm2l1_nd2nz(a_l1, src, 0, 0, BM, BK, 0, 0);
}
"""

_TRANSFER_OBJDUMP = """
00000000: 0000 MOV_OUT_TO_L1 X0, X1
00000004: 0000 MOV_OUT_TO_L1 X2, X3
00000008: 0000 MOV_OUT_TO_UB X4, X5
0000000c: 0000 MOV_OUT_TO_UB X6, X7
"""


def test_disassembly_uses_route_specific_counts_and_granularity() -> None:
    counts = gf._parse_objdump_pipe_counts(_TRANSFER_OBJDUMP)
    merged = gf._merge_disassembly_microarch_profile(_TRANSFER_SOURCE, _TRANSFER_OBJDUMP)

    assert counts["gm_to_l1_insns"] == 2
    assert counts["gm_to_ub_insns"] == 2
    assert merged is not None
    assert merged["gm_to_l1_bytes"] == 8192
    assert merged["gm_to_ub_bytes"] == 32768


def test_disassembly_preserves_base_bytes_without_transfer_evidence() -> None:
    objdump = "00000000: 0000 MOV_OUT_TO_UB X0, X1, #4096\n"
    merged = gf._merge_disassembly_microarch_profile("void empty_kernel() {}\n", objdump)

    assert merged is not None
    assert merged["gm_to_ub_bytes"] == 0


_GROUPED_LAYOUT_SOURCE = r"""
template <int BM, int BN, int BK, int TRANS_W, typename InT>
__global__ __mix__(1, 2) void gmm_persistent_16() {
    constexpr int ROWS = BM / 2;
    constexpr int A_STAGE = BM * BK;
    constexpr int B_STAGE = TRANS_W ? (BN * BK) : (BK * BN);
    __cbuf__ InT* a_l1_base = (__cbuf__ InT*)0;
    __cbuf__ InT* b_l1_base = (__cbuf__ InT*)(2 * A_STAGE * 2);
    constexpr int C_UB_BYTES = ROWS * BN * 4;
    constexpr int OUT_UB_BYTES = ROWS * BN * 2;
    __ubuf__ float* c_ub0 = (__ubuf__ float*)0;
    __ubuf__ float* c_ub1 = (__ubuf__ float*)(C_UB_BYTES);
    __ubuf__ InT* out_ub0 = (__ubuf__ InT*)(2 * C_UB_BYTES);
    __ubuf__ InT* out_ub1 = (__ubuf__ InT*)(2 * C_UB_BYTES + OUT_UB_BYTES);
    __cbuf__ InT* a_dst = a_l1_base + stg_id * A_STAGE;
    __cbuf__ InT* b_dst = b_l1_base + stg_id * B_STAGE;
}
gmm_persistent_16<64, 128, 64, 0, half>(nullptr);
"""


def test_static_ledger_sizes_grouped_matmul_tail_by_its_stage_stride() -> None:
    ledger = gf._build_ascendc_static_ledger(_GROUPED_LAYOUT_SOURCE)
    ub_buffers = ledger.waterlines["UB"].explicit_buffers

    assert ledger.waterlines["L1"].explicit_bytes == 49152
    assert ledger.waterlines["UB"].explicit_bytes == 49152
    assert ub_buffers["c_ub0"] == 16384
    assert ub_buffers["c_ub1"] == 16384
    assert ub_buffers["out_ub0"] == 8192
    assert ub_buffers["out_ub1"] == 8192


def test_static_ledger_reads_rendered_grouped_trackb_layout() -> None:
    from examples.ascend950pr_ops.grouped_matmul.kernels import _render_trackb_tu  # noqa: PLC0415

    source = _render_trackb_tu(
        bm=64,
        bn=128,
        bk=64,
        tl_dtype="float16",
        bias_dtype="float32",
        has_bias=True,
    )
    ledger = gf._build_ascendc_static_ledger(source)
    buffers = ledger.waterlines["L1"].explicit_buffers
    ub_buffers = ledger.waterlines["UB"].explicit_buffers

    assert buffers["b_l1_base"] == 32768
    assert buffers["b_hi_base"] == 32768
    assert buffers["b_lo_base"] == 32768
    assert ub_buffers["c_ub0"] == 16384
    assert ub_buffers["c_ub1"] == 16384
    assert ub_buffers["out_ub1"] == 8192


def test_disassembly_resolved_granularity_replaces_legacy_unit_prior() -> None:
    source = r"""
constexpr int BM = 64;
constexpr int BN = 64;
constexpr int BK = 64;
void run() {
    copy_gm_to_cbuf(dst, src, 0);
}
"""
    objdump = "00000000: 0000 MOV_OUT_TO_L1 X0, X1\n"

    merged = gf._merge_disassembly_microarch_profile(source, objdump)

    assert merged is not None
    assert merged["gm_to_l1_bytes"] == 8192


class _Proc:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class _Observed:
    def __init__(self) -> None:
        self.verify_calls = 0
        self.verify_env_exists = False
        self.run_env_exists = False
        self.temp_env: Path | None = None
        self.verify_cmd: list[str] = []


def _invoke_stage_c(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    run_payload: dict[str, object],
    verify_stdout: str,
    verify_rc: int = 0,
    with_telemetry: bool = True,
) -> tuple[gf.StageCResult, _Observed]:
    source = tmp_path / "source"
    source.mkdir()
    env_file = tmp_path / "env.sh"
    env_file.write_text('TILEFOUNDRY_NPU_DEVICE_TESTS="old"\n', encoding="utf-8")
    telemetry_path = tmp_path / "telemetry.json"
    if with_telemetry:
        telemetry_path.write_text('{"dur_us": 12.5}', encoding="utf-8")
    observed = _Observed()

    def fake_run(cmd: list[str], **_kwargs: object) -> _Proc:
        if cmd[1] == "run":
            run_env = Path(cmd[cmd.index("--env-file") + 1])
            observed.run_env_exists = run_env.is_file()
            observed.temp_env = run_env
            return _Proc(0, json.dumps(run_payload))
        observed.verify_calls += 1
        observed.verify_cmd = cmd
        observed.verify_env_exists = observed.temp_env is not None and observed.temp_env.is_file()
        return _Proc(verify_rc, verify_stdout)

    monkeypatch.setattr(gf, "_resolve_agent_gates_bin", lambda: "agent-gates")
    monkeypatch.setattr(gf.subprocess, "run", fake_run)
    ledger = gf._build_ascendc_static_ledger("void k() {}")
    result = gf.run_stage_c_agent_gates(
        ledger,
        source_root=source,
        device_test_rel="tests/test_device.py",
        env_file=env_file,
        telemetry_report_path=telemetry_path,
    )
    return result, observed


def test_stage_c_requires_verified_receipt_before_pass(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result, observed = _invoke_stage_c(
        monkeypatch,
        tmp_path,
        run_payload={"status": "PASS", "runId": "run-1", "receiptPath": "/runs/run-1"},
        verify_stdout=json.dumps({"runId": "run-1", "status": "PASS", "valid": True}),
    )

    assert result.passed is True
    assert result.accuracy_passed is True
    assert result.run_id == "run-1"
    assert result.telemetry == {"dur_us": 12.5}
    assert result.measured_latency_us == 12.5
    assert result.error_message == ""
    assert observed.verify_calls == 1
    assert observed.run_env_exists is True
    assert observed.verify_env_exists is True
    assert observed.verify_cmd == ["agent-gates", "verify", "--run-id", "run-1", "--json"]


def test_stage_c_rejects_receipt_not_marked_valid(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result, observed = _invoke_stage_c(
        monkeypatch,
        tmp_path,
        run_payload={"status": "PASS", "runId": "run-1"},
        verify_stdout=json.dumps({"runId": "run-1", "status": "PASS", "valid": False}),
    )

    assert result.passed is False
    assert result.telemetry == {}
    assert "valid=False" in result.error_message
    assert observed.verify_calls == 1


def test_stage_c_rejects_stale_receipt_and_drops_telemetry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result, observed = _invoke_stage_c(
        monkeypatch,
        tmp_path,
        run_payload={"status": "PASS", "runId": "run-1"},
        verify_stdout=json.dumps({"runId": "run-1", "status": "STALE", "valid": False}),
    )

    assert result.passed is False
    assert result.accuracy_passed is False
    assert result.telemetry == {}
    assert result.measured_latency_us == 0.0
    assert "receipt_verification_failed" in result.error_message
    assert "status=STALE" in result.error_message
    assert observed.verify_calls == 1


def test_stage_c_rejects_mismatched_verified_run_id(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result, observed = _invoke_stage_c(
        monkeypatch,
        tmp_path,
        run_payload={"status": "PASS", "runId": "run-1"},
        verify_stdout=json.dumps({"runId": "run-2", "status": "PASS", "valid": True}),
    )

    assert result.passed is False
    assert result.telemetry == {}
    assert "runId=run-2" in result.error_message
    assert observed.verify_calls == 1


def test_stage_c_rejects_malformed_verify_response(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result, observed = _invoke_stage_c(
        monkeypatch,
        tmp_path,
        run_payload={"status": "PASS", "runId": "run-1"},
        verify_stdout="receipt verification failed\n",
    )

    assert result.passed is False
    assert result.telemetry == {}
    assert "status=MISSING" in result.error_message
    assert observed.verify_calls == 1


def test_stage_c_rejects_pass_without_run_id_and_skips_verify(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result, observed = _invoke_stage_c(
        monkeypatch,
        tmp_path,
        run_payload={"status": "PASS"},
        verify_stdout="",
    )

    assert result.passed is False
    assert result.telemetry == {}
    assert "without a runId" in result.error_message
    assert observed.verify_calls == 0


def test_stage_c_does_not_verify_failed_run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result, observed = _invoke_stage_c(
        monkeypatch,
        tmp_path,
        run_payload={"status": "FAIL", "runId": "run-1"},
        verify_stdout="",
    )

    assert result.passed is False
    assert result.telemetry == {}
    assert observed.verify_calls == 0
