"""Live Ascend 950 NPU evolution test from HIR models using a configured provider."""

import asyncio
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tilefoundry_ascend.agent import (
    BayesianRuleTracker,
    GateFunnel,
    run_e2e_evolution,
)
from tilefoundry_ascend.agent.campaign.anchor_specs import parse_simple_env_file
from tilefoundry_ascend.agent.orchestrator import (
    KernelLLMBackend,
    MockLLMBackend,
    OpenAICompatHTTPBackend,
    StageCResult,
    compute_static_dynamic_residual,
    create_llm_backend_from_env,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_AR_MODEL_PATH = _REPO_ROOT / "examples" / "ascend950pr_ops" / "attention_residual" / "model.py"
_MLA_MODEL_PATH = _REPO_ROOT / "examples" / "ascend950pr_ops" / "mla" / "model.py"

_AR_TIER0_DSL = """
import tilelang.ascend.language as T

def make_ar_kernel():
    n_depth = 4
    d_dim = 128
    m_rows = 64
    v_rows = 256
    inv_d = 1.0 / float(d_dim)
    inv_sqrt_d = 1.0 / (float(d_dim) ** 0.5)

    @T.prim_func
    def main(
        V: T.Tensor((v_rows, d_dim), "bfloat16"),
        X: T.Tensor((m_rows, d_dim), "bfloat16"),
        W: T.Tensor((d_dim,), "bfloat16"),
        G: T.Tensor((d_dim,), "bfloat16"),
        Y: T.Tensor((m_rows, d_dim), "bfloat16"),
        alpha_val: T.float32,
        eps_val: T.float32,
    ):
        with T.Kernel(1) as cid:
            w_ub = T.alloc_shared((128,), "bfloat16")
            g_ub = T.alloc_shared((128,), "bfloat16")
            kw_ub = T.alloc_shared((128,), "float32")
            v_ub = T.alloc_shared((1, 128), "bfloat16")
            x_ub = T.alloc_shared((1, 128), "bfloat16")
            acc_ub = T.alloc_shared((1, 128), "float32")
            m_ub = T.alloc_shared((1, 8), "float32")
            l_ub = T.alloc_shared((1, 8), "float32")
            T.annotate_buffer_versions(
                {v_ub: 1, acc_ub: 1, m_ub: 1, l_ub: 1, kw_ub: 1, w_ub: 1, g_ub: 1, x_ub: 1}
            )
            T.copy(W[0:128], w_ub)
            T.copy(G[0:128], g_ub)
            with T.SimdVF():
                m32 = T.simd.pset(32, "PAT_ALL")
                m16 = T.simd.pset(16, "PAT_ALL")
                for c in T.serial(2):
                    off = c * 64
                    wf = T.simd.vcvt(T.simd.vld(w_ub[off], dist="US_B16"), "float32", mask=m16, part=0)
                    gf = T.simd.vcvt(T.simd.vld(g_ub[off], dist="US_B16"), "float32", mask=m16, part=0)
                    T.simd.vsts(kw_ub[off], T.simd.vmul(wf, gf, mask=m32), mask=m32, dist="NORM_B32")
            for m in T.serial(m_rows):
                with T.SimdVF():
                    m32 = T.simd.pset(32, "PAT_ALL")
                    vn = T.simd.vdup(T.float32(-1e38), "float32", mask=m32)
                    vz = T.simd.vdup(T.float32(0.0), "float32", mask=m32)
                    T.simd.vsts(m_ub[0, 0], vn, mask=m32, dist="ONEPT_B32")
                    T.simd.vsts(l_ub[0, 0], vz, mask=m32, dist="ONEPT_B32")
                    for c in T.serial(2):
                        T.simd.vsts(acc_ub[0, c * 64], vz, mask=m32, dist="NORM_B32")
                for n in T.Pipelined(n_depth, num_stages=1):
                    row_off = n * m_rows + m
                    T.copy(V[row_off, :128], v_ub[0, :128])
                    with T.SimdVF():
                        m32 = T.simd.pset(32, "PAT_ALL")
                        m16 = T.simd.pset(16, "PAT_ALL")
                        v_zero = T.simd.vdup(T.float32(0.0), "float32", mask=m32)
                        v_one = T.simd.vdup(T.float32(1.0), "float32", mask=m32)
                        sq_acc = T.simd.alloc_local((1,), "float32")
                        dot_acc = T.simd.alloc_local((1,), "float32")
                        sq_acc[0] = v_zero
                        dot_acc[0] = v_zero
                        for c in T.serial(2):
                            off = c * 64
                            vf = T.simd.vcvt(T.simd.vld(v_ub[0, off], dist="US_B16"), "float32", mask=m16, part=0)
                            kwf = T.simd.vld(kw_ub[off], dist="NORM_B32")
                            T.simd.vmula(sq_acc[0], vf, vf, mask=m32)
                            T.simd.vmula(dot_acc[0], vf, kwf, mask=m32)
                        sum_sq = T.simd.vdupv(T.simd.vcadd(sq_acc[0], mask=m32), mask=m32, pos="POS_LOWEST")
                        raw_dot = T.simd.vdupv(T.simd.vcadd(dot_acc[0], mask=m32), mask=m32, pos="POS_LOWEST")
                        rstd = T.simd.vdiv(v_one, T.simd.vsqrt(T.simd.vadds(T.simd.vmuls(sum_sq, T.float32(inv_d), mask=m32), eps_val, mask=m32), mask=m32), mask=m32)
                        sv = T.simd.vmuls(T.simd.vmul(raw_dot, rstd, mask=m32), T.float32(inv_sqrt_d), mask=m32)
                        m_old = T.simd.vld(m_ub[0, 0], dist="BRC_B32")
                        l_old = T.simd.vld(l_ub[0, 0], dist="BRC_B32")
                        m_new = T.simd.vmax(m_old, sv, mask=m32)
                        av = T.simd.vexp(T.simd.vsub(m_old, m_new, mask=m32), mask=m32)
                        pv = T.simd.vexp(T.simd.vsub(sv, m_new, mask=m32), mask=m32)
                        T.simd.vsts(m_ub[0, 0], m_new, mask=m32, dist="ONEPT_B32")
                        T.simd.vsts(l_ub[0, 0], T.simd.vadd(T.simd.vmul(l_old, av, mask=m32), pv, mask=m32), mask=m32, dist="ONEPT_B32")
                        for c in T.serial(2):
                            off = c * 64
                            vf = T.simd.vcvt(T.simd.vld(v_ub[0, off], dist="US_B16"), "float32", mask=m16, part=0)
                            a_old = T.simd.vld(acc_ub[0, off], dist="NORM_B32")
                            a_new = T.simd.vadd(T.simd.vmul(a_old, av, mask=m32), T.simd.vmul(vf, pv, mask=m32), mask=m32)
                            T.simd.vsts(acc_ub[0, off], a_new, mask=m32, dist="NORM_B32")
                T.copy(X[m, :128], x_ub[0, :128])
                with T.SimdVF():
                    m32 = T.simd.pset(32, "PAT_ALL")
                    m16 = T.simd.pset(16, "PAT_ALL")
                    v_one = T.simd.vdup(T.float32(1.0), "float32", mask=m32)
                    inv_l = T.simd.vdiv(v_one, T.simd.vld(l_ub[0, 0], dist="BRC_B32"), mask=m32)
                    s_agg = T.simd.vmuls(inv_l, alpha_val, mask=m32)
                    one_ma = T.float32(1.0) - alpha_val
                    for c in T.serial(2):
                        off = c * 64
                        xf = T.simd.vcvt(T.simd.vld(x_ub[0, off], dist="US_B16"), "float32", mask=m16, part=0)
                        af = T.simd.vld(acc_ub[0, off], dist="NORM_B32")
                        yf = T.simd.vadd(T.simd.vmuls(xf, one_ma, mask=m32), T.simd.vmul(af, s_agg, mask=m32), mask=m32)
                        T.simd.vsts(x_ub[0, off], T.simd.vcvt(yf, "bfloat16", mask=m16, part=0), mask=m32, dist="PK_B32", extent=64)
                T.copy(x_ub[0, :128], Y[m, :128])
                T.ascend_pipe_barrier("PIPE_ALL")
    return main
"""

_MLA_TIER0_DSL = """
import tilelang.ascend.language as T

def make_mla_kernel():
    b_dim = 1
    s_dim = 32
    s_kv = 128
    n_q = 16
    n_kv = 1
    d_nope = 128
    d_rope = 64
    q_total_rows = 512
    bm = 32
    bn = 64
    rows_per_sid = 16
    num_cores = 1

    @T.prim_func
    def main(
        Q_nope: T.Tensor((q_total_rows, d_nope), "bfloat16"),
        Q_rope: T.Tensor((q_total_rows, d_rope), "bfloat16"),
        K_nope: T.Tensor((b_dim, s_kv, n_kv, d_nope), "bfloat16"),
        K_rope: T.Tensor((b_dim, s_kv, n_kv, d_rope), "bfloat16"),
        V: T.Tensor((b_dim, s_kv, n_kv, d_nope), "bfloat16"),
        Out: T.Tensor((q_total_rows, d_nope), "bfloat16"),
        scale_val: T.float32,
    ):
        with T.MixedKernel(num_cores, sids=2) as (bx, sid):
            q_nope_l1 = T.alloc_l1((bm, d_nope), "bfloat16")
            q_rope_l1 = T.alloc_l1((bm, d_rope), "bfloat16")
            k_nope_l1 = T.alloc_l1((bn, d_nope), "bfloat16")
            k_rope_l1 = T.alloc_l1((bn, d_rope), "bfloat16")
            p_l1 = T.alloc_l1((bm, bn), "bfloat16")
            v0_l1 = T.alloc_l1((d_nope, bn), "bfloat16")
            s_l0c = T.alloc_l0c((bm, bn), "float32")
            o_l0c = T.alloc_l0c((bm, d_nope), "float32")
            q_nope_ub = T.alloc_shared((rows_per_sid, d_nope), "bfloat16")
            q_rope_ub = T.alloc_shared((rows_per_sid, d_rope), "bfloat16")
            S_ub = T.alloc_shared((rows_per_sid, bn), "float32")
            P_ub = T.alloc_shared((rows_per_sid, bn), "bfloat16")
            O_tmp_ub = T.alloc_shared((rows_per_sid, d_nope), "float32")
            O_ub = T.alloc_shared((rows_per_sid, d_nope), "float32")
            out_ub = T.alloc_shared((rows_per_sid, d_nope), "bfloat16")
            m_ub = T.alloc_shared((rows_per_sid, 8), "float32")
            l_ub = T.alloc_shared((rows_per_sid, 8), "float32")
            alpha_ub = T.alloc_shared((rows_per_sid, 8), "float32")
            T.annotate_buffer_versions(
                {
                    q_nope_l1: 1,
                    q_rope_l1: 1,
                    k_nope_l1: 1,
                    k_rope_l1: 1,
                    v0_l1: 1,
                    p_l1: 1,
                    s_l0c: 1,
                    o_l0c: 1,
                    S_ub: 1,
                    P_ub: 1,
                    O_tmp_ub: 1,
                    O_ub: 1,
                    out_ub: 1,
                    m_ub: 1,
                    l_ub: 1,
                    alpha_ub: 1,
                    q_nope_ub: 1,
                    q_rope_ub: 1,
                }
            )
            q_rows = s_dim * n_q
            m_tiles = q_rows // bm
            base_tiles = m_tiles // num_cores
            rem_tiles = m_tiles % num_cores
            t_start = bx * base_tiles + T.min(bx, rem_tiles)
            t_count = base_tiles + T.if_then_else(bx < rem_tiles, 1, 0)
            for local_t in T.serial(t_count):
                mt = t_start + local_t
                q_base = mt * bm
                for r in T.serial(rows_per_sid):
                    q_row = q_base + sid * rows_per_sid + r
                    s_i = q_row // n_q
                    h_loc = q_row % n_q
                    gm_r = s_i * n_q + h_loc
                    T.copy(Q_nope[gm_r, 0:d_nope], q_nope_ub[r, 0:d_nope])
                    T.copy(Q_rope[gm_r, 0:d_rope], q_rope_ub[r, 0:d_rope])
                T.dual_copy(q_nope_ub, q_nope_l1)
                T.dual_copy(q_rope_ub, q_rope_l1)
                with T.SimdVF():
                    m32 = T.simd.pset(32, "PAT_ALL")
                    vn = T.simd.vdup(T.float32(-1e38), "float32", mask=m32)
                    vz = T.simd.vdup(T.float32(0.0), "float32", mask=m32)
                    for r in T.serial(rows_per_sid):
                        T.simd.vsts(m_ub[r, 0], vn, mask=m32, dist="ONEPT_B32")
                        T.simd.vsts(l_ub[r, 0], vz, mask=m32, dist="ONEPT_B32")
                        for c in T.serial(2):
                            T.simd.vsts(O_ub[r, c * 64], vz, mask=m32, dist="NORM_B32")
                for kv in T.Pipelined(s_kv // bn, num_stages=1):
                    T.copy(K_nope[0, kv * bn : kv * bn + bn, 0, 0:d_nope], k_nope_l1)
                    T.copy(K_rope[0, kv * bn : kv * bn + bn, 0, 0:d_rope], k_rope_l1)
                    T.copy(V[0, kv * bn : kv * bn + bn, 0, 0:d_nope], v0_l1, transpose=True)
                    T.gemm(q_nope_l1, k_nope_l1, s_l0c, transpose_B=True, clear_accum=True)
                    T.gemm(q_rope_l1, k_rope_l1, s_l0c, transpose_B=True, clear_accum=False)
                    T.dual_copy(s_l0c, S_ub)
                    with T.SimdVF():
                        m32 = T.simd.pset(32, "PAT_ALL")
                        m16 = T.simd.pset(16, "PAT_ALL")
                        for r in T.serial(rows_per_sid):
                            sv = T.simd.vmuls(T.simd.vld(S_ub[r, 0], dist="NORM_B32"), scale_val, mask=m32)
                            mt_v = T.simd.vdupv(T.simd.vcmax(sv, mask=m32), mask=m32, pos="POS_LOWEST")
                            m_old = T.simd.vld(m_ub[r, 0], dist="BRC_B32")
                            l_old = T.simd.vld(l_ub[r, 0], dist="BRC_B32")
                            m_new = T.simd.vmax(m_old, mt_v, mask=m32)
                            av = T.simd.vexp(T.simd.vsub(m_old, m_new, mask=m32), mask=m32)
                            pf = T.simd.vexp(T.simd.vsub(sv, m_new, mask=m32), mask=m32)
                            rs = T.simd.vdupv(T.simd.vcadd(pf, mask=m32), mask=m32, pos="POS_LOWEST")
                            l_new = T.simd.vadd(T.simd.vmul(l_old, av, mask=m32), rs, mask=m32)
                            T.simd.vsts(m_ub[r, 0], m_new, mask=m32, dist="ONEPT_B32")
                            T.simd.vsts(l_ub[r, 0], l_new, mask=m32, dist="ONEPT_B32")
                            T.simd.vsts(alpha_ub[r, 0], av, mask=m32, dist="ONEPT_B32")
                            T.simd.vsts(P_ub[r, 0], T.simd.vcvt(pf, "bfloat16", mask=m16, part=0), mask=m32, dist="PK_B32", extent=64)
                    T.dual_copy(P_ub, p_l1)
                    T.gemm(p_l1, v0_l1, o_l0c, transpose_B=True, clear_accum=True)
                    T.dual_copy(o_l0c, O_tmp_ub)
                    with T.SimdVF():
                        m32 = T.simd.pset(32, "PAT_ALL")
                        for r in T.serial(rows_per_sid):
                            av = T.simd.vld(alpha_ub[r, 0], dist="BRC_B32")
                            for c in T.serial(2):
                                off = c * 64
                                oo = T.simd.vld(O_ub[r, off], dist="NORM_B32")
                                ot = T.simd.vld(O_tmp_ub[r, off], dist="NORM_B32")
                                T.simd.vsts(O_ub[r, off], T.simd.vadd(T.simd.vmul(oo, av, mask=m32), ot, mask=m32), mask=m32, dist="NORM_B32")
                with T.SimdVF():
                    m32 = T.simd.pset(32, "PAT_ALL")
                    m16 = T.simd.pset(16, "PAT_ALL")
                    v_one = T.simd.vdup(T.float32(1.0), "float32", mask=m32)
                    for r in T.serial(rows_per_sid):
                        inv_l = T.simd.vdiv(v_one, T.simd.vld(l_ub[r, 0], dist="BRC_B32"), mask=m32)
                        for c in T.serial(2):
                            off = c * 64
                            o_f = T.simd.vmul(T.simd.vld(O_ub[r, off], dist="NORM_B32"), inv_l, mask=m32)
                            T.simd.vsts(out_ub[r, off], T.simd.vcvt(o_f, "bfloat16", mask=m16, part=0), mask=m32, dist="PK_B32", extent=64)
                for r in T.serial(rows_per_sid):
                    q_row = q_base + sid * rows_per_sid + r
                    s_i = q_row // n_q
                    h_loc = q_row % n_q
                    gm_r = s_i * n_q + h_loc
                    T.copy(out_ub[r, 0:d_nope], Out[gm_r, 0:d_nope])
    return main
"""


def _require_live_npu_and_llm_backend() -> KernelLLMBackend:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Ascend NPU device is required for live e2e evolution test.")

    env_hint = os.environ.get("TILEFOUNDRY_NPU_ENV_FILE", "").strip()
    overrides = parse_simple_env_file(Path(env_hint).expanduser()) if env_hint else {}
    sources = (overrides, os.environ)

    def configured_value(*keys: str, default: str = "") -> str:
        return next(
            (value for source in sources for key in keys if (value := source.get(key, "").strip())),
            default,
        )

    kind = configured_value("TILEFOUNDRY_NPU_LLM_BACKEND", default="agy_cli").lower()
    bin_keys = ("TILEFOUNDRY_NPU_LLM_CLI_BIN",)
    if kind in ("agy_cli", "agy"):
        bin_keys += ("TILEFOUNDRY_NPU_AGY_BIN",)
    configured_bin = configured_value(*bin_keys)
    if kind in ("agy_cli", "agy") and configured_bin and not shutil.which(configured_bin):
        raise FileNotFoundError("The configured LLM CLI binary is missing or not executable.")

    try:
        backend = create_llm_backend_from_env(overrides=overrides)
    except FileNotFoundError:
        if not configured_bin and kind in ("agy_cli", "agy", "claude_cli", "codex_cli"):
            pytest.skip("The default LLM CLI executable is unavailable.")
        raise
    if isinstance(backend, MockLLMBackend):
        pytest.skip("Mock LLM backend configured for live NPU evolution.")
    return backend


@pytest.mark.parametrize(
    ("config", "expected_error"),
    [
        ({"BACKEND": "openai_compat", "EXPLOIT_MODEL": "test-model"}, None),
        ({"BACKEND": "openai_compat", "API_BASE": "invalid-url"}, ValueError),
        ({"BACKEND": "mock"}, pytest.skip.Exception),
        ({"BACKEND": "unsupported"}, ValueError),
        ({"BACKEND": "codex_cli"}, pytest.skip.Exception),
        ({"BACKEND": "agy_cli", "CLI_BIN": "missing-llm-bin"}, FileNotFoundError),
    ],
)
def test_configured_backend_readiness_offline(tmp_path: Path, monkeypatch, config, expected_error):
    """Check provider selection and skip behavior with simulated NPU availability."""
    monkeypatch.setattr(torch, "npu", SimpleNamespace(is_available=lambda: True), raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    for key in os.environ.copy():
        if key.startswith(("TILEFOUNDRY_NPU_LLM_", "TILEFOUNDRY_NPU_AGY_")):
            monkeypatch.delenv(key)
    env_file = tmp_path / "env.sh"
    env_file.write_text(
        "".join(f"TILEFOUNDRY_NPU_LLM_{key}={value}\n" for key, value in config.items()),
        encoding="utf-8",
    )
    monkeypatch.setenv("TILEFOUNDRY_NPU_ENV_FILE", str(env_file))
    if expected_error is None:
        assert isinstance(_require_live_npu_and_llm_backend(), OpenAICompatHTTPBackend)
    else:
        with pytest.raises(expected_error):
            _require_live_npu_and_llm_backend()


def _ar_npu_device_runner(_dsl_source: str, stage_a) -> StageCResult:
    import tilelang  # noqa: PLC0415

    from tests.ascend.benchmarks.attention_residual.golden import (  # noqa: PLC0415
        attention_residual as ar_golden,
    )
    from tilefoundry_ascend.agent.probes import run_msprof_detail  # noqa: PLC0415

    assert stage_a.prim_func is not None and stage_a.ledger is not None
    torch.manual_seed(7)
    n_depth, m_rows, d_dim = 4, 64, 128
    v_stack = torch.randn(n_depth, 1, m_rows, d_dim, dtype=torch.bfloat16)
    x = torch.randn(1, m_rows, d_dim, dtype=torch.bfloat16)
    w = torch.randn(d_dim, dtype=torch.bfloat16)
    gamma = torch.rand(d_dim, dtype=torch.bfloat16) + 0.5
    oracle = ar_golden(v_stack, x, w, gamma, alpha=0.5, epsilon=1e-6)

    v_flat = v_stack.view(n_depth * m_rows, d_dim).npu()
    x_flat = x.view(m_rows, d_dim).npu()
    w_npu = w.npu()
    g_npu = gamma.npu()

    kernel = tilelang.compile(stage_a.prim_func, target="ascend", out_idx=[4])
    out_npu = kernel(v_flat, x_flat, w_npu, g_npu, 0.5, 1e-6)
    torch.npu.synchronize()
    torch.testing.assert_close(
        out_npu.view(1, m_rows, d_dim).cpu().float(),
        oracle.float(),
        rtol=2e-2,
        atol=2e-2,
    )
    prof = run_msprof_detail(
        lambda: kernel(v_flat, x_flat, w_npu, g_npu, 0.5, 1e-6),
        n_warmup=2,
        n_repeat=5,
    )
    tel = {
        "dur_us": float(prof.dur_us),
        "aic_mac_ratio": float(prof.aic_mad),
        "aiv_vec_ratio": float(prof.aiv_vec),
        "mte2_wait_ratio": float(prof.aiv_mte2),
        "sync_stall_ratio": 0.0,
    }
    return StageCResult(
        passed=True,
        accuracy_passed=True,
        measured_latency_us=float(prof.dur_us),
        residual=compute_static_dynamic_residual(stage_a.ledger, tel),
        telemetry=tel,
    )


def _mla_npu_device_runner(_dsl_source: str, stage_a) -> StageCResult:
    import tilelang  # noqa: PLC0415

    from tests.ascend.benchmarks.mla.golden import mla as mla_golden  # noqa: PLC0415
    from tilefoundry_ascend.agent.probes import run_msprof_detail  # noqa: PLC0415

    assert stage_a.prim_func is not None and stage_a.ledger is not None
    torch.manual_seed(11)
    b_dim, s_dim, s_kv, n_q, n_kv, d_nope, d_rope = 1, 32, 128, 16, 1, 128, 64
    scale = 1.0 / ((d_nope + d_rope) ** 0.5)
    qn = torch.randn(b_dim, s_dim, n_q, d_nope, dtype=torch.bfloat16)
    qr = torch.randn(b_dim, s_dim, n_q, d_rope, dtype=torch.bfloat16)
    kn = torch.randn(b_dim, s_kv, n_kv, d_nope, dtype=torch.bfloat16)
    kr = torch.randn(b_dim, s_kv, n_kv, d_rope, dtype=torch.bfloat16)
    v = kn.clone()
    oracle = mla_golden(qn, qr, kn, kr, v, numKVHeads=n_kv, scaleValue=scale, inputLayout="BSND", is_causal=False)

    qn_f = qn.view(s_dim * n_q, d_nope).npu()
    qr_f = qr.view(s_dim * n_q, d_rope).npu()
    kn_n = kn.npu()
    kr_n = kr.npu()
    v_n = v.npu()

    kernel = tilelang.compile(stage_a.prim_func, target="ascend", out_idx=[5])
    out_npu = kernel(qn_f, qr_f, kn_n, kr_n, v_n, scale)
    torch.npu.synchronize()
    torch.testing.assert_close(
        out_npu.view(b_dim, s_dim, n_q, d_nope).cpu().float(),
        oracle.float(),
        rtol=2e-2,
        atol=2e-2,
    )
    prof = run_msprof_detail(
        lambda: kernel(qn_f, qr_f, kn_n, kr_n, v_n, scale),
        n_warmup=2,
        n_repeat=5,
    )
    tel = {
        "dur_us": float(prof.dur_us),
        "aic_mac_ratio": float(prof.aic_mad),
        "aiv_vec_ratio": float(prof.aiv_vec),
        "mte2_wait_ratio": float(prof.aiv_mte2),
        "sync_stall_ratio": 0.0,
    }
    return StageCResult(
        passed=True,
        accuracy_passed=True,
        measured_latency_us=float(prof.dur_us),
        residual=compute_static_dynamic_residual(stage_a.ledger, tel),
        telemetry=tel,
    )


def _safe_device_wrap(fn):
    def _wrapped(dsl_source: str, stage_a) -> StageCResult:
        try:
            return fn(dsl_source, stage_a)
        except Exception as exc:
            return StageCResult(
                passed=False,
                accuracy_passed=False,
                measured_latency_us=0.0,
                residual=None,
                telemetry={},
                error_message=str(exc)[-400:],
            )

    return _wrapped


def _log_progress(msg: str) -> None:
    out_dir = os.environ.get("TILEFOUNDRY_E2E_REPORT_DIR", "").strip()
    if out_dir:
        dest = Path(out_dir)
        dest.mkdir(parents=True, exist_ok=True)
        with (dest / "wp5_debug.log").open("a", encoding="utf-8") as fh:
            fh.write(msg + "\n")


def _save_trajectory(name: str, traj) -> None:
    out_dir = os.environ.get("TILEFOUNDRY_E2E_REPORT_DIR", "").strip()
    if out_dir:
        dest = Path(out_dir)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / f"{name}.json").write_text(traj.to_json(), encoding="utf-8")


def test_live_e2e_agent_evolution_on_npu() -> None:
    """Drive dual-loop evolution from AR and MLA HIR on a live Ascend 950 NPU."""
    backend = _require_live_npu_and_llm_backend()
    ar_hir = _AR_MODEL_PATH.read_text(encoding="utf-8")
    mla_hir = _MLA_MODEL_PATH.read_text(encoding="utf-8")
    selected = os.environ.get("TILEFOUNDRY_E2E_PROBLEM", "all").strip().lower()
    num_rounds = max(1, int(os.environ.get("TILEFOUNDRY_E2E_ROUNDS", "2")))

    ar_funnel = GateFunnel(device_runner=_safe_device_wrap(_ar_npu_device_runner))
    mla_funnel = GateFunnel(device_runner=_safe_device_wrap(_mla_npu_device_runner))

    try:
        async def _run_selected():
            results = {}
            if selected in ("all", "ar", "attention_residual"):
                _log_progress("Evaluating AR baseline")
                ar_base = ar_funnel.evaluate(_AR_TIER0_DSL, factory_name="make_ar_kernel")
                assert ar_base.passed, (
                    f"AR baseline failed ({ar_base.failed_stage}): {ar_base.error_message[-400:]}"
                )
                ar_traj = await run_e2e_evolution(
                    problem_name="attention_residual",
                    hir_source=ar_hir,
                    shape_contract="make_ar_kernel() returning fixed (256, 128) bfloat16 AttentionResidual",
                    backend=backend,
                    funnel=ar_funnel,
                    tracker=BayesianRuleTracker.from_seed_yaml(),
                    initial_dsl_source=_AR_TIER0_DSL,
                    factory_name="make_ar_kernel",
                    num_rounds=num_rounds,
                    k_rule_workers=2,
                )
                _save_trajectory("wp5_ar_trajectory", ar_traj)
                _log_progress(
                    f"AR evolution done: {ar_traj.baseline_latency_us:.3f} -> {ar_traj.best_latency_us:.3f} us"
                )
                results["ar"] = ar_traj

            if selected in ("all", "mla"):
                _log_progress("Evaluating MLA baseline")
                mla_base = mla_funnel.evaluate(_MLA_TIER0_DSL, factory_name="make_mla_kernel")
                assert mla_base.passed, (
                    f"MLA baseline failed ({mla_base.failed_stage}): {mla_base.error_message[-400:]}"
                )
                mla_traj = await run_e2e_evolution(
                    problem_name="mla",
                    hir_source=mla_hir,
                    shape_contract="make_mla_kernel() returning fixed (512, 128) bfloat16 MLA",
                    backend=backend,
                    funnel=mla_funnel,
                    tracker=BayesianRuleTracker.from_seed_yaml(),
                    initial_dsl_source=_MLA_TIER0_DSL,
                    factory_name="make_mla_kernel",
                    num_rounds=num_rounds,
                    k_rule_workers=2,
                )
                _save_trajectory("wp5_mla_trajectory", mla_traj)
                _log_progress(
                    f"MLA evolution done: {mla_traj.baseline_latency_us:.3f} -> {mla_traj.best_latency_us:.3f} us"
                )
                results["mla"] = mla_traj
            return results

        results = asyncio.run(_run_selected())
        assert results
        for traj in results.values():
            assert traj.baseline_latency_us > 0.0
            assert traj.best_latency_us <= traj.baseline_latency_us
            workers = [worker for round_ in traj.rounds for worker in round_.workers]
            assert workers, f"{traj.problem_name} evolution made no generation attempts"
            for kind in {worker.worker_kind for worker in workers}:
                attempts = [worker for worker in workers if worker.worker_kind == kind]
                assert any(worker.failed_stage != "llm_transport" for worker in attempts), (
                    f"{traj.problem_name} {kind} provider failed: {attempts[0].hypothesis}"
                )
    except Exception as exc:
        _log_progress(f"EXCEPTION: {type(exc).__name__}: {str(exc)[-800:]}")
        raise AssertionError(f"WP5 NPU evolution error: {type(exc).__name__}: {str(exc)[-800:]}") from None
