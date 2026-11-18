"""TileLang Ascend950PR Persistent-State SimdVF kernel for KimiDeltaAttention."""

import contextlib
import math

import tilelang
import tilelang.ascend.language as T
import torch

from tilefoundry_ascend.ascend.target import PURE_SIMDVF_UBUF_FLAGS

_KERNEL_CACHE: dict[tuple[object, ...], object] = {}


def _ascend_target_ctx():
    try:
        from tilelang import tvm
        from tilelang.backend.module import create_backend_context

        if tvm.target.Target.current(allow_none=True) is None:
            return create_backend_context("ascend").target
    except Exception:
        pass
    return contextlib.nullcontext()


def _select_kimi_delta_attention_config(
    dk: int = 128,
    dv: int = 128,
    tl_dtype: str = "bfloat16",
    *,
    t_dim: int = 64,
    use_qk_l2norm: bool = True,
    schedule_mode: str = "tuned",
) -> dict[str, object]:
    """Select persistent (B, H) UB-resident state regime and chunking for KDA."""
    dk_alloc = ((max(1, int(dk)) + 63) // 64) * 64
    dv_alloc = ((max(1, int(dv)) + 63) // 64) * 64
    # Persistent FP32 state [dk_alloc, dv_alloc] in UB (128x128*4B = 64KB < 256KB)
    state_bytes = dk_alloc * dv_alloc * 4
    if state_bytes <= 192 * 1024:
        regime = "persistent_ub_state_resident"
    else:
        regime = "chunked_dv_tile_state"
    chunk_t = min(max(1, int(t_dim)), 64 if schedule_mode == "tuned" else 16)
    return {
        "regime": regime,
        "dk": int(dk),
        "dv": int(dv),
        "tl_dtype": tl_dtype,
        "use_qk_l2norm": bool(use_qk_l2norm),
        "chunk_t": chunk_t,
        "pass_configs": {},
        "compile_flags": list(PURE_SIMDVF_UBUF_FLAGS) if schedule_mode == "tuned" else [],
    }


def _build_kimi_delta_attention_prim_func(
    dk: int = 128,
    dv: int = 128,
    tl_dtype: str = "bfloat16",
    use_qk_l2norm: bool = True,
    schedule_mode: str = "tuned",
    **_extra_kwargs,
):
    """Build Ascend 950 DV-Sharded T-Batched Register-Resident @T.prim_func for KimiDeltaAttention."""
    if dk <= 0 or dv <= 0:
        raise ValueError(f"dk and dv must be positive, got ({dk}, {dv})")
    if tl_dtype not in ("float16", "bfloat16", "float32"):
        raise ValueError(f"unsupported dtype {tl_dtype!r}")

    num_cores = 56
    dk_alloc = ((dk + 63) // 64) * 64
    dk_chunks = dk // 64
    dk_rem = dk % 64
    dv_tiles = (dv + 63) // 64
    dv_last = dv - (dv_tiles - 1) * 64
    t_blk = 16 if schedule_mode == "tuned" else 8

    b_dim = T.dynamic("b_dim")
    t_dim = T.dynamic("t_dim")
    h_dim = T.dynamic("h_dim")

    with _ascend_target_ctx():

        @T.prim_func
        def main(
            Q: T.Tensor((b_dim, t_dim, h_dim, dk), tl_dtype),
            K: T.Tensor((b_dim, t_dim, h_dim, dk), tl_dtype),
            V: T.Tensor((b_dim, t_dim, h_dim, dv), tl_dtype),
            G: T.Tensor((b_dim, t_dim, h_dim, dk), "float32"),
            Beta: T.Tensor((b_dim, t_dim, h_dim), "float32"),
            InitialState: T.Tensor((b_dim, h_dim, dk, dv), "float32"),
            Y: T.Tensor((b_dim, t_dim, h_dim, dv), tl_dtype),
            FinalState: T.Tensor((b_dim, h_dim, dk, dv), "float32"),
            scale_val: T.float32,
        ):
            with T.Kernel(num_cores) as cid:
                # Shard across (B, H, dv_tile) so state_ub[dk_alloc, 64] fits <=32KB in UB
                # and pred/y accumulators stay in SIMD vector registers across all dk rows.
                state_ub = T.alloc_shared((dk_alloc, 64), "float32")
                q_tile_ub = T.alloc_shared((t_blk, dk_alloc), tl_dtype)
                k_tile_ub = T.alloc_shared((t_blk, dk_alloc), tl_dtype)
                v_tile_ub = T.alloc_shared((t_blk, 64), tl_dtype)
                g_tile_ub = T.alloc_shared((t_blk, dk_alloc), "float32")
                beta_tile_ub = T.alloc_shared((t_blk,), "float32")
                out_tile_ub = T.alloc_shared((t_blk, 64), tl_dtype)
                qf_ub = T.alloc_shared((dk_alloc,), "float32")
                kf_ub = T.alloc_shared((dk_alloc,), "float32")
                g1_ub = T.alloc_shared((dk_alloc,), "float32")
                T.annotate_buffer_versions(
                    {
                        state_ub: 1,
                        q_tile_ub: 1,
                        k_tile_ub: 1,
                        v_tile_ub: 1,
                        g_tile_ub: 1,
                        beta_tile_ub: 1,
                        out_tile_ub: 1,
                        qf_ub: 1,
                        kf_ub: 1,
                        g1_ub: 1,
                    }
                )
                if schedule_mode == "tuned":
                    T.annotate_unlimit_memory("shared")

                total_tasks = b_dim * h_dim * dv_tiles
                base_tasks = total_tasks // num_cores
                rem_tasks = total_tasks % num_cores
                t_start = cid * base_tasks + T.min(cid, rem_tasks)
                t_count = base_tasks + T.if_then_else(cid < rem_tasks, 1, 0)
                t_blocks = (t_dim + t_blk - 1) // t_blk

                for ltask in T.serial(t_count):
                    tid = t_start + ltask
                    bh = tid // dv_tiles
                    dv_t = tid % dv_tiles
                    b_idx = bh // h_dim
                    h_idx = bh % h_dim
                    dv0 = dv_t * 64
                    dv_cur = T.min(64, dv - dv0)

                    for r in T.serial(dk):
                        T.copy(InitialState[b_idx, h_idx, r, dv0 : dv0 + dv_cur], state_ub[r, :dv_cur])

                    for tb in T.serial(t_blocks):
                        t0 = tb * t_blk
                        t_cur = T.min(t_blk, t_dim - t0)
                        for lt in T.serial(t_cur):
                            T.copy(Q[b_idx, t0 + lt, h_idx, :dk], q_tile_ub[lt, :dk])
                            T.copy(K[b_idx, t0 + lt, h_idx, :dk], k_tile_ub[lt, :dk])
                            T.copy(V[b_idx, t0 + lt, h_idx, dv0 : dv0 + dv_cur], v_tile_ub[lt, :dv_cur])
                            T.copy(G[b_idx, t0 + lt, h_idx, :dk], g_tile_ub[lt, :dk])
                            beta_tile_ub[lt] = Beta[b_idx, t0 + lt, h_idx]
                        T.ascend_pipe_barrier("PIPE_ALL")

                        with T.SimdVF():
                            m32 = T.simd.pset(32, "PAT_ALL")
                            m16 = T.simd.pset(16, "PAT_ALL")
                            mv32 = m32 if dv_last == 64 else T.simd.update_mask(dv_cur, width=32)
                            mv16 = m16 if dv_last == 64 else T.simd.update_mask(dv_cur * 2, width=16)
                            v_zero = T.simd.vdup(T.float32(0.0), "float32", mask=m32)
                            v_one = T.simd.vdup(T.float32(1.0), "float32", mask=m32)
                            v_eps = T.simd.vdup(T.float32(1.0e-6), "float32", mask=m32)
                            q_sq = T.simd.alloc_local((1,), "float32")
                            k_sq = T.simd.alloc_local((1,), "float32")
                            pred_reg = T.simd.alloc_local((1,), "float32")
                            y_reg = T.simd.alloc_local((1,), "float32")
                            vf_reg = T.simd.alloc_local((1,), "float32")

                            for lt in T.serial(t_cur):
                                beta_t = beta_tile_ub[lt]
                                q_sq[0] = v_zero
                                k_sq[0] = v_zero

                                for c in T.serial(dk_chunks):
                                    off = c * 64
                                    if tl_dtype == "float32":
                                        qv = T.simd.vld(q_tile_ub[lt, off], dist="NORM_B32")
                                        kv = T.simd.vld(k_tile_ub[lt, off], dist="NORM_B32")
                                    else:
                                        qv = T.simd.vcvt(T.simd.vld(q_tile_ub[lt, off], dist="US_B16"), "float32", mask=m16, part=0)
                                        kv = T.simd.vcvt(T.simd.vld(k_tile_ub[lt, off], dist="US_B16"), "float32", mask=m16, part=0)
                                    T.simd.vsts(qf_ub[off], qv, mask=m32, dist="NORM_B32")
                                    T.simd.vsts(kf_ub[off], kv, mask=m32, dist="NORM_B32")
                                    T.simd.vmula(q_sq[0], qv, qv, mask=m32)
                                    T.simd.vmula(k_sq[0], kv, kv, mask=m32)
                                if dk_rem > 0:
                                    off = dk_chunks * 64
                                    m32_t = T.simd.update_mask(dk_rem, width=32)
                                    if tl_dtype == "float32":
                                        qv_t = T.simd.vld(q_tile_ub[lt, off], dist="NORM_B32")
                                        kv_t = T.simd.vld(k_tile_ub[lt, off], dist="NORM_B32")
                                    else:
                                        m16_t = T.simd.update_mask(dk_rem * 2, width=16)
                                        qv_t = T.simd.vcvt(T.simd.vld(q_tile_ub[lt, off], dist="US_B16"), "float32", mask=m16_t, part=0)
                                        kv_t = T.simd.vcvt(T.simd.vld(k_tile_ub[lt, off], dist="US_B16"), "float32", mask=m16_t, part=0)
                                    T.simd.vsts(qf_ub[off], qv_t, mask=m32_t, dist="NORM_B32")
                                    T.simd.vsts(kf_ub[off], kv_t, mask=m32_t, dist="NORM_B32")
                                    q_sq[0] = T.simd.vadd(q_sq[0], T.simd.vmul(qv_t, qv_t, mask=m32_t), mask=m32)
                                    k_sq[0] = T.simd.vadd(k_sq[0], T.simd.vmul(kv_t, kv_t, mask=m32_t), mask=m32)

                                if use_qk_l2norm:
                                    q_inv = T.simd.vdiv(
                                        v_one,
                                        T.simd.vsqrt(
                                            T.simd.vadd(
                                                T.simd.vdupv(T.simd.vcadd(q_sq[0], mask=m32), mask=m32, pos="POS_LOWEST"),
                                                v_eps,
                                                mask=m32,
                                            ),
                                            mask=m32,
                                            precision="ftz_false",
                                        ),
                                        mask=m32,
                                        precision="exact",
                                    )
                                    k_inv = T.simd.vdiv(
                                        v_one,
                                        T.simd.vsqrt(
                                            T.simd.vadd(
                                                T.simd.vdupv(T.simd.vcadd(k_sq[0], mask=m32), mask=m32, pos="POS_LOWEST"),
                                                v_eps,
                                                mask=m32,
                                            ),
                                            mask=m32,
                                            precision="ftz_false",
                                        ),
                                        mask=m32,
                                        precision="exact",
                                    )
                                else:
                                    q_inv = v_one
                                    k_inv = v_one
                                q_scale_v = T.simd.vmuls(q_inv, scale_val, mask=m32)

                                for c in T.serial(dk_chunks):
                                    off = c * 64
                                    T.simd.vsts(
                                        qf_ub[off],
                                        T.simd.vmul(T.simd.vld(qf_ub[off], dist="NORM_B32"), q_scale_v, mask=m32),
                                        mask=m32,
                                        dist="NORM_B32",
                                    )
                                    T.simd.vsts(
                                        kf_ub[off],
                                        T.simd.vmul(T.simd.vld(kf_ub[off], dist="NORM_B32"), k_inv, mask=m32),
                                        mask=m32,
                                        dist="NORM_B32",
                                    )
                                    T.simd.vsts(
                                        g1_ub[off],
                                        T.simd.vexp(T.simd.vld(g_tile_ub[lt, off], dist="NORM_B32"), mask=m32, precision="ftz_false"),
                                        mask=m32,
                                        dist="NORM_B32",
                                    )
                                if dk_rem > 0:
                                    off = dk_chunks * 64
                                    m32_t = T.simd.update_mask(dk_rem, width=32)
                                    T.simd.vsts(
                                        qf_ub[off],
                                        T.simd.vmul(T.simd.vld(qf_ub[off], dist="NORM_B32"), q_scale_v, mask=m32_t),
                                        mask=m32_t,
                                        dist="NORM_B32",
                                    )
                                    T.simd.vsts(
                                        kf_ub[off],
                                        T.simd.vmul(T.simd.vld(kf_ub[off], dist="NORM_B32"), k_inv, mask=m32_t),
                                        mask=m32_t,
                                        dist="NORM_B32",
                                    )
                                    T.simd.vsts(
                                        g1_ub[off],
                                        T.simd.vexp(T.simd.vld(g_tile_ub[lt, off], dist="NORM_B32"), mask=m32_t, precision="ftz_false"),
                                        mask=m32_t,
                                        dist="NORM_B32",
                                    )

                                if tl_dtype == "float32":
                                    vf_reg[0] = T.simd.vld(v_tile_ub[lt, 0], dist="NORM_B32")
                                else:
                                    vf_reg[0] = T.simd.vcvt(T.simd.vld(v_tile_ub[lt, 0], dist="US_B16"), "float32", mask=mv16, part=0)
                                pred_reg[0] = v_zero
                                y_reg[0] = v_zero
                                T.simd.mem_bar("VST_VLD")

                                # Loop 1: Hbar = exp(g) * Hprev, pred_reg = k^T @ Hbar (100% in register)
                                for r in T.serial(dk):
                                    decay_v = T.simd.vld(g1_ub[r], dist="BRC_B32")
                                    k_norm_r = T.simd.vld(kf_ub[r], dist="BRC_B32")
                                    h_prev = T.simd.vld(state_ub[r, 0], dist="NORM_B32")
                                    h_bar = T.simd.vmul(h_prev, decay_v, mask=mv32)
                                    T.simd.vsts(state_ub[r, 0], h_bar, mask=mv32, dist="NORM_B32")
                                    pred_reg[0] = T.simd.vadd(pred_reg[0], T.simd.vmul(h_bar, k_norm_r, mask=mv32), mask=mv32)

                                # corr = (v - pred) * beta_t kept in pred_reg[0], then Loop 2: Hnew + y_reg
                                corr_v = T.simd.vmuls(T.simd.vsub(vf_reg[0], pred_reg[0], mask=mv32), beta_t, mask=mv32)
                                T.simd.mem_bar("VST_VLD")

                                for r in T.serial(dk):
                                    k_r = T.simd.vld(kf_ub[r], dist="BRC_B32")
                                    q_r = T.simd.vld(qf_ub[r], dist="BRC_B32")
                                    h_new = T.simd.vadd(
                                        T.simd.vld(state_ub[r, 0], dist="NORM_B32"),
                                        T.simd.vmul(k_r, corr_v, mask=mv32),
                                        mask=mv32,
                                    )
                                    T.simd.vsts(state_ub[r, 0], h_new, mask=mv32, dist="NORM_B32")
                                    y_reg[0] = T.simd.vadd(y_reg[0], T.simd.vmul(h_new, q_r, mask=mv32), mask=mv32)
                                T.simd.mem_bar("VST_VLD")

                                if tl_dtype == "float32":
                                    T.simd.vsts(out_tile_ub[lt, 0], y_reg[0], mask=mv32, dist="NORM_B32")
                                else:
                                    y16 = T.simd.vcvt(y_reg[0], tl_dtype, mask=mv16, part=0)
                                    T.simd.vsts(out_tile_ub[lt, 0], y16, mask=mv32, dist="PK_B32", extent=64)

                        T.ascend_pipe_barrier("PIPE_ALL")
                        for lt in T.serial(t_cur):
                            T.copy(out_tile_ub[lt, :dv_cur], Y[b_idx, t0 + lt, h_idx, dv0 : dv0 + dv_cur])
                        T.ascend_pipe_barrier("PIPE_ALL")

                    for r in T.serial(dk):
                        T.copy(state_ub[r, :dv_cur], FinalState[b_idx, h_idx, r, dv0 : dv0 + dv_cur])
                    T.ascend_pipe_barrier("PIPE_ALL")

    return main


def _get_kernel(dk: int, dv: int, tl_dtype: str, use_qk_l2norm: bool, t_dim: int, schedule_mode: str):
    cfg = _select_kimi_delta_attention_config(
        dk=dk,
        dv=dv,
        tl_dtype=tl_dtype,
        t_dim=t_dim,
        use_qk_l2norm=use_qk_l2norm,
        schedule_mode=schedule_mode,
    )
    key = (dk, dv, tl_dtype, bool(use_qk_l2norm), schedule_mode)
    compiled = _KERNEL_CACHE.get(key)
    if compiled is None:
        with _ascend_target_ctx():
            prog = _build_kimi_delta_attention_prim_func(
                dk=dk,
                dv=dv,
                tl_dtype=tl_dtype,
                use_qk_l2norm=use_qk_l2norm,
                schedule_mode=schedule_mode,
            )
        compiled = tilelang.compile(
            prog,
            target="ascend",
            out_idx=[6, 7],
            pass_configs=cfg.get("pass_configs") or None,
            compile_flags=cfg.get("compile_flags") or None,
        )
        _KERNEL_CACHE[key] = compiled
    return compiled


def kimi_delta_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    use_qk_l2norm: bool = True,
    scaleValue: float = -1.0,
    *,
    schedule_mode: str = "tuned",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute KimiDeltaAttention matching cannbot-ops level5/kimi_delta_attention proto.yaml."""
    if q.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"unsupported dtype {q.dtype}")
    dk = q.shape[-1]
    eff_scale = (dk ** -0.5) if scaleValue <= 0 else float(scaleValue)

    if q.device.type != "npu":
        qf = q.to(torch.float32)
        kf = k.to(torch.float32)
        vf = v.to(torch.float32)
        decay = g.to(torch.float32).exp()
        beta_f = beta.to(torch.float32)
        if use_qk_l2norm:
            qf = qf * torch.rsqrt(qf.square().sum(dim=-1, keepdim=True) + 1e-6)
            kf = kf * torch.rsqrt(kf.square().sum(dim=-1, keepdim=True) + 1e-6)
        qf = qf * eff_scale
        state = initial_state.to(torch.float32).clone()
        out = torch.empty(vf.shape, dtype=torch.float32, device=q.device)
        for tok in range(q.shape[1]):
            hbar = state * decay[:, tok].unsqueeze(-1)
            kt = kf[:, tok]
            pred = torch.sum(hbar * kt.unsqueeze(-1), dim=-2)
            corr = (vf[:, tok] - pred) * beta_f[:, tok].unsqueeze(-1)
            state = hbar + kt.unsqueeze(-1) * corr.unsqueeze(-2)
            out[:, tok] = torch.sum(state * qf[:, tok].unsqueeze(-1), dim=-2)
        return out.to(q.dtype), state

    tl_dtype = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
    }[q.dtype]
    compiled = _get_kernel(
        dk=dk,
        dv=v.shape[-1],
        tl_dtype=tl_dtype,
        use_qk_l2norm=bool(use_qk_l2norm),
        t_dim=q.shape[1],
        schedule_mode=schedule_mode,
    )
    return compiled(
        q.contiguous(),
        k.contiguous(),
        v.contiguous(),
        g.contiguous(),
        beta.contiguous(),
        initial_state.contiguous(),
        float(eff_scale),
    )


kernel = kimi_delta_attention

__all__ = [
    "_build_kimi_delta_attention_prim_func",
    "_select_kimi_delta_attention_config",
    "kernel",
    "kimi_delta_attention",
]
