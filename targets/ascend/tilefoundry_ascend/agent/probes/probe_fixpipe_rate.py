"""Standalone FixPipe bottleneck and K-accumulation span micro-probe."""


def build_fixpipe_gemm_func(
    m_dim: int = 3584, n_dim: int = 256, k_dim: int = 64, num_cores: int = 28
):
    """Build a Cube GEMM PrimFunc exposing FixPipe writeout pressure vs K span."""
    import tilelang.ascend.language as T

    tm, tn, tk = 128, 256, 64
    k_tiles = k_dim // tk
    m_tiles = m_dim // tm

    @T.prim_func
    def main(
        X: T.Tensor((m_dim, k_dim), "bfloat16"),
        W: T.Tensor((n_dim, k_dim), "bfloat16"),
        C: T.Tensor((m_dim, n_dim), "float32"),
    ):
        with T.Kernel(num_cores) as bx:
            res = T.alloc_l0c((tm, tn), "float32")
            x_l1 = T.alloc_l1((tm, tk), "bfloat16")
            w_l1 = T.alloc_l1((tn, tk), "bfloat16")
            for mt in T.serial(m_tiles):
                if mt % num_cores == bx:
                    for kt in T.Pipelined(k_tiles, num_stages=2):
                        T.copy(X[mt * tm : (mt + 1) * tm, kt * tk : (kt + 1) * tk], x_l1)
                        T.copy(W[0:tn, kt * tk : (kt + 1) * tk], w_l1)
                        T.gemm(x_l1, w_l1, res, transpose_B=True, clear_accum=kt == 0)
                    T.copy(res, C[mt * tm : (mt + 1) * tm, 0:tn])

    return main


def run_fixpipe_probe(m_dim: int = 3584, n_dim: int = 256) -> dict[str, float]:
    """Compare FixPipe utilization across short and enlarged K accumulation spans."""
    import tilelang
    from tilelang.profiler import do_bench
    import torch

    results: dict[str, float] = {}
    for label, k_dim in (("short_k", 64), ("long_k", 1024)):
        kernel = tilelang.compile(build_fixpipe_gemm_func(m_dim, n_dim, k_dim), out_idx=-1)
        x = torch.randn((m_dim, k_dim), dtype=torch.bfloat16, device="npu")
        w = torch.randn((n_dim, k_dim), dtype=torch.bfloat16, device="npu")
        out = kernel(x, w)
        torch.npu.synchronize()
        ref = torch.matmul(x.cpu().float(), w.cpu().float().t())
        torch.testing.assert_close(out.cpu(), ref, atol=1e-1, rtol=5e-2)
        prof = do_bench(lambda: kernel(x, w), backend="msprof_detail", _n_warmup=5, _n_repeat=8)
        results[f"{label}_dur_us"] = float(prof.dur_us)
        results[f"{label}_aic_mad"] = float(prof.aic_mad)
        results[f"{label}_aic_fixpipe"] = float(prof.aic_fixpipe)
    return results
