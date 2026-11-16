"""Standalone HBM sustainable bandwidth micro-probe for Ascend 950PR."""


def build_hbm_stream_func(num_elems: int = 56 * 8192 * 32, num_cores: int = 56):
    """Build a double-buffered GM streaming PrimFunc saturating HBM bandwidth."""
    import tilelang.ascend.language as T

    tile = 8192
    steps = num_elems // (num_cores * tile)

    @T.prim_func
    def main(
        A: T.Tensor((num_elems,), "float32"),
        B: T.Tensor((num_elems,), "float32"),
        C: T.Tensor((num_elems,), "float32"),
    ):
        with T.Kernel(num_cores) as bx:
            a_ub = T.alloc_shared((tile,), "float32")
            b_ub = T.alloc_shared((tile,), "float32")
            c_ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({a_ub: 2, b_ub: 2, c_ub: 2})
            for step in T.Pipelined(steps, num_stages=2):
                off = (step * num_cores + bx) * tile
                T.copy(A[off : off + tile], a_ub)
                T.copy(B[off : off + tile], b_ub)
                with T.SimdVF():
                    mask = T.simd.pset(32)
                    for i in range(tile // 64):
                        va = T.simd.vld(a_ub[i * 64])
                        vb = T.simd.vld(b_ub[i * 64])
                        T.simd.vsts(c_ub[i * 64], T.simd.vadd(va, vb, mask), mask)
                T.copy(c_ub, C[off : off + tile])

    return main


def run_hbm_bw_probe(num_elems: int = 56 * 8192 * 32, num_cores: int = 56) -> dict[str, float]:
    """Compile, verify, and benchmark sustainable HBM streaming bandwidth."""
    import tilelang
    from tilelang.profiler import do_bench
    import torch

    kernel = tilelang.compile(build_hbm_stream_func(num_elems, num_cores), out_idx=-1)
    a = torch.randn(num_elems, dtype=torch.float32, device="npu")
    b = torch.randn(num_elems, dtype=torch.float32, device="npu")
    out = kernel(a, b)
    torch.npu.synchronize()
    torch.testing.assert_close(out.cpu(), (a + b).cpu(), atol=1e-4, rtol=1e-4)
    profile = do_bench(lambda: kernel(a, b), backend="msprof_detail", _n_warmup=5, _n_repeat=8)
    dur_s = float(profile.dur_ns) * 1e-9
    total_bytes = float(3 * num_elems * 4)
    bw_bps = total_bytes / dur_s if dur_s > 0.0 else 0.0
    return {
        "dur_us": float(profile.dur_us),
        "bandwidth_bytes_per_sec": bw_bps,
        "bandwidth_gbps": bw_bps / 1e9,
        "aiv_mte2_ratio": float(profile.aiv_mte2),
        "aiv_mte3_ratio": float(profile.aiv_mte3),
    }
