"""Standalone HBM sustainable bandwidth micro-probe for Ascend 950PR."""

from collections.abc import Sequence

DEFAULT_HBM_BURST_BYTES: tuple[int, ...] = (4096, 8192, 16384, 32768, 65536)


def _burst_tile_elems(burst_bytes: int) -> int:
    return max(64, (int(burst_bytes) // 4 // 64) * 64)


def _default_burst_steps(burst_bytes: int, num_cores: int, total_stream_bytes: int = 224 * 1024 * 1024) -> int:
    tile_bytes = _burst_tile_elems(burst_bytes) * 4
    return max(64, total_stream_bytes // max(1, int(num_cores) * tile_bytes))


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


def build_hbm_burst_probe_func(
    burst_bytes: int = 32768,
    steps_per_core: int | None = None,
    num_cores: int = 56,
):
    """Build a single-slab double-buffered GM->UB burst probe for a given byte payload."""
    import tilelang.ascend.language as T

    tile_elems = _burst_tile_elems(burst_bytes)
    steps = (
        int(steps_per_core)
        if steps_per_core is not None
        else _default_burst_steps(burst_bytes, num_cores)
    )
    total_elems = num_cores * steps * tile_elems

    @T.prim_func
    def main(
        A: T.Tensor((total_elems,), "float32"),
        Out: T.Tensor((num_cores, 64), "float32"),
    ):
        with T.Kernel(num_cores) as bx:
            a_ub = T.alloc_shared((tile_elems,), "float32")
            sink_ub = T.alloc_shared((64,), "float32")
            T.annotate_buffer_versions({a_ub: 2, sink_ub: 1})
            with T.SimdVF():
                m32 = T.simd.pset(32, "PAT_ALL")
                z = T.simd.vdup(T.float32(0.0), "float32", mask=m32)
                T.simd.vsts(sink_ub[0], z, mask=m32, dist="NORM_B32")
            for step in T.Pipelined(steps, num_stages=2):
                off = (step * num_cores + bx) * tile_elems
                T.copy(A[off : off + tile_elems], a_ub)
                with T.SimdVF():
                    m32 = T.simd.pset(32, "PAT_ALL")
                    cur_sum = T.simd.vld(sink_ub[0])
                    head_v = T.simd.vld(a_ub[0])
                    tail_v = T.simd.vld(a_ub[tile_elems - 64])
                    delta = T.simd.vadd(head_v, tail_v, mask=m32)
                    T.simd.vsts(sink_ub[0], T.simd.vadd(cur_sum, delta, mask=m32), mask=m32)
            T.copy(sink_ub, Out[bx : bx + 1, 0:64])

    return main


def run_hbm_burst_bw_probe(
    burst_bytes_list: Sequence[int] = DEFAULT_HBM_BURST_BYTES,
    steps_per_core: int | None = None,
    num_cores: int = 56,
) -> dict[int, dict[str, float]]:
    """Measure GM->UB effective HBM bandwidth across contiguous payload byte sizes."""
    import tilelang
    from tilelang.profiler import do_bench
    import torch

    results: dict[int, dict[str, float]] = {}
    for raw_bytes in burst_bytes_list:
        req_bytes = int(raw_bytes)
        tile_elems = _burst_tile_elems(req_bytes)
        steps = (
            int(steps_per_core)
            if steps_per_core is not None
            else _default_burst_steps(req_bytes, num_cores)
        )
        total_elems = num_cores * steps * tile_elems
        fn = build_hbm_burst_probe_func(req_bytes, steps, num_cores)
        kernel = tilelang.compile(fn, out_idx=-1)
        a = torch.randn(total_elems, dtype=torch.float32, device="npu")
        out = kernel(a)
        torch.npu.synchronize()
        a_view = a.cpu().view(steps, num_cores, tile_elems)
        ref = (a_view[:, :, :64] + a_view[:, :, tile_elems - 64 :]).sum(dim=0)
        torch.testing.assert_close(out.cpu(), ref, atol=1e-2, rtol=1e-2)
        prof = do_bench(lambda: kernel(a), backend="msprof_detail", _n_warmup=5, _n_repeat=8)
        dur_s = float(prof.dur_ns) * 1e-9
        total_bytes = float(total_elems * 4 + num_cores * 64 * 4)
        bw_bps = total_bytes / dur_s if dur_s > 0.0 else 0.0
        results[req_bytes] = {
            "dur_us": float(prof.dur_us),
            "bandwidth_bytes_per_sec": bw_bps,
            "bandwidth_gbps": bw_bps / 1e9,
            "aiv_mte2_ratio": float(prof.aiv_mte2),
        }
    return results


def run_hbm_bw_probe(num_elems: int = 56 * 8192 * 32, num_cores: int = 56) -> dict[str, float]:
    """Compile, verify, and benchmark sustainable and segmented HBM bandwidth."""
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
    report: dict[str, float] = {
        "dur_us": float(profile.dur_us),
        "bandwidth_bytes_per_sec": bw_bps,
        "bandwidth_gbps": bw_bps / 1e9,
        "aiv_mte2_ratio": float(profile.aiv_mte2),
        "aiv_mte3_ratio": float(profile.aiv_mte3),
    }
    burst_metrics = run_hbm_burst_bw_probe(DEFAULT_HBM_BURST_BYTES, num_cores=num_cores)
    for b_size, entry in burst_metrics.items():
        report[f"burst_{b_size}_bps"] = float(entry["bandwidth_bytes_per_sec"])
        report[f"burst_{b_size}_gbps"] = float(entry["bandwidth_gbps"])
    return report
