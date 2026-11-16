"""Standalone MTE2 inflight queue and gather coalescing micro-probe."""


def build_mte2_coalesce_func(
    rows: int = 56 * 64, cols: int = 256, coalesced: bool = True, num_cores: int = 56
):
    """Build a PrimFunc comparing coalesced block copy against fragmented slice loads."""
    import tilelang.ascend.language as T

    rows_per_core = rows // num_cores
    sub_chunk = 8

    @T.prim_func
    def main(X: T.Tensor((rows, cols), "float32"), Y: T.Tensor((rows, cols), "float32")):
        with T.Kernel(num_cores) as bx:
            ub = T.alloc_shared((rows_per_core, cols), "float32")
            base = bx * rows_per_core
            if coalesced:
                T.copy(X[base : base + rows_per_core, 0:cols], ub)
            else:
                for r in T.serial(rows_per_core):
                    for seg in T.serial(cols // sub_chunk):
                        c0 = seg * sub_chunk
                        T.copy(X[base + r, c0 : c0 + sub_chunk], ub[r, c0 : c0 + sub_chunk])
            T.copy(ub, Y[base : base + rows_per_core, 0:cols])

    return main


def run_mte2_queue_probe(rows: int = 56 * 64, cols: int = 256) -> dict[str, float]:
    """Benchmark coalesced vs fragmented MTE2 DMA throughput on Ascend 950."""
    import tilelang
    from tilelang.profiler import do_bench
    import torch

    x = torch.randn((rows, cols), dtype=torch.float32, device="npu")
    timings: dict[str, float] = {}
    for label, flag in (("coalesced", True), ("fragmented", False)):
        kernel = tilelang.compile(build_mte2_coalesce_func(rows, cols, flag), out_idx=-1)
        out = kernel(x)
        torch.npu.synchronize()
        torch.testing.assert_close(out.cpu(), x.cpu(), atol=1e-5, rtol=1e-5)
        prof = do_bench(lambda: kernel(x), backend="msprof_detail", _n_warmup=5, _n_repeat=8)
        timings[f"{label}_dur_us"] = float(prof.dur_us)
        timings[f"{label}_aiv_mte2"] = float(prof.aiv_mte2)
    f_us = timings["fragmented_dur_us"]
    c_us = timings["coalesced_dur_us"]
    timings["coalesce_speedup"] = (f_us / c_us) if c_us > 0.0 else 0.0
    return timings
