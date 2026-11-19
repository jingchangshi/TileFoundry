"""Standalone SSBuffer cross-core scalar handoff latency micro-probe."""

from tilefoundry_ascend.agent.probes._cann_env import (
    ensure_cann_include_cpath,
    run_msprof_detail,
)


def build_ssbuf_handoff_func(rounds: int = 64, use_ssbuf: bool = True):
    """Build a MixCore 1:2 scalar handoff PrimFunc via SSBUF or GM scratch."""
    import tilelang.ascend.language as T  # noqa: PLC0415

    @T.prim_func
    def main_ss(seed: T.Tensor((16, 16), "float16"), out: T.Tensor((rounds, 1), "uint64")):
        with T.MixedKernel(1) as (_, sid):
            words = T.alloc_ss((2,), "uint32")
            packed = T.view(words, shape=(1,), dtype="uint64")
            T.annotate_buffer_versions({words: (3, "auto")})
            cube = T.alloc_l1((16, 16), "float16")
            T.copy(seed, cube)
            for i in T.Pipelined(rounds, num_stages=3, annotations={"enable_offset": True}):
                with T.Stage(0):
                    words[sid] = T.Cast("uint32", (i + 1) * (100 + sid))
                with T.Stage(2):
                    out[i, 0] = packed[0]

    @T.prim_func
    def main_gm(
        scratch: T.Tensor((rounds, 2), "int32"),
        out: T.Tensor((rounds, 1), "uint64"),
    ):
        with T.Kernel(1):
            for i in T.serial(rounds):
                with T.Cube():
                    T.ascend_cross_core_wait_flag(4, "PIPE_S", 0)
                    T.ascend_cross_core_wait_flag(4, "PIPE_S", 16)
                    lo = T.Cast("uint64", T.Cast("uint32", scratch[i, 0]))
                    hi = T.Cast("uint64", T.Cast("uint32", scratch[i, 1])) << 32
                    out[i, 0] = lo | hi
                    T.ascend_cross_core_set_flag(4, "PIPE_S", 0)
                    T.ascend_cross_core_set_flag(4, "PIPE_S", 16)
                with T.Vector() as sid:
                    scratch[i, sid] = T.Cast("int32", (i + 1) * (100 + sid))
                    T.ascend_cross_core_set_flag(4, "PIPE_S", 0)
                    T.ascend_cross_core_wait_flag(4, "PIPE_S", 0)

    return main_ss if use_ssbuf else main_gm


def run_ssbuf_probe(rounds: int = 64) -> dict[str, float]:
    """Verify dirty-init safety and benchmark SSBUF vs GM scalar handoff latency."""
    import tilelang  # noqa: PLC0415
    import torch  # noqa: PLC0415

    ensure_cann_include_cpath()
    seed = torch.zeros((16, 16), dtype=torch.float16).npu()
    expected = torch.tensor(
        [[(i + 1) * 100 + (((i + 1) * 101) << 32)] for i in range(rounds)],
        dtype=torch.uint64,
    )
    k_ss = tilelang.compile(build_ssbuf_handoff_func(rounds, use_ssbuf=True), out_idx=-1)
    out_ss = k_ss(seed)
    torch.npu.synchronize()
    torch.testing.assert_close(out_ss.cpu(), expected, rtol=0, atol=0)
    prof_ss = run_msprof_detail(lambda: k_ss(seed), n_warmup=5, n_repeat=10)

    k_gm = tilelang.compile(
        build_ssbuf_handoff_func(rounds, use_ssbuf=False),
        out_idx=-1,
        pass_configs={"tl.enable_auto_schedule": False},
    )
    scratch = torch.zeros((rounds, 2), dtype=torch.int32).npu()
    out_gm = k_gm(scratch)
    torch.npu.synchronize()
    torch.testing.assert_close(out_gm.cpu(), expected, rtol=0, atol=0)
    prof_gm = run_msprof_detail(lambda: k_gm(scratch), n_warmup=5, n_repeat=10)

    ss_us = float(prof_ss.dur_us)
    gm_us = float(prof_gm.dur_us)
    return {
        "ssbuf_dur_us": ss_us,
        "gm_dur_us": gm_us,
        "ssbuf_speedup": (gm_us / ss_us) if ss_us > 0.0 else 0.0,
    }
