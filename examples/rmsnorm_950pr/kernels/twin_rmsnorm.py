"""Independent TileLang Ascend Runtime Twin for BF16 RMSNorm (AIV Reg SIMD).

This is a *candidate* implementation written fresh and parameterised by a config
rows/num_stages/x_versions/y_versions/loop_mode/l2 policy. It does not import the
frozen v33 baseline; it is the device implementation under test (the TF side has
its own independent reference evaluator). The generated kernel is compiled by
TileLang and executed on the real Ascend950PR.
"""
import hashlib, json
from dataclasses import dataclass, asdict

import tilelang
import tilelang.ascend.language as T

BF = "bfloat16"


@dataclass(frozen=True)
class TwinConfig:
    case_id: str
    batch: int
    d: int
    rows: int
    loop_mode: str
    num_stages: int
    x_versions: int
    y_versions: int
    core_count: int = 56
    eps: float = 1e-6
    l2_read: str = "NOTALLOC_CLEAN"
    l2_write: str = "NORMAL_FV"

    def cache_key(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]


def build_twin(cfg: TwinConfig):
    batch, d, rows = cfg.batch, cfg.d, cfg.rows
    N_CORES = cfg.core_count
    CH = 64
    NCH = d // CH
    if d <= 0 or d % (4 * CH):
        raise ValueError("width must be a positive multiple of 256")
    if batch % rows:
        raise ValueError("rows must divide batch")
    tiles = batch // rows
    if tiles < N_CORES:
        raise ValueError("need at least one tile per core")
    nbase = tiles // N_CORES
    rem = tiles % N_CORES
    iters = nbase + (1 if rem else 0)
    dyn = cfg.loop_mode == "dyn"
    total = batch * d
    stages = cfg.num_stages
    xver = cfg.x_versions
    yver = cfg.y_versions

    @T.prim_func
    def main(X: T.Tensor((total,), BF), W: T.Tensor((d,), BF), Y: T.Tensor((total,), BF), eps: T.float32):
        with T.Kernel(N_CORES) as core:
            w_ub = T.alloc_shared((d,), BF)
            wf = T.alloc_shared((d,), "float32")
            x_ub = T.alloc_shared(rows * d, BF)
            y_ub = T.alloc_shared(rows * d, BF)
            T.annotate_buffer_versions({x_ub: xver, y_ub: yver})
            T.copy(W[:d], w_ub[:d])
            with T.SimdVF():
                full = T.simd.pset(32)
                for i in T.serial(NCH):
                    T.simd.vsts(wf[i * CH], T.simd.vcvt(T.simd.vld(w_ub[i * CH], dist="UNPK_B16"), "float32", part=0), full)
            start = core * nbase + T.min(core, rem)
            cnt = nbase + T.if_then_else(core < rem, T.int32(1), T.int32(0))
            for step in T.Pipelined(T.max(cnt, 1) if dyn else iters, num_stages=stages):
                s = step if dyn else T.min(step, cnt - 1)
                base = ((start + s) * rows) * d
                T.copy(X[base:base + rows * d], x_ub, l2_cache_ctrl=cfg.l2_read)
                with T.SimdVF():
                    full = T.simd.pset(32)
                    if rows == 4:
                        for r in range(4):
                            ro = r * d
                            acc0 = T.simd.alloc_var(T.float32); acc0 = T.simd.vdup(T.float32(0.0), T.float32, full)
                            acc1 = T.simd.alloc_var(T.float32); acc1 = T.simd.vdup(T.float32(0.0), T.float32, full)
                            acc2 = T.simd.alloc_var(T.float32); acc2 = T.simd.vdup(T.float32(0.0), T.float32, full)
                            acc3 = T.simd.alloc_var(T.float32); acc3 = T.simd.vdup(T.float32(0.0), T.float32, full)
                            for i in T.serial(NCH // 4):
                                v0 = T.simd.vcvt(T.simd.vld(x_ub[ro + (i * 4 + 0) * CH], dist="UNPK_B16"), "float32", part=0)
                                v1 = T.simd.vcvt(T.simd.vld(x_ub[ro + (i * 4 + 1) * CH], dist="UNPK_B16"), "float32", part=0)
                                v2 = T.simd.vcvt(T.simd.vld(x_ub[ro + (i * 4 + 2) * CH], dist="UNPK_B16"), "float32", part=0)
                                v3 = T.simd.vcvt(T.simd.vld(x_ub[ro + (i * 4 + 3) * CH], dist="UNPK_B16"), "float32", part=0)
                                T.simd.vmula(acc0, v0, v0, full)
                                T.simd.vmula(acc1, v1, v1, full)
                                T.simd.vmula(acc2, v2, v2, full)
                                T.simd.vmula(acc3, v3, v3, full)
                            acc = T.simd.vadd(T.simd.vadd(acc0, acc1, full), T.simd.vadd(acc2, acc3, full), full)
                            a1 = T.simd.vcadd(acc, full); a2 = T.simd.vcadd(a1, full); a3 = T.simd.vcadd(a2, full)
                            a4 = T.simd.vcadd(a3, full); a5 = T.simd.vcadd(a4, full); a6 = T.simd.vcadd(a5, full)
                            b = T.simd.vdupv(a6, full, pos="POS_LOWEST")
                            sc = T.simd.alloc_var(T.float32)
                            sc = T.simd.vdiv(T.simd.vdup(T.float32(1.0), "float32", full),
                                             T.simd.vsqrt(T.simd.vadds(T.simd.vmuls(b, T.float32(1.0 / d), full), eps, full), full), full)
                            for i in T.serial(NCH):
                                vf = T.simd.vcvt(T.simd.vld(x_ub[ro + i * CH], dist="UNPK_B16"), "float32", part=0)
                                wv = T.simd.vld(wf[i * CH])
                                yv = T.simd.vmul(T.simd.vmul(vf, sc, full), wv, full)
                                T.simd.vsts(y_ub[ro + i * CH], T.simd.vcvt(yv, BF), full, dist="PK_B32")
                    else:
                        for r in T.serial(rows):
                            ro = r * d
                            acc0 = T.simd.alloc_var(T.float32); acc0 = T.simd.vdup(T.float32(0.0), T.float32, full)
                            acc1 = T.simd.alloc_var(T.float32); acc1 = T.simd.vdup(T.float32(0.0), T.float32, full)
                            for i in T.serial(NCH // 2):
                                v0 = T.simd.vcvt(T.simd.vld(x_ub[ro + (i * 2) * CH], dist="UNPK_B16"), "float32", part=0)
                                v1 = T.simd.vcvt(T.simd.vld(x_ub[ro + (i * 2 + 1) * CH], dist="UNPK_B16"), "float32", part=0)
                                T.simd.vmula(acc0, v0, v0, full)
                                T.simd.vmula(acc1, v1, v1, full)
                            acc = T.simd.vadd(acc0, acc1, full)
                            a1 = T.simd.vcadd(acc, full); a2 = T.simd.vcadd(a1, full); a3 = T.simd.vcadd(a2, full)
                            a4 = T.simd.vcadd(a3, full); a5 = T.simd.vcadd(a4, full); a6 = T.simd.vcadd(a5, full)
                            b = T.simd.vdupv(a6, full, pos="POS_LOWEST")
                            rs = T.simd.vdiv(T.simd.vdup(T.float32(1.0), "float32", full),
                                             T.simd.vsqrt(T.simd.vadds(T.simd.vmuls(b, T.float32(1.0 / d), full), eps, full), full), full)
                            for i in T.serial(NCH):
                                vf = T.simd.vcvt(T.simd.vld(x_ub[ro + i * CH], dist="UNPK_B16"), "float32", part=0)
                                wv = T.simd.vld(wf[i * CH])
                                yv = T.simd.vmul(T.simd.vmul(vf, rs, full), wv, full)
                                T.simd.vsts(y_ub[ro + i * CH], T.simd.vcvt(yv, BF), full, dist="PK_B32")
                T.copy(y_ub, Y[base:base + rows * d], l2_cache_ctrl=cfg.l2_write)
    return main


_CACHE = {}


def compile_twin(cfg: TwinConfig):
    key = cfg.cache_key()
    if key not in _CACHE:
        _CACHE[key] = tilelang.compile(build_twin(cfg), target="ascend", out_idx=[2])
    return _CACHE[key]


def get_kernel_source(cfg: TwinConfig) -> str:
    k = compile_twin(cfg)
    try:
        return k.get_kernel_source()
    except Exception:
        return "<source unavailable>"