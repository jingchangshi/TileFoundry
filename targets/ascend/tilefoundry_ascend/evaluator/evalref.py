"""Reference evaluator for the Ascend950PR AIV Reg-SIMD instruction set.

This is an independent semantic model of the instructions (numpy, CPU). It is
NOT the Kernel under test and does not import the frozen v33 baseline. It is
used to (a) microtest each instruction for f32 vectors, and (b) evaluate the
instruction-level Scheduled HIR against an independent RMSNorm golden.

Vector register = 64 fp32 lanes (256 B / 4 B). Predicate = 64 bits.
"""
from __future__ import annotations

import numpy as np

LANES = 64


def bf16(x):
    """Round float32 values to bfloat16 precision, returned as float32."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    u = x.view(np.uint32)
    lsb = (u >> 16) & np.uint32(1)
    bias = np.uint32(0x7FFF) + lsb
    out = ((u + bias) >> 16).astype(np.uint32) << 16
    return out.view(np.float32)


# ---- instruction semantics (single issue) ----
def pset(elem_width: int, dist: str = "PAT_ALL"):
    if elem_width not in (8, 16, 32):
        raise ValueError("pset elem_width must be 8/16/32 bits")
    if dist != "PAT_ALL":
        raise NotImplementedError(dist)
    return np.ones(LANES, dtype=bool)


def vld(ub: np.ndarray, offset: int, dist: str = "NORM"):
    seg = np.asarray(ub[offset:offset + LANES], dtype=np.float32)
    if seg.shape[0] != LANES:
        raise ValueError("vld window out of range")
    return seg.copy()  # UNPK_B16 already widened bf16->f32 by the UB storage


def vcvt(src: np.ndarray, dtype: str = "float32", part: int = 0):
    if part not in (0, 1):
        raise ValueError("vcvt part must be 0 or 1")
    return np.asarray(src, dtype=np.float32).copy()


def vmula(acc, a, b, mask):
    acc = np.asarray(acc, dtype=np.float32).copy()
    upd = np.asarray(a, np.float32) * np.asarray(b, np.float32)
    return np.where(mask, acc + upd, acc).astype(np.float32)


def vadd(a, b, mask):
    return np.where(mask, np.asarray(a, np.float32) + np.asarray(b, np.float32),
                    np.asarray(a, np.float32)).astype(np.float32)


def vadds(a, s, mask):
    return np.where(mask, np.asarray(a, np.float32) + np.float32(s), np.asarray(a, np.float32)).astype(np.float32)


def vmuls(a, s, mask):
    return np.where(mask, np.asarray(a, np.float32) * np.float32(s), np.asarray(a, np.float32)).astype(np.float32)


def vmul(a, b, mask):
    return np.where(mask, np.asarray(a, np.float32) * np.asarray(b, np.float32), np.asarray(a, np.float32)).astype(np.float32)


def vcadd(src, mask):
    """Pairwise add reduction: dst[i] = src[2i] + src[2i+1] (low half)."""
    s = np.asarray(src, np.float32)
    out = np.zeros(LANES, dtype=np.float32)
    half = LANES // 2
    even = s[0:2 * half:2]
    odd = s[1:2 * half:2]
    out[:half] = np.where(mask[:half], even + odd, np.float32(0.0))
    return out


def vdupv(src, mask, pos="POS_LOWEST"):
    lane = 0 if pos == "POS_LOWEST" else LANES - 1
    return np.full(LANES, np.float32(np.asarray(src, np.float32)[lane]), dtype=np.float32)


def vdup(scalar, mask):
    return np.full(LANES, np.float32(scalar), dtype=np.float32)


def vsqrt(a, mask):
    return np.where(mask, np.sqrt(np.asarray(a, np.float32)), np.asarray(a, np.float32)).astype(np.float32)


def vdiv(a, b, mask):
    return np.where(mask, np.asarray(a, np.float32) / np.asarray(b, np.float32),
                    np.asarray(a, np.float32)).astype(np.float32)


def vsts(ub: np.ndarray, offset: int, src, mask, dist="PK_B32"):
    src = np.asarray(src, np.float32)
    packed = bf16(src)  # PK_B32: f32 -> bf16 rounding into the UB bf16 buffer
    ub[offset:offset + LANES] = np.where(mask, packed, ub[offset:offset + LANES])
    return ub


# ---- whole-tile scheduled-HIR evaluation (mirrors the v33 instruction schedule) ----
def evaluate_tile(x: np.ndarray, wf: np.ndarray, eps: float, rows: int, d: int) -> np.ndarray:
    """Evaluate one tile through the instruction-level schedule.

    x holds rows*d bf16 (already stored as bf16-rounded float32); wf holds d f32
    weights. Returns rows*d bf16-rounded float32 (the UB copy of Y).
    """
    CH = LANES
    NCH = d // CH
    full = pset(32)
    y = np.zeros(rows * d, dtype=np.float32)
    if rows == 4:
        for r in range(4):
            ro = r * d
            acc = [np.zeros(CH, np.float32) for _ in range(4)]
            for i in range(NCH // 4):
                for j in range(4):
                    v = vcvt(vld(x, ro + (i * 4 + j) * CH), "float32", 0)
                    acc[j] = vmula(acc[j], v, v, full)
            a = vadd(vadd(acc[0], acc[1], full), vadd(acc[2], acc[3], full), full)
            for _ in range(6):
                a = vcadd(a, full)
            b = vdupv(a, full, "POS_LOWEST")
            sc = vdiv(vdup(1.0, full), vsqrt(vadds(vmuls(b, np.float32(1.0 / d), full), eps, full), full), full)
            for i in range(NCH):
                vf = vcvt(vld(x, ro + i * CH), "float32", 0)
                wv = vld(wf, i * CH)
                yv = vmul(vmul(vf, sc, full), wv, full)
                vsts(y, ro + i * CH, yv, full, "PK_B32")
        return y
    # rows <= 2 looped path: 2 accumulators, NCH//2
    for r in range(rows):
        ro = r * d
        acc0 = np.zeros(CH, np.float32); acc1 = np.zeros(CH, np.float32)
        for i in range(NCH // 2):
            v0 = vcvt(vld(x, ro + (i * 2) * CH), "float32", 0)
            v1 = vcvt(vld(x, ro + (i * 2 + 1) * CH), "float32", 0)
            acc0 = vmula(acc0, v0, v0, full)
            acc1 = vmula(acc1, v1, v1, full)
        a = vadd(acc0, acc1, full)
        for _ in range(6):
            a = vcadd(a, full)
        b = vdupv(a, full, "POS_LOWEST")
        rs = vdiv(vdup(1.0, full), vsqrt(vadds(vmuls(b, np.float32(1.0 / d), full), eps, full), full), full)
        for i in range(NCH):
            vf = vcvt(vld(x, ro + i * CH), "float32", 0)
            wv = vld(wf, i * CH)
            yv = vmul(vmul(vf, rs, full), wv, full)
            vsts(y, ro + i * CH, yv, full, "PK_B32")
    return y


def evaluate_case(x, w, eps, rows, core_count, loop_mode):
    """Simulate the ragged per-core split and the tile pipeline on CPU.

    Returns the full (batch, d) output buffer (bf16-rounded float32).
    """
    batch, d = x.shape
    tiles = batch // rows
    nbase = tiles // core_count
    rem = tiles % core_count
    iters = nbase + (1 if rem else 0)
    wf = np.asarray(w, np.float32).copy()
    xf = bf16(np.asarray(x, np.float32).reshape(-1))
    out = np.zeros(batch * d, dtype=np.float32)
    for core in range(core_count):
        start = core * nbase + min(core, rem)
        cnt = nbase + (1 if core < rem else 0)
        steps = max(cnt, 1) if loop_mode == "dyn" else iters
        for step in range(steps):
            s = step if loop_mode == "dyn" else min(step, cnt - 1)
            tile = start + s
            if tile >= tiles:
                continue
            base = tile * rows * d
            xt = xf[base:base + rows * d]
            yt = evaluate_tile(xt, wf, eps, rows, d)
            out[base:base + rows * d] = yt
    return out.reshape(batch, d)


def microtests():
    """Deterministic instruction microtests against direct formulas."""
    rng = np.random.RandomState(0)
    full = pset(32)
    res = {}
    # pset granularity: 32 = elem width, 64 f32 lanes
    res["pset_lanes"] = {"ok": int(full.sum()) == 64, "lanes": int(full.sum())}
    # vcadd layout
    v = np.arange(64, dtype=np.float32)
    o = vcadd(v, full)
    res["vcadd_layout"] = {"ok": bool(np.allclose(o[:4], [1, 5, 9, 13]) and np.all(o[32:] == 0)),
                           "o0..3": o[:4].tolist()}
    # 6x vcadd == sum
    a = rng.randn(64).astype(np.float32)
    r = a
    for _ in range(6):
        r = vcadd(r, full)
    res["vcadd_6_sum"] = {"ok": bool(np.isclose(r[0], a.sum(), rtol=1e-4, atol=1e-4)),
                          "got": float(r[0]), "sum": float(a.sum())}
    # vld unpack (bf16 values preserved as f32)
    ub = bf16(rng.randn(128).astype(np.float32))
    res["vld_unpack"] = {"ok": bool(np.allclose(vld(ub, 64), ub[64:128]))}
    # vcvt identity f32
    res["vcvt_identity"] = {"ok": bool(np.allclose(vcvt(a, "float32", 0), a))}
    # mask inactive lanes leave acc unchanged
    mask = np.zeros(64, bool); mask[:32] = True
    acc = np.ones(64, np.float32)
    upd = vmula(acc, np.full(64, 2.0, np.float32), np.full(64, 3.0, np.float32), mask)
    res["mask_inactive"] = {"ok": bool(np.all(upd[32:] == 1.0) and np.all(upd[:32] == 7.0))}
    # vsts address window only
    yb = np.zeros(256, np.float32)
    vsts(yb, 128, np.full(64, 1.5, np.float32), full)
    res["vsts_address"] = {"ok": bool(np.all(yb[128:192] == 1.5) and np.all(yb[:128] == 0) and np.all(yb[192:] == 0))}
    # vdupv POS_LOWEST broadcasts lane0
    src = np.arange(64, dtype=np.float32)
    res["vdupv_lowest"] = {"ok": bool(np.all(vdupv(src, full, "POS_LOWEST") == 0.0))}
    # vsqrt + vdiv exact (not approximate rsqrt)
    x = np.full(64, 4.0, np.float32)
    res["sqrt_div_exact"] = {"ok": bool(np.allclose(vdiv(vdup(1.0, full), vsqrt(x, full), full), np.full(64, 0.5, np.float32)))}
    # full-tile RMSNorm sanity vs closed form
    d = 256; rows = 1
    xt = bf16(rng.randn(rows * d).astype(np.float32))
    wt = rng.randn(d).astype(np.float32)
    yt = evaluate_tile(xt, wt, 1e-6, rows, d)
    ref = bf16((xt.astype(np.float64) / np.sqrt((xt.astype(np.float64) ** 2).mean() + 1e-6) * wt.astype(np.float64)).astype(np.float32))
    res["tile_rmsnorm_vs_closedform"] = {"ok": bool(np.allclose(yt, ref, atol=2e-2, rtol=1e-2)),
                                         "max_abs": float(np.abs(yt - ref).max())}
    return res


__all__ = ["LANES", "bf16", "pset", "vld", "vcvt", "vmula", "vadd", "vadds", "vmuls",
           "vmul", "vcadd", "vdupv", "vdup", "vsqrt", "vdiv", "vsts",
           "evaluate_tile", "evaluate_case", "microtests"]
