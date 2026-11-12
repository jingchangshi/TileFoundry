"""CPU algorithm core for the single-kernel attention_residual design.

This is an executable algorithm model, not a TileFoundry integration, TileLang
device kernel, official golden, or replacement comparator. NumPy is required.
Use the original attachment harness for all formal NPU acceptance.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math

import numpy as np


def _validate(v_stack, x, w, gamma, alpha, epsilon):
    v, x, w, gamma = map(np.asarray, (v_stack, x, w, gamma))
    if v.ndim != 4:
        raise ValueError("v_stack must be [N,B,T,D]")
    n, b, t, d = v.shape
    if n < 2 or d < 1 or x.shape != (b, t, d):
        raise ValueError("invalid N/D or x shape")
    if w.shape != (d,) or gamma.shape != (d,):
        raise ValueError("w/gamma must be [D]")
    if not 0.05 <= alpha <= 0.95 or not 1e-8 <= epsilon <= 1e-3:
        raise ValueError("alpha/epsilon outside golden domain")
    return v, x, w, gamma


def _scores(v, w, gamma, d, epsilon, mode, gw=None):
    """v has shape [CN,R,D], all mathematical intermediates are FP32."""
    sq = np.sum(v * v, axis=-1, dtype=np.float32)
    rms = np.sqrt(sq / np.float32(d) + np.float32(epsilon))
    if mode == "literal_fp32":
        key = (v / rms[..., None]) * gamma
        return np.sum(key * w, axis=-1, dtype=np.float32) / np.float32(math.sqrt(d))
    if mode == "factored_gw":
        gw = gamma * w if gw is None else gw
        dot = np.sum(v * gw, axis=-1, dtype=np.float32)
        return (dot / rms) / np.float32(math.sqrt(d))
    raise ValueError(f"unknown score_mode: {mode}")


def attention_residual_blocked(
    v_stack, x, w, gamma, alpha, epsilon=1e-6, *,
    token_rows=1, block_n=1, score_mode="literal_fp32",
):
    """One algebraic model covers online, blocked_online and resident_n.

    The CPU implementation intentionally allocates temporary NumPy arrays.
    The NPU lowering must keep them on chip and use D-chunk register sums.
    No output, precision result, or speed here is a formal NPU result.
    """
    v, x, w, gamma = _validate(v_stack, x, w, gamma, alpha, epsilon)
    if type(token_rows) is not int or token_rows < 1 or type(block_n) is not int or block_n < 1:
        raise ValueError("token_rows/block_n must be positive integers")
    n, b, t, d = v.shape
    original_dtype = x.dtype
    vf = v.astype(np.float32).reshape(n, b * t, d)
    xf = x.astype(np.float32).reshape(b * t, d)
    wf, gf = w.astype(np.float32), gamma.astype(np.float32)
    gw = gf * wf if score_mode == "factored_gw" else None
    y = np.empty_like(xf)
    af = np.float32(alpha)
    for begin in range(0, b * t, token_rows):
        end = min(b * t, begin + token_rows)
        m = l = a = None
        for n_begin in range(0, n, block_n):
            values = vf[n_begin:min(n, n_begin + block_n), begin:end]
            scores = _scores(values, wf, gf, d, epsilon, score_mode, gw)
            block_m = np.max(scores, axis=0)
            exp = np.exp(scores - block_m[None, :])
            block_l = np.sum(exp, axis=0, dtype=np.float32)
            block_u = np.sum(exp[..., None] * values, axis=0, dtype=np.float32)
            if m is None:
                m, l, a = block_m, block_l, block_u
            else:
                new_m = np.maximum(m, block_m)
                rho, beta = np.exp(m - new_m), np.exp(block_m - new_m)
                l = rho * l + beta * block_l
                a = rho[:, None] * a + beta[:, None] * block_u
                m = new_m
        y[begin:end] = (np.float32(1) - af) * xf[begin:end] + af * (a / l[:, None])
    return y.reshape(b, t, d).astype(original_dtype)


def attention_residual_dense_model(v_stack, x, w, gamma, alpha, epsilon=1e-6):
    """Independent dense CPU formula for diagnostic model checks only."""
    v, x, w, gamma = _validate(v_stack, x, w, gamma, alpha, epsilon)
    n, b, t, d = v.shape
    original_dtype = x.dtype
    v = v.astype(np.float32)
    x = x.astype(np.float32)
    w, gamma = w.astype(np.float32), gamma.astype(np.float32)
    rms = np.sqrt(np.mean(v * v, axis=-1, keepdims=True, dtype=np.float32) + np.float32(epsilon))
    key = (v / rms) * gamma
    score = np.sum(key * w, axis=-1, dtype=np.float32) / np.float32(math.sqrt(d))
    exp = np.exp(score - np.max(score, axis=0, keepdims=True))
    p = exp / np.sum(exp, axis=0, keepdims=True, dtype=np.float32)
    agg = np.sum(p[..., None] * v, axis=0, dtype=np.float32)
    a = np.float32(alpha)
    return ((np.float32(1) - a) * x + a * agg).astype(original_dtype)


@dataclass(frozen=True)
class Candidate:
    family: str
    token_rows: int
    block_n: int
    v_versions: int
    d_chunk: int
    score_mode: str
    estimated_ub_bytes: int
    program_groups: int
    active_programs: int

    def as_dict(self):
        return asdict(self)


def enumerate_candidates(
    n, tokens, d, *, ub_capacity_bytes, aiv_count,
    compiler_reserve_bytes, reduction_scratch_bytes,
    element_bytes=2, alignment_bytes=32, vector_load_elements=1,
    token_rows=(1, 2, 4, 8, 16), d_chunks=(64, 128, 256, 512),
):
    """Return a finite, capacity-estimated space; not a legality/performance proof.

    Every numeric hardware/budget argument comes from the caller's pinned
    target. Backend allocations/spill must be checked after compilation.
    The conservative estimate retains a full FP32 acc even for resident_n.
    """
    if min(n, tokens, d, ub_capacity_bytes, aiv_count, element_bytes, alignment_bytes, vector_load_elements) < 1:
        raise ValueError("positive dimensions and target capacities required")
    if compiler_reserve_bytes < 0 or reduction_scratch_bytes < 0:
        raise ValueError("reserve/scratch must be nonnegative")

    def aligned(value):
        return ((value + alignment_bytes - 1) // alignment_bytes) * alignment_bytes

    # Conservatively pad to the D chunk and byte alignment for full-width
    # vector loads. Pass the backend's unmasked source-load width explicitly;
    # vector_load_elements=1 models a verified scalar/masked load path only.
    row_alignment = alignment_bytes // math.gcd(alignment_bytes, element_bytes)
    ns = sorted({c for c in (1, 2, 4, 8, 16, 32, 64, n) if c <= n})
    result = []
    for r in token_rows:
        if r < 1 or (tokens == 1 and r != 1):
            continue
        groups = (tokens + r - 1) // r
        for cn in ns:
            family = "resident_n" if cn == n else ("online" if cn == 1 else "blocked_online")
            for versions in (1, 2, 3):
                # A version names an N-block for streaming and a group for full residency.
                for bd in d_chunks:
                    if bd < 1:
                        continue
                    multiple = math.lcm(row_alignment, bd, vector_load_elements)
                    dp = ((d + multiple - 1) // multiple) * multiple
                    for mode in ("literal_fp32", "factored_gw"):
                        # Literal scoring retains gamma and w separately. Factored
                        # scoring retains gw after an in-kernel one-time multiply.
                        # Any temporary overlap during that preparation is charged
                        # to compiler_reserve_bytes and checked by final placement.
                        weights = 2 if mode == "literal_fp32" else 1
                        arrays = [4 * dp * weights,
                                  versions * r * cn * dp * element_bytes,
                                  4 * r * dp, r * dp * element_bytes,
                                  4 * r * cn, 16 * r]
                        estimate = (sum(aligned(value) for value in arrays)
                                    + compiler_reserve_bytes + reduction_scratch_bytes)
                        if estimate > ub_capacity_bytes:
                            continue
                        result.append(Candidate(family, r, cn, versions, bd, mode,
                                                estimate, groups, min(aiv_count, groups)))
    return result


def self_check():
    """Check block-state identity on small CPU examples, not official cases."""
    rng = np.random.default_rng(20261111)
    measurements = []
    for n, b, t, d in ((3, 2, 3, 257), (8, 1, 4, 256), (2, 1, 1, 257)):
        v = rng.uniform(-1, 1, (n, b, t, d)).astype(np.float32)
        x = rng.uniform(-1, 1, (b, t, d)).astype(np.float32)
        w = rng.uniform(-1, 1, d).astype(np.float32)
        gamma = rng.uniform(0.5, 1.5, d).astype(np.float32)
        expected = attention_residual_dense_model(v, x, w, gamma, 0.5)
        for cn in sorted({1, 2, n}):
            for r in (1, 2, 4):
                for mode in ("literal_fp32", "factored_gw"):
                    actual = attention_residual_blocked(v, x, w, gamma, 0.5,
                        token_rows=r, block_n=cn, score_mode=mode)
                    error = float(np.max(np.abs(actual - expected)))
                    if not np.isfinite(actual).all() or error > 5e-5:
                        raise AssertionError((n, b, t, d, cn, r, mode, error))
                    measurements.append(error)
    return {"scope": "CPU_algorithm_identity_diagnostic_only", "model_checks": len(measurements),
            "maximum_absolute_difference": max(measurements), "npu_executions": 0,
            "official_precision_acceptance": "NOT_RUN"}


if __name__ == "__main__":
    import json
    print(json.dumps(self_check(), indent=2))
