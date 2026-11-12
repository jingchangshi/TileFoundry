#!/usr/bin/env python3
"""Twin (T) CPU reference — the three computation roles, kept separate.

ROLE 1 (actual input) lives in twin_io.load_actual_inputs: recorded BF16 bytes +
eps f32 bits. Nothing here regenerates an input from a seed; make_input() below is
ONLY the fixture PRODUCER used once to materialise recorded bytes, never a consumer
entry point.

ROLE 2 (TF algorithm side) — tf_algo_*(): f32 arithmetic in the program's order with
an EXPLICIT BF16 cast at each conversion point. This models what TF's evaluator must
do; it is not the independent reference.

ROLE 3 (independent reference) — reference_*(): FP64 formula, then ONE BF16 rounding
from FP64 (bf16_bits_from_f64_once). A f64->f32->bf16 double rounding is NOT allowed
here: it is detected by reference_quantization_mismatch() and reported as
REFERENCE_QUANTIZATION_MISMATCH.

No device, no compiler/TIR import. See INDEPENDENCE.md.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

try:  # torch is the frozen CPU INPUT SAMPLER only (fixture production).
    import torch
except Exception:  # noqa: BLE001
    torch = None

# ----------------------------------------------------------- frozen protocol
EPS = 1e-06
ATOL = 0.02
RTOL = 0.01
SEEDS = [0, 1, 2]
PATTERNS = [("zero", 0), ("ones", 0), ("random", 0), ("random", 1),
            ("random", 2), ("alternating", 0)]
CASES = [("prefill_dyn", 8192, 4096, "dyn"),
         ("prefill_recompute", 8192, 4096, "recompute"),
         ("decode_dyn", 64, 4096, "dyn"),
         ("decode_recompute", 64, 4096, "recompute")]
PROTOCOL = {"reference": "cpu_float64_rmsnorm_then_bfloat16",
            "eps": EPS, "atol": ATOL, "rtol": RTOL,
            "patterns": [{"pattern": p, "seed": s} for p, s in PATTERNS],
            "seeds": SEEDS, "repetitions_per_input": 4,
            "stability_runs_per_case": 64}

VADD_N = 64          # f32 vadd-one fixture
RMS_ROW_SHAPE = (1, 256)   # T3 small-D single-row RMSNorm fixture (production D is 4096)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# ------------------------------------------------------- input (PRODUCER only)
def make_input(shape, pattern, seed):
    """Frozen-protocol sampler. PRODUCER ONLY — a consumer must read recorded bytes.

    Semantics copied from the frozen protocol/reference; torch samples, numpy computes.
    """
    if torch is None:
        raise RuntimeError("torch is required as the frozen CPU input sampler")
    g = torch.Generator(device="cpu").manual_seed(seed)
    rows, width = shape
    if pattern == "zero":
        x = torch.zeros(shape, dtype=torch.float32)
        w = torch.ones(width, dtype=torch.float32)
    elif pattern == "ones":
        x = torch.ones(shape, dtype=torch.float32)
        w = torch.linspace(0.25, 1.75, width)
    elif pattern == "random":
        x = torch.randn(shape, generator=g)
        w = torch.randn(width, generator=g)
    elif pattern == "alternating":
        idx = torch.arange(width)
        scale = torch.tensor([0.01, 1.0, 10.0, 100.0])[idx % 4]
        row = scale * torch.where(idx % 2 == 0, 1.0, -1.0)
        signs = torch.where(torch.arange(rows) % 2 == 0, 1.0, -1.0)
        x = signs[:, None] * row[None, :]
        w = torch.linspace(-1.0, 1.0, width)
    else:
        raise ValueError(pattern)
    return x.to(torch.bfloat16).contiguous(), w.to(torch.bfloat16).contiguous()


def bf16_bits_of_tensor(tensor) -> np.ndarray:
    """The bf16 bit patterns (uint16) of an already-bf16 torch tensor."""
    return tensor.view(torch.uint16).numpy().astype(np.uint16)


# --------------------------------------------------- BF16 rounding primitives
def bf16_bits_from_f32_cast(f32) -> np.ndarray:
    """EXPLICIT f32->bf16 cast (role 2's conversion point), RNE, returns uint16 bits.

    This is a deliberate cast, not an identity relabel: a value that is not
    bf16-representable changes here, which is what QUANTIZATION_BOUNDARY_MISMATCH
    checks.
    """
    u = np.ascontiguousarray(f32, dtype=np.float32).view(np.uint32).astype(np.uint64)
    lsb = (u >> np.uint64(16)) & np.uint64(1)
    u2 = (u + np.uint64(0x7FFF) + lsb) & np.uint64(0xFFFF0000)
    return (u2.astype(np.uint32) >> np.uint32(16)).astype(np.uint16)


def bf16_bits_from_f64_once(values) -> np.ndarray:
    """FP64 -> BF16 with ONE round-to-nearest-even step. The independent reference cast.

    Never goes through f32: doing f64->f32->bf16 rounds twice and can move a value
    across a bf16 boundary (see reference_quantization_mismatch / the bad control).
    """
    arr = np.asarray(values, dtype=np.float64)
    sign = ((arr.view(np.uint64) >> np.uint64(63)) & np.uint64(1)).astype(np.uint16)
    mag = np.abs(arr)
    bits = np.zeros(arr.shape, dtype=np.uint16)

    nan = np.isnan(mag)
    inf = np.isinf(mag)
    zero = mag == 0.0
    mant, exp2 = np.frexp(np.where(nan | inf | zero, 1.0, mag))  # mag = mant*2**exp2
    e = exp2 - 1                     # mag = (mant*2) * 2**e, mant*2 in [1,2)
    s = mant * 2.0

    normal = ~(nan | inf | zero) & (e >= -126)
    # significand with 7 fractional bits: q in [128, 256]
    q = np.rint(np.where(normal, s * 128.0, 0.0)).astype(np.int64)
    e_adj = e.copy()
    carry = normal & (q >= 256)
    q = np.where(carry, q // 2, q)
    e_adj = np.where(carry, e_adj + 1, e_adj)
    overflow = normal & (e_adj > 127)
    ok = normal & ~overflow
    bits[ok] = (((e_adj[ok] + 127).astype(np.uint16) << np.uint16(7))
                | (q[ok] - 128).astype(np.uint16))
    bits[overflow] = np.uint16(0x7F80)

    subnormal = ~(nan | inf | zero) & (e < -126)
    if np.any(subnormal):
        scaled = np.rint(mag[subnormal] * (2.0 ** 133)).astype(np.int64)
        scaled = np.clip(scaled, 0, 127)
        bits[subnormal] = scaled.astype(np.uint16)

    bits[inf] = np.uint16(0x7F80)
    bits[nan] = np.uint16(0x7FC0)
    bits = bits | (sign << np.uint16(15))
    return bits


def f32_values_from_bf16_bits(bits) -> np.ndarray:
    return (np.ascontiguousarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def reference_quantization_mismatch(exact_f64, ref_bits) -> bool:
    """True when a claimed independent reference did NOT do one FP64->BF16 rounding."""
    return not np.array_equal(np.asarray(ref_bits, dtype=np.uint16),
                              bf16_bits_from_f64_once(exact_f64))


def assert_single_round_reference(exact_f64, ref_bits, source_id="reference"):
    if reference_quantization_mismatch(exact_f64, ref_bits):
        from twin_io import TwinError
        raise TwinError("REFERENCE_QUANTIZATION_MISMATCH", "reference", source_id,
                        "one FP64->BF16 rounding", "different bits (double rounding?)")


# ---------------------------------------------------- ROLE 2: TF algorithm side
def tf_algo_vadd_one_f32(x_f32):
    """f32 elementwise x + 1.0. No BF16 in this fixture; exactly f32 arithmetic."""
    return (np.asarray(x_f32, dtype=np.float32) + np.float32(1.0)).astype(np.float32)


def tf_algo_rmsnorm_bf16(x_bits, w_bits, eps_f32):
    """TF algorithm side: widen bf16 exactly, compute f32, EXPLICIT bf16 cast."""
    x = f32_values_from_bf16_bits(x_bits)      # bf16 -> f32 is exact widening
    w = f32_values_from_bf16_bits(w_bits)
    d = np.float32(x.shape[-1])
    sq = (x * x).astype(np.float32)
    mean = (sq.sum(axis=-1, keepdims=True) / d).astype(np.float32)
    denom = np.sqrt(mean + np.float32(eps_f32)).astype(np.float32)
    rstd = (np.float32(1.0) / denom).astype(np.float32)
    y = (x * rstd * w[None, :]).astype(np.float32)
    return bf16_bits_from_f32_cast(y)          # explicit conversion point


# --------------------------------------------- ROLE 3: independent reference
def reference_vadd_one_f32(x_f32):
    """Independent f32 vadd-one: FP64 sum rounded ONCE to f32 (same value, proven)."""
    exact = np.asarray(x_f32, dtype=np.float64) + np.float64(1.0)
    return exact.astype(np.float32)


def reference_rmsnorm_bf16_once(x_bits, w_bits, eps_f32):
    """FP64 RMSNorm on bf16 inputs, then ONE bf16 rounding. Returns (exact64, bits)."""
    x = f32_values_from_bf16_bits(x_bits).astype(np.float64)
    w = f32_values_from_bf16_bits(w_bits).astype(np.float64)
    var = np.mean(x * x, axis=-1, keepdims=True)
    rstd = 1.0 / np.sqrt(var + np.float64(eps_f32))
    exact = x * rstd * w[None, :]
    return exact, bf16_bits_from_f64_once(exact)


# --------------------------------------------------------------- dispatch
def compute_reference(inputs, operation, protocol=None):
    """Compute the independent reference OutputBundle payload from loaded inputs."""
    arrays = inputs["arrays"]
    eps_f32 = inputs["eps_f32"]
    if operation == "vadd_one":
        x = np.asarray(arrays["x"], dtype=np.float32)
        got = reference_vadd_one_f32(x)
        return {"name": "out", "dtype": "float32", "values": got}
    if operation == "rmsnorm":
        _, bits = reference_rmsnorm_bf16_once(arrays["x"], arrays["w"], eps_f32)
        return {"name": "out", "dtype": "bfloat16", "values": bits, "exact64": None}
    from twin_io import TwinError
    raise TwinError("TWIN_UNSUPPORTED_OP", "reference", operation, "vadd_one|rmsnorm", operation)


def replay_logical(program_ref, inputs, plan_ref=None):
    """Minimal logical replay for the two T0-T4 examples (no Plan yet)."""
    from twin_io import TwinError
    if plan_ref is not None:
        raise TwinError("TWIN_UNSUPPORTED_PLAN", "replay", program_ref.get("entry", ""),
                        "LOGICAL_ONLY", "plan_ref supplied (T5 blocked_by R)")
    operation = program_ref.get("operation")
    arrays = inputs["arrays"]
    eps_f32 = inputs["eps_f32"]
    if operation == "vadd_one":
        return {"name": "out", "dtype": "float32",
                "values": tf_algo_vadd_one_f32(arrays["x"])}
    if operation == "rmsnorm":
        return {"name": "out", "dtype": "bfloat16",
                "values": tf_algo_rmsnorm_bf16(arrays["x"], arrays["w"], eps_f32)}
    raise TwinError("TWIN_UNSUPPORTED_OP", "replay", program_ref.get("entry", ""),
                    "vadd_one|rmsnorm", operation)
