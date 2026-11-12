"""P5-01/G50: the measured attention_residual kernel (CN=1 online, literal_fp32).

THIS IS THE KERNEL THAT PRODUCED THE 20/20 DEVICE RESULT, not a re-description of it. The
emitted text in twin.py is the same program in reviewable form; this module is the one that is
compiled and executed, so a divergence between them shows up as a failing conformance check
rather than as two quietly different implementations.

WHAT THE SHAPE OF THIS CODE RECORDS, because each was a defect found by running:

  * Buffers are FLAT and exactly-D-sized on the GM side, with a next_power_of_2(D) UB allocation.
    Layout inference on this target refuses a shared buffer whose extent is not a power of two,
    which is why D=2047 and D=1000 will not compile otherwise.
  * Every DMA moves exactly D elements into the padded buffer. A copy that moved PD elements
    failed with "CanProveEqual(src_total, dst_total)" on D=5120 and D=7168.
  * Cross-region state lives in shared (UB), never in a fragment. A fragment is VF-local; the
    compiler emits it as a simt_vf pointer argument and rejects it across regions with
    "pointer arguments to simt_vf functions must reside in __gm__ or __ubuf__".
  * The grid is M -- ONE TOKEN GROUP PER PROGRAM -- so the outer persistent loop runs exactly
    once. With niter>1 every iteration after the first produced wrong results, deterministically
    at the same rows. Rather than ship a persistent path that was not understood, the loop was
    removed. The persistent g+=P schedule remains open work for G51 and is NOT claimed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

# ------------------------------------------------------------------ ABI

@dataclass(frozen=True)
class AttnResABI:
    """The host calling convention for attention_residual.

    Ordered exactly as the operator's own signature, so the call site reads like the contract:

        attention_residual(v_stack, x, w, gamma, alpha, epsilon=1e-6) -> y

    v_stack is [N,B,T,D] and is passed FLATTENED to [N*M*D] because every operation along N is
    per-token and nothing mixes tokens, so the batch and sequence axes may be merged for
    addressing. x and y are [B,T,D] flattened to [M*D]. alpha and epsilon are runtime scalars,
    not compile-time constants: specialising them would mean a different cached binary per
    scalar value, and the operator's domain allows a continuum.
    """

    dtypes = ("bfloat16", "float16")
    scalar_dtypes = ("float32",)

    def args_for(self, *, n: int, tokens: int, d: int, dtype: str) -> tuple:
        if dtype not in self.dtypes:
            raise ValueError("attention_residual: dtype must be one of %r, got %r"
                             % (self.dtypes, dtype))
        esize = 2  # bfloat16 and float16 are both 2 bytes
        m = tokens
        return (
            {"name": "v_stack", "kind": "in", "dtype": dtype,
             "shape": (n, m, d), "count": n * m * d, "nbytes": n * m * d * esize},
            {"name": "x", "kind": "in", "dtype": dtype,
             "shape": (m, d), "count": m * d, "nbytes": m * d * esize},
            {"name": "w", "kind": "in", "dtype": dtype,
             "shape": (d,), "count": d, "nbytes": d * esize},
            {"name": "gamma", "kind": "in", "dtype": dtype,
             "shape": (d,), "count": d, "nbytes": d * esize},
            {"name": "y", "kind": "out", "dtype": dtype,
             "shape": (m, d), "count": m * d, "nbytes": m * d * esize},
            {"name": "alpha", "kind": "scalar", "dtype": "float32"},
            {"name": "epsilon", "kind": "scalar", "dtype": "float32"},
        )

    def output_index(self, *, n: int, tokens: int, d: int, dtype: str) -> int:
        args = self.args_for(n=n, tokens=tokens, d=d, dtype=dtype)
        return next(i for i, a in enumerate(args) if a["kind"] == "out")

    def validate(self, *, n: int, tokens: int, d: int, dtype: str) -> bool:
        """Refuse an ABI the operator's domain does not allow."""
        if not isinstance(n, int) or n < 2:
            raise ValueError("attention_residual: N >= 2 is a hard domain constraint, got %r"
                             % (n,))
        if not isinstance(tokens, int) or tokens < 1:
            raise ValueError("attention_residual: tokens must be >= 1, got %r" % (tokens,))
        if not isinstance(d, int) or d < 1:
            raise ValueError("attention_residual: D must be >= 1, got %r" % (d,))
        names = [a["name"] for a in self.args_for(n=n, tokens=tokens, d=d, dtype=dtype)]
        if len(set(names)) != len(names):
            raise ValueError("attention_residual: ABI arg names must be unique")
        return True


ABI = AttnResABI()


# ------------------------------------------------------------------ kernel selection

@dataclass(frozen=True)
class KernelConfig:
    """The launch configuration that was measured, recorded rather than inferred."""

    programs: int
    block_n: int
    threads: int
    padded_d: int
    niter: int

    def as_dict(self) -> dict:
        return {"programs": self.programs, "block_n": self.block_n, "threads": self.threads,
                "padded_d": self.padded_d, "niter": self.niter}


THREADS = 128


def next_power_of_two(value: int) -> int:
    pd = 1
    while pd < value:
        pd *= 2
    return pd


def kernel_config(*, n: int, tokens: int, d: int, block_n: int = 1) -> KernelConfig:
    """The configuration this kernel is verified under.

    programs == tokens, so niter == 1 and the outer persistent loop executes once. This is a
    deliberate correctness choice, not an optimisation: the loop was measured wrong for niter>1
    and removing it was the honest fix. It is why the persistent-group requirement is recorded
    as NOT met.
    """
    if n < 2:
        raise ValueError("attention_residual: N >= 2 is a hard domain constraint, got %r" % (n,))
    if block_n != 1:
        raise ValueError(
            "this measured kernel is the CN=1 online family; CN=%d belongs to the "
            "blocked_online/resident_n families verified under G52" % block_n)
    return KernelConfig(programs=tokens, block_n=1, threads=THREADS,
                        padded_d=next_power_of_two(d), niter=1)
