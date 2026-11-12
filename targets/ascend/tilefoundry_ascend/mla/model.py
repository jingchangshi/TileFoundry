"""P6-01/G60: the canonical online fused MLA contract.

WHAT MLA IS HERE, in one line: attention where Q and K are each a CONCATENATION of a nope part and
a rope part, and V is numerically identical to K_nope. That last clause is the whole reason the
operator exists -- it is a shared latent KV cache -- and it is also the clause most likely to be
misread. The interface keeps v as its own argument for compatibility with generic attention APIs,
and the fixture normalises it to a copy of k_nope so that both the oracle and the measured kernel
receive the same legal input. So:

    v == k_nope  is a VALUE CONTRACT, not a POINTER ALIASING requirement.

A kernel may read k_nope twice and never touch the v buffer. Nothing in this gate may assume the
two tensors occupy the same memory, and nothing may assume they do not.

THE ALGEBRA, with the clause that is easiest to get wrong marked:

    Q   = concat(q_nope, q_rope)          D_qk = d_nope + d_rope
    K   = concat(k_nope, k_rope)          D_qk = d_nope + d_rope
    V   = k_nope                          D_v  = d_nope
    s   = (Q . K) * scaleValue            scaleValue <= 0 means 1/sqrt(D_qk)
    p   = softmax(s, along KV)            causal: j <= i + (Skv - S)
    y   = p @ V

ROPE IS ADDED ONCE, AFTER THE NOPE SUM. The reference core computes
    score = scale * (dot(q_nope, k_nope) + dot(q_rope, k_rope))
i.e. two partial sums added once. Applying the scale to one part only, or accumulating the rope
part per-tile and scaling it again, both produce a plausible number and a different operator.

THE CAUSAL OFFSET IS RIGHT-ALIGNED, AND 'j <= i' IS WRONG. The mask is j <= i + (Skv - S). At
S == Skv that reduces to the familiar lower triangle, which is exactly why a plain 'j <= i' passes
a prefill self-attention test and then silently fails every cross-attention and MTP case where
S < Skv. The offset is not a detail; it is the difference between attending to the right KV
positions and attending to the wrong ones.

GQA: hkv = hq // G with G = N_q / N_kv, so query head h reads KV head h // G. The golden folds the
group axis into the matmul M dimension rather than materialising K/V per query head; the mapping
is what matters, and it is recorded as a function rather than an inline division.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

BSND = "BSND"
BNSD = "BNSD"
LAYOUTS = (BSND, BNSD)

FLOAT16 = "float16"
BFLOAT16 = "bfloat16"
DTYPES = (FLOAT16, BFLOAT16)

VERIFIED = "VERIFIED"
UNVERIFIED = "UNVERIFIED"
BLOCKED_ENV = "BLOCKED_ENV"
NOT_RUN = "NOT_RUN"


class MLAError(RuntimeError):
    """A violated contract in the MLA description."""

    def __init__(self, code: str, message: str, subject: str = "", expected=None, actual=None):
        self.code = code
        self.message = message
        self.subject = subject
        self.expected = expected
        self.actual = actual
        super().__init__("%s: %s" % (code, message))

    def as_dict(self) -> dict:
        return {"code": self.code, "message": self.message, "subject": self.subject,
                "expected": self.expected, "actual": self.actual}


@dataclass(frozen=True)
class MLAShape:
    """The five-tensor shape, validated once so no downstream stage re-derives it wrongly.

    The layout is applied here rather than at each use site: under BNSD the sequence and head axes
    are swapped, and a kernel that applies the layout in some places and not others reads the
    right number of elements from the wrong positions -- which is silent, not a crash.
    """

    batch: int
    seq_q: int
    seq_kv: int
    heads_q: int
    heads_kv: int
    d_nope: int
    d_rope: int
    layout: str = BSND

    def __post_init__(self):
        for name, value, minimum in (
                ("batch", self.batch, 1), ("seq_q", self.seq_q, 1),
                ("seq_kv", self.seq_kv, 1), ("heads_q", self.heads_q, 1),
                ("heads_kv", self.heads_kv, 1), ("d_nope", self.d_nope, 1),
                ("d_rope", self.d_rope, 1)):
            if not isinstance(value, int) or value < minimum:
                raise MLAError("MLA_BAD_SHAPE", "%s must be an int >= %d" % (name, minimum),
                               name, ">= %d" % minimum, value)
        if self.layout not in LAYOUTS:
            raise MLAError("MLA_BAD_LAYOUT", "layout must be BSND or BNSD", "layout", LAYOUTS,
                           self.layout)
        # GQA requires the query heads to divide evenly into KV heads. A non-divisible ratio
        # means the head mapping is undefined, so it is refused here instead of producing a
        # kernel that reads a fractional head index.
        if self.heads_q % self.heads_kv != 0:
            raise MLAError(
                "MLA_NONDIVISIBLE_HEADS",
                "N_q must be a multiple of N_kv so each query head maps to exactly one KV head",
                "heads", "N_q %% N_kv == 0",
                "%d %% %d = %d" % (self.heads_q, self.heads_kv,
                                   self.heads_q % self.heads_kv))
        # NOTE: S > Skv is LEGAL when the attention is NOT causal -- that is exactly the
        # cross-attention shape (official case 4: S=256, Skv=128, is_causal=False), where every
        # query row attends to all Skv positions. The causal requirement is therefore checked in
        # MLASpec, where is_causal is known, not here where it is not.
        # Putting the check here was a real bug: it rejected a legal official case.

    @property
    def d_qk(self) -> int:
        return self.d_nope + self.d_rope

    @property
    def d_v(self) -> int:
        return self.d_nope

    @property
    def group(self) -> int:
        """G = N_q / N_kv: how many query heads share one KV head."""
        return self.heads_q // self.heads_kv

    def kv_head_of(self, query_head: int) -> int:
        """hkv = hq // G -- the GQA mapping, as a function because it IS the contract."""
        if query_head < 0 or query_head >= self.heads_q:
            raise MLAError("MLA_HEAD_OUT_OF_RANGE", "query head out of range", "query_head",
                           "0..%d" % (self.heads_q - 1), query_head)
        return query_head // self.group

    def tensor_shape(self, name: str) -> tuple:
        """The shape of one of the five tensors in the declared layout."""
        if name in ("q_nope", "q_rope"):
            d = self.d_nope if name == "q_nope" else self.d_rope
            return ((self.batch, self.seq_q, self.heads_q, d) if self.layout == BSND
                    else (self.batch, self.heads_q, self.seq_q, d))
        if name in ("k_nope", "k_rope", "v"):
            d = self.d_rope if name == "k_rope" else self.d_nope
            return ((self.batch, self.seq_kv, self.heads_kv, d) if self.layout == BSND
                    else (self.batch, self.heads_kv, self.seq_kv, d))
        if name == "out":
            return ((self.batch, self.seq_q, self.heads_q, self.d_v) if self.layout == BSND
                    else (self.batch, self.heads_q, self.seq_q, self.d_v))
        raise MLAError("MLA_UNKNOWN_TENSOR", "unknown tensor name", "name",
                       ("q_nope", "q_rope", "k_nope", "k_rope", "v", "out"), name)

    def as_dict(self) -> dict:
        return {
            "batch": self.batch, "seq_q": self.seq_q, "seq_kv": self.seq_kv,
            "heads_q": self.heads_q, "heads_kv": self.heads_kv,
            "d_nope": self.d_nope, "d_rope": self.d_rope,
            "d_qk": self.d_qk, "d_v": self.d_v, "group": self.group,
            "layout": self.layout,
        }


def resolve_scale(scale_value: float, d_qk: int) -> float:
    """scaleValue <= 0 means 1/sqrt(D_qk); otherwise the value is used as given.

    The sentinel is a real branch, not a default: passing 0.0 through as a scale would zero every
    score and produce a uniform softmax, which looks like a working kernel.
    """
    if d_qk < 1:
        raise MLAError("MLA_BAD_D_QK", "D_qk must be positive", "d_qk", ">= 1", d_qk)
    if scale_value is None:
        return 1.0 / (d_qk ** 0.5)
    if scale_value <= 0:
        return 1.0 / (d_qk ** 0.5)
    return float(scale_value)


def causal_limit(query_row: int, seq_q: int, seq_kv: int) -> int:
    """The last KV index a query row may attend to: i + (Skv - S), RIGHT-ALIGNED.

    A plain 'i' is the mistake this function exists to prevent. At S == Skv the two agree, so a
    self-attention prefill test cannot distinguish them; every cross-attention and MTP case can.
    """
    if seq_q > seq_kv:
        raise MLAError(
            "MLA_CAUSAL_NEEDS_S_LE_SKV",
            "the right-aligned causal offset requires S <= Skv", "seq",
            "S <= Skv", "S=%d Skv=%d" % (seq_q, seq_kv))
    if query_row < 0 or query_row >= seq_q:
        raise MLAError("MLA_ROW_OUT_OF_RANGE", "query row out of range", "query_row",
                       "0..%d" % (seq_q - 1), query_row)
    return query_row + (seq_kv - seq_q)


def is_causal_masked(query_row: int, kv_index: int, seq_q: int, seq_kv: int) -> bool:
    """True when (i, j) is masked. Stated so the twins and the checker share one definition."""
    return kv_index > causal_limit(query_row, seq_q, seq_kv)


@dataclass(frozen=True)
class MLASpec:
    """A fully-specified single-kernel MLA."""

    shape: MLAShape
    dtype: str = BFLOAT16
    scale_value: float = -1.0
    is_causal: bool = True
    kv_block: int = 64
    launches: int = 1
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        # The right-aligned causal offset needs S <= Skv; with S > Skv the offset Skv - S is
        # negative and some rows would attend to no positions. This belongs HERE and not in
        # MLAShape, because it is only a constraint WHEN the mask is applied: the official
        # cross-attention case has S=256 > Skv=128 and is perfectly legal because it is
        # non-causal. Checking it on the shape alone rejected that case.
        if self.is_causal and self.shape.seq_q > self.shape.seq_kv:
            raise MLAError(
                "MLA_CAUSAL_NEEDS_S_LE_SKV",
                "the right-aligned causal contract requires S <= Skv; with S > Skv the offset "
                "Skv - S is negative and rows would attend to no positions",
                "seq", "S <= Skv", "S=%d Skv=%d" % (self.shape.seq_q, self.shape.seq_kv))
        if self.dtype not in DTYPES:
            raise MLAError("MLA_BAD_DTYPE", "unsupported storage dtype", "dtype", DTYPES,
                           self.dtype)

    @property
    def scale(self) -> float:
        return resolve_scale(self.scale_value, self.shape.d_qk)

    def kv_blocks(self) -> int:
        return (self.shape.seq_kv + self.kv_block - 1) // self.kv_block

    def as_dict(self) -> dict:
        return {
            "shape": self.shape.as_dict(),
            "dtype": self.dtype,
            "scale_value": self.scale_value,
            "scale_resolved": self.scale,
            "scale_is_default": self.scale_value is None or self.scale_value <= 0,
            "is_causal": self.is_causal,
            "kv_block": self.kv_block,
            "kv_blocks": self.kv_blocks(),
            "launches": self.launches,
            "metadata": dict(self.metadata),
        }


def build_mla(*, batch: int, seq_q: int, seq_kv: int, heads_q: int, heads_kv: int,
              d_nope: int, d_rope: int, layout: str = BSND, dtype: str = BFLOAT16,
              scale_value: float = -1.0, is_causal: bool = True,
              kv_block: int = 64) -> MLASpec:
    """Build a spec, refusing every combination the operator's own domain forbids."""
    if dtype not in DTYPES:
        raise MLAError("MLA_BAD_DTYPE", "only the declared storage dtypes are in the domain",
                       "dtype", DTYPES, dtype)
    if not isinstance(kv_block, int) or kv_block < 1:
        raise MLAError("MLA_BAD_KV_BLOCK", "kv_block must be an int >= 1", "kv_block",
                       ">= 1", kv_block)
    shape = MLAShape(batch=batch, seq_q=seq_q, seq_kv=seq_kv, heads_q=heads_q,
                     heads_kv=heads_kv, d_nope=d_nope, d_rope=d_rope, layout=layout)
    return MLASpec(
        shape=shape, dtype=dtype, scale_value=scale_value, is_causal=is_causal,
        kv_block=kv_block,
        metadata={
            "kernel": "mla",
            "abi": ("mla(q_nope, q_rope, k_nope, k_rope, v, numKVHeads, scaleValue=-1.0, "
                    "inputLayout='BSND', is_causal=False)"),
            "golden": "fixtures/mla/golden.py",
            "v_contract": ("v == k_nope is a VALUE contract; a kernel may read k_nope twice "
                           "and must not assume pointer aliasing either way"),
            "rope_added_once": "score = scale * (dot(q_nope,k_nope) + dot(q_rope,k_rope))",
            "causal": "j <= i + (Skv - S), right-aligned",
            "gqa": "hkv = hq // G",
            "all_masked_row": "weights set to 0, not NaN",
            "device_executed": False,
        },
    )