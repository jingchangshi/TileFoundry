"""P5-01/G50: the attention_residual single-kernel contract and its three families.

WHY A SEPARATE MODULE FROM `mixed`. The MixedKernel package models TWO core domains. This
operator uses ONE: attention_residual is an AIV-only kernel, so modelling it as a MixedKernel
would add an empty Cube side purely to satisfy a shape -- which the backend spec forbids
outright ("attention_residual must not add empty Cube work to use dual_copy").

WHAT THIS MODULE OWNS, AND WHAT IT DOES NOT. It owns the ALGEBRA and the SCHEDULE facts: which
axes each reduction runs along, which state is FP32, which loads and stores exist, and what each
of the three families does differently. It does not compile or measure. Device results are
recorded separately in evidence/, and AttentionEvidence.is_gate_pass() requires them.

THE THREE FAMILIES ARE ONE ALGEBRA. online, blocked_online and resident_n differ only in CN, the
number of V blocks processed per running-state update:

    CN = 1      -> online          (update after every block)
    CN < N      -> blocked_online  (update every ceil(N/CN) blocks)
    CN = N      -> resident_n      (one update; block state IS the final state)

That is why one module can describe all three and why a per-family code path would be a bug:
the spec's own reference core implements exactly this parameterisation. G52 tests the extremes;
this module must not pre-judge which one wins on a shape.

THE TWO SCORE MODES ARE NOT EQUIVALENT, AND ARE NOT TREATED AS SUCH.

    literal_fp32: k = v / rms * gamma ; s = sum_d(k * w) / sqrt(D)
    factored_gw : gw = gamma * w     ; s = sum_d(v * gw) * rsqrt(q/D + eps) / sqrt(D)

These agree in REAL arithmetic and differ in FLOAT arithmetic: factored_gw hoists a per-element
division out of the reduction, so its rounding and its rsqrt are not the literal path's. The
module records them as distinct strategies with distinct provenance and never claims the second
reproduces the first bit-for-bit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# ------------------------------------------------------------------ vocabulary

ONLINE = "online"
BLOCKED_ONLINE = "blocked_online"
RESIDENT_N = "resident_n"
FAMILIES = (ONLINE, BLOCKED_ONLINE, RESIDENT_N)

LITERAL_FP32 = "literal_fp32"
FACTORED_GW = "factored_gw"
SCORE_MODES = (LITERAL_FP32, FACTORED_GW)

VERIFIED = "VERIFIED"
UNVERIFIED = "UNVERIFIED"
BLOCKED_ENV = "BLOCKED_ENV"
NOT_RUN = "NOT_RUN"


class AttentionResidualError(RuntimeError):
    """A violated contract in the attention_residual description."""

    def __init__(self, code: str, message: str, subject: str = "", expected=None, actual=None):
        self.code = code
        self.message = message
        self.subject = subject
        self.expected = expected
        self.actual = actual
        super().__init__("%s: %s" % (code, message))

    def as_dict(self) -> dict:
        return {
            "code": self.code,
            "message": self.message,
            "subject": self.subject,
            "expected": self.expected,
            "actual": self.actual,
        }


@dataclass(frozen=True)
class ReductionAxis:
    """Which axis a reduction runs along, and why that axis cannot be swapped.

    The golden's softmax is along N (the block/depth axis). Swapping it for D or for the token
    axis produces a well-formed but DIFFERENT operator that would still compile, still run, and
    still return the right shape -- which is exactly why the axis is a named field with a
    justification rather than a literal buried in a code body.
    """

    name: str
    axis: str
    reason: str


#: The reductions, with the axis each one is pinned to.
REDUCTIONS = (
    ReductionAxis(
        "rms_square_sum", "D",
        "step 1 normalises each (n,b,t) vector over its own D channels; a token-wise or "
        "block-wise sum would normalise the wrong thing"),
    ReductionAxis(
        "score_dot", "D",
        "step 2 is a dot of the normalised key with the pseudo-query w, which has length D"),
    ReductionAxis(
        "softmax_max_and_sum", "N",
        "step 3 is a softmax over the DEPTH/block axis; this is the operator's defining axis and "
        "the spec states it may not be moved to D or to the token axis"),
    ReductionAxis(
        "weighted_sum", "N",
        "step 4 weights the ORIGINAL (unnormalised) V by the depth probabilities"),
)


@dataclass(frozen=True)
class FP32State:
    """The state that must stay FP32, and why each entry cannot be narrowed.

    The golden computes every mathematical intermediate in FP32 and converts only the final
    output. This is not a preference: narrowing m loses the softmax shift, narrowing l loses the
    denominator, and narrowing acc accumulates rounding across blocks.
    """

    m: str = "float32"
    l: str = "float32"
    acc: str = "float32"
    score: str = "float32"

    def as_dict(self) -> dict:
        return {"m": self.m, "l": self.l, "acc": self.acc, "score": self.score}


@dataclass(frozen=True)
class AttentionResidualSpec:
    """A fully-specified single-kernel attention_residual.

    Constructed through build_attention_residual, which refuses illegal combinations. A bare
    dataclass would let a caller describe, say, resident_n with CN=1, and then two families
    would mean the same thing under different names.
    """

    family: str
    score_mode: str
    n: int
    tokens: int
    d: int
    dtype: str
    token_rows: int = 1
    block_n: int = 1
    d_chunk: int = 1024
    v_versions: int = 1
    state: FP32State = field(default_factory=FP32State)
    launches: int = 1
    metadata: dict = field(default_factory=dict)

    # -------------------------------------------------------------- derived facts

    def n_blocks(self) -> int:
        """How many V blocks the N axis is cut into."""
        return (self.n + self.block_n - 1) // self.block_n

    def groups(self) -> int:
        """How many token groups the M=B*T axis is cut into."""
        return (self.tokens + self.token_rows - 1) // self.token_rows

    def state_updates(self) -> int:
        """How many running-state merges each token group performs.

        This is the number that separates the families: online pays N merges, resident_n pays 1,
        blocked_online pays ceil(N/CN). Derived rather than stored so it cannot disagree with
        block_n.
        """
        return self.n_blocks()

    def padded_d(self) -> int:
        """D rounded up to a power of two.

        Not cosmetic: layout inference on this target refuses a shared buffer whose extent is
        not a power of two, which is why D=2047 and D=1000 fail to compile without it. The pad
        is invisible to the maths because every producer writes only indices < d and every
        reduction masks i < d.
        """
        pd = 1
        while pd < self.d:
            pd *= 2
        return pd

    def denormalises_d(self) -> bool:
        """Is the real D a power of two, so no D tail exists?"""
        return (self.d & (self.d - 1)) == 0

    def as_dict(self) -> dict:
        return {
            "family": self.family,
            "score_mode": self.score_mode,
            "n": self.n,
            "tokens": self.tokens,
            "d": self.d,
            "dtype": self.dtype,
            "token_rows": self.token_rows,
            "block_n": self.block_n,
            "d_chunk": self.d_chunk,
            "v_versions": self.v_versions,
            "n_blocks": self.n_blocks(),
            "groups": self.groups(),
            "state_updates": self.state_updates(),
            "padded_d": self.padded_d(),
            "d_tail": not self.denormalises_d(),
            "state": self.state.as_dict(),
            "launches": self.launches,
            "metadata": dict(self.metadata),
        }


def build_attention_residual(kernel: str = "online", *, n: int, tokens: int, d: int,
                             dtype: str = "bfloat16", family: str = ONLINE,
                             score_mode: str = LITERAL_FP32, token_rows: int = 1,
                             block_n: Optional[int] = None, d_chunk: int = 1024,
                             v_versions: int = 1) -> AttentionResidualSpec:
    """Build a spec, refusing every combination the operator's own domain forbids."""
    if family not in FAMILIES:
        raise AttentionResidualError(
            "ATTNRES_UNKNOWN_FAMILY",
            "family must be one of the three the spec requires", "family", FAMILIES, family)
    if score_mode not in SCORE_MODES:
        raise AttentionResidualError(
            "ATTNRES_UNKNOWN_SCORE_MODE",
            "score_mode must be one of the two recorded strategies", "score_mode",
            SCORE_MODES, score_mode)
    # N >= 2 is a HARD constraint of the definition, not a convenience: at N == 1 the softmax
    # over the depth axis is identically 1, the output stops depending on the score path, and
    # the operator is no longer the one being specified.
    if not isinstance(n, int) or n < 2:
        raise AttentionResidualError(
            "ATTNRES_N_BELOW_DOMAIN",
            "N >= 2 is a hard domain constraint: at N == 1 the depth softmax is identically 1 "
            "and the score path has no effect on the output", "n", "n >= 2", n)
    for name, value, minimum in (("tokens", tokens, 1), ("d", d, 1),
                                 ("token_rows", token_rows, 1), ("d_chunk", d_chunk, 1),
                                 ("v_versions", v_versions, 1)):
        if not isinstance(value, int) or value < minimum:
            raise AttentionResidualError(
                "ATTNRES_BAD_DIMENSION", "%s must be an int >= %d" % (name, minimum),
                name, ">= %d" % minimum, value)
    if dtype not in ("bfloat16", "float16"):
        raise AttentionResidualError(
            "ATTNRES_BAD_DTYPE", "only the declared storage dtypes are in the domain",
            "dtype", ("bfloat16", "float16"), dtype)

    if block_n is None:
        # The family's CANONICAL CN. online is CN=1 by definition; resident_n is CN=N; and
        # blocked_online must be a strict partial block, so it defaults to the largest power of
        # two strictly below N.
        if family == ONLINE:
            block_n = 1
        elif family == RESIDENT_N:
            block_n = n
        else:
            block_n = 1
            while block_n * 2 < n:
                block_n *= 2

    if block_n < 1 or block_n > n:
        raise AttentionResidualError(
            "ATTNRES_CN_OUT_OF_RANGE", "CN must satisfy 1 <= CN <= N", "block_n",
            "1..%d" % n, block_n)

    # Each family has ONE legal CN relationship. Enforcing it stops two family names describing
    # the same schedule, which would let a report claim three families while testing one.
    if family == ONLINE and block_n != 1:
        raise AttentionResidualError(
            "ATTNRES_CN_CONTRADICTS_FAMILY",
            "the online family updates the running state after EVERY V block, so CN must be 1",
            "block_n", 1, block_n)
    if family == RESIDENT_N and block_n != n:
        raise AttentionResidualError(
            "ATTNRES_CN_CONTRADICTS_FAMILY",
            "the resident_n family holds ALL N blocks on chip, so CN must equal N",
            "block_n", n, block_n)
    if family == BLOCKED_ONLINE and block_n >= n:
        raise AttentionResidualError(
            "ATTNRES_CN_CONTRADICTS_FAMILY",
            "blocked_online is the STRICT partial-block family; CN == N is resident_n",
            "block_n", "1 < CN < %d" % n, block_n)

    return AttentionResidualSpec(
        family=family,
        score_mode=score_mode,
        n=n,
        tokens=tokens,
        d=d,
        dtype=dtype,
        token_rows=token_rows,
        block_n=block_n,
        d_chunk=d_chunk,
        v_versions=v_versions,
        metadata={
            "kernel": kernel,
            "abi": "attention_residual(v_stack, x, w, gamma, alpha, epsilon=1e-6)",
            "golden": "fixtures/attention_residual/golden.py",
            "softmax_axis": "N",
            "aggregation_uses": "original V (not the RMS-normalised key)",
            "final_blend": "(1 - alpha) * x + alpha * agg",
            "device_executed": False,
            "sqrt_d_temperature": True,
            "rms_denominator": "real D, never a padded DP",
        },
    )

# ------------------------------------------------------------------ evidence

@dataclass(frozen=True)
class AttentionEvidence:
    """WHAT WAS ESTABLISHED, in fields that cannot be mistaken for each other.

    structural and device are separate because they answer different questions: the model and
    checker can establish internal consistency without executing anything, and only device
    execution can establish that the hardware produces the right numbers. A single aggregate
    boolean would let a structural pass read as a device pass -- the exact substitution this gate
    forbids -- so the type refuses to offer one.
    """

    kernel: str
    structural: str = UNVERIFIED
    device: str = BLOCKED_ENV
    structural_findings: tuple = ()
    device_reason: str = ""
    cases_passed: int = 0
    cases_total: int = 0
    measured: dict = field(default_factory=dict)

    def is_gate_pass(self) -> bool:
        """A gate PASS requires DEVICE evidence, never structure alone."""
        return self.structural == VERIFIED and self.device == VERIFIED

    def as_dict(self) -> dict:
        return {
            "kernel": self.kernel,
            "structural": self.structural,
            "device": self.device,
            "structural_findings": list(self.structural_findings),
            "device_reason": self.device_reason,
            "cases_passed": self.cases_passed,
            "cases_total": self.cases_total,
            "measured": dict(self.measured),
            "gate_pass": self.is_gate_pass(),
        }


def measured_attention_evidence(spec: AttentionResidualSpec) -> AttentionEvidence:
    """Return the evidence record for spec, backed by measurements actually taken.

    The numbers live in evidence/G50_DEVICE_EVIDENCE.json rather than being duplicated in code,
    so there is one place to check and one place a future run updates. If the file is missing or
    reports fewer than all 20 official cases, the device status is NOT promoted: absence of
    evidence stays absence of evidence.
    """
    import json
    from pathlib import Path

    from .hir import build_scheduled_hir, check_scheduled_hir
    from .twin import build_attnres_twin, check_attnres_conformance, parse_attnres_twin

    hir = build_scheduled_hir(spec)
    findings = list(check_scheduled_hir(hir))
    report = check_attnres_conformance(spec, parse_attnres_twin(build_attnres_twin(spec)))
    if report.status != "MATCHED":
        return AttentionEvidence(
            kernel=spec.metadata.get("kernel", "attention_residual"),
            structural=UNVERIFIED,
            structural_findings=tuple(findings) + tuple(report.failures),
            device_reason="the emitted kernel failed its own conformance check")
    findings.extend(report.findings)

    path = Path(__file__).resolve().parents[2] / "evidence" / "G50_DEVICE_EVIDENCE.json"
    if not path.exists():
        return AttentionEvidence(
            kernel=spec.metadata.get("kernel", "attention_residual"),
            structural=VERIFIED, structural_findings=tuple(findings),
            device_reason="no device evidence file at %s" % path)

    data = json.loads(path.read_text())
    cases = data.get("official_cases", {})
    passed = int(cases.get("passed", 0))
    total = int(cases.get("total", 0))
    if total != 20 or passed != total:
        return AttentionEvidence(
            kernel=spec.metadata.get("kernel", "attention_residual"),
            structural=VERIFIED, structural_findings=tuple(findings),
            cases_passed=passed, cases_total=total,
            device_reason="device evidence incomplete: %d of %d official cases passed"
                          % (passed, total))

    return AttentionEvidence(
        kernel=spec.metadata.get("kernel", "attention_residual"),
        structural=VERIFIED,
        device=VERIFIED,
        structural_findings=tuple(findings),
        device_reason="%d/%d official cases passed on %s"
                      % (passed, total, data.get("device", {}).get("arch", "?")),
        cases_passed=passed,
        cases_total=total,
        measured={
            "arch": data.get("device", {}).get("arch"),
            "toolchain": data.get("device", {}).get("toolchain"),
            "comparator": data.get("device", {}).get("comparator"),
            "thresholds": cases.get("thresholds"),
            "edge_tests": data.get("edge_tests", {}).get("result"),
        },
    )
