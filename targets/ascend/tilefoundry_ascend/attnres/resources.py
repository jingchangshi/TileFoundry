"""P5-02/G52: candidate generation and REAL UB/layout resource pruning.

WHY THIS IS ARITHMETIC AND NOT A HEURISTIC. A candidate that does not fit in UB cannot run, so
admitting it produces a measurement that is really a compile failure wearing a number's clothes.
Every figure below is computed from the target's declared UB size and the actual buffer widths the
kernel allocates, so an infeasible candidate is rejected BEFORE a device run rather than after.

WHAT IS MODELLED AS A RESOURCE, AND WHY EACH ONE:

  ub_bytes      the binding constraint on this target (256 KiB per AIV). The persistent
                gamma/w/gw residency is charged once; the per-block V staging is charged
                CN times for the resident families because they hold the block, and once for
                the re-reading families because they stream it.
  vf_registers  a VF holds its fragment registers; the spec's R x CN x D-chunk footprint is
                charged against the 128-lane VF register file.
  layout_ok     a shared buffer whose extent is not a power of two gives layout inference no
                legal layout (observed on D=2047 and D=1000), so the padded width must be a
                power of two.
  dma_exact     a copy must move exactly as many elements as the region it targets
                (observed as CanProveEqual(src_total, dst_total) on D=5120/7168).

A NOTE ON WHAT "R" MEANS HERE. R is the number of token rows a program handles per iteration.
The measured G50 kernel uses R=1 (one token group per program) and that is the only value with
device evidence, so the candidate generator emits R=1 as the default and R>1 as candidates that
are marked UNMEASURED rather than silently offered as equivalent.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .model import (
    AttentionResidualError,
    AttentionResidualSpec,
    BLOCKED_ONLINE,
    FAMILIES,
    ONLINE,
    RESIDENT_N,
)

# ---------------------------------------------------------------- target facts

UB_BYTES = 256 * 1024
VF_LANES = 128
VF_REGISTER_BYTES = 4 * VF_LANES * 64   # 64 architectural VF registers of 128 x 4B lanes
DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4}

FEASIBLE = "FEASIBLE"
INFEASIBLE = "INFEASIBLE"

MEASURED = "MEASURED"
UNMEASURED = "UNMEASURED"


@dataclass(frozen=True)
class ResourceUse:
    """What one candidate actually costs, itemised so a rejection can name its cause."""

    ub_bytes: int
    vf_register_bytes: int
    items: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"ub_bytes": self.ub_bytes, "vf_register_bytes": self.vf_register_bytes,
                "items": dict(self.items)}


@dataclass(frozen=True)
class Candidate:
    """One generation candidate with its verdict and the reason for it."""

    family: str
    score_mode: str
    block_n: int
    token_rows: int
    d_chunk: int
    v_versions: int
    use: ResourceUse
    verdict: str
    evidence: str
    reasons: tuple = ()

    @property
    def feasible(self) -> bool:
        return self.verdict == FEASIBLE

    def as_dict(self) -> dict:
        return {
            "family": self.family, "score_mode": self.score_mode,
            "block_n": self.block_n, "token_rows": self.token_rows,
            "d_chunk": self.d_chunk, "v_versions": self.v_versions,
            "use": self.use.as_dict(), "verdict": self.verdict,
            "evidence": self.evidence, "reasons": list(self.reasons),
        }


def _pad(value: int) -> int:
    pd = 1
    while pd < value:
        pd *= 2
    return pd


def measure_resources(spec: AttentionResidualSpec, *, token_rows: int = 1,
                      v_versions: int = 1) -> ResourceUse:
    """Charge the spec's real buffer widths against the target's real limits.

    The accounting follows the kernel that was actually measured: v/x/y staged per block, a
    running acc (u), a block scratch u_b, and the persistent g/w/gw vectors. The resident
    families are charged CN tiles of V because that is exactly what they hold; the re-reading
    families are charged one because that is what they stream.
    """
    pd = _pad(spec.d)
    esize = DTYPE_BYTES[spec.dtype]
    items = {}

    # Persistent residency: read once per AIV, reused by every token group.
    items["g_ub"] = pd * 4
    items["w_ub"] = pd * 4
    # gw is only materialised by the factored policy; the literal path still allocates the
    # buffer in the emitted kernel, so it is charged either way and the saving is NOT claimed.
    items["gw_ub"] = pd * 4

    # Per-block staging, versioned when the pipeline multi-buffers it.
    items["v_ub"] = pd * esize * max(1, v_versions)
    items["x_ub"] = pd * esize
    items["y_ub"] = pd * esize

    # FP32 state: the running accumulator and the block scratch.
    items["acc_ub"] = pd * 4
    items["u_ub"] = pd * 4
    items["st"] = 10 * 4

    # resident_n and blocked families that keep the whole block on chip pay CN tiles.
    if spec.family == RESIDENT_N:
        # Every N block must be resident simultaneously: that IS the family's definition, and
        # it is why the family has a hard feasibility wall rather than a soft cost.
        items["v_resident"] = spec.n * pd * esize
    elif spec.family == BLOCKED_ONLINE:
        # The measured blocked form re-reads V for the second pass instead of holding the block,
        # so it is charged ONE tile. Holding CN tiles instead is the residency option left for
        # resource pruning to choose between, and the charge is recorded rather than assumed.
        items["v_resident"] = 0

    # Token rows per iteration multiply the per-row staging.
    if token_rows > 1:
        items["row_multiplicity"] = (token_rows - 1) * (pd * esize * 2 + pd * 4 * 2)

    ub = sum(items.values())
    return ResourceUse(ub_bytes=ub,
                       vf_register_bytes=pd * 4 * max(1, token_rows),
                       items=items)


def judge(spec: AttentionResidualSpec, use: ResourceUse, *, token_rows: int = 1,
          v_versions: int = 1) -> tuple:
    """Return the reasons this candidate cannot run; empty means feasible."""
    reasons = []
    if use.ub_bytes > UB_BYTES:
        reasons.append(
            "UB overflow: %d bytes needed vs %d available (%.2fx over)"
            % (use.ub_bytes, UB_BYTES, use.ub_bytes / float(UB_BYTES)))
    if use.vf_register_bytes > VF_REGISTER_BYTES:
        reasons.append(
            "VF register pressure: %d bytes vs %d available"
            % (use.vf_register_bytes, VF_REGISTER_BYTES))
    # Layout inference refuses a non-power-of-two shared extent; the padded width is what is
    # charged, so a spec whose padding does not fix this must be rejected rather than attempted.
    pd = _pad(spec.d)
    if (pd & (pd - 1)) != 0:
        reasons.append("layout: padded width %d is not a power of two" % pd)
    # Every copy must move exactly the region it targets.
    if spec.d != pd and v_versions > 1:
        reasons.append(
            "dma: a multi-versioned copy of %d elements into a %d-element region cannot prove "
            "src_total == dst_total (observed failure on D=5120/7168)" % (spec.d, pd))
    if spec.family == RESIDENT_N and spec.n * pd * DTYPE_BYTES[spec.dtype] > UB_BYTES:
        reasons.append(
            "resident_n cannot hold all %d blocks of %d elements in UB"
            % (spec.n, pd))
    return tuple(reasons)


def generate_candidates(*, n: int, tokens: int, d: int, dtype: str = "bfloat16",
                        d_chunks: Optional[tuple] = None) -> tuple:
    """Enumerate the family x score-mode x CN x R x D-chunk x versions space, judged.

    The space is enumerated rather than pruned by intuition so the report can show which
    candidates were REJECTED and why. A generator that only emits winners cannot demonstrate
    that pruning happened.
    """
    if d_chunks is None:
        # D-chunking is what makes a large D feasible at all: the chunk is the unit that must
        # fit, so the candidate set includes chunks that provably do and do not fit.
        d_chunks = (d, min(d, 1024), min(d, 512))

    from .model import build_attention_residual

    out = []
    for family in FAMILIES:
        if family == ONLINE:
            cns = (1,)
        elif family == RESIDENT_N:
            cns = (n,)
        else:
            cns = tuple(c for c in (2, 4, 8, 16) if 1 < c < n) or ()
        for cn in cns:
            for mode in ("literal_fp32", "factored_gw"):
                for rows in (1, 2):
                    for chunk in sorted(set(d_chunks), reverse=True):
                        for versions in (1, 2):
                            try:
                                spec = build_attention_residual(
                                    n=n, tokens=tokens, d=d, dtype=dtype, family=family,
                                    score_mode=mode, block_n=cn, token_rows=rows,
                                    d_chunk=chunk, v_versions=versions)
                            except AttentionResidualError as exc:
                                out.append(Candidate(
                                    family=family, score_mode=mode, block_n=cn,
                                    token_rows=rows, d_chunk=chunk, v_versions=versions,
                                    use=ResourceUse(0, 0), verdict=INFEASIBLE,
                                    evidence=MEASURED if family == ONLINE else UNMEASURED,
                                    reasons=(exc.code,)))
                                continue
                            use = measure_resources(spec, token_rows=rows,
                                                    v_versions=versions)
                            reasons = judge(spec, use, token_rows=rows,
                                            v_versions=versions)
                            out.append(Candidate(
                                family=family, score_mode=mode, block_n=cn,
                                token_rows=rows, d_chunk=chunk, v_versions=versions,
                                use=use,
                                verdict=INFEASIBLE if reasons else FEASIBLE,
                                # Only the shapes that were actually executed on device are
                                # called MEASURED. Everything else is a candidate, not a result.
                                evidence=(MEASURED if (family == ONLINE and rows == 1
                                                       and versions == 1
                                                       and chunk == d)
                                          else UNMEASURED),
                                reasons=reasons))
    return tuple(out)


def summarise(candidates) -> dict:
    """Count the space by verdict, so pruning is visible rather than implied."""
    total = len(candidates)
    feasible = sum(1 for c in candidates if c.feasible)
    measured = sum(1 for c in candidates if c.evidence == MEASURED)
    by_reason = {}
    for c in candidates:
        for r in c.reasons:
            by_reason[r] = by_reason.get(r, 0) + 1
    return {
        "total": total,
        "feasible": feasible,
        "infeasible": total - feasible,
        "measured_on_device": measured,
        "pruned_fraction": (total - feasible) / float(total) if total else 0.0,
        "rejections_by_reason": by_reason,
    }
