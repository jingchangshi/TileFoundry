"""P7-01/G69: static legality and resource filter, with NAMED rejections.

WHY THIS IS ARITHMETIC AND NOT A HEURISTIC. A candidate that cannot fit in L1 or L0C cannot run,
so admitting it produces a measurement that is really a compile failure wearing a number's
clothes (the same argument attnres/resources.py makes for UB). Every charge below is computed
from a DECLARED target capacity and from the buffers the kernel actually allocates, so an
infeasible candidate is rejected BEFORE a device run and the rejection names the space, the
requirement and the limit.

THE SIX SPACES THAT ARE CHECKED, because "does it fit" is six different questions on this
target and conflating them hides which one bound:

  UB   256 KiB per AIV. Scores, P operand, contribution, U/prefix, output staging, m/l/alpha.
  L1   512 KiB per AIC. Q tiles, the versioned K/Rope ring, optional separate V staging, P ring.
  L0A  64 KiB  per AIC. GEMM left operands: the Q panel (QK) and the P tile (PV).
  L0B  64 KiB  per AIC. GEMM right operands: the K panel (QK) and V (PV).
  L0C  256 KiB per AIC. FP32 accumulators: score, PV contribution, and -- for the deferred route
       ONLY -- the whole pending U that the route exists to keep off UB.
  SS   3 KiB   per MIX group. Deferred-rescale predicate and per-row factor publication. An
       AIV-only kernel (attention_residual) charges zero here, and that zero is recorded rather
       than silently omitted.
  events  finite per-namespace id budgets from completion/model.py (AIV/AIC ready and free 16
       ids x 2 banks, state 8, exit 4, device cross 16). A ring with V versions needs one id per
       live generation, so the count is derived from the versions, not from a constant.

WHY SOME CHARGES ARE max() AND SOME ARE sum(). QK and PV are serialized on the same Cube (spec
4.4: the solver may interleave them but cannot assume one Cube runs both), so their GEMM operand
staging is charged as a MAXIMUM: charging the sum would reject configurations the hardware can
run. UB, L1 and L0C hold simultaneously live objects whose lifetimes genuinely overlap across a
pipelined iteration, so those are charged as SUMS. Each Judgement records which rule applied
where, so a reader can disagree with a specific choice instead of the whole number.

WHAT THIS FILTER IS NOT. It does not compile, does not check the final lowering's actual
allocation, and does not decide numerics. Spec 8.4's pipeline is: static legality/resource ->
compile -> original comparator -> device timing. A FEASIBLE judgement is permission to compile,
never evidence that anything works. And because the real lowering may add scratch, every
FEASIBLE candidate is marked lower_bound_only: the numbers here are the author's declared
buffers, not a measurement of what TileLang allocated.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

from ..attnres.model import AttentionResidualError
from ..attnres.resources import measure_resources as measure_ar_ub
from ..attnres.resources import _pad as _ar_pad
from ..completion.model import ASCEND_EVENT_FACTS, AIC, AIV, DEVICE, KIND_CROSS, KIND_EXIT
from ..completion.model import KIND_FREE, KIND_READY, KIND_STATE
from ..memory import SPACE_FACTS
from . import generator as gen

FEASIBLE = "FEASIBLE"
INFEASIBLE = "INFEASIBLE"
#: The evidence column: what has actually been established about this candidate, separately from
#: whether it fits. A feasible candidate is not a result.
UNMEASURED = "UNMEASURED"
MEASURED = "MEASURED"
NUMERIC_FAILED = "NUMERIC_FAILED"
COMPILE_FAILED = "COMPILE_FAILED"
TIMEOUT = "TIMEOUT"

DTYPE_BYTES = {gen.FLOAT16: 2, gen.BFLOAT16: 2, "float32": 4}
#: SS facts from the target record (case/mla.md section 3: 3 KiB model limit, 32 B alignment).
SS_BYTES = 3 * 1024
SS_ALIGN = 32
#: The 128-column score packing path needs two 64-lane vector halves, i.e. at least 32 rows.
MIN_SCORE_ROWS = 32
#: Declared bound on |V| used by the theta math and the accumulator charge, NOT a measured value.
V_ABS_MAX = 8.0


class FilterError(RuntimeError):
    def __init__(self, code, message, subject="", expected=None, actual=None):
        self.code = code
        self.message = message
        self.subject = subject
        self.expected = expected
        self.actual = actual
        super().__init__("%s: %s" % (code, message))

    def as_dict(self):
        return {"code": self.code, "message": self.message, "subject": self.subject,
                "expected": self.expected, "actual": self.actual}


def _digest(payload) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                     default=str).encode()).hexdigest()[:16]


def _align_up(value: int, alignment: int) -> int:
    return int(math.ceil(float(value) / alignment) * alignment) if value > 0 else 0


# ------------------------------------------------------------------ target limits

@dataclass(frozen=True)
class TargetLimits:
    """The capacities the filter judges against, each with the record it came from.

    These are DECLARED nominal capacities, not usable amounts: the memory facts explicitly leave
    usable_bytes unmeasured, so a candidate that fits exactly is reported as fitting NOMINALLY
    and never as "will run". That distinction is carried in the reason text of any rejection and
    in the lower_bound_only flag of every feasible judgement.
    """

    ub_bytes: int = SPACE_FACTS["ascend.ub"].nominal_bytes
    l1_bytes: int = SPACE_FACTS["ascend.l1"].nominal_bytes
    l0a_bytes: int = SPACE_FACTS["ascend.l0a"].nominal_bytes
    l0b_bytes: int = SPACE_FACTS["ascend.l0b"].nominal_bytes
    l0c_bytes: int = SPACE_FACTS["ascend.l0c"].nominal_bytes
    ss_bytes: int = SS_BYTES
    aiv_ready_ids: int = 32
    aiv_free_ids: int = 32
    aiv_state_ids: int = 8
    aiv_exit_ids: int = 4
    aic_ready_ids: int = 32
    aic_free_ids: int = 32
    aic_state_ids: int = 8
    aic_exit_ids: int = 4
    cross_ids: int = 16
    source: str = ("memory.SPACE_FACTS nominal capacities (3510 records) + completion."
                   "ASCEND_EVENT_FACTS declared id budgets + mla.md SS 3KiB model limit")

    def space_limit(self, space: str) -> int:
        table = {"ub": self.ub_bytes, "l1": self.l1_bytes, "l0a": self.l0a_bytes,
                 "l0b": self.l0b_bytes, "l0c": self.l0c_bytes, "ss": self.ss_bytes}
        if space not in table:
            raise FilterError("FILTER_UNKNOWN_SPACE", "no declared capacity for this space",
                              "space", sorted(table), space)
        return table[space]

    def event_limit(self, core: str, kind: str) -> int:
        table = {
            (AIV, KIND_READY): self.aiv_ready_ids, (AIV, KIND_FREE): self.aiv_free_ids,
            (AIV, KIND_STATE): self.aiv_state_ids, (AIV, KIND_EXIT): self.aiv_exit_ids,
            (AIC, KIND_READY): self.aic_ready_ids, (AIC, KIND_FREE): self.aic_free_ids,
            (AIC, KIND_STATE): self.aic_state_ids, (AIC, KIND_EXIT): self.aic_exit_ids,
            (DEVICE, KIND_CROSS): self.cross_ids,
        }
        if (core, kind) not in table:
            raise FilterError("FILTER_UNKNOWN_EVENT_NAMESPACE",
                              "no declared id budget for this namespace", "namespace",
                              sorted(table), (core, kind))
        return table[(core, kind)]

    def as_dict(self) -> dict:
        return {
            "ub_bytes": self.ub_bytes, "l1_bytes": self.l1_bytes,
            "l0a_bytes": self.l0a_bytes, "l0b_bytes": self.l0b_bytes,
            "l0c_bytes": self.l0c_bytes, "ss_bytes": self.ss_bytes,
            "events": {"aiv_ready": self.aiv_ready_ids, "aiv_free": self.aiv_free_ids,
                       "aiv_state": self.aiv_state_ids, "aiv_exit": self.aiv_exit_ids,
                       "aic_ready": self.aic_ready_ids, "aic_free": self.aic_free_ids,
                       "aic_state": self.aic_state_ids, "aic_exit": self.aic_exit_ids,
                       "cross": self.cross_ids},
            "source": self.source,
        }


def target_limits() -> TargetLimits:
    """The limits, taken from the target records rather than typed in here."""
    events = {}
    for pool in ASCEND_EVENT_FACTS.pools:
        events[(pool.core_domain, pool.kind)] = int(pool.capacity) * int(pool.banks)
    return TargetLimits(
        aiv_ready_ids=events[(AIV, KIND_READY)], aiv_free_ids=events[(AIV, KIND_FREE)],
        aiv_state_ids=events[(AIV, KIND_STATE)], aiv_exit_ids=events[(AIV, KIND_EXIT)],
        aic_ready_ids=events[(AIC, KIND_READY)], aic_free_ids=events[(AIC, KIND_FREE)],
        aic_state_ids=events[(AIC, KIND_STATE)], aic_exit_ids=events[(AIC, KIND_EXIT)],
        cross_ids=events[(DEVICE, KIND_CROSS)])

# ------------------------------------------------------------------ records

@dataclass(frozen=True)
class Charge:
    """One allocation, named, with the space it lives in and the rule that combined it.

    The rule is 'sum' or 'max' and is recorded per item group so the aggregation is auditable: a
    reader who thinks the Q and P GEMM operand staging DO overlap can see exactly which number to
    change rather than rejecting the whole model.
    """

    item: str
    space: str
    bytes: int
    rule: str = "sum"
    versions: int = 1
    note: str = ""

    def as_dict(self) -> dict:
        return {"item": self.item, "space": self.space, "bytes": self.bytes, "rule": self.rule,
                "versions": self.versions, "note": self.note}


@dataclass(frozen=True)
class Rejection:
    """A named refusal: a stable code, the space, and the exact arithmetic.

    Codes are the contract; the message is for a human. Two candidates rejected for different
    reasons must not share a code, which is why legality codes are separate from capacity codes.
    """

    code: str
    space: str
    detail: str
    used: Optional[int] = None
    limit: Optional[int] = None

    def as_dict(self) -> dict:
        return {"code": self.code, "space": self.space, "detail": self.detail,
                "used": self.used, "limit": self.limit}


@dataclass(frozen=True)
class ResourceUse:
    charges: tuple
    event_use: dict = field(default_factory=dict)
    notes: tuple = ()

    def total(self, space: str) -> int:
        """The charge for one space. Items are summed unless their rule says they share time."""
        summed = 0
        maxed = {}
        for charge in self.charges:
            if charge.space != space:
                continue
            if charge.rule == "max":
                maxed[charge.item] = maxed.get(charge.item, 0) + charge.bytes
            else:
                summed += charge.bytes
        return summed + sum(maxed.values())

    def spaces(self) -> tuple:
        return tuple(sorted({c.space for c in self.charges}))

    def items(self) -> dict:
        out = {}
        for charge in self.charges:
            out[charge.item] = out.get(charge.item, 0) + charge.bytes
        return out

    def as_dict(self) -> dict:
        totals = {space: self.total(space) for space in self.spaces()}
        return {"totals_by_space": totals, "event_use": dict(self.event_use),
                "charges": [c.as_dict() for c in self.charges],
                "max_rules": [c.item for c in self.charges if c.rule == "max"],
                "notes": list(self.notes)}


@dataclass(frozen=True)
class Judgement:
    """The verdict for one candidate, with everything a report needs to justify it."""

    key: str
    operator: str
    family: str
    verdict: str
    evidence: str
    use: ResourceUse
    rejections: tuple = ()
    origin: str = gen.ORIGIN_SEED
    parent: str = ""
    changed: str = ""
    notes: tuple = ()

    @property
    def feasible(self) -> bool:
        return self.verdict == FEASIBLE

    def as_dict(self) -> dict:
        return {"key": self.key, "operator": self.operator, "family": self.family,
                "verdict": self.verdict, "evidence": self.evidence,
                "origin": self.origin, "parent": self.parent, "changed": self.changed,
                "rejections": [r.as_dict() for r in self.rejections],
                "use": self.use.as_dict(), "notes": list(self.notes),
                "lower_bound_only": True}


# ------------------------------------------------------------------ MLA charges

def _mla_effective_panel(candidate, d_nope: int) -> int:
    """The panel width a kernel can actually use: the candidate's panel if it tiles D_nope.

    When it does not, the charge uses the widest declared panel that DOES tile D_nope -- but the
    candidate is still rejected by mla_legality. Charging the tiling width keeps the resource
    number meaningful for a rejected candidate instead of reporting a buffer the kernel would
    never allocate.
    """
    if d_nope % candidate.output_panel == 0:
        return candidate.output_panel
    if d_nope % 128 == 0:
        return 128
    if d_nope % 64 == 0:
        return 64
    return d_nope


def mla_resources(candidate, spec, *, limits: Optional[TargetLimits] = None) -> ResourceUse:
    """Charge the MLA kernel's real buffers, parameterised by the candidate's decisions.

    The buffer list follows the G60 kernel's allocation sites one-for-one (q, qr, k, kr, p, score,
    pv, pa, vb, qa, kb, su, pu, tmp, out, m, l, alpha) plus the deferred route's pending L0C and
    UB prefix. Each note names its site, so a charge that looks wrong can be traced to an actual
    allocation instead of being argued about in aggregate.
    """
    limits = limits or target_limits()
    shape = spec.shape
    d_nope, d_rope = shape.d_nope, shape.d_rope
    d_qk = d_nope + d_rope
    es = DTYPE_BYTES[spec.dtype]
    panel = _mla_effective_panel(candidate, d_nope)
    shard_width = int(math.ceil(d_nope / float(candidate.output_shards)))
    # A shard owns a whole number of GEMM panels so its MMA operands are complete tiles.
    shard_cols = int(math.ceil(shard_width / float(panel)) * panel)
    shard_cols = min(shard_cols, d_nope)
    bq, bc = candidate.block_q, candidate.block_c
    half = bq // 2
    kv_versions = candidate.kv_versions
    p_versions = candidate.p_versions
    p_copies = 2 if candidate.precision == gen.PRECISION_HI_LO else 1
    charges = []

    # ---- L1 (AIC): Q reuse, the versioned KV ring, and the P hand-off ring.
    charges.append(Charge("q_l1", "l1", bq * d_qk * es,
                          note="one Q tile spanning nope+rope, reused across every KV tile"))
    charges.append(Charge("qr_l1", "l1", bq * d_rope * es,
                          note="Q_rope tile, appended once after the nope panels"))
    charges.append(Charge("kv_l1", "l1", kv_versions * bc * d_qk * es, versions=kv_versions,
                          note="versioned K_nope+K_rope ring; K and V share the tile when "
                               "kv_reuse is on"))
    if not candidate.kv_reuse:
        charges.append(Charge("v_l1", "l1", kv_versions * bc * d_nope * es,
                              versions=kv_versions,
                              note="separate V staging: V is read as its own tensor, so the "
                                   "K slot's last reader is PV and its FREE moves earlier"))
    charges.append(Charge("p_l1", "l1", p_versions * bq * bc * es * p_copies,
                          versions=p_versions,
                          note="P ring handed from the two AIV halves to the Cube; hi/lo "
                               "doubles it because both components are operands"))

    # ---- L0A / L0B (AIC): GEMM operands. QK and PV share the Cube, so these are max-ruled.
    charges.append(Charge("qa_l0a", "l0a", bq * panel * es, rule="max",
                          note="Q panel staged for QK; serialized with the PV operands on the "
                               "same Cube, hence the max rule"))
    charges.append(Charge("pa_l0a", "l0a", bq * bc * es * p_copies, rule="max",
                          note="P tile staged for PV (two tiles for hi/lo)"))
    charges.append(Charge("kb_l0b", "l0b", bc * panel * es, rule="max",
                          note="K panel staged for QK"))
    charges.append(Charge("vb_l0b", "l0b", min(panel, shard_cols) * bc * es, rule="max",
                          note="V staged for PV one PANEL at a time: PV is (BM x BC) @ (BC x BD), "
                               "so only the panel's columns are resident, not the whole shard"))

    # ---- L0C (AIC): FP32 accumulators.
    charges.append(Charge("score_l0c", "l0c", bq * bc * 4,
                          note="FP32 QK accumulator for one KV tile"))
    charges.append(Charge("pv_l0c", "l0c", bq * min(panel, shard_cols) * 4,
                          note="FP32 PV accumulator for one output-column panel"))
    if candidate.rescale == gen.RESCALE_DEFERRED:
        charges.append(Charge("pending_l0c", "l0c", bq * shard_cols * 4,
                              note="DEFERRED ONLY: the pending U that the route exists to keep "
                                   "in L0C instead of dragging through UB every tile"))

    # ---- UB (per AIV, half the query tile).
    charges.append(Charge("score_ub", "ub", half * bc * 4,
                          note="FP32 scores delivered by the L0C->UB dual copy"))
    charges.append(Charge("p_ub", "ub", half * bc * es * p_versions, versions=p_versions,
                          note="P operand before the UB->L1 pack"))
    charges.append(Charge("c_ub", "ub", half * min(panel, shard_cols) * 4,
                          note="FP32 PV contribution between FIX and the vector update; one PANEL "
                               "wide, which is what the dual copy delivers per call"))
    charges.append(Charge("u_ub", "ub", half * shard_cols * 4,
                          note="FP32 U accumulator spanning the program's own output columns; this "
                               "IS the output staging buffer, so it is charged once, not once per "
                               "name it is called by"))
    charges.append(Charge("cast_ub", "ub", half * shard_cols * es,
                          note="storage-dtype cast staging for the final store, alongside U and "
                               "not in addition to it"))
    charges.append(Charge("state_ub", "ub", bq * 4 * 3,
                          note="m, l and alpha per row, all FP32, all needed by the update"))
    if candidate.rescale == gen.RESCALE_DEFERRED:
        charges.append(Charge("prefix_ub", "ub", half * shard_cols * 4,
                              note="DEFERRED ONLY: drained prefix kept on the AIV side"))
        charges.append(Charge("factor_ub", "ub", bq * 4,
                              note="DEFERRED ONLY: per-row rescale factor for this boundary"))

    # ---- SS (MIX group scalar storage). Zero for canonical, and that zero is stated.
    ss_items = []
    if candidate.rescale == gen.RESCALE_DEFERRED:
        # Two AIVs publish disjoint bytes; the AIC reads both and makes the group decision.
        ss_items.append(Charge("ss_predicate", "ss", _align_up(8, SS_ALIGN) * 2,
                               note="each AIV publishes {needs_change, max_delta}; the AIC "
                                    "reads both and ORs them into the group predicate"))
        ss_items.append(Charge("ss_factor", "ss", _align_up(bq * 4, SS_ALIGN),
                               note="per-row rescale factor, published once per boundary and "
                                    "held until every consumer of that anchor retires"))
        ss_items.append(Charge("ss_epoch", "ss", _align_up(8, SS_ALIGN),
                               note="anchor generation tag, so a stale predicate cannot be "
                                    "mistaken for the current boundary"))
        charges.extend(ss_items)

    events = _mla_events(candidate)
    notes = []
    if not ss_items:
        notes.append("SS: canonical accumulation keeps U on UB/L0C and publishes nothing into "
                     "group scalar storage, so its SS charge is genuinely zero, not unmodelled")
    notes.append("L0A/L0B: QK and PV operand staging are charged as a maximum because the two "
                 "GEMMs are serialized on one Cube; a solver that keeps both resident would make "
                 "the sum the binding figure")
    notes.append("UB/L1/L0C: charged as sums because their lifetimes overlap across a pipelined "
                 "iteration; the final lowering may add scratch, so these are lower bounds")
    notes.append("capacities are NOMINAL target records; usable_bytes is unmeasured on this "
                 "target, so a candidate that fits nominally is not thereby known to run")
    return ResourceUse(charges=tuple(charges), event_use=events, notes=tuple(notes))


def _mla_events(candidate) -> dict:
    """Event ids the ring protocol needs, per namespace.

    One ready and one free id PER LIVE GENERATION for every ring, plus the state channels that
    order alpha against the U update and the P hand-off across cores. Counting ids per generation
    rather than as a constant is what makes "the pool is exhausted" a real diagnostic instead of a
    number that only ever grows.
    """
    kv = candidate.kv_versions
    p = candidate.p_versions
    score = 2   # the score ring is double-buffered by the L0C->UB dual copy
    pending_extra = 2 if candidate.rescale == gen.RESCALE_DEFERRED else 0
    aiv = {KIND_READY: kv + p + score, KIND_FREE: kv + p + score,
           KIND_STATE: 2 + (1 if candidate.rescale == gen.RESCALE_DEFERRED else 0),
           KIND_EXIT: 1}
    aic = {KIND_READY: kv + p, KIND_FREE: kv + p,
           KIND_STATE: 2 + pending_extra, KIND_EXIT: 1}
    cross = {KIND_CROSS: 2 + pending_extra}
    return {"aiv": aiv, "aic": aic, "device": cross}

# ------------------------------------------------------------------ MLA legality

def mla_legality(candidate, spec) -> tuple:
    """Structural refusals, each quoting the implemented constraint it violates.

    These are not preferences. Each code below corresponds to a limit of the schedule or kernel
    that exists (or a limit the spec states), so a candidate rejected here would either not
    compile or would compute a different operator. Rejecting them statically is what keeps a
    compile failure out of the timing table.
    """
    out = []
    shape = spec.shape
    group = shape.group
    bq, bc = candidate.block_q, candidate.block_c

    # The score path packs 128 KV columns into two 64-lane vector halves and converts them with a
    # 16-lane narrow pattern; fewer than 32 query rows leaves the second half of the pack empty.
    if bq < MIN_SCORE_ROWS:
        out.append(Rejection(
            "LEGAL_BQ_BELOW_SCORE_ROWS", "schedule",
            "block_q=%d < %d: the 128-column score packing path needs two 64-lane row halves, so "
            "a tile shorter than %d rows has no legal packing"
            % (bq, MIN_SCORE_ROWS, MIN_SCORE_ROWS), bq, MIN_SCORE_ROWS))
    if bq not in gen.BQ_CHOICES:
        out.append(Rejection(
            "LEGAL_BQ_NOT_IN_DECLARED_SET", "schedule",
            "block_q=%d is outside the generated set %s" % (bq, list(gen.BQ_CHOICES)),
            bq, list(gen.BQ_CHOICES)))
    if group % bq:
        out.append(Rejection(
            "LEGAL_GROUP_NOT_TILED", "schedule",
            "G=N_q/N_kv=%d is not a multiple of block_q=%d, so a query tile would straddle KV "
            "groups and read two different KV heads" % (group, bq), group, bq))

    # The KV loop advances by one packed tile and the implemented packing is 128 columns wide.
    if shape.seq_kv % bc:
        out.append(Rejection(
            "LEGAL_SEQ_KV_NOT_MULTIPLE_OF_BC", "schedule",
            "S_kv=%d is not a multiple of block_c=%d and the packed score loop has no masked "
            "KV tail" % (shape.seq_kv, bc), shape.seq_kv, bc))
    if bc != 128:
        out.append(Rejection(
            "LEGAL_BC_NOT_IMPLEMENTED", "schedule",
            "block_c=%d: the implemented score/pack path packs exactly 128 KV columns (two "
            "64-lane halves plus a 128-lane merge); a %d-column tile needs a packing path that "
            "does not exist, so this is a search-space entry rather than a runnable "
            "configuration" % (bc, bc), bc, 128))

    # GEMM panels must tile D_nope exactly; a short tiling silently computes fewer columns.
    if shape.d_nope % candidate.output_panel:
        out.append(Rejection(
            "LEGAL_PANEL_NOT_TILING_D_NOPE", "schedule",
            "D_nope=%d is not divisible by the output panel %d, and a panel that does not tile "
            "D silently drops the trailing columns instead of failing"
            % (shape.d_nope, candidate.output_panel), shape.d_nope, candidate.output_panel))
    if candidate.output_panel % 64:
        out.append(Rejection(
            "LEGAL_PANEL_NOT_MULTIPLE_OF_64", "schedule",
            "output panel %d is not a multiple of the 64-wide MMA panel"
            % candidate.output_panel, candidate.output_panel, 64))
    if shape.d_rope != 64:
        out.append(Rejection(
            "LEGAL_D_ROPE_MUST_BE_64", "schedule",
            "D_rope=%d: the rope GEMM is one 64-wide panel" % shape.d_rope, shape.d_rope, 64))

    # The deferred route's theta must be legal for the P operand AND the FP32 accumulator.
    if candidate.rescale == gen.RESCALE_DEFERRED:
        bound = gen.theta_legal_bound(spec.dtype, block_c=bc, v_abs_max=V_ABS_MAX)
        if candidate.theta > bound + 1e-9:
            out.append(Rejection(
                "LEGAL_THETA_EXCEEDS_OPERAND_BOUND", "numerics",
                "theta=%.6f exceeds the legal bound %.6f for P dtype %s at BC=%d: a tile of P "
                "would exceed the operand's finite range (or the FP32 accumulator's headroom, "
                "whichever binds) and every contribution from it would be inf"
                % (candidate.theta, bound, spec.dtype, bc), round(candidate.theta, 9),
                round(bound, 9)))

    # A shard must own whole MMA panels, and the shards must cover every real column once.
    shard_width = int(math.ceil(shape.d_nope / float(candidate.output_shards)))
    if shard_width % 64:
        out.append(Rejection(
            "LEGAL_SHARD_WIDTH_NOT_PANEL", "schedule",
            "sharding D_nope=%d into %d gives a %d-column shard, not a multiple of 64, so a "
            "program would own a partial MMA panel"
            % (shape.d_nope, candidate.output_shards, shard_width), shard_width, 64))
    if candidate.output_shards > 1:
        covered = shard_width * candidate.output_shards
        if covered < shape.d_nope:
            out.append(Rejection(
                "LEGAL_SHARD_UNDERCOVERS", "schedule",
                "%d shards of %d columns cover %d < D_nope=%d"
                % (candidate.output_shards, shard_width, covered, shape.d_nope),
                covered, shape.d_nope))
    return tuple(out)


def _capacity_rejections(use: ResourceUse, limits: TargetLimits) -> list:
    """One named rejection per space that overflows, or per namespace that exhausts its ids."""
    out = []
    for space in ("ub", "l1", "l0a", "l0b", "l0c", "ss"):
        used = use.total(space)
        limit = limits.space_limit(space)
        if used > limit:
            out.append(Rejection(
                "RES_%s_OVERFLOW" % space.upper(), space,
                "%s overflow: %d bytes required vs %d nominal, %.2fx over"
                % (space, used, limit, used / float(limit)), used, limit))
    for core in sorted(use.event_use):
        for kind in sorted(use.event_use[core]):
            count = use.event_use[core][kind]
            if not count:
                # A zero is a real statement (an AIV-only kernel needs no Cube events); it is
                # skipped only because 0 ids can never exhaust a declared capacity.
                continue
            limit = limits.event_limit(core, kind)
            if count > limit:
                out.append(Rejection(
                    "RES_EVENT_%s_%s_EXHAUSTED" % (core.upper(), kind.upper()), "events",
                    "%s/%s event pool cannot hold %d concurrent ids (capacity %d); the ring "
                    "would have to replay a generation, which is a protocol violation rather "
                    "than an optimisation" % (core, kind, count, limit), count, limit))
    return out


# ------------------------------------------------------------------ MLA judgement

def judge_mla(candidate, spec, *, limits: Optional[TargetLimits] = None) -> Judgement:
    """Static verdict for one MLA candidate: legality first, then the six spaces."""
    limits = limits or target_limits()
    use = mla_resources(candidate, spec, limits=limits)
    rejections = list(mla_legality(candidate, spec))
    rejections.extend(_capacity_rejections(use, limits))
    notes = list(use.notes)
    if candidate.rescale == gen.RESCALE_DEFERRED:
        bound = gen.theta_legal_bound(spec.dtype, block_c=candidate.block_c, v_abs_max=V_ABS_MAX)
        notes.append("theta legal bound for %s P at BC=%d is %.6f; this candidate uses %.6f"
                     % (spec.dtype, candidate.block_c, bound, candidate.theta))
    return Judgement(key=candidate.key(), operator="mla", family=candidate.family,
                     verdict=FEASIBLE if not rejections else INFEASIBLE,
                     evidence=UNMEASURED, use=use, rejections=tuple(rejections),
                     origin=candidate.origin, parent=candidate.parent,
                     changed=candidate.changed, notes=tuple(notes))


# ------------------------------------------------------------------ AR charges

def attention_residual_resources(candidate, *, limits: Optional[TargetLimits] = None):
    """Charge attention_residual, reusing the G52 UB/VF model and adding the other spaces.

    The UB and VF figures come from attnres/resources.py UNCHANGED (one model, one place to
    disagree with), and this function adds what that model deliberately did not cover: the L1
    staging of the V ring, the event ids the ring needs, and the explicit ZERO charges for L0 and
    SS. An AIV-only kernel allocates no Cube operands and publishes nothing into group scalar
    storage, and stating those zeros is what stops "unmodelled" from looking like "free".
    """
    limits = limits or target_limits()
    spec = candidate.build_spec()
    ub_use = measure_ar_ub(spec, token_rows=candidate.token_rows,
                           v_versions=candidate.v_versions)
    pd = _ar_pad(candidate.d)
    es = DTYPE_BYTES.get(candidate.dtype, 2)
    charges = []
    for item, value in ub_use.items.items():
        charges.append(Charge("ub_%s" % item, "ub", int(value),
                              note="inherited unchanged from attnres/resources."
                                   "measure_resources"))
    # L1: the V ring is what pipeline depth actually costs. x/y are staged per token group; the
    # persistent gamma/w/gw residency is charged once because that is the G51 schedule.
    charges.append(Charge("v_ring_l1", "l1", candidate.v_versions * pd * es,
                          versions=candidate.v_versions,
                          note="versioned V tile ring; the family decides how many blocks live "
                               "on chip but the ring decides how many are in flight"))
    charges.append(Charge("xy_l1", "l1", (pd * es) * 2,
                          note="x and y staging for one token group"))
    charges.append(Charge("persistent_l1", "l1", pd * es * 3,
                          note="persistent gamma/w and the factored gw copy, hoisted per AIV and "
                               "reused across every token group"))
    if candidate.prefetch != gen.PREFETCH_NONE:
        charges.append(Charge("prefetch_l1", "l1", pd * es,
                              note="the additional in-flight tile the prefetch recipe names"))
    charges.append(Charge("spill_ub", "ub", int(ub_use.vf_register_bytes) // 64,
                          note="SIMD spill allowance derived from the VF footprint the G52 model "
                               "already computes, so a vector-heavy policy is not charged zero"))
    events = {"aiv": {KIND_READY: candidate.v_versions + 3,
                      KIND_FREE: candidate.v_versions + 3,
                      KIND_STATE: 2, KIND_EXIT: 1},
              "aic": {KIND_READY: 0, KIND_FREE: 0, KIND_STATE: 0, KIND_EXIT: 0},
              "device": {KIND_CROSS: 0}}
    notes = [
        "L1: attention_residual's L1 use is the V ring plus x/y staging plus the persistent "
        "gamma/w; it is charged one term at a time rather than through the spec's MLA formula",
        "L0A/L0B/L0C: an AIV-only kernel issues no Cube work -- the backend spec forbids adding "
        "empty Cube work to use a Cube interface -- so these are charged ZERO and cannot "
        "produce a rejection",
        "SS: attention_residual has no group scalar communication, so it publishes nothing; the "
        "zero is recorded rather than omitted",
        "events: one AIV ready and one free id per V-ring generation, plus state and exit",
    ]
    if candidate.family == gen.RESIDENT_N:
        notes.append("resident_n: the family holds every block on chip, so its UB charge is the "
                     "one most likely to bind")
    return ResourceUse(charges=tuple(charges), event_use=events, notes=tuple(notes))


def attention_residual_legality(candidate) -> tuple:
    """Structural refusals for attention_residual, quoting the operator's own domain."""
    out = []
    try:
        spec = candidate.build_spec()
    except AttentionResidualError as exc:
        out.append(Rejection("LEGAL_" + str(exc.code), "schedule",
                             "%s (%s=%r)" % (exc.message, exc.subject, exc.actual),
                             exc.actual, exc.expected))
        return tuple(out)
    if candidate.block_n != spec.block_n:
        out.append(Rejection("LEGAL_CN_MISMATCH", "schedule",
                             "declared CN=%d but the built spec has CN=%d"
                             % (candidate.block_n, spec.block_n), candidate.block_n,
                             spec.block_n))
    if candidate.d_chunk < 1 or candidate.d_chunk > candidate.d:
        out.append(Rejection("LEGAL_D_CHUNK_OUT_OF_RANGE", "schedule",
                             "d_chunk=%d must be within 1..%d"
                             % (candidate.d_chunk, candidate.d), candidate.d_chunk,
                             "1..%d" % candidate.d))
    pads = 1
    while pads < candidate.d:
        pads *= 2
    if (pads & (pads - 1)) != 0:
        out.append(Rejection("LEGAL_PADDED_D_NOT_POWER_OF_TWO", "layout",
                             "padded D=%d is not a power of two, so layout inference has no "
                             "legal shared layout" % pads, pads, "power of two"))
    return tuple(out)


def judge_attention_residual(candidate, *, limits: Optional[TargetLimits] = None) -> Judgement:
    limits = limits or target_limits()
    try:
        use = attention_residual_resources(candidate, limits=limits)
        rejections = list(attention_residual_legality(candidate))
    except AttentionResidualError as exc:
        # A candidate whose spec cannot be built has no resource use to report. Reporting zero
        # and calling it infeasible keeps the record honest: nothing was fitted, nothing measured.
        return Judgement(key=candidate.key(), operator="attention_residual",
                         family=candidate.family, verdict=INFEASIBLE, evidence=UNMEASURED,
                         use=ResourceUse(charges=()), origin=candidate.origin,
                         parent=candidate.parent, changed=candidate.changed,
                         rejections=(Rejection("LEGAL_" + str(exc.code), "schedule",
                                               exc.message, exc.actual, exc.expected),))
    rejections.extend(_capacity_rejections(use, limits))
    notes = list(use.notes)
    # Device evidence for attention_residual exists only for the four configurations G52 swept.
    # Even a candidate that matches one of them is UNMEASURED until the harness records its own
    # run: reusing a family-level pass count as a per-candidate result is exactly the substitution
    # the gates forbid.
    if (candidate.token_rows == 1 and candidate.d_chunk == candidate.d
            and candidate.prefetch == gen.PREFETCH_NONE):
        notes.append("this configuration matches the shape G52 swept for this family, but a "
                     "family-level sweep is not a per-candidate measurement; the harness must "
                     "record one")
    return Judgement(key=candidate.key(), operator="attention_residual",
                     family=candidate.family,
                     verdict=FEASIBLE if not rejections else INFEASIBLE,
                     evidence=UNMEASURED, use=use, rejections=tuple(rejections),
                     origin=candidate.origin, parent=candidate.parent,
                     changed=candidate.changed, notes=tuple(notes))


# ------------------------------------------------------------------ batch judging

def judge_all(candidates, *, operator: str, spec=None,
              limits: Optional[TargetLimits] = None) -> tuple:
    """Judge a mixed list for one operator, preserving order and never dropping a candidate."""
    limits = limits or target_limits()
    out = []
    for candidate in candidates:
        if operator == "mla":
            if spec is None:
                raise FilterError("FILTER_SPEC_REQUIRED",
                                  "MLA judging needs the case specialization", "spec",
                                  "MLASpec", None)
            out.append(judge_mla(candidate, spec, limits=limits))
        elif operator == "attention_residual":
            out.append(judge_attention_residual(candidate, limits=limits))
        else:
            raise FilterError("FILTER_UNKNOWN_OPERATOR", "unknown operator", "operator",
                              ("mla", "attention_residual"), operator)
    return tuple(out)


def summarise(judgements) -> dict:
    """Count the space by verdict, origin, family and rejection code, so pruning is visible."""
    total = len(judgements)
    feasible = sum(1 for j in judgements if j.feasible)
    by_code = {}
    by_space = {}
    for judgement in judgements:
        for rejection in judgement.rejections:
            by_code[rejection.code] = by_code.get(rejection.code, 0) + 1
            by_space[rejection.space] = by_space.get(rejection.space, 0) + 1
    by_origin = {}
    for judgement in judgements:
        bucket = by_origin.setdefault(judgement.origin,
                                      {"total": 0, "feasible": 0, "infeasible": 0})
        bucket["total"] += 1
        bucket["feasible" if judgement.feasible else "infeasible"] += 1
    by_family = {}
    for judgement in judgements:
        bucket = by_family.setdefault(judgement.family, {"total": 0, "feasible": 0})
        bucket["total"] += 1
        if judgement.feasible:
            bucket["feasible"] += 1
    return {
        "total": total,
        "feasible": feasible,
        "infeasible": total - feasible,
        "pruned_fraction": (total - feasible) / float(total) if total else 0.0,
        "rejections_by_code": by_code,
        "rejections_by_space": by_space,
        "by_origin": by_origin,
        "by_family": by_family,
        "measured": sum(1 for j in judgements if j.evidence == MEASURED),
    }


__all__ = [
    "FEASIBLE", "INFEASIBLE", "UNMEASURED", "MEASURED", "NUMERIC_FAILED", "COMPILE_FAILED",
    "TIMEOUT", "DTYPE_BYTES", "SS_BYTES", "SS_ALIGN", "MIN_SCORE_ROWS", "V_ABS_MAX",
    "FilterError", "TargetLimits", "target_limits", "Charge", "Rejection", "ResourceUse",
    "Judgement", "mla_resources", "mla_legality", "judge_mla",
    "attention_residual_resources", "attention_residual_legality",
    "judge_attention_residual", "judge_all", "summarise",
]
