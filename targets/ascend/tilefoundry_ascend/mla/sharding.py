"""P6-03/G62: output-column sharding and the shape-specialized program mapping.

WHY THIS EXISTS. A decode case (B=1, S=1, Nq=128) has only 128 query rows and a handful of KV
heads, so it exposes too few jobs to fill 56 AIVs however the rows are tiled. cases/mla.md 6.2
gives the escape: split the OUTPUT COLUMNS. Each program still computes the COMPLETE softmax for
its query rows -- repeating the QK and softmax work -- and writes only the output columns it owns.

    program = (batch b, KV head h, query-row tile r0, output-column shard d0)

There is no cross-program merge, no GM probability matrix, and no second kernel. That is the whole
point: a shard is a self-contained copy of the single-kernel algorithm, so sharding costs
duplicated compute and buys nothing else. Split-KV would need a merge and is explicitly NOT allowed
here (see check_no_split_kv).

THE ROW INDEX r IS NOT THE QUERY POSITION, AND CONFLATING THEM IS THE TRAP. A tile of consecutive
r mixes query positions (and, under BSND, query heads). The causal mask needs the REAL i:

    BSND: r = i*G + g   ->  i = r // G,  g = r % G,  hq = h*G + g
    BNSD: r = g*S + i   ->  g = r // S,  i = r % S

Using r as the position is correct only when G == 1, which is exactly the configuration a prefill
test is likely to use -- so the bug passes the obvious test and then misattends for GQA. The
mapping is stated as functions here, in both layouts, and the causal helper refuses an un-mapped r.

THE COST IS REAL AND IS NOT HIDDEN. Sharding by k multiplies the QK and softmax work by k while
adding only occupancy. The cost model records that, so a caller decides on measurement rather than
on the intuition that more programs is better.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .model import BNSD, BSND, MLAError, MLAShape

#: Output-column panel widths the spec lists as the starting search set (cases/mla.md 5).
BD_CHOICES = (64, 128, 256)


def query_row_position(r: int, *, group: int, seq_q: int, layout: str) -> tuple:
    """Map a logical query-row index r to (query position i, group index g, query head hq).

    THE ONE FUNCTION EVERY CAUSAL CHECK MUST GO THROUGH. r is a PROGRAMMING row index: it orders
    rows so that memory access is contiguous. It is NOT a query position. Under BSND the group
    varies fastest (r = i*G + g); under BNSD the sequence varies fastest (r = g*S + i), which
    improves query read locality when the head axis is outer.

    The two layouts agree only at G == 1, so a test with G == 1 cannot distinguish a correct
    mapping from the bug of using r directly. That is why this is a function with a name rather
    than an inline division at each call site.
    """
    if layout not in (BSND, BNSD):
        raise MLAError("MLA_BAD_LAYOUT", "layout must be BSND or BNSD", "layout",
                       (BSND, BNSD), layout)
    if group < 1 or seq_q < 1:
        raise MLAError("MLA_BAD_SHAPE", "group and seq_q must be positive", "group/seq_q",
                       ">= 1", "group=%r seq_q=%r" % (group, seq_q))
    total = group * seq_q
    if r < 0 or r >= total:
        raise MLAError("MLA_ROW_OUT_OF_RANGE", "query row index out of range", "r",
                       "0..%d" % (total - 1), r)
    if layout == BSND:
        # head varies fastest, so consecutive r stays inside one query position
        return r // group, r % group, None
    # sequence varies fastest, so consecutive r stays inside one group
    return r % seq_q, r // seq_q, None


def query_head_of(r: int, *, kv_head: int, group: int, seq_q: int, layout: str) -> int:
    """The absolute query head hq a row belongs to: hq = h*G + g."""
    _, g, _ = query_row_position(r, group=group, seq_q=seq_q, layout=layout)
    return kv_head * group + g


def causal_position_of(r: int, *, group: int, seq_q: int, layout: str) -> int:
    """The REAL query position for the causal mask. Never returns r itself."""
    i, _, _ = query_row_position(r, group=group, seq_q=seq_q, layout=layout)
    return i


@dataclass(frozen=True)
class ColumnShard:
    """One program's output-column range [lo, hi) and the job it belongs to."""

    index: int
    lo: int
    hi: int

    @property
    def width(self) -> int:
        return self.hi - self.lo

    def owns(self, column: int) -> bool:
        return self.lo <= column < self.hi

    def as_dict(self) -> dict:
        return {"index": self.index, "lo": self.lo, "hi": self.hi, "width": self.width}


def column_shards(d_v: int, shard_count: int) -> tuple:
    """Split [0, d_v) into shard_count contiguous, non-overlapping, covering ranges.

    THE THREE PROPERTIES THAT MUST HOLD, each of which a plausible split can break:
      disjoint  no column is written by two programs -- an overlap is a data race, not a merge
      covering  every column is written by exactly one -- a gap is silently unwritten output
      balanced  widths differ by at most one -- an unbalanced split concentrates work

    A split that rounds the width up instead of distributing the remainder leaves a gap at the
    end; a split that rounds down twice can overlap. Both look fine on a power-of-two example.
    """
    if shard_count < 1:
        raise MLAError("MLA_BAD_SHARD_COUNT", "shard count must be >= 1", "shard_count",
                       ">= 1", shard_count)
    if d_v < 1:
        raise MLAError("MLA_BAD_SHARD_COUNT", "D_v must be >= 1", "d_v", ">= 1", d_v)
    if shard_count > d_v:
        raise MLAError(
            "MLA_TOO_MANY_SHARDS",
            "more column shards than output columns; some shard would own nothing",
            "shard_count", "<= %d" % d_v, shard_count)
    base, extra = divmod(d_v, shard_count)
    shards = []
    lo = 0
    for k in range(shard_count):
        width = base + (1 if k < extra else 0)
        hi = lo + width
        shards.append(ColumnShard(index=k, lo=lo, hi=hi))
        lo = hi
    return tuple(shards)


def check_shards(shards, d_v: int) -> tuple:
    """Verify disjoint / covering / balanced. Returns a tuple of violations; empty means valid.

    Stated as a check as well as a constructor because the shards may arrive from a plan, a cached
    dispatch decision or a hand-written candidate -- not only from column_shards().
    """
    violations = []
    seen = [0] * d_v
    for s in shards:
        if s.lo < 0 or s.hi > d_v or s.lo >= s.hi:
            violations.append(("shard_out_of_bounds",
                               "shard %d range [%d,%d) is not inside [0,%d)"
                               % (s.index, s.lo, s.hi, d_v)))
            continue
        for c in range(s.lo, s.hi):
            seen[c] += 1
    if any(n == 0 for n in seen):
        missing = [c for c, n in enumerate(seen) if n == 0]
        violations.append(("shard_gap",
                           "%d column(s) written by no shard, first at %d"
                           % (len(missing), missing[0])))
    if any(n > 1 for n in seen):
        dup = [c for c, n in enumerate(seen) if n > 1]
        violations.append(("shard_overlap",
                           "%d column(s) written by more than one shard, first at %d"
                           % (len(dup), dup[0])))
    widths = [s.width for s in shards]
    if widths and max(widths) - min(widths) > 1:
        violations.append(("shard_unbalanced",
                           "shard widths %r differ by more than one" % (widths,)))
    return tuple(violations)


@dataclass(frozen=True)
class ShardPlan:
    """A complete program decomposition: rows x kv-heads x batches x column shards."""

    batch: int
    seq_q: int
    seq_kv: int
    heads_q: int
    heads_kv: int
    d_v: int
    layout: str
    block_q: int
    shards: tuple
    program_count: int
    programs_per_row_tile: int

    def as_dict(self) -> dict:
        return {
            "batch": self.batch, "seq_q": self.seq_q, "seq_kv": self.seq_kv,
            "heads_q": self.heads_q, "heads_kv": self.heads_kv, "d_v": self.d_v,
            "layout": self.layout, "block_q": self.block_q,
            "shards": [s.as_dict() for s in self.shards],
            "program_count": self.program_count,
            "programs_per_row_tile": self.programs_per_row_tile,
            "duplicated_qk_softmax_work": self.programs_per_row_tile,
        }


def plan_programs(shape: MLAShape, *, block_q: int, shard_count: int = 1) -> ShardPlan:
    """Enumerate the programs for a shape, with or without column sharding.

    A program is (b, h, row tile, column shard). The row tiles within one (b, h) are the
    query-group tiles of width block_q, exactly as the unsharded kernel uses, so sharding only
    multiplies the grid and never changes the row tiling.
    """
    if block_q < 1:
        raise MLAError("MLA_BAD_TILE", "block_q must be >= 1", "block_q", ">= 1", block_q)
    group = shape.group
    if group % block_q:
        raise MLAError("MLA_GROUP_NOT_TILED", "query group must divide by block_q",
                       "group", "multiple of %d" % block_q, group)
    row_tiles = group // block_q
    shards = column_shards(shape.d_v, shard_count)
    violations = check_shards(shards, shape.d_v)
    if violations:
        raise MLAError("MLA_BAD_SHARDS", "; ".join(m for _, m in violations),
                       "shards", "disjoint, covering, balanced", violations)
    per_row_tile = len(shards)
    total = shape.batch * shape.heads_kv * shape.seq_q * row_tiles * per_row_tile
    return ShardPlan(
        batch=shape.batch, seq_q=shape.seq_q, seq_kv=shape.seq_kv,
        heads_q=shape.heads_q, heads_kv=shape.heads_kv, d_v=shape.d_v,
        layout=shape.layout, block_q=block_q, shards=shards,
        program_count=total, programs_per_row_tile=per_row_tile)


def program_axes(pid: int, plan: ShardPlan) -> dict:
    """Decode a flat program id into (b, h, r0, shard).

    The flat id's LEAST-significant axis is the column shard, so that shards of the same row tile
    land on nearby programs and read the same K tiles at roughly the same time. Stated explicitly
    because a different ordering changes locality without changing correctness, and a reader
    cannot tell which ordering was chosen from the arithmetic alone.
    """
    if pid < 0 or pid >= plan.program_count:
        raise MLAError("MLA_PROGRAM_OUT_OF_RANGE", "program id out of range", "pid",
                       "0..%d" % (plan.program_count - 1), pid)
    n_shards = plan.programs_per_row_tile
    tiles = plan.heads_q // plan.heads_kv // plan.block_q
    shard = pid % n_shards
    rest = pid // n_shards
    tile = rest % tiles
    rest //= tiles
    i = rest % plan.seq_q
    rest //= plan.seq_q
    h = rest % plan.heads_kv
    b = rest // plan.heads_kv
    return {"batch": b, "kv_head": h, "query_position": i, "row_tile": tile,
            "first_row": tile * plan.block_q, "shard": shard,
            "columns": plan.shards[shard]}


@dataclass(frozen=True)
class ShardCost:
    """The measured-in-principle cost of sharding, recorded so it is not assumed free."""

    programs: int
    row_tiles: int
    shards: int
    qk_work_multiplier: int
    softmax_work_multiplier: int
    pv_work_multiplier: float
    kv_read_multiplier: int

    def as_dict(self) -> dict:
        return {
            "programs": self.programs, "row_tiles": self.row_tiles, "shards": self.shards,
            "qk_work_multiplier": self.qk_work_multiplier,
            "softmax_work_multiplier": self.softmax_work_multiplier,
            "pv_work_multiplier": self.pv_work_multiplier,
            "kv_read_multiplier": self.kv_read_multiplier,
        }


def shard_cost(plan: ShardPlan) -> ShardCost:
    """What sharding multiplies.

    QK and softmax are repeated IN FULL by every shard of a row tile -- that is the price, and it
    grows linearly with the shard count. PV is split rather than repeated, so its total work is
    unchanged (multiplier 1.0). K reads are repeated per shard, because each shard needs the whole
    K tile to compute its own QK. V reads are NOT repeated: a shard reads only its own V columns.

    Recording PV as 1.0 and QK as k rather than lumping both is the difference between a cost model
    and a slogan. It also predicts the regime: sharding helps only when the shape is so small that
    occupancy, not arithmetic, is the binding constraint.
    """
    shards = len(plan.shards)
    return ShardCost(
        programs=plan.program_count,
        row_tiles=plan.program_count // shards if shards else 0,
        shards=shards,
        qk_work_multiplier=shards,
        softmax_work_multiplier=shards,
        pv_work_multiplier=1.0,
        kv_read_multiplier=shards,
    )


def check_no_split_kv(*, merges_across_programs: bool, gm_probability_matrix: bool,
                      second_kernel: bool, host_math: bool) -> tuple:
    """Refuse the alternatives the spec forbids for this variant.

    cases/mla.md 6.2 rules out multi-kernel split-KV as a fallback, and section 1 forbids host
    maths. A column shard needs NONE of these: it computes the full softmax itself, so there is
    nothing to merge. Each flag here is a way of quietly reintroducing the thing that was excluded.
    """
    violations = []
    if merges_across_programs:
        violations.append(("forbidden_cross_program_merge",
                           "a column shard must be self-contained; a cross-program merge is "
                           "split-KV by another name"))
    if gm_probability_matrix:
        violations.append(("forbidden_gm_probability",
                           "probabilities may not be materialised in GM"))
    if second_kernel:
        violations.append(("forbidden_second_kernel",
                           "the operator is one kernel launch; a second kernel is a different "
                           "operator with a different launch structure"))
    if host_math:
        violations.append(("forbidden_host_math",
                           "the host may only choose an implementation, validate metadata, "
                           "allocate the output and launch"))
    return tuple(violations)


def choose_shard_count(*, programs_unsharded: int, cores: int, d_v: int) -> int:
    """The smallest shard count that fills the device, or 1 when the shape already fills it.

    Deliberately CONSERVATIVE: it returns 1 whenever the unsharded grid already covers the cores,
    because sharding then multiplies real work for no occupancy gain. This is a lookup, not a
    performance claim -- the spec requires the choice to be made by measurement, and this only
    bounds the search.
    """
    if d_v < 1 or cores < 1:
        raise MLAError("MLA_BAD_SHARD_COUNT", "d_v and cores must be positive", "d_v/cores",
                       ">= 1", "d_v=%r cores=%r" % (d_v, cores))
    if programs_unsharded >= cores:
        return 1
    need = (cores + programs_unsharded - 1) // programs_unsharded
    return max(1, min(need, d_v))