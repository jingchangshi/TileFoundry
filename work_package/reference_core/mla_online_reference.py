"""Executable CPU algebra for the proposed single-kernel MLA algorithms.

Python float / math.fsum are used, so this is neither the original oracle nor
an fp32 hardware emulator. It checks blocking, causal indexing, online state,
deferred rescaling, and independent output-column shards. No device is used.
Input rows have already been mapped to one batch and one KV head.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Sequence

Matrix = Sequence[Sequence[float]]


@dataclass(frozen=True)
class RowState:
    anchor: float
    denominator: float
    numerator: tuple[float, ...]

    def output(self) -> list[float]:
        if self.denominator == 0:
            return [0.0] * len(self.numerator)
        return [x / self.denominator for x in self.numerator]


def _check(qn, qr, kn, kr, v, positions, query_length, kv_tile):
    if not qn or not kn or not v or kv_tile < 1:
        raise ValueError("Nonempty rows/KV and positive kv_tile are required")
    if len(qn) != len(qr) or len(qn) != len(positions):
        raise ValueError("Each Q row needs rope data and a logical query position")
    if len(kn) != len(kr) or len(kn) != len(v):
        raise ValueError("K_nope, K_rope, V need the same KV length")
    dn, dr, dv = len(qn[0]), len(qr[0]), len(v[0])
    if dn < 1 or dr < 1 or dv < 1:
        raise ValueError("Positive feature dimensions are required")
    for matrix, width in ((qn, dn), (qr, dr), (kn, dn), (kr, dr), (v, dv)):
        if any(len(row) != width for row in matrix):
            raise ValueError("Ragged or incompatible feature dimensions")
    if any(i < 0 or i >= query_length for i in positions):
        raise ValueError("Query positions must refer to original sequence indices")
    return dn, dr, dv


def _scores(qn, qr, kn, kr, positions, query_length, causal, scale, start, stop):
    offset = len(kn) - query_length
    rows = []
    for r, i in enumerate(positions):
        scores = []
        for j in range(start, stop):
            if causal and j > i + offset:
                scores.append(None)
            else:
                # Rope occurs once per (Q row, KV entry), after the nope sum.
                dot_nope = math.fsum(a * b for a, b in zip(qn[r], kn[j]))
                dot_rope = math.fsum(a * b for a, b in zip(qr[r], kr[j]))
                scores.append(scale * (dot_nope + dot_rope))
        rows.append(scores)
    return rows


def online_rows(qn: Matrix, qr: Matrix, kn: Matrix, kr: Matrix, v: Matrix, *,
                query_positions: Sequence[int], query_length: int,
                causal: bool = False, scale: float = -1.0, kv_tile: int = 64,
                probability_operand: Callable[[float], float] = float,
                output_columns: tuple[int, int] | None = None) -> list[RowState]:
    """Canonical per-KV-block online softmax with explicit output-column scope.

The caller can supply a P rounding function to explore operand error. The
denominator always uses unrounded P, matching the proposed numerical policy.
This does not implement the full five-tensor public ABI or its layout wrapper.
"""
    dn, dr, dv = _check(qn, qr, kn, kr, v, query_positions, query_length, kv_tile)
    if causal and query_length > len(kn):
        raise ValueError("The supplied MLA causal contract requires S <= Skv")
    scale = scale if scale > 0 else 1.0 / math.sqrt(dn + dr)
    lo, hi = output_columns if output_columns is not None else (0, dv)
    if not 0 <= lo < hi <= dv:
        raise ValueError("Invalid output-column shard")
    states = [RowState(-math.inf, 0.0, (0.0,) * (hi - lo)) for _ in qn]
    for begin in range(0, len(kn), kv_tile):
        end = min(begin + kv_tile, len(kn))
        scores = _scores(qn, qr, kn, kr, query_positions, query_length,
                         causal, scale, begin, end)
        for r, block in enumerate(scores):
            valid = [z for z in block if z is not None]
            if not valid:
                continue
            old = states[r]
            anchor = max(old.anchor, max(valid))
            alpha = 0.0 if old.denominator == 0 else math.exp(old.anchor - anchor)
            p = [0.0 if z is None else math.exp(z - anchor) for z in block]
            p_operand = [probability_operand(x) for x in p]
            numerator = tuple(alpha * old.numerator[d - lo] + math.fsum(
                p_operand[j - begin] * v[j][d] for j in range(begin, end))
                for d in range(lo, hi))
            states[r] = RowState(anchor, alpha * old.denominator + math.fsum(p), numerator)
    return states


def deferred_rows(qn: Matrix, qr: Matrix, kn: Matrix, kr: Matrix, v: Matrix, *,
                  query_positions: Sequence[int], query_length: int,
                  causal: bool = False, scale: float = -1.0, kv_tile: int = 64,
                  threshold: float = 0.0,
                  probability_operand: Callable[[float], float] = float,
                  output_columns: tuple[int, int] | None = None
                  ) -> tuple[list[RowState], int]:
    """Algebra for L0C-pending + UB-prefix accumulation with group-wide drain.

threshold bounds how far logits may rise above a retained scale anchor.
All rows drain together if any initialized row needs a new anchor. Hardware
must implement the AIV-to-AIC decision publication and completion protocol.
The returned drain count excludes the final drain and is a model diagnostic.
"""
    dn, dr, dv = _check(qn, qr, kn, kr, v, query_positions, query_length, kv_tile)
    if threshold < 0 or not math.isfinite(threshold):
        raise ValueError("Finite nonnegative threshold required")
    if causal and query_length > len(kn):
        raise ValueError("The supplied MLA causal contract requires S <= Skv")
    scale = scale if scale > 0 else 1.0 / math.sqrt(dn + dr)
    lo, hi = output_columns if output_columns is not None else (0, dv)
    if not 0 <= lo < hi <= dv:
        raise ValueError("Invalid output-column shard")
    width = hi - lo
    anchors = [-math.inf] * len(qn)
    denoms = [0.0] * len(qn)
    prefix = [[0.0] * width for _ in qn]
    pending = [[0.0] * width for _ in qn]
    drains = 0
    for begin in range(0, len(kn), kv_tile):
        end = min(begin + kv_tile, len(kn))
        scores = _scores(qn, qr, kn, kr, query_positions, query_length,
                         causal, scale, begin, end)
        block_max = [max((z for z in block if z is not None), default=-math.inf)
                     for block in scores]
        change = [denoms[r] > 0 and block_max[r] - anchors[r] > threshold
                  for r in range(len(qn))]
        if any(change):
            drains += 1
            for r in range(len(qn)):
                new_anchor = block_max[r] if change[r] else anchors[r]
                factor = (math.exp(anchors[r] - new_anchor)
                          if denoms[r] > 0 else 0.0)
                prefix[r] = [(x + y) * factor for x, y in zip(prefix[r], pending[r])]
                pending[r] = [0.0] * width
                denoms[r] *= factor
                anchors[r] = new_anchor
        for r, block in enumerate(scores):
            if block_max[r] == -math.inf:
                continue
            if denoms[r] == 0:
                anchors[r] = block_max[r]
            p = [0.0 if z is None else math.exp(z - anchors[r]) for z in block]
            operands = [probability_operand(x) for x in p]
            denoms[r] += math.fsum(p)
            for d in range(lo, hi):
                pending[r][d - lo] += math.fsum(
                    operands[j - begin] * v[j][d] for j in range(begin, end))
    states = [RowState(anchors[r], denoms[r], tuple(x + y for x, y in zip(prefix[r], pending[r])))
              for r in range(len(qn))]
    return states, drains


def merge_partial_states(parts: Sequence[RowState]) -> RowState:
    """Softmax merge algebra only; this is not a device split-KV protocol."""
    if not parts:
        raise ValueError("At least one partial state is required")
    width = len(parts[0].numerator)
    if any(len(p.numerator) != width for p in parts):
        raise ValueError("Partial states have different output widths")
    active = [p for p in parts if p.denominator > 0]
    if not active:
        return RowState(-math.inf, 0.0, (0.0,) * width)
    anchor = max(p.anchor for p in active)
    factors = [math.exp(p.anchor - anchor) for p in active]
    denominator = math.fsum(f * p.denominator for f, p in zip(factors, active))
    numerator = tuple(math.fsum(f * p.numerator[d] for f, p in zip(factors, active))
                      for d in range(width))
    return RowState(anchor, denominator, numerator)


def dense_rows(qn, qr, kn, kr, v, *, query_positions, query_length,
               causal=False, scale=-1.0):
    """Small CPU diagnostic reference, independent of block state updates."""
    dn, dr, dv = _check(qn, qr, kn, kr, v, query_positions, query_length, 1)
    scale = scale if scale > 0 else 1.0 / math.sqrt(dn + dr)
    scores = _scores(qn, qr, kn, kr, query_positions, query_length,
                     causal, scale, 0, len(kn))
    result = []
    for block in scores:
        valid = [z for z in block if z is not None]
        if not valid:
            result.append([0.0] * dv)
            continue
        maximum = max(valid)
        p = [0.0 if z is None else math.exp(z - maximum) for z in block]
        denom = math.fsum(p)
        result.append([math.fsum(p[j] * v[j][d] for j in range(len(v))) / denom
                       for d in range(dv)])
    return result


def _self_check():
    import random
    rng = random.Random(950)
    matrix = lambda rows, cols: [[rng.uniform(-2, 2) for _ in range(cols)] for _ in range(rows)]
    checks = 0
    max_error = 0.0
    for causal in (False, True):
        qn, qr, kn, kr = matrix(5, 7), matrix(5, 3), matrix(13, 7), matrix(13, 3)
        v = [row[:] for row in kn]
        # A row order different from query positions checks GQA-style flattening.
        args = dict(query_positions=[2, 0, 1, 2, 0], query_length=3, causal=causal)
        wanted = dense_rows(qn, qr, kn, kr, v, **args)
        for tile in (1, 2, 4, 8, 32):
            actual = [s.output() for s in online_rows(qn, qr, kn, kr, v, kv_tile=tile, **args)]
            err = max(abs(x-y) for a,b in zip(actual,wanted) for x,y in zip(a,b))
            assert err < 1e-12, (causal, tile, err)
            max_error = max(max_error, err)
            checks += 1
            for threshold in (0.0, 0.5, 3.0):
                states, drains = deferred_rows(qn, qr, kn, kr, v, kv_tile=tile, threshold=threshold, **args)
                err = max(abs(x-y) for a,b in zip([s.output() for s in states],wanted) for x,y in zip(a,b))
                assert err < 1e-12, (causal, tile, threshold, err)
                max_error = max(max_error, err)
                checks += 1
            left = online_rows(qn, qr, kn, kr, v, output_columns=(0,3), kv_tile=tile, **args)
            right = online_rows(qn, qr, kn, kr, v, output_columns=(3,7), kv_tile=tile, **args)
            combined = [a.output()+b.output() for a,b in zip(left,right)]
            assert max(abs(x-y) for a,b in zip(combined,wanted) for x,y in zip(a,b)) < 1e-12
            checks += 1
    return {"scope":"CPU_algebra_only", "checks":checks, "max_abs_error":max_error,
            "original_oracle_replaced":False, "npu_executions":0}


if __name__ == "__main__":
    import json
    print(json.dumps(_self_check(), indent=2))
