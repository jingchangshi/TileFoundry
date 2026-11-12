"""Executable mathematical kernel body: stable online softmax and rescaling.

Uses Python float (normally binary64), not the golden harness or an NPU kernel.
Port these equations into TF value-form SSA and TileLang AIV/AIC bodies. AIC
accumulators must be rescaled before each new block contribution is added.
"""
from dataclasses import dataclass
from math import exp, isfinite, sqrt


@dataclass(frozen=True)
class State:
    maximum: float
    denominator: float
    numerator: tuple[float, ...]

    @classmethod
    def empty(cls, width: int) -> "State":
        if type(width) is not int or width <= 0:
            raise ValueError("width must be a positive integer")
        return cls(float("-inf"), 0.0, (0.0,) * width)

    def normalized(self) -> tuple[float, ...]:
        # Defines the all-masked-row result without evaluating 0/0.
        if self.denominator == 0:
            return (0.0,) * len(self.numerator)
        return tuple(v / self.denominator for v in self.numerator)


def update(state: State, logits, values, mask=None) -> State:
    """Consume a K-wide score tile and K value vectors, each of width D.

    mask=False and score=-inf both contribute zero. A completely masked tile
    preserves state and never evaluates exp(-inf - -inf). Other nonfinite
    unmasked scores are rejected; they are outside this reference contract.
    """
    width = len(state.numerator)
    if len(logits) != len(values) or any(len(v) != width for v in values):
        raise ValueError("logits/values shape mismatch")
    if mask is None:
        mask = [True] * len(logits)
    if len(mask) != len(logits):
        raise ValueError("mask shape mismatch")
    active = []
    for score, value, valid in zip(logits, values, mask):
        if not valid or score == float("-inf"):
            continue
        if not isfinite(score) or not all(isfinite(x) for x in value):
            raise ValueError("unmasked scores/values must be finite")
        active.append((float(score), value))
    if not active:
        return state
    maximum = max(state.maximum, max(score for score, _ in active))
    alpha = exp(state.maximum - maximum) if state.denominator else 0.0
    weights = [(exp(score - maximum), value) for score, value in active]
    denominator = alpha * state.denominator + sum(p for p, _ in weights)
    numerator = tuple(alpha * state.numerator[d] + sum(p * v[d] for p, v in weights)
                      for d in range(width))
    return State(maximum, denominator, numerator)


def update_rows(states, logits_rows, values_rows, masks=None) -> tuple[State, ...]:
    """Batched rows; each row may use its own value tile or a shared reference."""
    rows = len(states)
    if len(logits_rows) != rows or len(values_rows) != rows:
        raise ValueError("row count mismatch")
    masks = [None] * rows if masks is None else masks
    if len(masks) != rows:
        raise ValueError("mask row count mismatch")
    return tuple(update(s, x, v, m) for s, x, v, m in zip(states, logits_rows, values_rows, masks))


def merge(left: State, right: State) -> State:
    """Merge independent KV partitions using unnormalized (m,l,u) states.

    This is algebra only. It does not supply cross-program synchronization or
    authorize a second kernel. A one-launch partitioned implementation needs
    its own validated on-device completion and reduction protocol.
    """
    if len(left.numerator) != len(right.numerator):
        raise ValueError("state width mismatch")
    if not left.denominator:
        return right
    if not right.denominator:
        return left
    maximum = max(left.maximum, right.maximum)
    a, b = exp(left.maximum - maximum), exp(right.maximum - maximum)
    return State(maximum, a * left.denominator + b * right.denominator,
                 tuple(a * u + b * v for u, v in zip(left.numerator, right.numerator)))


def attention_residual_row(v_blocks, x, w, gamma, alpha, epsilon=1e-6):
    """One [b,t] row; online depth-N aggregation with original V as values."""
    width = len(x)
    if width == 0 or len(w) != width or len(gamma) != width or len(v_blocks) < 2:
        raise ValueError("invalid AttentionResidual row shapes")
    if not 0.05 <= alpha <= 0.95 or not 1e-8 <= epsilon <= 1e-3:
        raise ValueError("alpha/epsilon outside attachment contract")
    state, gw = State.empty(width), [g * q for g, q in zip(gamma, w)]
    for v in v_blocks:
        if len(v) != width:
            raise ValueError("V row width mismatch")
        rms = sqrt(sum(a * a for a in v) / width + epsilon)
        score = sum(a * b for a, b in zip(v, gw)) / (rms * sqrt(width))
        state = update(state, [score], [v])
    return tuple((1 - alpha) * a + alpha * b for a, b in zip(x, state.normalized()))


def mla_rows(q_nope, q_rope, k_nope, k_rope, values, *, query_positions,
             query_length, scale=-1.0, causal=False, block_kv=2):
    """One KV head, multiple Q rows: tiled fused QK/softmax/PV mathematics.

    q rows can be different heads/positions. The caller maps BSND/BNSD and
    GQA to these rows. No full score/P matrix is allocated. The function uses
    exact Python-weight PV; production low-precision P is a separate policy.
    """
    rows, kv = len(q_nope), len(k_nope)
    if not rows or not kv or len(q_rope) != rows or len(query_positions) != rows:
        raise ValueError("invalid query rows")
    dn, dr = len(q_nope[0]), len(q_rope[0])
    if dn <= 0 or dr <= 0 or type(block_kv) is not int or block_kv <= 0:
        raise ValueError("invalid dimensions/tile")
    if len(k_rope) != kv or len(values) != kv:
        raise ValueError("KV row count mismatch")
    if any(len(v) != dn for v in [*q_nope, *k_nope, *values]) or any(len(v) != dr for v in [*q_rope, *k_rope]):
        raise ValueError("feature width mismatch")
    if any(not 0 <= i < query_length for i in query_positions) or (causal and query_length > kv):
        raise ValueError("query position/causal shape outside contract")
    scale = scale if scale > 0 else 1.0 / sqrt(dn + dr)
    states = tuple(State.empty(dn) for _ in range(rows))
    for first in range(0, kv, block_kv):
        last = min(kv, first + block_kv)
        scores = [[scale * (sum(a * b for a, b in zip(q_nope[r], k_nope[j]))
                            + sum(a * b for a, b in zip(q_rope[r], k_rope[j])))
                   for j in range(first, last)] for r in range(rows)]
        masks = [[not causal or j <= query_positions[r] + kv - query_length
                  for j in range(first, last)] for r in range(rows)]
        states = update_rows(states, scores, [values[first:last]] * rows, masks)
    return tuple(s.normalized() for s in states)
