"""Complete nested-list mathematical wrappers for the two semantic contracts.

Uses Python float, permits tiny unit-test dimensions, and never calls an NPU.
This is neither the official FP32/FP64 golden path nor a performance baseline.
The production entry must preserve Tensor ABI, dtype, device and contiguity.
"""
from online_softmax import attention_residual_row, mla_rows


def _shape(value):
    if not isinstance(value, (list, tuple)):
        return ()
    if not value:
        return (0,)
    child = _shape(value[0])
    if any(_shape(item) != child for item in value[1:]):
        raise ValueError("ragged nested-list tensor")
    return (len(value), *child)


def attention_residual(v_stack, x, w, gamma, alpha, epsilon=1e-6):
    """V[N,B,T,D], x[B,T,D], w[D], gamma[D] -> nested Y[B,T,D]."""
    shape = _shape(v_stack)
    if len(shape) != 4 or min(shape) <= 0:
        raise ValueError("v_stack must have shape [N,B,T,D]")
    n, b, t, d = shape
    if _shape(x) != (b, t, d) or _shape(w) != (d,) or _shape(gamma) != (d,):
        raise ValueError("AttentionResidual input shapes disagree")
    return [[list(attention_residual_row([v_stack[k][batch][token] for k in range(n)],
                                        x[batch][token], w, gamma, alpha, epsilon))
             for token in range(t)] for batch in range(b)]


def mla_dense(q_nope, q_rope, k_nope, k_rope, v, numKVHeads=1,
              scaleValue=-1.0, inputLayout="BSND", is_causal=False, *, block_kv=2):
    """Unpaged attention-only MLA with the original argument names.

    BSND input: Q[B,S,Nq,D], K/V[B,Skv,Nkv,D]. BNSD exchanges S and N.
    q_head//G selects the KV head. V must equal K_nope by value; distinct
    allocations are valid. Tiled online math never allocates full scores/P.
    """
    if inputLayout not in ("BSND", "BNSD"):
        raise ValueError("inputLayout must be BSND or BNSD")
    if type(numKVHeads) is not int or numKVHeads <= 0:
        raise ValueError("numKVHeads must be positive")
    shapes = [_shape(t) for t in (q_nope, q_rope, k_nope, k_rope, v)]
    if any(len(shape) != 4 or min(shape) <= 0 for shape in shapes):
        raise ValueError("MLA inputs must be nonempty rank-4 nested lists")
    logical = lambda s: s if inputLayout == "BSND" else (s[0], s[2], s[1], s[3])
    qn, qr, kn, kr, vs = [logical(s) for s in shapes]
    b, sequence, heads, dn = qn
    dr = qr[3]
    kv = kn[1]
    if (qr[:3] != qn[:3] or kn != (b, kv, numKVHeads, dn)
            or kr != (b, kv, numKVHeads, dr) or vs != kn or heads % numKVHeads):
        raise ValueError("MLA feature, batch, sequence or head shapes disagree")
    if is_causal and sequence > kv:
        raise ValueError("causal MLA requires S <= Skv")
    groups = heads // numKVHeads
    at = (lambda tensor, batch, pos, head: tensor[batch][pos][head]) if inputLayout == "BSND" else (
        lambda tensor, batch, pos, head: tensor[batch][head][pos])
    # Value equivalence is a mathematical precondition, not pointer aliasing.
    for batch in range(b):
        for pos in range(kv):
            for head in range(numKVHeads):
                if tuple(at(v, batch, pos, head)) != tuple(at(k_nope, batch, pos, head)):
                    raise ValueError("V must equal K_nope by value")
    out = [[[[0.0] * dn for _ in range(heads)] for _ in range(sequence)] for _ in range(b)]
    for batch in range(b):
        for kv_head in range(numKVHeads):
            row_map = [(pos, kv_head * groups + group)
                       for pos in range(sequence) for group in range(groups)]
            result = mla_rows(
                [at(q_nope, batch, pos, head) for pos, head in row_map],
                [at(q_rope, batch, pos, head) for pos, head in row_map],
                [at(k_nope, batch, pos, kv_head) for pos in range(kv)],
                [at(k_rope, batch, pos, kv_head) for pos in range(kv)],
                [at(v, batch, pos, kv_head) for pos in range(kv)],
                query_positions=[pos for pos, _ in row_map], query_length=sequence,
                scale=scaleValue, causal=is_causal, block_kv=block_kv)
            for (pos, head), row in zip(row_map, result):
                out[batch][pos][head] = list(row)
    if inputLayout == "BSND":
        return out
    return [[[out[batch][pos][head] for pos in range(sequence)]
             for head in range(heads)] for batch in range(b)]
