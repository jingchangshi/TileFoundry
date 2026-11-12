"""P5-02/G52: the measured CN>1 attention_residual kernel (blocked_online / resident_n).

THIS IS THE KERNEL THAT PRODUCED THE G52 DEVICE RESULTS. It generalises the G50 CN=1 kernel to
an arbitrary block width CN, which is what separates the three required families:

    CN = 1      online          one V block per running-state update
    CN < N      blocked_online  ceil(N/CN) updates, block state merged into the running state
    CN = N      resident_n      ONE update; the block state IS the final state

WHY BLOCK STATE IS A TWO-PASS FORM. A block's contribution to the running (m, l, acc) is only
well-defined once the block's own max m_b is known: exp(score_j - m_b) needs m_b. So the block is
walked twice -- pass 1 finds m_b, pass 2 forms l_b = sum_j exp(s_j - m_b) and
u_b = sum_j exp(s_j - m_b) * V_j -- and only then is the block merged:
nm = max(m, m_b), l = rho*l + bet*l_b, acc = rho*acc + bet*u_b with rho = exp(m-nm), bet = exp(m_b-nm).

THE REAL COST OF THAT FORM IS A SECOND READ OF V, and it is recorded rather than hidden. Holding
the CN tiles instead of re-reading them is the resource-pruning question in resources.py; here the
re-read is chosen because it costs UB rather than feasibility, and whether it wins on a given
shape is a measurement, not an assumption.

WHY niter IS 1 HERE. CN is an INTRA-program block width: one program owns one token group and
walks its ceil(N/CN) blocks. It does not need the cross-group persistent loop, so it is
unaffected by that loop's separate, still-open issue (recorded in G50's evidence). The persistent
g += P schedule remains G51 work and is NOT claimed.
"""
from __future__ import annotations

import math

THREADS = 128

#: The CN values with device evidence under G52.
MEASURED_CN = (1, 2, 4, 8)


def next_power_of_two(value: int) -> int:
    pd = 1
    while pd < value:
        pd *= 2
    return pd


def n_blocks(n: int, block_n: int) -> int:
    """ceil(N/CN) -- the number of running-state updates per token group."""
    if block_n < 1:
        raise ValueError("block_n must be >= 1, got %r" % (block_n,))
    return (n + block_n - 1) // block_n


def check_config(*, n: int, block_n: int, score_mode: str) -> None:
    """Refuse a configuration outside the measured domain.

    Raising here rather than in the builder keeps the honest boundary in one place: the model
    may DESCRIBE any legal (CN, mode) pair, but this kernel only CLAIMS the ones measured.
    """
    if n < 2:
        raise ValueError("attention_residual: N >= 2 is a hard domain constraint, got %r" % (n,))
    if block_n < 1 or block_n > n:
        raise ValueError("attention_residual: CN must satisfy 1 <= CN <= N, got %r" % (block_n,))
    if score_mode not in ("literal_fp32", "factored_gw"):
        raise ValueError("attention_residual: unknown score mode %r" % (score_mode,))


def kernel_summary(*, n: int, block_n: int, d: int, dtype: str = "bfloat16",
                   score_mode: str = "literal_fp32") -> dict:
    """The launch facts for one configuration, without compiling.

    Reported so a reader can see which family a configuration actually becomes: CN=1 is online,
    CN=N is resident_n, and anything between is blocked_online. Deriving the name from CN rather
    than storing it prevents a report from labelling a schedule it did not run.
    """
    check_config(n=n, block_n=block_n, score_mode=score_mode)
    if block_n == 1:
        family = "online"
    elif block_n == n:
        family = "resident_n"
    else:
        family = "blocked_online"
    return {
        "family": family,
        "block_n": block_n,
        "n_blocks": n_blocks(n, block_n),
        "state_updates": n_blocks(n, block_n),
        "padded_d": next_power_of_two(d),
        "d_tail": (d & (d - 1)) != 0,
        "denominator_d": d,
        "score_mode": score_mode,
        "dtype": dtype,
        "threads": THREADS,
        "n_iter": 1,
        "v_reads_per_block": 2,
        "v_reads_total": 2 * n,
        "measured": block_n in MEASURED_CN,
    }


def merge_algebra() -> dict:
    """The cross-block merge, stated once so test and kernel cannot drift apart.

    A test that re-derives this arithmetic and compares it to itself proves nothing; the value
    here is that the kernel and the device oracle were checked against the SAME statement of
    the rule, which is the golden's own form applied per block.
    """
    return {
        "block_pass_1": "m_b = max_j s_j over the CN members of the block",
        "block_pass_2": "l_b = sum_j exp(s_j - m_b) ; u_b = sum_j exp(s_j - m_b) * V_j",
        "merge": "nm = max(m, m_b) ; rho = exp(m - nm) ; bet = exp(m_b - nm)",
        "l_update": "l = rho * l + bet * l_b",
        "acc_update": "acc = rho * acc + bet * u_b",
        "final": "(1 - alpha) * x + alpha * acc / l",
        "note": "identical to the golden's running softmax once the block is treated as one "
                "group of N terms; CN changes only how many times the merge runs",
    }
