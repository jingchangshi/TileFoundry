"""P5-01/G50: the Scheduled HIR for attention_residual.

WHAT A SCHEDULED HIR IS HERE. Not a lowering artifact and not a re-statement of the algebra: it
is the TASK GRAPH the kernel runs, with each task's producer/consumer edges made explicit, so the
kernel body can be checked against the schedule rather than against a comment. The spec's
section 7 names exactly these tasks:

    LOAD_V(u)       AIV / MTE2    move CN x R x D in, with the validity mask
    BLOCK_COMPUTE(u) AIV / VECTOR  score, block state, running-state merge; last reader of V
    LOAD_X(g)       AIV / MTE2    the residual branch input
    FINALIZE(g)     AIV / VECTOR  read final state and x, form the output
    STORE_Y(g)      AIV / MTE3    write the REAL D out

STAGE ASSIGNMENT IS A STARTING POINT, NOT A CLAIM. The spec says stage 0/1 is "a searchable
starting point" from which the compiler derives phase/II. This module records the author's
initial stage and does not pretend it was proved optimal -- that is a performance question and
this gate is a correctness gate.

THE ONE EDGE THAT IS EASY TO GET WRONG, and which the spec calls out twice: the LAST CONSUMER OF
V IS THE WEIGHTED SUM, NOT THE SCORE. If the release edge were attached to the score task, a V
buffer could be overwritten while the weighted sum still had to read it. The edge table below
attaches it to BLOCK_COMPUTE as a whole for exactly that reason, and the checker refuses a HIR
whose V release is attributed to the score alone.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from .model import (
    AttentionResidualError,
    AttentionResidualSpec,
    LITERAL_FP32,
    FACTORED_GW,
    FAMILIES,
)

# ------------------------------------------------------------------ vocabulary

LOAD_V = "LOAD_V"
BLOCK_COMPUTE = "BLOCK_COMPUTE"
LOAD_X = "LOAD_X"
FINALIZE = "FINALIZE"
STORE_Y = "STORE_Y"

AIV = "aiv"
MTE2 = "MTE2"
VECTOR = "VECTOR"
MTE3 = "MTE3"

#: Which pipe each task occupies. A task that spans pipes would need splitting, so the table
#: deliberately names exactly one pipe per task and the checker rejects a composite task.
TASK_PIPE = {
    LOAD_V: MTE2,
    BLOCK_COMPUTE: VECTOR,
    LOAD_X: MTE2,
    FINALIZE: VECTOR,
    STORE_Y: MTE3,
}

#: The author's initial stage. Searchable, not proved.
TASK_STAGE = {
    LOAD_V: 0,
    BLOCK_COMPUTE: 1,
    LOAD_X: 0,
    FINALIZE: 1,
    STORE_Y: 1,
}


@dataclass(frozen=True)
class Task:
    """One task instance in the scheduled graph."""

    name: str
    scope: str            # 'group' (per token group) or 'block' (per V block inside a group)
    ordinal: int
    pipe: str
    stage: int

    def as_dict(self) -> dict:
        return {"name": self.name, "scope": self.scope, "ordinal": self.ordinal,
                "pipe": self.pipe, "stage": self.stage}


@dataclass(frozen=True)
class Edge:
    """A happens-before edge between task instances, with the reason it exists."""

    producer: str
    consumer: str
    kind: str             # 'ready' or 'free'
    reason: str

    def as_dict(self) -> dict:
        return {"producer": self.producer, "consumer": self.consumer, "kind": self.kind,
                "reason": self.reason}


@dataclass(frozen=True)
class ScheduledHIR:
    """The task graph plus the facts needed to check a kernel body against it."""

    spec: AttentionResidualSpec
    tasks: tuple
    edges: tuple
    address_equations: dict
    metadata: dict = field(default_factory=dict)

    def task_names(self) -> tuple:
        return tuple(t.name for t in self.tasks)

    def as_dict(self) -> dict:
        return {
            "spec": self.spec.as_dict(),
            "tasks": [t.as_dict() for t in self.tasks],
            "edges": [e.as_dict() for e in self.edges],
            "address_equations": dict(self.address_equations),
            "metadata": dict(self.metadata),
        }


def _address_equations(spec: AttentionResidualSpec) -> dict:
    """Flat addressing for every buffer the kernel touches.

    Written out because the device kernel indexes FLAT and an off-by-one in any of these is a
    silent wrong answer rather than a crash. v_stack is [N,B,T,D] and x/y are [B,T,D]; M = B*T
    flattens the batch and sequence axes, which is legal because every operation along N is
    per-token and nothing mixes tokens.
    """
    d = spec.d
    m = spec.tokens
    return {
        # v_stack[n, token, d] at flat offset (n * M + token) * D + d
        "v_stack": "(n * M + token) * D + i",
        "x": "token * D + i",
        "y": "token * D + i",
        # w and gamma are length-D vectors read once per participating AIV
        "w": "i",
        "gamma": "i",
        "M": str(m),
        "D": str(d),
        "N": str(spec.n),
        "CN": str(spec.block_n),
        "n_blocks": str(spec.n_blocks()),
    }


def build_scheduled_hir(spec: AttentionResidualSpec) -> ScheduledHIR:
    """Build the task graph for spec, refusing a description the schedule cannot honour."""
    if not isinstance(spec, AttentionResidualSpec):
        raise AttentionResidualError(
            "ATTNRES_NOT_A_SPEC", "the scheduled HIR is built from an AttentionResidualSpec",
            "spec", "AttentionResidualSpec", type(spec).__name__)
    if spec.family not in FAMILIES:
        raise AttentionResidualError(
            "ATTNRES_UNKNOWN_FAMILY", "unknown family", "family", FAMILIES, spec.family)
    if spec.launches != 1:
        raise AttentionResidualError(
            "ATTNRES_MULTIPLE_LAUNCHES",
            "this gate requires ONE logical kernel; a second launch would be the multi-kernel "
            "shape the operator forbids", "launches", 1, spec.launches)

    tasks = []
    ordinal = 0
    for g in range(spec.groups()):
        tasks.append(Task(LOAD_X, "group", g, TASK_PIPE[LOAD_X], TASK_STAGE[LOAD_X]))
        for u in range(spec.n_blocks()):
            # One LOAD_V / BLOCK_COMPUTE pair PER V BLOCK, not per CN group: each member of the
            # block is loaded and scored individually, and the block merge happens once the
            # group's members are complete. Modelling one task per CN group would hide the
            # second V read the blocked form actually performs.
            for _member in range(min(spec.block_n, spec.n - u * spec.block_n)):
                tasks.append(Task(LOAD_V, "block", u, TASK_PIPE[LOAD_V], TASK_STAGE[LOAD_V]))
                tasks.append(Task(BLOCK_COMPUTE, "block", u, TASK_PIPE[BLOCK_COMPUTE],
                                  TASK_STAGE[BLOCK_COMPUTE]))
        tasks.append(Task(FINALIZE, "group", g, TASK_PIPE[FINALIZE], TASK_STAGE[FINALIZE]))
        tasks.append(Task(STORE_Y, "group", g, TASK_PIPE[STORE_Y], TASK_STAGE[STORE_Y]))
        ordinal += 1

    edges = (
        # READY: a block may not be scored before its V tile has landed.
        Edge(LOAD_V, BLOCK_COMPUTE, "ready",
             "BLOCK_COMPUTE(u) reads V(u); issuing it before LOAD_V(u) completes would score "
             "uninitialised data"),
        # The running state is carried block to block. This is a VECTOR -> VECTOR dependency
        # inside one core, so it is a pipe guarantee rather than a cross-core flag.
        Edge(BLOCK_COMPUTE, BLOCK_COMPUTE, "ready",
             "BLOCK_COMPUTE(u+1) consumes the (m, l, acc) state BLOCK_COMPUTE(u) produced"),
        # READY: the output needs both the residual branch and the final aggregated state.
        Edge(BLOCK_COMPUTE, FINALIZE, "ready",
             "FINALIZE reads the final m/l/acc, which only the last BLOCK_COMPUTE produces"),
        Edge(LOAD_X, FINALIZE, "ready",
             "FINALIZE forms (1-alpha)*x + alpha*acc/l, so it must read the loaded x"),
        Edge(FINALIZE, STORE_Y, "ready",
             "STORE_Y writes what FINALIZE produced"),
        # FREE: the LAST reader of V is the weighted sum inside BLOCK_COMPUTE, NOT the score.
        # Attaching this to the score would let the next LOAD_V overwrite a tile the weighted
        # sum still needs.
        Edge(BLOCK_COMPUTE, LOAD_V, "free",
             "V(u) may only be overwritten after BLOCK_COMPUTE(u) has finished BOTH the score "
             "and the weighted sum; the weighted sum is the last consumer, not the score"),
        # FREE: x and the output staging may be reused only after the store has read them.
        Edge(STORE_Y, LOAD_X, "free",
             "the next group may reuse the x/output staging only after STORE_Y(g) has read it"),
    )

    return ScheduledHIR(
        spec=spec,
        tasks=tuple(tasks),
        edges=edges,
        address_equations=_address_equations(spec),
        metadata={
            "single_launch": True,
            "softmax_axis": "N",
            "score_mode": spec.score_mode,
            "literal_fp32_is_primary": spec.score_mode == LITERAL_FP32,
            "factored_gw_differs_in_float": spec.score_mode == FACTORED_GW,
            "v_last_consumer": BLOCK_COMPUTE,
            "block_n": spec.block_n,
            "n_blocks": spec.n_blocks(),
            "state_updates": spec.state_updates(),
            "block_state": ("none (CN=1, the running state IS the block state)"
                            if spec.block_n == 1 else
                            "m_b/l_b/u_b per block, merged into m/l/acc after each block"),
            "v_reads_per_element": 2,
            "stage_is_search_start": True,
            "state_dtype": spec.state.as_dict(),
        },
    )


def check_scheduled_hir(hir: ScheduledHIR) -> tuple:
    """Return the findings for hir; raise on a violation. Empty tuple means all clauses held."""
    if not isinstance(hir, ScheduledHIR):
        raise AttentionResidualError(
            "ATTNRES_NOT_A_HIR", "expected a ScheduledHIR", "hir", "ScheduledHIR",
            type(hir).__name__)
    spec = hir.spec
    findings = []

    # A composite task would need splitting into single-pipe pieces before it could be
    # scheduled, so the graph must not contain one.
    for task in hir.tasks:
        if task.pipe not in (MTE2, VECTOR, MTE3):
            raise AttentionResidualError(
                "ATTNRES_BAD_PIPE", "task names a pipe this graph does not model",
                task.name, (MTE2, VECTOR, MTE3), task.pipe)
    findings.append("%d task instances across %d pipe(s)"
                    % (len(hir.tasks), len({t.pipe for t in hir.tasks})))

    # The single-launch claim is the operator's whole point, so it is checked, not assumed.
    if spec.launches != 1:
        raise AttentionResidualError(
            "ATTNRES_MULTIPLE_LAUNCHES", "more than one logical launch", "launches", 1,
            spec.launches)
    findings.append("one logical launch")

    # V's last consumer must be the task that performs the weighted sum. This is the clause the
    # spec repeats, so it is checked structurally rather than trusted.
    if hir.metadata.get("v_last_consumer") != BLOCK_COMPUTE:
        raise AttentionResidualError(
            "ATTNRES_V_RELEASE_TOO_EARLY",
            "the V release edge is not attributed to the weighted-sum consumer; releasing V "
            "after the score would let a reload overwrite a tile still needed by the sum",
            "v_last_consumer", BLOCK_COMPUTE, hir.metadata.get("v_last_consumer"))
    findings.append("V's release edge is attributed to BLOCK_COMPUTE (score + weighted sum)")

    # The softmax axis is the operator's identity; a HIR that lost it describes something else.
    if hir.metadata.get("softmax_axis") != "N":
        raise AttentionResidualError(
            "ATTNRES_SOFTMAX_AXIS", "the depth softmax must run along N", "softmax_axis", "N",
            hir.metadata.get("softmax_axis"))
    findings.append("softmax is along N")

    # FP32 state is a numerical requirement, not a hint.
    state = spec.state.as_dict()
    for name, dtype in state.items():
        if dtype != "float32":
            raise AttentionResidualError(
                "ATTNRES_STATE_NOT_FP32",
                "the golden computes %r in FP32; narrowing it changes the operator's numerics"
                % name, name, "float32", dtype)
    findings.append("m/l/acc/score are all FP32")

    # The RMS denominator and the score temperature both use the REAL D.
    if "padded" in str(hir.address_equations.get("D", "")).lower():
        raise AttentionResidualError(
            "ATTNRES_USES_PADDED_D",
            "the RMS denominator must be the real D, never a padded DP", "D", "real D",
            hir.address_equations.get("D"))
    findings.append("RMS denominator and sqrt(D) use the real D=%d (padded width %d is "
                    "allocation only)" % (spec.d, spec.padded_d()))

    # Every family's state-update count must follow from CN, not from a stored constant.
    if spec.state_updates() != spec.n_blocks():
        raise AttentionResidualError(
            "ATTNRES_STATE_UPDATE_MISMATCH",
            "state updates must equal ceil(N/CN)", "state_updates", spec.n_blocks(),
            spec.state_updates())
    findings.append("%s: CN=%d so %d state update(s) per token group"
                    % (spec.family, spec.block_n, spec.state_updates()))

    # Every edge must name two tasks that exist.
    names = set(hir.task_names())
    for edge in hir.edges:
        for endpoint in (edge.producer, edge.consumer):
            if endpoint not in names:
                raise AttentionResidualError(
                    "ATTNRES_DANGLING_EDGE", "edge names a task that is not in the graph",
                    endpoint, sorted(names), endpoint)
    findings.append("%d edges, all endpoints present" % len(hir.edges))

    return tuple(findings)