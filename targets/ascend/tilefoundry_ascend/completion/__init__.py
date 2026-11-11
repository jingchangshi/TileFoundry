"""P4-01 / G40: completion obligations, physical events and protocol checking.

WHY THIS MODULE EXISTS. The pre-P4 sync layer (schedule/sync.py) built a FIXED six-channel
RMSNorm template (x_ready, x_free, y_ready, y_free, w_ready, w_conv) and then numbered the
channels from a single chip-wide counter starting at 0. Both halves of that are the defect
R-10 / FLAG-5 names:

  * the obligation SET did not depend on the program. A program with a different number of
    buffers, consumers, versions or cores produced the same six names, so an obligation the
    program really needed could not exist and one it did not need was still emitted;
  * the event NUMBER was a global sequence, so "flag 3" meant nothing in particular. Real
    hardware exposes finite event identifiers inside INDEPENDENT namespaces (per core domain,
    per event kind, per pipe pair / bank). The same number in two independent namespaces is
    legal, two overlapping lifetimes in ONE namespace are not, and a number may be reused once
    the earlier lifetime has provably completed.

This package derives obligations from the REAL resolved plan, allocates finite events per real
namespace with lifetime proofs, and checks a completion program against a target-aware ring
protocol. It does not estimate cycle latency and does not treat a count of Set/Wait nodes as a
deadlock proof.

THE DECOMPOSITION, and why each part is separate:

  model.py        target facts: core domains, pipe pairs, event namespaces, pool capacities,
                  and the named refusals. No program is read here.
  obligations.py  plan -> logical obligations. Every obligation names the real producer and
                  consumer access it came from, so an obligation that no access supports
                  cannot be invented and one the plan needs cannot be dropped silently.
  events.py       obligations + namespaces -> finite physical event ids, with use intervals.
                  Overlap/conflict/exhaustion/reuse are decided HERE, not by luck of order.
  schedule.py     obligations -> an instance-ordered completion trace (Set/Wait points with
                  generation, slot, guard and core), including initial empty slots, short
                  loops (trip 0 / 1 / below the pipeline depth), conditional guards and the
                  final drain.
  protocol.py     trace -> ring-protocol simulation + deadlock/completion analysis. This is a
                  real state machine per (buffer, slot), and it also decides guard
                  compatibility, so equal Set/Wait counts cannot pass it.
  emit.py         trace -> real typed TIR (FlagSet / FlagWait / Drain under If guards) and
                  back (a TIR reader), so verification reads the ACTUAL program.
  verify.py       obligations + allocation + trace/TIR -> CompletionReport. Independent
                  re-derivation from the plan, then coverage, protocol, namespace and
                  cross-core checks with named codes.
  plan.py         orchestration: build_completion_plan / attach_completion_plan.
"""
from __future__ import annotations

from .model import (
    AIC,
    AIV,
    DEVICE,
    ASCEND_EVENT_FACTS,
    KIND_CROSS,
    KIND_EXIT,
    KIND_FREE,
    KIND_READY,
    KIND_STATE,
    OBLIGATION_KINDS,
    CompletionError,
    EventNamespace,
    EventPool,
    Obligation,
    PIPE_FACTS,
    TargetEventFacts,
    core_for_pipe,
    is_async_pipe,
    Use,
    UseInterval,
    digest,
    event_facts_for,
)
from .guards import (
    ALWAYS,
    COND,
    NEVER,
    Guard,
    GUARD_ALWAYS,
    GUARD_NEVER,
    guard_implies,
    guards_agree,
    normalize_guard,
)
from .obligations import derive_obligations, obligations_from_edges
from .events import EventAssignment, allocate_events, check_assignment
from .protocol import ProtocolVerdict, simulate_trace
from .schedule import CompletionTrace, TraceRecord, build_completion_trace
from .emit import emit_trace_tir, read_trace_from_tir
from .verify import CompletionReport, verify_completion
from .plan import (
    reemit_into_unrolled,
    COMPLETION_PLAN_KEY,
    CompletionPlan,
    attach_completion_plan,
    build_completion_plan,
    require_completion_verified,
)

__all__ = [
    "AIC", "AIV", "DEVICE", "ASCEND_EVENT_FACTS", "CompletionError",
    "EventNamespace", "EventPool", "Obligation",
    "PIPE_FACTS", "core_for_pipe", "is_async_pipe",
    "TargetEventFacts", "Use", "UseInterval", "digest", "event_facts_for",
    "derive_obligations", "obligations_from_edges",
    "KIND_READY", "KIND_FREE", "KIND_STATE", "KIND_EXIT", "KIND_CROSS", "OBLIGATION_KINDS",
    "ALWAYS", "COND", "NEVER", "Guard", "GUARD_ALWAYS", "GUARD_NEVER",
    "guard_implies", "guards_agree", "normalize_guard",
    "EventAssignment", "allocate_events", "check_assignment",
    "ProtocolVerdict", "simulate_trace",
    "CompletionTrace", "TraceRecord", "build_completion_trace",
    "emit_trace_tir", "read_trace_from_tir",
    "CompletionReport", "verify_completion",
    "COMPLETION_PLAN_KEY", "CompletionPlan", "attach_completion_plan",
    "build_completion_plan", "require_completion_verified",
]
