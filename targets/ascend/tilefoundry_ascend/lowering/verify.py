"""Resolved-TIR verifier (M6/P5).

Events are re-parsed from the printed TIR text (not taken from the generator), then
replayed through per-slot state machines for the X and Y rings. In addition to the
per-slot/state checks, explicit COMPLETION-PROTOCOL rules are enforced, each with its
own stable tagged message:
  [M6-EMPTY-TIR]            the artifact carries no events
  [M6-UNKNOWN-STATEMENT]    a non-comment line that matches no known statement
  [M6-MISSING-DRAIN]        no final drain/completion barrier
  [M6-DRAIN-NOT-LAST]       the drain is not the final statement
  [M6-FLAG-MISMATCH]        a channel's set and wait name different flag ids
  [M6-MISSING-X-FREE-WAIT]  an x_load write needing a free-wait has none
  [M6-OUTSTANDING-GENERATION] a slot is left non-free (undrained) at the end
plus the per-slot state-machine errors (early free, wrong slot, gen mismatch).
Physical address/layout hazards are checked separately in layout_check.verify_layout.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_WAIT = re.compile(r"wait_flag\s+(?P<ch>\w+)\[(?P<slot>\d+)\]\s+gen=(?P<gen>\d+)\s+flag=(?P<flag>\d+)\s*#.*t=(?P<t>\d+)")
_SET = re.compile(r"set_flag\s+(?P<ch>\w+)\[(?P<slot>\d+)\]\s+gen=(?P<gen>\d+)\s+flag=(?P<flag>\d+)\s*#.*t=(?P<t>\d+)")
_OP = re.compile(r"(?P<op>mte2|mte3|simdvf)\s+(?P<task>\w+)\s+slot=(?P<slot>\d+)\s+t=(?P<t>\d+)")
_DRAIN = re.compile(r"^drain\(\)\s*(?:#.*)?$")


@dataclass
class PEvent:
    kind: str
    channel: str
    slot: int
    gen: int
    task: str
    t: int


def parse_events(text: str):
    out = []
    for line in text.splitlines():
        s = line.strip()
        m = _WAIT.match(s)
        if m:
            out.append(PEvent("WAIT", m.group("ch"), int(m.group("slot")), int(m.group("gen")), "?", int(m.group("t"))))
            continue
        m = _SET.match(s)
        if m:
            out.append(PEvent("SET", m.group("ch"), int(m.group("slot")), int(m.group("gen")), "?", int(m.group("t"))))
            continue
        m = _OP.match(s)
        if m:
            kind = {"mte2": "MTE2", "mte3": "MTE3", "simdvf": "VECTOR"}[m.group("op")]
            out.append(PEvent(kind, "", int(m.group("slot")), int(m.group("t")), m.group("task"), int(m.group("t"))))
    return out


def parse_flags(text: str):
    """channel -> {'set': {flag ids}, 'wait': {flag ids}} parsed from the text."""
    out = {}
    for line in text.splitlines():
        s = line.strip()
        m = _WAIT.match(s)
        if m:
            out.setdefault(m.group("ch"), {"set": set(), "wait": set()})["wait"].add(int(m.group("flag")))
            continue
        m = _SET.match(s)
        if m:
            out.setdefault(m.group("ch"), {"set": set(), "wait": set()})["set"].add(int(m.group("flag")))
    return out


def _channel_names_of(problem, plan):
    """The channel names the PLAN derived, or None when the plan carries no obligations.

    Imported lazily from the emitter so there is exactly ONE mapping from ring roles to
    derived channels: if the emitter's answer were recomputed differently here, the verifier
    and the emitter could disagree while both looked self-consistent.
    """
    from .materialize import channel_names

    try:
        return channel_names(plan)
    except KeyError:
        return None


def verify(problem, text: str, plan=None):
    """Re-parse and re-check the printed TIR.

    'plan' is optional so the signature stays compatible, but when supplied the derived channel
    names and pipe pairs come from IT. Without a plan the verifier cannot know which channels
    the program was supposed to have, and it says so rather than falling back to a literal list.
    """
    Vx = problem.policy["x_versions"]; Vy = problem.policy["y_versions"]
    ev = parse_events(text)
    errors = []
    raw = [l.strip() for l in text.splitlines()]
    # [M6-UNKNOWN-STATEMENT]
    unknown = []
    for l in raw:
        if not l:
            continue
        if l.startswith("#") or l.startswith("def ") or l.startswith("for "):
            continue
        if _WAIT.match(l) or _SET.match(l) or _OP.match(l) or _DRAIN.match(l):
            continue
        unknown.append(l)
    for u in unknown[:3]:
        errors.append("[M6-UNKNOWN-STATEMENT] unknown statement: " + u)
    # [M6-EMPTY-TIR]
    if not ev:
        errors.append("[M6-EMPTY-TIR] empty TIR: no events")
    # [M6-MISSING-DRAIN] / [M6-DRAIN-NOT-LAST]
    data_idx = [i for i, l in enumerate(raw) if l and not l.startswith("#")]
    drain_idx = [i for i, l in enumerate(raw) if _DRAIN.match(l)]
    if not drain_idx:
        errors.append("[M6-MISSING-DRAIN] missing drain: no completion/visibility barrier")
    elif data_idx and drain_idx[-1] != data_idx[-1]:
        errors.append("[M6-DRAIN-NOT-LAST] drain is not the final event")
    # [M6-FLAG-MISMATCH]
    flags = parse_flags(text)
    for ch, d in sorted(flags.items()):
        if (d["set"] and d["wait"] and d["set"] != d["wait"]) or len(d["set"]) > 1 or len(d["wait"]) > 1:
            errors.append("[M6-FLAG-MISMATCH] channel %s set %s wait %s"
                          % (ch, sorted(d["set"]), sorted(d["wait"])))
    # M6-UNKNOWN-CHANNEL: a WAIT/SET on a channel the PLAN never derived is rejected.
    # The known set is read from the plan's own obligations, so a program that invents a
    # channel -- or an emitter still using the old RMSNorm names -- is caught by name.
    names = _channel_names_of(problem, plan)
    known = set(names.values()) if names else set()
    for obligation in plan.get("obligations") or ():
        if obligation.get("channel"):
            known.add(str(obligation["channel"]))
    for e in ev:
        if e.kind in ("WAIT", "SET") and e.channel not in known:
            errors.append("[M6-UNKNOWN-CHANNEL] unknown sync channel: " + e.channel)
    # The ring a channel/op belongs to is identified by WHAT IT ORDERS, looked up from the plan
    # rather than by literal channel names: the names are derived now (see
    # lowering/materialize.channel_names), so a literal comparison would silently stop matching.
    names = _channel_names_of(problem, plan)
    x_channels = {names["x_ready"], names["x_free"]} if names else set()
    y_channels = {names["y_ready"], names["y_free"]} if names else set()

    # [M6-MISSING-X-FREE-WAIT] -- only when the plan actually HAS an x_free edge. A program
    # whose derivation produced no free edge (a write-only root, or a single-generation region)
    # has nothing to free, and demanding the wait would be a rule about a channel that does not
    # exist. The check is on the DERIVED channel, so dropping a derived wait still fails.
    x_ready_channel = names["x_ready"] if names else None
    x_free_channel = names["x_free"] if names else None
    y_ready_channel = names["y_ready"] if names else None
    y_free_channel = names["y_free"] if names else None
    if x_free_channel is not None:
        xload_mtes = [e for e in ev if e.kind == "MTE2" and e.t >= Vx]
        n_xfree_wait = sum(1 for e in ev
                           if e.kind == "WAIT" and e.channel == x_free_channel)
        if n_xfree_wait < len(xload_mtes):
            errors.append("[M6-MISSING-X-FREE-WAIT] %d x_free wait(s) for %d x_load write(s)"
                          " needing one" % (n_xfree_wait, len(xload_mtes)))
    # ---- per-slot state machine ----
    xstate = {i: "free" for i in range(Vx)}
    ystate = {i: "free" for i in range(Vy)}

    def _ver_for(e):
        if e.channel in x_channels or e.kind == "MTE2":
            return Vx
        if e.channel in y_channels or e.kind in ("MTE3", "VECTOR"):
            return Vy
        return None
    for e in ev:
        V = _ver_for(e)
        if V:
            if e.gen != e.t:
                errors.append(f"t={e.t}: {e.kind}/{e.channel} gen {e.gen} != logical iteration t")
            if e.slot != (e.gen % V):
                errors.append(f"t={e.t}: {e.kind}/{e.channel} slot {e.slot} != gen%{V} ({e.gen % V}); tile-index used as generation?")
        if e.kind == "WAIT" and e.channel == x_free_channel:
            if xstate.get(e.slot) != "free":
                errors.append(f"t={e.t}: wait x_free on slot {e.slot} in state {xstate.get(e.slot)}")
        elif e.kind == "MTE2":
            if e.slot not in xstate or xstate[e.slot] != "free":
                errors.append(f"t={e.t}: mte2 writes x slot {e.slot} not free ({xstate.get(e.slot)})")
            else:
                xstate[e.slot] = "x_written"
        elif e.kind == "SET" and e.channel == x_ready_channel:
            if xstate.get(e.slot) != "x_written":
                errors.append(f"t={e.t}: set x_ready slot {e.slot} in state {xstate.get(e.slot)}")
            else:
                xstate[e.slot] = "x_ready"
        elif e.kind == "WAIT" and e.channel == x_ready_channel:
            if xstate.get(e.slot) != "x_ready":
                errors.append(f"t={e.t}: wait x_ready slot {e.slot} in state {xstate.get(e.slot)}")
            else:
                xstate[e.slot] = "x_reading"
        elif e.kind == "SET" and e.channel == x_free_channel:
            if xstate.get(e.slot) != "x_reading":
                errors.append(f"t={e.t}: set x_free slot {e.slot} in state {xstate.get(e.slot)} (early free?)")
            else:
                xstate[e.slot] = "free"
        elif e.kind == "WAIT" and e.channel == y_free_channel:
            if ystate.get(e.slot) != "free":
                errors.append(f"t={e.t}: wait y_free slot {e.slot} in state {ystate.get(e.slot)}")
        elif e.kind == "VECTOR":
            if ystate.get(e.slot) != "free":
                errors.append(f"t={e.t}: main_vf writes y slot {e.slot} not free ({ystate.get(e.slot)})")
            else:
                ystate[e.slot] = "y_written"
        elif e.kind == "SET" and e.channel == y_ready_channel:
            if ystate.get(e.slot) != "y_written":
                errors.append(f"t={e.t}: set y_ready slot {e.slot} in state {ystate.get(e.slot)}")
            else:
                ystate[e.slot] = "y_ready"
        elif e.kind == "WAIT" and e.channel == y_ready_channel:
            if ystate.get(e.slot) != "y_ready":
                errors.append(f"t={e.t}: wait y_ready slot {e.slot} in state {ystate.get(e.slot)}")
            else:
                ystate[e.slot] = "y_reading"
        elif e.kind == "MTE3":
            if ystate.get(e.slot) != "y_reading":
                errors.append(f"t={e.t}: mte3 reads y slot {e.slot} in state {ystate.get(e.slot)}")
        elif e.kind == "SET" and e.channel == y_free_channel:
            if ystate.get(e.slot) != "y_reading":
                errors.append(f"t={e.t}: set y_free slot {e.slot} in state {ystate.get(e.slot)}")
            else:
                ystate[e.slot] = "free"
    leftover = {k: v for k, v in list(xstate.items()) + list(ystate.items()) if v != "free"}
    if leftover:
        errors.append("[M6-OUTSTANDING-GENERATION] slots not drained at end: %s" % leftover)
    return {"ok": not errors, "errors": errors, "events": len(ev),
            "x_versions": Vx, "y_versions": Vy}


# ---- deliberate corruption (for negative tests) ----
def _clone(ev):
    return [PEvent(e.kind, e.channel, e.slot, e.gen, e.task, e.t) for e in ev]


def mutate_drop_wait(events, channel, occurrence=0):
    ev = _clone(events); seen = 0; out = []
    for e in ev:
        if e.kind == "WAIT" and e.channel == channel:
            if seen == occurrence:
                seen += 1; continue
            seen += 1
        out.append(e)
    return out


def mutate_early_free(events, occurrence=0, free_channel=None, ready_channel=None):
    """Swap a FREE set to just before the read that consumes it -- an EARLY FREE.

    The channel names default to the legacy RMSNorm spellings so existing callers keep working,
    but a caller holding a DERIVED plan should pass the real names: mutating a channel that the
    program no longer contains produces an unchanged program, and an unchanged program passes,
    which is a negative test that cannot fail.
    """
    free_channel = free_channel or "x_free"
    ready_channel = ready_channel or "x_ready"
    ev = _clone(events)
    for i, e in enumerate(ev):
        if e.kind == "SET" and e.channel == free_channel:
            if occurrence == 0:
                j = i
                while j > 0 and not (ev[j].kind == "WAIT" and ev[j].channel == ready_channel):
                    j -= 1
                block = ev.pop(i)
                ev.insert(j, block)
                return ev
            occurrence -= 1
    return ev


def mutate_wrong_slot(events, occurrence=0, versions=None):
    """Shift one MTE2's slot to a DIFFERENT slot -- the tile-index-as-generation defect.

    WHY versions IS NOW A PARAMETER. This used to pick the ring width from the largest slot it
    happened to observe. On a program with a SINGLE logical iteration every slot is 0, so the
    width came out as 1 and the shift was '0 -> 0': the mutation changed nothing, the verifier
    passed, and the negative test silently reported success while proving nothing. The width
    must be the ring's declared version count, and a mutation that cannot change the slot is
    reported as a refusal to mutate rather than returned as an unchanged program.
    """
    ev = _clone(events)
    width = int(versions) if versions else 0
    if width <= 0:
        observed = [x.slot for x in ev if x.kind == "MTE2"]
        width = (max(observed) + 1) if observed else 0
    if width <= 1:
        raise ValueError(
            "cannot mutate a wrong slot: the program has a single slot, so any slot change"
            " would be indistinguishable from the correct program; pass the ring's version"
            " count to exercise this defect on a multi-version ring")
    seen = 0
    for e in ev:
        if e.kind == "MTE2":
            if seen == occurrence:
                e.slot = (e.slot + 1) % width
                return ev
            seen += 1
    return ev


def verify_text_events(problem, events, plan=None):
    """Re-print mutated events to text and verify (keeps the text-extraction path).

    Emits consistent per-channel flag ids and a trailing drain, so a mutation is
    rejected for the rule it violates, not for a missing drain/flag.
    """
    flags = {}
    nxt = 0
    for e in events:
        if e.kind in ("WAIT", "SET") and e.channel not in flags:
            flags[e.channel] = nxt; nxt += 1
    lines = []
    for e in events:
        if e.kind == "WAIT":
            lines.append("    wait_flag %s[%d] gen=%d flag=%d  # %s t=%d" % (e.channel, e.slot, e.gen, flags.get(e.channel, 0), e.task, e.t))
        elif e.kind == "SET":
            lines.append("    set_flag %s[%d] gen=%d flag=%d  # %s t=%d" % (e.channel, e.slot, e.gen, flags.get(e.channel, 0), e.task, e.t))
        elif e.kind == "MTE2":
            lines.append("    mte2 %s slot=%d t=%d" % (e.task, e.slot, e.t))
        elif e.kind == "MTE3":
            lines.append("    mte3 %s slot=%d t=%d" % (e.task, e.slot, e.t))
        elif e.kind == "VECTOR":
            lines.append("    simdvf %s slot=%d t=%d" % (e.task, e.slot, e.t))
    lines.append("  drain()")
    return verify(problem, chr(10).join(lines), plan)


_KNOWN_PIPES = {"mte2", "mte3", "vector", "scalar", "fix"}


def expected_pipes(plan) -> dict:
    """channel -> the (producer pipe, consumer pipe) the PLAN derived for it.

    WHY THIS IS DERIVED. This table used to be a literal dict keyed on the six RMSNorm channel
    names (x_ready, x_free, ...). Those names are now derived from the program, so a literal
    table stops matching the moment the program changes -- and a check that no longer matches
    its input does not fail loudly, it silently passes. The expected pair is therefore read
    from the same obligations the allocator used, which is the only set of pairs that can be
    wrong in the same way the plan is wrong.
    """
    pairs = {}
    for obligation in plan.get("obligations") or ():
        channel = str(obligation.get("channel") or "")
        if channel:
            pairs[channel] = (str(obligation.get("producer_pipe") or ""),
                              str(obligation.get("consumer_pipe") or ""))
    return pairs


def verify_pipes(plan) -> dict:
    """[M6-PIPE-MISMATCH]: every sync action must cross the pipe pair the plan derived.

    Pipes are a plan-level property (the TIR text does not encode them), so this checks
    plan['sync_actions'] against the derived channel->pipe map. Unknown pipes and same-pipe
    (no cross-engine) edges are also rejected.
    """
    errors = []
    table = expected_pipes(plan)
    for a in plan.get("sync_actions", []):
        ch = a.get("channel"); pp = a.get("producer_pipe"); cp = a.get("consumer_pipe")
        if pp not in _KNOWN_PIPES or cp not in _KNOWN_PIPES:
            errors.append("[M6-PIPE-MISMATCH] channel %s unknown pipe %s->%s" % (ch, pp, cp))
            continue
        exp = table.get(ch)
        if exp is None:
            # A sync action on a channel the plan never derived has no legal pipe pair at all.
            errors.append("[M6-PIPE-MISMATCH] channel %s is not in the plan's derived"
                          " obligations, so it has no legal pipe pair" % (ch,))
            continue
        if exp[0] and exp[1] and (pp, cp) != exp:
            errors.append("[M6-PIPE-MISMATCH] channel %s pipe %s->%s != expected %s->%s"
                          % (ch, pp, cp, exp[0], exp[1]))
        if pp == cp and exp[0] != exp[1]:
            errors.append("[M6-PIPE-MISMATCH] channel %s set/wait on the same pipe %s (its"
                          " derived edge does not cross engines)" % (ch, pp))
    return {"ok": not errors, "errors": errors, "rule": "M6-PIPE-MISMATCH"}


__all__ = ["PEvent", "parse_events", "parse_flags", "verify", "mutate_drop_wait",
           "mutate_early_free", "mutate_wrong_slot", "verify_text_events", "verify_pipes"]
