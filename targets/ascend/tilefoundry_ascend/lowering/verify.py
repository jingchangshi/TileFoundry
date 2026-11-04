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


def verify(problem, text: str):
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
    # M6-UNKNOWN-CHANNEL: a WAIT/SET on a channel outside the known sync set is rejected.
    _KNOWN_CHANNELS = {"x_ready", "x_free", "y_ready", "y_free", "w_ready", "w_conv"}
    for e in ev:
        if e.kind in ("WAIT", "SET") and e.channel not in _KNOWN_CHANNELS:
            errors.append("[M6-UNKNOWN-CHANNEL] unknown sync channel: " + e.channel)
    # [M6-MISSING-X-FREE-WAIT]
    xload_mtes = [e for e in ev if e.kind == "MTE2" and e.t >= Vx]
    n_xfree_wait = sum(1 for e in ev if e.kind == "WAIT" and e.channel == "x_free")
    if n_xfree_wait < len(xload_mtes):
        errors.append("[M6-MISSING-X-FREE-WAIT] %d x_free wait(s) for %d x_load write(s) needing one"
                      % (n_xfree_wait, len(xload_mtes)))
    # ---- per-slot state machine (unchanged semantics) ----
    xstate = {i: "free" for i in range(Vx)}
    ystate = {i: "free" for i in range(Vy)}
    def _ver_for(e):
        if e.channel in ("x_free", "x_ready") or e.kind == "MTE2":
            return Vx
        if e.channel in ("y_free", "y_ready") or e.kind in ("MTE3", "VECTOR"):
            return Vy
        return None
    for e in ev:
        V = _ver_for(e)
        if V:
            if e.gen != e.t:
                errors.append(f"t={e.t}: {e.kind}/{e.channel} gen {e.gen} != logical iteration t")
            if e.slot != (e.gen % V):
                errors.append(f"t={e.t}: {e.kind}/{e.channel} slot {e.slot} != gen%{V} ({e.gen % V}); tile-index used as generation?")
        if e.kind == "WAIT" and e.channel == "x_free":
            if xstate.get(e.slot) != "free":
                errors.append(f"t={e.t}: wait x_free on slot {e.slot} in state {xstate.get(e.slot)}")
        elif e.kind == "MTE2":
            if e.slot not in xstate or xstate[e.slot] != "free":
                errors.append(f"t={e.t}: mte2 writes x slot {e.slot} not free ({xstate.get(e.slot)})")
            else:
                xstate[e.slot] = "x_written"
        elif e.kind == "SET" and e.channel == "x_ready":
            if xstate.get(e.slot) != "x_written":
                errors.append(f"t={e.t}: set x_ready slot {e.slot} in state {xstate.get(e.slot)}")
            else:
                xstate[e.slot] = "x_ready"
        elif e.kind == "WAIT" and e.channel == "x_ready":
            if xstate.get(e.slot) != "x_ready":
                errors.append(f"t={e.t}: wait x_ready slot {e.slot} in state {xstate.get(e.slot)}")
            else:
                xstate[e.slot] = "x_reading"
        elif e.kind == "SET" and e.channel == "x_free":
            if xstate.get(e.slot) != "x_reading":
                errors.append(f"t={e.t}: set x_free slot {e.slot} in state {xstate.get(e.slot)} (early free?)")
            else:
                xstate[e.slot] = "free"
        elif e.kind == "WAIT" and e.channel == "y_free":
            if ystate.get(e.slot) != "free":
                errors.append(f"t={e.t}: wait y_free slot {e.slot} in state {ystate.get(e.slot)}")
        elif e.kind == "VECTOR":
            if ystate.get(e.slot) != "free":
                errors.append(f"t={e.t}: main_vf writes y slot {e.slot} not free ({ystate.get(e.slot)})")
            else:
                ystate[e.slot] = "y_written"
        elif e.kind == "SET" and e.channel == "y_ready":
            if ystate.get(e.slot) != "y_written":
                errors.append(f"t={e.t}: set y_ready slot {e.slot} in state {ystate.get(e.slot)}")
            else:
                ystate[e.slot] = "y_ready"
        elif e.kind == "WAIT" and e.channel == "y_ready":
            if ystate.get(e.slot) != "y_ready":
                errors.append(f"t={e.t}: wait y_ready slot {e.slot} in state {ystate.get(e.slot)}")
            else:
                ystate[e.slot] = "y_reading"
        elif e.kind == "MTE3":
            if ystate.get(e.slot) != "y_reading":
                errors.append(f"t={e.t}: mte3 reads y slot {e.slot} in state {ystate.get(e.slot)}")
        elif e.kind == "SET" and e.channel == "y_free":
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


def mutate_early_free(events, occurrence=0):
    """Swap a SET x_free to just before the matching VF read (x_ready wait)."""
    ev = _clone(events)
    for i, e in enumerate(ev):
        if e.kind == "SET" and e.channel == "x_free":
            if occurrence == 0:
                j = i
                while j > 0 and not (ev[j].kind == "WAIT" and ev[j].channel == "x_ready"):
                    j -= 1
                block = ev.pop(i)
                ev.insert(j, block)
                return ev
            occurrence -= 1
    return ev


def mutate_wrong_slot(events, occurrence=0):
    ev = _clone(events); seen = 0
    for e in ev:
        if e.kind == "MTE2":
            if seen == occurrence:
                e.slot = (e.slot + 1) % max(1, (max([x.slot for x in ev if x.kind == 'MTE2']) or 0) + 1)
                return ev
            seen += 1
    return ev


def verify_text_events(problem, events):
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
    return verify(problem, chr(10).join(lines))


#: The only legal pipe pair for each logical channel. A Set/Wait edge that does not
#: cross exactly this producer->consumer pipe pair is a pipe mismatch.
_EXPECTED_PIPES = {
    "x_ready": ("mte2", "vector"),
    "x_free": ("vector", "mte2"),
    "y_ready": ("vector", "mte3"),
    "y_free": ("mte3", "vector"),
    "w_ready": ("mte2", "vector"),
    "w_conv": ("vector", "vector"),
}
_KNOWN_PIPES = {"mte2", "mte3", "vector", "scalar", "fix"}


def verify_pipes(plan) -> dict:
    """[M6-PIPE-MISMATCH]: every sync action must cross the legal producer->consumer pipe.

    Pipes are a plan-level property (the TIR text does not encode them), so this checks
    plan['sync_actions'] against the canonical channel->pipe table. Unknown pipes and
    same-pipe (no cross-engine) edges are also rejected.
    """
    errors = []
    for a in plan.get("sync_actions", []):
        ch = a.get("channel"); pp = a.get("producer_pipe"); cp = a.get("consumer_pipe")
        if pp not in _KNOWN_PIPES or cp not in _KNOWN_PIPES:
            errors.append("[M6-PIPE-MISMATCH] channel %s unknown pipe %s->%s" % (ch, pp, cp))
            continue
        exp = _EXPECTED_PIPES.get(ch)
        if exp is not None and (pp, cp) != exp:
            errors.append("[M6-PIPE-MISMATCH] channel %s pipe %s->%s != expected %s->%s"
                          % (ch, pp, cp, exp[0], exp[1]))
        if pp == cp and ch not in ("w_conv",):
            errors.append("[M6-PIPE-MISMATCH] channel %s set/wait on the same pipe %s (no cross-engine edge)" % (ch, pp))
    return {"ok": not errors, "errors": errors, "rule": "M6-PIPE-MISMATCH"}


__all__ = ["PEvent", "parse_events", "parse_flags", "verify", "mutate_drop_wait",
           "mutate_early_free", "mutate_wrong_slot", "verify_text_events", "verify_pipes"]
