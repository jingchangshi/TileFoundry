"""Resolved-TIR verifier.

Events are re-parsed from the printed TIR text (not taken from the generator),
then replayed through per-slot state machines for the X and Y rings. Any use of
an unready buffer, write into an occupied slot, or generation mismatch is an
error. Deliberately broken variants (dropped wait, early free, wrong slot) must
be rejected.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_WAIT = re.compile(r"wait_flag\s+(\w+)\[(\d+)\]\s+gen=(\d+)\s+flag=\d+\s*#.*t=(\d+)")
_SET = re.compile(r"set_flag\s+(\w+)\[(\d+)\]\s+gen=(\d+)\s+flag=\d+\s*#.*t=(\d+)")
_OP = re.compile(r"(mte2|mte3|simdvf)\s+(\w+)\s+slot=(\d+)\s+t=(\d+)")


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
            out.append(PEvent("WAIT", m.group(1), int(m.group(2)), int(m.group(3)), "?", int(m.group(4))))
            continue
        m = _SET.match(s)
        if m:
            out.append(PEvent("SET", m.group(1), int(m.group(2)), int(m.group(3)), "?", int(m.group(4))))
            continue
        m = _OP.match(s)
        if m:
            kind = {"mte2": "MTE2", "mte3": "MTE3", "simdvf": "VECTOR"}[m.group(1)]
            out.append(PEvent(kind, "", int(m.group(3)), int(m.group(4)), m.group(2), int(m.group(4))))
    return out


def verify(problem, text: str):
    Vx = problem.policy["x_versions"]; Vy = problem.policy["y_versions"]
    ev = parse_events(text)
    xstate = {i: "free" for i in range(Vx)}
    ystate = {i: "free" for i in range(Vy)}
    errors = []
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
        errors.append(f"slots not drained at end: {leftover}")
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
                # move x_free set to before the previous WAIT x_ready
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
    """Re-print mutated events to text and verify (keeps the text-extraction path)."""
    from .materialize import _slot  # noqa
    lines = []
    for e in events:
        if e.kind == "WAIT":
            lines.append(f"    wait_flag {e.channel}[{e.slot}] gen={e.gen} flag=0  # {e.task} t={e.t}")
        elif e.kind == "SET":
            lines.append(f"    set_flag {e.channel}[{e.slot}] gen={e.gen} flag=0  # {e.task} t={e.t}")
        elif e.kind == "MTE2":
            lines.append(f"    mte2 {e.task} slot={e.slot} t={e.t}")
        elif e.kind == "MTE3":
            lines.append(f"    mte3 {e.task} slot={e.slot} t={e.t}")
        elif e.kind == "VECTOR":
            lines.append(f"    simdvf {e.task} slot={e.slot} t={e.t}")
    return verify(problem, "\n".join(lines))


__all__ = ["PEvent", "parse_events", "verify", "mutate_drop_wait", "mutate_early_free",
           "mutate_wrong_slot", "verify_text_events"]