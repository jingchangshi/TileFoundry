#!/usr/bin/env python3
"""task-17 BRUTEFORCE: independent exhaustive reference for fixed-II modular scheduling.

Pure-data problem view (field names are THIS script's own; a schema adapter is
added later by whoever consumes it):
  problem = {"stage_max": int,
             "tasks":   [{"name", "pipe", "latency", "resources": {res: units}, "stage": None | int}],
             "edges":   [{"u", "v", "d", "L"}],       # producer u -> consumer v
             "buffers": [{"name", "space", "slot_bytes", "writer", "readers": [name], "read_latency",
                          "versions": ("fixed", n) | ("auto", (lo, hi)),
                          "shift": ("fixed", k) | ("auto", (lo, hi))}],
             "resources": {res: capacity}, "capacity": {space: bytes}}
Feasibility (independently derived from the task-17 spec):
  t = II*stage + phase, 0 <= phase < II
  edge (u,v,d,L):    t_v + d*II >= t_u + L
  buffer (V,delta):  t_writer + (V+delta)*II >= max_r(t_reader_r + read_latency_r)
  resource ring:     circular [phase, phase+latency) weighted intervals; coverage <= capacity
                     everywhere incl. across the II wrap; same phase + same resource follows
                     declaration (source) order in issue_order
  capacity:          sum(V * slot_bytes) per space <= capacity[space]
Run: python3 docs/handoff/probes/bruteforce_schedule.py  (prints PASS/FAIL per test)
"""
from itertools import product as _product

def _validate(p):
    names = [t["name"] for t in p["tasks"]]
    assert len(names) == len(set(names)), "duplicate task name"
    have = set(names)
    assert all(e["u"] in have and e["v"] in have for e in p["edges"]), "edge endpoints must be tasks"
    assert all(b["writer"] in have and set(b["readers"]) <= have for b in p["buffers"]), "buffer users must be tasks"

def _ring_ok(intervals, cap, ii):
    """Circular coverage of (phase, latency, units) intervals must stay <= cap."""
    ev = {}
    for phase, length, units in intervals:
        if length <= 0:
            continue
        if length >= ii:  # covers the whole period
            ev[0] = ev.get(0, 0) + units; ev[ii] = ev.get(ii, 0) - units; continue
        end = phase + length
        ev[phase] = ev.get(phase, 0) + units
        if end > ii:  # wraps past the II boundary: split into two segments
            ev[ii] = ev.get(ii, 0) - units; ev[0] = ev.get(0, 0) + units; ev[end - ii] = ev.get(end - ii, 0) - units
        else:
            ev[end] = ev.get(end, 0) - units
    cur = 0
    for x in sorted(ev):
        if cur > cap:
            return False
        cur += ev[x]
    return cur <= cap

def _domain(spec):
    kind, val = spec
    return [val] if kind == "fixed" else list(range(val[0], val[1] + 1))

def solve(p, ii):
    """Exhaustive check of one fixed II. Returns ("SAT", assignment) or ("UNSAT", info)."""
    _validate(p)
    tasks, buffers = p["tasks"], p.get("buffers", [])
    t_dom = [[(t["stage"] if t["stage"] is not None else s, ph) for s in range(p["stage_max"] + 1)
              for ph in range(ii)] for t in tasks]
    b_dom = [[(v, d) for v in _domain(b["versions"]) for d in _domain(b["shift"])] for b in buffers]
    caps, spaces = p["resources"], p.get("capacity", {})
    tried, violated = 0, set()
    for combo in _product(*t_dom, *b_dom):
        tried += 1
        st = {t["name"]: combo[i] for i, t in enumerate(tasks)}
        t_ = {n: ii * s + ph for n, (s, ph) in st.items()}
        if any(t_[e["v"]] + e["d"] * ii < t_[e["u"]] + e["L"] for e in p["edges"]):
            violated.add("edge"); continue
        bv = {buffers[i]["name"]: combo[len(tasks) + i] for i in range(len(buffers))}
        if any(t_[b["writer"]] + (bv[b["name"]][0] + bv[b["name"]][1]) * ii
               < max(t_[r] + b["read_latency"] for r in b["readers"]) for b in buffers):
            violated.add("version"); continue
        use = {}
        for t in tasks:
            for res, units in [(t["pipe"], 1), *t.get("resources", {}).items()]:
                use.setdefault(res, []).append((st[t["name"]][1], t["latency"], units))
        if any(not _ring_ok(iv, caps[res], ii) for res, iv in use.items()):
            violated.add("ring"); continue
        fill = {}
        for i, b in enumerate(buffers):
            fill[b["space"]] = fill.get(b["space"], 0) + bv[b["name"]][0] * b["slot_bytes"]
        if any(fill.get(sp, 0) > cap for sp, cap in spaces.items()):
            violated.add("capacity"); continue
        order = sorted(tasks, key=lambda t: (t_[t["name"]], tasks.index(t)))
        return "SAT", {"ii": ii, "tasks": st, "t": t_, "buffers": bv, "issue_order": [t["name"] for t in order]}
    return "UNSAT", {"ii": ii, "tried": tried, "violated": sorted(violated)}

def min_ii(p, ii_max):
    """Smallest feasible II with the proof that every smaller II is UNSAT."""
    verdicts = {}
    for ii in range(1, ii_max + 1):
        verdicts[ii] = solve(p, ii)
        if verdicts[ii][0] == "SAT":
            return "SAT", ii, verdicts[ii][1], {k: "UNSAT" for k in range(1, ii)}
    return "UNSAT", None, None, {k: v[0] for k, v in verdicts.items()}

def _t(name, pipe, latency, stage=None, **kw):
    return {"name": name, "pipe": pipe, "latency": latency, "stage": stage, **kw}

def _buf(name, writer, readers, versions, space="smem", slot_bytes=1, read_latency=1):
    return {"name": name, "space": space, "slot_bytes": slot_bytes, "writer": writer, "readers": readers,
            "read_latency": read_latency, "versions": versions, "shift": ("fixed", 0)}

def _self_tests():
    r = []
    check = lambda name, cond: r.append((name, bool(cond)))
    p1 = {"stage_max": 1, "tasks": [_t("A", "v", 1)], "edges": [], "buffers": [], "resources": {"v": 1}, "capacity": {}}
    s1 = solve(p1, 2)
    check("T1 sat-and-placement", s1[0] == "SAT" and s1[1]["tasks"]["A"] == (0, 0))
    p2 = {"stage_max": 2, "tasks": [_t("A", "v", 1), _t("B", "v", 1)], "edges": [{"u": "A", "v": "B", "d": 0, "L": 1}],
          "buffers": [], "resources": {"v": 1}, "capacity": {}}
    check("T2-ii1-unsat", solve(p2, 1)[0] == "UNSAT")
    s2 = min_ii(p2, 4)
    check("T2-optimal-ii-2", s2[0] == "SAT" and s2[1] == 2 and s2[3] == {1: "UNSAT"})
    p3 = {"stage_max": 1, "tasks": [_t("W", "v", 1), _t("R", "w", 1)],
          "edges": [{"u": "W", "v": "R", "d": 0, "L": 1}], "buffers": [_buf("b", "W", ["R"], ("fixed", 1))],
          "resources": {"v": 1, "w": 1}, "capacity": {"smem": 16}}
    s3 = solve(p3, 2)
    check("T3a-fixed-v1-sat", s3[0] == "SAT" and s3[1]["buffers"]["b"] == (1, 0))
    p4 = {"stage_max": 1, "tasks": [_t("W", "v", 1, stage=0), _t("R", "w", 1, stage=1)],
          "edges": [{"u": "W", "v": "R", "d": 0, "L": 2}], "buffers": [_buf("b", "W", ["R"], ("fixed", 1))],
          "resources": {"v": 1, "w": 1}, "capacity": {"smem": 16}}
    check("T3b-fixed-v1-unsat", solve(p4, 2)[0] == "UNSAT")
    s4 = solve(dict(p4, buffers=[_buf("b", "W", ["R"], ("auto", (1, 2)))]), 2)
    check("T3b-auto-v2-sat", s4[0] == "SAT" and s4[1]["buffers"]["b"] == (2, 0))
    p5 = {"stage_max": 1, "tasks": [_t("P", "v", 3), _t("Q", "v", 3)], "edges": [], "buffers": [], "resources": {"v": 1}, "capacity": {}}
    check("T4-ii4-unsat", solve(p5, 4)[0] == "UNSAT")
    check("T4-ii5-unsat", solve(p5, 5)[0] == "UNSAT")
    s5 = min_ii(p5, 8)
    check("T4-optimal-ii-6", s5[0] == "SAT" and s5[1] == 6 and s5[3] == {k: "UNSAT" for k in range(1, 6)})
    base = {"stage_max": 0, "tasks": [_t("P", "v", 1), _t("Q", "w", 1)], "edges": [],
            "buffers": [_buf("b1", "P", ["P"], ("fixed", 1), slot_bytes=10, read_latency=1),
                        _buf("b2", "Q", ["Q"], ("fixed", 1), slot_bytes=10, read_latency=1)], "resources": {"v": 1, "w": 1}}
    check("T5-cap15-unsat", solve(dict(base, capacity={"smem": 15}), 1)[0] == "UNSAT")
    check("T5-cap30-sat", solve(dict(base, capacity={"smem": 30}), 1)[0] == "SAT")
    p6 = {"stage_max": 1, "tasks": [_t("A", "v", 1, stage=1), _t("B", "w", 1)],
          "edges": [{"u": "A", "v": "B", "d": 0, "L": 1}], "buffers": [], "resources": {"v": 1, "w": 1}, "capacity": {}}
    s6 = solve(p6, 2)
    check("T6-pinned-sat", s6[0] == "SAT" and s6[1]["tasks"]["A"] == (1, 0) and s6[1]["tasks"]["B"] == (1, 1))
    p6b = dict(p6, tasks=[_t("A", "v", 1, stage=1), _t("B", "w", 1, stage=0)])
    check("T6-pinned-unsat", solve(p6b, 2)[0] == "UNSAT")
    p7 = {"stage_max": 0, "tasks": [_t("P", "v", 1), _t("Q", "v", 1)], "edges": [], "buffers": [],
          "resources": {"v": 2}, "capacity": {}}
    s7 = solve(p7, 1)
    check("T7-cap2-sat-order", s7[0] == "SAT" and s7[1]["issue_order"] == ["P", "Q"])
    s7b = solve(dict(p7, tasks=[_t("Q", "v", 1), _t("P", "v", 1)]), 1)
    check("T7-order-flips", s7b[0] == "SAT" and s7b[1]["issue_order"] == ["Q", "P"])
    return r

if __name__ == "__main__":
    import sys
    fails = 0
    for name, ok in _self_tests():
        print(("PASS" if ok else "FAIL"), name)
        fails += 0 if ok else 1
    print("SUMMARY:", "ALL PASS" if fails == 0 else f"{fails} FAIL")
    sys.exit(1 if fails else 0)
