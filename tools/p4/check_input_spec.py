#!/usr/bin/env python3
"""Self-check for tools/p4/INPUT_SPEC.md: every field row must state a source, a
consumer and an error state, and the normative sections must be present.

This exists because a spec whose tables have empty cells is a RITUAL, not a contract.
Run: python tools/p4/check_input_spec.py  (exit 0 iff the checks hold)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

SPEC = Path(__file__).resolve().parent / "INPUT_SPEC.md"
REQUIRED_SECTIONS = [
    "输入来源与作废边界",
    "Field catalogue",
    "Error states",
    "Units (single table, no mixing)",
    "three easiest places",
]
REQUIRED_TOKENS = [
    "sat", "unsat", "unknown", "fallback", "ROW_WISE_W", "SELF_GENERATED_INPUT_SOURCE",
    "PROTOCOL_SHA_MISMATCH", "INPUT_IDENTITY_FAILED", "EMPTY_CASE_SET",
    "alias_conflicts", "byte_offset", "distance", "versions", "owner_loop",
    "facts_digest", "V01–V10", "c8e21a39", "PROTOCOL_NOT_FROZEN", "MODEL_AS_EVIDENCE",
]
FAILS = 0


def check(cond, label):
    global FAILS
    if not cond:
        FAILS += 1
    print("%-82s %s" % (label, "PASS" if cond else "FAIL"))


def main() -> int:
    text = SPEC.read_text()
    for s in REQUIRED_SECTIONS:
        check(s in text, "section present: %s" % s)
    for tok in REQUIRED_TOKENS:
        check(tok in text, "required token: %s" % tok)
    # Field rows are the 5-column tables in section 3 and section 2.
    rows = []
    for line in text.splitlines():
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) == 5 and cells[0] not in ("field", "ID", "state", "quantity") \
                and not set(cells[0]) <= set("-: "):
            rows.append(cells)
    check(len(rows) >= 25, "at least 25 field rows (got %d)" % len(rows))
    empty_source = [r[0] for r in rows if not r[2].strip()]
    empty_consumer = [r[0] for r in rows if not r[3].strip()]
    empty_error = [r[0] for r in rows if not r[4].strip()]
    check(not empty_source, "every field row has a SOURCE: %r" % empty_source)
    check(not empty_consumer, "every field row has a CONSUMER: %r" % empty_consumer)
    check(not empty_error, "every field row has an ERROR STATE: %r" % empty_error)
    # At least a majority of source cells must carry a file:line anchor (not prose only).
    anchored = [r for r in rows if re.search(r"[A-Za-z0-9_]+\.(py|sh|md|json):\d+", r[2])]
    check(len(anchored) >= len(rows) * 0.6,
          "majority of sources carry file:line anchors (%d/%d)" % (len(anchored), len(rows)))
    check("PROPOSED" in text and "NOT AN IMPLEMENTATION" in text,
          "spec is explicitly PROPOSED, not an implementation claim")
    print("RESULT:", "PASS" if FAILS == 0 else "FAILURES=%d" % FAILS)
    return 0 if FAILS == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
