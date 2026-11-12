#!/usr/bin/env python3
"""Regression test for the audit hole in view_spec_target.py.

Pre-fix, the cross-check PRINTED "NON-CONFORMING" and still exited 0: the conformance
column was decorative. A wrong target validator (or a later regression in the real one)
could therefore pass the audit. This test pins the post-fix contract by running the real
script as a subprocess against:

  * the committed I1 validator              -> exit 0, no NON-CONFORMING row;
  * a target that accepts an overlapping WR -> exit NON-zero, NON-CONFORMING on the WR row;
  * a target that rejects a read/read pair  -> exit NON-zero, NON-CONFORMING on the RR row.

The stub modules are generated in a tempdir, so the test carries its own fixtures and does
not depend on evidence directories.

Run: python tools/p3/test_view_spec_target_conformance.py   (exit 0 iff all three hold)
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TF_ROOT = HERE.parents[1]
SCRIPT = HERE / "view_spec_target.py"
I1 = TF_ROOT / "src/tilefoundry/ir/types/dtype_view.py"

STUB_HEAD = '''from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class DtypeViewSpec:
    root: str
    dtype: str
    element_bytes: int
    byte_offset: int
    extent: int
    alignment: int
    root_bytes: int
    access: str


class DtypeViewError(ValueError):
    def __init__(self, reason, message):
        super().__init__("%s: %s" % (reason, message))
        self.reason = reason
'''

ACCEPTS_ALL = STUB_HEAD + '''

def validate_view(view):
    return None


def validate_views(views):
    return None
'''

REJECTS_RR = STUB_HEAD + '''

def _r(v):
    return v.byte_offset, v.byte_offset + v.extent * v.element_bytes


def validate_views(views):
    if len(views) < 2:
        return
    sa, ea = _r(views[0])
    sb, eb = _r(views[1])
    if sa < eb and sb < ea:
        raise DtypeViewError("overlapping-write", "WRONG: read/read refused")


def validate_view(view):
    return None
'''

FAILS = 0


def check(cond: bool, label: str) -> None:
    global FAILS
    if not cond:
        FAILS += 1
    print("%-78s %s" % (label, "PASS" if cond else "FAIL"))


def table(stdout: str) -> dict[str, str]:
    """Parse the conformance table: case name -> last column token."""
    out = {}
    for line in stdout.splitlines():
        tok = line.split()
        if len(tok) >= 4 and tok[0] in CASE_NAMES:
            out[tok[0]] = tok[-1]
    return out


CASE_NAMES = {"valid_full_bf16_to_f32", "valid_read_read_overlap_diff_dtype", "misaligned",
              "range_exceeds_parent", "overlap_write_read_same_dtype", "extent_overflow",
              "stale_after_realloc"}


def run(path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), str(path)],
                          capture_output=True, text=True,
                          env={"PYTHONPATH": str(HERE), "PYTHONDONTWRITEBYTECODE": "1",
                               "PATH": "/usr/bin:/bin"})


def main() -> int:
    real = run(I1)
    rows = table(real.stdout)
    check(real.returncode == 0, "real I1 validator: exit 0 (got %d)" % real.returncode)
    check(rows.get("overlap_write_read_same_dtype") == "CONFORMS",
          "real I1 validator: write/read overlap CONFORMS (got %r)"
          % rows.get("overlap_write_read_same_dtype"))
    check(rows.get("valid_read_read_overlap_diff_dtype") == "CONFORMS",
          "real I1 validator: read/read overlap CONFORMS (got %r)"
          % rows.get("valid_read_read_overlap_diff_dtype"))
    check(rows.get("extent_overflow") == "SCOPE-GAP"
          and rows.get("stale_after_realloc") == "SCOPE-GAP",
          "declared scope gaps stay explicitly unverified (not silently conforming)")
    check("NON-CONFORMING" not in rows.values(),
          "real I1 validator: no NON-CONFORMING row: %r" % (sorted(rows.items()),))

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        (td / "accepts_all.py").write_text(ACCEPTS_ALL)
        (td / "rejects_rr.py").write_text(REJECTS_RR)

        bad = run(td / "accepts_all.py")
        bad_rows = table(bad.stdout)
        check(bad.returncode != 0,
              "target accepting a write/read overlap: exit NON-zero (got %d)" % bad.returncode)
        check(bad_rows.get("overlap_write_read_same_dtype") == "NON-CONFORMING",
              "  ... row is NON-CONFORMING: %r"
              % bad_rows.get("overlap_write_read_same_dtype"))
        check(bad_rows.get("valid_read_read_overlap_diff_dtype") == "CONFORMS",
              "  ... the unrelated read/read row still CONFORMS")

        bad2 = run(td / "rejects_rr.py")
        bad2_rows = table(bad2.stdout)
        check(bad2.returncode != 0,
              "target rejecting a read/read overlap: exit NON-zero (got %d)" % bad2.returncode)
        check(bad2_rows.get("valid_read_read_overlap_diff_dtype") == "NON-CONFORMING",
              "  ... row is NON-CONFORMING: %r"
              % bad2_rows.get("valid_read_read_overlap_diff_dtype"))
        check(bad2_rows.get("overlap_write_read_same_dtype") == "CONFORMS",
              "  ... the unrelated write/read row still CONFORMS")

    print("RESULT:", "PASS" if FAILS == 0 else "FAILURES=%d" % FAILS)
    return 0 if FAILS == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
