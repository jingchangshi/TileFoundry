#!/usr/bin/env python3
"""Independent SPEC + checks for viewing one GMEM root at another dtype (P3/P4).

Motivation: casting a BF16 GMEM root to F32 for FP32 accumulation has NO representation as
a view in the current lowering ("gmem cross-dtype reuse has no offset-view representation").
This module defines the construct's semantics independently of that implementation, so the
implementation can be checked against it.

SEMANTICS (one root, byte-addressed):
  root        : [base_byte, base_byte + nbytes) in GM, with an allocation generation.
  view        : byte_start = root.base_byte + byte_offset
                byte_length = count * dtype_width
                range       = [byte_start, byte_start + byte_length)
  alignment   : (root.base_byte + byte_offset) % dtype_width == 0
  aliasing    : two views alias iff they name the same root AND their ranges intersect.
                Alias identity is (root, byte range); dtype is NOT part of identity.

Illegal cases, each with its own rule tag and a discriminating negative:
  VIEW_MISALIGNED_OFFSET        offset not a multiple of the target dtype width
  VIEW_RANGE_OVERFLOWS_PARENT   range end beyond the parent allocation
  VIEW_EXTENT_OVERFLOW          element count above the representational cap
  VIEW_OVERLAP_INCOMPATIBLE     distinct views alias with a write on either side
  VIEW_STALE_AFTER_REALLOC      view generation != parent allocation generation

Run: python tools/p3/view_spec.py   (exit 0 iff the matrix is discriminating and valid
controls pass). Pure Python; no device code; read-only on other modules.
"""
from __future__ import annotations

from dataclasses import dataclass

DTYPE_WIDTH = {"u8": 1, "bf16": 2, "f16": 2, "f32": 4, "i32": 4, "i64": 8, "f64": 8}
MAX_ELEMENTS = 1 << 32  # policy representational cap for a single view

RULES = ("VIEW_MISALIGNED_OFFSET", "VIEW_RANGE_OVERFLOWS_PARENT", "VIEW_EXTENT_OVERFLOW",
         "VIEW_OVERLAP_INCOMPATIBLE", "VIEW_STALE_AFTER_REALLOC")


@dataclass(frozen=True)
class GmemRoot:
    name: str
    base_byte: int
    nbytes: int
    generation: int = 0

    @property
    def end(self) -> int:
        return self.base_byte + self.nbytes


@dataclass(frozen=True)
class OffsetView:
    name: str
    root: str
    generation: int
    byte_offset: int
    dtype: str
    count: int

    @property
    def width(self) -> int:
        return DTYPE_WIDTH[self.dtype]

    @property
    def byte_length(self) -> int:
        return self.count * self.width


def _tags(root: GmemRoot, view: OffsetView) -> list[str]:
    """Rules violated by ONE view relative to its parent root (order-independent)."""
    tags = []
    if root.generation != view.generation:
        tags.append("VIEW_STALE_AFTER_REALLOC")
    width = view.width
    if (root.base_byte + view.byte_offset) % width != 0:
        tags.append("VIEW_MISALIGNED_OFFSET")
    if view.count > MAX_ELEMENTS:
        tags.append("VIEW_EXTENT_OVERFLOW")
    start = root.base_byte + view.byte_offset
    if start + view.byte_length > root.end:
        tags.append("VIEW_RANGE_OVERFLOWS_PARENT")
    return tags


def _overlaps(views: list[OffsetView], roots: dict[str, GmemRoot],
              accesses: dict[str, str]) -> list[str]:
    """Incompatible aliasing: distinct views on one root whose ranges intersect and
    where at least one side writes. read/read is compatible."""
    tags = []
    for a in range(len(views)):
        for b in range(a + 1, len(views)):
            va, vb = views[a], views[b]
            if va.root != vb.root:
                continue
            ra, rb = roots[va.root], roots[vb.root]
            sa, ea = ra.base_byte + va.byte_offset, ra.base_byte + va.byte_offset + va.byte_length
            sb, eb = rb.base_byte + vb.byte_offset, rb.base_byte + vb.byte_offset + vb.byte_length
            if sa < eb and sb < ea:  # intersect
                if accesses.get(va.name) == "write" or accesses.get(vb.name) == "write":
                    tags.append("VIEW_OVERLAP_INCOMPATIBLE")
    return tags


def alias_conflicts(views: list[OffsetView], accesses: dict[str, str]) -> list[tuple]:
    """Pairs of distinct views on one root whose ranges intersect with a write on either
    side. PUBLIC entry point for the P4 alias/plan checks, so the view contract and the
    plan cannot drift apart."""
    out = []
    for a in range(len(views)):
        for b in range(a + 1, len(views)):
            va, vb = views[a], views[b]
            if va.root != vb.root:
                continue
            sa, ea = va.byte_offset, va.byte_offset + va.byte_length
            sb, eb = vb.byte_offset, vb.byte_offset + vb.byte_length
            if sa < eb and sb < ea and (accesses.get(va.name) == "write"
                                        or accesses.get(vb.name) == "write"):
                out.append((va.name, vb.name))
    return out


def check(roots: dict[str, GmemRoot], views: list[OffsetView],
          accesses: dict[str, str]) -> dict[str, list[str]]:
    """Return {rule: [messages]} for every violated rule."""
    out: dict[str, list[str]] = {r: [] for r in RULES}
    for view in views:
        if view.root not in roots:
            out["VIEW_STALE_AFTER_REALLOC"].append("%s: unknown parent root %r" % (view.name, view.root))
            continue
        root = roots[view.root]
        for tag in _tags(root, view):
            out[tag].append("%s: root=%s offset=%d dtype=%s count=%d range=[%d,%d) parent=[%d,%d)"
                            % (view.name, view.root, view.byte_offset, view.dtype, view.count,
                               root.base_byte + view.byte_offset,
                               root.base_byte + view.byte_offset + view.byte_length,
                               root.base_byte, root.end))
    for msg in _overlaps(views, roots, accesses):
        out[msg].append("distinct views alias with a write on one side")
    return out


def _matrix():
    """(name, roots, views, accesses, expected_rule_or_None)"""
    r = {"buf": GmemRoot("buf", 512, 1024, generation=3)}
    cases = []
    # valid control: bf16 root viewed at f32 covering exactly the allocation
    cases.append(("valid_cross_dtype_view", r,
                  [OffsetView("v_f32", "buf", 3, 0, "f32", 256)], {"v_f32": "read"}, None))
    # misaligned only
    cases.append(("misaligned_f32_offset", r,
                  [OffsetView("v", "buf", 3, 2, "f32", 8)], {"v": "read"}, "VIEW_MISALIGNED_OFFSET"))
    # range exceeds parent only (aligned, small count)
    cases.append(("range_exceeds_parent", r,
                  [OffsetView("v", "buf", 3, 768, "f32", 128)], {"v": "read"},
                  "VIEW_RANGE_OVERFLOWS_PARENT"))
    # extent cap only (u8, count above cap, fits in parent via huge root)
    big = {"big": GmemRoot("big", 0, 1 << 40, generation=1)}
    cases.append(("extent_overflow", big,
                  [OffsetView("v", "big", 1, 0, "u8", (1 << 32) + 1)], {"v": "read"},
                  "VIEW_EXTENT_OVERFLOW"))
    # incompatible overlap: bf16 writer [0,512) and f32 reader [256,512)
    cases.append(("overlap_write_read", r,
                  [OffsetView("w_bf16", "buf", 3, 0, "bf16", 256),
                   OffsetView("r_f32", "buf", 3, 256, "f32", 64)],
                  {"w_bf16": "write", "r_f32": "read"}, "VIEW_OVERLAP_INCOMPATIBLE"))
    # read/read overlap is COMPATIBLE
    cases.append(("overlap_read_read_ok", r,
                  [OffsetView("a", "buf", 3, 0, "bf16", 256),
                   OffsetView("b", "buf", 3, 256, "f32", 64)],
                  {"a": "read", "b": "read"}, None))
    # stale view only
    cases.append(("stale_after_realloc", r,
                  [OffsetView("v", "buf", 2, 0, "f32", 8)], {"v": "read"},
                  "VIEW_STALE_AFTER_REALLOC"))
    return cases


def main() -> int:
    cases = _matrix()
    failures = 0
    for name, roots, views, accesses, expected in cases:
        res = check(roots, views, accesses)
        fired = sorted(rule for rule, msgs in res.items() if msgs)
        if expected is None:
            ok = not fired
        else:
            ok = fired == [expected]  # DISCRIMINATING: exactly this rule
        failures += 0 if ok else 1
        print("%-26s expected=%-28s fired=%-32s %s"
              % (name, expected, fired or "[]", "PASS" if ok else "FAIL"))
    print("RESULT:", "ALL CASES DISCRIMINATING" if failures == 0 else "FAILURES=%d" % failures)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
