# Offset-view SPEC and checks (independent, CPU-only)

Artifact: tools/p3/view_spec.py (pure Python, no device code, read-only on other modules).

## 1. Semantics

A GMEM root is a byte-addressed allocation [base_byte, base_byte + nbytes) with an
allocation GENERATION. A view reinterprets part of that root at another dtype:

  byte_offset   BYTES from the root base (not elements)
  count         ELEMENT count in the target dtype
  width         target dtype width in bytes (u8 1, bf16/f16 2, f32/i32 4, i64/f64 8)
  byte_length   count * width
  range         [base_byte + byte_offset, base_byte + byte_offset + byte_length)
  alignment     (base_byte + byte_offset) % width == 0   (natural alignment for the dtype)
  aliasing      two views alias iff same root AND ranges intersect.
                Alias identity is (root, byte range); dtype is NOT part of the identity.

The motivating legal case (I1's blocker): a BF16 root viewed at F32 is legal iff the view
is aligned, stays inside the parent, does not overlap incompatibly, is within the count cap,
and is not stale.

## 2. Illegal cases, each with a discriminating check

| rule | condition | discriminating negative |
|---|---|---|
| VIEW_MISALIGNED_OFFSET | (base+offset) % width != 0 | f32 view at byte offset 2 |
| VIEW_RANGE_OVERFLOWS_PARENT | range end > parent end | f32 view at offset 768, count 128 in a 1024-byte root |
| VIEW_EXTENT_OVERFLOW | count > 2^32 (policy cap) | u8 view with count 2^32+1 in a 1 TiB root |
| VIEW_OVERLAP_INCOMPATIBLE | distinct views alias AND at least one writes | bf16 writer [0,512) + f32 reader [256,512) |
| VIEW_STALE_AFTER_REALLOC | view.generation != root.generation | view generation 2 against root generation 3 |

Each negative fires EXACTLY its own rule (the others are silent), and two valid controls
pass: the cross-dtype bf16->f32 view covering the whole allocation, and a read/read overlap.
Run: python tools/p3/view_spec.py -> "ALL CASES DISCRIMINATING".

## 3. Hardware/CANN forbidden vs our policy (honest ceiling)

This work is CPU-only and cites no CANN/ISA evidence. Therefore:

- NOT ESTABLISHED as hardware-forbidden: misaligned offset, range overflow, incompatible
  overlap, extent overflow, stale view. Natural alignment is LIKELY required by vector
  loads, but I have no selected-build evidence, so it is NOT claimed as a hardware rule.
- OUR POLICY (chosen to protect the compiler): all five. Overlapping views are not tracked
  by hardware, so an incompatible overlap is a race we refuse statically. The extent cap is
  a representational policy. Staleness is a compiler lifetime contract.

Consequence: passing these checks means the view obeys OUR contract, not that the device
would fault otherwise. Any claim that "hardware forbids X" needs a selected-build microtest;
until then X is policy. This ceiling is deliberate.


## 4. Target-side adaptation (I1's DtypeViewSpec, READ-ONLY)

Adapter: tools/p3/view_spec_target.py. It loads I1's class by path and adapts his objects
into GmemRoot/OffsetView; it never writes to his file.
State at wiring time: src/tilefoundry/ir/types/dtype_view.py was UNTRACKED (uncommitted),
sha256 61a28adf64440bda8a4c74cad92ad2e9e45f7319b3ce5fd0a262ddc611a6986a. Results are
valid for THAT sha only; re-run after he lands it.

On his real view objects (7/7 cases PASS):
  valid_full_bf16_to_f32                 my rules []                              he ACCEPTs
  valid_read_read_overlap_diff_dtype     my rules []                              he REJECTs (drift)
  misaligned                             VIEW_MISALIGNED_OFFSET                   he: misaligned
  range_exceeds_parent                   VIEW_RANGE_OVERFLOWS_PARENT              he: out-of-bounds
  overlap_write_read_same_dtype          VIEW_OVERLAP_INCOMPATIBLE                he ACCEPTs (drift)
  extent_overflow                        VIEW_EXTENT_OVERFLOW                     he ACCEPTs (no cap)
  stale_after_realloc                    VIEW_STALE_AFTER_REALLOC                 he ACCEPTs (no generation)

### Drift that must be reconciled (not papered over)
1. OVERLAP POLICY DIFFERS. This spec forbids any distinct-view overlap with a WRITE on
   either side (access-based). I1's validator forbids overlap only when the two views have
   DIFFERENT element sizes (dtype-based). Consequently:
     - same-dtype overlapping W x R: spec FORBIDS, his ACCEPTS;
     - different-dtype overlapping R x R: spec ALLOWS, his REJECTS.
   These are not the same contract. P4 must not mix them.
2. NO ALLOCATION GENERATION on his objects -> the STALE rule cannot be evaluated on them;
   the adapter had to supply a generation to exercise it.
3. NO COUNT CAP on his objects -> the EXTENT rule cannot be evaluated on them; he catches
   only the byte-range consequence (out-of-bounds).
4. His validator correctly REFUSES the motivating same-element-count bf16->f32 view as
   out-of-bounds (it needs twice the bytes), which matches the reported blocker.

## 5. P4 alias/plan wiring

view_spec.alias_conflicts(views, accesses) is the public entry point: pairs of distinct
views on one root whose ranges intersect with a write on either side. P4 alias/plan checks
must call THIS function (not a copy) so the plan and the view contract share one definition
and cannot drift apart. The rule is access-based, matching section 2.


### LEAD RULING (binding): ACCESS-BASED is the contract

Ruled 2026-11-04: the aliasing hazard is a WRITE and a READ against intersecting bytes
whose ORDER is undefined; the element width is irrelevant to that hazard. Therefore:
  * any two distinct views on one root with intersecting byte ranges where EITHER side is
    written are FORBIDDEN -- same-dtype W x R is exactly as unsafe as cross-dtype;
  * read/read is ALLOWED regardless of dtype.
I1's dtype-based validator (forbid only when element sizes differ) is NON-CONFORMING with
this contract and must change; this spec does not adapt to it.

Pinning cases (must hold forever):
  BOTH bf16, overlapping, one WRITE one READ      -> spec FORBIDS (VIEW_OVERLAP_INCOMPATIBLE)
  bf16 vs f32, overlapping, BOTH READ             -> spec ALLOWS

Re-run status (updated again). I1 COMMITTED dtype_view.py as 81bf059, sha256
23560f6f27c99b1af89cfa659ba3fa2930a839a8f51f5ab186f01347022b06c7; this supersedes the
untracked b7d29757 and its result is RETIRED. Per-case conformance against the committed sha:
  valid_full_bf16_to_f32              spec [] / ACCEPT                     ok
  valid_read_read_overlap_diff_dtype  spec [] / his ACCEPT                CONFORMS
  misaligned                          VIEW_MISALIGNED_OFFSET / misaligned  CONFORMS
  range_exceeds_parent                VIEW_RANGE_OVERFLOWS_PARENT / OOB    CONFORMS
  overlap_write_read_same_dtype       VIEW_OVERLAP_INCOMPATIBLE /
                                      his overlapping-write                CONFORMS
  extent_overflow                     VIEW_EXTENT_OVERFLOW / his ACCEPT    SCOPE GAP (expected)
  stale_after_realloc                 VIEW_STALE_AFTER_REALLOC / his ACCEPT SCOPE GAP (expected)
Pinning cases HOLD: same-dtype W x R FORBIDDEN, cross-dtype R x R ALLOWED.
I1's access field is REQUIRED with no default, so an undeclared write is impossible; his
docstring now states it carries no GENERATION and no ELEMENT-COUNT CAP, which matches this
spec's position that those two are spec/plan-level only.

STALENESS and the ELEMENT-COUNT CAP remain spec/plan-level only: his object carries
neither, so the target-side validator does not validate them; nothing may be read as target
validation of those two.

### Spec/plan-level only (not validated target-side)

STALENESS (allocation generation) and the ELEMENT-COUNT CAP live at the SPEC/PLAN level.
I1's DtypeViewSpec carries neither field, so the target-side validator does not validate
them. Nothing here may be read as "the target validates staleness/count-limit"; those two
rules are enforced only by this spec and (once wired) by P4. Stated so no claim outruns the
evidence.


## 6. Binding independence (do not conflate with the mesh-escape blocker)

I1's next blocker is that the gmem_view VALUE ESCAPES the mesh region at finalize
("value gmem_view escapes region mesh"). That is a SCOPING problem: where a value is bound
and how long it lives, NOT whether the byte-range alias is sound. alias_conflicts(views,
accesses) is therefore kept INDEPENDENT of any binding/scope: it takes views and accesses
only, and never asks where a value is defined. Do not fold region scoping into the alias
rule; they are different concerns with different evidence.
