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
