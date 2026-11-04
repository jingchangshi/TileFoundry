# P2 reference side: what the spec determines, and what it does not

Independent reference: `bf16_reference.py` (exact rationals, no torch/numpy/TF).
Microtests: `microtests.py` (18 boundary cases + 4 hand-derived self-checks).
Fuzz: `fuzz.py` (boundary-biased 20k patterns, first-divergence report, roundtrip drift).

## DETERMINED (the reference is authoritative for these)

- bf16 encoding/decoding: 1 sign, 8 exponent (bias 127), 7 mantissa. Same exponent range
  as f32, so f32 and bf16 share normal/subnormal boundaries.
- Round-to-nearest-even (ties to even), including the tie that produces the odd/even
  decision at the 7-bit mantissa.
- Subnormal boundary 2^-133 (min bf16 subnormal), half-way point 2^-134 (ties to zero).
- Overflow boundary: values >= (bf16 max + ulp/2) = 2^127*(255/128) + 2^119 round to inf;
  the tie itself rounds to inf (2^128 is not representable).
- Zero, +0/-0 sign preservation, infinity, and the NaN CLASS.
- f32 elementwise multiply and add: each rounded to binary32 (RNE) per IEEE-754.

**REFERENCE_REDUCTION_ORDER is a documented CHOICE, not a fact.** It states what this
reference models (2 accumulators, round-robin, f32 after every mul/add, left-to-right
vcadd chain). No spec I can cite fixes that order; it is an assumption recorded so it can
be challenged, and tooling/p2/cross_check_impl.py demonstrates below that a different
accumulation precision yields different bits on the same inputs.

## NOT DETERMINED by the spec (the honest ceiling)

1. **vcadd lane-reduction order.** The design says six vcadd calls; the order in which
   lanes are combined (pairwise tree vs sequential vs a hardware-specific hierarchy) is
   not fixed by any spec I can cite. `REFERENCE_REDUCTION_ORDER` assumes a left-to-right
   chain; a different tree changes the rounding of the sum for the same inputs.
2. **Number of accumulators.** The design mentions four/two. The kernel_v33 rows=4 path
   shows TWO (acc0, acc1). The reference defaults to 2 but is parameterised; which is
   canonical must come from the actual sequence, not the spec.
3. **Accumulate precision / fusion.** The reference assumes binary32 after every multiply
   and every add (no fusion). If the device fuses vmula (single rounding) or keeps wider
   intermediates, results differ.
4. **rsqrt path.** The kernel uses vdiv(1, vsqrt(...)); any approximate reciprocal-sqrt
   instruction has a precision not pinned by the spec. The reference assumes correctly
   rounded sqrt and division.
5. **Denormal handling on device (FTZ/DAZ).** The reference is IEEE (no flush). If the
   device flushes subnormals to zero, the subnormal cases diverge; nothing in the spec
   fixes flush behaviour for this target.
6. **bf16 pack rounding of y (PK_B32).** Assumed RNE. A truncating pack would differ and
   is not excluded by the instruction description.
7. **NaN payload.** torch's converter returns 0xFFFF while the reference returns the
   canonical 0x7FC0. Only the NaN CLASS is meaningful; the payload is not architecturally
   fixed, and the microtest compares classes, not payload bits.

## Consequence

Numeric agreement on random data can pin the DETERMINED parts only. It cannot prove the
reduction order, accumulator count, fusion, rsqrt precision or FTZ behaviour; those must be
pinned against the actual device instruction sequence (or declared explicitly as modelling
assumptions). Where they are not pinned, a passing random corpus is evidence of consistency
with one legal reduction, not proof of the hardware order.

## Status
microtests: ALL PASS (18 boundary cases + 4 self-checks).
fuzz: 20,000 boundary-biased patterns, NO DIVERGENCE; roundtrip drift 0.
(One real reference bug was found and fixed during this work: -0.0 lost its sign.)


## Two-independent-paths cross-check (I1's target-side bf16.py vs this reference)

tools/p2/cross_check_impl.py, read-only on the target-side module:
  encode: 20,000 exact-f32 inputs (boundary exponents + random words) -> AGREE
  decode: all 65,536 bf16 patterns compared as exact rationals -> AGREE
  reduction: his ORDER_LINEAR vs an independent bf16-step fold -> AGREE
  accumulator precision: bf16-step vs f32-step DOES differ on a crafted 64-lane vector
    (all terms in [1,2)): bf16-step 0x42C5 vs f32-step 0x4318. Same inputs, different
    bits. This is the CHOICE above made observable; it is NOT a defect on either side.

## P2 CEILING (nothing downstream may silently assume these)

1. vcadd lane-reduction order (tree vs sequential vs hardware hierarchy).
2. ACCUMULATOR COUNT -- explicit contract field, PROVISIONALLY ruled by the Lead
   (architect confirmation pending; this is NOT settled by me):
   - BASELINE OF RECORD FOR NUMERIC CLAIMS = TWO. Numeric claims are made against the
     shipped, executable reference/kernel_v33.py, whose rows=4 path uses TWO accumulators
     (acc0, acc1).
   - The design's FOUR is recorded as a DESIGN VARIANT / TARGET, not the contract: there is
     no executable four-accumulator form, and numerics must not be pinned to something
     unbuilt.
   - REQUIRED CONSEQUENCE IN CODE: the ISA must declare the accumulator count EXPLICITLY
     as a contract field (never implicit), and the reduction order must be stated as
     DEPENDING on that field, so a future four-accumulator variant cannot silently change
     numerics. bf16_reference.py carries ACCUMULATOR_COUNT_BASELINE = 2 as that explicit
     field, with the same provisional note.
   Status: PROVISIONAL Lead ruling; architect asked to confirm. Treat TWO as the claim
   baseline, FOUR as an unbuilt target, until that confirmation lands.
3. Fusion / wider intermediates (reference rounds f32 after every mul and add).
4. rsqrt precision (vdiv(1, vsqrt) vs an approximate rsqrt instruction).
5. Denormal FTZ/DAZ on device (reference is IEEE, no flush).
6. bf16 pack rounding of y (PK_B32) -- assumed RNE.
7. NaN payload (torch 0xFFFF vs canonical 0x7FC0); only the class is fixed.

Consequence: numeric agreement pins only the determined parts; it is NOT proof of the
hardware reduction order, and the accumulator count is an open design question.
