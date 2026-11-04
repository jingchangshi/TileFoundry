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
