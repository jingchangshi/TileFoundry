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
2. [RETRACTED -- see section 8; the text below is kept for the record] ACCUMULATOR COUNT -- was PROVISIONALLY ruled by the Lead
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


## 8. RETRACTION of the provisional accumulator ruling (do not build on item 2 above)

Retracted: the claim "TWO is the baseline of record" was WRONG. It mistook the rows<=2
fallback path for the main path. Verified against the pinned reference
reference/kernel_v33.py, whose SHA was re-checked as
c6c37fb751d0afe439006a953d5518724166a7375bc5201bfdb9acb61b93d909 == the P0 pin.

Corrected record (per-case, from the pinned source):
  * prefill 8192x4096: _rows_per_core(8192) == 4 (8192 % 4 == 0 and 8192//4 == 2048 >= 56).
    The rows==4 path (kernel_v33.py lines 127-140) declares FOUR fp32 chains
    acc0_0..acc0_3, one vmula chain per chunk lane group (chunk index % 4), and merges them
    BALANCED as ((acc0_0 + acc0_1) + (acc0_2 + acc0_3)). stages=4, yver=2.
  * decode 64x4096: _rows_per_core(64) == 1 (64//4=16 < 56, 64//2=32 < 56, 64//1=64 >= 56),
    so the rows<=2 fallback is used: TWO fp32 chains (acc0, acc1), merge (acc0 + acc1).
    stages=2, yver=1.

Two corrections that follow:
  (a) The accumulator count is PER-CASE, part of the algorithm / scheduled-HIR numerical
      contract -- NOT a universal primitive ISA field and NOT one global constant.
  (b) ALL accumulation is FP32 AFTER the BF16->FP32 conversion (vld dist="UNPK_B16" then
      vcvt to "float32"; vmula are float32). A reducer that rounds each step to BF16 is a
      TOY and is NOT v33 accumulation semantics. Any reference or test that does per-step
      BF16 rounding must be labelled a toy, not a contract.

Consequence for the P3 oracle: it models v33 (FP32 accumulation, 4 chains for prefill /
2 for decode, balanced merge), not the earlier global-2 assumption.


## 9. vcadd reduction order is UNSOURCED and UNRESOLVED

The earlier "6 vcadd" description is a PYTHON DOCSTRING reading, not proof. Against the
current TileLang C++ it is UNSOURCED: simd_inst.h 688-702 uses asc_reduce_sum with
PAT_VL1, and vcpadd 681-684 uses asc_pair_reduce_sum. Those are the actual primitives the
selected build uses; the exact lane reduction tree they implement must be established from
that C++ / an exact selected-build microtest, not inferred from a Python comment.

Therefore: the reduction order is UNRESOLVED pending an exact selected-build microtest.
Any "pairwise halving" statement in tools/p3 is a PLACEHOLDER model, not a claim.


## 10. check_reduction_order_observable is a TAUTOLOGY - do not cite it

The shipped check_reduction_order_observable is a tautology: replacing the pairwise
reducer with a constant 0 still passes it. It is NOT evidence of order observability and
must not be cited as such until I1 fixes it and adds the constant-reducer mutant as a
permanent negative. The vcadd order remains UNRESOLVED (section 9).

## 11. Credit scope (from the Lead/architect)

Software-exact quantizer results are accepted as the strongest positive evidence (65,280
finite codepoints; 32,639 adjacent intervals with midpoint +/- gap/2^40 and ties; overflow
controls; 8,192 add/mul predicate checks; 64 independent pairwise math cases). BUT the
native vcadd table and FP32-vs-BF16 semantics are NOT accepted. This is SOFTWARE ORACLE
CREDIT ONLY.
