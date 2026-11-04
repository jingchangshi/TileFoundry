# P3 numeric oracle: what it determines and what it assumes

Oracle: tools/p3/rmsnorm_oracle.py. Outputs: artifacts/run02/p3_oracle/run-P3-20261104T2305
(16 case/pattern files as uint16 BF16 bits + MANIFEST.json).

## ORACLE-DETERMINED (exact, from the frozen protocol and the pinned reference)

- Shapes and protocol: prefill 8192x4096, decode 64x4096; eps 1e-6; patterns
  zero/ones/random/alternating; seed 0. Protocol SHA a1676239...a78d0.
- Inputs are BF16 (single RNE), decoded exactly for the FP32 domain.
- ALL accumulation is FP32 after the BF16->FP32 conversion (no per-step BF16 rounding).
- Accumulator count PER CASE, from the pinned v33 (SHA c6c37fb7... == P0 pin):
  prefill rows=4 -> FOUR chains, balanced merge ((a0+a1)+(a2+a3));
  decode rows=1 -> rows<=2 fallback -> TWO chains, merge (a0+a1).
- Output y = FP32(x*scale*w) with a SINGLE RNE to BF16.
- BF16 encoder itself is validated against the exact-rational reference (5,000 random
  f64 patterns, 0 mismatches; NaN/inf/zero/subnormal/overflow covered).

## ASSUMED (not determined; must not be presented as v33 semantics)

1. LANE-REDUCTION (vcadd) ORDER IS UNRESOLVED. The pairwise-halving code is a PLACEHOLDER.
   The Python docstring is not proof: simd_inst.h 688-702 uses asc_reduce_sum with PAT_VL1
   and vcpadd 681-684 uses asc_pair_reduce_sum. Pin the exact tree with a selected-build
   microtest before any claim.
2. vmula is modelled as two FP32 roundings (mul then add), not a fused single-rounding FMA.
3. mean as b*(1/d) in FP32, scale as 1/sqrt(mean+eps) in FP32 with correctly rounded
   sqrt/div (an approximate rsqrt would differ).
4. Input generators: zero -> x=0,w=1; ones -> 1,1; alternating -> +/-1, w=1;
   random -> N(0,1) for both x and w, numpy default_rng(seed=0).
5. Loop mode (dyn/recompute) does not change the operator value; the same oracle output is
   recorded for both names.

## Per-case reference summary (seed 0)

prefill (chains=4): zero max|x|=0 sum(x^2)=0 max|y|=0 sum(y)=0 scale=1000;
  ones max|x|=1 sum(x^2)=3.35544e7 max|y|=1 sum(y)=3.35544e7 scale=1;
  alternating max|x|=1 sum(x^2)=3.35544e7 max|y|=1 sum(y)=0 scale=1;
  random max|x|=5 sum(x^2)=3.36091e7 max|y|=13.5625 sum(y)=-30864.4 scale=[0.971568,1.02539].
decode (chains=2): zero as above; ones sum(x^2)=262144;
  alternating max|y|=1 sum(y)=0; random max|x|=4.71875 sum(x^2)=262741 max|y|=10.5625
  sum(y)=-275.039 scale=[0.977443,1.0216].
dyn and recompute give identical values (as expected).

## Ceiling

The oracle is a faithful FP32-accumulation model EXCEPT where the lane-reduction order is
unresolved. Until a selected-build microtest pins that order, agreement with the oracle
cannot distinguish the correct tree from a placeholder that happens to sum similarly.


## Scalar semantics (visible in every diff)

The oracle multiplies by a FLOAT32-ROUNDED scalar (v33's scale lives in FP32). This is a
deliberate, stated choice, not an exact-scalar model. The distinction is observable:
  0x3f81 (bf16) * 0x1.02fa0be82fa0cp+0
    -> 0x3f82 with the f32-rounded scalar
    -> 0x3f83 with the exact scalar
Verified by "python tools/p3/rmsnorm_oracle.py --out X --scalar-demo". MANIFEST.json records
scalar_semantics accordingly. Any future diff must state which scalar semantics it used.

## Credit scope

Software oracle credit only. Native vcadd table and FP32-vs-BF16 semantics are NOT
accepted. check_reduction_order_observable must not be cited (tautology; see p2 REPORT 10).
