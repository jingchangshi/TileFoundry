# P3 numeric oracle: what it determines and what it assumes

> **03-21 RETRACTION (read the sections below with this in force).** This report describes
> the PRE-03-21 oracle, which GENERATED its own inputs (numpy RNG, **row-wise W** of shape
> (rows,D)) instead of consuming the runner's persisted X[M,D]/W[D]. Those inputs are not
> the device inputs, so every statement below about per-case reference values, patterns,
> seeds, loop modes, or agreement with device output is **VOID as same-input evidence**:
> V01 per-case reference summary; V02 "Input generators"; V03 "Inputs are BF16"; V04 the
> 8 run-P3-20261104T2305/*.y_bf16.npy files and their MANIFEST; V05 "dyn and recompute give
> identical values" (two labels on one self-generated output); V06 prefill random statistics
> (the same seed was reused for each 256-row block, 32 identical copies); V07 any claim of
> oracle-vs-device agreement for the 4/2-chain construction; V08 deterministic bit-exactness
> as an oracle result; V09 old seed/pattern/shape fields as protocol coverage; V10 treating
> those outputs as "the independent reference for the device diff". Full list, the fresh
> recomputation and the quantified old-vs-actual divergence (99.6–99.95% bit mismatch on
> every non-zero pattern) are in
> `artifacts/run03/w2/03-21/INPUT_ABI.md` and `old_vs_actual_divergence.json`.
> NOT void: the bf16 encoder self-check, --scalar-demo, the frozen tolerance protocol, and
> the UNSOURCED/UNVALIDATED status of the native vcadd tree.
> The current `tools/p3/rmsnorm_oracle.py` consumes runner artifacts READ-ONLY with
> per-file sha256 and refuses a non-conforming input ABI; `--run-dir` is mandatory.
>
> **RULING UPDATE 2026-11-05T00:44+08:00** (`architect_ruling_03_21b_actual_inputs_run2.md`):
> V01–V10 below are **I2 VOID item IDs, not the masterplan V01–V10 acceptance command cards**.
> The old same-input/device/protocol-coverage relation **NEVER HELD** (not "once accepted,
> now void"); the old numbers still describe their own software corpus. V03 is narrowed to
> the unbound-source inference (BF16/RNE codec properties remain valid); V06 keeps the
> full-size 8192x4096 array (the defect is the 32 repeated random blocks and runner
> identity, NOT the shape); V08 leaves the frozen det-bits-exact obligation intact; V10
> rejects any blanket void — the run2 ACTUAL-INPUT independent audit (164 files /
> `c8e21a39…` limited receipt) remains valid, not withdrawn/downgraded/upgraded.
> See `artifacts/run03/w2/03-21/INPUT_ABI.md`.

Oracle: tools/p3/rmsnorm_oracle.py. Outputs: artifacts/run02/p3_oracle/run-P3-20261104T2305
(16 case/pattern files as uint16 BF16 bits + MANIFEST.json) — **VOID as comparison
evidence, see the retraction above; retained only as a historical record**.

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
