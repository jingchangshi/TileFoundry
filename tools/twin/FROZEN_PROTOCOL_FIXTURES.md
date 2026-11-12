# Frozen protocol & fixture inventory for the T (Twin/reference) module

Status: protocol values are FROZEN by `artifacts/run02/HANDOFF.md` §5.9 (2026-11-05).
The file `config/numerical_protocol.proposed.json` still says `"status": "PROPOSED",
"approved_by": null`; both facts are recorded here — this doc does not upgrade the
file's own status and does not modify anything under `reference/`.

## 1. Frozen values

| Item | Value | Source (read-only) |
|---|---|---|
| reference | `cpu_float64_rmsnorm_then_bfloat16` | config/numerical_protocol.proposed.json |
| eps | **1e-6** (also as actual f32 bits: 0x358637bd) | HANDOFF §5.9; kernel_v33.py run(eps=1e-6) |
| atol | **0.02** | HANDOFF §5.9 |
| rtol | **0.01** | HANDOFF §5.9 |
| predicate | `|got-ref| <= atol + rtol*|ref|` for EVERY element; mismatch_count == 0 | HANDOFF §5.9 |
| patterns (order) | zero, ones, random(seed 0), random(1), random(2), alternating | run02 independent_reference/reference_report.json |
| seeds | 0,1,2 | config/numerical_protocol.proposed.json |
| repetitions / stability | 4 per input / 64 per case | same |
| cases | prefill 8192x4096 ×{dyn,recompute}; decode 64x4096 ×{dyn,recompute} | config/cases.json |

**Wing rule clarified (never to be flattened):** prefill has `max_abs = 0.03125 > atol`
and passes only through the rtol wing. A report must apply the elementwise rule;
`max_abs <= atol` is NOT the frozen test.

## 2. Determinism rules

- **zero / ones / alternating** use no RNG: device vs reference must be **bit-exact**.
- **random(seed 0/1/2)** inputs are produced by a torch CPU generator with
  `manual_seed(seed)`; binding device numbers to them is CONDITIONAL until an
  env-of-record run. The T module never regenerates an "actual" input from a seed —
  it reads recorded BF16 bytes (see twin-io/1).

## 3. Input construction (producer semantics only)

`make_input(shape, pattern, seed)` (reference.py, producer-only) is byte-for-byte the
frozen sampler:

| pattern | x | w |
|---|---|---|
| zero | zeros(shape, f32) | ones(width) |
| ones | ones(shape, f32) | linspace(0.25, 1.75, width) |
| random | randn(shape, generator(g)) | randn(width, generator(g)) |
| alternating | sign(row_i)*scale(j)*±1, scale=[0.01,1,10,100][j%4] | linspace(-1,1,width) |

x and w are then cast to bf16. Consumers load the RECORDED bf16 bytes and the eps
f32 bits; "pattern"/"seed" are notes, not a source.

## 4. Read-only provenance (sha256)

| file | sha256 |
|---|---|
| reference/kernel_v33.py | c6c37fb751d0afe439006a953d5518724166a7375bc5201bfdb9acb61b93d909 |
| reference/APPROVED_DESIGN.md | 2a02bd76afe6f12ac8a534e077cd2aea7cf91c8a94480da22103f6d370dcb8c5 |
| reference/v33_static_facts.json | 990f402c62e6eb322f47177fd307867b3e9b0e1fb43ab4d62a04e123a2583ed4 |
| config/numerical_protocol.proposed.json | 8c52ee193fd24924b05870c351a6b928bab4535e4f109ffc22236a17ceb2432b |
| config/cases.json | 04e0e3b8a4164382f96b73377484d662e1612ed6cf81120c5bda085fe22ded8f |
| run02 independent_reference/REPORT.md | 46cf6cd3f5b8a20a53aaee9f8cd885e014e923209990a00457e6794337669da9 |
| run02 independent_reference/reference_report.json | e1d5d25ba180c934bd3058212be23a464b9a97aa43d58c96e8aee19e4f87aa7f |

## 5. Recorded fixtures in this module

Bundles live in `tools/twin/fixtures/bundles.json` + `tools/twin/fixtures/actual/<case>/*.bin`.
Each tensor records shape/dtype/byte order/strides/nbytes/sha256; the eps record carries
the f32 **bits**. Fixtures use a small single-row width (D=256) for T3 CPU turn-around;
production cases remain the frozen 4096/8192 shapes above.

| fixture | kind | shape | notes |
|---|---|---|---|
| T-FIX-VADD-001 | f32 vadd-one | [64] | deterministic arange(64)+0.5; output f32, no BF16 |
| T-FIX-RMS-ROW-001..006 | single-row RMSNorm | [1,256] | patterns zero/ones/random(0,1,2)/alternating; eps bits frozen |

## 6. BF16 quantization boundary (design v1 §4)

1. Inputs are already bf16 when captured; the reference widens them to fp64 exactly.
2. TF algorithm side: bf16→f32 exact widening; f32 arithmetic in program order; an
   **explicit** f32→bf16 cast at each conversion point (never an identity relabel).
3. Independent reference: fp64 formula, then **ONE** fp64→bf16 RNE rounding
   (`bf16_bits_from_f64_once`). A f64→f32→bf16 path is double rounding and is
   rejected with `REFERENCE_QUANTIZATION_MISMATCH`.
4. Special values (NaN/Inf/overflow/subnormal) are explicit; the tolerance is never
   used to hide a non-finite value.
