# tools/twin — T (Twin/reference) module, CPU half

Status: **T WIP, UNCREDITED**. This is T0–T4 (design-independent CPU work) implementing
the Lead-approved `twin-io/1` schema from
`artifacts/run02/review/architect_t_twin_reference_module_design_v1.md` (SHA 4ecbbf22…).

T5 (Plan replay) is blocked_by R; T6 (real device) is blocked_by ENV+X/H. **No device is
touched here and no compiler/TIR module is imported.**

## Files

| file | role |
|---|---|
| `twin_io.py` | twin-io/1 schema, `capture_actual_inputs` / `load_actual_inputs`, `OutputBundle`, `compare_outputs` → `ReplayReport`, `TwinError` |
| `reference.py` | the three computation roles: TF-algorithm f32 + explicit BF16 cast; independent FP64 → ONE BF16 rounding; frozen sampler (producer only) |
| `fixtures.py` | materialises the recorded actual BF16 byte bundles and the T0–T4 evidence |
| `fixtures/` | `bundles.json` + `actual/<case>/*.bin` — the authoritative recorded inputs |
| `test_twin_io.py` | positives + decisive bad controls (also runs without pytest via `__main__`) |
| `FROZEN_PROTOCOL_FIXTURES.md` | frozen atol/rtol/eps, determinism rules, input patterns, source hashes |
| `INDEPENDENCE.md` | what "independent" structurally forbids |

## Run (CPU env; never let torch autoload the NPU backend)

```bash
# Point CPU_PY at an interpreter that has numpy + torch and does NOT autoload the
# NPU backend (this tree never imports the device stack). It does not need pytest.
CPU_PY=${CPU_PY:?set CPU_PY to the CPU interpreter (numpy+torch, no pytest)}
cd tools/twin
TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$CPU_PY" fixtures.py \
    --evidence-dir ../../artifacts/run03/w40_twin        # adjust to the pack path
TORCH_DEVICE_BACKEND_AUTOLOAD=0 "$CPU_PY" test_twin_io.py  # or: python -m pytest tools/twin
```

## Positive controls (T0–T4)

- `T-FIX-VADD-001` f32 vadd-one 64-lane — PASS, 64/64 elements, mismatch 0.
- `T-FIX-RMS-ROW-001..006` single-row RMSNorm (D=256) for zero/ones/random(0,1,2)/
  alternating — all PASS, mismatch 0; the independent reference is FP64 → one BF16
  rounding.

## Decisive bad controls (each must fail with a named code)

| control | expected code |
|---|---|
| flip one BM lane in the output | `NUMERIC_MISMATCH` (mismatch_count == 1, coord reported) |
| corrupt the recorded input sha256 | `INPUT_DIGEST_MISMATCH` |
| change eps f32 bits | `EPS_BITS_MISMATCH` |
| reference produced by f64→f32→bf16 double rounding | `REFERENCE_QUANTIZATION_MISMATCH` |
| missing recorded bytes (would need seed replay) | `INPUT_MISSING`/`INPUT_DIGEST_MISMATCH` |
| shape mismatch / NaN output | `OUTPUT_ABI_MISMATCH` / `NONFINITE_OUTPUT` |

## Write scope

Only `tools/twin/**` (this tree) and `artifacts/run03/w40_twin/**` (evidence). Nothing
under `src/` or `targets/ascend/` is touched, and F/P/L/R interfaces are unchanged.
