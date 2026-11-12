# Twin (T) independence boundary

The T module answers one question: *what should the numbers be, computed without the
compiler or its evaluator?* This file states what that forbids.

## Three roles, three code paths

| Role | Code | May it read compiler output? | May it use a candidate result? |
|---|---|---|---|
| 1 Actual input | `twin_io.capture/load_actual_inputs` | Only the recorded input bytes the runner materialised | No |
| 2 TF algorithm side | `reference.tf_algo_*` | It models what the TF evaluator DOES; it never consumes TF output as its own answer | No |
| 3 Independent reference | `reference.reference_* / compute_reference` | **No** | **No** |

## Explicit prohibitions (structurally tested)

- **No compiler/TIR import.** reference.py imports only numpy (+torch as the frozen
  input sampler). `test_twin_io.py::test_reference_module_is_independent_of_compiler_and_tir`
  parses the module AST and rejects `tilefoundry`, `tilelang`, `tilefoundry_ascend`,
  `evalref`, and the tokens `identity_evaluator`, `as_script`, `@prim_func`.
- **No TIR text, no `as_script`/type-string derivation.** A reference computed from
  compiler IR is not independent; it is the compiler grading itself.
- **No shared identity evaluator.** T must not call the TF evaluator, its reference
  helper, or a candidate's reduction/cast helpers. It may share read-only protocol
  parsing and the recorded actual-input bytes (design v1 §1).
- **No candidate golden as reference.** `compute_reference` derives its result from
  the loaded input bytes and the frozen formula only. The bad control
  `test_bad_control_double_rounding_reference_is_detected` shows a reference that
  takes a different rounding path is rejected, not tolerated.
- **No seed replay as actual input.** `load_actual_inputs` requires real recorded
  bytes with matching sha256/nbytes/dtype; a missing file is
  `INPUT_MISSING/INPUT_DIGEST_MISMATCH`, never "regenerate from seed"
  (`test_twin_io_never_regenerates_an_input_from_seed`).
- **No device.** This module starts no kernel and imports no device stack; T6 is out
  of scope until the ENV/X/H combination lands.

## Allowed sharing

- The frozen protocol constants and the input-sampler *semantics* (copied, with the
  source hashes recorded in `FROZEN_PROTOCOL_FIXTURES.md`).
- The recorded actual-input bytes and the twin-io/1 schema.
- Read-only serialization/hash utilities (sha256, endian packing).

## Error family (design v1 §6)

`INPUT_DIGEST_MISMATCH`, `INPUT_ABI_MISMATCH`, `INPUT_MISSING`, `EPS_BITS_MISMATCH`,
`OUTPUT_ABI_MISMATCH`, `NONFINITE_OUTPUT`, `NUMERIC_MISMATCH`,
`REFERENCE_QUANTIZATION_MISMATCH`, `TWIN_UNSUPPORTED_OP`, `TWIN_UNSUPPORTED_PLAN`.
Each is a structured object `(code, phase, source_id, expected, actual)`, not prose.
