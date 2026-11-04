# M6/P5 scope correction (binding)

This note corrects the claim attached to the M6/P5 work in commit d976f3d. It is the
authoritative scope; if any other message, comment, or summary states more, this wins.

## 1. Trace-model only -- NOT formal P5 TF TIR

The negative matrix (examples/rmsnorm_950pr/tools/negative_matrix.py) drives the verifier
over the CUSTOM TEXT emitted by lowering/materialize.py: a "# Resolved TF TIR" header, a
renderable def/for skeleton, and lines such as "mte2 x_load slot=.. t=..",
"simdvf main_compute ...", "wait_flag ...", "set_flag ...", "drain()". That text is built
by hand in the materializer. It is NOT produced by the public finalize path as formal TF
nodes.

Therefore the credit earned by the 13-rule matrix is FINITE TRACE-MODEL CREDIT only. It
is NOT evidence of P5 formal TIR and must not be cited as P5 formal TIR.

## 2. The layout rule does not check placed addresses

lowering/layout_check.py builds a NEW deterministic placement from buffer sizes,
alignment and version counts (largest-first, sequential within a space). It IGNORES the
plan. It therefore does NOT check the addresses that will actually be placed, and cannot
detect a hazard specific to the real placement. It is a structural feasibility check only.

## 3. What the work does legitimately assert

- The per-slot state machine and the 8 tagged rules fire on the trace model:
  [M6-EMPTY-TIR], [M6-UNKNOWN-STATEMENT], [M6-MISSING-DRAIN], [M6-DRAIN-NOT-LAST],
  [M6-FLAG-MISMATCH], [M6-MISSING-X-FREE-WAIT], [M6-OUTSTANDING-GENERATION],
  [M6-PIPE-MISMATCH], plus pre-existing mutation negatives and [M6-POST-LAYOUT-HAZARD]
  on a synthetic placement.
- The copy-path materializer emits a drain and per-buffer free-waits with generation /
  version counters -- in the trace model.
- Existing checks were not weakened on the trace model.

Nothing here asserts instruction-level or placement-level correctness of real emitted TF.

## 4. Real P5 fix

Emitting real TF nodes through the normal finalize path (rather than custom text) is a
separate, unscoped change. It is not attempted here.

## 5. Pending external review

The Architect is replaying the matrix and baseline (CPU, exact archive, ffi 0.1.15.post2)
and will issue counterexamples (no compute/stores; x_free after ready before VF; wrong load
generation on the same slot). This scope note stands regardless of those outcomes.
