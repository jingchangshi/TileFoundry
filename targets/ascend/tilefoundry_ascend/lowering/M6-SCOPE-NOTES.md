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


## 6. Architecture review: five bypasses, and the ruling

The Architect replayed d976f3d and found five bypasses in the trace model. RULING: M6
completion is BLOCKED and P5 is NOT granted. The 13 rules plus positives count as
trace-model progress only. Priority is the normal formal TF path (materialize must emit
real TF nodes through ConvertHIRToTIR / public finalize, layout must consume the actual
plan output, verify must run on that). No more auxiliary framework.

### Fixed here (the two trivially-wrong verifier bugs)
- BYPASS 5: "drain(); hidden_gm_write()" was accepted because the drain matcher was a
  PREFIX regex. Now the drain must be the whole statement with at most a trailing
  comment: "[M6-UNKNOWN-STATEMENT] unknown statement: drain(); hidden_gm_write() ...".
- BYPASS 4: an unknown sync channel was ACCEPTED. Now
  "[M6-UNKNOWN-CHANNEL] unknown sync channel: <name>".

Both are asserted as matrix rows (DRAIN_TRAILING_GARBAGE, UNKNOWN_CHANNEL); matrix is
15 rows, all rc=0.

### Recorded as KNOWN HOLES -- deliberately NOT closed with more regex
- HOLE 1: removing all simdvf/mte3/Y-protocol, leaving only the X handshake, verifies
  True. INDEPENDENTLY REPRODUCED here: 18 events, ok=True.
- HOLE 2: x_free moved to after wait_xready but before the first VF. REPORTED by the
  Architect; my naive line-reorder was itself rejected, so I did NOT reproduce his exact
  mutation and make no claim about it either way. Treat as open.
- HOLE 3: load slot 0->2 at t=4 with consumers unchanged verifies True. INDEPENDENTLY
  REPRODUCED here: 41 events, ok=True, no errors.

### Also recorded (not fixed, part of why P5 is not granted)
- [M6-PIPE-MISMATCH] exists only as a helper: pipeline.compile_case never calls
  verify_pipes, so the NORMAL materialize+verify path does not enforce it.
- layout_check builds a fresh placement and ignores the plan, so it does not check the
  addresses actually placed.
