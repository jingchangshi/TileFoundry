# P0 housekeeping: what is SIMPLIFIED, what is DROPPED

Recorded so the record stays honest. Based on the program review section B and the P0
recapture instruction.

## DROPPED (removed from the product path, kept only as history)

1. **Dual payload/enclosing-directory hashing ceremony.** The 165/164 confusion came from
   putting a manifest INSIDE an already-hashed payload. Gone: there is now exactly ONE
   domain. MANIFEST.json lives OUTSIDE payload/, and every hash in it is a payload file.
   The historical correction (archive/ARCHIVE-DOMAIN.md) stays as history, not a workflow.
2. **Scattered per-run anchors.** ANCHOR-*.json are superseded by the single immutable
   run dir + MANIFEST.json. Moved to payload-history, not consulted by the harness.
3. **Standalone tests that never merge into the harness.** Permanent-extra-obligation
   tests are dropped; the useful ones move into the canonical harness. A gate that never
   runs a required check protects nothing.
4. **Exact-English-message pins and repeated console matrices** as acceptance. The matrix
   of 13 PASS rows did not stop five false passes. Reserved for rule identity + offending
   artifact/source + real status.
5. **Custom-text success / case-template diagrams as phase acceptance.**

## SIMPLIFIED

1. **Per-run metadata** -> one immutable run dir + one small MANIFEST.json (source/dirty
   diff, build identity, environment/device, protocol, input set, command, child exits
   recorded ONCE).
2. **Replay policy** -> dependency-triggered: replay when implementation, relevant
   runtime/compiler dependency, protocol, input or generated plan changes, and once
   independently at FINAL. Documentation-only changes need provenance review, not a full
   numerical run.
3. **Repeated copies of inputs/goldens and repeated prose reports** -> one output per
   distinct protocol input; reuse content-identified data.

## KEPT

- The frozen precision protocol is unchanged and copied verbatim.
- The kernel SHA requirement c6c37fb751d0afe439006a953d5518724166a7375bc5201bfdb9acb61b93d909.
- Five pass snapshots: AutoSchedule, PrepareMultiBuffer, InsertSync, MaterializeMultiBuffer,
  LowerScheduledTIR.
