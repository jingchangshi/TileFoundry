# D0 end-to-end carrier: the loop does not survive parsing

`pipe_vadd.py` is the program supplied by Lead, which was measured as
`checked ok -> analyze ok -> finalize ok`. It reproduces here, and the measurement is
correct. What it does NOT do is exercise D0, and this file records why so nobody
cites it as pipeline evidence.

## Measured (venv-explorer python, PYTHONPATH=src:targets/ascend)

    checked:  OK
    analyze:  OK
    finalize: OK
    HIR LoopRegion count : 0      <-- the loop is GONE before lowering
    TIR For count        : 0
    authored manifest loops: 0
    as_script has tf.pipelined( : False

## What that rules out, and what it points at

- NOT a printer gap. `tir_printer.py` prints `tf.pipelined(...)` whenever
  `For.pipeline` is not None, and `python_printer.py` does the same for the HIR loop.
  There is no loop to print, so nothing is missing from the printers.
- NOT lowering dropping the field. `ConvertHIRToTIR` never sees a pipelined loop:
  the declaration is already gone at the end of parsing.
- The cause is dead-code elimination at parse time. In this program the value the loop
  produces (`s = tf.schedule((back, y[...]))`) is never consumed: the function returns
  `y`, an empty gmem Var allocated before the loop, and writing into it through the
  loop does not keep the loop alive. With no consumer the loop is removed, and a
  finalize that succeeds on a program with no loop is a VACUOUS success for D0.

That is why this file cannot serve as D0 completion evidence as written: the "all
zero" result is real but it is about a program with no pipeline in it.

## Also measured: two statements in the supplied program do not type-check here

`va`/`vb`/`mask` were allocated as `smem`, and f32 `T.ascend_vld` requires an element
register destination:

    VerifyError: ScheduleOp: T.ascend_vld ... reads bf16 loads state dist=UNPK_B16 and
    land in the carrier class; f32 loads state no dist and land in an element register

Allocating them as `rmem` (as the existing p3_ops fixtures do) resolves it. The file
here has that change applied, so what is measured above is the DCE finding alone and
not a residency error.

## What a real D0 carrier needs

A loop whose produced value is consumed by the function result, e.g. the loop leaving
a UB buffer that a post-loop `T.ascend_ub_to_gm()` stores into the returned tensor.
Until such a program parses with the loop intact, the D0 evidence is:
  * tests/test_d0_pipeline_controls.py -- 22 cases, function level, with mutation
    matrix (0 not caught), and
  * the BEFORE/AFTER pass registration, ordering and manifest survival measured on a
    real module that contains no pipeline (so the passes fire and are silent).
No claim is made that a pipelined loop has been lowered end to end.