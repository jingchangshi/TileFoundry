"""03-11: ascend_vmula micro -- typed finalize/eval, loop carry, partial mask, FMA model.

acc is READ|WRITE: a chain of vmula calls accumulates, and an inactive lane keeps the
value it carried in. The model is UNFUSED (multiply rounds, then add rounds); whether the
native issue fuses them is ASSUMED-UNKNOWN / UNSOURCED and no native credit is claimed.
The reference is independent Python double arithmetic rounded to f32 with struct; it never
calls the evaluator under test.

Exit codes: 0 PASS, 1 FAIL, 2 ERROR, 3 NOT_RUN.
"""
from __future__ import annotations

import os
import struct
import subprocess
import sys

import tilefoundry_ascend  # noqa: F401

if "torch_npu" in sys.modules:
    raise SystemExit("REFUSED: torch_npu is loaded; this check is CPU-only")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
FIX = os.path.join(ROOT, "targets", "ascend", "tests", "fixtures", "p3_ops", "vmula_micro.py")
ENTRY = "VmulaMicro"
LANES = 64


def fl32(x):
    return struct.unpack("<f", struct.pack("<f", float(x)))[0]


def f32(x):
    return struct.unpack("<i", struct.pack("<f", float(x)))[0]


def cli(*args):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([os.path.join(ROOT, "src"), os.path.join(ROOT, "targets", "ascend")])
    env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return subprocess.run([sys.executable, "-m", "tilefoundry.cli", *args], cwd=ROOT, env=env, capture_output=True, text=True)


def main() -> int:
    try:
        import torch
    except Exception as error:  # noqa: BLE001
        print(f"NOT_RUN torch is not importable: {error!r}")
        return 3

    from tilefoundry.cli.source import checked_program, load_authored_ir
    from tilefoundry.evaluator import evaluate
    from tilefoundry.evaluator.registry import schedule_eval_registry
    from tilefoundry.visitor_registry.verify import verify_module, verify_prim_function
    from tilefoundry_ascend.ir.ops import Vmula

    checks = {}
    directory = os.environ.get("TMPDIR", "/tmp")
    tir = os.path.join(directory, "p3_vmula_tir.py")

    r = cli("analyze", f"{FIX}:{ENTRY}")
    print(f"child analyze exit={r.returncode}")
    checks["public_analyze_exit0"] = r.returncode == 0
    if r.returncode:
        print(r.stderr[-300:])
    r = cli("schedule", "finalize", f"{FIX}:{ENTRY}", tir)
    print(f"child finalize exit={r.returncode}")
    checks["public_finalize_exit0"] = r.returncode == 0
    if r.returncode:
        print(r.stderr[-300:])
    if os.path.isfile(tir):
        module = load_authored_ir(f"{tir}:{ENTRY}")
        verify_module(module)
        verify_prim_function(module.entry_function())
        text = open(tir, encoding="utf-8").read()
        checks["public_reload_verified"] = True
        checks["tir_has_vmula_in_mode"] = "ascend_vmula" in text and "with T.SimdVF()" in text
        checks["tir_acc_is_read_write_order"] = "T.ascend_vmula(va, vb, mask, acc)" in text
    else:
        checks["public_reload_verified"] = False
        checks["tir_has_vmula_in_mode"] = False
        checks["tir_acc_is_read_write_order"] = False

    program = checked_program(f"{FIX}:{ENTRY}", None, command="p3-vmula")
    authored = program.entry

    # --- single issue: acc starts at 0, out = a*b ---------------------------
    xs = [1.0, -3.5, 0.25, 1024.0, -0.0, 3.1415927, -2.7182818, 7.5]
    ys = [2.0, 0.5, -8.0, 0.001953125, 5.0, -1.0, 4.0, -6.25]
    # Round the inputs to f32 FIRST: the program sees f32 values, so the reference must
    # multiply those same values, not the wider Python literals that produced them.
    xv = [fl32(xs[i % len(xs)]) for i in range(LANES)]
    yv = [fl32(ys[(i * 3) % len(ys)]) for i in range(LANES)]
    out = getattr(evaluate(authored, torch.tensor(xv, dtype=torch.float32), torch.tensor(yv, dtype=torch.float32)), "data", None)
    got = out.reshape(-1).view(torch.int32).tolist()
    # The program is acc + a*b with acc = +0, so the reference adds that +0 too:
    # IEEE says (+0) + (-0) = +0, which is exactly the difference from a bare product.
    expected = [f32(fl32(0.0 + fl32(xv[i] * yv[i]))) for i in range(LANES)]
    checks["single_issue_bits_exact"] = got == expected
    checks["single_issue_nonzero"] = any(b != 0 for b in got)

    # --- loop carry and partial mask by driving the evaluator twice ---------
    handler = schedule_eval_registry.lookup(Vmula)
    if handler is None:
        raise RuntimeError("no schedule evaluator for Vmula")

    class V:
        def __init__(self, t):
            self.data = t

    class Ctx:
        def __init__(self, args):
            self.args, self.op, self.result_type = args, Vmula(), None

    lane = lambda values: torch.tensor(values, dtype=torch.float32)  # noqa: E731
    a1 = lane([2.0] * LANES)
    b1 = lane([3.0] * LANES)
    acc0 = lane([10.0] * LANES)
    full = torch.ones(LANES, dtype=torch.bool)
    acc1 = handler(Ctx([V(a1), V(b1), V(full), V(acc0)])).data
    checks["carry_first_step"] = bool(torch.equal(acc1, lane([16.0] * LANES)))
    a2 = lane([-1.0] * LANES)
    b2 = lane([0.5] * LANES)
    acc2 = handler(Ctx([V(a2), V(b2), V(full), V(acc1)])).data
    checks["carry_second_step_reads_old_acc"] = bool(torch.equal(acc2, lane([15.5] * LANES)))
    checks["carry_would_differ_without_old_acc"] = not bool(torch.equal(acc2, lane([-0.5] * LANES)))

    half = torch.zeros(LANES, dtype=torch.bool)
    half[: LANES // 2] = True
    keep = lane([-7.0] * LANES)
    masked = handler(Ctx([V(lane([2.0] * LANES)), V(lane([5.0] * LANES)), V(half), V(keep)])).data
    checks["mask_active_half_acc"] = bool(torch.equal(masked[: LANES // 2], lane([3.0] * (LANES // 2))))
    checks["mask_inactive_keeps_old_acc"] = bool(torch.equal(masked[LANES // 2 :], keep[LANES // 2 :]))

    # --- FMA model is explicit: find a triple where fused != unfused -------
    found = None
    for acc_v in (1.0, 2.0 ** 24, -1.0, 3.0):
        for a_v in (1.0 + 2.0 ** -12, 1.0 + 2.0 ** -23, 1.0000001192092896):
            for b_v in (1.0 + 2.0 ** -12, 1.0 - 2.0 ** -12, 1.0000001192092896):
                unfused = fl32(acc_v + fl32(a_v * b_v))
                fused = fl32(acc_v + a_v * b_v)
                if unfused != fused:
                    found = (acc_v, a_v, b_v, unfused, fused)
                    break
            if found:
                break
        if found:
            break
    checks["fma_discriminator_found"] = found is not None
    if found:
        acc_v, a_v, b_v, unfused, fused = found
        model = handler(Ctx([V(lane([a_v] * LANES)), V(lane([b_v] * LANES)), V(full), V(lane([acc_v] * LANES))])).data
        model_bit = model.view(torch.int32).tolist()[0]
        checks["model_is_unfused"] = model_bit == f32(unfused)
        checks["model_differs_from_fused"] = model_bit != f32(fused)
        print(f"       FMA discriminator acc={acc_v!r} a={a_v!r} b={b_v!r}: unfused=0x{f32(unfused) & 0xFFFFFFFF:08X} fused=0x{f32(fused) & 0xFFFFFFFF:08X} model=0x{model_bit & 0xFFFFFFFF:08X}")

    for name, ok in checks.items():
        print(f"{'ok  ' if ok else 'FAIL'} {name}")
    if not all(checks.values()):
        print(f"{sum(not v for v in checks.values())}/{len(checks)} checks FAILED")
        return 1
    print(f"{len(checks)}/{len(checks)} checks passed")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as error:  # noqa: BLE001
        print(f"ERROR {type(error).__name__}: {error}")
        raise SystemExit(2)
