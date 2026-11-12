"""03-15/16/17: vcadd tree, vdupv broadcast, vmul, vsqrt, vdiv -- typed finalize/eval.

Each op: authored HIR chain through the public CLI analyze/finalize, reloaded with
load_authored_ir + verify_module + verify_prim_function, then evaluated against an
independent Python reference (struct-rounded f32; math.sqrt for vsqrt).

vcadd: the SIX source calls are an SSA chain 64->32->16->8->4->2 (width is an explicit
attribute); the model does NOT broadcast -- only dst[0:lanes/2] is written, and the final
broadcast is an EXPLICIT vdupv(position=0). Native tree/layout are UNSOURCED/UNVALIDATED.
A stage check proves one vcadd is a pairwise half-reduction, and the whole chain sums all
64 lanes; a dropped call or a constant-0 reducer changes the answer.

Exit codes: 0 PASS, 1 FAIL, 2 ERROR, 3 NOT_RUN.
"""
from __future__ import annotations

import argparse
import math
import os
import struct
import subprocess
import sys

import tilefoundry_ascend  # noqa: F401

if "torch_npu" in sys.modules:
    raise SystemExit("REFUSED: torch_npu is loaded; this check is CPU-only")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
FIX = os.path.join(ROOT, "targets", "ascend", "tests", "fixtures", "p3_ops", "arith.py")
LANES = 64
OPS = {"vmul": "VmulChain", "vsqrt": "VsqrtChain", "vdiv": "VdivChain", "vcadd": "VcaddChain", "vdupv": "VdupvChain"}


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--op", required=True, choices=sorted(OPS))
    args = parser.parse_args()
    entry = OPS[args.op]

    from tilefoundry.cli.source import checked_program, load_authored_ir
    from tilefoundry.evaluator import evaluate
    from tilefoundry.evaluator.registry import schedule_eval_registry
    from tilefoundry.visitor_registry.verify import verify_module, verify_prim_function
    import tilefoundry_ascend.ir.ops as ops_module

    checks = {}
    directory = os.environ.get("TMPDIR", "/tmp")
    tir = os.path.join(directory, f"p3_{args.op}_tir.py")
    r = cli("analyze", f"{FIX}:{entry}")
    print(f"child analyze exit={r.returncode}")
    checks["public_analyze_exit0"] = r.returncode == 0
    if r.returncode:
        print(r.stderr[-300:])
    r = cli("schedule", "finalize", f"{FIX}:{entry}", tir)
    print(f"child finalize exit={r.returncode}")
    checks["public_finalize_exit0"] = r.returncode == 0
    if r.returncode:
        print(r.stderr[-300:])
    if os.path.isfile(tir):
        module = load_authored_ir(f"{tir}:{entry}")
        verify_module(module)
        verify_prim_function(module.entry_function())
        text = open(tir, encoding="utf-8").read()
        checks["public_reload_verified"] = True
        checks["tir_has_op_in_mode"] = f"ascend_{args.op}" in text and "with T.SimdVF()" in text
    else:
        checks["public_reload_verified"] = False
        checks["tir_has_op_in_mode"] = False
    if args.op == "vcadd":
        checks["tir_has_six_calls"] = text.count("T.ascend_vcadd(") == 6 if os.path.isfile(tir) else False
        checks["tir_has_explicit_vdupv"] = "T.ascend_vdupv(" in text if os.path.isfile(tir) else False

    program = checked_program(f"{FIX}:{entry}", None, command=f"p3-{args.op}")
    authored = program.entry
    bases = [1.0, 4.0, 9.0, 16.0, 2.0, 0.25, 100.0, 0.0]
    xs = [fl32(bases[i % len(bases)]) for i in range(LANES)]
    ys = [fl32([2.0, -4.0, 0.5, 8.0][i % 4]) for i in range(LANES)]

    def run_chain():
        if args.op in ("vmul", "vdiv"):
            return getattr(evaluate(authored, torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)), "data", None)
        return getattr(evaluate(authored, torch.tensor(xs, dtype=torch.float32)), "data", None)

    out = run_chain()
    got = out.reshape(-1).view(torch.int32).tolist()
    if args.op == "vmul":
        expected = [f32(fl32(xs[i] * ys[i])) for i in range(LANES)]
    elif args.op == "vsqrt":
        expected = [f32(fl32(math.sqrt(xs[i]))) for i in range(LANES)]
    elif args.op == "vdiv":
        expected = [f32(fl32(xs[i] / ys[i])) if ys[i] != 0 else got[i] for i in range(LANES)]
    elif args.op == "vcadd":
        total = 0.0
        for v in xs:
            total += v
        expected = [f32(fl32(total))] * LANES
    else:  # vdupv(position=2)
        expected = [f32(xs[2])] * LANES
    checks["numeric_bits_exact_vs_independent_reference"] = got == expected
    checks["not_constant_zero"] = any(b != 0 for b in got)

    # mask is load-bearing for the predicated ops (not vcadd/vdupv which are not masked)
    if args.op in ("vmul", "vsqrt", "vdiv"):
        handler = schedule_eval_registry.lookup(getattr(ops_module, OPS[args.op].replace("Chain", "")))
        cls = getattr(ops_module, OPS[args.op].replace("Chain", ""))
        if handler is None:
            raise RuntimeError("no evaluator")
        class V:
            def __init__(self, t):
                self.data = t
        class Ctx:
            def __init__(self, a, op):
                self.args, self.op, self.result_type = a, op, None
        half = torch.zeros(LANES, dtype=torch.bool); half[: LANES // 2] = True
        keep = torch.full((LANES,), -7.0)
        a = torch.full((LANES,), 4.0)
        op_obj = cls()
        if args.op == "vmul":
            masked = handler(Ctx([V(a), V(torch.full((LANES,), 3.0)), V(half), V(keep)], op_obj)).data
        elif args.op == "vdiv":
            masked = handler(Ctx([V(a), V(torch.full((LANES,), 2.0)), V(half), V(keep)], op_obj)).data
        else:
            masked = handler(Ctx([V(a), V(half), V(keep)], op_obj)).data
        checks["mask_inactive_keeps_old_dst"] = bool(torch.equal(masked[LANES // 2 :], keep[LANES // 2 :]))

    # vcadd stage: one call with lanes=64 is a pairwise half reduction, not the full sum
    if args.op == "vcadd":
        handler = schedule_eval_registry.lookup(ops_module.Vcadd)
        class V:
            def __init__(self, t):
                self.data = t
        class Ctx:
            def __init__(self, a, op):
                self.args, self.op, self.result_type = a, op, None
        src = torch.arange(LANES, dtype=torch.float32)
        dst = torch.full((LANES,), -1.0)
        one = handler(Ctx([V(src), V(dst)], ops_module.Vcadd(lanes=64))).data
        checks["vcadd_pairwise_first"] = float(one[0]) == 0.0 + 1.0
        checks["vcadd_pairwise_last"] = float(one[31]) == 62.0 + 63.0
        checks["vcadd_leaves_upper_half"] = bool(torch.equal(one[32:], torch.full((32,), -1.0)))
        checks["vcadd_one_call_is_not_full_sum"] = float(one[0]) != sum(range(64))
        six = src
        for width in (64, 32, 16, 8, 4, 2):
            six = handler(Ctx([V(six), V(torch.full((LANES,), -1.0))], ops_module.Vcadd(lanes=width))).data
        checks["vcadd_six_calls_sum"] = float(six[0]) == float(sum(range(64)))
        missing = src
        for width in (64, 32, 16, 8, 4):
            missing = handler(Ctx([V(missing), V(torch.full((LANES,), -1.0))], ops_module.Vcadd(lanes=width))).data
        checks["vcadd_five_calls_not_sum"] = float(missing[0]) != float(sum(range(64)))

    for name, ok in checks.items():
        print(f"{'ok  ' if ok else 'FAIL'} {name}")
    print(f"first expected {[f'0x{v & 0xFFFFFFFF:08X}' for v in expected[:4]]} got {[f'0x{v & 0xFFFFFFFF:08X}' for v in got[:4]]}")
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
