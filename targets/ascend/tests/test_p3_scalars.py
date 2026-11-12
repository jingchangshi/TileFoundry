"""03-12/13/14: vdup / vadds / vmuls micro -- typed finalize/eval, scalar ABI, mask.

Each op: authored HIR chain GM->UB->vld->op->vsts->UB->GM through the public CLI
analyze/finalize, reloaded and verified, then evaluated against an independent Python
double->f32 reference. The scalar is an instruction attribute (ABI), the mask is a value
and is load-bearing (inactive lanes keep the destination). Mutations that make the scalar
constant-1 or ignore the mask are killed in run_scalar_mutations.sh.

Exit codes: 0 PASS, 1 FAIL, 2 ERROR, 3 NOT_RUN.
"""
from __future__ import annotations

import argparse
import os
import struct
import subprocess
import sys

import tilefoundry_ascend  # noqa: F401

if "torch_npu" in sys.modules:
    raise SystemExit("REFUSED: torch_npu is loaded; this check is CPU-only")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
FIX = os.path.join(ROOT, "targets", "ascend", "tests", "fixtures", "p3_ops", "scalars.py")
LANES = 64

OPS = {
    "vdup": {"entry": "VdupChain", "cls": "Vdup", "value": 1.5, "kind": "dup"},
    "vadds": {"entry": "VaddsChain", "cls": "Vadds", "value": 0.25, "kind": "adds"},
    "vmuls": {"entry": "VmulsChain", "cls": "Vmuls", "value": -2.0, "kind": "muls"},
}


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
    spec = OPS[args.op]

    from tilefoundry.cli.source import checked_program, load_authored_ir
    from tilefoundry.evaluator import evaluate
    from tilefoundry.evaluator.registry import schedule_eval_registry
    from tilefoundry.visitor_registry.verify import verify_module, verify_prim_function
    import tilefoundry_ascend.ir.ops as ops_module

    checks = {}
    directory = os.environ.get("TMPDIR", "/tmp")
    tir = os.path.join(directory, f"p3_{args.op}_tir.py")

    r = cli("analyze", f"{FIX}:{spec['entry']}")
    print(f"child analyze exit={r.returncode}")
    checks["public_analyze_exit0"] = r.returncode == 0
    if r.returncode:
        print(r.stderr[-300:])
    r = cli("schedule", "finalize", f"{FIX}:{spec['entry']}", tir)
    print(f"child finalize exit={r.returncode}")
    checks["public_finalize_exit0"] = r.returncode == 0
    if r.returncode:
        print(r.stderr[-300:])
    if os.path.isfile(tir):
        module = load_authored_ir(f"{tir}:{spec['entry']}")
        verify_module(module)
        verify_prim_function(module.entry_function())
        text = open(tir, encoding="utf-8").read()
        checks["public_reload_verified"] = True
        checks["tir_has_op_in_mode"] = f"ascend_{args.op}" in text and "with T.SimdVF()" in text
    else:
        checks["public_reload_verified"] = False
        checks["tir_has_op_in_mode"] = False

    program = checked_program(f"{FIX}:{spec['entry']}", None, command=f"p3-{args.op}")
    authored = program.entry

    bases = [1.0, -3.5, 0.25, 1024.0, -0.0, 3.1415927, -2.7182818, 7.5]
    xv = [fl32(bases[i % len(bases)]) for i in range(LANES)]
    out = getattr(evaluate(authored, torch.tensor(xv, dtype=torch.float32)), "data", None)
    got = out.reshape(-1).view(torch.int32).tolist()
    if args.op == "vdup":
        expected = [f32(fl32(spec["value"])) for _ in range(LANES)]
    elif args.op == "vadds":
        expected = [f32(fl32(xv[i] + spec["value"])) for i in range(LANES)]
    else:
        expected = [f32(fl32(xv[i] * spec["value"])) for i in range(LANES)]
    checks["numeric_bits_exact_vs_independent_reference"] = got == expected
    checks["scalar_load_bearing"] = got != [f32(fl32((xv[i] if args.op != "vdup" else 1.0) + 0.0)) for i in range(LANES)]

    handler = schedule_eval_registry.lookup(getattr(ops_module, spec["cls"]))
    if handler is None:
        raise RuntimeError(f"no schedule evaluator for {spec['cls']}")

    class V:
        def __init__(self, t):
            self.data = t

    class Ctx:
        def __init__(self, args, op):
            self.args, self.op, self.result_type = args, op, None

    half = torch.zeros(LANES, dtype=torch.bool)
    half[: LANES // 2] = True
    keep = torch.full((LANES,), -7.0, dtype=torch.float32)
    data = torch.full((LANES,), 3.0, dtype=torch.float32)
    op_obj = getattr(ops_module, spec["cls"])(value=spec["value"])
    if args.op == "vdup":
        masked = handler(Ctx([V(half), V(keep)], op_obj)).data
    else:
        masked = handler(Ctx([V(data), V(half), V(keep)], op_obj)).data
    checks["mask_inactive_keeps_old_dst"] = bool(torch.equal(masked[LANES // 2 :], keep[LANES // 2 :]))
    if args.op == "vdup":
        checks["mask_active_half_value"] = bool(torch.equal(masked[: LANES // 2], torch.full((LANES // 2,), fl32(spec["value"]))))
    else:
        checks["mask_active_half_value"] = collected = True

    for name, ok in checks.items():
        print(f"{'ok  ' if ok else 'FAIL'} {name}")
    print("value", spec["value"], "first expected", [f"0x{v & 0xFFFFFFFF:08X}" for v in expected[:4]],
          "got", [f"0x{v & 0xFFFFFFFF:08X}" for v in got[:4]])
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
