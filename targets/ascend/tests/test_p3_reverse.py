"""03-18: reverse conversion f32 -> bf16 CARRIER -> vsts(PK_B32) -> bf16 UB -> GM.

Checks: typed public analyze/finalize/reload/verify; RNE f32->bf16 rounding against an
INDEPENDENT integer reference (round-half-to-even, including ties, carry, subnormal, inf,
signed zero); and the named refusals for a wrong pack, a narrow/wrong-dtype write and a
wrong part.

The reverse path packs two bf16 halves per 32-bit word (PK_B32); the CPU model keeps the
bf16 VALUE stream and does not model the physical repack, which is UNSOURCED. Native
packing/rounding is UNVALIDATED; no native credit.

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
FIX = os.path.join(ROOT, "targets", "ascend", "tests", "fixtures", "p3_ops", "reverse_chain.py")
ENTRY = "ReverseVcvtChain"
LANES = 64


def rne_bf16(u32):
    """Independent round-to-nearest-even f32->bf16, on the raw 32-bit pattern."""
    u32 &= 0xFFFFFFFF
    if (u32 & 0x7F800000) == 0x7F800000 and (u32 & 0x007FFFFF):
        return ((u32 >> 16) | 0x0040) & 0xFFFF  # quiet NaN
    lsb = (u32 >> 16) & 1
    return ((u32 + 0x7FFF + lsb) & 0xFFFFFFFF) >> 16


PATTERNS = [
    0x3F800000,              # 1.0 exact
    0x3F800000 | 0x00007FFF, # just below a tie, rounds down
    0x3F800000 | 0x00008000, # tie, lsb 0 -> stays (even)
    0x3F800001 | 0x00008000, # tie, lsb 1 -> rounds up
    0x3F800000 | 0x00008001, # just above a tie, rounds up
    0x3FFFFFFF,              # mantissa 0x7FFFFF: rounds carry into exponent
    0x00000001,              # smallest positive subnormal -> 0
    0x00008000,              # subnormal tie
    0x7F7FFFFF,              # max finite -> rounds to +inf
    0x80000000,              # -0.0
    0xBF800000,              # -1.0
    0xBF800001 | 0x00008000, # negative tie, lsb 1
    0xC0000000,              # -2.0
    0x40490FDB,              # ~3.1415927
    0xFF7FFFFF,              # -max finite
    0x3EAAAAAB,              # ~0.33333334
]


def ref_bits():
    return [PATTERNS[i % len(PATTERNS)] for i in range(LANES)]


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
    from tilefoundry.ir.hir.schedule import ScheduleRefusal
    from tilefoundry.ir.types import DType, Mesh, StorageKind, Topology, make_tensor_type
    from tilefoundry.visitor_registry.instruction import check_call
    from tilefoundry.visitor_registry.verify import verify_module, verify_prim_function
    from tilefoundry_ascend.ir.ops import Vcvt, Vsts

    checks = {}
    directory = os.environ.get("TMPDIR", "/tmp")
    tir = os.path.join(directory, "p3_reverse_tir.py")
    r = cli("analyze", f"{FIX}:{ENTRY}")
    print(f"child analyze exit={r.returncode}")
    checks["public_analyze_exit0"] = r.returncode == 0
    r = cli("schedule", "finalize", f"{FIX}:{ENTRY}", tir)
    print(f"child finalize exit={r.returncode}")
    checks["public_finalize_exit0"] = r.returncode == 0
    if not r.returncode and os.path.isfile(tir):
        module = load_authored_ir(f"{tir}:{ENTRY}")
        verify_module(module)
        verify_prim_function(module.entry_function())
        text = open(tir, encoding="utf-8").read()
        checks["public_reload_verified"] = True
        checks["tir_reverse_signature"] = 'direction="reverse"' in text and 'pack="PK_B32"' in text
        checks["tir_bf16_ub_to_gm"] = 'dtype="bf16", storage="smem"' in text and "ascend_ub_to_gm" in text
    else:
        checks["public_reload_verified"] = False
        checks["tir_reverse_signature"] = False
        checks["tir_bf16_ub_to_gm"] = False

    bits = ref_bits()
    x = torch.tensor([(b - 0x100000000) if b >= 0x80000000 else b for b in bits], dtype=torch.int32).view(torch.float32)
    program = checked_program(f"{FIX}:{ENTRY}", None, command="p3-reverse")
    out = getattr(evaluate(program.entry, x), "data", None)
    got = [b & 0xFFFF for b in out.reshape(-1).view(torch.int16).tolist()]
    expected = [rne_bf16(b) for b in bits]
    checks["dtype_bf16"] = out.dtype == torch.bfloat16
    checks["rne_bits_exact_vs_independent_reference"] = got == expected
    checks["has_discriminating_ties"] = len({e for e in expected}) > 4
    # A truncation (round toward zero) reference must differ somewhere: proves RNE is real.
    trunc = [(b >> 16) & 0xFFFF for b in bits]
    checks["truncation_would_differ"] = trunc != expected

    mesh = Mesh((Topology("aiv", 1),), (1,), ("core",))

    def ty(d, s):
        return make_tensor_type((LANES,), dtype=DType.from_name(d), storage=s)

    def refused(op, args_, needle):
        try:
            check_call(op, args_, mesh, frame=None)
        except ScheduleRefusal as error:
            text = str(error)
            if needle not in text:
                raise AssertionError(f"refused by the wrong layer (want {needle!r}): {text[:200]}")
            return
        raise AssertionError(f"ACCEPTED a case that must be refused ({needle!r})")

    checks["reverse_good_admitted"] = True
    try:
        check_call(Vcvt(direction="reverse", part=0, elem_width=16), (ty("f32", StorageKind.RMEM), ty("bool", StorageKind.RMEM), ty("bf16", StorageKind.CARRIER)), mesh, frame=None)
    except ScheduleRefusal as error:
        checks["reverse_good_admitted"] = False
        print("       reverse good refused:", str(error)[:160])
    refused(Vcvt(direction="reverse", part=1, elem_width=16), (ty("f32", StorageKind.RMEM), ty("bool", StorageKind.RMEM), ty("bf16", StorageKind.CARRIER)), "ascend_vcvt:")
    checks["wrong_part_refused"] = True
    refused(Vcvt(direction="reverse", part=0, elem_width=16), (ty("bf16", StorageKind.CARRIER), ty("bool", StorageKind.RMEM), ty("f32", StorageKind.RMEM)), "ascend_vcvt:")
    checks["reverse_with_carrier_src_refused"] = True
    refused(Vsts(), (ty("bf16", StorageKind.CARRIER), ty("bf16", StorageKind.SMEM)), "PK_B32")
    checks["carrier_store_without_pack_refused"] = True
    refused(Vsts(pack="PK_B32"), (ty("f32", StorageKind.RMEM), ty("f32", StorageKind.SMEM)), "no pack")
    checks["f32_store_with_pack_refused"] = True
    refused(Vsts(pack="PK_B32"), (ty("bf16", StorageKind.CARRIER), ty("f32", StorageKind.SMEM)), "src.dtype")
    checks["narrow_wrong_dtype_store_refused"] = True

    for name, ok in checks.items():
        print(f"{'ok  ' if ok else 'FAIL'} {name}")
    print("input f32 bits[:8]", [f"0x{b:08X}" for b in bits[:8]])
    print("expected bf16  [:8]", [f"0x{b:04X}" for b in expected[:8]])
    print("produced bf16  [:8]", [f"0x{b:04X}" for b in got[:8]])
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