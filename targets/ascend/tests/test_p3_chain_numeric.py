"""WBS 03-05: the carrier -> vcvt -> f32 chain computes real numbers.

The reference is INDEPENDENT: it reads the 16 raw BF16 bits we chose and builds the IEEE
f32 word by a Python struct shift (bf16 -> f32 is a left shift of the 16-bit pattern into
the high half of a 32-bit word). It never calls torch's bfloat16 cast, never calls the
evaluator under test, and never reuses an output of the program. Bit patterns are
compared, not approximate floats, so signed zero and denormals count.

The input is deliberately discriminating: +-0, +-1, +-2, 0.5, 0.333, 3.14, the smallest
denormal, the largest finite bf16, negative denormals and a variety of exponents. A
constant-zero converter, an identity/ignore-source converter, a converter that ignores
its mask, or a wrong bit shift all disagree with the reference.

Public path, each child's exit recorded: CLI analyze, CLI schedule finalize, reload of
the FINALIZED artifact with verify_module/verify_prim_function, then evaluation of the
authored HIR against the independent reference.

Exit codes: 0 PASS, 1 FAIL, 2 ERROR, 3 NOT_RUN (no torch).
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
CHAIN = os.path.join(ROOT, "targets", "ascend", "tests", "fixtures", "p3_carrier", "bf16_vcvt_chain.py")
ENTRY = "Bf16VcvtChain"
LANES = 64

#: 16 raw bf16 bit patterns, repeated with a rotation so the 64 lanes are not uniform.
PATTERNS = [
    0x0000,  # +0
    0x8000,  # -0 (bit pattern must survive)
    0x3F80,  # 1.0
    0xBF80,  # -1.0
    0x4000,  # 2.0
    0xC000,  # -2.0
    0x3F00,  # 0.5
    0xBF00,  # -0.5
    0x0001,  # smallest positive denormal, 2^-133
    0x8001,  # smallest negative denormal
    0x3EAA,  # ~0.333
    0x4049,  # ~3.14
    0x7F7F,  # largest finite bf16 (~3.389e38)
    0xFF7F,  # -largest finite bf16
    0x3C00,  # 2^-7-ish normal
    0xBC00,  # negative normal
]


def input_bits(shift=0):
    return [PATTERNS[(i + shift) % len(PATTERNS)] for i in range(LANES)]


def reference_f32_bits(bits):
    """Independent BF16 -> IEEE f32, as a SIGNED 32-bit word (no torch cast)."""
    word = ((bits & 0xFFFF) << 16) & 0xFFFFFFFF
    return struct.unpack("<i", struct.pack("<I", word))[0]


def reference_float(bits):
    word = ((bits & 0xFFFF) << 16) & 0xFFFFFFFF
    return struct.unpack("<f", struct.pack("<I", word))[0]


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
    from tilefoundry_ascend.ir.ops import Vcvt

    checks = {}
    directory = os.environ.get("TMPDIR", "/tmp")
    tir_path = os.path.join(directory, "p3_numeric_chain_tir.py")

    # --- public path, one child each -----------------------------------------
    r_analyze = cli("analyze", f"{CHAIN}:{ENTRY}")
    print(f"child analyze exit={r_analyze.returncode}")
    checks["public_analyze_exit0"] = r_analyze.returncode == 0
    if not checks["public_analyze_exit0"]:
        print(r_analyze.stderr[-400:])
    r_finalize = cli("schedule", "finalize", f"{CHAIN}:{ENTRY}", tir_path)
    print(f"child finalize exit={r_finalize.returncode}")
    checks["public_finalize_exit0"] = r_finalize.returncode == 0
    if not checks["public_finalize_exit0"]:
        print(r_finalize.stderr[-400:])
    if os.path.isfile(tir_path):
        reloaded = load_authored_ir(f"{tir_path}:{ENTRY}")
        verify_module(reloaded)
        verify_prim_function(reloaded.entry_function())
        checks["public_reload_verified"] = True
        text = open(tir_path, encoding="utf-8").read()
        checks["public_tir_carries_chain"] = all(
            needle in text
            for needle in ('"carrier"', 'dist="UNPK_B16"', "ascend_pset16", "ascend_vcvt", "ascend_vsts")
        )
    else:
        checks["public_reload_verified"] = False
        checks["public_tir_carries_chain"] = False

    # --- numeric evaluation --------------------------------------------------
    program = checked_program(f"{CHAIN}:{ENTRY}", None, command="p3-numeric")
    authored = program.entry
    bits = input_bits()
    x = torch.tensor([(b - 0x10000) if b >= 0x8000 else b for b in bits], dtype=torch.int16).view(torch.bfloat16)
    produced = evaluate(authored, x)
    data = getattr(produced, "data", produced)
    expected_bits = [reference_f32_bits(b) for b in bits]
    got_bits = data.reshape(-1).view(torch.int32).tolist()

    checks["shape"] = tuple(data.shape) == (LANES,)
    checks["dtype_f32"] = data.dtype == torch.float32
    checks["bits_exact_vs_independent_reference"] = got_bits == expected_bits
    checks["nonzero_output"] = any(b != 0 for b in got_bits)
    checks["not_identity_of_input"] = got_bits != [(b & 0xFFFF) for b in bits]
    # A second, different input must produce a different output.
    bits2 = input_bits(shift=5)
    x2 = torch.tensor([(b - 0x10000) if b >= 0x8000 else b for b in bits2], dtype=torch.int16).view(torch.bfloat16)
    got2 = getattr(evaluate(authored, x2), "data", None).reshape(-1).view(torch.int32).tolist()
    checks["input_dependent"] = got2 != got_bits
    checks["second_input_also_exact"] = got2 == [reference_f32_bits(b) for b in bits2]
    # A genuinely wrong reference (the bf16 bit PATTERN read as a small integer, i.e. no
    # shift into the high half) must disagree with the real conversion. This is a real
    # comparison: it can equal got only if the program returned those small integers.
    checks["wrong_reference_no_shift_differs"] = got_bits != [
        struct.unpack("<i", struct.pack("<f", float(b)))[0] for b in bits
    ]

    # --- mask is load-bearing (drive the instruction evaluator directly) -----
    handler = schedule_eval_registry.lookup(Vcvt)
    if handler is None:
        raise RuntimeError("no schedule evaluator registered for Vcvt")

    class V:
        def __init__(self, tensor):
            self.data = tensor

    class Ctx:
        def __init__(self, args, op):
            self.args, self.op, self.result_type = args, op, None

    src = x
    half = torch.zeros(LANES, dtype=torch.bool)
    half[: LANES // 2] = True
    keep = torch.full((LANES,), -7.0, dtype=torch.float32)
    masked = handler(Ctx([V(src), V(half), V(keep)], Vcvt(direction="forward", part=0, elem_width=16))).data
    checks["mask_active_half_converted"] = masked[: LANES // 2].view(torch.int32).tolist() == expected_bits[: LANES // 2]
    checks["mask_inactive_half_kept"] = bool(torch.equal(masked[LANES // 2 :], keep[LANES // 2 :]))

    for name, ok in checks.items():
        print(f"{'ok  ' if ok else 'FAIL'} {name}")
    print("input raw bf16 bits (first 8):", [f"0x{b:04X}" for b in bits[:8]])
    print("expected f32 bits (first 8):", [f"0x{b & 0xFFFFFFFF:08X}" for b in expected_bits[:8]])
    print("produced f32 bits (first 8):", [f"0x{b & 0xFFFFFFFF:08X}" for b in got_bits[:8]])
    if not all(checks.values()):
        print(f"{sum(not ok for ok in checks.values())}/{len(checks)} checks FAILED")
        return 1
    print(f"{len(checks)}/{len(checks)} checks passed; reference floats (first 4):",
          [reference_float(b) for b in bits[:4]])
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as error:  # noqa: BLE001
        print(f"ERROR {type(error).__name__}: {error}")
        raise SystemExit(2)
