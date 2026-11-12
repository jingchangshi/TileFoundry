"""Run the restricted native TF product primitive on device 0 under external flock."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
import torch_npu

from tilefoundry.analysis import analyze
from tilefoundry.cli.source import checked_program
from tilefoundry.schedule import finalize
from tilefoundry_ascend.codegen.native_aiv import NativeAIVBinding
from tilefoundry_ascend.codegen.x_runtime import emit_for_codegen, compile_tilelang


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    if not args.compile_only:
        torch.npu.set_device(0)
    source = root / "tilefoundry_ascend/attnres/native_square.py"
    p = checked_program(str(source), None, command="native-aiv-device")
    m = finalize(analyze(p.module, p.entry, analysis=("memory",)), authored=p.authored)
    a = emit_for_codegen(m.metadata["ascend.p3_plan"], NativeAIVBinding(m))
    policy = {"tl.enable_auto_schedule": False}
    kernel = compile_tilelang(a, a.abi, {"source_path": str(output / "native_square_tl.py")}, policy)
    if args.compile_only:
        cce = kernel.get_kernel_source()
        (output / "native_square.cce").write_text(cce)
        print("COMPILE_PASS", len(cce), flush=True)
        return
    x = (torch.arange(256, dtype=torch.float32) - 113) / 16
    z = (torch.arange(256, dtype=torch.float32).remainder(17) - 7) / 8
    rows = []
    for name, right in (("product", z), ("square", x)):
        xn, zn = x.npu(), right.npu()
        out = torch.empty_like(xn)
        kernel(xn, zn, out)
        torch.npu.synchronize()
        actual = out.cpu()
        expected = x * right
        ok = bool(torch.equal(actual, expected))
        rows.append({"case": name, "passed": ok,
                     "max_abs_error": (actual - expected).abs().max().item()})
        print(name, "PASS" if ok else "FAIL", rows[-1], flush=True)
    cce = kernel.get_kernel_source()
    (output / "native_square.cce").write_text(cce)
    report = {"status": "PASS" if all(r["passed"] for r in rows) else "FAIL",
              "scope": "AR RMS product micro-case only; not full AR or G61/G70",
              "cases": rows, "pass_configs": policy,
              "author_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "tir_digest": a.tir_digest, "plan_digest": a.plan_digest,
              "tl_sha256": a.source_sha256,
              "cce_sha256": hashlib.sha256(cce.encode()).hexdigest(),
              "torch": torch.__version__}
    (output / "result.json").write_text(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
