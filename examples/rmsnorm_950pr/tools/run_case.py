#!/usr/bin/env python3
"""Run one Twin case on the real 950PR: numeric check, compile snapshot, paired
benchmark against the frozen v33 baseline. One case per process.

Exit codes (propagated to the caller, NOT swallowed):
  0  numerical PASS (0 mismatches under the frozen protocol) and bitwise-stable
  1  numerical FAIL or instability
  2  ERROR (unexpected exception)
  3  NOT_RUN (torch_npu absent) -- no device touched, no result
 75  BLOCKED_ENV (device not authorised) -- refused before importing torch_npu
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, statistics, sys, traceback
from pathlib import Path

BLOCKED_ENV = 75
NOT_RUN = 3
DEVICE_AUTH_ENV = "TF_DEVICE_AUTH"
FROZEN_PROTOCOL = ("/home/s00653124/workspace/TileFoundry_Ascend950PR_AgentTeam"
                   "/artifacts/run01/p0/precision_protocol.json")


def device_authorised() -> bool:
    return os.environ.get(DEVICE_AUTH_ENV, "").strip() == "1"


def torch_npu_present() -> bool:
    try:
        return importlib.util.find_spec("torch_npu") is not None
    except (ImportError, ValueError):
        return False


def refuse_unless_authorised(device_index: int) -> None:
    if not torch_npu_present():
        print("NOT_RUN: torch_npu is not importable. No device touched; nothing measured.",
              file=sys.stderr)
        raise SystemExit(NOT_RUN)
    if not device_authorised():
        print("\n".join([
            "BLOCKED_ENV: refusing to touch device %d." % device_index,
            "  torch_npu is present, so this script would open a device node.",
            "  Authorisation was not supplied; absence is a refusal, not a default.",
            "  Set %s=1 for this invocation only, after the admin has cleared the host." % DEVICE_AUTH_ENV,
        ]), file=sys.stderr)
        raise SystemExit(BLOCKED_ENV)
    print("DEVICE AUTHORISED (%s=1): device %d. Timings are PROVISIONAL (shared card)."
          % (DEVICE_AUTH_ENV, device_index), file=sys.stderr)


def sha256_text(t: str) -> str:
    return hashlib.sha256(t.encode()).hexdigest()


def load_frozen_protocol(path: str) -> dict:
    text = Path(path).read_text()
    data = json.loads(text)
    if data.get("status") != "FROZEN":
        raise SystemExit(
            "REFUSED: protocol %s has status=%r, not FROZEN. An unfrozen protocol "
            "must not be used to pass or fail a candidate." % (path, data.get("status")))
    return data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-id", required=True)
    ap.add_argument("--shape", type=int, nargs=2, required=True)
    ap.add_argument("--rows", type=int, required=True)
    ap.add_argument("--loop-mode", required=True, choices=["dyn", "recompute"])
    ap.add_argument("--stages", type=int, required=True)
    ap.add_argument("--xversions", type=int, required=True)
    ap.add_argument("--yversions", type=int, required=True)
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--windows", type=int, default=20)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--protocol", default=FROZEN_PROTOCOL,
                    help="FROZEN precision protocol (status must be FROZEN)")
    ap.add_argument("--baseline-kernel",
                    default="/home/s00653124/workspace/TileFoundry_Ascend950PR_AgentTeam/reference/kernel_v33.py")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    refuse_unless_authorised(args.device)
    protocol = load_frozen_protocol(args.protocol)

    import torch, torch_npu, tilelang
    try:
        tilelang.disable_cache()
    except Exception:
        pass
    torch.npu.set_device(args.device)
    sys.path.insert(0, "/home/s00653124/workspace/TileFoundry/examples/rmsnorm_950pr")
    from runtime_model import RuntimeModel, check_case, paired_benchmark
    from kernels.twin_rmsnorm import TwinConfig

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cfg = TwinConfig(args.case_id, args.shape[0], args.shape[1], args.rows, args.loop_mode,
                     args.stages, args.xversions, args.yversions)
    model = RuntimeModel(cfg)
    model.ensure()
    src = model.kernel_source()
    (out / "kernel_source.txt").write_text(src)
    devname = torch.npu.get_device_name(args.device)

    try:
        import importlib.metadata as _md
        tvm_ffi_ver = _md.version("apache-tvm-ffi")
    except Exception:
        tvm_ffi_ver = None

    protocol_text = Path(args.protocol).read_text()
    device_info = {
        "device_index": args.device, "device_name": devname,
        "torch": torch.__version__,
        "torch_npu": getattr(torch_npu, "__version__", None),
        "tilelang": getattr(tilelang, "__version__", None),
        "tilelang_file": getattr(tilelang, "__file__", None),
        "tvm_ffi": tvm_ffi_ver,
        "cann": os.environ.get("ASCEND_HOME_PATH"),
        "python": sys.version.split()[0], "python_exe": sys.executable,
        "ascend_rt_visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "protocol_id": protocol.get("protocol_id"),
        "protocol_path": str(Path(args.protocol).resolve()),
        "protocol_sha256": sha256_text(protocol_text),
        "kernel_source_sha256": sha256_text(src), "cache_key": cfg.cache_key(),
    }
    (out / "device_info.json").write_text(json.dumps(device_info, indent=2))

    numeric = check_case(model, torch, args.device, protocol, save_dir=str(out))
    (out / "numeric.json").write_text(json.dumps(numeric, indent=2))
    (out / "patterns_report.json").write_text(json.dumps(numeric["patterns"], indent=2))

    spec = importlib.util.spec_from_file_location("v33_base", args.baseline_kernel)
    base = importlib.util.module_from_spec(spec); sys.modules["v33_base"] = base
    spec.loader.exec_module(base)
    base.LOOP_MODE = args.loop_mode; base._KERNELS.clear()

    def baseline_run(x, w, eps):
        return base.run(x, w, eps)

    bench = paired_benchmark(model, baseline_run, torch, args.device,
                             windows=args.windows, iters=args.iters)
    (out / "paired_benchmark.json").write_text(json.dumps(bench, indent=2))

    acc = numeric["accuracy_summary"]
    stability_ok = bool(numeric["stability"]["all_bitwise_equal_first"])
    numeric_ok = bool(numeric["numerical_passed"])
    exit_code = 0 if (numeric_ok and stability_ok) else 1

    clearly_faster = (
        bench["baseline_median"] > bench["candidate_median"]
        and bench["speedup_median"] > 1.05
        and all(b > c for b, c in zip(bench["baseline_ms"], bench["candidate_ms"]))
    )
    perf_verdict = "CLEAR_GAIN" if clearly_faster else "NO_CLEAR_GAIN"

    summary = {
        "case_id": args.case_id, "device_family": "Ascend950PR", "device_name": devname,
        "protocol_id": protocol.get("protocol_id"),
        "protocol_sha256": device_info["protocol_sha256"],
        "numerical_passed": numeric_ok, "stability_ok": stability_ok,
        "accuracy": {
            "max_abs": acc["max_abs"], "max_rel": acc["max_rel"],
            "total_mismatch": acc["total_mismatch"],
            "atol": acc["atol"], "rtol": acc["rtol"],
            "max_abs_le_atol_alone": acc["max_abs_le_atol_alone"],
            "deterministic_patterns_exact": acc["deterministic_patterns_exact"],
            "pass_rule": acc["pass_rule"],
            "per_pattern": acc["per_pattern"],
        },
        "performance": {
            "verdict": perf_verdict,
            "baseline_median_ms": bench["baseline_median"],
            "candidate_median_ms": bench["candidate_median"],
            "speedup_median": bench["speedup_median"],
            "paired_windows": bench["windows"],
            "provisional_shared_card": True,
        },
        "exit_code": exit_code,
        "kernel_source_sha256": device_info["kernel_source_sha256"],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    (out / "exit_code.txt").write_text(str(exit_code))
    print(json.dumps(summary))
    return exit_code


if __name__ == "__main__":
    try:
        rc = main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        rc = 2
    sys.stdout.flush(); sys.stderr.flush()
    # os._exit avoids a hang in NPU/tilelang interpreter teardown; it carries the
    # REAL code computed above, not a constant 0.
    os._exit(rc)
