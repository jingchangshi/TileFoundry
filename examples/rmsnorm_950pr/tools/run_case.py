#!/usr/bin/env python3
"""Run one Twin case on the real 950PR: numeric check, compile snapshot, paired
benchmark against the frozen v33 baseline. One case per process."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, statistics, sys, traceback
from pathlib import Path


def sha256_text(t):
    return hashlib.sha256(t.encode()).hexdigest()


def main():
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
    ap.add_argument("--baseline-kernel", default="/home/s00653124/workspace/TileFoundry_Ascend950PR_AgentTeam/reference/kernel_v33.py")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

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

    device_info = {"device_index": args.device, "device_name": devname,
                   "torch": torch.__version__, "torch_npu": getattr(torch_npu, "__version__", None),
                   "tilelang": getattr(tilelang, "__version__", None),
                   "tilelang_file": getattr(tilelang, "__file__", None), "cann": __import__("os").environ.get("ASCEND_HOME_PATH"),
                   "kernel_source_sha256": sha256_text(src), "cache_key": cfg.cache_key()}
    (out / "device_info.json").write_text(json.dumps(device_info, indent=2))

    numeric = check_case(model, torch, args.device)
    (out / "numeric.json").write_text(json.dumps(numeric, indent=2))

    # baseline v33 module (fresh import; loop mode set; cache cleared)
    spec = importlib.util.spec_from_file_location("v33_base", args.baseline_kernel)
    base = importlib.util.module_from_spec(spec); sys.modules["v33_base"] = base; spec.loader.exec_module(base)
    base.LOOP_MODE = args.loop_mode; base._KERNELS.clear()

    def baseline_run(x, w, eps):
        return base.run(x, w, eps)

    bench = paired_benchmark(model, baseline_run, torch, args.device, windows=args.windows, iters=args.iters)
    (out / "paired_benchmark.json").write_text(json.dumps(bench, indent=2))

    summary = {"case_id": args.case_id, "device_family": "Ascend950PR", "device_name": devname,
               "numerical_passed": numeric["numerical_passed"],
               "stability_ok": numeric["stability"]["all_bitwise_equal_first"],
               "baseline_median_ms": bench["baseline_median"], "candidate_median_ms": bench["candidate_median"],
               "speedup_median": bench["speedup_median"],
               "kernel_source_sha256": device_info["kernel_source_sha256"]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))
    sys.stdout.flush(); os._exit(0)


if __name__ == "__main__":
    main()