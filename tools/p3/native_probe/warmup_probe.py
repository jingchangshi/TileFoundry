#!/usr/bin/env python3
"""06-06/§5 copy-only warmup: pure GM->UB->GM copy, NO vcadd. 30 s bound.

Purpose: window responsiveness preflight. If this does not return within 30 s the device
runtime is unhealthy and NO probe should run. Records the same phase timestamps and a
launch watchdog as the real probes. CPU --plan does not touch a device.

Not executed by I2; requires a Lead-approved window.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from device_guard import DeviceWindow, GuardError, verify_build  # noqa: E402
from probe_base import DEFAULT_TL, observed_bits, to_f32_tensor  # noqa: E402


def build_kernel(T):
    @T.prim_func
    def main(IN: T.Tensor((64,), "float32"), OUT: T.Tensor((64,), "float32")):
        with T.Kernel(1) as core:
            buf = T.alloc_shared((64,), "float32")
            T.copy(IN[:], buf[:])
            T.copy(buf[:], OUT[:])
    return main


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--launch-watchdog", type=int, default=30)
    ap.add_argument("--tilelang-root", default=str(DEFAULT_TL))
    args = ap.parse_args()
    if not args.execute:
        print(json.dumps({"probe": "copy_only_warmup", "status": "DESIGN; --execute required",
                          "bound_s": args.timeout, "launch_watchdog_s": args.launch_watchdog,
                          "note": "no device access in --plan"}, indent=2))
        return 0
    try:
        build = verify_build(Path(args.tilelang_root))
    except GuardError as exc:
        print("REFUSED", exc, file=sys.stderr)
        return 2
    import tilelang
    import tilelang.ascend.language as T
    import torch
    import torch_npu
    torch.npu.set_device(0)
    try:
        tilelang.disable_cache()
    except Exception:  # noqa: BLE001
        pass
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    phases, t0 = {}, time.monotonic()

    def mark(name):
        phases[name] = {"wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "elapsed_s": round(time.monotonic() - t0, 3)}
        (outdir / "phases.json").write_text(json.dumps(phases, indent=2))

    def watchdog_fire():
        mark("launch_watchdog_fired")
        (outdir / "WATCHDOG_FIRED.txt").write_text("copy warmup watchdog fired\n")
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(124)

    kernel = build_kernel(T)
    with DeviceWindow(args.out, args.timeout) as window:
        window.report.update({"build": build, "probe": "copy_only_warmup",
                              "launch_watchdog_s": args.launch_watchdog})
        mark("start"); mark("compile_start")
        compiled = tilelang.compile(kernel, target="ascend", out_idx=[1])
        mark("compile_done")
        src = compiled.get_kernel_source()
        (outdir / "kernel_source.txt").write_text(src)
        window.report["kernel_source_sha256"] = hashlib.sha256(src.encode()).hexdigest()
        bits = ["0x%08X" % (0x3F800000 + i) for i in range(64)]
        x = to_f32_tensor(torch, [int(b, 16) for b in bits], "npu:0")
        mark("tensors_ready")
        wd = threading.Timer(args.launch_watchdog, watchdog_fire); wd.daemon = True; wd.start()
        try:
            mark("launch_start")
            with torch.no_grad():
                y = compiled(x)
            mark("launch_returned")
            torch.npu.synchronize(); mark("sync_done")
            obs = observed_bits(torch, y); mark("readback_done")
        finally:
            wd.cancel()
        ok = [o.upper() for o in obs] == [b.upper() for b in bits]
        (outdir / "warmup.json").write_text(json.dumps(
            {"phases": phases, "copy_ok": ok, "input_bits": bits, "observed_bits": obs}, indent=2))
        window.report["phases"] = phases
        window.report["copy_ok"] = ok
        mark("json_written")
        print(json.dumps({"copy_only_warmup": True, "copy_ok": ok, "phases": phases}, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
