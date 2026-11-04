#!/usr/bin/env python3
"""Shared runner for the three 03-19 single-instruction native probes.

DESIGN ONLY (UNSOURCED / UNVALIDATED). Imported by vcadd_probe.py, vcvt_probe.py,
vmula_probe.py. THIS ROUND no probe is executed on a device: the scripts are delivered
for the approved V09 window of WBS 03-20. Each kernel has THREE input tensors and one
output (out_idx=3); exactly ONE arithmetic/conversion instruction is under test, the
surrounding vld/vsts/copy are TRANSPORT.

An observation is recorded, never "passed": a CPU-model match is not a native contract.
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
EXPECTED = HERE / "expected_bits.json"
DEFAULT_TL = Path("/home/s00653124/workspace/tilelang")
ZEROS = ["0x00000000"] * 64


def load_case(name: str) -> dict:
    d = json.loads(EXPECTED.read_text())
    for c in d["cases"]:
        if c["name"] == name:
            return c
    raise SystemExit("unknown case %r" % name)


def parse_bits(seq):
    return [int(x, 16) for x in seq]


def to_f32_tensor(torch, bits32, device):
    import numpy as np
    a = np.asarray(bits32, dtype=np.uint32).view(np.float32)
    return torch.from_numpy(a.copy()).to(device)


def to_bf16_tensor(torch, bits16, device):
    import numpy as np
    a = np.asarray(bits16, dtype=np.uint16)
    return torch.from_numpy(a.copy()).to(torch.int16).view(torch.bfloat16).to(device)


def observed_bits(torch, t):
    import numpy as np
    return ["0x%08X" % int(b) for b in np.ascontiguousarray(
        t.detach().to("cpu").to(torch.float32).numpy()).view(np.uint32)]


def make_tensors(torch, case, kind, dev):
    if kind == "vcvt":
        t0 = to_bf16_tensor(torch, parse_bits(case["input_bf16_bits"]), dev)
        t1 = to_f32_tensor(torch, parse_bits(case.get("dst_prefill_bits", ZEROS)), dev)
        t2 = to_f32_tensor(torch, parse_bits(ZEROS), dev)
    elif kind == "vmula":
        t0 = to_f32_tensor(torch, parse_bits(case["src0_bits"]), dev)
        t1 = to_f32_tensor(torch, parse_bits(case["src1_bits"]), dev)
        t2 = to_f32_tensor(torch, parse_bits(case["acc_in_bits"]), dev)
    else:
        t0 = to_f32_tensor(torch, parse_bits(case["input_bits"]), dev)
        t1 = to_f32_tensor(torch, parse_bits(case.get("dst_prefill_bits", ZEROS)), dev)
        t2 = to_f32_tensor(torch, parse_bits(ZEROS), dev)
    return t0, t1, t2


def compare(case: dict, obs: list[str]) -> dict:
    out = {"model_matches": {}, "native_status": "UNSOURCED / UNVALIDATED"}
    for key, want in case.get("predicted", {}).items():
        if isinstance(want, str) and want.startswith("0x") and len(want) == 10:
            out["model_matches"][key] = {
                "lane0": obs[0].upper() == want.upper(),
                "any_lane": any(o.upper() == want.upper() for o in obs),
                "all_lanes": all(o.upper() == want.upper() for o in obs),
            }
        elif isinstance(want, list) and want and isinstance(want[0], str):
            if len(want[0]) == 6:   # 0xXXXX bf16 codewords: not directly f32 bits
                out["model_matches"][key] = {"value_width": "bf16 codewords",
                                             "note": "compare via bf16 codeword of each observed f32 lane"}
                continue
            out["model_matches"][key] = {
                "exact_all_lanes": [o.upper() for o in obs] == [w.upper() for w in want],
                "lane0": obs[0].upper() == want[0].upper(),
            }
    prefill = case.get("dst_prefill_bits", ZEROS)
    out["layout"] = {
        "lanes_differing_from_prefill": [i for i, (o, p) in enumerate(zip(obs, prefill))
                                         if o.upper() != p.upper()],
        "note": "which lanes were written reveals PAT_VL1 / broadcast / group layout; "
                "unchanged inactive lanes distinguish MODE_MERGING from MODE_ZEROING",
    }
    return out


def run_probe(probe_name: str, kind: str, default_case: str, build_kernel,
              build_kernel_six=None) -> int:
    ap = argparse.ArgumentParser(description="03-19 native microtest (DESIGN, UNSOURCED)")
    ap.add_argument("--six-calls", action="store_true",
                    help="vcadd: six SOURCE calls + EXPLICIT vdupv(POS_LOWEST), never baked in")
    ap.add_argument("--case", default=default_case)
    ap.add_argument("--out", required=True)
    ap.add_argument("--execute", action="store_true",
                    help="REQUIRED to touch the device; --plan is CPU-only")
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--launch-watchdog", type=int, default=60,
                    help="host-side watchdog seconds around launch+synchronize; on expiry "
                         "record the phase and os._exit(124) WITHOUT killing the device or resetting")
    ap.add_argument("--tilelang-root", default=str(DEFAULT_TL))
    args = ap.parse_args()
    case = load_case(args.case)
    if case["kind"] != kind:
        print("case %r is a %s case, not %s" % (args.case, case["kind"], kind), file=sys.stderr)
        return 2
    from device_guard import DeviceWindow, GuardError, plan_text, verify_build
    if not args.execute:
        print(json.dumps(plan_text(probe_name + ".py", args.case, args.out, args.timeout),
                         indent=2))
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
    except Exception:
        pass
    if args.six_calls and build_kernel_six is None:
        print("REFUSED: --six-calls needs a six-call kernel builder", file=sys.stderr)
        return 2
    kernel = (build_kernel_six(T) if args.six_calls else build_kernel(T))
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    phases = {}
    t_start = time.monotonic()

    def mark(name):
        phases[name] = {"wall": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "elapsed_s": round(time.monotonic() - t_start, 3)}
        (outdir / "phases.json").write_text(json.dumps(phases, indent=2))

    def watchdog_fire():
        mark("launch_watchdog_fired")
        (outdir / "WATCHDOG_FIRED.txt").write_text(
            "launch watchdog fired after %ss; last phase=%s; host exited 124; device NOT "
            "reset/killed\n" % (args.launch_watchdog, sorted(phases)[-2] if len(phases) > 1 else "?"))
        # write a minimal device_window marker before the hard host exit
        (outdir / "device_window_watchdog.json").write_text(json.dumps(
            {"probe": probe_name, "case": args.case, "phases": phases,
             "watchdog_s": args.launch_watchdog, "status": "WATCHDOG_FIRED"}, indent=2))
        sys.stdout.flush(); sys.stderr.flush()
        os._exit(124)

    with DeviceWindow(args.out, args.timeout) as window:
        window.report.update({"build": build, "probe": probe_name, "case": args.case,
                              "tilelang": getattr(tilelang, "__version__", None),
                              "launch_watchdog_s": args.launch_watchdog})
        mark("start")
        mark("compile_start")
        compiled = tilelang.compile(kernel, target="ascend", out_idx=[3])
        mark("compile_done")
        src = compiled.get_kernel_source()
        (outdir / "kernel_source.txt").write_text(src)
        window.report["kernel_source_sha256"] = hashlib.sha256(src.encode()).hexdigest()
        try:
            window.report["compiled_repr"] = repr(compiled)[:300]
        except Exception:  # noqa: BLE001
            window.report["compiled_repr"] = None
        t0, t1, t2 = make_tensors(torch, case, kind, "npu:0")
        mark("tensors_ready")
        wd = threading.Timer(args.launch_watchdog, watchdog_fire)
        wd.daemon = True
        wd.start()
        try:
            mark("launch_start")
            with torch.no_grad():
                y = compiled(t0, t1, t2)
            mark("launch_returned")
            torch.npu.synchronize()
            mark("sync_done")
            obs = observed_bits(torch, y)
            mark("readback_done")
        finally:
            wd.cancel()
        verdict = compare(case, obs)
        result = {"case": args.case, "probe": probe_name, "build": build,
                  "status": "UNSOURCED / UNVALIDATED until architect 03-20 ruling",
                  "expected_bits_sha256": hashlib.sha256(EXPECTED.read_bytes()).hexdigest(),
                  "input_bits": case.get("input_bits") or case.get("input_bf16_bits")
                                or case.get("src0_bits"),
                  "dst_prefill_bits": case.get("dst_prefill_bits", ZEROS),
                  "observed_bits": obs, "comparison": verdict}
        (outdir / "observed_bits.json").write_text(json.dumps(result, indent=2))
        mark("json_written")
        window.report["phases"] = phases
        window.report["observation_captured"] = True
        print(json.dumps({"case": args.case, "observed_bits_lane0": obs[0],
                          "model_matches": verdict["model_matches"],
                          "phases": phases}, indent=2))
    return 0
