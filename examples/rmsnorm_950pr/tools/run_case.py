#!/usr/bin/env python3
"""Run one Twin case on the real 950PR: numeric check, compile snapshot, paired
benchmark against the frozen v33 baseline. One case per process."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, statistics, sys, traceback
from pathlib import Path


# --------------------------------------------------------------- device gate
#
# THIS PATH MAY NOT TOUCH A DEVICE WITHOUT EXPLICIT AUTHORISATION.
#
# The script used to import torch_npu and call torch.npu.set_device(0) as its first
# act, with no gate at all. During a host-wide device freeze that is a violation
# waiting to happen, and the cost is not hypothetical: in this program a device open
# at 13:46:26 was followed 32 s later by a host-wide docker exec stall, with the card
# already held by other tenants. A FAILED aclInit still OPENS the device node, so
# "it only errored" is not a defence -- the open happens either way.
#
# Two rules, in this order:
#   1. Presence is checked with find_spec, NEVER by importing. Importing torch_npu
#      loads libhccl and reaches the CANN stack, which is already device-adjacent;
#      the refusal has to happen before that, not after.
#   2. set_device is called only on the authorised path. It is never reached by
#      default, and never "just to see".
#
# Exit 75 is BLOCKED_ENV: the run did not happen. That is not a pass and not a
# failure -- there is no result, and reporting one would be a fabrication.
BLOCKED_ENV = 75      # consistent with the adapter; NOT_RUN below keeps suite meaning
NOT_RUN = 3           # suite convention: a dependency is unavailable
DEVICE_AUTH_ENV = "TF_DEVICE_AUTH"


def device_authorised() -> bool:
    """Explicit, per-invocation authorisation. Absence is refusal, never a default."""
    return os.environ.get(DEVICE_AUTH_ENV, "").strip() == "1"


def torch_npu_present() -> bool:
    """Is torch_npu installed? Answered WITHOUT importing it (see rule 1)."""
    try:
        return importlib.util.find_spec("torch_npu") is not None
    except (ImportError, ValueError):
        return False


def refuse_unless_authorised(device_index: int) -> None:
    """Refuse loudly, or return having done nothing at all.

    Deliberately takes no action when authorised: authorisation is a precondition
    this function checks, not a capability it grants.
    """
    if not torch_npu_present():
        print(
            "NOT_RUN: torch_npu is not importable, so this case cannot run.\n"
            "  No device was touched. This is NOT a pass: nothing was measured.",
            file=sys.stderr,
        )
        raise SystemExit(NOT_RUN)

    if not device_authorised():
        # One message, joined: multiple print() arguments would render as a tuple,
        # which is the kind of cosmetic bug that makes a refusal unreadable exactly
        # when someone needs to read it.
        message = "\n".join([
            "BLOCKED_ENV: refusing to touch device %d." % device_index,
            "  torch_npu is present, so this script would open a device node.",
            "  Authorisation was not supplied, and absence of authorisation is a",
            "  refusal -- not a default.",
            "",
            "  The run did not happen. This is not a pass and not a failure: there is",
            "  no result to report, and reporting one would be a fabrication.",
            "",
            "  Why: a device open at 13:46:26 in this program preceded a host-wide",
            "  exec stall 32 s later, with the card already held by other tenants.",
            "  A failed aclInit still opens the device node.",
            "",
            "  To authorise deliberately, set %s=1 for this invocation only, once the" % DEVICE_AUTH_ENV,
            "  host has been cleared by the admin.",
        ])
        print(message, file=sys.stderr)
        raise SystemExit(BLOCKED_ENV)

    # Reached only with authorisation. Said out loud so the log records that this
    # run was authorised rather than merely unguarded.
    print(
        "DEVICE AUTHORISED (%s=1): opening device %d.\n"
        "  Any timings below are PROVISIONAL; they describe a shared card, not"
        " exclusive use." % (DEVICE_AUTH_ENV, device_index),
        file=sys.stderr,
    )

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

    # Refuse BEFORE importing torch_npu. The order here is the protection.
    refuse_unless_authorised(args.device)

    import torch, torch_npu, tilelang
    try:
        tilelang.disable_cache()
    except Exception:
        pass
    # Reached only because the guard above authorised it. Never unconditional.
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