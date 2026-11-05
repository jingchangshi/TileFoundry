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
import os
from pathlib import Path

# Machine-independent root: from TF_WORKSPACE_ROOT, defaulting to ~/workspace.
# A literal home path makes the file unusable elsewhere; a placeholder would
# make it unusable HERE, so it is derived.
_WS_ROOT = os.environ.get("TF_WORKSPACE_ROOT") or str(Path.home() / "workspace")


BLOCKED_ENV = 75
NOT_RUN = 3
DEVICE_AUTH_ENV = "TF_DEVICE_AUTH"
# ONE canonical host lock, the same file run_all_cases.sh holds (HANDOFF §4 / Lead D4).
# run_all_cases.sh holds it once and exports TF_DEVICE_LOCK_HELD=1 to its children; this
# script MUST NOT take a second lock in that case. A standalone invocation takes the SAME
# lock, so there is exactly one holder on every path and never a second lock file.
#
# It is the TEAM lock, not a "device 0" lock: the FILE NAME is history (it was created
# when the team ran one fixed card) and is kept so the file everyone already meets at
# stays the file. Nothing about the name restricts which card may run -- the card comes
# from the ordered visible list, see resolve_device_selection().
TEAM_LOCK_PATH = "/home/.tf_device0.lock"  # no-machine-path: allow
FROZEN_PROTOCOL = (f"{_WS_ROOT}/TileFoundry_Ascend950PR_AgentTeam"
                   "/artifacts/run01/p0/precision_protocol.json")
_LOCK_FH = None


# --- tilelang provenance ----------------------------------------------------
# The device run must import the INSTALLED WHEEL from the shared prefix. There is
# deliberately NO source-tree fallback: with workspace/tilelang first on PYTHONPATH
# this script imports the checkout, and then the prefix passing its four layers says
# nothing about what actually ran on the card.
TILELANG_ROOT_ENV = "TILELANG_ROOT"
_ADAPTER_ENV_CONTEXT = (f"{_WS_ROOT}/AgentGates/agent_gates/adapters/tilefoundry-npu"
                        "/scripts/env_context.py")


def expected_tilelang_prefix() -> str:
    """Where the deployed tilelang wheel must live. Refuses; never guesses.

    Order: TILELANG_ROOT, then the adapter env context (the same SHARED_PREFIX
    configure.py provisions), then a NAMED refusal TILELANG_ROOT_UNSET.
    """
    root = (os.environ.get(TILELANG_ROOT_ENV) or "").strip()
    if root:
        return root
    ctx = Path(_ADAPTER_ENV_CONTEXT)
    if ctx.is_file():
        import runpy
        try:
            prefix = runpy.run_path(str(ctx)).get("SHARED_PREFIX")
        except Exception as exc:  # noqa: BLE001
            print("BLOCKED_ENV: adapter env context %s is unusable: %s" % (ctx, exc),
                  file=sys.stderr)
        else:
            if prefix:
                return str(prefix)
    print("\n".join([
        "BLOCKED_ENV: TILELANG_ROOT_UNSET",
        "  Neither %s nor the adapter env context" % TILELANG_ROOT_ENV,
        "  (%s) names the prefix that owns the" % _ADAPTER_ENV_CONTEXT,
        "  installed tilelang wheel. Falling back to a source checkout would measure",
        "  code nobody deploys, so this refuses rather than guesses.",
        "  Set %s=<venv root, e.g. .../envs/tfnpu-v1>." % TILELANG_ROOT_ENV,
    ]), file=sys.stderr)
    raise SystemExit(BLOCKED_ENV)


def assert_tilelang_from_prefix(module) -> dict:
    """The imported tilelang must be the wheel under <prefix>/lib/*/site-packages."""
    prefix = expected_tilelang_prefix()
    real_prefix = os.path.realpath(prefix)
    resolved = os.path.realpath(getattr(module, "__file__", "") or "")
    under_lib = resolved.startswith(os.path.join(real_prefix, "lib") + os.sep)
    in_site = (os.sep + "site-packages" + os.sep) in resolved
    if not (under_lib and in_site):
        looks_like_checkout = os.path.isfile(
            os.path.join(os.path.dirname(os.path.dirname(resolved)), "VERSION"))
        print("\n".join([
            "BLOCKED_ENV: TILELANG_NOT_FROM_PREFIX",
            "  tilelang resolved to %r," % resolved,
            "  which is NOT under %r." % os.path.join(real_prefix, "lib"),
            "  %s" % ("That path is a SOURCE CHECKOUT, not an installed wheel."
                     if looks_like_checkout else "Expected a wheel install."),
            "  Refusing to measure: the device run must import what the prefix verified.",
        ]), file=sys.stderr)
        raise SystemExit(BLOCKED_ENV)
    return {"expected_prefix": real_prefix, "tilelang_file": resolved,
            "version": getattr(module, "__version__", None)}


def _ensure_team_lock() -> None:
    """Acquire the ONE canonical TEAM lock, unless a parent already holds it.

    Exactly one holder per run: if TF_DEVICE_LOCK_HELD=1 the parent (run_all_cases.sh)
    holds the flock and this process must not re-acquire it (a second flock on the same
    open file description from a child is the duplicate-lock bug we are removing).
    """
    global _LOCK_FH
    if os.environ.get("TF_DEVICE_LOCK_HELD", "").strip() == "1":
        print("device lock: already held by parent (TF_DEVICE_LOCK_HELD=1); not re-acquiring",
              file=sys.stderr)
        return
    import fcntl
    try:
        fh = open(TEAM_LOCK_PATH, "a+")
    except OSError as exc:
        print("BLOCKED_ENV: cannot open the team lock %s: %s" % (TEAM_LOCK_PATH, exc),
              file=sys.stderr)
        raise SystemExit(BLOCKED_ENV)
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        fh.close()
        print("\n".join([
            "BLOCKED_ENV: another team measurement job holds %s." % TEAM_LOCK_PATH,
            "  Refusing to enter: a second concurrent device job must not run.",
            "  This is a refusal, not a failed attempt; nothing was opened.",
        ]), file=sys.stderr)
        raise SystemExit(BLOCKED_ENV)
    _LOCK_FH = fh
    print("team lock: acquired %s for this process" % TEAM_LOCK_PATH, file=sys.stderr)


sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools" / "p3" / "native_probe"))


def resolve_device_selection(requested: "int | None") -> dict:
    """Resolve a LOGICAL index inside the ordered visible list -- never assume a card.

    The ordered ASCEND_RT_VISIBLE_DEVICES list is the ONLY selector. Entry i of that
    list is the physical card a logical index i addresses. Two consequences, both
    deliberate:
      * no card number is baked in anywhere, so the same code runs on whatever list
        the operator granted;
      * a request outside the list is a NAMED refusal. Falling back to entry 0 would
        silently measure a different card than the one asked for -- the failure this
        function exists to prevent.
    """
    from device_guard import GuardError, check_device_identity
    try:
        ident = check_device_identity(dict(os.environ))
    except GuardError as exc:
        print("BLOCKED_ENV: %s" % exc, file=sys.stderr)
        raise SystemExit(BLOCKED_ENV)
    visible = list(ident["visible_ids"])
    if requested is None:
        logical = 0
        reason = "first entry of the ordered visible list (no --device given)"
    else:
        if not 0 <= requested < len(visible):
            print("\n".join([
                "BLOCKED_ENV: DEVICE_INDEX_NOT_VISIBLE",
                "  --device %d is outside the ordered visible list %r." % (requested, visible),
                "  A logical index addresses entry N of that list; a %d-entry list has no"
                "  entry %d. Refusing: substituting entry 0 would measure a different card."
                % (len(visible), requested),
                "  Widen ASCEND_RT_VISIBLE_DEVICES (or the adapter --device-ids) instead.",
            ]), file=sys.stderr)
            raise SystemExit(BLOCKED_ENV)
        logical = requested
        reason = "explicit --device argument"
    selection = {
        "visible_env": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "visible_list": visible,
        "logical_index": logical,
        "physical_device_id": visible[logical],
        "selection_reason": reason,
    }
    print("device selection: visible=%r logical=%d physical=%d (%s)"
          % (visible, logical, selection["physical_device_id"], reason), file=sys.stderr)
    return selection


def _device_identity_errors(uid: int, visible: str | None, ffi_version: str | None,
                            selection: dict) -> list[str]:
    """Explicit, testable checks: root, visible list, device ffi 0.1.12.

    CPU and device ffi must not be mixed: the device env is 0.1.12 (matching the
    tilelang-built libtvm_ffi.so); 0.1.15.post2 is the CPU env and is REFUSED here.
    """
    errors = []
    if uid != 0:
        errors.append("device work must run as root, uid is %d" % uid)
    try:
        from device_guard import check_device_identity
        env = dict(os.environ)
        env["ASCEND_RT_VISIBLE_DEVICES"] = str(visible)
        ident = check_device_identity(env)
        logical = selection.get("logical_index")
        if not isinstance(logical, int) or not 0 <= logical < len(ident["visible_ids"]):
            errors.append("logical index %r is outside visible list %r"
                          % (logical, ident["visible_ids"]))
    except Exception as exc:  # noqa: BLE001
        errors.append("device identity refused: %s" % exc)
    if ffi_version is None:
        errors.append("apache-tvm-ffi version is unknown; device env must be 0.1.12")
    elif not str(ffi_version).startswith("0.1.12"):
        errors.append("apache-tvm-ffi must be 0.1.12 in the device env, got %r "
                      "(0.1.15.post2 is the CPU env and must not be mixed)" % (ffi_version,))
    return errors


def _assert_device_identity(selection: dict) -> None:
    try:
        import importlib.metadata as _md
        ffi = _md.version("apache-tvm-ffi")
    except Exception:  # noqa: BLE001
        ffi = None
    errors = _device_identity_errors(os.geteuid(), os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
                                     ffi, selection)
    if errors:
        print("BLOCKED_ENV: device identity/ffi preconditions failed:", file=sys.stderr)
        for e in errors:
            print("  - " + e, file=sys.stderr)
        raise SystemExit(BLOCKED_ENV)
    print("device identity OK: uid=0, visible=%r, logical=%d, physical=%d, apache-tvm-ffi=%s"
          % (selection["visible_list"], selection["logical_index"],
             selection["physical_device_id"], ffi), file=sys.stderr)


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
    ap.add_argument("--device", type=int, default=None,
                    help="LOGICAL index into the ordered ASCEND_RT_VISIBLE_DEVICES list; "
                         "default: its first entry (no card number is assumed). An index "
                         "outside the list is refused by name (DEVICE_INDEX_NOT_VISIBLE).")
    ap.add_argument("--windows", type=int, default=20)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--protocol", default=FROZEN_PROTOCOL,
                    help="FROZEN precision protocol (status must be FROZEN)")
    ap.add_argument("--baseline-kernel",
                    default=f"{_WS_ROOT}/TileFoundry_Ascend950PR_AgentTeam/reference/kernel_v33.py")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    _ensure_team_lock()
    # The card is chosen from the ordered visible list, never from a constant.
    selection = resolve_device_selection(args.device)
    _assert_device_identity(selection)
    refuse_unless_authorised(selection["logical_index"])
    protocol = load_frozen_protocol(args.protocol)

    import torch, torch_npu, tilelang
    # Gate BEFORE touching the card: provenance of the imported tilelang.
    tilelang_provenance = assert_tilelang_from_prefix(tilelang)
    print("tilelang provenance OK: %s (%s)" % (tilelang_provenance["tilelang_file"],
                                               tilelang_provenance["version"]), file=sys.stderr)
    try:
        tilelang.disable_cache()
    except Exception:
        pass
    torch.npu.set_device(selection["logical_index"])
    sys.path.insert(0, f"{_WS_ROOT}/TileFoundry/examples/rmsnorm_950pr")
    from runtime_model import RuntimeModel, check_case, paired_benchmark
    from kernels.twin_rmsnorm import TwinConfig

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cfg = TwinConfig(args.case_id, args.shape[0], args.shape[1], args.rows, args.loop_mode,
                     args.stages, args.xversions, args.yversions)
    model = RuntimeModel(cfg)
    model.ensure()
    src = model.kernel_source()
    (out / "kernel_source.txt").write_text(src)
    devname = torch.npu.get_device_name(selection["logical_index"])

    try:
        import importlib.metadata as _md
        tvm_ffi_ver = _md.version("apache-tvm-ffi")
    except Exception:
        tvm_ffi_ver = None

    protocol_text = Path(args.protocol).read_text()
    # The TRIPLE that makes a run reproducible: which list was granted, which logical
    # index was used inside it, and which physical card that entry named.
    selection = dict(selection, device_name=devname)
    device_info = {
        "device_index": selection["logical_index"], "device_name": devname,
        "device_selection": selection,
        "torch": torch.__version__,
        "torch_npu": getattr(torch_npu, "__version__", None),
        "tilelang": getattr(tilelang, "__version__", None),
        "tilelang_file": getattr(tilelang, "__file__", None),
        "tilelang_provenance": tilelang_provenance,
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

    numeric = check_case(model, torch, selection["logical_index"], protocol, save_dir=str(out))
    (out / "numeric.json").write_text(json.dumps(numeric, indent=2))
    (out / "patterns_report.json").write_text(json.dumps(numeric["patterns"], indent=2))

    spec = importlib.util.spec_from_file_location("v33_base", args.baseline_kernel)
    base = importlib.util.module_from_spec(spec); sys.modules["v33_base"] = base
    spec.loader.exec_module(base)
    base.LOOP_MODE = args.loop_mode; base._KERNELS.clear()

    def baseline_run(x, w, eps):
        return base.run(x, w, eps)

    bench = paired_benchmark(model, baseline_run, torch, selection["logical_index"],
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
