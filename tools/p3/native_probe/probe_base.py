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
import inspect
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


class CaseDataError(Exception):
    """The case does not carry what the probe needs. Named, never defaulted.

    A silently substituted key is how vcvt_f32_to_bf16_rne_ftz died with
    KeyError 'input_bf16_bits' AFTER compiling: the harness had one branch for the
    whole vcvt kind and always read the WIDENING key.
    """


#: The one place a case element type maps to its bit-pattern key. The harness
#: reads this and nothing else, so a narrowing case cannot be driven by widening
#: assumptions.
SOURCE_KEY = {"bfloat16": "input_bf16_bits", "float32": "input_bits"}
WIDTH = {"bfloat16": 4, "float32": 8}


def source_dtype(case: dict) -> str:
    """The source element type, taken from the case data, not from `kind`.

    `kind` says which INSTRUCTION is under test (vcvt/vcadd/vmula). It does not
    say which DIRECTION: one vcvt case is f32->bf16 and another is bf16->f32.
    Deriving the dtype from kind is the defect this function replaces.
    """
    declared = case.get("source_dtype")
    if declared is not None:
        if declared not in SOURCE_KEY:
            raise CaseDataError("case %r declares unsupported source_dtype %r"
                                % (case.get("name"), declared))
        return declared
    # Backward-compatible inference for cases authored before the field existed.
    if "input_bf16_bits" in case:
        return "bfloat16"
    return "float32"


def _check_width(case, items, width, what):
    for item in items:
        ok = (isinstance(item, str) and item.startswith("0x")
              and len(item) == 2 + width)
        if not ok:
            raise CaseDataError(
                "case %r has a %s value %r that is not a 0x%02X-wide hex pattern"
                % (case.get("name"), what, item, width))


def input_bits_of(case: dict) -> list:
    """The source bit patterns, by the case DECLARED dtype. Refuses when absent."""
    dtype = source_dtype(case)
    key = SOURCE_KEY[dtype]
    if key not in case:
        raise CaseDataError(
            "case %r is %s-sourced and must carry %r; it carries %r"
            % (case.get("name"), dtype, key,
               sorted(k for k in case if k.endswith("_bits"))))
    _check_width(case, case[key], WIDTH[dtype], dtype)
    return case[key]


def dest_dtype(case: dict) -> str:
    return case.get("dest_dtype") or "float32"


def prefilled_bits_of(case: dict) -> list:
    """Destination prefill in the DESTINATION width, or zeros of that width."""
    width = WIDTH[dest_dtype(case)]
    prefill = case.get("dst_prefill_bits")
    if prefill is None:
        return ["0x%0*X" % (width, 0)] * 64
    _check_width(case, prefill, width, dest_dtype(case))
    return prefill


def observed_bits(torch, t):
    """Read the output in ITS OWN width: bf16 -> 16-bit codewords, f32 -> 32-bit.

    The previous revision always cast to float32 and viewed 32-bit words. For a
    narrowing case the output IS bfloat16, so each 16-bit codeword became an
    unrelated f32 bit pattern and was compared against 16-bit predictions: two
    different quantities, silently. Width follows the tensor, never a guess.
    """
    import numpy as np
    raw = t.detach().to("cpu").contiguous()
    if raw.dtype == torch.bfloat16:
        codes = raw.view(torch.int16).numpy().view(np.uint16)
        return ["0x%04X" % int(b) for b in codes]
    if raw.dtype == torch.float32:
        return ["0x%08X" % int(b) for b in raw.numpy().view(np.uint32)]
    raise CaseDataError("unsupported output dtype %r: refusing to reinterpret it"
                        % (raw.dtype,))


def make_tensors(torch, case, kind, dev):
    """Build (src, dst_prefill, aux) from the case DECLARED dtypes.

    vcvt is the only kind whose direction varies per case, so the DTYPE - not the
    kind - selects the constructor and the bit width. Reading the widening key for
    every vcvt case is exactly what produced KeyError input_bf16_bits on the
    narrowing case, after the kernel had already compiled.
    """
    if kind == "vcvt":
        dtype = source_dtype(case)
        src_bits = parse_bits(input_bits_of(case))
        t0 = (to_bf16_tensor(torch, src_bits, dev) if dtype == "bfloat16"
              else to_f32_tensor(torch, src_bits, dev))
        prefill = parse_bits(prefilled_bits_of(case))
        t1 = (to_bf16_tensor(torch, prefill, dev) if dest_dtype(case) == "bfloat16"
              else to_f32_tensor(torch, prefill, dev))
        t2 = to_f32_tensor(torch, parse_bits(ZEROS), dev)
    elif kind == "vmula":
        t0 = to_f32_tensor(torch, parse_bits(case["src0_bits"]), dev)
        t1 = to_f32_tensor(torch, parse_bits(case["src1_bits"]), dev)
        t2 = to_f32_tensor(torch, parse_bits(case["acc_in_bits"]), dev)
    else:
        t0 = to_f32_tensor(torch, parse_bits(input_bits_of(case)), dev)
        t1 = to_f32_tensor(torch, parse_bits(prefilled_bits_of(case)), dev)
        t2 = to_f32_tensor(torch, parse_bits(ZEROS), dev)
    return t0, t1, t2


def _lane_verdict(obs, want):
    """Per-lane comparison of an observation against a prediction, plus a summary.

    A note saying "compare via the bf16 codeword" is NOT a comparison - that was the
    gap: the narrowing evidence carried a description of what should be compared
    instead of the comparison itself. This returns the actual lane-by-lane result.
    """
    if len(obs) != len(want):
        return {"comparable": False,
                "why": "length mismatch: observed %d lanes, predicted %d" % (len(obs), len(want))}
    mismatched = [i for i, (o, w) in enumerate(zip(obs, want))
                  if o.upper() != w.upper()]
    return {"comparable": True,
            "lanes_compared": len(obs),
            "lanes_matching": len(obs) - len(mismatched),
            "mismatched_lanes": mismatched,
            "mismatched_observed": [obs[i] for i in mismatched],
            "mismatched_predicted": [want[i] for i in mismatched],
            "exact_all_lanes": not mismatched,
            "lane0": obs[0].upper() == want[0].upper()}


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
            if len(want[0]) == 6:   # 0xXXXX 16-bit codewords
                if len(obs[0]) != 6:
                    out["model_matches"][key] = {
                        "comparable": False,
                        "why": "predicted 16-bit codewords but observed %d-bit words; "
                               "the readback width does not match the prediction" % (
                                   len(obs[0]) - 2)}
                    continue
                out["model_matches"][key] = _lane_verdict(obs, want)
                continue
            out["model_matches"][key] = _lane_verdict(obs, want)
    # Width-aware: a bf16 destination has 16-bit prefill, an f32 destination 32-bit.
    # Using the 8-wide ZEROS constant against 6-wide bf16 observations would report
    # EVERY lane as differing from the prefill - a silent, total misread.
    prefill = prefilled_bits_of(case)
    out["layout"] = {
        "lanes_differing_from_prefill": [i for i, (o, p) in enumerate(zip(obs, prefill))
                                         if o.upper() != p.upper()],
        "note": "which lanes were written reveals PAT_VL1 / broadcast / group layout; "
                "unchanged inactive lanes distinguish MODE_MERGING from MODE_ZEROING",
    }
    # A part=0 NARROWING conversion writes one 16-bit result per 32-bit destination
    # word, so the even/odd slot split is a CHECKED property, not an assumption.
    if dest_dtype(case) == "bfloat16" and len(obs[0]) == 6:
        even = [obs[i] for i in range(0, len(obs), 2)]
        odd = [obs[i] for i in range(1, len(obs), 2)]
        # Compare UPPER-CASED on both sides: obs values are "0x.." from the
        # formatter, so a literal "0x0000" test against an uppercased string is
        # always false. That bug made every odd slot look written.
        zero = "0X0000"
        odd_prefill = {p.upper() for p in prefill}
        odd_zero = all(o.upper() == zero for o in odd)
        odd_holds_prefill = bool(odd) and {o.upper() for o in odd} <= odd_prefill
        preserved = odd_zero or odd_holds_prefill
        out["even_odd_slot_check"] = {
            "rule": case.get("layout", {}).get("rule"),
            "odd_slots_all_zero": odd_zero,
            "odd_slots_all_equal_prefill": odd_holds_prefill,
            "odd_slots_preserved_not_written": preserved,
            "even_slots_written": sum(1 for e in even if e.upper() != zero),
            "even_slots_total": len(even),
            "consistent_with_part0": preserved,
        }
    # FTZ/DAZ is a device property. State what was OBSERVED for the subnormal inputs
    # and never restate it from the case name.
    subnormals = case.get("subnormal_probe")
    if isinstance(subnormals, list) and subnormals:
        rows = []
        for item in subnormals:
            src_bits = item["src_f32_bits"]
            dst_slot = item["dst_slot"]
            observed = obs[dst_slot] if dst_slot < len(obs) else None
            rows.append({"src_f32_bits": src_bits,
                         "src_lane": item.get("src_lane"),
                         "dst_slot": dst_slot,
                         "observed": observed,
                         "flushed_to_zero": (observed is not None
                                             and observed.upper() == "0X0000"),
                         "expected_if_flushed": item.get("if_flushed"),
                         "expected_if_not_flushed": item.get("if_not_flushed")})
        flushed = [r["flushed_to_zero"] for r in rows]
        out["ftz_daz_observation"] = {
            "question": "are f32 subnormal inputs flushed to zero on this path?",
            "rows": rows,
            "all_flushed": all(flushed),
            "none_flushed": not any(flushed),
            "verdict": ("FTZ/DAZ INACTIVE on this path: the subnormal was converted, not "

                        "flushed" if not any(flushed) else
                        "FTZ/DAZ ACTIVE on this path: the subnormal was flushed to zero"
                        if all(flushed) else "MIXED: some subnormals flushed, some not"),
            "recorded_as": "native contract OBSERVATION under the frozen protocol; "
                           "NOT a precision-protocol failure",
            "not_derived_from": "the case name; the name says ftz and is not evidence",
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
    # A builder that accepts the case builds the direction the CASE declares.
    # One that does not keeps its old signature, so vcadd/vmula are untouched.
    def build_from_case(builder):
        try:
            params = inspect.signature(builder).parameters
        except (TypeError, ValueError):
            return builder(T)
        return builder(T, case) if len(params) >= 2 else builder(T)

    kernel = (build_from_case(build_kernel_six) if args.six_calls
              else build_from_case(build_kernel))
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
