#!/usr/bin/env python3
"""06-06 hardened elementwise diff of PERSISTED device tensors vs an INDEPENDENT
FP64->BF16 reference recomputed from the persisted X/W. CPU ONLY; archived inputs only.

What the verdict is built from (all must hold):
  1. BF16 CODEWORD comparison (uint16), NOT a Hamming distance: count of unequal
     codewords, first divergence row/col + hex, per pattern.
  2. Independent reference: numpy FP64 mean/rsqrt/product from <key>.x.npy / <key>.w.npy,
     ONE RNE to BF16. The runner golden.npy is used ONLY for input identity (its sha is
     recorded and a diagnostic codeword delta is reported); it is NEVER the pass value.
  3. FROZEN protocol atol/rtol/eps/repetitions/stability; status must be FROZEN.
  4. 4 repeats per input, 64 stability runs, and device pointer reuse are part of the
     OVERALL verdict (values_equal=false or storage not distinct => FAIL, not "reported").
  5. nonfinite device output => FAIL; empty/missing pattern set => REFUSED (never vacuous).

Exit: 0 PASS, 1 FAIL (a check failed), 2 REFUSED (inputs could not establish a judgement).

This validates the DECISION TOOL against ARCHIVED device outputs. It is NOT a new device
observation, not P6/V10 credit, and not a formal proof.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve()
TF_ROOT = HERE.parents[3]
sys.path.insert(0, str(TF_ROOT / "tools" / "p3"))
from rmsnorm_oracle import bf16_from_f64, bf16_to_f32, sha256_file  # noqa: E402

DEFAULT_PROTOCOL = (TF_ROOT.parent / "TileFoundry_Ascend950PR_AgentTeam"
                    / "artifacts/run01/p0/precision_protocol.json")
DETERMINISTIC = {"zero", "ones", "alternating"}

RULES = ("PROTOCOL_NOT_FROZEN", "TENSORS_JSON_MISSING", "TENSORS_JSON_INVALID",
         "EMPTY_PATTERN_SET", "TENSOR_FILE_MISSING", "MISSING_NUMERIC_RECORD",
         "INPUT_NOT_BF16", "NONFINITE_OUTPUT", "DETERMINISTIC_BITS_FAILED",
         "REPEATS_INCOMPLETE", "REPEATS_NOT_BITWISE_EQUAL", "STABILITY_FAILED",
         "REUSE_VALUES_MISMATCH", "REUSE_STORAGE_NOT_DISTINCT", "SHAPE_MISMATCH",
         "UNKNOWN_PATTERN")


class Refusal(RuntimeError):
    def __init__(self, code, message):
        super().__init__("%s: %s" % (code, message))
        self.code = code


def load_protocol(path: Path) -> dict:
    if not path.is_file():
        raise Refusal("PROTOCOL_NOT_FROZEN", "protocol %s missing" % path)
    text = path.read_text()
    data = json.loads(text)
    if data.get("status") != "FROZEN":
        raise Refusal("PROTOCOL_NOT_FROZEN",
                      "protocol %s status=%r is not FROZEN" % (path, data.get("status")))
    data["_path"] = str(path.resolve())
    data["_sha256"] = sha256_file(path)
    return data


def to_bits(arr: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(arr, dtype=np.float32)
    return (a.view(np.uint32) >> np.uint32(16)).astype(np.uint16)


def check_bf16(arr, name, key):
    """True iff every value is EXACTLY a bf16 codeword (inf/nan ARE codewords; they are
    rejected later as nonfinite OUTPUT, not mislabelled as a non-bf16 input)."""
    dec = bf16_to_f32(to_bits(arr))
    try:
        return bool(np.array_equal(dec, arr.astype(np.float32), equal_nan=True))
    except TypeError:
        return bool(np.array_equal(dec, arr.astype(np.float32)))


def fp64_reference(x_bits, w_bits, eps):
    x = bf16_to_f32(x_bits).astype(np.float64)
    w = bf16_to_f32(w_bits).astype(np.float64)
    mean = np.sum(x * x, axis=-1, keepdims=True) / x.shape[1]
    scale = 1.0 / np.sqrt(mean + eps)
    y = x * scale * w
    return bf16_from_f64(y), y


def first_divergence(dev_bits, ref_bits):
    idx = np.argwhere(dev_bits != ref_bits)
    if idx.size == 0:
        return None
    r, c = int(idx[0][0]), int(idx[0][1])
    return {"row": r, "col": c,
            "device_bf16": "0x%04X" % int(dev_bits[r, c]),
            "reference_bf16": "0x%04X" % int(ref_bits[r, c])}


def diff_key(cdir: Path, entry: dict, shape, protocol) -> dict:
    key = entry["key"]
    pattern = entry.get("pattern")
    if pattern not in DETERMINISTIC and pattern != "random":
        raise Refusal("UNKNOWN_PATTERN", "%s: pattern %r" % (key, pattern))
    rows, d = shape
    paths = {"x": cdir / (key + ".x.npy"), "w": cdir / (key + ".w.npy"),
             "device": cdir / (key + ".device.npy"), "golden": cdir / (key + ".golden.npy")}
    for name, p in paths.items():
        if not p.is_file():
            raise Refusal("TENSOR_FILE_MISSING", "%s: %s missing" % (key, p.name))
    x = np.load(paths["x"]); w = np.load(paths["w"])
    dev = np.load(paths["device"]); gold = np.load(paths["golden"])
    if tuple(x.shape) != (rows, d) or tuple(w.shape) != (d,):
        raise Refusal("SHAPE_MISMATCH", "%s: x%r w%r expected %r/D" % (key, x.shape, w.shape, (rows, d)))
    if tuple(dev.shape) != (rows, d) or tuple(gold.shape) != (rows, d):
        raise Refusal("SHAPE_MISMATCH", "%s: device/golden shape mismatch" % key)
    for name, arr in (("x", x), ("w", w), ("device", dev), ("golden", gold)):
        if not check_bf16(arr, name, key):
            raise Refusal("INPUT_NOT_BF16",
                          "%s: %s is not exactly bf16-representable" % (key, name))
    # Inputs must be finite; a nonfinite DEVICE output is a FAILURE (NONFINITE_OUTPUT),
    # handled below, not a refusal.
    if not (np.isfinite(x).all() and np.isfinite(w).all() and np.isfinite(gold).all()):
        raise Refusal("INPUT_NOT_BF16", "%s: input/golden carries non-finite values" % key)
    xb, wb = to_bits(x), to_bits(w)
    dev_bits, gold_bits = to_bits(dev), to_bits(gold)
    ref_bits, ref_exact = fp64_reference(xb, wb, float(protocol["eps"]))
    atol, rtol = float(protocol["atol"]), float(protocol["rtol"])
    got = bf16_to_f32(dev_bits).astype(np.float64)
    refv = bf16_to_f32(ref_bits).astype(np.float64)
    finite = bool(np.isfinite(got).all())
    diff = np.abs(got - refv) if finite else np.full(got.shape, np.inf)
    tol = atol + rtol * np.abs(refv)
    mismatch = int(np.count_nonzero(diff > tol)) if finite else int(got.size)
    max_abs = float(diff.max()) if finite else None
    max_rel = float((diff / np.maximum(np.abs(refv), 1e-12)).max()) if finite else None
    det = pattern in DETERMINISTIC
    codeword_delta = int(np.count_nonzero(dev_bits != ref_bits))
    det_exact = bool(codeword_delta == 0) if det else None
    failures = []
    if not finite:
        failures.append("NONFINITE_OUTPUT")
    if finite and mismatch != 0:
        failures.append("PROTOCOL_MISMATCH")
    if det and not det_exact:
        failures.append("DETERMINISTIC_BITS_FAILED")
    return {
        "key": key, "pattern": pattern, "seed": entry.get("seed"),
        "input_files": {k: {"sha256": sha256_file(p), "size": p.stat().st_size}
                        for k, p in paths.items()},
        "bf16_codeword": {"unequal_count": codeword_delta,
                          "elements": int(dev_bits.size),
                          "fraction": codeword_delta / dev_bits.size,
                          "first_divergence": first_divergence(dev_bits, ref_bits)},
        "protocol": {"mismatch_count": mismatch, "max_abs": max_abs, "max_rel": max_rel,
                     "atol": atol, "rtol": rtol,
                     "max_abs_le_atol": (None if max_abs is None else bool(max_abs <= atol)),
                     "rule": "per element |device-reference| <= atol + rtol*|reference|; mismatch_count must be 0"},
        "deterministic": {"required": det, "bf16_codewords_exact": det_exact},
        "golden_identity_only": {"sha256": sha256_file(paths["golden"]),
                                 "codeword_delta_vs_independent_ref": int(
                                     np.count_nonzero(gold_bits != ref_bits)),
                                 "note": "golden is NOT a pass value; diagnostic only"},
        "failures": failures,
        "pass": not failures,
        "pass_basis": ("bf16 codeword exact (deterministic)" if det and det_exact else
                       "all codewords equal" if codeword_delta == 0 else
                       ("within atol alone" if (max_abs is not None and max_abs <= atol) else
                        "PASS on rtol wing only: max_abs > atol")) if not failures else "FAIL",
    }


def check_repeats_stability_reuse(numeric: dict, protocol: dict) -> dict:
    out = {"rules": {}}
    reps_required = int(protocol.get("repetitions_per_input", 4))
    runs_required = int(protocol.get("stability_runs_per_case", 64))
    tests = numeric.get("tests") or []
    groups = {}
    for t in tests:
        groups.setdefault((t.get("pattern"), t.get("seed")), []).append(t)
    bad_groups = []
    for gkey, rows in groups.items():
        reps = len(rows)
        wrong = [r for r in rows if r.get("repeat_bitwise_equal") is False]
        if reps != reps_required:
            bad_groups.append("%s: %d repeats != %d" % (gkey, reps, reps_required))
        elif wrong:
            bad_groups.append("%s: %d repeat(s) not bitwise-equal" % (gkey, len(wrong)))
    out["rules"]["REPEATS_INCOMPLETE"] = [g for g in bad_groups if "!=" in g]
    out["rules"]["REPEATS_NOT_BITWISE_EQUAL"] = [g for g in bad_groups if "bitwise" in g]
    stab = numeric.get("stability") or {}
    if int(stab.get("runs", -1)) != runs_required or not stab.get("all_bitwise_equal_first") \
            or not stab.get("fixed_input"):
        out["rules"]["STABILITY_FAILED"] = [json.dumps(stab)]
    reuse = numeric.get("output_buffer_reuse") or {}
    if not reuse.get("values_equal"):
        out["rules"]["REUSE_VALUES_MISMATCH"] = [json.dumps(reuse)]
    if not reuse.get("device_storage_distinct") or \
            reuse.get("device_ptr_first") == reuse.get("device_ptr_second"):
        out["rules"]["REUSE_STORAGE_NOT_DISTINCT"] = [json.dumps(reuse)]
    out["repeats_required"] = reps_required
    out["stability_runs_required"] = runs_required
    out["pass"] = not any(out["rules"].values())
    return out


def diff_case(cdir: Path, protocol: dict) -> dict:
    tj_path = cdir / "tensors.json"
    if not tj_path.is_file():
        raise Refusal("TENSORS_JSON_MISSING", "%s missing" % tj_path)
    tj = json.loads(tj_path.read_text())
    entries = tj.get("tensors") or []
    if not entries:
        raise Refusal("EMPTY_PATTERN_SET", "%s lists zero patterns" % tj_path)
    shape = tuple(int(v) for v in tj["shape"])
    numeric_path = cdir / "numeric.json"
    if not numeric_path.is_file():
        raise Refusal("MISSING_NUMERIC_RECORD",
                      "%s: numeric.json absent, repeats/stability/reuse cannot be judged" % cdir)
    numeric = json.loads(numeric_path.read_text())
    out = {"case_dir": str(cdir), "case_id": tj.get("case_id", cdir.name),
           "shape": list(shape), "tensors_json_sha256": sha256_file(tj_path),
           "numeric_sha256": sha256_file(numeric_path), "patterns": []}
    for entry in entries:
        out["patterns"].append(diff_key(cdir, entry, shape, protocol))
    out["repeats_stability_reuse"] = check_repeats_stability_reuse(numeric, protocol)
    pat_ok = all(p["pass"] for p in out["patterns"])
    out["case_pass"] = bool(pat_ok and out["repeats_stability_reuse"]["pass"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases-dir", default=None,
                    help="parent directory with one subdirectory per case (tensors.json inside)")
    ap.add_argument("--case-dir", action="append", default=[],
                    help="explicit case directory (repeatable)")
    ap.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    report = {"tool": "direct_diff (06-06 hardened)", "scope":
              "validates the DECISION TOOL on ARCHIVED device outputs; NOT a new device observation",
              "rules": list(RULES), "cases": [], "refusals": []}
    try:
        protocol = load_protocol(Path(args.protocol))
    except Refusal as exc:
        report["refusals"].append(str(exc)); report["verdict"] = "REFUSED"
        _emit(report, args.out); print("REFUSED", exc, file=sys.stderr); return 2
    report["protocol"] = {"path": protocol["_path"], "sha256": protocol["_sha256"],
                          "status": protocol.get("status"),
                          "protocol_id": protocol.get("protocol_id"),
                          "atol": protocol.get("atol"), "rtol": protocol.get("rtol"),
                          "eps": protocol.get("eps"),
                          "repetitions_per_input": protocol.get("repetitions_per_input"),
                          "stability_runs_per_case": protocol.get("stability_runs_per_case")}
    if args.case_dir:
        dirs = [Path(p) for p in args.case_dir]
    elif args.cases_dir:
        base = Path(args.cases_dir)
        dirs = sorted(p for p in base.iterdir() if p.is_dir() and (p / "tensors.json").is_file())
        if not dirs:
            report["refusals"].append("EMPTY_PATTERN_SET: no case dirs with tensors.json under %s" % base)
    else:
        ap.error("--cases-dir or --case-dir required")
    try:
        for cdir in dirs:
            report["cases"].append(diff_case(cdir, protocol))
    except Refusal as exc:
        report["refusals"].append(str(exc)); report["verdict"] = "REFUSED"
        _emit(report, args.out); print("REFUSED", exc, file=sys.stderr); return 2
    if not report["cases"]:
        report["refusals"].append("EMPTY_PATTERN_SET: no cases judged")
        report["verdict"] = "REFUSED"; _emit(report, args.out); return 2
    ok = all(c["case_pass"] for c in report["cases"])
    report["verdict"] = "PASS" if ok else "FAIL"
    _emit(report, args.out)
    for c in report["cases"]:
        print("%-18s %s patterns=%d %s" % (c["case_id"], "%dx%d" % tuple(c["shape"]),
                                           len(c["patterns"]), "PASS" if c["case_pass"] else "FAIL"))
        for p in c["patterns"]:
            print("   %-12s codeword_diff=%-6d mismatch=%-4d max_abs=%-10s det=%-5s %s"
                  % (p["key"], p["bf16_codeword"]["unequal_count"],
                     p["protocol"]["mismatch_count"],
                     ("%.6g" % p["protocol"]["max_abs"]) if p["protocol"]["max_abs"] is not None else None,
                     p["deterministic"]["bf16_codewords_exact"], p["pass_basis"]))
    print("verdict:", report["verdict"])
    return 0 if ok else 1


def _emit(report, out):
    text = json.dumps(report, indent=2)
    if out:
        Path(out).write_text(text)
    return text


if __name__ == "__main__":
    raise SystemExit(main())
