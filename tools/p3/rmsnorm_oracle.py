#!/usr/bin/env python3
"""P3 independent numeric oracle over the ACTUAL runner inputs (03-21 input ABI).

WHY THIS FILE WAS REWRITTEN
---------------------------
Until 03-21 this oracle GENERATED its own inputs (numpy default_rng, and a row-wise
w of shape (rows, D)). The device runner persists its own inputs as X[M, D] and W[D]
(examples/rmsnorm_950pr/runtime_model.py:check_case -> save_dir). Row-wise w and a
runner W[D] are different inputs, so any "device vs oracle" comparison built on the
generated files was NOT a same-input comparison. The generated path is deleted: this
tool now only CONSUMES runner artifacts, read-only, and refuses by name when they are
not the documented layout.

INPUT ABI CONSUMED (produced by run_case.py -> runtime_model.check_case)
------------------------------------------------------------------------
  <run-dir>/tensors.json      {"case_id", "shape":[M,D], "tensors":[{key,pattern,seed,
                               files:[device,golden,x,w]}]}
  <run-dir>/<key>.x.npy       X[M,D] bf16 values carried as float32 (exact)
  <run-dir>/<key>.w.npy       W[D]   bf16 values carried as float32 (exact)
  <run-dir>/<key>.golden.npy  runner FP64->BF16 reference  (float32 of bf16)
  <run-dir>/<key>.device.npy  device output, float32 of bf16

Every file is hashed (sha256) and a combined input digest over the exact bf16 bits of
X and W is recorded; no comparison can be reported without those digests. A run-dir
with zero tensors is a REFUSAL (EMPTY_TENSOR_SET), never a vacuous pass.

WHICH CHECKS THE EXIT CODE DEPENDS ON
-------------------------------------
  1. input ABI: shapes/dtype/bf16-representability/tensors.json agreement
     (ROW_WISE_W, X_SHAPE_MISMATCH, W_SHAPE_MISMATCH, INPUT_NOT_BF16, ...);
  2. input identity: the runner's own FP64 reference recomputed from the persisted
     X/W must reproduce the persisted golden.npy BIT-EXACTLY. This is the check that
     makes "same input" an assertion instead of a story; the recomputation uses the
     runner's formula on CPU (torch), while the oracle's own numeric reference below
     stays independent of torch;
  3. numeric protocol (frozen): |device - independent ref| <= atol + rtol*|ref|,
     mismatch_count == 0, and deterministic patterns (zero/ones/alternating) must be
     BITS-exact. max_abs is reported separately and is NEVER the pass rule;
  4. diagnostics (NOT pass criteria): device vs the v33-semantics model, whose
     lane-reduction order is still a PLACEHOLDER.

The oracle's numeric model is unchanged: FP32 accumulation after BF16->FP32, chains
per case (prefill 4, decode 2), balanced merges, lane reduction UNRESOLVED, vmula as
two FP32 roundings (ASSUMED). Those assumptions are recorded in the manifest.

Exit codes: 0 all consumed runs PASS; 1 a consumed run FAILED a check;
            2 REFUSED (input ABI could not be established; nothing was judged).
Evidence scope: this is a recomputation over persisted runner inputs. It makes NO new
device observation; the device output hash is whatever the runner stored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "p2"))
import bf16_reference as p2  # noqa: E402

EPS = 1e-6
CH = 64
DEFAULT_PROTOCOL = Path("/home/s00653124/workspace/TileFoundry_Ascend950PR_AgentTeam"
                        "/artifacts/run01/p0/precision_protocol.json")
DETERMINISTIC_PATTERNS = {"zero", "ones", "alternating"}
BLOCK = 256

# NAMED ERROR STATES. Every one of these is a refusal or a failure with a stable code,
# so a caller can assert which rule fired instead of matching prose.
ABI_CODES = (
    "RUN_DIR_MISSING", "TENSORS_JSON_MISSING", "TENSORS_JSON_INVALID", "EMPTY_TENSOR_SET",
    "X_SHAPE_MISMATCH", "W_SHAPE_MISMATCH", "ROW_WISE_W", "INPUT_NOT_BF16",
    "INPUT_NONFINITE", "GOLDEN_NOT_BF16", "DEVICE_NOT_BF16", "GOLDEN_MISSING",
    "DEVICE_OUTPUT_MISSING", "SHAPE_DISAGREEMENT",
    "TENSOR_FILE_MISSING", "INPUT_SHA_MISMATCH", "CHAINS_UNKNOWN", "PROTOCOL_NOT_FROZEN",
)
CHECK_CODES = (
    "INPUT_IDENTITY_UNCHECKED", "INPUT_IDENTITY_FAILED", "DEVICE_PROTOCOL_FAILED",
    "DETERMINISTIC_BITS_FAILED", "EMPTY_PATTERN_SET", "MODEL_DIAG_MISSING",
)


class AbiError(RuntimeError):
    """The persisted inputs do not conform to the runner ABI: nothing may be judged."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__("%s: %s" % (code, message))
        self.code = code


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def bf16_from_f64(a: np.ndarray) -> np.ndarray:
    """Single RNE f64 -> bf16, from the f64 significand (no intermediate f32).

    Kept from the pre-03-21 oracle: it is validated against the exact-rational P2
    reference by --self-check, so it is the one piece of the old tool that still
    carries an independent proof obligation.
    """
    a = np.ascontiguousarray(a, dtype=np.float64)
    u = a.view(np.uint64)
    sign = (((u >> np.uint64(63)) & np.uint64(1)).astype(np.uint32) << np.uint32(15))
    exp = ((u >> np.uint64(52)) & np.uint64(0x7FF)).astype(np.int64)
    mant = u & np.uint64((1 << 52) - 1)
    out = sign.copy()
    is_nan_inf = exp == 0x7FF
    e = exp - 1023
    sig = mant | np.uint64(1 << 52)

    shift = np.uint64(45)
    q = sig >> shift
    rem = sig & ((np.uint64(1) << shift) - np.uint64(1))
    half = np.uint64(1) << (shift - np.uint64(1))
    q = q + ((rem > half) | ((rem == half) & ((q & np.uint64(1)) == np.uint64(1)))).astype(np.uint64)
    carry = q == np.uint64(256)
    q = np.where(carry, np.uint64(128), q)
    e = np.where(carry, e + 1, e)
    normal = (e >= -126) & (exp != 0)

    sub = (exp != 0) & (e < -126) & (e >= -134)
    k = np.clip(-(e + 81), 46, 53).astype(np.uint64)
    q2 = sig >> k
    rem2 = sig & ((np.uint64(1) << k) - np.uint64(1))
    half2 = np.uint64(1) << (k - np.uint64(1))
    q2 = q2 + ((rem2 > half2) | ((rem2 == half2) & ((q2 & np.uint64(1)) == np.uint64(1)))).astype(np.uint64)
    q2 = np.where(q2 >= np.uint64(128), np.uint64(128), q2)

    exp_bits = np.where(normal, (e + 127), 0).astype(np.uint32)
    mant_bits = np.where(normal, (q.astype(np.int64) - 128), q2.astype(np.int64)).astype(np.uint32)
    bits = out | (exp_bits << np.uint32(7)) | mant_bits
    bits = np.where(sub, out | q2.astype(np.uint32), bits)

    tiny = (exp == 0) | ((exp != 0) & (e < -134))
    bits = np.where(tiny, out, bits)
    bits = np.where(is_nan_inf, out | np.uint32(0x7F80), bits)
    bits = np.where(is_nan_inf & (mant != 0), out | np.uint32(0x7FC0), bits)
    bits = np.where(normal & (e > 127) & (~is_nan_inf), out | np.uint32(0x7F80), bits)
    return bits.astype(np.uint16)


def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint16)
    sign = np.where((bits & np.uint16(0x8000)) != 0, -1.0, 1.0).astype(np.float32)
    exp = ((bits >> np.uint16(7)) & np.uint16(0xFF)).astype(np.int64)
    mant = (bits & np.uint16(0x7F)).astype(np.int64)
    val = np.where(exp == 0,
                   mant.astype(np.float32) * np.float32(2.0 ** -133),
                   (128 + mant).astype(np.float32) * np.float32(2.0) ** (exp - 127 - 7))
    return (sign * val.astype(np.float32)).astype(np.float32)


def bf16_bits_of_f32(a: np.ndarray) -> np.ndarray:
    """bf16 bits of an array documented as 'bf16 values carried as float32'."""
    a = np.ascontiguousarray(a, dtype=np.float32)
    return (a.view(np.uint32) >> np.uint32(16)).astype(np.uint16)


def chains_for_case(case_id: str) -> int:
    if "prefill" in case_id:
        return 4
    if "decode" in case_id:
        return 2
    raise AbiError("CHAINS_UNKNOWN",
                   "case_id %r carries neither 'prefill' nor 'decode'; the accumulator "
                   "count is PER CASE and must not be guessed" % (case_id,))


def load_protocol(path: Path) -> dict:
    if not path.is_file():
        raise AbiError("PROTOCOL_NOT_FROZEN", "protocol file %s does not exist" % path)
    data = json.loads(path.read_text())
    if data.get("status") != "FROZEN":
        raise AbiError("PROTOCOL_NOT_FROZEN",
                       "protocol %s has status=%r, not FROZEN: an unfrozen protocol must "
                       "not pass or fail a candidate" % (path, data.get("status")))
    return data


def _require_file(path: Path, code: str) -> None:
    if not path.is_file():
        raise AbiError(code, "%s is missing" % path)


def read_inputs(run_dir: Path, entry: dict, shape: tuple[int, int]) -> dict:
    """Load and ABI-check one persisted pattern. Returns arrays + file digests."""
    rows, width = shape
    key = entry["key"]
    out = {"key": key, "pattern": entry.get("pattern"), "seed": entry.get("seed"),
           "files": {}, "input_digest": None}
    paths = {"x": run_dir / (key + ".x.npy"), "w": run_dir / (key + ".w.npy"),
             "golden": run_dir / (key + ".golden.npy"),
             "device": run_dir / (key + ".device.npy")}
    expected = {"x": "TENSOR_FILE_MISSING", "w": "TENSOR_FILE_MISSING",
                "golden": "GOLDEN_MISSING", "device": "DEVICE_OUTPUT_MISSING"}
    for kind, path in paths.items():
        _require_file(path, expected[kind])
        out["files"][kind] = {"name": path.name, "sha256": sha256_file(path),
                              "size": path.stat().st_size}
    # A saved sha in tensors.json (if a future runner writes one) must agree with the file.
    saved = entry.get("sha256") or {}
    for kind, digest in saved.items():
        if kind in out["files"] and out["files"][kind]["sha256"] != digest:
            raise AbiError("INPUT_SHA_MISMATCH",
                           "%s: tensors.json says %s for %s, file is %s"
                           % (key, digest, kind, out["files"][kind]["sha256"]))
    x = np.load(paths["x"])
    w = np.load(paths["w"])
    golden = np.load(paths["golden"])
    device = np.load(paths["device"])
    for name, arr in (("x", x), ("w", w), ("golden", golden), ("device", device)):
        if arr.dtype != np.float32:
            raise AbiError("INPUT_NOT_BF16",
                           "%s.%s.npy dtype is %s; the runner ABI stores bf16 values as "
                           "float32" % (key, name, arr.dtype))
    if x.ndim != 2 or tuple(x.shape) != (rows, width):
        raise AbiError("X_SHAPE_MISMATCH",
                       "%s: X is %r, tensors.json shape is %r" % (key, x.shape, (rows, width)))
    if w.ndim == 2:
        # The exact pre-03-21 defect: a row-wise w. It is NOT the runner's W[D] and must
        # never be silently broadcast or reshaped into one.
        raise AbiError("ROW_WISE_W",
                       "%s: W has shape %r (row-wise). The runner ABI is W[D] = %r; a "
                       "row-wise weight is a DIFFERENT input, not a same-input comparison"
                       % (key, w.shape, (width,)))
    if tuple(w.shape) != (width,):
        raise AbiError("W_SHAPE_MISMATCH",
                       "%s: W is %r, expected W[D] = %r" % (key, w.shape, (width,)))
    for name, arr in (("golden", golden), ("device", device)):
        if tuple(arr.shape) != (rows, width):
            raise AbiError("SHAPE_DISAGREEMENT",
                           "%s: %s is %r, X is %r" % (key, name, arr.shape, x.shape))
    for name, arr in (("x", x), ("w", w), ("golden", golden), ("device", device)):
        finite = np.isfinite(arr).all()
        if not finite:
            raise AbiError("INPUT_NONFINITE", "%s: %s carries non-finite values" % (key, name))
        if not np.array_equal(bf16_to_f32(bf16_bits_of_f32(arr)), arr):
            # Truncating a non-bf16 float32 into bf16 bits would smuggle a tampered
            # device/golden tensor past the bit checks, so this is a named refusal.
            code = ("INPUT_NOT_BF16" if name in ("x", "w")
                    else "GOLDEN_NOT_BF16" if name == "golden" else "DEVICE_NOT_BF16")
            raise AbiError(code,
                           "%s: %s carries values that are not exactly representable in "
                           "bf16; the runner ABI stores bf16 values as float32" % (key, name))
    xb = bf16_bits_of_f32(x)
    wb = bf16_bits_of_f32(w)
    h = hashlib.sha256()
    h.update(("X[%d,%d]" % (rows, width)).encode())
    h.update(np.ascontiguousarray(xb).tobytes())
    h.update(("W[%d]" % width).encode())
    h.update(np.ascontiguousarray(wb).tobytes())
    out["input_digest"] = h.hexdigest()
    out["xb"] = xb
    out["wb"] = wb
    out["golden_bits"] = bf16_bits_of_f32(golden)
    out["device_bits"] = bf16_bits_of_f32(device)
    out["x_f32"] = x
    out["w_f32"] = w
    return out


def identity_recompute(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    """Recompute the runner's own FP64 reference from the persisted X/W (bit-exact audit).

    Uses the runner's formula on CPU. This is an INPUT-IDENTITY check, not the independent
    numeric reference: it asks 'are these the inputs whose golden was stored', and it is
    allowed to share the runner's formula for exactly that reason.
    """
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        raise AbiError("INPUT_IDENTITY_UNCHECKED",
                       "torch is not importable (%s); the stored golden cannot be "
                       "reproduced from the stored X/W, so no same-input claim is made"
                       % (type(exc).__name__,)) from exc
    rows, width = x.shape
    out = np.empty((rows, width), dtype=np.uint16)
    with torch.no_grad():
        for r0 in range(0, rows, BLOCK):
            r1 = min(rows, r0 + BLOCK)
            xb = torch.from_numpy(np.ascontiguousarray(x[r0:r1])).to(torch.bfloat16)
            wb = torch.from_numpy(np.ascontiguousarray(w)).to(torch.bfloat16)
            xd = xb.to(torch.float64)
            wd = wb.to(torch.float64)
            exact = xd * torch.rsqrt((xd * xd).mean(dim=-1, keepdim=True) + eps) * wd
            refb = exact.to(torch.bfloat16).to(torch.float32).numpy()
            out[r0:r1] = bf16_bits_of_f32(refb)
    return out


def model_reference(xb: np.ndarray, wb: np.ndarray, chains: int, width: int,
                    eps: float, block: int = BLOCK) -> tuple[np.ndarray, dict]:
    """The v33-semantics numeric model over the ACTUAL input bits.

    FP32 after BF16->FP32; per-lane chunk accumulation (chunk c to chain c % chains);
    balanced merge; then the lane reduction, whose native order is UNRESOLVED and is a
    pairwise-halving PLACEHOLDER here. Returns bf16 bits and the exact fp64 expression
    (pre-bf16-rounding) for the max_abs-to-unrounded statistic.
    """
    rows = xb.shape[0]
    bits = np.empty((rows, width), dtype=np.uint16)
    exact = np.empty((rows, width), dtype=np.float64)
    nch = width // CH
    if nch * CH != width:
        raise AbiError("X_SHAPE_MISMATCH", "D=%d is not a multiple of the %d-lane chunk"
                       % (width, CH))
    wf = bf16_to_f32(wb)
    for r0 in range(0, rows, block):
        r1 = min(rows, r0 + block)
        n = r1 - r0
        xf = bf16_to_f32(xb[r0:r1])
        acc = [np.zeros((n, CH), dtype=np.float32) for _ in range(chains)]
        for c in range(nch):
            v = xf[:, c * CH:(c + 1) * CH]
            j = c % chains
            acc[j] = (acc[j] + (v * v)).astype(np.float32)
        if chains == 4:
            merged = ((acc[0] + acc[1]) + (acc[2] + acc[3])).astype(np.float32)
        elif chains == 2:
            merged = (acc[0] + acc[1]).astype(np.float32)
        else:
            raise AbiError("CHAINS_UNKNOWN", "unsupported chain count %r" % (chains,))
        while merged.shape[1] > 1:   # PLACEHOLDER lane reduction (order unresolved)
            merged = (merged[:, 0::2] + merged[:, 1::2]).astype(np.float32)
        total = merged[:, 0]
        mean = (total * np.float32(1.0 / width)).astype(np.float32)
        scale = (np.float32(1.0) / np.sqrt((mean + np.float32(eps)).astype(np.float32))).astype(np.float32)
        y = ((xf * scale[:, None]).astype(np.float32) * wf).astype(np.float32)
        bits[r0:r1] = bf16_from_f64(y.astype(np.float64))
        exact[r0:r1] = y.astype(np.float64)
    return bits, {"y_fp32": exact}


def fp64_reference(xb: np.ndarray, wb: np.ndarray, width: int, eps: float,
                   block: int = BLOCK) -> tuple[np.ndarray, np.ndarray]:
    """Independent reference: FP64 mean/rsqrt/product, ONE RNE to bf16.

    Numpy only. Deliberately NOT the v33-semantics chain model; the frozen protocol
    tolerance is applied against this, exactly like the runner's own FP64 golden.
    """
    rows = xb.shape[0]
    bits = np.empty((rows, width), dtype=np.uint16)
    exact = np.empty((rows, width), dtype=np.float64)
    wf = bf16_to_f32(wb).astype(np.float64)
    for r0 in range(0, rows, block):
        r1 = min(rows, r0 + block)
        xf = bf16_to_f32(xb[r0:r1]).astype(np.float64)
        mean = np.sum(xf * xf, axis=-1, keepdims=True) / width
        scale = 1.0 / np.sqrt(mean + eps)
        y = xf * scale * wf
        exact[r0:r1] = y
        bits[r0:r1] = bf16_from_f64(y)
    return bits, exact


def protocol_compare(device_bits: np.ndarray, ref_bits: np.ndarray, atol: float,
                     rtol: float) -> dict:
    got = bf16_to_f32(device_bits).astype(np.float64)
    ref = bf16_to_f32(ref_bits).astype(np.float64)
    finite = bool(np.isfinite(got).all())
    if not finite:
        return {"passed": False, "finite": False, "mismatch_count": int((~np.isfinite(got)).sum()),
                "max_abs": None, "max_rel": None,
                "pass_rule": "non-finite device output cannot pass"}
    diff = np.abs(got - ref)
    tol = atol + rtol * np.abs(ref)
    mism = int((diff > tol).sum())
    max_abs = float(diff.max()) if diff.size else 0.0
    denom = np.maximum(np.abs(ref), 1e-12)
    return {"passed": mism == 0, "finite": True, "mismatch_count": mism,
            "max_abs": max_abs, "max_rel": float((diff / denom).max()) if diff.size else 0.0,
            "max_abs_le_atol": bool(max_abs <= atol),
            "atol": atol, "rtol": rtol,
            "pass_rule": "per element |device-ref| <= atol + rtol*|ref|; mismatch_count must be 0"}


def model_diag(device_bits: np.ndarray, model_bits: np.ndarray) -> dict:
    diff_elems = int((device_bits != model_bits).sum())
    ulp = np.abs(bf16_to_f32(device_bits).astype(np.float64)
                 - bf16_to_f32(model_bits).astype(np.float64))
    idx = np.argwhere(device_bits != model_bits)
    first = None
    if idx.size:
        r, c = int(idx[0][0]), int(idx[0][1])
        first = {"row": r, "col": c,
                 "device_bits": "0x%04X" % int(device_bits[r, c]),
                 "model_bits": "0x%04X" % int(model_bits[r, c])}
    return {"bit_mismatch_count": diff_elems, "bit_mismatch_fraction": diff_elems / device_bits.size,
            "max_abs": float(ulp.max()) if ulp.size else 0.0, "first_divergence": first,
            "is_pass_criterion": False,
            "note": "diagnostic only: the lane-reduction order of the model is a PLACEHOLDER"}


def judge_pattern(rec: dict, chains: int, width: int, protocol: dict) -> dict:
    atol = float(protocol["atol"]); rtol = float(protocol["rtol"])
    eps = float(protocol.get("eps", EPS))
    identity_bits = identity_recompute(rec["x_f32"], rec["w_f32"], eps)
    identity_ok = bool(np.array_equal(identity_bits, rec["golden_bits"]))
    ref_bits, ref_exact = fp64_reference(rec["xb"], rec["wb"], width, eps)
    model_bits, model_exact = model_reference(rec["xb"], rec["wb"], chains, width, eps)
    proto = protocol_compare(rec["device_bits"], ref_bits, atol, rtol)
    det_required = rec["pattern"] in DETERMINISTIC_PATTERNS
    det_exact = bool(np.array_equal(rec["device_bits"], ref_bits)) if det_required else None
    got = bf16_to_f32(rec["device_bits"]).astype(np.float64)
    to_unrounded = float(np.abs(got - ref_exact).max()) if got.size else 0.0
    result = {
        "key": rec["key"], "pattern": rec["pattern"], "seed": rec["seed"],
        "input_digest": rec["input_digest"],
        "files": rec["files"],
        "input_identity": {"golden_reproduced_bit_exact": identity_ok,
                           "checked_by": "runner FP64 formula recomputed from persisted X/W"},
        "protocol": proto,
        "deterministic": {"required": det_required, "bits_exact_vs_independent_ref": det_exact},
        "max_abs_to_unrounded_fp64": to_unrounded,
        "model_diag": model_diag(rec["device_bits"], model_bits),
    }
    failures = []
    if not identity_ok:
        failures.append("INPUT_IDENTITY_FAILED")
    if not proto["passed"]:
        failures.append("DEVICE_PROTOCOL_FAILED")
    if det_required and not det_exact:
        failures.append("DETERMINISTIC_BITS_FAILED")
    result["failures"] = failures
    result["passed"] = not failures
    return result


def consume_run_dir(run_dir: Path, protocol: dict) -> dict:
    if not run_dir.is_dir():
        raise AbiError("RUN_DIR_MISSING", "%s is not a directory" % run_dir)
    tj_path = run_dir / "tensors.json"
    _require_file(tj_path, "TENSORS_JSON_MISSING")
    try:
        tj = json.loads(tj_path.read_text())
    except json.JSONDecodeError as exc:
        raise AbiError("TENSORS_JSON_INVALID", "%s: %s" % (tj_path, exc)) from exc
    if "shape" not in tj or "tensors" not in tj:
        raise AbiError("TENSORS_JSON_INVALID", "%s lacks shape/tensors" % tj_path)
    shape = tuple(int(v) for v in tj["shape"])
    if len(shape) != 2:
        raise AbiError("TENSORS_JSON_INVALID", "shape %r is not [M,D]" % (shape,))
    entries = tj["tensors"]
    if not entries:
        # The all(empty)==True trap: an empty corpus is a refusal, not a pass.
        raise AbiError("EMPTY_TENSOR_SET", "%s lists zero tensors" % tj_path)
    case_id = str(tj.get("case_id", run_dir.name))
    chains = chains_for_case(case_id)
    run = {"run_dir": str(run_dir), "case_id": case_id, "shape": list(shape),
           "chains": chains, "tensors_json": {"sha256": sha256_file(tj_path),
                                              "size": tj_path.stat().st_size},
           "patterns": [], "failures": []}
    di_path = run_dir / "device_info.json"
    if di_path.is_file():
        run["device_info"] = {"sha256": sha256_file(di_path),
                              "kernel_source_sha256": json.loads(di_path.read_text()).get(
                                  "kernel_source_sha256")}
    for entry in entries:
        rec = read_inputs(run_dir, entry, (shape[0], shape[1]))
        run["patterns"].append(judge_pattern(rec, chains, shape[1], protocol))
    for pat in run["patterns"]:
        for code in pat["failures"]:
            run["failures"].append("%s/%s: %s" % (pat["key"], pat["pattern"], code))
    run["passed"] = not run["failures"]
    return run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", action="append", default=[],
                    help="runner case directory with tensors.json + <key>.{x,w,golden,device}.npy")
    ap.add_argument("--out", help="directory for MANIFEST.json")
    ap.add_argument("--protocol", default=str(DEFAULT_PROTOCOL))
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--scalar-demo", action="store_true")
    args = ap.parse_args()

    if args.self_check:
        from fractions import Fraction
        import random
        rng = random.Random(7)
        bad = 0
        for _ in range(5000):
            v = np.frombuffer(rng.getrandbits(64).to_bytes(8, "little"), dtype=np.float64)[0]
            got = int(bf16_from_f64(np.array([v]))[0])
            if math.isnan(v):
                if (got & 0x7F80) != 0x7F80 or (got & 0x7F) == 0:
                    bad += 1
                continue
            if math.isinf(v):
                if (got & 0x7F80) != 0x7F80 or (got & 0x7F) != 0:
                    bad += 1
                continue
            if got != p2.fraction_to_bf16_bits(Fraction(v)):
                bad += 1
        print("self-check bad:", bad)
        return 0 if bad == 0 else 1
    if args.scalar_demo:
        from fractions import Fraction
        scalar = float.fromhex("0x1.02fa0be82fa0cp+0")
        lane = p2.bf16_bits_to_fraction(0x3f81)
        exact_bits = p2.fraction_to_bf16_bits(lane * Fraction(scalar))
        p32 = np.float32(np.float32(float(lane)) * np.float32(scalar))
        f32_bits = p2.fraction_to_bf16_bits(Fraction(float(p32)))
        print("scalar exact:", repr(scalar))
        print("exact-scalar bf16:    0x%04X" % exact_bits)
        print("f32-rounded-scalar bf16: 0x%04X" % f32_bits)
        return 0 if (exact_bits, f32_bits) == (0x3F83, 0x3F82) else 1

    if not args.run_dir:
        ap.error("--run-dir is required: this tool never generates its own inputs")
    if not args.out:
        ap.error("--out is required")

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "tool": {"path": str(Path(__file__).resolve()),
                 "sha256": sha256_file(Path(__file__).resolve())},
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "abi": {"name": "runner-tensors-v1",
                "inputs": "runner-persisted X[M,D], W[D], golden.npy, device.npy",
                "self_generated_inputs": False},
        "model": {"chunk": CH, "lane_reduction": "PLACEHOLDER pairwise-halving (UNRESOLVED)",
                  "vmula": "mul then add, two FP32 roundings (ASSUMED)",
                  "scalar_semantics": "FLOAT32-ROUNDED scalar (v33 scale is FP32)",
                  "accumulation": "FP32 after BF16->FP32",
                  "merge": {"4": "((a0+a1)+(a2+a3))", "2": "(a0+a1)"},
                  "chains_per_case": "from case_id (prefill=4, decode=2); never inferred"},
        "evidence_scope": ("recomputation over runner-persisted inputs; the device output is "
                           "the stored one and this tool makes no device observation"),
        "runs": [], "failures": [], "verdict": None,
    }
    code = 0
    try:
        protocol_path = Path(args.protocol)
        protocol = load_protocol(protocol_path)
        manifest["protocol"] = {
            "path": str(protocol_path.resolve()), "sha256": sha256_file(protocol_path),
            "status": protocol.get("status"), "protocol_id": protocol.get("protocol_id"),
            "atol": protocol.get("atol"), "rtol": protocol.get("rtol"),
            "eps": protocol.get("eps"), "seeds": protocol.get("seeds"),
        }
        for rd in args.run_dir:
            run = consume_run_dir(Path(rd), protocol)
            manifest["runs"].append(run)
            for line in run["failures"]:
                manifest["failures"].append("%s: %s" % (run["case_id"], line))
    except AbiError as exc:
        manifest["failures"].append(str(exc))
        manifest["verdict"] = "REFUSED"
        (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
        print("REFUSED %s" % exc, file=sys.stderr)
        return 2
    manifest["verdict"] = "PASS" if not manifest["failures"] else "FAIL"
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    for run in manifest["runs"]:
        print("%-22s %-14s chains=%d patterns=%d %s"
              % (run["case_id"], "%dx%d" % tuple(run["shape"]), run["chains"],
                 len(run["patterns"]), "PASS" if run["passed"] else "FAIL"))
        for pat in run["patterns"]:
            proto = pat["protocol"]
            print("  %-12s id=%-8s proto_mismatch=%-5s max_abs=%-10s det=%-5s model_bits_diff=%d (diag)"
                  % (pat["key"], "OK" if pat["input_identity"]["golden_reproduced_bit_exact"]
                     else "FAILED", proto["mismatch_count"],
                     "%.6g" % proto["max_abs"] if proto["max_abs"] is not None else None,
                     pat["deterministic"]["bits_exact_vs_independent_ref"],
                     pat["model_diag"]["bit_mismatch_count"]))
    for line in manifest["failures"]:
        print("FAILURE:", line, file=sys.stderr)
    print("verdict:", manifest["verdict"], "manifest:", out / "MANIFEST.json")
    if manifest["verdict"] != "PASS":
        code = 1
    return code


if __name__ == "__main__":
    raise SystemExit(main())
