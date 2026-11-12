"""twin-io/1 — the schema, producer/consumer ABI and comparator for the T module.

Approved design: artifacts/run02/review/architect_t_twin_reference_module_design_v1.md
(SHA 4ecbbf22..., Lead-approved minimal schema). This file implements only the
design-independent CPU half: recorded actual-input bundles, output bundles, and the
elementwise comparison report. It touches no device and no compiler interface.

Three roles are kept apart ON PURPOSE (see reference.py):
  1. ACTUAL INPUT  — read from the recorded BF16 bytes + eps f32 bits. The loader
     NEVER regenerates inputs from pattern/seed; "pattern" is a note, not a source.
  2. TF ALGORITHM SIDE — f32 arithmetic with an EXPLICIT BF16 cast at the program's
     conversion point (reference.py: tf_algo_*).
  3. INDEPENDENT REFERENCE — FP64 arithmetic, then ONE BF16 rounding (reference.py:
     reference_*). A f64->f32->bf16 double rounding is NOT this role and is rejected.

Errors are objects, not prose: TwinError(code, phase, source_id, expected, actual).
"""
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path

import numpy as np

SCHEMA = "twin-io/1"
BYTE_ORDER = "little"


class TwinError(RuntimeError):
    def __init__(self, code: str, phase: str, source_id: str = "",
                 expected=None, actual=None):
        self.code = code
        self.phase = phase
        self.source_id = source_id
        self.expected = expected
        self.actual = actual
        super().__init__("%s [%s] %s (expected=%r actual=%r)"
                         % (code, phase, source_id, expected, actual))

    def as_dict(self) -> dict:
        return {"code": self.code, "phase": self.phase, "source_id": self.source_id,
                "expected": self.expected, "actual": self.actual}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def _tensor_bytes(values, dtype: str) -> bytes:
    """Raw little-endian bytes for supported dtypes."""
    if dtype == "bfloat16":
        arr = np.ascontiguousarray(values, dtype=np.uint16)  # already bf16 bits
        return arr.tobytes()
    if dtype == "float32":
        arr = np.ascontiguousarray(values, dtype=np.float32)
        return arr.tobytes()
    raise TwinError("INPUT_ABI_MISMATCH", "capture", dtype, "bfloat16|float32", dtype)


def eps_f32_bits(eps) -> int:
    """The exact f32 bit pattern of eps (the ABI stores bits, not a JSON decimal)."""
    return int(np.float32(eps).view(np.uint32))


def eps_bits_to_f32(bits: int) -> np.float32:
    return np.frombuffer(struct.pack("<I", int(bits) & 0xFFFFFFFF), dtype=np.float32)[0]


# ------------------------------------------------------------------- capture
def capture_actual_inputs(tensors, eps_f32, case_ref, run_ref, storage_dir,
                          pattern=None, seed=None, source_manifest_digest=None):
    """Record the EXACT bytes that were fed to a runner, plus eps as f32 bits.

    tensors: [{"name": str, "values": np.ndarray, "dtype": "bfloat16"|"float32",
               "shape": [..]}]
    The bundle, not the seed, is the authoritative source; pattern/seed are notes.
    """
    storage = Path(storage_dir)
    storage.mkdir(parents=True, exist_ok=True)
    records = []
    for tensor in tensors:
        name = tensor["name"]
        dtype = tensor.get("dtype", "bfloat16")
        shape = [int(x) for x in (tensor.get("shape") or np.asarray(tensor["values"]).shape)]
        payload = _tensor_bytes(tensor["values"], dtype)
        filename = "%s.%s.bin" % (name, "bf16" if dtype == "bfloat16" else "f32")
        path = storage / filename
        path.write_bytes(payload)
        records.append({
            "name": name, "shape": shape, "dtype": dtype, "byte_order": BYTE_ORDER,
            "logical_strides": ([int(x) for x in np.cumprod([1] + shape[-1:0:-1])[::-1]]
                                if shape else []),
            "storage_path": filename, "nbytes": int(len(payload)),
            "sha256": sha256_bytes(payload),
        })
    return {
        "schema": SCHEMA, "kind": "ActualInputBundle",
        "run_id": run_ref.get("run_id"), "case_id": case_ref.get("case_id"),
        "input_id": case_ref.get("input_id", case_ref.get("case_id")),
        "tensors": records,
        "eps": {"dtype": "f32", "bits": eps_f32_bits(eps_f32),
                "hex": "0x%08x" % (eps_f32_bits(eps_f32) & 0xFFFFFFFF),
                "nominal": float(eps_f32)},
        "pattern": pattern, "seed": seed,
        "source_manifest_digest": source_manifest_digest,
    }


def load_actual_inputs(bundle, expected_refs=None, base_dir="."):
    """Load the recorded bytes and validate the ABI. Never regenerates from seed."""
    expected_refs = expected_refs or {}
    if bundle.get("schema") != SCHEMA or bundle.get("kind") != "ActualInputBundle":
        raise TwinError("INPUT_ABI_MISMATCH", "load", bundle.get("kind"),
                        SCHEMA, bundle.get("schema"))
    for key in ("run_id", "case_id", "input_id"):
        if key in expected_refs and expected_refs[key] != bundle.get(key):
            raise TwinError("INPUT_ABI_MISMATCH", "load", key,
                            expected_refs[key], bundle.get(key))
    if ("source_manifest_digest" in expected_refs
            and expected_refs["source_manifest_digest"] != bundle.get("source_manifest_digest")):
        raise TwinError("INPUT_DIGEST_MISMATCH", "load", "source_manifest_digest",
                        expected_refs["source_manifest_digest"],
                        bundle.get("source_manifest_digest"))
    if "eps_bits" in expected_refs and int(expected_refs["eps_bits"]) != int(bundle["eps"]["bits"]):
        raise TwinError("EPS_BITS_MISMATCH", "load", "eps.bits",
                        expected_refs["eps_bits"], bundle["eps"]["bits"])

    arrays = {}
    base = Path(base_dir)
    for record in bundle.get("tensors", []):
        path = base / record["storage_path"]
        if not path.is_file():
            raise TwinError("INPUT_MISSING", "load", record["name"],
                            record["storage_path"], "absent")
        payload = path.read_bytes()
        if len(payload) != int(record["nbytes"]):
            raise TwinError("INPUT_ABI_MISMATCH", "load", record["name"],
                            record["nbytes"], len(payload))
        if sha256_bytes(payload) != record["sha256"]:
            raise TwinError("INPUT_DIGEST_MISMATCH", "load", record["name"],
                            record["sha256"], sha256_bytes(payload))
        if record["dtype"] == "bfloat16":
            values = np.frombuffer(payload, dtype="<u2").reshape(record["shape"])
        elif record["dtype"] == "float32":
            values = np.frombuffer(payload, dtype="<f4").reshape(record["shape"])
        else:
            raise TwinError("INPUT_ABI_MISMATCH", "load", record["name"],
                            "bfloat16|float32", record["dtype"])
        if record["byte_order"] != BYTE_ORDER:
            raise TwinError("INPUT_ABI_MISMATCH", "load", record["name"],
                            BYTE_ORDER, record["byte_order"])
        arrays[record["name"]] = values
    return {"arrays": arrays, "eps_f32": eps_bits_to_f32(bundle["eps"]["bits"]),
            "bundle": bundle}


# ------------------------------------------------------------------ outputs
def output_bundle(got, dtype, *, run_ref=None, input_ref=None, program_ref=None,
                  protocol_ref=None, execution_kind="CPU_TWIN", coverage="FULL"):
    values = np.ascontiguousarray(got)
    payload = _tensor_bytes(values, dtype)
    return {
        "schema": SCHEMA, "kind": "OutputBundle",
        "run_id": (run_ref or {}).get("run_id"),
        "input_id": (input_ref or {}).get("input_id"),
        "program_id": (program_ref or {}).get("entry"),
        "plan_mode": (protocol_ref or {}).get("plan_mode", "LOGICAL_ONLY"),
        "dtype": dtype, "shape": list(values.shape),
        "nbytes": len(payload), "sha256": sha256_bytes(payload),
        "execution_kind": execution_kind, "coverage": coverage,
    }


def _as_f32_values(values, dtype):
    if dtype == "bfloat16":
        bits = np.ascontiguousarray(values, dtype=np.uint16)
        return (bits.astype(np.uint32) << 16).view(np.float32)
    return np.ascontiguousarray(values, dtype=np.float32)


def compare_outputs(got, reference, protocol, *, dtype="bfloat16", shape=None,
                    got_ref=None, reference_ref=None):
    """Apply the frozen predicate elementwise and return a ReplayReport.

    got/reference are arrays in *dtype* (bfloat16 values as uint16 bits). A missing,
    short or differently shaped output is an ABI failure, not a numeric pass. Any
    non-finite value is reported (NONFINITE_OUTPUT) and cannot be swept under the
    tolerance.
    """
    g = _as_f32_values(got, dtype)
    r = _as_f32_values(reference, dtype)
    report = {
        "schema": SCHEMA, "kind": "ReplayReport", "status": "FAIL", "code": None,
        "dtype": dtype, "n_elements": 0, "valid_elements": 0,
        "mismatch_count": 0, "mismatch_coords": [],
        "nonfinite_count": 0, "max_abs": None, "max_rel": None,
        "atol": protocol.get("atol"), "rtol": protocol.get("rtol"),
        "elementwise_rule": "|got-ref| <= atol + rtol*|ref|",
        "got_ref": got_ref, "reference_ref": reference_ref, "source_scope": "FULL",
    }
    if shape is not None and (list(g.shape) != list(shape) or list(r.shape) != list(shape)):
        report["code"] = "OUTPUT_ABI_MISMATCH"
        report["expected"] = list(shape)
        report["actual"] = [list(g.shape), list(r.shape)]
        return report
    if g.shape != r.shape or g.size == 0:
        report["code"] = "OUTPUT_ABI_MISMATCH"
        report["expected"] = "nonempty equal shapes"
        report["actual"] = [list(g.shape), list(r.shape)]
        return report
    report["n_elements"] = int(r.size)
    nonfinite = int(np.count_nonzero(~np.isfinite(g)) + np.count_nonzero(~np.isfinite(r)))
    report["nonfinite_count"] = nonfinite
    if nonfinite:
        bad = np.argwhere(~np.isfinite(g) | ~np.isfinite(r))
        report["mismatch_coords"] = [list(map(int, c)) for c in bad[:16]]
        report["code"] = "NONFINITE_OUTPUT"
        return report
    diff = np.abs(g - r)
    tol = np.float32(protocol["atol"]) + np.float32(protocol["rtol"]) * np.abs(r)
    bad = np.argwhere(diff > tol)
    # Small fixtures carry the full per-element diff; large ones keep the raw
    # outputs on disk and report the diagnostic statistics (never a summary-only
    # verdict: every element is still compared, only the LIST is size-capped).
    if int(r.size) <= 8192:
        report["element_diffs"] = [float(value) for value in diff.ravel()]
    report["valid_elements"] = int(r.size)
    report["mismatch_count"] = int(bad.shape[0])
    report["max_abs"] = float(diff.max())
    report["max_rel"] = float((diff / np.maximum(np.abs(r), 1e-12)).max())
    if bad.shape[0]:
        report["mismatch_coords"] = [list(map(int, c)) for c in bad[:16]]
        report["code"] = "NUMERIC_MISMATCH"
        return report
    report["status"] = "PASS"
    return report


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    return path
