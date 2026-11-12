#!/usr/bin/env python3
"""Materialise the recorded actual-input fixtures and the T0-T4 positive evidence.

The recorded .bin files ARE the actual input (role 1): consumers load them and never
regenerate from pattern/seed. This producer is run once per fixture set; its output
is committed so the bytes are stable and reviewable.

    python fixtures.py --bundles tools/twin/fixtures/bundles.json \
        --storage tools/twin/fixtures/actual \
        --evidence-dir <pack>/artifacts/run03/w40_twin
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

import reference as R
import twin_io as T

HERE = Path(__file__).resolve().parent


def _bundle(storage, case_id, input_id, tensors, eps=1e-06, pattern=None, seed=None,
            run_id="w40-twin-fixtures"):
    # Each bundle gets its OWN directory: different patterns reuse the tensor names
    # x/w, and a flat directory would let a later pattern overwrite an earlier
    # bundle's bytes (measured: INPUT_DIGEST_MISMATCH on load).
    case_dir = Path(storage) / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    bundle = T.capture_actual_inputs(
        tensors, eps, {"case_id": case_id, "input_id": input_id},
        {"run_id": run_id}, case_dir, pattern=pattern, seed=seed,
        source_manifest_digest="frozen-bf16-rmsnorm-frozen-v1")
    bundle["storage_dir"] = case_dir.name
    return bundle


def build(storage: Path, bundles_path: Path):
    storage = Path(storage)
    if storage.exists():
        shutil.rmtree(storage)  # a fixture set is rebuilt from scratch, never merged
    storage.mkdir(parents=True, exist_ok=True)
    bundles = {"schema": T.SCHEMA, "kind": "BundleSet", "bundles": {}}

    # T-FIX-VADD-001: f32 vadd-one, recorded input bytes.
    x = (np.arange(R.VADD_N, dtype=np.float32) + np.float32(0.5)).astype(np.float32)
    bundles["bundles"]["T-FIX-VADD-001"] = _bundle(
        storage, "vadd_one_64", "T-FIX-VADD-001",
        [{"name": "x", "values": x, "dtype": "float32", "shape": [R.VADD_N]}],
        pattern="deterministic_arange_plus_half", seed=None)

    # T-FIX-RMS-ROW-xxx: single-row RMSNorm, recorded BF16 bytes for each pattern.
    for order, (pattern, seed) in enumerate(R.PATTERNS, start=1):
        x_bf, w_bf = R.make_input(R.RMS_ROW_SHAPE, pattern, seed)
        fixture_id = "T-FIX-RMS-ROW-%03d" % order
        bundles["bundles"][fixture_id] = _bundle(
            storage, "rmsnorm_row_%s_s%d" % (pattern, seed), fixture_id,
            [{"name": "x", "values": R.bf16_bits_of_tensor(x_bf), "dtype": "bfloat16",
              "shape": list(R.RMS_ROW_SHAPE)},
             {"name": "w", "values": R.bf16_bits_of_tensor(w_bf), "dtype": "bfloat16",
              "shape": [R.RMS_ROW_SHAPE[1]]}],
            pattern=pattern, seed=seed)

    Path(bundles_path).parent.mkdir(parents=True, exist_ok=True)
    Path(bundles_path).write_text(json.dumps(bundles, indent=1, sort_keys=True) + "\n")
    return bundles


def run_evidence(bundles, storage, evidence_dir: Path):
    evidence_dir = Path(evidence_dir)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir = evidence_dir / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    summary = {"schema": T.SCHEMA, "kind": "TwinT0T4Evidence",
               "protocol": R.PROTOCOL, "cases": [], "status": "PASS"}
    expected_eps_bits = T.eps_f32_bits(R.EPS)
    for fixture_id, bundle in bundles["bundles"].items():
        operation = "vadd_one" if "VADD" in fixture_id else "rmsnorm"
        base = Path(storage) / bundle.get("storage_dir", "")
        loaded = T.load_actual_inputs(
            bundle, {"case_id": bundle["case_id"], "input_id": bundle["input_id"],
                     "eps_bits": expected_eps_bits}, base_dir=base)
        program_ref = {"entry": "fixture." + operation, "operation": operation,
                       "semantic_scope": "fixture"}
        reference_out = R.compute_reference(loaded, operation)
        algo_out = R.replay_logical(program_ref, loaded)
        dtype = reference_out["dtype"]
        report = T.compare_outputs(
            algo_out["values"], reference_out["values"], R.PROTOCOL, dtype=dtype,
            got_ref={"execution_kind": "TF_ALGO_F32_EXPLICIT_CAST"},
            reference_ref={"execution_kind": "REFERENCE_FP64_ONE_BF16_ROUND"})
        # Persist raw outputs so a later comparison does not rely on summaries.
        got_file = outputs_dir / (fixture_id + ".got." + ("bf16" if dtype == "bfloat16" else "f32") + ".bin")
        ref_file = outputs_dir / (fixture_id + ".expected." + ("bf16" if dtype == "bfloat16" else "f32") + ".bin")
        got_file.write_bytes(T._tensor_bytes(algo_out["values"], dtype))
        ref_file.write_bytes(T._tensor_bytes(reference_out["values"], dtype))
        if report["status"] != "PASS":
            summary["status"] = "FAIL"
        summary["cases"].append({
            "fixture_id": fixture_id, "case_id": bundle["case_id"],
            "operation": operation,
            "dtype": dtype,
            "input": {rec["name"]: rec["sha256"] for rec in bundle["tensors"]},
            "eps_bits": bundle["eps"]["bits"], "eps_hex": bundle["eps"]["hex"],
            "got_sha256": T.sha256_bytes(got_file.read_bytes()),
            "expected_sha256": T.sha256_bytes(ref_file.read_bytes()),
            "got_path": str(got_file), "expected_path": str(ref_file),
            "comparison": report,
        })
    T.write_json(evidence_dir / "twin_t0_t4_evidence.json", summary)
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundles", default=str(HERE / "fixtures" / "bundles.json"))
    parser.add_argument("--storage", default=str(HERE / "fixtures" / "actual"))
    parser.add_argument("--evidence-dir", default=None)
    args = parser.parse_args(argv)
    bundles = build(Path(args.storage), Path(args.bundles))
    print("bundles:", len(bundles["bundles"]))
    if args.evidence_dir:
        summary = run_evidence(bundles, Path(args.storage), Path(args.evidence_dir))
        print("evidence status:", summary["status"])
        if summary["status"] != "PASS":
            return 1
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
