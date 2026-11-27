"""NPU execution tests for Ascend 950PR calibration micro-probes."""

import json
import os
import sys
from pathlib import Path

import pytest
import torch

from tilefoundry_ascend.agent.probes import (
    run_fixpipe_probe,
    run_hbm_bw_probe,
    run_mte2_queue_probe,
    run_ssbuf_probe,
)


def _require_npu() -> None:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Ascend NPU device is required for calibration micro-probes.")


def test_ascend950_calibration_micro_probes() -> None:
    """Execute all four hardware micro-probes on Ascend 950PR and emit calibration JSON."""
    _require_npu()
    report: dict[str, dict[str, float]] = {}
    for key, fn in (
        ("hbm_bw", run_hbm_bw_probe),
        ("fixpipe_rate", run_fixpipe_probe),
        ("mte2_queue", run_mte2_queue_probe),
        ("ssbuf_latency", run_ssbuf_probe),
    ):
        report[key] = fn()
        sys.__stderr__.write(f"PROBE_{key.upper()}={json.dumps(report[key], sort_keys=True)}\n")
        sys.__stderr__.flush()
    sys.__stderr__.write("CALIBRATION_PROBE_REPORT=" + json.dumps(report, sort_keys=True) + "\n")
    sys.__stderr__.flush()
    report_path = os.environ.get("TILEFOUNDRY_PROBE_REPORT_PATH", "").strip()
    if report_path:
        Path(report_path).write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    assert report["hbm_bw"]["bandwidth_bytes_per_sec"] > 0.0
    assert report["hbm_bw"]["burst_4096_bps"] > 0.0
    assert report["hbm_bw"]["burst_32768_bps"] > report["hbm_bw"]["burst_4096_bps"]
    assert report["fixpipe_rate"]["short_k_dur_us"] > 0.0
    assert report["mte2_queue"]["coalesce_speedup"] > 1.0
    assert report["ssbuf_latency"]["ssbuf_dur_us"] > 0.0
