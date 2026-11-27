"""Lease-locked AgentGates L2 surrogate CANNBench batch evaluator on NPU."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def test_surrogate_anchor_batch_npu() -> None:
    """Evaluate configured or baseline 10-anchor surrogate batch inside AgentGates lease."""
    harness_path = Path(__file__).resolve().with_name("surrogate_batch_harness.py")
    spec = importlib.util.spec_from_file_location("surrogate_batch_harness", harness_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load surrogate harness from {harness_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    mod.run_surrogate_anchor_batch_npu()
