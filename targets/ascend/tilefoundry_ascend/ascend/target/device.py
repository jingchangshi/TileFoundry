"""Ascend 950PR / 950DT device resource model and runtime core detection."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass

from tilefoundry.ir.types import DType
from tilefoundry.target.base import Device

_AICORE_COUNT_RE = re.compile(r"Aicore\s+Count\s*:\s*(\d+)", re.IGNORECASE)


def detect_runtime_core_counts(default_aic: int = 28) -> tuple[int, int]:
    """Query live Ascend NPU `(aic_count, aiv_count)` safely, falling back offline.

    Checks ``torch.npu`` (unless ``TORCH_DEVICE_BACKEND_AUTOLOAD=0``) and
    ``npu-smi info -t common -i 0`` without raising when NPU drivers or tools
    are absent.
    """
    fallback_aic = max(1, int(default_aic))
    if os.environ.get("TORCH_DEVICE_BACKEND_AUTOLOAD") != "0":
        try:
            import torch  # noqa: PLC0415

            if hasattr(torch, "npu") and torch.npu.is_available():
                props = torch.npu.get_device_properties(0)
                for attr in ("aic_num", "cube_core_num", "core_num"):
                    val = getattr(props, attr, None)
                    if isinstance(val, int) and val > 0:
                        aiv_val = getattr(props, "aiv_num", None) or getattr(
                            props, "vector_core_num", None
                        )
                        aiv = int(aiv_val) if isinstance(aiv_val, int) and aiv_val > 0 else val * 2
                        return (val, aiv)
        except Exception:
            pass

    npu_smi = shutil.which("npu-smi")
    if npu_smi:
        try:
            proc = subprocess.run(
                [npu_smi, "info", "-t", "common", "-i", "0"],
                capture_output=True,
                text=True,
                timeout=2.0,
                check=False,
            )
            if proc.returncode == 0:
                match = _AICORE_COUNT_RE.search(proc.stdout)
                if match:
                    aic = int(match.group(1))
                    if aic > 0:
                        return (aic, aic * 2)
        except Exception:
            pass

    return (fallback_aic, fallback_aic * 2)


@dataclass(frozen=True)
class Ascend950PR(Device):
    """One Ascend 950 package (950PR / 950DT): coupled blocks, scratchpads, HBM, and rates."""

    name: str
    sm_count: int
    aic_count: int
    aiv_count: int
    hbm_capacity_bytes: int
    hbm_owner: str
    hbm_bandwidth_bytes_per_second: int
    hbm_burst_bandwidth_table: tuple[tuple[int, int], ...]
    l1_bytes_per_aic: int
    l0a_bytes_per_aic: int
    l0b_bytes_per_aic: int
    l0c_bytes_per_aic: int
    ub_nominal_bytes_per_aiv: int
    ub_simdvf_usable_bytes_per_aiv: int
    ub_usable_bytes_per_aiv: int
    rmem_bytes_per_lane: int
    _unit_flops: tuple[tuple[str, tuple[tuple[DType, int], ...]], ...]
    aiv_lane_events_per_second: int
    ssbuf_bytes_per_block: int
    ssbuf_latency_cycles: int
    clock_mhz: int
    l2_capacity_bytes: int

    @property
    def clock_ghz(self) -> float:
        """Return nominal AI Core clock frequency in GHz."""
        return round(self.clock_mhz / 1000.0, 6)

    @property
    def unit_flops_per_second(self) -> dict[str, dict[DType, int]]:
        """Return nominal compute throughput map per execution unit."""
        return {unit: dict(entries) for unit, entries in self._unit_flops}


__all__ = ["Ascend950PR", "detect_runtime_core_counts"]
