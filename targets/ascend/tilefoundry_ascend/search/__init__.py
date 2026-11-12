"""Native mainline search primitives.

This package exposes only the bounded subprocess protocol and the device-only
ranking boundary. Removed generator/filter/harness compatibility layers are
not re-exported.
"""
from .process import Deadline, ProcessResult, WorkerCommand, run_process
from .rank import (
    COMPILE_FAILED,
    COMPILE_TIMEOUT,
    LAUNCH_CONTRACT_VIOLATED,
    MEASUREMENT_SOURCE_DEVICE,
    MEASUREMENT_SOURCE_SIMULATED,
    MEASUREMENT_SOURCES,
    NUMERIC_FAILED,
    OK,
    PRUNED_STATIC,
    RUN_ERROR,
    RUN_TIMEOUT,
    STATUSES,
    CandidateMeasurement,
    RankError,
    Ranking,
    confirmation_required,
    rank,
    rankable,
    require_device_source,
    top_k,
)

__all__ = [
    "Deadline",
    "ProcessResult",
    "WorkerCommand",
    "run_process",
    "OK",
    "COMPILE_FAILED",
    "COMPILE_TIMEOUT",
    "NUMERIC_FAILED",
    "LAUNCH_CONTRACT_VIOLATED",
    "RUN_TIMEOUT",
    "RUN_ERROR",
    "PRUNED_STATIC",
    "STATUSES",
    "MEASUREMENT_SOURCE_DEVICE",
    "MEASUREMENT_SOURCE_SIMULATED",
    "MEASUREMENT_SOURCES",
    "RankError",
    "CandidateMeasurement",
    "Ranking",
    "rankable",
    "require_device_source",
    "rank",
    "top_k",
    "confirmation_required",
]
