"""Device-only ranking tests for the repaired search package."""
from __future__ import annotations

import pytest

from tilefoundry_ascend.search import (
    CandidateMeasurement,
    MEASUREMENT_SOURCE_DEVICE,
    MEASUREMENT_SOURCE_SIMULATED,
    RankError,
    confirmation_required,
    rank,
    rankable,
    require_device_source,
    top_k,
)


def _device(key: str, samples, footprint=0, spread_note="device-event"):
    return CandidateMeasurement(
        key=key,
        operator="attention_residual",
        measurement_source=MEASUREMENT_SOURCE_DEVICE,
        samples_us=tuple(samples),
        footprint_bytes=footprint,
        identity_key="binary-" + key,
        sync_note=spread_note,
        run_id="run-" + key,
    )


def _simulated(key: str, samples=(10.0, 10.0)):
    return CandidateMeasurement(
        key=key,
        operator="attention_residual",
        measurement_source=MEASUREMENT_SOURCE_SIMULATED,
        samples_us=tuple(samples),
        footprint_bytes=0,
        run_id="run-" + key,
    )


def test_simulated_is_recorded_but_never_rankable():
    sim = _simulated("sim")
    assert sim.rankable is False
    assert rankable(sim) is False
    with pytest.raises(RankError, match="RANK_SIMULATED_NOT_COMPARABLE"):
        require_device_source(sim)
    result = rank([sim])
    assert result.winner is None
    assert result.rejected == (sim,)
    assert result.comparable is False


def test_rank_prefers_device_and_breaks_ties_by_footprint():
    fast_big = _device("big", [8.0, 8.0], footprint=2048)
    slow_small = _device("small", [10.0, 10.0], footprint=128)
    equal_a = _device("equal-a", [8.0, 8.0], footprint=256)
    equal_b = _device("equal-b", [8.0, 8.0], footprint=512)
    sim = _simulated("sim", [0.1, 0.1])
    result = rank([slow_small, fast_big, equal_b, equal_a, sim])
    assert result.winner is equal_a
    assert result.ordered[:3] == (equal_a, equal_b, fast_big)
    assert sim in result.rejected
    assert top_k([slow_small, fast_big], 1) == (fast_big,)


def test_rank_refuses_malformed_measurements():
    with pytest.raises(RankError, match="RANK_REJECTED_HAS_SAMPLES"):
        CandidateMeasurement(key="bad", operator="mla", status="NUMERIC_FAILED", samples_us=(1.0,))
    with pytest.raises(RankError, match="RANK_OK_WITHOUT_SAMPLES"):
        CandidateMeasurement(key="empty", operator="mla")
    with pytest.raises(RankError, match="RANK_DEVICE_SYNC_NOTE_MISSING"):
        CandidateMeasurement(key="nosync", operator="mla", measurement_source=MEASUREMENT_SOURCE_DEVICE, samples_us=(1.0,), identity_key="id")
    with pytest.raises(RankError, match="RANK_BAD_SAMPLE"):
        _device("nan", [float("nan")])


def test_noise_confirmation_requires_two_device_candidates():
    left = _device("left", [10.0, 10.2, 10.1])
    right = _device("right", [10.1, 10.3, 10.0])
    assert confirmation_required(left, right) is True
    with pytest.raises(RankError, match="RANK_SIMULATED_NOT_COMPARABLE"):
        confirmation_required(left, _simulated("sim"))
