"""CPU-only exec workers exercise the actual search-to-process boundary; no NPU."""
import json
import sys
import time
from dataclasses import replace
from types import SimpleNamespace

import pytest

from tilefoundry_ascend.search.harness import (
    Budget, CacheIdentity, CandidateRun, DeviceTimer, MEASUREMENT_SOURCE_DEVICE,
    MEASUREMENT_SOURCE_SIMULATED, WorkerCandidateRun, run_candidate, search_case,
)
from tilefoundry_ascend.search.process import Deadline, WorkerCommand


WORKER = r'''
import json, os, pathlib, sys, time
stage = os.environ["TF_SEARCH_STAGE"]
with open(sys.argv[1], "a") as stream:
    stream.write(stage + "\n")
time.sleep(float(sys.argv[5]))
if stage == sys.argv[2]:
    time.sleep(30)
if stage == sys.argv[3]:
    sys.exit(7)
payload = dict(run_id=os.environ["TF_SEARCH_RUN_ID"], stage=stage,
               identity_key=os.environ["TF_SEARCH_IDENTITY_KEY"], failure=None)
if stage == "profile":
    n = int(os.environ["TF_SEARCH_TIMING_CALLS_PER_BATCH"])
    batches = int(os.environ["TF_SEARCH_CONFIRMATION_BATCHES"])
    payload.update(samples_us=[12.0] * (n * batches),
                   discarded_warmups=int(os.environ["TF_SEARCH_WARMUP_CALLS"]),
                   measurement_source="SIMULATED_CLOCK")
if stage == sys.argv[4]:
    payload["run_id"] = "stale-run"
pathlib.Path(os.environ["TF_SEARCH_RESULT_PATH"]).write_text(json.dumps(payload))
'''


def identity(key="candidate"):
    return CacheIdentity("test", "cpu-test", "python", "src", "plan", "float16",
                         "test", "shape", "test", key)


def contract(tmp_path, *, block="", fail="", stale="", device=False, delay=0):
    stages = {}
    for stage in ("compile", "numeric", "launch", "profile"):
        stages[stage] = WorkerCommand(
            (sys.executable, "-c", WORKER, str(tmp_path / "started"), block, fail, stale, str(delay)),
            str(tmp_path / (stage + ".log")), str(tmp_path / (stage + ".json")))
    return WorkerCandidateRun(**stages, measurement_source=(
        MEASUREMENT_SOURCE_DEVICE if device else MEASUREMENT_SOURCE_SIMULATED))


def budget(**kwargs):
    return Budget(warmup_calls=1, timing_calls_per_batch=2, confirmation_batches=1,
                  **kwargs)


def search(tmp_path, worker, *, seconds=0.6, compile_limit=3):
    calls = []
    candidates = [SimpleNamespace(key=lambda k=k: k) for k in ("first", "second")]
    settings = budget(case_search_deadline_s=seconds,
                      candidate_compile_deadline_s=compile_limit)

    def run_one(candidate, judgement):
        calls.append(candidate.key())
        # No manually passed Deadline: search_case must propagate it to this call.
        return run_candidate(worker, identity(candidate.key()), budget=settings)

    result = search_case(case_label="cpu-worker-test", operator="test", seeds=candidates,
                         advanced=(), judge=lambda c: SimpleNamespace(feasible=True),
                         run_one=run_one, budget=settings.case_budget(
                             seeds=2, neighbours=0, advanced=0))
    return result, calls


@pytest.mark.parametrize("stage", ["compile", "numeric", "launch", "profile"])
def test_real_blocking_worker_obeys_shared_case_deadline(tmp_path, stage):
    worker = contract(tmp_path, block=stage)
    started = time.monotonic()
    result, calls = search(tmp_path, worker)
    assert time.monotonic() - started < 2.5
    assert calls == ["first"]
    assert len(result.results) == 1
    row = result.results[0]
    assert row.status == ("COMPILE_TIMEOUT" if stage == "compile" else "RUN_TIMEOUT")
    assert row.stats is None and not row.rankable
    assert result.ranking.winner is None and not result.ranking.ordered
    stages = ["compile", "numeric", "launch", "profile"]
    assert (tmp_path / "started").read_text().splitlines() == stages[:stages.index(stage) + 1]
    assert "wall deadline" in result.stop_reason


def test_device_timeout_stops_candidates_even_with_case_time_remaining(tmp_path):
    # Device-labelled contract is a CPU blocking counterexample, never timing evidence.
    result, calls = search(tmp_path, contract(tmp_path, block="compile", device=True),
                           seconds=5, compile_limit=0.3)
    assert calls == ["first"]
    assert result.results[0].stats is None
    assert result.results[0].status == "COMPILE_TIMEOUT"
    assert result.ranking.winner is None
    assert "device worker timeout" in result.stop_reason


def test_completed_stages_consume_remaining_time_not_a_fresh_budget(tmp_path):
    # Each stage alone fits, but the sequence cannot fit in the case budget.
    result, calls = search(tmp_path, contract(tmp_path, delay=0.22), seconds=0.55)
    assert calls == ["first"]
    assert result.results[0].status == "RUN_TIMEOUT"
    assert result.results[0].stats is None
    assert "profile" not in (tmp_path / "started").read_text().splitlines()
    assert result.ranking.winner is None


def test_worker_success_retains_only_declared_simulated_samples(tmp_path):
    result = run_candidate(contract(tmp_path), identity(), budget=budget())
    assert result.status == "OK"
    assert result.measurement_source == MEASUREMENT_SOURCE_SIMULATED
    assert result.stats.samples_us == (12.0, 12.0)
    assert result.stats.discarded_warmups == 1


@pytest.mark.parametrize("stage,status", [("compile", "COMPILE_FAILED"),
    ("numeric", "RUN_ERROR"), ("launch", "RUN_ERROR"), ("profile", "RUN_ERROR")])
def test_native_failure_has_named_status_and_no_stats(tmp_path, stage, status):
    result = run_candidate(contract(tmp_path, fail=stage), identity(), budget=budget())
    assert result.status == status and result.stats is None
    assert stage in result.detail


def test_stale_receipt_and_missing_result_are_not_success(tmp_path):
    worker = contract(tmp_path, stale="numeric")
    result = run_candidate(worker, identity(), budget=budget())
    assert result.status == "RUN_ERROR" and result.stats is None
    assert "invocation and identity" in result.detail
    # Preexisting valid-looking JSON must be removed before an exec that writes nothing.
    path = tmp_path / "compile.json"
    path.write_text(json.dumps({"failure": None}))
    worker = replace(worker, compile=replace(worker.compile, argv=(sys.executable, "-c", "pass")))
    result = run_candidate(worker, identity(), budget=budget())
    assert result.status == "COMPILE_FAILED" and result.stats is None
    assert not path.exists()


def test_device_callable_is_refused_before_any_callback(tmp_path):
    def forbidden(*args):
        pytest.fail("unbounded device callback started")
    result = run_candidate(CandidateRun(forbidden, forbidden, forbidden, forbidden),
                           identity(), timer=DeviceTimer(sample=forbidden))
    assert result.status == "RUN_ERROR" and result.stats is None
    assert result.reason == "ENFORCED_WORKER_REQUIRED"


def test_expired_shared_deadline_never_starts_worker(tmp_path):
    now = [0.0]
    deadline = Deadline(1, clock=lambda: now[0])
    now[0] = 2.0
    result = run_candidate(contract(tmp_path), identity(), deadline=deadline)
    assert result.status == "COMPILE_TIMEOUT" and result.stats is None
    assert not (tmp_path / "started").exists()
