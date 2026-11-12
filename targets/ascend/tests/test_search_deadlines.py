"""Real worker deadlines, independent of NPU timing evidence."""
import sys
import time

import pytest

from tilefoundry_ascend.search.harness import Budget, HarnessError
from tilefoundry_ascend.search.process import Deadline, run_process


def test_all_origins_share_one_cap():
    with pytest.raises(HarnessError, match="BUDGET_ORIGINS_OVER_CAP"):
        Budget(max_candidates_per_case=39)
    for kwargs in ({"seed_candidates_max": -1},
                   {"case_search_deadline_s": 0},
                   {"candidate_compile_deadline_s": float("inf")}):
        with pytest.raises(HarnessError):
            Budget(**kwargs)


def test_shared_deadline_does_not_restart_per_stage():
    now = [10.0]
    deadline = Deadline(3.0, clock=lambda: now[0])
    assert deadline.remaining(2.0) == 2.0
    now[0] = 12.5
    assert deadline.remaining(2.0) == 0.5
    now[0] = 14.0
    assert deadline.remaining() == 0


def test_worker_timeout_interrupts_instead_of_waiting_for_return(tmp_path):
    result = run_process([sys.executable, "-c",
                          "import time; print(\"started\", flush=True); time.sleep(30)"],
                         stage="compile", timeout_s=0.2,
                         log_path=tmp_path / "timeout.log")
    assert result.status == "COMPILE_TIMEOUT"
    assert result.timed_out
    assert result.elapsed_seconds < 3
    assert (tmp_path / "timeout.log").read_text() == "started\n"


def test_worker_failure_retains_native_log(tmp_path):
    result = run_process([sys.executable, "-c",
                          "import sys; print(\"compiler error\", file=sys.stderr); sys.exit(7)"],
                         stage="compile", timeout_s=2, log_path=tmp_path / "fail.log")
    assert result.status == "COMPILE_FAILED"
    assert result.returncode == 7
    assert "compiler error" in (tmp_path / "fail.log").read_text()


def test_exhausted_case_budget_does_not_start_worker(tmp_path):
    now = [0.0]
    deadline = Deadline(1, clock=lambda: now[0])
    now[0] = 2.0
    marker = tmp_path / "should-not-exist"
    result = run_process([sys.executable, "-c",
                          "from pathlib import Path; Path(%r).touch()" % str(marker)],
                         stage="candidate", timeout_s=2, deadline=deadline,
                         log_path=tmp_path / "exhausted.log")
    assert result.status == "RUN_TIMEOUT"
    assert not marker.exists()
