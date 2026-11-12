"""Real worker deadlines, independent of NPU timing evidence."""
import os
from pathlib import Path
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


def test_timeout_kills_child_even_when_group_leader_exits_on_term(tmp_path):
    child_pid = tmp_path / "child.pid"
    child_code = ("import os,signal,time; from pathlib import Path; "
                  "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                  "Path(%r).write_text(str(os.getpid())); time.sleep(30)" % str(child_pid))
    parent_code = ("import subprocess,sys,time; "
                   "subprocess.Popen([sys.executable, '-c', %r]); time.sleep(30)" % child_code)
    result = run_process([sys.executable, "-c", parent_code], stage="compile",
                         timeout_s=0.5, log_path=tmp_path / "child.log")
    assert result.timed_out
    pid = int(child_pid.read_text())
    # An orphan may remain as a zombie until init reaps it, but cannot execute.
    state_file = Path("/proc/%d/stat" % pid)
    for _ in range(100):
        if not state_file.exists() or state_file.read_text().split()[2] == "Z":
            break
        time.sleep(0.01)
    else:
        pytest.fail("compiler child survived the worker deadline")


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
