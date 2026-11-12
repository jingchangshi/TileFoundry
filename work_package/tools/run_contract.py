#!/usr/bin/env python3
"""Run the unchanged attachment contract on a local Ascend candidate.

Only the standard library is imported before argument parsing. ``--help`` works
without PyTorch, CANN, TileLang, or an NPU. This runner checks numerical results;
it does not establish schedule conformance or measure performance.
"""

from __future__ import annotations

import argparse
import codecs
import contextlib
import datetime as dt
import functools
import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
from pathlib import Path
import platform
import re
import sys
import threading
import time
import traceback
import uuid


class PreflightError(RuntimeError):
    """The original contract cannot run in this environment."""


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def file_identity(path):
    path = Path(path).resolve()
    result = {"path": str(path), "exists": path.is_file()}
    if path.is_file():
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        result.update(sha256=digest.hexdigest(), bytes=path.stat().st_size)
    return result


def module_identity(module):
    result = {"name": module.__name__, "version": str(getattr(module, "__version__", "unknown"))}
    filename = getattr(module, "__file__", None)
    result["file"] = file_identity(filename) if filename else None
    return result


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load Python module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class CaseCollector:
    """Accept only JSON objects carrying a case result; validate after execution."""

    def __init__(self):
        self.records = []
        self.malformed = []
        self.non_case_json_count = 0

    def line(self, line):
        line = line.strip()
        if not line.startswith("{"):
            return
        try:
            value = json.loads(line)
        except (ValueError, TypeError):
            if '"case"' in line:
                self.malformed.append(line)
            return
        if not isinstance(value, dict) or "case" not in value:
            self.non_case_json_count += 1
            return
        self.records.append(value)

    def evaluate(self, expected):
        seen = {}
        unknown = []
        malformed = list(self.malformed)
        for record in self.records:
            label = record.get("case")
            if not isinstance(label, str):
                malformed.append(record)
                continue
            seen.setdefault(label, []).append(record)
            if label not in expected:
                unknown.append(label)
        duplicates = sorted(label for label, values in seen.items() if len(values) != 1)
        missing = sorted(set(expected) - set(seen))
        passed = [label for label in expected if len(seen.get(label, [])) == 1
                  and seen[label][0].get("passed") is True
                  and not seen[label][0].get("error")]
        failed = [label for label in expected if label in seen and label not in passed]
        complete = (len(expected) == 20 and len(set(expected)) == 20
                    and len(passed) == 20 and not missing and not duplicates
                    and not unknown and not malformed and len(self.records) == 20)
        return {
            "expected": len(expected), "expected_ids": expected,
            "executed_unique_expected": len(set(expected) & set(seen)),
            "result_record_count": len(self.records), "passed": len(passed),
            "failed": len(failed), "passed_ids": passed, "failed_ids": failed,
            "missing_ids": missing, "duplicate_ids": duplicates,
            "unknown_ids": sorted(set(unknown)), "malformed_records": malformed,
            "all_expected_passed": complete, "records": self.records,
        }


class FDTee:
    """Capture Python and native writes to stdout/stderr, preserving the console.

    POSIX pipes are used because PyTorch/CANN subprocesses can bypass Python's
    sys.stdout. Only stdout is interpreted as harness JSON; both streams are
    retained, unmodified, in the log. Arrival order between streams is not a
    reliable device-event timeline.
    """

    def __init__(self, path, collector):
        self.path = Path(path)
        self.collector = collector
        self.log = None
        self.lock = threading.Lock()
        self.saved = {}
        self.read_fds = {}
        self.threads = []
        self.errors = []

    def _read(self, fd, read_fd, console_fd):
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        pending = ""
        try:
            while True:
                chunk = os.read(read_fd, 65536)
                if not chunk:
                    break
                with self.lock:
                    if self.log is not None:
                        self.log.write(chunk)
                        self.log.flush()
                view = memoryview(chunk)
                while view:
                    try:
                        wrote = os.write(console_fd, view)
                    except (BrokenPipeError, OSError):
                        break  # A closed console must not discard the log.
                    view = view[wrote:]
                if fd == 1:
                    pending += decoder.decode(chunk)
                    while "\n" in pending:
                        line, pending = pending.split("\n", 1)
                        self.collector.line(line)
            if fd == 1:
                pending += decoder.decode(b"", final=True)
                if pending:
                    self.collector.line(pending)
        except Exception as exc:
            self.errors.append(f"fd {fd}: {type(exc).__name__}: {exc}")
        finally:
            with contextlib.suppress(OSError):
                os.close(read_fd)

    def __enter__(self):
        if os.name != "posix":
            raise PreflightError("This Ascend runner requires a POSIX environment for complete stdout/stderr capture.")
        sys.stdout.flush()
        sys.stderr.flush()
        self.log = self.path.open("wb", buffering=0)
        try:
            for fd in (1, 2):
                self.saved[fd] = os.dup(fd)
                read_fd, write_fd = os.pipe()
                self.read_fds[fd] = read_fd
                os.dup2(write_fd, fd)
                os.close(write_fd)
                thread = threading.Thread(target=self._read, args=(fd, read_fd, self.saved[fd]), daemon=True)
                self.threads.append(thread)
                thread.start()
        except Exception:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *unused):
        for stream in (sys.stdout, sys.stderr):
            with contextlib.suppress(Exception):
                stream.flush()
        # Flush the process C stdio buffers before restoring file descriptors.
        # This does not collect separate files written by a device profiler.
        with contextlib.suppress(Exception):
            import ctypes
            ctypes.CDLL(None).fflush(None)
        for fd, saved in self.saved.items():
            with contextlib.suppress(OSError):
                os.dup2(saved, fd)
        for thread in self.threads:
            thread.join(timeout=5)
        if any(thread.is_alive() for thread in self.threads):
            self.errors.append("A child process retained an output pipe; log capture did not reach EOF.")
        # Do not close a descriptor still owned by a reader thread. The process
        # will exit; an incomplete capture is a hard failure below.
        if not any(thread.is_alive() for thread in self.threads):
            for saved in self.saved.values():
                with contextlib.suppress(OSError):
                    os.close(saved)
        with self.lock:
            if self.log is not None:
                self.log.close()
                self.log = None


def package_versions():
    result = {}
    for name in ("torch", "torch-npu", "tilelang", "triton", "z3-solver", "apache-tvm", "PyYAML"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def select_device(torch, requested):
    npu = getattr(torch, "npu", None)
    if npu is None or not npu.is_available():
        raise PreflightError("torch.npu reports no available NPU.")
    if requested is None:
        index = int(npu.current_device())
    else:
        match = re.fullmatch(r"(?:npu:)?([0-9]+)", requested)
        if not match:
            raise PreflightError("--device must be an NPU index such as 0 or npu:0.")
        index = int(match.group(1))
    count = int(npu.device_count())
    if index >= count:
        raise PreflightError(f"Requested npu:{index}, but this process can see {count} NPU devices.")
    npu.set_device(index)
    selected = {"requested": requested, "selected": f"npu:{index}", "visible_device_count": count}
    for key, method in (("name", "get_device_name"), ("properties", "get_device_properties")):
        function = getattr(npu, method, None)
        if callable(function):
            try:
                selected[key] = str(function(index))
            except Exception as exc:
                selected[key + "_query_error"] = f"{type(exc).__name__}: {exc}"
    return selected


def prepare(args, report, fixture_root):
    case_dir = fixture_root / args.case
    paths = {
        "operator": args.operator_file,
        "test": case_dir / f"test_{args.case}.py",
        "golden": case_dir / "golden.py", "cases": case_dir / "cases.yaml",
        "proto": case_dir / "proto.yaml", "helper": fixture_root / "cannbench_contract.py",
    }
    report["files"] = {name: file_identity(path) for name, path in paths.items()}
    optional_csv = case_dir / "cases.csv"
    if optional_csv.is_file():
        report["files"]["cases_csv"] = file_identity(optional_csv)
    missing = [value["path"] for value in report["files"].values() if not value["exists"]]
    report["package_versions"] = package_versions()
    report["modules"] = {}
    # Collect all three dependency failures, rather than stopping at torch.
    dependencies = {}
    errors = []
    for name in ("torch", "torch_npu", "npu_compare"):
        try:
            module = importlib.import_module(name)
            dependencies[name] = module
            report["modules"][name] = module_identity(module)
        except Exception as exc:
            detail = f"{name}: {type(exc).__name__}: {exc}"
            report["modules"][name] = {"import_error": detail}
            errors.append(detail)
    if missing or errors:
        raise PreflightError("; ".join([*(f"Missing file: {p}" for p in missing), *errors]))
    comparator = getattr(dependencies["npu_compare"], "compare_case_result", None)
    if not callable(comparator):
        raise PreflightError("npu_compare.compare_case_result is missing or is not callable.")
    comparator_file = inspect.getsourcefile(comparator) or inspect.getfile(comparator)
    report["comparator"] = {
        "entrypoint": "npu_compare.compare_case_result", "accuracy_mode": "cann-bench",
        "implementation": file_identity(comparator_file),
        "threshold_policy": "Original helper/proto and installed comparator, unchanged.",
        "effective_thresholds": "Not inferred by this runner; pin and audit the comparator in P0.",
    }
    # Precedence is explicit. The unchanged test's parent search finds this copy.
    sys.path.insert(0, str(case_dir))
    sys.path.insert(0, str(fixture_root))
    helper = load_file("cannbench_contract", paths["helper"])
    test = load_file(f"_tf_contract_test_{args.case}_{report['run_id']}", paths["test"])
    report["modules"]["cannbench_contract"] = module_identity(helper)
    report["modules"]["test"] = module_identity(test)
    if Path(sys.modules["cannbench_contract"].__file__).resolve() != paths["helper"].resolve():
        raise PreflightError("The test imported a different cannbench_contract.py.")
    if not callable(getattr(test, "main", None)):
        raise PreflightError("The attachment test does not expose main(operator_api).")
    cases = helper.load_cases(case_dir, f"level4/{args.case}")
    expected = [str(case.get_case_id_str()) for case in cases]
    report["expected_case_ids"] = expected
    if len(expected) != 20 or len(set(expected)) != 20:
        raise PreflightError(f"Original contract must contain 20 unique cases; found {len(expected)} rows and {len(set(expected))} unique IDs.")
    report["device"] = select_device(dependencies["torch"], args.device)
    return dependencies["torch"], test, expected


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--case", required=True, choices=("attention_residual", "mla"))
    parser.add_argument("--operator-file", required=True, type=Path, help="Python candidate module; never imported in preflight-only mode.")
    parser.add_argument("--callable", dest="callable_name", help="Candidate function name (default: the --case name).")
    parser.add_argument("--device", help="Visible NPU index or npu:N; default: torch.npu.current_device().")
    parser.add_argument("--cache-dir", default="off", help="CANNBENCH_CACHE_DIR; default off disables the historical reference cache.")
    parser.add_argument("--report", required=True, type=Path, help="JSON report path; written on success, failure, or blocked preflight.")
    parser.add_argument("--log", type=Path, help="Complete stdout/stderr log; default: REPORT.log.")
    parser.add_argument("--preflight-only", action="store_true", help="Check dependencies/fixtures/device; do not import the candidate or run main/kernel.")
    return parser.parse_args(argv)


def write_report(path, report):
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        with contextlib.suppress(OSError):
            temporary.unlink()


def main(argv=None):
    args = parse_args(argv)
    started = time.monotonic()
    fixture_root = Path(__file__).resolve().parents[1] / "fixtures"
    args.operator_file = args.operator_file.resolve()
    args.report = args.report.resolve()
    args.log = args.log.resolve() if args.log else Path(str(args.report) + ".log")
    # Reports must never modify the original fixtures or candidate source.
    for output in (args.report, args.log):
        if output == args.operator_file or output == Path(__file__).resolve() or output.is_relative_to(fixture_root):
            print(f"Refusing to overwrite source/fixture with output: {output}", file=sys.stderr)
            return 2
    if args.report == args.log:
        print("--report and --log must name different files.", file=sys.stderr)
        return 2
    try:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.log.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(f"Cannot create report/log directory: {exc}", file=sys.stderr)
        return 2
    os.environ["CANNBENCH_CACHE_DIR"] = args.cache_dir
    report = {
        "schema_version": "tf-ascend.contract-run.v1", "run_id": uuid.uuid4().hex,
        "started_at_utc": utc_now(), "case": args.case,
        "operator_callable": args.callable_name or args.case,
        "mode": "preflight_only" if args.preflight_only else "numerical_contract",
        "status": "PREFLIGHT_ONLY" if args.preflight_only else "STARTED",
        "numerical_status": "NOT_RUN", "preflight_status": "STARTED",
        "kernel_execution_requested": not args.preflight_only,
        "candidate_imported": False, "contract_main_invoked": False,
        "main_returned_normally": False, "device_synchronize_succeeded": False,
        "interpreter": {"executable": file_identity(sys.executable), "version": sys.version,
                        "platform": platform.platform(), "machine": platform.machine()},
        "command": [sys.executable, str(Path(__file__).resolve()), *(argv if argv is not None else sys.argv[1:])],
        "cache_dir": os.environ["CANNBENCH_CACHE_DIR"],
        "log_path": str(args.log), "report_path": str(args.report),
        "schedule_conformance": "NOT_EVALUATED", "performance": "NOT_MEASURED",
    }
    collector = CaseCollector()
    tee = FDTee(args.log, collector)
    expected = []
    exit_code = 2
    try:
        with tee:
            print(f"[contract-runner] run_id={report['run_id']} case={args.case} mode={report['mode']}", flush=True)
            try:
                torch, test, expected = prepare(args, report, fixture_root)
                report["preflight_status"] = "PASS"
                if args.preflight_only:
                    print("[contract-runner] PREFLIGHT_ONLY: candidate not imported; main/kernel not invoked.", flush=True)
                    exit_code = 0
                else:
                    sys.path.insert(0, str(args.operator_file.parent))
                    operator = load_file(f"_tf_contract_candidate_{report['run_id']}", args.operator_file)
                    report["candidate_imported"] = True
                    report["modules"]["operator"] = module_identity(operator)
                    operator_api = getattr(operator, report["operator_callable"], None)
                    if not callable(operator_api):
                        raise RuntimeError(f"Candidate has no callable {report['operator_callable']!r}.")
                    actual_api = operator_api
                    @functools.wraps(actual_api)
                    def operator_api(*call_args, **call_kwargs):
                        try:
                            return actual_api(*call_args, **call_kwargs)
                        except (ModuleNotFoundError, ImportError) as exc:
                            # The unchanged Harness catches per-case exceptions.
                            # Preserve dependency failures before it stringifies
                            # them so they still receive the documented exit 2.
                            report.setdefault("runtime_dependency_errors", []).append(
                                {"type": type(exc).__name__, "message": str(exc)})
                            raise
                    report["contract_main_invoked"] = True
                    report["numerical_status"] = "RUNNING"
                    test.main(operator_api)
                    report["main_returned_normally"] = True
                    torch.npu.synchronize()
                    report["device_synchronize_succeeded"] = True
                    exit_code = 0  # Final status also requires every parsed case below.
            except (PreflightError, ModuleNotFoundError, ImportError) as exc:
                report["status"] = "PREFLIGHT_ONLY" if args.preflight_only else "BLOCKED_DEPENDENCY"
                report["preflight_status"] = "FAIL" if not report["contract_main_invoked"] else report["preflight_status"]
                report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
                if report["contract_main_invoked"]:
                    report["numerical_status"] = "INCOMPLETE"
                traceback.print_exc()
                exit_code = 2
            except BaseException as exc:
                report["status"] = "PREFLIGHT_ONLY" if args.preflight_only else "ERROR"
                report["preflight_status"] = "FAIL" if report["preflight_status"] == "STARTED" else report["preflight_status"]
                report["error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
                if report["contract_main_invoked"]:
                    report["numerical_status"] = "FAIL"
                traceback.print_exc()
                exit_code = 2 if args.preflight_only or report["preflight_status"] != "PASS" else 1
    except BaseException as exc:
        report["status"] = "PREFLIGHT_ONLY" if args.preflight_only else "ERROR"
        report["preflight_status"] = "FAIL" if report["preflight_status"] == "STARTED" else report["preflight_status"]
        report["runner_error"] = {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        exit_code = 2 if not report["contract_main_invoked"] else 1
        with contextlib.suppress(OSError):
            with args.log.open("a", encoding="utf-8") as handle:
                handle.write(report["runner_error"]["traceback"])
    expected = report.get("expected_case_ids", expected)
    report["results"] = collector.evaluate(expected)
    report["capture"] = {"method": "posix_fd_tee_stdout_and_stderr", "complete": not tee.errors, "errors": tee.errors}
    if args.preflight_only:
        report["status"] = "PREFLIGHT_ONLY"
        report["numerical_status"] = "NOT_RUN"
    elif report["contract_main_invoked"]:
        passed = (exit_code == 0 and report["main_returned_normally"]
                  and report["device_synchronize_succeeded"]
                  and report["results"]["all_expected_passed"] and not tee.errors)
        if passed:
            report["status"] = "PASS"
            report["numerical_status"] = "PASS"
        elif exit_code != 2:
            report["status"] = "FAIL"
            report["numerical_status"] = "FAIL"
            exit_code = 1
    if tee.errors:
        exit_code = 2 if args.preflight_only or not report["contract_main_invoked"] else 1
        if not args.preflight_only:
            report["status"] = "ERROR_INCOMPLETE_LOG"
            report["numerical_status"] = "INCOMPLETE" if report["contract_main_invoked"] else "NOT_RUN"
    if report.get("runtime_dependency_errors"):
        report["status"] = "BLOCKED_DEPENDENCY"
        report["numerical_status"] = "INCOMPLETE"
        exit_code = 2
    report["finished_at_utc"] = utc_now()
    report["elapsed_seconds"] = time.monotonic() - started
    report["exit_code"] = exit_code
    # Detect accidental input/fixture changes during this run. Hashing is not a
    # replacement for the work package's source-lock verification.
    changed = []
    for name, before in report.get("files", {}).items():
        try:
            after = file_identity(before["path"])
        except OSError as exc:
            report.setdefault("source_recheck_errors", {})[name] = str(exc)
            changed.append(name)
        else:
            if before.get("sha256") != after.get("sha256"):
                changed.append(name)
    report["source_files_changed_during_run"] = changed
    if changed:
        report["status"] = "PREFLIGHT_ONLY" if args.preflight_only else "FAIL_SOURCE_CHANGED"
        report["numerical_status"] = "NOT_RUN" if args.preflight_only else "INVALIDATED"
        exit_code = report["exit_code"] = 2 if args.preflight_only else 1
    try:
        report["log"] = file_identity(args.log)
        write_report(args.report, report)
    except OSError as exc:
        print(f"Cannot write final report: {exc}", file=sys.stderr)
        return 2
    print(f"[contract-runner] status={report['status']} numerical={report['numerical_status']} "
          f"passed={report['results']['passed']}/{report['results']['expected']} report={args.report}", flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
