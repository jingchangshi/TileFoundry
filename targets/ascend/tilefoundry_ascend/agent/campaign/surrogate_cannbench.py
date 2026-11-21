"""Four-tier host-outside-gate evaluation pyramid (L0->L1->L2->L3->L4) for campaigns."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import tomllib
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from tilefoundry_ascend.agent.campaign.anchor_specs import (
    AnchorCaseSpec,
    PerfTargetSpec,
    compute_case_eq3_score,
    default_source_root,
    get_default_anchors,
    get_operator_entrypoint,
    get_operator_module_relpath,
    get_operator_task,
    lookup_case_dims,
    parse_simple_env_file,
    resolve_cannbench_repo,
    resolve_cannbench_tasks_root,
)
from tilefoundry_ascend.agent.orchestrator.gate_funnel import (
    StageAResult,
    StageBResult,
    _bounded_compile_diagnostic,
    _parse_cce_res_usage_with_summary,
    build_stage_b_driver_source,
    run_stage_a_static,
    run_stage_b_compile,
    verify_agent_gates_receipt,
)
from tilefoundry_ascend.agent.orchestrator.llm_client import (
    KernelLLMBackend,
    extract_python_dsl,
)
from tilefoundry_ascend.agent.orchestrator.portfolio_runner import build_syntax_repair_request
from tilefoundry_ascend.ascend.cost_model.microarch_profile import (
    extract_microarch_profile_from_cce,
)

logger = logging.getLogger(__name__)


def wait_for_npu_quiescent(max_retries: int = 10):
    """Wait for NPU devices (davinci) to be closed in /proc/self/fd/."""
    for _ in range(max_retries):
        found = False
        try:
            fd_path = Path("/proc/self/fd")
            for fd in fd_path.iterdir():
                try:
                    if "davinci" in str(fd.readlink()):
                        found = True
                        break
                except OSError:
                    continue
        except OSError:
            pass
        if not found:
            return
        time.sleep(1.0)
    logger.warning("NPU still busy after waiting for quiescence.")

DEFAULT_AGENT_GATES_MAX_ATTEMPTS = 5
DEFAULT_AGENT_GATES_BACKOFF_SEC = 3.0

AGENT_GATES_ADAPTER = "tilefoundry-npu"
AGENT_GATES_DEVICE_FAST_GATE = "tilefoundry-npu-device-fast"
AGENT_GATES_L3_PROMOTION_GATE = "tilefoundry-npu-l3-promotion"
AGENT_GATES_L4_CANNBENCH_GATE = "tilefoundry-npu-cannbench"
FAST_REQUEST_SCHEMA = "tilefoundry-fast-request/v1"
FAST_RESPONSE_SCHEMA = "tilefoundry-fast-response/v1"
FAST_REQUEST_ENV = "TF_FAST_REQUEST"
FAST_SOURCE_ROOT_ENV = "TF_FAST_SOURCE_ROOT"
FAST_STATUS_OK = "ok"
FAST_STATUS_CANDIDATE_FAILURE = "candidate_failure"
FAST_STATUS_REQUEST_ERROR = "request_error"
FAST_STATUS_INFRASTRUCTURE_ERROR = "infrastructure_error"
DEFAULT_L3_PROMOTION_WAIT_SEC = 1920
DEFAULT_L4_CANNBENCH_WAIT_SEC = 5400
_V2_WAIT_SLICE_SEC = 60.0
_V2_SUBMIT_TIMEOUT_SEC = 60.0
_V2_POST_WAIT_TIMEOUT_SEC = 60.0
_V2_STRIPPED_PYTHON_ENV_KEYS = frozenset(
    {"PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE"}
)

FAILURE_KIND_CANDIDATE = "candidate"
FAILURE_KIND_INFRASTRUCTURE = "infrastructure"


@dataclass(frozen=True)
class AnchorEvalMetrics:
    """Single CANNBench anchor accuracy, latency, speedup, and Roofline eta record."""

    case_num: int
    regime_tag: str
    accuracy_passed: bool
    cand_us: float
    ref_perf_us: float
    t_hw_us: float
    speedup: float
    roofline_eta: float
    eq3_score: float
    error_message: str = ""

    def to_dict(self) -> dict[str, object]:
        """Serialize anchor evaluation metrics to a JSON-compatible dictionary."""
        return {
            "case_num": self.case_num,
            "regime_tag": self.regime_tag,
            "accuracy_passed": self.accuracy_passed,
            "cand_us": round(self.cand_us, 6),
            "ref_perf_us": round(self.ref_perf_us, 6),
            "t_hw_us": round(self.t_hw_us, 6),
            "speedup": round(self.speedup, 6),
            "roofline_eta": round(self.roofline_eta, 6),
            "eq3_score": round(self.eq3_score, 6),
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> AnchorEvalMetrics:
        """Reconstruct AnchorEvalMetrics from a JSON dictionary."""
        return cls(
            case_num=int(payload["case_num"]),
            regime_tag=str(payload["regime_tag"]),
            accuracy_passed=bool(payload["accuracy_passed"]),
            cand_us=float(payload["cand_us"]),
            ref_perf_us=float(payload["ref_perf_us"]),
            t_hw_us=float(payload["t_hw_us"]),
            speedup=float(payload["speedup"]),
            roofline_eta=float(payload["roofline_eta"]),
            eq3_score=float(payload["eq3_score"]),
            error_message=str(payload.get("error_message", "")),
        )


@dataclass(frozen=True)
class CandidateSurrogateResult:
    """Aggregated L2 surrogate CANNBench evaluation summary across active anchors."""

    candidate_id: str
    operator_name: str
    all_passed: bool
    avg_speedup: float
    min_case_speedup: float
    avg_roofline_eta: float
    surrogate_score: float
    anchors: tuple[AnchorEvalMetrics, ...] = ()
    error_message: str = ""

    def regime_headroom(self, regime_tag: str) -> float:
        """Return mean Roofline headroom (1 - eta) across anchors matching regime_tag."""
        matched = [a for a in self.anchors if a.regime_tag == regime_tag]
        if not matched:
            return max(0.0, 1.0 - self.avg_roofline_eta)
        return sum(max(0.0, 1.0 - a.roofline_eta) for a in matched) / len(matched)

    def for_regime(self, regime_tag: str) -> CandidateSurrogateResult:
        """Return CandidateSurrogateResult filtered strictly to anchors of regime_tag."""
        matched = tuple(a for a in self.anchors if a.regime_tag == regime_tag)
        if not matched:
            return CandidateSurrogateResult(
                candidate_id=self.candidate_id,
                operator_name=self.operator_name,
                all_passed=False,
                avg_speedup=0.0,
                min_case_speedup=0.0,
                avg_roofline_eta=0.0,
                surrogate_score=0.0,
                anchors=(),
                error_message=self.error_message,
            )
        return aggregate_surrogate_result(
            self.candidate_id,
            self.operator_name,
            matched,
            error_message=self.error_message,
        )

    def meets_perf_target(self, spec: PerfTargetSpec) -> bool:
        """Return True iff accuracy, score, speedup, eta, and latency limits pass."""
        if not self.all_passed or not self.anchors:
            return False
        if self.surrogate_score < spec.target_surrogate_score:
            return False
        if self.avg_speedup < spec.target_avg_speedup:
            return False
        if self.min_case_speedup < spec.target_min_case_speedup:
            return False
        if self.avg_roofline_eta < spec.target_roofline_eta:
            return False
        by_case = {a.case_num: a for a in self.anchors}
        for case_num, max_us in spec.per_anchor_max_us.items():
            anchor = by_case.get(int(case_num))
            if anchor is None or anchor.cand_us <= 0.0 or anchor.cand_us > float(max_us):
                return False
        return True

    def is_roofline_saturated(self, spec: PerfTargetSpec) -> bool:
        """Return True iff every active anchor has reached target_roofline_eta."""
        if not self.all_passed or not self.anchors:
            return False
        return all(a.roofline_eta >= spec.target_roofline_eta for a in self.anchors)

    def mean_latency_us(self) -> float:
        """Return arithmetic mean candidate latency in microseconds across anchors."""
        valid = [a.cand_us for a in self.anchors if a.cand_us > 0.0]
        if valid:
            return sum(valid) / len(valid)
        if self.avg_speedup > 0.0:
            return 1000.0 / self.avg_speedup
        return 1e9

    def to_dict(self) -> dict[str, object]:
        """Serialize CandidateSurrogateResult to a JSON-compatible dictionary."""
        return {
            "candidate_id": self.candidate_id,
            "operator_name": self.operator_name,
            "all_passed": self.all_passed,
            "avg_speedup": round(self.avg_speedup, 6),
            "min_case_speedup": round(self.min_case_speedup, 6),
            "avg_roofline_eta": round(self.avg_roofline_eta, 6),
            "surrogate_score": round(self.surrogate_score, 6),
            "anchors": [a.to_dict() for a in self.anchors],
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> CandidateSurrogateResult:
        """Reconstruct CandidateSurrogateResult from a JSON dictionary."""
        anchors = tuple(
            AnchorEvalMetrics.from_dict(item)
            for item in payload.get("anchors", ())
            if isinstance(item, Mapping)
        )
        return cls(
            candidate_id=str(payload["candidate_id"]),
            operator_name=str(payload["operator_name"]),
            all_passed=bool(payload["all_passed"]),
            avg_speedup=float(payload["avg_speedup"]),
            min_case_speedup=float(payload["min_case_speedup"]),
            avg_roofline_eta=float(payload["avg_roofline_eta"]),
            surrogate_score=float(payload["surrogate_score"]),
            anchors=anchors,
            error_message=str(payload.get("error_message", "")),
        )


@dataclass(frozen=True)
class L0L1CandidateCheck:
    """Host-CPU Stage L0 static + Stage L1 bisheng pre-filter result for one candidate."""

    candidate_id: str
    passed: bool
    failed_tier: str | None
    patched_module_source: str
    snippet_source: str
    repaired: bool = False
    stage_a: StageAResult | None = None
    stage_b: StageBResult | None = None
    error_message: str = ""


@dataclass(frozen=True)
class L3AcceptanceResult:
    """Outcome of AgentGates L3 full-contract acceptance verification."""

    passed: bool
    run_id: str = ""
    receipt_path: str = ""
    elapsed_sec: float = 0.0
    error_message: str = ""
    failure_kind: str = ""
    failure_category: str = ""
    pending: bool = False
    execution_id: str = ""
    candidate_sha256: str = ""
    target_path: str = ""
    device_id: str = ""
    state: str = ""
    execution_terminal: bool = True


@dataclass(frozen=True)
class L4CannbenchResult:
    """Outcome and per-case telemetry of official L4 tilefoundry-npu-cannbench gate."""

    passed: bool
    avg_speedup: float = 0.0
    score: float = 0.0
    per_case_speedups: dict[int, float] = field(default_factory=dict)
    per_case_elapsed_us: dict[int, float] = field(default_factory=dict)
    run_id: str = ""
    receipt_path: str = ""
    report_dir: str = ""
    error_message: str = ""
    failure_kind: str = ""
    execution_id: str = ""
    official_report_valid: bool = False
    all_cases_passed: bool = False
    all_timings_valid: bool = False
    case_count: int = 0
    passed_cases: int = 0
    device_id: str = ""
    state: str = ""
    execution_terminal: bool = True
    target_path: str = ""
    candidate_sha256: str = ""
    attempts: tuple[dict[str, str], ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Serialize L4 official CANNBench result to a JSON-compatible dictionary."""
        return {
            "passed": self.passed,
            "avg_speedup": round(self.avg_speedup, 6),
            "score": round(self.score, 6),
            "per_case_speedups": {str(k): v for k, v in sorted(self.per_case_speedups.items())},
            "per_case_elapsed_us": {str(k): v for k, v in sorted(self.per_case_elapsed_us.items())},
            "run_id": self.run_id,
            "receipt_path": self.receipt_path,
            "report_dir": self.report_dir,
            "error_message": self.error_message,
            "failure_kind": self.failure_kind,
            "execution_id": self.execution_id,
            "official_report_valid": self.official_report_valid,
            "all_cases_passed": self.all_cases_passed,
            "all_timings_valid": self.all_timings_valid,
            "case_count": self.case_count,
            "passed_cases": self.passed_cases,
            "device_id": self.device_id,
            "state": self.state,
            "execution_terminal": self.execution_terminal,
            "target_path": self.target_path,
            "candidate_sha256": self.candidate_sha256,
            "attempts": [dict(item) for item in self.attempts],
        }


def build_anchor_metrics(
    spec: AnchorCaseSpec,
    *,
    accuracy_passed: bool,
    cand_us: float,
    error_message: str = "",
) -> AnchorEvalMetrics:
    """Compute speedup, Roofline eta, and Eq.3 score from measured latency."""
    valid_lat = accuracy_passed and math.isfinite(cand_us) and cand_us > 0.0
    safe_lat = float(cand_us) if (math.isfinite(cand_us) and cand_us > 0.0) else 0.0
    speedup = (spec.ref_perf_us / safe_lat) if (valid_lat and spec.ref_perf_us > 0.0) else 0.0
    eta = (spec.t_hw_us / safe_lat) if (valid_lat and spec.t_hw_us > 0.0) else 0.0
    eq3 = compute_case_eq3_score(spec.ref_perf_us, safe_lat, spec.t_hw_us) if valid_lat else 0.0
    return AnchorEvalMetrics(
        case_num=spec.case_num,
        regime_tag=spec.regime_tag,
        accuracy_passed=bool(accuracy_passed and safe_lat > 0.0),
        cand_us=safe_lat,
        ref_perf_us=spec.ref_perf_us,
        t_hw_us=spec.t_hw_us,
        speedup=speedup,
        roofline_eta=eta,
        eq3_score=eq3,
        error_message=error_message,
    )


def aggregate_surrogate_result(
    candidate_id: str,
    operator_name: str,
    anchor_metrics: Sequence[AnchorEvalMetrics],
    *,
    error_message: str = "",
) -> CandidateSurrogateResult:
    """Aggregate per-anchor metrics into speedup, eta, and surrogate CANNBench score."""
    metrics = tuple(anchor_metrics)
    if not metrics:
        return CandidateSurrogateResult(
            candidate_id=candidate_id,
            operator_name=operator_name,
            all_passed=False,
            avg_speedup=0.0,
            min_case_speedup=0.0,
            avg_roofline_eta=0.0,
            surrogate_score=0.0,
            anchors=(),
            error_message=error_message or "no anchor metrics evaluated",
        )
    all_ok = all(m.accuracy_passed and m.cand_us > 0.0 for m in metrics)
    avg_sp = sum(m.speedup for m in metrics) / len(metrics)
    min_sp = min(m.speedup for m in metrics)
    avg_eta = sum(m.roofline_eta for m in metrics) / len(metrics)
    case_terms = [(0.20 + 0.70 * m.eq3_score) if m.accuracy_passed else 0.0 for m in metrics]
    comp_score = 0.05 if any(m.cand_us > 0.0 for m in metrics) else 0.0
    bonus = 0.05 if all_ok else 0.0
    surrogate_score = (comp_score + bonus + (sum(case_terms) / len(metrics))) * 100.0
    return CandidateSurrogateResult(
        candidate_id=candidate_id,
        operator_name=operator_name,
        all_passed=all_ok,
        avg_speedup=avg_sp,
        min_case_speedup=min_sp,
        avg_roofline_eta=avg_eta,
        surrogate_score=surrogate_score,
        anchors=metrics,
        error_message=error_message,
    )


_NPU_COMPARE_SHIM_SOURCE = '''"""Runtime bridge adapting kernel_eval.utils.compare to npu_compare API."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace


def _load_compare_tensors():
    cann_repo = Path({cann_repo!r})
    bench_src = str(cann_repo / "src")
    if bench_src not in sys.path:
        sys.path.insert(0, bench_src)
    from kernel_eval.utils.compare import compare_tensors

    return compare_tensors


_COMPARE_TENSORS = _load_compare_tensors()


def compare_case_result(
    *,
    case_id: str,
    actual: object,
    golden: object,
    inputs: object = (),
    compute: bool = True,
    accuracy_mode: str | None = None,
    native_ref_operator=None,
    extra: dict[str, object] | None = None,
):
    """Compare candidate tensor against golden oracle using CANNBench thresholds."""
    _ = (case_id, inputs, compute, accuracy_mode)
    custom_thresholds = (extra or {{}}).get("precision_thresholds") or {{}}
    dtype_str = str(getattr(actual, "dtype", "float32")).removeprefix("torch.")
    res = _COMPARE_TENSORS(
        actual,
        golden,
        dtype=dtype_str,
        native_output=None,
        custom_thresholds=custom_thresholds,
    )
    if not res.passed and native_ref_operator is not None:
        native_val = native_ref_operator(None)
        res = _COMPARE_TENSORS(
            actual,
            golden,
            dtype=dtype_str,
            native_output=native_val,
            custom_thresholds=custom_thresholds,
        )
    msg = res.error_msg or (
        f"MERE={{res.mere:.6e}}, MARE={{res.mare:.6e}}, max_diff={{res.max_diff:.6e}}"
    )
    return SimpleNamespace(passed=bool(res.passed), message=msg)
'''


def ensure_npu_compare_shim_dir(
    target_dir: str | Path,
    cannbench_repo: str | Path | None = None,
    env_file: str | Path | None = None,
) -> Path:
    """Create a runtime npu_compare.py shim delegating to CANNBench compare_tensors."""
    repo = resolve_cannbench_repo(cannbench_repo=cannbench_repo, env_file=env_file)
    out_dir = Path(target_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    shim_path = out_dir / "npu_compare.py"
    shim_path.write_text(
        _NPU_COMPARE_SHIM_SOURCE.format(cann_repo=str(repo)),
        encoding="utf-8",
    )
    return out_dir


def run_l0_static_precheck(
    module_source: str,
    *,
    builder_name: str,
    builder_kwargs: Mapping[str, object],
    config_selector_name: str | None = None,
    pass_configs: dict[str, object] | None = None,
    compile_flags: Sequence[str] | str | None = None,
    dims: Mapping[str, object] | None = None,
    var_bindings: Mapping[str, int] | None = None,
) -> StageAResult:
    """Run Tier L0 static AST, ledger, and router inspection on Host CPU."""
    return run_stage_a_static(
        module_source,
        factory_name=builder_name,
        factory_kwargs=builder_kwargs,
        pass_configs=pass_configs,
        compile_flags=compile_flags,
        config_selector_name=config_selector_name,
        dims=dims,
        var_bindings=var_bindings,
    )


def _resolve_env_values(env_file: str | Path | None = None) -> dict[str, str]:
    resolved = (
        str(env_file)
        if env_file is not None
        else os.environ.get("TILEFOUNDRY_NPU_ENV_FILE", "").strip()
    )
    if resolved:
        return parse_simple_env_file(resolved)
    return {}


def run_l1_compile_check(
    module_source: str,
    *,
    builder_name: str,
    builder_kwargs: Mapping[str, object],
    config_selector_name: str | None = None,
    pass_configs: Mapping[str, object] | None = None,
    compile_flags: Sequence[str] | str | None = None,
    dims: Mapping[str, object] | None = None,
    source_root: str | Path | None = None,
    env_file: str | Path | None = None,
    timeout_sec: int = 120,
) -> StageBResult:
    """Run Tier L1 TileLang + bisheng device compilation on Host CPU without NPU locks."""
    env_vals = _resolve_env_values(env_file)
    python_bin = (
        env_vals.get("TILEFOUNDRY_NPU_PYTHON")
        or os.environ.get("TILEFOUNDRY_NPU_PYTHON")
        or sys.executable
    )
    cann_setup = (
        env_vals.get("TILEFOUNDRY_NPU_CANN_SETUP")
        or os.environ.get("TILEFOUNDRY_NPU_CANN_SETUP")
        or ""
    ).strip()
    tl_source = (
        env_vals.get("TILEFOUNDRY_NPU_TILELANG_SOURCE")
        or os.environ.get("TILEFOUNDRY_NPU_TILELANG_SOURCE")
        or ""
    ).strip()
    if not cann_setup and not tl_source:
        return run_stage_b_compile(
            module_source,
            factory_name=builder_name,
            factory_kwargs=builder_kwargs,
            config_selector_name=config_selector_name,
            pass_configs=pass_configs,
            compile_flags=compile_flags,
            dims=dims,
            python_bin=python_bin,
            timeout_sec=timeout_sec,
        )

    t0 = time.monotonic()
    root = Path(source_root).resolve() if source_root is not None else default_source_root()
    pp_parts = [str(root / "src"), str(root / "targets" / "ascend")]
    if tl_source:
        pp_parts.insert(0, tl_source)
    if os.environ.get("PYTHONPATH"):
        pp_parts.append(os.environ["PYTHONPATH"])

    run_env = dict(os.environ)
    run_env["PYTHONPATH"] = os.pathsep.join(pp_parts)
    run_env["TILELANG_DISABLE_CACHE"] = "1"
    run_env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    run_env.setdefault("USE_CUDA", "0")

    with tempfile.TemporaryDirectory(prefix="tf_l1_compile_") as tmp_dir:
        tmp = Path(tmp_dir)
        cand_path = tmp / "candidate_dsl.py"
        cand_path.write_text(module_source, encoding="utf-8")
        kernel_dump_path = tmp / "lowered_kernel.cce"
        driver_path = tmp / "compile_driver.py"
        driver_path.write_text(
            build_stage_b_driver_source(
                cand_path,
                factory_name=builder_name,
                factory_kwargs=builder_kwargs,
                config_selector_name=config_selector_name,
                pass_configs=pass_configs,
                compile_flags=compile_flags,
                dims=dims,
                enable_device_compile=True,
                kernel_dump_path=kernel_dump_path,
            ),
            encoding="utf-8",
        )
        if cann_setup and Path(cann_setup).is_file() and shutil.which("bisheng") is None:
            bash_cmd = (
                f"set +u && source {shlex.quote(cann_setup)} >/dev/null 2>&1 "
                f"&& exec {shlex.quote(python_bin)} {shlex.quote(str(driver_path))}"
            )
            argv = ["/bin/bash", "-c", bash_cmd]
        else:
            argv = [python_bin, str(driver_path)]
        try:
            proc = subprocess.run(
                argv,
                text=True,
                capture_output=True,
                cwd=tmp_dir,
                env=run_env,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired:
            return StageBResult(
                passed=False,
                error_message=f"l1_compile_timeout:{timeout_sec}s",
                elapsed_sec=time.monotonic() - t0,
            )
        lowered_dump = (
            kernel_dump_path.read_text(encoding="utf-8", errors="ignore")
            if kernel_dump_path.is_file()
            else ""
        )
    combined_out = (proc.stdout or "") + "\n" + (proc.stderr or "")
    spill_bytes, vloopv2_hit, summary = _parse_cce_res_usage_with_summary(
        combined_out,
        asm_or_dump_text=lowered_dump,
    )
    ma_profile = (
        extract_microarch_profile_from_cce(lowered_dump).to_dict()
        if lowered_dump.strip()
        else None
    )
    if proc.returncode != 0:
        detail = _bounded_compile_diagnostic(proc.stderr or proc.stdout or "")
        return StageBResult(
            passed=False,
            error_message=f"l1_compile_failed (rc={proc.returncode}): {detail}",
            elapsed_sec=time.monotonic() - t0,
            spill_bytes=spill_bytes,
            vloopv2_hit=vloopv2_hit,
            compiler_summary=summary,
            microarch_profile=ma_profile,
        )
    return StageBResult(
        passed=True,
        elapsed_sec=time.monotonic() - t0,
        spill_bytes=spill_bytes,
        vloopv2_hit=vloopv2_hit,
        compiler_summary=summary,
        microarch_profile=ma_profile,
    )


async def run_l0_l1_round_filter(
    candidates: Sequence[tuple[str, str]],
    *,
    patch_fn: Callable[[str], str],
    l0_checker: Callable[[str], StageAResult],
    l1_compiler: Callable[[str], StageBResult],
    repair_backend: KernelLLMBackend | None = None,
    max_syntax_repairs: int = 1,
    slice_hint: str = "",
    backend_kind: str = "tilelang_dsl",
    stage_a_cost_ranker: Callable[
        [Sequence[tuple[str, str, StageAResult]]], Sequence[tuple[str, float]]
    ]
    | None = None,
    cost_prescreen_top_k: int | None = None,
) -> list[L0L1CandidateCheck]:
    """Filter candidate snippets through L0 static, optional CostModel prescreen, and L1 compile."""
    max_attempts = max(0, max_syntax_repairs)

    async def _try_repair(cur_src: str, diag: str) -> tuple[str | None, str, bool]:
        assert repair_backend is not None
        req = build_syntax_repair_request(
            cur_src,
            diag,
            slice_hint=slice_hint,
            backend_kind=backend_kind,
        )
        try:
            raw_reply = await repair_backend.generate(req)
        except Exception as exc:
            return None, f"{diag} | repair_transport: {type(exc).__name__}: {exc}", True
        try:
            repaired_src = extract_python_dsl(raw_reply)
        except Exception as exc:
            return None, f"{diag} | repair_parse: {type(exc).__name__}: {exc}", False
        return repaired_src, "", False

    async def _run_stage_a_phase(
        cand_id: str,
        cur_snippet: str,
        *,
        start_attempt: int = 0,
        repaired: bool = False,
    ) -> tuple[L0L1CandidateCheck | None, tuple[str, str, StageAResult, bool, int] | None]:
        snippet = cur_snippet
        is_repaired = repaired
        for attempt in range(start_attempt, max_attempts + 1):
            try:
                patched = patch_fn(snippet)
            except Exception as exc:
                err = f"ast_patch_error: {type(exc).__name__}: {exc}"
                if repair_backend is None or attempt >= max_attempts:
                    return (
                        L0L1CandidateCheck(
                            candidate_id=cand_id,
                            passed=False,
                            failed_tier="stage_a",
                            patched_module_source="",
                            snippet_source=snippet,
                            repaired=is_repaired,
                            error_message=err,
                        ),
                        None,
                    )
                next_snippet, rep_err, is_transport = await _try_repair(snippet, err)
                if next_snippet is None:
                    return (
                        L0L1CandidateCheck(
                            candidate_id=cand_id,
                            passed=False,
                            failed_tier="llm_transport" if is_transport else "stage_a",
                            patched_module_source="",
                            snippet_source=snippet,
                            repaired=is_repaired,
                            error_message=rep_err,
                        ),
                        None,
                    )
                snippet = next_snippet
                is_repaired = True
                continue

            stage_a = await asyncio.to_thread(l0_checker, patched)
            if not stage_a.passed:
                err = stage_a.error_message or "l0_static_failed"
                if repair_backend is None or attempt >= max_attempts:
                    return (
                        L0L1CandidateCheck(
                            candidate_id=cand_id,
                            passed=False,
                            failed_tier="stage_a",
                            patched_module_source=patched,
                            snippet_source=snippet,
                            repaired=is_repaired,
                            stage_a=stage_a,
                            error_message=err,
                        ),
                        None,
                    )
                next_snippet, rep_err, is_transport = await _try_repair(snippet, err)
                if next_snippet is None:
                    return (
                        L0L1CandidateCheck(
                            candidate_id=cand_id,
                            passed=False,
                            failed_tier="llm_transport" if is_transport else "stage_a",
                            patched_module_source=patched,
                            snippet_source=snippet,
                            repaired=is_repaired,
                            stage_a=stage_a,
                            error_message=rep_err,
                        ),
                        None,
                    )
                snippet = next_snippet
                is_repaired = True
                continue
            return None, (snippet, patched, stage_a, is_repaired, attempt)
        raise RuntimeError("unreachable repair termination")

    async def _run_stage_b_and_repairs(
        cand_id: str,
        snippet: str,
        patched: str,
        stage_a: StageAResult,
        repaired: bool,
        attempt: int,
    ) -> L0L1CandidateCheck:
        cur_snippet = snippet
        cur_patched = patched
        cur_stage_a = stage_a
        cur_repaired = repaired
        cur_attempt = attempt
        while True:
            stage_b = await asyncio.to_thread(l1_compiler, cur_patched)
            if stage_b.passed:
                return L0L1CandidateCheck(
                    candidate_id=cand_id,
                    passed=True,
                    failed_tier=None,
                    patched_module_source=cur_patched,
                    snippet_source=cur_snippet,
                    repaired=cur_repaired,
                    stage_a=cur_stage_a,
                    stage_b=stage_b,
                )
            err = stage_b.error_message or "l1_compile_failed"
            if repair_backend is None or cur_attempt >= max_attempts:
                return L0L1CandidateCheck(
                    candidate_id=cand_id,
                    passed=False,
                    failed_tier="stage_b",
                    patched_module_source=cur_patched,
                    snippet_source=cur_snippet,
                    repaired=cur_repaired,
                    stage_a=cur_stage_a,
                    stage_b=stage_b,
                    error_message=err,
                )
            next_snippet, rep_err, is_transport = await _try_repair(cur_snippet, err)
            if next_snippet is None:
                return L0L1CandidateCheck(
                    candidate_id=cand_id,
                    passed=False,
                    failed_tier="llm_transport" if is_transport else "stage_b",
                    patched_module_source=cur_patched,
                    snippet_source=cur_snippet,
                    repaired=cur_repaired,
                    stage_a=cur_stage_a,
                    stage_b=stage_b,
                    error_message=rep_err,
                )
            failed_chk, passed_state = await _run_stage_a_phase(
                cand_id,
                next_snippet,
                start_attempt=cur_attempt + 1,
                repaired=True,
            )
            if failed_chk is not None:
                return failed_chk
            assert passed_state is not None
            cur_snippet, cur_patched, cur_stage_a, cur_repaired, cur_attempt = passed_state

    async def _check_one(cand_id: str, snippet: str) -> L0L1CandidateCheck:
        failed_chk, passed_state = await _run_stage_a_phase(cand_id, snippet)
        if failed_chk is not None:
            return failed_chk
        assert passed_state is not None
        cur_snippet, patched, stage_a, repaired, attempt = passed_state
        return await _run_stage_b_and_repairs(
            cand_id, cur_snippet, patched, stage_a, repaired, attempt
        )

    use_prescreen = (
        stage_a_cost_ranker is not None
        and cost_prescreen_top_k is not None
        and int(cost_prescreen_top_k) > 0
    )
    if not use_prescreen:
        return list(await asyncio.gather(*(_check_one(cid, src) for cid, src in candidates)))

    assert stage_a_cost_ranker is not None
    top_k = int(cost_prescreen_top_k or 0)
    stage_a_outcomes = list(
        await asyncio.gather(*(_run_stage_a_phase(cid, src) for cid, src in candidates))
    )
    results_by_idx: dict[int, L0L1CandidateCheck] = {}
    passers: list[tuple[int, str, str, str, StageAResult, bool, int]] = []
    for idx, ((cid, _src), (failed_chk, passed_state)) in enumerate(
        zip(candidates, stage_a_outcomes)
    ):
        if failed_chk is not None:
            results_by_idx[idx] = failed_chk
        else:
            assert passed_state is not None
            cur_snippet, patched, stage_a, repaired, attempt = passed_state
            passers.append((idx, cid, cur_snippet, patched, stage_a, repaired, attempt))

    if len(passers) > top_k:
        rank_input = [(cid, patched, stage_a) for _idx, cid, _snip, patched, stage_a, _rep, _att in passers]
        ranked_pairs = list(stage_a_cost_ranker(rank_input))
        pred_by_id: dict[str, float] = {}
        order_by_id: dict[str, int] = {}
        for rank_idx, (rcid, rscore) in enumerate(ranked_pairs):
            sid = str(rcid)
            if sid not in pred_by_id:
                pred_by_id[sid] = float(rscore)
                order_by_id[sid] = rank_idx
        sorted_passers = sorted(
            passers,
            key=lambda item: (
                pred_by_id.get(item[1], float("inf")),
                order_by_id.get(item[1], len(passers) + item[0]),
                item[0],
            ),
        )
        kept_passers = sorted_passers[:top_k]
        for idx, cid, cur_snippet, patched, stage_a, repaired, _att in sorted_passers[top_k:]:
            pred_us = pred_by_id.get(cid, float("inf"))
            results_by_idx[idx] = L0L1CandidateCheck(
                candidate_id=cid,
                passed=False,
                failed_tier="cost_prescreen",
                patched_module_source=patched,
                snippet_source=cur_snippet,
                repaired=repaired,
                stage_a=stage_a,
                stage_b=None,
                error_message=f"cost_prescreen_pruned:pred_us={pred_us:.2f}",
            )
    else:
        kept_passers = passers

    if kept_passers:
        stage_b_checks = await asyncio.gather(
            *(
                _run_stage_b_and_repairs(cid, cur_snippet, patched, stage_a, repaired, attempt)
                for _idx, cid, cur_snippet, patched, stage_a, repaired, attempt in kept_passers
            )
        )
        for (idx, *_rest), chk in zip(kept_passers, stage_b_checks):
            results_by_idx[idx] = chk

    return [results_by_idx[i] for i in range(len(candidates))]


def _resolve_agent_gates_bin(source_root: Path | None = None) -> str:
    """Resolve the AgentGates CLI from explicit configuration or the installed entry point."""
    _ = source_root
    env_bin = os.environ.get("TILEFOUNDRY_NPU_AGENT_GATES_BIN", "").strip()
    if env_bin and Path(env_bin).is_file():
        return env_bin
    on_path = shutil.which("agent-gates")
    if on_path:
        return on_path
    raise FileNotFoundError(
        "Cannot locate 'agent-gates' binary. Set TILEFOUNDRY_NPU_AGENT_GATES_BIN or PATH."
    )


def _resolve_env_file_path(env_file: str | Path | None = None) -> Path:
    raw = (
        str(env_file)
        if env_file is not None
        else os.environ.get("TILEFOUNDRY_NPU_ENV_FILE", "").strip()
    )
    if raw and Path(raw).is_file():
        return Path(raw).resolve()
    raise FileNotFoundError("Valid tilefoundry-npu env.sh file is required for AgentGates runs.")


def _default_surrogate_cache_dir() -> Path:
    uid = os.getuid() if hasattr(os, "getuid") else 0
    return Path(tempfile.gettempdir()) / f"tf_surrogate_ref_cache_{uid}"


def _default_surrogate_tl_cache_dir() -> Path:
    uid = os.getuid() if hasattr(os, "getuid") else 0
    return Path(tempfile.gettempdir()) / f"tf_surrogate_tl_cache_{uid}"


def prewarm_anchor_reference_cache(
    operator_name: str,
    anchors: Sequence[AnchorCaseSpec],
    *,
    cannbench_repo: str | Path | None = None,
    env_file: str | Path | None = None,
    cache_dir: str | Path | None = None,
    source_root: str | Path | None = None,
) -> Path:
    """Populate CANNBench oracle and native reference cache on Host CPU without NPU locks."""
    root = Path(source_root).resolve() if source_root is not None else default_source_root()
    repo = resolve_cannbench_repo(cannbench_repo=cannbench_repo, env_file=env_file)
    resolved_cache = Path(cache_dir) if cache_dir is not None else _default_surrogate_cache_dir()
    resolved_cache.mkdir(parents=True, exist_ok=True)
    env_vals = parse_simple_env_file(env_file) if env_file is not None else {}
    python_bin = (
        env_vals.get("TILEFOUNDRY_NPU_PYTHON")
        or os.environ.get("TILEFOUNDRY_NPU_PYTHON")
        or sys.executable
    )

    run_env = dict(os.environ)
    run_env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    run_env["CANNBENCH_CACHE_DIR"] = str(resolved_cache)
    run_env["PYTHONDONTWRITEBYTECODE"] = "1"
    pp_parts = [str(root), str(root / "src"), str(root / "targets" / "ascend")]
    if run_env.get("PYTHONPATH"):
        pp_parts.append(run_env["PYTHONPATH"])
    run_env["PYTHONPATH"] = os.pathsep.join(pp_parts)

    case_ids = [int(a.case_num) for a in anchors]
    script = (
        "import json, sys\n"
        f"sys.path.insert(0, {str(root / 'targets' / 'ascend')!r})\n"
        "from tilefoundry_ascend.agent.campaign.anchor_specs import (\n"
        "    AnchorCaseSpec, load_anchor_harness_and_cases, load_cannbench_contract_module\n"
        ")\n"
        f"op_name = {operator_name!r}\n"
        f"case_ids = {json.dumps(case_ids)}\n"
        f"repo = {str(repo)!r}\n"
        f"root = {str(root)!r}\n"
        "contract = load_cannbench_contract_module(root)\n"
        "specs = [AnchorCaseSpec('', cid, '', 1.0, 1.0) for cid in case_ids]\n"
        "harness, cases = load_anchor_harness_and_cases(op_name, specs, cannbench_repo=repo, source_root=root)\n"
        "with contract.ReferenceCache(harness) as cache:\n"
        "    for case in cases:\n"
        "        def _prep(c=case, h=harness):\n"
        "            ts = h.prepare(c)\n"
        "            return {'inputs': ts, 'oracle': h.reference_output(c, ts)}\n"
        "        bundle = cache.get(case, 'oracle', _prep)\n"
        "        cache.get(case, 'native', lambda c=case, b=bundle, h=harness: h.reference_output(c, b['inputs'], native=True))\n"
    )
    proc = subprocess.run(
        [python_bin, "-c", script],
        text=True,
        capture_output=True,
        cwd=str(root),
        env=run_env,
        timeout=180,
    )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()[-1000:]
        raise RuntimeError(f"prewarm_anchor_reference_cache failed: {detail}")
    return resolved_cache


_PREWARM_TL_DRIVER_TEMPLATE = """import importlib.util
import inspect
import logging
import sys
import tilelang

sys.path.insert(0, {ascend_target_dir!r})
from tilefoundry_ascend.agent.campaign.anchor_specs import (
    get_default_regime_specs,
    resolve_regime_for_case,
)
from tilefoundry_ascend.agent.campaign.campaign_runner import (
    _HIR_DIM_ALIASES,
    _filter_callable_kwargs,
    resolve_regime_config,
)
from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (
    lookup_case_static_spec,
)

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s:%(name)s:%(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("tilefoundry_ascend.prewarm")

def _freeze(val):
    if isinstance(val, dict):
        return tuple((str(k), _freeze(v)) for k, v in sorted(val.items()))
    if isinstance(val, (list, tuple)):
        return tuple(_freeze(x) for x in val)
    if isinstance(val, set):
        return tuple(sorted(_freeze(x) for x in val))
    return val

op_name = {op_name!r}
case_ids = {case_ids_json}
cand_paths = {cand_paths_json}
repo = {repo!r}

regime_specs = get_default_regime_specs(op_name)
case_specs = []
for cid_num in case_ids:
    try:
        case_dims = lookup_case_static_spec(op_name, cid_num, cannbench_repo=repo)
        regime = resolve_regime_for_case(op_name, case_dims, regime_specs)
        case_specs.append((cid_num, case_dims, regime))
    except Exception as exc:
        logger.warning(
            "Prewarm static spec skipped for op=%s case=%s: %s: %s",
            op_name,
            cid_num,
            type(exc).__name__,
            exc,
        )

for idx, (cid, mod_path) in enumerate(cand_paths):
    try:
        mod_name = f"_tf_warm_{{idx}}"
        spec = importlib.util.spec_from_file_location(mod_name, mod_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load module spec from {{mod_path}}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception as exc:
        logger.warning(
            "Prewarm module load skipped for candidate=%s (%s): %s: %s",
            cid,
            mod_path,
            type(exc).__name__,
            exc,
        )
        continue

    seen_keys = set()
    get_kernel_fn = getattr(mod, "_get_kernel", None)
    combine_fn = getattr(mod, "_get_combine_kernel", None)

    for cid_num, case_dims, regime in case_specs:
        if getattr(regime, "runtime_backend", "") == "ascendc_trackb":
            continue
        try:
            resolved = resolve_regime_config(mod_path, regime, dims=case_dims)
            builder_fn = getattr(mod, resolved.builder_name, None)
            if not callable(builder_fn):
                raise ValueError(f"Builder {{resolved.builder_name!r}} not callable")

            pool = dict(regime.sample_builder_kwargs)
            for h_key, h_val in regime.representative_hir_dims.items():
                mapped = _HIR_DIM_ALIASES.get(str(h_key), str(h_key))
                pool.setdefault(mapped, h_val)
            for d_key, d_val in case_dims.items():
                mapped = _HIR_DIM_ALIASES.get(str(d_key), str(d_key))
                pool[mapped] = d_val
            raw_bk = {{}}
            raw_ck = {{}}
            sel_name = regime.config_selector_fn
            if sel_name and callable(getattr(mod, sel_name, None)):
                sel_fn = getattr(mod, sel_name)
                sel_out = sel_fn(**_filter_callable_kwargs(sel_fn, pool))
                if isinstance(sel_out, dict):
                    maybe_bk = sel_out.get("builder_kwargs")
                    if isinstance(maybe_bk, dict):
                        raw_bk = dict(maybe_bk)
                    maybe_ck = sel_out.get("combine_kwargs")
                    if isinstance(maybe_ck, dict):
                        raw_ck = dict(maybe_ck)
                    for sk, sv in sel_out.items():
                        if sk not in (
                            "pass_configs",
                            "compile_flags",
                            "builder_kwargs",
                            "combine_kwargs",
                        ):
                            pool[str(sk)] = sv
            pool.update(resolved.builder_kwargs)
            pool["builder_fn"] = builder_fn
            pool["pass_configs"] = dict(resolved.pass_configs)
            pool["compile_flags"] = tuple(resolved.compile_flags)
            pool["extra_builder_kwargs"] = dict(raw_bk) if raw_bk else None
            pool["extra_combine_kwargs"] = dict(raw_ck) if raw_ck else None

            used_get_kernel = False
            if callable(get_kernel_fn):
                gk_sig = inspect.signature(get_kernel_fn)
                has_req = all(
                    p.name in pool
                    for p in gk_sig.parameters.values()
                    if p.default is inspect.Parameter.empty
                    and p.kind not in (
                        inspect.Parameter.VAR_POSITIONAL,
                        inspect.Parameter.VAR_KEYWORD,
                    )
                )
                if has_req:
                    gk_kwargs = _filter_callable_kwargs(get_kernel_fn, pool)
                    sig_key = (
                        "_get_kernel",
                        resolved.builder_name,
                        tuple(
                            (k, _freeze(v))
                            for k, v in sorted(gk_kwargs.items())
                            if k != "builder_fn"
                        ),
                    )
                    if sig_key not in seen_keys:
                        seen_keys.add(sig_key)
                        get_kernel_fn(**gk_kwargs)
                    splits = int(pool.get("num_kv_splits", 1) or 1)
                    if splits > 1 and callable(combine_fn):
                        comb_kwargs = _filter_callable_kwargs(combine_fn, pool)
                        comb_key = (
                            "_get_combine_kernel",
                            tuple((k, _freeze(v)) for k, v in sorted(comb_kwargs.items())),
                        )
                        if comb_key not in seen_keys:
                            seen_keys.add(comb_key)
                            combine_fn(**comb_kwargs)
                    used_get_kernel = True

            if not used_get_kernel:
                generic_pool = dict(resolved.builder_kwargs)
                generic_pool.update(raw_bk)
                b_kwargs = _filter_callable_kwargs(builder_fn, generic_pool)
                pc_dict = dict(resolved.pass_configs)
                cf_tuple = tuple(resolved.compile_flags)
                sig_key = (
                    "generic",
                    resolved.builder_name,
                    tuple((k, _freeze(v)) for k, v in sorted(b_kwargs.items())),
                    _freeze(pc_dict),
                    cf_tuple,
                )
                if sig_key not in seen_keys:
                    seen_keys.add(sig_key)
                    prog = builder_fn(**b_kwargs)
                    tilelang.compile(
                        prog,
                        target="ascend",
                        pass_configs=pc_dict or None,
                        compile_flags=list(cf_tuple) if cf_tuple else None,
                    )
        except Exception as exc:
            logger.warning(
                "Prewarm compile skipped for candidate=%s case=%s: %s: %s",
                cid,
                cid_num,
                type(exc).__name__,
                exc,
            )
"""


def lookup_case_static_spec(
    operator_name: str,
    case_num: int,
    *,
    cannbench_repo: str | Path | None = None,
    env_file: str | Path | None = None,
) -> dict[str, object]:
    """Read static dims, tl_dtype, and attributes directly from cases.yaml."""
    repo = resolve_cannbench_repo(cannbench_repo=cannbench_repo, env_file=env_file)
    task = get_operator_task(operator_name)
    cases_path = resolve_cannbench_tasks_root(repo) / task / "cases.yaml"
    if not cases_path.is_file():
        raise FileNotFoundError(f"cases.yaml not found at {cases_path}")
    parsed = yaml.safe_load(cases_path.read_text(encoding="utf-8")) or {}
    cases_list = parsed.get("cases") or []
    target_id = int(case_num)
    entry: Mapping[str, object] | None = None
    for row in cases_list:
        if isinstance(row, Mapping) and int(row.get("case_id", -1)) == target_id:
            entry = row
            break
    if entry is None:
        raise KeyError(f"case_num={target_id} missing in {cases_path}")

    spec = lookup_case_dims(
        operator_name,
        target_id,
        cannbench_repo=repo,
        env_file=env_file,
    )
    raw_dtype = entry.get("dtype", "bfloat16")
    if isinstance(raw_dtype, (list, tuple)):
        tl_dtype = str(raw_dtype[0]) if raw_dtype else "bfloat16"
    else:
        tl_dtype = str(raw_dtype)
    tl_dtype = tl_dtype.strip() or "bfloat16"
    if tl_dtype.startswith("torch."):
        tl_dtype = tl_dtype[len("torch.") :]
    spec["tl_dtype"] = tl_dtype
    if "inputLayout" in spec and "input_layout" not in spec:
        spec["input_layout"] = str(spec["inputLayout"]).upper()
    if "is_causal" in spec:
        spec["is_causal"] = bool(spec["is_causal"])
    if (
        "B" in spec
        and "S" in spec
        and "m_rows" not in spec
        and isinstance(spec["B"], int)
        and isinstance(spec["S"], int)
    ):
        spec["m_rows"] = int(spec["B"]) * int(spec["S"])
    return spec


def prewarm_surrogate_tilelang_cache(
    operator_name: str,
    candidates: Sequence[tuple[str, str | Path]],
    anchors: Sequence[AnchorCaseSpec],
    *,
    cannbench_repo: str | Path | None = None,
    env_file: str | Path | None = None,
    cache_dir: str | Path | None = None,
    tl_cache_dir: str | Path | None = None,
    source_root: str | Path | None = None,
    timeout_sec: int = 300,
) -> Path:
    """Pre-compile candidate anchor kernels into TileLang disk cache on Host CPU."""
    root = Path(source_root).resolve() if source_root is not None else default_source_root()
    repo = resolve_cannbench_repo(cannbench_repo=cannbench_repo, env_file=env_file)
    resolved_ref_cache = Path(cache_dir) if cache_dir is not None else _default_surrogate_cache_dir()
    resolved_ref_cache.mkdir(parents=True, exist_ok=True)
    resolved_tl_cache = (
        Path(tl_cache_dir) if tl_cache_dir is not None else _default_surrogate_tl_cache_dir()
    )
    resolved_tl_cache.mkdir(parents=True, exist_ok=True)
    if not candidates or not anchors:
        return resolved_tl_cache

    env_vals = _resolve_env_values(env_file)
    python_bin = (
        env_vals.get("TILEFOUNDRY_NPU_PYTHON")
        or os.environ.get("TILEFOUNDRY_NPU_PYTHON")
        or sys.executable
    )
    cann_setup = (
        env_vals.get("TILEFOUNDRY_NPU_CANN_SETUP")
        or os.environ.get("TILEFOUNDRY_NPU_CANN_SETUP")
        or ""
    ).strip()
    tl_source = (
        env_vals.get("TILEFOUNDRY_NPU_TILELANG_SOURCE")
        or os.environ.get("TILEFOUNDRY_NPU_TILELANG_SOURCE")
        or ""
    ).strip()

    run_env = dict(os.environ)
    run_env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    run_env["ASCEND_RT_VISIBLE_DEVICES"] = ""
    run_env["USE_CUDA"] = "0"
    run_env["CANNBENCH_CACHE_DIR"] = str(resolved_ref_cache)
    run_env["TILELANG_CACHE_DIR"] = str(resolved_tl_cache)
    run_env["TILELANG_KERNEL_CACHE_USE_LIB_STAMP"] = "1"
    run_env["TILELANG_DISABLE_CACHE"] = "0"
    run_env["PYTHONDONTWRITEBYTECODE"] = "1"
    pp_parts = [str(root), str(root / "src"), str(root / "targets" / "ascend")]
    if tl_source:
        pp_parts.insert(0, tl_source)
    if run_env.get("PYTHONPATH"):
        pp_parts.append(run_env["PYTHONPATH"])
    run_env["PYTHONPATH"] = os.pathsep.join(pp_parts)

    case_ids = [int(a.case_num) for a in anchors]
    cand_paths = [(str(cid), str(Path(p).resolve())) for cid, p in candidates]

    with tempfile.TemporaryDirectory(prefix="tf_tl_prewarm_") as tmp_dir:
        driver_path = Path(tmp_dir) / "prewarm_tl_driver.py"
        driver_path.write_text(
            _PREWARM_TL_DRIVER_TEMPLATE.format(
                ascend_target_dir=str(root / "targets" / "ascend"),
                op_name=operator_name,
                case_ids_json=json.dumps(case_ids),
                cand_paths_json=json.dumps(cand_paths),
                repo=str(repo),
            ),
            encoding="utf-8",
        )
        if cann_setup and Path(cann_setup).is_file() and shutil.which("bisheng") is None:
            bash_cmd = (
                f"set +u && source {shlex.quote(cann_setup)} >/dev/null 2>&1 "
                f"&& exec {shlex.quote(python_bin)} {shlex.quote(str(driver_path))}"
            )
            argv = ["/bin/bash", "-c", bash_cmd]
        else:
            argv = [python_bin, str(driver_path)]
        proc = subprocess.run(
            argv,
            text=True,
            capture_output=True,
            cwd=str(root),
            env=run_env,
            timeout=timeout_sec,
        )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()[-1000:]
        raise RuntimeError(f"prewarm_surrogate_tilelang_cache failed: {detail}")
    if proc.stderr and proc.stderr.strip():
        logging.getLogger(__name__).warning(
            "prewarm_surrogate_tilelang_cache warnings: %s",
            proc.stderr.strip()[-1000:],
        )
    return resolved_tl_cache


_LLM_API_KEY_ENV = "TILEFOUNDRY_NPU_LLM_API_KEY_ENV"
_NON_AGENT_GATES_ENV_KEYS = frozenset(
    {"TILEFOUNDRY_NPU_AGENT_GATES_BIN", "TILEFOUNDRY_NPU_LLM_API_KEY"}
)


def _validate_llm_api_key_env(name: str) -> None:
    if name and (
        not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name)
        or name.startswith("TILEFOUNDRY_NPU_")
    ):
        raise ValueError(f"{_LLM_API_KEY_ENV} must name a separate environment variable")


def _format_env_line(key: str, value: str) -> str:
    if any(char in value for char in "\n\r\x00"):
        raise ValueError(f"{key} must be single-line")
    return f"{key}={shlex.quote(value)}"


def _write_temp_env_file(
    base_env: Path,
    dest_path: Path,
    overrides: Mapping[str, str],
) -> Path:
    values = parse_simple_env_file(base_env)
    values.update(overrides)
    _validate_llm_api_key_env(values.get(_LLM_API_KEY_ENV, "").strip())
    stripped_keys = set(overrides) | _NON_AGENT_GATES_ENV_KEYS
    lines = [
        line
        for line in base_env.read_text(encoding="utf-8").splitlines()
        if line.partition("=")[0].strip().removeprefix("export ").strip() not in stripped_keys
    ]
    for key, value in overrides.items():
        if key in _NON_AGENT_GATES_ENV_KEYS:
            continue
        lines.append(_format_env_line(key, value))
    dest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dest_path


_STRIPPED_HOST_CPU_ENV_KEYS = frozenset(
    {
        "ASCEND_RT_VISIBLE_DEVICES",
        "TILELANG_DISABLE_CACHE",
        "TORCH_DEVICE_BACKEND_AUTOLOAD",
        "USE_CUDA",
    }
)


def _clean_agent_gates_subprocess_env(extra_env: Mapping[str, str]) -> dict[str, str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("TILEFOUNDRY_NPU_") and k not in _STRIPPED_HOST_CPU_ENV_KEYS
    }
    env.setdefault("PYTHONNOUSERSITE", "1")
    env.update(extra_env)
    return env


def _parse_agent_gates_json(stdout: str) -> dict[str, object]:
    text = (stdout or "").strip()
    for line in reversed(text.splitlines()):
        cand = line.strip()
        if cand.startswith("{") and cand.endswith("}"):
            try:
                parsed = json.loads(cand)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
    idx = text.rfind("\n{")
    start = idx + 1 if idx >= 0 else text.find("{")
    if start < 0:
        return {}
    try:
        parsed = json.loads(text[start:])
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _is_transient_blocked_env(
    proc: subprocess.CompletedProcess[str],
    parsed: Mapping[str, object],
) -> bool:
    if proc.returncode == 0:
        return False
    if parsed.get("status") == "BLOCKED_ENV":
        return True
    if parsed.get("status") == "FAIL":
        gates = parsed.get("gates")
        if isinstance(gates, Sequence) and not isinstance(gates, (str, bytes)):
            for gate in gates:
                if not isinstance(gate, Mapping):
                    continue
                failures = gate.get("failures")
                if (
                    gate.get("gateId") == "tilefoundry-npu-device-fast"
                    and gate.get("status") == "NOT_RUN"
                    and isinstance(failures, Sequence)
                    and not isinstance(failures, (str, bytes))
                    and "DEPENDENCY_NOT_PASSED" in failures
                ):
                    return True
    return False


def _run_agent_gates_cmd(
    cmd: Sequence[str],
    *,
    cwd: str,
    env: Mapping[str, str],
    timeout_sec: int,
    max_attempts: int = DEFAULT_AGENT_GATES_MAX_ATTEMPTS,
    backoff_sec: float = DEFAULT_AGENT_GATES_BACKOFF_SEC,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
    wait_for_npu_quiescent()
    attempts = max(1, int(max_attempts))
    cmd_list = [str(part) for part in cmd]
    env_dict = dict(env)
    last_proc: subprocess.CompletedProcess[str] | None = None
    last_parsed: dict[str, object] = {}
    for attempt in range(1, attempts + 1):
        last_proc = subprocess.run(
            cmd_list,
            text=True,
            capture_output=True,
            cwd=cwd,
            env=env_dict,
            timeout=timeout_sec,
        )
        last_parsed = _parse_agent_gates_json(last_proc.stdout)
        if not (_is_transient_blocked_env(last_proc, last_parsed) and attempt < attempts):
            return last_proc, last_parsed
        delay = min(20.0, max(0.0, float(backoff_sec)) * (2 ** (attempt - 1)))
        logger.warning(
            "AgentGates transient BLOCKED_ENV (attempt %d/%d); retrying in %.1fs",
            attempt,
            attempts,
            delay,
        )
        if delay > 0.0:
            sleep_fn(delay)
    assert last_proc is not None
    return last_proc, last_parsed


def run_multi_operator_surrogate_batch_via_agent_gates(
    batches: Sequence[tuple[str, Sequence[tuple[str, str | Path]], Sequence[AnchorCaseSpec]]],
    *,
    source_root: str | Path | None = None,
    env_file: str | Path | None = None,
    device_id: str = "1",
    warmup: int = 3,
    repeat: int = 5,
    require_all_passed: bool = False,
    timeout_sec: int = 300,
    prewarm_cache: bool = True,
) -> tuple[dict[str, dict[str, CandidateSurrogateResult]], str, str, float]:
    """Evaluate candidate batches across operators inside one AgentGates device lease."""
    root = Path(source_root).resolve() if source_root is not None else default_source_root()
    resolved_env = _resolve_env_file_path(env_file)
    ag_bin = _resolve_agent_gates_bin(root)
    env_vals = parse_simple_env_file(resolved_env)
    cann_repo = resolve_cannbench_repo(
        cannbench_repo=env_vals.get("TILEFOUNDRY_NPU_CANNBENCH_REPO"),
        env_file=resolved_env,
    )
    cache_dir, tl_cache_dir = _fast_cache_roots(root, env_vals)

    if prewarm_cache:
        for op_name, cands, op_anchors in batches:
            prewarm_anchor_reference_cache(
                op_name,
                op_anchors,
                cannbench_repo=cann_repo,
                env_file=resolved_env,
                cache_dir=cache_dir,
                source_root=root,
            )
            prewarm_surrogate_tilelang_cache(
                op_name,
                cands,
                op_anchors,
                cannbench_repo=cann_repo,
                env_file=resolved_env,
                cache_dir=cache_dir,
                tl_cache_dir=tl_cache_dir,
                source_root=root,
            )

    request_root = _fast_request_root(root, env_vals)
    request_id = uuid.uuid4().hex
    request_dir = request_root / request_id
    request_path = request_root / f"{request_id}.request.json"
    response_path = request_root / f"{request_id}.response.json"

    staged: dict[str, dict[str, dict[str, str]]] = {}
    request_batches: list[dict[str, object]] = []
    seen_operators: set[str] = set()
    for op_name, cands, op_anchors in batches:
        if op_name in seen_operators:
            raise ValueError(f"duplicate operator_name {op_name!r} in one fast batch")
        seen_operators.add(op_name)
        bindings: dict[str, dict[str, str]] = {}
        entries: list[dict[str, str]] = []
        for index, (candidate_id, module_path) in enumerate(cands):
            if candidate_id in bindings:
                raise ValueError(
                    f"duplicate candidate_id {candidate_id!r} for operator {op_name!r}"
                )
            relative, digest = _stage_fast_candidate(
                request_dir,
                root,
                op_name,
                index,
                candidate_id,
                Path(module_path),
            )
            bindings[candidate_id] = {"module_path": relative, "sha256": digest}
            entries.append(
                {"candidate_id": candidate_id, "module_path": relative, "sha256": digest}
            )
        staged[op_name] = bindings
        request_batches.append(
            {
                "operator_name": op_name,
                "entrypoint": get_operator_entrypoint(op_name),
                "candidates": entries,
                "anchors": [a.to_dict() for a in op_anchors],
            }
        )

    payload: dict[str, object] = {
        "schema": FAST_REQUEST_SCHEMA,
        "request_id": request_id,
        "mode": "batch",
        "warmup": int(warmup),
        "repeat": int(repeat),
        "require_all_passed": bool(require_all_passed),
        "response_path": response_path.relative_to(root).as_posix(),
        "batches": request_batches,
        "msprof_targets": [],
    }
    request_sha256 = _write_fast_request(request_path, payload)
    workspace = _agent_gates_workspace_name(root)
    raw_resp, execution = _submit_fast_gate(
        agent_gates_bin=ag_bin,
        root=root,
        workspace=workspace,
        request_id=request_id,
        request_relative_path=request_path.relative_to(root).as_posix(),
        request_path=request_path,
        response_path=response_path,
        request_sha256=request_sha256,
        mode="batch",
        timeout_sec=float(timeout_sec),
        device_id=str(device_id),
    )
    _verify_fast_bindings(raw_resp, staged)
    _verify_fast_coverage(
        raw_resp,
        expected_candidates=sum(len(cands) for _op, cands, _anchors in batches),
        expected_msprof_targets=0,
    )
    operators_raw = _fast_response_mapping(raw_resp, "operators")
    out_by_op: dict[str, dict[str, CandidateSurrogateResult]] = {}
    failures: list[str] = []
    for op_name, cands, _anchors in batches:
        cands_raw = operators_raw.get(op_name)
        if not isinstance(cands_raw, Mapping):
            raise RuntimeError(f"fast response has no results for operator {op_name!r}")
        parsed_candidates: dict[str, CandidateSurrogateResult] = {}
        for candidate_id, _module_path in cands:
            item = cands_raw.get(candidate_id)
            if not isinstance(item, Mapping):
                raise RuntimeError(
                    f"fast response has no result for {op_name}/{candidate_id}"
                )
            result = CandidateSurrogateResult.from_dict(item)
            parsed_candidates[candidate_id] = result
            if not result.all_passed:
                failures.append(f"{op_name}/{candidate_id}")
        out_by_op[op_name] = parsed_candidates
    if require_all_passed and failures:
        raise RuntimeError(
            "AgentGates L2 batch rejected candidate(s): " + ", ".join(failures)
        )
    return (
        out_by_op,
        execution.run_id,
        execution.receipt_path,
        _fast_held_seconds(execution.report),
    )


def run_surrogate_batch_via_agent_gates(
    operator_name: str,
    candidates: Sequence[tuple[str, str | Path]],
    anchors: Sequence[AnchorCaseSpec] | None = None,
    *,
    source_root: str | Path | None = None,
    env_file: str | Path | None = None,
    device_id: str = "1",
    warmup: int = 3,
    repeat: int = 5,
    require_all_passed: bool = False,
    timeout_sec: int = 300,
) -> tuple[dict[str, CandidateSurrogateResult], str, str]:
    """Evaluate all L1-passing candidates of one round inside a single AgentGates window."""
    active_anchors = tuple(anchors) if anchors is not None else get_default_anchors(operator_name)
    by_op, run_id, receipt_path, _held = run_multi_operator_surrogate_batch_via_agent_gates(
        [(operator_name, candidates, active_anchors)],
        source_root=source_root,
        env_file=env_file,
        device_id=device_id,
        warmup=warmup,
        repeat=repeat,
        require_all_passed=require_all_passed,
        timeout_sec=timeout_sec,
    )
    return by_op.get(operator_name, {}), run_id, receipt_path


def bottleneck_anchor_for_regime(
    result: CandidateSurrogateResult,
    regime_tag: str,
) -> AnchorEvalMetrics | None:
    """Return the lowest-Eq.3-score evaluated anchor matching regime_tag."""
    matched = [a for a in result.anchors if a.regime_tag == regime_tag]
    if not matched:
        return None
    return min(matched, key=lambda a: (a.eq3_score, -a.cand_us, a.case_num))


def collect_bottleneck_msprof_via_agent_gates(
    operator_name: str,
    candidate_path: str | Path,
    anchors_by_regime: Sequence[AnchorCaseSpec] | Mapping[str, AnchorCaseSpec],
    *,
    source_root: str | Path | None = None,
    env_file: str | Path | None = None,
    device_id: str = "1",
    warmup: int = 1,
    repeat: int = 3,
    timeout_sec: int = 300,
) -> dict[str, dict[str, float]]:
    """Collect msprof_detail telemetry for bottleneck anchors inside an AgentGates lease."""
    if isinstance(anchors_by_regime, Mapping):
        regime_items = [(str(tag), spec) for tag, spec in anchors_by_regime.items()]
    else:
        regime_items = [(spec.regime_tag, spec) for spec in anchors_by_regime]
    if not regime_items:
        return {}

    unique_case_nums: list[int] = []
    seen_cases: set[int] = set()
    for _tag, spec in regime_items:
        cnum = int(spec.case_num)
        if cnum not in seen_cases:
            seen_cases.add(cnum)
            unique_case_nums.append(cnum)

    root = Path(source_root).resolve() if source_root is not None else default_source_root()
    resolved_env = _resolve_env_file_path(env_file)
    ag_bin = _resolve_agent_gates_bin(root)
    env_vals = parse_simple_env_file(resolved_env)
    cann_repo = resolve_cannbench_repo(
        cannbench_repo=env_vals.get("TILEFOUNDRY_NPU_CANNBENCH_REPO"),
        env_file=resolved_env,
    )
    cache_dir, tl_cache_dir = _fast_cache_roots(root, env_vals)
    cand_resolved = Path(candidate_path).resolve()

    unique_specs = [
        next(spec for _t, spec in regime_items if int(spec.case_num) == cnum)
        for cnum in unique_case_nums
    ]
    prewarm_anchor_reference_cache(
        operator_name,
        unique_specs,
        cannbench_repo=cann_repo,
        env_file=resolved_env,
        cache_dir=cache_dir,
        source_root=root,
    )
    prewarm_surrogate_tilelang_cache(
        operator_name,
        [("bottleneck_probe", cand_resolved)],
        unique_specs,
        cannbench_repo=cann_repo,
        env_file=resolved_env,
        cache_dir=cache_dir,
        tl_cache_dir=tl_cache_dir,
        source_root=root,
    )

    request_root = _fast_request_root(root, env_vals)
    request_id = uuid.uuid4().hex
    request_dir = request_root / request_id
    request_path = request_root / f"{request_id}.request.json"
    response_path = request_root / f"{request_id}.response.json"
    relative, digest = _stage_fast_candidate(
        request_dir,
        root,
        operator_name,
        0,
        "bottleneck_probe",
        cand_resolved,
    )
    payload: dict[str, object] = {
        "schema": FAST_REQUEST_SCHEMA,
        "request_id": request_id,
        "mode": "profile",
        "warmup": int(warmup),
        "repeat": int(repeat),
        "require_all_passed": False,
        "response_path": response_path.relative_to(root).as_posix(),
        "batches": [],
        "msprof_targets": [
            {
                "operator_name": operator_name,
                "candidate_id": "bottleneck_probe",
                "module_path": relative,
                "sha256": digest,
                "case_num": cnum,
                "warmup": int(warmup),
                "repeat": int(repeat),
            }
            for cnum in unique_case_nums
        ],
    }
    request_sha256 = _write_fast_request(request_path, payload)
    workspace = _agent_gates_workspace_name(root)
    raw_resp, _execution = _submit_fast_gate(
        agent_gates_bin=ag_bin,
        root=root,
        workspace=workspace,
        request_id=request_id,
        request_relative_path=request_path.relative_to(root).as_posix(),
        request_path=request_path,
        response_path=response_path,
        request_sha256=request_sha256,
        mode="profile",
        timeout_sec=float(timeout_sec),
        device_id=str(device_id),
    )
    _verify_fast_bindings(
        raw_resp,
        {operator_name: {"bottleneck_probe": {"module_path": relative, "sha256": digest}}},
    )
    _verify_fast_coverage(
        raw_resp,
        expected_candidates=1,
        expected_msprof_targets=len(unique_case_nums),
    )
    telemetry_raw = _fast_response_mapping(raw_resp, "msprof_telemetry").get(operator_name)
    by_case = (
        telemetry_raw.get("bottleneck_probe") if isinstance(telemetry_raw, Mapping) else None
    )
    if not isinstance(by_case, Mapping):
        raise RuntimeError(
            f"fast profile response has no msprof telemetry for {operator_name!r}"
        )
    out: dict[str, dict[str, float]] = {}
    for tag, spec in regime_items:
        raw_metrics = by_case.get(str(int(spec.case_num)))
        if isinstance(raw_metrics, Mapping) and "error" not in raw_metrics:
            out[tag] = {
                str(k): float(v)
                for k, v in raw_metrics.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            }
    return out


_AGENT_GATES_CONFIG_ENV = "AGENT_GATES_CONFIG"


def _agent_gates_config_path() -> Path:
    configured = os.environ.get(_AGENT_GATES_CONFIG_ENV, "").strip()
    return (
        Path(configured).expanduser()
        if configured
        else Path.home() / ".config/agent-gates/config.toml"
    )


def _agent_gates_config() -> Mapping[str, object]:
    return tomllib.loads(_agent_gates_config_path().read_text(encoding="utf-8"))


def _agent_gates_workspace_name(source_root: Path) -> str:
    """Resolve the configured V2 workspace whose source root is ``source_root``."""
    raw = _agent_gates_config()
    workspaces = raw.get("workspaces")
    if not isinstance(workspaces, Mapping):
        raise ValueError("AgentGates config must define [workspaces]")
    wanted = source_root.resolve()
    for name, entry in workspaces.items():
        if not isinstance(entry, Mapping):
            continue
        configured = entry.get("source_root")
        if isinstance(configured, str) and Path(configured).expanduser().resolve() == wanted:
            return str(name)
    raise ValueError(f"No AgentGates workspace is configured for source root {wanted}")


def _workspace_relative_candidate(source_root: Path, candidate_path: Path) -> str | None:
    try:
        relative = candidate_path.resolve().relative_to(source_root.resolve())
    except ValueError:
        return None
    return relative.as_posix()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class _V2GateExecution:
    """Outcome of one V2 submit/wait/report/verify sequence."""

    execution_id: str = ""
    run_id: str = ""
    receipt_path: str = ""
    report: Mapping[str, object] = field(default_factory=dict)
    verification: Mapping[str, object] = field(default_factory=dict)
    transport_error: str = ""
    elapsed_sec: float = 0.0
    device_id: str = ""
    state: str = ""
    execution_terminal: bool = True

    @property
    def execution_active(self) -> bool:
        """Return whether an execution may still be running and must be resumed."""
        return bool(self.execution_id) and not self.execution_terminal

    @property
    def verified(self) -> bool:
        """Return whether verify-requirements proved this run against current sources."""
        return (
            not self.transport_error
            and self.execution_id != ""
            and self.verification.get("satisfied") is True
        )


def _agent_gates_cli_env() -> dict[str, str]:
    """Build the CLI environment without project Python startup injection."""
    env = _clean_agent_gates_subprocess_env({})
    for key in _V2_STRIPPED_PYTHON_ENV_KEYS:
        env.pop(key, None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def _gate_row(report: Mapping[str, object], gate_id: str) -> Mapping[str, object] | None:
    for row in report.get("gates", ()):
        if isinstance(row, Mapping) and row.get("gateId") == gate_id:
            return row
    return None


def _gate_observation(
    report: Mapping[str, object], gate_id: str
) -> tuple[Mapping[str, object] | None, Mapping[str, object]]:
    row = _gate_row(report, gate_id)
    if row is None:
        return None, {}
    obs = row.get("observations")
    return row, dict(obs) if isinstance(obs, Mapping) else {}


def _gate_recorded_status(row: Mapping[str, object] | None) -> str:
    if row is None:
        return ""
    return str(row.get("recordedStatus") or row.get("status") or "")


def _as_positive_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0
    return max(0, int(value))


def _run_v2_gate(
    *,
    agent_gates_bin: str,
    root: Path,
    workspace: str,
    gate_id: str,
    target: str,
    candidate_path: str | None = None,
    timeout_sec: float,
    device_id: str = "",
    resume_execution_id: str = "",
    resume_run_id: str = "",
    on_execution_started: Callable[[str, str], None] | None = None,
) -> _V2GateExecution:
    """Run one V2 gate with a single monotonic deadline and one execution identity.

    A new submission first diagnoses the same route with the same target and
    candidate, so a blocked machine or an unusable pair is refused before an
    execution exists; a resumed execution goes straight to its wait and recovers
    the submitted run by its IDs. A bounded local wait slice continues the same
    execution; a terminal state wins even at the ``timed_out`` boundary.

    Note:
        An unproven terminal state stays recoverable. The submission callback
        persists the identity immediately so a killed campaign recovers it, and
        verification binds the receipt to the requested target and candidate;
        ``device_id`` is part of the execution identity and a resume keeps its
        card.
    """
    started = time.monotonic()
    deadline = started + max(1.0, float(timeout_sec))
    requested_device = str(device_id or "").strip()
    submit_cmd = [
        agent_gates_bin,
        "submit",
        "--workspace",
        workspace,
        "--adapter",
        AGENT_GATES_ADAPTER,
        "--gate",
        gate_id,
        "--target",
        target,
        "--json",
    ]
    if candidate_path is not None:
        submit_cmd.extend(["--candidate", candidate_path])
    if requested_device:
        submit_cmd.extend(["--device-id", requested_device])

    def _failed(
        execution_id: str,
        message: str,
        *,
        run_id: str = "",
        receipt_path: str = "",
        resolved_device_id: str = "",
        state: str = "",
        execution_terminal: bool = True,
    ) -> _V2GateExecution:
        return _V2GateExecution(
            execution_id=execution_id,
            run_id=run_id,
            receipt_path=receipt_path,
            transport_error=message[-1000:],
            elapsed_sec=time.monotonic() - started,
            device_id=resolved_device_id or requested_device,
            state=state,
            execution_terminal=execution_terminal,
        )

    if resume_execution_id:
        execution_id = resume_execution_id
        resolved_device = requested_device
    else:
        doctor_cmd = [
            agent_gates_bin,
            "doctor",
            "--adapter",
            AGENT_GATES_ADAPTER,
            "--source",
            str(root),
            "--gate",
            gate_id,
            "--target",
            target,
            "--json",
        ]
        if candidate_path is not None:
            doctor_cmd.extend(["--candidate", candidate_path])
        if requested_device:
            doctor_cmd.extend(["--device-id", requested_device])
        try:
            doctor_proc, doctor = _run_agent_gates_v2_cmd(
                doctor_cmd,
                root=root,
                timeout_sec=min(_V2_SUBMIT_TIMEOUT_SEC, max(1.0, deadline - started)),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return _failed("", f"{gate_id} doctor failed: {exc}")
        if doctor_proc.returncode != 0 or doctor.get("status") != "PASS":
            raw_failures = doctor.get("failures")
            failures = (
                "; ".join(str(item) for item in raw_failures)
                if isinstance(raw_failures, list)
                else ""
            )
            return _failed(
                "",
                failures
                or doctor_proc.stderr
                or doctor_proc.stdout
                or f"{gate_id} doctor did not pass for target {target}",
            )
        try:
            proc, parsed = _run_agent_gates_v2_cmd(
                submit_cmd,
                root=root,
                timeout_sec=min(_V2_SUBMIT_TIMEOUT_SEC, max(1.0, deadline - started)),
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return _failed("", f"{gate_id} submit failed: {exc}")
        if proc.returncode != 0:
            return _failed("", proc.stderr or proc.stdout or f"{gate_id} submit failed")
        execution_id = str(parsed.get("execution_id", ""))
        if not execution_id:
            return _failed("", f"{gate_id} submit returned no execution_id")
        resolved_device = str(
            parsed.get("device_id") or parsed.get("deviceId") or requested_device or ""
        )
        if on_execution_started is not None:
            on_execution_started(execution_id, resolved_device)

    run_id = resume_run_id
    last_state = ""
    if not run_id:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                terminal = last_state in {"FINISHED", "CANCELLED", "INTERRUPTED"}
                return _failed(
                    execution_id,
                    f"{gate_id} execution {execution_id} wait timed out",
                    resolved_device_id=resolved_device,
                    state=last_state or "RUNNING",
                    execution_terminal=terminal,
                )
            wait_seconds = min(_V2_WAIT_SLICE_SEC, remaining)
            wait_cmd = [
                agent_gates_bin,
                "wait",
                "--execution-id",
                execution_id,
                "--timeout",
                f"{wait_seconds:g}",
                "--json",
            ]
            try:
                proc, parsed = _run_agent_gates_v2_cmd(
                    wait_cmd,
                    root=root,
                    timeout_sec=wait_seconds + 5.0,
                )
            except subprocess.TimeoutExpired:
                continue
            except OSError as exc:
                return _failed(
                    execution_id,
                    f"{gate_id} execution {execution_id} wait failed: {exc}",
                    resolved_device_id=resolved_device,
                    state=last_state,
                    execution_terminal=False,
                )
            if proc.returncode != 0:
                return _failed(
                    execution_id,
                    proc.stderr or proc.stdout or f"{gate_id} execution {execution_id} wait failed",
                    resolved_device_id=resolved_device,
                    state=last_state,
                    execution_terminal=False,
                )
            status = parsed.get("status")
            state = str(status.get("state") or "") if isinstance(status, Mapping) else ""
            if state:
                last_state = state
            if state == "FINISHED":
                run_id = str(status.get("run_id") or "")
                break
            if state in {"CANCELLED", "INTERRUPTED"}:
                return _failed(
                    execution_id,
                    f"{gate_id} execution {execution_id} ended in state {state!r}",
                    resolved_device_id=resolved_device,
                    state=state,
                    execution_terminal=True,
                )
            if parsed.get("timed_out") is True:
                continue
            if state is None or not state:
                return _failed(
                    execution_id,
                    f"{gate_id} execution {execution_id} wait returned no status",
                    resolved_device_id=resolved_device,
                    state=last_state,
                    execution_terminal=False,
                )
            if state != "FINISHED":
                return _failed(
                    execution_id,
                    f"{gate_id} execution {execution_id} ended in state {state!r}",
                    resolved_device_id=resolved_device,
                    state=state,
                    execution_terminal=True,
                )
            run_id = str(status.get("run_id") or "")
            break
    if not run_id:
        return _failed(
            execution_id,
            f"{gate_id} execution {execution_id} finished without run_id",
            resolved_device_id=resolved_device,
            state=last_state or "FINISHED",
            execution_terminal=True,
        )

    try:
        evidence_root = _agent_gates_evidence_root()
    except (OSError, ValueError) as exc:
        return _failed(
            execution_id,
            f"{gate_id} evidence root unavailable: {exc}",
            run_id=run_id,
            resolved_device_id=resolved_device,
            state=last_state or "FINISHED",
        )
    report_cmd = [
        agent_gates_bin,
        "report",
        "--run-id",
        run_id,
        "--evidence-root",
        str(evidence_root),
        "--json",
    ]
    post_wait = min(_V2_POST_WAIT_TIMEOUT_SEC, max(1.0, deadline - time.monotonic()))
    try:
        proc, report = _run_agent_gates_v2_cmd(report_cmd, root=root, timeout_sec=post_wait)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return _failed(
            execution_id,
            f"{gate_id} report failed: {exc}",
            run_id=run_id,
            resolved_device_id=resolved_device,
            state=last_state or "FINISHED",
        )
    if not report or "receiptPath" not in report:
        return _failed(
            execution_id,
            proc.stderr or proc.stdout or f"{gate_id} report returned no receipt",
            run_id=run_id,
            resolved_device_id=resolved_device,
            state=last_state or "FINISHED",
        )
    receipt_path = str(report.get("receiptPath", ""))

    verify_cmd = [
        agent_gates_bin,
        "verify-requirements",
        "--workspace",
        workspace,
        "--adapter",
        AGENT_GATES_ADAPTER,
        "--gate",
        gate_id,
        "--run-id",
        run_id,
        "--json",
    ]
    if candidate_path is not None:
        verify_cmd.extend(["--target", target, "--candidate", candidate_path])
    post_wait = min(_V2_POST_WAIT_TIMEOUT_SEC, max(1.0, deadline - time.monotonic()))
    try:
        proc, verification = _run_agent_gates_v2_cmd(verify_cmd, root=root, timeout_sec=post_wait)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return _failed(
            execution_id,
            f"{gate_id} requirement verification failed: {exc}",
            run_id=run_id,
            receipt_path=receipt_path,
            resolved_device_id=resolved_device,
            state=last_state or "FINISHED",
        )
    if not verification or "satisfied" not in verification:
        return _failed(
            execution_id,
            proc.stderr or proc.stdout or f"{gate_id} requirement verification returned no verdict",
            run_id=run_id,
            receipt_path=receipt_path,
            resolved_device_id=resolved_device,
            state=last_state or "FINISHED",
        )
    return _V2GateExecution(
        execution_id=execution_id,
        run_id=run_id,
        receipt_path=receipt_path,
        report=report,
        verification=verification,
        elapsed_sec=time.monotonic() - started,
        device_id=resolved_device,
        state=last_state or "FINISHED",
        execution_terminal=True,
    )


_FAST_ID_SANITIZER = re.compile(r"[^A-Za-z0-9_.-]+")


def _fast_slug(value: str, *, fallback: str) -> str:
    slug = _FAST_ID_SANITIZER.sub("_", value).strip("._")
    return slug or fallback


def _fast_request_root(root: Path, env_vals: Mapping[str, str]) -> Path:
    """Resolve the request staging root under the AgentGates lease-managed build directory."""
    build_subdir = env_vals.get("TILEFOUNDRY_NPU_BUILD_SUBDIR", "").strip() or "build"
    if build_subdir in {".", ".."} or Path(build_subdir).name != build_subdir:
        raise ValueError(
            "TILEFOUNDRY_NPU_BUILD_SUBDIR must be one relative directory name, "
            f"got {build_subdir!r}"
        )
    return root / build_subdir / "agent_gates" / "fast"


def _fast_cache_roots(root: Path, env_vals: Mapping[str, str]) -> tuple[Path, Path]:
    """Resolve the reference and TileLang caches the device-fast child also uses."""
    raw = env_vals.get("TILEFOUNDRY_NPU_WORK_ROOT", "").strip()
    if not raw:
        raise ValueError(
            "AgentGates machine env has no TILEFOUNDRY_NPU_WORK_ROOT for device-fast"
        )
    work_root = Path(raw).expanduser()
    if not work_root.is_absolute():
        raise ValueError(
            f"TILEFOUNDRY_NPU_WORK_ROOT must be an absolute path, got {raw!r}"
        )
    resolved = work_root.resolve()
    if resolved == root or resolved.is_relative_to(root):
        raise ValueError(
            f"TILEFOUNDRY_NPU_WORK_ROOT must be outside the source root, got {resolved}"
        )
    base = resolved / "surrogate"
    return base / "reference", base / "tilelang"


def _stage_fast_candidate(
    request_dir: Path,
    root: Path,
    operator_name: str,
    index: int,
    candidate_id: str,
    module_path: Path,
) -> tuple[str, str]:
    """Copy a candidate into the request staging tree, returning (relative path, sha256)."""
    source = module_path.resolve()
    if not source.is_file():
        raise FileNotFoundError(f"candidate module {source} does not exist")
    if source.suffix != ".py":
        raise ValueError(f"candidate module {source} must be a Python file")
    operator_dir = request_dir / "candidates" / _fast_slug(operator_name, fallback="operator")
    operator_dir.mkdir(parents=True, exist_ok=True)
    staged = operator_dir / f"{index:03d}_{_fast_slug(candidate_id, fallback='candidate')}.py"
    if staged.exists():
        raise FileExistsError(f"staged candidate {staged} already exists")
    shutil.copyfile(source, staged)
    staged.chmod(0o444)
    return staged.relative_to(root).as_posix(), _sha256_file(staged)


def _write_fast_request(path: Path, payload: Mapping[str, object]) -> str:
    """Write the request once as an immutable descriptor and return its sha256."""
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fchmod(stream.fileno(), 0o444)
    return hashlib.sha256(data).hexdigest()


def _fast_response_mapping(raw: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"fast response field {key!r} must be a mapping")
    return value


def _require_fast_envelope(
    raw: Mapping[str, object],
    *,
    request_id: str,
    request_sha256: str,
    mode: str,
) -> str:
    """Validate the response envelope against the submitted request and return its status."""
    if raw.get("schema") != FAST_RESPONSE_SCHEMA:
        raise RuntimeError(
            f"fast response schema {raw.get('schema')!r} != {FAST_RESPONSE_SCHEMA!r}"
        )
    if raw.get("request_id") != request_id:
        raise RuntimeError(
            f"fast response request_id {raw.get('request_id')!r} != {request_id!r}"
        )
    if raw.get("request_sha256") != request_sha256:
        raise RuntimeError("fast response request_sha256 does not match the submitted request")
    if raw.get("mode") != mode:
        raise RuntimeError(f"fast response mode {raw.get('mode')!r} != {mode!r}")
    status = raw.get("status")
    if status not in (
        FAST_STATUS_OK,
        FAST_STATUS_CANDIDATE_FAILURE,
        FAST_STATUS_REQUEST_ERROR,
        FAST_STATUS_INFRASTRUCTURE_ERROR,
    ):
        raise RuntimeError(f"fast response status {status!r} is not recognized")
    return str(status)


def _raise_for_fast_status(status: str, raw: Mapping[str, object]) -> None:
    if status in (FAST_STATUS_REQUEST_ERROR, FAST_STATUS_INFRASTRUCTURE_ERROR):
        detail = str(raw.get("error", "")).strip() or "no detail"
        raise RuntimeError(f"fast gate {status}: {detail}")


def _verify_fast_bindings(
    raw: Mapping[str, object],
    expected: Mapping[str, Mapping[str, Mapping[str, str]]],
) -> None:
    """Prove every evaluated candidate matches its staged path and content digest."""
    bindings = _fast_response_mapping(raw, "candidates")
    for operator_name, candidates in expected.items():
        operator_bindings = bindings.get(operator_name)
        if not isinstance(operator_bindings, Mapping):
            raise RuntimeError(
                f"fast response has no candidate bindings for operator {operator_name!r}"
            )
        if set(operator_bindings) != set(candidates):
            raise RuntimeError(
                f"fast response candidate set for operator {operator_name!r} "
                "does not match the request"
            )
        for candidate_id, binding in candidates.items():
            actual = operator_bindings.get(candidate_id)
            if not isinstance(actual, Mapping):
                raise RuntimeError(
                    f"fast response binding for {operator_name}/{candidate_id} is malformed"
                )
            if actual.get("module_path") != binding["module_path"]:
                raise RuntimeError(
                    f"fast response module_path for {operator_name}/{candidate_id} "
                    "does not match the staged candidate"
                )
            if actual.get("sha256") != binding["sha256"]:
                raise RuntimeError(
                    f"fast response sha256 for {operator_name}/{candidate_id} "
                    "does not match the staged candidate"
                )
            if actual.get("post_sha256") != binding["sha256"]:
                raise RuntimeError(
                    f"fast gate {FAST_STATUS_INFRASTRUCTURE_ERROR}: candidate "
                    f"{operator_name}/{candidate_id} changed during evaluation"
                )


def _verify_fast_coverage(
    raw: Mapping[str, object],
    *,
    expected_candidates: int,
    expected_msprof_targets: int,
) -> None:
    coverage = _fast_response_mapping(raw, "coverage")
    expected_seen = coverage.get("expected_candidates")
    evaluated = coverage.get("evaluated_candidates")
    targets_expected = coverage.get("expected_msprof_targets")
    targets_observed = coverage.get("observed_msprof_targets")
    if expected_seen != expected_candidates or evaluated != expected_candidates:
        raise RuntimeError(
            "fast response candidate coverage "
            f"expected={expected_seen!r} evaluated={evaluated!r} "
            f"request={expected_candidates}"
        )
    if targets_expected != expected_msprof_targets or targets_observed != expected_msprof_targets:
        raise RuntimeError(
            "fast response msprof coverage "
            f"expected={targets_expected!r} observed={targets_observed!r} "
            f"request={expected_msprof_targets}"
        )


def _submit_fast_gate(
    *,
    agent_gates_bin: str,
    root: Path,
    workspace: str,
    request_id: str,
    request_relative_path: str,
    request_path: Path,
    response_path: Path,
    request_sha256: str,
    mode: str,
    timeout_sec: float,
    device_id: str,
) -> tuple[Mapping[str, object], _V2GateExecution]:
    """Submit the device-fast dynamic target, verify its receipt, and load the response."""
    if _sha256_file(request_path) != request_sha256:
        raise RuntimeError("fast request file changed after it was written")
    execution = _run_v2_gate(
        agent_gates_bin=agent_gates_bin,
        root=root,
        workspace=workspace,
        gate_id=AGENT_GATES_DEVICE_FAST_GATE,
        target=f"fast/{request_id}",
        candidate_path=request_relative_path,
        timeout_sec=timeout_sec,
        device_id=device_id,
    )
    if execution.transport_error:
        raise RuntimeError(f"AgentGates device-fast batch failed: {execution.transport_error}")
    if not execution.verified:
        failures = execution.verification.get("failures")
        response_detail = ""
        if response_path.is_file():
            try:
                failed_raw = json.loads(response_path.read_text(encoding="utf-8"))
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                failed_raw = None
            if isinstance(failed_raw, Mapping):
                status = str(failed_raw.get("status", "")).strip()
                error = str(failed_raw.get("error", "")).strip()
                if status:
                    response_detail = f"; response status {status}: {error or 'no detail'}"
        raise RuntimeError(
            "AgentGates device-fast receipt did not verify: "
            f"{failures or 'unsatisfied'}{response_detail}"
        )
    if not _source_identity_matches(execution.verification, root):
        raise RuntimeError(
            "AgentGates device-fast verified source identity does not match requested source"
        )
    if not response_path.is_file():
        row, _observation = _gate_observation(execution.report, AGENT_GATES_DEVICE_FAST_GATE)
        detail = (
            (row.get("failures") if row is not None else None)
            or execution.report.get("status")
            or "missing response"
        )
        raise RuntimeError(f"AgentGates device-fast batch produced no response: {detail}")
    raw = json.loads(response_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError("AgentGates device-fast response must be a JSON object")
    status = _require_fast_envelope(
        raw,
        request_id=request_id,
        request_sha256=request_sha256,
        mode=mode,
    )
    _raise_for_fast_status(status, raw)
    return raw, execution


def _fast_held_seconds(report: Mapping[str, object]) -> float:
    _row, observation = _gate_observation(report, AGENT_GATES_DEVICE_FAST_GATE)
    window = observation.get("deviceWindow")
    if not isinstance(window, Mapping):
        return 0.0
    held = window.get("heldSeconds")
    if isinstance(held, bool) or not isinstance(held, (int, float)):
        return 0.0
    return float(held)


def _source_identity_matches(verification: Mapping[str, object], root: Path) -> bool:
    identity = verification.get("current_source_identity")
    identity_root = identity.get("root") if isinstance(identity, Mapping) else None
    return bool(identity_root) and Path(str(identity_root)).resolve() == root


def _l3_evidence_payload(observation: Mapping[str, object]) -> Mapping[str, object]:
    for key in ("l3Evidence", "evidence"):
        candidate = observation.get(key)
        if isinstance(candidate, Mapping):
            return candidate
    return observation


def _l3_signed_failure_category(
    observation: Mapping[str, object],
    evidence: Mapping[str, object],
) -> str:
    """Return the adapter-signed category, falling back to the structured kind."""
    for container in (observation, evidence):
        raw = container.get("failureCategory")
        if raw is not None and str(raw).strip():
            return str(raw).strip().lower()
    structured = evidence.get("failure")
    if isinstance(structured, Mapping):
        return str(structured.get("kind") or "").strip().lower()
    return FAILURE_KIND_INFRASTRUCTURE


def _l3_structured_failure(
    observation: Mapping[str, object],
    evidence: Mapping[str, object],
) -> Mapping[str, object] | None:
    """Return the producer's highest-precedence structured failure, if any."""
    for container in (evidence, observation):
        structured = container.get("failure")
        if isinstance(structured, Mapping) and structured:
            return structured
    return None


def _l3_candidate_numeric_failure(
    marker: Mapping[str, object],
    evidence: Mapping[str, object],
    raw_category: str,
) -> bool:
    """Whether a producer marker binds a numeric regression to this candidate.

    A bare failed node/case is never enough. The marker must name a candidate
    comparison failure, and the signed summary must show that same node and
    case in a failed state.
    """
    if raw_category != "numeric":
        return False
    expected = {
        "kind": "candidate",
        "category": "numeric",
        "role": "candidate",
        "stage": "comparison",
    }
    for key, value in expected.items():
        if str(marker.get(key) or "").strip().lower() != value:
            return False
    if not str(marker.get("comparator") or "").strip():
        return False
    node_name = marker.get("node")
    case_id = marker.get("case")
    if not isinstance(node_name, str) or not node_name or case_id is None:
        return False
    comparisons = marker.get("comparisons")
    if not isinstance(comparisons, Sequence) or isinstance(comparisons, (str, bytes)):
        return False
    if not any(
        isinstance(row, Mapping) and row.get("passed") is False
        for row in comparisons
    ):
        return False
    failures = evidence.get("failures")
    if isinstance(failures, Sequence) and not isinstance(failures, (str, bytes)):
        blocked = {"infrastructure", "coverage"}
        if any(
            isinstance(row, Mapping)
            and str(row.get("kind") or "").strip().lower() in blocked
            for row in failures
        ):
            return False
    nodes = evidence.get("nodes")
    if not isinstance(nodes, Sequence) or isinstance(nodes, (str, bytes)):
        return False
    for node in nodes:
        if not isinstance(node, Mapping) or node.get("name") != node_name:
            continue
        if str(node.get("status") or "").strip().lower() not in {"failed", "error"}:
            return False
        cases = node.get("cases")
        if not isinstance(cases, Mapping):
            return False
        if str(cases.get(str(case_id)) or "").strip().lower() not in {"failed", "error"}:
            return False
        return True
    return False


def _l3_failure_kind(
    observation: Mapping[str, object],
    evidence: Mapping[str, object],
    coverage: Mapping[str, object] | None,
) -> str:
    """Classify a candidate defect only from explicit signed producer evidence.

    A failed JUnit/node/case row, a bare failure category, or a marker whose
    role/node/case/comparison does not line up is infrastructure. Missing or
    skipped coverage, interruptions, write errors and unreadable snapshots
    preserve the pending candidate and never consume stagnation.
    """
    _ = coverage
    interruption = evidence.get("interruption")
    if isinstance(interruption, Mapping):
        error_category = str(interruption.get("errorCategory") or "").strip()
        if error_category or interruption.get("signal") or interruption.get("interrupted"):
            return FAILURE_KIND_INFRASTRUCTURE
    write_errors = evidence.get("writeErrors")
    if (
        isinstance(write_errors, Sequence)
        and not isinstance(write_errors, (str, bytes))
        and len(write_errors) > 0
    ):
        return FAILURE_KIND_INFRASTRUCTURE
    phase = str(evidence.get("phase") or "").strip().lower()
    if phase != "finished":
        return FAILURE_KIND_INFRASTRUCTURE
    raw = _l3_signed_failure_category(observation, evidence)
    structured = _l3_structured_failure(observation, evidence)
    if structured is None:
        return FAILURE_KIND_INFRASTRUCTURE
    if _l3_candidate_numeric_failure(structured, evidence, raw):
        return FAILURE_KIND_CANDIDATE
    return FAILURE_KIND_INFRASTRUCTURE


def classify_l3_observation_failure(
    observation: Mapping[str, object],
) -> tuple[str, bool]:
    """Return ``(failure_kind, pending)`` for a signed L3 gate observation."""
    evidence = _l3_evidence_payload(observation)
    coverage_raw = evidence.get("coverage")
    kind = _l3_failure_kind(
        observation,
        evidence,
        coverage_raw if isinstance(coverage_raw, Mapping) else None,
    )
    return kind, kind != FAILURE_KIND_CANDIDATE


def l3_failure_result_from_observation(
    observation: Mapping[str, object],
    *,
    run_id: str = "",
    receipt_path: str = "",
    execution_id: str = "",
    candidate_sha256: str = "",
    target_path: str = "",
    device_id: str = "",
    state: str = "",
    execution_terminal: bool = True,
) -> L3AcceptanceResult:
    """Map an adapter L3 observation to the campaign failure result."""
    evidence = _l3_evidence_payload(observation)
    kind, pending = classify_l3_observation_failure(observation)
    return L3AcceptanceResult(
        passed=False,
        run_id=run_id,
        receipt_path=receipt_path,
        failure_kind=kind,
        failure_category=_l3_signed_failure_category(observation, evidence),
        pending=pending,
        execution_id=execution_id,
        candidate_sha256=candidate_sha256,
        target_path=target_path,
        device_id=device_id,
        state=state,
        execution_terminal=execution_terminal,
    )


def run_l3_full_contract_via_agent_gates(
    *,
    operator_name: str | None = None,
    candidate_module_path: str | Path | None = None,
    source_root: str | Path | None = None,
    env_file: str | Path | None = None,
    device_id: str = "1",
    timeout_sec: int = DEFAULT_L3_PROMOTION_WAIT_SEC,
    resume_execution_id: str = "",
    resume_run_id: str = "",
    on_execution_started: Callable[[str, str], None] | None = None,
) -> L3AcceptanceResult:
    """Run the L3 promotion gate through V2 and validate its signed promotion evidence.

    The adapter owns the interpreter, CANN/TileLang setup, cache policy and device.
    TileFoundry supplies only the canonical target and the workspace-relative candidate;
    the signed receipt must bind the actual imported candidate digest and full coverage.
    """
    _ = env_file
    result_device = str(device_id or "")
    started = time.monotonic()

    def _fail(
        message: str,
        *,
        execution_id: str = "",
        run_id: str = "",
        receipt: str = "",
        result_device_id: str = "",
        state: str = "",
        execution_terminal: bool = True,
    ) -> L3AcceptanceResult:
        return L3AcceptanceResult(
            passed=False,
            run_id=run_id,
            receipt_path=receipt,
            elapsed_sec=time.monotonic() - started,
            error_message=message[-1000:],
            failure_kind=FAILURE_KIND_INFRASTRUCTURE,
            execution_id=execution_id,
            candidate_sha256="",
            target_path="",
            device_id=result_device_id or result_device,
            state=state,
            execution_terminal=execution_terminal,
        )

    if operator_name is None or candidate_module_path is None:
        return _fail("L3 promotion requires an operator name and candidate module path")
    try:
        target_task = get_operator_task(operator_name)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        return _fail(f"L3 promotion target resolution failed: {exc}")
    root = Path(source_root).resolve() if source_root is not None else default_source_root()
    candidate = Path(candidate_module_path).resolve()
    relative = _workspace_relative_candidate(root, candidate)
    if relative is None:
        return _fail(f"L3 candidate {candidate} is outside workspace source root {root}")
    if not candidate.is_file():
        return _fail(f"L3 candidate module {candidate} does not exist")
    candidate_digest = _sha256_file(candidate)
    try:
        agent_gates_bin = _resolve_agent_gates_bin(root)
        workspace = _agent_gates_workspace_name(root)
    except (OSError, ValueError) as exc:
        return _fail(str(exc))
    target_path = str(candidate)

    execution = _run_v2_gate(
        agent_gates_bin=agent_gates_bin,
        root=root,
        workspace=workspace,
        gate_id=AGENT_GATES_L3_PROMOTION_GATE,
        target=target_task,
        candidate_path=relative,
        timeout_sec=timeout_sec,
        device_id=result_device,
        resume_execution_id=resume_execution_id,
        resume_run_id=resume_run_id,
        on_execution_started=on_execution_started,
    )
    if execution.transport_error:
        return _fail(
            execution.transport_error,
            execution_id=execution.execution_id,
            run_id=execution.run_id,
            receipt=execution.receipt_path,
            result_device_id=execution.device_id,
            state=execution.state,
            execution_terminal=execution.execution_terminal,
        )
    row, observation = _gate_observation(execution.report, AGENT_GATES_L3_PROMOTION_GATE)
    if row is None:
        return _fail(
            "L3 signed receipt has no promotion gate observation",
            execution_id=execution.execution_id,
            run_id=execution.run_id,
            receipt=execution.receipt_path,
            result_device_id=execution.device_id,
            state=execution.state,
            execution_terminal=execution.execution_terminal,
        )
    evidence = _l3_evidence_payload(observation)
    coverage_raw = evidence.get("coverage")
    coverage = coverage_raw if isinstance(coverage_raw, Mapping) else None
    recorded_status = _gate_recorded_status(row)

    def _reject(message: str, *, failure_kind: str) -> L3AcceptanceResult:
        return L3AcceptanceResult(
            passed=False,
            run_id=execution.run_id,
            receipt_path=execution.receipt_path,
            elapsed_sec=time.monotonic() - started,
            error_message=message[-1000:],
            failure_kind=failure_kind,
            failure_category=_l3_signed_failure_category(observation, evidence),
            pending=failure_kind != FAILURE_KIND_CANDIDATE,
            execution_id=execution.execution_id,
            candidate_sha256=candidate_digest,
            target_path=target_path,
            device_id=execution.device_id or result_device,
            state=execution.state,
            execution_terminal=execution.execution_terminal,
        )

    if recorded_status != "PASS":
        return _reject(
            f"L3 gate recorded status {recorded_status or 'missing'}: "
            f"{row.get('failures') or observation.get('error') or 'no detail'}",
            failure_kind=_l3_failure_kind(observation, evidence, coverage),
        )
    if not execution.verified:
        return _reject(
            f"L3 receipt did not verify: {execution.verification.get('failures') or 'unsatisfied'}",
            failure_kind=FAILURE_KIND_INFRASTRUCTURE,
        )
    if not _source_identity_matches(execution.verification, root):
        return _reject(
            "L3 verified source identity does not match requested source",
            failure_kind=FAILURE_KIND_INFRASTRUCTURE,
        )
    schema = str(evidence.get("schema", ""))
    profile = str(evidence.get("profile", ""))
    if schema != "tilefoundry-l3-evidence/v1" or profile != "l3-promotion":
        return _reject(
            f"L3 evidence schema/profile mismatch: schema={schema or 'missing'} "
            f"profile={profile or 'missing'}",
            failure_kind=FAILURE_KIND_INFRASTRUCTURE,
        )
    target = evidence.get("target")
    declared_task = target.get("task") if isinstance(target, Mapping) else None
    if not declared_task or str(declared_task) != target_task:
        return _reject(
            f"L3 evidence target task {declared_task or 'missing'} does not match {target_task}",
            failure_kind=FAILURE_KIND_INFRASTRUCTURE,
        )
    target_module = target.get("modulePath") if isinstance(target, Mapping) else None
    if not target_module or Path(str(target_module)).resolve() != Path(target_path):
        return _reject(
            "L3 evidence target does not match requested target module",
            failure_kind=FAILURE_KIND_INFRASTRUCTURE,
        )
    declared = evidence.get("candidate")
    declared_path = declared.get("modulePath") if isinstance(declared, Mapping) else None
    declared_digest = declared.get("sha256") if isinstance(declared, Mapping) else None
    if not declared_path or Path(str(declared_path)).resolve() != candidate:
        return _reject(
            "L3 evidence candidate path does not match the promoted candidate",
            failure_kind=FAILURE_KIND_INFRASTRUCTURE,
        )
    if str(declared_digest or "") != candidate_digest:
        return _reject(
            "L3 evidence candidate digest does not match the promoted candidate",
            failure_kind=FAILURE_KIND_INFRASTRUCTURE,
        )
    interpreter = evidence.get("interpreter")
    if not isinstance(interpreter, Mapping) or interpreter.get("match") is not True:
        return _reject(
            "L3 evidence interpreter did not match the adapter-resolved interpreter",
            failure_kind=FAILURE_KIND_INFRASTRUCTURE,
        )
    if coverage is None or coverage.get("qualified") is not True:
        return _reject(
            "L3 evidence did not qualify full contract coverage",
            failure_kind=_l3_failure_kind(observation, evidence, coverage),
        )
    nodes = coverage.get("nodes") if isinstance(coverage.get("nodes"), Mapping) else {}
    cases = coverage.get("cases") if isinstance(coverage.get("cases"), Mapping) else {}
    assert isinstance(nodes, Mapping) and isinstance(cases, Mapping)
    if (
        nodes.get("missing")
        or nodes.get("failed")
        or nodes.get("skipped")
        or nodes.get("blocked")
        or cases.get("missing")
        or cases.get("unexpected")
        or coverage.get("referenceCacheProblem")
    ):
        return _reject(
            "L3 evidence coverage is incomplete: "
            f"{dict(nodes)} / {dict(cases)} / "
            f"referenceCacheProblem={coverage.get('referenceCacheProblem')}",
            failure_kind=_l3_failure_kind(observation, evidence, coverage),
        )
    expected_cases = _as_positive_int(cases.get("expected"))
    completed_cases = _as_positive_int(cases.get("completed"))
    if expected_cases <= 0 or completed_cases != expected_cases:
        return _reject(
            f"L3 evidence case coverage incomplete: {completed_cases}/{expected_cases}",
            failure_kind=_l3_failure_kind(observation, evidence, coverage),
        )
    return L3AcceptanceResult(
        passed=True,
        run_id=execution.run_id,
        receipt_path=execution.receipt_path,
        elapsed_sec=time.monotonic() - started,
        execution_id=execution.execution_id,
        candidate_sha256=candidate_digest,
        target_path=target_path,
        device_id=execution.device_id or result_device,
        state=execution.state,
        execution_terminal=execution.execution_terminal,
    )


def _prewarm_micro_probe_cache(
    spec: object,
    *,
    source_root: Path,
    env_file: Path,
    tl_cache_dir: Path,
) -> None:
    env_vals = parse_simple_env_file(env_file)
    python_bin = (
        env_vals.get("TILEFOUNDRY_NPU_PYTHON")
        or os.environ.get("TILEFOUNDRY_NPU_PYTHON")
        or sys.executable
    )
    cann_setup = (env_vals.get("TILEFOUNDRY_NPU_CANN_SETUP") or "").strip()
    tl_source = (env_vals.get("TILEFOUNDRY_NPU_TILELANG_SOURCE") or "").strip()

    tl_cache_dir.mkdir(parents=True, exist_ok=True)
    run_env = dict(os.environ)
    run_env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
    run_env["ASCEND_RT_VISIBLE_DEVICES"] = ""
    run_env["USE_CUDA"] = "0"
    run_env["TILELANG_CACHE_DIR"] = str(tl_cache_dir)
    run_env["TILELANG_KERNEL_CACHE_USE_LIB_STAMP"] = "1"
    run_env["TILELANG_DISABLE_CACHE"] = "0"
    run_env["PYTHONDONTWRITEBYTECODE"] = "1"
    pp_parts = [str(source_root), str(source_root / "src"), str(source_root / "targets" / "ascend")]
    if tl_source:
        pp_parts.insert(0, tl_source)
    run_env["PYTHONPATH"] = os.pathsep.join(pp_parts)

    factory_name = str(getattr(spec, "factory_name"))
    sources = (
        ("base", str(getattr(spec, "baseline_probe_source"))),
        ("opt", str(getattr(spec, "optimized_probe_source"))),
    )
    script = (
        "import importlib, importlib.util, os, sys, tempfile\n"
        "from pathlib import Path\n"
        f"os.environ['TILELANG_CACHE_DIR'] = {str(tl_cache_dir)!r}\n"
        "os.environ['TILELANG_KERNEL_CACHE_USE_LIB_STAMP'] = '1'\n"
        "os.environ['TILELANG_DISABLE_CACHE'] = '0'\n"
        "import tilelang\n"
        "tl_env = importlib.import_module('tilelang.env')\n"
        "tl_kc = importlib.import_module('tilelang.cache.kernel_cache')\n"
        "tl_env.env.enable_cache()\n"
        "tl_kc.KernelCache._get_cache_namespace.cache_clear()\n"
        "tl_kc.KernelCache._get_tilelang_lib_stamp.cache_clear()\n"
        "tl_kc.KernelCache._create_dirs()\n"
        f"sources = {json.dumps(sources)}\n"
        f"factory = {factory_name!r}\n"
        "with tempfile.TemporaryDirectory(prefix='tf_probe_pre_') as td:\n"
        "    for tag, src in sources:\n"
        "        p = Path(td) / f'{tag}.py'\n"
        "        p.write_text(src, encoding='utf-8')\n"
        "        sp = importlib.util.spec_from_file_location(f'_tf_p_{tag}', str(p))\n"
        "        mod = importlib.util.module_from_spec(sp)\n"
        "        sp.loader.exec_module(mod)\n"
        "        fn = getattr(mod, factory)()\n"
        "        tilelang.compile(fn, target='ascend', out_idx=-1)\n"
    )
    with tempfile.TemporaryDirectory(prefix="tf_probe_cpu_") as tmp_dir:
        drv = Path(tmp_dir) / "prewarm_probe.py"
        drv.write_text(script, encoding="utf-8")
        if cann_setup and Path(cann_setup).is_file() and shutil.which("bisheng") is None:
            bash_cmd = (
                f"set +u && source {shlex.quote(cann_setup)} >/dev/null 2>&1 "
                f"&& exec {shlex.quote(python_bin)} {shlex.quote(str(drv))}"
            )
            argv = ["/bin/bash", "-c", bash_cmd]
        else:
            argv = [python_bin, str(drv)]
        try:
            subprocess.run(
                argv,
                text=True,
                capture_output=True,
                cwd=str(source_root),
                env=run_env,
                timeout=180,
            )
        except Exception as exc:
            logger.debug("Micro-probe CPU prewarm skipped: %s", exc)


def run_micro_probe_via_agent_gates(
    spec: object,
    *,
    source_root: str | Path | None = None,
    env_file: str | Path | None = None,
    device_id: str = "1",
    min_speedup: float = 1.08,
    timeout_sec: int = 360,
    prewarm_cache: bool = True,
) -> dict[str, float]:
    """Run hardware micro-probe verification strictly through AgentGates NPU lease."""
    root = Path(source_root).resolve() if source_root is not None else default_source_root()
    resolved_env = _resolve_env_file_path(env_file)
    ag_bin = _resolve_agent_gates_bin(root)
    tl_cache_dir = _default_surrogate_tl_cache_dir()

    if prewarm_cache:
        _prewarm_micro_probe_cache(
            spec,
            source_root=root,
            env_file=resolved_env,
            tl_cache_dir=tl_cache_dir,
        )

    fail_closed = {
        "baseline_dur_us": 0.0,
        "optimized_dur_us": 0.0,
        "speedup": 0.0,
        "numerical_ok": 0.0,
    }
    with tempfile.TemporaryDirectory(prefix="tf_probe_gate_") as tmp_dir:
        tmp = Path(tmp_dir)
        request_path = tmp / "micro_probe_request.json"
        response_path = tmp / "micro_probe_response.json"
        tmp_env = _write_temp_env_file(
            resolved_env,
            tmp / "env.sh",
            {
                "TILEFOUNDRY_NPU_DEVICE_TESTS": "tests/ascend/agent/test_surrogate_batch_npu.py",
            },
        )
        payload = {
            "probe_id": str(getattr(spec, "probe_id")),
            "probe_kind": str(getattr(spec, "probe_kind")),
            "factory_name": str(getattr(spec, "factory_name")),
            "baseline_probe_source": str(getattr(spec, "baseline_probe_source")),
            "optimized_probe_source": str(getattr(spec, "optimized_probe_source")),
            "line_count_baseline": int(getattr(spec, "line_count_baseline", 0)),
            "line_count_optimized": int(getattr(spec, "line_count_optimized", 0)),
            "tensor_shape": [int(x) for x in getattr(spec, "tensor_shape", (16, 128))],
            "tensor_dtype": str(getattr(spec, "tensor_dtype", "float32")),
            "min_speedup": float(min_speedup),
            "tilelang_cache_dir": str(tl_cache_dir),
            "response_path": str(response_path),
        }
        request_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        cmd = [
            ag_bin,
            "run",
            "--adapter",
            "tilefoundry-npu",
            "--source",
            str(root),
            "--env-file",
            str(tmp_env),
            "--device-id",
            str(device_id),
            "--gate",
            "tilefoundry-npu-device-fast",
            "--json",
        ]
        sub_env = _clean_agent_gates_subprocess_env(
            {
                "TF_MICRO_PROBE_REQUEST": str(request_path),
                "TF_SURROGATE_TL_CACHE_DIR": str(tl_cache_dir),
            }
        )
        proc, parsed = _run_agent_gates_cmd(
            cmd,
            cwd=str(root),
            env=sub_env,
            timeout_sec=timeout_sec,
        )
        run_id = str(parsed.get("runId", ""))
        run_passed = proc.returncode == 0 and parsed.get("status") == "PASS"
        if run_passed:
            verified, verify_error = (
                verify_agent_gates_receipt(ag_bin, run_id, source_root=root)
                if run_id
                else (False, "AgentGates micro-probe reported PASS without a runId")
            )
            if not verified:
                logger.warning(
                    "AgentGates micro-probe receipt verification failed: %s",
                    verify_error or "unknown failure",
                )
                return fail_closed
        if not response_path.is_file():
            logger.warning(
                "AgentGates micro-probe missing response (rc=%s): %s",
                proc.returncode,
                (proc.stderr or proc.stdout or "")[-400:],
            )
            return fail_closed
        raw_resp = json.loads(response_path.read_text(encoding="utf-8"))
        metrics_raw = raw_resp.get("metrics") or {}
        return {
            str(k): float(v)
            for k, v in metrics_raw.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        } or fail_closed


def _agent_gates_evidence_root() -> Path:
    """Read the configured AgentGates evidence root used by the V2 worker."""
    raw = _agent_gates_config()
    evidence_root = raw.get("evidence_root")
    if not isinstance(evidence_root, str) or not evidence_root.strip():
        raise ValueError("AgentGates evidence_root must be configured")
    return Path(evidence_root).expanduser().resolve()


def _extract_l4_metrics(
    parsed: Mapping[str, object],
    target_task: str,
) -> tuple[float, float, str, dict[int, float], dict[int, float]]:
    """Extract L4 aggregate and per-case metrics from signed gate observations."""
    avg_speedup = 0.0
    score = 0.0
    report_dir = ""
    per_case_speedups: dict[int, float] = {}
    per_case_elapsed: dict[int, float] = {}
    for gate_row in parsed.get("gates", ()):
        if not isinstance(gate_row, Mapping):
            continue
        if gate_row.get("gateId") != AGENT_GATES_L4_CANNBENCH_GATE:
            continue
        obs = gate_row.get("observations") or {}
        if not isinstance(obs, Mapping):
            continue
        avg_speedup = float(obs.get("avgSpeedup", 0.0))
        score = float(obs.get("score", 0.0))
        for res_row in obs.get("results", ()):
            if not isinstance(res_row, Mapping):
                continue
            if target_task and res_row.get("task") != target_task:
                continue
            report_dir = str(res_row.get("reportDir", ""))
            for case_row in res_row.get("cases", ()):
                if not isinstance(case_row, Mapping):
                    continue
                cid_str = str(case_row.get("case_id", ""))
                if "_" not in cid_str or not cid_str.rsplit("_", 1)[-1].isdigit():
                    continue
                cnum = int(cid_str.rsplit("_", 1)[-1])
                per_case_speedups[cnum] = float(case_row.get("speedup") or 0.0)
                per_case_elapsed[cnum] = float(case_row.get("elapsed_us") or 0.0)
    return avg_speedup, score, report_dir, per_case_speedups, per_case_elapsed


def _run_agent_gates_v2_cmd(
    cmd: Sequence[str],
    *,
    root: Path,
    timeout_sec: float,
) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
    """Run one V2 AgentGates command against the configured worker evidence."""
    proc = subprocess.run(
        [str(part) for part in cmd],
        text=True,
        capture_output=True,
        cwd=str(root),
        env=_agent_gates_cli_env(),
        timeout=timeout_sec,
    )
    return proc, _parse_agent_gates_json(proc.stdout)


def _l4_failure(
    error_message: str,
    *,
    failure_kind: str = FAILURE_KIND_INFRASTRUCTURE,
    execution_id: str = "",
    run_id: str = "",
    receipt_path: str = "",
    observation: Mapping[str, object] | None = None,
    device_id: str = "",
    state: str = "",
    execution_terminal: bool = True,
    target_path: str = "",
    candidate_sha256: str = "",
    attempts: tuple[dict[str, str], ...] = (),
) -> L4CannbenchResult:
    """Return a failed L4 result without metrics or promotion evidence."""
    obs = observation or {}
    return L4CannbenchResult(
        passed=False,
        run_id=run_id,
        receipt_path=receipt_path,
        error_message=error_message[-1000:],
        failure_kind=failure_kind,
        execution_id=execution_id,
        official_report_valid=obs.get("officialReportValid") is True,
        all_cases_passed=obs.get("allCasesPassed") is True,
        all_timings_valid=obs.get("allTimingsValid") is True,
        case_count=_as_positive_int(obs.get("caseCount")),
        passed_cases=_as_positive_int(obs.get("passedCases")),
        device_id=device_id,
        state=state,
        execution_terminal=execution_terminal,
        target_path=target_path,
        candidate_sha256=candidate_sha256,
        attempts=attempts,
    )


def _l4_candidate_matches(raw: object, expected: str, root: Path) -> bool:
    text = str(raw or "").strip()
    if not text:
        return False
    if text.endswith(expected) and (
        text == expected or text.endswith("/" + expected)
    ):
        return True
    path = Path(text)
    return path.is_absolute() and path.resolve() == (root / expected).resolve()


def _validate_l4_report(
    execution: _V2GateExecution,
    *,
    root: Path,
    target_task: str,
    expected_candidate: str,
) -> tuple[str, str]:
    """Validate signed L4 evidence; return ``(error_message, failure_kind)``.

    Only a complete official case set where every case passed is a candidate
    verdict. Missing cases, a missing or non-numeric passed count, and missing
    timings are unknown coverage: they preserve the pending candidate instead of
    blaming it. Report validation flags and a short passed-case count are
    candidate failures; identity, directory and signature problems are not.
    """
    if not execution.verified:
        return (
            f"L4 receipt did not verify: {execution.verification.get('failures') or 'unsatisfied'}",
            FAILURE_KIND_INFRASTRUCTURE,
        )
    if not _source_identity_matches(execution.verification, root):
        return (
            "L4 verified source identity does not match requested source",
            FAILURE_KIND_INFRASTRUCTURE,
        )
    row, obs = _gate_observation(execution.report, AGENT_GATES_L4_CANNBENCH_GATE)
    if row is None:
        return ("L4 signed receipt has no cannbench gate observation", FAILURE_KIND_INFRASTRUCTURE)
    status = _gate_recorded_status(row)
    if status != "PASS":
        return (
            f"L4 gate recorded status {status or 'missing'}: "
            f"{row.get('failures') or obs.get('error') or 'no detail'}",
            FAILURE_KIND_CANDIDATE if status == "FAIL" else FAILURE_KIND_INFRASTRUCTURE,
        )
    flags = {
        "officialReportValid": obs.get("officialReportValid"),
        "allCasesPassed": obs.get("allCasesPassed"),
        "allTimingsValid": obs.get("allTimingsValid"),
    }
    bad_flags = [name for name, value in flags.items() if value is not True]
    if bad_flags:
        return (
            f"L4 official report validation failed: {', '.join(bad_flags)}",
            FAILURE_KIND_CANDIDATE,
        )
    case_count = _as_positive_int(obs.get("caseCount"))
    passed_raw = obs.get("passedCases")
    passed_cases = _as_positive_int(passed_raw)
    timed_cases = _as_positive_int(obs.get("timedCases"))
    if case_count <= 0:
        return ("L4 signed observation reports no official cases", FAILURE_KIND_INFRASTRUCTURE)
    if isinstance(passed_raw, bool) or not isinstance(passed_raw, (int, float)):
        return (
            "L4 signed observation has no valid passed-case count",
            FAILURE_KIND_INFRASTRUCTURE,
        )
    if passed_cases != case_count:
        return (
            f"L4 passed {passed_cases}/{case_count} official cases",
            FAILURE_KIND_CANDIDATE,
        )
    if timed_cases != case_count:
        return (
            f"L4 timed {timed_cases}/{case_count} official cases",
            FAILURE_KIND_INFRASTRUCTURE,
        )
    results = obs.get("results")
    if not isinstance(results, Sequence) or isinstance(results, (str, bytes)):
        return ("L4 signed observation has no per-target results", FAILURE_KIND_INFRASTRUCTURE)
    matched = 0
    for res_row in results:
        if not isinstance(res_row, Mapping):
            continue
        if target_task and res_row.get("task") != target_task:
            continue
        matched += 1
        if res_row.get("reportValid") is not True:
            return (
                f"L4 official report is not valid for task {target_task or res_row.get('task')}",
                FAILURE_KIND_CANDIDATE,
            )
        if not str(res_row.get("reportDir", "")).strip():
            return ("L4 official report directory is missing", FAILURE_KIND_INFRASTRUCTURE)
        if not _l4_candidate_matches(res_row.get("candidate"), expected_candidate, root):
            return (
                f"L4 receipt candidate does not match {expected_candidate}",
                FAILURE_KIND_INFRASTRUCTURE,
            )
    if matched == 0:
        return (
            f"L4 signed observation has no result for task {target_task or '<any>'}",
            FAILURE_KIND_INFRASTRUCTURE,
        )
    return ("", "")


def run_l4_official_cannbench_via_agent_gates(
    *,
    operator_name: str | None = None,
    candidate_module_path: str | Path | None = None,
    workspace: str | None = None,
    source_root: str | Path | None = None,
    env_file: str | Path | None = None,
    device_id: str = "1",
    timeout_sec: int = DEFAULT_L4_CANNBENCH_WAIT_SEC,
    resume_execution_id: str = "",
    resume_run_id: str = "",
    on_execution_started: Callable[[str, str], None] | None = None,
) -> L4CannbenchResult:
    """Run official tilefoundry-npu-cannbench through V2 and verify its signed receipt.

    The campaign's actual entry always uses V2: the adapter owns environment, device
    lease and cache policy, while the candidate module is bound by path and signed
    observation. The default wait budgets queueing, prep and the 3600 s official bound.
    Metrics are returned only after the official-final validation flags, case counts,
    source identity and candidate binding check out. A resume keeps its original card,
    and a retry on another card is a fresh call so no timings are stitched across cards.
    """
    _ = env_file
    result_device = str(device_id or "")
    if operator_name is None:
        return _l4_failure("L4 cannbench requires a canonical operator name", device_id=result_device)
    try:
        target_task = get_operator_task(operator_name)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        return _l4_failure(
            f"L4 cannbench target resolution failed: {exc}", device_id=result_device
        )
    root = Path(source_root).resolve() if source_root is not None else default_source_root()
    if candidate_module_path is not None:
        expected_path = Path(candidate_module_path).resolve()
    else:
        expected_path = (root / get_operator_module_relpath(operator_name)).resolve()
    relative = _workspace_relative_candidate(root, expected_path)
    if relative is None:
        return _l4_failure(
            f"L4 candidate {expected_path} is outside workspace source root {root}",
            device_id=result_device,
        )
    try:
        candidate_digest = _sha256_file(expected_path)
    except OSError as exc:
        return _l4_failure(
            f"L4 candidate module {expected_path} cannot be read: {exc}",
            device_id=result_device,
            target_path=str(expected_path),
        )
    try:
        agent_gates_bin = _resolve_agent_gates_bin(root)
        resolved_workspace = workspace or _agent_gates_workspace_name(root)
    except (OSError, ValueError) as exc:
        return _l4_failure(str(exc), device_id=result_device)

    execution = _run_v2_gate(
        agent_gates_bin=agent_gates_bin,
        root=root,
        workspace=resolved_workspace,
        gate_id=AGENT_GATES_L4_CANNBENCH_GATE,
        target=target_task,
        candidate_path=relative,
        timeout_sec=timeout_sec,
        device_id=result_device,
        resume_execution_id=resume_execution_id,
        resume_run_id=resume_run_id,
        on_execution_started=on_execution_started,
    )
    if execution.transport_error:
        return _l4_failure(
            execution.transport_error,
            execution_id=execution.execution_id,
            run_id=execution.run_id,
            receipt_path=execution.receipt_path,
            device_id=execution.device_id or result_device,
            state=execution.state,
            execution_terminal=execution.execution_terminal,
            target_path=str(expected_path),
            candidate_sha256=candidate_digest,
        )
    _, observation = _gate_observation(execution.report, AGENT_GATES_L4_CANNBENCH_GATE)
    error, failure_kind = _validate_l4_report(
        execution,
        root=root,
        target_task=target_task,
        expected_candidate=relative,
    )
    if error:
        return _l4_failure(
            error,
            failure_kind=failure_kind,
            execution_id=execution.execution_id,
            run_id=execution.run_id,
            receipt_path=execution.receipt_path,
            observation=observation,
            device_id=execution.device_id or result_device,
            state=execution.state,
            execution_terminal=execution.execution_terminal,
            target_path=str(expected_path),
            candidate_sha256=candidate_digest,
        )
    avg_speedup, score, report_dir, speedups, elapsed = _extract_l4_metrics(
        execution.report, target_task
    )
    return L4CannbenchResult(
        passed=True,
        avg_speedup=avg_speedup,
        score=score,
        per_case_speedups=speedups,
        per_case_elapsed_us=elapsed,
        run_id=execution.run_id,
        receipt_path=execution.receipt_path,
        report_dir=report_dir,
        execution_id=execution.execution_id,
        official_report_valid=True,
        all_cases_passed=True,
        all_timings_valid=True,
        case_count=_as_positive_int(observation.get("caseCount")),
        passed_cases=_as_positive_int(observation.get("passedCases")),
        device_id=execution.device_id or result_device,
        state=execution.state,
        execution_terminal=execution.execution_terminal,
        target_path=str(expected_path),
        candidate_sha256=candidate_digest,
    )


__all__ = [
    "AGENT_GATES_ADAPTER",
    "AGENT_GATES_DEVICE_FAST_GATE",
    "AGENT_GATES_L3_PROMOTION_GATE",
    "AGENT_GATES_L4_CANNBENCH_GATE",
    "AnchorEvalMetrics",
    "CandidateSurrogateResult",
    "DEFAULT_L3_PROMOTION_WAIT_SEC",
    "DEFAULT_L4_CANNBENCH_WAIT_SEC",
    "FAST_REQUEST_ENV",
    "FAST_REQUEST_SCHEMA",
    "FAST_RESPONSE_SCHEMA",
    "FAST_SOURCE_ROOT_ENV",
    "FAST_STATUS_CANDIDATE_FAILURE",
    "FAST_STATUS_INFRASTRUCTURE_ERROR",
    "FAST_STATUS_OK",
    "FAST_STATUS_REQUEST_ERROR",
    "FAILURE_KIND_CANDIDATE",
    "FAILURE_KIND_INFRASTRUCTURE",
    "L0L1CandidateCheck",
    "L3AcceptanceResult",
    "L4CannbenchResult",
    "aggregate_surrogate_result",
    "bottleneck_anchor_for_regime",
    "build_anchor_metrics",
    "collect_bottleneck_msprof_via_agent_gates",
    "ensure_npu_compare_shim_dir",
    "lookup_case_static_spec",
    "prewarm_anchor_reference_cache",
    "prewarm_surrogate_tilelang_cache",
    "run_l0_l1_round_filter",
    "run_l0_static_precheck",
    "run_l1_compile_check",
    "run_l3_full_contract_via_agent_gates",
    "run_l4_official_cannbench_via_agent_gates",
    "run_micro_probe_via_agent_gates",
    "run_multi_operator_surrogate_batch_via_agent_gates",
    "run_surrogate_batch_via_agent_gates",
]
