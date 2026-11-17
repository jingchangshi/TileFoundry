"""Four-tier host-outside-gate evaluation pyramid (L0->L1->L2->L3->L4) for campaigns."""

from __future__ import annotations

import asyncio
import json
import math
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from tilefoundry_ascend.agent.campaign.anchor_specs import (
    AnchorCaseSpec,
    PerfTargetSpec,
    compute_case_eq3_score,
    default_source_root,
    get_default_anchors,
    get_operator_entrypoint,
    get_operator_task,
    parse_simple_env_file,
    resolve_cannbench_repo,
)
from tilefoundry_ascend.agent.orchestrator.gate_funnel import (
    StageAResult,
    StageBResult,
    build_stage_b_driver_source,
    run_stage_a_static,
    run_stage_b_compile,
)
from tilefoundry_ascend.agent.orchestrator.llm_client import (
    KernelLLMBackend,
    extract_python_dsl,
)
from tilefoundry_ascend.agent.orchestrator.portfolio_runner import build_syntax_repair_request


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
    cann_repo = {cann_repo!r}
    vendor_src = str(Path(cann_repo) / "vendor" / "cannbench" / "src")
    if vendor_src not in sys.path:
        sys.path.insert(0, vendor_src)
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
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()[-1500:]
        return StageBResult(
            passed=False,
            error_message=f"l1_compile_failed (rc={proc.returncode}): {detail}",
            elapsed_sec=time.monotonic() - t0,
        )
    return StageBResult(passed=True, elapsed_sec=time.monotonic() - t0)


async def run_l0_l1_round_filter(
    candidates: Sequence[tuple[str, str]],
    *,
    patch_fn: Callable[[str], str],
    l0_checker: Callable[[str], StageAResult],
    l1_compiler: Callable[[str], StageBResult],
    repair_backend: KernelLLMBackend | None = None,
    max_syntax_repairs: int = 1,
) -> list[L0L1CandidateCheck]:
    """Filter candidate snippets through L0 static and concurrent L1 compile checks."""

    async def _check_one(cand_id: str, snippet: str) -> L0L1CandidateCheck:
        cur_snippet = snippet
        repaired = False
        for attempt in range(max(0, max_syntax_repairs) + 1):
            try:
                patched = patch_fn(cur_snippet)
            except Exception as exc:
                err = f"ast_patch_error: {type(exc).__name__}: {exc}"
                if repair_backend is None or attempt >= max(0, max_syntax_repairs):
                    return L0L1CandidateCheck(
                        candidate_id=cand_id,
                        passed=False,
                        failed_tier="stage_a",
                        patched_module_source="",
                        snippet_source=cur_snippet,
                        repaired=repaired,
                        error_message=err,
                    )
                req = build_syntax_repair_request(cur_snippet, err)
                cur_snippet = extract_python_dsl(await repair_backend.generate(req))
                repaired = True
                continue

            stage_a = await asyncio.to_thread(l0_checker, patched)
            if not stage_a.passed:
                err = stage_a.error_message or "l0_static_failed"
                if repair_backend is None or attempt >= max(0, max_syntax_repairs):
                    return L0L1CandidateCheck(
                        candidate_id=cand_id,
                        passed=False,
                        failed_tier="stage_a",
                        patched_module_source=patched,
                        snippet_source=cur_snippet,
                        repaired=repaired,
                        stage_a=stage_a,
                        error_message=err,
                    )
                req = build_syntax_repair_request(cur_snippet, err)
                cur_snippet = extract_python_dsl(await repair_backend.generate(req))
                repaired = True
                continue

            stage_b = await asyncio.to_thread(l1_compiler, patched)
            if not stage_b.passed:
                err = stage_b.error_message or "l1_compile_failed"
                if repair_backend is None or attempt >= max(0, max_syntax_repairs):
                    return L0L1CandidateCheck(
                        candidate_id=cand_id,
                        passed=False,
                        failed_tier="stage_b",
                        patched_module_source=patched,
                        snippet_source=cur_snippet,
                        repaired=repaired,
                        stage_a=stage_a,
                        stage_b=stage_b,
                        error_message=err,
                    )
                req = build_syntax_repair_request(cur_snippet, err)
                cur_snippet = extract_python_dsl(await repair_backend.generate(req))
                repaired = True
                continue

            return L0L1CandidateCheck(
                candidate_id=cand_id,
                passed=True,
                failed_tier=None,
                patched_module_source=patched,
                snippet_source=cur_snippet,
                repaired=repaired,
                stage_a=stage_a,
                stage_b=stage_b,
            )
        raise RuntimeError("unreachable repair termination")

    return list(await asyncio.gather(*(_check_one(cid, src) for cid, src in candidates)))


def _resolve_agent_gates_bin(source_root: Path | None = None) -> str:
    env_bin = os.environ.get("TILEFOUNDRY_NPU_AGENT_GATES_BIN", "").strip()
    if env_bin and Path(env_bin).is_file():
        return env_bin
    on_path = shutil.which("agent-gates")
    if on_path:
        return on_path
    if source_root is not None:
        sibling = source_root.parent / "AgentGates" / ".venv" / "bin" / "agent-gates"
        if sibling.is_file():
            return str(sibling)
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

    entrypoint = get_operator_entrypoint(operator_name)
    case_ids = [int(a.case_num) for a in anchors]
    cand_paths = [(str(cid), str(Path(p).resolve())) for cid, p in candidates]

    with tempfile.TemporaryDirectory(prefix="tf_tl_prewarm_") as tmp_dir:
        driver_path = Path(tmp_dir) / "prewarm_tl_driver.py"
        driver_path.write_text(
            "import importlib.util, json, sys, tilelang\n"
            f"sys.path.insert(0, {str(root / 'targets' / 'ascend')!r})\n"
            "from tilefoundry_ascend.agent.campaign.anchor_specs import (\n"
            "    AnchorCaseSpec, load_anchor_harness_and_cases\n"
            ")\n"
            "class _CacheWarmed(Exception):\n"
            "    pass\n"
            "_orig_compile = tilelang.compile\n"
            "def _compile_only(*args, **kwargs):\n"
            "    _orig_compile(*args, **kwargs)\n"
            "    def _abort_launch(*_a, **_kw):\n"
            "        raise _CacheWarmed()\n"
            "    return _abort_launch\n"
            "tilelang.compile = _compile_only\n"
            f"op_name = {operator_name!r}\n"
            f"entrypoint = {entrypoint!r}\n"
            f"case_ids = {json.dumps(case_ids)}\n"
            f"cand_paths = {json.dumps(cand_paths)}\n"
            f"repo = {str(repo)!r}\n"
            f"root = {str(root)!r}\n"
            "specs = [AnchorCaseSpec('', cid, '', 1.0, 1.0) for cid in case_ids]\n"
            "harness, cases = load_anchor_harness_and_cases(\n"
            "    op_name, specs, cannbench_repo=repo, source_root=root\n"
            ")\n"
            "params_list = [harness.params(case, harness.prepare(case)) for case in cases]\n"
            "for cid, mod_path in cand_paths:\n"
            "    spec = importlib.util.spec_from_file_location(f'_tf_warm_{cid}', mod_path)\n"
            "    mod = importlib.util.module_from_spec(spec)\n"
            "    spec.loader.exec_module(mod)\n"
            "    fn = getattr(mod, entrypoint)\n"
            "    for params in params_list:\n"
            "        try:\n"
            "            fn(**params)\n"
            "        except _CacheWarmed:\n"
            "            pass\n"
            "        except Exception:\n"
            "            pass\n",
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
    return resolved_tl_cache


def _write_temp_env_file(
    base_env: Path,
    dest_path: Path,
    overrides: Mapping[str, str],
) -> Path:
    lines = [
        ln
        for ln in base_env.read_text(encoding="utf-8").splitlines()
        if not any(ln.startswith(f"{key}=") for key in overrides)
    ]
    for key, value in overrides.items():
        lines.append(f'{key}="{value}"')
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
    env.update(extra_env)
    return env


def _parse_agent_gates_json(stdout: str) -> dict[str, object]:
    idx = stdout.find("{")
    if idx < 0:
        return {}
    try:
        parsed = json.loads(stdout[idx:])
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


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
    cache_dir = _default_surrogate_cache_dir()
    tl_cache_dir = _default_surrogate_tl_cache_dir()

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

    with tempfile.TemporaryDirectory(prefix="tf_l2_batch_") as tmp_dir:
        tmp = Path(tmp_dir)
        request_path = tmp / "surrogate_request.json"
        response_path = tmp / "surrogate_response.json"
        tmp_env = _write_temp_env_file(
            resolved_env,
            tmp / "env.sh",
            {
                "TILEFOUNDRY_NPU_CPU_TESTS": "tests/ascend/agent/test_llm_watchdog.py",
                "TILEFOUNDRY_NPU_DEVICE_TESTS": "tests/ascend/agent/test_surrogate_batch_npu.py",
            },
        )
        payload = {
            "source_root": str(root),
            "cannbench_repo": str(cann_repo),
            "tilelang_cache_dir": str(tl_cache_dir),
            "warmup": int(warmup),
            "repeat": int(repeat),
            "require_all_passed": bool(require_all_passed),
            "response_path": str(response_path),
            "batches": [
                {
                    "operator_name": op_name,
                    "entrypoint": get_operator_entrypoint(op_name),
                    "candidates": [
                        {"candidate_id": cid, "module_path": str(Path(mod_path).resolve())}
                        for cid, mod_path in cands
                    ],
                    "anchors": [a.to_dict() for a in op_anchors],
                }
                for op_name, cands, op_anchors in batches
            ],
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
            "tilefoundry-npu-device",
            "--json",
        ]
        sub_env = _clean_agent_gates_subprocess_env(
            {
                "TF_SURROGATE_BATCH_REQUEST": str(request_path),
                "TF_SURROGATE_TL_CACHE_DIR": str(tl_cache_dir),
                "CANNBENCH_CACHE_DIR": str(cache_dir),
            }
        )
        proc = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            cwd=str(root),
            env=sub_env,
            timeout=timeout_sec,
        )
        parsed = _parse_agent_gates_json(proc.stdout)
        run_id = str(parsed.get("runId", ""))
        receipt_path = str(parsed.get("receiptPath", ""))
        held_sec = 0.0
        for gate_row in parsed.get("gates", ()):
            if isinstance(gate_row, dict) and gate_row.get("gateId") == "tilefoundry-npu-device":
                obs = gate_row.get("observations") or {}
                win = obs.get("deviceWindow") or {}
                held_sec = float(win.get("heldSeconds", 0.0))

        if not response_path.is_file():
            detail = (proc.stderr or proc.stdout or "surrogate batch missing response").strip()[-1000:]
            raise RuntimeError(
                f"AgentGates L2 surrogate batch failed (rc={proc.returncode}): {detail}"
            )
        raw_resp = json.loads(response_path.read_text(encoding="utf-8"))
        out_by_op: dict[str, dict[str, CandidateSurrogateResult]] = {}
        for op_name, cands_map in raw_resp.get("operators", {}).items():
            out_by_op[op_name] = {
                cid: CandidateSurrogateResult.from_dict(item)
                for cid, item in cands_map.items()
            }
        if proc.returncode != 0 or parsed.get("status") != "PASS":
            detail = (proc.stderr or proc.stdout or "agent-gates status FAIL").strip()[-1000:]
            if require_all_passed:
                raise RuntimeError(f"AgentGates L2 batch rejected candidate(s): {detail}")
        return out_by_op, run_id, receipt_path, held_sec


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


def run_l3_full_contract_via_agent_gates(
    *,
    source_root: str | Path | None = None,
    env_file: str | Path | None = None,
    device_id: str = "1",
    timeout_sec: int = 600,
) -> L3AcceptanceResult:
    """Run full 20-case + dynamic-cache test_npu_acceptance.py via AgentGates L3 gate."""
    t0 = time.monotonic()
    root = Path(source_root).resolve() if source_root is not None else default_source_root()
    resolved_env = _resolve_env_file_path(env_file)
    ag_bin = _resolve_agent_gates_bin(root)
    env_vals = parse_simple_env_file(resolved_env)
    cann_repo = resolve_cannbench_repo(
        cannbench_repo=env_vals.get("TILEFOUNDRY_NPU_CANNBENCH_REPO"),
        env_file=resolved_env,
    )
    cache_dir = _default_surrogate_cache_dir()

    with tempfile.TemporaryDirectory(prefix="tf_l3_gate_") as tmp_dir:
        tmp = Path(tmp_dir)
        shim_dir = ensure_npu_compare_shim_dir(tmp / "shim", cannbench_repo=cann_repo)
        tmp_env = _write_temp_env_file(
            resolved_env,
            tmp / "env.sh",
            {
                "TILEFOUNDRY_NPU_CPU_TESTS": "tests/ascend/agent/test_llm_watchdog.py",
                "TILEFOUNDRY_NPU_DEVICE_TESTS": (
                    "tests/ascend/agent/test_surrogate_batch_npu.py "
                    "tests/ascend/benchmarks/test_npu_acceptance.py"
                ),
            },
        )
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
            "tilefoundry-npu-device",
            "--json",
        ]
        prev_pp = os.environ.get("PYTHONPATH", "")
        new_pp = f"{shim_dir}{os.pathsep}{prev_pp}" if prev_pp else str(shim_dir)
        sub_env = _clean_agent_gates_subprocess_env(
            {
                "PYTHONPATH": new_pp,
                "CANNBENCH_CACHE_DIR": str(cache_dir),
                "TF_SURROGATE_SHIM_ONLY": "1",
                "TF_SURROGATE_SHIM_DIR": str(shim_dir),
            }
        )
        proc = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            cwd=str(root),
            env=sub_env,
            timeout=timeout_sec,
        )
    parsed = _parse_agent_gates_json(proc.stdout)
    ok = proc.returncode == 0 and parsed.get("status") == "PASS"
    err = "" if ok else (proc.stderr or proc.stdout or "L3 contract failed")[-1000:]
    return L3AcceptanceResult(
        passed=ok,
        run_id=str(parsed.get("runId", "")),
        receipt_path=str(parsed.get("receiptPath", "")),
        elapsed_sec=time.monotonic() - t0,
        error_message=err,
    )


def run_l4_official_cannbench_via_agent_gates(
    *,
    operator_name: str | None = None,
    source_root: str | Path | None = None,
    env_file: str | Path | None = None,
    device_id: str = "1",
    timeout_sec: int = 3600,
) -> L4CannbenchResult:
    """Run official tilefoundry-npu-cannbench gate once and parse per-case metrics."""
    root = Path(source_root).resolve() if source_root is not None else default_source_root()
    resolved_env = _resolve_env_file_path(env_file)
    ag_bin = _resolve_agent_gates_bin(root)

    with tempfile.TemporaryDirectory(prefix="tf_l4_gate_") as tmp_dir:
        tmp = Path(tmp_dir)
        overrides: dict[str, str] = {}
        if operator_name is not None:
            task = get_operator_task(operator_name)
            rel_cand = (
                "examples/ascend950pr_ops/attention_residual/kernels.py"
                if operator_name == "attention_residual"
                else "examples/ascend950pr_ops/mla/kernels.py"
            )
            overrides["TILEFOUNDRY_NPU_CANNBENCH_TARGETS"] = f"{task}:{rel_cand}"
        tmp_env = _write_temp_env_file(resolved_env, tmp / "env.sh", overrides)
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
            "tilefoundry-npu-cannbench",
            "--json",
        ]
        proc = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            cwd=str(root),
            env=_clean_agent_gates_subprocess_env({}),
            timeout=timeout_sec,
        )

    parsed = _parse_agent_gates_json(proc.stdout)
    ok = proc.returncode == 0 and parsed.get("status") == "PASS"
    run_id = str(parsed.get("runId", ""))
    receipt_path = str(parsed.get("receiptPath", ""))

    avg_speedup = 0.0
    score = 0.0
    report_dir = ""
    per_case_speedups: dict[int, float] = {}
    per_case_elapsed: dict[int, float] = {}

    target_task = get_operator_task(operator_name) if operator_name else ""
    for gate_row in parsed.get("gates", ()):
        if not isinstance(gate_row, dict) or gate_row.get("gateId") != "tilefoundry-npu-cannbench":
            continue
        obs = gate_row.get("observations") or {}
        avg_speedup = float(obs.get("avgSpeedup", 0.0))
        score = float(obs.get("score", 0.0))
        for res_row in obs.get("results", ()):
            if not isinstance(res_row, dict):
                continue
            if target_task and res_row.get("task") != target_task:
                continue
            report_dir = str(res_row.get("reportDir", ""))
            final_json = Path(report_dir) / "official_final.json"
            if final_json.is_file():
                final_data = json.loads(final_json.read_text(encoding="utf-8"))
                op_entry = final_data["operators"][0]
                avg_speedup = float(op_entry.get("avg_speedup", avg_speedup))
                score = float(op_entry.get("score", score))
                for case_row in op_entry.get("cases", ()):
                    cid_str = str(case_row.get("case_id", ""))
                    if "_" in cid_str and cid_str.rsplit("_", 1)[-1].isdigit():
                        cnum = int(cid_str.rsplit("_", 1)[-1])
                        per_case_speedups[cnum] = float(case_row.get("speedup", 0.0))
                        per_case_elapsed[cnum] = float(case_row.get("elapsed_us", 0.0))

    err = "" if ok else (proc.stderr or proc.stdout or "L4 CANNBench gate failed")[-1000:]
    return L4CannbenchResult(
        passed=ok,
        avg_speedup=avg_speedup,
        score=score,
        per_case_speedups=per_case_speedups,
        per_case_elapsed_us=per_case_elapsed,
        run_id=run_id,
        receipt_path=receipt_path,
        report_dir=report_dir,
        error_message=err,
    )


__all__ = [
    "AnchorEvalMetrics",
    "CandidateSurrogateResult",
    "L0L1CandidateCheck",
    "L3AcceptanceResult",
    "L4CannbenchResult",
    "aggregate_surrogate_result",
    "build_anchor_metrics",
    "ensure_npu_compare_shim_dir",
    "prewarm_anchor_reference_cache",
    "prewarm_surrogate_tilelang_cache",
    "run_l0_l1_round_filter",
    "run_l0_static_precheck",
    "run_l1_compile_check",
    "run_l3_full_contract_via_agent_gates",
    "run_l4_official_cannbench_via_agent_gates",
    "run_multi_operator_surrogate_batch_via_agent_gates",
    "run_surrogate_batch_via_agent_gates",
]
