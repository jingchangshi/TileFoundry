"""Three-stage verification funnel: Stage A static ledger -> Stage B bisheng -> Stage C NPU."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from tilefoundry_ascend.agent.orchestrator.residual_beam import (
    StaticDynamicResidual,
    compute_static_dynamic_residual,
)
from tilefoundry_ascend.ascend.passes.physical_ledger import (
    ObjectivePhysicalLedger,
    build_objective_physical_ledger,
)
from tilefoundry_ascend.ascend.schedule.envelope_inspector import (
    EnvelopeInspectionResult,
    inspect_tilelang_primfunc,
)
from tilefoundry_ascend.ascend.target import AscendTarget


@dataclass(frozen=True)
class StageAResult:
    """Output of millisecond Stage A static AST, ledger, and router inspection."""

    passed: bool
    prim_func: object | None
    ledger: ObjectivePhysicalLedger | None
    features: object | None = None
    route: object | None = None
    error_message: str = ""
    elapsed_ms: float = 0.0


@dataclass(frozen=True)
class StageBResult:
    """Output of Stage B Conda bisheng compilation check."""

    passed: bool
    error_message: str = ""
    elapsed_sec: float = 0.0


@dataclass(frozen=True)
class StageCResult:
    """Output of Stage C lease-locked AgentGates NPU verification and msprof telemetry."""

    passed: bool
    accuracy_passed: bool
    measured_latency_us: float
    residual: StaticDynamicResidual | None
    telemetry: dict[str, object]
    run_id: str = ""
    receipt_path: str = ""
    error_message: str = ""


@dataclass(frozen=True)
class FunnelEvaluation:
    """Aggregate result across Stage A, Stage B, and Stage C verification."""

    passed: bool
    failed_stage: str | None
    stage_a: StageAResult
    stage_b: StageBResult | None = None
    stage_c: StageCResult | None = None
    error_message: str = ""


def _resolve_prim_func_from_dsl(
    dsl_source: str,
    *,
    factory_name: str | None = None,
    factory_kwargs: Mapping[str, object] | None = None,
):
    """Load DSL source via a temporary module so @T.prim_func sees file lines cleanly."""
    import ast
    import importlib.util

    try:
        import tilelang
    except ImportError:
        tilelang = None
    _ = tilelang
    from tvm import IRModule, tirx

    ast.parse(dsl_source)
    with tempfile.TemporaryDirectory(prefix="tf_stage_a_") as tmp_dir:
        mod_path = Path(tmp_dir) / "candidate_dsl.py"
        mod_path.write_text(dsl_source, encoding="utf-8")
        spec = importlib.util.spec_from_file_location("_tf_stage_a_cand", str(mod_path))
        if spec is None or spec.loader is None:
            raise RuntimeError("failed to create module spec for candidate DSL")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        if factory_name and hasattr(mod, factory_name):
            target_obj = getattr(mod, factory_name)
            if callable(target_obj):
                return target_obj(**dict(factory_kwargs or {}))
            return target_obj

        for val in vars(mod).values():
            if isinstance(val, (tirx.PrimFunc, IRModule)):
                return val

        for name, val in vars(mod).items():
            if not name.startswith("_") and callable(val) and getattr(val, "__module__", None) == mod.__name__:
                return val(**dict(factory_kwargs or {}))
    raise ValueError("DSL source defined no callable factory or PrimFunc")


def run_stage_a_static(
    dsl_source: str,
    *,
    factory_name: str | None = None,
    factory_kwargs: Mapping[str, object] | None = None,
    target: AscendTarget | None = None,
    pass_configs: dict[str, object] | None = None,
    enable_track_router: bool = False,
) -> StageAResult:
    """Run millisecond Stage A AST, Two-Snapshot IR Diff, and optional track routing."""
    t0 = time.monotonic()
    try:
        prim_func = _resolve_prim_func_from_dsl(
            dsl_source,
            factory_name=factory_name,
            factory_kwargs=factory_kwargs,
        )
        inspection: EnvelopeInspectionResult = inspect_tilelang_primfunc(
            prim_func,
            target=target,
            pass_configs=pass_configs,
        )
    except Exception as exc:
        return StageAResult(
            passed=False,
            prim_func=None,
            ledger=None,
            features=None,
            route=None,
            error_message=f"stage_a_parse_or_lower_error: {type(exc).__name__}: {exc}",
            elapsed_ms=(time.monotonic() - t0) * 1000.0,
        )

    ledger = inspection.ledger
    features: object | None = None
    route: object | None = None
    if enable_track_router:
        from tilefoundry_ascend.agent.router.capability_gate import (
            extract_structural_features,
            route_execution_track,
        )

        features = extract_structural_features(prim_func, ledger=ledger, target=target)
        route = route_execution_track(features)
    if not ledger.is_legal:
        reasons: list[str] = []
        if ledger.max_overflow_bytes > 0:
            reasons.append(f"max_overflow_bytes={ledger.max_overflow_bytes}")
        if inspection.lowering_error:
            reasons.append(inspection.lowering_error)
        if not reasons:
            reasons.append("illegal_datapath_or_capacity")
        return StageAResult(
            passed=False,
            prim_func=prim_func,
            ledger=ledger,
            features=features,
            route=route,
            error_message="; ".join(reasons),
            elapsed_ms=(time.monotonic() - t0) * 1000.0,
        )
    return StageAResult(
        passed=True,
        prim_func=prim_func,
        ledger=ledger,
        features=features,
        route=route,
        elapsed_ms=(time.monotonic() - t0) * 1000.0,
    )


def _resolve_conda_python(explicit_python: str | None = None) -> str:
    candidate = (explicit_python or os.environ.get("TILEFOUNDRY_NPU_PYTHON", "")).strip()
    if candidate and Path(candidate).is_file():
        return candidate
    return sys.executable


def run_stage_b_compile(
    dsl_source: str,
    *,
    factory_name: str | None = None,
    factory_kwargs: Mapping[str, object] | None = None,
    python_bin: str | None = None,
    timeout_sec: int = 120,
    enable_device_compile: bool = True,
) -> StageBResult:
    """Verify full TileLang + bisheng device compilation in an isolated subprocess."""
    t0 = time.monotonic()
    py_exe = _resolve_conda_python(python_bin)
    kwargs_json = json.dumps(dict(factory_kwargs or {}))
    env = dict(os.environ)
    env["TILELANG_DISABLE_CACHE"] = "1"
    env.setdefault("USE_CUDA", "0")
    with tempfile.TemporaryDirectory(prefix="tf_stage_b_") as tmp_dir:
        cand_path = Path(tmp_dir) / "candidate_dsl.py"
        cand_path.write_text(dsl_source, encoding="utf-8")
        driver = (
            "import importlib.util, json, tilelang\n"
            "from tvm import tirx, IRModule\n"
            f"spec = importlib.util.spec_from_file_location('_tf_stage_b', {str(cand_path)!r})\n"
            "mod = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(mod)\n"
            f"fname = {factory_name!r}\n"
            f"kw = json.loads({kwargs_json!r})\n"
            "fn = getattr(mod, fname)(**kw) if fname and callable(getattr(mod, fname, None)) else None\n"
            "if fn is None:\n"
            "    for v in vars(mod).values():\n"
            "        if isinstance(v, (tirx.PrimFunc, IRModule)):\n"
            "            fn = v; break\n"
            "if fn is None:\n"
            "    for k, v in vars(mod).items():\n"
            "        if not k.startswith('_') and callable(v) and getattr(v, '__module__', None) == mod.__name__:\n"
            "            fn = v(**kw); break\n"
            f"art = tilelang.lower(fn, target='ascend', enable_device_compile={enable_device_compile!r})\n"
            "assert art.kernel_source\n"
        )
        try:
            proc = subprocess.run(
                [py_exe, "-c", driver],
                text=True,
                capture_output=True,
                cwd=tmp_dir,
                env=env,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired:
            return StageBResult(
                passed=False,
                error_message=f"stage_b_compile_timeout:{timeout_sec}s",
                elapsed_sec=time.monotonic() - t0,
            )
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()[-1500:]
        return StageBResult(
            passed=False,
            error_message=f"stage_b_compile_failed (rc={proc.returncode}): {detail}",
            elapsed_sec=time.monotonic() - t0,
        )
    return StageBResult(passed=True, elapsed_sec=time.monotonic() - t0)


def _resolve_agent_gates_bin() -> str:
    env_bin = os.environ.get("TILEFOUNDRY_NPU_AGENT_GATES_BIN", "").strip()
    if env_bin and Path(env_bin).is_file():
        return env_bin
    found = shutil.which("agent-gates")
    if found:
        return found
    raise FileNotFoundError(
        "Cannot locate 'agent-gates'. Set TILEFOUNDRY_NPU_AGENT_GATES_BIN or add to PATH."
    )


def run_stage_c_agent_gates(
    ledger: ObjectivePhysicalLedger,
    *,
    source_root: str | Path,
    device_id: str = "1",
    env_file: str | Path | None = None,
    device_test_rel: str | None = None,
    telemetry_report_path: str | Path | None = None,
    gate: str = "tilefoundry-npu-device",
    timeout_sec: int = 300,
) -> StageCResult:
    """Execute lease-locked NPU validation and msprof collection strictly via AgentGates."""
    ag_bin = _resolve_agent_gates_bin()
    src_path = Path(source_root).resolve()
    cmd = [
        ag_bin,
        "run",
        "--adapter",
        "tilefoundry-npu",
        "--source",
        str(src_path),
        "--gate",
        gate,
        "--device-id",
        str(device_id),
        "--json",
    ]
    with tempfile.TemporaryDirectory(prefix="tf_stage_c_env_") as tmp_dir:
        if device_test_rel and env_file and Path(env_file).is_file():
            tmp_env = Path(tmp_dir) / "env.sh"
            lines = [
                ln
                for ln in Path(env_file).read_text(encoding="utf-8").splitlines()
                if not ln.startswith("TILEFOUNDRY_NPU_DEVICE_TESTS=")
            ]
            lines.append(f'TILEFOUNDRY_NPU_DEVICE_TESTS="{device_test_rel}"')
            tmp_env.write_text("\n".join(lines) + "\n", encoding="utf-8")
            cmd.extend(["--env-file", str(tmp_env)])
        elif env_file and Path(env_file).is_file():
            cmd.extend(["--env-file", str(env_file)])

        proc = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
            cwd=str(src_path),
            timeout=timeout_sec,
        )

    parsed: dict[str, object] = {}
    for line in proc.stdout.splitlines():
        pass
    try:
        idx = proc.stdout.find("{")
        if idx >= 0:
            parsed = json.loads(proc.stdout[idx:])
    except json.JSONDecodeError:
        parsed = {}

    status = str(parsed.get("status", "FAIL"))
    run_id = str(parsed.get("runId", ""))
    receipt_path = str(parsed.get("receiptPath", ""))
    passed = proc.returncode == 0 and status == "PASS"

    telemetry: dict[str, object] = {}
    if telemetry_report_path and Path(telemetry_report_path).is_file():
        try:
            telemetry = json.loads(Path(telemetry_report_path).read_text(encoding="utf-8"))
        except Exception:
            telemetry = {}

    latency_us = float(telemetry.get("dur_us", telemetry.get("measured_latency_us", 0.0)))
    residual = compute_static_dynamic_residual(ledger, telemetry)
    err = "" if passed else (proc.stderr or proc.stdout or "agent-gates failed")[-1000:]
    return StageCResult(
        passed=passed,
        accuracy_passed=passed,
        measured_latency_us=latency_us,
        residual=residual,
        telemetry=telemetry,
        run_id=run_id,
        receipt_path=receipt_path,
        error_message=err,
    )


class GateFunnel:
    """Orchestrates Stage A static -> Stage B bisheng -> Stage C AgentGates NPU verification."""

    def __init__(
        self,
        *,
        target: AscendTarget | None = None,
        skip_stage_b: bool = False,
        compile_runner: Callable[[str], StageBResult] | None = None,
        device_runner: Callable[[str, StageAResult], StageCResult] | None = None,
        enable_track_router: bool = False,
    ) -> None:
        self._target = target or AscendTarget("ascend.950pr")
        self._skip_stage_b = skip_stage_b
        self._compile_runner = compile_runner
        self._device_runner = device_runner
        self._enable_track_router = enable_track_router

    def evaluate(
        self,
        dsl_source: str,
        *,
        factory_name: str | None = None,
        factory_kwargs: Mapping[str, object] | None = None,
    ) -> FunnelEvaluation:
        """Run Stage A, Stage B, and Stage C sequentially with early rejection."""
        stage_a = run_stage_a_static(
            dsl_source,
            factory_name=factory_name,
            factory_kwargs=factory_kwargs,
            target=self._target,
            enable_track_router=self._enable_track_router,
        )
        if not stage_a.passed or stage_a.ledger is None:
            return FunnelEvaluation(
                passed=False,
                failed_stage="stage_a",
                stage_a=stage_a,
                error_message=stage_a.error_message,
            )

        if self._skip_stage_b:
            stage_b = StageBResult(passed=True)
        elif self._compile_runner is not None:
            stage_b = self._compile_runner(dsl_source)
        else:
            stage_b = run_stage_b_compile(
                dsl_source,
                factory_name=factory_name,
                factory_kwargs=factory_kwargs,
            )
        if not stage_b.passed:
            return FunnelEvaluation(
                passed=False,
                failed_stage="stage_b",
                stage_a=stage_a,
                stage_b=stage_b,
                error_message=stage_b.error_message,
            )

        if self._device_runner is None:
            fallback_telemetry = {"dur_us": 0.0}
            residual = compute_static_dynamic_residual(stage_a.ledger, fallback_telemetry)
            stage_c = StageCResult(
                passed=True,
                accuracy_passed=True,
                measured_latency_us=0.0,
                residual=residual,
                telemetry=fallback_telemetry,
            )
        else:
            stage_c = self._device_runner(dsl_source, stage_a)

        if not stage_c.passed or not stage_c.accuracy_passed:
            return FunnelEvaluation(
                passed=False,
                failed_stage="stage_c",
                stage_a=stage_a,
                stage_b=stage_b,
                stage_c=stage_c,
                error_message=stage_c.error_message,
            )

        return FunnelEvaluation(
            passed=True,
            failed_stage=None,
            stage_a=stage_a,
            stage_b=stage_b,
            stage_c=stage_c,
        )


def prescreen_dsl_tile_kwargs(
    dsl_source: str,
    *,
    factory_name: str,
    candidate_kwargs_list: Sequence[Mapping[str, object]],
    target: AscendTarget | None = None,
) -> tuple[dict[str, object], StageAResult]:
    """Run Stage A across tile kwargs and pick the lowest-cost legal configuration."""
    if not candidate_kwargs_list:
        raise ValueError("candidate_kwargs_list must not be empty")
    legal: list[tuple[tuple[float, int, int], dict[str, object], StageAResult]] = []
    rejections: list[str] = []
    for idx, raw_kwargs in enumerate(candidate_kwargs_list):
        kwargs = dict(raw_kwargs)
        res = run_stage_a_static(
            dsl_source,
            factory_name=factory_name,
            factory_kwargs=kwargs,
            target=target,
        )
        ledger = res.ledger
        if not res.passed or ledger is None:
            rejections.append(f"{kwargs}: {res.error_message or 'stage_a_failed'}")
            continue
        if not ledger.is_legal or ledger.max_overflow_bytes > 0:
            rejections.append(f"{kwargs}: overflow={ledger.max_overflow_bytes}")
            continue
        degraded = [
            buf
            for wl in ledger.waterlines.values()
            for buf in wl.degraded_single_buffers
        ]
        if degraded:
            rejections.append(f"{kwargs}: degraded_single_buffers={degraded}")
            continue
        ii_rank = (
            float(ledger.steady_state_ii)
            if ledger.steady_state_ii is not None and ledger.steady_state_ii > 0
            else float("inf")
        )
        dom_cycles = ledger.pipe_cycles.get(
            ledger.dominant_pipe,
            max((v for v in ledger.pipe_cycles.values() if v > 0), default=0),
        )
        legal.append(((ii_rank, dom_cycles, idx), kwargs, res))
    if not legal:
        detail = "; ".join(rejections[:5])
        raise ValueError(f"no legal tile kwargs passed Stage A prescreen: {detail}")
    legal.sort(key=lambda item: item[0])
    _, best_kwargs, best_stage_a = legal[0]
    return best_kwargs, best_stage_a


def ledger_to_json(ledger: ObjectivePhysicalLedger | None) -> str:
    """Serialize an optional ledger safely for worker prompts."""
    if ledger is None:
        return json.dumps(
            build_objective_physical_ledger(
                analysis_mode="readonly_envelope",
                is_legal=False,
                steady_state_ii=None,
                pipe_cycles={},
                waterlines={},
            ).to_dict(),
            indent=2,
        )
    return ledger.to_json()


__all__ = [
    "FunnelEvaluation",
    "GateFunnel",
    "StageAResult",
    "StageBResult",
    "StageCResult",
    "ledger_to_json",
    "prescreen_dsl_tile_kwargs",
    "run_stage_a_static",
    "run_stage_b_compile",
    "run_stage_c_agent_gates",
]
