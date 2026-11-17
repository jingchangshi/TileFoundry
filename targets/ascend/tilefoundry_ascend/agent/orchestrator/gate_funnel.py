"""Three-stage verification funnel: Stage A static ledger -> Stage B bisheng -> Stage C NPU."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import contextlib
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


_HIR_TO_BUILDER_DIM_MAP: dict[str, str] = {
    "N": "n_depth",
    "B": "b_dim",
    "S": "s_dim",
    "S_kv": "s_kv",
    "D": "d_dim",
    "N_q": "n_q",
    "N_kv": "n_kv",
    "D_nope": "d_nope",
    "D_rope": "d_rope",
}


def _normalize_flag_list(raw_flags: object) -> list[str]:
    if raw_flags is None:
        return []
    if isinstance(raw_flags, str):
        text = raw_flags.strip()
        return text.split() if text else []
    if isinstance(raw_flags, (tuple, list)) or (
        hasattr(raw_flags, "__iter__") and not isinstance(raw_flags, (bytes, bytearray, Mapping))
    ):
        out: list[str] = []
        for item in raw_flags:
            val = str(getattr(item, "value", item)).strip()
            if val and val not in out:
                out.append(val)
        return out
    text = str(raw_flags).strip()
    return [text] if text else []


def _filter_kwargs_for_callable(fn: object, kwargs: Mapping[str, object]) -> dict[str, object]:
    import inspect

    sig = inspect.signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def _merge_dims_into_kwargs(
    base_kwargs: Mapping[str, object] | None,
    dims: Mapping[str, object] | None,
) -> dict[str, object]:
    merged = dict(base_kwargs or {})
    if not dims:
        return merged
    for key, val in dims.items():
        mapped = _HIR_TO_BUILDER_DIM_MAP.get(str(key), str(key))
        merged[mapped] = val
    return merged


def _split_selector_output(
    sel_out: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, object], list[str]]:
    pc: dict[str, object] = {}
    cf: list[str] = []
    builder_overrides: dict[str, object] = {}

    raw_pc = sel_out.get("pass_configs")
    if isinstance(raw_pc, Mapping):
        for k, v in raw_pc.items():
            if str(k) == "tl.device_compile_flags":
                for flag in _normalize_flag_list(v):
                    if flag not in cf:
                        cf.append(flag)
            else:
                pc[str(k)] = v
    elif raw_pc is not None:
        for flag in _normalize_flag_list(raw_pc):
            if flag not in cf:
                cf.append(flag)

    for flag in _normalize_flag_list(sel_out.get("compile_flags")):
        if flag not in cf:
            cf.append(flag)

    for k, v in sel_out.items():
        if k not in ("pass_configs", "compile_flags", "builder_kwargs"):
            builder_overrides[str(k)] = v
    raw_bk = sel_out.get("builder_kwargs")
    if isinstance(raw_bk, Mapping):
        for k, v in raw_bk.items():
            builder_overrides[str(k)] = v
    return builder_overrides, pc, cf


def _ascend_target_context():
    try:
        from tilelang import tvm
        from tilelang.backend.module import create_backend_context

        if tvm.target.Target.current(allow_none=True) is None:
            return create_backend_context("ascend").target
    except Exception:
        pass
    return contextlib.nullcontext()


def _resolve_prim_func_and_configs_from_dsl(
    dsl_source: str,
    *,
    factory_name: str | None = None,
    factory_kwargs: Mapping[str, object] | None = None,
    config_selector_name: str | None = None,
    dims: Mapping[str, object] | None = None,
) -> tuple[object, dict[str, object], list[str]]:
    """Load DSL source via a temp module, invoke selector if present, and build PrimFunc."""
    import ast
    import importlib.util

    try:
        import tilelang
    except ImportError:
        tilelang = None
    _ = tilelang
    from tvm import IRModule, tirx

    ast.parse(dsl_source)
    sel_pass_configs: dict[str, object] = {}
    sel_compile_flags: list[str] = []
    effective_kwargs = _merge_dims_into_kwargs(factory_kwargs, dims)

    with tempfile.TemporaryDirectory(prefix="tf_stage_a_") as tmp_dir:
        mod_path = Path(tmp_dir) / "candidate_dsl.py"
        mod_path.write_text(dsl_source, encoding="utf-8")
        spec = importlib.util.spec_from_file_location("_tf_stage_a_cand", str(mod_path))
        if spec is None or spec.loader is None:
            raise RuntimeError("failed to create module spec for candidate DSL")
        mod = importlib.util.module_from_spec(spec)
        with _ascend_target_context():
            spec.loader.exec_module(mod)

            if config_selector_name and hasattr(mod, config_selector_name):
                selector_obj = getattr(mod, config_selector_name)
                if callable(selector_obj):
                    sel_args = _filter_kwargs_for_callable(selector_obj, effective_kwargs)
                    sel_raw = selector_obj(**sel_args)
                    if isinstance(sel_raw, Mapping):
                        overrides, sel_pass_configs, sel_compile_flags = _split_selector_output(
                            sel_raw
                        )
                        effective_kwargs.update(overrides)

            if factory_name and hasattr(mod, factory_name):
                target_obj = getattr(mod, factory_name)
                if callable(target_obj):
                    call_kw = _filter_kwargs_for_callable(target_obj, effective_kwargs)
                    return target_obj(**call_kw), sel_pass_configs, sel_compile_flags
                return target_obj, sel_pass_configs, sel_compile_flags

            for val in vars(mod).values():
                if isinstance(val, (tirx.PrimFunc, IRModule)):
                    return val, sel_pass_configs, sel_compile_flags

            for name, val in vars(mod).items():
                if (
                    not name.startswith("_")
                    and callable(val)
                    and getattr(val, "__module__", None) == mod.__name__
                ):
                    call_kw = _filter_kwargs_for_callable(val, effective_kwargs)
                    return val(**call_kw), sel_pass_configs, sel_compile_flags
    raise ValueError("DSL source defined no callable factory or PrimFunc")


def _resolve_prim_func_from_dsl(
    dsl_source: str,
    *,
    factory_name: str | None = None,
    factory_kwargs: Mapping[str, object] | None = None,
):
    """Load DSL source via a temporary module so @T.prim_func sees file lines cleanly."""
    prim_func, _pc, _cf = _resolve_prim_func_and_configs_from_dsl(
        dsl_source,
        factory_name=factory_name,
        factory_kwargs=factory_kwargs,
    )
    return prim_func


def _merge_pass_configs_and_flags(
    sel_pc: Mapping[str, object] | None,
    sel_cf: Sequence[str] | str | None,
    ext_pc: Mapping[str, object] | None,
    ext_cf: Sequence[str] | str | None,
) -> tuple[dict[str, object], list[str]]:
    merged_pc: dict[str, object] = {}
    merged_cf: list[str] = []

    def _absorb(pc_map: Mapping[str, object] | None, cf_seq: Sequence[str] | str | None) -> None:
        if pc_map:
            for k, v in pc_map.items():
                if str(k) == "tl.device_compile_flags":
                    for flag in _normalize_flag_list(v):
                        if flag not in merged_cf:
                            merged_cf.append(flag)
                else:
                    merged_pc[str(k)] = v
        for flag in _normalize_flag_list(cf_seq):
            if flag not in merged_cf:
                merged_cf.append(flag)

    _absorb(sel_pc, sel_cf)
    _absorb(ext_pc, ext_cf)
    return merged_pc, merged_cf


def run_stage_a_static(
    dsl_source: str,
    *,
    factory_name: str | None = None,
    factory_kwargs: Mapping[str, object] | None = None,
    target: AscendTarget | None = None,
    pass_configs: dict[str, object] | None = None,
    compile_flags: Sequence[str] | str | None = None,
    config_selector_name: str | None = None,
    dims: Mapping[str, object] | None = None,
    var_bindings: Mapping[str, int] | None = None,
    enable_track_router: bool = False,
) -> StageAResult:
    """Run millisecond Stage A AST, Two-Snapshot IR Diff, and optional track routing."""
    t0 = time.monotonic()
    try:
        with _ascend_target_context():
            prim_func, sel_pc, sel_cf = _resolve_prim_func_and_configs_from_dsl(
                dsl_source,
                factory_name=factory_name,
                factory_kwargs=factory_kwargs,
                config_selector_name=config_selector_name,
                dims=dims,
            )
            merged_pc, merged_cf = _merge_pass_configs_and_flags(
                sel_pc,
                sel_cf,
                pass_configs,
                compile_flags,
            )
            inspection: EnvelopeInspectionResult = inspect_tilelang_primfunc(
                prim_func,
                target=target,
                pass_configs=merged_pc or None,
                compile_flags=merged_cf or None,
                var_bindings=var_bindings,
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


def build_stage_b_driver_source(
    cand_path: str | Path,
    *,
    factory_name: str | None = None,
    factory_kwargs: Mapping[str, object] | None = None,
    config_selector_name: str | None = None,
    pass_configs: Mapping[str, object] | None = None,
    compile_flags: Sequence[str] | str | None = None,
    dims: Mapping[str, object] | None = None,
    enable_device_compile: bool = True,
) -> str:
    """Build standalone Python driver lowering and device-compiling a candidate DSL file."""
    eff_kwargs = _merge_dims_into_kwargs(factory_kwargs, dims)
    ext_pc, ext_cf = _merge_pass_configs_and_flags(None, None, pass_configs, compile_flags)
    payload = json.dumps(
        {
            "cand_path": str(cand_path),
            "factory_name": factory_name,
            "factory_kwargs": eff_kwargs,
            "config_selector_name": config_selector_name,
            "pass_configs": ext_pc,
            "compile_flags": ext_cf,
            "enable_device_compile": bool(enable_device_compile),
        }
    )
    return (
        "import contextlib, importlib.util, inspect, json, tilelang\n"
        "from tilelang import tvm\n"
        "from tilelang.backend.module import create_backend_context\n"
        "from tvm import tirx, IRModule\n"
        "def _ascend_ctx():\n"
        "    if tvm.target.Target.current(allow_none=True) is None:\n"
        "        return create_backend_context('ascend').target\n"
        "    return contextlib.nullcontext()\n"
        f"_cfg = json.loads({payload!r})\n"
        "_spec = importlib.util.spec_from_file_location('_tf_stage_b', _cfg['cand_path'])\n"
        "_mod = importlib.util.module_from_spec(_spec)\n"
        "def _filt(obj, kw):\n"
        "    sig = inspect.signature(obj)\n"
        "    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):\n"
        "        return dict(kw)\n"
        "    return {k: v for k, v in kw.items() if k in sig.parameters}\n"
        "def _norm_flags(raw):\n"
        "    if raw is None:\n"
        "        return []\n"
        "    if isinstance(raw, str):\n"
        "        return raw.split()\n"
        "    return [str(getattr(x, 'value', x)).strip() for x in raw if str(x).strip()]\n"
        "_kw = dict(_cfg['factory_kwargs'])\n"
        "_sel_pc, _sel_cf = {}, []\n"
        "_fn = None\n"
        "with _ascend_ctx():\n"
        "    _spec.loader.exec_module(_mod)\n"
        "    _sname = _cfg['config_selector_name']\n"
        "    if _sname and callable(getattr(_mod, _sname, None)):\n"
        "        _sfn = getattr(_mod, _sname)\n"
        "        _sout = _sfn(**_filt(_sfn, _kw))\n"
        "        if isinstance(_sout, dict):\n"
        "            _raw_pc = _sout.get('pass_configs')\n"
        "            if isinstance(_raw_pc, dict):\n"
        "                for _k, _v in _raw_pc.items():\n"
        "                    if _k == 'tl.device_compile_flags':\n"
        "                        _sel_cf.extend(_norm_flags(_v))\n"
        "                    else:\n"
        "                        _sel_pc[str(_k)] = _v\n"
        "            elif _raw_pc is not None:\n"
        "                _sel_cf.extend(_norm_flags(_raw_pc))\n"
        "            _sel_cf.extend(_norm_flags(_sout.get('compile_flags')))\n"
        "            for _k, _v in _sout.items():\n"
        "                if _k not in ('pass_configs', 'compile_flags'):\n"
        "                    _kw[_k] = _v\n"
        "    _fname = _cfg['factory_name']\n"
        "    if _fname and hasattr(_mod, _fname):\n"
        "        _obj = getattr(_mod, _fname)\n"
        "        _fn = _obj(**_filt(_obj, _kw)) if callable(_obj) else _obj\n"
        "    if _fn is None:\n"
        "        for _v in vars(_mod).values():\n"
        "            if isinstance(_v, (tirx.PrimFunc, IRModule)):\n"
        "                _fn = _v; break\n"
        "    if _fn is None:\n"
        "        for _k, _v in vars(_mod).items():\n"
        "            if (not _k.startswith('_') and callable(_v)\n"
        "                    and getattr(_v, '__module__', None) == _mod.__name__):\n"
        "                _fn = _v(**_filt(_v, _kw)); break\n"
        "assert _fn is not None\n"
        "_merged_pc, _merged_cf = {}, []\n"
        "def _absorb_flags(items):\n"
        "    for _f in items:\n"
        "        if _f and _f not in _merged_cf:\n"
        "            _merged_cf.append(_f)\n"
        "_attrs = getattr(_fn, 'attrs', None)\n"
        "if _attrs and hasattr(_attrs, 'items'):\n"
        "    for _pk in ('tl.pass_configs', 'tilelang_pass_configs'):\n"
        "        if _pk in _attrs and hasattr(_attrs[_pk], 'items'):\n"
        "            for _ak, _av in _attrs[_pk].items():\n"
        "                _val = getattr(_av, 'value', _av)\n"
        "                if str(_ak) == 'tl.device_compile_flags':\n"
        "                    _absorb_flags(_norm_flags(_val))\n"
        "                else:\n"
        "                    _merged_pc[str(_ak)] = _val\n"
        "    for _ck in ('tir.cce_flags', 'tilelang_compile_flags'):\n"
        "        if _ck in _attrs:\n"
        "            _absorb_flags(_norm_flags(_attrs[_ck]))\n"
        "_merged_pc.update(_sel_pc)\n"
        "_absorb_flags(_sel_cf)\n"
        "_merged_pc.update(_cfg['pass_configs'])\n"
        "_absorb_flags(_cfg['compile_flags'])\n"
        "if _merged_cf:\n"
        "    _merged_pc['tl.device_compile_flags'] = list(_merged_cf)\n"
        "with tvm.transform.PassContext(config=_merged_pc):\n"
        "    _art = tilelang.lower(\n"
        "        _fn, target='ascend', enable_device_compile=_cfg['enable_device_compile']\n"
        "    )\n"
        "assert _art.kernel_source\n"
    )


def run_stage_b_compile(
    dsl_source: str,
    *,
    factory_name: str | None = None,
    factory_kwargs: Mapping[str, object] | None = None,
    config_selector_name: str | None = None,
    pass_configs: Mapping[str, object] | None = None,
    compile_flags: Sequence[str] | str | None = None,
    dims: Mapping[str, object] | None = None,
    python_bin: str | None = None,
    timeout_sec: int = 120,
    enable_device_compile: bool = True,
) -> StageBResult:
    """Verify full TileLang + bisheng device compilation in an isolated subprocess."""
    t0 = time.monotonic()
    py_exe = _resolve_conda_python(python_bin)
    env = dict(os.environ)
    env["TILELANG_DISABLE_CACHE"] = "1"
    env.setdefault("USE_CUDA", "0")
    with tempfile.TemporaryDirectory(prefix="tf_stage_b_") as tmp_dir:
        cand_path = Path(tmp_dir) / "candidate_dsl.py"
        cand_path.write_text(dsl_source, encoding="utf-8")
        driver = build_stage_b_driver_source(
            cand_path,
            factory_name=factory_name,
            factory_kwargs=factory_kwargs,
            config_selector_name=config_selector_name,
            pass_configs=pass_configs,
            compile_flags=compile_flags,
            dims=dims,
            enable_device_compile=enable_device_compile,
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
        config_selector_name: str | None = None,
        pass_configs: dict[str, object] | None = None,
        compile_flags: Sequence[str] | str | None = None,
        dims: Mapping[str, object] | None = None,
        var_bindings: Mapping[str, int] | None = None,
    ) -> FunnelEvaluation:
        """Run Stage A, Stage B, and Stage C sequentially with early rejection."""
        stage_a = run_stage_a_static(
            dsl_source,
            factory_name=factory_name,
            factory_kwargs=factory_kwargs,
            target=self._target,
            pass_configs=pass_configs,
            compile_flags=compile_flags,
            config_selector_name=config_selector_name,
            dims=dims,
            var_bindings=var_bindings,
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
                config_selector_name=config_selector_name,
                pass_configs=pass_configs,
                compile_flags=compile_flags,
                dims=dims,
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
    var_bindings: Mapping[str, int] | None = None,
    config_selector_name: str | None = None,
    pass_configs: dict[str, object] | None = None,
    compile_flags: Sequence[str] | str | None = None,
    dims: Mapping[str, object] | None = None,
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
            pass_configs=pass_configs,
            compile_flags=compile_flags,
            config_selector_name=config_selector_name,
            dims=dims,
            var_bindings=var_bindings,
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
    "build_stage_b_driver_source",
    "ledger_to_json",
    "prescreen_dsl_tile_kwargs",
    "run_stage_a_static",
    "run_stage_b_compile",
    "run_stage_c_agent_gates",
]
