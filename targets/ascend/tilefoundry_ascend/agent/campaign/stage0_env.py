"""Stage-0 seed preparation and marathon environment discovery utilities."""

from __future__ import annotations

import ast, asyncio, hashlib, logging, os, shutil, sys, tempfile, time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from tilefoundry_ascend.agent.campaign.helpers import compile_trackb_stage_b
from tilefoundry_ascend.agent.gates.gate_funnel import (
    check_npu_entrypoint_purity, evaluate_ascendc_candidate, extract_ordered_tensor_shapes, run_stage_a_static,
)
from tilefoundry_ascend.agent.gates.stage_b_gate import run_l1_compile_check
from tilefoundry_ascend.agent.npu import (
    _LLM_API_KEY_ENV, _NON_AGENT_GATES_ENV_KEYS, _format_env_line,
    _validate_llm_api_key_env, run_l3_full_contract_via_agent_gates,
)
from tilefoundry_ascend.agent.orchestrator.llm_client import KernelLLMBackend
from tilefoundry_ascend.agent.orchestrator.regime_slice_patcher import (
    _has_prim_func_builder, _render_trackb_tu_from_module, _resolve_selector_compile_flags, resolve_regime_config,
)
from tilefoundry_ascend.agent.orchestrator.seed_synthesizer import (
    GoldenEntrypointContract, OracleAccuracyResult, SeedSynthesisOutcome,
    extract_golden_entrypoint_contract, prepare_operator_seed_or_takeover,
    validate_entrypoint_signature_compatibility,
)
from tilefoundry_ascend.agent.specs.anchor_specs import (
    CannbenchOpsRegistry, RegimeTargetSpec, discover_module_trackb_symbols, extract_trackb_cpp_source,
    get_operator_module_relpath, get_operator_task, parse_simple_env_file,
)
from tilefoundry_ascend.ascend.json_record import atomic_write_text
from tilefoundry_ascend.ascend.passes.physical_ledger import ObjectivePhysicalLedger
from tilefoundry_ascend.ascend.sil.pytorch_frontend import PyTorchOpSemantics
from tilefoundry_ascend.ascend.sil.schema import ConstraintRelaxationMask
from tilefoundry_ascend.ascend.target.dialect_surface import CCE_CPP20_SURFACE

_LLM_ENV_PREFIX = "TILEFOUNDRY_NPU_LLM_"
_LEGACY_AGY_ENV_KEYS = frozenset({
    "TILEFOUNDRY_NPU_AGY_BIN", "TILEFOUNDRY_NPU_AGY_EXPLOIT_MODEL", "TILEFOUNDRY_NPU_AGY_EXPLORE_MODEL",
    "TILEFOUNDRY_NPU_AGY_ESCALATE_MODEL", "TILEFOUNDRY_NPU_AGY_COMMON_ARGS", "TILEFOUNDRY_NPU_AGY_MAX_CONCURRENCY",
})
_AGENT_GATES_ALLOWED_ENV_KEYS = frozenset({
    "TILEFOUNDRY_NPU_CAMPAIGN_DEVICE_IDS", "TILEFOUNDRY_NPU_CANN_SETUP", "TILEFOUNDRY_NPU_CONDA_SH",
    "TILEFOUNDRY_NPU_CONDA_ENV", "TILEFOUNDRY_NPU_CONDA_ENVS_PATH", "TILEFOUNDRY_NPU_PYTHON",
    "TILEFOUNDRY_NPU_TILELANG_SOURCE", "TILEFOUNDRY_NPU_TILELANG_BUILD", "TILEFOUNDRY_NPU_CPU_TESTS",
    "TILEFOUNDRY_NPU_DEVICE_TESTS", "TILEFOUNDRY_NPU_BUILD_SUBDIR", "TILEFOUNDRY_NPU_CANNBENCH_REPO",
    "TILEFOUNDRY_NPU_WORK_ROOT", "TILEFOUNDRY_NPU_CANNBENCH_ARCH", "TILEFOUNDRY_NPU_CANNBENCH_TARGETS_FILE",
    "TILEFOUNDRY_NPU_CANNBENCH_TARGETS", "TILEFOUNDRY_NPU_CANNBENCH_MCP_URL", "TILEFOUNDRY_NPU_CANNBENCH_MCP_TOKEN",
    "TILEFOUNDRY_NPU_CANNBENCH_MCP_TOKEN_FILE", "TILEFOUNDRY_NPU_CANNBENCH_BENCHMARK_VERSION",
    "TILEFOUNDRY_NPU_CANNBENCH_POOL_ID", "TILEFOUNDRY_NPU_CANNBENCH_AGGREGATION_TOKEN",
    "TILEFOUNDRY_NPU_CANNBENCH_INDEPENDENT_ENV_CONSENT", "TILEFOUNDRY_NPU_TILELANG_REPO",
    "TILEFOUNDRY_NPU_CANNBOT_DSL_REPO", "TILEFOUNDRY_NPU_OPS_TRANSFORMER_REPO", "TILEFOUNDRY_NPU_CANN_SERIES",
    "TILEFOUNDRY_CANN_SERIES", "TILEFOUNDRY_NPU_OCCUPANCY_POLICY", "TILEFOUNDRY_NPU_SHARED_MAX_EXT_PROCS",
    "TILEFOUNDRY_NPU_SHARED_MAX_HBM_MB", "TILEFOUNDRY_NPU_L4_EXTERNAL_PROBE_LIMIT",
})
_ADAPTER_TOOLCHAIN_ENV_KEYS = (
    "TILEFOUNDRY_NPU_CANN_SETUP", "TILEFOUNDRY_NPU_CONDA_SH", "TILEFOUNDRY_NPU_CONDA_ENV",
    "TILEFOUNDRY_NPU_CONDA_ENVS_PATH", "TILEFOUNDRY_NPU_CANNBENCH_ARCH",
)
_ADAPTER_PATH_ENV_KEYS = frozenset({"TILEFOUNDRY_NPU_CANN_SETUP", "TILEFOUNDRY_NPU_CONDA_SH"})


def short_operator_tag(operator_name: str) -> str:
    """Return short filesystem tag for an operator name or task id."""
    return (operator_name.rsplit("/", 1)[-1] if "/" in operator_name else operator_name).replace("_", "-")


def _find_cann_setup_script(seeded: Mapping[str, str]) -> str:
    if (direct := seeded.get("TILEFOUNDRY_NPU_CANN_SETUP", "").strip()) and Path(direct).is_file():
        return str(Path(direct).resolve())
    cands: list[Path] = []
    seen: set[Path] = set()
    add_cand = lambda path: cands.append(path.resolve()) if path.is_file() and path.resolve() not in seen and not seen.add(path.resolve()) else None
    roots = [Path(os.environ[k]).expanduser() for k in ("ASCEND_TOOLKIT_HOME", "ASCEND_HOME_PATH") if os.environ.get(k, "").strip()]
    for b in roots:
        add_cand(b / "set_env.sh"); add_cand(b.parent / "set_env.sh")
    vdirs = [r for b in roots for r in (b.parent, b.parent.parent) if r.is_dir()]
    if (bs := shutil.which("bisheng")):
        for par in Path(bs).resolve().parents[:5]:
            add_cand(par / "set_env.sh"); vdirs.extend(r for r in (par.parent, par.parent.parent) if r.is_dir())
    for vr in vdirs:
        add_cand(vr / "Ascend" / "9.3.0-weekly" / "cann-9.3.0" / "set_env.sh")
        for pat in ("Ascend/*/cann-*/set_env.sh", "*/cann-*/set_env.sh", "cann-*/set_env.sh"):
            for m in sorted(vr.glob(pat)):
                add_cand(m)
    opp_ok = lambda p: (p.parent / "opp" / "built-in" / "op_impl" / "ai_core" / "tbe" / "kernel" / "config").is_dir()
    return next((str(c) for c in cands if opp_ok(c)), str(cands[0]) if cands else "")


def discover_marathon_environment(
    *, operator_name: str, worktree: Path, state_dir: Path, base_env_file: str | Path | None = None, source_root: Path | None = None,
) -> dict[str, str]:
    """Discover runtime toolchain and workspace paths with no hardcoded machine paths."""
    ehint = base_env_file or os.environ.get("TILEFOUNDRY_NPU_ENV_FILE", "").strip()
    epath = p if (ehint and (p := Path(ehint).expanduser()).is_file()) else None
    seeded = parse_simple_env_file(epath) if epath else {}
    is_llm = lambda k: k.startswith(_LLM_ENV_PREFIX) or k in _LEGACY_AGY_ENV_KEYS
    seeded.update({k: v.strip() for k, v in os.environ.items() if not (epath and is_llm(k)) and k.startswith("TILEFOUNDRY_NPU_") and v.strip() and k not in seeded})
    wsp = list(dict.fromkeys([worktree.resolve().parent, *([source_root.resolve().parent] if source_root else [])]))
    if epath is None:
        for par in wsp:
            if (aenv := par / "AgentGates" / "agent_gates" / "adapters" / "tilefoundry-npu" / "env.sh").is_file():
                av = parse_simple_env_file(aenv)
                seeded.update({k: val for k in _ADAPTER_TOOLCHAIN_ENV_KEYS if k not in seeded and (val := av.get(k, "").strip()) and (k not in _ADAPTER_PATH_ENV_KEYS or Path(val).expanduser().is_file())})
                break
    py = seeded.get("TILEFOUNDRY_NPU_PYTHON") or os.environ.get("TILEFOUNDRY_NPU_PYTHON", "").strip() or sys.executable
    vals = {**seeded, "TILEFOUNDRY_NPU_PYTHON": py}
    parts = (Path(py).expanduser().resolve() if Path(py).expanduser().exists() else Path(py).expanduser()).parents
    if len(parts) >= 4 and parts[2].name == "envs":
        vals.setdefault("TILEFOUNDRY_NPU_CONDA_ENV", parts[1].name); vals.setdefault("TILEFOUNDRY_NPU_CONDA_ENVS_PATH", str(parts[2]))
        if (csh := parts[3] / "etc" / "profile.d" / "conda.sh").is_file():
            vals.setdefault("TILEFOUNDRY_NPU_CONDA_SH", str(csh))
    if (cset := _find_cann_setup_script(seeded)):
        vals["TILEFOUNDRY_NPU_CANN_SETUP"] = cset
    tl = next((p / "tilelang" for p in wsp if (p / "tilelang").is_dir()), wsp[0] / "tilelang")
    if tl.is_dir():
        vals.setdefault("TILEFOUNDRY_NPU_TILELANG_SOURCE", str(tl)); vals.setdefault("TILEFOUNDRY_NPU_TILELANG_REPO", str(tl))
    if (tl_src := Path(vals.get("TILEFOUNDRY_NPU_TILELANG_SOURCE") or tl) / "build").is_dir():
        vals.setdefault("TILEFOUNDRY_NPU_TILELANG_BUILD", str(tl_src))
    if not vals.get("TILEFOUNDRY_NPU_CANNBENCH_REPO") and (cb := next((p / "cann-bench" for p in wsp if (p / "cann-bench").is_dir() and (p / "cann-bench" / "tasks").is_dir()), None)):
        vals["TILEFOUNDRY_NPU_CANNBENCH_REPO"] = str(cb)
    for k, fn in (("TILEFOUNDRY_NPU_CANNBOT_DSL_REPO", "cannbot-dsl"), ("TILEFOUNDRY_NPU_OPS_TRANSFORMER_REPO", "ops-transformer")):
        if not vals.get(k) and (cand := next((p / fn for p in wsp if (p / fn).is_dir()), None)):
            vals[k] = str(cand)
    if not vals.get("TILEFOUNDRY_NPU_AGENT_GATES_BIN") and (ag := next((str(p / "AgentGates" / ".venv" / "bin" / "agent-gates") for p in wsp if (p / "AgentGates" / ".venv" / "bin" / "agent-gates").is_file()), shutil.which("agent-gates") or "")):
        vals["TILEFOUNDRY_NPU_AGENT_GATES_BIN"] = ag
    vals.setdefault("TILEFOUNDRY_NPU_CPU_TESTS", ""); vals.setdefault("TILEFOUNDRY_NPU_DEVICE_TESTS", ""); vals.setdefault("TILEFOUNDRY_NPU_BUILD_SUBDIR", "build")
    vals["TILEFOUNDRY_NPU_WORK_ROOT"] = str((state_dir / "cannbench-work").resolve())
    vals.setdefault("TILEFOUNDRY_NPU_CANNBENCH_ARCH", "ascend950")
    vals["TILEFOUNDRY_NPU_CANNBENCH_TARGETS"] = f"{get_operator_task(operator_name)}:{get_operator_module_relpath(operator_name)}"
    return vals


def ensure_marathon_env_file(
    *, operator_name: str, worktree: Path, state_dir: Path, env_file: str | Path | None = None, source_root: Path | None = None,
) -> Path:
    """Reuse an existing env.sh or materialize a discovered zero-config env.sh."""
    target = (Path(env_file).expanduser() if env_file is not None else (state_dir / "env.sh")).resolve()
    allowed = lambda k: k not in _NON_AGENT_GATES_ENV_KEYS and (k.startswith(_LLM_ENV_PREFIX) or k in _LEGACY_AGY_ENV_KEYS or k in _AGENT_GATES_ALLOWED_ENV_KEYS or k.startswith("TILEFOUNDRY_NPU_CANNBENCH_"))
    disc = {k: v for k, v in discover_marathon_environment(operator_name=operator_name, worktree=worktree, state_dir=state_dir, base_env_file=target if target.is_file() else None, source_root=source_root).items() if allowed(k)}
    _validate_llm_api_key_env(disc.get(_LLM_API_KEY_ENV, "").strip())
    if target.is_file():
        ex = parse_simple_env_file(target)
        if not any(not allowed(k) for k in ex) and not any(v and not ex.get(k) for k, v in disc.items()):
            return target
        merged = {**disc, **{k: v for k, v in ex.items() if v and allowed(k)}}
    else:
        merged = disc
    lines = [f"# Isolated marathon environment for {operator_name}", "# Plain KEY=VALUE assignments.", "", *(_format_env_line(k, merged[k]) for k in sorted(merged)), ""]
    atomic_write_text(target, "\n".join(lines))
    return target


def configure_marathon_process_env(*, operator_name: str, env_file: Path, worktree: Path | None = None, source_root: Path | None = None) -> dict[str, str]:
    """Configure coordinator process env vars cleanly for isolated worker subprocesses."""
    _ = (worktree, source_root)
    tdir = Path(tempfile.gettempdir()) / f"tf_marathon_{short_operator_tag(operator_name)}"
    tdir.mkdir(parents=True, exist_ok=True)
    os.environ.update({"TMPDIR": str(tdir), "TEMP": str(tdir), "TMP": str(tdir), "PYTHONUNBUFFERED": "1", "PYTHONDONTWRITEBYTECODE": "1", "TORCH_DEVICE_BACKEND_AUTOLOAD": "0", "USE_CUDA": "0", "TILEFOUNDRY_NPU_ENV_FILE": str(env_file)})
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    os.environ.pop("ASCEND_RT_VISIBLE_DEVICES", None)
    parsed = parse_simple_env_file(env_file)
    for k in ("TILEFOUNDRY_NPU_AGENT_GATES_BIN", "TILEFOUNDRY_NPU_CANNBENCH_REPO", "TILEFOUNDRY_NPU_TILELANG_SOURCE", "TILEFOUNDRY_NPU_TILELANG_BUILD"):
        if (val := parsed.get(k, "").strip()) and not os.environ.get(k):
            os.environ[k] = val
    if not os.environ.get("TILEFOUNDRY_NPU_AGENT_GATES_BIN", "").strip() and (ag := shutil.which("agent-gates")):
        os.environ["TILEFOUNDRY_NPU_AGENT_GATES_BIN"] = ag
    return parsed


def _stage0_contract_err(mod_src: str, entrypoint: str, track: str, *, contract: GoldenEntrypointContract | None = None) -> str:
    if track not in ("cce_trackb", "hybrid_trackb", "tilelang_dsl"):
        return f"unknown Stage-0 track {track!r}"
    try:
        tree = ast.parse(mod_src)
    except SyntaxError as exc:
        return f"seed module syntax error: {exc}"
    if not any(isinstance(n, ast.FunctionDef) and n.name == entrypoint for n in tree.body):
        return f"seed module does not define the public entrypoint {entrypoint}"
    if not any((isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "kernel" for t in n.targets) or isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id == "kernel") and isinstance(n.value, ast.Name) and n.value.id == entrypoint for n in tree.body):
        return f"seed module does not bind `kernel = {entrypoint}`"
    if contract is not None and (serr := validate_entrypoint_signature_compatibility(mod_src, contract)):
        return serr
    if (perr := check_npu_entrypoint_purity(mod_src, entrypoint, track=track, require_kernel_cache=True)):
        return "entrypoint purity rejection: " + "; ".join(perr)
    if track in ("cce_trackb", "hybrid_trackb") and (off := {t: c for t, c in CCE_CPP20_SURFACE.count_forbidden_tokens(mod_src).items() if c}):
        return "CCE C++20 seed references forbidden tokens: " + ", ".join(f"{t}={c}" for t, c in off.items())
    return ""


def _stage0_agg_engine_err(ledgers: Sequence[ObjectivePhysicalLedger | None], semantics: PyTorchOpSemantics | None) -> str:
    active = [lg for lg in ledgers if lg is not None]
    if semantics and semantics.has_gemm_contraction and not any(any(k in ("AIC.PIPE_M", "AIC.CUBE", "CUBE_L0C") and int(v) > 0 for k, v in lg.pipe_cycles.items()) for lg in active):
        return "aggregate L0 hardware-engine rejection: missing_cube_engine_primitives (AIC.PIPE_M == 0)"
    if semantics and not semantics.has_indirect_index and not any(lg.has_indirect_gather or any(k.endswith("PIPE_MTE2") and int(v) > 0 for k, v in lg.pipe_cycles.items()) for lg in active):
        return "aggregate L0 hardware-engine rejection: missing_hardware_dma_staging (PIPE_MTE2 == 0)"
    return ""


def build_stage0_static_validator(
    regime_specs: Sequence[RegimeTargetSpec], *, entrypoint: str, worktree: Path, env_file: Path | None,
    entrypoint_contract: GoldenEntrypointContract | None = None, semantics: PyTorchOpSemantics | None = None,
    op_proto: Mapping[str, object] | None = None,
    run_stage_a_fn: Callable[..., object] | None = None,
    run_l1_fn: Callable[..., object] | None = None,
) -> Callable[[str, str], tuple[bool, str]]:
    """Build the L0/L1 gate pair the bounded Stage-0 seed loop invokes per attempt."""
    assert regime_specs, "Stage-0 seed synthesis requires at least one regime specification"
    eff_sa_fn = run_stage_a_fn or run_stage_a_static
    eff_l1_fn = run_l1_fn or run_l1_compile_check

    def _run_ta(ms: str, rs: RegimeTargetSpec, *, enforce: bool) -> tuple[bool, str, ObjectivePhysicalLedger | None]:
        if (ce := _stage0_contract_err(ms, entrypoint, "tilelang_dsl", contract=entrypoint_contract)):
            return False, ce, None
        dims = dict(rs.representative_anchor_dims or rs.representative_hir_dims)
        try:
            res = resolve_regime_config(ms, rs, dims=dims)
        except (SyntaxError, ImportError, KeyError, NameError, RuntimeError, TypeError, ValueError) as exc:
            return False, f"seed config resolution failed: {type(exc).__name__}: {exc}", None
        sa = eff_sa_fn(
            ms, factory_name=res.builder_name, factory_kwargs=res.builder_kwargs, pass_configs=res.pass_configs or None,
            compile_flags=res.compile_flags or None, dims=dims or None, case_ratios=rs.representative_case_ratios,
            require_hardware_dma=bool(enforce and semantics and not semantics.has_indirect_index),
            require_cube_engine=bool(enforce and semantics and semantics.has_gemm_contraction),
            ordered_tensor_shapes=extract_ordered_tensor_shapes(case_entry=rs.representative_case_entry, dims=dims, proto=op_proto),
        )
        if not sa.passed:
            return False, f"L0 static/waterline rejection: {sa.error_message}", sa.ledger
        sb = eff_l1_fn(ms, builder_name=res.builder_name, builder_kwargs=res.builder_kwargs, config_selector_name=res.selector_name or None, pass_configs=res.pass_configs or None, compile_flags=res.compile_flags or None, source_root=worktree, env_file=env_file)
        return (True, "", sa.ledger) if sb.passed else (False, f"L1 device compilation rejection: {sb.error_message}", sa.ledger)

    def _run_tb(ms: str, rs: RegimeTargetSpec, *, track: str, enforce: bool) -> tuple[bool, str, ObjectivePhysicalLedger | None]:
        if (ce := _stage0_contract_err(ms, entrypoint, track, contract=entrypoint_contract)):
            return False, ce, None
        tb_syms = rs.trackb_symbols or discover_module_trackb_symbols(ms)
        if (ext := extract_trackb_cpp_source(ms, tb_syms)) is None:
            return False, "seed exposes no Track-B C++20 translation-unit string constant", None
        try:
            tu = _render_trackb_tu_from_module(ms, ext[1], rs, require_renderer=True)
        except FileNotFoundError:
            raise
        except (SyntaxError, AttributeError, ImportError, KeyError, NameError, OSError, RuntimeError, TypeError, ValueError) as exc:
            return False, f"Track-B _render_trackb_tu instantiation failed ({ext[0]}): {exc}", None
        ver = evaluate_ascendc_candidate(tu, freedom_mask=ConstraintRelaxationMask.top(), case_ratios=rs.representative_case_ratios, require_hardware_dma=bool(enforce and semantics and not semantics.has_indirect_index), require_cube_engine=bool(enforce and semantics and semantics.has_gemm_contraction))
        if not ver.passed:
            return False, f"Track-B L0 token purity ({ext[0]}): {ver.error_message}", ver.ledger
        comp = compile_trackb_stage_b(tu, _resolve_selector_compile_flags(ms, rs), env_file=env_file, source_root=worktree, verify_shared_link=True)
        return (True, "", ver.ledger) if comp.passed else (False, f"Track-B L1 bisheng compilation ({ext[0]}): {comp.error_message}", ver.ledger)

    def _validate(ms: str, track: str) -> tuple[bool, str]:
        if track not in ("cce_trackb", "hybrid_trackb", "tilelang_dsl"):
            return False, f"[{regime_specs[0].regime_tag}] {_stage0_contract_err(ms, entrypoint, track, contract=entrypoint_contract)}"
        fails = []
        for rs in regime_specs:
            if track == "cce_trackb":
                ok, err, _ = _run_tb(ms, rs, track=track, enforce=True)
            elif track == "hybrid_trackb":
                hp = _has_prim_func_builder(ms, rs.builder_candidates)
                ok, err, ltb = _run_tb(ms, rs, track=track, enforce=not hp)
                if ok and hp:
                    ok, err, lta = _run_ta(ms, rs, enforce=False)
                    if ok and (ae := _stage0_agg_engine_err((ltb, lta), semantics)):
                        ok, err = False, ae
            else:
                ok, err, _ = _run_ta(ms, rs, enforce=True)
            if not ok:
                fails.append(f"[{rs.regime_tag}] {err}")
        return (False, "; ".join(fails)) if fails else (True, "")

    return _validate


def build_stage0_oracle_validator(
    *, operator_name: str, staging_dir: Path, worktree: Path, device_id: str, op_logger: logging.Logger,
    blocked_env_max_retries: int = 12, blocked_env_backoff_sec: tuple[float, ...] = (10.0, 15.0, 20.0, 25.0, 30.0),
) -> Callable[[str], OracleAccuracyResult]:
    """Build the NPU accuracy validator that stages seed bytes by digest before L3 verification."""
    def _validate(module_source: str) -> OracleAccuracyResult:
        digest = hashlib.sha256(module_source.encode("utf-8")).hexdigest()
        staged = staging_dir / f"{digest}.py"
        atomic_write_text(staged, module_source)
        res, transient = None, False
        for ridx in range(max(0, int(blocked_env_max_retries)) + 1):
            res = run_l3_full_contract_via_agent_gates(operator_name=operator_name, candidate_module_path=staged, source_root=worktree, device_id=str(device_id))
            if res.passed:
                transient = False; break
            el = (res.error_message or "").lower()
            if res.failure_kind != "candidate" and ("cannot locate 'agent-gates'" in el or "env.sh file is required" in el or "filenotfounderror" in el):
                raise FileNotFoundError(res.error_message)
            transient = res.failure_kind != "candidate" and ("blocked_env" in el or "resource busy" in el)
            if not transient or ridx >= max(0, int(blocked_env_max_retries)):
                break
            wsec = float(blocked_env_backoff_sec[min(ridx, len(blocked_env_backoff_sec) - 1)]) if blocked_env_backoff_sec else 10.0
            op_logger.info("STAGE0_ORACLE_RETRY_BLOCKED_ENV op=%s retry=%d wait=%.1fs", operator_name, ridx + 1, wsec)
            time.sleep(wsec)
        assert res is not None
        fkind = "environment" if (not res.passed and transient) else (res.failure_kind or "candidate")
        return OracleAccuracyResult(passed=res.passed, error_message=res.error_message, candidate_sha256=res.candidate_sha256, l3_run_id=res.run_id, l3_receipt_path=res.receipt_path, l3_state=res.state, elapsed_sec=res.elapsed_sec, failure_kind=fkind)

    return _validate


def prepare_marathon_seed(
    *, operator_name: str, target_module: Path, staging_dir: Path, worktree: Path, env_file: Path | None,
    cannbench_repo: Path | None, llm_backend: KernelLLMBackend, cold_start: bool, device_id: str,
    op_logger: logging.Logger, pure_golden: bool = False,
    static_validator_factory: Callable[..., Callable[[str, str], tuple[bool, str]]] = build_stage0_static_validator,
    oracle_validator_factory: Callable[..., Callable[[str], OracleAccuracyResult]] = build_stage0_oracle_validator,
) -> SeedSynthesisOutcome:
    """Run bounded Stage-0 seed synthesis or explicit incumbent takeover before the campaign."""
    eff_cold = bool(cold_start or pure_golden)
    reg = CannbenchOpsRegistry(tasks_root=cannbench_repo, source_root=worktree, pure_golden=pure_golden)
    spec = reg.resolve(operator_name)
    if not spec.golden_path.is_file():
        raise FileNotFoundError(f"operator {operator_name} has no PyTorch golden reference at {spec.golden_path}")
    gsrc = spec.golden_path.read_text(encoding="utf-8")
    dtxt = spec.desc_path.read_text(encoding="utf-8") if spec.desc_path.is_file() else ""
    gcon = extract_golden_entrypoint_contract(gsrc, spec.entrypoint, proto=spec.proto, desc_text=dtxt)
    takeover = not eff_cold and target_module.is_file()
    init_path = target_module.parent / "__init__.py"
    created_init = not takeover and not init_path.exists()
    if created_init:
        target_module.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(init_path, f'"""Cold-start Ascend 950PR operator package for {spec.op_id}."""\n\nfrom .kernels import {spec.entrypoint}, kernel\n\n__all__ = [{spec.entrypoint!r}, "kernel"]\n')
    try:
        outcome = asyncio.run(prepare_operator_seed_or_takeover(
            operator_name=operator_name, entrypoint=spec.entrypoint, pytorch_golden_source=gsrc, proto=spec.proto,
            cases=spec.cases, legacy_regime_specs=(reg.get_regime_specs(operator_name) if takeover else None),
            llm_backend=llm_backend,
            l0_l1_validator_factory=lambda rs, *, semantics=None: static_validator_factory(rs, entrypoint=spec.entrypoint, worktree=worktree, env_file=env_file, entrypoint_contract=gcon, semantics=semantics, op_proto=spec.proto),
            oracle_accuracy_validator=oracle_validator_factory(operator_name=operator_name, staging_dir=staging_dir, worktree=worktree, device_id=device_id, op_logger=op_logger),
            existing_module_path=None if eff_cold else target_module, force_cold_start=eff_cold,
            max_seed_attempts=8, desc_text=dtxt, golden_contract=gcon,
        ))
    except (OSError, RuntimeError, ValueError):
        if created_init:
            init_path.unlink(missing_ok=True)
        raise
    if outcome.mode == "cold_start":
        digest = hashlib.sha256(outcome.module_source.encode("utf-8")).hexdigest()
        assert outcome.verification_state == "l3_verified" and outcome.candidate_sha256 == digest
        atomic_write_text(target_module, outcome.module_source)
    return outcome
