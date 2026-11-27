"""Thin single- and multi-operator marathon runner delegating to CampaignRunner."""

from __future__ import annotations

import argparse, asyncio, hashlib, json, logging, os, subprocess, sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, fields, replace
from pathlib import Path

from tilefoundry_ascend.agent.campaign.campaign_runner import (
    CAMPAIGN_DEVICE_IDS_ENV_KEY, CampaignConfig, ExtremeCampaignRunner,
    _add_common_campaign_arguments, _llm_backend_overrides, build_resilient_l2_evaluator,
    build_stage0_oracle_validator, build_stage0_static_validator, configure_marathon_process_env,
    discover_marathon_environment, ensure_marathon_env_file, parse_device_ids, prepare_marathon_seed,
    resolve_campaign_device_plan, short_operator_tag,
)
from tilefoundry_ascend.agent.flywheel.ablation_distiller import FlywheelDistiller
from tilefoundry_ascend.agent.gates.gate_funnel import GateFunnel
from tilefoundry_ascend.agent.gates.stage_b_gate import _ensure_tilelang_on_sys_path
from tilefoundry_ascend.agent.kb.bayesian_tracker import BayesianRuleTracker
from tilefoundry_ascend.agent.npu import (
    DEFAULT_L4_CANNBENCH_WAIT_SEC, L3AcceptanceResult,
    collect_bottleneck_msprof_via_agent_gates, run_l3_full_contract_via_agent_gates,
    run_l4_official_cannbench_via_agent_gates, run_micro_probe_via_agent_gates,
)
from tilefoundry_ascend.agent.orchestrator.llm_client import AgyCliLLMBackend, create_llm_backend_from_env
from tilefoundry_ascend.agent.orchestrator.seed_synthesizer import SeedSynthesisOutcome
from tilefoundry_ascend.agent.probes import LocalHardwareProbePort
from tilefoundry_ascend.agent.specs.anchor_specs import (
    CannbenchOpsRegistry, PerfTargetSpec, RegimeTargetSpec, default_source_root,
    get_operator_module_relpath,
)
from tilefoundry_ascend.ascend.cost_model.experience_lake import resolve_default_experience_lake_dir
from tilefoundry_ascend.ascend.json_record import atomic_write_text

logger = logging.getLogger("tilefoundry_ascend.marathon")
FastCreativeAgyBackend = AgyCliLLMBackend


@dataclass(frozen=True)
class MarathonConfig:
    """Configuration for zero-config single- or multi-operator marathon campaigns."""

    operator_name: str
    device_id: str = ""
    retry_device_id: str = ""
    device_ids: tuple[str, ...] = ()
    source_root: Path | None = None
    worktree: Path | None = None
    isolate_worktree: bool = False
    state_dir: Path | None = None
    target_module_path: Path | None = None
    env_file: Path | None = None
    checkpoint_path: Path | None = None
    report_path: Path | None = None
    campaign_kb_path: Path | None = None
    experience_lake_dir: Path | None = None
    log_path: Path | None = None
    max_rounds: int = 80
    round_regime_tags: tuple[str, ...] = ()
    max_wall_clock_min: float = 720.0
    k_rule_workers: int = 2
    max_syntax_repairs: int = 1
    min_surrogate_gain: float = 0.08
    stagnation_escalate_rounds: int = 1
    stagnation_freeze_rounds: int = 12
    stagnation_discount: float = 0.45
    target_avg_speedup: float = 2.985
    target_min_case_speedup: float = 2.95
    target_roofline_eta: float = 0.995
    target_surrogate_score: float = 99.8
    min_optimization_rounds: int = 15
    disable_early_goal_stop: bool = False
    run_final_l4: bool = True
    l4_regression_ratio: float = 1.0
    warmup: int = 3
    repeat: int = 5
    enable_tilesize_prescreen: bool = True
    cost_prescreen_top_k: int | None = None
    cost_model_mode: str = "off"
    allow_unreviewed_distilled_rules: bool = False
    salvage_beam_candidates: bool = False
    resume: bool | None = None
    escalate_model: str | None = None
    exploit_effort: str = "medium"
    auto_bootstrap_seed: bool = True
    cold_start: bool = False
    pure_golden: bool = False
    llm_backend: str | None = None
    exploit_model: str | None = None
    explore_model: str | None = None
    llm_api_base: str | None = None


def _resolve_worktree(cfg: MarathonConfig) -> tuple[Path, Path]:
    src = Path(cfg.source_root).expanduser().resolve() if cfg.source_root else default_source_root()
    if cfg.worktree is not None:
        return src, Path(cfg.worktree).expanduser().resolve()
    if not cfg.isolate_worktree:
        return src, src
    cand = src.parent / f"tilefoundry-marathon-{short_operator_tag(cfg.operator_name)}"
    if not (cand.is_dir() and (cand / ".git").exists()):
        subprocess.run(["git", "-C", str(src), "worktree", "add", "--detach", str(cand), "HEAD"], check=True, text=True, capture_output=True)
    return src, cand.resolve()


def run_operator_marathon(
    config: MarathonConfig | None = None,
    *,
    operator_name: str | None = None,
    max_rounds: int | None = None,
    resume: bool | None = None,
) -> int:
    """Configure workspace, environment, and run ExtremeCampaignRunner."""
    if config is None:
        if operator_name is None:
            raise ValueError("operator_name is required when config is None")
        cfg = MarathonConfig(
            operator_name=operator_name,
            max_rounds=80 if max_rounds is None else max_rounds,
            resume=resume,
        )
    else:
        cfg = replace(
            config,
            operator_name=config.operator_name if operator_name is None else operator_name,
            max_rounds=config.max_rounds if max_rounds is None else max_rounds,
            resume=config.resume if resume is None else resume,
        )
    src_root, worktree = _resolve_worktree(cfg)
    for e in (str(worktree / "targets" / "ascend"), str(worktree / "src"), str(worktree)):
        if e not in sys.path:
            sys.path.insert(0, e)
    short_op = short_operator_tag(cfg.operator_name)
    state_dir = Path(cfg.state_dir).expanduser().resolve() if cfg.state_dir else (worktree / ".tilefoundry-marathon" / cfg.operator_name).resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    op_tag = cfg.operator_name.rsplit("/", 1)[-1]
    skey = hashlib.sha256(str(state_dir).encode("utf-8")).hexdigest()[:12]
    staging_dir = (state_dir / "stage0_candidates").resolve() if state_dir.is_relative_to(worktree) else (worktree / ".tilefoundry-marathon" / "_stage0_staging" / skey / short_op).resolve()
    mirror: Path | None = None
    if cfg.target_module_path is not None:
        target_mod = Path(cfg.target_module_path).expanduser().resolve()
    elif cfg.pure_golden and state_dir.is_relative_to(worktree):
        target_mod = (state_dir / "pure_golden_ops" / op_tag / "kernels.py").resolve()
    elif cfg.pure_golden:
        target_mod = (worktree / ".tilefoundry-marathon" / "pure_golden_ops" / skey / op_tag / "kernels.py").resolve()
        mirror = (state_dir / "pure_golden_ops" / op_tag / "kernels.py").resolve()
    else:
        target_mod = (worktree / get_operator_module_relpath(cfg.operator_name)).resolve()
    ckpt = Path(cfg.checkpoint_path).expanduser().resolve() if cfg.checkpoint_path else (state_dir / "checkpoint.json")
    rpath = Path(cfg.report_path).expanduser().resolve() if cfg.report_path else (state_dir / "report.json")
    kbp = Path(cfg.campaign_kb_path).expanduser().resolve() if cfg.campaign_kb_path else (state_dir / "campaign_kb.yaml")
    lpath = Path(cfg.log_path).expanduser().resolve() if cfg.log_path else (state_dir / "marathon.log")
    lpath.parent.mkdir(parents=True, exist_ok=True)
    root_log = logging.getLogger("tilefoundry_ascend")
    root_log.setLevel(logging.INFO)
    if not any(isinstance(h, logging.FileHandler) and Path(getattr(h, "baseFilename", "")) == lpath for h in root_log.handlers):
        fh = logging.FileHandler(lpath, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root_log.addHandler(fh)
    op_log = logging.getLogger(f"tilefoundry_ascend.marathon.{short_op}")
    env_file = ensure_marathon_env_file(operator_name=cfg.operator_name, worktree=worktree, state_dir=state_dir, env_file=cfg.env_file, source_root=src_root)
    vis = parse_device_ids(os.environ.get("ASCEND_RT_VISIBLE_DEVICES", ""))
    env_vals = configure_marathon_process_env(operator_name=cfg.operator_name, env_file=env_file, worktree=worktree, source_root=src_root)
    _ensure_tilelang_on_sys_path(env_vals=env_vals, env_file=env_file, source_root=src_root)
    pdev, dpool = resolve_campaign_device_plan(device_id=str(cfg.device_id), device_ids=tuple(cfg.device_ids), env_device_ids=parse_device_ids(env_vals.get(CAMPAIGN_DEVICE_IDS_ENV_KEY, "")), visible_device_ids=vis)
    llm = create_llm_backend_from_env(overrides=_llm_backend_overrides(env_vals, cfg))
    rcb = env_vals.get("TILEFOUNDRY_NPU_CANNBENCH_REPO", "").strip() or os.environ.get("TILEFOUNDRY_NPU_CANNBENCH_REPO", "").strip()
    crepo = Path(rcb) if rcb else None
    resume_flag = ckpt.is_file() if cfg.resume is None else bool(cfg.resume)
    s_specs: Sequence[RegimeTargetSpec] | None = None
    seed_out: SeedSynthesisOutcome | None = None
    if cfg.auto_bootstrap_seed and not (resume_flag and ckpt.is_file() and target_mod.is_file()):
        try:
            seed_out = prepare_marathon_seed(
                operator_name=cfg.operator_name, target_module=target_mod, staging_dir=staging_dir,
                worktree=worktree, env_file=env_file, cannbench_repo=crepo, llm_backend=llm,
                cold_start=cfg.cold_start, device_id=pdev, op_logger=op_log, pure_golden=cfg.pure_golden,
                static_validator_factory=build_stage0_static_validator, oracle_validator_factory=build_stage0_oracle_validator,
            )
            s_specs = seed_out.regime_specs
        except FileNotFoundError:
            raise
        except (OSError, RuntimeError, ValueError) as exc:
            op_log.error("STAGE0_SEED_FAILED op=%s error=%s", cfg.operator_name, exc)
            return 2
    l2_ev = build_resilient_l2_evaluator(operator_name=cfg.operator_name, worktree=worktree, env_file=env_file, device_id=pdev, device_pool=dpool, warmup=cfg.warmup, repeat=cfg.repeat, op_logger=op_log)

    def _l3_run(mp: Path, *, resume_execution_id: str = "", resume_run_id: str = "", device_id: str = "", on_execution_started: Callable[[str, str], None] | None = None) -> L3AcceptanceResult:
        lk = {"device_pool": dpool} if (dpool and not resume_execution_id and not resume_run_id) else {}
        return run_l3_full_contract_via_agent_gates(operator_name=cfg.operator_name, candidate_module_path=mp, source_root=worktree, device_id=str(device_id or pdev), resume_execution_id=resume_execution_id, resume_run_id=resume_run_id, on_execution_started=on_execution_started, **lk)

    ldir = resolve_default_experience_lake_dir(worktree=worktree, checkpoint_path=ckpt, explicit_dir=cfg.experience_lake_dir)
    ptgt = PerfTargetSpec(operator_name=cfg.operator_name, target_avg_speedup=cfg.target_avg_speedup, target_min_case_speedup=cfg.target_min_case_speedup, target_roofline_eta=cfg.target_roofline_eta, target_surrogate_score=cfg.target_surrogate_score, min_optimization_rounds=cfg.min_optimization_rounds, per_anchor_max_us={})
    cf_names = {f.name for f in fields(CampaignConfig)}
    ccfg = CampaignConfig(
        operator_name=cfg.operator_name, max_rounds=cfg.max_rounds, max_wall_clock_min=cfg.max_wall_clock_min,
        k_rule_workers=cfg.k_rule_workers, max_syntax_repairs=cfg.max_syntax_repairs, min_surrogate_gain=cfg.min_surrogate_gain,
        stagnation_escalate_rounds=cfg.stagnation_escalate_rounds, stagnation_freeze_rounds=cfg.stagnation_freeze_rounds,
        disable_early_goal_stop=cfg.disable_early_goal_stop, run_final_l4=cfg.run_final_l4, l4_regression_ratio=cfg.l4_regression_ratio,
        device_id=pdev, retry_device_id=str(cfg.retry_device_id), device_ids=tuple(d for d in dpool if d != pdev),
        warmup=cfg.warmup, repeat=cfg.repeat, enable_tilesize_prescreen=cfg.enable_tilesize_prescreen,
        cost_prescreen_top_k=cfg.cost_prescreen_top_k, exploit_effort=cfg.exploit_effort, round_regime_tags=tuple(cfg.round_regime_tags),
        pure_golden=cfg.pure_golden, **{k: getattr(cfg, k) for k in ("cost_model_mode", "allow_unreviewed_distilled_rules") if k in cf_names},
    )
    tracker = BayesianRuleTracker.from_experience_lake(ldir, allow_unreviewed_distilled_rules=cfg.allow_unreviewed_distilled_rules)
    fly = FlywheelDistiller(funnel=GateFunnel(skip_stage_b=True), tracker=tracker, probe_port=LocalHardwareProbePort(), probe_runner=lambda sp: run_micro_probe_via_agent_gates(sp, source_root=worktree, env_file=env_file, device_id=pdev, min_speedup=1.08, timeout_sec=360, **({"device_pool": dpool} if dpool else {})))
    p_anc = None
    if cfg.pure_golden:
        preg = CannbenchOpsRegistry(tasks_root=crepo, source_root=worktree, pure_golden=True)
        p_anc = preg.get_anchors(cfg.operator_name)
        s_specs = s_specs or preg.get_regime_specs(cfg.operator_name)
    runner = ExtremeCampaignRunner(
        config=ccfg, target_module_path=target_mod, anchors=p_anc, regime_specs=s_specs,
        perf_target=ptgt, llm_backend=llm, tracker=tracker, flywheel_distiller=fly,
        l2_batch_evaluator=l2_ev,
        msprof_collector=lambda op, cp, ancs: collect_bottleneck_msprof_via_agent_gates(op, cp, ancs, source_root=worktree, env_file=env_file, device_id=pdev, timeout_sec=600, **({"device_pool": dpool} if dpool else {})),
        l3_contract_runner=_l3_run,
        l4_cannbench_runner=lambda mp, *, device_id="", resume_execution_id="", resume_run_id="", on_execution_started=None: run_l4_official_cannbench_via_agent_gates(operator_name=cfg.operator_name, candidate_module_path=mp, source_root=worktree, device_id=str(device_id or pdev), timeout_sec=DEFAULT_L4_CANNBENCH_WAIT_SEC, resume_execution_id=str(resume_execution_id or ""), resume_run_id=str(resume_run_id or ""), on_execution_started=on_execution_started),
        checkpoint_path=ckpt, campaign_kb_path=kbp, cost_calibration_path=state_dir / "cost_model_calibration.jsonl",
        experience_lake_dir=ldir, source_root=worktree, env_file=env_file, cannbench_repo=crepo,
        regime_blueprints=seed_out.blueprints_by_regime if seed_out else None,
        topology_axes=seed_out.semantics.base_axes if seed_out else None,
        stagnation_discount=cfg.stagnation_discount, op_logger=op_log,
    )
    report = asyncio.run(runner.run(resume=resume_flag))
    if mirror is not None and target_mod.is_file():
        mirror.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(mirror, target_mod.read_text(encoding="utf-8"))
        if (sinit := target_mod.parent / "__init__.py").is_file():
            atomic_write_text(mirror.parent / "__init__.py", sinit.read_text(encoding="utf-8"))
    rendered = json.dumps(report.to_dict(), indent=2)
    atomic_write_text(rpath, rendered + "\n")
    print(rendered)
    return 0


def run_operators_marathon(operators: Sequence[str], *, base_config: MarathonConfig, mode: str = "sequential", round_slice: int = 2) -> int:
    """Run sequential or round-robin marathon campaigns across *operators*."""
    ops = [op.strip() for op in operators if op.strip()]
    if not ops:
        raise ValueError("operators list must not be empty")
    if mode != "round-robin" or len(ops) == 1:
        for op in ops:
            if (rc := run_operator_marathon(base_config, operator_name=op)) != 0:
                return rc
        return 0
    slice_r = max(1, int(round_slice))
    done_rounds = {op: 0 for op in ops}
    while any(done_rounds[op] < base_config.max_rounds for op in ops):
        for op in ops:
            if done_rounds[op] >= base_config.max_rounds:
                continue
            target_r = min(base_config.max_rounds, done_rounds[op] + slice_r)
            if (rc := run_operator_marathon(base_config, operator_name=op, max_rounds=target_r, resume=(done_rounds[op] > 0 or base_config.resume))) != 0:
                return rc
            done_rounds[op] = target_r
    return 0


def build_marathon_arg_parser() -> argparse.ArgumentParser:
    """Construct standalone CLI parser for zero-config single- or multi-op marathon runs."""
    parser = argparse.ArgumentParser(prog="tilefoundry-ascend-marathon", description="Zero-config extreme score-sprint marathon runner.")
    sub = parser.add_subparsers(dest="command", required=True)
    rp = sub.add_parser("run", help="Run or auto-resume a score-sprint campaign.")
    _add_common_campaign_arguments(rp, required_op=False, default_rounds=80, default_wall=720.0)
    rp.add_argument("--operators", default="", help="Comma-separated operator list for multi-op marathon.")
    rp.add_argument("--marathon", action="store_true", help="Compatibility flag for unified marathon invocation.")
    rp.add_argument("--marathon-mode", choices=("sequential", "round-robin"), default="sequential")
    rp.add_argument("--worktree", default=None)
    rp.add_argument("--isolate-worktree", action="store_true")
    rp.add_argument("--state-dir", default=None)
    rp.add_argument("--target-file", "--target-module", dest="target_file", default=None)
    rp.add_argument("--report-json", default=None)
    rp.add_argument("--log-file", default=None)
    rp.add_argument("--no-final-l4", action="store_true")
    rp.add_argument("--fresh", action="store_true")
    rp.add_argument("--no-salvage", action="store_true")
    rp.add_argument("--cold-start", action="store_true")
    rp.add_argument("--pure-golden", dest="pure_golden", action="store_true", default=None)
    rp.add_argument("--no-pure-golden", dest="pure_golden", action="store_false")
    rp.add_argument("--experience-lake-dir", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute marathon CLI subcommand (defaults to 'run' when flags are passed directly)."""
    raw = list(sys.argv[1:] if argv is None else argv)
    if raw and raw[0] not in ("run", "-h", "--help"):
        raw = ["run", *raw]
    args = build_marathon_arg_parser().parse_args(raw)
    ops = [x.strip() for x in str(args.operators or "").split(",") if x.strip()]
    if not ops and not args.operator:
        raise SystemExit("Either --operator or --operators must be specified")
    primary_op = str(args.operator or ops[0])
    pure_golden = bool(args.cold_start) if args.pure_golden is None else bool(args.pure_golden)
    cfg = MarathonConfig(
        operator_name=primary_op, device_id=str(args.device_id), retry_device_id=str(args.retry_device_id),
        device_ids=tuple(args.device_ids), source_root=Path(args.source_root) if args.source_root else None,
        worktree=Path(args.worktree) if args.worktree else None, isolate_worktree=bool(args.isolate_worktree),
        state_dir=Path(args.state_dir) if args.state_dir else None,
        target_module_path=Path(args.target_file) if args.target_file else None,
        env_file=Path(args.env_file) if args.env_file else None,
        checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
        report_path=Path(args.report_json) if args.report_json else None,
        campaign_kb_path=Path(args.campaign_kb) if args.campaign_kb else None,
        experience_lake_dir=Path(args.experience_lake_dir) if args.experience_lake_dir else None,
        log_path=Path(args.log_file) if args.log_file else None,
        max_rounds=int(args.max_rounds), round_regime_tags=tuple(args.round_regime or ()),
        max_wall_clock_min=float(args.max_wall_clock_min), k_rule_workers=int(args.k_rule_workers),
        disable_early_goal_stop=bool(args.disable_early_goal_stop), run_final_l4=not bool(args.no_final_l4),
        resume=False if args.fresh else None, salvage_beam_candidates=not bool(args.no_salvage),
        cold_start=bool(args.cold_start), pure_golden=pure_golden,
        enable_tilesize_prescreen=bool(args.enable_tilesize_prescreen),
        cost_prescreen_top_k=int(args.cost_prescreen_top_k) if args.cost_prescreen_top_k is not None else None,
        cost_model_mode=str(args.cost_model_mode), allow_unreviewed_distilled_rules=bool(args.allow_unreviewed_distilled_rules),
        llm_backend=args.llm_backend, exploit_model=args.exploit_model, explore_model=args.explore_model,
        escalate_model=args.escalate_model, llm_api_base=args.llm_api_base,
    )
    if len(ops) > 1:
        return run_operators_marathon(ops, base_config=cfg, mode=str(args.marathon_mode))
    return run_operator_marathon(cfg)


__all__ = [
    "ExtremeCampaignRunner", "FastCreativeAgyBackend", "MarathonConfig", "atomic_write_text",
    "build_marathon_arg_parser", "build_resilient_l2_evaluator", "build_stage0_oracle_validator",
    "build_stage0_static_validator", "configure_marathon_process_env", "discover_marathon_environment",
    "ensure_marathon_env_file", "main", "prepare_marathon_seed", "run_operator_marathon",
    "run_operators_marathon", "short_operator_tag",
]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
