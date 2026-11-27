"""Command-line entrypoint for the unattended Ascend optimization campaign."""

from __future__ import annotations

import argparse, asyncio, json, os, sys
from collections.abc import Sequence
from pathlib import Path

from tilefoundry_ascend.agent.campaign.campaign_runner import (
    CAMPAIGN_DEVICE_IDS_ENV_KEY, CampaignConfig, PerfTargetCampaignRunner,
    _add_common_campaign_arguments, _llm_backend_overrides, parse_device_ids,
    resolve_campaign_device_plan,
)
from tilefoundry_ascend.agent.campaign.marathon import MarathonConfig, run_operator_marathon, run_operators_marathon
from tilefoundry_ascend.agent.flywheel.ablation_distiller import FlywheelDistiller
from tilefoundry_ascend.agent.gates.gate_funnel import GateFunnel
from tilefoundry_ascend.agent.gates.stage_b_gate import _ensure_tilelang_on_sys_path
from tilefoundry_ascend.agent.kb.bayesian_tracker import BayesianRuleTracker
from tilefoundry_ascend.agent.npu import run_micro_probe_via_agent_gates
from tilefoundry_ascend.agent.orchestrator.llm_client import create_llm_backend_from_env
from tilefoundry_ascend.agent.probes import LocalHardwareProbePort
from tilefoundry_ascend.agent.specs.anchor_specs import parse_simple_env_file


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser supporting standard and --marathon campaign modes."""
    parser = argparse.ArgumentParser(description="Run unattended performance optimization campaign.")
    _add_common_campaign_arguments(parser, required_op=False, default_rounds=None, default_wall=None)
    parser.add_argument("--operators", default="", help="Comma-separated operators for multi-op marathon.")
    parser.add_argument("--marathon", action="store_true", help="Use zero-config marathon runner with Stage-0 & auto env.")
    parser.add_argument("--marathon-mode", choices=("sequential", "round-robin"), default="sequential")
    parser.add_argument("--target-file", "--target-module", dest="target_file", default=None)
    parser.add_argument("--run-final-l4", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pure-golden", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one optimization campaign from CLI arguments."""
    parser = build_arg_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    ops = [x.strip() for x in str(args.operators or "").split(",") if x.strip()]
    if not ops and not args.operator:
        parser.error("one of --operator or --operators is required")
    primary_op = str(args.operator or ops[0])
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    os.environ.setdefault("USE_CUDA", "0")
    ehint = args.env_file or os.environ.get("TILEFOUNDRY_NPU_ENV_FILE", "").strip()
    evals = parse_simple_env_file(Path(ehint).expanduser()) if ehint else {}
    for k in ("TILEFOUNDRY_NPU_AGENT_GATES_BIN", "TILEFOUNDRY_NPU_CANNBENCH_REPO", "TILEFOUNDRY_NPU_TILELANG_SOURCE", "TILEFOUNDRY_NPU_TILELANG_BUILD"):
        if (val := evals.get(k, "").strip()):
            os.environ.setdefault(k, val)
    _ensure_tilelang_on_sys_path(env_vals=evals, env_file=ehint or None, source_root=args.source_root)
    eff_cost_mode = str(args.cost_model_mode).strip().lower()
    eff_prescreen = bool(args.enable_tilesize_prescreen) and eff_cost_mode != "off"
    if args.marathon or len(ops) > 1:
        cfg = MarathonConfig(
            operator_name=primary_op, device_id=str(args.device_id), retry_device_id=str(args.retry_device_id),
            device_ids=tuple(args.device_ids), source_root=Path(args.source_root) if args.source_root else None,
            target_module_path=Path(args.target_file) if args.target_file else None,
            env_file=Path(args.env_file) if args.env_file else None,
            checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
            campaign_kb_path=Path(args.campaign_kb) if args.campaign_kb else None,
            max_rounds=int(args.max_rounds) if args.max_rounds is not None else 80,
            max_wall_clock_min=float(args.max_wall_clock_min) if args.max_wall_clock_min is not None else 720.0,
            k_rule_workers=int(args.k_rule_workers), disable_early_goal_stop=bool(args.disable_early_goal_stop),
            run_final_l4=True, resume=True if args.resume else None,
            enable_tilesize_prescreen=eff_prescreen,
            cost_prescreen_top_k=int(args.cost_prescreen_top_k) if args.cost_prescreen_top_k is not None else None,
            cost_model_mode=eff_cost_mode, allow_unreviewed_distilled_rules=bool(args.allow_unreviewed_distilled_rules),
            pure_golden=bool(args.pure_golden), llm_backend=args.llm_backend, exploit_model=args.exploit_model,
            explore_model=args.explore_model, escalate_model=args.escalate_model, llm_api_base=args.llm_api_base,
        )
        return run_operators_marathon(ops, base_config=cfg, mode=str(args.marathon_mode)) if len(ops) > 1 else run_operator_marathon(cfg)
    if not args.checkpoint:
        parser.error("--checkpoint is required unless --marathon is set")
    pdev, pool = resolve_campaign_device_plan(
        device_id=str(args.device_id), device_ids=tuple(args.device_ids),
        env_device_ids=parse_device_ids(evals.get(CAMPAIGN_DEVICE_IDS_ENV_KEY, "")),
        visible_device_ids=parse_device_ids(os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")),
    )
    llm = create_llm_backend_from_env(overrides=_llm_backend_overrides(evals, args))
    cfg_c = CampaignConfig(
        operator_name=primary_op, max_rounds=int(args.max_rounds) if args.max_rounds is not None else 12,
        max_wall_clock_min=float(args.max_wall_clock_min) if args.max_wall_clock_min is not None else 180.0,
        k_rule_workers=int(args.k_rule_workers), disable_early_goal_stop=bool(args.disable_early_goal_stop),
        run_final_l4=bool(args.run_final_l4), device_id=pdev, retry_device_id=str(args.retry_device_id),
        device_ids=tuple(d for d in pool if d != pdev), enable_tilesize_prescreen=eff_prescreen,
        cost_prescreen_top_k=int(args.cost_prescreen_top_k) if args.cost_prescreen_top_k is not None else None,
        cost_model_mode=eff_cost_mode, allow_unreviewed_distilled_rules=bool(args.allow_unreviewed_distilled_rules),
        round_regime_tags=tuple(args.round_regime or ()), pure_golden=bool(args.pure_golden),
    )
    tracker = BayesianRuleTracker.from_seed_yaml()
    fly = FlywheelDistiller(
        funnel=GateFunnel(skip_stage_b=True), tracker=tracker, probe_port=LocalHardwareProbePort(),
        probe_runner=(lambda sp: run_micro_probe_via_agent_gates(sp, source_root=args.source_root, env_file=args.env_file, device_id=pdev, device_pool=pool)) if args.env_file else None,
    )
    runner = PerfTargetCampaignRunner(
        config=cfg_c, target_module_path=args.target_file, llm_backend=llm, tracker=tracker,
        flywheel_distiller=fly, checkpoint_path=args.checkpoint, campaign_kb_path=args.campaign_kb,
        source_root=args.source_root, env_file=args.env_file,
    )
    report = asyncio.run(runner.run(resume=bool(args.resume)))
    print(json.dumps(report.to_dict(), indent=2))
    return 0


__all__ = ["build_arg_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
