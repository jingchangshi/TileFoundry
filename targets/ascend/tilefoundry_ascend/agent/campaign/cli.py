"""CLI entrypoint for unattended Roofline-driven performance optimization campaigns."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from tilefoundry_ascend.agent.orchestrator.llm_client import create_llm_backend_from_env
from tilefoundry_ascend.agent.probes import LocalHardwareProbePort
from tilefoundry_ascend.agent.specs.anchor_specs import parse_simple_env_file
from tilefoundry_ascend.agent.campaign.campaign_runner import (
    CAMPAIGN_DEVICE_IDS_ENV_KEY,
    CampaignConfig,
    PerfTargetCampaignRunner,
    parse_device_ids,
    resolve_campaign_device_plan,
)
from tilefoundry_ascend.agent.campaign.marathon import (
    MarathonConfig,
    _add_llm_arguments,
    _llm_backend_overrides,
    run_operator_marathon,
)
from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (
    run_micro_probe_via_agent_gates,
)
from tilefoundry_ascend.agent.flywheel.ablation_distiller import FlywheelDistiller
from tilefoundry_ascend.agent.kb.bayesian_tracker import BayesianRuleTracker
from tilefoundry_ascend.agent.orchestrator.gate_funnel import GateFunnel
from tilefoundry_ascend.agent.orchestrator.llm_client import create_llm_backend_from_env


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct CLI parser for PerfTargetCampaignRunner."""
    parser = argparse.ArgumentParser(
        prog="tilefoundry-ascend-campaign",
        description="Run unattended Roofline-driven performance evolution campaigns.",
    )
    parser.add_argument(
        "--operator",
        required=True,
        help="Target operator to optimize (short name e.g. rms_norm or task id e.g. level2/rms_norm).",
    )
    parser.add_argument(
        "--marathon",
        action="store_true",
        help="Use zero-config extreme score-sprint runner with prewarmed L3 and auto env discovery.",
    )
    parser.add_argument("--max-rounds", type=int, default=None)
    parser.add_argument("--max-wall-clock-min", type=float, default=None)
    parser.add_argument("--checkpoint", default=None, help="Atomic checkpoint JSON path.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoint.")
    parser.add_argument("--run-final-l4", action="store_true", help="Run final official L4 gate.")
    parser.add_argument(
        "--device-id",
        default="",
        help=(
            "Authorized NPU device ID; defaults to the first card declared by "
            "TILEFOUNDRY_NPU_CAMPAIGN_DEVICE_IDS, else 1."
        ),
    )
    parser.add_argument(
        "--device-ids",
        type=parse_device_ids,
        default=(),
        help=(
            "Additional authorized NPU cards for whole-L4 attempts, "
            "comma-separated; overrides TILEFOUNDRY_NPU_CAMPAIGN_DEVICE_IDS. "
            "A blocked-env preemption moves to the next untried pool card."
        ),
    )
    parser.add_argument("--env-file", default=None, help="Path to tilefoundry-npu env.sh.")
    _add_llm_arguments(parser)
    parser.add_argument("--source-root", default=None, help="TileFoundry workspace root.")
    parser.add_argument("--target-file", default=None, help="Override target kernels.py path.")
    parser.add_argument("--campaign-kb", default=None, help="Optional YAML path for distilled rules.")
    parser.add_argument("--k-rule-workers", type=int, default=2)
    parser.add_argument(
        "--enable-tilesize-prescreen",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable tile-size prescreen guidance table in creative prompts.",
    )
    parser.add_argument(
        "--cost-prescreen-top-k",
        type=int,
        default=None,
        help="Keep only the top-K cost-model candidates during creative prescreen.",
    )
    parser.add_argument(
        "--cost-model-mode",
        choices=("advisory", "shadow", "off"),
        default="off",
        help="Cost model advisory gate policy mode.",
    )
    parser.add_argument(
        "--pure-golden",
        action="store_true",
        help="Isolate synthesis from authored HIR, DEFAULT_REGIME_SPECS, and repo incumbents.",
    )
    parser.add_argument(
        "--disable-early-goal-stop",
        action="store_true",
        help="Disable early stopping when Roofline/speedup goal is reached.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute a performance evolution campaign and emit JSON summary."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    eff_cost_model_mode = str(args.cost_model_mode).strip().lower()
    eff_enable_prescreen = bool(args.enable_tilesize_prescreen) and eff_cost_model_mode != "off"
    if args.marathon:
        m_cfg = MarathonConfig(
            operator_name=args.operator,
            device_id=str(args.device_id),
            device_ids=tuple(args.device_ids),
            source_root=Path(args.source_root) if args.source_root else None,
            target_module_path=Path(args.target_file) if args.target_file else None,
            env_file=Path(args.env_file) if args.env_file else None,
            checkpoint_path=Path(args.checkpoint) if args.checkpoint else None,
            campaign_kb_path=Path(args.campaign_kb) if args.campaign_kb else None,
            max_rounds=int(args.max_rounds) if args.max_rounds is not None else 80,
            max_wall_clock_min=(
                float(args.max_wall_clock_min)
                if args.max_wall_clock_min is not None
                else 720.0
            ),
            k_rule_workers=int(args.k_rule_workers),
            disable_early_goal_stop=bool(args.disable_early_goal_stop),
            run_final_l4=True,
            resume=True if args.resume else None,
            enable_tilesize_prescreen=eff_enable_prescreen,
            cost_prescreen_top_k=(
                int(args.cost_prescreen_top_k) if args.cost_prescreen_top_k is not None else None
            ),
            cost_model_mode=eff_cost_model_mode,
            pure_golden=bool(args.pure_golden),
            llm_backend=args.llm_backend,
            exploit_model=args.exploit_model,
            explore_model=args.explore_model,
            escalate_model=args.escalate_model,
            llm_api_base=args.llm_api_base,
        )
        return run_operator_marathon(m_cfg)

    if not args.checkpoint:
        parser.error("--checkpoint is required unless --marathon is set")

    env_hint = args.env_file or os.environ.get("TILEFOUNDRY_NPU_ENV_FILE", "").strip()
    env_values = parse_simple_env_file(Path(env_hint).expanduser()) if env_hint else {}
    primary_device_id, device_pool = resolve_campaign_device_plan(
        device_id=str(args.device_id),
        device_ids=tuple(args.device_ids),
        env_device_ids=parse_device_ids(
            env_values.get(CAMPAIGN_DEVICE_IDS_ENV_KEY, "")
        ),
        visible_device_ids=parse_device_ids(
            os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "")
        ),
    )
    llm_backend = create_llm_backend_from_env(overrides=_llm_backend_overrides(env_values, args))

    max_rounds = int(args.max_rounds) if args.max_rounds is not None else 12
    max_wall_clock_min = (
        float(args.max_wall_clock_min) if args.max_wall_clock_min is not None else 180.0
    )
    cfg = CampaignConfig(
        operator_name=args.operator,
        max_rounds=max_rounds,
        max_wall_clock_min=max_wall_clock_min,
        k_rule_workers=args.k_rule_workers,
        disable_early_goal_stop=bool(args.disable_early_goal_stop),
        run_final_l4=bool(args.run_final_l4),
        device_id=primary_device_id,
        device_ids=tuple(device for device in device_pool if device != primary_device_id),
        enable_tilesize_prescreen=eff_enable_prescreen,
        cost_prescreen_top_k=(
            int(args.cost_prescreen_top_k) if args.cost_prescreen_top_k is not None else None
        ),
        cost_model_mode=eff_cost_model_mode,
        pure_golden=bool(args.pure_golden),
    )
    tracker = BayesianRuleTracker.from_seed_yaml()
    flywheel = FlywheelDistiller(
        funnel=GateFunnel(skip_stage_b=True),
        tracker=tracker,
        probe_port=LocalHardwareProbePort(),
        probe_runner=(
            lambda spec: run_micro_probe_via_agent_gates(
                spec,
                source_root=args.source_root,
                env_file=args.env_file,
                device_id=primary_device_id,
                device_pool=device_pool,
            )
        )
        if args.env_file
        else None,
    )
    runner = PerfTargetCampaignRunner(
        config=cfg,
        target_module_path=args.target_file,
        llm_backend=llm_backend,
        tracker=tracker,
        flywheel_distiller=flywheel,
        checkpoint_path=args.checkpoint,
        campaign_kb_path=args.campaign_kb,
        source_root=args.source_root,
        env_file=args.env_file,
    )
    report = asyncio.run(runner.run(resume=bool(args.resume)))
    print(json.dumps(report.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
