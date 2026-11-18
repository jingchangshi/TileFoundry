"""CLI entrypoint for unattended Roofline-driven performance optimization campaigns."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

from tilefoundry_ascend.agent.campaign.campaign_runner import (
    CampaignConfig,
    PerfTargetCampaignRunner,
)
from tilefoundry_ascend.agent.campaign.marathon import (
    MarathonConfig,
    run_operator_marathon,
)
from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (
    run_micro_probe_via_agent_gates,
)
from tilefoundry_ascend.agent.flywheel.ablation_distiller import FlywheelDistiller
from tilefoundry_ascend.agent.kb.bayesian_tracker import BayesianRuleTracker
from tilefoundry_ascend.agent.orchestrator.gate_funnel import GateFunnel


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct CLI parser for PerfTargetCampaignRunner."""
    parser = argparse.ArgumentParser(
        prog="tilefoundry-ascend-campaign",
        description="Run unattended Roofline-driven performance evolution campaigns.",
    )
    parser.add_argument(
        "--operator",
        required=True,
        choices=("attention_residual", "mla"),
        help="Target production operator to optimize.",
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
    parser.add_argument("--device-id", default="1", help="Authorized NPU device ID.")
    parser.add_argument("--env-file", default=None, help="Path to tilefoundry-npu env.sh.")
    parser.add_argument("--source-root", default=None, help="TileFoundry workspace root.")
    parser.add_argument("--target-file", default=None, help="Override target kernels.py path.")
    parser.add_argument("--campaign-kb", default=None, help="Optional YAML path for distilled rules.")
    parser.add_argument("--k-rule-workers", type=int, default=2)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Execute a performance evolution campaign and emit JSON summary."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.marathon:
        m_cfg = MarathonConfig(
            operator_name=args.operator,
            device_id=str(args.device_id),
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
            run_final_l4=True if args.run_final_l4 else True,
            resume=True if args.resume else None,
        )
        return run_operator_marathon(m_cfg)

    if not args.checkpoint:
        parser.error("--checkpoint is required unless --marathon is set")

    max_rounds = int(args.max_rounds) if args.max_rounds is not None else 12
    max_wall_clock_min = (
        float(args.max_wall_clock_min) if args.max_wall_clock_min is not None else 180.0
    )
    cfg = CampaignConfig(
        operator_name=args.operator,
        max_rounds=max_rounds,
        max_wall_clock_min=max_wall_clock_min,
        k_rule_workers=args.k_rule_workers,
        run_final_l4=bool(args.run_final_l4),
        device_id=str(args.device_id),
    )
    tracker = BayesianRuleTracker.from_seed_yaml()
    flywheel = FlywheelDistiller(
        funnel=GateFunnel(skip_stage_b=True),
        tracker=tracker,
        probe_runner=(
            lambda spec: run_micro_probe_via_agent_gates(
                spec,
                source_root=args.source_root,
                env_file=args.env_file,
                device_id=str(args.device_id),
            )
        )
        if args.env_file
        else None,
    )
    runner = PerfTargetCampaignRunner(
        config=cfg,
        target_module_path=args.target_file,
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
