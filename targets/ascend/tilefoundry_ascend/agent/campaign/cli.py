"""CLI entrypoint for unattended Roofline-driven performance optimization campaigns."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from tilefoundry_ascend.agent.campaign.campaign_runner import (
    CampaignConfig,
    PerfTargetCampaignRunner,
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
    parser.add_argument("--max-rounds", type=int, default=12)
    parser.add_argument("--max-wall-clock-min", type=float, default=180.0)
    parser.add_argument("--checkpoint", required=True, help="Atomic checkpoint JSON path.")
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
    args = build_arg_parser().parse_args(argv)
    cfg = CampaignConfig(
        operator_name=args.operator,
        max_rounds=args.max_rounds,
        max_wall_clock_min=args.max_wall_clock_min,
        k_rule_workers=args.k_rule_workers,
        run_final_l4=bool(args.run_final_l4),
        device_id=str(args.device_id),
    )
    tracker = BayesianRuleTracker.from_seed_yaml()
    flywheel = FlywheelDistiller(
        funnel=GateFunnel(skip_stage_b=True),
        tracker=tracker,
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
