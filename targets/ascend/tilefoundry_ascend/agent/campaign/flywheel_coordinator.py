"""Regime-indexed flywheel checkpoint management and statistical distillation trigger."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from tilefoundry_ascend.agent.flywheel.ablation_distiller import (
    FlywheelDistiller,
    should_trigger_flywheel_checkpoint,
)
from tilefoundry_ascend.agent.kb.bayesian_tracker import (
    BayesianRuleTracker,
    append_winning_edge,
    save_distilled_rule_file,
    save_staging_candidate_rule,
)
from tilefoundry_ascend.agent.specs.anchor_specs import RegimeTargetSpec
from tilefoundry_ascend.ascend.json_record import JsonRecordMixin
from tilefoundry_ascend.ascend.sil.schema import TopologyAxes

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DistilledCheckpoint(JsonRecordMixin):
    """Best verified regime checkpoint used as baseline for subsequent distillation."""

    source: str = ""
    latency_us: float = 0.0
    sigma_paired_us: float = 0.0


class FlywheelCoordinator:
    """Coordinate statistically gated counterfactual distillation and rule persistence."""

    def __init__(self, *, flywheel: FlywheelDistiller | None = None) -> None:
        self._flywheel = flywheel
        self.last_distilled_ckpt_by_regime: dict[str, DistilledCheckpoint] = {}

    def seed_baseline(self, regime_tag: str, source: str, latency_us: float, sigma_us: float) -> None:
        """Initialize baseline checkpoint for a regime if latency is positive."""
        if latency_us > 0.0:
            self.last_distilled_ckpt_by_regime[regime_tag] = DistilledCheckpoint(
                source=source, latency_us=float(latency_us), sigma_paired_us=max(0.0, float(sigma_us))
            )

    def should_trigger(self, regime_tag: str, winner_latency_us: float, winner_sigma_us: float) -> bool:
        """Return True iff cumulative net gain over checkpoint exceeds 3 * sigma_paired."""
        if self._flywheel is None or winner_latency_us <= 0.0:
            return False
        ckpt = self.last_distilled_ckpt_by_regime.get(regime_tag)
        if ckpt is None or ckpt.latency_us <= 0.0:
            return False
        return should_trigger_flywheel_checkpoint(
            ckpt.latency_us, winner_latency_us, sigma_paired_us=max(ckpt.sigma_paired_us, float(winner_sigma_us))
        )

    def serialize_checkpoints(self) -> dict[str, dict[str, object]]:
        """Serialize regime checkpoints for campaign checkpoint persistence."""
        return {tag: ckpt.to_dict() for tag, ckpt in sorted(self.last_distilled_ckpt_by_regime.items())}

    def serialize_latencies(self) -> dict[str, float]:
        """Serialize regime checkpoint latencies for compact checkpoint compatibility."""
        return {tag: float(ckpt.latency_us) for tag, ckpt in sorted(self.last_distilled_ckpt_by_regime.items())}

    def restore_checkpoints(
        self, raw_ckpts: object, *, raw_latencies: object = None, fallback_source: str = ""
    ) -> None:
        """Restore regime checkpoints from serialized campaign checkpoint state."""
        self.last_distilled_ckpt_by_regime = {}
        if isinstance(raw_ckpts, Mapping):
            for tag, item in raw_ckpts.items():
                if isinstance(item, Mapping):
                    self.last_distilled_ckpt_by_regime[str(tag)] = DistilledCheckpoint.from_dict(item)
        if not self.last_distilled_ckpt_by_regime and isinstance(raw_latencies, Mapping):
            for tag, lat in raw_latencies.items():
                if isinstance(lat, (int, float)) and not isinstance(lat, bool) and float(lat) > 0.0:
                    self.last_distilled_ckpt_by_regime[str(tag)] = DistilledCheckpoint(
                        source=fallback_source, latency_us=float(lat), sigma_paired_us=1e-3 * float(lat)
                    )

    def run_distillation(
        self,
        *,
        operator_name: str,
        run_id: str,
        tracker: BayesianRuleTracker,
        baseline_source: str,
        winner_source: str,
        baseline_latency_us: float,
        winner_latency_us: float,
        sigma_paired_us: float,
        func_name: str,
        regime_spec: RegimeTargetSpec,
        topology_axes: TopologyAxes,
        batch_latency_evaluator: Callable[[Sequence[tuple[str, str]]], dict[str, float]],
        campaign_kb_path: Path | None,
        experience_lake_dir: Path | None,
        allow_unreviewed: bool,
        has_l2_executor: bool,
        real_npu_path: bool,
        source_sha256_fn: Callable[[str], str],
        write_text_fn: Callable[[Path, str], None],
        distill_shapes: Sequence[tuple[int, ...]] | None = None,
        rel_gain: float = 0.0,
        worker_kind: str = "",
        round_idx: int = 0,
        rule_id: str | None = None,
        symptom_family: str = "",
        proto_dims: Mapping[str, object] | None = None,
    ) -> list[str]:
        """Execute counterfactual distillation and advance regime checkpoint on admission."""
        if self._flywheel is None:
            return []
        ckpt = self.last_distilled_ckpt_by_regime.get(regime_spec.regime_tag)
        eff_base_src = ckpt.source if (ckpt is not None and ckpt.source.strip()) else baseline_source
        eff_base_lat = ckpt.latency_us if (ckpt is not None and ckpt.latency_us > 0.0) else baseline_latency_us
        eff_sigma = max(float(sigma_paired_us), float(ckpt.sigma_paired_us) if ckpt is not None else 0.0)
        verify_on_cand = bool(
            should_trigger_flywheel_checkpoint(eff_base_lat, winner_latency_us, sigma_paired_us=eff_sigma)
            and has_l2_executor
        )
        _ = (tracker, allow_unreviewed)
        distill_rep = self._flywheel.distill_from_pair(
            baseline_dsl=eff_base_src, winner_dsl=winner_source,
            baseline_latency_us=eff_base_lat, winner_latency_us=winner_latency_us,
            factory_name=func_name, factory_kwargs=regime_spec.sample_builder_kwargs,
            config_selector_name=regime_spec.config_selector_fn or None, dims=proto_dims,
            batch_latency_evaluator=batch_latency_evaluator, topology_axes=topology_axes,
            kb_output_path=campaign_kb_path, rule_id_prefix="R_FLYWHEEL",
            verify_on_candidate=verify_on_cand, max_loo_chunks=3,
            operator_name=operator_name, rule_scope="operator_local",
            active_symptom_family=symptom_family, ordered_tensor_shapes=distill_shapes,
            sigma_paired_us=eff_sigma,
        )

        if experience_lake_dir is not None:
            origin_ts = time.time()
            report_payload = distill_rep.to_dict()
            any_verified = False
            for outcome in distill_rep.distilled_outcomes:
                if outcome.rule is None or not outcome.admitted:
                    continue
                try:
                    save_staging_candidate_rule(experience_lake_dir, outcome.rule, ablation_report=report_payload)
                except (OSError, ValueError) as exc:
                    logger.debug("Persisting staging rule %s skipped: %s", outcome.rule.rule_id, exc)
                verified = bool(getattr(outcome, "verified_on_candidate", outcome.hardware_verified))
                any_verified = any_verified or verified
                if allow_unreviewed or (real_npu_path and verify_on_cand and verified):
                    try:
                        saved_path = save_distilled_rule_file(experience_lake_dir, outcome.rule)
                        meta = json.loads(saved_path.read_text(encoding="utf-8"))
                        if isinstance(meta, dict):
                            meta["origin_timestamp"] = origin_ts
                            meta["origin_operator"] = operator_name
                            write_text_fn(saved_path, json.dumps(meta, indent=2, sort_keys=True) + "\n")
                    except (OSError, ValueError, json.JSONDecodeError) as exc:
                        logger.debug("Persisting distilled rule %s skipped: %s", outcome.rule.rule_id, exc)
            if verify_on_cand and any_verified and winner_latency_us > 0.0:
                self.last_distilled_ckpt_by_regime[regime_spec.regime_tag] = DistilledCheckpoint(
                    source=winner_source, latency_us=float(winner_latency_us), sigma_paired_us=eff_sigma
                )
            try:
                append_winning_edge(
                    experience_lake_dir,
                    {
                        "run_id": run_id, "operator": operator_name, "origin_operator": operator_name,
                        "origin_timestamp": origin_ts, "timestamp": origin_ts, "is_real_npu": bool(real_npu_path),
                        "regime_tag": regime_spec.regime_tag, "round_idx": int(round_idx),
                        "worker_kind": str(worker_kind), "rule_id": rule_id, "symptom_family": symptom_family,
                        "parent_sha": source_sha256_fn(eff_base_src), "winner_sha": source_sha256_fn(winner_source),
                        "baseline_latency_us": float(eff_base_lat), "winner_latency_us": float(winner_latency_us),
                        "relative_gain": float(rel_gain),
                        "loo_chunks": [m.to_dict() for m in distill_rep.kept_positive_chunks],
                        "staged_rule_ids": list(distill_rep.admitted_rule_ids),
                        "distilled_rule_ids": list(distill_rep.admitted_rule_ids),
                        "topology_axes": topology_axes.to_dict(),
                    },
                )
            except (OSError, ValueError) as exc:
                logger.debug("Appending winning edge skipped: %s", exc)
        return list(distill_rep.admitted_rule_ids)


__all__ = ["DistilledCheckpoint", "FlywheelCoordinator"]
