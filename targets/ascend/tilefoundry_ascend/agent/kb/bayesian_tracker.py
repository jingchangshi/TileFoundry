"""Bayesian posterior tracker for causal rule realization rates and online pruning."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
import json
from pathlib import Path

from tilefoundry_ascend.agent.kb.schema import AscendCausalRule, load_seed_rules


@dataclass
class RulePosterior:
    """Beta-Binomial posterior state for a single AscendCausalRule."""

    rule_id: str
    alpha: float
    beta: float
    successes: int = 0
    failures: int = 0
    cumulative_speedup: float = 0.0
    pruned: bool = False

    @property
    def trials(self) -> int:
        """Return total observed postcondition evaluations."""
        return self.successes + self.failures

    @property
    def expected_rate(self) -> float:
        """Return posterior mean probability E[p] = alpha / (alpha + beta)."""
        denom = self.alpha + self.beta
        return (self.alpha / denom) if denom > 0.0 else 0.0

    @property
    def mean_speedup(self) -> float:
        """Return average relative latency speedup across positive realizations."""
        return (self.cumulative_speedup / self.successes) if self.successes > 0 else 0.0

    def to_dict(self) -> dict[str, object]:
        """Return JSON-serializable state."""
        return {
            "rule_id": self.rule_id,
            "alpha": self.alpha,
            "beta": self.beta,
            "successes": self.successes,
            "failures": self.failures,
            "cumulative_speedup": self.cumulative_speedup,
            "pruned": self.pruned,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RulePosterior:
        """Reconstruct RulePosterior from JSON payload."""
        return cls(
            rule_id=str(payload["rule_id"]),
            alpha=float(payload["alpha"]),
            beta=float(payload["beta"]),
            successes=int(payload.get("successes", 0)),
            failures=int(payload.get("failures", 0)),
            cumulative_speedup=float(payload.get("cumulative_speedup", 0.0)),
            pruned=bool(payload.get("pruned", False)),
        )


@dataclass
class BayesianRuleTracker:
    """Tracks factorized 3-axis causal rule efficacy and prunes persistently unrealized rules."""

    rules: dict[str, AscendCausalRule] = field(default_factory=dict)
    posteriors: dict[str, RulePosterior] = field(default_factory=dict)
    axis_posteriors: dict[str, dict[str, RulePosterior]] = field(default_factory=dict)

    @classmethod
    def from_rules(cls, rules: Iterable[AscendCausalRule]) -> BayesianRuleTracker:
        """Initialize tracker from seed or persisted AscendCausalRule entries."""
        tracker = cls()
        for rule in rules:
            tracker.admit_rule(rule)
        return tracker

    @classmethod
    def from_seed_yaml(cls, path: str | Path | None = None) -> BayesianRuleTracker:
        """Load seed rules YAML and initialize Bayesian priors."""
        return cls.from_rules(load_seed_rules(path))

    @staticmethod
    def _apply_outcome(
        state: RulePosterior,
        *,
        postcondition_realized: bool,
        speed_gain: float,
    ) -> None:
        if postcondition_realized:
            state.alpha += 1.0
            state.successes += 1
            if speed_gain > 0.0:
                state.cumulative_speedup += float(speed_gain)
        else:
            state.beta += 1.0
            state.failures += 1

    @staticmethod
    def _extract_axis_keys(topology_axes: object) -> tuple[str, str, str]:
        if isinstance(topology_axes, Mapping):
            ce = str(topology_axes["compute_engine"]).strip()
            mr = str(topology_axes["memory_ring"]).strip()
            ir = str(topology_axes["iteration_regularity"]).strip()
        else:
            ce = str(getattr(topology_axes, "compute_engine")).strip()
            mr = str(getattr(topology_axes, "memory_ring")).strip()
            ir = str(getattr(topology_axes, "iteration_regularity")).strip()
        if not ce or not mr or not ir:
            raise ValueError(f"Invalid topology_axes: {topology_axes!r}")
        return (f"ce:{ce}", f"mr:{mr}", f"ir:{ir}")

    def _ensure_axis_posterior(self, axis_key: str, rule_id: str) -> RulePosterior:
        bucket_map = self.axis_posteriors.setdefault(axis_key, {})
        if rule_id not in bucket_map:
            rule = self.rules[rule_id]
            bucket_map[rule_id] = RulePosterior(
                rule_id=rule_id,
                alpha=rule.alpha_prior,
                beta=rule.beta_prior,
            )
        return bucket_map[rule_id]

    def admit_rule(self, rule: AscendCausalRule) -> None:
        """Register or replace a rule while preserving existing statistics if active."""
        self.rules[rule.rule_id] = rule
        if rule.rule_id not in self.posteriors:
            self.posteriors[rule.rule_id] = RulePosterior(
                rule_id=rule.rule_id,
                alpha=rule.alpha_prior,
                beta=rule.beta_prior,
            )

    def record_outcome(
        self,
        rule_id: str,
        *,
        postcondition_realized: bool,
        speed_gain: float = 0.0,
        topology_axes: object | None = None,
    ) -> RulePosterior:
        """Update global and factorized 3-axis Beta posteriors after evaluating a kernel."""
        if rule_id not in self.rules:
            raise KeyError(f"unknown causal rule {rule_id!r}")
        state = self.posteriors[rule_id]
        self._apply_outcome(
            state,
            postcondition_realized=postcondition_realized,
            speed_gain=speed_gain,
        )
        if topology_axes is not None:
            for axis_key in self._extract_axis_keys(topology_axes):
                a_state = self._ensure_axis_posterior(axis_key, rule_id)
                self._apply_outcome(
                    a_state,
                    postcondition_realized=postcondition_realized,
                    speed_gain=speed_gain,
                )
        return state

    def prune_underperforming(
        self,
        *,
        min_trials: int = 5,
        prune_threshold: float = 0.25,
        topology_axes: object | None = None,
    ) -> tuple[str, ...]:
        """Mark rules with persistently low posterior realization rate as pruned."""
        newly_pruned: set[str] = set()
        if topology_axes is None:
            for rule_id, state in sorted(self.posteriors.items()):
                if not state.pruned and state.trials >= min_trials and state.expected_rate < prune_threshold:
                    state.pruned = True
                    newly_pruned.add(rule_id)
            for bucket_map in self.axis_posteriors.values():
                for rule_id, state in bucket_map.items():
                    if not state.pruned and state.trials >= min_trials and state.expected_rate < prune_threshold:
                        state.pruned = True
            return tuple(sorted(newly_pruned))

        axis_keys = self._extract_axis_keys(topology_axes)
        touched_rules: set[str] = set()
        for axis_key in axis_keys:
            bucket_map = self.axis_posteriors.get(axis_key, {})
            for rule_id, state in bucket_map.items():
                if not state.pruned and state.trials >= min_trials and state.expected_rate < prune_threshold:
                    state.pruned = True
                    touched_rules.add(rule_id)
        for rule_id in touched_rules:
            if self.is_pruned(rule_id, topology_axes=topology_axes):
                newly_pruned.add(rule_id)
        return tuple(sorted(newly_pruned))

    def unprune(self, rule_id: str, *, topology_axes: object | None = None) -> None:
        """Re-enable a previously pruned rule globally or within the given topology_axes."""
        if topology_axes is None:
            if rule_id in self.posteriors:
                self.posteriors[rule_id].pruned = False
            for bucket_map in self.axis_posteriors.values():
                if rule_id in bucket_map:
                    bucket_map[rule_id].pruned = False
            return
        for axis_key in self._extract_axis_keys(topology_axes):
            bucket_map = self.axis_posteriors.get(axis_key, {})
            if rule_id in bucket_map:
                bucket_map[rule_id].pruned = False

    def is_pruned(self, rule_id: str, *, topology_axes: object | None = None) -> bool:
        """Return whether rule_id is pruned globally or across a majority of the 3 axes."""
        if topology_axes is not None:
            axis_keys = self._extract_axis_keys(topology_axes)
            axis_states = [
                self.axis_posteriors.get(axis_key, {}).get(rule_id)
                for axis_key in axis_keys
            ]
            if any(st is not None for st in axis_states):
                # Require at least 2 of 3 factorized axes of this topology to be pruned
                # so sharing a single axis (e.g. ir:affine_simd) never causes bleed.
                return sum(1 for st in axis_states if st is not None and st.pruned) >= 2
            if any(rule_id in states for states in self.axis_posteriors.values()):
                return False
        global_state = self.posteriors.get(rule_id)
        return bool(global_state is not None and global_state.pruned)

    def active_rules(self, *, topology_axes: object | None = None) -> tuple[AscendCausalRule, ...]:
        """Return all non-pruned rules sorted deterministically."""
        return tuple(
            rule
            for rule_id, rule in sorted(self.rules.items())
            if not self.is_pruned(rule_id, topology_axes=topology_axes)
        )

    def _blended_axis_metrics(
        self,
        rule: AscendCausalRule,
        *,
        topology_axes: object,
    ) -> tuple[float, float]:
        """Blend factorized axis-level posteriors (ce/mr/ir)."""
        alpha = float(rule.alpha_prior)
        beta = float(rule.beta_prior)
        succ_acc = 0.0
        fail_acc = 0.0
        speedup_acc = 0.0

        axis_keys = self._extract_axis_keys(topology_axes)
        axis_weight = 1.0 / float(len(axis_keys))
        for axis_key in axis_keys:
            a_state = self.axis_posteriors.get(axis_key, {}).get(rule.rule_id)
            if a_state is not None and a_state.trials > 0:
                succ_acc += axis_weight * float(a_state.successes)
                fail_acc += axis_weight * float(a_state.failures)
                speedup_acc += axis_weight * float(a_state.cumulative_speedup)

        total_alpha = alpha + succ_acc
        total_beta = beta + fail_acc
        denom = total_alpha + total_beta
        expected_rate = (total_alpha / denom) if denom > 0.0 else 0.0
        mean_speedup = (speedup_acc / succ_acc) if succ_acc > 0.0 else 0.0
        return expected_rate, mean_speedup

    def rank_applicable_rules(
        self,
        context: Mapping[str, object],
        *,
        top_k: int | None = None,
        topology_axes: object | None = None,
    ) -> tuple[AscendCausalRule, ...]:
        """Filter matching non-pruned rules and rank by factorized 3-axis posterior rate and priority."""
        candidates: list[tuple[tuple[float, float, int, str], AscendCausalRule]] = []
        for rule in self.active_rules(topology_axes=topology_axes):
            if not rule.matches(context):
                continue
            if topology_axes is not None:
                expected_rate, mean_speedup = self._blended_axis_metrics(
                    rule, topology_axes=topology_axes
                )
            else:
                state = self.posteriors[rule.rule_id]
                expected_rate = state.expected_rate
                mean_speedup = state.mean_speedup
            score_key = (
                round(expected_rate, 6),
                round(mean_speedup, 6),
                -int(rule.priority),
                rule.rule_id,
            )
            candidates.append((score_key, rule))
        candidates.sort(key=lambda item: item[0], reverse=True)
        selected = [rule for _key, rule in candidates]
        if top_k is not None and top_k >= 0:
            selected = selected[:top_k]
        return tuple(selected)

    def to_dict(self) -> dict[str, object]:
        """Serialize global and factorized 3-axis tracker posteriors into JSON-safe dict."""
        payload: dict[str, object] = {
            "posteriors": {
                rid: state.to_dict() for rid, state in sorted(self.posteriors.items())
            }
        }
        if self.axis_posteriors:
            payload["axis_posteriors"] = {
                axis_key: {
                    rid: state.to_dict() for rid, state in sorted(states.items())
                }
                for axis_key, states in sorted(self.axis_posteriors.items())
            }
        return payload

    def save_json(self, path: str | Path) -> None:
        """Write tracker posteriors to disk as JSON."""
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def load_posteriors_dict(self, payload: Mapping[str, object]) -> None:
        """Restore global and factorized 3-axis posteriors from a serialized mapping."""
        saved = payload.get("posteriors", {})
        if isinstance(saved, Mapping):
            for rid, raw in saved.items():
                if rid in self.rules and isinstance(raw, Mapping):
                    self.posteriors[rid] = RulePosterior.from_dict(raw)
        saved_axes = payload.get("axis_posteriors", {})
        if isinstance(saved_axes, Mapping):
            for axis_key, states in saved_axes.items():
                if not isinstance(states, Mapping):
                    continue
                bucket_map = self.axis_posteriors.setdefault(str(axis_key), {})
                for rid, raw in states.items():
                    if rid in self.rules and isinstance(raw, Mapping):
                        bucket_map[rid] = RulePosterior.from_dict(raw)

    def load_posteriors_json(self, path: str | Path) -> None:
        """Restore tracked posteriors from disk for matching rule IDs."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            self.load_posteriors_dict(payload)


__all__ = ["BayesianRuleTracker", "RulePosterior"]
