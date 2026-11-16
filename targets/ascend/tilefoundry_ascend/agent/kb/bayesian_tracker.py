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
    """Tracks causal rule efficacy and prunes persistently unrealized rules."""

    rules: dict[str, AscendCausalRule] = field(default_factory=dict)
    posteriors: dict[str, RulePosterior] = field(default_factory=dict)

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
    ) -> RulePosterior:
        """Update Beta posterior after evaluating a candidate kernel transform."""
        if rule_id not in self.rules:
            raise KeyError(f"unknown causal rule {rule_id!r}")
        state = self.posteriors[rule_id]
        if postcondition_realized:
            state.alpha += 1.0
            state.successes += 1
            if speed_gain > 0.0:
                state.cumulative_speedup += float(speed_gain)
        else:
            state.beta += 1.0
            state.failures += 1
        return state

    def prune_underperforming(
        self,
        *,
        min_trials: int = 5,
        prune_threshold: float = 0.25,
    ) -> tuple[str, ...]:
        """Mark rules with persistently low posterior realization rate as pruned."""
        newly_pruned: list[str] = []
        for rule_id, state in sorted(self.posteriors.items()):
            if not state.pruned and state.trials >= min_trials:
                if state.expected_rate < prune_threshold:
                    state.pruned = True
                    newly_pruned.append(rule_id)
        return tuple(newly_pruned)

    def unprune(self, rule_id: str) -> None:
        """Re-enable a previously pruned rule."""
        if rule_id in self.posteriors:
            self.posteriors[rule_id].pruned = False

    def active_rules(self) -> tuple[AscendCausalRule, ...]:
        """Return all non-pruned rules sorted deterministically."""
        return tuple(
            rule
            for rule_id, rule in sorted(self.rules.items())
            if not self.posteriors[rule_id].pruned
        )

    def rank_applicable_rules(
        self,
        context: Mapping[str, object],
        *,
        top_k: int | None = None,
    ) -> tuple[AscendCausalRule, ...]:
        """Filter matching non-pruned rules and rank by posterior rate and priority."""
        candidates: list[tuple[tuple[float, float, int, str], AscendCausalRule]] = []
        for rule in self.active_rules():
            if not rule.matches(context):
                continue
            state = self.posteriors[rule.rule_id]
            score_key = (
                round(state.expected_rate, 6),
                round(state.mean_speedup, 6),
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
        """Serialize tracker posteriors and rule IDs into JSON-safe dict."""
        return {
            "posteriors": {
                rid: state.to_dict() for rid, state in sorted(self.posteriors.items())
            }
        }

    def save_json(self, path: str | Path) -> None:
        """Write tracker posteriors to disk as JSON."""
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def load_posteriors_json(self, path: str | Path) -> None:
        """Restore tracked posteriors from disk for matching rule IDs."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        saved = payload.get("posteriors", {})
        for rid, raw in saved.items():
            if rid in self.rules:
                self.posteriors[rid] = RulePosterior.from_dict(raw)


__all__ = ["BayesianRuleTracker", "RulePosterior"]
