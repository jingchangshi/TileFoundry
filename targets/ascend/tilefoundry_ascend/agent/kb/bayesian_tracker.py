"""Bayesian posterior tracker for causal rule realization rates and online pruning."""

from __future__ import annotations

import json
import os
import re
import tempfile
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

from tilefoundry_ascend.agent.kb.schema import (
    AscendCausalRule,
    load_seed_rules,
    normalize_runtime_backend,
)

_SAFE_KB_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_kb_slug(raw: str, default: str = "default") -> str:
    cleaned = _SAFE_KB_SLUG_RE.sub("_", str(raw or "").strip()).strip("._-")
    return cleaned or default


def _serialize_axes(topology_axes: object | None) -> dict[str, str] | None:
    if topology_axes is None:
        return None
    if isinstance(topology_axes, Mapping):
        return {
            "compute_engine": str(topology_axes.get("compute_engine", "")),
            "memory_ring": str(topology_axes.get("memory_ring", "")),
            "iteration_regularity": str(topology_axes.get("iteration_regularity", "")),
        }
    ce = str(getattr(topology_axes, "compute_engine", ""))
    mr = str(getattr(topology_axes, "memory_ring", ""))
    ir = str(getattr(topology_axes, "iteration_regularity", ""))
    if ce and mr and ir:
        return {"compute_engine": ce, "memory_ring": mr, "iteration_regularity": ir}
    return None


def append_kb_event_shard(
    lake_dir: str | Path,
    *,
    run_id: str,
    rule_id: str,
    postcondition_realized: bool,
    speed_gain: float = 0.0,
    topology_axes: object | None = None,
    operator_name: str = "",
    round_idx: int = 0,
    event_kind: str = "outcome",
) -> Path:
    """Append a single lock-free JSONL KB telemetry event into the sharded lake."""
    events_dir = Path(lake_dir) / "kb" / "events"
    events_dir.mkdir(parents=True, exist_ok=True)
    shard = events_dir / f"{_safe_kb_slug(run_id, 'default')}.jsonl"
    payload: dict[str, object] = {
        "timestamp": time.time(),
        "run_id": str(run_id),
        "event_kind": str(event_kind),
        "rule_id": str(rule_id),
        "postcondition_realized": bool(postcondition_realized),
        "speed_gain": float(speed_gain),
        "operator_name": str(operator_name),
        "round_idx": int(round_idx),
        "topology_axes": _serialize_axes(topology_axes),
    }
    with shard.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True) + "\n")
    return shard


def save_distilled_rule_file(lake_dir: str | Path, rule: AscendCausalRule) -> Path:
    """Atomically write or update one distilled causal rule JSON inside the shared KB lake."""
    rules_dir = Path(lake_dir) / "kb" / "distilled_rules"
    rules_dir.mkdir(parents=True, exist_ok=True)
    target = rules_dir / f"{_safe_kb_slug(rule.rule_id, 'rule')}.json"
    raw = json.dumps(rule.to_dict(), indent=2, sort_keys=True) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=f".{_safe_kb_slug(rule.rule_id, 'rule')}_", suffix=".json", dir=str(rules_dir))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(raw)
        tmp_path.replace(target)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
    return target


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
    pruned_at_round: int = -1

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
            "pruned_at_round": self.pruned_at_round,
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
            pruned_at_round=int(payload.get("pruned_at_round", -1)),
        )


@dataclass
class BayesianRuleTracker:
    """Tracks factorized 3-axis causal rule efficacy and prunes persistently unrealized rules."""

    rules: dict[str, AscendCausalRule] = field(default_factory=dict)
    posteriors: dict[str, RulePosterior] = field(default_factory=dict)
    axis_posteriors: dict[str, dict[str, RulePosterior]] = field(default_factory=dict)
    _round_counter: int = field(default=0, init=False, repr=False)
    _synced_event_counts: dict[str, int] = field(default_factory=dict, init=False, repr=False)

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

    def sync_from_experience_lake(
        self,
        lake_dir: str | Path | None,
        *,
        exclude_run_id: str | None = None,
    ) -> int:
        """Incrementally ingest shared distilled rules and peer KB events from Experience Lake."""
        if lake_dir is None or not str(lake_dir).strip():
            return 0
        kb_root = Path(lake_dir) / "kb"
        distilled_dir = kb_root / "distilled_rules"
        if distilled_dir.is_dir():
            for rpath in sorted(distilled_dir.glob("*.json")):
                try:
                    raw = json.loads(rpath.read_text(encoding="utf-8"))
                    if isinstance(raw, Mapping):
                        rule = AscendCausalRule.from_dict(raw)
                        existing = self.rules.get(rule.rule_id)
                        if existing is None:
                            self.admit_rule(rule)
                        elif rule.scope == "cross_op" and existing.scope == "operator_local":
                            self.rules[rule.rule_id] = rule
                except Exception:
                    continue

        events_dir = kb_root / "events"
        exclude_slug = _safe_kb_slug(exclude_run_id, "default") if exclude_run_id else None
        new_events: list[Mapping[str, object]] = []
        if events_dir.is_dir():
            for epath in sorted(events_dir.glob("*.jsonl")):
                if exclude_slug is not None and epath.stem == exclude_slug:
                    continue
                valid_objs: list[Mapping[str, object]] = []
                try:
                    for line in epath.read_text(encoding="utf-8", errors="replace").splitlines():
                        text = line.strip()
                        if not text:
                            continue
                        try:
                            obj = json.loads(text)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(obj, Mapping):
                            valid_objs.append(obj)
                except Exception:
                    continue
                prev = self._synced_event_counts.get(epath.name, 0)
                if len(valid_objs) <= prev:
                    continue
                new_events.extend(valid_objs[prev:])
                self._synced_event_counts[epath.name] = len(valid_objs)

        new_events.sort(key=lambda ev: float(ev.get("timestamp", 0.0) or 0.0))
        for ev in new_events:
            rid = str(ev.get("rule_id") or "")
            if not rid or rid not in self.rules:
                continue
            ridx = int(ev.get("round_idx", 0) or 0)
            axes_raw = ev.get("topology_axes")
            self.record_outcome(
                rid,
                postcondition_realized=bool(ev.get("postcondition_realized", False)),
                speed_gain=float(ev.get("speed_gain", 0.0) or 0.0),
                topology_axes=axes_raw if isinstance(axes_raw, Mapping) else None,
            )
            self.prune_underperforming(
                topology_axes=axes_raw if isinstance(axes_raw, Mapping) else None,
                round_idx=ridx,
            )
        self.promote_verified_local_rules()
        return len(new_events)

    @classmethod
    def from_experience_lake(
        cls,
        lake_dir: str | Path | None = None,
        *,
        seed_path: str | Path | None = None,
    ) -> BayesianRuleTracker:
        """Initialize from seed rules, then replay shared distilled rules and KB event shards."""
        tracker = cls.from_seed_yaml(seed_path)
        tracker.sync_from_experience_lake(lake_dir)
        return tracker

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
        round_idx: int | None = None,
    ) -> tuple[str, ...]:
        """Mark rules with persistently low posterior realization rate as pruned."""
        eff_round = int(round_idx) if round_idx is not None else self._round_counter
        newly_pruned: set[str] = set()
        if topology_axes is None:
            for rule_id, state in sorted(self.posteriors.items()):
                if not state.pruned and state.trials >= min_trials and state.expected_rate < prune_threshold:
                    state.pruned = True
                    state.pruned_at_round = eff_round
                    newly_pruned.add(rule_id)
            for bucket_map in self.axis_posteriors.values():
                for rule_id, state in bucket_map.items():
                    if not state.pruned and state.trials >= min_trials and state.expected_rate < prune_threshold:
                        state.pruned = True
                        state.pruned_at_round = eff_round
            return tuple(sorted(newly_pruned))

        axis_keys = self._extract_axis_keys(topology_axes)
        touched_rules: set[str] = set()
        for axis_key in axis_keys:
            bucket_map = self.axis_posteriors.get(axis_key, {})
            for rule_id, state in bucket_map.items():
                if not state.pruned and state.trials >= min_trials and state.expected_rate < prune_threshold:
                    state.pruned = True
                    state.pruned_at_round = eff_round
                    touched_rules.add(rule_id)
        for rule_id in touched_rules:
            if self.is_pruned(rule_id, topology_axes=topology_axes):
                newly_pruned.add(rule_id)
        return tuple(sorted(newly_pruned))

    def decay_and_revive(
        self,
        *,
        round_idx: int | None = None,
        min_cooldown_rounds: int = 10,
        decay_factor: float = 0.99,
    ) -> tuple[str, ...]:
        """Gently decay stale Beta evidence toward priors and revive rules past cooldown."""
        if round_idx is not None:
            self._round_counter = max(self._round_counter, int(round_idx))
        else:
            self._round_counter += 1
        eff_round = int(round_idx) if round_idx is not None else self._round_counter
        df = max(0.5, min(1.0, float(decay_factor)))

        revived: set[str] = set()
        for rid, state in sorted(self.posteriors.items()):
            rule = self.rules.get(rid)
            ap = float(rule.alpha_prior) if rule is not None else 2.0
            bp = float(rule.beta_prior) if rule is not None else 1.0
            if state.pruned and state.pruned_at_round >= 0 and (eff_round - state.pruned_at_round) >= int(min_cooldown_rounds):
                state.pruned = False
                state.pruned_at_round = -1
                state.alpha = ap + 1.0
                state.beta = bp + 1.0
                revived.add(rid)
            elif df < 1.0:
                state.alpha = ap + max(0.0, state.alpha - ap) * df
                state.beta = bp + max(0.0, state.beta - bp) * df

        for bucket_map in self.axis_posteriors.values():
            for rid, a_state in bucket_map.items():
                rule = self.rules.get(rid)
                ap = float(rule.alpha_prior) if rule is not None else 2.0
                bp = float(rule.beta_prior) if rule is not None else 1.0
                if (
                    a_state.pruned
                    and a_state.pruned_at_round >= 0
                    and (eff_round - a_state.pruned_at_round) >= int(min_cooldown_rounds)
                ):
                    a_state.pruned = False
                    a_state.pruned_at_round = -1
                    a_state.alpha = ap + 1.0
                    a_state.beta = bp + 1.0
                    revived.add(rid)
                elif df < 1.0:
                    a_state.alpha = ap + max(0.0, a_state.alpha - ap) * df
                    a_state.beta = bp + max(0.0, a_state.beta - bp) * df
        return tuple(sorted(revived))

    def promote_verified_local_rules(
        self,
        *,
        min_successes: int = 2,
        min_rate: float = 0.6,
    ) -> tuple[str, ...]:
        """Graduate proven operator-local rules to cross_op visibility."""
        promoted: list[str] = []
        for rid, rule in sorted(self.rules.items()):
            if rule.scope != "operator_local":
                continue
            state = self.posteriors.get(rid)
            if (
                state is not None
                and not state.pruned
                and state.successes >= int(min_successes)
                and state.expected_rate >= float(min_rate)
            ):
                self.rules[rid] = replace(rule, scope="cross_op")
                promoted.append(rid)
        return tuple(promoted)

    def unprune(self, rule_id: str, *, topology_axes: object | None = None) -> None:
        """Re-enable a previously pruned rule globally or within the given topology_axes."""
        if topology_axes is None:
            if rule_id in self.posteriors:
                self.posteriors[rule_id].pruned = False
                self.posteriors[rule_id].pruned_at_round = -1
            for bucket_map in self.axis_posteriors.values():
                if rule_id in bucket_map:
                    bucket_map[rule_id].pruned = False
                    bucket_map[rule_id].pruned_at_round = -1
            return
        for axis_key in self._extract_axis_keys(topology_axes):
            bucket_map = self.axis_posteriors.get(axis_key, {})
            if rule_id in bucket_map:
                bucket_map[rule_id].pruned = False
                bucket_map[rule_id].pruned_at_round = -1

    def is_pruned(self, rule_id: str, *, topology_axes: object | None = None) -> bool:
        """Return whether rule_id is pruned globally or across a majority of the 3 axes."""
        if topology_axes is not None:
            axis_keys = self._extract_axis_keys(topology_axes)
            axis_states = [
                self.axis_posteriors.get(axis_key, {}).get(rule_id)
                for axis_key in axis_keys
            ]
            if any(st is not None for st in axis_states):
                return sum(1 for st in axis_states if st is not None and st.pruned) >= 2
            if any(rule_id in states for states in self.axis_posteriors.values()):
                return False
        global_state = self.posteriors.get(rule_id)
        return bool(global_state is not None and global_state.pruned)

    def active_rules(
        self,
        *,
        topology_axes: object | None = None,
        operator_name: str = "",
        runtime_backend: str | None = None,
    ) -> tuple[AscendCausalRule, ...]:
        """Return all non-pruned rules visible to *operator_name* and *runtime_backend*."""
        eff_op = str(operator_name or "").strip()
        eff_backend = normalize_runtime_backend(runtime_backend)
        selected: list[AscendCausalRule] = []
        for rule_id, rule in sorted(self.rules.items()):
            if self.is_pruned(rule_id, topology_axes=topology_axes):
                continue
            if not rule.supports_backend(eff_backend):
                continue
            if (
                rule.scope == "operator_local"
                and eff_op
                and rule.origin_operator.strip()
                and rule.origin_operator.strip() != eff_op
            ):
                continue
            selected.append(rule)
        return tuple(selected)

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
        operator_name: str = "",
        runtime_backend: str | None = None,
    ) -> tuple[AscendCausalRule, ...]:
        """Filter matching rules by backend and rank by 3-axis posterior rate and priority."""
        eff_op = str(operator_name or "").strip() or str(context.get("operator_name") or "").strip()
        raw_backend = (
            runtime_backend
            if runtime_backend is not None
            else (context.get("runtime_backend") or context.get("backend_kind"))
        )
        eff_backend = normalize_runtime_backend(str(raw_backend)) if raw_backend else None
        candidates: list[tuple[tuple[float, float, int, str], AscendCausalRule]] = []
        for rule in self.active_rules(
            topology_axes=topology_axes,
            operator_name=eff_op,
            runtime_backend=eff_backend,
        ):
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
            candidates.append((score_key, rule.render_for_backend(eff_backend)))
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
        if self._synced_event_counts:
            payload["synced_event_counts"] = {
                str(fname): int(count)
                for fname, count in sorted(self._synced_event_counts.items())
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
        saved_synced = payload.get("synced_event_counts")
        if isinstance(saved_synced, Mapping):
            for fname, count in saved_synced.items():
                self._synced_event_counts[str(fname)] = max(
                    self._synced_event_counts.get(str(fname), 0),
                    int(count),
                )

    def load_posteriors_json(self, path: str | Path) -> None:
        """Restore tracked posteriors from disk for matching rule IDs."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            self.load_posteriors_dict(payload)


__all__ = [
    "BayesianRuleTracker",
    "RulePosterior",
    "append_kb_event_shard",
    "save_distilled_rule_file",
]
