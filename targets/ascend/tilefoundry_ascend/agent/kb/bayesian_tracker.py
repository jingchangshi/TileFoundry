"""Bayesian posterior tracker for causal rule realization rates and online pruning."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TypeVar

from tilefoundry_ascend.agent.kb.schema import (
    AscendCausalRule,
    SymptomFamily,
    load_seed_rules,
    normalize_runtime_backend,
)
from tilefoundry_ascend.ascend.json_record import JsonRecordMixin, atomic_write_text

_SAFE_KB_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")
_T = TypeVar("_T")


def _normalize_symptom_family(raw_family: str | SymptomFamily) -> str:
    """Normalize a SymptomFamily enum or raw string to its lowercase canonical identifier."""
    return (raw_family.value if isinstance(raw_family, SymptomFamily) else str(raw_family or "")).strip().lower()


def _safe_kb_slug(raw: str, default: str = "default") -> str:
    return _SAFE_KB_SLUG_RE.sub("_", str(raw or "").strip()).strip("._-") or default


def _serialize_axes(topology_axes: object | None) -> dict[str, str] | None:
    if topology_axes is None:
        return None
    keys = ("compute_engine", "memory_ring", "iteration_regularity")
    if isinstance(topology_axes, Mapping):
        return {k: str(topology_axes.get(k, "")) for k in keys}
    vals = {k: str(getattr(topology_axes, k, "")) for k in keys}
    return vals if all(vals.values()) else None


def _append_jsonl(path: Path, payload: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(payload), sort_keys=True) + "\n")
    return path


def _write_json(target: Path, payload: Mapping[str, object]) -> Path:
    atomic_write_text(target, json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return target


def append_kb_event_shard(
    lake_dir: str | Path, *, run_id: str, rule_id: str, postcondition_realized: bool,
    speed_gain: float = 0.0, topology_axes: object | None = None,
    operator_name: str = "", round_idx: int = 0, event_kind: str = "outcome",
) -> Path:
    """Append a single lock-free JSONL KB telemetry event into the sharded lake."""
    return _append_jsonl(Path(lake_dir) / "kb" / "events" / f"{_safe_kb_slug(run_id, 'default')}.jsonl", {
        "timestamp": time.time(), "run_id": str(run_id), "event_kind": str(event_kind),
        "rule_id": str(rule_id), "postcondition_realized": bool(postcondition_realized),
        "speed_gain": float(speed_gain), "operator_name": str(operator_name),
        "round_idx": int(round_idx), "topology_axes": _serialize_axes(topology_axes),
    })


def save_distilled_rule_file(lake_dir: str | Path, rule: AscendCausalRule) -> Path:
    """Atomically write or update one distilled causal rule JSON inside the shared KB lake."""
    return _write_json(Path(lake_dir) / "kb" / "distilled_rules" / f"{_safe_kb_slug(rule.rule_id, 'rule')}.json", rule.to_dict())


def save_staging_candidate_rule(
    lake_dir: str | Path, rule: AscendCausalRule, *, ablation_report: Mapping[str, object] | None = None
) -> Path:
    """Atomically write an unreviewed candidate rule draft to offline curation staging."""
    payload: dict[str, object] = {"timestamp": time.time(), "status": "pending_review", "rule": rule.to_dict()}
    if ablation_report is not None:
        payload["ablation_report"] = dict(ablation_report)
    return _write_json(Path(lake_dir) / "kb" / "staging_candidates" / f"{_safe_kb_slug(rule.rule_id, 'rule')}.json", payload)


def append_winning_edge(lake_dir: str | Path, record: Mapping[str, object]) -> Path:
    """Append one winning optimization edge record to ``kb/evidence/winning_edges.jsonl``."""
    return _append_jsonl(Path(lake_dir) / "kb" / "evidence" / "winning_edges.jsonl", {"timestamp": time.time(), **record})


def append_rule_misapplication(lake_dir: str | Path, record: Mapping[str, object]) -> Path:
    """Append one rule misapplication record to ``kb/evidence/rule_misapplications.jsonl``."""
    return _append_jsonl(Path(lake_dir) / "kb" / "evidence" / "rule_misapplications.jsonl", {"timestamp": time.time(), **record})


@dataclass(frozen=True)
class TabuEntry(JsonRecordMixin):
    """Intra-campaign rule cooldown scoped to one regime and parent source digest."""

    regime_tag: str
    rule_id: str
    parent_sha: str
    expire_round: int
    reason: str = ""


@dataclass(frozen=True)
class FamilyTabuEntry(JsonRecordMixin):
    """Intra-campaign symptom-family cooldown scoped to one regime and parent digest."""

    regime_tag: str
    symptom_family: str
    parent_sha: str
    expire_round: int
    reason: str = ""


@dataclass
class RulePosterior(JsonRecordMixin):
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


def _upsert_tabu(entries: list[_T], match_fn: Callable[[_T], bool], new_entry: _T) -> list[_T]:
    return [entry for entry in entries if not match_fn(entry)] + [new_entry]


def _tabu_active(
    entries: Sequence[_T],
    reg: str, sha: str, cur_round: int, key_match: Callable[[_T], bool],
) -> bool:
    return any(e.regime_tag == reg and e.parent_sha == sha and cur_round <= e.expire_round and key_match(e) for e in entries)


def _prune_bucket(bucket: Mapping[str, RulePosterior], min_trials: int, prune_threshold: float, eff_round: int) -> set[str]:
    pruned_ids: set[str] = set()
    for rule_id, state in sorted(bucket.items()):
        if not state.pruned and state.trials >= min_trials and state.expected_rate < prune_threshold:
            state.pruned, state.pruned_at_round = True, eff_round
            pruned_ids.add(rule_id)
    return pruned_ids


def _read_jsonl_mappings(path: Path) -> list[Mapping[str, object]]:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return []
    objs: list[Mapping[str, object]] = []
    for line in lines:
        if text := line.strip():
            try:
                if isinstance(obj := json.loads(text), Mapping):
                    objs.append(obj)
            except json.JSONDecodeError:
                continue
    return objs


@dataclass
class BayesianRuleTracker:
    """Tracks factorized 3-axis causal rule efficacy, scoped tabu cooldowns, and pruning."""

    rules: dict[str, AscendCausalRule] = field(default_factory=dict)
    posteriors: dict[str, RulePosterior] = field(default_factory=dict)
    axis_posteriors: dict[str, dict[str, RulePosterior]] = field(default_factory=dict)
    tabu_entries: list[TabuEntry] = field(default_factory=list)
    family_tabu_entries: list[FamilyTabuEntry] = field(default_factory=list)
    family_failure_streaks: dict[tuple[str, str, str], int] = field(default_factory=dict)
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
        self, lake_dir: str | Path | None, *, exclude_run_id: str | None = None, allow_unreviewed_distilled_rules: bool = False
    ) -> int:
        """Incrementally ingest peer KB posterior events from Experience Lake.

        Unreviewed distilled rules under ``kb/distilled_rules`` remain excluded from
        online active rules unless *allow_unreviewed_distilled_rules* is explicitly True.
        """
        if lake_dir is None or not str(lake_dir).strip():
            return 0
        kb_root = Path(lake_dir) / "kb"
        if allow_unreviewed_distilled_rules and (dist_dir := kb_root / "distilled_rules").is_dir():
            for rpath in sorted(dist_dir.glob("*.json")):
                try:
                    if isinstance(raw := json.loads(rpath.read_text(encoding="utf-8")), Mapping):
                        rule = AscendCausalRule.from_dict(raw)
                        if (existing := self.rules.get(rule.rule_id)) is None:
                            self.admit_rule(rule)
                        elif rule.scope == "cross_op" and existing.scope == "operator_local":
                            self.rules[rule.rule_id] = rule
                except (OSError, ValueError, KeyError, json.JSONDecodeError):
                    continue
        exclude_slug = _safe_kb_slug(exclude_run_id, "default") if exclude_run_id else None
        new_events: list[Mapping[str, object]] = []
        if (events_dir := kb_root / "events").is_dir():
            for epath in sorted(events_dir.glob("*.jsonl")):
                if exclude_slug is not None and epath.stem == exclude_slug:
                    continue
                valid_objs = _read_jsonl_mappings(epath)
                if len(valid_objs) > (prev := self._synced_event_counts.get(epath.name, 0)):
                    new_events.extend(valid_objs[prev:])
                    self._synced_event_counts[epath.name] = len(valid_objs)
        new_events.sort(key=lambda ev: float(ev.get("timestamp", 0.0) or 0.0))
        for ev in new_events:
            if (rid := str(ev.get("rule_id") or "")) and rid in self.rules:
                axes = ar if isinstance(ar := ev.get("topology_axes"), Mapping) else None
                self.record_outcome(rid, postcondition_realized=bool(ev.get("postcondition_realized", False)), speed_gain=float(ev.get("speed_gain", 0.0) or 0.0), topology_axes=axes)
                self.prune_underperforming(topology_axes=axes, round_idx=int(ev.get("round_idx", 0) or 0))
        if allow_unreviewed_distilled_rules:
            self.promote_verified_local_rules()
        return len(new_events)

    @classmethod
    def from_experience_lake(
        cls, lake_dir: str | Path | None = None, *, seed_path: str | Path | None = None,
        allow_unreviewed_distilled_rules: bool = False,
    ) -> BayesianRuleTracker:
        """Initialize from seed rules, then replay Experience Lake KB event shards."""
        tracker = cls.from_seed_yaml(seed_path)
        tracker.sync_from_experience_lake(lake_dir, allow_unreviewed_distilled_rules=allow_unreviewed_distilled_rules)
        return tracker

    def record_scoped_failure(
        self, *, regime_tag: str, rule_id: str, parent_sha: str, current_round: int,
        reason: str, cooldown_rounds: int = 2, family_cooldown_rounds: int = 1,
    ) -> None:
        """Record a scoped rule failure and escalate to family cooldown on streak >= 2."""
        reg, rid, sha = str(regime_tag).strip(), str(rule_id).strip(), str(parent_sha).strip()
        if not reg or not rid or not sha:
            return
        cur_round = int(current_round)
        self.tabu_entries = _upsert_tabu(
            self.tabu_entries, lambda e: e.regime_tag == reg and e.rule_id == rid and e.parent_sha == sha,
            TabuEntry(reg, rid, sha, cur_round + max(0, int(cooldown_rounds)), str(reason)),
        )
        if not (family := self.rules[rid].symptom_family if rid in self.rules else ""):
            return
        streak = self.family_failure_streaks[(reg, family, sha)] = self.family_failure_streaks.get((reg, family, sha), 0) + 1
        if streak >= 2:
            self.family_tabu_entries = _upsert_tabu(
                self.family_tabu_entries, lambda e: e.regime_tag == reg and e.symptom_family == family and e.parent_sha == sha,
                FamilyTabuEntry(reg, family, sha, cur_round + max(0, int(family_cooldown_rounds)), f"consecutive_family_failures:{streak}:{reason}"),
            )

    def reset_scoped_family_streak(self, *, regime_tag: str, rule_id: str, parent_sha: str) -> None:
        """Clear consecutive failure streak for the rule's symptom family on success."""
        if (rule := self.rules.get(str(rule_id).strip())) is not None and rule.symptom_family:
            self.family_failure_streaks.pop((str(regime_tag).strip(), rule.symptom_family, str(parent_sha).strip()), None)

    def is_family_tabued(
        self, *, regime_tag: str, symptom_family: str | SymptomFamily, parent_sha: str, current_round: int
    ) -> bool:
        """Return whether symptom_family is under active scoped cooldown."""
        reg, fam, sha = str(regime_tag).strip(), _normalize_symptom_family(symptom_family), str(parent_sha).strip()
        return bool(reg and fam and sha) and _tabu_active(
            self.family_tabu_entries, reg, sha, int(current_round), lambda e: _normalize_symptom_family(e.symptom_family) == fam
        )

    def is_rule_tabued(self, *, regime_tag: str, rule_id: str, parent_sha: str, current_round: int) -> bool:
        """Return whether rule_id or its symptom family is under active scoped cooldown."""
        reg, rid, sha = str(regime_tag).strip(), str(rule_id).strip(), str(parent_sha).strip()
        if not reg or not rid or not sha:
            return False
        cur_round = int(current_round)
        if _tabu_active(self.tabu_entries, reg, sha, cur_round, lambda e: e.rule_id == rid):
            return True
        rule = self.rules.get(rid)
        return bool(rule is not None and rule.symptom_family and self.is_family_tabued(
            regime_tag=reg, symptom_family=rule.symptom_family, parent_sha=sha, current_round=cur_round
        ))

    @staticmethod
    def _apply_outcome(state: RulePosterior, *, postcondition_realized: bool, speed_gain: float) -> None:
        if postcondition_realized:
            state.alpha, state.successes = state.alpha + 1.0, state.successes + 1
            if speed_gain > 0.0:
                state.cumulative_speedup += float(speed_gain)
        else:
            state.beta, state.failures = state.beta + 1.0, state.failures + 1

    @staticmethod
    def _extract_axis_keys(topology_axes: object) -> tuple[str, str, str]:
        getter = (lambda k: topology_axes[k]) if isinstance(topology_axes, Mapping) else (lambda k: getattr(topology_axes, k))
        ce, mr, ir = str(getter("compute_engine")).strip(), str(getter("memory_ring")).strip(), str(getter("iteration_regularity")).strip()
        if not ce or not mr or not ir:
            raise ValueError(f"Invalid topology_axes: {topology_axes!r}")
        return (f"ce:{ce}", f"mr:{mr}", f"ir:{ir}")

    def _ensure_axis_posterior(self, axis_key: str, rule_id: str) -> RulePosterior:
        bucket = self.axis_posteriors.setdefault(axis_key, {})
        if rule_id not in bucket:
            rule = self.rules[rule_id]
            bucket[rule_id] = RulePosterior(rule_id, rule.alpha_prior, rule.beta_prior)
        return bucket[rule_id]

    def admit_rule(self, rule: AscendCausalRule) -> None:
        """Register or replace a rule while preserving existing statistics if active."""
        self.rules[rule.rule_id] = rule
        self.posteriors.setdefault(rule.rule_id, RulePosterior(rule.rule_id, rule.alpha_prior, rule.beta_prior))

    def record_outcome(
        self, rule_id: str, *, postcondition_realized: bool, speed_gain: float = 0.0, topology_axes: object | None = None
    ) -> RulePosterior:
        """Update global and factorized 3-axis Beta posteriors after evaluating a kernel."""
        if rule_id not in self.rules:
            raise KeyError(f"unknown causal rule {rule_id!r}")
        state = self.posteriors[rule_id]
        self._apply_outcome(state, postcondition_realized=postcondition_realized, speed_gain=speed_gain)
        if topology_axes is not None:
            for ak in self._extract_axis_keys(topology_axes):
                self._apply_outcome(self._ensure_axis_posterior(ak, rule_id), postcondition_realized=postcondition_realized, speed_gain=speed_gain)
        return state

    def prune_underperforming(
        self, *, min_trials: int = 5, prune_threshold: float = 0.25,
        topology_axes: object | None = None, round_idx: int | None = None,
    ) -> tuple[str, ...]:
        """Mark rules with persistently low posterior realization rate as pruned."""
        eff_round = int(round_idx) if round_idx is not None else self._round_counter
        if topology_axes is None:
            newly_pruned = _prune_bucket(self.posteriors, min_trials, prune_threshold, eff_round)
            for bucket in self.axis_posteriors.values():
                _prune_bucket(bucket, min_trials, prune_threshold, eff_round)
            return tuple(sorted(newly_pruned))
        touched = set().union(*(_prune_bucket(self.axis_posteriors.get(ak, {}), min_trials, prune_threshold, eff_round) for ak in self._extract_axis_keys(topology_axes)))
        return tuple(sorted(rid for rid in touched if self.is_pruned(rid, topology_axes=topology_axes)))

    def decay_and_revive(self, *, round_idx: int | None = None, min_cooldown_rounds: int = 10, decay_factor: float = 0.99) -> tuple[str, ...]:
        """Gently decay stale Beta evidence toward priors and revive rules past cooldown."""
        self._round_counter = max(self._round_counter, int(round_idx)) if round_idx is not None else (self._round_counter + 1)
        eff_round, df, revived = (int(round_idx) if round_idx is not None else self._round_counter), max(0.5, min(1.0, float(decay_factor))), set()
        for bucket in (self.posteriors, *self.axis_posteriors.values()):
            for rid, st in sorted(bucket.items()):
                rule = self.rules.get(rid)
                ap, bp = (float(rule.alpha_prior), float(rule.beta_prior)) if rule is not None else (2.0, 1.0)
                if st.pruned and st.pruned_at_round >= 0 and (eff_round - st.pruned_at_round) >= int(min_cooldown_rounds):
                    st.pruned, st.pruned_at_round, st.alpha, st.beta = False, -1, ap + 1.0, bp + 1.0
                    revived.add(rid)
                elif df < 1.0:
                    st.alpha, st.beta = ap + max(0.0, st.alpha - ap) * df, bp + max(0.0, st.beta - bp) * df
        return tuple(sorted(revived))

    def promote_verified_local_rules(self, *, min_successes: int = 2, min_rate: float = 0.6) -> tuple[str, ...]:
        """Graduate proven operator-local rules to cross_op visibility."""
        promoted: list[str] = []
        for rid, rule in sorted(self.rules.items()):
            st = self.posteriors.get(rid)
            if rule.scope == "operator_local" and st is not None and not st.pruned and st.successes >= int(min_successes) and st.expected_rate >= float(min_rate):
                self.rules[rid] = replace(rule, scope="cross_op")
                promoted.append(rid)
        return tuple(promoted)

    def unprune(self, rule_id: str, *, topology_axes: object | None = None) -> None:
        """Re-enable a previously pruned rule globally or within the given topology_axes."""
        buckets = (self.posteriors, *self.axis_posteriors.values()) if topology_axes is None else (self.axis_posteriors.get(k, {}) for k in self._extract_axis_keys(topology_axes))
        for bucket in buckets:
            if rule_id in bucket:
                bucket[rule_id].pruned, bucket[rule_id].pruned_at_round = False, -1

    def is_pruned(self, rule_id: str, *, topology_axes: object | None = None) -> bool:
        """Return whether rule_id is pruned globally or across a majority of the 3 axes."""
        if topology_axes is not None:
            axis_states = [self.axis_posteriors.get(k, {}).get(rule_id) for k in self._extract_axis_keys(topology_axes)]
            if any(st is not None for st in axis_states):
                return sum(1 for st in axis_states if st is not None and st.pruned) >= 2
            if any(rule_id in states for states in self.axis_posteriors.values()):
                return False
        global_state = self.posteriors.get(rule_id)
        return bool(global_state is not None and global_state.pruned)

    def active_rules(self, *, topology_axes: object | None = None, operator_name: str = "", runtime_backend: str | None = None) -> tuple[AscendCausalRule, ...]:
        """Return all non-pruned rules visible to *operator_name* and *runtime_backend*."""
        eff_op, eff_backend = str(operator_name or "").strip(), normalize_runtime_backend(runtime_backend)
        return tuple(
            rule for rule_id, rule in sorted(self.rules.items())
            if not self.is_pruned(rule_id, topology_axes=topology_axes) and rule.supports_backend(eff_backend)
            and not (rule.scope == "operator_local" and eff_op and rule.origin_operator.strip() and rule.origin_operator.strip() != eff_op)
        )

    def _blended_axis_metrics(self, rule: AscendCausalRule, *, topology_axes: object) -> tuple[float, float]:
        """Blend factorized axis-level posteriors (ce/mr/ir)."""
        axis_keys = self._extract_axis_keys(topology_axes)
        aw, succ_acc, fail_acc, speed_acc = 1.0 / float(len(axis_keys)), 0.0, 0.0, 0.0
        for ak in axis_keys:
            if (a_st := self.axis_posteriors.get(ak, {}).get(rule.rule_id)) is not None and a_st.trials > 0:
                succ_acc += aw * float(a_st.successes)
                fail_acc += aw * float(a_st.failures)
                speed_acc += aw * float(a_st.cumulative_speedup)
        total_alpha = float(rule.alpha_prior) + succ_acc
        denom = total_alpha + float(rule.beta_prior) + fail_acc
        return (total_alpha / denom if denom > 0.0 else 0.0), (speed_acc / succ_acc if succ_acc > 0.0 else 0.0)

    def rank_applicable_rules(
        self, context: Mapping[str, object], *, top_k: int | None = None, topology_axes: object | None = None,
        operator_name: str = "", runtime_backend: str | None = None, regime_tag: str = "",
        parent_sha: str = "", current_round: int | None = None, symptom_ranking: Sequence[tuple[str | SymptomFamily, float]] = (),
    ) -> tuple[AscendCausalRule, ...]:
        """Filter matching rules by backend and scoped tabu, then rank by symptom and posterior."""
        eff_op = str(operator_name or "").strip() or str(context.get("operator_name") or "").strip()
        raw_backend = runtime_backend if runtime_backend is not None else (context.get("runtime_backend") or context.get("backend_kind"))
        eff_backend = normalize_runtime_backend(str(raw_backend)) if raw_backend else None
        has_scope = bool(regime_tag and parent_sha and current_round is not None)
        sev_by_fam: dict[str, float] = {}
        for raw_fam, raw_sev in symptom_ranking:
            fk = _normalize_symptom_family(raw_fam)
            if fk and not (has_scope and self.is_family_tabued(regime_tag=regime_tag, symptom_family=fk, parent_sha=parent_sha, current_round=int(current_round))):
                sev_by_fam[fk] = max(float(raw_sev), sev_by_fam.get(fk, float("-inf")))
        candidates: list[tuple[tuple[int, float, float, float, int, str], AscendCausalRule]] = []
        for rule in self.active_rules(topology_axes=topology_axes, operator_name=eff_op, runtime_backend=eff_backend):
            if (has_scope and self.is_rule_tabued(regime_tag=regime_tag, rule_id=rule.rule_id, parent_sha=parent_sha, current_round=int(current_round))) or not rule.matches(context):
                continue
            rate, speedup = self._blended_axis_metrics(rule, topology_axes=topology_axes) if topology_axes is not None else (self.posteriors[rule.rule_id].expected_rate, self.posteriors[rule.rule_id].mean_speedup)
            fk = _normalize_symptom_family(rule.symptom_family)
            fam_sev = sev_by_fam.get(fk, 0.0) if fk else 0.0
            boost_tier = 1 if fam_sev > 0.15 else 0
            candidates.append(((boost_tier, round(fam_sev, 6) if boost_tier else 0.0, round(rate, 6), round(speedup, 6), -int(rule.priority), rule.rule_id), rule.render_for_backend(eff_backend)))
        candidates.sort(key=lambda item: item[0], reverse=True)
        selected = [rule for _key, rule in candidates]
        return tuple(selected[:top_k] if top_k is not None and top_k >= 0 else selected)

    def select_top_rules(
        self,
        context: Mapping[str, object],
        *,
        top_k: int | None = None,
        topology_axes: object | None = None,
        operator_name: str = "",
        runtime_backend: str | None = None,
        regime_tag: str = "",
        parent_sha: str = "",
        current_round: int | None = None,
        symptom_ranking: Sequence[tuple[str | SymptomFamily, float]] = (),
    ) -> tuple[AscendCausalRule, ...]:
        """Select the highest-ranked matching rules after scoped tabu filtering."""
        return self.rank_applicable_rules(
            context,
            top_k=top_k,
            topology_axes=topology_axes,
            operator_name=operator_name,
            runtime_backend=runtime_backend,
            regime_tag=regime_tag,
            parent_sha=parent_sha,
            current_round=current_round,
            symptom_ranking=symptom_ranking,
        )

    def rank_matching_rules(
        self,
        context: Mapping[str, object],
        *,
        top_k: int | None = None,
        topology_axes: object | None = None,
        operator_name: str = "",
        runtime_backend: str | None = None,
        regime_tag: str = "",
        parent_sha: str = "",
        current_round: int | None = None,
        symptom_ranking: Sequence[tuple[str | SymptomFamily, float]] = (),
    ) -> tuple[AscendCausalRule, ...]:
        """Rank all matching rules after backend and scoped tabu filtering."""
        return self.rank_applicable_rules(
            context,
            top_k=top_k,
            topology_axes=topology_axes,
            operator_name=operator_name,
            runtime_backend=runtime_backend,
            regime_tag=regime_tag,
            parent_sha=parent_sha,
            current_round=current_round,
            symptom_ranking=symptom_ranking,
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize global, 3-axis posteriors, and scoped tabu state into JSON-safe dict."""
        payload: dict[str, object] = {"posteriors": {rid: st.to_dict() for rid, st in sorted(self.posteriors.items())}}
        if self.axis_posteriors:
            payload["axis_posteriors"] = {ak: {rid: st.to_dict() for rid, st in sorted(states.items())} for ak, states in sorted(self.axis_posteriors.items())}
        if self._synced_event_counts:
            payload["synced_event_counts"] = {str(fn): int(c) for fn, c in sorted(self._synced_event_counts.items())}
        for attr in ("tabu_entries", "family_tabu_entries"):
            if items := getattr(self, attr):
                payload[attr] = [e.to_dict() for e in items]
        if self.family_failure_streaks:
            payload["family_failure_streaks"] = [
                {"regime_tag": r, "symptom_family": f, "parent_sha": s, "count": int(c)}
                for (r, f, s), c in sorted(self.family_failure_streaks.items())
            ]
        return payload

    def save_json(self, path: str | Path) -> None:
        """Write tracker posteriors and scoped tabu state to disk as JSON."""
        atomic_write_text(path, json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    def _restore_bucket(self, dst: dict[str, RulePosterior], src: object) -> None:
        if isinstance(src, Mapping):
            for rid, raw in src.items():
                if rid in self.rules and isinstance(raw, Mapping):
                    dst[rid] = RulePosterior.from_dict(raw)

    def load_posteriors_dict(self, payload: Mapping[str, object]) -> None:
        """Restore global posteriors, 3-axis posteriors, and scoped tabu state from mapping."""
        self._restore_bucket(self.posteriors, payload.get("posteriors"))
        if isinstance(saved_axes := payload.get("axis_posteriors"), Mapping):
            for axis_key, states in saved_axes.items():
                if isinstance(states, Mapping):
                    self._restore_bucket(self.axis_posteriors.setdefault(str(axis_key), {}), states)
        if isinstance(saved_synced := payload.get("synced_event_counts"), Mapping):
            for fname, count in saved_synced.items():
                self._synced_event_counts[str(fname)] = max(self._synced_event_counts.get(str(fname), 0), int(count))
        if isinstance(raw_tabu := payload.get("tabu_entries"), list):
            self.tabu_entries = [TabuEntry.from_dict(i) for i in raw_tabu if isinstance(i, Mapping)]
        if isinstance(raw_fam := payload.get("family_tabu_entries"), list):
            self.family_tabu_entries = [FamilyTabuEntry.from_dict(i) for i in raw_fam if isinstance(i, Mapping)]
        if isinstance(raw_streaks := payload.get("family_failure_streaks"), list):
            self.family_failure_streaks = {
                (r, f, s): int(i["count"])
                for i in raw_streaks if isinstance(i, Mapping)
                and (r := str(i.get("regime_tag", "")).strip())
                and (f := str(i.get("symptom_family", "")).strip())
                and (s := str(i.get("parent_sha", "")).strip()) and int(i.get("count", 0)) > 0
            }

    @classmethod
    def from_dict(
        cls, payload: Mapping[str, object], *, rules: Iterable[AscendCausalRule] | None = None, seed_path: str | Path | None = None
    ) -> BayesianRuleTracker:
        """Reconstruct a BayesianRuleTracker including posteriors and scoped tabu state."""
        tracker = cls.from_rules(rules) if rules is not None else cls.from_seed_yaml(seed_path)
        tracker.load_posteriors_dict(payload)
        return tracker

    def load_posteriors_json(self, path: str | Path) -> None:
        """Restore tracked posteriors and scoped tabu state from disk for matching rule IDs."""
        if isinstance(payload := json.loads(Path(path).read_text(encoding="utf-8")), Mapping):
            self.load_posteriors_dict(payload)


__all__ = [
    "BayesianRuleTracker", "FamilyTabuEntry", "RulePosterior", "TabuEntry",
    "append_kb_event_shard", "append_rule_misapplication", "append_winning_edge",
    "save_distilled_rule_file", "save_staging_candidate_rule",
]
