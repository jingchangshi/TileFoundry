"""L3/L4 execution, card-pool rotation, pending recovery, and monotonic guardrails."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol, TypeVar

from tilefoundry_ascend.agent.campaign.helpers import (
    ensure_candidate_bytes, lookup_case_dims_safe, persist_runtime_state,
    source_sha256, write_source_and_verify,
)
from tilefoundry_ascend.agent.campaign.promotion_coordinator import PendingPromotion
from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (
    FAILURE_KIND_CANDIDATE, FAILURE_KIND_INFRASTRUCTURE, AnchorEvalMetrics,
    CandidateSurrogateResult, L3AcceptanceResult, L4CannbenchResult,
    run_l3_full_contract_via_agent_gates, run_l4_official_cannbench_via_agent_gates,
)
from tilefoundry_ascend.agent.contracts import NpuExecutionPort
from tilefoundry_ascend.agent.specs.anchor_specs import (
    AnchorCaseSpec, GuardrailCaseSpec, RegimeTargetSpec, get_operator_task,
    lookup_case_dims, lookup_case_metadata, resolve_regime_for_case,
)
from tilefoundry_ascend.agent.specs.fault_types import L3Resolution
from tilefoundry_ascend.ascend.cost_model.experience_lake import append_counterexample_boundary
from tilefoundry_ascend.ascend.json_record import JsonRecordMixin, atomic_write_text
from tilefoundry_ascend.ascend.passes.hir_tir_bridge import HirBlueprintPacket
from tilefoundry_ascend.ascend.sil.schema import TopologyAxes

logger = logging.getLogger(__name__)
_DEVICE_ID_PATTERN = re.compile(r"[0-9]+")
_L3_CASE_NUM_RE = re.compile(r"\bcase=(\d+)\b")
CAMPAIGN_DEVICE_IDS_ENV_KEY = "TILEFOUNDRY_NPU_CAMPAIGN_DEVICE_IDS"

SHAPE_DIM_KEYS: frozenset[str] = frozenset(
    {
        "b", "b_dim", "batch", "s", "s_dim", "s_q", "seq_len", "s_kv", "d", "d_dim", "d_nope",
        "d_rope", "m", "m_dim", "m_rows", "n", "n_dim", "n_depth", "n_q", "n_kv", "k", "k_dim",
        "half_dim", "q_rows", "q_total_rows", "v_rows", "epsilon", "tl_dtype", "dtype",
        "input_layout", "is_causal",
    }
)
NON_TAIL_DIM_KEYS: frozenset[str] = frozenset(
    {
        "total_elements", "rank", "num_inputs", "dtype_bytes", "case_id", "case_num",
        "group_count", "b", "b_dim", "batch", "n", "n_depth", "n_q", "n_kv", "numheads", "numkvheads",
    }
)


class DevicePoolConfig(Protocol):
    """Minimal configuration interface required to resolve an L4 physical-card pool."""
    @property
    def device_id(self) -> str: ...
    @property
    def device_ids(self) -> Sequence[str]: ...
    @property
    def retry_device_id(self) -> str: ...


class ResolvedRegimeConfigLike(Protocol):
    """Structural interface for resolved regime builder and compiler parameters."""
    @property
    def selector_name(self) -> str: ...
    @property
    def builder_kwargs(self) -> Mapping[str, object]: ...
    @property
    def pass_configs(self) -> Mapping[str, object]: ...
    @property
    def compile_flags(self) -> Sequence[str]: ...


class RoundRecordLike(Protocol):
    """Minimal round-record interface supporting dataclass replacement."""
    @property
    def round_idx(self) -> int: ...


_RoundT = TypeVar("_RoundT", bound=RoundRecordLike)


_T = TypeVar("_T")


def _call_with_optional_ctx(fn: Callable[..., _T], module_path: Path, **kwargs: object) -> _T:
    try:
        return fn(module_path, **kwargs)
    except TypeError as exc:
        if "unexpected keyword argument" not in str(exc):
            raise
        return fn(module_path)


def parse_device_ids(raw: str) -> tuple[str, ...]:
    """Parse one comma-separated ``--device-ids`` value into physical card IDs."""
    if not raw.strip():
        return ()
    parts = [part.strip() for part in raw.split(",")]
    if any(not part for part in parts):
        raise ValueError(f"device list contains an empty entry: {raw!r}")
    return tuple(parts)


def canonical_device_id(value: str, *, field: str) -> str:
    """Canonicalize one physical NPU ID exactly as AgentGates records it."""
    text = value.strip()
    if not _DEVICE_ID_PATTERN.fullmatch(text):
        raise ValueError(f"{field} must be a nonnegative decimal device ID, got {value!r}")
    return str(int(text))


def resolve_campaign_device_plan(
    *,
    device_id: str,
    device_ids: tuple[str, ...],
    env_device_ids: tuple[str, ...],
    visible_device_ids: tuple[str, ...] = (),
) -> tuple[str, tuple[str, ...]]:
    """Resolve primary card and ordered pool from CLI and managed env declarations."""
    canonical_cfg = tuple(
        canonical_device_id(v, field=f"configured_device_ids[{i}]")
        for i, v in enumerate(device_ids or env_device_ids)
    )
    explicit_primary = bool(device_id.strip())
    primary = canonical_device_id(device_id, field="device_id") if explicit_primary else (canonical_cfg[0] if canonical_cfg else "1")
    pool = [primary, *(d for d in canonical_cfg if d != primary)]
    if visible_device_ids:
        visible = {canonical_device_id(v, field=f"visible_device_ids[{i}]") for i, v in enumerate(visible_device_ids)}
        if explicit_primary and primary not in visible:
            raise ValueError(f"requested device {primary!r} is not visible (visible: {', '.join(sorted(visible))})")
        narrowed = [d for d in pool if d in visible]
        if not narrowed:
            raise ValueError(f"configured device pool has no card visible to this process (pool: {', '.join(pool)}; visible: {', '.join(sorted(visible))})")
        primary = primary if primary in visible else narrowed[0]
        pool = [primary, *(d for d in narrowed if d != primary)]
    return primary, tuple(pool)


def resolve_l4_device_pool(config: DevicePoolConfig) -> tuple[str, ...]:
    """Return the ordered physical-card pool for whole-L4 attempts."""
    pool = [canonical_device_id(config.device_id, field="device_id")]
    for i, v in enumerate(config.device_ids):
        dev = canonical_device_id(v, field=f"device_ids[{i}]")
        if dev not in pool:
            pool.append(dev)
    if config.retry_device_id.strip():
        rdev = canonical_device_id(config.retry_device_id, field="retry_device_id")
        if rdev not in pool:
            pool.append(rdev)
    return tuple(pool)


def is_occupancy_blocked_env(failure_kind: str, blocked_env_category: str) -> bool:
    """Return True only for a signed occupancy BLOCKED_ENV preemption."""
    return failure_kind == FAILURE_KIND_INFRASTRUCTURE and blocked_env_category == "occupancy"


@dataclass(frozen=True)
class PendingL4(JsonRecordMixin):
    """Official L4 execution state preserved across wait timeouts and restarts."""

    target_path: str = ""
    candidate_relative: str = ""
    candidate_sha256: str = ""
    execution_id: str = ""
    run_id: str = ""
    receipt_path: str = ""
    device_id: str = ""
    state: str = ""
    error_message: str = ""
    failure_kind: str = FAILURE_KIND_INFRASTRUCTURE
    execution_terminal: bool = True
    retry_used: bool = False
    device_pool: tuple[str, ...] = ()
    blocked_env_round_attempted_device_ids: tuple[str, ...] = ()
    blocked_env_rounds_used: int = 0
    blocked_env_round_limit: int | None = None
    blocked_env_category: str = ""
    device_rejected: bool = False
    rejected_device_ids: tuple[str, ...] = ()
    attempts: tuple[dict[str, str], ...] = ()
    prior_stop_reason: str = ""

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> PendingL4:
        """Reconstruct a pending L4 record from a checkpoint mapping."""
        norm = dict(payload)
        if "blocked_env_rounds_used" not in norm and "same_card_retries_used" in norm:
            norm["blocked_env_rounds_used"] = max(0, int(norm.get("same_card_retries_used") or 0))
        if "blocked_env_round_limit" not in norm and norm.get("same_card_retry_limit") is not None:
            raw_lim = norm["same_card_retry_limit"]
            norm["blocked_env_round_limit"] = max(0, int(raw_lim))  # type: ignore[arg-type]
        return super().from_dict(norm)


def promotion_attempt(
    result: L3AcceptanceResult, device_id: str, *, candidate_sha256: str = ""
) -> dict[str, str]:
    """Build one auditable L3 attempt entry with card and candidate identity."""
    return {
        "execution_id": result.execution_id, "run_id": result.run_id, "receipt_path": result.receipt_path,
        "error_message": result.error_message, "failure_kind": result.failure_kind,
        "failure_category": result.failure_category,
        "pending": str(result.pending or result.failure_kind != FAILURE_KIND_CANDIDATE or not result.execution_terminal).lower(),
        "device_id": result.device_id or device_id, "state": result.state,
        "candidate_sha256": result.candidate_sha256 or candidate_sha256,
        "execution_terminal": str(result.execution_terminal).lower(),
    }


def record_promotion_attempts(
    rounds: list[_RoundT],
    pending: PendingPromotion,
    attempts: tuple[dict[str, str], ...],
    *,
    promoted: bool,
    l3_resolution: str,
    stagnation_after: int,
) -> None:
    """Update the matching round record with promotion resolution and attempts."""
    for idx, record in enumerate(rounds):
        if record.round_idx == pending.round_idx:
            rounds[idx] = replace(
                record, promoted=promoted, l3_resolution=l3_resolution,
                stagnation_rounds_after=stagnation_after, promotion_attempts=attempts,
            )
            break


def l4_attempt(result: L4CannbenchResult) -> dict[str, str]:
    """Build one auditable L4 attempt entry with card and coverage fields."""
    return {
        "execution_id": result.execution_id, "run_id": result.run_id, "receipt_path": result.receipt_path,
        "error_message": result.error_message, "failure_kind": result.failure_kind, "device_id": result.device_id,
        "state": result.state, "execution_terminal": str(result.execution_terminal).lower(),
        "blocked_env_category": result.blocked_env_category, "device_rejected": str(result.device_rejected).lower(),
        "official_report_valid": str(result.official_report_valid).lower(),
        "all_cases_passed": str(result.all_cases_passed).lower(),
        "all_timings_valid": str(result.all_timings_valid).lower(),
        "case_count": str(result.case_count), "passed_cases": str(result.passed_cases),
    }


def l4_round_attempted_devices(pending: PendingL4) -> tuple[str, ...]:
    """Return cards already tried in the current blocked-env round."""
    return pending.blocked_env_round_attempted_device_ids or ((pending.device_id,) if pending.device_id else ())


def merge_l4_identity(result: L4CannbenchResult, pending: PendingL4) -> L4CannbenchResult:
    """Merge saved pending L4 identity fields onto an L4 result."""
    return replace(
        result, target_path=result.target_path or pending.target_path,
        candidate_sha256=result.candidate_sha256 or pending.candidate_sha256,
        device_id=result.device_id or pending.device_id,
        blocked_env_category=result.blocked_env_category or pending.blocked_env_category,
        attempts=pending.attempts or result.attempts,
    )


def pending_l4_after(
    pending: PendingL4, result: L4CannbenchResult, attempts: tuple[dict[str, str], ...], *, retry_used: bool
) -> PendingL4:
    """Advance PendingL4 state after an L4 attempt outcome."""
    rej = pending.rejected_device_ids
    if result.device_rejected and result.device_id and result.device_id not in rej:
        rej = (*rej, result.device_id)
    att = pending.blocked_env_round_attempted_device_ids
    if (
        result.device_id
        and (result.device_rejected or is_occupancy_blocked_env(result.failure_kind, result.blocked_env_category))
        and result.device_id not in att
    ):
        att = (*att, result.device_id)
    return replace(
        pending, target_path=result.target_path or pending.target_path,
        candidate_sha256=result.candidate_sha256 or pending.candidate_sha256,
        execution_id=result.execution_id or pending.execution_id, run_id=result.run_id or pending.run_id,
        receipt_path=result.receipt_path or pending.receipt_path, device_id=result.device_id or pending.device_id,
        state=result.state or pending.state, error_message=result.error_message or pending.error_message,
        failure_kind=result.failure_kind or pending.failure_kind, execution_terminal=result.execution_terminal,
        retry_used=retry_used, blocked_env_category=result.blocked_env_category,
        device_rejected=result.device_rejected, rejected_device_ids=rej,
        blocked_env_round_attempted_device_ids=att, attempts=attempts,
    )


def is_blocked_env_l4(result: L4CannbenchResult) -> bool:
    """Return True only for an occupancy-classified BLOCKED_ENV receipt."""
    return is_occupancy_blocked_env(result.failure_kind, result.blocked_env_category)


def is_rejected_card_l4(result: L4CannbenchResult) -> bool:
    """Return True when AgentGates refused the submitted card as unauthorized."""
    return result.device_rejected and result.failure_kind == FAILURE_KIND_INFRASTRUCTURE


def extract_l3_failed_case_nums(l3_res: L3AcceptanceResult) -> list[int]:
    """Extract deduplicated integer case numbers from an L3 failure result."""
    case_nums: list[int] = []
    seen: set[int] = set()
    for item in getattr(l3_res, "failed_case_nums", ()) or ():
        if isinstance(item, int) and not isinstance(item, bool) and int(item) not in seen:
            seen.add(int(item))
            case_nums.append(int(item))
    if not case_nums and l3_res.error_message:
        for m in _L3_CASE_NUM_RE.finditer(l3_res.error_message):
            cnum = int(m.group(1))
            if cnum not in seen:
                seen.add(cnum)
                case_nums.append(cnum)
    return case_nums


def extract_knob_signature(
    mod_src: str,
    regime_spec: RegimeTargetSpec,
    *,
    resolve_regime_config_fn: Callable[[str, RegimeTargetSpec, Mapping[str, object] | None], ResolvedRegimeConfigLike] | None = None,
) -> str:
    """Extract canonical sorted knob signature for deterministic boundary ledger."""
    if not mod_src.strip() or resolve_regime_config_fn is None:
        return ""
    anchor_dims = regime_spec.representative_anchor_dims or regime_spec.representative_hir_dims
    try:
        resolved = resolve_regime_config_fn(mod_src, regime_spec, anchor_dims)
    except (SyntaxError, AttributeError, ImportError, KeyError, NameError, OSError, RuntimeError, TypeError, ValueError):
        return ""
    parts: list[str] = []
    if resolved.selector_name:
        parts.extend(f"{k}={resolved.builder_kwargs[k]!r}" for k in sorted(resolved.builder_kwargs) if str(k).lower() not in SHAPE_DIM_KEYS)
    parts.extend(f"pc.{k}={resolved.pass_configs[k]!r}" for k in sorted(resolved.pass_configs))
    if resolved.compile_flags:
        parts.append(f"flags={' '.join(sorted(str(f) for f in resolved.compile_flags))}")
    return ",".join(parts)


def is_prime_gt2(value: int) -> bool:
    """Return True when value is an odd prime integer greater than 2."""
    if value < 3 or value % 2 == 0:
        return False
    d = 3
    while d * d <= value:
        if value % d == 0:
            return False
        d += 2
    return True


def case_dim_alignment_flags(case_dims: Mapping[str, object]) -> dict[str, bool]:
    """Compute odd, prime, and non-32B-aligned flags for counterexample dimensions."""
    tails = [
        int(v) for k, v in case_dims.items()
        if str(k).strip().lower() not in NON_TAIL_DIM_KEYS and isinstance(v, int) and not isinstance(v, bool) and v > 1
    ]
    elem_bytes = 4 if "32" in str(case_dims.get("tl_dtype") or case_dims.get("dtype") or "float16").lower() else 2
    has_odd = any(v % 2 == 1 for v in tails)
    has_prime = any(is_prime_gt2(v) for v in tails)
    non_32b = any((v * elem_bytes) % 32 != 0 for v in tails)
    is_unaligned = has_odd or has_prime or non_32b or any(v % 32 != 0 for v in tails)
    return {
        "has_odd_dim": has_odd, "has_prime_dim": has_prime, "non_32b_aligned": non_32b,
        "odd_or_prime_tail": bool(has_odd or has_prime), "has_unaligned_dim": is_unaligned, "is_unaligned": is_unaligned,
    }


@dataclass(frozen=True)
class _DevicePoolAdapter:
    device_id: str
    device_ids: tuple[str, ...]
    retry_device_id: str


class L3L4RecoveryCoordinator:
    """Coordinate L3/L4 execution, pending recovery, card rotation, and guardrails."""

    promotion_attempt = staticmethod(promotion_attempt)
    record_promotion_attempts = staticmethod(record_promotion_attempts)
    l4_attempt = staticmethod(l4_attempt)
    l4_round_attempted_devices = staticmethod(l4_round_attempted_devices)
    merge_l4_identity = staticmethod(merge_l4_identity)
    pending_l4_after = staticmethod(pending_l4_after)
    extract_l3_failed_case_nums = staticmethod(extract_l3_failed_case_nums)
    is_prime_gt2 = staticmethod(is_prime_gt2)
    case_dim_alignment_flags = staticmethod(case_dim_alignment_flags)
    is_blocked_env_l4 = staticmethod(is_blocked_env_l4)
    is_rejected_card_l4 = staticmethod(is_rejected_card_l4)

    def __init__(
        self, *, operator_name: str, target_module_path: Path, source_root: Path,
        device_id: str = "1", retry_device_id: str = "", device_ids: tuple[str, ...] = (),
        l4_device_pool: tuple[str, ...] | None = None, l4_blocked_env_max_retries: int = 2,
        l4_retry_backoff_sec: float = 45.0, env_file: Path | None = None,
        cannbench_repo: Path | None = None, checkpoint_path: Path | None = None,
        experience_lake_dir: Path | None = None, run_id: str = "",
        npu_port: NpuExecutionPort | None = None,
        l3_contract_runner: Callable[..., L3AcceptanceResult] | None = None,
        l4_cannbench_runner: Callable[..., L4CannbenchResult] | None = None,
        l2_batch_runner: Callable[[Sequence[tuple[str, Path]], Sequence[AnchorCaseSpec]], tuple[Mapping[str, CandidateSurrogateResult], Mapping[str, str]]] | None = None,
        lookup_case_dims_fn: Callable[[int, str], dict[str, object]] | None = None,
        resolve_regime_config_fn: Callable[[str, RegimeTargetSpec, Mapping[str, object] | None], ResolvedRegimeConfigLike] | None = None,
        persist_state_fn: Callable[[str, Mapping[str, object] | None], None] | None = None,
        write_and_verify_fn: Callable[[Path, str], str] | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        scoreboard_anchors: Sequence[AnchorCaseSpec] = (),
        regime_specs: Sequence[RegimeTargetSpec] = (),
        regime_blueprints: Mapping[str, HirBlueprintPacket] | None = None,
        topology_axes: TopologyAxes | Mapping[str, object] | None = None,
        guardrail_cases: list[GuardrailCaseSpec] | None = None,
        last_guardrail_metrics_by_cid: Mapping[str, Mapping[int, AnchorEvalMetrics]] | None = None,
        case_dims_cache: dict[int, dict[str, object]] | None = None,
    ) -> None:
        self._operator_name = operator_name
        self._target_module_path = Path(target_module_path).resolve()
        self._source_root = Path(source_root).resolve()
        self._device_id = canonical_device_id(device_id, field="device_id") if device_id.strip() else "1"
        self._retry_device_id = canonical_device_id(retry_device_id, field="retry_device_id") if retry_device_id.strip() else ""
        self._device_ids = tuple(device_ids)
        self._l4_device_pool = tuple(l4_device_pool) if l4_device_pool is not None else resolve_l4_device_pool(_DevicePoolAdapter(self._device_id, self._device_ids, self._retry_device_id))
        self._l4_blocked_env_max_retries, self._l4_retry_backoff_sec = max(0, int(l4_blocked_env_max_retries)), float(l4_retry_backoff_sec)
        self._env_file = Path(env_file).resolve() if env_file is not None else None
        self._cannbench_repo = Path(cannbench_repo).resolve() if cannbench_repo is not None else None
        self._checkpoint_path = Path(checkpoint_path).resolve() if checkpoint_path is not None else None
        self._experience_lake_dir = Path(experience_lake_dir).resolve() if experience_lake_dir is not None else None
        self._run_id, self._npu_port = run_id, npu_port
        self._l3_contract_runner, self._l4_cannbench_runner, self._l2_batch_runner = l3_contract_runner, l4_cannbench_runner, l2_batch_runner
        self._lookup_case_dims_fn, self._resolve_regime_config_fn = lookup_case_dims_fn, resolve_regime_config_fn
        self._persist_state_fn, self._write_and_verify_fn, self._sleep_fn = persist_state_fn, write_and_verify_fn, sleep_fn
        self.scoreboard_anchors: tuple[AnchorCaseSpec, ...] = tuple(scoreboard_anchors)
        self.regime_specs: tuple[RegimeTargetSpec, ...] = tuple(regime_specs)
        self.regime_blueprints: dict[str, HirBlueprintPacket] = dict(regime_blueprints or {})
        self.topology_axes: TopologyAxes | None = (
            topology_axes if isinstance(topology_axes, TopologyAxes)
            else (TopologyAxes.from_dict(topology_axes) if isinstance(topology_axes, Mapping) else None)
        )
        self.guardrail_cases: list[GuardrailCaseSpec] = guardrail_cases if guardrail_cases is not None else []
        self.last_guardrail_metrics_by_cid: Mapping[str, Mapping[int, AnchorEvalMetrics]] = last_guardrail_metrics_by_cid if last_guardrail_metrics_by_cid is not None else {}
        self.case_dims_cache: dict[int, dict[str, object]] = case_dims_cache if case_dims_cache is not None else {}
        self.pending_promotion: PendingPromotion | None = None
        self.active_promotion: PendingPromotion | None = None
        self.pending_l4: PendingL4 | None = None

    @property
    def l4_device_pool(self) -> tuple[str, ...]:
        """Return ordered authorized physical-card pool for L3/L4 execution."""
        return self._l4_device_pool

    def _persist_runtime_state(self, key: str, value: Mapping[str, object] | None) -> None:
        if self._persist_state_fn is not None:
            self._persist_state_fn(key, value)
        else:
            persist_runtime_state(self._checkpoint_path, key, value)

    def _write_source_and_verify(
        self, path: Path, source: str, *, expected_sha256: str | None = None
    ) -> str:
        if self._write_and_verify_fn is not None:
            return self._write_and_verify_fn(path, source)
        return write_source_and_verify(path, source, expected_sha256=expected_sha256)

    def run_l3_contract(
        self, module_path: Path, *, device_id: str | None = None,
        on_execution_started: Callable[..., None] | None = None,
    ) -> L3AcceptanceResult:
        """Execute a fresh L3 full-contract run on the requested device."""
        req_dev = str(device_id or self._device_id)
        if self._npu_port is not None:
            return self._npu_port.run_l3_contract(module_path, device_id=req_dev, on_execution_started=on_execution_started)
        if self._l3_contract_runner is not None:
            return _call_with_optional_ctx(self._l3_contract_runner, module_path, device_id=req_dev, on_execution_started=on_execution_started)
        return run_l3_full_contract_via_agent_gates(
            operator_name=self._operator_name, candidate_module_path=module_path,
            source_root=self._source_root, env_file=self._env_file,
            device_id=req_dev, device_pool=self._l4_device_pool, on_execution_started=on_execution_started,
        )

    def resume_l3_contract(
        self, module_path: Path, pending: PendingPromotion, *,
        on_execution_started: Callable[..., None] | None = None,
    ) -> L3AcceptanceResult:
        """Continue a saved L3 execution instead of submitting a duplicate."""
        req_dev = pending.device_id or self._device_id
        if self._npu_port is not None:
            return self._npu_port.run_l3_contract(
                module_path, device_id=req_dev, resume_execution_id=pending.execution_id,
                resume_run_id=pending.run_id, on_execution_started=on_execution_started,
            )
        if self._l3_contract_runner is not None:
            return _call_with_optional_ctx(
                self._l3_contract_runner, module_path, device_id=req_dev,
                resume_execution_id=pending.execution_id, resume_run_id=pending.run_id,
                on_execution_started=on_execution_started,
            )
        return run_l3_full_contract_via_agent_gates(
            operator_name=self._operator_name, candidate_module_path=module_path,
            source_root=self._source_root, env_file=self._env_file, device_id=req_dev,
            resume_execution_id=pending.execution_id, resume_run_id=pending.run_id, on_execution_started=on_execution_started,
        )

    def next_l3_alt_card(self, pending: PendingPromotion, attempted_device: str) -> str:
        """Select the next untried authorized pool card for an L3 retry."""
        tried = {str(it.get("device_id", "")).strip() for it in pending.attempts if str(it.get("device_id", "")).strip()}
        for d in (attempted_device, pending.device_id):
            if d.strip():
                tried.add(d.strip())
        untried = [dev for dev in self._l4_device_pool if dev not in tried]
        if not untried:
            return ""
        return untried[1] if len(untried) > 1 and untried[0] == self._l4_device_pool[0] else untried[0]

    def can_retry_on_alt_card(self, pending: PendingPromotion, attempted_device: str) -> bool:
        """Return True when an untried authorized card exists for L3 retry."""
        return bool(self.next_l3_alt_card(pending, attempted_device))

    def retry_l3_on_alt_card(
        self, pending: PendingPromotion, attempts: tuple[dict[str, str], ...]
    ) -> L3AcceptanceResult:
        """Start one fresh L3 execution on the next untried authorized pool card."""
        retry_dev = self.next_l3_alt_card(pending, pending.device_id)
        if not retry_dev:
            raise RuntimeError("no untried alternate device available for L3 retry")
        retry_pending = replace(
            pending, execution_id="", run_id="", receipt_path="", error_message="",
            attempts=attempts, device_id=retry_dev, execution_terminal=False, retry_used=True,
        )

        def _started(execution_id: str, device: str) -> None:
            started = replace(retry_pending, execution_id=execution_id, device_id=device, execution_terminal=False)
            self.active_promotion = self.pending_promotion = started
            self._persist_runtime_state("active_promotion", started.to_dict())

        res = self.run_l3_contract(self._target_module_path, device_id=retry_dev, on_execution_started=_started)
        self.active_promotion = None
        return res

    def _regime_spec_for_tag(self, regime_tag: str) -> RegimeTargetSpec:
        return next(
            (s for s in self.regime_specs if s.regime_tag == regime_tag),
            RegimeTargetSpec(regime_tag=regime_tag, builder_candidates=(), sample_builder_kwargs={}),
        )

    def resolve_pending_promotion(
        self, pending: PendingPromotion, *, incumbent_source: str,
        best_result: CandidateSurrogateResult, stagnation_by_regime: dict[str, int],
        rounds: list[_RoundT], promoted_rounds: list[int], rolled_back_rounds: list[int],
        active_anchors: Sequence[AnchorCaseSpec],
    ) -> tuple[str, CandidateSurrogateResult, dict[str, int], PendingPromotion | None, bool]:
        """Resume, retry on an untried card, or retain a recoverable pending L3."""
        _ = active_anchors
        ensure_err = ensure_candidate_bytes(
            path=self._target_module_path, source=pending.candidate_source, expected_sha256=pending.candidate_sha256
        )
        if ensure_err:
            upd = replace(pending, error_message=ensure_err)
            self.pending_promotion = upd
            self.active_promotion = upd if not upd.execution_terminal else None
            return incumbent_source, best_result, stagnation_by_regime, upd, True

        attempts = list(pending.attempts)
        current = pending

        def _keep_pending(res: L3AcceptanceResult, *, terminal: bool):
            nonlocal current
            attempts.append(promotion_attempt(res, current.device_id, candidate_sha256=current.candidate_sha256))
            current = replace(
                current, execution_id=res.execution_id or current.execution_id, run_id=res.run_id or current.run_id,
                receipt_path=res.receipt_path or current.receipt_path, error_message=res.error_message or current.error_message,
                attempts=tuple(attempts), device_id=res.device_id or current.device_id, execution_terminal=terminal,
            )
            if terminal:
                if err := self._write_source_and_verify(self._target_module_path, incumbent_source):
                    raise RuntimeError(err)
                self.active_promotion = None
                self._persist_runtime_state("active_promotion", None)
            else:
                self.active_promotion = current
                self._persist_runtime_state("active_promotion", current.to_dict())
            self.pending_promotion = current
            self._persist_runtime_state("pending_promotion", current.to_dict())
            record_promotion_attempts(
                rounds, current, tuple(attempts), promoted=False,
                l3_resolution=L3Resolution.INFRA_DEFERRED.value,
                stagnation_after=stagnation_by_regime.get(current.regime_tag, 0),
            )
            return incumbent_source, best_result, stagnation_by_regime, current, True

        def _promote():
            record_promotion_attempts(
                rounds, current, tuple(attempts), promoted=True,
                l3_resolution=L3Resolution.PROMOTED.value, stagnation_after=0,
            )
            if current.round_idx in rolled_back_rounds:
                rolled_back_rounds.remove(current.round_idx)
            if current.round_idx not in promoted_rounds:
                promoted_rounds.append(current.round_idx)
            stagnation_by_regime[current.regime_tag] = 0
            self.refresh_guardrail_incumbent_refs(current.candidate_id)
            self.pending_promotion = self.active_promotion = None
            self._persist_runtime_state("active_promotion", None)
            self._persist_runtime_state("pending_promotion", None)
            return current.candidate_source, current.candidate_result, stagnation_by_regime, None, False

        def _candidate_failure(res: L3AcceptanceResult):
            attempts.append(promotion_attempt(res, current.device_id, candidate_sha256=current.candidate_sha256))
            if err := self._write_source_and_verify(self._target_module_path, incumbent_source):
                raise RuntimeError(err)
            self.record_l3_guardrail_cases(res, incumbent_source, self._regime_spec_for_tag(current.regime_tag), topology_axes=self.topology_axes)
            if current.round_idx not in rolled_back_rounds:
                rolled_back_rounds.append(current.round_idx)
            stagnation_by_regime[current.regime_tag] = stagnation_by_regime.get(current.regime_tag, 0) + 1
            record_promotion_attempts(
                rounds, current, tuple(attempts), promoted=False,
                l3_resolution=L3Resolution.CANDIDATE_REJECTED.value,
                stagnation_after=stagnation_by_regime[current.regime_tag],
            )
            self.pending_promotion = self.active_promotion = None
            self._persist_runtime_state("active_promotion", None)
            self._persist_runtime_state("pending_promotion", None)
            return incumbent_source, best_result, stagnation_by_regime, None, False

        def _retry_alt_cards(start_res: L3AcceptanceResult | None, last_dev: str):
            nonlocal current
            tried_alt: set[str] = set()
            latest = start_res
            while self.can_retry_on_alt_card(current, last_dev):
                tdev = self.next_l3_alt_card(current, last_dev)
                if not tdev or tdev in tried_alt:
                    break
                tried_alt.add(tdev)
                rres = self.retry_l3_on_alt_card(current, tuple(attempts))
                attempts.append(promotion_attempt(rres, tdev or current.device_id, candidate_sha256=current.candidate_sha256))
                current = replace(
                    current, execution_id=rres.execution_id or current.execution_id, run_id=rres.run_id or current.run_id,
                    receipt_path=rres.receipt_path or current.receipt_path, error_message=rres.error_message or current.error_message,
                    attempts=tuple(attempts), device_id=tdev or rres.device_id or current.device_id,
                    execution_terminal=rres.execution_terminal, retry_used=True,
                )
                if rres.passed:
                    return _promote()
                if not rres.execution_terminal:
                    return _keep_pending(rres, terminal=False)
                if rres.failure_kind == FAILURE_KIND_CANDIDATE:
                    return _candidate_failure(rres)
                latest, last_dev = rres, (rres.device_id or tdev)
            if latest is not None:
                return _keep_pending(latest, terminal=latest.execution_terminal)
            self.pending_promotion = pending
            self.active_promotion = pending if not pending.execution_terminal else None
            return incumbent_source, best_result, stagnation_by_regime, pending, True

        if pending.execution_id and (not pending.execution_terminal or pending.run_id):
            res = self.resume_l3_contract(self._target_module_path, pending)
            attempts.append(promotion_attempt(res, pending.device_id, candidate_sha256=pending.candidate_sha256))
            current = replace(
                pending, execution_id=res.execution_id or pending.execution_id, run_id=res.run_id or pending.run_id,
                receipt_path=res.receipt_path or pending.receipt_path, error_message=res.error_message or pending.error_message,
                attempts=tuple(attempts), device_id=res.device_id or pending.device_id, execution_terminal=res.execution_terminal,
            )
            if res.passed:
                return _promote()
            if not res.execution_terminal:
                return _keep_pending(res, terminal=False)
            if res.failure_kind == FAILURE_KIND_CANDIDATE:
                return _candidate_failure(res)
            if self.can_retry_on_alt_card(current, res.device_id):
                return _retry_alt_cards(res, res.device_id)
            return _keep_pending(res, terminal=True)

        if self.can_retry_on_alt_card(current, current.device_id):
            return _retry_alt_cards(None, current.device_id)
        self.pending_promotion = pending
        self.active_promotion = pending if not pending.execution_terminal else None
        return incumbent_source, best_result, stagnation_by_regime, pending, True

    def run_l4_cannbench(
        self,
        module_path: Path,
        *,
        resume_execution_id: str = "",
        resume_run_id: str = "",
        device_id: str | None = None,
        on_execution_started: Callable[..., None] | None = None,
    ) -> L4CannbenchResult:
        """Execute or resume an official L4 CANNBench run."""
        req_dev = str(device_id or self._device_id)
        if self._npu_port is not None:
            return self._npu_port.run_l4_cannbench(
                module_path, device_id=req_dev, resume_execution_id=resume_execution_id,
                resume_run_id=resume_run_id, on_execution_started=on_execution_started,
            )
        if self._l4_cannbench_runner is not None:
            return _call_with_optional_ctx(
                self._l4_cannbench_runner, module_path, device_id=req_dev,
                resume_execution_id=resume_execution_id, resume_run_id=resume_run_id,
                on_execution_started=on_execution_started,
            )
        return run_l4_official_cannbench_via_agent_gates(
            operator_name=self._operator_name, candidate_module_path=module_path,
            source_root=self._source_root, env_file=self._env_file, device_id=req_dev,
            resume_execution_id=resume_execution_id, resume_run_id=resume_run_id,
            on_execution_started=on_execution_started,
        )

    def can_retry_l4_on_alt_card(self, pending: PendingL4, attempted_device: str) -> bool:
        """Return True when the legacy alternate retry card is configured and untried."""
        rdev, att = self._retry_device_id, (attempted_device or pending.device_id)
        return bool(rdev and not pending.retry_used and rdev != att and rdev not in l4_round_attempted_devices(pending))

    def effective_l4_round_limit(self, pending: PendingL4) -> int:
        """Return the stored quota, or the current config limit for legacy records."""
        return max(0, pending.blocked_env_round_limit if pending.blocked_env_round_limit is not None else self._l4_blocked_env_max_retries)

    def effective_l4_pool(self, pending: PendingL4) -> tuple[str, ...]:
        """Intersect a saved pool with the pool authorized by the current config."""
        if not pending.device_pool:
            return self._l4_device_pool
        allowed = set(self._l4_device_pool)
        return tuple(d for d in pending.device_pool if d in allowed) or self._l4_device_pool

    def _l4_record_started_cb(self, current_ref: list[PendingL4]) -> Callable[[str, str], None]:
        def _record_started(execution_id: str, device: str) -> None:
            upd = replace(
                current_ref[0], execution_id=execution_id or current_ref[0].execution_id,
                device_id=device or current_ref[0].device_id, state="SUBMITTED", execution_terminal=False,
            )
            current_ref[0] = upd
            self.pending_l4 = upd
            self._persist_runtime_state("pending_l4", upd.to_dict())
        return _record_started

    def advance_l4_card_failure(
        self, pending: PendingL4, result: L4CannbenchResult, attempts: list[dict[str, str]]
    ) -> tuple[L4CannbenchResult, PendingL4]:
        """Advance a terminal occupancy or refused-card L4 run across the card pool."""
        pool, limit = self.effective_l4_pool(pending), self.effective_l4_round_limit(pending)
        cur_ref = [
            replace(
                pending_l4_after(pending, result, tuple(attempts), retry_used=pending.retry_used),
                device_pool=pool, blocked_env_round_limit=limit,
            )
        ]
        while result.execution_terminal and (
            is_occupancy_blocked_env(result.failure_kind, result.blocked_env_category) or is_rejected_card_l4(result)
        ):
            cur = cur_ref[0]
            rejected = is_rejected_card_l4(result)
            attempted = l4_round_attempted_devices(cur)
            eligible = tuple(d for d in pool if d not in cur.rejected_device_ids)
            untried = tuple(d for d in eligible if d not in attempted)
            if untried:
                next_dev, next_rounds, next_att, delay = untried[0], cur.blocked_env_rounds_used, (*attempted, untried[0]), 0.0
            elif rejected or not eligible or cur.blocked_env_rounds_used >= limit:
                break
            else:
                next_rounds = cur.blocked_env_rounds_used + 1
                next_dev, next_att = eligible[0], (eligible[0],)
                delay = max(0.0, self._l4_retry_backoff_sec) * next_rounds
            cur_ref[0] = replace(
                cur, execution_id="", run_id="", receipt_path="", device_id=next_dev, state="SUBMITTED",
                error_message=result.error_message, failure_kind=FAILURE_KIND_INFRASTRUCTURE,
                execution_terminal=False, blocked_env_rounds_used=next_rounds,
                blocked_env_round_attempted_device_ids=next_att, blocked_env_category="", device_rejected=False,
            )
            self.pending_l4 = cur_ref[0]
            self._persist_runtime_state("pending_l4", cur_ref[0].to_dict())
            if delay > 0.0:
                self._sleep_fn(delay)
            result = self.run_l4_cannbench(
                self._target_module_path, device_id=next_dev, on_execution_started=self._l4_record_started_cb(cur_ref)
            )
            attempts.append(l4_attempt(result))
            cur_ref[0] = pending_l4_after(cur_ref[0], result, tuple(attempts), retry_used=cur_ref[0].retry_used)
            self.pending_l4 = cur_ref[0]
            self._persist_runtime_state("pending_l4", cur_ref[0].to_dict())
        return result, cur_ref[0]

    def resolve_pending_l4(
        self, pending: PendingL4, *, incumbent_source: str
    ) -> tuple[L4CannbenchResult, PendingL4 | None]:
        """Continue the saved L4 execution, or move the complete run to another card."""
        expected = pending.candidate_sha256 or source_sha256(incumbent_source)
        digest_error = ""
        try:
            cur_digest = hashlib.sha256(self._target_module_path.read_bytes()).hexdigest()
        except OSError as exc:
            digest_error = f"pending L4 candidate {self._target_module_path} cannot be read: {exc}"
        else:
            if cur_digest != expected:
                digest_error = (
                    f"pending L4 candidate {self._target_module_path} digest {cur_digest} "
                    f"does not match submitted {expected}; refusing to resume or replace it"
                )
        if digest_error:
            upd = replace(pending, candidate_sha256=expected, error_message=digest_error)
            self.pending_l4 = upd
            return L4CannbenchResult(
                passed=False, error_message=digest_error, failure_kind=FAILURE_KIND_INFRASTRUCTURE,
                execution_id=pending.execution_id, run_id=pending.run_id, receipt_path=pending.receipt_path,
                device_id=pending.device_id, state=pending.state, execution_terminal=pending.execution_terminal,
                target_path=pending.target_path, candidate_sha256=expected, attempts=pending.attempts,
            ), upd

        attempts = list(pending.attempts)
        eff_pool, eff_limit = self.effective_l4_pool(pending), self.effective_l4_round_limit(pending)
        cur_ref = [replace(pending, device_pool=eff_pool, blocked_env_round_limit=eff_limit)]
        started_cb = self._l4_record_started_cb(cur_ref)

        def _keep(res: L4CannbenchResult, *, retry_used: bool):
            upd = pending_l4_after(cur_ref[0], res, tuple(attempts), retry_used=retry_used)
            self.pending_l4 = upd
            self._persist_runtime_state("pending_l4", upd.to_dict())
            return merge_l4_identity(res, upd), upd

        def _finish(res: L4CannbenchResult):
            self.pending_l4 = None
            self._persist_runtime_state("pending_l4", None)
            return merge_l4_identity(res, cur_ref[0]), None

        def _retry_whole_l4():
            rdev = self._retry_device_id
            if not rdev:
                raise RuntimeError("retry_device_id is empty during whole-L4 retry")
            cur_ref[0] = replace(
                cur_ref[0], execution_id="", run_id="", receipt_path="", device_id=rdev, state="SUBMITTED",
                error_message="", failure_kind=FAILURE_KIND_INFRASTRUCTURE, execution_terminal=False,
                retry_used=True, blocked_env_category="", device_rejected=False,
            )
            rres = self.run_l4_cannbench(self._target_module_path, device_id=rdev, on_execution_started=started_cb)
            attempts.append(l4_attempt(rres))
            cur_ref[0] = replace(
                cur_ref[0], execution_id=rres.execution_id or cur_ref[0].execution_id,
                run_id=rres.run_id or cur_ref[0].run_id, receipt_path=rres.receipt_path or cur_ref[0].receipt_path,
                device_id=rres.device_id or cur_ref[0].device_id, state=rres.state or cur_ref[0].state,
                error_message=rres.error_message or cur_ref[0].error_message,
                failure_kind=rres.failure_kind or cur_ref[0].failure_kind,
                execution_terminal=rres.execution_terminal, attempts=tuple(attempts),
            )
            return _finish(rres) if (rres.passed or rres.failure_kind == FAILURE_KIND_CANDIDATE) else _keep(rres, retry_used=True)

        def _after_infra(res: L4CannbenchResult):
            if is_occupancy_blocked_env(res.failure_kind, res.blocked_env_category) or is_rejected_card_l4(res):
                res, cur_ref[0] = self.advance_l4_card_failure(cur_ref[0], res, attempts)
            if res.passed or res.failure_kind == FAILURE_KIND_CANDIDATE:
                return _finish(res)
            if not res.execution_terminal:
                return _keep(res, retry_used=cur_ref[0].retry_used)
            return _retry_whole_l4() if self.can_retry_l4_on_alt_card(cur_ref[0], res.device_id) else _keep(res, retry_used=cur_ref[0].retry_used)

        if pending.execution_id and (not pending.execution_terminal or pending.run_id):
            res = self.run_l4_cannbench(
                self._target_module_path, resume_execution_id=pending.execution_id,
                resume_run_id=pending.run_id, device_id=pending.device_id or self._device_id,
                on_execution_started=started_cb,
            )
            attempts.append(l4_attempt(res))
            cur_ref[0] = pending_l4_after(cur_ref[0], res, tuple(attempts), retry_used=cur_ref[0].retry_used)
            self.pending_l4 = cur_ref[0]
            self._persist_runtime_state("pending_l4", cur_ref[0].to_dict())
            return _after_infra(res)

        blocked = is_occupancy_blocked_env(pending.failure_kind, pending.blocked_env_category)
        rejected = pending.device_rejected and pending.failure_kind == FAILURE_KIND_INFRASTRUCTURE
        interrupted = not pending.execution_terminal and not pending.execution_id and not pending.run_id and pending.state == "SUBMITTED"
        if (blocked and pending.execution_terminal) or rejected or (interrupted and cur_ref[0].blocked_env_rounds_used <= eff_limit):
            if interrupted:
                idev = cur_ref[0].device_id if cur_ref[0].device_id in eff_pool else eff_pool[0]
                cur_ref[0] = replace(cur_ref[0], device_id=idev)
                res = self.run_l4_cannbench(self._target_module_path, device_id=idev, on_execution_started=started_cb)
                attempts.append(l4_attempt(res))
                cur_ref[0] = pending_l4_after(cur_ref[0], res, tuple(attempts), retry_used=cur_ref[0].retry_used)
                self.pending_l4 = cur_ref[0]
                self._persist_runtime_state("pending_l4", cur_ref[0].to_dict())
            else:
                res = L4CannbenchResult(
                    passed=False, error_message=pending.error_message, failure_kind=pending.failure_kind,
                    execution_id=pending.execution_id, run_id=pending.run_id, receipt_path=pending.receipt_path,
                    device_id=pending.device_id, state=pending.state, execution_terminal=pending.execution_terminal,
                    blocked_env_category=pending.blocked_env_category, device_rejected=pending.device_rejected,
                    target_path=pending.target_path, candidate_sha256=expected, attempts=pending.attempts,
                )
            return _after_infra(res)

        if self.can_retry_l4_on_alt_card(pending, pending.device_id):
            return _retry_whole_l4()
        self.pending_l4 = pending
        return L4CannbenchResult(
            passed=False,
            error_message=pending.error_message or "L4 infrastructure failure has no resumable execution and no other authorized card is configured",
            failure_kind=FAILURE_KIND_INFRASTRUCTURE, execution_id=pending.execution_id, run_id=pending.run_id,
            receipt_path=pending.receipt_path, device_id=pending.device_id, state=pending.state,
            execution_terminal=pending.execution_terminal, blocked_env_category=pending.blocked_env_category,
            device_rejected=pending.device_rejected, target_path=pending.target_path,
            candidate_sha256=expected, attempts=pending.attempts,
        ), pending

    def l4_candidate_relative(self) -> str:
        """Return target module path relative to source root, or filename fallback."""
        try:
            return self._target_module_path.resolve().relative_to(self._source_root.resolve()).as_posix()
        except (OSError, ValueError):
            return self._target_module_path.name

    def run_final_l4_tracked(self, *, incumbent_source: str, prior_stop_reason: str) -> L4CannbenchResult:
        """Submit the official L4 run and persist its identity before waiting."""
        pool = self._l4_device_pool
        attempts: list[dict[str, str]] = []
        cur_ref = [
            PendingL4(
                target_path=str(self._target_module_path), candidate_relative=self.l4_candidate_relative(),
                candidate_sha256=source_sha256(incumbent_source), device_id=pool[0], state="SUBMITTED",
                failure_kind=FAILURE_KIND_INFRASTRUCTURE, execution_terminal=False, device_pool=pool,
                blocked_env_round_attempted_device_ids=(pool[0],),
                blocked_env_round_limit=max(0, self._l4_blocked_env_max_retries),
                prior_stop_reason=prior_stop_reason,
            )
        ]
        res = self.run_l4_cannbench(
            self._target_module_path, device_id=pool[0], on_execution_started=self._l4_record_started_cb(cur_ref)
        )
        attempts.append(l4_attempt(res))
        cur_ref[0] = pending_l4_after(cur_ref[0], res, tuple(attempts), retry_used=False)
        if is_blocked_env_l4(res) or is_rejected_card_l4(res):
            res, cur_ref[0] = self.advance_l4_card_failure(cur_ref[0], res, attempts)
        if res.passed or res.failure_kind == FAILURE_KIND_CANDIDATE:
            self.pending_l4 = None
            self._persist_runtime_state("pending_l4", None)
        else:
            self.pending_l4 = cur_ref[0]
            self._persist_runtime_state("pending_l4", cur_ref[0].to_dict())
        return merge_l4_identity(res, cur_ref[0])

    def extract_knob_signature(self, mod_src: str, regime_spec: RegimeTargetSpec) -> str:
        """Extract canonical sorted knob signature for deterministic boundary ledger."""
        return extract_knob_signature(mod_src, regime_spec, resolve_regime_config_fn=self._resolve_regime_config_fn)

    def _lookup_case_dims_safe(self, case_num: int, regime_tag: str = "") -> dict[str, object]:
        if self._lookup_case_dims_fn is not None:
            cnum = int(case_num)
            if cnum in self.case_dims_cache:
                return dict(self.case_dims_cache[cnum])
            try:
                dims = dict(lookup_case_dims(self._operator_name, cnum, cannbench_repo=self._cannbench_repo, env_file=self._env_file))
            except (FileNotFoundError, OSError, KeyError, ValueError, TypeError):
                dims = dict(self._lookup_case_dims_fn(cnum, regime_tag))
            self.case_dims_cache[cnum] = dict(dims)
            return dims
        return lookup_case_dims_safe(
            self._operator_name, case_num, regime_tag=regime_tag, regime_specs=self.regime_specs,
            cache=self.case_dims_cache, cannbench_repo=self._cannbench_repo, env_file=self._env_file,
        )

    def evaluate_incumbent_on_new_anchors(
        self, incumbent_source: str, new_specs: Sequence[AnchorCaseSpec]
    ) -> list[AnchorEvalMetrics]:
        """Measure incumbent AnchorEvalMetrics on newly added probe cases."""
        if not new_specs or not incumbent_source.strip() or self._l2_batch_runner is None:
            return []
        try:
            with tempfile.TemporaryDirectory(prefix="tf_inc_dyn_anchor_") as tmp_dir:
                inc_file = Path(tmp_dir) / "incumbent.py"
                inc_file.write_text(incumbent_source, encoding="utf-8")
                l2_probe, _ = self._l2_batch_runner([("incumbent", inc_file)], list(new_specs))
            inc_res = l2_probe.get("incumbent")
            return list(inc_res.anchors) if inc_res is not None and inc_res.anchors else []
        except (OSError, subprocess.SubprocessError, TimeoutError, json.JSONDecodeError, RuntimeError, FileNotFoundError, KeyError, ValueError):
            return []

    def record_l3_guardrail_cases(
        self, l3_res: L3AcceptanceResult, incumbent_source: str,
        regime_spec: RegimeTargetSpec, *, topology_axes: TopologyAxes | None = None,
    ) -> list[GuardrailCaseSpec]:
        """Record all unseen non-scoreboard L3 failure probe cases monotonically."""
        scoreboard_cases = {int(a.case_num) for a in self.scoreboard_anchors}
        existing = {int(g.case_num) for g in self.guardrail_cases}
        targets = [c for c in extract_l3_failed_case_nums(l3_res) if c not in scoreboard_cases and c not in existing]
        if not targets:
            return []
        try:
            task_name = get_operator_task(self._operator_name)
        except (FileNotFoundError, KeyError, ValueError):
            task_name = f"operator/{self._operator_name}"
        bp = self.regime_blueprints.get(regime_spec.regime_tag)
        default_hw = float(bp.roofline.ideal_roofline_us) if bp and bp.roofline.ideal_roofline_us > 0.0 else 1.0
        eff_axes = topology_axes if topology_axes is not None else self.topology_axes
        axes_dict = eff_axes.to_dict() if eff_axes is not None else {}
        fstage = str(getattr(l3_res, "failure_stage", "") or "")
        probes, pending_specs = [], []
        for cnum in targets:
            dims = self._lookup_case_dims_safe(cnum, regime_spec.regime_tag)
            rtag = regime_spec.regime_tag
            if dims and self.regime_specs:
                try:
                    rtag = resolve_regime_for_case(self._operator_name, dims, self.regime_specs).regime_tag
                except (KeyError, ValueError, TypeError):
                    pass
            ref_us, hw_us = 1.0, default_hw
            try:
                mr, mh = lookup_case_metadata(self._operator_name, cnum, cannbench_repo=self._cannbench_repo, env_file=self._env_file)
                ref_us, hw_us = (float(mr) if mr > 0.0 else ref_us), (float(mh) if mh > 0.0 else hw_us)
            except (FileNotFoundError, OSError, KeyError, ValueError, json.JSONDecodeError):
                pass
            probes.append(AnchorCaseSpec(task=task_name, case_num=cnum, regime_tag=rtag, ref_perf_us=ref_us, t_hw_us=hw_us))
            pending_specs.append((cnum, rtag, ref_us, hw_us))
            if self._experience_lake_dir is not None:
                try:
                    append_counterexample_boundary(
                        self._experience_lake_dir,
                        {
                            "run_id": self._run_id, "operator": self._operator_name, "regime_tag": rtag,
                            "topology_axes": axes_dict, "case_num": int(cnum), "case_dims": dims,
                            "failure_stage": fstage, "ref_perf_us": float(ref_us), "t_hw_us": float(hw_us),
                            **case_dim_alignment_flags(dims),
                        },
                    )
                except (OSError, ValueError) as exc:
                    logger.debug("Appending counterexample boundary skipped: %s", exc)
        inc_by_case = {int(m.case_num): float(m.cand_us) for m in self.evaluate_incumbent_on_new_anchors(incumbent_source, probes) if m.cand_us > 0.0}
        added, seen_gr = [], set(existing)
        for cnum, rtag, ref_us, hw_us in pending_specs:
            if cnum in seen_gr:
                continue
            seen_gr.add(cnum)
            inc_ref = inc_by_case.get(cnum, 0.0) or (ref_us if ref_us > 1.0 else 0.0)
            spec = GuardrailCaseSpec(task=task_name, case_num=cnum, regime_tag=rtag, incumbent_ref_us=inc_ref, t_hw_us=hw_us)
            self.guardrail_cases.append(spec)
            added.append(spec)
        return added

    def refresh_guardrail_incumbent_refs(self, winner_candidate_id: str) -> None:
        """Update guardrail incumbent reference latencies from a newly promoted winner."""
        metrics_by_case = self.last_guardrail_metrics_by_cid.get(winner_candidate_id) if self.guardrail_cases else None
        if not metrics_by_case:
            return
        for idx, guardrail in enumerate(self.guardrail_cases):
            m = metrics_by_case.get(int(guardrail.case_num))
            if m is not None and m.accuracy_passed and m.cand_us > 0.0:
                self.guardrail_cases[idx] = replace(guardrail, incumbent_ref_us=float(m.cand_us))


__all__ = [
    "CAMPAIGN_DEVICE_IDS_ENV_KEY", "DevicePoolConfig", "L3L4RecoveryCoordinator",
    "NON_TAIL_DIM_KEYS", "PendingL4", "ResolvedRegimeConfigLike", "RoundRecordLike",
    "SHAPE_DIM_KEYS", "atomic_write_text", "canonical_device_id",
    "case_dim_alignment_flags", "ensure_candidate_bytes", "extract_knob_signature",
    "extract_l3_failed_case_nums", "is_blocked_env_l4", "is_occupancy_blocked_env",
    "is_prime_gt2", "is_rejected_card_l4", "l4_attempt", "l4_round_attempted_devices",
    "merge_l4_identity", "parse_device_ids", "pending_l4_after", "persist_runtime_state",
    "promotion_attempt", "record_promotion_attempts", "resolve_campaign_device_plan",
    "resolve_l4_device_pool", "source_sha256", "write_source_and_verify",
]
