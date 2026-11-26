"""L3/L4 execution, card-pool rotation, pending recovery, and monotonic guardrails."""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import os
import re
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol, TypeVar

from tilefoundry_ascend.agent.campaign.anchor_specs import (
    AnchorCaseSpec,
    GuardrailCaseSpec,
    RegimeTargetSpec,
    get_operator_task,
    lookup_case_dims,
    lookup_case_metadata,
    resolve_regime_for_case,
)
from tilefoundry_ascend.agent.campaign.fault_types import L3Resolution
from tilefoundry_ascend.agent.campaign.promotion_coordinator import PendingPromotion
from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (
    FAILURE_KIND_CANDIDATE,
    FAILURE_KIND_INFRASTRUCTURE,
    AnchorEvalMetrics,
    CandidateSurrogateResult,
    L3AcceptanceResult,
    L4CannbenchResult,
    run_l3_full_contract_via_agent_gates,
    run_l4_official_cannbench_via_agent_gates,
)
from tilefoundry_ascend.ascend.cost_model.experience_lake import (
    append_counterexample_boundary,
)
from tilefoundry_ascend.ascend.passes.hir_tir_bridge import HirBlueprintPacket
from tilefoundry_ascend.ascend.sil.schema import TopologyAxes

logger = logging.getLogger(__name__)
_T = TypeVar("_T")

_DEVICE_ID_PATTERN = re.compile(r"[0-9]+")
_L3_CASE_NUM_RE = re.compile(r"\bcase=(\d+)\b")
CAMPAIGN_DEVICE_IDS_ENV_KEY = "TILEFOUNDRY_NPU_CAMPAIGN_DEVICE_IDS"

SHAPE_DIM_KEYS: frozenset[str] = frozenset(
    {
        "b",
        "b_dim",
        "batch",
        "s",
        "s_dim",
        "s_q",
        "seq_len",
        "s_kv",
        "d",
        "d_dim",
        "d_nope",
        "d_rope",
        "m",
        "m_dim",
        "m_rows",
        "n",
        "n_dim",
        "n_depth",
        "n_q",
        "n_kv",
        "k",
        "k_dim",
        "half_dim",
        "q_rows",
        "q_total_rows",
        "v_rows",
        "epsilon",
        "tl_dtype",
        "dtype",
        "input_layout",
        "is_causal",
    }
)

NON_TAIL_DIM_KEYS: frozenset[str] = frozenset(
    {
        "total_elements",
        "rank",
        "num_inputs",
        "dtype_bytes",
        "case_id",
        "case_num",
        "group_count",
        "b",
        "b_dim",
        "batch",
        "n",
        "n_depth",
        "n_q",
        "n_kv",
        "numheads",
        "numkvheads",
    }
)


class DevicePoolConfig(Protocol):
    """Minimal configuration interface required to resolve an L4 physical-card pool."""

    @property
    def device_id(self) -> str:
        """Primary physical device ID."""

    @property
    def device_ids(self) -> Sequence[str]:
        """Configured physical device pool IDs."""

    @property
    def retry_device_id(self) -> str:
        """Legacy single alternate retry card ID."""


class ResolvedRegimeConfigLike(Protocol):
    """Structural interface for resolved regime builder and compiler parameters."""

    @property
    def selector_name(self) -> str:
        """Selected config selector function name."""

    @property
    def builder_kwargs(self) -> Mapping[str, object]:
        """Resolved builder keyword arguments."""

    @property
    def pass_configs(self) -> Mapping[str, object]:
        """Resolved compiler pass configurations."""

    @property
    def compile_flags(self) -> Sequence[str]:
        """Resolved compiler flags."""


class RoundRecordLike(Protocol):
    """Minimal round-record interface supporting dataclass replacement."""

    @property
    def round_idx(self) -> int:
        """1-based round index."""


_RoundT = TypeVar("_RoundT", bound=RoundRecordLike)


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
    configured = device_ids or env_device_ids
    canonical_configured = tuple(
        canonical_device_id(value, field=f"configured_device_ids[{index}]")
        for index, value in enumerate(configured)
    )
    explicit_primary = bool(device_id.strip())
    primary = canonical_device_id(device_id, field="device_id") if explicit_primary else ""
    if not primary:
        primary = canonical_configured[0] if canonical_configured else "1"
    pool = [primary]
    for canonical in canonical_configured:
        if canonical not in pool:
            pool.append(canonical)

    if visible_device_ids:
        visible = {
            canonical_device_id(value, field=f"visible_device_ids[{index}]")
            for index, value in enumerate(visible_device_ids)
        }
        if explicit_primary and primary not in visible:
            raise ValueError(
                f"requested device {primary!r} is not visible to this process "
                f"(visible: {', '.join(sorted(visible))})"
            )
        narrowed = [device for device in pool if device in visible]
        if not narrowed:
            raise ValueError(
                "configured device pool has no card visible to this process "
                f"(pool: {', '.join(pool)}; visible: {', '.join(sorted(visible))})"
            )
        primary = primary if primary in visible else narrowed[0]
        pool = [primary, *(device for device in narrowed if device != primary)]
    return primary, tuple(pool)


def resolve_l4_device_pool(config: DevicePoolConfig) -> tuple[str, ...]:
    """Return the ordered physical-card pool for whole-L4 attempts."""
    pool = [canonical_device_id(config.device_id, field="device_id")]
    for index, value in enumerate(config.device_ids):
        device_id = canonical_device_id(value, field=f"device_ids[{index}]")
        if device_id not in pool:
            pool.append(device_id)
    retry_device_id = config.retry_device_id.strip()
    if retry_device_id:
        retry_device = canonical_device_id(retry_device_id, field="retry_device_id")
        if retry_device not in pool:
            pool.append(retry_device)
    return tuple(pool)


def is_occupancy_blocked_env(failure_kind: str, blocked_env_category: str) -> bool:
    """Return True only for a signed occupancy BLOCKED_ENV preemption."""
    return (
        failure_kind == FAILURE_KIND_INFRASTRUCTURE
        and blocked_env_category == "occupancy"
    )


def atomic_write_text(path: Path, content: str) -> None:
    """Atomically write text to a file via temporary file, fsync, and replace."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            tmp_path = Path(fh.name)
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def source_sha256(source: str) -> str:
    """Return the SHA-256 hex digest of UTF-8 source text."""
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def write_source_and_verify(
    path: Path,
    source: str,
    *,
    expected_sha256: str | None = None,
) -> str:
    """Atomically write source and verify restored bytes match expected SHA-256."""
    expected = expected_sha256 or source_sha256(source)
    atomic_write_text(path, source)
    try:
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        return f"restored module {path} cannot be read back: {exc}"
    if actual != expected:
        return f"restored module {path} digest {actual} does not match expected {expected}"
    return ""


def ensure_candidate_bytes(
    *,
    path: Path,
    source: str,
    expected_sha256: str,
) -> str:
    """Restore submitted candidate bytes at path if missing or drifted."""
    expected = expected_sha256 or source_sha256(source)
    try:
        current = hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        current = ""
    if current == expected:
        return ""
    return write_source_and_verify(path, source, expected_sha256=expected)


def persist_runtime_state(
    checkpoint_path: Path | None,
    key: str,
    value: Mapping[str, object] | None,
) -> None:
    """Upsert a recoverable execution record into the checkpoint JSON file."""
    if checkpoint_path is None:
        return
    payload: dict[str, object] = {}
    if checkpoint_path.is_file():
        try:
            loaded = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if isinstance(loaded, Mapping):
                payload = dict(loaded)
        except (OSError, ValueError):
            payload = {}
    payload[key] = dict(value) if value is not None else None
    atomic_write_text(checkpoint_path, json.dumps(payload, indent=2) + "\n")


def invoke_with_optional_kwargs(
    runner: Callable[..., _T],
    *args: object,
    **kwargs: object,
) -> _T:
    """Pass only the optional keywords an injected runner actually accepts."""
    try:
        signature = inspect.signature(runner)
    except (TypeError, ValueError):
        return runner(*args)
    accepts_any = any(
        param.kind is inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )
    accepted = {
        name: value
        for name, value in kwargs.items()
        if accepts_any or name in signature.parameters
    }
    return runner(*args, **accepted)


@dataclass(frozen=True)
class PendingL4:
    """Official L4 execution state preserved across wait timeouts and restarts."""

    target_path: str
    candidate_relative: str
    candidate_sha256: str
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

    def to_dict(self) -> dict[str, object]:
        """Serialize the pending L4 record for the checkpoint and report."""
        return {
            "target_path": self.target_path,
            "candidate_relative": self.candidate_relative,
            "candidate_sha256": self.candidate_sha256,
            "execution_id": self.execution_id,
            "run_id": self.run_id,
            "receipt_path": self.receipt_path,
            "device_id": self.device_id,
            "state": self.state,
            "error_message": self.error_message,
            "failure_kind": self.failure_kind,
            "execution_terminal": self.execution_terminal,
            "retry_used": self.retry_used,
            "device_pool": list(self.device_pool),
            "blocked_env_round_attempted_device_ids": list(
                self.blocked_env_round_attempted_device_ids
            ),
            "blocked_env_rounds_used": self.blocked_env_rounds_used,
            "blocked_env_round_limit": self.blocked_env_round_limit,
            "blocked_env_category": self.blocked_env_category,
            "device_rejected": self.device_rejected,
            "rejected_device_ids": list(self.rejected_device_ids),
            "attempts": [dict(item) for item in self.attempts],
            "prior_stop_reason": self.prior_stop_reason,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> PendingL4:
        """Reconstruct a pending L4 record from a checkpoint mapping."""
        attempts = tuple(
            {str(k): str(v) for k, v in item.items()}
            for item in payload.get("attempts", ())
            if isinstance(item, Mapping)
        )
        return cls(
            target_path=str(payload.get("target_path", "")),
            candidate_relative=str(payload.get("candidate_relative", "")),
            candidate_sha256=str(payload.get("candidate_sha256", "")),
            execution_id=str(payload.get("execution_id", "")),
            run_id=str(payload.get("run_id", "")),
            receipt_path=str(payload.get("receipt_path", "")),
            device_id=str(payload.get("device_id", "")),
            state=str(payload.get("state", "")),
            error_message=str(payload.get("error_message", "")),
            failure_kind=str(payload.get("failure_kind", FAILURE_KIND_INFRASTRUCTURE)),
            execution_terminal=bool(payload.get("execution_terminal", True)),
            retry_used=bool(payload.get("retry_used", False)),
            device_pool=tuple(str(item) for item in payload.get("device_pool", ())),
            blocked_env_round_attempted_device_ids=tuple(
                str(item)
                for item in payload.get(
                    "blocked_env_round_attempted_device_ids",
                    (),
                )
            ),
            blocked_env_rounds_used=max(
                0,
                int(
                    payload.get(
                        "blocked_env_rounds_used",
                        payload.get("same_card_retries_used", 0),
                    )
                    or 0
                ),
            ),
            blocked_env_round_limit=(
                max(0, int(payload["blocked_env_round_limit"]))
                if payload.get("blocked_env_round_limit") is not None
                else (
                    max(0, int(payload["same_card_retry_limit"]))
                    if payload.get("same_card_retry_limit") is not None
                    else None
                )
            ),
            blocked_env_category=str(payload.get("blocked_env_category", "")),
            device_rejected=bool(payload.get("device_rejected", False)),
            rejected_device_ids=tuple(
                str(item) for item in payload.get("rejected_device_ids", ())
            ),
            attempts=attempts,
            prior_stop_reason=str(payload.get("prior_stop_reason", "")),
        )


def promotion_attempt(
    result: L3AcceptanceResult,
    device_id: str,
    *,
    candidate_sha256: str = "",
) -> dict[str, str]:
    """Build one auditable L3 attempt entry with card and candidate identity."""
    return {
        "execution_id": result.execution_id,
        "run_id": result.run_id,
        "receipt_path": result.receipt_path,
        "error_message": result.error_message,
        "failure_kind": result.failure_kind,
        "failure_category": result.failure_category,
        "pending": str(
            result.pending
            or result.failure_kind != FAILURE_KIND_CANDIDATE
            or not result.execution_terminal
        ).lower(),
        "device_id": result.device_id or device_id,
        "state": result.state,
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
                record,
                promoted=promoted,
                l3_resolution=l3_resolution,
                stagnation_rounds_after=stagnation_after,
                promotion_attempts=attempts,
            )
            break


def l4_attempt(result: L4CannbenchResult) -> dict[str, str]:
    """Build one auditable L4 attempt entry with card and coverage fields."""
    return {
        "execution_id": result.execution_id,
        "run_id": result.run_id,
        "receipt_path": result.receipt_path,
        "error_message": result.error_message,
        "failure_kind": result.failure_kind,
        "device_id": result.device_id,
        "state": result.state,
        "execution_terminal": str(result.execution_terminal).lower(),
        "blocked_env_category": result.blocked_env_category,
        "device_rejected": str(result.device_rejected).lower(),
        "official_report_valid": str(result.official_report_valid).lower(),
        "all_cases_passed": str(result.all_cases_passed).lower(),
        "all_timings_valid": str(result.all_timings_valid).lower(),
        "case_count": str(result.case_count),
        "passed_cases": str(result.passed_cases),
    }


def l4_round_attempted_devices(pending: PendingL4) -> tuple[str, ...]:
    """Return cards already tried in the current blocked-env round."""
    if pending.blocked_env_round_attempted_device_ids:
        return pending.blocked_env_round_attempted_device_ids
    return (pending.device_id,) if pending.device_id else ()


def merge_l4_identity(
    result: L4CannbenchResult,
    pending: PendingL4,
) -> L4CannbenchResult:
    """Merge saved pending L4 identity fields onto an L4 result."""
    return replace(
        result,
        target_path=result.target_path or pending.target_path,
        candidate_sha256=result.candidate_sha256 or pending.candidate_sha256,
        device_id=result.device_id or pending.device_id,
        blocked_env_category=result.blocked_env_category or pending.blocked_env_category,
        attempts=pending.attempts or result.attempts,
    )


def pending_l4_after(
    pending: PendingL4,
    result: L4CannbenchResult,
    attempts: tuple[dict[str, str], ...],
    *,
    retry_used: bool,
) -> PendingL4:
    """Advance PendingL4 state after an L4 attempt outcome."""
    rejected_device_ids = pending.rejected_device_ids
    if (
        result.device_rejected
        and result.device_id
        and result.device_id not in rejected_device_ids
    ):
        rejected_device_ids = (*rejected_device_ids, result.device_id)
    attempted_device_ids = pending.blocked_env_round_attempted_device_ids
    if (
        result.device_id
        and (
            result.device_rejected
            or is_occupancy_blocked_env(
                result.failure_kind,
                result.blocked_env_category,
            )
        )
        and result.device_id not in attempted_device_ids
    ):
        attempted_device_ids = (*attempted_device_ids, result.device_id)
    return replace(
        pending,
        target_path=result.target_path or pending.target_path,
        candidate_sha256=result.candidate_sha256 or pending.candidate_sha256,
        execution_id=result.execution_id or pending.execution_id,
        run_id=result.run_id or pending.run_id,
        receipt_path=result.receipt_path or pending.receipt_path,
        device_id=result.device_id or pending.device_id,
        state=result.state or pending.state,
        error_message=result.error_message or pending.error_message,
        failure_kind=result.failure_kind or pending.failure_kind,
        execution_terminal=result.execution_terminal,
        retry_used=retry_used,
        blocked_env_category=result.blocked_env_category,
        device_rejected=result.device_rejected,
        rejected_device_ids=rejected_device_ids,
        blocked_env_round_attempted_device_ids=attempted_device_ids,
        attempts=attempts,
    )


def is_blocked_env_l4(result: L4CannbenchResult) -> bool:
    """Return True only for an occupancy-classified BLOCKED_ENV receipt."""
    return is_occupancy_blocked_env(
        result.failure_kind,
        result.blocked_env_category,
    )


def is_rejected_card_l4(result: L4CannbenchResult) -> bool:
    """Return True when AgentGates refused the submitted card as unauthorized."""
    return (
        result.device_rejected
        and result.failure_kind == FAILURE_KIND_INFRASTRUCTURE
    )


def extract_l3_failed_case_nums(l3_res: L3AcceptanceResult) -> list[int]:
    """Extract deduplicated integer case numbers from an L3 failure result."""
    raw_cases = getattr(l3_res, "failed_case_nums", ()) or ()
    case_nums: list[int] = []
    seen: set[int] = set()
    for item in raw_cases:
        if isinstance(item, int) and not isinstance(item, bool):
            cnum = int(item)
            if cnum not in seen:
                seen.add(cnum)
                case_nums.append(cnum)
    if not case_nums and l3_res.error_message:
        for match in _L3_CASE_NUM_RE.finditer(l3_res.error_message):
            cnum = int(match.group(1))
            if cnum not in seen:
                seen.add(cnum)
                case_nums.append(cnum)
    return case_nums


def extract_knob_signature(
    mod_src: str,
    regime_spec: RegimeTargetSpec,
    *,
    resolve_regime_config_fn: Callable[
        [str, RegimeTargetSpec, Mapping[str, object] | None],
        ResolvedRegimeConfigLike,
    ]
    | None = None,
) -> str:
    """Extract canonical sorted knob signature for deterministic boundary ledger."""
    if not mod_src.strip() or resolve_regime_config_fn is None:
        return ""
    anchor_dims = (
        regime_spec.representative_anchor_dims or regime_spec.representative_hir_dims
    )
    try:
        resolved = resolve_regime_config_fn(mod_src, regime_spec, anchor_dims)
    except (
        SyntaxError,
        AttributeError,
        ImportError,
        KeyError,
        NameError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return ""
    parts: list[str] = []
    if resolved.selector_name:
        for key in sorted(resolved.builder_kwargs):
            if str(key).lower() in SHAPE_DIM_KEYS:
                continue
            parts.append(f"{key}={resolved.builder_kwargs[key]!r}")
    for key in sorted(resolved.pass_configs):
        parts.append(f"pc.{key}={resolved.pass_configs[key]!r}")
    if resolved.compile_flags:
        parts.append(
            f"flags={' '.join(sorted(str(flag) for flag in resolved.compile_flags))}"
        )
    return ",".join(parts)


def is_prime_gt2(value: int) -> bool:
    """Return True when value is an odd prime integer greater than 2."""
    if value < 3 or value % 2 == 0:
        return False
    divisor = 3
    while divisor * divisor <= value:
        if value % divisor == 0:
            return False
        divisor += 2
    return True


def case_dim_alignment_flags(case_dims: Mapping[str, object]) -> dict[str, bool]:
    """Compute odd, prime, and non-32B-aligned flags for counterexample dimensions."""
    tail_vals: list[int] = []
    for raw_key, raw_val in case_dims.items():
        key_norm = str(raw_key).strip().lower()
        if key_norm in NON_TAIL_DIM_KEYS:
            continue
        if isinstance(raw_val, int) and not isinstance(raw_val, bool) and raw_val > 1:
            tail_vals.append(int(raw_val))
    dtype_name = str(
        case_dims.get("tl_dtype") or case_dims.get("dtype") or "float16"
    ).lower()
    elem_bytes = 4 if "32" in dtype_name else 2
    has_odd = any(v % 2 == 1 for v in tail_vals)
    has_prime = any(is_prime_gt2(v) for v in tail_vals)
    non_32b = any((v * elem_bytes) % 32 != 0 for v in tail_vals)
    is_unaligned = (
        has_odd or has_prime or non_32b or any(v % 32 != 0 for v in tail_vals)
    )
    return {
        "has_odd_dim": has_odd,
        "has_prime_dim": has_prime,
        "non_32b_aligned": non_32b,
        "odd_or_prime_tail": bool(has_odd or has_prime),
        "has_unaligned_dim": is_unaligned,
        "is_unaligned": is_unaligned,
    }


@dataclass(frozen=True)
class _DevicePoolAdapter:
    device_id: str
    device_ids: tuple[str, ...]
    retry_device_id: str


class L3L4RecoveryCoordinator:
    """Coordinate L3/L4 execution, pending recovery, card rotation, and guardrails."""

    def __init__(
        self,
        *,
        operator_name: str,
        target_module_path: Path,
        source_root: Path,
        device_id: str = "1",
        retry_device_id: str = "",
        device_ids: tuple[str, ...] = (),
        l4_device_pool: tuple[str, ...] | None = None,
        l4_blocked_env_max_retries: int = 2,
        l4_retry_backoff_sec: float = 45.0,
        env_file: Path | None = None,
        cannbench_repo: Path | None = None,
        checkpoint_path: Path | None = None,
        experience_lake_dir: Path | None = None,
        run_id: str = "",
        l3_contract_runner: Callable[..., L3AcceptanceResult] | None = None,
        l4_cannbench_runner: Callable[..., L4CannbenchResult] | None = None,
        l2_batch_runner: Callable[
            [Sequence[tuple[str, Path]], Sequence[AnchorCaseSpec]],
            tuple[Mapping[str, CandidateSurrogateResult], Mapping[str, str]],
        ]
        | None = None,
        lookup_case_dims_fn: Callable[[int, str], dict[str, object]] | None = None,
        resolve_regime_config_fn: Callable[
            [str, RegimeTargetSpec, Mapping[str, object] | None],
            ResolvedRegimeConfigLike,
        ]
        | None = None,
        persist_state_fn: Callable[[str, Mapping[str, object] | None], None] | None = None,
        write_and_verify_fn: Callable[[Path, str], str] | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
        scoreboard_anchors: Sequence[AnchorCaseSpec] = (),
        regime_specs: Sequence[RegimeTargetSpec] = (),
        regime_blueprints: Mapping[str, HirBlueprintPacket] | None = None,
        topology_axes: TopologyAxes | Mapping[str, object] | None = None,
        guardrail_cases: list[GuardrailCaseSpec] | None = None,
        last_guardrail_metrics_by_cid: Mapping[str, Mapping[int, AnchorEvalMetrics]]
        | None = None,
        case_dims_cache: dict[int, dict[str, object]] | None = None,
    ) -> None:
        self._operator_name = operator_name
        self._target_module_path = Path(target_module_path).resolve()
        self._source_root = Path(source_root).resolve()
        self._device_id = (
            canonical_device_id(device_id, field="device_id")
            if device_id.strip()
            else "1"
        )
        stripped_retry = retry_device_id.strip()
        self._retry_device_id = (
            canonical_device_id(stripped_retry, field="retry_device_id")
            if stripped_retry
            else ""
        )
        self._device_ids = tuple(device_ids)
        if l4_device_pool is not None:
            self._l4_device_pool = tuple(l4_device_pool)
        else:
            self._l4_device_pool = resolve_l4_device_pool(
                _DevicePoolAdapter(
                    device_id=self._device_id,
                    device_ids=self._device_ids,
                    retry_device_id=self._retry_device_id,
                )
            )
        self._l4_blocked_env_max_retries = max(0, int(l4_blocked_env_max_retries))
        self._l4_retry_backoff_sec = float(l4_retry_backoff_sec)
        self._env_file = Path(env_file).resolve() if env_file is not None else None
        self._cannbench_repo = (
            Path(cannbench_repo).resolve() if cannbench_repo is not None else None
        )
        self._checkpoint_path = (
            Path(checkpoint_path).resolve() if checkpoint_path is not None else None
        )
        self._experience_lake_dir = (
            Path(experience_lake_dir).resolve()
            if experience_lake_dir is not None
            else None
        )
        self._run_id = run_id
        self._l3_contract_runner = l3_contract_runner
        self._l4_cannbench_runner = l4_cannbench_runner
        self._l2_batch_runner = l2_batch_runner
        self._lookup_case_dims_fn = lookup_case_dims_fn
        self._resolve_regime_config_fn = resolve_regime_config_fn
        self._persist_state_fn = persist_state_fn
        self._write_and_verify_fn = write_and_verify_fn
        self._sleep_fn = sleep_fn
        self.scoreboard_anchors: tuple[AnchorCaseSpec, ...] = tuple(scoreboard_anchors)
        self.regime_specs: tuple[RegimeTargetSpec, ...] = tuple(regime_specs)
        self.regime_blueprints: dict[str, HirBlueprintPacket] = dict(
            regime_blueprints or {}
        )
        if isinstance(topology_axes, TopologyAxes):
            self.topology_axes: TopologyAxes | None = topology_axes
        elif isinstance(topology_axes, Mapping):
            self.topology_axes = TopologyAxes.from_dict(topology_axes)
        else:
            self.topology_axes = None
        self.guardrail_cases: list[GuardrailCaseSpec] = (
            guardrail_cases if guardrail_cases is not None else []
        )
        self.last_guardrail_metrics_by_cid: Mapping[str, Mapping[int, AnchorEvalMetrics]] = (
            last_guardrail_metrics_by_cid
            if last_guardrail_metrics_by_cid is not None
            else {}
        )
        self.case_dims_cache: dict[int, dict[str, object]] = (
            case_dims_cache if case_dims_cache is not None else {}
        )
        self.pending_promotion: PendingPromotion | None = None
        self.active_promotion: PendingPromotion | None = None
        self.pending_l4: PendingL4 | None = None

    @property
    def l4_device_pool(self) -> tuple[str, ...]:
        """Return ordered authorized physical-card pool for L3/L4 execution."""
        return self._l4_device_pool

    def _persist_runtime_state(
        self,
        key: str,
        value: Mapping[str, object] | None,
    ) -> None:
        if self._persist_state_fn is not None:
            self._persist_state_fn(key, value)
            return
        persist_runtime_state(self._checkpoint_path, key, value)

    def _write_source_and_verify(
        self,
        path: Path,
        source: str,
        *,
        expected_sha256: str | None = None,
    ) -> str:
        if self._write_and_verify_fn is not None:
            return invoke_with_optional_kwargs(
                self._write_and_verify_fn,
                path,
                source,
                expected_sha256=expected_sha256,
            )
        return write_source_and_verify(path, source, expected_sha256=expected_sha256)

    def run_l3_contract(
        self,
        module_path: Path,
        *,
        device_id: str | None = None,
        on_execution_started: Callable[[str, str], None] | None = None,
    ) -> L3AcceptanceResult:
        """Execute a fresh L3 full-contract run on the requested device."""
        requested_device = str(device_id or self._device_id)
        if self._l3_contract_runner is not None:
            return invoke_with_optional_kwargs(
                self._l3_contract_runner,
                module_path,
                device_id=requested_device,
                device_pool=self._l4_device_pool,
                on_execution_started=on_execution_started,
            )
        return invoke_with_optional_kwargs(
            run_l3_full_contract_via_agent_gates,
            operator_name=self._operator_name,
            candidate_module_path=module_path,
            source_root=self._source_root,
            env_file=self._env_file,
            device_id=requested_device,
            device_pool=self._l4_device_pool,
            on_execution_started=on_execution_started,
        )

    def resume_l3_contract(
        self,
        module_path: Path,
        pending: PendingPromotion,
        *,
        on_execution_started: Callable[[str, str], None] | None = None,
    ) -> L3AcceptanceResult:
        """Continue a saved L3 execution instead of submitting a duplicate."""
        if self._l3_contract_runner is not None:
            return invoke_with_optional_kwargs(
                self._l3_contract_runner,
                module_path,
                resume_execution_id=pending.execution_id,
                resume_run_id=pending.run_id,
                device_id=pending.device_id or self._device_id,
                on_execution_started=on_execution_started,
            )
        return invoke_with_optional_kwargs(
            run_l3_full_contract_via_agent_gates,
            operator_name=self._operator_name,
            candidate_module_path=module_path,
            source_root=self._source_root,
            env_file=self._env_file,
            device_id=pending.device_id or self._device_id,
            resume_execution_id=pending.execution_id,
            resume_run_id=pending.run_id,
            on_execution_started=on_execution_started,
        )

    def next_l3_alt_card(self, pending: PendingPromotion, attempted_device: str) -> str:
        """Select the next untried authorized pool card for an L3 retry."""
        tried = {
            str(item.get("device_id", "")).strip()
            for item in pending.attempts
            if str(item.get("device_id", "")).strip()
        }
        if attempted_device.strip():
            tried.add(attempted_device.strip())
        if pending.device_id.strip():
            tried.add(pending.device_id.strip())
        untried = [dev for dev in self._l4_device_pool if dev not in tried]
        if not untried:
            return ""
        if len(untried) > 1 and untried[0] == self._l4_device_pool[0]:
            return untried[1]
        return untried[0]

    def can_retry_on_alt_card(self, pending: PendingPromotion, attempted_device: str) -> bool:
        """Return True when an untried authorized card exists for L3 retry."""
        return bool(self.next_l3_alt_card(pending, attempted_device))

    @staticmethod
    def promotion_attempt(
        result: L3AcceptanceResult,
        device_id: str,
        *,
        candidate_sha256: str = "",
    ) -> dict[str, str]:
        """Build one auditable L3 attempt entry."""
        return promotion_attempt(result, device_id, candidate_sha256=candidate_sha256)

    def retry_l3_on_alt_card(
        self,
        pending: PendingPromotion,
        attempts: tuple[dict[str, str], ...],
    ) -> L3AcceptanceResult:
        """Start one fresh L3 execution on the next untried authorized pool card."""
        retry_device = self.next_l3_alt_card(pending, pending.device_id)
        if not retry_device:
            raise RuntimeError("no untried alternate device available for L3 retry")
        retry_pending = replace(
            pending,
            execution_id="",
            run_id="",
            receipt_path="",
            error_message="",
            attempts=attempts,
            device_id=retry_device,
            execution_terminal=False,
            retry_used=True,
        )

        def _record_started(execution_id: str, device: str) -> None:
            started = replace(
                retry_pending,
                execution_id=execution_id,
                device_id=device,
                execution_terminal=False,
            )
            self.active_promotion = started
            self.pending_promotion = started
            self._persist_runtime_state("active_promotion", started.to_dict())

        result = self.run_l3_contract(
            self._target_module_path,
            device_id=retry_device,
            on_execution_started=_record_started,
        )
        self.active_promotion = None
        return result

    @staticmethod
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
        record_promotion_attempts(
            rounds,
            pending,
            attempts,
            promoted=promoted,
            l3_resolution=l3_resolution,
            stagnation_after=stagnation_after,
        )

    def _regime_spec_for_tag(self, regime_tag: str) -> RegimeTargetSpec:
        for spec in self.regime_specs:
            if spec.regime_tag == regime_tag:
                return spec
        return RegimeTargetSpec(
            regime_tag=regime_tag,
            builder_candidates=(),
            sample_builder_kwargs={},
        )

    def resolve_pending_promotion(
        self,
        pending: PendingPromotion,
        *,
        incumbent_source: str,
        best_result: CandidateSurrogateResult,
        stagnation_by_regime: dict[str, int],
        rounds: list[_RoundT],
        promoted_rounds: list[int],
        rolled_back_rounds: list[int],
        active_anchors: Sequence[AnchorCaseSpec],
    ) -> tuple[
        str,
        CandidateSurrogateResult,
        dict[str, int],
        PendingPromotion | None,
        bool,
    ]:
        """Resume, retry on an untried card, or retain a recoverable pending L3."""
        _ = active_anchors
        ensure_error = ensure_candidate_bytes(
            path=self._target_module_path,
            source=pending.candidate_source,
            expected_sha256=pending.candidate_sha256,
        )
        if ensure_error:
            updated = replace(pending, error_message=ensure_error)
            self.pending_promotion = updated
            self.active_promotion = updated if not updated.execution_terminal else None
            return incumbent_source, best_result, stagnation_by_regime, updated, True

        attempts = list(pending.attempts)
        current = pending

        def _keep_pending(
            result: L3AcceptanceResult,
            *,
            terminal: bool,
        ) -> tuple[
            str,
            CandidateSurrogateResult,
            dict[str, int],
            PendingPromotion | None,
            bool,
        ]:
            nonlocal current
            attempts.append(
                self.promotion_attempt(
                    result,
                    current.device_id,
                    candidate_sha256=current.candidate_sha256,
                )
            )
            current = replace(
                current,
                execution_id=result.execution_id or current.execution_id,
                run_id=result.run_id or current.run_id,
                receipt_path=result.receipt_path or current.receipt_path,
                error_message=result.error_message or current.error_message,
                attempts=tuple(attempts),
                device_id=result.device_id or current.device_id,
                execution_terminal=terminal,
            )
            if terminal:
                restore_error = self._write_source_and_verify(
                    self._target_module_path,
                    incumbent_source,
                )
                if restore_error:
                    raise RuntimeError(restore_error)
                self.active_promotion = None
                self._persist_runtime_state("active_promotion", None)
            else:
                self.active_promotion = current
                self._persist_runtime_state("active_promotion", current.to_dict())
            self.pending_promotion = current
            self._persist_runtime_state("pending_promotion", current.to_dict())
            self.record_promotion_attempts(
                rounds,
                current,
                tuple(attempts),
                promoted=False,
                l3_resolution=L3Resolution.INFRA_DEFERRED.value,
                stagnation_after=stagnation_by_regime.get(current.regime_tag, 0),
            )
            return incumbent_source, best_result, stagnation_by_regime, current, True

        def _promote() -> tuple[
            str,
            CandidateSurrogateResult,
            dict[str, int],
            PendingPromotion | None,
            bool,
        ]:
            nonlocal current
            self.record_promotion_attempts(
                rounds,
                current,
                tuple(attempts),
                promoted=True,
                l3_resolution=L3Resolution.PROMOTED.value,
                stagnation_after=0,
            )
            if current.round_idx in rolled_back_rounds:
                rolled_back_rounds.remove(current.round_idx)
            if current.round_idx not in promoted_rounds:
                promoted_rounds.append(current.round_idx)
            stagnation_by_regime[current.regime_tag] = 0
            self.refresh_guardrail_incumbent_refs(current.candidate_id)
            self.pending_promotion = None
            self.active_promotion = None
            self._persist_runtime_state("active_promotion", None)
            self._persist_runtime_state("pending_promotion", None)
            return (
                current.candidate_source,
                current.candidate_result,
                stagnation_by_regime,
                None,
                False,
            )

        def _candidate_failure(
            result: L3AcceptanceResult,
        ) -> tuple[
            str,
            CandidateSurrogateResult,
            dict[str, int],
            PendingPromotion | None,
            bool,
        ]:
            nonlocal current
            attempts.append(
                self.promotion_attempt(
                    result,
                    current.device_id,
                    candidate_sha256=current.candidate_sha256,
                )
            )
            restore_error = self._write_source_and_verify(
                self._target_module_path,
                incumbent_source,
            )
            if restore_error:
                raise RuntimeError(restore_error)
            self.record_l3_guardrail_cases(
                result,
                incumbent_source,
                self._regime_spec_for_tag(current.regime_tag),
                topology_axes=self.topology_axes,
            )
            if current.round_idx not in rolled_back_rounds:
                rolled_back_rounds.append(current.round_idx)
            stagnation_by_regime[current.regime_tag] = (
                stagnation_by_regime.get(current.regime_tag, 0) + 1
            )
            self.record_promotion_attempts(
                rounds,
                current,
                tuple(attempts),
                promoted=False,
                l3_resolution=L3Resolution.CANDIDATE_REJECTED.value,
                stagnation_after=stagnation_by_regime[current.regime_tag],
            )
            self.pending_promotion = None
            self.active_promotion = None
            self._persist_runtime_state("active_promotion", None)
            self._persist_runtime_state("pending_promotion", None)
            return incumbent_source, best_result, stagnation_by_regime, None, False

        def _retry_alt_cards(
            start_result: L3AcceptanceResult | None,
            last_device: str,
        ) -> tuple[
            str,
            CandidateSurrogateResult,
            dict[str, int],
            PendingPromotion | None,
            bool,
        ]:
            nonlocal current
            tried_alt_cards: set[str] = set()
            latest_result = start_result
            while self.can_retry_on_alt_card(current, last_device):
                target_dev = self.next_l3_alt_card(current, last_device)
                if not target_dev or target_dev in tried_alt_cards:
                    break
                tried_alt_cards.add(target_dev)
                retry_result = self.retry_l3_on_alt_card(current, tuple(attempts))
                attempts.append(
                    self.promotion_attempt(
                        retry_result,
                        target_dev or current.device_id,
                        candidate_sha256=current.candidate_sha256,
                    )
                )
                current = replace(
                    current,
                    execution_id=retry_result.execution_id or current.execution_id,
                    run_id=retry_result.run_id or current.run_id,
                    receipt_path=retry_result.receipt_path or current.receipt_path,
                    error_message=retry_result.error_message or current.error_message,
                    attempts=tuple(attempts),
                    device_id=target_dev or retry_result.device_id or current.device_id,
                    execution_terminal=retry_result.execution_terminal,
                    retry_used=True,
                )
                if retry_result.passed:
                    return _promote()
                if not retry_result.execution_terminal:
                    return _keep_pending(retry_result, terminal=False)
                if retry_result.failure_kind == FAILURE_KIND_CANDIDATE:
                    return _candidate_failure(retry_result)
                latest_result = retry_result
                last_device = retry_result.device_id or target_dev
            if latest_result is not None:
                return _keep_pending(
                    latest_result,
                    terminal=latest_result.execution_terminal,
                )
            self.pending_promotion = pending
            self.active_promotion = pending if not pending.execution_terminal else None
            return incumbent_source, best_result, stagnation_by_regime, pending, True

        if pending.execution_id and (not pending.execution_terminal or pending.run_id):
            result = self.resume_l3_contract(self._target_module_path, pending)
            attempts.append(
                self.promotion_attempt(
                    result,
                    pending.device_id,
                    candidate_sha256=pending.candidate_sha256,
                )
            )
            current = replace(
                pending,
                execution_id=result.execution_id or pending.execution_id,
                run_id=result.run_id or pending.run_id,
                receipt_path=result.receipt_path or pending.receipt_path,
                error_message=result.error_message or pending.error_message,
                attempts=tuple(attempts),
                device_id=result.device_id or pending.device_id,
                execution_terminal=result.execution_terminal,
            )
            if result.passed:
                return _promote()
            if not result.execution_terminal:
                return _keep_pending(result, terminal=False)
            if result.failure_kind == FAILURE_KIND_CANDIDATE:
                return _candidate_failure(result)
            if self.can_retry_on_alt_card(current, result.device_id):
                return _retry_alt_cards(result, result.device_id)
            return _keep_pending(result, terminal=True)

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
        on_execution_started: Callable[[str, str], None] | None = None,
    ) -> L4CannbenchResult:
        """Execute or resume an official L4 CANNBench run."""
        requested_device = str(device_id or self._device_id)
        if self._l4_cannbench_runner is not None:
            return invoke_with_optional_kwargs(
                self._l4_cannbench_runner,
                module_path,
                resume_execution_id=resume_execution_id,
                resume_run_id=resume_run_id,
                device_id=requested_device,
                on_execution_started=on_execution_started,
            )
        return invoke_with_optional_kwargs(
            run_l4_official_cannbench_via_agent_gates,
            operator_name=self._operator_name,
            candidate_module_path=module_path,
            source_root=self._source_root,
            env_file=self._env_file,
            device_id=requested_device,
            resume_execution_id=resume_execution_id,
            resume_run_id=resume_run_id,
            on_execution_started=on_execution_started,
        )

    @staticmethod
    def l4_attempt(result: L4CannbenchResult) -> dict[str, str]:
        """Build one auditable L4 attempt entry."""
        return l4_attempt(result)

    def can_retry_l4_on_alt_card(
        self,
        pending: PendingL4,
        attempted_device: str,
    ) -> bool:
        """Return True when the legacy alternate retry card is configured and untried."""
        retry_device = self._retry_device_id
        attempted = attempted_device or pending.device_id
        return bool(
            retry_device
            and not pending.retry_used
            and retry_device != attempted
            and retry_device not in self.l4_round_attempted_devices(pending)
        )

    @staticmethod
    def l4_round_attempted_devices(pending: PendingL4) -> tuple[str, ...]:
        """Return cards already tried in the current blocked-env round."""
        return l4_round_attempted_devices(pending)

    def effective_l4_round_limit(self, pending: PendingL4) -> int:
        """Return the stored quota, or the current config limit for legacy records."""
        if pending.blocked_env_round_limit is not None:
            return max(0, pending.blocked_env_round_limit)
        return max(0, self._l4_blocked_env_max_retries)

    def effective_l4_pool(self, pending: PendingL4) -> tuple[str, ...]:
        """Intersect a saved pool with the pool authorized by the current config."""
        if not pending.device_pool:
            return self._l4_device_pool
        allowed = set(self._l4_device_pool)
        narrowed = tuple(device for device in pending.device_pool if device in allowed)
        return narrowed or self._l4_device_pool

    def advance_l4_card_failure(
        self,
        pending: PendingL4,
        result: L4CannbenchResult,
        attempts: list[dict[str, str]],
    ) -> tuple[L4CannbenchResult, PendingL4]:
        """Advance a terminal occupancy or refused-card L4 run across the card pool."""
        pool = self.effective_l4_pool(pending)
        limit = self.effective_l4_round_limit(pending)
        current = replace(
            self.pending_l4_after(
                pending,
                result,
                tuple(attempts),
                retry_used=pending.retry_used,
            ),
            device_pool=pool,
            blocked_env_round_limit=limit,
        )

        def _record_started(execution_id: str, device: str) -> None:
            nonlocal current
            current = replace(
                current,
                execution_id=execution_id or current.execution_id,
                device_id=device or current.device_id,
                state="SUBMITTED",
                execution_terminal=False,
            )
            self.pending_l4 = current
            self._persist_runtime_state("pending_l4", current.to_dict())

        while result.execution_terminal and (
            is_occupancy_blocked_env(
                result.failure_kind,
                result.blocked_env_category,
            )
            or self.is_rejected_card_l4(result)
        ):
            rejected = self.is_rejected_card_l4(result)
            attempted = self.l4_round_attempted_devices(current)
            eligible = tuple(
                device for device in pool if device not in current.rejected_device_ids
            )
            untried = tuple(device for device in eligible if device not in attempted)
            if untried:
                next_device = untried[0]
                next_rounds = current.blocked_env_rounds_used
                next_attempted = (*attempted, next_device)
                delay = 0.0
            elif (
                rejected
                or not eligible
                or current.blocked_env_rounds_used >= limit
            ):
                break
            else:
                next_rounds = current.blocked_env_rounds_used + 1
                next_device = eligible[0]
                next_attempted = (next_device,)
                delay = max(0.0, self._l4_retry_backoff_sec) * next_rounds
            current = replace(
                current,
                execution_id="",
                run_id="",
                receipt_path="",
                device_id=next_device,
                state="SUBMITTED",
                error_message=result.error_message,
                failure_kind=FAILURE_KIND_INFRASTRUCTURE,
                execution_terminal=False,
                blocked_env_rounds_used=next_rounds,
                blocked_env_round_attempted_device_ids=next_attempted,
                blocked_env_category="",
                device_rejected=False,
            )
            self.pending_l4 = current
            self._persist_runtime_state("pending_l4", current.to_dict())
            if delay > 0.0:
                self._sleep_fn(delay)
            result = self.run_l4_cannbench(
                self._target_module_path,
                device_id=next_device,
                on_execution_started=_record_started,
            )
            attempts.append(self.l4_attempt(result))
            current = self.pending_l4_after(
                current,
                result,
                tuple(attempts),
                retry_used=current.retry_used,
            )
            self.pending_l4 = current
            self._persist_runtime_state("pending_l4", current.to_dict())
        return result, current

    @staticmethod
    def merge_l4_identity(
        result: L4CannbenchResult,
        pending: PendingL4,
    ) -> L4CannbenchResult:
        """Merge saved pending L4 identity fields onto an L4 result."""
        return merge_l4_identity(result, pending)

    @staticmethod
    def pending_l4_after(
        pending: PendingL4,
        result: L4CannbenchResult,
        attempts: tuple[dict[str, str], ...],
        *,
        retry_used: bool,
    ) -> PendingL4:
        """Advance PendingL4 state after an L4 attempt outcome."""
        return pending_l4_after(pending, result, attempts, retry_used=retry_used)

    def resolve_pending_l4(
        self,
        pending: PendingL4,
        *,
        incumbent_source: str,
    ) -> tuple[L4CannbenchResult, PendingL4 | None]:
        """Continue the saved L4 execution, or move the complete run to another card."""
        expected = pending.candidate_sha256 or source_sha256(incumbent_source)
        digest_error = ""
        try:
            current_digest = hashlib.sha256(
                self._target_module_path.read_bytes()
            ).hexdigest()
        except OSError as exc:
            digest_error = (
                f"pending L4 candidate {self._target_module_path} cannot be read: {exc}"
            )
        else:
            if current_digest != expected:
                digest_error = (
                    f"pending L4 candidate {self._target_module_path} digest "
                    f"{current_digest} does not match submitted {expected}; "
                    "refusing to resume or replace it"
                )
        if digest_error:
            failed = L4CannbenchResult(
                passed=False,
                error_message=digest_error,
                failure_kind=FAILURE_KIND_INFRASTRUCTURE,
                execution_id=pending.execution_id,
                run_id=pending.run_id,
                receipt_path=pending.receipt_path,
                device_id=pending.device_id,
                state=pending.state,
                execution_terminal=pending.execution_terminal,
                target_path=pending.target_path,
                candidate_sha256=expected,
                attempts=pending.attempts,
            )
            updated = replace(
                pending,
                candidate_sha256=expected,
                error_message=digest_error,
            )
            self.pending_l4 = updated
            return failed, updated

        attempts = list(pending.attempts)
        effective_pool = self.effective_l4_pool(pending)
        effective_limit = self.effective_l4_round_limit(pending)
        current = replace(
            pending,
            device_pool=effective_pool,
            blocked_env_round_limit=effective_limit,
        )

        def _record_started(execution_id: str, device: str) -> None:
            nonlocal current
            current = replace(
                current,
                execution_id=execution_id or current.execution_id,
                device_id=device or current.device_id,
                state="SUBMITTED",
                execution_terminal=False,
            )
            self.pending_l4 = current
            self._persist_runtime_state("pending_l4", current.to_dict())

        def _keep(
            result: L4CannbenchResult,
            *,
            retry_used: bool,
        ) -> tuple[L4CannbenchResult, PendingL4 | None]:
            nonlocal current
            updated = self.pending_l4_after(
                current,
                result,
                tuple(attempts),
                retry_used=retry_used,
            )
            self.pending_l4 = updated
            self._persist_runtime_state("pending_l4", updated.to_dict())
            return self.merge_l4_identity(result, updated), updated

        def _finish(
            result: L4CannbenchResult,
        ) -> tuple[L4CannbenchResult, PendingL4 | None]:
            self.pending_l4 = None
            self._persist_runtime_state("pending_l4", None)
            return self.merge_l4_identity(result, current), None

        def _retry_whole_l4() -> tuple[L4CannbenchResult, PendingL4 | None]:
            nonlocal current
            retry_device = self._retry_device_id
            if not retry_device:
                raise RuntimeError("retry_device_id is empty during whole-L4 retry")
            current = replace(
                current,
                execution_id="",
                run_id="",
                receipt_path="",
                device_id=retry_device,
                state="SUBMITTED",
                error_message="",
                failure_kind=FAILURE_KIND_INFRASTRUCTURE,
                execution_terminal=False,
                retry_used=True,
                blocked_env_category="",
                device_rejected=False,
            )
            retry_result = self.run_l4_cannbench(
                self._target_module_path,
                device_id=retry_device,
                on_execution_started=_record_started,
            )
            attempts.append(self.l4_attempt(retry_result))
            current = replace(
                current,
                execution_id=retry_result.execution_id or current.execution_id,
                run_id=retry_result.run_id or current.run_id,
                receipt_path=retry_result.receipt_path or current.receipt_path,
                device_id=retry_result.device_id or current.device_id,
                state=retry_result.state or current.state,
                error_message=retry_result.error_message or current.error_message,
                failure_kind=retry_result.failure_kind or current.failure_kind,
                execution_terminal=retry_result.execution_terminal,
                attempts=tuple(attempts),
            )
            if retry_result.passed or retry_result.failure_kind == FAILURE_KIND_CANDIDATE:
                return _finish(retry_result)
            return _keep(retry_result, retry_used=True)

        def _after_infrastructure(
            result: L4CannbenchResult,
        ) -> tuple[L4CannbenchResult, PendingL4 | None]:
            nonlocal current
            if (
                is_occupancy_blocked_env(
                    result.failure_kind,
                    result.blocked_env_category,
                )
                or self.is_rejected_card_l4(result)
            ):
                result, current = self.advance_l4_card_failure(
                    current,
                    result,
                    attempts,
                )
            if result.passed or result.failure_kind == FAILURE_KIND_CANDIDATE:
                return _finish(result)
            if not result.execution_terminal:
                return _keep(result, retry_used=current.retry_used)
            if self.can_retry_l4_on_alt_card(current, result.device_id):
                return _retry_whole_l4()
            return _keep(result, retry_used=current.retry_used)

        if pending.execution_id and (not pending.execution_terminal or pending.run_id):
            result = self.run_l4_cannbench(
                self._target_module_path,
                resume_execution_id=pending.execution_id,
                resume_run_id=pending.run_id,
                device_id=pending.device_id or self._device_id,
                on_execution_started=_record_started,
            )
            attempts.append(self.l4_attempt(result))
            current = self.pending_l4_after(
                current,
                result,
                tuple(attempts),
                retry_used=current.retry_used,
            )
            self.pending_l4 = current
            self._persist_runtime_state("pending_l4", current.to_dict())
            return _after_infrastructure(result)

        blocked_env_pending = is_occupancy_blocked_env(
            pending.failure_kind,
            pending.blocked_env_category,
        )
        rejected_pending = (
            pending.device_rejected
            and pending.failure_kind == FAILURE_KIND_INFRASTRUCTURE
        )
        interrupted_submission = (
            not pending.execution_terminal
            and not pending.execution_id
            and not pending.run_id
            and pending.state == "SUBMITTED"
        )
        if (
            (blocked_env_pending and pending.execution_terminal)
            or rejected_pending
            or (
                interrupted_submission
                and current.blocked_env_rounds_used <= effective_limit
            )
        ):
            if interrupted_submission:
                interrupted_device = (
                    current.device_id
                    if current.device_id in effective_pool
                    else effective_pool[0]
                )
                current = replace(current, device_id=interrupted_device)
                result = self.run_l4_cannbench(
                    self._target_module_path,
                    device_id=interrupted_device,
                    on_execution_started=_record_started,
                )
                attempts.append(self.l4_attempt(result))
                current = self.pending_l4_after(
                    current,
                    result,
                    tuple(attempts),
                    retry_used=current.retry_used,
                )
                self.pending_l4 = current
                self._persist_runtime_state("pending_l4", current.to_dict())
            else:
                result = L4CannbenchResult(
                    passed=False,
                    error_message=pending.error_message,
                    failure_kind=pending.failure_kind,
                    execution_id=pending.execution_id,
                    run_id=pending.run_id,
                    receipt_path=pending.receipt_path,
                    device_id=pending.device_id,
                    state=pending.state,
                    execution_terminal=pending.execution_terminal,
                    blocked_env_category=pending.blocked_env_category,
                    device_rejected=pending.device_rejected,
                    target_path=pending.target_path,
                    candidate_sha256=expected,
                    attempts=pending.attempts,
                )
            return _after_infrastructure(result)

        if self.can_retry_l4_on_alt_card(pending, pending.device_id):
            return _retry_whole_l4()
        self.pending_l4 = pending
        failed = L4CannbenchResult(
            passed=False,
            error_message=(
                pending.error_message
                or "L4 infrastructure failure has no resumable execution and no "
                "other authorized card is configured"
            ),
            failure_kind=FAILURE_KIND_INFRASTRUCTURE,
            execution_id=pending.execution_id,
            run_id=pending.run_id,
            receipt_path=pending.receipt_path,
            device_id=pending.device_id,
            state=pending.state,
            execution_terminal=pending.execution_terminal,
            blocked_env_category=pending.blocked_env_category,
            device_rejected=pending.device_rejected,
            target_path=pending.target_path,
            candidate_sha256=expected,
            attempts=pending.attempts,
        )
        return failed, pending

    def l4_candidate_relative(self) -> str:
        """Return target module path relative to source root, or filename fallback."""
        try:
            return self._target_module_path.resolve().relative_to(
                self._source_root.resolve()
            ).as_posix()
        except (OSError, ValueError):
            return self._target_module_path.name

    @staticmethod
    def is_blocked_env_l4(result: L4CannbenchResult) -> bool:
        """Return True only for an occupancy-classified BLOCKED_ENV receipt."""
        return is_blocked_env_l4(result)

    @staticmethod
    def is_rejected_card_l4(result: L4CannbenchResult) -> bool:
        """Return True when AgentGates refused the submitted card as unauthorized."""
        return is_rejected_card_l4(result)

    def run_final_l4_tracked(
        self,
        *,
        incumbent_source: str,
        prior_stop_reason: str,
    ) -> L4CannbenchResult:
        """Submit the official L4 run and persist its identity before waiting."""
        candidate_sha = source_sha256(incumbent_source)
        candidate_relative = self.l4_candidate_relative()
        pool = self._l4_device_pool
        attempts: list[dict[str, str]] = []
        current = PendingL4(
            target_path=str(self._target_module_path),
            candidate_relative=candidate_relative,
            candidate_sha256=candidate_sha,
            device_id=pool[0],
            state="SUBMITTED",
            failure_kind=FAILURE_KIND_INFRASTRUCTURE,
            execution_terminal=False,
            device_pool=pool,
            blocked_env_round_attempted_device_ids=(pool[0],),
            blocked_env_round_limit=max(0, self._l4_blocked_env_max_retries),
            prior_stop_reason=prior_stop_reason,
        )

        def _record_started(execution_id: str, device: str) -> None:
            nonlocal current
            current = replace(
                current,
                execution_id=execution_id or current.execution_id,
                device_id=device or current.device_id,
                state="SUBMITTED",
                execution_terminal=False,
            )
            self.pending_l4 = current
            self._persist_runtime_state("pending_l4", current.to_dict())

        result = self.run_l4_cannbench(
            self._target_module_path,
            device_id=pool[0],
            on_execution_started=_record_started,
        )
        attempts.append(self.l4_attempt(result))
        current = self.pending_l4_after(
            current,
            result,
            tuple(attempts),
            retry_used=False,
        )
        if self.is_blocked_env_l4(result) or self.is_rejected_card_l4(result):
            result, current = self.advance_l4_card_failure(current, result, attempts)
        if result.passed or result.failure_kind == FAILURE_KIND_CANDIDATE:
            self.pending_l4 = None
            self._persist_runtime_state("pending_l4", None)
            return self.merge_l4_identity(result, current)
        self.pending_l4 = current
        self._persist_runtime_state("pending_l4", current.to_dict())
        return self.merge_l4_identity(result, current)

    @staticmethod
    def extract_l3_failed_case_nums(l3_res: L3AcceptanceResult) -> list[int]:
        """Extract deduplicated integer case numbers from an L3 failure result."""
        return extract_l3_failed_case_nums(l3_res)

    def extract_knob_signature(
        self,
        mod_src: str,
        regime_spec: RegimeTargetSpec,
    ) -> str:
        """Extract canonical sorted knob signature for deterministic boundary ledger."""
        return extract_knob_signature(
            mod_src,
            regime_spec,
            resolve_regime_config_fn=self._resolve_regime_config_fn,
        )

    @staticmethod
    def is_prime_gt2(value: int) -> bool:
        """Return True when value is an odd prime integer greater than 2."""
        return is_prime_gt2(value)

    @staticmethod
    def case_dim_alignment_flags(case_dims: Mapping[str, object]) -> dict[str, bool]:
        """Compute odd, prime, and non-32B-aligned flags for counterexample dimensions."""
        return case_dim_alignment_flags(case_dims)

    def _lookup_case_dims_safe(
        self,
        case_num: int,
        regime_tag: str = "",
    ) -> dict[str, object]:
        cnum = int(case_num)
        if cnum in self.case_dims_cache:
            return dict(self.case_dims_cache[cnum])
        dims: dict[str, object] = {}
        try:
            dims = dict(
                lookup_case_dims(
                    self._operator_name,
                    cnum,
                    cannbench_repo=self._cannbench_repo,
                    env_file=self._env_file,
                )
            )
        except (FileNotFoundError, OSError, KeyError, ValueError, TypeError):
            if self._lookup_case_dims_fn is not None:
                dims = dict(self._lookup_case_dims_fn(cnum, regime_tag))
            else:
                for spec in self.regime_specs:
                    if regime_tag and spec.regime_tag == regime_tag:
                        dims = dict(
                            spec.representative_anchor_dims or spec.representative_hir_dims
                        )
                        break
                if not dims and self.regime_specs:
                    first = self.regime_specs[0]
                    dims = dict(
                        first.representative_anchor_dims or first.representative_hir_dims
                    )
        self.case_dims_cache[cnum] = dict(dims)
        return dims

    def evaluate_incumbent_on_new_anchors(
        self,
        incumbent_source: str,
        new_specs: Sequence[AnchorCaseSpec],
    ) -> list[AnchorEvalMetrics]:
        """Measure incumbent AnchorEvalMetrics on newly added probe cases."""
        if (
            not new_specs
            or not incumbent_source.strip()
            or self._l2_batch_runner is None
        ):
            return []
        try:
            with tempfile.TemporaryDirectory(prefix="tf_inc_dyn_anchor_") as tmp_dir:
                inc_file = Path(tmp_dir) / "incumbent.py"
                inc_file.write_text(incumbent_source, encoding="utf-8")
                l2_probe, _ = self._l2_batch_runner(
                    [("incumbent", inc_file)],
                    list(new_specs),
                )
            inc_res = l2_probe.get("incumbent")
            if inc_res is not None and inc_res.anchors:
                return list(inc_res.anchors)
        except (
            OSError,
            subprocess.SubprocessError,
            TimeoutError,
            json.JSONDecodeError,
            RuntimeError,
            FileNotFoundError,
            KeyError,
            ValueError,
        ):
            pass
        return []

    def record_l3_guardrail_cases(
        self,
        l3_res: L3AcceptanceResult,
        incumbent_source: str,
        regime_spec: RegimeTargetSpec,
        *,
        topology_axes: TopologyAxes | None = None,
    ) -> list[GuardrailCaseSpec]:
        """Record all unseen non-scoreboard L3 failure probe cases monotonically."""
        case_nums = self.extract_l3_failed_case_nums(l3_res)
        scoreboard_cases = {int(a.case_num) for a in self.scoreboard_anchors}
        existing_guardrails = {int(g.case_num) for g in self.guardrail_cases}
        target_cases = [
            c
            for c in case_nums
            if c not in scoreboard_cases and c not in existing_guardrails
        ]
        if not target_cases:
            return []
        try:
            task_name = get_operator_task(self._operator_name)
        except (FileNotFoundError, KeyError, ValueError):
            task_name = f"operator/{self._operator_name}"
        bp = self.regime_blueprints.get(regime_spec.regime_tag)
        default_t_hw = (
            float(bp.roofline.ideal_roofline_us)
            if bp is not None and bp.roofline.ideal_roofline_us > 0.0
            else 1.0
        )
        eff_axes = topology_axes if topology_axes is not None else self.topology_axes
        axes_dict = eff_axes.to_dict() if eff_axes is not None else {}
        failure_stage = str(getattr(l3_res, "failure_stage", "") or "")
        probe_specs: list[AnchorCaseSpec] = []
        pending_specs: list[tuple[int, str, float, float]] = []
        for cnum in target_cases:
            dims: dict[str, object] = {}
            try:
                dims = dict(
                    lookup_case_dims(
                        self._operator_name,
                        cnum,
                        cannbench_repo=self._cannbench_repo,
                        env_file=self._env_file,
                    )
                )
            except (FileNotFoundError, OSError, KeyError, ValueError, TypeError):
                if cnum in self.case_dims_cache:
                    dims = dict(self.case_dims_cache[cnum])
                elif self._lookup_case_dims_fn is not None:
                    dims = dict(self._lookup_case_dims_fn(cnum, regime_spec.regime_tag))
            true_regime_tag = regime_spec.regime_tag
            if dims and self.regime_specs:
                try:
                    true_regime_tag = resolve_regime_for_case(
                        self._operator_name,
                        dims,
                        self.regime_specs,
                    ).regime_tag
                except (KeyError, ValueError, TypeError):
                    pass
            ref_perf_us = 1.0
            t_hw_us = default_t_hw
            try:
                meta_ref, meta_hw = lookup_case_metadata(
                    self._operator_name,
                    cnum,
                    cannbench_repo=self._cannbench_repo,
                    env_file=self._env_file,
                )
                if meta_ref > 0.0:
                    ref_perf_us = float(meta_ref)
                if meta_hw > 0.0:
                    t_hw_us = float(meta_hw)
            except (FileNotFoundError, OSError, KeyError, ValueError, json.JSONDecodeError):
                pass
            probe_specs.append(
                AnchorCaseSpec(
                    task=task_name,
                    case_num=cnum,
                    regime_tag=true_regime_tag,
                    ref_perf_us=ref_perf_us,
                    t_hw_us=t_hw_us,
                )
            )
            pending_specs.append((cnum, true_regime_tag, ref_perf_us, t_hw_us))
            if self._experience_lake_dir is not None:
                eff_dims = dims or self._lookup_case_dims_safe(cnum, true_regime_tag)
                align_flags = self.case_dim_alignment_flags(eff_dims)
                try:
                    append_counterexample_boundary(
                        self._experience_lake_dir,
                        {
                            "run_id": self._run_id,
                            "operator": self._operator_name,
                            "regime_tag": true_regime_tag,
                            "topology_axes": axes_dict,
                            "case_num": int(cnum),
                            "case_dims": eff_dims,
                            "failure_stage": failure_stage,
                            "ref_perf_us": float(ref_perf_us),
                            "t_hw_us": float(t_hw_us),
                            **align_flags,
                        },
                    )
                except (OSError, ValueError) as exc:
                    logger.debug("Appending counterexample boundary skipped: %s", exc)

        inc_metrics = self.evaluate_incumbent_on_new_anchors(incumbent_source, probe_specs)
        inc_us_by_case = {
            int(m.case_num): float(m.cand_us) for m in inc_metrics if m.cand_us > 0.0
        }
        added: list[GuardrailCaseSpec] = []
        seen_guardrails = set(existing_guardrails)
        for cnum, true_regime_tag, ref_perf_us, t_hw_us in pending_specs:
            if cnum in seen_guardrails:
                continue
            seen_guardrails.add(cnum)
            inc_ref = inc_us_by_case.get(cnum, 0.0)
            if inc_ref <= 0.0 and ref_perf_us > 1.0:
                inc_ref = ref_perf_us
            spec = GuardrailCaseSpec(
                task=task_name,
                case_num=cnum,
                regime_tag=true_regime_tag,
                incumbent_ref_us=inc_ref,
                t_hw_us=t_hw_us,
            )
            self.guardrail_cases.append(spec)
            added.append(spec)
        return added

    def refresh_guardrail_incumbent_refs(self, winner_candidate_id: str) -> None:
        """Update guardrail incumbent reference latencies from a newly promoted winner."""
        if not self.guardrail_cases:
            return
        metrics_by_case = self.last_guardrail_metrics_by_cid.get(winner_candidate_id)
        if not metrics_by_case:
            return
        for idx, guardrail in enumerate(self.guardrail_cases):
            metric = metrics_by_case.get(int(guardrail.case_num))
            if metric is not None and metric.accuracy_passed and metric.cand_us > 0.0:
                self.guardrail_cases[idx] = replace(
                    guardrail,
                    incumbent_ref_us=float(metric.cand_us),
                )


__all__ = [
    "CAMPAIGN_DEVICE_IDS_ENV_KEY",
    "DevicePoolConfig",
    "L3L4RecoveryCoordinator",
    "NON_TAIL_DIM_KEYS",
    "PendingL4",
    "ResolvedRegimeConfigLike",
    "RoundRecordLike",
    "SHAPE_DIM_KEYS",
    "atomic_write_text",
    "canonical_device_id",
    "case_dim_alignment_flags",
    "ensure_candidate_bytes",
    "extract_knob_signature",
    "extract_l3_failed_case_nums",
    "invoke_with_optional_kwargs",
    "is_blocked_env_l4",
    "is_occupancy_blocked_env",
    "is_prime_gt2",
    "is_rejected_card_l4",
    "l4_attempt",
    "l4_round_attempted_devices",
    "merge_l4_identity",
    "parse_device_ids",
    "pending_l4_after",
    "persist_runtime_state",
    "promotion_attempt",
    "record_promotion_attempts",
    "resolve_campaign_device_plan",
    "resolve_l4_device_pool",
    "source_sha256",
    "write_source_and_verify",
]
