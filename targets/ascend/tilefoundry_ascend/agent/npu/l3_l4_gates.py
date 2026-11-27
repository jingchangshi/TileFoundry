"""L3 full/guardrail accuracy gate and L4 official CANNBench AgentGates client."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from tilefoundry_ascend.agent.npu.ag_transport import (
    AGENT_GATES_L3_PROMOTION_GATE, AGENT_GATES_L4_CANNBENCH_GATE,
    DEFAULT_L3_PROMOTION_WAIT_SEC, DEFAULT_L4_CANNBENCH_WAIT_SEC,
    FAILURE_KIND_CANDIDATE, FAILURE_KIND_INFRASTRUCTURE,
    _agent_gates_workspace_name, _as_positive_int, _blocked_env_category,
    _call_order_candidates_by_idle_preflight, _gate_observation, _gate_recorded_status,
    _is_fast_transient_build_lock, _resolve_agent_gates_bin, _resolve_env_file_path,
    _run_v2_gate, _sha256_file, _source_identity_matches, _V2GateExecution,
    _workspace_relative_candidate,
)
from tilefoundry_ascend.agent.specs.anchor_specs import (
    default_source_root, get_operator_module_relpath, get_operator_task, parse_simple_env_file,
)
from tilefoundry_ascend.ascend.json_record import JsonRecordMixin

logger = logging.getLogger(__name__)

_CANDIDATE_KERNEL_STAGES = frozenset({"kernel", "compile", "cache_idempotence"})
_CANDIDATE_KERNEL_DETAIL_SIGNALS = (
    "exceeded vf stack size", "ascend compilation failed", "bisheng", "ld.lld",
    "undefined symbol", "tvmerror", "l3 cache idempotence violation", "syntaxerror",
    "nameerror", "typeerror", "attributeerror", "keyerror", "valueerror", "importerror",
    "ub overflow", "out of ubuf", "kernel launch",
)
_INFRA_KERNEL_DETAIL_EXCLUDES = (
    "device offline", "device unavailable", "npu-smi", "drverror", "e39999", "ez9999",
    "aclrtgetdevice", "timeout", "reference cache", "no space left", "permission denied",
    "segmentation fault", "bus error",
)


@dataclass(frozen=True)
class L3AcceptanceResult:
    """Outcome of AgentGates L3 full-contract acceptance verification."""

    passed: bool
    run_id: str = ""
    receipt_path: str = ""
    elapsed_sec: float = 0.0
    error_message: str = ""
    failure_kind: str = ""
    failure_category: str = ""
    pending: bool = False
    execution_id: str = ""
    candidate_sha256: str = ""
    target_path: str = ""
    device_id: str = ""
    state: str = ""
    execution_terminal: bool = True
    blocked_env_category: str = ""
    failed_case_nums: tuple[int, ...] = ()
    failure_stage: str = ""


@dataclass(frozen=True)
class L4CannbenchResult(JsonRecordMixin):
    """Outcome and per-case telemetry of official L4 tilefoundry-npu-cannbench gate."""

    passed: bool
    avg_speedup: float = 0.0
    score: float = 0.0
    per_case_speedups: dict[int, float] = field(default_factory=dict)
    per_case_elapsed_us: dict[int, float] = field(default_factory=dict)
    run_id: str = ""
    receipt_path: str = ""
    report_dir: str = ""
    error_message: str = ""
    failure_kind: str = ""
    execution_id: str = ""
    official_report_valid: bool = False
    all_cases_passed: bool = False
    all_timings_valid: bool = False
    case_count: int = 0
    passed_cases: int = 0
    device_id: str = ""
    state: str = ""
    execution_terminal: bool = True
    blocked_env_category: str = ""
    device_rejected: bool = False
    target_path: str = ""
    candidate_sha256: str = ""
    attempts: tuple[dict[str, str], ...] = ()


def _l3_evidence_payload(obs: Mapping[str, object]) -> Mapping[str, object]:
    return next((c for k in ("l3Evidence", "evidence") if isinstance(c := obs.get(k), Mapping)), obs)


def _l3_signed_failure_category(obs: Mapping[str, object], ev: Mapping[str, object]) -> str:
    """Return the adapter-signed category, falling back to the structured kind."""
    for c in (obs, ev):
        if (raw := c.get("failureCategory")) is not None and str(raw).strip():
            return str(raw).strip().lower()
    st = ev.get("failure")
    return str(st.get("kind") or "").strip().lower() if isinstance(st, Mapping) else FAILURE_KIND_INFRASTRUCTURE


def _l3_structured_failure(obs: Mapping[str, object], ev: Mapping[str, object]) -> Mapping[str, object] | None:
    return next((st for c in (ev, obs) if isinstance(st := c.get("failure"), Mapping) and st), None)


def _is_seq(val: object) -> bool:
    return isinstance(val, Sequence) and not isinstance(val, (str, bytes))


def _is_collateral_missing_case_row(row: Mapping[str, object], failed_nodes: set[str]) -> bool:
    return str(row.get("kind") or "").strip().lower() == "coverage" and str(row.get("category") or "").strip().lower() == "missing_case" and str(row.get("node") or "").strip() in failed_nodes


def _l3_candidate_numeric_failure(marker: Mapping[str, object], ev: Mapping[str, object], raw_cat: str) -> bool:
    """Whether a producer marker binds a numeric regression to this candidate."""
    expected = {"kind": "candidate", "category": "numeric", "role": "candidate", "stage": "comparison"}
    if raw_cat != "numeric" or any(str(marker.get(k) or "").strip().lower() != v for k, v in expected.items()):
        return False
    node_name, case_id, cmps = marker.get("node"), marker.get("case"), marker.get("comparisons")
    if not str(marker.get("comparator") or "").strip() or not isinstance(node_name, str) or not node_name or case_id is None:
        return False
    if not _is_seq(cmps) or not any(isinstance(r, Mapping) and r.get("passed") is False for r in cmps):
        return False
    failures = ev.get("failures")
    if _is_seq(failures):
        rows = [r for r in failures if isinstance(r, Mapping)]
        cnodes = {str(r.get("node") or "").strip() for r in [marker, *rows] if str(r.get("kind") or "").strip().lower() == "candidate" and str(r.get("node") or "").strip()}
        if any(str(r.get("kind") or "").strip().lower() in {"infrastructure", "coverage"} and not _is_collateral_missing_case_row(r, cnodes) for r in rows):
            return False
    nodes = ev.get("nodes")
    if not _is_seq(nodes):
        return False
    for node in nodes:
        if isinstance(node, Mapping) and node.get("name") == node_name:
            cases = node.get("cases")
            return str(node.get("status") or "").strip().lower() in {"failed", "error"} and isinstance(cases, Mapping) and str(cases.get(str(case_id)) or "").strip().lower() in {"failed", "error"}
    return False


def _is_candidate_kernel_marker(row: Mapping[str, object], *, node_records: Mapping[str, Mapping[str, object]]) -> bool:
    role, stage = str(row.get("role") or "").strip().lower(), str(row.get("stage") or "").strip().lower()
    if role == "baseline" or stage not in _CANDIDATE_KERNEL_STAGES:
        return False
    node = node_records.get(nn) if isinstance(nn := row.get("node"), str) and nn else None
    if not isinstance(node, Mapping) or str(node.get("status") or "").strip().lower() not in {"failed", "error"}:
        return False
    if (cid := row.get("case")) is not None and (not isinstance(cases := node.get("cases"), Mapping) or str(cases.get(str(cid)) or "").strip().lower() not in {"failed", "error"}):
        return False
    detail = f"{row.get('exceptionType') or ''} {row.get('detail') or ''}".lower()
    if any(excl in detail for excl in _INFRA_KERNEL_DETAIL_EXCLUDES):
        return False
    return (str(row.get("kind") or "").strip().lower() == "candidate" and (role == "candidate" or stage in {"kernel", "compile"})) or any(s in detail for s in _CANDIDATE_KERNEL_DETAIL_SIGNALS)


def _l3_candidate_kernel_failure(structured: Mapping[str, object], ev: Mapping[str, object], raw_cat: str) -> bool:
    """Return True when L3 evidence unambiguously binds a candidate compile/kernel failure."""
    if raw_cat not in {"candidate", "infrastructure", "environment"} or not _is_seq(ev.get("nodes")):
        return False
    raw_failures = ev.get("failures")
    rows = ([r for r in raw_failures if isinstance(r, Mapping)] if _is_seq(raw_failures) else []) or [structured]
    node_records = {str(n.get("name")): n for n in ev["nodes"] if isinstance(n, Mapping) and n.get("name")}
    cnodes = {
        str(r.get("node") or "").strip() for r in rows
        if str(r.get("node") or "").strip()
        and (str(r.get("kind") or "").strip().lower() == "candidate" or _is_candidate_kernel_marker(r, node_records=node_records))
    }
    matched = False
    for row in rows:
        if _is_candidate_kernel_marker(row, node_records=node_records):
            matched = True
        elif not _is_collateral_missing_case_row(row, cnodes) and str(row.get("kind") or "").strip().lower() in {"infrastructure", "coverage"}:
            return False
    return matched


def _l3_failure_kind(obs: Mapping[str, object], ev: Mapping[str, object], coverage: Mapping[str, object] | None) -> str:
    """Classify a candidate defect only from explicit signed producer evidence."""
    intr, w_errs = ev.get("interruption"), ev.get("writeErrors")
    if isinstance(intr, Mapping) and (str(intr.get("errorCategory") or "").strip() or intr.get("signal") or intr.get("interrupted")):
        return FAILURE_KIND_INFRASTRUCTURE
    if (_is_seq(w_errs) and len(w_errs) > 0) or str(ev.get("phase") or "").strip().lower() != "finished":
        return FAILURE_KIND_INFRASTRUCTURE
    timing, interp, ref_c = ev.get("timing"), ev.get("interpreter"), ev.get("referenceCache")
    if (isinstance(timing, Mapping) and not timing.get("finishedAt")) or (isinstance(interp, Mapping) and interp.get("match") is False):
        return FAILURE_KIND_INFRASTRUCTURE
    if (isinstance(ref_c, Mapping) and (ref_c.get("errors") or (ref_c.get("required") is True and ref_c.get("usable") is False))) or (isinstance(coverage, Mapping) and coverage.get("referenceCacheProblem")):
        return FAILURE_KIND_INFRASTRUCTURE
    raw, st = _l3_signed_failure_category(obs, ev), _l3_structured_failure(obs, ev)
    if st is not None and (_l3_candidate_numeric_failure(st, ev, raw) or _l3_candidate_kernel_failure(st, ev, raw)):
        return FAILURE_KIND_CANDIDATE
    return FAILURE_KIND_INFRASTRUCTURE


def classify_l3_observation_failure(observation: Mapping[str, object]) -> tuple[str, bool]:
    """Return ``(failure_kind, pending)`` for a signed L3 gate observation."""
    ev = _l3_evidence_payload(observation)
    cov = ev.get("coverage")
    kind = _l3_failure_kind(observation, ev, cov if isinstance(cov, Mapping) else None)
    return kind, kind != FAILURE_KIND_CANDIDATE


def _parse_l3_case_num(raw: object) -> int | None:
    if not isinstance(raw, bool) and isinstance(raw, int):
        return raw if raw >= 0 else None
    return int(raw.strip()) if isinstance(raw, str) and raw.strip().isdigit() else None


def _extract_l3_failed_case_nums(ev: Mapping[str, object]) -> tuple[int, ...]:
    """Extract sorted deduplicated failed or errored case indices from L3 evidence."""
    found: set[int] = set()

    def _add(raw: object) -> None:
        if (parsed := _parse_l3_case_num(raw)) is not None:
            found.add(parsed)

    if isinstance(ev.get("failure"), Mapping):
        _add(ev["failure"].get("case"))
    if _is_seq(ev.get("failures")):
        for item in ev["failures"]:
            if isinstance(item, Mapping):
                _add(item.get("case"))
    if _is_seq(ev.get("nodes")):
        for node in ev["nodes"]:
            if not isinstance(node, Mapping):
                continue
            if isinstance(node.get("failure"), Mapping):
                _add(node["failure"].get("case"))
            if isinstance(cases := node.get("cases"), Mapping):
                for ck, cs in cases.items():
                    if str(cs or "").strip().lower() in {"failed", "error"}:
                        _add(ck)
    return tuple(sorted(found))


def _is_coverage_only_failure(row: object) -> bool:
    return not isinstance(row, Mapping) or not row or str(row.get("category") or "").strip().lower() == "missing_case" or str(row.get("stage") or "").strip().lower() == "coverage"


def _select_primary_l3_failure(ev: Mapping[str, object]) -> Mapping[str, object] | None:
    if _is_seq(nodes := ev.get("nodes")):
        for n in nodes:
            if isinstance(n, Mapping) and isinstance(n.get("failure"), Mapping) and not _is_coverage_only_failure(n["failure"]):
                return n["failure"]
    if _is_seq(fseq := ev.get("failures")):
        for item in fseq:
            if isinstance(item, Mapping) and not _is_coverage_only_failure(item):
                return item
    if isinstance(ev_f := ev.get("failure"), Mapping) and ev_f:
        return ev_f
    return next((item for item in fseq if isinstance(item, Mapping) and item), None) if _is_seq(fseq) else None


def _l3_failure_stage(ev: Mapping[str, object]) -> str:
    p = _select_primary_l3_failure(ev)
    return str(p.get("stage") or "").strip().lower() if isinstance(p, Mapping) else ""


def l3_failure_result_from_observation(
    observation: Mapping[str, object],
    *,
    run_id: str = "",
    receipt_path: str = "",
    execution_id: str = "",
    candidate_sha256: str = "",
    target_path: str = "",
    device_id: str = "",
    state: str = "",
    execution_terminal: bool = True,
    blocked_env_category: str = "",
) -> L3AcceptanceResult:
    """Map an adapter L3 observation to the campaign failure result."""
    ev = _l3_evidence_payload(observation)
    kind, pending = classify_l3_observation_failure(observation)
    return L3AcceptanceResult(
        passed=False, run_id=run_id, receipt_path=receipt_path, failure_kind=kind,
        failure_category=_l3_signed_failure_category(observation, ev), pending=pending,
        execution_id=execution_id, candidate_sha256=candidate_sha256, target_path=target_path,
        device_id=device_id, state=state, execution_terminal=execution_terminal,
        blocked_env_category=blocked_env_category, failed_case_nums=_extract_l3_failed_case_nums(ev),
        failure_stage=_l3_failure_stage(ev),
    )


def _format_l3_failure_detail(row: Mapping[str, object], obs: Mapping[str, object], ev: Mapping[str, object]) -> str:
    base_text = str(row.get("failures") or obs.get("error") or "no detail")
    failure = _select_primary_l3_failure(ev)
    if not isinstance(failure, Mapping):
        return base_text
    parts: list[str] = []
    header = " ".join(b for b in (
        f"case={failure['case']}" if failure.get("case") is not None else "",
        f"stage={failure['stage']}" if failure.get("stage") else "",
        f"type={failure['exceptionType']}" if failure.get("exceptionType") else "",
    ) if b)
    if header:
        parts.append(header)
    detail_raw = str(failure.get("detail") or "").strip()
    if detail_raw:
        idx = detail_raw.find("Compiler output:")
        if idx >= 0:
            parts.append(detail_raw[idx : idx + 900])
        else:
            parts.append(detail_raw if len(detail_raw) <= 900 else f"{detail_raw[:420]} ... {detail_raw[-450:]}")
    if _is_seq(cmps := failure.get("comparisons")):
        fc = next((c for c in cmps if isinstance(c, Mapping) and c.get("passed") is False), None) or next(
            (c for c in cmps if isinstance(c, Mapping) and (str(c.get("message") or "").strip() or str(c.get("detail") or "").strip())),
            None,
        )
        if isinstance(fc, Mapping):
            bits = []
            for key in ("name", "max_abs_diff", "max_rel_diff", "atol", "rtol", "detail", "message"):
                val = fc.get(key)
                if val is not None and str(val).strip():
                    vs = str(val).strip()
                    bits.append(f"{key}={vs[:600] if key in ('message', 'detail') and len(vs) > 600 else vs}")
            if bits:
                parts.append("comparison(" + ", ".join(bits) + ")")
    extracted = " | ".join(p for p in parts if p)
    return f"{base_text} | {extracted}" if extracted else base_text


def run_l3_full_contract_via_agent_gates(
    *, operator_name: str | None = None, candidate_module_path: str | Path | None = None,
    source_root: str | Path | None = None, env_file: str | Path | None = None,
    device_id: str = "1", device_pool: Sequence[str] = (), timeout_sec: int = DEFAULT_L3_PROMOTION_WAIT_SEC,
    resume_execution_id: str = "", resume_run_id: str = "", on_execution_started: Callable[[str, str], None] | None = None,
) -> L3AcceptanceResult:
    """Run the L3 promotion gate through AgentGates and validate its signed promotion evidence."""
    env_vals: dict[str, str] | None = None
    try:
        env_vals = parse_simple_env_file(_resolve_env_file_path(env_file))
    except FileNotFoundError:
        pass
    result_device, started = str(device_id or ""), time.monotonic()

    def _fail(msg: str, ex: _V2GateExecution | None = None, blocked_cat: str = "") -> L3AcceptanceResult:
        return L3AcceptanceResult(
            False, ex.run_id if ex else "", ex.receipt_path if ex else "", time.monotonic() - started,
            msg[-1000:], FAILURE_KIND_INFRASTRUCTURE, execution_id=ex.execution_id if ex else "",
            device_id=(ex.device_id if ex else "") or result_device, state=ex.state if ex else "",
            execution_terminal=ex.execution_terminal if ex else True, blocked_env_category=blocked_cat,
        )

    if operator_name is None or candidate_module_path is None:
        return _fail("L3 promotion requires an operator name and candidate module path")
    try:
        target_task = get_operator_task(operator_name)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        return _fail(f"L3 promotion target resolution failed: {exc}")
    root, candidate = (Path(source_root).resolve() if source_root is not None else default_source_root()), Path(candidate_module_path).resolve()
    if (relative := _workspace_relative_candidate(root, candidate)) is None:
        return _fail(f"L3 candidate {candidate} is outside workspace source root {root}")
    if not candidate.is_file():
        return _fail(f"L3 candidate module {candidate} does not exist")
    candidate_digest, target_path = _sha256_file(candidate), str(candidate)
    try:
        ag_bin = _resolve_agent_gates_bin(root)
        workspace = _agent_gates_workspace_name(root)
    except FileNotFoundError:
        raise
    except (OSError, ValueError) as exc:
        return _fail(str(exc))

    run_v2 = _run_v2_gate
    order_fn = _call_order_candidates_by_idle_preflight
    blocked_env_category = ""
    if resume_execution_id or resume_run_id:
        execution = run_v2(
            agent_gates_bin=ag_bin, root=root, workspace=workspace, gate_id=AGENT_GATES_L3_PROMOTION_GATE,
            target=target_task, candidate_path=relative, timeout_sec=timeout_sec, device_id=result_device,
            resume_execution_id=resume_execution_id, resume_run_id=resume_run_id, on_execution_started=on_execution_started,
        )
        blocked_env_category = _blocked_env_category(execution.report, (AGENT_GATES_L3_PROMOTION_GATE,))
    else:
        rejected_cards, initial_device_id = set(), result_device
        occ_backoff, total_occ_sleep, execution_opt = (15.0, 30.0, 45.0), 0.0, None
        for pass_idx in range(4):
            candidates = order_fn(initial_device_id, device_pool, excluded=rejected_cards, env_vals=env_vals)
            if not candidates:
                if not rejected_cards and initial_device_id:
                    candidates = (initial_device_id,)
                else:
                    break
            saw_occ = False
            for idx, card in enumerate(candidates):
                result_device = card
                attempt_exec = run_v2(
                    agent_gates_bin=ag_bin, root=root, workspace=workspace, gate_id=AGENT_GATES_L3_PROMOTION_GATE,
                    target=target_task, candidate_path=relative, timeout_sec=timeout_sec, device_id=card,
                    resume_execution_id="", resume_run_id="", on_execution_started=on_execution_started,
                )
                execution_opt = attempt_exec
                blocked_env_category = _blocked_env_category(attempt_exec.report, (AGENT_GATES_L3_PROMOTION_GATE,))
                if attempt_exec.device_rejected or blocked_env_category == "occupancy":
                    if attempt_exec.device_rejected and card:
                        rejected_cards.add(card)
                    else:
                        saw_occ = True
                    if idx + 1 < len(candidates):
                        logger.warning("L3_CONTAMINATION_RETEST task=%s tainted_device=%s reason=%s next_card=%s", target_task, card, "device_rejected" if attempt_exec.device_rejected else "occupancy", candidates[idx + 1])
                        continue
                else:
                    break
            else:
                if saw_occ and pass_idx < 3 and total_occ_sleep < 120.0 and order_fn(initial_device_id, device_pool, excluded=rejected_cards, env_vals=env_vals):
                    wait_sec = occ_backoff[min(pass_idx, len(occ_backoff) - 1)]
                    logger.info("L3_OCCUPANCY_BACKOFF task=%s pass=%d/4 sleep=%.1fs last_card=%s", target_task, pass_idx + 1, wait_sec, result_device)
                    time.sleep(wait_sec); total_occ_sleep += wait_sec
                    continue
            break
        assert execution_opt is not None
        execution = execution_opt

    if execution.transport_error:
        return _fail(execution.transport_error, execution, blocked_env_category)
    row, observation = _gate_observation(execution.report, AGENT_GATES_L3_PROMOTION_GATE)
    if row is None:
        return _fail("L3 signed receipt has no promotion gate observation", execution, blocked_env_category)
    evidence = _l3_evidence_payload(observation)
    coverage = evidence.get("coverage") if isinstance(evidence.get("coverage"), Mapping) else None
    recorded_status = _gate_recorded_status(row)
    failed_case_nums, failure_stage = _extract_l3_failed_case_nums(evidence), _l3_failure_stage(evidence)

    def _reject(msg: str, failure_kind: str) -> L3AcceptanceResult:
        return L3AcceptanceResult(
            False, execution.run_id, execution.receipt_path, time.monotonic() - started, msg[:1800], failure_kind,
            _l3_signed_failure_category(observation, evidence), failure_kind != FAILURE_KIND_CANDIDATE, execution.execution_id,
            candidate_digest, target_path, execution.device_id or result_device, execution.state,
            execution.execution_terminal, blocked_env_category, failed_case_nums, failure_stage,
        )

    if recorded_status != "PASS":
        return _reject(f"L3 gate recorded status {recorded_status or 'missing'}: {_format_l3_failure_detail(row, observation, evidence)}", _l3_failure_kind(observation, evidence, coverage))
    if not execution.verified:
        return _reject(f"L3 receipt did not verify: {execution.verification.get('failures') or 'unsatisfied'}", FAILURE_KIND_INFRASTRUCTURE)
    if not _source_identity_matches(execution.verification, root):
        return _reject("L3 verified source identity does not match requested source", FAILURE_KIND_INFRASTRUCTURE)
    schema, profile = str(evidence.get("schema", "")), str(evidence.get("profile", ""))
    if schema != "tilefoundry-l3-evidence/v1" or profile != "l3-promotion":
        return _reject(f"L3 evidence schema/profile mismatch: schema={schema or 'missing'} profile={profile or 'missing'}", FAILURE_KIND_INFRASTRUCTURE)
    target = evidence.get("target")
    if not (declared_task := target.get("task") if isinstance(target, Mapping) else None) or str(declared_task) != target_task:
        return _reject(f"L3 evidence target task {declared_task or 'missing'} does not match {target_task}", FAILURE_KIND_INFRASTRUCTURE)
    if not (target_module := target.get("modulePath") if isinstance(target, Mapping) else None) or Path(str(target_module)).resolve() != Path(target_path):
        return _reject("L3 evidence target does not match requested target module", FAILURE_KIND_INFRASTRUCTURE)
    declared = evidence.get("candidate")
    declared_path = declared.get("modulePath") if isinstance(declared, Mapping) else None
    declared_digest = declared.get("sha256") if isinstance(declared, Mapping) else None
    if not declared_path or Path(str(declared_path)).resolve() != candidate:
        return _reject("L3 evidence candidate path does not match the promoted candidate", FAILURE_KIND_INFRASTRUCTURE)
    if str(declared_digest or "") != candidate_digest:
        return _reject("L3 evidence candidate digest does not match the promoted candidate", FAILURE_KIND_INFRASTRUCTURE)
    if not isinstance(interp := evidence.get("interpreter"), Mapping) or interp.get("match") is not True:
        return _reject("L3 evidence interpreter did not match the adapter-resolved interpreter", FAILURE_KIND_INFRASTRUCTURE)
    if coverage is None or coverage.get("qualified") is not True:
        return _reject("L3 evidence did not qualify full contract coverage", _l3_failure_kind(observation, evidence, coverage))
    nodes = coverage.get("nodes") if isinstance(coverage.get("nodes"), Mapping) else {}
    cases = coverage.get("cases") if isinstance(coverage.get("cases"), Mapping) else {}
    if any(nodes.get(k) for k in ("missing", "failed", "skipped", "blocked")) or cases.get("missing") or cases.get("unexpected") or coverage.get("referenceCacheProblem"):
        return _reject(f"L3 evidence coverage is incomplete: {dict(nodes)} / {dict(cases)} / referenceCacheProblem={coverage.get('referenceCacheProblem')}", _l3_failure_kind(observation, evidence, coverage))
    exp_cases, comp_cases = _as_positive_int(cases.get("expected")), _as_positive_int(cases.get("completed"))
    if exp_cases <= 0 or comp_cases != exp_cases:
        return _reject(f"L3 evidence case coverage incomplete: {comp_cases}/{exp_cases}", _l3_failure_kind(observation, evidence, coverage))
    return L3AcceptanceResult(
        True, execution.run_id, execution.receipt_path, time.monotonic() - started, execution_id=execution.execution_id,
        candidate_sha256=candidate_digest, target_path=target_path, device_id=execution.device_id or result_device,
        state=execution.state, execution_terminal=execution.execution_terminal,
    )


def _extract_l4_metrics(parsed: Mapping[str, object], target_task: str) -> tuple[float, float, str, dict[int, float], dict[int, float]]:
    """Extract L4 aggregate and per-case metrics from signed gate observations."""
    avg_speedup, score, report_dir = 0.0, 0.0, ""
    per_case_speedups: dict[int, float] = {}
    per_case_elapsed: dict[int, float] = {}
    for gate_row in parsed.get("gates", ()):
        if not isinstance(gate_row, Mapping) or gate_row.get("gateId") != AGENT_GATES_L4_CANNBENCH_GATE:
            continue
        obs = gate_row.get("observations") or {}
        if not isinstance(obs, Mapping):
            continue
        avg_speedup, score = float(obs.get("avgSpeedup", 0.0)), float(obs.get("score", 0.0))
        for res_row in obs.get("results", ()):
            if not isinstance(res_row, Mapping) or (target_task and res_row.get("task") != target_task):
                continue
            report_dir = str(res_row.get("reportDir", ""))
            for case_row in res_row.get("cases", ()):
                if isinstance(case_row, Mapping):
                    cid_str = str(case_row.get("case_id", ""))
                    if "_" in cid_str and cid_str.rsplit("_", 1)[-1].isdigit():
                        cnum = int(cid_str.rsplit("_", 1)[-1])
                        per_case_speedups[cnum] = float(case_row.get("speedup") or 0.0)
                        per_case_elapsed[cnum] = float(case_row.get("elapsed_us") or 0.0)
    return avg_speedup, score, report_dir, per_case_speedups, per_case_elapsed


def _l4_failure(
    error_message: str, *, failure_kind: str = FAILURE_KIND_INFRASTRUCTURE, execution_id: str = "",
    run_id: str = "", receipt_path: str = "", observation: Mapping[str, object] | None = None,
    device_id: str = "", state: str = "", execution_terminal: bool = True,
    blocked_env_category: str = "", device_rejected: bool = False, target_path: str = "",
    candidate_sha256: str = "", attempts: tuple[dict[str, str], ...] = (),
) -> L4CannbenchResult:
    obs = observation or {}
    return L4CannbenchResult(
        passed=False, run_id=run_id, receipt_path=receipt_path, error_message=error_message[-1000:],
        failure_kind=failure_kind, execution_id=execution_id,
        official_report_valid=obs.get("officialReportValid") is True,
        all_cases_passed=obs.get("allCasesPassed") is True, all_timings_valid=obs.get("allTimingsValid") is True,
        case_count=_as_positive_int(obs.get("caseCount")), passed_cases=_as_positive_int(obs.get("passedCases")),
        device_id=device_id, state=state, execution_terminal=execution_terminal,
        blocked_env_category=blocked_env_category, device_rejected=device_rejected,
        target_path=target_path, candidate_sha256=candidate_sha256, attempts=attempts,
    )


def _l4_candidate_matches(raw: object, expected: str, root: Path) -> bool:
    text = str(raw or "").strip()
    if not text:
        return False
    if text.endswith(expected) and (text == expected or text.endswith("/" + expected)):
        return True
    path = Path(text)
    return path.is_absolute() and path.resolve() == (root / expected).resolve()


def _validate_l4_report(execution: _V2GateExecution, *, root: Path, target_task: str, expected_candidate: str) -> tuple[str, str]:
    """Validate signed L4 evidence; return ``(error_message, failure_kind)``."""
    if not execution.verified:
        return (f"L4 receipt did not verify: {execution.verification.get('failures') or 'unsatisfied'}", FAILURE_KIND_INFRASTRUCTURE)
    if not _source_identity_matches(execution.verification, root):
        return ("L4 verified source identity does not match requested source", FAILURE_KIND_INFRASTRUCTURE)
    row, obs = _gate_observation(execution.report, AGENT_GATES_L4_CANNBENCH_GATE)
    if row is None:
        return ("L4 signed receipt has no cannbench gate observation", FAILURE_KIND_INFRASTRUCTURE)
    status = _gate_recorded_status(row)
    if status != "PASS":
        return (f"L4 gate recorded status {status or 'missing'}: {row.get('failures') or obs.get('error') or 'no detail'}", FAILURE_KIND_CANDIDATE if status == "FAIL" else FAILURE_KIND_INFRASTRUCTURE)
    if bad := [n for n in ("officialReportValid", "allCasesPassed", "allTimingsValid") if obs.get(n) is not True]:
        return (f"L4 official report validation failed: {', '.join(bad)}", FAILURE_KIND_CANDIDATE)
    case_count, passed_raw = _as_positive_int(obs.get("caseCount")), obs.get("passedCases")
    passed_cases, timed_cases = _as_positive_int(passed_raw), _as_positive_int(obs.get("timedCases"))
    if case_count <= 0:
        return ("L4 signed observation reports no official cases", FAILURE_KIND_INFRASTRUCTURE)
    if isinstance(passed_raw, bool) or not isinstance(passed_raw, (int, float)):
        return ("L4 signed observation has no valid passed-case count", FAILURE_KIND_INFRASTRUCTURE)
    if passed_cases != case_count:
        return (f"L4 passed {passed_cases}/{case_count} official cases", FAILURE_KIND_CANDIDATE)
    if timed_cases != case_count:
        return (f"L4 timed {timed_cases}/{case_count} official cases", FAILURE_KIND_INFRASTRUCTURE)
    results = obs.get("results")
    if not _is_seq(results):
        return ("L4 signed observation has no per-target results", FAILURE_KIND_INFRASTRUCTURE)
    matched = 0
    for res_row in results:
        if not isinstance(res_row, Mapping) or (target_task and res_row.get("task") != target_task):
            continue
        matched += 1
        if res_row.get("reportValid") is not True:
            return (f"L4 official report is not valid for task {target_task or res_row.get('task')}", FAILURE_KIND_CANDIDATE)
        if not str(res_row.get("reportDir", "")).strip():
            return ("L4 official report directory is missing", FAILURE_KIND_INFRASTRUCTURE)
        if not _l4_candidate_matches(res_row.get("candidate"), expected_candidate, root):
            return (f"L4 receipt candidate does not match {expected_candidate}", FAILURE_KIND_INFRASTRUCTURE)
    if matched == 0:
        return (f"L4 signed observation has no result for task {target_task or '<any>'}", FAILURE_KIND_INFRASTRUCTURE)
    return ("", "")


def run_l4_official_cannbench_via_agent_gates(
    *, operator_name: str | None = None, candidate_module_path: str | Path | None = None,
    workspace: str | None = None, source_root: str | Path | None = None, env_file: str | Path | None = None,
    device_id: str = "1", timeout_sec: int = DEFAULT_L4_CANNBENCH_WAIT_SEC, resume_execution_id: str = "",
    resume_run_id: str = "", on_execution_started: Callable[[str, str], None] | None = None,
) -> L4CannbenchResult:
    """Run official tilefoundry-npu-cannbench through AgentGates and verify its signed receipt."""
    _ = env_file
    result_device = str(device_id or "")
    if operator_name is None:
        return _l4_failure("L4 cannbench requires a canonical operator name", device_id=result_device)
    try:
        target_task = get_operator_task(operator_name)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        return _l4_failure(f"L4 cannbench target resolution failed: {exc}", device_id=result_device)
    root = Path(source_root).resolve() if source_root is not None else default_source_root()
    expected_path = (Path(candidate_module_path).resolve() if candidate_module_path is not None else (root / get_operator_module_relpath(operator_name)).resolve())
    if (relative := _workspace_relative_candidate(root, expected_path)) is None:
        return _l4_failure(f"L4 candidate {expected_path} is outside workspace source root {root}", device_id=result_device)
    try:
        candidate_digest = _sha256_file(expected_path)
    except OSError as exc:
        return _l4_failure(f"L4 candidate module {expected_path} cannot be read: {exc}", device_id=result_device, target_path=str(expected_path))
    try:
        ag_bin = _resolve_agent_gates_bin(root)
        resolved_workspace = workspace or _agent_gates_workspace_name(root)
    except (OSError, ValueError) as exc:
        return _l4_failure(str(exc), device_id=result_device)

    run_v2, backoff = _run_v2_gate, (10.0, 15.0, 20.0, 25.0, 30.0)
    max_attempts, execution_opt = (1 if (resume_execution_id or resume_run_id) else 13), None
    for retry_idx in range(max_attempts):
        execution_opt = run_v2(
            agent_gates_bin=ag_bin, root=root, workspace=resolved_workspace, gate_id=AGENT_GATES_L4_CANNBENCH_GATE,
            target=target_task, candidate_path=relative, timeout_sec=timeout_sec, device_id=result_device,
            resume_execution_id=resume_execution_id, resume_run_id=resume_run_id, on_execution_started=on_execution_started,
        )
        if retry_idx >= max_attempts - 1 or not _is_fast_transient_build_lock(execution_opt):
            break
        ws = backoff[min(retry_idx, len(backoff) - 1)]
        logger.info("L4_GATE_RETRY_BUILD_LOCK op=%s retry=%d/12 wait=%.1fs", operator_name, retry_idx + 1, ws)
        time.sleep(ws)
    assert execution_opt is not None
    execution = execution_opt
    _row, observation = _gate_observation(execution.report, AGENT_GATES_L4_CANNBENCH_GATE)
    blocked_env_category = _blocked_env_category(execution.report)

    def _l4_exec_err(msg: str, *, kind: str = FAILURE_KIND_INFRASTRUCTURE, obs: Mapping[str, object] | None = None, rejected: bool = False) -> L4CannbenchResult:
        return _l4_failure(
            msg, failure_kind=kind, execution_id=execution.execution_id, run_id=execution.run_id,
            receipt_path=execution.receipt_path, observation=obs, device_id=execution.device_id or result_device,
            state=execution.state, execution_terminal=execution.execution_terminal, blocked_env_category=blocked_env_category,
            device_rejected=rejected, target_path=str(expected_path), candidate_sha256=candidate_digest,
        )

    if execution.transport_error:
        return _l4_exec_err(execution.transport_error, rejected=execution.device_rejected)
    error, failure_kind = _validate_l4_report(execution, root=root, target_task=target_task, expected_candidate=relative)
    if error:
        return _l4_exec_err(error, kind=failure_kind, obs=observation)
    avg_speedup, score, report_dir, speedups, elapsed = _extract_l4_metrics(execution.report, target_task)
    return L4CannbenchResult(
        True, avg_speedup, score, speedups, elapsed, execution.run_id, execution.receipt_path, report_dir,
        execution_id=execution.execution_id, official_report_valid=True, all_cases_passed=True, all_timings_valid=True,
        case_count=_as_positive_int(observation.get("caseCount")), passed_cases=_as_positive_int(observation.get("passedCases")),
        device_id=execution.device_id or result_device, state=execution.state, execution_terminal=execution.execution_terminal,
        target_path=str(expected_path), candidate_sha256=candidate_digest,
    )
