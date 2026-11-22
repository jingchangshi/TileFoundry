"""L3 promotion target contracts and machine-readable acceptance evidence.

The suite resolves one contract per canonical target, records every pytest node
and case transition to an atomically replaced JSON file, and checks full
coverage. Rewriting after each case leaves the last completed case and the
in-flight node after a SIGKILL or deadline; unknown values are ``null``, not
zero.

Notes:
    Environment variables are set by the AgentGates
    ``tilefoundry-npu-l3-promotion`` gate and are never exported into a parent
    shell.

    ``TF_L3_EVIDENCE_PATH``
        Absolute path of the evidence JSON file to write.
    ``TF_L3_TARGET_OPERATOR``
        Canonical target, e.g. ``level4/attention_residual`` or ``mla``.
    ``TF_L3_TARGET_MODULE_PATH``
        Candidate module used for this run. Its resolved path and SHA-256 are
        recorded and checked against the module the tests actually import.
    ``TF_L3_EXPECTED_PYTHON``
        Interpreter the gate resolved from the adapter. A mismatch is an
        environment block, not a candidate failure.
    ``TF_L3_CANNBENCH_REPO``
        CANNBench checkout used for comparator and task contracts when the
        official ``npu_compare`` module is not installed.
    ``TF_L3_CACHE_EXPECTATION``
        ``cold``, ``warm`` or ``any`` (default). ``cold`` requires at least one
        oracle miss with every miss persisted; later nodes may legitimately hit
        entries written by an earlier node. ``warm`` requires every oracle
        lookup to hit and nothing to be written.
    ``CANNBENCH_CACHE_MODE``
        ``off``, ``optional`` or ``required``. Required mode turns an unusable
        reference cache into an environment block and a failed cache write into
        a hard error.
"""

from __future__ import annotations

import hashlib
import json
import os
import signal
import sys
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType
from typing import TypeVar

import yaml

from tilefoundry_ascend.agent.campaign.anchor_specs import CannbenchOpsRegistry

EVIDENCE_ENV = "TF_L3_EVIDENCE_PATH"
PROFILE_ENV = "TF_L3_PROFILE"
TARGET_ENV = "TF_L3_TARGET_OPERATOR"
MODULE_ENV = "TF_L3_TARGET_MODULE_PATH"
EXPECTED_PYTHON_ENV = "TF_L3_EXPECTED_PYTHON"
CANNBENCH_REPO_ENV = "TF_L3_CANNBENCH_REPO"
CACHE_EXPECTATION_ENV = "TF_L3_CACHE_EXPECTATION"

SCHEMA = "tilefoundry-l3-evidence/v1"
PROFILE = "l3-promotion"
BLOCKED_EXIT_CODE = 75
DETAIL_LIMIT = 800

BENCHMARKS_DIR = Path(__file__).resolve().parent
AR_DIR = BENCHMARKS_DIR / "attention_residual"
MLA_DIR = BENCHMARKS_DIR / "mla"
SUPPLEMENTAL_MLA_PATH = BENCHMARKS_DIR / "cases" / "supplemental_mla.json"

AR_TASK = "level4/attention_residual"
MLA_TASK = "level4/mla"
AR_PARITY_CASES = ("1", "16", "20")
AR_DYNAMIC_CASES = (
    "ar_dynamic_shape_0",
    "ar_dynamic_shape_1",
    "ar_dynamic_cache_single_kernel",
)
MLA_DYNAMIC_CASES = (
    "mla_dynamic_shape_0",
    "mla_dynamic_shape_1",
    "mla_dynamic_cache_single_kernel",
)
AR_NODE = "test_attention_residual_cannbench_cases"
AR_PARITY_NODE = "test_attention_residual_baseline_vs_tuned_parity"
MLA_NODE = "test_mla_cannbench_cases"
MLA_SUPPLEMENTAL_NODE = "test_mla_supplemental_cases"
DYNAMIC_NODE = "test_dynamic_shape_kernel_cache_reuse"
GENERIC_NODE = "test_target_operator_full_cannbench_contract"


class L3ContractError(RuntimeError):
    """Raised when the promotion target contract cannot be resolved."""


FAILURE_KIND_CANDIDATE = "candidate"
FAILURE_KIND_INFRASTRUCTURE = "infrastructure"
FAILURE_KIND_COVERAGE = "coverage"
_FAILURE_PRIORITY = {
    FAILURE_KIND_INFRASTRUCTURE: 0,
    FAILURE_KIND_COVERAGE: 1,
    FAILURE_KIND_CANDIDATE: 2,
}


class CandidateComparisonError(AssertionError):
    """Numeric comparator explicitly returned a deviation for the candidate."""

    l3_failure_classified = True

    def __init__(self, message: str, failure: Mapping[str, object]) -> None:
        super().__init__(message)
        self.failure = dict(failure)


class CoverageContractError(AssertionError):
    """The executed cases do not match the declared contract coverage."""

    l3_failure_classified = True

    def __init__(self, message: str, failure: Mapping[str, object]) -> None:
        super().__init__(message)
        self.failure = dict(failure)


class ClassifiedTestFailure(AssertionError):
    """Aggregate failure whose structured evidence was already recorded."""

    l3_failure_classified = True


_CaseT = TypeVar("_CaseT")
_CacheKeyT = TypeVar("_CacheKeyT")


def verify_kernel_cache_idempotence(
    cases: Sequence[_CaseT],
    run_case_fn: Callable[[_CaseT], object],
    kernel_cache: Mapping[_CacheKeyT, object],
    *,
    perturb_case_fn: Callable[[_CaseT], object] | None = None,
) -> tuple[int, int]:
    """Verify two-pass kernel cache idempotence across replay and outer-axis perturbation."""
    c1 = len(kernel_cache)
    keys1 = set(kernel_cache.keys())
    circuit_breaker = max(1, len(cases)) * 2
    if c1 > circuit_breaker:
        raise ClassifiedTestFailure(
            f"Pass 1 _KERNEL_CACHE runaway growth: {c1} entries across "
            f"{len(cases)} cases (circuit_breaker={circuit_breaker})"
        )
    for case in cases:
        run_case_fn(case)
        if perturb_case_fn is not None:
            perturb_case_fn(case)
    c2 = len(kernel_cache)
    keys2 = set(kernel_cache.keys())
    pass2_delta = c2 - c1
    if pass2_delta != 0 or keys2 != keys1:
        added = len(keys2 - keys1)
        evicted = len(keys1 - keys2)
        raise ClassifiedTestFailure(
            f"L3 cache idempotence violation (Delta|Cache|={pass2_delta}): "
            f"pass1={c1}, pass2={c2}, added={added}, evicted={evicted}"
        )
    return (c1, pass2_delta)


@dataclass(frozen=True, kw_only=True)
class NodeContract:
    """One required pytest node and the case identities it must complete."""

    name: str
    required_cases: tuple[str, ...] = ()


@dataclass(frozen=True, kw_only=True)
class TargetContract:
    """Full L3 promotion contract for one canonical target."""

    operator: str
    task: str
    task_dir: Path
    nodes: tuple[NodeContract, ...]
    case_ids: tuple[str, ...]


@dataclass
class NodeRecord:
    """Mutable per-node evidence accumulated during one pytest session."""

    name: str
    required_cases: tuple[str, ...]
    node_id: str = ""
    status: str = "pending"
    cases: dict[str, str] = field(default_factory=dict)
    details: dict[str, str] = field(default_factory=dict)
    error: str = ""
    started_at: str = ""
    ended_at: str = ""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _truncate(text: str) -> str:
    cleaned = " ".join(str(text).split())
    return cleaned[:DETAIL_LIMIT]


def _failure_base(
    *,
    kind: str,
    category: str,
    stage: str,
    node: str | None,
    case_id: str | None,
    role: str | None,
    comparator: str | None,
    references: Sequence[str],
    comparisons: Sequence[Mapping[str, object]],
    exception_type: str | None,
    detail: str,
) -> dict[str, object]:
    """Build the single structured failure shape shared with both consumers."""
    assert kind in _FAILURE_PRIORITY
    return {
        "kind": kind,
        "category": category,
        "stage": stage,
        "node": node,
        "case": case_id,
        "role": role,
        "comparator": comparator,
        "references": [str(item) for item in references],
        "comparisons": [
            {
                "output": row.get("output"),
                "passed": bool(row.get("passed")),
                "message": _truncate(str(row.get("message") or "")),
            }
            for row in comparisons
        ],
        "exceptionType": exception_type,
        "detail": _truncate(detail),
    }


def comparison_failure(
    *,
    node: str,
    case_id: str,
    role: str,
    comparator: str,
    references: Sequence[str],
    comparisons: Sequence[Mapping[str, object]],
    detail: str = "",
) -> dict[str, object]:
    """Record an explicit comparator result; only a candidate deviation is candidate."""
    assert role in {"candidate", "baseline"}
    assert comparator
    normalized = [
        {
            "output": row.get("output"),
            "passed": bool(row.get("passed")),
            "message": _truncate(str(row.get("message") or "")),
        }
        for row in comparisons
    ]
    assert normalized
    assert any(row["passed"] is False for row in normalized)
    return _failure_base(
        kind=FAILURE_KIND_CANDIDATE if role == "candidate" else FAILURE_KIND_INFRASTRUCTURE,
        category="numeric" if role == "candidate" else "baseline_mismatch",
        stage="comparison",
        node=node,
        case_id=case_id,
        role=role,
        comparator=comparator,
        references=references,
        comparisons=normalized,
        exception_type=None,
        detail=detail or json.dumps(normalized, sort_keys=True),
    )


def infrastructure_failure(
    *,
    node: str | None,
    case_id: str | None,
    stage: str,
    category: str,
    detail: str,
    role: str | None = None,
    comparator: str | None = None,
    exception_type: str | None = None,
) -> dict[str, object]:
    """Record an execution/contract exception that must not blame the candidate."""
    assert category in {"exception", "assertion", "baseline_mismatch", "unclassified"}
    return _failure_base(
        kind=FAILURE_KIND_INFRASTRUCTURE,
        category=category,
        stage=stage,
        node=node,
        case_id=case_id,
        role=role,
        comparator=comparator,
        references=(),
        comparisons=(),
        exception_type=exception_type,
        detail=detail,
    )


def coverage_failure(
    *,
    node: str | None,
    case_id: str | None,
    category: str,
    detail: str,
) -> dict[str, object]:
    """Record a missing or unexpected contract case as coverage, never numeric."""
    assert category in {
        "case_identity",
        "unexpected_case",
        "missing_case",
        "missing_node",
        "skipped_node",
        "blocked_node",
        "coverage",
    }
    return _failure_base(
        kind=FAILURE_KIND_COVERAGE,
        category=category,
        stage="contract" if category == "case_identity" else "coverage",
        node=node,
        case_id=case_id,
        role=None,
        comparator=None,
        references=(),
        comparisons=(),
        exception_type=None,
        detail=detail,
    )


def _select_failure(rows: Sequence[Mapping[str, object]]) -> dict[str, object] | None:
    """Return the highest-precedence failure, preserving observation order."""
    selected: dict[str, object] | None = None
    selected_priority = 0
    for row in rows:
        priority = _FAILURE_PRIORITY.get(str(row.get("kind") or ""), 99)
        if selected is None or priority < selected_priority:
            selected = dict(row)
            selected_priority = priority
    return selected


def _as_count(value: object) -> int:
    """Coerce a JSON counter to a non-negative int, tolerating bad evidence."""
    if isinstance(value, (int, float, str, bytes)):
        try:
            return max(0, int(value))
        except (TypeError, ValueError):
            return 0
    return 0


def _counter_dict(value: object) -> dict[str, int]:
    counters: dict[str, int] = {}
    if isinstance(value, Mapping):
        for key in ("hits", "misses", "writes"):
            counters[key] = _as_count(value.get(key))
    return counters


def _node_short_name(node_id: str) -> str:
    name = node_id.rsplit("::", 1)[-1]
    return name.split("[", 1)[0]


def load_case_ids(task_dir: Path) -> tuple[str, ...]:
    """Load unique official case IDs from ``cases.yaml``.

    Missing, empty, duplicate or non-scalar IDs make the target contract
    unresolvable; a partially parsed list must never be treated as coverage.
    """
    payload = yaml.safe_load((Path(task_dir) / "cases.yaml").read_text(encoding="utf-8"))
    raw_cases = (payload or {}).get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise L3ContractError(f"{task_dir}/cases.yaml declares no cases")
    case_ids: list[str] = []
    for row in raw_cases:
        if not isinstance(row, dict) or "case_id" not in row:
            raise L3ContractError(f"{task_dir}/cases.yaml has a case without case_id")
        raw_id = row["case_id"]
        if not isinstance(raw_id, (int, str)) or str(raw_id).strip() == "":
            raise L3ContractError(f"{task_dir}/cases.yaml has an invalid case_id {raw_id!r}")
        case_ids.append(str(raw_id).strip())
    duplicates = sorted({case_id for case_id in case_ids if case_ids.count(case_id) > 1})
    if duplicates:
        raise L3ContractError(f"{task_dir}/cases.yaml has duplicate case_ids {duplicates}")
    return tuple(case_ids)


def load_supplemental_mla_ids(path: Path = SUPPLEMENTAL_MLA_PATH) -> tuple[str, ...]:
    """Load unique supplemental MLA case identities."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise L3ContractError(f"{path} declares no supplemental cases")
    case_ids: list[str] = []
    for row in raw_cases:
        if not isinstance(row, dict) or not isinstance(row.get("case_id"), str):
            raise L3ContractError(f"{path} has a supplemental case without a string case_id")
        case_id = row["case_id"].strip()
        if not case_id:
            raise L3ContractError(f"{path} has an empty supplemental case_id")
        case_ids.append(case_id)
    duplicates = sorted({case_id for case_id in case_ids if case_ids.count(case_id) > 1})
    if duplicates:
        raise L3ContractError(f"{path} has duplicate supplemental case_ids {duplicates}")
    return tuple(case_ids)


def _resolve_generic_contract(target: str, source_root: Path) -> TargetContract:
    spec = CannbenchOpsRegistry(source_root=source_root).resolve(target)
    case_ids = load_case_ids(spec.op_dir)
    return TargetContract(
        operator=spec.op_id,
        task=spec.task,
        task_dir=spec.op_dir,
        nodes=(NodeContract(name=GENERIC_NODE, required_cases=case_ids),),
        case_ids=case_ids,
    )


def resolve_contract(
    target: str,
    *,
    source_root: Path | None = None,
) -> TargetContract:
    """Resolve the full L3 contract for *target*.

    AttentionResidual and MLA use the in-repo CANNBench task definitions the
    acceptance tests actually execute.  Every other target is resolved through
    the campaign registry so its official ``cases.yaml`` defines coverage.
    """
    raw = str(target or "").strip()
    if not raw:
        raise L3ContractError("TF_L3_TARGET_OPERATOR is required for L3 promotion")
    short = raw.rsplit("/", 1)[-1].strip().lower()
    if short == "attention_residual":
        case_ids = load_case_ids(AR_DIR)
        return TargetContract(
            operator="attention_residual",
            task=AR_TASK,
            task_dir=AR_DIR,
            nodes=(
                NodeContract(name=AR_NODE, required_cases=case_ids),
                NodeContract(name=AR_PARITY_NODE, required_cases=AR_PARITY_CASES),
                NodeContract(name=DYNAMIC_NODE, required_cases=AR_DYNAMIC_CASES),
            ),
            case_ids=case_ids,
        )
    if short == "mla":
        case_ids = load_case_ids(MLA_DIR)
        supplemental = load_supplemental_mla_ids()
        return TargetContract(
            operator="mla",
            task=MLA_TASK,
            task_dir=MLA_DIR,
            nodes=(
                NodeContract(name=MLA_NODE, required_cases=case_ids),
                NodeContract(name=MLA_SUPPLEMENTAL_NODE, required_cases=supplemental),
                NodeContract(name=DYNAMIC_NODE, required_cases=MLA_DYNAMIC_CASES),
            ),
            case_ids=case_ids + supplemental,
        )
    root = source_root if source_root is not None else BENCHMARKS_DIR.parents[2]
    return _resolve_generic_contract(raw, root)


def cache_expectation() -> str:
    """Return ``cold``, ``warm`` or ``any`` for oracle reference-cache lookups."""
    raw = os.environ.get(CACHE_EXPECTATION_ENV, "any").strip().lower() or "any"
    if raw not in {"cold", "warm", "any"}:
        raise L3ContractError(
            f"{CACHE_EXPECTATION_ENV} must be cold, warm or any, got {raw!r}"
        )
    return raw


@dataclass(frozen=True, kw_only=True)
class CacheProbeResult:
    """Result of the write/read/delete probe on the promotion cache directory."""

    ok: bool
    cache_dir: str
    error: str = ""


def probe_reference_cache(cache_dir: str | Path) -> CacheProbeResult:
    """Write, read back and delete a probe file in *cache_dir*.

    Used before a promotion run and by the adapter preflight.  It verifies the
    exact directory the reference cache will use, not a parent or a shared
    cache.
    """
    root = Path(cache_dir)
    temporary: Path | None = None
    try:
        root.mkdir(parents=True, exist_ok=True)
        handle, name = tempfile.mkstemp(prefix=".l3-cache-probe-", dir=root)
        temporary = Path(name)
        with os.fdopen(handle, "wb") as stream:
            stream.write(b"tilefoundry-l3-reference-cache-probe")
            stream.flush()
            os.fsync(stream.fileno())
        if temporary.read_bytes() != b"tilefoundry-l3-reference-cache-probe":
            raise OSError("probe read-back mismatch")
        return CacheProbeResult(ok=True, cache_dir=str(root))
    except OSError as exc:
        return CacheProbeResult(
            ok=False,
            cache_dir=str(root),
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


class EvidenceRecorder:
    """Atomic per-case evidence writer for one L3 promotion session."""

    active = True

    def __init__(
        self,
        path: Path,
        contract: TargetContract,
        *,
        profile: str = PROFILE,
        expected_python: str = "",
    ) -> None:
        self.path = Path(path)
        self.contract = contract
        self.profile = profile
        self.expected_python = expected_python
        self.nodes: dict[str, NodeRecord] = {
            node.name: NodeRecord(name=node.name, required_cases=node.required_cases)
            for node in contract.nodes
        }
        self.started_at = _utc_now()
        self.updated_at = self.started_at
        self.finished_at = ""
        self.phase = "initialized"
        self.interrupted = True
        self.signal = ""
        self.error_category = ""
        self.error_detail = ""
        self.cache_observation: dict[str, object] = {}
        self.cache_preflight: dict[str, object] | None = None
        self.candidate: dict[str, object] = {}
        declared_target = os.environ.get(MODULE_ENV, "").strip()
        self.target_module_path = (
            str(Path(declared_target).resolve()) if declared_target else ""
        )
        self.interpreter = self.interpreter_identity()
        self.last_completed_node = ""
        self.last_completed_case: dict[str, object] | None = None
        self.write_errors: list[str] = []
        self.failures: list[dict[str, object]] = []
        self._write(strict=True)

    def interpreter_identity(self) -> dict[str, object]:
        expected = self.expected_python.strip()
        actual = Path(sys.executable).resolve()
        matches = not expected or Path(expected).resolve() == actual
        return {
            "sysExecutable": str(actual),
            "expected": str(Path(expected).resolve()) if expected else None,
            "match": matches,
            "pid": os.getpid(),
            "argv": list(sys.argv),
            "cwd": os.getcwd(),
            "version": sys.version,
            "prefix": sys.prefix,
            "basePrefix": sys.base_prefix,
            "condaEnv": os.environ.get("CONDA_DEFAULT_ENV") or None,
            "environment": {
                key: os.environ.get(key)
                for key in (
                    "CANNBENCH_CACHE_DIR",
                    "CANNBENCH_CACHE_MODE",
                    "TILELANG_CACHE_DIR",
                    "TILELANG_DISABLE_CACHE",
                    "TORCH_DEVICE_BACKEND_AUTOLOAD",
                    "PYTHONNOUSERSITE",
                    "TILEFOUNDRY_NPU_TILELANG_SOURCE",
                    "TILEFOUNDRY_NPU_TILELANG_BUILD",
                    "TILEFOUNDRY_NPU_CANNBENCH_REPO",
                    "TILEFOUNDRY_NPU_CANNBENCH_ARCH",
                    "TILEFOUNDRY_NPU_WORK_ROOT",
                )
            },
        }

    def note_module(self, role: str, module: ModuleType) -> None:
        """Record the provenance of a module already imported by this process."""
        modules = self.interpreter.setdefault("modules", {})
        assert isinstance(modules, dict)
        module_file = getattr(module, "__file__", None)
        version = getattr(module, "__version__", None)
        modules[role] = {
            "file": str(Path(module_file).resolve()) if module_file else None,
            "name": module.__name__,
            "version": str(version) if version is not None else None,
        }
        self._write()

    def note_candidate(self, module_path: str | Path) -> None:
        """Record the imported candidate module and verify the declared path."""
        resolved = Path(module_path).resolve()
        declared = os.environ.get(MODULE_ENV, "").strip()
        declared_resolved = str(Path(declared).resolve()) if declared else None
        digest = _sha256_file(resolved) if resolved.is_file() else None
        self.candidate = {
            "modulePath": str(resolved),
            "sha256": digest,
            "declaredPath": declared_resolved,
        }
        if not self.target_module_path:
            self.target_module_path = str(resolved)
        if declared_resolved is not None and declared_resolved != str(resolved):
            self.environment_block(
                "candidate_identity_mismatch",
                f"declared candidate {declared_resolved} loaded as {resolved}",
            )
        self._write()

    @contextmanager
    def node(self, name: str) -> Iterator[None]:
        """Mark a contract node as running and preserve failures for evidence."""
        record = self.nodes.get(name)
        if record is None:
            raise L3ContractError(f"{name} is not part of the {self.contract.operator} contract")
        self.node_started(name)
        try:
            yield
        except CandidateComparisonError as exc:
            self._append_failure(exc.failure)
            status = "blocked" if self.error_category else "failed"
            self.node_finished(name, status, f"{type(exc).__name__}: {exc}")
            raise
        except CoverageContractError as exc:
            self._append_failure(exc.failure)
            status = "blocked" if self.error_category else "failed"
            self.node_finished(name, status, f"{type(exc).__name__}: {exc}")
            raise
        except ClassifiedTestFailure as exc:
            status = "blocked" if self.error_category else "failed"
            self.node_finished(name, status, f"{type(exc).__name__}: {exc}")
            raise
        except Exception as exc:
            if not getattr(exc, "l3_failure_classified", False):
                self._append_failure(
                    infrastructure_failure(
                        node=name,
                        case_id=None,
                        stage="report",
                        category="assertion" if isinstance(exc, AssertionError) else "exception",
                        exception_type=type(exc).__name__,
                        detail=f"{type(exc).__name__}: {exc}",
                    )
                )
            status = "blocked" if self.error_category else "failed"
            self.node_finished(name, status, f"{type(exc).__name__}: {exc}")
            raise
        except BaseException as exc:
            status = "blocked" if self.error_category else "failed"
            self.node_finished(
                name,
                status,
                f"{type(exc).__name__}: {exc}",
                classify_missing_failure=False,
            )
            raise
        else:
            self.node_finished(name, "passed")

    def node_started(self, name: str) -> None:
        record = self.nodes[name]
        record.status = "running"
        record.started_at = _utc_now()
        self.phase = f"node:{name}"
        self._write()

    def node_finished(
        self,
        name: str,
        status: str,
        detail: str = "",
        *,
        classify_missing_failure: bool = True,
    ) -> None:
        """Finish *name*, classifying a failed node unless explicitly suppressed.

        Skip, exit and interruption events pass ``classify_missing_failure=False``
        so they remain coverage or environment outcomes rather than invented
        infrastructure failures.
        """
        record = self.nodes[name]
        record.status = status
        record.ended_at = _utc_now()
        if detail:
            record.error = _truncate(detail)
        if status == "passed":
            self.last_completed_node = name
        elif (
            status == "failed"
            and classify_missing_failure
            and not self._has_failure(node=name)
        ):
            self._append_failure(
                infrastructure_failure(
                    node=name,
                    case_id=None,
                    stage="report",
                    category="unclassified",
                    detail=detail or f"{name} failed without a structured failure",
                )
            )
        self.phase = "between_nodes"
        self._write()

    def note_report(self, node_id: str, when: str, outcome: str, longrepr: str = "") -> None:
        """Apply a pytest run report to the matching contract node."""
        name = _node_short_name(node_id)
        record = self.nodes.get(name)
        if record is None:
            return
        if not record.node_id:
            record.node_id = node_id
        if when == "setup":
            if outcome == "skipped":
                self.node_finished(name, "skipped", longrepr)
            elif outcome == "failed":
                self.node_finished(name, "failed", longrepr)
        elif when == "call":
            status = outcome
            if outcome != "passed" and self.error_category:
                status = "blocked"
            self.node_finished(name, status, longrepr if outcome != "passed" else "")
        elif when == "teardown" and outcome == "failed" and record.status != "failed":
            self.node_finished(name, "failed", longrepr)
        else:
            return
        self._write()

    def case_started(self, node: str, case_id: str) -> None:
        self.phase = f"case:{node}:{case_id}"
        self._write()

    def case_finished(
        self,
        node: str,
        case_id: str,
        status: str,
        detail: str = "",
        *,
        failure: Mapping[str, object] | None = None,
    ) -> None:
        record = self.nodes.get(node)
        if record is None:
            raise L3ContractError(f"{node} is not part of the {self.contract.operator} contract")
        record.cases[case_id] = status
        if detail:
            record.details[case_id] = _truncate(detail)
        if status == "passed":
            assert failure is None
            self._drop_case_failures(node, case_id)
        else:
            assert status in {"failed", "error"}
            if failure is not None:
                row = dict(failure)
                if row.get("node") is None:
                    row["node"] = node
                if row.get("case") is None:
                    row["case"] = case_id
                self._append_failure(row)
            elif not self._has_failure(node=node, case_id=case_id):
                self._append_failure(
                    infrastructure_failure(
                        node=node,
                        case_id=case_id,
                        stage="report",
                        category="unclassified",
                        detail=detail or f"{node}:{case_id} finished {status} without a structured failure",
                    )
                )
        self.last_completed_case = {"node": node, "caseId": case_id, "status": status}
        self._write()

    def record_failure(self, failure: Mapping[str, object]) -> None:
        """Persist one structured failure without changing case status."""
        self._append_failure(failure)
        self._write()

    def _append_failure(self, failure: Mapping[str, object]) -> dict[str, object]:
        row = dict(failure)
        assert row.get("kind") in _FAILURE_PRIORITY
        assert isinstance(row.get("node"), (str, type(None)))
        assert isinstance(row.get("case"), (str, type(None)))
        assert isinstance(row.get("references"), list)
        assert isinstance(row.get("comparisons"), list)
        if row not in self.failures:
            self.failures.append(row)
        return row

    def _has_failure(self, *, node: str, case_id: str | None = None) -> bool:
        return any(
            row.get("node") == node
            and (case_id is None or row.get("case") == case_id)
            for row in self.failures
        )

    def _drop_case_failures(self, node: str, case_id: str) -> None:
        self.failures[:] = [
            row
            for row in self.failures
            if not (row.get("node") == node and row.get("case") == case_id)
        ]

    def _case_failure(self, node: str, case_id: str) -> dict[str, object] | None:
        return _select_failure(
            [
                row
                for row in self.failures
                if row.get("node") == node and row.get("case") == case_id
            ]
        )

    def _node_failure(self, node: str) -> dict[str, object] | None:
        return _select_failure(
            [row for row in self.failures if row.get("node") == node]
        )

    def reference_cache(self, observation: dict[str, object]) -> None:
        """Attach the reference-cache observation collected by the harness."""
        self.cache_observation = self._merge_cache_observation(
            self.cache_observation, observation
        )
        if self.cache_preflight is not None:
            self.cache_observation["preflight"] = self.cache_preflight
        self._write()

    @staticmethod
    def _merge_cache_observation(
        previous: dict[str, object], observation: dict[str, object]
    ) -> dict[str, object]:
        """Sum cache counters across contract nodes.

        Each node builds its own ``ReferenceCache``, and a cold session still
        hits entries computed by an earlier node.  Receipts therefore record
        the whole session, while the policy check below expects at least one
        computed-and-written value for cold and only hits for warm.
        """
        current = dict(observation)
        raw_errors = current.get("errors")
        errors = [str(item) for item in raw_errors] if isinstance(raw_errors, (list, tuple)) else []
        current["errors"] = errors
        raw_by_kind = current.get("byKind")
        current_kinds: dict[str, dict[str, int]] = {
            str(kind): _counter_dict(counters)
            for kind, counters in raw_by_kind.items()
        } if isinstance(raw_by_kind, Mapping) else {}
        current["byKind"] = current_kinds
        if not previous:
            current["observations"] = 1
            return current
        merged = dict(previous)
        for key in ("hits", "misses", "writes"):
            merged[key] = _as_count(previous.get(key)) + _as_count(current.get(key))
        merged["usable"] = bool(previous.get("usable")) and bool(current.get("usable"))
        merged["required"] = bool(previous.get("required")) or bool(current.get("required"))
        raw_previous_errors = previous.get("errors")
        previous_errors = (
            [str(item) for item in raw_previous_errors]
            if isinstance(raw_previous_errors, (list, tuple))
            else []
        )
        merged["errors"] = previous_errors + errors
        raw_previous_kinds = previous.get("byKind")
        by_kind = {
            str(kind): _counter_dict(counters)
            for kind, counters in raw_previous_kinds.items()
        } if isinstance(raw_previous_kinds, Mapping) else {}
        for kind, counters in current_kinds.items():
            base = dict(by_kind.get(kind) or {})
            for key in ("hits", "misses", "writes"):
                base[key] = _as_count(base.get(key)) + _as_count(counters.get(key))
            by_kind[kind] = base
        merged["byKind"] = by_kind
        if (
            previous.get("mode") != current.get("mode")
            or previous.get("location") != current.get("location")
            or previous.get("expectation") != current.get("expectation")
        ):
            merged["errors"] = list(merged["errors"]) + [
                "reference cache policy changed between contract nodes"
            ]
            merged["usable"] = False
            merged["policyChanged"] = True
        merged["observations"] = max(1, _as_count(previous.get("observations"))) + 1
        return merged

    def note_cache_preflight(self, preflight: dict[str, object]) -> None:
        """Record the write/read/delete probe of the promotion cache directory."""
        self.cache_preflight = dict(preflight)
        self._write()

    def environment_block(self, category: str, detail: str) -> None:
        """Record an environment-level block that must not count as a candidate failure."""
        self.error_category = category
        self.error_detail = _truncate(detail)
        self.phase = f"blocked:{category}"
        self._write()

    def note_signal(self, signum: int) -> None:
        self.interrupted = True
        self.signal = signal.Signals(signum).name
        self.phase = f"signal:{self.signal}"
        self._write()

    def reference_cache_problem(self) -> str:
        """Return the cache policy violation for this run, if any."""
        observation = self.cache_observation
        if not observation:
            return ""
        if bool(observation.get("required")) and not bool(observation.get("usable")):
            return "reference_cache_unavailable"
        if bool(observation.get("required")) and observation.get("errors"):
            return "reference_cache_error"
        if observation.get("policyChanged"):
            return "reference_cache_policy_changed"
        expectation = str(observation.get("expectation") or "any")
        by_kind = observation.get("byKind") or {}
        oracle = by_kind.get("oracle") if isinstance(by_kind, dict) else None
        counters = oracle if isinstance(oracle, dict) else {}
        hits = int(counters.get("hits", 0))
        misses = int(counters.get("misses", 0))
        writes = int(counters.get("writes", 0))
        if expectation == "cold" and (misses <= 0 or writes != misses):
            return "reference_cache_expectation_cold"
        if expectation == "warm" and (misses != 0 or writes != 0 or hits <= 0):
            return "reference_cache_expectation_warm"
        return ""

    def coverage(self) -> dict[str, object]:
        """Compute contract coverage; partial, skipped or duplicate runs never qualify."""
        missing_nodes: list[str] = []
        failed_nodes: list[str] = []
        blocked_nodes: list[str] = []
        skipped_nodes: list[str] = []
        missing_cases: list[str] = []
        unexpected_cases: list[str] = []
        expected_checks = 0
        completed_checks = 0
        for node_contract in self.contract.nodes:
            record = self.nodes[node_contract.name]
            if record.status == "pending" or record.status == "running":
                missing_nodes.append(record.name)
            elif record.status == "failed":
                failed_nodes.append(record.name)
            elif record.status == "blocked":
                blocked_nodes.append(record.name)
            elif record.status == "skipped":
                skipped_nodes.append(record.name)
            for case_id in node_contract.required_cases:
                expected_checks += 1
                status = record.cases.get(case_id)
                if status == "passed":
                    completed_checks += 1
                else:
                    missing_cases.append(f"{record.name}:{case_id}")
            known = set(node_contract.required_cases)
            unexpected_cases.extend(
                f"{record.name}:{case_id}"
                for case_id in sorted(record.cases)
                if case_id not in known
            )
        cache_problem = self.reference_cache_problem()
        completed_nodes = sum(
            1
            for node_contract in self.contract.nodes
            if self.nodes[node_contract.name].status == "passed"
        )
        qualified = (
            not missing_nodes
            and not failed_nodes
            and not blocked_nodes
            and not skipped_nodes
            and not missing_cases
            and not unexpected_cases
            and not self.error_category
            and not self.write_errors
            and not cache_problem
        )
        return {
            "nodes": {
                "expected": len(self.contract.nodes),
                "completed": completed_nodes,
                "missing": missing_nodes,
                "failed": failed_nodes,
                "blocked": blocked_nodes,
                "skipped": skipped_nodes,
            },
            "cases": {
                "expected": expected_checks,
                "completed": completed_checks,
                "missing": missing_cases,
                "unexpected": unexpected_cases,
            },
            "referenceCacheProblem": cache_problem or None,
            "qualified": qualified,
        }

    def _contract_passed(self) -> bool:
        """Return True when every contract node and case completed successfully."""
        for node_contract in self.contract.nodes:
            record = self.nodes[node_contract.name]
            if record.status != "passed":
                return False
            if set(record.cases) != set(node_contract.required_cases):
                return False
            if any(status != "passed" for status in record.cases.values()):
                return False
        return True

    def finalize(self, *, interrupted: bool = False) -> None:
        """Close the evidence record; a missing finalize means the run was killed.

        A fully green run that could not persist evidence or violated the cache
        expectation records an infrastructure block, not a candidate rollback.
        """
        if interrupted or self.signal:
            self.interrupted = True
            if not self.error_category:
                self.error_category = "interrupted"
        else:
            self.interrupted = False
            cache_problem = self.reference_cache_problem()
            if (
                self.write_errors
                and not self.error_category
                and self._contract_passed()
            ):
                self.environment_block(
                    "evidence_write_error",
                    "; ".join(self.write_errors[-3:]),
                )
            elif cache_problem and not self.error_category and self._contract_passed():
                self.environment_block(
                    cache_problem,
                    "reference cache observation does not satisfy the promotion expectation",
                )
        self.phase = "finished"
        self.finished_at = _utc_now()
        self._write(strict=True)

    def _coverage_gap_failure(
        self, coverage: dict[str, object] | None = None
    ) -> dict[str, object] | None:
        """Describe a coverage gap that no case-level failure already explains."""
        observed = coverage if coverage is not None else self.coverage()
        if self.write_errors:
            return infrastructure_failure(
                node=None,
                case_id=None,
                stage="report",
                category="unclassified",
                detail="; ".join(self.write_errors[-3:]),
            )
        if self.error_category:
            return infrastructure_failure(
                node=None,
                case_id=None,
                stage="report",
                category="unclassified",
                detail=f"{self.error_category}: {self.error_detail}",
            )
        cache_problem = str(observed.get("referenceCacheProblem") or "")
        if cache_problem:
            return infrastructure_failure(
                node=None,
                case_id=None,
                stage="coverage",
                category="unclassified",
                detail=cache_problem,
            )
        nodes = observed.get("nodes")
        if isinstance(nodes, Mapping):
            for name in nodes.get("missing") or []:
                node_name = str(name)
                if not self._has_failure(node=node_name):
                    return coverage_failure(
                        node=node_name,
                        case_id=None,
                        category="missing_node",
                        detail=f"{node_name} did not reach a terminal state",
                    )
            for name in nodes.get("blocked") or []:
                node_name = str(name)
                if not self._has_failure(node=node_name):
                    return coverage_failure(
                        node=node_name,
                        case_id=None,
                        category="blocked_node",
                        detail=f"{node_name} was blocked",
                    )
            for name in nodes.get("skipped") or []:
                node_name = str(name)
                if not self._has_failure(node=node_name):
                    return coverage_failure(
                        node=node_name,
                        case_id=None,
                        category="skipped_node",
                        detail=f"{node_name} was skipped",
                    )
            for name in nodes.get("failed") or []:
                node_name = str(name)
                if not self._has_failure(node=node_name):
                    return infrastructure_failure(
                        node=node_name,
                        case_id=None,
                        stage="report",
                        category="unclassified",
                        detail=f"{node_name} failed without a structured failure",
                    )
        cases = observed.get("cases")
        if isinstance(cases, Mapping):
            for key in cases.get("missing") or []:
                node_name, _, case_id = str(key).partition(":")
                if not self._has_failure(node=node_name, case_id=case_id):
                    return coverage_failure(
                        node=node_name,
                        case_id=case_id,
                        category="missing_case",
                        detail=f"{key} was not completed",
                    )
            for key in cases.get("unexpected") or []:
                node_name, _, case_id = str(key).partition(":")
                if not self._has_failure(node=node_name, case_id=case_id):
                    return coverage_failure(
                        node=node_name,
                        case_id=case_id,
                        category="unexpected_case",
                        detail=f"{key} is not in the contract",
                    )
        if observed.get("qualified") is False and not self.failures:
            return coverage_failure(
                node=None,
                case_id=None,
                category="coverage",
                detail="contract coverage is incomplete",
            )
        return None

    def _cache_payload(self) -> dict[str, object] | None:
        if self.cache_observation:
            payload = dict(self.cache_observation)
            if self.cache_preflight is not None:
                payload["preflight"] = self.cache_preflight
            return payload
        if self.cache_preflight is not None:
            return {"preflight": self.cache_preflight}
        return None

    def payload(self) -> dict[str, object]:
        nodes: list[dict[str, object]] = []
        cases: dict[str, object] = {}
        for node_contract in self.contract.nodes:
            record = self.nodes[node_contract.name]
            nodes.append(
                {
                    "name": record.name,
                    "nodeId": record.node_id or None,
                    "status": record.status,
                    "requiredCases": list(node_contract.required_cases),
                    "cases": dict(sorted(record.cases.items())),
                    "details": dict(sorted(record.details.items())),
                    "failure": self._node_failure(record.name),
                    "error": record.error or None,
                    "startedAt": record.started_at or None,
                    "endedAt": record.ended_at or None,
                }
            )
            for case_id, status in sorted(record.cases.items()):
                cases[f"{record.name}:{case_id}"] = {
                    "status": status,
                    "detail": record.details.get(case_id),
                    "failure": self._case_failure(record.name, case_id),
                }
        coverage = self.coverage()
        failures = [dict(row) for row in self.failures]
        gap = self._coverage_gap_failure(coverage)
        if gap is not None and gap not in failures:
            failures.append(gap)
        return {
            "schema": SCHEMA,
            "profile": self.profile,
            "target": {
                "operator": self.contract.operator,
                "task": self.contract.task,
                "taskDir": str(self.contract.task_dir),
                "modulePath": self.target_module_path or None,
                "cases": list(self.contract.case_ids),
            },
            "candidate": self.candidate or None,
            "interpreter": self.interpreter,
            "timing": {
                "startedAt": self.started_at,
                "updatedAt": self.updated_at,
                "finishedAt": self.finished_at or None,
            },
            "phase": self.phase,
            "nodes": nodes,
            "cases": cases,
            "failure": _select_failure(failures),
            "failures": failures,
            "referenceCache": self._cache_payload(),
            "coverage": coverage,
            "writeErrors": list(self.write_errors),
            "interruption": {
                "interrupted": self.interrupted,
                "lastCompletedNode": self.last_completed_node or None,
                "lastCompletedCase": self.last_completed_case,
                "errorCategory": self.error_category or None,
                "detail": self.error_detail or None,
                "signal": self.signal or None,
            },
        }

    def _write(self, *, strict: bool = False) -> None:
        self.updated_at = _utc_now()
        temporary: Path | None = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            handle, name = tempfile.mkstemp(
                prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
            )
            temporary = Path(name)
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(self.payload(), stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            temporary = None
        except OSError as exc:
            self.write_errors.append(f"{type(exc).__name__}: {exc}")
            if strict:
                raise
            print(
                f"[l3-evidence] write failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
                flush=True,
            )
        finally:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass


class NullRecorder:
    """No-op observer used for ordinary (non-promotion) acceptance runs."""

    active = False

    def note_module(self, role: str, module: ModuleType) -> None:
        _ = (role, module)

    def note_candidate(self, module_path: str | Path) -> None:
        _ = module_path

    def node(self, name: str) -> AbstractContextManager[None]:
        _ = name
        return nullcontext()

    def node_started(self, name: str) -> None:
        _ = name

    def node_finished(
        self,
        name: str,
        status: str,
        detail: str = "",
        *,
        classify_missing_failure: bool = True,
    ) -> None:
        _ = (name, status, detail, classify_missing_failure)

    def note_report(self, node_id: str, when: str, outcome: str, longrepr: str = "") -> None:
        _ = (node_id, when, outcome, longrepr)

    def case_started(self, node: str, case_id: str) -> None:
        _ = (node, case_id)

    def case_finished(
        self,
        node: str,
        case_id: str,
        status: str,
        detail: str = "",
        *,
        failure: Mapping[str, object] | None = None,
    ) -> None:
        _ = (node, case_id, status, detail, failure)

    def record_failure(self, failure: Mapping[str, object]) -> None:
        _ = failure

    def reference_cache(self, observation: dict[str, object]) -> None:
        _ = observation

    def note_cache_preflight(self, preflight: dict[str, object]) -> None:
        _ = preflight

    def environment_block(self, category: str, detail: str) -> None:
        _ = (category, detail)

    def note_signal(self, signum: int) -> None:
        _ = signum

    def coverage(self) -> dict[str, object]:
        return {"qualified": False}

    def finalize(self, *, interrupted: bool = False) -> None:
        _ = interrupted


Recorder = EvidenceRecorder | NullRecorder

_RECORDER: Recorder | None = None


def reset_recorder() -> None:
    """Drop the session recorder (host tests and ``--forked`` isolation)."""
    global _RECORDER
    _RECORDER = None


def get_recorder() -> Recorder:
    """Return the session recorder, creating it from the environment once."""
    global _RECORDER
    if _RECORDER is not None:
        return _RECORDER
    raw_path = os.environ.get(EVIDENCE_ENV, "").strip()
    if not raw_path:
        _RECORDER = NullRecorder()
        return _RECORDER
    path = Path(raw_path)
    profile = os.environ.get(PROFILE_ENV, PROFILE).strip() or PROFILE
    expected_python = os.environ.get(EXPECTED_PYTHON_ENV, "").strip()
    target = os.environ.get(TARGET_ENV, "").strip()
    source_root = BENCHMARKS_DIR.parents[2]
    try:
        contract = resolve_contract(target, source_root=source_root)
    except Exception as exc:  # noqa: BLE001 - evidence must exist even for broken contracts
        contract = TargetContract(
            operator=target.rsplit("/", 1)[-1].strip(),
            task=target,
            task_dir=BENCHMARKS_DIR,
            nodes=(),
            case_ids=(),
        )
        recorder = EvidenceRecorder(
            path, contract, profile=profile, expected_python=expected_python
        )
        recorder.environment_block(
            "target_contract_error", f"{type(exc).__name__}: {exc}"
        )
        _RECORDER = recorder
        return _RECORDER
    recorder = EvidenceRecorder(
        path, contract, profile=profile, expected_python=expected_python
    )
    if not expected_python:
        recorder.environment_block(
            "interpreter_identity_missing",
            f"{EXPECTED_PYTHON_ENV} is required for L3 promotion evidence",
        )
    elif not recorder.interpreter["match"]:
        recorder.environment_block(
            "interpreter_mismatch",
            f"pytest interpreter {recorder.interpreter['sysExecutable']} does not match "
            f"adapter-resolved {recorder.interpreter['expected']}",
        )
    _RECORDER = recorder
    return _RECORDER


def active_recorder() -> EvidenceRecorder | None:
    """Return the evidence recorder when promotion evidence is active."""
    recorder = get_recorder()
    return recorder if isinstance(recorder, EvidenceRecorder) else None


def current_recorder() -> EvidenceRecorder | None:
    """Return an already-activated recorder without creating one.

    Hooks that run after tests must not resurrect a recorder from a
    monkeypatched environment (for example a host test that just cleared the
    evidence variables), otherwise pytest would exit non-zero after a fully
    passing run.
    """
    recorder = _RECORDER
    return recorder if isinstance(recorder, EvidenceRecorder) else None
