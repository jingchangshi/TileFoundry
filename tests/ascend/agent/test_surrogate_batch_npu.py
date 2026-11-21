"""Lease-locked AgentGates L2 surrogate CANNBench batch evaluator on NPU."""

from __future__ import annotations

import gc
import hashlib
import importlib
import importlib.util
import json
import math
import os
import re
import statistics
import sys
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from tilefoundry_ascend.agent.campaign.anchor_specs import (
    AnchorCaseSpec,
    default_source_root,
    get_default_anchors,
    get_operator_entrypoint,
    get_operator_module_relpath,
    load_anchor_harness_and_cases,
    load_cannbench_contract_module,
    resolve_cannbench_repo,
)
from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (
    FAST_REQUEST_ENV,
    FAST_REQUEST_SCHEMA,
    FAST_RESPONSE_SCHEMA,
    FAST_SOURCE_ROOT_ENV,
    FAST_STATUS_CANDIDATE_FAILURE,
    FAST_STATUS_INFRASTRUCTURE_ERROR,
    FAST_STATUS_OK,
    FAST_STATUS_REQUEST_ERROR,
    AnchorEvalMetrics,
    CandidateSurrogateResult,
    aggregate_surrogate_result,
    build_anchor_metrics,
    ensure_npu_compare_shim_dir,
)
from tilefoundry_ascend.agent.flywheel.ablation_distiller import (
    MicroProbeSpec,
    verify_micro_probe_hardware,
)

_ENV_SHIM_DIR = os.environ.get("TF_SURROGATE_SHIM_DIR", "").strip()
if (
    _ENV_SHIM_DIR
    and (_ENV_SHIM_DIR not in sys.path)
    and (Path(_ENV_SHIM_DIR) / "npu_compare.py").is_file()
):
    sys.path.insert(0, _ENV_SHIM_DIR)


def _require_npu_env() -> None:
    if not hasattr(torch, "npu"):
        try:
            importlib.import_module("torch_npu")
        except ImportError:
            pass
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        if os.environ.get(FAST_REQUEST_ENV, "").strip() or os.environ.get(
            "TF_SURROGATE_BATCH_REQUEST", ""
        ).strip():
            raise RuntimeError("Ascend NPU device is unavailable for surrogate batch request.")
        pytest.skip("Ascend NPU device is unavailable on this host.")


def _resolve_batch_cannbench_repo(source_root: Path) -> Path:
    explicit = os.environ.get("TF_SURROGATE_CANNBENCH_REPO", "").strip()
    if explicit and Path(explicit).is_dir():
        return Path(explicit).resolve()
    try:
        return resolve_cannbench_repo()
    except FileNotFoundError:
        sibling = source_root.parent / "cann-bench"
        if (sibling / "tasks").is_dir():
            return sibling.resolve()
        raise


def _ensure_npu_compare(cannbench_repo: Path, shim_dir: Path) -> None:
    if importlib.util.find_spec("npu_compare") is not None:
        return
    created = ensure_npu_compare_shim_dir(shim_dir, cannbench_repo=cannbench_repo)
    if str(created) not in sys.path:
        sys.path.insert(0, str(created))


def _load_candidate_entrypoint(module_path: Path, entrypoint: str, unique_tag: str):
    mod_name = f"_tf_surrogate_cand_{unique_tag}"
    spec = importlib.util.spec_from_file_location(mod_name, str(module_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load candidate module {module_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, entrypoint):
        raise AttributeError(f"{module_path} does not define {entrypoint!r}")
    return getattr(mod, entrypoint), mod


def _measure_npu_latency_us(fn_call, *, warmup: int, repeat: int) -> float:
    try:
        profiler = importlib.import_module("tilelang.profiler")
        lat_ms = profiler.do_bench(
            fn_call,
            _n_warmup=warmup,
            _n_repeat=repeat,
            backend="event",
            return_mode="median",
            device=torch.device("npu:0"),
        )
        return float(lat_ms) * 1000.0
    except Exception:
        for _ in range(max(1, warmup)):
            fn_call()
        torch.npu.synchronize()
        samples_ms: list[float] = []
        for _ in range(max(1, repeat)):
            start_ev = torch.npu.Event(enable_timing=True)
            end_ev = torch.npu.Event(enable_timing=True)
            start_ev.record()
            fn_call()
            end_ev.record()
            torch.npu.synchronize()
            samples_ms.append(float(start_ev.elapsed_time(end_ev)))
        return float(statistics.median(samples_ms)) * 1000.0


def _normalize_pmu_ratio(raw_val: object) -> float:
    val = float(raw_val)
    if val > 1.05:
        val /= 100.0
    return max(0.0, min(1.0, val))


def _extract_msprof_metrics(prof: object) -> dict[str, float | int]:
    aic_mad = _normalize_pmu_ratio(getattr(prof, "aic_mad", 0.0))
    aic_scalar = _normalize_pmu_ratio(getattr(prof, "aic_scalar", 0.0))
    aic_mte1 = _normalize_pmu_ratio(getattr(prof, "aic_mte1", 0.0))
    aic_mte2 = _normalize_pmu_ratio(getattr(prof, "aic_mte2", 0.0))
    aic_mte3 = _normalize_pmu_ratio(getattr(prof, "aic_mte3", 0.0))
    aic_fixpipe = _normalize_pmu_ratio(getattr(prof, "aic_fixpipe", 0.0))
    aic_total_cycles = int(getattr(prof, "aic_total_cycles", 0))

    aiv_vec = _normalize_pmu_ratio(getattr(prof, "aiv_vec", 0.0))
    aiv_scalar = _normalize_pmu_ratio(getattr(prof, "aiv_scalar", 0.0))
    aiv_mte2 = _normalize_pmu_ratio(getattr(prof, "aiv_mte2", 0.0))
    aiv_mte3 = _normalize_pmu_ratio(getattr(prof, "aiv_mte3", 0.0))
    aiv_total_cycles = int(getattr(prof, "aiv_total_cycles", 0))

    raw_dur_us = getattr(prof, "dur_us", None)
    if raw_dur_us is None:
        dur_us = float(getattr(prof, "dur_ns", 0.0)) / 1000.0
    else:
        dur_us = float(raw_dur_us)

    explicit_sync = getattr(prof, "sync_stall_ratio", None)
    if explicit_sync is not None:
        sync_stall_ratio = _normalize_pmu_ratio(explicit_sync)
    else:
        has_aic = (aic_total_cycles > 0) or any(
            v > 0.0 for v in (aic_mad, aic_scalar, aic_mte1, aic_mte2, aic_mte3, aic_fixpipe)
        )
        has_aiv = (aiv_total_cycles > 0) or any(
            v > 0.0 for v in (aiv_vec, aiv_scalar, aiv_mte2, aiv_mte3)
        )
        aic_active = (
            max(aic_mad, aic_mte1, aic_mte2, aic_mte3, aic_fixpipe, aic_scalar) if has_aic else 0.0
        )
        aiv_active = max(aiv_vec, aiv_mte2, aiv_mte3, aiv_scalar) if has_aiv else 0.0
        dominant_active = max(aic_active, aiv_active)
        if has_aic or has_aiv:
            sync_stall_ratio = max(0.0, 1.0 - min(1.0, dominant_active + 0.15))
        else:
            sync_stall_ratio = 0.0

    return {
        "aic_mac_ratio": aic_mad,
        "aiv_vec_ratio": aiv_vec,
        "mte2_wait_ratio": max(aiv_mte2, aic_mte2),
        "sync_stall_ratio": sync_stall_ratio,
        "aic_mad": aic_mad,
        "aic_fixpipe": aic_fixpipe,
        "aiv_vec": aiv_vec,
        "aiv_mte2": aiv_mte2,
        "aic_mte2": aic_mte2,
        "aic_total_cycles": aic_total_cycles,
        "aiv_total_cycles": aiv_total_cycles,
        "dur_us": dur_us,
    }


def _measure_msprof_detail_metrics(
    fn_call,
    *,
    warmup: int = 1,
    repeat: int = 3,
) -> dict[str, object]:
    try:
        profiler = importlib.import_module("tilelang.profiler")
        prof = profiler.do_bench(
            fn_call,
            _n_warmup=max(1, int(warmup)),
            _n_repeat=max(1, int(repeat)),
            backend="msprof_detail",
            device=torch.device("npu:0"),
        )
        return dict(_extract_msprof_metrics(prof))
    except Exception as exc:
        return {"error": f"msprof_detail_failed: {type(exc).__name__}: {exc}"[:300]}


def _default_batch_request(source_root: Path, cannbench_repo: Path) -> dict[str, object]:
    batches = []
    for op_name in ("attention_residual", "mla"):
        mod_path = (source_root / get_operator_module_relpath(op_name)).resolve()
        batches.append(
            {
                "operator_name": op_name,
                "entrypoint": get_operator_entrypoint(op_name),
                "candidates": [
                    {"candidate_id": f"{op_name}_baseline", "module_path": str(mod_path)}
                ],
                "anchors": [a.to_dict() for a in get_default_anchors(op_name)],
            }
        )
    return {
        "source_root": str(source_root),
        "cannbench_repo": str(cannbench_repo),
        "warmup": 3,
        "repeat": 5,
        "require_all_passed": True,
        "response_path": "",
        "batches": batches,
    }


def _evaluate_operator_batch(
    batch: dict[str, object],
    *,
    source_root: Path,
    cannbench_repo: Path,
    warmup: int,
    repeat: int,
    collect_msprof_detail: bool = False,
    msprof_out: dict[str, dict[str, dict[str, object]]] | None = None,
) -> dict[str, CandidateSurrogateResult]:
    op_name = str(batch["operator_name"])
    entrypoint = str(batch.get("entrypoint") or get_operator_entrypoint(op_name))
    anchors = [AnchorCaseSpec.from_dict(item) for item in batch["anchors"]]
    candidates = list(batch["candidates"])
    raw_msprof_cases = batch.get("msprof_case_nums")
    if raw_msprof_cases is not None:
        msprof_case_set = {int(c) for c in raw_msprof_cases}
    elif collect_msprof_detail:
        msprof_case_set = {int(a.case_num) for a in anchors}
    else:
        msprof_case_set = set()
    msprof_warmup = int(batch.get("msprof_warmup", 1))
    msprof_repeat = int(batch.get("msprof_repeat", 3))

    contract_mod = load_cannbench_contract_module(source_root=source_root)
    harness, cases = load_anchor_harness_and_cases(
        op_name,
        anchors,
        cannbench_repo=cannbench_repo,
        source_root=source_root,
    )
    case_by_num = {int(c.case_num): c for c in cases}

    loaded_cands: list[tuple[str, object, object]] = []
    for idx, cand in enumerate(candidates):
        cid = str(cand["candidate_id"])
        mod_path = Path(str(cand["module_path"]))
        tag = f"{op_name}_{idx}_{int(time.monotonic_ns())}"
        fn, mod_obj = _load_candidate_entrypoint(mod_path, entrypoint, tag)
        loaded_cands.append((cid, fn, mod_obj))

    per_cand_metrics: dict[str, list[AnchorEvalMetrics]] = {
        cid: [] for cid, _fn, _mod in loaded_cands
    }

    with contract_mod.ReferenceCache(harness) as cache:
        for spec in anchors:
            case = case_by_num[int(spec.case_num)]

            def _prep(c=case, h=harness):
                ts = h.prepare(c)
                return {"inputs": ts, "oracle": h.reference_output(c, ts)}

            bundle = cache.get(case, "oracle", _prep)
            tensors = bundle["inputs"]
            oracle = bundle["oracle"]

            def _native_fn(c=case, ts=tensors, h=harness, ch=cache):
                return ch.get(c, "native", lambda: h.reference_output(c, ts, native=True))

            for cid, fn, _mod in loaded_cands:
                npu_tensors = None
                params = None
                actual = None
                try:
                    npu_tensors = contract_mod.map_tensors(tensors, lambda x: x.npu())
                    params = harness.params(case, npu_tensors)
                    actual = fn(**params)
                    torch.npu.synchronize()
                    passed, details = harness.compare(
                        case, actual, oracle, tensors, _native_fn
                    )
                    err_msg = ""
                    if not passed:
                        err_msg = json.dumps(details)[:300]
                    cand_us = _measure_npu_latency_us(
                        lambda f=fn, p=params: f(**p),
                        warmup=warmup,
                        repeat=repeat,
                    )
                    per_cand_metrics[cid].append(
                        build_anchor_metrics(
                            spec,
                            accuracy_passed=passed,
                            cand_us=cand_us,
                            error_message=err_msg,
                        )
                    )
                    if passed and msprof_out is not None and int(spec.case_num) in msprof_case_set:
                        msprof_out.setdefault(cid, {})[str(int(spec.case_num))] = (
                            _measure_msprof_detail_metrics(
                                lambda f=fn, p=params: f(**p),
                                warmup=msprof_warmup,
                                repeat=msprof_repeat,
                            )
                        )
                except Exception as exc:
                    per_cand_metrics[cid].append(
                        build_anchor_metrics(
                            spec,
                            accuracy_passed=False,
                            cand_us=0.0,
                            error_message=f"{type(exc).__name__}: {exc}"[:300],
                        )
                    )
                finally:
                    actual = None
                    params = None
                    npu_tensors = None
                    torch.npu.empty_cache()

    for _cid, _fn, mod_obj in loaded_cands:
        cache_obj = getattr(mod_obj, "_KERNEL_CACHE", None)
        if isinstance(cache_obj, dict):
            cache_obj.clear()
    gc.collect()
    torch.npu.empty_cache()

    return {
        cid: aggregate_surrogate_result(cid, op_name, per_cand_metrics[cid])
        for cid, _fn, _mod in loaded_cands
    }


def _collect_msprof_targets(
    targets: list[Mapping[str, object]],
    *,
    source_root: Path,
    cannbench_repo: Path,
) -> dict[str, dict[str, dict[str, dict[str, object]]]]:
    by_op: dict[str, list[Mapping[str, object]]] = {}
    for row in targets:
        if isinstance(row, Mapping):
            op_name = str(row["operator_name"])
            by_op.setdefault(op_name, []).append(row)

    telemetry_out: dict[str, dict[str, dict[str, dict[str, object]]]] = {}
    contract_mod = load_cannbench_contract_module(source_root=source_root)
    for op_name, rows in by_op.items():
        entrypoint = get_operator_entrypoint(op_name)
        case_nums = sorted({int(r["case_num"]) for r in rows})
        specs = [AnchorCaseSpec("", cnum, "", 1.0, 1.0) for cnum in case_nums]
        harness, cases = load_anchor_harness_and_cases(
            op_name,
            specs,
            cannbench_repo=cannbench_repo,
            source_root=source_root,
        )
        case_by_num = {int(c.case_num): c for c in cases}

        loaded_mods: dict[tuple[str, str], tuple[object, object]] = {}
        for idx, row in enumerate(rows):
            cid = str(row["candidate_id"])
            mod_path = str(Path(str(row["module_path"])).resolve())
            key = (cid, mod_path)
            if key not in loaded_mods:
                ep = str(row.get("entrypoint") or entrypoint)
                tag = f"msprof_{op_name}_{idx}_{int(time.monotonic_ns())}"
                loaded_mods[key] = _load_candidate_entrypoint(Path(mod_path), ep, tag)

        with contract_mod.ReferenceCache(harness) as cache:
            for row in rows:
                cid = str(row["candidate_id"])
                mod_path = str(Path(str(row["module_path"])).resolve())
                cnum = int(row["case_num"])
                fn, _mod_obj = loaded_mods[(cid, mod_path)]
                case = case_by_num[cnum]

                def _prep(c=case, h=harness):
                    ts = h.prepare(c)
                    return {"inputs": ts, "oracle": h.reference_output(c, ts)}

                bundle = cache.get(case, "oracle", _prep)
                tensors = bundle["inputs"]
                npu_tensors = None
                params = None
                try:
                    npu_tensors = contract_mod.map_tensors(tensors, lambda x: x.npu())
                    params = harness.params(case, npu_tensors)
                    metrics = _measure_msprof_detail_metrics(
                        lambda f=fn, p=params: f(**p),
                        warmup=int(row.get("warmup", 1)),
                        repeat=int(row.get("repeat", 3)),
                    )
                except Exception as exc:
                    metrics = {"error": f"msprof_target_failed: {type(exc).__name__}: {exc}"[:300]}
                finally:
                    params = None
                    npu_tensors = None
                    torch.npu.empty_cache()
                telemetry_out.setdefault(op_name, {}).setdefault(cid, {})[str(cnum)] = metrics

        for _fn, mod_obj in loaded_mods.values():
            cache_obj = getattr(mod_obj, "_KERNEL_CACHE", None)
            if isinstance(cache_obj, dict):
                cache_obj.clear()
        gc.collect()
        torch.npu.empty_cache()
    return telemetry_out


_FAST_SHA256 = re.compile(r"[0-9a-f]{64}")


class _FastRequestError(RuntimeError):
    """A fast request contract violation reported as a request_error response."""


@dataclass(frozen=True)
class _FastCandidate:
    operator_name: str
    candidate_id: str
    module_path: Path
    relative_path: str
    sha256: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fast_int(value: object, field: str, *, default: int) -> int:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise _FastRequestError(f"fast request field {field!r} must be a positive integer")
    return value


def _fast_source_root(source_root_raw: str) -> Path:
    raw = source_root_raw.strip()
    if not raw:
        raise RuntimeError(
            f"fast request requires {FAST_SOURCE_ROOT_ENV} from the AgentGates adapter"
        )
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise RuntimeError(f"fast request environment {FAST_SOURCE_ROOT_ENV} must be absolute")
    resolved = path.resolve()
    if not resolved.is_dir():
        raise RuntimeError(
            f"fast request environment {FAST_SOURCE_ROOT_ENV} must be a directory: {resolved}"
        )
    return resolved


def _fast_env_dir(name: str, *, create: bool) -> Path:
    raw = os.environ.get(name, "").strip()
    if not raw:
        raise RuntimeError(f"fast request requires {name} from the AgentGates adapter")
    path = Path(raw).expanduser()
    if not path.is_absolute():
        raise RuntimeError(f"fast request environment {name} must be absolute")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    elif not path.is_dir():
        raise RuntimeError(f"fast request environment {name} must be a directory: {path}")
    return path.resolve()


def _fast_path(source_root: Path, value: object, field: str) -> Path:
    raw = str(value or "").strip()
    if not raw:
        raise _FastRequestError(f"fast request field {field!r} is required")
    relative = Path(raw)
    if relative.is_absolute() or ".." in relative.parts:
        raise _FastRequestError(f"fast request field {field!r} must be workspace-relative")
    unresolved = source_root / relative
    if unresolved.is_symlink():
        raise _FastRequestError(f"fast request field {field!r} must not be a symlink")
    try:
        resolved = unresolved.resolve()
        resolved.relative_to(source_root)
    except ValueError:
        raise _FastRequestError(
            f"fast request field {field!r} escapes the source root"
        ) from None
    return resolved


def _fast_candidate(
    source_root: Path,
    operator_name: str,
    row: Mapping[str, object],
    used: set[tuple[str, str]],
) -> _FastCandidate:
    candidate_id = str(row.get("candidate_id", "")).strip()
    if not candidate_id:
        raise _FastRequestError(
            f"fast request candidate under {operator_name!r} has no candidate_id"
        )
    key = (operator_name, candidate_id)
    if key in used:
        raise _FastRequestError(
            f"duplicate fast request candidate {operator_name}/{candidate_id}"
        )
    used.add(key)
    relative_path = str(row.get("module_path", "")).strip()
    module_path = _fast_path(source_root, relative_path, "module_path")
    if not module_path.is_file():
        raise _FastRequestError(
            f"fast request module {relative_path!r} is not a regular file"
        )
    sha256 = str(row.get("sha256", "")).strip().lower()
    if not _FAST_SHA256.fullmatch(sha256):
        raise _FastRequestError(
            f"fast request module {relative_path!r} has no valid sha256"
        )
    if _sha256_file(module_path) != sha256:
        raise _FastRequestError(
            f"fast request module {relative_path!r} does not match its declared sha256"
        )
    return _FastCandidate(
        operator_name=operator_name,
        candidate_id=candidate_id,
        module_path=module_path,
        relative_path=relative_path,
        sha256=sha256,
    )


def _fast_bindings(candidates: list[_FastCandidate]) -> dict[str, dict[str, dict[str, str]]]:
    bindings: dict[str, dict[str, dict[str, str]]] = {}
    for candidate in candidates:
        post_sha256 = _sha256_file(candidate.module_path)
        if post_sha256 != candidate.sha256:
            raise RuntimeError(
                f"fast gate {FAST_STATUS_INFRASTRUCTURE_ERROR}: candidate "
                f"{candidate.operator_name}/{candidate.candidate_id} changed during evaluation"
            )
        bindings.setdefault(candidate.operator_name, {})[candidate.candidate_id] = {
            "module_path": candidate.relative_path,
            "sha256": candidate.sha256,
            "post_sha256": post_sha256,
        }
    return bindings


def _fast_response_payload(
    *,
    request_id: str,
    request_sha256: str,
    mode: str,
    status: str,
    error: str = "",
    operators: Mapping[str, object] | None = None,
    msprof_telemetry: Mapping[str, object] | None = None,
    candidates: Mapping[str, object] | None = None,
    coverage: Mapping[str, int] | None = None,
    micro_probe: Mapping[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema": FAST_RESPONSE_SCHEMA,
        "request_id": request_id,
        "request_sha256": request_sha256,
        "mode": mode,
        "status": status,
        "error": error[-1000:],
        "operators": dict(operators or {}),
        "msprof_telemetry": dict(msprof_telemetry or {}),
        "candidates": dict(candidates or {}),
        "coverage": dict(
            coverage
            or {
                "expected_candidates": 0,
                "evaluated_candidates": 0,
                "expected_msprof_targets": 0,
                "observed_msprof_targets": 0,
            }
        ),
    }
    if micro_probe is not None:
        payload["micro_probe"] = dict(micro_probe)
    return payload


def _run_fast_batch(
    req: Mapping[str, object],
    *,
    source_root: Path,
    cannbench_repo: Path,
    response_path: Path,
    request_id: str,
    request_sha256: str,
) -> None:
    raw_batches = req.get("batches")
    if not isinstance(raw_batches, list) or not raw_batches:
        raise _FastRequestError("fast batch request requires a non-empty batches list")
    if req.get("msprof_targets") not in (None, []):
        raise _FastRequestError("fast batch request must not declare msprof_targets")
    warmup = _fast_int(req.get("warmup"), "warmup", default=3)
    repeat = _fast_int(req.get("repeat"), "repeat", default=5)

    used_candidates: set[tuple[str, str]] = set()
    operators: set[str] = set()
    candidates: list[_FastCandidate] = []
    normalized_batches: list[dict[str, object]] = []
    for row in raw_batches:
        if not isinstance(row, Mapping):
            raise _FastRequestError("fast batch entries must be objects")
        operator_name = str(row.get("operator_name", "")).strip()
        if not operator_name:
            raise _FastRequestError("fast batch entry has no operator_name")
        if operator_name in operators:
            raise _FastRequestError(f"duplicate fast batch operator {operator_name!r}")
        operators.add(operator_name)
        raw_candidates = row.get("candidates")
        if not isinstance(raw_candidates, list) or not raw_candidates:
            raise _FastRequestError(
                f"fast batch operator {operator_name!r} has no candidates"
            )
        normalized_candidates: list[dict[str, str]] = []
        for cand_row in raw_candidates:
            if not isinstance(cand_row, Mapping):
                raise _FastRequestError("fast batch candidates must be objects")
            candidate = _fast_candidate(source_root, operator_name, cand_row, used_candidates)
            candidates.append(candidate)
            normalized_candidates.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "module_path": str(candidate.module_path),
                }
            )
        raw_anchors = row.get("anchors")
        if not isinstance(raw_anchors, list) or not raw_anchors:
            raise _FastRequestError(
                f"fast batch operator {operator_name!r} has no anchors"
            )
        normalized_batches.append(
            {
                "operator_name": operator_name,
                "entrypoint": str(row.get("entrypoint") or get_operator_entrypoint(operator_name)),
                "candidates": normalized_candidates,
                "anchors": raw_anchors,
            }
        )

    results_by_op: dict[str, dict[str, dict[str, object]]] = {}
    failures: list[str] = []
    for batch in normalized_batches:
        op_name = str(batch["operator_name"])
        op_res = _evaluate_operator_batch(
            batch,
            source_root=source_root,
            cannbench_repo=cannbench_repo,
            warmup=warmup,
            repeat=repeat,
        )
        results_by_op[op_name] = {cid: res.to_dict() for cid, res in op_res.items()}
        failures.extend(
            f"{op_name}/{cid}" for cid, res in op_res.items() if not res.all_passed
        )
    payload = _fast_response_payload(
        request_id=request_id,
        request_sha256=request_sha256,
        mode="batch",
        status=FAST_STATUS_OK if not failures else FAST_STATUS_CANDIDATE_FAILURE,
        operators=results_by_op,
        candidates=_fast_bindings(candidates),
        coverage={
            "expected_candidates": len(candidates),
            "evaluated_candidates": sum(len(cands) for cands in results_by_op.values()),
            "expected_msprof_targets": 0,
            "observed_msprof_targets": 0,
        },
    )
    _atomic_write_json(response_path, payload)
    print(json.dumps(payload, sort_keys=True), flush=True)


def _run_fast_profile(
    req: Mapping[str, object],
    *,
    source_root: Path,
    cannbench_repo: Path,
    response_path: Path,
    request_id: str,
    request_sha256: str,
) -> None:
    raw_targets = req.get("msprof_targets")
    if not isinstance(raw_targets, list) or not raw_targets:
        raise _FastRequestError("fast profile request requires a non-empty msprof_targets list")
    if req.get("batches") not in (None, []):
        raise _FastRequestError("fast profile request must not declare batches")
    default_warmup = _fast_int(req.get("warmup"), "warmup", default=1)
    default_repeat = _fast_int(req.get("repeat"), "repeat", default=3)

    used_candidates: set[tuple[str, str]] = set()
    candidate_by_key: dict[tuple[str, str], _FastCandidate] = {}
    seen_targets: set[tuple[str, str, int]] = set()
    candidates: list[_FastCandidate] = []
    normalized_rows: list[dict[str, object]] = []
    for row in raw_targets:
        if not isinstance(row, Mapping):
            raise _FastRequestError("fast profile targets must be objects")
        operator_name = str(row.get("operator_name", "")).strip()
        if not operator_name:
            raise _FastRequestError("fast profile target has no operator_name")
        declared_candidate_id = str(row.get("candidate_id", "")).strip()
        if not declared_candidate_id:
            raise _FastRequestError(
                f"fast request candidate under {operator_name!r} has no candidate_id"
            )
        key = (operator_name, declared_candidate_id)
        candidate = candidate_by_key.get(key)
        if candidate is None:
            candidate = _fast_candidate(source_root, operator_name, row, used_candidates)
            candidate_by_key[key] = candidate
            candidates.append(candidate)
        elif (
            str(row.get("module_path", "")).strip() != candidate.relative_path
            or str(row.get("sha256", "")).strip().lower() != candidate.sha256
        ):
            raise _FastRequestError(
                f"fast profile candidate {operator_name}/{candidate.candidate_id} "
                "changes module_path or sha256 across targets"
            )
        case_num = row.get("case_num")
        if isinstance(case_num, bool) or not isinstance(case_num, int) or case_num <= 0:
            raise _FastRequestError(
                "fast profile target case_num must be a positive integer"
            )
        target_key = (operator_name, candidate.candidate_id, case_num)
        if target_key in seen_targets:
            raise _FastRequestError(
                f"duplicate fast profile target {operator_name}/"
                f"{candidate.candidate_id}#{case_num}"
            )
        seen_targets.add(target_key)
        normalized_rows.append(
            {
                "operator_name": operator_name,
                "candidate_id": candidate.candidate_id,
                "module_path": str(candidate.module_path),
                "case_num": case_num,
                "warmup": _fast_int(row.get("warmup"), "warmup", default=default_warmup),
                "repeat": _fast_int(row.get("repeat"), "repeat", default=default_repeat),
            }
        )

    telemetry = _collect_msprof_targets(
        normalized_rows,
        source_root=source_root,
        cannbench_repo=cannbench_repo,
    )
    observed_targets = sum(
        len(cases) for cands in telemetry.values() for cases in cands.values()
    )
    payload = _fast_response_payload(
        request_id=request_id,
        request_sha256=request_sha256,
        mode="profile",
        status=FAST_STATUS_OK,
        msprof_telemetry=telemetry,
        candidates=_fast_bindings(candidates),
        coverage={
            "expected_candidates": len(candidates),
            "evaluated_candidates": len(candidates),
            "expected_msprof_targets": len(raw_targets),
            "observed_msprof_targets": observed_targets,
        },
    )
    _atomic_write_json(response_path, payload)
    print(json.dumps(payload, sort_keys=True), flush=True)


def _run_fast_micro_probe(
    req: Mapping[str, object],
    *,
    source_root: Path,
    response_path: Path,
    request_id: str,
    request_sha256: str,
) -> None:
    raw_probe = req.get("probe")
    if not isinstance(raw_probe, Mapping):
        raise _FastRequestError("fast micro_probe request requires a probe object")
    probe_id = str(raw_probe.get("probe_id", "")).strip()
    probe_kind = str(raw_probe.get("probe_kind", "")).strip()
    factory_name = str(raw_probe.get("factory_name", "")).strip()
    tensor_dtype = str(raw_probe.get("tensor_dtype", "")).strip()
    if not probe_id or not probe_kind or not factory_name or not tensor_dtype:
        raise _FastRequestError(
            "fast micro_probe request probe requires probe_id, probe_kind, "
            "factory_name, and tensor_dtype"
        )
    line_counts: dict[str, int] = {}
    for field in ("line_count_baseline", "line_count_optimized"):
        value = raw_probe.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise _FastRequestError(
                f"fast request field 'probe.{field}' must be a non-negative integer"
            )
        line_counts[field] = value
    raw_shape = raw_probe.get("tensor_shape")
    if (
        not isinstance(raw_shape, list)
        or not 1 <= len(raw_shape) <= 8
        or any(
            isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0
            for dim in raw_shape
        )
    ):
        raise _FastRequestError(
            "fast request field 'probe.tensor_shape' must list 1 to 8 positive integers"
        )
    min_speedup = raw_probe.get("min_speedup")
    if (
        isinstance(min_speedup, bool)
        or not isinstance(min_speedup, (int, float))
        or not math.isfinite(min_speedup)
        or min_speedup <= 0.0
    ):
        raise _FastRequestError(
            "fast request field 'probe.min_speedup' must be a finite positive number"
        )

    raw_candidates = req.get("candidates")
    if not isinstance(raw_candidates, list) or len(raw_candidates) != 2:
        raise _FastRequestError(
            "fast micro_probe request requires baseline and optimized candidates"
        )
    used_candidates: set[tuple[str, str]] = set()
    candidates = [
        _fast_candidate(source_root, "micro_probe", row, used_candidates)
        for row in raw_candidates
        if isinstance(row, Mapping)
    ]
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    if len(candidates) != 2 or set(by_id) != {"baseline", "optimized"}:
        raise _FastRequestError(
            "fast micro_probe request requires baseline and optimized candidates"
        )
    spec = MicroProbeSpec(
        probe_id=probe_id,
        probe_kind=probe_kind,
        factory_name=factory_name,
        baseline_probe_source=by_id["baseline"].module_path.read_text(encoding="utf-8"),
        optimized_probe_source=by_id["optimized"].module_path.read_text(encoding="utf-8"),
        line_count_baseline=line_counts["line_count_baseline"],
        line_count_optimized=line_counts["line_count_optimized"],
        tensor_shape=tuple(raw_shape),
        tensor_dtype=tensor_dtype,
    )
    passed, metrics = verify_micro_probe_hardware(
        spec,
        min_speedup=float(min_speedup),
    )
    payload = _fast_response_payload(
        request_id=request_id,
        request_sha256=request_sha256,
        mode="micro_probe",
        status=FAST_STATUS_OK if passed else FAST_STATUS_CANDIDATE_FAILURE,
        candidates=_fast_bindings(candidates),
        coverage={
            "expected_candidates": len(candidates),
            "evaluated_candidates": len(candidates),
            "expected_msprof_targets": 0,
            "observed_msprof_targets": 0,
        },
        micro_probe={
            "probe_id": probe_id,
            "metrics": {str(key): float(value) for key, value in metrics.items()},
        },
    )
    _atomic_write_json(response_path, payload)
    print(json.dumps(payload, sort_keys=True), flush=True)


def _handle_fast_request(request_path_raw: str, source_root_raw: str, shim_dir: Path) -> None:
    request_path = Path(request_path_raw)
    if not request_path.is_file():
        raise RuntimeError(f"fast request {request_path} does not exist")
    request_bytes = request_path.read_bytes()
    request_sha256 = hashlib.sha256(request_bytes).hexdigest()
    try:
        req = json.loads(request_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"fast request {request_path} is not valid JSON: {exc}") from None
    if not isinstance(req, dict):
        raise RuntimeError(f"fast request {request_path} must be a JSON object")
    source_root = _fast_source_root(source_root_raw)
    response_path = _fast_path(source_root, req.get("response_path"), "response_path")
    request_id = str(req.get("request_id", "")).strip()
    mode = str(req.get("mode", "")).strip()

    def _fail(status: str, message: str) -> None:
        _atomic_write_json(
            response_path,
            _fast_response_payload(
                request_id=request_id,
                request_sha256=request_sha256,
                mode=mode,
                status=status,
                error=message,
            ),
        )

    try:
        if req.get("schema") != FAST_REQUEST_SCHEMA:
            raise _FastRequestError(
                f"fast request schema {req.get('schema')!r} != {FAST_REQUEST_SCHEMA!r}"
            )
        if not request_id:
            raise _FastRequestError("fast request has no request_id")
        if mode not in ("batch", "profile", "micro_probe"):
            raise _FastRequestError(f"fast request mode {mode!r} is not recognized")
        if mode == "micro_probe":
            tl_cache_raw = os.environ.get("TILELANG_CACHE_DIR", "").strip()
            if tl_cache_raw:
                _configure_tilelang_cache(tl_cache_raw)
            _run_fast_micro_probe(
                req,
                source_root=source_root,
                response_path=response_path,
                request_id=request_id,
                request_sha256=request_sha256,
            )
            return
        cannbench_repo = _fast_env_dir(
            "TILEFOUNDRY_NPU_CANNBENCH_REPO", create=False
        )
        tl_cache_dir = _fast_env_dir("TILELANG_CACHE_DIR", create=True)
        reference_cache_dir = _fast_env_dir("CANNBENCH_CACHE_DIR", create=True)
        os.environ["CANNBENCH_CACHE_DIR"] = str(reference_cache_dir)
        _ensure_npu_compare(cannbench_repo, shim_dir)
        _configure_tilelang_cache(str(tl_cache_dir))
        if mode == "batch":
            _run_fast_batch(
                req,
                source_root=source_root,
                cannbench_repo=cannbench_repo,
                response_path=response_path,
                request_id=request_id,
                request_sha256=request_sha256,
            )
        else:
            _run_fast_profile(
                req,
                source_root=source_root,
                cannbench_repo=cannbench_repo,
                response_path=response_path,
                request_id=request_id,
                request_sha256=request_sha256,
            )
    except _FastRequestError as exc:
        _fail(FAST_STATUS_REQUEST_ERROR, str(exc))
    except Exception as exc:
        _fail(FAST_STATUS_INFRASTRUCTURE_ERROR, f"{type(exc).__name__}: {exc}")
        raise


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_path, path)


def _configure_tilelang_cache(tl_cache_dir: str) -> None:
    resolved = Path(tl_cache_dir).resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    os.environ["TILELANG_CACHE_DIR"] = str(resolved)
    os.environ["TILELANG_KERNEL_CACHE_USE_LIB_STAMP"] = "1"
    os.environ["TILELANG_DISABLE_CACHE"] = "0"
    tl_env_mod = importlib.import_module("tilelang.env")
    tl_env_mod.env.enable_cache()
    kc_mod = importlib.import_module("tilelang.cache.kernel_cache")
    kc_mod.KernelCache._get_cache_namespace.cache_clear()
    kc_mod.KernelCache._get_tilelang_lib_stamp.cache_clear()
    kc_mod.KernelCache._create_dirs()


def test_fast_bindings_treat_candidate_change_as_infrastructure(tmp_path: Path) -> None:
    """A candidate rewritten after evaluation invalidates the run without blaming the request."""
    module = tmp_path / "kernel.py"
    module.write_text("x = 1\n", encoding="utf-8")
    candidate = _FastCandidate(
        operator_name="softmax",
        candidate_id="cand-1",
        module_path=module,
        relative_path="kernel.py",
        sha256=_sha256_file(module),
    )
    module.write_text("x = 2\n", encoding="utf-8")
    with pytest.raises(RuntimeError) as failure:
        _fast_bindings([candidate])
    assert not isinstance(failure.value, _FastRequestError)
    assert FAST_STATUS_INFRASTRUCTURE_ERROR in str(failure.value)
    assert "changed during evaluation" in str(failure.value)


def test_surrogate_anchor_batch_npu() -> None:
    """Evaluate configured or baseline 10-anchor surrogate batch inside AgentGates lease."""
    _require_npu_env()
    torch.npu.set_device(0)

    fast_req_env = os.environ.get(FAST_REQUEST_ENV, "").strip()
    req_env = os.environ.get("TF_SURROGATE_BATCH_REQUEST", "").strip()
    if fast_req_env and req_env:
        raise RuntimeError(
            f"{FAST_REQUEST_ENV} and TF_SURROGATE_BATCH_REQUEST must not both be set"
        )
    shim_only = os.environ.get("TF_SURROGATE_SHIM_ONLY", "").strip() == "1"
    shim_env_dir = os.environ.get("TF_SURROGATE_SHIM_DIR", "").strip()
    tl_cache_env = os.environ.get("TF_SURROGATE_TL_CACHE_DIR", "").strip()

    if shim_only:
        if tl_cache_env:
            _configure_tilelang_cache(tl_cache_env)
        if shim_env_dir and (Path(shim_env_dir) / "npu_compare.py").is_file():
            if shim_env_dir not in sys.path:
                sys.path.insert(0, shim_env_dir)
            importlib.import_module("npu_compare")
            return
        target_shim = (
            Path(shim_env_dir)
            if shim_env_dir
            else Path(tempfile.mkdtemp(prefix="tf_surrogate_shim_persist_"))
        )
        cann_repo = _resolve_batch_cannbench_repo(default_source_root())
        _ensure_npu_compare(cann_repo, target_shim)
        importlib.import_module("npu_compare")
        return

    with tempfile.TemporaryDirectory(prefix="tf_surrogate_shim_") as shim_dir:
        if fast_req_env:
            if not Path(fast_req_env).is_file():
                raise RuntimeError(f"{FAST_REQUEST_ENV} does not exist: {fast_req_env}")
            _handle_fast_request(
                fast_req_env,
                os.environ.get(FAST_SOURCE_ROOT_ENV, "").strip(),
                Path(shim_dir),
            )
            return
        if req_env and Path(req_env).is_file():
            req = json.loads(Path(req_env).read_text(encoding="utf-8"))
            source_root = Path(str(req["source_root"])).resolve()
            cann_repo = Path(str(req["cannbench_repo"])).resolve()
        else:
            source_root = default_source_root()
            cann_repo = _resolve_batch_cannbench_repo(source_root)
            req = _default_batch_request(source_root, cann_repo)

        _ensure_npu_compare(cann_repo, Path(shim_dir))

        if not os.environ.get("CANNBENCH_CACHE_DIR"):
            uid = os.getuid() if hasattr(os, "getuid") else 0
            cache_dir = Path(tempfile.gettempdir()) / f"tf_surrogate_ref_cache_{uid}"
            cache_dir.mkdir(parents=True, exist_ok=True)
            os.environ["CANNBENCH_CACHE_DIR"] = str(cache_dir)

        uid = os.getuid() if hasattr(os, "getuid") else 0
        default_tl_cache = str(Path(tempfile.gettempdir()) / f"tf_surrogate_tl_cache_{uid}")
        tl_cache_dir = (
            str(req.get("tilelang_cache_dir", "")).strip()
            or os.environ.get("TF_SURROGATE_TL_CACHE_DIR", "").strip()
            or default_tl_cache
        )
        _configure_tilelang_cache(tl_cache_dir)

        warmup = int(req.get("warmup", 3))
        repeat = int(req.get("repeat", 5))
        require_all_passed = bool(req.get("require_all_passed", False))
        collect_msprof_detail = bool(req.get("collect_msprof_detail", False))
        resp_path_raw = str(req.get("response_path", "")).strip()

        results_by_op: dict[str, dict[str, dict[str, object]]] = {}
        parsed_by_op: dict[str, dict[str, CandidateSurrogateResult]] = {}
        msprof_telemetry: dict[str, dict[str, dict[str, dict[str, object]]]] = {}
        for batch in req.get("batches", ()):
            op_msprof: dict[str, dict[str, dict[str, object]]] = {}
            op_res = _evaluate_operator_batch(
                batch,
                source_root=source_root,
                cannbench_repo=cann_repo,
                warmup=warmup,
                repeat=repeat,
                collect_msprof_detail=collect_msprof_detail,
                msprof_out=op_msprof,
            )
            op_name = str(batch["operator_name"])
            parsed_by_op[op_name] = op_res
            results_by_op[op_name] = {cid: res.to_dict() for cid, res in op_res.items()}
            if op_msprof:
                msprof_telemetry[op_name] = op_msprof

        raw_msprof_targets = req.get("msprof_targets") or []
        if raw_msprof_targets:
            target_tel = _collect_msprof_targets(
                list(raw_msprof_targets),
                source_root=source_root,
                cannbench_repo=cann_repo,
            )
            for op_name, cands_map in target_tel.items():
                op_bucket = msprof_telemetry.setdefault(op_name, {})
                for cid, cases_map in cands_map.items():
                    op_bucket.setdefault(cid, {}).update(cases_map)

        summary_payload: dict[str, object] = {"operators": results_by_op}
        if msprof_telemetry or collect_msprof_detail or raw_msprof_targets:
            summary_payload["msprof_telemetry"] = msprof_telemetry
        if resp_path_raw:
            _atomic_write_json(Path(resp_path_raw), summary_payload)
        print(json.dumps(summary_payload, sort_keys=True), flush=True)

        if require_all_passed:
            for op_name, op_res in parsed_by_op.items():
                for cid, res in op_res.items():
                    assert res.all_passed, f"{op_name}/{cid} failed: {res.to_dict()}"
                    for anchor in res.anchors:
                        assert anchor.cand_us > 0.0, f"{op_name}/{cid} #{anchor.case_num} invalid time"
