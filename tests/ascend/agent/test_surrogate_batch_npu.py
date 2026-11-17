"""Lease-locked AgentGates L2 surrogate CANNBench batch evaluator on NPU."""

from __future__ import annotations

import gc
import importlib
import importlib.util
import json
import os
import statistics
import sys
import tempfile
import time
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
    AnchorEvalMetrics,
    CandidateSurrogateResult,
    aggregate_surrogate_result,
    build_anchor_metrics,
    ensure_npu_compare_shim_dir,
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
        if os.environ.get("TF_SURROGATE_BATCH_REQUEST", "").strip():
            raise RuntimeError("Ascend NPU device is unavailable for surrogate batch request.")
        pytest.skip("Ascend NPU device is unavailable on this host.")


def _resolve_batch_cannbench_repo(source_root: Path) -> Path:
    explicit = os.environ.get("TF_SURROGATE_CANNBENCH_REPO", "").strip()
    if explicit and Path(explicit).is_dir():
        return Path(explicit).resolve()
    try:
        return resolve_cannbench_repo()
    except FileNotFoundError:
        sibling = source_root.parent / "cannbot-ops"
        if (sibling / "operators").is_dir():
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
) -> dict[str, CandidateSurrogateResult]:
    op_name = str(batch["operator_name"])
    entrypoint = str(batch.get("entrypoint") or get_operator_entrypoint(op_name))
    anchors = [AnchorCaseSpec.from_dict(item) for item in batch["anchors"]]
    candidates = list(batch["candidates"])

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


def test_surrogate_anchor_batch_npu() -> None:
    """Evaluate configured or baseline 10-anchor surrogate batch inside AgentGates lease."""
    _require_npu_env()
    torch.npu.set_device(0)

    req_env = os.environ.get("TF_SURROGATE_BATCH_REQUEST", "").strip()
    shim_only = os.environ.get("TF_SURROGATE_SHIM_ONLY", "").strip() == "1"
    shim_env_dir = os.environ.get("TF_SURROGATE_SHIM_DIR", "").strip()

    if shim_only:
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
        resp_path_raw = str(req.get("response_path", "")).strip()

        results_by_op: dict[str, dict[str, dict[str, object]]] = {}
        parsed_by_op: dict[str, dict[str, CandidateSurrogateResult]] = {}
        for batch in req["batches"]:
            op_res = _evaluate_operator_batch(
                batch,
                source_root=source_root,
                cannbench_repo=cann_repo,
                warmup=warmup,
                repeat=repeat,
            )
            op_name = str(batch["operator_name"])
            parsed_by_op[op_name] = op_res
            results_by_op[op_name] = {cid: res.to_dict() for cid, res in op_res.items()}

        summary_payload = {"operators": results_by_op}
        if resp_path_raw:
            _atomic_write_json(Path(resp_path_raw), summary_payload)
        print(json.dumps(summary_payload, sort_keys=True), flush=True)

        if require_all_passed:
            for op_name, op_res in parsed_by_op.items():
                for cid, res in op_res.items():
                    assert res.all_passed, f"{op_name}/{cid} failed: {res.to_dict()}"
                    for anchor in res.anchors:
                        assert anchor.cand_us > 0.0, f"{op_name}/{cid} #{anchor.case_num} invalid time"
