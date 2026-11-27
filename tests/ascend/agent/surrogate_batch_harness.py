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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from tilefoundry_ascend.agent import npu as ag_npu
from tilefoundry_ascend.agent.flywheel.ablation_distiller import (
    MicroProbeSpec,
    verify_micro_probe_hardware,
)
from tilefoundry_ascend.agent.npu.ag_transport import _sha256_file
from tilefoundry_ascend.agent.specs import anchor_specs as aspecs
from tilefoundry_ascend.ascend.json_record import atomic_write_text

_INCUMBENT_IDS = frozenset({"incumbent_slot", "__incumbent__", "incumbent_baseline"})
_SKIP_PILL_IDS = frozenset({"incumbent_slot", "__incumbent__"})
_FAST_SHA256 = re.compile(r"[0-9a-f]{64}")
_MSP_KEYS = (
    *((k, (k,)) for k in ("aic_scalar", "aic_mte1", "aic_mte3", "aic_fixpipe", "aiv_scalar",
                          "aiv_mte3", "aic_mte2", "aiv_mte2", "sync_stall_ratio")),
    ("aic_mac_ratio", ("aic_mad", "aic_mac_ratio")),
    ("aiv_vec_ratio", ("aiv_vec", "aiv_vec_ratio")),
    *((f"{k}_ratio", (f"{k}_ratio", k)) for k in ("aiv0_vec", "aiv1_vec")),
    ("l2_hit_rate", ("l2_hit_rate", "l2_cache_hit_rate")),
    ("bank_conflict_ratio", ("bank_conflict_ratio", "ub_bank_conflict_ratio",
                             "resource_conflict_ratio")),
)


def _ensure_npu(*, require: bool = False) -> None:
    if not hasattr(torch, "npu"):
        try:
            importlib.import_module("torch_npu")
        except ImportError:
            pass
    if not require or (hasattr(torch, "npu") and torch.npu.is_available()):
        return
    if os.environ.get(ag_npu.FAST_REQUEST_ENV, "").strip() or os.environ.get(
        "TF_SURROGATE_BATCH_REQUEST", ""
    ).strip():
        raise RuntimeError("Ascend NPU device is unavailable for surrogate batch request.")
    pytest.skip("Ascend NPU device is unavailable on this host.")


def _resolve_cannbench(source_root: Path) -> Path:
    if (exp := os.environ.get("TF_SURROGATE_CANNBENCH_REPO", "").strip()) and Path(exp).is_dir():
        return Path(exp).resolve()
    try:
        return aspecs.resolve_cannbench_repo()
    except FileNotFoundError:
        if ((sib := source_root.parent / "cann-bench") / "tasks").is_dir():
            return sib.resolve()
        raise


def _setup_env(cann_repo: Path | None, shim_dir: Path | None, tl_cache_dir: str) -> None:
    if cann_repo and shim_dir and importlib.util.find_spec("npu_compare") is None:
        shim = str(ag_npu.ensure_npu_compare_shim_dir(shim_dir, cannbench_repo=cann_repo))
        if shim not in sys.path:
            sys.path.insert(0, shim)
    if tl_cache_dir:
        resolved = Path(tl_cache_dir).resolve()
        resolved.mkdir(parents=True, exist_ok=True)
        os.environ.update({
            "TILELANG_CACHE_DIR": str(resolved),
            "TILELANG_KERNEL_CACHE_USE_LIB_STAMP": "1",
            "TILELANG_DISABLE_CACHE": "0",
        })
        importlib.import_module("tilelang.env").env.enable_cache()
        kc = importlib.import_module("tilelang.cache.kernel_cache").KernelCache
        kc._get_cache_namespace.cache_clear()
        kc._get_tilelang_lib_stamp.cache_clear()
        kc._create_dirs()


def _load_cand(path: Path, ep: str, tag: str) -> tuple[Callable[..., object], object]:
    mod_name = f"_tf_surrogate_cand_{tag}"
    spec = importlib.util.spec_from_file_location(mod_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load candidate module {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    if not hasattr(mod, ep):
        raise AttributeError(f"{path} does not define {ep!r}")
    return getattr(mod, ep), mod


def _clear_caches(mod_objs: Sequence[object]) -> None:
    for m in mod_objs:
        if isinstance(cache_obj := getattr(m, "_KERNEL_CACHE", None), dict):
            cache_obj.clear()
    gc.collect()
    if hasattr(getattr(torch, "npu", None), "empty_cache"):
        torch.npu.empty_cache()


def _time_events(fn_call: Callable[[], object], *, warmup: int, repeat: int) -> list[float]:
    for _ in range(max(1, warmup)):
        fn_call()
    torch.npu.synchronize()
    out: list[float] = []
    for _ in range(max(1, repeat)):
        s_ev, e_ev = torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
        s_ev.record()
        fn_call()
        e_ev.record()
        torch.npu.synchronize()
        out.append(float(s_ev.elapsed_time(e_ev)) * 1000.0)
    return out


def _measure_null_launch_us(warmup: int = 5, repeat: int = 20) -> float | None:
    npu = getattr(torch, "npu", None)
    if npu is None or not hasattr(npu, "Event") or not hasattr(npu, "synchronize"):
        return None
    if callable(avail := getattr(npu, "is_available", None)) and not avail():
        return None
    try:
        buf = torch.empty((1,), dtype=torch.float32, device="npu:0")
        vals = [v for v in _time_events(buf.zero_, warmup=warmup, repeat=repeat) if v > 0.0]
        med = float(statistics.median(vals)) if vals else 0.0
        return med if math.isfinite(med) and med > 0.0 else None
    except Exception:
        return None


def _measure_msprof(
    fn_call: Callable[[], object] | None, *, warmup: int = 1, repeat: int = 3
) -> dict[str, object]:
    if fn_call is None:
        return {"error": "msprof_detail_failed: no_callable"}

    def _num(*names: str) -> int | float | None:
        for name in names:
            raw = prof.get(name) if isinstance(prof, Mapping) else getattr(prof, name, None)
            if isinstance(raw, (int, float)) and not isinstance(raw, bool) and math.isfinite(raw):
                return raw
        return None

    try:
        prof = importlib.import_module("tilelang.profiler").do_bench(
            fn_call, _n_warmup=max(1, int(warmup)), _n_repeat=max(1, int(repeat)),
            backend="msprof_detail", device=torch.device("npu:0"),
        )
        m: dict[str, object] = {
            ok: max(0.0, min(1.0, (v := float(raw)) / 100.0 if v > 1.05 else v))
            for ok, names in _MSP_KEYS if (raw := _num(*names)) is not None
        }
        for dst, src in (("aic_mad", "aic_mac_ratio"), ("aiv_vec", "aiv_vec_ratio")):
            if src in m:
                m[dst] = m[src]
        if "mte2_wait_ratio" not in m and ("aiv_mte2" in m or "aic_mte2" in m):
            m["mte2_wait_ratio"] = max(float(m[k]) for k in ("aiv_mte2", "aic_mte2") if k in m)
        for k in ("aic_total_cycles", "aiv_total_cycles", "hbm_read_bytes", "hbm_write_bytes",
                  "actual_gm_bytes"):
            if (raw := _num(k)) is not None:
                m[k] = int(raw)
        if "actual_gm_bytes" not in m and ("hbm_read_bytes" in m or "hbm_write_bytes" in m):
            m["actual_gm_bytes"] = int(m.get("hbm_read_bytes", 0)) + int(
                m.get("hbm_write_bytes", 0))
        if (dur := _num("dur_us", "measured_latency_us")) is not None:
            m["dur_us"] = float(dur)
        elif (ns := _num("dur_ns")) is not None:
            m["dur_us"] = float(ns) / 1000.0
        return m
    except Exception as exc:
        return {"error": f"msprof_detail_failed: {type(exc).__name__}: {exc}"[:300]}


def _oracle_bundle(
    cache: object, harness: object, case: object
) -> tuple[object, object, Callable[[], object]]:
    b = cache.get(case, "oracle", lambda: {
        "inputs": (ts := harness.prepare(case)), "oracle": harness.reference_output(case, ts),
    })
    t, o = b["inputs"], b["oracle"]
    return t, o, lambda: cache.get(
        case, "native", lambda: harness.reference_output(case, t, native=True)
    )


def _load_harness(
    op_name: str, anchors: Sequence[aspecs.AnchorCaseSpec], source_root: Path, cann_repo: Path
) -> tuple[object, object, dict[int, object]]:
    cmod = aspecs.load_cannbench_contract_module(source_root=source_root)
    harness, cases = aspecs.load_anchor_harness_and_cases(
        op_name, anchors, cannbench_repo=cann_repo, source_root=source_root
    )
    return cmod, harness, {int(c.case_num): c for c in cases}


def _with_npu_params(
    cmod: object, harness: object, case: object, tensors: object, action: Callable[[object], object]
) -> object:
    try:
        return action(harness.params(case, cmod.map_tensors(tensors, lambda x: x.npu())))
    finally:
        if hasattr(getattr(torch, "npu", None), "empty_cache"):
            torch.npu.empty_cache()


def _run_case_acc(
    fn: Callable[..., object], harness: object, cmod: object, case: object,
    tensors: object, oracle: object, native_fn: Callable[[], object],
) -> tuple[bool, str]:
    try:
        def _check(params: object) -> tuple[bool, str]:
            actual = fn(**params)
            if hasattr(torch, "npu") and hasattr(torch.npu, "synchronize"):
                torch.npu.synchronize()
            ok, details = harness.compare(case, actual, oracle, tensors, native_fn)
            return (True, "") if ok else (False, json.dumps(details)[:300])

        return _with_npu_params(cmod, harness, case, tensors, _check)
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"[:300]


def _fail_m(spec: aspecs.AnchorCaseSpec, err: str) -> ag_npu.AnchorEvalMetrics:
    return ag_npu.build_anchor_metrics(
        spec, accuracy_passed=False, cand_us=0.0, error_message=err[:300]
    )


def _poison_pill_worker_main(req_path: Path, out_path: Path) -> None:
    p = json.loads(req_path.read_text(encoding="utf-8"))
    op, ep = str(p["operator_name"]), str(p["entrypoint"])
    sroot, crepo = Path(str(p["source_root"])), Path(str(p["cannbench_repo"]))
    anchors = [aspecs.AnchorCaseSpec.from_dict(item) for item in p["anchors"]]
    _ensure_npu()
    if hasattr(torch, "npu") and torch.npu.is_available():
        torch.npu.set_device(0)
    if tl_c := os.environ.get("TILELANG_CACHE_DIR", "").strip():
        _setup_env(None, None, tl_c)
    cmod, harness, by_num = _load_harness(op, anchors, sroot, crepo)
    if (rfd := os.environ.get("TF_POISON_PILL_READY_FD", "").strip()).isdigit():
        try:
            os.write(int(rfd), b"READY\n")
        except OSError:
            pass
    results, loaded = {}, []
    for idx, item in enumerate(p["candidates"]):
        cid = str(item["candidate_id"])
        results[cid] = {"all_passed": True, "error": "", "cases": {}}
        try:
            loaded.append((cid, *_load_cand(Path(str(item["module_path"])), ep,
                                            f"p_{op}_{idx}_{time.monotonic_ns()}")))
        except Exception as exc:
            err = f"{type(exc).__name__}: {exc}"[:300]
            results[cid] = {"all_passed": False, "error": err, "cases": {
                str(int(s.case_num)): {"passed": False, "error": err} for s in anchors}}
    try:
        with cmod.ReferenceCache(harness) as cache:
            for spec in anchors:
                cn = int(spec.case_num)
                t, o, nfn = _oracle_bundle(cache, harness, by_num[cn])
                for cid, fn, _m in loaded:
                    ok, err = _run_case_acc(fn, harness, cmod, by_num[cn], t, o, nfn)
                    results[cid]["cases"][str(cn)] = {"passed": ok, "error": err}
                    if not ok:
                        results[cid]["all_passed"] = False
                        results[cid]["error"] = results[cid]["error"] or err
    finally:
        _clear_caches([m for _c, _f, m in loaded])
    atomic_write_text(out_path, json.dumps({"candidates": results}, indent=2) + "\n")


def _screen_isolated(
    *, op_name: str, ep: str, candidates: Sequence[tuple[str, Path]],
    anchors: Sequence[aspecs.AnchorCaseSpec], source_root: Path, cann_repo: Path,
    timeout_sec: float = 30.0,
) -> tuple[bool, dict[str, tuple[bool, str, dict[int, tuple[bool, str]]]]]:
    if not candidates:
        return True, {}
    eff_to = max(0.05, float(timeout_sec))
    env = dict(os.environ)
    pp = [e for e in sys.path if e and (Path(e) / "npu_compare.py").is_file()] + [
        str(source_root), str(source_root / "src"), str(source_root / "targets" / "ascend"),
        *[x for x in env.get("PYTHONPATH", "").split(os.pathsep) if x],
    ]
    env["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(pp))
    with tempfile.TemporaryDirectory(prefix="tf_poison_pill_") as td:
        req_p, out_p = Path(td) / "req.json", Path(td) / "out.json"
        req_p.write_text(json.dumps({
            "operator_name": op_name, "entrypoint": ep,
            "candidates": [{"candidate_id": c, "module_path": str(p)} for c, p in candidates],
            "source_root": str(source_root), "cannbench_repo": str(cann_repo),
            "anchors": [a.to_dict() for a in anchors],
        }), encoding="utf-8")
        ok, err, _rc, _so, _se = ag_npu.run_poison_pill_subprocess(
            [sys.executable, str(Path(__file__).resolve()), "--poison-pill",
             str(req_p), str(out_p)],
            cwd=source_root, env=env,
            exec_timeout_sec=max(eff_to, min(eff_to * max(1, len(candidates)), 120.0)),
        )
        if not ok or not out_p.is_file():
            msg = err or "poison_pill_timeout_or_crash"
            return False, {c: (False, msg, {}) for c, _ in candidates}
        try:
            rc = json.loads(out_p.read_text(encoding="utf-8"))["candidates"]
            return True, {
                cid: (bool(rc[cid].get("all_passed", False)), str(rc[cid].get("error", ""))[:300],
                      {int(k): (bool(r.get("passed", False)), str(r.get("error", "")))
                       for k, r in rc[cid]["cases"].items() if isinstance(r, Mapping)})
                for cid, _ in candidates
            }
        except Exception as exc:
            msg = f"poison_pill_timeout_or_crash: invalid worker json: {exc}"[:300]
            return False, {c: (False, msg, {}) for c, _ in candidates}


def _measure_passed(
    passed: Sequence[tuple[int, str, Callable[..., object]]],
    loaded: Sequence[tuple[str, Callable[..., object], object]], params: object,
    spec: aspecs.AnchorCaseSpec, *, has_ev: bool, interleaved: bool, warmup: int, repeat: int,
) -> dict[str, ag_npu.AnchorEvalMetrics]:
    out: dict[str, ag_npu.AnchorEvalMetrics] = {}
    inc_ok = bool(loaded and loaded[0][0] in _SKIP_PILL_IDS and passed[0][0] == 0)
    if len(passed) >= 2 and has_ev and interleaved:
        stats = ag_npu.measure_paired_interleaved_micro_burst(
            [(c, fn) for _i, c, fn in passed], params, warmup=warmup, repeat=repeat,
            t_hw_us=spec.t_hw_us, max_vram_bytes=512 * 1024 * 1024,
        )
        for pos, (_i, cid, _fn) in enumerate(passed):
            cus, bk, rcv, mr, rmad = stats[cid]
            out[cid] = ag_npu.build_anchor_metrics(
                spec, accuracy_passed=True, cand_us=cus, ring_burst_k=bk,
                burst_batch_cv=(rcv if pos == 0 else rmad) if inc_ok else rcv,
                paired_speedup_vs_incumbent=(1.0 if pos == 0 else mr) if inc_ok else 0.0,
                paired_rel_mad=(0.0 if pos == 0 else rmad) if (inc_ok or pos > 0) else 0.0,
            )
        return out
    for _i, cid, fn in passed:
        bk, bcv, ret, probe_us = 1, 0.0, 0, None
        if has_ev and not (0.0 < float(spec.t_hw_us) < 5.0):
            s = _time_events(lambda f=fn, p=params: f(**p), warmup=max(6, warmup), repeat=repeat)
            cus = float(statistics.median(s))
            bcv = float(statistics.pstdev(s)) / max(cus, 1e-9) if len(s) >= 2 else 0.0
            probe_us = cus if 0.0 < cus < 8.0 else None
        if has_ev and ((0.0 < float(spec.t_hw_us) < 5.0) or probe_us is not None):
            cus, bk, bcv, ret = ag_npu.measure_npu_ring_burst_stats(
                fn, params, warmup=warmup, repeat=repeat, t_hw_us=spec.t_hw_us,
                first_probe_us=probe_us, max_vram_bytes=512 * 1024 * 1024)
        elif not has_ev:
            try:
                cus = float(importlib.import_module("tilelang.profiler").do_bench(
                    lambda f=fn, p=params: f(**p), _n_warmup=warmup, _n_repeat=repeat,
                    backend="event", return_mode="median", device=torch.device("npu:0"),
                )) * 1000.0
            except Exception:
                cus = float(statistics.median(
                    _time_events(lambda f=fn, p=params: f(**p), warmup=warmup, repeat=repeat)))
        out[cid] = ag_npu.build_anchor_metrics(
            spec, accuracy_passed=True, cand_us=cus, ring_burst_k=bk,
            burst_batch_cv=bcv, spike_retries=ret)
    return out


def _evaluate_operator_batch(
    batch: Mapping[str, object], *, source_root: Path, cann_repo: Path, warmup: int, repeat: int,
    collect_msprof_detail: bool = False,
    msprof_out: dict[str, dict[str, dict[str, object]]] | None = None,
    poison_pill_isolation: bool | None = None, poison_pill_timeout_sec: float = 30.0,
) -> dict[str, ag_npu.CandidateSurrogateResult]:
    op_name = str(batch["operator_name"])
    ep = str(batch.get("entrypoint") or aspecs.get_operator_entrypoint(op_name))
    anchors = [aspecs.AnchorCaseSpec.from_dict(item) for item in batch["anchors"]]
    candidates, raw_msp = list(batch["candidates"]), batch.get("msprof_case_nums")
    msp_cases = ({int(c) for c in raw_msp} if raw_msp is not None
                 else ({int(a.case_num) for a in anchors} if collect_msprof_detail else set()))
    mw, mr = int(batch.get("msprof_warmup", 1)), int(batch.get("msprof_repeat", 3))
    cmod, harness, by_num = _load_harness(op_name, anchors, source_root, cann_repo)
    has_ev = hasattr(getattr(torch, "npu", None), "Event")
    non_inc = [c for c in candidates if str(c["candidate_id"]) not in _INCUMBENT_IDS]
    use_pill = (bool(poison_pill_isolation) and bool(non_inc) if poison_pill_isolation is not None
                else len(candidates) >= 2 and bool(non_inc) and has_ev)
    per_cand: dict[str, list[ag_npu.AnchorEvalMetrics]] = {str(c["candidate_id"]): []
                                                           for c in candidates}
    phase_a_ok: set[str] = set()
    if use_pill:
        with cmod.ReferenceCache(harness) as wc:
            for spec in anchors:
                _oracle_bundle(wc, harness, by_num[int(spec.case_num)])
        pairs = [(str(c["candidate_id"]), Path(str(c["module_path"]))) for c in non_inc]
        kw = {"op_name": op_name, "ep": ep, "anchors": anchors, "source_root": source_root,
              "cann_repo": cann_repo,
              "timeout_sec": float(batch.get("poison_pill_timeout_sec", poison_pill_timeout_sec))}
        b_ok, b_res = _screen_isolated(candidates=pairs, **kw)
        for cid, mpath in pairs:
            ok_iso, b_err, pcase = (b_res[cid] if b_ok else (
                _screen_isolated(candidates=[(cid, mpath)], **kw)[1] if len(pairs) >= 2 else b_res
            ).get(cid, (False, "poison_pill_timeout_or_crash", {})))
            if ok_iso:
                phase_a_ok.add(cid)
            else:
                fb = b_err or "poison_pill_timeout_or_crash"
                per_cand[cid] = [_fail_m(
                    s, ce[1] if (ce := pcase.get(int(s.case_num))) and not ce[0] and ce[1] else fb
                ) for s in anchors]
    loaded = [
        (cid, *_load_cand(Path(str(c["module_path"])), ep, f"{op_name}_{i}_{time.monotonic_ns()}"))
        for i, c in enumerate(candidates)
        if not (use_pill and (cid := str(c["candidate_id"])) not in _INCUMBENT_IDS
                and cid not in phase_a_ok)
    ]
    with cmod.ReferenceCache(harness) as cache:
        for spec in anchors:
            case = by_num[int(spec.case_num)]
            tensors, oracle, native_fn = _oracle_bundle(cache, harness, case)
            passed, by_cid = [], {}
            for oidx, (cid, fn, _m) in enumerate(loaded):
                if use_pill and (cid in phase_a_ok or cid in _SKIP_PILL_IDS):
                    passed.append((oidx, cid, fn))
                elif (ra := _run_case_acc(fn, harness, cmod, case, tensors, oracle, native_fn))[0]:
                    passed.append((oidx, cid, fn))
                else:
                    by_cid[cid] = _fail_m(spec, ra[1])
            if passed:
                def _time_pass(base_params: object) -> None:
                    by_cid.update(_measure_passed(
                        passed, loaded, base_params, spec, has_ev=has_ev,
                        interleaved=bool(batch.get("paired_interleaved", True)),
                        warmup=warmup, repeat=repeat))
                    if msprof_out is not None and int(spec.case_num) in msp_cases:
                        for _i, cid, fn in passed:
                            msprof_out.setdefault(cid, {})[str(int(spec.case_num))] = (
                                _measure_msprof(lambda f=fn, p=base_params: f(**p),
                                                warmup=mw, repeat=mr))
                try:
                    _with_npu_params(cmod, harness, case, tensors, _time_pass)
                except Exception as exc:
                    for _i, cid, _fn in passed:
                        by_cid.setdefault(cid, _fail_m(spec, f"{type(exc).__name__}: {exc}"))
            for cid, _fn, _m in loaded:
                per_cand[cid].append(by_cid[cid])
    _clear_caches([m for _c, _f, m in loaded])
    return {str(c["candidate_id"]): ag_npu.aggregate_surrogate_result(
        str(c["candidate_id"]), op_name, per_cand[str(c["candidate_id"])]
    ) for c in candidates}


def _collect_msprof_targets(
    targets: Sequence[Mapping[str, object]], *, source_root: Path, cann_repo: Path
) -> dict[str, dict[str, dict[str, dict[str, object]]]]:
    by_op: dict[str, list[Mapping[str, object]]] = {}
    for row in targets:
        if isinstance(row, Mapping):
            by_op.setdefault(str(row["operator_name"]), []).append(row)
    out: dict[str, dict[str, dict[str, dict[str, object]]]] = {}
    for op_name, rows in by_op.items():
        ep = aspecs.get_operator_entrypoint(op_name)
        specs = [aspecs.AnchorCaseSpec("", cn, "", 1.0, 1.0)
                 for cn in sorted({int(r["case_num"]) for r in rows})]
        cmod, harness, by_num = _load_harness(op_name, specs, source_root, cann_repo)
        loaded = {(cid := str(r["candidate_id"]), mp := str(Path(str(r["module_path"])).resolve())):
                  _load_cand(Path(mp), str(r.get("entrypoint") or ep),
                             f"msp_{op_name}_{idx}_{time.monotonic_ns()}")
                  for idx, r in enumerate(rows)}
        with cmod.ReferenceCache(harness) as cache:
            for r in rows:
                cid, cn = str(r["candidate_id"]), int(r["case_num"])
                mp = str(Path(str(r["module_path"])).resolve())
                t, _o, _n = _oracle_bundle(cache, harness, by_num[cn])
                try:
                    metrics = _with_npu_params(
                        cmod, harness, by_num[cn], t,
                        lambda p, f=loaded[(cid, mp)][0], w=int(r.get("warmup", 1)),
                        rep=int(r.get("repeat", 3)): _measure_msprof(
                            lambda f=f, p=p: f(**p), warmup=w, repeat=rep))
                except Exception as exc:
                    metrics = {"error": f"msprof_target_failed: {type(exc).__name__}: {exc}"[:300]}
                out.setdefault(op_name, {}).setdefault(cid, {})[str(cn)] = metrics
        _clear_caches([m for _f, m in loaded.values()])
    return out


class _FastRequestError(RuntimeError):
    """Fast request contract violation reported as request_error."""


@dataclass(frozen=True)
class _FastCand:
    op_name: str
    cid: str
    path: Path
    rel_path: str
    sha256: str


def _req(cond: bool, msg: str) -> None:
    if not cond:
        raise _FastRequestError(msg)


def _fast_int(value: object, field: str, *, default: int | None = None, min_val: int = 1) -> int:
    if value is None and default is not None:
        return default
    _req(not isinstance(value, bool) and isinstance(value, int) and value >= min_val,
         f"fast request field {field!r} must be >= {min_val}")
    return int(value)


def _fast_abs_dir(env_name: str, raw_val: str | None = None, *, create: bool = False) -> Path:
    if not (raw := (os.environ.get(env_name, "") if raw_val is None else raw_val).strip()):
        raise RuntimeError(f"fast request requires {env_name} from the AgentGates adapter")
    if not (path := Path(raw).expanduser()).is_absolute():
        raise RuntimeError(f"fast request environment {env_name} must be absolute")
    if create:
        path.mkdir(parents=True, exist_ok=True)
    elif not path.is_dir():
        raise RuntimeError(f"fast request environment {env_name} must be a directory: {path}")
    return path.resolve()


def _fast_path(source_root: Path, value: object, field: str) -> Path:
    _req(bool(raw := str(value or "").strip()), f"fast request field {field!r} is required")
    rel = Path(raw)
    _req(not rel.is_absolute() and ".." not in rel.parts and not (source_root / rel).is_symlink(),
         f"fast request field {field!r} must be a safe relative path")
    try:
        resolved = (source_root / rel).resolve()
        resolved.relative_to(source_root)
    except ValueError:
        raise _FastRequestError(f"fast request field {field!r} escapes the source root") from None
    return resolved


def _fast_cand(
    source_root: Path, op: str, row: Mapping[str, object], used: set[tuple[str, str]]
) -> _FastCand:
    cid = str(row.get("candidate_id", "")).strip()
    _req(bool(cid) and (key := (op, cid)) not in used, f"invalid or duplicate candidate {op}/{cid}")
    used.add(key)
    rel = str(row.get("module_path", "")).strip()
    mod_path = _fast_path(source_root, rel, "module_path")
    _req(mod_path.is_file(), f"fast request module {rel!r} is not a regular file")
    sha = str(row.get("sha256", "")).strip().lower()
    _req(bool(_FAST_SHA256.fullmatch(sha)) and _sha256_file(mod_path) == sha,
         f"fast request module {rel!r} failed sha256 verification")
    return _FastCand(op, cid, mod_path, rel, sha)


def _fast_bindings(candidates: Sequence[_FastCand]) -> dict[str, dict[str, dict[str, str]]]:
    bindings: dict[str, dict[str, dict[str, str]]] = {}
    for c in candidates:
        if (post := _sha256_file(c.path)) != c.sha256:
            raise RuntimeError(
                f"fast gate {ag_npu.FAST_STATUS_INFRASTRUCTURE_ERROR}: {c.op_name}/{c.cid} changed")
        bindings.setdefault(c.op_name, {})[c.cid] = {
            "module_path": c.rel_path, "sha256": c.sha256, "post_sha256": post}
    return bindings


def _exec_batches(
    batches: Sequence[Mapping[str, object]], *, source_root: Path, cann_repo: Path,
    warmup: int, repeat: int, collect_msprof: bool = False,
) -> tuple[dict[str, dict[str, dict[str, object]]],
           dict[str, dict[str, ag_npu.CandidateSurrogateResult]],
           dict[str, dict[str, dict[str, dict[str, object]]]]]:
    res_by_op, parsed_by_op, msprof_tel = {}, {}, {}
    for batch in batches:
        op_msp: dict[str, dict[str, dict[str, object]]] = {}
        op_res = _evaluate_operator_batch(
            batch, source_root=source_root, cann_repo=cann_repo, warmup=warmup, repeat=repeat,
            collect_msprof_detail=collect_msprof, msprof_out=op_msp)
        op = str(batch["operator_name"])
        parsed_by_op[op], res_by_op[op] = op_res, {cid: r.to_dict() for cid, r in op_res.items()}
        if op_msp:
            msprof_tel[op] = op_msp
    return res_by_op, parsed_by_op, msprof_tel


def _handle_fast_request(request_path_raw: str, source_root_raw: str, shim_dir: Path) -> None:
    if not (req_path := Path(request_path_raw)).is_file():
        raise RuntimeError(f"fast request {req_path} does not exist")
    req_bytes = req_path.read_bytes()
    req_sha, req = hashlib.sha256(req_bytes).hexdigest(), json.loads(req_bytes.decode("utf-8"))
    if not isinstance(req, dict):
        raise RuntimeError(f"fast request {req_path} must be a JSON object")
    sroot = _fast_abs_dir(ag_npu.FAST_SOURCE_ROOT_ENV, source_root_raw)
    resp_p = _fast_path(sroot, req.get("response_path"), "response_path")
    req_id, mode = str(req.get("request_id", "")).strip(), str(req.get("mode", "")).strip()

    def _resp(
        status: str, *, error: str = "", ops: Mapping[str, object] | None = None,
        msp: Mapping[str, object] | None = None, cands: Mapping[str, object] | None = None,
        cov: tuple[int, int, int, int] = (0, 0, 0, 0), probe: Mapping[str, object] | None = None,
        null_us: float | None = None, stdout: bool = True,
    ) -> None:
        payload: dict[str, object] = {
            "schema": ag_npu.FAST_RESPONSE_SCHEMA, "request_id": req_id,
            "request_sha256": req_sha, "mode": mode, "status": status, "error": error[-1000:],
            "operators": dict(ops or {}), "msprof_telemetry": dict(msp or {}),
            "candidates": dict(cands or {}), "coverage": dict(zip(
                ("expected_candidates", "evaluated_candidates",
                 "expected_msprof_targets", "observed_msprof_targets"), cov, strict=True)),
        }
        if probe is not None:
            payload["micro_probe"] = dict(probe)
        if mode == "batch" and null_us is not None and math.isfinite(null_us) and null_us > 0.0:
            payload["null_launch_us"] = float(null_us)
        atomic_write_text(resp_p, json.dumps(payload, indent=2) + "\n")
        if stdout:
            print(json.dumps(payload, sort_keys=True), flush=True)

    try:
        _req(req.get("schema") == ag_npu.FAST_REQUEST_SCHEMA and bool(req_id)
             and mode in ("batch", "profile", "micro_probe"),
             f"invalid fast request: {req.get('schema')!r}/{mode!r}")
        if mode == "micro_probe":
            if tl_raw := os.environ.get("TILELANG_CACHE_DIR", "").strip():
                _setup_env(None, None, tl_raw)
            _req(isinstance(rp := req.get("probe"), Mapping), "fast micro_probe requires probe")
            rs = [str(rp.get(k, "")).strip()
                  for k in ("probe_id", "probe_kind", "factory_name", "tensor_dtype")]
            _req(all(rs), "fast micro_probe missing probe metadata fields")
            lcb = _fast_int(rp.get("line_count_baseline"), "probe.line_count_baseline", min_val=0)
            lco = _fast_int(rp.get("line_count_optimized"), "probe.line_count_optimized", min_val=0)
            sh, sp = rp.get("tensor_shape"), rp.get("min_speedup")
            _req(isinstance(sh, list) and 1 <= len(sh) <= 8
                 and all(isinstance(d, int) and not isinstance(d, bool) and d > 0 for d in sh),
                 "probe.tensor_shape must list 1 to 8 positive ints")
            _req(isinstance(sp, (int, float)) and not isinstance(sp, bool)
                 and math.isfinite(sp) and sp > 0, "probe.min_speedup must be positive")
            _req(isinstance(rc := req.get("candidates"), list) and len(rc) == 2,
                 "micro_probe needs 2")
            used_m: set[tuple[str, str]] = set()
            cm = [_fast_cand(sroot, "micro_probe", r, used_m) for r in rc if isinstance(r, Mapping)]
            by_id = {c.cid: c for c in cm}
            _req(len(cm) == 2 and set(by_id) == {"baseline", "optimized"},
                 "need baseline/optimized")
            spec = MicroProbeSpec(
                rs[0], rs[1], rs[2], by_id["baseline"].path.read_text(encoding="utf-8"),
                by_id["optimized"].path.read_text(encoding="utf-8"), lcb, lco, tuple(sh), rs[3])
            ok, metrics = verify_micro_probe_hardware(
                spec, probe_port=ag_npu.LocalHardwareProbePort())
            _resp(ag_npu.FAST_STATUS_OK if ok else ag_npu.FAST_STATUS_CANDIDATE_FAILURE,
                  cands=_fast_bindings(cm), cov=(2, 2, 0, 0),
                  probe={"probe_id": rs[0],
                         "metrics": {str(k): float(v) for k, v in metrics.items()}})
            return
        crepo = _fast_abs_dir("TILEFOUNDRY_NPU_CANNBENCH_REPO")
        tlc = _fast_abs_dir("TILELANG_CACHE_DIR", create=True)
        os.environ["CANNBENCH_CACHE_DIR"] = str(_fast_abs_dir("CANNBENCH_CACHE_DIR", create=True))
        _setup_env(crepo, shim_dir, str(tlc))
        if mode == "batch":
            _req(isinstance(raw_b := req.get("batches"), list) and bool(raw_b)
                 and req.get("msprof_targets") in (None, []),
                 "fast batch requires non-empty batches and no msprof_targets")
            w = _fast_int(req.get("warmup"), "warmup", default=3)
            r = _fast_int(req.get("repeat"), "repeat", default=5)
            used, ops, cands, batches = set(), set(), [], []
            for row in raw_b:
                _req(isinstance(row, Mapping)
                     and bool(op := str(row.get("operator_name", "")).strip())
                     and op not in ops, "fast batch entry requires unique operator_name")
                ops.add(op)
                _req(isinstance(rc := row.get("candidates"), list) and bool(rc)
                     and isinstance(ra := row.get("anchors"), list) and bool(ra),
                     f"fast batch operator {op!r} requires candidates and anchors")
                norm_c = []
                for crow in rc:
                    _req(isinstance(crow, Mapping), "fast batch candidates must be objects")
                    cands.append(cand := _fast_cand(sroot, op, crow, used))
                    norm_c.append({"candidate_id": cand.cid, "module_path": str(cand.path)})
                batches.append({
                    "operator_name": op, "candidates": norm_c, "anchors": ra,
                    "entrypoint": str(row.get("entrypoint") or aspecs.get_operator_entrypoint(op))})
            null_us = _measure_null_launch_us()
            res_by_op, parsed_by_op, _ = _exec_batches(
                batches, source_root=sroot, cann_repo=crepo, warmup=w, repeat=r)
            bad = [1 for m in parsed_by_op.values() for res in m.values() if not res.all_passed]
            _resp(ag_npu.FAST_STATUS_OK if not bad else ag_npu.FAST_STATUS_CANDIDATE_FAILURE,
                  ops=res_by_op, cands=_fast_bindings(cands), null_us=null_us,
                  cov=(len(cands), sum(len(c) for c in res_by_op.values()), 0, 0))
        else:
            _req(isinstance(raw_t := req.get("msprof_targets"), list) and bool(raw_t)
                 and req.get("batches") in (None, []),
                 "fast profile requires non-empty msprof_targets and no batches")
            dw = _fast_int(req.get("warmup"), "warmup", default=1)
            dr = _fast_int(req.get("repeat"), "repeat", default=3)
            used_p, by_key, seen_t, cands_p, norm_rows = set(), {}, set(), [], []
            for row in raw_t:
                _req(isinstance(row, Mapping)
                     and bool(op := str(row.get("operator_name", "")).strip())
                     and bool(decl := str(row.get("candidate_id", "")).strip()),
                     "fast profile target requires operator_name and candidate_id")
                if (cand := by_key.get((op, decl))) is None:
                    by_key[(op, decl)] = cand = _fast_cand(sroot, op, row, used_p)
                    cands_p.append(cand)
                else:
                    _req(str(row.get("module_path", "")).strip() == cand.rel_path
                         and str(row.get("sha256", "")).strip().lower() == cand.sha256,
                         f"fast profile candidate {op}/{cand.cid} changes module_path or sha256")
                cnum = _fast_int(row.get("case_num"), "case_num")
                _req((tkey := (op, cand.cid, cnum)) not in seen_t,
                     f"duplicate fast profile target {op}/{cand.cid}#{cnum}")
                seen_t.add(tkey)
                norm_rows.append({
                    "operator_name": op, "candidate_id": cand.cid, "module_path": str(cand.path),
                    "case_num": cnum, "warmup": _fast_int(row.get("warmup"), "warmup", default=dw),
                    "repeat": _fast_int(row.get("repeat"), "repeat", default=dr)})
            tel = _collect_msprof_targets(norm_rows, source_root=sroot, cann_repo=crepo)
            _resp(ag_npu.FAST_STATUS_OK, msp=tel, cands=_fast_bindings(cands_p), cov=(
                len(cands_p), len(cands_p), len(raw_t),
                sum(len(c) for m in tel.values() for c in m.values())))
    except _FastRequestError as exc:
        _resp(ag_npu.FAST_STATUS_REQUEST_ERROR, error=str(exc), stdout=False)
    except Exception as exc:
        _resp(ag_npu.FAST_STATUS_INFRASTRUCTURE_ERROR,
              error=f"{type(exc).__name__}: {exc}", stdout=False)
        raise


def run_surrogate_anchor_batch_npu() -> None:
    """Evaluate configured or baseline 10-anchor surrogate batch inside AgentGates lease."""
    _ensure_npu(require=True)
    torch.npu.set_device(0)
    fast_env = os.environ.get(ag_npu.FAST_REQUEST_ENV, "").strip()
    req_env = os.environ.get("TF_SURROGATE_BATCH_REQUEST", "").strip()
    if fast_env and req_env:
        raise RuntimeError(f"{ag_npu.FAST_REQUEST_ENV} and TF_SURROGATE_BATCH_REQUEST conflict")
    with tempfile.TemporaryDirectory(prefix="tf_surrogate_shim_") as shim_dir:
        if fast_env:
            if not Path(fast_env).is_file():
                raise RuntimeError(f"{ag_npu.FAST_REQUEST_ENV} does not exist: {fast_env}")
            _handle_fast_request(fast_env, os.environ.get(ag_npu.FAST_SOURCE_ROOT_ENV, "").strip(),
                                 Path(shim_dir))
            return
        if req_env and Path(req_env).is_file():
            req = json.loads(Path(req_env).read_text(encoding="utf-8"))
            sroot, crepo = Path(req["source_root"]).resolve(), Path(req["cannbench_repo"]).resolve()
        else:
            crepo = _resolve_cannbench(sroot := aspecs.default_source_root())
            req = {"warmup": 3, "repeat": 5, "require_all_passed": True, "batches": [{
                "operator_name": op, "entrypoint": aspecs.get_operator_entrypoint(op),
                "candidates": [{"candidate_id": f"{op}_baseline", "module_path": str(
                    (sroot / aspecs.get_operator_module_relpath(op)).resolve())}],
                "anchors": [a.to_dict() for a in aspecs.get_default_anchors(op)],
            } for op in ("attention_residual", "mla")]}
        uid = os.getuid() if hasattr(os, "getuid") else 0
        cdir = Path(tempfile.gettempdir()) / f"tf_surrogate_ref_cache_{uid}"
        if not os.environ.get("CANNBENCH_CACHE_DIR"):
            cdir.mkdir(parents=True, exist_ok=True)
            os.environ["CANNBENCH_CACHE_DIR"] = str(cdir)
        _setup_env(crepo, Path(shim_dir), str(req.get("tilelang_cache_dir", "")).strip()
                   or os.environ.get("TF_SURROGATE_TL_CACHE_DIR", "").strip()
                   or str(Path(tempfile.gettempdir()) / f"tf_surrogate_tl_cache_{uid}"))
        collect_msp = bool(req.get("collect_msprof_detail", False))
        res_by_op, parsed_by_op, msprof_tel = _exec_batches(
            req.get("batches", ()), source_root=sroot, cann_repo=crepo,
            warmup=int(req.get("warmup", 3)), repeat=int(req.get("repeat", 5)),
            collect_msprof=collect_msp)
        for op, cmap in _collect_msprof_targets(
            list(req.get("msprof_targets") or ()), source_root=sroot, cann_repo=crepo
        ).items():
            for cid, cases_map in cmap.items():
                msprof_tel.setdefault(op, {}).setdefault(cid, {}).update(cases_map)
        summary: dict[str, object] = {"operators": res_by_op}
        if msprof_tel or collect_msp or req.get("msprof_targets"):
            summary["msprof_telemetry"] = msprof_tel
        if resp_raw := str(req.get("response_path", "")).strip():
            atomic_write_text(Path(resp_raw), json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, sort_keys=True), flush=True)
        if bool(req.get("require_all_passed", False)):
            for op, op_res in parsed_by_op.items():
                for cid, res in op_res.items():
                    assert res.all_passed, f"{op}/{cid} failed"
                    assert all(a.cand_us > 0.0 for a in res.anchors), f"{op}/{cid} invalid time"


if __name__ == "__main__" and len(sys.argv) == 4 and sys.argv[1] == "--poison-pill":
    _poison_pill_worker_main(Path(sys.argv[2]), Path(sys.argv[3]))
