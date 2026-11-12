"""Independent Runtime Twin wrapper for BF16 RMSNorm on the real Ascend950PR.

Exposes a config-keyed run(x, w, eps) ABI matching the original run(x, w, eps)
that returns (BATCH, D) bf16. Numeric checks compare against an independent CPU
FP64 golden; timing uses NPU events around the kernel call only (compile and
host transfer excluded). The TF side has its own evaluator; this wrapper never
calls it as the answer.
"""
from __future__ import annotations

import copy, json, math
from dataclasses import asdict

import numpy as np

try:
    from .kernels.twin_rmsnorm import TwinConfig, compile_twin, get_kernel_source
except ImportError:  # top-level import path (examples/rmsnorm_950pr on sys.path)
    from kernels.twin_rmsnorm import TwinConfig, compile_twin, get_kernel_source


def _to_cpu(t):
    # tilelang output tensors carry storage without NPUStorageImpl metadata on
    # this torch_npu build; clone first to get a normal tensor.
    return t.clone().to("cpu")


class RuntimeModel:
    def __init__(self, cfg: TwinConfig):
        self.cfg = cfg
        self._kernel = None

    def ensure(self):
        if self._kernel is None:
            self._kernel = compile_twin(self.cfg)
        return self._kernel

    def run(self, x, w, eps=None):
        import torch
        k = self.ensure()
        eps = self.cfg.eps if eps is None else eps
        y = k(x.reshape(-1), w, float(eps))
        return y.reshape(self.cfg.batch, self.cfg.d)

    def kernel_source(self):
        return get_kernel_source(self.cfg)


def make_input(torch, shape, pattern, seed):
    g = torch.Generator(device="cpu").manual_seed(seed)
    rows, width = shape
    if pattern == "zero":
        x = torch.zeros(shape, dtype=torch.float32); w = torch.ones(width, dtype=torch.float32)
    elif pattern == "ones":
        x = torch.ones(shape, dtype=torch.float32); w = torch.linspace(0.25, 1.75, width)
    elif pattern == "random":
        x = torch.randn(shape, generator=g); w = torch.randn(width, generator=g)
    elif pattern == "alternating":
        idx = torch.arange(width)
        scale = torch.tensor([0.01, 1.0, 10.0, 100.0])[idx % 4]
        row = scale * torch.where(idx % 2 == 0, 1.0, -1.0)
        signs = torch.where(torch.arange(rows) % 2 == 0, 1.0, -1.0)
        x = signs[:, None] * row[None, :]; w = torch.linspace(-1.0, 1.0, width)
    else:
        raise ValueError(pattern)
    return x.to(torch.bfloat16).contiguous(), w.to(torch.bfloat16).contiguous()


def golden(torch, x, w, eps):
    xd, wd = x.to(torch.float64), w.to(torch.float64)
    exact = xd * torch.rsqrt((xd * xd).mean(dim=-1, keepdim=True) + eps) * wd
    return exact.to(torch.bfloat16), exact


def compare(torch, got, ref, exact, atol, rtol):
    """Per-element accuracy against the independent CPU-FP64 reference.

    The pass rule is |got-ref| <= atol + rtol*|ref| for EVERY element (mismatch_count
    == 0). We record max_abs and max_rel separately and NEVER state the pass in terms
    of "max_abs <= atol": a case can pass on the rtol wing with max_abs > atol (seen
    historically: max_abs 0.03125 > atol 0.02). max_abs_le_atol is its own boolean and
    is not used to decide pass/fail; the mismatch count is.
    """
    g = got.float(); r = ref.float()
    finite = bool(torch.isfinite(g).all().item())
    if not finite:
        return {"passed": False, "finite": False, "max_abs": None, "max_rel": None,
                "mismatch_count": int((~torch.isfinite(g)).sum()),
                "max_abs_le_atol": False,
                "pass_rule": "non-finite output cannot pass"}
    diff = (g - r).abs()
    tol = atol + rtol * r.abs()
    mism = int((diff > tol).sum())
    max_abs = float(diff.max())
    max_rel = float((diff / r.abs().clamp_min(1e-12)).max())
    return {"passed": mism == 0, "finite": True,
            "max_abs": max_abs, "max_rel": max_rel, "mismatch_count": mism,
            "max_abs_le_atol": bool(max_abs <= atol),
            "atol": atol, "rtol": rtol,
            "pass_rule": "mismatch_count==0 under |got-ref| <= atol + rtol*|ref|",
            "max_abs_to_unrounded_fp64": float((g.double() - exact).abs().max())}


def _input_schedule(protocol):
    """Turn the frozen protocol's patterns+seeds into (pattern, seed) pairs."""
    seeds = list(protocol.get("seeds", [0, 1, 2]))
    out = []
    for pat in protocol.get("patterns", ["zero", "ones", "random", "alternating"]):
        if pat == "random":
            for s in seeds:
                out.append(("random", s))
        else:
            out.append((pat, 0))
    return out


DETERMINISTIC_PATTERNS = {"zero", "ones", "alternating"}


def check_case(model: RuntimeModel, torch, device, protocol, save_dir=None):
    """Accuracy + stability + buffer-reuse, driven by the FROZEN protocol.

    Persists the DEVICE output tensor (cpu clone) per case/pattern to save_dir as
    <key>.device.npy, alongside <key>.golden.npy, <key>.x.npy, <key>.w.npy, so a
    direct elementwise diff against an independent reference is possible rather than
    relying only on aggregate stats.
    """
    import os
    import numpy as np
    cfg = model.cfg
    atol = float(protocol["atol"]); rtol = float(protocol["rtol"])
    eps = float(protocol.get("eps", cfg.eps))
    reps = int(protocol.get("repetitions_per_input", 4))
    stability = int(protocol.get("stability_runs_per_case", 64))
    schedule = _input_schedule(protocol)
    shape = (cfg.batch, cfg.d)
    model.ensure()
    dev = "npu:%d" % device
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    report = {"case_id": cfg.case_id, "shape": list(shape), "rows": cfg.rows,
              "loop_mode": cfg.loop_mode, "num_stages": cfg.num_stages,
              "x_versions": cfg.x_versions, "y_versions": cfg.y_versions,
              "atol": atol, "rtol": rtol,
              "protocol_id": protocol.get("protocol_id"),
              "pass_rule": "per element |got-ref| <= atol + rtol*|ref| (mismatch_count must be 0)",
              "tests": [], "patterns": []}
    saved = []
    last = None
    with torch.no_grad():
        for pattern, seed in schedule:
            xc, wc = make_input(torch, shape, pattern, seed)
            refb, exact = golden(torch, xc, wc, cfg.eps)
            x = xc.to(dev); w = wc.to(dev)
            torch.npu.synchronize()
            prev = None
            rep_stats = []
            ycpu = None
            for rep in range(reps):
                y = model.run(x, w, cfg.eps); torch.npu.synchronize()
                assert tuple(y.shape) == shape and y.dtype == torch.bfloat16, (y.shape, y.dtype)
                ycpu = _to_cpu(y)
                st = compare(torch, ycpu, refb, exact, atol, rtol)
                st.update(pattern=pattern, seed=seed, repetition=rep,
                          repeat_bitwise_equal=(None if prev is None else bool(torch.equal(prev, ycpu))))
                prev = ycpu
                report["tests"].append(st)
                rep_stats.append(st)
            key = pattern if pattern != "random" else "random_s%d" % seed
            if save_dir is not None:
                np.save(os.path.join(save_dir, key + ".device.npy"), ycpu.to(torch.float32).numpy())
                np.save(os.path.join(save_dir, key + ".golden.npy"), refb.to(torch.float32).numpy())
                np.save(os.path.join(save_dir, key + ".x.npy"), xc.to(torch.float32).numpy())
                np.save(os.path.join(save_dir, key + ".w.npy"), wc.to(torch.float32).numpy())
                saved.append({"key": key, "pattern": pattern, "seed": seed,
                              "files": [key + ".device.npy", key + ".golden.npy",
                                        key + ".x.npy", key + ".w.npy"]})
            worst_abs = max(s["max_abs"] for s in rep_stats if s["max_abs"] is not None)
            worst_rel = max(s["max_rel"] for s in rep_stats if s["max_rel"] is not None)
            mism = int(sum(s["mismatch_count"] for s in rep_stats))
            passed = all(s["passed"] for s in rep_stats)
            det_required = pattern in DETERMINISTIC_PATTERNS
            det_exact = (worst_abs == 0.0) if det_required else None
            report["patterns"].append({
                "pattern": pattern, "seed": seed, "key": key, "repetitions": reps,
                "max_abs": worst_abs, "max_rel": worst_rel, "total_mismatch": mism,
                "passed": passed, "max_abs_le_atol": bool(worst_abs <= atol),
                "deterministic_required": det_required, "deterministic_exact": det_exact,
                "pass_basis": ("max_abs == 0 (deterministic, must be exact)" if det_required
                               else ("within atol alone" if worst_abs <= atol
                                     else "PASS on rtol wing only (max_abs > atol)")),
            })
            last = (x, w)
    det_ok = all(p["deterministic_exact"] for p in report["patterns"] if p["deterministic_required"])
    report["deterministic_patterns_exact"] = bool(det_ok)
    report["numerical_passed"] = all(t["passed"] for t in report["tests"]) and det_ok
    tests = report["tests"]
    report["accuracy_summary"] = {
        "atol": atol, "rtol": rtol,
        "pass_rule": "per element |got-ref| <= atol + rtol*|ref|; every test 0 mismatches; deterministic patterns exact",
        "max_abs": max(t["max_abs"] for t in tests if t["max_abs"] is not None),
        "max_rel": max(t["max_rel"] for t in tests if t["max_rel"] is not None),
        "total_mismatch": int(sum(t["mismatch_count"] for t in tests)),
        "max_abs_le_atol_alone": bool(all(t["max_abs_le_atol"] for t in tests)),
        "deterministic_patterns_exact": bool(det_ok),
        "per_pattern": report["patterns"],
        "note": ("max_abs_le_atol_alone is reported for transparency and is NOT the pass "
                 "criterion; a case may pass on the rtol wing with max_abs > atol."),
    }
    x, w = last
    outs = [_to_cpu(model.run(x, w, cfg.eps)) for _ in range(stability)]
    report["stability"] = {"runs": stability,
                           "all_bitwise_equal_first": all(bool(torch.equal(outs[0], o)) for o in outs[1:]),
                           "fixed_input": True}
    r1 = model.run(x, w, cfg.eps); p1 = int(r1.data_ptr()); y1 = _to_cpu(r1)
    r2 = model.run(x, w, cfg.eps); p2 = int(r2.data_ptr()); y2 = _to_cpu(r2)
    report["output_buffer_reuse"] = {
        "device_ptr_first": p1, "device_ptr_second": p2,
        "device_storage_distinct": p1 != p2,
        "values_equal": bool(torch.equal(y1, y2)),
        "method": "device data_ptr captured BEFORE cpu clone; values compared on cpu clones",
        "note": ("host clone pointers are always distinct and prove nothing; reuse is "
                 "judged on device addresses plus value equality"),
    }
    if save_dir is not None:
        with open(os.path.join(save_dir, "tensors.json"), "w") as fh:
            json.dump({"case_id": cfg.case_id, "shape": list(shape),
                       "tensors": saved,
                       "convention": ("<key>.device.npy = device output as float32 of bf16; "
                                      "<key>.golden.npy = runner fp64->bf16 reference; "
                                      "<key>.x/.w = bf16 inputs as float32")}, fh, indent=2)
    return report


def paired_benchmark(model: RuntimeModel, baseline_run, torch, device, windows=20, iters=10, warmup=20):
    """Alternating baseline/candidate paired windows; returns raw ms lists."""
    cfg = model.cfg
    shape = (cfg.batch, cfg.d)
    torch.manual_seed(0)
    xc, wc = make_input(torch, shape, "random", 0)
    dev = "npu:%d" % device
    x = xc.to(dev); w = wc.to(dev)
    model.ensure()
    with torch.no_grad():
        for _ in range(warmup):
            baseline_run(x, w, cfg.eps); model.run(x, w, cfg.eps)
        torch.npu.synchronize()

        def win(fn):
            s = torch.npu.Event(enable_timing=True); e = torch.npu.Event(enable_timing=True)
            s.record()
            for _ in range(iters):
                fn(x, w, cfg.eps)
            e.record(); e.synchronize()
            return s.elapsed_time(e) / iters

        base_w, cand_w = [], []
        for i in range(windows):
            if i % 2 == 0:
                base_w.append(win(baseline_run)); cand_w.append(win(model.run))
            else:
                cand_w.append(win(model.run)); base_w.append(win(baseline_run))
    import statistics
    return {"case_id": cfg.case_id, "windows": windows, "iters_per_window": iters,
            "baseline_ms": base_w, "candidate_ms": cand_w,
            "baseline_median": statistics.median(base_w), "candidate_median": statistics.median(cand_w),
            "baseline_min": min(base_w), "candidate_min": min(cand_w),
            "speedup_median": statistics.median(base_w) / statistics.median(cand_w),
            "method": "NPU events, output allocation included, alternating paired windows"}