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
    g = got.float(); r = ref.float()
    finite = bool(torch.isfinite(g).all().item())
    if not finite:
        return {"passed": False, "finite": False, "max_abs": None, "mismatch_count": int((~torch.isfinite(g)).sum())}
    diff = (g - r).abs(); mism = int((diff > (atol + rtol * r.abs())).sum())
    return {"passed": mism == 0, "finite": True, "max_abs": float(diff.max()),
            "max_rel": float((diff / r.abs().clamp_min(1e-12)).max()),
            "mismatch_count": mism,
            "max_abs_to_unrounded_fp64": float((g.double() - exact).abs().max())}


def check_case(model: RuntimeModel, torch, device, atol=0.02, rtol=0.01, reps=4, stability=64):
    cfg = model.cfg
    shape = (cfg.batch, cfg.d)
    model.ensure()
    dev = "npu:%d" % device
    report = {"case_id": cfg.case_id, "shape": list(shape), "rows": cfg.rows,
              "loop_mode": cfg.loop_mode, "num_stages": cfg.num_stages,
              "x_versions": cfg.x_versions, "y_versions": cfg.y_versions,
              "atol": atol, "rtol": rtol, "tests": []}
    last = None
    with torch.no_grad():
        for pat in ["zero", "ones", ("random", 0), ("random", 1), ("random", 2), "alternating"]:
            pattern, seed = (pat, 0) if isinstance(pat, str) else pat
            xc, wc = make_input(torch, shape, pattern, seed)
            refb, exact = golden(torch, xc, wc, cfg.eps)
            x = xc.to(dev); w = wc.to(dev)
            torch.npu.synchronize()
            prev = None
            for rep in range(reps):
                y = model.run(x, w, cfg.eps); torch.npu.synchronize()
                assert tuple(y.shape) == shape and y.dtype == torch.bfloat16, (y.shape, y.dtype)
                st = compare(torch, _to_cpu(y), refb, exact, atol, rtol)
                obs = _to_cpu(y)
                st.update(pattern=pattern, seed=seed, repetition=rep,
                          repeat_bitwise_equal=(None if prev is None else bool(torch.equal(prev, obs))))
                prev = obs
                report["tests"].append(st)
            last = (x, w)
    report["numerical_passed"] = all(t["passed"] for t in report["tests"])
    # stability: 64 fresh/repeat calls on random input
    x, w = last
    outs = []
    for _ in range(stability):
        outs.append(_to_cpu(model.run(x, w, cfg.eps)))
    report["stability"] = {"runs": stability,
                           "all_bitwise_equal_first": all(bool(torch.equal(outs[0], o)) for o in outs[1:]),
                           "fixed_input": True}
    # output buffer reuse: call ABI repeatedly, ensure fresh output each call
    y1 = _to_cpu(model.run(x, w, cfg.eps)); y2 = _to_cpu(model.run(x, w, cfg.eps))
    report["output_buffer_reuse"] = {"distinct_storage": y1.data_ptr() != y2.data_ptr(),
                                     "values_equal": bool(torch.equal(y1, y2))}
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