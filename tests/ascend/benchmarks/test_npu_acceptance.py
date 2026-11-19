"""Ascend 950PR NPU numerical acceptance tests for adapted operators."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch

BENCHMARKS_DIR = Path(__file__).resolve().parent
AR_DIR = BENCHMARKS_DIR / "attention_residual"
MLA_DIR = BENCHMARKS_DIR / "mla"
SUPP_MLA_PATH = BENCHMARKS_DIR / "cases" / "supplemental_mla.json"

for _path in (BENCHMARKS_DIR, AR_DIR, MLA_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import cannbench_contract  # noqa: E402

_L3_TARGET_OP = os.environ.get("TF_L3_TARGET_OPERATOR", "").strip()
_L3_TARGET_MODULE_PATH = os.environ.get("TF_L3_TARGET_MODULE_PATH", "").strip()


def _normalize_target_op_short(target_op: str) -> str:
    raw = str(target_op or "").strip()
    return raw.rsplit("/", 1)[-1].strip().lower() if raw else ""


_L3_TARGET_OP_SHORT = _normalize_target_op_short(_L3_TARGET_OP)


def _should_run_acceptance_test(test_kind: str, target_op: str) -> bool:
    """Return True when acceptance suite *test_kind* applies to *target_op*."""
    short = _normalize_target_op_short(target_op)
    if not short:
        return test_kind in ("attention_residual", "mla", "dynamic_cache")
    if short == "attention_residual":
        return test_kind in ("attention_residual", "dynamic_cache")
    if short == "mla":
        return test_kind in ("mla", "dynamic_cache")
    return test_kind == "generic_cannbench"


def _load_override_module(module_path_str: str, tag: str):
    mod_path = Path(module_path_str).resolve()
    spec = importlib.util.spec_from_file_location(f"_tf_l3_override_{tag}", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load override candidate module: {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _load_scoped_operator_module(op_short_name: str):
    if _L3_TARGET_MODULE_PATH and _L3_TARGET_OP_SHORT == op_short_name:
        return _load_override_module(_L3_TARGET_MODULE_PATH, op_short_name)
    return importlib.import_module(f"examples.ascend950pr_ops.{op_short_name}.kernels")


def _require_npu_env() -> None:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Ascend NPU device is unavailable on this host.")
    if importlib.util.find_spec("npu_compare") is None:
        pytest.skip("npu_compare module is unavailable on PYTHONPATH.")


def _call_single_kernel(kernel_module, fn, **kwargs):
    orig = kernel_module._get_kernel
    calls = 0

    def wrapped(*args, **inner_kwargs):
        nonlocal calls
        calls += 1
        return orig(*args, **inner_kwargs)

    kernel_module._get_kernel = wrapped
    try:
        result = fn(**kwargs)
    finally:
        kernel_module._get_kernel = orig
    assert calls == 1, f"expected 1 kernel invocation, got {calls}"
    return result


def test_attention_residual_cannbench_cases() -> None:
    """Verify all 20 CANNBench AttentionResidual cases on NPU."""
    if not _should_run_acceptance_test("attention_residual", _L3_TARGET_OP):
        pytest.skip(f"Scoped to {_L3_TARGET_OP} L3 target.")
    _require_npu_env()
    import test_attention_residual  # noqa: PLC0415

    ar_kernels = _load_scoped_operator_module("attention_residual")
    test_attention_residual.main(ar_kernels.attention_residual)


def test_attention_residual_baseline_vs_tuned_parity() -> None:
    """Verify AttentionResidual baseline schedule numerical parity on NPU."""
    if not _should_run_acceptance_test("attention_residual", _L3_TARGET_OP):
        pytest.skip(f"Scoped to {_L3_TARGET_OP} L3 target.")
    _require_npu_env()
    ar_kernels = _load_scoped_operator_module("attention_residual")
    attention_residual = ar_kernels.attention_residual
    harness = cannbench_contract.Harness("level4/attention_residual", AR_DIR)
    target_cases = {1, 16, 20}
    checked = 0
    for case in harness.cases:
        if case.case_num not in target_cases:
            continue
        checked += 1
        tensors = harness.prepare(case)
        oracle = harness.reference_output(case, tensors)
        npu_tensors = cannbench_contract.map_tensors(tensors, lambda x: x.npu())
        params = harness.params(case, npu_tensors)
        actual = _call_single_kernel(
            ar_kernels,
            attention_residual,
            **params,
            schedule_mode="baseline",
        )
        passed, details = harness.compare(
            case,
            actual,
            oracle,
            tensors,
            lambda case=case, tensors=tensors: harness.reference_output(
                case, tensors, native=True
            ),
        )
        assert passed, f"{case.get_case_id_str()}: {details}"
    assert checked == len(target_cases)


def test_mla_cannbench_cases() -> None:
    """Verify all 20 CANNBench MLA cases on NPU."""
    if not _should_run_acceptance_test("mla", _L3_TARGET_OP):
        pytest.skip(f"Scoped to {_L3_TARGET_OP} L3 target.")
    _require_npu_env()
    import test_mla  # noqa: PLC0415

    mla_kernels = _load_scoped_operator_module("mla")
    test_mla.main(mla_kernels.mla)


def test_mla_supplemental_cases() -> None:
    """Verify supplemental GQA, tail, layout, and scale MLA cases on NPU."""
    if not _should_run_acceptance_test("mla", _L3_TARGET_OP):
        pytest.skip(f"Scoped to {_L3_TARGET_OP} L3 target.")
    _require_npu_env()
    mla_kernels = _load_scoped_operator_module("mla")
    mla = mla_kernels.mla
    payload = json.loads(SUPP_MLA_PATH.read_text(encoding="utf-8"))
    harness = cannbench_contract.Harness("level4/mla", MLA_DIR)
    for raw in payload["cases"]:
        case = cannbench_contract.Case(
            case_num=raw["case_id"],
            label=raw["key"],
            input_shapes=raw["input_shapes"],
            dtypes=raw["dtypes"],
            attrs=raw["attrs"],
            value_ranges=raw["value_range"],
        )
        tensors = cannbench_contract.DataGenerator().generate_input_tensors_from_case(
            input_shapes=case.input_shapes,
            dtypes=case.dtypes,
            value_ranges=case.value_ranges,
            seed=int(raw["seed"]),
        )
        params = {
            item.name: tensors[i] if i < len(tensors) else None
            for i, item in enumerate(harness.info.inputs)
        }
        params.update({k: v for k, v in case.attrs.items() if k not in params})
        tensors = harness.get_input(**params)
        oracle = harness.reference_output(case, tensors)
        def native_fn(case=case, tensors=tensors):
            return harness.reference_output(case, tensors, native=True)
        modes = ("tuned", "baseline")
        for mode in modes:
            npu_tensors = cannbench_contract.map_tensors(tensors, lambda x: x.npu())
            call_params = harness.params(case, npu_tensors)
            actual = _call_single_kernel(
                mla_kernels,
                mla,
                **call_params,
                schedule_mode=mode,
            )
            passed, details = harness.compare(case, actual, oracle, tensors, native_fn)
            assert passed, f"{case.get_case_id_str()} ({mode}): {details}"


def test_dynamic_shape_kernel_cache_reuse() -> None:
    """Verify T.dynamic parameterization reuses compiled kernels across dynamic shapes."""
    if not _should_run_acceptance_test("dynamic_cache", _L3_TARGET_OP):
        pytest.skip(f"Scoped to {_L3_TARGET_OP} L3 target.")
    _require_npu_env()

    torch.manual_seed(0)
    if _L3_TARGET_OP_SHORT != "mla":
        from tests.ascend.benchmarks.attention_residual.golden import (  # noqa: PLC0415
            attention_residual as ar_golden,
        )

        ar_kernels = _load_scoped_operator_module("attention_residual")
        attention_residual = ar_kernels.attention_residual
        ar_kernels._KERNEL_CACHE.clear()
        n_depth, d_dim = 4, 256
        w = torch.randn(d_dim, dtype=torch.bfloat16)
        gamma = torch.rand(d_dim, dtype=torch.bfloat16) + 0.5
        for b_dim, t_dim in ((1, 16), (2, 32)):
            v_stack = torch.randn(n_depth, b_dim, t_dim, d_dim, dtype=torch.bfloat16)
            x = torch.randn(b_dim, t_dim, d_dim, dtype=torch.bfloat16)
            ref = ar_golden(v_stack, x, w, gamma, alpha=0.5, epsilon=1e-6)
            actual = _call_single_kernel(
                ar_kernels,
                attention_residual,
                v_stack=v_stack.npu(),
                x=x.npu(),
                w=w.npu(),
                gamma=gamma.npu(),
                alpha=0.5,
                epsilon=1e-6,
            )
            torch.testing.assert_close(actual.cpu().float(), ref.float(), rtol=2e-2, atol=2e-2)
        assert len(ar_kernels._KERNEL_CACHE) == 1

    if _L3_TARGET_OP_SHORT != "attention_residual":
        from tests.ascend.benchmarks.mla.golden import mla as mla_golden  # noqa: PLC0415

        mla_kernels = _load_scoped_operator_module("mla")
        mla = mla_kernels.mla
        mla_kernels._KERNEL_CACHE.clear()
        n_q, n_kv, d_nope, d_rope = 16, 1, 512, 64
        for b_dim, s_dim, s_kv in ((2, 64, 128), (4, 128, 256)):
            qn = torch.randn(b_dim, s_dim, n_q, d_nope, dtype=torch.bfloat16)
            qr = torch.randn(b_dim, s_dim, n_q, d_rope, dtype=torch.bfloat16)
            kn = torch.randn(b_dim, s_kv, n_kv, d_nope, dtype=torch.bfloat16)
            kr = torch.randn(b_dim, s_kv, n_kv, d_rope, dtype=torch.bfloat16)
            v = kn.clone()
            ref = mla_golden(
                qn,
                qr,
                kn,
                kr,
                v,
                numKVHeads=n_kv,
                scaleValue=-1.0,
                inputLayout="BSND",
                is_causal=True,
            )
            actual = _call_single_kernel(
                mla_kernels,
                mla,
                q_nope=qn.npu(),
                q_rope=qr.npu(),
                k_nope=kn.npu(),
                k_rope=kr.npu(),
                v=v.npu(),
                numKVHeads=n_kv,
                scaleValue=-1.0,
                inputLayout="BSND",
                is_causal=True,
            )
            torch.testing.assert_close(actual.cpu().float(), ref.float(), rtol=2e-2, atol=2e-2)
        assert len(mla_kernels._KERNEL_CACHE) == 1


def test_target_operator_full_cannbench_contract() -> None:
    """Verify full CANNBench contract for any scoped non-AR/MLA L3 target operator."""
    if not _should_run_acceptance_test("generic_cannbench", _L3_TARGET_OP):
        pytest.skip("Scoped to generic CANNBench L3 targets outside AR/MLA.")
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Ascend NPU device is unavailable on this host.")

    from tilefoundry_ascend.agent.campaign.anchor_specs import (  # noqa: PLC0415
        CannbenchOpsRegistry,
    )
    from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (  # noqa: PLC0415
        ensure_npu_compare_shim_dir,
    )

    reg = CannbenchOpsRegistry()
    spec = reg.resolve(_L3_TARGET_OP)
    if _L3_TARGET_MODULE_PATH:
        mod = _load_override_module(_L3_TARGET_MODULE_PATH, _L3_TARGET_OP_SHORT or "target")
    else:
        repo_root = Path(__file__).resolve().parents[3]
        mod = _load_override_module(
            str(repo_root / spec.module_relpath),
            _L3_TARGET_OP_SHORT or "target",
        )
    operator_api = getattr(mod, spec.entrypoint)
    harness = cannbench_contract.Harness(spec.task, spec.op_dir)

    if importlib.util.find_spec("npu_compare") is not None:
        harness.run(operator_api)
        return

    with tempfile.TemporaryDirectory(prefix="tf_l3_shim_") as tmp_dir:
        shim_dir = ensure_npu_compare_shim_dir(tmp_dir)
        sys.path.insert(0, str(shim_dir))
        try:
            harness.run(operator_api)
        finally:
            if sys.path and sys.path[0] == str(shim_dir):
                sys.path.pop(0)
