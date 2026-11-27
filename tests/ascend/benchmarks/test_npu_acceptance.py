"""Ascend 950PR NPU numerical acceptance tests for adapted operators."""

from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
from collections.abc import Callable, Mapping
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
import l3_contract  # noqa: E402

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


def _environment_obstacle(recorder: l3_contract.Recorder, category: str, detail: str) -> None:
    """Report an environment block; promotion never degrades it to a skip."""
    if recorder.active:
        _note_runtime_modules(recorder)
        recorder.environment_block(category, detail)
        pytest.exit(
            f"L3 promotion blocked: {category}: {detail}",
            returncode=l3_contract.BLOCKED_EXIT_CODE,
        )
    pytest.skip(detail)


def _load_override_module(module_path_str: str, tag: str):
    mod_path = Path(module_path_str).resolve()
    spec = importlib.util.spec_from_file_location(f"_tf_l3_override_{tag}", mod_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load override candidate module: {mod_path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _note_runtime_modules(recorder: l3_contract.Recorder, kernel_module=None) -> None:
    if kernel_module is not None:
        recorder.note_module("candidate", kernel_module)
    recorder.note_module("cannbench_contract", cannbench_contract)
    for name in ("torch", "torch_npu", "tilelang", "npu_compare"):
        module = sys.modules.get(name)
        if module is not None:
            recorder.note_module(name, module)


def _require_candidate_file(recorder: l3_contract.Recorder, mod_path: Path) -> None:
    if not mod_path.is_file():
        _environment_obstacle(
            recorder,
            "candidate_missing",
            f"declared candidate module does not exist: {mod_path}",
        )


def _load_scoped_operator_module(op_short_name: str, recorder: l3_contract.Recorder):
    if _L3_TARGET_MODULE_PATH and _L3_TARGET_OP_SHORT == op_short_name:
        mod_path = Path(_L3_TARGET_MODULE_PATH)
        _require_candidate_file(recorder, mod_path)
        module = _load_override_module(str(mod_path), op_short_name)
        recorder.note_candidate(module.__file__)
    else:
        module = importlib.import_module(f"examples.ascend950pr_ops.{op_short_name}.kernels")
        recorder.note_candidate(module.__file__)
    _note_runtime_modules(recorder, module)
    return module


def _require_npu_env(recorder: l3_contract.Recorder) -> None:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        _environment_obstacle(
            recorder,
            "npu_unavailable",
            "Ascend NPU device is unavailable on this host.",
        )
    if not cannbench_contract.comparator_available():
        _environment_obstacle(
            recorder,
            "comparator_unavailable",
            "CANNBench precision comparator is unavailable (npu_compare and "
            "TF_L3_CANNBENCH_REPO are both missing).",
        )
    _note_runtime_modules(recorder)


def _run_harness(
    harness: cannbench_contract.Harness,
    operator_api,
    recorder: l3_contract.Recorder,
    node: str,
) -> None:
    """Run a contract node, turning a required-cache failure into a block."""
    try:
        harness.run(operator_api, observer=recorder, node=node)
    except cannbench_contract.ReferenceCacheError as exc:
        category = cannbench_contract.reference_cache_error_category(exc)
        detail = str(exc)
        if recorder.active:
            recorder.environment_block(category, detail)
            pytest.exit(
                f"L3 promotion blocked: {category}: {detail}",
                returncode=l3_contract.BLOCKED_EXIT_CODE,
            )
        raise


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


def _failure_for_exception(
    node: str,
    case_id: str,
    stage: str,
    exc: Exception,
) -> dict[str, object]:
    """Classify an execution-time exception without parsing its message."""
    return l3_contract.infrastructure_failure(
        node=node,
        case_id=case_id,
        stage=stage,
        category="assertion" if isinstance(exc, AssertionError) else "exception",
        exception_type=type(exc).__name__,
        detail=f"{type(exc).__name__}: {exc}",
    )


def _assert_candidate_close(
    *,
    node: str,
    case_id: str,
    actual: torch.Tensor,
    reference: torch.Tensor,
    rtol: float,
    atol: float,
) -> None:
    """Turn only an explicit numeric assert_close deviation into candidate evidence."""
    try:
        torch.testing.assert_close(actual, reference, rtol=rtol, atol=atol)
    except AssertionError as exc:
        detail = f"torch.testing.assert_close: {exc}"
        failure = l3_contract.comparison_failure(
            node=node,
            case_id=case_id,
            role="candidate",
            comparator="torch.testing.assert_close",
            references=("oracle",),
            comparisons=({"output": 0, "passed": False, "message": str(exc)},),
            detail=detail,
        )
        raise l3_contract.CandidateComparisonError(detail, failure) from exc


def _require_contract_cases(
    recorder: l3_contract.Recorder,
    node: str,
    expected: tuple[str, ...],
    actual: tuple[str, ...],
) -> None:
    """Fail closed when the executed case set differs from the contract."""
    _ = recorder
    if actual != expected:
        detail = f"{node}: case identity mismatch; expected {expected}, executed {actual}"
        raise l3_contract.CoverageContractError(
            detail,
            l3_contract.coverage_failure(
                node=node,
                case_id=None,
                category="case_identity",
                detail=detail,
            ),
        )


def test_attention_residual_cannbench_cases() -> None:
    """Verify all 20 CANNBench AttentionResidual cases on NPU."""
    recorder = l3_contract.get_recorder()
    with recorder.node(l3_contract.AR_NODE):
        if not _should_run_acceptance_test("attention_residual", _L3_TARGET_OP):
            pytest.skip(f"Scoped to {_L3_TARGET_OP} L3 target.")
        _require_npu_env(recorder)
        ar_kernels = _load_scoped_operator_module("attention_residual", recorder)
        harness = cannbench_contract.Harness(l3_contract.AR_TASK, AR_DIR)
        expected = l3_contract.load_case_ids(AR_DIR)
        actual = tuple(str(case.case_num) for case in harness.cases)
        _require_contract_cases(recorder, l3_contract.AR_NODE, expected, actual)
        _run_harness(
            harness,
            ar_kernels.attention_residual,
            recorder,
            l3_contract.AR_NODE,
        )


def test_attention_residual_baseline_vs_tuned_parity() -> None:
    """Verify AttentionResidual baseline schedule numerical parity on NPU."""
    recorder = l3_contract.get_recorder()
    with recorder.node(l3_contract.AR_PARITY_NODE):
        if not _should_run_acceptance_test("attention_residual", _L3_TARGET_OP):
            pytest.skip(f"Scoped to {_L3_TARGET_OP} L3 target.")
        _require_npu_env(recorder)
        ar_kernels = _load_scoped_operator_module("attention_residual", recorder)
        attention_residual = ar_kernels.attention_residual
        harness = cannbench_contract.Harness(l3_contract.AR_TASK, AR_DIR)
        harness_case_ids = {str(case.case_num) for case in harness.cases}
        missing = [
            case_id
            for case_id in l3_contract.AR_PARITY_CASES
            if case_id not in harness_case_ids
        ]
        if missing:
            detail = f"AttentionResidual parity cases missing from cases.yaml: {missing}"
            raise l3_contract.CoverageContractError(
                detail,
                l3_contract.coverage_failure(
                    node=l3_contract.AR_PARITY_NODE,
                    case_id=None,
                    category="missing_case",
                    detail=detail,
                ),
            )
        target_cases = {int(case_id) for case_id in l3_contract.AR_PARITY_CASES}
        checked = 0
        failures: list[str] = []
        for case in harness.cases:
            if case.case_num not in target_cases:
                continue
            checked += 1
            case_id = str(case.case_num)
            recorder.case_started(l3_contract.AR_PARITY_NODE, case_id)
            stage = "prepare"
            try:
                tensors = harness.prepare(case)
                stage = "reference"
                oracle = harness.reference_output(case, tensors)
                npu_tensors = cannbench_contract.map_tensors(tensors, lambda x: x.npu())
                params = harness.params(case, npu_tensors)
                stage = "kernel"
                actual = _call_single_kernel(
                    ar_kernels,
                    attention_residual,
                    **params,
                    schedule_mode="baseline",
                )
                stage = "comparison"
                passed, details = harness.compare(
                    case,
                    actual,
                    oracle,
                    tensors,
                    lambda case=case, tensors=tensors: harness.reference_output(
                        case, tensors, native=True
                    ),
                )
            except Exception as exc:
                failure = _failure_for_exception(
                    l3_contract.AR_PARITY_NODE, case_id, stage, exc
                )
                recorder.case_finished(
                    l3_contract.AR_PARITY_NODE,
                    case_id,
                    "failed",
                    f"{type(exc).__name__}: {exc}",
                    failure=failure,
                )
                raise l3_contract.ClassifiedTestFailure(
                    f"{case.get_case_id_str()}: {type(exc).__name__}: {exc}"
                ) from exc
            if passed:
                recorder.case_finished(l3_contract.AR_PARITY_NODE, case_id, "passed")
            else:
                failures.append(case_id)
                recorder.case_finished(
                    l3_contract.AR_PARITY_NODE,
                    case_id,
                    "failed",
                    f"baseline mismatch: {json.dumps(details)[:700]}",
                    failure=l3_contract.comparison_failure(
                        node=l3_contract.AR_PARITY_NODE,
                        case_id=case_id,
                        role="baseline",
                        comparator="cannbench",
                        references=("oracle", "native"),
                        comparisons=details,
                        detail=f"baseline mismatch: {json.dumps(details)[:700]}",
                    ),
                )
        if checked != len(target_cases):
            detail = (
                f"{l3_contract.AR_PARITY_NODE}: checked {checked} parity cases, "
                f"expected {len(target_cases)}"
            )
            raise l3_contract.CoverageContractError(
                detail,
                l3_contract.coverage_failure(
                    node=l3_contract.AR_PARITY_NODE,
                    case_id=None,
                    category="missing_case",
                    detail=detail,
                ),
            )
        expected_ids = tuple(sorted(str(case_num) for case_num in target_cases))
        if expected_ids != l3_contract.AR_PARITY_CASES:
            detail = (
                f"{l3_contract.AR_PARITY_NODE}: parity case identity mismatch; "
                f"expected {l3_contract.AR_PARITY_CASES}, got {expected_ids}"
            )
            raise l3_contract.CoverageContractError(
                detail,
                l3_contract.coverage_failure(
                    node=l3_contract.AR_PARITY_NODE,
                    case_id=None,
                    category="case_identity",
                    detail=detail,
                ),
            )
        if failures:
            raise l3_contract.ClassifiedTestFailure(
                f"{l3_contract.AR_PARITY_NODE}: baseline mismatch in {failures}"
            )


def test_mla_cannbench_cases() -> None:
    """Verify all 20 CANNBench MLA cases on NPU."""
    recorder = l3_contract.get_recorder()
    with recorder.node(l3_contract.MLA_NODE):
        if not _should_run_acceptance_test("mla", _L3_TARGET_OP):
            pytest.skip(f"Scoped to {_L3_TARGET_OP} L3 target.")
        _require_npu_env(recorder)
        mla_kernels = _load_scoped_operator_module("mla", recorder)
        harness = cannbench_contract.Harness(l3_contract.MLA_TASK, MLA_DIR)
        expected = l3_contract.load_case_ids(MLA_DIR)
        actual = tuple(str(case.case_num) for case in harness.cases)
        _require_contract_cases(recorder, l3_contract.MLA_NODE, expected, actual)
        _run_harness(
            harness,
            mla_kernels.mla,
            recorder,
            l3_contract.MLA_NODE,
        )


def test_mla_supplemental_cases() -> None:
    """Verify supplemental GQA, tail, layout, and scale MLA cases on NPU."""
    recorder = l3_contract.get_recorder()
    with recorder.node(l3_contract.MLA_SUPPLEMENTAL_NODE):
        if not _should_run_acceptance_test("mla", _L3_TARGET_OP):
            pytest.skip(f"Scoped to {_L3_TARGET_OP} L3 target.")
        _require_npu_env(recorder)
        mla_kernels = _load_scoped_operator_module("mla", recorder)
        mla = mla_kernels.mla
        payload = json.loads(SUPP_MLA_PATH.read_text(encoding="utf-8"))
        harness = cannbench_contract.Harness(l3_contract.MLA_TASK, MLA_DIR)
        expected = l3_contract.load_supplemental_mla_ids()
        actual = tuple(raw["case_id"] for raw in payload["cases"])
        _require_contract_cases(recorder, l3_contract.MLA_SUPPLEMENTAL_NODE, expected, actual)
        failures: list[str] = []
        for raw in payload["cases"]:
            case_id = raw["case_id"]
            recorder.case_started(l3_contract.MLA_SUPPLEMENTAL_NODE, case_id)
            case_failed = False
            case_detail = ""
            stage = "prepare"
            try:
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
                stage = "reference"
                oracle = harness.reference_output(case, tensors)

                def native_fn(case=case, tensors=tensors):
                    return harness.reference_output(case, tensors, native=True)

                modes = ("tuned", "baseline")
                for mode in modes:
                    role = "candidate" if mode == "tuned" else "baseline"
                    stage = "kernel"
                    npu_tensors = cannbench_contract.map_tensors(tensors, lambda x: x.npu())
                    call_params = harness.params(case, npu_tensors)
                    actual_output = _call_single_kernel(
                        mla_kernels,
                        mla,
                        **call_params,
                        schedule_mode=mode,
                    )
                    stage = "comparison"
                    passed, details = harness.compare(
                        case, actual_output, oracle, tensors, native_fn
                    )
                    if not passed:
                        case_failed = True
                        case_detail = f"{mode} mismatch: {json.dumps(details)[:700]}"
                        failures.append(f"{case_id}:{mode}")
                        recorder.record_failure(
                            l3_contract.comparison_failure(
                                node=l3_contract.MLA_SUPPLEMENTAL_NODE,
                                case_id=case_id,
                                role=role,
                                comparator="cannbench",
                                references=("oracle", "native"),
                                comparisons=details,
                                detail=case_detail,
                            )
                        )
            except Exception as exc:
                failure = _failure_for_exception(
                    l3_contract.MLA_SUPPLEMENTAL_NODE, case_id, stage, exc
                )
                recorder.case_finished(
                    l3_contract.MLA_SUPPLEMENTAL_NODE,
                    case_id,
                    "failed",
                    f"{type(exc).__name__}: {exc}",
                    failure=failure,
                )
                raise l3_contract.ClassifiedTestFailure(
                    f"{case_id}: {type(exc).__name__}: {exc}"
                ) from exc
            recorder.case_finished(
                l3_contract.MLA_SUPPLEMENTAL_NODE,
                case_id,
                "failed" if case_failed else "passed",
                case_detail,
            )
        if failures:
            raise l3_contract.ClassifiedTestFailure(
                f"{l3_contract.MLA_SUPPLEMENTAL_NODE}: comparator deviation in {failures}"
            )


def _record_dynamic_shape(recorder: l3_contract.Recorder, case_id: str, check) -> None:
    recorder.case_started(l3_contract.DYNAMIC_NODE, case_id)
    try:
        check()
    except l3_contract.CandidateComparisonError as exc:
        recorder.case_finished(
            l3_contract.DYNAMIC_NODE,
            case_id,
            "failed",
            str(exc),
            failure=exc.failure,
        )
        raise
    except Exception as exc:
        failure = _failure_for_exception(
            l3_contract.DYNAMIC_NODE, case_id, "kernel", exc
        )
        recorder.case_finished(
            l3_contract.DYNAMIC_NODE,
            case_id,
            "failed",
            f"{type(exc).__name__}: {exc}",
            failure=failure,
        )
        raise l3_contract.ClassifiedTestFailure(
            f"{case_id}: {type(exc).__name__}: {exc}"
        ) from exc
    recorder.case_finished(l3_contract.DYNAMIC_NODE, case_id, "passed")


def test_dynamic_shape_kernel_cache_reuse() -> None:
    """Verify T.dynamic parameterization reuses compiled kernels across dynamic shapes."""
    recorder = l3_contract.get_recorder()
    with recorder.node(l3_contract.DYNAMIC_NODE):
        if not _should_run_acceptance_test("dynamic_cache", _L3_TARGET_OP):
            pytest.skip(f"Scoped to {_L3_TARGET_OP} L3 target.")
        _require_npu_env(recorder)

        torch.manual_seed(0)
        if _L3_TARGET_OP_SHORT != "mla":
            from tests.ascend.benchmarks.attention_residual.golden import (  # noqa: PLC0415
                attention_residual as ar_golden,
            )

            ar_kernels = _load_scoped_operator_module("attention_residual", recorder)
            attention_residual = ar_kernels.attention_residual
            ar_kernels._KERNEL_CACHE.clear()
            n_depth, d_dim = 4, 256
            w = torch.randn(d_dim, dtype=torch.bfloat16)
            gamma = torch.rand(d_dim, dtype=torch.bfloat16) + 0.5
            for index, (b_dim, t_dim) in enumerate(((1, 16), (2, 32))):
                case_id = l3_contract.AR_DYNAMIC_CASES[index]

                def check(b_dim=b_dim, t_dim=t_dim):
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
                    _assert_candidate_close(
                        node=l3_contract.DYNAMIC_NODE,
                        case_id=case_id,
                        actual=actual.cpu().float(),
                        reference=ref.float(),
                        rtol=2e-2,
                        atol=2e-2,
                    )

                _record_dynamic_shape(recorder, case_id, check)

            _record_dynamic_shape(
                recorder,
                l3_contract.AR_DYNAMIC_CASES[2],
                lambda: _assert_single_kernel(ar_kernels._KERNEL_CACHE),
            )

        if _L3_TARGET_OP_SHORT != "attention_residual":
            from tests.ascend.benchmarks.mla.golden import mla as mla_golden  # noqa: PLC0415

            mla_kernels = _load_scoped_operator_module("mla", recorder)
            mla = mla_kernels.mla
            mla_kernels._KERNEL_CACHE.clear()
            n_q, n_kv, d_nope, d_rope = 16, 1, 512, 64
            for index, (b_dim, s_dim, s_kv) in enumerate(((2, 64, 128), (4, 128, 256))):
                case_id = l3_contract.MLA_DYNAMIC_CASES[index]

                def check(b_dim=b_dim, s_dim=s_dim, s_kv=s_kv):
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
                    _assert_candidate_close(
                        node=l3_contract.DYNAMIC_NODE,
                        case_id=case_id,
                        actual=actual.cpu().float(),
                        reference=ref.float(),
                        rtol=2e-2,
                        atol=2e-2,
                    )

                _record_dynamic_shape(recorder, case_id, check)

            _record_dynamic_shape(
                recorder,
                l3_contract.MLA_DYNAMIC_CASES[2],
                lambda: _assert_single_kernel(mla_kernels._KERNEL_CACHE),
            )


def _assert_single_kernel(kernel_cache: dict) -> None:
    assert len(kernel_cache) == 1, f"expected one compiled kernel, got {len(kernel_cache)}"


def _collect_tensors(value: object) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        collected: list[torch.Tensor] = []
        for item in value:
            collected.extend(_collect_tensors(item))
        return collected
    if isinstance(value, dict):
        collected = []
        for item in value.values():
            collected.extend(_collect_tensors(item))
        return collected
    return []


def _perturb_outer_axis_tensors(
    tensors: object,
    attrs: Mapping[str, object],
) -> object | None:
    if "group_list" in attrs or any(
        isinstance(val, (list, tuple, dict)) for val in attrs.values()
    ):
        return None
    flat = _collect_tensors(tensors)
    candidates = [
        tensor
        for tensor in flat
        if tensor.ndim >= 2 and tensor.shape[0] >= 64 and tensor.is_floating_point()
    ]
    if not candidates:
        return None
    for axis_key in ("dim", "axis"):
        raw_dim = attrs.get(axis_key)
        if isinstance(raw_dim, int) and not isinstance(raw_dim, bool):
            if any(raw_dim in (0, -tensor.ndim) for tensor in candidates):
                return None
    outer_dim = candidates[0].shape[0]
    if any(tensor.ndim == 1 and tensor.shape[0] == outer_dim for tensor in flat):
        return None
    if any(outer_dim in tensor.shape[1:] for tensor in flat if tensor.ndim >= 2):
        return None
    if any(
        isinstance(val, int) and not isinstance(val, bool) and val == outer_dim
        for val in attrs.values()
    ):
        return None
    step = 32 if outer_dim % 32 == 0 and outer_dim > 64 else 1
    return cannbench_contract.map_tensors(
        tensors,
        lambda t: t[:-step].contiguous()
        if isinstance(t, torch.Tensor) and t.ndim >= 2 and t.shape[0] == outer_dim
        else t,
    )


def _verify_generic_kernel_cache_idempotence(
    harness: cannbench_contract.Harness,
    operator_api: Callable[..., object],
    kernel_cache: Mapping[object, object],
    recorder: l3_contract.Recorder,
) -> tuple[int, int]:
    def _call_with_tensors(case: cannbench_contract.Case, host_tensors: object) -> None:
        npu_tensors = cannbench_contract.map_tensors(
            host_tensors,
            lambda tensor: tensor.npu(),
        )
        params = harness.params(case, npu_tensors)
        operator_api(**params)

    def _replay_case(case: cannbench_contract.Case) -> None:
        _call_with_tensors(case, harness.prepare(case))

    def _perturb_case(case: cannbench_contract.Case) -> None:
        tensors = harness.prepare(case)
        perturbed = _perturb_outer_axis_tensors(tensors, case.attrs)
        if perturbed is not None:
            _call_with_tensors(case, perturbed)

    try:
        return l3_contract.verify_kernel_cache_idempotence(
            harness.cases,
            _replay_case,
            kernel_cache,
            perturb_case_fn=_perturb_case,
        )
    except l3_contract.ClassifiedTestFailure as exc:
        detail = str(exc)
        last_case_id = str(harness.cases[-1].case_num)
        failure = l3_contract.comparison_failure(
            node=l3_contract.GENERIC_NODE,
            case_id=last_case_id,
            role="candidate",
            comparator="kernel_cache_idempotence",
            references=("two_pass_idempotence",),
            comparisons=({"output": 0, "passed": False, "message": detail},),
            detail=detail,
        )
        recorder.case_finished(
            l3_contract.GENERIC_NODE,
            last_case_id,
            "failed",
            detail,
            failure=failure,
        )
        raise


def test_target_operator_full_cannbench_contract() -> None:
    """Verify full CANNBench contract for any scoped non-AR/MLA L3 target operator."""
    recorder = l3_contract.get_recorder()
    with recorder.node(l3_contract.GENERIC_NODE):
        if not _should_run_acceptance_test("generic_cannbench", _L3_TARGET_OP):
            pytest.skip("Scoped to generic CANNBench L3 targets outside AR/MLA.")
        _require_npu_env(recorder)

        from tilefoundry_ascend.agent.specs.anchor_specs import (  # noqa: PLC0415
            CannbenchOpsRegistry,
        )

        reg = CannbenchOpsRegistry()
        spec = reg.resolve(_L3_TARGET_OP)
        if _L3_TARGET_MODULE_PATH:
            mod_path = Path(_L3_TARGET_MODULE_PATH)
        else:
            repo_root = Path(__file__).resolve().parents[3]
            mod_path = repo_root / spec.module_relpath
        _require_candidate_file(recorder, mod_path)
        mod = _load_override_module(str(mod_path), _L3_TARGET_OP_SHORT or "target")
        recorder.note_candidate(mod.__file__)
        _note_runtime_modules(recorder, mod)
        operator_api = getattr(mod, spec.entrypoint)
        harness = cannbench_contract.Harness(spec.task, spec.op_dir)
        expected = l3_contract.load_case_ids(spec.op_dir)
        actual = tuple(str(case.case_num) for case in harness.cases)
        _require_contract_cases(recorder, l3_contract.GENERIC_NODE, expected, actual)
        _run_harness(
            harness,
            operator_api,
            recorder,
            l3_contract.GENERIC_NODE,
        )
        kernel_cache = getattr(mod, "_KERNEL_CACHE", None)
        if isinstance(kernel_cache, dict) and harness.cases:
            _verify_generic_kernel_cache_idempotence(
                harness,
                operator_api,
                kernel_cache,
                recorder,
            )
