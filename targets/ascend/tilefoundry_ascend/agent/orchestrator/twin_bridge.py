"""CPU reference parity and CANNBench contract oracle for Ascend agents."""

from __future__ import annotations

import hashlib
import importlib.util
import inspect
import math
import runpy
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import yaml


def _max_abs_diff(ref_out: object, cand_out: object) -> float:
    if isinstance(ref_out, tuple) and isinstance(cand_out, (tuple, list)):
        if len(ref_out) != len(cand_out):
            return float("inf")
        return max(
            (_max_abs_diff(r_i, c_i) for r_i, c_i in zip(ref_out, cand_out, strict=True)),
            default=0.0,
        )
    if isinstance(ref_out, torch.Tensor) and isinstance(cand_out, torch.Tensor):
        if tuple(ref_out.shape) != tuple(cand_out.shape):
            return float("inf")
        diff = (
            cand_out.detach().cpu().to(torch.float32) - ref_out.detach().cpu().to(torch.float32)
        ).abs()
        return float(diff.max().item()) if diff.numel() > 0 else 0.0
    return float("inf")


def _compare_outputs(
    ref_out: object,
    cand_out: object,
    *,
    rtol: float,
    atol: float,
) -> tuple[bool, str]:
    """Compare oracle and candidate outputs, casting low-precision floats to float32."""
    if isinstance(ref_out, tuple):
        if not isinstance(cand_out, (tuple, list)) or len(cand_out) != len(ref_out):
            return (
                False,
                f"twin_output_structure_mismatch: expected {len(ref_out)} tuple elements",
            )
        for idx, (r_item, c_item) in enumerate(zip(ref_out, cand_out, strict=True)):
            ok, msg = _compare_outputs(r_item, c_item, rtol=rtol, atol=atol)
            if not ok:
                return False, f"output[{idx}]: {msg}"
        return True, f"matched {len(ref_out)} tuple outputs"

    if not isinstance(ref_out, torch.Tensor) or not isinstance(cand_out, torch.Tensor):
        return (
            False,
            f"twin_output_type_mismatch: ref={type(ref_out).__name__}, "
            f"cand={type(cand_out).__name__}",
        )

    if tuple(ref_out.shape) != tuple(cand_out.shape):
        return (
            False,
            f"twin_shape_mismatch: expected {tuple(ref_out.shape)}, got {tuple(cand_out.shape)}",
        )

    ref_cpu = ref_out.detach().cpu()
    cand_cpu = cand_out.detach().cpu()
    try:
        if ref_cpu.is_floating_point() or cand_cpu.is_floating_point():
            torch.testing.assert_close(
                cand_cpu.to(torch.float32),
                ref_cpu.to(torch.float32),
                rtol=rtol,
                atol=atol,
            )
        else:
            torch.testing.assert_close(cand_cpu, ref_cpu, rtol=0, atol=0)
    except AssertionError as exc:
        return False, f"twin_numeric_mismatch: {exc}"
    return True, f"shape={tuple(ref_cpu.shape)}, dtype={ref_cpu.dtype}"


def _load_golden_callable(
    golden_source: str,
    entry_name: str | None = None,
) -> tuple[Callable[..., object], str]:
    """Load a CPU-callable PyTorch golden entrypoint from Python source text."""
    with tempfile.TemporaryDirectory(prefix="tf_golden_ref_") as tmp_dir:
        path = Path(tmp_dir) / "golden_ref.py"
        path.write_text(golden_source, encoding="utf-8")
        ns = runpy.run_path(str(path))
    if entry_name:
        target = ns.get(entry_name)
        if callable(target):
            return target, entry_name
        raise ValueError(f"golden entrypoint {entry_name!r} not found in golden_source")
    for name, obj in ns.items():
        if name.startswith("_") or name == "get_input" or not callable(obj):
            continue
        if getattr(obj, "__module__", None) in (None, "<run_path>"):
            return obj, name
    raise ValueError("golden_source defines no top-level callable entrypoint")


def _synthesize_reference_tensor(
    shape: Sequence[int],
    dtype: torch.dtype,
    generator: torch.Generator,
    *,
    profile: str,
) -> torch.Tensor:
    """Create a deterministic CPU tensor for contract or stress parity checks."""
    target_shape = tuple(int(dim) for dim in shape)
    if dtype.is_floating_point:
        scale = 0.5 if profile == "contract" else 1.5
        vals = torch.randn(target_shape, generator=generator, dtype=torch.float32) * scale
        return vals.to(dtype)
    if dtype == torch.bool:
        return torch.randint(0, 2, target_shape, generator=generator, dtype=torch.int64).to(
            torch.bool
        )
    hi = 4 if profile == "contract" else 16
    return torch.randint(0, hi, target_shape, generator=generator, dtype=dtype)


def _default_shape_from_dims(test_dims: Mapping[str, int] | None) -> tuple[int, ...]:
    if not test_dims:
        return (4, 8)
    m_val = test_dims.get("M", test_dims.get("rows", test_dims.get("S_q", test_dims.get("S"))))
    n_val = test_dims.get("D", test_dims.get("cols", test_dims.get("N", test_dims.get("K"))))
    if m_val is not None and n_val is not None and m_val > 0 and n_val > 0:
        return (int(m_val), int(n_val))
    positive = [int(v) for v in test_dims.values() if int(v) > 0]
    if len(positive) >= 2:
        return (positive[0], positive[1])
    if len(positive) == 1:
        return (positive[0],)
    return (4, 8)


def _build_reference_inputs(
    ref_fn: Callable[..., object],
    *,
    profile: str,
    run_seed: int,
    sample_inputs: Sequence[torch.Tensor] | None,
    input_specs: Sequence[tuple[Sequence[int], str]] | None,
    attrs: Mapping[str, object],
    test_dims: Mapping[str, int] | None,
) -> list[torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(run_seed)
    if sample_inputs is not None:
        if profile == "contract":
            return [tensor.detach().cpu().clone() for tensor in sample_inputs]
        return [
            _synthesize_reference_tensor(
                tuple(tensor.shape),
                tensor.dtype,
                generator,
                profile="stress",
            )
            for tensor in sample_inputs
        ]
    if input_specs is not None:
        return [
            _synthesize_reference_tensor(
                shape,
                _str_to_torch_dtype(dtype_name),
                generator,
                profile=profile,
            )
            for shape, dtype_name in input_specs
        ]
    sig = inspect.signature(ref_fn)
    tensor_param_count = sum(
        1
        for pname, param in sig.parameters.items()
        if pname not in attrs
        and param.kind
        in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
        and param.default is inspect.Parameter.empty
    )
    count = max(1, tensor_param_count)
    shape = _default_shape_from_dims(test_dims)
    return [
        _synthesize_reference_tensor(shape, torch.float32, generator, profile=profile)
        for _ in range(count)
    ]


def _call_with_filtered_attrs(
    fn: Callable[..., object],
    inputs: Sequence[torch.Tensor],
    attrs: Mapping[str, object],
) -> object:
    if not attrs:
        return fn(*inputs)
    sig = inspect.signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return fn(*inputs, **attrs)
    matched = {k: v for k, v in attrs.items() if k in sig.parameters}
    return fn(*inputs, **matched)


def _promote_floating_inputs(
    inputs: Sequence[torch.Tensor],
    dtype: torch.dtype,
) -> list[torch.Tensor]:
    return [
        tensor.to(dtype=dtype) if tensor.is_floating_point() else tensor.clone()
        for tensor in inputs
    ]


def _run_golden_cpu(
    ref_fn: Callable[..., object],
    inputs: Sequence[torch.Tensor],
    attrs: Mapping[str, object],
) -> object:
    with torch.no_grad():
        for precision in (torch.float64, torch.float32):
            promoted = _promote_floating_inputs(inputs, precision)
            try:
                return _call_with_filtered_attrs(ref_fn, promoted, attrs)
            except (RuntimeError, TypeError, ValueError):
                continue
        raw_clones = [tensor.detach().cpu().clone() for tensor in inputs]
        return _call_with_filtered_attrs(ref_fn, raw_clones, attrs)


def verify_reference_cpu_parity(
    candidate_runner: Callable[..., object],
    *,
    golden_fn: Callable[..., object] | None = None,
    golden_source: str | None = None,
    entry_name: str | None = None,
    input_specs: Sequence[tuple[Sequence[int], str]] | None = None,
    sample_inputs: Sequence[torch.Tensor] | None = None,
    attrs: Mapping[str, object] | None = None,
    test_dims: Mapping[str, int] | None = None,
    rtol: float = 1e-2,
    atol: float = 1e-2,
    dual_profile: bool = True,
    seed: int = 0,
) -> tuple[bool, str]:
    """Compare *candidate_runner* against a PyTorch golden CPU reference callable."""
    if golden_fn is None and (golden_source is None or not golden_source.strip()):
        return (
            False,
            "reference_parity_missing_golden: provide golden_fn or golden_source",
        )

    call_attrs = dict(attrs or {})
    try:
        if golden_fn is not None:
            ref_fn = golden_fn
            ref_name = entry_name or getattr(golden_fn, "__name__", "golden")
        else:
            assert golden_source is not None
            ref_fn, ref_name = _load_golden_callable(golden_source, entry_name=entry_name)

        contract_inputs = _build_reference_inputs(
            ref_fn,
            profile="contract",
            run_seed=int(seed),
            sample_inputs=sample_inputs,
            input_specs=input_specs,
            attrs=call_attrs,
            test_dims=test_dims,
        )
        ref_out = _run_golden_cpu(ref_fn, contract_inputs, call_attrs)
        cand_inputs = [tensor.detach().cpu().clone() for tensor in contract_inputs]
        cand_out = _call_with_filtered_attrs(candidate_runner, cand_inputs, call_attrs)
        ok, detail = _compare_outputs(ref_out, cand_out, rtol=rtol, atol=atol)
        if not ok:
            return False, f"reference_cpu_parity_mismatch(contract): {detail}"

        stress_suffix = ""
        if dual_profile:
            stress_inputs = _build_reference_inputs(
                ref_fn,
                profile="stress",
                run_seed=int(seed) + 17,
                sample_inputs=sample_inputs,
                input_specs=input_specs,
                attrs=call_attrs,
                test_dims=test_dims,
            )
            stress_ref = _run_golden_cpu(ref_fn, stress_inputs, call_attrs)
            stress_cand_in = [tensor.detach().cpu().clone() for tensor in stress_inputs]
            stress_cand = _call_with_filtered_attrs(
                candidate_runner, stress_cand_in, call_attrs
            )
            stress_ok, stress_detail = _compare_outputs(
                stress_ref, stress_cand, rtol=rtol, atol=atol
            )
            stress_err = _max_abs_diff(stress_ref, stress_cand)
            if not stress_ok:
                return False, f"reference_cpu_parity_mismatch(stress): {stress_detail}"
            stress_suffix = f", stress_profile=ok(max_abs={stress_err:.3e})"
    except Exception as exc:
        return False, f"reference_parity_execution_error: {type(exc).__name__}: {exc}"

    return (
        True,
        f"reference_cpu_parity_passed: entry={ref_name}, {detail}{stress_suffix}",
    )


@dataclass(frozen=True)
class NumericalTwinResult:
    """Result of CPU numerical parity comparison between golden and candidate source."""

    passed: bool
    error_message: str = ""
    detail: str = ""


def _resolve_candidate_cpu_runner(
    candidate_source: str,
    *,
    entry_name: str,
) -> tuple[Callable[..., object] | None, str]:
    """Extract a CPU-evaluable callable from Python candidate module source."""
    if not candidate_source.strip():
        return None, "twin_empty_candidate_source"
    try:
        with tempfile.TemporaryDirectory(prefix="tf_twin_py_") as tmp_dir:
            cand_path = Path(tmp_dir) / "candidate_mod.py"
            cand_path.write_text(candidate_source, encoding="utf-8")
            ns = runpy.run_path(str(cand_path))
    except Exception as exc:
        return None, f"twin_candidate_exec_error: {type(exc).__name__}: {exc}"

    preferred_names = [
        "cpu_reference",
        "reference",
        "golden",
        entry_name,
        "forward",
        "kernel",
        "main",
    ]
    candidates: list[Callable[..., object]] = []
    seen_ids: set[int] = set()
    for name in preferred_names:
        obj = ns.get(name)
        if callable(obj) and id(obj) not in seen_ids:
            seen_ids.add(id(obj))
            candidates.append(obj)
    for name, obj in ns.items():
        if name.startswith("_") or not callable(obj) or id(obj) in seen_ids:
            continue
        if getattr(obj, "__module__", None) in (None, "<run_path>"):
            seen_ids.add(id(obj))
            candidates.append(obj)

    for obj in candidates:
        type_name = type(obj).__name__
        if type_name in ("PrimFunc", "IRModule"):
            continue
        try:
            sig = inspect.signature(obj)
        except (TypeError, ValueError):
            continue
        if len(sig.parameters) == 0:
            continue
        return obj, ""

    return None, "twin_no_cpu_executable_candidate: candidate defines no CPU-evaluable callable"


def verify_numerical_twin(
    reference_source: str,
    candidate_source: str,
    *,
    entry_name: str | None = None,
    input_specs: Sequence[tuple[Sequence[int], str]] | None = None,
    sample_inputs: Sequence[torch.Tensor] | None = None,
    attrs: Mapping[str, object] | None = None,
    anchor_dims: Mapping[str, int] | None = None,
    rtol: float = 1e-2,
    atol: float = 1e-2,
    dual_profile: bool = True,
) -> NumericalTwinResult:
    """Verify candidate_source against Python/PyTorch golden source on CPU."""
    if not reference_source.strip():
        return NumericalTwinResult(passed=False, error_message="twin_empty_reference_source")
    try:
        ref_fn, ref_name = _load_golden_callable(reference_source, entry_name=entry_name)
    except Exception as exc:
        return NumericalTwinResult(
            passed=False,
            error_message=f"twin_reference_load_error: {type(exc).__name__}: {exc}",
        )

    runner, err = _resolve_candidate_cpu_runner(
        candidate_source,
        entry_name=ref_name,
    )
    if runner is None:
        return NumericalTwinResult(passed=False, error_message=err)

    ok, detail = verify_reference_cpu_parity(
        runner,
        golden_fn=ref_fn,
        entry_name=ref_name,
        input_specs=input_specs,
        sample_inputs=sample_inputs,
        attrs=attrs,
        test_dims=anchor_dims,
        rtol=rtol,
        atol=atol,
        dual_profile=dual_profile,
    )
    return NumericalTwinResult(
        passed=ok,
        error_message="" if ok else detail,
        detail=detail,
    )



_CANNNBENCH_THRESHOLDS: dict[str, float] = {
    "float16": 2**-10,
    "bfloat16": 2**-7,
    "float32": 2**-13,
    "float64": 2**-13,
    "hifloat32": 2**-11,
    "float8_e4m3fn": 2**-3,
    "float8_e5m2": 2**-2,
    "int8": 0.0,
    "int16": 0.0,
    "int32": 0.0,
    "int64": 0.0,
    "uint8": 0.0,
    "uint16": 0.0,
    "uint32": 0.0,
    "uint64": 0.0,
    "bool": 0.0,
}
_SMALL_VALUE_THRESHOLDS: dict[str, float] = {
    "float16": 2**-11,
    "bfloat16": 2**-8,
    "float32": 2**-14,
    "float64": 2**-14,
}
_SMALL_VALUE_ERRORS: dict[str, float] = {
    "float16": 2**-16,
    "bfloat16": 2**-16,
    "float32": 2**-30,
    "float64": 2**-30,
}
_CANCEL_BOUNDARIES: dict[str, float] = {
    "float16": 2**-5,
    "bfloat16": 2**-3,
    "float32": 2**-8,
    "float64": 2**-8,
}
_BIT_VIEW_DTYPES: dict[torch.dtype, torch.dtype] = {
    torch.float16: torch.int16,
    torch.bfloat16: torch.int16,
    torch.float32: torch.int32,
    torch.float64: torch.int64,
}


def _clone_tensor_tree(value: object) -> object:
    """Deeply clone tensors inside nested lists/tuples/dicts to isolate in-place aliasing."""
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, list):
        return [_clone_tensor_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tensor_tree(item) for item in value)
    if isinstance(value, dict):
        return {k: _clone_tensor_tree(v) for k, v in value.items()}
    return value


def _flatten_output_tensors(value: object) -> list[object]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        out: list[object] = []
        for item in value:
            if isinstance(item, (list, tuple)):
                out.extend(_flatten_output_tensors(item))
            else:
                out.append(item)
        return out
    return [value]


def _dtype_str_from_torch(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def compare_cannbench_tensor(
    actual: object,
    golden: object,
    *,
    threshold: float | None = None,
    precision_thresholds: Mapping[str, float] | None = None,
    native_output: torch.Tensor | None = None,
) -> dict[str, object]:
    """Compare actual and golden tensors under CANNBench MERE/MARE and bit-exact rules."""
    if not isinstance(actual, torch.Tensor) or not isinstance(golden, torch.Tensor):
        return {
            "passed": False,
            "mere": float("inf"),
            "mare": float("inf"),
            "error_msg": f"expected Tensor outputs, got {type(actual).__name__} vs {type(golden).__name__}",
        }
    if tuple(actual.shape) != tuple(golden.shape):
        return {
            "passed": False,
            "mere": float("inf"),
            "mare": float("inf"),
            "error_msg": f"shape mismatch: actual={tuple(actual.shape)}, golden={tuple(golden.shape)}",
        }
    if not actual.is_contiguous():
        return {
            "passed": False,
            "mere": float("inf"),
            "mare": float("inf"),
            "error_msg": f"actual tensor is non-contiguous: shape={tuple(actual.shape)}, stride={actual.stride()}",
        }

    dtype_name = _dtype_str_from_torch(actual.dtype)
    if threshold is None:
        if precision_thresholds and dtype_name in precision_thresholds:
            eff_thresh = float(precision_thresholds[dtype_name])
        else:
            eff_thresh = float(_CANNNBENCH_THRESHOLDS.get(dtype_name, 2**-13))
    else:
        eff_thresh = float(threshold)

    actual_cpu = actual.detach().cpu()
    golden_cpu = golden.detach().cpu()

    if eff_thresh == 0.0 and actual_cpu.is_floating_point():
        bit_dtype = _BIT_VIEW_DTYPES.get(actual_cpu.dtype)
        if bit_dtype is not None:
            gold_cast = golden_cpu.to(actual_cpu.dtype).contiguous()
            out_bits = actual_cpu.contiguous().view(bit_dtype)
            gold_bits = gold_cast.view(bit_dtype)
            eq = torch.equal(out_bits, gold_bits)
            mism = 0 if eq else int((out_bits != gold_bits).sum().item())
            return {
                "passed": bool(eq),
                "threshold": eff_thresh,
                "mere": 0.0,
                "mare": 0.0,
                "mismatch_count": mism,
                "error_msg": "" if eq else f"bit-exact mismatch: {mism}/{actual_cpu.numel()} elements",
            }

    if actual_cpu.dtype == torch.bool or not actual_cpu.is_floating_point():
        gold_int = golden_cpu.to(actual_cpu.dtype)
        if torch.equal(actual_cpu, gold_int):
            return {"passed": True, "threshold": eff_thresh, "mere": 0.0, "mare": 0.0, "error_msg": ""}
        if actual_cpu.dtype == torch.bool:
            mism = int((actual_cpu != gold_int).sum().item())
            return {
                "passed": False,
                "threshold": eff_thresh,
                "mere": 0.0,
                "mare": 0.0,
                "error_msg": f"bool mismatch: {mism}/{actual_cpu.numel()} elements",
            }
        diff = torch.abs(actual_cpu.to(torch.int64) - gold_int.to(torch.int64))
        mism = int((diff > max(eff_thresh, 0.0)).sum().item())
        return {
            "passed": mism == 0,
            "threshold": eff_thresh,
            "mere": 0.0,
            "mare": 0.0,
            "mismatch_count": mism,
            "error_msg": "" if mism == 0 else f"integer diff > {eff_thresh}: {mism}/{actual_cpu.numel()} elements",
        }

    target_dtype = actual_cpu.dtype
    gold_trunc = golden_cpu.to(target_dtype).double()
    out_f64 = actual_cpu.to(torch.float64)

    nan_out = torch.isnan(out_f64)
    nan_gold = torch.isnan(gold_trunc)
    if bool((nan_out != nan_gold).any().item()):
        return {"passed": False, "threshold": eff_thresh, "mere": float("inf"), "mare": float("inf"), "error_msg": "NaN position mismatch"}

    inf_out = torch.isinf(out_f64)
    inf_gold = torch.isinf(gold_trunc)
    inf_match_mask = torch.zeros_like(out_f64, dtype=torch.bool)
    if bool((inf_out | inf_gold).any().item()):
        inf_mismatch = inf_out != inf_gold
        if bool(inf_mismatch.any().item()):
            max_finite = float(torch.finfo(target_dtype).max)
            out_m = inf_out & ~inf_gold
            gold_m = inf_gold & ~inf_out
            if bool(out_m.any().item()):
                out_f64 = out_f64.clone()
                out_f64[out_m] = torch.sign(out_f64[out_m]) * max_finite
            if bool(gold_m.any().item()):
                gold_trunc = gold_trunc.clone()
                gold_trunc[gold_m] = torch.sign(gold_trunc[gold_m]) * max_finite
        both_inf = inf_out & inf_gold
        if bool(both_inf.any().item()):
            if not bool((torch.sign(out_f64[both_inf]) == torch.sign(gold_trunc[both_inf])).all().item()):
                return {"passed": False, "threshold": eff_thresh, "mere": float("inf"), "mare": float("inf"), "error_msg": "Inf sign mismatch"}
            inf_match_mask[both_inf] = True

    diff = (out_f64 - gold_trunc).abs()
    gold_abs = torch.abs(gold_trunc)
    rel_err = diff / (gold_abs + 1e-7)
    valid_mask = ~(torch.isnan(rel_err) | torch.isinf(rel_err) | inf_match_mask)
    if not bool(valid_mask.any().item()):
        return {"passed": True, "threshold": eff_thresh, "mere": 0.0, "mare": 0.0, "error_msg": ""}

    valid_rel = rel_err[valid_mask]
    mere = float(valid_rel.mean().item())
    mare = float(valid_rel.max().item())
    mare_thresh = 10.0 * eff_thresh
    if mere < eff_thresh and mare < mare_thresh:
        return {"passed": True, "threshold": eff_thresh, "mere": mere, "mare": mare, "error_msg": ""}

    mismatch_mask = (rel_err > mare_thresh) & valid_mask
    sv_thresh = _SMALL_VALUE_THRESHOLDS.get(dtype_name, 2**-14)
    sv_err = _SMALL_VALUE_ERRORS.get(dtype_name, 2**-30)
    sv_mask = (gold_abs < sv_thresh) & valid_mask
    sv_npu_err = int((sv_mask & (diff > sv_err)).sum().item())
    if native_output is not None:
        cpu_f64 = native_output.detach().cpu().to(torch.float64)
        cpu_diff = (cpu_f64 - gold_trunc).abs()
    else:
        cpu_diff = torch.zeros_like(gold_trunc)
    sv_cpu_err = int((sv_mask & (cpu_diff > sv_err)).sum().item())
    sv_passed = (sv_npu_err / max(sv_cpu_err, 1)) <= 2 if bool(sv_mask.any().item()) else True

    cancel_bound = _CANCEL_BOUNDARIES.get(dtype_name, 2**-8)
    cancel_mask = (torch.abs(out_f64) < cancel_bound) & (gold_abs < cancel_bound) & (gold_abs >= sv_thresh) & valid_mask
    cancel_npu_err = int((cancel_mask & (rel_err > mare_thresh)).sum().item())
    cpu_rel = cpu_diff / (gold_abs + 1e-7)
    cancel_cpu_err = int((cancel_mask & (cpu_rel > mare_thresh)).sum().item())
    cancel_passed = (cancel_npu_err / max(cancel_cpu_err, 1)) <= 2 if bool(cancel_mask.any().item()) else True

    normal_mask = ~sv_mask & ~cancel_mask & valid_mask
    normal_err = int((mismatch_mask & normal_mask).sum().item())
    normal_cpu_err = int((normal_mask & (cpu_rel > mare_thresh)).sum().item())
    if normal_err == 0:
        normal_passed = True
    elif normal_cpu_err == 0:
        normal_passed = False
    else:
        normal_passed = (normal_err / normal_cpu_err) <= 2

    passed = normal_passed and sv_passed and cancel_passed
    return {
        "passed": bool(passed),
        "threshold": eff_thresh,
        "mere": mere,
        "mare": mare,
        "error_msg": "" if passed else f"MERE={mere:.3e} (limit={eff_thresh:.3e}), MARE={mare:.3e} (limit={mare_thresh:.3e})",
    }


def _str_to_torch_dtype(dtype_str: str) -> torch.dtype:
    key = dtype_str.strip().lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "float64": torch.float64,
        "fp64": torch.float64,
        "int8": torch.int8,
        "uint8": torch.uint8,
        "int16": torch.int16,
        "uint16": getattr(torch, "uint16", torch.int16),
        "int32": torch.int32,
        "uint32": getattr(torch, "uint32", torch.int32),
        "int64": torch.int64,
        "uint64": getattr(torch, "uint64", torch.int64),
        "bool": torch.bool,
    }
    if key not in mapping:
        raise ValueError(f"Unsupported dtype {dtype_str!r}")
    return mapping[key]


def _coerce_vr_endpoint(val: object) -> object:
    if not isinstance(val, str):
        return val
    low = val.strip().lower()
    if low in ("nan", "inf", "-inf", "+inf"):
        return low
    try:
        return float(val)
    except ValueError:
        return val


def _synthesize_contract_tensor(
    shape: Sequence[int] | None,
    dtype_str: str,
    value_range: object,
    generator: torch.Generator,
    *,
    param_name: str = "",
) -> torch.Tensor | None:
    """Synthesize an input tensor matching proto/cases value_range and Q4_0 rules."""
    if shape is None or (isinstance(shape, (list, tuple)) and any(x is None for x in shape)):
        return None
    if str(dtype_str).strip().lower() in ("none", ""):
        return None
    target_shape = [int(d) for d in shape]
    tdtype = _str_to_torch_dtype(dtype_str)

    if isinstance(value_range, (list, tuple)) and len(value_range) == 3 and value_range[0] == "unique":
        lo, hi = int(value_range[1]), int(value_range[2])
        numel = 1
        for d in target_shape:
            numel *= d
        span = hi - lo + 1
        if span < numel:
            raise ValueError(f"unique range [{lo}, {hi}] too small for {numel} elements")
        perm = torch.randperm(span, generator=generator)[:numel]
        return (perm + lo).reshape(target_shape).to(tdtype)

    if isinstance(value_range, (list, tuple)) and len(value_range) >= 2:
        lo_raw, hi_raw = _coerce_vr_endpoint(value_range[0]), _coerce_vr_endpoint(value_range[1])
    elif isinstance(value_range, (list, tuple)) and len(value_range) == 1:
        lo_raw = hi_raw = _coerce_vr_endpoint(value_range[0])
    elif value_range is not None:
        lo_raw = hi_raw = _coerce_vr_endpoint(value_range)
    else:
        if tdtype == torch.uint8:
            lo_raw, hi_raw = 0, 255
        elif tdtype.is_floating_point:
            lo_raw, hi_raw = -1.0, 1.0
        else:
            lo_raw, hi_raw = 0, 10

    def _to_float(v: object) -> float:
        if isinstance(v, str):
            if v == "inf" or v == "+inf":
                return float("inf")
            if v == "-inf":
                return float("-inf")
            if v == "nan":
                return float("nan")
            return float(v)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
        raise TypeError(f"Unsupported value_range endpoint: {v!r}")

    if tdtype == torch.bool:
        lo_i = max(0, min(1, int(_to_float(lo_raw))))
        hi_i = max(0, min(1, int(_to_float(hi_raw))))
        if lo_i > hi_i:
            lo_i, hi_i = hi_i, lo_i
        if lo_i == hi_i:
            return torch.full(target_shape, bool(lo_i), dtype=torch.bool)
        return torch.randint(lo_i, hi_i + 1, target_shape, generator=generator, dtype=torch.int64).to(torch.bool)

    if tdtype.is_floating_point:
        lo_f = _to_float(lo_raw)
        hi_f = _to_float(hi_raw)
        if math.isnan(lo_f) and math.isnan(hi_f):
            base = (torch.rand(target_shape, dtype=torch.float32, generator=generator) * 2.0 - 1.0).to(tdtype)
            nan_mask = torch.rand(target_shape, dtype=torch.float32, generator=generator) < 0.5
            base[nan_mask] = float("nan")
            return base
        if math.isinf(lo_f) or math.isinf(hi_f):
            if lo_f == hi_f:
                return torch.full(target_shape, lo_f, dtype=tdtype)
            base = (torch.rand(target_shape, dtype=torch.float32, generator=generator) * 2.0 - 1.0).to(tdtype)
            fmin = lo_f if math.isfinite(lo_f) else None
            fmax = hi_f if math.isfinite(hi_f) else None
            if fmin is not None or fmax is not None:
                base = torch.clamp(base, min=fmin, max=fmax)
            flat = base.flatten()
            n_edge = max(1, len(flat) // 20)
            if flat.numel() > 0:
                if lo_f == float("-inf"):
                    flat[:n_edge] = float("-inf")
                if hi_f == float("inf"):
                    flat[-n_edge:] = float("inf")
            return base
        if lo_f == hi_f:
            return torch.full(target_shape, lo_f, dtype=tdtype)
        finfo = torch.finfo(tdtype)
        lo_c = max(lo_f, float(finfo.min))
        hi_c = min(hi_f, float(finfo.max))
        vals = torch.rand(target_shape, dtype=torch.float64, generator=generator) * (hi_c - lo_c) + lo_c
        if lo_c < 0.0 < hi_c and vals.numel() > 0:
            near_zero = vals.abs() < 1e-4
            if bool(near_zero.any().item()):
                pos_step = torch.tensor(min(1e-2, hi_c * 0.5), dtype=torch.float64)
                neg_step = torch.tensor(max(-1e-2, lo_c * 0.5), dtype=torch.float64)
                vals[near_zero] = torch.where(vals[near_zero] >= 0.0, pos_step, neg_step)
        return torch.clamp(vals, float(finfo.min), float(finfo.max)).to(tdtype)

    lo_i = int(_to_float(lo_raw))
    hi_i = int(_to_float(hi_raw))
    if lo_i == hi_i:
        return torch.full(target_shape, lo_i, dtype=tdtype)
    if lo_i > hi_i:
        lo_i, hi_i = hi_i, lo_i
    if tdtype == torch.int64 and hi_i >= (1 << 63) - 1:
        hi_i = (1 << 63) - 2
    return torch.randint(lo_i, hi_i + 1, target_shape, generator=generator, dtype=tdtype)


class CannbenchContractOracle:
    """Faithful CANNBench contract input synthesizer and MERE/MARE + bit-exact accuracy oracle."""

    def __init__(
        self,
        operator_name: str | None = None,
        *,
        task_dir: str | Path | None = None,
        tasks_root: str | Path | None = None,
    ) -> None:
        if task_dir is not None:
            self.task_dir = Path(task_dir).expanduser().resolve()
            self.task = f"{self.task_dir.parent.name}/{self.task_dir.name}"
            self.op_name = self.task_dir.name
        elif operator_name is not None:
            from tilefoundry_ascend.agent.campaign.anchor_specs import (  # noqa: PLC0415
                CannbenchOpsRegistry,
            )

            registry = (
                CannbenchOpsRegistry(tasks_root=tasks_root)
                if tasks_root is not None
                else CannbenchOpsRegistry.default()
            )
            spec = registry.resolve(operator_name)
            self.task_dir = spec.op_dir
            self.task = spec.task
            self.op_name = spec.op_id
        else:
            raise ValueError("Either operator_name or task_dir must be provided")

        proto_raw = yaml.safe_load((self.task_dir / "proto.yaml").read_text(encoding="utf-8")) or {}
        self.proto: dict[str, object] = dict(proto_raw.get("operator") or proto_raw)
        raw_prec = self.proto.get("precision_thresholds")
        prec_map = raw_prec if isinstance(raw_prec, Mapping) else {}
        self.precision_thresholds: dict[str, float] = {
            str(k).lower(): float(v)
            for k, v in prec_map.items()
        }
        cases_raw = yaml.safe_load((self.task_dir / "cases.yaml").read_text(encoding="utf-8")) or {}
        self.cases: list[dict[str, object]] = [
            dict(row) for row in (cases_raw.get("cases") or ()) if isinstance(row, Mapping)
        ]
        self._cases_by_id: dict[int, dict[str, object]] = {
            int(c["case_id"]): c
            for c in self.cases
            if "case_id" in c and isinstance(c["case_id"], int)
        }

        ref_file = self.task_dir / "reference.py"
        if not ref_file.is_file():
            ref_file = self.task_dir / "golden.py"
        if not ref_file.is_file():
            raise FileNotFoundError(f"Neither reference.py nor golden.py found in {self.task_dir}")
        mod_spec = importlib.util.spec_from_file_location(
            f"_cannbench_oracle_{self.task.replace('/', '_')}", str(ref_file)
        )
        if mod_spec is None or mod_spec.loader is None:
            raise RuntimeError(f"Failed to load reference module {ref_file}")
        mod = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(mod)
        self.reference_module = mod
        self.reference_fn = getattr(mod, self.op_name, None)
        if not callable(self.reference_fn):
            for name, obj in vars(mod).items():
                if not name.startswith("_") and name != "get_input" and callable(obj):
                    self.reference_fn = obj
                    break
        if not callable(self.reference_fn):
            raise AttributeError(f"No reference callable found in {ref_file}")
        self.get_input = getattr(mod, "get_input", None)

    def get_case(self, case_num: int) -> dict[str, object]:
        """Return parsed cases.yaml dictionary for *case_num*."""
        cid = int(case_num)
        if cid not in self._cases_by_id:
            raise KeyError(f"case_id={cid} not found in {self.task_dir / 'cases.yaml'}")
        return self._cases_by_id[cid]

    def _deterministic_seed(self, case_num: int, seed: int | None) -> int:
        if seed is not None:
            return int(seed)
        label = f"{self.task}_{int(case_num)}"
        return int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:8], "big") % (2**31)

    def synthesize_case_inputs(
        self,
        case_num: int,
        *,
        seed: int | None = None,
    ) -> tuple[list[object], dict[str, object]]:
        """Synthesize faithful CPU tensors and call kwargs for *case_num*."""
        case = self.get_case(case_num)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self._deterministic_seed(case_num, seed))

        raw_shapes = case.get("input_shape") or []
        if isinstance(raw_shapes, (list, tuple)) and raw_shapes and isinstance(raw_shapes[0], int):
            shapes: list[object] = [raw_shapes]
        elif isinstance(raw_shapes, (list, tuple)):
            shapes = list(raw_shapes)
        else:
            shapes = []

        raw_dtypes = case.get("dtype") or ["float32"]
        if isinstance(raw_dtypes, str):
            dtypes: list[object] = [raw_dtypes]
        elif isinstance(raw_dtypes, (list, tuple)):
            dtypes = list(raw_dtypes)
        else:
            dtypes = ["float32"]
        if len(dtypes) == 1 and len(shapes) > 1:
            dtypes = dtypes * len(shapes)

        raw_vr = case.get("value_range")
        if not raw_vr or not isinstance(raw_vr, (list, tuple)):
            value_ranges: list[object] = [None] * len(shapes)
        elif not isinstance(raw_vr[0], (list, tuple)):
            value_ranges = [raw_vr] * len(shapes)
        else:
            value_ranges = list(raw_vr)
            if len(value_ranges) < len(shapes):
                value_ranges.extend([value_ranges[-1]] * (len(shapes) - len(value_ranges)))

        raw_proto_inputs = self.proto.get("inputs")
        proto_inputs = list(raw_proto_inputs) if isinstance(raw_proto_inputs, Sequence) else []
        tensors: list[object] = []
        for idx, sh in enumerate(shapes):
            dt = dtypes[idx] if idx < len(dtypes) else dtypes[-1]
            vr = value_ranges[idx] if idx < len(value_ranges) else value_ranges[-1]
            pname = (
                str(proto_inputs[idx].get("name", ""))
                if idx < len(proto_inputs) and isinstance(proto_inputs[idx], Mapping)
                else ""
            )
            if (
                sh is not None
                and isinstance(sh, list)
                and sh
                and isinstance(sh[0], list)
            ):
                sub_list: list[object] = []
                for j, sub_sh in enumerate(sh):
                    sub_dt = dt[j] if isinstance(dt, list) and j < len(dt) else dt
                    sub_vr = vr[j] if isinstance(vr, list) and j < len(vr) and isinstance(vr[j], (list, tuple)) else vr
                    sub_list.append(
                        _synthesize_contract_tensor(
                            sub_sh, str(sub_dt), sub_vr, gen, param_name=pname
                        )
                    )
                tensors.append(sub_list)
            else:
                sh_seq = sh if isinstance(sh, Sequence) and not isinstance(sh, (str, bytes)) else None
                tensors.append(
                    _synthesize_contract_tensor(
                        sh_seq, str(dt), vr, gen, param_name=pname
                    )
                )

        raw_attrs = case.get("attrs")
        attrs = dict(raw_attrs) if isinstance(raw_attrs, Mapping) else {}
        if callable(self.get_input):
            gi_params: dict[str, object] = {
                str(item.get("name", f"arg_{i}")): tensors[i] if i < len(tensors) else None
                for i, item in enumerate(proto_inputs)
                if isinstance(item, Mapping)
            }
            for k, v in attrs.items():
                gi_params.setdefault(str(k), v)
            gi_params.setdefault("skip2_exist", True)
            sig_gi = inspect.signature(self.get_input)
            if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_gi.parameters.values()):
                gi_out = self.get_input(**gi_params)
            else:
                filtered_gi = {k: v for k, v in gi_params.items() if k in sig_gi.parameters}
                gi_out = self.get_input(**filtered_gi)
            assert isinstance(gi_out, Iterable) or isinstance(gi_out, Sequence)
            tensors = list(gi_out)

        assert self.reference_fn is not None
        call_kwargs = self._build_call_params(self.reference_fn, case, tensors)
        return tensors, call_kwargs

    def _build_call_params(
        self,
        fn: Callable[..., object],
        case: Mapping[str, object],
        tensors: Sequence[object],
    ) -> dict[str, object]:
        sig = inspect.signature(fn)
        params: dict[str, object] = {}
        tensor_idx = 0
        raw_attrs = case.get("attrs")
        attrs = dict(raw_attrs) if isinstance(raw_attrs, Mapping) else {}
        raw_proto_inputs = self.proto.get("inputs")
        proto_inputs = list(raw_proto_inputs) if isinstance(raw_proto_inputs, Sequence) else []
        proto_tensor_names = [
            str(item.get("name", ""))
            for item in proto_inputs
            if isinstance(item, Mapping)
            and ("Tensor" in str(item.get("dtype", "")) or isinstance(item.get("dtype"), list))
        ]
        for pname, param in sig.parameters.items():
            ann = str(param.annotation) if param.annotation != inspect.Parameter.empty else ""
            is_tensor_param = (
                "Tensor" in ann
                or (
                    ann == ""
                    and pname not in attrs
                    and (pname in proto_tensor_names or tensor_idx < len(tensors))
                    and param.default is inspect.Parameter.empty
                )
            )
            if is_tensor_param and tensor_idx < len(tensors):
                params[pname] = tensors[tensor_idx]
                tensor_idx += 1
            elif pname in attrs:
                val = attrs[pname]
                if isinstance(val, str) and val.lower() in ("inf", "-inf", "+inf", "nan"):
                    val = float(val)
                params[pname] = val
            elif param.default is not inspect.Parameter.empty:
                params[pname] = param.default
        return params

    def reference_output(
        self,
        case_num: int,
        raw_tensors: Sequence[object] | None = None,
        *,
        seed: int | None = None,
        native: bool = False,
    ) -> object:
        """Compute cloned, aliasing-safe oracle reference output for *case_num*."""
        if raw_tensors is None:
            raw_tensors, _ = self.synthesize_case_inputs(case_num, seed=seed)
        case = self.get_case(case_num)
        cloned_obj = _clone_tensor_tree(list(raw_tensors))
        cloned = list(cloned_obj) if isinstance(cloned_obj, list) else list(raw_tensors)
        suffix = "_bench" if native else "_oracle"
        fn = getattr(self.reference_module, f"{self.op_name}{suffix}", self.reference_fn)
        assert callable(fn)
        if not native:
            def _promote_double(val: object) -> object:
                if isinstance(val, torch.Tensor):
                    return val.double() if val.is_floating_point() else val
                if isinstance(val, list):
                    return [_promote_double(item) for item in val]
                if isinstance(val, tuple):
                    return tuple(_promote_double(item) for item in val)
                return val

            promoted = [_promote_double(t) for t in cloned]
            try:
                with torch.no_grad():
                    return fn(**self._build_call_params(fn, case, promoted))
            except Exception:
                recloned = _clone_tensor_tree(list(raw_tensors))
                cloned = list(recloned) if isinstance(recloned, list) else list(raw_tensors)
        with torch.no_grad():
            return fn(**self._build_call_params(fn, case, cloned))

    def verify_candidate(
        self,
        candidate_fn: Callable[..., object],
        *,
        case_nums: Sequence[int] | None = None,
        seed: int | None = None,
    ) -> tuple[bool, list[dict[str, object]]]:
        """Validate candidate callable against oracle across selected cases."""
        targets = (
            [int(c) for c in case_nums]
            if case_nums is not None
            else [int(c["case_id"]) for c in self.cases if isinstance(c.get("case_id"), int)]
        )
        raw_proto_outputs = self.proto.get("outputs")
        proto_outputs = list(raw_proto_outputs) if isinstance(raw_proto_outputs, Sequence) else []
        case_reports: list[dict[str, object]] = []
        all_passed = True

        for cid in targets:
            case = self.get_case(cid)
            raw_tensors, _ = self.synthesize_case_inputs(cid, seed=seed)
            oracle_out = self.reference_output(cid, raw_tensors, seed=seed, native=False)
            cand_obj = _clone_tensor_tree(raw_tensors)
            cand_tensors = list(cand_obj) if isinstance(cand_obj, Sequence) else list(raw_tensors)
            cand_params = self._build_call_params(candidate_fn, case, cand_tensors)
            actual_out = candidate_fn(**cand_params)

            actual_list = _flatten_output_tensors(actual_out)
            gold_list = _flatten_output_tensors(oracle_out)
            if len(actual_list) != len(gold_list):
                all_passed = False
                case_reports.append(
                    {
                        "case_num": cid,
                        "passed": False,
                        "error_msg": f"output count mismatch: actual={len(actual_list)}, oracle={len(gold_list)}",
                    }
                )
                continue

            out_details: list[dict[str, object]] = []
            case_ok = True
            for i, (act_t, gold_t) in enumerate(zip(actual_list, gold_list)):
                if i < len(proto_outputs) and isinstance(proto_outputs[i], Mapping):
                    if not proto_outputs[i].get("compare", True):
                        continue
                res = compare_cannbench_tensor(
                    act_t,
                    gold_t,
                    precision_thresholds=self.precision_thresholds,
                )
                res["output_idx"] = i
                out_details.append(res)
                if not res["passed"]:
                    case_ok = False
            if not case_ok:
                all_passed = False
            case_reports.append(
                {
                    "case_num": cid,
                    "passed": case_ok,
                    "outputs": out_details,
                }
            )
        return all_passed, case_reports


__all__ = [
    "CannbenchContractOracle",
    "NumericalTwinResult",
    "compare_cannbench_tensor",
    "verify_numerical_twin",
    "verify_reference_cpu_parity",
]
