"""Runtime twin bridge and HIR/TIR navigation prompt formatting for Ascend agents."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import hashlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
import runpy
import tempfile
from typing import Any

import torch
import yaml

from tilefoundry.analysis import check_program
from tilefoundry.analysis.check import program_dim_vars
from tilefoundry.cli.source import load_authored_ir
from tilefoundry.evaluator import EvalError, TensorValue, TupleValue, evaluate, to_torch_dtype
from tilefoundry.ir.core.module import Module
from tilefoundry.ir.hir.pipeline import (
    normalize_pipeline_references,
    rebind_cloned_pipeline_references,
)
from tilefoundry.ir.types import TensorType, TupleType
from tilefoundry.ir.types.substitute import dim_vars_by_name
from tilefoundry.ir.types.utils import static_dim_value
from tilefoundry.runtime import DictResource, runtime_func, runtime_module
from tilefoundry_ascend.ascend.passes.hir_tir_bridge import HirTirNavigationPacket
from tilefoundry_ascend.ascend.passes.physical_ledger import ObjectivePhysicalLedger


def _coerce_output_to_cpu_tensors(value: object, expected_type: object) -> object:
    """Convert candidate runner outputs into CPU torch.Tensors matching *expected_type*."""
    if isinstance(expected_type, TupleType):
        if isinstance(value, TupleValue):
            items = value.elements
        elif isinstance(value, (tuple, list)):
            items = tuple(value)
        else:
            raise TypeError(
                f"Expected tuple output of length {len(expected_type.fields)}, "
                f"got {type(value).__name__}"
            )
        if len(items) != len(expected_type.fields):
            raise ValueError(
                f"Expected {len(expected_type.fields)} tuple outputs, got {len(items)}"
            )
        return tuple(
            _coerce_output_to_cpu_tensors(item, field_t)
            for item, field_t in zip(items, expected_type.fields, strict=True)
        )

    if isinstance(expected_type, TensorType):
        if isinstance(value, TensorValue):
            value = value.data
        try:
            target_dtype = to_torch_dtype(expected_type.dtype)
        except EvalError:
            target_dtype = None
        if isinstance(value, torch.Tensor):
            tensor = value.detach().cpu()
            if target_dtype is not None and tensor.dtype != target_dtype:
                tensor = tensor.to(dtype=target_dtype)
            return tensor
        if target_dtype is not None:
            return torch.as_tensor(value, dtype=target_dtype, device="cpu")
        return torch.as_tensor(value, device="cpu")

    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return value


def _make_twin_method(ir_fn, runner: Callable[..., object], cls_name: str):
    """Build a @runtime_func method preserving *ir_fn* name and non-const signature."""
    non_const_names = [
        param.name for param in ir_fn.params if not getattr(param, "is_const", False)
    ]
    sig_params = [
        inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        *(
            inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for name in non_const_names
        ),
    ]

    def _method(self, *args):
        raw_out = runner(*args)
        return _coerce_output_to_cpu_tensors(raw_out, ir_fn.return_type)

    _method.__name__ = ir_fn.name
    _method.__qualname__ = f"{cls_name}.{ir_fn.name}"
    _method.__signature__ = inspect.Signature(sig_params)
    return runtime_func(_method)


def _make_unimplemented_runner(fn_name: str) -> Callable[..., object]:
    """Build a fallback runner for auxiliary module functions not bound to the kernel."""

    def _unimplemented(*_args: object) -> object:
        raise NotImplementedError(f"Runtime twin function {fn_name!r} has no bound runner")

    return _unimplemented


def build_tilelang_runtime_twin(
    authored_mod: Module,
    kernel_runner: Callable[..., object],
    *,
    func_name: str | None = None,
):
    """Construct a @runtime_module twin class wrapping *kernel_runner* for *authored_mod*."""
    if func_name is not None:
        target_fn = authored_mod.lookup(func_name)
    elif authored_mod.entry is not None:
        target_fn = authored_mod.entry_function()
    elif len(authored_mod.functions) == 1:
        target_fn = authored_mod.functions[0]
    else:
        raise ValueError(
            f"Module {authored_mod.name!r} declares no entry and {len(authored_mod.functions)} "
            f"functions; specify func_name"
        )

    cls_name = f"{authored_mod.name}TileLangTwin"
    namespace: dict[str, object] = {"__module__": __name__}
    for fn in authored_mod.functions:
        runner = kernel_runner if fn.name == target_fn.name else _make_unimplemented_runner(fn.name)
        namespace[fn.name] = _make_twin_method(fn, runner, cls_name)

    twin_base = type(cls_name, (), namespace)
    return runtime_module(authored_mod)(twin_base)


def _load_authored_module(
    source: str,
    tmpdir: Path,
    *,
    test_dims: Mapping[str, int] | None,
) -> Module:
    """Load HIR module from *source* via build(**matched_dims) or load_authored_ir."""
    path = tmpdir / "authored_hir.py"
    path.write_text(source, encoding="utf-8")
    if test_dims:
        try:
            ns = runpy.run_path(str(path))
            build_fn = ns.get("build")
            if callable(build_fn):
                sig = inspect.signature(build_fn)
                matched = {
                    str(k): int(v)
                    for k, v in test_dims.items()
                    if str(k) in sig.parameters
                    and sig.parameters[str(k)].kind
                    in (
                        inspect.Parameter.POSITIONAL_OR_KEYWORD,
                        inspect.Parameter.KEYWORD_ONLY,
                    )
                }
                if matched:
                    built = build_fn(**matched)
                    if isinstance(built, Module):
                        return built
        except (ValueError, AssertionError, TypeError):
            pass
    return load_authored_ir(str(path))


def _check_module_dims(mod: Module, *, test_dims: Mapping[str, int] | None):
    """Specialize declared DimVars using *test_dims* and return (checked_mod, checked_fn)."""
    normalize_pipeline_references(mod)
    authored_fn = mod.entry_function()
    declared = set(program_dim_vars(mod, authored_fn).keys()) | set(
        dim_vars_by_name(mod.effective_topologies())
    )
    filtered = (
        {str(k): int(v) for k, v in test_dims.items() if str(k) in declared}
        if test_dims
        else {}
    )
    checked_mod, checked_fn = check_program(
        mod, authored_fn, dims=filtered if filtered else None
    )
    rebind_cloned_pipeline_references(checked_mod)
    return checked_mod, checked_fn


def _concrete_shape(ttype: TensorType, test_dims: Mapping[str, int] | None) -> tuple[int, ...]:
    """Resolve static integer shape for *ttype*, falling back to *test_dims* or DimVar.lo."""
    dims: list[int] = []
    for dim in ttype.shape:
        val = static_dim_value(dim)
        if val is not None:
            dims.append(int(val))
            continue
        dim_name = getattr(dim, "name", None)
        if test_dims and dim_name and str(dim_name) in test_dims:
            dims.append(int(test_dims[str(dim_name)]))
            continue
        lo = getattr(dim, "lo", None)
        if isinstance(lo, int) and lo > 0:
            dims.append(lo)
            continue
        raise ValueError(f"Unresolved dynamic shape dimension {dim!r} in {ttype.shape!r}")
    return tuple(dims)


def _synthesize_seeded_tensor(
    ttype: TensorType,
    generator: torch.Generator,
    *,
    test_dims: Mapping[str, int] | None,
    param_name: str = "",
    domain_profile: str = "contract",
) -> torch.Tensor:
    """Create a deterministic CPU tensor matching *ttype* shape, dtype, and profile."""
    shape = _concrete_shape(ttype, test_dims)
    target_dtype = to_torch_dtype(ttype.dtype)
    if target_dtype.is_floating_point:
        norm_name = str(param_name).strip().lower()
        if domain_profile == "contract" and norm_name in ("w", "weight_zero", "bias_zero"):
            return torch.zeros(shape, dtype=target_dtype)
        scale = 0.5 if domain_profile == "contract" else 1.0
        vals = torch.randn(shape, generator=generator, dtype=torch.float32) * scale
        return vals.to(target_dtype)
    if target_dtype == torch.bool:
        return torch.randint(0, 2, shape, generator=generator, dtype=torch.int64).to(target_dtype)
    return torch.randint(0, 4, shape, generator=generator, dtype=target_dtype)


def _max_abs_diff(ref_out: object, cand_out: object) -> float:
    if isinstance(ref_out, TensorValue):
        ref_out = ref_out.data
    elif isinstance(ref_out, TupleValue):
        ref_out = tuple(
            item.data if isinstance(item, TensorValue) else item for item in ref_out.elements
        )
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
        diff = (cand_out.detach().cpu().to(torch.float32) - ref_out.detach().cpu().to(torch.float32)).abs()
        return float(diff.max().item()) if diff.numel() > 0 else 0.0
    return float("inf")


def _compare_outputs(
    ref_out: object,
    cand_out: object,
    *,
    rtol: float,
    atol: float,
) -> tuple[bool, str]:
    """Compare oracle and twin outputs, casting low-precision floats to float32."""
    if isinstance(ref_out, TensorValue):
        ref_out = ref_out.data
    elif isinstance(ref_out, TupleValue):
        ref_out = tuple(
            item.data if isinstance(item, TensorValue) else item for item in ref_out.elements
        )

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


def verify_twin_cpu_reference(
    authored_hir_source: str,
    candidate_runner: Callable[..., object],
    *,
    test_dims: Mapping[str, int] | None = None,
    rtol: float = 1e-2,
    atol: float = 1e-2,
    dual_profile: bool = False,
) -> tuple[bool, str]:
    """Compare *candidate_runner* via a @runtime_module twin against CPU HIR evaluation."""
    def _eval_profile(
        checked_mod,
        checked_fn,
        *,
        profile: str,
        seed: int,
    ) -> tuple[bool, str, float]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        inputs: list[torch.Tensor] = []
        for param in checked_fn.params:
            if not isinstance(param.type, TensorType):
                return (
                    False,
                    f"twin_unsupported_param_type: {param.name} has {type(param.type).__name__}",
                    float("inf"),
                )
            inputs.append(
                _synthesize_seeded_tensor(
                    param.type,
                    generator,
                    test_dims=test_dims,
                    param_name=param.name,
                    domain_profile=profile,
                )
            )
        ref_out = evaluate(checked_fn, *inputs)
        twin_cls = build_tilelang_runtime_twin(
            checked_mod,
            candidate_runner,
            func_name=checked_fn.name,
        )
        twin = twin_cls()
        const_weights = {
            param.name: tensor
            for param, tensor in zip(checked_fn.params, inputs, strict=True)
            if getattr(param, "is_const", False)
        }
        if const_weights:
            twin.load(DictResource(const_weights))
        activations = [
            tensor
            for param, tensor in zip(checked_fn.params, inputs, strict=True)
            if not getattr(param, "is_const", False)
        ]
        cand_out = getattr(twin, checked_fn.name)(*activations)
        ok, detail = _compare_outputs(ref_out, cand_out, rtol=rtol, atol=atol)
        max_abs = _max_abs_diff(ref_out, cand_out)
        return ok, detail, max_abs

    try:
        with tempfile.TemporaryDirectory(prefix="tf_twin_ref_") as tmp_dir:
            authored_mod = _load_authored_module(
                authored_hir_source,
                Path(tmp_dir),
                test_dims=test_dims,
            )
        checked_mod, checked_fn = _check_module_dims(authored_mod, test_dims=test_dims)
        ok, detail, _ = _eval_profile(checked_mod, checked_fn, profile="contract", seed=0)
        if not ok:
            return False, detail
        stress_suffix = ""
        if dual_profile:
            stress_ok, _stress_detail, stress_err = _eval_profile(
                checked_mod,
                checked_fn,
                profile="stress",
                seed=17,
            )
            stress_tag = "ok" if stress_ok else "warn"
            stress_suffix = f", stress_profile={stress_tag}(max_abs={stress_err:.3e})"
    except Exception as exc:
        return False, f"twin_execution_error: {type(exc).__name__}: {exc}"

    return True, f"twin_cpu_reference_passed: entry={checked_fn.name}, {detail}{stress_suffix}"


_SCOPE_TOPOLOGY_MAP: dict[str, str] = {
    "UB": "smem(AIV)",
    "ub": "smem(AIV)",
    "smem": "smem(AIV)",
    "L1": "tmem(AIC)",
    "L0A": "tmem(AIC)",
    "L0B": "tmem(AIC)",
    "L0C": "tmem(AIC)",
    "SSBUF": "ssbuf(block)",
}


def _ub_peak_bytes(ledger: ObjectivePhysicalLedger) -> int:
    """Return peak allocated UB bytes from a physical ledger."""
    for key in ("UB", "ub", "smem"):
        waterline = ledger.waterlines.get(key)
        if waterline is not None:
            return int(waterline.peak_allocated_bytes)
    return 0


def _format_scope_pct(waterline) -> str:
    """Format peak / capacity percentage for one ScopeWaterline."""
    if waterline is None or waterline.capacity_bytes <= 0:
        return "n/a"
    pct = 100.0 * float(waterline.peak_allocated_bytes) / float(waterline.capacity_bytes)
    return f"{pct:.2f}%"


def _structural_reference_banner(packet: HirTirNavigationPacket) -> str:
    """Render the structural-only banner when requested and resolved HIR dims diverge."""
    return (
        f"[Structural Blueprint Reference Only: HIR resolved at "
        f"dims={packet.resolved_hir_dims} vs Anchor dims={packet.requested_anchor_dims}]"
    )


def format_hir_tir_navigation_section(
    packet: HirTirNavigationPacket,
    *,
    current_tl_ledger: ObjectivePhysicalLedger | None = None,
) -> str:
    """Format HIR roofline, legal TIR ops, Z3 schedule, and ledger delta sections."""
    rf = packet.roofline
    display_dims = rf.resolved_hir_dims or rf.anchor_dims
    dims_text = ", ".join(f"{k}={v}" for k, v in sorted(display_dims.items())) or "default"
    roofline_lines = [
        "### HIR ANALYTICAL ROOFLINE & TRAFFIC FLOOR",
        f"- entry: {packet.entry_name} (resolved_hir_dims: {dims_text})",
        f"- compute_flops: {rf.compute_flops}",
        f"- min_gm_bytes: {rf.min_gm_bytes}",
        f"- arithmetic_intensity: {rf.arithmetic_intensity:.4f}",
        f"- compute_floor_us: {rf.compute_floor_us:.4f}",
        f"- memory_floor_us: {rf.memory_floor_us:.4f}",
        f"- ideal_roofline_us: {rf.ideal_roofline_us:.4f}",
        f"- dominant_bound: {rf.dominant_bound}",
    ]
    if not rf.shape_exact_match:
        roofline_lines.append(f"- requested_anchor_dims: {rf.requested_anchor_dims}")
    if rf.fallback_reason:
        roofline_lines.append(f"- fallback_reason: {rf.fallback_reason}")

    cand_lines = ["### LEGAL ASCEND TIR CANDIDATE PRIMITIVES"]
    if packet.candidate_ops:
        cand_lines.extend(entry.format_for_prompt() for entry in packet.candidate_ops)
    else:
        cand_lines.append("- None extracted")

    blueprint_lines = ["### LOWERED ASCEND TIR BLUEPRINT & Z3 SCHEDULE"]
    if not packet.shape_exact_match:
        blueprint_lines.append(_structural_reference_banner(packet))
        if packet.fallback_reason:
            blueprint_lines.append(f"- fallback_reason: {packet.fallback_reason}")
    z3_text = json.dumps(packet.z3_pipeline_summary, sort_keys=True)
    blueprint_lines.append(f"Z3 Pipeline Summary: {z3_text}")
    if packet.tir_script:
        blueprint_lines.append(f"```python\n{packet.tir_script[:4000]}\n```")
    elif packet.lowering_diagnostics:
        blueprint_lines.append("Lowering Diagnostics:")
        blueprint_lines.extend(f"- {diag}" for diag in packet.lowering_diagnostics)
    else:
        blueprint_lines.append("- No lowered TIR script available.")

    sections = [
        "\n".join(roofline_lines),
        "\n".join(cand_lines),
        "\n".join(blueprint_lines),
    ]

    if current_tl_ledger is not None and packet.tir_ledger is not None:
        tir_led = packet.tir_ledger
        if packet.shape_exact_match:
            tir_ub = _ub_peak_bytes(tir_led)
            tl_ub = _ub_peak_bytes(current_tl_ledger)
            ub_delta = tl_ub - tir_ub
            gm_delta = current_tl_ledger.gm_roundtrip_bytes - tir_led.gm_roundtrip_bytes
            if (
                tir_led.steady_state_ii is not None
                and current_tl_ledger.steady_state_ii is not None
            ):
                ii_delta = str(current_tl_ledger.steady_state_ii - tir_led.steady_state_ii)
            else:
                ii_delta = "n/a"
            delta_lines = [
                "### TIR BLUEPRINT VS CURRENT TILELANG LEDGER DELTA",
                (
                    f"- steady_state_ii: tir={tir_led.steady_state_ii}, "
                    f"current_tl={current_tl_ledger.steady_state_ii}, delta={ii_delta}"
                ),
                (
                    f"- dominant_pipe: tir={tir_led.dominant_pipe}, "
                    f"current_tl={current_tl_ledger.dominant_pipe}"
                ),
                f"- ub_peak_bytes: tir={tir_ub}, current_tl={tl_ub}, delta={ub_delta:+d}",
                (
                    f"- gm_roundtrip_bytes: tir={tir_led.gm_roundtrip_bytes}, "
                    f"current_tl={current_tl_ledger.gm_roundtrip_bytes}, delta={gm_delta:+d}"
                ),
            ]
        else:
            delta_lines = [
                "### TIR BLUEPRINT VS CURRENT TILELANG LEDGER DELTA",
                _structural_reference_banner(packet),
            ]
            if packet.fallback_reason:
                delta_lines.append(f"- fallback_reason: {packet.fallback_reason}")
            delta_lines.append(
                f"- dominant_pipe: tir={tir_led.dominant_pipe}, "
                f"current_tl={current_tl_ledger.dominant_pipe}"
            )
            all_scopes = sorted(set(tir_led.waterlines) | set(current_tl_ledger.waterlines))
            for scope in all_scopes:
                topo = _SCOPE_TOPOLOGY_MAP.get(scope, scope)
                tir_pct = _format_scope_pct(tir_led.waterlines.get(scope))
                tl_pct = _format_scope_pct(current_tl_ledger.waterlines.get(scope))
                delta_lines.append(
                    f"- scope_utilization[{scope} -> {topo}]: "
                    f"tir={tir_pct}, current_tl={tl_pct}"
                )
            if not all_scopes:
                delta_lines.append("- scope_utilization: none")
        if (
            current_tl_ledger.min_gm_dma_bytes > 0
            or current_tl_ledger.uses_dynamic_offset_pipelining
            or current_tl_ledger.has_unlimit_memory_attr
            or current_tl_ledger.has_coupled_reduce_sfu_accum
        ):
            delta_lines.append(
                f"- microarch_flags: min_gm_dma_bytes={current_tl_ledger.min_gm_dma_bytes}, "
                f"dma_burst_eff={current_tl_ledger.dma_burst_efficiency_ratio:.3f}, "
                f"dyn_offset_pipe={current_tl_ledger.uses_dynamic_offset_pipelining}, "
                f"unlimit_mem={current_tl_ledger.has_unlimit_memory_attr}, "
                f"coupled_reduce_sfu={current_tl_ledger.has_coupled_reduce_sfu_accum}"
            )
        sections.append("\n".join(delta_lines))

    return "\n\n".join(sections)


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


def _clone_tensor_tree(value: Any) -> Any:
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


def _flatten_output_tensors(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        out: list[torch.Tensor] = []
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
    actual: torch.Tensor,
    golden: torch.Tensor,
    *,
    threshold: float | None = None,
    precision_thresholds: Mapping[str, float] | None = None,
    native_output: torch.Tensor | None = None,
) -> dict[str, object]:
    """Compare actual vs golden tensor using CANNBench MERE/MARE and threshold==0 bit-exact rules."""
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
    value_range: Any,
    generator: torch.Generator,
    *,
    param_name: str = "",
) -> torch.Tensor | None:
    """Synthesize a faithful input tensor matching proto/cases value_range and Q4_0/special rules."""
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
        norm_name = param_name.strip().lower()
        if "scale" in norm_name:
            lo_raw, hi_raw = 0.01, 0.08
        elif tdtype == torch.uint8:
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
        return float(v)  # type: ignore[arg-type]

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
        # Ensure non-zero weights when synthesizing floating tensors with symmetric zero-mean range
        if param_name.strip().lower() in ("w", "weight", "gamma", "scale", "scales") and vals.numel() > 0:
            zero_mask = vals == 0.0
            if bool(zero_mask.any().item()):
                vals[zero_mask] = 0.05
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


class CannbotContractOracle:
    """Faithful CANNBench contract input synthesizer and MERE/MARE + bit-exact accuracy oracle."""

    def __init__(
        self,
        operator_name: str | None = None,
        *,
        task_dir: str | Path | None = None,
        operators_root: str | Path | None = None,
    ) -> None:
        if task_dir is not None:
            self.task_dir = Path(task_dir).expanduser().resolve()
            self.task = f"{self.task_dir.parent.name}/{self.task_dir.name}"
            self.op_name = self.task_dir.name
        elif operator_name is not None:
            from tilefoundry_ascend.agent.campaign.anchor_specs import CannbotOpsRegistry

            registry = (
                CannbotOpsRegistry(operators_root=operators_root)
                if operators_root is not None
                else CannbotOpsRegistry.default()
            )
            spec = registry.resolve(operator_name)
            self.task_dir = spec.op_dir
            self.task = spec.task
            self.op_name = spec.op_id
        else:
            raise ValueError("Either operator_name or task_dir must be provided")

        proto_raw = yaml.safe_load((self.task_dir / "proto.yaml").read_text(encoding="utf-8")) or {}
        self.proto: dict[str, Any] = dict(proto_raw.get("operator") or proto_raw)
        self.precision_thresholds: dict[str, float] = {
            str(k).lower(): float(v)
            for k, v in (self.proto.get("precision_thresholds") or {}).items()
        }
        cases_raw = yaml.safe_load((self.task_dir / "cases.yaml").read_text(encoding="utf-8")) or {}
        self.cases: list[dict[str, Any]] = [
            dict(row) for row in (cases_raw.get("cases") or ()) if isinstance(row, Mapping)
        ]
        self._cases_by_id: dict[int, dict[str, Any]] = {
            int(c["case_id"]): c for c in self.cases if "case_id" in c
        }

        ref_file = self.task_dir / "reference.py"
        if not ref_file.is_file():
            ref_file = self.task_dir / "golden.py"
        if not ref_file.is_file():
            raise FileNotFoundError(f"Neither reference.py nor golden.py found in {self.task_dir}")
        mod_spec = importlib.util.spec_from_file_location(
            f"_cannbot_oracle_{self.task.replace('/', '_')}", str(ref_file)
        )
        if mod_spec is None or mod_spec.loader is None:
            raise RuntimeError(f"Failed to load reference module {ref_file}")
        mod = importlib.util.module_from_spec(mod_spec)
        mod_spec.loader.exec_module(mod)
        self.reference_module = mod
        self.reference_fn = getattr(mod, self.op_name, None)
        if not callable(self.reference_fn):
            # Fallback to first public function matching op schema
            for name, obj in vars(mod).items():
                if not name.startswith("_") and name != "get_input" and callable(obj):
                    self.reference_fn = obj
                    break
        if not callable(self.reference_fn):
            raise AttributeError(f"No reference callable found in {ref_file}")
        self.get_input = getattr(mod, "get_input", None)

    def get_case(self, case_num: int) -> dict[str, Any]:
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
    ) -> tuple[list[Any], dict[str, Any]]:
        """Synthesize faithful CPU tensors and call kwargs for *case_num*."""
        case = self.get_case(case_num)
        gen = torch.Generator(device="cpu")
        gen.manual_seed(self._deterministic_seed(case_num, seed))

        raw_shapes = case.get("input_shape") or []
        if raw_shapes and isinstance(raw_shapes[0], int):
            shapes: list[Any] = [raw_shapes]
        else:
            shapes = list(raw_shapes)

        raw_dtypes = case.get("dtype") or ["float32"]
        if isinstance(raw_dtypes, str):
            dtypes: list[Any] = [raw_dtypes]
        else:
            dtypes = list(raw_dtypes)
        if len(dtypes) == 1 and len(shapes) > 1:
            dtypes = dtypes * len(shapes)

        raw_vr = case.get("value_range")
        if not raw_vr:
            value_ranges: list[Any] = [None] * len(shapes)
        elif not isinstance(raw_vr[0], (list, tuple)):
            value_ranges = [raw_vr] * len(shapes)
        else:
            value_ranges = list(raw_vr)
            if len(value_ranges) < len(shapes):
                value_ranges.extend([value_ranges[-1]] * (len(shapes) - len(value_ranges)))

        proto_inputs = self.proto.get("inputs") or []
        tensors: list[Any] = []
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
                sub_list: list[Any] = []
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
                tensors.append(
                    _synthesize_contract_tensor(
                        sh, str(dt), vr, gen, param_name=pname
                    )
                )

        attrs = dict(case.get("attrs") or {})
        if callable(self.get_input):
            gi_params = {
                str(item.get("name", f"arg_{i}")): tensors[i] if i < len(tensors) else None
                for i, item in enumerate(proto_inputs)
                if isinstance(item, Mapping)
            }
            for k, v in attrs.items():
                gi_params.setdefault(str(k), v)
            gi_params.setdefault("skip2_exist", True)
            sig_gi = inspect.signature(self.get_input)
            if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig_gi.parameters.values()):
                tensors = list(self.get_input(**gi_params))
            else:
                filtered_gi = {k: v for k, v in gi_params.items() if k in sig_gi.parameters}
                tensors = list(self.get_input(**filtered_gi))

        call_kwargs = self._build_call_params(self.reference_fn, case, tensors)
        return tensors, call_kwargs

    def _build_call_params(
        self,
        fn: Callable[..., Any],
        case: Mapping[str, Any],
        tensors: Sequence[Any],
    ) -> dict[str, Any]:
        sig = inspect.signature(fn)
        params: dict[str, Any] = {}
        tensor_idx = 0
        attrs = dict(case.get("attrs") or {})
        proto_inputs = self.proto.get("inputs") or []
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
        raw_tensors: Sequence[Any] | None = None,
        *,
        seed: int | None = None,
        native: bool = False,
    ) -> Any:
        """Compute cloned, aliasing-safe oracle reference output for *case_num*."""
        if raw_tensors is None:
            raw_tensors, _ = self.synthesize_case_inputs(case_num, seed=seed)
        case = self.get_case(case_num)
        cloned = _clone_tensor_tree(list(raw_tensors))
        suffix = "_bench" if native else "_oracle"
        fn = getattr(self.reference_module, f"{self.op_name}{suffix}", self.reference_fn)
        if not native:
            def _promote_double(val: Any) -> Any:
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
                cloned = _clone_tensor_tree(list(raw_tensors))
        with torch.no_grad():
            return fn(**self._build_call_params(fn, case, cloned))

    def verify_candidate(
        self,
        candidate_fn: Callable[..., Any],
        *,
        case_nums: Sequence[int] | None = None,
        seed: int | None = None,
    ) -> tuple[bool, list[dict[str, Any]]]:
        """Validate candidate callable against oracle across selected cases."""
        targets = (
            [int(c) for c in case_nums]
            if case_nums is not None
            else [int(c["case_id"]) for c in self.cases]
        )
        proto_outputs = self.proto.get("outputs") or []
        case_reports: list[dict[str, Any]] = []
        all_passed = True

        for cid in targets:
            case = self.get_case(cid)
            raw_tensors, _ = self.synthesize_case_inputs(cid, seed=seed)
            oracle_out = self.reference_output(cid, raw_tensors, seed=seed, native=False)
            cand_tensors = _clone_tensor_tree(raw_tensors)
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

            out_details: list[dict[str, Any]] = []
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
    "CannbotContractOracle",
    "build_tilelang_runtime_twin",
    "compare_cannbench_tensor",
    "format_hir_tir_navigation_section",
    "verify_twin_cpu_reference",
]
