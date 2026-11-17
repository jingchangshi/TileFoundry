"""Runtime twin bridge and HIR/TIR navigation prompt formatting for Ascend agents."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import inspect
import json
from pathlib import Path
import runpy
import tempfile

import torch

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
) -> torch.Tensor:
    """Create a deterministic CPU tensor matching *ttype* shape and dtype."""
    shape = _concrete_shape(ttype, test_dims)
    target_dtype = to_torch_dtype(ttype.dtype)
    if target_dtype.is_floating_point:
        return torch.randn(shape, generator=generator, dtype=torch.float32).to(target_dtype)
    if target_dtype == torch.bool:
        return torch.randint(0, 2, shape, generator=generator, dtype=torch.int64).to(target_dtype)
    return torch.randint(0, 4, shape, generator=generator, dtype=target_dtype)


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
) -> tuple[bool, str]:
    """Compare *candidate_runner* via a @runtime_module twin against CPU HIR evaluation."""
    try:
        with tempfile.TemporaryDirectory(prefix="tf_twin_ref_") as tmp_dir:
            authored_mod = _load_authored_module(
                authored_hir_source,
                Path(tmp_dir),
                test_dims=test_dims,
            )
        checked_mod, checked_fn = _check_module_dims(authored_mod, test_dims=test_dims)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(0)
        inputs: list[torch.Tensor] = []
        for param in checked_fn.params:
            if not isinstance(param.type, TensorType):
                return (
                    False,
                    f"twin_unsupported_param_type: {param.name} has {type(param.type).__name__}",
                )
            inputs.append(
                _synthesize_seeded_tensor(param.type, generator, test_dims=test_dims)
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
    except Exception as exc:
        return False, f"twin_execution_error: {type(exc).__name__}: {exc}"

    ok, detail = _compare_outputs(ref_out, cand_out, rtol=rtol, atol=atol)
    if not ok:
        return False, detail
    return True, f"twin_cpu_reference_passed: entry={checked_fn.name}, {detail}"


def _ub_peak_bytes(ledger: ObjectivePhysicalLedger) -> int:
    """Return peak allocated UB bytes from a physical ledger."""
    for key in ("UB", "ub", "smem"):
        waterline = ledger.waterlines.get(key)
        if waterline is not None:
            return int(waterline.peak_allocated_bytes)
    return 0


def format_hir_tir_navigation_section(
    packet: HirTirNavigationPacket,
    *,
    current_tl_ledger: ObjectivePhysicalLedger | None = None,
) -> str:
    """Format HIR roofline, legal TIR ops, Z3 schedule, and ledger delta sections."""
    rf = packet.roofline
    dims_text = ", ".join(f"{k}={v}" for k, v in sorted(rf.anchor_dims.items())) or "default"
    roofline_lines = [
        "### HIR ANALYTICAL ROOFLINE & TRAFFIC FLOOR",
        f"- entry: {packet.entry_name} (anchor_dims: {dims_text})",
        f"- compute_flops: {rf.compute_flops}",
        f"- min_gm_bytes: {rf.min_gm_bytes}",
        f"- arithmetic_intensity: {rf.arithmetic_intensity:.4f}",
        f"- compute_floor_us: {rf.compute_floor_us:.4f}",
        f"- memory_floor_us: {rf.memory_floor_us:.4f}",
        f"- ideal_roofline_us: {rf.ideal_roofline_us:.4f}",
        f"- dominant_bound: {rf.dominant_bound}",
    ]

    cand_lines = ["### LEGAL ASCEND TIR CANDIDATE PRIMITIVES"]
    if packet.candidate_ops:
        cand_lines.extend(entry.format_for_prompt() for entry in packet.candidate_ops)
    else:
        cand_lines.append("- None extracted")

    blueprint_lines = ["### LOWERED ASCEND TIR BLUEPRINT & Z3 SCHEDULE"]
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
        tir_ub = _ub_peak_bytes(tir_led)
        tl_ub = _ub_peak_bytes(current_tl_ledger)
        ub_delta = tl_ub - tir_ub
        gm_delta = current_tl_ledger.gm_roundtrip_bytes - tir_led.gm_roundtrip_bytes
        if tir_led.steady_state_ii is not None and current_tl_ledger.steady_state_ii is not None:
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
        sections.append("\n".join(delta_lines))

    return "\n\n".join(sections)


__all__ = [
    "build_tilelang_runtime_twin",
    "format_hir_tir_navigation_section",
    "verify_twin_cpu_reference",
]
