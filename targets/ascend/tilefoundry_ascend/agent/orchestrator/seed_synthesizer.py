"""Stage-0 semantic seed synthesis from PyTorch golden references and workload contracts."""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Protocol

from tilefoundry_ascend.agent.kb.schema import normalize_runtime_backend
from tilefoundry_ascend.agent.orchestrator.llm_client import (
    GenerationRequest,
    KernelLLMBackend,
    extract_python_dsl,
)
from tilefoundry_ascend.agent.orchestrator.llm_constraints import (
    build_progressive_llm_constraints,
    format_g5_parent_monotonicity_lock,
)
from tilefoundry_ascend.agent.specs.anchor_specs import (
    RegimeTargetSpec,
    build_semantic_regime_specs,
    discover_module_trackb_symbols,
)
from tilefoundry_ascend.ascend.passes.hir_tir_bridge import HirBlueprintPacket
from tilefoundry_ascend.ascend.sil.pytorch_frontend import (
    PyTorchOpSemantics,
    PyTorchSemanticAnalyzer,
    RepresentativeCase,
)
from tilefoundry_ascend.ascend.sil.schema import CasePhysicsRatios, ConstraintRelaxationMask
from tilefoundry_ascend.ascend.target.dialect_surface import (
    DialectSurfaceSpec,
    get_dialect_surface,
)

logger = logging.getLogger(__name__)
SeedMode = Literal["cold_start", "incumbent_takeover"]
SeedVerificationState = Literal["l3_verified", "unverified_incumbent"]

_SCALAR_TYPE_MAP = {
    "int": "int", "int32": "int", "int64": "int",
    "float": "float", "float32": "float", "float64": "float",
    "bool": "bool", "str": "str", "string": "str",
    "listint": "list[int]", "list[int]": "list[int]",
    "listfloat": "list[float]", "list[float]": "list[float]", "list": "list",
}
_SYNTHETIC_DIM_ALIAS_RE = re.compile(r"^(?:total_elements|shape_\d+_dim_\d+|.+_dim_\d+)$")
_REDUNDANT_CASE_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("d_dim", ("D",)), ("dk", ("D", "d_dim")), ("half_dim", ("D", "d_dim")),
    ("b_dim", ("B",)), ("s_dim", ("S",)), ("m_rows", ("M",)), ("n_q", ("N_q",)),
    ("n_kv", ("N_kv",)), ("s_kv", ("S_kv",)), ("t_dim", ("S", "s_dim")),
)


@dataclass(frozen=True)
class OracleAccuracyResult:
    """Signed L3 accuracy evidence for one Stage-0 seed candidate."""

    passed: bool
    error_message: str = ""
    candidate_sha256: str = ""
    l3_run_id: str = ""
    l3_receipt_path: str = ""
    l3_state: str = ""
    elapsed_sec: float = 0.0
    failure_kind: str = "candidate"


@dataclass(frozen=True)
class SeedSynthesisOutcome:
    """Result of Stage-0 correctness seed synthesis or incumbent takeover preparation."""

    mode: SeedMode
    initial_track: str
    module_source: str
    semantics: PyTorchOpSemantics
    case_ratios: tuple[CasePhysicsRatios, ...]
    regime_specs: tuple[RegimeTargetSpec, ...]
    blueprints_by_regime: dict[str, HirBlueprintPacket]
    attempts_used: int
    verification_state: SeedVerificationState
    l0_l1_passed: bool
    oracle_accuracy_passed: bool
    candidate_sha256: str
    l3_run_id: str
    l3_receipt_path: str
    l3_state: str
    l3_elapsed_sec: float


@dataclass(frozen=True)
class GoldenParamSpec:
    """Normalized parameter specification extracted from golden.py and proto.yaml."""

    name: str
    annotation: str
    default_repr: str | None = None
    has_default: bool = False
    is_tensor: bool = False
    is_tensor_list: bool = False
    is_optional: bool = False


@dataclass(frozen=True)
class GoldenEntrypointContract:
    """Typed public entrypoint signature and concise mathematical specification."""

    entrypoint: str
    parameters: tuple[GoldenParamSpec, ...]
    return_annotation: str
    signature_line: str
    math_summary: str = ""
    output_names: tuple[str, ...] = ()
    output_dtypes: tuple[str, ...] = ()


class Stage0StaticValidatorFactory(Protocol):
    """Typed factory building a static Stage-0 validator for campaign regimes."""

    def __call__(
        self, regime_specs: tuple[RegimeTargetSpec, ...], *, semantics: PyTorchOpSemantics | None = None
    ) -> Callable[[str, str], tuple[bool, str]]: ...


def _sha256_text(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def select_fallback_track_from_semantics(
    semantics: PyTorchOpSemantics, ratios: Sequence[CasePhysicsRatios] = (),
    *, blueprints: Mapping[str, HirBlueprintPacket] | Sequence[HirBlueprintPacket] | None = None,
) -> str:
    """Select concrete Track-B CCE C++20 fallback track when Track-A lowering stalls."""
    if any(r.case_id < 1 for r in ratios):
        raise ValueError("case ratios must carry positive case_id")
    if semantics.base_axes.uses_cube or semantics.base_axes.compute_engine == "coupled_1c2v":
        return "hybrid_trackb"
    if blueprints is not None:
        packets = blueprints.values() if isinstance(blueprints, Mapping) else blueprints
        if any(p.roofline.cube_flops > 0 for p in packets):
            return "hybrid_trackb"
    return "cce_trackb"


def select_initial_track_from_semantics(
    semantics: PyTorchOpSemantics, ratios: Sequence[CasePhysicsRatios] = (),
    *, blueprints: Mapping[str, HirBlueprintPacket] | Sequence[HirBlueprintPacket] | None = None,
) -> str:
    """Prefer Track-A TileLang DSL initially unless semantics require direct Track-B indexing."""
    if any(r.case_id < 1 for r in ratios):
        raise ValueError("case ratios must carry positive case_id")
    if blueprints is not None:
        packets = blueprints.values() if isinstance(blueprints, Mapping) else blueprints
        for packet in packets:
            if not packet.entry_name.strip():
                raise ValueError("blueprint packet must carry a non-empty entry_name")
    return "cce_trackb" if (semantics.has_subbyte_or_block_quant or semantics.has_indirect_index) else "tilelang_dsl"


def _canonical_seed_track(initial_track: str) -> tuple[str, str]:
    backend = normalize_runtime_backend(initial_track)
    if backend == "tilelang_dsl":
        return "tilelang_dsl", "tilelang_dsl"
    if backend in ("ascendc_trackb", "hybrid_trackb"):
        return backend, "cce_cpp20"
    raise RuntimeError(f"unsupported Stage-0 seed track {initial_track!r}")


def _unwrap_proto_operator(proto: Mapping[str, object] | None) -> Mapping[str, object]:
    if not isinstance(proto, Mapping):
        return {}
    inner = proto.get("operator")
    if isinstance(inner, Mapping) and any(k in inner for k in ("inputs", "outputs", "attrs", "formula")):
        return inner
    return proto


def _normalize_scalar_annotation(raw_type: str) -> str:
    return _SCALAR_TYPE_MAP.get(raw_type.strip().lower(), raw_type.strip() or "object")


def _normalize_param_annotation(
    raw_ann: str, *, has_default: bool, default_is_none: bool,
    proto_input: Mapping[str, object] | None, proto_attr: Mapping[str, object] | None,
) -> tuple[str, bool, bool, bool]:
    cleaned = raw_ann.strip()
    is_opt = (
        default_is_none or "Optional[" in cleaned or "| None" in cleaned
        or "None |" in cleaned or bool((proto_input or {}).get("optional", False))
    )
    if "Tensor" in cleaned or (proto_input is not None and not cleaned):
        src = cleaned if "Tensor" in cleaned else str((proto_input or {}).get("shape", ""))
        is_list = "List[" in src or "list[" in src
        base = "List[torch.Tensor]" if is_list else "torch.Tensor"
        return (f"Optional[{base}]" if is_opt else base), True, is_list, is_opt
    if cleaned:
        if cleaned.startswith("Optional[") and cleaned.endswith("]"):
            return f"Optional[{_normalize_scalar_annotation(cleaned[9:-1])}]", False, False, True
        norm = _normalize_scalar_annotation(cleaned)
        return (f"Optional[{norm}]", False, False, True) if (default_is_none and not has_default) else (norm, False, False, is_opt)
    if proto_attr is not None and (attr_t := str(proto_attr.get("type", "")).strip()):
        norm = _normalize_scalar_annotation(attr_t)
        return (f"Optional[{norm}]", False, False, True) if default_is_none else (norm, False, False, is_opt)
    return "object", False, False, is_opt


def _extract_math_summary(proto: Mapping[str, object], desc_text: str) -> str:
    parts: list[str] = []
    if desc := str(proto.get("description", "")).strip():
        parts.append(desc)
    if formula := str(proto.get("formula", "")).strip():
        if compact := " ".join(ln.strip() for ln in formula.splitlines() if ln.strip()):
            parts.append(f"Formula: {compact[:220]}")
    if desc_text.strip():
        lines = [
            re.sub(r"^#+\s*", "", ln).strip()
            for ln in desc_text.splitlines()
            if ln.strip() and not ln.strip().startswith("---") and "Copyright" not in ln and "License" not in ln
        ]
        if lines:
            parts.append(" ".join(lines)[:240])
    return " | ".join(dict.fromkeys(p for p in parts if p))[:420]


def _collect_func_params(fn_node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[tuple[ast.arg, ast.expr | None]]:
    pos_args = [*fn_node.args.posonlyargs, *fn_node.args.args]
    pos_defs: list[ast.expr | None] = [None] * (len(pos_args) - len(fn_node.args.defaults)) + list(fn_node.args.defaults)
    return list(zip(pos_args, pos_defs)) + list(zip(fn_node.args.kwonlyargs, fn_node.args.kw_defaults))


def extract_golden_entrypoint_contract(
    golden_source: str, entrypoint: str, *, proto: Mapping[str, object] | None = None, desc_text: str = ""
) -> GoldenEntrypointContract:
    """Extract typed entrypoint parameters, defaults, return type, and math summary."""
    op_proto = _unwrap_proto_operator(proto)
    proto_inputs = {str(r["name"]): r for r in (op_proto.get("inputs") or ()) if isinstance(r, Mapping) and "name" in r}
    proto_attrs = {str(r["name"]): r for r in (op_proto.get("attrs") or ()) if isinstance(r, Mapping) and "name" in r}
    raw_outputs = [r for r in (op_proto.get("outputs") or ()) if isinstance(r, Mapping)]
    output_names = tuple(str(r.get("name", "")) for r in raw_outputs if r.get("name"))
    out_dtypes: list[str] = []
    for row in raw_outputs:
        if isinstance(raw_dt := row.get("dtype"), str) and raw_dt.strip():
            out_dtypes.append(raw_dt.strip())
        elif isinstance(raw_dt, Sequence) and not isinstance(raw_dt, (str, bytes)):
            if cands := [str(i).strip() for i in raw_dt if i is not None]:
                out_dtypes.append("/".join(cands))
    ret_ann = "tuple[torch.Tensor, ...]" if len(raw_outputs) > 1 else "torch.Tensor"

    fn_node: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    if golden_source.strip():
        try:
            top_defs = [n for n in ast.parse(golden_source).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            fn_node = next((n for n in top_defs if n.name == entrypoint), top_defs[0] if len(top_defs) == 1 else None)
        except SyntaxError:
            pass

    params: list[GoldenParamSpec] = []
    if fn_node is not None:
        for arg, d_node in _collect_func_params(fn_node):
            d_repr = ast.unparse(d_node) if d_node is not None else None
            ann, is_t, is_tl, is_opt = _normalize_param_annotation(
                ast.unparse(arg.annotation) if arg.annotation is not None else "",
                has_default=d_node is not None, default_is_none=(d_repr == "None"),
                proto_input=proto_inputs.get(arg.arg), proto_attr=proto_attrs.get(arg.arg),
            )
            params.append(GoldenParamSpec(arg.arg, ann, d_repr, d_node is not None, is_t, is_tl, is_opt))
        if fn_node.returns is not None and (raw_ret := ast.unparse(fn_node.returns).strip()):
            ret_ann = raw_ret
    else:
        for row in (op_proto.get("inputs") or ()):
            if isinstance(row, Mapping) and "name" in row:
                is_opt = bool(row.get("optional", False))
                ann, is_t, is_tl, _ = _normalize_param_annotation("", has_default=is_opt, default_is_none=is_opt, proto_input=row, proto_attr=None)
                params.append(GoldenParamSpec(str(row["name"]), ann, "None" if is_opt else None, is_opt, is_t, is_tl, is_opt))
        for row in (op_proto.get("attrs") or ()):
            if isinstance(row, Mapping) and "name" in row:
                has_def = "default" in row
                ann, _, _, is_opt = _normalize_param_annotation("", has_default=has_def, default_is_none=has_def and row["default"] is None, proto_input=None, proto_attr=row)
                params.append(GoldenParamSpec(str(row["name"]), ann, repr(row["default"]) if has_def else None, has_def, False, False, is_opt))

    rendered = ", ".join(
        f"{p.name}: {p.annotation} = {p.default_repr}" if (p.has_default and p.default_repr is not None) else f"{p.name}: {p.annotation}"
        for p in params
    )
    return GoldenEntrypointContract(
        entrypoint=entrypoint, parameters=tuple(params), return_annotation=ret_ann,
        signature_line=f"def {entrypoint}({rendered}) -> {ret_ann}:",
        math_summary=_extract_math_summary(op_proto, desc_text),
        output_names=output_names, output_dtypes=tuple(out_dtypes),
    )


def validate_entrypoint_signature_compatibility(module_source: str, contract: GoldenEntrypointContract) -> str:
    """Return an error when candidate entrypoint breaks ParamBuilder tensor/param ABI."""
    if not contract.parameters:
        return ""
    try:
        tree = ast.parse(module_source)
    except SyntaxError as exc:
        return f"seed module syntax error: {exc}"
    fn_node = next(
        (n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == contract.entrypoint), None
    )
    if fn_node is None:
        return f"seed module does not define the public entrypoint {contract.entrypoint}"
    cand_args = _collect_func_params(fn_node)
    expected_names, cand_names = [p.name for p in contract.parameters], [a.arg for a, _ in cand_args]
    if cand_names[: len(expected_names)] != expected_names:
        return f"entrypoint {contract.entrypoint!r} parameter signature mismatch for ParamBuilder: expected {expected_names}, got {cand_names}"
    for extra_arg, extra_def in cand_args[len(expected_names) :]:
        if extra_def is None:
            return f"entrypoint {contract.entrypoint!r} adds required parameter {extra_arg.arg!r} absent from golden signature"
    for param, (cand_arg, cand_def) in zip(contract.parameters, cand_args):
        cand_ann = ast.unparse(cand_arg.annotation) if cand_arg.annotation is not None else ""
        if param.is_tensor:
            if "Tensor" not in cand_ann:
                return f"entrypoint {contract.entrypoint!r} parameter {param.name!r} must carry a Tensor annotation ({param.annotation}) so ParamBuilder binds it as a tensor"
            if param.is_tensor_list and ("List[" not in cand_ann and "list[" not in cand_ann):
                return f"entrypoint {contract.entrypoint!r} parameter {param.name!r} must carry a List[torch.Tensor] annotation ({param.annotation})"
            if param.is_optional and cand_def is None and "Optional" not in cand_ann and "None" not in cand_ann:
                return f"entrypoint {contract.entrypoint!r} optional tensor {param.name!r} must preserve Optional/default None"
        else:
            if "Tensor" in cand_ann:
                return f"entrypoint {contract.entrypoint!r} scalar parameter {param.name!r} must not carry a Tensor annotation"
            if param.has_default and cand_def is None:
                return f"entrypoint {contract.entrypoint!r} parameter {param.name!r} must preserve default value ({param.default_repr})"
    return ""


def format_golden_entrypoint_prompt_block(contract: GoldenEntrypointContract) -> str:
    """Format exact entrypoint ABI and math contract for Stage-0 and optimization prompts."""
    lines = [
        "### EXACT GOLDEN ENTRYPOINT ABI (ParamBuilder-checked)",
        "```python", contract.signature_line, "```",
        "- Keep parameter names, order, defaults, and `torch.Tensor` / `Optional[torch.Tensor]` / `List[torch.Tensor]` annotations exact.",
    ]
    if contract.math_summary:
        lines.append(f"- Mathematical specification: {contract.math_summary}")
    if contract.output_names or contract.output_dtypes:
        lines.append(f"- Declared outputs: names={list(contract.output_names)} dtypes={list(contract.output_dtypes)}")
    return "\n".join(lines) + "\n\n"


def _prompt_type_name(value: object) -> str:
    if value is None:
        return "None"
    for py_t, name in ((bool, "bool"), (int, "int"), (float, "float"), (str, "str")):
        if isinstance(value, py_t):
            return name
    if isinstance(value, tuple):
        return f"tuple[{_prompt_type_name(value[0])}, ...]" if value else "tuple"
    if isinstance(value, list):
        return f"list[{_prompt_type_name(value[0])}]" if value else "list"
    raise ValueError(f"unsupported regime instantiation value {value!r} for the seed ABI prompt")


def _seed_parameter_specs(regime_specs: Sequence[RegimeTargetSpec]) -> tuple[tuple[str, object], ...]:
    merged: dict[str, object] = {}
    for regime in regime_specs:
        for key in sorted(regime.sample_builder_kwargs):
            val = regime.sample_builder_kwargs[key]
            if not (isinstance(val, (list, tuple)) and len(val) > 8) and key not in merged:
                merged[key] = val
    has_canon = any(not _SYNTHETIC_DIM_ALIAS_RE.match(k) for k in merged)
    filtered = {k: v for k, v in merged.items() if not _SYNTHETIC_DIM_ALIAS_RE.match(k)} if has_canon else dict(merged)
    for alias, parents in _REDUNDANT_CASE_ALIASES:
        if alias in filtered and any(p in filtered for p in parents) and len(filtered) > 6:
            filtered.pop(alias, None)
    return tuple(filtered.items())


def _render_signature(name: str, parameters: Sequence[tuple[str, object]], return_type: str) -> str:
    items = [f"{k}: {_prompt_type_name(v)} = {v!r}" for k, v in parameters] + ["**extra_dims: object"]
    return f"def {name}({', '.join(items)}) -> {return_type}:"


def _seed_module_abi(
    entrypoint: str, parameters: Sequence[tuple[str, object]], surface: DialectSurfaceSpec, track: str,
    *, golden_contract: GoldenEntrypointContract | None = None, regime_specs: Sequence[RegimeTargetSpec] = (),
) -> str:
    selector = _render_signature(f"_select_{entrypoint}_config", parameters, "dict[str, object]")
    builder = _render_signature(f"_build_{entrypoint}_prim_func", parameters, "object")
    get_kernel = _render_signature("_get_kernel", parameters, "object")
    dispatch = golden_contract.signature_line if (golden_contract and golden_contract.signature_line) else f"def {entrypoint}(...) -> object:"
    per_regime: list[str] = []
    if len(regime_specs) > 1:
        for reg in regime_specs:
            rp = _seed_parameter_specs((reg,))
            per_regime.extend((
                _render_signature(f"_select_{entrypoint}_{reg.regime_tag}_config", rp, "dict[str, object]"),
                _render_signature(f"_build_{entrypoint}_{reg.regime_tag}_prim_func", rp, "object"),
            ))
    multi_note = (
        f"- Multiple regimes detected: emit per-regime `_select_{entrypoint}_<tag>_config` and "
        f"`_build_{entrypoint}_<tag>_prim_func` functions so each regime edits orthogonal symbols, "
        f"and dispatch across them from `_select_{entrypoint}_config` / `{entrypoint}`.\n"
        if len(regime_specs) > 1 else ""
    )

    if track in ("cce_trackb", "hybrid_trackb"):
        header_list = ", ".join(f"`{h}`" for h in surface.required_headers_or_imports)
        lead_hdr = surface.required_headers_or_imports[0]
        forbidden = ", ".join(f"`{t}`" for t in surface.forbidden_tokens)
        sym_block = [*per_regime, selector, builder, get_kernel, "def _render_trackb_tu(**dims: object) -> str:", dispatch, f"{entrypoint.upper()}_TRACKB_CU: str", f"kernel = {entrypoint}"]
        hybrid_note = (
            '- `hybrid_trackb`: pair Track-B C++20 translation units with `@T.prim_func` builders compiled via `tilelang.compile(prog, target="ascend")` when splitting Cube/Vector/prologue/combine stages.\n'
            if track == "hybrid_trackb" else ""
        )
        return (
            "Required module symbols and exact Stage-0 instantiations:\n```python\n" + "\n".join(sym_block) + "\n```\n"
            f"- `{entrypoint.upper()}_TRACKB_CU` is a top-level C++20 translation-unit string constant (`{entrypoint.upper()}_TRACKB_CU = r\"\"\"...\"\"\"`, plain raw triple-quoted literal rather than `dedent(...)`) using only verified headers from {header_list} (starting with `{lead_hdr}`) that bisheng compiles standalone.\n"
            f"- Ascend 950PR (`dav-3510`) bisheng kernel attribute rule: explicitly annotate every `__global__` device kernel with `__vector__` (vector-only), `__cube__` (cube-only), or `__mix__(1, 2)` (mixed Cube+Vector), e.g. `__global__ __aicore__ __vector__ void {entrypoint}_k(...)`, and expose `extern \"C\" int launch_{entrypoint}(void* stream, ...)` invoking `<<<num_cores, nullptr, stream>>>(...)`.\n"
            "- `_render_trackb_tu` accepts the regime keyword arguments (plus `trackb_src`) as `**dims`, substitutes every shape/constexpr dimension into the C++20 source, and returns the complete instantiated top-level translation unit for those dims. It is called once per campaign regime before L0/L1, so every returned TU must compile standalone and contain exactly one launcher `<<<...>>>` call per active kernel.\n"
            f"- `_select_{entrypoint}_config` is called with the regime keyword arguments above and returns `pass_configs`, `compile_flags`, or `builder_kwargs` overrides.\n"
            f"- `_build_{entrypoint}_prim_func` is called with the same keyword arguments plus any selector `builder_kwargs`, and returns the Track-B kernel plan/translation-unit selection that `_get_kernel` compiles.\n"
            "- `_get_kernel` compiles and caches the Track-B shared library via:\n"
            "  - `from tilelang.contrib.bisheng import compile_ascend` (and import `PURE_SIMDVF_UBUF_FLAGS` from `tilefoundry_ascend.ascend.target` ONLY for pure `__simd_vf__` kernels without `__simt_vf__`)\n"
            '  - include flags: `-I{Path(tilelang.__file__).resolve().parent.parent / "src"}` plus existing CANN include directories (`include`, `x86_64-linux/include`, `aarch64-linux/include`, `runtime/include` under `ASCEND_HOME_PATH` or `ASCEND_TOOLKIT_HOME`); include `PURE_SIMDVF_UBUF_FLAGS` ONLY when the TU defines zero `__simt_vf__` functions (whenever `__simt_vf__` is present, omit `PURE_SIMDVF_UBUF_FLAGS` and never cast raw integer offsets to `__ubuf__ T*`, which overwrites the SIMT stack/data cache; use `__gm__` pointers or a `__gm__` scratch tensor instead)\n'
            '  - atomic `.so` caching under `cache_dir = Path(os.environ.get("TILELANG_CACHE_DIR") or tempfile.gettempdir()) / "tf_ascend_so_cache"`: compute persistent `(tu_sha256, compile_flags)` content-hashed `so_path = cache_dir / f"...so"`; if `not so_path.exists()`, compile into PID-staged `tmp_so = cache_dir / f"{so_path.stem}.{os.getpid()}.tmp.so"` via `compile_ascend(tu_src, target_format="so", npu_arch="dav-3510", options=opts, path_target=tmp_so)` and commit atomically with `os.replace(tmp_so, so_path)` (never key `_KERNEL_CACHE` or `.so` filenames on volatile per-case runtime extents such as `total_elements`, `m_rows`, `b_dim`, `s_dim`, `s_kv` unless quantized to a discrete bucket tag; reuse any existing `.so` across L3 subprocesses), append `str(so_path)` to `_COMPILED_OBJ_PATHS: list[str]`, load with `lib = ctypes.CDLL(str(so_path))` (retain `lib` on the cached runner to prevent GC), and pass `torch.npu.current_stream().npu_stream` alongside contiguous tensor `data_ptr()` (`ctypes.c_void_p`) values.\n'
            "- Track-B architecture preference: for regular dense elementwise, reduction, norm, and softmax workloads prefer UB-staged vectorized `__simd_vf__` (`asc_copy_gm2ub_align` + `simd_inst::*` + `asc_copy_ub2gm_align`) over scalar element-by-element `__gm__` loops inside `__simt_vf__`; for matmul and attention workloads prefer Cube (`asc_mmad` / `T.gemm`) + Vector tiling over scalar nested `__gm__` loops; reserve `__simt_vf__` for irregular gather/scatter/dynamic-index access patterns.\n"
            f"- `{entrypoint}` dispatches NPU tensors through the bisheng-compiled Track-B library; no torch/CPU fallback may remain on the NPU path.\n"
            f"{hybrid_note}{multi_note}- The module must not contain any forbidden Track-B symbol: {forbidden}.\n"
        )

    sym_block = [*per_regime, selector, builder, get_kernel, dispatch, f"kernel = {entrypoint}"]
    return (
        "Required module symbols and exact Stage-0 instantiations:\n```python\n" + "\n".join(sym_block) + "\n```\n"
        f"- `_select_{entrypoint}_config` is called with the regime keyword arguments above and returns optional `pass_configs`, `compile_flags`, and `builder_kwargs`.\n"
        f"- `_build_{entrypoint}_prim_func` is called with the same keyword arguments plus any selector `builder_kwargs`, and returns an inner `@T.prim_func`.\n"
        '- `_get_kernel` compiles that prim_func through `tilelang.compile(prog, target="ascend")` and caches the compiled callable, strictly excluding per-case volatile runtime extents (`total_elements`, `m_rows`, `b_dim`, `s_dim`, `s_kv`) from `_KERNEL_CACHE` / disk keys unless bucketed to discrete tags (use `T.dynamic(...)` inside `@T.prim_func` so all 20 cases share one compiled binary).\n'
        "- For matmul or coupled Cube+Vector operators use `with T.MixedKernel(28, sids=2) as (bx, sid):` with `T.alloc_l1`, `T.alloc_l0c`, `T.gemm`, `T.alloc_shared`, `T.SimdVF`.\n"
        "- For non-last-axis or high-rank reductions reshape/view on NPU as 3D `(outer, reduce_extent, inner)` and pad UB shared tiles to 64-element multiples with `T.simd.update_mask(rem, width=32/16)` plus explicit `-inf`/`NaN` row guards.\n"
        f"- `{entrypoint}` performs the operator on NPU tensors through `_get_kernel`; no torch/CPU fallback may remain on the NPU path.\n{multi_note}"
    )


def _compact_proto_for_prompt(proto: Mapping[str, object]) -> dict[str, object]:
    out: dict[str, object] = {}
    for k, v in proto.items():
        ks = str(k)
        if ks in ("inputs", "outputs", "attrs") and isinstance(v, (list, tuple)):
            out[ks] = [{str(ek): ev for ek, ev in e.items() if str(ek) != "description"} if isinstance(e, Mapping) else e for e in v]
        elif isinstance(v, Mapping):
            out[ks] = _compact_proto_for_prompt(v)
        else:
            out[ks] = v
    return out


def _compact_case_for_prompt(case: Mapping[str, object]) -> dict[str, object]:
    out: dict[str, object] = {}
    for k, v in case.items():
        if (ks := str(k)) in ("note", "operator", "value_range"):
            continue
        if ks == "dtype" and isinstance(v, (list, tuple)) and v and all(isinstance(i, str) for i in v) and len(set(v)) == 1:
            out[ks] = v[0]
        else:
            out[ks] = v
    return out


def _official_contract_section(
    proto: Mapping[str, object], cases: Sequence[Mapping[str, object]], regime_specs: Sequence[RegimeTargetSpec], *, entrypoint: str = ""
) -> str:
    lines = [
        "### OFFICIAL PROTO CONTRACT", "```json",
        json.dumps(_compact_proto_for_prompt(proto), sort_keys=True, ensure_ascii=False), "```",
        "### OFFICIAL CASES (dtypes and attrs are authoritative; every case must pass)", "```json",
        "\n".join(json.dumps(_compact_case_for_prompt(c), sort_keys=True, ensure_ascii=False) for c in cases), "```",
        "### CAMPAIGN REGIME INSTANTIATIONS (partitioned from golden+proto+cases semantics)",
    ]
    shared_sel = f"_select_{entrypoint}_config" if entrypoint else ""
    for reg in regime_specs:
        kw = ", ".join(f"{k}={v!r}" for k, v in sorted(reg.sample_builder_kwargs.items()))
        note = f" (or `{shared_sel}(...)` with the same kwargs)" if (shared_sel and reg.config_selector_fn != shared_sel) else ""
        lines.append(f"- regime `{reg.regime_tag}`: call `{reg.config_selector_fn}({kw})`{note}, then a builder from {list(reg.builder_candidates)} with those keyword arguments")
    return "\n".join(lines) + "\n\n"


def _blueprint_summary_section(blueprints: Mapping[str, HirBlueprintPacket], regime_specs: Sequence[RegimeTargetSpec]) -> str:
    lines = ["### PER-REGIME HIR BLUEPRINT SUMMARIES (all partitions must instantiate)"]
    for reg in regime_specs:
        if (pkt := blueprints.get(reg.regime_tag)) is None:
            raise ValueError(f"Stage-0 blueprint synthesis missing regime {reg.regime_tag!r}")
        rf = pkt.roofline
        rdims = "same" if pkt.resolved_hir_dims == pkt.requested_anchor_dims else str(pkt.resolved_hir_dims)
        lines.append(
            f"- regime `{reg.regime_tag}`: entry={pkt.entry_name}; anchor_dims={pkt.requested_anchor_dims}; "
            f"resolved_hir_dims={rdims}; ideal_roofline_us={float(rf.ideal_roofline_us):.4f}; "
            f"compute_floor_us={float(rf.compute_floor_us):.4f}; memory_floor_us={float(rf.memory_floor_us):.4f}; "
            f"dominant_bound={rf.dominant_bound}; shape_exact_match={pkt.shape_exact_match}; "
            f"fallback_reason={pkt.fallback_reason or 'none'}"
        )
    return "\n".join(lines) + "\n\n"


def _seed_feedback_section(last_error: str, rejected_source: str | None, rejected_error: str) -> str:
    if not last_error:
        return ""
    sections = ["### PREVIOUS ATTEMPT FAILURE DIAGNOSTIC", last_error, ""]
    if rejected_source is not None:
        sections.extend([format_g5_parent_monotonicity_lock(include_blueprint_retention=True), "", "### LAST REJECTED CANDIDATE SOURCE", "```python", rejected_source, "```", ""])
        if rejected_error and rejected_error != last_error:
            sections.extend(["### LAST REJECTED CANDIDATE DIAGNOSTIC", rejected_error, ""])
    return "\n".join(sections) + "\n"


def _build_blueprints(
    analyzer: PyTorchSemanticAnalyzer, semantics: PyTorchOpSemantics, regime_specs: Sequence[RegimeTargetSpec],
    *, proto: Mapping[str, object] | None = None,
) -> dict[str, HirBlueprintPacket]:
    return {
        reg.regime_tag: analyzer.build_universal_blueprint(
            semantics, reg.representative_anchor_dims or reg.representative_hir_dims,
            representative_case=RepresentativeCase.from_case_entry(reg.representative_case_entry, proto=proto) if reg.representative_case_entry else None,
        )
        for reg in regime_specs
    }


def _validate_stage0_entrypoint_and_alias(module_source: str, contract: GoldenEntrypointContract) -> str:
    try:
        tree = ast.parse(module_source)
    except SyntaxError as exc:
        return f"seed module syntax error: {exc}"
    if not any(isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == contract.entrypoint for n in tree.body):
        return f"seed module does not define the public entrypoint {contract.entrypoint}"
    kernel_target: str | None = None
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else ([node.target] if isinstance(node, ast.AnnAssign) else ())
        if any(isinstance(t, ast.Name) and t.id == "kernel" for t in targets):
            kernel_target = node.value.id if isinstance(node.value, ast.Name) else ""
    if kernel_target != contract.entrypoint:
        return f"seed module does not bind `kernel = {contract.entrypoint}`"
    return validate_entrypoint_signature_compatibility(module_source, contract)


def _extract_oracle_numeric_errors(oracle_detail: str) -> dict[str, float]:
    if not oracle_detail:
        return {}
    errors: dict[str, float] = {}
    for key in ("mere", "mare", "max_abs_diff", "max_diff", "abs_diff"):
        if match := re.search(rf"\b{key}\s*[=:]\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|nan|inf|-inf)\b", oracle_detail, flags=re.I):
            try:
                val = float(match.group(1))
                if math.isfinite(val) and val >= 0.0:
                    errors[key] = val
            except ValueError:
                pass
    return errors


def _is_oracle_converging(prev_m: Mapping[str, float], curr_m: Mapping[str, float]) -> bool:
    shared = [k for k in ("mere", "mare", "max_abs_diff", "max_diff", "abs_diff") if k in prev_m and k in curr_m]
    if not shared or not any(curr_m[k] < 0.95 * prev_m[k] for k in shared):
        return False
    return not any(k in prev_m and k in curr_m and curr_m[k] > 1.05 * prev_m[k] for k in ("max_abs_diff", "max_diff"))


async def prepare_operator_seed_or_takeover(
    *, operator_name: str, entrypoint: str, pytorch_golden_source: str, proto: Mapping[str, object],
    cases: Sequence[Mapping[str, object]], legacy_regime_specs: Sequence[RegimeTargetSpec] | None,
    llm_backend: KernelLLMBackend, l0_l1_validator_factory: Stage0StaticValidatorFactory,
    oracle_accuracy_validator: Callable[[str], OracleAccuracyResult], existing_module_path: Path | None = None,
    force_cold_start: bool = False, max_seed_attempts: int = 4, max_oracle_refinement_extension: int = 4,
    max_transport_retries: int = 2, max_env_retries: int = 3, desc_text: str = "",
    golden_contract: GoldenEntrypointContract | None = None,
) -> SeedSynthesisOutcome:
    """Execute unified Stage-0 preparation: pure PyTorch cold start or incumbent takeover."""
    analyzer = PyTorchSemanticAnalyzer()
    semantics = analyzer.analyze_source(pytorch_golden_source, entry_name=entrypoint, proto=proto)
    ratios = tuple(analyzer.compute_case_ratios(semantics, c) for c in cases)

    if not force_cold_start and existing_module_path is not None and existing_module_path.is_file():
        if not legacy_regime_specs:
            raise RuntimeError(f"incumbent takeover for {operator_name} requires campaign regime specs")
        incumbent_src = existing_module_path.read_text(encoding="utf-8")
        if normalize_runtime_backend(legacy_regime_specs[0].runtime_backend) not in ("tilelang_dsl", "ascendc_trackb", "hybrid_trackb"):
            raise ValueError(f"incumbent {operator_name!r} declares unsupported runtime backend {legacy_regime_specs[0].runtime_backend!r}")
        incumbent_regimes = tuple(legacy_regime_specs)
        return SeedSynthesisOutcome(
            mode="incumbent_takeover", initial_track="incumbent_detected", module_source=incumbent_src,
            semantics=semantics, case_ratios=ratios, regime_specs=incumbent_regimes,
            blueprints_by_regime=_build_blueprints(analyzer, semantics, incumbent_regimes, proto=proto),
            attempts_used=0, verification_state="unverified_incumbent", l0_l1_passed=False,
            oracle_accuracy_passed=False, candidate_sha256=_sha256_text(incumbent_src),
            l3_run_id="", l3_receipt_path="", l3_state="", l3_elapsed_sec=0.0,
        )

    provisional_regimes = build_semantic_regime_specs(
        operator_name=operator_name, entrypoint=entrypoint, cases=cases, proto=proto,
        base_axes=semantics.base_axes, has_reduction=semantics.has_reduction, case_ratios=ratios, runtime_backend="tilelang_dsl",
    )
    blueprints = _build_blueprints(analyzer, semantics, provisional_regimes, proto=proto)
    contract = golden_contract or extract_golden_entrypoint_contract(pytorch_golden_source, entrypoint, proto=proto, desc_text=desc_text)
    golden_abi_block = format_golden_entrypoint_prompt_block(contract)
    lines = pytorch_golden_source.splitlines()
    idx0 = next((i for i, ln in enumerate(lines) if ln.strip() and not ln.strip().startswith("#")), 0)
    clean_golden = "\n".join(lines[idx0:]).strip() or pytorch_golden_source.strip()
    golden_block = f"### PYTORCH MATHEMATICAL REFERENCE (GOLDEN ORACLE)\n```python\n{clean_golden}\n```\n\n"
    first_bp = next(iter(blueprints.values()), None)

    def _prepare_track_state(track_name: str) -> tuple[str, tuple[RegimeTargetSpec, ...], Callable[[str, str], tuple[bool, str]], str, str, str, str]:
        rt_backend, tier = _canonical_seed_track(track_name)
        specs = provisional_regimes if rt_backend == "tilelang_dsl" else build_semantic_regime_specs(
            operator_name=operator_name, entrypoint=entrypoint, cases=cases, proto=proto,
            base_axes=semantics.base_axes, has_reduction=semantics.has_reduction, case_ratios=ratios, runtime_backend=rt_backend,
        )
        validator = l0_l1_validator_factory(specs, semantics=semantics)
        surface = get_dialect_surface(tier)
        c_block = build_progressive_llm_constraints(
            escalation_tier=tier, freedom_mask=ConstraintRelaxationMask.top(), blueprint=first_bp,
            topology_axes=semantics.base_axes, case_ratios=ratios[0] if ratios else None,
        )
        abi_block = _seed_module_abi(entrypoint, _seed_parameter_specs(specs), surface, track_name, golden_contract=contract, regime_specs=specs)
        s_prompt = (
            f"Synthesize a self-contained Ascend 950PR ({surface.display_label}) module for operator '{operator_name}' "
            f"exposing `{entrypoint}` and `kernel = {entrypoint}` with exact numerical parity against the PyTorch "
            f"reference across all shapes. Zero PyTorch/CPU fallback is permitted on NPU tensors. {abi_block}"
        )
        return rt_backend, specs, validator, c_block, s_prompt, _official_contract_section(proto, cases, specs, entrypoint=entrypoint), _blueprint_summary_section(blueprints, specs)

    active_track = select_initial_track_from_semantics(semantics, ratios, blueprints=blueprints)
    runtime_backend, regime_specs, l0_l1_validator, constraint_block, system_prompt, contract_block, blueprint_block = _prepare_track_state(active_track)

    base_attempts, max_extra = max(1, max_seed_attempts), max(0, max_oracle_refinement_extension)
    last_error = last_rejected_error = ""
    last_rejected_source: str | None = None
    last_l0_l1_dsl: str | None = None
    attempt, attempts_used, transport_retries, track_a_static_failures, track_a_oracle_attempts = 1, 0, 0, 0, 0
    attempt_stages: list[Literal["static", "oracle"]] = []
    oracle_error_history: list[dict[str, float]] = []
    track_a_budget = max(2, max_seed_attempts // 2) if (active_track == "tilelang_dsl" and max_seed_attempts >= 4) else 0

    def _record_failure(err: str, cand_src: str, *, stage: Literal["static", "oracle"] = "static", oracle_errors: Mapping[str, float] | None = None) -> None:
        nonlocal active_track, attempt, blueprint_block, constraint_block, contract_block, l0_l1_validator
        nonlocal last_error, last_rejected_error, last_rejected_source, runtime_backend, regime_specs, system_prompt
        nonlocal track_a_static_failures, track_a_oracle_attempts
        last_error = last_rejected_error = err
        last_rejected_source = last_l0_l1_dsl if (stage == "static" and last_l0_l1_dsl is not None) else cand_src
        attempt_stages.append(stage)
        if stage == "oracle":
            if oracle_errors:
                oracle_error_history.append(dict(oracle_errors))
            else:
                oracle_error_history.clear()
        elif len(attempt_stages) >= 2 and attempt_stages[-1] == "static" and attempt_stages[-2] == "static":
            oracle_error_history.clear()
        if active_track == "tilelang_dsl":
            if stage == "static":
                track_a_static_failures += 1
            else:
                track_a_oracle_attempts += 1
        attempt += 1
        if active_track == "tilelang_dsl" and last_l0_l1_dsl is None and track_a_budget > 0 and track_a_static_failures >= track_a_budget and attempt <= base_attempts:
            if (fb := select_fallback_track_from_semantics(semantics, ratios, blueprints=blueprints)) != active_track:
                logger.info("STAGE0_TRACK_PIVOT op=%s attempt=%d from=%s to=%s static_failures=%d oracle_attempts=%d", operator_name, attempt, active_track, fb, track_a_static_failures, track_a_oracle_attempts)
                active_track = fb
                runtime_backend, regime_specs, l0_l1_validator, constraint_block, system_prompt, contract_block, blueprint_block = _prepare_track_state(active_track)
                last_rejected_source, last_rejected_error = None, ""

    def _should_continue() -> bool:
        if attempt <= base_attempts:
            return True
        if attempt > base_attempts + max_extra or not attempt_stages:
            return False
        trailing = 0
        for st in reversed(attempt_stages):
            if st == "oracle":
                break
            trailing += 1
        return trailing <= 1 and len(oracle_error_history) >= 2 and _is_oracle_converging(oracle_error_history[-2], oracle_error_history[-1])

    while _should_continue():
        attempts_used = attempt
        request = GenerationRequest(
            role="syntax_repair" if last_rejected_source is not None else "explore_creative",
            system_prompt=system_prompt,
            user_prompt=(
                f"{constraint_block}\n\n{golden_abi_block}{blueprint_block}{golden_block}{contract_block}"
                f"{_seed_feedback_section(last_error, last_rejected_source, last_rejected_error)}"
                "Output ONLY valid Python module source code."
            ),
            effort="high",
        )
        try:
            synthesized_src = extract_python_dsl(await llm_backend.generate(request))
        except (RuntimeError, ValueError, TimeoutError) as exc:
            last_error = f"LLM generation transport failure: {type(exc).__name__}: {exc}"
            if transport_retries >= max(0, max_transport_retries):
                raise RuntimeError(f"Stage-0 cold-start seed transport failed for {operator_name} on semantic attempt {attempt} after {transport_retries} retries: {last_error}") from exc
            transport_retries += 1
            continue

        transport_retries = 0
        if golden_contract is not None and (entry_err := _validate_stage0_entrypoint_and_alias(synthesized_src, golden_contract)):
            logger.info("STAGE0_ATTEMPT op=%s attempt=%d track=%s stage=entrypoint_contract detail=%s", operator_name, attempt, active_track, entry_err.replace("\n", " ")[:500])
            _record_failure(f"Stage-0 entrypoint contract rejection:\n{entry_err}", synthesized_src, stage="static")
            continue
        ok_static, static_err = l0_l1_validator(synthesized_src, active_track)
        if not ok_static:
            logger.info("STAGE0_ATTEMPT op=%s attempt=%d track=%s stage=static_l0_l1 detail=%s", operator_name, attempt, active_track, static_err.replace("\n", " ")[:500])
            _record_failure(f"L0/L1 static or compiler rejection:\n{static_err}", synthesized_src, stage="static")
            continue
        if active_track == "tilelang_dsl":
            last_l0_l1_dsl, track_a_static_failures = synthesized_src, 0

        evidence = oracle_accuracy_validator(synthesized_src)
        env_retries = 0
        while not evidence.passed and evidence.failure_kind == "environment" and env_retries < max(0, max_env_retries):
            env_retries += 1
            detail = evidence.error_message or f"L3 environment blocked state={evidence.l3_state}"
            logger.info("STAGE0_ATTEMPT_ENV_BLOCKED op=%s attempt=%d retry=%d/%d track=%s detail=%s", operator_name, attempt, env_retries, max(0, max_env_retries), active_track, detail.replace("\n", " ")[:500])
            evidence = oracle_accuracy_validator(synthesized_src)
        if not evidence.passed and evidence.failure_kind == "environment":
            detail = evidence.error_message or f"L3 environment blocked state={evidence.l3_state}"
            raise RuntimeError(f"Stage-0 cold-start oracle verification blocked by environment for {operator_name} on attempt {attempt} after {env_retries} retries: {detail}")

        if evidence.passed:
            digest = _sha256_text(synthesized_src)
            if evidence.candidate_sha256 != digest:
                raise RuntimeError(f"Stage-0 oracle evidence digest mismatch for {operator_name}: validator verified {evidence.candidate_sha256!r}, seed digest is {digest}")
            if not evidence.l3_run_id or not evidence.l3_receipt_path:
                raise RuntimeError(f"Stage-0 oracle evidence for {operator_name} lacks an L3 run_id/receipt")
            if runtime_backend in ("ascendc_trackb", "hybrid_trackb"):
                symbols = discover_module_trackb_symbols(synthesized_src)
                if not symbols:
                    raise RuntimeError("Track-B Stage-0 seed has no discoverable C++20 translation-unit string constant")
                render_sym = ("_render_trackb_tu",) if "def _render_trackb_tu(" in synthesized_src else ()
                verified_regimes = tuple(
                    replace(r, runtime_backend=runtime_backend, trackb_symbols=symbols, editable_symbols=tuple(dict.fromkeys([*r.editable_symbols, *symbols, *render_sym])))
                    for r in regime_specs
                )
            else:
                verified_regimes = tuple(regime_specs)
            return SeedSynthesisOutcome(
                mode="cold_start", initial_track=active_track, module_source=synthesized_src, semantics=semantics,
                case_ratios=ratios, regime_specs=verified_regimes, blueprints_by_regime=blueprints,
                attempts_used=attempt, verification_state="l3_verified", l0_l1_passed=True,
                oracle_accuracy_passed=True, candidate_sha256=digest, l3_run_id=evidence.l3_run_id,
                l3_receipt_path=evidence.l3_receipt_path, l3_state=evidence.l3_state, l3_elapsed_sec=evidence.elapsed_sec,
            )

        oracle_detail = evidence.error_message or f"L3 promotion state={evidence.l3_state}"
        logger.info("STAGE0_ATTEMPT op=%s attempt=%d track=%s stage=oracle_l3 detail=%s", operator_name, attempt, active_track, oracle_detail.replace("\n", " ")[:500])
        _record_failure(f"NPU numerical mismatch against PyTorch golden:\n{oracle_detail}", synthesized_src, stage="oracle", oracle_errors=_extract_oracle_numeric_errors(oracle_detail))

    raise RuntimeError(f"Stage-0 cold-start seed synthesis failed for {operator_name} after {attempts_used} attempts: {last_error}")


__all__ = [
    "GoldenEntrypointContract", "GoldenParamSpec", "OracleAccuracyResult", "SeedMode",
    "SeedSynthesisOutcome", "SeedVerificationState", "Stage0StaticValidatorFactory",
    "extract_golden_entrypoint_contract", "format_golden_entrypoint_prompt_block",
    "prepare_operator_seed_or_takeover", "select_fallback_track_from_semantics",
    "select_initial_track_from_semantics", "validate_entrypoint_signature_compatibility",
]
