"""Stage-0 semantic seed synthesis from PyTorch golden references and workload contracts."""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from tilefoundry_ascend.agent.campaign.anchor_specs import (
    RegimeTargetSpec,
    build_semantic_regime_specs,
    discover_module_trackb_symbols,
)
from tilefoundry_ascend.agent.kb.schema import normalize_runtime_backend
from tilefoundry_ascend.agent.orchestrator.llm_client import (
    GenerationRequest,
    KernelLLMBackend,
    extract_python_dsl,
)

logger = logging.getLogger(__name__)
from tilefoundry_ascend.agent.orchestrator.llm_constraints import (
    build_progressive_llm_constraints,
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

SeedMode = Literal["cold_start", "incumbent_takeover"]
SeedVerificationState = Literal["l3_verified", "unverified_incumbent"]


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


def _sha256_text(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _blueprint_packets_from(
    blueprints: Mapping[str, HirBlueprintPacket] | Sequence[HirBlueprintPacket] | None,
) -> tuple[HirBlueprintPacket, ...]:
    if blueprints is None:
        return ()
    if isinstance(blueprints, Mapping):
        return tuple(blueprints.values())
    return tuple(blueprints)


def _schedule_delta_from_packet(packet: HirBlueprintPacket) -> Mapping[str, object] | None:
    payload = packet.to_dict()
    raw_delta = payload.get("schedule_delta")
    return raw_delta if isinstance(raw_delta, Mapping) else None


def _delta_has_sram_or_roundtrip_pressure(delta: Mapping[str, object]) -> bool:
    for key in ("scheduled_ledger", "unscheduled_ledger"):
        raw_ledger = delta.get(key)
        if not isinstance(raw_ledger, Mapping):
            continue
        gm_roundtrip = raw_ledger.get("gm_roundtrip_bytes", 0)
        max_overflow = raw_ledger.get("max_overflow_bytes", 0)
        if (
            isinstance(gm_roundtrip, int)
            and not isinstance(gm_roundtrip, bool)
            and gm_roundtrip > 0
        ) or (
            isinstance(max_overflow, int)
            and not isinstance(max_overflow, bool)
            and max_overflow > 0
        ):
            return True
    return False


def _delta_recommends_facets(delta: Mapping[str, object], required: tuple[str, ...]) -> bool:
    raw_facets = delta.get("recommended_facets")
    if not isinstance(raw_facets, Sequence) or isinstance(raw_facets, (str, bytes)):
        return False
    present = {str(item) for item in raw_facets}
    return all(facet in present for facet in required)


def select_initial_track_from_semantics(
    semantics: PyTorchOpSemantics,
    ratios: Sequence[CasePhysicsRatios],
    *,
    blueprints: Mapping[str, HirBlueprintPacket] | Sequence[HirBlueprintPacket] | None = None,
) -> str:
    """Select Track-A TileLang DSL vs Track-B CCE C++20 from workload and Roofline physics."""
    if semantics.has_subbyte_or_block_quant or semantics.has_indirect_index:
        return "cce_trackb"
    if any(r.working_set_sram_ratio > 1.0 and r.stride_ratio > 1.0 for r in ratios):
        return "cce_trackb"

    packets = _blueprint_packets_from(blueprints)
    if not packets:
        return "tilelang_dsl"

    axes = semantics.base_axes
    if axes.compute_engine == "coupled_1c2v":
        for packet in packets:
            roofline = packet.roofline
            in_l0c_ub_zone = (
                roofline.cube_flops > 0
                and roofline.vector_flops > 0
                and 8.0 <= float(roofline.arithmetic_intensity) <= 128.0
                and (
                    axes.memory_ring
                    in ("bidirectional_cv_ring", "l0c_ub_epilogue", "persistent_onchip_state")
                    or float(roofline.vec_floor_us)
                    >= 0.25 * max(1e-9, float(roofline.cube_floor_us))
                )
            )
            if in_l0c_ub_zone:
                return "hybrid_trackb"
            delta = _schedule_delta_from_packet(packet)
            if delta is not None and (
                _delta_has_sram_or_roundtrip_pressure(delta)
                or _delta_recommends_facets(
                    delta, ("pipeline_and_sync_graph", "sram_layout_and_aliasing")
                )
            ):
                return "hybrid_trackb"

    if semantics.has_reduction and not axes.uses_cube:
        has_unaligned_tail = any(
            (not r.is_atom_aligned) and (r.inner_contiguous_bytes % 32 != 0)
            for r in ratios
        )
        for packet in packets:
            roofline = packet.roofline
            if has_unaligned_tail and (
                roofline.dominant_bound == "memory" or roofline.vector_flops > 0
            ):
                return "cce_trackb"
            delta = _schedule_delta_from_packet(packet)
            if delta is not None and _delta_recommends_facets(
                delta, ("exec_mode_and_spr", "pipeline_and_sync_graph")
            ):
                return "cce_trackb"

    return "tilelang_dsl"


def _canonical_seed_track(initial_track: str) -> tuple[str, str]:
    """Return (runtime_backend, escalation_tier) for a selected Stage-0 track."""
    backend = normalize_runtime_backend(initial_track)
    if backend == "tilelang_dsl":
        return "tilelang_dsl", "tilelang_dsl"
    if backend == "ascendc_trackb":
        return "ascendc_trackb", "cce_cpp20"
    if backend == "hybrid_trackb":
        return "hybrid_trackb", "cce_cpp20"
    raise RuntimeError(f"unsupported Stage-0 seed track {initial_track!r}")


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


def _unwrap_proto_operator(proto: Mapping[str, object] | None) -> Mapping[str, object]:
    if not isinstance(proto, Mapping):
        return {}
    inner = proto.get("operator")
    if isinstance(inner, Mapping) and (
        "inputs" in inner or "outputs" in inner or "attrs" in inner or "formula" in inner
    ):
        return inner
    return proto


def _normalize_scalar_annotation(raw_type: str) -> str:
    low = raw_type.strip().lower()
    if low in ("int", "int32", "int64"):
        return "int"
    if low in ("float", "float32", "float64"):
        return "float"
    if low in ("bool",):
        return "bool"
    if low in ("str", "string"):
        return "str"
    if low in ("listint", "list[int]"):
        return "list[int]"
    if low in ("listfloat", "list[float]"):
        return "list[float]"
    if low in ("list",):
        return "list"
    return raw_type.strip() or "object"


def _normalize_param_annotation(
    raw_ann: str,
    *,
    has_default: bool,
    default_is_none: bool,
    proto_input: Mapping[str, object] | None,
    proto_attr: Mapping[str, object] | None,
) -> tuple[str, bool, bool, bool]:
    cleaned = raw_ann.strip()
    is_optional = (
        default_is_none
        or "Optional[" in cleaned
        or "| None" in cleaned
        or "None |" in cleaned
        or bool((proto_input or {}).get("optional", False))
    )
    if "Tensor" in cleaned:
        is_list = "List[" in cleaned or "list[" in cleaned
        if is_list:
            ann = "Optional[List[torch.Tensor]]" if is_optional else "List[torch.Tensor]"
            return ann, True, True, is_optional
        ann = "Optional[torch.Tensor]" if is_optional else "torch.Tensor"
        return ann, True, False, is_optional
    if proto_input is not None and not cleaned:
        raw_shape = str(proto_input.get("shape", ""))
        is_list = "List[" in raw_shape or "list[" in raw_shape
        if is_list:
            ann = "Optional[List[torch.Tensor]]" if is_optional else "List[torch.Tensor]"
            return ann, True, True, is_optional
        ann = "Optional[torch.Tensor]" if is_optional else "torch.Tensor"
        return ann, True, False, is_optional
    if cleaned:
        if cleaned.startswith("Optional[") and cleaned.endswith("]"):
            inner = _normalize_scalar_annotation(cleaned[len("Optional[") : -1])
            return f"Optional[{inner}]", False, False, True
        norm = _normalize_scalar_annotation(cleaned)
        if default_is_none and not has_default:
            return f"Optional[{norm}]", False, False, True
        return norm, False, False, is_optional
    if proto_attr is not None:
        attr_type = str(proto_attr.get("type", "")).strip()
        if attr_type:
            norm = _normalize_scalar_annotation(attr_type)
            if default_is_none:
                return f"Optional[{norm}]", False, False, True
            return norm, False, False, is_optional
    return ("object", False, False, is_optional)


def _extract_math_summary(
    proto: Mapping[str, object],
    desc_text: str,
) -> str:
    parts: list[str] = []
    formula = str(proto.get("formula", "")).strip()
    desc = str(proto.get("description", "")).strip()
    if desc:
        parts.append(desc)
    if formula:
        compact_formula = " ".join(line.strip() for line in formula.splitlines() if line.strip())
        if compact_formula:
            parts.append(f"Formula: {compact_formula[:220]}")
    if desc_text.strip():
        lines = [
            re.sub(r"^#+\s*", "", line).strip()
            for line in desc_text.splitlines()
            if line.strip()
            and not line.strip().startswith("---")
            and "Copyright" not in line
            and "License" not in line
        ]
        if lines:
            parts.append(" ".join(lines)[:240])
    combined = " | ".join(dict.fromkeys(p for p in parts if p))
    return combined[:420]


def _find_entrypoint_def(
    tree: ast.Module,
    entrypoint: str,
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    top_defs = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    for node in top_defs:
        if node.name == entrypoint:
            return node
    if len(top_defs) == 1:
        return top_defs[0]
    return None


def extract_golden_entrypoint_contract(
    golden_source: str,
    entrypoint: str,
    *,
    proto: Mapping[str, object] | None = None,
    desc_text: str = "",
) -> GoldenEntrypointContract:
    """Extract typed entrypoint parameters, defaults, return type, and math summary."""
    op_proto = _unwrap_proto_operator(proto)
    proto_inputs = {
        str(row["name"]): row
        for row in (op_proto.get("inputs") or ())
        if isinstance(row, Mapping) and "name" in row
    }
    proto_attrs = {
        str(row["name"]): row
        for row in (op_proto.get("attrs") or ())
        if isinstance(row, Mapping) and "name" in row
    }
    raw_outputs = [
        row for row in (op_proto.get("outputs") or ()) if isinstance(row, Mapping)
    ]
    output_names = tuple(str(row.get("name", "")) for row in raw_outputs if row.get("name"))
    out_dtype_tokens: list[str] = []
    for row in raw_outputs:
        raw_dt = row.get("dtype")
        if isinstance(raw_dt, str) and raw_dt.strip():
            out_dtype_tokens.append(raw_dt.strip())
        elif isinstance(raw_dt, Sequence) and not isinstance(raw_dt, (str, bytes)):
            cands = [str(item).strip() for item in raw_dt if item is not None]
            if cands:
                out_dtype_tokens.append("/".join(cands))
    output_dtypes = tuple(out_dtype_tokens)
    math_summary = _extract_math_summary(op_proto, desc_text)

    params: list[GoldenParamSpec] = []
    return_annotation = (
        "tuple[torch.Tensor, ...]" if len(raw_outputs) > 1 else "torch.Tensor"
    )
    fn_node: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    if golden_source.strip():
        try:
            tree = ast.parse(golden_source)
            fn_node = _find_entrypoint_def(tree, entrypoint)
        except SyntaxError:
            fn_node = None

    if fn_node is not None:
        pos_args = [*fn_node.args.posonlyargs, *fn_node.args.args]
        pos_defaults: list[ast.expr | None] = [None] * (
            len(pos_args) - len(fn_node.args.defaults)
        ) + list(fn_node.args.defaults)
        all_args: list[tuple[ast.arg, ast.expr | None]] = list(zip(pos_args, pos_defaults))
        for kw_arg, kw_def in zip(fn_node.args.kwonlyargs, fn_node.args.kw_defaults):
            all_args.append((kw_arg, kw_def))

        for arg, default_node in all_args:
            raw_ann = ast.unparse(arg.annotation) if arg.annotation is not None else ""
            has_default = default_node is not None
            default_repr = ast.unparse(default_node) if default_node is not None else None
            default_is_none = default_repr == "None"
            ann, is_tensor, is_tensor_list, is_optional = _normalize_param_annotation(
                raw_ann,
                has_default=has_default,
                default_is_none=default_is_none,
                proto_input=proto_inputs.get(arg.arg),
                proto_attr=proto_attrs.get(arg.arg),
            )
            params.append(
                GoldenParamSpec(
                    name=arg.arg,
                    annotation=ann,
                    default_repr=default_repr,
                    has_default=has_default,
                    is_tensor=is_tensor,
                    is_tensor_list=is_tensor_list,
                    is_optional=is_optional,
                )
            )
        if fn_node.returns is not None:
            raw_ret = ast.unparse(fn_node.returns).strip()
            if raw_ret:
                return_annotation = raw_ret
    else:
        for row in (op_proto.get("inputs") or ()):
            if not isinstance(row, Mapping) or "name" not in row:
                continue
            pname = str(row["name"])
            is_opt = bool(row.get("optional", False))
            ann, is_tensor, is_tensor_list, _ = _normalize_param_annotation(
                "",
                has_default=is_opt,
                default_is_none=is_opt,
                proto_input=row,
                proto_attr=None,
            )
            params.append(
                GoldenParamSpec(
                    name=pname,
                    annotation=ann,
                    default_repr="None" if is_opt else None,
                    has_default=is_opt,
                    is_tensor=is_tensor,
                    is_tensor_list=is_tensor_list,
                    is_optional=is_opt,
                )
            )
        for row in (op_proto.get("attrs") or ()):
            if not isinstance(row, Mapping) or "name" not in row:
                continue
            pname = str(row["name"])
            has_def = "default" in row
            def_repr = repr(row["default"]) if has_def else None
            ann, _, _, is_opt = _normalize_param_annotation(
                "",
                has_default=has_def,
                default_is_none=has_def and row["default"] is None,
                proto_input=None,
                proto_attr=row,
            )
            params.append(
                GoldenParamSpec(
                    name=pname,
                    annotation=ann,
                    default_repr=def_repr,
                    has_default=has_def,
                    is_tensor=False,
                    is_tensor_list=False,
                    is_optional=is_opt,
                )
            )

    rendered_params = ", ".join(
        f"{p.name}: {p.annotation} = {p.default_repr}"
        if p.has_default and p.default_repr is not None
        else f"{p.name}: {p.annotation}"
        for p in params
    )
    signature_line = f"def {entrypoint}({rendered_params}) -> {return_annotation}:"
    return GoldenEntrypointContract(
        entrypoint=entrypoint,
        parameters=tuple(params),
        return_annotation=return_annotation,
        signature_line=signature_line,
        math_summary=math_summary,
        output_names=output_names,
        output_dtypes=output_dtypes,
    )


def validate_entrypoint_signature_compatibility(
    module_source: str,
    contract: GoldenEntrypointContract,
) -> str:
    """Return an error when candidate entrypoint breaks ParamBuilder tensor/param ABI."""
    if not contract.parameters:
        return ""
    try:
        tree = ast.parse(module_source)
    except SyntaxError as exc:
        return f"seed module syntax error: {exc}"
    fn_node = next(
        (
            node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == contract.entrypoint
        ),
        None,
    )
    if fn_node is None:
        return f"seed module does not define the public entrypoint {contract.entrypoint}"

    pos_args = [*fn_node.args.posonlyargs, *fn_node.args.args]
    pos_defaults: list[ast.expr | None] = [None] * (
        len(pos_args) - len(fn_node.args.defaults)
    ) + list(fn_node.args.defaults)
    cand_args: list[tuple[ast.arg, ast.expr | None]] = list(zip(pos_args, pos_defaults))
    for kw_arg, kw_def in zip(fn_node.args.kwonlyargs, fn_node.args.kw_defaults):
        cand_args.append((kw_arg, kw_def))

    expected_names = [p.name for p in contract.parameters]
    cand_names = [arg.arg for arg, _ in cand_args]
    if cand_names[: len(expected_names)] != expected_names:
        return (
            f"entrypoint {contract.entrypoint!r} parameter signature mismatch for ParamBuilder: "
            f"expected {expected_names}, got {cand_names}"
        )
    for extra_arg, extra_def in cand_args[len(expected_names) :]:
        if extra_def is None:
            return (
                f"entrypoint {contract.entrypoint!r} adds required parameter "
                f"{extra_arg.arg!r} absent from golden signature"
            )

    for param, (cand_arg, cand_def) in zip(contract.parameters, cand_args):
        cand_ann = ast.unparse(cand_arg.annotation) if cand_arg.annotation is not None else ""
        if param.is_tensor:
            if "Tensor" not in cand_ann:
                return (
                    f"entrypoint {contract.entrypoint!r} parameter {param.name!r} must carry "
                    f"a Tensor annotation ({param.annotation}) so ParamBuilder binds it as a tensor"
                )
            if param.is_tensor_list and ("List[" not in cand_ann and "list[" not in cand_ann):
                return (
                    f"entrypoint {contract.entrypoint!r} parameter {param.name!r} must carry "
                    f"a List[torch.Tensor] annotation ({param.annotation})"
                )
            if (
                param.is_optional
                and cand_def is None
                and "Optional" not in cand_ann
                and "None" not in cand_ann
            ):
                return (
                    f"entrypoint {contract.entrypoint!r} optional tensor {param.name!r} "
                    f"must preserve Optional/default None"
                )
        else:
            if "Tensor" in cand_ann:
                return (
                    f"entrypoint {contract.entrypoint!r} scalar parameter {param.name!r} "
                    f"must not carry a Tensor annotation"
                )
            if param.has_default and cand_def is None:
                return (
                    f"entrypoint {contract.entrypoint!r} parameter {param.name!r} must "
                    f"preserve default value ({param.default_repr})"
                )
    return ""


def format_golden_entrypoint_prompt_block(contract: GoldenEntrypointContract) -> str:
    """Format exact entrypoint ABI and math contract for Stage-0 and optimization prompts."""
    lines = [
        "### EXACT GOLDEN ENTRYPOINT ABI (ParamBuilder-checked)",
        "```python",
        contract.signature_line,
        "```",
        (
            "- Keep parameter names, order, defaults, and `torch.Tensor` / "
            "`Optional[torch.Tensor]` / `List[torch.Tensor]` annotations exact."
        ),
    ]
    if contract.math_summary:
        lines.append(f"- Mathematical specification: {contract.math_summary}")
    if contract.output_names or contract.output_dtypes:
        lines.append(
            f"- Declared outputs: names={list(contract.output_names)} "
            f"dtypes={list(contract.output_dtypes)}"
        )
    return "\n".join(lines) + "\n\n"


def _prompt_type_name(value: object) -> str:
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, tuple):
        return f"tuple[{_prompt_type_name(value[0])}, ...]" if value else "tuple"
    if isinstance(value, list):
        return f"list[{_prompt_type_name(value[0])}]" if value else "list"
    raise ValueError(f"unsupported regime instantiation value {value!r} for the seed ABI prompt")


def _seed_parameter_specs(
    regime_specs: Sequence[RegimeTargetSpec],
) -> tuple[tuple[str, object], ...]:
    ordered: dict[str, object] = {}
    for regime in regime_specs:
        for key in sorted(regime.sample_builder_kwargs):
            val = regime.sample_builder_kwargs[key]
            if isinstance(val, (list, tuple)) and len(val) > 8:
                continue
            if key not in ordered:
                ordered[key] = val
    return tuple(ordered.items())


def _render_signature(
    name: str,
    parameters: Sequence[tuple[str, object]],
    return_type: str,
) -> str:
    rendered = ", ".join(
        f"{key}: {_prompt_type_name(value)} = {value!r}" for key, value in parameters
    )
    return f"def {name}({rendered}) -> {return_type}:"


def _seed_module_abi(
    entrypoint: str,
    parameters: Sequence[tuple[str, object]],
    surface: DialectSurfaceSpec,
    track: str,
    *,
    golden_contract: GoldenEntrypointContract | None = None,
    regime_specs: Sequence[RegimeTargetSpec] = (),
) -> str:
    """Describe the concrete module symbols and invocation contract Stage-0 resolves."""
    selector = _render_signature(f"_select_{entrypoint}_config", parameters, "dict[str, object]")
    builder = _render_signature(f"_build_{entrypoint}_prim_func", parameters, "object")
    get_kernel = _render_signature("_get_kernel", parameters, "object")
    render_tu = "def _render_trackb_tu(**dims: object) -> str:"
    dispatch = (
        golden_contract.signature_line
        if golden_contract is not None and golden_contract.signature_line
        else f"def {entrypoint}(...) -> object:"
    )
    forbidden = ", ".join(f"`{token}`" for token in surface.forbidden_tokens)
    per_regime_lines: list[str] = []
    if len(regime_specs) > 1:
        for regime in regime_specs:
            tag = regime.regime_tag
            r_sel = _render_signature(
                f"_select_{entrypoint}_{tag}_config",
                parameters,
                "dict[str, object]",
            )
            r_bld = _render_signature(
                f"_build_{entrypoint}_{tag}_prim_func",
                parameters,
                "object",
            )
            per_regime_lines.extend((r_sel, r_bld))

    if track in ("cce_trackb", "hybrid_trackb"):
        symbol_block = [
            *per_regime_lines,
            selector,
            builder,
            get_kernel,
            render_tu,
            dispatch,
            f"{entrypoint.upper()}_TRACKB_CU: str",
            f"kernel = {entrypoint}",
        ]
        hybrid_note = (
            "- `hybrid_trackb`: pair Track-B C++20 translation units with `@T.prim_func` "
            "builders compiled via `tilelang.compile(prog, target=\"ascend\")` when "
            "splitting Cube/Vector/prologue/combine stages.\n"
            if track == "hybrid_trackb"
            else ""
        )
        multi_note = (
            f"- Multiple regimes detected: emit per-regime `_select_{entrypoint}_<tag>_config` "
            f"and `_build_{entrypoint}_<tag>_prim_func` functions so each regime edits "
            f"orthogonal symbols, and dispatch across them from `_select_{entrypoint}_config` "
            f"/ `{entrypoint}`.\n"
            if len(regime_specs) > 1
            else ""
        )
        return (
            "Required module symbols and exact Stage-0 instantiations:\n"
            "```python\n"
            + "\n".join(symbol_block)
            + "\n```\n"
            f"- `{entrypoint.upper()}_TRACKB_CU` is a top-level C++20 translation-unit string "
            "constant starting with `#include <tl_templates/ascend/...>` that bisheng compiles "
            "standalone.\n"
            "- `_render_trackb_tu` accepts the regime keyword arguments (plus `trackb_src`) as "
            "`**dims`, substitutes every shape/constexpr dimension into the C++20 source, and "
            "returns the complete instantiated top-level translation unit for those dims. It is "
            "called once per campaign regime before L0/L1, so every returned TU must compile "
            "standalone and contain exactly one launcher `<<<...>>>` call per active kernel.\n"
            f"- `_select_{entrypoint}_config` is called with the regime keyword arguments above "
            "and returns `pass_configs`, `compile_flags`, or `builder_kwargs` overrides.\n"
            f"- `_build_{entrypoint}_prim_func` is called with the same keyword arguments plus "
            "any selector `builder_kwargs`, and returns the Track-B kernel plan/translation-unit "
            "selection that `_get_kernel` compiles.\n"
            "- `_get_kernel` compiles and caches that Track-B plan for the dispatch callable.\n"
            f"- `{entrypoint}` dispatches NPU tensors through the bisheng-compiled Track-B "
            "library; no torch/CPU fallback may remain on the NPU path.\n"
            f"{hybrid_note}{multi_note}"
            f"- The module must not contain any forbidden Track-B symbol: {forbidden}.\n"
        )
    symbol_block = [
        *per_regime_lines,
        selector,
        builder,
        get_kernel,
        dispatch,
        f"kernel = {entrypoint}",
    ]
    multi_note = (
        f"- Multiple regimes detected: emit per-regime `_select_{entrypoint}_<tag>_config` "
        f"and `_build_{entrypoint}_<tag>_prim_func` functions so each regime edits "
        f"orthogonal symbols, and dispatch across them from `_select_{entrypoint}_config` "
        f"/ `{entrypoint}`.\n"
        if len(regime_specs) > 1
        else ""
    )
    return (
        "Required module symbols and exact Stage-0 instantiations:\n"
        "```python\n"
        + "\n".join(symbol_block)
        + "\n```\n"
        f"- `_select_{entrypoint}_config` is called with the regime keyword arguments above and "
        "returns optional `pass_configs`, `compile_flags`, and `builder_kwargs`.\n"
        f"- `_build_{entrypoint}_prim_func` is called with the same keyword arguments plus any "
        "selector `builder_kwargs`, and returns an inner `@T.prim_func`.\n"
        "- `_get_kernel` compiles that prim_func through "
        '`tilelang.compile(prog, target="ascend")` and caches the compiled callable.\n'
        f"- `{entrypoint}` performs the operator on NPU tensors through `_get_kernel`; no "
        "torch/CPU fallback may remain on the NPU path.\n"
        f"{multi_note}"
    )


def _seed_system_prompt(
    operator_name: str,
    entrypoint: str,
    surface: DialectSurfaceSpec,
    track: str,
    parameters: Sequence[tuple[str, object]],
    *,
    golden_contract: GoldenEntrypointContract | None = None,
    regime_specs: Sequence[RegimeTargetSpec] = (),
) -> str:
    return (
        f"Synthesize a self-contained Ascend 950PR ({surface.display_label}) module "
        f"for operator '{operator_name}' exposing `{entrypoint}` and "
        f"`kernel = {entrypoint}` with exact numerical parity against the PyTorch "
        "reference across all shapes. Zero PyTorch/CPU fallback is permitted on NPU tensors. "
        f"{_seed_module_abi(entrypoint, parameters, surface, track, golden_contract=golden_contract, regime_specs=regime_specs)}"
    )


def _official_contract_section(
    proto: Mapping[str, object],
    cases: Sequence[Mapping[str, object]],
    regime_specs: Sequence[RegimeTargetSpec],
    *,
    entrypoint: str = "",
) -> str:
    """Render proto.yaml, cases.yaml and the semantic regime instantiations for the prompt."""
    lines = [
        "### OFFICIAL PROTO CONTRACT",
        "```json",
        json.dumps(proto, indent=2, sort_keys=True, ensure_ascii=False),
        "```",
        "### OFFICIAL CASES (dtypes and attrs are authoritative; every case must pass)",
        "```json",
        json.dumps([dict(case) for case in cases], indent=2, sort_keys=True, ensure_ascii=False),
        "```",
        "### CAMPAIGN REGIME INSTANTIATIONS (partitioned from golden+proto+cases semantics)",
    ]
    shared_sel = f"_select_{entrypoint}_config" if entrypoint else ""
    for regime in regime_specs:
        kwargs = ", ".join(
            f"{key}={value!r}" for key, value in sorted(regime.sample_builder_kwargs.items())
        )
        dispatch_note = (
            f" (or `{shared_sel}({kwargs})`)"
            if shared_sel and regime.config_selector_fn != shared_sel
            else ""
        )
        lines.append(
            f"- regime `{regime.regime_tag}`: call `{regime.config_selector_fn}({kwargs})`"
            f"{dispatch_note}, then a builder from {list(regime.builder_candidates)} "
            "with those keyword arguments"
        )
    return "\n".join(lines) + "\n\n"


def _blueprint_summary_section(
    blueprints: Mapping[str, HirBlueprintPacket],
    regime_specs: Sequence[RegimeTargetSpec],
) -> str:
    """Render every regime's HIR Roofline blueprint so Stage-0 sizes all partitions."""
    lines = ["### PER-REGIME HIR BLUEPRINT SUMMARIES (all partitions must instantiate)"]
    for regime in regime_specs:
        packet = blueprints.get(regime.regime_tag)
        if packet is None:
            raise ValueError(
                f"Stage-0 blueprint synthesis missing regime {regime.regime_tag!r}"
            )
        roofline = packet.roofline
        kwargs = ", ".join(
            f"{key}={value!r}" for key, value in sorted(regime.sample_builder_kwargs.items())
        )
        lines.append(
            f"- regime `{regime.regime_tag}`: kwargs({kwargs}); entry={packet.entry_name}; "
            f"anchor_dims={packet.requested_anchor_dims}; "
            f"resolved_hir_dims={packet.resolved_hir_dims}; "
            f"ideal_roofline_us={float(roofline.ideal_roofline_us):.4f}; "
            f"compute_floor_us={float(roofline.compute_floor_us):.4f}; "
            f"memory_floor_us={float(roofline.memory_floor_us):.4f}; "
            f"dominant_bound={roofline.dominant_bound}; "
            f"shape_exact_match={packet.shape_exact_match}; "
            f"fallback_reason={packet.fallback_reason or 'none'}"
        )
    return "\n".join(lines) + "\n\n"


def _seed_feedback_section(
    last_error: str,
    rejected_source: str | None,
    rejected_error: str,
) -> str:
    if not last_error:
        return ""
    sections = ["### PREVIOUS ATTEMPT FAILURE DIAGNOSTIC", last_error, ""]
    if rejected_source is not None:
        sections.extend(
            [
                "### LAST REJECTED CANDIDATE SOURCE",
                "```python",
                rejected_source,
                "```",
                "",
            ]
        )
        if rejected_error and rejected_error != last_error:
            sections.extend(
                [
                    "### LAST REJECTED CANDIDATE DIAGNOSTIC",
                    rejected_error,
                    "",
                ]
            )
    return "\n".join(sections) + "\n"


def _build_blueprints(
    analyzer: PyTorchSemanticAnalyzer,
    semantics: PyTorchOpSemantics,
    regime_specs: Sequence[RegimeTargetSpec],
    *,
    authored_hir_source: str | None,
    scheduled_hir_source: str | None,
    proto: Mapping[str, object] | None = None,
) -> dict[str, HirBlueprintPacket]:
    return {
        regime.regime_tag: analyzer.build_universal_blueprint(
            semantics,
            regime.representative_anchor_dims or regime.representative_hir_dims,
            authored_hir_source=authored_hir_source,
            scheduled_hir_source=scheduled_hir_source,
            representative_case=(
                RepresentativeCase.from_case_entry(
                    regime.representative_case_entry,
                    proto=proto,
                )
                if regime.representative_case_entry
                else None
            ),
        )
        for regime in regime_specs
    }


def _bind_trackb_symbols(
    regime_specs: Sequence[RegimeTargetSpec],
    module_source: str,
    runtime_backend: str,
) -> tuple[RegimeTargetSpec, ...]:
    if runtime_backend not in ("ascendc_trackb", "hybrid_trackb"):
        return tuple(regime_specs)
    symbols = discover_module_trackb_symbols(module_source)
    if not symbols:
        raise RuntimeError(
            "Track-B Stage-0 seed has no discoverable C++20 translation-unit string constant"
        )
    return tuple(
        replace(
            regime,
            runtime_backend=runtime_backend,
            trackb_symbols=symbols,
            editable_symbols=tuple(dict.fromkeys([*regime.editable_symbols, *symbols])),
        )
        for regime in regime_specs
    )


async def prepare_operator_seed_or_takeover(
    *,
    operator_name: str,
    entrypoint: str,
    pytorch_golden_source: str,
    proto: Mapping[str, object],
    cases: Sequence[Mapping[str, object]],
    legacy_regime_specs: Sequence[RegimeTargetSpec] | None,
    llm_backend: KernelLLMBackend,
    l0_l1_validator_factory: Callable[
        [tuple[RegimeTargetSpec, ...]], Callable[[str, str], tuple[bool, str]]
    ],
    oracle_accuracy_validator: Callable[[str], OracleAccuracyResult],
    existing_module_path: Path | None = None,
    authored_hir_source: str | None = None,
    scheduled_hir_source: str | None = None,
    force_cold_start: bool = False,
    max_seed_attempts: int = 4,
    max_transport_retries: int = 2,
    desc_text: str = "",
    golden_contract: GoldenEntrypointContract | None = None,
) -> SeedSynthesisOutcome:
    """Execute unified Stage-0 preparation: pure PyTorch cold start or incumbent takeover."""
    analyzer = PyTorchSemanticAnalyzer()
    semantics = analyzer.analyze_source(
        pytorch_golden_source,
        entry_name=entrypoint,
        proto=proto,
    )
    ratios = tuple(analyzer.compute_case_ratios(semantics, case) for case in cases)

    if not force_cold_start and existing_module_path is not None and existing_module_path.is_file():
        if not legacy_regime_specs:
            raise RuntimeError(
                f"incumbent takeover for {operator_name} requires campaign regime specs"
            )
        incumbent_src = existing_module_path.read_text(encoding="utf-8")
        incumbent_backend = normalize_runtime_backend(legacy_regime_specs[0].runtime_backend)
        if incumbent_backend not in ("tilelang_dsl", "ascendc_trackb", "hybrid_trackb"):
            raise ValueError(
                f"incumbent {operator_name!r} declares unsupported runtime backend "
                f"{legacy_regime_specs[0].runtime_backend!r}"
            )
        incumbent_regimes = tuple(legacy_regime_specs)
        return SeedSynthesisOutcome(
            mode="incumbent_takeover",
            initial_track="incumbent_detected",
            module_source=incumbent_src,
            semantics=semantics,
            case_ratios=ratios,
            regime_specs=incumbent_regimes,
            blueprints_by_regime=_build_blueprints(
                analyzer,
                semantics,
                incumbent_regimes,
                authored_hir_source=authored_hir_source,
                scheduled_hir_source=scheduled_hir_source,
                proto=proto,
            ),
            attempts_used=0,
            verification_state="unverified_incumbent",
            l0_l1_passed=False,
            oracle_accuracy_passed=False,
            candidate_sha256=_sha256_text(incumbent_src),
            l3_run_id="",
            l3_receipt_path="",
            l3_state="",
            l3_elapsed_sec=0.0,
        )

    provisional_regimes = build_semantic_regime_specs(
        operator_name=operator_name,
        entrypoint=entrypoint,
        cases=cases,
        proto=proto,
        base_axes=semantics.base_axes,
        has_reduction=semantics.has_reduction,
        case_ratios=ratios,
        runtime_backend="tilelang_dsl",
    )
    blueprints = _build_blueprints(
        analyzer,
        semantics,
        provisional_regimes,
        authored_hir_source=authored_hir_source,
        scheduled_hir_source=scheduled_hir_source,
        proto=proto,
    )
    initial_track = select_initial_track_from_semantics(
        semantics,
        ratios,
        blueprints=blueprints,
    )
    runtime_backend, tier_name = _canonical_seed_track(initial_track)
    if runtime_backend == "tilelang_dsl":
        regime_specs = provisional_regimes
    else:
        regime_specs = build_semantic_regime_specs(
            operator_name=operator_name,
            entrypoint=entrypoint,
            cases=cases,
            proto=proto,
            base_axes=semantics.base_axes,
            has_reduction=semantics.has_reduction,
            case_ratios=ratios,
            runtime_backend=runtime_backend,
        )
    l0_l1_validator = l0_l1_validator_factory(regime_specs)
    surface = get_dialect_surface(tier_name)
    first_bp = next(iter(blueprints.values()), None)
    constraint_block = build_progressive_llm_constraints(
        escalation_tier=tier_name,
        freedom_mask=ConstraintRelaxationMask.top(),
        blueprint=first_bp,
    )
    contract = (
        golden_contract
        if golden_contract is not None
        else extract_golden_entrypoint_contract(
            pytorch_golden_source,
            entrypoint,
            proto=proto,
            desc_text=desc_text,
        )
    )
    parameters = _seed_parameter_specs(regime_specs)
    system_prompt = _seed_system_prompt(
        operator_name,
        entrypoint,
        surface,
        initial_track,
        parameters,
        golden_contract=contract,
        regime_specs=regime_specs,
    )
    golden_abi_block = format_golden_entrypoint_prompt_block(contract)
    golden_block = (
        "### PYTORCH MATHEMATICAL REFERENCE (GOLDEN ORACLE)\n"
        f"```python\n{pytorch_golden_source.strip()}\n```\n\n"
    )
    contract_block = _official_contract_section(
        proto,
        cases,
        regime_specs,
        entrypoint=entrypoint,
    )
    blueprint_block = _blueprint_summary_section(blueprints, regime_specs)

    last_error = ""
    last_rejected_source: str | None = None
    last_rejected_error = ""
    attempt = 1
    attempts_used = 0
    transport_retries = 0
    while attempt <= max(1, max_seed_attempts):
        attempts_used = attempt
        request = GenerationRequest(
            role=(
                "syntax_repair"
                if last_rejected_source is not None
                else "explore_creative"
            ),
            system_prompt=system_prompt,
            user_prompt=(
                f"{constraint_block}\n\n"
                f"{golden_abi_block}"
                f"{blueprint_block}"
                f"{golden_block}"
                f"{contract_block}"
                f"{_seed_feedback_section(last_error, last_rejected_source, last_rejected_error)}"
                "Output ONLY valid Python module source code."
            ),
            effort="high",
        )
        try:
            raw_reply = await llm_backend.generate(request)
            synthesized_src = extract_python_dsl(raw_reply)
        except (RuntimeError, ValueError, TimeoutError) as exc:
            last_error = f"LLM generation transport failure: {type(exc).__name__}: {exc}"
            if transport_retries >= max(0, max_transport_retries):
                raise RuntimeError(
                    f"Stage-0 cold-start seed transport failed for {operator_name} on semantic "
                    f"attempt {attempt} after {transport_retries} retries: {last_error}"
                ) from exc
            transport_retries += 1
            continue

        transport_retries = 0
        ok_static, static_err = l0_l1_validator(synthesized_src, initial_track)
        if not ok_static:
            logger.info(
                "STAGE0_ATTEMPT op=%s attempt=%d track=%s stage=static_l0_l1 detail=%s",
                operator_name,
                attempt,
                initial_track,
                static_err.replace("\n", " ")[:500],
            )
            last_error = f"L0/L1 static or compiler rejection:\n{static_err}"
            last_rejected_source = synthesized_src
            last_rejected_error = last_error
            attempt += 1
            continue

        evidence = oracle_accuracy_validator(synthesized_src)
        if evidence.passed:
            digest = _sha256_text(synthesized_src)
            if evidence.candidate_sha256 != digest:
                raise RuntimeError(
                    f"Stage-0 oracle evidence digest mismatch for {operator_name}: "
                    f"validator verified {evidence.candidate_sha256!r}, seed digest is {digest}"
                )
            if not evidence.l3_run_id or not evidence.l3_receipt_path:
                raise RuntimeError(
                    f"Stage-0 oracle evidence for {operator_name} lacks an L3 run_id/receipt"
                )
            verified_regimes = _bind_trackb_symbols(
                regime_specs,
                synthesized_src,
                runtime_backend,
            )
            return SeedSynthesisOutcome(
                mode="cold_start",
                initial_track=initial_track,
                module_source=synthesized_src,
                semantics=semantics,
                case_ratios=ratios,
                regime_specs=verified_regimes,
                blueprints_by_regime=blueprints,
                attempts_used=attempt,
                verification_state="l3_verified",
                l0_l1_passed=True,
                oracle_accuracy_passed=True,
                candidate_sha256=digest,
                l3_run_id=evidence.l3_run_id,
                l3_receipt_path=evidence.l3_receipt_path,
                l3_state=evidence.l3_state,
                l3_elapsed_sec=evidence.elapsed_sec,
            )
        oracle_detail = evidence.error_message or f"L3 promotion state={evidence.l3_state}"
        logger.info(
            "STAGE0_ATTEMPT op=%s attempt=%d track=%s stage=oracle_l3 detail=%s",
            operator_name,
            attempt,
            initial_track,
            oracle_detail.replace("\n", " ")[:500],
        )
        last_error = f"NPU numerical mismatch against PyTorch golden:\n{oracle_detail}"
        last_rejected_source = synthesized_src
        last_rejected_error = last_error
        attempt += 1

    raise RuntimeError(
        f"Stage-0 cold-start seed synthesis failed for {operator_name} after "
        f"{attempts_used} attempts: {last_error}"
    )


__all__ = [
    "GoldenEntrypointContract",
    "GoldenParamSpec",
    "OracleAccuracyResult",
    "SeedMode",
    "SeedSynthesisOutcome",
    "SeedVerificationState",
    "extract_golden_entrypoint_contract",
    "format_golden_entrypoint_prompt_block",
    "prepare_operator_seed_or_takeover",
    "select_initial_track_from_semantics",
    "validate_entrypoint_signature_compatibility",
]
