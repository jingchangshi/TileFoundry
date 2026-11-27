"""Six-tuple AscendCausalRule schema and safe AST inequality evaluator."""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path

import yaml

from tilefoundry_ascend.ascend.json_record import JsonRecordMixin, atomic_write_text
from tilefoundry_ascend.ascend.passes.observation_space import (
    AxisResidual5D,
    CompilerSummaryLike,
    ObservationVector5D,
    RooflineSummaryLike,
)
from tilefoundry_ascend.ascend.passes.physical_ledger import ObjectivePhysicalLedger
from tilefoundry_ascend.ascend.sil.schema import VALID_FREEDOM_FACETS, CasePhysicsRatios


class SymptomFamily(str, Enum):
    """Fixed 14-family hardware waste taxonomy shared across diagnosis and rule routing."""

    DMA_REDUNDANT_GM_ROUNDTRIP = "DMA_REDUNDANT_GM_ROUNDTRIP"
    DMA_UNALIGNED_SHORT_BURST = "DMA_UNALIGNED_SHORT_BURST"
    DMA_L2_THRASH_REUSE_MISS = "DMA_L2_THRASH_REUSE_MISS"
    COMPUTE_VECTOR_LOOP_DEGRADED = "COMPUTE_VECTOR_LOOP_DEGRADED"
    COMPUTE_INEFFECTIVE_THROUGHPUT = "COMPUTE_INEFFECTIVE_THROUGHPUT"
    COMPUTE_FIXPIPE_ACCUM_IMBALANCE = "COMPUTE_FIXPIPE_ACCUM_IMBALANCE"
    COMPUTE_1C2V_SUBCORE_IMBALANCE = "COMPUTE_1C2V_SUBCORE_IMBALANCE"
    SCALAR_LOOP_TRAP = "SCALAR_LOOP_TRAP"
    SRAM_REGISTER_SPILL_STACK = "SRAM_REGISTER_SPILL_STACK"
    SRAM_UB_L1_BANK_CONFLICT = "SRAM_UB_L1_BANK_CONFLICT"
    SYNC_PSEUDO_DOUBLE_BUFFER = "SYNC_PSEUDO_DOUBLE_BUFFER"
    SYNC_BARRIER_SERIALIZATION = "SYNC_BARRIER_SERIALIZATION"
    GRID_TAIL_WAVE_UNDERUTIL = "GRID_TAIL_WAVE_UNDERUTIL"
    UNMODELED_SILICON_RESIDUAL = "UNMODELED_SILICON_RESIDUAL"


DETERMINISTIC_SYMPTOM_FAMILIES: frozenset[SymptomFamily] = frozenset(
    f for f in SymptomFamily if f is not SymptomFamily.UNMODELED_SILICON_RESIDUAL
)

_SYMPTOM_SPECS: tuple[tuple[SymptomFamily, str, tuple[str, ...]], ...] = (
    (SymptomFamily.SCALAR_LOOP_TRAP, "compute_issue", ("exec_mode_and_spr", "dma_and_cache_hierarchy")),
    (SymptomFamily.SRAM_REGISTER_SPILL_STACK, "onchip_sram_reg", ("tile_and_stage_knobs", "exec_mode_and_spr")),
    (SymptomFamily.DMA_REDUNDANT_GM_ROUNDTRIP, "offchip_dma", ("dma_and_cache_hierarchy", "sram_layout_and_aliasing", "pipeline_and_sync_graph")),
    (SymptomFamily.DMA_UNALIGNED_SHORT_BURST, "offchip_dma", ("tile_and_stage_knobs", "dma_and_cache_hierarchy")),
    (SymptomFamily.DMA_L2_THRASH_REUSE_MISS, "offchip_dma", ("grid_and_workspace_topology", "dma_and_cache_hierarchy")),
    (SymptomFamily.SYNC_BARRIER_SERIALIZATION, "pipeline_sync", ("pipeline_and_sync_graph", "tile_and_stage_knobs")),
    (SymptomFamily.SYNC_PSEUDO_DOUBLE_BUFFER, "pipeline_sync", ("pipeline_and_sync_graph", "sram_layout_and_aliasing")),
    (SymptomFamily.COMPUTE_INEFFECTIVE_THROUGHPUT, "compute_issue", ("exec_mode_and_spr", "tile_and_stage_knobs")),
    (SymptomFamily.COMPUTE_VECTOR_LOOP_DEGRADED, "compute_issue", ("exec_mode_and_spr", "alg_compute_and_monoid")),
    (SymptomFamily.COMPUTE_1C2V_SUBCORE_IMBALANCE, "compute_issue", ("grid_and_workspace_topology", "pipeline_and_sync_graph")),
    (SymptomFamily.COMPUTE_FIXPIPE_ACCUM_IMBALANCE, "compute_issue", ("tile_and_stage_knobs", "sram_layout_and_aliasing")),
    (SymptomFamily.SRAM_UB_L1_BANK_CONFLICT, "onchip_sram_reg", ("sram_layout_and_aliasing",)),
    (SymptomFamily.GRID_TAIL_WAVE_UNDERUTIL, "grid_wave", ("grid_and_workspace_topology", "tile_and_stage_knobs")),
    (SymptomFamily.UNMODELED_SILICON_RESIDUAL, "unmodeled_residual", VALID_FREEDOM_FACETS),
)
SYMPTOM_FAMILY_TO_AXIS: dict[SymptomFamily, str] = {f: ax for f, ax, _ in _SYMPTOM_SPECS}
SYMPTOM_FAMILY_TO_FACETS: dict[SymptomFamily, tuple[str, ...]] = {f: fc for f, _, fc in _SYMPTOM_SPECS}
_PHYSICAL_PRIORITY_INDEX = {f: idx for idx, (f, _, _) in enumerate(_SYMPTOM_SPECS)}

_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod,
}
_CMP_OPS = {
    ast.Eq: operator.eq, ast.NotEq: operator.ne, ast.Lt: operator.lt, ast.LtE: operator.le,
    ast.Gt: operator.gt, ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b, ast.NotIn: lambda a, b: a not in b,
}
_SAFE_FUNCS = {"abs": abs, "max": max, "min": min}
_ALLOWED_AST_NODES = (
    ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.USub, ast.UAdd,
    ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Compare,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Constant,
    ast.Name, ast.Attribute, ast.Load, ast.Tuple, ast.List, ast.Call,
)


class UnsafePredicateError(ValueError):
    """Raised when a causal rule predicate uses disallowed syntax or symbols."""


def _flatten_attribute(node: ast.AST) -> str | None:
    parts: list[str] = []
    cur: ast.AST = node
    while isinstance(cur, ast.Attribute):
        if cur.attr.startswith("_"):
            raise UnsafePredicateError(f"private attribute access forbidden: {cur.attr!r}")
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        if cur.id.startswith("_"):
            raise UnsafePredicateError(f"private name access forbidden: {cur.id!r}")
        return ".".join(reversed([*parts, cur.id]))
    return None


def validate_rule_predicate(expr: str) -> ast.Expression:
    """Parse and validate a predicate string against the safe AST whitelist."""
    if not isinstance(expr, str) or not expr.strip():
        raise UnsafePredicateError("predicate must be a non-empty string")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise UnsafePredicateError(f"invalid predicate syntax {expr!r}: {exc}") from exc
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_AST_NODES):
            raise UnsafePredicateError(f"disallowed AST node {type(node).__name__} in predicate {expr!r}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _SAFE_FUNCS:
                raise UnsafePredicateError(f"only {sorted(_SAFE_FUNCS)} calls are permitted")
            if node.keywords:
                raise UnsafePredicateError("keyword arguments are not allowed in predicate calls")
        if isinstance(node, ast.Attribute) and _flatten_attribute(node) is None:
            raise UnsafePredicateError("attribute chains must root at an identifier")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float, str, bool, type(None))):
            raise UnsafePredicateError(f"unsupported constant {node.value!r}")
    return tree


def _eval_node(node: ast.AST, env: Mapping[str, object]) -> object:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, env)
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise KeyError(f"unknown predicate variable {node.id!r}")
        return env[node.id]
    if isinstance(node, ast.Attribute):
        if (dotted := _flatten_attribute(node)) is None:
            raise UnsafePredicateError("invalid attribute root")
        if dotted in env:
            return env[dotted]
        parts, cur = dotted.split("."), env
        for idx, part in enumerate(parts):
            if isinstance(cur, Mapping) and part in cur:
                cur = cur[part]
                continue
            if isinstance(cur, Mapping) and (rem := ".".join(parts[idx:])) in cur:
                return cur[rem]
            raise KeyError(f"unknown predicate attribute {dotted!r}")
        return cur
    if isinstance(node, (ast.Tuple, ast.List)):
        vals = [_eval_node(el, env) for el in node.elts]
        return tuple(vals) if isinstance(node, ast.Tuple) else vals
    if isinstance(node, ast.UnaryOp):
        val = _eval_node(node.operand, env)
        if isinstance(node.op, ast.Not):
            return not bool(val)
        return (-float(val) if isinstance(val, float) else -int(val)) if isinstance(node.op, ast.USub) else val
    if isinstance(node, ast.BoolOp):
        is_and = isinstance(node.op, ast.And)
        for val in node.values:
            if (res := bool(_eval_node(val, env))) != is_and:
                return res
        return is_and
    if isinstance(node, ast.BinOp):
        return _BIN_OPS[type(node.op)](_eval_node(node.left, env), _eval_node(node.right, env))
    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, env)
        for op, comp in zip(node.ops, node.comparators):
            if not _CMP_OPS[type(op)](left, right := _eval_node(comp, env)):
                return False
            left = right
        return True
    if isinstance(node, ast.Call):
        return _SAFE_FUNCS[node.func.id](*(_eval_node(arg, env) for arg in node.args))
    raise UnsafePredicateError(f"unsupported AST node {type(node).__name__}")


def safe_eval_predicate(expr: str, env: Mapping[str, object]) -> bool:
    """Safely evaluate a causal rule guard, precondition, or postcondition."""
    return bool(_eval_node(validate_rule_predicate(expr), env))


_BACKEND_ALIASES = {
    "tilelang": "tilelang_dsl", "tilelang_dsl": "tilelang_dsl",
    "inline_cce_vf": "tilelang_dsl", "inline_ascendc_vf": "tilelang_dsl",
    "ascendc": "ascendc_trackb", "ascendc_template": "ascendc_trackb",
    "ascendc_trackb": "ascendc_trackb", "cce": "ascendc_trackb", "cpp20": "ascendc_trackb",
    "cce_cpp20": "ascendc_trackb", "cce_trackb": "ascendc_trackb",
    "tl_templates_cpp20": "ascendc_trackb", "hybrid": "hybrid_trackb", "hybrid_trackb": "hybrid_trackb",
}


def normalize_runtime_backend(backend: str | None) -> str | None:
    """Normalize runtime backend aliases to canonical KB backend identifiers."""
    if backend is None:
        return None
    cleaned = str(backend).strip().lower().replace("-", "_")
    return _BACKEND_ALIASES.get(cleaned, cleaned) if cleaned else None


@dataclass(frozen=True)
class AscendCausalRule(JsonRecordMixin):
    """Six-tuple causal optimization rule grounded in physical hardware telemetry."""

    rule_id: str
    regime_guard: str
    precondition: str
    transform_template: str
    expected_postcondition: str
    source_provenance: str
    alpha_prior: float = 2.0
    beta_prior: float = 1.0
    rationale: str = ""
    priority: int = 100
    scope: str = "cross_op"
    origin_operator: str = ""
    compatible_backends: tuple[str, ...] = ("tilelang_dsl", "ascendc_trackb", "hybrid_trackb")
    cpp20_transform_template: str = ""
    symptom_family: str = ""
    anti_precondition: str = "False"
    mechanism_tag: str = ""
    concept_description: str = ""

    def __post_init__(self) -> None:
        if not self.rule_id or not self.transform_template.strip():
            raise ValueError("rule_id and transform_template must be non-empty")
        if self.scope not in ("cross_op", "operator_local"):
            raise ValueError(f"invalid rule scope {self.scope!r}; expected 'cross_op' or 'operator_local'")
        backends = tuple(dict.fromkeys(n for raw in self.compatible_backends if (n := normalize_runtime_backend(str(raw)))))
        if not backends:
            raise ValueError("compatible_backends must contain at least one backend")
        object.__setattr__(self, "compatible_backends", backends)
        fam = str(self.symptom_family or "").strip()
        if fam:
            valid = tuple(f.value for f in SymptomFamily if f in DETERMINISTIC_SYMPTOM_FAMILIES)
            if fam not in valid:
                raise ValueError(f"invalid symptom_family {fam!r}; expected one of {valid}")
        object.__setattr__(self, "symptom_family", fam)
        object.__setattr__(self, "anti_precondition", str(self.anti_precondition or "False").strip() or "False")
        object.__setattr__(self, "mechanism_tag", str(self.mechanism_tag or "").strip() or self.rule_id.lower())
        object.__setattr__(self, "concept_description", str(self.concept_description or "").strip())
        for pred in (self.regime_guard, self.precondition, self.anti_precondition, self.expected_postcondition):
            validate_rule_predicate(pred)

    def supports_backend(self, runtime_backend: str | None) -> bool:
        """Return whether this rule applies to *runtime_backend* without dialect leakage."""
        norm = normalize_runtime_backend(runtime_backend)
        return True if norm is None else norm in {b for raw in self.compatible_backends if (b := normalize_runtime_backend(str(raw)))}

    def render_for_backend(self, runtime_backend: str | None = None) -> AscendCausalRule:
        """Return rule rendered with Track-B CCE (C++20) guidance on C++20 backends."""
        norm, cpp_guide = normalize_runtime_backend(runtime_backend), self.cpp20_transform_template.strip()
        if not cpp_guide:
            return self
        if norm == "ascendc_trackb":
            return replace(self, transform_template=cpp_guide)
        if norm == "hybrid_trackb" and cpp_guide not in self.transform_template:
            return replace(self, transform_template=f"{self.transform_template.strip()}\n\nTrack-B CCE (C++20):\n{cpp_guide}")
        return self

    def as_six_tuple(self) -> tuple[str, str, str, str, str, str]:
        """Return canonical 6-tuple fields."""
        return (self.rule_id, self.regime_guard, self.precondition, self.transform_template, self.expected_postcondition, self.source_provenance)

    def matches(self, context: Mapping[str, object]) -> bool:
        """Return whether regime_guard and precondition hold while anti_precondition is false."""
        return (
            safe_eval_predicate(self.regime_guard, context)
            and safe_eval_predicate(self.precondition, context)
            and not safe_eval_predicate(self.anti_precondition, context)
        )

    def postcondition_realized(self, context: Mapping[str, object]) -> bool:
        """Return whether expected_postcondition holds in post-edit context."""
        return safe_eval_predicate(self.expected_postcondition, context)

    @classmethod
    def from_dict(cls, entry: Mapping[str, object]) -> AscendCausalRule:
        """Deserialize rule from YAML/JSON mapping."""
        norm = dict(entry)
        raw_b = norm.get("compatible_backends")
        if isinstance(raw_b, str):
            norm["compatible_backends"] = [b.strip() for b in raw_b.split(",") if b.strip()]
        elif isinstance(raw_b, Sequence):
            norm["compatible_backends"] = [str(b).strip() for b in raw_b if str(b).strip()]
        norm["scope"] = norm.get("scope") or "cross_op"
        norm["anti_precondition"] = norm.get("anti_precondition") or "False"
        return super().from_dict(norm)


_DEFAULT_PIPE_KEYS = (
    "AIC.PIPE_M", "AIC.PIPE_FIX", "AIC.PIPE_MTE1", "AIC.PIPE_MTE2", "AIC.PIPE_MTE3", "AIC.PIPE_S",
    "AIV0.PIPE_V", "AIV0.PIPE_MTE1", "AIV0.PIPE_MTE2", "AIV0.PIPE_MTE3", "AIV0.PIPE_S",
    "AIV1.PIPE_V", "AIV1.PIPE_MTE1", "AIV1.PIPE_MTE2", "AIV1.PIPE_MTE3", "AIV1.PIPE_S",
)
_DEFAULT_SCOPES = ("UB", "L1", "L0A", "L0B", "L0C", "SSBUF")
_RESIDUAL_DEFAULTS: dict[str, int | float | bool | str] = {
    "measured_latency_us": 0.0, "predicted_bottleneck_util": 0.0,
    "measured_aic_mac_ratio": 0.0, "measured_aiv_vec_ratio": 0.0,
    "l2_aic_mac_ratio": 0.0, "l2_aiv_vec_ratio": 0.0,
    "aic_mac_ratio": 0.0, "aiv_vec_ratio": 0.0, "l2_pmu_available": 0,
    "measured_mte2_wait_ratio": 0.0, "measured_sync_stall_ratio": 0.0,
    "unexplained_gap_ratio": 0.0, "transfer_inflation_ratio": 1.0, "effective_compute_eta": 0.0,
    "pseudo_double_buffer_serialized": False, "spill_bytes": 0, "stack_bytes": 0,
    "vloopv2_hit": True, "l2_hit_rate": 1.0, "bank_conflict_ratio": 0.0,
    "vf_fusion_breaks": 0, "alias_serialized_buffers": 0, "dual_issue_ipc_collapse": 0.0,
    "dcache_dcci_flushes": 0, "has_local_membar_in_vf_call": False,
}
_TELEMETRY_CONTEXT_DEFAULTS: dict[str, int | float | bool | str] = {
    "num_aic_cores": 28, "num_aiv_cores": 56, "ub_bank_conflict_risk": 0.0,
    "l1_bank_conflict_risk": 0.0, "ub_scalar_compute_loops": 0, "l1_vec_regs": 0,
    "vec_regs": 0, "l1_scalar_regs": 0, "contig_atom_ratio": 1.0,
    "working_set_sram_ratio": 0.0, "scalar_compute_loop_fallback": 0.0,
    "dcache_bypass_narrow_store": 0.0, "pipeline_depth": 1, "has_empty_group": False,
    "has_loop_carried_state": False, "compute_engine": "", "memory_ring": "",
    "iteration_regularity": "",
}
_PMU_L2_FIELD_NAMES = frozenset({
    "l2_aic_mac_ratio", "l2_aiv_vec_ratio", "l2_mte2_wait_ratio", "l2_fixpipe_ratio",
    "l2_cache_hit_rate", "l2_bank_conflict_ratio", "l2_sync_stall_ratio", "l2_pseudo_double_buffer",
})
_EXPLICIT_L2_PMU_KEYS = frozenset({
    "l2_aic_mac_ratio", "l2_aiv_vec_ratio", "measured_aic_mac_ratio", "measured_aiv_vec_ratio",
    "aic_mac_ratio", "aiv_vec_ratio", "aic_mad", "aiv_vec", "measured_mte2_wait_ratio",
    "mte2_wait_ratio", "aiv_mte2", "aic_mte2", "measured_sync_stall_ratio", "sync_stall_ratio",
    "bank_conflict_ratio", "ub_bank_conflict_ratio", "resource_conflict_ratio", "l2_hit_rate",
    "l2_cache_hit_rate", "aic_fixpipe_ratio", "aic_fixpipe", "pseudo_double_buffer_serialized",
})
_AIC_RATIO_KEYS = ("l2_aic_mac_ratio", "measured_aic_mac_ratio", "aic_mac_ratio")
_AIV_RATIO_KEYS = ("l2_aiv_vec_ratio", "measured_aiv_vec_ratio", "aiv_vec_ratio")
_AXIS_KEYS = ("compute_engine", "memory_ring", "iteration_regularity")
_MAX_HAZARD_KEYS = frozenset({
    "spill_bytes", "stack_bytes", "vf_fusion_breaks", "alias_serialized_buffers",
    "dual_issue_ipc_collapse", "dcache_dcci_flushes", "scalar_compute_loop_fallback",
    "dcache_bypass_narrow_store",
})
_MISSING = object()


def _get_field(source: object, key: str, default: object = _MISSING) -> object:
    return source.get(key, default) if isinstance(source, Mapping) else getattr(source, key, default)


def _coerce(raw_val: object, default_val: int | float | bool | str) -> int | float | bool | str:
    if isinstance(default_val, bool):
        return bool(raw_val)
    if isinstance(default_val, int):
        return int(raw_val) if isinstance(raw_val, (int, float, str)) else int(default_val)
    if isinstance(default_val, float):
        return float(raw_val) if isinstance(raw_val, (int, float, str)) else float(default_val)
    return str(raw_val)


def _rf_opt(source: object, *keys: str) -> float | None:
    for key in keys:
        if (raw := _get_field(source, key)) is not _MISSING and not isinstance(raw, bool) and isinstance(raw, (int, float)):
            return float(raw)
    return None


def _rf(source: object, *keys: str, default: float = 0.0) -> float:
    return v if (v := _rf_opt(source, *keys)) is not None else default


def _ri(source: object, *keys: str, default: int = 0) -> int:
    return int(v) if (v := _rf_opt(source, *keys)) is not None else default


def _rb(source: object, *keys: str, default: bool = False) -> bool:
    for key in keys:
        if (raw := _get_field(source, key)) is not _MISSING:
            return bool(raw)
    return default


def _as_int(val: object, default: int = 0) -> int:
    return int(val) if isinstance(val, (bool, int, float, str)) else default


def _as_float(val: object, default: float = 0.0) -> float:
    return float(val) if isinstance(val, (int, float, str)) and not isinstance(val, bool) else default


def _sync_aliases(target: dict[str, object], value: float, keys: tuple[str, ...]) -> None:
    for key in keys:
        target[key] = value


def _apply_topology_axes(target: dict[str, object], source: object | None) -> None:
    if source is not None:
        for ak in _AXIS_KEYS:
            if (val := _get_field(source, ak)) is not _MISSING:
                target[ak] = str(val)


def _extract_typed_defaults(
    source: object, defaults: Mapping[str, int | float | bool | str]
) -> dict[str, int | float | bool | str]:
    return {
        k: _coerce(raw_val, dv) for k, dv in defaults.items() if (raw_val := _get_field(source, k)) is not _MISSING
    }


def _pmu_flag_from_mapping(src: Mapping[str, object]) -> int | None:
    if "l2_pmu_available" in src:
        raw_flag = src["l2_pmu_available"]
        return (1 if int(raw_flag) != 0 else 0) if isinstance(raw_flag, (bool, int, float)) else 0
    return 1 if (not bool(src.get("is_analytical_proxy", False)) and any(k in src for k in _EXPLICIT_L2_PMU_KEYS)) else None


def _resolve_l2_pmu_flag(
    residual: object | None, embedded_obs: ObservationVector5D | None, extra: Mapping[str, object] | None
) -> int:
    available = 0
    if residual is not None:
        if isinstance(residual, Mapping) and "l2_pmu_available" in residual:
            available = _pmu_flag_from_mapping(residual) or 0
        elif embedded_obs is not None:
            if embedded_obs.has_l2_telemetry and not bool(getattr(embedded_obs, "is_analytical_proxy", False)):
                if any(f in _PMU_L2_FIELD_NAMES for f in embedded_obs.measured_l2_fields) or (
                    not embedded_obs.measured_l2_fields
                    and (embedded_obs.compute_issue.l2_aiv_vec_ratio > 0.0 or embedded_obs.compute_issue.l2_aic_mac_ratio > 0.0)
                ):
                    available = 1
        elif isinstance(residual, Mapping):
            available = _pmu_flag_from_mapping(residual) or 0
        elif (
            any(
                float(getattr(residual, k, 0.0) or 0.0) > 0.0
                for k in ("measured_aiv_vec_ratio", "measured_aic_mac_ratio", "measured_mte2_wait_ratio", "measured_sync_stall_ratio", "bank_conflict_ratio")
            )
            or float(getattr(residual, "l2_hit_rate", 1.0) or 1.0) < 1.0
            or bool(getattr(residual, "pseudo_double_buffer_serialized", False))
        ):
            available = 1
    if extra and (extra_flag := _pmu_flag_from_mapping(extra)) is not None:
        available = extra_flag
    return available


def _ledger_to_namespace(ledger: ObjectivePhysicalLedger) -> dict[str, object]:
    ns: dict[str, object] = {key: 0 for key in _DEFAULT_PIPE_KEYS}
    ns.update({key: int(val) for key, val in ledger.pipe_cycles.items()})
    ns["PIPE_M"], ns["PIPE_FIX"] = int(ns["AIC.PIPE_M"]), int(ns["AIC.PIPE_FIX"])
    ns["PIPE_V"] = max(int(ns["AIV0.PIPE_V"]), int(ns["AIV1.PIPE_V"]))
    ns["PIPE_MTE2"] = max(int(ns["AIC.PIPE_MTE2"]), int(ns["AIV0.PIPE_MTE2"]), int(ns["AIV1.PIPE_MTE2"]))
    ns["PIPE_MTE3"] = max(int(ns["AIV0.PIPE_MTE3"]), int(ns["AIV1.PIPE_MTE3"]))
    cube_cycles = int(ledger.pipe_cycles.get("AIC.PIPE_M", 0)) + int(ledger.pipe_cycles.get("AIC.CUBE", 0))
    vec_cycles = int(ledger.pipe_cycles.get("AIV0.PIPE_V", 0)) + int(ledger.pipe_cycles.get("AIV1.PIPE_V", 0))
    ns.update({
        "is_legal": ledger.is_legal, "analysis_mode": ledger.analysis_mode,
        "steady_state_ii": ledger.steady_state_ii if ledger.steady_state_ii is not None else 0,
        "dominant_pipe": ledger.dominant_pipe, "mte2_inflight_per_aiv": ledger.mte2_inflight_per_aiv,
        "unaligned_dma_padding_bytes": ledger.unaligned_dma_padding_bytes,
        "max_overflow_bytes": ledger.max_overflow_bytes, "has_opaque_regions": bool(ledger.opaque_dynamic_regions),
        "cube_cycles": cube_cycles, "vec_cycles": vec_cycles,
        "l0_cube_cycles": cube_cycles, "l0_vec_cycles": vec_cycles,
        "cube_to_vector_ratio": round(cube_cycles / float(max(1, vec_cycles)), 6) if cube_cycles > 0 else 0.0,
        "wave_occupancy": float(ledger.wave_occupancy), "l0_wave_occupancy": float(ledger.wave_occupancy),
        "grid_blocks": int(ledger.grid_blocks), "l0_grid_blocks": int(ledger.grid_blocks),
        "active_grid_blocks": int(ledger.active_grid_blocks),
        "dma_burst_efficiency_ratio": float(ledger.dma_burst_efficiency_ratio),
        "has_simtvf": any("simt_vf" in r.lower() or "simtvf" in r.lower() for r in ledger.opaque_dynamic_regions),
        "has_dynamic_control": bool(ledger.opaque_dynamic_regions),
    })
    for attr in ("in_loop_pipe_all_barriers", "gm_roundtrip_bytes", "dominant_contig_bytes", "effective_cube_m", "in_loop_ub_reload_stores", "min_gm_dma_bytes"):
        ns[attr] = int(getattr(ledger, attr))
    for attr in ("grid_is_symbolic", "has_indirect_gather", "has_lowbit_quant", "has_online_rescale_loop", "uses_unreserved_ubuf", "uses_dynamic_offset_pipelining", "has_unlimit_memory_attr", "has_coupled_reduce_sfu_accum"):
        ns[attr] = bool(getattr(ledger, attr))
    for key, default_val in _TELEMETRY_CONTEXT_DEFAULTS.items():
        ns[key] = _coerce(getattr(ledger, key, default_val), default_val)
    if isinstance(raw_prof := getattr(ledger, "microarch_profile", None), Mapping):
        stage_hint = max(_as_int(raw_prof.get("pipeline_depth"), 1), _as_int(raw_prof.get("num_stages"), 1))
        if stage_hint > 1 or "pipeline_depth" in raw_prof or "num_stages" in raw_prof:
            ns["pipeline_depth"] = ns["num_stages"] = max(1, stage_hint)
        if not ns.get("compute_engine") and raw_prof.get("compute_engine"):
            ns["compute_engine"] = str(raw_prof["compute_engine"])
    total_bloat = degraded_total = any_overflow = 0
    wl_fields = ("capacity_bytes", "explicit_bytes", "implicit_bloat_bytes", "peak_allocated_bytes", "overflow_bytes")
    for scope in _DEFAULT_SCOPES:
        wl = ledger.waterlines.get(scope)
        if wl is not None:
            total_bloat, degraded_total, any_overflow = total_bloat + wl.implicit_bloat_bytes, degraded_total + len(wl.degraded_single_buffers), max(any_overflow, wl.overflow_bytes)
        ns[scope] = {**{k: getattr(wl, k) if wl else 0 for k in wl_fields}, "degraded_count": len(wl.degraded_single_buffers) if wl else 0}
    ub_ns = ns["UB"]
    assert isinstance(ub_ns, dict)
    ub_cap = int(ub_ns["capacity_bytes"])
    ub_ratio = round(float(int(ub_ns["peak_allocated_bytes"])) / float(max(1, ub_cap)), 6) if ub_cap > 0 else 0.0
    ns["ub_utilization_ratio"] = ns["ub_occupancy_ratio"] = ub_ratio
    ns["implicit_bloat_bytes"], ns["degraded_buffer_count"], ns["any_overflow_bytes"] = total_bloat, degraded_total, any_overflow
    return ns


def _embedded_observation(residual: object | None) -> ObservationVector5D | None:
    """Recover the 5D observation carried by a residual object or serialized mapping."""
    if residual is None or isinstance(residual, ObservationVector5D):
        return residual
    raw_obs = _get_field(residual, "observation", None)
    if isinstance(raw_obs, ObservationVector5D):
        return raw_obs
    return ObservationVector5D.from_dict(raw_obs) if isinstance(raw_obs, Mapping) else None


def _extract_extra_signals(source: object | None) -> dict[str, float]:
    out: dict[str, float] = {}
    if source is None:
        return out
    for key in ("scalar_compute_loop_fallback", "dcache_bypass_narrow_store", "ub_scalar_compute_loops"):
        if (val := _rf_opt(source, key)) is not None:
            out[key] = val
    if isinstance(raw_extra := _get_field(source, "extra_signals", None), Mapping):
        for key, val in raw_extra.items():
            if isinstance(val, (int, float)) and not isinstance(val, bool):
                sk = str(key)
                out[sk] = max(out[sk], float(val)) if sk in out else float(val)
    return out


def _merge_extra_signals(target: dict[str, float], incoming: Mapping[str, float]) -> None:
    for key, val in incoming.items():
        target[key] = max(target[key], val) if key in target else val


def _apply_case_ratios(ctx: dict[str, object], raw_ratios: object | None) -> None:
    if isinstance(raw_ratios, CasePhysicsRatios):
        ctx["contig_atom_ratio"], ctx["working_set_sram_ratio"] = float(raw_ratios.contig_atom_ratio), float(raw_ratios.working_set_sram_ratio)
    elif isinstance(raw_ratios, Mapping):
        for key in ("contig_atom_ratio", "working_set_sram_ratio"):
            if (val := _rf_opt(raw_ratios, key)) is not None:
                ctx[key] = val


def _apply_signal_clamps(
    ctx: dict[str, object], extra_signals_map: dict[str, float], *, extra: Mapping[str, object] | None = None
) -> None:
    ctx["extra_signals"] = dict(extra_signals_map)
    extra_ub = _rf_opt(extra, "ub_scalar_compute_loops") if extra is not None else None
    ub_loops = max(0, int(extra_ub)) if extra_ub is not None else max(_as_int(ctx.get("ub_scalar_compute_loops")), int(extra_signals_map.get("ub_scalar_compute_loops", 0)))
    ctx["ub_scalar_compute_loops"] = ub_loops
    ctx["scalar_compute_loop_fallback"] = max(_as_float(ctx.get("scalar_compute_loop_fallback")), float(extra_signals_map.get("scalar_compute_loop_fallback", 0.0)), float(ub_loops))
    ctx["dcache_bypass_narrow_store"] = max(_as_float(ctx.get("dcache_bypass_narrow_store")), float(extra_signals_map.get("dcache_bypass_narrow_store", 0.0)))


def _populate_state_context(
    ledger: ObjectivePhysicalLedger, *, features: object | None = None, residual: object | None = None,
    compiler_summary: CompilerSummaryLike | Mapping[str, object] | None = None,
    extra: Mapping[str, object] | None = None, target: object | None = None,
    roofline: RooflineSummaryLike | Mapping[str, object] | None = None, case_ratios: object | None = None,
) -> tuple[dict[str, object], dict[str, float], ObservationVector5D | None]:
    ctx = _ledger_to_namespace(ledger)
    if features is not None:
        if hasattr(features, "to_dict"):
            ctx.update(features.to_dict())
        elif isinstance(features, Mapping):
            ctx.update(features)
        for key in (*_AXIS_KEYS, "has_empty_group", "has_loop_carried_state"):
            if hasattr(features, key):
                ctx[key] = getattr(features, key)
        _apply_topology_axes(ctx, getattr(features, "topology_axes", None))
    if target is not None:
        dev = getattr(target, "device", target)
        for attr, k in (("aic_count", "num_aic_cores"), ("aiv_count", "num_aiv_cores")):
            if hasattr(dev, attr):
                ctx[k] = int(getattr(dev, attr))

    res_defaults: dict[str, int | float | bool | str] = dict(_RESIDUAL_DEFAULTS)
    extra_signals_map: dict[str, float] = {}
    has_l1_summary = compiler_summary is not None
    if compiler_summary is not None:
        comp_vals = _extract_typed_defaults(compiler_summary, _RESIDUAL_DEFAULTS)
        sw, vl = _as_int(_get_field(compiler_summary, "software_loops", 0)), _as_int(_get_field(compiler_summary, "vloopv2_loops", 0))
        ctx["l1_vec_regs"] = ctx["vec_regs"] = _as_int(_get_field(compiler_summary, "vec_regs", 0))
        ctx["l1_scalar_regs"] = _as_int(_get_field(compiler_summary, "scalar_regs", 0))
        _merge_extra_signals(extra_signals_map, _extract_extra_signals(compiler_summary))
        if sw == 0 and vl == 0 and "vloopv2_hit" in comp_vals:
            comp_vals["vloopv2_hit"] = True
        res_defaults.update(comp_vals)

    embedded_obs: ObservationVector5D | None = None
    if residual is not None:
        res_vals = _extract_typed_defaults(residual, _RESIDUAL_DEFAULTS)
        if has_l1_summary:
            for key, val in res_vals.items():
                if key in _MAX_HAZARD_KEYS:
                    cur = res_defaults[key]
                    res_defaults[key] = max(_as_float(cur), _as_float(val)) if isinstance(cur, float) or isinstance(val, float) else max(_as_int(cur), _as_int(val))
                elif key == "vloopv2_hit":
                    res_defaults[key] = bool(res_defaults[key]) and bool(val)
                elif key == "has_local_membar_in_vf_call":
                    res_defaults[key] = bool(res_defaults[key]) or bool(val)
                else:
                    res_defaults[key] = val
        else:
            res_defaults.update(res_vals)
        ctx.update(_extract_typed_defaults(residual, _TELEMETRY_CONTEXT_DEFAULTS))
        embedded_obs = _embedded_observation(residual)
        for keys, obs_attr in ((_AIC_RATIO_KEYS, "l2_aic_mac_ratio"), (_AIV_RATIO_KEYS, "l2_aiv_vec_ratio")):
            val = _rf_opt(residual, *keys)
            if val is None and embedded_obs is not None:
                val = float(getattr(embedded_obs.compute_issue, obs_attr))
            if val is not None:
                for k in keys:
                    res_defaults[k] = val
        if embedded_obs is not None:
            obs_payload = embedded_obs.to_dict()
            for key, ax_key in (("transfer_inflation_ratio", "offchip_dma"), ("effective_compute_eta", "compute_issue")):
                if _get_field(residual, key) is _MISSING:
                    ax_p = obs_payload.get(ax_key)
                    raw_val = obs_payload.get(key) or (ax_p.get(key, ax_p.get(f"l2_{key}")) if isinstance(ax_p, Mapping) else None)
                    if isinstance(raw_val, (int, float)) and not isinstance(raw_val, bool):
                        res_defaults[key] = float(raw_val)
            for sig_k, src_val in (
                ("ub_scalar_compute_loops", _as_float(getattr(embedded_obs.compute_issue, "l0_ub_scalar_compute_loops", 0))),
                ("scalar_compute_loop_fallback", _as_float(getattr(embedded_obs.compute_issue, "l1_scalar_compute_loop_fallback", 0.0))),
                ("dcache_bypass_narrow_store", _as_float(getattr(embedded_obs.offchip_dma, "l1_dcache_bypass_narrow_store", 0.0))),
            ):
                if src_val > 0.0:
                    _merge_extra_signals(extra_signals_map, {sig_k: src_val})
            if not has_l1_summary and embedded_obs.has_l1_compiler_summary:
                has_l1_summary = True
                ctx["l1_vec_regs"] = ctx["vec_regs"] = int(embedded_obs.onchip_sram_reg.l1_vec_regs)
                ctx["l1_scalar_regs"] = int(embedded_obs.onchip_sram_reg.l1_scalar_regs)
                res_defaults["vloopv2_hit"] = bool(res_defaults["vloopv2_hit"]) and not (embedded_obs.compute_issue.l1_software_loops > 0 and embedded_obs.compute_issue.l1_vloopv2_loops == 0)
                res_defaults["has_local_membar_in_vf_call"] = bool(res_defaults["has_local_membar_in_vf_call"]) or bool(embedded_obs.pipeline_sync.l1_local_membar_in_vf)
                res_defaults["dual_issue_ipc_collapse"] = max(_as_float(res_defaults["dual_issue_ipc_collapse"]), float(embedded_obs.compute_issue.l1_dual_issue_collapse))
                for rk, sv in (
                    ("spill_bytes", embedded_obs.onchip_sram_reg.l1_spill_bytes), ("stack_bytes", embedded_obs.onchip_sram_reg.l1_stack_bytes),
                    ("vf_fusion_breaks", embedded_obs.compute_issue.l1_vf_fusion_breaks),
                    ("alias_serialized_buffers", embedded_obs.pipeline_sync.l1_alias_serialized_buffers),
                    ("dcache_dcci_flushes", embedded_obs.offchip_dma.l1_dcache_dcci_flushes),
                ):
                    res_defaults[rk] = max(_as_int(res_defaults[rk]), int(sv))
        _merge_extra_signals(extra_signals_map, _extract_extra_signals(residual))
    elif roofline is not None and (static_obs := ledger.to_observation_5d(roofline=roofline)).transfer_inflation_ratio > 0.0:
        res_defaults["transfer_inflation_ratio"] = float(static_obs.transfer_inflation_ratio)

    ctx["has_l1_compiler_summary"] = has_l1_summary
    ctx.update(res_defaults)
    residuals = (embedded_obs or ledger.to_observation_5d(roofline=roofline)).compute_axis_residuals()
    ctx.update({
        "compute_issue_gap": residuals.compute_issue_gap, "offchip_dma_gap": residuals.offchip_dma_gap,
        "onchip_sram_reg_gap": residuals.onchip_sram_reg_gap, "pipeline_sync_gap": residuals.pipeline_sync_gap,
        "grid_wave_gap": residuals.grid_wave_gap, "dominant_axis": residuals.dominant_axis,
    })
    if _as_int(ctx.get("vec_regs")) == 0 and _as_int(ctx.get("l1_vec_regs")) > 0:
        ctx["vec_regs"] = _as_int(ctx["l1_vec_regs"])
    _apply_case_ratios(ctx, case_ratios)
    _apply_signal_clamps(ctx, extra_signals_map)
    ctx["l2_pmu_available"] = _resolve_l2_pmu_flag(residual, embedded_obs, extra)
    obs_stages = max(1, _as_int(getattr(embedded_obs, "num_stages", 1), 1)) if embedded_obs is not None else 1
    if embedded_obs is not None:
        if not ctx.get("compute_engine") and embedded_obs.compute_engine:
            ctx["compute_engine"] = embedded_obs.compute_engine
        if _as_int(ctx.get("pipeline_depth"), 1) <= 1 and obs_stages > 1:
            ctx["pipeline_depth"] = obs_stages
        ctx["num_stages"] = max(_as_int(ctx.get("pipeline_depth"), 1), obs_stages)
    else:
        ctx.setdefault("num_stages", max(1, _as_int(ctx.get("pipeline_depth"), 1)))
    ctx["has_l2_pmu"] = bool(ctx["l2_pmu_available"]) or bool(getattr(embedded_obs, "has_l2_pmu", False))
    if _as_float(ctx.get("ub_bank_conflict_risk")) == 0.0 and _as_float(ctx.get("bank_conflict_ratio")) > 0.0:
        ctx["ub_bank_conflict_risk"] = _as_float(ctx["bank_conflict_ratio"])
    return ctx, extra_signals_map, embedded_obs


def build_predicate_context(
    ledger: ObjectivePhysicalLedger, *, features: object | None = None,
    residual: Mapping[str, object] | object | None = None,
    compiler_summary: CompilerSummaryLike | Mapping[str, object] | None = None,
    prev_ledger: ObjectivePhysicalLedger | None = None, prev_residual: object | None = None,
    prev_compiler_summary: object | None = None, prev_case_ratios: object | None = None,
    extra: Mapping[str, object] | None = None, target: object | None = None,
    roofline: RooflineSummaryLike | Mapping[str, object] | None = None, case_ratios: object | None = None,
) -> dict[str, object]:
    """Assemble a flat and dotted-path dictionary for safe rule predicate evaluation.

    Both the current and previous levels expose L0, L1, L2, and 5D axis gaps.
    """
    ctx, extra_signals_map, embedded_obs = _populate_state_context(
        ledger, features=features, residual=residual, compiler_summary=compiler_summary,
        extra=extra, target=target, roofline=roofline, case_ratios=case_ratios,
    )
    if any(x is not None for x in (prev_ledger, prev_residual, prev_compiler_summary, prev_case_ratios)):
        prev_ns, _, _ = _populate_state_context(
            prev_ledger if prev_ledger is not None else ledger,
            residual=prev_residual, compiler_summary=prev_compiler_summary,
            target=target, roofline=roofline, case_ratios=prev_case_ratios,
        )
        ctx["prev"] = prev_ns
    else:
        ctx["prev"] = dict(ctx)
    if extra:
        ctx.update(extra)
        for src_k, dst_k, fn in (
            ("ub_occupancy_ratio", "ub_utilization_ratio", _as_float),
            ("ub_utilization_ratio", "ub_occupancy_ratio", _as_float),
            ("vec_regs", "l1_vec_regs", _as_int), ("l1_vec_regs", "vec_regs", _as_int),
            ("case_contig_atom_ratio", "contig_atom_ratio", _as_float),
            ("case_working_set_sram_ratio", "working_set_sram_ratio", _as_float),
        ):
            if src_k in extra and dst_k not in extra:
                ctx[dst_k] = fn(extra[src_k])
        if "case_ratios" in extra:
            _apply_case_ratios(ctx, extra.get("case_ratios"))
        _merge_extra_signals(extra_signals_map, _extract_extra_signals(extra))
        _apply_signal_clamps(ctx, extra_signals_map, extra=extra)
        if (eaiv := _rf_opt(extra, "aiv_vec_ratio", "l2_aiv_vec_ratio", "measured_aiv_vec_ratio")) is not None:
            _sync_aliases(ctx, eaiv, _AIV_RATIO_KEYS)
        if (eaic := _rf_opt(extra, "aic_mac_ratio", "l2_aic_mac_ratio", "measured_aic_mac_ratio")) is not None:
            _sync_aliases(ctx, eaic, _AIC_RATIO_KEYS)
        ctx["l2_pmu_available"] = _resolve_l2_pmu_flag(residual, embedded_obs, extra)
        ctx["has_l2_pmu"] = bool(ctx["l2_pmu_available"]) or (embedded_obs is not None and bool(getattr(embedded_obs, "has_l2_pmu", False)))
        _apply_topology_axes(ctx, extra.get("topology_axes"))
    return ctx


def _observation_to_signal_dict(obs: ObservationVector5D, *, case_ratios: object | None = None) -> dict[str, object]:
    p = obs.to_dict()
    cm = v if isinstance(v := p.get("compute_issue"), Mapping) else {}
    dm = v if isinstance(v := p.get("offchip_dma"), Mapping) else {}
    sm = v if isinstance(v := p.get("onchip_sram_reg"), Mapping) else {}
    ym = v if isinstance(v := p.get("pipeline_sync"), Mapping) else {}
    gm = v if isinstance(v := p.get("grid_irregularity"), Mapping) else {}
    sw_loops, vl_loops = _ri(cm, "l1_software_loops"), _ri(cm, "l1_vloopv2_loops")
    vec_cycles, cube_cycles = _ri(cm, "l0_vec_cycles"), _ri(cm, "l0_cube_cycles")
    fixpipe_cycles = _ri(cm, "l0_fixpipe_cycles", default=_ri(dm, "l0_fixpipe_cycles", default=_as_int(getattr(obs.compute_issue, "l0_fixpipe_cycles", getattr(obs.offchip_dma, "l0_fixpipe_cycles", 0)))))
    ub_scalar_loops = _ri(cm, "l0_ub_scalar_compute_loops", default=_as_int(getattr(obs.compute_issue, "l0_ub_scalar_compute_loops", 0)))
    scalar_fb = _rf(cm, "l1_scalar_compute_loop_fallback", default=_as_float(getattr(obs.compute_issue, "l1_scalar_compute_loop_fallback", 0.0)))
    has_pmu = bool(getattr(obs, "has_l2_pmu", False)) or (obs.has_l2_telemetry and not bool(getattr(obs, "is_analytical_proxy", False)) and any(f in _PMU_L2_FIELD_NAMES for f in obs.measured_l2_fields))
    obs_stages, dyn_pipe = max(1, _as_int(getattr(obs, "num_stages", 1), 1)), _rb(ym, "l0_uses_dynamic_offset_pipe")
    l1_vec_regs, wave_occ, grid_blocks = _ri(sm, "l1_vec_regs"), _rf(gm, "l0_wave_occupancy", default=1.0), _ri(gm, "l0_grid_blocks")
    out: dict[str, object] = {
        "transfer_inflation_ratio": _rf(p, "transfer_inflation_ratio", default=_rf(dm, "transfer_inflation_ratio", "l2_transfer_inflation_ratio", default=1.0)),
        "effective_compute_eta": _rf(p, "effective_compute_eta", default=_rf(cm, "effective_compute_eta", "l2_effective_compute_eta")),
        "roofline_eta": _rf(cm, "l2_roofline_eta"), "gm_roundtrip_bytes": _ri(dm, "l0_gm_roundtrip_bytes"),
        "cube_to_vector_ratio": _rf(cm, "l0_cube_to_vec_ratio"), "dominant_contig_bytes": _ri(dm, "l0_dominant_contig_bytes"),
        "dma_burst_efficiency_ratio": _rf(dm, "l0_dma_burst_efficiency", default=1.0),
        "unaligned_dma_padding_bytes": _ri(dm, "l0_unaligned_padding_bytes"),
        "min_gm_dma_bytes": _ri(dm, "l0_min_gm_dma_bytes", default=_as_int(getattr(obs.offchip_dma, "l0_min_gm_dma_bytes", 0))),
        "mte2_inflight_per_aiv": _ri(dm, "l0_mte2_inflight_per_aiv"), "l2_hit_rate": _rf(dm, "l2_cache_hit_rate", default=1.0),
        "measured_mte2_wait_ratio": _rf(dm, "l2_mte2_wait_ratio"), "vloopv2_hit": not (sw_loops > 0 and vl_loops == 0),
        "vf_fusion_breaks": _ri(cm, "l1_vf_fusion_breaks"), "dual_issue_ipc_collapse": _rf(cm, "l1_dual_issue_collapse"),
        "aic_fixpipe_ratio": _rf(dm, "l2_fixpipe_ratio"),
        "AIC.PIPE_M": cube_cycles, "AIC.PIPE_FIX": fixpipe_cycles, "PIPE_FIX": fixpipe_cycles,
        "AIV0.PIPE_V": vec_cycles, "AIV1.PIPE_V": 0 if _rb(cm, "l0_aiv_subcore_imbalance") else vec_cycles,
        "l0_cube_cycles": cube_cycles, "l0_vec_cycles": vec_cycles,
        "spill_bytes": _ri(sm, "l1_spill_bytes"), "stack_bytes": _ri(sm, "l1_stack_bytes"),
        "l1_vec_regs": l1_vec_regs, "vec_regs": l1_vec_regs,
        "contig_atom_ratio": _rf(p, "contig_atom_ratio", default=_rf(dm, "contig_atom_ratio", default=_as_float(getattr(obs, "contig_atom_ratio", getattr(obs.offchip_dma, "contig_atom_ratio", 1.0)), 1.0))),
        "working_set_sram_ratio": _rf(p, "working_set_sram_ratio", default=_rf(sm, "working_set_sram_ratio", default=_as_float(getattr(obs, "working_set_sram_ratio", getattr(obs.onchip_sram_reg, "working_set_sram_ratio", 0.0))))),
        "scalar_compute_loop_fallback": max(scalar_fb, float(ub_scalar_loops)), "ub_scalar_compute_loops": ub_scalar_loops,
        "dcache_bypass_narrow_store": _rf(dm, "l1_dcache_bypass_narrow_store", default=_as_float(getattr(obs.offchip_dma, "l1_dcache_bypass_narrow_store", 0.0))),
        "compute_engine": str(getattr(obs, "compute_engine", "") or ""),
        "num_stages": obs_stages, "pipeline_depth": obs_stages,
        "has_dynamic_double_buffer": dyn_pipe, "uses_dynamic_offset_pipelining": dyn_pipe,
        "has_l2_pmu": has_pmu, "l2_pmu_available": 1 if has_pmu else 0,
        "max_overflow_bytes": _ri(sm, "l0_max_overflow_bytes"), "degraded_buffer_count": _ri(sm, "l0_degraded_single_buffers"),
        "bank_conflict_ratio": _rf(sm, "l2_bank_conflict_ratio"),
        "ub_bank_conflict_risk": _rf(sm, "l0_ub_bank_conflict_risk"), "l1_bank_conflict_risk": _rf(sm, "l0_l1_bank_conflict_risk"),
        "pseudo_double_buffer_serialized": _rb(ym, "l2_pseudo_double_buffer"),
        "alias_serialized_buffers": _ri(ym, "l1_alias_serialized_buffers"),
        "measured_sync_stall_ratio": _rf(ym, "l2_sync_stall_ratio"),
        "in_loop_pipe_all_barriers": _ri(ym, "l0_in_loop_pipe_all_barriers"),
        "has_local_membar_in_vf_call": _rb(ym, "l1_local_membar_in_vf"),
        "wave_occupancy": wave_occ, "l0_wave_occupancy": wave_occ,
        "grid_blocks": grid_blocks, "l0_grid_blocks": grid_blocks,
        "num_aic_cores": 28, "num_aiv_cores": 56, "grid_is_symbolic": False,
        "has_l1_compiler_summary": _rb(p, "has_l1_compiler_summary"),
        "measured_latency_us": _rf(gm, "l2_measured_latency_us"),
    }
    if case_ratios is not None:
        _apply_case_ratios(out, case_ratios)
    obs_aiv, obs_aic = _rf(cm, "l2_aiv_vec_ratio"), _rf(cm, "l2_aic_mac_ratio")
    if obs_aiv > 0.0 or obs_aic > 0.0 or "l2_aiv_vec_ratio" in obs.measured_l2_fields or "l2_aic_mac_ratio" in obs.measured_l2_fields:
        _sync_aliases(out, obs_aiv, _AIV_RATIO_KEYS)
        _sync_aliases(out, obs_aic, _AIC_RATIO_KEYS)
    return out


def _resolve_simplex_compute_engine(
    sig: Mapping[str, object], *, topology_axes: object | None, l0_cube_cycles: int,
    l0_vec_cycles: int, cube_ratio: float, aic_mac: float, aiv_vec: float, pipe_fix: int,
) -> str:
    """Resolve canonical compute_engine from topology_axes, signals, or pipe activity."""
    valid_engines = ("aiv_only", "aic_only", "coupled_1c2v")
    for source in (topology_axes, sig.get("topology_axes")):
        if source is not None and (cand := str(_get_field(source, "compute_engine", "") or "").strip()) in valid_engines:
            return cand
    if (raw_ce := str(sig.get("compute_engine", "") or "").strip()) in valid_engines:
        return raw_ce
    has_cube = l0_cube_cycles > 0 or cube_ratio > 0.0 or aic_mac > 0.0 or pipe_fix > 0
    has_vec = l0_vec_cycles > 0 or aiv_vec > 0.0
    return "coupled_1c2v" if (has_cube and has_vec) else ("aic_only" if has_cube else ("aiv_only" if has_vec else ""))


def _resolve_simplex_has_l2_pmu(
    sig: Mapping[str, object], *, aiv_vec: float, aic_mac: float, mte2_wait: float,
    sync_stall: float, bank_conflict: float, fixpipe_ratio: float, l2_hit: float,
) -> bool:
    """Determine whether real L2 PMU counters are present in the signal payload."""
    if _rb(sig, "is_analytical_proxy"):
        return False
    if "has_l2_pmu" in sig:
        return bool(sig["has_l2_pmu"])
    if "l2_pmu_available" in sig:
        raw_flag = sig["l2_pmu_available"]
        return bool(int(raw_flag)) if isinstance(raw_flag, (int, float)) and not isinstance(raw_flag, bool) else bool(raw_flag)
    has_pos = aiv_vec > 0.0 or aic_mac > 0.0 or mte2_wait > 0.0 or sync_stall > 0.0 or bank_conflict > 0.0 or fixpipe_ratio > 0.0 or l2_hit < 1.0
    return any(key in sig for key in (_EXPLICIT_L2_PMU_KEYS | {"aic_mac"})) and has_pos


def normalize_symptom_simplex(
    raw_weights: Mapping[SymptomFamily, float], *, roofline_eta: float = 0.0, has_l2_pmu: bool = False
) -> dict[SymptomFamily, float]:
    """Normalize deterministic hardware symptoms independently of (1 - eta)."""
    _ = has_l2_pmu
    shares: dict[SymptomFamily, float] = {fam: 0.0 for fam in SymptomFamily}
    det_weights = {
        fam: (max(0.0, v) if math.isfinite(v) else 0.0)
        for fam in DETERMINISTIC_SYMPTOM_FAMILIES
        for v in (float(raw_weights.get(fam, 0.0) or 0.0),)
    }
    s_raw = sum(det_weights.values())
    safe_eta = float(roofline_eta) if math.isfinite(roofline_eta) else 0.0
    if s_raw > 1e-12:
        for fam, weight in det_weights.items():
            shares[fam] = min(1.0, max(0.0, weight / s_raw))
        shares[SymptomFamily.UNMODELED_SILICON_RESIDUAL] = 0.0
        if abs(diff := 1.0 - sum(shares.values())) > 0.0:
            top = max(DETERMINISTIC_SYMPTOM_FAMILIES, key=lambda fam: shares[fam])
            shares[top] = min(1.0, max(0.0, shares[top] + diff))
    else:
        shares[SymptomFamily.UNMODELED_SILICON_RESIDUAL] = 1.0 if safe_eta < 1.0 else 0.0
    return shares


def compute_symptom_simplex(
    obs_or_context: ObservationVector5D | Mapping[str, object], *,
    case_ratios: object | None = None, topology_axes: object | None = None,
) -> list[tuple[SymptomFamily, float]]:
    """Compute topology-gated simplex shares over all 14 SymptomFamily categories."""
    if isinstance(obs_or_context, ObservationVector5D):
        sig: dict[str, object] = _observation_to_signal_dict(obs_or_context, case_ratios=case_ratios)
    else:
        sig = dict(obs_or_context)
        if case_ratios is not None:
            _apply_case_ratios(sig, case_ratios)

    t_inf, eff_eta, roof_eta = _rf(sig, "transfer_inflation_ratio", default=1.0), _rf(sig, "effective_compute_eta"), _rf(sig, "roofline_eta")
    gm_rt, cube_r, contig_b = _ri(sig, "gm_roundtrip_bytes"), _rf(sig, "cube_to_vector_ratio"), _ri(sig, "dominant_contig_bytes")
    burst_e, unalign_p, mte2_in = _rf(sig, "dma_burst_efficiency_ratio", default=1.0), _ri(sig, "unaligned_dma_padding_bytes"), _ri(sig, "mte2_inflight_per_aiv")
    l2_hit, mte2_wait = _rf(sig, "l2_hit_rate", default=1.0), _rf(sig, "measured_mte2_wait_ratio", "mte2_wait_ratio")
    vloop_hit, vf_breaks, dual_col = _rb(sig, "vloopv2_hit", default=True), _ri(sig, "vf_fusion_breaks"), _rf(sig, "dual_issue_ipc_collapse")
    aiv_vec, aic_mac, fix_r = _rf(sig, "measured_aiv_vec_ratio", "aiv_vec_ratio"), _rf(sig, "measured_aic_mac_ratio", "aic_mac_ratio"), _rf(sig, "aic_fixpipe_ratio", "aic_fixpipe")
    pipe_m, pipe_fix, aiv0_v, aiv1_v = _ri(sig, "AIC.PIPE_M", "PIPE_M"), _ri(sig, "AIC.PIPE_FIX", "PIPE_FIX"), _ri(sig, "AIV0.PIPE_V"), _ri(sig, "AIV1.PIPE_V")
    l0_cube = _ri(sig, "l0_cube_cycles", "AIC.PIPE_M", "PIPE_M")
    l0_vec = _ri(sig, "l0_vec_cycles", default=(aiv0_v + aiv1_v) if (aiv0_v or aiv1_v) else _ri(sig, "PIPE_V"))
    l1_vregs, has_l1 = _ri(sig, "l1_vec_regs", "vec_regs"), _rb(sig, "has_l1_compiler_summary")
    extra_map = raw_extra if isinstance(raw_extra := sig.get("extra_signals"), Mapping) else {}
    scalar_fb = _rf(sig, "scalar_compute_loop_fallback", default=_rf(extra_map, "scalar_compute_loop_fallback"))
    dcache_amp = _rf(sig, "dcache_bypass_narrow_store", default=_rf(extra_map, "dcache_bypass_narrow_store"))
    spill_b, stack_b = _ri(sig, "spill_bytes"), _ri(sig, "stack_bytes", "l1_stack_bytes")
    max_ovf, deg_bufs = _ri(sig, "max_overflow_bytes", "any_overflow_bytes"), _ri(sig, "degraded_buffer_count")
    bank_c, ub_risk, l1_risk = _rf(sig, "bank_conflict_ratio"), _rf(sig, "ub_bank_conflict_risk"), _rf(sig, "l1_bank_conflict_risk")
    pseudo_db, alias_ser = _rb(sig, "pseudo_double_buffer_serialized"), _ri(sig, "alias_serialized_buffers")
    sync_stall, pipe_all, local_mb = _rf(sig, "measured_sync_stall_ratio", "sync_stall_ratio"), _ri(sig, "in_loop_pipe_all_barriers"), _rb(sig, "has_local_membar_in_vf_call")
    wave_occ, grid_b = _rf(sig, "wave_occupancy", default=1.0), _ri(sig, "grid_blocks")
    num_aic, num_aiv, grid_sym = max(1, _ri(sig, "num_aic_cores", default=28)), max(1, _ri(sig, "num_aiv_cores", default=56)), _rb(sig, "grid_is_symbolic")

    ce = _resolve_simplex_compute_engine(sig, topology_axes=topology_axes, l0_cube_cycles=l0_cube, l0_vec_cycles=l0_vec, cube_ratio=cube_r, aic_mac=aic_mac, aiv_vec=aiv_vec, pipe_fix=pipe_fix)
    num_stages = _ri(sig, "num_stages", "pipeline_depth", default=1)
    has_dyn_db = _rb(sig, "has_dynamic_double_buffer") or _rb(sig, "uses_dynamic_offset_pipelining") or _rb(sig, "l0_uses_dynamic_offset_pipe")
    has_l2_pmu = _resolve_simplex_has_l2_pmu(sig, aiv_vec=aiv_vec, aic_mac=aic_mac, mte2_wait=mte2_wait, sync_stall=sync_stall, bank_conflict=bank_c, fixpipe_ratio=fix_r, l2_hit=l2_hit)

    inf_s = min(1.0, 0.35 + 0.65 * min(1.0, (t_inf - 1.30) / 1.70)) if t_inf > 1.30 else 0.0
    rt_s = 0.65 if (gm_rt > 0 and cube_r > 0.0) else (0.45 if gm_rt > 0 else 0.0)
    byp_s = min(1.0, 0.85 + 0.15 * min(1.0, (dcache_amp - 1.0) / 31.0)) if dcache_amp > 1.0 else 0.0
    l2_s = 0.0
    if t_inf <= 1.30:
        if l2_hit < 0.65:
            l2_s = min(1.0, (1.0 - l2_hit) * (1.0 if mte2_wait >= 0.15 else 0.85))
        elif l2_hit < 0.80 and mte2_wait > 0.20:
            l2_s = min(1.0, max(1.0 - l2_hit, mte2_wait * 0.8))

    comp_u, eta_obs = max(aiv_vec, aic_mac), (eff_eta if eff_eta > 0.0 else roof_eta)
    fix_s = min(1.0, max(0.35, (float(pipe_fix) / float(pipe_m)) - 1.0)) if (pipe_m > 0 and pipe_fix > 1.15 * float(pipe_m)) else (min(1.0, fix_r) if (fix_r > 0.45 and aic_mac < 0.45) else 0.0)
    pseudo_s = min(1.0, max(0.75 if pseudo_db else 0.0, min(1.0, 0.30 * float(alias_ser)))) if (num_stages > 1 or has_dyn_db) else 0.0
    wave_w = (1.0 - wave_occ / float(math.ceil(wave_occ))) if wave_occ > 0.0 else 0.0
    act_c = num_aic if cube_r > 0.0 else num_aiv
    under_s = max(0.0, 1.0 - float(grid_b) / float(act_c)) if (not grid_sym and wave_occ > 0.0 and 0 < grid_b < act_c) else 0.0
    has_simtvf = _rb(sig, "has_simtvf")
    if has_l2_pmu:
        scalar_s = (min(0.40, 0.15 * scalar_fb) if scalar_fb > 0.0 else 0.0) if (aiv_vec >= 0.08 or aic_mac >= 0.05) else (1.0 if (scalar_fb > 0.0 or (aiv_vec < 0.05 and aic_mac < 0.02)) else 0.0)
    elif scalar_fb > 0.0:
        scalar_s = min(0.55, 0.25 * scalar_fb) if (l0_cube > 0 or l0_vec > 0 or has_simtvf) else 1.0
    else:
        scalar_s = 1.0 if ((l0_cube == 0 and l0_vec == 0 and not has_simtvf) or (has_l1 and l1_vregs == 0)) else 0.0

    raw_weights: dict[SymptomFamily, float] = {
        SymptomFamily.DMA_REDUNDANT_GM_ROUNDTRIP: min(1.0, max(0.0, inf_s, rt_s)),
        SymptomFamily.DMA_UNALIGNED_SHORT_BURST: min(1.0, max(0.0, 1.0 - burst_e, 0.55 if (0 < contig_b < 16384) else 0.0, 0.65 if unalign_p > 0 else 0.0, 0.70 if mte2_in > 16 else 0.0, byp_s)),
        SymptomFamily.DMA_L2_THRASH_REUSE_MISS: max(0.0, l2_s),
        SymptomFamily.COMPUTE_VECTOR_LOOP_DEGRADED: min(1.0, max(0.0, 0.65 if not vloop_hit else 0.0, min(1.0, 0.30 * float(vf_breaks)), dual_col if dual_col > 0.25 else 0.0)),
        SymptomFamily.COMPUTE_INEFFECTIVE_THROUGHPUT: min(1.0, (1.0 - eta_obs) * comp_u) if (comp_u > 0.55 and 0.0 < eta_obs < 0.30) else 0.0,
        SymptomFamily.COMPUTE_FIXPIPE_ACCUM_IMBALANCE: max(0.0, fix_s),
        SymptomFamily.COMPUTE_1C2V_SUBCORE_IMBALANCE: (0.80 if cube_r > 0.0 else 0.70) if (aiv0_v > 0 and aiv1_v == 0) else 0.0,
        SymptomFamily.SCALAR_LOOP_TRAP: max(0.0, scalar_s),
        SymptomFamily.SRAM_REGISTER_SPILL_STACK: min(1.0, max(0.0, min(1.0, float(spill_b) / 256.0) if spill_b > 0 else 0.0, 0.55 if stack_b > 0 else 0.0, 0.85 if max_ovf > 0 else 0.0, 0.65 if deg_bufs > 0 else 0.0)),
        SymptomFamily.SRAM_UB_L1_BANK_CONFLICT: min(1.0, max(0.0, bank_c if bank_c > 0.08 else 0.0, ub_risk, l1_risk)) if spill_b == 0 else 0.0,
        SymptomFamily.SYNC_PSEUDO_DOUBLE_BUFFER: max(0.0, pseudo_s),
        SymptomFamily.SYNC_BARRIER_SERIALIZATION: min(1.0, max(0.0, sync_stall if sync_stall > 0.20 else 0.0, 0.65 if local_mb else 0.0, 0.55 if pipe_all > 0 else 0.0)) if pseudo_s == 0.0 else 0.0,
        SymptomFamily.GRID_TAIL_WAVE_UNDERUTIL: min(1.0, max(0.0, wave_w if wave_w > 0.18 else 0.0, under_s if under_s > 0.18 else 0.0)),
    }
    for fam in ((SymptomFamily.COMPUTE_1C2V_SUBCORE_IMBALANCE, SymptomFamily.COMPUTE_FIXPIPE_ACCUM_IMBALANCE) if ce == "aiv_only" else ((SymptomFamily.COMPUTE_1C2V_SUBCORE_IMBALANCE, SymptomFamily.COMPUTE_VECTOR_LOOP_DEGRADED) if ce == "aic_only" else ())):
        raw_weights[fam] = 0.0
    cs = float(max(0, l0_cube + l0_vec))
    excess: dict[SymptomFamily, float] = {
        SymptomFamily.DMA_REDUNDANT_GM_ROUNDTRIP: float(gm_rt) + max(0.0, t_inf - 1.0) * cs,
        SymptomFamily.DMA_UNALIGNED_SHORT_BURST: float(unalign_p), SymptomFamily.DMA_L2_THRASH_REUSE_MISS: max(0.0, 1.0 - l2_hit) * cs,
        SymptomFamily.COMPUTE_VECTOR_LOOP_DEGRADED: (float(vf_breaks) + dual_col) * cs,
        SymptomFamily.COMPUTE_INEFFECTIVE_THROUGHPUT: max(0.0, 1.0 - eta_obs) * cs if eta_obs > 0.0 else 0.0,
        SymptomFamily.COMPUTE_FIXPIPE_ACCUM_IMBALANCE: float(max(0, pipe_fix - pipe_m)),
        SymptomFamily.COMPUTE_1C2V_SUBCORE_IMBALANCE: float(abs(aiv0_v - aiv1_v)),
        SymptomFamily.SCALAR_LOOP_TRAP: cs + scalar_fb * 64.0,
        SymptomFamily.SRAM_REGISTER_SPILL_STACK: float(spill_b + stack_b + max_ovf + deg_bufs * 64),
        SymptomFamily.SRAM_UB_L1_BANK_CONFLICT: max(bank_c, ub_risk, l1_risk) * cs,
        SymptomFamily.SYNC_PSEUDO_DOUBLE_BUFFER: float(alias_ser) * cs,
        SymptomFamily.SYNC_BARRIER_SERIALIZATION: sync_stall * cs + float(pipe_all * 64),
        SymptomFamily.GRID_TAIL_WAVE_UNDERUTIL: max(wave_w, under_s) * cs, SymptomFamily.UNMODELED_SILICON_RESIDUAL: 0.0,
    }
    shares = normalize_symptom_simplex(raw_weights, roofline_eta=roof_eta if roof_eta > 0.0 else eff_eta, has_l2_pmu=has_l2_pmu)
    return sorted(shares.items(), key=lambda item: (-item[1], -(excess.get(item[0], 0.0) if raw_weights.get(item[0], 0.0) > 0.0 else 0.0), _PHYSICAL_PRIORITY_INDEX[item[0]]))


def rank_symptom_families(
    obs_or_context: ObservationVector5D | Mapping[str, object], *,
    case_ratios: object | None = None, topology_axes: object | None = None,
) -> list[tuple[SymptomFamily, float]]:
    """Compute simplex-conserved shares over all 14 SymptomFamily categories."""
    return compute_symptom_simplex(obs_or_context, case_ratios=case_ratios, topology_axes=topology_axes)


def should_escalate_diagnostic_or_creative(
    simplex: Sequence[tuple[SymptomFamily | str, float]], *, stagnation_rounds: int
) -> bool:
    """Escalate when stalled on dominant unmodeled residual or high simplex entropy."""
    if stagnation_rounds < 1 or not simplex:
        return False
    top_raw = max(simplex, key=lambda item: float(item[1]))[0]
    if (top_raw.value if isinstance(top_raw, SymptomFamily) else str(top_raw)) == SymptomFamily.UNMODELED_SILICON_RESIDUAL.value:
        return True
    weights = [max(0.0, float(val)) for _, val in simplex]
    if (total := sum(weights)) <= 0.0:
        return True
    log_k = math.log(len(SymptomFamily))
    return (-sum((w / total) * math.log(w / total) for w in weights if w > 0.0) / log_k) > (1.0 - 1.0 / log_k)


def default_seed_rules_path() -> Path:
    """Return path to the bundled pure-English causal seed rule library."""
    return Path(__file__).resolve().with_name("seed_rules.yaml")


def load_seed_rules(path: str | Path | None = None, *, require_symptom_family: bool | None = None) -> tuple[AscendCausalRule, ...]:
    """Load and validate AscendCausalRule entries from YAML."""
    target_path = Path(path) if path is not None else default_seed_rules_path()
    is_bundled = path is None or target_path.resolve() == default_seed_rules_path().resolve()
    raw = yaml.safe_load(target_path.read_text(encoding="utf-8"))
    items = raw["rules"] if isinstance(raw, Mapping) and "rules" in raw else (raw if isinstance(raw, list) else None)
    if items is None:
        raise ValueError(f"invalid rules YAML root in {target_path}")
    rules: list[AscendCausalRule] = []
    for entry in items:
        rule = AscendCausalRule.from_dict(entry)
        enforce = require_symptom_family if require_symptom_family is not None else (is_bundled or not rule.source_provenance.startswith("flywheel:"))
        if enforce and (not rule.symptom_family or SymptomFamily(rule.symptom_family) not in DETERMINISTIC_SYMPTOM_FAMILIES):
            raise ValueError(f"seed rule {rule.rule_id!r} in {target_path} must declare a valid deterministic symptom_family")
        rules.append(rule)
    return tuple(rules)


def save_rules_yaml(rules: tuple[AscendCausalRule, ...] | list[AscendCausalRule], path: str | Path) -> None:
    """Persist AscendCausalRule entries to YAML cleanly in sorted order."""
    atomic_write_text(path, yaml.safe_dump({"rules": [r.to_dict() for r in rules]}, sort_keys=False, allow_unicode=False))


__all__ = [
    "AscendCausalRule", "DETERMINISTIC_SYMPTOM_FAMILIES", "SYMPTOM_FAMILY_TO_AXIS",
    "SYMPTOM_FAMILY_TO_FACETS", "SymptomFamily", "UnsafePredicateError",
    "build_predicate_context", "compute_symptom_simplex", "default_seed_rules_path",
    "load_seed_rules", "normalize_runtime_backend", "normalize_symptom_simplex",
    "rank_symptom_families", "safe_eval_predicate", "save_rules_yaml",
    "should_escalate_diagnostic_or_creative", "validate_rule_predicate",
]
