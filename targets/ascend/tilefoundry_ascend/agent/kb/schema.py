"""Six-tuple AscendCausalRule schema and safe AST inequality evaluator."""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path

import yaml

from tilefoundry_ascend.ascend.passes.observation_space import (
    AxisResidual5D,
    CompilerSummaryLike,
    ObservationVector5D,
    RooflineSummaryLike,
)
from tilefoundry_ascend.ascend.passes.physical_ledger import ObjectivePhysicalLedger


class SymptomFamily(str, Enum):
    """Fixed 12-family hardware waste taxonomy shared across diagnosis and rule routing."""

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


SYMPTOM_FAMILY_TO_AXIS: dict[SymptomFamily, str] = {
    SymptomFamily.DMA_REDUNDANT_GM_ROUNDTRIP: "offchip_dma",
    SymptomFamily.DMA_UNALIGNED_SHORT_BURST: "offchip_dma",
    SymptomFamily.DMA_L2_THRASH_REUSE_MISS: "offchip_dma",
    SymptomFamily.COMPUTE_VECTOR_LOOP_DEGRADED: "compute_issue",
    SymptomFamily.COMPUTE_INEFFECTIVE_THROUGHPUT: "compute_issue",
    SymptomFamily.COMPUTE_FIXPIPE_ACCUM_IMBALANCE: "compute_issue",
    SymptomFamily.COMPUTE_1C2V_SUBCORE_IMBALANCE: "compute_issue",
    SymptomFamily.SCALAR_LOOP_TRAP: "compute_issue",
    SymptomFamily.SRAM_REGISTER_SPILL_STACK: "onchip_sram_reg",
    SymptomFamily.SRAM_UB_L1_BANK_CONFLICT: "onchip_sram_reg",
    SymptomFamily.SYNC_PSEUDO_DOUBLE_BUFFER: "pipeline_sync",
    SymptomFamily.SYNC_BARRIER_SERIALIZATION: "pipeline_sync",
    SymptomFamily.GRID_TAIL_WAVE_UNDERUTIL: "grid_wave",
}

SYMPTOM_FAMILY_TO_FACETS: dict[SymptomFamily, tuple[str, ...]] = {
    SymptomFamily.DMA_REDUNDANT_GM_ROUNDTRIP: (
        "dma_and_cache_hierarchy",
        "sram_layout_and_aliasing",
        "pipeline_and_sync_graph",
    ),
    SymptomFamily.DMA_UNALIGNED_SHORT_BURST: (
        "tile_and_stage_knobs",
        "dma_and_cache_hierarchy",
    ),
    SymptomFamily.DMA_L2_THRASH_REUSE_MISS: (
        "grid_and_workspace_topology",
        "dma_and_cache_hierarchy",
    ),
    SymptomFamily.COMPUTE_VECTOR_LOOP_DEGRADED: (
        "exec_mode_and_spr",
        "alg_compute_and_monoid",
    ),
    SymptomFamily.COMPUTE_INEFFECTIVE_THROUGHPUT: (
        "exec_mode_and_spr",
        "tile_and_stage_knobs",
    ),
    SymptomFamily.COMPUTE_FIXPIPE_ACCUM_IMBALANCE: (
        "tile_and_stage_knobs",
        "sram_layout_and_aliasing",
    ),
    SymptomFamily.COMPUTE_1C2V_SUBCORE_IMBALANCE: (
        "grid_and_workspace_topology",
        "pipeline_and_sync_graph",
    ),
    SymptomFamily.SCALAR_LOOP_TRAP: (
        "exec_mode_and_spr",
        "dma_and_cache_hierarchy",
    ),
    SymptomFamily.SRAM_REGISTER_SPILL_STACK: (
        "tile_and_stage_knobs",
        "exec_mode_and_spr",
    ),
    SymptomFamily.SRAM_UB_L1_BANK_CONFLICT: ("sram_layout_and_aliasing",),
    SymptomFamily.SYNC_PSEUDO_DOUBLE_BUFFER: (
        "pipeline_and_sync_graph",
        "sram_layout_and_aliasing",
    ),
    SymptomFamily.SYNC_BARRIER_SERIALIZATION: (
        "pipeline_and_sync_graph",
        "tile_and_stage_knobs",
    ),
    SymptomFamily.GRID_TAIL_WAVE_UNDERUTIL: (
        "grid_and_workspace_topology",
        "tile_and_stage_knobs",
    ),
}

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
}

_CMP_OPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
    ast.In: lambda a, b: a in b,
    ast.NotIn: lambda a, b: a not in b,
}

_SAFE_FUNCS = {
    "abs": abs,
    "max": max,
    "min": min,
}

_ALLOWED_AST_NODES = (
    ast.Expression,
    ast.BoolOp,
    ast.And,
    ast.Or,
    ast.UnaryOp,
    ast.Not,
    ast.USub,
    ast.UAdd,
    ast.BinOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Compare,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
    ast.Constant,
    ast.Name,
    ast.Attribute,
    ast.Load,
    ast.Tuple,
    ast.List,
    ast.Call,
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
        parts.append(cur.id)
        return ".".join(reversed(parts))
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
            raise UnsafePredicateError(
                f"disallowed AST node {type(node).__name__} in predicate {expr!r}"
            )
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _SAFE_FUNCS:
                raise UnsafePredicateError(f"only {sorted(_SAFE_FUNCS)} calls are permitted")
            if node.keywords:
                raise UnsafePredicateError("keyword arguments are not allowed in predicate calls")
        if isinstance(node, ast.Attribute):
            if _flatten_attribute(node) is None:
                raise UnsafePredicateError("attribute chains must root at an identifier")
        if isinstance(node, ast.Constant) and not isinstance(
            node.value, (int, float, str, bool, type(None))
        ):
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
        dotted = _flatten_attribute(node)
        if dotted is None:
            raise UnsafePredicateError("invalid attribute root")
        if dotted in env:
            return env[dotted]
        parts = dotted.split(".")
        cur: object = env
        for idx, part in enumerate(parts):
            if isinstance(cur, Mapping) and part in cur:
                cur = cur[part]
                continue
            rem = ".".join(parts[idx:])
            if isinstance(cur, Mapping) and rem in cur:
                return cur[rem]
            raise KeyError(f"unknown predicate attribute {dotted!r}")
        return cur
    if isinstance(node, ast.Tuple):
        return tuple(_eval_node(el, env) for el in node.elts)
    if isinstance(node, ast.List):
        return [_eval_node(el, env) for el in node.elts]
    if isinstance(node, ast.UnaryOp):
        val = _eval_node(node.operand, env)
        if isinstance(node.op, ast.Not):
            return not bool(val)
        if isinstance(node.op, ast.USub):
            return -float(val) if isinstance(val, float) else -int(val)
        return val
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            for val in node.values:
                if not bool(_eval_node(val, env)):
                    return False
            return True
        for val in node.values:
            if bool(_eval_node(val, env)):
                return True
        return False
    if isinstance(node, ast.BinOp):
        lhs = _eval_node(node.left, env)
        rhs = _eval_node(node.right, env)
        return _BIN_OPS[type(node.op)](lhs, rhs)
    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, env)
        for op, comp in zip(node.ops, node.comparators):
            right = _eval_node(comp, env)
            if not _CMP_OPS[type(op)](left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.Call):
        fn = _SAFE_FUNCS[node.func.id]
        args = [_eval_node(arg, env) for arg in node.args]
        return fn(*args)
    raise UnsafePredicateError(f"unsupported AST node {type(node).__name__}")


def safe_eval_predicate(expr: str, env: Mapping[str, object]) -> bool:
    """Safely evaluate a causal rule guard, precondition, or postcondition."""
    tree = validate_rule_predicate(expr)
    return bool(_eval_node(tree, env))


_BACKEND_ALIASES = {
    "tilelang": "tilelang_dsl",
    "ascendc": "ascendc_trackb",
    "ascendc_template": "ascendc_trackb",
    "cce_cpp20": "ascendc_trackb",
    "cce_trackb": "ascendc_trackb",
    "tl_templates_cpp20": "ascendc_trackb",
    "hybrid": "hybrid_trackb",
}


def normalize_runtime_backend(backend: str | None) -> str | None:
    """Normalize runtime backend aliases to canonical KB backend identifiers."""
    if backend is None:
        return None
    cleaned = str(backend).strip()
    if not cleaned:
        return None
    return _BACKEND_ALIASES.get(cleaned, cleaned)


@dataclass(frozen=True)
class AscendCausalRule:
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

    def __post_init__(self) -> None:
        if not self.rule_id or not self.transform_template.strip():
            raise ValueError("rule_id and transform_template must be non-empty")
        if self.scope not in ("cross_op", "operator_local"):
            raise ValueError(
                f"invalid rule scope {self.scope!r}; expected 'cross_op' or 'operator_local'"
            )
        normalized_backends = tuple(
            norm
            for raw in self.compatible_backends
            if (norm := normalize_runtime_backend(str(raw)))
        )
        if not normalized_backends:
            raise ValueError("compatible_backends must contain at least one backend")
        object.__setattr__(self, "compatible_backends", normalized_backends)
        cleaned_family = str(self.symptom_family or "").strip()
        if cleaned_family:
            try:
                SymptomFamily(cleaned_family)
            except ValueError as exc:
                valid = tuple(fam.value for fam in SymptomFamily)
                raise ValueError(
                    f"invalid symptom_family {cleaned_family!r}; expected one of {valid}"
                ) from exc
        object.__setattr__(self, "symptom_family", cleaned_family)
        anti_expr = str(self.anti_precondition or "False").strip() or "False"
        object.__setattr__(self, "anti_precondition", anti_expr)
        validate_rule_predicate(self.regime_guard)
        validate_rule_predicate(self.precondition)
        validate_rule_predicate(self.anti_precondition)
        validate_rule_predicate(self.expected_postcondition)

    def supports_backend(self, runtime_backend: str | None) -> bool:
        """Return whether this rule applies to *runtime_backend*."""
        norm = normalize_runtime_backend(runtime_backend)
        if norm is None:
            return True
        return norm in self.compatible_backends

    def render_for_backend(self, runtime_backend: str | None = None) -> AscendCausalRule:
        """Return rule rendered with Track-B CCE (C++20) guidance on C++20 backends."""
        norm = normalize_runtime_backend(runtime_backend)
        cpp_guide = self.cpp20_transform_template.strip()
        if not cpp_guide:
            return self
        if norm == "ascendc_trackb":
            return replace(self, transform_template=cpp_guide)
        if norm == "hybrid_trackb" and cpp_guide not in self.transform_template:
            combined = (
                f"{self.transform_template.strip()}\n\n"
                f"Track-B CCE (C++20):\n{cpp_guide}"
            )
            return replace(self, transform_template=combined)
        return self

    def as_six_tuple(self) -> tuple[str, str, str, str, str, str]:
        """Return canonical 6-tuple fields."""
        return (
            self.rule_id,
            self.regime_guard,
            self.precondition,
            self.transform_template,
            self.expected_postcondition,
            self.source_provenance,
        )

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

    def to_dict(self) -> dict[str, object]:
        """Serialize rule into a YAML/JSON-compatible mapping."""
        payload: dict[str, object] = {
            "rule_id": self.rule_id,
            "regime_guard": self.regime_guard,
            "precondition": self.precondition,
            "anti_precondition": self.anti_precondition,
            "symptom_family": self.symptom_family,
            "transform_template": self.transform_template,
            "expected_postcondition": self.expected_postcondition,
            "source_provenance": self.source_provenance,
            "alpha_prior": self.alpha_prior,
            "beta_prior": self.beta_prior,
            "rationale": self.rationale,
            "priority": self.priority,
            "scope": self.scope,
            "origin_operator": self.origin_operator,
            "compatible_backends": list(self.compatible_backends),
            "cpp20_transform_template": self.cpp20_transform_template,
        }
        return payload

    @classmethod
    def from_dict(cls, entry: Mapping[str, object]) -> AscendCausalRule:
        """Deserialize rule from YAML/JSON mapping."""
        raw_backends = entry.get("compatible_backends")
        if raw_backends is None:
            backends: tuple[str, ...] = ("tilelang_dsl", "ascendc_trackb", "hybrid_trackb")
        elif isinstance(raw_backends, str):
            backends = tuple(b.strip() for b in raw_backends.split(",") if b.strip())
        else:
            backends = tuple(str(b).strip() for b in raw_backends if str(b).strip())
        return cls(
            rule_id=str(entry["rule_id"]),
            regime_guard=str(entry["regime_guard"]),
            precondition=str(entry["precondition"]),
            transform_template=str(entry["transform_template"]),
            expected_postcondition=str(entry["expected_postcondition"]),
            source_provenance=str(entry["source_provenance"]),
            alpha_prior=float(entry.get("alpha_prior", 2.0)),
            beta_prior=float(entry.get("beta_prior", 1.0)),
            rationale=str(entry.get("rationale", "")),
            priority=int(entry.get("priority", 100)),
            scope=str(entry.get("scope", "cross_op") or "cross_op"),
            origin_operator=str(entry.get("origin_operator", "") or ""),
            compatible_backends=backends,
            cpp20_transform_template=str(entry.get("cpp20_transform_template", "") or ""),
            symptom_family=str(entry.get("symptom_family", "") or ""),
            anti_precondition=str(entry.get("anti_precondition", "False") or "False"),
        )


_DEFAULT_PIPE_KEYS = (
    "AIC.PIPE_M",
    "AIC.PIPE_FIX",
    "AIC.PIPE_MTE1",
    "AIC.PIPE_MTE2",
    "AIC.PIPE_MTE3",
    "AIC.PIPE_S",
    "AIV0.PIPE_V",
    "AIV0.PIPE_MTE1",
    "AIV0.PIPE_MTE2",
    "AIV0.PIPE_MTE3",
    "AIV0.PIPE_S",
    "AIV1.PIPE_V",
    "AIV1.PIPE_MTE1",
    "AIV1.PIPE_MTE2",
    "AIV1.PIPE_MTE3",
    "AIV1.PIPE_S",
)

_DEFAULT_SCOPES = ("UB", "L1", "L0A", "L0B", "L0C", "SSBUF")

_RESIDUAL_DEFAULTS: dict[str, int | float | bool | str] = {
    "measured_latency_us": 0.0,
    "predicted_bottleneck_util": 0.0,
    "measured_aic_mac_ratio": 0.0,
    "measured_aiv_vec_ratio": 0.0,
    "l2_aic_mac_ratio": 0.0,
    "l2_aiv_vec_ratio": 0.0,
    "measured_mte2_wait_ratio": 0.0,
    "measured_sync_stall_ratio": 0.0,
    "unexplained_gap_ratio": 0.0,
    "transfer_inflation_ratio": 1.0,
    "effective_compute_eta": 0.0,
    "pseudo_double_buffer_serialized": False,
    "spill_bytes": 0,
    "stack_bytes": 0,
    "vloopv2_hit": True,
    "l2_hit_rate": 1.0,
    "bank_conflict_ratio": 0.0,
    "vf_fusion_breaks": 0,
    "alias_serialized_buffers": 0,
    "dual_issue_ipc_collapse": 0.0,
    "dcache_dcci_flushes": 0,
    "has_local_membar_in_vf_call": False,
}

_TELEMETRY_CONTEXT_DEFAULTS: dict[str, int | float | bool | str] = {
    "num_aic_cores": 28,
    "num_aiv_cores": 56,
    "ub_bank_conflict_risk": 0.0,
    "l1_bank_conflict_risk": 0.0,
    "l1_vec_regs": 0,
    "l1_scalar_regs": 0,
    "has_empty_group": False,
    "has_loop_carried_state": False,
    "compute_engine": "",
    "memory_ring": "",
    "iteration_regularity": "",
}


def _ledger_to_namespace(ledger: ObjectivePhysicalLedger) -> dict[str, object]:
    ns: dict[str, object] = {key: 0 for key in _DEFAULT_PIPE_KEYS}
    for key, val in ledger.pipe_cycles.items():
        ns[key] = int(val)
    ns["PIPE_M"] = int(ns["AIC.PIPE_M"])
    ns["PIPE_FIX"] = int(ns["AIC.PIPE_FIX"])
    ns["PIPE_V"] = max(int(ns["AIV0.PIPE_V"]), int(ns["AIV1.PIPE_V"]))
    ns["PIPE_MTE2"] = max(
        int(ns["AIC.PIPE_MTE2"]),
        int(ns["AIV0.PIPE_MTE2"]),
        int(ns["AIV1.PIPE_MTE2"]),
    )
    ns["PIPE_MTE3"] = max(int(ns["AIV0.PIPE_MTE3"]), int(ns["AIV1.PIPE_MTE3"]))
    ns["is_legal"] = ledger.is_legal
    ns["analysis_mode"] = ledger.analysis_mode
    ns["steady_state_ii"] = ledger.steady_state_ii if ledger.steady_state_ii is not None else 0
    ns["dominant_pipe"] = ledger.dominant_pipe
    ns["mte2_inflight_per_aiv"] = ledger.mte2_inflight_per_aiv
    ns["unaligned_dma_padding_bytes"] = ledger.unaligned_dma_padding_bytes
    ns["max_overflow_bytes"] = ledger.max_overflow_bytes
    ns["has_opaque_regions"] = bool(ledger.opaque_dynamic_regions)

    cube_cycles = int(ledger.pipe_cycles.get("AIC.PIPE_M", 0)) + int(
        ledger.pipe_cycles.get("AIC.CUBE", 0)
    )
    vec_cycles = int(ledger.pipe_cycles.get("AIV0.PIPE_V", 0)) + int(
        ledger.pipe_cycles.get("AIV1.PIPE_V", 0)
    )
    ns["cube_cycles"] = cube_cycles
    ns["vec_cycles"] = vec_cycles
    ns["l0_cube_cycles"] = cube_cycles
    ns["l0_vec_cycles"] = vec_cycles
    ns["cube_to_vector_ratio"] = (
        round(cube_cycles / float(max(1, vec_cycles)), 6) if cube_cycles > 0 else 0.0
    )
    ns["wave_occupancy"] = float(ledger.wave_occupancy)
    ns["l0_wave_occupancy"] = float(ledger.wave_occupancy)
    ns["grid_blocks"] = int(ledger.grid_blocks)
    ns["l0_grid_blocks"] = int(ledger.grid_blocks)
    ns["active_grid_blocks"] = int(ledger.active_grid_blocks)
    ns["grid_is_symbolic"] = bool(ledger.grid_is_symbolic)
    ns["in_loop_pipe_all_barriers"] = int(ledger.in_loop_pipe_all_barriers)
    ns["gm_roundtrip_bytes"] = int(ledger.gm_roundtrip_bytes)
    ns["dominant_contig_bytes"] = int(ledger.dominant_contig_bytes)
    ns["effective_cube_m"] = int(ledger.effective_cube_m)
    ns["has_indirect_gather"] = bool(ledger.has_indirect_gather)
    ns["has_lowbit_quant"] = bool(ledger.has_lowbit_quant)
    ns["has_online_rescale_loop"] = bool(ledger.has_online_rescale_loop)
    ns["in_loop_ub_reload_stores"] = int(ledger.in_loop_ub_reload_stores)
    ns["uses_unreserved_ubuf"] = bool(ledger.uses_unreserved_ubuf)
    ns["min_gm_dma_bytes"] = int(ledger.min_gm_dma_bytes)
    ns["dma_burst_efficiency_ratio"] = float(ledger.dma_burst_efficiency_ratio)
    ns["uses_dynamic_offset_pipelining"] = bool(ledger.uses_dynamic_offset_pipelining)
    ns["has_unlimit_memory_attr"] = bool(ledger.has_unlimit_memory_attr)
    ns["has_coupled_reduce_sfu_accum"] = bool(ledger.has_coupled_reduce_sfu_accum)
    ns["has_simtvf"] = any(
        "simt_vf" in region.lower() or "simtvf" in region.lower()
        for region in ledger.opaque_dynamic_regions
    )
    ns["has_dynamic_control"] = bool(ledger.opaque_dynamic_regions)

    for key, default_val in _TELEMETRY_CONTEXT_DEFAULTS.items():
        raw_val = getattr(ledger, key, default_val)
        if isinstance(default_val, bool):
            ns[key] = bool(raw_val)
        elif isinstance(default_val, int):
            ns[key] = int(raw_val)
        elif isinstance(default_val, float):
            ns[key] = float(raw_val)
        else:
            ns[key] = str(raw_val)

    total_bloat = 0
    degraded_total = 0
    any_overflow = 0
    for scope in _DEFAULT_SCOPES:
        wl = ledger.waterlines.get(scope)
        if wl is None:
            ns[scope] = {
                "capacity_bytes": 0,
                "explicit_bytes": 0,
                "implicit_bloat_bytes": 0,
                "peak_allocated_bytes": 0,
                "overflow_bytes": 0,
                "degraded_count": 0,
            }
        else:
            total_bloat += wl.implicit_bloat_bytes
            degraded_total += len(wl.degraded_single_buffers)
            any_overflow = max(any_overflow, wl.overflow_bytes)
            ns[scope] = {
                "capacity_bytes": wl.capacity_bytes,
                "explicit_bytes": wl.explicit_bytes,
                "implicit_bloat_bytes": wl.implicit_bloat_bytes,
                "peak_allocated_bytes": wl.peak_allocated_bytes,
                "overflow_bytes": wl.overflow_bytes,
                "degraded_count": len(wl.degraded_single_buffers),
            }
    ub_ns = ns["UB"]
    assert isinstance(ub_ns, dict)
    ub_peak = int(ub_ns["peak_allocated_bytes"])
    ub_cap = int(ub_ns["capacity_bytes"])
    ub_ratio = round(float(ub_peak) / float(max(1, ub_cap)), 6) if ub_cap > 0 else 0.0
    ns["ub_utilization_ratio"] = ub_ratio
    ns["ub_occupancy_ratio"] = ub_ratio
    ns["implicit_bloat_bytes"] = total_bloat
    ns["degraded_buffer_count"] = degraded_total
    ns["any_overflow_bytes"] = any_overflow
    return ns


def _extract_typed_defaults(
    source: Mapping[str, object] | object,
    defaults: Mapping[str, int | float | bool | str],
) -> dict[str, int | float | bool | str]:
    resolved: dict[str, int | float | bool | str] = {}
    for key, default_val in defaults.items():
        if isinstance(source, Mapping) and key in source:
            raw_val = source[key]
        elif hasattr(source, key):
            raw_val = getattr(source, key)
        else:
            continue
        if isinstance(default_val, bool):
            resolved[key] = bool(raw_val)
        elif isinstance(default_val, int):
            resolved[key] = int(raw_val)
        elif isinstance(default_val, float):
            resolved[key] = float(raw_val)
        else:
            resolved[key] = str(raw_val)
    return resolved


def _embedded_observation(
    residual: Mapping[str, object] | object | None,
) -> ObservationVector5D | None:
    """Recover the 5D observation carried by a residual object or serialized mapping."""
    if residual is None:
        return None
    if isinstance(residual, ObservationVector5D):
        return residual
    raw_observation: object = (
        residual.get("observation")
        if isinstance(residual, Mapping)
        else getattr(residual, "observation", None)
    )
    if isinstance(raw_observation, ObservationVector5D):
        return raw_observation
    if isinstance(raw_observation, Mapping):
        return ObservationVector5D.from_dict(raw_observation)
    return None


def _axis_residuals(
    ledger: ObjectivePhysicalLedger,
    residual: Mapping[str, object] | object | None,
    roofline: RooflineSummaryLike | Mapping[str, object] | None = None,
) -> AxisResidual5D:
    """Compute axis residuals from the measured observation, else the ledger's static L0 view."""
    observation = _embedded_observation(residual)
    if observation is None:
        observation = ledger.to_observation_5d(roofline=roofline)
    return observation.compute_axis_residuals()


def _axis_gap_context(residuals: AxisResidual5D) -> dict[str, float | str]:
    """Flatten the five axis gaps plus the dominant axis into predicate-context keys."""
    context: dict[str, float | str] = {
        "compute_issue_gap": residuals.compute_issue_gap,
        "offchip_dma_gap": residuals.offchip_dma_gap,
        "onchip_sram_reg_gap": residuals.onchip_sram_reg_gap,
        "pipeline_sync_gap": residuals.pipeline_sync_gap,
        "grid_wave_gap": residuals.grid_wave_gap,
        "dominant_axis": residuals.dominant_axis,
    }
    return context


def build_predicate_context(
    ledger: ObjectivePhysicalLedger,
    *,
    features: object | None = None,
    residual: Mapping[str, object] | object | None = None,
    compiler_summary: CompilerSummaryLike | Mapping[str, object] | None = None,
    prev_ledger: ObjectivePhysicalLedger | None = None,
    extra: Mapping[str, object] | None = None,
    target: object | None = None,
    roofline: RooflineSummaryLike | Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Assemble a flat and dotted-path dictionary for safe rule predicate evaluation.

    Both the current and previous levels expose the five 5D axis gaps and dominant_axis.
    """
    ctx = _ledger_to_namespace(ledger)
    if features is not None:
        if hasattr(features, "to_dict"):
            ctx.update(features.to_dict())
        elif isinstance(features, Mapping):
            ctx.update(features)
        for key in (
            "compute_engine",
            "memory_ring",
            "iteration_regularity",
            "has_empty_group",
            "has_loop_carried_state",
        ):
            if hasattr(features, key):
                ctx[key] = getattr(features, key)
        feat_axes = getattr(features, "topology_axes", None)
        if isinstance(feat_axes, Mapping):
            for ak in ("compute_engine", "memory_ring", "iteration_regularity"):
                if ak in feat_axes:
                    ctx[ak] = str(feat_axes[ak])
        elif feat_axes is not None:
            for ak in ("compute_engine", "memory_ring", "iteration_regularity"):
                if hasattr(feat_axes, ak):
                    ctx[ak] = str(getattr(feat_axes, ak))

    if target is not None:
        dev = getattr(target, "device", target)
        if hasattr(dev, "aic_count"):
            ctx["num_aic_cores"] = int(dev.aic_count)
        if hasattr(dev, "aiv_count"):
            ctx["num_aiv_cores"] = int(dev.aiv_count)

    res_defaults: dict[str, int | float | bool | str] = dict(_RESIDUAL_DEFAULTS)
    has_l1_summary = compiler_summary is not None
    if compiler_summary is not None:
        comp_vals = _extract_typed_defaults(compiler_summary, _RESIDUAL_DEFAULTS)
        if isinstance(compiler_summary, Mapping):
            sw = int(compiler_summary.get("software_loops", 0) or 0)
            vl = int(compiler_summary.get("vloopv2_loops", 0) or 0)
            ctx["l1_vec_regs"] = int(compiler_summary.get("vec_regs", 0) or 0)
            ctx["l1_scalar_regs"] = int(compiler_summary.get("scalar_regs", 0) or 0)
        else:
            sw = int(compiler_summary.software_loops)
            vl = int(compiler_summary.vloopv2_loops)
            ctx["l1_vec_regs"] = int(compiler_summary.vec_regs)
            ctx["l1_scalar_regs"] = int(compiler_summary.scalar_regs)
        if sw == 0 and vl == 0 and "vloopv2_hit" in comp_vals:
            comp_vals["vloopv2_hit"] = True
        res_defaults.update(comp_vals)
    if residual is not None:
        res_defaults.update(_extract_typed_defaults(residual, _RESIDUAL_DEFAULTS))
        ctx.update(_extract_typed_defaults(residual, _TELEMETRY_CONTEXT_DEFAULTS))
        if (
            res_defaults.get("measured_aic_mac_ratio", 0.0)
            and not res_defaults.get("l2_aic_mac_ratio", 0.0)
        ):
            res_defaults["l2_aic_mac_ratio"] = res_defaults["measured_aic_mac_ratio"]
        if (
            res_defaults.get("measured_aiv_vec_ratio", 0.0)
            and not res_defaults.get("l2_aiv_vec_ratio", 0.0)
        ):
            res_defaults["l2_aiv_vec_ratio"] = res_defaults["measured_aiv_vec_ratio"]
        embedded_obs = _embedded_observation(residual)
        if embedded_obs is not None:
            obs_payload = embedded_obs.to_dict()
            dma_payload = obs_payload.get("offchip_dma")
            compute_payload = obs_payload.get("compute_issue")
            for key, axis_payload in (
                ("transfer_inflation_ratio", dma_payload),
                ("effective_compute_eta", compute_payload),
            ):
                if (
                    isinstance(residual, Mapping)
                    and key in residual
                ) or (not isinstance(residual, Mapping) and hasattr(residual, key)):
                    continue
                raw_val = obs_payload.get(key)
                if raw_val is None and isinstance(axis_payload, Mapping):
                    raw_val = axis_payload.get(key)
                    if raw_val is None:
                        raw_val = axis_payload.get(f"l2_{key}")
                if isinstance(raw_val, (int, float)) and not isinstance(raw_val, bool):
                    res_defaults[key] = float(raw_val)
            if not has_l1_summary and embedded_obs.has_l1_compiler_summary:
                has_l1_summary = True
                ctx["l1_vec_regs"] = int(embedded_obs.onchip_sram_reg.l1_vec_regs)
                ctx["l1_scalar_regs"] = int(embedded_obs.onchip_sram_reg.l1_scalar_regs)
                res_defaults["spill_bytes"] = max(
                    int(res_defaults["spill_bytes"]),
                    int(embedded_obs.onchip_sram_reg.l1_spill_bytes),
                )
                res_defaults["stack_bytes"] = max(
                    int(res_defaults["stack_bytes"]),
                    int(embedded_obs.onchip_sram_reg.l1_stack_bytes),
                )
                obs_vloop_hit = not (
                    embedded_obs.compute_issue.l1_software_loops > 0
                    and embedded_obs.compute_issue.l1_vloopv2_loops == 0
                )
                res_defaults["vloopv2_hit"] = (
                    bool(res_defaults["vloopv2_hit"]) and obs_vloop_hit
                )
                res_defaults["vf_fusion_breaks"] = max(
                    int(res_defaults["vf_fusion_breaks"]),
                    int(embedded_obs.compute_issue.l1_vf_fusion_breaks),
                )
                res_defaults["dual_issue_ipc_collapse"] = max(
                    float(res_defaults["dual_issue_ipc_collapse"]),
                    float(embedded_obs.compute_issue.l1_dual_issue_collapse),
                )
                res_defaults["alias_serialized_buffers"] = max(
                    int(res_defaults["alias_serialized_buffers"]),
                    int(embedded_obs.pipeline_sync.l1_alias_serialized_buffers),
                )
                res_defaults["has_local_membar_in_vf_call"] = (
                    bool(res_defaults["has_local_membar_in_vf_call"])
                    or bool(embedded_obs.pipeline_sync.l1_local_membar_in_vf)
                )
                res_defaults["dcache_dcci_flushes"] = max(
                    int(res_defaults["dcache_dcci_flushes"]),
                    int(embedded_obs.offchip_dma.l1_dcache_dcci_flushes),
                )
    elif roofline is not None:
        static_obs = ledger.to_observation_5d(roofline=roofline)
        if static_obs.transfer_inflation_ratio > 0.0:
            res_defaults["transfer_inflation_ratio"] = float(static_obs.transfer_inflation_ratio)
    ctx["has_l1_compiler_summary"] = has_l1_summary
    ctx.update(res_defaults)
    ctx.update(_axis_gap_context(_axis_residuals(ledger, residual, roofline=roofline)))

    if (
        float(ctx.get("ub_bank_conflict_risk", 0.0)) == 0.0
        and float(ctx.get("bank_conflict_ratio", 0.0)) > 0.0
    ):
        ctx["ub_bank_conflict_risk"] = float(ctx["bank_conflict_ratio"])

    if prev_ledger is not None:
        prev_ns = _ledger_to_namespace(prev_ledger)
        prev_ns.update(_axis_gap_context(_axis_residuals(prev_ledger, None, roofline=roofline)))
        for key, val in _RESIDUAL_DEFAULTS.items():
            prev_ns.setdefault(key, val)
        for key, val in _TELEMETRY_CONTEXT_DEFAULTS.items():
            prev_ns.setdefault(key, val)
        if roofline is not None:
            prev_obs = prev_ledger.to_observation_5d(roofline=roofline)
            if prev_obs.transfer_inflation_ratio > 0.0:
                prev_ns["transfer_inflation_ratio"] = float(prev_obs.transfer_inflation_ratio)
        ctx["prev"] = prev_ns
    else:
        ctx["prev"] = dict(ctx)

    if extra:
        ctx.update(extra)
        if "ub_occupancy_ratio" in extra and "ub_utilization_ratio" not in extra:
            ctx["ub_utilization_ratio"] = float(extra["ub_occupancy_ratio"])
        elif "ub_utilization_ratio" in extra and "ub_occupancy_ratio" not in extra:
            ctx["ub_occupancy_ratio"] = float(extra["ub_utilization_ratio"])
        extra_axes = extra.get("topology_axes")
        if isinstance(extra_axes, Mapping):
            for ak in ("compute_engine", "memory_ring", "iteration_regularity"):
                if ak in extra_axes:
                    ctx[ak] = str(extra_axes[ak])
        elif extra_axes is not None:
            for ak in ("compute_engine", "memory_ring", "iteration_regularity"):
                if hasattr(extra_axes, ak):
                    ctx[ak] = str(getattr(extra_axes, ak))
    return ctx


def _read_signal_float(
    mapping: Mapping[str, object],
    *keys: str,
    default: float = 0.0,
) -> float:
    for key in keys:
        if key not in mapping:
            continue
        raw = mapping[key]
        if isinstance(raw, bool):
            continue
        if isinstance(raw, (int, float)):
            return float(raw)
    return default


def _read_signal_int(
    mapping: Mapping[str, object],
    *keys: str,
    default: int = 0,
) -> int:
    for key in keys:
        if key not in mapping:
            continue
        raw = mapping[key]
        if isinstance(raw, bool):
            continue
        if isinstance(raw, int):
            return raw
        if isinstance(raw, float):
            return int(raw)
    return default


def _read_signal_bool(
    mapping: Mapping[str, object],
    *keys: str,
    default: bool = False,
) -> bool:
    for key in keys:
        if key not in mapping:
            continue
        return bool(mapping[key])
    return default


def _observation_to_signal_dict(obs: ObservationVector5D) -> dict[str, object]:
    payload = obs.to_dict()
    compute = payload.get("compute_issue")
    dma = payload.get("offchip_dma")
    sram = payload.get("onchip_sram_reg")
    sync = payload.get("pipeline_sync")
    grid = payload.get("grid_irregularity")
    compute_map = compute if isinstance(compute, Mapping) else {}
    dma_map = dma if isinstance(dma, Mapping) else {}
    sram_map = sram if isinstance(sram, Mapping) else {}
    sync_map = sync if isinstance(sync, Mapping) else {}
    grid_map = grid if isinstance(grid, Mapping) else {}

    transfer_inflation = _read_signal_float(
        payload,
        "transfer_inflation_ratio",
        default=_read_signal_float(
            dma_map,
            "transfer_inflation_ratio",
            "l2_transfer_inflation_ratio",
            default=1.0,
        ),
    )
    effective_eta = _read_signal_float(
        payload,
        "effective_compute_eta",
        default=_read_signal_float(
            compute_map,
            "effective_compute_eta",
            "l2_effective_compute_eta",
            default=0.0,
        ),
    )
    sw_loops = _read_signal_int(compute_map, "l1_software_loops")
    vl_loops = _read_signal_int(compute_map, "l1_vloopv2_loops")
    vec_cycles = _read_signal_int(compute_map, "l0_vec_cycles")
    subcore_imbalance = _read_signal_bool(compute_map, "l0_aiv_subcore_imbalance")
    return {
        "transfer_inflation_ratio": transfer_inflation,
        "effective_compute_eta": effective_eta,
        "roofline_eta": _read_signal_float(compute_map, "l2_roofline_eta"),
        "gm_roundtrip_bytes": _read_signal_int(dma_map, "l0_gm_roundtrip_bytes"),
        "cube_to_vector_ratio": _read_signal_float(compute_map, "l0_cube_to_vec_ratio"),
        "dominant_contig_bytes": _read_signal_int(dma_map, "l0_dominant_contig_bytes"),
        "dma_burst_efficiency_ratio": _read_signal_float(
            dma_map, "l0_dma_burst_efficiency", default=1.0
        ),
        "unaligned_dma_padding_bytes": _read_signal_int(dma_map, "l0_unaligned_padding_bytes"),
        "mte2_inflight_per_aiv": _read_signal_int(dma_map, "l0_mte2_inflight_per_aiv"),
        "l2_hit_rate": _read_signal_float(dma_map, "l2_cache_hit_rate", default=1.0),
        "measured_mte2_wait_ratio": _read_signal_float(dma_map, "l2_mte2_wait_ratio"),
        "vloopv2_hit": not (sw_loops > 0 and vl_loops == 0),
        "vf_fusion_breaks": _read_signal_int(compute_map, "l1_vf_fusion_breaks"),
        "dual_issue_ipc_collapse": _read_signal_float(compute_map, "l1_dual_issue_collapse"),
        "measured_aiv_vec_ratio": _read_signal_float(compute_map, "l2_aiv_vec_ratio"),
        "measured_aic_mac_ratio": _read_signal_float(compute_map, "l2_aic_mac_ratio"),
        "l2_aiv_vec_ratio": _read_signal_float(compute_map, "l2_aiv_vec_ratio"),
        "l2_aic_mac_ratio": _read_signal_float(compute_map, "l2_aic_mac_ratio"),
        "aic_fixpipe_ratio": _read_signal_float(dma_map, "l2_fixpipe_ratio"),
        "AIC.PIPE_M": _read_signal_int(compute_map, "l0_cube_cycles"),
        "AIC.PIPE_FIX": 0,
        "AIV0.PIPE_V": vec_cycles,
        "AIV1.PIPE_V": 0 if subcore_imbalance else vec_cycles,
        "l0_cube_cycles": _read_signal_int(compute_map, "l0_cube_cycles"),
        "l0_vec_cycles": vec_cycles,
        "spill_bytes": _read_signal_int(sram_map, "l1_spill_bytes"),
        "stack_bytes": _read_signal_int(sram_map, "l1_stack_bytes"),
        "l1_vec_regs": _read_signal_int(sram_map, "l1_vec_regs"),
        "max_overflow_bytes": _read_signal_int(sram_map, "l0_max_overflow_bytes"),
        "degraded_buffer_count": _read_signal_int(sram_map, "l0_degraded_single_buffers"),
        "bank_conflict_ratio": _read_signal_float(sram_map, "l2_bank_conflict_ratio"),
        "ub_bank_conflict_risk": _read_signal_float(sram_map, "l0_ub_bank_conflict_risk"),
        "l1_bank_conflict_risk": _read_signal_float(sram_map, "l0_l1_bank_conflict_risk"),
        "pseudo_double_buffer_serialized": _read_signal_bool(sync_map, "l2_pseudo_double_buffer"),
        "alias_serialized_buffers": _read_signal_int(sync_map, "l1_alias_serialized_buffers"),
        "measured_sync_stall_ratio": _read_signal_float(sync_map, "l2_sync_stall_ratio"),
        "in_loop_pipe_all_barriers": _read_signal_int(sync_map, "l0_in_loop_pipe_all_barriers"),
        "has_local_membar_in_vf_call": _read_signal_bool(sync_map, "l1_local_membar_in_vf"),
        "wave_occupancy": _read_signal_float(grid_map, "l0_wave_occupancy", default=1.0),
        "l0_wave_occupancy": _read_signal_float(grid_map, "l0_wave_occupancy", default=1.0),
        "grid_blocks": _read_signal_int(grid_map, "l0_grid_blocks", default=0),
        "l0_grid_blocks": _read_signal_int(grid_map, "l0_grid_blocks", default=0),
        "num_aic_cores": 28,
        "num_aiv_cores": 56,
        "grid_is_symbolic": False,
        "has_l1_compiler_summary": _read_signal_bool(payload, "has_l1_compiler_summary"),
    }


def rank_symptom_families(
    obs_or_context: ObservationVector5D | Mapping[str, object],
) -> list[tuple[SymptomFamily, float]]:
    """Score and rank all 12 hardware SymptomFamily categories in descending severity."""
    sig: Mapping[str, object] = (
        _observation_to_signal_dict(obs_or_context)
        if isinstance(obs_or_context, ObservationVector5D)
        else obs_or_context
    )

    transfer_inflation = _read_signal_float(sig, "transfer_inflation_ratio", default=1.0)
    effective_eta = _read_signal_float(sig, "effective_compute_eta", default=0.0)
    roofline_eta = _read_signal_float(sig, "roofline_eta", default=0.0)
    gm_roundtrip = _read_signal_int(sig, "gm_roundtrip_bytes")
    cube_ratio = _read_signal_float(sig, "cube_to_vector_ratio")
    contig_bytes = _read_signal_int(sig, "dominant_contig_bytes")
    burst_eff = _read_signal_float(sig, "dma_burst_efficiency_ratio", default=1.0)
    unaligned_pad = _read_signal_int(sig, "unaligned_dma_padding_bytes")
    mte2_inflight = _read_signal_int(sig, "mte2_inflight_per_aiv")
    l2_hit = _read_signal_float(sig, "l2_hit_rate", default=1.0)
    mte2_wait = _read_signal_float(sig, "measured_mte2_wait_ratio", "mte2_wait_ratio")

    vloopv2_hit = _read_signal_bool(sig, "vloopv2_hit", default=True)
    vf_breaks = _read_signal_int(sig, "vf_fusion_breaks")
    dual_collapse = _read_signal_float(sig, "dual_issue_ipc_collapse")
    aiv_vec = _read_signal_float(sig, "measured_aiv_vec_ratio", "aiv_vec_ratio")
    aic_mac = _read_signal_float(sig, "measured_aic_mac_ratio", "aic_mac_ratio")
    fixpipe_ratio = _read_signal_float(sig, "aic_fixpipe_ratio", "aic_fixpipe")
    pipe_m = _read_signal_int(sig, "AIC.PIPE_M", "PIPE_M")
    pipe_fix = _read_signal_int(sig, "AIC.PIPE_FIX", "PIPE_FIX")
    aiv0_v = _read_signal_int(sig, "AIV0.PIPE_V")
    aiv1_v = _read_signal_int(sig, "AIV1.PIPE_V")
    l0_cube_cycles = _read_signal_int(sig, "l0_cube_cycles", "AIC.PIPE_M", "PIPE_M")
    l0_vec_cycles = _read_signal_int(
        sig,
        "l0_vec_cycles",
        default=(aiv0_v + aiv1_v) if (aiv0_v or aiv1_v) else _read_signal_int(sig, "PIPE_V"),
    )
    l1_vec_regs = _read_signal_int(sig, "l1_vec_regs", "vec_regs")
    has_l1_summary = _read_signal_bool(sig, "has_l1_compiler_summary")

    spill_bytes = _read_signal_int(sig, "spill_bytes")
    stack_bytes = _read_signal_int(sig, "stack_bytes", "l1_stack_bytes")
    max_overflow = _read_signal_int(sig, "max_overflow_bytes", "any_overflow_bytes")
    degraded_bufs = _read_signal_int(sig, "degraded_buffer_count")
    bank_conflict = _read_signal_float(sig, "bank_conflict_ratio")
    ub_bank_risk = _read_signal_float(sig, "ub_bank_conflict_risk")
    l1_bank_risk = _read_signal_float(sig, "l1_bank_conflict_risk")

    pseudo_db = _read_signal_bool(sig, "pseudo_double_buffer_serialized")
    alias_ser = _read_signal_int(sig, "alias_serialized_buffers")
    sync_stall = _read_signal_float(sig, "measured_sync_stall_ratio", "sync_stall_ratio")
    pipe_all = _read_signal_int(sig, "in_loop_pipe_all_barriers")
    local_membar = _read_signal_bool(sig, "has_local_membar_in_vf_call")

    wave_occ = _read_signal_float(sig, "wave_occupancy", default=1.0)
    grid_blocks = _read_signal_int(sig, "grid_blocks")
    num_aic = max(1, _read_signal_int(sig, "num_aic_cores", default=28))
    num_aiv = max(1, _read_signal_int(sig, "num_aiv_cores", default=56))
    grid_symbolic = _read_signal_bool(sig, "grid_is_symbolic")

    inflation_score = (
        min(1.0, 0.35 + 0.65 * min(1.0, (transfer_inflation - 1.30) / 1.70))
        if transfer_inflation > 1.30
        else 0.0
    )
    roundtrip_score = (
        0.65 if (gm_roundtrip > 0 and cube_ratio > 0.0) else (0.45 if gm_roundtrip > 0 else 0.0)
    )
    dma_roundtrip_score = min(1.0, max(inflation_score, roundtrip_score))

    short_burst_score = min(
        1.0,
        max(
            max(0.0, 1.0 - burst_eff),
            0.55 if (0 < contig_bytes < 16384) else 0.0,
            0.65 if unaligned_pad > 0 else 0.0,
            0.70 if mte2_inflight > 16 else 0.0,
        ),
    )

    l2_thrash_score = 0.0
    if transfer_inflation <= 1.30:
        if l2_hit < 0.65:
            l2_thrash_score = min(1.0, (1.0 - l2_hit) * (1.0 if mte2_wait >= 0.15 else 0.85))
        elif l2_hit < 0.80 and mte2_wait > 0.20:
            l2_thrash_score = min(1.0, max(1.0 - l2_hit, mte2_wait * 0.8))

    vec_loop_score = min(
        1.0,
        max(
            0.65 if not vloopv2_hit else 0.0,
            min(1.0, 0.30 * float(vf_breaks)),
            dual_collapse if dual_collapse > 0.25 else 0.0,
        ),
    )

    compute_util = max(aiv_vec, aic_mac)
    eff_eta_obs = effective_eta if effective_eta > 0.0 else roofline_eta
    ineffective_compute_score = (
        min(1.0, (1.0 - eff_eta_obs) * compute_util)
        if (compute_util > 0.55 and 0.0 < eff_eta_obs < 0.30)
        else 0.0
    )

    fixpipe_score = 0.0
    if pipe_m > 0 and pipe_fix > 1.15 * float(pipe_m):
        fixpipe_score = min(1.0, max(0.35, (float(pipe_fix) / float(pipe_m)) - 1.0))
    elif fixpipe_ratio > 0.45 and aic_mac < 0.45:
        fixpipe_score = min(1.0, fixpipe_ratio)

    subcore_imbalance_score = 0.0
    if aiv0_v > 0 and aiv1_v == 0:
        subcore_imbalance_score = 0.80 if cube_ratio > 0.0 else 0.70

    sram_spill_score = min(
        1.0,
        max(
            min(1.0, float(spill_bytes) / 256.0) if spill_bytes > 0 else 0.0,
            0.55 if stack_bytes > 0 else 0.0,
            0.85 if max_overflow > 0 else 0.0,
            0.65 if degraded_bufs > 0 else 0.0,
        ),
    )

    bank_conflict_score = (
        min(
            1.0,
            max(
                bank_conflict if bank_conflict > 0.08 else 0.0,
                ub_bank_risk,
                l1_bank_risk,
            ),
        )
        if spill_bytes == 0
        else 0.0
    )

    pseudo_db_score = min(
        1.0,
        max(
            0.75 if pseudo_db else 0.0,
            min(1.0, 0.30 * float(alias_ser)),
        ),
    )

    barrier_score = (
        min(
            1.0,
            max(
                sync_stall if sync_stall > 0.20 else 0.0,
                0.65 if local_membar else 0.0,
                0.55 if pipe_all > 0 else 0.0,
            ),
        )
        if (not pseudo_db and alias_ser == 0)
        else 0.0
    )

    wave_waste = (
        (1.0 - wave_occ / float(math.ceil(wave_occ))) if wave_occ > 0.0 else 0.0
    )
    active_cores = num_aic if cube_ratio > 0.0 else num_aiv
    underfill_score = (
        max(0.0, 1.0 - float(grid_blocks) / float(active_cores))
        if (not grid_symbolic and wave_occ > 0.0 and 0 < grid_blocks < active_cores)
        else 0.0
    )
    grid_wave_score = min(
        1.0,
        max(
            wave_waste if wave_waste > 0.18 else 0.0,
            underfill_score if underfill_score > 0.18 else 0.0,
        ),
    )

    is_scalar_trap = (l0_cube_cycles == 0 and l0_vec_cycles == 0) or (
        has_l1_summary and l1_vec_regs == 0 and aiv_vec < 0.02 and aic_mac == 0.0
    )
    scalar_loop_score = 1.0 if is_scalar_trap else 0.0

    scored: list[tuple[SymptomFamily, float]] = [
        (SymptomFamily.DMA_REDUNDANT_GM_ROUNDTRIP, round(dma_roundtrip_score, 6)),
        (SymptomFamily.DMA_UNALIGNED_SHORT_BURST, round(short_burst_score, 6)),
        (SymptomFamily.DMA_L2_THRASH_REUSE_MISS, round(l2_thrash_score, 6)),
        (SymptomFamily.COMPUTE_VECTOR_LOOP_DEGRADED, round(vec_loop_score, 6)),
        (SymptomFamily.COMPUTE_INEFFECTIVE_THROUGHPUT, round(ineffective_compute_score, 6)),
        (SymptomFamily.COMPUTE_FIXPIPE_ACCUM_IMBALANCE, round(fixpipe_score, 6)),
        (SymptomFamily.COMPUTE_1C2V_SUBCORE_IMBALANCE, round(subcore_imbalance_score, 6)),
        (SymptomFamily.SCALAR_LOOP_TRAP, round(scalar_loop_score, 6)),
        (SymptomFamily.SRAM_REGISTER_SPILL_STACK, round(sram_spill_score, 6)),
        (SymptomFamily.SRAM_UB_L1_BANK_CONFLICT, round(bank_conflict_score, 6)),
        (SymptomFamily.SYNC_PSEUDO_DOUBLE_BUFFER, round(pseudo_db_score, 6)),
        (SymptomFamily.SYNC_BARRIER_SERIALIZATION, round(barrier_score, 6)),
        (SymptomFamily.GRID_TAIL_WAVE_UNDERUTIL, round(grid_wave_score, 6)),
    ]
    scored.sort(
        key=lambda item: (
            -item[1],
            0 if item[0] == SymptomFamily.SCALAR_LOOP_TRAP else 1,
            item[0].value,
        )
    )
    return scored


def default_seed_rules_path() -> Path:
    """Return path to the bundled pure-English causal seed rule library."""
    return Path(__file__).resolve().with_name("seed_rules.yaml")


def load_seed_rules(
    path: str | Path | None = None,
    *,
    require_symptom_family: bool | None = None,
) -> tuple[AscendCausalRule, ...]:
    """Load and validate AscendCausalRule entries from YAML."""
    target_path = Path(path) if path is not None else default_seed_rules_path()
    is_bundled_seed = (
        path is None or target_path.resolve() == default_seed_rules_path().resolve()
    )
    raw = yaml.safe_load(target_path.read_text(encoding="utf-8"))
    if isinstance(raw, Mapping) and "rules" in raw:
        items = raw["rules"]
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError(f"invalid rules YAML root in {target_path}")
    rules: list[AscendCausalRule] = []
    for entry in items:
        rule = AscendCausalRule.from_dict(entry)
        enforce_family = (
            require_symptom_family
            if require_symptom_family is not None
            else (is_bundled_seed or not rule.source_provenance.startswith("flywheel:"))
        )
        if enforce_family and not rule.symptom_family:
            raise ValueError(
                f"seed rule {rule.rule_id!r} in {target_path} must declare a valid "
                "non-empty symptom_family"
            )
        rules.append(rule)
    return tuple(rules)


def save_rules_yaml(rules: tuple[AscendCausalRule, ...] | list[AscendCausalRule], path: str | Path) -> None:
    """Persist AscendCausalRule entries to YAML cleanly in sorted order."""
    payload = {"rules": [rule.to_dict() for rule in rules]}
    Path(path).write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )


__all__ = [
    "AscendCausalRule",
    "SYMPTOM_FAMILY_TO_AXIS",
    "SYMPTOM_FAMILY_TO_FACETS",
    "SymptomFamily",
    "UnsafePredicateError",
    "build_predicate_context",
    "default_seed_rules_path",
    "load_seed_rules",
    "normalize_runtime_backend",
    "rank_symptom_families",
    "safe_eval_predicate",
    "save_rules_yaml",
    "validate_rule_predicate",
]
