"""Six-tuple AscendCausalRule schema and safe AST inequality evaluator."""

from __future__ import annotations

import ast
from collections.abc import Mapping
from dataclasses import dataclass
import operator
from pathlib import Path
import yaml

from tilefoundry_ascend.ascend.passes.physical_ledger import ObjectivePhysicalLedger

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

    def __post_init__(self) -> None:
        if not self.rule_id or not self.transform_template.strip():
            raise ValueError("rule_id and transform_template must be non-empty")
        if self.scope not in ("cross_op", "operator_local"):
            raise ValueError(f"invalid rule scope {self.scope!r}; expected 'cross_op' or 'operator_local'")
        validate_rule_predicate(self.regime_guard)
        validate_rule_predicate(self.precondition)
        validate_rule_predicate(self.expected_postcondition)

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
        """Return whether both regime_guard and precondition hold in context."""
        return safe_eval_predicate(self.regime_guard, context) and safe_eval_predicate(
            self.precondition, context
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
            "transform_template": self.transform_template,
            "expected_postcondition": self.expected_postcondition,
            "source_provenance": self.source_provenance,
            "alpha_prior": self.alpha_prior,
            "beta_prior": self.beta_prior,
            "rationale": self.rationale,
            "priority": self.priority,
            "scope": self.scope,
            "origin_operator": self.origin_operator,
        }
        return payload

    @classmethod
    def from_dict(cls, entry: Mapping[str, object]) -> AscendCausalRule:
        """Deserialize rule from YAML/JSON mapping."""
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
    "measured_mte2_wait_ratio": 0.0,
    "measured_sync_stall_ratio": 0.0,
    "unexplained_gap_ratio": 0.0,
    "pseudo_double_buffer_serialized": False,
    "spill_bytes": 0,
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
    ns["cube_to_vector_ratio"] = (
        round(cube_cycles / float(max(1, vec_cycles)), 6) if cube_cycles > 0 else 0.0
    )
    ns["wave_occupancy"] = float(ledger.wave_occupancy)
    ns["grid_blocks"] = int(ledger.grid_blocks)
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
    ns["ub_utilization_ratio"] = round(float(ub_peak) / float(max(1, ub_cap)), 6)
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


def build_predicate_context(
    ledger: ObjectivePhysicalLedger,
    *,
    features: object | None = None,
    residual: Mapping[str, object] | object | None = None,
    compiler_summary: Mapping[str, object] | object | None = None,
    prev_ledger: ObjectivePhysicalLedger | None = None,
    extra: Mapping[str, object] | None = None,
    target: object | None = None,
) -> dict[str, object]:
    """Assemble a flat and dotted-path dictionary for safe rule predicate evaluation."""
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
    if compiler_summary is not None:
        comp_vals = _extract_typed_defaults(compiler_summary, _RESIDUAL_DEFAULTS)
        # Preserve optimistic vloopv2_hit=True when BishengResourceSummary had no loop diagnostics
        if isinstance(compiler_summary, Mapping):
            sw = int(compiler_summary.get("software_loops", 0) or 0)
            vl = int(compiler_summary.get("vloopv2_loops", 0) or 0)
        else:
            sw = int(getattr(compiler_summary, "software_loops", 0) or 0)
            vl = int(getattr(compiler_summary, "vloopv2_loops", 0) or 0)
        if sw == 0 and vl == 0 and "vloopv2_hit" in comp_vals:
            comp_vals["vloopv2_hit"] = True
        res_defaults.update(comp_vals)
    if residual is not None:
        res_defaults.update(_extract_typed_defaults(residual, _RESIDUAL_DEFAULTS))
        ctx.update(_extract_typed_defaults(residual, _TELEMETRY_CONTEXT_DEFAULTS))
    ctx.update(res_defaults)

    # Also allow bank_conflict_ratio (from TelemetryFeedback) to seed ub_bank_conflict_risk
    if float(ctx.get("ub_bank_conflict_risk", 0.0)) == 0.0 and float(ctx.get("bank_conflict_ratio", 0.0)) > 0.0:
        ctx["ub_bank_conflict_risk"] = float(ctx["bank_conflict_ratio"])

    if prev_ledger is not None:
        prev_ns = _ledger_to_namespace(prev_ledger)
        for key, val in _RESIDUAL_DEFAULTS.items():
            prev_ns.setdefault(key, val)
        for key, val in _TELEMETRY_CONTEXT_DEFAULTS.items():
            prev_ns.setdefault(key, val)
        ctx["prev"] = prev_ns
    else:
        ctx["prev"] = dict(ctx)

    if extra:
        ctx.update(extra)
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


def default_seed_rules_path() -> Path:
    """Return path to the bundled pure-English causal seed rule library."""
    return Path(__file__).resolve().with_name("seed_rules.yaml")


def load_seed_rules(path: str | Path | None = None) -> tuple[AscendCausalRule, ...]:
    """Load and validate AscendCausalRule entries from YAML."""
    target_path = Path(path) if path is not None else default_seed_rules_path()
    raw = yaml.safe_load(target_path.read_text(encoding="utf-8"))
    if isinstance(raw, Mapping) and "rules" in raw:
        items = raw["rules"]
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError(f"invalid rules YAML root in {target_path}")
    rules: list[AscendCausalRule] = []
    for entry in items:
        rules.append(AscendCausalRule.from_dict(entry))
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
    "UnsafePredicateError",
    "build_predicate_context",
    "default_seed_rules_path",
    "load_seed_rules",
    "safe_eval_predicate",
    "save_rules_yaml",
    "validate_rule_predicate",
]
