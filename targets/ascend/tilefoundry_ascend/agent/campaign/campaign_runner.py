"""Unattended performance-driven campaign runner with checkpointing and L3 rollback guard."""

from __future__ import annotations

import argparse
import ast
import asyncio
import importlib.util
import inspect
import json
import os
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from tilefoundry_ascend.agent.campaign.anchor_specs import (
    DEFAULT_REGIME_SPECS,
    AnchorCaseSpec,
    PerfTargetSpec,
    RegimeTargetSpec,
    append_dynamic_anchor,
    default_source_root,
    get_default_anchors,
    get_default_perf_target,
    get_default_regime_specs,
    get_operator_module_relpath,
)
from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (
    CandidateSurrogateResult,
    L0L1CandidateCheck,
    L3AcceptanceResult,
    L4CannbenchResult,
    run_l0_l1_round_filter,
    run_l1_compile_check,
    run_l3_full_contract_via_agent_gates,
    run_l4_official_cannbench_via_agent_gates,
    run_surrogate_batch_via_agent_gates,
)
from tilefoundry_ascend.agent.flywheel.ablation_distiller import FlywheelDistiller
from tilefoundry_ascend.agent.kb.bayesian_tracker import BayesianRuleTracker
from tilefoundry_ascend.agent.kb.schema import (
    AscendCausalRule,
    build_predicate_context,
    load_seed_rules,
)
from tilefoundry_ascend.agent.orchestrator.gate_funnel import (
    GateFunnel,
    StageAResult,
    StageBResult,
    run_stage_a_static,
)
from tilefoundry_ascend.agent.orchestrator.llm_client import (
    CREATIVE_RESPONSE_SCHEMA,
    GenerationRequest,
    KernelLLMBackend,
    create_llm_backend_from_env,
    extract_creative_payload,
    extract_python_dsl,
)
from tilefoundry_ascend.agent.orchestrator.portfolio_runner import (
    CREATIVE_FOCUS_HINTS,
    TILELANG_ASCEND_PRIMITIVE_MENU,
    build_prescreen_summary_from_kwargs,
    summarize_excluded_mechanisms,
)
from tilefoundry_ascend.agent.orchestrator.residual_beam import (
    BeamCandidate,
    TwoLaneBeamSearch,
    compute_static_dynamic_residual,
    extract_structural_signature,
)
from tilefoundry_ascend.agent.orchestrator.twin_bridge import format_hir_tir_navigation_section
from tilefoundry_ascend.ascend.passes.hir_tir_bridge import (
    HirTirNavigationPacket,
    compile_hir_navigation_packet,
)
from tilefoundry_ascend.ascend.passes.physical_ledger import (
    ObjectivePhysicalLedger,
    build_objective_physical_ledger,
)
from tilefoundry_ascend.ascend.target import AscendTarget


@dataclass(frozen=True)
class CampaignConfig:
    """Configuration controlling stopping policy, escalation, and gate thresholds."""

    operator_name: str
    max_rounds: int = 12
    max_wall_clock_min: float = 180.0
    k_rule_workers: int = 2
    max_syntax_repairs: int = 1
    min_l2_gain_for_l3: float = 0.015
    min_surrogate_gain: float = 0.25
    min_flywheel_gain: float = 0.03
    stagnation_escalate_rounds: int = 2
    stagnation_freeze_rounds: int = 4
    disable_early_goal_stop: bool = False
    run_final_l4: bool = False
    l4_regression_ratio: float = 0.95
    device_id: str = "1"
    warmup: int = 3
    repeat: int = 5


@dataclass(frozen=True)
class CampaignRoundRecord:
    """Telemetry snapshot recorded at the completion of one campaign round."""

    round_idx: int
    regime_tag: str
    builder_name: str
    parent_lane: str
    creative_effort: str
    l0_l1_passed_ids: tuple[str, ...]
    l0_l1_failed: tuple[tuple[str, str, str], ...]
    best_round_speedup: float
    best_so_far_speedup: float
    promoted: bool
    rolled_back_l3: bool
    stagnation_rounds_after: int
    admitted_rule_ids: tuple[str, ...] = ()
    best_round_surrogate_score: float = 0.0
    best_so_far_surrogate_score: float = 0.0

    def to_dict(self) -> dict[str, object]:
        """Serialize CampaignRoundRecord into a JSON-compatible dictionary."""
        return {
            "round_idx": self.round_idx,
            "regime_tag": self.regime_tag,
            "builder_name": self.builder_name,
            "parent_lane": self.parent_lane,
            "creative_effort": self.creative_effort,
            "l0_l1_passed_ids": list(self.l0_l1_passed_ids),
            "l0_l1_failed": [list(item) for item in self.l0_l1_failed],
            "best_round_speedup": round(self.best_round_speedup, 6),
            "best_so_far_speedup": round(self.best_so_far_speedup, 6),
            "best_round_surrogate_score": round(self.best_round_surrogate_score, 6),
            "best_so_far_surrogate_score": round(self.best_so_far_surrogate_score, 6),
            "promoted": self.promoted,
            "rolled_back_l3": self.rolled_back_l3,
            "stagnation_rounds_after": self.stagnation_rounds_after,
            "admitted_rule_ids": list(self.admitted_rule_ids),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> CampaignRoundRecord:
        """Reconstruct CampaignRoundRecord from a JSON mapping."""
        failed_raw = payload.get("l0_l1_failed", ())
        failed = tuple(
            (str(item[0]), str(item[1]), str(item[2]))
            for item in failed_raw
            if isinstance(item, Sequence) and len(item) >= 3
        )
        return cls(
            round_idx=int(payload["round_idx"]),
            regime_tag=str(payload["regime_tag"]),
            builder_name=str(payload["builder_name"]),
            parent_lane=str(payload["parent_lane"]),
            creative_effort=str(payload["creative_effort"]),
            l0_l1_passed_ids=tuple(str(x) for x in payload.get("l0_l1_passed_ids", ())),
            l0_l1_failed=failed,
            best_round_speedup=float(payload["best_round_speedup"]),
            best_so_far_speedup=float(payload["best_so_far_speedup"]),
            promoted=bool(payload["promoted"]),
            rolled_back_l3=bool(payload["rolled_back_l3"]),
            stagnation_rounds_after=int(payload["stagnation_rounds_after"]),
            admitted_rule_ids=tuple(str(x) for x in payload.get("admitted_rule_ids", ())),
            best_round_surrogate_score=float(payload.get("best_round_surrogate_score", 0.0)),
            best_so_far_surrogate_score=float(payload.get("best_so_far_surrogate_score", 0.0)),
        )


@dataclass
class CampaignReport:
    """Final convergence summary emitted by PerfTargetCampaignRunner."""

    operator_name: str
    stop_reason: str
    total_rounds: int
    baseline_result: CandidateSurrogateResult
    best_result: CandidateSurrogateResult
    rounds: tuple[CampaignRoundRecord, ...]
    promoted_rounds: tuple[int, ...] = ()
    rolled_back_rounds: tuple[int, ...] = ()
    escalated_rounds: tuple[int, ...] = ()
    frozen_regimes: tuple[str, ...] = ()
    active_anchors: tuple[AnchorCaseSpec, ...] = ()
    admitted_rule_ids: tuple[str, ...] = ()
    l4_result: L4CannbenchResult | None = None

    def to_dict(self) -> dict[str, object]:
        """Serialize CampaignReport into a JSON-compatible dictionary."""
        return {
            "operator_name": self.operator_name,
            "stop_reason": self.stop_reason,
            "total_rounds": self.total_rounds,
            "baseline_result": self.baseline_result.to_dict(),
            "best_result": self.best_result.to_dict(),
            "rounds": [r.to_dict() for r in self.rounds],
            "promoted_rounds": list(self.promoted_rounds),
            "rolled_back_rounds": list(self.rolled_back_rounds),
            "escalated_rounds": list(self.escalated_rounds),
            "frozen_regimes": list(self.frozen_regimes),
            "active_anchors": [a.to_dict() for a in self.active_anchors],
            "admitted_rule_ids": list(self.admitted_rule_ids),
            "l4_result": self.l4_result.to_dict() if self.l4_result else None,
        }


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            tmp_path = Path(fh.name)
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


_FORBIDDEN_SLICE_SYMBOLS: frozenset[str] = frozenset(
    {"attention_residual", "mla", "_get_kernel", "_KERNEL_CACHE"}
)

_HIR_DIM_ALIASES: dict[str, str] = {
    "N": "n_depth",
    "B": "b_dim",
    "S": "s_dim",
    "S_kv": "s_kv",
    "D": "d_dim",
    "N_q": "n_q",
    "N_kv": "n_kv",
    "D_nope": "d_nope",
    "D_rope": "d_rope",
}


@dataclass(frozen=True)
class ResolvedRegimeConfig:
    """Resolved builder, tile hyperparameters, and pass/compile flags for a regime."""

    builder_name: str
    builder_kwargs: dict[str, object]
    pass_configs: dict[str, object]
    compile_flags: tuple[str, ...]
    selector_name: str = ""


def _filter_callable_kwargs(fn: object, kwargs: Mapping[str, object]) -> dict[str, object]:
    sig = inspect.signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def _norm_flag_seq(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        text = raw.strip()
        return text.split() if text else []
    if isinstance(raw, (tuple, list)) or (
        hasattr(raw, "__iter__") and not isinstance(raw, (bytes, bytearray, Mapping))
    ):
        out: list[str] = []
        for item in raw:
            val = str(getattr(item, "value", item)).strip()
            if val and val not in out:
                out.append(val)
        return out
    text = str(raw).strip()
    return [text] if text else []


def resolve_regime_config(
    module_source_or_path: str | Path,
    regime_spec: RegimeTargetSpec,
    dims: Mapping[str, object] | None = None,
) -> ResolvedRegimeConfig:
    """Invoke regime config_selector_fn dynamically to resolve builder kwargs and flags."""
    if isinstance(module_source_or_path, Path):
        mod_path = module_source_or_path.resolve()
        src_text = mod_path.read_text(encoding="utf-8")
        is_existing_file = True
    else:
        raw_str = str(module_source_or_path)
        is_existing_file = "\n" not in raw_str and Path(raw_str).is_file()
        if is_existing_file:
            mod_path = Path(raw_str).resolve()
            src_text = mod_path.read_text(encoding="utf-8")
        else:
            mod_path = None
            src_text = raw_str

    def _load_and_resolve(file_path: Path) -> ResolvedRegimeConfig:
        spec = importlib.util.spec_from_file_location("_tf_resolve_cfg", str(file_path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Failed to load module spec from {file_path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        context: dict[str, object] = dict(regime_spec.sample_builder_kwargs)
        for h_key, h_val in regime_spec.representative_hir_dims.items():
            mapped = _HIR_DIM_ALIASES.get(str(h_key), str(h_key))
            if mapped not in context:
                context[mapped] = h_val
        if dims:
            for d_key, d_val in dims.items():
                mapped = _HIR_DIM_ALIASES.get(str(d_key), str(d_key))
                context[mapped] = d_val

        pass_configs: dict[str, object] = {}
        compile_flags: list[str] = []
        selector_name = regime_spec.config_selector_fn
        if selector_name and callable(getattr(mod, selector_name, None)):
            selector_fn = getattr(mod, selector_name)
            sel_args = _filter_callable_kwargs(selector_fn, context)
            sel_out = selector_fn(**sel_args)
            if isinstance(sel_out, Mapping):
                raw_pc = sel_out.get("pass_configs")
                if isinstance(raw_pc, Mapping):
                    for k, v in raw_pc.items():
                        if str(k) == "tl.device_compile_flags":
                            for fl in _norm_flag_seq(v):
                                if fl not in compile_flags:
                                    compile_flags.append(fl)
                        else:
                            pass_configs[str(k)] = v
                elif raw_pc is not None:
                    for fl in _norm_flag_seq(raw_pc):
                        if fl not in compile_flags:
                            compile_flags.append(fl)
                for fl in _norm_flag_seq(sel_out.get("compile_flags")):
                    if fl not in compile_flags:
                        compile_flags.append(fl)
                for k, v in sel_out.items():
                    if k not in ("pass_configs", "compile_flags"):
                        context[str(k)] = v

        builder_name = ""
        for cand in regime_spec.builder_candidates:
            if callable(getattr(mod, cand, None)):
                builder_name = cand
                break
        if not builder_name:
            for cand in regime_spec.editable_symbols:
                if cand != selector_name and callable(getattr(mod, cand, None)):
                    builder_name = cand
                    break
        if not builder_name:
            raise ValueError(
                f"No builder callable matching {regime_spec.builder_candidates} in module."
            )
        builder_fn = getattr(mod, builder_name)
        builder_kwargs = _filter_callable_kwargs(builder_fn, context)
        return ResolvedRegimeConfig(
            builder_name=builder_name,
            builder_kwargs=builder_kwargs,
            pass_configs=pass_configs,
            compile_flags=tuple(compile_flags),
            selector_name=selector_name,
        )

    if mod_path is not None and is_existing_file:
        return _load_and_resolve(mod_path)
    with tempfile.TemporaryDirectory(prefix="tf_resolve_cfg_") as tmp_dir:
        tmp_file = Path(tmp_dir) / "module_src.py"
        tmp_file.write_text(src_text, encoding="utf-8")
        return _load_and_resolve(tmp_file)


@dataclass(frozen=True)
class _TopLevelSymbol:
    name: str
    kind: str
    start: int
    end: int


def _stmt_line_span(node: ast.stmt) -> tuple[int, int]:
    decs = getattr(node, "decorator_list", ())
    start = min((dec.lineno for dec in decs), default=node.lineno)
    end = node.end_lineno or node.lineno
    return start, end


def _collect_top_level_symbols(tree: ast.Module) -> list[_TopLevelSymbol]:
    items: list[_TopLevelSymbol] = []
    for node in tree.body:
        start, end = _stmt_line_span(node)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            items.append(_TopLevelSymbol(node.name, "func", start, end))
        elif isinstance(node, ast.Assign):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                items.append(_TopLevelSymbol(node.targets[0].id, "assign", start, end))
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                items.append(_TopLevelSymbol(node.target.id, "assign", start, end))
    return items


def _find_top_level_func(tree: ast.Module, names: Sequence[str]) -> ast.FunctionDef | None:
    by_name = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    for name in names:
        if name in by_name:
            return by_name[name]
    return None


def _func_line_span(node: ast.FunctionDef) -> tuple[int, int]:
    start = min((dec.lineno for dec in node.decorator_list), default=node.lineno)
    end = node.end_lineno or node.lineno
    return start, end


def _other_regime_forbidden_symbols(regime_spec: RegimeTargetSpec) -> set[str]:
    forbidden: set[str] = set(_FORBIDDEN_SLICE_SYMBOLS)
    own_editable = set(regime_spec.editable_symbols)
    for specs in DEFAULT_REGIME_SPECS.values():
        for spec in specs:
            if spec.regime_tag == regime_spec.regime_tag:
                continue
            for sym in spec.editable_symbols:
                if sym not in own_editable:
                    forbidden.add(sym)
            if spec.config_selector_fn and spec.config_selector_fn not in own_editable:
                forbidden.add(spec.config_selector_fn)
    return forbidden


def extract_regime_snippet(
    module_source: str,
    regime_spec: RegimeTargetSpec,
) -> tuple[str, str]:
    """Extract target regime multi-symbol slice (selector, builders, helpers) cleanly."""
    tree = ast.parse(module_source)
    lines = module_source.splitlines(keepends=True)

    if regime_spec.editable_symbols:
        symbols = _collect_top_level_symbols(tree)
        editable_set = set(regime_spec.editable_symbols)
        prefix = regime_spec.allowed_helper_prefix
        matched = [
            sym
            for sym in symbols
            if sym.name in editable_set or (prefix and sym.name.startswith(prefix))
        ]
        if matched:
            func_names = {sym.name for sym in matched if sym.kind == "func"}
            primary = ""
            for cand in regime_spec.builder_candidates:
                if cand in func_names:
                    primary = cand
                    break
            if not primary:
                primary = next(
                    (sym.name for sym in matched if sym.kind == "func"),
                    matched[0].name,
                )
            blocks = [
                "".join(lines[sym.start - 1 : sym.end]).rstrip()
                for sym in matched
            ]
            return primary, "\n\n".join(blocks).rstrip() + "\n"

    node = _find_top_level_func(tree, regime_spec.builder_candidates)
    if node is None:
        raise ValueError(
            f"No builder function matching {regime_spec.builder_candidates} found in module."
        )
    start, end = _func_line_span(node)
    return node.name, "".join(lines[start - 1 : end]).rstrip() + "\n"


def _patch_single_func_into_module(
    module_source: str,
    func_name: str,
    candidate_dsl: str,
) -> str:
    cand_tree = ast.parse(candidate_dsl)
    cand_node = _find_top_level_func(cand_tree, (func_name,))
    if cand_node is None:
        raise ValueError(f"Candidate source must define function {func_name!r}.")

    cand_lines = candidate_dsl.splitlines(keepends=True)
    c_start, c_end = _func_line_span(cand_node)
    replacement = "".join(cand_lines[c_start - 1 : c_end])
    if not replacement.endswith("\n"):
        replacement += "\n"

    orig_tree = ast.parse(module_source)
    orig_node = _find_top_level_func(orig_tree, (func_name,))
    if orig_node is None:
        raise ValueError(f"Original module missing target function {func_name!r}.")
    o_start, o_end = _func_line_span(orig_node)
    orig_lines = module_source.splitlines(keepends=True)
    patched = "".join([*orig_lines[: o_start - 1], replacement, *orig_lines[o_end:]])
    ast.parse(patched)
    return patched


def patch_regime_into_module(
    module_source: str,
    regime_or_func: RegimeTargetSpec | str,
    candidate_raw_source: str,
) -> str:
    """Splice validated single-function or multi-symbol regime slice into module_source."""
    candidate_dsl = extract_python_dsl(candidate_raw_source)
    if isinstance(regime_or_func, str):
        return _patch_single_func_into_module(module_source, regime_or_func, candidate_dsl)

    regime_spec = regime_or_func
    orig_tree = ast.parse(module_source)
    if not regime_spec.editable_symbols:
        orig_node = _find_top_level_func(orig_tree, regime_spec.builder_candidates)
        if orig_node is None:
            raise ValueError(
                f"Original module missing builder in {regime_spec.builder_candidates}."
            )
        return _patch_single_func_into_module(module_source, orig_node.name, candidate_dsl)

    orig_symbols = _collect_top_level_symbols(orig_tree)
    orig_by_name = {sym.name: sym for sym in orig_symbols}
    editable_set = set(regime_spec.editable_symbols)
    prefix = regime_spec.allowed_helper_prefix

    required_symbols = [sym for sym in regime_spec.editable_symbols if sym in orig_by_name]
    if not required_symbols:
        orig_node = _find_top_level_func(orig_tree, regime_spec.builder_candidates)
        if orig_node is None:
            raise ValueError(
                f"Original module missing regime symbols {regime_spec.editable_symbols}."
            )
        return _patch_single_func_into_module(module_source, orig_node.name, candidate_dsl)

    cand_tree = ast.parse(candidate_dsl)
    cand_symbols: list[_TopLevelSymbol] = []
    seen_cand: set[str] = set()
    forbidden = _other_regime_forbidden_symbols(regime_spec)

    for node in cand_tree.body:
        start, end = _stmt_line_span(node)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name, kind = node.name, "func"
        elif isinstance(node, ast.Assign):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                raise ValueError("Candidate assignment must target a single variable name.")
            name, kind = node.targets[0].id, "assign"
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name, kind = node.target.id, "assign"
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        elif (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            continue
        else:
            raise ValueError(
                f"Candidate snippet contains unsupported top-level {type(node).__name__}."
            )

        if name in seen_cand:
            raise ValueError(f"Candidate snippet defines duplicate symbol {name!r}.")
        seen_cand.add(name)

        if name in forbidden:
            raise ValueError(f"Candidate cannot define or modify protected symbol {name!r}.")
        if name not in editable_set:
            if not prefix or not name.startswith(prefix):
                raise ValueError(
                    f"Unauthorized top-level symbol {name!r}; expected {editable_set} "
                    f"or prefix {prefix!r}."
                )
        cand_symbols.append(_TopLevelSymbol(name, kind, start, end))

    cand_by_name = {sym.name: sym for sym in cand_symbols}
    missing = [name for name in required_symbols if name not in cand_by_name]
    if missing:
        raise ValueError(f"Candidate snippet missing required regime symbol(s): {missing}.")

    cand_lines = candidate_dsl.splitlines(keepends=True)
    cand_slices = {
        sym.name: "".join(cand_lines[sym.start - 1 : sym.end]).rstrip() + "\n"
        for sym in cand_symbols
    }

    orig_regime_items = [
        sym
        for sym in orig_symbols
        if sym.name in editable_set or (prefix and sym.name.startswith(prefix))
    ]
    new_helpers = [
        sym
        for sym in cand_symbols
        if sym.name not in editable_set and sym.name not in orig_by_name
    ]
    insert_line_idx = orig_regime_items[0].start - 1

    orig_lines = module_source.splitlines(keepends=True)
    for item in reversed(orig_regime_items):
        if item.name in cand_slices:
            orig_lines[item.start - 1 : item.end] = [cand_slices[item.name]]
        else:
            del_end = item.end
            while del_end < len(orig_lines) and orig_lines[del_end].strip() == "":
                del_end += 1
            del orig_lines[item.start - 1 : del_end]

    if new_helpers:
        helper_block = "".join(cand_slices[h.name] + "\n" for h in new_helpers)
        orig_lines[insert_line_idx:insert_line_idx] = [helper_block]

    patched = "".join(orig_lines)
    ast.parse(patched)
    return patched


def _surrogate_rank_key(res: CandidateSurrogateResult) -> tuple[float, float, float]:
    return (res.surrogate_score, res.min_case_speedup, res.avg_speedup)


def _fallback_ledger() -> ObjectivePhysicalLedger:
    return build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=100,
        pipe_cycles={"AIV0.PIPE_V": 100},
        waterlines={},
    )


def _format_anchor_telemetry(
    result: CandidateSurrogateResult,
    regime_tag: str,
) -> str:
    rows = []
    for anchor in result.anchors:
        rows.append(
            {
                "case_num": anchor.case_num,
                "regime_tag": anchor.regime_tag,
                "target_regime": anchor.regime_tag == regime_tag,
                "accuracy_passed": anchor.accuracy_passed,
                "cand_us": round(anchor.cand_us, 4),
                "ref_perf_us": round(anchor.ref_perf_us, 4),
                "t_hw_us": round(anchor.t_hw_us, 4),
                "speedup": round(anchor.speedup, 4),
                "roofline_eta": round(anchor.roofline_eta, 4),
                "headroom": round(max(0.0, 1.0 - anchor.roofline_eta), 4),
            }
        )
    return json.dumps(
        {
            "operator": result.operator_name,
            "avg_speedup": round(result.avg_speedup, 4),
            "min_case_speedup": round(result.min_case_speedup, 4),
            "avg_roofline_eta": round(result.avg_roofline_eta, 4),
            "surrogate_score": round(result.surrogate_score, 4),
            "anchors": rows,
        },
        indent=2,
    )


class PerfTargetCampaignRunner:
    """Host-CPU multi-round optimization controller driving L0->L1->L2->L3->L4 pyramid."""

    def __init__(
        self,
        *,
        config: CampaignConfig,
        target_module_path: str | Path | None = None,
        perf_target: PerfTargetSpec | None = None,
        anchors: Sequence[AnchorCaseSpec] | None = None,
        regime_specs: Sequence[RegimeTargetSpec] | None = None,
        llm_backend: KernelLLMBackend | None = None,
        tracker: BayesianRuleTracker | None = None,
        beam: TwoLaneBeamSearch | None = None,
        flywheel_distiller: FlywheelDistiller | None = None,
        l0_checker: Callable[[str, RegimeTargetSpec], StageAResult] | None = None,
        l1_compiler: Callable[[str, RegimeTargetSpec], StageBResult] | None = None,
        l2_batch_evaluator: Callable[
            [Sequence[tuple[str, Path]], Sequence[AnchorCaseSpec]],
            Mapping[str, CandidateSurrogateResult],
        ]
        | None = None,
        l3_contract_runner: Callable[[Path], L3AcceptanceResult] | None = None,
        l4_cannbench_runner: Callable[[Path], L4CannbenchResult] | None = None,
        checkpoint_path: str | Path | None = None,
        campaign_kb_path: str | Path | None = None,
        source_root: str | Path | None = None,
        env_file: str | Path | None = None,
        cannbench_repo: str | Path | None = None,
        authored_hir_path: str | Path | None = None,
        scheduled_hir_path: str | Path | None = None,
        enable_track_router: bool = False,
        target: AscendTarget | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config
        self._source_root = (
            Path(source_root).resolve() if source_root is not None else default_source_root()
        )
        self._target_module_path = (
            Path(target_module_path).resolve()
            if target_module_path is not None
            else (self._source_root / get_operator_module_relpath(config.operator_name)).resolve()
        )
        self._perf_target = perf_target or get_default_perf_target(config.operator_name)
        self._initial_anchors = tuple(
            anchors if anchors is not None else get_default_anchors(config.operator_name)
        )
        self._regime_specs = tuple(
            regime_specs
            if regime_specs is not None
            else get_default_regime_specs(config.operator_name)
        )
        self._env_file = Path(env_file).resolve() if env_file is not None else None
        self._cannbench_repo = Path(cannbench_repo).resolve() if cannbench_repo is not None else None
        self._backend = llm_backend
        self._tracker = tracker or BayesianRuleTracker.from_seed_yaml()
        self._beam = beam or TwoLaneBeamSearch()
        self._flywheel = flywheel_distiller
        self._l0_checker = l0_checker
        self._l1_compiler = l1_compiler
        self._l2_batch_evaluator = l2_batch_evaluator
        self._l3_contract_runner = l3_contract_runner
        self._l4_cannbench_runner = l4_cannbench_runner
        self._checkpoint_path = Path(checkpoint_path).resolve() if checkpoint_path else None
        self._campaign_kb_path = Path(campaign_kb_path).resolve() if campaign_kb_path else None
        self._enable_track_router = bool(enable_track_router)
        self._target = target or AscendTarget("ascend.950pr")
        self._clock = clock

        if authored_hir_path is not None:
            self._authored_hir_path: Path | None = Path(authored_hir_path).resolve()
        else:
            sibling_model = self._target_module_path.parent / "model.py"
            self._authored_hir_path = sibling_model.resolve() if sibling_model.is_file() else None

        if scheduled_hir_path is not None:
            self._scheduled_hir_path: Path | None = Path(scheduled_hir_path).resolve()
        else:
            sibling_sched = self._target_module_path.parent / "scheduled.py"
            self._scheduled_hir_path = sibling_sched.resolve() if sibling_sched.is_file() else None

        self._regime_nav_packets: dict[str, HirTirNavigationPacket] = (
            self._compile_regime_nav_packets()
        )

    def _compile_regime_nav_packets(self) -> dict[str, HirTirNavigationPacket]:
        if self._authored_hir_path is None or not self._authored_hir_path.is_file():
            return {}
        try:
            authored_hir_source = self._authored_hir_path.read_text(encoding="utf-8")
        except OSError:
            return {}
        scheduled_hir_source: str | None = None
        if self._scheduled_hir_path is not None and self._scheduled_hir_path.is_file():
            try:
                scheduled_hir_source = self._scheduled_hir_path.read_text(encoding="utf-8")
            except OSError:
                scheduled_hir_source = None

        packets: dict[str, HirTirNavigationPacket] = {}
        for regime_spec in self._regime_specs:
            try:
                packets[regime_spec.regime_tag] = compile_hir_navigation_packet(
                    authored_hir_source,
                    scheduled_hir_source=scheduled_hir_source,
                    anchor_dims=regime_spec.representative_hir_dims,
                    target=self._target,
                )
            except Exception:
                continue
        return packets

    def _get_backend(self) -> KernelLLMBackend:
        if self._backend is None:
            self._backend = create_llm_backend_from_env()
        return self._backend

    def _default_l0_checker(
        self,
        module_source: str,
        regime_spec: RegimeTargetSpec,
    ) -> StageAResult:
        try:
            resolved = resolve_regime_config(module_source, regime_spec)
        except Exception as exc:
            return StageAResult(
                passed=False,
                prim_func=None,
                ledger=None,
                error_message=f"stage_a_config_resolve_error: {type(exc).__name__}: {exc}",
            )
        return run_stage_a_static(
            module_source,
            factory_name=resolved.builder_name,
            factory_kwargs=resolved.builder_kwargs,
            pass_configs=resolved.pass_configs or None,
            compile_flags=resolved.compile_flags or None,
            target=self._target,
            enable_track_router=self._enable_track_router,
        )

    def _run_l0(self, module_source: str, regime_spec: RegimeTargetSpec) -> StageAResult:
        if self._l0_checker is not None:
            return self._l0_checker(module_source, regime_spec)
        return self._default_l0_checker(module_source, regime_spec)

    def _run_l1(self, module_source: str, regime_spec: RegimeTargetSpec) -> StageBResult:
        if self._l1_compiler is not None:
            return self._l1_compiler(module_source, regime_spec)
        try:
            resolved = resolve_regime_config(module_source, regime_spec)
        except Exception as exc:
            return StageBResult(
                passed=False,
                error_message=f"l1_config_resolve_error: {type(exc).__name__}: {exc}",
            )
        return run_l1_compile_check(
            module_source,
            builder_name=resolved.builder_name,
            builder_kwargs=resolved.builder_kwargs,
            config_selector_name=regime_spec.config_selector_fn or None,
            pass_configs=resolved.pass_configs or None,
            compile_flags=resolved.compile_flags or None,
            source_root=self._source_root,
            env_file=self._env_file,
        )

    def _run_l2_batch(
        self,
        candidates: Sequence[tuple[str, Path]],
        active_anchors: Sequence[AnchorCaseSpec],
    ) -> dict[str, CandidateSurrogateResult]:
        if self._l2_batch_evaluator is not None:
            return dict(self._l2_batch_evaluator(candidates, active_anchors))
        results, _run_id, _receipt = run_surrogate_batch_via_agent_gates(
            self._config.operator_name,
            candidates,
            active_anchors,
            source_root=self._source_root,
            env_file=self._env_file,
            device_id=self._config.device_id,
            warmup=self._config.warmup,
            repeat=self._config.repeat,
            require_all_passed=False,
        )
        return results

    def _run_l3_contract(self, module_path: Path) -> L3AcceptanceResult:
        if self._l3_contract_runner is not None:
            return self._l3_contract_runner(module_path)
        return run_l3_full_contract_via_agent_gates(
            source_root=self._source_root,
            env_file=self._env_file,
            device_id=self._config.device_id,
        )

    def _run_l4_cannbench(self, module_path: Path) -> L4CannbenchResult:
        if self._l4_cannbench_runner is not None:
            return self._l4_cannbench_runner(module_path)
        return run_l4_official_cannbench_via_agent_gates(
            operator_name=self._config.operator_name,
            source_root=self._source_root,
            env_file=self._env_file,
            device_id=self._config.device_id,
        )

    def _is_regime_satisfied(
        self,
        regime_spec: RegimeTargetSpec,
        best_result: CandidateSurrogateResult,
        active_anchors: Sequence[AnchorCaseSpec],
    ) -> bool:
        anchor_nums = {
            a.case_num for a in active_anchors if a.regime_tag == regime_spec.regime_tag
        }
        matched = [a for a in best_result.anchors if a.case_num in anchor_nums]
        if not matched:
            return False
        for metric in matched:
            if not metric.accuracy_passed:
                return False
            if metric.roofline_eta < self._perf_target.target_roofline_eta:
                return False
            if metric.speedup < self._perf_target.target_min_case_speedup:
                return False
            max_us = self._perf_target.per_anchor_max_us.get(metric.case_num)
            if max_us is not None and metric.cand_us > max_us:
                return False
        return True

    def _allow_early_stop(self, rounds_completed: int) -> bool:
        if self._config.disable_early_goal_stop:
            return False
        return rounds_completed > 0 and rounds_completed >= self._perf_target.min_optimization_rounds

    def _select_active_regime(
        self,
        current_regime: str | None,
        frozen_regimes: set[str],
        best_result: CandidateSurrogateResult,
        active_anchors: Sequence[AnchorCaseSpec],
        *,
        allow_satisfied_fallback: bool = False,
    ) -> RegimeTargetSpec | None:
        by_tag = {r.regime_tag: r for r in self._regime_specs}
        if current_regime and current_regime not in frozen_regimes and current_regime in by_tag:
            spec = by_tag[current_regime]
            if not self._is_regime_satisfied(spec, best_result, active_anchors):
                return spec

        candidates = [
            spec
            for spec in self._regime_specs
            if spec.regime_tag not in frozen_regimes
            and not self._is_regime_satisfied(spec, best_result, active_anchors)
        ]
        if not candidates and allow_satisfied_fallback:
            if current_regime and current_regime not in frozen_regimes and current_regime in by_tag:
                return by_tag[current_regime]
            candidates = [
                spec for spec in self._regime_specs if spec.regime_tag not in frozen_regimes
            ]
        if not candidates:
            return None
        candidates.sort(
            key=lambda s: best_result.regime_headroom(s.regime_tag),
            reverse=True,
        )
        return candidates[0]

    @staticmethod
    def _slice_contract_instruction(func_name: str, regime_spec: RegimeTargetSpec) -> str:
        if not regime_spec.editable_symbols:
            return f"function '{func_name}'"
        syms = ", ".join(regime_spec.editable_symbols)
        helper_note = (
            f" plus optional '{regime_spec.allowed_helper_prefix}*' helpers"
            if regime_spec.allowed_helper_prefix
            else ""
        )
        return (
            f"ALL required slice symbols ({syms}){helper_note}; do NOT define or modify "
            "public entrypoints or symbols from other regimes"
        )

    def _build_exploit_regime_request(
        self,
        *,
        snippet_source: str,
        func_name: str,
        regime_spec: RegimeTargetSpec,
        rule: AscendCausalRule,
        best_result: CandidateSurrogateResult,
        nav_packet: HirTirNavigationPacket | None = None,
        parent_ledger: ObjectivePhysicalLedger | None = None,
        track: str = "unconstrained",
    ) -> GenerationRequest:
        slice_desc = self._slice_contract_instruction(func_name, regime_spec)
        sys_prompt = (
            "You are an Ascend 950PR TileLang DSL performance engineer. Apply the selected causal "
            f"optimization rule cleanly while preserving exact numerical semantics, dynamic shape "
            f"compatibility, and {slice_desc}. Output ONLY Python code for this regime slice."
        )
        if not self._enable_track_router or track in ("", "unconstrained"):
            track_section = (
                f"### TILELANG ASCEND PRIMITIVE REFERENCE\n{TILELANG_ASCEND_PRIMITIVE_MENU}\n\n"
            )
        else:
            track_section = f"### TARGET TRACK\n{track}\n\n"
        if nav_packet is not None:
            nav_text = format_hir_tir_navigation_section(
                nav_packet,
                current_tl_ledger=parent_ledger,
            )
            nav_section = f"{nav_text}\n\n"
        else:
            nav_section = ""
        user_prompt = (
            f"### TARGET REGIME\n{regime_spec.regime_tag}: {regime_spec.description}\n\n"
            f"{track_section}"
            f"{nav_section}"
            f"### SELECTED CAUSAL RULE\n"
            f"Rule ID: {rule.rule_id}\n"
            f"Precondition: {rule.precondition}\n"
            f"Transform Template:\n{rule.transform_template.strip()}\n"
            f"Expected Postcondition: {rule.expected_postcondition}\n\n"
            f"### SURROGATE ANCHOR TELEMETRY\n"
            f"{_format_anchor_telemetry(best_result, regime_spec.regime_tag)}\n\n"
            f"### TARGET FUNCTION SNIPPET\n```python\n{snippet_source}\n```\n"
        )
        return GenerationRequest(
            role="exploit_rule",
            system_prompt=sys_prompt,
            user_prompt=user_prompt,
            json_schema=None,
            effort="medium",
        )

    def _build_creative_regime_request(
        self,
        *,
        snippet_source: str,
        func_name: str,
        regime_spec: RegimeTargetSpec,
        best_result: CandidateSurrogateResult,
        effort: str,
        excluded_mechanisms: Sequence[str] = (),
        focus_hint: str = "",
        prescreen_summary: str = "",
        nav_packet: HirTirNavigationPacket | None = None,
        parent_ledger: ObjectivePhysicalLedger | None = None,
        track: str = "unconstrained",
    ) -> GenerationRequest:
        slice_desc = self._slice_contract_instruction(func_name, regime_spec)
        sys_prompt = (
            "You are an Ascend 950PR (dav-3510) systems architect. Propose an architectural "
            f"optimization grounded in Roofline limits (56 AIV / 28 MixCore blocks, 3 KiB SSBUF, "
            "L0C->UB direct path, UB->L1 direct path, 64-elem UB alignment, Split-KV parallelism). "
            f"Preserve exact numerical semantics, signatures, and {slice_desc}. Return JSON "
            "matching the schema."
        )
        if excluded_mechanisms:
            constraint_lines = "\n".join(f"- {item}" for item in excluded_mechanisms)
        else:
            constraint_lines = "- None recorded yet; explore any high-impact structural axis."
        neg_section = (
            "### NEGATIVE CONSTRAINTS (ALREADY EXPLORED BY PARALLEL WORKERS)\n"
            f"{constraint_lines}\n"
            "Propose a structurally orthogonal transformation that does NOT duplicate the "
            "mechanisms listed above.\n\n"
        )
        if not self._enable_track_router or track in ("", "unconstrained"):
            track_section = (
                f"### TILELANG ASCEND PRIMITIVE REFERENCE\n{TILELANG_ASCEND_PRIMITIVE_MENU}\n\n"
            )
        else:
            track_section = f"### EXECUTION TRACK\n{track}\n\n"
        if nav_packet is not None:
            nav_text = format_hir_tir_navigation_section(
                nav_packet,
                current_tl_ledger=parent_ledger,
            )
            nav_section = f"{nav_text}\n\n"
        else:
            nav_section = ""
        focus_section = f"### ARCHITECTURAL FOCUS HINT\n{focus_hint}\n\n" if focus_hint else ""
        prescreen_section = (
            f"### TILE PRESCREEN GUIDANCE\n{prescreen_summary}\n\n"
            if prescreen_summary
            else ""
        )
        user_prompt = (
            f"### TARGET REGIME\n{regime_spec.regime_tag}: {regime_spec.description}\n\n"
            f"{track_section}"
            f"{nav_section}"
            f"{focus_section}"
            f"{neg_section}"
            f"{prescreen_section}"
            f"### SURROGATE ANCHOR TELEMETRY\n"
            f"{_format_anchor_telemetry(best_result, regime_spec.regime_tag)}\n\n"
            f"### TARGET FUNCTION SNIPPET\n```python\n{snippet_source}\n```\n"
        )
        return GenerationRequest(
            role="explore_creative",
            system_prompt=sys_prompt,
            user_prompt=user_prompt,
            json_schema=CREATIVE_RESPONSE_SCHEMA,
            effort=effort,
        )

    def _evaluate_single_baseline(
        self,
        incumbent_source: str,
        active_anchors: Sequence[AnchorCaseSpec],
    ) -> CandidateSurrogateResult:
        with tempfile.TemporaryDirectory(prefix="tf_camp_base_") as tmp_dir:
            mod_path = Path(tmp_dir) / "incumbent.py"
            mod_path.write_text(incumbent_source, encoding="utf-8")
            results = self._run_l2_batch([("incumbent_baseline", mod_path)], active_anchors)
        if "incumbent_baseline" not in results:
            raise RuntimeError("Baseline evaluation missing incumbent_baseline result.")
        return results["incumbent_baseline"]

    def _save_checkpoint(
        self,
        *,
        incumbent_source: str,
        baseline_result: CandidateSurrogateResult,
        best_result: CandidateSurrogateResult,
        active_anchors: Sequence[AnchorCaseSpec],
        frozen_regimes: set[str],
        current_regime: str | None,
        stagnation_rounds: int,
        rounds: Sequence[CampaignRoundRecord],
        promoted_rounds: Sequence[int],
        rolled_back_rounds: Sequence[int],
        escalated_rounds: Sequence[int],
        admitted_rule_ids: Sequence[str],
        stop_reason: str,
        l4_result: L4CannbenchResult | None,
    ) -> None:
        if self._checkpoint_path is None:
            return
        payload = {
            "version": 1,
            "operator_name": self._config.operator_name,
            "target_module_path": str(self._target_module_path),
            "incumbent_source": incumbent_source,
            "baseline_result": baseline_result.to_dict(),
            "best_result": best_result.to_dict(),
            "active_anchors": [a.to_dict() for a in active_anchors],
            "frozen_regimes": sorted(frozen_regimes),
            "current_regime": current_regime,
            "stagnation_rounds": stagnation_rounds,
            "rounds": [r.to_dict() for r in rounds],
            "promoted_rounds": list(promoted_rounds),
            "rolled_back_rounds": list(rolled_back_rounds),
            "escalated_rounds": list(escalated_rounds),
            "admitted_rule_ids": list(admitted_rule_ids),
            "tracker_state": self._tracker.to_dict(),
            "beam_state": self._beam.to_dict(),
            "stop_reason": stop_reason,
            "l4_result": l4_result.to_dict() if l4_result else None,
        }
        _atomic_write_text(
            self._checkpoint_path,
            json.dumps(payload, indent=2) + "\n",
        )

    def _restore_from_checkpoint(
        self,
    ) -> tuple[
        str,
        CandidateSurrogateResult,
        CandidateSurrogateResult,
        list[AnchorCaseSpec],
        set[str],
        str | None,
        int,
        list[CampaignRoundRecord],
        list[int],
        list[int],
        list[int],
        list[str],
    ]:
        assert self._checkpoint_path is not None
        payload = json.loads(self._checkpoint_path.read_text(encoding="utf-8"))
        incumbent_source = str(payload["incumbent_source"])
        _atomic_write_text(self._target_module_path, incumbent_source)
        if self._campaign_kb_path is not None and self._campaign_kb_path.is_file():
            for rule in load_seed_rules(self._campaign_kb_path):
                self._tracker.admit_rule(rule)
        if isinstance(payload.get("tracker_state"), Mapping):
            saved_posteriors = payload["tracker_state"].get("posteriors", {})
            if isinstance(saved_posteriors, Mapping):
                tmp_path = self._checkpoint_path.with_suffix(".tracker.tmp")
                tmp_path.write_text(
                    json.dumps({"posteriors": saved_posteriors}),
                    encoding="utf-8",
                )
                try:
                    self._tracker.load_posteriors_json(tmp_path)
                finally:
                    tmp_path.unlink(missing_ok=True)
        if isinstance(payload.get("beam_state"), Mapping):
            self._beam = TwoLaneBeamSearch.from_dict(payload["beam_state"])
        return (
            incumbent_source,
            CandidateSurrogateResult.from_dict(payload["baseline_result"]),
            CandidateSurrogateResult.from_dict(payload["best_result"]),
            [AnchorCaseSpec.from_dict(item) for item in payload.get("active_anchors", ())],
            set(str(x) for x in payload.get("frozen_regimes", ())),
            payload.get("current_regime"),
            int(payload.get("stagnation_rounds", 0)),
            [CampaignRoundRecord.from_dict(r) for r in payload.get("rounds", ())],
            [int(x) for x in payload.get("promoted_rounds", ())],
            [int(x) for x in payload.get("rolled_back_rounds", ())],
            [int(x) for x in payload.get("escalated_rounds", ())],
            [str(x) for x in payload.get("admitted_rule_ids", ())],
        )

    async def _execute_single_round(
        self,
        *,
        round_idx: int,
        regime_spec: RegimeTargetSpec,
        incumbent_source: str,
        best_result: CandidateSurrogateResult,
        active_anchors: Sequence[AnchorCaseSpec],
        stagnation_rounds: int,
        admitted_rule_ids: list[str],
    ) -> tuple[
        CampaignRoundRecord,
        str,
        CandidateSurrogateResult,
        int,
        bool,
        bool,
        bool,
    ]:
        backend = self._get_backend()
        func_name, default_snippet = extract_regime_snippet(incumbent_source, regime_spec)

        escalated = stagnation_rounds >= self._config.stagnation_escalate_rounds
        parent_lane = "exploit"
        parent_snippet = default_snippet
        stone_ledger: ObjectivePhysicalLedger | None = None
        stone_track: str | None = None
        creative_effort = "high"
        if escalated:
            creative_effort = "xhigh"
            if self._beam.stepping_stone_beam:
                stone = self._beam.stepping_stone_beam[0]
                self._beam.mark_parent_selected(stone)
                try:
                    patch_regime_into_module(incumbent_source, regime_spec, stone.dsl_source)
                    parent_snippet = stone.dsl_source
                    parent_lane = "stepping_stone"
                    stone_ledger = stone.ledger
                    stone_track = stone.track
                except Exception:
                    parent_lane = "stepping_stone"
            else:
                parent_lane = "stepping_stone"

        base_stage_a = self._run_l0(incumbent_source, regime_spec)
        base_ledger = base_stage_a.ledger if base_stage_a.ledger is not None else _fallback_ledger()
        parent_ledger = stone_ledger if stone_ledger is not None else base_ledger
        if self._enable_track_router:
            parent_track = (
                stone_track
                if stone_track is not None
                else (
                    base_stage_a.route.track
                    if base_stage_a.route is not None
                    else "unconstrained"
                )
            )
            parent_features = base_stage_a.features
        else:
            parent_track = "unconstrained"
            parent_features = None

        parent_residual = compute_static_dynamic_residual(
            parent_ledger,
            {
                "dur_us": best_result.mean_latency_us(),
                "roofline_eta": best_result.avg_roofline_eta,
            },
        )
        ctx = build_predicate_context(
            parent_ledger,
            features=parent_features,
            residual=parent_residual,
        )
        ranked_rules = list(
            self._tracker.rank_applicable_rules(ctx, top_k=self._config.k_rule_workers)
        )
        nav_packet = self._regime_nav_packets.get(regime_spec.regime_tag)

        worker_specs: list[tuple[str, str, AscendCausalRule | None, GenerationRequest]] = []
        for idx, rule in enumerate(ranked_rules[: self._config.k_rule_workers]):
            worker_specs.append(
                (
                    f"r{round_idx}_rule_{idx}",
                    "rule_exploit",
                    rule,
                    self._build_exploit_regime_request(
                        snippet_source=parent_snippet,
                        func_name=func_name,
                        regime_spec=regime_spec,
                        rule=rule,
                        best_result=best_result,
                        nav_packet=nav_packet,
                        parent_ledger=parent_ledger,
                        track=parent_track,
                    ),
                )
            )
        excluded = summarize_excluded_mechanisms(
            ranked_rules,
            self._beam.exploited_signatures,
        )
        prescreen_summary = build_prescreen_summary_from_kwargs(
            regime_spec.sample_builder_kwargs
        )
        deficit = max(0, self._config.k_rule_workers - len(ranked_rules))
        total_creative = 1 + deficit
        for c_idx in range(total_creative):
            wid = f"r{round_idx}_creative" if c_idx == 0 else f"r{round_idx}_creative_{c_idx}"
            hint = (
                CREATIVE_FOCUS_HINTS[c_idx % len(CREATIVE_FOCUS_HINTS)]
                if total_creative > 1
                else ""
            )
            worker_specs.append(
                (
                    wid,
                    "creative_explore",
                    None,
                    self._build_creative_regime_request(
                        snippet_source=parent_snippet,
                        func_name=func_name,
                        regime_spec=regime_spec,
                        best_result=best_result,
                        effort=creative_effort,
                        excluded_mechanisms=excluded,
                        focus_hint=hint,
                        prescreen_summary=prescreen_summary,
                        nav_packet=nav_packet,
                        parent_ledger=parent_ledger,
                        track=parent_track,
                    ),
                )
            )

        async def _call_worker(
            wid: str,
            wkind: str,
            rule: AscendCausalRule | None,
            req: GenerationRequest,
        ) -> tuple[str, str, AscendCausalRule | None, str | None, str, bool]:
            try:
                raw = await backend.generate(req)
            except Exception as exc:
                return wid, wkind, rule, None, f"llm_transport:{type(exc).__name__}:{exc}", True
            try:
                if wkind == "creative_explore":
                    snippet = extract_creative_payload(raw)["dsl_source"]
                else:
                    snippet = extract_python_dsl(raw)
                return wid, wkind, rule, snippet, "", False
            except Exception as exc:
                return wid, wkind, rule, None, f"llm_parse:{type(exc).__name__}:{exc}", False

        worker_replies = await asyncio.gather(
            *(_call_worker(wid, wkind, rule, req) for wid, wkind, rule, req in worker_specs)
        )

        raw_candidates: list[tuple[str, str]] = []
        failed_entries: list[tuple[str, str, str]] = []
        meta_by_id: dict[str, tuple[str, AscendCausalRule | None]] = {}
        for wid, wkind, rule, snippet, err, is_transport_error in worker_replies:
            meta_by_id[wid] = (wkind, rule)
            if snippet is None:
                stage_tag = "llm_transport" if is_transport_error else "llm_generate"
                failed_entries.append((wid, stage_tag, err[:200]))
                if rule is not None and not is_transport_error:
                    self._tracker.record_outcome(rule.rule_id, postcondition_realized=False)
            else:
                raw_candidates.append((wid, snippet))

        l0_l1_checks: list[L0L1CandidateCheck] = []
        if raw_candidates:
            l0_l1_checks = await run_l0_l1_round_filter(
                raw_candidates,
                patch_fn=lambda snip: patch_regime_into_module(
                    incumbent_source, regime_spec, snip
                ),
                l0_checker=lambda mod_src: self._run_l0(mod_src, regime_spec),
                l1_compiler=lambda mod_src: self._run_l1(mod_src, regime_spec),
                repair_backend=backend if self._config.max_syntax_repairs > 0 else None,
                max_syntax_repairs=self._config.max_syntax_repairs,
            )

        passed_checks = [chk for chk in l0_l1_checks if chk.passed]
        for chk in l0_l1_checks:
            if not chk.passed:
                failed_entries.append(
                    (chk.candidate_id, chk.failed_tier or "l0_l1", chk.error_message[:200])
                )
                _wkind, rule = meta_by_id[chk.candidate_id]
                if rule is not None:
                    self._tracker.record_outcome(rule.rule_id, postcondition_realized=False)

        l2_results: dict[str, CandidateSurrogateResult] = {}
        if passed_checks:
            with tempfile.TemporaryDirectory(prefix=f"tf_camp_r{round_idx}_") as tmp_dir:
                cand_paths: list[tuple[str, Path]] = []
                for chk in passed_checks:
                    cand_file = Path(tmp_dir) / f"{chk.candidate_id}.py"
                    cand_file.write_text(chk.patched_module_source, encoding="utf-8")
                    cand_paths.append((chk.candidate_id, cand_file))
                l2_results = self._run_l2_batch(cand_paths, active_anchors)

        beam_updates: list[BeamCandidate] = []
        passing_candidates: list[tuple[L0L1CandidateCheck, CandidateSurrogateResult]] = []
        for chk in passed_checks:
            res = l2_results.get(chk.candidate_id)
            if res is None:
                continue
            wkind, rule = meta_by_id[chk.candidate_id]
            ledger = (
                chk.stage_a.ledger
                if chk.stage_a and chk.stage_a.ledger is not None
                else _fallback_ledger()
            )
            track = (
                chk.stage_a.route.track
                if self._enable_track_router and chk.stage_a and chk.stage_a.route is not None
                else "unconstrained"
            )
            lat_us = res.mean_latency_us()
            beam_updates.append(
                BeamCandidate(
                    candidate_id=chk.candidate_id,
                    round_idx=round_idx,
                    worker_kind=wkind,
                    rule_id=rule.rule_id if rule else None,
                    dsl_source=chk.snippet_source,
                    track=track,
                    ledger=ledger,
                    residual=compute_static_dynamic_residual(
                        ledger,
                        {"dur_us": lat_us, "roofline_eta": res.avg_roofline_eta},
                    ),
                    latency_us=lat_us,
                    accuracy_passed=res.all_passed,
                    structural_signature=extract_structural_signature(
                        chk.snippet_source, track=track
                    ),
                    parent_id=f"round_{round_idx - 1}",
                    features=(
                        chk.stage_a.features
                        if (self._enable_track_router and chk.stage_a)
                        else None
                    ),
                )
            )
            if rule is not None:
                surrogate_gain = (
                    (res.surrogate_score - best_result.surrogate_score)
                    if res.all_passed
                    else 0.0
                )
                realized = res.all_passed and _surrogate_rank_key(res) > _surrogate_rank_key(
                    best_result
                )
                self._tracker.record_outcome(
                    rule.rule_id,
                    postcondition_realized=realized,
                    speed_gain=max(0.0, surrogate_gain),
                )
            if res.all_passed:
                passing_candidates.append((chk, res))

        self._beam.update(beam_updates)

        best_round_speedup = best_result.avg_speedup
        best_round_surrogate_score = best_result.surrogate_score
        promoted = False
        rolled_back_l3 = False
        new_incumbent = incumbent_source
        new_best = best_result
        round_admitted: list[str] = []

        if passing_candidates:
            top_chk, top_res = max(
                passing_candidates,
                key=lambda item: _surrogate_rank_key(item[1]),
            )
            best_round_speedup = top_res.avg_speedup
            best_round_surrogate_score = top_res.surrogate_score
            surrogate_gain = top_res.surrogate_score - best_result.surrogate_score
            rel_gain = (
                (top_res.avg_speedup - best_result.avg_speedup)
                / max(best_result.avg_speedup, 1e-6)
            )
            tail_guard_ok = top_res.min_case_speedup >= 0.97 * best_result.min_case_speedup
            dual_criterion_ok = surrogate_gain >= self._config.min_surrogate_gain or (
                top_res.min_case_speedup >= 1.03 * best_result.min_case_speedup
                and top_res.surrogate_score >= best_result.surrogate_score - 1e-6
            )
            if tail_guard_ok and dual_criterion_ok:
                _atomic_write_text(self._target_module_path, top_chk.patched_module_source)
                l3_res = self._run_l3_contract(self._target_module_path)
                if l3_res.passed:
                    promoted = True
                    new_incumbent = top_chk.patched_module_source
                    new_best = top_res
                    stagnation_after = 0
                    if rel_gain >= self._config.min_flywheel_gain and self._flywheel is not None:
                        try:
                            distill_rep = self._flywheel.distill_from_pair(
                                baseline_dsl=incumbent_source,
                                winner_dsl=top_chk.patched_module_source,
                                baseline_latency_us=best_result.mean_latency_us(),
                                winner_latency_us=top_res.mean_latency_us(),
                                factory_name=func_name,
                                factory_kwargs=regime_spec.sample_builder_kwargs,
                                kb_output_path=self._campaign_kb_path,
                                rule_id_prefix=f"R_{self._config.operator_name.upper()}",
                            )
                            round_admitted.extend(distill_rep.admitted_rule_ids)
                            admitted_rule_ids.extend(distill_rep.admitted_rule_ids)
                        except Exception:
                            pass
                else:
                    _atomic_write_text(self._target_module_path, incumbent_source)
                    rolled_back_l3 = True
                    stagnation_after = stagnation_rounds + 1
            else:
                stagnation_after = stagnation_rounds + 1
        else:
            stagnation_after = stagnation_rounds + 1

        record = CampaignRoundRecord(
            round_idx=round_idx,
            regime_tag=regime_spec.regime_tag,
            builder_name=func_name,
            parent_lane=parent_lane,
            creative_effort=creative_effort,
            l0_l1_passed_ids=tuple(c.candidate_id for c in passed_checks),
            l0_l1_failed=tuple(failed_entries),
            best_round_speedup=best_round_speedup,
            best_so_far_speedup=new_best.avg_speedup,
            promoted=promoted,
            rolled_back_l3=rolled_back_l3,
            stagnation_rounds_after=stagnation_after,
            admitted_rule_ids=tuple(round_admitted),
            best_round_surrogate_score=best_round_surrogate_score,
            best_so_far_surrogate_score=new_best.surrogate_score,
        )
        return (
            record,
            new_incumbent,
            new_best,
            stagnation_after,
            promoted,
            rolled_back_l3,
            escalated,
        )

    async def run(self, *, resume: bool = False) -> CampaignReport:
        """Execute unattended optimization campaign until stopping criterion triggers."""
        campaign_start = self._clock()
        if resume and self._checkpoint_path is not None and self._checkpoint_path.is_file():
            (
                incumbent_source,
                baseline_result,
                best_result,
                active_anchors,
                frozen_regimes,
                current_regime,
                stagnation_rounds,
                rounds,
                promoted_rounds,
                rolled_back_rounds,
                escalated_rounds,
                admitted_rule_ids,
            ) = self._restore_from_checkpoint()
        else:
            incumbent_source = self._target_module_path.read_text(encoding="utf-8")
            active_anchors = list(self._initial_anchors)
            baseline_result = self._evaluate_single_baseline(incumbent_source, active_anchors)
            best_result = baseline_result
            frozen_regimes = set()
            current_regime = None
            stagnation_rounds = 0
            rounds = []
            promoted_rounds = []
            rolled_back_rounds = []
            escalated_rounds = []
            admitted_rule_ids = []
            self._save_checkpoint(
                incumbent_source=incumbent_source,
                baseline_result=baseline_result,
                best_result=best_result,
                active_anchors=active_anchors,
                frozen_regimes=frozen_regimes,
                current_regime=current_regime,
                stagnation_rounds=stagnation_rounds,
                rounds=rounds,
                promoted_rounds=promoted_rounds,
                rolled_back_rounds=rolled_back_rounds,
                escalated_rounds=escalated_rounds,
                admitted_rule_ids=admitted_rule_ids,
                stop_reason="in_progress",
                l4_result=None,
            )

        stop_reason = ""
        if self._allow_early_stop(len(rounds)):
            if best_result.meets_perf_target(self._perf_target):
                stop_reason = "goal_met"
            elif best_result.is_roofline_saturated(self._perf_target):
                stop_reason = "roofline_saturated"

        start_round = (rounds[-1].round_idx + 1) if rounds else 1
        budget_sec = self._config.max_wall_clock_min * 60.0

        for round_idx in range(start_round, max(0, self._config.max_rounds) + 1):
            if stop_reason:
                break
            if (self._clock() - campaign_start) >= budget_sec:
                stop_reason = "budget_exhausted"
                break

            if stagnation_rounds >= self._config.stagnation_freeze_rounds and current_regime:
                frozen_regimes.add(current_regime)
                stagnation_rounds = 0
                current_regime = None

            active_regime = self._select_active_regime(
                current_regime,
                frozen_regimes,
                best_result,
                active_anchors,
                allow_satisfied_fallback=not self._allow_early_stop(len(rounds)),
            )
            if active_regime is None:
                stop_reason = "all_regimes_converged"
                break
            current_regime = active_regime.regime_tag

            (
                rec,
                incumbent_source,
                best_result,
                stagnation_rounds,
                promoted,
                rolled_back,
                escalated,
            ) = await self._execute_single_round(
                round_idx=round_idx,
                regime_spec=active_regime,
                incumbent_source=incumbent_source,
                best_result=best_result,
                active_anchors=active_anchors,
                stagnation_rounds=stagnation_rounds,
                admitted_rule_ids=admitted_rule_ids,
            )
            rounds.append(rec)
            if promoted:
                promoted_rounds.append(round_idx)
            if rolled_back:
                rolled_back_rounds.append(round_idx)
            if escalated:
                escalated_rounds.append(round_idx)

            early_stop_allowed = self._allow_early_stop(len(rounds))
            if early_stop_allowed and best_result.meets_perf_target(self._perf_target):
                stop_reason = "goal_met"
            elif early_stop_allowed and best_result.is_roofline_saturated(self._perf_target):
                stop_reason = "roofline_saturated"
            elif stagnation_rounds >= self._config.stagnation_freeze_rounds:
                frozen_regimes.add(active_regime.regime_tag)
                stagnation_rounds = 0
                current_regime = None
                next_reg = self._select_active_regime(
                    None,
                    frozen_regimes,
                    best_result,
                    active_anchors,
                    allow_satisfied_fallback=not early_stop_allowed,
                )
                if next_reg is None:
                    stop_reason = "all_regimes_converged"

            self._save_checkpoint(
                incumbent_source=incumbent_source,
                baseline_result=baseline_result,
                best_result=best_result,
                active_anchors=active_anchors,
                frozen_regimes=frozen_regimes,
                current_regime=current_regime,
                stagnation_rounds=stagnation_rounds,
                rounds=rounds,
                promoted_rounds=promoted_rounds,
                rolled_back_rounds=rolled_back_rounds,
                escalated_rounds=escalated_rounds,
                admitted_rule_ids=admitted_rule_ids,
                stop_reason=stop_reason or "in_progress",
                l4_result=None,
            )
            if stop_reason:
                break

        if not stop_reason:
            stop_reason = "budget_exhausted"

        l4_result: L4CannbenchResult | None = None
        if self._config.run_final_l4:
            l4_result = self._run_l4_cannbench(self._target_module_path)
            anchor_nums = {a.case_num for a in active_anchors}
            regressed_nums = sorted(
                cnum
                for cnum, sp in l4_result.per_case_speedups.items()
                if cnum not in anchor_nums and sp < self._config.l4_regression_ratio
            )
            if regressed_nums:
                for cnum in regressed_nums:
                    active_anchors = list(
                        append_dynamic_anchor(
                            active_anchors,
                            operator_name=self._config.operator_name,
                            case_num=cnum,
                            regime_tag="tail_regression",
                            cannbench_repo=self._cannbench_repo,
                            env_file=self._env_file,
                        )
                    )
                catch_regime = self._regime_specs[0]
                catch_idx = (rounds[-1].round_idx + 1) if rounds else 1
                (
                    rec,
                    incumbent_source,
                    best_result,
                    stagnation_rounds,
                    promoted,
                    rolled_back,
                    escalated,
                ) = await self._execute_single_round(
                    round_idx=catch_idx,
                    regime_spec=catch_regime,
                    incumbent_source=incumbent_source,
                    best_result=best_result,
                    active_anchors=active_anchors,
                    stagnation_rounds=stagnation_rounds,
                    admitted_rule_ids=admitted_rule_ids,
                )
                rounds.append(rec)
                if promoted:
                    promoted_rounds.append(catch_idx)
                if rolled_back:
                    rolled_back_rounds.append(catch_idx)
                if escalated:
                    escalated_rounds.append(catch_idx)
                l4_result = self._run_l4_cannbench(self._target_module_path)

        self._save_checkpoint(
            incumbent_source=incumbent_source,
            baseline_result=baseline_result,
            best_result=best_result,
            active_anchors=active_anchors,
            frozen_regimes=frozen_regimes,
            current_regime=current_regime,
            stagnation_rounds=stagnation_rounds,
            rounds=rounds,
            promoted_rounds=promoted_rounds,
            rolled_back_rounds=rolled_back_rounds,
            escalated_rounds=escalated_rounds,
            admitted_rule_ids=admitted_rule_ids,
            stop_reason=stop_reason,
            l4_result=l4_result,
        )

        return CampaignReport(
            operator_name=self._config.operator_name,
            stop_reason=stop_reason,
            total_rounds=len(rounds),
            baseline_result=baseline_result,
            best_result=best_result,
            rounds=tuple(rounds),
            promoted_rounds=tuple(promoted_rounds),
            rolled_back_rounds=tuple(rolled_back_rounds),
            escalated_rounds=tuple(escalated_rounds),
            frozen_regimes=tuple(sorted(frozen_regimes)),
            active_anchors=tuple(active_anchors),
            admitted_rule_ids=tuple(admitted_rule_ids),
            l4_result=l4_result,
        )


def build_arg_parser() -> argparse.ArgumentParser:
    """Construct CLI parser for PerfTargetCampaignRunner."""
    parser = argparse.ArgumentParser(
        prog="tilefoundry-ascend-campaign-runner",
        description="Run unattended performance optimization campaigns.",
    )
    parser.add_argument(
        "--operator",
        required=True,
        choices=("attention_residual", "mla"),
        help="Target production operator to optimize.",
    )
    parser.add_argument("--max-rounds", type=int, default=12)
    parser.add_argument("--max-wall-clock-min", type=float, default=180.0)
    parser.add_argument("--device-id", default="1", help="Authorized NPU device ID.")
    parser.add_argument(
        "--disable-early-goal-stop",
        action="store_true",
        help="Continue optimizing through max-rounds even after target thresholds pass.",
    )
    parser.add_argument(
        "--min-surrogate-gain",
        type=float,
        default=0.25,
        help="Minimum surrogate_score gain on [0, 100] scale required for L3 promotion.",
    )
    parser.add_argument("--checkpoint", default=None, help="Atomic checkpoint JSON path.")
    parser.add_argument("--report-json", default=None, help="Optional final report JSON path.")
    parser.add_argument("--run-final-l4", action="store_true", help="Run final official L4 gate.")
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoint.")
    parser.add_argument("--k-rule-workers", type=int, default=2)
    parser.add_argument("--source-root", default=None, help="TileFoundry workspace root.")
    parser.add_argument("--env-file", default=None, help="Path to tilefoundry-npu env.sh.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute PerfTargetCampaignRunner from CLI arguments and emit JSON summary."""
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    cfg = CampaignConfig(
        operator_name=args.operator,
        max_rounds=args.max_rounds,
        max_wall_clock_min=args.max_wall_clock_min,
        k_rule_workers=args.k_rule_workers,
        min_surrogate_gain=args.min_surrogate_gain,
        disable_early_goal_stop=bool(args.disable_early_goal_stop),
        run_final_l4=bool(args.run_final_l4),
        device_id=str(args.device_id),
    )
    tracker = BayesianRuleTracker.from_seed_yaml()
    flywheel = FlywheelDistiller(
        funnel=GateFunnel(skip_stage_b=True),
        tracker=tracker,
    )
    runner = PerfTargetCampaignRunner(
        config=cfg,
        tracker=tracker,
        flywheel_distiller=flywheel,
        checkpoint_path=args.checkpoint,
        source_root=args.source_root,
        env_file=args.env_file,
    )
    report = asyncio.run(runner.run(resume=bool(args.resume)))
    rendered = json.dumps(report.to_dict(), indent=2)
    if args.report_json:
        _atomic_write_text(Path(args.report_json), rendered + "\n")
    print(rendered)
    return 0


__all__ = [
    "CampaignConfig",
    "CampaignReport",
    "CampaignRoundRecord",
    "PerfTargetCampaignRunner",
    "ResolvedRegimeConfig",
    "build_arg_parser",
    "extract_regime_snippet",
    "main",
    "patch_regime_into_module",
    "resolve_regime_config",
]


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
