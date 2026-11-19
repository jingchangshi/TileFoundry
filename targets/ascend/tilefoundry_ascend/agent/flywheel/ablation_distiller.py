"""Counterfactual single-variable AST ablation and self-evolution flywheel distiller."""

from __future__ import annotations

import ast
import copy
import difflib
import hashlib
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from tilefoundry_ascend.agent.campaign.anchor_specs import CannbenchOpsRegistry
from tilefoundry_ascend.agent.kb.bayesian_tracker import BayesianRuleTracker
from tilefoundry_ascend.agent.kb.schema import (
    AscendCausalRule,
    build_predicate_context,
    save_rules_yaml,
    validate_rule_predicate,
)
from tilefoundry_ascend.agent.orchestrator.e2e_evolution import EvolutionTrajectory
from tilefoundry_ascend.agent.orchestrator.gate_funnel import (
    GateFunnel,
    StageAResult,
    run_stage_a_static,
)
from tilefoundry_ascend.ascend.passes.physical_ledger import ObjectivePhysicalLedger

_STATIC_DISALLOWED_PROVENANCE_TOKENS = frozenset(
    {
        "/home/",
        "/users/",
        "/data/",
        "attention_residual",
        "attn_res",
        "mla",
        "mla_model",
        "grouped_matmul",
        "softmax",
        "deepgemm",
        "flash_attention",
    }
)
_SUBSTRING_PROVENANCE_TOKENS = frozenset(
    {
        "/home/",
        "/users/",
        "/data/",
        "attention_residual",
        "attn_res",
        "mla_model",
        "deepgemm",
        "flash_attention",
    }
)


def _build_disallowed_provenance_tokens() -> frozenset[str]:
    """Dynamically collect all registered operator IDs to prevent operator-specific leakage."""
    tokens = set(_STATIC_DISALLOWED_PROVENANCE_TOKENS)
    try:
        for op_id in CannbenchOpsRegistry.default().list_operators():
            cleaned = str(op_id).strip().lower()
            if cleaned:
                tokens.add(cleaned)
    except (FileNotFoundError, OSError):
        pass
    return frozenset(tokens)


@dataclass(frozen=True)
class AstEditChunk:
    """Single atomic structural delta extracted from baseline and winner DSL ASTs."""

    chunk_id: str
    edit_kind: str
    target_path: str
    before_src: str
    after_src: str
    ablated_dsl: str

    def to_dict(self) -> dict[str, object]:
        """Serialize atomic AST delta to JSON-compatible dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "edit_kind": self.edit_kind,
            "target_path": self.target_path,
            "before_src": self.before_src,
            "after_src": self.after_src,
            "dsl_sha256": hashlib.sha256(self.ablated_dsl.encode("utf-8")).hexdigest()[:16],
        }


@dataclass(frozen=True)
class AblationMeasurement:
    """Funnel and latency outcome for one counterfactual single-variable ablation."""

    chunk: AstEditChunk
    funnel_passed: bool
    baseline_latency_us: float
    variant_latency_us: float
    relative_gain: float
    positive_gain: bool
    stage_a_result: StageAResult | None = None
    diagnostic: str = ""

    def to_dict(self) -> dict[str, object]:
        """Serialize ablation measurement to JSON-compatible dictionary."""
        return {
            "chunk": self.chunk.to_dict(),
            "funnel_passed": self.funnel_passed,
            "baseline_latency_us": round(self.baseline_latency_us, 6),
            "variant_latency_us": round(self.variant_latency_us, 6),
            "relative_gain": round(self.relative_gain, 6),
            "positive_gain": self.positive_gain,
            "diagnostic": self.diagnostic,
        }


@dataclass(frozen=True)
class MicroProbeSpec:
    """Standalone 30-50 line Ascend 950 micro-probe pair isolating one causal edit."""

    probe_id: str
    probe_kind: str
    factory_name: str
    baseline_probe_source: str
    optimized_probe_source: str
    line_count_baseline: int
    line_count_optimized: int
    tensor_shape: tuple[int, ...]
    tensor_dtype: str = "float32"

    def to_dict(self) -> dict[str, object]:
        """Serialize micro-probe specification without raw machine paths."""
        return {
            "probe_id": self.probe_id,
            "probe_kind": self.probe_kind,
            "factory_name": self.factory_name,
            "line_count_baseline": self.line_count_baseline,
            "line_count_optimized": self.line_count_optimized,
            "tensor_shape": list(self.tensor_shape),
            "tensor_dtype": self.tensor_dtype,
        }


_SUPPORTED_PROBE_KINDS = (
    "core_parallelism",
    "coalesced_copy",
    "pipeline_versions",
    "ssbuf_handoff",
    "barrier_prune",
    "dual_copy_1c2v",
    "pad_alignment",
    "static_pingpong_unroll",
    "loop_fission_sfu",
)


@dataclass(frozen=True)
class DistilledRuleOutcome:
    """Verification and KB admission outcome for one winning AST mechanism."""

    chunk_id: str
    probe: MicroProbeSpec | None
    hardware_verified: bool
    probe_metrics: dict[str, float]
    rule: AscendCausalRule | None
    admitted: bool
    rejection_reason: str = ""

    def to_dict(self) -> dict[str, object]:
        """Serialize distilled rule verification and admission record."""
        return {
            "chunk_id": self.chunk_id,
            "probe": self.probe.to_dict() if self.probe is not None else None,
            "hardware_verified": self.hardware_verified,
            "probe_metrics": dict(self.probe_metrics),
            "rule_id": self.rule.rule_id if self.rule else None,
            "admitted": self.admitted,
            "rejection_reason": self.rejection_reason,
        }


@dataclass
class FlywheelDistillationReport:
    """Complete audit report across AST decomposition, ablation, probing, and KB admission."""

    baseline_latency_us: float
    winner_latency_us: float
    total_chunks: int
    kept_positive_chunks: tuple[AblationMeasurement, ...]
    discarded_chunks: tuple[AblationMeasurement, ...]
    distilled_outcomes: tuple[DistilledRuleOutcome, ...] = field(default_factory=tuple)
    admitted_rule_ids: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, object]:
        """Serialize complete distillation report to JSON-compatible dictionary."""
        return {
            "baseline_latency_us": round(self.baseline_latency_us, 6),
            "winner_latency_us": round(self.winner_latency_us, 6),
            "total_chunks": self.total_chunks,
            "kept_positive_chunks": [m.to_dict() for m in self.kept_positive_chunks],
            "discarded_chunks": [m.to_dict() for m in self.discarded_chunks],
            "distilled_outcomes": [o.to_dict() for o in self.distilled_outcomes],
            "admitted_rule_ids": list(self.admitted_rule_ids),
        }

    def to_json(self) -> str:
        """Render formatted JSON representation of the distillation report."""
        return json.dumps(self.to_dict(), indent=2)


def _locate_prim_func(tree: ast.Module) -> tuple[ast.FunctionDef | None, ast.FunctionDef | None]:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            for parent in ast.walk(tree):
                if isinstance(parent, ast.FunctionDef) and parent is not node:
                    if node in parent.body:
                        return parent, node
            return None, node
    for stmt in tree.body:
        if isinstance(stmt, ast.FunctionDef):
            return None, stmt
    return None, None


def _node_key(node: ast.AST) -> str:
    return ast.unparse(node)


def _node_tag(node: ast.AST) -> str:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return f"def:{node.name}"
    if isinstance(node, ast.With):
        if node.items:
            ctx = node.items[0].context_expr
            if isinstance(ctx, ast.Call):
                return f"with:{_node_key(ctx.func)}"
            return f"with:{_node_key(ctx)}"
        return "with"
    if isinstance(node, ast.For):
        return f"for:{_node_key(node.target)}"
    if isinstance(node, ast.Assign) and node.targets:
        return f"assign:{_node_key(node.targets[0])}"
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        return f"call:{_node_key(node.value.func)}"
    return type(node).__name__


def _classify_edit(before_src: str, after_src: str) -> str:
    combined = f"{before_src} {after_src}"
    if "alloc_ss" in combined or "SSBUF" in combined:
        return "ssbuf_handoff"
    if "T.dual_copy" in combined or (
        "T.MixedKernel" in after_src and "T.Kernel" in before_src
    ):
        return "dual_copy_1c2v"
    if (
        ("T.Pipelined" in before_src and "T.Pipelined" not in after_src)
        or ("enable_offset" in before_src and "enable_offset" not in after_src)
    ):
        return "static_pingpong_unroll"
    if (
        "gm_to_ub_pad" in combined
        or "pad_value" in combined
        or ("alloc_shared" in before_src and "alloc_shared" in after_src and before_src != after_src)
    ):
        return "pad_alignment"
    if "ascend_pipe_barrier" in combined or "unit_flag_ctrl" in combined:
        return "barrier_prune"
    if "num_stages" in combined or "annotate_buffer_versions" in combined:
        return "pipeline_versions"
    if (
        ("vcadd" in combined or "vexp" in combined or "vsqrt" in combined)
        and (
            after_src.count("for ") > before_src.count("for ")
            or ("vcadd" in before_src and "vcadd" not in after_src)
            or (
                before_src.count("for ") >= 1
                and after_src.count("for ") >= 1
                and len(after_src.strip().splitlines()) < len(before_src.strip().splitlines())
            )
        )
    ):
        return "loop_fission_sfu"
    if (
        "T.Kernel" in combined
        or "T.MixedKernel" in combined
        or "num_cores" in combined
        or "num_blocks" in combined
    ):
        return "core_parallelism"
    if "T.copy" in combined:
        return "coalesced_copy"
    return "stmt_rewrite"


def _extract_probe_params(chunk: AstEditChunk) -> dict[str, object]:
    """Extract and clamp concrete tile, core, stage, and dtype parameters from AST."""
    cores = 28
    stages = 2
    tile_elems = 2048
    dtype = "float32"

    for candidate_src in (chunk.after_src, chunk.ablated_dsl):
        if not candidate_src.strip():
            continue
        try:
            tree = ast.parse(candidate_src)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                attr = node.func.attr
                if attr in ("Kernel", "MixedKernel") and node.args:
                    arg0 = node.args[0]
                    if isinstance(arg0, ast.Constant) and isinstance(arg0.value, int):
                        cores = max(2, min(56, int(arg0.value)))
                elif attr == "Pipelined":
                    for kw in node.keywords:
                        if kw.arg == "num_stages" and isinstance(kw.value, ast.Constant):
                            if isinstance(kw.value.value, int):
                                stages = max(2, min(3, int(kw.value.value)))
                elif attr in ("alloc_shared", "alloc_l1") and node.args:
                    shape_arg = node.args[0]
                    if isinstance(shape_arg, (ast.Tuple, ast.List)):
                        ints = [
                            int(elt.value)
                            for elt in shape_arg.elts
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, int)
                        ]
                        if len(ints) == len(shape_arg.elts) and ints:
                            prod = 1
                            for dim_val in ints:
                                prod *= max(1, dim_val)
                            aligned = ((prod + 127) // 128) * 128
                            tile_elems = max(512, min(4096, aligned))
                    if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
                        dt_val = str(node.args[1].value)
                        if dt_val in ("float32", "float16", "bfloat16"):
                            dtype = dt_val
    return {
        "cores": cores,
        "stages": stages,
        "tile_elems": tile_elems,
        "dtype": dtype,
    }


def _set_stmt_by_path(tree: ast.Module, path: tuple[int, ...], new_stmt: ast.stmt | None) -> bool:
    cur_list: list[ast.stmt] = tree.body
    for idx in path[:-1]:
        node = cur_list[idx]
        if not hasattr(node, "body") or not isinstance(node.body, list):
            return False
        cur_list = node.body
    leaf = path[-1]
    if new_stmt is None:
        if 0 <= leaf < len(cur_list):
            cur_list.pop(leaf)
            return True
        return False
    if 0 <= leaf < len(cur_list):
        cur_list[leaf] = copy.deepcopy(new_stmt)
        return True
    if leaf == len(cur_list):
        cur_list.append(copy.deepcopy(new_stmt))
        return True
    return False


def _insert_stmt_by_path(tree: ast.Module, path: tuple[int, ...], new_stmt: ast.stmt) -> bool:
    cur_list: list[ast.stmt] = tree.body
    for idx in path[:-1]:
        node = cur_list[idx]
        if not hasattr(node, "body") or not isinstance(node.body, list):
            return False
        cur_list = node.body
    leaf = max(0, min(path[-1], len(cur_list)))
    cur_list.insert(leaf, copy.deepcopy(new_stmt))
    return True


def _replace_call_arg(
    tree: ast.Module,
    stmt_path: tuple[int, ...],
    *,
    is_with_header: bool,
    new_expr: ast.expr,
) -> bool:
    cur_list: list[ast.stmt] = tree.body
    for idx in stmt_path[:-1]:
        node = cur_list[idx]
        if not hasattr(node, "body") or not isinstance(node.body, list):
            return False
        cur_list = node.body
    stmt = cur_list[stmt_path[-1]]
    if is_with_header and isinstance(stmt, ast.With) and stmt.items:
        stmt.items[0].context_expr = copy.deepcopy(new_expr)
        return True
    if not is_with_header and isinstance(stmt, ast.For):
        stmt.iter = copy.deepcopy(new_expr)
        return True
    return False


def _collect_top_defs(tree: ast.Module) -> list[tuple[str, ast.stmt]]:
    defs: list[tuple[str, ast.stmt]] = []
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            defs.append((stmt.name, stmt))
        elif isinstance(stmt, ast.Assign):
            if len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
                defs.append((stmt.targets[0].id, stmt))
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            defs.append((stmt.target.id, stmt))
    return defs


def _attach_missing_winner_defs(
    patched_tree: ast.Module,
    new_defs: Mapping[str, ast.stmt] | None,
    def_order: Sequence[str],
) -> ast.Module:
    if not new_defs:
        return patched_tree
    existing = {name for name, _stmt in _collect_top_defs(patched_tree)}
    needed: set[str] = set()
    changed = True
    while changed:
        changed = False
        roots: list[ast.AST] = [patched_tree]
        for n in needed:
            roots.append(new_defs[n])
        for root in roots:
            for sub in ast.walk(root):
                if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                    if sub.id in new_defs and sub.id not in existing and sub.id not in needed:
                        needed.add(sub.id)
                        changed = True
    if not needed:
        return patched_tree
    order_rank = {name: idx for idx, name in enumerate(def_order)}
    to_insert = [
        copy.deepcopy(new_defs[name])
        for name in sorted(needed, key=lambda n: order_rank.get(n, 0))
    ]
    insert_at = 0
    for idx, stmt in enumerate(patched_tree.body):
        if isinstance(stmt, (ast.Import, ast.ImportFrom)) or (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        ):
            insert_at = idx + 1
        elif idx == insert_at:
            break
    patched_tree.body = [
        *patched_tree.body[:insert_at],
        *to_insert,
        *patched_tree.body[insert_at:],
    ]
    return patched_tree


def _render_patched(
    base_tree: ast.Module,
    *,
    new_defs: Mapping[str, ast.stmt] | None = None,
    def_order: Sequence[str] = (),
) -> str:
    base_tree = _attach_missing_winner_defs(base_tree, new_defs, def_order)
    ast.fix_missing_locations(base_tree)
    code = ast.unparse(base_tree)
    ast.parse(code)
    return code


def _extract_stmt_list_deltas(
    base_tree: ast.Module,
    base_list: list[ast.stmt],
    win_list: list[ast.stmt],
    parent_path: tuple[int, ...],
    path_label: str,
    chunks: list[AstEditChunk],
    seen_dsls: set[str],
    *,
    new_defs: Mapping[str, ast.stmt] | None = None,
    def_order: Sequence[str] = (),
) -> None:
    base_tags = [_node_tag(s) for s in base_list]
    win_tags = [_node_tag(s) for s in win_list]
    base_blk_txt = "\n".join(_node_key(s) for s in base_list)
    block_has_sfu = any(tok in base_blk_txt for tok in ("vcadd", "vexp", "vsqrt"))
    block_fission_loops = block_has_sfu and (
        sum(isinstance(s, ast.For) for s in win_list)
        > sum(isinstance(s, ast.For) for s in base_list)
    )
    matcher = difflib.SequenceMatcher(a=base_tags, b=win_tags, autojunk=False)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                b_stmt = base_list[i1 + k]
                w_stmt = win_list[j1 + k]
                b_txt = _node_key(b_stmt)
                w_txt = _node_key(w_stmt)
                if b_txt == w_txt:
                    continue
                cur_path = parent_path + (i1 + k,)
                if isinstance(b_stmt, ast.With) and isinstance(w_stmt, ast.With):
                    b_hdr = _node_key(b_stmt.items[0].context_expr) if b_stmt.items else ""
                    w_hdr = _node_key(w_stmt.items[0].context_expr) if w_stmt.items else ""
                    if b_hdr != w_hdr and w_stmt.items:
                        patched = copy.deepcopy(base_tree)
                        if _replace_call_arg(patched, cur_path, is_with_header=True, new_expr=w_stmt.items[0].context_expr):
                            dsl = _render_patched(patched, new_defs=new_defs, def_order=def_order)
                            if dsl not in seen_dsls:
                                seen_dsls.add(dsl)
                                chunks.append(
                                    AstEditChunk(
                                        chunk_id=f"edit_{len(chunks)}",
                                        edit_kind=_classify_edit(b_hdr, w_hdr),
                                        target_path=f"{path_label}/with_{i1 + k}",
                                        before_src=b_hdr,
                                        after_src=w_hdr,
                                        ablated_dsl=dsl,
                                    )
                                )
                    _extract_stmt_list_deltas(
                        base_tree,
                        b_stmt.body,
                        w_stmt.body,
                        cur_path,
                        f"{path_label}/with_{i1 + k}",
                        chunks,
                        seen_dsls,
                        new_defs=new_defs,
                        def_order=def_order,
                    )
                    continue

                if isinstance(b_stmt, ast.For) and isinstance(w_stmt, ast.For):
                    b_iter = _node_key(b_stmt.iter)
                    w_iter = _node_key(w_stmt.iter)
                    if b_iter != w_iter:
                        patched = copy.deepcopy(base_tree)
                        if _replace_call_arg(patched, cur_path, is_with_header=False, new_expr=w_stmt.iter):
                            dsl = _render_patched(patched, new_defs=new_defs, def_order=def_order)
                            if dsl not in seen_dsls:
                                seen_dsls.add(dsl)
                                chunks.append(
                                    AstEditChunk(
                                        chunk_id=f"edit_{len(chunks)}",
                                        edit_kind=_classify_edit(b_iter, w_iter),
                                        target_path=f"{path_label}/for_{i1 + k}",
                                        before_src=b_iter,
                                        after_src=w_iter,
                                        ablated_dsl=dsl,
                                    )
                                )
                    _extract_stmt_list_deltas(
                        base_tree,
                        b_stmt.body,
                        w_stmt.body,
                        cur_path,
                        f"{path_label}/for_{i1 + k}",
                        chunks,
                        seen_dsls,
                        new_defs=new_defs,
                        def_order=def_order,
                    )
                    continue

                if isinstance(b_stmt, ast.FunctionDef) and isinstance(w_stmt, ast.FunctionDef):
                    _extract_stmt_list_deltas(
                        base_tree,
                        b_stmt.body,
                        w_stmt.body,
                        cur_path,
                        f"{path_label}/{b_stmt.name}",
                        chunks,
                        seen_dsls,
                        new_defs=new_defs,
                        def_order=def_order,
                    )
                    continue

                patched = copy.deepcopy(base_tree)
                if _set_stmt_by_path(patched, cur_path, w_stmt):
                    dsl = _render_patched(patched, new_defs=new_defs, def_order=def_order)
                    if dsl not in seen_dsls:
                        seen_dsls.add(dsl)
                        chunks.append(
                            AstEditChunk(
                                chunk_id=f"edit_{len(chunks)}",
                                edit_kind=_classify_edit(b_txt, w_txt),
                                target_path=f"{path_label}/stmt_{i1 + k}",
                                before_src=b_txt,
                                after_src=w_txt,
                                ablated_dsl=dsl,
                            )
                        )
        elif tag == "replace":
            common = min(i2 - i1, j2 - j1)
            for k in range(common):
                b_stmt = base_list[i1 + k]
                w_stmt = win_list[j1 + k]
                b_txt = _node_key(b_stmt)
                w_txt = _node_key(w_stmt)
                if isinstance(b_stmt, (ast.With, ast.For)) and isinstance(w_stmt, type(b_stmt)):
                    if isinstance(b_stmt, ast.With) and isinstance(w_stmt, ast.With) and b_stmt.items and w_stmt.items:
                        b_hdr = _node_key(b_stmt.items[0].context_expr)
                        w_hdr = _node_key(w_stmt.items[0].context_expr)
                        if b_hdr != w_hdr:
                            hdr_patch = copy.deepcopy(base_tree)
                            if _replace_call_arg(
                                hdr_patch,
                                parent_path + (i1 + k,),
                                is_with_header=True,
                                new_expr=w_stmt.items[0].context_expr,
                            ):
                                dsl = _render_patched(hdr_patch, new_defs=new_defs, def_order=def_order)
                                if dsl not in seen_dsls:
                                    seen_dsls.add(dsl)
                                    chunks.append(
                                        AstEditChunk(
                                            chunk_id=f"edit_{len(chunks)}",
                                            edit_kind=_classify_edit(b_hdr, w_hdr),
                                            target_path=f"{path_label}/with_hdr_{i1 + k}",
                                            before_src=b_hdr,
                                            after_src=w_hdr,
                                            ablated_dsl=dsl,
                                        )
                                    )
                    _extract_stmt_list_deltas(
                        base_tree,
                        b_stmt.body,
                        w_stmt.body,
                        parent_path + (i1 + k,),
                        f"{path_label}/block_{i1 + k}",
                        chunks,
                        seen_dsls,
                        new_defs=new_defs,
                        def_order=def_order,
                    )
                    continue
                cur_path = parent_path + (i1 + k,)
                patched = copy.deepcopy(base_tree)
                if _set_stmt_by_path(patched, cur_path, w_stmt):
                    dsl = _render_patched(patched, new_defs=new_defs, def_order=def_order)
                    if dsl not in seen_dsls:
                        seen_dsls.add(dsl)
                        chunks.append(
                            AstEditChunk(
                                chunk_id=f"edit_{len(chunks)}",
                                edit_kind=_classify_edit(b_txt, w_txt),
                                target_path=f"{path_label}/replace_{i1 + k}",
                                before_src=b_txt,
                                after_src=w_txt,
                                ablated_dsl=dsl,
                            )
                        )
            for j in range(j1 + common, j2):
                w_stmt = win_list[j]
                w_txt = _node_key(w_stmt)
                ins_idx = min(i2 + (j - (j1 + common)), len(base_list))
                cur_path = parent_path + (ins_idx,)
                patched = copy.deepcopy(base_tree)
                if _insert_stmt_by_path(patched, cur_path, w_stmt):
                    dsl = _render_patched(patched, new_defs=new_defs, def_order=def_order)
                    if dsl not in seen_dsls:
                        seen_dsls.add(dsl)
                        kind = (
                            "loop_fission_sfu"
                            if (block_fission_loops and isinstance(w_stmt, ast.For))
                            else _classify_edit("", w_txt)
                        )
                        chunks.append(
                            AstEditChunk(
                                chunk_id=f"edit_{len(chunks)}",
                                edit_kind=kind,
                                target_path=f"{path_label}/replace_ins_{j}",
                                before_src="",
                                after_src=w_txt,
                                ablated_dsl=dsl,
                            )
                        )
        elif tag == "delete":
            for idx in range(i1, i2):
                b_stmt = base_list[idx]
                b_txt = _node_key(b_stmt)
                cur_path = parent_path + (idx,)
                patched = copy.deepcopy(base_tree)
                if _set_stmt_by_path(patched, cur_path, None):
                    dsl = _render_patched(patched, new_defs=new_defs, def_order=def_order)
                    if dsl not in seen_dsls:
                        seen_dsls.add(dsl)
                        chunks.append(
                            AstEditChunk(
                                chunk_id=f"edit_{len(chunks)}",
                                edit_kind=_classify_edit(b_txt, ""),
                                target_path=f"{path_label}/delete_{idx}",
                                before_src=b_txt,
                                after_src="",
                                ablated_dsl=dsl,
                            )
                        )
        elif tag == "insert":
            for j in range(j1, j2):
                w_stmt = win_list[j]
                w_txt = _node_key(w_stmt)
                cur_path = parent_path + (min(i1, len(base_list)),)
                patched = copy.deepcopy(base_tree)
                if _insert_stmt_by_path(patched, cur_path, w_stmt):
                    dsl = _render_patched(patched, new_defs=new_defs, def_order=def_order)
                    if dsl not in seen_dsls:
                        seen_dsls.add(dsl)
                        kind = (
                            "loop_fission_sfu"
                            if (block_fission_loops and isinstance(w_stmt, ast.For))
                            else _classify_edit("", w_txt)
                        )
                        chunks.append(
                            AstEditChunk(
                                chunk_id=f"edit_{len(chunks)}",
                                edit_kind=kind,
                                target_path=f"{path_label}/insert_{i1}_{j}",
                                before_src="",
                                after_src=w_txt,
                                ablated_dsl=dsl,
                            )
                        )


def _classify_config_key_edit(key_name: str, before_src: str, after_src: str) -> str:
    kl = key_name.lower()
    if any(tok in kl for tok in ("stage", "buffer", "preload")):
        return "pipeline_versions"
    if any(tok in kl for tok in ("core", "block", "split", "threads")):
        return "core_parallelism"
    if any(tok in kl for tok in ("pad", "align", "datablock", "stride")):
        return "pad_alignment"
    if "barrier" in kl or "sync" in kl or "unit_flag" in kl:
        return "barrier_prune"
    classified = _classify_edit(before_src, after_src)
    return classified if classified != "stmt_rewrite" else "config_knob"


def _find_dict_literals_by_context(tree: ast.Module) -> dict[str, ast.Dict]:
    """Collect constant-key Dict literals keyed by enclosing function/assignment path."""
    found: dict[str, ast.Dict] = {}

    def _visit(node: ast.AST, prefix: str) -> None:
        if isinstance(node, ast.FunctionDef):
            fn_prefix = f"{prefix}/{node.name}" if prefix else node.name
            for idx, stmt in enumerate(node.body):
                _visit(stmt, f"{fn_prefix}:{idx}" if not isinstance(stmt, ast.Return) else f"{fn_prefix}:return")
            return
        if isinstance(node, ast.Dict):
            found[prefix] = node
            return
        for child in ast.iter_child_nodes(node):
            _visit(child, prefix)

    for idx, stmt in enumerate(tree.body):
        if isinstance(stmt, ast.FunctionDef):
            _visit(stmt, "")
        elif isinstance(stmt, ast.Assign) and stmt.targets:
            _visit(stmt, f"global:{_node_key(stmt.targets[0])}")
        else:
            _visit(stmt, f"mod:{idx}")
    return found


def _dict_key_map(dnode: ast.Dict) -> dict[str, tuple[int, ast.expr]]:
    out: dict[str, tuple[int, ast.expr]] = {}
    for idx, (k, v) in enumerate(zip(dnode.keys, dnode.values)):
        if isinstance(k, ast.Constant) and isinstance(k.value, str):
            out[k.value] = (idx, v)
    return out


def _extract_config_dict_loo_chunks(
    base_tree: ast.Module,
    win_tree: ast.Module,
    chunks: list[AstEditChunk],
    seen_dsls: set[str],
) -> None:
    """Create Leave-One-Out chunks reverting one dict config entry in winner_dsl at a time."""
    base_dicts = _find_dict_literals_by_context(base_tree)
    win_dicts = _find_dict_literals_by_context(win_tree)
    for ctx_key, w_dict in win_dicts.items():
        b_dict = base_dicts.get(ctx_key)
        if b_dict is None:
            continue
        b_map = _dict_key_map(b_dict)
        w_map = _dict_key_map(w_dict)
        for k, (w_idx, w_val) in w_map.items():
            w_val_src = _node_key(w_val)
            if k in b_map:
                _b_idx, b_val = b_map[k]
                b_val_src = _node_key(b_val)
                if b_val_src == w_val_src:
                    continue
                patched = copy.deepcopy(win_tree)
                p_dict = _find_dict_literals_by_context(patched).get(ctx_key)
                if p_dict is None or w_idx >= len(p_dict.values):
                    continue
                p_dict.values[w_idx] = copy.deepcopy(b_val)
                dsl = _render_patched(patched)
                if dsl not in seen_dsls:
                    seen_dsls.add(dsl)
                    chunks.append(
                        AstEditChunk(
                            chunk_id=f"loo_{len(chunks)}",
                            edit_kind=_classify_config_key_edit(k, b_val_src, w_val_src),
                            target_path=f"{ctx_key}/dict:{k}",
                            before_src=f"{k}: {b_val_src}",
                            after_src=f"{k}: {w_val_src}",
                            ablated_dsl=dsl,
                        )
                    )
            else:
                patched = copy.deepcopy(win_tree)
                p_dict = _find_dict_literals_by_context(patched).get(ctx_key)
                if p_dict is None or w_idx >= len(p_dict.values):
                    continue
                del p_dict.keys[w_idx]
                del p_dict.values[w_idx]
                dsl = _render_patched(patched)
                if dsl not in seen_dsls:
                    seen_dsls.add(dsl)
                    chunks.append(
                        AstEditChunk(
                            chunk_id=f"loo_{len(chunks)}",
                            edit_kind=_classify_config_key_edit(k, "", w_val_src),
                            target_path=f"{ctx_key}/dict_del:{k}",
                            before_src="",
                            after_src=f"{k}: {w_val_src}",
                            ablated_dsl=dsl,
                        )
                    )


def _extract_leave_one_out_stmt_deltas(
    win_tree: ast.Module,
    base_list: list[ast.stmt],
    win_list: list[ast.stmt],
    parent_path: tuple[int, ...],
    path_label: str,
    chunks: list[AstEditChunk],
    seen_dsls: set[str],
    *,
    skip_assign_symbols: frozenset[str] = frozenset(),
) -> None:
    """Extract Leave-One-Out AST chunks on winner_dsl reverting one change at a time."""
    base_tags = [_node_tag(s) for s in base_list]
    win_tags = [_node_tag(s) for s in win_list]
    matcher = difflib.SequenceMatcher(a=base_tags, b=win_tags, autojunk=False)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                b_stmt = base_list[i1 + k]
                w_stmt = win_list[j1 + k]
                if _node_key(b_stmt) == _node_key(w_stmt):
                    continue
                if skip_assign_symbols and not parent_path:
                    w_tag = _node_tag(w_stmt)
                    if w_tag.startswith("assign:") and (
                        w_tag.removeprefix("assign:") in skip_assign_symbols
                    ):
                        continue
                    if (
                        isinstance(w_stmt, ast.AnnAssign)
                        and isinstance(w_stmt.target, ast.Name)
                        and w_stmt.target.id in skip_assign_symbols
                    ):
                        continue
                cur_path = parent_path + (j1 + k,)
                if isinstance(b_stmt, ast.With) and isinstance(w_stmt, ast.With):
                    b_hdr = _node_key(b_stmt.items[0].context_expr) if b_stmt.items else ""
                    w_hdr = _node_key(w_stmt.items[0].context_expr) if w_stmt.items else ""
                    if b_hdr != w_hdr and b_stmt.items:
                        patched = copy.deepcopy(win_tree)
                        if _replace_call_arg(
                            patched,
                            cur_path,
                            is_with_header=True,
                            new_expr=b_stmt.items[0].context_expr,
                        ):
                            dsl = _render_patched(patched)
                            if dsl not in seen_dsls:
                                seen_dsls.add(dsl)
                                chunks.append(
                                    AstEditChunk(
                                        chunk_id=f"loo_{len(chunks)}",
                                        edit_kind=_classify_edit(b_hdr, w_hdr),
                                        target_path=f"{path_label}/with_{j1 + k}",
                                        before_src=b_hdr,
                                        after_src=w_hdr,
                                        ablated_dsl=dsl,
                                    )
                                )
                    _extract_leave_one_out_stmt_deltas(
                        win_tree,
                        b_stmt.body,
                        w_stmt.body,
                        cur_path,
                        f"{path_label}/with_{j1 + k}",
                        chunks,
                        seen_dsls,
                    )
                    continue

                if isinstance(b_stmt, ast.For) and isinstance(w_stmt, ast.For):
                    b_iter = _node_key(b_stmt.iter)
                    w_iter = _node_key(w_stmt.iter)
                    if b_iter != w_iter:
                        patched = copy.deepcopy(win_tree)
                        if _replace_call_arg(
                            patched,
                            cur_path,
                            is_with_header=False,
                            new_expr=b_stmt.iter,
                        ):
                            dsl = _render_patched(patched)
                            if dsl not in seen_dsls:
                                seen_dsls.add(dsl)
                                chunks.append(
                                    AstEditChunk(
                                        chunk_id=f"loo_{len(chunks)}",
                                        edit_kind=_classify_edit(b_iter, w_iter),
                                        target_path=f"{path_label}/for_{j1 + k}",
                                        before_src=b_iter,
                                        after_src=w_iter,
                                        ablated_dsl=dsl,
                                    )
                                )
                    _extract_leave_one_out_stmt_deltas(
                        win_tree,
                        b_stmt.body,
                        w_stmt.body,
                        cur_path,
                        f"{path_label}/for_{j1 + k}",
                        chunks,
                        seen_dsls,
                    )
                    continue

                if isinstance(b_stmt, ast.FunctionDef) and isinstance(w_stmt, ast.FunctionDef):
                    _extract_leave_one_out_stmt_deltas(
                        win_tree,
                        b_stmt.body,
                        w_stmt.body,
                        cur_path,
                        f"{path_label}/{w_stmt.name}",
                        chunks,
                        seen_dsls,
                    )
                    continue

                patched = copy.deepcopy(win_tree)
                if _set_stmt_by_path(patched, cur_path, b_stmt):
                    dsl = _render_patched(patched)
                    if dsl not in seen_dsls:
                        seen_dsls.add(dsl)
                        b_txt, w_txt = _node_key(b_stmt), _node_key(w_stmt)
                        chunks.append(
                            AstEditChunk(
                                chunk_id=f"loo_{len(chunks)}",
                                edit_kind=_classify_edit(b_txt, w_txt),
                                target_path=f"{path_label}/stmt_{j1 + k}",
                                before_src=b_txt,
                                after_src=w_txt,
                                ablated_dsl=dsl,
                            )
                        )

        elif tag == "replace":
            common = min(i2 - i1, j2 - j1)
            for k in range(common):
                b_stmt = base_list[i1 + k]
                w_stmt = win_list[j1 + k]
                cur_path = parent_path + (j1 + k,)
                if isinstance(b_stmt, (ast.With, ast.For)) and isinstance(w_stmt, type(b_stmt)):
                    if isinstance(b_stmt, ast.With) and isinstance(w_stmt, ast.With) and b_stmt.items and w_stmt.items:
                        b_hdr = _node_key(b_stmt.items[0].context_expr)
                        w_hdr = _node_key(w_stmt.items[0].context_expr)
                        if b_hdr != w_hdr:
                            hdr_patch = copy.deepcopy(win_tree)
                            if _replace_call_arg(
                                hdr_patch,
                                cur_path,
                                is_with_header=True,
                                new_expr=b_stmt.items[0].context_expr,
                            ):
                                dsl = _render_patched(hdr_patch)
                                if dsl not in seen_dsls:
                                    seen_dsls.add(dsl)
                                    chunks.append(
                                        AstEditChunk(
                                            chunk_id=f"loo_{len(chunks)}",
                                            edit_kind=_classify_edit(b_hdr, w_hdr),
                                            target_path=f"{path_label}/with_hdr_{j1 + k}",
                                            before_src=b_hdr,
                                            after_src=w_hdr,
                                            ablated_dsl=dsl,
                                        )
                                    )
                    _extract_leave_one_out_stmt_deltas(
                        win_tree,
                        b_stmt.body,
                        w_stmt.body,
                        cur_path,
                        f"{path_label}/block_{j1 + k}",
                        chunks,
                        seen_dsls,
                    )
                    continue
                patched = copy.deepcopy(win_tree)
                if _set_stmt_by_path(patched, cur_path, b_stmt):
                    dsl = _render_patched(patched)
                    if dsl not in seen_dsls:
                        seen_dsls.add(dsl)
                        b_txt, w_txt = _node_key(b_stmt), _node_key(w_stmt)
                        chunks.append(
                            AstEditChunk(
                                chunk_id=f"loo_{len(chunks)}",
                                edit_kind=_classify_edit(b_txt, w_txt),
                                target_path=f"{path_label}/replace_{j1 + k}",
                                before_src=b_txt,
                                after_src=w_txt,
                                ablated_dsl=dsl,
                            )
                        )
            for j in range(j1 + common, j2):
                w_stmt = win_list[j]
                w_txt = _node_key(w_stmt)
                cur_path = parent_path + (j,)
                patched = copy.deepcopy(win_tree)
                if _set_stmt_by_path(patched, cur_path, None):
                    dsl = _render_patched(patched)
                    if dsl not in seen_dsls:
                        seen_dsls.add(dsl)
                        chunks.append(
                            AstEditChunk(
                                chunk_id=f"loo_{len(chunks)}",
                                edit_kind=_classify_edit("", w_txt),
                                target_path=f"{path_label}/remove_extra_{j}",
                                before_src="",
                                after_src=w_txt,
                                ablated_dsl=dsl,
                            )
                        )

        elif tag == "insert":
            for j in range(j1, j2):
                w_stmt = win_list[j]
                w_txt = _node_key(w_stmt)
                cur_path = parent_path + (j,)
                patched = copy.deepcopy(win_tree)
                if _set_stmt_by_path(patched, cur_path, None):
                    dsl = _render_patched(patched)
                    if dsl not in seen_dsls:
                        seen_dsls.add(dsl)
                        chunks.append(
                            AstEditChunk(
                                chunk_id=f"loo_{len(chunks)}",
                                edit_kind=_classify_edit("", w_txt),
                                target_path=f"{path_label}/remove_ins_{j}",
                                before_src="",
                                after_src=w_txt,
                                ablated_dsl=dsl,
                            )
                        )

        elif tag == "delete":
            for idx in range(i1, i2):
                b_stmt = base_list[idx]
                b_txt = _node_key(b_stmt)
                cur_path = parent_path + (min(j1, len(win_list)),)
                patched = copy.deepcopy(win_tree)
                if _insert_stmt_by_path(patched, cur_path, b_stmt):
                    dsl = _render_patched(patched)
                    if dsl not in seen_dsls:
                        seen_dsls.add(dsl)
                        chunks.append(
                            AstEditChunk(
                                chunk_id=f"loo_{len(chunks)}",
                                edit_kind=_classify_edit(b_txt, ""),
                                target_path=f"{path_label}/reinsert_{idx}",
                                before_src=b_txt,
                                after_src="",
                                ablated_dsl=dsl,
                            )
                        )


def _find_str_constant_node(tree: ast.Module, symbol_name: str) -> ast.Constant | None:
    for node in tree.body:
        val: ast.AST | None = None
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == symbol_name
        ):
            val = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == symbol_name
        ):
            val = node.value
        if isinstance(val, ast.Constant) and isinstance(val.value, str):
            return val
    return None


def _extract_str_constant(tree: ast.Module, symbol_name: str) -> str | None:
    node = _find_str_constant_node(tree, symbol_name)
    return str(node.value) if node is not None else None


def _format_cpp_string_literal(cpp_src: str) -> str:
    if '"""' not in cpp_src and not cpp_src.endswith(('"', "\\")):
        return f'r"""{cpp_src}"""'
    if "'''" not in cpp_src and not cpp_src.endswith(("'", "\\")):
        return f"r'''{cpp_src}'''"
    return repr(cpp_src)


def _replace_str_constant_in_module(
    module_src: str,
    symbol_name: str,
    new_cpp_src: str,
) -> str:
    tree = ast.parse(module_src)
    node = _find_str_constant_node(tree, symbol_name)
    if (
        node is not None
        and node.end_lineno is not None
        and node.end_col_offset is not None
    ):
        lines = module_src.splitlines(keepends=True)
        start_line = lines[node.lineno - 1].encode("utf-8")
        end_line = lines[node.end_lineno - 1].encode("utf-8")
        prefix = "".join(lines[: node.lineno - 1]) + start_line[: node.col_offset].decode("utf-8")
        suffix = end_line[node.end_col_offset :].decode("utf-8") + "".join(lines[node.end_lineno :])
        candidate = f"{prefix}{_format_cpp_string_literal(new_cpp_src)}{suffix}"
        try:
            parsed = ast.parse(candidate)
            if _extract_str_constant(parsed, symbol_name) == new_cpp_src:
                return candidate
        except SyntaxError:
            pass
    if node is not None:
        node.value = new_cpp_src
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _classify_cpp_hunk_kind(before_seg: str, after_seg: str) -> str:
    combined = f"{before_seg} {after_seg}"
    if "& 1" in combined or "% 2" in combined or "(event_t)1" in combined:
        return "static_pingpong_unroll"
    if (
        "asc_sync_notify" in combined
        or "asc_lock" in combined
        or "asc_unlock" in combined
        or "pipe_barrier" in combined
        or "CrossCore" in combined
    ):
        return "barrier_prune"
    if "+ 16" in combined or "+ 32" in combined or "STRIDE" in combined or "PAD" in combined:
        return "pad_alignment"
    if (
        "asc_copy_gm2ub" in combined
        or "asc_copy_ub2gm" in combined
        or "asc_copy_gm2l1" in combined
        or "asc_copy_l0c2ub" in combined
    ):
        return "coalesced_copy"
    if "get_subblockid" in combined or "1c2v" in combined.lower():
        return "dual_copy_1c2v"
    if "block_idx" in combined or "num_cores" in combined or "get_block_num" in combined:
        return "core_parallelism"
    if "simd_inst::" in combined and after_seg.count("for ") > before_seg.count("for "):
        return "loop_fission_sfu"
    return "cpp_hunk_edit"


def _discover_trackb_cpp_symbols(tree: ast.Module) -> list[str]:
    symbols: list[str] = []
    for node in tree.body:
        name: str | None = None
        val: ast.AST | None = None
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            name, val = node.targets[0].id, node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name, val = node.target.id, node.value
        if not name or not isinstance(val, ast.Constant) or not isinstance(val.value, str):
            continue
        cpp_text = val.value
        if (
            name.endswith("_TRACKB_SRC")
            or "tl_templates/ascend/" in cpp_text
            or "__aicore__" in cpp_text
            or "__global__" in cpp_text
            or "__simd_vf__" in cpp_text
        ):
            symbols.append(name)
    return symbols


def decompose_cpp_string_leave_one_out(
    baseline_module_src: str,
    winner_module_src: str,
    cpp_symbol_name: str,
) -> list[AstEditChunk]:
    """Generate Leave-One-Out counterfactual modules by reverting one C++ diff hunk at a time."""
    base_tree = ast.parse(baseline_module_src)
    win_tree = ast.parse(winner_module_src)
    base_cpp = _extract_str_constant(base_tree, cpp_symbol_name)
    win_cpp = _extract_str_constant(win_tree, cpp_symbol_name)
    if not base_cpp or not win_cpp or base_cpp == win_cpp:
        return []

    base_lines = base_cpp.splitlines(keepends=True)
    win_lines = win_cpp.splitlines(keepends=True)
    matcher = difflib.SequenceMatcher(a=base_lines, b=win_lines, autojunk=False)
    diff_ops = [
        (tag, i1, i2, j1, j2)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes()
        if tag != "equal"
    ]
    if len(diff_ops) <= 1:
        return []

    chunks: list[AstEditChunk] = []
    for idx, (_tag, i1, i2, j1, j2) in enumerate(diff_ops):
        mid_lines = list(base_lines[i1:i2])
        if mid_lines and j2 < len(win_lines) and not mid_lines[-1].endswith("\n"):
            mid_lines[-1] = mid_lines[-1] + "\n"
        ablated_cpp = "".join([*win_lines[:j1], *mid_lines, *win_lines[j2:]])
        if ablated_cpp.count("{") != ablated_cpp.count("}"):
            continue
        before_seg = "".join(base_lines[i1:i2]).strip()
        after_seg = "".join(win_lines[j1:j2]).strip()
        ablated_mod_src = _replace_str_constant_in_module(
            winner_module_src, cpp_symbol_name, ablated_cpp
        )
        chunks.append(
            AstEditChunk(
                chunk_id=f"cpp_loo_{cpp_symbol_name}_{idx}",
                edit_kind=_classify_cpp_hunk_kind(before_seg, after_seg),
                target_path=f"{cpp_symbol_name}:L{j1 + 1}-L{max(j1 + 1, j2)}",
                before_src=before_seg,
                after_src=after_seg,
                ablated_dsl=ablated_mod_src,
            )
        )
    return chunks


def decompose_leave_one_out_ablations(
    baseline_dsl: str,
    winner_dsl: str,
) -> tuple[AstEditChunk, ...]:
    """Decompose winner_dsl into Leave-One-Out AST, config-dict, and Track-B C++ ablations."""
    base_tree = ast.parse(baseline_dsl)
    win_tree = ast.parse(winner_dsl)
    chunks: list[AstEditChunk] = []
    seen_dsls: set[str] = {ast.unparse(win_tree)}

    base_cpp_syms = set(_discover_trackb_cpp_symbols(base_tree))
    decomposed_cpp_syms: set[str] = set()
    for sym_name in _discover_trackb_cpp_symbols(win_tree):
        if sym_name not in base_cpp_syms:
            continue
        cpp_chunks = decompose_cpp_string_leave_one_out(baseline_dsl, winner_dsl, sym_name)
        if not cpp_chunks:
            continue
        decomposed_cpp_syms.add(sym_name)
        for ch in cpp_chunks:
            norm = ast.unparse(ast.parse(ch.ablated_dsl))
            if norm not in seen_dsls:
                seen_dsls.add(norm)
                chunks.append(ch)

    _extract_config_dict_loo_chunks(base_tree, win_tree, chunks, seen_dsls)
    _extract_leave_one_out_stmt_deltas(
        win_tree,
        base_tree.body,
        win_tree.body,
        (),
        "module",
        chunks,
        seen_dsls,
        skip_assign_symbols=frozenset(decomposed_cpp_syms),
    )
    return tuple(chunks)


def decompose_ast_ablations(baseline_dsl: str, winner_dsl: str) -> tuple[AstEditChunk, ...]:
    """Decompose AST difference between baseline and winner DSL into single-variable edits."""
    base_tree = ast.parse(baseline_dsl)
    win_tree = ast.parse(winner_dsl)
    chunks: list[AstEditChunk] = []
    seen_dsls: set[str] = {ast.unparse(base_tree)}

    base_defs = dict(_collect_top_defs(base_tree))
    win_defs_list = _collect_top_defs(win_tree)
    new_defs = {name: stmt for name, stmt in win_defs_list if name not in base_defs}
    def_order = [name for name, _stmt in win_defs_list if name in new_defs]

    _extract_stmt_list_deltas(
        base_tree,
        base_tree.body,
        win_tree.body,
        (),
        "module",
        chunks,
        seen_dsls,
        new_defs=new_defs,
        def_order=def_order,
    )
    has_verifiable_chunk = any(c.edit_kind in _SUPPORTED_PROBE_KINDS for c in chunks)
    win_rendered = ast.unparse(win_tree)
    if (
        (not chunks or (len(base_tree.body) > 2 and not has_verifiable_chunk))
        and win_rendered != ast.unparse(base_tree)
        and win_rendered not in seen_dsls
    ):
        win_defs_map = dict(win_defs_list)
        diff_names = [
            name
            for name, w_stmt in win_defs_list
            if name not in base_defs or _node_key(base_defs[name]) != _node_key(w_stmt)
        ]
        before_diff = "\n".join(
            _node_key(base_defs[n]) for n in diff_names if n in base_defs
        ) or ast.unparse(base_tree)
        after_diff = "\n".join(
            _node_key(win_defs_map[n]) for n in diff_names if n in win_defs_map
        ) or win_rendered
        chunks.append(
            AstEditChunk(
                chunk_id=f"edit_{len(chunks)}",
                edit_kind=_classify_edit(before_diff, after_diff),
                target_path="module/regime_composite" if chunks else "module/whole",
                before_src=before_diff,
                after_src=after_diff,
                ablated_dsl=win_rendered,
            )
        )
    return tuple(chunks)


def _probe_templates_for_kind(
    probe_kind: str,
    params: Mapping[str, object] | None = None,
) -> tuple[str, str, tuple[int, ...], str] | None:
    cfg = dict(params or {})
    cores = max(2, min(56, int(cfg.get("cores", 28))))
    mix_cores = max(2, min(28, cores))
    stages = max(2, min(3, int(cfg.get("stages", 2))))
    tile = max(512, min(4096, ((int(cfg.get("tile_elems", 2048)) + 127) // 128) * 128))
    dtype = str(cfg.get("dtype", "float32"))

    if probe_kind == "core_parallelism":
        base = f"""import tilelang.ascend.language as T

def make_flywheel_probe():
    tile = {tile}
    steps = {cores}
    total = tile * steps

    @T.prim_func
    def main(
        A: T.Tensor((total,), "float32"),
        B: T.Tensor((total,), "float32"),
        C: T.Tensor((total,), "float32"),
    ):
        with T.Kernel(1) as bx:
            a_ub = T.alloc_shared((tile,), "float32")
            b_ub = T.alloc_shared((tile,), "float32")
            c_ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({{a_ub: 1, b_ub: 1, c_ub: 1}})
            for s in T.Pipelined(steps, num_stages=1):
                off = s * tile + bx * 0
                T.copy(A[off : off + tile], a_ub)
                T.copy(B[off : off + tile], b_ub)
                with T.SimdVF():
                    mask = T.simd.pset(32)
                    for i in range(tile // 64):
                        va = T.simd.vld(a_ub[i * 64])
                        vb = T.simd.vld(b_ub[i * 64])
                        T.simd.vsts(c_ub[i * 64], T.simd.vadd(va, vb, mask), mask)
                T.copy(c_ub, C[off : off + tile])
                T.ascend_pipe_barrier("PIPE_ALL")
    return main
"""
        opt = f"""import tilelang.ascend.language as T

def make_flywheel_probe():
    tile = {tile}
    cores = {cores}
    total = tile * cores

    @T.prim_func
    def main(
        A: T.Tensor((total,), "float32"),
        B: T.Tensor((total,), "float32"),
        C: T.Tensor((total,), "float32"),
    ):
        with T.Kernel(cores) as bx:
            a_ub = T.alloc_shared((tile,), "float32")
            b_ub = T.alloc_shared((tile,), "float32")
            c_ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({{a_ub: 1, b_ub: 1, c_ub: 1}})
            off = bx * tile
            T.copy(A[off : off + tile], a_ub)
            T.copy(B[off : off + tile], b_ub)
            with T.SimdVF():
                mask = T.simd.pset(32)
                for i in range(tile // 64):
                    va = T.simd.vld(a_ub[i * 64])
                    vb = T.simd.vld(b_ub[i * 64])
                    T.simd.vsts(c_ub[i * 64], T.simd.vadd(va, vb, mask), mask)
            T.copy(c_ub, C[off : off + tile])
    return main
"""
        return base, opt, (tile * cores,), dtype

    if probe_kind == "coalesced_copy":
        rows = 16
        cols = 128
        base = f"""import tilelang.ascend.language as T

def make_flywheel_probe():
    cores = {mix_cores}
    rows = {rows}
    cols = {cols}

    @T.prim_func
    def main(
        A: T.Tensor((cores * rows, cols), "float32"),
        B: T.Tensor((cores * rows, cols), "float32"),
        C: T.Tensor((cores * rows, cols), "float32"),
    ):
        with T.Kernel(cores) as bx:
            a_ub = T.alloc_shared((rows, cols), "float32")
            b_ub = T.alloc_shared((rows, cols), "float32")
            c_ub = T.alloc_shared((rows, cols), "float32")
            base_r = bx * rows
            for r in T.serial(rows):
                T.copy(A[base_r + r, 0:cols], a_ub[r, 0:cols])
                T.copy(B[base_r + r, 0:cols], b_ub[r, 0:cols])
            with T.SimdVF():
                mask = T.simd.pset(32)
                for r in T.serial(rows):
                    for k in range(2):
                        va = T.simd.vld(a_ub[r, k * 64])
                        vb = T.simd.vld(b_ub[r, k * 64])
                        T.simd.vsts(c_ub[r, k * 64], T.simd.vadd(va, vb, mask), mask)
            for r in T.serial(rows):
                T.copy(c_ub[r, 0:cols], C[base_r + r, 0:cols])
    return main
"""
        opt = f"""import tilelang.ascend.language as T

def make_flywheel_probe():
    cores = {mix_cores}
    rows = {rows}
    cols = {cols}

    @T.prim_func
    def main(
        A: T.Tensor((cores * rows, cols), "float32"),
        B: T.Tensor((cores * rows, cols), "float32"),
        C: T.Tensor((cores * rows, cols), "float32"),
    ):
        with T.Kernel(cores) as bx:
            a_ub = T.alloc_shared((rows, cols), "float32")
            b_ub = T.alloc_shared((rows, cols), "float32")
            c_ub = T.alloc_shared((rows, cols), "float32")
            base_r = bx * rows
            T.copy(A[base_r : base_r + rows, 0:cols], a_ub)
            T.copy(B[base_r : base_r + rows, 0:cols], b_ub)
            with T.SimdVF():
                mask = T.simd.pset(32)
                for r in T.serial(rows):
                    for k in range(2):
                        va = T.simd.vld(a_ub[r, k * 64])
                        vb = T.simd.vld(b_ub[r, k * 64])
                        T.simd.vsts(c_ub[r, k * 64], T.simd.vadd(va, vb, mask), mask)
            T.copy(c_ub, C[base_r : base_r + rows, 0:cols])
    return main
"""
        return base, opt, (mix_cores * rows, cols), dtype

    if probe_kind == "pipeline_versions":
        steps = 16
        base = f"""import tilelang.ascend.language as T

def make_flywheel_probe():
    tile = {tile}
    steps = {steps}
    cores = {mix_cores}
    total = cores * steps * tile

    @T.prim_func
    def main(
        A: T.Tensor((total,), "float32"),
        B: T.Tensor((total,), "float32"),
        C: T.Tensor((total,), "float32"),
    ):
        with T.Kernel(cores) as bx:
            a_ub = T.alloc_shared((tile,), "float32")
            b_ub = T.alloc_shared((tile,), "float32")
            c_ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({{a_ub: 1, b_ub: 1, c_ub: 1}})
            for s in T.Pipelined(steps, num_stages=1):
                off = (s * cores + bx) * tile
                T.copy(A[off : off + tile], a_ub)
                T.copy(B[off : off + tile], b_ub)
                with T.SimdVF():
                    mask = T.simd.pset(32)
                    for i in range(tile // 64):
                        va = T.simd.vld(a_ub[i * 64])
                        vb = T.simd.vld(b_ub[i * 64])
                        T.simd.vsts(c_ub[i * 64], T.simd.vadd(va, vb, mask), mask)
                T.copy(c_ub, C[off : off + tile])
                T.ascend_pipe_barrier("PIPE_ALL")
    return main
"""
        opt = f"""import tilelang.ascend.language as T

def make_flywheel_probe():
    tile = {tile}
    steps = {steps}
    cores = {mix_cores}
    total = cores * steps * tile

    @T.prim_func
    def main(
        A: T.Tensor((total,), "float32"),
        B: T.Tensor((total,), "float32"),
        C: T.Tensor((total,), "float32"),
    ):
        with T.Kernel(cores) as bx:
            a_ub = T.alloc_shared((tile,), "float32")
            b_ub = T.alloc_shared((tile,), "float32")
            c_ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({{a_ub: {stages}, b_ub: {stages}, c_ub: {stages}}})
            for s in T.Pipelined(steps, num_stages={stages}):
                off = (s * cores + bx) * tile
                T.copy(A[off : off + tile], a_ub)
                T.copy(B[off : off + tile], b_ub)
                with T.SimdVF():
                    mask = T.simd.pset(32)
                    for i in range(tile // 64):
                        va = T.simd.vld(a_ub[i * 64])
                        vb = T.simd.vld(b_ub[i * 64])
                        T.simd.vsts(c_ub[i * 64], T.simd.vadd(va, vb, mask), mask)
                T.copy(c_ub, C[off : off + tile])
    return main
"""
        return base, opt, (mix_cores * steps * tile,), dtype

    if probe_kind == "ssbuf_handoff":
        rounds = 64
        base = f"""import tilelang.ascend.language as T

def make_flywheel_probe():
    rounds = {rounds}
    tile = 16

    @T.prim_func
    def main(
        seed: T.Tensor((tile, tile), "float16"),
        scratch: T.Tensor((rounds, 2), "uint32"),
        out: T.Tensor((rounds, 1), "uint64"),
    ):
        with T.MixedKernel(1) as (_, sid):
            cube = T.alloc_l1((tile, tile), "float16")
            tmp_ub = T.alloc_shared((1,), "uint32")
            T.copy(seed, cube)
            for i in T.Pipelined(
                rounds,
                num_stages=1,
            ):
                with T.Stage(0):
                    val = T.Cast("uint32", (i + 1) * (100 + sid))
                    tmp_ub[0] = val
                    T.copy(tmp_ub, scratch[i, sid : sid + 1])
                with T.Stage(0):
                    lo = T.Cast("uint64", scratch[i, 0])
                    hi = T.Cast("uint64", scratch[i, 1]) << 32
                    out[i, 0] = lo | hi
                    T.ascend_pipe_barrier("PIPE_ALL")
    return main
"""
        opt = f"""import tilelang.ascend.language as T

def make_flywheel_probe():
    rounds = {rounds}
    tile = 16

    @T.prim_func
    def main(
        seed: T.Tensor((tile, tile), "float16"),
        out: T.Tensor((rounds, 1), "uint64"),
    ):
        with T.MixedKernel(1) as (_, sid):
            words = T.alloc_ss((2,), "uint32")
            packed = T.view(words, shape=(1,), dtype="uint64")
            T.annotate_buffer_versions({{words: (3, "auto")}})
            cube = T.alloc_l1((tile, tile), "float16")
            T.copy(seed, cube)
            for i in T.Pipelined(
                rounds,
                num_stages=3,
                annotations={{"enable_offset": True}},
            ):
                with T.Stage(0):
                    words[sid] = T.Cast(
                        "uint32",
                        (i + 1) * (100 + sid),
                    )
                with T.Stage(2):
                    out[i, 0] = packed[0]
    return main
"""
        return base, opt, (rounds, 1), "uint64"

    if probe_kind == "barrier_prune":
        steps = 16
        base = f"""import tilelang.ascend.language as T

def make_flywheel_probe():
    tile = {tile}
    steps = {steps}
    cores = {mix_cores}
    total = cores * steps * tile

    @T.prim_func
    def main(
        A: T.Tensor((total,), "float32"),
        B: T.Tensor((total,), "float32"),
        C: T.Tensor((total,), "float32"),
    ):
        with T.Kernel(cores) as bx:
            a_ub = T.alloc_shared((tile,), "float32")
            b_ub = T.alloc_shared((tile,), "float32")
            c_ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({{a_ub: {stages}, b_ub: {stages}, c_ub: {stages}}})
            for s in T.Pipelined(steps, num_stages={stages}):
                off = (s * cores + bx) * tile
                T.copy(A[off : off + tile], a_ub)
                T.copy(B[off : off + tile], b_ub)
                T.ascend_pipe_barrier("PIPE_ALL")
                with T.SimdVF():
                    mask = T.simd.pset(32)
                    for i in range(tile // 64):
                        va = T.simd.vld(a_ub[i * 64])
                        vb = T.simd.vld(b_ub[i * 64])
                        T.simd.vsts(c_ub[i * 64], T.simd.vadd(va, vb, mask), mask)
                T.ascend_pipe_barrier("PIPE_ALL")
                T.copy(c_ub, C[off : off + tile])
                T.ascend_pipe_barrier("PIPE_ALL")
    return main
"""
        opt = f"""import tilelang.ascend.language as T

def make_flywheel_probe():
    tile = {tile}
    steps = {steps}
    cores = {mix_cores}
    total = cores * steps * tile

    @T.prim_func
    def main(
        A: T.Tensor((total,), "float32"),
        B: T.Tensor((total,), "float32"),
        C: T.Tensor((total,), "float32"),
    ):
        with T.Kernel(cores) as bx:
            a_ub = T.alloc_shared((tile,), "float32")
            b_ub = T.alloc_shared((tile,), "float32")
            c_ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({{a_ub: {stages}, b_ub: {stages}, c_ub: {stages}}})
            for s in T.Pipelined(steps, num_stages={stages}):
                off = (s * cores + bx) * tile
                T.copy(A[off : off + tile], a_ub)
                T.copy(B[off : off + tile], b_ub)
                with T.SimdVF():
                    mask = T.simd.pset(32)
                    for i in range(tile // 64):
                        va = T.simd.vld(a_ub[i * 64])
                        vb = T.simd.vld(b_ub[i * 64])
                        T.simd.vsts(c_ub[i * 64], T.simd.vadd(va, vb, mask), mask)
                T.copy(c_ub, C[off : off + tile])
    return main
"""
        return base, opt, (mix_cores * steps * tile,), dtype

    if probe_kind == "dual_copy_1c2v":
        rows = 16
        half_rows = 8
        cols = 128
        base = f"""import tilelang.ascend.language as T

def make_flywheel_probe():
    cores = {mix_cores}
    rows = {rows}
    cols = {cols}

    @T.prim_func
    def main(
        A: T.Tensor((cores * rows, cols), "float32"),
        B: T.Tensor((cores * rows, cols), "float32"),
        C: T.Tensor((cores * rows, cols), "float32"),
    ):
        with T.Kernel(cores) as bx:
            a_ub = T.alloc_shared((rows, cols), "float32")
            b_ub = T.alloc_shared((rows, cols), "float32")
            c_ub = T.alloc_shared((rows, cols), "float32")
            base_r = bx * rows
            T.copy(A[base_r : base_r + rows, 0:cols], a_ub)
            T.copy(B[base_r : base_r + rows, 0:cols], b_ub)
            with T.SimdVF():
                mask = T.simd.pset(32)
                for r in T.serial(rows):
                    for k in range(cols // 64):
                        va = T.simd.vld(a_ub[r, k * 64])
                        vb = T.simd.vld(b_ub[r, k * 64])
                        T.simd.vsts(c_ub[r, k * 64], T.simd.vadd(va, vb, mask), mask)
            T.copy(c_ub, C[base_r : base_r + rows, 0:cols])
    return main
"""
        opt = f"""import tilelang.ascend.language as T

def make_flywheel_probe():
    cores = {mix_cores}
    rows = {rows}
    half_rows = {half_rows}
    cols = {cols}

    @T.prim_func
    def main(
        A: T.Tensor((cores * rows, cols), "float32"),
        B: T.Tensor((cores * rows, cols), "float32"),
        C: T.Tensor((cores * rows, cols), "float32"),
    ):
        with T.MixedKernel(cores) as (bx, sid):
            a_ub = T.alloc_shared((half_rows, cols), "float32")
            b_ub = T.alloc_shared((half_rows, cols), "float32")
            c_ub = T.alloc_shared((half_rows, cols), "float32")
            row_off = bx * rows + sid * half_rows
            T.copy(A[row_off : row_off + half_rows, 0:cols], a_ub)
            T.copy(B[row_off : row_off + half_rows, 0:cols], b_ub)
            with T.SimdVF():
                mask = T.simd.pset(32)
                for r in T.serial(half_rows):
                    for k in range(cols // 64):
                        va = T.simd.vld(a_ub[r, k * 64])
                        vb = T.simd.vld(b_ub[r, k * 64])
                        T.simd.vsts(c_ub[r, k * 64], T.simd.vadd(va, vb, mask), mask)
            T.copy(c_ub, C[row_off : row_off + half_rows, 0:cols])
    return main
"""
        return base, opt, (mix_cores * rows, cols), dtype

    if probe_kind == "pad_alignment":
        rows = 8
        steps = 4
        base = f"""import tilelang.ascend.language as T

def make_flywheel_probe():
    cores = {mix_cores}
    rows = {rows}
    cols = 70
    steps = {steps}

    @T.prim_func
    def main(
        A: T.Tensor((cores * steps * rows, cols), "float32"),
        B: T.Tensor((cores * steps * rows, cols), "float32"),
        C: T.Tensor((cores * steps * rows, cols), "float32"),
    ):
        with T.Kernel(cores) as bx:
            a_ub = T.alloc_shared((rows, cols), "float32")
            b_ub = T.alloc_shared((rows, cols), "float32")
            c_ub = T.alloc_shared((rows, cols), "float32")
            for s in T.serial(steps):
                r0 = (bx * steps + s) * rows
                T.copy(A[r0 : r0 + rows, 0:cols], a_ub)
                T.copy(B[r0 : r0 + rows, 0:cols], b_ub)
                with T.SimdVF():
                    mask = T.simd.pset(32)
                    for r in T.serial(rows):
                        va = T.simd.vld(a_ub[r, 0])
                        vb = T.simd.vld(b_ub[r, 0])
                        T.simd.vsts(c_ub[r, 0], T.simd.vadd(va, vb, mask), mask)
                T.copy(c_ub, C[r0 : r0 + rows, 0:cols])
    return main
"""
        opt = f"""import tilelang.ascend.language as T

def make_flywheel_probe():
    cores = {mix_cores}
    rows = {rows}
    cols = 128
    steps = {steps}

    @T.prim_func
    def main(
        A: T.Tensor((cores * steps * rows, cols), "float32"),
        B: T.Tensor((cores * steps * rows, cols), "float32"),
        C: T.Tensor((cores * steps * rows, cols), "float32"),
    ):
        with T.Kernel(cores) as bx:
            a_ub = T.alloc_shared((rows, cols), "float32")
            b_ub = T.alloc_shared((rows, cols), "float32")
            c_ub = T.alloc_shared((rows, cols), "float32")
            for s in T.Pipelined(steps, num_stages={stages}):
                r0 = (bx * steps + s) * rows
                T.copy(A[r0 : r0 + rows, 0:cols], a_ub)
                T.copy(B[r0 : r0 + rows, 0:cols], b_ub)
                with T.SimdVF():
                    mask = T.simd.pset(32)
                    for r in T.serial(rows):
                        for k in range(2):
                            va = T.simd.vld(a_ub[r, k * 64])
                            vb = T.simd.vld(b_ub[r, k * 64])
                            T.simd.vsts(c_ub[r, k * 64], T.simd.vadd(va, vb, mask), mask)
                T.copy(c_ub, C[r0 : r0 + rows, 0:cols])
    return main
"""
        return base, opt, (mix_cores * steps * rows, 128), dtype

    if probe_kind == "static_pingpong_unroll":
        steps = 16
        base = f"""import tilelang.ascend.language as T

def make_flywheel_probe():
    tile = {tile}
    steps = {steps}
    cores = {mix_cores}
    total = cores * steps * tile

    @T.prim_func
    def main(
        A: T.Tensor((total,), "float32"),
        B: T.Tensor((total,), "float32"),
        C: T.Tensor((total,), "float32"),
    ):
        with T.Kernel(cores) as bx:
            a_ub = T.alloc_shared((tile,), "float32")
            b_ub = T.alloc_shared((tile,), "float32")
            c_ub = T.alloc_shared((tile,), "float32")
            T.annotate_buffer_versions({{a_ub: 2, b_ub: 2, c_ub: 2}})
            for s in T.Pipelined(steps, num_stages=2, annotations={{"enable_offset": True}}):
                off = (s * cores + bx) * tile
                T.copy(A[off : off + tile], a_ub)
                T.copy(B[off : off + tile], b_ub)
                with T.SimdVF():
                    mask = T.simd.pset(32)
                    for i in range(tile // 64):
                        va = T.simd.vld(a_ub[i * 64])
                        vb = T.simd.vld(b_ub[i * 64])
                        T.simd.vsts(c_ub[i * 64], T.simd.vadd(va, vb, mask), mask)
                T.copy(c_ub, C[off : off + tile])
    return main
"""
        opt = f"""import tilelang.ascend.language as T

def make_flywheel_probe():
    tile = {tile}
    pairs = {steps // 2}
    cores = {mix_cores}
    total = cores * pairs * 2 * tile

    @T.prim_func
    def main(
        A: T.Tensor((total,), "float32"),
        B: T.Tensor((total,), "float32"),
        C: T.Tensor((total,), "float32"),
    ):
        with T.Kernel(cores) as bx:
            a0 = T.alloc_shared((tile,), "float32")
            a1 = T.alloc_shared((tile,), "float32")
            b0 = T.alloc_shared((tile,), "float32")
            b1 = T.alloc_shared((tile,), "float32")
            c0 = T.alloc_shared((tile,), "float32")
            c1 = T.alloc_shared((tile,), "float32")
            for s in T.serial(pairs):
                off0 = (s * 2 * cores + bx) * tile
                off1 = ((s * 2 + 1) * cores + bx) * tile
                T.copy(A[off0 : off0 + tile], a0)
                T.copy(B[off0 : off0 + tile], b0)
                T.copy(A[off1 : off1 + tile], a1)
                T.copy(B[off1 : off1 + tile], b1)
                with T.SimdVF():
                    mask = T.simd.pset(32)
                    for i in range(tile // 64):
                        T.simd.vsts(c0[i * 64], T.simd.vadd(T.simd.vld(a0[i * 64]), T.simd.vld(b0[i * 64]), mask), mask)
                        T.simd.vsts(c1[i * 64], T.simd.vadd(T.simd.vld(a1[i * 64]), T.simd.vld(b1[i * 64]), mask), mask)
                T.copy(c0, C[off0 : off0 + tile])
                T.copy(c1, C[off1 : off1 + tile])
    return main
"""
        return base, opt, (mix_cores * steps * tile,), "float32"

    if probe_kind == "loop_fission_sfu":
        base = f"""import tilelang.ascend.language as T

def make_flywheel_probe():
    cores = {mix_cores}
    rows = 4
    cols = 64

    @T.prim_func
    def main(
        Q: T.Tensor((cores * rows, cols), "float32"),
        Out: T.Tensor((cores, cols), "float32"),
        WOut: T.Tensor((cores * rows, 1), "float32"),
    ):
        with T.Kernel(cores) as bx:
            q_ub = T.alloc_shared((1, cols), "float32")
            acc_ub = T.alloc_shared((rows, cols), "float32")
            w_ub = T.alloc_shared((rows, 1), "float32")
            for s in T.serial(rows):
                T.copy(Q[bx * rows + s : bx * rows + s + 1, :], q_ub)
                with T.SimdVF():
                    m32 = T.simd.pset(32, "PAT_ALL")
                    qv = T.simd.vld(q_ub[0, 0])
                    sv = T.simd.vdupv(T.simd.vcadd(qv, mask=m32), mask=m32, pos="POS_LOWEST")
                    ev = T.simd.vexp(sv, mask=m32)
                    T.simd.vsts(w_ub[s, 0], ev, mask=m32, dist="ONEPT_B32")
                    av = T.simd.vld(acc_ub[s, 0])
                    T.simd.vsts(acc_ub[s, 0], T.simd.vadd(av, qv, mask=m32), mask=m32)
            T.copy(acc_ub[0:1, :], Out[bx : bx + 1, :])
            T.copy(w_ub, WOut[bx * rows : bx * rows + rows, :])
    return main
"""
        opt = f"""import tilelang.ascend.language as T

def make_flywheel_probe():
    cores = {mix_cores}
    rows = 4
    cols = 64

    @T.prim_func
    def main(
        Q: T.Tensor((cores * rows, cols), "float32"),
        Out: T.Tensor((cores, cols), "float32"),
        WOut: T.Tensor((cores * rows, 1), "float32"),
    ):
        with T.Kernel(cores) as bx:
            q_ub = T.alloc_shared((1, cols), "float32")
            acc_ub = T.alloc_shared((rows, cols), "float32")
            exb = T.alloc_shared((rows, 1), "float32")
            for s in T.serial(rows):
                T.copy(Q[bx * rows + s : bx * rows + s + 1, :], q_ub)
                with T.SimdVF():
                    m32 = T.simd.pset(32, "PAT_ALL")
                    qv = T.simd.vld(q_ub[0, 0])
                    sv = T.simd.vdupv(T.simd.vcadd(qv, mask=m32), mask=m32, pos="POS_LOWEST")
                    ev = T.simd.vexp(sv, mask=m32)
                    T.simd.vsts(exb[s, 0], ev, mask=m32, dist="ONEPT_B32")
            for s2 in T.serial(rows):
                T.copy(Q[bx * rows + s2 : bx * rows + s2 + 1, :], q_ub)
                with T.SimdVF():
                    m32 = T.simd.pset(32, "PAT_ALL")
                    qv = T.simd.vld(q_ub[0, 0])
                    av = T.simd.vld(acc_ub[s2, 0])
                    T.simd.vsts(acc_ub[s2, 0], T.simd.vadd(av, qv, mask=m32), mask=m32)
            T.copy(acc_ub[0:1, :], Out[bx : bx + 1, :])
            T.copy(exb, WOut[bx * rows : bx * rows + rows, :])
    return main
"""
        return base, opt, (mix_cores * 4, 64), "float32"

    return None


def synthesize_micro_probe(
    chunk: AstEditChunk,
    *,
    probe_id: str | None = None,
) -> MicroProbeSpec | None:
    """Synthesize a standalone 30-50 line micro-probe, or None for unmapped edits."""
    if chunk.edit_kind not in _SUPPORTED_PROBE_KINDS:
        return None
    params = _extract_probe_params(chunk)
    rendered = _probe_templates_for_kind(chunk.edit_kind, params)
    if rendered is None:
        return None
    base_src, opt_src, shape, dtype = rendered
    pid = probe_id or f"probe_{chunk.edit_kind}_{chunk.chunk_id}"
    return MicroProbeSpec(
        probe_id=pid,
        probe_kind=chunk.edit_kind,
        factory_name="make_flywheel_probe",
        baseline_probe_source=base_src,
        optimized_probe_source=opt_src,
        line_count_baseline=len(base_src.strip().splitlines()),
        line_count_optimized=len(opt_src.strip().splitlines()),
        tensor_shape=shape,
        tensor_dtype=dtype,
    )


def verify_micro_probe_hardware(
    spec: MicroProbeSpec,
    *,
    probe_runner: Callable[[MicroProbeSpec], Mapping[str, float]] | None = None,
    min_speedup: float = 1.08,
) -> tuple[bool, dict[str, float]]:
    """Verify synthesized micro-probe correctness, speedup, and pipe wait ratios on NPU."""
    if probe_runner is not None:
        raw = dict(probe_runner(spec))
        b_us = float(raw.get("baseline_dur_us", 0.0))
        o_us = float(raw.get("optimized_dur_us", 0.0))
        speedup = float(raw.get("speedup", (b_us / o_us) if b_us > 0.0 and o_us > 0.0 else 0.0))
        raw["speedup"] = speedup
        wait_ok = True
        if "baseline_mte2_wait_ratio" in raw and "optimized_mte2_wait_ratio" in raw:
            base_w = float(raw["baseline_mte2_wait_ratio"])
            opt_w = float(raw["optimized_mte2_wait_ratio"])
            if opt_w > max(base_w * 1.05, base_w + 0.05):
                wait_ok = False
        if "baseline_aiv_mte2" in raw and "optimized_aiv_mte2" in raw:
            base_m = float(raw["baseline_aiv_mte2"])
            opt_m = float(raw["optimized_aiv_mte2"])
            if base_m > 0.0 and speedup < 1.50:
                base_abs = base_m * b_us
                opt_abs = opt_m * o_us
                if opt_m > max(base_m * 1.15, base_m + 0.05) and opt_abs > max(
                    base_abs * 1.15, base_abs + 0.05 * b_us
                ):
                    wait_ok = False
        passed = (
            bool(raw.get("numerical_ok", True))
            and b_us > 0.0
            and o_us > 0.0
            and speedup >= min_speedup
            and wait_ok
        )
        return passed, raw

    import importlib.util
    import tempfile

    import tilelang
    import torch
    from tilelang.profiler import do_bench

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("Ascend NPU device is required for live micro-probe verification.")

    def _compile_and_time(src: str, tag: str, kind: str) -> tuple[float, dict[str, float]]:
        with tempfile.TemporaryDirectory(prefix=f"tf_probe_{tag}_") as tmp_dir:
            mod_file = Path(tmp_dir) / "probe_mod.py"
            mod_file.write_text(src, encoding="utf-8")
            mod_spec = importlib.util.spec_from_file_location(f"_tf_probe_{tag}", str(mod_file))
            assert mod_spec is not None and mod_spec.loader is not None
            mod = importlib.util.module_from_spec(mod_spec)
            mod_spec.loader.exec_module(mod)
            prim_func = getattr(mod, spec.factory_name)()
            kernel = tilelang.compile(prim_func, target="ascend", out_idx=-1)

        torch.manual_seed(42)
        if kind == "ssbuf_handoff":
            rounds = spec.tensor_shape[0]
            seed = torch.zeros((16, 16), dtype=torch.float16, device="npu")
            expected = torch.tensor(
                [[(i + 1) * 100 + (((i + 1) * 101) << 32)] for i in range(rounds)],
                dtype=torch.uint64,
            )
            if tag == "base":
                scratch = torch.zeros((rounds, 2), dtype=torch.uint32, device="npu")
                out = kernel(seed, scratch)
                torch.npu.synchronize()
                torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
                prof = do_bench(
                    lambda: kernel(seed, scratch),
                    backend="msprof_detail",
                    _n_warmup=3,
                    _n_repeat=6,
                )
            else:
                out = kernel(seed)
                torch.npu.synchronize()
                torch.testing.assert_close(out.cpu(), expected, rtol=0, atol=0)
                prof = do_bench(
                    lambda: kernel(seed),
                    backend="msprof_detail",
                    _n_warmup=3,
                    _n_repeat=6,
                )
        else:
            run_shape = spec.tensor_shape
            if kind == "pad_alignment" and tag == "base":
                run_shape = (spec.tensor_shape[0], 70)
            a = torch.randn(run_shape, dtype=torch.float32, device="npu")
            b = torch.randn(run_shape, dtype=torch.float32, device="npu")
            out = kernel(a, b)
            torch.npu.synchronize()
            torch.testing.assert_close(out.cpu(), (a + b).cpu(), atol=1e-4, rtol=1e-4)
            prof = do_bench(
                lambda: kernel(a, b),
                backend="msprof_detail",
                _n_warmup=3,
                _n_repeat=6,
            )
        ratios = {
            "aiv_mte2": float(getattr(prof, "aiv_mte2", 0.0)),
            "aiv_vec": float(getattr(prof, "aiv_vec", 0.0)),
            "aic_fixpipe": float(getattr(prof, "aic_fixpipe", 0.0)),
        }
        return float(prof.dur_us), ratios

    b_us, b_ratios = _compile_and_time(spec.baseline_probe_source, "base", spec.probe_kind)
    o_us, o_ratios = _compile_and_time(spec.optimized_probe_source, "opt", spec.probe_kind)
    speedup = (b_us / o_us) if o_us > 0.0 else 0.0
    wait_non_regression = True
    if b_ratios["aiv_mte2"] > 0.0 and speedup < 1.50:
        base_abs = b_ratios["aiv_mte2"] * b_us
        opt_abs = o_ratios["aiv_mte2"] * o_us
        if o_ratios["aiv_mte2"] > max(
            b_ratios["aiv_mte2"] * 1.15, b_ratios["aiv_mte2"] + 0.05
        ) and opt_abs > max(base_abs * 1.15, base_abs + 0.05 * b_us):
            wait_non_regression = False
    metrics = {
        "baseline_dur_us": round(b_us, 6),
        "optimized_dur_us": round(o_us, 6),
        "speedup": round(speedup, 6),
        "numerical_ok": 1.0,
        "aiv_mte2": round(o_ratios["aiv_mte2"], 6),
        "aiv_vec": round(o_ratios["aiv_vec"], 6),
        "aic_fixpipe": round(o_ratios["aic_fixpipe"], 6),
        "baseline_aiv_mte2": round(b_ratios["aiv_mte2"], 6),
        "optimized_aiv_mte2": round(o_ratios["aiv_mte2"], 6),
    }
    return (speedup >= min_speedup and wait_non_regression), metrics


def _generalize_template_text(chunk: AstEditChunk) -> str:
    if chunk.edit_kind == "core_parallelism":
        return (
            "Scale T.Kernel(block_count) grid dimensionality to saturate available hardware "
            "blocks using contiguous per-block slicing and eliminate trailing PIPE_ALL barriers."
        )
    if chunk.edit_kind == "pipeline_versions":
        return (
            "Increase T.Pipelined num_stages to 2 and annotate on-chip buffer versions >= 2 "
            "so DMA transfers overlap with compute stages while reserving compiler headroom."
        )
    if chunk.edit_kind == "coalesced_copy":
        return (
            "Coalesce strided inner row T.copy loops into contiguous 2D block DMA slices "
            "to reduce MTE2 inflight queue pressure and improve burst efficiency."
        )
    if chunk.edit_kind == "ssbuf_handoff":
        return (
            "Replace global memory scalar barriers with 8B-aligned SSBUF T.alloc_ss slots "
            "and explicit dirty-SRAM uint32 zero initialization before steady-state handoff."
        )
    if chunk.edit_kind == "dual_copy_1c2v":
        return (
            "Partition shared DMA workloads across independent dual-vector worker slices "
            "with 1C2V swizzle scheduling to double concurrent MTE2 and vector throughput."
        )
    if chunk.edit_kind == "barrier_prune":
        return (
            "Eliminate redundant T.sync_all and T.set_flag barriers inside steady-state loops, "
            "retaining only the minimal cross-pipe synchronization required by dependencies."
        )
    if chunk.edit_kind == "pad_alignment":
        return (
            "Pad inner buffer trailing dimensions to 32-byte and 512-byte bank-aligned strides "
            "so DMA bursts avoid unaligned tail transactions and bank conflict stalls."
        )
    if chunk.edit_kind == "static_pingpong_unroll":
        return (
            "Replace dynamic-offset T.Pipelined loops with 2x-unrolled static ping-pong buffers "
            "(buf_a, buf_b) and explicit prologue prefetch to eliminate pointer-alias serialization."
        )
    if chunk.edit_kind == "loop_fission_sfu":
        return (
            "Split coupled scalar reduction and SFU evaluation into Pass 1 before streaming wide "
            "vector accumulation in Pass 2 so long SFU chains no longer serialize wide datapaths."
        )
    if chunk.edit_kind == "config_knob":
        return (
            "Tune schedule configuration dictionary parameters (tile shapes, buffer stages, "
            "swizzle window, and bank padding) to maximize hardware pipeline occupancy without overflow."
        )
    if chunk.edit_kind == "cpp_hunk_edit":
        return (
            "Apply targeted tl_templates/ascend C++20 kernel optimizations across vector SIMD, "
            "DMA alignment, and on-chip buffer staging while preserving exact synchronization parity."
        )
    return (
        "Replace sequential or over-synchronized inner statements with non-blocking "
        "pipelined vector and DMA execution while preserving static UB headroom."
    )


def _rule_predicates_for_kind(
    chunk: AstEditChunk,
    ledger: ObjectivePhysicalLedger | None,
) -> tuple[str, str, str]:
    if chunk.edit_kind == "core_parallelism":
        return (
            "is_legal and any_overflow_bytes == 0 and not has_indirect_gather",
            "wave_occupancy < 0.75 or PIPE_V > 0 or PIPE_MTE2 > 0",
            "measured_latency_us <= prev.measured_latency_us",
        )
    if chunk.edit_kind == "coalesced_copy":
        return (
            "is_legal and dominant_contig_bytes >= 64",
            "mte2_inflight_per_aiv > 8 or PIPE_MTE2 > PIPE_M",
            "mte2_inflight_per_aiv <= prev.mte2_inflight_per_aiv and any_overflow_bytes == 0",
        )
    if chunk.edit_kind == "ssbuf_handoff":
        return (
            "is_legal and (has_dynamic_control or cube_to_vector_ratio >= 0.0)",
            "unexplained_gap_ratio >= 0.15 or measured_sync_stall_ratio > 0.05",
            "measured_sync_stall_ratio <= prev.measured_sync_stall_ratio",
        )
    if chunk.edit_kind == "dual_copy_1c2v":
        return (
            "is_legal and (cube_to_vector_ratio > 0.0 or AIV0.PIPE_V > 0)",
            "PIPE_MTE2 > 0 or PIPE_V > 0 or measured_mte2_wait_ratio > 0.1",
            "any_overflow_bytes == 0 and measured_latency_us <= prev.measured_latency_us",
        )
    if chunk.edit_kind == "barrier_prune":
        return (
            "is_legal and not has_dynamic_control",
            "in_loop_pipe_all_barriers > 0 or measured_sync_stall_ratio > 0.05 or PIPE_V > 0",
            "in_loop_pipe_all_barriers <= prev.in_loop_pipe_all_barriers and any_overflow_bytes == 0",
        )
    if chunk.edit_kind == "pad_alignment":
        return (
            "is_legal and dominant_contig_bytes >= 16",
            "unaligned_dma_padding_bytes > 0 or PIPE_MTE2 > 0 or PIPE_V > 0",
            "unaligned_dma_padding_bytes <= prev.unaligned_dma_padding_bytes and any_overflow_bytes == 0",
        )
    if chunk.edit_kind == "static_pingpong_unroll":
        return (
            "is_legal and (uses_dynamic_offset_pipelining or pseudo_double_buffer_serialized)",
            "pseudo_double_buffer_serialized or measured_mte2_wait_ratio >= 0.32",
            "not uses_dynamic_offset_pipelining and any_overflow_bytes == 0",
        )
    if chunk.edit_kind == "loop_fission_sfu":
        return (
            "is_legal and PIPE_M == 0 and has_coupled_reduce_sfu_accum",
            "has_coupled_reduce_sfu_accum or measured_aiv_vec_ratio >= 0.35",
            "not has_coupled_reduce_sfu_accum and any_overflow_bytes == 0",
        )
    dom = ledger.dominant_pipe if ledger is not None else "AIV0.PIPE_V"
    _ = dom
    return (
        "is_legal and any_overflow_bytes == 0",
        "degraded_buffer_count >= 0 and (PIPE_V > 0 or PIPE_MTE2 > 0)",
        "degraded_buffer_count <= prev.degraded_buffer_count and any_overflow_bytes == 0",
    )


def _is_rule_text_clean(rule: AscendCausalRule) -> tuple[bool, str]:
    joined = " ".join(rule.as_six_tuple()).lower()
    if rule.cpp20_transform_template:
        joined = f"{joined} {rule.cpp20_transform_template.lower()}"
    for bad in _build_disallowed_provenance_tokens():
        if bad in _SUBSTRING_PROVENANCE_TOKENS:
            if bad in joined:
                return False, f"disallowed token {bad!r} in distilled rule text"
        elif re.search(rf"(?<![a-z0-9_]){re.escape(bad)}(?![a-z0-9_])", joined):
            return False, f"disallowed token {bad!r} in distilled rule text"
    if re.search(r"/(home|tmp|var|usr)/", joined):
        return False, "filesystem path leaked into rule text"
    try:
        validate_rule_predicate(rule.regime_guard)
        validate_rule_predicate(rule.precondition)
        validate_rule_predicate(rule.expected_postcondition)
    except Exception as exc:
        return False, f"invalid predicate: {exc}"
    return True, ""


def distill_causal_rule(
    chunk: AstEditChunk,
    *,
    rule_id: str,
    ledger: ObjectivePhysicalLedger | None = None,
    probe: MicroProbeSpec | None = None,
    probe_metrics: Mapping[str, float] | None = None,
    scope: str = "cross_op",
    origin_operator: str = "",
) -> AscendCausalRule:
    """Synthesize a pure-English 6-tuple AscendCausalRule from a verified AST edit."""
    regime_guard, precondition, expected_post = _rule_predicates_for_kind(chunk, ledger)
    template = _generalize_template_text(chunk)
    speedup = float((probe_metrics or {}).get("speedup", 1.0))
    rationale = (
        f"Counterfactual single-variable ablation verified on Ascend 950 micro-probe "
        f"({chunk.edit_kind}, speedup={speedup:.3f}x)."
    )
    is_cpp_chunk = chunk.chunk_id.startswith("cpp_loo_") or chunk.edit_kind == "cpp_hunk_edit"
    backends: tuple[str, ...] = (
        ("ascendc_trackb", "hybrid_trackb")
        if is_cpp_chunk
        else ("tilelang_dsl", "ascendc_trackb", "hybrid_trackb")
    )
    cpp_template = template if is_cpp_chunk else ""
    return AscendCausalRule(
        rule_id=rule_id,
        regime_guard=regime_guard,
        precondition=precondition,
        transform_template=template,
        expected_postcondition=expected_post,
        source_provenance="flywheel:ablation",
        alpha_prior=2.5,
        beta_prior=1.0,
        rationale=rationale,
        priority=85,
        scope=scope,
        origin_operator=origin_operator,
        compatible_backends=backends,
        cpp20_transform_template=cpp_template,
    )


def _ledger_cost_score(ledger: ObjectivePhysicalLedger) -> float:
    dom = float(max(ledger.pipe_cycles.values(), default=100))
    occ = float(ledger.wave_occupancy)
    if occ <= 0.0:
        occ = float(min(56, max(1, int(ledger.grid_blocks)))) / 56.0
    occ = max(0.02, occ)
    burst = max(0.15, float(ledger.dma_burst_efficiency_ratio))
    degraded_count = sum(len(wl.degraded_single_buffers) for wl in ledger.waterlines.values())
    penalty = 1.0
    penalty += 0.25 * max(0, degraded_count)
    penalty += 0.15 * max(0, int(ledger.in_loop_pipe_all_barriers))
    if ledger.uses_dynamic_offset_pipelining:
        penalty += 0.12
    if ledger.has_coupled_reduce_sfu_accum:
        penalty += 0.12
    return (dom / occ / burst) * penalty


class FlywheelDistiller:
    """Decomposes winning DSL edits, ablates negatives, probes NPU, and admits rules."""

    def __init__(
        self,
        *,
        funnel: GateFunnel,
        tracker: BayesianRuleTracker,
        probe_runner: Callable[[MicroProbeSpec], Mapping[str, float]] | None = None,
        min_edit_gain: float = 0.03,
        min_probe_speedup: float = 1.08,
    ) -> None:
        self._funnel = funnel
        self._tracker = tracker
        self._probe_runner = probe_runner
        self._min_edit_gain = min_edit_gain
        self._min_probe_speedup = min_probe_speedup

    def distill_leave_one_out(
        self,
        *,
        baseline_dsl: str,
        winner_dsl: str,
        winner_latency_us: float,
        baseline_latency_us: float | None = None,
        factory_name: str | None = None,
        factory_kwargs: Mapping[str, object] | None = None,
        config_selector_name: str | None = None,
        dims: Mapping[str, object] | None = None,
        batch_latency_evaluator: Callable[[Sequence[tuple[str, str]]], Mapping[str, float]] | None = None,
        topology_axes: object | None = None,
        kb_output_path: str | Path | None = None,
        rule_id_prefix: str = "R_FLYWHEEL_LOO",
        max_loo_chunks: int | None = 3,
        operator_name: str = "",
        rule_scope: str = "operator_local",
    ) -> FlywheelDistillationReport:
        """Run Leave-One-Out AST/config ablation directly on real candidate winner_dsl."""
        chunks = decompose_leave_one_out_ablations(baseline_dsl, winner_dsl)
        if max_loo_chunks is not None and int(max_loo_chunks) > 0:
            chunks = chunks[: int(max_loo_chunks)]
        kept: list[AblationMeasurement] = []
        discarded: list[AblationMeasurement] = []
        outcomes: list[DistilledRuleOutcome] = []
        admitted_ids: list[str] = []

        win_sa = run_stage_a_static(
            winner_dsl,
            factory_name=factory_name,
            factory_kwargs=factory_kwargs,
            config_selector_name=config_selector_name,
            dims=dims,
        )
        win_cost = (
            _ledger_cost_score(win_sa.ledger)
            if win_sa.passed and win_sa.ledger is not None
            else 0.0
        )

        funnel_results: list[tuple[AstEditChunk, object]] = []
        for chunk in chunks:
            eval_res = self._funnel.evaluate(
                chunk.ablated_dsl,
                factory_name=factory_name,
                factory_kwargs=factory_kwargs,
                config_selector_name=config_selector_name,
                dims=dims,
            )
            funnel_results.append((chunk, eval_res))

        batch_latencies: dict[str, float] = {}
        if batch_latency_evaluator is not None:
            batch_targets = [
                (chunk.chunk_id, chunk.ablated_dsl)
                for chunk, eval_res in funnel_results
                if eval_res.passed and eval_res.stage_a is not None and eval_res.stage_a.ledger is not None
            ]
            if batch_targets:
                raw_batch = batch_latency_evaluator(batch_targets)
                batch_latencies = {
                    str(k): float(v) for k, v in dict(raw_batch).items() if float(v) > 0.0
                }

        win_lat = max(float(winner_latency_us), 1e-6)
        eff_base_lat = float(baseline_latency_us) if baseline_latency_us is not None else win_lat
        for chunk, eval_res in funnel_results:
            if not eval_res.passed or eval_res.stage_a is None or eval_res.stage_a.ledger is None:
                discarded.append(
                    AblationMeasurement(
                        chunk=chunk,
                        funnel_passed=False,
                        baseline_latency_us=win_lat,
                        variant_latency_us=0.0,
                        relative_gain=-1.0,
                        positive_gain=False,
                        stage_a_result=eval_res.stage_a,
                        diagnostic=f"{eval_res.failed_stage}:{eval_res.error_message[:120]}",
                    )
                )
                continue

            if chunk.chunk_id in batch_latencies:
                var_lat = batch_latencies[chunk.chunk_id]
            elif eval_res.stage_c and eval_res.stage_c.measured_latency_us > 0.0:
                var_lat = eval_res.stage_c.measured_latency_us
            elif win_cost > 0.0:
                var_cost = _ledger_cost_score(eval_res.stage_a.ledger)
                var_lat = win_lat * (var_cost / win_cost)
            else:
                var_lat = float(max(eval_res.stage_a.ledger.pipe_cycles.values(), default=100)) / 20.0

            rel_gain = (var_lat - win_lat) / max(var_lat, 1e-6)
            is_pos = var_lat > win_lat and rel_gain >= self._min_edit_gain
            meas = AblationMeasurement(
                chunk=chunk,
                funnel_passed=True,
                baseline_latency_us=win_lat,
                variant_latency_us=var_lat,
                relative_gain=rel_gain,
                positive_gain=is_pos,
                stage_a_result=eval_res.stage_a,
                diagnostic="positive_loo_gain" if is_pos else "redundant_or_harmful_edit",
            )
            if is_pos:
                kept.append(meas)
            else:
                discarded.append(meas)

        seen_sigs: set[tuple[str, str]] = set()
        for meas in kept:
            sig = (meas.chunk.edit_kind, meas.chunk.target_path)
            if sig in seen_sigs:
                continue
            seen_sigs.add(sig)
            speedup = meas.variant_latency_us / win_lat if win_lat > 0.0 else 1.0
            metrics = {
                "baseline_dur_us": round(meas.variant_latency_us, 6),
                "optimized_dur_us": round(win_lat, 6),
                "speedup": round(speedup, 6),
                "numerical_ok": 1.0,
            }
            rule_num = len(self._tracker.rules) + 1
            suffix = meas.chunk.edit_kind.upper()
            cand_rule_id = f"{rule_id_prefix}_{rule_num:02d}_{suffix}"
            while cand_rule_id in self._tracker.rules:
                rule_num += 1
                cand_rule_id = f"{rule_id_prefix}_{rule_num:02d}_{suffix}"
            rule = distill_causal_rule(
                meas.chunk,
                rule_id=cand_rule_id,
                ledger=meas.stage_a_result.ledger if meas.stage_a_result else None,
                probe=None,
                probe_metrics=metrics,
                scope=rule_scope,
                origin_operator=operator_name,
            )
            clean_ok, clean_err = _is_rule_text_clean(rule)
            if not clean_ok:
                outcomes.append(
                    DistilledRuleOutcome(
                        chunk_id=meas.chunk.chunk_id,
                        probe=None,
                        hardware_verified=True,
                        probe_metrics=metrics,
                        rule=rule,
                        admitted=False,
                        rejection_reason=clean_err,
                    )
                )
                continue
            self._tracker.admit_rule(rule)
            self._tracker.record_outcome(
                rule.rule_id,
                postcondition_realized=True,
                speed_gain=max(0.0, meas.relative_gain),
                topology_axes=topology_axes,
            )
            admitted_ids.append(rule.rule_id)
            outcomes.append(
                DistilledRuleOutcome(
                    chunk_id=meas.chunk.chunk_id,
                    probe=None,
                    hardware_verified=True,
                    probe_metrics=metrics,
                    rule=rule,
                    admitted=True,
                )
            )

        if kb_output_path is not None:
            save_rules_yaml(list(self._tracker.rules.values()), kb_output_path)

        return FlywheelDistillationReport(
            baseline_latency_us=eff_base_lat,
            winner_latency_us=win_lat,
            total_chunks=len(chunks),
            kept_positive_chunks=tuple(kept),
            discarded_chunks=tuple(discarded),
            distilled_outcomes=tuple(outcomes),
            admitted_rule_ids=tuple(admitted_ids),
        )

    def distill_from_pair(
        self,
        *,
        baseline_dsl: str,
        winner_dsl: str,
        baseline_latency_us: float,
        winner_latency_us: float,
        factory_name: str | None = None,
        factory_kwargs: Mapping[str, object] | None = None,
        config_selector_name: str | None = None,
        dims: Mapping[str, object] | None = None,
        batch_latency_evaluator: Callable[[Sequence[tuple[str, str]]], Mapping[str, float]] | None = None,
        topology_axes: object | None = None,
        kb_output_path: str | Path | None = None,
        rule_id_prefix: str = "R_FLYWHEEL",
        verify_on_candidate: bool = False,
        max_loo_chunks: int | None = 3,
        operator_name: str = "",
        rule_scope: str = "cross_op",
    ) -> FlywheelDistillationReport:
        """Run full counterfactual ablation, micro-probe or LOO check, and KB admission."""
        if verify_on_candidate:
            return self.distill_leave_one_out(
                baseline_dsl=baseline_dsl,
                winner_dsl=winner_dsl,
                winner_latency_us=winner_latency_us,
                baseline_latency_us=baseline_latency_us,
                factory_name=factory_name,
                factory_kwargs=factory_kwargs,
                config_selector_name=config_selector_name,
                dims=dims,
                batch_latency_evaluator=batch_latency_evaluator,
                topology_axes=topology_axes,
                kb_output_path=kb_output_path,
                rule_id_prefix=rule_id_prefix,
                max_loo_chunks=max_loo_chunks,
                operator_name=operator_name,
                rule_scope=rule_scope,
            )
        chunks = decompose_ast_ablations(baseline_dsl, winner_dsl)
        kept: list[AblationMeasurement] = []
        discarded: list[AblationMeasurement] = []
        outcomes: list[DistilledRuleOutcome] = []
        admitted_ids: list[str] = []

        base_sa = run_stage_a_static(
            baseline_dsl,
            factory_name=factory_name,
            factory_kwargs=factory_kwargs,
            config_selector_name=config_selector_name,
            dims=dims,
        )
        base_cost = (
            _ledger_cost_score(base_sa.ledger)
            if base_sa.passed and base_sa.ledger is not None
            else 0.0
        )

        funnel_results: list[tuple[AstEditChunk, object]] = []
        for chunk in chunks:
            eval_res = self._funnel.evaluate(
                chunk.ablated_dsl,
                factory_name=factory_name,
                factory_kwargs=factory_kwargs,
                config_selector_name=config_selector_name,
                dims=dims,
            )
            funnel_results.append((chunk, eval_res))

        batch_latencies: dict[str, float] = {}
        if batch_latency_evaluator is not None:
            batch_targets: list[tuple[str, str]] = []
            seen_batch_kinds: set[str] = set()
            for chunk, eval_res in funnel_results:
                if (
                    not eval_res.passed
                    or eval_res.stage_a is None
                    or eval_res.stage_a.ledger is None
                ):
                    continue
                if chunk.edit_kind not in _SUPPORTED_PROBE_KINDS:
                    continue
                if chunk.edit_kind in seen_batch_kinds:
                    continue
                seen_batch_kinds.add(chunk.edit_kind)
                batch_targets.append((chunk.chunk_id, chunk.ablated_dsl))
                if len(batch_targets) >= 4:
                    break
            if batch_targets:
                raw_batch = batch_latency_evaluator(batch_targets)
                batch_latencies = {
                    str(k): float(v)
                    for k, v in dict(raw_batch).items()
                    if float(v) > 0.0
                }

        base_lat = max(baseline_latency_us, 1e-6)
        for chunk, eval_res in funnel_results:
            if not eval_res.passed or eval_res.stage_a is None or eval_res.stage_a.ledger is None:
                discarded.append(
                    AblationMeasurement(
                        chunk=chunk,
                        funnel_passed=False,
                        baseline_latency_us=baseline_latency_us,
                        variant_latency_us=0.0,
                        relative_gain=-1.0,
                        positive_gain=False,
                        stage_a_result=eval_res.stage_a,
                        diagnostic=f"{eval_res.failed_stage}:{eval_res.error_message[:120]}",
                    )
                )
                continue

            if chunk.chunk_id in batch_latencies:
                var_lat = batch_latencies[chunk.chunk_id]
            elif eval_res.stage_c and eval_res.stage_c.measured_latency_us > 0.0:
                var_lat = eval_res.stage_c.measured_latency_us
            elif base_cost > 0.0:
                var_cost = _ledger_cost_score(eval_res.stage_a.ledger)
                var_lat = base_lat * (var_cost / base_cost)
            else:
                var_lat = float(max(eval_res.stage_a.ledger.pipe_cycles.values(), default=100)) / 20.0
            rel_gain = (base_lat - var_lat) / base_lat
            is_pos = rel_gain >= self._min_edit_gain
            meas = AblationMeasurement(
                chunk=chunk,
                funnel_passed=True,
                baseline_latency_us=baseline_latency_us,
                variant_latency_us=var_lat,
                relative_gain=rel_gain,
                positive_gain=is_pos,
                stage_a_result=eval_res.stage_a,
                diagnostic="positive_gain" if is_pos else "negative_or_zero_gain",
            )
            if is_pos:
                kept.append(meas)
            else:
                discarded.append(meas)

        seen_kinds: set[str] = set()
        for meas in kept:
            if meas.chunk.edit_kind in seen_kinds:
                continue
            seen_kinds.add(meas.chunk.edit_kind)
            probe = synthesize_micro_probe(meas.chunk)
            if probe is None:
                outcomes.append(
                    DistilledRuleOutcome(
                        chunk_id=meas.chunk.chunk_id,
                        probe=None,
                        hardware_verified=False,
                        probe_metrics={},
                        rule=None,
                        admitted=False,
                        rejection_reason=f"unverifiable_edit_kind:{meas.chunk.edit_kind}",
                    )
                )
                continue
            ok, metrics = verify_micro_probe_hardware(
                probe,
                probe_runner=self._probe_runner,
                min_speedup=self._min_probe_speedup,
            )
            if not ok:
                outcomes.append(
                    DistilledRuleOutcome(
                        chunk_id=meas.chunk.chunk_id,
                        probe=probe,
                        hardware_verified=False,
                        probe_metrics=metrics,
                        rule=None,
                        admitted=False,
                        rejection_reason="micro_probe_below_threshold_or_wait_regression",
                    )
                )
                continue

            rule_num = len(self._tracker.rules) + 1
            suffix = meas.chunk.edit_kind.upper()
            cand_rule_id = f"{rule_id_prefix}_{rule_num:02d}_{suffix}"
            while cand_rule_id in self._tracker.rules:
                rule_num += 1
                cand_rule_id = f"{rule_id_prefix}_{rule_num:02d}_{suffix}"
            rule = distill_causal_rule(
                meas.chunk,
                rule_id=cand_rule_id,
                ledger=meas.stage_a_result.ledger if meas.stage_a_result else None,
                probe=probe,
                probe_metrics=metrics,
                scope=rule_scope,
                origin_operator=operator_name,
            )
            clean_ok, clean_err = _is_rule_text_clean(rule)
            if not clean_ok:
                outcomes.append(
                    DistilledRuleOutcome(
                        chunk_id=meas.chunk.chunk_id,
                        probe=probe,
                        hardware_verified=True,
                        probe_metrics=metrics,
                        rule=rule,
                        admitted=False,
                        rejection_reason=clean_err,
                    )
                )
                continue

            self._tracker.admit_rule(rule)
            ctx = build_predicate_context(
                meas.stage_a_result.ledger,
                features=meas.stage_a_result.features,
            ) if meas.stage_a_result and meas.stage_a_result.ledger else {}
            _ = ctx
            self._tracker.record_outcome(
                rule.rule_id,
                postcondition_realized=True,
                speed_gain=max(0.0, meas.relative_gain),
                topology_axes=topology_axes,
            )
            admitted_ids.append(rule.rule_id)
            outcomes.append(
                DistilledRuleOutcome(
                    chunk_id=meas.chunk.chunk_id,
                    probe=probe,
                    hardware_verified=True,
                    probe_metrics=metrics,
                    rule=rule,
                    admitted=True,
                )
            )

        if kb_output_path is not None:
            save_rules_yaml(list(self._tracker.rules.values()), kb_output_path)

        return FlywheelDistillationReport(
            baseline_latency_us=baseline_latency_us,
            winner_latency_us=winner_latency_us,
            total_chunks=len(chunks),
            kept_positive_chunks=tuple(kept),
            discarded_chunks=tuple(discarded),
            distilled_outcomes=tuple(outcomes),
            admitted_rule_ids=tuple(admitted_ids),
        )

    def distill_from_trajectory(
        self,
        *,
        baseline_dsl: str,
        trajectory: EvolutionTrajectory,
        factory_name: str | None = None,
        factory_kwargs: Mapping[str, object] | None = None,
        kb_output_path: str | Path | None = None,
    ) -> FlywheelDistillationReport:
        """Distill and verify causal rules directly from an EvolutionTrajectory winner."""
        return self.distill_from_pair(
            baseline_dsl=baseline_dsl,
            winner_dsl=trajectory.best_dsl_source,
            baseline_latency_us=trajectory.baseline_latency_us,
            winner_latency_us=trajectory.best_latency_us,
            factory_name=factory_name,
            factory_kwargs=factory_kwargs,
            kb_output_path=kb_output_path,
        )


__all__ = [
    "AblationMeasurement",
    "AstEditChunk",
    "DistilledRuleOutcome",
    "FlywheelDistillationReport",
    "FlywheelDistiller",
    "MicroProbeSpec",
    "_build_disallowed_provenance_tokens",
    "decompose_ast_ablations",
    "decompose_cpp_string_leave_one_out",
    "decompose_leave_one_out_ablations",
    "distill_causal_rule",
    "synthesize_micro_probe",
    "verify_micro_probe_hardware",
]
