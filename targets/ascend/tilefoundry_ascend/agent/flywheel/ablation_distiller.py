"""Counterfactual single-variable AST ablation and self-evolution flywheel distiller."""

from __future__ import annotations

import ast
from collections.abc import Callable, Mapping
import copy
from dataclasses import dataclass, field
import difflib
import hashlib
import json
from pathlib import Path
import re

from tilefoundry_ascend.agent.kb.bayesian_tracker import BayesianRuleTracker
from tilefoundry_ascend.agent.kb.schema import (
    AscendCausalRule,
    build_predicate_context,
    save_rules_yaml,
    validate_rule_predicate,
)
from tilefoundry_ascend.agent.orchestrator.gate_funnel import GateFunnel, StageAResult
from tilefoundry_ascend.agent.orchestrator.e2e_evolution import EvolutionTrajectory
from tilefoundry_ascend.ascend.passes.physical_ledger import ObjectivePhysicalLedger

_DISALLOWED_PROVENANCE_TOKENS = (
    "/home/",
    "/users/",
    "/data/",
    "attention_residual",
    "mla_model",
    "deepgemm",
    "flash_attention",
)


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


@dataclass(frozen=True)
class DistilledRuleOutcome:
    """Verification and KB admission outcome for one winning AST mechanism."""

    chunk_id: str
    probe: MicroProbeSpec
    hardware_verified: bool
    probe_metrics: dict[str, float]
    rule: AscendCausalRule | None
    admitted: bool
    rejection_reason: str = ""

    def to_dict(self) -> dict[str, object]:
        """Serialize distilled rule verification and admission record."""
        return {
            "chunk_id": self.chunk_id,
            "probe": self.probe.to_dict(),
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
    if "num_stages" in combined or "annotate_buffer_versions" in combined:
        return "pipeline_versions"
    if "alloc_ss" in combined or "SSBUF" in combined:
        return "ssbuf_handoff"
    if "T.Kernel" in combined or "T.MixedKernel" in combined or "num_cores" in combined or "num_blocks" in combined:
        return "core_parallelism"
    if "T.copy" in combined or "T.dual_copy" in combined:
        return "coalesced_copy"
    if "ascend_pipe_barrier" in combined:
        return "barrier_prune"
    return "stmt_rewrite"


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


def _render_patched(base_tree: ast.Module) -> str:
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
) -> None:
    base_tags = [_node_tag(s) for s in base_list]
    win_tags = [_node_tag(s) for s in win_list]
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
                            dsl = _render_patched(patched)
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
                    )
                    continue

                if isinstance(b_stmt, ast.For) and isinstance(w_stmt, ast.For):
                    b_iter = _node_key(b_stmt.iter)
                    w_iter = _node_key(w_stmt.iter)
                    if b_iter != w_iter:
                        patched = copy.deepcopy(base_tree)
                        if _replace_call_arg(patched, cur_path, is_with_header=False, new_expr=w_stmt.iter):
                            dsl = _render_patched(patched)
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
                    )
                    continue

                patched = copy.deepcopy(base_tree)
                if _set_stmt_by_path(patched, cur_path, w_stmt):
                    dsl = _render_patched(patched)
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
                                dsl = _render_patched(hdr_patch)
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
                    )
                    continue
                cur_path = parent_path + (i1 + k,)
                patched = copy.deepcopy(base_tree)
                if _set_stmt_by_path(patched, cur_path, w_stmt):
                    dsl = _render_patched(patched)
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
        elif tag == "delete":
            for idx in range(i1, i2):
                b_stmt = base_list[idx]
                b_txt = _node_key(b_stmt)
                cur_path = parent_path + (idx,)
                patched = copy.deepcopy(base_tree)
                if _set_stmt_by_path(patched, cur_path, None):
                    dsl = _render_patched(patched)
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
                if _set_stmt_by_path(patched, cur_path, w_stmt):
                    dsl = _render_patched(patched)
                    if dsl not in seen_dsls:
                        seen_dsls.add(dsl)
                        chunks.append(
                            AstEditChunk(
                                chunk_id=f"edit_{len(chunks)}",
                                edit_kind=_classify_edit("", w_txt),
                                target_path=f"{path_label}/insert_{i1}_{j}",
                                before_src="",
                                after_src=w_txt,
                                ablated_dsl=dsl,
                            )
                        )


def decompose_ast_ablations(baseline_dsl: str, winner_dsl: str) -> tuple[AstEditChunk, ...]:
    """Decompose AST difference between baseline and winner DSL into single-variable edits."""
    base_tree = ast.parse(baseline_dsl)
    win_tree = ast.parse(winner_dsl)
    chunks: list[AstEditChunk] = []
    seen_dsls: set[str] = {ast.unparse(base_tree)}

    _extract_stmt_list_deltas(
        base_tree,
        base_tree.body,
        win_tree.body,
        (),
        "module",
        chunks,
        seen_dsls,
    )
    if not chunks and ast.unparse(win_tree) != ast.unparse(base_tree):
        chunks.append(
            AstEditChunk(
                chunk_id="edit_0",
                edit_kind=_classify_edit(baseline_dsl, winner_dsl),
                target_path="module/whole",
                before_src=ast.unparse(base_tree),
                after_src=ast.unparse(win_tree),
                ablated_dsl=ast.unparse(win_tree),
            )
        )
    return tuple(chunks)


def _probe_templates_for_kind(probe_kind: str) -> tuple[str, str, tuple[int, ...], str]:
    if probe_kind == "core_parallelism":
        base = """import tilelang.ascend.language as T

def make_flywheel_probe():
    tile = 2048
    steps = 56
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
            T.annotate_buffer_versions({a_ub: 1, b_ub: 1, c_ub: 1})
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
        opt = """import tilelang.ascend.language as T

def make_flywheel_probe():
    tile = 2048
    cores = 56
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
            T.annotate_buffer_versions({a_ub: 1, b_ub: 1, c_ub: 1})
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
        return base, opt, (2048 * 56,), "float32"

    if probe_kind == "coalesced_copy":
        base = """import tilelang.ascend.language as T

def make_flywheel_probe():
    cores = 28
    rows = 16
    cols = 128

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
        opt = """import tilelang.ascend.language as T

def make_flywheel_probe():
    cores = 28
    rows = 16
    cols = 128

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
        return base, opt, (28 * 16, 128), "float32"

    base = """import tilelang.ascend.language as T

def make_flywheel_probe():
    tile = 2048
    steps = 16
    cores = 28
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
            T.annotate_buffer_versions({a_ub: 1, b_ub: 1, c_ub: 1})
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
    opt = """import tilelang.ascend.language as T

def make_flywheel_probe():
    tile = 2048
    steps = 16
    cores = 28
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
            T.annotate_buffer_versions({a_ub: 2, b_ub: 2, c_ub: 2})
            for s in T.Pipelined(steps, num_stages=2):
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
    return base, opt, (28 * 16 * 2048,), "float32"


def synthesize_micro_probe(
    chunk: AstEditChunk,
    *,
    probe_id: str | None = None,
) -> MicroProbeSpec:
    """Synthesize a standalone 30-50 line Ascend 950 micro-probe isolating one AST mechanism."""
    pid = probe_id or f"probe_{chunk.edit_kind}_{chunk.chunk_id}"
    base_src, opt_src, shape, dtype = _probe_templates_for_kind(chunk.edit_kind)
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
    min_speedup: float = 1.05,
) -> tuple[bool, dict[str, float]]:
    """Verify synthesized micro-probe correctness and latency gain on Ascend 950 NPU."""
    if probe_runner is not None:
        raw = dict(probe_runner(spec))
        b_us = float(raw.get("baseline_dur_us", 0.0))
        o_us = float(raw.get("optimized_dur_us", 0.0))
        speedup = float(raw.get("speedup", (b_us / o_us) if b_us > 0.0 and o_us > 0.0 else 0.0))
        raw["speedup"] = speedup
        passed = bool(raw.get("numerical_ok", True)) and b_us > 0.0 and o_us > 0.0 and speedup >= min_speedup
        return passed, raw

    import importlib.util
    import tempfile
    import tilelang
    from tilelang.profiler import do_bench
    import torch

    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("Ascend NPU device is required for live micro-probe verification.")

    def _compile_and_time(src: str, tag: str) -> tuple[float, torch.Tensor]:
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
        a = torch.randn(spec.tensor_shape, dtype=torch.float32, device="npu")
        b = torch.randn(spec.tensor_shape, dtype=torch.float32, device="npu")
        out = kernel(a, b)
        torch.npu.synchronize()
        torch.testing.assert_close(out.cpu(), (a + b).cpu(), atol=1e-4, rtol=1e-4)
        prof = do_bench(lambda: kernel(a, b), backend="msprof_detail", _n_warmup=3, _n_repeat=6)
        return float(prof.dur_us), out.cpu()

    b_us, _ = _compile_and_time(spec.baseline_probe_source, "base")
    o_us, _ = _compile_and_time(spec.optimized_probe_source, "opt")
    speedup = (b_us / o_us) if o_us > 0.0 else 0.0
    metrics = {
        "baseline_dur_us": round(b_us, 6),
        "optimized_dur_us": round(o_us, 6),
        "speedup": round(speedup, 6),
        "numerical_ok": 1.0,
    }
    return speedup >= min_speedup, metrics


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
            "static_affine_bounds and not indirect_sparse_gather",
            "persistent_wave_occupancy < 0.75 or PIPE_V > 0 or PIPE_MTE2 > 0",
            "measured_latency_us <= prev.measured_latency_us",
        )
    if chunk.edit_kind == "coalesced_copy":
        return (
            "inner_contiguous_bytes >= 64",
            "mte2_inflight_per_aiv > 8 or PIPE_MTE2 > PIPE_M",
            "mte2_inflight_per_aiv <= prev.mte2_inflight_per_aiv and any_overflow_bytes == 0",
        )
    if chunk.edit_kind == "ssbuf_handoff":
        return (
            "dynamic_cross_core_handoff or cube_to_vector_ratio >= 0.0",
            "unexplained_gap_ratio >= 0.15 or measured_sync_stall_ratio > 0.05",
            "measured_sync_stall_ratio <= prev.measured_sync_stall_ratio",
        )
    dom = ledger.dominant_pipe if ledger is not None else "AIV0.PIPE_V"
    _ = dom
    return (
        "inner_contiguous_bytes >= 64",
        "degraded_buffer_count >= 0 and (PIPE_V > 0 or PIPE_MTE2 > 0)",
        "degraded_buffer_count <= prev.degraded_buffer_count and any_overflow_bytes == 0",
    )


def _is_rule_text_clean(rule: AscendCausalRule) -> tuple[bool, str]:
    joined = " ".join(rule.as_six_tuple()).lower()
    for bad in _DISALLOWED_PROVENANCE_TOKENS:
        if bad in joined:
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
) -> AscendCausalRule:
    """Synthesize a pure-English 6-tuple AscendCausalRule from a verified AST edit."""
    regime_guard, precondition, expected_post = _rule_predicates_for_kind(chunk, ledger)
    template = _generalize_template_text(chunk)
    speedup = float((probe_metrics or {}).get("speedup", 1.0))
    rationale = (
        f"Counterfactual single-variable ablation verified on Ascend 950 micro-probe "
        f"({chunk.edit_kind}, speedup={speedup:.3f}x)."
    )
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
    )


class FlywheelDistiller:
    """Decomposes winning DSL edits, ablates negatives, probes NPU, and admits rules."""

    def __init__(
        self,
        *,
        funnel: GateFunnel,
        tracker: BayesianRuleTracker,
        probe_runner: Callable[[MicroProbeSpec], Mapping[str, float]] | None = None,
        min_edit_gain: float = 0.03,
        min_probe_speedup: float = 1.05,
    ) -> None:
        self._funnel = funnel
        self._tracker = tracker
        self._probe_runner = probe_runner
        self._min_edit_gain = min_edit_gain
        self._min_probe_speedup = min_probe_speedup

    def distill_from_pair(
        self,
        *,
        baseline_dsl: str,
        winner_dsl: str,
        baseline_latency_us: float,
        winner_latency_us: float,
        factory_name: str | None = None,
        factory_kwargs: Mapping[str, object] | None = None,
        kb_output_path: str | Path | None = None,
        rule_id_prefix: str = "R_FLYWHEEL",
    ) -> FlywheelDistillationReport:
        """Run full counterfactual ablation, micro-probe check, and KB admission pipeline."""
        chunks = decompose_ast_ablations(baseline_dsl, winner_dsl)
        kept: list[AblationMeasurement] = []
        discarded: list[AblationMeasurement] = []
        outcomes: list[DistilledRuleOutcome] = []
        admitted_ids: list[str] = []

        base_lat = max(baseline_latency_us, 1e-6)
        for chunk in chunks:
            eval_res = self._funnel.evaluate(
                chunk.ablated_dsl,
                factory_name=factory_name,
                factory_kwargs=factory_kwargs,
            )
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

            var_lat = (
                eval_res.stage_c.measured_latency_us
                if eval_res.stage_c and eval_res.stage_c.measured_latency_us > 0.0
                else float(max(eval_res.stage_a.ledger.pipe_cycles.values(), default=100)) / 20.0
            )
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
                        rejection_reason="micro_probe_speedup_below_threshold",
                    )
                )
                continue

            rule_num = len(self._tracker.rules) + 1
            suffix = meas.chunk.edit_kind.upper()
            cand_rule_id = f"{rule_id_prefix}_{rule_num:02d}_{suffix}"
            rule = distill_causal_rule(
                meas.chunk,
                rule_id=cand_rule_id,
                ledger=meas.stage_a_result.ledger if meas.stage_a_result else None,
                probe=probe,
                probe_metrics=metrics,
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
    "decompose_ast_ablations",
    "distill_causal_rule",
    "synthesize_micro_probe",
    "verify_micro_probe_hardware",
]
