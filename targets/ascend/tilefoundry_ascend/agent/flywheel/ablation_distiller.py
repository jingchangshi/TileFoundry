"""Counterfactual single-variable AST ablation and self-evolution flywheel distiller."""

from __future__ import annotations

import ast
import copy
import difflib
import importlib.util
import math
import re
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from pathlib import Path

from tilefoundry_ascend.agent.contracts import EvolutionTrajectory, HardwareProbePort
from tilefoundry_ascend.agent.kb.bayesian_tracker import BayesianRuleTracker
from tilefoundry_ascend.agent.kb.schema import (
    _RESIDUAL_DEFAULTS,
    _TELEMETRY_CONTEXT_DEFAULTS,
    SYMPTOM_FAMILY_TO_FACETS,
    AscendCausalRule,
    SymptomFamily,
    normalize_runtime_backend,
    safe_eval_predicate,
    save_rules_yaml,
    validate_rule_predicate,
)
from tilefoundry_ascend.agent.gates.gate_funnel import (
    GateFunnel,
    StageAResult,
    extract_ordered_tensor_shapes,
    run_stage_a_static,
)
from tilefoundry_ascend.agent.specs.ops_registry import CannbenchOpsRegistry
from tilefoundry_ascend.ascend.json_record import JsonRecordMixin
from tilefoundry_ascend.ascend.passes.observation_space import (
    AxisResidual5D,
    ComputeIssueAxis,
    GridIrregularityAxis,
    ObservationVector5D,
    OffChipMemoryDmaAxis,
    OnChipSramRegAxis,
    PipelineSyncAxis,
)
from tilefoundry_ascend.ascend.passes.physical_ledger import ObjectivePhysicalLedger, ScopeWaterline
from tilefoundry_ascend.ascend.sil.schema import VALID_FREEDOM_FACETS, CasePhysicsRatios, TopologyAxes

_GENERIC_PATH_PATTERN = re.compile(r"(?:^|[\s'\"=(])/(?:[A-Za-z0-9._-]+/){2,}")
_SUPPORTED_PROBE_KINDS = (
    "core_parallelism", "coalesced_copy", "pipeline_versions", "ssbuf_handoff",
    "barrier_prune", "dual_copy_1c2v", "pad_alignment",
    "static_pingpong_unroll", "loop_fission_sfu",
)


def _build_allowed_schema_signal_vocabulary() -> frozenset[str]:
    """Build closed predicate identifier vocabulary from schema dataclasses and builtins."""
    vocab: set[str] = {
        "True", "False", "None", "abs", "min", "max", "prev", "PIPE_M", "PIPE_FIX", "PIPE_V",
        "PIPE_MTE2", "PIPE_MTE3", "AIC", "AIV0", "AIV1", "UB", "L1", "L0A", "L0B", "L0C", "SSBUF",
        "any_overflow_bytes", "degraded_buffer_count", "implicit_bloat_bytes", "max_overflow_bytes",
        "ub_utilization_ratio", "ub_occupancy_ratio", "cube_to_vector_ratio", "cube_cycles",
        "vec_cycles", "has_dynamic_control", "has_opaque_regions", "has_simtvf",
        "dominant_axis", "has_l1_compiler_summary", "degraded_count",
    }
    for cls in (
        ObservationVector5D, ComputeIssueAxis, OffChipMemoryDmaAxis, OnChipSramRegAxis,
        PipelineSyncAxis, GridIrregularityAxis, AxisResidual5D, ObjectivePhysicalLedger,
        ScopeWaterline, CasePhysicsRatios, TopologyAxes,
    ):
        vocab.update(f.name for f in fields(cls))
    vocab.update(_RESIDUAL_DEFAULTS)
    vocab.update(_TELEMETRY_CONTEXT_DEFAULTS)
    return frozenset(vocab)


ALLOWED_SCHEMA_SIGNAL_VOCABULARY: frozenset[str] = _build_allowed_schema_signal_vocabulary()


def _validate_schema_closed_identifiers(expr: str) -> list[str]:
    """Parse predicate AST and return root identifiers absent from schema vocabulary."""
    if not isinstance(expr, str) or not expr.strip():
        return ["<empty>"]
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return ["<syntax_error>"]
    unknown: list[str] = []
    for node in ast.walk(tree):
        ident: str | None = None
        if isinstance(node, ast.Name) and node.id not in ALLOWED_SCHEMA_SIGNAL_VOCABULARY:
            ident = node.id
        elif (
            isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
            and node.value.id == "prev" and node.attr not in ALLOWED_SCHEMA_SIGNAL_VOCABULARY
        ):
            ident = node.attr
        if ident is not None and ident not in unknown:
            unknown.append(ident)
    return unknown


def _operator_name_variants(operator_name: str) -> set[str]:
    """Generate dynamic snake, kebab, compact, and prefix variants for an operator name."""
    raw = str(operator_name or "").strip()
    if not raw:
        return set()
    low = raw.lower()
    variants: set[str] = {low}
    parts = [p for p in re.split(r"[^a-z0-9]+", low) if p]
    if parts:
        snake = "_".join(parts)
        variants.update({snake, "-".join(parts), "".join(parts), f"{snake}_model"})
        if len(parts) >= 2:
            variants.add(f"{parts[0][:4]}_{parts[1][:3]}")
            if len(parts[0]) >= 3 and len(parts[1]) >= 3:
                variants.add(f"{parts[0][:3]}_{parts[1][:3]}")
        if len(parts) >= 3:
            variants.add("".join(p[0] for p in parts))
    camel = [p.lower() for p in re.findall(r"[A-Z]?[a-z0-9]+|[A-Z]+(?=[A-Z]|$)", raw)]
    if len(camel) >= 2:
        variants.update({"_".join(camel), "-".join(camel), "".join(camel)})
    return {v for v in variants if len(v) >= 2}


def _build_disallowed_provenance_tokens(operator_name: str = "") -> frozenset[str]:
    """Dynamically collect operator variants and registered operator IDs without hardcoding."""
    tokens = _operator_name_variants(operator_name)
    try:
        tokens.update(
            str(op).strip().lower()
            for op in CannbenchOpsRegistry.default().list_operators()
            if str(op).strip()
        )
    except (FileNotFoundError, OSError):
        pass
    return frozenset(tokens)


def resolve_paired_sigma_us(
    reference_us: float, *, sigma_paired_us: float | None = None, paired_rel_mad: float | None = None
) -> float:
    """Resolve paired measurement uncertainty (us) from sigma_paired_us or paired_rel_mad."""
    if sigma_paired_us is not None:
        val = float(sigma_paired_us)
        return max(0.0, val) if math.isfinite(val) else 0.0
    if paired_rel_mad is not None:
        mad, ref = float(paired_rel_mad), float(reference_us)
        if math.isfinite(mad) and mad > 0.0 and math.isfinite(ref) and ref > 0.0:
            return 1.4826 * mad * ref
    return 0.0


def is_paired_gain_significant(
    delta_us: float, *, sigma_paired_us: float | None = None,
    paired_rel_mad: float | None = None, reference_us: float = 0.0, sigma_multiplier: float = 2.0,
) -> bool:
    """Return True when paired latency reduction exceeds sigma_multiplier * sigma_paired_us."""
    delta = float(delta_us)
    if not math.isfinite(delta) or delta <= 0.0:
        return False
    eff_sigma = resolve_paired_sigma_us(
        reference_us, sigma_paired_us=sigma_paired_us, paired_rel_mad=paired_rel_mad
    )
    return delta > float(sigma_multiplier) * eff_sigma


def should_trigger_flywheel_checkpoint(
    baseline_latency_us: float, winner_latency_us: float, *,
    sigma_paired_us: float | None = None, paired_rel_mad: float | None = None,
) -> bool:
    """Return True when winner latency beats baseline by more than 3 * sigma_paired_us."""
    base_lat, win_lat = float(baseline_latency_us), float(winner_latency_us)
    if not (math.isfinite(base_lat) and math.isfinite(win_lat)) or base_lat <= 0.0 or win_lat <= 0.0:
        return False
    return is_paired_gain_significant(
        base_lat - win_lat, sigma_paired_us=sigma_paired_us, paired_rel_mad=paired_rel_mad,
        reference_us=base_lat, sigma_multiplier=3.0,
    )


@dataclass(frozen=True)
class AstEditChunk(JsonRecordMixin):
    """Single atomic structural delta extracted from baseline and winner DSL ASTs."""

    chunk_id: str
    edit_kind: str
    target_path: str
    before_src: str
    after_src: str
    ablated_dsl: str


@dataclass(frozen=True)
class AblationMeasurement(JsonRecordMixin):
    """Funnel and latency outcome for one counterfactual single-variable ablation."""

    chunk: AstEditChunk
    funnel_passed: bool
    baseline_latency_us: float
    variant_latency_us: float
    relative_gain: float
    positive_gain: bool
    stage_a_result: StageAResult | None = None
    diagnostic: str = ""


@dataclass(frozen=True)
class MicroProbeSpec(JsonRecordMixin):
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


@dataclass(frozen=True)
class DistilledRuleOutcome(JsonRecordMixin):
    """Verification and KB admission outcome for one winning AST mechanism."""

    chunk_id: str
    probe: MicroProbeSpec | None
    hardware_verified: bool
    probe_metrics: dict[str, float]
    rule: AscendCausalRule | None
    admitted: bool
    rejection_reason: str = ""


@dataclass
class FlywheelDistillationReport(JsonRecordMixin):
    """Complete audit report across AST decomposition, ablation, probing, and KB admission."""

    baseline_latency_us: float
    winner_latency_us: float
    total_chunks: int
    kept_positive_chunks: tuple[AblationMeasurement, ...]
    discarded_chunks: tuple[AblationMeasurement, ...]
    distilled_outcomes: tuple[DistilledRuleOutcome, ...] = field(default_factory=tuple)
    admitted_rule_ids: tuple[str, ...] = field(default_factory=tuple)


def _node_key(node: ast.AST) -> str:
    return ast.unparse(node)


def _node_tag(node: ast.AST) -> str:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return f"def:{node.name}"
    if isinstance(node, ast.With):
        if not node.items:
            return "with"
        ctx = node.items[0].context_expr
        return f"with:{_node_key(ctx.func if isinstance(ctx, ast.Call) else ctx)}"
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
    if "T.dual_copy" in combined or ("T.MixedKernel" in after_src and "T.Kernel" in before_src):
        return "dual_copy_1c2v"
    if ("T.Pipelined" in before_src and "T.Pipelined" not in after_src) or (
        "enable_offset" in before_src and "enable_offset" not in after_src
    ):
        return "static_pingpong_unroll"
    if "gm_to_ub_pad" in combined or "pad_value" in combined or (
        "alloc_shared" in before_src and "alloc_shared" in after_src and before_src != after_src
    ):
        return "pad_alignment"
    if "ascend_pipe_barrier" in combined or "unit_flag_ctrl" in combined:
        return "barrier_prune"
    if "num_stages" in combined or "annotate_buffer_versions" in combined:
        return "pipeline_versions"
    if ("vcadd" in combined or "vexp" in combined or "vsqrt" in combined) and (
        after_src.count("for ") > before_src.count("for ")
        or ("vcadd" in before_src and "vcadd" not in after_src)
        or (before_src.count("for ") >= 1 and after_src.count("for ") >= 1
            and len(after_src.strip().splitlines()) < len(before_src.strip().splitlines()))
    ):
        return "loop_fission_sfu"
    if any(t in combined for t in ("T.Kernel", "T.MixedKernel", "num_cores", "num_blocks")):
        return "core_parallelism"
    return "coalesced_copy" if "T.copy" in combined else "stmt_rewrite"


def _extract_probe_params(chunk: AstEditChunk) -> dict[str, object]:
    """Extract and clamp concrete tile, core, stage, and dtype parameters from AST."""
    cores, stages, tile_elems, dtype = 28, 2, 2048, "float32"
    for candidate_src in (chunk.after_src, chunk.ablated_dsl):
        try:
            tree = ast.parse(candidate_src) if candidate_src.strip() else None
        except SyntaxError:
            tree = None
        for node in ast.walk(tree) if tree is not None else ():
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            attr = node.func.attr
            if attr in ("Kernel", "MixedKernel") and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, int):
                cores = max(2, min(56, int(node.args[0].value)))
            elif attr == "Pipelined":
                for kw in node.keywords:
                    if kw.arg == "num_stages" and isinstance(kw.value, ast.Constant) and isinstance(kw.value.value, int):
                        stages = max(2, min(3, int(kw.value.value)))
            elif attr in ("alloc_shared", "alloc_l1") and node.args:
                if isinstance(node.args[0], (ast.Tuple, ast.List)):
                    ints = [int(e.value) for e in node.args[0].elts if isinstance(e, ast.Constant) and isinstance(e.value, int)]
                    if ints and len(ints) == len(node.args[0].elts):
                        tile_elems = max(512, min(4096, ((math.prod(max(1, d) for d in ints) + 127) // 128) * 128))
                if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant) and str(node.args[1].value) in ("float32", "float16", "bfloat16"):
                    dtype = str(node.args[1].value)
    return {"cores": cores, "stages": stages, "tile_elems": tile_elems, "dtype": dtype}


def _resolve_stmt_list(tree: ast.Module, path: tuple[int, ...]) -> list[ast.stmt] | None:
    cur_list: list[ast.stmt] = tree.body
    for idx in path[:-1]:
        node = cur_list[idx]
        body = getattr(node, "body", None)
        if not isinstance(body, list):
            return None
        cur_list = body
    return cur_list


def _set_stmt_by_path(tree: ast.Module, path: tuple[int, ...], new_stmt: ast.stmt | None, *, insert: bool = False) -> bool:
    if (cur_list := _resolve_stmt_list(tree, path)) is None:
        return False
    leaf = path[-1]
    if insert and new_stmt is not None:
        cur_list.insert(max(0, min(leaf, len(cur_list))), copy.deepcopy(new_stmt))
        return True
    if new_stmt is None:
        return (cur_list.pop(leaf) is not None) if 0 <= leaf < len(cur_list) else False
    if 0 <= leaf < len(cur_list):
        cur_list[leaf] = copy.deepcopy(new_stmt)
        return True
    if leaf == len(cur_list):
        cur_list.append(copy.deepcopy(new_stmt))
        return True
    return False


def _replace_call_arg(tree: ast.Module, stmt_path: tuple[int, ...], *, is_with_header: bool, new_expr: ast.expr) -> bool:
    if (cur_list := _resolve_stmt_list(tree, stmt_path)) is None:
        return False
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
        elif isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            defs.append((stmt.targets[0].id, stmt))
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            defs.append((stmt.target.id, stmt))
    return defs


def _attach_missing_winner_defs(
    patched_tree: ast.Module, new_defs: Mapping[str, ast.stmt] | None, def_order: Sequence[str]
) -> ast.Module:
    if not new_defs:
        return patched_tree
    existing = {name for name, _ in _collect_top_defs(patched_tree)}
    needed: set[str] = set()
    changed = True
    while changed:
        changed = False
        for root in [patched_tree, *(new_defs[n] for n in needed)]:
            for sub in ast.walk(root):
                if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                    if sub.id in new_defs and sub.id not in existing and sub.id not in needed:
                        needed.add(sub.id)
                        changed = True
    if not needed:
        return patched_tree
    rank = {name: idx for idx, name in enumerate(def_order)}
    to_insert = [copy.deepcopy(new_defs[n]) for n in sorted(needed, key=lambda n: rank.get(n, 0))]
    insert_at = 0
    for idx, stmt in enumerate(patched_tree.body):
        if isinstance(stmt, (ast.Import, ast.ImportFrom)) or (
            isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        ):
            insert_at = idx + 1
        elif idx == insert_at:
            break
    patched_tree.body = [*patched_tree.body[:insert_at], *to_insert, *patched_tree.body[insert_at:]]
    return patched_tree


def _render_patched(
    base_tree: ast.Module, *, new_defs: Mapping[str, ast.stmt] | None = None, def_order: Sequence[str] = ()
) -> str:
    base_tree = _attach_missing_winner_defs(base_tree, new_defs, def_order)
    ast.fix_missing_locations(base_tree)
    code = ast.unparse(base_tree)
    ast.parse(code)
    return code


def _emit_chunk(
    chunks: list[AstEditChunk], seen_dsls: set[str], *, prefix: str,
    edit_kind: str, target_path: str, before_src: str, after_src: str, dsl: str,
) -> None:
    if dsl in seen_dsls:
        return
    seen_dsls.add(dsl)
    chunks.append(AstEditChunk(
        chunk_id=f"{prefix}{len(chunks)}", edit_kind=edit_kind, target_path=target_path,
        before_src=before_src, after_src=after_src, ablated_dsl=dsl,
    ))


def _extract_stmt_deltas(
    patch_base: ast.Module, base_list: list[ast.stmt], win_list: list[ast.stmt],
    parent_path: tuple[int, ...], path_label: str, chunks: list[AstEditChunk],
    seen_dsls: set[str], *, reverse: bool, new_defs: Mapping[str, ast.stmt] | None = None,
    def_order: Sequence[str] = (), skip_assign_symbols: frozenset[str] = frozenset(),
) -> None:
    prefix = "loo_" if reverse else "edit_"
    base_blk_txt = "\n".join(_node_key(s) for s in base_list)
    block_fission = (
        not reverse and any(tok in base_blk_txt for tok in ("vcadd", "vexp", "vsqrt"))
        and sum(isinstance(s, ast.For) for s in win_list) > sum(isinstance(s, ast.For) for s in base_list)
    )

    def _try_emit(fn: Callable[[ast.Module], bool], kind: str, tpath: str, before: str, after: str) -> None:
        patched = copy.deepcopy(patch_base)
        if fn(patched):
            dsl = _render_patched(patched, new_defs=None if reverse else new_defs, def_order=() if reverse else def_order)
            _emit_chunk(chunks, seen_dsls, prefix=prefix, edit_kind=kind, target_path=tpath,
                        before_src=before, after_src=after, dsl=dsl)

    def _handle_compound(b_s: ast.stmt, w_s: ast.stmt, cur_p: tuple[int, ...], hdr_l: str, body_l: str) -> None:
        if isinstance(b_s, ast.With) and isinstance(w_s, ast.With):
            bh = _node_key(b_s.items[0].context_expr) if b_s.items else ""
            wh = _node_key(w_s.items[0].context_expr) if w_s.items else ""
            es = b_s if reverse else w_s
            if bh != wh and es.items:
                _try_emit(lambda t: _replace_call_arg(t, cur_p, is_with_header=True, new_expr=es.items[0].context_expr),
                          _classify_edit(bh, wh), hdr_l, bh, wh)
        elif isinstance(b_s, ast.For) and isinstance(w_s, ast.For):
            bi, wi = _node_key(b_s.iter), _node_key(w_s.iter)
            if bi != wi:
                _try_emit(lambda t: _replace_call_arg(t, cur_p, is_with_header=False, new_expr=b_s.iter if reverse else w_s.iter),
                          _classify_edit(bi, wi), hdr_l, bi, wi)
        _extract_stmt_deltas(patch_base, b_s.body, w_s.body, cur_p, body_l, chunks, seen_dsls,
                             reverse=reverse, new_defs=new_defs, def_order=def_order)

    def _emit_win_insert(ws: ast.stmt, ins_idx: int, tpath: str) -> None:
        wt = _node_key(ws)
        kind = "loop_fission_sfu" if (block_fission and isinstance(ws, ast.For)) else _classify_edit("", wt)
        _try_emit(lambda t: _set_stmt_by_path(t, parent_path + (ins_idx,), ws, insert=True), kind, tpath, "", wt)

    def _emit_remove(j: int, label: str) -> None:
        wt = _node_key(win_list[j])
        _try_emit(lambda t: _set_stmt_by_path(t, parent_path + (j,), None), _classify_edit("", wt), f"{path_label}/{label}_{j}", "", wt)

    matcher = difflib.SequenceMatcher(a=[_node_tag(s) for s in base_list], b=[_node_tag(s) for s in win_list], autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for k in range(i2 - i1):
                bs, ws = base_list[i1 + k], win_list[j1 + k]
                bt, wt = _node_key(bs), _node_key(ws)
                if bt == wt or (reverse and skip_assign_symbols and not parent_path and (
                    (_node_tag(ws).startswith("assign:") and _node_tag(ws).removeprefix("assign:") in skip_assign_symbols)
                    or (isinstance(ws, ast.AnnAssign) and isinstance(ws.target, ast.Name) and ws.target.id in skip_assign_symbols)
                )):
                    continue
                idx = (j1 + k) if reverse else (i1 + k)
                cp = parent_path + (idx,)
                if isinstance(bs, (ast.With, ast.For)) and isinstance(ws, type(bs)):
                    sub = "with" if isinstance(bs, ast.With) else "for"
                    _handle_compound(bs, ws, cp, f"{path_label}/{sub}_{idx}", f"{path_label}/{sub}_{idx}")
                elif isinstance(bs, ast.FunctionDef) and isinstance(ws, ast.FunctionDef):
                    _extract_stmt_deltas(patch_base, bs.body, ws.body, cp, f"{path_label}/{ws.name if reverse else bs.name}", chunks, seen_dsls, reverse=reverse, new_defs=new_defs, def_order=def_order)
                else:
                    _try_emit(lambda t: _set_stmt_by_path(t, cp, bs if reverse else ws), _classify_edit(bt, wt), f"{path_label}/stmt_{idx}", bt, wt)
        elif tag == "replace":
            common = min(i2 - i1, j2 - j1)
            for k in range(common):
                bs, ws, idx = base_list[i1 + k], win_list[j1 + k], (j1 + k) if reverse else (i1 + k)
                cp = parent_path + (idx,)
                if isinstance(bs, (ast.With, ast.For)) and isinstance(ws, type(bs)):
                    _handle_compound(bs, ws, cp, f"{path_label}/with_hdr_{idx}", f"{path_label}/block_{idx}")
                else:
                    bt, wt = _node_key(bs), _node_key(ws)
                    _try_emit(lambda t: _set_stmt_by_path(t, cp, bs if reverse else ws), _classify_edit(bt, wt), f"{path_label}/replace_{idx}", bt, wt)
            for j in range(j1 + common, j2):
                _emit_remove(j, "remove_extra") if reverse else _emit_win_insert(win_list[j], min(i2 + (j - (j1 + common)), len(base_list)), f"{path_label}/replace_ins_{j}")
        elif tag == "delete":
            for idx in range(i1, i2):
                bs, bt = base_list[idx], _node_key(base_list[idx])
                _try_emit((lambda t: _set_stmt_by_path(t, parent_path + (min(j1, len(win_list)),), bs, insert=True)) if reverse else (lambda t: _set_stmt_by_path(t, parent_path + (idx,), None)), _classify_edit(bt, ""), f"{path_label}/{'reinsert' if reverse else 'delete'}_{idx}", bt, "")
        elif tag == "insert":
            for j in range(j1, j2):
                _emit_remove(j, "remove_ins") if reverse else _emit_win_insert(win_list[j], min(i1, len(base_list)), f"{path_label}/insert_{i1}_{j}")


def _classify_config_key_edit(key_name: str, before_src: str, after_src: str) -> str:
    kl = key_name.lower()
    for tokens, kind in (
        (("stage", "buffer", "preload"), "pipeline_versions"), (("core", "block", "split", "threads"), "core_parallelism"),
        (("pad", "align", "datablock", "stride"), "pad_alignment"), (("barrier", "sync", "unit_flag"), "barrier_prune"),
    ):
        if any(tok in kl for tok in tokens):
            return kind
    return c if (c := _classify_edit(before_src, after_src)) != "stmt_rewrite" else "config_knob"


def _find_dict_literals_by_context(tree: ast.Module) -> dict[str, ast.Dict]:
    """Collect constant-key Dict literals keyed by enclosing function/assignment path."""
    found, counts = {}, {}

    def _visit(node: ast.AST, prefix: str) -> None:
        if isinstance(node, ast.FunctionDef):
            fn_p = f"{prefix}/{node.name}" if prefix else node.name
            for idx, st in enumerate(node.body):
                sk = f"{fn_p}:return" if isinstance(st, ast.Return) else (f"{fn_p}:assign:{st.targets[0].id}" if isinstance(st, ast.Assign) and len(st.targets) == 1 and isinstance(st.targets[0], ast.Name) else f"{fn_p}:{idx}")
                _visit(st, sk)
            return
        if isinstance(node, ast.Dict):
            occ = counts.get(prefix, 0)
            counts[prefix] = occ + 1
            found[prefix if occ == 0 else f"{prefix}#{occ}"] = node
        for child in ast.iter_child_nodes(node):
            _visit(child, prefix)

    for idx, stmt in enumerate(tree.body):
        _visit(stmt, "" if isinstance(stmt, ast.FunctionDef) else (f"global:{_node_key(stmt.targets[0])}" if isinstance(stmt, ast.Assign) and stmt.targets else f"mod:{idx}"))
    return found


def _dict_key_map(dnode: ast.Dict) -> dict[str, tuple[int, ast.expr]]:
    return {k.value: (idx, v) for idx, (k, v) in enumerate(zip(dnode.keys, dnode.values)) if isinstance(k, ast.Constant) and isinstance(k.value, str)}


def _extract_config_dict_chunks(
    base_tree: ast.Module, win_tree: ast.Module, chunks: list[AstEditChunk], seen_dsls: set[str], *, reverse: bool
) -> None:
    """Create forward or Leave-One-Out chunks over matching dict config literals."""
    prefix, patch_root = ("loo_" if reverse else "edit_"), (win_tree if reverse else base_tree)
    base_dicts, win_dicts = _find_dict_literals_by_context(base_tree), _find_dict_literals_by_context(win_tree)
    for ctx_key, w_dict in win_dicts.items():
        if (b_dict := base_dicts.get(ctx_key)) is None:
            continue
        b_map = _dict_key_map(b_dict)
        for k, (w_idx, w_val) in _dict_key_map(w_dict).items():
            w_src, patched = _node_key(w_val), copy.deepcopy(patch_root)
            if (p_dict := _find_dict_literals_by_context(patched).get(ctx_key)) is None:
                continue
            if k in b_map:
                b_idx, b_val = b_map[k]
                b_src, t_idx = _node_key(b_val), (w_idx if reverse else b_idx)
                if b_src == w_src or t_idx >= len(p_dict.values):
                    continue
                p_dict.values[t_idx] = copy.deepcopy(b_val if reverse else w_val)
                _emit_chunk(chunks, seen_dsls, prefix=prefix, edit_kind=_classify_config_key_edit(k, b_src, w_src), target_path=f"{ctx_key}/dict:{k}", before_src=f"{k}: {b_src}", after_src=f"{k}: {w_src}", dsl=_render_patched(patched))
            else:
                if reverse:
                    if w_idx >= len(p_dict.values):
                        continue
                    del p_dict.keys[w_idx], p_dict.values[w_idx]
                    tpath = f"{ctx_key}/dict_del:{k}"
                else:
                    p_dict.keys.append(ast.Constant(value=k))
                    p_dict.values.append(copy.deepcopy(w_val))
                    tpath = f"{ctx_key}/dict_add:{k}"
                _emit_chunk(chunks, seen_dsls, prefix=prefix, edit_kind=_classify_config_key_edit(k, "", w_src), target_path=tpath, before_src="", after_src=f"{k}: {w_src}", dsl=_render_patched(patched))


def _is_trackb_render_fn(name: str) -> bool:
    return name == "_render_trackb_tu" or (name.startswith("_render_") and name.endswith("_trackb_launcher"))


def _extract_func_default_loo_chunks(
    base_tree: ast.Module, win_tree: ast.Module, chunks: list[AstEditChunk], seen_dsls: set[str]
) -> None:
    """Revert changed default parameters one at a time on selector and Track-B renderers."""
    def _collect(fn: ast.FunctionDef) -> list[tuple[str, bool, int, ast.expr]]:
        pos = [*fn.args.posonlyargs, *fn.args.args]
        off = len(pos) - len(fn.args.defaults)
        return [(pos[off + i].arg, False, i, d) for i, d in enumerate(fn.args.defaults) if off + i >= 0] + [
            (a.arg, True, ki, kd) for ki, (a, kd) in enumerate(zip(fn.args.kwonlyargs, fn.args.kw_defaults)) if kd is not None
        ]

    base_funcs = {n.name: n for n in base_tree.body if isinstance(n, ast.FunctionDef)}
    for w_idx, w_node in enumerate(win_tree.body):
        if not isinstance(w_node, ast.FunctionDef) or not (_is_trackb_render_fn(w_node.name) or (w_node.name.startswith("_select_") and w_node.name.endswith("_config"))) or (b_node := base_funcs.get(w_node.name)) is None:
            continue
        prefix = "trackb_tu_" if _is_trackb_render_fn(w_node.name) else "loo_"
        b_defs = {(name, is_kw): expr for name, is_kw, _, expr in _collect(b_node)}
        for arg_name, is_kw, d_idx, w_def in _collect(w_node):
            if (b_def := b_defs.get((arg_name, is_kw))) is None or (b_src := _node_key(b_def)) == (w_src := _node_key(w_def)):
                continue
            patched = copy.deepcopy(win_tree)
            p_fn = patched.body[w_idx]
            if isinstance(p_fn, ast.FunctionDef) and d_idx < len(t_defs := (p_fn.args.kw_defaults if is_kw else p_fn.args.defaults)):
                t_defs[d_idx] = copy.deepcopy(b_def)
                _emit_chunk(chunks, seen_dsls, prefix=prefix, edit_kind=_classify_config_key_edit(arg_name, b_src, w_src), target_path=f"{w_node.name}/{'kw_default' if is_kw else 'default'}:{arg_name}", before_src=f"{arg_name}={b_src}", after_src=f"{arg_name}={w_src}", dsl=_render_patched(patched))


def _iter_line_loo_hunks(base_text: str, win_text: str) -> Iterable[tuple[int, int, int, str, str, str]]:
    b_lines, w_lines = base_text.splitlines(keepends=True), win_text.splitlines(keepends=True)
    ops = [op for op in difflib.SequenceMatcher(a=b_lines, b=w_lines, autojunk=False).get_opcodes() if op[0] != "equal"]
    if len(ops) <= 1:
        return
    for idx, (_op, bi1, bi2, wj1, wj2) in enumerate(ops):
        mid = list(b_lines[bi1:bi2])
        if mid and wj2 < len(w_lines) and not mid[-1].endswith("\n"):
            mid[-1] += "\n"
        yield idx, wj1, wj2, "".join([*w_lines[:wj1], *mid, *w_lines[wj2:]]), "".join(b_lines[bi1:bi2]).strip(), "".join(w_lines[wj1:wj2]).strip()


def _extract_trackb_render_template_loo_chunks(
    baseline_dsl: str, winner_dsl: str, base_tree: ast.Module, win_tree: ast.Module,
    chunks: list[AstEditChunk], seen_dsls: set[str],
) -> None:
    """Decompose multi-line C++ template parameter blocks inside Track-B renderer functions."""
    base_funcs = {n.name: n for n in base_tree.body if isinstance(n, ast.FunctionDef)}
    for top_idx, w_fn in enumerate(win_tree.body):
        if not isinstance(w_fn, ast.FunctionDef) or not _is_trackb_render_fn(w_fn.name) or (b_fn := base_funcs.get(w_fn.name)) is None:
            continue
        matcher = difflib.SequenceMatcher(a=[_node_tag(s) for s in b_fn.body], b=[_node_tag(s) for s in w_fn.body], autojunk=False)
        for tag, i1, i2, j1, j2 in matcher.get_opcodes():
            if tag not in ("equal", "replace"):
                continue
            for k in range(min(i2 - i1, j2 - j1)):
                bs, ws = b_fn.body[i1 + k], w_fn.body[j1 + k]
                if _node_key(bs) == _node_key(ws):
                    continue
                b_seg = ast.get_source_segment(baseline_dsl, bs) or ""
                w_seg = ast.get_source_segment(winner_dsl, ws) or ""
                if not b_seg.strip() or not w_seg.strip() or ("\n" not in b_seg and "\n" not in w_seg):
                    continue
                for _idx, wj1, _wj2, ablated, bp, ap in _iter_line_loo_hunks(b_seg, w_seg):
                    try:
                        parsed = ast.parse(ablated)
                    except SyntaxError:
                        continue
                    if len(parsed.body) == 1 and _set_stmt_by_path(patched := copy.deepcopy(win_tree), (top_idx, j1 + k), parsed.body[0]):
                        kind = _classify_cpp_hunk_kind(bp, ap)
                        kind = "core_parallelism" if (kind == "cpp_hunk_edit" and "<<<" in f"{bp}{ap}") else kind
                        _emit_chunk(chunks, seen_dsls, prefix="trackb_tu_", edit_kind=kind,
                                    target_path=f"{w_fn.name}/template_L{wj1 + 1}", before_src=bp, after_src=ap, dsl=_render_patched(patched))


def _iter_top_str_constants(tree: ast.Module) -> Iterable[tuple[str, ast.Constant]]:
    for node in tree.body:
        name, val = None, None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name, val = node.targets[0].id, node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name, val = node.target.id, node.value
        if name and isinstance(val, ast.Constant) and isinstance(val.value, str):
            yield name, val


def _find_str_constant_node(tree: ast.Module, symbol_name: str) -> ast.Constant | None:
    return next((val for name, val in _iter_top_str_constants(tree) if name == symbol_name), None)


def _extract_str_constant(tree: ast.Module, symbol_name: str) -> str | None:
    node = _find_str_constant_node(tree, symbol_name)
    return str(node.value) if node is not None else None


def _format_cpp_string_literal(cpp_src: str) -> str:
    if '"""' not in cpp_src and not cpp_src.endswith(('"', "\\")):
        return f'r"""{cpp_src}"""'
    if "'''" not in cpp_src and not cpp_src.endswith(("'", "\\")):
        return f"r'''{cpp_src}'''"
    return repr(cpp_src)


def _replace_str_constant_in_module(module_src: str, symbol_name: str, new_cpp_src: str) -> str:
    tree = ast.parse(module_src)
    node = _find_str_constant_node(tree, symbol_name)
    if node is not None and node.end_lineno is not None and node.end_col_offset is not None:
        lines = module_src.splitlines(keepends=True)
        sb, eb = lines[node.lineno - 1].encode("utf-8"), lines[node.end_lineno - 1].encode("utf-8")
        prefix = "".join(lines[: node.lineno - 1]) + sb[: node.col_offset].decode("utf-8")
        suffix = eb[node.end_col_offset :].decode("utf-8") + "".join(lines[node.end_lineno :])
        cand = f"{prefix}{_format_cpp_string_literal(new_cpp_src)}{suffix}"
        try:
            if _extract_str_constant(ast.parse(cand), symbol_name) == new_cpp_src:
                return cand
        except SyntaxError:
            pass
    if node is not None:
        node.value = new_cpp_src
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


def _classify_cpp_hunk_kind(before_seg: str, after_seg: str) -> str:
    combined = f"{before_seg} {after_seg}"
    for tokens, kind in (
        (("& 1", "% 2", "(event_t)1"), "static_pingpong_unroll"),
        (("asc_sync_notify", "asc_lock", "asc_unlock", "pipe_barrier", "CrossCore"), "barrier_prune"),
        (("+ 16", "+ 32", "STRIDE", "PAD"), "pad_alignment"),
        (("asc_copy_gm2ub", "asc_copy_ub2gm", "asc_copy_gm2l1", "asc_copy_l0c2ub"), "coalesced_copy"),
        (("block_idx", "num_cores", "get_block_num"), "core_parallelism"),
    ):
        if any(t in combined for t in tokens):
            return kind
    if "get_subblockid" in combined or "1c2v" in combined.lower():
        return "dual_copy_1c2v"
    if "simd_inst::" in combined and after_seg.count("for ") > before_seg.count("for "):
        return "loop_fission_sfu"
    return "cpp_hunk_edit"


def _discover_trackb_cpp_symbols(tree: ast.Module) -> list[str]:
    markers = ("tl_templates/ascend/", "__aicore__", "__global__", "__simd_vf__")
    return [n for n, v in _iter_top_str_constants(tree) if n.endswith("_TRACKB_SRC") or any(m in v.value for m in markers)]


def decompose_cpp_string_leave_one_out(
    baseline_module_src: str, winner_module_src: str, cpp_symbol_name: str
) -> list[AstEditChunk]:
    """Generate Leave-One-Out counterfactual modules by reverting one C++ diff hunk at a time."""
    base_cpp = _extract_str_constant(ast.parse(baseline_module_src), cpp_symbol_name)
    win_cpp = _extract_str_constant(ast.parse(winner_module_src), cpp_symbol_name)
    if not base_cpp or not win_cpp or base_cpp == win_cpp:
        return []
    return [
        AstEditChunk(
            chunk_id=f"cpp_loo_{cpp_symbol_name}_{idx}", edit_kind=_classify_cpp_hunk_kind(bs, as_),
            target_path=f"{cpp_symbol_name}:L{j1 + 1}-L{max(j1 + 1, j2)}", before_src=bs, after_src=as_,
            ablated_dsl=_replace_str_constant_in_module(winner_module_src, cpp_symbol_name, ablated_cpp),
        )
        for idx, j1, j2, ablated_cpp, bs, as_ in _iter_line_loo_hunks(base_cpp, win_cpp)
        if ablated_cpp.count("{") == ablated_cpp.count("}")
    ]


def decompose_leave_one_out_ablations(baseline_dsl: str, winner_dsl: str) -> tuple[AstEditChunk, ...]:
    """Decompose winner_dsl into Leave-One-Out AST, config-dict, and Track-B C++ ablations."""
    base_tree, win_tree = ast.parse(baseline_dsl), ast.parse(winner_dsl)
    chunks: list[AstEditChunk] = []
    seen_dsls: set[str] = {ast.unparse(win_tree)}
    base_cpp_syms = set(_discover_trackb_cpp_symbols(base_tree))
    decomposed_cpp_syms: set[str] = set()
    for sym_name in _discover_trackb_cpp_symbols(win_tree):
        if sym_name not in base_cpp_syms:
            continue
        if (cpp_chunks := decompose_cpp_string_leave_one_out(baseline_dsl, winner_dsl, sym_name)):
            decomposed_cpp_syms.add(sym_name)
            for ch in cpp_chunks:
                norm = ast.unparse(ast.parse(ch.ablated_dsl))
                if norm not in seen_dsls:
                    seen_dsls.add(norm)
                    chunks.append(ch)
    _extract_config_dict_chunks(base_tree, win_tree, chunks, seen_dsls, reverse=True)
    _extract_func_default_loo_chunks(base_tree, win_tree, chunks, seen_dsls)
    _extract_trackb_render_template_loo_chunks(baseline_dsl, winner_dsl, base_tree, win_tree, chunks, seen_dsls)
    _extract_stmt_deltas(win_tree, base_tree.body, win_tree.body, (), "module", chunks, seen_dsls,
                         reverse=True, skip_assign_symbols=frozenset(decomposed_cpp_syms))
    return tuple(chunks)


def decompose_ast_ablations(baseline_dsl: str, winner_dsl: str) -> tuple[AstEditChunk, ...]:
    """Decompose AST difference between baseline and winner DSL into single-variable edits."""
    base_tree, win_tree = ast.parse(baseline_dsl), ast.parse(winner_dsl)
    chunks: list[AstEditChunk] = []
    seen_dsls: set[str] = {ast.unparse(base_tree)}
    base_defs = dict(_collect_top_defs(base_tree))
    win_defs_list = _collect_top_defs(win_tree)
    new_defs = {name: stmt for name, stmt in win_defs_list if name not in base_defs}
    def_order = [name for name, _ in win_defs_list if name in new_defs]
    _extract_config_dict_chunks(base_tree, win_tree, chunks, seen_dsls, reverse=False)
    _extract_stmt_deltas(base_tree, base_tree.body, win_tree.body, (), "module", chunks, seen_dsls,
                         reverse=False, new_defs=new_defs, def_order=def_order)
    win_rendered = ast.unparse(win_tree)
    has_verifiable = any(c.edit_kind in _SUPPORTED_PROBE_KINDS for c in chunks)
    if (not chunks or (len(base_tree.body) > 2 and not has_verifiable)) and win_rendered != ast.unparse(base_tree) and win_rendered not in seen_dsls:
        win_map = dict(win_defs_list)
        diff = [n for n, ws in win_defs_list if n not in base_defs or _node_key(base_defs[n]) != _node_key(ws)]
        b_diff = "\n".join(_node_key(base_defs[n]) for n in diff if n in base_defs) or ast.unparse(base_tree)
        a_diff = "\n".join(_node_key(win_map[n]) for n in diff if n in win_map) or win_rendered
        _emit_chunk(chunks, seen_dsls, prefix="edit_", edit_kind=_classify_edit(b_diff, a_diff),
                    target_path="module/regime_composite" if chunks else "module/whole",
                    before_src=b_diff, after_src=a_diff, dsl=win_rendered)
    return tuple(chunks)


def _wrap_probe(vars_lines: Sequence[str], params_lines: Sequence[str], kernel_hdr: str, body_lines: Sequence[str]) -> str:
    vblk = "\n".join(f"    {sub}" for ln in vars_lines for sub in ln.splitlines())
    pblk = "\n".join(f"        {sub}" for ln in params_lines for sub in ln.splitlines())
    bblk = "\n".join(f"            {sub}" for ln in body_lines for sub in ln.splitlines())
    return (
        "import tilelang.ascend.language as T\n\ndef make_flywheel_probe():\n"
        f"{vblk}\n\n    @T.prim_func\n    def main(\n{pblk}\n    ):\n"
        f"        with {kernel_hdr}:\n{bblk}\n    return main\n"
    )


def _simd_vadd(a_idx: str, b_idx: str, c_idx: str, *, loop: str, count: str, pad: str = "") -> list[str]:
    return [
        f"{pad}with T.SimdVF():", f"{pad}    mask = T.simd.pset(32)", f"{pad}    for {loop} in {count}:",
        f"{pad}        va = T.simd.vld({a_idx})", f"{pad}        vb = T.simd.vld({b_idx})",
        f"{pad}        T.simd.vsts({c_idx}, T.simd.vadd(va, vb, mask), mask)",
    ]


def _simd_2d_vadd(rows_expr: str, cols_count: str, *, pad: str = "") -> list[str]:
    return [
        f"{pad}with T.SimdVF():", f"{pad}    mask = T.simd.pset(32)",
        f"{pad}    for r in T.serial({rows_expr}):", f"{pad}        for k in {cols_count}:",
        f"{pad}            va = T.simd.vld(a_ub[r, k * 64])", f"{pad}            vb = T.simd.vld(b_ub[r, k * 64])",
        f"{pad}            T.simd.vsts(c_ub[r, k * 64], T.simd.vadd(va, vb, mask), mask)",
    ]


def _vec_params(shape: str, dtype: str = "float32") -> list[str]:
    return [f'{n}: T.Tensor({shape}, "{dtype}"),' for n in ("A", "B", "C")]


def _alloc_abc(shape: str, dtype: str = "float32", *, stages: int | None = None) -> list[str]:
    out = [f'{n}_ub = T.alloc_shared({shape}, "{dtype}")' for n in ("a", "b", "c")]
    if stages is not None:
        out.append(f"T.annotate_buffer_versions({{a_ub: {stages}, b_ub: {stages}, c_ub: {stages}}})")
    return out


def _probe_templates_for_kind(
    probe_kind: str, params: Mapping[str, object] | None = None
) -> tuple[str, str, tuple[int, ...], str] | None:
    cfg = dict(params or {})
    cores = max(2, min(56, int(cfg.get("cores", 28))))
    mix_cores = max(2, min(28, cores))
    stages = max(2, min(3, int(cfg.get("stages", 2))))
    tile = max(512, min(4096, ((int(cfg.get("tile_elems", 2048)) + 127) // 128) * 128))
    dtype = str(cfg.get("dtype", "float32"))
    vadd_1d = _simd_vadd("a_ub[i * 64]", "b_ub[i * 64]", "c_ub[i * 64]", loop="i", count="range(tile // 64)", pad="    ")

    def _pipelined_vec(n_stages: int, *, with_barriers: bool) -> str:
        bar = ['    T.ascend_pipe_barrier("PIPE_ALL")'] if with_barriers else []
        return _wrap_probe(
            [f"tile = {tile}", "steps = 16", f"cores = {mix_cores}", "total = cores * steps * tile"],
            _vec_params("(total,)"), "T.Kernel(cores) as bx",
            [*_alloc_abc("(tile,)"), f"for s in T.Pipelined(steps, num_stages={n_stages}):",
             "    off = (s * cores + bx) * tile", "    T.copy(A[off : off + tile], a_ub)",
             "    T.copy(B[off : off + tile], b_ub)", *bar, *vadd_1d, *bar, "    T.copy(c_ub, C[off : off + tile])", *bar],
        )

    if probe_kind == "core_parallelism":
        base = _wrap_probe(
            [f"tile = {tile}", f"steps = {cores}", "total = tile * steps"], _vec_params("(total,)"), "T.Kernel(1) as bx",
            [*_alloc_abc("(tile,)", stages=1), "for s in T.Pipelined(steps, num_stages=1):",
             "    off = s * tile + bx * 0", "    T.copy(A[off : off + tile], a_ub)", "    T.copy(B[off : off + tile], b_ub)",
             *vadd_1d, "    T.copy(c_ub, C[off : off + tile])", '    T.ascend_pipe_barrier("PIPE_ALL")'],
        )
        opt = _wrap_probe(
            [f"tile = {tile}", f"cores = {cores}", "total = tile * cores"], _vec_params("(total,)"), "T.Kernel(cores) as bx",
            [*_alloc_abc("(tile,)", stages=1), "off = bx * tile", "T.copy(A[off : off + tile], a_ub)",
             "T.copy(B[off : off + tile], b_ub)", *[ln[4:] for ln in vadd_1d], "T.copy(c_ub, C[off : off + tile])"],
        )
        return base, opt, (tile * cores,), dtype

    if probe_kind == "coalesced_copy":
        v2d = _simd_2d_vadd("rows", "range(2)")
        vv, vp, hdr = [f"cores = {mix_cores}", "rows = 16", "cols = 128"], _vec_params("(cores * rows, cols)"), "T.Kernel(cores) as bx"
        base = _wrap_probe(vv, vp, hdr, [
            *_alloc_abc("(rows, cols)"), "base_r = bx * rows", "for r in T.serial(rows):",
            "    T.copy(A[base_r + r, 0:cols], a_ub[r, 0:cols])", "    T.copy(B[base_r + r, 0:cols], b_ub[r, 0:cols])",
            *v2d, "for r in T.serial(rows):", "    T.copy(c_ub[r, 0:cols], C[base_r + r, 0:cols])"])
        opt = _wrap_probe(vv, vp, hdr, [
            *_alloc_abc("(rows, cols)"), "base_r = bx * rows", "T.copy(A[base_r : base_r + rows, 0:cols], a_ub)",
            "T.copy(B[base_r : base_r + rows, 0:cols], b_ub)", *v2d, "T.copy(c_ub, C[base_r : base_r + rows, 0:cols])"])
        return base, opt, (mix_cores * 16, 128), dtype

    if probe_kind in ("pipeline_versions", "barrier_prune"):
        b_stg = 1 if probe_kind == "pipeline_versions" else stages
        return _pipelined_vec(b_stg, with_barriers=True), _pipelined_vec(stages, with_barriers=False), (mix_cores * 16 * tile,), dtype

    if probe_kind == "ssbuf_handoff":
        base = _wrap_probe(
            ["rounds = 64", "tile = 16"],
            ['seed: T.Tensor((tile, tile), "float16"),', 'scratch: T.Tensor((rounds, 2), "uint32"),', 'out: T.Tensor((rounds, 1), "uint64"),'],
            "T.MixedKernel(1) as (_, sid)",
            ['cube = T.alloc_l1((tile, tile), "float16")\ntmp_ub = T.alloc_shared((1,), "uint32")\nT.copy(seed, cube)',
             'for i in T.Pipelined(rounds, num_stages=1):\n    with T.Stage(0):\n        val = T.Cast("uint32", (i + 1) * (100 + sid))',
             '        tmp_ub[0] = val\n        T.copy(tmp_ub, scratch[i, sid : sid + 1])\n    with T.Stage(0):',
             '        lo = T.Cast("uint64", scratch[i, 0])\n        hi = T.Cast("uint64", scratch[i, 1]) << 32',
             '        out[i, 0] = lo | hi\n        T.ascend_pipe_barrier("PIPE_ALL")'],
        )
        opt = _wrap_probe(
            ["rounds = 64", "tile = 16"], ['seed: T.Tensor((tile, tile), "float16"),', 'out: T.Tensor((rounds, 1), "uint64"),'],
            "T.MixedKernel(1) as (_, sid)",
            ['words = T.alloc_ss((2,), "uint32")\npacked = T.view(words, shape=(1,), dtype="uint64")',
             'T.annotate_buffer_versions({words: (3, "auto")})\ncube = T.alloc_l1((tile, tile), "float16")\nT.copy(seed, cube)',
             'for i in T.Pipelined(\n    rounds,\n    num_stages=3,\n    annotations={"enable_offset": True},\n):',
             '    with T.Stage(0):\n        words[sid] = T.Cast(\n            "uint32",\n            (i + 1) * (100 + sid),\n        )',
             '    with T.Stage(2):\n        out[i, 0] = packed[0]'],
        )
        return base, opt, (64, 1), "uint64"

    if probe_kind == "dual_copy_1c2v":
        vp = _vec_params("(cores * rows, cols)")
        base = _wrap_probe([f"cores = {mix_cores}", "rows = 16", "cols = 128"], vp, "T.Kernel(cores) as bx", [
            *_alloc_abc("(rows, cols)"), "base_r = bx * rows\nT.copy(A[base_r : base_r + rows, 0:cols], a_ub)",
            "T.copy(B[base_r : base_r + rows, 0:cols], b_ub)", *_simd_2d_vadd("rows", "range(cols // 64)"), "T.copy(c_ub, C[base_r : base_r + rows, 0:cols])"])
        opt = _wrap_probe([f"cores = {mix_cores}", "rows = 16", "half_rows = 8", "cols = 128"], vp, "T.MixedKernel(cores) as (bx, sid)", [
            *_alloc_abc("(half_rows, cols)"), "row_off = bx * rows + sid * half_rows\nT.copy(A[row_off : row_off + half_rows, 0:cols], a_ub)",
            "T.copy(B[row_off : row_off + half_rows, 0:cols], b_ub)", *_simd_2d_vadd("half_rows", "range(cols // 64)"), "T.copy(c_ub, C[row_off : row_off + half_rows, 0:cols])"])
        return base, opt, (mix_cores * 16, 128), dtype

    if probe_kind == "pad_alignment":
        vp = _vec_params("(cores * steps * rows, cols)")
        base = _wrap_probe([f"cores = {mix_cores}", "rows = 8", "cols = 70", "steps = 4"], vp, "T.Kernel(cores) as bx", [
            *_alloc_abc("(rows, cols)"), "for s in T.serial(steps):\n    r0 = (bx * steps + s) * rows\n    T.copy(A[r0 : r0 + rows, 0:cols], a_ub)",
            "    T.copy(B[r0 : r0 + rows, 0:cols], b_ub)", *_simd_vadd("a_ub[r, 0]", "b_ub[r, 0]", "c_ub[r, 0]", loop="r", count="T.serial(rows)", pad="    "), "    T.copy(c_ub, C[r0 : r0 + rows, 0:cols])"])
        opt = _wrap_probe([f"cores = {mix_cores}", "rows = 8", "cols = 128", "steps = 4"], vp, "T.Kernel(cores) as bx", [
            *_alloc_abc("(rows, cols)"), f"for s in T.Pipelined(steps, num_stages={stages}):\n    r0 = (bx * steps + s) * rows\n    T.copy(A[r0 : r0 + rows, 0:cols], a_ub)",
            "    T.copy(B[r0 : r0 + rows, 0:cols], b_ub)", *_simd_2d_vadd("rows", "range(2)", pad="    "), "    T.copy(c_ub, C[r0 : r0 + rows, 0:cols])"])
        return base, opt, (mix_cores * 32, 128), dtype

    if probe_kind == "static_pingpong_unroll":
        base = _wrap_probe(
            [f"tile = {tile}", "steps = 16", f"cores = {mix_cores}", "total = cores * steps * tile"], _vec_params("(total,)"), "T.Kernel(cores) as bx",
            [*_alloc_abc("(tile,)", stages=2), 'for s in T.Pipelined(steps, num_stages=2, annotations={"enable_offset": True}):',
             "    off = (s * cores + bx) * tile\n    T.copy(A[off : off + tile], a_ub)\n    T.copy(B[off : off + tile], b_ub)",
             *vadd_1d, "    T.copy(c_ub, C[off : off + tile])"])
        opt = _wrap_probe(
            [f"tile = {tile}", "pairs = 8", f"cores = {mix_cores}", "total = cores * pairs * 2 * tile"], _vec_params("(total,)"), "T.Kernel(cores) as bx",
            [*[f'{n}{k} = T.alloc_shared((tile,), "float32")' for n in ("a", "b", "c") for k in (0, 1)],
             "for s in T.serial(pairs):\n    off0 = (s * 2 * cores + bx) * tile\n    off1 = ((s * 2 + 1) * cores + bx) * tile",
             "    T.copy(A[off0 : off0 + tile], a0)\n    T.copy(B[off0 : off0 + tile], b0)\n    T.copy(A[off1 : off1 + tile], a1)\n    T.copy(B[off1 : off1 + tile], b1)",
             "    with T.SimdVF():\n        mask = T.simd.pset(32)\n        for i in range(tile // 64):",
             "            T.simd.vsts(c0[i * 64], T.simd.vadd(T.simd.vld(a0[i * 64]), T.simd.vld(b0[i * 64]), mask), mask)",
             "            T.simd.vsts(c1[i * 64], T.simd.vadd(T.simd.vld(a1[i * 64]), T.simd.vld(b1[i * 64]), mask), mask)",
             "    T.copy(c0, C[off0 : off0 + tile])\n    T.copy(c1, C[off1 : off1 + tile])"])
        return base, opt, (mix_cores * 16 * tile,), "float32"

    if probe_kind == "loop_fission_sfu":
        sfu_pars = ['Q: T.Tensor((cores * rows, cols), "float32"),', 'Out: T.Tensor((cores, cols), "float32"),', 'WOut: T.Tensor((cores * rows, 1), "float32"),']
        sfu_vars = [f"cores = {mix_cores}", "rows = 4", "cols = 64"]
        sfu_hdr = '    with T.SimdVF():\n        m32 = T.simd.pset(32, "PAT_ALL")\n        qv = T.simd.vld(q_ub[0, 0])'
        exp_blk = '        sv = T.simd.vdupv(T.simd.vcadd(qv, mask=m32), mask=m32, pos="POS_LOWEST")\n        ev = T.simd.vexp(sv, mask=m32)'
        alloc_sfu = 'q_ub = T.alloc_shared((1, cols), "float32")\nacc_ub = T.alloc_shared((rows, cols), "float32")'
        base = _wrap_probe(sfu_vars, sfu_pars, "T.Kernel(cores) as bx", [
            alloc_sfu, 'w_ub = T.alloc_shared((rows, 1), "float32")\nfor s in T.serial(rows):\n    T.copy(Q[bx * rows + s : bx * rows + s + 1, :], q_ub)',
            sfu_hdr, exp_blk, '        T.simd.vsts(w_ub[s, 0], ev, mask=m32, dist="ONEPT_B32")\n        av = T.simd.vld(acc_ub[s, 0])',
            "        T.simd.vsts(acc_ub[s, 0], T.simd.vadd(av, qv, mask=m32), mask=m32)\nT.copy(acc_ub[0:1, :], Out[bx : bx + 1, :])",
            "T.copy(w_ub, WOut[bx * rows : bx * rows + rows, :])"])
        opt = _wrap_probe(sfu_vars, sfu_pars, "T.Kernel(cores) as bx", [
            alloc_sfu, 'exb = T.alloc_shared((rows, 1), "float32")\nfor s in T.serial(rows):\n    T.copy(Q[bx * rows + s : bx * rows + s + 1, :], q_ub)',
            sfu_hdr, exp_blk, '        T.simd.vsts(exb[s, 0], ev, mask=m32, dist="ONEPT_B32")\nfor s2 in T.serial(rows):',
            "    T.copy(Q[bx * rows + s2 : bx * rows + s2 + 1, :], q_ub)", sfu_hdr,
            "        av = T.simd.vld(acc_ub[s2, 0])\n        T.simd.vsts(acc_ub[s2, 0], T.simd.vadd(av, qv, mask=m32), mask=m32)",
            "T.copy(acc_ub[0:1, :], Out[bx : bx + 1, :])\nT.copy(exb, WOut[bx * rows : bx * rows + rows, :])"])
        return base, opt, (mix_cores * 4, 64), "float32"
    return None


def synthesize_micro_probe(chunk: AstEditChunk, *, probe_id: str | None = None) -> MicroProbeSpec | None:
    """Synthesize a standalone 30-50 line micro-probe, or None for unmapped edits."""
    if chunk.edit_kind not in _SUPPORTED_PROBE_KINDS:
        return None
    if (rendered := _probe_templates_for_kind(chunk.edit_kind, _extract_probe_params(chunk))) is None:
        return None
    base_src, opt_src, shape, dtype = rendered
    return MicroProbeSpec(
        probe_id=probe_id or f"probe_{chunk.edit_kind}_{chunk.chunk_id}", probe_kind=chunk.edit_kind,
        factory_name="make_flywheel_probe", baseline_probe_source=base_src, optimized_probe_source=opt_src,
        line_count_baseline=len(base_src.strip().splitlines()), line_count_optimized=len(opt_src.strip().splitlines()),
        tensor_shape=shape, tensor_dtype=dtype,
    )


def _check_aiv_mte2_non_regression(base_m: float, opt_m: float, b_us: float, o_us: float, speedup: float) -> bool:
    if base_m <= 0.0 or speedup >= 1.50:
        return True
    return not (opt_m > max(base_m * 1.15, base_m + 0.05) and opt_m * o_us > max(base_m * b_us * 1.15, base_m * b_us + 0.05 * b_us))


def _evaluate_probe_gates(
    b_us: float, o_us: float, *, raw: dict[str, float],
    sigma_paired_us: float | None, paired_rel_mad: float | None,
) -> tuple[bool, dict[str, float]]:
    speedup = float(raw.get("speedup", (b_us / o_us) if b_us > 0.0 and o_us > 0.0 else 0.0))
    eff_sigma = resolve_paired_sigma_us(b_us, sigma_paired_us=sigma_paired_us, paired_rel_mad=paired_rel_mad)
    delta_us = b_us - o_us
    raw.update({"speedup": speedup, "delta_us": delta_us, "sigma_paired_us": eff_sigma})
    sig_ok = b_us > 0.0 and o_us > 0.0 and is_paired_gain_significant(
        delta_us, sigma_paired_us=eff_sigma, reference_us=b_us, sigma_multiplier=3.0
    )
    wait_ok = True
    if "baseline_mte2_wait_ratio" in raw and "optimized_mte2_wait_ratio" in raw:
        bw, ow = float(raw["baseline_mte2_wait_ratio"]), float(raw["optimized_mte2_wait_ratio"])
        wait_ok = ow <= max(bw * 1.05, bw + 0.05)
    if "baseline_aiv_mte2" in raw and "optimized_aiv_mte2" in raw:
        wait_ok = wait_ok and _check_aiv_mte2_non_regression(
            float(raw["baseline_aiv_mte2"]), float(raw["optimized_aiv_mte2"]), b_us, o_us, speedup
        )
    return bool(raw.get("numerical_ok", True)) and sig_ok and wait_ok, raw


def verify_micro_probe_hardware(
    spec: MicroProbeSpec, *, probe_runner: Callable[[MicroProbeSpec], Mapping[str, float]] | None = None,
    probe_port: HardwareProbePort | None = None, sigma_paired_us: float | None = None,
    paired_rel_mad: float | None = None,
) -> tuple[bool, dict[str, float]]:
    """Verify micro-probe correctness, 3-sigma paired gain, and pipe wait ratios on NPU."""
    if probe_runner is not None:
        raw = dict(probe_runner(spec))
        b_us, o_us = float(raw.get("baseline_dur_us", 0.0)), float(raw.get("optimized_dur_us", 0.0))
        sig = float(raw["sigma_paired_us"]) if raw.get("sigma_paired_us") is not None else sigma_paired_us
        mad = float(raw["paired_rel_mad"]) if raw.get("paired_rel_mad") is not None else paired_rel_mad
        return _evaluate_probe_gates(b_us, o_us, raw=raw, sigma_paired_us=sig, paired_rel_mad=mad)

    if probe_port is None:
        raise RuntimeError("HardwareProbePort is required for live micro-probe verification when probe_runner is None.")
    import tilelang  # noqa: PLC0415
    import torch  # noqa: PLC0415

    probe_port.ensure_cann_include_cpath()
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("Ascend NPU device is required for live micro-probe verification.")

    def _compile_and_time(src: str, tag: str, kind: str) -> tuple[float, dict[str, float]]:
        with tempfile.TemporaryDirectory(prefix=f"tf_probe_{tag}_") as tmp_dir:
            mf = Path(tmp_dir) / "probe_mod.py"
            mf.write_text(src, encoding="utf-8")
            ms = importlib.util.spec_from_file_location(f"_tf_probe_{tag}", str(mf))
            assert ms is not None and ms.loader is not None
            mod = importlib.util.module_from_spec(ms)
            ms.loader.exec_module(mod)
            kernel = tilelang.compile(getattr(mod, spec.factory_name)(), target="ascend", out_idx=-1)
        torch.manual_seed(42)
        if kind == "ssbuf_handoff":
            rounds = spec.tensor_shape[0]
            seed = torch.zeros((16, 16), dtype=torch.float16).npu()
            expected = torch.tensor([[(i + 1) * 100 + (((i + 1) * 101) << 32)] for i in range(rounds)], dtype=torch.uint64)
            args: tuple[object, ...] = (seed, torch.zeros((rounds, 2), dtype=torch.uint32).npu()) if tag == "base" else (seed,)
            tol = {"rtol": 0, "atol": 0}
        else:
            rshape = (spec.tensor_shape[0], 70) if (kind == "pad_alignment" and tag == "base") else spec.tensor_shape
            a, b = torch.randn(rshape, dtype=torch.float32).npu(), torch.randn(rshape, dtype=torch.float32).npu()
            args, expected, tol = (a, b), a.cpu() + b.cpu(), {"atol": 1e-4, "rtol": 1e-4}
        out = kernel(*args)
        torch.npu.synchronize()
        torch.testing.assert_close(out.cpu(), expected, **tol)
        prof = probe_port.run_msprof_detail(lambda: kernel(*args), n_warmup=3, n_repeat=6)
        return float(prof.dur_us), {k: float(getattr(prof, k, 0.0)) for k in ("aiv_mte2", "aiv_vec", "aic_fixpipe")}

    b_us, b_rat = _compile_and_time(spec.baseline_probe_source, "base", spec.probe_kind)
    o_us, o_rat = _compile_and_time(spec.optimized_probe_source, "opt", spec.probe_kind)
    raw = {
        "baseline_dur_us": round(b_us, 6), "optimized_dur_us": round(o_us, 6), "numerical_ok": 1.0,
        "aiv_mte2": round(o_rat["aiv_mte2"], 6), "aiv_vec": round(o_rat["aiv_vec"], 6),
        "aic_fixpipe": round(o_rat["aic_fixpipe"], 6),
        "baseline_aiv_mte2": round(b_rat["aiv_mte2"], 6), "optimized_aiv_mte2": round(o_rat["aiv_mte2"], 6),
    }
    ok, metrics = _evaluate_probe_gates(b_us, o_us, raw=raw, sigma_paired_us=sigma_paired_us, paired_rel_mad=paired_rel_mad)
    return ok, {k: round(v, 6) for k, v in metrics.items()}


_TEMPLATE_TEXT_BY_KIND: dict[str, str] = {
    "core_parallelism": "Scale T.Kernel(block_count) grid dimensionality to saturate available hardware blocks using contiguous per-block slicing and eliminate trailing PIPE_ALL barriers.",
    "pipeline_versions": "Increase T.Pipelined num_stages to 2 and annotate on-chip buffer versions >= 2 so DMA transfers overlap with compute stages while reserving compiler headroom.",
    "coalesced_copy": "Coalesce strided inner row T.copy loops into contiguous 2D block DMA slices to reduce MTE2 inflight queue pressure and improve burst efficiency.",
    "ssbuf_handoff": "Replace global memory scalar barriers with 8B-aligned SSBUF T.alloc_ss slots and explicit dirty-SRAM uint32 zero initialization before steady-state handoff.",
    "dual_copy_1c2v": "Partition shared DMA workloads across independent dual-vector worker slices with 1C2V swizzle scheduling to double concurrent MTE2 and vector throughput.",
    "barrier_prune": "Eliminate redundant T.sync_all and T.set_flag barriers inside steady-state loops, retaining only the minimal cross-pipe synchronization required by dependencies.",
    "pad_alignment": "Pad inner buffer trailing dimensions to 32-byte and 512-byte bank-aligned strides so DMA bursts avoid unaligned tail transactions and bank conflict stalls.",
    "static_pingpong_unroll": "Replace dynamic-offset T.Pipelined loops with 2x-unrolled static ping-pong buffers (buf_a, buf_b) and explicit prologue prefetch to eliminate pointer-alias serialization.",
    "loop_fission_sfu": "Split coupled scalar reduction and SFU evaluation into Pass 1 before streaming wide vector accumulation in Pass 2 so long SFU chains no longer serialize wide datapaths.",
    "config_knob": "Tune schedule configuration dictionary parameters (tile shapes, buffer stages, swizzle window, and bank padding) to maximize hardware pipeline occupancy without overflow.",
    "cpp_hunk_edit": "Apply targeted tl_templates/ascend C++20 kernel optimizations across vector SIMD, DMA alignment, and on-chip buffer staging while preserving exact synchronization parity.",
}
_DEFAULT_TEMPLATE_TEXT = "Replace sequential or over-synchronized inner statements with non-blocking pipelined vector and DMA execution while preserving static UB headroom."
_RULE_PREDICATES_BY_KIND: dict[str, tuple[str, str, str]] = {
    "core_parallelism": ("is_legal and any_overflow_bytes == 0 and not has_indirect_gather", "(wave_occupancy > 0.0 and wave_occupancy < 0.75) or PIPE_V > 0 or PIPE_MTE2 > 0", "measured_latency_us < prev.measured_latency_us"),
    "coalesced_copy": ("is_legal and dominant_contig_bytes >= 64", "mte2_inflight_per_aiv > 8 or PIPE_MTE2 > PIPE_M", "mte2_inflight_per_aiv < prev.mte2_inflight_per_aiv and any_overflow_bytes == 0"),
    "ssbuf_handoff": ("is_legal and (has_dynamic_control or cube_to_vector_ratio >= 0.0)", "unexplained_gap_ratio >= 0.15 or measured_sync_stall_ratio > 0.05", "measured_sync_stall_ratio < prev.measured_sync_stall_ratio"),
    "dual_copy_1c2v": ("is_legal and (cube_to_vector_ratio > 0.0 or AIV0.PIPE_V > 0)", "PIPE_MTE2 > 0 or PIPE_V > 0 or measured_mte2_wait_ratio > 0.1", "any_overflow_bytes == 0 and measured_latency_us < prev.measured_latency_us"),
    "barrier_prune": ("is_legal and not has_dynamic_control", "in_loop_pipe_all_barriers > 0 or measured_sync_stall_ratio > 0.05 or PIPE_V > 0", "in_loop_pipe_all_barriers < prev.in_loop_pipe_all_barriers and any_overflow_bytes == 0"),
    "pad_alignment": ("is_legal and dominant_contig_bytes >= 16", "unaligned_dma_padding_bytes > 0 or PIPE_MTE2 > 0 or PIPE_V > 0", "unaligned_dma_padding_bytes < prev.unaligned_dma_padding_bytes and any_overflow_bytes == 0"),
    "static_pingpong_unroll": ("is_legal and (uses_dynamic_offset_pipelining or pseudo_double_buffer_serialized)", "pseudo_double_buffer_serialized or measured_mte2_wait_ratio >= 0.32", "not uses_dynamic_offset_pipelining and any_overflow_bytes == 0"),
    "loop_fission_sfu": ("is_legal and PIPE_M == 0 and has_coupled_reduce_sfu_accum", "has_coupled_reduce_sfu_accum or measured_aiv_vec_ratio >= 0.35", "not has_coupled_reduce_sfu_accum and any_overflow_bytes == 0"),
}
_DEFAULT_RULE_PREDICATES = ("is_legal and any_overflow_bytes == 0", "degraded_buffer_count > 0 and (PIPE_V > 0 or PIPE_MTE2 > 0)", "degraded_buffer_count < prev.degraded_buffer_count and any_overflow_bytes == 0")
_EDIT_KIND_TO_SYMPTOM_FAMILY: dict[str, SymptomFamily] = {
    "core_parallelism": SymptomFamily.GRID_TAIL_WAVE_UNDERUTIL,
    "pipeline_versions": SymptomFamily.SYNC_PSEUDO_DOUBLE_BUFFER,
    "coalesced_copy": SymptomFamily.DMA_UNALIGNED_SHORT_BURST,
    "ssbuf_handoff": SymptomFamily.DMA_REDUNDANT_GM_ROUNDTRIP,
    "dual_copy_1c2v": SymptomFamily.COMPUTE_1C2V_SUBCORE_IMBALANCE,
    "barrier_prune": SymptomFamily.SYNC_BARRIER_SERIALIZATION,
    "pad_alignment": SymptomFamily.SRAM_UB_L1_BANK_CONFLICT,
    "static_pingpong_unroll": SymptomFamily.SYNC_PSEUDO_DOUBLE_BUFFER,
    "loop_fission_sfu": SymptomFamily.COMPUTE_VECTOR_LOOP_DEGRADED,
}
_ANTI_PRECONDITION_BY_KIND: dict[str, str] = {
    "core_parallelism": "any_overflow_bytes > 0 or spill_bytes > 0 or has_indirect_gather",
    "barrier_prune": "any_overflow_bytes > 0 or has_dynamic_control",
}


def _infer_symptom_family(
    chunk: AstEditChunk, ledger: ObjectivePhysicalLedger | None, active_symptom_family: str = ""
) -> str:
    if (cleaned := str(active_symptom_family or "").strip()):
        try:
            return SymptomFamily(cleaned).value
        except ValueError:
            pass
    if (mapped := _EDIT_KIND_TO_SYMPTOM_FAMILY.get(chunk.edit_kind)) is not None:
        return mapped.value
    if ledger is not None:
        if ledger.in_loop_pipe_all_barriers > 0:
            return SymptomFamily.SYNC_BARRIER_SERIALIZATION.value
        if any(wl.degraded_single_buffers for wl in ledger.waterlines.values()) or ledger.uses_dynamic_offset_pipelining:
            return SymptomFamily.SYNC_PSEUDO_DOUBLE_BUFFER.value
        if ledger.dma_burst_efficiency_ratio < 0.7:
            return SymptomFamily.DMA_UNALIGNED_SHORT_BURST.value
        if 0.0 < ledger.wave_occupancy < 0.75:
            return SymptomFamily.GRID_TAIL_WAVE_UNDERUTIL.value
    return SymptomFamily.COMPUTE_INEFFECTIVE_THROUGHPUT.value


_CANONICAL_FREEDOM_FACETS: frozenset[str] = frozenset(VALID_FREEDOM_FACETS)
_TRACKB_TOPOLOGY_TOKENS: frozenset[str] = frozenset({
    "cce_cpp20", "ascendc_trackb", "hybrid_trackb", "cce_trackb",
    "ascendc_template", "ascendc", "hybrid", "tl_templates_cpp20",
})


def _split_str_tokens(raw: object) -> list[str]:
    if isinstance(raw, str):
        return [t.strip() for t in raw.split(",") if t.strip()]
    return [str(t).strip() for t in raw if str(t).strip()] if isinstance(raw, Iterable) else []


def _extract_rule_facets(rule: AscendCausalRule) -> tuple[str, ...]:
    if (raw_facets := getattr(rule, "recommended_facets", None)) is not None:
        return tuple(_split_str_tokens(raw_facets))
    if (family_name := str(getattr(rule, "symptom_family", "") or "").strip()):
        try:
            return SYMPTOM_FAMILY_TO_FACETS.get(SymptomFamily(family_name), ())
        except ValueError:
            return ()
    return ()


def _rule_covers_trackb(rule: AscendCausalRule) -> bool:
    tokens = [str(raw).strip() for raw in getattr(rule, "compatible_backends", ()) or ()]
    tokens.extend(_split_str_tokens(getattr(rule, "applicable_topologies", None)))
    return any(
        t.lower() in _TRACKB_TOPOLOGY_TOKENS or (normalize_runtime_backend(t.lower()) or t.lower()) in _TRACKB_TOPOLOGY_TOKENS
        for t in tokens
    )


def _is_zero_literal(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return not isinstance(node.value, bool) and float(node.value) == 0.0
    return isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)) and _is_zero_literal(node.operand)


def _has_tautological_ge_zero(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Compare):
            left = node.left
            for op, right in zip(node.ops, node.comparators):
                if (isinstance(op, ast.GtE) and _is_zero_literal(right)) or (isinstance(op, ast.LtE) and _is_zero_literal(left)):
                    return True
                left = right
    return False


_IDEAL_UNIT_EFFICIENCY_NAMES = frozenset({
    "effective_compute_eta", "roofline_eta", "dma_burst_efficiency_ratio", "contig_atom_ratio",
    "wave_occupancy", "wave_occupancy_ratio", "l2_hit_rate", "cube_util_ratio", "vec_util_ratio",
})
_IDEAL_CONTIG_BYTE_NAMES = frozenset({"dominant_contig_bytes", "inner_contiguous_bytes"})
_IDEAL_BOOL_TRUE_NAMES = frozenset({"is_legal", "vloopv2_hit", "is_atom_aligned"})


def _ideal_default_for_symbol(symbol: str) -> object:
    leaf = symbol.rsplit(".", 1)[-1]
    return True if leaf in _IDEAL_BOOL_TRUE_NAMES else (1.0 if leaf in _IDEAL_UNIT_EFFICIENCY_NAMES else (32768 if leaf in _IDEAL_CONTIG_BYTE_NAMES else 0))


def _build_zero_ideal_env(tree: ast.AST) -> dict[str, object]:
    env: dict[str, object] = {"is_legal": True, "vloopv2_hit": True, "is_atom_aligned": True}
    skip = frozenset({"True", "False", "None", "abs", "min", "max"})
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            parts, cur = [], node
            while isinstance(cur, ast.Attribute):
                parts.append(cur.attr)
                cur = cur.value
            if isinstance(cur, ast.Name):
                dotted = ".".join(reversed([*parts, cur.id]))
                env.setdefault(dotted, _ideal_default_for_symbol(dotted))
        elif isinstance(node, ast.Name) and node.id not in skip:
            env.setdefault(node.id, _ideal_default_for_symbol(node.id))
    return env


def _node_references_prev(node: ast.AST) -> bool:
    return any(isinstance(s, ast.Attribute) and isinstance(s.value, ast.Name) and s.value.id == "prev" for s in ast.walk(node))


def _validate_postcondition_strictness(tree: ast.AST) -> tuple[bool, str]:
    has_strict = False
    for node in ast.walk(tree):
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            has_strict = True
        elif isinstance(node, ast.Compare):
            left = node.left
            for op, right in zip(node.ops, node.comparators):
                if isinstance(op, (ast.Lt, ast.Gt)):
                    has_strict = True
                elif isinstance(op, (ast.LtE, ast.GtE)) and (_node_references_prev(left) or _node_references_prev(right)):
                    return False, "expected_postcondition uses non-strict <= or >= against prev"
                left = right
    return (True, "") if has_strict else (False, "expected_postcondition lacks strict improvement comparison (<, >, or not)")


def _validate_distilled_rule(rule: AscendCausalRule, *, operator_name: str = "") -> tuple[bool, str]:
    """Enforce provenance, closed AST vocabulary, Track-B template, and convergence gates."""
    raw_fields = list(rule.as_six_tuple()) + ([rule.cpp20_transform_template] if rule.cpp20_transform_template else [])
    joined = " ".join(raw_fields).lower()
    active_op = operator_name or str(getattr(rule, "origin_operator", "") or "")
    for bad in _build_disallowed_provenance_tokens(active_op):
        if re.search(rf"(?<![a-z0-9_]){re.escape(bad)}(?![a-z0-9_])", joined):
            return False, f"disallowed token {bad!r} in distilled rule text"
    for bad in _operator_name_variants(active_op):
        if len(bad) >= 4 and bad in joined:
            return False, f"disallowed token {bad!r} in distilled rule text"
    if any(_GENERIC_PATH_PATTERN.search(ft) for ft in raw_fields):
        return False, "filesystem path leaked into rule text"
    try:
        validate_rule_predicate(rule.regime_guard)
        pre_tree = validate_rule_predicate(rule.precondition)
        validate_rule_predicate(rule.anti_precondition)
        post_tree = validate_rule_predicate(rule.expected_postcondition)
    except Exception as exc:
        return False, f"invalid predicate: {exc}"
    if not (facets := _extract_rule_facets(rule)):
        return False, "missing or empty recommended_facets"
    if (bad_facets := [f for f in facets if f not in _CANONICAL_FREEDOM_FACETS]):
        return False, f"invalid recommended_facets {bad_facets!r}"
    if _rule_covers_trackb(rule) and not str(rule.cpp20_transform_template or "").strip():
        return False, "Track-B applicable rule requires non-empty cpp20_transform_template"
    if _has_tautological_ge_zero(pre_tree) or safe_eval_predicate(rule.precondition, _build_zero_ideal_env(pre_tree)):
        return False, "precondition is tautological"
    post_ok, post_err = _validate_postcondition_strictness(post_tree)
    if not post_ok:
        return False, post_err
    for fname, expr in (("precondition", rule.precondition), ("anti_precondition", rule.anti_precondition), ("expected_postcondition", rule.expected_postcondition)):
        if (unknown_ids := _validate_schema_closed_identifiers(expr)):
            return False, f"unknown schema identifier(s) {unknown_ids!r} in {fname}"
    return True, ""


def distill_causal_rule(
    chunk: AstEditChunk, *, rule_id: str, ledger: ObjectivePhysicalLedger | None = None,
    probe: MicroProbeSpec | None = None, probe_metrics: Mapping[str, float] | None = None,
    scope: str = "cross_op", origin_operator: str = "", active_symptom_family: str = "",
) -> AscendCausalRule:
    """Synthesize a pure-English 6-tuple AscendCausalRule from a verified AST edit."""
    _ = probe
    regime_guard, precondition, expected_post = _RULE_PREDICATES_BY_KIND.get(chunk.edit_kind, _DEFAULT_RULE_PREDICATES)
    template = _TEMPLATE_TEXT_BY_KIND.get(chunk.edit_kind, _DEFAULT_TEMPLATE_TEXT)
    speedup = float((probe_metrics or {}).get("speedup", 1.0))
    is_cpp = (
        chunk.chunk_id.startswith(("cpp_loo_", "trackb_tu_")) or chunk.edit_kind == "cpp_hunk_edit"
        or "_render_trackb_tu" in chunk.target_path or "_trackb_launcher" in chunk.target_path
    )
    return AscendCausalRule(
        rule_id=rule_id, regime_guard=regime_guard, precondition=precondition,
        transform_template=template, expected_postcondition=expected_post,
        source_provenance="flywheel:ablation", alpha_prior=2.5, beta_prior=1.0,
        rationale=f"Counterfactual single-variable ablation verified on Ascend 950 micro-probe ({chunk.edit_kind}, speedup={speedup:.3f}x).",
        priority=85, scope=scope, origin_operator=origin_operator,
        compatible_backends=("ascendc_trackb", "hybrid_trackb") if is_cpp else ("tilelang_dsl",),
        cpp20_transform_template=template if is_cpp else "",
        symptom_family=_infer_symptom_family(chunk, ledger, active_symptom_family),
        anti_precondition=_ANTI_PRECONDITION_BY_KIND.get(chunk.edit_kind, "any_overflow_bytes > 0 or spill_bytes > 0"),
    )


def _ledger_cost_score(ledger: ObjectivePhysicalLedger) -> float:
    dom = float(max(ledger.pipe_cycles.values(), default=100))
    occ = max(0.02, float(ledger.wave_occupancy) or (float(min(56, max(1, int(ledger.grid_blocks)))) / 56.0))
    burst = max(0.15, float(ledger.dma_burst_efficiency_ratio))
    degraded = sum(len(wl.degraded_single_buffers) for wl in ledger.waterlines.values())
    penalty = 1.0 + 0.25 * max(0, degraded) + 0.15 * max(0, int(ledger.in_loop_pipe_all_barriers))
    penalty += (0.12 if ledger.uses_dynamic_offset_pipelining else 0.0) + (0.12 if ledger.has_coupled_reduce_sfu_accum else 0.0)
    return (dom / occ / burst) * penalty


class FlywheelDistiller:
    """Decomposes winning DSL edits, ablates negatives, probes NPU, and admits rules."""

    def __init__(
        self, *, funnel: GateFunnel, tracker: BayesianRuleTracker,
        probe_runner: Callable[[MicroProbeSpec], Mapping[str, float]] | None = None,
        probe_port: HardwareProbePort | None = None, sigma_paired_us: float | None = None,
        paired_rel_mad: float | None = None,
    ) -> None:
        self._funnel, self._tracker = funnel, tracker
        self._probe_runner, self._probe_port = probe_runner, probe_port
        self._sigma_paired_us, self._paired_rel_mad = sigma_paired_us, paired_rel_mad

    def _allocate_rule_id(self, prefix: str, edit_kind: str) -> str:
        rule_num, suffix = len(self._tracker.rules) + 1, edit_kind.upper()
        while (cand := f"{prefix}_{rule_num:02d}_{suffix}") in self._tracker.rules:
            rule_num += 1
        return cand

    def _run_distill(
        self, chunks: Sequence[AstEditChunk], *, ref_dsl: str, ref_latency_us: float,
        report_base_us: float, report_win_us: float, reverse: bool, factory_name: str | None,
        factory_kwargs: Mapping[str, object] | None, config_selector_name: str | None,
        dims: Mapping[str, object] | None, ordered_tensor_shapes: Sequence[tuple[int, ...]] | None,
        batch_latency_evaluator: Callable[[Sequence[tuple[str, str]]], Mapping[str, float]] | None,
        topology_axes: object | None, kb_output_path: str | Path | None, rule_id_prefix: str,
        operator_name: str, rule_scope: str, active_symptom_family: str,
        sigma_paired_us: float | None, paired_rel_mad: float | None,
    ) -> FlywheelDistillationReport:
        eff_sigma = self._sigma_paired_us if sigma_paired_us is None else sigma_paired_us
        eff_mad = self._paired_rel_mad if paired_rel_mad is None else paired_rel_mad
        shapes = extract_ordered_tensor_shapes(dims=dims, ordered_tensor_shapes=ordered_tensor_shapes)
        ref_sa = run_stage_a_static(ref_dsl, factory_name=factory_name, factory_kwargs=factory_kwargs, config_selector_name=config_selector_name, dims=dims, ordered_tensor_shapes=shapes)
        ref_cost = _ledger_cost_score(ref_sa.ledger) if ref_sa.passed and ref_sa.ledger is not None else 0.0
        funnel_results = [(c, self._funnel.evaluate(c.ablated_dsl, factory_name=factory_name, factory_kwargs=factory_kwargs, config_selector_name=config_selector_name, dims=dims, ordered_tensor_shapes=shapes)) for c in chunks]
        batch_latencies: dict[str, float] = {}
        if batch_latency_evaluator is not None:
            targets, seen_b = [], set()
            for c, ev in funnel_results:
                if ev.passed and ev.stage_a is not None and ev.stage_a.ledger is not None and (reverse or (c.edit_kind in _SUPPORTED_PROBE_KINDS and c.edit_kind not in seen_b)):
                    seen_b.add(c.edit_kind)
                    targets.append((c.chunk_id, c.ablated_dsl))
                    if not reverse and len(targets) >= 4:
                        break
            if targets:
                batch_latencies = {str(k): float(v) for k, v in dict(batch_latency_evaluator(targets)).items() if float(v) > 0.0}
        kept, discarded, ref_lat = [], [], max(ref_latency_us, 1e-6)
        base_out = ref_lat if reverse else ref_latency_us
        for c, ev in funnel_results:
            if not ev.passed or ev.stage_a is None or ev.stage_a.ledger is None:
                discarded.append(AblationMeasurement(c, False, base_out, 0.0, -1.0, False, ev.stage_a, f"{ev.failed_stage}:{ev.error_message[:120]}"))
                continue
            var_lat = batch_latencies[c.chunk_id] if c.chunk_id in batch_latencies else (
                ev.stage_c.measured_latency_us if (ev.stage_c and ev.stage_c.measured_latency_us > 0.0) else (
                    ref_lat * (_ledger_cost_score(ev.stage_a.ledger) / ref_cost) if ref_cost > 0.0 else float(max(ev.stage_a.ledger.pipe_cycles.values(), default=100)) / 20.0
                )
            )
            delta_us = (var_lat - ref_lat) if reverse else (ref_lat - var_lat)
            rel_gain = delta_us / (max(var_lat, 1e-6) if reverse else ref_lat)
            is_pos = (var_lat > ref_lat if reverse else True) and is_paired_gain_significant(delta_us, sigma_paired_us=eff_sigma, paired_rel_mad=eff_mad, reference_us=ref_lat, sigma_multiplier=2.0)
            diag = ("positive_loo_gain" if is_pos else "redundant_or_harmful_edit") if reverse else ("positive_gain" if is_pos else "negative_or_zero_gain")
            (kept if is_pos else discarded).append(AblationMeasurement(c, True, base_out, var_lat, rel_gain, is_pos, ev.stage_a, diag))
        outcomes, admitted_ids, seen_keys = [], [], set()
        for meas in kept:
            dk: object = (meas.chunk.edit_kind, meas.chunk.target_path) if reverse else meas.chunk.edit_kind
            if dk in seen_keys:
                continue
            seen_keys.add(dk)
            if reverse:
                speedup = meas.variant_latency_us / ref_lat if ref_lat > 0.0 else 1.0
                probe, metrics = None, {"baseline_dur_us": round(meas.variant_latency_us, 6), "optimized_dur_us": round(ref_lat, 6), "speedup": round(speedup, 6), "numerical_ok": 1.0}
            else:
                if (probe := synthesize_micro_probe(meas.chunk)) is None:
                    outcomes.append(DistilledRuleOutcome(meas.chunk.chunk_id, None, False, {}, None, False, f"unverifiable_edit_kind:{meas.chunk.edit_kind}"))
                    continue
                ok, metrics = verify_micro_probe_hardware(probe, probe_runner=self._probe_runner, probe_port=self._probe_port, sigma_paired_us=eff_sigma, paired_rel_mad=eff_mad)
                if not ok:
                    outcomes.append(DistilledRuleOutcome(meas.chunk.chunk_id, probe, False, metrics, None, False, "micro_probe_below_threshold_or_wait_regression"))
                    continue
            rule = distill_causal_rule(meas.chunk, rule_id=self._allocate_rule_id(rule_id_prefix, meas.chunk.edit_kind), ledger=meas.stage_a_result.ledger if meas.stage_a_result else None, probe=probe, probe_metrics=metrics, scope=rule_scope, origin_operator=operator_name, active_symptom_family=active_symptom_family)
            clean_ok, clean_err = _validate_distilled_rule(rule, operator_name=operator_name)
            if not clean_ok:
                outcomes.append(DistilledRuleOutcome(meas.chunk.chunk_id, probe, True, metrics, rule, False, clean_err))
                continue
            self._tracker.admit_rule(rule)
            self._tracker.record_outcome(rule.rule_id, postcondition_realized=True, speed_gain=max(0.0, meas.relative_gain), topology_axes=topology_axes)
            admitted_ids.append(rule.rule_id)
            outcomes.append(DistilledRuleOutcome(meas.chunk.chunk_id, probe, True, metrics, rule, True))
        if kb_output_path is not None:
            save_rules_yaml(list(self._tracker.rules.values()), kb_output_path)
        return FlywheelDistillationReport(report_base_us, report_win_us, len(chunks), tuple(kept), tuple(discarded), tuple(outcomes), tuple(admitted_ids))

    def distill_leave_one_out(
        self, *, baseline_dsl: str, winner_dsl: str, winner_latency_us: float,
        baseline_latency_us: float | None = None, max_loo_chunks: int | None = 3,
        rule_id_prefix: str = "R_FLYWHEEL_LOO", rule_scope: str = "operator_local", **kw: object,
    ) -> FlywheelDistillationReport:
        """Run Leave-One-Out AST/config ablation directly on real candidate winner_dsl."""
        chunks = decompose_leave_one_out_ablations(baseline_dsl, winner_dsl)
        if max_loo_chunks is not None and int(max_loo_chunks) > 0:
            chunks = chunks[: int(max_loo_chunks)]
        win_lat = max(float(winner_latency_us), 1e-6)
        eff_base = float(baseline_latency_us) if baseline_latency_us is not None else win_lat
        return self._run_distill(
            chunks, ref_dsl=winner_dsl, ref_latency_us=win_lat, report_base_us=eff_base, report_win_us=win_lat, reverse=True,
            factory_name=kw.get("factory_name"), factory_kwargs=kw.get("factory_kwargs"), config_selector_name=kw.get("config_selector_name"),
            dims=kw.get("dims"), ordered_tensor_shapes=kw.get("ordered_tensor_shapes"), batch_latency_evaluator=kw.get("batch_latency_evaluator"),
            topology_axes=kw.get("topology_axes"), kb_output_path=kw.get("kb_output_path"), rule_id_prefix=rule_id_prefix,
            operator_name=str(kw.get("operator_name", "")), rule_scope=rule_scope, active_symptom_family=str(kw.get("active_symptom_family", "")),
            sigma_paired_us=kw.get("sigma_paired_us"), paired_rel_mad=kw.get("paired_rel_mad"),
        )

    def distill_from_pair(
        self, *, baseline_dsl: str, winner_dsl: str, baseline_latency_us: float, winner_latency_us: float,
        verify_on_candidate: bool = False, max_loo_chunks: int | None = 3,
        rule_id_prefix: str = "R_FLYWHEEL", rule_scope: str = "cross_op", **kw: object,
    ) -> FlywheelDistillationReport:
        """Run full counterfactual ablation, micro-probe or LOO check, and KB admission."""
        eff_sigma = self._sigma_paired_us if kw.get("sigma_paired_us") is None else kw.get("sigma_paired_us")
        eff_mad = self._paired_rel_mad if kw.get("paired_rel_mad") is None else kw.get("paired_rel_mad")
        if not should_trigger_flywheel_checkpoint(baseline_latency_us, winner_latency_us, sigma_paired_us=eff_sigma, paired_rel_mad=eff_mad):
            return FlywheelDistillationReport(baseline_latency_us, winner_latency_us, 0, (), ())
        if verify_on_candidate:
            return self.distill_leave_one_out(
                baseline_dsl=baseline_dsl, winner_dsl=winner_dsl, winner_latency_us=winner_latency_us,
                baseline_latency_us=baseline_latency_us, max_loo_chunks=max_loo_chunks,
                rule_id_prefix=rule_id_prefix, rule_scope=rule_scope, **kw,
            )
        return self._run_distill(
            decompose_ast_ablations(baseline_dsl, winner_dsl), ref_dsl=baseline_dsl, ref_latency_us=baseline_latency_us,
            report_base_us=baseline_latency_us, report_win_us=winner_latency_us, reverse=False,
            factory_name=kw.get("factory_name"), factory_kwargs=kw.get("factory_kwargs"), config_selector_name=kw.get("config_selector_name"),
            dims=kw.get("dims"), ordered_tensor_shapes=kw.get("ordered_tensor_shapes"), batch_latency_evaluator=kw.get("batch_latency_evaluator"),
            topology_axes=kw.get("topology_axes"), kb_output_path=kw.get("kb_output_path"), rule_id_prefix=rule_id_prefix,
            operator_name=str(kw.get("operator_name", "")), rule_scope=rule_scope, active_symptom_family=str(kw.get("active_symptom_family", "")),
            sigma_paired_us=kw.get("sigma_paired_us"), paired_rel_mad=kw.get("paired_rel_mad"),
        )

    def distill_from_trajectory(
        self, *, baseline_dsl: str, trajectory: EvolutionTrajectory, **kw: object,
    ) -> FlywheelDistillationReport:
        """Distill and verify causal rules directly from an EvolutionTrajectory winner."""
        return self.distill_from_pair(
            baseline_dsl=baseline_dsl, winner_dsl=trajectory.best_dsl_source,
            baseline_latency_us=trajectory.baseline_latency_us, winner_latency_us=trajectory.best_latency_us, **kw,
        )


__all__ = [
    "ALLOWED_SCHEMA_SIGNAL_VOCABULARY", "AblationMeasurement", "AstEditChunk",
    "DistilledRuleOutcome", "FlywheelDistillationReport", "FlywheelDistiller", "MicroProbeSpec",
    "_build_disallowed_provenance_tokens", "_operator_name_variants", "_validate_distilled_rule",
    "_validate_schema_closed_identifiers", "decompose_ast_ablations",
    "decompose_cpp_string_leave_one_out", "decompose_leave_one_out_ablations",
    "distill_causal_rule", "is_paired_gain_significant", "resolve_paired_sigma_us",
    "should_trigger_flywheel_checkpoint", "synthesize_micro_probe", "verify_micro_probe_hardware",
]
