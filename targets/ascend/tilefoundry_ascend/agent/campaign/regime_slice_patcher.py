"""Regime config resolution, slice extraction, Track-B context view, and AST patching."""

from __future__ import annotations

import ast
import copy
import importlib.util
import re
import tempfile
import textwrap
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from tilefoundry_ascend.agent.orchestrator import cpp_slice_protocol as _cpp_slice_protocol_mod
from tilefoundry_ascend.agent.orchestrator.cpp_slice_protocol import (
    DEFAULT_PROTECTED_EXPORTS, _SEARCH_REPLACE_PATTERNS, TopLevelSymbol as _TopLevelSymbol,
    _char_span_to_line_span, _disambiguate_matches_by_line_window, _find_all_substring_spans,
    _is_bare_cpp_symbol_patch as _looks_like_bare_trackb_cpp, _parse_scoped_search_replace_hunks,
    _scoped_editable_units, _split_cpp_units, apply_scoped_trackb_patch, build_trackb_context_view,
    collect_referenced_helper_names as _collect_referenced_helper_names,
    collect_top_level_ast_nodes as _collect_top_level_ast_nodes,
    collect_top_level_symbols as _collect_top_level_symbols, extract_cpp_diagnostic_lines,
    filter_callable_kwargs as _filter_callable_kwargs, find_top_level_func as _find_top_level_func,
    find_top_level_str_assign_spans as _find_top_level_str_assign_span,
    format_cpp_string_literal as _format_raw_cpp_literal, normalized_line_matches as _normalized_line_matches,
    parse_scoped_search_replace_blocks, py_char_offset as _ast_char_offset,
    resolve_instantiated_trackb_kernel, scan_launched_trackb_kernels, select_active_trackb_kernel,
    splice_cpp_units_into_tu, split_tracka_editable_and_readonly_tail, stmt_line_span as _func_line_span,
)
from tilefoundry_ascend.agent.orchestrator.llm_client import extract_python_dsl
from tilefoundry_ascend.agent.specs.anchor_specs import (
    RegimeTargetSpec, extract_trackb_cpp_source, get_operator_entrypoint, parse_simple_env_file,
)
from tilefoundry_ascend.ascend.json_record import atomic_write_text as _atomic_write_text
from tilefoundry_ascend.ascend.sil.schema import CasePhysicsRatios

DEFAULT_READONLY_HOST_EXPORTS: tuple[str, ...] = (
    "kernel", "ref_kernel", "build_kernel", "_get_kernel", "_KERNEL_CACHE", "_ENTRY_CACHE",
)
_HIR_DIM_ALIASES: dict[str, str] = {
    "N": "n_depth", "B": "b_dim", "S": "s_dim", "S_kv": "s_kv", "D": "d_dim",
    "N_q": "n_q", "N_kv": "n_kv", "D_nope": "d_nope", "D_rope": "d_rope", "inner": "inner_dim",
}


@dataclass(frozen=True)
class ResolvedRegimeConfig:
    """Resolved builder, tile hyperparameters, and pass/compile flags for a regime."""

    builder_name: str
    builder_kwargs: dict[str, object]
    pass_configs: dict[str, object]
    compile_flags: tuple[str, ...]
    selector_name: str = ""
    combine_kwargs: dict[str, object] = field(default_factory=dict)


def _derive_dynamic_protected_symbols(
    module_source: str,
    regime_spec: RegimeTargetSpec,
    *,
    operator_entrypoint: str = "",
    sibling_specs: Sequence[RegimeTargetSpec] = (),
) -> set[str]:
    """Derive protected entrypoint, cache, and sibling symbols from AST and specs."""
    protected: set[str] = {"kernel"}
    if operator_entrypoint:
        protected.add(operator_entrypoint)
    if module_source.strip():
        try:
            tree = ast.parse(module_source)
        except SyntaxError:
            tree = None
        if tree is not None:
            top_funcs: dict[str, ast.FunctionDef] = {}
            for node in tree.body:
                if isinstance(node, ast.FunctionDef):
                    top_funcs[node.name] = node
                    if node.name == "_ascend_target_ctx" or node.name == "_get_kernel" or (node.name.startswith("_get_") and node.name.endswith("_kernel")):
                        protected.add(node.name)
                elif isinstance(node, ast.Assign):
                    protected.update(t.id for t in node.targets if isinstance(t, ast.Name) and t.id.endswith("_CACHE"))
                elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id.endswith("_CACHE"):
                    protected.add(node.target.id)
            for root_name in tuple(protected):
                if (fn := top_funcs.get(root_name)) is not None:
                    protected.update(
                        sub.func.id for sub in ast.walk(fn)
                        if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id in top_funcs
                    )
    own_editable = set(regime_spec.editable_symbols)
    own_builders = set(regime_spec.builder_candidates)
    for spec in sibling_specs:
        if spec.regime_tag == regime_spec.regime_tag:
            continue
        protected.update(s for s in spec.editable_symbols if s not in own_editable)
        if spec.config_selector_fn and spec.config_selector_fn not in own_editable:
            protected.add(spec.config_selector_fn)
        protected.update(c for c in spec.builder_candidates if c not in own_editable and c not in own_builders)
    protected.difference_update(own_editable)
    return protected


def _norm_flag_seq(raw: object) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return raw.strip().split() if raw.strip() else []
    if isinstance(raw, (tuple, list)) or (hasattr(raw, "__iter__") and not isinstance(raw, (bytes, bytearray, Mapping))):
        return list(dict.fromkeys(val for item in raw if (val := str(getattr(item, "value", item)).strip())))
    return [str(raw).strip()] if str(raw).strip() else []


def _load_module_and_run_selector(
    file_path: Path, regime_spec: RegimeTargetSpec, dims: Mapping[str, object] | None = None
) -> tuple[object, dict[str, object], dict[str, object], list[str], str, dict[str, object]]:
    spec = importlib.util.spec_from_file_location("_tf_resolve_cfg", str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module spec from {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    context: dict[str, object] = dict(regime_spec.sample_builder_kwargs)
    for h_key, h_val in regime_spec.representative_hir_dims.items():
        context.setdefault(_HIR_DIM_ALIASES.get(str(h_key), str(h_key)), h_val)
    for d_key, d_val in (dims or {}).items():
        context[_HIR_DIM_ALIASES.get(str(d_key), str(d_key))] = d_val
    if "half_dim" not in context and isinstance(raw_d := context.get("d_dim"), int) and not isinstance(raw_d, bool):
        context["half_dim"] = max(1, raw_d // 2)
    pass_configs, compile_flags, combine_kwargs = {}, [], {}

    def _add_flags(raw: object) -> None:
        compile_flags.extend(fl for fl in _norm_flag_seq(raw) if fl not in compile_flags)

    def _absorb(raw_pc: object) -> None:
        if isinstance(raw_pc, Mapping):
            for k, v in raw_pc.items():
                _add_flags(v) if str(k) == "tl.device_compile_flags" else pass_configs.__setitem__(str(k), v)
        elif raw_pc is not None:
            _add_flags(raw_pc)

    _absorb(context.pop("pass_configs", None))
    _add_flags(context.pop("compile_flags", None))
    selector_name = regime_spec.config_selector_fn
    if not (selector_name and callable(getattr(mod, selector_name, None))):
        selector_name = next((s for s in regime_spec.editable_symbols if s.startswith("_select_") and callable(getattr(mod, s, None))), selector_name)
    if selector_name and callable(sel_fn := getattr(mod, selector_name, None)) and isinstance(sel_out := sel_fn(**_filter_callable_kwargs(sel_fn, context)), Mapping):
        _absorb(sel_out.get("pass_configs"))
        _add_flags(sel_out.get("compile_flags"))
        for k, v in sel_out.items():
            if k not in ("pass_configs", "compile_flags", "builder_kwargs", "combine_kwargs"):
                context[str(k)] = v
        if isinstance(raw_bk := sel_out.get("builder_kwargs"), Mapping):
            context.update({str(k): v for k, v in raw_bk.items()})
        if isinstance(raw_ck := sel_out.get("combine_kwargs"), Mapping):
            combine_kwargs = {str(k): v for k, v in raw_ck.items()}
    return mod, context, pass_configs, compile_flags, selector_name, combine_kwargs


def _resolve_selector_compile_flags(
    module_source: str, regime_spec: RegimeTargetSpec, dims: Mapping[str, object] | None = None
) -> tuple[str, ...]:
    try:
        with tempfile.TemporaryDirectory(prefix="tf_resolve_flags_") as tmp_dir:
            tmp_file = Path(tmp_dir) / "module_src.py"
            tmp_file.write_text(module_source, encoding="utf-8")
            return tuple(_load_module_and_run_selector(tmp_file, regime_spec, dims=dims)[3])
    except (SyntaxError, AttributeError, ImportError, KeyError, NameError, OSError, RuntimeError, TypeError, ValueError):
        return ()


def resolve_regime_config(
    module_source_or_path: str | Path, regime_spec: RegimeTargetSpec, dims: Mapping[str, object] | None = None
) -> ResolvedRegimeConfig:
    """Invoke regime config_selector_fn dynamically to resolve builder kwargs and flags."""
    if isinstance(module_source_or_path, Path):
        mod_path: Path | None = module_source_or_path.resolve()
        src_text = mod_path.read_text(encoding="utf-8")
    else:
        raw_str = str(module_source_or_path)
        if "\n" not in raw_str and Path(raw_str).is_file():
            mod_path = Path(raw_str).resolve()
            src_text = mod_path.read_text(encoding="utf-8")
        else:
            mod_path, src_text = None, raw_str

    def _load_and_resolve(file_path: Path) -> ResolvedRegimeConfig:
        mod, ctx, pass_configs, compile_flags, sel_name, combine_kwargs = _load_module_and_run_selector(
            file_path, regime_spec, dims=dims
        )
        builder_name = next((c for c in regime_spec.builder_candidates if callable(getattr(mod, c, None))), "")
        if not builder_name:
            builder_name = next((c for c in regime_spec.editable_symbols if c != sel_name and callable(getattr(mod, c, None))), "")
        if not builder_name:
            raise ValueError(f"No builder callable matching {regime_spec.builder_candidates} in module.")
        return ResolvedRegimeConfig(
            builder_name=builder_name, builder_kwargs=_filter_callable_kwargs(getattr(mod, builder_name), ctx),
            pass_configs=pass_configs, compile_flags=tuple(compile_flags),
            selector_name=sel_name, combine_kwargs=combine_kwargs,
        )

    if mod_path is not None:
        return _load_and_resolve(mod_path)
    with tempfile.TemporaryDirectory(prefix="tf_resolve_cfg_") as tmp_dir:
        tmp_file = Path(tmp_dir) / "module_src.py"
        tmp_file.write_text(src_text, encoding="utf-8")
        return _load_and_resolve(tmp_file)


def _extract_trackb_cpp_source(module_source: str, regime_spec: RegimeTargetSpec) -> tuple[str, str] | None:
    return extract_trackb_cpp_source(module_source, regime_spec.trackb_symbols)


def _top_level_symbol_sources(module_source: str) -> dict[str, str]:
    tree = ast.parse(module_source)
    lines = module_source.splitlines(keepends=True)
    return {sym.name: "".join(lines[sym.start - 1 : sym.end]).strip() for sym in _collect_top_level_symbols(tree)}


def _reject_trackb_context_markers(candidate_text: str, regime_tag: str) -> None:
    markers = ("READONLY_SIBLING_DECLARATION", "READ-ONLY SIBLING KERNEL", "READ-ONLY HELPER")
    if found := [m for m in markers if m in candidate_text]:
        raise ValueError(
            f"Track-B candidate for regime {regime_tag!r} echoes read-only markers {found}; "
            "reply with a full symbol unit or SEARCH/REPLACE hunk instead"
        )


def _strip_trackb_patch_hunks(patch_text: str) -> str:
    residual = patch_text
    for pattern in _SEARCH_REPLACE_PATTERNS:
        residual = pattern.sub("", residual)
    return residual.strip()


def _parses_as_python(candidate_text: str) -> bool:
    try:
        ast.parse(candidate_text)
        return True
    except SyntaxError:
        return False


def _render_trackb_tu_from_module(
    module_source: str, cpp_source: str, regime_spec: RegimeTargetSpec, *,
    require_renderer: bool, anchor_dims_override: Mapping[str, object] | None = None,
) -> str:
    tree = ast.parse(module_source)
    if _find_top_level_func(tree, ("_render_trackb_tu",)) is None:
        if require_renderer:
            raise ValueError("Track-B seed must declare def _render_trackb_tu(**dims: object) -> str")
        return cpp_source
    dims: dict[str, object] = {
        **regime_spec.sample_builder_kwargs, **regime_spec.representative_hir_dims,
        **regime_spec.representative_anchor_dims, **(anchor_dims_override or {}),
    }
    try:
        resolved = resolve_regime_config(module_source, regime_spec, dims=dims)
        alias_sources = {v: k for k, v in _HIR_DIM_ALIASES.items()}
        for k, v in resolved.builder_kwargs.items():
            if not ((src := alias_sources.get(k)) is not None and src in dims and dims[src] != v):
                dims.setdefault(k, v)
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    dims["trackb_src"] = cpp_source
    with tempfile.TemporaryDirectory(prefix="tf_trackb_inst_") as tmp_dir:
        module_file = Path(tmp_dir) / "candidate_mod.py"
        module_file.write_text(module_source, encoding="utf-8")
        spec = importlib.util.spec_from_file_location("_tf_trackb_render", str(module_file))
        if spec is None or spec.loader is None:
            if require_renderer:
                raise RuntimeError("cannot import the seed module to render its Track-B TU")
            return cpp_source
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        render = getattr(module, "_render_trackb_tu", None)
        if not callable(render):
            if require_renderer:
                raise ValueError("Track-B seed must declare def _render_trackb_tu(**dims: object) -> str")
            return cpp_source
        rendered = render(**dims)
    if not isinstance(rendered, str) or not rendered.strip():
        if require_renderer:
            raise ValueError("_render_trackb_tu returned an empty C++ translation unit")
        return cpp_source
    return rendered


def _trackb_kernel_names(cpp_source: str) -> set[str]:
    return {u.name for u in _split_cpp_units(cpp_source)[1] if u.kind == "kernel" and u.name}


def _trackb_units_by_name(cpp_source: str) -> dict[str, str]:
    return {u.name: u.text for u in _split_cpp_units(cpp_source)[1] if u.name}


def _launched_trackb_kernels(instantiated_tu: str, raw_cpp_source: str, kernels: set[str]) -> tuple[set[str], set[str]]:
    launched, primary, _ = scan_launched_trackb_kernels(instantiated_tu, raw_cpp_source, kernels)
    return launched, primary


def _ordered_top_level_trackb_kernels(instantiated_tu: str, raw_cpp_source: str, kernels: set[str]) -> tuple[str, ...]:
    return scan_launched_trackb_kernels(instantiated_tu, raw_cpp_source, kernels, fallback_full=True)[2]


def _active_trackb_kernel(
    instantiated_tu: str, raw_cpp_source: str, *, regime_tag: str, module_source: str | None = None
) -> str:
    if (resolved := resolve_instantiated_trackb_kernel(instantiated_tu, raw_cpp_source)) is not None:
        return resolved
    kernels = sorted(_trackb_kernel_names(raw_cpp_source))
    if len(kernels) == 1:
        return kernels[0]
    if not kernels:
        raise ValueError(f"Track-B instantiation for regime {regime_tag!r} contains no __global__ kernel")
    launched, primary = _launched_trackb_kernels(instantiated_tu, raw_cpp_source, set(kernels))
    if len(primary) == 1:
        return next(iter(primary))
    if module_source and (mod_sel := select_active_trackb_kernel(raw_cpp_source, regime_tag=regime_tag, module_source=module_source)) is not None:
        return mod_sel
    if not launched:
        raise ValueError(
            f"Track-B instantiation for regime {regime_tag!r} contains multiple kernels {kernels} "
            "and no launcher stub; the instantiated translation unit must launch exactly one primary kernel"
        )
    raise ValueError(
        f"Track-B instantiation for regime {regime_tag!r} launches kernels {sorted(launched)} with no unique "
        f"top-level launch target among {sorted(primary)}; the launcher stub must launch exactly one primary kernel outside conditional branches"
    )


def _normalized_trackb_unit(text: str) -> str:
    return " ".join(text.split())


def _assert_trackb_replacement_scoped(
    orig_cpp: str, new_cpp: str, regime_spec: RegimeTargetSpec, *,
    module_source: str, active_kernel: str | None, allowed_added_units: Collection[str] = (),
) -> None:
    orig_units, new_units = _trackb_units_by_name(orig_cpp), _trackb_units_by_name(new_cpp)
    if active_kernel is None:
        active_kernel = select_active_trackb_kernel(orig_cpp, regime_tag=regime_spec.regime_tag, module_source=module_source)
    if active_kernel is None:
        raise ValueError(
            f"Track-B full replacement for regime {regime_spec.regime_tag!r} has no resolvable active kernel; "
            "reply with SEARCH/REPLACE hunks or one bare C++ unit"
        )
    editable = {u.name for u in _scoped_editable_units(orig_cpp, active_kernel, allow_exclusive_helpers=True)}
    if dropped := sorted(set(orig_units) - set(new_units)):
        raise ValueError(f"Track-B replacement for regime {regime_spec.regime_tag!r} drops units {dropped}; reply with SEARCH/REPLACE hunks or one bare C++ unit per changed function")
    if added := sorted(set(new_units) - set(orig_units) - set(allowed_added_units)):
        raise ValueError(f"Track-B replacement for regime {regime_spec.regime_tag!r} adds units {added}; reply with SEARCH/REPLACE hunks or one bare C++ unit per changed function")
    changed = [n for n in orig_units if _normalized_trackb_unit(orig_units[n]) != _normalized_trackb_unit(new_units[n])]
    if frozen := sorted(n for n in changed if n not in editable):
        raise ValueError(
            f"Track-B replacement for regime {regime_spec.regime_tag!r} edits frozen sibling kernels or shared helpers {frozen}; "
            f"only active kernel {active_kernel!r} and its exclusive helpers may change"
        )


def _case_ratio_predicate_entries(ratios: CasePhysicsRatios | None) -> dict[str, object]:
    return {} if ratios is None else {f"case_{k.removeprefix('case_')}": v for k, v in ratios.to_dict().items()}


def _detect_trackb_dead_code_edit(parent_source: str, candidate_source: str, regime_spec: RegimeTargetSpec) -> str | None:
    if regime_spec.runtime_backend not in ("ascendc_trackb", "hybrid_trackb") or not regime_spec.trackb_symbols:
        return None
    p_syms, c_syms = _top_level_symbol_sources(parent_source), _top_level_symbol_sources(candidate_source)
    if not p_syms or not c_syms:
        return None
    changed = {n for n in (set(p_syms) | set(c_syms)) if p_syms.get(n, "") != c_syms.get(n, "")}
    if not changed:
        return None
    active_syms = set(regime_spec.trackb_symbols)
    if regime_spec.config_selector_fn:
        active_syms.add(regime_spec.config_selector_fn)
    if not (changed & active_syms):
        return (
            f"DEAD_CODE_TRACKB_EDIT: regime {regime_spec.regime_tag!r} executes runtime_backend={regime_spec.runtime_backend} "
            f"via {list(regime_spec.trackb_symbols)}; candidate modified only inactive fallback symbols {sorted(changed)}."
        )
    return None


def _resolve_cann_root_from_env_file(env_file: str | Path | None) -> Path | None:
    if env_file is None or not (p := Path(env_file)).is_file():
        return None
    if not (setup_raw := parse_simple_env_file(p).get("TILEFOUNDRY_NPU_CANN_SETUP", "").strip()):
        return None
    parent = Path(setup_raw).expanduser().parent
    return parent if parent.is_dir() else None


def _resolve_operator_entrypoint_safe(operator_name: str) -> str:
    try:
        return get_operator_entrypoint(operator_name)
    except (KeyError, FileNotFoundError):
        return str(operator_name).rsplit("/", 1)[-1].strip()


def _other_regime_forbidden_symbols(
    regime_spec: RegimeTargetSpec, *, operator_entrypoint: str = "",
    sibling_specs: Sequence[RegimeTargetSpec] = (), module_source: str = "",
) -> set[str]:
    return _derive_dynamic_protected_symbols(
        module_source, regime_spec, operator_entrypoint=operator_entrypoint, sibling_specs=sibling_specs
    )


def extract_regime_snippet(
    module_source: str, regime_spec: RegimeTargetSpec, *,
    operator_entrypoint: str = "", sibling_specs: Sequence[RegimeTargetSpec] = (),
) -> tuple[str, str]:
    """Extract target regime multi-symbol slice (selector, builders, helpers) cleanly."""
    tree = ast.parse(module_source)
    lines = module_source.splitlines(keepends=True)
    symbols, ast_by_name = _collect_top_level_symbols(tree), _collect_top_level_ast_nodes(tree)
    editable_set, prefix = set(regime_spec.editable_symbols), regime_spec.allowed_helper_prefix

    if regime_spec.runtime_backend in ("ascendc_trackb", "hybrid_trackb") and regime_spec.trackb_symbols:
        tb_set = set(regime_spec.trackb_symbols)
        tb_sel = [
            s for s in symbols
            if s.name in tb_set or s.name == "_render_trackb_tu"
            or (regime_spec.config_selector_fn and s.name == regime_spec.config_selector_fn)
            or (s.name.startswith("_select_") and (not editable_set or s.name in editable_set))
        ]
        if tb_sel:
            fnames = {s.name for s in tb_sel if s.kind == "func"}
            primary = regime_spec.config_selector_fn if (regime_spec.config_selector_fn in fnames) else next((s.name for s in tb_sel if s.kind == "func"), tb_sel[0].name)
            return primary, "\n\n".join("".join(lines[s.start - 1 : s.end]).rstrip() for s in tb_sel).rstrip() + "\n"

    seed_symbols = [s for s in symbols if s.name in editable_set or (prefix and s.name.startswith(prefix))] if editable_set else []
    if not seed_symbols:
        if (node := _find_top_level_func(tree, regime_spec.builder_candidates)) is None:
            raise ValueError(f"No builder function matching {regime_spec.builder_candidates} found in module.")
        seed_symbols = [s for s in symbols if s.name == node.name]
    seed_names = [s.name for s in seed_symbols]
    seed_set = set(seed_names)
    for sym in symbols:
        if sym.name in {"_plan_workspace_bytes", f"_{regime_spec.regime_tag}_plan_workspace_bytes"} and sym.name not in seed_set:
            seed_symbols.append(sym); seed_names.append(sym.name); seed_set.add(sym.name)
    closure_helpers: set[str] = set()
    if regime_spec.runtime_backend not in ("ascendc_trackb", "hybrid_trackb"):
        excluded = _other_regime_forbidden_symbols(regime_spec, operator_entrypoint=operator_entrypoint, sibling_specs=sibling_specs, module_source=module_source)
        if editable_set:
            excluded.update(c for c in regime_spec.builder_candidates if c not in editable_set)
        priv_cands = {s.name for s in symbols if s.kind == "func" and s.name.startswith("_") and s.name not in excluded and s.name not in seed_set}
        closure_helpers = _collect_referenced_helper_names(ast_by_name, helper_names=priv_cands, seed_roots=[n for n in seed_names if n != regime_spec.config_selector_fn] or seed_names)
    selected = [s for s in symbols if s.name in seed_set or s.name in closure_helpers]
    func_names = {s.name for s in selected if s.kind == "func"}
    primary = next((c for c in regime_spec.builder_candidates if c in func_names), "") or next((s.name for s in selected if s.kind == "func"), selected[0].name)
    return primary, "\n\n".join("".join(lines[s.start - 1 : s.end]).rstrip() for s in selected).rstrip() + "\n"


def split_readonly_host_tail(
    snippet_source: str, protected_exports: Collection[str] = DEFAULT_READONLY_HOST_EXPORTS
) -> tuple[str, str]:
    """Separate protected module-tail bindings from *snippet_source* into a read-only tail."""
    protected_set = {str(n) for n in protected_exports if str(n)}
    if not protected_set or not snippet_source.strip():
        return snippet_source, ""
    try:
        tree = ast.parse(snippet_source)
    except SyntaxError:
        return snippet_source, ""
    lines = snippet_source.splitlines(keepends=True)
    spans: list[tuple[int, int]] = []
    tail_blocks: list[str] = []
    for node in tree.body:
        names = (
            [node.name] if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
            else ([t.id for t in node.targets if isinstance(t, ast.Name)] if isinstance(node, ast.Assign)
                  else ([node.target.id] if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) else []))
        )
        if any(n in protected_set for n in names):
            s, e = _func_line_span(node)
            spans.append((s, e))
            if block := "".join(lines[s - 1 : e]).rstrip():
                tail_blocks.append(block)
    if not spans:
        return snippet_source, ""
    rem = list(lines)
    for s, e in reversed(spans):
        del_e = e
        while del_e < len(rem) and rem[del_e].strip() == "":
            del_e += 1
        del rem[s - 1 : del_e]
    editable = "".join(rem).rstrip()
    return (f"{editable}\n" if editable else ""), "\n\n".join(tail_blocks).rstrip()


def _is_prim_func_decorator(dec: ast.expr) -> bool:
    return (isinstance(dec, ast.Name) and dec.id == "prim_func") or (isinstance(dec, ast.Attribute) and dec.attr == "prim_func") or (isinstance(dec, ast.Call) and _is_prim_func_decorator(dec.func))


def _has_prim_func_builder(module_source: str, builder_candidates: Sequence[str]) -> bool:
    if not builder_candidates:
        return False
    try:
        tree = ast.parse(module_source)
    except SyntaxError:
        return False
    cand_set = set(builder_candidates)
    return any(
        any(isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) and sub is not node and any(_is_prim_func_decorator(d) for d in sub.decorator_list) for sub in ast.walk(node))
        for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in cand_set
    )


def _extract_func_signature_header(lines: Sequence[str], node: ast.FunctionDef) -> str:
    raw = (
        "".join(lines[node.lineno - 1 : node.body[0].lineno - 1]).rstrip()
        if (node.body and node.body[0].lineno > node.lineno)
        else "".join(lines[node.lineno - 1 : node.lineno]).split(":", 1)[0].rstrip() + ":"
    )
    return textwrap.dedent(raw).strip()


def _extract_regime_abi_context(module_source: str, regime_spec: RegimeTargetSpec, *, golden_abi_block: str = "") -> str:
    tree = ast.parse(module_source)
    lines = module_source.splitlines(keepends=True)
    top_funcs = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    editable_set, prefix = set(regime_spec.editable_symbols), regime_spec.allowed_helper_prefix
    seeds = [
        n.name for n in tree.body
        if isinstance(n, ast.FunctionDef) and (n.name in editable_set or (prefix and n.name.startswith(prefix)) or n.name in regime_spec.builder_candidates)
    ]
    if not seeds:
        return golden_abi_block
    visited: set[str] = set()
    slice_funcs, helper_funcs, prim_sigs, seen_prim = [], [], [], set()
    queue = list(seeds)
    while queue:
        if (fname := queue.pop(0)) in visited:
            continue
        visited.add(fname)
        if (fnode := top_funcs.get(fname)) is None:
            continue
        if fname in editable_set or fname in regime_spec.builder_candidates:
            slice_funcs.append(fnode)
        elif not (prefix and fname.startswith(prefix)):
            helper_funcs.append(fnode)
        for sub in ast.walk(fnode):
            if isinstance(sub, ast.FunctionDef) and sub is not fnode and any(_is_prim_func_decorator(d) for d in sub.decorator_list):
                if (sig := "@T.prim_func\n" + _extract_func_signature_header(lines, sub)) not in seen_prim:
                    seen_prim.add(sig)
                    prim_sigs.append(sig)
            elif isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name) and sub.func.id in top_funcs and sub.func.id not in visited:
                queue.append(sub.func.id)
    blocks = [_extract_func_signature_header(lines, n) for n in (*helper_funcs, *slice_funcs)] + prim_sigs
    if not blocks:
        return golden_abi_block
    return (
        f"{golden_abi_block}### IMMUTABLE HOST / PRIMFUNC ABI CONTRACT\n```python\n{chr(10).join([blocks[0]] + [chr(10) + b for b in blocks[1:]]) if blocks else ''}\n```\n"
        "Any custom @T.prim_func returned for this slice MUST keep the exact tensor/scalar parameter order, ranks, and dtypes above expected by _get_kernel "
        "(do NOT substitute the PyTorch golden entrypoint parameter list, and do NOT import tilelang.language as T or use T.Scope). "
        "_select_*_config() may return optional 'builder_kwargs': {...} forwarded to custom builders for compile-time constexpr specialization.\n\n"
    )


def apply_ast_patch(module_source: str, target_func: str, candidate_dsl: str) -> str:
    """Replace *target_func* in *module_source* using Python AST line anchors."""
    orig_tree, cand_tree = ast.parse(module_source), ast.parse(candidate_dsl)
    if (cand_node := _find_top_level_func(cand_tree, (target_func,))) is None:
        raise ValueError(f"Candidate source must define function {target_func!r}.")
    c_start, c_end = _func_line_span(cand_node)
    replacement = "".join(candidate_dsl.splitlines(keepends=True)[c_start - 1 : c_end])
    if not replacement.endswith("\n"):
        replacement += "\n"
    if (orig_node := _find_top_level_func(orig_tree, (target_func,))) is None:
        raise ValueError(f"Original module missing target function {target_func!r}.")
    o_start, o_end = _func_line_span(orig_node)
    orig_lines = module_source.splitlines(keepends=True)
    patched = "".join([*orig_lines[: o_start - 1], replacement, *orig_lines[o_end:]])
    ast.parse(patched)
    return patched


def _replace_trackb_constant(module_source: str, sym_name: str, cpp_source: str) -> str:
    hits = _find_top_level_str_assign_span(ast.parse(module_source), module_source.splitlines(keepends=True), target_names={sym_name})
    if not hits:
        raise ValueError(f"Track-B symbol {sym_name!r} assignment not found in module_source.")
    _, _, start_pos, end_pos = hits[0]
    patched = module_source[:start_pos] + _format_raw_cpp_literal(cpp_source) + module_source[end_pos:]
    ast.parse(patched)
    return patched


def _patch_anchor_miss_error_type() -> type[ValueError]:
    exc_cls = getattr(_cpp_slice_protocol_mod, "PatchAnchorMissError", None)
    return exc_cls if (isinstance(exc_cls, type) and issubclass(exc_cls, ValueError)) else ValueError


def _format_patch_anchor_miss_diagnostic(
    module_source: str, regime_spec: RegimeTargetSpec, reason: str, *, active_kernel: str | None = None
) -> str:
    if (extracted := _extract_trackb_cpp_source(module_source, regime_spec)) is None:
        return f"PATCH_ANCHOR_MISS: {reason}"
    _, orig_cpp = extracted
    resolved = active_kernel or select_active_trackb_kernel(orig_cpp, regime_tag=regime_spec.regime_tag, module_source=module_source)
    editable = _scoped_editable_units(orig_cpp, resolved, allow_exclusive_helpers=True)
    target = next((u for u in editable if u.name == resolved), editable[0] if editable else None)
    if target is None:
        return f"PATCH_ANCHOR_MISS: {reason} (active_kernel={resolved})"
    u_lines = [ln.rstrip() for ln in target.text.splitlines() if ln.strip()]
    return f"PATCH_ANCHOR_MISS: {reason} (active_kernel={resolved}; first_lines={' | '.join(u_lines[:3])!r}; last_lines={' | '.join(u_lines[-3:])!r})"


def _is_anchor_miss_value_error(exc: ValueError) -> bool:
    miss_cls = _patch_anchor_miss_error_type()
    if miss_cls is not ValueError and isinstance(exc, miss_cls):
        return True
    msg = str(exc).lower()
    return any(t in msg for t in ("did not match", "ambiguous search", "outside active kernel", "outside the editable scope", "not found in c++ source", "must contain at least one valid", "identical", "applied == 0"))


def _apply_trackb_module_edit(
    module_source: str, regime_spec: RegimeTargetSpec, payload: str, *,
    transform_fn: Callable[..., tuple[str, int]], noop_msg: str, active_kernel: str | None = None,
) -> str:
    if (extracted := _extract_trackb_cpp_source(module_source, regime_spec)) is None:
        raise ValueError(f"Original module missing Track-B C++ constant in {regime_spec.trackb_symbols}.")
    sym_name, orig_cpp = extracted
    resolved = active_kernel or select_active_trackb_kernel(orig_cpp, regime_tag=regime_spec.regime_tag, module_source=module_source)
    miss_cls = _patch_anchor_miss_error_type()
    try:
        updated_cpp, applied = transform_fn(orig_cpp, payload, active_kernel=resolved)
    except ValueError as exc:
        if "PATCH_ANCHOR_MISS:" in str(exc):
            raise
        if _is_anchor_miss_value_error(exc):
            raise miss_cls(_format_patch_anchor_miss_diagnostic(module_source, regime_spec, str(exc), active_kernel=resolved)) from exc
        raise
    if applied == 0 or updated_cpp.strip() == orig_cpp.strip():
        raise miss_cls(_format_patch_anchor_miss_diagnostic(module_source, regime_spec, noop_msg, active_kernel=resolved))
    return _replace_trackb_constant(module_source, sym_name, updated_cpp)


def _apply_scoped_trackb_patch_to_module(
    module_source: str, regime_spec: RegimeTargetSpec, patch_text: str, *, active_kernel: str | None = None
) -> str:
    return _apply_trackb_module_edit(
        module_source, regime_spec, patch_text, transform_fn=apply_scoped_trackb_patch,
        noop_msg="Track-B SEARCH/REPLACE produced identical C++ source", active_kernel=active_kernel,
    )


def _apply_spliced_trackb_units_to_module(
    module_source: str, regime_spec: RegimeTargetSpec, candidate_cpp: str, *, active_kernel: str | None = None
) -> str:
    return _apply_trackb_module_edit(
        module_source, regime_spec, candidate_cpp, transform_fn=splice_cpp_units_into_tu,
        noop_msg="Track-B C++ unit splice produced identical C++ source", active_kernel=active_kernel,
    )


def _trackb_python_residual(patch_text: str) -> str:
    residual = _strip_trackb_patch_hunks(patch_text)
    return residual if (residual and _parses_as_python(residual)) else ""


def _extract_p2_focus_lines_and_diag(p2_diag: Mapping[str, object] | None) -> tuple[tuple[int, ...], str]:
    if not p2_diag:
        return (), ""
    focus: set[int] = set()
    diag_parts: list[str] = []
    for key in ("diagnostic_text", "error_message", "compiler_diagnostic"):
        if isinstance(raw_val := p2_diag.get(key), str) and raw_val.strip():
            diag_parts.append(raw_val.strip())
            focus.update(extract_cpp_diagnostic_lines(raw_val))
    raw_hotspots = p2_diag.get("p2_hotspots") or p2_diag.get("source_line_hotspots") or ()
    if isinstance(raw_hotspots, Sequence) and not isinstance(raw_hotspots, (str, bytes)):
        for spot in list(raw_hotspots):
            loc = spot.get("source_location", "") if isinstance(spot, Mapping) else getattr(spot, "source_location", spot)
            if loc_str := str(loc or "").strip():
                focus.update(int(m.group(1)) for m in re.finditer(r":(\d+)(?::\d+)?\b", loc_str) if int(m.group(1)) > 0)
    if isinstance(raw_lines := p2_diag.get("focus_lines"), Sequence) and not isinstance(raw_lines, (str, bytes)):
        focus.update(item for item in raw_lines if isinstance(item, int) and not isinstance(item, bool) and item > 0)
    return tuple(sorted(focus)), "\n".join(diag_parts)


def _compact_trackb_snippet_source(
    snippet_source: str, regime_spec: RegimeTargetSpec, *, incumbent_source: str = "",
    max_bytes: int | None = None, cpp_max_bytes: int | None = None, active_kernel: str | None = None,
    focus_lines: Sequence[int] = (), diagnostic_text: str = "",
) -> str:
    _ = (max_bytes, cpp_max_bytes)
    if regime_spec.runtime_backend not in ("ascendc_trackb", "hybrid_trackb") or not regime_spec.trackb_symbols:
        return snippet_source
    try:
        tree = ast.parse(snippet_source)
    except SyntaxError:
        return snippet_source
    hits = _find_top_level_str_assign_span(tree, snippet_source.splitlines(keepends=True), target_names=set(regime_spec.trackb_symbols))
    updated = snippet_source
    for sym_name, cpp_text, start_pos, end_pos in reversed(hits):
        if "/* TRACK-B SCOPED SLICE:" in cpp_text:
            continue
        view = build_trackb_context_view(
            cpp_text, regime_tag=regime_spec.regime_tag, module_source=incumbent_source or snippet_source,
            trackb_symbol=sym_name, active_kernel=active_kernel, focus_lines=focus_lines, diagnostic_text=diagnostic_text,
        )
        wrapped = view if view.startswith("\n") else f"\n{view}\n"
        updated = updated[:start_pos] + _format_raw_cpp_literal(wrapped) + updated[end_pos:]
    return updated


def _is_real_tir_primfunc(obj: object) -> bool:
    if obj is None or isinstance(obj, (tuple, list, dict, str, bytes, int, float, bool)):
        return False
    try:
        import tilelang  # noqa: F401, PLC0415
        import tvm  # noqa: PLC0415

        prim_cls = getattr(getattr(tvm, "tirx", None), "PrimFunc", None) or getattr(getattr(tvm, "tir", None), "PrimFunc", None)
        return isinstance(prim_cls, type) and isinstance(obj, prim_cls)
    except ImportError:
        return False


def _cow_helper_name(regime_tag: str, helper_name: str) -> str:
    cow_prefix = f"_{regime_tag}_"
    return helper_name if helper_name.startswith(cow_prefix) else f"{cow_prefix}{helper_name.lstrip('_')}"


class _CowHelperRenamer(ast.NodeTransformer):
    """Rename CoW helper FunctionDefs and ast.Name references inside active symbols."""

    def __init__(self, rename_map: Mapping[str, str]) -> None:
        self._rename_map = dict(rename_map)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        if node.name in self._rename_map:
            node.name = self._rename_map[node.name]
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        if node.name in self._rename_map:
            node.name = self._rename_map[node.name]
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self._rename_map:
            node.id = self._rename_map[node.id]
        return node


def _ast_references_any(node: ast.AST, names: Collection[str]) -> bool:
    target_set = set(names)
    return bool(target_set) and any(isinstance(sub, ast.Name) and sub.id in target_set for sub in ast.walk(node))


def _render_with_cow_renames(
    node: ast.stmt, raw_source: str, rename_map: Mapping[str, str], *, force_rename_self: str = ""
) -> tuple[str, ast.stmt]:
    if not (rename_map and ((force_rename_self and force_rename_self in rename_map) or _ast_references_any(node, rename_map))):
        return raw_source, node
    transformed = _CowHelperRenamer(rename_map).visit(copy.deepcopy(node))
    ast.fix_missing_locations(transformed)
    return ast.unparse(transformed).rstrip() + "\n", transformed


def _collect_shared_non_active_symbols(
    orig_symbols: Sequence[_TopLevelSymbol], orig_ast_by_name: Mapping[str, ast.stmt],
    *, active_own_symbols: set[str], forbidden: set[str],
) -> set[str]:
    called_by_any = {
        sub.id for cname, cnode in orig_ast_by_name.items()
        for sub in ast.walk(cnode)
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load) and sub.id in orig_ast_by_name and sub.id != cname
    }
    non_active_roots = [
        s.name for s in orig_symbols
        if s.name not in active_own_symbols and (s.name in forbidden or not s.name.startswith("_") or s.kind != "func" or s.name not in called_by_any)
    ]
    eligible = set(orig_ast_by_name) - active_own_symbols
    return _collect_referenced_helper_names(orig_ast_by_name, helper_names=eligible, seed_roots=non_active_roots)


def _resolve_tracka_editable_set(tree: ast.Module, regime_spec: RegimeTargetSpec) -> set[str]:
    top_names = {sym.name for sym in _collect_top_level_symbols(tree)}
    if regime_spec.editable_symbols and any(sym in top_names for sym in regime_spec.editable_symbols):
        editable = set(regime_spec.editable_symbols)
    else:
        if (orig_node := _find_top_level_func(tree, regime_spec.builder_candidates)) is None:
            raise ValueError(f"Original module missing regime symbols or builder {regime_spec.builder_candidates}.")
        editable = {orig_node.name}
    if regime_spec.config_selector_fn and regime_spec.config_selector_fn in top_names:
        editable.add(regime_spec.config_selector_fn)
    return editable


def _protected_python_spans(
    source: str, regime_spec: RegimeTargetSpec, *, editable_set: set[str],
    operator_entrypoint: str, sibling_specs: Sequence[RegimeTargetSpec],
) -> tuple[int | None, dict[str, tuple[int, int]], set[str], str]:
    tail_exports = set(DEFAULT_PROTECTED_EXPORTS) | ({operator_entrypoint} if operator_entrypoint else set())
    tail_exports.difference_update(editable_set)
    _, readonly_tail = split_tracka_editable_and_readonly_tail(source, protected_exports=frozenset(tail_exports))
    tail_start_line = (source.count("\n", 0, tc) + 1) if (readonly_tail and (tc := source.rfind(readonly_tail)) >= 0) else None
    sib_selectors = {s.config_selector_fn for s in sibling_specs if s.regime_tag != regime_spec.regime_tag and s.config_selector_fn}
    allowed_own = (set(editable_set) - sib_selectors) | {"_plan_workspace_bytes", f"_{regime_spec.regime_tag}_plan_workspace_bytes"}
    forbidden = (_other_regime_forbidden_symbols(regime_spec, operator_entrypoint=operator_entrypoint, sibling_specs=sibling_specs, module_source=source) | tail_exports) - allowed_own
    protected_spans: dict[str, tuple[int, int]] = {}
    try:
        tree = ast.parse(source)
        protected_spans = {sym.name: (sym.start, sym.end) for sym in _collect_top_level_symbols(tree) if sym.name in forbidden}
    except SyntaxError:
        pass
    return tail_start_line, protected_spans, forbidden, readonly_tail


def _apply_scoped_python_search_replace_to_module(
    module_source: str, regime_spec: RegimeTargetSpec, candidate_dsl: str,
    *, operator_entrypoint: str = "", sibling_specs: Sequence[RegimeTargetSpec] = (),
) -> str:
    hunks = _parse_scoped_search_replace_hunks(candidate_dsl)
    if not hunks:
        raise ValueError("Track-A Python candidate contains no valid SEARCH/REPLACE hunks")
    orig_tree = ast.parse(module_source)
    editable_set = _resolve_tracka_editable_set(orig_tree, regime_spec)
    tail_exports = (set(DEFAULT_PROTECTED_EXPORTS) | ({operator_entrypoint} if operator_entrypoint else set())) - editable_set
    _, orig_spans, protected_names, orig_readonly_tail = _protected_python_spans(
        module_source, regime_spec, editable_set=editable_set, operator_entrypoint=operator_entrypoint, sibling_specs=sibling_specs
    )
    orig_ast_by_name = _collect_top_level_ast_nodes(orig_tree)

    cur = module_source
    for hunk in hunks:
        if not hunk.search_block.strip():
            raise ValueError("Empty SEARCH block is forbidden in Track-A Python patch")
        tail_start_line, protected_spans, _, _ = _protected_python_spans(
            cur, regime_spec, editable_set=editable_set, operator_entrypoint=operator_entrypoint, sibling_specs=sibling_specs
        )
        protected_spans = protected_spans or orig_spans
        spans = _find_all_substring_spans(cur, hunk.search_block) or [
            (s, e) for s, e, _ in _normalized_line_matches(cur, hunk.search_block, strip_common_indent=False, keep_trailing_newline=False)
        ]
        if not spans:
            raise ValueError("SEARCH block did not match target Track-A Python source")
        picked = _disambiguate_matches_by_line_window(cur, spans, hunk.line_window) if len(spans) > 1 else 0
        m_start, m_end = spans[picked]
        m_s_line, m_e_line = _char_span_to_line_span(cur, m_start, m_end)
        if tail_start_line is not None and m_e_line >= tail_start_line:
            raise ValueError(f"Track-A SEARCH/REPLACE hunk at lines {m_s_line}-{m_e_line} targets read-only host tail (starts at line {tail_start_line})")
        for sym_name, (sym_s, sym_e) in protected_spans.items():
            if m_s_line <= sym_e and m_e_line >= sym_s:
                raise ValueError(f"Track-A SEARCH/REPLACE hunk at lines {m_s_line}-{m_e_line} targets protected symbol {sym_name!r} (lines {sym_s}-{sym_e})")
        cur = cur[:m_start] + hunk.replace_block + cur[m_end:]

    patched_ast_by_name = _collect_top_level_ast_nodes(ast.parse(cur))
    for name in sorted(protected_names):
        o_node, n_node = orig_ast_by_name.get(name), patched_ast_by_name.get(name)
        if o_node is None and n_node is not None:
            raise ValueError(f"Track-A SEARCH/REPLACE cannot define protected symbol {name!r}.")
        if o_node is not None and (n_node is None or ast.dump(o_node) != ast.dump(n_node)):
            raise ValueError(f"Track-A SEARCH/REPLACE cannot modify protected symbol {name!r}.")
    if orig_readonly_tail and split_tracka_editable_and_readonly_tail(cur, protected_exports=frozenset(tail_exports))[1] != orig_readonly_tail:
        raise ValueError("Track-A SEARCH/REPLACE modified read-only host tail.")
    if cur == module_source or cur.strip() == module_source.strip():
        raise ValueError("Track-A SEARCH/REPLACE produced identical Python source")
    return cur


def patch_regime_into_module(
    module_source: str, regime_or_func: RegimeTargetSpec | str, candidate_raw_source: str,
    *, operator_entrypoint: str = "", sibling_specs: Sequence[RegimeTargetSpec] = (), active_kernel: str | None = None,
) -> str:
    """Merge a scoped or Python regime slice into module_source without losing siblings."""
    candidate_dsl = extract_python_dsl(candidate_raw_source)
    if isinstance(regime_or_func, str):
        return apply_ast_patch(module_source, regime_or_func, candidate_dsl)

    regime_spec = regime_or_func
    python_snippet = candidate_dsl
    if regime_spec.runtime_backend in ("ascendc_trackb", "hybrid_trackb") and regime_spec.trackb_symbols:
        _reject_trackb_context_markers(candidate_dsl, regime_spec.regime_tag)
        sr_blocks = parse_scoped_search_replace_blocks(candidate_dsl)
        if ("<<<<<<<" in candidate_dsl or ">>>>>>>" in candidate_dsl) and not sr_blocks:
            raise _patch_anchor_miss_error_type()(_format_patch_anchor_miss_diagnostic(
                module_source, regime_spec, "Malformed SEARCH/REPLACE block markers in Track-B candidate", active_kernel=active_kernel
            ))
        if sr_blocks:
            module_source = _apply_scoped_trackb_patch_to_module(module_source, regime_spec, candidate_dsl, active_kernel=active_kernel)
            if not (python_snippet := _trackb_python_residual(candidate_dsl)):
                return module_source
        elif not _parses_as_python(candidate_dsl) and _looks_like_bare_trackb_cpp(candidate_dsl):
            return _apply_spliced_trackb_units_to_module(module_source, regime_spec, candidate_dsl, active_kernel=active_kernel)
    else:
        has_sr = any(tok in candidate_dsl for tok in ("<<<<<<<", ">>>>>>>", "<<< SEARCH", ">>> REPLACE"))
        sr_hunks = _parse_scoped_search_replace_hunks(candidate_dsl)
        if has_sr and not sr_hunks:
            raise ValueError("Malformed SEARCH/REPLACE block markers in Track-A Python candidate")
        if sr_hunks:
            return _apply_scoped_python_search_replace_to_module(
                module_source, regime_spec, candidate_dsl, operator_entrypoint=operator_entrypoint, sibling_specs=sibling_specs
            )

    orig_tree = ast.parse(module_source)
    orig_symbols = _collect_top_level_symbols(orig_tree)
    orig_by_name = {sym.name: sym for sym in orig_symbols}
    orig_ast_by_name = _collect_top_level_ast_nodes(orig_tree)

    if regime_spec.editable_symbols and any(s in orig_by_name for s in regime_spec.editable_symbols):
        effective_editable = tuple(regime_spec.editable_symbols)
    else:
        if (orig_node := _find_top_level_func(orig_tree, regime_spec.builder_candidates)) is None:
            msg = f"Original module missing regime symbols {regime_spec.editable_symbols}." if regime_spec.editable_symbols else f"Original module missing builder in {regime_spec.builder_candidates}."
            raise ValueError(msg)
        effective_editable = (orig_node.name,)

    editable_set, prefix, cow_prefix = set(effective_editable), regime_spec.allowed_helper_prefix, f"_{regime_spec.regime_tag}_"
    forbidden = _other_regime_forbidden_symbols(regime_spec, operator_entrypoint=operator_entrypoint, sibling_specs=sibling_specs) - editable_set
    cand_tree = ast.parse(python_snippet)
    cand_symbols: list[_TopLevelSymbol] = []
    cand_ast_by_name: dict[str, ast.stmt] = {}
    seen_cand: set[str] = set()
    for node in cand_tree.body:
        start, end = _func_line_span(node)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name, kind = node.name, "func"
        elif isinstance(node, ast.Assign):
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                raise ValueError("Candidate assignment must target a single variable name.")
            name, kind = node.targets[0].id, "assign"
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name, kind = node.target.id, "assign"
        elif isinstance(node, (ast.Import, ast.ImportFrom)) or (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)):
            continue
        else:
            raise ValueError(f"Candidate snippet contains unsupported top-level {type(node).__name__}.")
        if name in seen_cand:
            raise ValueError(f"Candidate snippet defines duplicate symbol {name!r}.")
        seen_cand.add(name)
        cand_symbols.append(_TopLevelSymbol(name, kind, start, end))
        cand_ast_by_name[name] = node
    if not cand_symbols:
        raise ValueError("Candidate snippet must define at least one authorized regime symbol.")

    priv_pool = {s.name for s in (*orig_symbols, *cand_symbols) if s.kind == "func" and s.name.startswith("_") and s.name not in forbidden and s.name not in editable_set}
    active_asts = {**orig_ast_by_name, **cand_ast_by_name}
    active_roots = [n for n in (*effective_editable, *(s.name for s in cand_symbols if (prefix and s.name.startswith(prefix)) or s.name.startswith(cow_prefix))) if n in active_asts]
    ref_priv = _collect_referenced_helper_names(active_asts, helper_names=priv_pool, seed_roots=active_roots)
    ws_names = {"_plan_workspace_bytes", f"_{regime_spec.regime_tag}_plan_workspace_bytes"}

    for sym in cand_symbols:
        name = sym.name
        if name in forbidden:
            raise ValueError(f"Candidate cannot define or modify protected symbol {name!r}.")
        if name in editable_set or name in ws_names or (prefix and name.startswith(prefix)) or name.startswith(cow_prefix):
            continue
        is_ref_priv = sym.kind == "func" and name.startswith("_") and name in ref_priv
        if not is_ref_priv:
            if name in orig_by_name:
                raise ValueError(f"Candidate cannot define or modify protected symbol {name!r}.")
            raise ValueError(f"Unauthorized top-level symbol {name!r}; expected {editable_set} or prefix {prefix!r}.")

    cand_lines = python_snippet.splitlines(keepends=True)
    cand_slices = {s.name: "".join(cand_lines[s.start - 1 : s.end]).rstrip() + "\n" for s in cand_symbols}
    active_own = editable_set | ws_names | {s.name for s in orig_symbols if (prefix and s.name.startswith(prefix)) or s.name.startswith(cow_prefix)}
    shared_reachable = _collect_shared_non_active_symbols(orig_symbols, orig_ast_by_name, active_own_symbols=active_own, forbidden=forbidden)
    edited_helpers = {
        s.name for s in cand_symbols
        if s.kind == "func" and s.name not in editable_set and (s.name not in orig_ast_by_name or ast.dump(cand_ast_by_name[s.name]) != ast.dump(orig_ast_by_name[s.name]))
    }
    cow_targets = edited_helpers & shared_reachable
    if cow_targets:
        changed = True
        while changed:
            changed = False
            for hname in list(shared_reachable - cow_targets - editable_set):
                if (hnode := cand_ast_by_name.get(hname) or orig_ast_by_name.get(hname)) is not None and _ast_references_any(hnode, cow_targets):
                    cow_targets.add(hname)
                    changed = True
    rename_map = {k: v for h in sorted(cow_targets) if (k := h) != (v := _cow_helper_name(regime_spec.regime_tag, h))}

    rendered_by_target: dict[str, str] = {}
    rendered_ast_by_target: dict[str, ast.stmt] = {}
    for sym in cand_symbols:
        if sym.name in shared_reachable and sym.name not in rename_map and sym.name not in editable_set:
            continue
        tname = rename_map.get(sym.name, sym.name)
        rendered_by_target[tname], rendered_ast_by_target[tname] = _render_with_cow_renames(
            cand_ast_by_name[sym.name], cand_slices[sym.name], rename_map, force_rename_self=sym.name
        )

    orig_lines = module_source.splitlines(keepends=True)
    if rename_map:
        for hname in (*effective_editable, *(h for h in ref_priv if h not in shared_reachable and h not in editable_set)):
            if hname in orig_ast_by_name and hname not in cand_ast_by_name:
                osym = orig_by_name[hname]
                r_src, r_node = _render_with_cow_renames(orig_ast_by_name[hname], "".join(orig_lines[osym.start - 1 : osym.end]).rstrip() + "\n", rename_map)
                if r_node is not orig_ast_by_name[hname]:
                    rendered_by_target[hname], rendered_ast_by_target[hname] = r_src, r_node

    merged_asts = {**orig_ast_by_name, **rendered_ast_by_target}
    prunable_orig = {s.name for s in orig_symbols if s.name not in editable_set and s.name not in shared_reachable and ((prefix and s.name.startswith(prefix)) or s.name.startswith(cow_prefix))}
    prunable_cand = {n for n in rendered_ast_by_target if n not in editable_set and n.startswith(cow_prefix)}
    seed_roots = [n for n in merged_asts if n not in prunable_orig and n not in prunable_cand] + [w for w in ws_names if w in merged_asts]
    live_helpers = _collect_referenced_helper_names(merged_asts, helper_names=prunable_orig | {n for n in rendered_ast_by_target if n not in editable_set}, seed_roots=seed_roots) | (ws_names & set(merged_asts))
    new_helpers = [n for n in rendered_by_target if n not in editable_set and n not in orig_by_name and (n not in prunable_cand or n in live_helpers)]
    insert_anchor = next((s.name for s in orig_symbols if s.name in editable_set or (prefix and s.name.startswith(prefix)) or s.name.startswith(cow_prefix)), orig_symbols[0].name)

    for item in reversed(orig_symbols):
        if item.name in rendered_by_target and not (item.name in prunable_cand and item.name not in live_helpers):
            orig_lines[item.start - 1 : item.end] = [rendered_by_target[item.name]]
        elif (item.name in rendered_by_target and item.name in prunable_cand and item.name not in live_helpers) or (item.name in prunable_orig and item.name not in live_helpers):
            de = item.end
            while de < len(orig_lines) and orig_lines[de].strip() == "":
                de += 1
            del orig_lines[item.start - 1 : de]
        if item.name == insert_anchor and new_helpers:
            orig_lines[item.start - 1 : item.start - 1] = ["".join(rendered_by_target[h] + "\n" for h in new_helpers)]

    patched = "".join(orig_lines)
    ast.parse(patched)
    if tb_edited := sorted(set(cand_slices) & set(regime_spec.trackb_symbols)):
        orig_raw_lines = module_source.splitlines(keepends=True)
        for sym_name in tb_edited:
            o_hits = _find_top_level_str_assign_span(orig_tree, orig_raw_lines, target_names={sym_name})
            n_hits = _find_top_level_str_assign_span(ast.parse(patched), patched.splitlines(keepends=True), target_names={sym_name})
            if not o_hits or not n_hits:
                raise ValueError(f"Track-B symbols {tb_edited} must be assigned a raw C++ string literal.")
            orig_cpp, cand_cpp = o_hits[0][1], n_hits[0][1]
            orig_units, cand_units = _trackb_units_by_name(orig_cpp), _trackb_units_by_name(cand_cpp)
            allowed_added: set[str] = set()
            if cand_units and (set(orig_units) - set(cand_units)):
                kname = active_kernel or select_active_trackb_kernel(orig_cpp, regime_tag=regime_spec.regime_tag, module_source=module_source)
                merged_cpp, _ = splice_cpp_units_into_tu(orig_cpp, cand_cpp, active_kernel=kname)
                allowed_added = set(_trackb_units_by_name(merged_cpp)) - set(orig_units)
                patched = _replace_trackb_constant(patched, sym_name, merged_cpp)
                cand_cpp = merged_cpp
            _assert_trackb_replacement_scoped(orig_cpp, cand_cpp, regime_spec, module_source=module_source, active_kernel=active_kernel, allowed_added_units=allowed_added)
    if "_render_trackb_tu" in cand_slices:
        seen_tags: set[str] = set()
        for spec in (regime_spec, *sibling_specs):
            if spec.regime_tag not in seen_tags and spec.trackb_symbols:
                seen_tags.add(spec.regime_tag)
                if (spec_tb := _extract_trackb_cpp_source(patched, spec)) is not None:
                    _render_trackb_tu_from_module(patched, spec_tb[1], spec, require_renderer=True)
    return patched


__all__ = [
    "DEFAULT_READONLY_HOST_EXPORTS", "ResolvedRegimeConfig", "_CowHelperRenamer", "_HIR_DIM_ALIASES",
    "_TopLevelSymbol", "_active_trackb_kernel", "_apply_scoped_python_search_replace_to_module",
    "_apply_scoped_trackb_patch_to_module", "_apply_spliced_trackb_units_to_module",
    "_assert_trackb_replacement_scoped", "_ast_char_offset", "_ast_references_any", "_atomic_write_text",
    "_case_ratio_predicate_entries", "_collect_referenced_helper_names", "_collect_shared_non_active_symbols",
    "_collect_top_level_ast_nodes", "_collect_top_level_symbols", "_compact_trackb_snippet_source",
    "_cow_helper_name", "_derive_dynamic_protected_symbols", "_detect_trackb_dead_code_edit",
    "_extract_func_signature_header", "_extract_p2_focus_lines_and_diag", "_extract_regime_abi_context",
    "_extract_trackb_cpp_source", "_filter_callable_kwargs", "_find_top_level_func",
    "_find_top_level_str_assign_span", "_format_raw_cpp_literal", "_func_line_span", "_has_prim_func_builder",
    "_is_prim_func_decorator", "_is_real_tir_primfunc", "_launched_trackb_kernels",
    "_load_module_and_run_selector", "_norm_flag_seq", "_normalized_trackb_unit",
    "_ordered_top_level_trackb_kernels", "_other_regime_forbidden_symbols", "_parses_as_python",
    "_reject_trackb_context_markers", "_render_trackb_tu_from_module", "_render_with_cow_renames",
    "_replace_trackb_constant", "_resolve_cann_root_from_env_file", "_resolve_operator_entrypoint_safe",
    "_resolve_selector_compile_flags", "_strip_trackb_patch_hunks", "_top_level_symbol_sources",
    "_trackb_kernel_names", "_trackb_python_residual", "_trackb_units_by_name", "apply_ast_patch",
    "extract_regime_snippet", "patch_regime_into_module", "resolve_regime_config", "split_readonly_host_tail",
]
