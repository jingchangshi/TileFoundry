"""Regime config resolution, slice extraction, Track-B context view, and AST patching."""

from __future__ import annotations

import ast
import copy
import importlib.util
import inspect
import os
import re
import tempfile
import textwrap
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from tilefoundry_ascend.agent.campaign.anchor_specs import (
    CasePhysicsRatios,
    RegimeTargetSpec,
    extract_trackb_cpp_source,
    get_operator_entrypoint,
    parse_simple_env_file,
)
from tilefoundry_ascend.agent.orchestrator import (
    cpp_slice_protocol as _cpp_slice_protocol_mod,
)
from tilefoundry_ascend.agent.orchestrator.cpp_slice_protocol import (
    _LAUNCH_CONFIG_RE,
    _SEARCH_REPLACE_PATTERNS,
    _scoped_editable_units,
    _split_cpp_units,
    apply_scoped_trackb_patch,
    build_trackb_context_view,
    extract_cpp_diagnostic_lines,
    parse_scoped_search_replace_blocks,
    resolve_instantiated_trackb_kernel,
    select_active_trackb_kernel,
    splice_cpp_units_into_tu,
    strip_cpp_comments_and_strings,
)
from tilefoundry_ascend.agent.orchestrator.llm_client import extract_python_dsl

DEFAULT_READONLY_HOST_EXPORTS: tuple[str, ...] = (
    "kernel",
    "ref_kernel",
    "build_kernel",
    "_get_kernel",
    "_KERNEL_CACHE",
    "_ENTRY_CACHE",
)


def _atomic_write_text(path: Path, content: str) -> None:
    """Atomically write *content* to *path* using a same-directory temporary file."""
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
                    if (
                        node.name == "_ascend_target_ctx"
                        or (node.name.startswith("_get_") and node.name.endswith("_kernel"))
                        or node.name == "_get_kernel"
                    ):
                        protected.add(node.name)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id.endswith("_CACHE"):
                            protected.add(target.id)
                elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                    if node.target.id.endswith("_CACHE"):
                        protected.add(node.target.id)
            for root_name in tuple(protected):
                fn = top_funcs.get(root_name)
                if fn is None:
                    continue
                for sub in ast.walk(fn):
                    if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                        if sub.func.id in top_funcs:
                            protected.add(sub.func.id)
    own_editable = set(regime_spec.editable_symbols)
    own_builders = set(regime_spec.builder_candidates)
    for spec in sibling_specs:
        if spec.regime_tag == regime_spec.regime_tag:
            continue
        for sym in spec.editable_symbols:
            if sym not in own_editable:
                protected.add(sym)
        if spec.config_selector_fn and spec.config_selector_fn not in own_editable:
            protected.add(spec.config_selector_fn)
        for cand in spec.builder_candidates:
            if cand not in own_editable and cand not in own_builders:
                protected.add(cand)
    protected.difference_update(own_editable)
    return protected


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
    "inner": "inner_dim",
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


def _filter_callable_kwargs(fn: object, kwargs: Mapping[str, object]) -> dict[str, object]:
    """Filter *kwargs* down to parameters accepted by *fn*."""
    if not callable(fn):
        return {}
    sig = inspect.signature(fn)
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def _norm_flag_seq(raw: object) -> list[str]:
    """Normalize a compiler flag scalar or sequence into unique stripped strings."""
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


def _load_module_and_run_selector(
    file_path: Path,
    regime_spec: RegimeTargetSpec,
    dims: Mapping[str, object] | None = None,
) -> tuple[object, dict[str, object], dict[str, object], list[str], str, dict[str, object]]:
    """Load *file_path* and execute the regime config selector with merged dimensions."""
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
    if "half_dim" not in context:
        raw_d = context.get("d_dim")
        if isinstance(raw_d, int) and not isinstance(raw_d, bool):
            context["half_dim"] = max(1, raw_d // 2)

    pass_configs: dict[str, object] = {}
    compile_flags: list[str] = []
    combine_kwargs: dict[str, object] = {}

    def _absorb_pass_configs(raw_pc: object) -> None:
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

    _absorb_pass_configs(context.pop("pass_configs", None))
    for fl in _norm_flag_seq(context.pop("compile_flags", None)):
        if fl not in compile_flags:
            compile_flags.append(fl)

    selector_name = regime_spec.config_selector_fn
    if not (selector_name and callable(getattr(mod, selector_name, None))):
        for cand_sym in regime_spec.editable_symbols:
            if cand_sym.startswith("_select_") and callable(getattr(mod, cand_sym, None)):
                selector_name = cand_sym
                break
    if selector_name and callable(getattr(mod, selector_name, None)):
        selector_fn = getattr(mod, selector_name)
        sel_args = _filter_callable_kwargs(selector_fn, context)
        sel_out = selector_fn(**sel_args)
        if isinstance(sel_out, Mapping):
            _absorb_pass_configs(sel_out.get("pass_configs"))
            for fl in _norm_flag_seq(sel_out.get("compile_flags")):
                if fl not in compile_flags:
                    compile_flags.append(fl)
            for k, v in sel_out.items():
                if k not in (
                    "pass_configs",
                    "compile_flags",
                    "builder_kwargs",
                    "combine_kwargs",
                ):
                    context[str(k)] = v
            raw_bk = sel_out.get("builder_kwargs")
            if isinstance(raw_bk, Mapping):
                for k, v in raw_bk.items():
                    context[str(k)] = v
            raw_ck = sel_out.get("combine_kwargs")
            if isinstance(raw_ck, Mapping):
                combine_kwargs = {str(k): v for k, v in raw_ck.items()}
    return mod, context, pass_configs, compile_flags, selector_name, combine_kwargs


def _resolve_selector_compile_flags(
    module_source: str,
    regime_spec: RegimeTargetSpec,
    dims: Mapping[str, object] | None = None,
) -> tuple[str, ...]:
    """Invoke regime config selector if present and return its compile flags."""
    try:
        with tempfile.TemporaryDirectory(prefix="tf_resolve_flags_") as tmp_dir:
            tmp_file = Path(tmp_dir) / "module_src.py"
            tmp_file.write_text(module_source, encoding="utf-8")
            _, _, _, compile_flags, _, _ = _load_module_and_run_selector(
                tmp_file, regime_spec, dims=dims
            )
            return tuple(compile_flags)
    except (
        SyntaxError,
        AttributeError,
        ImportError,
        KeyError,
        NameError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return ()


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
        mod, context, pass_configs, compile_flags, selector_name, combine_kwargs = (
            _load_module_and_run_selector(file_path, regime_spec, dims=dims)
        )
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
            combine_kwargs=combine_kwargs,
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
    """Collect top-level function and single-target assignment spans from *tree*."""
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


def _extract_trackb_cpp_source(
    module_source: str,
    regime_spec: RegimeTargetSpec,
) -> tuple[str, str] | None:
    """Return `(symbol_name, cpp_source)` for the first Track-B C++20 string constant."""
    return extract_trackb_cpp_source(module_source, regime_spec.trackb_symbols)


def _top_level_symbol_sources(module_source: str) -> dict[str, str]:
    """Return top-level symbol source text keyed by symbol name."""
    tree = ast.parse(module_source)
    lines = module_source.splitlines(keepends=True)
    return {
        sym.name: "".join(lines[sym.start - 1 : sym.end]).strip()
        for sym in _collect_top_level_symbols(tree)
    }


def _reject_trackb_context_markers(candidate_text: str, regime_tag: str) -> None:
    """Reject a Track-B reply that echoes read-only sibling declaration markers."""
    markers = ("READONLY_SIBLING_DECLARATION", "READ-ONLY SIBLING KERNEL", "READ-ONLY HELPER")
    found = [marker for marker in markers if marker in candidate_text]
    if found:
        raise ValueError(
            f"Track-B candidate for regime {regime_tag!r} echoes read-only markers {found}; "
            "reply with a full symbol unit or SEARCH/REPLACE hunk instead"
        )


def _strip_trackb_patch_hunks(patch_text: str) -> str:
    """Return *patch_text* with its SEARCH/REPLACE blocks removed."""
    residual = patch_text
    for pattern in _SEARCH_REPLACE_PATTERNS:
        residual = pattern.sub("", residual)
    return residual.strip()


def _looks_like_bare_trackb_cpp(candidate_text: str) -> bool:
    """Return True when a Track-B reply carries bare C++ units rather than Python."""
    if _parses_as_python(candidate_text):
        return False
    if re.search(r"(?m)^\s*(?:async\s+)?def\s+", candidate_text):
        return False
    if "<<<<<<< SEARCH" in candidate_text or (
        "=======" in candidate_text and ">>>>>>> REPLACE" in candidate_text
    ):
        return False
    _, units = _split_cpp_units(candidate_text)
    if any(unit.kind in ("kernel", "helper") and unit.name for unit in units):
        return True
    return bool(
        re.search(r"(?m)^\s*#include\b", candidate_text)
        or re.search(r"(?m)^\s*(?:static\s+inline|constexpr)\b", candidate_text)
    )


def _parses_as_python(candidate_text: str) -> bool:
    """Return True when *candidate_text* is a syntactically valid Python snippet."""
    try:
        ast.parse(candidate_text)
    except SyntaxError:
        return False
    return True


def _render_trackb_tu_from_module(
    module_source: str,
    cpp_source: str,
    regime_spec: RegimeTargetSpec,
    *,
    require_renderer: bool,
    anchor_dims_override: Mapping[str, object] | None = None,
) -> str:
    """Render a module's Track-B launcher with the shared dimension merge."""
    tree = ast.parse(module_source)
    if _find_top_level_func(tree, ("_render_trackb_tu",)) is None:
        if require_renderer:
            raise ValueError(
                "Track-B seed must declare def _render_trackb_tu(**dims: object) -> str"
            )
        return cpp_source
    dims: dict[str, object] = {
        **regime_spec.sample_builder_kwargs,
        **regime_spec.representative_hir_dims,
        **regime_spec.representative_anchor_dims,
    }
    if anchor_dims_override:
        dims.update(anchor_dims_override)
    try:
        resolved = resolve_regime_config(module_source, regime_spec, dims=dims)
        alias_sources = {target: source for source, target in _HIR_DIM_ALIASES.items()}
        for key, value in resolved.builder_kwargs.items():
            source = alias_sources.get(key)
            if source is not None and source in dims and dims[source] != value:
                continue
            dims.setdefault(key, value)
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
                raise ValueError(
                    "Track-B seed must declare def _render_trackb_tu(**dims: object) -> str"
                )
            return cpp_source
        rendered = render(**dims)
    if not isinstance(rendered, str) or not rendered.strip():
        if require_renderer:
            raise ValueError("_render_trackb_tu returned an empty C++ translation unit")
        return cpp_source
    return rendered


def _active_trackb_kernel(
    instantiated_tu: str,
    raw_cpp_source: str,
    *,
    regime_tag: str,
    module_source: str | None = None,
) -> str:
    """Resolve the launched kernel from instantiation truth rather than name heuristics."""
    resolved = resolve_instantiated_trackb_kernel(instantiated_tu, raw_cpp_source)
    if resolved is not None:
        return resolved
    kernels = sorted(_trackb_kernel_names(raw_cpp_source))
    if len(kernels) == 1:
        return kernels[0]
    if not kernels:
        raise ValueError(
            f"Track-B instantiation for regime {regime_tag!r} contains no __global__ kernel"
        )
    launched, primary = _launched_trackb_kernels(
        instantiated_tu,
        raw_cpp_source,
        set(kernels),
    )
    if len(primary) == 1:
        return next(iter(primary))
    if module_source:
        mod_selected = select_active_trackb_kernel(
            raw_cpp_source,
            regime_tag=regime_tag,
            module_source=module_source,
        )
        if mod_selected is not None:
            return mod_selected
    if not launched:
        raise ValueError(
            f"Track-B instantiation for regime {regime_tag!r} contains multiple kernels "
            f"{kernels} and no launcher stub; the instantiated translation unit must launch "
            "exactly one primary kernel"
        )
    raise ValueError(
        f"Track-B instantiation for regime {regime_tag!r} launches kernels {sorted(launched)} "
        f"with no unique top-level launch target among {sorted(primary)}; the launcher stub "
        "must launch exactly one primary kernel outside conditional branches"
    )


def _launched_trackb_kernels(
    instantiated_tu: str,
    raw_cpp_source: str,
    kernels: set[str],
) -> tuple[set[str], set[str]]:
    """Return all launched kernels and those launched at the launcher's top level."""
    masked = strip_cpp_comments_and_strings(instantiated_tu)
    launch_region = (
        masked[len(raw_cpp_source) :]
        if instantiated_tu.startswith(raw_cpp_source)
        else masked
    )
    launches: list[tuple[int, str]] = []
    depth = 0
    scanned = 0
    for match in _LAUNCH_CONFIG_RE.finditer(launch_region):
        depth += launch_region.count("{", scanned, match.start()) - launch_region.count(
            "}", scanned, match.start()
        )
        scanned = match.start()
        name = match.group(1)
        if name in kernels:
            launches.append((depth, name))
    if not launches:
        return set(), set()
    shallowest = min(depth for depth, _name in launches)
    return (
        {name for _depth, name in launches},
        {name for depth, name in launches if depth == shallowest},
    )


def _ordered_top_level_trackb_kernels(
    instantiated_tu: str,
    raw_cpp_source: str,
    kernels: set[str],
) -> tuple[str, ...]:
    """Return distinct top-level launched `__global__` kernels in launch order."""
    masked = strip_cpp_comments_and_strings(instantiated_tu)
    regions = (
        [masked[len(raw_cpp_source) :], masked]
        if instantiated_tu.startswith(raw_cpp_source) and len(instantiated_tu) > len(raw_cpp_source)
        else [masked]
    )
    for launch_region in regions:
        launches: list[tuple[int, str]] = []
        depth = 0
        scanned = 0
        for match in _LAUNCH_CONFIG_RE.finditer(launch_region):
            depth += launch_region.count("{", scanned, match.start()) - launch_region.count(
                "}", scanned, match.start()
            )
            scanned = match.start()
            name = match.group(1)
            if name in kernels:
                launches.append((depth, name))
        if launches:
            shallowest = min(depth for depth, _name in launches)
            ordered: list[str] = []
            for depth, name in launches:
                if depth == shallowest and name not in ordered:
                    ordered.append(name)
            return tuple(ordered)
    return ()


def _trackb_kernel_names(cpp_source: str) -> set[str]:
    """Return the set of `__global__` kernel names defined in *cpp_source*."""
    _, units = _split_cpp_units(cpp_source)
    return {unit.name for unit in units if unit.kind == "kernel" and unit.name}


def _trackb_units_by_name(cpp_source: str) -> dict[str, str]:
    """Return named top-level Track-B unit text keyed by symbol name."""
    _, units = _split_cpp_units(cpp_source)
    return {unit.name: unit.text for unit in units if unit.name}


def _normalized_trackb_unit(text: str) -> str:
    """Collapse whitespace so pure reformatting is not treated as a unit edit."""
    return " ".join(text.split())


def _assert_trackb_replacement_scoped(
    orig_cpp: str,
    new_cpp: str,
    regime_spec: RegimeTargetSpec,
    *,
    module_source: str,
    active_kernel: str | None,
    allowed_added_units: Collection[str] = (),
) -> None:
    """Reject a full-TU replacement that edits frozen siblings or shared helpers."""
    orig_units = _trackb_units_by_name(orig_cpp)
    new_units = _trackb_units_by_name(new_cpp)
    if active_kernel is None:
        active_kernel = select_active_trackb_kernel(
            orig_cpp,
            regime_tag=regime_spec.regime_tag,
            module_source=module_source,
        )
    if active_kernel is None:
        raise ValueError(
            f"Track-B full replacement for regime {regime_spec.regime_tag!r} has no resolvable "
            "active kernel; reply with SEARCH/REPLACE hunks or one bare C++ unit"
        )
    editable = {
        unit.name
        for unit in _scoped_editable_units(
            orig_cpp,
            active_kernel,
            allow_exclusive_helpers=True,
        )
    }
    dropped = sorted(set(orig_units) - set(new_units))
    if dropped:
        raise ValueError(
            f"Track-B replacement for regime {regime_spec.regime_tag!r} drops units {dropped}; "
            "reply with SEARCH/REPLACE hunks or one bare C++ unit per changed function"
        )
    allowed_added = set(allowed_added_units)
    added = sorted(set(new_units) - set(orig_units) - allowed_added)
    if added:
        raise ValueError(
            f"Track-B replacement for regime {regime_spec.regime_tag!r} adds units {added}; "
            "reply with SEARCH/REPLACE hunks or one bare C++ unit per changed function"
        )
    changed = sorted(
        name
        for name in orig_units
        if _normalized_trackb_unit(orig_units[name])
        != _normalized_trackb_unit(new_units[name])
    )
    frozen = sorted(name for name in changed if name not in editable)
    if frozen:
        raise ValueError(
            f"Track-B replacement for regime {regime_spec.regime_tag!r} edits frozen sibling "
            f"kernels or shared helpers {frozen}; only active kernel {active_kernel!r} and its "
            "exclusive helpers may change"
        )


def _case_ratio_predicate_entries(ratios: CasePhysicsRatios | None) -> dict[str, object]:
    """Return collision-free dimensionless predicate keys for the representative case."""
    if ratios is None:
        return {}
    return {
        f"case_{key.removeprefix('case_')}": value for key, value in ratios.to_dict().items()
    }


def _detect_trackb_dead_code_edit(
    parent_source: str,
    candidate_source: str,
    regime_spec: RegimeTargetSpec,
) -> str | None:
    """Reject candidates that edit only inactive TileLang fallbacks in a Track-B regime."""
    if (
        regime_spec.runtime_backend not in ("ascendc_trackb", "hybrid_trackb")
        or not regime_spec.trackb_symbols
    ):
        return None
    parent_syms = _top_level_symbol_sources(parent_source)
    cand_syms = _top_level_symbol_sources(candidate_source)
    if not parent_syms or not cand_syms:
        return None
    changed = {
        name
        for name in (set(parent_syms) | set(cand_syms))
        if parent_syms.get(name, "") != cand_syms.get(name, "")
    }
    if not changed:
        return None
    active_syms = set(regime_spec.trackb_symbols)
    if regime_spec.config_selector_fn:
        active_syms.add(regime_spec.config_selector_fn)
    if not (changed & active_syms):
        return (
            f"DEAD_CODE_TRACKB_EDIT: regime {regime_spec.regime_tag!r} executes "
            f"runtime_backend={regime_spec.runtime_backend} via "
            f"{list(regime_spec.trackb_symbols)}; "
            f"candidate modified only inactive fallback symbols {sorted(changed)}."
        )
    return None


def _resolve_cann_root_from_env_file(env_file: str | Path | None) -> Path | None:
    """Resolve the CANN toolkit root directory from a managed env file."""
    if env_file is None:
        return None
    p = Path(env_file)
    if not p.is_file():
        return None
    vals = parse_simple_env_file(p)
    setup_raw = vals.get("TILEFOUNDRY_NPU_CANN_SETUP", "").strip()
    if not setup_raw:
        return None
    parent = Path(setup_raw).expanduser().parent
    return parent if parent.is_dir() else None


def _find_top_level_func(tree: ast.Module, names: Sequence[str]) -> ast.FunctionDef | None:
    """Find the first top-level `ast.FunctionDef` matching one of *names*."""
    by_name = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for name in names:
        if name in by_name:
            return by_name[name]
    return None


def _func_line_span(node: ast.FunctionDef) -> tuple[int, int]:
    """Return 1-based inclusive `(start, end)` line numbers including decorators."""
    start = min((dec.lineno for dec in node.decorator_list), default=node.lineno)
    end = node.end_lineno or node.lineno
    return start, end


def _resolve_operator_entrypoint_safe(operator_name: str) -> str:
    """Resolve the public operator entrypoint function name without failing."""
    try:
        return get_operator_entrypoint(operator_name)
    except (KeyError, FileNotFoundError):
        return str(operator_name).rsplit("/", 1)[-1].strip()


def _other_regime_forbidden_symbols(
    regime_spec: RegimeTargetSpec,
    *,
    operator_entrypoint: str = "",
    sibling_specs: Sequence[RegimeTargetSpec] = (),
    module_source: str = "",
) -> set[str]:
    """Return symbol names protected from edits by the active regime."""
    return _derive_dynamic_protected_symbols(
        module_source,
        regime_spec,
        operator_entrypoint=operator_entrypoint,
        sibling_specs=sibling_specs,
    )


def _is_trackb_regime_spec(regime_spec: RegimeTargetSpec) -> bool:
    """Return True when *regime_spec* actively executes Track-B C++ symbols."""
    return (
        regime_spec.runtime_backend in ("ascendc_trackb", "hybrid_trackb")
        and bool(regime_spec.trackb_symbols)
    )


def _filter_trackb_slice_symbols(
    symbols: Sequence[_TopLevelSymbol],
    regime_spec: RegimeTargetSpec,
    editable_set: set[str],
) -> list[_TopLevelSymbol]:
    """Keep only `_select_*`, `_render_trackb_tu`, and `trackb_symbols` for Track-B slices."""
    trackb_sym_set = set(regime_spec.trackb_symbols)
    kept: list[_TopLevelSymbol] = []
    for sym in symbols:
        if sym.name in trackb_sym_set or sym.name == "_render_trackb_tu":
            kept.append(sym)
        elif regime_spec.config_selector_fn and sym.name == regime_spec.config_selector_fn:
            kept.append(sym)
        elif sym.name.startswith("_select_") and (not editable_set or sym.name in editable_set):
            kept.append(sym)
    return kept


def extract_regime_snippet(
    module_source: str,
    regime_spec: RegimeTargetSpec,
    *,
    operator_entrypoint: str = "",
    sibling_specs: Sequence[RegimeTargetSpec] = (),
) -> tuple[str, str]:
    """Extract target regime multi-symbol slice (selector, builders, helpers) cleanly."""
    tree = ast.parse(module_source)
    lines = module_source.splitlines(keepends=True)
    symbols = _collect_top_level_symbols(tree)
    ast_by_name = _collect_top_level_ast_nodes(tree)
    editable_set = set(regime_spec.editable_symbols)
    prefix = regime_spec.allowed_helper_prefix

    if _is_trackb_regime_spec(regime_spec):
        trackb_selected = _filter_trackb_slice_symbols(symbols, regime_spec, editable_set)
        if trackb_selected:
            func_names = {sym.name for sym in trackb_selected if sym.kind == "func"}
            primary = ""
            if regime_spec.config_selector_fn and regime_spec.config_selector_fn in func_names:
                primary = regime_spec.config_selector_fn
            if not primary:
                primary = next(
                    (sym.name for sym in trackb_selected if sym.kind == "func"),
                    trackb_selected[0].name,
                )
            blocks = [
                "".join(lines[sym.start - 1 : sym.end]).rstrip() for sym in trackb_selected
            ]
            return primary, "\n\n".join(blocks).rstrip() + "\n"

    seed_symbols: list[_TopLevelSymbol] = []
    if editable_set:
        seed_symbols = [
            sym
            for sym in symbols
            if sym.name in editable_set or (prefix and sym.name.startswith(prefix))
        ]
    if not seed_symbols:
        node = _find_top_level_func(tree, regime_spec.builder_candidates)
        if node is None:
            raise ValueError(
                f"No builder function matching {regime_spec.builder_candidates} found in module."
            )
        seed_symbols = [sym for sym in symbols if sym.name == node.name]

    seed_names = [sym.name for sym in seed_symbols]
    seed_name_set = set(seed_names)
    ws_planner_names = {
        "_plan_workspace_bytes",
        f"_{regime_spec.regime_tag}_plan_workspace_bytes",
    }
    for sym in symbols:
        if sym.name in ws_planner_names and sym.name not in seed_name_set:
            seed_symbols.append(sym)
            seed_names.append(sym.name)
            seed_name_set.add(sym.name)
    closure_helpers: set[str] = set()
    if regime_spec.runtime_backend not in ("ascendc_trackb", "hybrid_trackb"):
        excluded = _other_regime_forbidden_symbols(
            regime_spec,
            operator_entrypoint=operator_entrypoint,
            sibling_specs=sibling_specs,
            module_source=module_source,
        )
        if editable_set:
            for cand in regime_spec.builder_candidates:
                if cand not in editable_set:
                    excluded.add(cand)
        private_func_candidates = {
            sym.name
            for sym in symbols
            if sym.kind == "func"
            and sym.name.startswith("_")
            and sym.name not in excluded
            and sym.name not in seed_name_set
        }
        closure_seed_names = [
            name for name in seed_names if name != regime_spec.config_selector_fn
        ] or seed_names
        closure_helpers = _collect_referenced_helper_names(
            ast_by_name,
            helper_names=private_func_candidates,
            seed_roots=closure_seed_names,
        )
    selected = [
        sym
        for sym in symbols
        if sym.name in seed_name_set or sym.name in closure_helpers
    ]
    func_names = {sym.name for sym in selected if sym.kind == "func"}
    primary = ""
    for cand in regime_spec.builder_candidates:
        if cand in func_names:
            primary = cand
            break
    if not primary:
        primary = next(
            (sym.name for sym in selected if sym.kind == "func"),
            selected[0].name,
        )
    blocks = ["".join(lines[sym.start - 1 : sym.end]).rstrip() for sym in selected]
    return primary, "\n\n".join(blocks).rstrip() + "\n"


def split_readonly_host_tail(
    snippet_source: str,
    protected_exports: Collection[str] = DEFAULT_READONLY_HOST_EXPORTS,
) -> tuple[str, str]:
    """Separate protected module-tail bindings from *snippet_source* into a read-only tail."""
    protected_set = {str(name) for name in protected_exports if str(name)}
    if not protected_set or not snippet_source.strip():
        return snippet_source, ""
    try:
        tree = ast.parse(snippet_source)
    except SyntaxError:
        return snippet_source, ""
    lines = snippet_source.splitlines(keepends=True)
    protected_spans: list[tuple[int, int]] = []
    tail_blocks: list[str] = []
    for node in tree.body:
        names: list[str] = []
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names = [node.name]
        elif isinstance(node, ast.Assign):
            names = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names = [node.target.id]
        if any(name in protected_set for name in names):
            start, end = _stmt_line_span(node)
            protected_spans.append((start, end))
            block = "".join(lines[start - 1 : end]).rstrip()
            if block:
                tail_blocks.append(block)
    if not protected_spans:
        return snippet_source, ""
    remaining = list(lines)
    for start, end in reversed(protected_spans):
        del_end = end
        while del_end < len(remaining) and remaining[del_end].strip() == "":
            del_end += 1
        del remaining[start - 1 : del_end]
    editable = "".join(remaining).rstrip()
    editable_out = f"{editable}\n" if editable else ""
    readonly_tail = "\n\n".join(tail_blocks).rstrip()
    return editable_out, readonly_tail


def _is_prim_func_decorator(dec: ast.expr) -> bool:
    """Return True when *dec* denotes a `prim_func` decorator."""
    if isinstance(dec, ast.Name):
        return dec.id == "prim_func"
    if isinstance(dec, ast.Attribute):
        return dec.attr == "prim_func"
    if isinstance(dec, ast.Call):
        return _is_prim_func_decorator(dec.func)
    return False


def _has_prim_func_builder(
    module_source: str,
    builder_candidates: Sequence[str],
) -> bool:
    """Return True when any builder in *builder_candidates* contains a `@T.prim_func`."""
    if not builder_candidates:
        return False
    try:
        tree = ast.parse(module_source)
    except SyntaxError:
        return False
    candidate_set = set(builder_candidates)
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name not in candidate_set:
            continue
        for sub in ast.walk(node):
            if not isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)) or sub is node:
                continue
            if any(_is_prim_func_decorator(dec) for dec in sub.decorator_list):
                return True
    return False


def _extract_func_signature_header(lines: Sequence[str], node: ast.FunctionDef) -> str:
    """Extract the `def ...:` header signature for *node* from *lines*."""
    if node.body and node.body[0].lineno > node.lineno:
        end_idx = node.body[0].lineno - 1
        raw = "".join(lines[node.lineno - 1 : end_idx]).rstrip()
    else:
        raw = "".join(lines[node.lineno - 1 : node.lineno]).split(":", 1)[0].rstrip() + ":"
    return textwrap.dedent(raw).strip()


def _extract_regime_abi_context(
    module_source: str,
    regime_spec: RegimeTargetSpec,
    *,
    golden_abi_block: str = "",
) -> str:
    """Extract slice, helper builder, and @T.prim_func ABI signatures for the target regime."""
    tree = ast.parse(module_source)
    lines = module_source.splitlines(keepends=True)
    top_funcs = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    editable_set = set(regime_spec.editable_symbols)
    prefix = regime_spec.allowed_helper_prefix
    seed_names = [
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and (
            node.name in editable_set
            or (prefix and node.name.startswith(prefix))
            or node.name in regime_spec.builder_candidates
        )
    ]
    if not seed_names:
        return golden_abi_block

    visited: set[str] = set()
    ordered_slice_funcs: list[ast.FunctionDef] = []
    ordered_helper_funcs: list[ast.FunctionDef] = []
    prim_func_sigs: list[str] = []
    seen_prim_sigs: set[str] = set()
    queue = list(seed_names)
    while queue:
        fname = queue.pop(0)
        if fname in visited:
            continue
        visited.add(fname)
        fnode = top_funcs.get(fname)
        if fnode is None:
            continue
        is_prefix_helper = bool(prefix and fname.startswith(prefix))
        if fname in editable_set or fname in regime_spec.builder_candidates:
            ordered_slice_funcs.append(fnode)
        elif not is_prefix_helper:
            ordered_helper_funcs.append(fnode)
        for sub in ast.walk(fnode):
            if isinstance(sub, ast.FunctionDef) and sub is not fnode:
                if any(_is_prim_func_decorator(dec) for dec in sub.decorator_list):
                    sig = "@T.prim_func\n" + _extract_func_signature_header(lines, sub)
                    if sig not in seen_prim_sigs:
                        seen_prim_sigs.add(sig)
                        prim_func_sigs.append(sig)
            elif isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                callee = sub.func.id
                if callee in top_funcs and callee not in visited:
                    queue.append(callee)

    blocks: list[str] = []
    for hnode in ordered_helper_funcs:
        blocks.append(_extract_func_signature_header(lines, hnode))
    for snode in ordered_slice_funcs:
        blocks.append(_extract_func_signature_header(lines, snode))
    blocks.extend(prim_func_sigs)
    if not blocks:
        return golden_abi_block
    sigs_text = "\n\n".join(blocks)
    primfunc_abi = (
        "### IMMUTABLE HOST / PRIMFUNC ABI CONTRACT\n"
        f"```python\n{sigs_text}\n```\n"
        "Any custom @T.prim_func returned for this slice MUST keep the exact tensor/scalar "
        "parameter order, ranks, and dtypes above expected by _get_kernel (do NOT substitute "
        "the PyTorch golden entrypoint parameter list, and do NOT import tilelang.language as T "
        "or use T.Scope). _select_*_config() may return optional 'builder_kwargs': {...} "
        "forwarded to custom builders for compile-time constexpr specialization.\n\n"
    )
    return f"{golden_abi_block}{primfunc_abi}"


def apply_ast_patch(
    module_source: str,
    target_func: str,
    candidate_dsl: str,
) -> str:
    """Replace *target_func* in *module_source* using Python AST line anchors."""
    orig_tree = ast.parse(module_source)
    cand_tree = ast.parse(candidate_dsl)

    cand_node = _find_top_level_func(cand_tree, (target_func,))
    if cand_node is None:
        raise ValueError(f"Candidate source must define function {target_func!r}.")

    cand_lines = candidate_dsl.splitlines(keepends=True)
    c_start, c_end = _func_line_span(cand_node)
    replacement = "".join(cand_lines[c_start - 1 : c_end])
    if not replacement.endswith("\n"):
        replacement += "\n"

    orig_node = _find_top_level_func(orig_tree, (target_func,))
    if orig_node is None:
        raise ValueError(f"Original module missing target function {target_func!r}.")

    o_start, o_end = _func_line_span(orig_node)
    orig_lines = module_source.splitlines(keepends=True)

    patched = "".join([*orig_lines[: o_start - 1], replacement, *orig_lines[o_end:]])
    ast.parse(patched)
    return patched


def _collect_top_level_ast_nodes(tree: ast.Module) -> dict[str, ast.stmt]:
    """Collect top-level function and single-target assignment AST nodes by name."""
    nodes: dict[str, ast.stmt] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            nodes[node.name] = node
        elif isinstance(node, ast.Assign):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                nodes[node.targets[0].id] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            nodes[node.target.id] = node
    return nodes


def _collect_referenced_helper_names(
    merged_ast_by_name: Mapping[str, ast.stmt],
    *,
    helper_names: set[str],
    seed_roots: Sequence[str],
) -> set[str]:
    """Compute transitive closure of helper symbols referenced from *seed_roots*."""
    live: set[str] = set()
    queue = list(seed_roots)
    visited: set[str] = set()
    while queue:
        sym = queue.pop(0)
        if sym in visited:
            continue
        visited.add(sym)
        node = merged_ast_by_name.get(sym)
        if node is None:
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                ref = sub.id
                if ref in helper_names and ref not in live:
                    live.add(ref)
                    if ref not in visited:
                        queue.append(ref)
    return live


def _ast_char_offset(lines: Sequence[str], lineno: int, col_offset: int) -> int:
    """Convert 1-based AST line and UTF-8 byte column offset to character index."""
    prefix = sum(len(lines[idx]) for idx in range(lineno - 1))
    line_bytes = lines[lineno - 1].encode("utf-8")
    return prefix + len(line_bytes[:col_offset].decode("utf-8"))


def _format_raw_cpp_literal(cpp_text: str) -> str:
    """Format *cpp_text* as a raw triple-quoted Python string literal when safe."""
    if '"""' not in cpp_text and not cpp_text.endswith('"'):
        return f'r"""{cpp_text}"""'
    return repr(cpp_text)


def _find_top_level_str_assign_span(
    tree: ast.Module,
    lines: Sequence[str],
    target_names: set[str],
) -> list[tuple[str, str, int, int]]:
    """Find character spans of top-level string constant assignments in *target_names*."""
    hits: list[tuple[str, str, int, int]] = []
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
        if (
            name in target_names
            and isinstance(val, ast.Constant)
            and isinstance(val.value, str)
            and val.end_lineno is not None
            and val.end_col_offset is not None
        ):
            start_pos = _ast_char_offset(lines, val.lineno, val.col_offset)
            end_pos = _ast_char_offset(lines, val.end_lineno, val.end_col_offset)
            hits.append((name, val.value, start_pos, end_pos))
    return hits


def _replace_trackb_constant(module_source: str, sym_name: str, cpp_source: str) -> str:
    """Rewrite one top-level Track-B C++ string constant in *module_source*."""
    orig_tree = ast.parse(module_source)
    lines = module_source.splitlines(keepends=True)
    hits = _find_top_level_str_assign_span(orig_tree, lines, {sym_name})
    if not hits:
        raise ValueError(f"Track-B symbol {sym_name!r} assignment not found in module_source.")
    _, _, start_pos, end_pos = hits[0]
    patched = (
        module_source[:start_pos]
        + _format_raw_cpp_literal(cpp_source)
        + module_source[end_pos:]
    )
    ast.parse(patched)
    return patched


def _patch_anchor_miss_error_type() -> type[ValueError]:
    """Resolve PatchAnchorMissError from cpp_slice_protocol when available."""
    exc_cls = getattr(_cpp_slice_protocol_mod, "PatchAnchorMissError", None)
    if isinstance(exc_cls, type) and issubclass(exc_cls, ValueError):
        return exc_cls
    return ValueError


def _format_patch_anchor_miss_diagnostic(
    module_source: str,
    regime_spec: RegimeTargetSpec,
    reason: str,
    *,
    active_kernel: str | None = None,
) -> str:
    """Format an objective PATCH_ANCHOR_MISS diagnostic with active unit anchor lines."""
    extracted = _extract_trackb_cpp_source(module_source, regime_spec)
    if extracted is None:
        return f"PATCH_ANCHOR_MISS: {reason}"
    _sym_name, orig_cpp = extracted
    resolved_kernel = active_kernel or select_active_trackb_kernel(
        orig_cpp,
        regime_tag=regime_spec.regime_tag,
        module_source=module_source,
    )
    editable_units = _scoped_editable_units(
        orig_cpp, resolved_kernel, allow_exclusive_helpers=True
    )
    target_unit = next(
        (u for u in editable_units if u.name == resolved_kernel),
        editable_units[0] if editable_units else None,
    )
    if target_unit is None:
        return f"PATCH_ANCHOR_MISS: {reason} (active_kernel={resolved_kernel})"
    unit_lines = [ln.rstrip() for ln in target_unit.text.splitlines() if ln.strip()]
    first_lines = " | ".join(unit_lines[:3])
    last_lines = " | ".join(unit_lines[-3:])
    return (
        f"PATCH_ANCHOR_MISS: {reason} "
        f"(active_kernel={resolved_kernel}; first_lines={first_lines!r}; "
        f"last_lines={last_lines!r})"
    )


def _is_anchor_miss_value_error(exc: ValueError) -> bool:
    """Return True when *exc* represents an anchor miss in Track-B patching."""
    miss_cls = _patch_anchor_miss_error_type()
    if miss_cls is not ValueError and isinstance(exc, miss_cls):
        return True
    msg = str(exc).lower()
    return any(
        tok in msg
        for tok in (
            "did not match",
            "ambiguous search",
            "outside active kernel",
            "outside the editable scope",
            "not found in c++ source",
            "must contain at least one valid",
            "identical",
            "applied == 0",
        )
    )


def _apply_scoped_trackb_patch_to_module(
    module_source: str,
    regime_spec: RegimeTargetSpec,
    patch_text: str,
    *,
    active_kernel: str | None = None,
) -> str:
    """Apply SEARCH/REPLACE hunks to the active Track-B kernel inside *module_source*."""
    extracted = _extract_trackb_cpp_source(module_source, regime_spec)
    if extracted is None:
        raise ValueError(
            f"Original module missing Track-B C++ constant in {regime_spec.trackb_symbols}."
        )
    sym_name, orig_cpp = extracted
    resolved_kernel = active_kernel or select_active_trackb_kernel(
        orig_cpp,
        regime_tag=regime_spec.regime_tag,
        module_source=module_source,
    )
    miss_cls = _patch_anchor_miss_error_type()
    try:
        updated_cpp, applied = apply_scoped_trackb_patch(
            orig_cpp,
            patch_text,
            active_kernel=resolved_kernel,
        )
    except ValueError as exc:
        if "PATCH_ANCHOR_MISS:" in str(exc):
            raise
        if _is_anchor_miss_value_error(exc):
            diag = _format_patch_anchor_miss_diagnostic(
                module_source,
                regime_spec,
                str(exc),
                active_kernel=resolved_kernel,
            )
            raise miss_cls(diag) from exc
        raise
    if applied == 0 or updated_cpp.strip() == orig_cpp.strip():
        diag = _format_patch_anchor_miss_diagnostic(
            module_source,
            regime_spec,
            "Track-B SEARCH/REPLACE produced identical C++ source",
            active_kernel=resolved_kernel,
        )
        raise miss_cls(diag)
    return _replace_trackb_constant(module_source, sym_name, updated_cpp)


def _apply_spliced_trackb_units_to_module(
    module_source: str,
    regime_spec: RegimeTargetSpec,
    candidate_cpp: str,
    *,
    active_kernel: str | None = None,
) -> str:
    """Replace same-name bare C++ units of the active kernel without touching siblings."""
    extracted = _extract_trackb_cpp_source(module_source, regime_spec)
    if extracted is None:
        raise ValueError(
            f"Original module missing Track-B C++ constant in {regime_spec.trackb_symbols}."
        )
    sym_name, orig_cpp = extracted
    resolved_kernel = active_kernel or select_active_trackb_kernel(
        orig_cpp,
        regime_tag=regime_spec.regime_tag,
        module_source=module_source,
    )
    miss_cls = _patch_anchor_miss_error_type()
    try:
        updated_cpp, applied = splice_cpp_units_into_tu(
            orig_cpp,
            candidate_cpp,
            active_kernel=resolved_kernel,
        )
    except ValueError as exc:
        if "PATCH_ANCHOR_MISS:" in str(exc):
            raise
        if _is_anchor_miss_value_error(exc):
            diag = _format_patch_anchor_miss_diagnostic(
                module_source,
                regime_spec,
                str(exc),
                active_kernel=resolved_kernel,
            )
            raise miss_cls(diag) from exc
        raise
    if applied == 0 or updated_cpp.strip() == orig_cpp.strip():
        diag = _format_patch_anchor_miss_diagnostic(
            module_source,
            regime_spec,
            "Track-B C++ unit splice produced identical C++ source",
            active_kernel=resolved_kernel,
        )
        raise miss_cls(diag)
    return _replace_trackb_constant(module_source, sym_name, updated_cpp)


def _trackb_python_residual(patch_text: str) -> str:
    """Return the valid Python snippet outside SEARCH/REPLACE blocks, else empty string."""
    residual = _strip_trackb_patch_hunks(patch_text)
    if not residual:
        return ""
    try:
        ast.parse(residual)
    except SyntaxError:
        return ""
    return residual


def _extract_p2_focus_lines_and_diag(
    p2_diag: Mapping[str, object] | None,
) -> tuple[tuple[int, ...], str]:
    """Extract sorted source focus line numbers and raw diagnostic text from p2_diag."""
    if not p2_diag:
        return (), ""
    focus: set[int] = set()
    diag_parts: list[str] = []
    for key in ("diagnostic_text", "error_message", "compiler_diagnostic"):
        raw_val = p2_diag.get(key)
        if isinstance(raw_val, str) and raw_val.strip():
            diag_parts.append(raw_val.strip())
            focus.update(extract_cpp_diagnostic_lines(raw_val))
    raw_hotspots = p2_diag.get("p2_hotspots") or p2_diag.get("source_line_hotspots") or ()
    if isinstance(raw_hotspots, Sequence) and not isinstance(raw_hotspots, (str, bytes)):
        for spot in list(raw_hotspots):
            if isinstance(spot, Mapping):
                loc = spot.get("source_location") or ""
            elif hasattr(spot, "source_location"):
                loc = getattr(spot, "source_location", "") or ""
            else:
                loc = spot
            loc_str = str(loc).strip()
            if not loc_str:
                continue
            for m in re.finditer(r":(\d+)(?::\d+)?\b", loc_str):
                ln = int(m.group(1))
                if ln > 0:
                    focus.add(ln)
    raw_lines = p2_diag.get("focus_lines")
    if isinstance(raw_lines, Sequence) and not isinstance(raw_lines, (str, bytes)):
        for item in raw_lines:
            if isinstance(item, int) and not isinstance(item, bool) and item > 0:
                focus.add(item)
    return tuple(sorted(focus)), "\n".join(diag_parts)


def _compact_trackb_snippet_source(
    snippet_source: str,
    regime_spec: RegimeTargetSpec,
    *,
    incumbent_source: str = "",
    max_bytes: int | None = None,
    cpp_max_bytes: int | None = None,
    active_kernel: str | None = None,
    focus_lines: Sequence[int] = (),
    diagnostic_text: str = "",
) -> str:
    """Render complete active closure plus single-line sibling prototypes without truncation."""
    _ = (max_bytes, cpp_max_bytes)
    if regime_spec.runtime_backend not in ("ascendc_trackb", "hybrid_trackb"):
        return snippet_source
    if not regime_spec.trackb_symbols:
        return snippet_source
    try:
        tree = ast.parse(snippet_source)
    except SyntaxError:
        return snippet_source
    lines = snippet_source.splitlines(keepends=True)
    hits = _find_top_level_str_assign_span(tree, lines, set(regime_spec.trackb_symbols))
    replacements: list[tuple[int, int, str]] = []
    for sym_name, cpp_text, start_pos, end_pos in hits:
        if "/* TRACK-B SCOPED SLICE:" in cpp_text:
            continue
        view = build_trackb_context_view(
            cpp_text,
            regime_tag=regime_spec.regime_tag,
            module_source=incumbent_source or snippet_source,
            trackb_symbol=sym_name,
            active_kernel=active_kernel,
            focus_lines=focus_lines,
            diagnostic_text=diagnostic_text,
        )
        wrapped = view if view.startswith("\n") else f"\n{view}\n"
        replacements.append((start_pos, end_pos, _format_raw_cpp_literal(wrapped)))
    if not replacements:
        return snippet_source
    updated = snippet_source
    for start_pos, end_pos, literal in reversed(replacements):
        updated = updated[:start_pos] + literal + updated[end_pos:]
    return updated


def _is_real_tir_primfunc(obj: object) -> bool:
    """Return True iff *obj* is a concrete TVM `PrimFunc` instance."""
    if obj is None or isinstance(obj, (tuple, list, dict, str, bytes, int, float, bool)):
        return False
    try:
        import tilelang  # noqa: F401, PLC0415
        import tvm  # noqa: PLC0415

        tirx_mod = getattr(tvm, "tirx", None)
        tir_mod = getattr(tvm, "tir", None)
        prim_cls = getattr(tirx_mod, "PrimFunc", None) or getattr(tir_mod, "PrimFunc", None)
        return isinstance(prim_cls, type) and isinstance(obj, prim_cls)
    except ImportError:
        return False


def _cow_helper_name(regime_tag: str, helper_name: str) -> str:
    """Return idempotent Copy-on-Write helper name scoped under regime_tag."""
    cow_prefix = f"_{regime_tag}_"
    if helper_name.startswith(cow_prefix):
        return helper_name
    return f"{cow_prefix}{helper_name.lstrip('_')}"


class _CowHelperRenamer(ast.NodeTransformer):
    """Rename CoW helper FunctionDefs and ast.Name references inside active symbols."""

    def __init__(self, rename_map: Mapping[str, str]) -> None:
        self._rename_map = dict(rename_map)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        """Rename matching FunctionDef and recurse into its body."""
        if node.name in self._rename_map:
            node.name = self._rename_map[node.name]
        return self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        """Rename matching AsyncFunctionDef and recurse into its body."""
        if node.name in self._rename_map:
            node.name = self._rename_map[node.name]
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> ast.AST:
        """Rename matching `ast.Name` identifiers."""
        if node.id in self._rename_map:
            node.id = self._rename_map[node.id]
        return node


def _ast_references_any(node: ast.AST, names: Collection[str]) -> bool:
    """Return True if *node* contains any ast.Name referencing a symbol in *names*."""
    if not names:
        return False
    target_set = set(names)
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name) and sub.id in target_set:
            return True
    return False


def _render_with_cow_renames(
    node: ast.stmt,
    raw_source: str,
    rename_map: Mapping[str, str],
    *,
    force_rename_self: str = "",
) -> tuple[str, ast.stmt]:
    """Apply _CowHelperRenamer when needed, preserving raw_source when unchanged."""
    needs_rename = bool(
        rename_map
        and (
            (force_rename_self and force_rename_self in rename_map)
            or _ast_references_any(node, rename_map)
        )
    )
    if not needs_rename:
        return raw_source, node
    transformed = _CowHelperRenamer(rename_map).visit(copy.deepcopy(node))
    ast.fix_missing_locations(transformed)
    return ast.unparse(transformed).rstrip() + "\n", transformed


def _collect_shared_non_active_symbols(
    orig_symbols: Sequence[_TopLevelSymbol],
    orig_ast_by_name: Mapping[str, ast.stmt],
    *,
    active_own_symbols: set[str],
    forbidden: set[str],
) -> set[str]:
    """Find top-level symbols reachable from sibling regimes or non-active roots."""
    called_by_any: set[str] = set()
    for caller_name, caller_node in orig_ast_by_name.items():
        for sub in ast.walk(caller_node):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                ref = sub.id
                if ref in orig_ast_by_name and ref != caller_name:
                    called_by_any.add(ref)

    non_active_roots = [
        sym.name
        for sym in orig_symbols
        if sym.name not in active_own_symbols
        and (
            sym.name in forbidden
            or not sym.name.startswith("_")
            or sym.kind != "func"
            or sym.name not in called_by_any
        )
    ]
    shared_reachable: set[str] = set()
    queue = list(non_active_roots)
    visited: set[str] = set()
    while queue:
        cur = queue.pop(0)
        if cur in visited:
            continue
        visited.add(cur)
        if cur in active_own_symbols:
            continue
        node = orig_ast_by_name.get(cur)
        if node is None:
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                ref = sub.id
                if ref in orig_ast_by_name and ref not in active_own_symbols:
                    shared_reachable.add(ref)
                    if ref not in visited:
                        queue.append(ref)
    return shared_reachable


def patch_regime_into_module(
    module_source: str,
    regime_or_func: RegimeTargetSpec | str,
    candidate_raw_source: str,
    *,
    operator_entrypoint: str = "",
    sibling_specs: Sequence[RegimeTargetSpec] = (),
    active_kernel: str | None = None,
) -> str:
    """Merge a scoped or Python regime slice into module_source without losing siblings."""
    candidate_dsl = extract_python_dsl(candidate_raw_source)
    if isinstance(regime_or_func, str):
        return apply_ast_patch(module_source, regime_or_func, candidate_dsl)

    regime_spec = regime_or_func
    python_snippet = candidate_dsl
    if (
        regime_spec.runtime_backend in ("ascendc_trackb", "hybrid_trackb")
        and regime_spec.trackb_symbols
    ):
        _reject_trackb_context_markers(candidate_dsl, regime_spec.regime_tag)
        search_replace_blocks = parse_scoped_search_replace_blocks(candidate_dsl)
        if (
            ("<<<<<<<" in candidate_dsl or ">>>>>>>" in candidate_dsl)
            and not search_replace_blocks
        ):
            diag = _format_patch_anchor_miss_diagnostic(
                module_source,
                regime_spec,
                "Malformed SEARCH/REPLACE block markers in Track-B candidate",
                active_kernel=active_kernel,
            )
            raise _patch_anchor_miss_error_type()(diag)
        if search_replace_blocks:
            module_source = _apply_scoped_trackb_patch_to_module(
                module_source,
                regime_spec,
                candidate_dsl,
                active_kernel=active_kernel,
            )
            python_snippet = _trackb_python_residual(candidate_dsl)
            if not python_snippet:
                return module_source
        elif not _parses_as_python(candidate_dsl) and _looks_like_bare_trackb_cpp(
            candidate_dsl
        ):
            return _apply_spliced_trackb_units_to_module(
                module_source,
                regime_spec,
                candidate_dsl,
                active_kernel=active_kernel,
            )
    orig_tree = ast.parse(module_source)
    orig_symbols = _collect_top_level_symbols(orig_tree)
    orig_by_name = {sym.name: sym for sym in orig_symbols}
    orig_ast_by_name = _collect_top_level_ast_nodes(orig_tree)

    if regime_spec.editable_symbols:
        required_symbols = [sym for sym in regime_spec.editable_symbols if sym in orig_by_name]
        if required_symbols:
            effective_editable = tuple(regime_spec.editable_symbols)
        else:
            orig_node = _find_top_level_func(orig_tree, regime_spec.builder_candidates)
            if orig_node is None:
                raise ValueError(
                    f"Original module missing regime symbols {regime_spec.editable_symbols}."
                )
            effective_editable = (orig_node.name,)
    else:
        orig_node = _find_top_level_func(orig_tree, regime_spec.builder_candidates)
        if orig_node is None:
            raise ValueError(
                f"Original module missing builder in {regime_spec.builder_candidates}."
            )
        effective_editable = (orig_node.name,)

    editable_set = set(effective_editable)
    prefix = regime_spec.allowed_helper_prefix
    cow_prefix = f"_{regime_spec.regime_tag}_"

    cand_tree = ast.parse(python_snippet)
    cand_symbols: list[_TopLevelSymbol] = []
    cand_ast_by_name: dict[str, ast.stmt] = {}
    seen_cand: set[str] = set()
    forbidden = _other_regime_forbidden_symbols(
        regime_spec,
        operator_entrypoint=operator_entrypoint,
        sibling_specs=sibling_specs,
    )
    forbidden.difference_update(editable_set)

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
        cand_symbols.append(_TopLevelSymbol(name, kind, start, end))
        cand_ast_by_name[name] = node

    if not cand_symbols:
        raise ValueError(
            "Candidate snippet must define at least one authorized regime symbol."
        )

    private_func_pool = {
        sym.name
        for sym in (*orig_symbols, *cand_symbols)
        if sym.kind == "func"
        and sym.name.startswith("_")
        and sym.name not in forbidden
        and sym.name not in editable_set
    }
    active_check_asts = {**orig_ast_by_name, **cand_ast_by_name}
    active_seed_roots = [
        name
        for name in (
            *effective_editable,
            *(
                sym.name
                for sym in cand_symbols
                if (prefix and sym.name.startswith(prefix))
                or sym.name.startswith(cow_prefix)
            ),
        )
        if name in active_check_asts
    ]
    referenced_private_funcs = _collect_referenced_helper_names(
        active_check_asts,
        helper_names=private_func_pool,
        seed_roots=active_seed_roots,
    )

    ws_planner_names = {
        "_plan_workspace_bytes",
        f"_{regime_spec.regime_tag}_plan_workspace_bytes",
    }
    for sym in cand_symbols:
        name = sym.name
        if name in forbidden:
            raise ValueError(f"Candidate cannot define or modify protected symbol {name!r}.")
        if name in editable_set or name in ws_planner_names:
            continue
        if (prefix and name.startswith(prefix)) or name.startswith(cow_prefix):
            continue
        is_referenced_private_func = (
            sym.kind == "func"
            and name.startswith("_")
            and name in referenced_private_funcs
        )
        if name in orig_by_name:
            if not is_referenced_private_func:
                raise ValueError(
                    f"Candidate cannot define or modify protected symbol {name!r}."
                )
        elif not is_referenced_private_func:
            raise ValueError(
                f"Unauthorized top-level symbol {name!r}; expected {editable_set} "
                f"or prefix {prefix!r}."
            )

    cand_lines = python_snippet.splitlines(keepends=True)
    cand_slices = {
        sym.name: "".join(cand_lines[sym.start - 1 : sym.end]).rstrip() + "\n"
        for sym in cand_symbols
    }

    active_own_symbols = (
        set(editable_set)
        | ws_planner_names
        | {
            sym.name
            for sym in orig_symbols
            if (prefix and sym.name.startswith(prefix)) or sym.name.startswith(cow_prefix)
        }
    )
    shared_reachable = _collect_shared_non_active_symbols(
        orig_symbols,
        orig_ast_by_name,
        active_own_symbols=active_own_symbols,
        forbidden=forbidden,
    )

    edited_cand_helpers = {
        sym.name
        for sym in cand_symbols
        if sym.kind == "func"
        and sym.name not in editable_set
        and (
            sym.name not in orig_ast_by_name
            or ast.dump(cand_ast_by_name[sym.name]) != ast.dump(orig_ast_by_name[sym.name])
        )
    }
    cow_targets = edited_cand_helpers & shared_reachable
    if cow_targets:
        changed_closure = True
        while changed_closure:
            changed_closure = False
            for hname in list(shared_reachable):
                if hname in cow_targets or hname in editable_set:
                    continue
                hnode = cand_ast_by_name.get(hname) or orig_ast_by_name.get(hname)
                if hnode is not None and _ast_references_any(hnode, cow_targets):
                    cow_targets.add(hname)
                    changed_closure = True
    rename_map = {
        hname: _cow_helper_name(regime_spec.regime_tag, hname)
        for hname in sorted(cow_targets)
    }
    rename_map = {k: v for k, v in rename_map.items() if k != v}

    rendered_by_target: dict[str, str] = {}
    rendered_ast_by_target: dict[str, ast.stmt] = {}
    for sym in cand_symbols:
        if (
            sym.name in shared_reachable
            and sym.name not in rename_map
            and sym.name not in editable_set
        ):
            continue
        target_name = rename_map.get(sym.name, sym.name)
        rendered_src, rendered_node = _render_with_cow_renames(
            cand_ast_by_name[sym.name],
            cand_slices[sym.name],
            rename_map,
            force_rename_self=sym.name,
        )
        rendered_by_target[target_name] = rendered_src
        rendered_ast_by_target[target_name] = rendered_node

    orig_lines = module_source.splitlines(keepends=True)
    if rename_map:
        for ename in effective_editable:
            if ename in orig_ast_by_name and ename not in cand_ast_by_name:
                orig_sym = orig_by_name[ename]
                orig_raw = (
                    "".join(orig_lines[orig_sym.start - 1 : orig_sym.end]).rstrip() + "\n"
                )
                rendered_src, rendered_node = _render_with_cow_renames(
                    orig_ast_by_name[ename],
                    orig_raw,
                    rename_map,
                )
                if rendered_node is not orig_ast_by_name[ename]:
                    rendered_by_target[ename] = rendered_src
                    rendered_ast_by_target[ename] = rendered_node
        for hname in referenced_private_funcs:
            if (
                hname in orig_ast_by_name
                and hname not in cand_ast_by_name
                and hname not in shared_reachable
                and hname not in editable_set
            ):
                orig_sym = orig_by_name[hname]
                orig_raw = (
                    "".join(orig_lines[orig_sym.start - 1 : orig_sym.end]).rstrip() + "\n"
                )
                rendered_src, rendered_node = _render_with_cow_renames(
                    orig_ast_by_name[hname],
                    orig_raw,
                    rename_map,
                )
                if rendered_node is not orig_ast_by_name[hname]:
                    rendered_by_target[hname] = rendered_src
                    rendered_ast_by_target[hname] = rendered_node

    merged_ast_by_name = dict(orig_ast_by_name)
    merged_ast_by_name.update(rendered_ast_by_target)
    prunable_orig_helpers = {
        sym.name
        for sym in orig_symbols
        if sym.name not in editable_set
        and sym.name not in shared_reachable
        and ((prefix and sym.name.startswith(prefix)) or sym.name.startswith(cow_prefix))
    }
    prunable_cand_cow = {
        name
        for name in rendered_ast_by_target
        if name not in editable_set and name.startswith(cow_prefix)
    }
    all_helper_names = prunable_orig_helpers | {
        name for name in rendered_ast_by_target if name not in editable_set
    }
    seed_roots = [
        name
        for name in merged_ast_by_name
        if name not in prunable_orig_helpers and name not in prunable_cand_cow
    ]
    for ws_name in ws_planner_names:
        if ws_name in merged_ast_by_name and ws_name not in seed_roots:
            seed_roots.append(ws_name)
    live_helpers = _collect_referenced_helper_names(
        merged_ast_by_name,
        helper_names=all_helper_names,
        seed_roots=seed_roots,
    )
    live_helpers.update(ws_planner_names & set(merged_ast_by_name))

    new_helper_names = [
        name
        for name in rendered_by_target
        if name not in editable_set
        and name not in orig_by_name
        and (name not in prunable_cand_cow or name in live_helpers)
    ]
    insert_anchor = next(
        (
            sym.name
            for sym in orig_symbols
            if sym.name in editable_set
            or (prefix and sym.name.startswith(prefix))
            or sym.name.startswith(cow_prefix)
        ),
        orig_symbols[0].name,
    )

    for item in reversed(orig_symbols):
        if item.name in rendered_by_target:
            if item.name in prunable_cand_cow and item.name not in live_helpers:
                del_end = item.end
                while del_end < len(orig_lines) and orig_lines[del_end].strip() == "":
                    del_end += 1
                del orig_lines[item.start - 1 : del_end]
            else:
                orig_lines[item.start - 1 : item.end] = [rendered_by_target[item.name]]
        elif item.name in prunable_orig_helpers and item.name not in live_helpers:
            del_end = item.end
            while del_end < len(orig_lines) and orig_lines[del_end].strip() == "":
                del_end += 1
            del orig_lines[item.start - 1 : del_end]
        if item.name == insert_anchor and new_helper_names:
            helper_block = "".join(
                rendered_by_target[hname] + "\n" for hname in new_helper_names
            )
            orig_lines[item.start - 1 : item.start - 1] = [helper_block]

    patched = "".join(orig_lines)
    ast.parse(patched)
    if set(cand_slices) & set(regime_spec.trackb_symbols):
        orig_lines_raw = module_source.splitlines(keepends=True)
        for sym_name in sorted(set(cand_slices) & set(regime_spec.trackb_symbols)):
            orig_hits = _find_top_level_str_assign_span(orig_tree, orig_lines_raw, {sym_name})
            patched_tree = ast.parse(patched)
            patched_lines = patched.splitlines(keepends=True)
            new_hits = _find_top_level_str_assign_span(
                patched_tree, patched_lines, {sym_name}
            )
            if not orig_hits or not new_hits:
                raise ValueError(
                    f"Track-B symbols {sorted(set(cand_slices) & set(regime_spec.trackb_symbols))} "
                    "must be assigned a raw C++ string literal."
                )
            orig_cpp = orig_hits[0][1]
            cand_cpp = new_hits[0][1]
            orig_units = _trackb_units_by_name(orig_cpp)
            cand_units = _trackb_units_by_name(cand_cpp)
            allowed_added_units: set[str] = set()
            if cand_units and (set(orig_units) - set(cand_units)):
                kernel_name = active_kernel or select_active_trackb_kernel(
                    orig_cpp,
                    regime_tag=regime_spec.regime_tag,
                    module_source=module_source,
                )
                merged_cpp, _applied = splice_cpp_units_into_tu(
                    orig_cpp,
                    cand_cpp,
                    active_kernel=kernel_name,
                )
                allowed_added_units = set(_trackb_units_by_name(merged_cpp)) - set(orig_units)
                patched = _replace_trackb_constant(patched, sym_name, merged_cpp)
                cand_cpp = merged_cpp
            _assert_trackb_replacement_scoped(
                orig_cpp,
                cand_cpp,
                regime_spec,
                module_source=module_source,
                active_kernel=active_kernel,
                allowed_added_units=allowed_added_units,
            )
    if "_render_trackb_tu" in cand_slices:
        seen_tags: set[str] = set()
        for spec in (regime_spec, *sibling_specs):
            if spec.regime_tag in seen_tags or not spec.trackb_symbols:
                continue
            seen_tags.add(spec.regime_tag)
            spec_tb = _extract_trackb_cpp_source(patched, spec)
            if spec_tb is not None:
                _render_trackb_tu_from_module(
                    patched,
                    spec_tb[1],
                    spec,
                    require_renderer=True,
                )
    return patched


__all__ = [
    "DEFAULT_READONLY_HOST_EXPORTS",
    "ResolvedRegimeConfig",
    "_CowHelperRenamer",
    "_HIR_DIM_ALIASES",
    "_TopLevelSymbol",
    "_active_trackb_kernel",
    "_apply_scoped_trackb_patch_to_module",
    "_apply_spliced_trackb_units_to_module",
    "_assert_trackb_replacement_scoped",
    "_ast_char_offset",
    "_ast_references_any",
    "_atomic_write_text",
    "_case_ratio_predicate_entries",
    "_collect_referenced_helper_names",
    "_collect_shared_non_active_symbols",
    "_collect_top_level_ast_nodes",
    "_collect_top_level_symbols",
    "_compact_trackb_snippet_source",
    "_cow_helper_name",
    "_derive_dynamic_protected_symbols",
    "_detect_trackb_dead_code_edit",
    "_extract_func_signature_header",
    "_extract_p2_focus_lines_and_diag",
    "_extract_regime_abi_context",
    "_extract_trackb_cpp_source",
    "_filter_callable_kwargs",
    "_find_top_level_func",
    "_find_top_level_str_assign_span",
    "_format_raw_cpp_literal",
    "_func_line_span",
    "_has_prim_func_builder",
    "_is_prim_func_decorator",
    "_is_real_tir_primfunc",
    "_launched_trackb_kernels",
    "_load_module_and_run_selector",
    "_norm_flag_seq",
    "_normalized_trackb_unit",
    "_ordered_top_level_trackb_kernels",
    "_other_regime_forbidden_symbols",
    "_parses_as_python",
    "_reject_trackb_context_markers",
    "_render_trackb_tu_from_module",
    "_render_with_cow_renames",
    "_replace_trackb_constant",
    "_resolve_cann_root_from_env_file",
    "_resolve_operator_entrypoint_safe",
    "_resolve_selector_compile_flags",
    "_strip_trackb_patch_hunks",
    "_top_level_symbol_sources",
    "_trackb_kernel_names",
    "_trackb_python_residual",
    "_trackb_units_by_name",
    "apply_ast_patch",
    "extract_regime_snippet",
    "patch_regime_into_module",
    "resolve_regime_config",
    "split_readonly_host_tail",
]
