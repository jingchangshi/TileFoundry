"""Scoped C++20 slicing, patching, and Track-A-to-Track-B escalation protocol."""

from __future__ import annotations

import ast
import contextlib
import functools
import inspect
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from tilefoundry_ascend.ascend.compile_cache.cpp_closure import (
    _CppUnit,
    _extract_unit_name,
    _helper_call_closure,
    _host_launcher_units,
    _looks_like_cpp_def_start,
    _referenced_helpers_in_text,
    _split_cpp_units,
    extract_reachable_trackb_source,
    select_active_trackb_kernel,
    strip_cpp_comments_and_strings,
    strip_markdown_fence_lines,
)
from tilefoundry_ascend.ascend.target.dialect_surface import CCE_CPP20_SURFACE

_SEARCH_REPLACE_WINDOW_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"<<<<<<<\s*SEARCH(?:\s*\[L?(?P<start>\d+)\s*-\s*L?(?P<end>\d+)\])?\s*\n"
        r"(?P<search>.*?)\n=======\s*\n(?P<replace>.*?)\n>>>>>>>\s*REPLACE",
        re.DOTALL,
    ),
    re.compile(
        r"<<<\s*SEARCH(?:\s*\[L?(?P<start>\d+)\s*-\s*L?(?P<end>\d+)\])?\s*\n"
        r"(?P<search>.*?)\n===\s*\n(?P<replace>.*?)\n>>>\s*REPLACE",
        re.DOTALL,
    ),
)
_SEARCH_REPLACE_PATTERNS = _SEARCH_REPLACE_WINDOW_PATTERNS

_LAUNCH_CONFIG_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*(?:<[^;{}]*>)?\s*<<<")
_SIBLING_MARKER_RE = re.compile(
    r"(?:/\*\s*\[READ-ONLY SIBLING KERNEL:|//\s*\[READONLY_SIBLING_DECLARATION\])"
)
_DEVICE_QUALIFIER_RE = re.compile(
    r"__(?:global|aicore|simd_vf|simt_vf|simd_callee|simt_callee)__\b"
)
_DIAG_FILE_LINE_RE = re.compile(
    r"(?:\.(?:cce|cpp|cc|cu|h)|<source>|input):(\d+)(?::\d+)?:\s*(?:fatal\s+)?(?:error|warning):"
)
_DIAG_WORD_LINE_RE = re.compile(r"\bline\s+(\d+)\b")
_TRACKB_CPP_QUALIFIERS: tuple[str, ...] = (
    'extern "C" __global__', "__global__", "__aicore__", "__simd_vf__", "__simt_vf__",
)

DEFAULT_PROTECTED_EXPORTS: frozenset[str] = frozenset(
    {"kernel", "ref_kernel", "build_kernel", "_ENTRY_CACHE", "__all__"}
)
_NON_REPAIRABLE_FAILURE_CODES: frozenset[str] = frozenset(
    {
        "NOOP_IDENTICAL", "UB_SRAM_OVERFLOW", "UB_OVERFLOW",
        "L1_SRAM_OVERFLOW", "L0C_SRAM_OVERFLOW", "VF_STACK_OVERFLOW",
    }
)
TRACKB_REPLY_CONTRACT_PROMPT = (
    "### TRACK-B REPLY CONTRACT\n"
    "- Prefer Mode A (complete bare C++20 definition of the active target kernel or exclusive "
    "helper, keeping exact symbol name and signature) or Mode B (line-window anchored "
    "`<<<<<<< SEARCH [Lstart-Lend]` / `=======` / `>>>>>>> REPLACE` hunks using `[L0001]` line "
    "numbers from the active closure view).\n"
    "- Never edit or echo `// [READONLY_SIBLING_DECLARATION]` or `/* [READ-ONLY HELPER] */` lines."
)


class PatchAnchorMissError(ValueError):
    """Raised when a Track-B C++ patch fails to match or produces zero effective change."""

    fault_plane: str = "SYNTAX_AST"
    fault_code: str = "PATCH_ANCHOR_MISS"


class AmbiguousPatchAnchorError(PatchAnchorMissError):
    """Raised when a SEARCH hunk matches multiple spans without unique window resolution."""

    fault_plane: str = "SYNTAX_AST"
    fault_code: str = "PATCH_AMBIGUOUS_ANCHOR"


class NoopIdenticalPatchError(PatchAnchorMissError):
    """Raised when an applied Track-B patch produces normalized-identical C++ source."""

    fault_plane: str = "NOOP_IDENTICAL"
    fault_code: str = "NOOP_IDENTICAL"


@dataclass(frozen=True)
class RepairClosureSlice:
    """Self-contained repair context spanning active device call-graph and host ABI."""

    editable_device_source: str
    sibling_prototypes: tuple[str, ...]
    readonly_host_abi_tail: str
    diagnostic_excerpt: str


@dataclass(frozen=True)
class TopLevelSymbol:
    """Top-level Python function or single-target assignment line span."""

    name: str
    kind: str
    start: int
    end: int


@dataclass(frozen=True)
class _SearchReplaceHunk:
    search_block: str
    replace_block: str
    line_window: tuple[int, int] | None = None


@functools.lru_cache(maxsize=512)
def cached_callable_signature_info(
    fn: object,
) -> tuple[inspect.Signature, bool, tuple[str, ...], tuple[tuple[str, object], ...]] | None:
    """Inspect and cache signature metadata for an external user module callable."""
    if not callable(fn):
        return None
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None
    params = tuple(sig.parameters.values())
    has_vk = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params)
    pos = tuple(
        p.name
        for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    )
    defs = tuple(
        (p.name, p.default)
        for p in params
        if p.default is not inspect.Parameter.empty
        and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
    )
    return sig, has_vk, pos, defs


def filter_callable_kwargs(
    fn: object,
    kwargs: Mapping[str, object],
    *,
    aliases: Mapping[str, tuple[str, ...]] | None = None,
) -> dict[str, object]:
    """Filter *kwargs* down to parameters accepted by an external *fn*."""
    info = cached_callable_signature_info(fn)
    if info is None:
        return {}
    sig, has_vk, _pos, _defs = info
    if has_vk:
        return dict(kwargs)
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    if not aliases:
        return filtered
    low_kw = {str(k).lower(): v for k, v in kwargs.items()}
    for pname in sig.parameters:
        if pname in filtered:
            continue
        pl = pname.lower()
        if pl in low_kw:
            filtered[pname] = low_kw[pl]
            continue
        for alias in aliases.get(pl, ()):
            if alias in kwargs or alias.lower() in low_kw:
                filtered[pname] = kwargs[alias] if alias in kwargs else low_kw[alias.lower()]
                break
    return filtered


def format_trackb_reply_contract(active_kernel: str | None = None) -> str:
    """Render Track-B patch instructions preferring Mode A symbol defs or Mode B [Lstart-Lend]."""
    if active_kernel:
        return f"{TRACKB_REPLY_CONTRACT_PROMPT}\n- Active editable kernel scope: `{active_kernel}` and its exclusive helpers."
    return TRACKB_REPLY_CONTRACT_PROMPT


def extract_cpp_diagnostic_lines(diagnostic_text: str) -> tuple[int, ...]:
    """Extract sorted unique positive 1-indexed line numbers from C++ diagnostics."""
    if not diagnostic_text:
        return ()
    lines = {
        int(m.group(1))
        for pat in (_DIAG_FILE_LINE_RE, _DIAG_WORD_LINE_RE)
        for m in pat.finditer(diagnostic_text)
        if int(m.group(1)) > 0
    }
    return tuple(sorted(lines))


def classify_stage_failure_for_repair(failure_code: str) -> bool:
    """Return True only when syntactic or API repair can resolve the failure."""
    return str(failure_code or "").strip().upper() not in _NON_REPAIRABLE_FAILURE_CODES


def stmt_line_span(node: ast.stmt) -> tuple[int, int]:
    """Return 1-based inclusive `(start, end)` line numbers including decorators."""
    decs = getattr(node, "decorator_list", ())
    return min((d.lineno for d in decs), default=node.lineno), (node.end_lineno or node.lineno)


def _extract_top_level_bound_names(node: ast.stmt) -> set[str]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return {node.name}
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return {node.target.id}
    if isinstance(node, ast.Assign):
        return {t.id for t in node.targets if isinstance(t, ast.Name)}
    return set()


def collect_top_level_symbols(tree: ast.Module) -> list[TopLevelSymbol]:
    """Collect top-level function and single-target assignment spans from *tree*."""
    items: list[TopLevelSymbol] = []
    for node in tree.body:
        start, end = stmt_line_span(node)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            items.append(TopLevelSymbol(node.name, "func", start, end))
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            items.append(TopLevelSymbol(node.targets[0].id, "assign", start, end))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            items.append(TopLevelSymbol(node.target.id, "assign", start, end))
    return items


def collect_top_level_ast_nodes(tree: ast.Module) -> dict[str, ast.stmt]:
    """Collect top-level function and single-target assignment AST nodes by name."""
    nodes: dict[str, ast.stmt] = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            nodes[node.name] = node
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            nodes[node.targets[0].id] = node
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            nodes[node.target.id] = node
    return nodes


def find_top_level_func(tree: ast.Module, names: Sequence[str]) -> ast.FunctionDef | None:
    """Find the first top-level `ast.FunctionDef` matching one of *names*."""
    by_name = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    return next((by_name[name] for name in names if name in by_name), None)


def collect_referenced_helper_names(
    ast_by_name: Mapping[str, ast.stmt],
    *,
    helper_names: set[str],
    seed_roots: Sequence[str],
) -> set[str]:
    """Compute transitive closure of helper symbols referenced from *seed_roots*."""
    live: set[str] = set()
    queue, visited = list(seed_roots), set()
    while queue:
        sym = queue.pop(0)
        if sym in visited:
            continue
        visited.add(sym)
        if (node := ast_by_name.get(sym)) is None:
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
                ref = sub.id
                if ref in helper_names and ref not in live:
                    live.add(ref)
                    if ref not in visited:
                        queue.append(ref)
    return live


def py_char_offset(lines: Sequence[str], lineno: int, col_offset: int) -> int:
    """Convert 1-based AST line and UTF-8 byte column offset to character index."""
    prefix = sum(len(lines[idx]) for idx in range(lineno - 1))
    return prefix + len(lines[lineno - 1].encode("utf-8")[:col_offset].decode("utf-8"))


def format_cpp_string_literal(cpp_text: str) -> str:
    """Format *cpp_text* as a raw triple-quoted Python string literal when safe."""
    return f'r"""{cpp_text}"""' if ('"""' not in cpp_text and not cpp_text.endswith('"')) else repr(cpp_text)


def find_top_level_str_assign_spans(
    tree: ast.Module,
    lines: Sequence[str],
    *,
    target_names: set[str] | None = None,
    require_cpp_tokens: bool = False,
) -> list[tuple[str, str, int, int]]:
    """Find character spans of top-level string constant assignments."""
    hits: list[tuple[str, str, int, int]] = []
    for node in tree.body:
        name: str | None = None
        val: ast.AST | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name, val = node.targets[0].id, node.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            name, val = node.target.id, node.value
        if (
            name is None
            or (target_names is not None and name not in target_names)
            or not isinstance(val, ast.Constant)
            or not isinstance(val.value, str)
            or val.end_lineno is None
            or val.end_col_offset is None
        ):
            continue
        if require_cpp_tokens and not any(t in val.value for t in ("__global__", "__aicore__", "__simd_vf__", "__simt_vf__")):
            continue
        hits.append((name, val.value, py_char_offset(lines, val.lineno, val.col_offset), py_char_offset(lines, val.end_lineno, val.end_col_offset)))
    return hits


def _find_python_cpp_literal_spans(
    source: str, *, require_cpp_tokens: bool = True
) -> list[tuple[str, str, int, int]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    return find_top_level_str_assign_spans(
        tree, source.splitlines(keepends=True), require_cpp_tokens=require_cpp_tokens
    )


def split_tracka_editable_and_readonly_tail(
    module_source: str, protected_exports: frozenset[str] = DEFAULT_PROTECTED_EXPORTS
) -> tuple[str, str]:
    """Separate editable kernel builders from protected module-tail bindings."""
    if not module_source.strip():
        return "", ""
    lines = module_source.splitlines(keepends=True)
    split_lineno = len(lines) + 1
    try:
        tree = ast.parse(module_source)
        for node in tree.body:
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.FunctionDef, ast.AsyncFunctionDef)):
                if _extract_top_level_bound_names(node) & protected_exports:
                    split_lineno = min(split_lineno, stmt_line_span(node)[0])
                    break
    except SyntaxError:
        if escaped := "|".join(re.escape(name) for name in sorted(protected_exports)):
            tail_re = re.compile(rf"^(?:(?:async\s+)?def\s+(?:{escaped})\b|(?:{escaped})\s*[:=])")
            for idx, raw_line in enumerate(lines, start=1):
                if tail_re.match(raw_line):
                    split_lineno = idx
                    break
    if split_lineno > len(lines):
        return module_source.rstrip() + "\n", ""
    editable = "".join(lines[: split_lineno - 1]).rstrip()
    return (f"{editable}\n" if editable else ""), "".join(lines[split_lineno - 1 :]).strip()


def extract_trackb_launcher_preview(module_source: str, *, max_bytes: int | None = None) -> str:
    """Extract C++ extern-C launcher stubs and host binding lines without byte truncation."""
    _ = max_bytes
    if not module_source.strip():
        return ""
    previews: list[str] = []
    if any(tok in module_source for tok in _TRACKB_CPP_QUALIFIERS) or 'extern "C"' in module_source:
        _, units = _split_cpp_units(module_source)
        host_units, _ = _host_launcher_units(units)
        for unit in host_units:
            if (snippet := unit.text.strip()) and snippet not in previews:
                previews.append(snippet)
    if not previews:
        for match in re.finditer(r'extern\s+"C"\s+', module_source):
            start = match.start()
            if (brace_open := module_source.find("{", start)) < 0:
                continue
            depth = 0
            end = -1
            for idx in range(brace_open, len(module_source)):
                if module_source[idx] == "{":
                    depth += 1
                elif module_source[idx] == "}":
                    depth -= 1
                    if depth == 0:
                        end = idx + 1
                        break
            if end > start and (snippet := module_source[start:end].strip()) and snippet not in previews:
                previews.append(snippet)
    if not previews:
        try:
            tree = ast.parse(module_source)
            lines = module_source.splitlines()
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and "trackb" in node.name:
                    assert node.end_lineno is not None
                    previews.append("\n".join(lines[node.lineno - 1 : min(node.lineno + 6, node.end_lineno)]).strip())
                    break
        except SyntaxError:
            pass
    return "\n\n".join(previews).strip()


def scan_launched_trackb_kernels(
    instantiated_tu: str, raw_cpp_source: str, kernels: set[str], *, fallback_full: bool = False
) -> tuple[set[str], set[str], tuple[str, ...]]:
    """Scan launcher `<<<...>>>` sites tracking brace depth to find shallowest launched kernels."""
    masked = strip_cpp_comments_and_strings(instantiated_tu)
    has_tail = instantiated_tu.startswith(raw_cpp_source) and len(instantiated_tu) > len(raw_cpp_source)
    regions = ([masked[len(raw_cpp_source) :], masked] if (has_tail and fallback_full) else [masked[len(raw_cpp_source) :] if instantiated_tu.startswith(raw_cpp_source) else masked])
    for launch_region in regions:
        launches: list[tuple[int, str]] = []
        depth = scanned = 0
        for match in _LAUNCH_CONFIG_RE.finditer(launch_region):
            depth += launch_region.count("{", scanned, match.start()) - launch_region.count("}", scanned, match.start())
            scanned = match.start()
            if (name := match.group(1)) in kernels:
                launches.append((depth, name))
        if launches:
            shallowest = min(d for d, _ in launches)
            ordered = tuple(dict.fromkeys(name for d, name in launches if d == shallowest))
            return {name for _, name in launches}, set(ordered), ordered
    return set(), set(), ()


def resolve_instantiated_trackb_kernel(instantiated_tu: str, raw_cpp_source: str) -> str | None:
    """Resolve the unique active `__global__` kernel from instantiated launcher truth."""
    _, units = _split_cpp_units(raw_cpp_source)
    kernels = {u.name for u in units if u.kind == "kernel" and u.name}
    if not kernels or not instantiated_tu.strip():
        return None
    launched, _, _ = scan_launched_trackb_kernels(instantiated_tu, raw_cpp_source, kernels)
    return next(iter(launched)) if len(launched) == 1 else None


def _extract_directive_lines(cpp_source: str, units: Sequence[_CppUnit]) -> list[tuple[int, str]]:
    unit_ranges = [(u.start_line, u.end_line) for u in units]
    return [
        (lineno, raw_line.rstrip("\r\n"))
        for lineno, raw_line in enumerate(cpp_source.splitlines(), start=1)
        if raw_line.strip().startswith("#") and not any(s <= lineno <= e for s, e in unit_ranges)
    ]


def _format_closure_block(
    lines_with_nums: Sequence[tuple[int, str]], *, include_line_numbers: bool, out_lines: list[str], line_map: dict[int, int]
) -> None:
    for tu_lineno, raw_line in lines_with_nums:
        out_lines.append(f"[L{tu_lineno:04d}] {raw_line}" if include_line_numbers else raw_line)
        line_map[len(out_lines)] = tu_lineno


def _build_cpp_closure_entries(
    clean_cpp: str, target_unit: _CppUnit, units: Sequence[_CppUnit], *, excluded_starts: set[int] = frozenset()
) -> tuple[list[tuple[int, str, object]], list[str]]:
    helpers_by_name = {u.name: u for u in units if u.kind == "helper" and u.name}
    reached_names = _helper_call_closure(target_unit, helpers_by_name)
    entries: list[tuple[int, str, object]] = [(d_ln, "directive", (d_ln, d_txt)) for d_ln, d_txt in _extract_directive_lines(clean_cpp, units)]
    sibling_protos: list[str] = []
    for unit in units:
        if unit.start_char in excluded_starts:
            continue
        if unit.kind == "global_decl":
            entries.append((unit.start_line, "unit", unit))
        elif unit.kind == "helper":
            if unit.start_char == target_unit.start_char or (unit.name and unit.name in reached_names):
                entries.append((unit.start_line, "unit", unit))
            elif sig := unit.signature.rstrip(";").strip():
                entries.append((unit.start_line, "ro_helper", f"{sig};"))
                sibling_protos.append(f"{sig};")
        elif unit.kind == "kernel":
            if unit.start_char == target_unit.start_char:
                entries.append((unit.start_line, "unit", unit))
            else:
                sig = unit.signature.rstrip(";").strip()
                decl = f"{sig};" if sig else f"void {unit.name}();"
                entries.append((unit.start_line, "ro_sibling", decl))
                sibling_protos.append(decl)
    entries.sort(key=lambda item: item[0])
    return entries, sibling_protos


def render_active_symbol_closure_with_line_map(
    cpp_source: str, active_kernel: str, *, include_line_numbers: bool = True
) -> tuple[str, dict[int, int]]:
    """Render 100% lossless active closure with line numbers and single-line sibling decls."""
    clean_cpp = strip_markdown_fence_lines(cpp_source)
    _, units = _split_cpp_units(clean_cpp)
    target_unit = next((u for u in units if u.kind == "kernel" and u.name == active_kernel), None) or next((u for u in units if u.name == active_kernel), None)
    if target_unit is None:
        raise ValueError(f"Active kernel {active_kernel!r} not found in C++ source")
    entries, _ = _build_cpp_closure_entries(clean_cpp, target_unit, units)
    out_lines: list[str] = []
    line_map: dict[int, int] = {}
    prev_kind = ""
    for _, kind, payload in entries:
        if out_lines and (kind == "unit" or prev_kind != kind):
            out_lines.append("")
        if kind == "directive":
            assert isinstance(payload, tuple)
            _format_closure_block([payload], include_line_numbers=include_line_numbers, out_lines=out_lines, line_map=line_map)
        elif kind == "unit":
            assert isinstance(payload, _CppUnit)
            _format_closure_block([(payload.start_line + i, ln) for i, ln in enumerate(payload.text.rstrip("\r\n").splitlines())], include_line_numbers=include_line_numbers, out_lines=out_lines, line_map=line_map)
        elif kind == "ro_helper":
            out_lines.append(f"/* [READ-ONLY HELPER] */ {payload}")
        elif kind == "ro_sibling":
            out_lines.append(f"// [READONLY_SIBLING_DECLARATION] {payload}")
        prev_kind = kind
    return "\n".join(out_lines).strip(), line_map


def render_active_symbol_closure_view(cpp_source: str, active_kernel: str, *, include_line_numbers: bool = True) -> str:
    """Render 100% lossless active closure with optional [L0001] line-number anchors."""
    return render_active_symbol_closure_with_line_map(cpp_source, active_kernel, include_line_numbers=include_line_numbers)[0]


def build_trackb_context_view(
    cpp_source: str, *, active_kernel: str | None = None, regime_tag: str = "",
    module_source: str = "", trackb_symbol: str = "", launcher_preview: str = "",
    max_bytes: int = 49152, focus_lines: Sequence[int] = (), focus_window: int = 25,
    diagnostic_text: str = "", include_line_numbers: bool = True,
) -> str:
    """Build a lossless active-closure C++20 context view with single-line sibling decls."""
    _ = (max_bytes, focus_lines, focus_window, diagnostic_text)
    clean_cpp = strip_markdown_fence_lines(cpp_source)
    _, units = _split_cpp_units(clean_cpp)
    kernels = [u for u in units if u.kind == "kernel"]
    target_name = select_active_trackb_kernel(clean_cpp, active_kernel=active_kernel, regime_tag=regime_tag, module_source=module_source)
    if target_name is None and kernels:
        target_name = kernels[0].name
    preview = launcher_preview.strip() or extract_trackb_launcher_preview(module_source)
    header_parts: list[str] = []
    if trackb_symbol:
        header_parts.append(f"/* TRACK-B SCOPED SLICE: symbol={trackb_symbol} active={target_name} */")
    if preview:
        header_parts.append(f"/* LAUNCHER BINDING PREVIEW */\n{preview}")
    if target_name is not None:
        closure_body = render_active_symbol_closure_view(clean_cpp, target_name, include_line_numbers=include_line_numbers)
    else:
        out_lines: list[str] = []
        line_map: dict[int, int] = {}
        if directives := _extract_directive_lines(clean_cpp, units):
            _format_closure_block(directives, include_line_numbers=include_line_numbers, out_lines=out_lines, line_map=line_map)
        for unit in units:
            if out_lines:
                out_lines.append("")
            _format_closure_block(
                [(unit.start_line + i, ln) for i, ln in enumerate(unit.text.rstrip("\r\n").splitlines())],
                include_line_numbers=include_line_numbers, out_lines=out_lines, line_map=line_map,
            )
        closure_body = "\n".join(out_lines).strip()
    return "\n\n".join(p for p in (*header_parts, closure_body) if p).strip()


extract_scoped_trackb_kernel_view = build_trackb_context_view


def _parse_scoped_search_replace_hunks(patch_text: str) -> list[_SearchReplaceHunk]:
    cleaned_patch = strip_markdown_fence_lines(patch_text)
    for pat in _SEARCH_REPLACE_WINDOW_PATTERNS:
        hunks: list[_SearchReplaceHunk] = []
        for m in pat.finditer(cleaned_patch):
            s_raw, e_raw = m.group("start"), m.group("end")
            win = (min(int(s_raw), int(e_raw)), max(int(s_raw), int(e_raw))) if (s_raw is not None and e_raw is not None) else None
            hunks.append(_SearchReplaceHunk(strip_markdown_fence_lines(m.group("search")), strip_markdown_fence_lines(m.group("replace")), win))
        if hunks:
            return hunks
    return []


def parse_scoped_search_replace_blocks(patch_text: str) -> list[tuple[str, str]]:
    """Parse SEARCH/REPLACE patch hunks from scoped Track-B candidate output."""
    return [(h.search_block, h.replace_block) for h in _parse_scoped_search_replace_hunks(patch_text)]


def _reject_introduced_forbidden_tokens(search_block: str, replace_block: str) -> None:
    if introduced := CCE_CPP20_SURFACE.introduced_forbidden_tokens(search_block, replace_block):
        raise ValueError(f"Track-B CCE patch introduces forbidden symbols: {', '.join(introduced)}")


def _leading_ws_len(line: str) -> int:
    return len(line) - len(line.lstrip(" \t"))


def _common_indent(lines: Sequence[str]) -> int:
    indents = [_leading_ws_len(line) for line in lines if line.strip()]
    return min(indents) if indents else 0


def normalized_line_matches(
    text: str, search_block: str, *, strip_common_indent: bool = True, keep_trailing_newline: bool = True
) -> list[tuple[int, int, int]]:
    """Locate *search_block* windows after stripping trailing whitespace and optional indent."""
    text_lines, search_lines = text.splitlines(keepends=True), search_block.splitlines()
    if not search_lines or not any(line.strip() for line in search_lines):
        return []
    s_comm = _common_indent(search_lines) if strip_common_indent else 0
    norm_search = [(ln.rstrip()[s_comm:] if (strip_common_indent and ln.strip()) else ln.rstrip()) for ln in search_lines]
    offsets, pos = [], 0
    for ln in text_lines:
        offsets.append(pos)
        pos += len(ln)
    win, matches = len(search_lines), []
    for idx in range(len(text_lines) - win + 1):
        win_lines = text_lines[idx : idx + win]
        w_comm = _common_indent(win_lines) if strip_common_indent else 0
        if [(ln.rstrip()[w_comm:] if (strip_common_indent and ln.strip()) else ln.rstrip()) for ln in win_lines] == norm_search:
            last = win_lines[-1]
            matches.append((offsets[idx], offsets[idx + win - 1] + (len(last) if keep_trailing_newline else len(last.rstrip("\r\n"))), w_comm))
    return matches


def _reindent_replacement(replace_block: str, search_block: str, window_indent: int) -> str:
    sc = _common_indent(search_block.splitlines())
    return "\n".join(("" if not ln.strip() else " " * window_indent + ln[min(sc, _leading_ws_len(ln)) :]) for ln in replace_block.splitlines())


def _scoped_editable_units(cpp_source: str, active_kernel: str, *, allow_exclusive_helpers: bool) -> list[_CppUnit]:
    _, units = _split_cpp_units(cpp_source)
    kernels = [u for u in units if u.kind == "kernel"]
    target_unit = next((u for u in kernels if u.name == active_kernel), None) or next((u for u in units if u.name == active_kernel), None)
    if target_unit is None:
        raise ValueError(f"Active kernel {active_kernel!r} not found in C++ source")
    editable = [target_unit]
    if allow_exclusive_helpers and (helpers_by_name := {u.name: u for u in units if u.kind == "helper" and u.name}):
        shared: set[str] = set()
        for kernel in kernels:
            if kernel.name != target_unit.name:
                shared |= _helper_call_closure(kernel, helpers_by_name)
        editable.extend(helpers_by_name[name] for name in (_helper_call_closure(target_unit, helpers_by_name) - shared))
    return sorted(editable, key=lambda u: u.start_char)


def _find_all_substring_spans(text: str, needle: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    pos, step = 0, max(1, len(needle))
    while (idx := text.find(needle, pos)) >= 0:
        spans.append((idx, idx + len(needle)))
        pos = idx + step
    return spans


def _char_span_to_line_span(text: str, start_char: int, end_char: int) -> tuple[int, int]:
    start_line = text.count("\n", 0, start_char) + 1
    adj_end = end_char - 1 if end_char > start_char and text[end_char - 1] == "\n" else end_char
    return start_line, text.count("\n", 0, max(start_char, adj_end)) + 1


def _disambiguate_matches_by_line_window(
    text: str, spans: Sequence[tuple[int, int]], line_window: tuple[int, int] | None
) -> int:
    line_spans = [_char_span_to_line_span(text, s, e) for s, e in spans]
    if line_window is not None:
        w_start, w_end = line_window
        win_center = (w_start + w_end) / 2.0
        qualifying: list[tuple[int, float]] = []
        for idx, (ma, mb) in enumerate(line_spans):
            if (max(0, min(mb, w_end) - max(ma, w_start) + 1) / max(1, mb - ma + 1)) >= 0.5:
                qualifying.append((idx, abs(((ma + mb) / 2.0) - win_center)))
        if qualifying:
            min_dist = min(dist for _, dist in qualifying)
            best = [idx for idx, dist in qualifying if dist == min_dist]
            if len(best) == 1:
                return best[0]
    formatted = ", ".join(f"[L{la}-L{lb}]" for la, lb in line_spans)
    raise AmbiguousPatchAnchorError(
        f"Ambiguous SEARCH block matched {len(spans)} locations at {formatted}; provide [Lstart-Lend] or more context"
    )


def _is_normalized_identical(a: str, b: str) -> bool:
    return a.strip() == b.strip() or " ".join(a.split()) == " ".join(b.split())


def _is_bare_cpp_symbol_patch(patch_text: str) -> bool:
    if "<<<<<<< SEARCH" in patch_text or "<<< SEARCH" in patch_text:
        return False
    clean_patch = strip_markdown_fence_lines(patch_text)
    if _find_python_cpp_literal_spans(clean_patch, require_cpp_tokens=False):
        return False
    _, units = _split_cpp_units(clean_patch)
    return any(u.kind in ("kernel", "helper") and u.name and _DEVICE_QUALIFIER_RE.search(u.signature) for u in units)


def apply_scoped_trackb_patch(
    cpp_source: str, patch_text: str, *, active_kernel: str | None = None, allow_exclusive_helpers: bool = True
) -> tuple[str, int]:
    """Apply Mode-A symbol replacement or Mode-B SEARCH/REPLACE hunks to *cpp_source*."""
    if _SIBLING_MARKER_RE.search(patch_text):
        raise ValueError("Patch attempted to copy or edit a read-only sibling kernel marker")
    hunks = _parse_scoped_search_replace_hunks(patch_text)
    if not hunks:
        if _is_bare_cpp_symbol_patch(patch_text):
            return splice_cpp_units_into_tu(cpp_source, patch_text, active_kernel=active_kernel)
        raise PatchAnchorMissError("patch_text must contain at least one valid SEARCH/REPLACE block")

    updated, applied = cpp_source, 0
    for hunk in hunks:
        search_block, replace_block = hunk.search_block, hunk.replace_block
        if not search_block.strip():
            raise PatchAnchorMissError("Empty SEARCH block is forbidden")
        _reject_introduced_forbidden_tokens(search_block, replace_block)
        scope = (
            [(u.start_char, u.end_char) for u in _scoped_editable_units(updated, active_kernel, allow_exclusive_helpers=allow_exclusive_helpers)]
            if active_kernel is not None else [(0, len(updated))]
        )
        exact_spans = _find_all_substring_spans(updated, search_block)
        is_exact = bool(exact_spans)
        candidates = [(s, e, 0) for s, e in exact_spans] if is_exact else normalized_line_matches(updated, search_block)
        if not candidates:
            raise PatchAnchorMissError("SEARCH block did not match target C++ source")
        picked = _disambiguate_matches_by_line_window(updated, [(s, e) for s, e, _ in candidates], hunk.line_window) if len(candidates) > 1 else 0
        m_start, m_end, win_indent = candidates[picked]
        if not any(m_start >= s and m_end <= e for s, e in scope):
            raise ValueError(f"Scoped patch targets read-only code outside active kernel {active_kernel!r}")
        if is_exact:
            replacement = replace_block
        else:
            replacement = _reindent_replacement(replace_block, search_block, win_indent)
            if updated[m_start:m_end].endswith("\n") and not replacement.endswith("\n"):
                replacement += "\n"
        updated = updated[:m_start] + replacement + updated[m_end:]
        applied += 1

    if applied == 0 or _is_normalized_identical(updated, cpp_source):
        raise NoopIdenticalPatchError("Track-B patch applied zero effective changes to target C++ source")
    return updated, applied


def splice_cpp_units_into_tu(
    cpp_source: str, candidate_cpp: str, *, active_kernel: str | None = None
) -> tuple[str, int]:
    """Replace same-name top-level C++ units of *cpp_source* with *candidate_cpp* units."""
    _, candidate_units = _split_cpp_units(strip_markdown_fence_lines(candidate_cpp))
    replacements = [u for u in candidate_units if u.kind in ("kernel", "helper") and u.name]
    if not replacements:
        raise ValueError("candidate_cpp does not contain any bare C++ kernel or helper definition")
    seen_names: set[str] = set()
    for unit in replacements:
        if unit.name in seen_names:
            raise ValueError(f"Candidate C++ unit {unit.name!r} is defined more than once")
        seen_names.add(unit.name)

    _, units = _split_cpp_units(cpp_source)
    all_existing_names = {u.name for u in units if u.name}
    editable = (
        _scoped_editable_units(cpp_source, active_kernel, allow_exclusive_helpers=True)
        if active_kernel is not None else [u for u in units if u.kind in ("kernel", "helper") and u.name]
    )
    editable_by_name = {u.name: u for u in editable}

    targets: list[tuple[int, int, str]] = []
    replaced_units: list[_CppUnit] = []
    new_helpers: list[_CppUnit] = []
    for unit in replacements:
        if (target := editable_by_name.get(unit.name)) is not None:
            _reject_introduced_forbidden_tokens(target.text, unit.text)
            targets.append((target.start_char, target.end_char, unit.text))
            replaced_units.append(unit)
        elif unit.kind == "helper" and unit.name not in all_existing_names:
            new_helpers.append(unit)
        elif active_kernel is not None:
            raise ValueError(f"Candidate C++ unit {unit.name!r} is outside the editable scope of active kernel {active_kernel!r}")
        else:
            raise ValueError(f"Candidate C++ unit {unit.name!r} not found in C++ source")

    if new_helpers:
        new_by_name = {h.name: h for h in new_helpers}
        reached = set().union(*(_helper_call_closure(s, new_by_name) for s in replaced_units)) if replaced_units else set()
        unreached = [h.name for h in new_helpers if h.name not in reached]
        if not targets or unreached:
            bad = unreached[0] if unreached else new_helpers[0].name
            if active_kernel is not None:
                raise ValueError(f"Candidate C++ unit {bad!r} is outside the editable scope of active kernel {active_kernel!r}")
            raise ValueError(f"Candidate C++ unit {bad!r} not found in C++ source")
        for h in new_helpers:
            _reject_introduced_forbidden_tokens("", h.text)

    updated = cpp_source
    for start_c, end_c, rep_text in sorted(targets, reverse=True):
        updated = updated[:start_c] + rep_text + updated[end_c:]
    if new_helpers:
        min_start = min(start for start, _, _ in targets)
        prefix = "" if (min_start == 0 or updated[min_start - 1] == "\n") else "\n"
        updated = updated[:min_start] + prefix + "\n\n".join(h.text.strip() for h in new_helpers) + "\n\n" + updated[min_start:]
    if _is_normalized_identical(updated, cpp_source):
        raise NoopIdenticalPatchError("Spliced C++ units are identical to parent Track-B translation unit")
    return updated, len(targets) + len(new_helpers)


splice_cpp_units_by_symbol = splice_cpp_units_into_tu


def escalate_primfunc_to_trackb_tu(prim_func: object) -> str:
    """Lower an incumbent TileLang @T.prim_func into an operator-specific C++20 TU."""
    import tilelang  # noqa: PLC0415
    from tilelang import tvm  # noqa: PLC0415
    from tilelang.backend.module import create_backend_context  # noqa: PLC0415

    target_ctx = create_backend_context("ascend").target if tvm.target.Target.current(allow_none=True) is None else contextlib.nullcontext()
    with target_ctx:
        lowered = tilelang.lower(prim_func, target="ascend")
    kernel_src = str(getattr(lowered, "kernel_source", "") or "").strip()
    if not kernel_src:
        raise RuntimeError("tilelang.lower returned empty kernel_source for Track-B escalation")
    return kernel_src


def _eval_static_str_expr(expr: ast.AST, str_consts: Mapping[str, str]) -> str | None:
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return expr.value
    if isinstance(expr, ast.Name) and expr.id in str_consts:
        return str_consts[expr.id]
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
        left, right = _eval_static_str_expr(expr.left, str_consts), _eval_static_str_expr(expr.right, str_consts)
        return (left + right) if (left is not None and right is not None) else None
    return None


def _detect_trackb_prologue_line_offset(raw_src: str, *, parent_source: str = "") -> int:
    """Count lines prepended before trackb_src using pure AST inspection."""
    for candidate_src in (raw_src, parent_source):
        if not candidate_src or "def _render_trackb_tu" not in candidate_src:
            continue
        try:
            tree = ast.parse(candidate_src)
        except SyntaxError:
            continue
        str_consts = {name: val for name, val, _, _ in find_top_level_str_assign_spans(tree, candidate_src.splitlines(keepends=True))}
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "_render_trackb_tu":
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Return) and isinstance(sub.value, ast.BinOp) and isinstance(sub.value.op, ast.Add):
                        if (left_val := _eval_static_str_expr(sub.value.left, str_consts)) is not None:
                            return left_val.count("\n")
                        if left_src := ast.get_source_segment(candidate_src, sub.value.left):
                            return left_src.count("\\n")
    return 0


def _scope_trackb_python_or_cpp(
    source: str, diag: str, *, parent_source: str = "", prologue_line_offset: int = 0,
    focus_lines: Sequence[int] = (), active_kernel: str | None = None,
) -> str:
    merged_focus = tuple(sorted({*extract_cpp_diagnostic_lines(diag), *(int(ln) for ln in focus_lines if int(ln) > 0)}))
    try:
        ast.parse(source)
    except SyntaxError:
        return build_trackb_context_view(source, active_kernel=active_kernel, focus_lines=merged_focus, diagnostic_text=diag, include_line_numbers=False) if any(tok in source for tok in _TRACKB_CPP_QUALIFIERS) else source
    if not (hits := _find_python_cpp_literal_spans(source, require_cpp_tokens=True)):
        return source
    offset = int(prologue_line_offset) if prologue_line_offset > 0 else _detect_trackb_prologue_line_offset(source, parent_source=parent_source)
    adj_focus = tuple(ln - offset for ln in merged_focus if ln > offset) if offset > 0 else merged_focus
    updated = source
    for sym_name, cpp_text, start_pos, end_pos in reversed(hits):
        view = build_trackb_context_view(
            cpp_text, active_kernel=active_kernel, module_source=source, trackb_symbol=sym_name,
            focus_lines=adj_focus, diagnostic_text="" if offset > 0 else diag, include_line_numbers=False,
        )
        updated = updated[:start_pos] + format_cpp_string_literal(view if view.startswith("\n") else f"\n{view}\n") + updated[end_pos:]
    ast.parse(updated)
    return updated


def _looks_like_python_primfunc_slice(source: str) -> bool:
    stripped = source.strip()
    if not stripped or parse_scoped_search_replace_blocks(stripped) or any(t in stripped for t in _TRACKB_CPP_QUALIFIERS):
        return False
    if "@T.prim_func" in stripped or re.search(r"(?m)^\s*(?:async\s+)?def\s+[A-Za-z_]\w*\s*\(", stripped):
        return True
    try:
        return any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in ast.parse(stripped).body)
    except SyntaxError:
        return False


def compact_creative_source_slice(
    source: str, *, max_bytes: int | None = None, focus_lines: Sequence[int] = (),
    diagnostic_text: str = "", active_kernel: str | None = None, parent_source: str = "", prologue_line_offset: int = 0,
) -> str:
    """Scope Track-B C++ sources losslessly without hardcoded byte truncation."""
    _ = max_bytes
    if any(tok in source for tok in _TRACKB_CPP_QUALIFIERS):
        return _scope_trackb_python_or_cpp(source, diagnostic_text, parent_source=parent_source, prologue_line_offset=prologue_line_offset, focus_lines=focus_lines, active_kernel=active_kernel)
    return source


def _score_name_for_regime(name: str, regime_tag: str) -> int:
    if not (tag := regime_tag.strip().lower()):
        return 0
    low = name.lower()
    if tag in low:
        return 100 + len(tag)
    parts, words = re.split(r"[^a-z0-9]+", tag), set(re.split(r"[^a-z0-9]+", low))
    return sum(10 + len(t) for t in parts if len(t) >= 3 and t in low) + sum(3 for t in parts if len(t) >= 2 and t in words)


def _py_node_source_slice(lines: Sequence[str], node: ast.stmt) -> str:
    start, end = stmt_line_span(node)
    return "".join(lines[start - 1 : end]).rstrip()


def _extract_cpp_closure_parts(
    cpp_source: str, *, active_kernel: str | None = None, regime_tag: str = "",
    module_source: str = "", diagnostic_lines: Sequence[int] = (),
) -> tuple[str, tuple[str, ...], str]:
    clean_cpp = strip_markdown_fence_lines(cpp_source)
    _, units = _split_cpp_units(clean_cpp)
    kernels = [u for u in units if u.kind == "kernel" and u.name]
    helpers_by_name = {u.name: u for u in units if u.kind == "helper" and u.name}
    host_launchers, _ = _host_launcher_units(units)
    host_starts = {u.start_char for u in host_launchers}

    target_name = active_kernel
    if target_name is None and diagnostic_lines:
        for ln in diagnostic_lines:
            if (k := next((u.name for u in kernels if u.start_line <= ln <= u.end_line), None)):
                target_name = k; break
            if (h := next((u for u in helpers_by_name.values() if u.start_line <= ln <= u.end_line and u.name), None)) and (k := next((u.name for u in kernels if h.name in _helper_call_closure(u, helpers_by_name)), None)):
                target_name = k; break
    if target_name is None:
        target_name = select_active_trackb_kernel(clean_cpp, regime_tag=regime_tag, module_source=module_source) or (kernels[0].name if kernels else None)
    target_unit = next((u for u in kernels if u.name == target_name), None) or (next((u for u in units if u.name == target_name), None) if target_name is not None else None)
    host_tail = "\n\n".join(u.text.strip() for u in host_launchers if u.text.strip())
    if target_unit is None:
        directives = _extract_directive_lines(clean_cpp, units)
        parts = (["\n".join(txt for _, txt in directives)] if directives else []) + [u.text.strip() for u in units if u.start_char not in host_starts and u.text.strip()]
        return "\n\n".join(p for p in parts if p).strip(), (), host_tail

    entries, sibling_protos = _build_cpp_closure_entries(clean_cpp, target_unit, units, excluded_starts=host_starts)
    device_lines, prev_kind = [], ""
    for _, kind, payload in entries:
        if kind not in ("directive", "unit"):
            continue
        if device_lines and (kind == "unit" or prev_kind != "directive"):
            device_lines.append("")
        if isinstance(payload, tuple):
            device_lines.append(payload[1])
        elif isinstance(payload, _CppUnit):
            device_lines.append(payload.text.strip())
        prev_kind = kind
    return "\n".join(device_lines).strip(), tuple(sibling_protos), host_tail


def _extract_hybrid_trackb_repair_closure(
    module_source: str, tree: ast.Module, cpp_spans: Sequence[tuple[str, str, int, int]],
    *, active_symbols: Sequence[str], regime_tag: str, diagnostic_text: str, protected_exports: frozenset[str],
) -> RepairClosureSlice:
    lines = module_source.splitlines(keepends=True)
    funcs = {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assigns = {n.name: stmt for n in collect_top_level_symbols(tree) if n.kind == "assign" for stmt in (collect_top_level_ast_nodes(tree)[n.name],)}
    ordered_defs = [(s.kind, s.name) for s in collect_top_level_symbols(tree)]
    cpp_names = {name for name, _, _, _ in cpp_spans}
    active_sym_set = set(active_symbols)
    picked = next((s for s in cpp_spans if s[0] in active_sym_set), None)
    if picked is None and regime_tag:
        picked = max(cpp_spans, key=lambda s: _score_name_for_regime(s[0], regime_tag))
    active_cpp_name, active_cpp_text, _, _ = picked or cpp_spans[0]

    offset = _detect_trackb_prologue_line_offset(module_source)
    raw_diag = extract_cpp_diagnostic_lines(diagnostic_text)
    adj_diag = tuple(ln - offset for ln in raw_diag if ln > offset) if offset > 0 else raw_diag
    explicit_kernel = next((sym for sym in active_symbols if sym not in cpp_names and sym not in funcs), None)
    editable_cpp, cpp_protos, cpp_host_tail = _extract_cpp_closure_parts(
        active_cpp_text, active_kernel=explicit_kernel, regime_tag=regime_tag, module_source=module_source, diagnostic_lines=adj_diag
    )

    fn_segments = {name: _py_node_source_slice(lines, node) for name, node in funcs.items()}
    dead_tracka = {
        name for name, seg in fn_segments.items()
        if "@T.prim_func" in seg or (name.startswith("_build_") and "trackb" not in name.lower() and not any(t in seg for t in ("compile_ascend", "_render_", 'extern "C"', "ctypes", "<<<", "TilingData")))
    }
    selector_names = [n for n in funcs if n.startswith(("_select_", "select_")) and n not in dead_tracka]
    kept_selectors = set(([n for n in selector_names if _score_name_for_regime(n, regime_tag) > 0] if regime_tag else []) or selector_names)

    abi_seeds = [
        name for name, seg in fn_segments.items()
        if name not in dead_tracka
        and not (name.startswith(("_select_", "select_")) and name not in kept_selectors)
        and (
            name in active_sym_set or name == "_render_trackb_tu" or name.startswith("_render_")
            or "trackb" in name.lower() or "workspace" in name.lower() or name.startswith(("_select_", "select_"))
            or any(t in seg for t in ('extern "C"', "<<<", "TilingData", "block_dim", "compile_ascend", "ctypes"))
        )
    ]
    all_py_defs: dict[str, ast.stmt] = {**funcs, **assigns}
    eligible_refs = {
        ref for ref in all_py_defs
        if ref not in dead_tracka and ref not in cpp_names
        and not (ref.startswith(("_select_", "select_")) and ref not in kept_selectors)
    }
    abi_visited = {s for s in abi_seeds if s in eligible_refs} | collect_referenced_helper_names(
        all_py_defs, helper_names=eligible_refs, seed_roots=abi_seeds
    )
    abi_visited.update(name for kind, name in ordered_defs if kind == "assign" and name in protected_exports)

    abi_parts: list[str] = []
    seen_assigns: set[int] = set()
    for kind, name in ordered_defs:
        if name not in abi_visited:
            continue
        if kind == "func":
            abi_parts.append(fn_segments[name])
        elif id(assigns[name]) not in seen_assigns:
            seen_assigns.add(id(assigns[name]))
            abi_parts.append(_py_node_source_slice(lines, assigns[name]))
    if cpp_host_tail:
        abi_parts.append(cpp_host_tail)

    extra_protos = list(cpp_protos)
    for cname, ctext, _, _ in cpp_spans:
        if cname != active_cpp_name:
            for u in _split_cpp_units(ctext)[1]:
                if u.kind == "kernel" and u.name:
                    decl = f"{u.signature.rstrip(';').strip()};" if u.signature.strip() else f"void {u.name}();"
                    if decl not in extra_protos:
                        extra_protos.append(decl)
    return RepairClosureSlice(editable_cpp, tuple(extra_protos), "\n\n".join(p for p in abi_parts if p).strip(), diagnostic_text.strip())


def _extract_tracka_repair_closure(
    module_source: str, *, active_symbols: Sequence[str], regime_tag: str, diagnostic_text: str, protected_exports: frozenset[str]
) -> RepairClosureSlice:
    editable_raw, readonly_tail = split_tracka_editable_and_readonly_tail(module_source, protected_exports)
    if not editable_raw.strip():
        return RepairClosureSlice("", (), readonly_tail, diagnostic_text.strip())
    try:
        tree = ast.parse(editable_raw)
    except SyntaxError:
        return RepairClosureSlice(editable_raw.strip(), (), readonly_tail, diagnostic_text.strip())

    lines = editable_raw.splitlines(keepends=True)
    imports: list[tuple[set[str], str]] = []
    funcs: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    all_defs: dict[str, ast.stmt] = {}
    ordered_defs: list[tuple[str, str]] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.append(({a.asname or a.name.split(".")[0] for a in node.names}, _py_node_source_slice(lines, node)))
        elif isinstance(node, ast.ImportFrom):
            imports.append(({a.asname or a.name for a in node.names}, _py_node_source_slice(lines, node)))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs[node.name] = node
            all_defs[node.name] = node
            ordered_defs.append(("func", node.name))
        elif isinstance(node, ast.ClassDef):
            all_defs[node.name] = node
            ordered_defs.append(("class", node.name))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            for bname in _extract_top_level_bound_names(node):
                all_defs[bname] = node
                ordered_defs.append(("assign", bname))

    builder_fns = {n for n, fn in funcs.items() if "@T.prim_func" in _py_node_source_slice(lines, fn) or n.startswith("_build_") or n in ("make_kernel", "main")}
    selector_fns = {n for n in funcs if n.startswith(("_select_", "select_"))}
    seeds = [s for s in active_symbols if s in all_defs]
    diag_lines = set(extract_cpp_diagnostic_lines(diagnostic_text)) | {int(m.group(1)) for m in re.finditer(r":(\d+)(?::\d+)?\b", diagnostic_text) if int(m.group(1)) > 0}
    for name, fnode in funcs.items():
        if any(fnode.lineno <= ln <= (fnode.end_lineno or fnode.lineno) for ln in diag_lines) and name not in seeds:
            seeds.append(name)
    if not seeds and regime_tag:
        for pool in (selector_fns, builder_fns):
            if scored := [(_score_name_for_regime(n, regime_tag), n) for n in pool if _score_name_for_regime(n, regime_tag) > 0]:
                top = max(sc for sc, _ in scored)
                seeds.extend(n for sc, n in scored if sc == top and n not in seeds)
    if seeds and not any(s in selector_fns for s in seeds):
        if len(selector_fns) == 1:
            seeds.extend(selector_fns)
        elif regime_tag:
            seeds.extend(n for n in selector_fns if _score_name_for_regime(n, regime_tag) > 0 and n not in seeds)
    if not seeds:
        seeds.extend(sorted(selector_fns | builder_fns) if (builder_fns or selector_fns) else [n for _, n in ordered_defs])

    allowed_pool = set(all_defs) - ((builder_fns | selector_fns) - set(seeds))
    visited = ({s for s in seeds if s in all_defs} | collect_referenced_helper_names(all_defs, helper_names=allowed_pool, seed_roots=seeds))
    used_names = set(visited).union(*({sub.id for sub in ast.walk(all_defs[sym]) if isinstance(sub, ast.Name)} for sym in visited if sym in all_defs))

    parts = [imp_src for bound, imp_src in imports if (bound & used_names) or "*" in bound]
    seen_assigns: set[int] = set()
    for kind, name in ordered_defs:
        if name not in visited:
            continue
        if kind == "assign":
            if id(all_defs[name]) not in seen_assigns:
                seen_assigns.add(id(all_defs[name]))
                parts.append(_py_node_source_slice(lines, all_defs[name]))
        else:
            parts.append(_py_node_source_slice(lines, all_defs[name]))

    sibling_protos = tuple(
        f"{'async def' if isinstance(funcs[name], ast.AsyncFunctionDef) else 'def'} {name}({ast.unparse(funcs[name].args)})"
        f"{f' -> {ast.unparse(funcs[name].returns)}' if funcs[name].returns is not None else ''}: ..."
        for kind, name in ordered_defs if kind == "func" and name not in visited
    )
    return RepairClosureSlice("\n\n".join(p for p in parts if p.strip()).strip(), sibling_protos, readonly_tail, diagnostic_text.strip())


def extract_repair_closure_slice(
    module_source: str, *, active_symbols: Sequence[str] = (), regime_tag: str = "",
    backend_kind: str = "tilelang_dsl", diagnostic_text: str = "", protected_exports: frozenset[str] | None = None,
) -> RepairClosureSlice:
    """Extract lossless reachable call-graph closure and host ABI slice without byte caps."""
    if not module_source.strip():
        return RepairClosureSlice("", (), "", diagnostic_text.strip())
    eff_protected = DEFAULT_PROTECTED_EXPORTS if protected_exports is None else protected_exports
    is_trackb = str(backend_kind or "").strip().lower() in ("ascendc_trackb", "hybrid_trackb", "ascendc_template", "cce_cpp20", "cce_trackb")
    try:
        parsed_tree: ast.Module | None = ast.parse(module_source)
    except SyntaxError:
        parsed_tree = None
    if parsed_tree is not None and (cpp_spans := _find_python_cpp_literal_spans(module_source, require_cpp_tokens=True)) and (is_trackb or not _looks_like_python_primfunc_slice(module_source)):
        return _extract_hybrid_trackb_repair_closure(
            module_source, parsed_tree, cpp_spans, active_symbols=active_symbols,
            regime_tag=regime_tag, diagnostic_text=diagnostic_text, protected_exports=eff_protected,
        )
    if parsed_tree is None and any(tok in module_source for tok in _TRACKB_CPP_QUALIFIERS) and (is_trackb or not _looks_like_python_primfunc_slice(module_source)):
        editable_cpp, sibling_protos, host_tail = _extract_cpp_closure_parts(
            module_source, active_kernel=next((s for s in active_symbols if s), None),
            regime_tag=regime_tag, module_source=module_source, diagnostic_lines=extract_cpp_diagnostic_lines(diagnostic_text),
        )
        return RepairClosureSlice(editable_cpp, sibling_protos, host_tail, diagnostic_text.strip())
    return _extract_tracka_repair_closure(module_source, active_symbols=active_symbols, regime_tag=regime_tag, diagnostic_text=diagnostic_text, protected_exports=eff_protected)


__all__ = [
    "AmbiguousPatchAnchorError", "DEFAULT_PROTECTED_EXPORTS", "NoopIdenticalPatchError",
    "PatchAnchorMissError", "RepairClosureSlice", "TRACKB_REPLY_CONTRACT_PROMPT", "TopLevelSymbol",
    "_TRACKB_CPP_QUALIFIERS", "_detect_trackb_prologue_line_offset", "_find_python_cpp_literal_spans",
    "_looks_like_python_primfunc_slice", "apply_scoped_trackb_patch", "build_trackb_context_view",
    "cached_callable_signature_info", "classify_stage_failure_for_repair", "collect_referenced_helper_names",
    "collect_top_level_ast_nodes", "collect_top_level_symbols", "compact_creative_source_slice",
    "escalate_primfunc_to_trackb_tu", "extract_cpp_diagnostic_lines", "extract_reachable_trackb_source",
    "extract_repair_closure_slice", "extract_scoped_trackb_kernel_view",
    "extract_trackb_launcher_preview", "filter_callable_kwargs", "find_top_level_func",
    "find_top_level_str_assign_spans", "format_cpp_string_literal", "format_trackb_reply_contract",
    "normalized_line_matches", "parse_scoped_search_replace_blocks", "py_char_offset",
    "render_active_symbol_closure_view", "render_active_symbol_closure_with_line_map",
    "resolve_instantiated_trackb_kernel", "scan_launched_trackb_kernels",
    "select_active_trackb_kernel", "splice_cpp_units_by_symbol", "splice_cpp_units_into_tu",
    "split_tracka_editable_and_readonly_tail", "stmt_line_span", "strip_cpp_comments_and_strings",
    "strip_markdown_fence_lines",
]
