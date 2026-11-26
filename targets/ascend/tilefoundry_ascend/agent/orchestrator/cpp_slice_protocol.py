"""Scoped C++20 slicing, patching, and Track-A-to-Track-B escalation protocol."""

from __future__ import annotations

import ast
import contextlib
import re
from collections.abc import Mapping, Sequence
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

_SEARCH_REPLACE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"<<<<<<<\s*SEARCH(?:\s*\[L?\d+\s*-\s*L?\d+\])?\s*\n(.*?)\n=======\s*\n(.*?)\n>>>>>>>\s*REPLACE",
        re.DOTALL,
    ),
    re.compile(
        r"<<<\s*SEARCH(?:\s*\[L?\d+\s*-\s*L?\d+\])?\s*\n(.*?)\n===\s*\n(.*?)\n>>>\s*REPLACE",
        re.DOTALL,
    ),
)
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


TRACKB_REPLY_CONTRACT_PROMPT = (
    "### TRACK-B REPLY CONTRACT\n"
    "- Prefer Mode A (complete bare C++20 definition of the active target kernel or exclusive "
    "helper, keeping exact symbol name and signature) or Mode B (line-window anchored "
    "`<<<<<<< SEARCH [Lstart-Lend]` / `=======` / `>>>>>>> REPLACE` hunks using `[L0001]` line "
    "numbers from the active closure view).\n"
    "- Never edit or echo `// [READONLY_SIBLING_DECLARATION]` or `/* [READ-ONLY HELPER] */` lines."
)


def format_trackb_reply_contract(active_kernel: str | None = None) -> str:
    """Render Track-B patch instructions preferring Mode A symbol defs or Mode B [Lstart-Lend]."""
    if active_kernel:
        return (
            f"{TRACKB_REPLY_CONTRACT_PROMPT}\n"
            f"- Active editable kernel scope: `{active_kernel}` and its exclusive helpers."
        )
    return TRACKB_REPLY_CONTRACT_PROMPT


def extract_cpp_diagnostic_lines(diagnostic_text: str) -> tuple[int, ...]:
    """Extract sorted unique positive 1-indexed line numbers from C++ diagnostics."""
    if not diagnostic_text:
        return ()
    lines: set[int] = set()
    for pat in (_DIAG_FILE_LINE_RE, _DIAG_WORD_LINE_RE):
        for match in pat.finditer(diagnostic_text):
            line_num = int(match.group(1))
            if line_num > 0:
                lines.add(line_num)
    return tuple(sorted(lines))


DEFAULT_PROTECTED_EXPORTS: frozenset[str] = frozenset(
    {"kernel", "ref_kernel", "build_kernel", "_ENTRY_CACHE", "__all__"}
)

_NON_REPAIRABLE_FAILURE_CODES: frozenset[str] = frozenset(
    {
        "NOOP_IDENTICAL",
        "UB_SRAM_OVERFLOW",
        "UB_OVERFLOW",
        "L1_SRAM_OVERFLOW",
        "L0C_SRAM_OVERFLOW",
        "VF_STACK_OVERFLOW",
    }
)


@dataclass(frozen=True)
class RepairClosureSlice:
    """Self-contained repair context spanning active device call-graph and host ABI."""

    editable_device_source: str
    sibling_prototypes: tuple[str, ...]
    readonly_host_abi_tail: str
    diagnostic_excerpt: str


def classify_stage_failure_for_repair(failure_code: str) -> bool:
    """Return True only when syntactic or API repair can resolve the failure."""
    return str(failure_code or "").strip().upper() not in _NON_REPAIRABLE_FAILURE_CODES


def _extract_top_level_bound_names(node: ast.stmt) -> set[str]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return {node.name}
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return {node.target.id}
    if isinstance(node, ast.Assign):
        return {t.id for t in node.targets if isinstance(t, ast.Name)}
    return set()


def split_tracka_editable_and_readonly_tail(
    module_source: str,
    protected_exports: frozenset[str] = DEFAULT_PROTECTED_EXPORTS,
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
                names = _extract_top_level_bound_names(node)
                if names & protected_exports:
                    decs = getattr(node, "decorator_list", ())
                    node_start = min((d.lineno for d in decs), default=node.lineno)
                    split_lineno = min(split_lineno, node_start)
                    break
    except SyntaxError:
        escaped = "|".join(re.escape(name) for name in sorted(protected_exports))
        if escaped:
            tail_re = re.compile(
                rf"^(?:(?:async\s+)?def\s+(?:{escaped})\b|(?:{escaped})\s*[:=])"
            )
            for idx, raw_line in enumerate(lines, start=1):
                if tail_re.match(raw_line):
                    split_lineno = idx
                    break
    if split_lineno > len(lines):
        return module_source.rstrip() + "\n", ""
    editable = "".join(lines[: split_lineno - 1]).rstrip()
    editable_out = f"{editable}\n" if editable else ""
    readonly_tail = "".join(lines[split_lineno - 1 :]).strip()
    return editable_out, readonly_tail


def extract_trackb_launcher_preview(
    module_source: str,
    *,
    max_bytes: int | None = None,
) -> str:
    """Extract C++ extern-C launcher stubs and host binding lines without byte truncation."""
    _ = max_bytes
    if not module_source.strip():
        return ""
    previews: list[str] = []
    if any(tok in module_source for tok in _TRACKB_CPP_QUALIFIERS) or 'extern "C"' in module_source:
        _, units = _split_cpp_units(module_source)
        host_units, _ = _host_launcher_units(units)
        for unit in host_units:
            snippet = unit.text.strip()
            if snippet and snippet not in previews:
                previews.append(snippet)

    if not previews:
        for match in re.finditer(r'extern\s+"C"\s+', module_source):
            start = match.start()
            brace_open = module_source.find("{", start)
            if brace_open < 0:
                continue
            depth = 0
            end = -1
            for idx in range(brace_open, len(module_source)):
                ch = module_source[idx]
                if ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        end = idx + 1
                        break
            if end > start:
                snippet = module_source[start:end].strip()
                if snippet and snippet not in previews:
                    previews.append(snippet)

    if not previews:
        try:
            tree = ast.parse(module_source)
            lines = module_source.splitlines()
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and "trackb" in node.name:
                    assert node.end_lineno is not None
                    hdr = "\n".join(
                        lines[node.lineno - 1 : min(node.lineno + 6, node.end_lineno)]
                    )
                    previews.append(hdr.strip())
                    break
        except SyntaxError:
            pass

    return "\n\n".join(previews).strip()


def resolve_instantiated_trackb_kernel(
    instantiated_tu: str,
    raw_cpp_source: str,
) -> str | None:
    """Resolve the unique active ``__global__`` kernel from instantiated launcher truth.

    The module renderer appends an ``extern "C"`` launcher stub to the raw Track-B
    source, so the ``<<<...>>>`` call sites in that tail name the kernel chosen for
    the representative case. Comments and literals are masked before scanning;
    repeated launches of one kernel resolve, while distinct launched kernels are
    ambiguous and return ``None``.
    """
    _, units = _split_cpp_units(raw_cpp_source)
    kernels = {u.name for u in units if u.kind == "kernel" and u.name}
    if not kernels or not instantiated_tu.strip():
        return None
    masked = strip_cpp_comments_and_strings(instantiated_tu)
    launch_region = (
        masked[len(raw_cpp_source) :]
        if instantiated_tu.startswith(raw_cpp_source)
        else masked
    )
    launched = {
        match.group(1)
        for match in _LAUNCH_CONFIG_RE.finditer(launch_region)
        if match.group(1) in kernels
    }
    if len(launched) == 1:
        return next(iter(launched))
    return None


def _extract_directive_lines(
    cpp_source: str,
    units: Sequence[_CppUnit],
) -> list[tuple[int, str]]:
    """Return 1-indexed preprocessor directive lines outside parsed C++ units."""
    unit_ranges = [(u.start_line, u.end_line) for u in units]
    directives: list[tuple[int, str]] = []
    for lineno, raw_line in enumerate(cpp_source.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped.startswith("#"):
            continue
        if any(start <= lineno <= end for start, end in unit_ranges):
            continue
        directives.append((lineno, raw_line.rstrip("\r\n")))
    return directives


def _format_closure_block(
    lines_with_nums: Sequence[tuple[int, str]],
    *,
    include_line_numbers: bool,
    out_lines: list[str],
    line_map: dict[int, int],
) -> None:
    """Append a lossless TU block and record output-to-TU line mappings."""
    for tu_lineno, raw_line in lines_with_nums:
        rendered = f"[L{tu_lineno:04d}] {raw_line}" if include_line_numbers else raw_line
        out_lines.append(rendered)
        line_map[len(out_lines)] = tu_lineno


def render_active_symbol_closure_with_line_map(
    cpp_source: str,
    active_kernel: str,
    *,
    include_line_numbers: bool = True,
) -> tuple[str, dict[int, int]]:
    """Render 100% lossless active closure with line numbers and single-line sibling decls."""
    clean_cpp = strip_markdown_fence_lines(cpp_source)
    _, units = _split_cpp_units(clean_cpp)
    kernels = [u for u in units if u.kind == "kernel"]
    helpers = [u for u in units if u.kind == "helper"]

    target_unit = next((u for u in kernels if u.name == active_kernel), None)
    if target_unit is None:
        target_unit = next((u for u in units if u.name == active_kernel), None)
    if target_unit is None:
        raise ValueError(f"Active kernel {active_kernel!r} not found in C++ source")

    helpers_by_name = {h.name: h for h in helpers if h.name}
    reached_names = _helper_call_closure(target_unit, helpers_by_name)
    directives = _extract_directive_lines(clean_cpp, units)

    entries: list[tuple[int, str, object]] = []
    for d_lineno, d_text in directives:
        entries.append((d_lineno, "directive", (d_lineno, d_text)))
    for unit in units:
        if unit.kind == "global_decl":
            entries.append((unit.start_line, "unit", unit))
        elif unit.kind == "helper":
            if unit.start_char == target_unit.start_char or (
                unit.name and unit.name in reached_names
            ):
                entries.append((unit.start_line, "unit", unit))
            else:
                entries.append((unit.start_line, "ro_helper", unit))
        elif unit.kind == "kernel":
            if unit.start_char == target_unit.start_char:
                entries.append((unit.start_line, "unit", unit))
            else:
                entries.append((unit.start_line, "ro_sibling", unit))

    entries.sort(key=lambda item: item[0])

    out_lines: list[str] = []
    line_map: dict[int, int] = {}
    prev_kind = ""
    for _, kind, payload in entries:
        if kind == "directive":
            if out_lines and prev_kind != "directive":
                out_lines.append("")
            d_pair = payload
            assert isinstance(d_pair, tuple)
            _format_closure_block(
                [d_pair],
                include_line_numbers=include_line_numbers,
                out_lines=out_lines,
                line_map=line_map,
            )
        elif kind == "unit":
            unit = payload
            assert isinstance(unit, _CppUnit)
            if out_lines:
                out_lines.append("")
            unit_lines = [
                (unit.start_line + idx, raw_ln)
                for idx, raw_ln in enumerate(unit.text.rstrip("\r\n").splitlines())
            ]
            _format_closure_block(
                unit_lines,
                include_line_numbers=include_line_numbers,
                out_lines=out_lines,
                line_map=line_map,
            )
        elif kind == "ro_helper":
            unit = payload
            assert isinstance(unit, _CppUnit)
            sig = unit.signature.rstrip(";").strip()
            if sig:
                if out_lines and prev_kind != "ro_helper":
                    out_lines.append("")
                out_lines.append(f"/* [READ-ONLY HELPER] */ {sig};")
        elif kind == "ro_sibling":
            unit = payload
            assert isinstance(unit, _CppUnit)
            sig = unit.signature.rstrip(";").strip()
            decl = f"{sig};" if sig else f"void {unit.name}();"
            if out_lines and prev_kind != "ro_sibling":
                out_lines.append("")
            out_lines.append(f"// [READONLY_SIBLING_DECLARATION] {decl}")
        prev_kind = kind

    return "\n".join(out_lines).strip(), line_map


def render_active_symbol_closure_view(
    cpp_source: str,
    active_kernel: str,
    *,
    include_line_numbers: bool = True,
) -> str:
    """Render 100% lossless active closure with optional [L0001] line-number anchors."""
    view_text, _ = render_active_symbol_closure_with_line_map(
        cpp_source,
        active_kernel,
        include_line_numbers=include_line_numbers,
    )
    return view_text


def build_trackb_context_view(
    cpp_source: str,
    *,
    active_kernel: str | None = None,
    regime_tag: str = "",
    module_source: str = "",
    trackb_symbol: str = "",
    launcher_preview: str = "",
    max_bytes: int = 49152,
    focus_lines: Sequence[int] = (),
    focus_window: int = 25,
    diagnostic_text: str = "",
    include_line_numbers: bool = True,
) -> str:
    """Build a lossless active-closure C++20 context view with single-line sibling decls."""
    _ = (max_bytes, focus_lines, focus_window, diagnostic_text)
    clean_cpp = strip_markdown_fence_lines(cpp_source)
    _, units = _split_cpp_units(clean_cpp)
    kernels = [u for u in units if u.kind == "kernel"]
    target_name = select_active_trackb_kernel(
        clean_cpp,
        active_kernel=active_kernel,
        regime_tag=regime_tag,
        module_source=module_source,
    )
    if target_name is None and kernels:
        target_name = kernels[0].name

    preview = launcher_preview.strip() or extract_trackb_launcher_preview(module_source)
    header_parts: list[str] = []
    if trackb_symbol:
        header_parts.append(
            f"/* TRACK-B SCOPED SLICE: symbol={trackb_symbol} active={target_name} */"
        )
    if preview:
        header_parts.append(f"/* LAUNCHER BINDING PREVIEW */\n{preview}")

    if target_name is not None:
        closure_body = render_active_symbol_closure_view(
            clean_cpp,
            target_name,
            include_line_numbers=include_line_numbers,
        )
    else:
        directives = _extract_directive_lines(clean_cpp, units)
        out_lines: list[str] = []
        line_map: dict[int, int] = {}
        if directives:
            _format_closure_block(
                directives,
                include_line_numbers=include_line_numbers,
                out_lines=out_lines,
                line_map=line_map,
            )
        for unit in units:
            if out_lines:
                out_lines.append("")
            unit_lines = [
                (unit.start_line + idx, raw_ln)
                for idx, raw_ln in enumerate(unit.text.rstrip("\r\n").splitlines())
            ]
            _format_closure_block(
                unit_lines,
                include_line_numbers=include_line_numbers,
                out_lines=out_lines,
                line_map=line_map,
            )
        closure_body = "\n".join(out_lines).strip()

    all_parts = [p for p in (*header_parts, closure_body) if p]
    return "\n\n".join(all_parts).strip()


extract_scoped_trackb_kernel_view = build_trackb_context_view


@dataclass(frozen=True)
class _SearchReplaceHunk:
    """Parsed SEARCH/REPLACE patch hunk with optional [Lstart-Lend] line window."""

    search_block: str
    replace_block: str
    line_window: tuple[int, int] | None = None


def _parse_scoped_search_replace_hunks(patch_text: str) -> list[_SearchReplaceHunk]:
    """Parse SEARCH/REPLACE hunks including optional [Lstart-Lend] anchor windows."""
    cleaned_patch = strip_markdown_fence_lines(patch_text)
    for pat in _SEARCH_REPLACE_WINDOW_PATTERNS:
        hunks: list[_SearchReplaceHunk] = []
        for m in pat.finditer(cleaned_patch):
            start_raw = m.group("start")
            end_raw = m.group("end")
            window: tuple[int, int] | None = None
            if start_raw is not None and end_raw is not None:
                a = int(start_raw)
                b = int(end_raw)
                window = (min(a, b), max(a, b))
            hunks.append(
                _SearchReplaceHunk(
                    search_block=strip_markdown_fence_lines(m.group("search")),
                    replace_block=strip_markdown_fence_lines(m.group("replace")),
                    line_window=window,
                )
            )
        if hunks:
            return hunks
    return []


def parse_scoped_search_replace_blocks(patch_text: str) -> list[tuple[str, str]]:
    """Parse SEARCH/REPLACE patch hunks from scoped Track-B candidate output."""
    return [
        (h.search_block, h.replace_block)
        for h in _parse_scoped_search_replace_hunks(patch_text)
    ]


def _reject_introduced_forbidden_tokens(search_block: str, replace_block: str) -> None:
    """Reject a Track-B replacement that increases boundary-delimited forbidden token counts."""
    introduced = CCE_CPP20_SURFACE.introduced_forbidden_tokens(search_block, replace_block)
    if introduced:
        joined = ", ".join(introduced)
        raise ValueError(f"Track-B CCE patch introduces forbidden symbols: {joined}")


def _leading_ws_len(line: str) -> int:
    return len(line) - len(line.lstrip(" \t"))


def _common_indent(lines: Sequence[str]) -> int:
    indents = [_leading_ws_len(line) for line in lines if line.strip()]
    return min(indents) if indents else 0


def _normalized_line_matches(
    text: str,
    search_block: str,
) -> list[tuple[int, int, int]]:
    """Locate *search_block* windows after stripping line-tail whitespace and common indent."""
    text_lines = text.splitlines(keepends=True)
    search_lines = search_block.splitlines()
    if not search_lines or not any(line.strip() for line in search_lines):
        return []
    search_common = _common_indent(search_lines)
    normalized_search = [
        line.rstrip()[search_common:] if line.strip() else "" for line in search_lines
    ]
    offsets: list[int] = []
    pos = 0
    for line in text_lines:
        offsets.append(pos)
        pos += len(line)
    window = len(search_lines)
    matches: list[tuple[int, int, int]] = []
    for idx in range(len(text_lines) - window + 1):
        window_lines = text_lines[idx : idx + window]
        window_common = _common_indent(window_lines)
        normalized_window = [
            line.rstrip()[window_common:] if line.strip() else "" for line in window_lines
        ]
        if normalized_window != normalized_search:
            continue
        start = offsets[idx]
        end = offsets[idx + window - 1] + len(window_lines[-1])
        matches.append((start, end, window_common))
    return matches


def _reindent_replacement(replace_block: str, search_block: str, window_indent: int) -> str:
    search_common = _common_indent(search_block.splitlines())
    lines: list[str] = []
    for line in replace_block.splitlines():
        if not line.strip():
            lines.append("")
            continue
        lines.append(" " * window_indent + line[min(search_common, _leading_ws_len(line)) :])
    return "\n".join(lines)


def _scoped_editable_units(
    cpp_source: str,
    active_kernel: str,
    *,
    allow_exclusive_helpers: bool,
) -> list[_CppUnit]:
    """Return the active kernel unit plus helpers reachable only from that kernel."""
    _, units = _split_cpp_units(cpp_source)
    kernels = [u for u in units if u.kind == "kernel"]
    target_unit = next((u for u in kernels if u.name == active_kernel), None)
    if target_unit is None:
        target_unit = next((u for u in units if u.name == active_kernel), None)
    if target_unit is None:
        raise ValueError(f"Active kernel {active_kernel!r} not found in C++ source")
    editable = [target_unit]
    if allow_exclusive_helpers:
        helpers_by_name = {u.name: u for u in units if u.kind == "helper" and u.name}
        if helpers_by_name:
            shared: set[str] = set()
            for kernel in kernels:
                if kernel.name == target_unit.name:
                    continue
                shared |= _helper_call_closure(kernel, helpers_by_name)
            exclusive = _helper_call_closure(target_unit, helpers_by_name) - shared
            editable.extend(helpers_by_name[name] for name in exclusive)
    return sorted(editable, key=lambda u: u.start_char)


def _find_all_substring_spans(text: str, needle: str) -> list[tuple[int, int]]:
    """Return all non-overlapping (start_char, end_char) spans of *needle* in *text*."""
    spans: list[tuple[int, int]] = []
    pos = 0
    step = max(1, len(needle))
    while True:
        idx = text.find(needle, pos)
        if idx < 0:
            break
        spans.append((idx, idx + len(needle)))
        pos = idx + step
    return spans


def _char_span_to_line_span(text: str, start_char: int, end_char: int) -> tuple[int, int]:
    """Convert a character range into an inclusive 1-indexed (start_line, end_line) span."""
    start_line = text.count("\n", 0, start_char) + 1
    adj_end = end_char - 1 if end_char > start_char and text[end_char - 1] == "\n" else end_char
    end_line = text.count("\n", 0, max(start_char, adj_end)) + 1
    return start_line, end_line


def _disambiguate_matches_by_line_window(
    text: str,
    spans: Sequence[tuple[int, int]],
    line_window: tuple[int, int] | None,
) -> int:
    """Select the unique match matching *line_window*, else raise AmbiguousPatchAnchorError."""
    line_spans = [_char_span_to_line_span(text, s, e) for s, e in spans]
    if line_window is not None:
        w_start, w_end = line_window
        win_center = (w_start + w_end) / 2.0
        qualifying: list[tuple[int, float]] = []
        for idx, (ma, mb) in enumerate(line_spans):
            ov_start = max(ma, w_start)
            ov_end = min(mb, w_end)
            overlap_lines = max(0, ov_end - ov_start + 1)
            match_lines = max(1, mb - ma + 1)
            if (overlap_lines / match_lines) >= 0.5:
                dist = abs(((ma + mb) / 2.0) - win_center)
                qualifying.append((idx, dist))
        if qualifying:
            min_dist = min(dist for _, dist in qualifying)
            best = [idx for idx, dist in qualifying if dist == min_dist]
            if len(best) == 1:
                return best[0]
    formatted = ", ".join(f"[L{la}-L{lb}]" for la, lb in line_spans)
    raise AmbiguousPatchAnchorError(
        f"Ambiguous SEARCH block matched {len(spans)} locations at {formatted}; "
        "provide [Lstart-Lend] or more context"
    )


def _is_normalized_identical(a: str, b: str) -> bool:
    """Return True when *a* and *b* are identical after whitespace normalization."""
    if a.strip() == b.strip():
        return True
    return " ".join(a.split()) == " ".join(b.split())


def _is_bare_cpp_symbol_patch(patch_text: str) -> bool:
    """Return True when *patch_text* is a bare Mode-A C++ kernel/helper definition."""
    if "<<<<<<< SEARCH" in patch_text or "<<< SEARCH" in patch_text:
        return False
    clean_patch = strip_markdown_fence_lines(patch_text)
    if _find_python_cpp_literal_spans(clean_patch, require_cpp_tokens=False):
        return False
    _, units = _split_cpp_units(clean_patch)
    return any(
        u.kind in ("kernel", "helper")
        and bool(u.name)
        and bool(_DEVICE_QUALIFIER_RE.search(u.signature))
        for u in units
    )


def apply_scoped_trackb_patch(
    cpp_source: str,
    patch_text: str,
    *,
    active_kernel: str | None = None,
    allow_exclusive_helpers: bool = True,
) -> tuple[str, int]:
    """Apply Mode-A symbol replacement or Mode-B SEARCH/REPLACE hunks to *cpp_source*.

    With *active_kernel* set, edits must stay within the active kernel and its
    exclusive helpers. Ambiguous SEARCH hunks require a `[Lstart-Lend]` window
    and raise AmbiguousPatchAnchorError listing every `[La-Lb]` span otherwise.
    """
    if _SIBLING_MARKER_RE.search(patch_text):
        raise ValueError("Patch attempted to copy or edit a read-only sibling kernel marker")

    hunks = _parse_scoped_search_replace_hunks(patch_text)
    if not hunks:
        if _is_bare_cpp_symbol_patch(patch_text):
            return splice_cpp_units_into_tu(
                cpp_source,
                patch_text,
                active_kernel=active_kernel,
            )
        raise PatchAnchorMissError(
            "patch_text must contain at least one valid SEARCH/REPLACE block"
        )

    updated = cpp_source
    applied = 0
    for hunk in hunks:
        search_block = hunk.search_block
        replace_block = hunk.replace_block
        if not search_block.strip():
            raise PatchAnchorMissError("Empty SEARCH block is forbidden")
        _reject_introduced_forbidden_tokens(search_block, replace_block)

        if active_kernel is not None:
            editable = _scoped_editable_units(
                updated,
                active_kernel,
                allow_exclusive_helpers=allow_exclusive_helpers,
            )
            scope = [(unit.start_char, unit.end_char) for unit in editable]
        else:
            scope = [(0, len(updated))]

        def _in_scope(start_char: int, end_char: int) -> bool:
            return any(start_char >= start and end_char <= end for start, end in scope)

        exact_spans = _find_all_substring_spans(updated, search_block)
        if not exact_spans:
            fallback = _normalized_line_matches(updated, search_block)
            if not fallback:
                raise PatchAnchorMissError("SEARCH block did not match target C++ source")
            if len(fallback) > 1:
                picked_idx = _disambiguate_matches_by_line_window(
                    updated,
                    [(s, e) for s, e, _ in fallback],
                    hunk.line_window,
                )
            else:
                picked_idx = 0
            match_start, match_end, window_indent = fallback[picked_idx]
            if not _in_scope(match_start, match_end):
                raise ValueError(
                    f"Scoped patch targets read-only code outside active kernel {active_kernel!r}"
                )
            replacement = _reindent_replacement(replace_block, search_block, window_indent)
            if updated[match_start:match_end].endswith("\n") and not replacement.endswith("\n"):
                replacement = replacement + "\n"
            updated = updated[:match_start] + replacement + updated[match_end:]
            applied += 1
            continue

        if len(exact_spans) > 1:
            picked_idx = _disambiguate_matches_by_line_window(
                updated,
                exact_spans,
                hunk.line_window,
            )
        else:
            picked_idx = 0
        match_start, match_end = exact_spans[picked_idx]
        if not _in_scope(match_start, match_end):
            raise ValueError(
                f"Scoped patch targets read-only code outside active kernel {active_kernel!r}"
            )

        updated = updated[:match_start] + replace_block + updated[match_end:]
        applied += 1

    if applied == 0 or _is_normalized_identical(updated, cpp_source):
        raise NoopIdenticalPatchError(
            "Track-B patch applied zero effective changes to target C++ source"
        )
    return updated, applied


def splice_cpp_units_into_tu(
    cpp_source: str,
    candidate_cpp: str,
    *,
    active_kernel: str | None = None,
) -> tuple[str, int]:
    """Replace same-name top-level C++ units of *cpp_source* with *candidate_cpp* units.

    Bare kernel and helper definitions from *candidate_cpp* are spliced in place by
    unit name. With *active_kernel* set, only that kernel and its exclusive helpers
    are replaceable, so a candidate can never overwrite a read-only sibling kernel.
    """
    clean_candidate = strip_markdown_fence_lines(candidate_cpp)
    _, candidate_units = _split_cpp_units(clean_candidate)
    replacements = [u for u in candidate_units if u.kind in ("kernel", "helper") and u.name]
    if not replacements:
        raise ValueError("candidate_cpp does not contain any bare C++ kernel or helper definition")
    replacement_names: set[str] = set()
    for unit in replacements:
        if unit.name in replacement_names:
            raise ValueError(f"Candidate C++ unit {unit.name!r} is defined more than once")
        replacement_names.add(unit.name)

    _, units = _split_cpp_units(cpp_source)
    all_existing_names = {u.name for u in units if u.name}
    if active_kernel is not None:
        editable = _scoped_editable_units(
            cpp_source,
            active_kernel,
            allow_exclusive_helpers=True,
        )
    else:
        editable = [u for u in units if u.kind in ("kernel", "helper") and u.name]
    editable_by_name = {u.name: u for u in editable}

    targets: list[tuple[int, int, str]] = []
    replaced_units: list[_CppUnit] = []
    new_helpers: list[_CppUnit] = []
    for unit in replacements:
        target = editable_by_name.get(unit.name)
        if target is not None:
            _reject_introduced_forbidden_tokens(target.text, unit.text)
            targets.append((target.start_char, target.end_char, unit.text))
            replaced_units.append(unit)
            continue
        if unit.kind == "helper" and unit.name not in all_existing_names:
            new_helpers.append(unit)
            continue
        if active_kernel is not None:
            raise ValueError(
                f"Candidate C++ unit {unit.name!r} is outside the editable scope "
                f"of active kernel {active_kernel!r}"
            )
        raise ValueError(f"Candidate C++ unit {unit.name!r} not found in C++ source")

    if new_helpers:
        new_helpers_by_name = {h.name: h for h in new_helpers}
        reached_new: set[str] = set()
        for seed_unit in replaced_units:
            reached_new |= _helper_call_closure(seed_unit, new_helpers_by_name)
        unreached = [h.name for h in new_helpers if h.name not in reached_new]
        if not targets or unreached:
            bad_name = unreached[0] if unreached else new_helpers[0].name
            if active_kernel is not None:
                raise ValueError(
                    f"Candidate C++ unit {bad_name!r} is outside the editable scope "
                    f"of active kernel {active_kernel!r}"
                )
            raise ValueError(f"Candidate C++ unit {bad_name!r} not found in C++ source")
        for helper_unit in new_helpers:
            _reject_introduced_forbidden_tokens("", helper_unit.text)

    updated = cpp_source
    for start_char, end_char, replacement_text in sorted(targets, reverse=True):
        updated = updated[:start_char] + replacement_text + updated[end_char:]
    if new_helpers:
        min_start = min(start for start, _, _ in targets)
        helper_blob = "\n\n".join(h.text.strip() for h in new_helpers)
        prefix = "" if (min_start == 0 or updated[min_start - 1] == "\n") else "\n"
        updated = updated[:min_start] + prefix + helper_blob + "\n\n" + updated[min_start:]
    if _is_normalized_identical(updated, cpp_source):
        raise NoopIdenticalPatchError(
            "Spliced C++ units are identical to parent Track-B translation unit"
        )
    return updated, len(targets) + len(new_helpers)


splice_cpp_units_by_symbol = splice_cpp_units_into_tu


def escalate_primfunc_to_trackb_tu(prim_func: object) -> str:
    """Lower an incumbent TileLang @T.prim_func into an operator-specific C++20 TU."""
    import tilelang  # noqa: PLC0415
    from tilelang import tvm  # noqa: PLC0415
    from tilelang.backend.module import create_backend_context  # noqa: PLC0415

    target_ctx = (
        create_backend_context("ascend").target
        if tvm.target.Target.current(allow_none=True) is None
        else contextlib.nullcontext()
    )
    with target_ctx:
        lowered = tilelang.lower(prim_func, target="ascend")
    kernel_src = str(getattr(lowered, "kernel_source", "") or "").strip()
    if not kernel_src:
        raise RuntimeError("tilelang.lower returned empty kernel_source for Track-B escalation")
    return kernel_src


_TRACKB_CPP_QUALIFIERS: tuple[str, ...] = (
    'extern "C" __global__',
    "__global__",
    "__aicore__",
    "__simd_vf__",
    "__simt_vf__",
)


def _py_char_offset(lines: Sequence[str], lineno: int, col_offset: int) -> int:
    prefix = sum(len(lines[idx]) for idx in range(lineno - 1))
    line_bytes = lines[lineno - 1].encode("utf-8")
    return prefix + len(line_bytes[:col_offset].decode("utf-8"))


def _format_cpp_string_literal(cpp_text: str) -> str:
    if '"""' not in cpp_text and not cpp_text.endswith('"'):
        return f'r"""{cpp_text}"""'
    return repr(cpp_text)


def _eval_static_str_expr(expr: ast.AST, str_consts: Mapping[str, str]) -> str | None:
    if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
        return expr.value
    if isinstance(expr, ast.Name) and expr.id in str_consts:
        return str_consts[expr.id]
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
        left_val = _eval_static_str_expr(expr.left, str_consts)
        right_val = _eval_static_str_expr(expr.right, str_consts)
        if left_val is not None and right_val is not None:
            return left_val + right_val
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
        str_consts: dict[str, str] = {}
        for node in tree.body:
            if (
                isinstance(node, ast.Assign)
                and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                str_consts[node.targets[0].id] = node.value.value
            elif (
                isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            ):
                str_consts[node.target.id] = node.value.value
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "_render_trackb_tu":
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Return) and sub.value is not None:
                        if isinstance(sub.value, ast.BinOp) and isinstance(sub.value.op, ast.Add):
                            left_val = _eval_static_str_expr(sub.value.left, str_consts)
                            if left_val is not None:
                                return left_val.count("\n")
                            left_src = ast.get_source_segment(candidate_src, sub.value.left) or ""
                            if left_src:
                                return left_src.count("\\n")
    return 0


def _find_python_cpp_literal_spans(
    source: str,
    *,
    require_cpp_tokens: bool = True,
) -> list[tuple[str, str, int, int]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    lines = source.splitlines(keepends=True)
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
            name is not None
            and isinstance(val, ast.Constant)
            and isinstance(val.value, str)
            and val.end_lineno is not None
            and val.end_col_offset is not None
        ):
            if require_cpp_tokens and not any(
                tok in val.value
                for tok in ("__global__", "__aicore__", "__simd_vf__", "__simt_vf__")
            ):
                continue
            start_pos = _py_char_offset(lines, val.lineno, val.col_offset)
            end_pos = _py_char_offset(lines, val.end_lineno, val.end_col_offset)
            hits.append((name, val.value, start_pos, end_pos))
    return hits


def _scope_trackb_python_or_cpp(
    source: str,
    diag: str,
    *,
    parent_source: str = "",
    prologue_line_offset: int = 0,
    focus_lines: Sequence[int] = (),
    active_kernel: str | None = None,
) -> str:
    """Scope Track-B C++ or hybrid Python source using lossless active-symbol closure."""
    diag_focus = extract_cpp_diagnostic_lines(diag)
    merged_focus = tuple(sorted({*diag_focus, *(int(ln) for ln in focus_lines if int(ln) > 0)}))
    try:
        ast.parse(source)
    except SyntaxError:
        if any(tok in source for tok in _TRACKB_CPP_QUALIFIERS):
            return build_trackb_context_view(
                source,
                active_kernel=active_kernel,
                focus_lines=merged_focus,
                diagnostic_text=diag,
                include_line_numbers=False,
            )
        return source

    hits = _find_python_cpp_literal_spans(source, require_cpp_tokens=True)
    if not hits:
        return source

    prologue_offset = (
        int(prologue_line_offset)
        if prologue_line_offset > 0
        else _detect_trackb_prologue_line_offset(source, parent_source=parent_source)
    )
    adjusted_focus = (
        tuple(ln - prologue_offset for ln in merged_focus if ln > prologue_offset)
        if prologue_offset > 0
        else merged_focus
    )
    diag_for_view = "" if prologue_offset > 0 else diag
    replacements: list[tuple[int, int, str]] = []
    for sym_name, cpp_text, start_pos, end_pos in hits:
        view = build_trackb_context_view(
            cpp_text,
            active_kernel=active_kernel,
            module_source=source,
            trackb_symbol=sym_name,
            focus_lines=adjusted_focus,
            diagnostic_text=diag_for_view,
            include_line_numbers=False,
        )
        wrapped = view if view.startswith("\n") else f"\n{view}\n"
        replacements.append((start_pos, end_pos, _format_cpp_string_literal(wrapped)))

    updated = source
    for start_pos, end_pos, literal in reversed(replacements):
        updated = updated[:start_pos] + literal + updated[end_pos:]
    ast.parse(updated)
    return updated


def _looks_like_python_primfunc_slice(source: str) -> bool:
    """Return True when *source* is a Python @T.prim_func or def slice without Track-B C++."""
    stripped = source.strip()
    if not stripped:
        return False
    if parse_scoped_search_replace_blocks(stripped):
        return False
    if any(tok in stripped for tok in _TRACKB_CPP_QUALIFIERS):
        return False
    if "@T.prim_func" in stripped:
        return True
    if re.search(r"(?m)^\s*(?:async\s+)?def\s+[A-Za-z_]\w*\s*\(", stripped):
        return True
    try:
        tree = ast.parse(stripped)
    except SyntaxError:
        return False
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        for node in tree.body
    )


def compact_creative_source_slice(
    source: str,
    *,
    max_bytes: int | None = None,
    focus_lines: Sequence[int] = (),
    diagnostic_text: str = "",
    active_kernel: str | None = None,
    parent_source: str = "",
    prologue_line_offset: int = 0,
) -> str:
    """Scope Track-B C++ sources losslessly without hardcoded byte truncation."""
    _ = max_bytes
    if any(tok in source for tok in _TRACKB_CPP_QUALIFIERS):
        return _scope_trackb_python_or_cpp(
            source,
            diagnostic_text,
            parent_source=parent_source,
            prologue_line_offset=prologue_line_offset,
            focus_lines=focus_lines,
            active_kernel=active_kernel,
        )
    return source


def _score_name_for_regime(name: str, regime_tag: str) -> int:
    tag = regime_tag.strip().lower()
    if not tag:
        return 0
    low = name.lower()
    if tag in low:
        return 100 + len(tag)
    long_tokens = [tok for tok in re.split(r"[^a-z0-9]+", tag) if len(tok) >= 3]
    word_tokens = [tok for tok in re.split(r"[^a-z0-9]+", tag) if len(tok) >= 2]
    name_words = set(re.split(r"[^a-z0-9]+", low))
    score = sum(10 + len(tok) for tok in long_tokens if tok in low)
    score += sum(3 for tok in word_tokens if tok in name_words)
    return score


def _py_node_source_slice(lines: Sequence[str], node: ast.stmt) -> str:
    decs = getattr(node, "decorator_list", ())
    start = min((d.lineno for d in decs), default=node.lineno)
    end = node.end_lineno or node.lineno
    return "".join(lines[start - 1 : end]).rstrip()


def _format_python_prototype(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    args_src = ast.unparse(node.args)
    ret_src = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
    return f"{prefix} {node.name}({args_src}){ret_src}: ..."


def _extract_python_diag_lines(diagnostic_text: str) -> tuple[int, ...]:
    lines = set(extract_cpp_diagnostic_lines(diagnostic_text))
    if diagnostic_text:
        for match in re.finditer(r":(\d+)(?::\d+)?\b", diagnostic_text):
            ln = int(match.group(1))
            if ln > 0:
                lines.add(ln)
    return tuple(sorted(lines))


def _resolve_active_kernel_from_diagnostics(
    units: Sequence[_CppUnit],
    diag_lines: Sequence[int],
    helpers_by_name: Mapping[str, _CppUnit],
) -> str | None:
    if not diag_lines:
        return None
    kernels = [u for u in units if u.kind == "kernel" and u.name]
    for ln in diag_lines:
        for kernel in kernels:
            if kernel.start_line <= ln <= kernel.end_line:
                return kernel.name
    for ln in diag_lines:
        for helper in helpers_by_name.values():
            if helper.start_line <= ln <= helper.end_line and helper.name:
                for kernel in kernels:
                    if helper.name in _helper_call_closure(kernel, helpers_by_name):
                        return kernel.name
    return None


def _extract_cpp_closure_parts(
    cpp_source: str,
    *,
    active_kernel: str | None = None,
    regime_tag: str = "",
    module_source: str = "",
    diagnostic_lines: Sequence[int] = (),
) -> tuple[str, tuple[str, ...], str]:
    clean_cpp = strip_markdown_fence_lines(cpp_source)
    _, units = _split_cpp_units(clean_cpp)
    kernels = [u for u in units if u.kind == "kernel" and u.name]
    helpers = [u for u in units if u.kind == "helper"]
    helpers_by_name = {h.name: h for h in helpers if h.name}
    host_launchers, _ = _host_launcher_units(units)
    host_launcher_starts = {u.start_char for u in host_launchers}

    target_name = active_kernel
    if target_name is None:
        target_name = _resolve_active_kernel_from_diagnostics(
            units, diagnostic_lines, helpers_by_name
        )
    if target_name is None:
        target_name = select_active_trackb_kernel(
            clean_cpp,
            regime_tag=regime_tag,
            module_source=module_source,
        )
    if target_name is None and kernels:
        target_name = kernels[0].name

    target_unit = next((u for u in kernels if u.name == target_name), None)
    if target_unit is None and target_name is not None:
        target_unit = next((u for u in units if u.name == target_name), None)

    directives = _extract_directive_lines(clean_cpp, units)
    if target_unit is None:
        device_parts = ["\n".join(text for _, text in directives)] if directives else []
        for unit in units:
            if unit.start_char in host_launcher_starts:
                continue
            txt = unit.text.strip()
            if txt:
                device_parts.append(txt)
        host_tail = "\n\n".join(u.text.strip() for u in host_launchers if u.text.strip())
        return "\n\n".join(p for p in device_parts if p).strip(), (), host_tail

    reached_names = _helper_call_closure(target_unit, helpers_by_name)
    entries: list[tuple[int, str, object]] = []
    for d_lineno, d_text in directives:
        entries.append((d_lineno, "directive", d_text))
    sibling_protos: list[str] = []
    for unit in units:
        if unit.start_char in host_launcher_starts:
            continue
        if unit.kind == "global_decl":
            entries.append((unit.start_line, "unit", unit))
        elif unit.kind == "helper":
            if unit.start_char == target_unit.start_char or (
                unit.name and unit.name in reached_names
            ):
                entries.append((unit.start_line, "unit", unit))
            else:
                sig = unit.signature.rstrip(";").strip()
                if sig:
                    sibling_protos.append(f"{sig};")
        elif unit.kind == "kernel":
            if unit.start_char == target_unit.start_char:
                entries.append((unit.start_line, "unit", unit))
            else:
                sig = unit.signature.rstrip(";").strip()
                decl = f"{sig};" if sig else f"void {unit.name}();"
                sibling_protos.append(decl)

    entries.sort(key=lambda item: item[0])
    device_lines: list[str] = []
    prev_kind = ""
    for _, kind, payload in entries:
        if kind == "directive":
            if device_lines and prev_kind != "directive":
                device_lines.append("")
            assert isinstance(payload, str)
            device_lines.append(payload)
        else:
            assert isinstance(payload, _CppUnit)
            if device_lines:
                device_lines.append("")
            device_lines.append(payload.text.strip())
        prev_kind = kind

    host_tail = "\n\n".join(u.text.strip() for u in host_launchers if u.text.strip())
    return "\n".join(device_lines).strip(), tuple(sibling_protos), host_tail


def _is_dead_tracka_builder(name: str, seg: str) -> bool:
    if "@T.prim_func" in seg:
        return True
    if name.startswith("_build_") and "trackb" not in name.lower():
        if not any(
            tok in seg
            for tok in ("compile_ascend", "_render_", 'extern "C"', "ctypes", "<<<", "TilingData")
        ):
            return True
    return False


def _is_host_abi_seed(name: str, seg: str) -> bool:
    if name == "_render_trackb_tu" or name.startswith("_render_"):
        return True
    if "trackb" in name.lower() or "workspace" in name.lower():
        return True
    if name.startswith(("_select_", "select_")):
        return True
    return any(
        tok in seg
        for tok in ('extern "C"', "<<<", "TilingData", "block_dim", "compile_ascend", "ctypes")
    )


def _extract_hybrid_trackb_repair_closure(
    module_source: str,
    tree: ast.Module,
    cpp_spans: Sequence[tuple[str, str, int, int]],
    *,
    active_symbols: Sequence[str],
    regime_tag: str,
    diagnostic_text: str,
    protected_exports: frozenset[str],
) -> RepairClosureSlice:
    lines = module_source.splitlines(keepends=True)
    funcs: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    assigns: dict[str, ast.stmt] = {}
    ordered_defs: list[tuple[str, str]] = []
    cpp_names = {name for name, _, _, _ in cpp_spans}

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs[node.name] = node
            ordered_defs.append(("func", node.name))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            for bname in _extract_top_level_bound_names(node):
                assigns[bname] = node
                ordered_defs.append(("assign", bname))

    active_sym_set = set(active_symbols)
    picked_span = next((s for s in cpp_spans if s[0] in active_sym_set), None)
    if picked_span is None and regime_tag:
        picked_span = max(
            cpp_spans,
            key=lambda s: _score_name_for_regime(s[0], regime_tag),
        )
    if picked_span is None:
        picked_span = cpp_spans[0]
    active_cpp_name, active_cpp_text, _, _ = picked_span

    prologue_offset = _detect_trackb_prologue_line_offset(module_source)
    raw_diag_lines = extract_cpp_diagnostic_lines(diagnostic_text)
    adj_diag_lines = (
        tuple(ln - prologue_offset for ln in raw_diag_lines if ln > prologue_offset)
        if prologue_offset > 0
        else raw_diag_lines
    )
    explicit_kernel = next(
        (sym for sym in active_symbols if sym not in cpp_names and sym not in funcs),
        None,
    )
    editable_cpp, cpp_protos, cpp_host_tail = _extract_cpp_closure_parts(
        active_cpp_text,
        active_kernel=explicit_kernel,
        regime_tag=regime_tag,
        module_source=module_source,
        diagnostic_lines=adj_diag_lines,
    )

    fn_segments = {name: _py_node_source_slice(lines, node) for name, node in funcs.items()}
    dead_tracka = {
        name for name, seg in fn_segments.items() if _is_dead_tracka_builder(name, seg)
    }

    abi_seeds: list[str] = []
    selector_names: list[str] = [
        name for name in funcs if name.startswith(("_select_", "select_")) and name not in dead_tracka
    ]
    matched_selectors = (
        [name for name in selector_names if _score_name_for_regime(name, regime_tag) > 0]
        if regime_tag
        else []
    )
    kept_selectors = set(matched_selectors or selector_names)

    for name, seg in fn_segments.items():
        if name in dead_tracka:
            continue
        if name.startswith(("_select_", "select_")) and name not in kept_selectors:
            continue
        if name in active_sym_set or _is_host_abi_seed(name, seg):
            abi_seeds.append(name)

    abi_visited: set[str] = set()
    queue = list(abi_seeds)
    while queue:
        sym = queue.pop(0)
        if sym in abi_visited or sym in dead_tracka or sym in cpp_names:
            continue
        node = funcs.get(sym) or assigns.get(sym)
        if node is None:
            continue
        abi_visited.add(sym)
        for sub in ast.walk(node):
            if isinstance(sub, ast.Name):
                ref = sub.id
                if (
                    ref not in abi_visited
                    and ref not in dead_tracka
                    and ref not in cpp_names
                    and (ref in funcs or ref in assigns)
                ):
                    if ref.startswith(("_select_", "select_")) and ref not in kept_selectors:
                        continue
                    queue.append(ref)

    for kind, name in ordered_defs:
        if kind == "assign" and name in protected_exports and name not in abi_visited:
            abi_visited.add(name)

    abi_parts: list[str] = []
    seen_assign_nodes: set[int] = set()
    for kind, name in ordered_defs:
        if name not in abi_visited:
            continue
        if kind == "func":
            abi_parts.append(fn_segments[name])
        else:
            stmt = assigns[name]
            if id(stmt) not in seen_assign_nodes:
                seen_assign_nodes.add(id(stmt))
                abi_parts.append(_py_node_source_slice(lines, stmt))

    if cpp_host_tail:
        abi_parts.append(cpp_host_tail)

    extra_protos: list[str] = list(cpp_protos)
    for cname, ctext, _, _ in cpp_spans:
        if cname == active_cpp_name:
            continue
        _, c_units = _split_cpp_units(ctext)
        for unit in c_units:
            if unit.kind == "kernel" and unit.name:
                sig = unit.signature.rstrip(";").strip()
                decl = f"{sig};" if sig else f"void {unit.name}();"
                if decl not in extra_protos:
                    extra_protos.append(decl)

    return RepairClosureSlice(
        editable_device_source=editable_cpp,
        sibling_prototypes=tuple(extra_protos),
        readonly_host_abi_tail="\n\n".join(p for p in abi_parts if p).strip(),
        diagnostic_excerpt=diagnostic_text.strip(),
    )


def _extract_tracka_repair_closure(
    module_source: str,
    *,
    active_symbols: Sequence[str],
    regime_tag: str,
    diagnostic_text: str,
    protected_exports: frozenset[str],
) -> RepairClosureSlice:
    editable_raw, readonly_tail = split_tracka_editable_and_readonly_tail(
        module_source, protected_exports
    )
    if not editable_raw.strip():
        return RepairClosureSlice(
            editable_device_source="",
            sibling_prototypes=(),
            readonly_host_abi_tail=readonly_tail,
            diagnostic_excerpt=diagnostic_text.strip(),
        )
    try:
        tree = ast.parse(editable_raw)
    except SyntaxError:
        return RepairClosureSlice(
            editable_device_source=editable_raw.strip(),
            sibling_prototypes=(),
            readonly_host_abi_tail=readonly_tail,
            diagnostic_excerpt=diagnostic_text.strip(),
        )

    lines = editable_raw.splitlines(keepends=True)
    imports: list[tuple[set[str], str]] = []
    funcs: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    classes: dict[str, ast.ClassDef] = {}
    assigns: dict[str, ast.stmt] = {}
    ordered_defs: list[tuple[str, str]] = []

    for node in tree.body:
        if isinstance(node, ast.Import):
            bound = {alias.asname or alias.name.split(".")[0] for alias in node.names}
            imports.append((bound, _py_node_source_slice(lines, node)))
        elif isinstance(node, ast.ImportFrom):
            bound = {alias.asname or alias.name for alias in node.names}
            imports.append((bound, _py_node_source_slice(lines, node)))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs[node.name] = node
            ordered_defs.append(("func", node.name))
        elif isinstance(node, ast.ClassDef):
            classes[node.name] = node
            ordered_defs.append(("class", node.name))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            for bname in _extract_top_level_bound_names(node):
                assigns[bname] = node
                ordered_defs.append(("assign", bname))

    all_defs: dict[str, ast.stmt] = {**assigns, **classes, **funcs}
    fn_segments = {name: _py_node_source_slice(lines, node) for name, node in funcs.items()}
    builder_fns = {
        name
        for name, seg in fn_segments.items()
        if "@T.prim_func" in seg or name.startswith("_build_") or name in ("make_kernel", "main")
    }
    selector_fns = {name for name in funcs if name.startswith(("_select_", "select_"))}

    seeds: list[str] = [s for s in active_symbols if s in all_defs]
    diag_lines = _extract_python_diag_lines(diagnostic_text)
    if diag_lines:
        for name, fnode in funcs.items():
            end_ln = fnode.end_lineno or fnode.lineno
            if any(fnode.lineno <= ln <= end_ln for ln in diag_lines):
                if name not in seeds:
                    seeds.append(name)

    if not seeds and regime_tag:
        scored_selectors = [
            (_score_name_for_regime(n, regime_tag), n)
            for n in selector_fns
            if _score_name_for_regime(n, regime_tag) > 0
        ]
        scored_builders = [
            (_score_name_for_regime(n, regime_tag), n)
            for n in builder_fns
            if _score_name_for_regime(n, regime_tag) > 0
        ]
        if scored_selectors:
            top_s = max(sc for sc, _ in scored_selectors)
            seeds.extend(n for sc, n in scored_selectors if sc == top_s and n not in seeds)
        if scored_builders:
            top_b = max(sc for sc, _ in scored_builders)
            seeds.extend(n for sc, n in scored_builders if sc == top_b and n not in seeds)

    if seeds and not any(s in selector_fns for s in seeds):
        if len(selector_fns) == 1:
            seeds.extend(selector_fns)
        elif regime_tag:
            for sname in selector_fns:
                if _score_name_for_regime(sname, regime_tag) > 0 and sname not in seeds:
                    seeds.append(sname)

    if not seeds:
        if builder_fns or selector_fns:
            seeds.extend(sorted(selector_fns | builder_fns))
        else:
            seeds.extend(ordered_name for _, ordered_name in ordered_defs)

    sibling_exclusions = (builder_fns | selector_fns) - set(seeds)
    visited: set[str] = set()
    queue = list(seeds)
    while queue:
        sym = queue.pop(0)
        if sym in visited or sym not in all_defs:
            continue
        visited.add(sym)
        for sub in ast.walk(all_defs[sym]):
            if isinstance(sub, ast.Name):
                ref = sub.id
                if (
                    ref in all_defs
                    and ref not in visited
                    and ref not in sibling_exclusions
                ):
                    queue.append(ref)

    used_names: set[str] = set(visited)
    for sym in visited:
        used_names.update(sub.id for sub in ast.walk(all_defs[sym]) if isinstance(sub, ast.Name))

    parts: list[str] = []
    for bound, imp_src in imports:
        if bound & used_names or "*" in bound:
            parts.append(imp_src)

    seen_assigns: set[int] = set()
    for kind, name in ordered_defs:
        if name not in visited:
            continue
        if kind == "assign":
            stmt = assigns[name]
            if id(stmt) not in seen_assigns:
                seen_assigns.add(id(stmt))
                parts.append(_py_node_source_slice(lines, stmt))
        else:
            parts.append(_py_node_source_slice(lines, all_defs[name]))

    sibling_protos = tuple(
        _format_python_prototype(funcs[name])
        for kind, name in ordered_defs
        if kind == "func" and name not in visited
    )

    return RepairClosureSlice(
        editable_device_source="\n\n".join(p for p in parts if p.strip()).strip(),
        sibling_prototypes=sibling_protos,
        readonly_host_abi_tail=readonly_tail,
        diagnostic_excerpt=diagnostic_text.strip(),
    )


def extract_repair_closure_slice(
    module_source: str,
    *,
    active_symbols: Sequence[str] = (),
    regime_tag: str = "",
    backend_kind: str = "tilelang_dsl",
    diagnostic_text: str = "",
    protected_exports: frozenset[str] | None = None,
) -> RepairClosureSlice:
    """Extract lossless reachable call-graph closure and host ABI slice without byte caps."""
    if not module_source.strip():
        return RepairClosureSlice(
            editable_device_source="",
            sibling_prototypes=(),
            readonly_host_abi_tail="",
            diagnostic_excerpt=diagnostic_text.strip(),
        )
    eff_protected = DEFAULT_PROTECTED_EXPORTS if protected_exports is None else protected_exports
    is_trackb_tier = str(backend_kind or "").strip().lower() in (
        "ascendc_trackb",
        "hybrid_trackb",
        "ascendc_template",
        "cce_cpp20",
        "cce_trackb",
    )
    parsed_tree: ast.Module | None = None
    try:
        parsed_tree = ast.parse(module_source)
    except SyntaxError:
        parsed_tree = None

    if parsed_tree is not None:
        cpp_spans = _find_python_cpp_literal_spans(module_source, require_cpp_tokens=True)
        if cpp_spans and (is_trackb_tier or not _looks_like_python_primfunc_slice(module_source)):
            return _extract_hybrid_trackb_repair_closure(
                module_source,
                parsed_tree,
                cpp_spans,
                active_symbols=active_symbols,
                regime_tag=regime_tag,
                diagnostic_text=diagnostic_text,
                protected_exports=eff_protected,
            )
        return _extract_tracka_repair_closure(
            module_source,
            active_symbols=active_symbols,
            regime_tag=regime_tag,
            diagnostic_text=diagnostic_text,
            protected_exports=eff_protected,
        )

    if any(tok in module_source for tok in _TRACKB_CPP_QUALIFIERS) and (
        is_trackb_tier or not _looks_like_python_primfunc_slice(module_source)
    ):
        diag_lines = extract_cpp_diagnostic_lines(diagnostic_text)
        explicit_kernel = next((sym for sym in active_symbols if sym), None)
        editable_cpp, sibling_protos, host_tail = _extract_cpp_closure_parts(
            module_source,
            active_kernel=explicit_kernel,
            regime_tag=regime_tag,
            module_source=module_source,
            diagnostic_lines=diag_lines,
        )
        return RepairClosureSlice(
            editable_device_source=editable_cpp,
            sibling_prototypes=sibling_protos,
            readonly_host_abi_tail=host_tail,
            diagnostic_excerpt=diagnostic_text.strip(),
        )

    return _extract_tracka_repair_closure(
        module_source,
        active_symbols=active_symbols,
        regime_tag=regime_tag,
        diagnostic_text=diagnostic_text,
        protected_exports=eff_protected,
    )


__all__ = [
    "AmbiguousPatchAnchorError",
    "DEFAULT_PROTECTED_EXPORTS",
    "NoopIdenticalPatchError",
    "PatchAnchorMissError",
    "RepairClosureSlice",
    "TRACKB_REPLY_CONTRACT_PROMPT",
    "_TRACKB_CPP_QUALIFIERS",
    "_detect_trackb_prologue_line_offset",
    "_find_python_cpp_literal_spans",
    "_looks_like_python_primfunc_slice",
    "apply_scoped_trackb_patch",
    "build_trackb_context_view",
    "classify_stage_failure_for_repair",
    "compact_creative_source_slice",
    "escalate_primfunc_to_trackb_tu",
    "extract_cpp_diagnostic_lines",
    "extract_reachable_trackb_source",
    "extract_repair_closure_slice",
    "extract_scoped_trackb_kernel_view",
    "extract_trackb_launcher_preview",
    "format_trackb_reply_contract",
    "parse_scoped_search_replace_blocks",
    "render_active_symbol_closure_view",
    "render_active_symbol_closure_with_line_map",
    "resolve_instantiated_trackb_kernel",
    "select_active_trackb_kernel",
    "splice_cpp_units_by_symbol",
    "splice_cpp_units_into_tu",
    "split_tracka_editable_and_readonly_tail",
    "strip_cpp_comments_and_strings",
    "strip_markdown_fence_lines",
]
