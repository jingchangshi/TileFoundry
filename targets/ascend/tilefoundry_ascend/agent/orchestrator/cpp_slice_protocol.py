"""Scoped C++20 slicing, patching, and Track-A-to-Track-B escalation protocol."""

from __future__ import annotations

import ast
import contextlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from tilefoundry_ascend.ascend.target.dialect_surface import CCE_CPP20_SURFACE

_SEARCH_REPLACE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"<<<<<<<\s*SEARCH\s*\n(.*?)\n=======\s*\n(.*?)\n>>>>>>>\s*REPLACE",
        re.DOTALL,
    ),
    re.compile(
        r"<<<\s*SEARCH\s*\n(.*?)\n===\s*\n(.*?)\n>>>\s*REPLACE",
        re.DOTALL,
    ),
)

_UNIT_START_RE = re.compile(
    r"^\s*(?:template\s*<|__(?:global|aicore|simd_vf|simt_vf|simd_callee|simt_callee)__\b|"
    r"extern\s+\"C\"|struct\s+\w+|(?:static\s+)?constexpr\b|using\s+\w+|"
    r"(?:static\s+)?(?:inline\s+|"
    r"__(?:global|aicore|simd_vf|simt_vf|simd_callee|simt_callee)__\b))"
)
_FUNC_NAME_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_CALL_NAME_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*(?:<[^;{}()]*>)?\s*\(")
_TEMPLATE_CALL_ARGS_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\s*<([^;{}()]+)>\s*\(")
_IDENT_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\b")
_C_CONTROL_KEYWORDS = frozenset({"if", "for", "while", "switch", "return", "sizeof", "alignof"})
_SIBLING_MARKER_RE = re.compile(r"/\*\s*\[READ-ONLY SIBLING KERNEL:")
_LAUNCH_CONFIG_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*(?:<[^;{}]*>)?\s*<<<")
_MARKDOWN_FENCE_LINE_RE = re.compile(r"^\s*```[A-Za-z0-9_+.-]*\s*$")
_DEVICE_QUALIFIER_RE = re.compile(
    r"__(?:global|aicore|simd_vf|simt_vf|simd_callee|simt_callee)__\b"
)
_DIAG_FILE_LINE_RE = re.compile(
    r"(?:\.(?:cce|cpp|cc|cu|h)|<source>|input):(\d+)(?::\d+)?:\s*(?:fatal\s+)?(?:error|warning):"
)
_DIAG_WORD_LINE_RE = re.compile(r"\bline\s+(\d+)\b")


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


def strip_markdown_fence_lines(text: str) -> str:
    """Strip leading, trailing, or embedded Markdown code fence lines from LLM C++ output."""
    if "```" not in text:
        return text
    kept = [
        line
        for line in text.splitlines(keepends=True)
        if not _MARKDOWN_FENCE_LINE_RE.match(line.rstrip("\r\n"))
    ]
    return "".join(kept).strip("\n") + ("\n" if text.endswith("\n") and kept else "")


@dataclass(frozen=True)
class _CppUnit:
    """Top-level parsed C++20 declaration or function block."""

    kind: str
    name: str
    signature: str
    text: str
    start_char: int
    end_char: int
    line_count: int
    start_line: int = 1
    end_line: int = 1


def strip_cpp_comments_and_strings(text: str) -> str:
    """Blank C++ comments and string/char literals while preserving byte offsets and newlines."""
    chars = list(text)
    n = len(chars)
    i = 0
    while i < n:
        ch = chars[i]
        nxt = chars[i + 1] if i + 1 < n else ""
        if ch == "/" and nxt == "/":
            while i < n and chars[i] != "\n":
                chars[i] = " "
                i += 1
        elif ch == "/" and nxt == "*":
            chars[i] = " "
            chars[i + 1] = " "
            i += 2
            while i + 1 < n and not (chars[i] == "*" and chars[i + 1] == "/"):
                if chars[i] != "\n":
                    chars[i] = " "
                i += 1
            if i + 1 < n:
                chars[i] = " "
                chars[i + 1] = " "
                i += 2
        elif ch in ('"', "'"):
            quote = ch
            chars[i] = " "
            i += 1
            while i < n:
                cur = chars[i]
                if cur != "\n":
                    chars[i] = " "
                if cur == "\\" and i + 1 < n:
                    if chars[i + 1] != "\n":
                        chars[i + 1] = " "
                    i += 2
                    continue
                i += 1
                if cur == quote:
                    break
        else:
            i += 1
    return "".join(chars)


def _extract_unit_name(header_text: str, *, is_struct: bool = False) -> str:
    if is_struct:
        m_st = re.search(r"\b(?:struct|using)\s+([A-Za-z_][A-Za-z0-9_]*)", header_text)
        if m_st:
            return m_st.group(1)
    names = [m.group(1) for m in _FUNC_NAME_RE.finditer(header_text)]
    for cand in reversed(names):
        if cand not in _C_CONTROL_KEYWORDS:
            return cand
    return ""


def _looks_like_cpp_def_start(masked_lines: Sequence[str], idx: int) -> bool:
    """Return True when a top-level statement opens a brace before terminating with ';'."""
    if not masked_lines[idx].strip():
        return False
    for k in range(idx, len(masked_lines)):
        mln = masked_lines[k]
        if mln.lstrip().startswith("#"):
            return False
        brace_pos = mln.find("{")
        semi_pos = mln.find(";")
        if brace_pos >= 0 and (semi_pos < 0 or brace_pos < semi_pos):
            return True
        if semi_pos >= 0:
            return False
    return False


def _split_cpp_units(cpp_source: str) -> tuple[list[str], list[_CppUnit]]:
    lines = cpp_source.splitlines(keepends=True)
    masked_lines = strip_cpp_comments_and_strings(cpp_source).splitlines(keepends=True)
    line_offsets: list[int] = []
    pos = 0
    for ln in lines:
        line_offsets.append(pos)
        pos += len(ln)

    header_directives: list[str] = []
    units: list[_CppUnit] = []
    i = 0
    total = len(lines)
    while i < total:
        raw_ln = lines[i]
        stripped = raw_ln.strip()
        if not stripped:
            i += 1
            continue
        if stripped.startswith("#"):
            header_directives.append(raw_ln.rstrip("\r\n"))
            i += 1
            continue
        if (
            _UNIT_START_RE.match(stripped)
            or "{" in masked_lines[i]
            or _looks_like_cpp_def_start(masked_lines, i)
        ):
            start_i = i
            start_char = line_offsets[start_i]
            depth = 0
            seen_brace = False
            j = i
            while j < total:
                mln = masked_lines[j]
                depth += mln.count("{") - mln.count("}")
                if "{" in mln:
                    seen_brace = True
                if seen_brace and depth <= 0:
                    break
                if not seen_brace and lines[j].rstrip().endswith(";"):
                    break
                j += 1
            if j >= total:
                j = total - 1
            end_char = line_offsets[j] + len(lines[j])
            block_text = cpp_source[start_char:end_char]
            sig_head = block_text.split("{", 1)[0].strip()
            is_struct = bool(
                re.match(
                    r"^(?:template\s*<[^>]*>\s*)?(?:struct|enum|using|typedef)\b",
                    sig_head,
                )
                or (
                    re.match(r"^(?:static\s+)?constexpr\b", sig_head)
                    and ("(" not in sig_head or "=" in sig_head.split("(", 1)[0])
                )
            )
            if is_struct or (not seen_brace and block_text.rstrip().endswith(";")):
                kind = "global_decl"
            elif "__global__" in sig_head:
                kind = "kernel"
            else:
                kind = "helper"
            name = _extract_unit_name(sig_head, is_struct=is_struct)
            sig_clean = " ".join(sig_head.splitlines()).strip()
            units.append(
                _CppUnit(
                    kind=kind,
                    name=name,
                    signature=sig_clean,
                    text=block_text,
                    start_char=start_char,
                    end_char=end_char,
                    line_count=(j - start_i + 1),
                    start_line=start_i + 1,
                    end_line=j + 1,
                )
            )
            i = j + 1
            continue
        if stripped.endswith(";"):
            start_char = line_offsets[i]
            end_char = start_char + len(raw_ln)
            units.append(
                _CppUnit(
                    kind="global_decl",
                    name=_extract_unit_name(stripped, is_struct=True),
                    signature=stripped.rstrip(";").strip(),
                    text=raw_ln,
                    start_char=start_char,
                    end_char=end_char,
                    line_count=1,
                    start_line=i + 1,
                    end_line=i + 1,
                )
            )
        i += 1
    return header_directives, units


def extract_trackb_launcher_preview(module_source: str, *, max_bytes: int = 1600) -> str:
    """Extract compact C++ extern-C launcher stubs and host binding lines from *module_source*."""
    if not module_source.strip():
        return ""
    previews: list[str] = []
    for match in re.finditer(
        r'extern\s+"C"\s+[\s\S]{0,1200}?\n\}',
        module_source,
    ):
        snippet = match.group(0).strip()
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

    joined = "\n\n".join(previews).strip()
    if len(joined.encode("utf-8")) > max_bytes:
        joined = joined.encode("utf-8")[: max_bytes - 32].decode("utf-8", errors="ignore")
        joined = joined.rstrip() + "\n/* ... launcher preview truncated ... */"
    return joined


def _host_launcher_units(
    units: Sequence[_CppUnit],
    *,
    outside_cpp: str = "",
) -> tuple[list[_CppUnit], set[str]]:
    """Collect non-device host launcher units and kernel names they launch via <<<...>>>."""
    host_units = [
        u
        for u in units
        if u.kind == "helper"
        and not _DEVICE_QUALIFIER_RE.search(u.signature)
        and "<<<" in u.text
    ]
    prioritized: list[_CppUnit] = []
    fallback: list[_CppUnit] = []
    for unit in host_units:
        if outside_cpp and unit.name and re.search(rf"\b{re.escape(unit.name)}\b", outside_cpp):
            prioritized.append(unit)
        else:
            fallback.append(unit)
    ordered = prioritized if prioritized else fallback
    launched: set[str] = set()
    for unit in ordered:
        masked = strip_cpp_comments_and_strings(unit.text)
        for m in _LAUNCH_CONFIG_RE.finditer(masked):
            launched.add(m.group(1))
    return ordered, launched


def select_active_trackb_kernel(
    cpp_source: str,
    *,
    active_kernel: str | None = None,
    regime_tag: str | None = "",
    module_source: str | None = "",
) -> str | None:
    """Resolve the active ``__global__`` kernel name for scoped Track-B context extraction."""
    clean_cpp = strip_markdown_fence_lines(cpp_source)
    _, units = _split_cpp_units(clean_cpp)
    kernels = [u for u in units if u.kind == "kernel" and u.name]
    if not kernels:
        return None
    if active_kernel:
        for unit in kernels:
            if unit.name == active_kernel:
                return unit.name
        return None
    mod_src = str(module_source or "")
    outside_cpp = mod_src.replace(cpp_source, "").replace(clean_cpp, "") if mod_src else ""
    _, cpp_launched_names = _host_launcher_units(units, outside_cpp=outside_cpp)
    outside_launched_names = (
        {
            unit.name
            for unit in kernels
            if re.search(rf"\b{re.escape(unit.name)}\s*(?:<[^>]*>)?\s*<<<", outside_cpp)
        }
        if outside_cpp
        else set()
    )
    launched_names = outside_launched_names or cpp_launched_names
    tag = str(regime_tag or "").lower()
    if tag:
        tokens = [tok for tok in re.split(r"[^a-z0-9]+", tag) if len(tok) >= 3]
        scored = sorted(
            kernels,
            key=lambda u: (
                -sum(1 for t in tokens if t in u.name.lower()),
                -(1 if u.name in launched_names else 0),
                -u.line_count,
            ),
        )
        if scored and any(t in scored[0].name.lower() for t in tokens):
            return scored[0].name
    if outside_launched_names:
        launched = [u for u in kernels if u.name in outside_launched_names]
        return max(launched, key=lambda u: u.line_count).name
    if cpp_launched_names:
        launched = [u for u in kernels if u.name in cpp_launched_names]
        if launched:
            return max(launched, key=lambda u: u.line_count).name
    if outside_cpp:
        called = [
            unit
            for unit in kernels
            if f"{unit.name}<" in outside_cpp or f"{unit.name}(" in outside_cpp
        ]
        if called:
            return max(called, key=lambda u: u.line_count).name
    return kernels[0].name


def extract_reachable_trackb_source(
    cpp_source: str,
    *,
    module_source: str | None = None,
    regime_tag: str | None = None,
) -> tuple[str, str]:
    """Extract the launched entry kernel and its static reachable C++20 subgraph."""
    clean_cpp = strip_markdown_fence_lines(cpp_source)
    directives, units = _split_cpp_units(clean_cpp)
    entry_kernel = select_active_trackb_kernel(
        clean_cpp,
        regime_tag=regime_tag or "",
        module_source=module_source or "",
    )
    if entry_kernel is None:
        return "", clean_cpp

    entry_unit = next((u for u in units if u.kind == "kernel" and u.name == entry_kernel), None)
    if entry_unit is None:
        return "", clean_cpp

    callables_by_name = {
        u.name: u for u in units if u.kind in ("kernel", "helper") and u.name
    }
    reached_names = _helper_call_closure(entry_unit, callables_by_name)
    reached_names.add(entry_kernel)

    host_launchers, _ = _host_launcher_units(units)
    entry_launch_re = re.compile(rf"\b{re.escape(entry_kernel)}\s*(?:<[^;{{}}]*>)?\s*<<<")
    kept_launcher_starts = {
        u.start_char
        for u in host_launchers
        if entry_launch_re.search(strip_cpp_comments_and_strings(u.text))
    }

    kept_units: list[_CppUnit] = []
    for unit in units:
        if unit.kind == "global_decl":
            kept_units.append(unit)
        elif unit.start_char == entry_unit.start_char:
            kept_units.append(unit)
        elif unit.kind in ("kernel", "helper") and unit.name in reached_names:
            kept_units.append(unit)
        elif unit.start_char in kept_launcher_starts:
            kept_units.append(unit)

    parts: list[str] = []
    if directives:
        parts.append("\n".join(directives))
    for unit in sorted(kept_units, key=lambda u: u.start_char):
        text = unit.text.strip()
        if text:
            parts.append(text)
    return entry_kernel, "\n\n".join(parts) + ("\n" if parts else "")


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


def _merge_line_spans(spans: Sequence[tuple[int, int]]) -> list[tuple[int, int]]:
    ordered = sorted((s, e) for s, e in spans if s < e)
    if not ordered:
        return []
    merged: list[tuple[int, int]] = [ordered[0]]
    for start, end in ordered[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _render_line_spans(
    lines: Sequence[str],
    spans: Sequence[tuple[int, int]],
    marker: str,
) -> str:
    if not spans:
        return ""
    chunks: list[str] = []
    last_idx = len(spans) - 1
    for idx, (start, end) in enumerate(spans):
        seg = "".join(lines[start:end])
        if idx < last_idx:
            chunks.append(seg.rstrip("\r\n"))
        else:
            chunks.append(seg)
    return marker.join(chunks)


def _truncate_kernel_body(
    kernel_text: str,
    max_bytes: int,
    *,
    unit_start_line: int = 1,
    focus_lines: Sequence[int] = (),
    focus_window: int = 25,
) -> str:
    """Truncate a C++ body to *max_bytes* while optionally preserving focused line windows."""
    if len(kernel_text.encode("utf-8")) <= max_bytes:
        return kernel_text
    lines = kernel_text.splitlines(keepends=True)
    if len(lines) <= 20:
        encoded = kernel_text.encode("utf-8")[: max(128, max_bytes - 48)]
        return encoded.decode("utf-8", errors="ignore").rstrip() + "\n}"
    middle_marker = (
        "\n    /* [MIDDLE BODY OMITTED TO FIT <15KB SCOPED VIEW; PATCH VISIBLE LINES] */\n"
    )
    budget = max(256, max_bytes - len(middle_marker.encode("utf-8")))
    head_budget = budget // 2
    tail_budget = budget - head_budget

    head_lines: list[str] = []
    used = 0
    for ln in lines:
        b = len(ln.encode("utf-8"))
        if used + b > head_budget and head_lines:
            break
        head_lines.append(ln)
        used += b

    tail_lines: list[str] = []
    used_tail = 0
    for ln in reversed(lines[len(head_lines) :]):
        b = len(ln.encode("utf-8"))
        if used_tail + b > tail_budget and tail_lines:
            break
        tail_lines.append(ln)
        used_tail += b
    tail_lines.reverse()

    rel_indices = sorted(
        {
            int(ln) - int(unit_start_line)
            for ln in focus_lines
            if 0 <= int(ln) - int(unit_start_line) < len(lines)
        }
    )
    omitted_start = len(head_lines)
    omitted_end = len(lines) - len(tail_lines)
    if not any(omitted_start <= rel < omitted_end for rel in rel_indices):
        return "".join(head_lines).rstrip() + middle_marker + "".join(tail_lines).lstrip("\n")

    focus_marker = "\n    /* [LINES OMITTED OUTSIDE FOCUS WINDOW; PATCH VISIBLE LINES] */\n"
    total_lines = len(lines)
    line_bytes = [len(ln.encode("utf-8")) for ln in lines]
    win = max(0, int(focus_window))

    def _pick_head(budget_limit: int, min_lines: int) -> int:
        end = 0
        acc = 0
        while end < total_lines:
            nb = acc + line_bytes[end]
            if end >= min_lines and nb > budget_limit:
                break
            if end > 0 and nb > int(max_bytes * 0.35):
                break
            acc = nb
            end += 1
        return max(1, end)

    def _pick_tail(budget_limit: int, min_lines: int, floor_idx: int) -> int:
        start = total_lines
        acc = 0
        while start > floor_idx:
            nb = acc + line_bytes[start - 1]
            if (total_lines - start) >= min_lines and nb > budget_limit:
                break
            if start < total_lines and nb > int(max_bytes * 0.25):
                break
            acc = nb
            start -= 1
        return min(total_lines - 1, start) if total_lines > floor_idx else floor_idx

    min_head = min(6, total_lines)
    head_end = _pick_head(int(max_bytes * 0.20), min_head)
    min_tail = min(4, max(0, total_lines - head_end))
    tail_start = _pick_tail(int(max_bytes * 0.15), min_tail, head_end)

    full_spans = _merge_line_spans(
        [
            (0, head_end),
            *((max(0, rel - win), min(total_lines, rel + win + 1)) for rel in rel_indices),
            (tail_start, total_lines),
        ]
    )
    rendered = _render_line_spans(lines, full_spans, focus_marker)
    if len(rendered.encode("utf-8")) <= max_bytes:
        return rendered

    head_end = min(head_end, min_head)
    tail_start = max(tail_start, total_lines - min(4, max(0, total_lines - head_end)))
    trimmed_spans = _merge_line_spans(
        [
            (0, head_end),
            *((max(0, rel - win), min(total_lines, rel + win + 1)) for rel in rel_indices),
            (tail_start, total_lines),
        ]
    )
    rendered = _render_line_spans(lines, trimmed_spans, focus_marker)
    if len(rendered.encode("utf-8")) <= max_bytes:
        return rendered

    best_rendered = ""
    for r in range(win, -1, -1):
        cand_spans = _merge_line_spans(
            [
                (0, head_end),
                *((max(0, rel - r), min(total_lines, rel + r + 1)) for rel in rel_indices),
                (tail_start, total_lines),
            ]
        )
        cand = _render_line_spans(lines, cand_spans, focus_marker)
        if len(cand.encode("utf-8")) <= max_bytes:
            return cand
        if r == 0:
            best_rendered = cand

    for h_lim, t_lim in ((2, 2), (1, 1)):
        cand_spans = _merge_line_spans(
            [
                (0, min(h_lim, total_lines)),
                *((rel, rel + 1) for rel in rel_indices),
                (max(total_lines - t_lim, min(h_lim, total_lines)), total_lines),
            ]
        )
        cand = _render_line_spans(lines, cand_spans, focus_marker)
        if len(cand.encode("utf-8")) <= max_bytes:
            return cand
        best_rendered = cand

    encoded = best_rendered.encode("utf-8")[: max(128, max_bytes - 48)]
    return encoded.decode("utf-8", errors="ignore").rstrip() + "\n}"


def build_trackb_context_view(
    cpp_source: str,
    *,
    active_kernel: str | None = None,
    regime_tag: str = "",
    module_source: str = "",
    trackb_symbol: str = "",
    launcher_preview: str = "",
    max_bytes: int = 14800,
    focus_lines: Sequence[int] = (),
    focus_window: int = 25,
    diagnostic_text: str = "",
) -> str:
    """Build a bounded (<15 KB) scoped C++20 context view for Track-B editing."""
    diag_lines = extract_cpp_diagnostic_lines(diagnostic_text) if diagnostic_text else ()
    resolved_focus = tuple(
        sorted({int(ln) for ln in (*focus_lines, *diag_lines) if int(ln) > 0})
    )
    directives, units = _split_cpp_units(cpp_source)
    target_name = select_active_trackb_kernel(
        cpp_source,
        active_kernel=active_kernel,
        regime_tag=regime_tag,
        module_source=module_source,
    )
    global_decls = [u.text.strip() for u in units if u.kind == "global_decl"]
    kernels = [u for u in units if u.kind == "kernel"]
    helpers = [u for u in units if u.kind == "helper"]

    target_unit = next((u for u in kernels if u.name == target_name), None)
    if target_unit is None and kernels:
        target_unit = kernels[0]
        target_name = target_unit.name

    preview = launcher_preview.strip() or extract_trackb_launcher_preview(module_source)

    prefix_parts: list[str] = []
    if trackb_symbol:
        prefix_parts.append(f"/* TRACK-B SCOPED SLICE: symbol={trackb_symbol} active={target_name} */")
    if directives:
        prefix_parts.append("\n".join(directives))
    if global_decls:
        prefix_parts.append("\n\n".join(global_decls))
    if preview:
        prefix_parts.append(f"/* LAUNCHER BINDING PREVIEW */\n{preview}")

    sibling_summaries: list[str] = []
    for k in kernels:
        if target_unit is not None and k.name == target_unit.name:
            continue
        sibling_summaries.append(
            f"/* [READ-ONLY SIBLING KERNEL: {k.name} ({k.line_count} lines; edits forbidden)] */\n"
            f"{k.signature};"
        )
    if sibling_summaries:
        prefix_parts.append("\n\n".join(sibling_summaries))

    helpers_by_name = {h.name: h for h in helpers if h.name}
    helper_calls: set[str] = set()
    if target_unit is not None:
        helper_calls = _helper_call_closure(target_unit, helpers_by_name)
    ref_helpers = [h for h in helpers if h.name and h.name in helper_calls]
    other_helpers = [h for h in helpers if h not in ref_helpers]

    helper_summaries = [
        f"/* [READ-ONLY HELPER] */ {h.signature};" for h in other_helpers if h.signature
    ]
    ref_summaries = [
        f"/* [HELPER SUMMARY] */ {h.signature};" for h in ref_helpers if h.signature
    ]

    static_header = "\n\n".join(p for p in prefix_parts if p)
    footer = "\n\n".join(helper_summaries)
    overhead_bytes = len((static_header + "\n\n" + footer).encode("utf-8")) + 64

    body_parts: list[tuple[str, _CppUnit | None]] = []
    if static_header:
        body_parts.append((static_header, None))

    if target_unit is not None:
        ref_full_text = "\n\n".join(h.text.strip() for h in ref_helpers)
        full_with_helpers = (
            overhead_bytes
            + len(ref_full_text.encode("utf-8"))
            + len(target_unit.text.encode("utf-8"))
        )
        if full_with_helpers <= max_bytes:
            for rh in ref_helpers:
                rh_text = rh.text.strip()
                if rh_text:
                    body_parts.append((rh_text, rh))
            body_parts.append((target_unit.text.strip(), target_unit))
        else:
            focused_helpers = [
                h
                for h in ref_helpers
                if any(h.start_line <= fl <= h.end_line for fl in resolved_focus)
            ]
            unfocused_ref_helpers = [h for h in ref_helpers if h not in focused_helpers]
            ref_sum_text = "\n".join(
                f"/* [HELPER SUMMARY] */ {h.signature};"
                for h in unfocused_ref_helpers
                if h.signature
            )
            if ref_sum_text:
                body_parts.append((ref_sum_text, None))

            target_has_focus = any(
                target_unit.start_line <= fl <= target_unit.end_line for fl in resolved_focus
            )
            if focused_helpers:
                fixed_bytes = (
                    len("\n\n".join(p for p, _ in body_parts if p).encode("utf-8"))
                    + (len(("\n\n" + footer).encode("utf-8")) if footer else 0)
                    + 16
                )
                avail_code = max(512, max_bytes - fixed_bytes)
                helper_cap = max(
                    256,
                    int(avail_code * (0.45 if target_has_focus else 0.65)),
                )
                raw_focused = [h.text.strip() for h in focused_helpers if h.text.strip()]
                raw_focused_bytes = sum(len(t.encode("utf-8")) for t in raw_focused)
                if raw_focused_bytes + 4 * len(raw_focused) <= helper_cap:
                    for fh in focused_helpers:
                        fh_text = fh.text.strip()
                        if fh_text:
                            body_parts.append((fh_text, fh))
                else:
                    per_helper_budget = max(
                        256,
                        (helper_cap - 4 * len(focused_helpers)) // len(focused_helpers),
                    )
                    for fh in focused_helpers:
                        fh_text = fh.text.strip()
                        if not fh_text:
                            continue
                        body_parts.append(
                            (
                                _truncate_kernel_body(
                                    fh_text,
                                    per_helper_budget,
                                    unit_start_line=fh.start_line,
                                    focus_lines=resolved_focus,
                                    focus_window=focus_window,
                                ),
                                fh,
                            )
                        )

            used_parts = [p for p, _ in body_parts if p]
            if footer:
                used_parts.append(footer)
            used_now = len("\n\n".join(used_parts).encode("utf-8")) + 4
            remaining = max(256, max_bytes - used_now)
            body_parts.append(
                (
                    _truncate_kernel_body(
                        target_unit.text.strip(),
                        remaining,
                        unit_start_line=target_unit.start_line,
                        focus_lines=resolved_focus,
                        focus_window=focus_window,
                    ),
                    target_unit,
                )
            )
    elif ref_helpers:
        body_parts.append(("\n".join(ref_summaries), None))

    if footer:
        body_parts.append((footer, None))

    non_empty_parts = [(p, unit) for p, unit in body_parts if p]
    result = "\n\n".join(p for p, _ in non_empty_parts).strip()
    if len(result.encode("utf-8")) > max_bytes:
        remapped_focus: list[int] = []
        if resolved_focus:
            cpp_lines = cpp_source.splitlines()
            cur_line = 1
            for part_idx, (part_text, unit) in enumerate(non_empty_parts):
                p_text = part_text.strip() if part_idx in (0, len(non_empty_parts) - 1) else part_text
                p_lines = p_text.splitlines()
                if unit is not None:
                    for fl in resolved_focus:
                        if not (unit.start_line <= fl <= unit.end_line) or not (1 <= fl <= len(cpp_lines)):
                            continue
                        orig_line = cpp_lines[fl - 1].rstrip()
                        raw_rel = fl - unit.start_line
                        if 0 <= raw_rel < len(p_lines) and p_lines[raw_rel].rstrip() == orig_line:
                            remapped_focus.append(cur_line + raw_rel)
                            continue
                        matches = [
                            idx
                            for idx, pln in enumerate(p_lines)
                            if pln.rstrip() == orig_line
                        ]
                        if matches:
                            best_idx = min(matches, key=lambda idx: abs(idx - raw_rel))
                            remapped_focus.append(cur_line + best_idx)
                cur_line += len(p_lines) + 1
        result = _truncate_kernel_body(
            result,
            max_bytes,
            unit_start_line=1,
            focus_lines=tuple(sorted(set(remapped_focus))),
            focus_window=focus_window,
        )
    return result


def parse_scoped_search_replace_blocks(patch_text: str) -> list[tuple[str, str]]:
    """Parse SEARCH/REPLACE patch hunks from scoped Track-B candidate output."""
    cleaned_patch = strip_markdown_fence_lines(patch_text)
    for pat in _SEARCH_REPLACE_PATTERNS:
        matches = [
            (strip_markdown_fence_lines(m.group(1)), strip_markdown_fence_lines(m.group(2)))
            for m in pat.finditer(cleaned_patch)
        ]
        if matches:
            return matches
    return []


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


def _referenced_helpers_in_text(
    text: str, helpers_by_name: Mapping[str, _CppUnit]
) -> set[str]:
    masked = strip_cpp_comments_and_strings(text)
    refs: set[str] = set()
    for match in _CALL_NAME_RE.finditer(masked):
        name = match.group(1)
        if name in helpers_by_name:
            refs.add(name)
    for match in _TEMPLATE_CALL_ARGS_RE.finditer(masked):
        for ident in _IDENT_RE.finditer(match.group(1)):
            name = ident.group(1)
            if name in helpers_by_name:
                refs.add(name)
    for match in _LAUNCH_CONFIG_RE.finditer(masked):
        name = match.group(1)
        if name in helpers_by_name:
            refs.add(name)
    return refs


def _helper_call_closure(unit: _CppUnit, helpers_by_name: Mapping[str, _CppUnit]) -> set[str]:
    reached: set[str] = set()
    pending: list[_CppUnit] = [unit]
    while pending:
        current = pending.pop()
        for name in _referenced_helpers_in_text(current.text, helpers_by_name):
            if name not in reached:
                reached.add(name)
                pending.append(helpers_by_name[name])
    return reached


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


def apply_scoped_trackb_patch(
    cpp_source: str,
    patch_text: str,
    *,
    active_kernel: str | None = None,
    allow_exclusive_helpers: bool = True,
) -> tuple[str, int]:
    """Apply scoped SEARCH/REPLACE hunks to *cpp_source*, forbidding read-only sibling edits.

    With *active_kernel* set, hunks must match the active kernel body or a helper
    reachable only from that kernel; helpers shared with sibling kernels stay
    read-only. When the exact substring lookup misses, a unique match ignoring
    line-tail whitespace and common indentation is accepted.
    """
    if _SIBLING_MARKER_RE.search(patch_text):
        raise ValueError("Patch attempted to copy or edit a read-only sibling kernel marker")

    hunks = parse_scoped_search_replace_blocks(patch_text)
    if not hunks:
        raise ValueError("patch_text must contain at least one valid SEARCH/REPLACE block")

    updated = cpp_source
    applied = 0
    for search_block, replace_block in hunks:
        if not search_block.strip():
            raise ValueError("Empty SEARCH block is forbidden")
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

        match_count = updated.count(search_block)
        if match_count == 0:
            fallback = _normalized_line_matches(updated, search_block)
            if not fallback:
                raise ValueError("SEARCH block did not match target C++ source")
            if len(fallback) > 1:
                raise ValueError(
                    f"Ambiguous SEARCH block matched {len(fallback)} locations; "
                    "provide more context"
                )
            match_start, match_end, window_indent = fallback[0]
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

        if match_count > 1:
            raise ValueError(
                f"Ambiguous SEARCH block matched {match_count} locations; provide more context"
            )
        match_start = updated.index(search_block)
        match_end = match_start + len(search_block)
        if not _in_scope(match_start, match_end):
            raise ValueError(
                f"Scoped patch targets read-only code outside active kernel {active_kernel!r}"
            )

        updated = updated[:match_start] + replace_block + updated[match_end:]
        applied += 1

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
    return updated, len(targets) + len(new_helpers)


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


CREATIVE_SOURCE_MAX_BYTES = 14_336

_TRACKB_CPP_QUALIFIERS: tuple[str, ...] = (
    'extern "C" __global__',
    "__global__",
    "__aicore__",
    "__simd_vf__",
    "__simt_vf__",
)

_OMITTED_SPAN_MARKER_RE = re.compile(
    r"^\s*/\*\s*(?:"
    r"\.\.\.\s*(?P<count>\d+)\s+lines?\s+omitted\s*\.\.\."
    r"|\[MIDDLE BODY OMITTED[^\n]*\]"
    r"|\[LINES OMITTED OUTSIDE FOCUS WINDOW[^\n]*\]"
    r")\s*\*/\s*$"
)

_ANY_FOLDING_MARKER_RE = re.compile(
    r"/\*\s*\.\.\.\s*(?:\d+\s+)?lines?\s+omitted\s*\.\.\.\s*\*/"
    r"|READ-ONLY SIBLING KERNEL"
    r"|READ-ONLY HELPER"
    r"|MIDDLE BODY OMITTED"
    r"|LINES OMITTED OUTSIDE FOCUS WINDOW"
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


def _compact_trackb_repair_source(
    raw_src: str,
    diag: str,
    max_bytes: int = 14800,
    *,
    parent_source: str = "",
    prologue_line_offset: int = 0,
    focus_lines: Sequence[int] = (),
    active_kernel: str | None = None,
) -> str:
    """Slice oversized Track-B repair source while keeping hybrid Python AST valid."""
    if len(raw_src.encode("utf-8")) <= max_bytes:
        return raw_src
    diag_focus = extract_cpp_diagnostic_lines(diag)
    merged_focus = tuple(sorted({*diag_focus, *(int(ln) for ln in focus_lines if int(ln) > 0)}))
    try:
        ast.parse(raw_src)
    except SyntaxError:
        if "__global__" in raw_src or "__aicore__" in raw_src or "__simd_vf__" in raw_src:
            return build_trackb_context_view(
                raw_src,
                active_kernel=active_kernel,
                max_bytes=max_bytes,
                focus_lines=merged_focus,
                diagnostic_text=diag,
            )
        return raw_src

    hits = _find_python_cpp_literal_spans(raw_src, require_cpp_tokens=True)
    if not hits:
        return raw_src

    prologue_offset = (
        int(prologue_line_offset)
        if prologue_line_offset > 0
        else _detect_trackb_prologue_line_offset(raw_src, parent_source=parent_source)
    )
    adjusted_focus = (
        tuple(ln - prologue_offset for ln in merged_focus if ln > prologue_offset)
        if prologue_offset > 0
        else merged_focus
    )
    diag_for_view = "" if prologue_offset > 0 else diag

    raw_hit_bytes = [len(raw_src[s:e].encode("utf-8")) for _, _, s, e in hits]
    non_cpp_outer_bytes = max(0, len(raw_src.encode("utf-8")) - sum(raw_hit_bytes))
    avail_total = max(0, max_bytes - non_cpp_outer_bytes - 16 * len(hits))
    for _pass in range(2):
        per_hit_budget = max(256, avail_total // len(hits))
        small_used = sum(b for b in raw_hit_bytes if b <= per_hit_budget)
        large_count = sum(1 for b in raw_hit_bytes if b > per_hit_budget)
        large_budget = (
            max(per_hit_budget, (avail_total - small_used) // large_count)
            if large_count > 0
            else per_hit_budget
        )

        replacements: list[tuple[int, int, str]] = []
        for idx, (sym_name, cpp_text, start_pos, end_pos) in enumerate(hits):
            cpp_budget = per_hit_budget if raw_hit_bytes[idx] <= per_hit_budget else large_budget
            view = build_trackb_context_view(
                cpp_text,
                active_kernel=active_kernel,
                module_source=raw_src,
                trackb_symbol=sym_name,
                max_bytes=cpp_budget,
                focus_lines=adjusted_focus,
                diagnostic_text=diag_for_view,
            )
            wrapped = view if view.startswith("\n") else f"\n{view}\n"
            replacements.append((start_pos, end_pos, _format_cpp_string_literal(wrapped)))

        updated = raw_src
        for start_pos, end_pos, literal in reversed(replacements):
            updated = updated[:start_pos] + literal + updated[end_pos:]
        ast.parse(updated)
        actual_bytes = len(updated.encode("utf-8"))
        if actual_bytes <= max_bytes or avail_total <= 512:
            return updated
        avail_total = max(256, avail_total - (actual_bytes - max_bytes) - 32 * len(hits))
    return updated


def _match_omitted_anchor_span(
    orig_lines: Sequence[str],
    pre_anchor: Sequence[str],
    post_anchor: Sequence[str],
    *,
    orig_cursor: int,
    expected_count: int | None,
    post_reaches_eof: bool,
) -> tuple[int, int] | None:
    n_pre = len(pre_anchor)
    n_post = len(post_anchor)
    orig_norm = [ln.rstrip() for ln in orig_lines]
    pre_starts = [
        idx
        for idx in range(orig_cursor, len(orig_norm) - n_pre + 1)
        if orig_norm[idx : idx + n_pre] == list(pre_anchor)
    ]
    post_starts = [
        idx
        for idx in range(orig_cursor + n_pre + 1, len(orig_norm) - n_post + 1)
        if orig_norm[idx : idx + n_post] == list(post_anchor)
    ]
    pairs: list[tuple[int, int]] = []
    for pre_i in pre_starts:
        pre_end = pre_i + n_pre
        for post_j in post_starts:
            if pre_end >= post_j:
                continue
            if expected_count is not None and (post_j - pre_end) != expected_count:
                continue
            pairs.append((pre_end, post_j))
    if len(pairs) == 1:
        return pairs[0]
    if len(pairs) > 1 and post_reaches_eof:
        eof_pairs = [pair for pair in pairs if pair[1] + n_post == len(orig_norm)]
        if len(eof_pairs) == 1:
            return eof_pairs[0]
    return None


def _splice_cpp_omitted_spans(cand_cpp: str, orig_cpp: str) -> tuple[str, bool]:
    """Restore omitted C++ spans using exact anchor lines from pre-compaction source."""
    cand_lines = cand_cpp.splitlines()
    orig_lines = orig_cpp.splitlines()
    marker_hits: list[tuple[int, int | None]] = []
    for idx, line in enumerate(cand_lines):
        m = _OMITTED_SPAN_MARKER_RE.match(line)
        if m is not None:
            raw_count = m.group("count")
            marker_hits.append((idx, int(raw_count) if raw_count is not None else None))
    if not marker_hits:
        return cand_cpp, not bool(_ANY_FOLDING_MARKER_RE.search(cand_cpp))

    rebuilt: list[str] = []
    cand_cursor = 0
    orig_cursor = 0
    for k, (m_idx, expected_count) in enumerate(marker_hits):
        prev_limit = (marker_hits[k - 1][0] + 1) if k > 0 else 0
        next_limit = marker_hits[k + 1][0] if k + 1 < len(marker_hits) else len(cand_lines)

        b_end = m_idx
        while b_end > prev_limit and not cand_lines[b_end - 1].strip():
            b_end -= 1
        a_start = m_idx + 1
        while a_start < next_limit and not cand_lines[a_start].strip():
            a_start += 1
        if b_end <= prev_limit or a_start >= next_limit:
            return cand_cpp, False

        avail_pre = b_end - prev_limit
        avail_post = next_limit - a_start
        anchor_sizes = sorted(
            (
                (np_pre, np_post)
                for np_pre in range(1, min(4, avail_pre) + 1)
                for np_post in range(1, min(4, avail_post) + 1)
            ),
            key=lambda pair: (pair[0] + pair[1], min(pair[0], pair[1])),
            reverse=True,
        )
        matched_span: tuple[int, int] | None = None
        for np_pre, np_post in anchor_sizes:
            pre_anchor = [ln.rstrip() for ln in cand_lines[b_end - np_pre : b_end]]
            post_anchor = [ln.rstrip() for ln in cand_lines[a_start : a_start + np_post]]
            if not any(s.strip() for s in pre_anchor) or not any(s.strip() for s in post_anchor):
                continue
            post_reaches_eof = (k + 1 == len(marker_hits)) and (
                a_start + np_post == len(cand_lines)
            )
            matched_span = _match_omitted_anchor_span(
                orig_lines,
                pre_anchor,
                post_anchor,
                orig_cursor=orig_cursor,
                expected_count=expected_count,
                post_reaches_eof=post_reaches_eof,
            )
            if matched_span is not None:
                break
        if matched_span is None:
            return cand_cpp, False

        orig_pre_end, orig_post_start = matched_span
        rebuilt.extend(cand_lines[cand_cursor:b_end])
        rebuilt.extend(orig_lines[orig_pre_end:orig_post_start])
        cand_cursor = a_start
        orig_cursor = orig_post_start

    rebuilt.extend(cand_lines[cand_cursor:])
    restored = "\n".join(rebuilt)
    if cand_cpp.endswith("\n"):
        restored += "\n"
    if _ANY_FOLDING_MARKER_RE.search(restored):
        return cand_cpp, False
    return restored, True


def _collect_orig_cpp_map(sources: Sequence[str]) -> tuple[dict[str, str], list[str]]:
    by_sym: dict[str, str] = {}
    bare_list: list[str] = []
    for src in sources:
        if not src or not src.strip():
            continue
        hits = _find_python_cpp_literal_spans(src, require_cpp_tokens=False)
        if hits:
            for sym_name, cpp_text, _, _ in hits:
                if sym_name not in by_sym:
                    by_sym[sym_name] = cpp_text
                bare_list.append(cpp_text)
        else:
            bare_list.append(src)
    return by_sym, bare_list


def _resplice_trackb_omitted_spans(
    repaired_src: str,
    *,
    pre_compaction_source: str,
    parent_source: str = "",
    extra_sources: Sequence[str] = (),
) -> str:
    """Splice unchanged omitted C++ spans back into a repair reply when resolvable."""
    if not repaired_src or parse_scoped_search_replace_blocks(repaired_src):
        return repaired_src
    if not _ANY_FOLDING_MARKER_RE.search(repaired_src):
        return repaired_src

    by_sym, bare_list = _collect_orig_cpp_map(
        (pre_compaction_source, *extra_sources, parent_source)
    )
    if not by_sym and not bare_list:
        return repaired_src

    rep_hits = _find_python_cpp_literal_spans(repaired_src, require_cpp_tokens=False)
    if rep_hits:
        replacements: list[tuple[int, int, str]] = []
        for sym_name, cpp_text, start_pos, end_pos in rep_hits:
            if not _ANY_FOLDING_MARKER_RE.search(cpp_text):
                continue
            candidates: list[str] = []
            if sym_name in by_sym:
                candidates.append(by_sym[sym_name])
            for raw_cand in bare_list:
                if raw_cand not in candidates:
                    candidates.append(raw_cand)
            spliced_val: str | None = None
            for orig_cpp in candidates:
                res, ok = _splice_cpp_omitted_spans(cpp_text, orig_cpp)
                if ok:
                    spliced_val = res
                    break
            if spliced_val is not None:
                replacements.append((start_pos, end_pos, _format_cpp_string_literal(spliced_val)))
        if not replacements:
            return repaired_src
        updated = repaired_src
        for start_pos, end_pos, literal in reversed(replacements):
            updated = updated[:start_pos] + literal + updated[end_pos:]
        try:
            ast.parse(updated)
        except SyntaxError:
            return repaired_src
        return updated

    for orig_cpp in bare_list:
        res, ok = _splice_cpp_omitted_spans(repaired_src, orig_cpp)
        if ok:
            return res
    return repaired_src


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


def _is_active_python_fn(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    source: str,
    focus_lines: Sequence[int],
    active_kernel: str | None,
) -> bool:
    if active_kernel and node.name == active_kernel:
        return True
    end_ln = node.end_lineno or node.lineno
    if any(node.lineno <= fl <= end_ln for fl in focus_lines):
        return True
    if node.name in ("main", "kernel", "make_kernel") or node.name.startswith("_select_"):
        return True
    seg = ast.get_source_segment(source, node) or ""
    return any(
        tok in seg
        for tok in ("@T.prim_func", "T.Kernel(", "T.MixedKernel(", "T.Persistent(")
    )


def _fold_cold_stmt_spans(
    stmts: Sequence[ast.stmt],
    *,
    focus_lines: Sequence[int],
    window: int,
    replacements: list[tuple[int, int, str]],
    lines: Sequence[str],
) -> None:
    if len(stmts) <= 4:
        return
    cold_run: list[ast.stmt] = []

    def _flush() -> None:
        if len(cold_run) < 2:
            cold_run.clear()
            return
        first = cold_run[0]
        last = cold_run[-1]
        start_ln = first.lineno
        end_ln = last.end_lineno or last.lineno
        indent = " " * first.col_offset
        marker = (
            f"{indent}# [READ-ONLY COLD BLOCK FOLDED ({len(cold_run)} statements)]\n"
            f"{indent}...\n"
        )
        replacements.append((start_ln, end_ln, marker))
        cold_run.clear()

    for idx, stmt in enumerate(stmts):
        s_ln = stmt.lineno
        e_ln = stmt.end_lineno or s_ln
        is_edge = idx == 0 or idx == len(stmts) - 1
        near_focus = any(s_ln - window <= fl <= e_ln + window for fl in focus_lines)
        is_structural = isinstance(
            stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Return, ast.With, ast.AsyncWith)
        )
        if is_edge or near_focus or is_structural:
            _flush()
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.With, ast.AsyncWith)):
                _fold_cold_stmt_spans(
                    stmt.body,
                    focus_lines=focus_lines,
                    window=window,
                    replacements=replacements,
                    lines=lines,
                )
            elif isinstance(stmt, ast.For):
                _fold_cold_stmt_spans(
                    stmt.body,
                    focus_lines=focus_lines,
                    window=window,
                    replacements=replacements,
                    lines=lines,
                )
        else:
            cold_run.append(stmt)
    _flush()


def _compact_python_source_slice(
    source: str,
    *,
    max_bytes: int,
    focus_lines: Sequence[int] = (),
    active_kernel: str | None = None,
) -> str:
    if len(source.encode("utf-8")) <= max_bytes:
        return source
    try:
        tree = ast.parse(source)
    except SyntaxError:
        encoded = source.encode("utf-8")[: max(128, max_bytes - 32)]
        return encoded.decode("utf-8", errors="ignore").rstrip() + "\n# ...\n"

    lines = source.splitlines(keepends=True)
    inactive_repls: list[tuple[int, int, str]] = []
    active_fns: list[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if _is_active_python_fn(
                node,
                source=source,
                focus_lines=focus_lines,
                active_kernel=active_kernel,
            ):
                active_fns.append(node)
            elif node.body:
                b_start = node.body[0].lineno
                b_end = node.end_lineno or node.body[-1].end_lineno or b_start
                indent = " " * max(4, node.body[0].col_offset)
                stub = (
                    f"{indent}# [READ-ONLY HELPER FOLDED TO FIT <=14KB BUDGET]\n"
                    f"{indent}...\n"
                )
                inactive_repls.append((b_start, b_end, stub))

    def _apply_line_replacements(repls: Sequence[tuple[int, int, str]]) -> str:
        if not repls:
            return source
        buf = list(lines)
        for s_ln, e_ln, text in sorted(repls, key=lambda item: item[0], reverse=True):
            buf[s_ln - 1 : e_ln] = [text]
        return "".join(buf)

    updated = _apply_line_replacements(inactive_repls)
    if len(updated.encode("utf-8")) <= max_bytes:
        ast.parse(updated)
        return updated

    for window in (15, 6, 1, 0):
        stmt_repls = list(inactive_repls)
        for fn in active_fns:
            _fold_cold_stmt_spans(
                fn.body,
                focus_lines=focus_lines,
                window=window,
                replacements=stmt_repls,
                lines=lines,
            )
        cand = _apply_line_replacements(stmt_repls)
        if len(cand.encode("utf-8")) <= max_bytes:
            ast.parse(cand)
            return cand
        updated = cand

    lines_out = updated.splitlines(keepends=True)
    while len("".join(lines_out).encode("utf-8")) > max_bytes and len(lines_out) > 2:
        lines_out.pop()
    trimmed = "".join(lines_out)
    try:
        ast.parse(trimmed)
        return trimmed
    except SyntaxError:
        return trimmed


def compact_creative_source_slice(
    source: str,
    *,
    max_bytes: int = CREATIVE_SOURCE_MAX_BYTES,
    focus_lines: Sequence[int] = (),
    diagnostic_text: str = "",
    active_kernel: str | None = None,
    parent_source: str = "",
    prologue_line_offset: int = 0,
) -> str:
    """Fold oversized creative-exploration parent source into an AST-safe <=14KB focus view."""
    if len(source.encode("utf-8")) <= max_bytes:
        return source
    if any(tok in source for tok in _TRACKB_CPP_QUALIFIERS):
        compacted = _compact_trackb_repair_source(
            source,
            diagnostic_text,
            max_bytes=max_bytes,
            parent_source=parent_source,
            prologue_line_offset=prologue_line_offset,
            focus_lines=focus_lines,
            active_kernel=active_kernel,
        )
        if len(compacted.encode("utf-8")) <= max_bytes:
            return compacted
        source = compacted
    return _compact_python_source_slice(
        source,
        max_bytes=max_bytes,
        focus_lines=focus_lines,
        active_kernel=active_kernel,
    )


__all__ = [
    "CREATIVE_SOURCE_MAX_BYTES",
    "_ANY_FOLDING_MARKER_RE",
    "_OMITTED_SPAN_MARKER_RE",
    "_TRACKB_CPP_QUALIFIERS",
    "_compact_trackb_repair_source",
    "_detect_trackb_prologue_line_offset",
    "_find_python_cpp_literal_spans",
    "_looks_like_python_primfunc_slice",
    "_resplice_trackb_omitted_spans",
    "_splice_cpp_omitted_spans",
    "apply_scoped_trackb_patch",
    "build_trackb_context_view",
    "compact_creative_source_slice",
    "escalate_primfunc_to_trackb_tu",
    "extract_cpp_diagnostic_lines",
    "extract_reachable_trackb_source",
    "extract_trackb_launcher_preview",
    "parse_scoped_search_replace_blocks",
    "resolve_instantiated_trackb_kernel",
    "select_active_trackb_kernel",
    "splice_cpp_units_into_tu",
    "strip_cpp_comments_and_strings",
    "strip_markdown_fence_lines",
]
