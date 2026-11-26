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
_SIBLING_MARKER_RE = re.compile(
    r"(?:/\*\s*\[READ-ONLY SIBLING KERNEL:|//\s*\[READONLY_SIBLING_DECLARATION\])"
)
_LAUNCH_CONFIG_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*(?:<[^;{}]*>)?\s*<<<")
_MARKDOWN_FENCE_LINE_RE = re.compile(r"^\s*```[A-Za-z0-9_+.-]*\s*$")
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

    def _kernel_rank(unit: _CppUnit) -> tuple[int, int]:
        is_f32_fallback = (
            "f32" in unit.name.lower() or "fp32" in unit.name.lower()
        ) and ("f32" not in tag and "fp32" not in tag)
        return (0 if is_f32_fallback else 1, unit.line_count)

    if tag:
        tokens = [tok for tok in re.split(r"[^a-z0-9]+", tag) if len(tok) >= 3]
        scored = sorted(
            kernels,
            key=lambda u: (
                -sum(1 for t in tokens if t in u.name.lower()),
                -(1 if u.name in launched_names else 0),
                -_kernel_rank(u)[0],
                -_kernel_rank(u)[1],
            ),
        )
        if scored and any(t in scored[0].name.lower() for t in tokens):
            return scored[0].name
    if outside_launched_names:
        launched = [u for u in kernels if u.name in outside_launched_names]
        return max(launched, key=_kernel_rank).name
    if cpp_launched_names:
        launched = [u for u in kernels if u.name in cpp_launched_names]
        if launched:
            return max(launched, key=_kernel_rank).name
    if outside_cpp:
        called = [
            unit
            for unit in kernels
            if f"{unit.name}<" in outside_cpp or f"{unit.name}(" in outside_cpp
        ]
        if called:
            return max(called, key=_kernel_rank).name
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


CREATIVE_SOURCE_MAX_BYTES = 14_336

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
    """Scope Track-B C++ sources losslessly or fold oversized pure-Python parent sources."""
    if len(source.encode("utf-8")) <= max_bytes:
        return source
    if any(tok in source for tok in _TRACKB_CPP_QUALIFIERS):
        return _scope_trackb_python_or_cpp(
            source,
            diagnostic_text,
            parent_source=parent_source,
            prologue_line_offset=prologue_line_offset,
            focus_lines=focus_lines,
            active_kernel=active_kernel,
        )
    return _compact_python_source_slice(
        source,
        max_bytes=max_bytes,
        focus_lines=focus_lines,
        active_kernel=active_kernel,
    )


__all__ = [
    "AmbiguousPatchAnchorError",
    "CREATIVE_SOURCE_MAX_BYTES",
    "NoopIdenticalPatchError",
    "PatchAnchorMissError",
    "TRACKB_REPLY_CONTRACT_PROMPT",
    "_TRACKB_CPP_QUALIFIERS",
    "_detect_trackb_prologue_line_offset",
    "_find_python_cpp_literal_spans",
    "_looks_like_python_primfunc_slice",
    "apply_scoped_trackb_patch",
    "build_trackb_context_view",
    "compact_creative_source_slice",
    "escalate_primfunc_to_trackb_tu",
    "extract_cpp_diagnostic_lines",
    "extract_reachable_trackb_source",
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
    "strip_cpp_comments_and_strings",
    "strip_markdown_fence_lines",
]
