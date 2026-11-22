"""Scoped C++20 slicing, patching, and Track-A-to-Track-B escalation protocol."""

from __future__ import annotations

import ast
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
                re.match(r"^(?:struct|enum|using|typedef)\b", sig_head)
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


def select_active_trackb_kernel(
    cpp_source: str,
    *,
    active_kernel: str | None = None,
    regime_tag: str = "",
    module_source: str = "",
) -> str | None:
    """Resolve the active ``__global__`` kernel name for scoped Track-B context extraction."""
    _, units = _split_cpp_units(cpp_source)
    kernels = [u for u in units if u.kind == "kernel" and u.name]
    if not kernels:
        return None
    if active_kernel:
        for unit in kernels:
            if unit.name == active_kernel:
                return unit.name
        return None
    tag = str(regime_tag or "").lower()
    if tag:
        tokens = [tok for tok in re.split(r"[^a-z0-9]+", tag) if len(tok) >= 3]
        scored = sorted(
            kernels,
            key=lambda u: (
                -sum(1 for t in tokens if t in u.name.lower()),
                -u.line_count,
            ),
        )
        if scored and any(t in scored[0].name.lower() for t in tokens):
            return scored[0].name
    if module_source:
        outside_cpp = module_source.replace(cpp_source, "")
        launched = [
            unit
            for unit in kernels
            if re.search(rf"\b{re.escape(unit.name)}\s*(?:<[^>]*>)?\s*<<<", outside_cpp)
        ]
        if launched:
            return max(launched, key=lambda u: u.line_count).name
        called = [
            unit
            for unit in kernels
            if f"{unit.name}<" in outside_cpp or f"{unit.name}(" in outside_cpp
        ]
        if called:
            return max(called, key=lambda u: u.line_count).name
    return kernels[0].name


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


def _truncate_kernel_body(kernel_text: str, max_bytes: int) -> str:
    if len(kernel_text.encode("utf-8")) <= max_bytes:
        return kernel_text
    lines = kernel_text.splitlines(keepends=True)
    if len(lines) <= 20:
        encoded = kernel_text.encode("utf-8")[: max(128, max_bytes - 48)]
        return encoded.decode("utf-8", errors="ignore").rstrip() + "\n}"
    marker = "\n    /* [MIDDLE BODY OMITTED TO FIT <15KB SCOPED VIEW; PATCH VISIBLE LINES] */\n"
    budget = max(256, max_bytes - len(marker.encode("utf-8")))
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
    return "".join(head_lines).rstrip() + marker + "".join(tail_lines).lstrip("\n")


def build_trackb_context_view(
    cpp_source: str,
    *,
    active_kernel: str | None = None,
    regime_tag: str = "",
    module_source: str = "",
    trackb_symbol: str = "",
    launcher_preview: str = "",
    max_bytes: int = 14800,
) -> str:
    """Build a bounded (<15 KB) scoped C++20 context view for Track-B editing."""
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

    body_parts: list[str] = []
    if static_header:
        body_parts.append(static_header)

    if target_unit is not None:
        ref_full_text = "\n\n".join(h.text.strip() for h in ref_helpers)
        full_with_helpers = (
            overhead_bytes
            + len(ref_full_text.encode("utf-8"))
            + len(target_unit.text.encode("utf-8"))
        )
        if full_with_helpers <= max_bytes:
            if ref_full_text:
                body_parts.append(ref_full_text)
            body_parts.append(target_unit.text.strip())
        else:
            ref_sum_text = "\n".join(ref_summaries)
            if ref_sum_text:
                body_parts.append(ref_sum_text)
            used_now = len("\n\n".join(body_parts).encode("utf-8")) + len(footer.encode("utf-8")) + 64
            remaining = max(1024, max_bytes - used_now)
            body_parts.append(_truncate_kernel_body(target_unit.text.strip(), remaining))
    elif ref_helpers:
        body_parts.append("\n".join(ref_summaries))

    if footer:
        body_parts.append(footer)

    result = "\n\n".join(p for p in body_parts if p).strip()
    if len(result.encode("utf-8")) > max_bytes:
        result = _truncate_kernel_body(result, max_bytes)
    return result


def parse_scoped_search_replace_blocks(patch_text: str) -> list[tuple[str, str]]:
    """Parse SEARCH/REPLACE patch hunks from scoped Track-B candidate output."""
    for pat in _SEARCH_REPLACE_PATTERNS:
        matches = [(m.group(1), m.group(2)) for m in pat.finditer(patch_text)]
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
    _, candidate_units = _split_cpp_units(candidate_cpp)
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

    lowered = tilelang.lower(prim_func, target="ascend")
    kernel_src = str(getattr(lowered, "kernel_source", "") or "").strip()
    if not kernel_src:
        raise RuntimeError("tilelang.lower returned empty kernel_source for Track-B escalation")
    return kernel_src


__all__ = [
    "apply_scoped_trackb_patch",
    "build_trackb_context_view",
    "escalate_primfunc_to_trackb_tu",
    "extract_trackb_launcher_preview",
    "parse_scoped_search_replace_blocks",
    "resolve_instantiated_trackb_kernel",
    "select_active_trackb_kernel",
    "splice_cpp_units_into_tu",
    "strip_cpp_comments_and_strings",
]
