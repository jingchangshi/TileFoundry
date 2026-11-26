"""Dependency-free C++20 unit splitting and reachable kernel closure extraction."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

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
_LAUNCH_CONFIG_RE = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*(?:<[^;{}]*>)?\s*<<<")
_MARKDOWN_FENCE_LINE_RE = re.compile(r"^\s*```[A-Za-z0-9_+.-]*\s*$")
_DEVICE_QUALIFIER_RE = re.compile(
    r"__(?:global|aicore|simd_vf|simt_vf|simd_callee|simt_callee)__\b"
)


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
