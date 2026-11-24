"""Parse ``bisheng --cce-res-usage`` output and CCE assembly dumps.

Also covers the Tier-1 behavior registry.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
from typing import Any

from tilefoundry_ascend.ascend.passes.physical_ledger import ObjectivePhysicalLedger

_UNIT_MULTIPLIER: dict[str, int] = {
    "": 1,
    "b": 1,
    "byte": 1,
    "bytes": 1,
    "kb": 1024,
    "kib": 1024,
    "mb": 1024 * 1024,
    "mib": 1024 * 1024,
}


@dataclass(frozen=True)
class BishengResourceSummary:
    """Extracted hardware register, spill, stack, loop, and compiler-hazard telemetry."""

    vec_regs: int = 0
    scalar_regs: int = 0
    spill_bytes: int = 0
    stack_bytes: int = 0
    vloopv2_loops: int = 0
    software_loops: int = 0
    has_local_membar_in_vf_call: bool = False
    vf_fusion_breaks: int = 0
    alias_serialized_buffers: int = 0
    dual_issue_ipc_collapse: float = 0.0
    dcache_dcci_flushes: int = 0
    extra_signals: dict[str, float] = field(default_factory=dict)

    @property
    def has_spill(self) -> bool:
        """Return True if bisheng emitted register or stack spill."""
        return self.spill_bytes > 0

    @property
    def vloopv2_hit(self) -> bool:
        """Return True if VLOOPv2 fired and no fallback software loop remains."""
        return self.vloopv2_loops > 0 and self.software_loops == 0

    def to_dict(self) -> dict[str, object]:
        """Serialize summary to a plain JSON-safe dictionary."""
        out = asdict(self)
        out["extra_signals"] = {str(k): float(v) for k, v in self.extra_signals.items()}
        return out

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> BishengResourceSummary:
        """Reconstruct BishengResourceSummary from a mapping."""
        raw_extra = payload.get("extra_signals") or {}
        extra = (
            {str(k): float(v) for k, v in raw_extra.items()}
            if isinstance(raw_extra, Mapping)
            else {}
        )
        return cls(
            vec_regs=int(payload.get("vec_regs", 0) or 0),
            scalar_regs=int(payload.get("scalar_regs", 0) or 0),
            spill_bytes=int(payload.get("spill_bytes", 0) or 0),
            stack_bytes=int(payload.get("stack_bytes", 0) or 0),
            vloopv2_loops=int(payload.get("vloopv2_loops", 0) or 0),
            software_loops=int(payload.get("software_loops", 0) or 0),
            has_local_membar_in_vf_call=bool(
                payload.get("has_local_membar_in_vf_call", False)
            ),
            vf_fusion_breaks=max(0, int(payload.get("vf_fusion_breaks", 0) or 0)),
            alias_serialized_buffers=max(
                0, int(payload.get("alias_serialized_buffers", 0) or 0)
            ),
            dual_issue_ipc_collapse=max(
                0.0, min(1.0, float(payload.get("dual_issue_ipc_collapse", 0.0) or 0.0))
            ),
            dcache_dcci_flushes=max(0, int(payload.get("dcache_dcci_flushes", 0) or 0)),
            extra_signals=extra,
        )


def _match_int(patterns: tuple[str, ...], text: str) -> int | None:
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            val = int(m.group(1))
            unit = (m.group(2) if m.lastindex and m.lastindex >= 2 else "") or ""
            mult = _UNIT_MULTIPLIER.get(unit.strip().lower(), 1)
            return val * mult
    return None


def _match_float(patterns: tuple[str, ...], text: str) -> float | None:
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)
        if m:
            return float(m.group(1))
    return None


def _detect_local_membar_in_vf(res_text: str = "", asm_or_dump_text: str = "") -> bool:
    combined = "\n".join(part for part in (res_text, asm_or_dump_text) if part)
    if re.search(
        r"has_local_membar_in_vf_call\s*[:=]\s*(true|1|yes)",
        combined,
        flags=re.IGNORECASE,
    ):
        return True
    has_vf = bool(
        re.search(
            r"(__simd_vf__|__no_simd_vf_fusion__|asc_vf_call|T\.SimdVF|simdvf|vf_call)",
            combined,
            flags=re.IGNORECASE,
        )
    )
    has_membar = bool(
        re.search(
            r"(LocalMemBar|local_membar|pipe_barrier\s*\(\s*PIPE_ALL|PIPE_ALL)",
            combined,
            flags=re.IGNORECASE,
        )
    )
    return has_vf and has_membar


def _detect_vf_fusion_breaks(res_text: str = "", asm_or_dump_text: str = "") -> int:
    """Count VF fusion breaks, ignoring comments and prototypes that double-count launches."""
    combined = "\n".join(part for part in (res_text, asm_or_dump_text) if part)
    explicit = _match_int((r"\bvf_fusion_breaks\s*[:=]\s*(\d+)",), combined)
    if explicit is not None:
        return max(0, explicit)
    raw_src = asm_or_dump_text if asm_or_dump_text.strip() else combined
    src_no_comments = re.sub(r"/\*.*?\*/", " ", raw_src, flags=re.DOTALL)
    src_no_comments = re.sub(r"//[^\n]*", "", src_no_comments)
    src_lines = [
        ln
        for ln in src_no_comments.splitlines()
        if not re.match(r"^\s*(?:#\s*(?:define|include)|extern\b)", ln)
    ]
    clean_src = "\n".join(src_lines)
    explicit_markers = re.findall(r"\bvf_fusion_break\b", clean_src, flags=re.IGNORECASE)
    call_launches = re.findall(
        r"(?:\basc_vf_call\s*(?:<[^>\n]*>\s*)?\(|\bT\.SimdVF\s*\(|\bSimdVF\s*\()",
        clean_src,
        flags=re.IGNORECASE,
    )
    if call_launches:
        launch_count = len(call_launches)
    else:
        launch_count = len(re.findall(r"\b__simd_vf__\b", clean_src, flags=re.IGNORECASE))
    if launch_count >= 2:
        return (launch_count - 1) + len(explicit_markers)
    return len(explicit_markers)


def _detect_alias_serialized_buffers(res_text: str = "", asm_or_dump_text: str = "") -> int:
    combined = "\n".join(part for part in (res_text, asm_or_dump_text) if part)
    explicit = _match_int((r"\balias_serialized_buffers\s*[:=]\s*(\d+)",), combined)
    if explicit is not None:
        return max(0, explicit)
    indicators = re.findall(
        r"(?:enable_offset\s*[:=]\s*(?:true|1|yes)"
        r"|\bmay_alias\b|\balias_hazard\b|\bbuffer_aliasing\b"
        r"|\bserialized_ping_pong\b|\bping_pong_serialized\b"
        r"|\bdegraded_single_buffer\b|\bsingle_buffer_fallback\b"
        r"|\bunrestricted_alias\b|\bdynamic_offset\b)",
        combined,
        flags=re.IGNORECASE,
    )
    return len(indicators)


def _detect_dual_issue_ipc_collapse(res_text: str = "", asm_or_dump_text: str = "") -> float:
    combined = "\n".join(part for part in (res_text, asm_or_dump_text) if part)
    explicit = _match_float(
        (r"\bdual_issue_ipc_collapse\s*[:=]\s*([0-9]*\.?[0-9]+)",),
        combined,
    )
    if explicit is not None:
        return max(0.0, min(1.0, explicit))

    src = asm_or_dump_text if asm_or_dump_text.strip() else combined
    lines = [ln.strip() for ln in src.splitlines() if ln.strip()]

    raw_fma_chains = 0
    prev_fma_dst: str | None = None
    fma_run = 0
    fma_re = re.compile(
        r"^\s*(?:vmad|vfma|vmula|vmla|Reg::MulAdd|T\.simd\.vmula)\b[\s(]+([A-Za-z_][A-Za-z0-9_\[\]]*)",
        flags=re.IGNORECASE,
    )

    sfu_chains = 0
    sfu_run = 0
    sfu_re = re.compile(
        r"\b(?:vexp|vsqrt|vdiv|vln|vrec|T\.simd\.v(?:exp|sqrt|div|ln)|Reg::(?:Exp|Sqrt|Div))\b",
        flags=re.IGNORECASE,
    )

    for ln in lines:
        m_fma = fma_re.search(ln)
        if m_fma:
            dst = m_fma.group(1).lower()
            if dst == prev_fma_dst:
                fma_run += 1
                if fma_run == 2:
                    raw_fma_chains += 1
            else:
                prev_fma_dst = dst
                fma_run = 1
        else:
            prev_fma_dst = None
            fma_run = 0

        if sfu_re.search(ln):
            sfu_run += 1
            if sfu_run == 2:
                sfu_chains += 1
        else:
            sfu_run = 0

    explicit_markers = len(
        re.findall(
            r"\b(?:raw_fma_chain|single_accumulator|sfu_stall|dual_issue_collapse)\b",
            combined,
            flags=re.IGNORECASE,
        )
    )
    severity = 0.35 * raw_fma_chains + 0.30 * sfu_chains + 0.35 * explicit_markers
    return max(0.0, min(1.0, round(severity, 4)))


def _detect_dcache_dcci_flushes(res_text: str = "", asm_or_dump_text: str = "") -> int:
    combined = "\n".join(part for part in (res_text, asm_or_dump_text) if part)
    explicit = _match_int((r"\bdcache_dcci_flushes\s*[:=]\s*(\d+)",), combined)
    if explicit is not None:
        return max(0, explicit)
    matches = re.findall(
        r"\b(?:dcci(?:_all)?|DataCacheCleanAndInvalid|dcache_invalidate)\b",
        combined,
        flags=re.IGNORECASE,
    )
    return len(matches)


def _strip_cpp_comments(text: str) -> str:
    stripped = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", stripped)


def _extract_for_loops(text: str) -> list[tuple[str, str]]:
    loops: list[tuple[str, str]] = []
    n = len(text)
    for match in re.finditer(r"\bfor\s*\(", text):
        header_start = match.end()
        pos = header_start
        paren_depth = 1
        while pos < n and paren_depth > 0:
            ch = text[pos]
            if ch == "(":
                paren_depth += 1
            elif ch == ")":
                paren_depth -= 1
            pos += 1
        if paren_depth != 0:
            continue
        header = text[header_start : pos - 1]
        while pos < n and text[pos].isspace():
            pos += 1
        if pos >= n:
            continue
        if text[pos] == "{":
            pos += 1
            start = pos
            brace_depth = 1
            while pos < n and brace_depth > 0:
                ch = text[pos]
                if ch == "{":
                    brace_depth += 1
                elif ch == "}":
                    brace_depth -= 1
                pos += 1
            if brace_depth == 0:
                loops.append((header, text[start : pos - 1]))
        else:
            semi = text.find(";", pos)
            if semi != -1:
                loops.append((header, text[pos : semi + 1]))
    return loops


def _extract_for_loop_bodies(text: str) -> list[str]:
    return [body for _, body in _extract_for_loops(text)]


def _strip_subscripts(text: str) -> str:
    out: list[str] = []
    depth = 0
    for ch in text:
        if ch == "[":
            if depth == 0:
                out.append("[]")
            depth += 1
        elif ch == "]":
            if depth > 0:
                depth -= 1
        elif depth == 0:
            out.append(ch)
    return "".join(out)


_VEC_MATRIX_INST_RE = re.compile(
    r"(?:simd_inst::|\bvloop_?v2\b|\basc_mmad(?:_mx)?\b|\bascend_gemm_l1\b"
    r"|\basc_simd_\w+|\basc_v(?!f_call\b)\w+|\bReg::|\bvec_(?!reg))",
    flags=re.IGNORECASE,
)

_SIMT_LAUNCH_BOUNDS_RE = re.compile(r"__launch_bounds__\s*\(\s*(\d+)\s*\)")
_SIMT_VF_CALL_DIM3_RE = re.compile(
    r"\basc_vf_call\s*<[^>]+>\s*\(\s*dim3\s*\(\s*(\d+)"
    r"(?:\s*,\s*(\d+)\s*(?:,\s*(\d+)\s*)?)?\s*\)"
)
_GM_PTR_NAME_RE = re.compile(
    r"__gm__\s+[^;(),={}]*?\*+\s*(?:__restrict__\s*)?([A-Za-z_]\w*)"
)
_UB_PTR_NAME_RE = re.compile(
    r"__ubuf__\s+[^;(),={}]*?\*+\s*(?:__restrict__\s*)?([A-Za-z_]\w*)"
)

_ASC_PRIM_CALL_RE = re.compile(r"\basc_\w+\s*(?:<[^;{}()]*>)?\s*\(")
_NESTED_FOR_RE = re.compile(r"\bfor\s*\(")

_SCALAR_COMPUTE_OP_RE = re.compile(
    r"(?:\+=|-=|\*=|/=|%=|&=|\|=|\^=|<<|>>|[+\-*/%&|^~]"
    r"|\b(?:expf?|sqrtf?|rsqrtf?|logf?|tanhf?|fabsf?|fmaxf?|fminf?|powf?|fmaf?)\s*\()",
)

_BYPASS_DCACHE_RE = re.compile(r"\b(?:tl::)?write_gm_bypass_dcache\b")


def _static_loop_bound_lt_threshold(header: str, threshold: int = 32) -> bool:
    parts = header.split(";")
    if len(parts) != 3:
        return False
    init_part, cond_part, step_part = (part.strip() for part in parts)
    cond_lt = re.fullmatch(r"([A-Za-z_]\w*)\s*(<=|<)\s*(\d+)[uUlL]*", cond_part)
    if cond_lt is not None:
        var = cond_lt.group(1)
        upper = int(cond_lt.group(3)) + (1 if cond_lt.group(2) == "<=" else 0)
    else:
        cond_gt = re.fullmatch(r"(\d+)[uUlL]*\s*(>=|>)\s*([A-Za-z_]\w*)", cond_part)
        if cond_gt is None:
            return False
        var = cond_gt.group(3)
        upper = int(cond_gt.group(1)) + (1 if cond_gt.group(2) == ">=" else 0)
    init_m = re.search(rf"\b{re.escape(var)}\s*=\s*(-?\d+)[uUlL]*\b", init_part)
    if init_m is None:
        return False
    start = int(init_m.group(1))
    if re.fullmatch(rf"(?:\+\+\s*{re.escape(var)}|{re.escape(var)}\s*\+\+)", step_part):
        step = 1
    else:
        add_m = re.fullmatch(rf"{re.escape(var)}\s*\+=\s*(\d+)[uUlL]*", step_part)
        if add_m is None:
            return False
        step = int(add_m.group(1))
    if step <= 0:
        return False
    trips = max(0, (upper - start + step - 1) // step)
    return trips < threshold

_TYPE_BYTES_PATTERNS: tuple[tuple[re.Pattern[str], float], ...] = (
    (re.compile(r"\b(?:u?int8(?:_t)?|fp8\w*|float8\w*|e4m3\w*|e5m2\w*|char|bool)\b"), 1.0),
    (
        re.compile(
            r"\b(?:half|__fp16|fp16|float16|bfloat16|bf16|__bf16|u?int16(?:_t)?)\b"
        ),
        2.0,
    ),
    (re.compile(r"\b(?:float32|fp32|float|u?int32(?:_t)?|int)\b"), 4.0),
    (re.compile(r"\b(?:double|u?int64(?:_t)?)\b"), 8.0),
)


def _elem_bytes_from_type_str(type_str: str) -> float | None:
    low = type_str.lower()
    for pattern, size in _TYPE_BYTES_PATTERNS:
        if pattern.search(low):
            return size
    return None


def _infer_gm_elem_bytes(text: str, target_ptrs: set[str] | None = None) -> float:
    tmpl = re.search(r"\b(?:tl::)?write_gm_bypass_dcache\s*<\s*([A-Za-z0-9_]+)", text)
    if tmpl:
        inferred = _elem_bytes_from_type_str(tmpl.group(1))
        if inferred is not None:
            return inferred
    for decl_match in re.finditer(
        r"__gm__\s+([^;(),={}]*?)\*+\s*(?:__restrict__\s*)?([A-Za-z_]\w*)",
        text,
    ):
        type_part, var_name = decl_match.group(1), decl_match.group(2)
        if target_ptrs and var_name not in target_ptrs:
            continue
        inferred = _elem_bytes_from_type_str(type_part)
        if inferred is not None:
            return inferred
    inferred_any = _elem_bytes_from_type_str(text)
    if inferred_any is not None:
        return inferred_any
    return 2.0


_CPP_CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*(?:<[^;{}()]*>)?\s*\(")
_CPP_TMPL_ARGS_RE = re.compile(r"\b[A-Za-z_]\w*\s*<([^;{}()]+)>\s*\(")
_CPP_LAUNCH_RE = re.compile(r"\b([A-Za-z_]\w*)\s*(?:<[^;{}]*>)?\s*<<<")


def _extract_header_fn_name(sig_head: str) -> str:
    cleaned = re.sub(r"\b(?:const|noexcept|override|final)\s*$", "", sig_head.strip()).rstrip()
    if not cleaned.endswith(")"):
        return ""
    depth = 0
    open_idx = -1
    for idx in range(len(cleaned) - 1, -1, -1):
        if cleaned[idx] == ")":
            depth += 1
        elif cleaned[idx] == "(":
            depth -= 1
            if depth == 0:
                open_idx = idx
                break
    if open_idx <= 0:
        return ""
    ident = re.search(r"([A-Za-z_]\w*)\s*$", cleaned[:open_idx].rstrip())
    if ident is None:
        return ""
    name = ident.group(1)
    if name in ("if", "for", "while", "switch", "decltype", "__launch_bounds__", "__mix__"):
        return ""
    return name


def _extract_reachable_trackb_text(src: str) -> str:
    """Return the launched __global__ kernel and its transitive callee bodies."""
    masked = _strip_cpp_comments(src)
    blocks: list[tuple[str, bool, str]] = []
    depth = 0
    stmt_start = 0
    i = 0
    n = len(masked)
    while i < n:
        ch = masked[i]
        if ch == ";" and depth == 0:
            stmt_start = i + 1
        elif ch == "{" and depth == 0:
            sig_head = masked[stmt_start:i].strip()
            fn_name = _extract_header_fn_name(sig_head)
            brace_depth = 1
            j = i + 1
            while j < n and brace_depth > 0:
                if masked[j] == "{":
                    brace_depth += 1
                elif masked[j] == "}":
                    brace_depth -= 1
                j += 1
            if fn_name:
                blocks.append((fn_name, "__global__" in sig_head, src[stmt_start:j]))
            stmt_start = j
            i = j
            continue
        elif ch == "{":
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
            if depth == 0:
                stmt_start = i + 1
        i += 1

    kernels = {name for name, is_global, _ in blocks if is_global}
    if not kernels:
        return src
    launched = [m.group(1) for m in _CPP_LAUNCH_RE.finditer(masked) if m.group(1) in kernels]
    entry = launched[0] if launched else next(name for name, is_global, _ in blocks if is_global)
    by_name = {name: text for name, _, text in blocks}
    reached: set[str] = set()
    pending = [entry]
    while pending:
        cur_name = pending.pop()
        if cur_name in reached or cur_name not in by_name:
            continue
        reached.add(cur_name)
        body_masked = _strip_cpp_comments(by_name[cur_name])
        for cm in _CPP_CALL_RE.finditer(body_masked):
            if cm.group(1) in by_name and cm.group(1) not in reached:
                pending.append(cm.group(1))
        for tm in _CPP_TMPL_ARGS_RE.finditer(body_masked):
            for ident in re.findall(r"\b([A-Za-z_]\w*)\b", tm.group(1)):
                if ident in by_name and ident not in reached:
                    pending.append(ident)
    kept = [text for name, _, text in blocks if name in reached]
    return "\n\n".join(kept) if kept else src


def _has_parallel_simt_kernel(cleaned: str) -> bool:
    if not re.search(r"\b__simt_vf__\b", cleaned):
        return False
    bounds = [int(m.group(1)) for m in _SIMT_LAUNCH_BOUNDS_RE.finditer(cleaned)]
    dim3_threads = [
        int(m.group(1)) * int(m.group(2) or 1) * int(m.group(3) or 1)
        for m in _SIMT_VF_CALL_DIM3_RE.finditer(cleaned)
    ]
    if bounds and max(bounds) < 32:
        return False
    if dim3_threads and max(dim3_threads) < 32:
        return False
    return bool((bounds and max(bounds) >= 32) or (dim3_threads and max(dim3_threads) >= 32))


def count_reachable_ub_scalar_compute_loops(cpp_source: str) -> int:
    """Count innermost reachable scalar compute loops operating on __ubuf__ buffers."""
    if not cpp_source or not cpp_source.strip():
        return 0
    src = (
        _extract_reachable_trackb_text(cpp_source)
        if re.search(r"\b__global__\b", cpp_source)
        else cpp_source
    )
    cleaned = _strip_cpp_comments(src)
    if _has_parallel_simt_kernel(cleaned):
        return 0
    ub_ptrs = set(_UB_PTR_NAME_RE.findall(cleaned))
    if not ub_ptrs:
        return 0
    ptr_alt = "|".join(re.escape(p) for p in sorted(ub_ptrs))
    buf_access_re = re.compile(
        rf"\b(?:{ptr_alt})\s*\[[^\]]+\]|\*\s*\(?\s*(?:{ptr_alt})\b"
    )
    count = 0
    for header, body in _extract_for_loops(cleaned):
        if _NESTED_FOR_RE.search(body):
            continue
        if _VEC_MATRIX_INST_RE.search(body) or _ASC_PRIM_CALL_RE.search(body):
            continue
        if _static_loop_bound_lt_threshold(header, 32):
            continue
        if not buf_access_re.search(body):
            continue
        norm_body = body.replace("++", " ").replace("--", " ").replace("->", " ")
        body_no_idx = _strip_subscripts(norm_body)
        if _SCALAR_COMPUTE_OP_RE.search(body_no_idx):
            count += 1
    return count


def _detect_scalar_compute_loop_fallback(
    res_text: str = "",
    asm_or_dump_text: str = "",
) -> dict[str, float]:
    combined = "\n".join(part for part in (res_text, asm_or_dump_text) if part)
    explicit = _match_float(
        (r"\bscalar_compute_loop_fallback\s*[:=]\s*([0-9]*\.?[0-9]+)",),
        combined,
    )
    if explicit is not None:
        return {"scalar_compute_loop_fallback": float(explicit)} if explicit > 0.0 else {}
    if re.search(
        r"\bscalar_compute_loop_fallback\s*[:=]\s*(true|yes)",
        combined,
        flags=re.IGNORECASE,
    ):
        return {"scalar_compute_loop_fallback": 1.0}

    src = asm_or_dump_text if asm_or_dump_text.strip() else combined
    if re.search(r"\b__global__\b", src):
        src = _extract_reachable_trackb_text(src)
    ub_loops = count_reachable_ub_scalar_compute_loops(src)
    if ub_loops > 0:
        return {"scalar_compute_loop_fallback": float(ub_loops)}
    cleaned = _strip_cpp_comments(src)
    if _VEC_MATRIX_INST_RE.search(cleaned) or _has_parallel_simt_kernel(cleaned):
        return {}

    gm_ptrs = set(_GM_PTR_NAME_RE.findall(cleaned))
    gm_alt = "|".join(re.escape(p) for p in sorted(gm_ptrs))
    gm_write_re = (
        re.compile(
            rf"(?:\b(?:{gm_alt})\s*\[[^\]]+\]|\*\s*\(?\s*(?:{gm_alt})[^;={{}}]*\))"
            r"\s*(?:[+\-*/%&|^]?=)(?!=)"
        )
        if gm_ptrs
        else None
    )
    generic_write_re = re.compile(
        r"(?:\b[A-Za-z_]\w*\s*\[[^\]]+\]|\*\s*\([^)]+\))\s*(?:[+\-*/%&|^]?=)(?!=)"
    )

    for header, body in _extract_for_loops(cleaned):
        if _NESTED_FOR_RE.search(body):
            continue
        if _VEC_MATRIX_INST_RE.search(body) or _ASC_PRIM_CALL_RE.search(body):
            continue
        if _static_loop_bound_lt_threshold(header, 32):
            continue
        has_write = (
            bool(gm_write_re.search(body))
            if gm_write_re is not None
            else bool(generic_write_re.search(body))
        )
        if not has_write:
            continue
        norm_body = body.replace("++", " ").replace("--", " ").replace("->", " ")
        body_no_idx = _strip_subscripts(norm_body)
        if _SCALAR_COMPUTE_OP_RE.search(body_no_idx):
            return {"scalar_compute_loop_fallback": 1.0}
    return {}


def detect_scalar_compute_loop_fallback(
    tir_ledger: ObjectivePhysicalLedger | None = None,
    *,
    ub_scalar_compute_loops: int = 0,
    cpp_source: str = "",
    extra_signals: Mapping[str, object] | None = None,
) -> float:
    """Return scalar-compute fallback severity from ledger, C++ source, and signals."""
    loop_count = max(0, int(ub_scalar_compute_loops))
    if tir_ledger is not None:
        loop_count = max(loop_count, max(0, int(tir_ledger.ub_scalar_compute_loops)))
    explicit = 0.0
    if extra_signals is not None:
        raw_sig = extra_signals.get("scalar_compute_loop_fallback", 0.0)
        if isinstance(raw_sig, (int, float)) and not isinstance(raw_sig, bool):
            explicit = max(explicit, float(raw_sig))
        raw_loops = extra_signals.get("ub_scalar_compute_loops", 0)
        if isinstance(raw_loops, (int, float)) and not isinstance(raw_loops, bool):
            loop_count = max(loop_count, max(0, int(raw_loops)))
    if cpp_source:
        cpp_loops = count_reachable_ub_scalar_compute_loops(cpp_source)
        loop_count = max(loop_count, cpp_loops)
        if loop_count == 0:
            detected = _detect_scalar_compute_loop_fallback(asm_or_dump_text=cpp_source)
            explicit = max(explicit, float(detected.get("scalar_compute_loop_fallback", 0.0)))
    return max(explicit, float(loop_count))


def _detect_dcache_bypass_narrow_store(
    res_text: str = "",
    asm_or_dump_text: str = "",
) -> dict[str, float]:
    combined = "\n".join(part for part in (res_text, asm_or_dump_text) if part)
    explicit = _match_float(
        (r"\bdcache_bypass_narrow_store\s*[:=]\s*([0-9]*\.?[0-9]+)",),
        combined,
    )
    if explicit is not None:
        if explicit <= 0.0:
            return {}
        return {"dcache_bypass_narrow_store": max(4.0, min(64.0, float(explicit)))}

    src = asm_or_dump_text if asm_or_dump_text.strip() else combined
    cleaned = _strip_cpp_comments(src)
    if _BYPASS_DCACHE_RE.search(cleaned):
        elem_bytes = _infer_gm_elem_bytes(cleaned)
        amp = max(4.0, min(64.0, 64.0 / elem_bytes))
        return {"dcache_bypass_narrow_store": amp}

    gm_ptrs = set(
        re.findall(
            r"__gm__\s+[^;(),={}]*?\*+\s*(?:__restrict__\s*)?([A-Za-z_]\w*)",
            cleaned,
        )
    )
    fallback_gm_ptrs = set(re.findall(r"\b(gm_[A-Za-z0-9_]+|[A-Za-z0-9_]+_gm)\b", cleaned))
    candidate_ptrs = gm_ptrs | fallback_gm_ptrs
    if not candidate_ptrs:
        return {}

    matched_ptrs: set[str] = set()
    ptr_alt = "|".join(re.escape(p) for p in sorted(candidate_ptrs))
    store_re = re.compile(
        rf"\b({ptr_alt})\s*\[[^\]]+\]\s*(?:[+\-*/%&|^]?=)(?!=)"
        rf"|\*\s*\(\s*({ptr_alt})\b[^)]*\)\s*(?:[+\-*/%&|^]?=)(?!=)"
    )
    for body in _extract_for_loop_bodies(cleaned):
        for match in store_re.finditer(body):
            ptr_name = match.group(1) or match.group(2)
            if ptr_name:
                matched_ptrs.add(ptr_name)

    if not matched_ptrs:
        return {}
    elem_bytes = _infer_gm_elem_bytes(cleaned, target_ptrs=matched_ptrs)
    if elem_bytes >= 32.0:
        return {}
    amp = max(4.0, min(64.0, 64.0 / elem_bytes))
    return {"dcache_bypass_narrow_store": amp}


def _detect_spill_and_stack(res_text: str = "", asm_or_dump_text: str = "") -> dict[str, int]:
    combined = "\n".join(part for part in (res_text, asm_or_dump_text) if part)
    vec_regs = _match_int(
        (
            r"(?:vector\s+register\s+usage|used\s+vector\s+registers?|number\s+of\s+vector\s+registers?|vec_regs?|vecregs?)\s*[:=]\s*(\d+)",
            r"\bvec_reg_num\s*[:=]\s*(\d+)",
        ),
        combined,
    )
    scalar_regs = _match_int(
        (
            r"(?:scalar\s+register\s+usage|used\s+scalar\s+registers?|number\s+of\s+scalar\s+registers?|scalar_regs?|scalarregs?)\s*[:=]\s*(\d+)",
            r"\bscalar_reg_num\s*[:=]\s*(\d+)",
        ),
        combined,
    )
    spill_bytes = _match_int(
        (
            r"(?:register\s+spill|stack\s+spill\s+size|spill\s+stack\s+size|spill_bytes|spillsize|spill\s+size|spill)\s*[:=]\s*(\d+)\s*(bytes?|kib|mib|kb|mb|b)?",
        ),
        combined,
    )
    stack_bytes = _match_int(
        (
            r"(?:stack\s+frame\s+size|stack_bytes|stacksize|stack\s+size|frame\s+size)\s*[:=]\s*(\d+)\s*(bytes?|kib|mib|kb|mb|b)?",
        ),
        combined,
    )
    return {
        "vec_regs": vec_regs or 0,
        "scalar_regs": scalar_regs or 0,
        "spill_bytes": spill_bytes or 0,
        "stack_bytes": stack_bytes or 0,
    }


_VLOOPV2_MISS_DIAG_RE = re.compile(
    r"(?:cannot\s+use\s+vloop_?v2|vloop_?v2\s*(?:miss|failed|fallback|disabled|not\s+applied)|"
    r"vloop_?v2\s*[:=]\s*(?:false|0|no|miss))",
    flags=re.IGNORECASE,
)


def _detect_loop_vectorization(res_text: str = "", asm_or_dump_text: str = "") -> dict[str, int]:
    combined = "\n".join(part for part in (res_text, asm_or_dump_text) if part)
    explicit_vloopv2 = _match_int((r"(?:vloopv2_loops|vloopv2\s+loops?)\s*[:=]\s*(\d+)",), combined)
    if explicit_vloopv2 is not None:
        vloopv2_loops = explicit_vloopv2
    else:
        asm_src = asm_or_dump_text if asm_or_dump_text.strip() else combined
        cleaned_src = _VLOOPV2_MISS_DIAG_RE.sub(" ", asm_src)
        matches = re.findall(r"\bVLOOP_?V2\b(?!\s*[:=]\s*0)", cleaned_src, flags=re.IGNORECASE)
        vloopv2_loops = len(matches)

    explicit_sw_loops = _match_int(
        (r"(?:software_loops|software\s+loops?|scalar\s+loops?)\s*[:=]\s*(\d+)",),
        combined,
    )
    if explicit_sw_loops is not None:
        software_loops = explicit_sw_loops
    else:
        asm_src = asm_or_dump_text if asm_or_dump_text.strip() else combined
        sw_markers = re.findall(
            r"(?:software\s+loop|unvectorized\s+loop|FALLBACK_SW_LOOP|\bVLOOP\b(?!_?V2))",
            asm_src,
            flags=re.IGNORECASE,
        )
        miss_markers = _VLOOPV2_MISS_DIAG_RE.findall(asm_src)
        back_branches = re.findall(
            r"\b(?:b|b\.[a-z]{2}|cbn?z|tbn?z|jump)\s+[^\n]*?\.L(?:BB)?[0-9_]+",
            asm_src,
            flags=re.IGNORECASE,
        )
        if sw_markers or miss_markers:
            software_loops = len(sw_markers) + len(miss_markers)
        elif back_branches and vloopv2_loops == 0:
            software_loops = len(back_branches)
        else:
            software_loops = 0
    return {"vloopv2_loops": vloopv2_loops, "software_loops": software_loops}


def _penalty_spill_and_stack(summary: BishengResourceSummary, ctx: Mapping[str, Any]) -> float:
    base_ref = float(ctx.get("base_ref", 1.0))
    penalty = 0.0
    if summary.spill_bytes > 0:
        spill_factor = min(1.5, 0.20 + 0.004 * float(summary.spill_bytes))
        penalty += base_ref * spill_factor + (float(summary.spill_bytes) * 0.01)
    if summary.stack_bytes > 1024 and summary.spill_bytes == 0:
        penalty += min(base_ref * 0.25, float(summary.stack_bytes) * 0.002)
    return penalty


def _penalty_software_loop(summary: BishengResourceSummary, ctx: Mapping[str, Any]) -> float:
    if summary.software_loops > 0 and summary.vloopv2_loops == 0:
        base_ref = float(ctx.get("base_ref", 1.0))
        vec_floor_us = float(ctx.get("vec_floor_us", 0.0))
        sw_factor = min(2.0, 0.35 * float(summary.software_loops))
        return max(vec_floor_us, base_ref * 0.5) * sw_factor
    return 0.0


def _penalty_local_membar(summary: BishengResourceSummary, ctx: Mapping[str, Any]) -> float:
    if summary.has_local_membar_in_vf_call:
        vec_floor_us = float(ctx.get("vec_floor_us", 0.0))
        return max(1.0, vec_floor_us * 0.25)
    return 0.0


def _penalty_vf_fusion_breaks(summary: BishengResourceSummary, ctx: Mapping[str, Any]) -> float:
    if summary.vf_fusion_breaks <= 0:
        return 0.0
    base_ref = float(ctx.get("base_ref", 1.0))
    vec_floor_us = float(ctx.get("vec_floor_us", 0.0))
    per_break = max(1.0, max(vec_floor_us * 0.18, base_ref * 0.12))
    return per_break * float(min(6, summary.vf_fusion_breaks))


def _penalty_alias_serialized(summary: BishengResourceSummary, ctx: Mapping[str, Any]) -> float:
    if summary.alias_serialized_buffers <= 0:
        return 0.0
    base_ref = float(ctx.get("base_ref", 1.0))
    vec_floor_us = float(ctx.get("vec_floor_us", 0.0))
    hbm_floor_us = float(ctx.get("hbm_floor_us", 0.0))
    scale = min(2.5, 0.30 * float(summary.alias_serialized_buffers))
    return max(hbm_floor_us, vec_floor_us, base_ref * 0.25) * scale


def _penalty_dual_issue_collapse(summary: BishengResourceSummary, ctx: Mapping[str, Any]) -> float:
    sev = max(0.0, min(1.0, float(summary.dual_issue_ipc_collapse)))
    if sev <= 0.0:
        return 0.0
    base_ref = float(ctx.get("base_ref", 1.0))
    vec_floor_us = float(ctx.get("vec_floor_us", 0.0))
    return max(vec_floor_us, base_ref * 0.20) * 0.45 * sev


def _penalty_dcache_dcci(summary: BishengResourceSummary, ctx: Mapping[str, Any]) -> float:
    if summary.dcache_dcci_flushes <= 0:
        return 0.0
    base_ref = float(ctx.get("base_ref", 1.0))
    return min(2.0 * base_ref, float(summary.dcache_dcci_flushes) * max(1.5, 0.08 * base_ref))


def _ctx_float(ctx: Mapping[str, object], key: str, default: float = 0.0) -> float:
    raw = ctx.get(key)
    if isinstance(raw, bool):
        return default
    if isinstance(raw, (int, float)):
        return float(raw)
    return default


def _penalty_scalar_compute_loop_fallback(
    summary: BishengResourceSummary,
    ctx: Mapping[str, object],
) -> float:
    severity = float(summary.extra_signals.get("scalar_compute_loop_fallback", 0.0) or 0.0)
    if severity <= 0.0:
        return 0.0
    base_ref = max(1.0, _ctx_float(ctx, "base_ref", 1.0))
    vec_floor_us = max(0.0, _ctx_float(ctx, "vec_floor_us", 0.0))
    if vec_floor_us > 0.0:
        return min(1.5 * vec_floor_us, 0.35 * severity * max(vec_floor_us, 1.0))
    return min(1.5 * base_ref, 0.35 * severity * base_ref)


def _penalty_dcache_bypass_narrow_store(
    summary: BishengResourceSummary,
    ctx: Mapping[str, object],
) -> float:
    amp = float(summary.extra_signals.get("dcache_bypass_narrow_store", 0.0) or 0.0)
    if amp <= 1.0:
        return 0.0
    base_ref = max(1.0, _ctx_float(ctx, "base_ref", 1.0))
    hbm_floor_us = max(0.0, _ctx_float(ctx, "hbm_floor_us", 0.0))
    eff_hbm = hbm_floor_us if hbm_floor_us > 0.0 else base_ref * 0.25
    return eff_hbm * 0.5 * max(0.0, amp - 1.0)


def _invoke_parse_fn(parse_fn: Callable[..., Any], res_text: str, asm_or_dump_text: str) -> Any:
    try:
        sig = inspect.signature(parse_fn)
        pos_params = [
            p
            for p in sig.parameters.values()
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())
        if len(pos_params) == 1 and not has_varargs:
            combined = "\n".join(part for part in (res_text, asm_or_dump_text) if part)
            return parse_fn(combined)
    except (ValueError, TypeError):
        pass
    return parse_fn(res_text, asm_or_dump_text)


def _invoke_penalty_fn(
    penalty_fn: Callable[..., float],
    summary: BishengResourceSummary,
    ctx: Mapping[str, Any],
) -> float:
    try:
        sig = inspect.signature(penalty_fn)
        pos_params = [
            p
            for p in sig.parameters.values()
            if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        ]
        has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values())
        if len(pos_params) == 1 and not has_varargs:
            p_name = pos_params[0].name.lower()
            if p_name in ("ctx", "context"):
                return float(penalty_fn(ctx))
            return float(penalty_fn(summary))
    except (ValueError, TypeError):
        pass
    return float(penalty_fn(summary, ctx))


@dataclass(frozen=True)
class CompilerBehaviorDetector:
    """One compiler-level phenomenon parsed from dumps and penalized in Tier-1."""

    name: str
    parse_fn: Callable[..., Any]
    penalty_fn: Callable[..., float]
    summary_field: str | None = None
    description: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("CompilerBehaviorDetector.name must be a non-empty string")
        if not callable(self.parse_fn):
            raise TypeError("CompilerBehaviorDetector.parse_fn must be callable")
        if not callable(self.penalty_fn):
            raise TypeError("CompilerBehaviorDetector.penalty_fn must be callable")


class CompilerBehaviorRegistry:
    """Extensible registry of Tier-1 compiler behavior parsers and latency penalty hooks."""

    def __init__(self, detectors: list[CompilerBehaviorDetector] | None = None) -> None:
        self._detectors: dict[str, CompilerBehaviorDetector] = {}
        if detectors:
            for det in detectors:
                self._detectors[det.name] = det

    def register(
        self,
        detector_or_name: CompilerBehaviorDetector | str,
        parse_fn: Callable[..., Any] | None = None,
        penalty_fn: Callable[..., float] | None = None,
        *,
        summary_field: str | None = None,
        description: str = "",
        overwrite: bool = True,
    ) -> CompilerBehaviorDetector:
        """Register a built-in or custom compiler behavior detector."""
        if isinstance(detector_or_name, CompilerBehaviorDetector):
            det = detector_or_name
        else:
            if parse_fn is None or penalty_fn is None:
                raise ValueError("parse_fn and penalty_fn are required when registering by name")
            det = CompilerBehaviorDetector(
                name=str(detector_or_name),
                parse_fn=parse_fn,
                penalty_fn=penalty_fn,
                summary_field=summary_field,
                description=description,
            )
        if not overwrite and det.name in self._detectors:
            raise KeyError(f"Compiler detector {det.name!r} is already registered")
        self._detectors[det.name] = det
        return det

    def unregister(self, name: str) -> bool:
        """Remove a detector by name if present."""
        return self._detectors.pop(name, None) is not None

    def get(self, name: str) -> CompilerBehaviorDetector | None:
        """Return detector by name, or None."""
        return self._detectors.get(name)

    def list_detectors(self) -> tuple[str, ...]:
        """Return ordered detector names."""
        return tuple(self._detectors.keys())

    @property
    def detectors(self) -> tuple[CompilerBehaviorDetector, ...]:
        """Return all registered detectors in registration order."""
        return tuple(self._detectors.values())

    def clone(self) -> CompilerBehaviorRegistry:
        """Return an independent shallow copy of this registry."""
        return CompilerBehaviorRegistry(list(self._detectors.values()))

    def parse(self, res_text: str = "", asm_or_dump_text: str = "") -> BishengResourceSummary:
        """Run registered detectors over compiler, assembly, or IR text."""
        combined = "\n".join(part for part in (res_text, asm_or_dump_text) if part)
        if not combined.strip():
            return BishengResourceSummary()

        fields: dict[str, Any] = {
            "vec_regs": 0,
            "scalar_regs": 0,
            "spill_bytes": 0,
            "stack_bytes": 0,
            "vloopv2_loops": 0,
            "software_loops": 0,
            "has_local_membar_in_vf_call": False,
            "vf_fusion_breaks": 0,
            "alias_serialized_buffers": 0,
            "dual_issue_ipc_collapse": 0.0,
            "dcache_dcci_flushes": 0,
        }
        extra_signals: dict[str, float] = {}

        for det in self._detectors.values():
            raw = _invoke_parse_fn(det.parse_fn, res_text, asm_or_dump_text)
            if isinstance(raw, Mapping):
                for k, v in raw.items():
                    sk = str(k)
                    if sk in fields:
                        fields[sk] = v
                    elif v is not None and float(v) != 0.0:
                        extra_signals[sk] = float(v)
            elif det.summary_field and det.summary_field in fields:
                fields[det.summary_field] = raw
            elif raw is not None:
                val = float(raw) if not isinstance(raw, bool) else (1.0 if raw else 0.0)
                if val != 0.0:
                    extra_signals[det.name] = val

        return BishengResourceSummary(
            vec_regs=int(fields["vec_regs"] or 0),
            scalar_regs=int(fields["scalar_regs"] or 0),
            spill_bytes=int(fields["spill_bytes"] or 0),
            stack_bytes=int(fields["stack_bytes"] or 0),
            vloopv2_loops=int(fields["vloopv2_loops"] or 0),
            software_loops=int(fields["software_loops"] or 0),
            has_local_membar_in_vf_call=bool(fields["has_local_membar_in_vf_call"]),
            vf_fusion_breaks=max(0, int(fields["vf_fusion_breaks"] or 0)),
            alias_serialized_buffers=max(0, int(fields["alias_serialized_buffers"] or 0)),
            dual_issue_ipc_collapse=max(
                0.0, min(1.0, float(fields["dual_issue_ipc_collapse"] or 0.0))
            ),
            dcache_dcci_flushes=max(0, int(fields["dcache_dcci_flushes"] or 0)),
            extra_signals=extra_signals,
        )

    def compute_tier1_penalties(
        self,
        summary: BishengResourceSummary,
        context: Mapping[str, Any] | None = None,
    ) -> tuple[float, dict[str, float]]:
        """Evaluate every registered detector's Tier-1 penalty and return (total_us, breakdown)."""
        base_ctx = dict(context or {})
        total = 0.0
        breakdown: dict[str, float] = {}
        for det in self._detectors.values():
            det_ctx = dict(base_ctx)
            if det.summary_field and hasattr(summary, det.summary_field):
                det_ctx["signal"] = getattr(summary, det.summary_field)
            else:
                det_ctx["signal"] = summary.extra_signals.get(det.name, 0.0)
            val = max(0.0, _invoke_penalty_fn(det.penalty_fn, summary, det_ctx))
            if val > 0.0:
                breakdown[det.name] = val
                total += val
        return total, breakdown


def _build_default_registry() -> CompilerBehaviorRegistry:
    reg = CompilerBehaviorRegistry()
    reg.register(
        "register_stack_spill",
        _detect_spill_and_stack,
        _penalty_spill_and_stack,
        summary_field="spill_bytes",
        description="(a) Vector/scalar register pressure and stack spill bytes",
    )
    reg.register(
        "loop_vectorization_vloopv2",
        _detect_loop_vectorization,
        _penalty_software_loop,
        summary_field="software_loops",
        description="(b) Hardware VLOOPv2 zero-overhead loop hit vs software branch fallback",
    )
    reg.register(
        "vf_local_membar",
        _detect_local_membar_in_vf,
        _penalty_local_membar,
        summary_field="has_local_membar_in_vf_call",
        description="(c) LocalMemBar/PIPE_ALL serialization inside SimdVF / asc_vf_call",
    )
    reg.register(
        "vf_fusion_breaks",
        _detect_vf_fusion_breaks,
        _penalty_vf_fusion_breaks,
        summary_field="vf_fusion_breaks",
        description="(d) Fragmented asc_vf_call/SimdVF launches forcing UB store->load roundtrips",
    )
    reg.register(
        "alias_serialized_buffers",
        _detect_alias_serialized_buffers,
        _penalty_alias_serialized,
        summary_field="alias_serialized_buffers",
        description="(e) Dynamic offset / unrestricted pointer aliasing serializing MTE2/V/MTE3 ping-pong",
    )
    reg.register(
        "dual_issue_ipc_collapse",
        _detect_dual_issue_ipc_collapse,
        _penalty_dual_issue_collapse,
        summary_field="dual_issue_ipc_collapse",
        description="(f) Single-accumulator RAW FMA chains or back-to-back SFU chains collapsing IPC",
    )
    reg.register(
        "dcache_dcci_flushes",
        _detect_dcache_dcci_flushes,
        _penalty_dcache_dcci,
        summary_field="dcache_dcci_flushes",
        description="(g) Scalar dcci D-cache invalidation storms inside loops",
    )
    reg.register(
        "scalar_compute_loop_fallback",
        _detect_scalar_compute_loop_fallback,
        _penalty_scalar_compute_loop_fallback,
        description="(h) Scalar for-loop arithmetic/bitwise compute with zero vector/matrix ops",
    )
    reg.register(
        "dcache_bypass_narrow_store",
        _detect_dcache_bypass_narrow_store,
        _penalty_dcache_bypass_narrow_store,
        description="(i) Bypass-DCache or sub-32B elementwise GM stores causing 64B amplification",
    )
    return reg


DEFAULT_COMPILER_BEHAVIOR_REGISTRY: CompilerBehaviorRegistry = _build_default_registry()


def register_compiler_detector(
    name: str,
    parse_fn: Callable[..., Any],
    penalty_fn: Callable[..., float],
    *,
    registry: CompilerBehaviorRegistry | None = None,
    summary_field: str | None = None,
    description: str = "",
    overwrite: bool = True,
) -> CompilerBehaviorDetector:
    """Register a custom compiler behavior detector on *registry* (default: global registry)."""
    target_reg = registry if registry is not None else DEFAULT_COMPILER_BEHAVIOR_REGISTRY
    return target_reg.register(
        name,
        parse_fn,
        penalty_fn,
        summary_field=summary_field,
        description=description,
        overwrite=overwrite,
    )


def parse_bisheng_res_usage(
    res_text: str = "",
    asm_or_dump_text: str = "",
    *,
    registry: CompilerBehaviorRegistry | None = None,
) -> BishengResourceSummary:
    """Parse ``bisheng --cce-res-usage`` output and optional ``.s`` / IR / source dumps."""
    active_registry = registry if registry is not None else DEFAULT_COMPILER_BEHAVIOR_REGISTRY
    return active_registry.parse(res_text, asm_or_dump_text)


parse_bisheng_resources = parse_bisheng_res_usage

__all__ = [
    "BishengResourceSummary",
    "CompilerBehaviorDetector",
    "CompilerBehaviorRegistry",
    "DEFAULT_COMPILER_BEHAVIOR_REGISTRY",
    "count_reachable_ub_scalar_compute_loops",
    "detect_scalar_compute_loop_fallback",
    "parse_bisheng_res_usage",
    "parse_bisheng_resources",
    "register_compiler_detector",
]
