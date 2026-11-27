"""Parse ``bisheng --cce-res-usage`` output and CCE assembly dumps.

Also covers the Tier-1 behavior registry.
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any

from tilefoundry_ascend.ascend.json_record import JsonRecordMixin
from tilefoundry_ascend.ascend.passes.physical_ledger import ObjectivePhysicalLedger

_UNIT_MULTIPLIER: dict[str, int] = {"": 1, "b": 1, "byte": 1, "bytes": 1, "kb": 1024, "kib": 1024, "mb": 1048576, "mib": 1048576}
_VEC_MATRIX_INST_RE = re.compile(r"(?:simd_inst::|\bvloop_?v2\b|\basc_mmad(?:_mx)?\b|\bascend_gemm_l1\b|\basc_simd_\w+|\basc_v(?!f_call\b)\w+|\bReg::|\bvec_(?!reg))", flags=re.IGNORECASE)
_SIMT_LAUNCH_BOUNDS_RE = re.compile(r"__launch_bounds__\s*\(\s*(\d+)\s*\)")
_SIMT_VF_CALL_DIM3_RE = re.compile(r"\basc_vf_call\s*<[^>]+>\s*\(\s*dim3\s*\(\s*(\d+)(?:\s*,\s*(\d+)\s*(?:,\s*(\d+)\s*)?)?\s*\)")
_GM_PTR_NAME_RE = re.compile(r"__gm__\s+[^;(),={}]*?\*+\s*(?:__restrict__\s*)?([A-Za-z_]\w*)")
_UB_PTR_NAME_RE = re.compile(r"__ubuf__\s+[^;(),={}]*?\*+\s*(?:__restrict__\s*)?([A-Za-z_]\w*)")
_ASC_PRIM_CALL_RE = re.compile(r"\basc_\w+\s*(?:<[^;{}()]*>)?\s*\(")
_NESTED_FOR_RE = re.compile(r"\bfor\s*\(")
_SCALAR_COMPUTE_OP_RE = re.compile(r"(?:\+=|-=|\*=|/=|%=|&=|\|=|\^=|<<|>>|[+\-*/%&|^~]|\b(?:expf?|sqrtf?|rsqrtf?|logf?|tanhf?|fabsf?|fmaxf?|fminf?|powf?|fmaf?)\s*\()")
_BYPASS_DCACHE_RE = re.compile(r"\b(?:tl::)?write_gm_bypass_dcache\b")
_VLOOPV2_MISS_DIAG_RE = re.compile(r"(?:cannot\s+use\s+vloop_?v2|vloop_?v2\s*(?:miss|failed|fallback|disabled|not\s+applied)|vloop_?v2\s*[:=]\s*(?:false|0|no|miss))", flags=re.IGNORECASE)
_CPP_CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*(?:<[^;{}()]*>)?\s*\(")
_CPP_TMPL_ARGS_RE = re.compile(r"\b[A-Za-z_]\w*\s*<([^;{}()]+)>\s*\(")
_CPP_LAUNCH_RE = re.compile(r"\b([A-Za-z_]\w*)\s*(?:<[^;{}]*>)?\s*<<<")
_TYPE_BYTES_PATTERNS: tuple[tuple[re.Pattern[str], float], ...] = (
    (re.compile(r"\b(?:u?int8(?:_t)?|fp8\w*|float8\w*|e4m3\w*|e5m2\w*|char|bool)\b"), 1.0),
    (re.compile(r"\b(?:half|__fp16|fp16|float16|bfloat16|bf16|__bf16|u?int16(?:_t)?)\b"), 2.0),
    (re.compile(r"\b(?:float32|fp32|float|u?int32(?:_t)?|int)\b"), 4.0),
    (re.compile(r"\b(?:double|u?int64(?:_t)?)\b"), 8.0),
)


@dataclass(frozen=True)
class BishengResourceSummary(JsonRecordMixin):
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

    def __post_init__(self) -> None:
        for attr in ("vf_fusion_breaks", "alias_serialized_buffers", "dcache_dcci_flushes"):
            object.__setattr__(self, attr, max(0, int(getattr(self, attr) or 0)))
        object.__setattr__(self, "dual_issue_ipc_collapse", max(0.0, min(1.0, float(self.dual_issue_ipc_collapse or 0.0))))
        raw = self.extra_signals if isinstance(self.extra_signals, Mapping) else {}
        object.__setattr__(self, "extra_signals", {str(k): float(v) for k, v in raw.items()})

    @property
    def has_spill(self) -> bool:
        return self.spill_bytes > 0

    @property
    def vloopv2_hit(self) -> bool:
        return self.vloopv2_loops > 0 and self.software_loops == 0


def _match_int(patterns: tuple[str, ...], text: str) -> int | None:
    for pat in patterns:
        if (m := re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)):
            unit = (m.group(2) if m.lastindex and m.lastindex >= 2 else "") or ""
            return int(m.group(1)) * _UNIT_MULTIPLIER.get(unit.strip().lower(), 1)
    return None


def _match_float(patterns: tuple[str, ...], text: str) -> float | None:
    for pat in patterns:
        if (m := re.search(pat, text, flags=re.IGNORECASE | re.MULTILINE)):
            return float(m.group(1))
    return None


def _combined(res_text: str, asm_or_dump_text: str) -> str:
    return "\n".join(p for p in (res_text, asm_or_dump_text) if p)


def _strip_cpp_comments(text: str) -> str:
    return re.sub(r"//[^\n]*", "", re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL))


def _scan_balanced(text: str, open_idx: int, open_ch: str, close_ch: str) -> int:
    depth, pos, n = 1, open_idx + 1, len(text)
    while pos < n and depth > 0:
        depth += (text[pos] == open_ch) - (text[pos] == close_ch)
        pos += 1
    return pos if depth == 0 else -1


def _detect_local_membar_in_vf(res_text: str = "", asm_or_dump_text: str = "") -> bool:
    comb = _combined(res_text, asm_or_dump_text)
    if re.search(r"has_local_membar_in_vf_call\s*[:=]\s*(true|1|yes)", comb, flags=re.IGNORECASE):
        return True
    return bool(re.search(r"(__simd_vf__|__no_simd_vf_fusion__|asc_vf_call|T\.SimdVF|simdvf|vf_call)", comb, flags=re.IGNORECASE)) and bool(re.search(r"(LocalMemBar|local_membar|pipe_barrier\s*\(\s*PIPE_ALL|PIPE_ALL)", comb, flags=re.IGNORECASE))


def _detect_vf_fusion_breaks(res_text: str = "", asm_or_dump_text: str = "") -> int:
    comb = _combined(res_text, asm_or_dump_text)
    if (exp := _match_int((r"\bvf_fusion_breaks\s*[:=]\s*(\d+)",), comb)) is not None:
        return max(0, exp)
    clean_src = "\n".join(ln for ln in _strip_cpp_comments(asm_or_dump_text if asm_or_dump_text.strip() else comb).splitlines() if not re.match(r"^\s*(?:#\s*(?:define|include)|extern\b)", ln))
    marks = re.findall(r"\bvf_fusion_break\b", clean_src, flags=re.IGNORECASE)
    calls = re.findall(r"(?:\basc_vf_call\s*(?:<[^>\n]*>\s*)?\(|\bT\.SimdVF\s*\(|\bSimdVF\s*\()", clean_src, flags=re.IGNORECASE)
    launches = len(calls) if calls else len(re.findall(r"\b__simd_vf__\b", clean_src, flags=re.IGNORECASE))
    return ((launches - 1) + len(marks)) if launches >= 2 else len(marks)


def _detect_alias_serialized_buffers(res_text: str = "", asm_or_dump_text: str = "") -> int:
    comb = _combined(res_text, asm_or_dump_text)
    if (exp := _match_int((r"\balias_serialized_buffers\s*[:=]\s*(\d+)",), comb)) is not None:
        return max(0, exp)
    return len(re.findall(r"(?:enable_offset\s*[:=]\s*(?:true|1|yes)|\bmay_alias\b|\balias_hazard\b|\bbuffer_aliasing\b|\bserialized_ping_pong\b|\bping_pong_serialized\b|\bdegraded_single_buffer\b|\bsingle_buffer_fallback\b|\bunrestricted_alias\b|\bdynamic_offset\b)", comb, flags=re.IGNORECASE))


def _detect_dual_issue_ipc_collapse(res_text: str = "", asm_or_dump_text: str = "") -> float:
    comb = _combined(res_text, asm_or_dump_text)
    if (exp := _match_float((r"\bdual_issue_ipc_collapse\s*[:=]\s*([0-9]*\.?[0-9]+)",), comb)) is not None:
        return max(0.0, min(1.0, exp))
    src = asm_or_dump_text if asm_or_dump_text.strip() else comb
    fma_re = re.compile(r"^\s*(?:vmad|vfma|vmula|vmla|Reg::MulAdd|T\.simd\.vmula)\b[\s(]+([A-Za-z_][A-Za-z0-9_\[\]]*)", flags=re.IGNORECASE)
    sfu_re = re.compile(r"\b(?:vexp|vsqrt|vdiv|vln|vrec|T\.simd\.v(?:exp|sqrt|div|ln)|Reg::(?:Exp|Sqrt|Div))\b", flags=re.IGNORECASE)
    raw_fma, sfu_chains, fma_run, sfu_run, prev_dst = 0, 0, 0, 0, None
    for ln in (ln.strip() for ln in src.splitlines() if ln.strip()):
        if (mf := fma_re.search(ln)):
            dst = mf.group(1).lower()
            fma_run, prev_dst = (fma_run + 1 if dst == prev_dst else 1), dst
            raw_fma += (fma_run == 2)
        else:
            prev_dst, fma_run = None, 0
        sfu_run = (sfu_run + 1) if sfu_re.search(ln) else 0
        sfu_chains += (sfu_run == 2)
    marks = len(re.findall(r"\b(?:raw_fma_chain|single_accumulator|sfu_stall|dual_issue_collapse)\b", comb, flags=re.IGNORECASE))
    return max(0.0, min(1.0, round(0.35 * raw_fma + 0.30 * sfu_chains + 0.35 * marks, 4)))


def _detect_dcache_dcci_flushes(res_text: str = "", asm_or_dump_text: str = "") -> int:
    comb = _combined(res_text, asm_or_dump_text)
    if (exp := _match_int((r"\bdcache_dcci_flushes\s*[:=]\s*(\d+)",), comb)) is not None:
        return max(0, exp)
    return len(re.findall(r"\b(?:dcci(?:_all)?|DataCacheCleanAndInvalid|dcache_invalidate)\b", comb, flags=re.IGNORECASE))


def _extract_for_loops(text: str) -> list[tuple[str, str]]:
    loops, n = [], len(text)
    for m in re.finditer(r"\bfor\s*\(", text):
        if (ep := _scan_balanced(text, m.end() - 1, "(", ")")) < 0:
            continue
        pos = ep
        while pos < n and text[pos].isspace():
            pos += 1
        if pos < n and text[pos] == "{" and (eb := _scan_balanced(text, pos, "{", "}")) > pos:
            loops.append((text[m.end() : ep - 1], text[pos + 1 : eb - 1]))
        elif pos < n and (semi := text.find(";", pos)) != -1:
            loops.append((text[m.end() : ep - 1], text[pos : semi + 1]))
    return loops


def _strip_subscripts(text: str) -> str:
    out, depth = [], 0
    for ch in text:
        if ch == "[":
            if depth == 0:
                out.append("[]")
            depth += 1
        elif ch == "]":
            depth = max(0, depth - 1)
        elif depth == 0:
            out.append(ch)
    return "".join(out)


def _static_loop_bound_lt_threshold(header: str, threshold: int = 32) -> bool:
    parts = [p.strip() for p in header.split(";")]
    if len(parts) != 3:
        return False
    clt = re.fullmatch(r"([A-Za-z_]\w*)\s*(<=|<)\s*(\d+)[uUlL]*", parts[1])
    cgt = re.fullmatch(r"(\d+)[uUlL]*\s*(>=|>)\s*([A-Za-z_]\w*)", parts[1]) if not clt else None
    if clt:
        var, upper = clt.group(1), int(clt.group(3)) + (clt.group(2) == "<=")
    elif cgt:
        var, upper = cgt.group(3), int(cgt.group(1)) + (cgt.group(2) == ">=")
    else:
        return False
    if not (init_m := re.search(rf"\b{re.escape(var)}\s*=\s*(-?\d+)[uUlL]*\b", parts[0])):
        return False
    step = 1 if re.fullmatch(rf"(?:\+\+\s*{re.escape(var)}|{re.escape(var)}\s*\+\+)", parts[2]) else (int(am.group(1)) if (am := re.fullmatch(rf"{re.escape(var)}\s*\+=\s*(\d+)[uUlL]*", parts[2])) else 0)
    return step > 0 and max(0, (upper - int(init_m.group(1)) + step - 1) // step) < threshold


def _elem_bytes_from_type_str(type_str: str) -> float | None:
    low = type_str.lower()
    return next((size for pat, size in _TYPE_BYTES_PATTERNS if pat.search(low)), None)


def _infer_gm_elem_bytes(text: str, target_ptrs: set[str] | None = None) -> float:
    if (tmpl := re.search(r"\b(?:tl::)?write_gm_bypass_dcache\s*<\s*([A-Za-z0-9_]+)", text)) and (sz := _elem_bytes_from_type_str(tmpl.group(1))) is not None:
        return sz
    for dm in re.finditer(r"__gm__\s+([^;(),={}]*?)\*+\s*(?:__restrict__\s*)?([A-Za-z_]\w*)", text):
        if (not target_ptrs or dm.group(2) in target_ptrs) and (sz := _elem_bytes_from_type_str(dm.group(1))) is not None:
            return sz
    return _elem_bytes_from_type_str(text) or 2.0


def _extract_header_fn_name(sig_head: str) -> str:
    cleaned = re.sub(r"\b(?:const|noexcept|override|final)\s*$", "", sig_head.strip()).rstrip()
    if not cleaned.endswith(")"):
        return ""
    depth, open_idx = 0, -1
    for idx in range(len(cleaned) - 1, -1, -1):
        depth += (cleaned[idx] == ")") - (cleaned[idx] == "(")
        if depth == 0:
            open_idx = idx
            break
    ident = re.search(r"([A-Za-z_]\w*)\s*$", cleaned[:open_idx].rstrip()) if open_idx > 0 else None
    return "" if (not ident or ident.group(1) in ("if", "for", "while", "switch", "decltype", "__launch_bounds__", "__mix__")) else ident.group(1)


def _extract_reachable_trackb_text(src: str) -> str:
    masked, blocks, depth, stmt_start, i, n = _strip_cpp_comments(src), [], 0, 0, 0, len(src)
    masked_n = len(masked)
    while i < masked_n:
        ch = masked[i]
        if ch == ";" and depth == 0:
            stmt_start = i + 1
        elif ch == "{" and depth == 0:
            sig_head, j = masked[stmt_start:i].strip(), _scan_balanced(masked, i, "{", "}")
            if j < 0:
                break
            if (fn := _extract_header_fn_name(sig_head)):
                blocks.append((fn, "__global__" in sig_head, src[stmt_start:j]))
            stmt_start = i = j
            continue
        elif ch in "{}":
            depth = max(0, depth + (1 if ch == "{" else -1))
            if depth == 0:
                stmt_start = i + 1
        i += 1
    _ = n
    kernels = {name for name, is_g, _ in blocks if is_g}
    if not kernels:
        return src
    launched = [m.group(1) for m in _CPP_LAUNCH_RE.finditer(masked) if m.group(1) in kernels]
    entry = launched[0] if launched else next(name for name, is_g, _ in blocks if is_g)
    by_name, reached, pending = {nm: t for nm, _, t in blocks}, set(), [entry]
    while pending:
        if (cur := pending.pop()) in reached or cur not in by_name:
            continue
        reached.add(cur)
        bm = _strip_cpp_comments(by_name[cur])
        pending.extend(m.group(1) for m in _CPP_CALL_RE.finditer(bm) if m.group(1) in by_name and m.group(1) not in reached)
        for tm in _CPP_TMPL_ARGS_RE.finditer(bm):
            pending.extend(idn for idn in re.findall(r"\b([A-Za-z_]\w*)\b", tm.group(1)) if idn in by_name and idn not in reached)
    return "\n\n".join(t for nm, _, t in blocks if nm in reached) or src


def _has_parallel_simt_kernel(cleaned: str) -> bool:
    if not re.search(r"\b__simt_vf__\b", cleaned):
        return False
    bounds = [int(m.group(1)) for m in _SIMT_LAUNCH_BOUNDS_RE.finditer(cleaned)]
    dim3s = [int(m.group(1)) * int(m.group(2) or 1) * int(m.group(3) or 1) for m in _SIMT_VF_CALL_DIM3_RE.finditer(cleaned)]
    return not ((bounds and max(bounds) < 32) or (dim3s and max(dim3s) < 32)) and bool((bounds and max(bounds) >= 32) or (dim3s and max(dim3s) >= 32))


def _iter_scalar_candidate_loops(cleaned: str) -> Iterator[str]:
    for hdr, body in _extract_for_loops(cleaned):
        if not (_NESTED_FOR_RE.search(body) or _VEC_MATRIX_INST_RE.search(body) or _ASC_PRIM_CALL_RE.search(body) or _static_loop_bound_lt_threshold(hdr, 32)):
            yield body


def _has_scalar_compute_op(body: str) -> bool:
    return bool(_SCALAR_COMPUTE_OP_RE.search(_strip_subscripts(body.replace("++", " ").replace("--", " ").replace("->", " "))))


def count_reachable_ub_scalar_compute_loops(cpp_source: str) -> int:
    """Count innermost reachable scalar compute loops operating on __ubuf__ buffers."""
    if not cpp_source or not cpp_source.strip():
        return 0
    cleaned = _strip_cpp_comments(_extract_reachable_trackb_text(cpp_source) if re.search(r"\b__global__\b", cpp_source) else cpp_source)
    if _has_parallel_simt_kernel(cleaned) or not (ub_ptrs := set(_UB_PTR_NAME_RE.findall(cleaned))):
        return 0
    alt = "|".join(re.escape(p) for p in sorted(ub_ptrs))
    acc_re = re.compile(rf"\b(?:{alt})\s*\[[^\]]+\]|\*\s*\(?\s*(?:{alt})\b")
    return sum(1 for b in _iter_scalar_candidate_loops(cleaned) if acc_re.search(b) and _has_scalar_compute_op(b))


def _detect_scalar_compute_loop_fallback(res_text: str = "", asm_or_dump_text: str = "") -> dict[str, float]:
    comb = _combined(res_text, asm_or_dump_text)
    if (exp := _match_float((r"\bscalar_compute_loop_fallback\s*[:=]\s*([0-9]*\.?[0-9]+)",), comb)) is not None:
        return {"scalar_compute_loop_fallback": float(exp)} if exp > 0.0 else {}
    if re.search(r"\bscalar_compute_loop_fallback\s*[:=]\s*(true|yes)", comb, flags=re.IGNORECASE):
        return {"scalar_compute_loop_fallback": 1.0}
    src = asm_or_dump_text if asm_or_dump_text.strip() else comb
    if re.search(r"\b__global__\b", src):
        src = _extract_reachable_trackb_text(src)
    if (ub_loops := count_reachable_ub_scalar_compute_loops(src)) > 0:
        return {"scalar_compute_loop_fallback": float(ub_loops)}
    cleaned = _strip_cpp_comments(src)
    if _VEC_MATRIX_INST_RE.search(cleaned) or _has_parallel_simt_kernel(cleaned):
        return {}
    gm_ptrs = set(_GM_PTR_NAME_RE.findall(cleaned))
    gm_alt = "|".join(re.escape(p) for p in sorted(gm_ptrs))
    wre = re.compile(rf"(?:\b(?:{gm_alt})\s*\[[^\]]+\]|\*\s*\(?\s*(?:{gm_alt})[^;={{}}]*\))\s*(?:[+\-*/%&|^]?=)(?!=)") if gm_ptrs else re.compile(r"(?:\b[A-Za-z_]\w*\s*\[[^\]]+\]|\*\s*\([^)]+\))\s*(?:[+\-*/%&|^]?=)(?!=)")
    return {"scalar_compute_loop_fallback": 1.0} if any(wre.search(b) and _has_scalar_compute_op(b) for b in _iter_scalar_candidate_loops(cleaned)) else {}


def detect_scalar_compute_loop_fallback(
    tir_ledger: ObjectivePhysicalLedger | None = None, *, ub_scalar_compute_loops: int = 0,
    cpp_source: str = "", extra_signals: Mapping[str, object] | None = None,
) -> float:
    """Return scalar-compute fallback severity from ledger, C++ source, and signals."""
    loops = max(0, int(ub_scalar_compute_loops), int(tir_ledger.ub_scalar_compute_loops) if tir_ledger is not None else 0)
    explicit = 0.0
    if extra_signals is not None:
        if isinstance(rs := extra_signals.get("scalar_compute_loop_fallback", 0.0), (int, float)) and not isinstance(rs, bool):
            explicit = max(explicit, float(rs))
        if isinstance(rl := extra_signals.get("ub_scalar_compute_loops", 0), (int, float)) and not isinstance(rl, bool):
            loops = max(loops, max(0, int(rl)))
    if cpp_source:
        loops = max(loops, count_reachable_ub_scalar_compute_loops(cpp_source))
        if loops == 0:
            explicit = max(explicit, float(_detect_scalar_compute_loop_fallback(asm_or_dump_text=cpp_source).get("scalar_compute_loop_fallback", 0.0)))
    return max(explicit, float(loops))


def _detect_dcache_bypass_narrow_store(res_text: str = "", asm_or_dump_text: str = "") -> dict[str, float]:
    comb = _combined(res_text, asm_or_dump_text)
    if (exp := _match_float((r"\bdcache_bypass_narrow_store\s*[:=]\s*([0-9]*\.?[0-9]+)",), comb)) is not None:
        return {"dcache_bypass_narrow_store": max(4.0, min(64.0, float(exp)))} if exp > 0.0 else {}
    cleaned = _strip_cpp_comments(asm_or_dump_text if asm_or_dump_text.strip() else comb)
    if _BYPASS_DCACHE_RE.search(cleaned):
        return {"dcache_bypass_narrow_store": max(4.0, min(64.0, 64.0 / _infer_gm_elem_bytes(cleaned)))}
    ptrs = set(_GM_PTR_NAME_RE.findall(cleaned)) | set(re.findall(r"\b(gm_[A-Za-z0-9_]+|[A-Za-z0-9_]+_gm)\b", cleaned))
    if not ptrs:
        return {}
    alt = "|".join(re.escape(p) for p in sorted(ptrs))
    sre = re.compile(rf"\b({alt})\s*\[[^\]]+\]\s*(?:[+\-*/%&|^]?=)(?!=)|\*\s*\(\s*({alt})\b[^)]*\)\s*(?:[+\-*/%&|^]?=)(?!=)")
    matched = {m.group(1) or m.group(2) for _, body in _extract_for_loops(cleaned) for m in sre.finditer(body) if (m.group(1) or m.group(2))}
    return {} if (not matched or (eb := _infer_gm_elem_bytes(cleaned, target_ptrs=matched)) >= 32.0) else {"dcache_bypass_narrow_store": max(4.0, min(64.0, 64.0 / eb))}


def _detect_spill_and_stack(res_text: str = "", asm_or_dump_text: str = "") -> dict[str, int]:
    comb, sz = _combined(res_text, asm_or_dump_text), r"(bytes?|kib|mib|kb|mb|b)?"
    return {
        "vec_regs": _match_int((r"(?:vector\s+register\s+usage|used\s+vector\s+registers?|number\s+of\s+vector\s+registers?|vec_regs?|vecregs?)\s*[:=]\s*(\d+)", r"\bvec_reg_num\s*[:=]\s*(\d+)"), comb) or 0,
        "scalar_regs": _match_int((r"(?:scalar\s+register\s+usage|used\s+scalar\s+registers?|number\s+of\s+scalar\s+registers?|scalar_regs?|scalarregs?)\s*[:=]\s*(\d+)", r"\bscalar_reg_num\s*[:=]\s*(\d+)"), comb) or 0,
        "spill_bytes": _match_int((rf"(?:register\s+spill|stack\s+spill\s+size|spill\s+stack\s+size|spill_bytes|spillsize|spill\s+size|spill)\s*[:=]\s*(\d+)\s*{sz}",), comb) or 0,
        "stack_bytes": _match_int((rf"(?:stack\s+frame\s+size|stack_bytes|stacksize|stack\s+size|frame\s+size)\s*[:=]\s*(\d+)\s*{sz}",), comb) or 0,
    }


def _detect_loop_vectorization(res_text: str = "", asm_or_dump_text: str = "") -> dict[str, int]:
    comb = _combined(res_text, asm_or_dump_text)
    asm = asm_or_dump_text if asm_or_dump_text.strip() else comb
    ev2 = _match_int((r"(?:vloopv2_loops|vloopv2\s+loops?)\s*[:=]\s*(\d+)",), comb)
    v2 = ev2 if ev2 is not None else len(re.findall(r"\bVLOOP_?V2\b(?!\s*[:=]\s*0)", _VLOOPV2_MISS_DIAG_RE.sub(" ", asm), flags=re.IGNORECASE))
    if (esw := _match_int((r"(?:software_loops|software\s+loops?|scalar\s+loops?)\s*[:=]\s*(\d+)",), comb)) is not None:
        return {"vloopv2_loops": v2, "software_loops": esw}
    sw_m, miss_m = re.findall(r"(?:software\s+loop|unvectorized\s+loop|FALLBACK_SW_LOOP|\bVLOOP\b(?!_?V2))", asm, flags=re.IGNORECASE), _VLOOPV2_MISS_DIAG_RE.findall(asm)
    back_b = re.findall(r"\b(?:b|b\.[a-z]{2}|cbn?z|tbn?z|jump)\s+[^\n]*?\.L(?:BB)?[0-9_]+", asm, flags=re.IGNORECASE)
    return {"vloopv2_loops": v2, "software_loops": (len(sw_m) + len(miss_m)) if (sw_m or miss_m) else (len(back_b) if (back_b and v2 == 0) else 0)}


def _ctx_float(ctx: Mapping[str, object], key: str, default: float = 0.0) -> float:
    return float(raw) if (isinstance(raw := ctx.get(key), (int, float)) and not isinstance(raw, bool)) else default


def _penalty_spill_and_stack(s: BishengResourceSummary, ctx: Mapping[str, Any]) -> float:
    b = float(ctx.get("base_ref", 1.0))
    return (b * min(1.5, 0.20 + 0.004 * s.spill_bytes) + s.spill_bytes * 0.01) if s.spill_bytes > 0 else (min(b * 0.25, float(s.stack_bytes) * 0.002) if s.stack_bytes > 1024 else 0.0)


def _penalty_software_loop(s: BishengResourceSummary, ctx: Mapping[str, Any]) -> float:
    return (max(float(ctx.get("vec_floor_us", 0.0)), float(ctx.get("base_ref", 1.0)) * 0.5) * min(2.0, 0.35 * s.software_loops)) if (s.software_loops > 0 and s.vloopv2_loops == 0) else 0.0


def _penalty_local_membar(s: BishengResourceSummary, ctx: Mapping[str, Any]) -> float:
    return max(1.0, float(ctx.get("vec_floor_us", 0.0)) * 0.25) if s.has_local_membar_in_vf_call else 0.0


def _penalty_vf_fusion_breaks(s: BishengResourceSummary, ctx: Mapping[str, Any]) -> float:
    return (max(1.0, float(ctx.get("vec_floor_us", 0.0)) * 0.18, float(ctx.get("base_ref", 1.0)) * 0.12) * float(min(6, s.vf_fusion_breaks))) if s.vf_fusion_breaks > 0 else 0.0


def _penalty_alias_serialized(s: BishengResourceSummary, ctx: Mapping[str, Any]) -> float:
    return (max(float(ctx.get("hbm_floor_us", 0.0)), float(ctx.get("vec_floor_us", 0.0)), float(ctx.get("base_ref", 1.0)) * 0.25) * min(2.5, 0.30 * s.alias_serialized_buffers)) if s.alias_serialized_buffers > 0 else 0.0


def _penalty_dual_issue_collapse(s: BishengResourceSummary, ctx: Mapping[str, Any]) -> float:
    sev = max(0.0, min(1.0, float(s.dual_issue_ipc_collapse)))
    return (max(float(ctx.get("vec_floor_us", 0.0)), float(ctx.get("base_ref", 1.0)) * 0.20) * 0.45 * sev) if sev > 0.0 else 0.0


def _penalty_dcache_dcci(s: BishengResourceSummary, ctx: Mapping[str, Any]) -> float:
    b = float(ctx.get("base_ref", 1.0))
    return min(2.0 * b, float(s.dcache_dcci_flushes) * max(1.5, 0.08 * b)) if s.dcache_dcci_flushes > 0 else 0.0


def _penalty_scalar_compute_loop_fallback(s: BishengResourceSummary, ctx: Mapping[str, object]) -> float:
    if (sev := float(s.extra_signals.get("scalar_compute_loop_fallback", 0.0) or 0.0)) <= 0.0:
        return 0.0
    b, vf = max(1.0, _ctx_float(ctx, "base_ref", 1.0)), max(0.0, _ctx_float(ctx, "vec_floor_us", 0.0))
    return min(1.5 * vf, 0.35 * sev * max(vf, 1.0)) if vf > 0.0 else min(1.5 * b, 0.35 * sev * b)


def _penalty_dcache_bypass_narrow_store(s: BishengResourceSummary, ctx: Mapping[str, object]) -> float:
    if (amp := float(s.extra_signals.get("dcache_bypass_narrow_store", 0.0) or 0.0)) <= 1.0:
        return 0.0
    eh = h if (h := max(0.0, _ctx_float(ctx, "hbm_floor_us", 0.0))) > 0.0 else max(1.0, _ctx_float(ctx, "base_ref", 1.0)) * 0.25
    return eh * 0.5 * max(0.0, amp - 1.0)


def _positional_params(fn: Callable[..., Any]) -> list[str] | None:
    try:
        sig = inspect.signature(fn)
    except (ValueError, TypeError):
        return None
    if any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in sig.parameters.values()):
        return None
    return [p.name for p in sig.parameters.values() if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)]


def _invoke_parse_fn(parse_fn: Callable[..., Any], res_text: str, asm_or_dump_text: str) -> Any:
    params = _positional_params(parse_fn)
    return parse_fn(_combined(res_text, asm_or_dump_text)) if (params is not None and len(params) == 1) else parse_fn(res_text, asm_or_dump_text)


def _invoke_penalty_fn(penalty_fn: Callable[..., float], summary: BishengResourceSummary, ctx: Mapping[str, Any]) -> float:
    params = _positional_params(penalty_fn)
    if params is not None and len(params) == 1:
        return float(penalty_fn(ctx) if params[0].lower() in ("ctx", "context") else penalty_fn(summary))
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
        if not callable(self.parse_fn) or not callable(self.penalty_fn):
            raise TypeError("CompilerBehaviorDetector parse_fn and penalty_fn must be callable")


class CompilerBehaviorRegistry:
    """Extensible registry of Tier-1 compiler behavior parsers and latency penalty hooks."""

    def __init__(self, detectors: list[CompilerBehaviorDetector] | None = None) -> None:
        self._detectors: dict[str, CompilerBehaviorDetector] = {d.name: d for d in (detectors or ())}

    def register(
        self, detector_or_name: CompilerBehaviorDetector | str, parse_fn: Callable[..., Any] | None = None,
        penalty_fn: Callable[..., float] | None = None, *, summary_field: str | None = None,
        description: str = "", overwrite: bool = True,
    ) -> CompilerBehaviorDetector:
        """Register a built-in or custom compiler behavior detector."""
        if isinstance(detector_or_name, CompilerBehaviorDetector):
            det = detector_or_name
        else:
            if parse_fn is None or penalty_fn is None:
                raise ValueError("parse_fn and penalty_fn are required when registering by name")
            det = CompilerBehaviorDetector(str(detector_or_name), parse_fn, penalty_fn, summary_field, description)
        if not overwrite and det.name in self._detectors:
            raise KeyError(f"Compiler detector {det.name!r} is already registered")
        self._detectors[det.name] = det
        return det

    def unregister(self, name: str) -> bool:
        return self._detectors.pop(name, None) is not None

    def get(self, name: str) -> CompilerBehaviorDetector | None:
        return self._detectors.get(name)

    def list_detectors(self) -> tuple[str, ...]:
        return tuple(self._detectors.keys())

    @property
    def detectors(self) -> tuple[CompilerBehaviorDetector, ...]:
        return tuple(self._detectors.values())

    def clone(self) -> CompilerBehaviorRegistry:
        return CompilerBehaviorRegistry(list(self._detectors.values()))

    def parse(self, res_text: str = "", asm_or_dump_text: str = "") -> BishengResourceSummary:
        """Run registered detectors over compiler, assembly, or IR text."""
        if not _combined(res_text, asm_or_dump_text).strip():
            return BishengResourceSummary()
        fields = BishengResourceSummary().to_dict()
        extra_signals: dict[str, float] = {}
        fields.pop("extra_signals", None)
        for det in self._detectors.values():
            raw = _invoke_parse_fn(det.parse_fn, res_text, asm_or_dump_text)
            if isinstance(raw, Mapping):
                for k, v in raw.items():
                    if (sk := str(k)) in fields:
                        fields[sk] = v
                    elif v is not None and float(v) != 0.0:
                        extra_signals[sk] = float(v)
            elif det.summary_field and det.summary_field in fields:
                fields[det.summary_field] = raw
            elif raw is not None and (val := float(raw) if not isinstance(raw, bool) else (1.0 if raw else 0.0)) != 0.0:
                extra_signals[det.name] = val
        return BishengResourceSummary.from_dict({**fields, "extra_signals": extra_signals})

    def compute_tier1_penalties(self, summary: BishengResourceSummary, context: Mapping[str, Any] | None = None) -> tuple[float, dict[str, float]]:
        """Evaluate every registered detector's Tier-1 penalty and return (total_us, breakdown)."""
        base_ctx, total, breakdown = dict(context or {}), 0.0, {}
        for det in self._detectors.values():
            sig = getattr(summary, det.summary_field) if (det.summary_field and hasattr(summary, det.summary_field)) else summary.extra_signals.get(det.name, 0.0)
            if (val := max(0.0, _invoke_penalty_fn(det.penalty_fn, summary, {**base_ctx, "signal": sig}))) > 0.0:
                breakdown[det.name] = val
                total += val
        return total, breakdown


_DEFAULT_DETECTORS: tuple[tuple[str, Callable[..., Any], Callable[..., float], str | None, str], ...] = (
    ("register_stack_spill", _detect_spill_and_stack, _penalty_spill_and_stack, "spill_bytes", "(a) Vector/scalar register pressure and stack spill bytes"),
    ("loop_vectorization_vloopv2", _detect_loop_vectorization, _penalty_software_loop, "software_loops", "(b) Hardware VLOOPv2 zero-overhead loop hit vs software branch fallback"),
    ("vf_local_membar", _detect_local_membar_in_vf, _penalty_local_membar, "has_local_membar_in_vf_call", "(c) LocalMemBar/PIPE_ALL serialization inside SimdVF / asc_vf_call"),
    ("vf_fusion_breaks", _detect_vf_fusion_breaks, _penalty_vf_fusion_breaks, "vf_fusion_breaks", "(d) Fragmented asc_vf_call/SimdVF launches forcing UB store->load roundtrips"),
    ("alias_serialized_buffers", _detect_alias_serialized_buffers, _penalty_alias_serialized, "alias_serialized_buffers", "(e) Dynamic offset / unrestricted pointer aliasing serializing MTE2/V/MTE3 ping-pong"),
    ("dual_issue_ipc_collapse", _detect_dual_issue_ipc_collapse, _penalty_dual_issue_collapse, "dual_issue_ipc_collapse", "(f) Single-accumulator RAW FMA chains or back-to-back SFU chains collapsing IPC"),
    ("dcache_dcci_flushes", _detect_dcache_dcci_flushes, _penalty_dcache_dcci, "dcache_dcci_flushes", "(g) Scalar dcci D-cache invalidation storms inside loops"),
    ("scalar_compute_loop_fallback", _detect_scalar_compute_loop_fallback, _penalty_scalar_compute_loop_fallback, None, "(h) Scalar for-loop arithmetic/bitwise compute with zero vector/matrix ops"),
    ("dcache_bypass_narrow_store", _detect_dcache_bypass_narrow_store, _penalty_dcache_bypass_narrow_store, None, "(i) Bypass-DCache or sub-32B elementwise GM stores causing 64B amplification"),
)


def _build_default_registry() -> CompilerBehaviorRegistry:
    reg = CompilerBehaviorRegistry()
    for n, pf, penf, sf, d in _DEFAULT_DETECTORS:
        reg.register(n, pf, penf, summary_field=sf, description=d)
    return reg


DEFAULT_COMPILER_BEHAVIOR_REGISTRY: CompilerBehaviorRegistry = _build_default_registry()


def register_compiler_detector(
    name: str, parse_fn: Callable[..., Any], penalty_fn: Callable[..., float], *,
    registry: CompilerBehaviorRegistry | None = None, summary_field: str | None = None, description: str = "", overwrite: bool = True,
) -> CompilerBehaviorDetector:
    return (registry if registry is not None else DEFAULT_COMPILER_BEHAVIOR_REGISTRY).register(name, parse_fn, penalty_fn, summary_field=summary_field, description=description, overwrite=overwrite)


def parse_bisheng_res_usage(res_text: str = "", asm_or_dump_text: str = "", *, registry: CompilerBehaviorRegistry | None = None) -> BishengResourceSummary:
    return (registry if registry is not None else DEFAULT_COMPILER_BEHAVIOR_REGISTRY).parse(res_text, asm_or_dump_text)


parse_bisheng_resources = parse_bisheng_res_usage

__all__ = [
    "BishengResourceSummary", "CompilerBehaviorDetector", "CompilerBehaviorRegistry",
    "DEFAULT_COMPILER_BEHAVIOR_REGISTRY", "count_reachable_ub_scalar_compute_loops",
    "detect_scalar_compute_loop_fallback", "parse_bisheng_res_usage",
    "parse_bisheng_resources", "register_compiler_detector",
]
