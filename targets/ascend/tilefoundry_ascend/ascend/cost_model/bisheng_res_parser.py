"""Parser for ``bisheng --cce-res-usage`` output, CCE assembly/IR dumps, and Tier-1 behavior registry."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, field
import inspect
import re
from typing import Any

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
    combined = "\n".join(part for part in (res_text, asm_or_dump_text) if part)
    explicit = _match_int((r"\bvf_fusion_breaks\s*[:=]\s*(\d+)",), combined)
    if explicit is not None:
        return max(0, explicit)
    src = asm_or_dump_text if asm_or_dump_text.strip() else combined
    launches = re.findall(
        r"(?:\basc_vf_call\b|\bT\.SimdVF\b|\b__simd_vf__\b|\bSimdVF\s*\()",
        src,
        flags=re.IGNORECASE,
    )
    explicit_markers = re.findall(r"\bvf_fusion_break\b", src, flags=re.IGNORECASE)
    if len(launches) >= 2:
        return (len(launches) - 1) + len(explicit_markers)
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
    """Plugin descriptor for one compiler-level phenomenon parsed from dumps and penalized in Tier-1."""

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
        """Run all registered detectors on compiler/asm/IR text to produce BishengResourceSummary."""
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
    "parse_bisheng_res_usage",
    "parse_bisheng_resources",
    "register_compiler_detector",
]
