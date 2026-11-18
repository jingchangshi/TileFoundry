"""Deterministic Ascend 950 (dav-3510) SIL hardware redline and pitfall verifier."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
import re

from tilefoundry_ascend.ascend.sil.schema import (
    VALID_FREEDOM_FACETS,
    AdvisoryEnvelope,
    ConstraintRelaxationMask,
    HardwareLegalityBounds,
    PolicyEnvelope,
    RegimeSlice,
    SchedulePolicy,
    VerificationReport,
)

_INT4_DTYPES: frozenset[str] = frozenset({"i4", "u4", "int4", "q4_0"})
_CROSSCORE_FLAG_MODES: frozenset[str] = frozenset(
    {"mode4_flag", "mode2_flag", "cross_core_flag"}
)
_SIMT_VECTOR_MODES: frozenset[str] = frozenset({"simtvf", "hybrid"})
_HF32_MIN_FP32_THRESHOLD: float = 2.0**-11


def _normalize_dma_path(path: str) -> str:
    return path.replace(" ", "").upper()


def _resolve_fp32_threshold(precision_thresholds: dict[str, float]) -> float | None:
    for key in ("float32", "fp32", "f32"):
        if key in precision_thresholds:
            return float(precision_thresholds[key])
    return None


def _is_prefill_regime(regime_tag: str) -> bool:
    return "prefill" in regime_tag.lower()


class SILVerifier:
    """Enforces dav-3510 hardware bounds, DMA legality, sync invariants, and precision guards."""

    def verify_policy(
        self,
        policy: SchedulePolicy,
        advisory: AdvisoryEnvelope,
        *,
        regime: RegimeSlice | None = None,
    ) -> VerificationReport:
        """Verify a single SchedulePolicy against an AdvisoryEnvelope."""
        errors: list[str] = []
        warnings: list[str] = []
        suggested_fixes: dict[str, object] = {}

        bounds = advisory.legality_bounds
        if regime is None:
            for item in advisory.regimes:
                if item.regime_tag == policy.regime_tag:
                    regime = item
                    break

        tag = policy.regime_tag
        axes = policy.topology_axes

        # 1. L1 capacity & Half-L1 ping/pong isolation
        if policy.l1_estimated_bytes > bounds.l1_bytes:
            errors.append(
                f"[{tag}] L1 footprint {policy.l1_estimated_bytes} B exceeds dav-3510 L1 limit {bounds.l1_bytes} B"
            )
        if policy.l1_half_bank_split and policy.l1_stages >= 2 and policy.l1_estimated_bytes > 0:
            stages_per_half_bank = math.ceil(policy.l1_stages / 2)
            half_bank_bytes = (policy.l1_estimated_bytes * stages_per_half_bank) / policy.l1_stages
            if half_bank_bytes > bounds.l1_half_bank_bytes:
                errors.append(
                    f"[{tag}] Half-L1 slot footprint {int(math.ceil(half_bank_bytes))} B "
                    f"exceeds {bounds.l1_half_bank_bytes} B per ping/pong bank "
                    f"(l1_stages={policy.l1_stages})"
                )

        # 2. L0C accumulator tile capacity
        uses_cube = (
            axes.uses_cube
            or bool(policy.cube_operand_dtype)
            or policy.cv_datapath != "none"
            or policy.l0c_buffers > 1
        )
        bm, bn, _ = policy.tile_mnk
        if uses_cube:
            l0c_bytes = bm * bn * 4 * policy.l0c_buffers
            if l0c_bytes > bounds.l0c_bytes:
                errors.append(
                    f"[{tag}] L0C tile footprint bm*bn*4*l0c_buffers={l0c_bytes} B "
                    f"exceeds dav-3510 L0C limit {bounds.l0c_bytes} B"
                )

        # 3. UB capacity & 256KB unlock legality + cross-axis (coupled_1c2v x irregular_simt) carveout
        is_coupled_simt = (
            axes.compute_engine == "coupled_1c2v"
            and axes.iteration_regularity == "irregular_simt"
        )
        if policy.vector_mode in _SIMT_VECTOR_MODES and policy.ub_256kb_unlock:
            errors.append(
                f"[{tag}] ub_256kb_unlock=True is forbidden when vector_mode={policy.vector_mode!r}"
            )
            suggested_fixes["ub_256kb_unlock"] = False
        elif is_coupled_simt and policy.ub_256kb_unlock:
            errors.append(
                f"[{tag}] ub_256kb_unlock=True is forbidden for coupled_1c2v + irregular_simt "
                f"(32KB SIMT D-cache carveout)"
            )
            suggested_fixes["ub_256kb_unlock"] = False

        if policy.vector_mode == "simdvf":
            ub_limit = (
                bounds.ub_unreserved_bytes if policy.ub_256kb_unlock else bounds.ub_simdvf_bytes
            )
        else:
            ub_limit = bounds.ub_simtvf_bytes

        if is_coupled_simt:
            ub_limit = min(ub_limit, bounds.ub_simtvf_bytes)

        if policy.ub_estimated_bytes > ub_limit:
            if is_coupled_simt:
                errors.append(
                    f"[{tag}] UB footprint {policy.ub_estimated_bytes} B exceeds "
                    f"coupled_1c2v + irregular_simt cap {ub_limit} B (32KB SIMT D-cache carveout)"
                )
            else:
                errors.append(
                    f"[{tag}] UB footprint {policy.ub_estimated_bytes} B exceeds "
                    f"{policy.vector_mode} limit {ub_limit} B"
                )
                if (
                    policy.vector_mode == "simdvf"
                    and not policy.ub_256kb_unlock
                    and policy.ub_estimated_bytes <= bounds.ub_unreserved_bytes
                ):
                    suggested_fixes["ub_256kb_unlock"] = True

        # 4. Forbidden 3510 DMA paths
        forbidden_map = {_normalize_dma_path(p): p for p in bounds.forbidden_dma_paths}
        for path in policy.dma_paths_used:
            norm = _normalize_dma_path(path)
            if norm in forbidden_map:
                errors.append(
                    f"[{tag}] Forbidden dav-3510 DMA path {path!r} "
                    f"(forbidden={bounds.forbidden_dma_paths})"
                )

        # 5. Illegal direct Cube INT4
        if (
            bounds.forbid_cube_int4
            and policy.cube_operand_dtype.strip().lower() in _INT4_DTYPES
            and policy.w4_mode == "none"
        ):
            errors.append(
                f"[{tag}] Direct Cube INT4 ({policy.cube_operand_dtype!r}) with w4_mode='none' "
                f"is illegal on dav-3510; use 'aiv_unpack_f16' or 'mxa8w4_bitshift_x64'"
            )
            suggested_fixes["w4_mode"] = "aiv_unpack_f16"
            suggested_fixes["w4_mode_alternatives"] = ["aiv_unpack_f16", "mxa8w4_bitshift_x64"]

        # 6. Silent uint16_t MMAD wrap
        if max(policy.tile_mnk) > bounds.max_mmad_mnk:
            errors.append(
                f"[{tag}] tile_mnk={policy.tile_mnk} exceeds max_mmad_mnk={bounds.max_mmad_mnk} "
                f"(silent uint16_t MMAD wrap hazard)"
            )

        # 7. unit_flag_2_3 deadlock hazard
        if (
            policy.sync_mode in ("unit_flag_2_3", "unit_flag")
            and policy.l0c_buffers > 1
            and regime is not None
            and regime.has_unaligned_tail
        ):
            errors.append(
                f"[{tag}] {policy.sync_mode} deadlock hazard with l0c_buffers={policy.l0c_buffers} > 1 "
                f"and unaligned tail"
            )
            suggested_fixes["l0c_buffers"] = 1

        # 8. UB bank conflict check (512B stride across 16 banks * 32B datablocks)
        if (
            policy.ub_row_stride_bytes > 0
            and (policy.ub_row_stride_bytes % 512 == 0)
            and policy.ub_bank_pad_datablocks == 0
        ):
            warnings.append(
                f"[{tag}] UB row stride {policy.ub_row_stride_bytes} B is a multiple of 512 B "
                f"with ub_bank_pad_datablocks=0 (16-bank conflict risk)"
            )
            suggested_fixes["ub_bank_pad_datablocks"] = 1
            suggested_fixes["ub_rw_shift_datablocks"] = 8

        # 9. CrossCore depth > 15 guard (sync_mode OR coupled_1c2v axis)
        if (
            policy.sync_mode in _CROSSCORE_FLAG_MODES
            and policy.reverse_ack_period > bounds.max_crosscore_depth
        ):
            errors.append(
                f"[{tag}] reverse_ack_period={policy.reverse_ack_period} exceeds hardware "
                f"max_crosscore_depth={bounds.max_crosscore_depth} on sync_mode={policy.sync_mode!r}"
            )
            suggested_fixes["reverse_ack_period"] = min(14, bounds.max_crosscore_depth)
        elif (
            axes.compute_engine == "coupled_1c2v"
            and policy.reverse_ack_period > bounds.max_crosscore_depth
        ):
            errors.append(
                f"[{tag}] coupled_1c2v axis reverse_ack_period={policy.reverse_ack_period} exceeds "
                f"hardware max_crosscore_depth={bounds.max_crosscore_depth}"
            )
            suggested_fixes["reverse_ack_period"] = min(14, bounds.max_crosscore_depth)

        # 10. l0c2ub_split & ragged_group empty-subblock sync parity guard
        if (
            policy.cv_datapath.startswith(("l0c2ub_split", "l0c_to_ub_split"))
            and not policy.ensure_empty_subblock_sync
        ):
            errors.append(
                f"[{tag}] cv_datapath={policy.cv_datapath!r} requires "
                f"ensure_empty_subblock_sync=True to prevent deadlock on empty tail sub-blocks"
            )
            suggested_fixes["ensure_empty_subblock_sync"] = True
        elif (
            axes.iteration_regularity == "ragged_group"
            and not policy.ensure_empty_subblock_sync
        ):
            errors.append(
                f"[{tag}] topology_axes.iteration_regularity='ragged_group' requires "
                f"ensure_empty_subblock_sync=True to prevent deadlock on empty tail sub-blocks"
            )
            suggested_fixes["ensure_empty_subblock_sync"] = True

        # 11. SIMT thread count warning
        if policy.vector_mode in _SIMT_VECTOR_MODES and policy.simt_threads > 1024:
            warnings.append(
                f"[{tag}] simt_threads={policy.simt_threads} > 1024 on {policy.vector_mode}; "
                f"__launch_bounds__(512) recommended for 0 register spill"
            )
            suggested_fixes["simt_threads"] = 512

        # 12. HF32 precision guard
        fp32_thresh = _resolve_fp32_threshold(advisory.precision_thresholds)
        if policy.hf32_allowed and fp32_thresh is not None and fp32_thresh < _HF32_MIN_FP32_THRESHOLD:
            errors.append(
                f"[{tag}] hf32_allowed=True violates fp32 precision threshold "
                f"{fp32_thresh:g} (< 2^-11)"
            )
            suggested_fixes["hf32_allowed"] = False

        # 13. Non-IEEE unshifted exp guard on inf/nan inputs
        if policy.fast_unshifted_exp and advisory.has_nan_inf_cases:
            errors.append(
                f"[{tag}] fast_unshifted_exp=True is unsafe when advisory.has_nan_inf_cases=True"
            )
            suggested_fixes["fast_unshifted_exp"] = False

        # 14. bidirectional_cv_ring Prefill M < 82 warning
        if axes.memory_ring == "bidirectional_cv_ring" and _is_prefill_regime(tag):
            rep_m = regime.representative_dims.get("M") if regime is not None else None
            if bm < 82 or (rep_m is not None and 0 < rep_m < 82):
                warnings.append(
                    f"[{tag}] mte2_bound_below_82: M < 82 on bidirectional_cv_ring Prefill "
                    f"(tile_m={bm}, rep_m={rep_m})"
                )

        return VerificationReport(
            passed=len(errors) == 0,
            errors=tuple(errors),
            warnings=tuple(warnings),
            suggested_fixes=suggested_fixes,
        )

    def verify_envelope(
        self,
        policy_envelope: PolicyEnvelope,
        advisory: AdvisoryEnvelope,
    ) -> VerificationReport:
        """Verify all per-regime SchedulePolicies in a PolicyEnvelope."""
        errors: list[str] = []
        warnings: list[str] = []
        suggested_fixes: dict[str, object] = {}

        if policy_envelope.op_name != advisory.op_name:
            errors.append(
                f"PolicyEnvelope.op_name {policy_envelope.op_name!r} does not match "
                f"AdvisoryEnvelope.op_name {advisory.op_name!r}"
            )

        regimes_by_tag = {r.regime_tag: r for r in advisory.regimes}
        for regime in advisory.regimes:
            if regime.regime_tag not in policy_envelope.policies:
                errors.append(
                    f"Missing SchedulePolicy for regime {regime.regime_tag!r} in PolicyEnvelope"
                )

        for tag, policy in policy_envelope.policies.items():
            sub_report = self.verify_policy(
                policy,
                advisory,
                regime=regimes_by_tag.get(tag),
            )
            errors.extend(sub_report.errors)
            warnings.extend(sub_report.warnings)
            if sub_report.suggested_fixes:
                if len(policy_envelope.policies) == 1:
                    suggested_fixes.update(sub_report.suggested_fixes)
                else:
                    suggested_fixes[tag] = dict(sub_report.suggested_fixes)

        return VerificationReport(
            passed=len(errors) == 0,
            errors=tuple(errors),
            warnings=tuple(warnings),
            suggested_fixes=suggested_fixes,
        )

    def verify(
        self,
        policy_or_envelope: PolicyEnvelope | SchedulePolicy,
        advisory: AdvisoryEnvelope,
    ) -> VerificationReport:
        """Dispatch verification for either a PolicyEnvelope or a single SchedulePolicy."""
        if isinstance(policy_or_envelope, PolicyEnvelope):
            return self.verify_envelope(policy_or_envelope, advisory)
        if isinstance(policy_or_envelope, SchedulePolicy):
            return self.verify_policy(policy_or_envelope, advisory)
        raise TypeError(
            f"Expected PolicyEnvelope or SchedulePolicy, got {type(policy_or_envelope).__name__}"
        )

    @staticmethod
    def verify_source_against_facets(
        source_text: str,
        mask: object,
        *,
        max_flag_depth: int = 15,
    ) -> VerificationReport:
        """Verify AscendC source text against the 7-facet MECE ConstraintRelaxationMask + hardware invariants."""
        return verify_source_against_facets(source_text, mask, max_flag_depth=max_flag_depth)


def verify_policy_envelope(
    policy_envelope: PolicyEnvelope,
    advisory: AdvisoryEnvelope,
) -> VerificationReport:
    """Verify a PolicyEnvelope against an AdvisoryEnvelope using SILVerifier."""
    return SILVerifier().verify_envelope(policy_envelope, advisory)


_REVERSE_ACK_RE = re.compile(r"\breverseAck\s*=\s*(\d+)")
_HW_MUTEX_CALL_RE = re.compile(r"\b(?:HwMutexLock|HwMutexUnlock)\s*(?:<[^>]*>)?\s*\(\s*(-?\d+)")


def _strip_cpp_comments_and_strings(src: str) -> str:
    """Replace C/C++ comments and string literals with spaces while keeping structure."""
    out: list[str] = []
    i = 0
    n = len(src)
    while i < n:
        ch = src[i]
        nxt = src[i + 1] if i + 1 < n else ""
        if ch == "/" and nxt == "/":
            i += 2
            while i < n and src[i] != "\n":
                out.append(" ")
                i += 1
        elif ch == "/" and nxt == "*":
            i += 2
            out.append("  ")
            while i < n:
                if src[i] == "*" and i + 1 < n and src[i + 1] == "/":
                    out.append("  ")
                    i += 2
                    break
                out.append("\n" if src[i] == "\n" else " ")
                i += 1
        elif ch in ('"', "'"):
            quote = ch
            out.append(" ")
            i += 1
            while i < n:
                c = src[i]
                out.append("\n" if c == "\n" else " ")
                i += 1
                if c == "\\":
                    if i < n:
                        out.append("\n" if src[i] == "\n" else " ")
                        i += 1
                elif c == quote:
                    break
        else:
            out.append(ch)
            i += 1
    return "".join(out)


@dataclass(frozen=True)
class _FlagEvent:
    kind: str  # "set" or "wait"
    flag_id: str


@dataclass(frozen=True)
class _TermStmt:
    kind: str  # "return", "continue", "break"


@dataclass(frozen=True)
class _BlockNode:
    stmts: tuple[object, ...]


@dataclass(frozen=True)
class _IfNode:
    then_branch: object
    else_branch: object | None
    cond: str = ""


@dataclass(frozen=True)
class _LoopNode:
    body: object


def _normalize_flag_id(raw_args: str) -> str:
    text = " ".join(raw_args.strip().split())
    if "," in text:
        text = text.split(",")[-1].strip()
    if re.fullmatch(r"-?\d+[uUlL]*", text):
        return text.rstrip("uUlL")
    return text or "0"


def _is_literal_flag_id(flag_id: str) -> bool:
    return bool(re.fullmatch(r"-?\d+", flag_id))


def _record_flag_issue(
    fid: str,
    msg: str,
    *,
    errors: list[str],
    warnings: list[str] | None = None,
    warn_symbolic: bool = False,
) -> None:
    if not _is_literal_flag_id(fid) and warn_symbolic and warnings is not None:
        warnings.append(msg)
    else:
        errors.append(msg)


def _consume_balanced(src: str, idx: int, open_ch: str, close_ch: str) -> int:
    """Given src[idx] == open_ch, return index right after matching close_ch."""
    n = len(src)
    depth = 0
    while idx < n:
        if src[idx] == open_ch:
            depth += 1
        elif src[idx] == close_ch:
            depth -= 1
            if depth == 0:
                return idx + 1
        idx += 1
    return n


def _match_flag_call_at(src: str, idx: int) -> tuple[_FlagEvent, int] | None:
    """Try to parse a SetFlag/WaitFlag call starting at idx."""
    m = re.match(
        r"(CrossCoreSetFlag|CrossCoreWaitFlag|SetFlag|WaitFlag)\b",
        src[idx:],
    )
    if not m:
        return None
    if idx > 0 and (src[idx - 1].isalnum() or src[idx - 1] == "_"):
        return None
    name = m.group(1)
    pos = idx + len(name)
    n = len(src)
    while pos < n and src[pos].isspace():
        pos += 1
    if pos < n and src[pos] == "<":
        pos = _consume_balanced(src, pos, "<", ">")
        while pos < n and src[pos].isspace():
            pos += 1
    if pos >= n or src[pos] != "(":
        return None
    end_paren = _consume_balanced(src, pos, "(", ")")
    args = src[pos + 1 : end_paren - 1]
    kind = "set" if "SetFlag" in name else "wait"
    return _FlagEvent(kind=kind, flag_id=_normalize_flag_id(args)), end_paren


def _strip_outer_parens(text: str) -> str:
    cur = text.strip()
    while cur.startswith("(") and cur.endswith(")") and _consume_balanced(cur, 0, "(", ")") == len(cur):
        cur = cur[1:-1].strip()
    return cur


def _classify_core_role_cond(cond: str) -> str | None:
    """Classify pure 1C2V core-role branch guards (ASCEND_IS_AIC / ASCEND_IS_AIV)."""
    text = " ".join((cond or "").strip().split())
    if re.match(r"constexpr\b", text):
        rem = text[len("constexpr") :].strip()
        if rem.startswith("(") and rem.endswith(")") and _consume_balanced(rem, 0, "(", ")") == len(rem):
            text = rem[1:-1].strip()
    text = _strip_outer_parens(text)
    negated = False
    if text.startswith("!"):
        negated = True
        text = _strip_outer_parens(text[1:])

    role: str | None = None
    if text == "ASCEND_IS_AIC" or re.fullmatch(
        r"(?:AscendC::)?g_coreType\s*==\s*(?:AscendC::)?AIC", text
    ):
        role = "aic"
    elif text == "ASCEND_IS_AIV" or re.fullmatch(
        r"(?:AscendC::)?g_coreType\s*==\s*(?:AscendC::)?AIV", text
    ):
        role = "aiv"
    elif re.fullmatch(r"(?:AscendC::)?g_coreType\s*!=\s*(?:AscendC::)?AIV", text):
        role = "aic"
    elif re.fullmatch(r"(?:AscendC::)?g_coreType\s*!=\s*(?:AscendC::)?AIC", text):
        role = "aiv"

    if role is None:
        return None
    if negated:
        return "aiv" if role == "aic" else "aic"
    return role


def _try_parse_if_at(src: str, idx: int) -> tuple[_IfNode, int] | None:
    """Try to parse `if [constexpr] (<cond> | ASCEND_IS_AIC | ASCEND_IS_AIV) ... [else ...]`."""
    n = len(src)
    if not re.match(r"if\b", src[idx:]) or (idx > 0 and (src[idx - 1].isalnum() or src[idx - 1] == "_")):
        return None
    pos = idx + 2
    while pos < n and src[pos].isspace():
        pos += 1
    if re.match(r"constexpr\b", src[pos:]):
        pos += len("constexpr")
        while pos < n and src[pos].isspace():
            pos += 1
    if pos < n and src[pos] == "(":
        end_paren = _consume_balanced(src, pos, "(", ")")
        cond = src[pos + 1 : end_paren - 1].strip()
        pos = end_paren
    else:
        m_role = re.match(r"(ASCEND_IS_AIC|ASCEND_IS_AIV)\b", src[pos:])
        if not m_role:
            return None
        cond = m_role.group(1)
        pos += len(cond)

    then_node, pos = _parse_single_stmt(src, pos)
    while pos < n and src[pos].isspace():
        pos += 1
    else_node: object | None = None
    if re.match(r"else\b", src[pos:]) and (
        pos == 0 or not (src[pos - 1].isalnum() or src[pos - 1] == "_")
    ):
        pos += 4
        else_node, pos = _parse_single_stmt(src, pos)
    return _IfNode(then_branch=then_node, else_branch=else_node, cond=cond), pos


def _parse_stmt_sequence(src: str, idx: int, stop_on_rbrace: bool) -> tuple[_BlockNode, int]:
    """Parse a sequence of C++ control-flow statements and flag events."""
    n = len(src)
    items: list[object] = []
    while idx < n:
        while idx < n and src[idx].isspace():
            idx += 1
        if idx >= n:
            break
        if src[idx] == "}":
            if stop_on_rbrace:
                idx += 1
            break
        if src[idx] == "{":
            sub_block, idx = _parse_stmt_sequence(src, idx + 1, stop_on_rbrace=True)
            items.append(sub_block)
            continue

        # Check control-flow keywords
        if_hit = _try_parse_if_at(src, idx)
        if if_hit is not None:
            if_node, idx = if_hit
            items.append(if_node)
            continue

        if re.match(r"(?:for|while)\b", src[idx:]) and (
            idx == 0 or not (src[idx - 1].isalnum() or src[idx - 1] == "_")
        ):
            kw = "for" if src[idx:].startswith("for") else "while"
            pos = idx + len(kw)
            while pos < n and src[pos].isspace():
                pos += 1
            if pos < n and src[pos] == "(":
                pos = _consume_balanced(src, pos, "(", ")")
                body_node, pos = _parse_single_stmt(src, pos)
                items.append(_LoopNode(body=body_node))
                idx = pos
                continue

        if re.match(r"do\b", src[idx:]) and (idx == 0 or not (src[idx - 1].isalnum() or src[idx - 1] == "_")):
            pos = idx + 2
            body_node, pos = _parse_single_stmt(src, pos)
            while pos < n and src[pos].isspace():
                pos += 1
            if re.match(r"while\b", src[pos:]):
                pos += 5
                while pos < n and src[pos].isspace():
                    pos += 1
                if pos < n and src[pos] == "(":
                    pos = _consume_balanced(src, pos, "(", ")")
                while pos < n and src[pos] != ";":
                    pos += 1
                if pos < n and src[pos] == ";":
                    pos += 1
            items.append(_LoopNode(body=body_node))
            idx = pos
            continue

        # Check flag call
        flag_hit = _match_flag_call_at(src, idx)
        if flag_hit is not None:
            ev, idx = flag_hit
            items.append(ev)
            continue

        # Check early terminator
        m_term = re.match(r"(return|continue|break)\b", src[idx:])
        if m_term and (idx == 0 or not (src[idx - 1].isalnum() or src[idx - 1] == "_")):
            kind = m_term.group(1)
            pos = idx + len(kind)
            # Scan until ';' for any inline flag call in return expression (rare)
            while pos < n and src[pos] != ";" and src[pos] != "}":
                pos += 1
            if pos < n and src[pos] == ";":
                pos += 1
            items.append(_TermStmt(kind=kind))
            idx = pos
            continue

        idx += 1

    return _BlockNode(stmts=tuple(items)), idx


def _parse_single_stmt(src: str, idx: int) -> tuple[object, int]:
    """Parse one statement (either a { ... } block or up to ';')."""
    n = len(src)
    while idx < n and src[idx].isspace():
        idx += 1
    if idx >= n:
        return _BlockNode(stmts=()), idx
    if src[idx] == "{":
        return _parse_stmt_sequence(src, idx + 1, stop_on_rbrace=True)
    if_hit = _try_parse_if_at(src, idx)
    if if_hit is not None:
        return if_hit

    items: list[object] = []
    while idx < n:
        if src[idx] == ";":
            idx += 1
            break
        if src[idx] == "}":
            break
        flag_hit = _match_flag_call_at(src, idx)
        if flag_hit is not None:
            ev, idx = flag_hit
            items.append(ev)
            continue
        m_term = re.match(r"(return|continue|break)\b", src[idx:])
        if m_term and (idx == 0 or not (src[idx - 1].isalnum() or src[idx - 1] == "_")):
            items.append(_TermStmt(kind=m_term.group(1)))
            idx += len(m_term.group(1))
            continue
        idx += 1
    return _BlockNode(stmts=tuple(items)), idx


def _merge_delta_maps(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0) + v
    return out


def _eval_cfg_node(
    node: object,
    *,
    max_flag_depth: int,
    errors: list[str],
    incoming_common: dict[str, int],
    incoming_aic: dict[str, int],
    incoming_aiv: dict[str, int],
    core_role: str | None = None,
    warnings: list[str] | None = None,
    warn_symbolic: bool = False,
) -> tuple[dict[str, int], dict[str, int], dict[str, int], str | None]:
    """Evaluate per-flag-ID Set/Wait deltas across CFG + 1C2V core-role branches."""
    if isinstance(node, _FlagEvent):
        if _is_literal_flag_id(node.flag_id):
            fid_int = int(node.flag_id)
            if fid_int < 0 or fid_int > max_flag_depth:
                errors.append(
                    f"Flag ID {fid_int} exceeds max_flag_depth={max_flag_depth} (or is negative)"
                )
        delta = {node.flag_id: 1 if node.kind == "set" else -1}
        if core_role == "aic":
            return ({}, delta, {}, None)
        if core_role == "aiv":
            return ({}, {}, delta, None)
        return (delta, {}, {}, None)

    if isinstance(node, _TermStmt):
        return ({}, {}, {}, node.kind)

    if isinstance(node, _LoopNode):
        b_common, b_aic, b_aiv, _ = _eval_cfg_node(
            node.body,
            max_flag_depth=max_flag_depth,
            errors=errors,
            incoming_common={},
            incoming_aic={},
            incoming_aiv={},
            core_role=core_role,
            warnings=warnings,
            warn_symbolic=warn_symbolic,
        )
        for fid, val in sorted(b_common.items()):
            if val != 0:
                _record_flag_issue(
                    fid,
                    f"Loop body has unbalanced CrossCore flag {fid!r} (net delta={val}; expected 0)",
                    errors=errors,
                    warnings=warnings,
                    warn_symbolic=warn_symbolic,
                )
        b_cv = _merge_delta_maps(b_aic, b_aiv)
        for fid, val in sorted(b_cv.items()):
            if val != 0:
                _record_flag_issue(
                    fid,
                    f"Loop body has unbalanced 1C2V CrossCore flag {fid!r} (net delta={val}; expected 0)",
                    errors=errors,
                    warnings=warnings,
                    warn_symbolic=warn_symbolic,
                )
        return ({}, {}, {}, None)

    if isinstance(node, _IfNode):
        branch_role = _classify_core_role_cond(node.cond)
        if branch_role in ("aic", "aiv") and core_role is None:
            opp_role = "aiv" if branch_role == "aic" else "aic"
            tc, taic, taiv, _tterm = _eval_cfg_node(
                node.then_branch,
                max_flag_depth=max_flag_depth,
                errors=errors,
                incoming_common={},
                incoming_aic=dict(incoming_aic),
                incoming_aiv=dict(incoming_aiv),
                core_role=branch_role,
                warnings=warnings,
                warn_symbolic=warn_symbolic,
            )
            ec: dict[str, int] = {}
            eaic: dict[str, int] = {}
            eaiv: dict[str, int] = {}
            if node.else_branch is not None:
                if (
                    isinstance(node.else_branch, _IfNode)
                    and _classify_core_role_cond(node.else_branch.cond) == opp_role
                    and node.else_branch.else_branch is None
                ):
                    else_target = node.else_branch.then_branch
                else:
                    else_target = node.else_branch
                ec, eaic, eaiv, _eterm = _eval_cfg_node(
                    else_target,
                    max_flag_depth=max_flag_depth,
                    errors=errors,
                    incoming_common={},
                    incoming_aic=dict(incoming_aic),
                    incoming_aiv=dict(incoming_aiv),
                    core_role=opp_role,
                    warnings=warnings,
                    warn_symbolic=warn_symbolic,
                )
            out_aic = _merge_delta_maps(taic, eaic)
            out_aiv = _merge_delta_maps(taiv, eaiv)
            if tc:
                if branch_role == "aic":
                    out_aic = _merge_delta_maps(out_aic, tc)
                else:
                    out_aiv = _merge_delta_maps(out_aiv, tc)
            if ec:
                if opp_role == "aic":
                    out_aic = _merge_delta_maps(out_aic, ec)
                else:
                    out_aiv = _merge_delta_maps(out_aiv, ec)
            return ({}, out_aic, out_aiv, None)

        tc, taic, taiv, tterm = _eval_cfg_node(
            node.then_branch,
            max_flag_depth=max_flag_depth,
            errors=errors,
            incoming_common=dict(incoming_common),
            incoming_aic=dict(incoming_aic),
            incoming_aiv=dict(incoming_aiv),
            core_role=core_role,
            warnings=warnings,
            warn_symbolic=warn_symbolic,
        )
        if node.else_branch is not None:
            ec, eaic, eaiv, eterm = _eval_cfg_node(
                node.else_branch,
                max_flag_depth=max_flag_depth,
                errors=errors,
                incoming_common=dict(incoming_common),
                incoming_aic=dict(incoming_aic),
                incoming_aiv=dict(incoming_aiv),
                core_role=core_role,
                warnings=warnings,
                warn_symbolic=warn_symbolic,
            )
        else:
            ec, eaic, eaiv, eterm = {}, {}, {}, None

        then_cv = _merge_delta_maps(taic, taiv)
        else_cv = _merge_delta_maps(eaic, eaiv)

        if tterm is not None and eterm is None:
            for fid, val in sorted(_merge_delta_maps(tc, then_cv).items()):
                if val != 0:
                    _record_flag_issue(
                        fid,
                        f"Early-exit ({tterm}) branch leaves unbalanced CrossCore flag {fid!r} (delta={val})",
                        errors=errors,
                        warnings=warnings,
                        warn_symbolic=warn_symbolic,
                    )
            return (ec, eaic, eaiv, None)
        if eterm is not None and tterm is None:
            for fid, val in sorted(_merge_delta_maps(ec, else_cv).items()):
                if val != 0:
                    _record_flag_issue(
                        fid,
                        f"Early-exit ({eterm}) branch leaves unbalanced CrossCore flag {fid!r} (delta={val})",
                        errors=errors,
                        warnings=warnings,
                        warn_symbolic=warn_symbolic,
                    )
            return (tc, taic, taiv, None)
        if tterm is not None and eterm is not None:
            for label, d_all in (("then", _merge_delta_maps(tc, then_cv)), ("else", _merge_delta_maps(ec, else_cv))):
                for fid, val in sorted(d_all.items()):
                    if val != 0:
                        _record_flag_issue(
                            fid,
                            f"Early-exit {label} branch leaves unbalanced CrossCore flag {fid!r} (delta={val})",
                            errors=errors,
                            warnings=warnings,
                            warn_symbolic=warn_symbolic,
                        )
            return ({}, {}, {}, tterm)

        for fid in sorted(set(tc) | set(ec)):
            td = tc.get(fid, 0)
            ed = ec.get(fid, 0)
            if td != ed:
                _record_flag_issue(
                    fid,
                    f"Unbalanced CrossCore flag {fid!r} across branches (then_delta={td}, else_delta={ed})",
                    errors=errors,
                    warnings=warnings,
                    warn_symbolic=warn_symbolic,
                )
        for fid in sorted(set(then_cv) | set(else_cv)):
            td = then_cv.get(fid, 0)
            ed = else_cv.get(fid, 0)
            if td != ed:
                _record_flag_issue(
                    fid,
                    f"Unbalanced 1C2V CrossCore flag {fid!r} across branches (then_delta={td}, else_delta={ed})",
                    errors=errors,
                    warnings=warnings,
                    warn_symbolic=warn_symbolic,
                )
        return (tc, taic, taiv, None)

    if isinstance(node, _BlockNode):
        run_common: dict[str, int] = {}
        run_aic: dict[str, int] = {}
        run_aiv: dict[str, int] = {}
        term_kind: str | None = None
        for stmt in node.stmts:
            comb_common = _merge_delta_maps(incoming_common, run_common)
            comb_aic = _merge_delta_maps(incoming_aic, run_aic)
            comb_aiv = _merge_delta_maps(incoming_aiv, run_aiv)
            dc, daic, daiv, st_term = _eval_cfg_node(
                stmt,
                max_flag_depth=max_flag_depth,
                errors=errors,
                incoming_common=comb_common,
                incoming_aic=comb_aic,
                incoming_aiv=comb_aiv,
                core_role=core_role,
                warnings=warnings,
                warn_symbolic=warn_symbolic,
            )
            if term_kind is not None:
                continue
            for target_run, target_inc, d_map in (
                (run_common, incoming_common, dc),
                (run_aic, incoming_aic, daic),
                (run_aiv, incoming_aiv, daiv),
            ):
                for fid, delta_val in d_map.items():
                    target_run[fid] = target_run.get(fid, 0) + delta_val
                    depth_now = target_inc.get(fid, 0) + target_run[fid]
                    if depth_now > max_flag_depth:
                        _record_flag_issue(
                            fid,
                            f"In-flight SetFlag depth {depth_now} for flag {fid!r} exceeds max_flag_depth={max_flag_depth}",
                            errors=errors,
                            warnings=warnings,
                            warn_symbolic=warn_symbolic,
                        )
            if st_term is not None:
                term_kind = st_term
                if st_term == "return" and core_role is None:
                    exit_common = _merge_delta_maps(incoming_common, run_common)
                    exit_cv = _merge_delta_maps(
                        _merge_delta_maps(incoming_aic, run_aic),
                        _merge_delta_maps(incoming_aiv, run_aiv),
                    )
                    for fid in sorted(set(exit_common) | set(exit_cv)):
                        c_val = exit_common.get(fid, 0)
                        cv_val = exit_cv.get(fid, 0)
                        if c_val != 0:
                            _record_flag_issue(
                                fid,
                                f"Return path leaves unbalanced CrossCore flag {fid!r} (net delta={c_val})",
                                errors=errors,
                                warnings=warnings,
                                warn_symbolic=warn_symbolic,
                            )
                        if cv_val != 0:
                            _record_flag_issue(
                                fid,
                                f"Return path leaves unbalanced 1C2V CrossCore flag {fid!r} (net delta={cv_val})",
                                errors=errors,
                                warnings=warnings,
                                warn_symbolic=warn_symbolic,
                            )
        return (run_common, run_aic, run_aiv, term_kind)

    return ({}, {}, {}, None)


def _eval_cfg_flag_balance(
    node: object,
    *,
    max_flag_depth: int,
    errors: list[str],
    incoming_delta: dict[str, int] | None = None,
    warnings: list[str] | None = None,
    warn_symbolic: bool = False,
) -> tuple[dict[str, int], str | None]:
    """Evaluate per-flag-ID Set/Wait deltas across CFG + 1C2V core-role branches."""
    inc = dict(incoming_delta or {})
    r_common, r_aic, r_aiv, term_kind = _eval_cfg_node(
        node,
        max_flag_depth=max_flag_depth,
        errors=errors,
        incoming_common=inc,
        incoming_aic={},
        incoming_aiv={},
        core_role=None,
        warnings=warnings,
        warn_symbolic=warn_symbolic,
    )
    r_cv = _merge_delta_maps(r_aic, r_aiv)
    # Prevent unguarded common flags from silently canceling core-role guarded flags
    # (unguarded WaitFlag runs on BOTH AIC and AIV, whereas SetFlag in AIC runs only on AIC).
    for fid in sorted(set(r_common) & (set(r_aic) | set(r_aiv))):
        c_val = r_common.get(fid, 0)
        cv_val = r_cv.get(fid, 0)
        if c_val != 0 and (r_aic.get(fid, 0) != 0 or r_aiv.get(fid, 0) != 0):
            _record_flag_issue(
                fid,
                f"Unguarded CrossCore flag {fid!r} (delta={c_val}) mixed with core-role guarded flag (1C2V delta={cv_val})",
                errors=errors,
                warnings=warnings,
                warn_symbolic=warn_symbolic,
            )
    combined = _merge_delta_maps(r_common, r_cv)
    return (combined, term_kind)


_FORBIDDEN_DMA_SOURCE_RE = re.compile(
    r"(?:L1\s*->\s*GM|GM\s*->\s*L0A|GM\s*->\s*L0B|\bL1_TO_GM\b|\bGM_TO_L0A\b|\bGM_TO_L0B\b)",
    re.IGNORECASE,
)
_UB_BYTES_DECL_RE = re.compile(r"\bub_bytes\s*=\s*(\d+)", re.IGNORECASE)
_L1_BYTES_DECL_RE = re.compile(r"\bl1_bytes\s*=\s*(\d+)", re.IGNORECASE)
_TILE_DIM_DECL_RE = re.compile(r"\b(?:CUBE_)?TILE_([MNK])\s*=\s*(\d+)")

_FACET_ALG_COMPUTE_RE = re.compile(
    r"\b(?:Reg::[A-Za-z_]\w*|Mmad|T\.gemm|asc_reduce_[A-Za-z0-9_]+)\b"
)
_FACET_TILE_STAGE_RE = re.compile(
    r"\b(?:CUBE_TILE_M|CUBE_TILE_N|CUBE_TILE_K|TILE_M|TILE_N|TILE_K|"
    r"tile_mnk|l1_stages|row_pack_factor|multi_row_unroll)\b\s*="
)
_FACET_EXEC_MODE_SPR_RE = re.compile(
    r"\b(?:__simt_vf__|__simd_vf__|__launch_bounds__|LoadDist::[A-Za-z0-9_]+|"
    r"StoreDist::[A-Za-z0-9_]+|DIST_[A-Z0-9_]+|SetCtrlSpr|SetMxScaleSPR|Simt::[A-Za-z0-9_]+)\b"
)
_FACET_SRAM_LAYOUT_ALIAS_RE = re.compile(
    r"\b(?:l1_slot_aliasing|L1_SLOT_ALIAS|inplace_l1_alias|reuse_l1_slot|alias_l1_slot|"
    r"UB_BANK_PAD_DATABLOCKS|ub_bank_pad_datablocks|ub_rw_shift_datablocks|l1_half_bank_split)\b"
    r"|TOTAL_L1_SIZE\s*>>"
)
_FACET_DMA_CACHE_RE = re.compile(
    r"\b(?:DataCopyPad|DataCopy|Nd2NzParams|Load2D|Load3D|LoadData2D|LoadData3D|"
    r"FixpipeParamsArch3510|FixPipeDeqScale|SetL2CacheHint|CACHE_MODE_DISABLE|dcci)\b"
)
_FACET_PIPELINE_SYNC_RE = re.compile(
    r"\b(?:STAGE_[A-Z0-9_]+|CrossCoreSetFlag|CrossCoreWaitFlag|PipeBarrier|LocalMemBar|"
    r"HwMutexLock|HwMutexUnlock|enUnitFlag|unitFlag)\b"
    r"|\b(?:cmatrixInitVal|initC)\s*=\s*false\b"
    r"|\b(?<![A-Za-z0-9_])(?:SetFlag|WaitFlag)\b(?=\s*(?:<|\())"
)
_FACET_GRID_WORKSPACE_RE = re.compile(
    r"\b(?:ComputeSwatSerpentineCoord|GroupedGemmRollingDispatch|endBlockIdx|"
    r"streamk|asc_atomic_cas|atomicAdd|atomicMax|atomicMin|SetAtomicAdd|gm_workspace)\b"
)


def _coerce_facet_mask(mask: object) -> ConstraintRelaxationMask:
    if mask is None:
        return ConstraintRelaxationMask.bottom()
    if isinstance(mask, ConstraintRelaxationMask):
        return mask
    if isinstance(mask, Mapping):
        return ConstraintRelaxationMask.from_dict(mask)
    if all(hasattr(mask, f) for f in VALID_FREEDOM_FACETS):
        return ConstraintRelaxationMask(
            **{f: bool(getattr(mask, f)) for f in VALID_FREEDOM_FACETS}
        )
    raise TypeError(
        f"freedom_mask must be ConstraintRelaxationMask, Mapping, or None, got {type(mask).__name__}"
    )


def verify_source_against_facets(
    source_text: str,
    mask: ConstraintRelaxationMask | Mapping[str, object] | None = None,
    *,
    max_flag_depth: int = 15,
    bounds: HardwareLegalityBounds | None = None,
) -> VerificationReport:
    """Enforce the MECE 7-facet AST partition locks + unconditional dav-3510 hardware legality invariants.

    When ``mask == ConstraintRelaxationMask.top()`` (``mask.is_unrestricted_top`` is True),
    ZERO shell syntax locks apply; only physical hardware legality invariants are enforced.
    """
    if int(max_flag_depth) < 1:
        raise ValueError(f"max_flag_depth must be >= 1, got {max_flag_depth}")

    hw_bounds = bounds if bounds is not None else HardwareLegalityBounds()
    eff_max_depth = min(int(max_flag_depth), hw_bounds.max_crosscore_depth)
    mask_obj = _coerce_facet_mask(mask)
    raw_text = str(source_text)
    stripped = _strip_cpp_comments_and_strings(raw_text)
    errors: list[str] = []
    warnings: list[str] = []
    suggested_fixes: dict[str, object] = {}

    # 1. Unconditional SILVerifier hardware legality invariants (always checked, including at top())
    for bad_dma in _FORBIDDEN_DMA_SOURCE_RE.findall(raw_text):
        errors.append(
            f"[hw_legality] Forbidden dav-3510 DMA path {bad_dma.strip()!r} "
            f"(forbidden={hw_bounds.forbidden_dma_paths})"
        )

    for m in _REVERSE_ACK_RE.finditer(stripped):
        depth_val = int(m.group(1))
        if depth_val > eff_max_depth:
            errors.append(
                f"[hw_legality] reverseAck={depth_val} exceeds max_flag_depth={eff_max_depth}"
            )
            suggested_fixes["max_flag_depth"] = eff_max_depth

    for m in _HW_MUTEX_CALL_RE.finditer(stripped):
        mutex_id = int(m.group(1))
        if mutex_id < 0 or mutex_id >= hw_bounds.hardware_mutex_count:
            errors.append(
                f"[hw_legality] Hardware mutex ID {mutex_id} outside valid range 0..{hw_bounds.hardware_mutex_count - 1}"
            )

    root_block, _ = _parse_stmt_sequence(stripped, 0, stop_on_rbrace=False)
    top_delta, _ = _eval_cfg_flag_balance(
        root_block,
        max_flag_depth=eff_max_depth,
        errors=errors,
        incoming_delta={},
        warnings=warnings,
        warn_symbolic=mask_obj.is_unrestricted_top,
    )
    for fid, net in sorted(top_delta.items()):
        if net != 0:
            _record_flag_issue(
                fid,
                f"[hw_legality] Unbalanced CrossCore SetFlag/WaitFlag count for flag {fid!r} "
                f"(net delta={net}; expected 0)",
                errors=errors,
                warnings=warnings,
                warn_symbolic=mask_obj.is_unrestricted_top,
            )

    for m in _UB_BYTES_DECL_RE.finditer(stripped):
        ub_val = int(m.group(1))
        if ub_val > hw_bounds.ub_unreserved_bytes:
            errors.append(
                f"[hw_legality] Declared ub_bytes={ub_val} exceeds max UB capacity {hw_bounds.ub_unreserved_bytes}"
            )
    for m in _L1_BYTES_DECL_RE.finditer(stripped):
        l1_val = int(m.group(1))
        if l1_val > hw_bounds.l1_bytes:
            errors.append(
                f"[hw_legality] Declared l1_bytes={l1_val} exceeds max L1 capacity {hw_bounds.l1_bytes}"
            )
    tile_vals: dict[str, int] = {}
    for m in _TILE_DIM_DECL_RE.finditer(stripped):
        dim_ch, dim_val = m.group(1), int(m.group(2))
        tile_vals[dim_ch] = max(tile_vals.get(dim_ch, 0), dim_val)
        if dim_val > hw_bounds.max_mmad_mnk:
            errors.append(
                f"[hw_legality] Tile dimension {dim_ch}={dim_val} exceeds max_mmad_mnk={hw_bounds.max_mmad_mnk}"
            )
    if "M" in tile_vals and "N" in tile_vals:
        l0c_req = tile_vals["M"] * tile_vals["N"] * 4
        if l0c_req > hw_bounds.l0c_bytes:
            errors.append(
                f"[hw_legality] Tile L0C footprint {tile_vals['M']}*{tile_vals['N']}*4={l0c_req} B "
                f"exceeds L0C limit {hw_bounds.l0c_bytes} B"
            )

    # 2. Shell syntax locks (bypassed completely when mask == ConstraintRelaxationMask.top())
    if not mask_obj.is_unrestricted_top:
        if not mask_obj.alg_compute_and_monoid:
            comp_hits = sorted(set(_FACET_ALG_COMPUTE_RE.findall(stripped)))
            if comp_hits:
                errors.append(
                    f"[facet:alg_compute_and_monoid=locked] Algorithmic compute/monoid edits are locked: {comp_hits}"
                )
                suggested_fixes["unlock_alg_compute_and_monoid"] = True

        if not mask_obj.tile_and_stage_knobs:
            tile_hits = sorted(set(_FACET_TILE_STAGE_RE.findall(stripped)))
            if tile_hits:
                errors.append(
                    f"[facet:tile_and_stage_knobs=locked] Tile/stage knob mutations are locked: {tile_hits}"
                )
                suggested_fixes["unlock_tile_and_stage_knobs"] = True

        if not mask_obj.exec_mode_and_spr:
            exec_hits = sorted(set(_FACET_EXEC_MODE_SPR_RE.findall(stripped)))
            if exec_hits:
                errors.append(
                    f"[facet:exec_mode_and_spr=locked] Execution mode / register layout / SPR edits are locked: {exec_hits}"
                )
                suggested_fixes["unlock_exec_mode_and_spr"] = True

        if not mask_obj.sram_layout_and_aliasing:
            alias_hits = sorted(set(_FACET_SRAM_LAYOUT_ALIAS_RE.findall(stripped)))
            if alias_hits:
                errors.append(
                    f"[facet:sram_layout_and_aliasing=locked] SRAM layout / bank pad / slot aliasing is locked: {alias_hits}"
                )
                suggested_fixes["unlock_sram_layout_and_aliasing"] = True

        if not mask_obj.dma_and_cache_hierarchy:
            dma_hits = sorted(set(_FACET_DMA_CACHE_RE.findall(stripped)))
            if dma_hits:
                errors.append(
                    f"[facet:dma_and_cache_hierarchy=locked] DMA / FixPipe / L2 cache hierarchy edits are locked: {dma_hits}"
                )
                suggested_fixes["unlock_dma_and_cache_hierarchy"] = True

        if not mask_obj.pipeline_and_sync_graph:
            sync_hits = sorted(
                {m.group(0).strip() for m in _FACET_PIPELINE_SYNC_RE.finditer(stripped)}
            )
            if sync_hits:
                errors.append(
                    f"[facet:pipeline_and_sync_graph=locked] Pipeline stage-DAG / async sync graph edits are locked: {sync_hits}"
                )
                suggested_fixes["unlock_pipeline_and_sync_graph"] = True

        if not mask_obj.grid_and_workspace_topology:
            grid_hits = sorted(set(_FACET_GRID_WORKSPACE_RE.findall(stripped)))
            if grid_hits:
                errors.append(
                    f"[facet:grid_and_workspace_topology=locked] Grid mapping / GM workspace / GM atomics edits are locked: {grid_hits}"
                )
                suggested_fixes["unlock_grid_and_workspace_topology"] = True

    return VerificationReport(
        passed=len(errors) == 0,
        errors=tuple(errors),
        warnings=tuple(warnings),
        suggested_fixes=suggested_fixes,
    )
