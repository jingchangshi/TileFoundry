"""Deterministic Ascend 950 (dav-3510) SIL hardware redline and pitfall verifier."""

from __future__ import annotations

import ast
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass

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
_CROSSCORE_FLAG_MODES: frozenset[str] = frozenset({"mode4_flag", "mode2_flag", "cross_core_flag"})
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
        fixes: dict[str, object] = {}
        bounds = advisory.legality_bounds
        if regime is None:
            regime = next((r for r in advisory.regimes if r.regime_tag == policy.regime_tag), None)
        tag, axes = policy.regime_tag, policy.topology_axes

        if policy.l1_estimated_bytes > bounds.l1_bytes:
            errors.append(
                f"[{tag}] L1 footprint {policy.l1_estimated_bytes} B exceeds dav-3510 L1 limit {bounds.l1_bytes} B"
            )
        if policy.l1_half_bank_split and policy.l1_stages >= 2 and policy.l1_estimated_bytes > 0:
            half_b = (policy.l1_estimated_bytes * math.ceil(policy.l1_stages / 2)) / policy.l1_stages
            if half_b > bounds.l1_half_bank_bytes:
                errors.append(
                    f"[{tag}] Half-L1 slot footprint {int(math.ceil(half_b))} B "
                    f"exceeds {bounds.l1_half_bank_bytes} B per ping/pong bank (l1_stages={policy.l1_stages})"
                )

        uses_cube = axes.uses_cube or bool(policy.cube_operand_dtype) or policy.cv_datapath != "none" or policy.l0c_buffers > 1
        bm, bn, _ = policy.tile_mnk
        if uses_cube and (l0c_b := bm * bn * 4 * policy.l0c_buffers) > bounds.l0c_bytes:
            errors.append(
                f"[{tag}] L0C tile footprint bm*bn*4*l0c_buffers={l0c_b} B exceeds dav-3510 L0C limit {bounds.l0c_bytes} B"
            )

        is_coupled_simt = axes.compute_engine == "coupled_1c2v" and axes.iteration_regularity == "irregular_simt"
        if policy.vector_mode in _SIMT_VECTOR_MODES and policy.ub_256kb_unlock:
            errors.append(f"[{tag}] ub_256kb_unlock=True is forbidden when vector_mode={policy.vector_mode!r}")
            fixes["ub_256kb_unlock"] = False
        elif is_coupled_simt and policy.ub_256kb_unlock:
            errors.append(
                f"[{tag}] ub_256kb_unlock=True is forbidden for coupled_1c2v + irregular_simt (32KB SIMT D-cache carveout)"
            )
            fixes["ub_256kb_unlock"] = False

        ub_limit = (
            (bounds.ub_unreserved_bytes if policy.ub_256kb_unlock else bounds.ub_simdvf_bytes)
            if policy.vector_mode == "simdvf"
            else bounds.ub_simtvf_bytes
        )
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
                    f"[{tag}] UB footprint {policy.ub_estimated_bytes} B exceeds {policy.vector_mode} limit {ub_limit} B"
                )
                if policy.vector_mode == "simdvf" and not policy.ub_256kb_unlock and policy.ub_estimated_bytes <= bounds.ub_unreserved_bytes:
                    fixes["ub_256kb_unlock"] = True

        forbidden_set = {_normalize_dma_path(p) for p in bounds.forbidden_dma_paths}
        for path in policy.dma_paths_used:
            if _normalize_dma_path(path) in forbidden_set:
                errors.append(
                    f"[{tag}] Forbidden dav-3510 DMA path {path!r} (forbidden={bounds.forbidden_dma_paths})"
                )

        if bounds.forbid_cube_int4 and policy.cube_operand_dtype.strip().lower() in _INT4_DTYPES and policy.w4_mode == "none":
            errors.append(
                f"[{tag}] Direct Cube INT4 ({policy.cube_operand_dtype!r}) with w4_mode='none' "
                f"is illegal on dav-3510; use 'aiv_unpack_f16' or 'mxa8w4_bitshift_x64'"
            )
            fixes["w4_mode"] = "aiv_unpack_f16"
            fixes["w4_mode_alternatives"] = ["aiv_unpack_f16", "mxa8w4_bitshift_x64"]

        if max(policy.tile_mnk) > bounds.max_mmad_mnk:
            errors.append(
                f"[{tag}] tile_mnk={policy.tile_mnk} exceeds max_mmad_mnk={bounds.max_mmad_mnk} "
                f"(silent uint16_t MMAD wrap hazard)"
            )

        if policy.sync_mode in ("unit_flag_2_3", "unit_flag") and policy.l0c_buffers > 1 and regime is not None and regime.has_unaligned_tail:
            errors.append(
                f"[{tag}] {policy.sync_mode} deadlock hazard with l0c_buffers={policy.l0c_buffers} > 1 and unaligned tail"
            )
            fixes["l0c_buffers"] = 1

        if policy.ub_row_stride_bytes > 0 and policy.ub_row_stride_bytes % 512 == 0 and policy.ub_bank_pad_datablocks == 0:
            warnings.append(
                f"[{tag}] UB row stride {policy.ub_row_stride_bytes} B is a multiple of 512 B "
                f"with ub_bank_pad_datablocks=0 (16-bank conflict risk)"
            )
            fixes.update({"ub_bank_pad_datablocks": 1, "ub_rw_shift_datablocks": 8})

        if policy.reverse_ack_period > bounds.max_crosscore_depth:
            if policy.sync_mode in _CROSSCORE_FLAG_MODES:
                errors.append(
                    f"[{tag}] reverse_ack_period={policy.reverse_ack_period} exceeds hardware "
                    f"max_crosscore_depth={bounds.max_crosscore_depth} on sync_mode={policy.sync_mode!r}"
                )
                fixes["reverse_ack_period"] = min(14, bounds.max_crosscore_depth)
            elif axes.compute_engine == "coupled_1c2v":
                errors.append(
                    f"[{tag}] coupled_1c2v axis reverse_ack_period={policy.reverse_ack_period} exceeds "
                    f"hardware max_crosscore_depth={bounds.max_crosscore_depth}"
                )
                fixes["reverse_ack_period"] = min(14, bounds.max_crosscore_depth)

        if not policy.ensure_empty_subblock_sync:
            if policy.cv_datapath.startswith(("l0c2ub_split", "l0c_to_ub_split")):
                errors.append(
                    f"[{tag}] cv_datapath={policy.cv_datapath!r} requires "
                    f"ensure_empty_subblock_sync=True to prevent deadlock on empty tail sub-blocks"
                )
                fixes["ensure_empty_subblock_sync"] = True
            elif axes.iteration_regularity == "ragged_group":
                errors.append(
                    f"[{tag}] topology_axes.iteration_regularity='ragged_group' requires "
                    f"ensure_empty_subblock_sync=True to prevent deadlock on empty tail sub-blocks"
                )
                fixes["ensure_empty_subblock_sync"] = True

        if policy.vector_mode in _SIMT_VECTOR_MODES and policy.simt_threads > 1024:
            warnings.append(
                f"[{tag}] simt_threads={policy.simt_threads} > 1024 on {policy.vector_mode}; "
                f"__launch_bounds__(512) recommended for 0 register spill"
            )
            fixes["simt_threads"] = 512

        fp32_thresh = _resolve_fp32_threshold(advisory.precision_thresholds)
        if policy.hf32_allowed and fp32_thresh is not None and fp32_thresh < _HF32_MIN_FP32_THRESHOLD:
            errors.append(f"[{tag}] hf32_allowed=True violates fp32 precision threshold {fp32_thresh:g} (< 2^-11)")
            fixes["hf32_allowed"] = False

        if policy.fast_unshifted_exp and advisory.has_nan_inf_cases:
            errors.append(f"[{tag}] fast_unshifted_exp=True is unsafe when advisory.has_nan_inf_cases=True")
            fixes["fast_unshifted_exp"] = False

        if axes.memory_ring == "bidirectional_cv_ring" and _is_prefill_regime(tag):
            rep_m = regime.representative_dims.get("M") if regime is not None else None
            if bm < 82 or (rep_m is not None and 0 < rep_m < 82):
                warnings.append(
                    f"[{tag}] mte2_bound_below_82: M < 82 on bidirectional_cv_ring Prefill (tile_m={bm}, rep_m={rep_m})"
                )

        return VerificationReport(
            passed=not errors, errors=tuple(errors), warnings=tuple(warnings), suggested_fixes=fixes
        )

    def verify_envelope(
        self,
        policy_envelope: PolicyEnvelope,
        advisory: AdvisoryEnvelope,
    ) -> VerificationReport:
        """Verify all per-regime SchedulePolicies in a PolicyEnvelope."""
        errors: list[str] = []
        warnings: list[str] = []
        fixes: dict[str, object] = {}
        if policy_envelope.op_name != advisory.op_name:
            errors.append(
                f"PolicyEnvelope.op_name {policy_envelope.op_name!r} does not match "
                f"AdvisoryEnvelope.op_name {advisory.op_name!r}"
            )
        regimes_by_tag = {r.regime_tag: r for r in advisory.regimes}
        for r in advisory.regimes:
            if r.regime_tag not in policy_envelope.policies:
                errors.append(f"Missing SchedulePolicy for regime {r.regime_tag!r} in PolicyEnvelope")
        for tag, policy in policy_envelope.policies.items():
            sub = self.verify_policy(policy, advisory, regime=regimes_by_tag.get(tag))
            errors.extend(sub.errors)
            warnings.extend(sub.warnings)
            if sub.suggested_fixes:
                if len(policy_envelope.policies) == 1:
                    fixes.update(sub.suggested_fixes)
                else:
                    fixes[tag] = dict(sub.suggested_fixes)
        return VerificationReport(
            passed=not errors, errors=tuple(errors), warnings=tuple(warnings), suggested_fixes=fixes
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
        raise TypeError(f"Expected PolicyEnvelope or SchedulePolicy, got {type(policy_or_envelope).__name__}")

    @staticmethod
    def verify_source_against_facets(
        source_text: str,
        mask: object,
        *,
        max_flag_depth: int = 15,
    ) -> VerificationReport:
        """Verify AscendC source text against 7-facet MECE locks and hardware invariants."""
        return verify_source_against_facets(source_text, mask, max_flag_depth=max_flag_depth)

    @staticmethod
    def verify_tilelang_ast(source_text: str) -> VerificationReport:
        """Verify TileLang Python DSL AST for legal T.Stage and T.Parallel usage."""
        return verify_tilelang_ast(source_text)


def verify_policy_envelope(policy_envelope: PolicyEnvelope, advisory: AdvisoryEnvelope) -> VerificationReport:
    """Verify a PolicyEnvelope against an AdvisoryEnvelope using SILVerifier."""
    return SILVerifier().verify_envelope(policy_envelope, advisory)


_REVERSE_ACK_RE = re.compile(r"\breverseAck\s*=\s*(\d+)")
_HW_MUTEX_CALL_RE = re.compile(r"\b(?:HwMutexLock|HwMutexUnlock)\s*(?:<[^>]*>)?\s*\(\s*(-?\d+)")


def _strip_cpp_comments_and_strings(src: str) -> str:
    out: list[str] = []
    i, n = 0, len(src)
    while i < n:
        ch, nxt = src[i], (src[i + 1] if i + 1 < n else "")
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
                if c == "\\" and i < n:
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
    kind: str
    flag_id: str
    family: str = "crosscore"
    pipe_pair: tuple[str, str] = ("", "")
    event_expr: str = ""


@dataclass(frozen=True)
class _TermStmt:
    kind: str


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
    return text.rstrip("uUlL") if re.fullmatch(r"-?\d+[uUlL]*", text) else (text or "0")


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
    n, depth = len(src), 0
    while idx < n:
        depth += (src[idx] == open_ch) - (src[idx] == close_ch)
        if depth == 0:
            return idx + 1
        idx += 1
    return n


def _split_top_level_args(raw_args: str) -> list[str]:
    parts: list[str] = []
    cur: list[str] = []
    p = a = b = 0
    for ch in raw_args:
        if ch == "(":
            p += 1
        elif ch == ")" and p > 0:
            p -= 1
        elif ch == "<":
            a += 1
        elif ch == ">" and a > 0:
            a -= 1
        elif ch == "[":
            b += 1
        elif ch == "]" and b > 0:
            b -= 1
        if ch == "," and p == a == b == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    tail = "".join(cur).strip()
    if tail or parts:
        parts.append(tail)
    return parts


def _is_literal_event_expr(expr: str) -> bool:
    return bool(re.fullmatch(r"(?:-?\d+[uUlL]*|EVENT_ID\d+)", expr.strip()))


def _match_flag_call_at(src: str, idx: int) -> tuple[_FlagEvent, int] | None:
    m = re.match(
        r"(CrossCoreSetFlag|CrossCoreWaitFlag|SetFlag|WaitFlag|asc_sync_notify|asc_sync_wait|asc_lock|asc_unlock)\b",
        src[idx:],
    )
    if not m or (idx > 0 and (src[idx - 1].isalnum() or src[idx - 1] == "_")):
        return None
    name = m.group(1)
    pos, n = idx + len(name), len(src)
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
    if name in ("asc_lock", "asc_unlock"):
        parts = _split_top_level_args(args)
        pipe = " ".join(parts[0].split()) if parts else ""
        slot = " ".join(parts[1].split()) if len(parts) > 1 else "0"
        return _FlagEvent("set" if name == "asc_lock" else "wait", f"lock:{pipe}:{slot}", "lock", (pipe, ""), slot), end_paren
    if name in ("asc_sync_notify", "asc_sync_wait"):
        parts = _split_top_level_args(args)
        src_p = " ".join(parts[0].split()) if parts else ""
        dst_p = " ".join(parts[1].split()) if len(parts) > 1 else ""
        ev_raw = " ".join(parts[2].split()) if len(parts) > 2 else "0"
        ev_norm = re.sub(r"^\(\s*event_t\s*\)\s*", "", ev_raw).strip()
        return (
            _FlagEvent("set" if name == "asc_sync_notify" else "wait", f"sync:{src_p}:{dst_p}:{ev_norm}", "asc_sync", (src_p, dst_p), ev_norm),
            end_paren,
        )
    return _FlagEvent("set" if "SetFlag" in name else "wait", _normalize_flag_id(args), "crosscore"), end_paren


def _strip_outer_parens(text: str) -> str:
    cur = text.strip()
    while cur.startswith("(") and cur.endswith(")") and _consume_balanced(cur, 0, "(", ")") == len(cur):
        cur = cur[1:-1].strip()
    return cur


def _classify_core_role_cond(cond: str) -> str | None:
    text = " ".join((cond or "").strip().split())
    if re.match(r"constexpr\b", text):
        rem = text[len("constexpr") :].strip()
        if rem.startswith("(") and rem.endswith(")") and _consume_balanced(rem, 0, "(", ")") == len(rem):
            text = rem[1:-1].strip()
    text = _strip_outer_parens(text)
    negated = text.startswith("!")
    if negated:
        text = _strip_outer_parens(text[1:])
    if text == "ASCEND_IS_AIC" or re.fullmatch(r"g_coreType\s*(?:==\s*AIC|!=\s*AIV)", text):
        role = "aic"
    elif text == "ASCEND_IS_AIV" or re.fullmatch(r"g_coreType\s*(?:==\s*AIV|!=\s*AIC)", text):
        role = "aiv"
    else:
        return None
    return ("aiv" if role == "aic" else "aic") if negated else role


def _word_boundary_ok(src: str, idx: int) -> bool:
    return idx == 0 or not (src[idx - 1].isalnum() or src[idx - 1] == "_")


def _try_parse_if_at(src: str, idx: int) -> tuple[_IfNode, int] | None:
    n = len(src)
    if not re.match(r"if\b", src[idx:]) or not _word_boundary_ok(src, idx):
        return None
    pos = idx + 2
    while pos < n and src[pos].isspace():
        pos += 1
    if re.match(r"constexpr\b", src[pos:]):
        pos += len("constexpr")
        while pos < n and src[pos].isspace():
            pos += 1
    if pos < n and src[pos] == "(":
        end_p = _consume_balanced(src, pos, "(", ")")
        cond, pos = src[pos + 1 : end_p - 1].strip(), end_p
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
    if re.match(r"else\b", src[pos:]) and _word_boundary_ok(src, pos):
        else_node, pos = _parse_single_stmt(src, pos + 4)
    return _IfNode(then_branch=then_node, else_branch=else_node, cond=cond), pos


def _parse_stmt_sequence(src: str, idx: int, stop_on_rbrace: bool) -> tuple[_BlockNode, int]:
    n, items = len(src), []
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
            sub, idx = _parse_stmt_sequence(src, idx + 1, stop_on_rbrace=True)
            items.append(sub)
        elif (if_hit := _try_parse_if_at(src, idx)) is not None:
            items.append(if_hit[0])
            idx = if_hit[1]
        elif re.match(r"(?:for|while)\b", src[idx:]) and _word_boundary_ok(src, idx):
            pos = idx + (3 if src[idx:].startswith("for") else 5)
            while pos < n and src[pos].isspace():
                pos += 1
            if pos < n and src[pos] == "(":
                pos = _consume_balanced(src, pos, "(", ")")
                body_node, idx = _parse_single_stmt(src, pos)
                items.append(_LoopNode(body=body_node))
            else:
                idx += 1
        elif re.match(r"do\b", src[idx:]) and _word_boundary_ok(src, idx):
            body_node, pos = _parse_single_stmt(src, idx + 2)
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
        elif (flag_hit := _match_flag_call_at(src, idx)) is not None:
            items.append(flag_hit[0])
            idx = flag_hit[1]
        elif (m_term := re.match(r"(return|continue|break)\b", src[idx:])) and _word_boundary_ok(src, idx):
            pos = idx + len(m_term.group(1))
            while pos < n and src[pos] not in ";}":
                pos += 1
            if pos < n and src[pos] == ";":
                pos += 1
            items.append(_TermStmt(kind=m_term.group(1)))
            idx = pos
        else:
            idx += 1
    return _BlockNode(stmts=tuple(items)), idx


def _parse_single_stmt(src: str, idx: int) -> tuple[object, int]:
    n = len(src)
    while idx < n and src[idx].isspace():
        idx += 1
    if idx >= n:
        return _BlockNode(stmts=()), idx
    if src[idx] == "{":
        return _parse_stmt_sequence(src, idx + 1, stop_on_rbrace=True)
    if (if_hit := _try_parse_if_at(src, idx)) is not None:
        return if_hit
    items: list[object] = []
    while idx < n and src[idx] not in ";}":
        if (flag_hit := _match_flag_call_at(src, idx)) is not None:
            items.append(flag_hit[0])
            idx = flag_hit[1]
        elif (m_term := re.match(r"(return|continue|break)\b", src[idx:])) and _word_boundary_ok(src, idx):
            items.append(_TermStmt(kind=m_term.group(1)))
            idx += len(m_term.group(1))
        else:
            idx += 1
    if idx < n and src[idx] == ";":
        idx += 1
    return _BlockNode(stmts=tuple(items)), idx


def _merge_delta_maps(a: dict[str, int], b: dict[str, int]) -> dict[str, int]:
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0) + v
    return out


def _collect_asc_sync_analysis(root_block: object, errors: list[str]) -> frozenset[tuple[str, str]]:
    notify_counts: dict[tuple[str, str], int] = {}
    wait_counts: dict[tuple[str, str], int] = {}
    notify_evs: dict[tuple[str, str], set[str]] = {}
    wait_evs: dict[tuple[str, str], set[str]] = {}
    in_loop_pairs: set[tuple[str, str]] = set()
    out_loop_pairs: set[tuple[str, str]] = set()
    in_loop_sym: set[tuple[str, str]] = set()

    def _walk(node: object, in_loop: bool) -> None:
        if isinstance(node, _FlagEvent) and node.family == "asc_sync":
            pair = node.pipe_pair
            counts, evs = (notify_counts, notify_evs) if node.kind == "set" else (wait_counts, wait_evs)
            counts[pair] = counts.get(pair, 0) + 1
            evs.setdefault(pair, set()).add(node.event_expr)
            if in_loop:
                in_loop_pairs.add(pair)
                if not _is_literal_event_expr(node.event_expr):
                    in_loop_sym.add(pair)
            else:
                out_loop_pairs.add(pair)
        elif isinstance(node, _LoopNode):
            _walk(node.body, True)
        elif isinstance(node, _IfNode):
            _walk(node.then_branch, in_loop)
            if node.else_branch is not None:
                _walk(node.else_branch, in_loop)
        elif isinstance(node, _BlockNode):
            for stmt in node.stmts:
                _walk(stmt, in_loop)

    _walk(root_block, False)
    pingpong = set(in_loop_pairs & out_loop_pairs)
    for pair in sorted(set(notify_counts) | set(wait_counts)):
        nc, wc = notify_counts.get(pair, 0), wait_counts.get(pair, 0)
        if nc == 0 or wc == 0:
            errors.append(
                f"[hw_legality] Unpaired asc_sync_notify/wait on pipe pair {pair[0]}->{pair[1]} (notify={nc}, wait={wc})"
            )
            continue
        n_evs, w_evs = notify_evs.get(pair, set()), wait_evs.get(pair, set())
        if all(_is_literal_event_expr(ev) for ev in n_evs | w_evs) and n_evs != w_evs:
            errors.append(
                f"[hw_legality] Mismatched asc_sync_notify/wait event IDs on {pair[0]}->{pair[1]} "
                f"(notify={sorted(n_evs)}, wait={sorted(w_evs)})"
            )
        elif pair in in_loop_sym:
            pingpong.add(pair)
    return frozenset(pingpong)


def _eval_cfg_flag_balance(
    root: object,
    *,
    max_flag_depth: int,
    errors: list[str],
    incoming_delta: dict[str, int] | None = None,
    warnings: list[str] | None = None,
    warn_symbolic: bool = False,
    skip_asc_sync_pairs: frozenset[tuple[str, str]] = frozenset(),
) -> tuple[dict[str, int], str | None]:
    """Evaluate per-flag-ID Set/Wait, asc_sync, and asc_lock deltas across CFG branches."""

    def _rec(fid: str, msg: str) -> None:
        _record_flag_issue(fid, msg, errors=errors, warnings=warnings, warn_symbolic=warn_symbolic)

    def _emit_nonzero(dmap: dict[str, int], fmt: str) -> None:
        for fid, val in sorted(dmap.items()):
            if val != 0:
                _rec(fid, fmt.format(fid=repr(fid), val=val))

    def _eval(
        node: object,
        inc_c: dict[str, int],
        inc_aic: dict[str, int],
        inc_aiv: dict[str, int],
        role: str | None,
    ) -> tuple[dict[str, int], dict[str, int], dict[str, int], str | None]:
        if isinstance(node, _FlagEvent):
            if node.family == "asc_sync" and node.pipe_pair in skip_asc_sync_pairs:
                return ({}, {}, {}, None)
            if node.family == "crosscore" and _is_literal_flag_id(node.flag_id):
                if not (0 <= (fid_int := int(node.flag_id)) <= max_flag_depth):
                    errors.append(f"Flag ID {fid_int} exceeds max_flag_depth={max_flag_depth} (or is negative)")
            delta = {node.flag_id: 1 if node.kind == "set" else -1}
            if role == "aic":
                return ({}, delta, {}, None)
            if role == "aiv":
                return ({}, {}, delta, None)
            return (delta, {}, {}, None)

        if isinstance(node, _TermStmt):
            return ({}, {}, {}, node.kind)

        if isinstance(node, _LoopNode):
            bc, baic, baiv, _ = _eval(node.body, {}, {}, {}, role)
            _emit_nonzero(bc, "Loop body has unbalanced CrossCore flag {fid} (net delta={val}; expected 0)")
            _emit_nonzero(_merge_delta_maps(baic, baiv), "Loop body has unbalanced 1C2V CrossCore flag {fid} (net delta={val}; expected 0)")
            return ({}, {}, {}, None)

        if isinstance(node, _IfNode):
            branch_role = _classify_core_role_cond(node.cond)
            if branch_role in ("aic", "aiv") and role is None:
                opp = "aiv" if branch_role == "aic" else "aic"
                tc, taic, taiv, _ = _eval(node.then_branch, {}, dict(inc_aic), dict(inc_aiv), branch_role)
                ec: dict[str, int] = {}
                eaic: dict[str, int] = {}
                eaiv: dict[str, int] = {}
                if node.else_branch is not None:
                    else_target = (
                        node.else_branch.then_branch
                        if isinstance(node.else_branch, _IfNode)
                        and _classify_core_role_cond(node.else_branch.cond) == opp
                        and node.else_branch.else_branch is None
                        else node.else_branch
                    )
                    ec, eaic, eaiv, _ = _eval(else_target, {}, dict(inc_aic), dict(inc_aiv), opp)
                out_aic, out_aiv = _merge_delta_maps(taic, eaic), _merge_delta_maps(taiv, eaiv)
                for d_c, d_role in ((tc, branch_role), (ec, opp)):
                    if d_c:
                        if d_role == "aic":
                            out_aic = _merge_delta_maps(out_aic, d_c)
                        else:
                            out_aiv = _merge_delta_maps(out_aiv, d_c)
                return ({}, out_aic, out_aiv, None)

            tc, taic, taiv, tterm = _eval(node.then_branch, dict(inc_c), dict(inc_aic), dict(inc_aiv), role)
            if node.else_branch is not None:
                ec, eaic, eaiv, eterm = _eval(node.else_branch, dict(inc_c), dict(inc_aic), dict(inc_aiv), role)
            else:
                ec, eaic, eaiv, eterm = {}, {}, {}, None
            then_cv, else_cv = _merge_delta_maps(taic, taiv), _merge_delta_maps(eaic, eaiv)

            if tterm is not None and eterm is None:
                _emit_nonzero(_merge_delta_maps(tc, then_cv), f"Early-exit ({tterm}) branch leaves unbalanced CrossCore flag {{fid}} (delta={{val}})")
                return (ec, eaic, eaiv, None)
            if eterm is not None and tterm is None:
                _emit_nonzero(_merge_delta_maps(ec, else_cv), f"Early-exit ({eterm}) branch leaves unbalanced CrossCore flag {{fid}} (delta={{val}})")
                return (tc, taic, taiv, None)
            if tterm is not None and eterm is not None:
                for label, d_all in (("then", _merge_delta_maps(tc, then_cv)), ("else", _merge_delta_maps(ec, else_cv))):
                    _emit_nonzero(d_all, f"Early-exit {label} branch leaves unbalanced CrossCore flag {{fid}} (delta={{val}})")
                return ({}, {}, {}, tterm)

            for prefix, m_t, m_e in (("CrossCore", tc, ec), ("1C2V CrossCore", then_cv, else_cv)):
                for fid in sorted(set(m_t) | set(m_e)):
                    if (td := m_t.get(fid, 0)) != (ed := m_e.get(fid, 0)):
                        _rec(fid, f"Unbalanced {prefix} flag {fid!r} across branches (then_delta={td}, else_delta={ed})")
            return (tc, taic, taiv, None)

        if isinstance(node, _BlockNode):
            rc: dict[str, int] = {}
            raic: dict[str, int] = {}
            raiv: dict[str, int] = {}
            term_kind: str | None = None
            for stmt in node.stmts:
                dc, daic, daiv, st_term = _eval(
                    stmt,
                    _merge_delta_maps(inc_c, rc),
                    _merge_delta_maps(inc_aic, raic),
                    _merge_delta_maps(inc_aiv, raiv),
                    role,
                )
                if term_kind is not None:
                    continue
                for trun, tinc, dmap in ((rc, inc_c, dc), (raic, inc_aic, daic), (raiv, inc_aiv, daiv)):
                    for fid, dv in dmap.items():
                        trun[fid] = trun.get(fid, 0) + dv
                        depth = tinc.get(fid, 0) + trun[fid]
                        if not fid.startswith(("lock:", "sync:")) and depth > max_flag_depth:
                            _rec(fid, f"In-flight SetFlag depth {depth} for flag {fid!r} exceeds max_flag_depth={max_flag_depth}")
                if st_term is not None:
                    term_kind = st_term
                    if st_term == "return" and role is None:
                        _emit_nonzero(_merge_delta_maps(inc_c, rc), "Return path leaves unbalanced CrossCore flag {fid} (net delta={val})")
                        _emit_nonzero(
                            _merge_delta_maps(_merge_delta_maps(inc_aic, raic), _merge_delta_maps(inc_aiv, raiv)),
                            "Return path leaves unbalanced 1C2V CrossCore flag {fid} (net delta={val})",
                        )
            return (rc, raic, raiv, term_kind)

        return ({}, {}, {}, None)

    r_common, r_aic, r_aiv, term_kind = _eval(root, dict(incoming_delta or {}), {}, {}, None)
    r_cv = _merge_delta_maps(r_aic, r_aiv)
    for fid in sorted(set(r_common) & (set(r_aic) | set(r_aiv))):
        c_val = r_common.get(fid, 0)
        if c_val != 0 and (r_aic.get(fid, 0) != 0 or r_aiv.get(fid, 0) != 0):
            _rec(
                fid,
                f"Unguarded CrossCore flag {fid!r} (delta={c_val}) mixed with core-role guarded flag (1C2V delta={r_cv.get(fid, 0)})",
            )
    return _merge_delta_maps(r_common, r_cv), term_kind


_FORBIDDEN_DMA_SOURCE_RE = re.compile(
    r"(?:L1\s*->\s*GM|GM\s*->\s*L0A|GM\s*->\s*L0B|\bL1_TO_GM\b|\bGM_TO_L0A\b|\bGM_TO_L0B\b)",
    re.IGNORECASE,
)
_UB_BYTES_DECL_RE = re.compile(r"\bub_bytes\s*=\s*(\d+)", re.IGNORECASE)
_L1_BYTES_DECL_RE = re.compile(r"\bl1_bytes\s*=\s*(\d+)", re.IGNORECASE)
_TILE_DIM_DECL_RE = re.compile(r"\b(?:CUBE_)?TILE_([MNK])\s*=\s*(\d+)")

_FACET_RULES: tuple[tuple[str, re.Pattern[str], str], ...] = (
    (
        "alg_compute_and_monoid",
        re.compile(r"\b(?:Reg::[A-Za-z_]\w*|Mmad|T\.gemm|ascend_gemm_l1|asc_mmad(?:_mx|_sparse)?|asc_reduce_[A-Za-z0-9_]+)\b"),
        "Algorithmic compute/monoid edits are locked",
    ),
    (
        "tile_and_stage_knobs",
        re.compile(r"\b(?:CUBE_TILE_M|CUBE_TILE_N|CUBE_TILE_K|TILE_M|TILE_N|TILE_K|tile_mnk|l1_stages|row_pack_factor|multi_row_unroll)\b\s*="),
        "Tile/stage knob mutations are locked",
    ),
    (
        "exec_mode_and_spr",
        re.compile(
            r"\b(?:__simt_vf__|__simd_vf__|__launch_bounds__|LoadDist::[A-Za-z0-9_]+|"
            r"StoreDist::[A-Za-z0-9_]+|DIST_[A-Z0-9_]+|SetCtrlSpr|SetMxScaleSPR|Simt::[A-Za-z0-9_]+|simd_inst::[A-Za-z0-9_]+)\b"
        ),
        "Execution mode / register layout / SPR edits are locked",
    ),
    (
        "sram_layout_and_aliasing",
        re.compile(
            r"\b(?:l1_slot_aliasing|L1_SLOT_ALIAS|inplace_l1_alias|reuse_l1_slot|alias_l1_slot|"
            r"UB_BANK_PAD_DATABLOCKS|ub_bank_pad_datablocks|ub_rw_shift_datablocks|l1_half_bank_split)\b|TOTAL_L1_SIZE\s*>>"
        ),
        "SRAM layout / bank pad / slot aliasing is locked",
    ),
    (
        "dma_and_cache_hierarchy",
        re.compile(
            r"\b(?:DataCopyPad|DataCopy|Nd2NzParams|Load2D|Load3D|LoadData2D|LoadData3D|"
            r"FixpipeParamsArch3510|FixPipeDeqScale|SetL2CacheHint|CACHE_MODE_DISABLE|dcci|asc_copy_[A-Za-z0-9_]+)\b"
        ),
        "DMA / FixPipe / L2 cache hierarchy edits are locked",
    ),
    (
        "pipeline_and_sync_graph",
        re.compile(
            r"\b(?:STAGE_[A-Z0-9_]+|CrossCoreSetFlag|CrossCoreWaitFlag|PipeBarrier|LocalMemBar|"
            r"HwMutexLock|HwMutexUnlock|enUnitFlag|unitFlag|asc_sync_notify|asc_sync_wait|"
            r"asc_sync_block_arrive|asc_sync_block_wait|asc_sync_pipe|asc_lock|asc_unlock)\b"
            r"|\b(?:cmatrixInitVal|initC)\s*=\s*false\b|\b(?<![A-Za-z0-9_])(?:SetFlag|WaitFlag)\b(?=\s*(?:<|\())"
        ),
        "Pipeline stage-DAG / async sync graph edits are locked",
    ),
    (
        "grid_and_workspace_topology",
        re.compile(
            r"\b(?:ComputeSwatSerpentineCoord|GroupedGemmRollingDispatch|endBlockIdx|"
            r"streamk|asc_atomic_cas|atomicAdd|atomicMax|atomicMin|SetAtomicAdd|gm_workspace)\b"
        ),
        "Grid mapping / GM workspace / GM atomics edits are locked",
    ),
)


def _coerce_facet_mask(mask: object) -> ConstraintRelaxationMask:
    if mask is None:
        return ConstraintRelaxationMask.bottom()
    if isinstance(mask, ConstraintRelaxationMask):
        return mask
    if isinstance(mask, Mapping):
        return ConstraintRelaxationMask.from_dict(mask)
    if all(hasattr(mask, f) for f in VALID_FREEDOM_FACETS):
        return ConstraintRelaxationMask(**{f: bool(getattr(mask, f)) for f in VALID_FREEDOM_FACETS})
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
    """Enforce MECE 7-facet AST locks and unconditional dav-3510 hardware invariants.

    When ``mask == ConstraintRelaxationMask.top()`` (``mask.is_unrestricted_top`` is True),
    ZERO shell syntax locks apply; only physical hardware legality invariants are enforced.
    """
    if int(max_flag_depth) < 1:
        raise ValueError(f"max_flag_depth must be >= 1, got {max_flag_depth}")

    hw = bounds if bounds is not None else HardwareLegalityBounds()
    eff_max_depth = min(int(max_flag_depth), hw.max_crosscore_depth)
    mask_obj = _coerce_facet_mask(mask)
    raw_text = str(source_text)
    stripped = _strip_cpp_comments_and_strings(raw_text)
    errors: list[str] = []
    warnings: list[str] = []
    fixes: dict[str, object] = {}

    for bad_dma in _FORBIDDEN_DMA_SOURCE_RE.findall(raw_text):
        errors.append(
            f"[hw_legality] Forbidden dav-3510 DMA path {bad_dma.strip()!r} (forbidden={hw.forbidden_dma_paths})"
        )
    for m in _REVERSE_ACK_RE.finditer(stripped):
        if (dv := int(m.group(1))) > eff_max_depth:
            errors.append(f"[hw_legality] reverseAck={dv} exceeds max_flag_depth={eff_max_depth}")
            fixes["max_flag_depth"] = eff_max_depth
    for m in _HW_MUTEX_CALL_RE.finditer(stripped):
        if not (0 <= (mid := int(m.group(1))) < hw.hardware_mutex_count):
            errors.append(f"[hw_legality] Hardware mutex ID {mid} outside valid range 0..{hw.hardware_mutex_count - 1}")

    root_block, _ = _parse_stmt_sequence(stripped, 0, stop_on_rbrace=False)
    pingpong_pairs = _collect_asc_sync_analysis(root_block, errors)
    top_delta, _ = _eval_cfg_flag_balance(
        root_block, max_flag_depth=eff_max_depth, errors=errors, warnings=warnings, skip_asc_sync_pairs=pingpong_pairs
    )
    for fid, net in sorted(top_delta.items()):
        if net != 0:
            _record_flag_issue(
                fid,
                f"[hw_legality] Unbalanced CrossCore/asc_sync count for flag {fid!r} (net delta={net}; expected 0)",
                errors=errors,
                warnings=warnings,
            )

    for m in _UB_BYTES_DECL_RE.finditer(stripped):
        if (uv := int(m.group(1))) > hw.ub_unreserved_bytes:
            errors.append(f"[hw_legality] Declared ub_bytes={uv} exceeds max UB capacity {hw.ub_unreserved_bytes}")
    for m in _L1_BYTES_DECL_RE.finditer(stripped):
        if (lv := int(m.group(1))) > hw.l1_bytes:
            errors.append(f"[hw_legality] Declared l1_bytes={lv} exceeds max L1 capacity {hw.l1_bytes}")
    tile_vals: dict[str, int] = {}
    for m in _TILE_DIM_DECL_RE.finditer(stripped):
        dim_ch, dim_val = m.group(1), int(m.group(2))
        tile_vals[dim_ch] = max(tile_vals.get(dim_ch, 0), dim_val)
        if dim_val > hw.max_mmad_mnk:
            errors.append(f"[hw_legality] Tile dimension {dim_ch}={dim_val} exceeds max_mmad_mnk={hw.max_mmad_mnk}")
    if "M" in tile_vals and "N" in tile_vals and (l0c_req := tile_vals["M"] * tile_vals["N"] * 4) > hw.l0c_bytes:
        errors.append(
            f"[hw_legality] Tile L0C footprint {tile_vals['M']}*{tile_vals['N']}*4={l0c_req} B exceeds L0C limit {hw.l0c_bytes} B"
        )

    if not mask_obj.is_unrestricted_top:
        for facet, pat, msg in _FACET_RULES:
            if not getattr(mask_obj, facet):
                hits = sorted({m.group(0).strip() for m in pat.finditer(stripped)})
                if hits:
                    errors.append(f"[facet:{facet}=locked] {msg}: {hits}")
                    fixes[f"unlock_{facet}"] = True

    return VerificationReport(passed=not errors, errors=tuple(errors), warnings=tuple(warnings), suggested_fixes=fixes)


def _is_t_dsl_call(node: ast.AST, *attrs: str) -> bool:
    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in attrs
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "T"
    )


def _extract_int_constant(node: ast.AST) -> int | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return int(node.value)
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, int)
        and not isinstance(node.operand.value, bool)
    ):
        return -int(node.operand.value)
    return None


def _pipelined_has_stage_offset(call: ast.Call) -> bool:
    stage_expr: ast.AST | None = call.args[1] if len(call.args) >= 2 else next(
        (kw.value for kw in call.keywords if kw.arg == "num_stages"), None
    )
    if stage_expr is not None:
        sv = _extract_int_constant(stage_expr)
        return sv >= 2 if sv is not None else True
    for kw in call.keywords:
        if kw.arg == "annotations" and isinstance(kw.value, ast.Dict):
            for k_node, v_node in zip(kw.value.keys, kw.value.values):
                if isinstance(k_node, ast.Constant) and k_node.value == "enable_offset" and isinstance(v_node, ast.Constant) and bool(v_node.value):
                    return True
    return False


def _extract_stage_index(call: ast.Call) -> int | None:
    if call.args:
        return _extract_int_constant(call.args[0])
    return next((_extract_int_constant(kw.value) for kw in call.keywords if kw.arg in ("stage", "stage_id")), None)


def verify_tilelang_ast(source_text: str) -> VerificationReport:
    """Enforce Stage-A TileLang Python AST rules for T.Stage and T.Parallel bodies."""
    try:
        tree = ast.parse(source_text)
    except SyntaxError as exc:
        return VerificationReport(passed=False, errors=(f"[tilelang_ast] SyntaxError at line {exc.lineno}: {exc.msg}",))

    errors: list[str] = []

    def _visit(node: ast.AST, in_par: bool, in_simd: bool, in_pipe: bool) -> None:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for child in ast.iter_child_nodes(node):
                _visit(child, False, False, False)
            return
        if isinstance(node, (ast.With, ast.AsyncWith)):
            next_simd = in_simd or any(_is_t_dsl_call(item.context_expr, "SimdVF") for item in node.items)
            for item in node.items:
                _visit(item.context_expr, in_par, in_simd, in_pipe)
                if item.optional_vars is not None:
                    _visit(item.optional_vars, in_par, in_simd, in_pipe)
            for stmt in node.body:
                _visit(stmt, in_par, next_simd, in_pipe)
            return
        if isinstance(node, (ast.For, ast.AsyncFor)):
            _visit(node.target, in_par, in_simd, in_pipe)
            _visit(node.iter, in_par, in_simd, in_pipe)
            next_par = in_par or _is_t_dsl_call(node.iter, "Parallel")
            next_pipe = in_pipe or (
                isinstance(node.iter, ast.Call)
                and _is_t_dsl_call(node.iter, "Pipelined")
                and _pipelined_has_stage_offset(node.iter)
            )
            for stmt in node.body:
                _visit(stmt, next_par, in_simd, next_pipe)
            for stmt in node.orelse:
                _visit(stmt, in_par, in_simd, in_pipe)
            return
        if isinstance(node, ast.Call):
            if _is_t_dsl_call(node, "Stage"):
                if in_simd:
                    errors.append(f"[tilelang_ast] T.Stage(...) is forbidden inside T.SimdVF() (line {node.lineno})")
                if in_par:
                    errors.append(f"[tilelang_ast] T.Stage(...) is forbidden inside T.Parallel() (line {node.lineno})")
                if (s_idx := _extract_stage_index(node)) is not None and s_idx != 0 and not in_pipe:
                    errors.append(
                        f"[tilelang_ast] Nonzero T.Stage({s_idx}) requires an enclosing T.Pipelined(..., num_stages>=2) context (line {node.lineno})"
                    )
            if in_par and _is_t_dsl_call(node, "copy", "gemm", "gemm_v0"):
                assert isinstance(node.func, ast.Attribute)
                errors.append(
                    f"[tilelang_ast] Block-level T.{node.func.attr}(...) is forbidden inside T.Parallel() loop body (line {node.lineno})"
                )
        for child in ast.iter_child_nodes(node):
            _visit(child, in_par, in_simd, in_pipe)

    _visit(tree, False, False, False)
    return VerificationReport(passed=not errors, errors=tuple(errors))
