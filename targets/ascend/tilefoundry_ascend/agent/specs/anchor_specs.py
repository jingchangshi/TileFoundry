"""Stratified CANNBench anchor registry, counterexample injection, and scoring."""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import yaml

from tilefoundry_ascend.agent.specs.ops_registry import (
    ATTENTION_RESIDUAL_ANCHORS, DEFAULT_ANCHORS, DEFAULT_PERF_TARGETS, DEFAULT_REGIME_SPECS,
    MLA_ANCHORS, OPERATOR_METADATA, AnchorCaseSpec, CannbenchOpSpec, CannbenchOpsRegistry,
    CasePhysicsRatios, GuardrailCaseSpec, PerfTargetSpec, PhaseSignature, RegimeSpec,
    RegimeTargetSpec, SemanticRegimeGroup, _analyze_case_geometry, _has_flat_operator_layout,
    _is_prime_gt2, _normalize_shape_list, build_semantic_regime_groups,
    build_semantic_regime_specs, campaign_case_regime_tag, default_source_root,
    discover_module_trackb_symbols, extract_schema_driven_case_dims, extract_trackb_cpp_source,
    get_default_anchors, get_default_perf_target, get_default_regime_specs,
    get_operator_entrypoint, get_operator_module_relpath, get_operator_task,
    infer_module_runtime_backend, load_cannbench_contract_module, parse_simple_env_file,
    resolve_cannbench_repo, resolve_cannbench_tasks_root, semantic_regime_representative,
    semantic_regime_tag,
)
from tilefoundry_ascend.ascend.cost_model.experience_lake import load_counterexample_boundaries


def _load_operator_yamls(
    operator_name: str, *, cannbench_repo: str | Path | None = None, env_file: str | Path | None = None
) -> tuple[CannbenchOpsRegistry | None, str, Sequence[Mapping[str, object]], Sequence[Mapping[str, object]], Mapping[int, tuple[float, float]]]:
    tasks_root = resolve_cannbench_tasks_root(resolve_cannbench_repo(cannbench_repo=cannbench_repo, env_file=env_file))
    registry: CannbenchOpsRegistry | None = None
    try:
        registry = CannbenchOpsRegistry(tasks_root=tasks_root)
    except (FileNotFoundError, KeyError, OSError, ValueError):
        registry = None
    if registry is not None and registry.has_operator(operator_name):
        spec = registry.resolve(operator_name)
        ri = spec.proto.get("inputs")
        return registry, spec.task, spec.cases, (ri if isinstance(ri, (list, tuple)) else ()), spec.metadata_950pr

    task = get_operator_task(operator_name)
    cases_path = tasks_root / task / "cases.yaml"
    if not cases_path.is_file():
        raise FileNotFoundError(f"cases.yaml not found at {cases_path}")
    parsed = yaml.safe_load(cases_path.read_text(encoding="utf-8")) or {}
    cases = [r for r in (parsed.get("cases") or ()) if isinstance(r, Mapping)]
    proto_inputs: Sequence[Mapping[str, object]] = ()
    if (proto_path := cases_path.parent / "proto.yaml").is_file():
        op_blk = (yaml.safe_load(proto_path.read_text(encoding="utf-8")) or {}).get("operator") or {}
        if isinstance(op_blk, Mapping) and isinstance(op_blk.get("inputs"), (list, tuple)):
            proto_inputs = op_blk["inputs"]
    meta_map: dict[int, tuple[float, float]] = {}
    if (meta_path := tasks_root / "metadata" / "950pr.json").is_file() and "/" in task:
        try:
            lvl, op_k = task.split("/", 1)
            for ks, e in ((json.loads(meta_path.read_text(encoding="utf-8")).get(lvl) or {}).get(op_k) or {}).items():
                if isinstance(e, Mapping):
                    meta_map[int(ks)] = (float(e.get("baseline_perf_us", 0.0) or 0.0), float(e.get("t_hw_us", 0.0) or 0.0))
        except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
            pass
    return registry, task, cases, proto_inputs, meta_map


def load_anchor_harness_and_cases(
    operator_name: str, anchors: Sequence[AnchorCaseSpec] | None = None, *,
    cannbench_repo: str | Path | None = None, env_file: str | Path | None = None,
    source_root: str | Path | None = None,
) -> tuple[object, list[object]]:
    """Instantiate cannbench_contract.Harness and filter active anchor Case objects."""
    task = get_operator_task(operator_name)
    task_dir = resolve_cannbench_tasks_root(resolve_cannbench_repo(cannbench_repo=cannbench_repo, env_file=env_file)) / task
    if not task_dir.is_dir():
        raise FileNotFoundError(f"CANNBench task directory missing: {task_dir}")
    harness = load_cannbench_contract_module(source_root=source_root).Harness(task, task_dir)
    active = tuple(anchors) if anchors is not None else get_default_anchors(operator_name)
    by_num = {int(c.case_num): c for c in harness.cases}
    selected = []
    for spec in active:
        if int(spec.case_num) not in by_num:
            raise KeyError(f"case_num={spec.case_num} missing in {task_dir / 'cases.yaml'}")
        selected.append(by_num[int(spec.case_num)])
    return harness, selected


def lookup_case_metadata(
    operator_name: str, case_num: int, *, cannbench_repo: str | Path | None = None, env_file: str | Path | None = None
) -> tuple[float, float]:
    """Read (baseline_perf_us, t_hw_us) from <tasks_root>/metadata/950pr.json."""
    meta_path = resolve_cannbench_tasks_root(resolve_cannbench_repo(cannbench_repo=cannbench_repo, env_file=env_file)) / "metadata" / "950pr.json"
    lvl, op_key = get_operator_task(operator_name).split("/", 1)
    entry = json.loads(meta_path.read_text(encoding="utf-8"))[lvl][op_key][str(int(case_num))]
    return float(entry["baseline_perf_us"]), float(entry["t_hw_us"])


def lookup_case_dims(
    operator_name: str, case_num: int, *, cannbench_repo: str | Path | None = None, env_file: str | Path | None = None
) -> dict[str, object]:
    """Extract tensor dimensions and dispatch attributes from <tasks_root>/<task>/cases.yaml."""
    registry, task, cases_list, proto_inputs, _ = _load_operator_yamls(operator_name, cannbench_repo=cannbench_repo, env_file=env_file)
    tid = int(case_num)
    if (entry := next((r for r in cases_list if int(r.get("case_id", -1)) == tid), None)) is None:
        raise KeyError(f"case_num={tid} missing in {task}/cases.yaml")
    dims = extract_schema_driven_case_dims(entry, proto_inputs=proto_inputs)
    if registry is not None and registry.has_operator(operator_name):
        try:
            if (rtag := registry.get_case_regime_tag(operator_name, tid)) is not None:
                dims["regime_tag"] = rtag
        except (FileNotFoundError, KeyError, OSError, ValueError):
            pass
    return dims


def resolve_regime_for_case(
    operator_name: str, case_dims: Mapping[str, object], regime_specs: Sequence[RegimeTargetSpec]
) -> RegimeTargetSpec:
    """Resolve the active dispatch regime matching structural case dimensions."""
    _ = operator_name
    if not regime_specs:
        raise ValueError("regime_specs must not be empty")
    by_tag = {s.regime_tag: s for s in regime_specs}
    if (tag := str(case_dims.get("regime_tag", "")).strip()) in by_tag:
        return by_tag[tag]
    layout = str(case_dims.get("inputLayout", case_dims.get("input_layout", "BSND"))).upper()
    d_raw = case_dims.get("D", case_dims.get("d_dim", 0))
    d_dim = int(d_raw) if isinstance(d_raw, int) and not isinstance(d_raw, bool) else 0
    b_dim, s_dim = int(case_dims.get("B", case_dims.get("b_dim", 1))), int(case_dims.get("S", case_dims.get("s_dim", case_dims.get("S_q", 1))))

    if "bnsd_gather" in by_tag or "split_kv_decode" in by_tag:
        for ok, rtag in ((layout == "BNSD", "bnsd_gather"), (s_dim <= 2, "split_kv_decode"), (True, "prefill_compute")):
            if ok and rtag in by_tag:
                return by_tag[rtag]
    if "unaligned_d" in by_tag or "small_decode" in by_tag:
        for ok, rtag in ((d_dim > 0 and d_dim % 64 != 0, "unaligned_d"), (b_dim * s_dim < 16, "small_decode"), (True, "aligned_prefill")):
            if ok and rtag in by_tag:
                return by_tag[rtag]
    if d_dim > 1 and d_dim % 16 != 0 and "unaligned_tail" in by_tag:
        return by_tag["unaligned_tail"]
    if isinstance(te := case_dims.get("total_elements"), int):
        if int(te) <= 65536 and "small_latency" in by_tag:
            return by_tag["small_latency"]
        if int(te) >= 1048576 and "large_throughput" in by_tag:
            return by_tag["large_throughput"]
    return regime_specs[0]


def append_dynamic_anchor(
    anchors: Sequence[AnchorCaseSpec], *, operator_name: str, case_num: int, regime_tag: str = "tail_regression",
    ref_perf_us: float | None = None, t_hw_us: float | None = None,
    cannbench_repo: str | Path | None = None, env_file: str | Path | None = None,
) -> tuple[AnchorCaseSpec, ...]:
    """Append a non-anchor regression case idempotently to the active anchor list."""
    cid = int(case_num)
    if any(int(a.case_num) == cid for a in anchors):
        return tuple(anchors)
    if ref_perf_us is None or t_hw_us is None:
        m_ref, m_hw = lookup_case_metadata(operator_name, cid, cannbench_repo=cannbench_repo, env_file=env_file)
        ref_perf_us = m_ref if ref_perf_us is None else float(ref_perf_us)
        t_hw_us = m_hw if t_hw_us is None else float(t_hw_us)
    return (*anchors, AnchorCaseSpec(get_operator_task(operator_name), cid, regime_tag, float(ref_perf_us), float(t_hw_us)))


_NON_TAIL_DIM_KEYS = frozenset({
    "total_elements", "rank", "num_inputs", "dtype_bytes", "case_id", "case_num",
    "group_count", "b", "b_dim", "batch", "n_q", "n_kv", "numheads", "numkvheads",
})
_TOPOLOGY_AXIS_KEYS = ("compute_engine", "memory_ring", "iteration_regularity")


def _optional_positive_float(raw: object) -> float | None:
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        return float(raw) if float(raw) > 0.0 else None
    if isinstance(raw, str):
        try:
            return val if (val := float(raw.strip())) > 0.0 else None
        except ValueError:
            return None
    return None


def _coerce_case_int(raw: object) -> int | None:
    if isinstance(raw, bool) or raw is None:
        return None
    if isinstance(raw, (int, float)):
        return int(raw)
    if isinstance(raw, str):
        try:
            return int(raw.strip())
        except ValueError:
            return None
    return None


def _append_anchor_with_fallback(
    anchors: tuple[AnchorCaseSpec, ...], *, operator_name: str, case_num: int, regime_tag: str,
    ref_perf_us: float | None = None, t_hw_us: float | None = None,
    cannbench_repo: str | Path | None = None, env_file: str | Path | None = None,
) -> tuple[AnchorCaseSpec, ...]:
    cid = int(case_num)
    if any(int(a.case_num) == cid for a in anchors):
        return anchors
    for use_meta in (True, False):
        try:
            return append_dynamic_anchor(
                anchors, operator_name=operator_name, case_num=cid, regime_tag=regime_tag,
                ref_perf_us=ref_perf_us if use_meta else float(ref_perf_us or 0.0),
                t_hw_us=t_hw_us if use_meta else float(t_hw_us or 0.0),
                cannbench_repo=cannbench_repo, env_file=env_file,
            )
        except (FileNotFoundError, KeyError, OSError, ValueError, json.JSONDecodeError):
            continue
    return (*anchors, AnchorCaseSpec(anchors[0].task if anchors else operator_name, cid, regime_tag, float(ref_perf_us or 0.0), float(t_hw_us or 0.0)))


def _record_has_unaligned_dim(record: Mapping[str, object]) -> bool:
    if any(bool(record.get(k)) for k in ("has_unaligned_dim", "odd_or_prime_tail", "is_unaligned")):
        return True
    if not isinstance(raw_dims := record.get("case_dims"), Mapping):
        return False
    if (rs := raw_dims.get("input_shape") or raw_dims.get("shapes")) is not None:
        if any(sh and sh[-1] > 1 and (sh[-1] % 2 == 1 or _is_prime_gt2(sh[-1]) or sh[-1] % 32 != 0) for sh in _normalize_shape_list(rs)):
            return True
    return any(
        str(k).strip().lower() not in _NON_TAIL_DIM_KEYS and isinstance(v, int) and not isinstance(v, bool)
        and v > 1 and (v % 2 == 1 or _is_prime_gt2(v) or v % 32 != 0)
        for k, v in raw_dims.items()
    )


def inject_lake_counterexample_anchors(
    initial_anchors: Sequence[AnchorCaseSpec], *, operator_name: str, regime_specs: Sequence[RegimeTargetSpec],
    experience_lake_dir: str | Path | None, cannbench_repo: str | Path | None = None,
    env_file: str | Path | None = None, max_injected: int | None = None,
    topology_axes: Mapping[str, object] | None = None,
) -> list[AnchorCaseSpec]:
    """Inject high-frequency historical L3/L4 counterexamples or tail seeds into anchors."""
    if experience_lake_dir is None or (max_injected is not None and max_injected <= 0) or not str(experience_lake_dir).strip():
        return list(initial_anchors)
    lake_path = Path(str(experience_lake_dir).strip())
    anchors_t, existing = tuple(initial_anchors), {int(a.case_num) for a in initial_anchors}

    if same_op := load_counterexample_boundaries(lake_path, operator=operator_name):
        freq: Counter[int] = Counter()
        rep_map: dict[int, Mapping[str, object]] = {}
        for rec in same_op:
            if (cid := _coerce_case_int(rec.get("case_num"))) is None or cid in existing:
                continue
            freq[cid] += 1
            if (prev := rep_map.get(cid)) is None or (not str(prev.get("regime_tag", "")).strip() and str(rec.get("regime_tag", "")).strip()):
                rep_map[cid] = rec
        ordered = sorted(freq.items(), key=lambda it: (-it[1], it[0]))
        for cid, _cnt in (ordered if max_injected is None else ordered[:max_injected]):
            rep = rep_map[cid]
            rtag = str(rep.get("regime_tag", "")).strip() or (
                resolve_regime_for_case(operator_name, rep["case_dims"], regime_specs).regime_tag
                if isinstance(rep.get("case_dims"), Mapping) and rep["case_dims"] and regime_specs
                else (regime_specs[0].regime_tag if regime_specs else "tail_regression")
            )
            anchors_t = _append_anchor_with_fallback(
                anchors_t, operator_name=operator_name, case_num=cid, regime_tag=rtag,
                ref_perf_us=_optional_positive_float(rep.get("ref_perf_us")),
                t_hw_us=_optional_positive_float(rep.get("t_hw_us")), cannbench_repo=cannbench_repo, env_file=env_file,
            )
        return list(anchors_t)

    if not (all_recs := load_counterexample_boundaries(lake_path)):
        return list(anchors_t)
    t_axes = {k: str(topology_axes.get(k, "")).strip().lower() for k in _TOPOLOGY_AXIS_KEYS if str(topology_axes.get(k, "")).strip()} if isinstance(topology_axes, Mapping) else {}
    filtered = [r for r in all_recs if not t_axes or not isinstance(ra := r.get("topology_axes"), Mapping) or not ra or all(str(ra.get(k, "")).strip().lower() in ("", exp) for k, exp in t_axes.items())]
    if not any(_record_has_unaligned_dim(r) for r in filtered):
        return list(anchors_t)
    try:
        _, _, cases_list, _, meta_map = _load_operator_yamls(operator_name, cannbench_repo=cannbench_repo, env_file=env_file)
    except (FileNotFoundError, KeyError, OSError, ValueError, yaml.YAMLError):
        return list(anchors_t)
    cands = [
        (-(_coerce_case_int(geom.get("unaligned_score")) or 0), _coerce_case_int(geom.get("total_elems")) or 1, cid, raw, ref_u, hw_u)
        for raw in cases_list
        if (cid := _coerce_case_int(raw.get("case_id"))) is not None and cid >= 0 and cid not in existing
        for ref_u, hw_u in (meta_map.get(cid, (0.0, 0.0)),)
        if bool((geom := _analyze_case_geometry(raw, ref_u, hw_u)).get("is_unaligned"))
    ]
    if not cands:
        return list(anchors_t)
    cands.sort(key=lambda it: (it[0], it[1], it[2]))
    _, _, best_cid, best_raw, best_ref, best_hw = cands[0]
    tags = {s.regime_tag for s in regime_specs}
    chosen_tag = "unaligned_tail" if "unaligned_tail" in tags else ("unaligned_d" if "unaligned_d" in tags else (resolve_regime_for_case(operator_name, extract_schema_driven_case_dims(best_raw), regime_specs).regime_tag if regime_specs else "unaligned_tail"))
    return list(_append_anchor_with_fallback(
        anchors_t, operator_name=operator_name, case_num=best_cid, regime_tag=chosen_tag,
        ref_perf_us=(best_ref if best_ref > 0.0 else None), t_hw_us=(best_hw if best_hw > 0.0 else None),
        cannbench_repo=cannbench_repo, env_file=env_file,
    ))


def compute_case_eq3_score(ref_perf_us: float, cand_us: float, t_hw_us: float) -> float:
    """Compute CANNBench Eq.3 per-case performance score in [0, +inf)."""
    if cand_us <= 0.0 or t_hw_us <= 0.0:
        return 0.0
    baseline = float(ref_perf_us) if ref_perf_us > 0.0 else max(t_hw_us * 10.0, 10.0)
    denom = (cand_us - t_hw_us) + (baseline - t_hw_us)
    if denom <= 0.0:
        return 0.0
    return max(0.0, (baseline - t_hw_us) / denom)


def compute_case_aligned_eta(anchor: AnchorCaseSpec, measured_dur_us: float) -> float:
    """Compute Roofline efficiency eta = T_hw / T_measured clamped to [0, 1.0]."""
    if not math.isfinite(measured_dur_us) or measured_dur_us <= 0.0 or anchor.t_hw_us <= 0.0:
        return 0.0
    return max(0.0, min(1.0, float(anchor.t_hw_us) / float(measured_dur_us)))


_ANCHOR_DIM_ALIASES: dict[str, str] = {
    "N": "n_depth", "B": "b_dim", "S": "s_dim", "S_q": "s_dim", "S_kv": "s_kv",
    "D": "d_dim", "N_q": "n_q", "N_kv": "n_kv", "D_nope": "d_nope", "D_rope": "d_rope",
    "S1": "s1_dim", "S2": "s2_dim", "N1": "n_q", "N2": "n_kv", "topK": "top_k", "T": "t_dim",
}


def build_anchor_var_bindings(
    dims: Mapping[str, object] | None, *, kwargs: Mapping[str, object] | None = None
) -> dict[str, int]:
    """Build rule-blind symbolic integer bindings from anchor dims and builder kwargs."""
    out: dict[str, int] = {str(k): int(v) for k, v in (kwargs or {}).items() if isinstance(v, int) and not isinstance(v, bool)}
    explicit: set[str] = set()
    for k, v in (dims or {}).items():
        if isinstance(v, int) and not isinstance(v, bool):
            ks, iv = str(k), int(v)
            out[ks] = iv
            explicit.add(ks)
            if alias := _ANCHOR_DIM_ALIASES.get(ks):
                out[alias] = iv
                explicit.add(alias)
    bv, sv = out.get("B", out.get("b_dim")), out.get("S", out.get("S_q", out.get("s_dim")))
    nqv, nkvv, nv = out.get("N_q", out.get("n_q")), out.get("N_kv", out.get("n_kv")), out.get("N", out.get("n_depth"))
    if bv is not None and sv is not None and "m_rows" not in explicit:
        out["m_rows"] = int(bv) * int(sv)
    if bv is not None and sv is not None and nqv is not None and "q_total_rows" not in explicit:
        out["q_total_rows"] = int(bv) * int(sv) * int(nqv)
    if sv is not None and nqv is not None and nkvv is not None and int(nkvv) > 0 and "q_rows" not in explicit:
        out["q_rows"] = int(sv) * (int(nqv) // int(nkvv))
    if nv is not None and bv is not None and sv is not None and "v_rows" not in explicit:
        out["v_rows"] = int(nv) * int(bv) * int(sv)
    return out


@dataclass(frozen=True)
class InputDomainEnvelope:
    """Operator input domain contract, safe fast-math notes, and constexpr hints."""

    operator: str
    tensor_distributions: tuple[str, ...]
    algebraic_invariants: tuple[str, ...]
    safe_fast_math_notes: tuple[str, ...]
    constexpr_specialization_hints: tuple[str, ...]

    def format_prompt_block(self) -> str:
        """Render domain assumptions and safe fast-math contract for prompt injection."""
        lines = [f"### INPUT DOMAIN & FAST-MATH ENVELOPE ({self.operator})"]
        for hdr, items in (
            ("Tensor Distributions:", self.tensor_distributions),
            ("Algebraic Invariants:", self.algebraic_invariants),
            ("Safe Fast-Math Transformations:", self.safe_fast_math_notes),
            ("Constexpr Builder Specialization:", self.constexpr_specialization_hints),
        ):
            if items:
                lines.append(hdr)
                lines.extend(f"- {it}" for it in items)
        return "\n".join(lines)


INPUT_DOMAIN_ENVELOPES: dict[str, InputDomainEnvelope] = {
    "attention_residual": InputDomainEnvelope(
        "attention_residual",
        ("v_stack and x are fp16/bf16 bounded normal tensors (~randn).", "w is typically zero-initialized or small bounded normal, and gamma is near unit scale."),
        ("RMSNorm(V) normalizes across D with epsilon=1e-6 before depth scoring.", "Normalized dot-product logits in fp32 stay tightly bounded around zero."),
        ("Unshifted exp(score) weights and single-pass vfma accumulation without online max-shift rescaling stay finite and within rtol=1e-2, atol=1e-2 on normalized inputs.", "Keep reductions and exponential weights in fp32 before casting final outputs."),
        ("Return optional 'builder_kwargs': {...} from _select_*_config() to pass static compile-time switches (for example unroll, fission, or fast-exp mode) to builders.",),
    ),
    "mla": InputDomainEnvelope(
        "mla",
        ("q_nope, q_rope, k_nope, k_rope, and v are bounded fp16/bf16 normal tensors (~randn).", "scale = 1 / sqrt(d_nope + d_rope) bounds QK^T magnitudes across heads."),
        ("Causal masking zeroes upper-triangular future KV positions when is_causal=True.", "Group query ratio g_ratio = n_q // n_kv shares K/V heads per head group."),
        ("Retain fp32 online max-shift softmax stabilization across long KV sequence blocks.", "Scale folding and exp2(x * log2(e)) approximations are legal within rtol=1e-2, atol=1e-2."),
        ("Return optional 'builder_kwargs': {...} from _select_*_config() to specialize static head counts, stage splits, or fast-math flags as compile-time constants.",),
    ),
}


def get_input_domain_envelope(operator: str) -> InputDomainEnvelope | None:
    """Return InputDomainEnvelope for *operator*, or None if unregistered."""
    return INPUT_DOMAIN_ENVELOPES.get(str(operator).strip().lower())


__all__ = [
    "ATTENTION_RESIDUAL_ANCHORS", "AnchorCaseSpec", "CannbenchOpSpec", "CannbenchOpsRegistry",
    "DEFAULT_ANCHORS", "DEFAULT_PERF_TARGETS", "DEFAULT_REGIME_SPECS", "GuardrailCaseSpec",
    "INPUT_DOMAIN_ENVELOPES", "InputDomainEnvelope", "MLA_ANCHORS", "OPERATOR_METADATA",
    "PerfTargetSpec", "PhaseSignature", "RegimeSpec", "RegimeTargetSpec", "SemanticRegimeGroup",
    "append_dynamic_anchor", "build_anchor_var_bindings", "build_semantic_regime_groups",
    "build_semantic_regime_specs", "campaign_case_regime_tag",
    "compute_case_aligned_eta", "compute_case_eq3_score", "default_source_root",
    "discover_module_trackb_symbols", "extract_schema_driven_case_dims", "get_default_anchors",
    "get_default_perf_target", "get_default_regime_specs", "get_input_domain_envelope",
    "get_operator_entrypoint", "get_operator_module_relpath", "get_operator_task",
    "infer_module_runtime_backend", "inject_lake_counterexample_anchors",
    "load_anchor_harness_and_cases", "load_cannbench_contract_module", "lookup_case_dims",
    "lookup_case_metadata", "parse_simple_env_file", "resolve_cannbench_repo",
    "resolve_cannbench_tasks_root", "resolve_regime_for_case", "semantic_regime_representative",
    "semantic_regime_tag",
]
