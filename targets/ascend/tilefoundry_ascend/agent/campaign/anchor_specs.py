"""Stratified CANNBench anchor registry and official scoring helpers for campaigns."""

from __future__ import annotations

import importlib.util
import json
import os
import shlex
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class AnchorCaseSpec:
    """Single CANNBench anchor case with calibrated reference and Roofline baselines."""

    task: str
    case_num: int
    regime_tag: str
    ref_perf_us: float
    t_hw_us: float

    def to_dict(self) -> dict[str, object]:
        """Serialize anchor specification to a JSON-compatible dictionary."""
        return {
            "task": self.task,
            "case_num": self.case_num,
            "regime_tag": self.regime_tag,
            "ref_perf_us": self.ref_perf_us,
            "t_hw_us": self.t_hw_us,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> AnchorCaseSpec:
        """Reconstruct AnchorCaseSpec from a JSON mapping."""
        return cls(
            task=str(payload["task"]),
            case_num=int(payload["case_num"]),
            regime_tag=str(payload["regime_tag"]),
            ref_perf_us=float(payload["ref_perf_us"]),
            t_hw_us=float(payload["t_hw_us"]),
        )


@dataclass(frozen=True)
class PerfTargetSpec:
    """Operator-level performance and Roofline convergence targets."""

    operator_name: str
    target_avg_speedup: float
    target_min_case_speedup: float
    target_roofline_eta: float
    per_anchor_max_us: dict[int, float] = field(default_factory=dict)
    target_surrogate_score: float = 90.0
    min_optimization_rounds: int = 3

    def to_dict(self) -> dict[str, object]:
        """Serialize target thresholds to a JSON-compatible dictionary."""
        return {
            "operator_name": self.operator_name,
            "target_avg_speedup": self.target_avg_speedup,
            "target_min_case_speedup": self.target_min_case_speedup,
            "target_roofline_eta": self.target_roofline_eta,
            "target_surrogate_score": self.target_surrogate_score,
            "min_optimization_rounds": self.min_optimization_rounds,
            "per_anchor_max_us": {str(k): v for k, v in sorted(self.per_anchor_max_us.items())},
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> PerfTargetSpec:
        """Reconstruct PerfTargetSpec from a JSON mapping."""
        raw_max = payload.get("per_anchor_max_us") or {}
        per_anchor = (
            {int(k): float(v) for k, v in raw_max.items()}
            if isinstance(raw_max, Mapping)
            else {}
        )
        return cls(
            operator_name=str(payload["operator_name"]),
            target_avg_speedup=float(payload["target_avg_speedup"]),
            target_min_case_speedup=float(payload["target_min_case_speedup"]),
            target_roofline_eta=float(payload["target_roofline_eta"]),
            per_anchor_max_us=per_anchor,
            target_surrogate_score=float(payload.get("target_surrogate_score", 90.0)),
            min_optimization_rounds=int(payload.get("min_optimization_rounds", 3)),
        )


@dataclass(frozen=True)
class RegimeTargetSpec:
    """Target dispatch regime inside an operator kernel module."""

    regime_tag: str
    builder_candidates: tuple[str, ...]
    sample_builder_kwargs: dict[str, object]
    description: str = ""
    representative_hir_dims: dict[str, int] = field(default_factory=dict)
    config_selector_fn: str = ""
    editable_symbols: tuple[str, ...] = ()
    allowed_helper_prefix: str = ""

    def to_dict(self) -> dict[str, object]:
        """Serialize regime target specification to a JSON-compatible dictionary."""
        return {
            "regime_tag": self.regime_tag,
            "builder_candidates": list(self.builder_candidates),
            "sample_builder_kwargs": dict(self.sample_builder_kwargs),
            "description": self.description,
            "representative_hir_dims": {
                str(k): int(v) for k, v in self.representative_hir_dims.items()
            },
            "config_selector_fn": self.config_selector_fn,
            "editable_symbols": list(self.editable_symbols),
            "allowed_helper_prefix": self.allowed_helper_prefix,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> RegimeTargetSpec:
        """Reconstruct RegimeTargetSpec from a JSON mapping."""
        raw_kwargs = payload.get("sample_builder_kwargs") or {}
        raw_dims = payload.get("representative_hir_dims") or {}
        return cls(
            regime_tag=str(payload["regime_tag"]),
            builder_candidates=tuple(str(x) for x in payload.get("builder_candidates", ())),
            sample_builder_kwargs=dict(raw_kwargs) if isinstance(raw_kwargs, Mapping) else {},
            description=str(payload.get("description", "")),
            representative_hir_dims=(
                {str(k): int(v) for k, v in raw_dims.items()}
                if isinstance(raw_dims, Mapping)
                else {}
            ),
            config_selector_fn=str(payload.get("config_selector_fn", "")),
            editable_symbols=tuple(str(x) for x in payload.get("editable_symbols", ())),
            allowed_helper_prefix=str(payload.get("allowed_helper_prefix", "")),
        )


RegimeSpec = RegimeTargetSpec


OPERATOR_METADATA: dict[str, tuple[str, str, str]] = {
    "attention_residual": (
        "level4/attention_residual",
        "examples/ascend950pr_ops/attention_residual/kernels.py",
        "attention_residual",
    ),
    "mla": (
        "level4/mla",
        "examples/ascend950pr_ops/mla/kernels.py",
        "mla",
    ),
}

ATTENTION_RESIDUAL_ANCHORS: tuple[AnchorCaseSpec, ...] = (
    AnchorCaseSpec("level4/attention_residual", 2, "small_decode", 1.5, 0.5),
    AnchorCaseSpec("level4/attention_residual", 4, "unaligned_d", 9.93, 3.3099),
    AnchorCaseSpec("level4/attention_residual", 14, "unaligned_d", 67.033, 22.3442),
    AnchorCaseSpec("level4/attention_residual", 16, "aligned_prefill", 1098.175, 366.0583),
    AnchorCaseSpec("level4/attention_residual", 17, "aligned_prefill", 1037.167, 345.7223),
)

MLA_ANCHORS: tuple[AnchorCaseSpec, ...] = (
    AnchorCaseSpec("level4/mla", 1, "prefill_compute", 69.16, 23.0568),
    AnchorCaseSpec("level4/mla", 6, "prefill_compute", 223.696, 74.5654),
    AnchorCaseSpec("level4/mla", 7, "split_kv_decode", 4.558, 1.5192),
    AnchorCaseSpec("level4/mla", 11, "split_kv_decode", 174.88, 102.9511),
    AnchorCaseSpec("level4/mla", 13, "bnsd_gather", 69.13, 23.0568),
)

DEFAULT_ANCHORS: dict[str, tuple[AnchorCaseSpec, ...]] = {
    "attention_residual": ATTENTION_RESIDUAL_ANCHORS,
    "mla": MLA_ANCHORS,
}

DEFAULT_PERF_TARGETS: dict[str, PerfTargetSpec] = {
    "attention_residual": PerfTargetSpec(
        operator_name="attention_residual",
        target_avg_speedup=2.85,
        target_min_case_speedup=1.60,
        target_roofline_eta=0.90,
        target_surrogate_score=97.5,
        min_optimization_rounds=3,
        per_anchor_max_us={},
    ),
    "mla": PerfTargetSpec(
        operator_name="mla",
        target_avg_speedup=2.50,
        target_min_case_speedup=1.35,
        target_roofline_eta=0.85,
        target_surrogate_score=92.0,
        min_optimization_rounds=3,
        per_anchor_max_us={7: 8.0, 13: 95.0},
    ),
}

DEFAULT_REGIME_SPECS: dict[str, tuple[RegimeTargetSpec, ...]] = {
    "attention_residual": (
        RegimeTargetSpec(
            regime_tag="small_decode",
            builder_candidates=(
                "_build_ar_small_decode_func",
                "_build_attention_residual_prim_func",
            ),
            sample_builder_kwargs={
                "n_depth": 2,
                "d_dim": 4096,
                "tl_dtype": "bfloat16",
                "tile_m": 1,
                "schedule_mode": "tuned",
            },
            description="Small-row decode regime (m_rows < 28) needing depth/width core sharding.",
            representative_hir_dims={"N": 2, "B": 1, "S": 1, "D": 4096},
            config_selector_fn="_select_ar_small_decode_config",
            editable_symbols=(
                "_select_ar_small_decode_config",
                "_build_ar_small_decode_func",
            ),
            allowed_helper_prefix="_ar_small_decode_",
        ),
        RegimeTargetSpec(
            regime_tag="unaligned_d",
            builder_candidates=(
                "_build_ar_unaligned_func",
                "_build_attention_residual_prim_func",
            ),
            sample_builder_kwargs={
                "n_depth": 8,
                "d_dim": 1000,
                "tl_dtype": "bfloat16",
                "tile_m": 1,
                "schedule_mode": "tuned",
            },
            description=(
                "Non-64-aligned D tail regime enabling padded UB row pitch and multi-row "
                "buffering."
            ),
            representative_hir_dims={"N": 8, "B": 1, "S": 4, "D": 1000},
            config_selector_fn="_select_ar_unaligned_config",
            editable_symbols=(
                "_select_ar_unaligned_config",
                "_build_ar_unaligned_func",
            ),
            allowed_helper_prefix="_ar_unaligned_",
        ),
        RegimeTargetSpec(
            regime_tag="aligned_prefill",
            builder_candidates=(
                "_build_ar_aligned_func",
                "_build_attention_residual_prim_func",
            ),
            sample_builder_kwargs={
                "n_depth": 8,
                "d_dim": 4096,
                "tl_dtype": "bfloat16",
                "tile_m": 6,
                "schedule_mode": "tuned",
            },
            description="Aligned prefill regime with 56-core multi-row SimdVF pipelining.",
            representative_hir_dims={"N": 8, "B": 1, "S": 6, "D": 512},
            config_selector_fn="_select_ar_aligned_config",
            editable_symbols=(
                "_select_ar_aligned_config",
                "_build_ar_aligned_func",
            ),
            allowed_helper_prefix="_ar_aligned_",
        ),
    ),
    "mla": (
        RegimeTargetSpec(
            regime_tag="split_kv_decode",
            builder_candidates=(
                "_build_mla_split_kv_decode_func",
                "_build_mla_prim_func",
            ),
            sample_builder_kwargs={
                "b_dim": 1,
                "s_dim": 1,
                "s_kv": 2048,
                "n_q": 128,
                "n_kv": 1,
                "d_nope": 512,
                "d_rope": 64,
                "tl_dtype": "float16",
                "input_layout": "BSND",
                "is_causal": False,
                "bm": 64,
                "aligned_group_q": True,
                "schedule_mode": "tuned",
            },
            description=(
                "Decode and small-Sq regime targeting Split-KV FlashDecoding across 28 MixCores."
            ),
            representative_hir_dims={
                "B": 1,
                "S": 1,
                "S_kv": 64,
                "N_q": 16,
                "N_kv": 1,
                "D_nope": 512,
                "D_rope": 64,
            },
            config_selector_fn="_select_mla_decode_config",
            editable_symbols=(
                "_select_mla_decode_config",
                "_build_mla_split_kv_decode_func",
                "_build_mla_split_kv_combine_func",
            ),
            allowed_helper_prefix="_mla_decode_",
        ),
        RegimeTargetSpec(
            regime_tag="bnsd_gather",
            builder_candidates=(
                "_build_mla_bnsd_gather_func",
                "_build_mla_prim_func",
            ),
            sample_builder_kwargs={
                "n_q": 128,
                "n_kv": 1,
                "d_nope": 512,
                "d_rope": 64,
                "tl_dtype": "float16",
                "input_layout": "BNSD",
                "is_causal": True,
                "bm": 64,
                "aligned_group_q": False,
                "schedule_mode": "tuned",
            },
            description="Non-aligned BNSD Q staging regime replacing scalar row copies.",
            representative_hir_dims={
                "B": 1,
                "S": 2,
                "S_kv": 32,
                "N_q": 16,
                "N_kv": 1,
                "D_nope": 128,
                "D_rope": 64,
            },
            config_selector_fn="_select_mla_bnsd_config",
            editable_symbols=(
                "_select_mla_bnsd_config",
                "_build_mla_bnsd_gather_func",
            ),
            allowed_helper_prefix="_mla_bnsd_",
        ),
        RegimeTargetSpec(
            regime_tag="prefill_compute",
            builder_candidates=(
                "_build_mla_prefill_func",
                "_build_mla_prim_func",
            ),
            sample_builder_kwargs={
                "n_q": 128,
                "n_kv": 1,
                "d_nope": 512,
                "d_rope": 64,
                "tl_dtype": "float16",
                "input_layout": "BSND",
                "is_causal": True,
                "bm": 128,
                "aligned_group_q": True,
                "schedule_mode": "tuned",
            },
            description="Prefill and large-batch compute/pipeline overlap regime.",
            representative_hir_dims={
                "B": 1,
                "S": 8,
                "S_kv": 32,
                "N_q": 8,
                "N_kv": 1,
                "D_nope": 128,
                "D_rope": 64,
            },
            config_selector_fn="_select_mla_prefill_config",
            editable_symbols=(
                "_select_mla_prefill_config",
                "_build_mla_prefill_func",
            ),
            allowed_helper_prefix="_mla_prefill_",
        ),
    ),
}


def get_operator_task(operator_name: str) -> str:
    """Return CANNBench task path (level4/<op>) for an operator name."""
    if operator_name not in OPERATOR_METADATA:
        raise KeyError(f"unsupported campaign operator {operator_name!r}")
    return OPERATOR_METADATA[operator_name][0]


def get_operator_module_relpath(operator_name: str) -> str:
    """Return workspace-relative kernel module path for an operator name."""
    if operator_name not in OPERATOR_METADATA:
        raise KeyError(f"unsupported campaign operator {operator_name!r}")
    return OPERATOR_METADATA[operator_name][1]


def get_operator_entrypoint(operator_name: str) -> str:
    """Return public Python callable name for an operator name."""
    if operator_name not in OPERATOR_METADATA:
        raise KeyError(f"unsupported campaign operator {operator_name!r}")
    return OPERATOR_METADATA[operator_name][2]


def get_default_anchors(operator_name: str) -> tuple[AnchorCaseSpec, ...]:
    """Return stratified anchor cases for attention_residual or mla."""
    if operator_name not in DEFAULT_ANCHORS:
        raise KeyError(f"unsupported campaign operator {operator_name!r}")
    return DEFAULT_ANCHORS[operator_name]


def get_default_perf_target(operator_name: str) -> PerfTargetSpec:
    """Return default performance and Roofline target thresholds for an operator."""
    if operator_name not in DEFAULT_PERF_TARGETS:
        raise KeyError(f"unsupported campaign operator {operator_name!r}")
    return DEFAULT_PERF_TARGETS[operator_name]


def get_default_regime_specs(operator_name: str) -> tuple[RegimeTargetSpec, ...]:
    """Return ordered optimization regime specifications for an operator."""
    if operator_name not in DEFAULT_REGIME_SPECS:
        raise KeyError(f"unsupported campaign operator {operator_name!r}")
    return DEFAULT_REGIME_SPECS[operator_name]


def parse_simple_env_file(env_file: str | Path) -> dict[str, str]:
    """Parse NAME=VALUE assignments from a plain shell config file."""
    path = Path(env_file)
    if not path.is_file():
        return {}
    parsed: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, raw_val = line.split("=", 1)
        key = key.strip()
        raw_val = raw_val.strip()
        if not key:
            continue
        try:
            tokens = shlex.split(raw_val)
            value = tokens[0] if len(tokens) == 1 else raw_val
        except ValueError:
            value = raw_val.strip("\"'")
        parsed[key] = value
    return parsed


def resolve_cannbench_repo(
    cannbench_repo: str | Path | None = None,
    env_file: str | Path | None = None,
) -> Path:
    """Resolve TILEFOUNDRY_NPU_CANNBENCH_REPO from explicit arg, env, or env.sh."""
    candidates: list[str] = []
    if cannbench_repo is not None:
        candidates.append(str(cannbench_repo))
    env_val = os.environ.get("TILEFOUNDRY_NPU_CANNBENCH_REPO", "").strip()
    if env_val:
        candidates.append(env_val)
    resolved_env_file = (
        str(env_file)
        if env_file is not None
        else os.environ.get("TILEFOUNDRY_NPU_ENV_FILE", "").strip()
    )
    if resolved_env_file:
        parsed = parse_simple_env_file(resolved_env_file)
        if parsed.get("TILEFOUNDRY_NPU_CANNBENCH_REPO"):
            candidates.append(parsed["TILEFOUNDRY_NPU_CANNBENCH_REPO"])

    for raw in candidates:
        path = Path(raw).expanduser()
        if path.is_dir() and (path / "operators").is_dir():
            return path.resolve()
    raise FileNotFoundError(
        "Cannot locate TILEFOUNDRY_NPU_CANNBENCH_REPO. "
        "Pass cannbench_repo or set TILEFOUNDRY_NPU_CANNBENCH_REPO."
    )


def default_source_root() -> Path:
    """Resolve TileFoundry repository root relative to this module."""
    return Path(__file__).resolve().parents[5]


def load_cannbench_contract_module(source_root: str | Path | None = None):
    """Dynamically load tests/ascend/benchmarks/cannbench_contract.py."""
    root = Path(source_root).resolve() if source_root is not None else default_source_root()
    contract_path = root / "tests" / "ascend" / "benchmarks" / "cannbench_contract.py"
    if not contract_path.is_file():
        raise FileNotFoundError(f"cannbench_contract.py not found under {contract_path}")
    spec = importlib.util.spec_from_file_location("_tf_cannbench_contract", str(contract_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to create module spec for {contract_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_anchor_harness_and_cases(
    operator_name: str,
    anchors: Sequence[AnchorCaseSpec] | None = None,
    *,
    cannbench_repo: str | Path | None = None,
    env_file: str | Path | None = None,
    source_root: str | Path | None = None,
):
    """Instantiate cannbench_contract.Harness and filter active anchor Case objects."""
    task = get_operator_task(operator_name)
    repo = resolve_cannbench_repo(cannbench_repo=cannbench_repo, env_file=env_file)
    task_dir = repo / "operators" / task
    if not task_dir.is_dir():
        raise FileNotFoundError(f"CANNBench task directory missing: {task_dir}")
    contract_mod = load_cannbench_contract_module(source_root=source_root)
    harness = contract_mod.Harness(task, task_dir)
    active = tuple(anchors) if anchors is not None else get_default_anchors(operator_name)
    by_num = {int(c.case_num): c for c in harness.cases}
    selected = []
    for spec in active:
        if int(spec.case_num) not in by_num:
            raise KeyError(f"case_num={spec.case_num} missing in {task_dir / 'cases.yaml'}")
        selected.append(by_num[int(spec.case_num)])
    return harness, selected


def lookup_case_metadata(
    operator_name: str,
    case_num: int,
    *,
    cannbench_repo: str | Path | None = None,
    env_file: str | Path | None = None,
) -> tuple[float, float]:
    """Read (baseline_perf_us, t_hw_us) from operators/metadata/950pr.json."""
    repo = resolve_cannbench_repo(cannbench_repo=cannbench_repo, env_file=env_file)
    meta_path = repo / "operators" / "metadata" / "950pr.json"
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    task = get_operator_task(operator_name)
    level, op_key = task.split("/", 1)
    entry = payload[level][op_key][str(int(case_num))]
    return float(entry["baseline_perf_us"]), float(entry["t_hw_us"])


def append_dynamic_anchor(
    anchors: Sequence[AnchorCaseSpec],
    *,
    operator_name: str,
    case_num: int,
    regime_tag: str = "tail_regression",
    ref_perf_us: float | None = None,
    t_hw_us: float | None = None,
    cannbench_repo: str | Path | None = None,
    env_file: str | Path | None = None,
) -> tuple[AnchorCaseSpec, ...]:
    """Append a non-anchor regression case idempotently to the active anchor list."""
    cid = int(case_num)
    if any(int(a.case_num) == cid for a in anchors):
        return tuple(anchors)
    if ref_perf_us is None or t_hw_us is None:
        meta_ref, meta_hw = lookup_case_metadata(
            operator_name,
            cid,
            cannbench_repo=cannbench_repo,
            env_file=env_file,
        )
        resolved_ref = meta_ref if ref_perf_us is None else float(ref_perf_us)
        resolved_hw = meta_hw if t_hw_us is None else float(t_hw_us)
    else:
        resolved_ref = float(ref_perf_us)
        resolved_hw = float(t_hw_us)
    new_spec = AnchorCaseSpec(
        task=get_operator_task(operator_name),
        case_num=cid,
        regime_tag=regime_tag,
        ref_perf_us=resolved_ref,
        t_hw_us=resolved_hw,
    )
    return (*anchors, new_spec)


def compute_case_eq3_score(ref_perf_us: float, cand_us: float, t_hw_us: float) -> float:
    """Compute CANNBench Eq.3 per-case performance score in [0, +inf)."""
    if cand_us <= 0.0 or t_hw_us <= 0.0:
        return 0.0
    baseline = float(ref_perf_us) if ref_perf_us > 0.0 else max(t_hw_us * 10.0, 10.0)
    denom = (cand_us - t_hw_us) + (baseline - t_hw_us)
    if denom <= 0.0:
        return 0.0
    return max(0.0, (baseline - t_hw_us) / denom)


__all__ = [
    "ATTENTION_RESIDUAL_ANCHORS",
    "AnchorCaseSpec",
    "DEFAULT_ANCHORS",
    "DEFAULT_PERF_TARGETS",
    "DEFAULT_REGIME_SPECS",
    "MLA_ANCHORS",
    "OPERATOR_METADATA",
    "PerfTargetSpec",
    "RegimeSpec",
    "RegimeTargetSpec",
    "append_dynamic_anchor",
    "compute_case_eq3_score",
    "default_source_root",
    "get_default_anchors",
    "get_default_perf_target",
    "get_default_regime_specs",
    "get_operator_entrypoint",
    "get_operator_module_relpath",
    "get_operator_task",
    "load_anchor_harness_and_cases",
    "load_cannbench_contract_module",
    "lookup_case_metadata",
    "parse_simple_env_file",
    "resolve_cannbench_repo",
]
