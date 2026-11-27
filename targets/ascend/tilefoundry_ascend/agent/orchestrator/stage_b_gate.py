"""Canonical L2 Stage-B Compile Gate with post-selector CompileTaskSpec quotienting."""

from __future__ import annotations

import ast
import contextlib
import fcntl
import functools
import hashlib
import importlib.util
import inspect
import json
import linecache
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import types
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tilefoundry_ascend.agent.contracts import (
    CasePlanExtractor,
    CompileTaskSpec,
    RegimeCompilePlan,
    StageBCompileReceipt,
)
from tilefoundry_ascend.agent.specs.anchor_specs import build_anchor_var_bindings
from tilefoundry_ascend.agent.specs.ops_registry import (
    discover_module_trackb_symbols,
    extract_schema_driven_case_dims,
    extract_trackb_cpp_source,
    infer_module_runtime_backend,
    parse_simple_env_file,
)
from tilefoundry_ascend.ascend.compile_cache.artifact_store import (
    ArtifactStore,
    collect_new_tl_cache_binaries,
    compute_compile_task_key as compute_artifact_task_key,
    extract_reachable_closure_source,
    is_verified_binary_file,
    merge_tl_cache_tree,
    resolve_unified_tl_cache_dir,
    seed_tl_cache_tree,
    snapshot_tl_cache_binaries,
)
from tilefoundry_ascend.ascend.compile_cache.cpp_closure import (
    extract_reachable_trackb_source,
    strip_cpp_comments_and_strings,
)
from tilefoundry_ascend.ascend.cost_model.bisheng_res_parser import (
    BishengResourceSummary,
    parse_bisheng_res_usage,
)
from tilefoundry_ascend.ascend.cost_model.microarch_profile import extract_microarch_profile_from_cce
from tilefoundry_ascend.ascend.json_record import atomic_write_text
from tilefoundry_ascend.ascend.target import PURE_SIMDVF_UBUF_FLAGS

_PREWARM_MANIFEST_FILENAME = "prewarm_manifest_v1.json"
_COMPILE_DIAGNOSTIC_LIMIT = 1500
_HIR_TO_BUILDER_DIM_MAP: dict[str, str] = {
    "N": "n_depth", "B": "b_dim", "S": "s_dim", "S1": "s1_dim", "S2": "s2_dim",
    "S_kv": "s_kv", "D": "d_dim", "N_q": "n_q", "N1": "n_q", "N_kv": "n_kv", "N2": "n_kv",
    "G": "g_ratio", "D_nope": "d_nope", "D_rope": "d_rope", "M": "m_dim", "K": "k_dim",
    "H": "num_heads", "E": "num_experts", "topK": "top_k", "topk": "top_k",
    "sparse_size": "top_k", "T": "t_dim",
}
_PARAM_DIM_FALLBACK_ALIASES: dict[str, tuple[str, ...]] = {
    "m": ("M", "m_dim", "m_rows"), "m_dim": ("M", "m", "m_rows"),
    "n": ("N", "n_dim", "n_depth"), "n_dim": ("N", "n"), "k": ("K", "k_dim"),
    "k_dim": ("K", "k"), "d": ("D", "d_dim", "hidden_dim"), "d_dim": ("D", "d", "hidden_dim"),
    "b": ("B", "b_dim", "batch"), "b_dim": ("B", "b", "batch"),
    "s": ("S", "s_dim", "seq_len"), "s_dim": ("S", "s", "seq_len"),
    "s1_dim": ("S1", "S", "s_dim"), "s2_dim": ("S2", "S_kv", "s_kv"),
    "n_q": ("N_q", "N1", "num_heads"), "n_kv": ("N_kv", "N2"),
    "top_k": ("topK", "topk", "sparse_size", "k_sparse"), "t_dim": ("T", "S", "s_dim"),
}
_CANN_INCLUDE_SUBDIRS: tuple[str, ...] = (
    "include", "x86_64-linux/asc", "x86_64-linux/asc/include", "x86_64-linux/include",
    "aarch64-linux/asc", "aarch64-linux/asc/include", "aarch64-linux/include",
    "runtime/include", "asc", "asc/include", "tools/tikcpp/tikcfw", "compiler/tikcpp/tikcfw",
)
_VLOOPV2_MISS_RE = re.compile(
    r"(?:vloop_?v2\s*(?:miss|failed|fallback|disabled|not\s+applied)|vloop_?v2\s*[:=]\s*(?:false|0|no|miss)|cannot\s+use\s+vloop_?v2)",
    re.IGNORECASE,
)
_COMPILE_ERROR_RE = re.compile(
    r"(?i)(?:\[error\]|\btvm\s*error\b|\bfatal\s*:|\bCheck\s+failed:|(?:^|[\s:(])(?:Assertion|Type|Value|Runtime|Internal|Key|Attribute|Name|Syntax)Error:|(?:^|[\s:\[(])(?:fatal\s+)?error:)"
)
_DEPRECATED_WARN_RE = re.compile(
    r"(?:warning:\s+.*\[-Wdeprecated-declarations\]|note:\s+'[^']+'\s+has been explicitly marked deprecated here|note:\s+expanded from macro 'ASC_DEPRECATED')"
)
_NON_ERROR_DIAG_HEADER_RE = re.compile(r"^(?:note:|warning:|in file included from|/\S+:\d+:\d+:\s*(?:warning|note):)", re.IGNORECASE)
_TVM_BT_RE = re.compile(r"^\s*(?:\[bt\]\s*\(\d+\)|File \"<unknown>\", line 0, in )")


@dataclass(frozen=True)
class StageBResult:
    """Output of Stage B Conda bisheng compilation check."""

    passed: bool
    error_message: str = ""
    elapsed_sec: float = 0.0
    spill_bytes: int = 0
    vloopv2_hit: bool = True
    compiler_summary: BishengResourceSummary | None = None
    microarch_profile: dict[str, object] | None = None
    full_diagnostic: str = ""
    generated_cce: str = ""
    lowered_tir: str = ""
    fault_plane: str = ""
    fault_code: str = ""
    failed_tier: str = ""
    artifact_id: str = ""
    binary_paths: tuple[str, ...] = ()


def _stage_b_fail(
    t0: float,
    error_message: str,
    fault_code: str,
    *,
    fault_plane: str = "STAGE_B_COMPILE",
    spill_bytes: int = 0,
    vloopv2_hit: bool = True,
    summary: BishengResourceSummary | None = None,
    ma_profile: dict[str, object] | None = None,
    full_diag: str = "",
    cce: str = "",
    tir: str = "",
) -> StageBResult:
    return StageBResult(
        passed=False, error_message=error_message, elapsed_sec=time.monotonic() - t0,
        spill_bytes=spill_bytes, vloopv2_hit=vloopv2_hit, compiler_summary=summary,
        microarch_profile=ma_profile, full_diagnostic=full_diag, generated_cce=cce,
        lowered_tir=tir, fault_plane=fault_plane, fault_code=fault_code, failed_tier="stage_b",
    )


def _json_safe(val: object) -> object:
    if isinstance(val, Mapping):
        return {str(k): _json_safe(v) for k, v in sorted(val.items(), key=lambda kv: str(kv[0]))}
    if isinstance(val, (list, tuple)):
        return [_json_safe(i) for i in val]
    if isinstance(val, (set, frozenset)):
        return sorted((_json_safe(i) for i in val), key=lambda x: json.dumps(x, sort_keys=True, separators=(",", ":")))
    return val if isinstance(val, (str, int, float, bool)) or val is None else str(val)


def _freeze_pair_tuple(raw: Mapping[str, Any] | Sequence[tuple[str, Any]] | None) -> tuple[tuple[str, Any], ...]:
    if not raw:
        return ()
    items = raw.items() if isinstance(raw, Mapping) else raw
    return tuple(sorted(((str(k), _json_safe(v)) for k, v in items), key=lambda kv: kv[0]))


def compute_compile_task_key(
    clean_unit_src: str, target: str, builder_name: str,
    static_kwargs: tuple[tuple[str, Any], ...], pass_configs: tuple[tuple[str, Any], ...],
    compile_flags: tuple[str, ...],
) -> str:
    """Compute canonical SHA-256 task key incorporating all 6 specialization inputs."""
    payload = json.dumps(
        {
            "clean_unit_src": str(clean_unit_src),
            "target": str(target),
            "builder_name": str(builder_name),
            "static_kwargs": [[str(k), _json_safe(v)] for k, v in sorted(static_kwargs, key=lambda kv: str(kv[0]))],
            "pass_configs": [[str(k), _json_safe(v)] for k, v in sorted(pass_configs, key=lambda kv: str(kv[0]))],
            "compile_flags": list(dict.fromkeys(str(f).strip() for f in compile_flags if str(f).strip())),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def has_unreserved_vf_stack_overflow(spec: CompileTaskSpec, stack_bytes: int, compiler_diag: str) -> bool:
    """Decide VF stack overflow strictly against this specialization's flags."""
    low = (compiler_diag or "").lower()
    if "exceeded vf stack size" in low or "stack overflow" in low:
        return True
    return int(stack_bytes) > 0 and spec.disables_vf_stack_ubuf


def build_post_selector_regime_plan(
    module_source: str, regime_tag: str, cases_by_id: Mapping[Any, Mapping[str, Any]], extractor: CasePlanExtractor,
    *, env_file: str | Path | None = None, source_root: str | Path | None = None,
) -> RegimeCompilePlan:
    """Evaluate extractor over all regime cases first, then quotient by task_key."""
    tasks_by_case: dict[str, tuple[CompileTaskSpec, ...]] = {}
    unique_specs: dict[str, CompileTaskSpec] = {}
    fast_loaded = (
        _load_plan_module(module_source, env_file=env_file, source_root=source_root)
        if isinstance(extractor, ModuleSelectorPlanExtractor)
        else None
    )
    for case_id in sorted(cases_by_id, key=str):
        cid_str, case_spec = str(case_id), cases_by_id[case_id]
        emitted = (
            extractor._extract_case_tasks_loaded(fast_loaded, case_spec)
            if fast_loaded is not None and isinstance(extractor, ModuleSelectorPlanExtractor)
            else tuple(extractor.extract_case_tasks(module_source, cid_str, dict(case_spec)))
        )
        tasks_by_case[cid_str] = emitted
        for spec in emitted:
            if spec.task_key not in unique_specs:
                unique_specs[spec.task_key] = spec
    return RegimeCompilePlan(regime_tag=str(regime_tag), tasks_by_case_id=tasks_by_case, unique_specs_by_key=unique_specs)


def _normalize_flag_list(raw_flags: object) -> list[str]:
    if raw_flags is None:
        return []
    if isinstance(raw_flags, str):
        return raw_flags.strip().split() if raw_flags.strip() else []
    if isinstance(raw_flags, (tuple, list)) or (hasattr(raw_flags, "__iter__") and not isinstance(raw_flags, (bytes, bytearray, Mapping))):
        return list(dict.fromkeys(v for item in raw_flags if (v := str(getattr(item, "value", item)).strip())))
    return [s] if (s := str(raw_flags).strip()) else []


def resolve_trackb_compile_flags(cpp_source: str, selector_flags: Sequence[str] | str | None = None) -> tuple[str, ...]:
    """Resolve Track-B bisheng flags, stripping SIMD-VF UBUF flags on __simt_vf__."""
    normalized = _normalize_flag_list(selector_flags)
    if re.search(r"\b__simt_vf__\b", strip_cpp_comments_and_strings(cpp_source)):
        simdvf = frozenset(PURE_SIMDVF_UBUF_FLAGS)
        return tuple(f for f in normalized if f not in simdvf)
    return tuple(dict.fromkeys([*PURE_SIMDVF_UBUF_FLAGS, *normalized]))


def _normalize_trackb_unit_source(cpp_source: str, *, regime_tag: str = "") -> str:
    stripped = cpp_source.strip()
    if regime_tag and stripped:
        masked = strip_cpp_comments_and_strings(stripped)
        if len(re.findall(r"\b__global__\b", masked)) > 1 and "<<<" not in masked:
            _, extracted = extract_reachable_trackb_source(stripped, regime_tag=regime_tag)
            return extracted.strip()
    return stripped


@functools.lru_cache(maxsize=512)
def _cached_sig_info(fn: object) -> tuple[inspect.Signature, bool, tuple[str, ...], tuple[tuple[str, Any], ...]] | None:
    try:
        sig = inspect.signature(fn)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    has_vk = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    pos = tuple(p.name for p in sig.parameters.values() if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD))
    defs = tuple((p.name, p.default) for p in sig.parameters.values() if p.default is not inspect.Parameter.empty and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD))
    return sig, has_vk, pos, defs


def _filter_kwargs_for_callable(fn: object, kwargs: Mapping[str, object]) -> dict[str, object]:
    info = _cached_sig_info(fn)
    if info is None or info[1]:
        return dict(kwargs)
    sig = info[0]
    filtered = {k: v for k, v in kwargs.items() if k in sig.parameters}
    low_kw = {str(k).lower(): v for k, v in kwargs.items()}
    for pname in sig.parameters:
        if pname in filtered:
            continue
        pl = pname.lower()
        if pl in low_kw:
            filtered[pname] = low_kw[pl]
            continue
        for alias in _PARAM_DIM_FALLBACK_ALIASES.get(pl, ()):
            if alias in kwargs or alias.lower() in low_kw:
                filtered[pname] = kwargs[alias] if alias in kwargs else low_kw[alias.lower()]
                break
    return filtered


def _merge_dims_into_kwargs(base_kwargs: Mapping[str, object] | None, dims: Mapping[str, object] | None) -> dict[str, object]:
    merged = dict(base_kwargs or {})
    for key, val in (dims or {}).items():
        ks = str(key)
        merged.setdefault(ks, val)
        merged[_HIR_TO_BUILDER_DIM_MAP.get(ks, ks)] = val
    return merged


def _split_selector_output(sel_out: Mapping[str, object]) -> tuple[dict[str, object], dict[str, object], list[str]]:
    pc, cf, overrides = {}, [], {}
    if isinstance(raw_pc := sel_out.get("pass_configs"), Mapping):
        for k, v in raw_pc.items():
            if str(k) == "tl.device_compile_flags":
                cf.extend(f for f in _normalize_flag_list(v) if f not in cf)
            else:
                pc[str(k)] = v
    elif raw_pc is not None:
        cf.extend(f for f in _normalize_flag_list(raw_pc) if f not in cf)
    cf.extend(f for f in _normalize_flag_list(sel_out.get("compile_flags")) if f not in cf)
    for k, v in sel_out.items():
        if k not in ("pass_configs", "compile_flags", "builder_kwargs"):
            overrides[str(k)] = v
    if isinstance(raw_bk := sel_out.get("builder_kwargs"), Mapping):
        overrides.update({str(k): v for k, v in raw_bk.items()})
    return overrides, pc, cf


def _merge_pass_configs_and_flags(
    sel_pc: Mapping[str, object] | None, sel_cf: Sequence[str] | str | None,
    ext_pc: Mapping[str, object] | None, ext_cf: Sequence[str] | str | None,
) -> tuple[dict[str, object], list[str]]:
    merged_pc: dict[str, object] = {}
    merged_cf: list[str] = []
    for pc_map, cf_seq in ((sel_pc, sel_cf), (ext_pc, ext_cf)):
        for k, v in (pc_map or {}).items():
            if str(k) == "tl.device_compile_flags":
                merged_cf.extend(f for f in _normalize_flag_list(v) if f not in merged_cf)
            else:
                merged_pc[str(k)] = v
        merged_cf.extend(f for f in _normalize_flag_list(cf_seq) if f not in merged_cf)
    return merged_pc, merged_cf


def _bind_callable_static_kwargs(fn: Callable[..., Any], args: tuple[Any, ...], kwargs: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    info = _cached_sig_info(fn)
    if info is None:
        return _freeze_pair_tuple({k: v for k, v in kwargs.items() if not callable(v)})
    _, _, pos_names, defaults = info
    items: dict[str, Any] = dict(defaults)
    for idx, val in enumerate(args):
        if idx < len(pos_names):
            items[pos_names[idx]] = val
    items.update(kwargs)
    return tuple(sorted(((str(k), _json_safe(v)) for k, v in items.items() if not callable(v)), key=lambda kv: kv[0]))


class _MetaDevice:
    type = "npu"
    index = 0

    def __str__(self) -> str:
        return "npu:0"


class _MetaDtype:
    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return f"torch.{self.name}"

    def __eq__(self, other: object) -> bool:
        return self.name == other.name if isinstance(other, _MetaDtype) else str(other) in (self.name, f"torch.{self.name}")

    def __hash__(self) -> int:
        return hash(self.name)


_META_DTYPE_CACHE: dict[str, object] = {}


def _resolve_torch_dtype(name: str) -> object:
    clean = str(name or "bfloat16").strip().removeprefix("torch.") or "bfloat16"
    if clean in _META_DTYPE_CACHE:
        return _META_DTYPE_CACHE[clean]
    try:
        import torch  # noqa: PLC0415

        if (dt := getattr(torch, clean, None)) is not None:
            _META_DTYPE_CACHE[clean] = dt
            return dt
    except Exception:
        pass
    dt_obj = _MetaDtype(clean)
    _META_DTYPE_CACHE[clean] = dt_obj
    return dt_obj


class _MetaTensor:
    device = _MetaDevice()
    is_npu = True

    def __init__(self, shape: Sequence[int], dtype_obj: object) -> None:
        self.shape, self.dtype = tuple(int(x) for x in shape), dtype_obj

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def dim(self) -> int:
        return len(self.shape)

    def numel(self) -> int:
        return math.prod(self.shape) if self.shape else 1

    def size(self, axis: int | None = None) -> tuple[int, ...] | int:
        return self.shape if axis is None else self.shape[axis]

    def view(self, *shape_args: Any) -> _MetaTensor:
        raw = list(shape_args[0]) if len(shape_args) == 1 and isinstance(shape_args[0], (tuple, list)) else list(shape_args)
        neg_idx, known, out = -1, 1, []
        for idx, dim in enumerate(raw):
            iv = int(dim)
            if iv == -1:
                neg_idx = idx
                out.append(1)
            else:
                known *= max(1, iv)
                out.append(iv)
        if neg_idx >= 0:
            out[neg_idx] = max(1, self.numel() // max(1, known))
        return _MetaTensor(out, self.dtype)

    reshape = view
    contiguous = to = __getitem__ = lambda self, *_a, **_k: self
    float = lambda self: _MetaTensor(self.shape, _resolve_torch_dtype("float32"))
    half = lambda self: _MetaTensor(self.shape, _resolve_torch_dtype("float16"))
    bfloat16 = lambda self: _MetaTensor(self.shape, _resolve_torch_dtype("bfloat16"))
    data_ptr = lambda self: 0


@dataclass(frozen=True)
class _CapturedPrimFunc:
    builder_name: str
    static_kwargs: tuple[tuple[str, Any], ...]


class _NoopCacheDict(dict[Any, Any]):
    get = lambda self, _k, default=None: default
    __setitem__ = lambda self, _k, _v: None


class _ModuleProxy:
    def __init__(self, target: Any, overrides: Mapping[str, Any]) -> None:
        self._target, self._overrides = target, overrides

    def __getattr__(self, name: str) -> Any:
        return self._overrides[name] if name in self._overrides else getattr(self._target, name)


class _DummyCLib:
    def __getattr__(self, name: str) -> Any:
        fn = lambda *_a, **_k: 0
        setattr(self, name, fn)
        return fn


@dataclass
class _LoadedPlanModule:
    module_source: str
    mod_sha: str
    mod: types.ModuleType
    records: list[tuple[str, tuple[tuple[str, Any], ...], dict[str, Any], tuple[str, ...]]]
    trackb_records: list[tuple[str, str, tuple[str, ...]]]
    entrypoint_name: str
    selectors: tuple[str, ...]
    builders: tuple[str, ...]
    backend_kind: str
    closure_cache: dict[tuple[str, ...], str]
    spec_cache: dict[tuple[Any, ...], CompileTaskSpec]
    trackb_tu_by_key: dict[str, str]


_PLAN_MODULE_CACHE: dict[str, _LoadedPlanModule] = {}
_PLAN_MODULE_LOCK = threading.Lock()
_LAST_PLAN_ENTRY: tuple[str, _LoadedPlanModule] | None = None


def _resolve_tilelang_source_dir(
    *, env_vals: Mapping[str, str] | None = None, env_file: str | Path | None = None, source_root: str | Path | None = None,
) -> str | None:
    vals = env_vals if env_vals is not None else _resolve_env_values(env_file)
    cands: list[Path] = []
    for raw in (
        vals.get("TILEFOUNDRY_NPU_TILELANG_SOURCE", ""), vals.get("TILEFOUNDRY_NPU_TILELANG_REPO", ""),
        os.environ.get("TILEFOUNDRY_NPU_TILELANG_SOURCE", ""), os.environ.get("TILEFOUNDRY_NPU_TILELANG_REPO", ""),
    ):
        if txt := str(raw or "").strip():
            cands.append(Path(txt).expanduser())
    if source_root is not None:
        cands.append(Path(source_root).expanduser().resolve().parent / "tilelang")
    cands.append(Path(__file__).resolve().parents[5].parent / "tilelang")
    for cand in cands:
        if (cand / "tilelang").is_dir():
            return str(cand.resolve())
    return None


def _ensure_tilelang_on_sys_path(
    *, env_vals: Mapping[str, str] | None = None, env_file: str | Path | None = None, source_root: str | Path | None = None,
) -> str | None:
    """Ensure host CPU autoload guard and resolved tilelang source root are on sys.path."""
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")
    os.environ.setdefault("USE_CUDA", "0")
    if (tl_dir := _resolve_tilelang_source_dir(env_vals=env_vals, env_file=env_file, source_root=source_root)) is None:
        return None
    if tl_dir not in sys.path:
        if importlib.util.find_spec("tilelang") is None:
            sys.path.insert(0, tl_dir)
        else:
            sys.path.append(tl_dir)
        importlib.invalidate_caches()
    return tl_dir


def _load_plan_module(
    module_source: str, *, env_vals: Mapping[str, str] | None = None, env_file: str | Path | None = None, source_root: str | Path | None = None,
) -> _LoadedPlanModule:
    global _LAST_PLAN_ENTRY
    with _PLAN_MODULE_LOCK:
        last = _LAST_PLAN_ENTRY
        if last is not None and (last[0] is module_source or last[0] == module_source):
            return last[1]
        mod_sha = hashlib.sha256(module_source.encode("utf-8")).hexdigest()
        if (cached := _PLAN_MODULE_CACHE.get(mod_sha)) is not None:
            _LAST_PLAN_ENTRY = (module_source, cached)
            return cached

    _ensure_tilelang_on_sys_path(env_vals=env_vals, env_file=env_file, source_root=source_root)
    mod_filename = f"<tf_plan_{mod_sha[:12]}>"
    linecache.cache[mod_filename] = (len(module_source), None, module_source.splitlines(keepends=True), mod_filename)
    mod = types.ModuleType(f"_tf_plan_{mod_sha[:12]}")
    exec(compile(module_source, mod_filename, "exec"), mod.__dict__)

    ep, funcs, prim_builders, kernel_alias = "", [], [], ""
    with contextlib.suppress(SyntaxError):
        for node in ast.parse(module_source).body:
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name) and tgt.id == "kernel" and isinstance(node.value, ast.Name):
                        kernel_alias = node.value.id
            elif isinstance(node, ast.FunctionDef):
                funcs.append(node.name)
                if "@T.prim_func" in (ast.get_source_segment(module_source, node) or ""):
                    prim_builders.append(node.name)
    ep = kernel_alias if kernel_alias in funcs else next(
        (c for c in funcs if not c.startswith(("_", "build_")) and callable(getattr(mod, c, None))),
        "kernel" if "kernel" in funcs else "",
    )
    selectors = tuple(f for f in funcs if f.startswith(("_select_", "select_")) and f.endswith("_config")) or tuple(
        f for f in funcs if f.startswith(("_select_", "select_"))
    )
    builders = tuple(dict.fromkeys([
        *(f for f in funcs if (f.startswith(("_build_", "build_")) or f == "make_kernel") and not f.endswith("_trackb_callable")),
        *prim_builders,
    ]))
    backend, _ = infer_module_runtime_backend(module_source, ep)
    records: list[tuple[str, tuple[tuple[str, Any], ...], dict[str, Any], tuple[str, ...]]] = []
    tb_records: list[tuple[str, str, tuple[str, ...]]] = []

    for bname in builders:
        if not callable(orig_b := getattr(mod, bname, None)):
            continue
        def _mk_spy(name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
            @functools.wraps(fn)
            def _spy(*a: Any, **kw: Any) -> _CapturedPrimFunc:
                return _CapturedPrimFunc(name, _bind_callable_static_kwargs(fn, a, kw))
            _spy.__name__ = name
            return _spy
        setattr(mod, bname, _mk_spy(bname, orig_b))

    if isinstance(getattr(mod, "_KERNEL_CACHE", None), dict):
        setattr(mod, "_KERNEL_CACHE", _NoopCacheDict())

    def _cap_compile(prog: Any, *, target: str = "ascend", out_idx: Any = None, pass_configs: Mapping[str, Any] | None = None, compile_flags: Sequence[str] | str | None = None, **_kw: Any) -> Any:
        _ = target
        pc = dict(pass_configs) if isinstance(pass_configs, Mapping) else {}
        cf = tuple(f for f in _normalize_flag_list(compile_flags) if not f.startswith("-I"))
        records.append((prog.builder_name, prog.static_kwargs, pc, cf) if isinstance(prog, _CapturedPrimFunc) else ("unknown_builder", (), pc, cf))
        n_out = len(out_idx) if isinstance(out_idx, (list, tuple)) and out_idx else 1
        return lambda *a, **_k: (t := (a[0] if a and isinstance(a[0], _MetaTensor) else _MetaTensor((1,), _resolve_torch_dtype("float32")))) if n_out == 1 else tuple(t for _ in range(n_out))

    if (orig_tl := getattr(mod, "tilelang", None)) is not None:
        setattr(mod, "tilelang", _ModuleProxy(orig_tl, {"compile": _cap_compile, "lower": _cap_compile}))
    if hasattr(mod, "compile_ascend"):
        setattr(mod, "compile_ascend", lambda src, *, options=None, **_k: tb_records.append(("_render_trackb_tu", str(src), tuple(f for f in _normalize_flag_list(options) if not f.startswith("-I")))) or "stub.so")
    if (orig_ct := getattr(mod, "ctypes", None)) is not None:
        dlib = _DummyCLib()
        setattr(mod, "ctypes", _ModuleProxy(orig_ct, {"CDLL": lambda *_a, **_k: dlib}))
    if hasattr(mod, "_get_split_kv_workspace"):
        dt = _resolve_torch_dtype("float32")
        setattr(mod, "_get_split_kv_workspace", lambda _d, sp, qr, dn: (_MetaTensor((sp, qr, dn), dt), _MetaTensor((sp, qr), dt)))
    if hasattr(mod, "_get_track_b_lib"):
        tblib = _DummyCLib()
        setattr(mod, "_get_track_b_lib", lambda *_a, **_k: tblib)
    if (orig_th := getattr(mod, "torch", None)) is not None:
        dt32 = _resolve_torch_dtype("float32")
        setattr(mod, "torch", _ModuleProxy(orig_th, {
            "empty_like": lambda t, **_k: _MetaTensor(getattr(t, "shape", (1,)), getattr(t, "dtype", dt32)),
            "empty": lambda *s, **_k: _MetaTensor(s[0] if len(s) == 1 and isinstance(s[0], (tuple, list)) else s, dt32),
            "npu": types.SimpleNamespace(current_stream=lambda *_a, **_k: types.SimpleNamespace(npu_stream=0)),
        }))

    loaded = _LoadedPlanModule(module_source, mod_sha, mod, records, tb_records, ep, selectors, builders, backend, {}, {}, {})
    with _PLAN_MODULE_LOCK:
        if len(_PLAN_MODULE_CACHE) >= 64:
            _PLAN_MODULE_CACHE.clear()
        _PLAN_MODULE_CACHE[mod_sha] = loaded
        _LAST_PLAN_ENTRY = (module_source, loaded)
    return loaded


def _extract_dim_pool(
    case_spec: Mapping[str, Any], *, proto_inputs: Sequence[Mapping[str, Any]] = (),
    default_builder_kwargs: Mapping[str, Any] | None = None, schedule_mode: str = "tuned",
) -> dict[str, Any]:
    case_pool: dict[str, Any] = {}
    if "input_shape" in case_spec:
        case_pool.update({str(k): v for k, v in extract_schema_driven_case_dims(case_spec, proto_inputs=proto_inputs).items()})
        if isinstance(attrs := case_spec.get("attrs"), Mapping):
            case_pool.update({str(k): v for k, v in attrs.items()})
        case_pool.update({str(k): v for k, v in case_spec.items() if k not in ("input_shape", "dtype", "attrs", "value_range")})
        if "dtype" in case_spec and "tl_dtype" not in case_pool:
            rdt = case_spec["dtype"]
            fdt = rdt[0] if isinstance(rdt, (list, tuple)) and rdt else rdt
            case_pool["tl_dtype"] = str(fdt or "bfloat16").strip().removeprefix("torch.") or "bfloat16"
        for k, v in list(case_pool.items()):
            if mapped := _HIR_TO_BUILDER_DIM_MAP.get(k):
                case_pool.setdefault(mapped, v)
        for k, v in build_anchor_var_bindings(case_pool).items():
            case_pool.setdefault(k, v)
    else:
        for k, v in case_spec.items():
            if k != "value_range":
                case_pool[ks := str(k)] = v
                if mapped := _HIR_TO_BUILDER_DIM_MAP.get(ks):
                    case_pool.setdefault(mapped, v)
        if (bv := case_pool.get("b_dim", case_pool.get("B"))) is not None and (sv := case_pool.get("s_dim", case_pool.get("S"))) is not None:
            case_pool.setdefault("m_rows", int(bv) * int(sv))
    pool: dict[str, Any] = dict(default_builder_kwargs) if default_builder_kwargs else {}
    pool.update(case_pool)
    pool.setdefault("schedule_mode", schedule_mode)
    rtl = pool.get("tl_dtype", "bfloat16")
    pool["tl_dtype"] = (rtl[6:] or "bfloat16") if isinstance(rtl, str) and rtl.startswith("torch.") else (rtl or "bfloat16")
    return pool


@functools.lru_cache(maxsize=256)
def _cached_ep_meta(ep_fn: Callable[..., Any]) -> tuple[tuple[tuple[str, bool], ...], int, tuple[str, ...]] | None:
    if (info := _cached_sig_info(ep_fn)) is None:
        return None
    req: list[tuple[str, bool]] = []
    opt: list[str] = []
    n_t = 0
    for p in info[0].parameters.values():
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if p.default is inspect.Parameter.empty:
            is_s = any(t in str(p.annotation or "") for t in ("float", "int", "bool", "str")) or p.name in ("alpha", "beta", "epsilon", "eps", "scale", "scaleValue")
            req.append((p.name, not is_s))
            n_t += int(not is_s)
        else:
            opt.append(p.name)
    return tuple(req), n_t, tuple(opt)


def _build_entrypoint_meta_args(ep_fn: Callable[..., Any], case_spec: Mapping[str, Any], pool: Mapping[str, Any]) -> tuple[list[Any], dict[str, Any]] | None:
    if (meta := _cached_ep_meta(ep_fn)) is None:
        return None
    req_params, n_tensors, opt_names = meta
    shapes: list[list[int]] = []
    raw_in = case_spec.get("input_shape")
    if isinstance(raw_in, (list, tuple)) and raw_in:
        shapes = [[int(d) for d in raw_in]] if isinstance(raw_in[0], int) else [
            [int(d) for d in sh] for sh in raw_in if isinstance(sh, (list, tuple)) and sh and isinstance(sh[0], int)
        ]
    if len(shapes) < n_tensors and "n_depth" in pool and "d_dim" in pool and n_tensors >= 4:
        b, s, n, d = int(pool.get("b_dim", 1)), int(pool.get("s_dim", pool.get("m_rows", 1))), int(pool["n_depth"]), int(pool["d_dim"])
        shapes = [[n, b, s, d], [b, s, d], [d], [d]] if n_tensors == 4 else []
    elif len(shapes) < n_tensors and "d_nope" in pool and "d_rope" in pool and n_tensors in (4, 5):
        b, s, skv, nq, nkv = int(pool.get("b_dim", 1)), int(pool.get("s_dim", 1)), int(pool.get("s_kv", 128)), int(pool.get("n_q", 16)), int(pool.get("n_kv", 1))
        dn, dr = int(pool["d_nope"]), int(pool["d_rope"])
        if str(pool.get("input_layout", pool.get("inputLayout", "BSND"))).upper() == "BNSD":
            shapes = [[b, nq, s, dn], [b, nq, s, dr], [b, nkv, skv, dn], [b, nkv, skv, dr], [b, nkv, skv, dn]]
        else:
            shapes = [[b, s, nq, dn], [b, s, nq, dr], [b, skv, nkv, dn], [b, skv, nkv, dr], [b, skv, nkv, dn]]
    if len(shapes) < n_tensors:
        return None
    rdt = case_spec.get("dtype")
    dt_list = [str(d).strip() for d in rdt if str(d).strip()] if isinstance(rdt, (list, tuple)) else ([rdt.strip()] if isinstance(rdt, str) and rdt.strip() else [])
    def_dt = str(pool.get("tl_dtype", "bfloat16"))
    pos_args: list[Any] = []
    t_idx = 0
    for pname, is_t in req_params:
        if is_t:
            dt_s = dt_list[t_idx] if t_idx < len(dt_list) and dt_list[t_idx].lower() != "none" else def_dt
            pos_args.append(_MetaTensor(shapes[t_idx], _resolve_torch_dtype(dt_s)))
            t_idx += 1
        else:
            pos_args.append(float(pool.get(pname, 1.0)))
    kw_args = {on: pool[on] for on in opt_names if on in pool}
    if "numKVHeads" in opt_names and "numKVHeads" not in kw_args and "n_kv" in pool:
        kw_args["numKVHeads"] = pool["n_kv"]
    if "inputLayout" in opt_names and "inputLayout" not in kw_args and "input_layout" in pool:
        kw_args["inputLayout"] = pool["input_layout"]
    return pos_args, kw_args


_TB_POOL_KEYS = frozenset({"bm", "d_nope", "d_dim", "tl_dtype", "m_rows", "inner_dim", "trackb_bm"})


class ModuleSelectorPlanExtractor:
    """Evaluate candidate module selectors on host CPU to emit exact CompileTaskSpecs."""

    def __init__(
        self, *, target: str = "dav-3510", builder_name: str = "", config_selector_name: str = "",
        default_builder_kwargs: Mapping[str, Any] | None = None, default_pass_configs: Mapping[str, Any] | None = None,
        default_compile_flags: Sequence[str] | str | None = None, proto_inputs: Sequence[Mapping[str, Any]] = (),
        schedule_mode: str = "tuned",
    ) -> None:
        self.target, self.builder_name, self.config_selector_name = str(target), str(builder_name), str(config_selector_name)
        self.default_builder_kwargs = dict(default_builder_kwargs or {})
        self.default_pass_configs = dict(default_pass_configs or {})
        self.default_compile_flags = tuple(_normalize_flag_list(default_compile_flags))
        self._proto_inputs = tuple(proto_inputs)
        self.schedule_mode = str(schedule_mode)

    def _spec_from_capture(
        self, loaded: _LoadedPlanModule, builder_name: str, static_kwargs: tuple[tuple[str, Any], ...],
        pass_configs: Mapping[str, Any], compile_flags: tuple[str, ...], *,
        selector_name: str = "", clean_unit_override: str | None = None,
    ) -> CompileTaskSpec:
        merged_pc, merged_cf = _merge_pass_configs_and_flags(pass_configs, compile_flags, self.default_pass_configs, self.default_compile_flags)
        pc_t, cf_t = _freeze_pair_tuple(merged_pc), tuple(merged_cf)
        src_sel = selector_name if clean_unit_override is None else clean_unit_override
        ckey = json.dumps([self.target, builder_name, static_kwargs, pc_t, cf_t, src_sel], sort_keys=True)
        if clean_unit_override is None and ckey in loaded.spec_cache:
            return loaded.spec_cache[ckey]
        if clean_unit_override is not None:
            clean_src = clean_unit_override
        else:
            sym_key = tuple(s for s in (selector_name, builder_name) if s)
            clean_src = loaded.closure_cache.get(sym_key)
            if clean_src is None:
                clean_src = extract_reachable_closure_source(loaded.module_source, active_symbols=sym_key) if sym_key else loaded.module_source
                loaded.closure_cache[sym_key] = clean_src
        spec = CompileTaskSpec(compute_compile_task_key(clean_src, self.target, builder_name, static_kwargs, pc_t, cf_t), builder_name, static_kwargs, pc_t, cf_t)
        if clean_unit_override is not None:
            loaded.trackb_tu_by_key[spec.task_key] = clean_unit_override
        else:
            loaded.spec_cache[ckey] = spec
        return spec

    def _collect_captured_specs(self, loaded: _LoadedPlanModule, pool: Mapping[str, Any], sel_name: str = "") -> list[CompileTaskSpec]:
        specs = [self._spec_from_capture(loaded, bn, kw, pc, cf, selector_name=sel_name) for bn, kw, pc, cf in loaded.records]
        for tb_name, cpp_src, tb_cf in loaded.trackb_records:
            norm_tu = _normalize_trackb_unit_source(cpp_src)
            eff_cf = resolve_trackb_compile_flags(norm_tu, tb_cf)
            tb_kw = _freeze_pair_tuple({k: v for k, v in pool.items() if k in _TB_POOL_KEYS})
            specs.append(self._spec_from_capture(loaded, tb_name, tb_kw, {}, eff_cf, selector_name=sel_name, clean_unit_override=norm_tu))
        return specs

    def _extract_case_tasks_loaded(self, loaded: _LoadedPlanModule, case_spec: Mapping[str, Any]) -> tuple[CompileTaskSpec, ...]:
        pool = _extract_dim_pool(case_spec, proto_inputs=self._proto_inputs, default_builder_kwargs=self.default_builder_kwargs, schedule_mode=self.schedule_mode)
        if loaded.entrypoint_name and callable(ep_fn := getattr(loaded.mod, loaded.entrypoint_name, None)):
            if (meta_args := _build_entrypoint_meta_args(ep_fn, case_spec, pool)) is not None:
                loaded.records.clear()
                loaded.trackb_records.clear()
                try:
                    ep_fn(*meta_args[0], **meta_args[1])
                    if ep_specs := self._collect_captured_specs(loaded, pool):
                        return tuple(ep_specs)
                except Exception:
                    pass

        sel_name = str(case_spec.get("config_selector_name") or self.config_selector_name or (loaded.selectors[0] if loaded.selectors else ""))
        overrides: dict[str, Any] = {}
        sel_pc: dict[str, Any] = {}
        sel_cf: list[str] = []
        raw_bk: dict[str, Any] = {}
        raw_ck: dict[str, Any] = {}
        if sel_name and callable(sel_fn := getattr(loaded.mod, sel_name, None)):
            if isinstance(sel_out := sel_fn(**_filter_kwargs_for_callable(sel_fn, pool)), Mapping):
                overrides, sel_pc, sel_cf = _split_selector_output(sel_out)
                if isinstance(sel_out.get("builder_kwargs"), Mapping):
                    raw_bk = dict(sel_out["builder_kwargs"])
                if isinstance(sel_out.get("combine_kwargs"), Mapping):
                    raw_ck = dict(sel_out["combine_kwargs"])
                pool.update(overrides)

        render_fn = getattr(loaded.mod, "_render_trackb_tu", None)
        if (bool(pool.get("use_trackb")) or (loaded.backend_kind == "ascendc_trackb" and callable(render_fn))) and callable(render_fn):
            norm_tu = _normalize_trackb_unit_source(str(render_fn(**_filter_kwargs_for_callable(render_fn, pool))))
            tb_kw = _freeze_pair_tuple({k: v for k, v in pool.items() if k in _TB_POOL_KEYS})
            return (self._spec_from_capture(loaded, "_render_trackb_tu", tb_kw, sel_pc, resolve_trackb_compile_flags(norm_tu, sel_cf), selector_name=sel_name, clean_unit_override=norm_tu),)

        b_name = str(case_spec.get("builder_name") or self.builder_name or (loaded.builders[0] if loaded.builders else ""))
        b_fn = getattr(loaded.mod, b_name, None) if b_name else None
        specs: list[CompileTaskSpec] = []
        if callable(b_fn):
            loaded.records.clear()
            loaded.trackb_records.clear()
            if callable(gk_fn := getattr(loaded.mod, "_get_kernel", None)):
                pool_gk = {**pool, "builder_fn": b_fn, "pass_configs": dict(sel_pc), "compile_flags": tuple(sel_cf),
                           "extra_builder_kwargs": ({**overrides, **raw_bk} or None)}
                if (gk_info := _cached_sig_info(gk_fn)) and all(
                    p.name in pool_gk for p in gk_info[0].parameters.values()
                    if p.default is inspect.Parameter.empty and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
                ):
                    try:
                        gk_fn(**_filter_kwargs_for_callable(gk_fn, pool_gk))
                    except Exception:
                        loaded.records.clear()
            if loaded.records:
                if raw_bk:
                    loaded.records[:] = [
                        (bn, _freeze_pair_tuple({**dict(kw), "extra_builder_kwargs": raw_bk}), pc, cf)
                        for bn, kw, pc, cf in loaded.records
                    ]
                specs.extend(self._collect_captured_specs(loaded, pool, sel_name=sel_name))
            else:
                res = b_fn(**_filter_kwargs_for_callable(b_fn, pool))
                b_kw = res.static_kwargs if isinstance(res, _CapturedPrimFunc) else _freeze_pair_tuple(_filter_kwargs_for_callable(b_fn, pool))
                if raw_bk:
                    b_kw = _freeze_pair_tuple({**dict(b_kw), "extra_builder_kwargs": raw_bk})
                specs.append(self._spec_from_capture(loaded, b_name, b_kw, sel_pc, tuple(sel_cf), selector_name=sel_name))

        if int(pool.get("num_kv_splits", 1) or 1) > 1 and callable(comb_fn := getattr(loaded.mod, "_get_combine_kernel", None)):
            loaded.records.clear()
            loaded.trackb_records.clear()
            comb_pool = {**pool, "pass_configs": dict(sel_pc), "compile_flags": tuple(sel_cf), "extra_combine_kwargs": raw_ck or None}
            with contextlib.suppress(Exception):
                comb_fn(**_filter_kwargs_for_callable(comb_fn, comb_pool))
            if loaded.records:
                cb_kw = _freeze_pair_tuple(_filter_kwargs_for_callable(comb_fn, comb_pool))
                loaded.records[:] = [
                    ("_get_combine_kernel", cb_kw, pc, cf) if bn == "unknown_builder" else (bn, kw, pc, cf)
                    for bn, kw, pc, cf in loaded.records
                ]
            specs.extend(self._collect_captured_specs(loaded, pool, sel_name=sel_name))
        return tuple(specs)

    def extract_case_tasks(self, module_source: str, case_id: str, case_spec: dict[str, Any]) -> tuple[CompileTaskSpec, ...]:
        """Extract ordered CompileTaskSpecs emitted by candidate selector for *case_spec*."""
        _ = case_id
        return self._extract_case_tasks_loaded(_load_plan_module(module_source), case_spec)


def _bounded_compile_diagnostic(output_text: str, *, limit: int = _COMPILE_DIAGNOSTIC_LIMIT) -> str:
    """Keep primary compiler errors without deprecation cascades plus a bounded output tail."""
    text = output_text.strip()
    if len(text) <= limit:
        return text
    raw_lines = text.splitlines()
    kept: list[str] = []
    skip_cascade = skip_bt = False
    for line in raw_lines:
        if _COMPILE_ERROR_RE.search(line):
            skip_cascade = skip_bt = False
            kept.append(line)
        elif _DEPRECATED_WARN_RE.search(line):
            skip_cascade, skip_bt = True, False
        elif line.strip() == "Stack trace:":
            skip_bt, skip_cascade = True, False
        elif skip_bt:
            if not (_TVM_BT_RE.search(line) or not line.strip()):
                skip_bt = False
                kept.append(line)
        elif skip_cascade:
            st = line.strip()
            if not st or _NON_ERROR_DIAG_HEADER_RE.match(st):
                skip_cascade = False
                if st:
                    kept.append(line)
        else:
            kept.append(line)
    lines = kept if (kept and any(_COMPILE_ERROR_RE.search(l) for l in kept)) else raw_lines
    eff = "\n".join(lines).strip()
    if len(eff) <= limit:
        return eff
    if not (err_idxs := [i for i, ln in enumerate(lines) if _COMPILE_ERROR_RE.search(ln)]):
        return eff[-limit:]
    ei = err_idxs[0]
    tail = eff[-limit:]
    if lines[ei].strip() in tail:
        return tail
    ctx = "\n".join(lines[ei : min(len(lines), ei + 7)]).strip()
    head = ctx if len(ctx) <= limit // 2 else lines[ei].strip()[: limit // 2]
    prefix = f"{head}\n[... compiler output truncated ...]\n"
    return prefix[:limit] if len(prefix) >= limit else prefix + eff[-(limit - len(prefix)) :]


_OVERFLOW_TABLES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("VF_STACK_OVERFLOW", ("exceeded vf stack size", "vf_stack", "vf stack", "stack frame size", "stack overflow")),
    ("L0C_SRAM_OVERFLOW", ("l0c_sram_overflow", "l0c overflow", "l0c_overflow", "l0c capacity", "exceeds l0c", "exceeded l0c")),
    ("L1_SRAM_OVERFLOW", ("l1_sram_overflow", "l1 overflow", "l1_overflow", "cbuf overflow", "l1 capacity", "cbuf capacity", "exceeds l1", "exceeded l1")),
    ("UB_SRAM_OVERFLOW", ("ub_sram_overflow", "ub_overflow", "ubuf_overflow", "ub overflow", "ubuf overflow", "ub capacity", "ubuf capacity", "exceeds ub", "exceeded ub", "exceeds ubuf", "exceeded ubuf")),
)


def _classify_stage_b_overflow_fault_code(combined_diag: str, summary: BishengResourceSummary | None, default_code: str) -> str:
    if summary is not None and int(summary.stack_bytes) > 0:
        return "VF_STACK_OVERFLOW"
    low = (combined_diag or "").lower()
    for code, toks in _OVERFLOW_TABLES:
        if any(t in low for t in toks):
            return code
    caps = ("overflow", "exceed", "capacity", "out of memory", "insufficient")
    if any(c in low for c in caps):
        if "l0c" in low or "__cc__" in low:
            return "L0C_SRAM_OVERFLOW"
        if any(t in low for t in ("l1 ", "l1_", "cbuf", "__cbuf__")):
            return "L1_SRAM_OVERFLOW"
        if "ubuf" in low or "__ubuf__" in low or re.search(r"\bub\b", low):
            return "UB_SRAM_OVERFLOW"
    return default_code


def _parse_cce_res_usage_with_summary(output_text: str, *, asm_or_dump_text: str = "") -> tuple[int, bool, BishengResourceSummary]:
    summary = parse_bisheng_res_usage(output_text or "", asm_or_dump_text=asm_or_dump_text or "")
    vloop_hit = summary.vloopv2_loops > 0 and summary.software_loops == 0 and not _VLOOPV2_MISS_RE.search(f"{output_text}\n{asm_or_dump_text}")
    return int(summary.spill_bytes), vloop_hit, summary


def parse_cce_res_usage(output_text: str) -> tuple[int, bool]:
    """Parse bisheng --cce-res-usage diagnostics into (spill_bytes, vloopv2_hit)."""
    sp, vl, _ = _parse_cce_res_usage_with_summary(output_text)
    return sp, vl


@contextlib.contextmanager
def ascend_compile_session(target: str = "ascend", pass_configs: Mapping[str, object] | None = None):
    """Enter contiguous TVM target and optional PassContext for Ascend lowering."""
    from tilelang import tvm  # noqa: PLC0415
    from tilelang.backend.module import create_backend_context  # noqa: PLC0415

    with contextlib.ExitStack() as stack:
        if tvm.target.Target.current(allow_none=True) is None:
            stack.enter_context(create_backend_context(target).target)
        if pass_configs is not None:
            stack.enter_context(tvm.transform.PassContext(config=dict(pass_configs)))
        yield


def _resolve_cann_root(cann_root: str | Path | None = None, *, bisheng_bin: str | Path | None = None) -> Path | None:
    if cann_root is not None and str(cann_root).strip():
        return Path(cann_root).expanduser()
    for ek in ("ASCEND_TOOLKIT_HOME", "ASCEND_HOME_PATH", "BISHENG_HOME"):
        if raw := os.environ.get(ek, "").strip():
            cand = Path(raw).expanduser()
            if ek == "BISHENG_HOME" and not any((cand / s).is_dir() for s in _CANN_INCLUDE_SUBDIRS[:4]):
                for p in cand.parents:
                    if any((p / s).is_dir() for s in _CANN_INCLUDE_SUBDIRS[:4]):
                        return p
            return cand
    for raw in (*(os.environ.get(k, "").strip() for k in ("ASCEND_OPP_PATH", "ASCEND_AICPU_PATH")),):
        if raw and any(((p := Path(raw).expanduser().parent) / s).is_dir() for s in _CANN_INCLUDE_SUBDIRS[:4]):
            return p
    if rbin := (bisheng_bin or shutil.which("bisheng")):
        for p in Path(str(rbin)).resolve().parents:
            if any((p / s).is_dir() for s in _CANN_INCLUDE_SUBDIRS[:4]):
                return p
    return None


def _resolve_bisheng_bin(bisheng_bin: str | Path | None = None, cann_root: str | Path | None = None) -> str | None:
    if bisheng_bin is not None:
        raw = str(bisheng_bin).strip()
        return (str(Path(raw)) if Path(raw).is_file() else shutil.which(raw)) if raw else None
    if (bh := os.environ.get("BISHENG_HOME", "").strip()) and (cand := Path(bh).expanduser() / "bin" / "bisheng").is_file():
        return str(cand)
    if found := shutil.which("bisheng"):
        return found
    if (rc := _resolve_cann_root(cann_root)) is not None:
        return next((str(rc / rel) for rel in ("tools/ccec_compiler/bin/bisheng", "bin/bisheng") if (rc / rel).is_file()), None)
    return None


def _detect_cann_include_dirs(cann_root: str | Path | None = None, *, bisheng_bin: str | Path | None = None) -> list[str]:
    roots: list[Path] = [Path(str(cann_root)).expanduser()] if (cann_root is not None and str(cann_root).strip()) else []
    roots.extend(Path(r).expanduser() for k in ("ASCEND_HOME_PATH", "ASCEND_TOOLKIT_HOME") if (r := os.environ.get(k, "").strip()))
    if inf := _resolve_cann_root(cann_root, bisheng_bin=bisheng_bin):
        roots.append(inf)
    found: list[str] = []
    seen: set[str] = set()
    for root in roots:
        for sub in _CANN_INCLUDE_SUBDIRS:
            if (inc := root / sub).is_dir() and (rk := str(inc.resolve())) not in seen:
                seen.add(rk)
                found.append(str(inc))
    return found


def _inject_cann_cpath_env(env_map: dict[str, str] | None = None) -> list[str]:
    if incs := _detect_cann_include_dirs():
        tgt = os.environ if env_map is None else env_map
        tgt["CPATH"] = os.pathsep.join(dict.fromkeys([*incs, *(p for p in tgt.get("CPATH", "").split(os.pathsep) if p)]))
    return incs


def _resolve_toolchain_tool(tool_names: Sequence[str], *, bisheng_bin: str | Path | None = None) -> str | None:
    if bisheng_bin is not None and (bp := Path(str(bisheng_bin))).is_file():
        if hit := next((str(c) for n in tool_names if (c := bp.resolve().with_name(n)).is_file()), None):
            return hit
    return next((w for n in tool_names if (w := shutil.which(n))), None)


def _verify_object_has_instantiated_kernel(obj_path: Path, *, bisheng_bin: str | Path | None = None, subproc_run: Callable[..., Any] | None = None) -> tuple[bool, str]:
    if not obj_path.is_file():
        return False, "stage_b_ascendc_missing_object: object file was not generated"
    run_fn = subproc_run or subprocess.run
    if nm_bin := _resolve_toolchain_tool(("llvm-nm", "nm"), bisheng_bin=bisheng_bin):
        if (proc := run_fn([nm_bin, "--defined-only", str(obj_path)], text=True, capture_output=True, timeout=15, check=False)).returncode == 0:
            if any(re.search(r"\b[Tt]\s+(?!\$)\S+", ln.strip()) for ln in (proc.stdout or "").splitlines()):
                return True, ""
            return False, "stage_b_ascendc_uninstantiated_template: object file contains zero instantiated code symbols (.text is empty); provide concrete template instantiation via _render_trackb_tu"
    if od_bin := _resolve_toolchain_tool(("llvm-objdump", "objdump"), bisheng_bin=bisheng_bin):
        if (proc := run_fn([od_bin, "-h", str(obj_path)], text=True, capture_output=True, timeout=15, check=False)).returncode == 0:
            m = re.search(r"\.text(?:\.\S+)?\s+([0-9a-fA-F]+)\s", proc.stdout or "")
            return (True, "") if (m and int(m.group(1), 16) > 0) else (False, "stage_b_ascendc_uninstantiated_template: .text section size is 0; provide concrete template instantiation via _render_trackb_tu")
    return (True, "") if obj_path.stat().st_size > 2048 else (False, "stage_b_ascendc_uninstantiated_template: object file contains no verifiable instantiated kernel code")


def _disassemble_object(obj_path: Path, *, bisheng_bin: str | Path | None = None, subproc_run: Callable[..., Any] | None = None) -> str:
    od_bin = _resolve_toolchain_tool(("llvm-objdump", "objdump"), bisheng_bin=bisheng_bin)
    if not od_bin or not obj_path.is_file():
        return ""
    proc = (subproc_run or subprocess.run)([od_bin, "-d", str(obj_path)], text=True, capture_output=True, timeout=30, check=False)
    return proc.stdout if proc.returncode == 0 else ""


_OBJDUMP_ZERO_KEYS = (
    "total_insns", "cube_mmad", "fixpipe", "mte1", "mte2", "mte3", "vec", "vloopv2",
    "dcci", "sync", "sw_back_branches", "vf_sw_back_branches", "scalar_regs", "vec_regs",
    "gm_to_l1_insns", "gm_to_ub_insns", "l1_to_l0a_insns", "l1_to_l0b_insns",
    "l0c_to_ub_insns", "l0c_to_l1_insns", "l0c_to_gm_insns", "ub_to_gm_insns", "ub_to_l1_insns",
)


def _parse_objdump_pipe_counts(objdump_text: str) -> dict[str, int]:
    counts = dict.fromkeys(_OBJDUMP_ZERO_KEYS, 0)
    if not objdump_text.strip():
        return counts
    xregs, vregs, in_vf = set(), set(), False
    for raw_ln in objdump_text.splitlines():
        if lbl := re.match(r"^[0-9a-f]+\s+<([^>]+)>:\s*$", raw_ln.strip()):
            in_vf = ".vector.thread" in lbl.group(1)
            continue
        if not (m := re.match(r"^\s*([0-9a-f]+):\s+[0-9a-f]+\s+([A-Za-z0-9_.]+)(.*)$", raw_ln)):
            continue
        pc, mnem, rest = int(m.group(1), 16), m.group(2).upper(), m.group(3)
        counts["total_insns"] += 1
        xregs.update(int(r) for r in re.findall(r"\bX(\d+)\b", rest))
        vregs.update(int(r) for r in re.findall(r"\bV(\d+)\b", rest))
        if mnem.startswith("VLOOP"):
            counts["vloopv2"] += int("VLOOPV2" in mnem or "VLOOP_V2" in mnem)
        elif mnem.startswith("MMAD"):
            counts["cube_mmad"] += 1
        elif mnem.startswith("FIX_"):
            counts["fixpipe"] += 1
            for tok, key in (("L1", "l0c_to_l1_insns"), ("CBUF", "l0c_to_l1_insns"), ("UB", "l0c_to_ub_insns"), ("GM", "l0c_to_gm_insns")):
                if tok in mnem:
                    counts[key] += 1
                    break
        elif mnem.startswith("LOAD_L1_TO_L0"):
            counts["mte1"] += 1
            counts["l1_to_l0b_insns" if "L0B" in mnem else "l1_to_l0a_insns"] += 1
        elif mnem.startswith("MOV_OUT_TO_"):
            counts["mte2"] += 1
            if "L1" in mnem or "CBUF" in mnem:
                counts["gm_to_l1_insns"] += 1
            elif "UB" in mnem:
                counts["gm_to_ub_insns"] += 1
        elif mnem.startswith("MOV_UB_TO_"):
            counts["mte3"] += 1
            counts["ub_to_l1_insns" if ("L1" in mnem or "CBUF" in mnem) else "ub_to_gm_insns"] += 1
        elif mnem.startswith("DCCI"):
            counts["dcci"] += 1
        elif mnem.startswith(("SET_FLAG", "WAIT_FLAG", "SET_INTRA", "WAIT_INTRA", "BAR.")):
            counts["sync"] += 1
        elif mnem.startswith("V") and not mnem.startswith("VF"):
            counts["vec"] += 1
        if mnem.startswith(("JUMP", "B.", "CBNZ", "CBZ", "TBNZ", "TBZ")) or mnem == "B":
            tm, hm = re.search(r"\+0x([0-9a-f]+)>", rest, re.IGNORECASE), re.search(r"#(-?\d+)", rest)
            if (tm and int(tm.group(1), 16) <= pc) or (not tm and hm and int(hm.group(1)) <= pc):
                counts["sw_back_branches"] += 1
                counts["vf_sw_back_branches"] += int(in_vf)
    counts["scalar_regs"], counts["vec_regs"] = len(xregs), len(vregs)
    return counts


def _augment_bisheng_telemetry_text(compiler_out: str, objdump_text: str) -> str:
    dis = _parse_objdump_pipe_counts(objdump_text)
    vec_regs = max([*(int(m) for m in re.findall(r"\bVReg\s+number\s*:\s*(\d+)", compiler_out)), dis["vec_regs"], 0])
    stack_b = max((int(m) for m in re.findall(r"(?:^|[,\s])Stack\s+size\s*:\s*(\d+)\s*bytes", compiler_out, re.IGNORECASE)), default=0)
    extra: list[str] = []
    if vec_regs > 0 and not re.search(r"(?:vector\s+register\s+usage|vec_reg_num)", compiler_out, re.IGNORECASE):
        extra.append(f"Vector Register Usage: {vec_regs}")
    if dis["scalar_regs"] > 0 and not re.search(r"(?:scalar\s+register\s+usage|scalar_reg_num)", compiler_out, re.IGNORECASE):
        extra.append(f"Scalar Register Usage: {dis['scalar_regs']}")
    if stack_b > 0 and not re.search(r"stack\s+frame\s+size", compiler_out, re.IGNORECASE):
        extra.append(f"Stack Frame Size: {stack_b} bytes")
    if objdump_text.strip():
        vloop_cnt = dis["vloopv2"]
        extra.extend((f"vloopv2_loops: {vloop_cnt}", f"software_loops: {dis['vf_sw_back_branches'] if vloop_cnt > 0 else dis['sw_back_branches']}"))
    return f"{compiler_out}\n" + "\n".join(extra) if extra else compiler_out


def _detect_tilelang_include_dir() -> str | None:
    try:
        import tilelang  # noqa: PLC0415

        if ((sd := Path(tilelang.__file__).resolve().parents[1] / "src") / "tl_templates" / "ascend").is_dir():
            return str(sd)
    except Exception:
        pass
    for raw in (os.environ.get("TILEFOUNDRY_NPU_TILELANG_SOURCE", "").strip(), os.environ.get("TILEFOUNDRY_NPU_TILELANG_REPO", "").strip()):
        if raw and ((sd := Path(raw).expanduser() / "src") / "tl_templates" / "ascend").is_dir():
            return str(sd)
    sib = Path(__file__).resolve().parents[5].parent / "tilelang" / "src"
    return str(sib) if (sib / "tl_templates" / "ascend").is_dir() else None


@dataclass(frozen=True)
class _AscendcCompileContext:
    extra_flags: list[str]
    has_explicit_x: bool
    std_flags: list[str]
    include_flags: list[str]
    env: dict[str, str]


def _prepare_ascendc_compile_context(
    *, compile_flags: Sequence[str] | str | None, cann_root: str | Path | None,
    resolved_bisheng: str, extra_include_dirs: Sequence[str | Path] | str | Path | None,
) -> _AscendcCompileContext:
    extra_flags = _normalize_flag_list(compile_flags)
    std_flags = [] if any(f.startswith("-std=") for f in extra_flags) else ["-std=c++20"]
    include_flags: list[str] = []
    seen = set(extra_flags)
    raw_incs: list[str] = ([str(extra_include_dirs)] if isinstance(extra_include_dirs, (str, Path)) else [str(i) for i in (extra_include_dirs or ())])
    if tl_inc := _detect_tilelang_include_dir():
        raw_incs.append(tl_inc)
    raw_incs.extend(_detect_cann_include_dirs(cann_root, bisheng_bin=resolved_bisheng))
    for inc in raw_incs:
        if (s := inc.strip()) and Path(s).is_dir() and (flag := f"-I{s}") not in seen:
            seen.add(flag)
            include_flags.append(flag)
    env = dict(os.environ)
    if (rc := _resolve_cann_root(cann_root, bisheng_bin=resolved_bisheng)) is not None and rc.is_dir() and not env.get("ASCEND_HOME_PATH"):
        env["ASCEND_HOME_PATH"] = str(rc).strip()
    return _AscendcCompileContext(extra_flags, "-x" in extra_flags, std_flags, include_flags, env)


def _atomic_copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(dst.parent), prefix=f".{dst.name}.")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        shutil.copy2(src, tmp_path)
        os.replace(tmp_path, dst)
    finally:
        tmp_path.unlink(missing_ok=True)


def compute_trackb_shared_so_filename(cpp_source: str, compile_flags: Sequence[str] | str | None = None) -> str:
    """Compute canonical cached shared-object filename for a Track-B TU and flags."""
    non_inc = [f for f in _normalize_flag_list(compile_flags) if not f.startswith("-I")]
    return f"trackb_{hashlib.sha256((cpp_source + '|' + ' '.join(non_inc)).encode('utf-8')).hexdigest()[:16]}.so"


def _build_trackb_shared_link_cmd(resolved_bisheng: str, *, arch: str, ctx: _AscendcCompileContext, src_path: Path, so_path: Path) -> list[str]:
    return [
        resolved_bisheng, "-O2", "-fPIC", *ctx.std_flags, "-mllvm", "-cce-aicore-dcpreload-args=false",
        f"--npu-arch={arch}", *ctx.include_flags, *ctx.extra_flags, "-Wl,--no-mmap-output-file", "-shared",
        str(src_path), "-o", str(so_path),
    ]


def compile_trackb_shared_so(
    cpp_source: str, *, cache_dir: str | Path, compile_flags: Sequence[str] | str | None = None,
    arch: str = "dav-3510", bisheng_bin: str | Path | None = None, cann_root: str | Path | None = None,
    extra_include_dirs: Sequence[str | Path] | str | Path | None = None, timeout_sec: int = 120,
    subproc_run: Callable[..., Any] | None = None,
) -> tuple[bool, Path | None, str]:
    """Compile a Track-B C++20 TU into a content-addressed cached .so via bisheng."""
    if not isinstance(cpp_source, str) or not cpp_source.strip():
        return (False, None, "stage_b_ascendc_empty_source")
    cache_root = Path(cache_dir).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    target_so = cache_root / compute_trackb_shared_so_filename(cpp_source, compile_flags)
    if is_verified_binary_file(target_so):
        return (True, target_so, "")
    if (resolved_bisheng := _resolve_bisheng_bin(bisheng_bin, cann_root=cann_root)) is None:
        return (False, None, "stage_b_ascendc_bisheng_not_found")
    ctx = _prepare_ascendc_compile_context(compile_flags=compile_flags, cann_root=cann_root, resolved_bisheng=resolved_bisheng, extra_include_dirs=extra_include_dirs)
    with tempfile.TemporaryDirectory(prefix="tf_trackb_shared_") as tmp_dir:
        src_path = Path(tmp_dir) / ("kernel.cpp" if ctx.has_explicit_x else "kernel.asc")
        tmp_so = Path(tmp_dir) / "kernel.so"
        src_path.write_text(cpp_source, encoding="utf-8")
        try:
            proc = (subproc_run or subprocess.run)(
                _build_trackb_shared_link_cmd(resolved_bisheng, arch=arch, ctx=ctx, src_path=src_path, so_path=tmp_so),
                text=True, capture_output=True, cwd=tmp_dir, env=ctx.env, timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired:
            return (False, None, f"stage_b_ascendc_shared_link_timeout:{timeout_sec}s")
        raw = (proc.stderr or "") + "\n" + (proc.stdout or "")
        if proc.returncode != 0 or not is_verified_binary_file(tmp_so):
            return (False, None, f"stage_b_ascendc_shared_link_failed (rc={proc.returncode}): {_bounded_compile_diagnostic(raw)}")
        _atomic_copy_file(tmp_so, target_so)
        return (True, target_so, "")


def build_stage_b_driver_source(
    cand_path: str | Path, *, factory_name: str | None = None, factory_kwargs: Mapping[str, object] | None = None,
    config_selector_name: str | None = None, pass_configs: Mapping[str, object] | None = None,
    compile_flags: Sequence[str] | str | None = None, dims: Mapping[str, object] | None = None,
    enable_device_compile: bool = True, kernel_dump_path: str | Path | None = None,
    tir_dump_path: str | Path | None = None, compile_for_cache: bool = False,
    emit_cache_bins_path: Path | str | None = None,
) -> str:
    """Build standalone Python driver lowering and device-compiling a candidate DSL file."""
    if not enable_device_compile:
        raise ValueError("enable_device_compile=False bypass has been removed")
    ext_pc, ext_cf = _merge_pass_configs_and_flags(None, None, pass_configs, compile_flags)
    payload = json.dumps({
        "cand_path": str(cand_path), "factory_name": factory_name,
        "factory_kwargs": _merge_dims_into_kwargs(factory_kwargs, dims),
        "config_selector_name": config_selector_name, "pass_configs": ext_pc, "compile_flags": ext_cf,
        "kernel_dump_path": str(kernel_dump_path) if kernel_dump_path is not None else None,
        "tir_dump_path": str(tir_dump_path) if tir_dump_path is not None else None,
        "compile_for_cache": bool(compile_for_cache),
        "emit_cache_bins_path": str(emit_cache_bins_path) if emit_cache_bins_path is not None else None,
    })
    template = '''
import contextlib, importlib.util, inspect, json, os, pathlib, tilelang
from tilelang import tvm
from tilelang.backend.module import create_backend_context
from tvm import tirx, IRModule

@contextlib.contextmanager
def _ascend_compile_session(pc=None):
    with contextlib.ExitStack() as st:
        if tvm.target.Target.current(allow_none=True) is None:
            st.enter_context(create_backend_context("ascend").target)
        if pc is not None:
            st.enter_context(tvm.transform.PassContext(config=dict(pc)))
        yield

_cfg = json.loads(__PAYLOAD__)
_src = pathlib.Path(_cfg["cand_path"]).read_text(encoding="utf-8")
_has_prim = "prim_func" in _src
_spec = importlib.util.spec_from_file_location("_tf_stage_b", _cfg["cand_path"])
_mod = importlib.util.module_from_spec(_spec)
_filt = lambda obj, kw: dict(kw) if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in inspect.signature(obj).parameters.values()) else {k: v for k, v in kw.items() if k in inspect.signature(obj).parameters}
_nflags = lambda raw: ([] if raw is None else (raw.split() if isinstance(raw, str) else [str(getattr(x, "value", x)).strip() for x in raw if str(x).strip()]))

def _collect_bins(obj):
    found, seen = [], set()
    def _add(raw):
        if raw:
            try:
                p = pathlib.Path(str(raw))
                if p.is_file() and (p.suffix in (".so", ".o", ".bin") or ".so" in p.suffixes) and p.stat().st_size > 0:
                    rp = str(p.resolve())
                    if rp not in seen:
                        seen.add(rp); found.append(rp)
            except OSError:
                pass
    objs = list(obj) if isinstance(obj, (tuple, list)) else [obj]
    for cur in list(objs):
        if cur is not None:
            objs.extend(s for sa in ("adapter", "lib", "kernel_lib", "artifact") if (s := getattr(cur, sa, None)) is not None)
    tl_res = str(pathlib.Path(e).resolve()) if (e := os.environ.get("TILELANG_CACHE_DIR", "")) else ""
    for cur in objs:
        if cur is None: continue
        if isinstance(cur, (str, pathlib.Path)): _add(cur); continue
        if isinstance(cur, dict):
            for v in cur.values():
                if isinstance(v, (str, pathlib.Path)): _add(v)
            continue
        for attr in ("kernel_path", "libpath", "lib_path", "so_path", "binary_path", "artifact_path", "path", "_name", "_tilelang_cache_path"):
            _add(getattr(cur, attr, None))
        for d in ([getattr(cur, "_tilelang_cache_path", None), pathlib.Path(tl_res) / str(getattr(cur, "cache_key", None) or getattr(cur, "_tilelang_cache_key", None) or "")] if tl_res else [getattr(cur, "_tilelang_cache_path", None)]):
            if d:
                try:
                    if (dp := pathlib.Path(str(d))).is_dir():
                        for cand in sorted(dp.rglob("*")): _add(cand)
                except OSError: pass
        if (ck := getattr(cur, "cache_key", None) or getattr(cur, "_tilelang_cache_key", None)) and tl_res:
            for cand in sorted(pathlib.Path(tl_res).glob(f"*{ck}*")): _add(cand)
    return found

_kw, _sel_pc, _sel_cf, _raw_bk, _raw_ck, _fn, _compiled = dict(_cfg["factory_kwargs"]), {}, [], {}, {}, None, None
with _ascend_compile_session():
    _spec.loader.exec_module(_mod)
    if (_sn := _cfg["config_selector_name"]) and callable(_sfn := getattr(_mod, _sn, None)):
        if isinstance(_so := _sfn(**_filt(_sfn, _kw)), dict):
            if isinstance(_rpc := _so.get("pass_configs"), dict):
                for _k, _v in _rpc.items():
                    if _k == "tl.device_compile_flags": _sel_cf.extend(_nflags(_v))
                    else: _sel_pc[str(_k)] = _v
            elif _rpc is not None: _sel_cf.extend(_nflags(_rpc))
            _sel_cf.extend(_nflags(_so.get("compile_flags")))
            _raw_bk = dict(_so["builder_kwargs"]) if isinstance(_so.get("builder_kwargs"), dict) else {}
            _raw_ck = dict(_so["combine_kwargs"]) if isinstance(_so.get("combine_kwargs"), dict) else {}
            _kw.update({k: v for k, v in _so.items() if k not in ("pass_configs", "compile_flags", "builder_kwargs", "combine_kwargs")})
    _fname, _used_cache_path = _cfg["factory_name"], False
    if _cfg["compile_for_cache"]:
        _cpc, _ccf = {**_sel_pc, **_cfg["pass_configs"]}, list(dict.fromkeys([*_sel_cf, *_cfg["compile_flags"]]))
        _gkp = {**_kw, "pass_configs": _cpc, "compile_flags": tuple(_ccf), "extra_builder_kwargs": _raw_bk or _kw.get("extra_builder_kwargs"), "extra_combine_kwargs": _raw_ck or _kw.get("extra_combine_kwargs")}
        _vk = (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        _bobj = getattr(_mod, _fname, None) if (_fname and hasattr(_mod, _fname)) else None
        for _ds in (getattr(_mod, "kernel", None), getattr(_mod, _sn, None) if _sn else None, _bobj):
            if callable(_ds):
                try:
                    for _dp in inspect.signature(_ds).parameters.values():
                        if _dp.name not in _gkp and _dp.default is not inspect.Parameter.empty and _dp.kind not in _vk: _gkp[_dp.name] = _dp.default
                except (TypeError, ValueError): pass
        if _fname == "_get_combine_kernel" and callable(_cfn := getattr(_mod, "_get_combine_kernel", None)):
            _compiled, _used_cache_path = _cfn(**_filt(_cfn, _gkp)), True
        else:
            if callable(_bobj): _gkp["builder_fn"] = _bobj
            if callable(_gk := getattr(_mod, "_get_kernel", None)) and all(_p.name in _gkp for _p in inspect.signature(_gk).parameters.values() if _p.default is inspect.Parameter.empty and _p.kind not in _vk):
                _compiled, _used_cache_path = _gk(**_filt(_gk, _gkp)), True
    if not _used_cache_path:
        _kwb = {**_kw, **_raw_bk}
        if _fname and hasattr(_mod, _fname):
            _o = getattr(_mod, _fname)
            _fn = _o if (_fname == "kernel" and not _has_prim and callable(_o)) else (_o(**_filt(_o, _kwb)) if callable(_o) else _o)
        if _fn is None:
            _fn = next((_v for _v in vars(_mod).values() if isinstance(_v, (tirx.PrimFunc, IRModule))), None)
        if _fn is None and callable(_ko := getattr(_mod, "kernel", None)) and not _has_prim: _fn = _ko
        if _fn is None:
            _fn = next((_v(**_filt(_v, _kwb)) for _k, _v in vars(_mod).items() if not _k.startswith("_") and callable(_v) and getattr(_v, "__module__", None) == _mod.__name__), None)
        assert _fn is not None
_merged_pc = None
if _compiled is None and _fn is not None:
    if not isinstance(_fn, (tirx.PrimFunc, IRModule)) and callable(_fn):
        _ch = getattr(_mod, "compile_kernel", getattr(_mod, "build_kernel", None))
        if not callable(_ch): raise RuntimeError("stage_b_non_kernel_callable: plain Python function lacks @T.prim_func or compile_kernel")
        _compiled = _ch()
    else:
        _mpc, _mcf = {}, []
        _abs = lambda seq: _mcf.extend(_f for _f in seq if _f and _f not in _mcf)
        if (_attrs := getattr(_fn, "attrs", None)) and hasattr(_attrs, "items"):
            for _pk in ("tl.pass_configs", "tilelang_pass_configs"):
                if _pk in _attrs and hasattr(_attrs[_pk], "items"):
                    for _ak, _av in _attrs[_pk].items():
                        _val = getattr(_av, "value", _av)
                        if str(_ak) == "tl.device_compile_flags": _abs(_nflags(_val))
                        else: _mpc[str(_ak)] = _val
            for _ck in ("tir.cce_flags", "tilelang_compile_flags"):
                if _ck in _attrs: _abs(_nflags(_attrs[_ck]))
        _mpc.update(_sel_pc); _abs(_sel_cf); _mpc.update(_cfg["pass_configs"]); _abs(_cfg["compile_flags"])
        if not _cfg["compile_for_cache"]: _abs(["--cce-res-usage"])
        if _mcf: _mpc["tl.device_compile_flags"] = list(_mcf)
        if _cfg.get("tir_dump_path"): pathlib.Path(_cfg["tir_dump_path"]).write_text(str(_fn), encoding="utf-8")
        _merged_pc = _mpc or None
        with _ascend_compile_session(_merged_pc):
            if _cfg["compile_for_cache"] and callable(getattr(tilelang, "compile", None)):
                _compiled = tilelang.compile(_fn, target="ascend", pass_configs=_merged_pc, compile_flags=_mcf or None)
            else:
                try: _compiled = tilelang.lower(_fn, target="ascend", enable_device_compile=True)
                except Exception:
                    try:
                        _pre = tilelang.lower(_fn, target="ascend", enable_device_compile=False)
                        if _cfg.get("tir_dump_path"): pathlib.Path(_cfg["tir_dump_path"]).write_text(str(getattr(_pre, "device_mod", None) or _fn), encoding="utf-8")
                        if _cfg.get("kernel_dump_path") and getattr(_pre, "kernel_source", None): pathlib.Path(_cfg["kernel_dump_path"]).write_text(str(_pre.kernel_source), encoding="utf-8")
                    except Exception: pass
                    raise
                if _cfg.get("tir_dump_path"): pathlib.Path(_cfg["tir_dump_path"]).write_text(str(getattr(_compiled, "device_mod", None) or _fn), encoding="utf-8")
                assert _compiled.kernel_source
                if _cfg.get("kernel_dump_path"): pathlib.Path(_cfg["kernel_dump_path"]).write_text(str(_compiled.kernel_source), encoding="utf-8")
if _compiled is not None and _cfg.get("kernel_dump_path") and not pathlib.Path(_cfg["kernel_dump_path"]).is_file():
    _tgts = [_compiled[0] if isinstance(_compiled, (tuple, list)) and _compiled else _compiled]
    _ksrc = next((v for c in _tgts + [getattr(_tgts[0], "adapter", None)] if c is not None for v in ((c.get_kernel_source() if callable(getattr(c, "get_kernel_source", None)) else None), getattr(c, "kernel_source", None)) if isinstance(v, str) and v.strip()), "")
    if _ksrc: pathlib.Path(_cfg["kernel_dump_path"]).write_text(_ksrc, encoding="utf-8")
if _cfg.get("emit_cache_bins_path"):
    (_ep := pathlib.Path(_cfg["emit_cache_bins_path"])).parent.mkdir(parents=True, exist_ok=True)
    _ep.write_text(json.dumps(_collect_bins(_compiled)), encoding="utf-8")
'''
    return template.lstrip().replace("__PAYLOAD__", repr(payload))


def _resolve_artifact_store(artifact_store: ArtifactStore | None) -> ArtifactStore:
    return artifact_store if artifact_store is not None else ArtifactStore(resolve_unified_tl_cache_dir())


def _merge_cache_locked(worker_cache: Path, target_cache: Path) -> None:
    lock_path = target_cache / ".prewarm_manifest_v1.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as fh:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            merge_tl_cache_tree(worker_cache, target_cache)
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def _collect_built_bins(
    work_dir: Path, worker_cache: Path, target_cache: Path, pre_bins: set[str], emit_path: Path | None = None
) -> list[Path]:
    bins: list[Path] = []
    if emit_path is not None and emit_path.is_file():
        with contextlib.suppress(OSError, json.JSONDecodeError):
            for raw in json.loads(emit_path.read_text(encoding="utf-8")):
                p = Path(str(raw))
                try:
                    cand = target_cache / p.relative_to(worker_cache)
                except ValueError:
                    cand = p
                if is_verified_binary_file(cand) and cand not in bins:
                    bins.append(cand)
    for rel in collect_new_tl_cache_binaries(worker_cache, pre_bins):
        if is_verified_binary_file(cand := target_cache / rel) and cand not in bins:
            bins.append(cand)
    for ext_p in sorted(work_dir.rglob("*")):
        if not ext_p.is_relative_to(worker_cache) and is_verified_binary_file(ext_p) and ext_p not in bins:
            bins.append(ext_p)
    return bins


def run_stage_b_compile(
    dsl_source: str, *, factory_name: str | None = None, factory_kwargs: Mapping[str, object] | None = None,
    config_selector_name: str | None = None, pass_configs: Mapping[str, object] | None = None,
    compile_flags: Sequence[str] | str | None = None, dims: Mapping[str, object] | None = None,
    python_bin: str | None = None, timeout_sec: int = 120, enable_device_compile: bool = True,
    artifact_store: ArtifactStore | None = None, artifact_id: str = "", regime_tag: str = "",
    spec_key: str = "", subproc_run: Callable[..., Any] | None = None,
) -> StageBResult:
    """Verify full TileLang + bisheng device compilation in an isolated subprocess."""
    if not enable_device_compile:
        raise ValueError("enable_device_compile=False bypass has been removed")
    t0 = time.monotonic()
    py_exe = c if (c := (python_bin or os.environ.get("TILEFOUNDRY_NPU_PYTHON", "")).strip()) and Path(c).is_file() else sys.executable
    env = dict(os.environ)
    env.setdefault("USE_CUDA", "0")
    _inject_cann_cpath_env(env)
    store = _resolve_artifact_store(artifact_store)
    store.root.mkdir(parents=True, exist_ok=True)
    norm_pc, norm_cf = _merge_pass_configs_and_flags(None, None, pass_configs, compile_flags)
    closure_src = extract_reachable_closure_source(dsl_source, regime_tag=regime_tag) if regime_tag else dsl_source
    target_aid = artifact_id or compute_artifact_task_key(normalized_kernel_unit_src=closure_src, cann_target="dav-3510", pass_configs=norm_pc, compile_flags=norm_cf, static_spec_signature=spec_key)
    spec_chk = CompileTaskSpec(target_aid, str(factory_name or ""), _freeze_pair_tuple(factory_kwargs), _freeze_pair_tuple(norm_pc), tuple(norm_cf))

    with tempfile.TemporaryDirectory(prefix="tf_stage_b_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        w_cache = tmp_root / "tl_cache"
        seed_tl_cache_tree(store.root, w_cache)
        pre_bins = snapshot_tl_cache_binaries(w_cache)
        env.update({"TILELANG_DISABLE_CACHE": "0", "TILELANG_CACHE_DIR": str(w_cache), "TILELANG_KERNEL_CACHE_USE_LIB_STAMP": "1"})
        cand_path, kdump, tir_dump = tmp_root / "candidate_dsl.py", tmp_root / "lowered_kernel.cce", tmp_root / "lowered_tir.tir"
        cand_path.write_text(dsl_source, encoding="utf-8")
        driver = build_stage_b_driver_source(
            cand_path, factory_name=factory_name, factory_kwargs=factory_kwargs, config_selector_name=config_selector_name,
            pass_configs=pass_configs, compile_flags=compile_flags, dims=dims, kernel_dump_path=kdump, tir_dump_path=tir_dump,
        )
        try:
            proc = (subproc_run or subprocess.run)([py_exe, "-c", driver], text=True, capture_output=True, cwd=tmp_dir, env=env, timeout=timeout_sec)
        except subprocess.TimeoutExpired:
            return _stage_b_fail(
                t0, f"stage_b_compile_timeout:{timeout_sec}s", "STAGE_B_COMPILE_TIMEOUT",
                cce=(kdump.read_text(encoding="utf-8", errors="ignore") if kdump.is_file() else ""),
                tir=(tir_dump.read_text(encoding="utf-8", errors="ignore") if tir_dump.is_file() else ""),
            )
        ldump = kdump.read_text(encoding="utf-8", errors="ignore") if kdump.is_file() else ""
        ltir = tir_dump.read_text(encoding="utf-8", errors="ignore") if tir_dump.is_file() else ""
        comb = (proc.stdout or "") + "\n" + (proc.stderr or "")
        sp, vl, summary = _parse_cce_res_usage_with_summary(comb, asm_or_dump_text=ldump)
        ma_prof = extract_microarch_profile_from_cce(ldump).to_dict() if ldump.strip() else None
        if proc.returncode != 0:
            raw_d = proc.stderr or proc.stdout or ""
            return _stage_b_fail(
                t0, f"stage_b_compile_failed (rc={proc.returncode}): {_bounded_compile_diagnostic(raw_d)}",
                _classify_stage_b_overflow_fault_code(f"{raw_d}\n{ldump}", summary, "STAGE_B_COMPILE_FAILED"),
                spill_bytes=sp, vloopv2_hit=vl, summary=summary, ma_profile=ma_prof, full_diag=raw_d, cce=ldump, tir=ltir,
            )
        if has_unreserved_vf_stack_overflow(spec_chk, summary.stack_bytes, comb):
            return _stage_b_fail(
                t0, f"stage_b_vf_stack_overflow: stack_bytes={summary.stack_bytes}", "VF_STACK_OVERFLOW",
                spill_bytes=sp, vloopv2_hit=vl, summary=summary, ma_profile=ma_prof, full_diag=comb, cce=ldump, tir=ltir,
            )
        _merge_cache_locked(w_cache, store.root)
        recorded_aid, recorded_bins = "", ()
        if built := _collect_built_bins(tmp_root, w_cache, store.root, pre_bins):
            rec = store.record_verified_artifact(
                target_aid, binary_paths=built, backend_kind="tilelang", flags=" ".join(norm_cf),
                spec_key=spec_key, closure_src=closure_src, metadata={"stack_bytes": int(summary.stack_bytes)}, copy_binaries=True,
            )
            recorded_aid, recorded_bins = target_aid, rec.binary_paths
        elif (ex := store.get_verified_artifact(target_aid)) is not None:
            recorded_aid, recorded_bins = ex.artifact_id, ex.binary_paths
    return StageBResult(
        passed=True, elapsed_sec=time.monotonic() - t0, spill_bytes=sp, vloopv2_hit=vl, compiler_summary=summary,
        microarch_profile=ma_prof, generated_cce=ldump, lowered_tir=ltir, artifact_id=recorded_aid, binary_paths=recorded_bins,
    )


def run_stage_b_ascendc_compile(
    cpp_source: str, *, arch: str = "dav-3510", compile_flags: Sequence[str] | str | None = None,
    bisheng_bin: str | Path | None = None, cann_root: str | Path | None = None,
    extra_include_dirs: Sequence[str | Path] | str | Path | None = None, timeout_sec: int = 120,
    enable_device_compile: bool = True, verify_shared_link: bool = False,
    emit_shared_so_cache_dir: str | Path | None = None, artifact_store: ArtifactStore | None = None,
    artifact_id: str = "", regime_tag: str = "", spec_key: str = "",
    subproc_run: Callable[..., Any] | None = None, disassembly_profile_fn: Callable[[str, str], dict[str, object] | None] | None = None,
    bisheng_resolver: Callable[..., str | None] | None = None,
    verify_instantiated_fn: Callable[..., tuple[bool, str]] | None = None,
    disassemble_fn: Callable[..., str] | None = None,
) -> StageBResult:
    """Verify dav-3510 C++20 translation unit compilation via bisheng in an isolated tempdir."""
    if not enable_device_compile:
        raise ValueError("enable_device_compile=False bypass has been removed")
    t0 = time.monotonic()
    if not isinstance(cpp_source, str) or not cpp_source.strip():
        return _stage_b_fail(t0, "stage_b_ascendc_empty_source", "STAGE_B_ASCENDC_EMPTY_SOURCE")
    resolve_bs = bisheng_resolver or _resolve_bisheng_bin
    if (resolved_bisheng := resolve_bs(bisheng_bin, cann_root=cann_root)) is None:
        return _stage_b_fail(t0, "stage_b_ascendc_bisheng_not_found", "BISHENG_NOT_FOUND", fault_plane="INFRA_TOOLCHAIN")
    run_fn = subproc_run or subprocess.run
    ctx = _prepare_ascendc_compile_context(compile_flags=compile_flags, cann_root=cann_root, resolved_bisheng=resolved_bisheng, extra_include_dirs=extra_include_dirs)
    extra_flags = ctx.extra_flags
    store = _resolve_artifact_store(artifact_store)
    spec_chk = CompileTaskSpec(artifact_id or "trackb", "_trackb_tu", (), (), tuple(extra_flags))
    with tempfile.TemporaryDirectory(prefix="tf_stage_b_ascendc_") as tmp_dir:
        src_path = Path(tmp_dir) / ("kernel.cpp" if ctx.has_explicit_x else "kernel.asc")
        obj_path, so_path = Path(tmp_dir) / "kernel.o", Path(tmp_dir) / "kernel.so"
        src_path.write_text(cpp_source, encoding="utf-8")
        should_link = verify_shared_link and 'extern "C"' in cpp_source and "<<<" in cpp_source and not ctx.has_explicit_x and bool(ctx.env.get("ASCEND_HOME_PATH"))
        cached_so = (Path(emit_shared_so_cache_dir).expanduser().resolve() / compute_trackb_shared_so_filename(cpp_source, extra_flags)) if (should_link and emit_shared_so_cache_dir is not None) else None
        cached_hit = cached_so is not None and is_verified_binary_file(cached_so)

        def _invoke(legacy: bool) -> Any:
            af = ["-x", "cce", f"--cce-aicore-arch={arch}"] if legacy else [f"--npu-arch={arch}"]
            df = [] if any("Wdeprecated-declarations" in str(f) for f in extra_flags) else ["-Wno-deprecated-declarations"]
            return run_fn([resolved_bisheng, *af, "--cce-aicore-only", "--cce-res-usage", "-O2", *ctx.std_flags, *ctx.include_flags, *df, *extra_flags, "-c", str(src_path), "-o", str(obj_path)], text=True, capture_output=True, cwd=tmp_dir, env=ctx.env, timeout=timeout_sec)

        link_exec, link_fut = None, None
        try:
            if should_link and not cached_hit:
                link_exec = ThreadPoolExecutor(max_workers=2)
                link_fut = link_exec.submit(run_fn, _build_trackb_shared_link_cmd(resolved_bisheng, arch=arch, ctx=ctx, src_path=src_path, so_path=so_path), text=True, capture_output=True, cwd=tmp_dir, env=ctx.env, timeout=timeout_sec)
            try:
                proc = _invoke(False)
                if proc.returncode != 0:
                    le = ((proc.stderr or "") + "\n" + (proc.stdout or "")).lower()
                    if "--npu-arch" in le and any(t in le for t in ("unknown", "unrecognized", "unsupported", "invalid")):
                        proc = _invoke(True)
            except subprocess.TimeoutExpired:
                return _stage_b_fail(t0, f"stage_b_ascendc_compile_timeout:{timeout_sec}s", "STAGE_B_ASCENDC_COMPILE_TIMEOUT")
            comb = (proc.stdout or "") + "\n" + (proc.stderr or "")
            if proc.returncode != 0 or not obj_path.is_file():
                sp, vl, summary = _parse_cce_res_usage_with_summary(comb, asm_or_dump_text=cpp_source)
                raw_d = proc.stderr or proc.stdout or "bisheng failed without diagnostics"
                return _stage_b_fail(
                    t0, f"stage_b_ascendc_compile_failed (rc={proc.returncode}): {_bounded_compile_diagnostic(raw_d)}",
                    _classify_stage_b_overflow_fault_code(raw_d, summary, "STAGE_B_ASCENDC_COMPILE_FAILED"),
                    spill_bytes=sp, vloopv2_hit=vl, summary=summary, ma_profile=(extract_microarch_profile_from_cce(cpp_source).to_dict() if cpp_source.strip() else None),
                    full_diag=raw_d, cce=cpp_source,
                )
            inst_ok, inst_err = (verify_instantiated_fn or _verify_object_has_instantiated_kernel)(obj_path, bisheng_bin=resolved_bisheng, subproc_run=run_fn)
            od_txt = (disassemble_fn or _disassemble_object)(obj_path, bisheng_bin=resolved_bisheng, subproc_run=run_fn)
            tel_out = _augment_bisheng_telemetry_text(comb, od_txt)
            dump_txt = f"{od_txt}\n{cpp_source}" if od_txt.strip() else cpp_source
            sp, vl, summary = _parse_cce_res_usage_with_summary(tel_out, asm_or_dump_text=dump_txt)
            ma_prof = disassembly_profile_fn(cpp_source, od_txt) if disassembly_profile_fn is not None else extract_microarch_profile_from_cce(cpp_source).to_dict()
            if not inst_ok:
                return _stage_b_fail(t0, inst_err, "STAGE_B_ASCENDC_UNINSTANTIATED_TEMPLATE", spill_bytes=sp, vloopv2_hit=vl, summary=summary, ma_profile=ma_prof, cce=cpp_source)
            if has_unreserved_vf_stack_overflow(spec_chk, summary.stack_bytes, tel_out):
                return _stage_b_fail(t0, f"stage_b_ascendc_vf_stack_overflow: stack_bytes={summary.stack_bytes}", "VF_STACK_OVERFLOW", spill_bytes=sp, vloopv2_hit=vl, summary=summary, ma_profile=ma_prof, full_diag=tel_out, cce=cpp_source)
            so_opt: Path | None = None
            if should_link:
                if cached_hit:
                    so_opt = cached_so
                elif link_fut is not None:
                    try:
                        lp = link_fut.result()
                    except subprocess.TimeoutExpired:
                        return _stage_b_fail(t0, f"stage_b_ascendc_shared_link_timeout:{timeout_sec}s", "STAGE_B_ASCENDC_SHARED_LINK_TIMEOUT", spill_bytes=sp, vloopv2_hit=vl, summary=summary, ma_profile=ma_prof)
                    lraw = (lp.stderr or "") + "\n" + (lp.stdout or "")
                    if lp.returncode != 0 or not so_path.is_file():
                        return _stage_b_fail(t0, f"stage_b_ascendc_shared_link_failed (rc={lp.returncode}): {_bounded_compile_diagnostic(lraw)}", _classify_stage_b_overflow_fault_code(lraw, summary, "STAGE_B_ASCENDC_SHARED_LINK_FAILED"), spill_bytes=sp, vloopv2_hit=vl, summary=summary, ma_profile=ma_prof, full_diag=lraw)
                    if cached_so is not None:
                        _atomic_copy_file(so_path, cached_so)
                    so_opt = so_path
        finally:
            if link_fut is not None:
                with contextlib.suppress(Exception):
                    link_fut.result()
            if link_exec is not None:
                link_exec.shutdown(wait=True)

        recorded_aid, recorded_bins = "", ()
        if built := [p for p in (obj_path, so_opt) if p is not None and p.is_file() and p.stat().st_size > 0]:
            closure_src = _normalize_trackb_unit_source(cpp_source, regime_tag=regime_tag)
            recorded_aid = artifact_id or compute_artifact_task_key(normalized_kernel_unit_src=closure_src, cann_target=arch, pass_configs=None, compile_flags=extra_flags, static_spec_signature=spec_key)
            rec = store.record_verified_artifact(recorded_aid, binary_paths=built, backend_kind="ascendc", arch=arch, flags=" ".join(extra_flags), spec_key=spec_key, closure_src=closure_src, metadata={"stack_bytes": int(summary.stack_bytes)}, copy_binaries=True)
            recorded_bins = rec.binary_paths
    return StageBResult(
        passed=True, elapsed_sec=time.monotonic() - t0, spill_bytes=sp, vloopv2_hit=vl,
        compiler_summary=summary, microarch_profile=ma_prof, generated_cce=cpp_source,
        artifact_id=recorded_aid, binary_paths=recorded_bins,
    )


def _load_manifest_dict(manifest_path: Path) -> dict[str, Any]:
    base: dict[str, Any] = {"version": 1, "regime_tag": "", "tasks": {}, "plan_index": {}}
    if not manifest_path.is_file():
        return base
    with contextlib.suppress(OSError, json.JSONDecodeError):
        if isinstance(raw := json.loads(manifest_path.read_text(encoding="utf-8")), Mapping):
            base["regime_tag"] = str(raw.get("regime_tag", ""))
            if isinstance(raw.get("tasks"), Mapping):
                base["tasks"] = dict(raw["tasks"])
            if isinstance(raw.get("plan_index"), Mapping):
                base["plan_index"] = dict(raw["plan_index"])
    return base


def validate_prewarm_manifest_v1(manifest_path: str | Path, plan: RegimeCompilePlan) -> tuple[bool, dict[str, Any]]:
    """Validate prewarm_manifest_v1.json against all CompileTaskSpecs in *plan*."""
    mpath = Path(manifest_path)
    manifest = _load_manifest_dict(mpath)
    tasks = manifest.get("tasks")
    if not mpath.is_file() or not isinstance(tasks, Mapping):
        return False, {"manifest_path": str(mpath), "regime_tag": plan.regime_tag, "total_tasks": len(plan.unique_specs_by_key), "cache_hits": 0, "missing_tasks": sorted(plan.unique_specs_by_key), "failed_tasks": [], "stale_tasks": []}
    missing, failed, stale, hits = [], [], [], 0
    for tkey, spec in plan.unique_specs_by_key.items():
        if not isinstance(entry := tasks.get(tkey), Mapping):
            missing.append(tkey)
        elif entry.get("passed") is False or str(entry.get("status", "")).upper() == "FAIL" or has_unreserved_vf_stack_overflow(spec, int(entry.get("stack_bytes", 0) or 0), ""):
            failed.append(tkey)
        else:
            raw_bins = ([ap] if (ap := str(entry.get("artifact_path", "") or "").strip()) else []) + (
                [str(b).strip() for b in entry["binary_paths"] if str(b).strip()] if isinstance(entry.get("binary_paths"), Sequence) and not isinstance(entry.get("binary_paths"), (str, bytes)) else []
            )
            if not raw_bins or not any(is_verified_binary_file(Path(b)) for b in raw_bins):
                missing.append(tkey)
                stale.append(tkey)
            else:
                hits += 1
    return (hits == len(plan.unique_specs_by_key) and not missing and not failed), {
        "manifest_path": str(mpath), "regime_tag": plan.regime_tag, "total_tasks": len(plan.unique_specs_by_key),
        "cache_hits": hits, "missing_tasks": missing, "failed_tasks": failed, "stale_tasks": stale,
    }


def _resolve_env_values(env_file: str | Path | None) -> dict[str, str]:
    eff = str(env_file) if env_file is not None else os.environ.get("TILEFOUNDRY_NPU_ENV_FILE", "").strip()
    return parse_simple_env_file(eff) if eff else {}


def _build_cpu_compile_env(env_vals: Mapping[str, str], tl_cache_dir: Path, source_root: Path | None = None) -> tuple[dict[str, str], Callable[[Path], list[str]]]:
    py = env_vals.get("TILEFOUNDRY_NPU_PYTHON") or os.environ.get("TILEFOUNDRY_NPU_PYTHON") or sys.executable
    cann_setup = (env_vals.get("TILEFOUNDRY_NPU_CANN_SETUP") or os.environ.get("TILEFOUNDRY_NPU_CANN_SETUP") or "").strip()
    tl_src = _resolve_tilelang_source_dir(env_vals=env_vals, source_root=source_root) or ""
    root = source_root or Path(__file__).resolve().parents[5]
    pp = [*( [tl_src] if tl_src else [] ), str(root), str(root / "src"), str(root / "targets" / "ascend"), *( [os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [] )]
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(pp), "TILELANG_DISABLE_CACHE": "0", "TILELANG_KERNEL_CACHE_USE_LIB_STAMP": "1", "TILELANG_CACHE_DIR": str(tl_cache_dir), "TORCH_DEVICE_BACKEND_AUTOLOAD": "0", "ASCEND_RT_VISIBLE_DEVICES": ""}
    env.setdefault("USE_CUDA", "0")
    need_src = bool(cann_setup and Path(cann_setup).is_file() and shutil.which("bisheng") is None)
    return env, (lambda sp: ["/bin/bash", "-c", f"set +u && source {shlex.quote(cann_setup)} >/dev/null 2>&1 && exec {shlex.quote(py)} {shlex.quote(str(sp))}"] if need_src else [py, str(sp)])


def _run_driver_spec(
    spec: CompileTaskSpec | None, module_source: str, work_dir: Path, *,
    resolved_tl_cache: Path, env_vals: Mapping[str, str], source_root: Path | None = None,
    timeout_sec: int = 120, fallback_cfg: tuple[str, Mapping[str, object], str | None, Mapping[str, object] | None, Sequence[str] | str | None, Mapping[str, object] | None] | None = None,
) -> tuple[bool, int, str, tuple[Path, ...], str]:
    w_cache = work_dir / "tl_cache"
    seed_tl_cache_tree(resolved_tl_cache, w_cache)
    pre_bins = snapshot_tl_cache_binaries(w_cache)
    run_env, build_argv = _build_cpu_compile_env(env_vals, w_cache, source_root)
    cand_path, drv_path, kdump = work_dir / "candidate_dsl.py", work_dir / "compile_driver.py", work_dir / "lowered_kernel.cce"
    emit_path = work_dir / "compiled_bins.json"
    cand_path.write_text(module_source, encoding="utf-8")
    if spec is not None:
        drv_src = build_stage_b_driver_source(
            cand_path, factory_name=spec.builder_name, factory_kwargs=dict(spec.static_kwargs),
            pass_configs=dict(spec.pass_configs), compile_flags=spec.compile_flags,
            enable_device_compile=True, kernel_dump_path=kdump, compile_for_cache=True, emit_cache_bins_path=emit_path,
        )
        fc_default, to_code = "STAGE_B_COMPILE_FAILED", "STAGE_B_COMPILE_TIMEOUT"
    else:
        assert fallback_cfg is not None
        bn, bkw, csn, pc, cf, dims = fallback_cfg
        drv_src = build_stage_b_driver_source(cand_path, factory_name=bn, factory_kwargs=bkw, config_selector_name=csn, pass_configs=pc, compile_flags=cf, dims=dims, enable_device_compile=True, kernel_dump_path=kdump)
        fc_default, to_code = "L1_COMPILE_FAILED", "L1_COMPILE_TIMEOUT"
    drv_path.write_text(drv_src, encoding="utf-8")
    try:
        proc = subprocess.run(build_argv(drv_path), text=True, capture_output=True, cwd=str(work_dir), env=run_env, timeout=timeout_sec)
    except subprocess.TimeoutExpired as exc:
        cap = "".join(p for p in (exc.stdout, exc.stderr) if isinstance(p, str))
        return False, 0, cap or f"compile_timeout:{timeout_sec}s", (), to_code
    ldump = kdump.read_text(encoding="utf-8", errors="ignore") if kdump.is_file() else ""
    comb = (proc.stdout or "") + "\n" + (proc.stderr or "")
    _, _, summary = _parse_cce_res_usage_with_summary(comb, asm_or_dump_text=ldump)
    s_bytes = int(summary.stack_bytes)
    if proc.returncode != 0:
        return False, s_bytes, comb, (), _classify_stage_b_overflow_fault_code(f"{comb}\n{ldump}", summary, fc_default)
    if spec is not None and has_unreserved_vf_stack_overflow(spec, s_bytes, comb):
        return False, s_bytes, comb, (), "VF_STACK_OVERFLOW"
    _merge_cache_locked(w_cache, resolved_tl_cache)
    return True, s_bytes, comb, tuple(_collect_built_bins(work_dir, w_cache, resolved_tl_cache, pre_bins, emit_path)), ""


def _is_trackb_spec(spec: CompileTaskSpec) -> bool:
    low = str(spec.builder_name or "").strip().lower()
    return low in ("_render_trackb_tu", "_trackb_tu") or "trackb" in low


def _resolve_trackb_spec_cpp_source(
    spec: CompileTaskSpec, module_source: str, *, env_vals: Mapping[str, str] | None = None, source_root: Path | None = None
) -> tuple[bool, str]:
    if not _is_trackb_spec(spec):
        return False, ""
    is_explicit = spec.builder_name in ("_render_trackb_tu", "_trackb_tu")
    if module_source.strip():
        loaded = _load_plan_module(module_source, env_vals=env_vals, source_root=source_root)
        if (cached_tu := loaded.trackb_tu_by_key.get(spec.task_key)) is not None:
            return True, cached_tu
        fn = getattr(loaded.mod, spec.builder_name, None)
        if fn is None and spec.builder_name == "_trackb_tu":
            fn = getattr(loaded.mod, "_render_trackb_tu", None)
        raw_fn = getattr(fn, "__wrapped__", fn) if callable(fn) else None
        if callable(raw_fn):
            res = raw_fn(**_filter_kwargs_for_callable(raw_fn, dict(spec.static_kwargs)))
            if isinstance(res, str):
                return True, _normalize_trackb_unit_source(res)
            if not is_explicit:
                return False, ""
        if is_explicit and (tb := extract_trackb_cpp_source(module_source, discover_module_trackb_symbols(module_source))):
            return True, _normalize_trackb_unit_source(tb[1])
    return is_explicit, ""


def _run_trackb_spec(
    spec: CompileTaskSpec, cpp_source: str, work_dir: Path, *,
    env_vals: Mapping[str, str], source_root: Path | None = None, timeout_sec: int = 120, regime_tag: str = "",
) -> tuple[bool, int, str, tuple[Path, ...], str]:
    cann_setup = (env_vals.get("TILEFOUNDRY_NPU_CANN_SETUP") or "").strip()
    cann_root = str(p.parent) if cann_setup and (p := Path(cann_setup).expanduser().resolve()).is_file() else None
    tl_dir = _resolve_tilelang_source_dir(env_vals=env_vals, source_root=source_root)
    extra_incs = [str(Path(tl_dir) / "src")] if tl_dir and (Path(tl_dir) / "src" / "tl_templates" / "ascend").is_dir() else None
    eff_flags = resolve_trackb_compile_flags(cpp_source, spec.compile_flags) if cpp_source.strip() else spec.compile_flags
    res = run_stage_b_ascendc_compile(
        cpp_source, compile_flags=eff_flags, cann_root=cann_root, extra_include_dirs=extra_incs, timeout_sec=timeout_sec,
        artifact_store=ArtifactStore(work_dir / "ascendc_store"), artifact_id=spec.task_key, regime_tag=regime_tag, spec_key=spec.task_key,
    )
    sb = int(res.compiler_summary.stack_bytes) if res.compiler_summary is not None else 0
    bins = tuple(Path(p) for p in res.binary_paths if is_verified_binary_file(Path(p)))
    diag = res.error_message or res.full_diagnostic or ""
    if not res.passed:
        fc = res.fault_code or _classify_stage_b_overflow_fault_code(diag, res.compiler_summary, "STAGE_B_ASCENDC_COMPILE_FAILED")
        return False, sb, diag, (), fc
    if has_unreserved_vf_stack_overflow(spec, sb, diag):
        return False, sb, f"stage_b_ascendc_vf_stack_overflow: stack_bytes={sb}", bins, "VF_STACK_OVERFLOW"
    if not bins:
        return False, sb, "stage_b_ascendc_missing_object: no verified binary artifact produced", (), "STAGE_B_ASCENDC_MISSING_OBJECT"
    return True, sb, diag, bins, ""


class StageBCompileGate:
    """Compile cache-missed CompileTaskSpecs on CPU and emit prewarm_manifest_v1.json."""

    def __init__(
        self, plan_or_store: RegimeCompilePlan | ArtifactStore, artifact_store: ArtifactStore | None = None, *,
        compile_fn: Callable[[CompileTaskSpec, str, Path], tuple[bool, int, str, Sequence[Path | str]]] | None = None,
        max_workers: int | None = None, timeout_sec: int = 120, env_file: str | Path | None = None,
        source_root: str | Path | None = None,
    ) -> None:
        if isinstance(plan_or_store, ArtifactStore):
            self._default_plan, self.artifact_store = None, plan_or_store
        else:
            if artifact_store is None:
                raise ValueError("artifact_store is required when plan is passed first")
            self._default_plan, self.artifact_store = plan_or_store, artifact_store
        self._compile_fn, self._max_workers, self._timeout_sec = compile_fn, max_workers, timeout_sec
        self._env_vals = _resolve_env_values(env_file)
        self._source_root = Path(source_root).resolve() if source_root is not None else None

    def run(self, module_source: str = "", *, plan: RegimeCompilePlan | None = None) -> StageBCompileReceipt:
        """Execute Stage-B CPU compilation for cache misses and write prewarm_manifest_v1.json."""
        t0 = time.monotonic()
        if (active_plan := plan if plan is not None else self._default_plan) is None:
            raise ValueError("RegimeCompilePlan must be provided to StageBCompileGate")
        tl_cache = self.artifact_store.root
        tl_cache.mkdir(parents=True, exist_ok=True)
        manifest_path = tl_cache / _PREWARM_MANIFEST_FILENAME
        existing = _load_manifest_dict(manifest_path)

        cached_res: dict[str, tuple[bool, int, str, tuple[Path, ...], str]] = {}
        missed: list[CompileTaskSpec] = []
        for tkey, spec in active_plan.unique_specs_by_key.items():
            if (rec := self.artifact_store.get_verified_artifact(tkey)) is not None:
                prev = existing["tasks"].get(tkey)
                sb = int(rec.metadata.get("stack_bytes", prev.get("stack_bytes", 0) if isinstance(prev, Mapping) else 0) or 0)
                bins = tuple(Path(p) for p in rec.binary_paths if is_verified_binary_file(Path(p)))
                cached_res[tkey] = (False, sb, f"stage_b_vf_stack_overflow: stack_bytes={sb}", bins, "VF_STACK_OVERFLOW") if has_unreserved_vf_stack_overflow(spec, sb, "") else (True, sb, "", bins, "")
            else:
                missed.append(spec)

        compiled_res: dict[str, tuple[bool, int, str, tuple[Path, ...], str]] = {}
        with contextlib.ExitStack() as stack:
            if missed:
                tmp_root = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="tf_stageb_gate_")))
                if module_source.strip() and any(_is_trackb_spec(s) for s in missed):
                    _load_plan_module(module_source, env_vals=self._env_vals, source_root=self._source_root)

                def _run_one(idx: int, spec: CompileTaskSpec) -> tuple[bool, int, str, tuple[Path, ...], str]:
                    (w_dir := tmp_root / f"spec_{idx}").mkdir(parents=True, exist_ok=True)
                    if self._compile_fn is not None:
                        ok, sb, diag, raw_bins = self._compile_fn(spec, module_source, w_dir)
                        bins = tuple(Path(p) for p in raw_bins if is_verified_binary_file(Path(p)))
                        if ok and has_unreserved_vf_stack_overflow(spec, sb, diag):
                            return False, int(sb), diag, bins, "VF_STACK_OVERFLOW"
                        fc = "" if ok else _classify_stage_b_overflow_fault_code(diag, BishengResourceSummary(stack_bytes=int(sb)), "STAGE_B_COMPILE_FAILED")
                        return bool(ok), int(sb), str(diag), bins, fc
                    try:
                        is_tb, cpp_src = _resolve_trackb_spec_cpp_source(
                            spec, module_source, env_vals=self._env_vals, source_root=self._source_root
                        )
                    except Exception as exc:
                        return False, 0, f"stage_b_ascendc_render_failed: {type(exc).__name__}: {exc}", (), "STAGE_B_ASCENDC_RENDER_FAILED"
                    if is_tb:
                        return _run_trackb_spec(
                            spec, cpp_src, w_dir, env_vals=self._env_vals, source_root=self._source_root,
                            timeout_sec=self._timeout_sec, regime_tag=active_plan.regime_tag,
                        )
                    return _run_driver_spec(spec, module_source, w_dir, resolved_tl_cache=tl_cache, env_vals=self._env_vals, source_root=self._source_root, timeout_sec=self._timeout_sec)

                with ThreadPoolExecutor(max_workers=self._max_workers or min(16, os.cpu_count() or 8, max(1, len(missed)))) as pool:
                    futs = {s.task_key: pool.submit(_run_one, i, s) for i, s in enumerate(missed)}
                    compiled_res = {tk: f.result() for tk, f in futs.items()}

            all_res = {**cached_res, **compiled_res}
            max_sb, first_fc, first_diag, all_ok = 0, "", "", True
            lock_path = tl_cache / ".prewarm_manifest_v1.lock"
            with lock_path.open("a+", encoding="utf-8") as fh:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                try:
                    latest = _load_manifest_dict(manifest_path)
                    mtasks, mplan = dict(latest["tasks"]), dict(latest["plan_index"])
                    for tkey, spec in active_plan.unique_specs_by_key.items():
                        ok, sb, diag, bins, fc = all_res[tkey]
                        max_sb = max(max_sb, int(sb))
                        if ok and has_unreserved_vf_stack_overflow(spec, sb, diag):
                            ok, fc = False, "VF_STACK_OVERFLOW"
                        rbins = bins
                        if ok and tkey in compiled_res and bins:
                            bk = "ascendc_trackb" if _is_trackb_spec(spec) else "tilelang"
                            rec = self.artifact_store.record_verified_artifact(tkey, binary_paths=bins, backend_kind=bk, flags=" ".join(spec.compile_flags), metadata={"stack_bytes": int(sb), "regime_tag": active_plan.regime_tag}, copy_binaries=True)
                            rbins = tuple(Path(p) for p in rec.binary_paths)
                        if not ok:
                            all_ok = False
                            if not first_fc:
                                first_fc, first_diag = fc or "STAGE_B_COMPILE_FAILED", diag
                        mtasks[tkey] = {
                            "builder_name": spec.builder_name, "artifact_path": str(rbins[0]) if rbins else "",
                            "binary_paths": [str(p) for p in rbins], "stack_bytes": int(sb),
                            "compile_flags": list(spec.compile_flags), "passed": bool(ok),
                            "status": "PASS" if ok else "FAIL", "fault_code": "" if ok else (fc or "STAGE_B_COMPILE_FAILED"),
                        }
                    mplan.update({str(cid): [s.task_key for s in cspecs] for cid, cspecs in active_plan.tasks_by_case_id.items()})
                    atomic_write_text(manifest_path, json.dumps({"version": 1, "regime_tag": active_plan.regime_tag, "tasks": mtasks, "plan_index": mplan}, indent=2, sort_keys=True) + "\n")
                finally:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        return StageBCompileReceipt(all_ok, str(manifest_path), max_sb, first_fc, _bounded_compile_diagnostic(first_diag) if first_diag else "", time.monotonic() - t0)

    compile_plan = run


def run_l1_compile_check(
    module_source: str, *, builder_name: str, builder_kwargs: Mapping[str, object],
    config_selector_name: str | None = None, pass_configs: Mapping[str, object] | None = None,
    compile_flags: Sequence[str] | str | None = None, dims: Mapping[str, object] | None = None,
    tail_anchor_dims: Mapping[str, object] | None = None, static_spec_dims_list: Sequence[Mapping[str, object]] | None = None,
    source_root: str | Path | None = None, env_file: str | Path | None = None, timeout_sec: int = 120,
    artifact_store: ArtifactStore | None = None, artifact_id: str = "", regime_tag: str = "", spec_key: str = "",
    tl_cache_dir: str | Path | None = None, anchor_case_specs: Sequence[tuple[int, str, Mapping[str, object]]] | None = None,
    operator_name: str = "", cases_yaml_sha256: str = "",
) -> StageBResult:
    """Run Tier L1 TileLang + bisheng device compilation on Host CPU via StageBCompileGate."""
    _ = (spec_key, operator_name, cases_yaml_sha256)
    t0 = time.monotonic()
    env_vals = _resolve_env_values(env_file)
    root = Path(source_root).resolve() if source_root is not None else Path(__file__).resolve().parents[5]
    store = artifact_store if artifact_store is not None else ArtifactStore(resolve_unified_tl_cache_dir(tl_cache_dir, env_vals=env_vals, source_root=root))
    store.root.mkdir(parents=True, exist_ok=True)
    cann_setup = (env_vals.get("TILEFOUNDRY_NPU_CANN_SETUP") or os.environ.get("TILEFOUNDRY_NPU_CANN_SETUP") or "").strip()
    tl_src = (env_vals.get("TILEFOUNDRY_NPU_TILELANG_SOURCE") or os.environ.get("TILEFOUNDRY_NPU_TILELANG_SOURCE") or "").strip()
    if not cann_setup and not tl_src:
        return run_stage_b_compile(
            module_source, factory_name=builder_name, factory_kwargs=builder_kwargs, config_selector_name=config_selector_name,
            pass_configs=pass_configs, compile_flags=compile_flags, dims=dims,
            python_bin=env_vals.get("TILEFOUNDRY_NPU_PYTHON") or os.environ.get("TILEFOUNDRY_NPU_PYTHON") or sys.executable,
            timeout_sec=timeout_sec, artifact_store=store, artifact_id=artifact_id, regime_tag=regime_tag,
        )

    cases_by_id: dict[str, dict[str, Any]] = {str(cn): dict(cd) for cn, _, cd in (anchor_case_specs or ())}
    if dims:
        cases_by_id.setdefault("primary", dict(dims))
    if tail_anchor_dims:
        cases_by_id["tail_anchor"] = dict(tail_anchor_dims)
    for idx, sdims in enumerate(static_spec_dims_list or ()):
        cases_by_id[f"static_{idx}"] = dict(sdims)
    if not cases_by_id:
        cases_by_id["primary"] = {}

    norm_pc, norm_cf = _merge_pass_configs_and_flags(None, None, pass_configs, compile_flags)
    plan = build_post_selector_regime_plan(
        module_source, regime_tag or "default", cases_by_id,
        ModuleSelectorPlanExtractor(builder_name=builder_name, config_selector_name=config_selector_name or "", default_builder_kwargs=builder_kwargs, default_pass_configs=norm_pc, default_compile_flags=norm_cf),
        env_file=env_file, source_root=root,
    )
    if not plan.unique_specs_by_key:
        with tempfile.TemporaryDirectory(prefix="tf_l1_fallback_") as tmp_dir:
            ok, sb, diag, _, fc = _run_driver_spec(
                None, module_source, Path(tmp_dir), resolved_tl_cache=store.root, env_vals=env_vals, source_root=root,
                timeout_sec=timeout_sec, fallback_cfg=(builder_name, builder_kwargs, config_selector_name, pass_configs, compile_flags, dims),
            )
            if not ok:
                return _stage_b_fail(t0, f"l1_compile_failed: {_bounded_compile_diagnostic(diag)}", fc or "L1_COMPILE_FAILED", summary=BishengResourceSummary(stack_bytes=sb), full_diag=diag)
        return StageBResult(passed=True, elapsed_sec=time.monotonic() - t0)

    receipt = StageBCompileGate(plan, store, timeout_sec=timeout_sec, env_file=env_file, source_root=root).run(module_source)
    if not receipt.passed:
        pfx = "l1_vf_stack_overflow" if receipt.fault_code == "VF_STACK_OVERFLOW" else "l1_compile_failed"
        return _stage_b_fail(t0, f"{pfx}: {receipt.diagnostic}", receipt.fault_code or "L1_COMPILE_FAILED", summary=BishengResourceSummary(stack_bytes=receipt.max_stack_bytes), full_diag=receipt.diagnostic)
    first_key = next(iter(plan.unique_specs_by_key))
    art = store.get_verified_artifact(first_key)
    return StageBResult(passed=True, elapsed_sec=receipt.elapsed_sec, compiler_summary=BishengResourceSummary(stack_bytes=receipt.max_stack_bytes), artifact_id=art.artifact_id if art else first_key, binary_paths=art.binary_paths if art else ())


__all__ = [
    "ModuleSelectorPlanExtractor", "StageBCompileGate", "StageBResult", "_resolve_artifact_store",
    "ascend_compile_session", "build_post_selector_regime_plan", "build_stage_b_driver_source",
    "compile_trackb_shared_so", "compute_compile_task_key", "compute_trackb_shared_so_filename",
    "has_unreserved_vf_stack_overflow", "parse_cce_res_usage", "resolve_trackb_compile_flags",
    "run_l1_compile_check", "run_stage_b_ascendc_compile", "run_stage_b_compile", "validate_prewarm_manifest_v1",
]
