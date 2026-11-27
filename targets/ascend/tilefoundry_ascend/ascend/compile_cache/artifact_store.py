"""Closure-addressed source slicing and binary artifact verification store."""

from __future__ import annotations

import ast
import contextlib
import errno
import fcntl
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol

UnitKind = Literal["tilelang_dsl", "cce_cpp20"]
VALID_UNIT_KINDS: tuple[UnitKind, ...] = ("tilelang_dsl", "cce_cpp20")


class CompileTaskSpecLike(Protocol):
    """L0 structural port matching L1 CompileTaskSpec specialization fields."""

    @property
    def builder_name(self) -> str: ...
    @property
    def static_kwargs(self) -> tuple[tuple[str, object], ...]: ...
    @property
    def pass_configs(self) -> tuple[tuple[str, object], ...]: ...
    @property
    def compile_flags(self) -> tuple[str, ...]: ...
    @property
    def unit_kind(self) -> UnitKind: ...
    @property
    def unit_source(self) -> str: ...

from tilefoundry_ascend.ascend.compile_cache.cpp_closure import (
    _split_cpp_units, extract_reachable_trackb_source, select_active_trackb_kernel, strip_cpp_comments_and_strings,
)
from tilefoundry_ascend.ascend.json_record import JsonRecordMixin, atomic_write_text

_CPP_MARKERS = ("__global__", "__aicore__", "__ubuf__", "__simd_vf__", "__simt_vf__", "#include", 'extern "C"')
_BINARY_SUFFIXES = frozenset({".so", ".o", ".bin"})
_CACHE_INTERNAL_FILES = frozenset({
    "prewarm_manifest_v1.json", ".prewarm_manifest_v1.lock",
    "artifact_manifest_v1.json", "artifact_index_v1.jsonl", ".artifact_store_v1.lock",
})
_SKIPPED_TL_CACHE_DIRS = frozenset({"artifact_blobs", "stage_b_diag_v1"})


def is_verified_binary_file(path: Path) -> bool:
    """Return True iff *path* is an existing non-empty compiled binary file."""
    try:
        return (
            path.is_file()
            and (path.suffix in _BINARY_SUFFIXES or ".so" in path.suffixes)
            and path.stat().st_size > 0
        )
    except OSError:
        return False


def _is_verified_binary_file(path: Path) -> bool:
    return is_verified_binary_file(path)


def _is_cache_internal_file(item: Path) -> bool:
    name = item.name
    return name in _CACHE_INTERNAL_FILES or name.endswith(".lock") or name.startswith((".prewarm_manifest_v1.", ".artifact_manifest."))


def _walk_tl_cache_tree(base_dir: Path) -> Iterator[tuple[Path, list[str], list[str]]]:
    for root_str, dirnames, filenames in os.walk(base_dir):
        dirnames[:] = [d for d in sorted(dirnames) if d not in _SKIPPED_TL_CACHE_DIRS]
        yield Path(root_str), dirnames, sorted(filenames)


def _mirror_walk_dirs(src_dir: Path, dst_dir: Path, root: Path, dirnames: list[str]) -> Path:
    rel_root = root.relative_to(src_dir)
    dst_root = dst_dir if rel_root == Path(".") else dst_dir / rel_root
    dst_root.mkdir(parents=True, exist_ok=True)
    for dname in dirnames:
        (dst_root / dname).mkdir(parents=True, exist_ok=True)
    return dst_root


def seed_tl_cache_tree(src_dir: Path, dst_dir: Path) -> None:
    """Seed an isolated worker cache from *src_dir* via hardlinks or copy fallback."""
    if not src_dir.is_dir():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    for root, dirnames, filenames in _walk_tl_cache_tree(src_dir):
        dst_root = _mirror_walk_dirs(src_dir, dst_dir, root, dirnames)
        for fname in filenames:
            if fname == "compiled_entries.jsonl":
                continue
            item, target = root / fname, dst_root / fname
            if _is_cache_internal_file(item) or target.exists():
                continue
            try:
                os.link(item, target)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    shutil.copy2(item, target)


def _atomic_copy_into_cache(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(dst.parent), prefix=f".{dst.name}.")
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        shutil.copy2(src, tmp_path)
        os.replace(tmp_path, dst)
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def merge_tl_cache_tree(src_dir: Path, dst_dir: Path) -> None:
    """Merge an isolated worker's TileLang cache outputs into *dst_dir*."""
    if not src_dir.is_dir():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    for root, dirnames, filenames in _walk_tl_cache_tree(src_dir):
        dst_root = _mirror_walk_dirs(src_dir, dst_dir, root, dirnames)
        for fname in filenames:
            item, target = root / fname, dst_root / fname
            if _is_cache_internal_file(item):
                continue
            if fname == "compiled_entries.jsonl" and target.is_file():
                existing, addition = target.read_text(encoding="utf-8"), item.read_text(encoding="utf-8")
                if not addition or addition == existing:
                    continue
                if existing and not existing.endswith("\n"):
                    existing += "\n"
                atomic_write_text(target, existing + addition)
            elif not target.exists():
                _atomic_copy_into_cache(item, target)


def snapshot_tl_cache_binaries(cache_dir: Path) -> dict[Path, tuple[int, int]]:
    """Snapshot relative binary paths to (size, mtime_ns) inside *cache_dir*."""
    if not cache_dir.is_dir():
        return {}
    return {
        (root / fn).relative_to(cache_dir): ((st := (root / fn).stat()).st_size, st.st_mtime_ns)
        for root, _, filenames in _walk_tl_cache_tree(cache_dir)
        for fn in filenames
        if is_verified_binary_file(root / fn)
    }


def collect_new_tl_cache_binaries(cache_dir: Path, pre_snapshot: Mapping[Path, tuple[int, int]]) -> tuple[Path, ...]:
    """Return relative paths of new or updated binaries in *cache_dir*."""
    return tuple(rel for rel, sig in snapshot_tl_cache_binaries(cache_dir).items() if pre_snapshot.get(rel) != sig)


def _extract_cpp_string_literal(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str) and any(t in node.value for t in _CPP_MARKERS):
        return node.value
    return None


def _prune_unused_cpp_globals(cpp_tu: str) -> str:
    directives, units = _split_cpp_units(cpp_tu)
    if not units:
        return cpp_tu.strip() + ("\n" if cpp_tu.strip() else "")
    masked = strip_cpp_comments_and_strings("\n".join(u.text for u in units if u.kind != "global_decl"))
    kept = ["\n".join(directives)] if directives else []
    kept_units = [u for u in units if u.kind != "global_decl" or not u.name or re.search(rf"\b{re.escape(u.name)}\b", masked)]
    kept.extend(text for u in sorted(kept_units, key=lambda u: u.start_char) if (text := u.text.strip()))
    return "\n\n".join(kept) + ("\n" if kept else "")


def _extract_cpp_closure(cpp_source: str, *, module_source: str, regime_tag: str, active_kernel: str = "") -> str:
    target_tag = regime_tag
    _, units = _split_cpp_units(cpp_source)
    if active_kernel and any(u.kind == "kernel" and u.name == active_kernel for u in units):
        target_tag = regime_tag or active_kernel
    elif select_active_trackb_kernel(cpp_source, active_kernel=active_kernel or None, regime_tag=regime_tag, module_source=module_source) is None and regime_tag:
        tag_parts = [p for p in re.split(r"[^a-z0-9]+", regime_tag.lower()) if len(p) >= 2]
        kernels = [u for u in units if u.kind == "kernel" and u.name]
        best = max(kernels, key=lambda u: (sum(1 for p in tag_parts if p in u.name.lower()), u.line_count), default=None)
        if best is not None and any(p in best.name.lower() for p in tag_parts):
            target_tag = best.name
    _, extracted = extract_reachable_trackb_source(cpp_source, module_source=module_source, regime_tag=target_tag)
    return _prune_unused_cpp_globals(extracted)


def _score_name_for_regime(name: str, regime_tag: str) -> int:
    tag = regime_tag.strip().lower()
    if not tag:
        return 0
    low = name.lower()
    if tag in low:
        return 100 + len(tag)
    tokens = [tok for tok in re.split(r"[^a-z0-9]+", tag) if len(tok) >= 2]
    name_words = set(re.split(r"[^a-z0-9]+", low))
    return sum(10 + len(t) for t in tokens if len(t) >= 3 and t in low) + sum(3 for t in tokens if t in name_words)


def _collect_ast_names(node: ast.AST) -> set[str]:
    return {sub.id for sub in ast.walk(node) if isinstance(sub, ast.Name)}


def _node_source_slice(lines: Sequence[str], node: ast.stmt) -> str:
    start = min((d.lineno for d in getattr(node, "decorator_list", ())), default=node.lineno)
    return "".join(lines[start - 1 : (node.end_lineno or node.lineno)]).rstrip()


def _is_regime_candidate_fn(name: str, seg: str) -> bool:
    return bool(
        "@T.prim_func" in seg
        or re.match(r"^_?(?:select|build)_.*(?:_config|_func|_callable|_kernel)$", name)
        or re.match(r"^_?(?:kernel_[a-z0-9_]+|[a-z0-9_]+_kernel)$", name)
    )


def _resolve_branch_for_regime(fnode: ast.FunctionDef | ast.AsyncFunctionDef, regime_tag: str, lines: Sequence[str]) -> str | None:
    if not (tag := regime_tag.strip().lower()):
        return None
    for stmt in fnode.body:
        cur: ast.If | None = stmt if isinstance(stmt, ast.If) else None
        while cur is not None:
            body_seg = "\n".join(_node_source_slice(lines, s) for s in cur.body)
            if tag in ast.unparse(cur.test).lower() or tag in body_seg.lower():
                return f"def {fnode.name}({ast.unparse(fnode.args)}):\n{body_seg}"
            cur = cur.orelse[0] if len(cur.orelse) == 1 and isinstance(cur.orelse[0], ast.If) else None
    return None


def extract_reachable_closure_source(module_source: str, *, active_symbols: Sequence[str] = (), regime_tag: str = "") -> str:
    """Extract normalized reachable Python and Track-B C++ closure for one regime."""
    if not module_source.strip():
        return ""
    try:
        tree = ast.parse(module_source)
    except SyntaxError:
        return module_source.strip()

    lines = module_source.splitlines(keepends=True)
    imports: list[tuple[set[str], str]] = []
    funcs, classes, assigns, cpp_constants, ordered_defs = {}, {}, {}, {}, []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imports.append(({a.asname or a.name.split(".")[0] for a in node.names}, _node_source_slice(lines, node)))
        elif isinstance(node, ast.ImportFrom):
            imports.append(({a.asname or a.name for a in node.names}, _node_source_slice(lines, node)))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs[node.name] = node; ordered_defs.append(("func", node.name))
        elif isinstance(node, ast.ClassDef):
            classes[node.name] = node; ordered_defs.append(("class", node.name))
        elif (isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)) or (
            isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        ):
            name = (node.targets[0] if isinstance(node, ast.Assign) else node.target).id
            assigns[name] = node; ordered_defs.append(("assign", name))
            if (cpp_val := _extract_cpp_string_literal(node.value)) is not None:
                cpp_constants[name] = cpp_val

    all_defs: dict[str, ast.stmt] = {**assigns, **classes, **funcs}
    fn_segments = {name: _node_source_slice(lines, node) for name, node in funcs.items()}
    regime_entry_fns = {n for n, seg in fn_segments.items() if _is_regime_candidate_fn(n, seg)}
    seeds, excluded_siblings, active_cpp_kernel = [s for s in active_symbols if s in all_defs], set(), ""

    if not seeds and regime_tag:
        selectors, builders, other = [], [], []
        for name in funcs:
            if (score := _score_name_for_regime(name, regime_tag)) <= 0:
                continue
            bucket = selectors if name.startswith(("_select_", "select_")) else builders if name in regime_entry_fns else other
            bucket.append((score, name))
        for bucket in (selectors, builders):
            if bucket:
                top = max(s for s, _ in bucket)
                seeds.extend(n for s, n in bucket if s == top)
        if not seeds and other:
            top = max(s for s, _ in other)
            seeds.extend(n for s, n in other if s == top)
        if not builders:
            matched_cpp = []
            for cname, cpp_txt in cpp_constants.items():
                cscore = _score_name_for_regime(cname, regime_tag)
                kname = select_active_trackb_kernel(cpp_txt, regime_tag=regime_tag, module_source=module_source) or ""
                if (best_score := max(cscore, _score_name_for_regime(kname, regime_tag))) > 0:
                    matched_cpp.append((best_score, cname, kname))
            if matched_cpp:
                matched_cpp.sort(key=lambda r: r[0], reverse=True)
                for cscore, cname, kname in matched_cpp:
                    if cscore == matched_cpp[0][0]:
                        if cname not in seeds:
                            seeds.append(cname)
                        active_cpp_kernel = active_cpp_kernel or kname
                excluded_siblings.update(c for c in cpp_constants if c not in seeds)
        if seeds:
            excluded_siblings.update(r for r in regime_entry_fns if r not in seeds)
        else:
            for fname, fnode in funcs.items():
                if (branch := _resolve_branch_for_regime(fnode, regime_tag, lines)) is not None:
                    if bdefs := [n for n in _collect_ast_names(ast.parse(branch)) if n in all_defs and n != fname]:
                        seeds.extend(bdefs)
                        break
        if not seeds and cpp_constants:
            seeds.extend(cpp_constants.keys())

    if not seeds:
        for fname in ("kernel", "_get_kernel", "_render_trackb_tu", *funcs, *cpp_constants):
            if fname in all_defs and fname not in seeds and (
                fname in ("kernel", "_get_kernel", "_render_trackb_tu") or fname in regime_entry_fns or fname.startswith("_render_") or fname in cpp_constants
            ):
                seeds.append(fname)
    if not seeds:
        return module_source.strip()
    if "_get_kernel" in funcs and "_get_kernel" not in seeds and not active_symbols:
        if any(s in funcs and ("prim_func" in fn_segments.get(s, "") or s.startswith("_build_")) for s in seeds):
            seeds.append("_get_kernel")

    visited: set[str] = set()
    queue = list(seeds)
    while queue:
        if (sym := queue.pop(0)) in visited or sym not in all_defs:
            continue
        visited.add(sym)
        queue.extend(r for r in sorted(_collect_ast_names(all_defs[sym])) if r in all_defs and r not in visited and r not in excluded_siblings)

    seed_cpp = set(seeds) & set(cpp_constants)
    for fn_name, fnode in funcs.items():
        if fn_name not in visited and fn_name not in excluded_siblings and any(t in fn_segments[fn_name] for t in ("compile_ascend", "_render_trackb_tu")):
            if (refs := _collect_ast_names(fnode)) & seed_cpp:
                visited.add(fn_name)
                visited.update(r for r in sorted(refs) if r in all_defs and r not in excluded_siblings)

    used_names = set(visited).union(*(_collect_ast_names(all_defs[s]) for s in visited))
    parts = [imp_src for bound, imp_src in imports if bound & used_names or "*" in bound]
    for kind, name in ordered_defs:
        if name not in visited:
            continue
        if kind == "assign" and name in cpp_constants:
            sliced = _extract_cpp_closure(cpp_constants[name], module_source=module_source, regime_tag=regime_tag, active_kernel=active_cpp_kernel)
            parts.append(f'{name} = r"""\n{sliced}"""')
        else:
            parts.append(_node_source_slice(lines, all_defs[name]))
    return "\n\n".join(part for part in parts if part.strip()).strip()


def _canonical_json_value(val: object) -> object:
    if isinstance(val, Mapping):
        return {str(k): _canonical_json_value(v) for k, v in sorted(val.items(), key=lambda kv: str(kv[0]))}
    if isinstance(val, (list, tuple)):
        return [_canonical_json_value(item) for item in val]
    if isinstance(val, (set, frozenset)):
        return sorted((_canonical_json_value(i) for i in val), key=lambda x: json.dumps(x, sort_keys=True, separators=(",", ":")))
    return val if isinstance(val, (str, int, float, bool)) or val is None else str(val)


def _canonical_kv_map(
    raw: Mapping[str, object] | Sequence[tuple[str, object]] | None,
) -> dict[str, object]:
    if raw is None:
        return {}
    items = raw.items() if isinstance(raw, Mapping) else raw
    return {str(k): _canonical_json_value(v) for k, v in sorted(items, key=lambda kv: str(kv[0]))}


def compute_compile_task_key(
    spec: CompileTaskSpecLike | None = None,
    *,
    normalized_kernel_unit_src: str = "",
    cann_target: str = "dav-3510",
    pass_configs: Mapping[str, object] | Sequence[tuple[str, object]] | None = None,
    compile_flags: Sequence[str] = (),
    static_spec_signature: str = "",
    compiler_stamp: str = "",
    unit_kind: UnitKind | None = None,
    builder_name: str = "",
    static_kwargs: Mapping[str, object] | Sequence[tuple[str, object]] | None = None,
) -> str:
    """Compute canonical SHA256 task key shared across Stage-B, Prewarm, and L1..L4."""
    eff_unit_kind = (
        unit_kind
        if unit_kind is not None
        else (spec.unit_kind if spec is not None else "tilelang_dsl")
    )
    if eff_unit_kind not in VALID_UNIT_KINDS:
        raise ValueError(
            f"Unsupported unit_kind {eff_unit_kind!r}; expected one of {VALID_UNIT_KINDS}"
        )
    eff_src = (
        normalized_kernel_unit_src
        if normalized_kernel_unit_src
        else (spec.unit_source if spec is not None else "")
    )
    eff_builder = builder_name if builder_name else (spec.builder_name if spec is not None else "")
    eff_static_kwargs = (
        static_kwargs
        if static_kwargs is not None
        else (spec.static_kwargs if spec is not None else None)
    )
    eff_pass_configs = (
        pass_configs if pass_configs is not None else (spec.pass_configs if spec is not None else None)
    )
    eff_flags = compile_flags if compile_flags else (spec.compile_flags if spec is not None else ())
    payload = json.dumps(
        {
            "builder_name": str(eff_builder),
            "cann_target": str(cann_target),
            "compile_flags": sorted({str(f).strip() for f in eff_flags if str(f).strip()}),
            "compiler_stamp": str(compiler_stamp),
            "normalized_kernel_unit_src": str(eff_src),
            "pass_configs": _canonical_kv_map(eff_pass_configs),
            "static_kwargs": _canonical_kv_map(eff_static_kwargs),
            "static_spec_signature": str(static_spec_signature),
            "unit_kind": str(eff_unit_kind),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolve_unified_tl_cache_dir(
    tl_cache_dir: str | Path | None = None, *, env_vals: Mapping[str, str] | None = None, source_root: Path | None = None
) -> Path:
    """Resolve the canonical unified TileLang and Track-B ArtifactStore cache directory."""
    _ = source_root
    if tl_cache_dir is not None and str(tl_cache_dir).strip():
        return Path(tl_cache_dir).expanduser().resolve()
    for env_map in ((env_vals, os.environ) if env_vals is not None else (os.environ,)):
        for key in ("TILELANG_CACHE_DIR", "TF_SURROGATE_TL_CACHE_DIR", "TILEFOUNDRY_NPU_ARTIFACT_STORE"):
            if (val := str(env_map.get(key, "") or "").strip()):
                return Path(val).expanduser().resolve()
        if (val := str(env_map.get("TILEFOUNDRY_NPU_WORK_ROOT", "") or "").strip()):
            return Path(val).expanduser().resolve() / "surrogate" / "tilelang"
    uid = os.getuid() if hasattr(os, "getuid") else 0
    return (Path(tempfile.gettempdir()) / f"tf_surrogate_tl_cache_{uid}").resolve()


def compute_artifact_id(
    *,
    closure_src: str,
    flags: str = "",
    arch: str = "dav-3510",
    cann_series: str = "cann_9_3",
    spec_key: str = "",
    compiler_stamp: str = "",
    unit_kind: UnitKind = "tilelang_dsl",
) -> str:
    """Compute SHA256 hex digest identifying a compiled kernel artifact."""
    target_tag = f"{arch}:{cann_series}" if cann_series and cann_series != "cann_9_3" else str(arch)
    return compute_compile_task_key(
        normalized_kernel_unit_src=str(closure_src),
        cann_target=target_tag,
        compile_flags=tuple(tok for tok in str(flags).split() if tok),
        static_spec_signature=str(spec_key),
        compiler_stamp=str(compiler_stamp),
        unit_kind=unit_kind,
    )


@dataclass(frozen=True)
class ArtifactRecord(JsonRecordMixin):
    """Persisted metadata for one verified compiled kernel artifact."""

    artifact_id: str
    backend_kind: str = "tilelang"
    binary_paths: tuple[str, ...] = ()
    arch: str = "dav-3510"
    cann_series: str = "cann_9_3"
    flags: str = ""
    spec_key: str = ""
    compiler_stamp: str = ""
    closure_sha256: str = ""
    metadata: dict[str, object] = field(default_factory=dict)
    recorded_at: float = 0.0

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ArtifactRecord:
        """Reconstruct ArtifactRecord from a JSON mapping."""
        raw_bins, raw_meta = payload.get("binary_paths"), payload.get("metadata")
        norm = dict(payload)
        norm.setdefault("artifact_id", "")
        norm["binary_paths"] = tuple(str(p) for p in raw_bins if str(p).strip()) if isinstance(raw_bins, Sequence) and not isinstance(raw_bins, (str, bytes)) else ()
        norm["metadata"] = dict(raw_meta) if isinstance(raw_meta, Mapping) else {}
        norm["recorded_at"] = float(payload.get("recorded_at", 0.0) or 0.0)
        return super().from_dict(norm)


class ArtifactStore:
    """Disk-backed store of closure-addressed verified TileLang and Track-B binaries."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.manifest_path = self.root / "artifact_manifest_v1.json"
        self.index_path = self.root / "artifact_index_v1.jsonl"
        self.blobs_dir = self.root / "artifact_blobs"
        self._lock_path = self.root / ".artifact_store_v1.lock"

    def _load_records_unlocked(self) -> dict[str, dict[str, object]]:
        records: dict[str, dict[str, object]] = {}
        if self.manifest_path.is_file():
            with contextlib.suppress(OSError, UnicodeDecodeError, json.JSONDecodeError):
                raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                if isinstance(raw, Mapping) and isinstance(entries := raw.get("artifacts"), Mapping):
                    records = {str(k): dict(v) for k, v in entries.items() if isinstance(v, Mapping)}
        if not records and self.index_path.is_file():
            with contextlib.suppress(OSError, UnicodeDecodeError, json.JSONDecodeError):
                for line in self.index_path.read_text(encoding="utf-8").splitlines():
                    if line.strip() and isinstance(row := json.loads(line), Mapping) and row.get("artifact_id"):
                        records[str(row["artifact_id"])] = dict(row)
        return records

    def _save_manifest_unlocked(self, records: Mapping[str, Mapping[str, object]]) -> None:
        atomic_write_text(self.manifest_path, json.dumps({"version": 1, "artifacts": dict(records)}, indent=2, sort_keys=True) + "\n")

    def record_verified_artifact(
        self, artifact_id: str, *, binary_paths: Sequence[str | Path] = (), backend_kind: str = "tilelang",
        arch: str = "dav-3510", cann_series: str = "cann_9_3", flags: str = "", spec_key: str = "",
        compiler_stamp: str = "", closure_src: str = "", metadata: Mapping[str, object] | None = None, copy_binaries: bool = False,
    ) -> ArtifactRecord:
        """Persist a verified artifact record and optionally copy binary files into blobs."""
        if not (aid := str(artifact_id or "").strip()):
            raise ValueError("artifact_id must not be empty")
        self.root.mkdir(parents=True, exist_ok=True)
        resolved_bins: list[str] = []
        for raw_p in binary_paths:
            if not _is_verified_binary_file(src_p := Path(raw_p).resolve()):
                continue
            if copy_binaries:
                dest_p = self.blobs_dir / aid / src_p.name
                if src_p != dest_p:
                    _atomic_copy_into_cache(src_p, dest_p)
                resolved_bins.append(str(dest_p))
            else:
                resolved_bins.append(str(src_p))
        record = ArtifactRecord(
            artifact_id=aid, backend_kind=str(backend_kind), binary_paths=tuple(dict.fromkeys(resolved_bins)),
            arch=str(arch), cann_series=str(cann_series), flags=str(flags), spec_key=str(spec_key), compiler_stamp=str(compiler_stamp),
            closure_sha256=hashlib.sha256(closure_src.encode("utf-8")).hexdigest() if closure_src else "",
            metadata=dict(metadata) if metadata is not None else {}, recorded_at=time.time(),
        )
        with self._lock_path.open("a+", encoding="utf-8") as lock_fh:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            try:
                records = self._load_records_unlocked()
                records[aid] = (row_dict := record.to_dict())
                self._save_manifest_unlocked(records)
                with self.index_path.open("a", encoding="utf-8") as idx_fh:
                    idx_fh.write(json.dumps(row_dict, sort_keys=True) + "\n")
                    idx_fh.flush()
            finally:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
        return record

    def get_verified_artifact(self, artifact_id: str) -> ArtifactRecord | None:
        """Return verified ArtifactRecord iff all recorded binary files exist with size > 0."""
        aid = str(artifact_id or "").strip()
        raw = self._load_records_unlocked().get(aid) if aid else None
        if not isinstance(raw, Mapping):
            return None
        record = ArtifactRecord.from_dict(raw)
        if not record.binary_paths or any(not _is_verified_binary_file(Path(p)) for p in record.binary_paths):
            return None
        return record

    def has_verified_artifact(self, artifact_id: str) -> bool:
        """Return True iff artifact_id has on-disk verified binary files with st_size > 0."""
        return self.get_verified_artifact(artifact_id) is not None
