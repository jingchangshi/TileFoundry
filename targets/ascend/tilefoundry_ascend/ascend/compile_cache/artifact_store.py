"""Closure-addressed source slicing and binary artifact verification store."""

from __future__ import annotations

import ast
import errno
import fcntl
import hashlib
import json
import os
import re
import shutil
import tempfile
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from tilefoundry_ascend.ascend.compile_cache.cpp_closure import (
    _split_cpp_units,
    extract_reachable_trackb_source,
    select_active_trackb_kernel,
    strip_cpp_comments_and_strings,
)

_CPP_MARKERS: tuple[str, ...] = (
    "__global__",
    "__aicore__",
    "__ubuf__",
    "__simd_vf__",
    "__simt_vf__",
    "#include",
    'extern "C"',
)
_BINARY_SUFFIXES: frozenset[str] = frozenset({".so", ".o", ".bin"})


def is_verified_binary_file(path: Path) -> bool:
    """Return True iff *path* is an existing non-empty compiled binary file."""
    try:
        if not path.is_file():
            return False
        if path.suffix not in _BINARY_SUFFIXES and ".so" not in path.suffixes:
            return False
        return path.stat().st_size > 0
    except OSError:
        return False


def _is_verified_binary_file(path: Path) -> bool:
    return is_verified_binary_file(path)


_CACHE_INTERNAL_FILES: frozenset[str] = frozenset(
    {
        "prewarm_manifest_v1.json",
        ".prewarm_manifest_v1.lock",
        "artifact_manifest_v1.json",
        "artifact_index_v1.jsonl",
        ".artifact_store_v1.lock",
    }
)


def _is_cache_internal_file(item: Path) -> bool:
    name = item.name
    if name in _CACHE_INTERNAL_FILES:
        return True
    return name.startswith(".prewarm_manifest_v1.") or name.startswith(".artifact_manifest.")


def seed_tl_cache_tree(src_dir: Path, dst_dir: Path) -> None:
    """Seed an isolated worker cache from *src_dir* via hardlinks or copy fallback."""
    if not src_dir.is_dir():
        return
    dst_dir.mkdir(parents=True, exist_ok=True)
    for item in sorted(src_dir.rglob("*")):
        rel = item.relative_to(src_dir)
        if _is_cache_internal_file(item):
            continue
        if item.name == "compiled_entries.jsonl":
            continue
        if rel.parts and rel.parts[0] == "artifact_blobs":
            continue
        target = dst_dir / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(item, target)
        except OSError as exc:
            if exc.errno == errno.EEXIST:
                continue
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
    for item in sorted(src_dir.rglob("*")):
        if _is_cache_internal_file(item):
            continue
        rel = item.relative_to(src_dir)
        target = dst_dir / rel
        if item.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        if item.name == "compiled_entries.jsonl" and target.is_file():
            existing = target.read_text(encoding="utf-8")
            addition = item.read_text(encoding="utf-8")
            if existing and not existing.endswith("\n"):
                existing += "\n"
            fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), prefix=".compiled_entries.")
            tmp_path = Path(tmp_name)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    fh.write(existing + addition)
                    fh.flush()
                    os.fsync(fh.fileno())
                os.replace(tmp_path, target)
            finally:
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
        elif not target.exists():
            _atomic_copy_into_cache(item, target)


def snapshot_tl_cache_binaries(cache_dir: Path) -> dict[Path, tuple[int, int]]:
    """Snapshot relative binary paths to (size, mtime_ns) inside *cache_dir*."""
    snap: dict[Path, tuple[int, int]] = {}
    if not cache_dir.is_dir():
        return snap
    for item in cache_dir.rglob("*"):
        if is_verified_binary_file(item):
            st = item.stat()
            snap[item.relative_to(cache_dir)] = (st.st_size, st.st_mtime_ns)
    return snap


def collect_new_tl_cache_binaries(
    cache_dir: Path,
    pre_snapshot: Mapping[Path, tuple[int, int]],
) -> tuple[Path, ...]:
    """Return relative paths of new or updated binaries in *cache_dir*."""
    post_new: list[Path] = []
    all_post: list[Path] = []
    if not cache_dir.is_dir():
        return ()
    for item in sorted(cache_dir.rglob("*")):
        if not is_verified_binary_file(item):
            continue
        rel = item.relative_to(cache_dir)
        all_post.append(rel)
        st = item.stat()
        prev = pre_snapshot.get(rel)
        if prev is None or prev != (st.st_size, st.st_mtime_ns):
            post_new.append(rel)
    return tuple(post_new or all_post)


def _extract_cpp_string_literal(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        if any(tok in node.value for tok in _CPP_MARKERS):
            return node.value
    return None


def _prune_unused_cpp_globals(cpp_tu: str) -> str:
    directives, units = _split_cpp_units(cpp_tu)
    if not units:
        return cpp_tu.strip() + ("\n" if cpp_tu.strip() else "")
    non_globals = [u for u in units if u.kind != "global_decl"]
    non_global_masked = strip_cpp_comments_and_strings("\n".join(u.text for u in non_globals))
    kept: list[str] = []
    if directives:
        kept.append("\n".join(directives))
    kept_units = []
    for unit in units:
        if unit.kind != "global_decl":
            kept_units.append(unit)
            continue
        if not unit.name or re.search(rf"\b{re.escape(unit.name)}\b", non_global_masked):
            kept_units.append(unit)
    for unit in sorted(kept_units, key=lambda u: u.start_char):
        text = unit.text.strip()
        if text:
            kept.append(text)
    return "\n\n".join(kept) + ("\n" if kept else "")


def _extract_cpp_closure(
    cpp_source: str,
    *,
    module_source: str,
    regime_tag: str,
    active_kernel: str = "",
) -> str:
    if active_kernel:
        _, units = _split_cpp_units(cpp_source)
        if any(u.kind == "kernel" and u.name == active_kernel for u in units):
            _, extracted = extract_reachable_trackb_source(
                cpp_source,
                module_source=module_source,
                regime_tag=regime_tag or active_kernel,
            )
            return _prune_unused_cpp_globals(extracted)
    selected = select_active_trackb_kernel(
        cpp_source,
        active_kernel=active_kernel or None,
        regime_tag=regime_tag,
        module_source=module_source,
    )
    if selected is None and regime_tag:
        tag_low = regime_tag.lower()
        tag_parts = [p for p in re.split(r"[^a-z0-9]+", tag_low) if len(p) >= 2]
        _, units = _split_cpp_units(cpp_source)
        kernels = [u for u in units if u.kind == "kernel" and u.name]
        best = max(
            kernels,
            key=lambda u: (
                sum(1 for p in tag_parts if p in u.name.lower()),
                u.line_count,
            ),
            default=None,
        )
        if best is not None and any(p in best.name.lower() for p in tag_parts):
            _, units_all = _split_cpp_units(cpp_source)
            callables = {
                u.name: u for u in units_all if u.kind in ("kernel", "helper") and u.name
            }
            _, extracted = extract_reachable_trackb_source(
                cpp_source,
                module_source=module_source,
                regime_tag=best.name,
            )
            if best.name in callables:
                return _prune_unused_cpp_globals(extracted)
    _, extracted = extract_reachable_trackb_source(
        cpp_source,
        module_source=module_source,
        regime_tag=regime_tag,
    )
    return _prune_unused_cpp_globals(extracted)


def _score_name_for_regime(name: str, regime_tag: str) -> int:
    tag = regime_tag.strip().lower()
    if not tag:
        return 0
    low = name.lower()
    if tag in low:
        return 100 + len(tag)
    long_tokens = [tok for tok in re.split(r"[^a-z0-9]+", tag) if len(tok) >= 3]
    word_tokens = [tok for tok in re.split(r"[^a-z0-9]+", tag) if len(tok) >= 2]
    name_words = set(re.split(r"[^a-z0-9]+", low))
    score = sum(10 + len(tok) for tok in long_tokens if tok in low)
    score += sum(3 for tok in word_tokens if tok in name_words)
    return score


def _collect_ast_names(node: ast.AST) -> set[str]:
    return {sub.id for sub in ast.walk(node) if isinstance(sub, ast.Name)}


def _node_source_slice(lines: Sequence[str], node: ast.stmt) -> str:
    decs = getattr(node, "decorator_list", ())
    start = min((d.lineno for d in decs), default=node.lineno)
    end = node.end_lineno or node.lineno
    return "".join(lines[start - 1 : end]).rstrip()


def _is_regime_candidate_fn(name: str, seg: str) -> bool:
    if "@T.prim_func" in seg:
        return True
    if re.match(r"^_?(?:select|build)_.*(?:_config|_func|_callable|_kernel)$", name):
        return True
    return bool(re.match(r"^_?(?:kernel_[a-z0-9_]+|[a-z0-9_]+_kernel)$", name))


def _resolve_branch_for_regime(
    fnode: ast.FunctionDef | ast.AsyncFunctionDef,
    regime_tag: str,
    lines: Sequence[str],
) -> str | None:
    tag = regime_tag.strip().lower()
    if not tag:
        return None
    for stmt in fnode.body:
        if not isinstance(stmt, ast.If):
            continue
        cur: ast.If | None = stmt
        while cur is not None:
            test_seg = ast.unparse(cur.test).lower()
            body_seg = "\n".join(_node_source_slice(lines, s) for s in cur.body).lower()
            if tag in test_seg or tag in body_seg:
                header = f"def {fnode.name}({ast.unparse(fnode.args)}):"
                body = "\n".join(_node_source_slice(lines, s) for s in cur.body)
                return f"{header}\n{body}"
            if len(cur.orelse) == 1 and isinstance(cur.orelse[0], ast.If):
                cur = cur.orelse[0]
            else:
                cur = None
    return None


def extract_reachable_closure_source(
    module_source: str,
    *,
    active_symbols: Sequence[str] = (),
    regime_tag: str = "",
) -> str:
    """Extract normalized reachable Python and Track-B C++ closure for one regime."""
    if not module_source.strip():
        return ""
    try:
        tree = ast.parse(module_source)
    except SyntaxError:
        return module_source.strip()

    lines = module_source.splitlines(keepends=True)
    imports: list[tuple[set[str], str]] = []
    funcs: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    classes: dict[str, ast.ClassDef] = {}
    assigns: dict[str, ast.stmt] = {}
    cpp_constants: dict[str, str] = {}
    ordered_defs: list[tuple[str, str]] = []

    for node in tree.body:
        if isinstance(node, ast.Import):
            bound = {alias.asname or alias.name.split(".")[0] for alias in node.names}
            imports.append((bound, _node_source_slice(lines, node)))
        elif isinstance(node, ast.ImportFrom):
            bound = {alias.asname or alias.name for alias in node.names}
            imports.append((bound, _node_source_slice(lines, node)))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs[node.name] = node
            ordered_defs.append(("func", node.name))
        elif isinstance(node, ast.ClassDef):
            classes[node.name] = node
            ordered_defs.append(("class", node.name))
        elif isinstance(node, ast.Assign) and len(node.targets) == 1:
            tgt = node.targets[0]
            if isinstance(tgt, ast.Name):
                assigns[tgt.id] = node
                ordered_defs.append(("assign", tgt.id))
                cpp_val = _extract_cpp_string_literal(node.value)
                if cpp_val is not None:
                    cpp_constants[tgt.id] = cpp_val
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            assigns[node.target.id] = node
            ordered_defs.append(("assign", node.target.id))
            cpp_val = _extract_cpp_string_literal(node.value)
            if cpp_val is not None:
                cpp_constants[node.target.id] = cpp_val

    all_defs: dict[str, ast.stmt] = {**assigns, **classes, **funcs}
    fn_segments = {name: _node_source_slice(lines, node) for name, node in funcs.items()}
    regime_entry_fns = {
        name for name, seg in fn_segments.items() if _is_regime_candidate_fn(name, seg)
    }

    seeds: list[str] = [s for s in active_symbols if s in all_defs]
    excluded_siblings: set[str] = set()
    active_cpp_kernel = ""

    if not seeds and regime_tag:
        scored_selectors: list[tuple[int, str]] = []
        scored_builders: list[tuple[int, str]] = []
        scored_other: list[tuple[int, str]] = []
        for name in funcs:
            score = _score_name_for_regime(name, regime_tag)
            if score <= 0:
                continue
            if name.startswith("_select_") or name.startswith("select_"):
                scored_selectors.append((score, name))
            elif name in regime_entry_fns:
                scored_builders.append((score, name))
            else:
                scored_other.append((score, name))

        if scored_selectors:
            max_sel = max(s for s, _ in scored_selectors)
            seeds.extend(n for s, n in scored_selectors if s == max_sel)
        if scored_builders:
            max_bld = max(s for s, _ in scored_builders)
            seeds.extend(n for s, n in scored_builders if s == max_bld)
        if not seeds and scored_other:
            max_oth = max(s for s, _ in scored_other)
            seeds.extend(n for s, n in scored_other if s == max_oth)

        matched_cpp: list[tuple[int, str, str]] = []
        for cname, cpp_txt in cpp_constants.items():
            cscore = _score_name_for_regime(cname, regime_tag)
            kname = select_active_trackb_kernel(
                cpp_txt,
                regime_tag=regime_tag,
                module_source=module_source,
            )
            kscore = _score_name_for_regime(kname or "", regime_tag)
            if cscore > 0 or kscore > 0:
                matched_cpp.append((max(cscore, kscore), cname, kname or ""))
        if matched_cpp:
            matched_cpp.sort(key=lambda row: row[0], reverse=True)
            top_cscore = matched_cpp[0][0]
            for cscore, cname, kname in matched_cpp:
                if cscore == top_cscore:
                    if cname not in seeds:
                        seeds.append(cname)
                    if not active_cpp_kernel and kname:
                        active_cpp_kernel = kname
            for cname in cpp_constants:
                if cname not in seeds:
                    excluded_siblings.add(cname)

        if seeds:
            for rname in regime_entry_fns:
                if rname not in seeds:
                    excluded_siblings.add(rname)
        else:
            for fname, fnode in funcs.items():
                branch = _resolve_branch_for_regime(fnode, regime_tag, lines)
                if branch is not None:
                    branch_names = _collect_ast_names(ast.parse(branch))
                    branch_defs = [n for n in branch_names if n in all_defs and n != fname]
                    if branch_defs:
                        seeds.extend(branch_defs)
                        break

        if not seeds and cpp_constants:
            seeds.extend(cpp_constants.keys())

    if not seeds:
        for fallback in ("kernel", "_get_kernel", "_render_trackb_tu"):
            if fallback in all_defs:
                seeds.append(fallback)
        for fname in funcs:
            if fname in regime_entry_fns or fname.startswith("_render_"):
                if fname not in seeds:
                    seeds.append(fname)
        for cname in cpp_constants:
            if cname not in seeds:
                seeds.append(cname)

    if not seeds:
        return module_source.strip()

    if "_get_kernel" in funcs and "_get_kernel" not in seeds and not active_symbols:
        for s in list(seeds):
            if s in funcs and (
                "prim_func" in fn_segments.get(s, "") or s.startswith("_build_")
            ):
                seeds.append("_get_kernel")
                break

    visited: set[str] = set()
    queue: list[str] = list(seeds)
    while queue:
        sym = queue.pop(0)
        if sym in visited or sym not in all_defs:
            continue
        visited.add(sym)
        for ref in sorted(_collect_ast_names(all_defs[sym])):
            if ref in all_defs and ref not in visited and ref not in excluded_siblings:
                queue.append(ref)

    for fn_name, fnode in list(funcs.items()):
        if fn_name in visited or fn_name in excluded_siblings:
            continue
        if any(tok in fn_segments[fn_name] for tok in ("compile_ascend", "_render_trackb_tu")):
            refs = _collect_ast_names(fnode)
            if refs & set(seeds) & set(cpp_constants):
                visited.add(fn_name)
                for ref in sorted(refs):
                    if ref in all_defs and ref not in visited and ref not in excluded_siblings:
                        visited.add(ref)

    used_names: set[str] = set(visited)
    for sym in visited:
        used_names.update(_collect_ast_names(all_defs[sym]))

    parts: list[str] = []
    for bound, imp_src in imports:
        if bound & used_names or "*" in bound:
            parts.append(imp_src)

    for kind, name in ordered_defs:
        if name not in visited:
            continue
        if kind == "assign" and name in cpp_constants:
            sliced_cpp = _extract_cpp_closure(
                cpp_constants[name],
                module_source=module_source,
                regime_tag=regime_tag,
                active_kernel=active_cpp_kernel,
            )
            parts.append(f'{name} = r"""\n{sliced_cpp}"""')
        else:
            parts.append(_node_source_slice(lines, all_defs[name]))

    return "\n\n".join(part for part in parts if part.strip()).strip()


def _canonical_json_value(val: object) -> object:
    if isinstance(val, Mapping):
        return {
            str(k): _canonical_json_value(v)
            for k, v in sorted(val.items(), key=lambda kv: str(kv[0]))
        }
    if isinstance(val, (list, tuple)):
        return [_canonical_json_value(item) for item in val]
    if isinstance(val, (set, frozenset)):
        items = [_canonical_json_value(item) for item in val]
        return sorted(items, key=lambda x: json.dumps(x, sort_keys=True, separators=(",", ":")))
    if isinstance(val, (str, int, float, bool)) or val is None:
        return val
    return str(val)


def compute_compile_task_key(
    *,
    normalized_kernel_unit_src: str,
    cann_target: str = "dav-3510",
    pass_configs: Mapping[str, object] | None = None,
    compile_flags: Sequence[str] = (),
    static_spec_signature: str = "",
    compiler_stamp: str = "",
) -> str:
    """Compute canonical SHA256 task key shared across Stage-B, Prewarm, and L1..L4."""
    sorted_pass_configs = (
        _canonical_json_value(pass_configs) if pass_configs is not None else {}
    )
    sorted_compile_flags = sorted(
        {str(flag).strip() for flag in compile_flags if str(flag).strip()}
    )
    payload = json.dumps(
        {
            "cann_target": str(cann_target),
            "compile_flags": sorted_compile_flags,
            "compiler_stamp": str(compiler_stamp),
            "normalized_kernel_unit_src": str(normalized_kernel_unit_src),
            "pass_configs": sorted_pass_configs,
            "static_spec_signature": str(static_spec_signature),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def resolve_unified_tl_cache_dir(
    tl_cache_dir: str | Path | None = None,
    *,
    env_vals: Mapping[str, str] | None = None,
    source_root: Path | None = None,
) -> Path:
    """Resolve the canonical unified TileLang and Track-B ArtifactStore cache directory."""
    _ = source_root
    if tl_cache_dir is not None and str(tl_cache_dir).strip():
        return Path(tl_cache_dir).expanduser().resolve()
    if env_vals is not None:
        env_tl = str(env_vals.get("TILELANG_CACHE_DIR", "") or "").strip()
        if env_tl:
            return Path(env_tl).expanduser().resolve()
        env_sf = str(env_vals.get("TF_SURROGATE_TL_CACHE_DIR", "") or "").strip()
        if env_sf:
            return Path(env_sf).expanduser().resolve()
        env_store = str(env_vals.get("TILEFOUNDRY_NPU_ARTIFACT_STORE", "") or "").strip()
        if env_store:
            return Path(env_store).expanduser().resolve()
        env_work = str(env_vals.get("TILEFOUNDRY_NPU_WORK_ROOT", "") or "").strip()
        if env_work:
            return Path(env_work).expanduser().resolve() / "surrogate" / "tilelang"
    for key in (
        "TILELANG_CACHE_DIR",
        "TF_SURROGATE_TL_CACHE_DIR",
        "TILEFOUNDRY_NPU_ARTIFACT_STORE",
    ):
        raw_val = os.environ.get(key, "").strip()
        if raw_val:
            return Path(raw_val).expanduser().resolve()
    raw_work = os.environ.get("TILEFOUNDRY_NPU_WORK_ROOT", "").strip()
    if raw_work:
        return Path(raw_work).expanduser().resolve() / "surrogate" / "tilelang"
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
) -> str:
    """Compute SHA256 hex digest identifying a compiled kernel artifact."""
    flag_items = tuple(tok for tok in str(flags).split() if tok)
    target_tag = (
        f"{arch}:{cann_series}"
        if cann_series and cann_series != "cann_9_3"
        else str(arch)
    )
    return compute_compile_task_key(
        normalized_kernel_unit_src=str(closure_src),
        cann_target=target_tag,
        pass_configs=None,
        compile_flags=flag_items,
        static_spec_signature=str(spec_key),
        compiler_stamp=str(compiler_stamp),
    )


@dataclass(frozen=True)
class ArtifactRecord:
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

    def to_dict(self) -> dict[str, object]:
        """Serialize record into a JSON-compatible dictionary."""
        return {
            "artifact_id": self.artifact_id,
            "backend_kind": self.backend_kind,
            "binary_paths": list(self.binary_paths),
            "arch": self.arch,
            "cann_series": self.cann_series,
            "flags": self.flags,
            "spec_key": self.spec_key,
            "compiler_stamp": self.compiler_stamp,
            "closure_sha256": self.closure_sha256,
            "metadata": dict(self.metadata),
            "recorded_at": float(self.recorded_at),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> ArtifactRecord:
        """Reconstruct ArtifactRecord from a JSON mapping."""
        raw_bins = payload.get("binary_paths")
        bins = (
            tuple(str(p) for p in raw_bins if str(p).strip())
            if isinstance(raw_bins, Sequence) and not isinstance(raw_bins, (str, bytes))
            else ()
        )
        raw_meta = payload.get("metadata")
        meta = dict(raw_meta) if isinstance(raw_meta, Mapping) else {}
        return cls(
            artifact_id=str(payload.get("artifact_id", "")),
            backend_kind=str(payload.get("backend_kind", "tilelang")),
            binary_paths=bins,
            arch=str(payload.get("arch", "dav-3510")),
            cann_series=str(payload.get("cann_series", "cann_9_3")),
            flags=str(payload.get("flags", "")),
            spec_key=str(payload.get("spec_key", "")),
            compiler_stamp=str(payload.get("compiler_stamp", "")),
            closure_sha256=str(payload.get("closure_sha256", "")),
            metadata=meta,
            recorded_at=float(payload.get("recorded_at", 0.0) or 0.0),
        )


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
            try:
                raw = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                if isinstance(raw, Mapping):
                    entries = raw.get("artifacts")
                    if isinstance(entries, Mapping):
                        for key, row in entries.items():
                            if isinstance(row, Mapping):
                                records[str(key)] = dict(row)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                pass
        if not records and self.index_path.is_file():
            try:
                for line in self.index_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if isinstance(row, Mapping) and row.get("artifact_id"):
                        records[str(row["artifact_id"])] = dict(row)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError):
                pass
        return records

    def _save_manifest_unlocked(self, records: Mapping[str, Mapping[str, object]]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "artifacts": dict(records)}
        encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
        fd, tmp_name = tempfile.mkstemp(dir=str(self.root), prefix=".artifact_manifest.")
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(encoded)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp_path, self.manifest_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    def record_verified_artifact(
        self,
        artifact_id: str,
        *,
        binary_paths: Sequence[str | Path] = (),
        backend_kind: str = "tilelang",
        arch: str = "dav-3510",
        cann_series: str = "cann_9_3",
        flags: str = "",
        spec_key: str = "",
        compiler_stamp: str = "",
        closure_src: str = "",
        metadata: Mapping[str, object] | None = None,
        copy_binaries: bool = False,
    ) -> ArtifactRecord:
        """Persist a verified artifact record and optionally copy binary files into blobs."""
        aid = str(artifact_id or "").strip()
        if not aid:
            raise ValueError("artifact_id must not be empty")
        self.root.mkdir(parents=True, exist_ok=True)
        resolved_bins: list[str] = []
        for raw_p in binary_paths:
            src_p = Path(raw_p).resolve()
            if not _is_verified_binary_file(src_p):
                continue
            if copy_binaries:
                dest_dir = self.blobs_dir / aid
                dest_dir.mkdir(parents=True, exist_ok=True)
                dest_p = dest_dir / src_p.name
                if src_p != dest_p:
                    fd, tmp_name = tempfile.mkstemp(dir=str(dest_dir), prefix=f".{src_p.name}.")
                    os.close(fd)
                    tmp_p = Path(tmp_name)
                    try:
                        shutil.copy2(src_p, tmp_p)
                        os.replace(tmp_p, dest_p)
                    finally:
                        if tmp_p.exists():
                            tmp_p.unlink(missing_ok=True)
                resolved_bins.append(str(dest_p))
            else:
                resolved_bins.append(str(src_p))

        dedup_bins = tuple(dict.fromkeys(resolved_bins))
        closure_sha = (
            hashlib.sha256(closure_src.encode("utf-8")).hexdigest() if closure_src else ""
        )
        record = ArtifactRecord(
            artifact_id=aid,
            backend_kind=str(backend_kind),
            binary_paths=dedup_bins,
            arch=str(arch),
            cann_series=str(cann_series),
            flags=str(flags),
            spec_key=str(spec_key),
            compiler_stamp=str(compiler_stamp),
            closure_sha256=closure_sha,
            metadata=dict(metadata) if metadata is not None else {},
            recorded_at=time.time(),
        )
        with self._lock_path.open("a+", encoding="utf-8") as lock_fh:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX)
            try:
                records = self._load_records_unlocked()
                row_dict = record.to_dict()
                records[aid] = row_dict
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
        if not aid:
            return None
        records = self._load_records_unlocked()
        raw = records.get(aid)
        if not isinstance(raw, Mapping):
            return None
        record = ArtifactRecord.from_dict(raw)
        if not record.binary_paths:
            return None
        for bin_path_str in record.binary_paths:
            if not _is_verified_binary_file(Path(bin_path_str)):
                return None
        return record

    def has_verified_artifact(self, artifact_id: str) -> bool:
        """Return True iff artifact_id has on-disk verified binary files with st_size > 0."""
        return self.get_verified_artifact(artifact_id) is not None
