#!/usr/bin/env python3
"""Offline AST auditor verifying TILELANG_DSL_SURFACE sync with upstream TileLang."""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys
from pathlib import Path

from tilefoundry_ascend.ascend.target.dialect_surface import TILELANG_DSL_SURFACE


def _default_tilelang_root() -> Path:
    env_root = os.environ.get("TILELANG_ROOT", "").strip()
    if env_root:
        return Path(env_root)
    repo_root = Path(__file__).resolve().parents[2]
    return repo_root / "tilelang"


def _parse_module(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _collect_str_tuple_assign(tree: ast.Module, target_name: str) -> set[str]:
    names: set[str] = set()
    for stmt in tree.body:
        if not isinstance(stmt, ast.Assign):
            continue
        if not any(isinstance(t, ast.Name) and t.id == target_name for t in stmt.targets):
            continue
        if isinstance(stmt.value, (ast.Tuple, ast.List)):
            for elt in stmt.value.elts:
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    names.add(elt.value)
    return names


def _collect_imported_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for stmt in tree.body:
        if isinstance(stmt, ast.ImportFrom):
            for alias in stmt.names:
                if alias.name != "*":
                    names.add(alias.asname or alias.name)
        elif isinstance(stmt, ast.Import):
            for alias in stmt.names:
                names.add(alias.asname or alias.name)
    return names


def _collect_defs_with_params(path: Path) -> dict[str, set[str]]:
    if not path.is_file():
        return {}
    tree = _parse_module(path)
    defs: dict[str, set[str]] = {}
    for stmt in tree.body:
        if isinstance(stmt, ast.FunctionDef):
            all_args = stmt.args.posonlyargs + stmt.args.args + stmt.args.kwonlyargs
            defs[stmt.name] = {arg.arg for arg in all_args}
    return defs


def _extract_signature_kwargs(signature: str) -> set[str]:
    cleaned = re.sub(r"\[[^\]]*\]", "", signature)
    kwargs: set[str] = set()
    for arg_list in re.findall(r"\(([^()]*)\)", cleaned):
        kwargs.update(re.findall(r"(?:^|,)\s*([A-Za-z_][A-Za-z0-9_]*)\s*=", arg_list))
    return kwargs


def _collect_ascend_exports_and_params(
    tilelang_root: Path,
) -> tuple[set[str], dict[str, set[str]], dict[str, set[str]]]:
    asc_dir = tilelang_root / "tilelang" / "ascend" / "language"
    common_dir = tilelang_root / "tilelang" / "language"
    init_tree = _parse_module(asc_dir / "__init__.py")
    common_tree = _parse_module(common_dir / "common.py")

    exports = _collect_str_tuple_assign(init_tree, "_ASCEND_API_ALL")
    exports.update(_collect_imported_names(init_tree))
    exports.update(_collect_str_tuple_assign(common_tree, "_LOCAL_EXPORTS"))
    exports.update(_collect_imported_names(common_tree))

    for sub in ("dma.py", "mode.py", "sync.py", "math.py"):
        sub_path = asc_dir / sub
        if sub_path.is_file():
            sub_tree = _parse_module(sub_path)
            exports.update(_collect_str_tuple_assign(sub_tree, "__all__"))
            exports.update(_collect_defs_with_params(sub_path).keys())

    for sub in ("fill_op.py", "symbolics.py", "loop.py", "allocate.py", "reduce_op.py"):
        sub_path = common_dir / sub
        if sub_path.is_file():
            exports.update(_collect_defs_with_params(sub_path).keys())

    lang_params: dict[str, set[str]] = {}
    for path in (
        common_dir / "loop.py",
        common_dir / "allocate.py",
        common_dir / "reduce_op.py",
        common_dir / "fill_op.py",
        common_dir / "symbolics.py",
        asc_dir / "copy_op.py",
        asc_dir / "gemm_op.py",
        asc_dir / "reduce_op.py",
        asc_dir / "kernel.py",
        asc_dir / "schedule_hint.py",
        asc_dir / "frame.py",
        asc_dir / "annotations.py",
        asc_dir / "allocate.py",
        asc_dir / "sync.py",
    ):
        lang_params.update(_collect_defs_with_params(path))

    simd_params = _collect_defs_with_params(asc_dir / "simd.py")
    return exports, lang_params, simd_params


def _collect_skill_documented_symbols(tilelang_root: Path) -> set[str]:
    skill_dir = tilelang_root / ".agents" / "skills" / "tilelang-ascend"
    documented: set[str] = set()
    for rel in ("SKILL.md", "references/synchronization.md"):
        doc_path = skill_dir / rel
        if not doc_path.is_file():
            continue
        text = doc_path.read_text(encoding="utf-8")
        for sym in re.findall(r"\bT\.(?:simd\.[a-zA-Z0-9_]+|[A-Za-z_][A-Za-z0-9_]*)\b", text):
            documented.add(sym)
    return documented


_REQUIRED_SKILL_PRIMITIVES: tuple[str, ...] = (
    "T.Kernel",
    "T.MixedKernel",
    "T.Pipelined",
    "T.Persistent",
    "T.SimdVF",
    "T.SimtVF",
    "T.Task",
    "T.PerCoreTask",
    "T.Stage",
    "T.copy",
    "T.dual_copy",
    "T.gemm",
    "T.alloc_shared",
    "T.alloc_l1",
    "T.alloc_l0a",
    "T.alloc_l0b",
    "T.alloc_l0c",
    "T.alloc_ss",
    "T.alloc_reducer",
    "T.reducer_init",
    "T.reducer_update",
    "T.finalize_reducer",
    "T.annotate_buffer_versions",
    "T.annotate_unlimit_memory",
    "T.assume_no_conflict",
    "T.assume_conflict",
    "T.ascend_pipe_barrier",
    "T.simd.vld",
    "T.simd.vsts",
)


def audit_surface_sync(tilelang_root: Path) -> list[str]:
    """Return audit errors for missing or stale TILELANG_DSL_SURFACE entries."""
    asc_init = tilelang_root / "tilelang" / "ascend" / "language" / "__init__.py"
    simd_py = tilelang_root / "tilelang" / "ascend" / "language" / "simd.py"
    skill_md = tilelang_root / ".agents" / "skills" / "tilelang-ascend" / "SKILL.md"
    if not asc_init.is_file() or not simd_py.is_file():
        return [f"Missing upstream TileLang ascend language sources under {tilelang_root}"]

    exports, lang_params, simd_params = _collect_ascend_exports_and_params(tilelang_root)
    documented = _collect_skill_documented_symbols(tilelang_root) if skill_md.is_file() else set()
    errors: list[str] = []

    surface_by_sym = {entry.symbol: entry for entry in TILELANG_DSL_SURFACE.primitives}
    for req_sym in _REQUIRED_SKILL_PRIMITIVES:
        if documented and req_sym not in documented:
            errors.append(f"Expected upstream skill symbol missing in SKILL docs: {req_sym}")
        if req_sym not in surface_by_sym:
            errors.append(f"Upstream skill primitive missing from TILELANG_DSL_SURFACE: {req_sym}")

    for entry in TILELANG_DSL_SURFACE.primitives:
        sym = entry.symbol
        if sym.startswith("T.simd."):
            fn_name = sym.removeprefix("T.simd.")
            if fn_name not in simd_params:
                errors.append(f"Stale T.simd primitive not found in simd.py: {sym}")
                continue
            sig_kwargs = _extract_signature_kwargs(entry.signature)
            missing_kwargs = sorted(sig_kwargs - simd_params[fn_name])
            if missing_kwargs:
                errors.append(
                    f"Stale kwargs {missing_kwargs} in {sym} signature vs simd.py"
                )
        elif sym.startswith("T."):
            fn_name = sym.removeprefix("T.")
            if fn_name not in exports:
                errors.append(f"Stale TileLang primitive not exported in __init__.py: {sym}")
                continue
            if fn_name in lang_params:
                sig_kwargs = _extract_signature_kwargs(entry.signature)
                missing_kwargs = sorted(sig_kwargs - lang_params[fn_name])
                if missing_kwargs:
                    errors.append(
                        f"Stale kwargs {missing_kwargs} in {sym} signature vs upstream def"
                    )
        else:
            if sym not in exports:
                errors.append(f"Stale symbol not exported in tilelang.ascend.language: {sym}")
    return errors


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint auditing TILELANG_DSL_SURFACE against upstream TileLang."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tilelang-root",
        type=Path,
        default=_default_tilelang_root(),
        help="Path to upstream tilelang checkout (defaults to TILELANG_ROOT or ../tilelang).",
    )
    args = parser.parse_args(argv)
    errors = audit_surface_sync(args.tilelang_root)
    if errors:
        for err in errors:
            print(f"ERROR: {err}", file=sys.stderr)
        return 1
    print(
        f"OK: verified {len(TILELANG_DSL_SURFACE.primitives)} TILELANG_DSL_SURFACE "
        f"primitives against {args.tilelang_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
