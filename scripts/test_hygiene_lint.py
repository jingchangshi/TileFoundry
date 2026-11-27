#!/usr/bin/env python3
"""Enforce test file naming and private-attribute monkeypatch hygiene."""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

VALID_RULES = ("version_filename", "private_monkeypatch")
VERSION_STEM_RE = re.compile(r"(?:^test_v\d+|_v\d+(?:_\d+)?$)")
PROD_ROOTS = frozenset({"tilefoundry_ascend", "tilefoundry", "examples"})
STDLIB_ROOTS = sys.stdlib_module_names | frozenset({"pytest", "unittest", "builtins"})
PATCH_CALL_NAMES = frozenset({"patch"})


@dataclass(frozen=True)
class Violation:
    """Single test hygiene violation finding."""

    path: str
    line: int
    rule: str
    detail: str


@dataclass(frozen=True)
class LintReport:
    """Summary report produced by a test hygiene lint run."""

    root: str
    files_scanned: int
    elapsed_ms: float
    total_violations: int
    counts_by_rule: dict[str, int]
    violations: list[Violation]


def _default_root() -> Path:
    return Path(__file__).resolve().parent.parent / "tests" / "ascend"


def _is_private_member(name: str) -> bool:
    if not name.startswith("_"):
        return False
    return not (name.startswith("__") and name.endswith("__"))


def _string_literal(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)
        return f"{parent}.{node.attr}" if parent else None
    return None


def _root_module(node: ast.AST) -> str | None:
    dotted = _dotted_name(node)
    if not dotted:
        return None
    return dotted.split(".", 1)[0]


def _collect_provenance(
    tree: ast.Module,
) -> tuple[set[str], set[str], set[str], set[str], set[str]]:
    """Collect stdlib aliases, local stub names, and mock.patch aliases."""
    stdlib_names: set[str] = set()
    prod_names: set[str] = set()
    local_stub_names: set[str] = set()
    patch_funcs: set[str] = set()
    patch_obj_funcs: set[str] = set()
    mock_modules: set[str] = set()
    local_classes: set[str] = set()
    fixture_funcs: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                bound = alias.asname or alias.name.split(".", 1)[0]
                root = alias.name.split(".", 1)[0]
                if alias.name == "unittest.mock" or (
                    alias.name == "unittest" and alias.asname is None
                ):
                    mock_modules.add(bound)
                if root in STDLIB_ROOTS:
                    stdlib_names.add(bound)
                elif root in PROD_ROOTS:
                    prod_names.add(bound)
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            root = mod.split(".", 1)[0]
            for alias in node.names:
                bound = alias.asname or alias.name
                if mod == "unittest.mock" or (mod == "unittest" and alias.name == "mock"):
                    if alias.name == "patch":
                        patch_funcs.add(bound)
                    elif alias.name == "mock":
                        mock_modules.add(bound)
                if mod == "types" and alias.name == "SimpleNamespace":
                    local_stub_names.add(bound)
                if root in STDLIB_ROOTS:
                    stdlib_names.add(bound)
                elif root in PROD_ROOTS:
                    prod_names.add(bound)
        elif isinstance(node, ast.ClassDef):
            local_classes.add(node.name)
            local_stub_names.add(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for dec in node.decorator_list:
                dec_name = _dotted_name(dec.func if isinstance(dec, ast.Call) else dec) or ""
                if dec_name.endswith("fixture"):
                    fixture_funcs.add(node.name)

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Call):
                callee = _dotted_name(node.value.func)
                if callee and callee.split(".", 1)[0] in (local_classes | {"SimpleNamespace"}):
                    local_stub_names.add(target.id)
        elif isinstance(node, ast.FunctionDef) and node.name in fixture_funcs:
            returns_prod = any(
                isinstance(sub, ast.Return)
                and sub.value is not None
                and (_root_module(sub.value) in prod_names)
                for sub in ast.walk(node)
            )
            if not returns_prod:
                local_stub_names.add(node.name)

    for mod_name in mock_modules:
        if mod_name == "unittest":
            patch_funcs.add("unittest.mock.patch")
            patch_obj_funcs.add("unittest.mock.patch.object")
        else:
            patch_funcs.add(f"{mod_name}.patch")
            patch_obj_funcs.add(f"{mod_name}.patch.object")
    for p_name in list(patch_funcs):
        patch_obj_funcs.add(f"{p_name}.object")

    return stdlib_names, prod_names, local_stub_names, patch_funcs, patch_obj_funcs


def _is_allowed_target_obj(
    node: ast.AST,
    stdlib_names: set[str],
    prod_names: set[str],
    local_stub_names: set[str],
) -> bool:
    root = _root_module(node)
    if root is None:
        return False
    if root in prod_names:
        return False
    return root in stdlib_names or root in local_stub_names


def _check_dotted_target(
    rel_path: str,
    lineno: int,
    target_str: str,
) -> Violation | None:
    parts = target_str.split(".")
    if len(parts) < 2:
        return None
    root = parts[0]
    if root in STDLIB_ROOTS:
        return None
    private_parts = [part for part in parts[1:] if _is_private_member(part)]
    if not private_parts:
        return None
    return Violation(
        rel_path,
        lineno,
        "private_monkeypatch",
        f"cannot patch private member `{private_parts[0]}` via target `{target_str}`",
    )


def _kwarg_value(call: ast.Call, name: str) -> ast.AST | None:
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    return None


def _scan_private_monkeypatches(rel_path: str, tree: ast.Module) -> list[Violation]:
    stdlib_names, prod_names, local_stub_names, patch_funcs, patch_obj_funcs = (
        _collect_provenance(tree)
    )
    findings: list[Violation] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        callee = _dotted_name(node.func)
        if not callee:
            continue

        if callee == "monkeypatch.setattr" or callee.endswith(".monkeypatch.setattr"):
            target_arg = node.args[0] if node.args else _kwarg_value(node, "target")
            name_arg = node.args[1] if len(node.args) >= 2 else _kwarg_value(node, "name")
            target_str = _string_literal(target_arg)
            if target_str is not None and "." in target_str:
                viol = _check_dotted_target(rel_path, node.lineno, target_str)
                if viol is not None:
                    findings.append(viol)
                continue
            attr_str = _string_literal(name_arg)
            if (
                target_arg is not None
                and attr_str is not None
                and _is_private_member(attr_str)
                and not _is_allowed_target_obj(
                    target_arg, stdlib_names, prod_names, local_stub_names
                )
            ):
                target_desc = _dotted_name(target_arg) or "<expr>"
                findings.append(
                    Violation(
                        rel_path,
                        node.lineno,
                        "private_monkeypatch",
                        f"monkeypatch.setattr targets private `{attr_str}` on `{target_desc}`",
                    )
                )
        elif callee in patch_obj_funcs or callee.endswith(".patch.object"):
            target_arg = node.args[0] if node.args else _kwarg_value(node, "target")
            attr_arg = node.args[1] if len(node.args) >= 2 else _kwarg_value(node, "attribute")
            attr_str = _string_literal(attr_arg)
            if (
                target_arg is not None
                and attr_str is not None
                and _is_private_member(attr_str)
                and not _is_allowed_target_obj(
                    target_arg, stdlib_names, prod_names, local_stub_names
                )
            ):
                target_desc = _dotted_name(target_arg) or "<expr>"
                findings.append(
                    Violation(
                        rel_path,
                        node.lineno,
                        "private_monkeypatch",
                        f"{callee} targets private `{attr_str}` on `{target_desc}`",
                    )
                )
        elif callee in patch_funcs or callee.endswith(".patch"):
            target_arg = node.args[0] if node.args else _kwarg_value(node, "target")
            target_str = _string_literal(target_arg)
            if target_str is not None:
                viol = _check_dotted_target(rel_path, node.lineno, target_str)
                if viol is not None:
                    findings.append(viol)

    return findings


def run_lint(root: Path, rules: set[str] | None = None) -> LintReport:
    """Scan *root* and return a LintReport for the requested rules."""
    t0 = time.perf_counter()
    active = set(VALID_RULES) if rules is None else set(rules)
    resolved_root = root.resolve()
    violations: list[Violation] = []
    files_scanned = 0

    for file_path in sorted(resolved_root.rglob("*.py")):
        rel_parts = file_path.relative_to(resolved_root).parts
        if "benchmarks" in rel_parts:
            continue
        files_scanned += 1
        rel_path = file_path.relative_to(resolved_root).as_posix()

        if "version_filename" in active and VERSION_STEM_RE.search(file_path.stem):
            violations.append(
                Violation(
                    rel_path,
                    1,
                    "version_filename",
                    f"versioned test filename `{file_path.name}` is forbidden",
                )
            )

        if "private_monkeypatch" in active:
            source = file_path.read_text(encoding="utf-8")
            if (
                "monkeypatch" in source
                or "patch" in source
            ):
                try:
                    tree = ast.parse(source, filename=str(file_path))
                except SyntaxError:
                    continue
                violations.extend(_scan_private_monkeypatches(rel_path, tree))

    violations.sort(key=lambda item: (item.path, item.line, item.rule, item.detail))
    counts = {rule: 0 for rule in VALID_RULES}
    for item in violations:
        counts[item.rule] += 1
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return LintReport(
        root=str(resolved_root),
        files_scanned=files_scanned,
        elapsed_ms=round(elapsed_ms, 2),
        total_violations=len(violations),
        counts_by_rule=counts,
        violations=violations,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the Ascend test hygiene linter."""
    parser = argparse.ArgumentParser(
        description="Check tests/ascend against version-filename and private-monkeypatch rules."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=_default_root(),
        help="Root directory to scan (default: tests/ascend).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON report.",
    )
    args = parser.parse_args(argv)
    if not args.root.is_dir():
        print(f"error: root directory does not exist: {args.root}", file=sys.stderr)
        return 2

    report = run_lint(args.root)
    if args.json:
        payload = {
            "root": report.root,
            "files_scanned": report.files_scanned,
            "elapsed_ms": report.elapsed_ms,
            "total_violations": report.total_violations,
            "counts_by_rule": report.counts_by_rule,
            "violations": [asdict(item) for item in report.violations],
        }
        print(json.dumps(payload, indent=2))
    else:
        for item in report.violations:
            print(f"{item.path}:{item.line}: [{item.rule}] {item.detail}")
        summary = ", ".join(f"{rule}={report.counts_by_rule[rule]}" for rule in VALID_RULES)
        print(
            f"Test hygiene lint scanned {report.files_scanned} files in {report.elapsed_ms:.2f}ms: "
            f"{report.total_violations} violation(s) ({summary})"
        )
    return 1 if report.total_violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
