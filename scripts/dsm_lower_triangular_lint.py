#!/usr/bin/env python3
"""Enforce the 6-layer lower-triangular DSM architecture for tilefoundry_ascend."""

from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

VALID_CATEGORIES = ("layers", "l3_l4", "scc", "sys_modules", "reflection")

BANNED_REFLECTION_NAMES = frozenset(
    {
        "_invoke_with_optional_kwargs",
        "_sync_coordinators",
        "_PROPAGATED_HOOKS",
        "_owner_override",
        "_resolve_hook",
        "_bind_facade_globals",
        "_FACADE_NS",
        "_hook",
    }
)

_L0_PREFIXES = (
    "ascend.target.",
    "ascend.sil.",
    "ascend.passes.",
    "ascend.compile_cache.",
    "ascend.templates.",
)
_L0_EXACT = frozenset(
    {
        "__init__",
        "ascend.__init__",
        "ascend.json_record",
        "ascend.cost_model.bisheng_res_parser",
        "ascend.cost_model.microarch_profile",
    }
)

_L1_PREFIXES = ("agent.specs.", "agent.kb.")
_L1_EXACT = frozenset(
    {
        "agent.contracts",
        "ascend.cost_model.experience_lake",
        "ascend.cost_model.cost_gate_policy",
    }
)

_L2_PREFIXES = ("ascend.schedule.", "agent.router.", "agent.gates.")

_L3_PREFIXES = ("agent.orchestrator.", "agent.synthesis.", "agent.flywheel.")
_L3_EXACT = frozenset(
    {
        "agent.cost_advisor",
        "ascend.cost_model.hybrid_cost_model",
        "ascend.cost_model.pipe_schedule_eval",
        "ascend.cost_model.__init__",
    }
)

_L4_PREFIXES = ("agent.probes.", "agent.npu.")

_L5_PREFIXES = ("agent.campaign.",)
_L5_EXACT = frozenset({"agent.__init__"})

_LAYER_RULES: tuple[tuple[int, tuple[str, ...], frozenset[str]], ...] = (
    (0, _L0_PREFIXES, _L0_EXACT),
    (1, _L1_PREFIXES, _L1_EXACT),
    (2, _L2_PREFIXES, frozenset()),
    (3, _L3_PREFIXES, _L3_EXACT),
    (4, _L4_PREFIXES, frozenset()),
    (5, _L5_PREFIXES, _L5_EXACT),
)


@dataclass(frozen=True)
class Violation:
    """Single DSM architecture violation finding."""

    path: str
    line: int
    category: str
    message: str


@dataclass(frozen=True)
class LintReport:
    """Structured DSM lint summary across a package root."""

    root: str
    files_scanned: int
    elapsed_ms: float
    total_violations: int
    counts_by_category: dict[str, int]
    violations: list[Violation]


def classify_module(module_name: str) -> int | None:
    """Return layer index 0..5 for a tilefoundry_ascend module, or None."""
    if module_name == "tilefoundry_ascend.__init__":
        short = "__init__"
    elif module_name.startswith("tilefoundry_ascend."):
        short = module_name[len("tilefoundry_ascend.") :]
    else:
        return None
    for layer, prefixes, exact in _LAYER_RULES:
        if short in exact or any(short.startswith(prefix) for prefix in prefixes):
            return layer
    return None


def _module_name_from_path(root: Path, file_path: Path) -> str:
    rel_parts = file_path.relative_to(root).with_suffix("").parts
    return "tilefoundry_ascend." + ".".join(rel_parts)


def _resolve_target(name: str, known_modules: set[str]) -> str:
    pkg_init = f"{name}.__init__"
    if pkg_init in known_modules:
        return pkg_init
    return name


def _resolve_import_from_base(
    src_module: str, node: ast.ImportFrom
) -> str | None:
    if node.level == 0:
        base = node.module or ""
        if base == "tilefoundry_ascend" or base.startswith("tilefoundry_ascend."):
            return base
        return None
    if src_module.endswith(".__init__"):
        pkg = src_module[: -len(".__init__")]
    else:
        pkg = src_module.rsplit(".", 1)[0]
    pkg_parts = pkg.split(".")
    up = node.level - 1
    if up > len(pkg_parts):
        return None
    anchor = pkg_parts[: len(pkg_parts) - up]
    if node.module:
        anchor.extend(node.module.split("."))
    base = ".".join(anchor)
    if base == "tilefoundry_ascend" or base.startswith("tilefoundry_ascend."):
        return base
    return None


_BASE_SCAN_TOKENS = (
    "modules",
    "_invoke_with_optional_kwargs",
    "_sync_coordinators",
    "_PROPAGATED_HOOKS",
    "_owner_override",
    "_resolve_hook",
    "_bind_facade_globals",
    "_FACADE_NS",
    "_hook",
)


def _iter_import_nodes(tree: ast.Module) -> list[ast.Import | ast.ImportFrom]:
    imports: list[ast.Import | ast.ImportFrom] = []
    stack: list[ast.stmt] = list(tree.body)
    while stack:
        stmt = stack.pop()
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            imports.append(stmt)
            continue
        for attr in ("body", "orelse", "finalbody"):
            sub = getattr(stmt, attr, None)
            if sub:
                stack.extend(sub)
        handlers = getattr(stmt, "handlers", None)
        if handlers:
            for handler in handlers:
                stack.extend(handler.body)
        cases = getattr(stmt, "cases", None)
        if cases:
            for case in cases:
                stack.extend(case.body)
    imports.reverse()
    return imports


def _extract_internal_imports(
    src_module: str,
    nodes: list[ast.AST] | list[ast.Import | ast.ImportFrom],
    known_modules: set[str],
) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    seen_per_line: set[tuple[int, str]] = set()

    def add(line: int, target: str) -> None:
        key = (line, target)
        if key not in seen_per_line:
            seen_per_line.add(key)
            found.append(key)

    for node in nodes:
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if name == "tilefoundry_ascend" or name.startswith("tilefoundry_ascend."):
                    add(node.lineno, _resolve_target(name, known_modules))
        elif isinstance(node, ast.ImportFrom):
            base = _resolve_import_from_base(src_module, node)
            if base is None:
                continue
            for alias in node.names:
                if alias.name == "*":
                    add(node.lineno, _resolve_target(base, known_modules))
                    continue
                sub = f"{base}.{alias.name}"
                sub_init = f"{sub}.__init__"
                if sub_init in known_modules:
                    add(node.lineno, sub_init)
                elif sub in known_modules:
                    add(node.lineno, sub)
                else:
                    add(node.lineno, _resolve_target(base, known_modules))
    return found


def _is_sys_modules_expr(node: ast.AST, sys_aliases: set[str], mod_aliases: set[str]) -> bool:
    if (
        isinstance(node, ast.Attribute)
        and node.attr == "modules"
        and isinstance(node.value, ast.Name)
        and node.value.id in sys_aliases
    ):
        return True
    return isinstance(node, ast.Name) and node.id in mod_aliases


def _is_safe_external_probe_key(node: ast.AST) -> bool:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        val = node.value
        return "tilefoundry_ascend" not in val and not val.startswith(".")
    return False


def _scan_full_ast(
    rel_path: str,
    rel_parts: tuple[str, ...],
    tree: ast.AST,
    check_sys_modules: bool,
    check_reflection: bool,
) -> tuple[list[ast.Import | ast.ImportFrom], list[Violation]]:
    nodes = list(ast.walk(tree))
    imports: list[ast.Import | ast.ImportFrom] = []
    sys_aliases: set[str] = {"sys"}
    mod_aliases: set[str] = set()
    in_campaign = len(rel_parts) >= 2 and rel_parts[0] == "agent" and rel_parts[1] == "campaign"
    inspect_aliases: set[str] = {"inspect"}
    signature_aliases: set[str] = set()

    for node in nodes:
        if isinstance(node, ast.Import):
            imports.append(node)
            for alias in node.names:
                if alias.name == "sys":
                    sys_aliases.add(alias.asname or "sys")
                elif in_campaign and alias.name == "inspect":
                    inspect_aliases.add(alias.asname or "inspect")
        elif isinstance(node, ast.ImportFrom):
            imports.append(node)
            if node.module == "sys":
                for alias in node.names:
                    if alias.name == "modules":
                        mod_aliases.add(alias.asname or "modules")
            elif in_campaign and node.module == "inspect":
                for alias in node.names:
                    if alias.name == "signature":
                        signature_aliases.add(alias.asname or "signature")

    violations: list[Violation] = []
    seen: set[tuple[str, int, str]] = set()

    def record(cat: str, line: int, msg: str) -> None:
        key = (cat, line, msg)
        if key not in seen:
            seen.add(key)
            violations.append(Violation(rel_path, line, cat, msg))

    for node in nodes:
        if check_sys_modules:
            if isinstance(node, ast.Subscript) and _is_sys_modules_expr(
                node.value, sys_aliases, mod_aliases
            ):
                if not _is_safe_external_probe_key(node.slice):
                    record(
                        "sys_modules",
                        node.lineno,
                        "sys.modules subscript lookup targeting internal or dynamic module",
                    )
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and _is_sys_modules_expr(node.func.value, sys_aliases, mod_aliases)
                and node.func.attr in {"get", "pop", "setdefault"}
            ):
                key_node = node.args[0] if node.args else None
                if key_node is None or not _is_safe_external_probe_key(key_node):
                    record(
                        "sys_modules",
                        node.lineno,
                        f"sys.modules.{node.func.attr} lookup targeting internal or dynamic module",
                    )
            elif isinstance(node, ast.Compare):
                for left, op, right in zip(
                    [node.left, *node.comparators[:-1]], node.ops, node.comparators
                ):
                    if isinstance(op, (ast.In, ast.NotIn)) and _is_sys_modules_expr(
                        right, sys_aliases, mod_aliases
                    ):
                        if not _is_safe_external_probe_key(left):
                            record(
                                "sys_modules",
                                node.lineno,
                                "sys.modules membership probe targeting internal or dynamic module",
                            )
        if check_reflection:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                if node.name in BANNED_REFLECTION_NAMES:
                    record("reflection", node.lineno, f"banned reflection symbol `{node.name}`")
            elif isinstance(node, ast.Name):
                if node.id in BANNED_REFLECTION_NAMES:
                    record("reflection", node.lineno, f"banned reflection symbol `{node.id}`")
                elif in_campaign and node.id in signature_aliases:
                    record("reflection", node.lineno, "banned reflection symbol `inspect.signature`")
            elif isinstance(node, ast.Attribute):
                if node.attr in BANNED_REFLECTION_NAMES:
                    record("reflection", node.lineno, f"banned reflection symbol `{node.attr}`")
                elif (
                    in_campaign
                    and node.attr == "signature"
                    and isinstance(node.value, ast.Name)
                    and node.value.id in inspect_aliases
                ):
                    record("reflection", node.lineno, "banned reflection symbol `inspect.signature`")
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name in BANNED_REFLECTION_NAMES:
                        record(
                            "reflection", node.lineno, f"banned reflection symbol `{alias.name}`"
                        )
                    if alias.asname and alias.asname in BANNED_REFLECTION_NAMES:
                        record(
                            "reflection", node.lineno, f"banned reflection symbol `{alias.asname}`"
                        )
                if (
                    in_campaign
                    and node.module == "inspect"
                    and any(alias.name == "signature" for alias in node.names)
                ):
                    record("reflection", node.lineno, "banned reflection symbol `inspect.signature`")
    return imports, violations


def _tarjan_sccs(
    nodes: list[str], adj: dict[str, set[str]]
) -> list[list[str]]:
    index_counter = 0
    indices: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    sccs: list[list[str]] = []

    for start in nodes:
        if start in indices:
            continue
        call_stack: list[tuple[str, list[str], int]] = [
            (start, sorted(adj.get(start, ())), 0)
        ]
        indices[start] = index_counter
        lowlink[start] = index_counter
        index_counter += 1
        stack.append(start)
        on_stack.add(start)

        while call_stack:
            v, neighbors, pos = call_stack[-1]
            if pos < len(neighbors):
                call_stack[-1] = (v, neighbors, pos + 1)
                w = neighbors[pos]
                if w not in indices:
                    indices[w] = index_counter
                    lowlink[w] = index_counter
                    index_counter += 1
                    stack.append(w)
                    on_stack.add(w)
                    call_stack.append((w, sorted(adj.get(w, ())), 0))
                elif w in on_stack:
                    if indices[w] < lowlink[v]:
                        lowlink[v] = indices[w]
            else:
                call_stack.pop()
                if call_stack:
                    parent = call_stack[-1][0]
                    if lowlink[v] < lowlink[parent]:
                        lowlink[parent] = lowlink[v]
                if lowlink[v] == indices[v]:
                    comp: list[str] = []
                    while True:
                        w = stack.pop()
                        on_stack.discard(w)
                        comp.append(w)
                        if w == v:
                            break
                    sccs.append(sorted(comp))
    return sccs


def run_lint(
    root: Path, categories: set[str] | None = None
) -> LintReport:
    """Scan *root* and return a LintReport for the requested categories."""
    t0 = time.perf_counter()
    active = set(VALID_CATEGORIES) if categories is None else set(categories)
    resolved_root = root.resolve()
    py_files = sorted(resolved_root.rglob("*.py"))

    module_to_path: dict[str, Path] = {}
    module_to_rel: dict[str, str] = {}
    for file_path in py_files:
        mod_name = _module_name_from_path(resolved_root, file_path)
        module_to_path[mod_name] = file_path
        module_to_rel[mod_name] = file_path.relative_to(resolved_root).as_posix()

    known_modules = set(module_to_path)
    adj: dict[str, set[str]] = {mod: set() for mod in known_modules}
    edge_lines: dict[tuple[str, str], int] = {}
    violations: list[Violation] = []

    for mod_name in sorted(known_modules):
        file_path = module_to_path[mod_name]
        rel_path = module_to_rel[mod_name]
        rel_parts = file_path.relative_to(resolved_root).parts
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(file_path))

        src_layer = classify_module(mod_name)
        if "layers" in active and src_layer is None:
            violations.append(
                Violation(
                    rel_path,
                    1,
                    "layers",
                    f"unclassified module `{mod_name}` outside L0..L5 DSM table",
                )
            )

        in_campaign = len(rel_parts) >= 2 and rel_parts[0] == "agent" and rel_parts[1] == "campaign"
        check_sys = "sys_modules" in active
        check_refl = "reflection" in active
        need_full = (check_sys or check_refl) and (
            any(tok in source for tok in _BASE_SCAN_TOKENS)
            or (check_refl and in_campaign and "signature" in source)
        )
        if need_full:
            import_nodes, file_violations = _scan_full_ast(
                rel_path, rel_parts, tree, check_sys, check_refl
            )
            violations.extend(file_violations)
        else:
            import_nodes = _iter_import_nodes(tree)

        imports = _extract_internal_imports(mod_name, import_nodes, known_modules)
        for lineno, dst_mod in imports:
            if dst_mod in known_modules:
                adj[mod_name].add(dst_mod)
                edge_lines.setdefault((mod_name, dst_mod), lineno)

            dst_layer = classify_module(dst_mod)
            if src_layer is None or dst_layer is None:
                continue
            if (src_layer == 3 and dst_layer == 4) or (src_layer == 4 and dst_layer == 3):
                if "l3_l4" in active:
                    violations.append(
                        Violation(
                            rel_path,
                            lineno,
                            "l3_l4",
                            f"L{src_layer}/L{dst_layer} orthogonality violation: "
                            f"`{mod_name}` -> `{dst_mod}`",
                        )
                    )
            elif dst_layer > src_layer and "layers" in active:
                violations.append(
                    Violation(
                        rel_path,
                        lineno,
                        "layers",
                        f"upper-triangular import L{src_layer}->L{dst_layer}: "
                        f"`{mod_name}` -> `{dst_mod}`",
                    )
                )

    if "scc" in active:
        for mod_name in sorted(known_modules):
            if mod_name in adj[mod_name]:
                violations.append(
                    Violation(
                        module_to_rel[mod_name],
                        edge_lines.get((mod_name, mod_name), 1),
                        "scc",
                        f"self-import cycle on `{mod_name}`",
                    )
                )
        for comp in _tarjan_sccs(sorted(known_modules), adj):
            if len(comp) > 1:
                head = comp[0]
                line = 1
                for nxt in comp[1:]:
                    if (head, nxt) in edge_lines:
                        line = edge_lines[(head, nxt)]
                        break
                violations.append(
                    Violation(
                        module_to_rel[head],
                        line,
                        "scc",
                        f"SCC cycle ({len(comp)} modules): " + " <-> ".join(comp),
                    )
                )

    violations.sort(key=lambda item: (item.path, item.line, item.category, item.message))
    counts = {cat: 0 for cat in VALID_CATEGORIES}
    for item in violations:
        counts[item.category] += 1
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return LintReport(
        root=str(resolved_root),
        files_scanned=len(py_files),
        elapsed_ms=round(elapsed_ms, 2),
        total_violations=len(violations),
        counts_by_category=counts,
        violations=violations,
    )


def _default_root() -> Path:
    repo_root = Path(__file__).resolve().parent.parent
    return repo_root / "targets" / "ascend" / "tilefoundry_ascend"


def _parse_categories(raw: str | None) -> set[str] | None:
    if raw is None:
        return None
    parts = [part.strip() for part in raw.split(",") if part.strip()]
    if not parts:
        raise ValueError("--only requires at least one category")
    invalid = [part for part in parts if part not in VALID_CATEGORIES]
    if invalid:
        raise ValueError(
            f"unknown categories {invalid}; expected subset of {list(VALID_CATEGORIES)}"
        )
    return set(parts)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the DSM lower-triangular architecture linter."""
    parser = argparse.ArgumentParser(
        description="Check tilefoundry_ascend against the 6-layer lower-triangular DSM."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=_default_root(),
        help="Path to tilefoundry_ascend package root.",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated categories: layers,l3_l4,scc,sys_modules,reflection.",
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
    try:
        categories = _parse_categories(args.only)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    report = run_lint(args.root, categories)
    if args.json:
        payload = {
            "root": report.root,
            "files_scanned": report.files_scanned,
            "elapsed_ms": report.elapsed_ms,
            "total_violations": report.total_violations,
            "counts_by_category": report.counts_by_category,
            "violations": [asdict(item) for item in report.violations],
        }
        print(json.dumps(payload, indent=2))
    else:
        for item in report.violations:
            print(f"{item.path}:{item.line}: [{item.category}] {item.message}")
        summary = ", ".join(
            f"{cat}={report.counts_by_category[cat]}" for cat in VALID_CATEGORIES
        )
        print(
            f"DSM lint scanned {report.files_scanned} files in {report.elapsed_ms:.2f}ms: "
            f"{report.total_violations} violation(s) ({summary})"
        )
    return 1 if report.total_violations else 0


if __name__ == "__main__":
    raise SystemExit(main())
