"""Deterministic AST static semantic gate for TileLang Ascend kernels."""

from __future__ import annotations

import ast
from dataclasses import dataclass

_SIMD_NO_OFF_CALLEES: frozenset[str] = frozenset(
    {
        "T.simd.vld",
        "T.simd.vsts",
        "T.simd.pld",
        "T.simd.pst",
    }
)
_LEAF_SCOPE_WITH_NAMES: frozenset[str] = frozenset({"T.Task", "T.SimdVF", "T.SimtVF"})
_TILE_OP_NAMES: frozenset[str] = frozenset({"T.copy", "T.gemm"})


@dataclass(frozen=True)
class SemanticViolation:
    """Single deterministic AST semantic contract violation."""

    code: str
    lineno: int
    message: str


def _dotted_name(expr: ast.expr) -> str:
    parts: list[str] = []
    cur = expr
    while isinstance(cur, ast.Attribute):
        parts.append(cur.attr)
        cur = cur.value
    if isinstance(cur, ast.Name):
        parts.append(cur.id)
        return ".".join(reversed(parts))
    return ""


def _call_qualname(expr: ast.expr) -> str:
    if isinstance(expr, ast.Call):
        return _dotted_name(expr.func)
    return ""


def _is_prim_func_decorator(dec: ast.expr) -> bool:
    target = dec.func if isinstance(dec, ast.Call) else dec
    return isinstance(target, ast.Attribute) and target.attr == "prim_func"


def _top_level_bound_names(stmt: ast.stmt) -> tuple[str, ...]:
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return (stmt.name,)
    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        return (stmt.target.id,)
    if isinstance(stmt, ast.Assign):
        return tuple(t.id for t in stmt.targets if isinstance(t, ast.Name))
    return ()


def _has_enable_offset_annotation(expr: ast.expr) -> bool:
    if not isinstance(expr, ast.Call):
        return False
    for kw in expr.keywords:
        if kw.arg != "annotations":
            continue
        if isinstance(kw.value, ast.Dict):
            for key, val in zip(kw.value.keys, kw.value.values):
                if (
                    isinstance(key, ast.Constant)
                    and key.value == "enable_offset"
                    and isinstance(val, ast.Constant)
                    and val.value is True
                ):
                    return True
        if isinstance(kw.value, ast.Call) and _dotted_name(kw.value.func) == "dict":
            for inner_kw in kw.value.keywords:
                if (
                    inner_kw.arg == "enable_offset"
                    and isinstance(inner_kw.value, ast.Constant)
                    and inner_kw.value.value is True
                ):
                    return True
    return False


def _stage_index_gt_zero(expr: ast.expr) -> bool:
    if not isinstance(expr, ast.Call):
        return False
    arg_expr: ast.expr | None = expr.args[0] if expr.args else None
    if arg_expr is None:
        for kw in expr.keywords:
            if kw.arg == "stage":
                arg_expr = kw.value
                break
    if (
        isinstance(arg_expr, ast.Constant)
        and isinstance(arg_expr.value, int)
        and not isinstance(arg_expr.value, bool)
    ):
        return arg_expr.value > 0
    return False


class TileLangSemanticGate(ast.NodeVisitor):
    """Static AST checker for TileLang Ascend scope, loop, stage, and SIMD contracts."""

    def __init__(self, protected_exports: frozenset[str] = frozenset()) -> None:
        self._protected = protected_exports
        self._in_prim_func = False
        self._parallel_depth = 0
        self._simtvf_depth = 0
        self._leaf_scope_depth = 0
        self._pipelined_offset_stack: list[bool] = []
        self.violations: list[SemanticViolation] = []

    def check_module(self, tree: ast.Module) -> tuple[SemanticViolation, ...]:
        """Validate module AST and return deterministic violations in source order."""
        if self._protected:
            for stmt in tree.body:
                for name in _top_level_bound_names(stmt):
                    if name in self._protected:
                        self.violations.append(
                            SemanticViolation(
                                code="PROTECTED_SYMBOL_REBIND",
                                lineno=stmt.lineno,
                                message=(
                                    f"Top-level symbol '{name}' is in protected_exports "
                                    "and cannot be rebound by candidate code."
                                ),
                            )
                        )
        self.visit(tree)
        return tuple(self.violations)

    def _visit_func_node(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        if self._in_prim_func:
            self.violations.append(
                SemanticViolation(
                    code="NESTED_DEF_IN_PRIM_FUNC",
                    lineno=node.lineno,
                    message=(
                        f"Move helper '{node.name}' outside @T.prim_func "
                        "and pass buffers/scalars explicitly."
                    ),
                )
            )
            return
        is_prim = any(_is_prim_func_decorator(dec) for dec in node.decorator_list)
        prev_prim = self._in_prim_func
        if is_prim:
            self._in_prim_func = True
        try:
            self.generic_visit(node)
        finally:
            self._in_prim_func = prev_prim

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Check for nested defs inside @T.prim_func and track prim_func scope."""
        self._visit_func_node(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Check for nested async defs inside @T.prim_func and track prim_func scope."""
        self._visit_func_node(node)

    def visit_For(self, node: ast.For) -> None:
        """Validate T.Pipelined and T.Parallel nesting and offset annotations."""
        iter_name = _call_qualname(node.iter)
        entered_parallel = False
        entered_pipelined = False
        if iter_name == "T.Parallel":
            entered_parallel = True
            self._parallel_depth += 1
        elif iter_name == "T.Pipelined":
            entered_pipelined = True
            if self._parallel_depth > 0:
                self.violations.append(
                    SemanticViolation(
                        code="PIPELINED_INSIDE_PARALLEL",
                        lineno=node.lineno,
                        message="T.Pipelined loop cannot be nested inside T.Parallel.",
                    )
                )
            self._pipelined_offset_stack.append(_has_enable_offset_annotation(node.iter))
        try:
            self.generic_visit(node)
        finally:
            if entered_pipelined:
                self._pipelined_offset_stack.pop()
            if entered_parallel:
                self._parallel_depth -= 1

    def visit_With(self, node: ast.With) -> None:
        """Validate T.Stage, T.Task, T.SimdVF, and T.SimtVF context nesting."""
        pushed_pipelines = 0
        added_leaf = 0
        added_simtvf = 0
        try:
            for item in node.items:
                ctx_name = _call_qualname(item.context_expr)
                if ctx_name == "T.Pipelined":
                    if self._parallel_depth > 0:
                        self.violations.append(
                            SemanticViolation(
                                code="PIPELINED_INSIDE_PARALLEL",
                                lineno=node.lineno,
                                message="T.Pipelined cannot be nested inside T.Parallel.",
                            )
                        )
                    self._pipelined_offset_stack.append(
                        _has_enable_offset_annotation(item.context_expr)
                    )
                    pushed_pipelines += 1
                elif ctx_name in _LEAF_SCOPE_WITH_NAMES:
                    self._leaf_scope_depth += 1
                    added_leaf += 1
                    if ctx_name == "T.SimtVF":
                        self._simtvf_depth += 1
                        added_simtvf += 1
                elif ctx_name == "T.Stage":
                    if self._leaf_scope_depth > 0 or self._parallel_depth > 0:
                        self.violations.append(
                            SemanticViolation(
                                code="STAGE_INSIDE_LEAF_SCOPE",
                                lineno=node.lineno,
                                message=(
                                    "T.Stage cannot be nested inside T.Task, "
                                    "T.SimdVF, T.SimtVF, or T.Parallel."
                                ),
                            )
                        )
                    if _stage_index_gt_zero(item.context_expr) and not (
                        self._pipelined_offset_stack and self._pipelined_offset_stack[-1]
                    ):
                        self.violations.append(
                            SemanticViolation(
                                code="NONZERO_STAGE_WITHOUT_ENABLE_OFFSET",
                                lineno=node.lineno,
                                message=(
                                    "Nonzero T.Stage(k) requires enclosing "
                                    "T.Pipelined(..., annotations={'enable_offset': True})."
                                ),
                            )
                        )
            self.generic_visit(node)
        finally:
            self._simtvf_depth -= added_simtvf
            self._leaf_scope_depth -= added_leaf
            for _ in range(pushed_pipelines):
                self._pipelined_offset_stack.pop()

    def visit_Call(self, node: ast.Call) -> None:
        """Validate T.copy/T.gemm parallel placement and T.simd.* off= prohibitions."""
        callee = _dotted_name(node.func)
        if callee in _TILE_OP_NAMES and self._parallel_depth > 0 and self._simtvf_depth == 0:
            self.violations.append(
                SemanticViolation(
                    code="TILE_OP_INSIDE_PARALLEL",
                    lineno=node.lineno,
                    message=(
                        f"{callee} cannot appear inside T.Parallel outside T.SimtVF."
                    ),
                )
            )
        if callee in _SIMD_NO_OFF_CALLEES and any(kw.arg == "off" for kw in node.keywords):
            self.violations.append(
                SemanticViolation(
                    code="SIMD_FORBIDDEN_OFF_KWARG",
                    lineno=node.lineno,
                    message=(
                        f"{callee} does not accept off=; "
                        "index the buffer slice directly (buf[offset])."
                    ),
                )
            )
        self.generic_visit(node)


def check_tilelang_semantic_contracts(
    source: str,
    *,
    protected_exports: frozenset[str] = frozenset(),
) -> tuple[SemanticViolation, ...]:
    """Run deterministic AST semantic checks on TileLang Python source."""
    tree = ast.parse(source)
    gate = TileLangSemanticGate(protected_exports=protected_exports)
    return gate.check_module(tree)


__all__ = [
    "SemanticViolation",
    "TileLangSemanticGate",
    "check_tilelang_semantic_contracts",
]
