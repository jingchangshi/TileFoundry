
from typing import Sequence
import ast

def _func_line_span(node: ast.FunctionDef) -> tuple[int, int]:
    start = min((dec.lineno for dec in node.decorator_list), default=node.lineno)
    end = node.end_lineno or node.lineno
    return start, end

def _find_top_level_func(tree: ast.Module, names: Sequence[str]) -> ast.FunctionDef | None:
    by_name = {
        node.name: node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }
    for name in names:
        if name in by_name:
            return by_name[name]
    return None

def apply_ast_patch(module_source: str, target_func: str, candidate_dsl: str) -> str:
    orig_tree = ast.parse(module_source)
    cand_tree = ast.parse(candidate_dsl)
    
    cand_node = _find_top_level_func(cand_tree, (target_func,))
    cand_lines = candidate_dsl.splitlines(keepends=True)
    c_start, c_end = _func_line_span(cand_node)
    replacement = "".join(cand_lines[c_start - 1 : c_end])
    if not replacement.endswith("\n"):
        replacement += "\n"
        
    orig_node = _find_top_level_func(orig_tree, (target_func,))
    o_start, o_end = _func_line_span(orig_node)
    orig_lines = module_source.splitlines(keepends=True)
    
    patched = "".join([*orig_lines[: o_start - 1], replacement, *orig_lines[o_end:]])
    ast.parse(patched)
    return patched

# Test
orig = """
def foo():
    return 1

def bar():
    return 2
"""
cand = """
def foo():
    return 3
"""
from typing import Sequence
print(apply_ast_patch(orig, "foo", cand))
