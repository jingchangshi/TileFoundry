
import ast

def replace_func_node(module_source, func_name, new_func_source):
    orig_tree = ast.parse(module_source)
    
    # Find the function node to replace
    class FuncFinder(ast.NodeTransformer):
        def __init__(self, func_name, replacement_node):
            self.func_name = func_name
            self.replacement_node = replacement_node
            self.replaced = False

        def visit_FunctionDef(self, node):
            if node.name == self.func_name:
                self.replaced = True
                return self.replacement_node
            return node

    cand_tree = ast.parse(new_func_source)
    cand_node = None
    for node in cand_tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            cand_node = node
            break
    
    if cand_node is None:
        raise ValueError(f"Function {func_name} not found in candidate source.")
        
    transformer = FuncFinder(func_name, cand_node)
    new_tree = transformer.visit(orig_tree)
    
    if not transformer.replaced:
        raise ValueError(f"Function {func_name} not found in original source.")
        
    return ast.unparse(new_tree)

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

print(replace_func_node(orig, "foo", cand))
