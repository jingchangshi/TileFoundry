"""M2b closure checks on typed IR objects.

Structural facts come from the loaded TIR tree (params, MeshScope, Evaluate
nodes and operand types), not source text.  Negative cases accept only the
parse/verify error they are meant to raise.  CPU equality executes the authored
HIR when torch is importable, otherwise it is reported NOT_RUN.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from tilefoundry.cli.source import load_authored_ir
from tilefoundry.parser.ast_pattern import ParseError
from tilefoundry.ir.core.errors import VerifyError
from tilefoundry.visitor_registry.verify import verify_module, verify_prim_function


def _walk(stmt, found):
    name = type(stmt).__name__
    if name in ("Sequential",):
        for child in stmt.body:
            _walk(child, found)
        return
    if name == "MeshScope":
        found["meshes"].append(stmt.mesh)
        _walk(stmt.body, found)
        return
    if name == "Evaluate":
        found["evaluates"].append(stmt)
        return
    body = getattr(stmt, "body", None)
    if body is None:
        return
    if isinstance(body, (tuple, list)):
        for child in body:
            _walk(child, found)
    else:
        _walk(body, found)


def _op_of(stmt):
    for attr in ("callable",):
        value = getattr(stmt, attr, None)
        if value is not None and type(value).__name__ != "type":
            return value
    return None


def verify_artifact(path: Path, entry: str):
    m = load_authored_ir(f"{path}:{entry}")
    verify_module(m)
    verify_prim_function(m.entry_function(), module_fns=m)
    return m


def main(artifact: str, entry: str) -> int:
    p = Path(artifact)
    out = {"artifact": str(p), "entry": entry, "structural": {}, "negatives": {}}
    m = verify_artifact(p, entry)
    fn = m.entry_function()
    params = {q.name: q.type for q in fn.params}
    out["structural"]["params"] = sorted(params)
    out["structural"]["in_gmem"] = str(getattr(params.get("x"), "storage", "")) == "2"
    out["structural"]["out_gmem"] = str(getattr(params.get("out"), "storage", "")) == "2"
    found = {"meshes": [], "evaluates": []}
    _walk(fn.body, found)
    out["structural"]["mesh_names"] = [list(mesh.names) if hasattr(mesh, "names") else str(mesh) for mesh in found["meshes"]]
    copies = []
    for stmt in found["evaluates"]:
        op = getattr(stmt, "callable", None)
        if type(op).__name__ == "Copy":
            copies.append(stmt)
    out["structural"]["copy_count"] = len(copies)
    storages = set()
    for stmt in copies:
        for arg in getattr(stmt, "args", ()) or ():
            storage = getattr(getattr(arg, "type", None), "storage", None)
            if storage is not None:
                storages.add(int(storage))
    out["structural"]["copy_operand_storages"] = sorted(storages)
    out["structural"]["has_smem_staging"] = 3 in storages  # StorageKind.SMEM
    out["structural"]["has_aiv_mesh"] = any("aiv" in (mesh.names if hasattr(mesh, "names") else ()) for mesh in found["meshes"])

    scratch = p.parent

    def negative(name, mutated, expected):
        path = scratch / f"__negative_{name}.py"
        path.write_text(mutated)
        try:
            verify_artifact(path, entry)
        except expected as exc:
            out["negatives"][name] = {"rejected": True, "error": type(exc).__name__}
        except Exception as exc:
            out["negatives"][name] = {"rejected": False, "unexpected": repr(exc)[:160]}
        else:
            out["negatives"][name] = {"rejected": False, "unexpected": "accepted"}

    text = p.read_text()
    negative("shape_layout_mismatch", text.replace("shape=(64,)", "shape=(32,)"), VerifyError)
    negative("unknown_storage", text.replace('storage="smem"', 'storage="l2"'), ParseError)

    structural_ok = all(v for k, v in out["structural"].items() if isinstance(v, bool))
    negatives_ok = all(v.get("rejected") for v in out["negatives"].values())
    try:
        import torch
        from tilefoundry.evaluator import evaluate
        authored = load_authored_ir(str(Path(__file__).resolve().parents[1] / "copy_slice.py") + ":CopySlice")
        x = torch.randn(64)
        result = evaluate(authored.entry_function(), x)
        data = result.data if hasattr(result, "data") else result
        equal = bool(torch.equal(data, x))
        out["cpu_semantics"] = {"status": "PASS" if equal else "FAIL", "equal": equal}
        cpu_ok = equal
    except Exception as exc:
        out["cpu_semantics"] = {"status": "NOT_RUN", "reason": repr(exc)[:200]}
        cpu_ok = False
    out["static_ok"] = bool(structural_ok and negatives_ok)
    out["ok"] = bool(out["static_ok"] and cpu_ok)
    print(json.dumps(out, indent=2))
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
