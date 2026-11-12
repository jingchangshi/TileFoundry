"""Structural identity over ACTUAL IR nodes, including real Op attributes.

The compiler compares an entry with itself before and after a transaction, so the
identity must be a property of the program's real content. Two rules make this
different from a name/kind digest:

1. An Op instance is not a dataclass. Its published attribute parameters
   (Op.params() with kind == "attribute") ARE its program content: a
   Vadds(value=1.0) and a Vadds(value=7.0) are different programs, as are
   Vcvt(part=0) / Vcvt(part=1) and Pset(elem_width=32) / Pset(elem_width=16).
   Reading only kind and name would make every instruction differ by attribute
   collide.
2. A Var is a BINDER, not a string. The walk assigns every binder its own local
   label (param0:x, let3:x, for1:i, ...) and references are serialized by that
   label. Two program positions spelled with the same name but bound by
   different declarations therefore do not compare equal.

Fields that are evidence ABOUT a run rather than the program (analysis metadata,
authored source locators) and fields that are typed object references rather
than program structure (resolved anchor/buffer refs, normalized stage rows) are
not part of the identity: they legitimately differ between two views of one
program.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import fields, is_dataclass

from tilefoundry.ir.core import Var
from tilefoundry.ir.core.op import op_identifier

__all__ = [
    "function_identity",
    "module_identity",
    "node_identity_payload",
]

#: Program-structure-irrelevant fields, by name. See the module docstring.
_SKIP_FIELDS = frozenset({
    "metadata",                 # analysis evidence, not the program
    "origin",                   # authored source locator
    "normalized_stages",        # resolved rows holding live object references
    "anchor_ref",               # resolved object reference
    "source_ref",               # resolved object reference
    "buffer_ref",               # resolved object reference
    "annotation",               # cached type spelling of a Var
    "_sealed",
    "_display_name",
    "variants",                 # separately-identified specializations
})


def _scalar_text(value) -> str | None:
    """Canonical text for a plain literal, or None when it is not one."""
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "bool:%s" % value
    if isinstance(value, int):
        return "int:%d" % value
    if isinstance(value, float):
        return "float:%r" % value
    if isinstance(value, str):
        return "str:%s" % value
    if isinstance(value, bytes):
        return "bytes:%s" % value.hex()
    return None


def _type_text(type_) -> str:
    """A complete, deterministic spelling of a value's type."""
    if type_ is None:
        return "none"
    return "type:%s" % (type_,)


def _var_text(var: Var, binders: dict[int, str]) -> str:
    label = binders.get(id(var))
    if label is not None:
        return "ref:%s" % label
    return "free:%s" % (var.name,)


def _describe(node, binders: dict[int, str]) -> str:
    scalar = _scalar_text(node)
    if scalar is not None:
        return scalar
    if isinstance(node, Var):
        return _var_text(node, binders)
    if isinstance(node, tuple):
        return "(" + ",".join(_describe(item, binders) for item in node) + ")"
    if isinstance(node, list):
        return "[" + ",".join(_describe(item, binders) for item in node) + "]"
    if isinstance(node, dict):
        return "{" + ",".join(
            "%s:%s" % (_describe(key, binders), _describe(value, binders))
            for key, value in sorted(node.items(), key=lambda pair: repr(pair[0]))
        ) + "}"
    if is_dataclass(node):
        parts = []
        for field in fields(node):
            if field.name in _SKIP_FIELDS:
                continue
            parts.append("%s=%s" % (field.name, _describe(getattr(node, field.name, None), binders)))
        return "%s(%s)" % (type(node).__name__, ";".join(parts))
    # An Op instance: its published attribute parameters are its content.
    declared = getattr(type(node), "params", None)
    if callable(declared):
        try:
            attributes = [param.name for param in declared()
                          if getattr(param, "kind", "") == "attribute"]
        except (TypeError, AttributeError):
            attributes = []
        try:
            name = op_identifier(type(node))
        except (AttributeError, TypeError):
            name = type(node).__name__
        body = ";".join(
            "%s=%s" % (attribute, _describe(getattr(node, attribute, None), binders))
            for attribute in attributes
        )
        return "op:%s(%s)" % (name, body)
    return "object:%s:%r" % (type(node).__name__, node)


def _bind(node, binders: dict[int, str], counter: list[int]) -> None:
    """Record every binder before references are serialized."""
    if node is None:
        return
    if isinstance(node, (tuple, list)):
        for item in node:
            _bind(item, binders, counter)
        return
    if not is_dataclass(node):
        return
    kind = type(node).__name__
    if kind == "LetStmt":
        binders[id(node.var)] = "let%d:%s" % (counter[0], node.var.name)
        counter[0] += 1
    elif kind in ("For", "While"):
        induction = getattr(node, "induction_var", None)
        if induction is not None:
            binders[id(induction)] = "for%d:%s" % (counter[0], induction.name)
            counter[0] += 1
    elif kind == "MeshScope":
        binders[id(node.binding)] = "mesh%d:%s" % (counter[0], node.binding.name)
        counter[0] += 1
    elif kind == "PrimFunction":
        for index, param in enumerate(node.params):
            binders[id(param)] = "param%d:%s" % (index, param.name)
    for field in fields(node):
        if field.name in _SKIP_FIELDS:
            continue
        _bind(getattr(node, field.name, None), binders, counter)


def _describe_target(target) -> str:
    if target is None:
        return "none"
    return "%s/%s" % (type(target).__name__, getattr(target, "name", ""))


def node_identity_payload(function) -> dict:
    """The JSON payload one function's identity is computed over."""
    binders: dict[int, str] = {}
    for index, param in enumerate(getattr(function, "params", ()) or ()):
        binders[id(param)] = "param%d:%s" % (index, param.name)
    _bind(getattr(function, "body", None), binders, [0])
    return {
        "name": getattr(function, "name", None),
        "params": [
            {"label": binders.get(id(param)), "type": _type_text(getattr(param, "type", None))}
            for param in getattr(function, "params", ()) or ()
        ],
        "body": _describe(getattr(function, "body", None), binders),
        "output_count": getattr(function, "output_count", None),
        "target": _describe_target(getattr(function, "target", None)),
    }


def function_identity(function) -> str:
    """A sha256 identity of one function's actual program content."""
    text = json.dumps(node_identity_payload(function), sort_keys=True)
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def module_identity(module) -> str:
    """A sha256 identity over every actual function of a module."""
    payload = [node_identity_payload(function)
               for function in getattr(module, "functions", ()) or ()]
    text = json.dumps(payload, sort_keys=True)
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()
