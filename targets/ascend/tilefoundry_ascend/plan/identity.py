"""Canonical identity bytes over ACTUAL program content.

One encoder; the four content boundaries (source / target / problem / plan)
share it. The encoder refuses instead of guessing:

- every node carries a type tag; there is NO repr or str(type) fallback;
- a Var is a LEXICAL binder. Params, LetStmt vars, For induction vars and
  MeshScope bindings are labelled where they actually occur; a use of a Var
  that is not lexically in scope is a dangling reference and is refused. The
  name is never an identity. LetStmt.value encodes in the OUTER environment;
  a new binder is visible to the following siblings (and to a For/Mesh body),
  and If arms get independent copies;
- floats are IEEE-754 binary64 big-endian hex (so -0.0/+0.0/inf/NaN bit
  patterns are distinct); ints are exact decimal strings; bytes are hex;
- sets are sorted by the canonical bytes of their elements, dicts by the
  canonical bytes of their keys; the same key with two different values is
  refused rather than silently overwritten;
- an Op instance is its registered identifier plus its declared ATTRIBUTE
  parameters. Its input parameters are NOT instance content: the operands
  ride on Evaluate.args / Call.args, which are encoded there;
- diagnostic-only fields (loc, metadata, annotation, PipelineSpec.origin,
  normalized_stages) are program evidence, not program content;
- isl relations are read POSITIONALLY through the dimension API (per basic
  map, exact integer/rational coefficients). No name string or repr enters
  the bytes, so an alpha-renamed relation with the same positions encodes
  identically while a re-bound AccessRelation.values entry does not;
- Target identity queries the ACTUAL instance's facts; unavailable/unknown
  facts are encoded explicitly and never backfilled.

Anything without a declared encoding raises IdentityRefusal with the field
path; there is no fallback branch.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import struct
from dataclasses import fields, is_dataclass
from enum import Enum

import isl

from tilefoundry.ir.core import Var
from tilefoundry.ir.core.op import op_identifier
from tilefoundry.visitor_registry.access_relation import (AccessRelation,
                                                          _parameter_names)

__all__ = [
    "IDENTITY_ENCODING_VERSION",
    "source_labels",
    "IDENTITY_UNSUPPORTED_VALUE",
    "IdentityRefusal",
    "entry_canonical_bytes",
    "function_identity",
    "module_bytes",
    "module_identity",
    "node_identity_payload",
    "source_bytes",
    "target_bytes",
]

#: Bumping this invalidates every stored anchor. It is not runtime dispatch.
#: Bumped to 2 by task-47: the problem recipe now binds the FULL
#: content of every record (SOURCE_SNAPSHOT_DESIGN §4) instead of id lists, so
#: every anchor computed under version 1 describes a weaker claim than the one
#: the code now makes and must be invalidated. This is the design's invalidation
#: mechanism, not a runtime version dispatch.
IDENTITY_ENCODING_VERSION = 2
IDENTITY_UNSUPPORTED_VALUE = "IDENTITY_UNSUPPORTED_VALUE"

_STMT_KINDS = frozenset({
    "Sequential", "LetStmt", "For", "If", "MeshScope", "ExecutionScope",
    "Evaluate", "Return",
})
_STMT_MODULES = "tilefoundry.ir.tir.stmts"

#: Fields that are evidence about a program, not the program (see docstring).
_DIAGNOSTIC_FIELDS = frozenset({
    "loc", "metadata", "annotation", "origin", "normalized_stages",
    # View.ssa declares itself an in-run HIR SSA identity that is never
    # serialized, and measured across three fresh processes of the same
    # program it differed every run: it is a memory address. Encoding it would
    # put an object address into an identity, the exact defect removed from
    # plan bytes in task-44. A View is still bound by content: its address
    # recipe is base/starts/sizes/type/stem/whole_tensor.
    "ssa",
})

#: Design section 3: a legal pure-dimension Op computes loop bounds and IS
#: program content, but it is not a registered instruction: it carries no
#: _op_schema and no reference_name, so op_identifier() cannot name it. The
#: family is declared here explicitly and encoded by module-qualified class
#: identity. Input operands still ride on the enclosing Call.args (encoded
#: there); declared attribute parameters are read off the instance.
_DIM_OP_MODULES = "tilefoundry.ir.types.dim"
_DIM_OP_KINDS = frozenset({
    "DimAdd", "DimSub", "DimMul", "DimFloorDiv", "DimMod", "DimMin", "DimMax",
    "DimConst", "DimVar",
})


def _op_identity(kind, node, path: str) -> tuple:
    """Explicit naming, in priority order: registered schema, declared
    reference_name, then the declared pure-dimension family. No str(type)."""
    if getattr(kind, "_op_schema", None) is not None:
        return "op", "%s.%s" % (kind._op_schema.dialect, kind._op_schema.name)
    reference = getattr(kind, "reference_name", "")
    if reference:
        return "op", reference
    if kind.__module__ == _DIM_OP_MODULES and kind.__name__ in _DIM_OP_KINDS:
        return "dim_op", "%s.%s" % (kind.__module__, kind.__qualname__)
    raise IdentityRefusal(path,
                          "an Op is neither registered nor in the declared pure-dimension "
                          "family: %s.%s" % (kind.__module__, kind.__qualname__))


class IdentityRefusal(ValueError):
    """No canonical bytes exist for this value at this path."""

    def __init__(self, path: str, detail: str) -> None:
        self.code = IDENTITY_UNSUPPORTED_VALUE
        self.path = path
        super().__init__("%s: %s (path=%s)" % (IDENTITY_UNSUPPORTED_VALUE, detail, path))


def _dump(payload) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def _tag(tag: str, payload) -> dict:
    return {"#": tag, "v": payload, "enc": IDENTITY_ENCODING_VERSION}


def _op_instance(node) -> bool:
    declared = getattr(type(node), "params", None)
    return callable(declared) and not is_dataclass(node) and not isinstance(node, type)


def _encode(node, env: dict, path: str, depth: int):
    if depth > 64:
        raise IdentityRefusal(path, "value nesting exceeds the supported depth")
    if node is None:
        return _tag("none", None)
    kind = type(node)
    if kind is bool:
        return _tag("bool", bool(node))
    if kind is int:
        return _tag("int", str(int(node)))
    if kind is float:
        return _tag("float", struct.pack(">d", node).hex())
    if kind is str:
        return _tag("str", node)
    if kind is bytes:
        return _tag("bytes", bytes(node).hex())
    if isinstance(node, Var):
        label = env.get(id(node))
        if label is None:
            raise IdentityRefusal(path, "a Var is used outside its lexical scope")
        return _tag("ref", {"label": label,
                            "type": _encode(node.type, env, path + ".type", depth + 1),
                            "is_const": bool(node.is_const)})
    if isinstance(node, (tuple, list)):
        return _tag("seq", [_encode(item, env, "%s[%d]" % (path, index), depth + 1)
                            for index, item in enumerate(node)])
    if isinstance(node, (set, frozenset)):
        rows = sorted({_dump(_encode(item, env, "%s{elem}" % path, depth + 1))
                       for item in node})
        return _tag("set", [json.loads(row) for row in rows])
    if isinstance(node, dict):
        rows = []
        for key, value in node.items():
            encoded_key = _encode(key, env, path + ".key", depth + 1)
            rows.append((_dump(encoded_key), encoded_key,
                         _encode(value, env, path + ".value", depth + 1)))
        rows.sort(key=lambda row: row[0])
        merged = []
        for raw_key, encoded_key, encoded_value in rows:
            if merged and merged[-1][0] == raw_key:
                if merged[-1][2] != encoded_value:
                    raise IdentityRefusal(path, "two distinct values share one canonical key")
                continue
            merged.append((raw_key, encoded_key, encoded_value))
        return _tag("map", [[key, value] for _, key, value in merged])
    if isinstance(node, Enum):
        return _tag("enum", {"type": "%s.%s" % (kind.__module__, kind.__qualname__),
                             "name": node.name,
                             "value": _encode(node.value, env, path + ".value", depth + 1)})
    if isinstance(node, AccessRelation):
        return _tag("access_relation", _encode_access_relation(node, env, path, depth))
    if kind.__module__ == "isl":
        if kind.__name__ == "map":
            return _tag("isl_map", _encode_isl_map(node, path))
        raise IdentityRefusal(path, "no positional encoding for isl.%s" % kind.__name__)
    if _op_instance(node):
        tag, identifier = _op_identity(kind, node, path)
        attributes = []
        for param in kind.params():
            if param.kind != "attribute":
                continue  # operands ride on Evaluate/Call args, not on the instance
            attributes.append({
                "name": param.name,
                "value": _encode(getattr(node, param.name, None), env,
                                 "%s.op.%s" % (path, param.name), depth + 1),
            })
        return _tag(tag, {"id": identifier, "attributes": attributes})
    if is_dataclass(node):
        payload = {}
        for field in fields(node):
            if field.name in _DIAGNOSTIC_FIELDS:
                continue
            payload[field.name] = _encode(getattr(node, field.name, None), env,
                                          "%s.%s" % (path, field.name), depth + 1)
        return _tag("dataclass", {"type": "%s.%s" % (kind.__module__, kind.__qualname__),
                                  "fields": payload})
    raise IdentityRefusal(path, "no canonical encoding for %s.%s" % (kind.__module__, kind.__qualname__))


def _positional_rows(container, path: str):
    rows = []
    for dim_type in (isl.dim_type.CST, isl.dim_type.PARAM, isl.dim_type.IN,
                     isl.dim_type.OUT, isl.dim_type.DIV):
        try:
            count = container.dim(dim_type)
        except (AttributeError, isl.Error) as error:
            raise IdentityRefusal(path, "isl dimensions are not readable: %s" % error) from error
        for index in range(count):
            try:
                value = container.get_coefficient_val(dim_type, index)
                rows.append({"dim": dim_type.name, "index": index,
                             "num": str(value.num_si()), "den": str(value.den_si())})
            except (AttributeError, isl.Error) as error:
                raise IdentityRefusal(path, "isl coefficient is not readable: %s" % error) from error
    return rows


def _encode_isl_map(node, path: str):
    try:
        dims = {dt.name: node.dim(dt) for dt in
                (isl.dim_type.PARAM, isl.dim_type.IN, isl.dim_type.OUT, isl.dim_type.DIV)}
        listed = node.get_basic_map_list()
        basic = []
        for index in range(listed.n_basic_map()):
            item = listed.at(index)
            constraints = item.constraint_list()
            rows = []
            for position in range(constraints.n_constraint()):
                constraint = constraints.at(position)
                rows.append({"equality": bool(constraint.is_equality()),
                             "coefficients": _positional_rows(
                                 constraint, "%s.basic[%d].c[%d]" % (path, index, position))})
            basic.append({"dims": {dt.name: item.dim(dt) for dt in
                                   (isl.dim_type.PARAM, isl.dim_type.IN,
                                    isl.dim_type.OUT, isl.dim_type.DIV)},
                          "constraints": rows})
        if not basic:
            raise IdentityRefusal(path, "an empty relation carries no positional content")
        return {"dims": dims, "basic_maps": basic}
    except IdentityRefusal:
        raise
    except (AttributeError, isl.Error, TypeError) as error:
        raise IdentityRefusal(path, "isl relation is not positionally readable: %s" % error) from error


def _encode_access_relation(relation, env, path: str, depth: int):
    names = list(_parameter_names(relation.relation))
    if set(names) != set(relation.values):
        raise IdentityRefusal(path, "relation parameter names and bindings disagree")
    params = relation.relation.dim(isl.dim_type.PARAM)
    if len(names) != params:
        raise IdentityRefusal(path, "relation parameter names do not map to every position")
    positions = []
    for name in names:
        index = relation.relation.find_dim_by_name(isl.dim_type.PARAM, name)
        if index < 0 or index in [p[0] for p in positions]:
            raise IdentityRefusal(path, "relation parameter position is ambiguous: %r" % name)
        positions.append((index, name))
    bindings = [{"position": index,
                 "value": _encode(relation.values[name], env,
                                  "%s.values[%s]" % (path, name), depth + 1)}
                for index, name in sorted(positions)]
    return {"relation": _encode_isl_map(relation.relation, path + ".relation"),
            "bindings": bindings,
            "may_access": bool(relation.may_access),
            "affine": _encode(relation.affine, env, path + ".affine", depth + 1)}


class _Scope:
    """One shared occurrence counter; binder labels are positional, never names."""

    def __init__(self) -> None:
        self.statement = 0

    def occurrence(self) -> str:
        label = "s%d" % self.statement
        self.statement += 1
        return label


def _statement_kind(node, path: str) -> str:
    kind = type(node)
    if kind.__module__ != _STMT_MODULES or kind.__name__ not in _STMT_KINDS:
        raise IdentityRefusal(path, "unsupported statement %s.%s"
                              % (kind.__module__, kind.__name__))
    return kind.__name__


def _encode_statements(items, env, scope, path, depth):
    """Encode a straight-line body; LetStmt binders thread to later siblings."""
    encoded = []
    for index, item in enumerate(items):
        encoded.append(_encode_statement(item, env, scope, "%s[%d]" % (path, index), depth))
    return encoded


def _encode_statement(node, env, scope, path, depth):
    if depth > 48:
        raise IdentityRefusal(path, "statement nesting exceeds the supported depth")
    kind = _statement_kind(node, path)
    occurrence = scope.occurrence()
    if kind == "Sequential":
        return {"#": "stmt", "kind": kind, "occurrence": occurrence,
                "body": _encode_statements(node.body, env, scope, path + ".body", depth + 1)}
    if kind == "LetStmt":
        value = _encode(node.value, env, path + ".value", depth + 1)  # outer scope
        env[id(node.var)] = occurrence
        return {"#": "stmt", "kind": kind, "occurrence": occurrence, "value": value,
                "definition": {"label": occurrence,
                               "type": _encode(node.var.type, env, path + ".var.type", depth + 1)}}
    if kind == "For":
        bounds = {name: _encode(getattr(node, name), env, "%s.%s" % (path, name), depth + 1)
                  for name in ("start", "stop", "step")}
        child = dict(env)
        label = occurrence
        child[id(node.induction_var)] = label
        return {"#": "stmt", "kind": kind, "occurrence": occurrence, "bounds": bounds,
                "induction": {"label": label,
                              "type": _encode(node.induction_var.type, child,
                                              path + ".iv.type", depth + 1)},
                "pipeline": _encode(getattr(node, "pipeline", None), child,
                                    path + ".pipeline", depth + 1),
                "body": _encode_statements(
                    [node.body] if not isinstance(node.body, (list, tuple)) else node.body,
                    child, scope, path + ".body", depth + 1)}
    if kind == "MeshScope":
        mesh = _encode(node.mesh, env, path + ".mesh", depth + 1)
        child = dict(env)
        child[id(node.binding)] = occurrence
        return {"#": "stmt", "kind": kind, "occurrence": occurrence, "mesh": mesh,
                "binding": {"label": occurrence,
                            "type": _encode(node.binding.type, child,
                                            path + ".binding.type", depth + 1)},
                "body": _encode_statements(
                    [node.body] if not isinstance(node.body, (list, tuple)) else node.body,
                    child, scope, path + ".body", depth + 1)}
    if kind == "ExecutionScope":
        return {"#": "stmt", "kind": kind, "occurrence": occurrence,
                "execution_kind": _encode(node.kind, env, path + ".kind", depth + 1),
                "body": _encode_statements(
                    [node.body] if not isinstance(node.body, (list, tuple)) else node.body,
                    dict(env), scope, path + ".body", depth + 1)}
    if kind == "If":
        then_env, else_env = dict(env), dict(env)
        return {"#": "stmt", "kind": kind, "occurrence": occurrence,
                "cond": _encode(node.cond, env, path + ".cond", depth + 1),
                "then": _encode_statements([node.then_body], then_env, scope,
                                           path + ".then", depth + 1),
                "else": _encode_statements([node.else_body], else_env, scope,
                                           path + ".else", depth + 1)
                if node.else_body is not None else None}
    payload = {"#": "stmt", "kind": kind, "occurrence": occurrence}
    for field in fields(node):
        if field.name in _DIAGNOSTIC_FIELDS:
            continue
        payload[field.name] = _encode(getattr(node, field.name, None), env,
                                      "%s.%s" % (path, field.name), depth + 1)
    return payload


def _encode_function(function, path: str) -> dict:
    if type(function).__name__ != "PrimFunction":
        raise IdentityRefusal(path, "only a real PrimFunction carries source bytes")
    env: dict = {}
    scope = _Scope()
    params = []
    for index, param in enumerate(getattr(function, "params", ()) or ()):
        label = "p%d" % index
        env[id(param)] = label
        params.append({"label": label,
                       "type": _encode(param.type, env, "%s.params[%d].type" % (path, index), 0),
                       "is_const": bool(param.is_const)})
    body = getattr(function, "body", None)
    statements = body if isinstance(body, (list, tuple)) else [body]
    return {
        "enc": IDENTITY_ENCODING_VERSION,
        "name": _encode(getattr(function, "name", None), env, path + ".name", 0),
        "params": params,
        "output_count": _encode(getattr(function, "output_count", None), env,
                                path + ".output_count", 0),
        "target": _encode_target(getattr(function, "target", None), path + ".target"),
        "body": _encode_statements(statements, env, scope, path + ".body", 0),
    }


def _capability(target, name: str, path: str):
    """Call one declared target capability, or refuse a target that lacks it.

    A missing method is a missing capability: it is named, never swallowed by
    catching the AttributeError it would raise, and never backfilled.
    """
    method = getattr(target, name, None)
    if method is None:
        raise IdentityRefusal(path, "target does not publish %s" % name)
    return method()


def _fact(target, class_path: str, path: str):
    """One queried fact. A DECLINED fact is a structured record, never prose.

    The target's answer "I have no projection for this fact class" is a fact
    about the target, but the sentence itself is human-readable text that can
    change with no change of meaning. Identity bytes carry the refusal kind and
    the fact class name only, so rewording the message cannot move the source,
    problem or plan identity.
    """
    module_name, _, name = class_path.rpartition(".")
    from tilefoundry.target.base import UnsupportedCapabilityError
    cls = getattr(importlib.import_module(module_name), name)
    try:
        return _encode(target.get_facts(cls), {}, "%s.%s" % (path, name), 0)
    except UnsupportedCapabilityError:
        return _tag("declined", _encode(name, {}, path, 0))


def _target_payload(target, path: str) -> dict:
    if target is None:
        return _tag("none", None)
    from tilefoundry.target.facts import MemoryExecutionFacts, TopologyFacts
    classes = {
        "MemoryExecutionFacts": MemoryExecutionFacts,
        "MemoryHierarchyFacts": "tilefoundry_ascend.tf_target.MemoryHierarchyFacts",
        "TopologyFacts": TopologyFacts,
        "TopologyLevelFacts": "tilefoundry_ascend.tf_target.TopologyLevelFacts",
    }
    if getattr(target, "get_facts", None) is None:
        raise IdentityRefusal(path, "target does not publish get_facts")
    facts = {}
    for name in sorted(classes):
        entry = classes[name]
        if isinstance(entry, str):
            facts[name] = _fact(target, entry, path)
        else:
            from tilefoundry.target.base import UnsupportedCapabilityError
            try:
                facts[name] = _encode(target.get_facts(entry), {}, "%s.%s" % (path, name), 0)
            except UnsupportedCapabilityError:
                # Structured: which fact class was declined, never the sentence.
                facts[name] = _tag("declined", _encode(name, {}, path, 0))
    semantics = ({field.name: getattr(target, field.name, None)
                  for field in fields(target)} if is_dataclass(target) else {})
    spaces = dict(sorted(_capability(target, "space_facts", path).items()))
    return {
        "enc": IDENTITY_ENCODING_VERSION,
        "type": _encode("%s.%s" % (type(target).__module__, type(target).__qualname__),
                        {}, path + ".type", 0),
        "name": _encode(getattr(target, "name", None), {}, path + ".name", 0),
        "semantics": _encode(semantics, {}, path + ".semantics", 0),
        "facts": facts,
        "spaces": _encode(spaces, {}, path + ".spaces", 0),
        "events": _encode(_capability(target, "target_event_facts", path),
                          {}, path + ".events", 0),
        "completion_ops": _encode(sorted(cls.__qualname__
                                         for cls in _capability(
                                             target, "completion_ops", path).values()),
                                  {}, path + ".completion_ops", 0),
        "core_owners": _encode(dict(sorted(_capability(
            target, "core_owners", path).items())), {}, path + ".core_owners", 0),
    }


def _encode_target(target, path: str):
    return _target_payload(target, path)


def target_bytes(target) -> bytes:
    """Canonical bytes of the ACTUAL target instance and its queried facts."""
    return _dump(_target_payload(target, "$.target"))


def source_bytes(function) -> bytes:
    """Canonical bytes of one entry function: lexical binders, full body, target."""
    return _dump(_encode_function(function, "$"))


def entry_canonical_bytes(function) -> bytes:
    return source_bytes(function)


def module_bytes(module) -> bytes:
    payload = [_encode_function(function, "$.functions[%d]" % index)
               for index, function in enumerate(getattr(module, "functions", ()) or ())]
    return _dump({"enc": IDENTITY_ENCODING_VERSION, "functions": payload})


def node_identity_payload(function) -> dict:
    """The JSON payload one function's identity is computed over."""
    return _encode_function(function, "$")


def function_identity(function) -> str:
    """A sha256 identity of one function's actual program content."""
    return "sha256:" + hashlib.sha256(source_bytes(function)).hexdigest()


def module_identity(module) -> str:
    """A sha256 identity over every actual function of a module."""
    return "sha256:" + hashlib.sha256(module_bytes(module)).hexdigest()


def source_labels(function) -> dict:
    """id(node or bound Var) -> positional label, for problem content binding.

    This is the SAME traversal order _encode_statements uses, so a label here
    names the same statement occurrence the source bytes name: problem content
    is anchored to positions in the actual TIR, never to Python object
    addresses. A caller that hands a record pointing at an object outside this
    table gets no label, and _encode then refuses that Var by path instead of
    silently inventing an identity for it.
    """
    from tilefoundry.ir.core import Var

    if type(function).__name__ != "PrimFunction":
        raise IdentityRefusal("$", "only a real PrimFunction carries TIR labels")
    labels: dict = {}
    scope = _Scope()
    for index, param in enumerate(getattr(function, "params", ()) or ()):
        labels[id(param)] = "p%d" % index
    body = getattr(function, "body", None)

    def walk(node):
        label = scope.occurrence()
        labels[id(node)] = label
        kind = type(node)
        if kind.__name__ == "Sequential":
            for child in node.body:
                walk(child)
            return
        if kind.__name__ == "LetStmt":
            labels[id(node.var)] = label
            return
        if kind.__name__ == "For":
            labels[id(node.induction_var)] = label
            walk(node.body)
            return
        if kind.__name__ == "MeshScope":
            labels[id(node.binding)] = label
            walk(node.body)
            return
        if kind.__name__ == "ExecutionScope":
            walk(node.body)
            return
        if kind.__name__ == "If":
            walk(node.then_body)
            if node.else_body is not None:
                walk(node.else_body)
            return
        for field in fields(node):
            if field.name in _DIAGNOSTIC_FIELDS:
                continue
            value = getattr(node, field.name, None)
            if isinstance(value, (list, tuple)):
                for item in value:
                    if hasattr(item, "__dataclass_fields__"):
                        walk(item)
            elif hasattr(value, "__dataclass_fields__") and kind.__name__ not in (
                    "Evaluate",):
                walk(value)
        return

    statements = body if isinstance(body, (list, tuple)) else [body]
    for statement in statements:
        walk(statement)
    return labels
