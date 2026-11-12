"""Capture authored native pipeline declarations before HIR dead-code elimination.

Only the AST locates a declaration. Presence is checked against the typed
LoopRegion.pipeline that actually survives in the checked HIR. If a declaration
disappears, compilation is refused rather than shrinking its own expectation.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass

from tilefoundry.ir.hir.pipeline import PipelineError

__all__ = [
    "PipelineIntent",
    "PipelineIntentSet",
    "register_pipeline_intents",
    "verify_pipeline_intents",
]


@dataclass(frozen=True)
class PipelineIntent:
    """One authored tf.pipelined declaration, located in source."""

    function_name: str
    source_key: tuple[int, int]
    max_stage: int | None
    max_buffer_versions: int | None
    fixed_ii: int | None
    enable_offset: bool | None

    @property
    def origin(self) -> tuple[int, int]:
        return self.source_key


PipelineIntentSet = tuple[PipelineIntent, ...]


def _is_tf_pipelined_callee(call: ast.AST) -> bool:
    return (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "tf"
        and call.func.attr == "pipelined"
    )


def _literal_or_none(node: ast.AST) -> object:
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return None


def register_pipeline_intents(
    node: ast.FunctionDef, *, function_name: str, dialect: str
) -> PipelineIntentSet:
    """Capture every authored tf.pipelined declaration from the AST."""
    if dialect != "hir":
        return ()
    intents: list[PipelineIntent] = []
    seen: set[tuple[int, int]] = set()
    for candidate in ast.walk(node):
        if not isinstance(candidate, ast.For):
            continue
        iterator = candidate.iter
        if not _is_tf_pipelined_callee(iterator):
            continue
        key = (iterator.lineno, iterator.col_offset)
        if key in seen:
            continue
        seen.add(key)
        values: dict[str, object] = {}
        for keyword in getattr(iterator, "keywords", ()):
            if keyword.arg is None:
                continue
            values[keyword.arg] = _literal_or_none(keyword.value)
        enable_offset = values.get("enable_offset")
        enable_offset = enable_offset if isinstance(enable_offset, bool) else None
        intents.append(PipelineIntent(
            function_name=function_name,
            source_key=key,
            max_stage=values.get("max_stage") if isinstance(values.get("max_stage"), int) else None,
            max_buffer_versions=values.get("max_buffer_versions") if isinstance(values.get("max_buffer_versions"), int) else None,
            fixed_ii=values.get("fixed_ii") if isinstance(values.get("fixed_ii"), int) else None,
            enable_offset=enable_offset,
        ))
    return tuple(intents)


def _reachable_pipeline_loops(function) -> dict[tuple[int, int], object]:
    from tilefoundry.analysis.effects import reachable_loops  # noqa: PLC0415

    found: dict[tuple[int, int], object] = {}
    for expr in reachable_loops(function):
        spec = getattr(expr, "pipeline", None)
        if spec is None:
            continue
        origin = getattr(spec, "origin", None)
        key = tuple(origin) if isinstance(origin, tuple) else None
        if key is not None and key in found:
            raise PipelineError("PIPELINE_SOURCE_DUPLICATE", "two surviving loops claim authored source %r" % (key,))
        if key is not None:
            found[key] = spec
    return found


def verify_pipeline_intents(intents: PipelineIntentSet, function) -> None:
    """Every captured declaration must still be a typed pipeline on the surviving HIR."""
    if not intents:
        return
    keys = [intent.source_key for intent in intents]
    if len(set(keys)) != len(keys):
        raise PipelineError("PIPELINE_SOURCE_DUPLICATE", "the authored source declares one pipeline position twice: %r" % (keys,))
    actual = _reachable_pipeline_loops(function)
    missing = [intent for intent in intents if intent.source_key not in actual]
    if missing:
        raise PipelineError(
            "PIPELINE_DECLARATION_DROPPED_BY_DCE",
            "the authored source declares tf.pipelined at %s, but no loop carrying that "
            "declaration survived parsing/dead-code elimination in %r; a pipeline "
            "declaration may not be dropped silently"
            % ([(intent.source_key, intent.function_name) for intent in missing],
               getattr(function, "name", "<anonymous>")),
        )
    declared = {intent.source_key for intent in intents}
    unbound = sorted(key for key in actual if key not in declared)
    if unbound:
        raise PipelineError(
            "PIPELINE_SOURCE_UNBOUND",
            "surviving pipelined loop(s) at %s have no authored declaration in this function" % (unbound,),
        )
