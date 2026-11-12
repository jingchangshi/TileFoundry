"""Authored pipeline intent: register before parse/DCE, verify after.

The parser emits an HIR body by reachability: a statement whose value nothing
consumes does not reach the returned expression, so the loop node it built is
dropped. That is correct for private dead values, but a `tf.pipelined` loop is a
DECLARATION, and a declaration that silently disappears while the program still
finalizes is a vacuous success.

So the intent is captured from the authored AST BEFORE any body is built and
before dead-code elimination can remove a loop, then checked against the typed
HIR that actually survived. The expected set is never derived from the surviving
HIR: an empty expectation taken from the checked side would let a lost
declaration shrink its own expectation and always pass.

Only the AST is used to LOCATE a declaration. What counts as "still present" is
a typed `LoopRegion.pipeline` (a `PipelineSpec`) reachable from the parsed
function body -- never a string match, never a reconstructed fake loop.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass

from tilefoundry.ir.hir.pipeline import PipelineError

__all__ = [
    "PIPELINE_INTENT_SCHEMA",
    "PipelineIntent",
    "PipelineIntentSet",
    "register_pipeline_intents",
    "verify_pipeline_intents",
]

#: Schema tag for the captured intent record, so a later revision is
#: distinguishable rather than silently reinterpreted under the old rules.
PIPELINE_INTENT_SCHEMA = "pipeline-intent/1"


@dataclass(frozen=True)
class PipelineIntent:
    """One authored `tf.pipelined` declaration, located in the source.

    `source_key` is the authored position of the `tf.pipelined(...)` call. It is
    the same position the parser records on `PipelineSpec.origin`, so a surviving
    loop can be matched back to the declaration it came from within one parse.
    It is a source locator, not a semantic hash: a reprint that moves a line does
    not change what was declared.
    """

    schema: str
    function_name: str
    source_key: tuple[int, int]
    num_stages: int | None
    enable_offset: bool | None

    @property
    def origin(self) -> tuple[int, int]:
        return self.source_key


PipelineIntentSet = tuple[PipelineIntent, ...]


def _is_tf_pipelined_callee(call: ast.AST) -> bool:
    """Whether *call* is exactly `tf.pipelined(...)`, the only accepted spelling."""
    return (
        isinstance(call, ast.Call)
        and isinstance(call.func, ast.Attribute)
        and isinstance(call.func.value, ast.Name)
        and call.func.value.id == "tf"
        and call.func.attr == "pipelined"
    )


def _literal_or_none(node: ast.AST) -> object:
    """The authored literal value, or None when it is not a literal.

    A non-literal is not refused here: the parser already refuses a dynamic
    `num_stages`/`annotations` by name. This record is only rich enough to make
    a lost declaration legible in the refusal message.
    """
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        return None


def register_pipeline_intents(
    node: ast.FunctionDef, *, function_name: str, dialect: str
) -> PipelineIntentSet:
    """Capture every authored `tf.pipelined` loop intent from the AST.

    Called before the body is walked, so a declaration is recorded even when the
    loop it belongs to is later eliminated as dead. Only HIR carries this
    declaration today; TIR has no body-level DCE, so the check is HIR-only.
    """
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
        if key in seen:  # a defensive dedupe: one authored call is one intent
            continue
        seen.add(key)
        num_stages = None
        enable_offset = None
        for keyword in getattr(iterator, "keywords", ()):  # ast.Call
            if keyword.arg == "num_stages":
                value = _literal_or_none(keyword.value)
                num_stages = value if isinstance(value, int) and not isinstance(value, bool) else None
            elif keyword.arg == "annotations":
                value = _literal_or_none(keyword.value)
                if isinstance(value, dict):
                    flag = value.get("enable_offset", False)
                    enable_offset = flag if isinstance(flag, bool) else None
        intents.append(
            PipelineIntent(
                schema=PIPELINE_INTENT_SCHEMA,
                function_name=function_name,
                source_key=key,
                num_stages=num_stages,
                enable_offset=enable_offset,
            )
        )
    return tuple(intents)


def _reachable_pipeline_loops(function) -> dict[tuple[int, int], object]:
    """{origin: PipelineSpec} for loops that actually survived into *function*.

    Only the typed HIR reachable from the returned body counts. A `LoopRegion`
    that was constructed and then dropped is NOT here, which is exactly what
    makes a lost declaration visible.
    """
    from tilefoundry.analysis.effects import reachable_loops  # noqa: PLC0415

    # "Reachable from the returned value" is the effect-aware DCE root set: the same
    # predicate analysis.effects uses to decide retention. Sharing it keeps the two
    # answers (which loops survived, which loops are roots) from drifting apart.
    found: dict[tuple[int, int], object] = {}
    for expr in reachable_loops(function):
        spec = getattr(expr, "pipeline", None)
        if spec is None:
            continue
        origin = getattr(spec, "origin", None)
        key = tuple(origin) if isinstance(origin, tuple) else None
        if key is not None and key in found:
            raise PipelineError(
                "PIPELINE_SOURCE_DUPLICATE",
                "two surviving loops both claim authored source %r" % (key,),
            )
        if key is not None:
            found[key] = spec
    return found


def verify_pipeline_intents(intents: PipelineIntentSet, function) -> None:
    """Every captured intent must still be a typed pipeline on the surviving HIR.

    Raises `PIPELINE_DECLARATION_DROPPED_BY_DCE` when a declared loop is gone,
    `PIPELINE_SOURCE_DUPLICATE` for two intents at one source, and
    `PIPELINE_SOURCE_UNBOUND` for a surviving pipeline loop that no intent
    accounts for. An empty intent set is legal (an ordinary program declares no
    pipeline); it is never treated as a satisfied expectation.
    """
    if not intents:
        return
    keys = [intent.source_key for intent in intents]
    if len(set(keys)) != len(keys):
        raise PipelineError(
            "PIPELINE_SOURCE_DUPLICATE",
            "the authored source declares one pipeline position twice: %r" % (keys,),
        )
    actual = _reachable_pipeline_loops(function)
    missing = [intent for intent in intents if intent.source_key not in actual]
    if missing:
        raise PipelineError(
            "PIPELINE_DECLARATION_DROPPED_BY_DCE",
            "the authored source declares tf.pipelined at %s, but no loop carrying "
            "that declaration survived parsing/dead-code elimination in %r; a "
            "pipeline declaration may not be dropped silently"
            % (
                [
                    {
                        "source": intent.source_key,
                        "function": intent.function_name,
                        "num_stages": intent.num_stages,
                        "enable_offset": intent.enable_offset,
                    }
                    for intent in missing
                ],
                getattr(function, "name", "<anonymous>"),
            ),
        )
    declared = {intent.source_key for intent in intents}
    unbound = sorted(key for key in actual if key not in declared)
    if unbound:
        raise PipelineError(
            "PIPELINE_SOURCE_UNBOUND",
            "surviving pipelined loop(s) at %s have no authored declaration in this "
            "function" % (unbound,),
        )
