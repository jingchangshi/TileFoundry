"""Pipeline declaration surface: the typed "this loop is pipelined" intent.

D0 carries DECLARATION only. A PipelineSpec says a loop was authored with a
stage budget; it is not a schedule, it does not imply a solver ran, and nothing
here generates offset plans. Consumers that cannot act on the intent must refuse
by name rather than silently downgrade it.

Single source of truth: stop is NOT copied in here. The authoritative loop bounds
are LoopRegion.start/extent/step (extent is the exclusive stop). The canonical
descriptor exports them alongside the spec, so a clone or a substitute cannot
leave two copies of one expression out of step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = [
    "PIPELINE_SCHEMA_VERSION",
    "PipelineError",
    "PipelineSpec",
]


#: Schema tag recorded in every spec, so a later revision is distinguishable
#: rather than silently reinterpreted under the old rules.
PIPELINE_SCHEMA_VERSION = "pipeline/1"

#: Annotation keys the surface accepts. Anything else is refused by name: an
#: annotation that is quietly ignored is a value the author believes in and the
#: compiler does not have.
SUPPORTED_ANNOTATIONS = frozenset({"enable_offset"})


class PipelineError(ValueError):
    """A pipeline declaration the surface refuses, with a stable named code.

    Carried as an attribute rather than only in the message so a test can assert
    WHICH rule refused, and so a caller never has to match on prose.
    """

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__("%s: %s" % (code, detail))


@dataclass(frozen=True)
class PipelineSpec:
    """Immutable authored intent that a loop is pipelined.

    Fields are deliberately minimal. num_stages is a loop-level BUDGET, not a
    per-buffer version count; buffer versions live on the buffer declarations
    (Empty.buffers) and are never folded in here.
    """

    num_stages: int
    enable_offset: bool = False
    origin: Any | None = None
    schema_version: str = PIPELINE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        # bool is an int subclass in Python, and a stage budget of True/False is
        # meaningless. Reject it explicitly rather than letting it through as 1/0.
        if type(self.num_stages) is not int:
            raise PipelineError(
                "PIPELINE_NUM_STAGES_INVALID",
                "num_stages must be a plain int, got %s (%r)"
                % (type(self.num_stages).__name__, self.num_stages),
            )
        if self.num_stages < 1:
            raise PipelineError(
                "PIPELINE_NUM_STAGES_INVALID",
                "num_stages must be >= 1, got %r" % (self.num_stages,),
            )
        if type(self.enable_offset) is not bool:
            raise PipelineError(
                "PIPELINE_ENABLE_OFFSET_INVALID",
                "enable_offset must be a bool, got %s (%r)"
                % (type(self.enable_offset).__name__, self.enable_offset),
            )
        if self.schema_version != PIPELINE_SCHEMA_VERSION:
            raise PipelineError(
                "PIPELINE_SCHEMA_UNSUPPORTED",
                "unknown pipeline schema %r" % (self.schema_version,),
            )

    @classmethod
    def from_surface(
        cls,
        num_stages: Any,
        annotations: Any = None,
        origin: Any | None = None,
    ) -> "PipelineSpec":
        """Build from the authored surface, refusing anything D0 does not support.

        D0 accepts a literal stage budget and a small closed annotation set. A
        symbolic or dynamic expression is refused here rather than deferred: the
        surface has no way to represent "resolve this later" without pretending a
        solver already ran.
        """
        if annotations is None:
            annotations = {}
        if not isinstance(annotations, dict):
            raise PipelineError(
                "PIPELINE_ANNOTATION_UNSUPPORTED",
                "annotations must be a dict, got %s" % (type(annotations).__name__,),
            )
        unknown = tuple(sorted(k for k in annotations if k not in SUPPORTED_ANNOTATIONS))
        if unknown:
            raise PipelineError(
                "PIPELINE_ANNOTATION_UNSUPPORTED",
                "unsupported annotation key(s) %s; D0 accepts only %s"
                % (unknown, tuple(sorted(SUPPORTED_ANNOTATIONS))),
            )
        enable_offset = annotations.get("enable_offset", False)
        return cls(num_stages=num_stages, enable_offset=enable_offset, origin=origin)
