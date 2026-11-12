"""D0 pipeline declaration checks: what was declared must still be there.

These are the checkers behind the "bad control" tests. Their job is to FAIL BY
NAME when a declaration is lost or silently changed between two views of the same
program, which is the only way the corresponding tests can ever be red for the
right reason.

Expected values are built from the AUTHORED side. Deriving them from the side
being checked would let a lost declaration pass by shrinking the expectation with
it, so an empty expectation is itself refused rather than accepted vacuously.
"""

from __future__ import annotations

from tilefoundry.analysis.pipeline_reader import (
    collect_buffer_versions,
    collect_pipeline_declarations,
)
from tilefoundry.ir.hir.pipeline import PipelineError

__all__ = [
    "verify_buffer_versions_unchanged",
    "verify_pipeline_preserved",
]


def verify_pipeline_preserved(authored, lowered) -> None:
    """Every loop that declared a pipeline must still declare it in `lowered`.

    `authored` is the source of truth: the set is never narrowed to match what
    the lowered side happens to contain.
    """
    expected = collect_pipeline_declarations(authored)
    actual = collect_pipeline_declarations(lowered)
    if not expected:
        raise PipelineError(
            "PIPELINE_EXPECTATION_MISSING",
            "the authored side declares no pipeline at all, so there is nothing to "
            "preserve and this check would pass vacuously",
        )
    lost = sorted(key for key in expected if key not in actual)
    if lost:
        raise PipelineError(
            "PIPELINE_DECLARATION_LOST",
            "these loops declared a pipeline and no longer carry one after "
            "lowering/cloning: %s" % (lost,),
        )
    changed = sorted(key for key in expected if expected[key] != actual[key])
    if changed:
        raise PipelineError(
            "PIPELINE_DECLARATION_CHANGED",
            "the pipeline declaration changed: %s"
            % ([(key, expected[key], actual[key]) for key in changed],),
        )


def verify_buffer_versions_unchanged(authored, actual) -> None:
    """The stage budget is NOT a buffer version count; a change is a declared defect.

    Budget and versions are different levels: the budget is per loop, versions are
    per buffer. Folding one into the other (a "budget 2" quietly re-reading every
    buffer as 2 versions) would change what runs while the declaration still reads
    as authored.
    """
    expected = collect_buffer_versions(authored)
    seen = collect_buffer_versions(actual)
    if not expected:
        raise PipelineError(
            "PIPELINE_EXPECTATION_MISSING",
            "the authored side declares no buffer versions, so this check would "
            "pass vacuously",
        )
    differing = sorted(
        key for key in expected if key in seen and seen[key] != expected[key]
    )
    if differing:
        raise PipelineError(
            "BUFFER_VERSION_DECLARATION_CHANGED",
            "these buffers changed version count: %s"
            % ([(key, expected[key], seen[key]) for key in differing],),
        )
    missing = sorted(key for key in expected if key not in seen)
    if missing:
        raise PipelineError(
            "BUFFER_VERSION_DECLARATION_CHANGED",
            "these buffer version declarations disappeared: %s" % (missing,),
        )
