"""Ascend compiler passes."""

from tilefoundry_ascend.ascend.passes.pipeline import (
    ResolveAscendPipelines,
    ValidateAscendPipelines,
)

__all__ = ["ResolveAscendPipelines", "ValidateAscendPipelines"]
