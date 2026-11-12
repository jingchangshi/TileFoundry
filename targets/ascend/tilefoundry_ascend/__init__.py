"""TileFoundry Ascend 950PR native mainline package.

Only the single native compiler path is registered. Historical material lives
under targets/ascend/_archive_historical and is never imported by this package.
"""
from tilefoundry_ascend import ir as _ir  # noqa: F401  (registers the surface)
from tilefoundry_ascend.tf_target import Ascend950PRTarget  # noqa: F401
from tilefoundry_ascend.passes import pipeline as _pipeline  # noqa: F401
from tilefoundry_ascend.passes import p3_chain as _p3_chain  # noqa: F401

__all__ = ["Ascend950PRTarget"]
