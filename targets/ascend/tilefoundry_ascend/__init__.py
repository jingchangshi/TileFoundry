"""Ascend targets for TileFoundry."""

import os

if not os.environ.get("ASCEND_TOOLKIT_HOME") and not os.environ.get("ASCEND_HOME_PATH"):
    os.environ.setdefault("TORCH_DEVICE_BACKEND_AUTOLOAD", "0")

from tilefoundry_ascend.ascend import (  # noqa: E402
    PURE_SIMDVF_UBUF_FLAGS,
    Ascend950PR,
    AscendTarget,
    DaVinci950,
    NodeResourceMask,
)

__all__ = [
    "PURE_SIMDVF_UBUF_FLAGS",
    "Ascend950PR",
    "AscendTarget",
    "DaVinci950",
    "NodeResourceMask",
]
