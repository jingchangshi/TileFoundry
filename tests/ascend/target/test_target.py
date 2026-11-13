"""Idempotent Ascend target registration check."""

from __future__ import annotations

import tilefoundry_ascend  # noqa: F401
from tilefoundry.target import registered_targets


def test_registration_is_idempotent():
    assert {"ascend"} <= registered_targets().keys()
