"""Ascend document-backed values, fact evidence, and instruction capabilities."""

from __future__ import annotations

import pytest

from tests.core.target._hardware import (
    bare_projection_cases,
    check_bare_projection,
    check_document_evidence,
    check_hardware_defaults,
)
from tests.core.target._instructions import check_architecture_capabilities
from tilefoundry_ascend.ascend.target import AscendTarget
from tilefoundry_ascend.ascend.target import spec as ascend_spec


@pytest.mark.parametrize(
    ("construct", "message"),
    bare_projection_cases(
        lambda architecture: AscendTarget(architecture=architecture),
        AscendTarget,
        "ascend",
        (
            "AscendTarget.architecture must be an installed ID string, a document path, or "
            "a DaVinci950; got _BareArchitecture",
            "AscendTarget.device must be an installed ID string, a document path, or "
            "an Ascend950PR; got _BareDevice",
        ),
    ),
)
def test_document_backends_reject_bare_projection_value_markers(construct, message: str) -> None:
    check_bare_projection(construct, message)


def test_no_installed_number_is_repeated_as_a_python_default() -> None:
    check_hardware_defaults(
        AscendTarget.hardware, (ascend_spec.DAVINCI_950_ID, ascend_spec.ASCEND_950PR_ID)
    )


def test_architecture_and_device_evidence_preserve_ownership() -> None:
    owners = {
        "ascend.davinci_950": {
            "memory.tmem.owner": "aic",
            "memory.smem.owner": "aiv",
            "memory.rmem.owner": "lane",
        },
        "ascend.950pr": {"memory.hbm.owner": "target"},
    }
    check_document_evidence(AscendTarget.hardware, owners)


def test_instruction_capabilities_and_execution_resources_reconcile() -> None:
    check_architecture_capabilities(AscendTarget, "tilefoundry_ascend.")
