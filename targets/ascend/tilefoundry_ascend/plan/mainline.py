"""Frozen native mainline transaction surface.

This module is the one place where the compiler turns a checked TIR function
into a rewritten symbolic entry. It has no compatibility schema: the transaction
is fixed by INTERFACE_FREEZE_NATIVE_MAINLINE.md. Until MAIN-A supplies the real
analyzer/solver/resolver, the named refusal below is intentional and never an
APPLIED metadata record.
"""
from __future__ import annotations

from dataclasses import replace

__all__ = [
    "MainlineError",
    "MainlineNotImplemented",
    "PlanError",
    "SINGLE_TRANSACTION",
    "apply_symbolic_pipeline",
    "compile_mainline",
    "install_entry",
    "verify_actual_program",
]

SINGLE_TRANSACTION = (
    "freeze", "analyze", "solve", "resolve", "completion",
    "storage", "apply", "verify", "install", "export",
)


class PlanError(ValueError):
    """A malformed plan or failed installation, with a stable named code."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__("%s: %s" % (code, detail))


class MainlineError(RuntimeError):
    """A named refusal from the single native transaction."""


class MainlineNotImplemented(MainlineError):
    """The real transaction stage is not implemented yet; never report APPLIED."""


def install_entry(module, rewritten_entry, *, conformance=None):
    """Install a rewritten REAL function only after actual conformance is verified.

    The module is immutable and is never mutated on refusal. A same-name fresh
    PrimFunction is not proof: the caller's metadata/status is not accepted, and
    the real verify_actual_program path is not implemented in this candidate.
    """
    entry_name = getattr(module, "entry", None)
    if not entry_name:
        raise PlanError("PLAN_ENTRY_UNKNOWN", "module has no entry name")
    functions = tuple(getattr(module, "functions", ()) or ())
    if not any(getattr(function, "name", None) == entry_name for function in functions):
        raise PlanError("PLAN_ENTRY_UNKNOWN", "module entry %r is not a function" % (entry_name,))
    if getattr(rewritten_entry, "name", None) != entry_name:
        raise PlanError("PLAN_ENTRY_MISMATCH", "rewritten entry name does not match module.entry")
    if conformance is None:
        raise PlanError(
            "INSTALL_CANDIDATE_UNVERIFIED",
            "install_entry requires the actual verify_actual_program conformance result; "
            "a same-name candidate and caller metadata are not verification",
        )
    if isinstance(conformance, dict):
        raise PlanError(
            "INSTALL_CANDIDATE_UNVERIFIED",
            "install_entry refuses a self-reported conformance dict; actual verification is required",
        )
    # The actual verifier and TIR/TL conformance path are not implemented yet.
    # Fail closed rather than installing an unverified candidate.
    raise MainlineNotImplemented("INSTALL_VERIFIER_NOT_IMPLEMENTED")


def apply_symbolic_pipeline(entry, plan, completion, storage):
    raise MainlineNotImplemented("MAINLINE_APPLY_NOT_IMPLEMENTED")


def verify_actual_program(original, rewritten, plan, completion, storage):
    raise MainlineNotImplemented("MAINLINE_VERIFY_NOT_IMPLEMENTED")


def compile_mainline(module, authored=None):
    raise MainlineNotImplemented("MAINLINE_CHAIN_NOT_IMPLEMENTED")
