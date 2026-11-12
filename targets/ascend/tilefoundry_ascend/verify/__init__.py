"""CPU verification entry points for real lowered native TIR."""

from .executor import ExecutionRefusal, run_entry, supported_instructions

__all__ = ["ExecutionRefusal", "run_entry", "supported_instructions"]
