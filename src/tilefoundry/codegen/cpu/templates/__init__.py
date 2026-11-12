"""Jinja2 templates for CPU (host) codegen boilerplate.

Only the host-module wrapper is rendered here; stmt/op emission stays in the
Python walker.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

_TEMPLATE_DIR = Path(__file__).parent


@lru_cache(maxsize=1)
def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(_TEMPLATE_DIR)),
        undefined=StrictUndefined,
        trim_blocks=False,
        lstrip_blocks=False,
        keep_trailing_newline=True,
    )


def render(name: str, **vars) -> str:
    return _env().get_template(name).render(**vars)


__all__ = ["render"]
