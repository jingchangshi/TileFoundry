"""Native attention_residual package.

Only the native builder is exported. Archived twin/HIR/kernel modules are not
restored and this package never imports them.
"""
from .build import AttnResBuildError, AttnResSpec, build_attnres, build_attnres_module

__all__ = [
    "AttnResBuildError",
    "AttnResSpec",
    "build_attnres",
    "build_attnres_module",
]
