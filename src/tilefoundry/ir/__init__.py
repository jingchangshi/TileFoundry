"""tilefoundry.ir — IR tree root.

Subpackages:
- `ir.core`   — spec 001 (Expr/Op/Call/...)
- `ir.types`  — spec 002 (TensorType/DType/...)
- `ir.types.shard` — spec 003 (Mesh/ShardLayout/...)
- `ir.hir`    — spec 004
- `ir.tir`    — spec 005

Side-effect-free; each subpackage declares its own exports.
"""

from __future__ import annotations
