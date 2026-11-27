"""Shared JSON record serialization mixin and atomic text write helper."""

from __future__ import annotations

import functools
import json
import os
import tempfile
import types
from collections.abc import Mapping, Sequence
from dataclasses import MISSING, Field, fields, is_dataclass
from pathlib import Path
from typing import Any, ClassVar, Literal, Self, Union, get_args, get_origin, get_type_hints


def atomic_write_text(path: Path | str, content: str) -> None:
    """Atomically replace *path* with UTF-8 *content* via a same-directory temp file."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=str(target.parent),
            prefix=f".{target.name}.",
            suffix=".tmp",
            delete=False,
        ) as fh:
            tmp_path = Path(fh.name)
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, target)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def json_clone(value: Any) -> Any:
    """Recursively convert records, mappings, and sequences into JSON-compatible values."""
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {str(k): json_clone(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_clone(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


@functools.lru_cache(maxsize=None)
def _dataclass_field_hints(cls: type[Any]) -> tuple[tuple[Field[Any], Any], ...]:
    hints = get_type_hints(cls)
    return tuple((f, hints.get(f.name, Any)) for f in fields(cls))


def _default_for_field(f: Field[Any]) -> tuple[bool, Any]:
    if f.default is not MISSING:
        return True, f.default
    if f.default_factory is not MISSING:
        return True, f.default_factory()
    return False, None


def _unwrap_optional(hint: Any) -> tuple[bool, Any]:
    origin = get_origin(hint)
    if origin in (Union, types.UnionType):
        args = get_args(hint)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) < len(args):
            inner = (
                non_none[0]
                if len(non_none) == 1
                else Union.__getitem__(tuple(non_none))
            )
            return True, inner
    return False, hint


def _coerce_value(hint: Any, raw: Any, *, has_default: bool = False, default_val: Any = None) -> Any:
    is_opt, inner = _unwrap_optional(hint)
    if raw is None:
        if is_opt:
            return None
        if has_default:
            return default_val
        origin = get_origin(inner)
        if origin is tuple:
            return ()
        if origin is dict:
            return {}
        return None

    if isinstance(inner, type) and hasattr(inner, "from_dict"):
        if isinstance(raw, inner):
            return raw
        if isinstance(raw, Mapping):
            return inner.from_dict(raw)
        if has_default:
            return default_val
        raise TypeError(f"Expected {inner.__name__} or Mapping, got {type(raw).__name__}")

    origin = get_origin(inner)
    args = get_args(inner)

    if origin is tuple:
        if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence):
            raise TypeError(f"Expected sequence for tuple field, got {type(raw).__name__}")
        if len(args) == 2 and args[1] is Ellipsis:
            elem_hint = args[0]
            return tuple(_coerce_value(elem_hint, item) for item in raw)
        if args:
            return tuple(
                _coerce_value(args[min(i, len(args) - 1)], item) for i, item in enumerate(raw)
            )
        return tuple(raw)

    if origin is dict:
        if not isinstance(raw, Mapping):
            if is_opt:
                return None
            raise TypeError(f"Expected Mapping for dict field, got {type(raw).__name__}")
        val_hint = args[1] if len(args) == 2 else Any
        return {str(k): _coerce_value(val_hint, v) for k, v in raw.items()}

    if origin is Literal:
        return str(raw) if all(isinstance(a, str) for a in args) else raw

    if inner is bool:
        return bool(raw)
    if inner is int:
        return int(raw)
    if inner is float:
        return float(raw)
    if inner is str:
        return str(raw)
    return raw


class JsonRecordMixin:
    """Reusable JSON/dict serialization mixin for dataclasses."""

    _json_sort_keys: ClassVar[bool] = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize dataclass fields into a JSON-compatible dictionary."""
        if not is_dataclass(self):
            raise TypeError(f"{type(self).__name__} must be a dataclass to use JsonRecordMixin")
        return {f.name: json_clone(getattr(self, f.name)) for f in fields(self)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> Self:
        """Reconstruct a dataclass instance from a serialized mapping."""
        if not is_dataclass(cls):
            raise TypeError(f"{cls.__name__} must be a dataclass to use JsonRecordMixin")
        kwargs: dict[str, Any] = {}
        for f, hint in _dataclass_field_hints(cls):
            has_default, default_val = _default_for_field(f)
            is_opt, inner = _unwrap_optional(hint)
            origin = get_origin(inner)
            if f.name in payload:
                raw = payload[f.name]
                if raw is None and is_opt:
                    kwargs[f.name] = None
                elif raw is None and has_default:
                    kwargs[f.name] = default_val
                else:
                    kwargs[f.name] = _coerce_value(
                        hint, raw, has_default=has_default, default_val=default_val
                    )
            elif has_default:
                kwargs[f.name] = default_val
            elif not is_opt and origin is tuple:
                kwargs[f.name] = ()
            elif not is_opt and origin is dict:
                kwargs[f.name] = {}
            else:
                raise KeyError(f.name)
        return cls(**kwargs)

    def to_json(self, *, indent: int | None = 2) -> str:
        """Serialize record into formatted JSON text."""
        return json.dumps(self.to_dict(), indent=indent, sort_keys=self._json_sort_keys)

    @classmethod
    def from_json(cls, raw: str) -> Self:
        """Deserialize record from a JSON string."""
        return cls.from_dict(json.loads(raw))


__all__ = [
    "JsonRecordMixin",
    "atomic_write_text",
    "json_clone",
]
