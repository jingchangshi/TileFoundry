"""Resolve typed PipelineSpec references on HIR loop regions.

The schema lives in ``tilefoundry.ir.hir.pipeline_spec``. This module walks HIR
values, binds each authored reference to the real child or buffer object, and
rebinds references after specialization clones a module.
"""

from __future__ import annotations

from dataclasses import replace

from tilefoundry.ir.hir.loop_region import LoopRegion
from tilefoundry.ir.hir.pipeline_spec import (
    AUTHORED,
    BUFFER_SCHEDULE_INVALID,
    MANUAL_DEFAULT,
    BufferSchedule,
    PipelineError,
    PipelineSpec,
    TaskIntent,
    _buffer_name,
    normalize_manual_stages,
)

__all__ = [
    "AUTHORED",
    "BUFFER_SCHEDULE_INVALID",
    "BufferSchedule",
    "MANUAL_DEFAULT",
    "PipelineError",
    "PipelineSpec",
    "TaskIntent",
    "normalize_manual_stages",
    "normalize_pipeline_references",
    "rebind_cloned_pipeline_references",
]


def normalize_pipeline_references(module) -> None:
    """Resolve typed references and manual stages on every reachable HIR LoopRegion."""
    def walk(node):
        if isinstance(node, LoopRegion) and node.pipeline is not None:
            pipeline = node.pipeline
            if pipeline.task_intents or pipeline.buffer_schedules:
                body_args = tuple(getattr(node.body, "args", ()) or ())
                target_name = type(getattr(node.body, "target", None)).__name__.lower()
                children = body_args if target_name in ("group", "sequence") and body_args else (node.body,)
                buffers = []
                seen_ids = set()

                def add_buffer(item, *, skip_existing_name=False):
                    if item is None or id(item) in seen_ids:
                        return
                    name = _buffer_name(item)
                    if skip_existing_name and name in existing_names:
                        return
                    seen_ids.add(id(item))
                    buffers.append(item)

                seen_param_names = set()
                for item in tuple(getattr(node, "params", ())) + tuple(getattr(node, "args", ())):
                    name = _buffer_name(item)
                    if name in seen_param_names:
                        continue
                    seen_param_names.add(name)
                    add_buffer(item)
                existing_names = {_buffer_name(item) for item in buffers}

                def walk_allocations(statement):
                    fields = getattr(statement, "__dataclass_fields__", None)
                    if not fields:
                        return
                    if type(statement).__name__ == "LetStmt":
                        value = getattr(statement, "value", None)
                        if type(getattr(value, "target", None)).__name__ in ("AllocTensor", "TensorView", "PtrOf"):
                            add_buffer(getattr(statement, "var", None), skip_existing_name=True)
                    for field_name in fields:
                        if field_name in ("type", "metadata", "origin"):
                            continue
                        child = getattr(statement, field_name, None)
                        if isinstance(child, tuple):
                            for item in child:
                                walk_allocations(item)
                        elif hasattr(child, "__dataclass_fields__"):
                            walk_allocations(child)

                walk_allocations(node.body)
                resolved = pipeline.resolve_references(children, tuple(buffers))
                rows = normalize_manual_stages(children, resolved.task_intents, pipeline=resolved)
                object.__setattr__(node, "pipeline", replace(resolved, normalized_stages=rows))
        for field_name in getattr(node, "__dataclass_fields__", ()) or ():
            if field_name in ("type", "metadata", "origin"):
                continue
            value = getattr(node, field_name, None)
            if isinstance(value, tuple):
                for item in value:
                    walk(item)
            elif hasattr(value, "__dataclass_fields__"):
                walk(value)

    for function in getattr(module, "functions", ()) or ():
        walk(function)


def rebind_cloned_pipeline_references(module) -> None:
    """Clear authored object refs on a cloned module and resolve them locally."""
    def walk(node):
        if isinstance(node, LoopRegion) and node.pipeline is not None:
            pipeline = node.pipeline
            if pipeline.task_intents or pipeline.buffer_schedules:
                object.__setattr__(node, "pipeline", replace(
                    pipeline,
                    task_intents=tuple(replace(intent, anchor_ref=None) for intent in pipeline.task_intents),
                    buffer_schedules=tuple(replace(schedule, buffer_ref=None) for schedule in pipeline.buffer_schedules),
                ))
        for field_name in getattr(node, "__dataclass_fields__", ()) or ():
            if field_name in ("type", "metadata", "origin"):
                continue
            value = getattr(node, field_name, None)
            if isinstance(value, tuple):
                for item in value:
                    walk(item)
            elif hasattr(value, "__dataclass_fields__"):
                walk(value)

    for function in getattr(module, "functions", ()) or ():
        walk(function)
    normalize_pipeline_references(module)
