"""Single typed PipelineSpec for the native Ascend 950PR mainline.

There is one schema. max_stage bounds authored stages; max_buffer_versions bounds
auto buffer versions; they are independent. Fixed versions exist only when the
author wrote BufferSchedule(versions=("fixed", n)). There is no second field for
the stage count, no version tag branch, and no legacy stage/version mapping.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

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

MANUAL_DEFAULT = "MANUAL_DEFAULT"
AUTHORED = "AUTHORED"
BUFFER_SCHEDULE_INVALID = "BUFFER_SCHEDULE_INVALID"


class PipelineError(ValueError):
    """A pipeline declaration refused by a stable named code."""

    def __init__(self, code: str, detail: str) -> None:
        self.code = code
        self.detail = detail
        super().__init__("%s: %s" % (code, detail))


@dataclass(frozen=True)
class TaskIntent:
    """One typed whole-task intent bound to a real direct child by object identity."""

    task_id: str
    anchor: int
    stage: int | None = None
    issue_after: tuple = ()
    cost_override: int | None = None
    anchor_ref: object | None = field(default=None, compare=False, repr=False)
    source_ref: object | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.task_id) is not str or not self.task_id:
            raise PipelineError("TASK_INTENT_INVALID", "task_id must be a non-empty str")
        if type(self.anchor) is not int or isinstance(self.anchor, bool) or self.anchor < 0:
            raise PipelineError("TASK_INTENT_INVALID", "anchor must be a non-negative direct-child index")
        if self.stage is not None and (type(self.stage) is not int or isinstance(self.stage, bool) or self.stage < 0):
            raise PipelineError("TASK_INTENT_INVALID", "stage must be a non-negative int or None")
        if type(self.issue_after) is not tuple or any(type(x) is not str or not x for x in self.issue_after):
            raise PipelineError("TASK_INTENT_INVALID", "issue_after must be a tuple of task ids")
        if self.cost_override is not None and (type(self.cost_override) is not int or self.cost_override < 0):
            raise PipelineError("TASK_INTENT_INVALID", "cost_override must be a non-negative int or None")


@dataclass(frozen=True)
class BufferSchedule:
    """One published version domain for a REAL buffer root object.

    versions is ("fixed", n) or ("auto", min, max). A fixed value is authored;
    it is never derived from max_stage or from any fallback.
    """

    root: str
    versions: tuple
    owner: str = ""
    space: str = ""
    layout: str = ""
    buffer_ref: object | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.root) is not str or not self.root:
            raise PipelineError(BUFFER_SCHEDULE_INVALID, "root must be a non-empty str")
        if (type(self.versions) is not tuple or not self.versions
                or self.versions[0] not in ("fixed", "auto")):
            raise PipelineError(BUFFER_SCHEDULE_INVALID, "versions must be ('fixed', n) or ('auto', min, max)")
        if self.versions[0] == "fixed":
            if len(self.versions) != 2 or type(self.versions[1]) is not int or isinstance(self.versions[1], bool) or self.versions[1] < 1:
                raise PipelineError(BUFFER_SCHEDULE_INVALID, "fixed versions must be a positive int")
        elif (len(self.versions) != 3
              or any(type(x) is not int or isinstance(x, bool) or x < 1 for x in self.versions[1:])):
            raise PipelineError(BUFFER_SCHEDULE_INVALID, "auto versions must be (min,max) positive ints")
        elif self.versions[1] > self.versions[2]:
            raise PipelineError(BUFFER_SCHEDULE_INVALID, "auto min must not exceed max")


@dataclass(frozen=True)
class PipelineSpec:
    """The single authored pipeline declaration.

    task_intents and buffer_schedules carry object references after
    resolve_references(); before that only source locators are populated.
    """

    max_stage: int
    max_buffer_versions: int
    fixed_ii: int | None = None
    enable_offset: bool = False
    origin: Any | None = None
    task_intents: tuple = ()
    buffer_schedules: tuple = ()
    normalized_stages: tuple = field(default=(), compare=False, repr=False)

    def __post_init__(self) -> None:
        if type(self.max_stage) is not int or isinstance(self.max_stage, bool) or self.max_stage < 1:
            raise PipelineError("PIPELINE_MAX_STAGE_INVALID", "max_stage must be a positive plain int")
        if type(self.max_buffer_versions) is not int or isinstance(self.max_buffer_versions, bool) or self.max_buffer_versions < 1:
            raise PipelineError("PIPELINE_MAX_BUFFER_VERSIONS_INVALID", "max_buffer_versions must be a positive plain int")
        if self.fixed_ii is not None and (type(self.fixed_ii) is not int or isinstance(self.fixed_ii, bool) or self.fixed_ii < 1):
            raise PipelineError("PIPELINE_FIXED_II_INVALID", "fixed_ii must be a positive plain int or None")
        if type(self.enable_offset) is not bool:
            raise PipelineError("PIPELINE_ENABLE_OFFSET_INVALID", "enable_offset must be a bool")
        if not isinstance(self.task_intents, tuple) or any(not isinstance(x, TaskIntent) for x in self.task_intents):
            raise PipelineError("TASK_INTENT_INVALID", "task_intents must be TaskIntent tuple")
        if not isinstance(self.buffer_schedules, tuple) or any(not isinstance(x, BufferSchedule) for x in self.buffer_schedules):
            raise PipelineError(BUFFER_SCHEDULE_INVALID, "buffer_schedules must be BufferSchedule tuple")
        ids = [x.task_id for x in self.task_intents]
        if len(ids) != len(set(ids)):
            raise PipelineError("TASK_INTENT_INVALID", "duplicate task_id")
        for intent in self.task_intents:
            if intent.task_id in intent.issue_after:
                raise PipelineError("TASK_INTENT_INVALID", "task cannot issue_after itself")
            missing = [x for x in intent.issue_after if x not in ids]
            if missing:
                raise PipelineError("TASK_INTENT_INVALID", "issue_after references unknown task ids")
        roots = [x.root for x in self.buffer_schedules]
        if len(roots) != len(set(roots)):
            raise PipelineError(BUFFER_SCHEDULE_INVALID, "duplicate BufferSchedule root")

    @classmethod
    def from_surface(
        cls,
        *,
        max_stage: Any,
        max_buffer_versions: Any,
        fixed_ii: Any = None,
        enable_offset: Any = False,
        origin: Any | None = None,
        task_intents: Any = (),
        buffer_schedules: Any = (),
    ) -> "PipelineSpec":
        """Build the one schema from authored literals; no annotations side channel."""
        return cls(
            max_stage=max_stage,
            max_buffer_versions=max_buffer_versions,
            fixed_ii=fixed_ii,
            enable_offset=enable_offset,
            origin=origin,
            task_intents=tuple(task_intents),
            buffer_schedules=tuple(buffer_schedules),
        )

    def resolve_references(self, direct_children, buffers):
        """Bind task anchors and buffer roots to the REAL objects in this module."""
        children = tuple(direct_children)
        if len(self.task_intents) > len(children):
            raise PipelineError("TASK_INTENT_INVALID", "more task intents than direct children")
        intents = []
        for intent in self.task_intents:
            if intent.anchor >= len(children):
                raise PipelineError("TASK_INTENT_INVALID", "anchor is outside the direct-child list")
            if intent.anchor_ref is not None:
                if intent.anchor_ref is not children[intent.anchor]:
                    raise PipelineError("TASK_INTENT_REBIND", "resolved anchor_ref does not match the current child object")
                intents.append(intent)
            else:
                intents.append(replace(intent, anchor_ref=children[intent.anchor]))
        slots = {}
        for buffer in buffers or ():
            root = _buffer_name(buffer)
            slots.setdefault(root, []).append(buffer)
        schedules = []
        for schedule in self.buffer_schedules:
            matches = slots.get(schedule.root, [])
            if len(matches) != 1:
                raise PipelineError(BUFFER_SCHEDULE_INVALID, "root must name exactly one real buffer object, got %d" % len(matches))
            if schedule.buffer_ref is not None and all(schedule.buffer_ref is not item for item in matches):
                raise PipelineError("BUFFER_SCHEDULE_REBIND", "resolved buffer_ref does not match the current buffer object")
            owner, space, layout = _buffer_facts(matches[0])
            schedules.append(replace(
                schedule,
                buffer_ref=matches[0] if schedule.buffer_ref is None else schedule.buffer_ref,
                owner=schedule.owner or owner,
                space=schedule.space or space,
                layout=schedule.layout or layout,
            ))
        return replace(self, task_intents=tuple(intents), buffer_schedules=tuple(schedules))


def _buffer_name(buffer) -> str:
    return str(getattr(buffer, "root", getattr(buffer, "name", "")))


def _buffer_facts(buffer) -> tuple[str, str, str]:
    type_ = getattr(buffer, "type", None)
    owner = getattr(buffer, "owner", None) or getattr(type_, "owner", None) or ""
    space = getattr(buffer, "storage", None) or getattr(type_, "storage", None) or ""
    layout = getattr(type_, "layout", None) or getattr(buffer, "layout", None) or ""
    return str(getattr(owner, "value", owner) or ""), str(getattr(space, "value", space) or ""), str(layout)


def normalize_manual_stages(direct_children, intents, *, pipeline=None):
    """Manual stage rule: staged siblings make the list manual; unlabeled ones are stage 0."""
    children = tuple(direct_children)
    intents = tuple(intents)
    if any(intent.anchor >= len(children) for intent in intents):
        raise PipelineError("TASK_INTENT_INVALID", "anchor outside direct children")
    if len({intent.anchor for intent in intents}) != len(intents):
        raise PipelineError("TASK_INTENT_INVALID", "one child may have only one TaskIntent")
    manual = any(intent.stage is not None for intent in intents)
    if not manual:
        return ()
    by_anchor = {intent.anchor: intent for intent in intents}
    rows = []
    for index, child in enumerate(children):
        intent = by_anchor.get(index)
        if intent is not None and intent.stage is not None:
            rows.append((child, intent.stage, AUTHORED))
        else:
            rows.append((child, 0, MANUAL_DEFAULT))
    if pipeline is not None and not pipeline.enable_offset and any(stage > 0 for _child, stage, _src in rows):
        raise PipelineError("PIPELINE_OFFSET_STAGE_CONFLICT", "explicit non-zero stage requires enable_offset=True")
    return tuple(rows)


def normalize_pipeline_references(module) -> None:
    """Resolve typed references and manual stages on every reachable HIR LoopRegion."""
    from tilefoundry.ir.hir.loop_region import LoopRegion

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
    from tilefoundry.ir.hir.loop_region import LoopRegion

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
