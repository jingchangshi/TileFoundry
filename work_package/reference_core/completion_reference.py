"""Logical ring-slot protocol checker, independent of estimated cycle latency.

An event says completion actually occurred in the modeled execution. This
does not prove that a hardware Set/Wait implements it: namespace allocation,
memory visibility, conditional control flow and device execution need TF's
target-aware checks. The model deliberately knows every required consumer.
"""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Buffer:
    name: str
    versions: int
    consumers: tuple[str, ...]


@dataclass(frozen=True)
class Event:
    kind: str  # write_begin | ready | read_begin | read_done | free
    buffer: str
    slot: int
    generation: int
    consumer: str = ""


@dataclass
class Slot:
    generation: int | None = None
    retired_generation: int = -1
    ready: bool = False
    active: set[str] = field(default_factory=set)
    done: set[str] = field(default_factory=set)


class RingProtocol:
    def __init__(self, buffers):
        buffers = tuple(buffers)
        self.buffers = {b.name: b for b in buffers}
        if len(self.buffers) != len(buffers):
            raise ValueError("duplicate buffer declarations")
        for b in buffers:
            if type(b.versions) is not int or b.versions <= 0 or len(set(b.consumers)) != len(b.consumers):
                raise ValueError("invalid buffer versions/consumer list")
        self.slots = {(b.name, s): Slot() for b in buffers for s in range(b.versions)}

    def accept(self, event: Event):
        if event.buffer not in self.buffers:
            raise ValueError("unknown buffer")
        b = self.buffers[event.buffer]
        if type(event.generation) is not int or type(event.slot) is not int or event.generation < 0 or event.slot != event.generation % b.versions:
            raise ValueError("slot/generation mismatch")
        slot = self.slots[(b.name, event.slot)]
        if event.kind == "write_begin":
            if slot.generation is not None:
                raise ValueError(f"overwrite before FREE: {b.name}[{event.slot}] generation {slot.generation}")
            if event.generation <= slot.retired_generation:
                raise ValueError("stale/replayed write generation")
            slot.generation, slot.ready = event.generation, False
            return
        if slot.generation != event.generation:
            raise ValueError("event refers to wrong or nonexistent generation")
        if event.kind == "ready":
            if slot.ready:
                raise ValueError("duplicate READY")
            slot.ready = True
        elif event.kind == "read_begin":
            if not slot.ready or event.consumer not in b.consumers or event.consumer in slot.active | slot.done:
                raise ValueError("read without READY or invalid/duplicate consumer")
            slot.active.add(event.consumer)
        elif event.kind == "read_done":
            if event.consumer not in slot.active:
                raise ValueError("read completion without matching read")
            slot.active.remove(event.consumer)
            slot.done.add(event.consumer)
        elif event.kind == "free":
            if not slot.ready or slot.active or slot.done != set(b.consumers):
                raise ValueError("FREE before every last consumer completed")
            slot.retired_generation = slot.generation
            slot.generation, slot.ready = None, False
            slot.done.clear()
        else:
            raise ValueError(f"unknown event kind: {event.kind}")

    def finish(self):
        occupied = [key for key, slot in self.slots.items() if slot.generation is not None]
        if occupied:
            raise ValueError(f"region exit with live asynchronous storage: {occupied}")
