"""Authored SimdVF execution-region probe (M3).

This is the smallest authored program that enters a declared execution mode
through the public surface. It exists so the region can be replayed by anyone
with the interpreter of record, rather than living in a scratch file:

    docker exec -u s00653124 tf-ascend950pr bash -c '\
      export PYTHONPATH=<repo>/src:<repo>/targets/ascend; \
      cd <repo> && $HOME/.conda/envs/tilefoundry-npu/bin/python3.12 \
      -m tilefoundry.cli analyze examples/rmsnorm_950pr/vf_probe.py:VfInsideMesh'

``/opt/tf-venv`` is retired: a transcript from it is provisional and must not be
quoted as evidence. Nothing here runs on the host.

Reading the two entries:

``VfInsideMesh`` stages an input through UB and enters ``T.SimdVF`` under the
AIV mesh, which is the placement the mode declares. It analyzes today, and it is
*refused* at ``schedule finalize``: the pointwise add inside the region is a
legacy operation that declares no execution placement, and a mode admits only
operations written for it. That refusal is the intended current answer, not a
shape to work around -- the positive chain needs a real inside-only operation.

``VfWithoutMesh`` enters the same mode with no mesh in force. A module that
declares an AIV level says what the program may use, not that every region in it
runs there, so this is refused at type inference.

The f32 element width here is the declared pset width of the f32 subset. It is
not a lane count; the AIV vector unit computes over 64 lanes regardless.
"""
from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_ascend import Ascend950PRTarget

LANES = 64


@module(entry="vf", target=Ascend950PRTarget(), topologies=(Topology("aiv", 1),))
class VfInsideMesh:
    """A mode entered under the AIV mesh, with a real GM->UB stage either side."""

    @func
    def vf(x: Tensor[(LANES,), "f32"]) -> Tensor[(LANES,), "f32"]:
        with Mesh(("aiv",), layout=(1,), names=("core",)) as aiv:
            ub = tf.empty(Tensor[(LANES,), "f32", (LANES,), "smem"])
            staged = tf.schedule((x, ub), op=T.copy())
            with T.SimdVF():
                y = staged + staged
            out = tf.empty(Tensor[(LANES,), "f32", (LANES,), "gmem"])
            return tf.schedule((y, out), op=T.copy())


@module(entry="vf", target=Ascend950PRTarget(), topologies=(Topology("aiv", 1),))
class VfWithoutMesh:
    """A mode with no mesh in force: a declaration is not a domain."""

    @func
    def vf(x: Tensor[(LANES,), "f32"]) -> Tensor[(LANES,), "f32"]:
        with T.SimdVF():
            y = x + x
            return y