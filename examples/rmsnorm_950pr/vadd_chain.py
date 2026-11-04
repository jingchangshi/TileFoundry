"""M3: a real SimdVF region computing a predicated vector add.

The chain is the whole boundary, not a fragment of it:

    GM --ascend_gm_to_ub--> UB --ascend_vld--> vreg
    vreg+mask --ascend_vadd--> vreg --ascend_vsts--> UB --ascend_ub_to_gm--> GM

The two transfers are declared outside-only and the four register instructions
inside-only, so the shape of the program is the placement rule rather than an
example of it. The predicate is an operand produced by ``ascend_pset``: the mask
is a value in the program, which is what makes it checkable against the data it
selects instead of being an attribute nothing can verify.

``ascend_pset(elem_width=32)`` states a 32-bit element width. That is not a lane
count and is never read as one: the AIV vector unit computes over 64 lanes, and
the mask has one bit per lane.

Replay with the interpreter of record, INSIDE the container (never on the host):

    docker exec -u s00653124 tf-ascend950pr bash -c '\
      export PYTHONPATH=<repo>/src:<repo>/targets/ascend; \
      cd <repo> && $HOME/.conda/envs/tilefoundry-npu/bin/python3.12 \
      -m tilefoundry.cli schedule finalize \
      examples/rmsnorm_950pr/vadd_chain.py:VaddChain <out.py>'

The conda env is the env of record. ``/opt/tf-venv`` is retired and must not be
quoted as evidence; a transcript from it is provisional by definition. The output
path must not be under /tmp: the loader walks the containing directory, so a
temporary directory of unrelated packages will fail the import for reasons that
have nothing to do with this program.
"""
from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_ascend import Ascend950PRTarget

LANES = 64


@module(entry="vadd", target=Ascend950PRTarget(), topologies=(Topology("aiv", 1),))
class VaddChain:
    """One predicated lane add, staged through UB either side of the mode."""

    @func
    def vadd(x: Tensor[(LANES,), "f32"]) -> Tensor[(LANES,), "f32"]:
        with Mesh(("aiv",), layout=(1,), names=("core",)) as aiv:
            loaded = tf.empty(Tensor[(LANES,), "f32", (LANES,), "smem"])
            staged = tf.schedule((x, loaded), op=T.ascend_gm_to_ub())
            with T.SimdVF():
                left = tf.empty(Tensor[(LANES,), "f32", (LANES,), "rmem"])
                right = tf.empty(Tensor[(LANES,), "f32", (LANES,), "rmem"])
                mask = tf.empty(Tensor[(LANES,), "bool", (LANES,), "rmem"])
                acc = tf.empty(Tensor[(LANES,), "f32", (LANES,), "rmem"])
                hl = tf.schedule((staged, left), op=T.ascend_vld())
                hr = tf.schedule((staged, right), op=T.ascend_vld())
                pred = tf.schedule((mask,), op=T.ascend_pset(elem_width=32))
                added = tf.schedule((hl, hr, pred, acc), op=T.ascend_vadd())
                stored = tf.empty(Tensor[(LANES,), "f32", (LANES,), "smem"])
                back = tf.schedule((added, stored), op=T.ascend_vsts())
            out = tf.empty(Tensor[(LANES,), "f32", (LANES,), "gmem"])
            return tf.schedule((back, out), op=T.ascend_ub_to_gm())