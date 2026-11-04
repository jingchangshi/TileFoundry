"""M3 end-to-end: one predicated SimdVF vadd, authored as a normal source artifact.

This is the smallest program whose finalized TIR contains an ExecutionScope. It is
written here as a source file rather than assembled inline in a test, so the artifact
the acceptance test loads and the artifact a verifier is pointed at are the same kind
of object a user would write.

    GM --ascend_gm_to_ub--> UB --ascend_vld--> vreg
    vreg+mask --ascend_vadd--> vreg --ascend_vsts--> UB --ascend_ub_to_gm--> GM

The two transfers are declared outside-only and the four register instructions
inside-only, so the placement is the shape of the program rather than an example of
it. The predicate is a VALUE produced by ascend_pset, not an attribute, which is what
makes the mask checkable against the data it selects.

ascend_pset(elem_width=32) states a 32-bit ELEMENT width. It is never read as a lane
count: the AIV vector unit computes over 64 lanes and the mask is one bit per lane.

Replay with the interpreter of record, INSIDE the container (never on the host). The
conda env is the env of record; /opt/tf-venv is retired and is not evidence:

    docker exec -e PYTHONPATH=<repo>/src:<repo>/targets/ascend \
        -e TORCH_DEVICE_BACKEND_AUTOLOAD=0 tf-ascend950pr \
        /home/s00653124/.conda/envs/tilefoundry-npu/bin/python3.12 \
        -m tilefoundry.cli schedule finalize \
        examples/rmsnorm_950pr/vf_vadd.py:VfVadd <out.py>

The interpreter is named by ABSOLUTE path on purpose: under docker exec HOME is /root,
so a $HOME-relative path resolves nowhere. The output path must not be under /tmp --
the loader walks the output directory, and a scratch directory of unrelated packages
fails the import for reasons unrelated to this program.
"""
from __future__ import annotations

from tilefoundry.dsl import *
from tilefoundry_ascend import Ascend950PRTarget

LANES = 64


@module(entry="vadd", target=Ascend950PRTarget(), topologies=(Topology("aiv", 1),))
class VfVadd:
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
