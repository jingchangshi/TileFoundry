"""Authored HIR: mathematical RMSNorm (independent of v33).

Used for TF-native parser/type-inference evidence via:
  python -m tilefoundry.cli analyze examples/rmsnorm_950pr/model.py:RMSNormAuthored
The Ascend Scheduled HIR (instruction level) lives in
targets/ascend/tilefoundry_ascend (schedule/program.py) and is evaluated by its
own reference instruction evaluator.
"""
from tilefoundry.dsl import *
from tilefoundry.target import CpuTarget

BATCH = 8
D = 4096
EPS = 1e-6


@module(entry="rmsnorm", target=CpuTarget())
class RMSNormAuthored:
    @func
    def rmsnorm(x: Tensor[(BATCH, D), "bf16"], w: Tensor[(D,), "bf16"]) -> Tensor[(BATCH, D), "bf16"]:
        # Y = x * rsqrt(mean(x^2, -1) + eps) * w  (no Qwen "1+w")
        return tf.rms_norm(x, w, eps=EPS)
