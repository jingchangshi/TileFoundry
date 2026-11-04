"""P2 report: BF16 encoding/rounding and v33 FP32 reduction contract, as JSON.

Reflects the Lead RETRACTION: the accumulator count is PER CASE (rows=4 -> four chains,
balanced merge ((0+1)+(2+3)); rows<=2 fallback -> two, merge (0+1)), NOT a universal ISA
field; accumulation is FP32 after BF16 conversion, so the per-step-BF16 reducer is marked
RETRACTED and kept only as a contrast.

    PYTHONPATH=<repo>/src:<repo>/targets/ascend python -m tilefoundry_ascend.bf16_report
"""
from __future__ import annotations

import json
import math

from . import bf16
from .ir.isa import ISA, PER_CASE_ACCUMULATOR_CONTRACT


def main() -> int:
    samples = [0.0, -0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 1.0 + 2.0 ** -8, 1.0 + 3 * 2.0 ** -8,
               65504.0, 2.0 ** -130, math.inf]
    lanes = [bf16.encode(1.0)] + [bf16.encode(2.0 ** -8)] * (bf16.LANES - 1)
    rows4 = PER_CASE_ACCUMULATOR_CONTRACT["rows=4"]
    rows2 = PER_CASE_ACCUMULATOR_CONTRACT["rows<=2"]
    report = {
        "stage": "P2",
        "retraction": {
            "status": "RETRACTED",
            "wrong_claim": "global ISA accumulator_count = TWO as the claim baseline",
            "reason": (
                "reference/kernel_v33.py (SHA c6c37fb7..., P0 pin verified) uses FOUR "
                "acc0_0..acc0_3 chains on the rows=4 path (lines 127-140) and TWO only in "
                "the rows<=2 fallback (lines 253-260); the count is a per-case algorithm / "
                "scheduled-HIR contract, not an ISA field"
            ),
            "also_retracted": (
                "per-step-BF16 reduction; v33 accumulates in FP32 after BF16 conversion"
            ),
        },
        "encoding": [{"value": repr(value), "bits": f"0x{bf16.encode(value):04X}"} for value in samples],
        "carrier": {"low": "0x3F80", "high": "0x4040",
                    "packed": f"0x{bf16.pack_pair(0x3F80, 0x4040):08X}"},
        "per_case_accumulator_contract": PER_CASE_ACCUMULATOR_CONTRACT,
        "isa_vcadd": {
            "accumulator_count": ISA["vcadd"].accumulator_count,
            "reduction_order": ISA["vcadd"].reduction_order,
        },
        "reduction_fp32_per_case": {
            "rows=4": f"0x{bf16.vcadd_v33(lanes, rows4['chains'], rows4['merge']):04X}",
            "rows<=2": f"0x{bf16.vcadd_v33(lanes, rows2['chains'], rows2['merge']):04X}",
        },
        "retracted_per_step_bf16": f"0x{bf16.vcadd(lanes, bf16.ORDER_LINEAR):04X}",
        "order_is_observable": (
            bf16.vcadd_v33(lanes, rows4["chains"], rows4["merge"])
            != bf16.vcadd(lanes, bf16.ORDER_LINEAR)
        ),
        "isa_surface": {
            name: {
                "lane_count": ISA[name].lane_count,
                "element_widths": list(ISA[name].element_widths),
                "predicate_rule": ISA[name].predicate_rule,
                "rounding": ISA[name].rounding,
                "reduction_order": ISA[name].reduction_order,
            }
            for name in ("vadd", "vmul", "vmula", "vcadd")
        },
    }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
