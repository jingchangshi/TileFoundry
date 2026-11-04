"""P2 report: BF16 encoding/rounding and reduction-order surface, as JSON.

    PYTHONPATH=<repo>/src:<repo>/targets/ascend python -m tilefoundry_ascend.bf16_report
"""
from __future__ import annotations

import json
import math

from . import bf16
from .ir.isa import ISA


def main() -> int:
    samples = [0.0, -0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 1.0 + 2.0 ** -8, 1.0 + 3 * 2.0 ** -8,
               65504.0, 2.0 ** -130, math.inf]
    lanes = [bf16.encode(1.0)] + [bf16.encode(2.0 ** -8)] * (bf16.LANES - 1)
    report = {
        "stage": "P2",
        "encoding": [{"value": repr(value), "bits": f"0x{bf16.encode(value):04X}"} for value in samples],
        "carrier": {"low": f"0x{0x3F80:04X}", "high": f"0x{0x4040:04X}",
                    "packed": f"0x{bf16.pack_pair(0x3F80, 0x4040):08X}"},
        "reduction_order_crafted_vector": {
            "linear": f"0x{bf16.vcadd(lanes, bf16.ORDER_LINEAR):04X}",
            "pairwise": f"0x{bf16.vcadd(lanes, bf16.ORDER_PAIRWISE):04X}",
            "exact_then_round": f"0x{bf16.vcadd(lanes, bf16.ORDER_EXACT):04X}",
            "order_is_observable": bf16.vcadd(lanes, bf16.ORDER_LINEAR) != bf16.vcadd(lanes, bf16.ORDER_EXACT),
        },
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
