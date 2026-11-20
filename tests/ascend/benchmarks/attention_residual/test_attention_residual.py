# test-mode: standalone
# compute-kind: compute
# api-name: attention_residual
# api-kind: torch-function
# kernels: attention_residual
"""Standalone CANNBench AttentionResidual correctness harness.

Locates ``cannbench_contract.py`` in this checkout or an archived source
snapshot and delegates every case to the shared harness.
"""
from pathlib import Path
import sys

for parent in Path(__file__).resolve().parents:
    if (parent / 'cannbench_contract.py').is_file():
        sys.path.insert(0, str(parent))
        break
from cannbench_contract import Harness, load_cases


def main(operator_api, observer=None, node=None):
    Harness('level4/attention_residual', Path(__file__).resolve().parent).run(
        operator_api, observer=observer, node=node
    )
