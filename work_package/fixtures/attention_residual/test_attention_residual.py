# test-mode: standalone
# compute-kind: compute
# api-name: attention_residual
# api-kind: torch-function
# kernels: attention_residual
from pathlib import Path
import sys

# Find the shared helper in this checkout or an archived source snapshot.
for parent in Path(__file__).resolve().parents:
    if (parent / 'cannbench_contract.py').is_file():
        sys.path.insert(0, str(parent))
        break
from cannbench_contract import Harness, load_cases


def main(operator_api):
    Harness('level4/attention_residual', Path(__file__).resolve().parent).run(operator_api)
