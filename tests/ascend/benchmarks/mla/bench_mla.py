# api-name: mla
# api-kind: torch-function
# kernels: mla
from pathlib import Path
import test_mla as harness


def build_operator_api(operator_module):
    return operator_module.mla


def build_bench_cases():
    return [{'id': str(c.case_num), 'spec': c, 'warmup': 3, 'repeats': 5}
            for c in harness.load_cases(Path(__file__).resolve().parent, 'level4/mla')]


def build_bench_case_fn(operator_api, case):
    return harness.Harness('level4/mla', Path(__file__).resolve().parent).bench_fn(operator_api, case)
