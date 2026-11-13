# api-name: attention_residual
# api-kind: torch-function
# kernels: attention_residual
from pathlib import Path
import test_attention_residual as harness


def build_operator_api(operator_module):
    return operator_module.attention_residual


def build_bench_cases():
    return [{'id': str(c.case_num), 'spec': c, 'warmup': 3, 'repeats': 5}
            for c in harness.load_cases(Path(__file__).resolve().parent, 'level4/attention_residual')]


def build_bench_case_fn(operator_api, case):
    return harness.Harness('level4/attention_residual', Path(__file__).resolve().parent).bench_fn(operator_api, case)
