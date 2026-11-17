"""Offline unit tests for WP6 counterfactual AST ablation distiller and KB admission."""

import json
from pathlib import Path

from tilefoundry_ascend.agent import BayesianRuleTracker, FlywheelDistiller, GateFunnel
from tilefoundry_ascend.agent.flywheel import (
    decompose_ast_ablations,
    synthesize_micro_probe,
)
from tilefoundry_ascend.agent.kb import load_seed_rules, safe_eval_predicate
from tilefoundry_ascend.agent.orchestrator import (
    StageBResult,
    StageCResult,
    compute_static_dynamic_residual,
    run_stage_a_static,
)

_BASELINE_DSL = """
import tilelang.ascend.language as T

def make_kernel():
    @T.prim_func
    def main(
        X: T.Tensor((128, 128), "bfloat16"),
        W: T.Tensor((128, 128), "bfloat16"),
        Y: T.Tensor((128, 128), "float32"),
    ):
        with T.Kernel(1):
            x_l1 = T.alloc_l1((128, 64), "bfloat16")
            w_l1 = T.alloc_l1((128, 64), "bfloat16")
            acc = T.alloc_l0c((128, 128), "float32")
            for kt in T.Pipelined(2, num_stages=1):
                T.copy(X[0:128, kt * 64 : (kt + 1) * 64], x_l1)
                T.copy(W[0:128, kt * 64 : (kt + 1) * 64], w_l1)
                T.gemm(x_l1, w_l1, acc, transpose_B=True, clear_accum=kt == 0)
            T.copy(acc, Y[0:128, 0:128])
    return main
"""

_WINNER_WITH_DECOY_DSL = """
import tilelang.ascend.language as T

def make_kernel():
    @T.prim_func
    def main(
        X: T.Tensor((128, 128), "bfloat16"),
        W: T.Tensor((128, 128), "bfloat16"),
        Y: T.Tensor((128, 128), "float32"),
    ):
        with T.Kernel(8):
            x_l1 = T.alloc_l1((128, 64), "bfloat16")
            w_l1 = T.alloc_l1((128, 64), "bfloat16")
            acc = T.alloc_l0c((128, 128), "float32")
            T.ascend_pipe_barrier("PIPE_ALL")
            for kt in T.Pipelined(2, num_stages=2):
                T.copy(X[0:128, kt * 64 : (kt + 1) * 64], x_l1)
                T.copy(W[0:128, kt * 64 : (kt + 1) * 64], w_l1)
                T.gemm(x_l1, w_l1, acc, transpose_B=True, clear_accum=kt == 0)
            T.copy(acc, Y[0:128, 0:128])
    return main
"""


def _mock_ablation_device(dsl_source: str, stage_a) -> StageCResult:
    assert stage_a.ledger is not None
    if "T.Kernel(8)" in dsl_source:
        lat = 42.0
    elif "num_stages=2" in dsl_source:
        lat = 78.0
    elif "ascend_pipe_barrier" in dsl_source:
        lat = 126.0
    else:
        lat = 110.0
    tel = {"dur_us": lat, "aic_mad": 0.65, "aiv_vec": 0.25, "aiv_mte2": 0.18}
    return StageCResult(
        passed=True,
        accuracy_passed=True,
        measured_latency_us=lat,
        residual=compute_static_dynamic_residual(stage_a.ledger, tel),
        telemetry=tel,
    )


def test_single_variable_ablation_and_negative_filtering(tmp_path: Path) -> None:
    """Verify AST diff splits multi-edit winner, drops negative edit, probes, and admits rules."""
    chunks = decompose_ast_ablations(_BASELINE_DSL, _WINNER_WITH_DECOY_DSL)
    kinds = {c.edit_kind for c in chunks}
    assert "core_parallelism" in kinds
    assert "pipeline_versions" in kinds
    assert "barrier_prune" in kinds
    for chunk in chunks:
        probe = synthesize_micro_probe(chunk)
        assert probe is not None
        assert 25 <= probe.line_count_baseline <= 50
        assert 25 <= probe.line_count_optimized <= 50
        assert "from __future__ import annotations" not in probe.baseline_probe_source
        assert "from __future__ import annotations" not in probe.optimized_probe_source
        sa_base = run_stage_a_static(probe.baseline_probe_source, factory_name=probe.factory_name)
        sa_opt = run_stage_a_static(probe.optimized_probe_source, factory_name=probe.factory_name)
        assert sa_base.passed and sa_opt.passed

    tracker = BayesianRuleTracker.from_seed_yaml()
    initial_count = len(tracker.rules)
    funnel = GateFunnel(
        compile_runner=lambda _dsl: StageBResult(passed=True),
        device_runner=_mock_ablation_device,
    )
    out_yaml = tmp_path / "distilled_kb.yaml"
    distiller = FlywheelDistiller(
        funnel=funnel,
        tracker=tracker,
        probe_runner=lambda _spec: {
            "baseline_dur_us": 40.0,
            "optimized_dur_us": 12.0,
            "numerical_ok": 1.0,
        },
        min_edit_gain=0.05,
        min_probe_speedup=1.2,
    )
    report = distiller.distill_from_pair(
        baseline_dsl=_BASELINE_DSL,
        winner_dsl=_WINNER_WITH_DECOY_DSL,
        baseline_latency_us=110.0,
        winner_latency_us=35.0,
        factory_name="make_kernel",
        kb_output_path=out_yaml,
    )

    assert len(report.kept_positive_chunks) == 2
    assert len(report.discarded_chunks) >= 1
    assert any(m.chunk.edit_kind == "barrier_prune" for m in report.discarded_chunks)
    assert len(report.admitted_rule_ids) == 2
    assert len(tracker.rules) == initial_count + 2

    reloaded = load_seed_rules(out_yaml)
    assert len(reloaded) == initial_count + 2
    admitted = [r for r in reloaded if r.rule_id in set(report.admitted_rule_ids)]
    for rule in admitted:
        assert rule.source_provenance == "flywheel:ablation"
        assert safe_eval_predicate(
            rule.regime_guard,
            {
                "is_legal": True,
                "any_overflow_bytes": 0,
                "has_indirect_gather": False,
                "dominant_contig_bytes": 256,
                "has_dynamic_control": False,
                "cube_to_vector_ratio": 1.0,
                "AIV0.PIPE_V": 10,
            },
        )
    summary = json.loads(report.to_json())
    assert summary["admitted_rule_ids"] == list(report.admitted_rule_ids)


def _wrap_prim_func(inner_body: str, *, header: str = "with T.Kernel(4, threads=128) as bx:") -> str:
    indented = "\n".join(f"            {line}" for line in inner_body.strip().splitlines())
    return (
        "import tilelang.language as T\n\n\n"
        "def make_kernel():\n"
        "    @T.prim_func\n"
        "    def main(\n"
        '        A: T.Tensor((16, 128), "float32"),\n'
        '        B: T.Tensor((16, 128), "float32"),\n'
        "    ):\n"
        f"        {header}\n"
        f"{indented}\n"
        "    return main\n"
    )


def test_all_seven_mechanism_probes_extract_params_and_pass_stage_a() -> None:
    """Verify all 7 mechanism classes synthesize distinct valid 25-50 line micro-probes."""
    cases = [
        (
            "core_parallelism",
            _wrap_prim_func('T.copy(A[bx, 0:128], B[bx, 0:128])', header="with T.Kernel(2, threads=128) as bx:"),
            _wrap_prim_func('T.copy(A[bx, 0:128], B[bx, 0:128])', header="with T.Kernel(24, threads=128) as bx:"),
        ),
        (
            "pipeline_versions",
            _wrap_prim_func(
                "for ko in T.Pipelined(4, num_stages=1):\n"
                "    T.copy(A[bx, 0:128], B[bx, 0:128])"
            ),
            _wrap_prim_func(
                "for ko in T.Pipelined(4, num_stages=3):\n"
                "    T.copy(A[bx, 0:128], B[bx, 0:128])"
            ),
        ),
        (
            "coalesced_copy",
            _wrap_prim_func("for i in T.serial(4):\n    T.copy(A[i, 0], B[i, 0])"),
            _wrap_prim_func("for i in T.serial(4):\n    T.copy(A[0:4, 0:128], B[0:4, 0:128])"),
        ),
        (
            "ssbuf_handoff",
            _wrap_prim_func('flag = T.alloc_shared((16,), "uint32")'),
            _wrap_prim_func('flag = T.alloc_ss((16,), "uint32")'),
        ),
        (
            "dual_copy_1c2v",
            _wrap_prim_func("T.copy(A[bx, 0:128], B[bx, 0:128])", header="with T.Kernel(8, threads=128) as bx:"),
            _wrap_prim_func("T.dual_copy(A[bx, 0:128], B[bx, 0:128])", header="with T.MixedKernel(8) as bx:"),
        ),
        (
            "barrier_prune",
            _wrap_prim_func('T.ascend_pipe_barrier("PIPE_V")\nT.copy(A[bx, 0:128], B[bx, 0:128])'),
            _wrap_prim_func('T.copy(A[bx, 0:128], B[bx, 0:128])'),
        ),
        (
            "pad_alignment",
            _wrap_prim_func('buf = T.alloc_shared((16, 70), "float16")'),
            _wrap_prim_func('buf = T.alloc_shared((16, 96), "float16")'),
        ),
    ]

    seen_kinds: set[str] = set()
    for expected_kind, base_src, opt_src in cases:
        chunks = decompose_ast_ablations(base_src, opt_src)
        matching = [c for c in chunks if c.edit_kind == expected_kind]
        assert matching, f"expected edit_kind {expected_kind}, got {[c.edit_kind for c in chunks]}"
        seen_kinds.add(expected_kind)
        probe = synthesize_micro_probe(matching[0])
        assert probe is not None
        assert probe.probe_kind == expected_kind
        assert 25 <= probe.line_count_baseline <= 50
        assert 25 <= probe.line_count_optimized <= 50
        assert probe.baseline_probe_source != probe.optimized_probe_source
        assert "from __future__ import annotations" not in probe.baseline_probe_source
        assert "from __future__ import annotations" not in probe.optimized_probe_source
        sa_base = run_stage_a_static(probe.baseline_probe_source, factory_name=probe.factory_name)
        sa_opt = run_stage_a_static(probe.optimized_probe_source, factory_name=probe.factory_name)
        assert sa_base.passed, f"{expected_kind} baseline failed Stage A: {sa_base.error_message}"
        assert sa_opt.passed, f"{expected_kind} optimized failed Stage A: {sa_opt.error_message}"

    assert len(seen_kinds) == 7

    chunks_low = decompose_ast_ablations(
        _wrap_prim_func("T.copy(A[bx, 0:128], B[bx, 0:128])", header="with T.Kernel(2, threads=128) as bx:"),
        _wrap_prim_func("T.copy(A[bx, 0:128], B[bx, 0:128])", header="with T.Kernel(8, threads=128) as bx:"),
    )
    chunks_high = decompose_ast_ablations(
        _wrap_prim_func("T.copy(A[bx, 0:128], B[bx, 0:128])", header="with T.Kernel(2, threads=128) as bx:"),
        _wrap_prim_func("T.copy(A[bx, 0:128], B[bx, 0:128])", header="with T.Kernel(24, threads=128) as bx:"),
    )
    p_low = synthesize_micro_probe(chunks_low[0])
    p_high = synthesize_micro_probe(chunks_high[0])
    assert p_low is not None and p_high is not None
    assert p_low.optimized_probe_source != p_high.optimized_probe_source


def test_unverifiable_stmt_rewrite_rejected_fail_closed(tmp_path: Path) -> None:
    """Verify arbitrary stmt_rewrite edits return no probe and never enter the KB."""
    from tilefoundry_ascend.agent.flywheel.ablation_distiller import FlywheelDistiller
    from tilefoundry_ascend.agent.kb.bayesian_tracker import BayesianRuleTracker
    from tilefoundry_ascend.agent.orchestrator.gate_funnel import (
        GateFunnel,
        StageCResult,
    )

    base_dsl = (
        "import tilelang.language as T\n\n\n"
        "def make_kernel():\n"
        "    @T.prim_func\n"
        "    def main(\n"
        '        A: T.Tensor((2, 128), "float32"),\n'
        '        B: T.Tensor((2, 128), "float32"),\n'
        "    ):\n"
        "        with T.Kernel(2, threads=128) as bx:\n"
        '            buf = T.alloc_shared((128,), "float32")\n'
        "            T.copy(A[bx, 0:128], buf)\n"
        "            for i in T.serial(128):\n"
        "                buf[i] = buf[i] + 1.0\n"
        "            T.copy(buf, B[bx, 0:128])\n"
        "    return main\n"
    )
    win_dsl = base_dsl.replace("buf[i] = buf[i] + 1.0", "buf[i] = buf[i] + 2.0")

    chunks = decompose_ast_ablations(base_dsl, win_dsl)
    assert len(chunks) == 1
    assert chunks[0].edit_kind == "stmt_rewrite"
    assert synthesize_micro_probe(chunks[0]) is None

    probe_calls: list[str] = []

    def _unexpected_probe_runner(spec):
        probe_calls.append(spec.probe_kind)
        return {"baseline_dur_us": 100.0, "optimized_dur_us": 50.0, "speedup": 2.0}

    funnel = GateFunnel(
        skip_stage_b=True,
        device_runner=lambda _dsl, _sa: StageCResult(
            passed=True,
            accuracy_passed=True,
            measured_latency_us=50.0,
            residual=None,
            telemetry={"dur_us": 50.0},
        ),
    )
    tracker = BayesianRuleTracker()
    distiller = FlywheelDistiller(
        funnel=funnel,
        tracker=tracker,
        probe_runner=_unexpected_probe_runner,
    )
    report = distiller.distill_from_pair(
        baseline_dsl=base_dsl,
        winner_dsl=win_dsl,
        baseline_latency_us=100.0,
        winner_latency_us=50.0,
        factory_name="make_kernel",
        kb_output_path=tmp_path / "flywheel_kb.yaml",
    )

    assert probe_calls == []
    assert report.admitted_rule_ids == ()
    assert len(report.distilled_outcomes) == 1
    outcome = report.distilled_outcomes[0]
    assert outcome.probe is None
    assert outcome.admitted is False
    assert outcome.hardware_verified is False
    assert outcome.rule is None
    assert outcome.rejection_reason == "unverifiable_edit_kind:stmt_rewrite"
    assert json.loads(report.to_json())["admitted_rule_ids"] == []


def test_hardware_gate_enforces_108x_and_wait_ratio_non_regression(tmp_path: Path) -> None:
    """Verify micro-probe gate enforces 1.08x speedup default and wait-ratio checks."""
    from tilefoundry_ascend.agent.flywheel.ablation_distiller import (
        FlywheelDistiller,
        verify_micro_probe_hardware,
    )
    from tilefoundry_ascend.agent.kb.schema import validate_rule_predicate
    from tilefoundry_ascend.agent.kb.bayesian_tracker import BayesianRuleTracker
    from tilefoundry_ascend.agent.orchestrator.gate_funnel import (
        GateFunnel,
        StageCResult,
    )

    base_dsl = (
        "import tilelang.language as T\n\n\n"
        "def make_kernel():\n"
        "    @T.prim_func\n"
        "    def main(\n"
        '        A: T.Tensor((2, 128), "float16"),\n'
        '        B: T.Tensor((2, 128), "float16"),\n'
        "    ):\n"
        "        with T.Kernel(2, threads=128) as bx:\n"
        '            buf = T.alloc_shared((16, 70), "float16")\n'
        "            T.copy(A[bx, 0:70], buf[0, 0:70])\n"
        "            T.copy(buf[0, 0:70], B[bx, 0:70])\n"
        "    return main\n"
    )
    win_dsl = base_dsl.replace('T.alloc_shared((16, 70), "float16")', 'T.alloc_shared((16, 96), "float16")')

    chunks = decompose_ast_ablations(base_dsl, win_dsl)
    pad_chunks = [c for c in chunks if c.edit_kind == "pad_alignment"]
    assert pad_chunks
    probe = synthesize_micro_probe(pad_chunks[0])
    assert probe is not None

    ok_106, _ = verify_micro_probe_hardware(
        probe,
        probe_runner=lambda _s: {
            "numerical_ok": True,
            "baseline_dur_us": 106.0,
            "optimized_dur_us": 100.0,
        },
    )
    assert ok_106 is False

    ok_wait_reg, _ = verify_micro_probe_hardware(
        probe,
        probe_runner=lambda _s: {
            "numerical_ok": True,
            "baseline_dur_us": 120.0,
            "optimized_dur_us": 100.0,
            "baseline_mte2_wait_ratio": 0.10,
            "optimized_mte2_wait_ratio": 0.30,
        },
    )
    assert ok_wait_reg is False

    ok_aiv_reg, _ = verify_micro_probe_hardware(
        probe,
        probe_runner=lambda _s: {
            "numerical_ok": True,
            "baseline_dur_us": 120.0,
            "optimized_dur_us": 100.0,
            "baseline_aiv_mte2": 0.40,
            "optimized_aiv_mte2": 0.70,
        },
    )
    assert ok_aiv_reg is False

    funnel = GateFunnel(
        skip_stage_b=True,
        device_runner=lambda _dsl, _sa: StageCResult(
            passed=True,
            accuracy_passed=True,
            measured_latency_us=60.0,
            residual=None,
            telemetry={"dur_us": 60.0},
        ),
    )
    tracker = BayesianRuleTracker()
    distiller = FlywheelDistiller(
        funnel=funnel,
        tracker=tracker,
        probe_runner=lambda _s: {
            "numerical_ok": True,
            "baseline_dur_us": 120.0,
            "optimized_dur_us": 100.0,
            "baseline_mte2_wait_ratio": 0.15,
            "optimized_mte2_wait_ratio": 0.10,
            "baseline_aiv_mte2": 0.40,
            "optimized_aiv_mte2": 0.35,
        },
    )
    assert distiller._min_probe_speedup == 1.08
    report = distiller.distill_from_pair(
        baseline_dsl=base_dsl,
        winner_dsl=win_dsl,
        baseline_latency_us=100.0,
        winner_latency_us=60.0,
        factory_name="make_kernel",
        kb_output_path=tmp_path / "flywheel_pad_kb.yaml",
    )
    assert len(report.admitted_rule_ids) == 1
    admitted_outcome = [o for o in report.distilled_outcomes if o.admitted][0]
    assert admitted_outcome.rule is not None
    validate_rule_predicate(admitted_outcome.rule.regime_guard)
    validate_rule_predicate(admitted_outcome.rule.precondition)
    validate_rule_predicate(admitted_outcome.rule.expected_postcondition)


def test_up2_static_pingpong_and_loop_fission_probes() -> None:
    """Verify UP2 static_pingpong_unroll and loop_fission_sfu probes synthesize and pass Stage A."""
    cases = [
        (
            "static_pingpong_unroll",
            _wrap_prim_func(
                "for i in T.Pipelined(4, num_stages=2):\n"
                "    T.copy(A[bx, 0:128], B[bx, 0:128])"
            ),
            _wrap_prim_func(
                "for i in T.serial(2):\n"
                "    T.copy(A[bx, 0:128], B[bx, 0:128])"
            ),
        ),
        (
            "loop_fission_sfu",
            _wrap_prim_func(
                "for s in T.serial(4):\n"
                "    acc_v = T.simd.vcadd(q_v)\n"
                "    B[bx, s] = acc_v"
            ),
            _wrap_prim_func(
                "for s1 in T.serial(4):\n"
                "    acc_v = T.simd.vcadd(q_v)\n"
                "for s2 in T.serial(4):\n"
                "    B[bx, s2] = A[bx, s2]"
            ),
        ),
    ]
    seen: set[str] = set()
    for kind, base_src, opt_src in cases:
        chunks = decompose_ast_ablations(base_src, opt_src)
        matching = [c for c in chunks if c.edit_kind == kind]
        assert matching, f"expected {kind}, got {[c.edit_kind for c in chunks]}"
        seen.add(kind)
        probe = synthesize_micro_probe(matching[0])
        assert probe is not None
        assert probe.probe_kind == kind
        assert 25 <= probe.line_count_baseline <= 50
        assert 25 <= probe.line_count_optimized <= 50
        sa_base = run_stage_a_static(probe.baseline_probe_source, factory_name=probe.factory_name)
        sa_opt = run_stage_a_static(probe.optimized_probe_source, factory_name=probe.factory_name)
        assert sa_base.passed, f"{kind} baseline failed Stage A: {sa_base.error_message}"
        assert sa_opt.passed, f"{kind} optimized failed Stage A: {sa_opt.error_message}"
    assert seen == {"static_pingpong_unroll", "loop_fission_sfu"}

