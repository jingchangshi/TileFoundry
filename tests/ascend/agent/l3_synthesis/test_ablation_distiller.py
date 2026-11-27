"""Tests for counterfactual ablation distillation, HardwareProbePort, Trajectory, and KB."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tilefoundry_ascend.agent import BayesianRuleTracker, FlywheelDistiller, GateFunnel
from tilefoundry_ascend.agent.contracts import (
    EvolutionRoundRecord,
    EvolutionTrajectory,
    HardwareProbePort,
    WorkerStepRecord,
)
from tilefoundry_ascend.agent.flywheel import decompose_ast_ablations, synthesize_micro_probe
from tilefoundry_ascend.agent.flywheel.ablation_distiller import (
    AstEditChunk,
    _build_disallowed_provenance_tokens,
    _validate_distilled_rule,
    decompose_leave_one_out_ablations,
    distill_causal_rule,
    is_paired_gain_significant,
    resolve_paired_sigma_us,
    should_trigger_flywheel_checkpoint,
    verify_micro_probe_hardware,
)
from tilefoundry_ascend.agent.kb import (
    AscendCausalRule,
    SymptomFamily,
    load_seed_rules,
    safe_eval_predicate,
    validate_rule_predicate,
)
from tilefoundry_ascend.agent.orchestrator import (
    StageBResult,
    StageCResult,
    compute_static_dynamic_residual,
    run_stage_a_static,
)
from tilefoundry_ascend.agent.specs.anchor_specs import CannbenchOpsRegistry
from tilefoundry_ascend.ascend.sil.schema import TopologyAxes

_REPO_ROOT = Path(__file__).resolve().parents[4]

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


@dataclass
class _StubProbeProfile:
    dur_us: float = 12.0
    aiv_mte2: float = 0.2
    aiv_vec: float = 0.5
    aic_fixpipe: float = 0.1


class _FakeHardwareProbePort:
    def __init__(self) -> None:
        self.cpath_calls = 0
        self.msprof_calls = 0

    def ensure_cann_include_cpath(self, env: dict[str, str] | None = None) -> dict[str, str]:
        self.cpath_calls += 1
        return dict(env or {})

    def run_msprof_detail(self, *args: object, **kwargs: object) -> _StubProbeProfile:
        _ = (args, kwargs)
        self.msprof_calls += 1
        return _StubProbeProfile()


def _wrap_prim_func(
    inner_body: str, *, header: str = "with T.Kernel(4, threads=128) as bx:"
) -> str:
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


def test_flywheel_has_zero_l3_to_l4_imports_and_uses_contracts() -> None:
    """Enforce zero L3->L4 imports in agent/flywheel and contract-based port injection."""
    flywheel_dir = _REPO_ROOT / "targets" / "ascend" / "tilefoundry_ascend" / "agent" / "flywheel"
    forbidden_prefixes = (
        "tilefoundry_ascend.agent.probes",
        "tilefoundry_ascend.agent.campaign",
        "agent.probes",
        "agent.campaign",
    )
    imported_modules: list[str] = []
    contracts_symbols: set[str] = set()
    for py_file in sorted(flywheel_dir.glob("*.py")):
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_modules.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                imported_modules.append(mod)
                if mod == "tilefoundry_ascend.agent.contracts":
                    contracts_symbols.update(alias.name for alias in node.names)

    for mod in imported_modules:
        assert not mod.startswith(forbidden_prefixes), f"Forbidden L3->L4 import: {mod}"
    assert {"EvolutionTrajectory", "HardwareProbePort"} <= contracts_symbols


def test_hardware_probe_port_injection_and_gate_thresholds(tmp_path: Path) -> None:
    """Verify HardwareProbePort injection, 3-sigma significance, and wait-ratio gates."""
    chunks = decompose_ast_ablations(_BASELINE_DSL, _WINNER_WITH_DECOY_DSL)
    probe = synthesize_micro_probe(next(c for c in chunks if c.edit_kind == "core_parallelism"))
    assert probe is not None

    with pytest.raises(RuntimeError, match="HardwareProbePort is required"):
        verify_micro_probe_hardware(probe, probe_runner=None, probe_port=None)

    fake_port = _FakeHardwareProbePort()
    assert isinstance(fake_port, object)
    port_ref: HardwareProbePort = fake_port
    if not (hasattr(torch, "npu") and torch.npu.is_available()):
        with pytest.raises(RuntimeError, match="Ascend NPU device is required"):
            verify_micro_probe_hardware(probe, probe_runner=None, probe_port=port_ref)
        assert fake_port.cpath_calls == 1

    assert resolve_paired_sigma_us(100.0, paired_rel_mad=0.05) == pytest.approx(7.413)
    assert is_paired_gain_significant(20.0, sigma_paired_us=5.0, sigma_multiplier=3.0) is True
    assert is_paired_gain_significant(10.0, sigma_paired_us=5.0, sigma_multiplier=3.0) is False
    assert should_trigger_flywheel_checkpoint(100.0, 80.0, sigma_paired_us=5.0) is True
    assert should_trigger_flywheel_checkpoint(100.0, 92.0, sigma_paired_us=5.0) is False

    ok_low, _ = verify_micro_probe_hardware(
        probe,
        probe_runner=lambda _s: {
            "numerical_ok": True,
            "baseline_dur_us": 100.5,
            "optimized_dur_us": 100.0,
            "sigma_paired_us": 1.0,
        },
    )
    assert ok_low is False

    ok_wait, _ = verify_micro_probe_hardware(
        probe,
        probe_runner=lambda _s: {
            "numerical_ok": True,
            "baseline_dur_us": 120.0,
            "optimized_dur_us": 100.0,
            "baseline_mte2_wait_ratio": 0.10,
            "optimized_mte2_wait_ratio": 0.30,
        },
    )
    assert ok_wait is False

    ok_aiv, _ = verify_micro_probe_hardware(
        probe,
        probe_runner=lambda _s: {
            "numerical_ok": True,
            "baseline_dur_us": 120.0,
            "optimized_dur_us": 100.0,
            "baseline_aiv_mte2": 0.40,
            "optimized_aiv_mte2": 0.70,
        },
    )
    assert ok_aiv is False

    distiller = FlywheelDistiller(
        funnel=GateFunnel(skip_stage_b=True),
        tracker=BayesianRuleTracker(),
        probe_port=port_ref,
    )
    assert distiller._probe_port is fake_port
    _ = tmp_path


def test_single_variable_ablation_probes_and_kb_admission(tmp_path: Path) -> None:
    """Decompose edits, drop negative decoy, validate 9 probe kinds, and admit YAML rules."""
    cases = [
        (
            "core_parallelism",
            _wrap_prim_func(
                "T.copy(A[bx, 0:128], B[bx, 0:128])", header="with T.Kernel(2, threads=128) as bx:"
            ),
            _wrap_prim_func(
                "T.copy(A[bx, 0:128], B[bx, 0:128])", header="with T.Kernel(24, threads=128) as bx:"
            ),
        ),
        (
            "pipeline_versions",
            _wrap_prim_func(
                "for ko in T.Pipelined(4, num_stages=1):\n    T.copy(A[bx, 0:128], B[bx, 0:128])"
            ),
            _wrap_prim_func(
                "for ko in T.Pipelined(4, num_stages=3):\n    T.copy(A[bx, 0:128], B[bx, 0:128])"
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
            _wrap_prim_func(
                "T.copy(A[bx, 0:128], B[bx, 0:128])", header="with T.Kernel(8, threads=128) as bx:"
            ),
            _wrap_prim_func(
                "T.dual_copy(A[bx, 0:128], B[bx, 0:128])", header="with T.MixedKernel(8) as bx:"
            ),
        ),
        (
            "barrier_prune",
            _wrap_prim_func('T.ascend_pipe_barrier("PIPE_V")\nT.copy(A[bx, 0:128], B[bx, 0:128])'),
            _wrap_prim_func("T.copy(A[bx, 0:128], B[bx, 0:128])"),
        ),
        (
            "pad_alignment",
            _wrap_prim_func('buf = T.alloc_shared((16, 70), "float16")'),
            _wrap_prim_func('buf = T.alloc_shared((16, 96), "float16")'),
        ),
        (
            "static_pingpong_unroll",
            _wrap_prim_func(
                "for i in T.Pipelined(4, num_stages=2):\n    T.copy(A[bx, 0:128], B[bx, 0:128])"
            ),
            _wrap_prim_func("for i in T.serial(2):\n    T.copy(A[bx, 0:128], B[bx, 0:128])"),
        ),
        (
            "loop_fission_sfu",
            _wrap_prim_func(
                "for s in T.serial(4):\n    acc_v = T.simd.vcadd(q_v)\n    B[bx, s] = acc_v"
            ),
            _wrap_prim_func(
                "for s1 in T.serial(4):\n    acc_v = T.simd.vcadd(q_v)\n"
                "for s2 in T.serial(4):\n    B[bx, s2] = A[bx, s2]"
            ),
        ),
    ]
    seen_kinds: set[str] = set()
    for kind, b_src, o_src in cases:
        chunks = decompose_ast_ablations(b_src, o_src)
        matched = [c for c in chunks if c.edit_kind == kind]
        assert matched, f"missing {kind}"
        seen_kinds.add(kind)
        probe = synthesize_micro_probe(matched[0])
        assert probe is not None
        assert 25 <= probe.line_count_baseline <= 50
        assert 25 <= probe.line_count_optimized <= 50
        assert run_stage_a_static(
            probe.baseline_probe_source, factory_name=probe.factory_name
        ).passed
        assert run_stage_a_static(
            probe.optimized_probe_source, factory_name=probe.factory_name
        ).passed
    assert len(seen_kinds) == 9

    tracker = BayesianRuleTracker.from_seed_yaml()
    initial_count = len(tracker.rules)
    out_yaml = tmp_path / "distilled_kb.yaml"
    distiller = FlywheelDistiller(
        funnel=GateFunnel(
            compile_runner=lambda _d: StageBResult(passed=True),
            device_runner=_mock_ablation_device,
        ),
        tracker=tracker,
        probe_runner=lambda _s: {
            "baseline_dur_us": 40.0, "optimized_dur_us": 12.0, "numerical_ok": 1.0
        },
    )
    report = distiller.distill_from_pair(
        baseline_dsl=_BASELINE_DSL,
        winner_dsl=_WINNER_WITH_DECOY_DSL,
        baseline_latency_us=110.0,
        winner_latency_us=35.0,
        factory_name="make_kernel",
        topology_axes=TopologyAxes("coupled_1c2v", "gm_l1_l0_fixpipe", "affine_simd"),
        kb_output_path=out_yaml,
    )
    assert len(report.kept_positive_chunks) == 2
    assert any(m.chunk.edit_kind == "barrier_prune" for m in report.discarded_chunks)
    assert len(report.admitted_rule_ids) == 2
    assert len(tracker.rules) == initial_count + 2
    reloaded = load_seed_rules(out_yaml)
    assert len(reloaded) == initial_count + 2
    for rid in report.admitted_rule_ids:
        assert rid in tracker.axis_posteriors.get("mr:gm_l1_l0_fixpipe", {})
        rule = next(r for r in reloaded if r.rule_id == rid)
        assert rule.source_provenance == "flywheel:ablation"
        assert safe_eval_predicate(
            rule.regime_guard,
            {"is_legal": True, "any_overflow_bytes": 0, "has_indirect_gather": False},
        )
    assert json.loads(report.to_json())["admitted_rule_ids"] == list(report.admitted_rule_ids)


def test_distill_from_trajectory_and_fail_closed_stmt_rewrite(tmp_path: Path) -> None:
    """Distill from EvolutionTrajectory and reject unverifiable stmt_rewrite edits."""
    step = WorkerStepRecord(
        worker_id="w1",
        worker_kind="creative_explore",
        rule_id=None,
        repaired=False,
        passed=True,
        failed_stage=None,
        latency_us=35.0,
        estimated_cycles=100.0,
        dominant_pipe="AIC.PIPE_M",
        unexplained_gap_ratio=0.1,
        hypothesis="Scale blocks and pipeline stages",
        dsl_sha256="abc123",
    )
    round_rec = EvolutionRoundRecord(
        round_idx=1,
        parent_id="seed",
        parent_latency_us=110.0,
        parent_dominant_pipe="AIC.PIPE_M",
        workers=(step,),
        exploit_beam_ids=("w1",),
        stepping_stone_beam_ids=(),
        pruned_rule_ids=(),
        best_round_latency_us=35.0,
        best_so_far_latency_us=35.0,
    )
    traj = EvolutionTrajectory(
        problem_name="demo_gemm",
        track="unconstrained",
        total_rounds=1,
        baseline_candidate_id="seed",
        baseline_latency_us=110.0,
        baseline_estimated_cycles=400.0,
        best_candidate_id="w1",
        best_latency_us=35.0,
        best_estimated_cycles=100.0,
        speedup_vs_baseline=110.0 / 35.0,
        best_dsl_source=_WINNER_WITH_DECOY_DSL,
        rounds=(round_rec,),
    )
    assert json.loads(traj.to_json())["best_candidate_id"] == "w1"

    tracker = BayesianRuleTracker()
    distiller = FlywheelDistiller(
        funnel=GateFunnel(
            compile_runner=lambda _d: StageBResult(passed=True),
            device_runner=_mock_ablation_device,
        ),
        tracker=tracker,
        probe_runner=lambda _s: {
            "baseline_dur_us": 40.0, "optimized_dur_us": 10.0, "numerical_ok": 1.0
        },
    )
    traj_report = distiller.distill_from_trajectory(
        baseline_dsl=_BASELINE_DSL,
        trajectory=traj,
        factory_name="make_kernel",
        kb_output_path=tmp_path / "traj_kb.yaml",
    )
    assert len(traj_report.admitted_rule_ids) == 2

    base_stmt = (
        "import tilelang.language as T\n\n\n"
        "def make_kernel():\n"
        "    @T.prim_func\n"
        '    def main(A: T.Tensor((2, 128), "float32"), B: T.Tensor((2, 128), "float32")):\n'
        "        with T.Kernel(2) as bx:\n"
        '            buf = T.alloc_shared((128,), "float32")\n'
        "            T.copy(A[bx, 0:128], buf)\n"
        "            for i in T.serial(128):\n"
        "                buf[i] = buf[i] + 1.0\n"
        "            T.copy(buf, B[bx, 0:128])\n"
        "    return main\n"
    )
    win_stmt = base_stmt.replace("buf[i] + 1.0", "buf[i] + 2.0")
    probe_calls: list[str] = []
    stmt_distiller = FlywheelDistiller(
        funnel=GateFunnel(
            skip_stage_b=True,
            device_runner=lambda _d, _sa: StageCResult(
                passed=True,
                accuracy_passed=True,
                measured_latency_us=50.0,
                residual=None,
                telemetry={"dur_us": 50.0},
            ),
        ),
        tracker=BayesianRuleTracker(),
        probe_runner=lambda sp: (
            probe_calls.append(sp.probe_kind),
            {"baseline_dur_us": 100.0, "optimized_dur_us": 50.0},
        )[1],
    )
    stmt_report = stmt_distiller.distill_from_pair(
        baseline_dsl=base_stmt,
        winner_dsl=win_stmt,
        baseline_latency_us=100.0,
        winner_latency_us=50.0,
        factory_name="make_kernel",
        kb_output_path=tmp_path / "stmt_kb.yaml",
    )
    assert probe_calls == []
    assert stmt_report.admitted_rule_ids == ()
    assert (
        stmt_report.distilled_outcomes[0].rejection_reason == "unverifiable_edit_kind:stmt_rewrite"
    )


def test_leave_one_out_cpp_ablation_and_kb_admission_guards(tmp_path: Path) -> None:
    """Verify Track-B C++ LOO decomposition, provenance guards, and predicate validation."""
    base_tb = (
        '_SOFTMAX_TRACKB_SRC = r"""\n'
        "#include <tl_templates/ascend/common.h>\n"
        "__aicore__ void kernel_entry(int n_iters) {\n"
        "  int stride = 64;\n"
        "  for (int i = 0; i < n_iters; ++i) {\n"
        "    asc_sync_notify(PIPE_MTE2, PIPE_V, EVENT_ID0);\n"
        "  }\n"
        "  simd_inst::vsts_norm(vec_out, ub_dst, 0, m32);\n"
        "}\n"
        '"""\n'
    )
    win_tb = (
        '_SOFTMAX_TRACKB_SRC = r"""\n'
        "#include <tl_templates/ascend/common.h>\n"
        "__aicore__ void kernel_entry(int n_iters) {\n"
        "  int stride = 64 + 16;\n"
        "  for (int i = 0; i < n_iters; ++i) {\n"
        "    asc_sync_notify(PIPE_MTE2, PIPE_V, (event_t)(i & 1));\n"
        "  }\n"
        "  simd_inst::vsts_pack_b32(vec_out16, ub_dst, 0, m32);\n"
        "}\n"
        '"""\n'
    )
    loo_chunks = decompose_leave_one_out_ablations(base_tb, win_tb)
    assert len(loo_chunks) == 3
    for ch in loo_chunks:
        rule = distill_causal_rule(ch, rule_id=f"R_{ch.chunk_id}")
        assert rule.compatible_backends == ("ascendc_trackb", "hybrid_trackb")
        assert _validate_distilled_rule(rule) == (True, "")

    base_kw = {
        "rule_id": "R_GUARD",
        "regime_guard": "is_legal and any_overflow_bytes == 0",
        "precondition": "dma_burst_efficiency_ratio < 0.6 and PIPE_MTE2 > 0",
        "transform_template": "Coalesce bursts to saturate memory pipe.",
        "expected_postcondition": "dma_burst_efficiency_ratio > prev.dma_burst_efficiency_ratio",
        "source_provenance": "flywheel:ablation",
        "compatible_backends": ("tilelang_dsl",),
        "symptom_family": SymptomFamily.DMA_UNALIGNED_SHORT_BURST.value,
    }
    assert _validate_distilled_rule(AscendCausalRule(**base_kw)) == (True, "")
    for overrides in (
        {"precondition": "ub_spill_bytes >= 0"},
        {"expected_postcondition": "x <= prev.x"},
        {"symptom_family": ""},
        {"precondition": "unknown_sig > 0"},
        {"transform_template": "Read /tmp/a/b/c"},
        {"compatible_backends": ("ascendc_trackb",), "cpp20_transform_template": ""},
    ):
        assert _validate_distilled_rule(AscendCausalRule(**{**base_kw, **overrides}))[0] is False
    base_rule = AscendCausalRule(**base_kw)
    bad_facets = SimpleNamespace(
        **{**base_rule.__dict__, "recommended_facets": ("bad_facet",)},
        as_six_tuple=base_rule.as_six_tuple,
    )
    assert _validate_distilled_rule(bad_facets)[0] is False

    disallowed = _build_disallowed_provenance_tokens()
    all_ops = CannbenchOpsRegistry.default().list_operators()
    assert len(all_ops) >= 60
    for op_id in all_ops[:15]:
        assert op_id.lower() in disallowed
        leaky = AscendCausalRule(**{**base_kw, "transform_template": f"Tune for {op_id} kernel."})
        assert _validate_distilled_rule(leaky)[0] is False

    clean_chunk = AstEditChunk("c_ok", "pad_alignment", "module", "64", "64 + 16", win_tb)
    clean_rule = distill_causal_rule(clean_chunk, rule_id="R_CLEAN")
    validate_rule_predicate(clean_rule.precondition)

    loo_tracker = BayesianRuleTracker()
    loo_distiller = FlywheelDistiller(
        funnel=GateFunnel(
            skip_stage_b=True,
            device_runner=lambda src, _sa: StageCResult(
                passed=True,
                accuracy_passed=True,
                measured_latency_us=20.0 if "T.Kernel(8)" in src else 45.0,
                residual=None,
                telemetry={"dur_us": 20.0 if "T.Kernel(8)" in src else 45.0},
            ),
        ),
        tracker=loo_tracker,
    )
    loo_report = loo_distiller.distill_from_pair(
        baseline_dsl=_BASELINE_DSL,
        winner_dsl=_WINNER_WITH_DECOY_DSL,
        baseline_latency_us=60.0,
        winner_latency_us=20.0,
        factory_name="make_kernel",
        verify_on_candidate=True,
        kb_output_path=tmp_path / "loo_kb.yaml",
    )
    assert loo_report.total_chunks >= 1
    assert loo_report.admitted_rule_ids
