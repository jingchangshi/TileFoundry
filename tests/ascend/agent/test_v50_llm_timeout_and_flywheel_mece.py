"""Tests for B7 LLM semaphore/timeout decoupling and B8 slice/flywheel gates."""

from __future__ import annotations

import ast
import asyncio
import importlib.util
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from tilefoundry_ascend.agent import (
    BayesianRuleTracker,
    GateFunnel,
    MockLLMBackend,
    PortfolioRunner,
    TwoLaneBeamSearch,
)
from tilefoundry_ascend.agent.flywheel.ablation_distiller import (
    AstEditChunk,
    FlywheelDistiller,
    _is_rule_text_clean,
    _validate_distilled_rule,
    distill_causal_rule,
)
from tilefoundry_ascend.agent.kb.schema import (
    AscendCausalRule,
    SymptomFamily,
    load_seed_rules,
    safe_eval_predicate,
)
from tilefoundry_ascend.agent.orchestrator import (
    BeamCandidate,
    StageBResult,
    StageCResult,
    compute_static_dynamic_residual,
    extract_structural_signature,
    portfolio_runner as portfolio_mod,
)
from tilefoundry_ascend.agent.orchestrator import cpp_slice_protocol as slice_mod
from tilefoundry_ascend.agent.orchestrator.cpp_slice_protocol import compact_creative_source_slice
from tilefoundry_ascend.agent.orchestrator.llm_client import (
    AgyCliLLMBackend,
    GenerationRequest,
    StreamWatchdogConfig,
)
from tilefoundry_ascend.agent.orchestrator.portfolio_runner import build_creative_request
from tilefoundry_ascend.ascend.passes.physical_ledger import (
    build_objective_physical_ledger,
    build_scope_waterline,
)


def _write_fake_agy_script(tmp_path: Path, script_name: str = "fake_agy") -> tuple[Path, Path]:
    state_dir = tmp_path / "agy_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    bin_path = tmp_path / script_name
    body = "\n".join(
        [
            f"#!{sys.executable}",
            "import json, os, sys, time",
            "from pathlib import Path",
            "state_dir = Path(os.environ['TF_FAKE_AGY_STATE_DIR'])",
            "counter = state_dir / 'count.txt'",
            "idx = int(counter.read_text(encoding='utf-8')) + 1 if counter.exists() else 1",
            "counter.write_text(str(idx), encoding='utf-8')",
            "argv = sys.argv[1:]",
            "model = argv[argv.index('--model') + 1] if '--model' in argv else ''",
            "effort = next((a.split('=', 1)[1] for a in argv if a.startswith('--effort=')), '')",
            "pt = next((a.split('=', 1)[1] for a in argv if a.startswith('--print-timeout=')), '')",
            "rec = {'idx': idx, 'model': model, 'effort': effort, 'print_timeout': pt}",
            "(state_dir / f'call_{idx}.json').write_text(json.dumps(rec), encoding='utf-8')",
            "scenario = os.environ.get('TF_FAKE_AGY_SCENARIO', 'fast_ok')",
            "if scenario == 'fast_after_sem':",
            "    time.sleep(0.15)",
            "    print(json.dumps({'event': 'init'}), flush=True)",
            "    res = {'event': 'result', 'result': {'status': 'SUCCESS', 'response': 'sem_ok'}}",
            "    print(json.dumps(res), flush=True)",
            "    sys.exit(0)",
            "if scenario == 'hang_on_high':",
            "    print(json.dumps({'event': 'init'}), flush=True)",
            "    if model == 'gemini-4-hydrogen-high':",
            "        time.sleep(0.8)",
            "        sys.exit(1)",
            "    time.sleep(0.05)",
            "    res = {'event': 'result', 'result': {'status': 'SUCCESS', 'response': 'medium_ok'}}",
            "    print(json.dumps(res), flush=True)",
            "    sys.exit(0)",
            "if scenario == 'hang_always':",
            "    print(json.dumps({'event': 'init'}), flush=True)",
            "    time.sleep(0.8)",
            "    sys.exit(1)",
        ]
    )
    bin_path.write_text(body + "\n", encoding="utf-8")
    bin_path.chmod(0o755)
    return bin_path, state_dir


def test_agy_semaphore_wait_excluded_from_active_ceiling_and_startup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Queueing on _sem for > active_ceiling must not trip startup or active_ceiling."""
    bin_path, state_dir = _write_fake_agy_script(tmp_path)
    monkeypatch.setenv("TF_FAKE_AGY_STATE_DIR", str(state_dir))
    monkeypatch.setenv("TF_FAKE_AGY_SCENARIO", "fast_after_sem")

    backend = AgyCliLLMBackend(
        bin_path=str(bin_path),
        max_concurrency=1,
        watchdog=StreamWatchdogConfig(
            startup_timeout_sec=0.8,
            idle_stall_timeout_sec=0.8,
            active_ceiling_timeout_sec=1.0,
            max_retries=0,
        ),
    )

    async def _run() -> str:
        await backend._sem.acquire()
        queued = asyncio.create_task(
            backend.generate(GenerationRequest("exploit_rule", "sys", "usr"))
        )
        await asyncio.sleep(1.5)
        backend._sem.release()
        return await queued

    t0 = time.monotonic()
    reply = asyncio.run(_run())
    elapsed = time.monotonic() - t0
    assert reply == "sem_ok"
    assert elapsed >= 1.5


def test_agy_creative_tiered_ceilings_and_consecutive_timeout_cooldown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify 480s/720s creative ceilings, --print-timeout sync, and 2-hit cooldown."""
    bin_path, state_dir = _write_fake_agy_script(tmp_path)
    monkeypatch.setenv("TF_FAKE_AGY_STATE_DIR", str(state_dir))

    default_backend = AgyCliLLMBackend(bin_path=str(bin_path))
    std_creative = GenerationRequest("explore_creative", "sys", "usr")
    ps_flag_req = GenerationRequest("explore_creative", "sys", "usr", paradigm_shift=True)
    ps_prompt_req = GenerationRequest(
        "explore_creative",
        "sys",
        "Propose a ParadigmShift full_slice_rewrite across cores.",
    )
    exploit_req = GenerationRequest("exploit_rule", "sys", "usr")

    assert default_backend.watchdog_for_request(std_creative).active_ceiling_timeout_sec == 480.0
    assert "--print-timeout=480s" in default_backend.build_command(std_creative, "P")
    assert default_backend.watchdog_for_request(ps_flag_req).active_ceiling_timeout_sec == 720.0
    assert "--print-timeout=720s" in default_backend.build_command(ps_flag_req, "P")
    assert default_backend.watchdog_for_request(ps_prompt_req).active_ceiling_timeout_sec == 720.0
    assert "--print-timeout=720s" in default_backend.build_command(ps_prompt_req, "P")
    assert default_backend.watchdog_for_request(exploit_req).active_ceiling_timeout_sec == 360.0
    assert "--print-timeout=360s" in default_backend.build_command(exploit_req, "P")

    monkeypatch.setenv("TF_FAKE_AGY_SCENARIO", "hang_on_high")
    backend = AgyCliLLMBackend(
        bin_path=str(bin_path),
        watchdog=StreamWatchdogConfig(
            startup_timeout_sec=0.5,
            idle_stall_timeout_sec=0.5,
            active_ceiling_timeout_sec=0.25,
            max_retries=0,
        ),
    )

    req_alpha = GenerationRequest("explore_creative", "sys", "usr", session_scope="regime_alpha")
    req_beta = GenerationRequest("explore_creative", "sys", "usr", session_scope="regime_beta")

    with pytest.raises(TimeoutError, match="active_ceiling_timeout"):
        asyncio.run(backend.generate(req_alpha))
    with pytest.raises(TimeoutError, match="active_ceiling_timeout"):
        asyncio.run(backend.generate(req_beta))
    with pytest.raises(TimeoutError, match="active_ceiling_timeout"):
        asyncio.run(backend.generate(req_alpha))

    reply = asyncio.run(backend.generate(req_alpha))
    assert reply == "medium_ok"

    call4 = json.loads((state_dir / "call_4.json").read_text(encoding="utf-8"))
    assert call4["model"] == "gemini-4-hydrogen-medium"
    assert call4["effort"] == "medium"
    assert "xhigh" not in call4["model"]

    with pytest.raises(TimeoutError, match="active_ceiling_timeout"):
        asyncio.run(backend.generate(req_alpha))
    call5 = json.loads((state_dir / "call_5.json").read_text(encoding="utf-8"))
    assert call5["model"] == "gemini-4-hydrogen-high"

    monkeypatch.setenv("TF_FAKE_AGY_SCENARIO", "hang_always")
    with pytest.raises(TimeoutError, match="active_ceiling_timeout"):
        asyncio.run(backend.generate(req_alpha))
    with pytest.raises(TimeoutError, match="active_ceiling_timeout"):
        asyncio.run(backend.generate(req_alpha))
    call7 = json.loads((state_dir / "call_7.json").read_text(encoding="utf-8"))
    assert call7["model"] == "gemini-4-hydrogen-medium"
    with pytest.raises(TimeoutError, match="active_ceiling_timeout"):
        asyncio.run(backend.generate(req_alpha))
    call8 = json.loads((state_dir / "call_8.json").read_text(encoding="utf-8"))
    assert call8["model"] == "gemini-4-hydrogen-high"


def _sample_small_dsl() -> str:
    return (
        "import tilelang.ascend.language as T\n\n"
        "def make_kernel():\n"
        "    @T.prim_func\n"
        "    def main(\n"
        "        X: T.Tensor((128, 128), 'bfloat16'),\n"
        "        W: T.Tensor((128, 128), 'bfloat16'),\n"
        "        Y: T.Tensor((128, 128), 'float32'),\n"
        "    ):\n"
        "        with T.Kernel(2):\n"
        "            x_l1 = T.alloc_l1((128, 64), 'bfloat16')\n"
        "            w_l1 = T.alloc_l1((128, 64), 'bfloat16')\n"
        "            acc = T.alloc_l0c((128, 128), 'float32')\n"
        "            for kt in T.Pipelined(2, num_stages=2):\n"
        "                T.copy(X[0:128, kt * 64 : (kt + 1) * 64], x_l1)\n"
        "                T.copy(W[0:128, kt * 64 : (kt + 1) * 64], w_l1)\n"
        "                T.gemm(x_l1, w_l1, acc, transpose_B=True, clear_accum=kt == 0)\n"
        "            T.copy(acc, Y[0:128, 0:128])\n"
        "    return main\n"
    )


def test_slice_helpers_sink_to_cpp_slice_protocol_and_creative_14kb_budget() -> None:
    """Verify slice helper unification, deleted folding helpers, and lossless creative slicing."""
    for removed in (
        "_ANY_FOLDING_MARKER_RE",
        "_compact_trackb_repair_source",
        "_resplice_trackb_omitted_spans",
        "_splice_cpp_omitted_spans",
    ):
        assert not hasattr(portfolio_mod, removed)
        assert not hasattr(slice_mod, removed)
    for attr in (
        "_detect_trackb_prologue_line_offset",
        "_find_python_cpp_literal_spans",
        "_looks_like_python_primfunc_slice",
    ):
        assert getattr(portfolio_mod, attr) is getattr(slice_mod, attr)

    helpers = "\n\n".join(
        "\n".join(
            [
                f"def _static_readonly_helper_{idx}(x):",
                *(f"    val_{j} = x + {idx * 100 + j}" for j in range(35)),
                "    return val_34",
            ]
        )
        for idx in range(25)
    )
    large_py_dsl = helpers + "\n\n" + _sample_small_dsl()

    folded = compact_creative_source_slice(large_py_dsl)
    assert folded == large_py_dsl
    ast.parse(folded)
    assert "@T.prim_func" in folded
    assert "T.gemm" in folded
    assert "READ-ONLY HELPER FOLDED" not in folded

    trackb_cpp = (
        "#include <tl_templates/ascend/common.h>\n"
        "__aicore__ inline float live_helper(float v) { return v + 1.0f; }\n"
        'extern "C" __global__ __vector__ void active_k(__gm__ float* x) {\n'
        "    x[0] = live_helper(x[0]);\n"
        "}\n"
        'extern "C" __global__ __vector__ void sibling_k(__gm__ float* y) {\n'
        "    y[0] = 2.0f;\n"
        "}\n"
    )
    scoped_cpp = compact_creative_source_slice(trackb_cpp, active_kernel="active_k")
    assert "x[0] = live_helper(x[0]);" in scoped_cpp
    assert "live_helper(float v)" in scoped_cpp
    assert "// [READONLY_SIBLING_DECLARATION]" in scoped_cpp
    assert "y[0] = 2.0f;" not in scoped_cpp
    assert "READ-ONLY HELPER FOLDED" not in scoped_cpp

    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=10,
        pipe_cycles={"AIC.PIPE_M": 64, "AIV0.PIPE_V": 32},
        waterlines={
            "UB": build_scope_waterline(
                scope="UB",
                capacity_bytes=192 * 1024,
                explicit_buffers={"ub": 4096},
                implicit_buffers={},
            )
        },
    )
    res = compute_static_dynamic_residual(ledger, {"dur_us": 20.0})
    parent = BeamCandidate(
        candidate_id="p_large",
        round_idx=0,
        worker_kind="seed",
        rule_id=None,
        dsl_source=large_py_dsl,
        track="developer",
        ledger=ledger,
        residual=res,
        latency_us=20.0,
        accuracy_passed=True,
        structural_signature=extract_structural_signature(large_py_dsl),
    )
    req = build_creative_request(parent)
    src_block = req.user_prompt.split("### CURRENT KERNEL SOURCE\n```python\n", 1)[1].rsplit(
        "\n```\n", 1
    )[0]
    assert src_block == large_py_dsl.strip()
    ast.parse(src_block)


def test_portfolio_plan_workers_respects_cost_gate_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify PortfolioRunner.plan_workers suppresses prescreen when CostGatePolicy disengages."""
    _ = monkeypatch

    sa = GateFunnel(skip_stage_b=True).evaluate(
        _sample_small_dsl(),
        factory_name="make_kernel",
    )
    assert sa.stage_a is not None and sa.stage_a.ledger is not None
    res = compute_static_dynamic_residual(sa.stage_a.ledger, {"dur_us": 40.0})
    parent = BeamCandidate(
        candidate_id="seed_0",
        round_idx=0,
        worker_kind="seed",
        rule_id=None,
        dsl_source=_sample_small_dsl(),
        track="developer",
        ledger=sa.stage_a.ledger,
        residual=res,
        latency_us=40.0,
        accuracy_passed=True,
        features=sa.stage_a.features,
    )
    runner = PortfolioRunner(
        backend=MockLLMBackend(),
        funnel=GateFunnel(
            compile_runner=lambda _s: StageBResult(passed=True),
            device_runner=lambda _s, st: StageCResult(
                passed=True,
                accuracy_passed=True,
                measured_latency_us=20.0,
                residual=compute_static_dynamic_residual(st.ledger, {"dur_us": 20.0}),
                telemetry={"dur_us": 20.0},
            ),
        ),
        tracker=BayesianRuleTracker.from_seed_yaml(),
        enable_tilesize_prescreen=True,
        cost_model_mode="advisory",
    )

    engaged_specs = runner.plan_workers(
        parent,
        round_idx=1,
        factory_kwargs={"bm": 64, "bn": 64, "bk": 64},
        bucket_clean_samples=30,
        out_of_sample_spearman_rho=0.72,
    )
    engaged_creative = next(s for s in engaged_specs if s.worker_kind == "creative_explore")
    assert "### TILE PRESCREEN GUIDANCE" in engaged_creative.request.user_prompt

    disengaged_specs = runner.plan_workers(
        parent,
        round_idx=2,
        factory_kwargs={"bm": 64, "bn": 64, "bk": 64},
        bucket_clean_samples=2,
        out_of_sample_spearman_rho=0.72,
    )
    disengaged_creative = next(s for s in disengaged_specs if s.worker_kind == "creative_explore")
    assert "### TILE PRESCREEN GUIDANCE" not in disengaged_creative.request.user_prompt


def test_flywheel_distiller_strict_monotonic_and_dual_track_gates(tmp_path: Path) -> None:
    """Verify tautological preconditions, <= postconditions, bad facets, and Track-B gates."""
    base_kwargs = {
        "rule_id": "R_GATE_TEST",
        "regime_guard": "is_legal and any_overflow_bytes == 0",
        "precondition": "degraded_buffer_count > 0 and PIPE_V > 0",
        "transform_template": "Apply coalesced 2D vector buffering without barriers.",
        "expected_postcondition": "degraded_buffer_count < prev.degraded_buffer_count",
        "source_provenance": "flywheel:ablation",
        "compatible_backends": ("tilelang_dsl",),
        "symptom_family": SymptomFamily.SYNC_PSEUDO_DOUBLE_BUFFER.value,
    }

    taut_rule = AscendCausalRule(**{**base_kwargs, "precondition": "degraded_buffer_count >= 0"})
    ok, reason = _validate_distilled_rule(taut_rule)
    assert ok is False
    assert "tautological" in reason

    taut_and_rule = AscendCausalRule(
        **{**base_kwargs, "precondition": "degraded_buffer_count >= 0 and PIPE_V > 0"}
    )
    ok_and, reason_and = _validate_distilled_rule(taut_and_rule)
    assert ok_and is False
    assert "tautological" in reason_and

    nonstrict_rule = AscendCausalRule(
        **{
            **base_kwargs,
            "expected_postcondition": "degraded_buffer_count <= prev.degraded_buffer_count",
        }
    )
    ok_post, reason_post = _validate_distilled_rule(nonstrict_rule)
    assert ok_post is False
    assert "non-strict" in reason_post

    no_facet_rule = AscendCausalRule(**{**base_kwargs, "symptom_family": ""})
    ok_facet, reason_facet = _validate_distilled_rule(no_facet_rule)
    assert ok_facet is False
    assert "recommended_facets" in reason_facet

    bad_facet_rule = AscendCausalRule(**base_kwargs)
    bad_facet_ns = SimpleNamespace(
        **{**bad_facet_rule.__dict__, "recommended_facets": ("unknown_facet",)},
        as_six_tuple=bad_facet_rule.as_six_tuple,
    )
    ok_bf, reason_bf = _validate_distilled_rule(bad_facet_ns)
    assert ok_bf is False
    assert "recommended_facets" in reason_bf

    missing_cpp_rule = AscendCausalRule(
        **{
            **base_kwargs,
            "compatible_backends": ("tilelang_dsl", "ascendc_trackb", "hybrid_trackb"),
            "cpp20_transform_template": "",
        }
    )
    ok_tb, reason_tb = _validate_distilled_rule(missing_cpp_rule)
    assert ok_tb is False
    assert "cpp20_transform_template" in reason_tb

    fallback_chunk = AstEditChunk(
        chunk_id="c_fallback",
        edit_kind="unclassified_edit",
        target_path="main:0",
        before_src="a = 1",
        after_src="a = 2",
        ablated_dsl=_sample_small_dsl(),
    )
    fb_rule = distill_causal_rule(fallback_chunk, rule_id="R_FALLBACK_OK")
    assert fb_rule.precondition == "degraded_buffer_count > 0 and (PIPE_V > 0 or PIPE_MTE2 > 0)"
    assert (
        fb_rule.expected_postcondition
        == "degraded_buffer_count < prev.degraded_buffer_count and any_overflow_bytes == 0"
    )
    assert fb_rule.compatible_backends == ("tilelang_dsl",)
    assert _validate_distilled_rule(fb_rule) == (True, "")
    assert _is_rule_text_clean(fb_rule) == (True, "")
    assert (
        safe_eval_predicate(
            fb_rule.precondition,
            {"degraded_buffer_count": 0, "PIPE_V": 0, "PIPE_MTE2": 0},
        )
        is False
    )

    for kind in (
        "core_parallelism",
        "coalesced_copy",
        "ssbuf_handoff",
        "dual_copy_1c2v",
        "barrier_prune",
        "pad_alignment",
        "static_pingpong_unroll",
        "loop_fission_sfu",
        "other_kind",
    ):
        ch = AstEditChunk("c_k", kind, "main:0", "x", "y", _sample_small_dsl())
        r = distill_causal_rule(ch, rule_id=f"R_{kind.upper()}")
        assert _validate_distilled_rule(r) == (True, ""), f"Failed for {kind}"

    tracker = BayesianRuleTracker()
    funnel = GateFunnel(
        compile_runner=lambda _s: StageBResult(passed=True),
        device_runner=lambda s, st: StageCResult(
            passed=True,
            accuracy_passed=True,
            measured_latency_us=80.0 if "T.Kernel(2)" in s else 40.0,
            residual=compute_static_dynamic_residual(st.ledger, {"dur_us": 40.0}),
            telemetry={"dur_us": 40.0},
        ),
    )
    distiller = FlywheelDistiller(
        funnel=funnel,
        tracker=tracker,
        probe_runner=lambda _p: {
            "baseline_dur_us": 40.0,
            "optimized_dur_us": 10.0,
            "numerical_ok": 1.0,
        },
    )
    base_dsl = _sample_small_dsl()
    win_dsl = base_dsl.replace("with T.Kernel(2):", "with T.Kernel(8):")

    orig_distill = distill_causal_rule

    def _inject_bad_rule(chunk: AstEditChunk, *, rule_id: str, **kwargs: object) -> AscendCausalRule:
        good = orig_distill(
            chunk,
            rule_id=rule_id,
            ledger=kwargs.get("ledger"),
            probe=kwargs.get("probe"),
            probe_metrics=kwargs.get("probe_metrics"),
            scope=str(kwargs.get("scope", "cross_op")),
            origin_operator=str(kwargs.get("origin_operator", "")),
            active_symptom_family=str(kwargs.get("active_symptom_family", "")),
        )
        return replace(good, precondition="degraded_buffer_count >= 0")

    import tilefoundry_ascend.agent.flywheel.ablation_distiller as ad_mod  # noqa: PLC0415

    saved_fn = ad_mod.distill_causal_rule
    ad_mod.distill_causal_rule = _inject_bad_rule
    kb_path = tmp_path / "rejected_kb.yaml"
    try:
        report = distiller.distill_from_pair(
            baseline_dsl=base_dsl,
            winner_dsl=win_dsl,
            baseline_latency_us=80.0,
            winner_latency_us=40.0,
            factory_name="make_kernel",
            kb_output_path=kb_path,
        )
    finally:
        ad_mod.distill_causal_rule = saved_fn

    assert report.admitted_rule_ids == ()
    assert len(report.distilled_outcomes) == 1
    assert report.distilled_outcomes[0].admitted is False
    assert "tautological" in report.distilled_outcomes[0].rejection_reason
    assert len(tracker.rules) == 0
    assert load_seed_rules(kb_path) == ()
