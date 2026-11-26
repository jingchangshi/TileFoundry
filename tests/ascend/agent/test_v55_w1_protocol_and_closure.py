"""Unit tests for unified code protocol, zero-tool sandbox, closure slicing, and async lane."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import sys
import time
from pathlib import Path

import pytest

from tilefoundry_ascend.agent import (
    BayesianRuleTracker,
    BeamCandidate,
    GateFunnel,
    PortfolioRunner,
    TwoLaneBeamSearch,
)
from tilefoundry_ascend.agent.orchestrator import (
    AgyCliLLMBackend,
    GenerationRequest,
    MockLLMBackend,
    StageAResult,
    StageBResult,
    StageCResult,
    StreamWatchdogConfig,
    build_creative_request,
    build_exploit_request,
    build_syntax_repair_request,
    compute_static_dynamic_residual,
    extract_creative_payload,
    extract_python_dsl,
    llm_client,
)
from tilefoundry_ascend.agent.orchestrator.cpp_slice_protocol import (
    DEFAULT_PROTECTED_EXPORTS,
    RepairClosureSlice,
    classify_stage_failure_for_repair,
    compact_creative_source_slice,
    extract_repair_closure_slice,
    extract_trackb_launcher_preview,
    split_tracka_editable_and_readonly_tail,
)
from tilefoundry_ascend.ascend.passes import build_objective_physical_ledger

REPO_ROOT = Path(__file__).resolve().parents[3]

_SEED_DSL = """
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
            for kt in T.Pipelined(2, num_stages=2):
                T.copy(X[0:128, kt * 64 : (kt + 1) * 64], x_l1)
                T.copy(W[0:128, kt * 64 : (kt + 1) * 64], w_l1)
                T.gemm(x_l1, w_l1, acc, transpose_B=True, clear_accum=kt == 0)
            T.copy(acc, Y[0:128, 0:128])
    return main
"""

_FAKE_SANDBOX_AGY = """
import json
import os
import sys
import time
from pathlib import Path


def _emit(payload):
    print(json.dumps(payload), flush=True)


def main():
    state_dir = Path(os.environ["TF_W1_STATE_DIR"])
    scenario = os.environ["TF_W1_SCENARIO"]
    cwd = Path.cwd()
    agents_dir = cwd / ".agents"
    no_tools_md = agents_dir / "rules" / "no_tools.md"
    hooks_json = agents_dir / "hooks.json"
    snapshot = {
        "has_no_tools_md": no_tools_md.is_file(),
        "no_tools_text": no_tools_md.read_text(encoding="utf-8") if no_tools_md.is_file() else "",
        "has_hooks_json": hooks_json.is_file(),
        "hooks_data": json.loads(hooks_json.read_text(encoding="utf-8")) if hooks_json.is_file() else {},
        "argv": sys.argv[1:],
    }
    (state_dir / "snapshot.json").write_text(json.dumps(snapshot), encoding="utf-8")
    _emit({"event": "init", "conversation_id": "conv_w1"})

    if scenario == "hook_deny_once_recover":
        _emit(
            {
                "event": "step_update",
                "step_update": {
                    "step_index": 1,
                    "step_type": "tool",
                    "tool_name": "run_command",
                    "state": "ERROR",
                    "error": {"message": "Tool call denied by pre-tool hook: Tools disabled"},
                },
            }
        )
        _emit(
            {
                "event": "result",
                "result": {
                    "status": "SUCCESS",
                    "response": (
                        "# target_bottleneck: DMA_WAIT\\n"
                        "# structural_hypothesis: recovered in turn\\n"
                        "```python\\nx = 9\\n```"
                    ),
                },
            }
        )
        return 0

    if scenario == "hook_deny_twice_abort":
        for idx in (1, 2):
            _emit(
                {
                    "event": "step_update",
                    "step_update": {
                        "step_index": idx,
                        "step_type": "tool",
                        "tool_name": "view_file",
                        "state": "ERROR",
                        "error": {"message": "Denied by pre-tool hook"},
                    },
                }
            )
        time.sleep(10.0)
        return 0

    if scenario == "repair_streaming":
        for idx in range(5):
            time.sleep(0.08)
            _emit(
                {
                    "event": "step_update",
                    "step_update": {
                        "step_index": 1,
                        "step_type": "agent_response",
                        "state": "ACTIVE",
                        "text_delta": f"# tok_{idx}\\n",
                    },
                }
            )
        _emit(
            {
                "event": "result",
                "result": {"status": "SUCCESS", "response": "```python\\nfixed = 1\\n```"},
            }
        )
        return 0

    _emit(
        {
            "event": "result",
            "result": {"status": "SUCCESS", "response": "```python\\nok = 1\\n```"},
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


def _write_fake_sandbox_agy(tmp_path: Path) -> tuple[Path, Path]:
    bin_path = tmp_path / "fake_agy_w1"
    bin_path.write_text(f"#!{sys.executable}\n{_FAKE_SANDBOX_AGY.lstrip()}", encoding="utf-8")
    bin_path.chmod(0o755)
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return bin_path, state_dir


def _make_parent_candidate(dsl_source: str = _SEED_DSL) -> BeamCandidate:
    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=16,
        pipe_cycles={"AIC.PIPE_M": 64, "AIV0.PIPE_V": 32},
        waterlines={},
        gm_roundtrip_bytes=1024,
    )
    res = compute_static_dynamic_residual(ledger, {"dur_us": 40.0})
    return BeamCandidate(
        candidate_id="seed_0",
        round_idx=0,
        worker_kind="seed",
        rule_id=None,
        dsl_source=dsl_source,
        track="unconstrained",
        ledger=ledger,
        residual=res,
        latency_us=40.0,
        accuracy_passed=True,
    )


def test_unified_header_fenced_protocol_and_no_json_schema() -> None:
    """Drop CREATIVE_RESPONSE_SCHEMA/json_schema and parse header comments plus code fences."""
    assert not hasattr(llm_client, "CREATIVE_RESPONSE_SCHEMA")
    assert "CREATIVE_RESPONSE_SCHEMA" not in llm_client.__all__
    field_names = {f.name for f in dataclasses.fields(GenerationRequest)}
    assert "json_schema" not in field_names

    raw = (
        "# target_bottleneck: DMA_REDUNDANT_GM_ROUNDTRIP\n"
        "# structural_hypothesis: Fuse add and rmsnorm in UB\n"
        "```python\n"
        "@T.prim_func\n"
        "def fused():\n"
        "    pass\n"
        "```\n"
    )
    parsed = extract_creative_payload(raw)
    assert parsed["target_local_metric"] == "DMA_REDUNDANT_GM_ROUNDTRIP"
    assert parsed["physical_hypothesis"] == "Fuse add and rmsnorm in UB"
    assert "@T.prim_func" in parsed["dsl_source"]

    alt_raw = (
        "# target_local_metric: aic_fixpipe\n"
        "# physical_hypothesis: Bypass GM roundtrip\n"
        "```cpp\n"
        "__global__ void k() {}\n"
        "```\n"
    )
    alt = extract_creative_payload(alt_raw)
    assert alt["target_local_metric"] == "aic_fixpipe"
    assert alt["physical_hypothesis"] == "Bypass GM roundtrip"
    assert "__global__ void k() {}" in alt["dsl_source"]

    wrapped_json = json.dumps({"dsl_source": "def f():\n    return 1"})
    with pytest.raises(ValueError, match="llm_prose_refusal_no_code"):
        extract_python_dsl(wrapped_json)


def test_agy_sandbox_no_tools_flag_stripping_and_hook_denial_handling(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Strip plan/skip-perms flags, inject no-tool rules/hooks, and allow one hook denial."""
    assert llm_client.PURE_SYNTHESIS_ROLES == frozenset(
        {"explore_creative", "exploit_rule", "syntax_repair", "escalate_trackb"}
    )
    bin_path, state_dir = _write_fake_sandbox_agy(tmp_path)
    monkeypatch.setenv("TF_W1_STATE_DIR", str(state_dir))
    monkeypatch.setenv("TF_W1_SCENARIO", "ok")

    backend = AgyCliLLMBackend(
        bin_path=str(bin_path),
        common_args=(
            "--input-format text --sandbox --mode plan "
            "--mode=always --dangerously-skip-permissions"
        ),
        watchdog=StreamWatchdogConfig(
            startup_timeout_sec=5.0,
            idle_stall_timeout_sec=0.15,
            active_ceiling_timeout_sec=2.0,
            max_retries=0,
        ),
    )
    for role in llm_client.PURE_SYNTHESIS_ROLES:
        cmd = backend.build_command(GenerationRequest(role, "s", "u"), "P")
        assert "--dangerously-skip-permissions" not in cmd
        assert "--mode" not in cmd
        assert all(not tok.startswith("--mode=") for tok in cmd)
        assert "--json-schema" not in cmd

    audit_cmd = backend.build_command(GenerationRequest("repo_audit", "s", "u"), "P")
    assert "--dangerously-skip-permissions" in audit_cmd
    assert "--mode" in audit_cmd

    asyncio.run(backend.generate(GenerationRequest("repo_audit", "s", "u")))
    audit_snap = json.loads((state_dir / "snapshot.json").read_text(encoding="utf-8"))
    assert audit_snap["has_no_tools_md"] is False
    assert audit_snap["has_hooks_json"] is False

    monkeypatch.setenv("TF_W1_SCENARIO", "hook_deny_once_recover")
    recovered = asyncio.run(backend.generate(GenerationRequest("explore_creative", "s", "u")))
    creative_payload = extract_creative_payload(recovered)
    assert creative_payload["dsl_source"] == "x = 9"
    snap = json.loads((state_dir / "snapshot.json").read_text(encoding="utf-8"))
    assert snap["has_no_tools_md"] is True
    assert "trigger: always_on" in snap["no_tools_text"]
    assert snap["has_hooks_json"] is True
    hook_groups = snap["hooks_data"]["no-tools"]["PreToolUse"]
    assert hook_groups[0]["matcher"] == "*"
    assert "deny" in hook_groups[0]["hooks"][0]["command"]

    monkeypatch.setenv("TF_W1_SCENARIO", "hook_deny_twice_abort")
    with pytest.raises(llm_client.LLMGenerationError) as exc_info:
        asyncio.run(backend.generate(GenerationRequest("exploit_rule", "s", "u")))
    assert exc_info.value.fault_code == "SANDBOX_TOOL_LOOP_ABORTED"

    monkeypatch.setenv("TF_W1_SCENARIO", "repair_streaming")
    repaired = asyncio.run(backend.generate(GenerationRequest("syntax_repair", "s", "u")))
    assert extract_python_dsl(repaired) == "fixed = 1"


def test_repair_classifier_and_split_tracka_tail() -> None:
    """Classify deterministic capacity/noop failures and split Track-A host tail cleanly."""
    for non_repairable in (
        "NOOP_IDENTICAL",
        "UB_SRAM_OVERFLOW",
        "UB_OVERFLOW",
        "L1_SRAM_OVERFLOW",
        "L0C_SRAM_OVERFLOW",
        "VF_STACK_OVERFLOW",
    ):
        assert classify_stage_failure_for_repair(non_repairable) is False

    for repairable in (
        "STAGE_A_PARSE_OR_LOWER_ERROR",
        "TILELANG_AST_VIOLATION",
        "STAGE_B_COMPILE_FAILED",
        "",
    ):
        assert classify_stage_failure_for_repair(repairable) is True

    mod_src = (
        "import tilelang.ascend.language as T\n\n"
        "def _build_op(bm: int = 32):\n"
        "    @T.prim_func\n"
        "    def main():\n"
        "        pass\n"
        "    return main\n\n"
        "def kernel(x):\n"
        "    return x\n\n"
        "ref_kernel = kernel\n"
        "__all__ = ['kernel']\n"
    )
    editable, tail = split_tracka_editable_and_readonly_tail(mod_src, DEFAULT_PROTECTED_EXPORTS)
    assert "def _build_op" in editable
    assert "def kernel" not in editable
    assert tail.startswith("def kernel(x):")
    assert "ref_kernel = kernel" in tail

    broken_src = (
        "def _build_op():\n"
        "    if True\n"
        "        pass\n"
        "kernel = _build_op\n"
    )
    b_edit, b_tail = split_tracka_editable_and_readonly_tail(broken_src)
    assert "def _build_op" in b_edit
    assert b_tail == "kernel = _build_op"


def test_extract_repair_closure_slice_trackb_and_tracka_zero_byte_caps() -> None:
    """Extract lossless reachable call-graph closures for hybrid Track-B and Track-A."""
    mla_src = (REPO_ROOT / "examples" / "ascend950pr_ops" / "mla" / "kernels.py").read_text(
        encoding="utf-8"
    )
    mla_closure = extract_repair_closure_slice(
        mla_src,
        regime_tag="decode_main",
        backend_kind="hybrid_trackb",
        diagnostic_text="kernel.cce:120:5: error: invalid vector cast",
    )
    assert isinstance(mla_closure, RepairClosureSlice)
    assert "_build_mla_prim_func" not in mla_closure.editable_device_source
    assert "_build_mla_split_kv_decode_func" not in mla_closure.editable_device_source
    assert "_build_mla_prefill_func" not in mla_closure.editable_device_source
    assert "mla_kv_reuse_c1c1c2" in mla_closure.editable_device_source
    assert any("mla_fast_combine" in proto for proto in mla_closure.sibling_prototypes)
    assert "def _select_mla_decode_config" in mla_closure.readonly_host_abi_tail
    assert "def _render_trackb_tu" in mla_closure.readonly_host_abi_tail
    assert 'extern "C" int launch_mla_trackb' in mla_closure.readonly_host_abi_tail
    assert "def _build_mla_trackb_callable" in mla_closure.readonly_host_abi_tail

    giant_kernel_lines = [
        "#include <tl_templates/ascend/common.h>",
        "struct TilingData { int m; int block_dim; };",
        "constexpr int kBlock = 64;",
        "__simd_vf__ inline float used_helper(float x) { return x + 1.0f; }",
        "__simd_vf__ inline float unused_helper(float x) { return x - 1.0f; }",
        'extern "C" __global__ __vector__ void big_active_k(__gm__ float* x, TilingData td) {',
        *(f"    float v_{i:04d} = used_helper(x[{i}]); /* padding_{i:04d} */" for i in range(950)),
        "    x[0] = v_0949;",
        "}",
        'extern "C" __global__ __vector__ void unused_sibling_k(__gm__ float* y) {',
        "    y[0] = unused_helper(y[0]);",
        "}",
        'extern "C" int launch_big(void* stream, void* x, TilingData td) {',
        "    big_active_k<<<td.block_dim, nullptr, stream>>>((__gm__ float*)x, td);",
        "    return 0;",
        "}",
    ]
    giant_cpp = "\n".join(giant_kernel_lines)
    assert len(giant_cpp.encode("utf-8")) > 50_000
    cpp_closure = extract_repair_closure_slice(
        giant_cpp,
        active_symbols=("big_active_k",),
        backend_kind="ascendc_trackb",
        diagnostic_text="kernel.cce:10: error: test",
    )
    assert "v_0949 = used_helper" in cpp_closure.editable_device_source
    assert "struct TilingData" in cpp_closure.editable_device_source
    assert "constexpr int kBlock = 64;" in cpp_closure.editable_device_source
    assert "unused_sibling_k" not in cpp_closure.editable_device_source
    assert any("unused_sibling_k" in proto for proto in cpp_closure.sibling_prototypes)
    assert any("unused_helper" in proto for proto in cpp_closure.sibling_prototypes)
    assert 'extern "C" int launch_big' in cpp_closure.readonly_host_abi_tail

    preview = extract_trackb_launcher_preview(giant_cpp)
    assert "launch_big" in preview
    assert "truncated" not in preview

    tracka_mod = (
        "import tilelang.ascend.language as T\n\n"
        "def _used_helper(x):\n"
        "    return x + 1\n\n"
        "def _dead_helper(y):\n"
        "    return y * 2\n\n"
        "def _select_decode_config(d: int):\n"
        "    return {'bm': 32}\n\n"
        "def _build_decode_func(bm: int = 32):\n"
        "    @T.prim_func\n"
        "    def main():\n"
        "        _used_helper(bm)\n"
        "    return main\n\n"
        "def _select_prefill_config(d: int):\n"
        "    return {'bm': 128}\n\n"
        "def _build_prefill_func(bm: int = 128):\n"
        "    @T.prim_func\n"
        "    def main():\n"
        "        _dead_helper(bm)\n"
        "    return main\n\n"
        "kernel = _build_decode_func\n"
        "__all__ = ['kernel']\n"
    )
    py_closure = extract_repair_closure_slice(
        tracka_mod,
        regime_tag="decode",
        backend_kind="tilelang_dsl",
    )
    assert "def _used_helper" in py_closure.editable_device_source
    assert "def _build_decode_func" in py_closure.editable_device_source
    assert "def _select_decode_config" in py_closure.editable_device_source
    assert "def _dead_helper" not in py_closure.editable_device_source
    assert "def _build_prefill_func(bm: int=128): ..." in py_closure.sibling_prototypes
    assert "def _select_prefill_config(d: int): ..." in py_closure.sibling_prototypes
    assert "kernel = _build_decode_func" in py_closure.readonly_host_abi_tail

    assert compact_creative_source_slice(tracka_mod, max_bytes=64) == tracka_mod


def test_portfolio_prompts_skip_repair_and_speculative_creative_lane() -> None:
    """Verify unified prompts, zero parent truncation, repair skipping, and async creative lane."""
    tracka_with_tail = f"{_SEED_DSL.strip()}\n\nkernel = make_kernel\n__all__ = ['kernel']\n"
    parent = _make_parent_candidate(tracka_with_tail)
    rule = BayesianRuleTracker.from_seed_yaml().active_rules()[0]

    exploit_req = build_exploit_request(parent, rule)
    creative_req = build_creative_request(parent)
    for req in (exploit_req, creative_req):
        combined = f"{req.system_prompt}\n{req.user_prompt}"
        assert "# target_bottleneck:" in combined
        assert "# structural_hypothesis:" in combined
        assert "Return JSON matching the schema" not in combined
        assert "Never nest `def`" in combined
        assert "<read_only_host_tail>" in req.user_prompt
        assert "kernel = make_kernel" in req.user_prompt
        code_block = req.user_prompt.split("### CURRENT KERNEL SOURCE\n```python\n", 1)[1].split(
            "\n```\n", 1
        )[0]
        assert "kernel = make_kernel" not in code_block

    big_parent_lines = [
        "import tilelang.ascend.language as T",
        "def make_kernel():",
        "    @T.prim_func",
        "    def main():",
        *(f"        v_{i:04d} = {i}" for i in range(1100)),
        "        pass",
        "    return main",
    ]
    big_parent = "\n".join(big_parent_lines)
    assert len(big_parent.encode("utf-8")) > 20_000
    repair_req = build_syntax_repair_request(
        big_parent,
        "syntax error",
        parent_source=big_parent,
    )
    assert "v_1099 = 1099" in repair_req.user_prompt
    assert "parent source truncated" not in repair_req.user_prompt

    seen_backends: list[str | None] = []

    class _BackendSpyTracker(BayesianRuleTracker):
        def rank_applicable_rules(
            self,
            context,
            *,
            top_k: int | None = None,
            topology_axes: object | None = None,
            operator_name: str = "",
            runtime_backend: str | None = None,
            regime_tag: str = "",
            parent_sha: str = "",
            current_round: int | None = None,
            symptom_ranking=(),
        ):
            seen_backends.append(runtime_backend)
            return super().rank_applicable_rules(
                context,
                top_k=top_k,
                topology_axes=topology_axes,
                operator_name=operator_name,
                runtime_backend=runtime_backend,
                regime_tag=regime_tag,
                parent_sha=parent_sha,
                current_round=current_round,
                symptom_ranking=symptom_ranking,
            )

    tracker = _BackendSpyTracker.from_seed_yaml()
    skip_backend = MockLLMBackend(
        responses={
            "exploit_rule": [_SEED_DSL, _SEED_DSL],
            "explore_creative": [
                f"# target_bottleneck: x\n# structural_hypothesis: y\n```python\n{_SEED_DSL}\n```"
            ],
            "syntax_repair": [_SEED_DSL],
        }
    )
    skip_funnel = GateFunnel(
        compile_runner=lambda _src: StageBResult(
            passed=False,
            fault_plane="STAGE_B",
            fault_code="UB_SRAM_OVERFLOW",
            error_message="UB overflow",
        )
    )
    skip_runner = PortfolioRunner(
        backend=skip_backend,
        funnel=skip_funnel,
        tracker=tracker,
        beam=TwoLaneBeamSearch(),
        k_rule_workers=2,
        max_syntax_repairs=1,
    )
    skip_report = asyncio.run(
        skip_runner.run_round(
            _make_parent_candidate(),
            round_idx=1,
            factory_name="make_kernel",
            escalation_tier="ascendc_trackb",
        )
    )
    assert seen_backends == ["ascendc_trackb"]
    assert skip_report.repaired_worker_ids == []
    assert [c for c in skip_backend.calls if c.role == "syntax_repair"] == []

    class _SlowCreativeBackend:
        def __init__(self) -> None:
            self.creative_gate = asyncio.Event()

        async def generate(self, req: GenerationRequest, *, effort_override: str | None = None) -> str:
            _ = effort_override
            if req.role == "explore_creative":
                await self.creative_gate.wait()
                return (
                    "# target_bottleneck: DMA\n"
                    "# structural_hypothesis: Async speculative winner\n"
                    f"```python\n{_SEED_DSL}\n```"
                )
            return _SEED_DSL

    slow_backend = _SlowCreativeBackend()
    ok_funnel = GateFunnel(
        compile_runner=lambda _dsl: StageBResult(passed=True),
        device_runner=lambda _dsl, st_a: StageCResult(
            passed=True,
            accuracy_passed=True,
            measured_latency_us=25.0,
            residual=compute_static_dynamic_residual(st_a.ledger, {"dur_us": 25.0}),
            telemetry={"dur_us": 25.0},
        ),
    )

    async def _run_speculative_rounds() -> None:
        spec_runner = PortfolioRunner(
            backend=slow_backend,
            funnel=ok_funnel,
            tracker=BayesianRuleTracker.from_seed_yaml(),
            beam=TwoLaneBeamSearch(),
            k_rule_workers=2,
        )
        t0 = time.monotonic()
        r1 = await spec_runner.run_round(
            _make_parent_candidate(),
            round_idx=1,
            factory_name="make_kernel",
        )
        assert (time.monotonic() - t0) < 0.5
        assert len(r1.evaluated_candidates) == 2
        assert spec_runner._pending_speculative_creative is not None
        assert not spec_runner._pending_speculative_creative.done()

        slow_backend.creative_gate.set()
        await asyncio.sleep(0.01)
        assert spec_runner._pending_speculative_creative.done()

        slow_backend.creative_gate.clear()
        r2 = await spec_runner.run_round(
            _make_parent_candidate(),
            round_idx=2,
            factory_name="make_kernel",
        )
        creative_cands = [c for c in r2.evaluated_candidates if c.worker_kind == "creative_explore"]
        assert len(creative_cands) == 1
        assert creative_cands[0].candidate_id == "r1_creative"
        assert creative_cands[0].hypothesis == "Async speculative winner"

        assert spec_runner._pending_speculative_creative is not None
        assert not spec_runner._pending_speculative_creative.done()
        spec_runner.cancel_pending_speculative()
        assert spec_runner._pending_speculative_creative is None

    asyncio.run(_run_speculative_rounds())


def test_tracka_search_replace_extraction_prompt_and_creative_semaphore(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify SEARCH/REPLACE extraction, Track-A delta contract, and _creative_sem isolation."""
    reply_with_hunk = (
        "# target_bottleneck: PIPE_V_STALL\n"
        "# structural_hypothesis: Double vector tile size via minimal hunk\n"
        "```python\n"
        "<<<<<<< SEARCH [L12-L15]\n"
        "    tile = 32\n"
        "=======\n"
        "    tile = 64\n"
        ">>>>>>> REPLACE\n"
        "```\n"
    )
    payload = extract_creative_payload(reply_with_hunk)
    assert payload["target_local_metric"] == "PIPE_V_STALL"
    assert payload["physical_hypothesis"] == "Double vector tile size via minimal hunk"
    assert "<<<<<<< SEARCH [L12-L15]" in payload["dsl_source"]
    assert ">>>>>>> REPLACE" in payload["dsl_source"]
    assert extract_python_dsl(reply_with_hunk) == payload["dsl_source"]

    parent = _make_parent_candidate(
        f"{_SEED_DSL.strip()}\n\nkernel = make_kernel\n__all__ = ['kernel']\n"
    )
    rule = BayesianRuleTracker.from_seed_yaml().active_rules()[0]
    exploit_req = build_exploit_request(parent, rule)
    assert "# target_bottleneck:" in exploit_req.system_prompt
    assert "# structural_hypothesis:" in exploit_req.system_prompt
    assert "<<<<<<< SEARCH" in exploit_req.system_prompt
    assert ">>>>>>> REPLACE" in exploit_req.system_prompt
    assert "<read_only_host_tail>" in exploit_req.system_prompt

    creative_req = build_creative_request(parent)
    assert "# target_bottleneck:" in creative_req.system_prompt
    assert "# structural_hypothesis:" in creative_req.system_prompt
    assert "<<<<<<< SEARCH" in creative_req.system_prompt

    repair_req = build_syntax_repair_request(_SEED_DSL, "syntax error")
    assert "<<<<<<< SEARCH" in repair_req.system_prompt

    bin_path, state_dir = _write_fake_sandbox_agy(tmp_path)
    monkeypatch.setenv("TF_W1_STATE_DIR", str(state_dir))
    monkeypatch.setenv("TF_W1_SCENARIO", "ok")
    backend = AgyCliLLMBackend(
        bin_path=str(bin_path),
        max_concurrency=1,
        watchdog=StreamWatchdogConfig(
            startup_timeout_sec=5.0,
            idle_stall_timeout_sec=2.0,
            active_ceiling_timeout_sec=5.0,
            max_retries=0,
        ),
    )

    async def _verify_semaphore_isolation() -> None:
        creative_entered = asyncio.Event()
        release_creative = asyncio.Event()
        seen_order: list[str] = []

        async def _fake_attempt(
            req: GenerationRequest,
            _full_prompt: str,
            *,
            effort_override: str | None = None,
            model_override: str | None = None,
        ) -> str:
            _ = (effort_override, model_override)
            seen_order.append(f"start:{req.role}")
            if req.role == "explore_creative":
                creative_entered.set()
                await release_creative.wait()
            seen_order.append(f"end:{req.role}")
            return "```python\nx = 1\n```"

        monkeypatch.setattr(backend, "_run_single_attempt", _fake_attempt)
        creative_task = asyncio.create_task(
            backend.generate(GenerationRequest("explore_creative", "s", "u"))
        )
        await creative_entered.wait()
        assert backend._creative_sem.locked()
        assert not backend._sem.locked()

        exploit_res = await asyncio.wait_for(
            backend.generate(GenerationRequest("exploit_rule", "s", "u")),
            timeout=1.0,
        )
        assert "x = 1" in exploit_res
        assert "end:exploit_rule" in seen_order
        assert "end:explore_creative" not in seen_order

        release_creative.set()
        await creative_task

    asyncio.run(_verify_semaphore_isolation())

