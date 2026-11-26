"""Unit tests for Track-B 48KB active kernel view and LLM prose-refusal fast-fail."""

from __future__ import annotations

import asyncio
import inspect
import json
import re
from pathlib import Path

import pytest

from tilefoundry_ascend.agent import (
    BayesianRuleTracker,
    GateFunnel,
    MockLLMBackend,
    PortfolioRunner,
    TwoLaneBeamSearch,
)
from tilefoundry_ascend.agent.orchestrator import (
    BeamCandidate,
    StageBResult,
    StageCResult,
    compute_static_dynamic_residual,
    extract_creative_payload,
    extract_python_dsl,
    extract_structural_signature,
)
from tilefoundry_ascend.agent.orchestrator.cpp_slice_protocol import (
    apply_scoped_trackb_patch,
    build_trackb_context_view,
    select_active_trackb_kernel,
)
from tilefoundry_ascend.ascend.passes import (
    build_objective_physical_ledger,
    build_scope_waterline,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _make_large_cpp_tu(
    *,
    active_lines: int = 380,
    sibling_count: int = 6,
    sibling_lines: int = 60,
    extra_sibling_sig_pad: int = 0,
) -> tuple[str, str, str]:
    header = "#include <tl_templates/ascend/common.h>\nstruct TuMeta { int n; };\n"
    reachable_helper = (
        "__aicore__ inline float active_reachable_helper(float val) {\n"
        + "".join(f"    val = val + 0.001f * (float){i};\n" for i in range(1, 50))
        + "    return val;\n"
        "}\n"
    )
    unreachable_helper = (
        "__aicore__ inline float sibling_only_helper(float val) {\n"
        + "".join(f"    val = val * 1.001f + (float){i};\n" for i in range(1, 30))
        + "    return val;\n"
        "}\n"
    )
    active_body = [
        'extern "C" __global__ __aicore__ void active_target_kernel(__gm__ float* x, __gm__ float* y) {'
    ]
    for idx in range(1, active_lines + 1):
        active_body.append(
            f"    float acc_{idx:03d} = x[{idx}] + active_reachable_helper((float){idx});"
        )
    active_body.append("    y[0] = acc_001 + acc_%03d;" % active_lines)
    active_body.append("}")
    active_kernel_src = "\n".join(active_body) + "\n"

    siblings: list[str] = []
    for s_idx in range(sibling_count):
        sig_args = ", ".join(
            f"__gm__ float* p_{s_idx}_{a}" for a in range(1 + extra_sibling_sig_pad)
        )
        lines = [f'extern "C" __global__ __aicore__ void sibling_kernel_{s_idx}({sig_args}) {{']
        for l_idx in range(sibling_lines):
            lines.append(
                f"    float s_{l_idx:03d} = sibling_only_helper((float){l_idx}) + p_{s_idx}_0[0];"
            )
        lines.append("}")
        siblings.append("\n".join(lines) + "\n")

    full_tu = header + reachable_helper + unreachable_helper + active_kernel_src + "".join(siblings)
    return full_tu, active_kernel_src, reachable_helper


def test_build_trackb_context_view_default_48kb_preserves_25kb_active_kernel_and_helper() -> None:
    """Keep 25KB active kernel and reachable helper verbatim under default 48KB budget."""
    sig = inspect.signature(build_trackb_context_view)
    assert sig.parameters["max_bytes"].default == 49152

    full_tu, active_src, helper_src = _make_large_cpp_tu(
        active_lines=380,
        sibling_count=6,
        sibling_lines=65,
    )
    assert len(active_src.encode("utf-8")) > 24000
    assert len(full_tu.encode("utf-8")) > 45000

    view = build_trackb_context_view(
        full_tu,
        active_kernel="active_target_kernel",
        regime_tag="decode_main",
        trackb_symbol="_TEST_TRACKB_SRC",
    )
    assert len(view.encode("utf-8")) <= 49152
    assert "MIDDLE BODY OMITTED" not in view
    assert "LINES OMITTED" not in view
    assert active_src.strip() in view
    assert helper_src.strip() in view
    assert "[READ-ONLY SIBLING KERNEL: sibling_kernel_0" in view
    assert "/* [READ-ONLY HELPER] */" in view

    anchor = "    float acc_200 = x[200] + active_reachable_helper((float)200);"
    assert anchor in view
    patch = (
        "<<<<<<< SEARCH\n"
        f"{anchor}\n"
        "=======\n"
        "    float acc_200 = x[200] + active_reachable_helper((float)201);\n"
        ">>>>>>> REPLACE\n"
    )
    patched, count = apply_scoped_trackb_patch(
        full_tu,
        patch,
        active_kernel="active_target_kernel",
    )
    assert count == 1
    assert "active_reachable_helper((float)201)" in patched


def test_build_trackb_context_view_folds_siblings_before_truncating_active_kernel() -> None:
    """Fold oversized sibling/helper summaries first so active kernel stays untruncated."""
    full_tu, active_src, helper_src = _make_large_cpp_tu(
        active_lines=380,
        sibling_count=24,
        sibling_lines=10,
        extra_sibling_sig_pad=60,
    )
    assert len(active_src.encode("utf-8")) + len(helper_src.encode("utf-8")) < 40000
    assert len(full_tu.encode("utf-8")) > 50000

    view = build_trackb_context_view(
        full_tu,
        active_kernel="active_target_kernel",
        max_bytes=49152,
    )
    assert len(view.encode("utf-8")) <= 49152
    assert "MIDDLE BODY OMITTED" not in view
    assert active_src.strip() in view
    assert helper_src.strip() in view


def test_build_trackb_context_view_mla_kv_reuse_no_middle_omitted() -> None:
    """Preserve mla_kv_reuse_c1c1c2 and reachable helpers verbatim at default 48KB."""
    mod_path = REPO_ROOT / "examples" / "ascend950pr_ops" / "mla" / "kernels.py"
    mod_src = mod_path.read_text(encoding="utf-8")
    match = re.search(r'_MLA_TRACKB_SRC\s*=\s*r?"""([\s\S]*?)"""', mod_src)
    assert match is not None
    mla_cpp = match.group(1)

    active = select_active_trackb_kernel(
        mla_cpp,
        regime_tag="decode_main",
        module_source=mod_src,
    )
    assert active == "mla_kv_reuse_c1c1c2"

    view = build_trackb_context_view(
        mla_cpp,
        active_kernel=active,
        regime_tag="decode_main",
        module_source=mod_src,
        trackb_symbol="_MLA_TRACKB_SRC",
    )
    assert len(view.encode("utf-8")) <= 49152
    assert "MIDDLE BODY OMITTED" not in view
    assert "LINES OMITTED" not in view
    assert "mla_kv_reuse_c1c1c2" in view
    assert "vf_softmax_tail" in view
    assert "[READ-ONLY SIBLING KERNEL: mla_fast_combine" in view


@pytest.mark.parametrize(
    "prose_reply",
    [
        "",
        "   \n\t ",
        "This task cannot be completed because the kernel is missing.",
        "I cannot complete this task without additional context.\n### Gaps Found\n- missing",
        "### Gaps Found\n- item 1\n- item 2",
        "hello",
        json.dumps({"response": "### Gaps Found\n- Cannot complete task"}),
        json.dumps({"structured_output": {"dsl_source": "I cannot complete this task."}}),
        "```python\nI cannot complete this task.\n```",
    ],
)
def test_extract_python_dsl_fast_fails_on_prose_refusals(prose_reply: str) -> None:
    """Raise llm_prose_refusal_no_code when response lacks executable DSL or patch hunks."""
    with pytest.raises(ValueError, match="llm_prose_refusal_no_code"):
        extract_python_dsl(prose_reply)


@pytest.mark.parametrize(
    ("valid_reply", "expected_substr"),
    [
        ("def foo():\n    return 1\n", "def foo():"),
        ("x = 1", "x = 1"),
        ("import tilelang\n", "import tilelang"),
        ("def broken(:\n    pass", "def broken(:"),
        (json.dumps({"dsl_source": "def foo():\n    pass\n"}), "def foo():"),
        (
            json.dumps({"structured_output": {"dsl_source": "```python\nx = 2\n```"}}),
            "x = 2",
        ),
        ("```python\ndef bar():\n    return 3\n```", "def bar():"),
        ("```cpp\n__global__ void k() {}\n```", "__global__ void k()"),
        (
            "<<<<<<< SEARCH\nint a = 1;\n=======\nint a = 2;\n>>>>>>> REPLACE\n",
            "<<<<<<< SEARCH",
        ),
        (
            "#include <tl_templates/ascend/common.h>\n__global__ void k() {}\n",
            "__global__ void k()",
        ),
        ("__aicore__ inline void helper() {}\n", "__aicore__"),
        ("constexpr int K_TILE = 64;\n", "constexpr int K_TILE"),
    ],
)
def test_extract_python_dsl_accepts_code_patches_and_cce(
    valid_reply: str,
    expected_substr: str,
) -> None:
    """Keep valid Python, broken Python def syntax, SEARCH/REPLACE, and raw CCE accepted."""
    extracted = extract_python_dsl(valid_reply)
    assert expected_substr in extracted


def test_extract_creative_payload_and_portfolio_fast_fail_skip_syntax_repair() -> None:
    """Propagate prose refusal as llm_parse failure without invoking syntax_repair."""
    refusal_json = json.dumps(
        {
            "physical_hypothesis": "none",
            "target_local_metric": "latency_us",
            "dsl_source": "I cannot complete this task.\n### Gaps Found\n- item 1",
        }
    )
    with pytest.raises(ValueError, match="llm_prose_refusal_no_code"):
        extract_creative_payload(refusal_json)

    seed_dsl = "def make_kernel():\n    return 1\n"
    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=16,
        pipe_cycles={"AIV0.PIPE_V": 64},
        waterlines={
            "UB": build_scope_waterline(
                scope="UB",
                capacity_bytes=192 * 1024,
                explicit_buffers={"ub": 2048},
                implicit_buffers={},
            )
        },
        gm_roundtrip_bytes=1024,
    )
    parent = BeamCandidate(
        candidate_id="seed_0",
        round_idx=0,
        worker_kind="seed",
        rule_id=None,
        dsl_source=seed_dsl,
        track="unconstrained",
        ledger=ledger,
        residual=compute_static_dynamic_residual(ledger, {"dur_us": 10.0}),
        latency_us=10.0,
        accuracy_passed=True,
        structural_signature=extract_structural_signature(seed_dsl),
    )
    prose_text = "This task cannot be completed.\n### Gaps Found\n- missing context"
    backend = MockLLMBackend(
        responses={
            "exploit_rule": [prose_text, prose_text],
            "explore_creative": [refusal_json],
            "syntax_repair": [seed_dsl],
        }
    )
    funnel = GateFunnel(
        compile_runner=lambda _dsl: StageBResult(passed=True),
        device_runner=lambda _dsl, st_a: StageCResult(
            passed=True,
            accuracy_passed=True,
            measured_latency_us=9.0,
            residual=compute_static_dynamic_residual(st_a.ledger, {"dur_us": 9.0}),
            telemetry={"dur_us": 9.0},
        ),
    )
    runner = PortfolioRunner(
        backend=backend,
        funnel=funnel,
        tracker=BayesianRuleTracker.from_seed_yaml(),
        beam=TwoLaneBeamSearch(),
        k_rule_workers=1,
        max_syntax_repairs=2,
    )
    report = asyncio.run(runner.run_round(parent, round_idx=1, factory_name="make_kernel"))
    assert report.evaluated_candidates == []
    assert any(stage == "llm_parse" and "llm_prose_refusal_no_code" in err for _, stage, err in report.failed_candidates)
    assert [req.role for req in backend.calls if req.role == "syntax_repair"] == []
