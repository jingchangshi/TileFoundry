"""Unit tests for lossless AST closure views, Mode A/B patching, and zero-tool LLM fast-fail."""

from __future__ import annotations

import asyncio
import inspect
import json
import sys
import time
from pathlib import Path

import pytest

from tilefoundry_ascend.agent.orchestrator import cpp_slice_protocol
from tilefoundry_ascend.agent.orchestrator.cpp_slice_protocol import (
    TRACKB_REPLY_CONTRACT_PROMPT,
    AmbiguousPatchAnchorError,
    NoopIdenticalPatchError,
    PatchAnchorMissError,
    apply_scoped_trackb_patch,
    build_trackb_context_view,
    extract_scoped_trackb_kernel_view,
    format_trackb_reply_contract,
    render_active_symbol_closure_view,
    render_active_symbol_closure_with_line_map,
    splice_cpp_units_by_symbol,
    splice_cpp_units_into_tu,
)
from tilefoundry_ascend.agent.orchestrator.llm_client import (
    _PROSE_REFUSAL_ERROR,
    AgyCliLLMBackend,
    GenerationRequest,
    LLMGenerationError,
    LLMTimeoutError,
    ProseRefusalError,
    StreamWatchdogConfig,
    extract_python_dsl,
)

_FAKE_TOOL_AGY_SCRIPT = """
import json
import os
import sys
import time
from pathlib import Path


def _emit(payload):
    print(json.dumps(payload), flush=True)


def main():
    prompt = ""
    for tok in sys.argv[1:]:
        if tok.startswith("--print="):
            prompt = tok.split("=", 1)[1]
    state_dir = Path(os.environ["TF_FAKE_TOOL_STATE_DIR"])
    scenario = os.environ["TF_FAKE_TOOL_SCENARIO"]
    counter = state_dir / "attempts.txt"
    attempt = int(counter.read_text(encoding="utf-8")) + 1 if counter.exists() else 1
    counter.write_text(str(attempt), encoding="utf-8")
    (state_dir / f"prompt_{attempt}.txt").write_text(prompt, encoding="utf-8")

    if scenario == "startup_stall":
        time.sleep(30.0)
        return 0

    _emit({"event": "init", "conversation_id": f"conv_{attempt}"})

    if scenario == "tool_once_then_ok":
        if attempt == 1:
            _emit(
                {
                    "event": "step_update",
                    "step_update": {
                        "step_index": 1,
                        "step_type": "tool",
                        "tool_name": "view_file",
                        "state": "ACTIVE",
                    },
                }
            )
            time.sleep(30.0)
            return 0
        _emit(
            {
                "event": "result",
                "result": {"status": "SUCCESS", "response": "def fast_ok():\\n    return 7"},
            }
        )
        return 0

    if scenario == "tool_always":
        _emit(
            {
                "event": "step_update",
                "step_update": {
                    "step_index": 1,
                    "step_type": "tool",
                    "tool_name": "run_command",
                    "state": "ACTIVE",
                },
            }
        )
        time.sleep(30.0)
        return 0

    if scenario == "always_503":
        log_path = ""
        for tok in sys.argv[1:]:
            if tok.startswith("--log-file="):
                log_path = tok.split("=", 1)[1]
        lp = Path(log_path)
        lp.parent.mkdir(parents=True, exist_ok=True)
        lp.write_text(
            "E1117 10:00:00 client.go:90] UNAVAILABLE (code 503): No capacity available\\n"
            "E1117 10:00:00 client.go:91] UNAVAILABLE (code 503): No capacity available\\n",
            encoding="utf-8",
        )
        time.sleep(30.0)
        return 0

    raise ValueError(scenario)


if __name__ == "__main__":
    raise SystemExit(main())
"""


def _write_fake_tool_agy(tmp_path: Path) -> tuple[Path, Path]:
    bin_path = tmp_path / "fake_agy_tool"
    bin_path.write_text(f"#!{sys.executable}\n{_FAKE_TOOL_AGY_SCRIPT.lstrip()}", encoding="utf-8")
    bin_path.chmod(0o755)
    state_dir = tmp_path / "state"
    state_dir.mkdir(parents=True, exist_ok=True)
    return bin_path, state_dir


def test_lossless_active_closure_line_map_and_readonly_sibling_declaration() -> None:
    """Render 100% lossless active closure with [L0001] anchors and single-line sibling decls."""
    header = [
        "#include <tl_templates/ascend/common.h>",
        "struct ActiveParams { int count; };",
        "__simd_vf__ inline float active_leaf_helper(float v) {",
        "    return v * 2.0f + 1.0f;",
        "}",
        "__simd_vf__ inline float active_mid_helper(float v) {",
        "    return active_leaf_helper(v) + 0.5f;",
        "}",
        "__simd_vf__ inline float sibling_only_helper(float v) {",
        "    return v - 3.0f;",
        "}",
    ]
    active_lines = ['extern "C" __global__ __vector__ void kernel_active(__gm__ float* x) {']
    for idx in range(1, 780):
        active_lines.append(
            f"    float reg_{idx:03d} = active_mid_helper(x[{idx}]) + 0.25f; /* pad_{idx:03d} */"
        )
    active_lines.append("    x[0] = reg_779;")
    active_lines.append("}")
    sibling_lines = [
        'extern "C" __global__ __vector__ void kernel_sibling(__gm__ float* y) {',
        "    float sib_secret = sibling_only_helper(y[0]);",
        "    y[0] = sib_secret;",
        "}",
    ]
    cpp = "\n".join(header + active_lines + sibling_lines) + "\n"
    assert len(cpp.encode("utf-8")) > 52000

    view, line_map = render_active_symbol_closure_with_line_map(
        cpp,
        "kernel_active",
        include_line_numbers=True,
    )
    assert view == render_active_symbol_closure_view(cpp, "kernel_active")
    assert "MIDDLE BODY OMITTED" not in view
    assert "LINES OMITTED" not in view
    assert "sib_secret" not in view
    assert (
        '// [READONLY_SIBLING_DECLARATION] extern "C" __global__ __vector__ '
        "void kernel_sibling(__gm__ float* y);"
    ) in view
    assert "float reg_001 = active_mid_helper" in view
    assert "float reg_400 = active_mid_helper" in view
    assert "float reg_779 = active_mid_helper" in view
    assert "active_leaf_helper(float v)" in view

    tu_lines = cpp.splitlines()
    rendered_lines = view.splitlines()
    for out_ln, tu_ln in line_map.items():
        prefix = f"[L{tu_ln:04d}] "
        assert rendered_lines[out_ln - 1].startswith(prefix)
        assert rendered_lines[out_ln - 1][len(prefix) :] == tu_lines[tu_ln - 1]

    scoped_view = build_trackb_context_view(
        cpp,
        active_kernel="kernel_active",
        max_bytes=4096,
    )
    assert extract_scoped_trackb_kernel_view is build_trackb_context_view
    assert "float reg_400 = active_mid_helper" in scoped_view
    assert "// [READONLY_SIBLING_DECLARATION]" in scoped_view
    assert "MIDDLE BODY OMITTED" not in scoped_view


def test_mode_a_whole_symbol_replacement_and_alias() -> None:
    """Delegate bare C++ unit patches without SEARCH markers to splice_cpp_units_into_tu."""
    assert splice_cpp_units_by_symbol is splice_cpp_units_into_tu
    cpp = (
        "#include <tl_templates/ascend/common.h>\n"
        'extern "C" __global__ __aicore__ void kernel_active(__gm__ float* x) {\n'
        "    x[0] = 1.0f;\n"
        "}\n"
        'extern "C" __global__ __aicore__ void kernel_sibling(__gm__ float* y) {\n'
        "    y[0] = 9.0f;\n"
        "}\n"
    )
    mode_a_patch = (
        "```cpp\n"
        'extern "C" __global__ __aicore__ void kernel_active(__gm__ float* x) {\n'
        "    x[0] = 42.0f;\n"
        "}\n"
        "```\n"
    )
    updated, count = apply_scoped_trackb_patch(cpp, mode_a_patch, active_kernel="kernel_active")
    assert count == 1
    assert "x[0] = 42.0f;" in updated
    assert "y[0] = 9.0f;" in updated

    sibling_mode_a = (
        'extern "C" __global__ __aicore__ void kernel_sibling(__gm__ float* y) {\n'
        "    y[0] = 99.0f;\n"
        "}\n"
    )
    with pytest.raises(ValueError, match="outside the editable scope"):
        apply_scoped_trackb_patch(cpp, sibling_mode_a, active_kernel="kernel_active")


def test_mode_b_line_window_disambiguation_and_ambiguous_fault() -> None:
    """Disambiguate symmetric duplicated branches with [Lstart-Lend] and fail loudly otherwise."""
    cpp = (
        "#include <tl_templates/ascend/common.h>\n"
        'extern "C" __global__ __vector__ void sym_kernel(__gm__ float* x, int flag) {\n'
        "    if (flag > 0) {\n"
        "        float acc = x[0];\n"
        "        acc = acc * 2.0f;\n"
        "        x[0] = acc;\n"
        "    } else {\n"
        "        float acc = x[0];\n"
        "        acc = acc * 2.0f;\n"
        "        x[0] = acc;\n"
        "    }\n"
        "}\n"
    )
    branch_body = (
        "        float acc = x[0];\n"
        "        acc = acc * 2.0f;\n"
        "        x[0] = acc;"
    )
    plain_ambig = f"<<<<<<< SEARCH\n{branch_body}\n=======\n        x[0] = 5.0f;\n>>>>>>> REPLACE\n"
    with pytest.raises(AmbiguousPatchAnchorError) as exc_info:
        apply_scoped_trackb_patch(cpp, plain_ambig, active_kernel="sym_kernel")
    err = exc_info.value
    assert isinstance(err, PatchAnchorMissError)
    assert err.fault_plane == "SYNTAX_AST"
    assert err.fault_code == "PATCH_AMBIGUOUS_ANCHOR"
    assert "[L4-L6]" in str(err)
    assert "[L8-L10]" in str(err)

    win_branch_2 = (
        "<<<<<<< SEARCH [L8-L10]\n"
        f"{branch_body}\n"
        "=======\n"
        "        float acc = x[0];\n"
        "        acc = acc * 3.0f;\n"
        "        x[0] = acc;\n"
        ">>>>>>> REPLACE\n"
    )
    updated, count = apply_scoped_trackb_patch(cpp, win_branch_2, active_kernel="sym_kernel")
    assert count == 1
    assert updated.count("acc = acc * 2.0f;") == 1
    assert updated.count("acc = acc * 3.0f;") == 1

    norm_branch_1 = (
        "<<<<<<< SEARCH [L3-L7]\n"
        "float acc = x[0];\n"
        "acc = acc * 2.0f;\n"
        "x[0] = acc;\n"
        "=======\n"
        "float acc = x[0];\n"
        "acc = acc * 4.0f;\n"
        "x[0] = acc;\n"
        ">>>>>>> REPLACE\n"
    )
    updated_norm, norm_count = apply_scoped_trackb_patch(
        cpp,
        norm_branch_1,
        active_kernel="sym_kernel",
    )
    assert norm_count == 1
    assert "        acc = acc * 4.0f;" in updated_norm
    assert updated_norm.count("acc = acc * 2.0f;") == 1

    out_of_window = (
        f"<<<<<<< SEARCH [L20-L25]\n{branch_body}\n=======\n        x[0] = 0.0f;\n>>>>>>> REPLACE\n"
    )
    with pytest.raises(AmbiguousPatchAnchorError) as out_exc:
        apply_scoped_trackb_patch(cpp, out_of_window, active_kernel="sym_kernel")
    assert out_exc.value.fault_code == "PATCH_AMBIGUOUS_ANCHOR"


def test_noop_identical_patch_raises_structured_fault() -> None:
    """Raise NoopIdenticalPatchError with NOOP_IDENTICAL attributes for zero-change edits."""
    cpp = (
        "#include <tl_templates/ascend/common.h>\n"
        'extern "C" __global__ __vector__ void kernel_active(__gm__ float* x) {\n'
        "    x[0] = 1.0f;\n"
        "}\n"
    )
    ws_noop = (
        "<<<<<<< SEARCH\n"
        "    x[0] = 1.0f;\n"
        "=======\n"
        "        x[0] =   1.0f;\n"
        ">>>>>>> REPLACE\n"
    )
    with pytest.raises(NoopIdenticalPatchError) as exc_b:
        apply_scoped_trackb_patch(cpp, ws_noop, active_kernel="kernel_active")
    assert isinstance(exc_b.value, PatchAnchorMissError)
    assert exc_b.value.fault_plane == "NOOP_IDENTICAL"
    assert exc_b.value.fault_code == "NOOP_IDENTICAL"

    mode_a_noop = (
        'extern "C" __global__ __vector__ void kernel_active(__gm__ float* x) {\n'
        "  x[0] = 1.0f;\n"
        "}\n"
    )
    with pytest.raises(NoopIdenticalPatchError) as exc_a:
        apply_scoped_trackb_patch(cpp, mode_a_noop, active_kernel="kernel_active")
    assert exc_a.value.fault_plane == "NOOP_IDENTICAL"
    assert exc_a.value.fault_code == "NOOP_IDENTICAL"


def test_trackb_reply_contract_and_deleted_truncation_symbols() -> None:
    """Verify Mode A/B prompt contract and removal of legacy middle-truncation helpers."""
    assert "Mode A" in TRACKB_REPLY_CONTRACT_PROMPT
    assert "<<<<<<< SEARCH [Lstart-Lend]" in TRACKB_REPLY_CONTRACT_PROMPT
    contract = format_trackb_reply_contract("target_k")
    assert "target_k" in contract
    assert "[READONLY_SIBLING_DECLARATION]" in contract

    for deleted_attr in (
        "_truncate_kernel_body",
        "_OMITTED_SPAN_MARKER_RE",
        "_ANY_FOLDING_MARKER_RE",
        "_compact_trackb_repair_source",
        "_resplice_trackb_omitted_spans",
    ):
        assert not hasattr(cpp_slice_protocol, deleted_attr)
    src = inspect.getsource(cpp_slice_protocol)
    assert "MIDDLE BODY OMITTED" not in src
    assert "LINES OMITTED OUTSIDE FOCUS WINDOW" not in src


def test_watchdog_zero_tool_calls_fast_retry_and_hard_abort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Terminate on first tool event, allow at most one fast no-tool retry, else abort."""
    assert StreamWatchdogConfig().max_sandbox_tool_calls == 0

    bin_path, state_dir = _write_fake_tool_agy(tmp_path)
    monkeypatch.setenv("TF_FAKE_TOOL_STATE_DIR", str(state_dir))
    monkeypatch.setenv("TF_FAKE_TOOL_SCENARIO", "tool_once_then_ok")

    backend = AgyCliLLMBackend(
        bin_path=str(bin_path),
        watchdog=StreamWatchdogConfig(
            startup_timeout_sec=5.0,
            idle_stall_timeout_sec=5.0,
            active_ceiling_timeout_sec=10.0,
            max_retries=3,
            retry_backoff_sec=2.0,
        ),
    )
    t0 = time.monotonic()
    reply = asyncio.run(backend.generate(GenerationRequest("exploit_rule", "sys", "usr")))
    elapsed = time.monotonic() - t0
    assert reply == "def fast_ok():\n    return 7"
    assert elapsed < 1.5
    assert (state_dir / "attempts.txt").read_text(encoding="utf-8") == "2"
    retried_prompt = (state_dir / "prompt_2.txt").read_text(encoding="utf-8")
    assert "ZERO TOOL CALLS ALLOWED" in retried_prompt

    (state_dir / "attempts.txt").unlink(missing_ok=True)
    monkeypatch.setenv("TF_FAKE_TOOL_SCENARIO", "tool_always")
    t1 = time.monotonic()
    with pytest.raises(LLMGenerationError, match="sandbox_tool_loop_aborted") as exc_tool:
        asyncio.run(backend.generate(GenerationRequest("exploit_rule", "sys", "usr")))
    assert (time.monotonic() - t1) < 1.5
    assert (state_dir / "attempts.txt").read_text(encoding="utf-8") == "2"
    assert exc_tool.value.fault_plane == "LLM_TRANSPORT"
    assert exc_tool.value.fault_code == "SANDBOX_TOOL_LOOP_ABORTED"
    assert exc_tool.value.retriable is False


def test_structured_fault_attributes_prose_and_exhausted_503_timeout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Attach fault_plane, fault_code, and retriable on prose refusal and exhausted 503/timeout."""
    assert _PROSE_REFUSAL_ERROR.fault_plane == "SYNTAX_AST"
    assert _PROSE_REFUSAL_ERROR.fault_code == "LLM_PROSE_NO_CODE"
    assert _PROSE_REFUSAL_ERROR.retriable is False

    with pytest.raises(ProseRefusalError) as prose_exc:
        extract_python_dsl("I cannot complete this task without more context.")
    assert isinstance(prose_exc.value, ValueError)
    assert isinstance(prose_exc.value, LLMGenerationError)
    assert prose_exc.value.fault_plane == "SYNTAX_AST"
    assert prose_exc.value.fault_code == "LLM_PROSE_NO_CODE"
    assert prose_exc.value.retriable is False

    bin_path, state_dir = _write_fake_tool_agy(tmp_path)
    monkeypatch.setenv("TF_FAKE_TOOL_STATE_DIR", str(state_dir))
    monkeypatch.setenv("TF_FAKE_TOOL_SCENARIO", "always_503")
    backend_503 = AgyCliLLMBackend(
        bin_path=str(bin_path),
        watchdog=StreamWatchdogConfig(
            startup_timeout_sec=5.0,
            idle_stall_timeout_sec=5.0,
            active_ceiling_timeout_sec=10.0,
            max_503_hits=2,
            max_retries=0,
            capacity_503_max_retries=0,
        ),
    )
    with pytest.raises(LLMGenerationError, match="agy_503_no_capacity_x2") as exc_503:
        asyncio.run(backend_503.generate(GenerationRequest("exploit_rule", "sys", "usr")))
    assert exc_503.value.fault_plane == "LLM_TRANSPORT"
    assert exc_503.value.fault_code == "LLM_TIMEOUT_OR_503"
    assert exc_503.value.retriable is False

    (state_dir / "attempts.txt").unlink(missing_ok=True)
    monkeypatch.setenv("TF_FAKE_TOOL_SCENARIO", "startup_stall")
    backend_timeout = AgyCliLLMBackend(
        bin_path=str(bin_path),
        watchdog=StreamWatchdogConfig(
            startup_timeout_sec=0.15,
            idle_stall_timeout_sec=0.15,
            active_ceiling_timeout_sec=1.0,
            max_retries=0,
        ),
    )
    with pytest.raises(LLMTimeoutError, match="startup_timeout") as exc_to:
        asyncio.run(backend_timeout.generate(GenerationRequest("exploit_rule", "sys", "usr")))
    assert isinstance(exc_to.value, TimeoutError)
    assert isinstance(exc_to.value, LLMGenerationError)
    assert exc_to.value.fault_plane == "LLM_TRANSPORT"
    assert exc_to.value.fault_code == "LLM_TIMEOUT_OR_503"
    assert exc_to.value.retriable is False
