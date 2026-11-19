"""Unit tests for P0 L3 operator scoping and P1 Track-B AscendC C++20 hot-path wiring."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import examples.ascend950pr_ops.grouped_matmul.kernels as gmm_kernels
import examples.ascend950pr_ops.mla.kernels as mla_kernels
import examples.ascend950pr_ops.softmax.kernels as softmax_kernels
import tests.ascend.benchmarks.test_npu_acceptance as npu_acceptance
from tilefoundry_ascend.agent.campaign.anchor_specs import (
    CannbenchOpsRegistry,
    RegimeTargetSpec,
    classify_regime_runtime_backend,
    discover_module_trackb_symbols,
)
from tilefoundry_ascend.agent.campaign.campaign_runner import (
    CampaignConfig,
    PerfTargetCampaignRunner,
    _detect_trackb_dead_code_edit,
    _extract_trackb_cpp_source,
    extract_regime_snippet,
    patch_regime_into_module,
)
from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (
    AnchorEvalMetrics,
    CandidateSurrogateResult,
)
from tilefoundry_ascend.agent.kb.schema import load_seed_rules
from tilefoundry_ascend.agent.orchestrator.gate_funnel import StageBResult
from tilefoundry_ascend.agent.orchestrator.residual_beam import compute_static_dynamic_residual
from tilefoundry_ascend.ascend.passes.physical_ledger import build_objective_physical_ledger

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_p0_npu_acceptance_scoping_and_tuned_only_guard() -> None:
    """Verify P0 normalizes levelX/<op>, isolates non-AR/MLA targets, and uses tuned mode when scoped."""
    assert npu_acceptance._normalize_target_op_short("") == ""
    assert npu_acceptance._normalize_target_op_short("level4/mla") == "mla"
    assert npu_acceptance._normalize_target_op_short("level2/softmax") == "softmax"

    # Unscoped: run legacy AR + MLA + dynamic cache tests, skip generic CANNBench hook
    for kind, expected in (
        ("attention_residual", True),
        ("mla", True),
        ("dynamic_cache", True),
        ("generic_cannbench", False),
    ):
        assert npu_acceptance._should_run_acceptance_test(kind, "") is expected

    # Scoped to AR (short or level-qualified)
    for target in ("attention_residual", "level4/attention_residual"):
        assert npu_acceptance._should_run_acceptance_test("attention_residual", target) is True
        assert npu_acceptance._should_run_acceptance_test("dynamic_cache", target) is True
        assert npu_acceptance._should_run_acceptance_test("mla", target) is False
        assert npu_acceptance._should_run_acceptance_test("generic_cannbench", target) is False

    # Scoped to MLA (short or level-qualified)
    for target in ("mla", "level4/mla"):
        assert npu_acceptance._should_run_acceptance_test("mla", target) is True
        assert npu_acceptance._should_run_acceptance_test("dynamic_cache", target) is True
        assert npu_acceptance._should_run_acceptance_test("attention_residual", target) is False
        assert npu_acceptance._should_run_acceptance_test("generic_cannbench", target) is False

    # Scoped to generic operator (e.g., level2/softmax, level3/grouped_matmul)
    for target in ("level2/softmax", "level3/grouped_matmul", "softmax"):
        assert npu_acceptance._should_run_acceptance_test("generic_cannbench", target) is True
        assert npu_acceptance._should_run_acceptance_test("attention_residual", target) is False
        assert npu_acceptance._should_run_acceptance_test("mla", target) is False
        assert npu_acceptance._should_run_acceptance_test("dynamic_cache", target) is False

    src = (REPO_ROOT / "tests" / "ascend" / "benchmarks" / "test_npu_acceptance.py").read_text(
        encoding="utf-8"
    )
    assert 'modes = ("tuned",) if _L3_TARGET_OP_SHORT else ("tuned", "baseline")' in src
    assert 'if _L3_TARGET_OP_SHORT == "mla":' in src
    assert "def test_target_operator_full_cannbench_contract()" in src


def test_p1a_trackb_discovery_classification_and_ast_splicing() -> None:
    """Verify AST Track-B symbol discovery, softmax D<=16 routing, roundtrip, and constant patching."""
    expected_symbols = {
        "softmax": ("_SOFTMAX_TRACKB_SRC", "_select_trackb_softmax_plan"),
        "grouped_matmul": ("_GMM_TRACKB_SRC",),
        "mla": ("_MLA_TRACKB_SRC",),
        "q4_0_dequant_matmul": ("_Q4_0_ASC_SRC",),
        "kimi_delta_attention": ("_TRACK_B_ASC_SRC",),
        "mhc_sinkhorn": ("_TRACK_B_CODE",),
        "apply_rotary_pos_emb": ("_ROPE_ASC_SRC",),
        "rms_norm": (),
    }
    for op_name, expected in expected_symbols.items():
        mod_text = (REPO_ROOT / "examples" / "ascend950pr_ops" / op_name / "kernels.py").read_text(
            encoding="utf-8"
        )
        assert discover_module_trackb_symbols(mod_text) == expected

    # Backend classification including softmax inner<=1 & D<=16 branch
    assert classify_regime_runtime_backend(
        "softmax",
        "extreme_aspect",
        {"d_dim": 16, "inner": 1},
        ("_SOFTMAX_TRACKB_SRC", "_select_trackb_softmax_plan"),
    ) == "tilelang_dsl"
    assert classify_regime_runtime_backend(
        "softmax",
        "inner_axis",
        {"d_dim": 16, "inner": 8},
        ("_SOFTMAX_TRACKB_SRC", "_select_trackb_softmax_plan"),
    ) == "ascendc_trackb"
    assert classify_regime_runtime_backend(
        "softmax",
        "large_throughput",
        {"d_dim": 128, "inner": 1},
        ("_SOFTMAX_TRACKB_SRC", "_select_trackb_softmax_plan"),
    ) == "ascendc_trackb"
    assert classify_regime_runtime_backend("mla", "split_kv_decode", {"S": 1}, ("_MLA_TRACKB_SRC",)) == "hybrid_trackb"
    assert classify_regime_runtime_backend("grouped_matmul", "mid_balanced", {"M": 64}, ("_GMM_TRACKB_SRC",)) == "ascendc_trackb"
    assert classify_regime_runtime_backend("rms_norm", "mid_balanced", {"D": 128}, ()) == "tilelang_dsl"

    reg = CannbenchOpsRegistry.default()
    gmm_specs = reg.get_regime_specs("grouped_matmul")
    assert gmm_specs and all(s.runtime_backend == "ascendc_trackb" for s in gmm_specs)
    assert all("_GMM_TRACKB_SRC" in s.editable_symbols for s in gmm_specs)

    rebuilt = RegimeTargetSpec.from_dict(gmm_specs[0].to_dict())
    assert rebuilt == gmm_specs[0]
    assert RegimeTargetSpec.from_dict(
        {"regime_tag": "legacy", "builder_candidates": ["_build_fn"], "sample_builder_kwargs": {}}
    ).runtime_backend == "tilelang_dsl"

    # Full extract + partial Track-B constant splice roundtrip on grouped_matmul
    gmm_src = (REPO_ROOT / "examples" / "ascend950pr_ops" / "grouped_matmul" / "kernels.py").read_text(
        encoding="utf-8"
    )
    primary, snippet = extract_regime_snippet(gmm_src, gmm_specs[0])
    assert primary == "_build_grouped_matmul_prim_func"
    assert "_GMM_TRACKB_SRC" in snippet
    updated_tb = '_GMM_TRACKB_SRC = r"""\n' + gmm_kernels._GMM_TRACKB_SRC + '\n// trackb tuned marker\n"""\n'
    patched = patch_regime_into_module(gmm_src, gmm_specs[0], updated_tb)
    sym_name, extracted_cpp = _extract_trackb_cpp_source(patched, gmm_specs[0]) or ("", "")
    assert sym_name == "_GMM_TRACKB_SRC"
    assert "// trackb tuned marker" in extracted_cpp
    assert "def _build_grouped_matmul_prim_func(" in patched


class _NpuTensorProxy:
    """Minimal CPU-backed tensor proxy with is_npu=True for unit-testing mla dispatch."""

    def __init__(self, tensor: torch.Tensor) -> None:
        self._t = tensor
        self.is_npu = True

    @property
    def shape(self):
        return self._t.shape

    @property
    def dtype(self):
        return self._t.dtype

    @property
    def device(self):
        return self._t.device

    def contiguous(self) -> _NpuTensorProxy:
        return _NpuTensorProxy(self._t.contiguous())

    def view(self, *shape) -> _NpuTensorProxy:
        return _NpuTensorProxy(self._t.view(*shape))

    def transpose(self, dim0: int, dim1: int) -> _NpuTensorProxy:
        return _NpuTensorProxy(self._t.transpose(dim0, dim1))

    def reshape(self, *shape) -> _NpuTensorProxy:
        return _NpuTensorProxy(self._t.reshape(*shape))


def test_p1b_mla_dispatch_honors_use_trackb_and_trackb_bm(monkeypatch) -> None:
    """Verify mla() honors optional cfg['use_trackb'], cfg['trackb_bm'], and merges builder_kwargs."""
    mla_kernels._DISPATCH_CACHE.clear()
    captured: list[dict[str, object]] = []

    def _fake_get_kernel(*args, **kwargs):
        captured.append({"args": args, "kwargs": dict(kwargs)})
        return lambda qn, *_rest: torch.zeros_like(qn._t)

    monkeypatch.setattr(mla_kernels, "_get_kernel", _fake_get_kernel)
    monkeypatch.setattr(
        mla_kernels,
        "_select_mla_decode_config",
        lambda **_kw: {
            "bm": 128,
            "aligned_group_q": True,
            "num_kv_splits": 4,
            "use_trackb": True,
            "trackb_bm": 32,
            "builder_kwargs": {"custom_const": 7},
        },
    )

    qn = _NpuTensorProxy(torch.zeros((2, 2, 128, 512), dtype=torch.float16))
    qr = _NpuTensorProxy(torch.zeros((2, 2, 128, 64), dtype=torch.float16))
    kn = _NpuTensorProxy(torch.zeros((2, 1024, 1, 512), dtype=torch.float16))
    kr = _NpuTensorProxy(torch.zeros((2, 1024, 1, 64), dtype=torch.float16))
    v = _NpuTensorProxy(torch.zeros((2, 1024, 1, 512), dtype=torch.float16))

    out = mla_kernels.mla(qn, qr, kn, kr, v)
    assert out.shape == (2, 2, 128, 512)
    assert len(captured) == 1
    assert captured[0]["args"][7] == 32  # trackb_bm override applied (default would be 64)
    assert captured[0]["kwargs"]["extra_builder_kwargs"] == {"custom_const": 7, "_trackb": True}

    # Verify explicit use_trackb=False routes back to TileLang path cleanly
    mla_kernels._DISPATCH_CACHE.clear()
    captured.clear()
    monkeypatch.setattr(
        mla_kernels,
        "_select_mla_decode_config",
        lambda **_kw: {
            "bm": 64,
            "aligned_group_q": True,
            "num_kv_splits": 1,
            "use_trackb": False,
        },
    )
    mla_kernels.mla(qn, qr, kn, kr, v)
    assert len(captured) == 1
    assert captured[0]["args"][7] == 64
    assert captured[0]["kwargs"]["extra_builder_kwargs"] is None


def test_p1c_dead_code_guard_l0_l1_and_prompt_routing(monkeypatch) -> None:
    """Verify dead-code guard, L0 AscendC validation, prompt tiering, and L1 Track-B compile."""
    spec = RegimeTargetSpec(
        regime_tag="large_throughput",
        builder_candidates=("_build_softmax_prim_func",),
        sample_builder_kwargs={"D": 256},
        config_selector_fn="_select_softmax_config",
        editable_symbols=(
            "_select_softmax_config",
            "_build_softmax_prim_func",
            "_SOFTMAX_TRACKB_SRC",
        ),
        runtime_backend="ascendc_trackb",
        trackb_symbols=("_SOFTMAX_TRACKB_SRC",),
    )

    parent_src = (
        '_SOFTMAX_TRACKB_SRC = r"""\n'
        '#include <tl_templates/ascend/common.h>\n'
        '__global__ __vector__ void k() {}\n'
        '"""\n\n'
        "def _select_softmax_config(**_kw):\n"
        "    return {'bm': 64}\n\n"
        "def _build_softmax_prim_func():\n"
        "    return None\n"
    )
    dead_cand = parent_src.replace("return None", "return 123")
    err = _detect_trackb_dead_code_edit(parent_src, dead_cand, spec)
    assert err is not None and "DEAD_CODE_TRACKB_EDIT" in err

    live_cpp_cand = parent_src.replace("void k() {}", "void k() { int x = 1; }")
    assert _detect_trackb_dead_code_edit(parent_src, live_cpp_cand, spec) is None

    live_sel_cand = parent_src.replace("return {'bm': 64}", "return {'bm': 128}")
    assert _detect_trackb_dead_code_edit(parent_src, live_sel_cand, spec) is None

    tilelang_spec = RegimeTargetSpec(
        regime_tag="extreme_aspect",
        builder_candidates=("_build_softmax_prim_func",),
        sample_builder_kwargs={},
        runtime_backend="tilelang_dsl",
        trackb_symbols=(),
    )
    assert _detect_trackb_dead_code_edit(parent_src, dead_cand, tilelang_spec) is None

    # Construct lightweight runner instance to test _default_l0_checker, _run_l1, and prompts
    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(operator_name="level2/softmax", max_rounds=1),
        target_module_path=REPO_ROOT / "examples" / "ascend950pr_ops" / "softmax" / "kernels.py",
    )

    # Valid Track-B L0 check
    full_sm_src = (REPO_ROOT / "examples" / "ascend950pr_ops" / "softmax" / "kernels.py").read_text(
        encoding="utf-8"
    )
    real_sm_spec = next(
        s for s in CannbenchOpsRegistry.default().get_regime_specs("softmax") if s.trackb_symbols
    )
    sa_ok = runner._default_l0_checker(full_sm_src, real_sm_spec)
    assert sa_ok.passed is True
    assert sa_ok.backend_kind == "ascendc"

    # Illegal L1->GM DMA in Track-B string must fail L0
    illegal_src = parent_src.replace("void k() {}", "void k() { /* L1->GM */ }")
    sa_bad = runner._default_l0_checker(illegal_src, spec)
    assert sa_bad.passed is False
    assert "L1->GM" in sa_bad.error_message

    # Prompt tiering + active hot-path section
    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=None,
        pipe_cycles={"AIV0.PIPE_V": 100},
        waterlines={},
    )
    surrogate_res = CandidateSurrogateResult(
        candidate_id="base",
        operator_name="level2/softmax",
        all_passed=True,
        avg_speedup=1.0,
        min_case_speedup=1.0,
        avg_roofline_eta=0.2,
        surrogate_score=50.0,
        anchors=(
            AnchorEvalMetrics(
                case_num=1,
                regime_tag="large_throughput",
                accuracy_passed=True,
                cand_us=10.0,
                ref_perf_us=10.0,
                t_hw_us=2.0,
                speedup=1.0,
                roofline_eta=0.2,
                eq3_score=50.0,
            ),
        ),
    )
    rule = load_seed_rules()[0]
    exp_req = runner._build_exploit_regime_request(
        snippet_source=parent_src,
        func_name="_build_softmax_prim_func",
        regime_spec=spec,
        rule=rule,
        best_result=surrogate_res,
        parent_ledger=ledger,
        parent_residual=compute_static_dynamic_residual(ledger, {"dur_us": 10.0}),
    )
    cre_req = runner._build_creative_regime_request(
        snippet_source=parent_src,
        func_name="_build_softmax_prim_func",
        regime_spec=spec,
        best_result=surrogate_res,
        effort="high",
        parent_ledger=ledger,
        parent_residual=compute_static_dynamic_residual(ledger, {"dur_us": 10.0}),
    )
    for req in (exp_req, cre_req):
        assert "### LAYER 2: TARGET DIALECT API SURFACE (tier=ascendc_template)" in req.user_prompt
        assert "### ACTIVE RUNTIME HOT PATH" in req.user_prompt
        assert "_SOFTMAX_TRACKB_SRC" in req.user_prompt

    # L1 Track-B dispatch (mocked compile)
    seen_cpp: list[str] = []

    def _fake_compile(cpp_source: str, **_kwargs) -> StageBResult:
        seen_cpp.append(cpp_source)
        return StageBResult(passed=True, elapsed_sec=0.05)

    monkeypatch.setattr(
        "tilefoundry_ascend.agent.campaign.campaign_runner.run_stage_b_ascendc_compile",
        _fake_compile,
    )
    sb_res = runner._run_l1(live_cpp_cand, spec)
    assert sb_res.passed is True
    assert len(seen_cpp) == 1
    assert "int x = 1;" in seen_cpp[0]

    # Hybrid Track-B L0 check fails when AscendC source violates redline even if TileLang passes
    mla_specs = CannbenchOpsRegistry.default().get_regime_specs("mla")
    mla_spec = mla_specs[0]
    assert mla_spec.runtime_backend == "hybrid_trackb"
    assert "_MLA_TRACKB_SRC" in mla_spec.trackb_symbols

    monkeypatch.setattr(
        "tilefoundry_ascend.agent.campaign.campaign_runner.run_stage_a_static",
        lambda *_a, **_kw: SimpleNamespace(
            passed=True,
            prim_func=None,
            ledger=ledger,
            features=None,
            route=None,
            error_message="",
            elapsed_ms=1.0,
        ),
    )
    mla_src = (REPO_ROOT / "examples" / "ascend950pr_ops" / "mla" / "kernels.py").read_text(
        encoding="utf-8"
    )
    hybrid_runner = PerfTargetCampaignRunner(
        config=CampaignConfig(operator_name="level4/mla", max_rounds=1),
        target_module_path=REPO_ROOT / "examples" / "ascend950pr_ops" / "mla" / "kernels.py",
    )
    assert hybrid_runner._default_l0_checker(mla_src, mla_spec).passed is True
    bad_mla_src = mla_src.replace(
        "__global__ __mix__(1, 2) void mla_kv_reuse_c1c1c2(",
        "// GM->L0A forbidden\n__global__ __mix__(1, 2) void mla_kv_reuse_c1c1c2(",
    )
    bad_mla_sa = hybrid_runner._default_l0_checker(bad_mla_src, mla_spec)
    assert bad_mla_sa.passed is False
    assert "trackb_ascendc(_MLA_TRACKB_SRC)" in bad_mla_sa.error_message

