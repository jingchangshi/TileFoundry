"""Regression tests for V4.9.1 Stage-C hotfixes."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest
import tilelang
from tvm import tirx

from tilefoundry_ascend.agent.campaign.anchor_specs import RegimeTargetSpec
from tilefoundry_ascend.agent.campaign.campaign_runner import (
    CampaignConfig,
    PerfTargetCampaignRunner,
)
from tilefoundry_ascend.agent.orchestrator.gate_funnel import (
    _check_custom_kernel_redlines,
    _collect_global_kernel_names,
)
from tilefoundry_ascend.ascend.passes.hir_tir_bridge import (
    HirBlueprintPacket,
    HirRooflineSummary,
)
from tilefoundry_ascend.ascend.schedule.envelope_inspector import (
    _MAX_SAFE_SPEC_INT,
    _ensure_positive_denominators,
)
from tilefoundry_ascend.ascend.sil.pytorch_frontend import (
    PyTorchOpSemantics,
    PyTorchSemanticAnalyzer,
)
from tilefoundry_ascend.ascend.sil.schema import ConstraintRelaxationMask


def test_ensure_positive_denominators_multi_var_floordiv_no_overflow() -> None:
    """Multi-var floor-div denominator converges positive without int32 overflow."""
    q_total_rows = tirx.Var("q_total_rows", "int32")
    total_groups = tirx.Var("total_groups", "int32")
    denom = tirx.floordiv(
        tirx.floordiv(q_total_rows, total_groups * 128) * 128 + 63,
        64,
    )
    spec_var_map: dict[object, object] = {
        q_total_rows: tirx.IntImm("int32", 128),
        total_groups: tirx.IntImm("int32", 128),
    }
    recorded: list[object] = [q_total_rows, total_groups]

    _ensure_positive_denominators(
        [denom],
        spec_var_map,
        recorded,
        [q_total_rows, total_groups],
        is_handle_or_data=lambda _v: False,
        tirx=tirx,
        tvm_mod=tilelang.tvm,
    )

    analyzer = tilelang.tvm.arith.Analyzer()
    sub = tirx.stmt_functor.substitute(denom, spec_var_map)
    val = int(analyzer.simplify(sub).value)
    assert val > 0
    assert int(spec_var_map[q_total_rows].value) <= _MAX_SAFE_SPEC_INT
    assert int(spec_var_map[total_groups].value) <= _MAX_SAFE_SPEC_INT


def test_collect_global_kernel_names_expands_macro_token_pasting() -> None:
    """Macro-defined __global__ kernels expand for chevron launch validation."""
    src = """
    #define DEF_STAGE1_MIX(SUFFIX, T) \\
    extern "C" __global__ __aicore__ __mix__(1, 2) void kda_wy_stage1_mix_##SUFFIX( \\
        __gm__ T* q, int32_t B) { \\
      asc_init(); \\
    }

    DEF_STAGE1_MIX(f16_64, half)
    DEF_STAGE1_MIX(bf16_128, bfloat16_t)

    __aicore__ inline void helper_fn(__gm__ half* q) {}

    void host_launch(void* stream, __gm__ half* q, int32_t B) {
      kda_wy_stage1_mix_f16_64<<<28, nullptr, stream>>>(q, B);
      kda_wy_stage1_mix_bf16_128<<<28, nullptr, stream>>>(q, B);
    }
    """
    names = _collect_global_kernel_names(src)
    assert "kda_wy_stage1_mix_f16_64" in names
    assert "kda_wy_stage1_mix_bf16_128" in names
    assert "non_global_chevron_launch" not in " ".join(_check_custom_kernel_redlines(src))

    bad_src = src + "\nvoid bad(void* s, __gm__ half* q) { helper_fn<<<1, nullptr, s>>>(q); }\n"
    errs = _check_custom_kernel_redlines(bad_src)
    assert any("non_global_chevron_launch" in err for err in errs)


def test_pytorch_semantics_distinguishes_vector_einsum_from_gemm_einsum() -> None:
    """Vector-only einsum keeps has_matmul=True while setting has_gemm_contraction=False."""
    vec_src = """
import torch

def attention_residual(v_stack, x, w, gamma, alpha):
    v = v_stack.to(torch.float32)
    k = v * gamma.to(torch.float32)
    s = torch.einsum("nbtd,d->nbt", k, w.to(torch.float32))
    p = torch.softmax(s, dim=0)
    agg = torch.einsum("nbt,nbtd->btd", p, v)
    return (1.0 - alpha) * x.to(torch.float32) + alpha * agg
"""
    gemm_src = """
import torch

def matmul_einsum(q, k):
    return torch.einsum("bhqd,bhkd->bhqk", q, k)
"""
    analyzer = PyTorchSemanticAnalyzer()
    vec_sem = analyzer.analyze_source(vec_src, entry_name="attention_residual")
    assert vec_sem.has_matmul is True
    assert vec_sem.has_gemm_contraction is False
    assert vec_sem.base_axes.compute_engine == "aiv_only"
    assert vec_sem.base_axes.memory_ring == "gm_ub_stream"

    gemm_sem = analyzer.analyze_source(gemm_src, entry_name="matmul_einsum")
    assert gemm_sem.has_matmul is True
    assert gemm_sem.has_gemm_contraction is True

    default_sem = PyTorchOpSemantics(
        entry_name="op",
        has_matmul=True,
        has_reduction=False,
        reduction_axes=(),
        has_online_exp_max=False,
        has_indirect_index=False,
        has_subbyte_or_block_quant=False,
        has_recurrent_loop=False,
        mutates_inputs=False,
    )
    assert default_sem.has_gemm_contraction is True


def test_default_l0_checker_uses_differential_baseline_for_trackb(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Track-B Stage-A preserves incumbent forbidden tokens while rejecting growth."""
    base_src = dedent(
        """
        _TU = r'''
        #include <tl_templates/ascend/common.h>
        using namespace AscendC;
        extern "C" __global__ __aicore__ __vector__ void vec_k(
            __gm__ half* x, __gm__ half* y) {
            asc_init();
            __ubuf__ half* u = (__ubuf__ half*)0;
            asc_copy_gm2ub_align(u, x, 128);
            asc_copy_ub2gm_align(y, u, 128);
        }
        '''
        def _sel():
            return {"compile_flags": []}
        """
    )
    mod_path = tmp_path / "kernels.py"
    mod_path.write_text(base_src, encoding="utf-8")
    spec = RegimeTargetSpec(
        regime_tag="r0",
        builder_candidates=(),
        sample_builder_kwargs={},
        config_selector_fn="_sel",
        runtime_backend="ascendc_trackb",
        trackb_symbols=("_TU",),
        editable_symbols=("_sel", "_TU"),
    )
    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(operator_name="level4/gqa", max_rounds=1),
        target_module_path=mod_path,
        anchors=(),
        regime_specs=(spec,),
        regime_blueprints={
            "r0": HirBlueprintPacket(
                entry_name="gqa",
                roofline=HirRooflineSummary(
                    anchor_dims={"M": 64, "D": 64},
                    compute_flops=1000,
                    min_gm_bytes=1024,
                    arithmetic_intensity=1.0,
                    compute_floor_us=1.0,
                    memory_floor_us=1.0,
                    ideal_roofline_us=1.0,
                    dominant_bound="memory",
                ),
            )
        },
        l2_batch_evaluator=lambda _c, _a: {},
    )
    monkeypatch.setattr(runner, "_resolve_operator_semantics_for_gates", lambda: None)
    monkeypatch.setattr(
        runner,
        "_resolve_trackb_freedom_mask",
        lambda _spec: ConstraintRelaxationMask.top(),
    )

    same_res = runner._default_l0_checker(base_src, spec)
    assert same_res.passed is True

    grown_src = base_src.replace(
        "using namespace AscendC;",
        "using namespace AscendC;\n        using namespace AscendC;",
    )
    grown_res = runner._default_l0_checker(grown_src, spec)
    assert grown_res.passed is False
    assert "cce_forbidden_tokens" in grown_res.error_message
