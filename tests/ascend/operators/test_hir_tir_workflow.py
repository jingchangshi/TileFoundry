"""End-to-end CPU HIR evaluation, analysis, pipeline solving, and TIR verification."""

from __future__ import annotations

import pathlib

import torch

from tests.ascend.benchmarks.attention_residual.golden import (
    attention_residual as ar_golden,
)
from tests.ascend.benchmarks.mla.golden import mla as mla_golden
from tilefoundry.analysis import analyze, check_program
from tilefoundry.cli.source import load_authored_ir
from tilefoundry.evaluator import evaluate
from tilefoundry.inspection import as_script
from tilefoundry.schedule.candidates import candidates as report_candidates
from tilefoundry.schedule.facts import listing as facts_listing
from tilefoundry.schedule.finalize import finalize
from tilefoundry.visitor_registry.verify import verify_prim_function
from tilefoundry_ascend import AscendTarget

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
AR_DIR = REPO_ROOT / "examples" / "ascend950pr_ops" / "attention_residual"
MLA_DIR = REPO_ROOT / "examples" / "ascend950pr_ops" / "mla"


def _lower(mod):
    authored = mod.entry_function()
    checked_mod, checked_fn = check_program(mod, authored)
    analyzed = analyze(checked_mod, checked_fn, analysis="memory")
    return finalize(analyzed, authored=authored)


def test_attention_residual_model_and_scheduled_match_golden():
    model_mod = load_authored_ir(f"{AR_DIR / 'model.py'}:AttentionResidualModel")
    sched_mod = load_authored_ir(f"{AR_DIR / 'scheduled.py'}:AttentionResidualScheduled")

    n, b, s, d = 4, 1, 4, 64
    v_stack = torch.randn(n, b, s, d, dtype=torch.bfloat16)
    x = torch.randn(b, s, d, dtype=torch.bfloat16)
    w = torch.randn(d, dtype=torch.bfloat16)
    gamma = torch.rand(d, dtype=torch.bfloat16) + 0.5
    ref = ar_golden(v_stack, x, w, gamma, alpha=0.5, epsilon=1e-6)

    model_out = evaluate(model_mod.entry_function(), v_stack, x, w, gamma).data
    sched_out = evaluate(sched_mod.entry_function(), v_stack, x, w, gamma).data
    torch.testing.assert_close(model_out.float(), ref.float(), rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(sched_out.float(), ref.float(), rtol=2e-2, atol=2e-2)


def test_mla_model_and_scheduled_match_golden():
    model_mod = load_authored_ir(f"{MLA_DIR / 'model.py'}:MlaModel")
    sched_mod = load_authored_ir(f"{MLA_DIR / 'scheduled.py'}:MlaScheduled")

    b, s, s_kv, n_q, n_kv, d_nope, d_rope = 1, 16, 16, 4, 1, 64, 64
    q_nope = torch.randn(b, s, n_q, d_nope, dtype=torch.float16)
    q_rope = torch.randn(b, s, n_q, d_rope, dtype=torch.float16)
    k_nope = torch.randn(b, s_kv, n_kv, d_nope, dtype=torch.float16)
    k_rope = torch.randn(b, s_kv, n_kv, d_rope, dtype=torch.float16)
    v = k_nope.clone()
    ref = mla_golden(
        q_nope,
        q_rope,
        k_nope,
        k_rope,
        v,
        numKVHeads=n_kv,
        scaleValue=-1.0,
        inputLayout="BSND",
        is_causal=True,
    )

    model_out = evaluate(model_mod.entry_function(), q_nope, q_rope, k_nope, k_rope, v).data
    torch.testing.assert_close(model_out.float(), ref.float(), rtol=2e-2, atol=2e-2)

    q_nope_bnsd = q_nope.permute(0, 2, 1, 3).contiguous()
    q_rope_bnsd = q_rope.permute(0, 2, 1, 3).contiguous()
    k_nope_bnsd = k_nope.permute(0, 2, 1, 3).contiguous()
    k_rope_bnsd = k_rope.permute(0, 2, 1, 3).contiguous()
    v_bnsd = v.permute(0, 2, 1, 3).contiguous()
    qi = torch.arange(s).view(s, 1)
    kj = torch.arange(s_kv).view(1, s_kv)
    mask_2d = torch.where(kj > qi + (s_kv - s), torch.tensor(-1e9), torch.tensor(0.0)).to(
        torch.float32
    )
    causal_mask = mask_2d.unsqueeze(0).expand(n_q, s, s_kv).reshape(n_q * s, s_kv).contiguous()
    ref_bnsd = mla_golden(
        q_nope_bnsd,
        q_rope_bnsd,
        k_nope_bnsd,
        k_rope_bnsd,
        v_bnsd,
        numKVHeads=n_kv,
        scaleValue=-1.0,
        inputLayout="BNSD",
        is_causal=True,
    )
    sched_out = evaluate(
        sched_mod.entry_function(),
        q_nope_bnsd,
        q_rope_bnsd,
        k_nope_bnsd,
        k_rope_bnsd,
        v_bnsd,
        causal_mask,
    ).data
    torch.testing.assert_close(sched_out.float(), ref_bnsd.float(), rtol=2e-2, atol=2e-2)


def test_ascend_schedule_cli_and_finalize_workflow():
    target = AscendTarget("ascend.950pr")
    facts = facts_listing(target)
    assert any(row["id"] == "T.ascend_mmad" for row in facts["instructions"])

    for op_dir, model_cls, sched_cls in (
        (AR_DIR, "AttentionResidualModel", "AttentionResidualScheduled"),
        (MLA_DIR, "MlaModel", "MlaScheduled"),
    ):
        model_mod = load_authored_ir(f"{op_dir / 'model.py'}:{model_cls}")
        checked_mod, checked_fn = check_program(model_mod, model_mod.entry_function())
        cand_report = report_candidates(checked_mod, checked_fn)
        assert cand_report["lines"]

        sched_mod = load_authored_ir(f"{op_dir / 'scheduled.py'}:{sched_cls}")
        lowered = _lower(sched_mod)
        verify_prim_function(lowered.entry_function(), module_fns=lowered)
        pipelines = lowered.metadata.get("ascend.pipelines")
        assert isinstance(pipelines, dict) and pipelines
        tir_text = as_script(lowered)
        assert "tf.pipelined(" in tir_text

    assert (AR_DIR / "tir" / "attention_residual.py").is_file()
    assert (MLA_DIR / "tir" / "mla.py").is_file()


def test_zero_simtvf_in_scheduled_tir_and_kernels():
    """Verify both Ascend 950PR operators use SimdVF (zero SimtVF) across layers 2-3."""
    checked_paths = (
        AR_DIR / "scheduled.py",
        AR_DIR / "tir" / "attention_residual.py",
        AR_DIR / "kernels.py",
        MLA_DIR / "scheduled.py",
        MLA_DIR / "tir" / "mla.py",
        MLA_DIR / "kernels.py",
    )
    for path in checked_paths:
        text = path.read_text(encoding="utf-8")
        assert "SimtVF" not in text and "simtvf" not in text, f"Unexpected SimtVF in {path}"
        assert "SimdVF" in text or "simdvf" in text, f"Expected SimdVF in {path}"
