"""Integration tests for pure-golden mode, P2 deep-dive, online dedup, and diagnosis."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from textwrap import dedent

import pytest

from tilefoundry_ascend.agent.campaign import (
    AnchorCaseSpec,
    CampaignConfig,
    CampaignReport,
    PerfTargetCampaignRunner,
    PerfTargetSpec,
    RegimeTargetSpec,
    aggregate_surrogate_result,
    build_anchor_metrics,
    build_arg_parser,
    marathon,
)
from tilefoundry_ascend.agent.campaign.anchor_specs import CannbenchOpsRegistry
from tilefoundry_ascend.agent.campaign.marathon import MarathonConfig, run_operator_marathon
from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (
    CandidateSurrogateResult,
    L3AcceptanceResult,
)
from tilefoundry_ascend.agent.kb.bayesian_tracker import BayesianRuleTracker
from tilefoundry_ascend.agent.kb.schema import (
    SYMPTOM_FAMILY_TO_FACETS,
    AscendCausalRule,
    SymptomFamily,
)
from tilefoundry_ascend.agent.orchestrator.gate_funnel import StageAResult, StageBResult
from tilefoundry_ascend.agent.orchestrator.llm_client import GenerationRequest
from tilefoundry_ascend.agent.orchestrator.residual_beam import (
    BeamCandidate,
    StaticDynamicResidual,
    TwoLaneBeamSearch,
)
from tilefoundry_ascend.agent.router import CapabilityGate, route_execution_track
from tilefoundry_ascend.ascend.cost_model import BishengResourceSummary
from tilefoundry_ascend.ascend.cost_model.experience_lake import (
    ExperienceLake,
    compute_source_sha256,
)
from tilefoundry_ascend.ascend.passes import (
    HirBlueprintPacket,
    HirRooflineSummary,
    build_objective_physical_ledger,
)
from tilefoundry_ascend.ascend.sil.schema import ConstraintRelaxationMask, TopologyAxes


class _CapturingBackend:
    """Async mock LLM backend recording requests and yielding scripted replies."""

    def __init__(self, reply_fn: object) -> None:
        self._reply_fn = reply_fn
        self.requests: list[GenerationRequest] = []

    async def generate(self, req: GenerationRequest) -> str:
        self.requests.append(req)
        return str(self._reply_fn(req, len(self.requests) - 1))


_DUAL_MODULE_SRC = dedent(
    """
    def _build_reg_a(tile_m=16):
        def main():
            return tile_m
        return main

    def _build_reg_b(tile_m=8):
        def main():
            return tile_m
        return main

    def op_entry(x):
        return _build_reg_a(), _build_reg_b(), x
    """
)

_REG_A_IMPROVED = dedent(
    """
    def _build_reg_a(tile_m=32):
        def main():
            return tile_m
        return main
    """
)

_REG_A_STONE = dedent(
    """
    def _build_reg_a(tile_m=64):
        def main():
            return tile_m
        return main
    """
)

_REG_B_IMPROVED = dedent(
    """
    def _build_reg_b(tile_m=32):
        def main():
            return tile_m
        return main
    """
)


def _dual_regimes() -> tuple[RegimeTargetSpec, RegimeTargetSpec]:
    return (
        RegimeTargetSpec(
            regime_tag="reg_a",
            description="Regime A",
            editable_symbols=("_build_reg_a",),
            builder_candidates=("_build_reg_a",),
            sample_builder_kwargs={},
            representative_hir_dims={"M": 128, "N": 128},
            representative_anchor_dims={"M": 128, "N": 128},
        ),
        RegimeTargetSpec(
            regime_tag="reg_b",
            description="Regime B",
            editable_symbols=("_build_reg_b",),
            builder_candidates=("_build_reg_b",),
            sample_builder_kwargs={},
            representative_hir_dims={"M": 64, "N": 64},
            representative_anchor_dims={"M": 64, "N": 64},
        ),
    )


def _dual_anchors() -> tuple[AnchorCaseSpec, AnchorCaseSpec]:
    return (
        AnchorCaseSpec("synthetic/op", 1, "reg_a", 100.0, 20.0),
        AnchorCaseSpec("synthetic/op", 2, "reg_b", 80.0, 16.0),
    )


def _perf_target(eta: float = 0.85) -> PerfTargetSpec:
    return PerfTargetSpec(
        operator_name="synthetic/op",
        target_avg_speedup=2.5,
        target_min_case_speedup=2.0,
        target_roofline_eta=eta,
        target_surrogate_score=90.0,
    )


def _make_stage_a(
    *,
    gm_bytes: int = 262144,
    ub_overflow: int = 0,
    steady_ii: int = 10,
    pipe_cycles: Mapping[str, int] | None = None,
) -> StageAResult:
    waterlines = {}
    if ub_overflow > 0:
        from tilefoundry_ascend.ascend.passes.physical_ledger import ScopeWaterline

        waterlines["UB"] = ScopeWaterline(
            scope="UB",
            peak_bytes=262144 + ub_overflow,
            limit_bytes=262144,
            overflow_bytes=ub_overflow,
        )
    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=steady_ii,
        pipe_cycles=dict(pipe_cycles or {"AIV0.PIPE_V": 120, "AIV0.PIPE_MTE2": 200}),
        gm_roundtrip_bytes=gm_bytes,
        min_gm_dma_bytes=gm_bytes,
        waterlines=waterlines,
    )
    route = route_execution_track(None, ledger=ledger)
    feats = route.features
    return StageAResult(
        passed=True,
        prim_func=None,
        ledger=ledger,
        features=feats,
        route=route,
    )


def _make_bisheng_summary(
    *,
    spill_bytes: int = 0,
    vloopv2_loops: int = 1,
    scalar_regs: int = 4,
) -> BishengResourceSummary:
    return BishengResourceSummary(
        spill_bytes=spill_bytes,
        vloopv2_loops=vloopv2_loops,
        scalar_regs=scalar_regs,
        vec_regs=16,
    )


def _make_blueprint(ideal_us: float = 16.0) -> HirBlueprintPacket:
    return HirBlueprintPacket(
        entry_name="op_entry",
        roofline=HirRooflineSummary(
            anchor_dims={"M": 128, "N": 128},
            compute_flops=1024 * 1024,
            min_gm_bytes=262144,
            arithmetic_intensity=4.0,
            compute_floor_us=ideal_us,
            memory_floor_us=ideal_us,
            ideal_roofline_us=ideal_us,
            dominant_bound="memory",
        ),
    )


def _make_result(
    cid: str,
    anchors: Sequence[AnchorCaseSpec],
    lats: Mapping[int, float],
    *,
    passed: bool = True,
) -> CandidateSurrogateResult:
    metrics = [
        build_anchor_metrics(
            anc,
            accuracy_passed=passed,
            cand_us=float(lats[anc.case_num]),
        )
        for anc in anchors
    ]
    return aggregate_surrogate_result(cid, "synthetic/op", metrics)


def _write_synthetic_cannbench_task(
    tasks_root: Path,
    *,
    level: str,
    op_name: str,
    golden_source: str,
    entrypoint: str,
    proto: dict[str, object],
    cases: list[dict[str, object]],
    desc_md: str = "",
) -> Path:
    import yaml

    op_dir = tasks_root / level / op_name
    op_dir.mkdir(parents=True, exist_ok=True)
    (op_dir / "golden.py").write_text(dedent(golden_source), encoding="utf-8")
    (op_dir / "proto.yaml").write_text(
        yaml.safe_dump({"operator": proto}, sort_keys=False),
        encoding="utf-8",
    )
    (op_dir / "cases.yaml").write_text(
        yaml.safe_dump({"cases": cases}, sort_keys=False),
        encoding="utf-8",
    )
    if desc_md:
        (op_dir / "desc.md").write_text(dedent(desc_md), encoding="utf-8")
    meta_dir = tasks_root / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_path = meta_dir / "950pr.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    level_meta = meta.setdefault(level, {})
    op_meta = level_meta.setdefault(op_name, {})
    for row in cases:
        op_meta[str(row["case_id"])] = {"ref_perf_us": 100.0, "t_hw_us": 20.0}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return op_dir


def _real_cannbench_repo() -> Path | None:
    env_repo = os.environ.get("TILEFOUNDRY_NPU_CANNBENCH_REPO", "").strip()
    candidates = (
        Path(env_repo).expanduser() if env_repo else None,
        Path.home() / "workspace" / "cann-bench",
    )
    for candidate in candidates:
        if candidate is not None and (candidate / "tasks" / "level1").is_dir():
            return candidate
    return None


def test_pure_golden_mode_cli_marathon_provenance_and_typed_abi(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parser = build_arg_parser()
    parsed = parser.parse_args(["--operator", "level1/exp_custom", "--pure-golden"])
    assert parsed.pure_golden is True

    cann_repo = tmp_path / "cann-bench"
    tasks_root = cann_repo / "tasks"
    _write_synthetic_cannbench_task(
        tasks_root,
        level="level3",
        op_name="fused_norm",
        entrypoint="fused_norm",
        golden_source="""
        import torch
        from typing import Optional

        def fused_norm(
            x: torch.Tensor,
            residual: Optional[torch.Tensor] = None,
            eps: float = 1e-6,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            y = x if residual is None else x + residual
            var = (y * y).mean(dim=-1, keepdim=True)
            out = y * torch.rsqrt(var + eps)
            return out, var
        """,
        proto={
            "inputs": [
                {"name": "x", "dtype": ["float16", "bfloat16"]},
                {"name": "residual", "dtype": ["float16", "bfloat16"], "optional": True},
            ],
            "attrs": [{"name": "eps", "dtype": "float", "default": 1e-6}],
            "outputs": [
                {"name": "out", "dtype": ["float16"]},
                {"name": "var", "dtype": ["float32"]},
            ],
        },
        cases=[
            {
                "case_id": 1,
                "input_shape": [[64, 256], [64, 256]],
                "dtype": ["float16", "float16"],
            },
            {
                "case_id": 2,
                "input_shape": [[128, 513], [128, 513]],
                "dtype": ["float16", "float16"],
            },
        ],
        desc_md="Fused residual add and RMS normalization returning (out, var).",
    )

    mod_dir = tmp_path / "mod_dir"
    mod_dir.mkdir(parents=True)
    target_mod = mod_dir / "kernels.py"
    target_mod.write_text(_DUAL_MODULE_SRC, encoding="utf-8")
    sibling_model = mod_dir / "model.py"
    sibling_model.write_text("raise SyntaxError('should never be read in pure_golden')\n", encoding="utf-8")
    sibling_sched = mod_dir / "scheduled.py"
    sibling_sched.write_text("raise SyntaxError('should never be read in pure_golden')\n", encoding="utf-8")

    regimes = (
        RegimeTargetSpec(
            regime_tag="reg_a",
            description="Aligned",
            editable_symbols=("_build_reg_a",),
            builder_candidates=("_build_reg_a",),
            sample_builder_kwargs={},
            representative_hir_dims={"M": 64, "D": 256},
            representative_anchor_dims={"M": 64, "D": 256},
            representative_case_entry={
                "case_id": 1,
                "input_shape": [[64, 256], [64, 256]],
                "dtype": ["float16", "float16"],
            },
        ),
        RegimeTargetSpec(
            regime_tag="reg_b",
            description="Tail",
            editable_symbols=("_build_reg_b",),
            builder_candidates=("_build_reg_b",),
            sample_builder_kwargs={},
            representative_hir_dims={"M": 128, "D": 513},
            representative_anchor_dims={"M": 128, "D": 513},
            representative_case_entry={
                "case_id": 2,
                "input_shape": [[128, 513], [128, 513]],
                "dtype": ["float16", "float16"],
            },
        ),
    )
    anchors = (
        AnchorCaseSpec("level3/fused_norm", 1, "reg_a", 100.0, 20.0),
        AnchorCaseSpec("level3/fused_norm", 2, "reg_b", 80.0, 16.0),
    )

    backend = _CapturingBackend(
        lambda req, _idx: (
            json.dumps(
                {
                    "physical_hypothesis": "increase tile_m",
                    "target_local_metric": "latency_us",
                    "dsl_source": _REG_A_IMPROVED,
                }
            )
            if req.role == "explore_creative"
            else _REG_A_IMPROVED
        )
    )

    tracker = BayesianRuleTracker.from_rules(
        [
            AscendCausalRule(
                rule_id="R_TEST",
                symptom_family=SymptomFamily.DMA_REDUNDANT_GM_ROUNDTRIP.value,
                regime_guard="is_legal",
                precondition="PIPE_V > 0",
                transform_template="Increase tile size.",
                expected_postcondition="PIPE_V <= prev.PIPE_V",
                source_provenance="test",
            )
        ]
    )

    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(
            operator_name="level3/fused_norm",
            max_rounds=1,
            k_rule_workers=1,
            pure_golden=True,
        ),
        target_module_path=target_mod,
        cannbench_repo=cann_repo,
        anchors=anchors,
        regime_specs=regimes,
        perf_target=PerfTargetSpec("level3/fused_norm", 2.0, 1.5, 0.9, 90.0),
        llm_backend=backend,
        tracker=tracker,
        l0_checker=lambda src, _r: _make_stage_a(),
        l1_compiler=lambda _s, _r: StageBResult(
            passed=True, compiler_summary=_make_bisheng_summary()
        ),
        l2_batch_evaluator=lambda cands, ancs: {
            cid: _make_result(cid, ancs, {1: 80.0, 2: 70.0}) for cid, _p in cands
        },
        l3_contract_runner=lambda _p: L3AcceptanceResult(passed=True),
    )

    report = asyncio.run(runner.run())
    assert isinstance(report, CampaignReport)
    assert report.pure_golden_mode is True
    assert report.blueprint_provenance["pure_golden_mode"] is True
    assert report.blueprint_provenance["source_mode"] == "pure_golden"
    assert str(report.blueprint_provenance["golden_path"]).endswith("golden.py")

    report_dict = report.to_dict()
    assert report_dict["pure_golden_mode"] is True
    assert report_dict["blueprint_provenance"] == report.blueprint_provenance
    assert "stagnation_diagnosis" in report_dict

    exploit_reqs = [r for r in backend.requests if r.role == "exploit_rule"]
    creative_reqs = [r for r in backend.requests if r.role == "explore_creative"]
    assert exploit_reqs and creative_reqs
    for req in exploit_reqs + creative_reqs:
        assert "### EXACT GOLDEN ENTRYPOINT ABI" in req.user_prompt
        assert "def fused_norm(" in req.user_prompt
        assert "Optional[torch.Tensor]" in req.user_prompt
        assert "tuple[torch.Tensor, torch.Tensor]" in req.user_prompt
        assert "### IMMUTABLE HOST / PRIMFUNC ABI CONTRACT" in req.user_prompt

    seen_configs: list[CampaignConfig] = []

    class _StubReport:
        def to_dict(self) -> dict[str, object]:
            return {"status": "ok"}

    class _CapturingRunner:
        def __init__(self, **kwargs: object) -> None:
            cfg = kwargs["config"]
            assert isinstance(cfg, CampaignConfig)
            seen_configs.append(cfg)

        async def run(self, *, resume: bool) -> _StubReport:
            return _StubReport()

    worktree = tmp_path / "wt"
    worktree.mkdir()
    monkeypatch.setattr(marathon, "_resolve_worktree", lambda _c: (worktree, worktree))
    monkeypatch.setattr(marathon, "_ensure_sys_paths", lambda _w: None)
    monkeypatch.setattr(marathon, "_configure_marathon_logging", lambda _p: None)
    monkeypatch.setattr(marathon, "ensure_marathon_env_file", lambda **_k: worktree / "env.sh")
    monkeypatch.setattr(
        marathon,
        "configure_marathon_process_env",
        lambda **_k: {"TILEFOUNDRY_NPU_CANNBENCH_REPO": str(cann_repo)},
    )
    monkeypatch.setattr(marathon, "create_llm_backend_from_env", lambda **_k: backend)
    monkeypatch.setattr(marathon, "build_resilient_l2_evaluator", lambda **_k: (lambda c, a: {}))
    monkeypatch.setattr(marathon, "run_l3_full_contract_via_agent_gates", lambda **_k: None)
    monkeypatch.setattr(marathon, "resolve_default_experience_lake_dir", lambda **_k: None)
    monkeypatch.setattr(marathon, "ExtremeCampaignRunner", _CapturingRunner)

    state_dir = tmp_path / "state"
    rc = run_operator_marathon(
        MarathonConfig(
            operator_name="level3/fused_norm",
            state_dir=state_dir,
            auto_bootstrap_seed=False,
            salvage_beam_candidates=False,
            pure_golden=True,
        )
    )
    assert rc == 0
    assert seen_configs and seen_configs[0].pure_golden is True


def test_five_category_synthetic_and_optional_real_cannbench_pure_golden_e2e(
    tmp_path: Path,
) -> None:
    cann_repo = tmp_path / "cann-bench"
    tasks_root = cann_repo / "tasks"
    categories = [
        (
            "level1",
            "transcendental_elem",
            "transcendental_elem",
            """
            import torch

            def transcendental_elem(x: torch.Tensor) -> torch.Tensor:
                return torch.rsqrt(torch.exp(x) + 1.0)
            """,
            {"inputs": [{"name": "x"}], "outputs": [{"name": "y", "dtype": ["float16"]}]},
            [{"case_id": 1, "input_shape": [[64, 256]], "dtype": "float16"}],
            {"M": 64, "D": 256},
        ),
        (
            "level2",
            "tail_reduce",
            "tail_reduce",
            """
            import torch

            def tail_reduce(x: torch.Tensor) -> torch.Tensor:
                return torch.softmax(x, dim=-1)
            """,
            {"inputs": [{"name": "x"}], "outputs": [{"name": "y", "dtype": ["float16"]}]},
            [{"case_id": 1, "input_shape": [[33, 513]], "dtype": "float16"}],
            {"M": 33, "D": 513},
        ),
        (
            "level3",
            "bmm_trans",
            "bmm_trans",
            """
            import torch

            def bmm_trans(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
                return torch.bmm(a, b.transpose(-1, -2))
            """,
            {
                "inputs": [{"name": "a"}, {"name": "b"}],
                "outputs": [{"name": "c", "dtype": ["float16"]}],
            },
            [
                {
                    "case_id": 1,
                    "input_shape": [[4, 64, 128], [4, 256, 128]],
                    "dtype": ["float16", "float16"],
                    "attrs": {"transpose_b": True},
                }
            ],
            {"B": 4, "M": 64, "N": 256, "K": 128},
        ),
        (
            "level3",
            "gmm_like",
            "gmm_like",
            """
            import torch

            def gmm_like(x: torch.Tensor, w: torch.Tensor, group_list: torch.Tensor) -> torch.Tensor:
                return torch.matmul(x, w)
            """,
            {
                "inputs": [{"name": "x"}, {"name": "w"}, {"name": "group_list"}],
                "outputs": [{"name": "y", "dtype": ["float16"]}],
            },
            [
                {
                    "case_id": 1,
                    "input_shape": [[128, 64], [64, 128], [4]],
                    "dtype": ["float16", "float16", "int64"],
                }
            ],
            {"M": 128, "N": 128, "K": 64, "E": 4},
        ),
        (
            "level3",
            "opt_multi_out",
            "opt_multi_out",
            """
            import torch
            from typing import Optional

            def opt_multi_out(
                x: torch.Tensor,
                bias: Optional[torch.Tensor] = None,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                y = x if bias is None else x + bias
                return y.to(torch.int8), y.mean(dim=-1)
            """,
            {
                "inputs": [{"name": "x"}, {"name": "bias", "optional": True}],
                "outputs": [
                    {"name": "q", "dtype": ["int8"]},
                    {"name": "s", "dtype": ["float32"]},
                ],
            },
            [{"case_id": 1, "input_shape": [[32, 128], [128]], "dtype": ["float16", "float16"]}],
            {"M": 32, "D": 128},
        ),
    ]

    for level, op_name, entry, golden_src, proto, cases, dims in categories:
        _write_synthetic_cannbench_task(
            tasks_root,
            level=level,
            op_name=op_name,
            entrypoint=entry,
            golden_source=golden_src,
            proto=proto,
            cases=cases,
        )
        mod_path = tmp_path / f"{op_name}_kernels.py"
        mod_path.write_text(_DUAL_MODULE_SRC, encoding="utf-8")
        regime = RegimeTargetSpec(
            regime_tag="reg_a",
            description="Single",
            editable_symbols=("_build_reg_a",),
            builder_candidates=("_build_reg_a",),
            sample_builder_kwargs={},
            representative_hir_dims=dims,
            representative_anchor_dims=dims,
            representative_case_entry=cases[0],
        )
        anc = AnchorCaseSpec(f"{level}/{op_name}", 1, "reg_a", 100.0, 20.0)
        backend = _CapturingBackend(lambda _r, _i: _REG_A_IMPROVED)
        runner = PerfTargetCampaignRunner(
            config=CampaignConfig(
                operator_name=f"{level}/{op_name}",
                max_rounds=1,
                k_rule_workers=0,
                pure_golden=True,
            ),
            target_module_path=mod_path,
            cannbench_repo=cann_repo,
            anchors=(anc,),
            regime_specs=(regime,),
            perf_target=PerfTargetSpec(f"{level}/{op_name}", 2.0, 1.5, 0.9, 90.0),
            llm_backend=backend,
            l0_checker=lambda _s, _r: _make_stage_a(),
            l1_compiler=lambda _s, _r: StageBResult(
                passed=True, compiler_summary=_make_bisheng_summary()
            ),
            l2_batch_evaluator=lambda cands, ancs: {
                cid: _make_result(cid, ancs, {1: 80.0}) for cid, _p in cands
            },
            l3_contract_runner=lambda _p: L3AcceptanceResult(passed=True),
        )
        report = asyncio.run(runner.run())
        assert report.pure_golden_mode is True
        assert report.blueprint_provenance["source_mode"] == "pure_golden"
        assert f"def {entry}(" in backend.requests[0].user_prompt

    real_repo = _real_cannbench_repo()
    if real_repo is not None:
        real_reg = CannbenchOpsRegistry(tasks_root=real_repo, pure_golden=True)
        for op_id in (
            "level1/exp",
            "level2/softmax",
            "level3/grouped_matmul",
            "level4/gqa",
            "level3/add_rms_norm_dynamic_quant",
        ):
            spec = real_reg.resolve(op_id)
            real_mod = tmp_path / f"real_{spec.task.replace('/', '_')}_kernels.py"
            real_mod.write_text(_DUAL_MODULE_SRC, encoding="utf-8")
            real_regimes = real_reg.get_regime_specs(op_id)
            first_reg = replace(
                real_regimes[0],
                editable_symbols=("_build_reg_a",),
                builder_candidates=("_build_reg_a",),
            )
            first_anc = replace(spec.anchors[0], regime_tag=first_reg.regime_tag)
            real_backend = _CapturingBackend(lambda _r, _i: _REG_A_IMPROVED)
            real_runner = PerfTargetCampaignRunner(
                config=CampaignConfig(
                    operator_name=op_id,
                    max_rounds=1,
                    k_rule_workers=0,
                    pure_golden=True,
                ),
                target_module_path=real_mod,
                cannbench_repo=real_repo,
                anchors=(first_anc,),
                regime_specs=(first_reg,),
                perf_target=spec.perf_target,
                llm_backend=real_backend,
                l0_checker=lambda _s, _r: _make_stage_a(),
                l1_compiler=lambda _s, _r: StageBResult(
                    passed=True, compiler_summary=_make_bisheng_summary()
                ),
                l2_batch_evaluator=lambda cands, ancs: {
                    cid: _make_result(cid, ancs, {first_anc.case_num: first_anc.ref_perf_us * 0.9})
                    for cid, _p in cands
                },
                l3_contract_runner=lambda _p: L3AcceptanceResult(passed=True),
            )
            real_report = asyncio.run(real_runner.run())
            assert real_report.pure_golden_mode is True
            assert real_report.blueprint_provenance["golden_path"] == str(spec.golden_path)
            assert f"def {spec.entrypoint}(" in real_backend.requests[0].user_prompt


def test_online_cost_model_dedup_all_regime_baseline_and_stepping_stone_deltas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target_mod = tmp_path / "kernels.py"
    target_mod.write_text(_DUAL_MODULE_SRC, encoding="utf-8")
    lake_dir = tmp_path / "experience_lake"
    regimes = _dual_regimes()
    anchors = _dual_anchors()

    l0_calls: list[str] = []
    l1_calls: list[str] = []

    def _l0_fn(_src: str, rspec: RegimeTargetSpec) -> StageAResult:
        l0_calls.append(rspec.regime_tag)
        return _make_stage_a()

    def _l1_fn(_src: str, rspec: RegimeTargetSpec) -> StageBResult:
        l1_calls.append(rspec.regime_tag)
        return StageBResult(
            passed=True,
            compiler_summary=_make_bisheng_summary(
                scalar_regs=7 if rspec.regime_tag == "reg_a" else 9
            ),
        )

    backend = _CapturingBackend(lambda _r, _i: _REG_A_IMPROVED)
    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(
            operator_name="synthetic/op",
            max_rounds=1,
            k_rule_workers=0,
            min_surrogate_gain=0.1,
        ),
        target_module_path=target_mod,
        anchors=anchors,
        regime_specs=regimes,
        perf_target=_perf_target(),
        llm_backend=backend,
        l0_checker=_l0_fn,
        l1_compiler=_l1_fn,
        l2_batch_evaluator=lambda cands, ancs: {
            cid: _make_result(
                cid,
                ancs,
                {1: 100.0, 2: 80.0} if cid == "incumbent_baseline" else {1: 60.0, 2: 75.0},
            )
            for cid, _p in cands
        },
        l3_contract_runner=lambda _p: L3AcceptanceResult(passed=True),
        experience_lake_dir=lake_dir,
        regime_blueprints={"reg_a": _make_blueprint(), "reg_b": _make_blueprint()},
    )

    seen_exclude_run_ids: list[str | None] = []
    orig_replay = runner._cost_model.replay_experience_lake

    def _wrapped_replay(
        lake: ExperienceLake | str | Path,
        *,
        persist_calibration: bool = False,
        exclude_run_id: str | None = None,
    ) -> dict[str, int]:
        seen_exclude_run_ids.append(exclude_run_id)
        return orig_replay(
            lake,
            persist_calibration=persist_calibration,
            exclude_run_id=exclude_run_id,
        )

    monkeypatch.setattr(runner._cost_model, "replay_experience_lake", _wrapped_replay)

    report = asyncio.run(runner.run())
    assert report.total_rounds == 1
    assert "reg_a" in l0_calls and "reg_b" in l0_calls
    assert "reg_a" in l1_calls and "reg_b" in l1_calls

    base_sha = compute_source_sha256(_DUAL_MODULE_SRC)
    cand_a_path = lake_dir / "candidates" / f"{base_sha}_reg_a.json"
    cand_b_path = lake_dir / "candidates" / f"{base_sha}_reg_b.json"
    assert cand_a_path.is_file()
    assert cand_b_path.is_file()
    cand_a_json = json.loads(cand_a_path.read_text(encoding="utf-8"))
    cand_b_json = json.loads(cand_b_path.read_text(encoding="utf-8"))
    assert cand_a_json["bisheng_summary"]["scalar_regs"] == 7
    assert cand_b_json["bisheng_summary"]["scalar_regs"] == 9

    lake = ExperienceLake(lake_dir)
    _cands, obs = lake.load_records()
    baseline_obs = [o for o in obs if o.cand_sha == base_sha and o.eval_tier == "baseline"]
    assert sorted(o.case_num for o in baseline_obs) == [1, 2]

    assert seen_exclude_run_ids and all(rid == runner._run_id for rid in seen_exclude_run_ids)
    assert runner._cost_model._replayed_obs_keys
    assert runner._cost_model._replayed_pair_keys
    replay_counts = orig_replay(lake_dir)
    assert replay_counts["observations"] == 0
    assert replay_counts["pair_deltas"] == 0

    stone_beam = runner.beam_for_regime("reg_a")
    assert stone_beam.exploit_beam
    recorded_cand = stone_beam.exploit_beam[0]
    assert recorded_cand.case_latencies_us == {1: 60.0, 2: 75.0}
    assert isinstance(recorded_cand.policy, dict) and recorded_cand.policy

    stone_module_path = tmp_path / "stone_kernels.py"
    stone_module_path.write_text(_DUAL_MODULE_SRC, encoding="utf-8")
    stone_runner = PerfTargetCampaignRunner(
        config=CampaignConfig(
            operator_name="synthetic/op",
            max_rounds=1,
            k_rule_workers=0,
            stagnation_escalate_rounds=0,
        ),
        target_module_path=stone_module_path,
        anchors=anchors,
        regime_specs=regimes,
        perf_target=_perf_target(),
        llm_backend=_CapturingBackend(lambda _r, _i: _REG_A_IMPROVED),
        l0_checker=lambda _s, _r: _make_stage_a(),
        l1_compiler=lambda _s, _r: StageBResult(
            passed=True, compiler_summary=_make_bisheng_summary()
        ),
        l2_batch_evaluator=lambda cands, ancs: {
            cid: _make_result(
                cid,
                ancs,
                {1: 100.0, 2: 80.0} if cid == "incumbent_baseline" else {1: 55.0, 2: 70.0},
            )
            for cid, _p in cands
        },
        l3_contract_runner=lambda _p: L3AcceptanceResult(passed=False),
        regime_blueprints={"reg_a": _make_blueprint(), "reg_b": _make_blueprint()},
    )
    stone_ledger = _make_stage_a().ledger
    assert stone_ledger is not None
    stone_res = StaticDynamicResidual(
        measured_latency_us=70.0,
        predicted_bottleneck_util=0.5,
        measured_aic_mac_ratio=0.0,
        measured_aiv_vec_ratio=0.4,
        measured_mte2_wait_ratio=0.2,
        measured_sync_stall_ratio=0.1,
        unexplained_gap_ratio=0.1,
    )
    stone = BeamCandidate(
        candidate_id="stone_r0",
        round_idx=0,
        worker_kind="creative_explore",
        rule_id=None,
        dsl_source=_REG_A_STONE,
        track="unconstrained",
        ledger=stone_ledger,
        residual=stone_res,
        latency_us=70.0,
        accuracy_passed=True,
        case_latencies_us={1: 71.0, 2: 79.0},
        policy={"tile_m": 64, "sentinel_parent_knob": 999},
    )
    stone_runner.beam_for_regime("reg_a").stepping_stone_beam = [stone]

    captured_calls: list[dict[str, object]] = []
    orig_record = stone_runner._record_surrogate_observations

    def _spy_record(**kwargs: object) -> None:
        captured_calls.append(dict(kwargs))
        orig_record(**kwargs)

    monkeypatch.setattr(stone_runner, "_record_surrogate_observations", _spy_record)

    captured_pair_deltas: list[tuple[dict[str, object], float]] = []
    orig_pair = stone_runner._cost_model.observe_pair_delta

    def _spy_pair(
        cand_wl: Mapping[str, object],
        cand_us: float,
        parent_wl: Mapping[str, object],
        parent_us: float,
        **kwargs: object,
    ) -> None:
        captured_pair_deltas.append((dict(parent_wl), float(parent_us)))
        orig_pair(cand_wl, cand_us, parent_wl, parent_us, **kwargs)

    monkeypatch.setattr(stone_runner._cost_model, "observe_pair_delta", _spy_pair)

    asyncio.run(stone_runner.run())
    child_calls = [c for c in captured_calls if c.get("worker_kind") != "baseline"]
    assert child_calls
    last_call = child_calls[-1]
    assert last_call["parent_policy"] == {"tile_m": 64, "sentinel_parent_knob": 999}
    parent_res_arg = last_call["parent_result"]
    assert isinstance(parent_res_arg, CandidateSurrogateResult)
    assert {a.case_num: a.cand_us for a in parent_res_arg.anchors} == {1: 71.0, 2: 79.0}
    assert captured_pair_deltas
    assert {us for _wl, us in captured_pair_deltas} == {71.0, 79.0}


def test_p2_deep_dive_trigger_prompt_injection_and_symptom_priority_unlock(
    tmp_path: Path,
) -> None:
    residual = StaticDynamicResidual(
        measured_latency_us=42.0,
        predicted_bottleneck_util=0.8,
        measured_aic_mac_ratio=0.0,
        measured_aiv_vec_ratio=0.45,
        measured_mte2_wait_ratio=0.40,
        measured_sync_stall_ratio=0.25,
        unexplained_gap_ratio=0.20,
    )
    assert PerfTargetCampaignRunner._format_msprof_section(residual) != ""
    p2_text = PerfTargetCampaignRunner._format_msprof_section(
        None,
        p2_diag={
            "p2_hotspots": [
                {"source_location": "k.py:10", "instruction_count": 30, "instruction_share": 0.45},
                {"source_location": "k.py:18", "instruction_count": 20},
                {"source_location": "k.py:25", "instruction_count": 10},
                {"source_location": "k.py:30", "instruction_count": 5},
            ],
            "p2_mte_waitflag_share": 0.375,
            "p2_issue_interval_us": {"MTE2": {"p50_us": 1.25, "p90_us": 2.5}},
        },
    )
    assert "### HARDWARE MSPROF TELEMETRY" in p2_text
    assert "k.py:10" in p2_text and "k.py:25" in p2_text and "k.py:30" not in p2_text
    assert "p2_mte_waitflag_share: 0.3750" in p2_text
    assert "MTE2(p50=1.250us, p90=2.500us)" in p2_text

    target_mod = tmp_path / "kernels.py"
    target_mod.write_text(_DUAL_MODULE_SRC, encoding="utf-8")
    regimes = (_dual_regimes()[0],)
    anchors = (_dual_anchors()[0],)

    high_prior_sync = AscendCausalRule(
        rule_id="R_HIGH_SYNC",
        symptom_family=SymptomFamily.SYNC_BARRIER_SERIALIZATION.value,
        regime_guard="is_legal",
        precondition="PIPE_V > 0",
        transform_template="Adjust sync barriers.",
        expected_postcondition="PIPE_V <= prev.PIPE_V",
        source_provenance="test",
        alpha_prior=30.0,
        beta_prior=1.0,
    )
    low_prior_dma = AscendCausalRule(
        rule_id="R_LOW_DMA",
        symptom_family=SymptomFamily.DMA_REDUNDANT_GM_ROUNDTRIP.value,
        regime_guard="is_legal",
        precondition="PIPE_V > 0",
        transform_template="Fuse GM roundtrip into UB.",
        expected_postcondition="PIPE_V <= prev.PIPE_V",
        source_provenance="test",
        alpha_prior=2.0,
        beta_prior=2.0,
    )
    tracker = BayesianRuleTracker.from_rules([high_prior_sync, low_prior_dma])

    p2_calls: list[tuple[str, int]] = []

    def _p2_collector(
        op_name: str, cand_path: str, anc_seq: Sequence[AnchorCaseSpec]
    ) -> Mapping[str, object]:
        p2_calls.append((op_name, int(anc_seq[0].case_num)))
        return {
            "reg_a": {
                "p2_hotspots": [{"source_location": "op.py:42", "instruction_count": 50}],
                "p2_mte_waitflag_share": 0.41,
                "p2_issue_interval_us": {"PIPE_V": {"p50_us": 0.8, "p90_us": 1.6}},
            }
        }

    backend = _CapturingBackend(lambda _r, _i: _REG_A_IMPROVED)
    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(
            operator_name="synthetic/op",
            max_rounds=3,
            k_rule_workers=1,
            stagnation_escalate_rounds=2,
            stagnation_freeze_rounds=5,
        ),
        target_module_path=target_mod,
        anchors=anchors,
        regime_specs=regimes,
        perf_target=_perf_target(),
        llm_backend=backend,
        tracker=tracker,
        l0_checker=lambda _s, _r: _make_stage_a(
            gm_bytes=262144,
            pipe_cycles={"AIV0.PIPE_V": 100, "AIV1.PIPE_V": 100, "AIV0.PIPE_MTE2": 240},
        ),
        l1_compiler=lambda _s, _r: StageBResult(
            passed=True, compiler_summary=_make_bisheng_summary()
        ),
        l2_batch_evaluator=lambda cands, ancs: {
            cid: _make_result(cid, ancs, {1: 100.0}) for cid, _p in cands
        },
        msprof_collector=lambda _op, _p, _a: {
            "reg_a": {
                "dur_us": 100.0,
                "aiv_vec_ratio": 0.25,
                "mte2_wait_ratio": 0.45,
                "l2_hit_rate": 0.58,
                "sync_stall_ratio": 0.60,
            }
        },
        p2_deep_dive_collector=_p2_collector,
        regime_blueprints={"reg_a": _make_blueprint(ideal_us=15.0)},
    )

    report = asyncio.run(runner.run())
    assert report.total_rounds == 3
    assert len(p2_calls) == 1
    round3_exploit = [r for r in backend.requests if r.role == "exploit_rule"][-1]
    assert "Rule ID: R_LOW_DMA" in round3_exploit.user_prompt
    assert "op.py:42" in round3_exploit.user_prompt
    assert "p2_mte_waitflag_share: 0.4100" in round3_exploit.user_prompt

    mask = runner._freedom_masks["reg_a"]
    for facet in SYMPTOM_FAMILY_TO_FACETS[SymptomFamily.DMA_REDUNDANT_GM_ROUNDTRIP]:
        assert mask.is_unlocked(facet) is True

    failing_p2_runner = PerfTargetCampaignRunner(
        config=CampaignConfig(
            operator_name="synthetic/op",
            max_rounds=3,
            k_rule_workers=0,
            stagnation_escalate_rounds=2,
            stagnation_freeze_rounds=5,
        ),
        target_module_path=target_mod,
        anchors=anchors,
        regime_specs=regimes,
        perf_target=_perf_target(),
        llm_backend=_CapturingBackend(lambda _r, _i: _REG_A_IMPROVED),
        l0_checker=lambda _s, _r: _make_stage_a(
            gm_bytes=262144,
            pipe_cycles={"AIV0.PIPE_V": 100, "AIV1.PIPE_V": 100, "AIV0.PIPE_MTE2": 240},
        ),
        l1_compiler=lambda _s, _r: StageBResult(
            passed=True, compiler_summary=_make_bisheng_summary()
        ),
        l2_batch_evaluator=lambda cands, ancs: {
            cid: _make_result(cid, ancs, {1: 100.0}) for cid, _p in cands
        },
        msprof_collector=lambda _op, _p, _a: {
            "reg_a": {
                "dur_us": 100.0,
                "aiv_vec_ratio": 0.25,
                "mte2_wait_ratio": 0.45,
                "l2_hit_rate": 0.58,
                "sync_stall_ratio": 0.60,
            }
        },
        p2_deep_dive_collector=lambda _op, _p, _a: (_ for _ in ()).throw(RuntimeError("p2 fail")),
        regime_blueprints={"reg_a": _make_blueprint(ideal_us=15.0)},
    )
    res_fail_safe = asyncio.run(failing_p2_runner.run())
    assert res_fail_safe.total_rounds == 3


def test_stagnation_diagnosis_six_taxonomy_and_checkpoint_roundtrip(
    tmp_path: Path,
) -> None:
    expected_taxonomy = {
        "roofline_saturated",
        "rule_starvation",
        "l0_l1_compile_wall",
        "cost_prescreen_mismatch",
        "l3_accuracy_instability",
        "hardware_unmodeled_stall",
    }

    def _run_diag_scenario(
        name: str,
        *,
        rules: Sequence[AscendCausalRule],
        l0_fn: object,
        l1_fn: object,
        l2_lat: float,
        l3_passed: bool,
        prescreen: bool = False,
        top_k: int | None = None,
        cost_model_mode: str = "auto",
        k_rule_workers: int | None = None,
        max_rounds: int = 2,
        ckpt_path: Path | None = None,
        resume: bool = False,
    ) -> CampaignReport:
        mod_path = tmp_path / f"{name}.py"
        if not mod_path.is_file():
            mod_path.write_text(_DUAL_MODULE_SRC, encoding="utf-8")
        regimes = (_dual_regimes()[0],)
        anchors = (_dual_anchors()[0],)
        tracker = BayesianRuleTracker.from_rules(rules)
        backend = _CapturingBackend(
            lambda req, idx: (
                json.dumps(
                    {
                        "physical_hypothesis": f"hyp_{idx}",
                        "target_local_metric": "latency_us",
                        "dsl_source": _REG_A_IMPROVED.replace(
                            "tile_m=32", f"tile_m={32 + idx}"
                        ),
                    }
                )
                if req.role == "explore_creative"
                else _REG_A_IMPROVED.replace("tile_m=32", f"tile_m={32 + idx}")
            )
        )
        runner = PerfTargetCampaignRunner(
            config=CampaignConfig(
                operator_name="synthetic/op",
                max_rounds=max_rounds,
                k_rule_workers=(
                    k_rule_workers if k_rule_workers is not None else (1 if rules else 2)
                ),
                enable_tilesize_prescreen=prescreen,
                cost_prescreen_top_k=top_k,
                cost_model_mode=cost_model_mode,
                min_surrogate_gain=0.1,
                stagnation_freeze_rounds=10,
            ),
            target_module_path=mod_path,
            anchors=anchors,
            regime_specs=regimes,
            perf_target=_perf_target(eta=0.85),
            llm_backend=backend,
            tracker=tracker,
            l0_checker=l0_fn,
            l1_compiler=l1_fn,
            l2_batch_evaluator=lambda cands, ancs: {
                cid: _make_result(
                    cid,
                    ancs,
                    {1: 100.0 if cid == "incumbent_baseline" else l2_lat},
                )
                for cid, _p in cands
            },
            l3_contract_runner=lambda _p: L3AcceptanceResult(
                passed=l3_passed,
                failure_kind=("" if l3_passed else "candidate"),
                execution_terminal=True,
            ),
            checkpoint_path=ckpt_path,
            regime_blueprints={"reg_a": _make_blueprint()},
        )
        return asyncio.run(runner.run(resume=resume))

    dummy_rule = AscendCausalRule(
        rule_id="R_ANY",
        symptom_family=SymptomFamily.DMA_REDUNDANT_GM_ROUNDTRIP.value,
        regime_guard="is_legal",
        precondition="PIPE_V > 0",
        transform_template="Tune tile.",
        expected_postcondition="PIPE_V <= prev.PIPE_V",
        source_provenance="test",
    )
    ok_l0 = lambda _s, _r: _make_stage_a()
    ok_l1 = lambda _s, _r: StageBResult(passed=True, compiler_summary=_make_bisheng_summary())

    rep_sat = _run_diag_scenario(
        "sat",
        rules=[dummy_rule],
        l0_fn=ok_l0,
        l1_fn=ok_l1,
        l2_lat=22.0,
        l3_passed=True,
    )
    assert rep_sat.stagnation_diagnosis["primary_stagnation_cause"] == "roofline_saturated"

    rep_starv = _run_diag_scenario(
        "starv",
        rules=[],
        l0_fn=ok_l0,
        l1_fn=ok_l1,
        l2_lat=100.0,
        l3_passed=True,
    )
    assert rep_starv.stagnation_diagnosis["primary_stagnation_cause"] == "rule_starvation"
    reg_starv = rep_starv.stagnation_diagnosis["regimes"]["reg_a"]
    assert reg_starv["rule_starvation_rounds"] == 2

    rep_wall = _run_diag_scenario(
        "wall",
        rules=[dummy_rule],
        l0_fn=lambda src, _r: (
            _make_stage_a()
            if "tile_m=16" in src
            else StageAResult(passed=False, prim_func=None, ledger=None, error_message="ub overflow")
        ),
        l1_fn=ok_l1,
        l2_lat=100.0,
        l3_passed=True,
    )
    assert rep_wall.stagnation_diagnosis["primary_stagnation_cause"] == "l0_l1_compile_wall"
    assert rep_wall.stagnation_diagnosis["regimes"]["reg_a"]["l0_reject_reasons"]["ub_overflow"] > 0

    rep_l1_wall = _run_diag_scenario(
        "l1_wall",
        rules=[dummy_rule],
        l0_fn=ok_l0,
        l1_fn=lambda _s, _r: StageBResult(passed=False, error_message="bisheng compile failed"),
        l2_lat=100.0,
        l3_passed=True,
    )
    assert rep_l1_wall.stagnation_diagnosis["primary_stagnation_cause"] == "l0_l1_compile_wall"
    assert rep_l1_wall.stagnation_diagnosis["regimes"]["reg_a"]["l1_compiler_penalty_rejects"] > 0

    dummy_rule_2 = AscendCausalRule(
        rule_id="R_ANY_2",
        symptom_family=SymptomFamily.DMA_REDUNDANT_GM_ROUNDTRIP.value,
        regime_guard="is_legal",
        precondition="PIPE_V > 0",
        transform_template="Tune tile 2.",
        expected_postcondition="PIPE_V <= prev.PIPE_V",
        source_provenance="test",
    )
    dummy_rule_3 = AscendCausalRule(
        rule_id="R_ANY_3",
        symptom_family=SymptomFamily.DMA_REDUNDANT_GM_ROUNDTRIP.value,
        regime_guard="is_legal",
        precondition="PIPE_V > 0",
        transform_template="Tune tile 3.",
        expected_postcondition="PIPE_V <= prev.PIPE_V",
        source_provenance="test",
    )
    rep_prescreen = _run_diag_scenario(
        "prescreen",
        rules=[dummy_rule, dummy_rule_2, dummy_rule_3],
        l0_fn=lambda src, _r: (
            _make_stage_a(pipe_cycles={"AIV0.PIPE_V": 50000})
            if "tile_m=16" not in src
            else _make_stage_a()
        ),
        l1_fn=ok_l1,
        l2_lat=100.0,
        l3_passed=True,
        prescreen=True,
        top_k=1,
        cost_model_mode="coarse_only",
        k_rule_workers=3,
    )
    assert rep_prescreen.stagnation_diagnosis["primary_stagnation_cause"] == "cost_prescreen_mismatch"
    assert rep_prescreen.stagnation_diagnosis["regimes"]["reg_a"]["cost_prescreen_filtered_count"] >= 2

    rep_l3 = _run_diag_scenario(
        "l3_rb",
        rules=[dummy_rule],
        l0_fn=ok_l0,
        l1_fn=ok_l1,
        l2_lat=50.0,
        l3_passed=False,
    )
    assert rep_l3.stagnation_diagnosis["primary_stagnation_cause"] == "l3_accuracy_instability"
    assert rep_l3.stagnation_diagnosis["regimes"]["reg_a"]["l3_rollback_count"] == 2

    ckpt = tmp_path / "diag_ckpt.json"
    rep_stall_p1 = _run_diag_scenario(
        "stall",
        rules=[dummy_rule],
        l0_fn=ok_l0,
        l1_fn=ok_l1,
        l2_lat=100.0,
        l3_passed=True,
        max_rounds=1,
        ckpt_path=ckpt,
    )
    assert rep_stall_p1.stagnation_diagnosis["regimes"]["reg_a"]["l2_no_gain_rounds"] == 1
    rep_stall_p2 = _run_diag_scenario(
        "stall",
        rules=[dummy_rule],
        l0_fn=ok_l0,
        l1_fn=ok_l1,
        l2_lat=100.0,
        l3_passed=True,
        max_rounds=2,
        ckpt_path=ckpt,
        resume=True,
    )
    assert rep_stall_p2.stagnation_diagnosis["primary_stagnation_cause"] == "hardware_unmodeled_stall"
    assert rep_stall_p2.stagnation_diagnosis["regimes"]["reg_a"]["l2_no_gain_rounds"] == 2
    assert rep_stall_p2.stagnation_diagnosis["regimes"]["reg_a"]["top_symptom_families"]

    observed_causes = {
        rep_sat.stagnation_diagnosis["primary_stagnation_cause"],
        rep_starv.stagnation_diagnosis["primary_stagnation_cause"],
        rep_wall.stagnation_diagnosis["primary_stagnation_cause"],
        rep_prescreen.stagnation_diagnosis["primary_stagnation_cause"],
        rep_l3.stagnation_diagnosis["primary_stagnation_cause"],
        rep_stall_p2.stagnation_diagnosis["primary_stagnation_cause"],
    }
    assert observed_causes == expected_taxonomy

    legacy_payload = json.loads(ckpt.read_text(encoding="utf-8"))
    legacy_payload.pop("stagnation_diagnosis_stats", None)
    ckpt.write_text(json.dumps(legacy_payload, indent=2), encoding="utf-8")
    resumed_legacy = _run_diag_scenario(
        "stall",
        rules=[dummy_rule],
        l0_fn=ok_l0,
        l1_fn=ok_l1,
        l2_lat=100.0,
        l3_passed=True,
        max_rounds=3,
        ckpt_path=ckpt,
        resume=True,
    )
    assert resumed_legacy.stagnation_diagnosis["regimes"]["reg_a"]["l2_no_gain_rounds"] == 1


def test_hybrid_trackb_without_prim_func_and_freedom_mask_preserves_alg_monoid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify Track-B-only hybrid regimes run L0/L1 via bisheng and keep alg facet unlocked."""
    hyb_file = tmp_path / "hyb_op.py"
    hyb_src = dedent(
        '''
        import torch

        HYB_TRACKB_CU = """
        #include <tl_templates/ascend/kernel_entry.h>
        __global__ void hyb_k(__gm__ float* q, __gm__ float* y) {
            __simd_vf__ void (*fn)() = nullptr;
            asc_copy_gm2ub_align(nullptr, nullptr, 64);
        }
        """

        def _select_hyb_config(**kwargs) -> dict[str, object]:
            return {"pass_configs": {}, "compile_flags": []}

        def _build_hyb_prim_func(**kwargs) -> str:
            return HYB_TRACKB_CU

        def _render_trackb_tu(trackb_src: str = HYB_TRACKB_CU, **dims: object) -> str:
            return trackb_src + "\\nvoid launch() { hyb_k<<<1, 1, 1>>>(nullptr, nullptr); }\\n"

        def _get_kernel(**kwargs) -> object:
            return object()

        def hyb_op(q: torch.Tensor) -> torch.Tensor:
            return q

        kernel = hyb_op
        '''
    )
    hyb_file.write_text(hyb_src, encoding="utf-8")
    regime = RegimeTargetSpec(
        regime_tag="hyb_reg",
        description="hybrid track-b only",
        builder_candidates=("_build_hyb_prim_func",),
        config_selector_fn="_select_hyb_config",
        representative_hir_dims={"total_elements": 64},
        sample_builder_kwargs={"total_elements": 64},
        editable_symbols=("HYB_TRACKB_CU", "_select_hyb_config"),
        runtime_backend="hybrid_trackb",
        trackb_symbols=("HYB_TRACKB_CU",),
    )
    runner = PerfTargetCampaignRunner(
        config=CampaignConfig(operator_name="hyb_op", max_rounds=1, pure_golden=True),
        target_module_path=hyb_file,
        anchors=(
            AnchorCaseSpec(
                task="hyb_op",
                case_num=1,
                regime_tag="hyb_reg",
                ref_perf_us=10.0,
                t_hw_us=1.0,
            ),
        ),
        regime_specs=(regime,),
        perf_target=PerfTargetSpec(
            operator_name="hyb_op",
            target_avg_speedup=1.0,
            target_min_case_speedup=1.0,
            target_roofline_eta=0.5,
        ),
        source_root=tmp_path,
        l2_batch_evaluator=lambda _c, _a: {},
    )
    mask = runner._resolve_trackb_freedom_mask(regime)
    assert mask.alg_compute_and_monoid is True
    assert mask.exec_mode_and_spr is True
    assert mask.dma_and_cache_hierarchy is True

    stage_a = runner._run_l0(hyb_src, regime)
    assert stage_a.passed is True, stage_a.error_message

    monkeypatch.setattr(
        runner,
        "_compile_trackb_stage_b",
        lambda _cpp: StageBResult(passed=True),
    )
    stage_b = runner._run_l1(hyb_src, regime)
    assert stage_b.passed is True, stage_b.error_message
