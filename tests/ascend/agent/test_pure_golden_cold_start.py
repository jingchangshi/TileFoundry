"""Tests for pure-golden Stage-0 cold-start hardening and semantic analysis."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from textwrap import dedent

import pytest

from tests.ascend.benchmarks.cannbench_contract import ParamBuilder
from tilefoundry_ascend.agent.campaign import marathon
from tilefoundry_ascend.agent.campaign import surrogate_cannbench
from tilefoundry_ascend.agent.campaign.anchor_specs import (
    DEFAULT_REGIME_SPECS,
    AnchorCaseSpec,
    CannbenchOpsRegistry,
    RegimeTargetSpec,
    build_semantic_regime_specs,
    discover_module_trackb_symbols,
    extract_schema_driven_case_dims,
    extract_trackb_cpp_source,
)
from tilefoundry_ascend.agent.campaign.marathon import (
    MarathonConfig,
    _stage0_entrypoint_contract_error,
    build_marathon_arg_parser,
    build_stage0_oracle_validator,
    build_stage0_static_validator,
    run_operator_marathon,
)
from tilefoundry_ascend.agent.campaign.seed_synthesizer import (
    OracleAccuracyResult,
    _canonical_seed_track,
    _seed_module_abi,
    _seed_parameter_specs,
    extract_golden_entrypoint_contract,
    format_golden_entrypoint_prompt_block,
    prepare_operator_seed_or_takeover,
    validate_entrypoint_signature_compatibility,
)
from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (
    FAST_RESPONSE_SCHEMA,
    L3AcceptanceResult,
    _V2GateExecution,
    _collect_profile_via_agent_gates,
    _fast_wire_operator_name,
    _format_l3_failure_detail,
    _submit_fast_gate,
    run_multi_operator_surrogate_batch_via_agent_gates,
)
from tilefoundry_ascend.agent.orchestrator import gate_funnel
from tilefoundry_ascend.agent.orchestrator.gate_funnel import (
    StageAResult,
    StageBResult,
    check_npu_entrypoint_purity,
    run_stage_b_ascendc_compile,
)
from tilefoundry_ascend.agent.orchestrator.llm_client import GenerationRequest, MockLLMBackend
from tilefoundry_ascend.ascend.sil.pytorch_frontend import (
    PyTorchSemanticAnalyzer,
    RepresentativeCase,
)
from tilefoundry_ascend.ascend.target.dialect_surface import (
    CCE_CPP20_SURFACE,
    get_dialect_surface,
)


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


class _ScriptedBackend:
    """Async LLM backend replaying fixed module outputs and capturing prompts."""

    def __init__(self, replies: Sequence[str]) -> None:
        self._replies = list(replies)
        self.requests: list[GenerationRequest] = []

    async def generate(self, req: GenerationRequest) -> str:
        self.requests.append(req)
        idx = min(len(self.requests) - 1, len(self._replies) - 1)
        return self._replies[idx]


def test_analyze_source_traces_class_forward_aliases_conv_sdpa_and_sfu() -> None:
    analyzer = PyTorchSemanticAnalyzer()
    source = dedent(
        """
        import torch
        import torch.nn.functional as F

        class _ConvBlock:
            def _inner_norm(self, y):
                return torch.rsqrt(y.pow(2) + 1e-6)

            def forward(self, x, w):
                out = F.conv2d(x, w, padding=1)
                return self._inner_norm(F.silu(out))

        def _entry_impl(x, w):
            blk = _ConvBlock()
            return blk(x, w)

        conv_alias = _entry_impl

        class _DeadClass:
            def forward(self, z):
                return torch.gather(z, dim=0, index=z.long())
        """
    )
    sem = analyzer.analyze_source(source, entry_name="conv_alias")
    assert sem.has_matmul is True
    assert sem.has_conv is True
    assert sem.has_indirect_index is False
    assert sem.transcendental_op_count >= 3

    sdpa_source = dedent(
        """
        import torch.nn.functional as F

        def sdpa_op(q, k, v):
            return F.scaled_dot_product_attention(q, k, v, is_causal=True)
        """
    )
    sdpa_sem = analyzer.analyze_source(sdpa_source, entry_name="sdpa_op")
    assert sdpa_sem.has_matmul is True
    assert sdpa_sem.has_reduction is True
    assert sdpa_sem.has_online_exp_max is True
    assert sdpa_sem.base_axes.compute_engine == "coupled_1c2v"
    assert sdpa_sem.base_axes.memory_ring == "bidirectional_cv_ring"

    sfu_source = dedent(
        """
        import torch
        import torch.nn.functional as F

        def sfu_vec(x):
            a = torch.rsqrt(torch.exp(x) + 1.0)
            b = F.gelu(F.silu(torch.sigmoid(torch.tanh(a))))
            c = torch.sin(b) + torch.cos(b) + torch.erf(b)
            return torch.sqrt(torch.log(c.abs() + 1.0))
        """
    )
    sfu_sem = analyzer.analyze_source(sfu_source, entry_name="sfu_vec")
    pkt = analyzer.build_universal_blueprint(sfu_sem, {"M": 64, "D": 128})
    assert pkt.roofline.special_ops >= 64 * 128 * 9
    assert pkt.roofline.vector_flops > pkt.roofline.special_ops * 8


def test_bmm_3d_conv2d_and_representative_case_enhancements() -> None:
    analyzer = PyTorchSemanticAnalyzer()
    bmm_src = dedent(
        """
        import torch

        def bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return torch.bmm(a, b)
        """
    )
    bmm_sem = analyzer.analyze_source(bmm_src, entry_name="bmm")
    bmm_case_entry = {
        "case_id": 1,
        "input_shape": [[4, 128, 64], [4, 64, 256]],
        "dtype": "float16",
    }
    rep = RepresentativeCase.from_case_entry(bmm_case_entry)
    assert rep.input_dtypes == ("float16", "float16")
    bmm_dims = {
        k: v
        for k, v in extract_schema_driven_case_dims(bmm_case_entry).items()
        if isinstance(v, int) and not isinstance(v, bool)
    }
    pkt = analyzer.build_universal_blueprint(bmm_sem, bmm_dims, representative_case=rep)
    assert pkt.roofline.resolved_hir_dims == {"M": 128, "N": 256, "K": 64, "B": 4, "E": 1}
    assert pkt.roofline.cube_flops == 2 * 4 * 128 * 256 * 64
    assert pkt.roofline.min_gm_bytes == 2 * (4 * 128 * 64 + 4 * 64 * 256 + 4 * 128 * 256)

    bmm_trans = RepresentativeCase.from_case_entry(
        {
            "case_id": 2,
            "input_shape": [[4, 64, 128], [4, 256, 64]],
            "dtype": ["bfloat16", "bfloat16"],
            "attrs": {"transpose_a": True, "transpose_b": True},
        }
    )
    pkt_t = analyzer.build_universal_blueprint(bmm_sem, bmm_dims, representative_case=bmm_trans)
    assert pkt_t.roofline.resolved_hir_dims == {"M": 128, "N": 256, "K": 64, "B": 4, "E": 1}

    conv_src = dedent(
        """
        import torch.nn.functional as F

        def conv2d_op(x, w):
            return F.conv2d(x, w, stride=1, padding=1)
        """
    )
    conv_sem = analyzer.analyze_source(conv_src, entry_name="conv2d_op")
    conv_case = {
        "case_id": 1,
        "input_shape": [[2, 32, 16, 16], [64, 32, 3, 3]],
        "dtype": ["float16", "float16"],
        "attrs": {"strides": [1, 1], "pads": [1, 1, 1, 1], "dilations": [1, 1]},
    }
    conv_rep = RepresentativeCase.from_case_entry(conv_case)
    conv_dims = {
        k: v
        for k, v in extract_schema_driven_case_dims(conv_case).items()
        if isinstance(v, int) and not isinstance(v, bool)
    }
    conv_pkt = analyzer.build_universal_blueprint(
        conv_sem,
        conv_dims,
        representative_case=conv_rep,
    )
    assert conv_pkt.roofline.resolved_hir_dims == {
        "M": 256,
        "N": 64,
        "K": 32 * 9,
        "B": 2,
        "E": 1,
    }
    assert conv_pkt.roofline.cube_flops == 2 * 2 * 256 * 64 * (32 * 9)

    optional_and_nested = RepresentativeCase.from_case_entry(
        {
            "case_id": 3,
            "input_shape": [[8, 16], None, [[16, 32], [32, 32]]],
            "dtype": ["float16", "float16", ["bfloat16", "float32"]],
        },
        proto={
            "outputs": [
                {"name": "y", "dtype": ["int8"]},
                {"name": "scale", "dtype": ["float32"]},
            ]
        },
    )
    assert optional_and_nested.input_shapes == ((8, 16), (16, 32), (32, 32))
    assert optional_and_nested.input_dtypes == ("float16", "bfloat16", "float32")
    assert optional_and_nested.output_dtypes == ("int8", "float32")


def test_schema_dims_supports_all_dtypes_and_avoids_spurious_vm_dims() -> None:
    for dtype_name in (
        "int8",
        "uint8",
        "int16",
        "int32",
        "int64",
        "bool",
        "float16",
        "bfloat16",
        "float32",
        "float64",
        "float8_e4m3fn",
        "float8_e5m2",
    ):
        dims = extract_schema_driven_case_dims(
            {"case_id": 1, "input_shape": [[64, 128], [128, 256]], "dtype": [dtype_name, dtype_name]}
        )
        assert dims["tl_dtype"] == dtype_name
        assert "prog_len" not in dims
        assert "num_regs" not in dims
        assert "w_dim" not in dims

    vm_dims = extract_schema_driven_case_dims(
        {"case_id": 1, "input_shape": [[4096, 4], [64, 1024]], "dtype": ["int32", "bfloat16"]},
        proto_inputs=[{"name": "program"}, {"name": "init_regs"}],
    )
    assert vm_dims["tl_dtype"] == "bfloat16"
    assert vm_dims["prog_len"] == 4096
    assert vm_dims["num_regs"] == 64
    assert vm_dims["w_dim"] == 1024


def test_golden_contract_extraction_parambuilder_and_stage0_guard(tmp_path: Path) -> None:
    golden = dedent(
        """
        from typing import List, Optional
        import torch

        def sample_op(
            x: torch.Tensor,
            bias: Optional[torch.Tensor] = None,
            experts: List[torch.Tensor] = (),
            scale: float = 1.0,
        ) -> torch.Tensor:
            return x * scale
        """
    )
    contract = extract_golden_entrypoint_contract(
        golden,
        "sample_op",
        proto={
            "description": "Scaled sample operator",
            "formula": "y = x * scale",
            "outputs": [{"name": "y", "dtype": ["float16"]}],
        },
        desc_text="# SampleOp\nComputes scaled activation with optional bias.",
    )
    assert contract.signature_line == (
        "def sample_op(x: torch.Tensor, bias: Optional[torch.Tensor] = None, "
        "experts: List[torch.Tensor] = (), scale: float = 1.0) -> torch.Tensor:"
    )
    assert "Scaled sample operator" in contract.math_summary
    assert "EXACT GOLDEN ENTRYPOINT ABI" in format_golden_entrypoint_prompt_block(contract)

    ns: dict[str, object] = {}
    exec(
        "from typing import List, Optional\nimport torch\n"
        + contract.signature_line
        + "\n    return x\n",
        ns,
    )
    fn = ns["sample_op"]
    assert callable(fn)

    class _FakeCase:
        input_shapes = [[4, 8], [8], [[8, 8], [8, 8]]]
        attrs = {"scale": 0.5}

    import torch

    bound = ParamBuilder().build_call_params(
        fn,
        _FakeCase(),
        [torch.zeros(4, 8), torch.ones(8), [torch.zeros(8, 8), torch.ones(8, 8)]],
    )
    assert set(bound) == {"x", "bias", "experts", "scale"}
    assert isinstance(bound["x"], torch.Tensor)
    assert isinstance(bound["bias"], torch.Tensor)
    assert isinstance(bound["experts"], list) and len(bound["experts"]) == 2
    assert bound["scale"] == 0.5

    bad_untyped = dedent(
        """
        def sample_op(x, bias=None, experts=(), scale=1.0):
            return x
        kernel = sample_op
        """
    )
    err = validate_entrypoint_signature_compatibility(bad_untyped, contract)
    assert "Tensor annotation" in err

    bad_order = dedent(
        """
        from typing import List, Optional
        import torch

        def sample_op(
            bias: Optional[torch.Tensor] = None,
            x: torch.Tensor = None,
            experts: List[torch.Tensor] = (),
            scale: float = 1.0,
        ):
            return x
        kernel = sample_op
        """
    )
    assert "parameter signature mismatch" in validate_entrypoint_signature_compatibility(
        bad_order,
        contract,
    )

    spec = RegimeTargetSpec(
        regime_tag="vec",
        builder_candidates=("_build_sample_op_prim_func",),
        sample_builder_kwargs={"M": 4, "D": 8},
        representative_hir_dims={"M": 4, "D": 8},
        config_selector_fn="_select_sample_op_config",
    )
    validator = build_stage0_static_validator(
        (spec,),
        entrypoint="sample_op",
        worktree=tmp_path,
        env_file=None,
        entrypoint_contract=contract,
    )
    passed, l0_err = validator(bad_untyped, "tilelang_dsl")
    assert passed is False
    assert "Tensor annotation" in l0_err


def test_hybrid_trackb_stage0_validation_and_seed_synthesis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert _canonical_seed_track("hybrid_trackb") == ("hybrid_trackb", "cce_cpp20")

    hybrid_module = dedent(
        '''
        import torch

        OP_TRACKB_CU = """
        #include <tl_templates/ascend/kernel_entry.h>
        __global__ void k() {}
        """

        def _render_trackb_tu(**dims: object) -> str:
            return str(dims.get("trackb_src", "")) + f"// M={dims['M']}\\n"

        def _select_gather_op_config(M: int = 64, D: int = 128) -> dict[str, object]:
            return {"pass_configs": {}, "compile_flags": []}

        def _build_gather_op_prim_func(M: int = 64, D: int = 128):
            import tilelang.ascend.language as T

            @T.prim_func
            def main(X: T.Tensor((M, D), "float16")):
                pass

            return main

        def _get_kernel(M: int = 64, D: int = 128):
            import tilelang
            prog = _build_gather_op_prim_func(M, D)
            return tilelang.compile(prog, target="ascend"), OP_TRACKB_CU

        def gather_op(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
            return _get_kernel(int(x.shape[0]), int(x.shape[1]))[0](x, index)

        kernel = gather_op
        '''
    )
    spec = RegimeTargetSpec(
        regime_tag="coupled_cv",
        builder_candidates=("_build_gather_op_prim_func",),
        sample_builder_kwargs={"M": 64, "D": 128},
        representative_hir_dims={"M": 64, "D": 128},
        config_selector_fn="_select_gather_op_config",
        runtime_backend="hybrid_trackb",
        trackb_symbols=("OP_TRACKB_CU",),
    )

    tb_evals: list[str] = []
    ta_static: list[str] = []
    monkeypatch.setattr(
        marathon,
        "evaluate_ascendc_candidate",
        lambda src, **kw: tb_evals.append(src) or StageAResult(passed=True, prim_func=None, ledger=None),
    )
    monkeypatch.setattr(
        marathon,
        "run_stage_b_ascendc_compile",
        lambda src, **kw: StageBResult(passed=True),
    )
    monkeypatch.setattr(
        marathon,
        "run_stage_a_static",
        lambda src, **kw: ta_static.append(str(kw.get("factory_name"))) or StageAResult(passed=True, prim_func=None, ledger=None),
    )
    monkeypatch.setattr(
        marathon,
        "run_l1_compile_check",
        lambda src, **kw: StageBResult(passed=True),
    )

    contract = extract_golden_entrypoint_contract(
        "import torch\ndef gather_op(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:\n    return torch.gather(x, 0, index)\n",
        "gather_op",
    )
    validator = build_stage0_static_validator(
        (spec,),
        entrypoint="gather_op",
        worktree=tmp_path,
        env_file=None,
        entrypoint_contract=contract,
    )
    passed, err = validator(hybrid_module, "hybrid_trackb")
    assert passed is True, err
    assert len(tb_evals) == 1
    assert ta_static == ["_build_gather_op_prim_func"]

    forbidden_hybrid = hybrid_module + "\n_FORBIDDEN = 'TPipe'\n"
    passed_bad, err_bad = validator(forbidden_hybrid, "hybrid_trackb")
    assert passed_bad is False
    assert "TPipe=" in err_bad


def test_multi_regime_orthogonal_seed_abi_and_skip_oversized_attrs() -> None:
    analyzer = PyTorchSemanticAnalyzer()
    golden = dedent(
        """
        import torch

        def my_op(x: torch.Tensor, group_list=None) -> torch.Tensor:
            return torch.exp(x)
        """
    )
    sem = analyzer.analyze_source(golden, entry_name="my_op")
    cases = [
        {
            "case_id": 1,
            "input_shape": [[32, 64]],
            "dtype": ["float16"],
            "attrs": {"group_list": list(range(1, 20))},
        },
        {
            "case_id": 2,
            "input_shape": [[4096, 4096]],
            "dtype": ["float16"],
            "attrs": {"group_list": [1024, 2048, 3072, 4096]},
        },
    ]
    ratios = tuple(analyzer.compute_case_ratios(sem, c) for c in cases)
    regimes = build_semantic_regime_specs(
        operator_name="my_op",
        entrypoint="my_op",
        cases=cases,
        proto={"inputs": [{"name": "x"}], "outputs": [{"name": "y"}]},
        base_axes=sem.base_axes,
        has_reduction=sem.has_reduction,
        case_ratios=ratios,
        runtime_backend="tilelang_dsl",
    )
    assert len(regimes) == 2
    sym_sets = [set(r.editable_symbols) for r in regimes]
    assert sym_sets[0].isdisjoint(sym_sets[1])
    for regime in regimes:
        assert regime.config_selector_fn == f"_select_my_op_{regime.regime_tag}_config"
        assert regime.builder_candidates[0] == f"_build_my_op_{regime.regime_tag}_prim_func"
    seed_params = dict(_seed_parameter_specs(regimes))
    assert "group_list" in seed_params and len(seed_params["group_list"]) == 4


def test_pure_golden_registry_ast_check_and_marathon_isolation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks_root = tmp_path / "cann-bench" / "tasks"
    op_dir = tasks_root / "level4" / "mla"
    op_dir.mkdir(parents=True)
    (op_dir / "golden.py").write_text(
        dedent(
            """
            import torch

            def mla(q_nope: torch.Tensor, q_rope: torch.Tensor, k_nope: torch.Tensor, k_rope: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                scores = torch.matmul(q_nope, k_nope.transpose(-1, -2))
                return torch.matmul(torch.softmax(scores, dim=-1), v)
            """
        ),
        encoding="utf-8",
    )
    (op_dir / "proto.yaml").write_text(
        dedent(
            """
            operator:
              name: mla
              inputs:
              - name: q_nope
              - name: q_rope
              - name: k_nope
              - name: k_rope
              - name: v
              outputs:
              - name: y
                dtype: [float16]
            """
        ),
        encoding="utf-8",
    )
    (op_dir / "cases.yaml").write_text(
        dedent(
            """
            cases:
            - case_id: 1
              input_shape: [[1, 8, 16, 128], [1, 8, 16, 64], [1, 32, 1, 128], [1, 32, 1, 64], [1, 32, 1, 128]]
              dtype: [float16, float16, float16, float16, float16]
            - case_id: 7
              input_shape: [[1, 1, 16, 128], [1, 1, 16, 64], [1, 2048, 1, 128], [1, 2048, 1, 64], [1, 2048, 1, 128]]
              dtype: [float16, float16, float16, float16, float16]
            """
        ),
        encoding="utf-8",
    )

    worktree = tmp_path / "repo"
    incumbent_file = worktree / "examples" / "ascend950pr_ops" / "mla" / "kernels.py"
    incumbent_file.parent.mkdir(parents=True)
    incumbent_file.write_text("def mla(*args):\n    pass\n", encoding="utf-8")

    default_reg = CannbenchOpsRegistry(tasks_root=tasks_root, source_root=worktree)
    regimes_missing_symbols = default_reg.get_regime_specs("mla")
    assert {r.regime_tag for r in regimes_missing_symbols} != {
        r.regime_tag for r in DEFAULT_REGIME_SPECS["mla"]
    }

    pure_reg = CannbenchOpsRegistry(tasks_root=tasks_root, source_root=worktree, pure_golden=True)
    assert pure_reg.get_case_regime_tag("mla", 1) is not None
    pure_anchors = pure_reg.get_anchors("mla")
    assert all(a.regime_tag not in {"split_kv_decode", "bnsd_gather", "prefill_compute"} for a in pure_anchors)

    parser = build_marathon_arg_parser()
    cold_args = parser.parse_args(["run", "--operator", "mla", "--cold-start", "--experience-lake-dir", "/tmp/lake"])
    assert cold_args.cold_start is True
    assert cold_args.pure_golden is None
    assert cold_args.experience_lake_dir == "/tmp/lake"
    no_pg_args = parser.parse_args(["run", "--operator", "mla", "--cold-start", "--no-pure-golden"])
    assert no_pg_args.pure_golden is False

    captured: dict[str, object] = {}

    class _CapturingRunner:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

        async def run(self, *, resume: bool) -> object:
            target_p = Path(str(captured["target_module_path"]))
            target_p.parent.mkdir(parents=True, exist_ok=True)
            target_p.write_text("kernel = object()\n", encoding="utf-8")
            return _StubReport()

    class _StubReport:
        def to_dict(self) -> dict[str, object]:
            return {"status": "ok"}

    monkeypatch.setattr(marathon, "_resolve_worktree", lambda _cfg: (worktree, worktree))
    monkeypatch.setattr(marathon, "_ensure_sys_paths", lambda _w: None)
    monkeypatch.setattr(marathon, "_configure_marathon_logging", lambda _p: None)
    monkeypatch.setattr(marathon, "ensure_marathon_env_file", lambda **_kw: worktree / "env.sh")
    monkeypatch.setattr(marathon, "configure_marathon_process_env", lambda **_kw: {"TILEFOUNDRY_NPU_CANNBENCH_REPO": str(tasks_root.parent)})
    monkeypatch.setattr(marathon, "create_llm_backend_from_env", lambda **_kw: MockLLMBackend())
    monkeypatch.setattr(marathon, "build_resilient_l2_evaluator", lambda **_kw: (lambda c, a: {}))
    monkeypatch.setattr(marathon, "build_prewarmed_l3_runner", lambda **_kw: (lambda _p: None))
    monkeypatch.setattr(marathon, "resolve_default_experience_lake_dir", lambda **_kw: None)
    monkeypatch.setattr(marathon, "ExtremeCampaignRunner", _CapturingRunner)

    real_read_text = Path.read_text

    def _guarded_read(self: Path, *args: object, **kwargs: object) -> str:
        if "ascend950pr_ops" in str(self):
            raise AssertionError(f"pure_golden must never read {self}")
        return real_read_text(self, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", _guarded_read)
    state_dir = worktree / ".tilefoundry-marathon" / "state"
    rc = run_operator_marathon(
        MarathonConfig(
            operator_name="mla",
            state_dir=state_dir,
            auto_bootstrap_seed=False,
            salvage_beam_candidates=False,
            pure_golden=True,
        )
    )
    assert rc == 0
    assert captured["target_module_path"] == (state_dir / "pure_golden_ops" / "mla" / "kernels.py").resolve()
    assert captured["authored_hir_path"] is None
    assert captured["scheduled_hir_path"] is None

    ext_state_dir = tmp_path / "external_state"
    captured.clear()
    ext_rc = run_operator_marathon(
        MarathonConfig(
            operator_name="mla",
            state_dir=ext_state_dir,
            auto_bootstrap_seed=False,
            salvage_beam_candidates=False,
            pure_golden=True,
        )
    )
    assert ext_rc == 0
    ext_target = Path(str(captured["target_module_path"]))
    assert ext_target.is_relative_to(worktree / ".tilefoundry-marathon" / "pure_golden_ops")
    assert "ascend950pr_ops" not in str(ext_target)
    assert (ext_state_dir / "pure_golden_ops" / "mla" / "kernels.py").is_file()


@pytest.mark.skipif(
    _real_cannbench_repo() is None,
    reason="real cann-bench tasks repository is not available",
)
def test_real_cannbench_five_operators_pure_golden_contracts_and_blueprints() -> None:
    repo = _real_cannbench_repo()
    assert repo is not None
    registry = CannbenchOpsRegistry(tasks_root=repo, pure_golden=True)
    analyzer = PyTorchSemanticAnalyzer()

    for op_id in (
        "level1/exp",
        "level2/softmax",
        "level3/grouped_matmul",
        "level4/gqa",
        "level3/add_rms_norm_dynamic_quant",
    ):
        spec = registry.resolve(op_id)
        golden_src = spec.golden_path.read_text(encoding="utf-8")
        desc_txt = spec.desc_path.read_text(encoding="utf-8") if spec.desc_path.is_file() else ""
        contract = extract_golden_entrypoint_contract(
            golden_src,
            spec.entrypoint,
            proto=spec.proto,
            desc_text=desc_txt,
        )
        assert contract.parameters
        assert any(p.is_tensor for p in contract.parameters)
        assert validate_entrypoint_signature_compatibility(golden_src, contract) == ""

        sem = analyzer.analyze_source(golden_src, entry_name=spec.entrypoint, proto=spec.proto)
        regimes = registry.get_regime_specs(op_id)
        assert len(regimes) >= 2
        if len(regimes) > 1:
            sym_sets = [set(r.editable_symbols) for r in regimes]
            for i in range(len(sym_sets)):
                for j in range(i + 1, len(sym_sets)):
                    assert sym_sets[i].isdisjoint(sym_sets[j]), (op_id, regimes[i].regime_tag)
        for regime in regimes:
            rep = RepresentativeCase.from_case_entry(
                regime.representative_case_entry,
                proto=spec.proto,
            )
            pkt = analyzer.build_universal_blueprint(
                sem,
                regime.representative_anchor_dims or regime.representative_hir_dims,
                representative_case=rep,
            )
            assert pkt.roofline.ideal_roofline_us > 0.0

    quant_spec = registry.resolve("level3/add_rms_norm_dynamic_quant")
    quant_case = quant_spec.cases[0]
    quant_rep = RepresentativeCase.from_case_entry(quant_case, proto=quant_spec.proto)
    assert len(quant_rep.output_dtypes) == 3
    assert set(quant_rep.output_dtypes) == {"int8", "float16", "float32"}


def test_seed_synthesizer_preserves_hybrid_trackb_and_typed_signature(
    tmp_path: Path,
) -> None:
    golden = dedent(
        """
        import torch

        def attn_like(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            scores = torch.matmul(q, k.transpose(-1, -2))
            return torch.matmul(torch.softmax(scores, dim=-1), v)
        """
    )
    cases = [
        {
            "case_id": 1,
            "input_shape": [[1, 32, 16, 128], [1, 32, 16, 128], [1, 32, 16, 128]],
            "dtype": ["float16", "float16", "float16"],
        },
        {
            "case_id": 2,
            "input_shape": [[2, 512, 16, 128], [2, 512, 16, 128], [2, 512, 16, 128]],
            "dtype": ["float16", "float16", "float16"],
        },
    ]
    hybrid_seed = dedent(
        '''
        import torch

        ATTN_LIKE_TRACKB_CU = """
        #include <tl_templates/ascend/kernel_entry.h>
        __global__ void attn_k() {}
        """

        def attn_like(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("NPU")

        kernel = attn_like
        '''
    )
    backend = _ScriptedBackend([hybrid_seed])
    proto = {
        "inputs": [{"name": "q"}, {"name": "k"}, {"name": "v"}],
        "outputs": [{"name": "y", "dtype": ["float16"]}],
    }

    import hashlib

    outcome = asyncio.run(
        prepare_operator_seed_or_takeover(
            operator_name="attn_like",
            entrypoint="attn_like",
            pytorch_golden_source=golden,
            proto=proto,
            cases=cases,
            legacy_regime_specs=None,
            llm_backend=backend,
            l0_l1_validator_factory=lambda _rs: (lambda _s, _t: (True, "")),
            oracle_accuracy_validator=lambda s: OracleAccuracyResult(
                passed=True,
                candidate_sha256=hashlib.sha256(s.encode("utf-8")).hexdigest(),
                l3_run_id="run-1",
                l3_receipt_path=str(tmp_path / "receipt.json"),
                l3_state="passed",
            ),
            force_cold_start=True,
        )
    )
    assert outcome.mode == "cold_start"
    assert outcome.initial_track == "hybrid_trackb"
    assert all(r.runtime_backend == "hybrid_trackb" for r in outcome.regime_specs)
    assert "def attn_like(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:" in backend.requests[0].system_prompt
    assert "def _select_attn_like_" in backend.requests[0].system_prompt
    assert "hybrid_trackb" in backend.requests[0].system_prompt


def test_stage0_trackb_allows_ascendc_primitives_without_facet_locks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify Stage-0 Track-B validates hardware legality without incremental facet locks."""
    regime = RegimeTargetSpec(
        regime_tag="reduce_tail",
        description="tail reduction",
        builder_candidates=("_build_sm_prim_func",),
        config_selector_fn="_select_sm_config",
        representative_hir_dims={"total_elements": 128},
        sample_builder_kwargs={"total_elements": 128},
        editable_symbols=("SM_TRACKB_CU", "_select_sm_config"),
        runtime_backend="ascendc_trackb",
        trackb_symbols=("SM_TRACKB_CU",),
    )
    compiled_sources: list[str] = []

    def _fake_compile(cpp_source: str, **_kw: object) -> StageBResult:
        compiled_sources.append(cpp_source)
        return StageBResult(passed=True)

    monkeypatch.setattr(marathon, "run_stage_b_ascendc_compile", _fake_compile)
    validator = build_stage0_static_validator(
        [regime],
        entrypoint="sm_op",
        worktree=tmp_path,
        env_file=None,
    )
    seed_src = dedent(
        '''
        import torch

        SM_TRACKB_CU = """
        #include <tl_templates/ascend/kernel_entry.h>
        __launch_bounds__(512)
        __global__ void sm_k(__gm__ float* x, __gm__ float* y) {
            __simd_vf__ void (*fn)() = nullptr;
            asc_copy_gm2ub_align(nullptr, nullptr, 64);
            asc_sync_notify(0, 1);
            asc_sync_wait(0, 1);
        }
        """

        def _select_sm_config(**kwargs) -> dict[str, object]:
            return {"pass_configs": {}, "compile_flags": []}

        def _build_sm_prim_func(**kwargs) -> str:
            return SM_TRACKB_CU

        def _render_trackb_tu(trackb_src: str = SM_TRACKB_CU, **dims: object) -> str:
            return trackb_src + "\\nvoid launch() { sm_k<<<1, 1, 1>>>(nullptr, nullptr); }\\n"

        def _get_kernel(**kwargs) -> object:
            return object()

        def sm_op(x: torch.Tensor) -> torch.Tensor:
            return x

        kernel = sm_op
        '''
    )
    passed, err = validator(seed_src, "cce_trackb")
    assert passed is True, err
    assert len(compiled_sources) == 1

    bad_src = seed_src.replace(
        "asc_sync_wait(0, 1);",
        "asc_sync_wait(0, 1);\n            using namespace AscendC;",
    )
    bad_ok, bad_err = validator(bad_src, "cce_trackb")
    assert bad_ok is False
    assert "AscendC" in bad_err


def test_trackb_static_cpp_string_discovery_and_extraction() -> None:
    """Verify dedent, string concatenation, strip, and vector/simd markers are resolved."""
    src_dedent = dedent(
        '''
        from textwrap import dedent
        OP_A_TRACKB_CU = dedent("""
            #include <tl_templates/ascend/common.h>
            __global__ __vector__ void k() {}
        """).strip()
        '''
    )
    syms = discover_module_trackb_symbols(src_dedent)
    assert syms == ("OP_A_TRACKB_CU",)
    extracted = extract_trackb_cpp_source(src_dedent, syms)
    assert extracted is not None
    assert extracted[0] == "OP_A_TRACKB_CU"
    assert "__global__ __vector__" in extracted[1]

    src_concat = dedent(
        '''
        import textwrap
        OP_B_TRACKB_CU = (
            textwrap.dedent("""
                #include "custom.h"
                __simd_vf__ void vf() {}
            """)
            + "\\n__global__ __aicore__ __vector__ void run() {}\\n"
        )
        '''
    )
    syms_b = discover_module_trackb_symbols(src_concat)
    assert syms_b == ("OP_B_TRACKB_CU",)
    extracted_b = extract_trackb_cpp_source(src_concat, syms_b)
    assert extracted_b is not None
    assert "__simd_vf__" in extracted_b[1]
    assert "__vector__" in extracted_b[1]


def test_trackb_entrypoint_purity_and_stage0_contract_with_compile_ascend() -> None:
    """Verify Track-B purity skips tilelang.compile while guarding fallbacks and missing TU."""
    src_trackb = dedent(
        '''
        from textwrap import dedent
        import torch
        from tilelang.contrib.bisheng import compile_ascend

        VEC_OP_TRACKB_CU = dedent("""
            #include <tl_templates/ascend/common.h>
            __global__ __aicore__ __vector__ void k(__gm__ float* x) {}
        """)

        def _get_kernel() -> object:
            return compile_ascend(VEC_OP_TRACKB_CU, target_format="so")

        def vec_op(x: torch.Tensor) -> torch.Tensor:
            if x.device.type != "npu":
                return x * 2
            _get_kernel()
            return x

        kernel = vec_op
        '''
    )
    assert check_npu_entrypoint_purity(src_trackb, "vec_op", track="cce_trackb") == []
    assert _stage0_entrypoint_contract_error(src_trackb, "vec_op", "hybrid_trackb") == ""

    src_fallback = src_trackb.replace(
        "_get_kernel()\n    return x",
        "_get_kernel()\n    return torch.softmax(x, dim=-1)",
    )
    errs_fb = check_npu_entrypoint_purity(src_fallback, "vec_op", track="cce_trackb")
    assert any("npu_host_fallback_forbidden" in e for e in errs_fb)

    src_missing = dedent(
        '''
        import torch

        def vec_op(x: torch.Tensor) -> torch.Tensor:
            return x

        kernel = vec_op
        '''
    )
    errs_miss = check_npu_entrypoint_purity(src_missing, "vec_op", track="cce_trackb")
    assert any("missing_trackb_dispatch" in e for e in errs_miss)


def test_seed_module_abi_trackb_prompt_includes_compile_ascend_recipe() -> None:
    """Verify Track-B prompt specifies raw TU literal, __vector__ attr, and compile_ascend."""
    prompt = _seed_module_abi(
        "softmax",
        (("d_dim", 128),),
        get_dialect_surface("cce_cpp20"),
        "cce_trackb",
    )
    assert "compile_ascend" in prompt
    assert "PURE_SIMDVF_UBUF_FLAGS" in prompt
    assert "__global__ __aicore__ __vector__" in prompt
    assert '__mix__(1, 2)' in prompt
    assert 'SOFTMAX_TRACKB_CU = r"""' in prompt
    assert "torch.npu.current_stream().npu_stream" in prompt


def test_entrypoint_purity_allows_non_npu_guards_self_calls_and_ctypes_lib() -> None:
    """Verify purity skips CPU guards, helpers, and ctypes handles while blocking NPU fallbacks."""
    clean_src = dedent(
        '''
        import torch
        import torch.nn.functional as F
        from tilelang.contrib.bisheng import compile_ascend

        RMS_NORM_TRACKB_CU = r"""
        #include <tl_templates/ascend/common.h>
        __global__ __aicore__ __vector__ void k(__gm__ float* x) {}
        """

        def _get_kernel(lib: object) -> object:
            return lib

        def _launch_helper(x: torch.Tensor, lib: object) -> torch.Tensor:
            lib.rms_norm(x.data_ptr())
            return x

        def rms_norm(x: torch.Tensor, recurse: bool = False) -> torch.Tensor:
            if not x.is_npu:
                return F.softmax(x, dim=-1)
            if x.device.type == "cpu":
                return F.rms_norm(x, x.shape[-1:])
            if x.device.type == "npu":
                if recurse:
                    return _launch_helper(x, _get_kernel(None))
                return rms_norm(x.contiguous(), recurse=True)
            else:
                return F.softmax(x, dim=-1)

        kernel = rms_norm
        '''
    )
    assert check_npu_entrypoint_purity(clean_src, "rms_norm", track="cce_trackb") == []

    bad_else = dedent(
        '''
        import torch
        import torch.nn.functional as F

        RMS_NORM_TRACKB_CU = r"""
        #include <tl_templates/ascend/common.h>
        __global__ __aicore__ __vector__ void k(__gm__ float* x) {}
        """

        def rms_norm(x: torch.Tensor) -> torch.Tensor:
            if not x.is_npu:
                return x
            return F.rms_norm(x, x.shape[-1:])

        kernel = rms_norm
        '''
    )
    errs = check_npu_entrypoint_purity(bad_else, "rms_norm", track="cce_trackb")
    assert any("npu_host_fallback_forbidden:rms_norm->rms_norm" in e for e in errs)


def test_l3_failure_detail_preserves_compiler_and_comparison_diagnostics() -> None:
    """Verify failed L3 status formatting preserves compiler stderr and comparison metrics."""
    row = {"failures": ["NONZERO_EXIT", "OBSERVATION_MISMATCH: exitCodeZero"]}
    obs = {"error": "failed"}
    evidence = {
        "failure": {
            "case": "1",
            "stage": "kernel",
            "exceptionType": "RuntimeError",
            "detail": (
                "RuntimeError: Ascend compilation failed. Command: bisheng ... "
                "Compiler output: ld.lld: error: undefined symbol: expf"
            ),
            "comparisons": [
                {"name": "y", "passed": False, "max_abs_diff": 0.25, "atol": 1e-3}
            ],
        }
    }
    msg = _format_l3_failure_detail(row, obs, evidence)
    assert "case=1 stage=kernel type=RuntimeError" in msg
    assert "ld.lld: error: undefined symbol: expf" in msg
    assert "max_abs_diff=0.25" in msg
    assert "atol=0.001" in msg


def test_stage_b_ascendc_verify_shared_link_catches_linker_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify optional shared-library link verification catches ld.lld undefined symbol errors."""
    cann_root = tmp_path / "cann"
    (cann_root / "include").mkdir(parents=True)
    bisheng = cann_root / "bin" / "bisheng"
    bisheng.parent.mkdir(parents=True)
    bisheng.write_text("#!/bin/sh\n", encoding="utf-8")

    tu = (
        '#include <tl_templates/ascend/common.h>\n'
        '__global__ __aicore__ __vector__ void k(__gm__ float* x) {}\n'
        'extern "C" int launch_k(void* s, void* x) { k<<<1, nullptr, s>>>((__gm__ float*)x); return 0; }\n'
    )
    monkeypatch.setattr(
        gate_funnel, "_verify_object_has_instantiated_kernel", lambda *_a, **_kw: (True, "")
    )
    monkeypatch.setattr(gate_funnel, "_disassemble_object", lambda *_a, **_kw: "")

    commands: list[list[str]] = []

    def _run_c_pass_link_fail(cmd, **_kw):
        cmd_list = [str(x) for x in cmd]
        if cmd_list[0] != str(bisheng):
            return subprocess.CompletedProcess(cmd_list, 0, stdout="", stderr="")
        commands.append(cmd_list)
        out_path = Path(cmd_list[cmd_list.index("-o") + 1])
        if "-shared" in cmd_list:
            return subprocess.CompletedProcess(
                cmd_list, 1, stdout="", stderr="ld.lld: error: undefined symbol: expf\n"
            )
        out_path.write_bytes(b"\x7fELF")
        return subprocess.CompletedProcess(cmd_list, 0, stdout="", stderr="")

    monkeypatch.setattr(gate_funnel.subprocess, "run", _run_c_pass_link_fail)

    res_no_link = run_stage_b_ascendc_compile(
        tu, bisheng_bin=bisheng, cann_root=cann_root, verify_shared_link=False
    )
    assert res_no_link.passed is True
    assert len(commands) == 1

    commands.clear()
    res_fail_link = run_stage_b_ascendc_compile(
        tu, bisheng_bin=bisheng, cann_root=cann_root, verify_shared_link=True
    )
    assert res_fail_link.passed is False
    assert "stage_b_ascendc_shared_link_failed" in res_fail_link.error_message
    assert "undefined symbol: expf" in res_fail_link.error_message
    assert len(commands) == 2

    def _run_both_pass(cmd, **_kw):
        cmd_list = [str(x) for x in cmd]
        out_path = Path(cmd_list[cmd_list.index("-o") + 1])
        out_path.write_bytes(b"\x7fELF")
        return subprocess.CompletedProcess(cmd_list, 0, stdout="", stderr="")

    monkeypatch.setattr(gate_funnel.subprocess, "run", _run_both_pass)
    res_ok = run_stage_b_ascendc_compile(
        tu, bisheng_bin=bisheng, cann_root=cann_root, verify_shared_link=True
    )
    assert res_ok.passed is True


def test_stage0_oracle_validator_retries_transient_blocked_env(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify transient BLOCKED_ENV resource-busy results retry with backoff before returning."""
    results = [
        L3AcceptanceResult(
            passed=False,
            failure_kind="infrastructure",
            error_message="L3 gate recorded status BLOCKED_ENV: ['resource busy: build:/repo/build']",
        ),
        L3AcceptanceResult(
            passed=False,
            failure_kind="infrastructure",
            error_message="L3 gate recorded status BLOCKED_ENV: ['resource busy: build:/repo/build']",
        ),
        L3AcceptanceResult(
            passed=True,
            run_id="run-ok",
            receipt_path=str(tmp_path / "receipt.json"),
            candidate_sha256="abc",
            state="FINISHED",
        ),
    ]
    calls: list[str] = []
    sleeps: list[float] = []

    def _fake_l3(**kwargs: object) -> L3AcceptanceResult:
        calls.append(str(kwargs.get("operator_name")))
        return results[len(calls) - 1]

    monkeypatch.setattr(marathon, "run_l3_full_contract_via_agent_gates", _fake_l3)
    monkeypatch.setattr(marathon.time, "sleep", lambda sec: sleeps.append(float(sec)))

    oracle = build_stage0_oracle_validator(
        operator_name="level4/gqa",
        staging_dir=tmp_path / "staging",
        worktree=tmp_path,
        device_id="0",
        op_logger=logging.getLogger("test.oracle"),
        blocked_env_max_retries=4,
        blocked_env_backoff_sec=(10.0, 15.0),
    )
    outcome = oracle("kernel = object()\n")
    assert outcome.passed is True
    assert outcome.l3_run_id == "run-ok"
    assert len(calls) == 3
    assert sleeps == [10.0, 15.0]


def test_cce_cpp20_surface_prompt_includes_dav3510_compiler_rules() -> None:
    """Verify CCE C++20 surface reference teaches dav-3510 GM qualifiers, math, and callee rules."""
    rendered = CCE_CPP20_SURFACE.render_prompt_reference()
    assert "__gm__ T*" in rendered
    assert "__builtin_expf" in rendered
    assert "__simt_callee__" in rendered
    assert "MODE_ZEROING" in rendered
    assert 'if x.device.type != "npu":' in rendered


def test_fast_wire_operator_name_normalization_in_batch_and_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify fast batch and profile serialize slashless wire names and restore caller keys."""
    assert _fast_wire_operator_name("level2/softmax") == "softmax"
    assert (
        _fast_wire_operator_name("level3/add_rms_norm_dynamic_quant")
        == "add_rms_norm_dynamic_quant"
    )

    source_root = tmp_path / "repo"
    source_root.mkdir()
    cand = source_root / "cand.py"
    cand.write_text("kernel = object()\n", encoding="utf-8")
    env_file = source_root / "env.sh"
    work_root = tmp_path / "work"
    work_root.mkdir()
    env_file.write_text(
        f"TILEFOUNDRY_NPU_CANNBENCH_REPO={tmp_path / 'cb'}\n"
        f"TILEFOUNDRY_NPU_WORK_ROOT={work_root}\n",
        encoding="utf-8",
    )
    anchor = AnchorCaseSpec("level2/softmax", 1, "reduce_tail", 10.0, 5.0)
    monkeypatch.setattr(surrogate_cannbench, "_resolve_agent_gates_bin", lambda _r: "agent-gates")
    monkeypatch.setattr(surrogate_cannbench, "_agent_gates_workspace_name", lambda _r: "tilefoundry")
    monkeypatch.setattr(surrogate_cannbench, "get_operator_entrypoint", lambda _op: "softmax")
    monkeypatch.setattr(
        surrogate_cannbench, "prewarm_anchor_reference_cache", lambda *_a, **_kw: None
    )
    monkeypatch.setattr(
        surrogate_cannbench, "prewarm_surrogate_tilelang_cache", lambda *_a, **_kw: None
    )

    seen_payloads: list[dict[str, object]] = []

    def _fake_submit(
        *,
        request_id: str,
        request_path: Path,
        response_path: Path,
        mode: str,
        **_kw: object,
    ) -> tuple[dict[str, object], _V2GateExecution]:
        req_payload = json.loads(request_path.read_text(encoding="utf-8"))
        seen_payloads.append(req_payload)
        bindings = {
            row["operator_name"]: {
                c["candidate_id"]: {
                    "module_path": c["module_path"],
                    "sha256": c["sha256"],
                    "post_sha256": c["sha256"],
                }
                for c in ([row] if mode == "profile" else row["candidates"])
            }
            for row in (
                req_payload["msprof_targets"] if mode == "profile" else req_payload["batches"]
            )
        }
        if mode == "batch":
            resp: dict[str, object] = {
                "schema": FAST_RESPONSE_SCHEMA,
                "request_id": request_id,
                "mode": "batch",
                "status": "ok",
                "candidates": bindings,
                "coverage": {
                    "expected_candidates": 1,
                    "evaluated_candidates": 1,
                    "expected_msprof_targets": 0,
                    "observed_msprof_targets": 0,
                },
                "operators": {
                    "softmax": {
                        "c0": {
                            "candidate_id": "c0",
                            "operator_name": "softmax",
                            "all_passed": True,
                            "avg_speedup": 2.0,
                            "min_case_speedup": 2.0,
                            "avg_roofline_eta": 1.0,
                            "surrogate_score": 88.0,
                            "anchors": [
                                {
                                    "case_num": 1,
                                    "regime_tag": "reduce_tail",
                                    "ref_perf_us": 10.0,
                                    "cand_us": 5.0,
                                    "t_hw_us": 5.0,
                                    "speedup": 2.0,
                                    "roofline_eta": 1.0,
                                    "eq3_score": 88.0,
                                    "accuracy_passed": True,
                                }
                            ],
                        }
                    }
                },
            }
        else:
            resp = {
                "schema": FAST_RESPONSE_SCHEMA,
                "request_id": request_id,
                "mode": "profile",
                "status": "ok",
                "candidates": bindings,
                "coverage": {
                    "expected_candidates": 1,
                    "evaluated_candidates": 1,
                    "expected_msprof_targets": 1,
                    "observed_msprof_targets": 1,
                },
                "msprof_telemetry": {
                    "softmax": {"bottleneck_probe": {"1": {"aiv_vec_ratio": 0.72}}}
                },
            }
        return resp, _V2GateExecution(
            execution_id="exec-fast-1",
            report={
                "status": "PASS",
                "gates": [{"gateId": "tilefoundry-npu-device-fast", "observations": {"heldSeconds": 1.0}}],
            },
            verification={"satisfied": True},
            run_id="run-fast-1",
            receipt_path=str(response_path),
            elapsed_sec=1.0,
        )

    monkeypatch.setattr(surrogate_cannbench, "_submit_fast_gate", _fake_submit)
    by_op, run_id, _rcp, _held = run_multi_operator_surrogate_batch_via_agent_gates(
        [("level2/softmax", [("c0", cand)], [anchor])],
        source_root=source_root,
        env_file=env_file,
        prewarm_cache=False,
    )
    assert run_id == "run-fast-1"
    assert seen_payloads[0]["batches"][0]["operator_name"] == "softmax"
    assert "level2/softmax" in by_op
    assert by_op["level2/softmax"]["c0"].surrogate_score == 88.0

    profile = _collect_profile_via_agent_gates(
        "level2/softmax",
        cand,
        [anchor],
        candidate_id="bottleneck_probe",
        source_root=source_root,
        env_file=env_file,
    )
    assert seen_payloads[1]["msprof_targets"][0]["operator_name"] == "softmax"
    assert profile["reduce_tail"]["aiv_vec_ratio"] == 0.72


def test_l3_failure_detail_prefers_node_comparison_message_over_coverage() -> None:
    """Verify node failure message is prioritized ahead of trailing coverage missing_case."""
    row = {"failures": ["NONZERO_EXIT", "INVALID_TEST_COUNT"]}
    obs = {"error": "failed"}
    evidence = {
        "failure": {
            "case": "2",
            "category": "missing_case",
            "stage": "coverage",
            "detail": "test_target_operator_full_cannbench_contract:2 was not completed",
        },
        "nodes": [
            {
                "name": "test_target_operator_full_cannbench_contract",
                "failure": {
                    "case": "1",
                    "category": "numeric",
                    "comparisons": [
                        {
                            "message": (
                                "RuntimeError: rtStreamSynchronize execution failed, "
                                "reason=vector core exception"
                            )
                        }
                    ],
                },
            }
        ],
    }
    msg = _format_l3_failure_detail(row, obs, evidence)
    assert "reason=vector core exception" in msg
    assert "was not completed" not in msg


def test_submit_fast_gate_retries_transient_build_lock_backoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify _submit_fast_gate retries BLOCKED_ENV resource busy build lock before succeeding."""
    req_path = tmp_path / "req.json"
    req_bytes = b'{"schema": "tilefoundry-fast-request/v1"}\n'
    req_path.write_bytes(req_bytes)
    req_sha = hashlib.sha256(req_bytes).hexdigest()
    resp_path = tmp_path / "resp.json"
    resp_payload = {
        "schema": FAST_RESPONSE_SCHEMA,
        "request_id": "req1",
        "request_sha256": req_sha,
        "mode": "batch",
        "status": "ok",
    }
    resp_path.write_text(json.dumps(resp_payload), encoding="utf-8")

    executions = [
        _V2GateExecution(
            report={
                "status": "BLOCKED_ENV",
                "gates": [
                    {
                        "gateId": "tilefoundry-npu-device-fast",
                        "status": "BLOCKED_ENV",
                        "failures": ["resource busy: build:/repo/build (waited 0.0s)"],
                        "observations": {},
                    }
                ],
            },
            verification={"satisfied": False, "failures": ["resource busy: build:/repo/build"]},
        ),
        _V2GateExecution(
            execution_id="exec-fast-ok",
            report={"status": "PASS", "gates": []},
            verification={"satisfied": True},
            run_id="run-fast-ok",
        ),
    ]
    calls: list[str] = []
    sleeps: list[float] = []

    def _fake_v2(**kwargs: object) -> _V2GateExecution:
        calls.append(str(kwargs.get("target")))
        return executions[len(calls) - 1]

    monkeypatch.setattr(surrogate_cannbench, "_run_v2_gate", _fake_v2)
    monkeypatch.setattr(surrogate_cannbench, "_source_identity_matches", lambda *_a: True)
    monkeypatch.setattr(surrogate_cannbench.time, "sleep", lambda sec: sleeps.append(float(sec)))

    raw, final_exec = _submit_fast_gate(
        agent_gates_bin="agent-gates",
        root=tmp_path,
        workspace="tilefoundry",
        request_id="req1",
        request_relative_path="req.json",
        request_path=req_path,
        response_path=resp_path,
        request_sha256=req_sha,
        mode="batch",
        timeout_sec=60.0,
        device_id="1",
    )
    assert final_exec.run_id == "run-fast-ok"
    assert raw["status"] == "ok"
    assert len(calls) == 2
    assert sleeps == [8.0]


def test_prepare_seed_retries_same_candidate_on_environment_blockage(tmp_path: Path) -> None:
    """Verify environment blockage retries same seed without burning LLM attempts."""
    golden = dedent(
        """
        import torch

        def vec_op(x: torch.Tensor) -> torch.Tensor:
            return torch.exp(x)
        """
    )
    proto = {
        "name": "vec_op",
        "inputs": [{"name": "x", "dtype": "float32"}],
        "outputs": [{"name": "y", "dtype": "float32"}],
    }
    cases = [{"case_id": 1, "input_shape": [[64, 128]], "dtype": ["float32"], "attrs": {}}]
    seed_src = dedent(
        """
        import torch
        import tilelang
        import tilelang.ascend.language as T

        def _select_vec_op_config(M: int = 64, D: int = 128) -> dict[str, object]:
            return {"pass_configs": {}, "compile_flags": []}

        def _build_vec_op_prim_func(M: int = 64, D: int = 128):
            @T.prim_func
            def main(X: T.Tensor((M, D), "float32")):
                pass
            return main

        def _get_kernel(M: int = 64, D: int = 128):
            return tilelang.compile(_build_vec_op_prim_func(M, D), target="ascend")

        def vec_op(x: torch.Tensor) -> torch.Tensor:
            if x.device.type != "npu":
                return torch.exp(x)
            return _get_kernel(int(x.shape[0]), int(x.shape[1]))(x)

        kernel = vec_op
        """
    )
    digest = hashlib.sha256(seed_src.strip().encode("utf-8")).hexdigest()
    backend = _ScriptedBackend([seed_src.strip()])
    oracle_calls = 0

    def _blocked_then_ok(module_source: str) -> OracleAccuracyResult:
        nonlocal oracle_calls
        oracle_calls += 1
        assert hashlib.sha256(module_source.encode("utf-8")).hexdigest() == digest
        if oracle_calls <= 2:
            return OracleAccuracyResult(
                passed=False,
                error_message="L3 gate recorded status BLOCKED_ENV: ['resource busy: build:/repo/build']",
                l3_state="FINISHED",
                failure_kind="environment",
            )
        return OracleAccuracyResult(
            passed=True,
            candidate_sha256=digest,
            l3_run_id="run-env-recovered",
            l3_receipt_path=str(tmp_path / "receipt.json"),
            l3_state="FINISHED",
        )

    outcome = asyncio.run(
        prepare_operator_seed_or_takeover(
            operator_name="level1/vec_op",
            entrypoint="vec_op",
            pytorch_golden_source=golden,
            proto=proto,
            cases=cases,
            legacy_regime_specs=None,
            llm_backend=backend,
            l0_l1_validator_factory=lambda _regimes: (lambda _src, _track: (True, "")),
            oracle_accuracy_validator=_blocked_then_ok,
            force_cold_start=True,
            max_seed_attempts=1,
            max_env_retries=3,
        )
    )
    assert outcome.oracle_accuracy_passed is True
    assert outcome.attempts_used == 1
    assert oracle_calls == 3
    assert len(backend.requests) == 1

    backend_exhausted = _ScriptedBackend([seed_src.strip()])
    exhausted_calls = 0

    def _always_blocked(_module_source: str) -> OracleAccuracyResult:
        nonlocal exhausted_calls
        exhausted_calls += 1
        return OracleAccuracyResult(
            passed=False,
            error_message="L3 gate recorded status BLOCKED_ENV: ['resource busy: build:/repo/build']",
            l3_state="FINISHED",
            failure_kind="environment",
        )

    with pytest.raises(RuntimeError, match="blocked by environment"):
        asyncio.run(
            prepare_operator_seed_or_takeover(
                operator_name="level1/vec_op",
                entrypoint="vec_op",
                pytorch_golden_source=golden,
                proto=proto,
                cases=cases,
                legacy_regime_specs=None,
                llm_backend=backend_exhausted,
                l0_l1_validator_factory=lambda _regimes: (lambda _src, _track: (True, "")),
                oracle_accuracy_validator=_always_blocked,
                force_cold_start=True,
                max_seed_attempts=2,
                max_env_retries=2,
            )
        )
    assert exhausted_calls == 3
    assert len(backend_exhausted.requests) == 1


def test_validate_stage0_track_b_omits_simdvf_ubuf_flags_for_simtvf(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify Stage-0 Track-B omits PURE_SIMDVF_UBUF_FLAGS whenever __simt_vf__ appears."""
    recorded_flags: list[list[str]] = []

    def _fake_compile(tu_src: str, **kwargs: object) -> StageBResult:
        _ = tu_src
        raw_flags = kwargs.get("compile_flags")
        assert isinstance(raw_flags, Sequence)
        recorded_flags.append([str(item) for item in raw_flags])
        return StageBResult(passed=True)

    monkeypatch.setattr(marathon, "evaluate_ascendc_candidate", lambda *_a, **_kw: StageAResult(passed=True, prim_func=None, ledger=None))
    monkeypatch.setattr(marathon, "run_stage_b_ascendc_compile", _fake_compile)

    contract = extract_golden_entrypoint_contract(
        "import torch\ndef vec_op(x: torch.Tensor) -> torch.Tensor:\n    return x\n",
        "vec_op",
    )
    spec = RegimeTargetSpec(
        regime_tag="stream",
        builder_candidates=("_build_vec_op_prim_func",),
        sample_builder_kwargs={"M": 64, "D": 128},
        representative_hir_dims={"M": 64, "D": 128},
        config_selector_fn="_select_vec_op_config",
        runtime_backend="ascendc_trackb",
        trackb_symbols=("VEC_OP_TRACKB_CU",),
    )
    template = dedent(
        '''
        import ctypes
        import torch
        from tilelang.contrib.bisheng import compile_ascend

        VEC_OP_TRACKB_CU = r"""
        #include <tl_templates/ascend/common.h>
        __VF_ATTR__ void helper(__gm__ float* x) {}
        extern "C" __global__ __aicore__ __vector__ void vec_op_k(__gm__ float* x) {}
        extern "C" int launch_vec_op(void* stream, void* x) {
            vec_op_k<<<1, nullptr, stream>>>((__gm__ float*)x);
            return 0;
        }
        """

        def _render_trackb_tu(**dims: object) -> str:
            _ = dims
            return VEC_OP_TRACKB_CU

        def _select_vec_op_config(M: int = 64, D: int = 128) -> dict[str, object]:
            return {"compile_flags": []}

        def _build_vec_op_prim_func(M: int = 64, D: int = 128) -> object:
            return VEC_OP_TRACKB_CU

        def _get_kernel(M: int = 64, D: int = 128) -> object:
            compile_ascend(VEC_OP_TRACKB_CU, target_format="so", npu_arch="dav-3510")
            return ctypes.CDLL("dummy.so")

        def vec_op(x: torch.Tensor) -> torch.Tensor:
            if x.device.type != "npu":
                return x
            return _get_kernel()(x)

        kernel = vec_op
        '''
    )
    validator = build_stage0_static_validator(
        (spec,),
        entrypoint="vec_op",
        worktree=tmp_path,
        env_file=None,
        entrypoint_contract=contract,
    )
    ok_simd, err_simd = validator(template.replace("__VF_ATTR__", "__simd_vf__ inline"), "cce_trackb")
    assert ok_simd is True, err_simd
    ok_simt, err_simt = validator(template.replace("__VF_ATTR__", "__simt_vf__"), "cce_trackb")
    assert ok_simt is True, err_simt
    assert len(recorded_flags) == 2
    assert "--cce-disable-vf-stack-reserved-ubuf" in recorded_flags[0]
    assert "--cce-disable-vf-stack-reserved-ubuf" not in recorded_flags[1]


def test_cce_cpp20_surface_warns_against_simtvf_ubuf_flags_and_raw_ub_casts() -> None:
    """Verify CCE C++20 reference forbids PURE_SIMDVF_UBUF_FLAGS and raw __ubuf__ casts in __simt_vf__."""
    rendered = CCE_CPP20_SURFACE.render_prompt_reference()
    assert "PURE_SIMDVF_UBUF_FLAGS" in rendered
    assert "__simt_vf__" in rendered
    assert "(__ubuf__" in rendered
    assert "507035" in rendered

