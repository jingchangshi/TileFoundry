"""Tests for pure-golden Stage-0 cold-start hardening and semantic analysis."""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from textwrap import dedent

import pytest

from tests.ascend.benchmarks.cannbench_contract import ParamBuilder
from tilefoundry_ascend.agent.campaign import marathon
from tilefoundry_ascend.agent.campaign.anchor_specs import (
    DEFAULT_REGIME_SPECS,
    CannbenchOpsRegistry,
    RegimeTargetSpec,
    build_semantic_regime_specs,
    extract_schema_driven_case_dims,
)
from tilefoundry_ascend.agent.campaign.marathon import (
    MarathonConfig,
    build_marathon_arg_parser,
    build_stage0_static_validator,
    run_operator_marathon,
)
from tilefoundry_ascend.agent.campaign.seed_synthesizer import (
    OracleAccuracyResult,
    _canonical_seed_track,
    _seed_parameter_specs,
    extract_golden_entrypoint_contract,
    format_golden_entrypoint_prompt_block,
    prepare_operator_seed_or_takeover,
    validate_entrypoint_signature_compatibility,
)
from tilefoundry_ascend.agent.orchestrator.gate_funnel import StageAResult, StageBResult
from tilefoundry_ascend.agent.orchestrator.llm_client import GenerationRequest, MockLLMBackend
from tilefoundry_ascend.ascend.sil.pytorch_frontend import (
    PyTorchSemanticAnalyzer,
    RepresentativeCase,
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
