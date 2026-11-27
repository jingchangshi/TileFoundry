"""Tests for EnvelopeInspector, Dual-Mode Physical Ledger, Pass Tap, and 6-Op Stage-A."""

import ast
import contextlib
import importlib
import importlib.util
import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import tilelang.ascend.language as T
import torch
from tilelang import tvm
from tilelang.utils.language import DataType as TLDataType
from tvm import tirx

from examples.ascend950pr_ops.add_rms_norm_dynamic_quant import kernels as rms_kernels
from examples.ascend950pr_ops.attention_residual import kernels as ar_kernels
from examples.ascend950pr_ops.grouped_matmul import kernels as gmm_kernels
from examples.ascend950pr_ops.kimi_delta_attention import kernels as kda_kernels
from examples.ascend950pr_ops.mla import kernels as mla_kernels
from examples.ascend950pr_ops.sparse_flash_attention import kernels as sfa_kernels
from tilefoundry.vendor.tilelang_z3 import Buffer, Capacity, Edge, Problem, Resource, Reuse, Task
from tilefoundry_ascend.agent.campaign import campaign_runner as cr_mod
from tilefoundry_ascend.agent.campaign import helpers as helpers_mod
from tilefoundry_ascend.agent.campaign import stage_gate_coordinator as sgc_mod
from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (
    extract_ordered_tensor_shapes,
    run_l0_static_precheck,
)
from tilefoundry_ascend.agent.kb import build_predicate_context, load_seed_rules
from tilefoundry_ascend.agent.orchestrator.gate_funnel import (
    StageAResult,
    StageBResult,
    _build_ascendc_static_ledger,
    _safe_eval_cpp_int,
    build_stage_b_driver_source,
    run_stage_a_static,
)
from tilefoundry_ascend.agent.orchestrator.stage_b_gate import run_stage_b_ascendc_compile
from tilefoundry_ascend.agent.specs.anchor_specs import (
    AnchorCaseSpec,
    PerfTargetSpec,
    RegimeTargetSpec,
)
from tilefoundry_ascend.ascend.passes import (
    ObjectivePhysicalLedger,
    ScopeWaterline,
    build_objective_physical_ledger,
    build_scope_waterline,
)
from tilefoundry_ascend.ascend.schedule import (
    ReadOnlyAscendPassTap,
    bitwise_conflict_pairs,
    inspect_tilelang_primfunc,
    pipe_cycles_from_problem,
    solve_ascend_pipeline,
    specialize_primfunc_for_anchor,
    task_resource_mask,
)
from tilefoundry_ascend.ascend.schedule.envelope_inspector import (
    _count_in_loop_pipe_all_barriers,
    _count_tirx_ub_scalar_compute_loops,
    _detect_tilelang_aiv_split_factor,
    _effective_tirx_call_name,
    _try_solve_tilelang_pipelined_ii,
)
from tilefoundry_ascend.ascend.target import (
    DEFAULT_CANN_SERIES,
    AscendSpaceFacts,
    AscendTarget,
    NodeResourceMask,
    resolve_cann_series,
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load_dynamic_module(tmp_path: Path, filename: str, mod_name: str, src: str) -> object:
    mod_path = tmp_path / filename
    mod_path.write_text(src, encoding="utf-8")
    spec = importlib.util.spec_from_file_location(mod_name, str(mod_path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _assert_stage_a_legal(
    prog: object,
    pass_configs: dict[str, object] | None,
    compile_flags: list[str] | tuple[str, ...] | None,
    label: str,
    var_bindings: dict[str, int] | None = None,
) -> None:
    insp = inspect_tilelang_primfunc(
        prog, pass_configs=pass_configs, compile_flags=compile_flags, var_bindings=var_bindings
    )
    assert insp.lowering_error is None, f"{label}: {insp.lowering_error}"
    assert insp.ledger.is_legal, f"{label}: overflow={insp.ledger.max_overflow_bytes}"


def _sample_gemm_primfunc():
    @T.prim_func
    def sample_gemm(
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

    return sample_gemm


@pytest.mark.parametrize("op_case", ["sfa", "mla_decode", "kda_128", "kda_256", "gmm"])
def test_contract_d_knobs_and_stage_a_legality(op_case: str) -> None:
    """Verify Contract-D knobs and Stage-A legality for KDA, GroupedMatmul, SFA, and MLA."""
    if op_case in ("kda_128", "kda_256"):
        dv = 128 if op_case == "kda_128" else 256
        cfg = kda_kernels._select_kimi_delta_attention_config(dk=128, dv=dv, tl_dtype="bfloat16", schedule_mode="tuned")
        assert int(cfg["chunk_t"]) >= 1 and (int(cfg["dv_tile"]) == 64 if dv == 256 else int(cfg["dv_tile"]) >= 64)
        _assert_stage_a_legal(kda_kernels._build_kimi_delta_attention_prim_func(**cfg), dict(cfg["pass_configs"]), list(cfg["compile_flags"]), op_case)
    elif op_case == "gmm":
        gmm_cfg = gmm_kernels._select_grouped_matmul_config(
            group_sizes=(512, 512), k_dim=1024, n_dim=2048, tl_dtype="float16",
            schedule_mode="tuned", bm_override=64, bn_override=128, bk_override=64, num_stages_override=1,
        )
        assert (gmm_cfg["bm"], gmm_cfg["bn"], gmm_cfg["bk"], gmm_cfg["num_stages"]) == (64, 128, 64, 1)
        gmm_src = (REPO_ROOT / "examples/ascend950pr_ops/grouped_matmul/kernels.py").read_text("utf-8")
        ast.parse(gmm_src)
        assert "linear_tile" in gmm_src
    elif op_case == "sfa":
        sfa_cfg = sfa_kernels._select_sparse_flash_attention_config(
            g_ratio=16, dk=128, dv=128, dtype="float16", top_k=512, schedule_mode="tuned"
        )
        assert all(k in sfa_cfg for k in ("bm", "bn", "num_stages")) and sfa_cfg["use_pipelined"] is True
        _assert_stage_a_legal(
            sfa_kernels._build_sparse_flash_attention_prim_func(**sfa_cfg),
            sfa_cfg.get("pass_configs") if isinstance(sfa_cfg.get("pass_configs"), dict) else None,
            list(sfa_cfg.get("compile_flags") or ()),
            "sfa",
            var_bindings={"b_dim": 1, "s1_dim": 1, "s2_dim": 128, "n_kv": 1, "top_k": 64, "is_causal_rt": 0},
        )
    else:
        for sel in (
            mla_kernels._select_mla_decode_config,
            mla_kernels._select_mla_bnsd_config,
            mla_kernels._select_mla_prefill_config,
        ):
            assert all(k in sel(schedule_mode="tuned") for k in ("bm", "bn", "num_stages"))
        dec_cfg = mla_kernels._select_mla_decode_config(
            b_dim=1, s_dim=1, s_kv=64, n_q=128, n_kv=1, schedule_mode="tuned"
        )
        _assert_stage_a_legal(
            mla_kernels._build_mla_split_kv_decode_func(**dec_cfg),
            dict(dec_cfg["pass_configs"]), list(dec_cfg["compile_flags"]), "mla_decode",
        )
        assert math.isfinite(float(dec_cfg["bm"]))


def test_envelope_inspector_positional_and_zero_divisor_safety() -> None:
    """Bind unnamed/renamed dynamic dims positionally and clamp zero/unbound divisors."""
    m_dyn, n_dyn = tvm.tirx.SizeVar("unnamed_m", "int32"), tvm.tirx.SizeVar("unnamed_n", "int32")

    @T.prim_func
    def dynamic_kernel(
        X: T.Tensor((m_dyn, n_dyn), "bfloat16"),
        Y: T.Tensor((m_dyn, n_dyn), "bfloat16"),
        tile_m: T.int32,
    ):
        with T.Kernel(T.ceildiv(m_dyn, tile_m)):
            x_ub = T.alloc_shared((64, 64), "bfloat16")
            T.copy(X[0:64, 0:64], x_ub)
            T.copy(x_ub, Y[0:64, 0:64])

    spec_fn, _ = specialize_primfunc_for_anchor(
        dynamic_kernel, var_bindings={"tile_m": 0}, ordered_tensor_shapes=((256, 64), (256, 64))
    )
    assert spec_fn is not None
    insp = inspect_tilelang_primfunc(
        dynamic_kernel, var_bindings={"tile_m": 0}, ordered_tensor_shapes=((256, 64), (256, 64))
    )
    assert insp.ledger is not None and "Divide by zero" not in (insp.lowering_error or "")
    assert insp.ledger.grid_blocks >= 1

    decode_split = tvm.tirx.SizeVar("decode_split", "int32")
    m_rows_dyn = tvm.tirx.SizeVar("m_rows_dyn", "int32")
    d_dyn = tvm.tirx.SizeVar("d_dyn", "int32")

    @T.prim_func
    def split_kernel(
        Q: T.Tensor((decode_split, d_dyn), "bfloat16"),
        Out: T.Tensor((m_rows_dyn, d_dyn), "bfloat16"),
        hc_dim_dyn: T.int32,
    ):
        with T.Kernel(m_rows_dyn // hc_dim_dyn):
            q_ub = T.alloc_shared((64, 64), "bfloat16")
            for _r in T.serial(decode_split // hc_dim_dyn):
                T.copy(Q[0:64, 0:64], q_ub)
                T.copy(q_ub, Out[0:64, 0:64])

    assert specialize_primfunc_for_anchor(
        split_kernel, var_bindings=None, ordered_tensor_shapes=((128, 64), (128, 64))
    )[0] is not None
    for bindings in ({"hc_dim_dyn": 0}, None):
        res = inspect_tilelang_primfunc(
            split_kernel, var_bindings=bindings, ordered_tensor_shapes=((128, 64), (128, 64))
        )
        assert res.ledger is not None and "Divide by zero" not in (res.lowering_error or "")
        assert res.ledger.grid_blocks >= 1


def test_envelope_inspector_compound_and_subtract_denominators() -> None:
    """Raise compound FloorDiv and subtracted denominator vars during specialization."""
    m_dyn, n_dyn = tvm.tirx.SizeVar("M", "int32"), tvm.tirx.SizeVar("N", "int32")

    @T.prim_func
    def nested_div_kernel(
        X: T.Tensor((m_dyn, n_dyn), "bfloat16"), Y: T.Tensor((m_dyn, n_dyn), "bfloat16")
    ):
        with T.Kernel(T.floordiv(m_dyn, T.floordiv(n_dyn, 64))):
            x_ub = T.alloc_shared((64, 32), "bfloat16")
            T.copy(X[0:64, 0:32], x_ub)
            T.copy(x_ub, Y[0:64, 0:32])

    for bindings, shapes in (({"M": 256, "N": 32}, None), (None, ((256, 32), (256, 32)))):
        spec_fn, _ = specialize_primfunc_for_anchor(
            nested_div_kernel, var_bindings=bindings, ordered_tensor_shapes=shapes
        )
        assert spec_fn is not None and not spec_fn.same_as(nested_div_kernel)
        for param in spec_fn.params:
            buf = spec_fn.buffer_map[param]
            assert isinstance(buf.shape[0], tvm.tirx.IntImm) and int(buf.shape[0].value) == 256
            assert isinstance(buf.shape[1], tvm.tirx.IntImm) and int(buf.shape[1].value) >= 64
        undef = tvm.tirx.analysis.undefined_vars(spec_fn.body)
        assert all(not isinstance(v, tvm.tirx.SizeVar) for v in undef)
        insp = inspect_tilelang_primfunc(
            nested_div_kernel, var_bindings=bindings, ordered_tensor_shapes=shapes
        )
        assert insp.ledger is not None and "Divide by zero" not in (insp.lowering_error or "")
        assert insp.ledger.grid_blocks >= 1

    @T.prim_func
    def deep_nested(
        X: T.Tensor((m_dyn, n_dyn), "bfloat16"), Y: T.Tensor((m_dyn, n_dyn), "bfloat16")
    ):
        with T.Kernel(T.floordiv(m_dyn, T.floordiv(T.floordiv(n_dyn, 64), 2))):
            ub = T.alloc_shared((64, 32), "bfloat16")
            T.copy(X[0:64, 0:32], ub)
            T.copy(ub, Y[0:64, 0:32])

    @T.prim_func
    def sub_denom(
        X: T.Tensor((m_dyn, n_dyn), "bfloat16"), Y: T.Tensor((m_dyn, n_dyn), "bfloat16")
    ):
        with T.Kernel(T.floordiv(m_dyn, n_dyn - 32)):
            ub = T.alloc_shared((64, 32), "bfloat16")
            T.copy(X[0:64, 0:32], ub)
            T.copy(ub, Y[0:64, 0:32])

    s_deep, _ = specialize_primfunc_for_anchor(deep_nested, var_bindings={"M": 256, "N": 32})
    assert all(int(s_deep.buffer_map[p].shape[1].value) >= 128 for p in s_deep.params)
    s_sub, _ = specialize_primfunc_for_anchor(sub_denom, var_bindings={"M": 256, "N": 16})
    assert all(int(s_sub.buffer_map[p].shape[1].value) > 32 for p in s_sub.params)


def test_ordered_tensor_shapes_extraction_and_l0_callsite_wiring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Wire extracted ordered_tensor_shapes through precheck, campaign runner, and marathon."""
    assert extract_ordered_tensor_shapes(case_entry={"input_shape": [16, 128]}) == ((16, 128),)
    assert extract_ordered_tensor_shapes(
        case_entry={"input_shape": [[4, 128], [128]], "output_shape": [[4, 128]]}
    ) == ((4, 128), (128,), (4, 128))
    assert extract_ordered_tensor_shapes(
        case_entry={"input_shapes": [[2, 8, 64]]},
        dims={"B": 2, "S": 8, "D": 64},
        proto={"outputs": [{"name": "y", "shape": "[B, S, D]"}]},
    ) == ((2, 8, 64), (2, 8, 64))

    dsl_src = (
        "import tilelang\nimport tilelang.ascend.language as T\nfrom tilelang import tvm\n\n"
        "def make_kernel():\n"
        "    m_u = tvm.tirx.SizeVar('unnamed_m', 'int32')\n"
        "    n_u = tvm.tirx.SizeVar('unnamed_n', 'int32')\n"
        "    @T.prim_func\n"
        "    def main_fn(X: T.Tensor((m_u, n_u), 'bfloat16'), "
        "Y: T.Tensor((m_u, n_u), 'bfloat16')):\n"
        "        with T.Kernel(T.floordiv(m_u, T.floordiv(n_u, 64))):\n"
        "            ub = T.alloc_shared((64, 32), 'bfloat16')\n"
        "            T.copy(X[0:64, 0:32], ub)\n"
        "            T.copy(ub, Y[0:64, 0:32])\n"
        "    return main_fn\n"
    )
    precheck = run_l0_static_precheck(
        dsl_src,
        builder_name="make_kernel",
        builder_kwargs={},
        case_entry={"input_shape": [[256, 32], [256, 32]]},
    )
    assert precheck.ledger is not None and "Divide by zero" not in (precheck.error_message or "")
    assert precheck.ledger.grid_blocks >= 1

    seen_cr: list[object] = []
    seen_mar: list[object] = []

    def _fake_cr(*_a, **kw) -> StageAResult:
        seen_cr.append(kw.get("ordered_tensor_shapes"))
        return StageAResult(passed=True, prim_func=None, ledger=precheck.ledger)

    def _fake_mar(*_a, **kw) -> StageAResult:
        seen_mar.append(kw.get("ordered_tensor_shapes"))
        return StageAResult(passed=True, prim_func=None, ledger=precheck.ledger)

    for mod in (helpers_mod, sgc_mod, cr_mod):
        monkeypatch.setattr(mod, "run_stage_a_static", _fake_cr)
    monkeypatch.setattr(cr_mod, "run_l1_compile_check", lambda *_a, **_kw: StageBResult(passed=True))

    mod_path = tmp_path / "mod.py"
    mod_src = (
        dsl_src
        + "_KERNEL_CACHE = {}\n"
        + "def _build_combine(**_kw):\n    return make_kernel()\n"
        + "def main(*_args, **_kwargs):\n"
        + "    if 'k' not in _KERNEL_CACHE:\n"
        + "        _KERNEL_CACHE['k'] = tilelang.compile(make_kernel())\n"
        + "    return _KERNEL_CACHE['k']\n"
        + "kernel = main\n"
    )
    mod_path.write_text(mod_src, encoding="utf-8")

    regime = RegimeTargetSpec(
        regime_tag="small",
        builder_candidates=("make_kernel",),
        sample_builder_kwargs={},
        representative_anchor_dims={"M": 256, "N": 32},
        editable_symbols=("make_kernel", "_build_combine"),
        representative_case_entry={"case_id": 1, "input_shape": [[256, 32], [256, 32]]},
    )
    runner = cr_mod.PerfTargetCampaignRunner(
        config=cr_mod.CampaignConfig(operator_name="custom_op", max_rounds=1),
        target_module_path=mod_path,
        cannbench_repo=tmp_path,
        perf_target=PerfTargetSpec("custom_op", 1.0, 1.0, 0.5, 50.0),
        anchors=(AnchorCaseSpec("custom_op", 1, "small", ref_perf_us=10.0, t_hw_us=5.0),),
        regime_specs=(regime,),
    )
    cr_cand = dsl_src + "\ndef _build_combine(**_kw):\n    fn = make_kernel()\n    return fn\n"
    assert runner._stage_gate_coordinator._default_l0_checker(cr_cand, regime).passed is True
    assert seen_cr == [((256, 32), (256, 32)), ((256, 32), (256, 32))]

    monkeypatch.setattr(cr_mod, "run_stage_a_static", _fake_mar)
    validator = cr_mod.build_stage0_static_validator(
        (regime,), entrypoint="main", worktree=tmp_path, env_file=None
    )
    ok, err = validator(mod_src, "tilelang_dsl")
    assert ok is True and err == ""
    assert seen_mar == [((256, 32), (256, 32))]


def test_mode_a_24bit_co_schedule_and_ledger() -> None:
    """Verify 24-bit resource masks and Mode A Z3 solve emitting ObjectivePhysicalLedger."""
    m_aiv0, m_aiv1 = NodeResourceMask.for_aiv0("VECTOR"), NodeResourceMask.for_aiv1("VECTOR")
    m_aic, m_dual = NodeResourceMask.for_aic("CUBE"), NodeResourceMask.for_dual_aiv("VECTOR")
    t_aiv0 = Task(name="vec0", resources=m_aiv0.to_solver_resources(), latency=1, stage=0)
    t_aiv1 = Task(name="vec1", resources=m_aiv1.to_solver_resources(), latency=1, stage=0)
    t_aic = Task(name="cube0", resources=m_aic.to_solver_resources(), latency=1, stage=0)
    t_dual = Task(name="dual_vec", resources=m_dual.to_solver_resources(), latency=1, stage=0)
    res_pool = (
        Resource("aiv", "aiv0", "VECTOR", 1),
        Resource("aiv", "aiv1", "VECTOR", 1),
        Resource("aic", "aic", "CUBE", 1),
    )

    p_prob = Problem(tasks=(t_aiv0, t_aiv1, t_aic), resources=res_pool, max_stage=0, manual=True)
    assert bitwise_conflict_pairs(p_prob.tasks) == ()
    p_res = solve_ascend_pipeline(p_prob)
    assert p_res.assignment.ii == 1 and p_res.assignment.phases == {"vec0": 0, "vec1": 0, "cube0": 0}

    d_prob = Problem(
        tasks=(t_aiv0, t_aiv1, t_dual, t_aic), resources=res_pool, max_stage=0, manual=True
    )
    assert bitwise_conflict_pairs(d_prob.tasks) == ((0, 2), (1, 2))
    d_res = solve_ascend_pipeline(d_prob)
    assert d_res.assignment.ii == 2
    assert d_res.assignment.phases["dual_vec"] not in (
        d_res.assignment.phases["vec0"],
        d_res.assignment.phases["vec1"],
    )
    assert task_resource_mask(t_dual) == m_dual

    facts = AscendTarget("ascend.950pr").get_facts(AscendSpaceFacts)
    loop_prob = Problem(
        tasks=(
            Task("load_a", (("aiv", "aiv0", "mte2", 1),), 1, 0),
            Task("load_b", (("aiv", "aiv0", "mte2", 1),), 1, 0),
            Task("vec_add", (("aiv", "aiv0", "vector", 1),), 1, 1),
            Task("store_c", (("aiv", "aiv0", "mte3", 1),), 1, 2),
        ),
        buffers=tuple(
            Buffer(n, 256, "UB", "aiv", auto_min=1, auto_max=2) for n in ("a_ub", "b_ub", "c_ub")
        ),
        edges=(
            Edge("load_a", "vec_add", 0, 1),
            Edge("load_b", "vec_add", 0, 1),
            Edge("vec_add", "store_c", 0, 1),
        ),
        reuses=(
            Reuse("a_ub", "vec_add", "load_a", 0, 1),
            Reuse("b_ub", "vec_add", "load_b", 0, 1),
            Reuse("c_ub", "store_c", "vec_add", 0, 1),
        ),
        resources=(
            Resource("aiv", "aiv0", "mte2", 1),
            Resource("aiv", "aiv0", "vector", 1),
            Resource("aiv", "aiv0", "mte3", 1),
        ),
        capacities=(Capacity("UB", "aiv", facts.ub_simdvf_usable_bytes),),
        max_stage=2,
        manual=True,
    )
    s_res = solve_ascend_pipeline(loop_prob)
    explicit_ub = {
        b.name: b.slot_bytes * s_res.buffer_versions.get(b.name, 1) for b in s_res.problem.buffers
    }
    ledger = build_objective_physical_ledger(
        analysis_mode="exact_modulo_z3",
        is_legal=True,
        steady_state_ii=s_res.assignment.ii,
        pipe_cycles=pipe_cycles_from_problem(s_res.problem, s_res.assignment),
        waterlines={
            "UB": build_scope_waterline(
                scope="UB",
                capacity_bytes=facts.ub_simdvf_usable_bytes,
                explicit_buffers=explicit_ub,
                implicit_buffers={},
            )
        },
    )
    assert isinstance(ledger, ObjectivePhysicalLedger) and ledger.analysis_mode == "exact_modulo_z3"
    assert ledger.is_legal is True and ledger.waterlines["UB"].overflow_bytes == 0
    assert ledger.dominant_pipe != "NONE"
    assert json.loads(ledger.to_json())["analysis_mode"] == "exact_modulo_z3"


def test_mode_b_two_snapshot_diff_dynamic_while_ub_overflow_and_rule_blindness() -> None:
    """Verify Two-Snapshot IR Diff catches passes, dynamic while/MTE2, UB overflow, and purity."""
    insp = inspect_tilelang_primfunc(_sample_gemm_primfunc())
    ldg = insp.ledger
    assert ldg.analysis_mode == "exact_modulo_z3" and ldg.steady_state_ii > 1 and ldg.is_legal
    assert "tl.MaterializeMultiBuffer" in insp.lowered_passes_observed
    assert "tl.MergeUBAllocations" in insp.lowered_passes_observed
    assert ldg.waterlines["L1"].explicit_bytes == 2 * (128 * 64 * 2)
    assert ldg.waterlines["L1"].peak_allocated_bytes >= ldg.waterlines["L1"].explicit_bytes
    assert ldg.waterlines["L0C"].overflow_bytes == 0
    assert ldg.dominant_pipe in ("AIC.PIPE_FIX", "AIC.PIPE_M", "AIC.PIPE_MTE2")

    @T.prim_func
    def dynamic_fragmented(
        X: T.Tensor((32, 256), "float32"),
        Y: T.Tensor((32, 256), "float32"),
        steps: T.Tensor((1,), "int32"),
    ):
        with T.Kernel(1):
            ub = T.alloc_shared((32, 256), "float32")
            it = T.alloc_var("int32", 0)
            while it < steps[0]:
                for seg in T.serial(32):
                    c0 = seg * 8
                    T.copy(X[0, c0 : c0 + 5], ub[0, c0 : c0 + 5])
                T.copy(ub, Y[0:32, 0:256])
                it += 1

    res_dyn = inspect_tilelang_primfunc(
        dynamic_fragmented, pass_configs={"tl.enable_auto_schedule": False}
    )
    assert "while_loop" in res_dyn.ledger.opaque_dynamic_regions
    assert "data_dependent_branch" in res_dyn.ledger.opaque_dynamic_regions
    assert res_dyn.ledger.mte2_inflight_per_aiv == 32
    assert res_dyn.ledger.unaligned_dma_padding_bytes > 0

    @T.prim_func
    def huge_ub_kernel(
        X: T.Tensor((2, 256, 128), "float32"), Y: T.Tensor((2, 256, 128), "float32")
    ):
        with T.Kernel(1):
            a_ub, b_ub = T.alloc_shared((256, 128), "float32"), T.alloc_shared((256, 128), "float32")
            for k in T.Pipelined(2, num_stages=2):
                T.copy(X[k, 0:256, 0:128], a_ub)
                T.copy(a_ub, b_ub)
                T.copy(b_ub, Y[k, 0:256, 0:128])

    res_ov = inspect_tilelang_primfunc(huge_ub_kernel)
    assert res_ov.ledger.is_legal is False
    assert (
        res_ov.ledger.max_overflow_bytes > 0
        or "lowering_rejected" in res_ov.ledger.opaque_dynamic_regions
    )

    ascend_root = REPO_ROOT / "targets" / "ascend" / "tilefoundry_ascend" / "ascend"
    assert all("tilefoundry_ascend.agent" not in p.read_text("utf-8") for p in ascend_root.rglob("*.py"))


def test_fp1_trip_scaling_dual_aiv_split_pipelined_ii_and_ledger_roundtrip() -> None:
    """Verify loop trip scaling, 1C2V split, Mode A II + degradation, and ledger roundtrip."""
    @T.prim_func
    def vec_copy(a: T.Tensor((4, 60), "float32"), c: T.Tensor((4, 60), "float32")):
        with T.Kernel(1):
            a_ub = T.alloc_shared((1, 60), "float32")
            for k in T.serial(4):
                T.copy(a[k : k + 1, 0:60], a_ub)
                T.copy(a_ub, c[k : k + 1, 0:60])

    r_vec = inspect_tilelang_primfunc(vec_copy, pass_configs={"tl.enable_auto_schedule": False})
    assert r_vec.ledger.analysis_mode == "readonly_envelope"
    assert isinstance(r_vec.ledger.waterlines["UB"], ScopeWaterline)
    assert r_vec.ledger.opaque_dynamic_regions == ()
    assert r_vec.ledger.pipe_cycles["AIV0.PIPE_MTE2"] > 0
    assert r_vec.ledger.pipe_cycles["AIV0.PIPE_MTE3"] > 0
    assert r_vec.ledger.unaligned_dma_padding_bytes == 64

    @T.prim_func
    def dyn_loop(
        a: T.Tensor((4, 64), "float32"),
        c: T.Tensor((4, 64), "float32"),
        n_steps: T.Tensor((1,), "int32"),
    ):
        with T.Kernel(1):
            a_ub = T.alloc_shared((1, 64), "float32")
            for k in T.serial(n_steps[0]):
                T.copy(a[k : k + 1, 0:64], a_ub)
                T.copy(a_ub, c[k : k + 1, 0:64])

    r_dyn = inspect_tilelang_primfunc(dyn_loop, pass_configs={"tl.enable_auto_schedule": False})
    assert any(tag.startswith("dynamic_for:") for tag in r_dyn.ledger.opaque_dynamic_regions)

    @T.prim_func
    def single_aiv(X: T.Tensor((128, 128), "float32"), Y: T.Tensor((128, 128), "float32")):
        with T.Kernel(1):
            x_ub = T.alloc_shared((128, 128), "float32")
            T.copy(X[0:128, 0:128], x_ub)
            T.copy(x_ub, Y[0:128, 0:128])

    @T.prim_func
    def mixed_dual(X: T.Tensor((128, 128), "float32"), Y: T.Tensor((128, 128), "float32")):
        with T.MixedKernel(1, sids=2) as (bx, sid):
            x_ub = T.alloc_shared((64, 128), "float32")
            x_l1 = T.alloc_l1((128, 128), "float32")
            for _r in T.serial(4):
                T.copy(X[sid * 64 : (sid + 1) * 64, 0:128], x_ub)
            with T.SimdVF():
                m32 = T.simd.pset(32, "PAT_ALL")
                for r in T.serial(8):
                    v = T.simd.vld(x_ub[r, 0], dist="NORM_B32")
                    T.simd.vsts(x_ub[r, 0], v, mask=m32, dist="NORM_B32")
            T.dual_copy(x_ub, x_l1)
            T.copy(x_ub, Y[sid * 64 : (sid + 1) * 64, 0:128])

    @T.prim_func
    def guarded_sid(X: T.Tensor((128, 128), "float32")):
        with T.MixedKernel(1, sids=2) as (bx, sid):
            x_ub = T.alloc_shared((64, 128), "float32")
            if sid == 0:
                with T.SimdVF():
                    m32 = T.simd.pset(32, "PAT_ALL")
                    for r in T.serial(4):
                        v = T.simd.vld(x_ub[r, 0], dist="NORM_B32")
                        T.simd.vsts(x_ub[r, 0], v, mask=m32, dist="NORM_B32")

    assert _detect_tilelang_aiv_split_factor(single_aiv, tirx=tirx) == 1
    assert _detect_tilelang_aiv_split_factor(mixed_dual, tirx=tirx) == 2
    cthread_iv = tirx.IterVar(tvm.ir.Range(0, 2), tirx.Var("sid", "int32"), 0, "cthread")
    synth_fn = tirx.PrimFunc([], tirx.AttrStmt(cthread_iv, "thread_extent", 2, tirx.Evaluate(0)))
    assert _detect_tilelang_aiv_split_factor(synth_fn, tirx=tirx) == 2

    assert inspect_tilelang_primfunc(single_aiv).ledger.pipe_cycles.get("AIV1.PIPE_V", 0) == 0
    s1 = inspect_tilelang_primfunc(mixed_dual, aiv_split_factor=1).ledger.pipe_cycles
    s2 = inspect_tilelang_primfunc(mixed_dual).ledger.pipe_cycles
    for p in ("PIPE_V", "PIPE_MTE2", "PIPE_MTE3"):
        assert s2[f"AIV1.{p}"] > 0 and s2[f"AIV0.{p}"] == s1[f"AIV0.{p}"] // 2
    g_res = inspect_tilelang_primfunc(guarded_sid).ledger.pipe_cycles
    assert g_res.get("AIV0.PIPE_V", 0) == 4 and g_res.get("AIV1.PIPE_V", 0) == 0

    @T.prim_func
    def mixed_pipelined(
        X: T.Tensor((128, 128), "bfloat16"),
        W: T.Tensor((128, 128), "bfloat16"),
        Y: T.Tensor((128, 128), "float32"),
    ):
        with T.MixedKernel(1, sids=2) as (bx, sid):
            x_l1, w_l1 = T.alloc_l1((128, 64), "bfloat16"), T.alloc_l1((128, 64), "bfloat16")
            acc, out_ub = T.alloc_l0c((128, 128), "float32"), T.alloc_shared((64, 128), "float32")
            for kt in T.Pipelined(2, num_stages=2):
                T.copy(X[0:128, kt * 64 : (kt + 1) * 64], x_l1)
                T.copy(W[0:128, kt * 64 : (kt + 1) * 64], w_l1)
                T.gemm(x_l1, w_l1, acc, transpose_B=True, clear_accum=kt == 0)
            T.dual_copy(acc, out_ub)
            T.copy(out_ub, Y[sid * 64 : (sid + 1) * 64, 0:128])

    res_mp = inspect_tilelang_primfunc(mixed_pipelined)
    assert res_mp.ledger.analysis_mode == "exact_modulo_z3" and res_mp.ledger.steady_state_ii > 1
    spaces = AscendTarget("ascend.950pr").get_facts(AscendSpaceFacts)
    deg_ii = _try_solve_tilelang_pipelined_ii(
        mixed_pipelined, spaces, tirx=tirx, DataTypeCls=TLDataType,
        aiv_split_factor=2, has_simtvf=False, unreserved_ubuf=False,
        degraded_names=("x_l1", "w_l1"),
    )
    assert deg_ii is not None and deg_ii >= res_mp.ledger.steady_state_ii

    wl_ub = ScopeWaterline(
        "UB", 245760, 32768, 4096, 36864, 0, {"a_ub": 16384, "b_ub": 16384},
        {"a_ub_expansion": 4096}, ("b_ub",),
    )
    assert ScopeWaterline.from_dict(wl_ub.to_dict()) == wl_ub
    ldg_rt = ObjectivePhysicalLedger(
        "exact_modulo_z3", True, 8, {"AIC.PIPE_M": 8}, "AIC.PIPE_M",
        {"UB": wl_ub}, 2, 16, ("simt_vf",),
    )
    assert ObjectivePhysicalLedger.from_dict(json.loads(ldg_rt.to_json())) == ldg_rt


def test_symbolic_grid_burst_unlimit_pipelining_fission_and_dav3510_signals(
    tmp_path: Path,
) -> None:
    """Verify symbolic grids, burst efficiency, unlimit_memory, fission, and dav-3510 metrics."""
    src = """
import tilelang.ascend.language as T
def make_dyn_vec():
    m_rows = T.dynamic("m_rows")
    @T.prim_func
    def main(X: T.Tensor((m_rows, 64), "float32"), Y: T.Tensor((m_rows, 64), "float32")):
        with T.Kernel(T.ceildiv(m_rows, 64)) as bx:
            ub = T.alloc_shared((64, 64), "float32")
            T.copy(X[bx * 64 : bx * 64 + 64, 0:64], ub)
            T.copy(ub, Y[bx * 64 : bx * 64 + 64, 0:64])
    return main
"""
    mod = _load_dynamic_module(tmp_path, "dyn_grid.py", "_tf_dyn_grid_test", src)
    r13 = next(r for r in load_seed_rules() if r.rule_id == "R13_SPLIT_KV_WAVE_RECOVERY")
    vec_fn = mod.make_dyn_vec()

    hi = inspect_tilelang_primfunc(vec_fn, var_bindings={"m_rows": 4096})
    assert hi.ledger.grid_blocks == 64 and not hi.ledger.grid_is_symbolic
    assert hi.ledger.wave_occupancy > 1.0 and not r13.matches(build_predicate_context(hi.ledger))

    unb = inspect_tilelang_primfunc(vec_fn, var_bindings=None)
    assert unb.ledger.grid_is_symbolic and unb.ledger.wave_occupancy == 1.0
    assert unb.ledger.grid_blocks == 56 and not r13.matches(build_predicate_context(unb.ledger))
    assert ObjectivePhysicalLedger.from_dict(unb.ledger.to_dict()) == unb.ledger

    lo = inspect_tilelang_primfunc(vec_fn, var_bindings={"m_rows": 128})
    assert lo.ledger.grid_blocks == 2 and 0.0 < lo.ledger.wave_occupancy < 0.5
    assert r13.matches(build_predicate_context(lo.ledger)) is True

    @T.prim_func
    def copy_8k(X: T.Tensor((16, 1024), "float32"), Y: T.Tensor((16, 1024), "float32")):
        with T.Kernel(1):
            with T.SimdVF():
                ub = T.alloc_shared((2, 1024), "float32")
                T.copy(X[0:2, 0:1024], ub)
                T.copy(ub, Y[0:2, 0:1024])

    @T.prim_func
    def copy_32k(X: T.Tensor((16, 1024), "float32"), Y: T.Tensor((16, 1024), "float32")):
        with T.Kernel(1):
            with T.SimdVF():
                ub = T.alloc_shared((8, 1024), "float32")
                T.copy(X[0:8, 0:1024], ub)
                T.copy(ub, Y[0:8, 0:1024])

    r8k, r32k = inspect_tilelang_primfunc(copy_8k).ledger, inspect_tilelang_primfunc(copy_32k).ledger
    assert r8k.min_gm_dma_bytes == 8192 and r32k.min_gm_dma_bytes == 32768
    assert r8k.dma_burst_efficiency_ratio < r32k.dma_burst_efficiency_ratio
    assert (r8k.pipe_cycles["AIV0.PIPE_MTE2"] / 8192.0) > (
        r32k.pipe_cycles["AIV0.PIPE_MTE2"] / 32768.0
    )

    @T.prim_func
    def unlimit_fn(X: T.Tensor((16, 1024), "float16"), Y: T.Tensor((16, 1024), "float16")):
        with T.Kernel(1):
            T.annotate_unlimit_memory("shared")
            buf = T.alloc_shared((2, 1024), "float16")
            for i in T.Pipelined(4, num_stages=2):
                T.copy(X[i * 2 : (i + 1) * 2, :], buf)
                T.copy(buf, Y[i * 2 : (i + 1) * 2, :])

    @T.prim_func
    def static_pp(X: T.Tensor((16, 1024), "float16"), Y: T.Tensor((16, 1024), "float16")):
        with T.Kernel(1):
            va, vb = T.alloc_shared((2, 1024), "float16"), T.alloc_shared((2, 1024), "float16")
            for i in T.serial(2):
                T.copy(X[i * 4 : i * 4 + 2, :], va)
                T.copy(X[i * 4 + 2 : i * 4 + 4, :], vb)
                T.copy(va, Y[i * 4 : i * 4 + 2, :])
                T.copy(vb, Y[i * 4 + 2 : i * 4 + 4, :])

    ru = inspect_tilelang_primfunc(unlimit_fn).ledger
    assert ru.has_unlimit_memory_attr and ru.uses_dynamic_offset_pipelining
    assert ru.waterlines["UB"].capacity_bytes in (248 * 1024, 256 * 1024)
    assert inspect_tilelang_primfunc(static_pp).ledger.uses_dynamic_offset_pipelining is False

    @T.prim_func
    def coupled_fn(
        Q: T.Tensor((4, 64), "float32"),
        Out: T.Tensor((1, 64), "float32"),
        ScalarOut: T.Tensor((4, 1), "float32"),
    ):
        with T.Kernel(1):
            q_ub = T.alloc_shared((1, 64), "float32")
            acc_ub, row_ub = T.alloc_shared((4, 64), "float32"), T.alloc_shared((4, 1), "float32")
            for s in T.serial(4):
                T.copy(Q[s : s + 1, :], q_ub)
                with T.SimdVF():
                    m32 = T.simd.pset(32, "PAT_ALL")
                    qv = T.simd.vld(q_ub[0, 0])
                    sv = T.simd.vdupv(T.simd.vcadd(qv, mask=m32), mask=m32, pos="POS_LOWEST")
                    T.simd.vsts(row_ub[s, 0], T.simd.vexp(sv, mask=m32), mask=m32, dist="ONEPT_B32")
                    av = T.simd.vld(acc_ub[s, 0])
                    T.simd.vsts(acc_ub[s, 0], T.simd.vadd(av, qv, mask=m32), mask=m32)
            T.copy(acc_ub[0:1, :], Out)
            T.copy(row_ub, ScalarOut)

    @T.prim_func
    def fission_fn(
        Q: T.Tensor((4, 64), "float32"),
        Out: T.Tensor((1, 64), "float32"),
        ScalarOut: T.Tensor((4, 1), "float32"),
    ):
        with T.Kernel(1):
            q_ub = T.alloc_shared((1, 64), "float32")
            acc_ub, exb = T.alloc_shared((4, 64), "float32"), T.alloc_shared((4, 1), "float32")
            for s in T.serial(4):
                T.copy(Q[s : s + 1, :], q_ub)
                with T.SimdVF():
                    m32 = T.simd.pset(32, "PAT_ALL")
                    qv = T.simd.vld(q_ub[0, 0])
                    sv = T.simd.vdupv(T.simd.vcadd(qv, mask=m32), mask=m32, pos="POS_LOWEST")
                    T.simd.vsts(exb[s, 0], T.simd.vexp(sv, mask=m32), mask=m32, dist="ONEPT_B32")
            for s in T.serial(4):
                T.copy(Q[s : s + 1, :], q_ub)
                with T.SimdVF():
                    m32 = T.simd.pset(32, "PAT_ALL")
                    qv, av = T.simd.vld(q_ub[0, 0]), T.simd.vld(acc_ub[s, 0])
                    T.simd.vsts(acc_ub[s, 0], T.simd.vadd(av, qv, mask=m32), mask=m32)
            T.copy(acc_ub[0:1, :], Out)
            T.copy(exb, ScalarOut)

    assert inspect_tilelang_primfunc(coupled_fn).ledger.has_coupled_reduce_sfu_accum is True
    assert inspect_tilelang_primfunc(fission_fn).ledger.has_coupled_reduce_sfu_accum is False

    def _gemm_small(dt: str):
        @T.prim_func
        def g(A: T.Tensor((32, 64), dt), B: T.Tensor((64, 32), dt), C: T.Tensor((32, 32), "float32")):
            with T.Kernel(1):
                al, bl = T.alloc_l1((32, 64), dt), T.alloc_l1((64, 32), dt)
                cl = T.alloc_l0c((32, 32), "float32")
                T.copy(A[0:32, 0:64], al)
                T.copy(B[0:64, 0:32], bl)
                T.gemm(al, bl, cl, clear_accum=True)
                T.copy(cl, C[0:32, 0:32])

        return inspect_tilelang_primfunc(g, pass_configs={"tl.enable_auto_schedule": False}).ledger

    r_bf16, r_fp8 = _gemm_small("bfloat16"), _gemm_small("float8_e4m3fn")
    assert r_bf16.pipe_cycles["AIC.PIPE_M"] == 22 and r_fp8.pipe_cycles["AIC.PIPE_M"] == 14
    assert 0.0 < r_bf16.l1_bank_conflict_risk <= 0.5

    @T.prim_func
    def ub_conf(X: T.Tensor((16, 128), "float32"), Y: T.Tensor((16, 128), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((16, 128), "float32")
            T.copy(X[0:16, 0:128], ub)
            T.copy(ub, Y[0:16, 0:128])

    @T.prim_func
    def ub_pad(X: T.Tensor((16, 136), "float32"), Y: T.Tensor((16, 136), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((16, 136), "float32")
            T.copy(X[0:16, 0:136], ub)
            T.copy(ub, Y[0:16, 0:136])

    no_auto = {"tl.enable_auto_schedule": False}
    assert inspect_tilelang_primfunc(ub_conf, pass_configs=no_auto).ledger.ub_bank_conflict_risk == 1.0
    assert inspect_tilelang_primfunc(ub_pad, pass_configs=no_auto).ledger.ub_bank_conflict_risk == 0.0

    @T.prim_func
    def guarded_pipelined(
        X: T.Tensor((128, 128), "bfloat16"),
        W: T.Tensor((128, 128), "bfloat16"),
        Flag: T.Tensor((4,), "int32"),
        Y: T.Tensor((128, 128), "float32"),
    ):
        with T.Kernel(1):
            x_l1, w_l1 = T.alloc_l1((128, 64), "bfloat16"), T.alloc_l1((128, 64), "bfloat16")
            acc = T.alloc_l0c((128, 128), "float32")
            for kt in T.Pipelined(2, num_stages=2):
                T.copy(X[0:128, kt * 64 : (kt + 1) * 64], x_l1)
                T.copy(W[0:128, kt * 64 : (kt + 1) * 64], w_l1)
                if Flag[kt] > 0:
                    T.gemm(x_l1, w_l1, acc, transpose_B=True, clear_accum=kt == 0)
            T.copy(acc, Y[0:128, 0:128])

    spaces = AscendTarget("ascend.950pr").get_facts(AscendSpaceFacts)
    ana_ii = _try_solve_tilelang_pipelined_ii(
        guarded_pipelined, spaces, tirx=tirx, DataTypeCls=TLDataType,
        aiv_split_factor=1, has_simtvf=False, unreserved_ubuf=False,
    )
    rg = inspect_tilelang_primfunc(guarded_pipelined, pass_configs=no_auto).ledger
    assert ana_ii is not None and ana_ii >= 1 and rg.steady_state_ii == ana_ii
    assert "data_dependent_branch" in rg.opaque_dynamic_regions
    assert rg.analysis_mode == "readonly_envelope"


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("(uint16_t)BN", 64),
        ("(uint32_t)(BN * 2)", 128),
        ("(int32_t)BM + (int)4", 36),
        ("(size_t)sizeof(float)", 4),
        ("static_cast<uint16_t>(BN) * sizeof(__bf16)", 128),
        ("sizeof(int32_t) + sizeof(uint32_t)", 8),
        ("sizeof(half) + sizeof(bfloat16_t)", 4),
        ("sizeof(uint16_t) + sizeof(int16_t)", 4),
        ("sizeof(int8_t) + sizeof(uint8_t)", 2),
    ],
)
def test_cpp_int_eval_strips_casts_and_rewrites_sizeof(expr: str, expected: int) -> None:
    """Verify _safe_eval_cpp_int strips C/static casts and rewrites sizeof widths."""
    assert _safe_eval_cpp_int(expr, {"BN": 64, "BM": 32}) == expected


def test_trackb_ledger_stage_b_failures_and_tirx_extern(tmp_path: Path) -> None:
    """Verify Track-B ledger widths, Stage-B failure CCE retention, tirx extern, and driver."""
    tu_src = gmm_kernels._render_trackb_tu(m_dim=128, k_dim=256, n_dim=512, tl_dtype="float16")
    gmm_ldg = _build_ascendc_static_ledger(tu_src)
    assert gmm_ldg.min_gm_dma_bytes > 0 and gmm_ldg.dominant_contig_bytes > 0
    assert 0.0 < gmm_ldg.dma_burst_efficiency_ratio <= 1.0

    for dst_type, eb in (("float", 4), ("__bf16", 2), ("int8_t", 1)):
        cpp_src = (
            f'extern "C" __global__ __cube__ void l0c_probe(__gm__ {dst_type}* out_gm) {{\n'
            f"    __cc__ float* l0c = (__cc__ float*)(0x0000);\n"
            f"    asc_copy_l0c2gm(out_gm, l0c, (uint16_t)(32 + sizeof({dst_type})), "
            f"(uint16_t)16, (uint32_t)64, (uint16_t)16, "
            f"static_cast<asc_store_l2_cache_mode>(0), static_cast<asc_unit_flag_mode>(0), "
            f"static_cast<asc_quant_mode>(0), static_cast<asc_relu_pre_mode>(0), "
            f"false, true, false, false);\n}}\n"
        )
        led = _build_ascendc_static_ledger(cpp_src)
        assert led.dominant_contig_bytes == (32 + eb) * eb
        assert led.min_gm_dma_bytes == (32 + eb) * 16 * eb

    sample_cpp = 'extern "C" __global__ __vector__ void k(__gm__ float* out) { asc_init(); }\n'
    res_fail = run_stage_b_ascendc_compile(
        sample_cpp,
        bisheng_resolver=lambda *_a, **_kw: "/usr/bin/false",
        subproc_run=lambda *_a, **_kw: SimpleNamespace(returncode=1, stdout="", stderr="err: bad"),
    )
    assert res_fail.passed is False and res_fail.generated_cce == sample_cpp

    def _ok_obj(cmd, **_kw):
        Path(cmd[cmd.index("-o") + 1]).write_bytes(b"\x7fELF" + b"\x00" * 64)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    res_inst = run_stage_b_ascendc_compile(
        sample_cpp,
        bisheng_resolver=lambda *_a, **_kw: "/usr/bin/false",
        subproc_run=_ok_obj,
        verify_instantiated_fn=lambda *_a, **_kw: (False, "uninstantiated"),
        disassemble_fn=lambda *_a, **_kw: "",
    )
    assert res_inst.passed is False and res_inst.generated_cce == sample_cpp

    for intrin in ("call_extern", "call_pure_extern", "call_intrin", "call_llvm_intrin"):
        for sym in ("simd_inst::vadd", "pipe_barrier", "asc_copy_gm2ubuf", "tl.ascend_copy"):
            assert _effective_tirx_call_name(SimpleNamespace(op=f"tirx.{intrin}", args=[sym])) == sym

    @T.prim_func
    def extern_vec(x: T.Tensor((128,), "float32"), y: T.Tensor((128,), "float32")) -> None:
        with T.Kernel(1):
            ub = T.alloc_shared((128,), "float32")
            T.copy(x[0:128], ub)
            for _k in T.serial(4):
                T.call_extern("handle", "simd_inst::vadd", ub.data, ub.data, ub.data, 128)
                T.call_extern("handle", "pipe_barrier", "PIPE_ALL")
            T.copy(ub, y[0:128])

    assert _count_in_loop_pipe_all_barriers(extern_vec.body, tirx=tirx) == 1
    assert _count_tirx_ub_scalar_compute_loops(extern_vec.body, {"ub"}, tirx=tirx) == 0
    ext_res = inspect_tilelang_primfunc(extern_vec)
    assert ext_res.ledger.ub_scalar_compute_loops == 0 and ext_res.ledger.in_loop_pipe_all_barriers == 1

    k_py, tir_d, cce_d = tmp_path / "k.py", tmp_path / "out.tir", tmp_path / "out.cce"
    k_py.write_text("def make_kernel():\n    return 'dummy_fn'\n", encoding="utf-8")
    drv = build_stage_b_driver_source(
        k_py, factory_name="make_kernel", tir_dump_path=str(tir_d), kernel_dump_path=str(cce_d)
    )
    calls: list[bool] = []
    fail_flag = [False]

    def _fake_lower(fn, *, target: str, enable_device_compile: bool):
        assert fn == "dummy_fn" and target == "ascend"
        calls.append(enable_device_compile)
        if enable_device_compile and fail_flag[0]:
            raise RuntimeError("device compile failed")
        return SimpleNamespace(device_mod="LOWERED_TIR", kernel_source="// CCE_SRC")

    @contextlib.contextmanager
    def _fake_sess(_pc):
        yield

    _, rest = drv.split("_cfg = json.loads(", 1)
    cfg_line, suffix = rest.split("\n", 1)
    exec_body = suffix[suffix.index("        with _ascend_compile_session(_merged_pc):") :]
    ns = {
        "_cfg": json.loads(eval(cfg_line.rstrip(")"))),
        "_fn": "dummy_fn",
        "_merged_pc": {},
        "pathlib": importlib.import_module("pathlib"),
        "tilelang": SimpleNamespace(lower=_fake_lower),
        "_ascend_compile_session": _fake_sess,
    }
    exec("if True:\n" + exec_body, ns)
    assert calls == [True] and tir_d.read_text() == "LOWERED_TIR"
    calls.clear()
    fail_flag[0] = True
    tir_d.unlink()
    cce_d.unlink()
    with pytest.raises(RuntimeError, match="device compile failed"):
        exec("if True:\n" + exec_body, ns)
    assert calls == [True, False] and cce_d.read_text() == "// CCE_SRC"


def test_readonly_pass_tap_device_mod_reconcile_and_cann93(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify ReadOnlyAscendPassTap immutability, device_mod reconciliation, and CANN 9.3 lock."""
    gemm_fn = _sample_gemm_primfunc()
    h0, s0 = tvm.ir.structural_hash(gemm_fn), gemm_fn.script()
    res = inspect_tilelang_primfunc(gemm_fn)
    assert tvm.ir.structural_hash(gemm_fn) == h0 and gemm_fn.script() == s0
    for pname in (
        "tl.MaterializeMultiBuffer",
        "tl.InsertNd2Nz",
        "tl.NormalizeAscendFractalStorage",
        "tl.MergeUBAllocations",
    ):
        assert pname in res.lowered_passes_observed
    assert (
        "tl.AscendLowerTileOp" in res.lowered_passes_observed
        or "tl.LowerTileOp" in res.lowered_passes_observed
    )
    assert res.ledger.uses_dynamic_offset_pipelining is True
    assert isinstance(ReadOnlyAscendPassTap(), object)

    @T.prim_func
    def narrow_barrier(X: T.Tensor((16, 64), "float32"), Y: T.Tensor((16, 64), "float32")):
        with T.Kernel(1):
            ub = T.alloc_shared((16, 64), "float32")
            for i in T.serial(16):
                T.copy(X[i, 0:4], ub[i, 0:4])
                T.ascend_pipe_barrier("PIPE_ALL")
                for j in T.serial(64):
                    ub[i, j] = ub[i, j] + T.float32(1.0)
                T.copy(ub[i, 0:4], Y[i, 0:4])

    rn = inspect_tilelang_primfunc(narrow_barrier, pass_configs={"tl.enable_auto_schedule": False})
    assert rn.ledger.in_loop_pipe_all_barriers >= 1 and rn.ledger.min_gm_dma_bytes == 16
    assert 0.0 < rn.ledger.dma_burst_efficiency_ratio < 1.0 and rn.ledger.ub_scalar_compute_loops >= 1

    @T.prim_func
    def asym_sid():
        with T.MixedKernel(1, sids=2) as (bx, sid):
            x_ub = T.alloc_shared((64, 128), "float32")
            trips = 6 if sid == 0 else 2
            if sid == 0:
                with T.SimdVF():
                    m32 = T.simd.pset(32, "PAT_ALL")
                    for r in T.serial(6):
                        T.simd.vsts(x_ub[r, 0], T.simd.vld(x_ub[r, 0], dist="NORM_B32"), mask=m32, dist="NORM_B32")
            else:
                with T.SimdVF():
                    m32 = T.simd.pset(32, "PAT_ALL")
                    for r in T.serial(2):
                        T.simd.vsts(x_ub[r, 0], T.simd.vld(x_ub[r, 0], dist="NORM_B32"), mask=m32, dist="NORM_B32")
            _ = trips

    ra = inspect_tilelang_primfunc(asym_sid).ledger
    assert ra.pipe_cycles["AIV0.PIPE_V"] == 6 and ra.pipe_cycles["AIV1.PIPE_V"] == 2
    assert ra.ub_scalar_compute_loops == 0

    for var in (
        "TILEFOUNDRY_CANN_SERIES",
        "TILEFOUNDRY_NPU_CANN_SERIES",
        "ASCEND_HOME_PATH",
        "ASCEND_TOOLKIT_HOME",
    ):
        monkeypatch.delenv(var, raising=False)
    assert resolve_cann_series() == DEFAULT_CANN_SERIES == "cann_9_3"
    monkeypatch.setenv("ASCEND_HOME_PATH", "/opt/Ascend/cann-9.3.0")
    assert resolve_cann_series() == "cann_9_3"
    assert resolve_cann_series(cann_home="/opt/Ascend/cann-9.3.0") == "cann_9_3"
    assert resolve_cann_series("cann_9_3") == "cann_9_3"


@pytest.mark.parametrize(
    ("op_name", "params", "has_kernel_alias"),
    [
        (
            "kimi_delta_attention",
            ("q", "k", "v", "g", "beta", "initial_state", "use_qk_l2norm", "scaleValue", "schedule_mode"),
            True,
        ),
        (
            "grouped_matmul",
            ("x", "weight", "bias", "group_list", "split_item", "transpose_weight", "schedule_mode"),
            False,
        ),
        (
            "sparse_flash_attention",
            ("query", "key", "value", "sparseIndices", "scaleValue", "inputLayout", "is_causal", "schedule_mode"),
            True,
        ),
        (
            "mla",
            ("q_nope", "q_rope", "k_nope", "k_rope", "v", "numKVHeads", "scaleValue", "inputLayout", "is_causal", "schedule_mode", "num_kv_splits_override"),
            False,
        ),
        ("add_rms_norm_dynamic_quant", ("x1", "x2", "gamma", "epsilon", "schedule_mode"), True),
        ("attention_residual", ("v_stack", "x", "w", "gamma", "alpha", "epsilon", "schedule_mode"), True),
    ],
)
def test_six_ops_public_entrypoint_signatures_and_exports(
    op_name: str, params: tuple[str, ...], has_kernel_alias: bool
) -> None:
    """Verify public entrypoint signatures and exports for all 6 Ascend 950PR operators."""
    mod = importlib.import_module(f"examples.ascend950pr_ops.{op_name}.kernels")
    pkg = importlib.import_module(f"examples.ascend950pr_ops.{op_name}")
    entry = getattr(mod, op_name)
    assert callable(entry) and getattr(pkg, op_name) is entry
    sig = inspect.signature(entry)
    assert tuple(sig.parameters.keys()) == params
    assert sig.parameters["schedule_mode"].default == "tuned"
    if has_kernel_alias:
        assert getattr(mod, "kernel") is entry


def test_add_rms_norm_dynamic_quant_reg_resident_and_smoke() -> None:
    """Verify RMSNorm reg_resident tightening, overrides, Stage-A static, and CPU smoke."""
    sel = rms_kernels._select_add_rms_norm_dynamic_quant_config
    c_reg = sel(d_dim=128, tl_dtype="bfloat16", m_rows=112, schedule_mode="tuned")
    assert c_reg["tile_m"] <= 2 and c_reg["regime"] == "aligned_reg_resident"
    c_m = sel(d_dim=128, tl_dtype="bfloat16", m_rows=524288, schedule_mode="tuned")
    assert c_m["tile_m"] > 2 and c_m["regime"] == "aligned_single_pass_ub"
    assert sel(d_dim=128, tl_dtype="bfloat16", m_rows=112, schedule_mode="tuned", tile_m_override=4)["regime"] == "aligned_single_pass_ub"
    c_ov = sel(d_dim=2048, tl_dtype="bfloat16", m_rows=4096, schedule_mode="tuned", tile_m_override=4, num_stages_override=1)
    assert c_ov["tile_m"] == 4 and c_ov["num_stages"] == 1

    src = (REPO_ROOT / "examples/ascend950pr_ops/add_rms_norm_dynamic_quant/kernels.py").read_text("utf-8")
    res = run_stage_a_static(
        src, factory_name="_build_add_rms_norm_dynamic_quant_prim_func",
        config_selector_name="_select_add_rms_norm_dynamic_quant_config", dims={"D": 4096, "M": 4096},
    )
    assert res.passed, res.error_message
    yq, xo, sc = rms_kernels.kernel(
        torch.randn(4, 256, dtype=torch.bfloat16),
        torch.randn(4, 256, dtype=torch.bfloat16),
        torch.randn(256, dtype=torch.bfloat16),
        epsilon=1e-6,
    )
    assert (yq.dtype, yq.shape) == (torch.int8, (4, 256))
    assert (xo.dtype, xo.shape) == (torch.bfloat16, (4, 256))
    assert (sc.dtype, sc.shape) == (torch.float32, (4,))


@pytest.mark.parametrize(
    ("d_dim", "m_rows", "dt"),
    [(128, 112, "bfloat16"), (128, 524288, "float16"), (2053, 4093, "float16"), (7168, 2048, "bfloat16")],
)
def test_add_rms_norm_dynamic_quant_stage_a_shape(d_dim: int, m_rows: int, dt: str) -> None:
    """Verify AddRMSNormDynamicQuant Stage-A legality across aligned and tail shapes."""
    cfg = rms_kernels._select_add_rms_norm_dynamic_quant_config(
        d_dim=d_dim, tl_dtype=dt, m_rows=m_rows, schedule_mode="tuned"
    )
    prog = rms_kernels._build_add_rms_norm_dynamic_quant_prim_func(**cfg)
    _assert_stage_a_legal(prog, dict(cfg["pass_configs"]), list(cfg["compile_flags"]), f"rms_{d_dim}")


def test_attention_residual_saturation_and_overrides() -> None:
    """Verify AttentionResidual 2D saturation and knob overrides."""
    for m_rows, d_dim, min_tasks in ((1, 4096, 16), (4, 4096, 56), (16, 3584, 56), (32, 2560, 56)):
        cfg = ar_kernels._select_ar_small_decode_config(
            n_depth=4, d_dim=d_dim, tl_dtype="bfloat16", schedule_mode="tuned", m_rows=m_rows
        )
        d_sub = int(cfg["d_subtile"])
        assert d_sub >= 256 and d_sub % 256 == 0 and m_rows * ((d_dim + d_sub - 1) // d_sub) >= min_tasks
        assert int(cfg["tile_m"]) == 1 and int(cfg["num_stages"]) == 2


@pytest.mark.parametrize("regime", ["small_m1", "small_tail", "unaligned", "aligned"])
def test_attention_residual_stage_a_regime(regime: str) -> None:
    """Verify AttentionResidual Stage-A legality across all regimes."""
    regimes = {
        "small_m1": (ar_kernels._select_ar_small_decode_config, ar_kernels._build_ar_small_decode_func, {"n_depth": 2, "d_dim": 4096, "m_rows": 1}, {"N": 2, "B": 1, "S": 1, "D": 4096}),
        "small_tail": (ar_kernels._select_ar_small_decode_config, ar_kernels._build_ar_small_decode_func, {"n_depth": 4, "d_dim": 3584, "m_rows": 16}, None),
        "unaligned": (ar_kernels._select_ar_unaligned_config, ar_kernels._build_ar_unaligned_func, {"n_depth": 3, "d_dim": 2047, "m_rows": 333}, {"N": 8, "B": 1, "S": 16, "D": 1000}),
        "aligned": (ar_kernels._select_ar_aligned_config, ar_kernels._build_ar_aligned_func, {"n_depth": 8, "d_dim": 4096, "m_rows": 8192}, {"N": 8, "B": 1, "S": 4096, "D": 4096}),
    }
    sel, bld, kw, s_dims = regimes[regime]
    if s_dims is not None:
        ar_src = (REPO_ROOT / "examples/ascend950pr_ops/attention_residual/kernels.py").read_text("utf-8")
        res = run_stage_a_static(ar_src, factory_name=bld.__name__, config_selector_name=sel.__name__, dims=s_dims)
        assert res.passed, f"{regime}: {res.error_message}"
    else:
        cfg = sel(**kw, schedule_mode="tuned")
        _assert_stage_a_legal(bld(**kw, **cfg, schedule_mode="tuned"), dict(cfg["pass_configs"]), list(cfg["compile_flags"]), regime)
