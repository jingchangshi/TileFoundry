"""Unit tests for CANNBench YAML parsing, CannbenchOpsRegistry, and anchor specs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from tilefoundry_ascend.agent.specs.anchor_specs import (
    AnchorCaseSpec,
    CannbenchOpsRegistry,
    GuardrailCaseSpec,
    PerfTargetSpec,
    RegimeTargetSpec,
    append_dynamic_anchor,
    build_anchor_var_bindings,
    build_semantic_regime_groups,
    build_semantic_regime_specs,
    campaign_case_regime_tag,
    compute_case_aligned_eta,
    compute_case_eq3_score,
    discover_module_trackb_symbols,
    extract_schema_driven_case_dims,
    extract_trackb_cpp_source,
    get_default_anchors,
    get_default_perf_target,
    get_input_domain_envelope,
    get_operator_entrypoint,
    get_operator_module_relpath,
    get_operator_task,
    infer_module_runtime_backend,
    inject_lake_counterexample_anchors,
    load_anchor_harness_and_cases,
    lookup_case_dims,
    lookup_case_metadata,
    parse_simple_env_file,
    resolve_cannbench_repo,
    resolve_cannbench_tasks_root,
    resolve_regime_for_case,
    semantic_regime_representative,
    semantic_regime_tag,
)
from tilefoundry_ascend.ascend.cost_model.experience_lake import append_counterexample_boundary
from tilefoundry_ascend.ascend.sil.pytorch_frontend import PyTorchSemanticAnalyzer
from tilefoundry_ascend.ascend.sil.schema import CasePhysicsRatios, TopologyAxes


def _write_synthetic_op(
    op_dir: Path,
    *,
    op_name: str,
    golden_src: str,
    cases: list[dict[str, object]],
    proto: dict[str, object] | None = None,
) -> None:
    op_dir.mkdir(parents=True, exist_ok=True)
    proto_doc = proto or {
        "operator": {
            "name": op_name,
            "category": "Vector",
            "inputs": [{"name": "x", "shape": "[M, D]", "dtype": ["float16"]}],
            "outputs": [{"name": "y", "shape": "[M, D]", "dtype": ["float16"]}],
        }
    }
    (op_dir / "proto.yaml").write_text(yaml.safe_dump(proto_doc), encoding="utf-8")
    (op_dir / "cases.yaml").write_text(yaml.safe_dump({"cases": cases}), encoding="utf-8")
    (op_dir / "golden.py").write_text(golden_src, encoding="utf-8")


def _make_manifest_tasks_root(root: Path) -> tuple[Path, Path]:
    src_root = root / "repo"
    tasks_root = root / "cann-bench" / "tasks"
    tasks_root.mkdir(parents=True, exist_ok=True)
    (tasks_root / "metadata").mkdir(parents=True, exist_ok=True)

    vec_golden = "import torch\ndef custom_vec(x):\n    return torch.relu(x)\n"
    vec_cases = [
        {"case_id": 1, "input_shape": [[1, 64]], "dtype": ["float16"]},
        {"case_id": 2, "input_shape": [[8, 97]], "dtype": ["float16"]},
        {"case_id": 3, "input_shape": [[64, 256]], "dtype": ["float16"]},
        {"case_id": 4, "input_shape": [[512, 1024]], "dtype": ["float16"]},
        {"case_id": 5, "input_shape": [[2048, 2048]], "dtype": ["float16"]},
    ]
    _write_synthetic_op(
        tasks_root / "level1" / "custom_vec",
        op_name="custom_vec",
        golden_src=vec_golden,
        cases=vec_cases,
    )

    kda_golden = "import torch\ndef demo_kda(q, k, v):\n    return q + k + v\n"
    kda_cases = [
        {
            "case_id": 1,
            "input_shape": [[1, 64, 16, 128], [1, 64, 16, 128], [1, 64, 16, 128]],
            "dtype": ["float16", "float16", "float16"],
        },
        {
            "case_id": 2,
            "input_shape": [[1, 64, 16, 256], [1, 64, 16, 256], [1, 64, 16, 256]],
            "dtype": ["float16", "float16", "float16"],
        },
    ]
    kda_proto = {
        "operator": {
            "name": "demo_kda",
            "category": "FusedComposite",
            "inputs": [{"name": "q"}, {"name": "k"}, {"name": "v"}],
            "outputs": [{"name": "y"}],
        }
    }
    _write_synthetic_op(
        tasks_root / "level4" / "demo_kda",
        op_name="demo_kda",
        golden_src=kda_golden,
        cases=kda_cases,
        proto=kda_proto,
    )

    manifest = {
        "operators": [
            {"id": "custom_vec", "path": "tasks/level1/custom_vec", "level": "L1"},
            {"id": "demo_kda", "path": "tasks/level4/demo_kda", "level": "L4"},
        ]
    }
    (tasks_root / "suite_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    meta = {
        "level1": {
            "custom_vec": {
                "1": {"baseline_perf_us": 4.0, "t_hw_us": 1.0},
                "2": {"baseline_perf_us": 10.0, "t_hw_us": 2.5},
                "3": {"baseline_perf_us": 20.0, "t_hw_us": 5.0},
                "4": {"baseline_perf_us": 80.0, "t_hw_us": 20.0},
                "5": {"baseline_perf_us": 200.0, "t_hw_us": 50.0},
            }
        },
        "level4": {
            "demo_kda": {
                "1": {"baseline_perf_us": 30.0, "t_hw_us": 10.0},
                "2": {"baseline_perf_us": 60.0, "t_hw_us": 20.0},
            }
        },
    }
    (tasks_root / "metadata" / "950pr.json").write_text(json.dumps(meta), encoding="utf-8")

    mod_dir = src_root / "examples" / "ascend950pr_ops" / "demo_kda"
    mod_dir.mkdir(parents=True, exist_ok=True)
    (mod_dir / "kernels.py").write_text(
        '''
_TRACK_B_SRC = r"""
#include <tl_templates/ascend/common.h>
__global__ __vector__ void kda_kernel() {}
"""

def _select_demo_kda_config(dk: int = 128, dv: int = 128):
    return {"chunk_t": 32}

def _build_demo_kda_prim_func(dk: int = 128, dv: int = 128):
    return None

def _render_trackb_tu(**dims):
    return _TRACK_B_SRC

def _render_demo_kda_trackb_launcher(**dims):
    return _TRACK_B_SRC

def _get_track_b_lib():
    return None

def _run_track_b(q, k, v):
    return _get_track_b_lib()

def demo_kda(q, k, v):
    dk = q.shape[-1]
    if dk <= 128 and v.shape[-1] <= 128:
        return _run_track_b(q, k, v)
    return tilelang.compile(None)
''',
        encoding="utf-8",
    )
    return src_root, tasks_root


def test_schema_driven_case_dims_dtypes_and_leak_guards() -> None:
    """Verify schema-driven dimension extraction across dtypes, layouts, and non-leakage."""
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
            {
                "case_id": 1,
                "input_shape": [[64, 128], [128, 256]],
                "dtype": [dtype_name, dtype_name],
            }
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

    norm_dims = extract_schema_driven_case_dims(
        {"case_id": 1, "input_shape": [[64, 128], [128]], "dtype": ["float16", "float16"]},
        proto_inputs=[{"name": "x", "shape": "[M, D]"}, {"name": "gamma", "shape": "[D]"}],
    )
    assert norm_dims["M"] == 64 and norm_dims["D"] == 128
    assert norm_dims["x_dim_0"] == 64 and norm_dims["gamma_dim_0"] == 128
    for leaked in ("K", "half_dim", "dk", "dv", "hc_mult"):
        assert leaked not in norm_dims

    mhc_dims = extract_schema_driven_case_dims(
        {"case_id": 1, "input_shape": [[4, 8, 8]], "dtype": ["float32"]},
        proto_inputs=[{"name": "comb", "shape": "[B, hc_mult, hc_mult]"}],
    )
    assert mhc_dims["hc_mult"] == 8

    rope_dims = extract_schema_driven_case_dims(
        {
            "case_id": 1,
            "input_shape": [[2, 16, 8, 64], [2, 16, 8, 64], [16, 32], [16, 32]],
            "dtype": ["float16", "float16", "float16", "float16"],
            "attrs": {"rotaryMode": "half"},
        },
        proto_inputs=[{"name": "query"}, {"name": "key"}, {"name": "cos"}, {"name": "sin"}],
    )
    assert rope_dims["half_dim"] == 32

    mla_dims = extract_schema_driven_case_dims(
        {
            "case_id": 7,
            "input_shape": [[1, 1, 128, 512], [1, 1, 128, 64], [1, 2048, 1, 512]],
            "dtype": ["float16", "float16", "float16"],
            "attrs": {"inputLayout": "BSND", "is_causal": True},
        },
        proto_inputs=[{"name": "q_nope"}, {"name": "q_rope"}, {"name": "kv_cache"}],
    )
    assert mla_dims["D_nope"] == 512
    assert mla_dims["D_rope"] == 64
    assert mla_dims["S_kv"] == 2048
    assert mla_dims["N_kv"] == 1
    assert mla_dims["is_causal"] is True

    bnsd_dims = extract_schema_driven_case_dims(
        {
            "case_id": 2,
            "input_shape": [[2, 8, 16, 64], [2, 2, 32, 64], [2, 2, 32, 64]],
            "dtype": ["float16", "float16", "float16"],
            "attrs": {"inputLayout": "BNSD", "dim": -1},
        },
        proto_inputs=[{"name": "query"}, {"name": "key"}, {"name": "value"}],
    )
    assert bnsd_dims["B"] == 2 and bnsd_dims["N_q"] == 8 and bnsd_dims["S"] == 16
    assert bnsd_dims["N_kv"] == 2 and bnsd_dims["dk"] == 64 and bnsd_dims["dv"] == 64
    assert bnsd_dims["inner"] == 1


def test_registry_discovery_stratification_and_semantic_regimes(tmp_path: Path) -> None:
    """Verify manifest/level/flat discovery, stratification, and hybrid regime selection."""
    src_root, tasks_root = _make_manifest_tasks_root(tmp_path)
    reg = CannbenchOpsRegistry(tasks_root=tasks_root, source_root=src_root)
    assert reg.tasks_root == tasks_root.resolve()
    assert reg.list_operators() == ("custom_vec", "demo_kda")
    assert reg.list_tasks() == ("level1/custom_vec", "level4/demo_kda")
    assert reg.has_operator("custom_vec") is True
    assert reg.has_operator("tasks/level1/custom_vec") is True
    assert reg.has_operator("unknown_op") is False

    vec_spec = reg.resolve("custom_vec")
    assert 4 <= len(vec_spec.anchors) <= 5
    tags = {a.regime_tag for a in vec_spec.anchors}
    assert {"small_latency", "unaligned_tail", "large_throughput"} <= tags
    assert vec_spec.perf_target.target_surrogate_score == pytest.approx(95.0)
    assert reg.get_perf_target("custom_vec") == vec_spec.perf_target
    assert reg.get_case_regime_tag("custom_vec", 1) is not None
    with pytest.raises(KeyError):
        reg.get_case_regime_tag("custom_vec", 999)

    pure_regimes = reg.get_regime_specs("demo_kda", pure_golden=True)
    assert all(r.runtime_backend == "tilelang_dsl" for r in pure_regimes)
    assert all(r.trackb_symbols == () for r in pure_regimes)

    hybrid_regimes = reg.get_regime_specs("demo_kda", pure_golden=False)
    by_dk = {int(r.representative_anchor_dims["q_dim_3"]): r for r in hybrid_regimes}
    assert by_dk[128].runtime_backend == "hybrid_trackb"
    assert "_render_trackb_tu" in by_dk[128].editable_symbols
    assert "_render_demo_kda_trackb_launcher" in by_dk[128].editable_symbols
    assert by_dk[256].runtime_backend == "tilelang_dsl"
    assert by_dk[256].trackb_symbols == ()

    level_root = tmp_path / "level_only"
    _write_synthetic_op(
        level_root / "level2" / "op_x",
        op_name="op_x",
        golden_src="import torch\ndef op_x(x):\n    return x\n",
        cases=[{"case_id": 1, "input_shape": [[32, 64]], "dtype": ["float16"]}],
    )
    level_reg = CannbenchOpsRegistry(tasks_root=level_root, source_root=src_root)
    assert level_reg.list_tasks() == ("level2/op_x",)

    flat_root = tmp_path / "flat_only"
    _write_synthetic_op(
        flat_root / "rms_norm",
        op_name="rms_norm",
        golden_src="import torch\ndef rms_norm(x):\n    return x\n",
        cases=[{"case_id": 1, "input_shape": [[32, 64]], "dtype": ["float16"]}],
    )
    flat_reg = CannbenchOpsRegistry(tasks_root=flat_root, source_root=src_root)
    assert flat_reg.list_tasks() == ("level2/rms_norm",)

    _write_synthetic_op(
        flat_root / "unregistered_flat_op",
        op_name="unregistered_flat_op",
        golden_src="import torch\ndef f(x):\n    return x\n",
        cases=[{"case_id": 1, "input_shape": [[8, 8]], "dtype": ["float16"]}],
    )
    with pytest.raises(ValueError, match="without an official task mapping"):
        CannbenchOpsRegistry(tasks_root=flat_root, source_root=src_root)


def test_trackb_symbol_discovery_and_runtime_backend_inference() -> None:
    """Verify AST Track-B C++20 symbol extraction and reachable backend classification."""
    src_dedent = '''
from textwrap import dedent
OP_A_TRACKB_CU = dedent("""
    #include <tl_templates/ascend/common.h>
    __global__ __vector__ void k() {}
""").strip()
def _select_op_a_trackb_plan():
    return {}
'''
    syms = discover_module_trackb_symbols(src_dedent)
    assert syms == ("OP_A_TRACKB_CU", "_select_op_a_trackb_plan")
    extracted = extract_trackb_cpp_source(src_dedent, syms)
    assert extracted is not None and extracted[0] == "OP_A_TRACKB_CU"
    assert "__global__ __vector__" in extracted[1]

    src_concat = '''
import textwrap
OP_B_CU = (
    textwrap.dedent("""
        #include "custom.h"
        __simd_vf__ void vf() {}
    """)
    + "\\n__global__ __aicore__ __vector__ void run() {}\\n"
)
def _unused_helper():
    return tilelang.compile(None)
def op_b():
    return _get_track_b_lib()
'''
    assert infer_module_runtime_backend(src_concat, "op_b") == ("ascendc_trackb", ("OP_B_CU",))

    hybrid_src = '''
OP_H_CU = "#include <tl_templates/ascend/common.h>\\n__aicore__ void h() {}"
def _helper():
    return OP_H_CU
def op_h(flag):
    if flag:
        return _helper()
    return tilelang.compile(None)
'''
    assert infer_module_runtime_backend(hybrid_src, "op_h") == ("hybrid_trackb", ("OP_H_CU",))
    pure_src = "def pure():\n    return tilelang.compile(None)\n"
    assert infer_module_runtime_backend(pure_src, "pure") == ("tilelang_dsl", ())


def test_semantic_regime_groups_tag_and_json_records() -> None:
    """Verify semantic regime grouping, phase tags, and JsonRecordMixin round-trips."""
    sig = (
        "aiv_vector_or_gemv",
        "contiguous",
        "sram_1pass",
        "unaligned_predicated",
        "balanced_waves",
    )
    assert semantic_regime_tag(sig) == "vec_contig_sram_unaligned_filled"

    golden = "import torch\ndef vec_op(x):\n    return torch.sum(x, dim=-1)\n"
    cases = [
        {"case_id": 1, "input_shape": [[1, 32]], "dtype": ["float16"]},
        {"case_id": 2, "input_shape": [[64, 97]], "dtype": ["float16"]},
        {"case_id": 3, "input_shape": [[1024, 4096]], "dtype": ["float16"]},
    ]
    proto = {"inputs": [{"name": "x"}], "outputs": [{"name": "y"}]}
    analyzer = PyTorchSemanticAnalyzer()
    sem = analyzer.analyze_source(golden, entry_name="vec_op", proto=proto)
    ratios = tuple(analyzer.compute_case_ratios(sem, c) for c in cases)
    groups = build_semantic_regime_groups(
        operator_name="vec_op",
        cases=cases,
        proto=proto,
        base_axes=sem.base_axes,
        has_reduction=sem.has_reduction,
        case_ratios=ratios,
    )
    assert len(groups) >= 2
    cid, rep_dims, _rep_sig = semantic_regime_representative(groups[0])
    assert cid == groups[0].representative_case_id
    assert "total_elements" in rep_dims
    tag = campaign_case_regime_tag(
        ratios[0], cases[0], base_axes=sem.base_axes, has_reduction=sem.has_reduction
    )
    assert isinstance(tag, str) and "_" in tag

    specs = build_semantic_regime_specs(
        operator_name="vec_op",
        entrypoint="vec_op",
        cases=cases,
        proto=proto,
        base_axes=sem.base_axes,
        has_reduction=sem.has_reduction,
        case_ratios=ratios,
        runtime_backend="tilelang_dsl",
    )
    assert len(specs) == len(groups)
    roundtrip_reg = RegimeTargetSpec.from_dict(specs[0].to_dict())
    assert roundtrip_reg.regime_tag == specs[0].regime_tag

    guardrail = GuardrailCaseSpec(
        "level4/mla", 7, "split_kv_decode", incumbent_ref_us=12.0, t_hw_us=2.0
    )
    assert guardrail.to_anchor_spec() == AnchorCaseSpec(
        "level4/mla", 7, "split_kv_decode", 12.0, 2.0
    )
    assert GuardrailCaseSpec.from_dict(guardrail.to_dict()) == guardrail
    perf = PerfTargetSpec("vec_op", 2.5, 1.3, 0.9, {1: 4.5, 2: 12.0}, 95.0, 3)
    assert PerfTargetSpec.from_dict(perf.to_dict()) == perf


def test_anchor_specs_helpers_env_and_lake_injection(tmp_path: Path) -> None:
    """Verify env parsing, metadata/dim lookup, regime resolution, scoring, and injection."""
    env_file = tmp_path / "env.sh"
    repo_root = tmp_path / "cann-bench"
    tasks_root = repo_root / "tasks"
    op_dir = tasks_root / "level4" / "attention_residual"
    _write_synthetic_op(
        op_dir,
        op_name="attention_residual",
        golden_src="import torch\ndef attention_residual(v):\n    return v\n",
        cases=[
            {"case_id": 2, "input_shape": [[2, 1, 1, 4096]], "dtype": ["bfloat16"]},
            {"case_id": 4, "input_shape": [[8, 1, 16, 1000]], "dtype": ["bfloat16"]},
            {"case_id": 15, "input_shape": [[8, 1, 32, 513]], "dtype": ["bfloat16"]},
        ],
        proto={
            "operator": {
                "name": "attention_residual",
                "inputs": [{"name": "v", "shape": "[N, B, S, D]", "dtype": ["bfloat16"]}],
                "outputs": [{"name": "y", "shape": "[B, S, D]", "dtype": ["bfloat16"]}],
            }
        },
    )
    (tasks_root / "metadata").mkdir(parents=True, exist_ok=True)
    meta = {
        "level4": {
            "attention_residual": {
                "2": {"baseline_perf_us": 1.5, "t_hw_us": 0.5},
                "4": {"baseline_perf_us": 9.93, "t_hw_us": 3.31},
                "15": {"baseline_perf_us": 18.0, "t_hw_us": 6.0},
            }
        }
    }
    (tasks_root / "metadata" / "950pr.json").write_text(json.dumps(meta), encoding="utf-8")
    env_file.write_text(
        f'export TILEFOUNDRY_NPU_CANNBENCH_REPO="{repo_root}"\n',
        encoding="utf-8",
    )
    assert parse_simple_env_file(env_file)["TILEFOUNDRY_NPU_CANNBENCH_REPO"] == str(repo_root)
    assert resolve_cannbench_repo(env_file=env_file) == repo_root.resolve()
    assert resolve_cannbench_tasks_root(repo_root) == tasks_root.resolve()
    assert lookup_case_metadata("attention_residual", 4, env_file=env_file) == pytest.approx(
        (9.93, 3.31)
    )
    dims4 = lookup_case_dims("attention_residual", 4, env_file=env_file)
    assert dims4["D"] == 1000

    ar_regimes = (
        RegimeTargetSpec("small_decode", ("_b0",), {}),
        RegimeTargetSpec("unaligned_d", ("_b1",), {}),
        RegimeTargetSpec("aligned_prefill", ("_b2",), {}),
    )
    assert (
        resolve_regime_for_case("attention_residual", {"B": 1, "S": 1, "D": 4096}, ar_regimes)
        .regime_tag
        == "small_decode"
    )
    assert (
        resolve_regime_for_case("attention_residual", {"B": 1, "S": 16, "D": 1000}, ar_regimes)
        .regime_tag
        == "unaligned_d"
    )
    assert (
        resolve_regime_for_case("attention_residual", {"B": 1, "S": 64, "D": 4096}, ar_regimes)
        .regime_tag
        == "aligned_prefill"
    )

    anchors = (AnchorCaseSpec("level4/attention_residual", 2, "small_decode", 1.5, 0.5),)
    appended = append_dynamic_anchor(
        anchors,
        operator_name="attention_residual",
        case_num=4,
        regime_tag="unaligned_d",
        env_file=env_file,
    )
    assert [a.case_num for a in appended] == [2, 4]
    assert (
        append_dynamic_anchor(appended, operator_name="attention_residual", case_num=2) == appended
    )

    lake_dir = tmp_path / "lake"
    for cid in (10, 11, 12, 13):
        append_counterexample_boundary(
            lake_dir,
            {
                "operator": "attention_residual",
                "regime_tag": "unaligned_d",
                "case_num": cid,
                "ref_perf_us": 50.0 + cid,
                "t_hw_us": 10.0 + cid,
            },
        )
    injected = inject_lake_counterexample_anchors(
        anchors,
        operator_name="attention_residual",
        regime_specs=ar_regimes,
        experience_lake_dir=lake_dir,
    )
    assert len(anchors) == 1
    assert [a.case_num for a in injected] == [2, 10, 11, 12, 13]

    other_lake = tmp_path / "cross_lake"
    append_counterexample_boundary(
        other_lake,
        {
            "operator": "other_op",
            "case_num": 99,
            "case_dims": {"D": 513},
            "topology_axes": {"compute_engine": "aiv_only"},
        },
    )
    cross_injected = inject_lake_counterexample_anchors(
        anchors,
        operator_name="attention_residual",
        regime_specs=ar_regimes,
        experience_lake_dir=other_lake,
        env_file=env_file,
        topology_axes={"compute_engine": "aiv_only"},
    )
    assert [a.case_num for a in cross_injected] == [2, 15]

    assert compute_case_eq3_score(10.0, 10.0, 2.0) == pytest.approx(0.5)
    assert compute_case_eq3_score(10.0, 0.0, 2.0) == pytest.approx(0.0)
    assert compute_case_aligned_eta(anchors[0], 2.0) == pytest.approx(0.25)
    assert compute_case_aligned_eta(anchors[0], 0.1) == pytest.approx(1.0)
    assert compute_case_aligned_eta(anchors[0], float("nan")) == pytest.approx(0.0)

    bindings = build_anchor_var_bindings(
        {"N": 8, "B": 2, "S": 16, "N_q": 16, "N_kv": 2}, kwargs={"tile_m": 4}
    )
    assert bindings["m_rows"] == 32
    assert bindings["q_total_rows"] == 512
    assert bindings["q_rows"] == 128
    assert bindings["v_rows"] == 256
    assert bindings["tile_m"] == 4

    env_block = get_input_domain_envelope("attention_residual")
    assert env_block is not None
    assert "INPUT DOMAIN & FAST-MATH ENVELOPE" in env_block.format_prompt_block()
    assert get_input_domain_envelope("unregistered") is None

    if (Path(__file__).resolve().parents[5] / "cann-bench" / "tasks").is_dir():
        live_reg = CannbenchOpsRegistry.default()
        assert len(live_reg.list_operators()) == 60
        assert get_operator_task("rms_norm") == "level2/rms_norm"
        assert get_operator_entrypoint("level2/rms_norm") == "rms_norm"
        assert (
            get_operator_module_relpath("rms_norm")
            == "examples/ascend950pr_ops/rms_norm/kernels.py"
        )
        assert len(get_default_anchors("level2/rms_norm")) == 5
        assert get_default_perf_target("rms_norm").target_surrogate_score >= 95.0
        harness, selected = load_anchor_harness_and_cases(
            "attention_residual",
            anchors=(AnchorCaseSpec("level4/attention_residual", 2, "small_decode", 1.5, 0.5),),
        )
        assert harness is not None and len(selected) == 1 and int(selected[0].case_num) == 2
