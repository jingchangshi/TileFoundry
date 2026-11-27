"""Consolidated L0 tests for PyTorch SIL frontend, physics ratios, blueprints, and verifier."""

from __future__ import annotations

import copy
import dataclasses
import importlib.util
import json
import math
import re
import shutil
import time
from collections.abc import Mapping
from dataclasses import fields, replace
from pathlib import Path
from textwrap import dedent

import pytest
import torch
import yaml

from tilefoundry_ascend.agent.campaign import marathon
from tilefoundry_ascend.agent.campaign.marathon import MarathonConfig, build_marathon_arg_parser
from tilefoundry_ascend.agent.specs.anchor_specs import extract_schema_driven_case_dims
from tilefoundry_ascend.ascend.passes.hir_tir_bridge import BlueprintProvenance, HirBlueprintPacket, HirRooflineSummary
from tilefoundry_ascend.ascend.passes.physical_ledger import (
    ObjectivePhysicalLedger,
    build_objective_physical_ledger,
    build_scope_waterline,
)
from tilefoundry_ascend.ascend.schedule.regime_solver import (
    SOLVER_MODE_ALGEBRAIC,
    SOLVER_MODE_Z3,
    RegimePiecewiseSolver,
    RegimePlanResult,
    normalize_solver_mode,
    recompute_regime_for_live_buffers,
)
from tilefoundry_ascend.ascend.sil import (
    DEFAULT_OPS_ROOT,
    VALID_COMPUTE_ENGINES,
    VALID_ITERATION_REGULARITIES,
    VALID_MEMORY_RINGS,
    OpAdvisoryExtractor,
)
from tilefoundry_ascend.ascend.sil.pytorch_frontend import (
    PyTorchOpSemantics,
    PyTorchSemanticAnalyzer,
    RepresentativeCase,
)
from tilefoundry_ascend.ascend.sil.schema import (
    CasePhysicsKind,
    CasePhysicsRatios,
    CasePhysicsRegimeCase,
    CasePhysicsRegimeGroup,
    ConstraintRelaxationMask,
    RegimeSlice,
    TopologyAxes,
    case_workload_bytes,
    flatten_int_shapes,
    has_empty_group_attrs,
    partition_case_regimes,
    select_case_regime_representative,
    shape_numel,
)
from tilefoundry_ascend.ascend.sil.verifier import SILVerifier, verify_tilelang_ast
from tilefoundry_ascend.ascend.target import (
    TILELANG_DSL_SURFACE,
    AscendSpaceFacts,
    AscendTarget,
    load_cann_surface,
)
from tilefoundry_ascend.ascend.target.dialect_surface import filter_primitives_on_demand
from tilefoundry_ascend.ascend.templates import registry as template_registry
from tilefoundry_ascend.ascend.templates.registry import unlock_cross_topology_paradigm_shift

_REPO_ROOT = Path(__file__).resolve().parents[3]
_AR_KERNELS_PATH = _REPO_ROOT / "examples" / "ascend950pr_ops" / "attention_residual" / "kernels.py"
_SEED_RULES_PATH = _REPO_ROOT / "targets" / "ascend" / "tilefoundry_ascend" / "agent" / "kb" / "seed_rules.yaml"

_STOCK_OPS: tuple[tuple[str, str], ...] = (
    ("level1", "swi_glu"),
    ("level2", "rms_norm"),
    ("level2", "softmax"),
    ("level2", "apply_rotary_pos_emb"),
    ("level3", "add_rms_norm_dynamic_quant"),
    ("level3", "mhc_sinkhorn"),
    ("level3", "grouped_matmul"),
    ("level3", "q4_0_dequant_matmul"),
    ("level4", "attention_residual"),
    ("level4", "mla"),
    ("level4", "gqa"),
    ("level4", "dynamic_mx_quant"),
    ("level4", "sparse_flash_attention"),
    ("level5", "kimi_delta_attention"),
    ("level5", "tensor_program_vm"),
)
_STOCK_IDS = tuple(name for _, name in _STOCK_OPS)

_EXPECTED_SEMANTICS: dict[
    str, tuple[bool, bool, tuple[int, ...], bool, bool, bool, bool, bool, bool]
] = {
    "swi_glu": (False, False, (), False, False, False, False, False, False),
    "rms_norm": (False, True, (), False, False, False, False, False, True),
    "softmax": (False, True, (), True, False, False, False, False, True),
    "apply_rotary_pos_emb": (False, False, (), False, False, False, False, True, False),
    "add_rms_norm_dynamic_quant": (False, True, (-1,), False, False, False, False, True, True),
    "mhc_sinkhorn": (False, True, (-2, -1), True, False, False, True, True, True),
    "grouped_matmul": (True, False, (), False, False, False, True, False, False),
    "q4_0_dequant_matmul": (True, False, (), False, False, True, False, False, False),
    "attention_residual": (True, True, (-1, 0), True, False, False, False, False, True),
    "mla": (True, True, (-1,), True, False, False, True, False, True),
    "gqa": (True, True, (-1,), True, False, False, True, False, True),
    "dynamic_mx_quant": (False, True, (-1,), False, False, True, False, True, False),
    "sparse_flash_attention": (True, True, (), False, True, False, True, False, False),
    "kimi_delta_attention": (False, True, (-2, -1), True, False, False, True, False, False),
    "tensor_program_vm": (False, True, (), False, False, False, True, False, False),
}

_EXPECTED_AXES: dict[str, str] = {
    "swi_glu": "aiv_only/gm_ub_stream/affine_simd",
    "rms_norm": "aiv_only/gm_ub_stream/affine_simd",
    "softmax": "aiv_only/gm_ub_stream/affine_simd",
    "apply_rotary_pos_emb": "aiv_only/gm_ub_stream/affine_simd",
    "add_rms_norm_dynamic_quant": "aiv_only/gm_ub_stream/affine_simd",
    "mhc_sinkhorn": "aiv_only/persistent_onchip_state/affine_simd",
    "grouped_matmul": "coupled_1c2v/l0c_ub_epilogue/affine_simd",
    "q4_0_dequant_matmul": "coupled_1c2v/l0c_ub_epilogue/affine_simd",
    "attention_residual": "aiv_only/gm_ub_stream/affine_simd",
    "mla": "coupled_1c2v/bidirectional_cv_ring/affine_simd",
    "gqa": "coupled_1c2v/bidirectional_cv_ring/affine_simd",
    "dynamic_mx_quant": "aiv_only/gm_ub_stream/affine_simd",
    "sparse_flash_attention": "coupled_1c2v/l0c_ub_epilogue/irregular_simt",
    "kimi_delta_attention": "aiv_only/persistent_onchip_state/affine_simd",
    "tensor_program_vm": "aiv_only/persistent_onchip_state/affine_simd",
}

_AXES_AIV = TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd")
_AXES_CUBE = TopologyAxes("aic_only", "gm_l1_l0_fixpipe", "affine_simd")
_AXES_COUPLED = TopologyAxes("coupled_1c2v", "l0c_ub_epilogue", "affine_simd")
_AXES_PERSISTENT = TopologyAxes("coupled_1c2v", "persistent_onchip_state", "affine_simd")
_AXES_PERSISTENT_AIV = TopologyAxes("aiv_only", "persistent_onchip_state", "affine_simd")


def _yaml_mapping(path: Path) -> dict[str, object]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return {str(k): v for k, v in raw.items()}


def _op_path(op_name: str) -> Path:
    level = next(lvl for lvl, name in _STOCK_OPS if name == op_name)
    return DEFAULT_OPS_ROOT / level / op_name


def _load_op(op_name: str) -> tuple[str, dict[str, object], tuple[dict[str, object], ...]]:
    op_dir = _op_path(op_name)
    source = (op_dir / "golden.py").read_text(encoding="utf-8")
    proto = _yaml_mapping(op_dir / "proto.yaml")
    cases = tuple(_yaml_mapping(op_dir / "cases.yaml")["cases"])
    return source, proto, cases


def _schema_dim_ints(case: Mapping[str, object], proto: Mapping[str, object]) -> dict[str, int]:
    raw_inputs = proto.get("inputs")
    proto_inputs = tuple(raw_inputs) if isinstance(raw_inputs, list) else ()
    raw_dims = extract_schema_driven_case_dims(case, proto_inputs=proto_inputs)
    return {str(k): v for k, v in raw_dims.items() if isinstance(v, int) and not isinstance(v, bool)}


def _scrub_proto_identity(proto: Mapping[str, object], alias: str) -> dict[str, object]:
    copied = copy.deepcopy(dict(proto))
    for key in ("id", "name", "op_name", "cannbot_task", "schema"):
        if key in copied:
            copied[key] = alias
    if isinstance(copied.get("operator"), dict):
        for key in ("id", "name", "schema"):
            if key in copied["operator"]:
                copied["operator"][key] = alias
    return copied


def _ratio(**kw: object) -> CasePhysicsRatios:
    defaults: dict[str, object] = {
        "case_id": 1,
        "stride_ratio": 1.0,
        "contig_atom_ratio": 16.0,
        "is_atom_aligned": True,
        "working_set_sram_ratio": 0.25,
        "working_set_vreg_ratio": 0.5,
        "wave_occupancy_ratio": 1.0,
        "oi_ridge_ratio": 1.0,
        "inner_contiguous_bytes": 32,
        "has_empty_group": False,
    }
    defaults.update(kw)
    return CasePhysicsRatios(**defaults)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def analyzer() -> PyTorchSemanticAnalyzer:
    if not DEFAULT_OPS_ROOT.is_dir():
        pytest.skip(f"cann-bench tasks root missing: {DEFAULT_OPS_ROOT}")
    return PyTorchSemanticAnalyzer()


@pytest.fixture(scope="module")
def extractor() -> OpAdvisoryExtractor:
    if not DEFAULT_OPS_ROOT.is_dir():
        pytest.skip(f"cann-bench tasks root missing: {DEFAULT_OPS_ROOT}")
    return OpAdvisoryExtractor(DEFAULT_OPS_ROOT)


@pytest.mark.parametrize("op_name", _STOCK_IDS)
def test_stock_operators_semantics_obfuscation_ratios_and_regimes(
    analyzer: PyTorchSemanticAnalyzer,
    extractor: OpAdvisoryExtractor,
    op_name: str,
) -> None:
    """Verify AST semantics, obfuscation invariance, 20-case physics ratios, and regimes."""
    assert (_op_path(op_name) / "golden.py").is_file()
    source, proto, cases = _load_op(op_name)
    sem = analyzer.analyze_source(source, entry_name=op_name, proto=proto)
    assert (
        sem.has_matmul,
        sem.has_reduction,
        sem.reduction_axes,
        sem.has_online_exp_max,
        sem.has_indirect_index,
        sem.has_subbyte_or_block_quant,
        sem.has_recurrent_loop,
        sem.mutates_inputs,
        sem.preserves_input_shape,
    ) == _EXPECTED_SEMANTICS[op_name]
    assert sem.base_axes.key() == _EXPECTED_AXES[op_name]

    alias = "anon_entry"
    obf_src, reps = re.subn(rf"^def\s+{re.escape(op_name)}\s*\(", f"def {alias}(", source, flags=re.M)
    assert reps == 1
    obf = analyzer.analyze_source(obf_src, entry_name=alias, proto=_scrub_proto_identity(proto, alias))
    assert obf.base_axes == sem.base_axes and obf.preserves_input_shape == sem.preserves_input_shape

    assert len(cases) == 20
    reps_by_sig: dict[tuple[str, ...], tuple[int, dict[str, int]]] = {}
    for case in cases:
        ratios = analyzer.compute_case_ratios(sem, case)
        assert analyzer.compute_case_ratios(obf, case).to_dict() == ratios.to_dict()
        assert ratios.stride_ratio >= 1.0 and ratios.contig_atom_ratio > 0.0
        sig = ratios.phase_signature(sem.base_axes)
        assert sig[0] in ("aic_cube", "coupled_1c2v", "aiv_vector_or_gemv")
        eff = ratios.effective_axes(sem.base_axes)
        assert eff.compute_engine in VALID_COMPUTE_ENGINES
        assert eff.memory_ring in VALID_MEMORY_RINGS
        assert eff.iteration_regularity in VALID_ITERATION_REGULARITIES
        assert CasePhysicsRatios.from_dict(ratios.to_dict()) == ratios
        shapes = flatten_int_shapes(case.get("input_shape"))
        scale = shape_numel(shapes[0]) if shapes else 1
        dims = _schema_dim_ints(case, proto)
        if sig not in reps_by_sig or scale > reps_by_sig[sig][0]:
            reps_by_sig[sig] = (scale, dims)

    for _scale, dims in reps_by_sig.values():
        pkt = analyzer.build_universal_blueprint(sem, dims)
        rf = pkt.roofline
        assert rf is not None and rf.compute_flops == rf.cube_flops + rf.vector_flops > 0
        assert rf.ideal_roofline_us == max(rf.compute_floor_us, rf.memory_floor_us) > 0.0

    level = next(lvl for lvl, name in _STOCK_OPS if name == op_name)
    adv = extractor.extract(op_name, level=level)
    assert 2 <= len(adv.regimes) <= 4
    covered = [cid for reg in adv.regimes for cid in reg.case_nums]
    assert sorted(covered) == list(range(1, 21))


def test_ast_reachability_shape_preservation_and_quant_distinction(
    analyzer: PyTorchSemanticAnalyzer,
) -> None:
    """Follow reachable helpers only, reject dead/string clues, and distinguish int8 vs quant."""
    dead = dedent(
        """
        import torch
        def flash_attention_q4_quant_matmul(x):
            \"\"\"Softmax int4 quantized matmul with indirect gather.\"\"\"
            note = "mhc sinkhorn softmax matmul int4"
            return x + 1.0
        def dead_matmul(x, w):
            return torch.matmul(x, w)
        """
    )
    sem_dead = analyzer.analyze_source(
        dead, entry_name="flash_attention_q4_quant_matmul", proto={"description": "int4 matmul"}
    )
    assert not any((sem_dead.has_matmul, sem_dead.has_reduction, sem_dead.has_subbyte_or_block_quant))

    nested = dedent(
        """
        import torch
        def entry(x):
            def inner(y):
                return torch.sum(y, dim=1)
            return inner(x)
        def dead_outer(x):
            return torch.matmul(x, x)
        """
    )
    sem_nested = analyzer.analyze_source(nested, entry_name="entry")
    assert sem_nested.has_reduction is True and sem_nested.reduction_axes == (1,)
    assert sem_nested.has_matmul is False

    norm_src = dedent(
        """
        import torch
        def normalized(x):
            mean = x.mean(dim=-1, keepdim=True)
            centered = x - mean
            var = (centered * centered).mean(dim=-1, keepdim=True)
            return centered / torch.sqrt(var + 1e-5)
        """
    )
    assert analyzer.analyze_source(norm_src, entry_name="normalized").preserves_input_shape is True
    assert (
        analyzer.analyze_source(
            "import torch\ndef f(x):\n    return torch.softmax(x, dim=-1).sum(dim=-1)\n",
            entry_name="f",
        ).preserves_input_shape
        is False
    )
    assert (
        analyzer.analyze_source(
            "import torch\ndef f(x, idx):\n    if idx.dtype != torch.int64:\n        idx = idx.long()\n    return torch.gather(x, 1, idx)\n",
            entry_name="f",
        ).preserves_input_shape
        is False
    )

    int8_src = dedent(
        """
        import torch
        def per_token_int8(x: torch.Tensor):
            s = x.abs().amax(dim=-1, keepdim=True) / 127.0
            return torch.clamp((x / s).round(), -128, 127).to(torch.int8), s.squeeze(-1)
        """
    )
    assert analyzer.analyze_source(int8_src, entry_name="per_token_int8").has_subbyte_or_block_quant is False
    for lvl, op_name in (("level3", "q4_0_dequant_matmul"), ("level3", "dequant_swiglu_quant")):
        op_dir = DEFAULT_OPS_ROOT / lvl / op_name
        if op_dir.is_dir():
            s = analyzer.analyze_source(
                (op_dir / "golden.py").read_text(encoding="utf-8"),
                entry_name=op_name,
                proto=_yaml_mapping(op_dir / "proto.yaml"),
            )
            assert s.has_subbyte_or_block_quant is True


def test_blueprint_dtype_pricing_missing_dims_and_clustering(
    analyzer: PyTorchSemanticAnalyzer,
    extractor: OpAdvisoryExtractor,
) -> None:
    """Verify exact per-tensor dtype pricing, missing anchor validation, and phase clustering."""
    gmm_src, gmm_proto, gmm_cases = _load_op("grouped_matmul")
    gmm_sem = analyzer.analyze_source(gmm_src, entry_name="grouped_matmul", proto=gmm_proto)
    rep = RepresentativeCase.from_case_entry(next(c for c in gmm_cases if c["case_id"] == 7))
    assert rep.input_dtypes == ("bfloat16", "bfloat16", "float32")
    with pytest.raises(ValueError, match="positive integer case_id"):
        RepresentativeCase.from_case_entry({"input_shape": [[4, 8]], "dtype": ["float16"]})

    dims = {"M": 1, "N": 8192, "K": 8192}
    hw = RepresentativeCase(
        case_id=7,
        input_shapes=((1, 8192), (1, 8192, 8192)),
        input_dtypes=("float16", "float32"),
        output_dtype="float32",
    )
    sw = replace(hw, input_dtypes=("float32", "float16"))
    p_hw = analyzer.build_universal_blueprint(gmm_sem, dims, representative_case=hw)
    p_sw = analyzer.build_universal_blueprint(gmm_sem, dims, representative_case=sw)
    assert p_hw.roofline.min_gm_bytes == 268_484_608
    assert p_sw.roofline.min_gm_bytes == 134_283_264
    assert p_hw.provenance.traffic_basis == "declared_io_exact"
    assert p_sw.roofline.ideal_roofline_us < p_hw.roofline.ideal_roofline_us

    q4_src, q4_proto, _ = _load_op("q4_0_dequant_matmul")
    q4_sem = analyzer.analyze_source(q4_src, entry_name="q4_0_dequant_matmul", proto=q4_proto)
    q4_pkt = analyzer.build_universal_blueprint(q4_sem, {"in_features": 160, "out_features": 512, "total_elements": 160})
    assert q4_pkt.roofline.resolved_hir_dims == {"M": 1, "N": 512, "K": 160, "B": 1, "E": 1}
    with pytest.raises(ValueError, match="matmul analytical Roofline requires M"):
        analyzer.build_universal_blueprint(q4_sem, {"in_features": 160, "out_features": 512})

    mla_src, mla_proto, _ = _load_op("mla")
    mla_sem = analyzer.analyze_source(mla_src, entry_name="mla", proto=mla_proto)
    with pytest.raises(ValueError, match="attention requires one of anchor dims"):
        analyzer.build_universal_blueprint(mla_sem, {"B": 2})

    sm_tags = {cid: r.regime_tag for r in extractor.extract("softmax", level="level2").regimes for cid in r.case_nums}
    assert set(sm_tags.values()) == {"reduce_tail", "reduce_tail_subatom", "reduce_strided", "reduce_strided_twopass"}
    q4_tags: dict[str, set[int]] = {}
    for r in extractor.extract("q4_0_dequant_matmul", level="level3").regimes:
        q4_tags.setdefault(r.regime_tag, set()).update(r.case_nums)
    assert q4_tags == {"gemv_small": {1, 2, 3, 4, 5, 19, 20}, "gemv": {6, 7, 9}, "mixcore": {8, 10, 11, 12, 13, 14, 15, 16, 17, 18}}
    mla_tags: dict[str, set[int]] = {}
    for r in extractor.extract("mla", level="level4").regimes:
        mla_tags.setdefault(r.regime_tag, set()).update(r.case_nums)
    assert mla_tags == {"prefill_split": {1, 2, 5, 6, 13, 14, 15, 20}, "prefill": {3, 4}, "decode_split": {7, 8, 17, 18, 19}, "decode": {9, 10, 11, 12, 16}}


def test_phase_band_transitions_partition_and_validation() -> None:
    """Verify phase band thresholds, effective axes demotions, partitioning, and schema errors."""
    assert _ratio(wave_occupancy_ratio=0.5).phase_signature(_AXES_COUPLED)[4] == "balanced_waves"
    assert _ratio(wave_occupancy_ratio=0.499).phase_signature(_AXES_COUPLED)[4] == "under_occupied_split"
    assert _ratio(wave_occupancy_ratio=0.0, has_empty_group=True).phase_signature(_AXES_COUPLED)[4] == "ragged_groups"
    assert _ratio(oi_ridge_ratio=0.25).phase_signature(_AXES_COUPLED)[0] == "coupled_1c2v"
    assert _ratio(oi_ridge_ratio=0.24).phase_signature(_AXES_COUPLED)[0] == "aiv_vector_or_gemv"
    assert _ratio(oi_ridge_ratio=0.25).phase_signature(_AXES_CUBE)[0] == "aic_cube"
    assert _ratio(working_set_vreg_ratio=1.0).phase_signature(_AXES_AIV)[2] == "vreg_resident"
    assert _ratio(working_set_vreg_ratio=1.1, working_set_sram_ratio=1.0).phase_signature(_AXES_AIV)[2] == "sram_1pass"
    assert _ratio(working_set_vreg_ratio=1.1, working_set_sram_ratio=1.1).phase_signature(_AXES_AIV)[2] == "gm_multipass"
    assert _ratio(contig_atom_ratio=0.99).phase_signature(_AXES_AIV)[3] == "sub_atom_simt"
    assert _ratio(contig_atom_ratio=1.0, is_atom_aligned=False).phase_signature(_AXES_AIV)[3] == "unaligned_predicated"
    assert _ratio(stride_ratio=2.0).phase_signature(_AXES_AIV)[1] == "strided_or_gather"

    assert _ratio(oi_ridge_ratio=0.1).effective_axes(_AXES_COUPLED) == _AXES_AIV
    assert _ratio(working_set_vreg_ratio=4.0, working_set_sram_ratio=8.0).effective_axes(_AXES_PERSISTENT) == _AXES_COUPLED
    assert _ratio(working_set_vreg_ratio=4.0, working_set_sram_ratio=8.0).effective_axes(_AXES_PERSISTENT_AIV) == _AXES_AIV
    assert case_workload_bytes(((2, 2), (3,)), ("int8", "float32")) == 16

    with pytest.raises(ValueError, match="stride_ratio must be >= 1.0"):
        _ratio(stride_ratio=0.5)
    with pytest.raises(ValueError, match="contig_atom_ratio must be > 0.0"):
        _ratio(contig_atom_ratio=0.0)

    r_wave = {1: _ratio(case_id=1, wave_occupancy_ratio=1.0), 2: _ratio(case_id=2, wave_occupancy_ratio=0.1)}
    g_wave = partition_case_regimes(
        r_wave, {1: {"rows": 8}, 2: {"rows": 8}}, {1: 1024, 2: 2048},
        kind="reduction", axes=_AXES_AIV, min_partitions=2, max_partitions=4,
    )
    assert [g.regime_tag for g in g_wave] == ["reduce_tail_wave_filled", "reduce_tail_wave_underfill"]

    r_work = {i: _ratio(case_id=i, wave_occupancy_ratio=1.0) for i in range(1, 5)}
    g_work = partition_case_regimes(
        r_work, {i: {"rows": 8} for i in range(1, 5)}, {1: 4096, 2: 512, 3: 2048, 4: 1024},
        kind="reduction", axes=_AXES_AIV, min_partitions=2, max_partitions=4,
    )
    assert {g.regime_tag for g in g_work} == {"reduce_tail_latency", "reduce_tail_throughput"}

    reg_slice = RegimeSlice(
        regime_tag="main", case_nums=(1,), representative_dims={"M": 128}, has_unaligned_tail=False, case_ratios=(_ratio(),), representative_case_id=1
    )
    assert RegimeSlice.from_json(reg_slice.to_json()) == reg_slice
    assert select_case_regime_representative(
        (
            CasePhysicsRegimeCase(
                case_id=1, dims={"M": 4}, phase_signature=_ratio().phase_signature(_AXES_AIV), workload_bytes=10
            ),
            CasePhysicsRegimeCase(
                case_id=2, dims={"M": 8}, phase_signature=_ratio().phase_signature(_AXES_AIV), workload_bytes=10
            ),
        )
    ).case_id == 2
    assert has_empty_group_attrs({}) is False


def test_semantic_first_blueprints_ledger_scope_and_caliper(tmp_path: Path) -> None:
    """Verify golden-only blueprint generation, GM traffic scope, and <=3-step solver caliper."""
    analyzer = PyTorchSemanticAnalyzer()
    mla_golden = dedent(
        """
        import torch
        def forward(q, k, v):
            return torch.matmul(torch.softmax(torch.matmul(q, k.transpose(-2, -1)), dim=-1), v)
        """
    )
    sem = analyzer.analyze_source(mla_golden, entry_name="forward")
    rep = RepresentativeCase(
        case_id=1,
        input_shapes=((4, 32, 256, 576), (4, 8, 4096, 576), (4, 8, 4096, 512)),
        input_dtypes=("bfloat16", "bfloat16", "bfloat16"),
        output_dtype="bfloat16",
    )
    pkt = analyzer.build_universal_blueprint(
        sem, {"B": 4, "S_q": 256, "S_kv": 4096, "N_q": 32, "N_kv": 8, "D_qk": 576, "D_v": 512}, representative_case=rep
    )
    assert pkt.provenance == BlueprintProvenance("pytorch_golden", "declared_io_exact", "representative_anchor")
    assert HirBlueprintPacket.from_dict(json.loads(json.dumps(pkt.to_dict()))) == pkt

    per_tile = build_objective_physical_ledger(
        analysis_mode="readonly_envelope", is_legal=True, steady_state_ii=16,
        pipe_cycles={"AIV0.PIPE_MTE2": 64}, waterlines={}, grid_blocks=28, active_grid_blocks=14,
        gm_roundtrip_bytes=65536, gm_traffic_scope="per_tile",
    )
    full_grid = build_objective_physical_ledger(
        analysis_mode="readonly_envelope", is_legal=True, steady_state_ii=16,
        pipe_cycles={"AIV0.PIPE_MTE2": 64}, waterlines={}, grid_blocks=28,
        gm_roundtrip_bytes=1_048_576, gm_traffic_scope="full_grid",
    )
    assert per_tile.total_gm_bytes() == 65536 * 14 and full_grid.total_gm_bytes(56) == 1_048_576
    assert ObjectivePhysicalLedger.from_dict(full_grid.to_dict()) == full_grid

    assert (SOLVER_MODE_ALGEBRAIC, SOLVER_MODE_Z3) == ("algebraic_piecewise", "z3_smt")
    assert normalize_solver_mode("analytical_piecewise") == "algebraic_piecewise"
    target = AscendTarget("ascend.950pr")
    facts = target.get_facts(AscendSpaceFacts)
    solver = RegimePiecewiseSolver(target)
    wide = solver.solve_regime(
        {"regime_tag": "opaque", "representative_dims": {"M": 64}},
        op_name="anon_reduce",
        topology_axes=_AXES_AIV,
        roofline=HirRooflineSummary({"M": 64, "D": 65536, "elements": 64 * 65536}, 64 * 65536 * 11, 64 * 65536 * 4, 2.75, 1.0, 10.0, 10.0, "memory"),
        input_dtypes=("bfloat16",),
    )
    assert wide.policy["requires_twopass_reduction"] is True and wide.caliper_steps == 0
    assert RegimePlanResult.from_dict(dict(wide.to_summary_dict(), solver_mode="analytical_piecewise")).solver_mode == SOLVER_MODE_ALGEBRAIC

    cube_base = solver.solve_regime({"regime_tag": "cube", "representative_dims": {"M": 256, "N": 256, "K": 128}}, op_name="mm", topology_axes=_AXES_COUPLED, uses_cube=True)
    ub_over = build_objective_physical_ledger(
        analysis_mode="readonly_envelope", is_legal=False, steady_state_ii=None, pipe_cycles={"AIV0.PIPE_V": 120},
        waterlines={
            "UB": build_scope_waterline(
                scope="UB",
                capacity_bytes=facts.ub_simdvf_usable_bytes,
                explicit_buffers={"buf": int(facts.ub_simdvf_usable_bytes * 3.2)},
                implicit_buffers={},
            )
        },
    )
    ub_cal = recompute_regime_for_live_buffers(solver, cube_base, ub_over)
    assert 1 <= ub_cal.caliper_steps <= 3 and ub_cal.caliper_action == "ub_halving"
    assert ub_cal.policy["ub_estimated_bytes"] <= facts.ub_simdvf_usable_bytes

    l1_over = build_objective_physical_ledger(
        analysis_mode="readonly_envelope", is_legal=False, steady_state_ii=None, pipe_cycles={"AIC.PIPE_M": 120},
        waterlines={
            "L1": build_scope_waterline(
                scope="L1",
                capacity_bytes=facts.l1_bytes,
                explicit_buffers={"a": int(facts.l1_bytes * 1.8)},
                implicit_buffers={},
            )
        },
    )
    l1_cal = recompute_regime_for_live_buffers(solver, dataclasses.replace(cube_base, policy=dict(cube_base.policy, l1_stages=3, l1_half_bank_split=True)), l1_over)
    assert 1 <= l1_cal.caliper_steps <= 3 and l1_cal.caliper_action == "l1_stage_backoff"

    if DEFAULT_OPS_ROOT.is_dir():
        for lvl, op in (("level1", "swi_glu"), ("level2", "softmax")):
            src_dir, dst_dir = DEFAULT_OPS_ROOT / lvl / op, tmp_path / lvl / op
            dst_dir.mkdir(parents=True)
            for fn in ("golden.py", "proto.yaml", "cases.yaml"):
                shutil.copy2(src_dir / fn, dst_dir / fn)
            s = analyzer.analyze_source((dst_dir / "golden.py").read_text(encoding="utf-8"), entry_name=op)
            assert analyzer.build_blueprint(s, {"M": 64, "D": 512, "total_elements": 64 * 512}).roofline.ideal_roofline_us > 0.0


def test_gemm_contraction_associative_recurrence_and_sil_verifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify GEMM contraction, associative recurrence, T.Stage verifier, and 7-facet unlock."""
    analyzer = PyTorchSemanticAnalyzer()
    ar_src = dedent(
        """
        import math, torch
        def attention_residual(v_stack, x, w, gamma, alpha, epsilon=1e-6):
            v = v_stack.to(torch.float32)
            rms = torch.sqrt(v.pow(2).mean(dim=-1, keepdim=True) + float(epsilon))
            s = torch.einsum("nbtd,d->nbt", v / rms * gamma.to(torch.float32), w.to(torch.float32)) / math.sqrt(v.shape[-1])
            agg = torch.einsum("nbt,nbtd->btd", torch.softmax(s, dim=0), v)
            return (1.0 - float(alpha)) * x.to(torch.float32) + float(alpha) * agg
        """
    )
    ar_sem = analyzer.analyze_source(ar_src, entry_name="attention_residual")
    assert ar_sem.has_gemm_contraction is False and ar_sem.base_axes.compute_engine == "aiv_only"
    gemm_sem = analyzer.analyze_source(
        "import torch\ndef batched_gemm(a, b):\n    return torch.einsum('bik,bkj->bij', a, b)\n",
        entry_name="batched_gemm",
    )
    assert gemm_sem.has_gemm_contraction is True and gemm_sem.base_axes.compute_engine in ("aic_only", "coupled_1c2v")

    kda_src = dedent(
        """
        import torch
        def rec(q, k, v, g, beta, init_state):
            mem = init_state.to(torch.float32).clone()
            out = torch.empty(v.shape, dtype=torch.float32, device=q.device)
            for t in range(q.shape[1]):
                faded = mem * g[:, t].unsqueeze(-1)
                pred = torch.sum(faded * k[:, t].unsqueeze(-1), dim=-2)
                mem = faded + k[:, t].unsqueeze(-1) * ((v[:, t] - pred) * beta[:, t].unsqueeze(-1)).unsqueeze(-2)
                out[:, t] = torch.sum(mem * q[:, t].unsqueeze(-1), dim=-2)
            return out, mem
        """
    )
    kda_sem = analyzer.analyze_source(kda_src, entry_name="rec")
    assert kda_sem.has_associative_outer_recurrence is True and kda_sem.base_axes.memory_ring == "persistent_onchip_state"
    rnn_sem = analyzer.analyze_source(
        "import torch\ndef rnn(x, w, h):\n    for t in range(x.shape[1]):\n        h = torch.tanh(w @ h)\n    return h\n",
        entry_name="rnn",
    )
    assert rnn_sem.has_recurrent_loop is True and rnn_sem.has_associative_outer_recurrence is False

    assert not hasattr(marathon, "build_prewarmed_l3_runner")
    assert "cost_model_mode" in {f.name for f in fields(MarathonConfig)}
    assert MarathonConfig(operator_name="mla").cost_model_mode == "off"
    assert build_marathon_arg_parser().parse_args(["run", "--operator", "mla", "--cost-model-mode", "advisory"]).cost_model_mode == "advisory"

    verifier = SILVerifier()
    t0 = time.monotonic()
    for bad, needle in (
        ("import tilelang.ascend.language as T\n@T.prim_func\ndef k(A, B):\n    for i in T.Parallel(1):\n        T.copy(A, B)\n", "T.copy"),
        ("import tilelang.ascend.language as T\n@T.prim_func\ndef k():\n    for n in T.Pipelined(4, num_stages=2):\n        with T.SimdVF():\n            with T.Stage(1):\n                pass\n", "T.SimdVF"),
        ("import tilelang.ascend.language as T\n@T.prim_func\ndef k():\n    for n in T.serial(4):\n        with T.Stage(1):\n            pass\n", "T.Stage(1)"),
    ):
        rep = verify_tilelang_ast(bad)
        assert not rep.passed and any(needle in err for err in rep.errors)
    assert verifier.verify_tilelang_ast(
        "import tilelang.ascend.language as T\n@T.prim_func\ndef k(A, B):\n    for n in T.Pipelined(4, num_stages=2):\n        with T.Stage(0):\n            T.copy(A, B)\n"
    ).passed is True
    assert (time.monotonic() - t0) * 1000.0 < 500.0

    spec = importlib.util.spec_from_file_location("_test_ar_k", str(_AR_KERNELS_PATH))
    assert spec is not None and spec.loader is not None
    ar_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ar_mod)
    ar_mod._KERNEL_CACHE.clear()
    ar_mod._DISPATCH_CACHE.clear()
    compile_calls: list[dict[str, object]] = []
    monkeypatch.setattr(ar_mod.tilelang, "compile", lambda _p, **kw: compile_calls.append(dict(kw)) or (lambda *_a: torch.zeros((1, 256), dtype=torch.bfloat16)))
    ar_mod._get_kernel(2, 256, "bfloat16", 1, "tuned", pass_configs=["--cce-disable-vf-stack-reserved-ubuf"])
    assert compile_calls[0]["pass_configs"] is None
    assert "--cce-disable-vf-stack-reserved-ubuf" in (compile_calls[0]["compile_flags"] or [])
    small_cfg = ar_mod._select_ar_small_decode_config(n_depth=2, d_dim=4096, tl_dtype="bfloat16", schedule_mode="tuned", m_rows=1)
    assert "--cce-disable-vf-stack-reserved-ubuf" not in small_cfg["compile_flags"]
    large_cfg = ar_mod._select_ar_aligned_config(n_depth=8, d_dim=4096, tl_dtype="bfloat16", schedule_mode="tuned", m_rows=2048)
    assert "--cce-disable-vf-stack-reserved-ubuf" in large_cfg["compile_flags"]

    assert "unlock_cross_topology_paradigm_shift" in template_registry.__all__
    assert unlock_cross_topology_paradigm_shift(ConstraintRelaxationMask.bottom()) == ConstraintRelaxationMask.top()
    assert len(yaml.safe_load(_SEED_RULES_PATH.read_text(encoding="utf-8"))["rules"]) >= 41
    by_sym_93 = {e.symbol: e for e in load_cann_surface("cann_9_3").primitives}
    assert by_sym_93["T.Stage"].facet == "pipeline_and_sync_graph"
    assert "AscendMGroupedTileScheduler" in {
        e.symbol for e in filter_primitives_on_demand(
            TILELANG_DSL_SURFACE,
            topology_axes=TopologyAxes("coupled_1c2v", "l0c_ub_epilogue", "ragged_group"),
            freedom_mask=ConstraintRelaxationMask.bottom(),
        )
    }
