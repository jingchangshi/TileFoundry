"""Tests for the PyTorch semantic frontend, physics ratios, and analytical blueprints."""

from __future__ import annotations

import copy
import json
import math
import re
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from textwrap import dedent

import pytest
import yaml

from tilefoundry_ascend.agent.campaign.anchor_specs import extract_schema_driven_case_dims
from tilefoundry_ascend.ascend.schedule.regime_solver import RegimePiecewiseSolver
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
    RegimeSlice,
    TopologyAxes,
    case_workload_bytes,
    flatten_int_shapes,
    has_empty_group_attrs,
    partition_case_regimes,
    select_case_regime_representative,
    shape_numel,
)

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
    "attention_residual": "coupled_1c2v/bidirectional_cv_ring/affine_simd",
    "mla": "coupled_1c2v/bidirectional_cv_ring/affine_simd",
    "gqa": "coupled_1c2v/bidirectional_cv_ring/affine_simd",
    "dynamic_mx_quant": "aiv_only/gm_ub_stream/affine_simd",
    "sparse_flash_attention": "coupled_1c2v/l0c_ub_epilogue/irregular_simt",
    "kimi_delta_attention": "aiv_only/persistent_onchip_state/affine_simd",
    "tensor_program_vm": "aiv_only/persistent_onchip_state/affine_simd",
}

_EXPECTED_WORK_MODEL: dict[str, str] = {
    "swi_glu": "analytical_elementwise",
    "rms_norm": "analytical_reduction",
    "softmax": "analytical_reduction",
    "apply_rotary_pos_emb": "analytical_elementwise",
    "add_rms_norm_dynamic_quant": "analytical_reduction",
    "mhc_sinkhorn": "analytical_reduction",
    "grouped_matmul": "analytical_matmul",
    "q4_0_dequant_matmul": "analytical_matmul_quant",
    "attention_residual": "analytical_attention_online",
    "mla": "analytical_attention_online",
    "gqa": "analytical_attention_online",
    "dynamic_mx_quant": "analytical_reduction",
    "sparse_flash_attention": "analytical_attention",
    "kimi_delta_attention": "analytical_reduction",
    "tensor_program_vm": "analytical_reduction",
}

_AXES_AIV = TopologyAxes("aiv_only", "gm_ub_stream", "affine_simd")
_AXES_CUBE = TopologyAxes("aic_only", "gm_l1_l0_fixpipe", "affine_simd")
_AXES_COUPLED = TopologyAxes("coupled_1c2v", "l0c_ub_epilogue", "affine_simd")
_AXES_PERSISTENT = TopologyAxes("coupled_1c2v", "persistent_onchip_state", "affine_simd")
_AXES_PERSISTENT_AIV = TopologyAxes("aiv_only", "persistent_onchip_state", "affine_simd")


def _string_mapping(raw: object) -> dict[str, object]:
    assert isinstance(raw, dict)
    result: dict[str, object] = {}
    for key, value in raw.items():
        assert isinstance(key, str)
        result[key] = value
    return result


def _yaml_mapping(path: Path) -> dict[str, object]:
    return _string_mapping(yaml.safe_load(path.read_text(encoding="utf-8")))


def _case_entries(path: Path) -> tuple[dict[str, object], ...]:
    raw_cases = _yaml_mapping(path).get("cases")
    assert isinstance(raw_cases, list)
    return tuple(_string_mapping(item) for item in raw_cases)


def _proto_inputs(proto: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    raw_inputs = proto.get("inputs")
    if not isinstance(raw_inputs, list):
        return ()
    return tuple(_string_mapping(item) for item in raw_inputs)


def _case_attrs(case: Mapping[str, object]) -> Mapping[str, object]:
    raw_attrs = case.get("attrs")
    if isinstance(raw_attrs, dict):
        return _string_mapping(raw_attrs)
    return {}


def _op_path(op_name: str) -> Path:
    for level, name in _STOCK_OPS:
        if name == op_name:
            return DEFAULT_OPS_ROOT / level / name
    raise KeyError(op_name)


def _load_op(op_name: str) -> tuple[str, dict[str, object], tuple[dict[str, object], ...]]:
    op_dir = _op_path(op_name)
    source = (op_dir / "golden.py").read_text(encoding="utf-8")
    proto = _yaml_mapping(op_dir / "proto.yaml")
    return source, proto, _case_entries(op_dir / "cases.yaml")


def _schema_dim_ints(
    case: Mapping[str, object],
    proto_inputs: tuple[Mapping[str, object], ...],
) -> dict[str, int]:
    raw_dims = extract_schema_driven_case_dims(case, proto_inputs=proto_inputs)
    return {
        str(key): value
        for key, value in raw_dims.items()
        if isinstance(value, int) and not isinstance(value, bool)
    }


def _scrub_proto_identity(proto: Mapping[str, object], alias: str) -> dict[str, object]:
    copied = copy.deepcopy(dict(proto))
    for key in ("id", "name", "op_name", "cannbot_task", "schema"):
        if key in copied:
            copied[key] = alias
    op_block = copied.get("operator")
    if isinstance(op_block, dict):
        for key in ("id", "name", "schema"):
            if key in op_block:
                op_block[key] = alias
    return copied


def _extractor_regime_tags(extractor: OpAdvisoryExtractor, op_name: str) -> dict[int, str]:
    level = next(lvl for lvl, name in _STOCK_OPS if name == op_name)
    advisory = extractor.extract(op_name, level=level)
    tags: dict[int, str] = {}
    for regime in advisory.regimes:
        for case_num in regime.case_nums:
            assert case_num not in tags
            tags[case_num] = regime.regime_tag
    assert sorted(tags) == list(range(1, 21))
    return tags


@pytest.fixture(scope="module")
def ops_root() -> Path:
    if not DEFAULT_OPS_ROOT.is_dir():
        pytest.skip(f"cann-bench tasks root missing: {DEFAULT_OPS_ROOT}")
    return DEFAULT_OPS_ROOT


@pytest.fixture(scope="module")
def analyzer(ops_root: Path) -> PyTorchSemanticAnalyzer:
    return PyTorchSemanticAnalyzer()


@pytest.fixture(scope="module")
def extractor(ops_root: Path) -> OpAdvisoryExtractor:
    return OpAdvisoryExtractor(DEFAULT_OPS_ROOT)


@pytest.mark.parametrize("op_name", _STOCK_IDS)
def test_stock_operator_semantics_and_base_axes(
    analyzer: PyTorchSemanticAnalyzer,
    op_name: str,
) -> None:
    """AST analysis recovers the structural semantics of every stock operator."""
    source, proto, _cases = _load_op(op_name)
    semantics = analyzer.analyze_source(source, entry_name=op_name, proto=proto)
    observed = (
        semantics.has_matmul,
        semantics.has_reduction,
        semantics.reduction_axes,
        semantics.has_online_exp_max,
        semantics.has_indirect_index,
        semantics.has_subbyte_or_block_quant,
        semantics.has_recurrent_loop,
        semantics.mutates_inputs,
        semantics.preserves_input_shape,
    )
    assert observed == _EXPECTED_SEMANTICS[op_name]
    assert semantics.entry_name == op_name
    assert semantics.base_axes.key() == _EXPECTED_AXES[op_name]


def test_ast_ignores_dead_helpers_names_docstrings_and_strings(
    analyzer: PyTorchSemanticAnalyzer,
) -> None:
    """Names, docstrings, arbitrary strings, and unreachable helpers never create semantics."""
    source = dedent(
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
    semantics = analyzer.analyze_source(
        source,
        entry_name="flash_attention_q4_quant_matmul",
        proto={
            "description": "quantized softmax attention with int4 matmul and indirect gather",
            "name": "dead_matmul",
        },
    )
    assert semantics.has_matmul is False
    assert semantics.has_reduction is False
    assert semantics.reduction_axes == ()
    assert semantics.has_online_exp_max is False
    assert semantics.has_indirect_index is False
    assert semantics.has_subbyte_or_block_quant is False
    assert semantics.has_recurrent_loop is False
    assert semantics.mutates_inputs is False


def test_ast_follows_reachable_top_level_and_nested_helpers(
    analyzer: PyTorchSemanticAnalyzer,
) -> None:
    """Only helpers actually called by the entry contribute reduction, matmul, or loop evidence."""
    source = dedent(
        """
        import torch

        def entry(x):
            def inner(y):
                return torch.sum(y, dim=1)
            return inner(x)

        def dead_outer(x):
            def dead_inner(y):
                return torch.matmul(y, y)
            return dead_inner(x)
        """
    )
    semantics = analyzer.analyze_source(source, entry_name="entry")
    assert semantics.has_reduction is True
    assert semantics.reduction_axes == (1,)
    assert semantics.has_matmul is False
    live_source = dedent(
        """
        import torch

        def entry(x):
            return _live(x)

        def _live(x):
            return torch.softmax(x, dim=-1)
        """
    )
    live = analyzer.analyze_source(live_source, entry_name="entry")
    assert live.has_reduction is True
    assert live.reduction_axes == (-1,)
    assert live.has_online_exp_max is True
    sole_source = dedent(
        """
        import torch

        def gather(x, index):
            return torch.gather(x, dim=1, index=index)
        """
    )
    sole = analyzer.analyze_source(sole_source, entry_name="exp")
    assert sole.entry_name == "exp"
    assert sole.has_indirect_index is True


def test_dtype_bytes_parameter_scales_ratios_and_traffic(
    analyzer: PyTorchSemanticAnalyzer,
) -> None:
    """Real case dtypes override the fallback, while absent dtypes scale declared traffic."""
    case = {"case_id": 1, "input_shape": [[16, 32]]}
    semantics = _semantics()
    byte = analyzer.compute_case_ratios(semantics, case, dtype_bytes=1.0)
    half = analyzer.compute_case_ratios(semantics, case, dtype_bytes=2.0)
    word = analyzer.compute_case_ratios(semantics, case, dtype_bytes=4.0)
    assert byte.inner_contiguous_bytes == 32
    assert half.inner_contiguous_bytes == 2 * byte.inner_contiguous_bytes
    assert word.inner_contiguous_bytes == 4 * byte.inner_contiguous_bytes
    typed_case = {**case, "dtype": ["float16"]}
    assert (
        analyzer.compute_case_ratios(semantics, typed_case, dtype_bytes=8.0)
        == half
    )
    with pytest.raises(ValueError, match="dtype_bytes must be finite and positive"):
        analyzer.compute_case_ratios(semantics, case, dtype_bytes=0.0)
    with pytest.raises(ValueError, match="dtype_bytes must be finite and positive"):
        analyzer.compute_case_ratios(semantics, case, dtype_bytes=math.nan)


def test_blueprint_dtype_bytes_controls_traffic_basis(
    analyzer: PyTorchSemanticAnalyzer,
) -> None:
    """Fallback widths price traffic but never claim a provable lower bound."""
    source, proto, _cases = _load_op("q4_0_dequant_matmul")
    quant = analyzer.analyze_source(source, entry_name="q4_0_dequant_matmul", proto=proto)
    dims = {"in_features": 160, "out_features": 512, "total_elements": 160}
    byte_packet = analyzer.build_universal_blueprint(quant, dims, dtype_bytes=1.0)
    wide_packet = analyzer.build_universal_blueprint(quant, dims, dtype_bytes=2.0)
    assert byte_packet.roofline.min_gm_bytes == 160 + 512 + (160 * 512) // 2
    assert wide_packet.roofline.min_gm_bytes == 2 * (160 + 512) + (160 * 512) // 2
    assert byte_packet.provenance is not None
    assert byte_packet.provenance.semantic_source == "pytorch_golden"
    assert byte_packet.provenance.traffic_basis == "estimated_shape_dtype"
    assert wide_packet.provenance is not None
    assert wide_packet.provenance.traffic_basis == "estimated_shape_dtype"
    grouped_source, grouped_proto, _ = _load_op("grouped_matmul")
    grouped = analyzer.analyze_source(
        grouped_source, entry_name="grouped_matmul", proto=grouped_proto
    )
    matmul_dims = {"M": 128, "N": 64, "K": 32}
    fallback = analyzer.build_universal_blueprint(grouped, matmul_dims, dtype_bytes=2.0)
    assert fallback.roofline.min_gm_bytes == 2 * (
        128 * 32 + 128 * 64 + 64 * 32
    )
    assert fallback.roofline.cube_flops == 2 * 128 * 64 * 32
    assert fallback.roofline.vector_flops == 0
    assert fallback.provenance is not None
    assert fallback.provenance.semantic_source == "pytorch_golden"
    assert fallback.provenance.traffic_basis == "estimated_shape_dtype"
    with pytest.raises(ValueError, match="dtype_bytes must be finite and positive"):
        analyzer.build_universal_blueprint(grouped, matmul_dims, dtype_bytes=math.inf)


def test_representative_case_pairs_declared_shapes_and_dtypes() -> None:
    """Case payloads pair each flattened shape with its own dtype and tolerate desync."""
    _, _, cases = _load_op("grouped_matmul")
    mixed = next(case for case in cases if case["case_id"] == 7)
    representative = RepresentativeCase.from_case_entry(mixed)
    assert representative.case_id == 7
    assert representative.input_shapes == ((128, 512), (8, 512, 512), (8, 512))
    assert representative.input_dtypes == ("bfloat16", "bfloat16", "float32")
    assert representative.output_dtype is None
    desynced = RepresentativeCase.from_case_entry(
        {
            "case_id": 9,
            "input_shape": [[4, 8], [8]],
            "dtype": [["float16"], ["bfloat16"], ["float32"]],
        }
    )
    assert desynced.input_shapes == ((4, 8), (8,))
    assert desynced.input_dtypes == (None, None)
    with pytest.raises(ValueError, match="positive integer case_id"):
        RepresentativeCase.from_case_entry(
            {"input_shape": [[4, 8]], "dtype": ["float16"]}
        )


def test_blueprint_prices_each_declared_input_dtype(
    analyzer: PyTorchSemanticAnalyzer,
) -> None:
    """Swapping declared input dtypes changes traffic while declared I/O stays provable."""
    source, proto, _cases = _load_op("grouped_matmul")
    semantics = analyzer.analyze_source(
        source, entry_name="grouped_matmul", proto=proto
    )
    dims = {"M": 1, "N": 8192, "K": 8192}
    activation_elements = 8192
    weight_elements = 8192 * 8192
    output_elements = 8192
    half_weight = RepresentativeCase(
        case_id=7,
        input_shapes=((1, 8192), (1, 8192, 8192)),
        input_dtypes=("float16", "float32"),
        output_dtype="float32",
    )
    swapped = replace(half_weight, input_dtypes=("float32", "float16"))
    undeclared_output = replace(half_weight, output_dtype=None)
    half_packet = analyzer.build_universal_blueprint(
        semantics, dims, representative_case=half_weight
    )
    swapped_packet = analyzer.build_universal_blueprint(
        semantics, dims, representative_case=swapped
    )
    undeclared_packet = analyzer.build_universal_blueprint(
        semantics, dims, representative_case=undeclared_output
    )
    half_bytes = (
        2 * activation_elements + 4 * weight_elements + 4 * output_elements
    )
    swapped_bytes = (
        4 * activation_elements + 2 * weight_elements + 4 * output_elements
    )
    assert half_bytes == 268_484_608
    assert swapped_bytes == 134_283_264
    assert half_packet.roofline.min_gm_bytes == half_bytes
    assert swapped_packet.roofline.min_gm_bytes == swapped_bytes
    assert half_bytes - swapped_bytes == 2 * (weight_elements - activation_elements)
    for packet in (half_packet, swapped_packet):
        assert packet.provenance is not None
        assert packet.provenance.semantic_source == "pytorch_golden"
        assert packet.provenance.traffic_basis == "declared_io_exact"
        assert packet.roofline.dominant_bound == "memory"
        assert packet.roofline.ideal_roofline_us == packet.roofline.memory_floor_us
        assert packet.roofline.memory_floor_us == pytest.approx(
            packet.roofline.min_gm_bytes / analyzer.hbm_bytes_per_second * 1e6
        )
        prompt = packet.format_for_prompt()
        assert f"min_gm_bytes={packet.roofline.min_gm_bytes}" in prompt
        assert f"memory_floor_us={packet.roofline.memory_floor_us!r}" in prompt
        assert f"ideal_roofline_us={packet.roofline.ideal_roofline_us!r}" in prompt
    assert (
        swapped_packet.roofline.ideal_roofline_us
        < half_packet.roofline.ideal_roofline_us
    )
    assert undeclared_packet.roofline.min_gm_bytes == (
        2 * activation_elements + 4 * weight_elements + 2 * output_elements
    )
    assert undeclared_packet.provenance is not None
    assert undeclared_packet.provenance.traffic_basis == "estimated_shape_dtype"


def test_attention_blueprint_prices_declared_qkv_dtypes(
    analyzer: PyTorchSemanticAnalyzer,
) -> None:
    """Attention traffic charges Q, K, and V at their declared widths, not one shared width."""
    source, proto, cases = _load_op("gqa")
    semantics = analyzer.analyze_source(source, entry_name="gqa", proto=proto)
    case = cases[0]
    dims = _schema_dim_ints(case, _proto_inputs(proto))
    declared = RepresentativeCase.from_case_entry(case)
    half_kv = replace(
        declared,
        input_dtypes=("float16", "float32", "float16"),
        output_dtype="float16",
    )
    wide_kv = replace(
        declared,
        input_dtypes=("float32", "float16", "float32"),
        output_dtype="float16",
    )
    q_elements = 2 * 128 * 32 * 128
    kv_elements = 2 * 128 * 8 * 128
    half_kv_bytes = 2 * q_elements + (4 + 2) * kv_elements + 2 * q_elements
    wide_kv_bytes = 4 * q_elements + (2 + 4) * kv_elements + 2 * q_elements
    half_packet = analyzer.build_universal_blueprint(
        semantics, dims, representative_case=half_kv
    )
    wide_packet = analyzer.build_universal_blueprint(
        semantics, dims, representative_case=wide_kv
    )
    assert half_packet.roofline.min_gm_bytes == half_kv_bytes == 5_767_168
    assert wide_packet.roofline.min_gm_bytes == wide_kv_bytes == 7_864_320
    for packet in (half_packet, wide_packet):
        assert packet.provenance is not None
        assert packet.provenance.semantic_source == "pytorch_golden"
        assert packet.provenance.traffic_basis == "declared_io_exact"
        assert packet.roofline.special_ops > 0
        assert packet.roofline.memory_floor_us == pytest.approx(
            packet.roofline.min_gm_bytes / analyzer.hbm_bytes_per_second * 1e6
        )
        prompt = packet.format_for_prompt()
        assert f"min_gm_bytes={packet.roofline.min_gm_bytes}" in prompt
        assert f"memory_floor_us={packet.roofline.memory_floor_us!r}" in prompt
        assert f"ideal_roofline_us={packet.roofline.ideal_roofline_us!r}" in prompt
    assert (
        wide_packet.roofline.ideal_roofline_us
        > half_packet.roofline.ideal_roofline_us
    )


def test_preserves_input_shape_follows_keepdim_broadcast_provenance(
    analyzer: PyTorchSemanticAnalyzer,
) -> None:
    """Only binary combination of entry data with keepdim-reduced data preserves the input shape."""
    combined = analyzer.analyze_source(
        dedent(
            """
            import torch

            def normalized(x):
                mean = x.mean(dim=-1, keepdim=True)
                centered = x - mean
                variance = (centered * centered).mean(dim=-1, keepdim=True)
                return centered / torch.sqrt(variance + 1e-5)
            """
        ),
        entry_name="normalized",
    )
    assert combined.preserves_input_shape is True
    isolated = analyzer.analyze_source(
        dedent(
            """
            import torch

            def isolated(x):
                return x.mean(dim=-1, keepdim=True) + 1.0
            """
        ),
        entry_name="isolated",
    )
    assert isolated.preserves_input_shape is False
    shrinking = analyzer.analyze_source(
        dedent(
            """
            import torch

            def shrunk(x):
                return x - x.mean(dim=-1, keepdim=False)
            """
        ),
        entry_name="shrunk",
    )
    assert shrinking.preserves_input_shape is False


def test_softmax_shape_preservation_requires_a_shape_preserving_return_path(
    analyzer: PyTorchSemanticAnalyzer,
) -> None:
    """A softmax that is reduced or left out of the returned value must not count."""
    reduced = analyzer.analyze_source(
        dedent(
            """
            import torch

            def reduced(x):
                return torch.softmax(x, dim=-1).sum(dim=-1)
            """
        ),
        entry_name="reduced",
    )
    assert reduced.preserves_input_shape is False
    staged = analyzer.analyze_source(
        dedent(
            """
            import torch

            def staged(x):
                probabilities = torch.softmax(x, dim=-1)
                return torch.sum(probabilities, dim=-1)
            """
        ),
        entry_name="staged",
    )
    assert staged.preserves_input_shape is False
    unrelated = analyzer.analyze_source(
        dedent(
            """
            import torch

            def unrelated(x):
                torch.softmax(x, dim=-1)
                return torch.sum(x, dim=-1)
            """
        ),
        entry_name="unrelated",
    )
    assert unrelated.preserves_input_shape is False
    helper = analyzer.analyze_source(
        dedent(
            """
            import torch

            def _normalize(x):
                return torch.softmax(x, dim=-1)

            def through_helper(x):
                return _normalize(x)
            """
        ),
        entry_name="through_helper",
    )
    assert helper.preserves_input_shape is True


def test_self_referential_reassignment_on_the_return_path_terminates(
    analyzer: PyTorchSemanticAnalyzer,
) -> None:
    """A name reassigned through its own conversion must not re-resolve forever."""
    reassigned = analyzer.analyze_source(
        dedent(
            """
            import torch

            def gather_like(x, index, dim):
                if index.dtype != torch.int64:
                    index = index.long()
                gathered = torch.gather(x, dim, index)
                return gathered
            """
        ),
        entry_name="gather_like",
    )
    assert reassigned.preserves_input_shape is False
    returned_conversion = analyzer.analyze_source(
        dedent(
            """
            import torch

            def returned_conversion(index):
                index = index.long()
                return index
            """
        ),
        entry_name="returned_conversion",
    )
    assert returned_conversion.preserves_input_shape is False


def test_obfuscated_softmax_and_shrinking_sum_price_output_write_back(
    analyzer: PyTorchSemanticAnalyzer,
) -> None:
    """Shape-preserving ops charge a full write-back while reducing ops charge one output row."""
    source, proto, _cases = _load_op("softmax")
    obfuscated_source, replacements = re.subn(
        r"^def\s+softmax\s*\(",
        "def anon_entry(",
        source,
        flags=re.MULTILINE,
    )
    assert replacements == 1
    semantics = analyzer.analyze_source(
        obfuscated_source,
        entry_name="anon_entry",
        proto=_scrub_proto_identity(proto, "anon_entry"),
    )
    assert semantics.preserves_input_shape is True
    wide = RepresentativeCase(
        case_id=1,
        input_shapes=((8192, 8192),),
        input_dtypes=("bfloat16",),
    )
    wide_packet = analyzer.build_universal_blueprint(
        semantics,
        {"total_elements": 8192 * 8192},
        representative_case=wide,
    )
    assert wide_packet.roofline.min_gm_bytes == 268_435_456
    narrow = RepresentativeCase(
        case_id=2,
        input_shapes=((4096, 1024),),
        input_dtypes=("float16",),
    )
    narrow_packet = analyzer.build_universal_blueprint(
        semantics,
        {"total_elements": 4096 * 1024},
        representative_case=narrow,
    )
    assert narrow_packet.roofline.min_gm_bytes == 16_777_216

    shrinking = analyzer.analyze_source(
        dedent(
            """
            import torch

            def anon_sum(x):
                return torch.sum(x, dim=-1)
            """
        ),
        entry_name="anon_sum",
    )
    assert shrinking.preserves_input_shape is False
    shrink_packet = analyzer.build_universal_blueprint(
        shrinking,
        {"total_elements": 4096 * 1024},
        representative_case=narrow,
    )
    assert shrink_packet.roofline.min_gm_bytes == 8_396_800


@pytest.mark.parametrize("op_name", _STOCK_IDS)
def test_entry_and_proto_obfuscation_preserves_semantics_and_ratios(
    analyzer: PyTorchSemanticAnalyzer,
    op_name: str,
) -> None:
    """Renaming the entry function and proto identity leaves semantics and ratios intact."""
    source, proto, cases = _load_op(op_name)
    semantics = analyzer.analyze_source(source, entry_name=op_name, proto=proto)
    alias = "anon_entry"
    obfuscated_source, replacements = re.subn(
        rf"^def\s+{re.escape(op_name)}\s*\(",
        f"def {alias}(",
        source,
        flags=re.MULTILINE,
    )
    assert replacements == 1
    obfuscated = analyzer.analyze_source(
        obfuscated_source,
        entry_name=alias,
        proto=_scrub_proto_identity(proto, alias),
    )
    assert obfuscated.entry_name == alias
    assert obfuscated.has_matmul == semantics.has_matmul
    assert obfuscated.has_reduction == semantics.has_reduction
    assert obfuscated.reduction_axes == semantics.reduction_axes
    assert obfuscated.has_online_exp_max == semantics.has_online_exp_max
    assert obfuscated.has_indirect_index == semantics.has_indirect_index
    assert obfuscated.has_subbyte_or_block_quant == semantics.has_subbyte_or_block_quant
    assert obfuscated.has_recurrent_loop == semantics.has_recurrent_loop
    assert obfuscated.mutates_inputs == semantics.mutates_inputs
    assert obfuscated.preserves_input_shape == semantics.preserves_input_shape
    assert obfuscated.base_axes == semantics.base_axes
    for case in cases:
        original = analyzer.compute_case_ratios(semantics, case)
        renamed = analyzer.compute_case_ratios(obfuscated, case)
        assert renamed.to_dict() == original.to_dict()


@pytest.mark.parametrize("op_name", _STOCK_IDS)
def test_all_cases_produce_valid_physics_ratios(
    analyzer: PyTorchSemanticAnalyzer,
    op_name: str,
) -> None:
    """Every official case yields finite ratios, literal phases, and a JSON-stable payload."""
    source, proto, cases = _load_op(op_name)
    semantics = analyzer.analyze_source(source, entry_name=op_name, proto=proto)
    assert len(cases) == 20
    case_ids: list[int] = []
    for case in cases:
        ratios = analyzer.compute_case_ratios(semantics, case)
        attrs = _case_attrs(case)
        case_id = case["case_id"]
        assert isinstance(case_id, int)
        case_ids.append(case_id)
        assert ratios.case_id == case_id
        for value in (
            ratios.stride_ratio,
            ratios.contig_atom_ratio,
            ratios.working_set_sram_ratio,
            ratios.working_set_vreg_ratio,
            ratios.wave_occupancy_ratio,
            ratios.oi_ridge_ratio,
        ):
            assert math.isfinite(value)
            assert value >= 0.0
        assert ratios.stride_ratio >= 1.0
        assert ratios.contig_atom_ratio > 0.0
        assert ratios.inner_contiguous_bytes >= 1
        assert ratios.has_empty_group == has_empty_group_attrs(attrs)
        engine, layout, residency, alignment, wave = ratios.phase_signature(semantics.base_axes)
        assert engine in ("aic_cube", "coupled_1c2v", "aiv_vector_or_gemv")
        assert layout in ("contiguous", "strided_or_gather")
        assert residency in ("vreg_resident", "sram_1pass", "gm_multipass")
        assert alignment in ("aligned_vector", "unaligned_predicated", "sub_atom_simt")
        assert wave in ("balanced_waves", "under_occupied_split", "ragged_groups")
        effective = ratios.effective_axes(semantics.base_axes)
        assert effective.compute_engine in VALID_COMPUTE_ENGINES
        assert effective.memory_ring in VALID_MEMORY_RINGS
        assert effective.iteration_regularity in VALID_ITERATION_REGULARITIES
        assert CasePhysicsRatios.from_dict(ratios.to_dict()) == ratios
    assert case_ids == list(range(1, 21))


def test_advisory_envelope_transports_case_physics_ratios(
    extractor: OpAdvisoryExtractor,
) -> None:
    """The advisory carries one JSON-stable physics-ratio payload per official case."""
    advisory = extractor.extract("softmax", level="level2")
    raw_payloads = advisory.structural_features["case_physics_ratios"]
    assert isinstance(raw_payloads, list)
    assert len(raw_payloads) == 20
    restored = []
    for raw in raw_payloads:
        assert isinstance(raw, dict)
        restored.append(CasePhysicsRatios.from_dict(raw))
    assert [ratios.case_id for ratios in restored] == list(range(1, 21))
    assert advisory.regimes
    by_case = {
        ratios.case_id: ratios
        for regime in advisory.regimes
        for ratios in regime.case_ratios
    }
    assert by_case == {ratios.case_id: ratios for ratios in restored}


@pytest.mark.parametrize("op_name", _STOCK_IDS)
def test_advisory_regimes_follow_public_physical_partition(
    extractor: OpAdvisoryExtractor,
    op_name: str,
) -> None:
    """Every stock advisory carries 2-4 physical regimes with representative evidence."""
    level = next(lvl for lvl, name in _STOCK_OPS if name == op_name)
    advisory = extractor.extract(op_name, level=level)
    assert 2 <= len(advisory.regimes) <= 4
    covered: list[int] = []
    for regime in advisory.regimes:
        assert regime.phase_coordinates
        assert regime.representative_case_id in regime.case_nums
        ratios_by_id = {ratios.case_id: ratios for ratios in regime.case_ratios}
        assert regime.representative_ratios() == ratios_by_id[regime.representative_case_id]
        covered.extend(regime.case_nums)
    assert sorted(covered) == list(range(1, 21))
    assert len(set(covered)) == len(covered)


def test_softmax_phase_bands_match_extractor_regimes(
    analyzer: PyTorchSemanticAnalyzer,
    extractor: OpAdvisoryExtractor,
) -> None:
    """Contiguous, strided, and two-pass softmax cases agree across analyzer and extractor."""
    source, proto, cases = _load_op("softmax")
    semantics = analyzer.analyze_source(source, entry_name="softmax", proto=proto)
    tags = _extractor_regime_tags(extractor, "softmax")
    assert set(tags.values()) == {
        "reduce_tail",
        "reduce_tail_subatom",
        "reduce_strided",
        "reduce_strided_twopass",
    }
    signatures = {}
    for case in cases:
        ratios = analyzer.compute_case_ratios(semantics, case)
        case_id = ratios.case_id
        signature = ratios.phase_signature(semantics.base_axes)
        signatures[case_id] = (ratios, signature)
        tag = tags[case_id]
        if tag == "reduce_tail":
            assert signature[1] == "contiguous"
            assert signature[3] != "sub_atom_simt"
        elif tag == "reduce_tail_subatom":
            assert signature[1] == "contiguous"
            assert signature[3] == "sub_atom_simt"
        elif tag == "reduce_strided":
            assert signature[1] == "strided_or_gather"
            assert signature[2] != "gm_multipass"
        else:
            assert tag == "reduce_strided_twopass"
            assert signature[1] == "strided_or_gather"
            assert signature[2] == "gm_multipass"
    case1, phase1 = signatures[1]
    assert phase1[2] == "vreg_resident"
    case2, phase2 = signatures[2]
    assert phase2[2] == "sram_1pass"
    case4, phase4 = signatures[4]
    assert phase4 == (
        "aiv_vector_or_gemv",
        "strided_or_gather",
        "gm_multipass",
        "aligned_vector",
        "under_occupied_split",
    )
    assert case4.stride_ratio == 8192.0
    case7, phase7 = signatures[7]
    assert phase7[0] == "aiv_vector_or_gemv"
    assert phase7[3] == "unaligned_predicated"
    assert case7.stride_ratio == 1.0
    case13, phase13 = signatures[13]
    assert phase13[3] == "sub_atom_simt"
    assert case13.contig_atom_ratio < 1.0
    case16, phase16 = signatures[16]
    assert case16.stride_ratio == 2049.0
    assert phase16[4] == "under_occupied_split"


def test_q4_gemv_and_mixcore_phase_clustering(
    analyzer: PyTorchSemanticAnalyzer,
    extractor: OpAdvisoryExtractor,
) -> None:
    """Q4 GEMV/MixCore tags partition exactly along the analyzer engine and wave phases."""
    source, proto, cases = _load_op("q4_0_dequant_matmul")
    semantics = analyzer.analyze_source(source, entry_name="q4_0_dequant_matmul", proto=proto)
    tags = _extractor_regime_tags(extractor, "q4_0_dequant_matmul")
    tag_cases: dict[str, set[int]] = {}
    for case_id, tag in tags.items():
        tag_cases.setdefault(tag, set()).add(case_id)
    assert tag_cases == {
        "gemv_small": {1, 2, 3, 4, 5, 19, 20},
        "gemv": {6, 7, 9},
        "mixcore": {8, 10, 11, 12, 13, 14, 15, 16, 17, 18},
    }
    signatures = {}
    for case in cases:
        ratios = analyzer.compute_case_ratios(semantics, case)
        signatures[ratios.case_id] = (ratios, ratios.phase_signature(semantics.base_axes))
        engine, _layout, _residency, _alignment, wave = signatures[ratios.case_id][1]
        tag = tags[ratios.case_id]
        if tag == "gemv_small":
            assert engine == "aiv_vector_or_gemv"
            assert ratios.oi_ridge_ratio < 0.25
            assert wave == "under_occupied_split"
        elif tag == "gemv":
            assert engine == "aiv_vector_or_gemv"
            assert ratios.oi_ridge_ratio < 0.25
            assert wave == "balanced_waves"
        else:
            assert engine == "coupled_1c2v"
            assert ratios.oi_ridge_ratio >= 0.25
    case1, phase1 = signatures[1]
    assert case1.oi_ridge_ratio == pytest.approx(1 / 82)
    assert phase1[0] == "aiv_vector_or_gemv"
    assert case1.effective_axes(semantics.base_axes) == TopologyAxes(
        "aiv_only", "gm_ub_stream", "affine_simd"
    )
    case8, phase8 = signatures[8]
    assert case8.oi_ridge_ratio == pytest.approx(16 / 11)
    assert phase8[0] == "coupled_1c2v"
    case18, _phase18 = signatures[18]
    assert case18.oi_ridge_ratio == pytest.approx(128 / 11)


def test_mla_prefill_decode_split_clustering(
    analyzer: PyTorchSemanticAnalyzer,
    extractor: OpAdvisoryExtractor,
) -> None:
    """MLA prefill, decode, and split groups follow structural phases regardless of layout."""
    source, proto, cases = _load_op("mla")
    semantics = analyzer.analyze_source(source, entry_name="mla", proto=proto)
    tags = _extractor_regime_tags(extractor, "mla")
    expected = {
        "prefill_split": {1, 2, 5, 6, 13, 14, 15, 20},
        "prefill": {3, 4},
        "decode_split": {7, 8, 17, 18, 19},
        "decode": {9, 10, 11, 12, 16},
    }
    observed: dict[str, set[int]] = {}
    for case_id, tag in tags.items():
        observed.setdefault(tag, set()).add(case_id)
    assert observed == expected
    for case in cases:
        ratios = analyzer.compute_case_ratios(semantics, case)
        _engine, layout, _residency, _alignment, _wave = ratios.phase_signature(
            semantics.base_axes
        )
        attrs = _case_attrs(case)
        case_input_shape = flatten_int_shapes(case.get("input_shape"))
        lead = case_input_shape[0]
        layout_name = str(attrs.get("inputLayout", "BSND")).upper()
        sequence_axis = 2 if layout_name == "BNSD" else 1
        query_length = int(lead[sequence_axis])
        tag = tags[ratios.case_id]
        assert (layout == "strided_or_gather") is (layout_name == "BNSD")
        assert tag.startswith("prefill") is (query_length > 2)
        if layout_name == "BNSD":
            assert ratios.stride_ratio == 2.0
        else:
            assert ratios.stride_ratio == 1.0


@pytest.mark.parametrize("op_name", _STOCK_IDS)
def test_universal_blueprint_roofline_is_honest(
    analyzer: PyTorchSemanticAnalyzer,
    op_name: str,
) -> None:
    """Each phase signature has a non-fabricated Roofline packet with consistent floors."""
    source, proto, cases = _load_op(op_name)
    semantics = analyzer.analyze_source(source, entry_name=op_name, proto=proto)
    proto_inputs = _proto_inputs(proto)
    representatives: dict[tuple[str, ...], tuple[int, dict[str, int]]] = {}
    for case in cases:
        ratios = analyzer.compute_case_ratios(semantics, case)
        signature = ratios.phase_signature(semantics.base_axes)
        shapes = flatten_int_shapes(case.get("input_shape"))
        scale = shape_numel(shapes[0]) if shapes else 1
        dims = _schema_dim_ints(case, proto_inputs)
        previous = representatives.get(signature)
        if previous is None or scale > previous[0]:
            representatives[signature] = (scale, dims)
    assert representatives
    for dims in representatives.values():
        packet = analyzer.build_universal_blueprint(semantics, dims[1])
        roofline = packet.roofline
        assert roofline is not None
        assert packet.entry_name == op_name
        assert packet.requested_anchor_dims == dims[1]
        assert packet.resolved_hir_dims == roofline.resolved_hir_dims
        assert packet.shape_exact_match is True
        assert packet.fallback_reason is None
        assert packet.lowering_diagnostics == ("analytical_pytorch_frontend",)
        assert packet.provenance is not None
        assert packet.provenance.semantic_source == "pytorch_golden"
        assert packet.provenance.traffic_basis == "estimated_shape_dtype"
        assert roofline.compute_flops > 0
        assert roofline.compute_flops == roofline.cube_flops + roofline.vector_flops
        assert roofline.min_gm_bytes > 0
        assert roofline.anchor_dims == roofline.resolved_hir_dims
        assert roofline.arithmetic_intensity == pytest.approx(
            round(roofline.compute_flops / roofline.min_gm_bytes, 4)
        )
        if roofline.cube_flops == 0:
            assert roofline.compute_floor_us == roofline.vec_floor_us
        else:
            assert roofline.compute_floor_us == max(
                roofline.cube_floor_us, roofline.vec_floor_us
            )
        assert roofline.ideal_roofline_us == max(
            roofline.compute_floor_us, roofline.memory_floor_us
        )
        assert roofline.ideal_roofline_us > 0.0
        if roofline.compute_floor_us > roofline.memory_floor_us:
            assert roofline.dominant_bound == "compute"
        elif roofline.memory_floor_us > roofline.compute_floor_us:
            assert roofline.dominant_bound == "memory"
        else:
            assert roofline.dominant_bound == "balanced"


def test_q4_gemv_blueprint_recovers_missing_m(
    analyzer: PyTorchSemanticAnalyzer,
) -> None:
    """A vector activation with in_features/out_features resolves to M=1 without guessing."""
    source, proto, _cases = _load_op("q4_0_dequant_matmul")
    semantics = analyzer.analyze_source(source, entry_name="q4_0_dequant_matmul", proto=proto)
    dims = {"in_features": 160, "out_features": 512, "total_elements": 160}
    packet = analyzer.build_universal_blueprint(semantics, dims)
    roofline = packet.roofline
    assert roofline is not None
    assert roofline.resolved_hir_dims == {"M": 1, "N": 512, "K": 160, "B": 1, "E": 1}
    assert roofline.cube_flops == 2 * 1 * 512 * 160
    assert roofline.vector_flops == 2 * (160 * 512 + 5 * 512)
    assert roofline.min_gm_bytes == 2 * (160 + 512) + (160 * 512) // 2
    with pytest.raises(ValueError, match="matmul analytical Roofline requires M"):
        analyzer.build_universal_blueprint(
            semantics, {"in_features": 160, "out_features": 512}
        )


def test_sparse_attention_blueprint_uses_attention_work(
    analyzer: PyTorchSemanticAnalyzer,
) -> None:
    """An indirect coupled matmul with attention anchor dims is modelled as dense attention."""
    source, proto, cases = _load_op("sparse_flash_attention")
    semantics = analyzer.analyze_source(source, entry_name="sparse_flash_attention", proto=proto)
    proto_inputs = _proto_inputs(proto)
    dims = _schema_dim_ints(cases[0], proto_inputs)
    packet = analyzer.build_universal_blueprint(semantics, dims)
    roofline = packet.roofline
    assert roofline is not None
    assert packet.provenance is not None
    assert packet.provenance.semantic_source == "pytorch_golden"
    assert roofline.resolved_hir_dims == {
        "B": 2,
        "S_q": 256,
        "S_kv": 4096,
        "N_q": 32,
        "N_kv": 8,
        "D_qk": 128,
        "D_v": 128,
    }
    score_pairs = 2 * 32 * 256 * 4096
    assert roofline.cube_flops == 2 * score_pairs * (128 + 128)
    assert roofline.vector_flops == 5 * score_pairs
    assert roofline.min_gm_bytes == 2 * 2 * (
        32 * 256 * 128 + 8 * 4096 * 128 + 8 * 4096 * 128 + 32 * 256 * 128
    )


def test_blueprint_refuses_missing_anchor_dims(
    analyzer: PyTorchSemanticAnalyzer,
) -> None:
    """Each analytical work model raises on missing dims instead of emitting a default Roofline."""
    mla_source, mla_proto, _ = _load_op("mla")
    mla = analyzer.analyze_source(mla_source, entry_name="mla", proto=mla_proto)
    with pytest.raises(ValueError, match="attention requires one of anchor dims"):
        analyzer.build_universal_blueprint(mla, {"B": 2})
    with pytest.raises(ValueError, match="attention analytical Roofline requires"):
        analyzer.build_universal_blueprint(mla, {"B": 2, "S_q": 4})
    grouped_source, grouped_proto, _ = _load_op("grouped_matmul")
    grouped = analyzer.analyze_source(
        grouped_source, entry_name="grouped_matmul", proto=grouped_proto
    )
    with pytest.raises(ValueError, match="matmul requires one of anchor dims"):
        analyzer.build_universal_blueprint(grouped, {"M": 128})
    swi_glu_source, swi_glu_proto, _ = _load_op("swi_glu")
    swi_glu = analyzer.analyze_source(swi_glu_source, entry_name="swi_glu", proto=swi_glu_proto)
    with pytest.raises(ValueError, match="analytical Roofline requires total_elements"):
        analyzer.build_universal_blueprint(swi_glu, {"rows": 0, "cols": 0})


def _ratio(
    *,
    case_id: int = 1,
    stride_ratio: float = 1.0,
    contig_atom_ratio: float = 16.0,
    is_atom_aligned: bool = True,
    working_set_sram_ratio: float = 0.25,
    working_set_vreg_ratio: float = 0.5,
    wave_occupancy_ratio: float = 1.0,
    oi_ridge_ratio: float = 1.0,
    inner_contiguous_bytes: int = 32,
    has_empty_group: bool = False,
) -> CasePhysicsRatios:
    return CasePhysicsRatios(
        case_id=case_id,
        stride_ratio=stride_ratio,
        contig_atom_ratio=contig_atom_ratio,
        is_atom_aligned=is_atom_aligned,
        working_set_sram_ratio=working_set_sram_ratio,
        working_set_vreg_ratio=working_set_vreg_ratio,
        wave_occupancy_ratio=wave_occupancy_ratio,
        oi_ridge_ratio=oi_ridge_ratio,
        inner_contiguous_bytes=inner_contiguous_bytes,
        has_empty_group=has_empty_group,
    )


def test_phase_band_boundaries() -> None:
    """Wave, OI, residency, alignment, and empty-group boundaries select exact phase literals."""
    assert _ratio(wave_occupancy_ratio=0.5).phase_signature(_AXES_COUPLED)[4] == "balanced_waves"
    assert (
        _ratio(wave_occupancy_ratio=0.4999).phase_signature(_AXES_COUPLED)[4]
        == "under_occupied_split"
    )
    assert (
        _ratio(wave_occupancy_ratio=0.0, has_empty_group=True).phase_signature(_AXES_COUPLED)[4]
        == "ragged_groups"
    )
    assert _ratio(oi_ridge_ratio=0.25).phase_signature(_AXES_COUPLED)[0] == "coupled_1c2v"
    assert (
        _ratio(oi_ridge_ratio=0.2499).phase_signature(_AXES_COUPLED)[0]
        == "aiv_vector_or_gemv"
    )
    assert _ratio(oi_ridge_ratio=0.25).phase_signature(_AXES_CUBE)[0] == "aic_cube"
    assert _ratio(oi_ridge_ratio=1.0).phase_signature(_AXES_AIV)[0] == "aiv_vector_or_gemv"
    assert (
        _ratio(working_set_vreg_ratio=1.0, working_set_sram_ratio=0.5).phase_signature(
            _AXES_AIV
        )[2]
        == "vreg_resident"
    )
    assert (
        _ratio(working_set_vreg_ratio=1.0001, working_set_sram_ratio=1.0).phase_signature(
            _AXES_AIV
        )[2]
        == "sram_1pass"
    )
    assert (
        _ratio(working_set_vreg_ratio=1.0001, working_set_sram_ratio=1.0001).phase_signature(
            _AXES_AIV
        )[2]
        == "gm_multipass"
    )
    assert _ratio(contig_atom_ratio=1.0).phase_signature(_AXES_AIV)[3] == "aligned_vector"
    assert (
        _ratio(contig_atom_ratio=0.999).phase_signature(_AXES_AIV)[3] == "sub_atom_simt"
    )
    assert (
        _ratio(contig_atom_ratio=1.0, is_atom_aligned=False).phase_signature(_AXES_AIV)[3]
        == "unaligned_predicated"
    )
    assert (
        _ratio(stride_ratio=1.0).phase_signature(_AXES_AIV)[1] == "contiguous"
    )
    assert (
        _ratio(stride_ratio=2.0).phase_signature(_AXES_AIV)[1] == "strided_or_gather"
    )


def test_effective_axes_transitions() -> None:
    """Phase bands demote cube work, route multipass persistence, and mark ragged/simt work."""
    assert _ratio(oi_ridge_ratio=0.1).effective_axes(_AXES_COUPLED) == TopologyAxes(
        "aiv_only", "gm_ub_stream", "affine_simd"
    )
    multipass = _ratio(
        working_set_vreg_ratio=4.0,
        working_set_sram_ratio=8.0,
    )
    assert multipass.effective_axes(_AXES_PERSISTENT) == TopologyAxes(
        "coupled_1c2v", "l0c_ub_epilogue", "affine_simd"
    )
    assert multipass.effective_axes(_AXES_PERSISTENT_AIV) == TopologyAxes(
        "aiv_only", "gm_ub_stream", "affine_simd"
    )
    assert _ratio(has_empty_group=True).effective_axes(_AXES_COUPLED).iteration_regularity == (
        "ragged_group"
    )
    assert _ratio(contig_atom_ratio=0.5).effective_axes(_AXES_AIV).iteration_regularity == (
        "irregular_simt"
    )
    assert _ratio(contig_atom_ratio=0.5).effective_axes(_AXES_COUPLED).iteration_regularity == (
        "affine_simd"
    )


def test_case_physics_ratios_validation_and_roundtrip() -> None:
    """Invalid ratios are rejected and dict/JSON roundtrips preserve every field."""
    valid = _ratio(case_id=7, stride_ratio=2.0, contig_atom_ratio=7.5)
    assert CasePhysicsRatios.from_dict(json.loads(json.dumps(valid.to_dict()))) == valid
    with pytest.raises(ValueError, match="stride_ratio must be >= 1.0"):
        _ratio(stride_ratio=0.5)
    with pytest.raises(ValueError, match="contig_atom_ratio must be > 0.0"):
        _ratio(contig_atom_ratio=0.0)
    with pytest.raises(ValueError, match="must be finite and non-negative"):
        _ratio(working_set_sram_ratio=-1.0)
    with pytest.raises(ValueError, match="must be finite and non-negative"):
        _ratio(wave_occupancy_ratio=math.inf)
    with pytest.raises(ValueError, match="case_id must be positive"):
        _ratio(case_id=0)
    with pytest.raises(ValueError, match="case_id must be an int"):
        CasePhysicsRatios.from_dict({**valid.to_dict(), "case_id": True})
    missing_alignment = valid.to_dict()
    del missing_alignment["inner_contiguous_bytes"]
    with pytest.raises(ValueError, match="inner_contiguous_bytes must be an int"):
        CasePhysicsRatios.from_dict(missing_alignment)


def test_regime_slice_case_ratios_roundtrip() -> None:
    """RegimeSlice persists physics ratios, representative id, and phase evidence."""
    first = _ratio(case_id=1)
    second = _ratio(case_id=2, wave_occupancy_ratio=0.1)
    regime = RegimeSlice(
        regime_tag="main",
        case_nums=(1, 2),
        representative_dims={"M": 128, "N": 128},
        has_unaligned_tail=False,
        case_ratios=(first, second),
        representative_case_id=1,
        phase_coordinates=(
            "aiv_vector_or_gemv/contiguous/vreg_resident/aligned_vector/balanced_waves",
        ),
        rationale="shared phase evidence",
    )
    assert regime.representative_ratios() == first
    assert RegimeSlice.from_json(regime.to_json()) == regime
    payload = regime.to_dict()
    payload["case_ratios"] = "not-a-sequence"
    with pytest.raises(ValueError, match="case_ratios must be a sequence"):
        RegimeSlice.from_dict(payload)
    payload["case_ratios"] = [first.to_dict(), "ignored"]
    filtered = RegimeSlice.from_dict(payload)
    assert filtered.case_ratios == (first,)
    with pytest.raises(ValueError, match="case_ratios must contain CasePhysicsRatios"):
        RegimeSlice(
            regime_tag="main",
            case_nums=(1,),
            representative_dims={"M": 1},
            has_unaligned_tail=False,
            case_ratios=("bad",),
        )
    with pytest.raises(ValueError, match="representative_case_id must belong to case_nums"):
        RegimeSlice(
            regime_tag="main",
            case_nums=(1,),
            representative_dims={"M": 1},
            has_unaligned_tail=False,
            representative_case_id=2,
        )


def _partition(
    ratios: dict[int, CasePhysicsRatios],
    dims: dict[int, dict[str, int]] | None = None,
    workloads: dict[int, int] | None = None,
    *,
    kind: CasePhysicsKind = "reduction",
    axes: TopologyAxes = _AXES_AIV,
    min_partitions: int = 2,
    max_partitions: int = 4,
) -> tuple[CasePhysicsRegimeGroup, ...]:
    return partition_case_regimes(
        ratios,
        dims if dims is not None else {case_id: {"rows": 8, "cols": 16} for case_id in ratios},
        workloads
        if workloads is not None
        else {case_id: 1024 * case_id for case_id in ratios},
        kind=kind,
        axes=axes,
        min_partitions=min_partitions,
        max_partitions=max_partitions,
    )


def test_case_workload_bytes_uses_each_declared_dtype() -> None:
    """Workload bytes multiply each input numel by its own dtype width."""
    assert case_workload_bytes(((2, 3),), ("float32",)) == 24
    assert case_workload_bytes(((2, 3),), ("float16",)) == 12
    assert case_workload_bytes(((2, 2), (3,)), ("int8", "float32")) == 16
    assert case_workload_bytes(((2, 2), (3,)), ()) == 14


def test_partition_single_phase_group_splits_by_phase_band() -> None:
    """One raw tag with differing wave bands splits by wave and preserves coordinates."""
    ratios = {
        1: _ratio(case_id=1, wave_occupancy_ratio=1.0),
        2: _ratio(case_id=2, wave_occupancy_ratio=1.0),
        3: _ratio(case_id=3, wave_occupancy_ratio=0.1),
    }
    groups = _partition(ratios)
    assert [group.regime_tag for group in groups] == [
        "reduce_tail_wave_filled",
        "reduce_tail_wave_underfill",
    ]
    assert groups[0].case_ids == (1, 2)
    assert groups[1].case_ids == (3,)
    for group in groups:
        assert group.phase_coordinates
        assert "reduce_tail_wave_" in group.description
        assert "split by wave band" in group.rationale
    assert groups[0].representative_case.case_id == 2
    assert groups[0].representative_case.workload_bytes == 2048


def test_partition_identical_phase_group_splits_by_workload() -> None:
    """A tag whose five coordinates coincide splits into latency and throughput halves."""
    ratios = {
        case_id: _ratio(case_id=case_id, wave_occupancy_ratio=1.0)
        for case_id in range(1, 5)
    }
    workloads = {1: 4096, 2: 512, 3: 2048, 4: 1024}
    groups = _partition(ratios, workloads=workloads)
    groups_by_tag = {group.regime_tag: group for group in groups}
    assert set(groups_by_tag) == {"reduce_tail_latency", "reduce_tail_throughput"}
    latency = groups_by_tag["reduce_tail_latency"]
    throughput = groups_by_tag["reduce_tail_throughput"]
    assert latency.case_ids == (2, 4)
    assert throughput.case_ids == (1, 3)
    assert throughput.representative_case.case_id == 1
    assert latency.phase_coordinates == throughput.phase_coordinates
    assert "dtype-aware workload scale" in latency.rationale


def test_partition_folds_split_variants_under_cap() -> None:
    """Five attention tags fold decode/prefill split variants into three regimes."""
    ratios = {
        1: _ratio(case_id=1, wave_occupancy_ratio=1.0),
        2: _ratio(case_id=2, wave_occupancy_ratio=0.1),
        3: _ratio(case_id=3, wave_occupancy_ratio=1.0),
        4: _ratio(case_id=4, wave_occupancy_ratio=0.1),
        5: _ratio(case_id=5, has_empty_group=True),
    }
    dims = {
        1: {"S_q": 1, "M": 1},
        2: {"S_q": 1, "M": 1},
        3: {"S_q": 8, "M": 8},
        4: {"S_q": 8, "M": 8},
        5: {"S_q": 8, "M": 8},
    }
    groups = _partition(ratios, dims, kind="attention", axes=_AXES_COUPLED)
    assert {group.regime_tag for group in groups} == {"decode", "prefill", "empty_group"}
    assert sorted(case_id for group in groups for case_id in group.case_ids) == [1, 2, 3, 4, 5]
    assert any("_split variants folded" in group.description for group in groups)


def test_partition_merges_above_cap_without_names_or_case_ids() -> None:
    """Five non-folding tags merge at the nearest phase distance and keep raw evidence."""
    ratios = {
        1: _ratio(case_id=1, wave_occupancy_ratio=1.0),
        2: _ratio(case_id=2, stride_ratio=2.0),
        3: _ratio(case_id=3, working_set_vreg_ratio=2.0, working_set_sram_ratio=2.0),
        4: _ratio(
            case_id=4,
            stride_ratio=2.0,
            working_set_vreg_ratio=2.0,
            working_set_sram_ratio=2.0,
        ),
        5: _ratio(case_id=5, has_empty_group=True),
    }
    groups = _partition(ratios)
    assert len(groups) == 4
    assert sorted(case_id for group in groups for case_id in group.case_ids) == [1, 2, 3, 4, 5]
    assert any("merged at 5D phase distance" in group.rationale for group in groups)
    coordinates = {coordinate for group in groups for coordinate in group.phase_coordinates}
    assert len(coordinates) == 5


def test_partition_bounds_validation_and_representative_rule() -> None:
    """Partition inputs are checked and representative selection is shared and deterministic."""
    ratios = {case_id: _ratio(case_id=case_id) for case_id in (1, 2)}
    with pytest.raises(ValueError, match="at least one official case"):
        _partition({})
    with pytest.raises(ValueError, match="partition bounds"):
        _partition(ratios, min_partitions=3, max_partitions=2)
    with pytest.raises(ValueError, match="case dims mismatch"):
        _partition(ratios, dims={1: {"M": 1}})
    with pytest.raises(ValueError, match="case workload bytes mismatch"):
        _partition(ratios, workloads={1: 10})
    with pytest.raises(ValueError, match="ratios carry case_id"):
        _partition({1: _ratio(case_id=2)})
    with pytest.raises(ValueError, match="workload_bytes must be non-negative"):
        _partition(ratios, workloads={1: 10, 2: -1})
    cases = (
        CasePhysicsRegimeCase(
            case_id=1,
            dims={"M": 4},
            phase_signature=_ratio().phase_signature(_AXES_AIV),
            workload_bytes=10,
        ),
        CasePhysicsRegimeCase(
            case_id=2,
            dims={"M": 8},
            phase_signature=_ratio().phase_signature(_AXES_AIV),
            workload_bytes=10,
        ),
    )
    assert select_case_regime_representative(cases).case_id == 2
    with pytest.raises(ValueError, match="needs at least one case"):
        select_case_regime_representative(())


def test_regime_solver_consumes_case_ratios() -> None:
    """The regime solver reports physics phases and splits tails for under-occupied waves."""
    solver = RegimePiecewiseSolver()
    balanced = _ratio(wave_occupancy_ratio=1.0)
    mapping_regime = {
        "regime_tag": "main",
        "representative_dims": {"M": 128, "N": 128, "K": 64},
        "case_ratios": [balanced.to_dict()],
    }
    balanced_plan = solver.solve_regime(
        mapping_regime, op_name="anon", topology_axes=_AXES_COUPLED
    )
    assert balanced_plan.policy["tail_wave_split"] is False
    assert (
        "Physics phase coupled_1c2v/contiguous/vreg_resident/aligned_vector/balanced_waves"
        in balanced_plan.notes
    )
    split = _ratio(
        stride_ratio=2.0,
        working_set_vreg_ratio=2.0,
        working_set_sram_ratio=4.0,
        wave_occupancy_ratio=0.1,
        oi_ridge_ratio=0.5,
    )
    split_plan = solver.solve_regime(
        mapping_regime | {"case_ratios": [split.to_dict()]},
        op_name="anon",
        topology_axes=_AXES_AIV,
    )
    assert split_plan.policy["tail_wave_split"] is True
    assert (
        "Physics phase aiv_vector_or_gemv/strided_or_gather/gm_multipass/"
        "aligned_vector/under_occupied_split"
    ) in split_plan.notes
    slice_regime = RegimeSlice(
        regime_tag="main",
        case_nums=(1,),
        representative_dims={"M": 128, "N": 128, "K": 64},
        has_unaligned_tail=False,
        case_ratios=(split,),
    )
    slice_plan = solver.solve_regime(
        slice_regime, op_name="anon", topology_axes=_AXES_AIV
    )
    assert slice_plan.policy["tail_wave_split"] is True
    assert (
        "Physics phase aiv_vector_or_gemv/strided_or_gather/gm_multipass/"
        "aligned_vector/under_occupied_split"
    ) in slice_plan.notes


def test_stock_operator_manifest_is_complete() -> None:
    """The stock manifest names fifteen distinct operators with matching level directories."""
    assert len(_STOCK_OPS) == 15
    assert len(set(_STOCK_IDS)) == 15
    for level, name in _STOCK_OPS:
        assert level in ("level1", "level2", "level3", "level4", "level5")
        if DEFAULT_OPS_ROOT.is_dir():
            assert (_op_path(name) / "golden.py").is_file()
            assert (_op_path(name) / "proto.yaml").is_file()
            assert (_op_path(name) / "cases.yaml").is_file()


def _semantics(
    *,
    entry_name: str = "op",
    reduction_axes: tuple[int, ...] = (1,),
) -> PyTorchOpSemantics:
    return PyTorchOpSemantics(
        entry_name=entry_name,
        has_matmul=False,
        has_reduction=True,
        reduction_axes=reduction_axes,
        has_online_exp_max=False,
        has_indirect_index=False,
        has_subbyte_or_block_quant=False,
        has_recurrent_loop=False,
        mutates_inputs=False,
    )


def test_semantics_object_validates_entry_and_axes() -> None:
    """PyTorchOpSemantics rejects empty entry names and normalizes reduction axes."""
    semantics = _semantics()
    assert semantics.reduction_axes == (1,)
    with pytest.raises(ValueError, match="entry_name must be a non-empty string"):
        _semantics(entry_name="  ")
    with pytest.raises(ValueError, match="must contain only ints"):
        _semantics(reduction_axes=(True,))


def test_subbyte_or_block_quant_distinguishes_8bit_elementwise_from_dequant(
    analyzer: PyTorchSemanticAnalyzer,
) -> None:
    """Standard int8/uint8 output stays False while dequant and sub-byte ops set True."""
    elem_int8_src = dedent(
        """
        import torch

        def per_token_int8(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            scale = x.abs().amax(dim=-1, keepdim=True) / 127.0
            y = torch.clamp((x / scale).round(), -128, 127).to(torch.int8)
            return y, scale.squeeze(-1)
        """
    )
    elem_sem = analyzer.analyze_source(elem_int8_src, entry_name="per_token_int8")
    assert elem_sem.has_subbyte_or_block_quant is False

    for lvl, op_name in (
        ("level3", "q4_0_dequant_matmul"),
        ("level3", "dequant_swiglu_quant"),
        ("level3", "weight_quant_batch_matmul"),
    ):
        op_dir = DEFAULT_OPS_ROOT / lvl / op_name
        if not op_dir.is_dir():
            continue
        source = (op_dir / "golden.py").read_text(encoding="utf-8")
        proto = _yaml_mapping(op_dir / "proto.yaml")
        sem = analyzer.analyze_source(source, entry_name=op_name, proto=proto)
        assert sem.has_subbyte_or_block_quant is True, op_name
