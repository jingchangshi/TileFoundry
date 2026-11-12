"""Tests for the native AR/MLA builders and device-only search boundary.

These tests load the two new builder modules directly as files because the
archived attnres package surface is intentionally not restored by CLEAN-CODE-2.
The search package is imported normally to cover its repaired __init__.
"""
from __future__ import annotations

import importlib.util
import sys
from dataclasses import fields, is_dataclass
from pathlib import Path

import pytest

from tilefoundry.ir.core.module import Module
from tilefoundry.ir.tir.prim_function import PrimFunction
from tilefoundry.ir.tir.stmts import ExecutionScope, For
from tilefoundry.visitor_registry.verify import verify_prim_function
from tilefoundry_ascend.search import (
    CandidateMeasurement,
    Deadline,
    MEASUREMENT_SOURCE_DEVICE,
    MEASUREMENT_SOURCE_SIMULATED,
    WorkerCommand,
    confirmation_required,
    rank,
    rankable,
    require_device_source,
    top_k,
)
from tilefoundry_ascend.search.process import run_process

ROOT = Path(__file__).resolve().parents[1] / "tilefoundry_ascend"


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ar = _load("_mainline_attnres_build", "attnres/build.py")
mla = _load("_mainline_mla_build", "mla/build.py")


def _walk(node):
    yield node
    if isinstance(node, (tuple, list)):
        for item in node:
            yield from _walk(item)
        return
    if not is_dataclass(node):
        return
    for field in fields(node):
        if field.name in ("type", "metadata", "annotation", "pipeline"):
            continue
        yield from _walk(getattr(node, field.name))


def _node_counts(body):
    counts = {}
    for node in _walk(body):
        name = type(node).__name__
        counts[name] = counts.get(name, 0) + 1
    return counts


def _ar_spec(**overrides):
    values = dict(n=6, tokens=333, d=2047, token_rows=8, block_n=4, v_versions=2, family=ar.BLOCKED_ONLINE, score_mode=ar.FACTORED_GW)
    values.update(overrides)
    return ar.AttnResSpec(**values)


def _mla_spec(**overrides):
    values = dict(batch=1, seq_q=64, seq_kv=512, heads_q=32, heads_kv=8, d_nope=448, d_rope=64, layout=mla.BNSD, kv_block=128, block_q=32, shard_count=4, precision=mla.PRECISION_HI_LO)
    values.update(overrides)
    return mla.MLASpec(**values)


def test_ar_builder_returns_native_module_for_and_scope():
    spec = _ar_spec()
    module = ar.build_attnres_module(spec)
    assert isinstance(module, Module)
    function = module.entry_function()
    assert isinstance(function, PrimFunction)
    assert function.name == "attention_residual"
    assert module.entry == "attention_residual"
    assert [node for node in _walk(function.body) if isinstance(node, For)]
    assert [node for node in _walk(function.body) if isinstance(node, ExecutionScope)]
    assert module.metadata["operator"] == "attention_residual"
    assert module.metadata["contract"]["device_validated"] is False
    assert module.metadata["contract"]["fp32_state"] == ("m", "l", "acc", "score")
    assert "rms" in module.metadata["contract"]["score_modes"][ar.FACTORED_GW]
    verify_prim_function(function, module_fns=module)


def test_ar_structure_is_independent_of_n_and_m():
    small = ar.build_attnres_module(_ar_spec(n=2, tokens=1, d=256, token_rows=1, block_n=1, d_chunk=256, family=ar.ONLINE, score_mode=ar.LITERAL_FP32))
    larger = ar.build_attnres_module(_ar_spec(n=17, tokens=333, d=256, token_rows=1, block_n=1, d_chunk=256, family=ar.ONLINE, score_mode=ar.LITERAL_FP32))
    assert _node_counts(small.entry_function().body) == _node_counts(larger.entry_function().body)
    assert sum(1 for node in _walk(small.entry_function().body) if isinstance(node, For)) == sum(1 for node in _walk(larger.entry_function().body) if isinstance(node, For))


def test_ar_rejects_family_cn_and_dtype_errors():
    with pytest.raises(ar.AttnResBuildError, match="ATTNRES_FAMILY_CN_CONTRADICTION"):
        _ar_spec(family=ar.ONLINE, block_n=4)
    with pytest.raises(ar.AttnResBuildError, match="ATTNRES_BAD_DTYPE"):
        _ar_spec(dtype="float32")
    with pytest.raises(ar.AttnResBuildError, match="ATTNRES_BAD_D_DIMENSION|ATTNRES_BAD_DIMENSION"):
        _ar_spec(d=128)


def test_mla_row_mapping_gqa_causal_and_grid():
    spec = _mla_spec()
    assert spec.group == 4
    assert spec.query_row_position(0) == (0, 0)
    assert spec.query_row_position(spec.seq_q) == (0, 1)
    assert spec.query_head(0, kv_head=3) == 12
    assert spec.kv_head_of(7) == 1
    assert spec.causal_position_of(spec.seq_q * 2 + 5) == 5
    assert spec.causal_limit(5) == 5 + (spec.seq_kv - spec.seq_q)
    assert spec.is_causal_masked(5, spec.causal_limit(5) + 1) is True
    assert spec.is_causal_masked(5, spec.causal_limit(5)) is False
    ranges = spec.shard_ranges()
    assert len(ranges) == 4
    assert ranges[0][0] == 0 and ranges[-1][0] + ranges[-1][1] == spec.d_v
    assert all(width > 0 for _, width in ranges)
    row_tiles = (spec.group * spec.seq_q + spec.block_q - 1) // spec.block_q
    assert spec.program_count() == spec.batch * spec.heads_kv * row_tiles * spec.shard_count


def test_mla_builder_returns_native_module_and_contract():
    spec = _mla_spec()
    module = mla.build_mla_module(spec)
    function = module.entry_function()
    assert isinstance(function, PrimFunction)
    assert function.name == "mla"
    assert [node for node in _walk(function.body) if isinstance(node, For)]
    assert [node for node in _walk(function.body) if isinstance(node, ExecutionScope)]
    assert module.metadata["program_count"] == spec.program_count()
    assert module.metadata["contract"]["rope_added_once"].startswith("score = scale")
    assert module.metadata["contract"]["precision"] == mla.PRECISION_HI_LO
    assert module.metadata["contract"]["device_validated"] is False
    verify_prim_function(function, module_fns=module)


def test_mla_rejects_legality_errors():
    with pytest.raises(mla.MLABuildError, match="MLA_D_ROPE_MUST_BE_64"):
        _mla_spec(d_rope=32)
    with pytest.raises(mla.MLABuildError, match="MLA_NONDIVISIBLE_HEADS"):
        _mla_spec(heads_q=30, heads_kv=8)
    with pytest.raises(mla.MLABuildError, match="MLA_CAUSAL_NEEDS_S_LE_SKV"):
        _mla_spec(seq_q=1024, seq_kv=512, is_causal=True)
    with pytest.raises(mla.MLABuildError, match="MLA_V_SHAPE_MISMATCH"):
        _mla_spec().check_v_metadata(v_shape=(1, 512, 8, 512), v_dtype="bfloat16", v_layout="BNSD", k_shape=(1, 512, 8, 448), k_dtype="bfloat16", k_layout="BNSD")


def test_builders_do_not_use_direct_tilelang_compile():
    for path in (ROOT / "attnres/build.py", ROOT / "mla/build.py"):
        text = path.read_text(encoding="utf-8")
        assert "tilelang.compile(" not in text
        assert "from tilelang import compile" not in text
        assert "attnres.twin" not in text
        assert "mla.kernel" not in text


def test_search_package_and_process_contract_import():
    assert Deadline.__name__ == "Deadline"
    assert WorkerCommand.__name__ == "WorkerCommand"
    assert callable(run_process)
    import tilefoundry_ascend.search as search_pkg
    assert not hasattr(search_pkg, "SimulatedClock")
    assert not hasattr(search_pkg, "HarnessError")
