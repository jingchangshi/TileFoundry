"""Tests for EnvelopeInspector positional dynamic binding and zero-divisor safety."""

from tilefoundry_ascend.ascend.schedule.envelope_inspector import (
    inspect_tilelang_primfunc,
    specialize_primfunc_for_anchor,
)


def test_envelope_inspector_binds_positional_shapes_and_avoids_zero_divisors() -> None:
    """Bind unnamed dynamic dims from ordered_tensor_shapes and clamp zero/unbound divisors."""
    import tilelang.ascend.language as T  # noqa: PLC0415
    from tilelang import tvm  # noqa: PLC0415

    m_dyn = tvm.tirx.SizeVar("unnamed_m", "int32")
    n_dyn = tvm.tirx.SizeVar("unnamed_n", "int32")

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
        dynamic_kernel,
        var_bindings={"tile_m": 0},
        ordered_tensor_shapes=((256, 64), (256, 64)),
    )
    assert spec_fn is not None

    inspection = inspect_tilelang_primfunc(
        dynamic_kernel,
        var_bindings={"tile_m": 0},
        ordered_tensor_shapes=((256, 64), (256, 64)),
    )
    assert inspection.ledger is not None
    assert "Divide by zero" not in (inspection.lowering_error or "")
    assert inspection.ledger.grid_blocks >= 1


def test_envelope_inspector_binds_renamed_decode_split_and_clamps_hc_divisor() -> None:
    """Bind renamed decode_split dims positionally and clamp m_rows_dyn // hc_dim_dyn divisors."""
    import tilelang.ascend.language as T  # noqa: PLC0415
    from tilelang import tvm  # noqa: PLC0415

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

    spec_unbound, _ = specialize_primfunc_for_anchor(
        split_kernel,
        var_bindings=None,
        ordered_tensor_shapes=((128, 64), (128, 64)),
    )
    assert spec_unbound is not None

    insp_zero = inspect_tilelang_primfunc(
        split_kernel,
        var_bindings={"hc_dim_dyn": 0},
        ordered_tensor_shapes=((128, 64), (128, 64)),
    )
    assert insp_zero.ledger is not None
    assert "Divide by zero" not in (insp_zero.lowering_error or "")
    assert insp_zero.ledger.grid_blocks >= 1

    insp_unbound = inspect_tilelang_primfunc(
        split_kernel,
        var_bindings=None,
        ordered_tensor_shapes=((128, 64), (128, 64)),
    )
    assert insp_unbound.ledger is not None
    assert "Divide by zero" not in (insp_unbound.lowering_error or "")
    assert insp_unbound.ledger.grid_blocks >= 1


def test_envelope_inspector_raises_compound_floordiv_denominator_and_specializes() -> None:
    """Raise compound denominator vars so nested FloorDiv(N, 64) at N=32 specializes cleanly."""
    import tilelang.ascend.language as T  # noqa: PLC0415
    from tilelang import tvm  # noqa: PLC0415

    m_dyn = tvm.tirx.SizeVar("M", "int32")
    n_dyn = tvm.tirx.SizeVar("N", "int32")

    @T.prim_func
    def nested_div_kernel(
        X: T.Tensor((m_dyn, n_dyn), "bfloat16"),
        Y: T.Tensor((m_dyn, n_dyn), "bfloat16"),
    ):
        with T.Kernel(T.floordiv(m_dyn, T.floordiv(n_dyn, 64))):
            x_ub = T.alloc_shared((64, 32), "bfloat16")
            T.copy(X[0:64, 0:32], x_ub)
            T.copy(x_ub, Y[0:64, 0:32])

    spec_fn, _ = specialize_primfunc_for_anchor(
        nested_div_kernel,
        var_bindings={"M": 256, "N": 32},
    )
    assert spec_fn is not None
    assert not spec_fn.same_as(nested_div_kernel)
    for param in spec_fn.params:
        buf = spec_fn.buffer_map.get(param)
        assert buf is not None
        assert len(buf.shape) == 2
        assert isinstance(buf.shape[0], tvm.tirx.IntImm)
        assert isinstance(buf.shape[1], tvm.tirx.IntImm)
        assert int(buf.shape[0].value) == 256
        assert int(buf.shape[1].value) >= 64
        assert not tvm.tirx.analysis.undefined_vars(spec_fn.body) or all(
            not isinstance(v, tvm.tirx.SizeVar)
            for v in tvm.tirx.analysis.undefined_vars(spec_fn.body)
        )

    spec_pos, _ = specialize_primfunc_for_anchor(
        nested_div_kernel,
        var_bindings=None,
        ordered_tensor_shapes=((256, 32), (256, 32)),
    )
    assert spec_pos is not None
    assert not spec_pos.same_as(nested_div_kernel)
    for param in spec_pos.params:
        buf = spec_pos.buffer_map[param]
        assert int(buf.shape[0].value) == 256
        assert int(buf.shape[1].value) >= 64

    insp_bound = inspect_tilelang_primfunc(
        nested_div_kernel,
        var_bindings={"M": 256, "N": 32},
    )
    assert insp_bound.ledger is not None
    assert "Divide by zero" not in (insp_bound.lowering_error or "")
    assert insp_bound.ledger.grid_blocks >= 1

    insp_pos = inspect_tilelang_primfunc(
        nested_div_kernel,
        var_bindings=None,
        ordered_tensor_shapes=((256, 32), (256, 32)),
    )
    assert insp_pos.ledger is not None
    assert "Divide by zero" not in (insp_pos.lowering_error or "")
    assert insp_pos.ledger.grid_blocks >= 1


def test_envelope_inspector_multi_level_and_subtract_denominators() -> None:
    """Handle multi-level nested FloorDiv and subtracted expressions in denominators."""
    import tilelang.ascend.language as T  # noqa: PLC0415
    from tilelang import tvm  # noqa: PLC0415

    m_dyn = tvm.tirx.SizeVar("M", "int32")
    n_dyn = tvm.tirx.SizeVar("N", "int32")

    @T.prim_func
    def deep_nested_kernel(
        X: T.Tensor((m_dyn, n_dyn), "bfloat16"),
        Y: T.Tensor((m_dyn, n_dyn), "bfloat16"),
    ):
        with T.Kernel(T.floordiv(m_dyn, T.floordiv(T.floordiv(n_dyn, 64), 2))):
            x_ub = T.alloc_shared((64, 32), "bfloat16")
            T.copy(X[0:64, 0:32], x_ub)
            T.copy(x_ub, Y[0:64, 0:32])

    spec_deep, _ = specialize_primfunc_for_anchor(
        deep_nested_kernel,
        var_bindings={"M": 256, "N": 32},
    )
    assert not spec_deep.same_as(deep_nested_kernel)
    for param in spec_deep.params:
        buf = spec_deep.buffer_map[param]
        assert int(buf.shape[1].value) >= 128

    @T.prim_func
    def sub_denom_kernel(
        X: T.Tensor((m_dyn, n_dyn), "bfloat16"),
        Y: T.Tensor((m_dyn, n_dyn), "bfloat16"),
    ):
        with T.Kernel(T.floordiv(m_dyn, n_dyn - 32)):
            x_ub = T.alloc_shared((64, 32), "bfloat16")
            T.copy(X[0:64, 0:32], x_ub)
            T.copy(x_ub, Y[0:64, 0:32])

    spec_sub, _ = specialize_primfunc_for_anchor(
        sub_denom_kernel,
        var_bindings={"M": 256, "N": 16},
    )
    assert not spec_sub.same_as(sub_denom_kernel)
    for param in spec_sub.params:
        buf = spec_sub.buffer_map[param]
        assert int(buf.shape[1].value) > 32


def test_ordered_tensor_shapes_extraction_and_l0_callsite_wiring(
    tmp_path,
    monkeypatch,
) -> None:
    """Wire extracted ordered_tensor_shapes through precheck, campaign runner, and marathon."""
    from tilefoundry_ascend.agent.campaign import campaign_runner as cr_mod  # noqa: PLC0415
    from tilefoundry_ascend.agent.campaign import marathon as marathon_mod  # noqa: PLC0415
    from tilefoundry_ascend.agent.specs.anchor_specs import (  # noqa: PLC0415
        AnchorCaseSpec,
        PerfTargetSpec,
        RegimeTargetSpec,
    )
    from tilefoundry_ascend.agent.campaign.surrogate_cannbench import (  # noqa: PLC0415
        extract_ordered_tensor_shapes,
        run_l0_static_precheck,
    )
    from tilefoundry_ascend.agent.orchestrator.gate_funnel import (  # noqa: PLC0415
        StageAResult,
        StageBResult,
    )

    assert extract_ordered_tensor_shapes(case_entry={"input_shape": [16, 128]}) == ((16, 128),)
    assert extract_ordered_tensor_shapes(
        case_entry={
            "input_shape": [[4, 128], [128]],
            "output_shape": [[4, 128]],
        }
    ) == ((4, 128), (128,), (4, 128))
    assert extract_ordered_tensor_shapes(
        case_entry={"input_shapes": [[2, 8, 64]]},
        dims={"B": 2, "S": 8, "D": 64},
        proto={"outputs": [{"name": "y", "shape": "[B, S, D]"}]},
    ) == ((2, 8, 64), (2, 8, 64))

    dsl_src = (
        "import tilelang.ascend.language as T\n"
        "from tilelang import tvm\n\n"
        "def make_kernel():\n"
        "    m_u = tvm.tirx.SizeVar('unnamed_m', 'int32')\n"
        "    n_u = tvm.tirx.SizeVar('unnamed_n', 'int32')\n"
        "    @T.prim_func\n"
        "    def main(X: T.Tensor((m_u, n_u), 'bfloat16'), "
        "Y: T.Tensor((m_u, n_u), 'bfloat16')):\n"
        "        with T.Kernel(T.floordiv(m_u, T.floordiv(n_u, 64))):\n"
        "            ub = T.alloc_shared((64, 32), 'bfloat16')\n"
        "            T.copy(X[0:64, 0:32], ub)\n"
        "            T.copy(ub, Y[0:64, 0:32])\n"
        "    return main\n"
    )
    precheck_res = run_l0_static_precheck(
        dsl_src,
        builder_name="make_kernel",
        builder_kwargs={},
        case_entry={"input_shape": [[256, 32], [256, 32]]},
    )
    assert precheck_res.ledger is not None
    assert "Divide by zero" not in (precheck_res.error_message or "")
    assert precheck_res.ledger.grid_blocks >= 1

    seen_cr_shapes: list[object] = []
    seen_marathon_shapes: list[object] = []

    def _fake_cr_stage_a(*_args, **kwargs) -> StageAResult:
        seen_cr_shapes.append(kwargs.get("ordered_tensor_shapes"))
        return StageAResult(passed=True, prim_func=None, ledger=precheck_res.ledger)

    def _fake_marathon_stage_a(*_args, **kwargs) -> StageAResult:
        seen_marathon_shapes.append(kwargs.get("ordered_tensor_shapes"))
        return StageAResult(passed=True, prim_func=None, ledger=precheck_res.ledger)

    monkeypatch.setattr(cr_mod, "run_stage_a_static", _fake_cr_stage_a)
    monkeypatch.setattr(marathon_mod, "run_stage_a_static", _fake_marathon_stage_a)
    monkeypatch.setattr(
        marathon_mod,
        "_stage0_entrypoint_contract_error",
        lambda *_a, **_kw: "",
    )
    monkeypatch.setattr(
        marathon_mod,
        "run_l1_compile_check",
        lambda *_a, **_kw: StageBResult(passed=True),
    )

    mod_path = tmp_path / "mod.py"
    mod_src = (
        dsl_src
        + "\ndef _build_combine(**_kw):\n    return make_kernel()\n"
        + "\ndef main(*_args, **_kwargs):\n    return make_kernel()\n"
    )
    mod_path.write_text(mod_src, encoding="utf-8")

    regime = RegimeTargetSpec(
        regime_tag="small",
        builder_candidates=("make_kernel",),
        sample_builder_kwargs={},
        representative_anchor_dims={"M": 256, "N": 32},
        editable_symbols=("make_kernel", "_build_combine"),
        representative_case_entry={
            "case_id": 1,
            "input_shape": [[256, 32], [256, 32]],
        },
    )
    runner = cr_mod.PerfTargetCampaignRunner(
        config=cr_mod.CampaignConfig(operator_name="custom_op", max_rounds=1),
        target_module_path=mod_path,
        perf_target=PerfTargetSpec(
            operator_name="custom_op",
            target_avg_speedup=1.0,
            target_min_case_speedup=1.0,
            target_roofline_eta=0.5,
            target_surrogate_score=50.0,
        ),
        anchors=(
            AnchorCaseSpec(
                task="custom_op",
                case_num=1,
                regime_tag="small",
                ref_perf_us=10.0,
                t_hw_us=5.0,
            ),
        ),
        regime_specs=(regime,),
    )
    monkeypatch.setattr(runner, "_resolve_operator_semantics_for_gates", lambda: None)
    monkeypatch.setattr(
        runner,
        "_resolve_combine_gate_spec",
        lambda *_a, **_kw: ("_build_combine", {}),
    )
    cr_res = runner._default_l0_checker(mod_src, regime)
    assert cr_res.passed is True
    assert seen_cr_shapes == [((256, 32), (256, 32)), ((256, 32), (256, 32))]

    ok, err, _ = marathon_mod._run_stage0_track_a_gates(
        mod_src,
        regime,
        entrypoint="main",
        worktree=tmp_path,
        env_file=None,
        enforce_hardware_engines=False,
    )
    assert ok is True, err
    assert err == ""
    assert seen_marathon_shapes == [((256, 32), (256, 32))]
