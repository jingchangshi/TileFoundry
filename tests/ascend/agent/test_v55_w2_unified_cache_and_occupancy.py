"""Tests for unified ArtifactStore caching, overflow classification, and occupancy errors."""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from tilefoundry_ascend.agent.campaign import surrogate_cannbench
from tilefoundry_ascend.agent.campaign.anchor_specs import AnchorCaseSpec
from tilefoundry_ascend.agent.orchestrator.gate_funnel import (
    StageAResult,
    StageBResult,
    _resolve_artifact_store,
    build_stage_b_driver_source,
    run_stage_b_ascendc_compile,
    run_stage_b_compile,
)
from tilefoundry_ascend.ascend.compile_cache.artifact_store import (
    ArtifactStore,
    collect_new_tl_cache_binaries,
    compute_artifact_id,
    compute_compile_task_key,
    merge_tl_cache_tree,
    resolve_unified_tl_cache_dir,
    seed_tl_cache_tree,
    snapshot_tl_cache_binaries,
)
from tilefoundry_ascend.ascend.passes.physical_ledger import build_objective_physical_ledger


def test_unified_task_key_delegation_and_cache_dir_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify compute_compile_task_key canonicalization, delegation, and cache dir priority."""
    k1 = compute_compile_task_key(
        normalized_kernel_unit_src="void k() {}",
        cann_target="dav-3510",
        pass_configs={"b": 2, "a": 1},
        compile_flags=["-DNDEBUG", "-O2", "-O2"],
        static_spec_signature="D=512",
        compiler_stamp="s1",
    )
    k2 = compute_compile_task_key(
        normalized_kernel_unit_src="void k() {}",
        cann_target="dav-3510",
        pass_configs={"a": 1, "b": 2},
        compile_flags=["-O2", "-DNDEBUG"],
        static_spec_signature="D=512",
        compiler_stamp="s1",
    )
    assert k1 == k2

    aid = compute_artifact_id(
        closure_src="void k() {}",
        flags="-O2",
        arch="dav-3510",
        spec_key="D=512",
        compiler_stamp="s1",
    )
    assert aid == compute_compile_task_key(
        normalized_kernel_unit_src="void k() {}",
        cann_target="dav-3510",
        pass_configs=None,
        compile_flags=("-O2",),
        static_spec_signature="D=512",
        compiler_stamp="s1",
    )

    explicit = tmp_path / "explicit_tl"
    assert resolve_unified_tl_cache_dir(explicit) == explicit.resolve()

    work_root = tmp_path / "work"
    assert resolve_unified_tl_cache_dir(
        env_vals={"TILEFOUNDRY_NPU_WORK_ROOT": str(work_root)}
    ) == (work_root / "surrogate" / "tilelang").resolve()

    env_tl = tmp_path / "env_tl"
    assert (
        resolve_unified_tl_cache_dir(
            env_vals={
                "TILELANG_CACHE_DIR": str(env_tl),
                "TILEFOUNDRY_NPU_WORK_ROOT": str(work_root),
            }
        )
        == env_tl.resolve()
    )

    monkeypatch.setenv("TILELANG_CACHE_DIR", str(env_tl))
    store = _resolve_artifact_store(None)
    assert store.root == env_tl.resolve()


def test_stage_b_compile_enables_cache_and_classifies_overflows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify run_stage_b_compile cache seeding, merge, and SRAM overflow codes."""
    cache_root = tmp_path / "unified_cache"
    monkeypatch.setenv("TILELANG_CACHE_DIR", str(cache_root))
    seed_marker = cache_root / "seeded.txt"
    cache_root.mkdir(parents=True, exist_ok=True)
    seed_marker.write_text("seed", encoding="utf-8")

    observed_envs: list[dict[str, str]] = []

    def _fake_ok_run(cmd, *, text, capture_output, cwd, env, timeout):
        _ = (cmd, text, capture_output, timeout)
        observed_envs.append(
            {
                "TILELANG_DISABLE_CACHE": env.get("TILELANG_DISABLE_CACHE", ""),
                "TILELANG_CACHE_DIR": env.get("TILELANG_CACHE_DIR", ""),
                "TILELANG_KERNEL_CACHE_USE_LIB_STAMP": env.get(
                    "TILELANG_KERNEL_CACHE_USE_LIB_STAMP", ""
                ),
            }
        )
        worker_cache = Path(env["TILELANG_CACHE_DIR"])
        assert (worker_cache / "seeded.txt").is_file()
        (worker_cache / "compiled_kernel.so").write_bytes(b"ELF_STAGE_B_SO")
        Path(cwd, "lowered_kernel.cce").write_text("// lowered\n", encoding="utf-8")
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    with monkeypatch.context() as mp:
        mp.setattr(
            "tilefoundry_ascend.agent.orchestrator.gate_funnel.subprocess.run",
            _fake_ok_run,
        )
        res_ok = run_stage_b_compile(
            "def kernel():\n    pass\n",
            pass_configs={"a": 1},
            compile_flags=["-O2"],
            spec_key="D=2048",
        )
    assert res_ok.passed is True
    assert observed_envs[0]["TILELANG_DISABLE_CACHE"] == "0"
    assert observed_envs[0]["TILELANG_KERNEL_CACHE_USE_LIB_STAMP"] == "1"
    assert observed_envs[0]["TILELANG_CACHE_DIR"] != str(cache_root)
    assert (cache_root / "compiled_kernel.so").is_file()
    store = ArtifactStore(cache_root)
    assert res_ok.artifact_id
    assert store.has_verified_artifact(res_ok.artifact_id) is True

    cases = [
        ("bisheng: error: exceeded vf stack size (32 > 0)", "VF_STACK_OVERFLOW"),
        ("Stack Frame Size: 64 bytes\nfatal error", "VF_STACK_OVERFLOW"),
        ("error: __ubuf__ capacity exceeded (300 KB > 256 KB)", "UB_SRAM_OVERFLOW"),
        ("error: L1 cbuf capacity overflow", "L1_SRAM_OVERFLOW"),
        ("error: L0C matrix buffer overflow", "L0C_SRAM_OVERFLOW"),
    ]
    for stderr_msg, expected_code in cases:
        with monkeypatch.context() as mp:
            mp.setattr(
                "tilefoundry_ascend.agent.orchestrator.gate_funnel.subprocess.run",
                lambda *a, _msg=stderr_msg, **kw: subprocess.CompletedProcess(
                    args=[], returncode=1, stdout="", stderr=_msg
                ),
            )
            res_fail = run_stage_b_compile("def kernel():\n    pass\n")
            assert res_fail.passed is False
            assert res_fail.fault_code == expected_code

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_bisheng = fake_bin / "bisheng"
    fake_bisheng.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "sys.stderr.write('error: exceeded vf stack size\\n')\n"
        "sys.exit(1)\n",
        encoding="utf-8",
    )
    fake_bisheng.chmod(0o755)
    sb_tb = run_stage_b_ascendc_compile(
        'extern "C" __global__ __vector__ void k() {}\n',
        bisheng_bin=fake_bisheng,
    )
    assert sb_tb.passed is False
    assert sb_tb.fault_code == "VF_STACK_OVERFLOW"


def test_run_l1_static_spec_dims_scan_and_prewarm_unified_hits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify L1 static width scan, VF_STACK_OVERFLOW rejection, and Stage-B->Prewarm hits."""
    assert not hasattr(surrogate_cannbench, "_build_prewarm_task_key")

    repo_root = Path(__file__).resolve().parents[3]
    cann_repo = tmp_path / "cann_repo"
    ar_dir = cann_repo / "tasks" / "level4" / "attention_residual"
    ar_dir.mkdir(parents=True, exist_ok=True)
    cases_payload = {
        "cases": [
            {
                "case_id": 2,
                "input_shape": [[2, 1, 1, 4096], [1, 1, 4096], [4096], [4096]],
                "dtype": "bfloat16",
                "attrs": {"alpha": 0.9, "epsilon": 1e-6},
            }
        ]
    }
    (ar_dir / "cases.yaml").write_text(yaml.safe_dump(cases_payload), encoding="utf-8")

    fake_root = tmp_path / "fake_root"
    fake_root.mkdir(parents=True, exist_ok=True)
    (fake_root / "src").symlink_to(repo_root / "src")
    (fake_root / "targets").symlink_to(repo_root / "targets")
    (fake_root / "tests" / "ascend" / "benchmarks").mkdir(parents=True, exist_ok=True)
    (fake_root / "tests" / "ascend" / "benchmarks" / "cannbench_contract.py").write_text(
        "class Harness:\n    pass\n",
        encoding="utf-8",
    )

    tl_stub = tmp_path / "tl_stub"
    tl_pkg_backend = tl_stub / "tilelang" / "backend"
    tl_pkg_backend.mkdir(parents=True)
    (tl_pkg_backend / "__init__.py").write_text("", encoding="utf-8")
    (tl_pkg_backend / "module.py").write_text(
        "from types import SimpleNamespace\n"
        "import contextlib\n"
        "def create_backend_context(_t):\n"
        "    return SimpleNamespace(target=contextlib.nullcontext())\n",
        encoding="utf-8",
    )
    (tl_stub / "tvm.py").write_text(
        "import contextlib\n"
        "class _Target:\n"
        "    @staticmethod\n"
        "    def current(allow_none=True):\n"
        "        return 'ascend'\n"
        "class target:\n"
        "    Target = _Target\n"
        "class transform:\n"
        "    @staticmethod\n"
        "    def PassContext(config=None):\n"
        "        return contextlib.nullcontext()\n"
        "class tirx:\n"
        "    PrimFunc = dict\n"
        "class IRModule:\n"
        "    pass\n",
        encoding="utf-8",
    )
    (tl_stub / "tilelang" / "__init__.py").write_text(
        "import os, tvm\n"
        "from pathlib import Path\n"
        "class _Art:\n"
        "    def __init__(self, src):\n"
        "        self.kernel_source = src\n"
        "        self.device_mod = src\n"
        "def lower(fn, target='ascend', enable_device_compile=True):\n"
        "    d = int(fn.get('d', 0))\n"
        "    if d == 512 and fn.get('overflow'):\n"
        "        raise RuntimeError('exceeded vf stack size (32 > 0)')\n"
        "    cache_dir = Path(os.environ['TILELANG_CACHE_DIR'])\n"
        "    cache_dir.mkdir(parents=True, exist_ok=True)\n"
        "    (cache_dir / f'k_{d}.so').write_bytes(b'ELF_L1_SO')\n"
        "    return _Art('// ok')\n"
        "def compile(prog, target='ascend', out_idx=None, pass_configs=None, compile_flags=None):\n"
        "    return lower(prog, target=target)\n",
        encoding="utf-8",
    )

    work_root = tmp_path / "work_root"
    env_file = tmp_path / "env.sh"
    env_file.write_text(
        f"TILEFOUNDRY_NPU_WORK_ROOT={work_root}\n"
        f"TILEFOUNDRY_NPU_CANNBENCH_REPO={cann_repo}\n"
        f"TILEFOUNDRY_NPU_TILELANG_SOURCE={tl_stub}\n"
        f"TILEFOUNDRY_NPU_PYTHON={sys.executable}\n",
        encoding="utf-8",
    )

    good_src = (
        "import tilelang\n"
        "def _select_ar_small_decode_config(n_depth=2, d_dim=4096, tl_dtype='bfloat16', "
        "schedule_mode='tuned', m_rows=1):\n"
        "    return {'tile_m': 1, 'pass_configs': {}, 'compile_flags': []}\n"
        "def _build_ar_small_decode_func(n_depth=2, d_dim=4096, tl_dtype='bfloat16', "
        "tile_m=1, schedule_mode='tuned'):\n"
        "    return {'d': int(d_dim), 'overflow': False}\n"
        "def _get_kernel(n_depth, d_dim, tl_dtype, tile_m, schedule_mode, *, "
        "builder_fn=None, pass_configs=None, compile_flags=None):\n"
        "    prog = builder_fn(n_depth, d_dim, tl_dtype, tile_m, schedule_mode)\n"
        "    return tilelang.compile(prog, target='ascend', out_idx=[4], "
        "pass_configs=pass_configs, compile_flags=compile_flags)\n"
    )
    bad_src = good_src.replace("'overflow': False", "'overflow': True")

    bad_res = surrogate_cannbench.run_l1_compile_check(
        bad_src,
        builder_name="_build_ar_small_decode_func",
        builder_kwargs={"n_depth": 2, "d_dim": 4096, "tl_dtype": "bfloat16", "tile_m": 1, "schedule_mode": "tuned"},
        config_selector_name="_select_ar_small_decode_config",
        dims={"D": 4096},
        static_spec_dims_list=[{"D": 512}, {"D": 2048}, {"D": 4096}],
        source_root=fake_root,
        env_file=env_file,
    )
    assert bad_res.passed is False
    assert bad_res.fault_code == "VF_STACK_OVERFLOW"

    tl_cache = work_root / "surrogate" / "tilelang"
    good_res = surrogate_cannbench.run_l1_compile_check(
        good_src,
        builder_name="_build_ar_small_decode_func",
        builder_kwargs={"n_depth": 2, "d_dim": 4096, "tl_dtype": "bfloat16", "tile_m": 1, "schedule_mode": "tuned"},
        config_selector_name="_select_ar_small_decode_config",
        dims={"D": 4096},
        static_spec_dims_list=[{"D": 512}, {"D": 2048}, {"D": 4096}],
        source_root=fake_root,
        env_file=env_file,
        regime_tag="small_decode",
    )
    assert good_res.passed is True
    assert (tl_cache / "k_4096.so").is_file()
    assert (tl_cache / "k_2048.so").is_file()
    assert (tl_cache / "k_512.so").is_file()
    assert (tl_cache / "prewarm_manifest_v1.json").is_file()
    store = ArtifactStore(tl_cache)
    assert store.has_verified_artifact(good_res.artifact_id) is True

    cand_file = tmp_path / "cand_good.py"
    cand_file.write_text(good_src, encoding="utf-8")
    anchors = (AnchorCaseSpec("level4/attention_residual", 2, "small_decode", 1.5, 0.5),)
    pw_res = surrogate_cannbench.prewarm_surrogate_tilelang_cache_with_stats(
        "attention_residual",
        [("c_good", cand_file)],
        anchors,
        cannbench_repo=cann_repo,
        env_file=env_file,
        cache_dir=work_root / "surrogate" / "reference",
        tl_cache_dir=tl_cache,
        source_root=fake_root,
    )
    assert pw_res.prewarm_stats["compiled_tasks"] == 0
    assert pw_res.prewarm_stats["cache_hits"] == 1


def test_trackb_stage_b_to_prewarm_achieves_zero_recompile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify Track-B Stage-B artifact key aligns with multi-case Prewarm for zero recompile."""
    repo_root = Path(__file__).resolve().parents[3]
    cann_repo = tmp_path / "cann_repo"
    mla_dir = cann_repo / "tasks" / "level4" / "mla"
    mla_dir.mkdir(parents=True, exist_ok=True)
    cases_payload = {
        "cases": [
            {
                "case_id": 3,
                "input_shape": [[1, 1, 128, 512], [1, 1, 128, 64], [1, 1024, 1, 512], [1, 1024, 1, 64]],
                "dtype": "float16",
                "attrs": {"is_causal": False},
            },
            {
                "case_id": 7,
                "input_shape": [[1, 1, 128, 512], [1, 1, 128, 64], [1, 4096, 1, 512], [1, 4096, 1, 64]],
                "dtype": "float16",
                "attrs": {"is_causal": False},
            },
        ]
    }
    (mla_dir / "cases.yaml").write_text(yaml.safe_dump(cases_payload), encoding="utf-8")

    fake_root = tmp_path / "fake_root"
    fake_root.mkdir(parents=True, exist_ok=True)
    (fake_root / "src").symlink_to(repo_root / "src")
    (fake_root / "targets").symlink_to(repo_root / "targets")
    (fake_root / "tests" / "ascend" / "benchmarks").mkdir(parents=True, exist_ok=True)
    (fake_root / "tests" / "ascend" / "benchmarks" / "cannbench_contract.py").write_text(
        "class Harness:\n    pass\n",
        encoding="utf-8",
    )

    tl_stub = tmp_path / "tl_stub"
    tl_stub.mkdir(parents=True, exist_ok=True)
    (tl_stub / "tilelang.py").write_text(
        "from pathlib import Path\n"
        "def compile_ascend(src, target_format='so', npu_arch='dav-3510', options=None, path_target=None):\n"
        "    p = Path(path_target)\n"
        "    p.parent.mkdir(parents=True, exist_ok=True)\n"
        "    p.write_bytes(b'ELF_TRACKB_SO')\n"
        "    return str(p)\n",
        encoding="utf-8",
    )

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_bisheng = fake_bin / "bisheng"
    fake_bisheng.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "if '-o' in args:\n"
        "    Path(args[args.index('-o') + 1]).write_bytes(b'ELF_OBJ_' * 64)\n"
        "sys.exit(0)\n",
        encoding="utf-8",
    )
    fake_bisheng.chmod(0o755)
    fake_nm = fake_bin / "llvm-nm"
    fake_nm.write_text(
        "#!/usr/bin/env python3\n"
        "print('0000000000000000 T mla_kv_reuse')\n",
        encoding="utf-8",
    )
    fake_nm.chmod(0o755)

    work_root = tmp_path / "work_root"
    tl_cache = work_root / "surrogate" / "tilelang"
    env_file = tmp_path / "env.sh"
    env_file.write_text(
        f"TILEFOUNDRY_NPU_WORK_ROOT={work_root}\n"
        f"TILEFOUNDRY_NPU_CANNBENCH_REPO={cann_repo}\n"
        f"TILEFOUNDRY_NPU_TILELANG_SOURCE={tl_stub}\n"
        f"TILEFOUNDRY_NPU_PYTHON={sys.executable}\n",
        encoding="utf-8",
    )

    mla_src = (
        "import tilelang\n"
        "def _select_mla_decode_config(b_dim=1, s_dim=1, s_kv=2048, n_q=128, n_kv=1, "
        "d_nope=512, d_rope=64, tl_dtype='float16', input_layout='BSND', is_causal=False):\n"
        "    return {'bm': 64, 'compile_flags': ['-O2']}\n"
        "def _select_mla_bnsd_config(b_dim=2, s_dim=128, s_kv=128, n_q=128, n_kv=1, "
        "d_nope=512, d_rope=64, tl_dtype='float16', input_layout='BNSD', is_causal=True):\n"
        "    return {'bm': 64, 'compile_flags': ['-O2']}\n"
        "def _select_mla_prefill_config(b_dim=2, s_dim=128, s_kv=128, n_q=128, n_kv=1, "
        "d_nope=512, d_rope=64, tl_dtype='float16', input_layout='BSND', is_causal=True):\n"
        "    return {'bm': 64, 'compile_flags': ['-O2']}\n"
        '_MLA_TRACKB_SRC = r"""\n'
        "#include <tl_templates/ascend/common.h>\n"
        "template <int BM, int DN>\n"
        '__global__ __vector__ void mla_kv_reuse(__gm__ float* q, int s_kv) {\n'
        "    q[0] = float(BM + DN + s_kv);\n"
        "}\n"
        '"""\n'
        "def _render_trackb_tu(**dims):\n"
        "    bm = int(dims.get('bm', 64) or 64)\n"
        "    dn = int(dims.get('d_nope', dims.get('D_nope', 512)) or 512)\n"
        "    return (\n"
        "        _MLA_TRACKB_SRC\n"
        "        + f'extern \"C\" int launch_mla(void* s, float* q, int skv) ' "
        "'{ mla_kv_reuse<' + str(bm) + ',' + str(dn) + '><<<28, nullptr, s>>>(q, skv); return 0; }\\n'\n"
        "    )\n"
    )
    mla_mod_path = tmp_path / "mla_cand.py"
    mla_mod_path.write_text(mla_src, encoding="utf-8")

    monkeypatch.syspath_prepend(str(tl_stub))
    ns: dict[str, object] = {}
    exec(compile(mla_src, str(mla_mod_path), "exec"), ns)
    rendered_tu = ns["_render_trackb_tu"](bm=64, d_nope=512)
    assert isinstance(rendered_tu, str)

    from tilefoundry_ascend.agent.orchestrator.gate_funnel import resolve_trackb_compile_flags

    eff_flags = resolve_trackb_compile_flags(rendered_tu, ["-O2"])
    store = ArtifactStore(tl_cache)
    sb_res = run_stage_b_ascendc_compile(
        rendered_tu,
        compile_flags=eff_flags,
        bisheng_bin=fake_bisheng,
        artifact_store=store,
    )
    assert sb_res.passed is True
    assert store.has_verified_artifact(sb_res.artifact_id) is True

    anchors = (
        AnchorCaseSpec("level4/mla", 3, "split_kv_decode", 10.0, 2.0),
        AnchorCaseSpec("level4/mla", 7, "split_kv_decode", 12.0, 3.0),
    )
    pw_res = surrogate_cannbench.prewarm_surrogate_tilelang_cache_with_stats(
        "mla",
        [("c_mla", mla_mod_path)],
        anchors,
        cannbench_repo=cann_repo,
        env_file=env_file,
        cache_dir=work_root / "surrogate" / "reference",
        tl_cache_dir=tl_cache,
        source_root=fake_root,
    )
    assert pw_res.prewarm_stats["compiled_tasks"] == 0
    assert pw_res.prewarm_stats["cache_hits"] == 1
    manifest = json.loads((tl_cache / "prewarm_manifest_v1.json").read_text(encoding="utf-8"))
    assert sb_res.artifact_id in manifest["tasks"]


def test_round_filter_skips_repair_on_capacity_overflows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify run_l0_l1_round_filter skips syntax_repair on physical overflow fault codes."""
    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=4,
        pipe_cycles={"AIV0.PIPE_V": 16},
        waterlines={},
    )
    repair_calls: list[str] = []
    monkeypatch.setattr(
        surrogate_cannbench,
        "build_syntax_repair_request",
        lambda cur_src, diag, **_kw: SimpleNamespace(
            role="syntax_repair",
            user_prompt=f"{cur_src}\n{diag}",
        ),
    )

    class _CountingRepairBackend:
        async def generate(self, req: object) -> str:
            repair_calls.append(str(getattr(req, "user_prompt", "")))
            return "def ok_fn():\n    return 'repaired'\n"

    overflow_codes = (
        "NOOP_IDENTICAL",
        "UB_SRAM_OVERFLOW",
        "UB_OVERFLOW",
        "L1_SRAM_OVERFLOW",
        "L0C_SRAM_OVERFLOW",
        "VF_STACK_OVERFLOW",
    )

    def _l0(src: str) -> StageAResult:
        if "stagea_ub" in src:
            return StageAResult(
                passed=False,
                prim_func=None,
                ledger=None,
                error_message="ub overflow",
                fault_plane="STAGE_A_STATIC",
                fault_code="UB_OVERFLOW",
                failed_tier="stage_a",
            )
        return StageAResult(passed=True, prim_func=None, ledger=ledger)

    def _l1(src: str) -> StageBResult:
        for code in overflow_codes:
            if code in src:
                return StageBResult(
                    passed=False,
                    error_message=f"failed:{code}",
                    fault_plane="STAGE_B_COMPILE",
                    fault_code=code,
                    failed_tier="stage_b",
                )
        if "repaired" in src:
            return StageBResult(passed=True)
        return StageBResult(
            passed=False,
            error_message="undeclared identifier",
            fault_plane="STAGE_B_COMPILE",
            fault_code="STAGE_B_COMPILE_FAILED",
            failed_tier="stage_b",
        )

    cands = [("c_sa_ub", "def f():\n    return 'stagea_ub'\n")] + [
        (f"c_{code.lower()}", f"def f_{i}():\n    return {code!r}\n")
        for i, code in enumerate(overflow_codes)
    ] + [("c_syntax", "def f_syn():\n    return 'syntax'\n")]

    res = asyncio.run(
        surrogate_cannbench.run_l0_l1_round_filter(
            cands,
            patch_fn=lambda snip: snip,
            l0_checker=_l0,
            l1_compiler=_l1,
            repair_backend=_CountingRepairBackend(),
            max_syntax_repairs=1,
            parent_source="def parent():\n    return 'parent'\n",
        )
    )
    assert len(repair_calls) == 1
    assert res[-1].passed is True
    assert res[-1].repaired is True
    for item in res[:-1]:
        assert item.passed is False
        assert item.repaired is False


def test_occupancy_exhausted_error_and_null_launch_us(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify OccupancyExhaustedError on rc=75 and L2BatchResults.null_launch_us."""
    req_path = tmp_path / "req.json"
    req_bytes = b'{"schema": "tilefoundry-fast-request/v1"}\n'
    req_path.write_bytes(req_bytes)
    req_sha = hashlib.sha256(req_bytes).hexdigest()
    resp_path = tmp_path / "resp.json"

    monkeypatch.setattr(surrogate_cannbench.time, "sleep", lambda _s: None)
    monkeypatch.setattr(
        surrogate_cannbench,
        "order_candidates_by_idle_preflight",
        lambda device_id, pool=(), excluded=(): ("0", "1"),
    )

    def _occ_v2(**kwargs: object) -> surrogate_cannbench._V2GateExecution:
        dev = str(kwargs.get("device_id") or "")
        return surrogate_cannbench._V2GateExecution(
            execution_id=f"exec-{dev}",
            run_id=f"run-{dev}",
            report={
                "status": "BLOCKED_ENV",
                "gates": [
                    {
                        "gateId": surrogate_cannbench.AGENT_GATES_DEVICE_FAST_GATE,
                        "status": "BLOCKED_ENV",
                        "recordedStatus": "BLOCKED_ENV",
                        "failures": ["BLOCKED_EXIT_CODE: 75 occupancy busy"],
                        "observations": {"errorCategory": "occupancy"},
                    }
                ],
            },
            verification={"satisfied": False, "failures": ["BLOCKED_EXIT_CODE: 75"]},
            device_id=dev,
        )

    monkeypatch.setattr(surrogate_cannbench, "_run_v2_gate", _occ_v2)
    with pytest.raises(surrogate_cannbench.OccupancyExhaustedError) as exc_info:
        surrogate_cannbench._submit_fast_gate(
            agent_gates_bin="agent-gates",
            root=tmp_path,
            workspace="tilefoundry",
            request_id="req-occ",
            request_relative_path="req.json",
            request_path=req_path,
            response_path=resp_path,
            request_sha256=req_sha,
            mode="batch",
            timeout_sec=15.0,
            device_id="0",
            device_pool=("0", "1"),
        )
    err = exc_info.value
    assert err.fault_plane == "INFRA_OCCUPANCY"
    assert err.exit_code == 75
    assert err.attempts >= 2
    assert "occupancy" in err.receipt_reasons

    src_root = tmp_path / "src_root"
    src_root.mkdir()
    cand_path = src_root / "cand.py"
    cand_path.write_text("def k():\n    return 1\n", encoding="utf-8")
    env_file = tmp_path / "env.sh"
    env_file.write_text(
        f"TILEFOUNDRY_NPU_WORK_ROOT={tmp_path / 'work'}\n"
        f"TILEFOUNDRY_NPU_CANNBENCH_REPO={tmp_path / 'cann'}\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(surrogate_cannbench, "_resolve_agent_gates_bin", lambda _r: "agent-gates")
    monkeypatch.setattr(surrogate_cannbench, "_agent_gates_workspace_name", lambda _r: "ws")
    monkeypatch.setattr(surrogate_cannbench, "resolve_cannbench_repo", lambda **_k: tmp_path)

    def _fake_submit_with_null(
        *,
        request_id: str,
        request_path: Path,
        response_path: Path,
        **_kw: object,
    ) -> tuple[dict[str, object], surrogate_cannbench._V2GateExecution]:
        req_payload = json.loads(request_path.read_text(encoding="utf-8"))
        c_entry = req_payload["batches"][0]["candidates"][0]
        resp = {
            "schema": surrogate_cannbench.FAST_RESPONSE_SCHEMA,
            "request_id": request_id,
            "mode": "batch",
            "status": "ok",
            "null_launch_us": 2.35,
            "candidates": {
                "mla": {
                    "c1": {
                        "module_path": c_entry["module_path"],
                        "sha256": c_entry["sha256"],
                        "post_sha256": c_entry["sha256"],
                    }
                }
            },
            "coverage": {
                "expected_candidates": 1,
                "evaluated_candidates": 1,
                "expected_msprof_targets": 0,
                "observed_msprof_targets": 0,
            },
            "operators": {
                "mla": {
                    "c1": {
                        "candidate_id": "c1",
                        "operator_name": "mla",
                        "all_passed": True,
                        "avg_speedup": 2.0,
                        "min_case_speedup": 2.0,
                        "avg_roofline_eta": 0.7,
                        "surrogate_score": 70.0,
                        "anchors": [],
                    }
                }
            },
        }
        return resp, surrogate_cannbench._V2GateExecution(
            execution_id="exec-null",
            run_id="run-null",
            receipt_path=str(response_path),
            report={"status": "PASS", "gates": []},
            verification={"satisfied": True},
        )

    monkeypatch.setattr(surrogate_cannbench, "_submit_fast_gate", _fake_submit_with_null)
    anchors = (AnchorCaseSpec("level4/mla", 3, "split_kv_decode", 10.0, 2.0),)
    by_op, _run_id, _rcp, _held = surrogate_cannbench.run_multi_operator_surrogate_batch_via_agent_gates(
        [("mla", [("c1", cand_path)], anchors)],
        source_root=src_root,
        env_file=env_file,
        prewarm_cache=False,
    )
    op_res = by_op["mla"]
    assert isinstance(op_res, surrogate_cannbench.L2BatchResults)
    assert op_res.null_launch_us == pytest.approx(2.35)


def test_collect_new_tl_cache_binaries_and_merge_skip_internal_files(tmp_path: Path) -> None:
    """Verify collect_new_tl_cache_binaries returns empty on cache hit and merge skips locks."""
    cache_dir = tmp_path / "tl_cache"
    cache_dir.mkdir()
    existing_so = cache_dir / "existing_0.so"
    existing_so.write_bytes(b"ELF_EXISTING_0")
    (cache_dir / "existing_1.so").write_bytes(b"ELF_EXISTING_1")

    snap = snapshot_tl_cache_binaries(cache_dir)
    assert collect_new_tl_cache_binaries(cache_dir, snap) == ()

    added_so = cache_dir / "new_2.so"
    added_so.write_bytes(b"ELF_NEW_2")
    assert collect_new_tl_cache_binaries(cache_dir, snap) == (Path("new_2.so"),)

    src_dir = tmp_path / "worker_cache"
    dst_dir = tmp_path / "shared_cache"
    src_dir.mkdir()
    dst_dir.mkdir()
    for fname in (
        "artifact_manifest_v1.json",
        "artifact_index_v1.jsonl",
        "prewarm_manifest_v1.json",
        "custom.lock",
    ):
        (src_dir / fname).write_text("worker_internal", encoding="utf-8")
    (src_dir / "shared.so").write_bytes(b"SAME_BYTES")
    (dst_dir / "shared.so").write_bytes(b"SAME_BYTES")
    (src_dir / "fresh.so").write_bytes(b"FRESH_BYTES")

    merge_tl_cache_tree(src_dir, dst_dir)
    assert not (dst_dir / "artifact_manifest_v1.json").exists()
    assert not (dst_dir / "artifact_index_v1.jsonl").exists()
    assert not (dst_dir / "prewarm_manifest_v1.json").exists()
    assert not (dst_dir / "custom.lock").exists()
    assert (dst_dir / "fresh.so").read_bytes() == b"FRESH_BYTES"
    assert (dst_dir / "shared.so").read_bytes() == b"SAME_BYTES"


def test_record_trackb_stage_b_artifacts_and_plan_zero_subprocesses(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify record_trackb_stage_b_artifacts_and_plan yields zero-subprocess prewarm hits."""
    repo_root = Path(__file__).resolve().parents[3]
    cann_repo = tmp_path / "cann_repo"
    mla_dir = cann_repo / "tasks" / "level4" / "mla"
    mla_dir.mkdir(parents=True, exist_ok=True)
    cases_path = mla_dir / "cases.yaml"
    cases_payload = {
        "cases": [
            {
                "case_id": 3,
                "input_shape": [
                    [1, 1, 128, 512],
                    [1, 1, 128, 64],
                    [1, 1024, 1, 512],
                    [1, 1024, 1, 64],
                ],
                "dtype": "float16",
                "attrs": {"is_causal": False},
            }
        ]
    }
    cases_path.write_text(yaml.safe_dump(cases_payload), encoding="utf-8")
    cases_sha = hashlib.sha256(cases_path.read_bytes()).hexdigest()

    fake_root = tmp_path / "fake_root"
    fake_root.mkdir(parents=True, exist_ok=True)
    (fake_root / "src").symlink_to(repo_root / "src")
    (fake_root / "targets").symlink_to(repo_root / "targets")
    (fake_root / "tests" / "ascend" / "benchmarks").mkdir(parents=True, exist_ok=True)
    (fake_root / "tests" / "ascend" / "benchmarks" / "cannbench_contract.py").write_text(
        "class Harness:\n    pass\n",
        encoding="utf-8",
    )

    tl_cache = tmp_path / "work" / "surrogate" / "tilelang"
    tl_cache.mkdir(parents=True, exist_ok=True)
    so_bin = tmp_path / "trackb_mla.so"
    so_bin.write_bytes(b"ELF_TRACKB_PLAN")

    mla_src = (
        '_MLA_TRACKB_SRC = r"""\n'
        "#include <tl_templates/ascend/common.h>\n"
        'extern "C" __global__ __vector__ void mla_kv_reuse(__gm__ float* q) {\n'
        "    q[0] = 1.0f;\n"
        "}\n"
        '"""\n'
    )
    mla_path = tmp_path / "mla_trackb.py"
    mla_path.write_text(mla_src, encoding="utf-8")
    norm_unit = (
        "#include <tl_templates/ascend/common.h>\n"
        'extern "C" __global__ __vector__ void mla_kv_reuse(__gm__ float* q) {\n'
        "    q[0] = 1.0f;\n"
        "}\n"
    )
    env_file = tmp_path / "env.sh"
    env_file.write_text(
        f"TILEFOUNDRY_NPU_CANNBENCH_REPO={cann_repo}\n"
        f"TILEFOUNDRY_NPU_PYTHON={sys.executable}\n",
        encoding="utf-8",
    )
    recorded = surrogate_cannbench.record_trackb_stage_b_artifacts_and_plan(
        module_source=mla_src,
        operator_name="mla",
        anchor_entries=[(3, "split_kv_decode", norm_unit, ["-O2"], [so_bin])],
        tl_cache_dir=tl_cache,
        cases_yaml_sha256=cases_sha,
        env_vals={"TILEFOUNDRY_NPU_PYTHON": sys.executable},
    )
    assert recorded == 1

    def _fail_subprocess(*_a: object, **_kw: object) -> object:
        raise AssertionError("prewarm should hit plan_index with zero subprocesses")

    monkeypatch.setattr(surrogate_cannbench.subprocess, "run", _fail_subprocess)
    anchors = (AnchorCaseSpec("level4/mla", 3, "split_kv_decode", 10.0, 2.0),)
    pw_res = surrogate_cannbench.prewarm_surrogate_tilelang_cache_with_stats(
        "mla",
        [("c_tb", mla_path)],
        anchors,
        cannbench_repo=cann_repo,
        env_file=env_file,
        cache_dir=tmp_path / "work" / "surrogate" / "reference",
        tl_cache_dir=tl_cache,
        source_root=fake_root,
    )
    assert pw_res.prewarm_stats["compiled_tasks"] == 0
    assert pw_res.prewarm_stats["cache_hits"] == 1


def test_tracka_anchor_case_specs_and_compile_for_cache_driver_prewarm_agreement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify anchor_case_specs and compile_for_cache align keys for zero-subprocess prewarm."""
    repo_root = Path(__file__).resolve().parents[3]
    cann_repo = tmp_path / "cann_repo"
    task_dir = cann_repo / "tasks" / "level3" / "add_rms_norm_dynamic_quant"
    task_dir.mkdir(parents=True, exist_ok=True)
    cases_path = task_dir / "cases.yaml"
    cases_payload = {
        "cases": [
            {
                "case_id": 1,
                "input_shape": [[8, 128], [8, 128], [128]],
                "dtype": "float16",
                "attrs": {"epsilon": 1e-6},
            },
            {
                "case_id": 2,
                "input_shape": [[32, 128], [32, 128], [128]],
                "dtype": "float16",
                "attrs": {"epsilon": 1e-6},
            },
        ]
    }
    cases_path.write_text(yaml.safe_dump(cases_payload), encoding="utf-8")
    cases_sha = hashlib.sha256(cases_path.read_bytes()).hexdigest()

    fake_root = tmp_path / "fake_root"
    fake_root.mkdir(parents=True, exist_ok=True)
    (fake_root / "src").symlink_to(repo_root / "src")
    (fake_root / "targets").symlink_to(repo_root / "targets")
    (fake_root / "tests" / "ascend" / "benchmarks").mkdir(parents=True, exist_ok=True)
    (fake_root / "tests" / "ascend" / "benchmarks" / "cannbench_contract.py").write_text(
        "class Harness:\n    pass\n",
        encoding="utf-8",
    )

    tl_stub = tmp_path / "tl_stub"
    tl_pkg_backend = tl_stub / "tilelang" / "backend"
    tl_pkg_backend.mkdir(parents=True)
    (tl_pkg_backend / "__init__.py").write_text("", encoding="utf-8")
    (tl_pkg_backend / "module.py").write_text(
        "from types import SimpleNamespace\n"
        "import contextlib\n"
        "def create_backend_context(_t):\n"
        "    return SimpleNamespace(target=contextlib.nullcontext())\n",
        encoding="utf-8",
    )
    (tl_stub / "tvm.py").write_text(
        "import contextlib\n"
        "class _Target:\n"
        "    @staticmethod\n"
        "    def current(allow_none=True):\n"
        "        return 'ascend'\n"
        "class target:\n"
        "    Target = _Target\n"
        "class transform:\n"
        "    @staticmethod\n"
        "    def PassContext(config=None):\n"
        "        return contextlib.nullcontext()\n"
        "class tirx:\n"
        "    PrimFunc = dict\n"
        "class IRModule:\n"
        "    pass\n",
        encoding="utf-8",
    )
    (tl_stub / "tilelang" / "__init__.py").write_text(
        "import os, tvm\n"
        "from pathlib import Path\n"
        "class _Kernel:\n"
        "    def __init__(self, kp, cce):\n"
        "        self.kernel_path = str(kp)\n"
        "        self._cce = cce\n"
        "    def get_kernel_source(self):\n"
        "        return self._cce\n"
        "def lower(fn, target='ascend', enable_device_compile=True):\n"
        "    cache_dir = Path(os.environ['TILELANG_CACHE_DIR'])\n"
        "    cache_dir.mkdir(parents=True, exist_ok=True)\n"
        "    so_p = cache_dir / f\"k_{fn['d']}_{fn['m']}.so\"\n"
        "    so_p.write_bytes(b'ELF_LOWER_SO')\n"
        "    return SimpleArt(so_p, '// diag cce')\n"
        "class SimpleArt:\n"
        "    def __init__(self, kp, cce):\n"
        "        self.kernel_path = str(kp)\n"
        "        self.kernel_source = cce\n"
        "        self.device_mod = cce\n"
        "def compile(prog, target='ascend', out_idx=None, pass_configs=None, compile_flags=None):\n"
        "    flags = list(compile_flags or [])\n"
        "    assert '--cce-res-usage' not in flags\n"
        "    cache_dir = Path(os.environ['TILELANG_CACHE_DIR'])\n"
        "    cache_dir.mkdir(parents=True, exist_ok=True)\n"
        "    so_p = cache_dir / f\"k_{prog['d']}_{prog['m']}.so\"\n"
        "    if not so_p.exists():\n"
        "        so_p.write_bytes(b'ELF_COMPILE_SO')\n"
        "    return _Kernel(so_p, '// compiled cce')\n",
        encoding="utf-8",
    )

    mod_src = (
        "import tilelang\n"
        "def _select_add_rms_norm_dynamic_quant_config("
        "d_dim=128, tl_dtype='float16', m_rows=8, schedule_mode='tuned'):\n"
        "    return {'tile_m': 8, 'pass_configs': {}, 'compile_flags': ['-O2']}\n"
        "def _build_add_rms_norm_dynamic_quant_prim_func("
        "d_dim=128, tl_dtype='float16', tile_m=8, schedule_mode='tuned'):\n"
        "    return {'d': int(d_dim), 'm': int(tile_m)}\n"
        "def _get_kernel(d_dim, tl_dtype, m_rows, schedule_mode, *, "
        "builder_fn=None, pass_configs=None, compile_flags=None):\n"
        "    prog = builder_fn(d_dim, tl_dtype, 8, schedule_mode)\n"
        "    return tilelang.compile(prog, target='ascend', out_idx=[3], "
        "pass_configs=pass_configs, compile_flags=compile_flags)\n"
    )
    cand_path = tmp_path / "cand_arndq.py"
    cand_path.write_text(mod_src, encoding="utf-8")
    emit_json = tmp_path / "emit_bins.json"
    drv_src = build_stage_b_driver_source(
        cand_path,
        factory_name="_build_add_rms_norm_dynamic_quant_prim_func",
        factory_kwargs={"d_dim": 128, "tl_dtype": "float16", "schedule_mode": "tuned"},
        config_selector_name="_select_add_rms_norm_dynamic_quant_config",
        dims={"d_dim": 128, "tl_dtype": "float16", "m_rows": 8},
        compile_for_cache=True,
        emit_cache_bins_path=emit_json,
    )
    drv_env = {
        **dict(surrogate_cannbench.os.environ),
        "PYTHONPATH": str(tl_stub),
        "TILELANG_CACHE_DIR": str(tmp_path / "drv_cache"),
    }
    proc = subprocess.run(
        [sys.executable, "-c", drv_src],
        text=True,
        capture_output=True,
        env=drv_env,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    emitted = json.loads(emit_json.read_text(encoding="utf-8"))
    assert len(emitted) == 1 and emitted[0].endswith("k_128_8.so")

    work_root = tmp_path / "work_root"
    tl_cache = work_root / "surrogate" / "tilelang"
    env_file = tmp_path / "env.sh"
    env_file.write_text(
        f"TILEFOUNDRY_NPU_WORK_ROOT={work_root}\n"
        f"TILEFOUNDRY_NPU_CANNBENCH_REPO={cann_repo}\n"
        f"TILEFOUNDRY_NPU_TILELANG_SOURCE={tl_stub}\n"
        f"TILEFOUNDRY_NPU_PYTHON={sys.executable}\n",
        encoding="utf-8",
    )
    anchor_case_specs = [
        (1, "default", {"d_dim": 128, "tl_dtype": "float16", "m_rows": 8, "is_causal": False}),
        (2, "default", {"d_dim": 128, "tl_dtype": "float16", "m_rows": 32, "is_causal": False}),
    ]
    res = surrogate_cannbench.run_l1_compile_check(
        mod_src,
        builder_name="_build_add_rms_norm_dynamic_quant_prim_func",
        builder_kwargs={
            "d_dim": 128,
            "tl_dtype": "float16",
            "schedule_mode": "tuned",
            "cache_g_f32": True,
        },
        config_selector_name="_select_add_rms_norm_dynamic_quant_config",
        pass_configs={},
        compile_flags=["-O2"],
        dims={"d_dim": 128, "tl_dtype": "float16", "m_rows": 8, "is_causal": False},
        source_root=fake_root,
        env_file=env_file,
        tl_cache_dir=tl_cache,
        anchor_case_specs=anchor_case_specs,
        operator_name="add_rms_norm_dynamic_quant",
        cases_yaml_sha256=cases_sha,
    )
    assert res.passed is True
    assert (tl_cache / "prewarm_manifest_v1.json").is_file()

    def _fail_subprocess(*_a: object, **_kw: object) -> object:
        raise AssertionError("prewarm should hit plan_index with zero subprocesses")

    monkeypatch.setattr(surrogate_cannbench.subprocess, "run", _fail_subprocess)
    anchors = (
        AnchorCaseSpec("level3/add_rms_norm_dynamic_quant", 1, "default", 5.0, 2.0),
        AnchorCaseSpec("level3/add_rms_norm_dynamic_quant", 2, "default", 9.0, 4.0),
    )
    pw_res = surrogate_cannbench.prewarm_surrogate_tilelang_cache_with_stats(
        "add_rms_norm_dynamic_quant",
        [("c_cand", cand_path)],
        anchors,
        cannbench_repo=cann_repo,
        env_file=env_file,
        cache_dir=work_root / "surrogate" / "reference",
        tl_cache_dir=tl_cache,
        source_root=fake_root,
    )
    assert pw_res.prewarm_stats["compiled_tasks"] == 0
    assert pw_res.prewarm_stats["cache_hits"] == 2


def test_round_filter_skip_repair_when_survivors_exist(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify skip_repair_when_survivors_exist skips repair when a survivor exists."""
    ledger = build_objective_physical_ledger(
        analysis_mode="readonly_envelope",
        is_legal=True,
        steady_state_ii=4,
        pipe_cycles={"AIV0.PIPE_V": 16},
        waterlines={},
    )
    repair_calls: list[str] = []
    monkeypatch.setattr(
        surrogate_cannbench,
        "build_syntax_repair_request",
        lambda cur_src, diag, **_kw: SimpleNamespace(
            role="syntax_repair",
            user_prompt=f"{cur_src}\n{diag}",
        ),
    )

    class _RepairBackend:
        async def generate(self, req: object) -> str:
            repair_calls.append(str(getattr(req, "user_prompt", "")))
            return "def repaired_ok():\n    return 'repaired'\n"

    def _l0(_src: str) -> StageAResult:
        return StageAResult(passed=True, prim_func=None, ledger=ledger)

    def _l1(src: str) -> StageBResult:
        if "clean_ok" in src or "repaired" in src:
            return StageBResult(passed=True)
        return StageBResult(
            passed=False,
            error_message="undeclared identifier",
            fault_plane="STAGE_B_COMPILE",
            fault_code="STAGE_B_COMPILE_FAILED",
            failed_tier="stage_b",
        )

    res_with_survivor = asyncio.run(
        surrogate_cannbench.run_l0_l1_round_filter(
            [
                ("c_ok", "def f_ok():\n    return 'clean_ok'\n"),
                ("c_bad", "def f_bad():\n    return 'broken'\n"),
            ],
            patch_fn=lambda snip: snip,
            l0_checker=_l0,
            l1_compiler=_l1,
            repair_backend=_RepairBackend(),
            max_syntax_repairs=1,
            parent_source="def parent():\n    return 'parent'\n",
            skip_repair_when_survivors_exist=True,
        )
    )
    assert len(repair_calls) == 0
    assert res_with_survivor.wall_repair_sec == 0.0
    assert res_with_survivor[0].passed is True
    assert res_with_survivor[1].passed is False

    res_no_survivor = asyncio.run(
        surrogate_cannbench.run_l0_l1_round_filter(
            [
                ("c_bad1", "def f_b1():\n    return 'broken1'\n"),
                ("c_bad2", "def f_b2():\n    return 'broken2'\n"),
            ],
            patch_fn=lambda snip: snip,
            l0_checker=_l0,
            l1_compiler=_l1,
            repair_backend=_RepairBackend(),
            max_syntax_repairs=1,
            parent_source="def parent():\n    return 'parent'\n",
            skip_repair_when_survivors_exist=True,
        )
    )
    assert len(repair_calls) == 2
    assert all(item.passed and item.repaired for item in res_no_survivor)


def test_run_l1_compile_check_skips_cached_keys_and_populates_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify run_l1_compile_check skips compile workers on cache hits and updates manifest."""
    repo_root = Path(__file__).resolve().parents[3]
    cann_repo = tmp_path / "cann_repo"
    task_dir = cann_repo / "tasks" / "level3" / "add_rms_norm_dynamic_quant"
    task_dir.mkdir(parents=True, exist_ok=True)
    cases_path = task_dir / "cases.yaml"
    cases_payload = {
        "cases": [
            {
                "case_id": 1,
                "input_shape": [[8, 128], [8, 128], [128]],
                "dtype": "float16",
                "attrs": {"epsilon": 1e-6},
            },
            {
                "case_id": 2,
                "input_shape": [[32, 128], [32, 128], [128]],
                "dtype": "float16",
                "attrs": {"epsilon": 1e-6},
            },
        ]
    }
    cases_path.write_text(yaml.safe_dump(cases_payload), encoding="utf-8")
    cases_sha = hashlib.sha256(cases_path.read_bytes()).hexdigest()

    fake_root = tmp_path / "fake_root"
    fake_root.mkdir(parents=True, exist_ok=True)
    (fake_root / "src").symlink_to(repo_root / "src")
    (fake_root / "targets").symlink_to(repo_root / "targets")
    (fake_root / "tests" / "ascend" / "benchmarks").mkdir(parents=True, exist_ok=True)
    (fake_root / "tests" / "ascend" / "benchmarks" / "cannbench_contract.py").write_text(
        "class Harness:\n    pass\n",
        encoding="utf-8",
    )

    tl_stub = tmp_path / "tl_stub"
    tl_pkg_backend = tl_stub / "tilelang" / "backend"
    tl_pkg_backend.mkdir(parents=True)
    (tl_pkg_backend / "__init__.py").write_text("", encoding="utf-8")
    (tl_pkg_backend / "module.py").write_text(
        "from types import SimpleNamespace\n"
        "import contextlib\n"
        "def create_backend_context(_t):\n"
        "    return SimpleNamespace(target=contextlib.nullcontext())\n",
        encoding="utf-8",
    )
    (tl_stub / "tvm.py").write_text(
        "import contextlib\n"
        "class _Target:\n"
        "    @staticmethod\n"
        "    def current(allow_none=True):\n"
        "        return 'ascend'\n"
        "class target:\n"
        "    Target = _Target\n"
        "class transform:\n"
        "    @staticmethod\n"
        "    def PassContext(config=None):\n"
        "        return contextlib.nullcontext()\n"
        "class tirx:\n"
        "    PrimFunc = dict\n"
        "class IRModule:\n"
        "    pass\n",
        encoding="utf-8",
    )
    (tl_stub / "tilelang" / "__init__.py").write_text(
        "import os, tvm\n"
        "from pathlib import Path\n"
        "class _Kernel:\n"
        "    def __init__(self, kp, cce):\n"
        "        self.kernel_path = str(kp)\n"
        "        self._cce = cce\n"
        "    def get_kernel_source(self):\n"
        "        return self._cce\n"
        "class _Art:\n"
        "    def __init__(self, kp, cce):\n"
        "        self.kernel_path = str(kp)\n"
        "        self.kernel_source = cce\n"
        "        self.device_mod = cce\n"
        "def lower(fn, target='ascend', enable_device_compile=True):\n"
        "    cache_dir = Path(os.environ['TILELANG_CACHE_DIR'])\n"
        "    cache_dir.mkdir(parents=True, exist_ok=True)\n"
        "    so_p = cache_dir / f\"k_{fn['d']}_{fn['m']}.so\"\n"
        "    so_p.write_bytes(b'ELF_LOWER_SO')\n"
        "    return _Art(so_p, '// diag cce')\n"
        "def compile(prog, target='ascend', out_idx=None, pass_configs=None, compile_flags=None):\n"
        "    cache_dir = Path(os.environ['TILELANG_CACHE_DIR'])\n"
        "    cache_dir.mkdir(parents=True, exist_ok=True)\n"
        "    so_p = cache_dir / f\"k_{prog['d']}_{prog['m']}.so\"\n"
        "    if not so_p.exists():\n"
        "        so_p.write_bytes(b'ELF_COMPILE_SO')\n"
        "    return _Kernel(so_p, '// compiled cce')\n",
        encoding="utf-8",
    )

    mod_src = (
        "import tilelang\n"
        "def _select_add_rms_norm_dynamic_quant_config("
        "d_dim=128, tl_dtype='float16', m_rows=8, schedule_mode='tuned'):\n"
        "    return {'tile_m': int(m_rows), 'pass_configs': {}, 'compile_flags': ['-O2']}\n"
        "def _build_add_rms_norm_dynamic_quant_prim_func("
        "d_dim=128, tl_dtype='float16', tile_m=8, schedule_mode='tuned'):\n"
        "    return {'d': int(d_dim), 'm': int(tile_m)}\n"
        "def _get_kernel(d_dim, tl_dtype, m_rows, schedule_mode, *, "
        "builder_fn=None, pass_configs=None, compile_flags=None):\n"
        "    prog = builder_fn(d_dim, tl_dtype, m_rows, schedule_mode)\n"
        "    return tilelang.compile(prog, target='ascend', out_idx=[3], "
        "pass_configs=pass_configs, compile_flags=compile_flags)\n"
    )
    work_root = tmp_path / "work_root"
    tl_cache = work_root / "surrogate" / "tilelang"
    env_file = tmp_path / "env.sh"
    env_file.write_text(
        f"TILEFOUNDRY_NPU_WORK_ROOT={work_root}\n"
        f"TILEFOUNDRY_NPU_CANNBENCH_REPO={cann_repo}\n"
        f"TILEFOUNDRY_NPU_TILELANG_SOURCE={tl_stub}\n"
        f"TILEFOUNDRY_NPU_PYTHON={sys.executable}\n",
        encoding="utf-8",
    )
    anchor_case_specs = [
        (1, "default", {"d_dim": 128, "tl_dtype": "float16", "m_rows": 8, "is_causal": False}),
        (2, "default", {"d_dim": 128, "tl_dtype": "float16", "m_rows": 32, "is_causal": False}),
    ]
    static_widths = [
        {"d_dim": 128, "tl_dtype": "float16", "m_rows": 8, "is_causal": False},
        {"d_dim": 512, "tl_dtype": "float16", "m_rows": 8, "is_causal": False},
    ]

    spawned_scripts: list[str] = []
    real_subproc_run = subprocess.run

    def _recording_run(cmd, *args, **kwargs):
        if isinstance(cmd, (list, tuple)) and len(cmd) >= 2:
            spawned_scripts.append(Path(str(cmd[-1])).name)
        return real_subproc_run(cmd, *args, **kwargs)

    monkeypatch.setattr(surrogate_cannbench.subprocess, "run", _recording_run)

    call_kwargs = {
        "builder_name": "_build_add_rms_norm_dynamic_quant_prim_func",
        "builder_kwargs": {
            "d_dim": 128,
            "tl_dtype": "float16",
            "schedule_mode": "tuned",
            "cache_g_f32": True,
        },
        "config_selector_name": "_select_add_rms_norm_dynamic_quant_config",
        "pass_configs": {},
        "compile_flags": ["-O2"],
        "dims": {"d_dim": 128, "tl_dtype": "float16", "m_rows": 8, "is_causal": False},
        "static_spec_dims_list": static_widths,
        "source_root": fake_root,
        "env_file": env_file,
        "tl_cache_dir": tl_cache,
        "anchor_case_specs": anchor_case_specs,
        "operator_name": "add_rms_norm_dynamic_quant",
        "cases_yaml_sha256": cases_sha,
        "regime_tag": "default",
    }

    res1 = surrogate_cannbench.run_l1_compile_check(mod_src, **call_kwargs)
    assert res1.passed is True
    assert spawned_scripts.count("anchor_plan.py") == 1
    assert spawned_scripts.count("diag_driver.py") == 1
    assert spawned_scripts.count("compile_driver.py") == 3

    manifest_path = tl_cache / "prewarm_manifest_v1.json"
    manifest1 = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest1["tasks"]) >= 3
    assert len(manifest1["plan_index"]) >= 3

    spawned_scripts.clear()
    manifest_path.unlink()

    res2 = surrogate_cannbench.run_l1_compile_check(mod_src, **call_kwargs)
    assert res2.passed is True
    assert spawned_scripts.count("anchor_plan.py") == 1
    assert spawned_scripts.count("diag_driver.py") == 0
    assert spawned_scripts.count("compile_driver.py") == 0
    assert res2.spill_bytes == res1.spill_bytes
    assert res2.vloopv2_hit == res1.vloopv2_hit
    assert res2.compiler_summary == res1.compiler_summary
    assert res2.microarch_profile == res1.microarch_profile

    manifest2 = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert set(manifest2["tasks"]) == set(manifest1["tasks"])
    assert set(manifest2["plan_index"]) == set(manifest1["plan_index"])
    store = ArtifactStore(tl_cache)
    assert store.has_verified_artifact(res2.artifact_id) is True

    tb_src = (
        '_TB_CPP = r"""\n'
        "#include <tl_templates/ascend/common.h>\n"
        'extern "C" __global__ __vector__ void k() {}\n'
        '"""\n'
        "def _render_trackb_tu(**_kw):\n"
        "    return _TB_CPP\n"
    )
    cand_tb = tmp_path / "cand_tb.py"
    cand_tb.write_text(tb_src, encoding="utf-8")
    plan_script = tmp_path / "tb_anchor_plan.py"
    plan_script.write_text(
        surrogate_cannbench._L1_ANCHOR_PLAN_TEMPLATE.format(
            ascend_target_dir=str(fake_root / "targets" / "ascend"),
            op_name="mla",
            mod_path=str(cand_tb),
            anchor_specs_json=json.dumps([[3, "split_kv_decode", {"d_nope": 512}]]),
            def_builder="_render_trackb_tu",
            def_bk_json=json.dumps({"cache_g_f32": True}),
            def_sel="",
            def_pc_json=json.dumps({}),
            def_cf_json=json.dumps([]),
        ),
        encoding="utf-8",
    )
    tb_plan = real_subproc_run(
        [sys.executable, str(plan_script)],
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert tb_plan.returncode == 0, tb_plan.stderr
    assert json.loads(tb_plan.stdout.strip().splitlines()[-1]) == {"tasks": []}


def test_tl_cache_tree_skips_blob_and_diag_dirs(tmp_path: Path) -> None:
    """Verify seed, merge, snapshot, and collect skip artifact_blobs and stage_b_diag_v1."""
    src_dir = tmp_path / "src_cache"
    dst_dir = tmp_path / "dst_cache"
    merge_dir = tmp_path / "merge_cache"
    (src_dir / "sub" / "empty_ok").mkdir(parents=True)
    (src_dir / "artifact_blobs" / "deep").mkdir(parents=True)
    (src_dir / "nested" / "artifact_blobs").mkdir(parents=True)
    (src_dir / "stage_b_diag_v1" / "deep").mkdir(parents=True)
    (src_dir / "nested" / "stage_b_diag_v1").mkdir(parents=True)

    (src_dir / "keep.so").write_bytes(b"ELF_KEEP")
    (src_dir / "sub" / "nested.so").write_bytes(b"ELF_NESTED")
    (src_dir / "compiled_entries.jsonl").write_text('{"k": 1}\n', encoding="utf-8")
    (src_dir / "prewarm_manifest_v1.json").write_text("{}", encoding="utf-8")
    (src_dir / "artifact_blobs" / "deep" / "blob.so").write_bytes(b"ELF_BLOB")
    (src_dir / "nested" / "artifact_blobs" / "blob2.so").write_bytes(b"ELF_BLOB2")
    (src_dir / "stage_b_diag_v1" / "deep" / "diag.so").write_bytes(b"ELF_DIAG")
    (src_dir / "nested" / "stage_b_diag_v1" / "diag2.so").write_bytes(b"ELF_DIAG2")

    seed_tl_cache_tree(src_dir, dst_dir)
    assert (dst_dir / "keep.so").read_bytes() == b"ELF_KEEP"
    assert (dst_dir / "sub" / "nested.so").read_bytes() == b"ELF_NESTED"
    assert (dst_dir / "sub" / "empty_ok").is_dir()
    assert not (dst_dir / "compiled_entries.jsonl").exists()
    assert not (dst_dir / "prewarm_manifest_v1.json").exists()
    assert not (dst_dir / "artifact_blobs").exists()
    assert not (dst_dir / "nested" / "artifact_blobs").exists()
    assert not (dst_dir / "stage_b_diag_v1").exists()
    assert not (dst_dir / "nested" / "stage_b_diag_v1").exists()

    snap = snapshot_tl_cache_binaries(src_dir)
    assert set(snap) == {Path("keep.so"), Path("sub/nested.so")}
    assert collect_new_tl_cache_binaries(src_dir, {}) == (
        Path("keep.so"),
        Path("sub/nested.so"),
    )

    merge_tl_cache_tree(src_dir, merge_dir)
    assert (merge_dir / "keep.so").read_bytes() == b"ELF_KEEP"
    assert (merge_dir / "sub" / "nested.so").read_bytes() == b"ELF_NESTED"
    assert (merge_dir / "compiled_entries.jsonl").read_text(encoding="utf-8") == '{"k": 1}\n'
    assert not (merge_dir / "artifact_blobs").exists()
    assert not (merge_dir / "nested" / "artifact_blobs").exists()
    assert not (merge_dir / "stage_b_diag_v1").exists()
    assert not (merge_dir / "nested" / "stage_b_diag_v1").exists()


def test_merge_tl_cache_tree_skips_hardlinked_same_inode_without_reading_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify merge_tl_cache_tree short-circuits same-inode hardlinks before read_bytes."""
    shared_dir = tmp_path / "shared_cache"
    worker_dir = tmp_path / "worker_cache"
    shared_dir.mkdir()
    (shared_dir / "seeded.so").write_bytes(b"ELF_HARDLINKED")

    seed_tl_cache_tree(shared_dir, worker_dir)
    assert (worker_dir / "seeded.so").stat().st_ino == (shared_dir / "seeded.so").stat().st_ino

    (worker_dir / "new_out.so").write_bytes(b"ELF_NEW_OUT")
    (shared_dir / "separate.so").write_bytes(b"SAME_CONTENT")
    (worker_dir / "separate.so").write_bytes(b"SAME_CONTENT")

    orig_read_bytes = Path.read_bytes
    read_paths: list[Path] = []

    def _guarded_read_bytes(self: Path) -> bytes:
        read_paths.append(self)
        if self.name == "seeded.so":
            raise AssertionError(f"read_bytes called on hardlinked file: {self}")
        return orig_read_bytes(self)

    with monkeypatch.context() as mp:
        mp.setattr(Path, "read_bytes", _guarded_read_bytes)
        merge_tl_cache_tree(worker_dir, shared_dir)

    assert all(p.name != "seeded.so" for p in read_paths)
    assert any(p.name == "separate.so" for p in read_paths)
    assert (shared_dir / "new_out.so").read_bytes() == b"ELF_NEW_OUT"
    assert (shared_dir / "seeded.so").read_bytes() == b"ELF_HARDLINKED"
