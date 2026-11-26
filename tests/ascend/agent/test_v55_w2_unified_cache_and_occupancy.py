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
    run_stage_b_ascendc_compile,
    run_stage_b_compile,
)
from tilefoundry_ascend.ascend.compile_cache.artifact_store import (
    ArtifactStore,
    compute_artifact_id,
    compute_compile_task_key,
    resolve_unified_tl_cache_dir,
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
