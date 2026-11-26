"""CPU unit tests for v5.3 WP-4 prewarm cache, dual-shape Stage-B, and batch isolation."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from tests.ascend.agent import test_surrogate_batch_npu as batch_harness
from tilefoundry_ascend.agent.campaign import marathon, surrogate_cannbench
from tilefoundry_ascend.agent.campaign.anchor_specs import AnchorCaseSpec
from tilefoundry_ascend.agent.campaign.campaign_runner import resolve_campaign_device_plan


def _setup_prewarm_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
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
            },
            {
                "case_id": 4,
                "input_shape": [[8, 1, 16, 1000], [1, 16, 1000], [1000], [1000]],
                "dtype": "bfloat16",
                "attrs": {"alpha": 0.9, "epsilon": 1e-6},
            },
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

    tl_stub_dir = tmp_path / "tl_stub"
    tl_stub_dir.mkdir(parents=True, exist_ok=True)
    (tl_stub_dir / "tilelang.py").write_text(
        "import json, os\n"
        "from pathlib import Path\n"
        "def compile(prog, target='ascend', out_idx=None, pass_configs=None, compile_flags=None):\n"
        "    cache_dir = Path(os.environ['TILELANG_CACHE_DIR'])\n"
        "    cache_dir.mkdir(parents=True, exist_ok=True)\n"
        "    if prog.get('fail'):\n"
        "        raise RuntimeError('simulated_bisheng_compile_failure')\n"
        "    log_path = cache_dir / 'compiled_entries.jsonl'\n"
        "    with log_path.open('a', encoding='utf-8') as fh:\n"
        "        fh.write(json.dumps({'prog': prog, 'target': target}) + '\\n')\n"
        "    so_path = cache_dir / f\"k_{prog.get('tag', 'ok')}_{prog.get('n', 0)}.so\"\n"
        "    so_path.write_bytes(b'ELF_STUB')\n"
        "    return {'cached': prog}\n",
        encoding="utf-8",
    )

    work_root = tmp_path / "work_root"
    env_file = tmp_path / "env.sh"
    env_file.write_text(
        f"TILEFOUNDRY_NPU_WORK_ROOT={work_root}\n"
        f"TILEFOUNDRY_NPU_CANNBENCH_REPO={cann_repo}\n"
        f"TILEFOUNDRY_NPU_TILELANG_SOURCE={tl_stub_dir}\n"
        f"TILEFOUNDRY_NPU_PYTHON={sys.executable}\n",
        encoding="utf-8",
    )
    return fake_root, cann_repo, tl_stub_dir, work_root, env_file


def _write_ar_candidate(path: Path, *, tag: str, fail_compile: bool = False) -> Path:
    fail_flag = "True" if fail_compile else "False"
    path.write_text(
        "import tilelang\n"
        "def _select_ar_small_decode_config(n_depth=2, d_dim=4096, tl_dtype='bfloat16', "
        "schedule_mode='tuned', m_rows=1):\n"
        "    return {'tile_m': 1, 'pass_configs': {}, 'compile_flags': []}\n"
        "def _build_ar_small_decode_func(n_depth=2, d_dim=4096, tl_dtype='bfloat16', "
        "tile_m=1, schedule_mode='tuned'):\n"
        f"    return {{'tag': {tag!r}, 'n': n_depth, 'fail': {fail_flag}}}\n"
        "def _select_ar_unaligned_config(n_depth=8, d_dim=1000, tl_dtype='bfloat16', "
        "schedule_mode='tuned', m_rows=4):\n"
        "    return {'tile_m': 1, 'pass_configs': {}, 'compile_flags': []}\n"
        "def _build_ar_unaligned_func(n_depth=8, d_dim=1000, tl_dtype='bfloat16', "
        "tile_m=1, schedule_mode='tuned'):\n"
        f"    return {{'tag': {tag!r}, 'n': n_depth, 'fail': {fail_flag}}}\n"
        "def _get_kernel(n_depth, d_dim, tl_dtype, tile_m, schedule_mode, *, "
        "builder_fn=None, pass_configs=None, compile_flags=None):\n"
        "    prog = builder_fn(n_depth, d_dim, tl_dtype, tile_m, schedule_mode)\n"
        "    return tilelang.compile(prog, target='ascend', out_idx=[4], "
        "pass_configs=pass_configs, compile_flags=compile_flags)\n",
        encoding="utf-8",
    )
    return path


def test_prewarm_manifest_cache_hit_skips_subprocess_and_invalidates_on_stamp_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hit manifest with zero subprocesses on reuse; invalidate on stamp or artifact loss."""
    fake_root, cann_repo, tl_stub_dir, work_root, env_file = _setup_prewarm_fixture(tmp_path)
    cand_a = _write_ar_candidate(tmp_path / "cand_a.py", tag="a")
    anchors = (
        AnchorCaseSpec("level4/attention_residual", 2, "small_decode", 1.5, 0.5),
        AnchorCaseSpec("level4/attention_residual", 4, "unaligned_d", 9.9, 3.3),
    )
    ref_cache = work_root / "surrogate" / "reference"
    tl_cache = work_root / "surrogate" / "tilelang"

    spawn_count = 0
    real_run = subprocess.run

    def _counting_run(*args, **kwargs):
        nonlocal spawn_count
        spawn_count += 1
        return real_run(*args, **kwargs)

    monkeypatch.setattr(surrogate_cannbench.subprocess, "run", _counting_run)

    res1 = surrogate_cannbench.prewarm_surrogate_tilelang_cache_with_stats(
        "attention_residual",
        [("c_a", cand_a)],
        anchors,
        cannbench_repo=cann_repo,
        env_file=env_file,
        cache_dir=ref_cache,
        tl_cache_dir=tl_cache,
        source_root=fake_root,
    )
    assert res1.prewarm_stats == {
        "total_tasks": 2,
        "cache_hits": 0,
        "compiled_tasks": 2,
        "failed_candidates": 0,
    }
    assert spawn_count == 3
    assert (tl_cache / "prewarm_manifest_v1.json").is_file()

    spawn_count = 0
    res2 = surrogate_cannbench.prewarm_surrogate_tilelang_cache_with_stats(
        "attention_residual",
        [("c_a", cand_a)],
        anchors,
        cannbench_repo=cann_repo,
        env_file=env_file,
        cache_dir=ref_cache,
        tl_cache_dir=tl_cache,
        source_root=fake_root,
    )
    assert spawn_count == 0
    assert res2.prewarm_stats == {
        "total_tasks": 2,
        "cache_hits": 2,
        "compiled_tasks": 0,
        "failed_candidates": 0,
    }

    stub_file = tl_stub_dir / "tilelang.py"
    stub_file.write_text(stub_file.read_text(encoding="utf-8") + "\n# rebuilt\n", encoding="utf-8")
    spawn_count = 0
    res3 = surrogate_cannbench.prewarm_surrogate_tilelang_cache_with_stats(
        "attention_residual",
        [("c_a", cand_a)],
        anchors,
        cannbench_repo=cann_repo,
        env_file=env_file,
        cache_dir=ref_cache,
        tl_cache_dir=tl_cache,
        source_root=fake_root,
    )
    assert spawn_count == 3
    assert res3.prewarm_stats["compiled_tasks"] == 2
    assert res3.prewarm_stats["cache_hits"] == 0

    for artifact in list(tl_cache.rglob("*.so")) + list(tl_cache.rglob("compiled_entries.jsonl")):
        artifact.unlink()
    spawn_count = 0
    res4 = surrogate_cannbench.prewarm_surrogate_tilelang_cache_with_stats(
        "attention_residual",
        [("c_a", cand_a)],
        anchors,
        cannbench_repo=cann_repo,
        env_file=env_file,
        cache_dir=ref_cache,
        tl_cache_dir=tl_cache,
        source_root=fake_root,
    )
    assert spawn_count == 3
    assert res4.prewarm_stats["compiled_tasks"] == 2


def test_prewarm_fault_isolation_strips_bad_candidate_and_skips_device_when_none_valid(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bad non-incumbent candidate fails cleanly in prewarm while valid sibling proceeds."""
    fake_root, cann_repo, _tl_stub, _work_root, env_file = _setup_prewarm_fixture(tmp_path)
    inc_cand = _write_ar_candidate(tmp_path / "inc.py", tag="inc")
    good_cand = _write_ar_candidate(tmp_path / "good.py", tag="good")
    bad_cand = _write_ar_candidate(tmp_path / "bad.py", tag="bad", fail_compile=True)
    anchors = (AnchorCaseSpec("level4/attention_residual", 2, "small_decode", 1.5, 0.5),)

    monkeypatch.setattr(surrogate_cannbench, "_resolve_agent_gates_bin", lambda _r: "agent-gates")
    monkeypatch.setattr(surrogate_cannbench, "_agent_gates_workspace_name", lambda _r: "ws")
    monkeypatch.setattr(
        surrogate_cannbench,
        "prewarm_anchor_reference_cache",
        lambda *_a, **_k: tmp_path / "ref",
    )

    submitted_cands: list[list[str]] = []

    def _fake_submit(
        *,
        request_id: str,
        request_path: Path,
        response_path: Path,
        **_kw: object,
    ) -> tuple[dict[str, object], surrogate_cannbench._V2GateExecution]:
        req_payload = json.loads(request_path.read_text(encoding="utf-8"))
        batch = req_payload["batches"][0]
        submitted_cands.append([c["candidate_id"] for c in batch["candidates"]])
        bindings = {
            "attention_residual": {
                c["candidate_id"]: {
                    "module_path": c["module_path"],
                    "sha256": c["sha256"],
                    "post_sha256": c["sha256"],
                }
                for c in batch["candidates"]
            }
        }
        ops = {
            c["candidate_id"]: {
                "candidate_id": c["candidate_id"],
                "operator_name": "attention_residual",
                "all_passed": True,
                "avg_speedup": 2.0,
                "min_case_speedup": 2.0,
                "avg_roofline_eta": 0.8,
                "surrogate_score": 80.0,
                "anchors": [
                    {
                        "case_num": 2,
                        "regime_tag": "small_decode",
                        "accuracy_passed": True,
                        "cand_us": 0.75,
                        "ref_perf_us": 1.5,
                        "t_hw_us": 0.5,
                        "speedup": 2.0,
                        "roofline_eta": 0.66,
                        "eq3_score": 80.0,
                    }
                ],
            }
            for c in batch["candidates"]
        }
        resp = {
            "schema": surrogate_cannbench.FAST_RESPONSE_SCHEMA,
            "request_id": request_id,
            "mode": "batch",
            "status": "ok",
            "candidates": bindings,
            "coverage": {
                "expected_candidates": len(batch["candidates"]),
                "evaluated_candidates": len(batch["candidates"]),
                "expected_msprof_targets": 0,
                "observed_msprof_targets": 0,
            },
            "operators": {"attention_residual": ops},
        }
        return resp, surrogate_cannbench._V2GateExecution(
            execution_id="exec-1",
            run_id="run-1",
            receipt_path=str(response_path),
            report={"status": "PASS", "gates": []},
            verification={"satisfied": True},
        )

    monkeypatch.setattr(surrogate_cannbench, "_submit_fast_gate", _fake_submit)

    by_op, run_id, _rcp, _held = (
        surrogate_cannbench.run_multi_operator_surrogate_batch_via_agent_gates(
            [
                (
                    "attention_residual",
                    [("incumbent_slot", inc_cand), ("c_good", good_cand), ("c_bad", bad_cand)],
                    anchors,
                )
            ],
            source_root=fake_root,
            env_file=env_file,
        )
    )
    assert run_id == "run-1"
    assert submitted_cands == [["incumbent_slot", "c_good"]]
    op_res = by_op["attention_residual"]
    assert isinstance(op_res, surrogate_cannbench.L2BatchResults)
    assert op_res["incumbent_slot"].all_passed is True
    assert op_res["c_good"].all_passed is True
    assert op_res["c_bad"].all_passed is False
    assert "prewarm_compile_failed" in op_res["c_bad"].error_message
    assert op_res.prewarm_stats["failed_candidates"] == 1
    assert op_res.prewarm_stats["compiled_tasks"] == 2
    assert op_res.l2_prewarm_sec > 0.0

    submitted_cands.clear()
    by_op_all_bad, run_id_empty, _rcp2, held_empty = (
        surrogate_cannbench.run_multi_operator_surrogate_batch_via_agent_gates(
            [
                (
                    "attention_residual",
                    [("incumbent_slot", inc_cand), ("c_bad", bad_cand)],
                    anchors,
                )
            ],
            source_root=fake_root,
            env_file=env_file,
        )
    )
    assert submitted_cands == []
    assert run_id_empty == ""
    assert held_empty == 0.0
    assert by_op_all_bad["attention_residual"]["c_bad"].all_passed is False

    bad_inc = _write_ar_candidate(tmp_path / "bad_inc.py", tag="bad_inc", fail_compile=True)
    with pytest.raises(RuntimeError, match="incumbent_slot"):
        surrogate_cannbench.run_multi_operator_surrogate_batch_via_agent_gates(
            [
                (
                    "attention_residual",
                    [("incumbent_slot", bad_inc), ("c_good", good_cand)],
                    anchors,
                )
            ],
            source_root=fake_root,
            env_file=env_file,
        )


def test_run_l1_compile_check_dual_shape_single_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """run_l1_compile_check lowers both primary and distinct tail_anchor_dims in one subprocess."""
    tl_stub = tmp_path / "tl_stub"
    tl_stub.mkdir()
    (tl_stub / "tilelang.py").write_text(
        "class _Art:\n"
        "    def __init__(self, src):\n"
        "        self.kernel_source = src\n"
        "        self.device_mod = src\n"
        "def lower(fn, target='ascend', enable_device_compile=True):\n"
        "    if isinstance(fn, dict) and fn.get('m', 0) > 100:\n"
        "        raise RuntimeError('tail_ubuf_overflow')\n"
        "    return _Art('// ok')\n",
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
    tl_pkg_backend = tl_stub / "tilelang" / "backend"
    tl_pkg_backend.mkdir(parents=True)
    (tl_stub / "tilelang" / "__init__.py").write_text(
        "from tilelang import *  # stub\n", encoding="utf-8"
    )
    (tl_stub / "tilelang.py").unlink()
    (tl_stub / "tilelang" / "__init__.py").write_text(
        "import tvm\n"
        "class _Art:\n"
        "    def __init__(self, src):\n"
        "        self.kernel_source = src\n"
        "        self.device_mod = src\n"
        "def lower(fn, target='ascend', enable_device_compile=True):\n"
        "    if isinstance(fn, dict) and fn.get('m', 0) > 100:\n"
        "        raise RuntimeError('tail_ubuf_overflow')\n"
        "    return _Art('// ok')\n",
        encoding="utf-8",
    )
    (tl_pkg_backend / "__init__.py").write_text("", encoding="utf-8")
    (tl_pkg_backend / "module.py").write_text(
        "from types import SimpleNamespace\n"
        "import contextlib\n"
        "def create_backend_context(_t):\n"
        "    return SimpleNamespace(target=contextlib.nullcontext())\n",
        encoding="utf-8",
    )

    env_file = tmp_path / "env.sh"
    env_file.write_text(
        f"TILEFOUNDRY_NPU_TILELANG_SOURCE={tl_stub}\n"
        f"TILEFOUNDRY_NPU_PYTHON={sys.executable}\n",
        encoding="utf-8",
    )
    src = (
        "def build_op(m=32):\n"
        "    return {'m': int(m)}\n"
    )

    calls = 0
    real_run = subprocess.run

    def _counting_run(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_run(*args, **kwargs)

    monkeypatch.setattr(surrogate_cannbench.subprocess, "run", _counting_run)

    pass_res = surrogate_cannbench.run_l1_compile_check(
        src,
        builder_name="build_op",
        builder_kwargs={},
        dims={"m": 32},
        tail_anchor_dims={"m": 64},
        env_file=env_file,
    )
    assert pass_res.passed is True
    assert calls == 1

    calls = 0
    fail_res = surrogate_cannbench.run_l1_compile_check(
        src,
        builder_name="build_op",
        builder_kwargs={},
        dims={"m": 32},
        tail_anchor_dims={"m": 256},
        env_file=env_file,
    )
    assert fail_res.passed is False
    assert calls == 1
    assert "tail_ubuf_overflow" in fail_res.full_diagnostic


def test_marathon_defaults_real_npu_flag_no_occupancy_fallback_and_paired_incumbent(
    tmp_path: Path,
) -> None:
    """Verify cost_model_mode default off, is_real_npu, occupancy raise, and paired fallback."""
    assert marathon.MarathonConfig(operator_name="mla").cost_model_mode == "off"
    parsed_default = marathon.build_marathon_arg_parser().parse_args(["run", "--operator", "mla"])
    assert parsed_default.cost_model_mode == "off"
    parsed_shadow = marathon.build_marathon_arg_parser().parse_args(
        ["run", "--operator", "mla", "--cost-model-mode", "shadow"]
    )
    assert parsed_shadow.cost_model_mode == "shadow"

    target = tmp_path / "k.py"
    target.write_text("def k():\n    return 1\n", encoding="utf-8")
    anchors = (AnchorCaseSpec("level4/mla", 3, "decode", 10.0, 5.0),)
    inc_res = surrogate_cannbench.CandidateSurrogateResult(
        candidate_id="incumbent_slot",
        operator_name="mla",
        all_passed=True,
        avg_speedup=1.0,
        min_case_speedup=1.0,
        avg_roofline_eta=0.5,
        surrogate_score=50.0,
    )

    for err_msg in (
        "AgentGates BLOCKED_EXIT_CODE: 75 occupancy",
        "BLOCKED_ENV on npu:0",
        "device_rejected by gate",
    ):
        calls: list[list[str]] = []

        def _occ_fail(_op, cands, _anc, **_kw):
            calls.append([cid for cid, _ in cands])
            raise RuntimeError(err_msg)

        ev = marathon.build_resilient_l2_evaluator(
            operator_name="mla",
            worktree=tmp_path,
            env_file=tmp_path / "env.sh",
            device_id="0",
            device_pool=("0",),
            batch_fn=_occ_fail,
        )
        assert getattr(ev, "is_real_npu", False) is True
        with pytest.raises(RuntimeError):
            ev([("incumbent_slot", target), ("c1", target), ("c2", target)], anchors)
        assert len(calls) == 1

    fallback_calls: list[list[str]] = []

    def _paired_fallback(_op, cands, _anc, **_kw):
        cids = [cid for cid, _ in cands]
        fallback_calls.append(cids)
        if len(cids) > 2:
            raise RuntimeError("transient worker timeout")
        res_map = {"incumbent_slot": inc_res}
        for cid in cids:
            if cid != "incumbent_slot":
                res_map[cid] = surrogate_cannbench.CandidateSurrogateResult(
                    candidate_id=cid,
                    operator_name="mla",
                    all_passed=True,
                    avg_speedup=2.0,
                    min_case_speedup=2.0,
                    avg_roofline_eta=0.8,
                    surrogate_score=80.0,
                )
        return (
            surrogate_cannbench.L2BatchResults(
                res_map,
                {k: f"rid_{cids[-1]}" for k in res_map},
                prewarm_stats={"total_tasks": 1, "cache_hits": 1, "compiled_tasks": 0, "failed_candidates": 0},
                l2_prewarm_sec=0.02,
                l2_npu_sec=1.5,
            ),
            f"rid_{cids[-1]}",
            "rcp",
        )

    ev_fb = marathon.build_resilient_l2_evaluator(
        operator_name="mla",
        worktree=tmp_path,
        env_file=tmp_path / "env.sh",
        device_id="0",
        device_pool=("0",),
        batch_fn=_paired_fallback,
    )
    with pytest.raises(RuntimeError, match="transient worker timeout"):
        ev_fb([("incumbent_slot", target), ("c1", target), ("c2", target)], anchors)
    assert fallback_calls == [["incumbent_slot", "c1", "c2"]]


def test_single_card_lock_overrides_env_multi_card_pool(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI --device-id 0 --device-ids 0 locks pool to ('0',) even when env.sh declares 0,1."""
    primary, pool = resolve_campaign_device_plan(
        device_id="0",
        device_ids=("0",),
        env_device_ids=("0", "1"),
    )
    assert primary == "0"
    assert pool == ("0",)
    assert surrogate_cannbench._normalize_device_pool(primary, pool) == ("0",)

    env_path = tmp_path / "env.sh"
    env_path.write_text(
        "TILEFOUNDRY_NPU_LLM_BACKEND=mock\n"
        "TILEFOUNDRY_NPU_CAMPAIGN_DEVICE_IDS=0,1\n",
        encoding="utf-8",
    )
    seen_pools: list[tuple[str, tuple[str, ...]]] = []

    def _fake_build_l2(**kwargs: object):
        seen_pools.append((str(kwargs["device_id"]), tuple(kwargs["device_pool"])))
        return lambda _c, _a: surrogate_cannbench.L2BatchResults({}, {})

    class _Runner:
        def __init__(self, **kwargs: object) -> None:
            cfg = kwargs["config"]
            seen_pools.append((str(cfg.device_id), tuple(cfg.device_ids)))

        async def run(self, *, resume: bool):
            _ = resume
            return SimpleNamespace(to_dict=lambda: {"status": "ok"})

    monkeypatch.setattr(marathon, "ExtremeCampaignRunner", _Runner)
    monkeypatch.setattr(marathon, "_ensure_sys_paths", lambda _w: None)
    monkeypatch.setattr(marathon, "build_resilient_l2_evaluator", _fake_build_l2)
    monkeypatch.setattr(marathon, "salvage_unpromoted_beam_winner", lambda **_k: None)
    target = tmp_path / "kernels.py"
    target.write_text("def k():\n    return 1\n", encoding="utf-8")
    assert (
        marathon.main(
            [
                "run",
                "--operator",
                "mla",
                "--device-id",
                "0",
                "--device-ids",
                "0",
                "--source-root",
                str(tmp_path),
                "--target-file",
                str(target),
                "--checkpoint",
                str(tmp_path / "checkpoint.json"),
                "--env-file",
                str(env_path),
                "--no-salvage",
            ]
        )
        == 0
    )
    assert seen_pools == [("0", ("0",)), ("0", ())]


def test_poison_pill_subprocess_isolates_crash_and_timeout_without_breaking_paired_burst(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phase-A subprocess screening isolates crashes/timeouts while paired Phase-B stays intact."""
    src_root = tmp_path / "src_root"
    bench_dir = src_root / "tests" / "ascend" / "benchmarks"
    bench_dir.mkdir(parents=True)
    (src_root / "src").symlink_to(Path(__file__).resolve().parents[3] / "src")
    (src_root / "targets").symlink_to(Path(__file__).resolve().parents[3] / "targets")
    (src_root / "tests" / "__init__.py").write_text("", encoding="utf-8")
    (src_root / "tests" / "ascend" / "__init__.py").write_text("", encoding="utf-8")
    (src_root / "tests" / "ascend" / "agent").mkdir(parents=True)
    (src_root / "tests" / "ascend" / "agent" / "__init__.py").write_text("", encoding="utf-8")
    (src_root / "tests" / "ascend" / "agent" / "test_surrogate_batch_npu.py").symlink_to(
        Path(__file__).resolve().parents[3] / "tests" / "ascend" / "agent" / "test_surrogate_batch_npu.py"
    )
    (bench_dir / "__init__.py").write_text("", encoding="utf-8")
    (bench_dir / "cannbench_contract.py").write_text(
        "class _FakeTensor:\n"
        "    def npu(self):\n"
        "        return self\n"
        "    def clone(self):\n"
        "        return self\n"
        "    def numel(self):\n"
        "        return 1024\n"
        "    def element_size(self):\n"
        "        return 2\n"
        "class ReferenceCache:\n"
        "    def __init__(self, _h):\n"
        "        self._d = {}\n"
        "    def __enter__(self):\n"
        "        return self\n"
        "    def __exit__(self, *_a):\n"
        "        return None\n"
        "    def get(self, case, tag, fn):\n"
        "        k = (int(case.case_num), tag)\n"
        "        if k not in self._d:\n"
        "            self._d[k] = fn()\n"
        "        return self._d[k]\n"
        "def map_tensors(obj, fn):\n"
        "    return {k: fn(v) for k, v in obj.items()}\n",
        encoding="utf-8",
    )

    cann_repo = tmp_path / "cann_repo"
    (cann_repo / "tasks").mkdir(parents=True)

    inc_mod = tmp_path / "inc_slot.py"
    inc_mod.write_text("def run_op(x):\n    return 'inc'\n", encoding="utf-8")
    good_mod = tmp_path / "good_cand.py"
    good_mod.write_text("def run_op(x):\n    return 'good'\n", encoding="utf-8")
    crash_mod = tmp_path / "crash_cand.py"
    crash_mod.write_text("import os\ndef run_op(x):\n    os._exit(11)\n", encoding="utf-8")
    hang_mod = tmp_path / "hang_cand.py"
    hang_mod.write_text("import time\ndef run_op(x):\n    time.sleep(10)\n    return 'good'\n", encoding="utf-8")

    class _FakeHarness:
        def prepare(self, _case):
            return {"x": 1}

        def reference_output(self, _case, _ts, native=False):
            _ = native
            return "ref"

        def params(self, _case, tensors):
            return {"x": tensors["x"]}

        def compare(self, _case, actual, _oracle, _tensors, _native_fn):
            return (actual in {"inc", "good"}), {}

    anchors = [AnchorCaseSpec("level4/mla", 3, "decode", 20.0, 6.0)]
    monkeypatch.setattr(
        batch_harness,
        "load_anchor_harness_and_cases",
        lambda _op, anc, **_k: (_FakeHarness(), [SimpleNamespace(case_num=int(a.case_num)) for a in anc]),
    )
    monkeypatch.setattr(
        batch_harness,
        "load_cannbench_contract_module",
        lambda **_k: SimpleNamespace(
            ReferenceCache=type(
                "_RC",
                (),
                {
                    "__init__": lambda self, _h: setattr(self, "_d", {}),
                    "__enter__": lambda self: self,
                    "__exit__": lambda self, *_a: None,
                    "get": lambda self, c, t, fn: self._d.setdefault((int(c.case_num), t), fn()),
                },
            ),
            map_tensors=lambda obj, _fn: dict(obj),
        ),
    )

    class _FakeClockEvent:
        _clock = [0.0]

        def __init__(self, enable_timing: bool = True) -> None:
            _ = enable_timing
            self._t = 0.0

        def record(self) -> None:
            self._t = self._clock[0]

        def elapsed_time(self, other: _FakeClockEvent) -> float:
            return (other._t - self._t) / 1000.0

    fake_npu = SimpleNamespace(
        Event=_FakeClockEvent,
        synchronize=lambda: None,
        empty_cache=lambda: None,
        is_available=lambda: False,
        set_device=lambda _idx: None,
    )
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)

    site_dir = tmp_path / "site_patch"
    site_dir.mkdir()
    (site_dir / "sitecustomize.py").write_text(
        "from types import SimpleNamespace\n"
        "from tilefoundry_ascend.agent.campaign import anchor_specs\n"
        "class _T:\n"
        "    def npu(self): return self\n"
        "class _H:\n"
        "    def prepare(self, _c): return {'x': _T()}\n"
        "    def reference_output(self, _c, _t, native=False): return 'ref'\n"
        "    def params(self, _c, ts): return {'x': ts['x']}\n"
        "    def compare(self, _c, actual, _o, _t, _n): return (actual in {'inc', 'good'}), {}\n"
        "anchor_specs.load_anchor_harness_and_cases = lambda _op, anc, **_k: (_H(), [SimpleNamespace(case_num=int(a.case_num)) for a in anc])\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTHONPATH", f"{site_dir}:{src_root}:{src_root / 'src'}:{src_root / 'targets' / 'ascend'}")

    pass_ok, pass_err, pass_cases = batch_harness._screen_candidate_isolated(
        operator_name="mla",
        entrypoint="run_op",
        candidate_id="c_good",
        module_path=good_mod,
        anchors=anchors,
        source_root=src_root,
        cannbench_repo=cann_repo,
        timeout_sec=5.0,
    )
    assert pass_ok is True, pass_err
    assert pass_cases[3][0] is True

    crash_ok, crash_err, _ = batch_harness._screen_candidate_isolated(
        operator_name="mla",
        entrypoint="run_op",
        candidate_id="c_crash",
        module_path=crash_mod,
        anchors=anchors,
        source_root=src_root,
        cannbench_repo=cann_repo,
        timeout_sec=5.0,
    )
    assert crash_ok is False
    assert "poison_pill_timeout_or_crash" in crash_err

    hang_ok, hang_err, _ = batch_harness._screen_candidate_isolated(
        operator_name="mla",
        entrypoint="run_op",
        candidate_id="c_hang",
        module_path=hang_mod,
        anchors=anchors,
        source_root=src_root,
        cannbench_repo=cann_repo,
        timeout_sec=1.5,
    )
    assert hang_ok is False
    assert "poison_pill_timeout_or_crash" in hang_err
    assert "timeout" in hang_err

    paired_calls: list[list[str]] = []

    def _fake_paired(named_fns, _params, **_kw):
        cids = [cid for cid, _fn in named_fns]
        paired_calls.append(cids)
        return {
            "incumbent_slot": (20.0, 1, 0.01, 1.0, 0.0),
            "c_good": (10.0, 1, 0.01, 2.0, 0.01),
        }

    monkeypatch.setattr(batch_harness, "measure_paired_interleaved_micro_burst", _fake_paired)

    batch_res = batch_harness._evaluate_operator_batch(
        {
            "operator_name": "mla",
            "entrypoint": "run_op",
            "candidates": [
                {"candidate_id": "incumbent_slot", "module_path": str(inc_mod)},
                {"candidate_id": "c_good", "module_path": str(good_mod)},
                {"candidate_id": "c_crash", "module_path": str(crash_mod)},
                {"candidate_id": "c_hang", "module_path": str(hang_mod)},
            ],
            "anchors": [anchors[0].to_dict()],
        },
        source_root=src_root,
        cannbench_repo=cann_repo,
        warmup=1,
        repeat=3,
        poison_pill_isolation=True,
        poison_pill_timeout_sec=2.0,
    )
    assert paired_calls == [["incumbent_slot", "c_good"]]
    assert batch_res["incumbent_slot"].all_passed is True
    assert batch_res["c_good"].all_passed is True
    assert batch_res["c_good"].anchors[0].paired_speedup_vs_incumbent == pytest.approx(2.0)
    assert batch_res["c_crash"].all_passed is False
    assert "poison_pill_timeout_or_crash" in batch_res["c_crash"].anchors[0].error_message
    assert batch_res["c_hang"].all_passed is False
    assert "poison_pill_timeout_or_crash" in batch_res["c_hang"].anchors[0].error_message


def test_prewarm_anchor_reference_cache_marker_skips_and_invalidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Skip reference prewarm subprocess when marker and .pt files match; re-run on loss."""
    fake_root, cann_repo, _tl_stub, work_root, env_file = _setup_prewarm_fixture(tmp_path)
    anchors = (
        AnchorCaseSpec("level4/attention_residual", 2, "small_decode", 1.5, 0.5),
        AnchorCaseSpec("level4/attention_residual", 4, "unaligned_d", 9.9, 3.3),
    )
    ref_cache = work_root / "surrogate" / "reference"
    op_dir = ref_cache / "operators" / "level4__attention_residual"
    marker_path = op_dir / ".ref_prewarm_v1.json"

    spawn_count = 0

    def _fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal spawn_count
        spawn_count += 1
        op_dir.mkdir(parents=True, exist_ok=True)
        (op_dir / "identity.json").write_text(
            '{"task": "level4/attention_residual"}',
            encoding="utf-8",
        )
        for cid in (2, 4):
            (op_dir / f"case_{cid}.oracle.pt").write_bytes(b"ORACLE_PT")
            (op_dir / f"case_{cid}.native.pt").write_bytes(b"NATIVE_PT")
        return subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")

    monkeypatch.setattr(surrogate_cannbench.subprocess, "run", _fake_run)

    out1 = surrogate_cannbench.prewarm_anchor_reference_cache(
        "attention_residual",
        anchors,
        cannbench_repo=cann_repo,
        env_file=env_file,
        cache_dir=ref_cache,
        source_root=fake_root,
    )
    assert out1 == ref_cache
    assert spawn_count == 1
    assert marker_path.is_file()
    marker_data = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker_data["case_ids"] == [2, 4]
    assert marker_data["files"] == [
        "case_2.native.pt",
        "case_2.oracle.pt",
        "case_4.native.pt",
        "case_4.oracle.pt",
    ]

    surrogate_cannbench.prewarm_anchor_reference_cache(
        "attention_residual",
        anchors,
        cannbench_repo=cann_repo,
        env_file=env_file,
        cache_dir=ref_cache,
        source_root=fake_root,
    )
    assert spawn_count == 1

    (op_dir / "case_4.native.pt").unlink()
    surrogate_cannbench.prewarm_anchor_reference_cache(
        "attention_residual",
        anchors,
        cannbench_repo=cann_repo,
        env_file=env_file,
        cache_dir=ref_cache,
        source_root=fake_root,
    )
    assert spawn_count == 2
    assert (op_dir / "case_4.native.pt").is_file()

    surrogate_cannbench.prewarm_anchor_reference_cache(
        "attention_residual",
        anchors[:1],
        cannbench_repo=cann_repo,
        env_file=env_file,
        cache_dir=ref_cache,
        source_root=fake_root,
    )
    assert spawn_count == 2

    cases_path = cann_repo / "tasks" / "level4" / "attention_residual" / "cases.yaml"
    cases_path.write_text(
        cases_path.read_text(encoding="utf-8") + "\n# changed\n",
        encoding="utf-8",
    )
    surrogate_cannbench.prewarm_anchor_reference_cache(
        "attention_residual",
        anchors,
        cannbench_repo=cann_repo,
        env_file=env_file,
        cache_dir=ref_cache,
        source_root=fake_root,
    )
    assert spawn_count == 3
    updated_marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert updated_marker["cases_yaml_sha256"] != marker_data["cases_yaml_sha256"]

