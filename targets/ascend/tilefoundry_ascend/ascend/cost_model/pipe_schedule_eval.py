"""Ascend 950 (dav-3510 / A5) continuous 6-pipe + AVE schedule evaluator."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from tilefoundry_ascend.ascend.cost_model.microarch_profile import MicroarchProfile
from tilefoundry_ascend.ascend.target import AscendSpaceFacts, AscendTarget

MS_PIPE_SLOPE_US_PER_BYTE: dict[str, float] = {
    "MTE1": 5.54e-6,
    "MTE2": 6.39e-6,
    "MTE3": 5.65e-6,
    "FIXPIPE": 5.40e-6,
}

MS_PIPE_INTERCEPT_US: dict[str, float] = {
    "CUBE": 0.081,
    "VECTOR": 0.026,
    "MTE1": 0.021,
    "MTE2": 0.276,
    "MTE3": 0.150,
    "FIXPIPE": 0.137,
}

DIRECTION_SLOPE_MULT: dict[str, float] = {
    "ub_to_l1": 0.50,
    "l0c_to_ub": 0.877,
}

_CUBE_DTYPE_ALIASES: dict[str, str] = {
    "float16": "f16",
    "fp16": "f16",
    "bfloat16": "bf16",
    "float32": "f32",
    "fp32": "f32",
    "int8": "i8",
    "uint8": "u8",
    "e4m3": "fp8e4m3",
    "fp8_e4m3": "fp8e4m3",
    "float8_e4m3": "fp8e4m3",
    "float8_e4m3fn": "fp8e4m3",
    "e5m2": "fp8e5m2",
    "fp8_e5m2": "fp8e5m2",
    "float8_e5m2": "fp8e5m2",
    "float4_e2m1": "f4e2m1",
    "fp4_e2m1": "f4e2m1",
    "float4_e1m2": "f4e1m2",
    "fp4_e1m2": "f4e1m2",
}


def resolve_cube_macs_safe(
    facts: AscendSpaceFacts,
    dtype_name: object,
    *,
    hf32: bool = False,
) -> int:
    """Resolve Cube MACs/cycle safely from canonical or full TIR dtype names."""
    raw = str(getattr(dtype_name, "name", dtype_name) or "").strip().lower()
    if not raw:
        raise ValueError(f"Unsupported empty Cube operand dtype {dtype_name!r}")
    canon = _CUBE_DTYPE_ALIASES.get(raw, raw)
    if hf32 and canon in ("f32", "hf32"):
        canon = "hf32"
    try:
        return int(facts.cube_macs_per_cycle(canon, hf32=hf32))
    except (ValueError, KeyError, TypeError) as exc:
        raise ValueError(
            f"Unsupported Cube operand dtype {dtype_name!r} (canonical={canon!r})"
        ) from exc


@dataclass(frozen=True)
class Ascend950PipeScheduleResult:
    """Continuous nanosecond breakdown across Ascend 950 pipes, AVE ports, and bubbles."""

    pipe_ns: dict[str, float]
    ave_compute_ns: float
    ave_io_ns: float
    steady_state_ii_ns: float
    drain_bubble_ns: float
    loop_total_ns: float
    dominant_pipe: str
    total_ns: float

    def to_dict(self) -> dict[str, object]:
        return {
            "pipe_ns": {k: round(v, 3) for k, v in sorted(self.pipe_ns.items())},
            "ave_compute_ns": round(self.ave_compute_ns, 3),
            "ave_io_ns": round(self.ave_io_ns, 3),
            "steady_state_ii_ns": round(self.steady_state_ii_ns, 3),
            "drain_bubble_ns": round(self.drain_bubble_ns, 3),
            "loop_total_ns": round(self.loop_total_ns, 3),
            "dominant_pipe": self.dominant_pipe,
            "total_ns": round(self.total_ns, 3),
        }


def normalize_ns_to_z3_ticks(
    ns_values: Sequence[float],
    *,
    max_ticks: int = 12,
) -> list[int]:
    """Map continuous ns latencies cleanly into integer Z3 ticks in [1, max_ticks]."""
    if not ns_values:
        return []
    cap = max(2, int(max_ticks))
    positive = [max(1.0, float(v)) for v in ns_values]
    max_v = max(positive)
    if max_v <= 0.0:
        return [1 for _ in positive]
    return [min(cap, max(1, int(round((val / max_v) * cap)))) for val in positive]


class Ascend950PipeScheduleEvaluator:
    """Evaluator fusing AVE Vector ports/SFU physics with 6-pipe affine overlap."""

    def __init__(self, target: AscendTarget | None = None) -> None:
        self.target = target or AscendTarget("ascend.950pr")
        self.facts = self.target.get_facts(AscendSpaceFacts)
        self.clock_ghz = float(getattr(self.facts, "clock_ghz", 1.65) or 1.65)

    def evaluate_profile(
        self,
        profile: MicroarchProfile | Mapping[str, object],
        *,
        cube_flops: float = 0.0,
        vec_flops: float = 0.0,
        cube_dtype: str = "f16",
        hf32: bool = False,
        total_tiles: int = 0,
        active_aic_cores: int | None = None,
        active_aiv_cores: int | None = None,
        dma_burst_efficiency: float = 1.0,
        is_coupled_1c2v: bool = False,
        ledger_steady_state_ii: int = 0,
        pipe_scale_multipliers: Mapping[str, float] | None = None,
        profile_is_per_core_full_loop: bool = False,
    ) -> Ascend950PipeScheduleResult:
        """Compute single-wave continuous pipe ns, AVE score, steady II, and drain bubble."""
        prof = MicroarchProfile.coerce(profile)
        scales = dict(pipe_scale_multipliers or {})

        def _scale(pipe: str) -> float:
            raw = float(scales.get(pipe, 1.0) or 1.0)
            return max(0.5, min(2.0, raw))

        aic_cores = max(1, int(active_aic_cores or self.facts.num_aic_cores))
        aiv_cores = max(1, int(active_aiv_cores or self.facts.num_aiv_cores))
        burst_eff = max(0.10, min(1.0, float(dma_burst_efficiency or 1.0)))

        is_cube_work = cube_flops > 0.0 or prof.cube_mmad_count > 0 or prof.cube_macs_total > 0
        active_grid = aic_cores if is_cube_work else aiv_cores
        if profile_is_per_core_full_loop:
            waves = 1.0
            block_from_chip_scale = 1.0 / float(max(1, total_tiles if total_tiles > 0 else active_grid))
        elif total_tiles > 0:
            waves = float(max(1, math.ceil(total_tiles / max(1, active_grid))))
            block_from_chip_scale = 1.0 / float(max(1, total_tiles))
        else:
            waves = 1.0
            block_from_chip_scale = 1.0 / float(active_grid)

        cube_macs_per_cyc = float(resolve_cube_macs_safe(self.facts, cube_dtype, hf32=hf32))
        block_macs_flops = float(prof.cube_macs_total) * 2.0
        chip_cube_flops = float(cube_flops) * block_from_chip_scale
        per_block_cube_flops = max(block_macs_flops, chip_cube_flops)
        if per_block_cube_flops > 0.0:
            aic_peak_flops_per_ns = cube_macs_per_cyc * 2.0 * self.clock_ghz
            cube_compute_ns = per_block_cube_flops / max(1e-6, aic_peak_flops_per_ns)
            cube_intercept_ns = MS_PIPE_INTERCEPT_US["CUBE"] * 1000.0 * max(1, prof.cube_mmad_count)
            cube_ns = (cube_compute_ns + cube_intercept_ns) * prof.l1_bank_conflict_factor * _scale("CUBE")
        elif prof.cube_mmad_count > 0:
            cube_ns = (
                (MS_PIPE_INTERCEPT_US["CUBE"] * 1000.0 * prof.cube_mmad_count)
                * prof.l1_bank_conflict_factor
                * _scale("CUBE")
            )
        else:
            cube_ns = 0.0

        ave_weighted_issues = (
            0.5 * float(prof.fp_alu_ops)
            + 1.0 * float(prof.int_alu_ops)
            + 4.0 * float(prof.sfu_f16_ops)
            + 2.0 * float(prof.sfu_f32_ops)
            + 3.0 * float(prof.reduce_ops)
            + 1.0 * float(prof.slot0_only_ops)
        )
        vec_block_flops = float(vec_flops) * block_from_chip_scale
        aiv_peak_flops_per_ns = 128.0 * self.clock_ghz
        vec_flop_ns = vec_block_flops / max(1e-6, aiv_peak_flops_per_ns)
        ave_issue_ns = (ave_weighted_issues * 8.0) / max(0.1, self.clock_ghz)
        scalar_issue_ns = float(prof.scalar_ops) / max(0.1, self.clock_ghz)
        cv_vec_factor = 0.5 if is_coupled_1c2v else 1.0
        ave_compute_ns = max(vec_flop_ns, ave_issue_ns, scalar_issue_ns) * cv_vec_factor

        ub_load_ns = (float(prof.ub_load_bytes) / 512.0) / max(0.1, self.clock_ghz)
        ub_store_ns = (float(prof.ub_store_bytes) / 256.0) / max(0.1, self.clock_ghz)
        if ub_store_ns >= ub_load_ns:
            ave_io_ns = ub_store_ns
        else:
            ave_io_ns = ub_store_ns + 0.5 * (ub_load_ns - ub_store_ns)
        ave_io_ns *= max(1.0, prof.ub_bank_conflict_factor) * max(1.0, prof.simt_shm_conflict_factor)

        if ave_compute_ns > 0.0 or ave_io_ns > 0.0:
            vec_intercept_ns = MS_PIPE_INTERCEPT_US["VECTOR"] * 1000.0
            vector_ns = (max(ave_compute_ns, ave_io_ns) + vec_intercept_ns) * _scale("VECTOR")
        else:
            vector_ns = 0.0

        mte1_bytes = float(prof.l1_to_l0a_bytes + prof.l1_to_l0b_bytes)
        mte1_ns = (
            (mte1_bytes * MS_PIPE_SLOPE_US_PER_BYTE["MTE1"] + MS_PIPE_INTERCEPT_US["MTE1"]) * 1000.0 * _scale("MTE1")
            if mte1_bytes > 0.0
            else 0.0
        )

        mte2_bytes = float(prof.gm_to_l1_bytes + prof.gm_to_ub_bytes)
        mte2_ns = (
            ((mte2_bytes * MS_PIPE_SLOPE_US_PER_BYTE["MTE2"] / burst_eff) + MS_PIPE_INTERCEPT_US["MTE2"])
            * 1000.0
            * _scale("MTE2")
            if mte2_bytes > 0.0
            else 0.0
        )

        mte3_eff_bytes = float(prof.ub_to_gm_bytes) + DIRECTION_SLOPE_MULT["ub_to_l1"] * float(prof.ub_to_l1_bytes)
        mte3_ns = (
            (mte3_eff_bytes * MS_PIPE_SLOPE_US_PER_BYTE["MTE3"] + MS_PIPE_INTERCEPT_US["MTE3"])
            * 1000.0
            * _scale("MTE3")
            if mte3_eff_bytes > 0.0
            else 0.0
        )

        fixpipe_bpc = float(getattr(self.facts, "fixpipe_bytes_per_cycle", 128) or 128)
        fixpipe_slope_us_per_b = 1.0 / (max(1.0, fixpipe_bpc) * self.clock_ghz * 1e3)
        fix_eff_bytes = (
            DIRECTION_SLOPE_MULT["l0c_to_ub"] * float(prof.l0c_to_ub_bytes)
            + float(prof.l0c_to_l1_bytes)
            + float(prof.l0c_to_gm_bytes)
        )
        fixpipe_ns = (
            (fix_eff_bytes * fixpipe_slope_us_per_b + MS_PIPE_INTERCEPT_US["FIXPIPE"])
            * 1000.0
            * _scale("FIXPIPE")
            if fix_eff_bytes > 0.0
            else 0.0
        )

        pipe_ns: dict[str, float] = {
            "CUBE": cube_ns * waves,
            "VECTOR": vector_ns * waves,
            "MTE1": mte1_ns * waves,
            "MTE2": mte2_ns * waves,
            "MTE3": mte3_ns * waves,
            "FIXPIPE": fixpipe_ns * waves,
        }
        non_zero = [(k, v) for k, v in pipe_ns.items() if v > 0.0]
        if non_zero:
            dom_pipe, max_pipe_block_ns = max(non_zero, key=lambda item: (item[1], item[0]))
        else:
            dom_pipe, max_pipe_block_ns = "VECTOR", 0.0

        if ledger_steady_state_ii > 0:
            z3_ii_ns = (float(ledger_steady_state_ii) / max(0.1, self.clock_ghz)) * waves
            if z3_ii_ns > max_pipe_block_ns and max_pipe_block_ns > 0.0:
                max_pipe_block_ns = 0.7 * max_pipe_block_ns + 0.3 * z3_ii_ns

        depth = max(1, prof.pipeline_depth)
        trips = max(1, prof.inner_loop_trips)
        total_active_sum = sum(pipe_ns.values())
        per_iter_ii_ns = max_pipe_block_ns / float(trips)
        non_dom_tail_ns = max(0.0, total_active_sum - max_pipe_block_ns)
        if depth > 1 and total_active_sum > max_pipe_block_ns:
            drain_bubble_ns = min(non_dom_tail_ns, float(depth - 1) * per_iter_ii_ns * 0.65)
            overlap_efficiency = max(0.0, min(1.0, float(trips - 1) / float(trips + depth - 1)))
            residual_non_dom = non_dom_tail_ns * (1.0 - overlap_efficiency)
            loop_total_ns = max_pipe_block_ns + drain_bubble_ns + 0.20 * residual_non_dom
        else:
            drain_bubble_ns = non_dom_tail_ns * 0.65
            loop_total_ns = max_pipe_block_ns + drain_bubble_ns

        sync_ns = waves * (float(prof.cv_sync_barriers) * 80.0 + float(prof.pipe_all_barriers) * 180.0)
        total_ns = loop_total_ns + sync_ns
        return Ascend950PipeScheduleResult(
            pipe_ns=pipe_ns,
            ave_compute_ns=ave_compute_ns * waves,
            ave_io_ns=ave_io_ns * waves,
            steady_state_ii_ns=per_iter_ii_ns,
            drain_bubble_ns=drain_bubble_ns,
            loop_total_ns=loop_total_ns,
            dominant_pipe=dom_pipe,
            total_ns=total_ns,
        )


__all__ = [
    "DIRECTION_SLOPE_MULT",
    "MS_PIPE_INTERCEPT_US",
    "MS_PIPE_SLOPE_US_PER_BYTE",
    "Ascend950PipeScheduleEvaluator",
    "Ascend950PipeScheduleResult",
    "normalize_ns_to_z3_ticks",
    "resolve_cube_macs_safe",
]
