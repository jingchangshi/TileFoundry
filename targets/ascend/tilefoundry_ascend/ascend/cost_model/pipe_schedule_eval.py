"""Ascend 950 (dav-3510 / A5) continuous 6-pipe + AVE schedule evaluator."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from tilefoundry_ascend.ascend.cost_model.microarch_profile import MicroarchProfile
from tilefoundry_ascend.ascend.target import AscendSpaceFacts, AscendTarget
from tilefoundry_ascend.ascend.target.facts import resolve_cube_macs_safe

MS_PIPE_SLOPE_US_PER_BYTE: dict[str, float] = {
    "MTE1": 5.54e-6, "MTE2": 6.39e-6, "MTE3": 5.65e-6, "FIXPIPE": 5.40e-6,
}
MS_PIPE_INTERCEPT_US: dict[str, float] = {
    "CUBE": 0.081, "VECTOR": 0.026, "MTE1": 0.021,
    "MTE2": 0.276, "MTE3": 0.150, "FIXPIPE": 0.137,
}
DIRECTION_SLOPE_MULT: dict[str, float] = {"ub_to_l1": 0.50, "l0c_to_ub": 0.877}


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
        """Serialize rounded schedule latencies to a JSON-safe dictionary."""
        out: dict[str, object] = {
            "pipe_ns": {k: round(v, 3) for k, v in sorted(self.pipe_ns.items())},
            "dominant_pipe": self.dominant_pipe,
        }
        for key in ("ave_compute_ns", "ave_io_ns", "steady_state_ii_ns", "drain_bubble_ns", "loop_total_ns", "total_ns"):
            out[key] = round(float(getattr(self, key)), 3)
        return out


def normalize_ns_to_z3_ticks(ns_values: Sequence[float], *, max_ticks: int = 12) -> list[int]:
    """Map continuous ns latencies cleanly into integer Z3 ticks in [1, max_ticks]."""
    if not ns_values:
        return []
    cap = max(2, int(max_ticks))
    positive = [max(1.0, float(v)) for v in ns_values]
    max_v = max(positive)
    return [1 for _ in positive] if max_v <= 0.0 else [min(cap, max(1, int(round((v / max_v) * cap)))) for v in positive]


def _dma_pipe_ns(pipe: str, eff_bytes: float, slope_us_per_b: float, scale: float) -> float:
    return (eff_bytes * slope_us_per_b + MS_PIPE_INTERCEPT_US[pipe]) * 1000.0 * scale if eff_bytes > 0.0 else 0.0


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
        _scale = lambda p: max(0.5, min(2.0, float(scales.get(p, 1.0) or 1.0)))  # noqa: E731
        aic_cores = max(1, int(active_aic_cores or self.facts.num_aic_cores))
        aiv_cores = max(1, int(active_aiv_cores or self.facts.num_aiv_cores))
        burst_eff = max(0.10, min(1.0, float(dma_burst_efficiency or 1.0)))
        active_grid = aic_cores if (cube_flops > 0.0 or prof.cube_mmad_count > 0 or prof.cube_macs_total > 0) else aiv_cores
        if profile_is_per_core_full_loop:
            waves, block_scale = 1.0, 1.0 / float(max(1, total_tiles if total_tiles > 0 else active_grid))
        elif total_tiles > 0:
            waves, block_scale = float(max(1, math.ceil(total_tiles / max(1, active_grid)))), 1.0 / float(max(1, total_tiles))
        else:
            waves, block_scale = 1.0, 1.0 / float(active_grid)

        macs_per_cyc = float(resolve_cube_macs_safe(self.facts, cube_dtype, hf32=hf32))
        per_block_cube = max(float(prof.cube_macs_total) * 2.0, float(cube_flops) * block_scale)
        if per_block_cube > 0.0:
            cube_compute = per_block_cube / max(1e-6, macs_per_cyc * 2.0 * self.clock_ghz)
            cube_ns = (cube_compute + MS_PIPE_INTERCEPT_US["CUBE"] * 1000.0 * max(1, prof.cube_mmad_count)) * prof.l1_bank_conflict_factor * _scale("CUBE")
        elif prof.cube_mmad_count > 0:
            cube_ns = MS_PIPE_INTERCEPT_US["CUBE"] * 1000.0 * prof.cube_mmad_count * prof.l1_bank_conflict_factor * _scale("CUBE")
        else:
            cube_ns = 0.0

        ave_issues = 0.5 * prof.fp_alu_ops + prof.int_alu_ops + 4.0 * prof.sfu_f16_ops + 2.0 * prof.sfu_f32_ops + 3.0 * prof.reduce_ops + prof.slot0_only_ops
        clk = max(0.1, self.clock_ghz)
        vec_flop_ns = (float(vec_flops) * block_scale) / max(1e-6, 128.0 * self.clock_ghz)
        ave_compute_ns = max(vec_flop_ns, (ave_issues * 8.0) / clk, float(prof.scalar_ops) / clk) * (0.5 if is_coupled_1c2v else 1.0)
        ub_load_ns, ub_store_ns = (float(prof.ub_load_bytes) / 512.0) / clk, (float(prof.ub_store_bytes) / 256.0) / clk
        ave_io_ns = (ub_store_ns if ub_store_ns >= ub_load_ns else ub_store_ns + 0.5 * (ub_load_ns - ub_store_ns)) * max(1.0, prof.ub_bank_conflict_factor) * max(1.0, prof.simt_shm_conflict_factor)
        vector_ns = ((max(ave_compute_ns, ave_io_ns) + MS_PIPE_INTERCEPT_US["VECTOR"] * 1000.0) * _scale("VECTOR")) if (ave_compute_ns > 0.0 or ave_io_ns > 0.0) else 0.0

        fix_bpc = float(getattr(self.facts, "fixpipe_bytes_per_cycle", 128) or 128)
        fix_slope = 1.0 / (max(1.0, fix_bpc) * self.clock_ghz * 1e3)
        pipe_ns = {
            k: v * waves
            for k, v in {
                "CUBE": cube_ns,
                "VECTOR": vector_ns,
                "MTE1": _dma_pipe_ns("MTE1", float(prof.l1_to_l0a_bytes + prof.l1_to_l0b_bytes), MS_PIPE_SLOPE_US_PER_BYTE["MTE1"], _scale("MTE1")),
                "MTE2": _dma_pipe_ns("MTE2", float(prof.gm_to_l1_bytes + prof.gm_to_ub_bytes), MS_PIPE_SLOPE_US_PER_BYTE["MTE2"] / burst_eff, _scale("MTE2")),
                "MTE3": _dma_pipe_ns("MTE3", float(prof.ub_to_gm_bytes) + DIRECTION_SLOPE_MULT["ub_to_l1"] * float(prof.ub_to_l1_bytes), MS_PIPE_SLOPE_US_PER_BYTE["MTE3"], _scale("MTE3")),
                "FIXPIPE": _dma_pipe_ns("FIXPIPE", DIRECTION_SLOPE_MULT["l0c_to_ub"] * float(prof.l0c_to_ub_bytes) + float(prof.l0c_to_l1_bytes + prof.l0c_to_gm_bytes), fix_slope, _scale("FIXPIPE")),
            }.items()
        }
        non_zero = [(k, v) for k, v in pipe_ns.items() if v > 0.0]
        dom_pipe, max_pipe_ns = max(non_zero, key=lambda item: (item[1], item[0])) if non_zero else ("VECTOR", 0.0)
        if ledger_steady_state_ii > 0:
            z3_ii_ns = (float(ledger_steady_state_ii) / clk) * waves
            if z3_ii_ns > max_pipe_ns > 0.0:
                max_pipe_ns = 0.7 * max_pipe_ns + 0.3 * z3_ii_ns

        depth, trips = max(1, prof.pipeline_depth), max(1, prof.inner_loop_trips)
        total_active = sum(pipe_ns.values())
        per_iter_ii_ns, non_dom_tail = max_pipe_ns / float(trips), max(0.0, total_active - max_pipe_ns)
        if depth > 1 and total_active > max_pipe_ns:
            drain_bubble_ns = min(non_dom_tail, float(depth - 1) * per_iter_ii_ns * 0.65)
            overlap_eff = max(0.0, min(1.0, float(trips - 1) / float(trips + depth - 1)))
            loop_total_ns = max_pipe_ns + drain_bubble_ns + 0.20 * non_dom_tail * (1.0 - overlap_eff)
        else:
            drain_bubble_ns = non_dom_tail * 0.65
            loop_total_ns = max_pipe_ns + drain_bubble_ns

        sync_ns = waves * (float(prof.cv_sync_barriers) * 80.0 + float(prof.pipe_all_barriers) * 180.0)
        return Ascend950PipeScheduleResult(
            pipe_ns=pipe_ns, ave_compute_ns=ave_compute_ns * waves, ave_io_ns=ave_io_ns * waves,
            steady_state_ii_ns=per_iter_ii_ns, drain_bubble_ns=drain_bubble_ns,
            loop_total_ns=loop_total_ns, dominant_pipe=dom_pipe, total_ns=loop_total_ns + sync_ns,
        )


__all__ = [
    "DIRECTION_SLOPE_MULT", "MS_PIPE_INTERCEPT_US", "MS_PIPE_SLOPE_US_PER_BYTE",
    "Ascend950PipeScheduleEvaluator", "Ascend950PipeScheduleResult",
    "normalize_ns_to_z3_ticks", "resolve_cube_macs_safe",
]
