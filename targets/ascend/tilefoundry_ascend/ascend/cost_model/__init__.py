"""Ascend 950 bisheng resource parser and 3-tier hybrid cost model."""

from tilefoundry_ascend.ascend.cost_model.bisheng_res_parser import (
    DEFAULT_COMPILER_BEHAVIOR_REGISTRY,
    BishengResourceSummary,
    CompilerBehaviorDetector,
    CompilerBehaviorRegistry,
    parse_bisheng_res_usage,
    parse_bisheng_resources,
    register_compiler_detector,
)
from tilefoundry_ascend.ascend.cost_model.hybrid_cost_model import (
    CUBE_MACS_PER_CYCLE,
    DEFAULT_CV_SYNC_EVENT_US,
    DEFAULT_FIXPIPE_BYTES_PER_CYCLE_PER_AIC,
    DEFAULT_LAUNCH_OVERHEAD_US,
    CalibrationSample,
    CannbenchScorePrediction,
    CostFeatures,
    CostModelEvalReport,
    CostPrediction,
    HybridAscendCostModel,
    build_cost_features_from_ledger,
    evaluate_cost_model_ranking,
    predict_cannbench_eq3_score,
    topology_bucket_for_axes,
)

_BACKTEST_EXPORTS = frozenset(
    {
        "BacktestSummary",
        "OfficialCaseRecord",
        "load_official_reports",
        "run_cost_model_backtest",
        "run_p2_prescreen_ablation",
    }
)


def __getattr__(name: str) -> object:
    if name in _BACKTEST_EXPORTS:
        from tilefoundry_ascend.ascend.cost_model import calibration_backtest

        return getattr(calibration_backtest, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "CUBE_MACS_PER_CYCLE",
    "DEFAULT_COMPILER_BEHAVIOR_REGISTRY",
    "DEFAULT_CV_SYNC_EVENT_US",
    "DEFAULT_FIXPIPE_BYTES_PER_CYCLE_PER_AIC",
    "DEFAULT_LAUNCH_OVERHEAD_US",
    "BacktestSummary",
    "BishengResourceSummary",
    "CalibrationSample",
    "CannbenchScorePrediction",
    "CompilerBehaviorDetector",
    "CompilerBehaviorRegistry",
    "CostFeatures",
    "CostModelEvalReport",
    "CostPrediction",
    "HybridAscendCostModel",
    "OfficialCaseRecord",
    "build_cost_features_from_ledger",
    "evaluate_cost_model_ranking",
    "load_official_reports",
    "parse_bisheng_res_usage",
    "parse_bisheng_resources",
    "predict_cannbench_eq3_score",
    "register_compiler_detector",
    "run_cost_model_backtest",
    "run_p2_prescreen_ablation",
    "topology_bucket_for_axes",
]
