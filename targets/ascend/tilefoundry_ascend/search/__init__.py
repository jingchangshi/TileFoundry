"""P7-01/G69: bounded, target-aware performance search for the two single-kernel operators.

THE PACKAGE IN ONE PARAGRAPH. The search enumerates the discrete schedule decisions the specs make
searchable (generator), refuses the configurations that cannot run and says exactly why (filter),
and then compiles, verifies, checks the launch contract and times what survives, with a cache
identity that invalidates and statistics over retained raw samples (harness). Nothing here ranks
by a cost model, and nothing here counts a feasible configuration as a measured one.

THE SPLIT, and why each part is a separate module:

  generator.py  the candidate space and a stable identity per candidate. It only says which
                configurations EXIST; it never says which is fast.
  filter.py     static legality plus UB/L1/L0A/L0B/L0C/SS/event-slot capacity, each rejection a
                named code with the arithmetic attached.
  harness.py    the compile -> numeric -> launch -> timing loop, the cache key, the statistics
                (median/p10/p90/spread over raw samples), the tie/noise rule, and the per-case
                search driver including the local neighbourhood expansion.

WHAT THE PACKAGE IS CAREFUL NOT TO CLAIM. A FEASIBLE candidate is not a result. A timing taken
with the injectable simulated clock is not a device measurement and is labelled as such. The
resource figures are the author's declared buffers, so every judgement is marked
lower_bound_only until the real lowering is compiled and inspected. The search reports the
fastest MEASURED candidate within the recorded budget, never a global optimum.
"""
from .generator import (
    AttentionResidualCandidate,
    GeneratorError,
    MLACandidate,
    ORIGINS,
    ORIGIN_ADVANCED,
    ORIGIN_NEIGHBOR,
    ORIGIN_SEED,
    PRECISION_HI_LO,
    PRECISION_POLICIES,
    PRECISION_SINGLE,
    RESCALE_CANONICAL,
    RESCALE_DEFERRED,
    RESCALE_MODES,
    attention_residual_neighbours,
    attention_residual_seeds,
    deduplicate,
    mla_advanced_around,
    mla_neighbours,
    mla_seeds,
    theta_legal_bound,
    theta_search_bound,
)
from .filter import (
    COMPILE_FAILED,
    FEASIBLE,
    INFEASIBLE,
    Judgement,
    MEASURED,
    NUMERIC_FAILED,
    Rejection,
    TargetLimits,
    UNMEASURED,
    attention_residual_resources,
    judge_all,
    judge_attention_residual,
    judge_mla,
    summarise as summarise_filter,
    target_limits,
)
from .harness import (
    Budget,
    CacheIdentity,
    CandidateResult,
    CandidateRun,
    CaseBudget,
    DeviceTimer,
    DeviceTimerUnavailable,
    HarnessError,
    LAUNCH_CONTRACT_VIOLATED,
    MEASUREMENT_SOURCE_DEVICE,
    MEASUREMENT_SOURCE_SIMULATED,
    OK,
    P0_BUDGET,
    PRUNED_STATIC,
    Ranking,
    SearchRound,
    SimulatedClock,
    Stats,
    compute_stats,
    confirmation_required,
    expand_neighbourhood,
    plan_digest,
    pruned_result,
    rank,
    run_candidate,
    search_case,
    source_digest,
    summarise_rounds,
    toolchain_identity,
    top_k,
)

__all__ = [
    "MLACandidate", "AttentionResidualCandidate", "GeneratorError",
    "ORIGINS", "ORIGIN_SEED", "ORIGIN_NEIGHBOR", "ORIGIN_ADVANCED",
    "RESCALE_CANONICAL", "RESCALE_DEFERRED", "RESCALE_MODES",
    "PRECISION_SINGLE", "PRECISION_HI_LO", "PRECISION_POLICIES",
    "mla_seeds", "attention_residual_seeds", "mla_neighbours",
    "attention_residual_neighbours", "mla_advanced_around", "deduplicate",
    "theta_legal_bound", "theta_search_bound",
    "FEASIBLE", "INFEASIBLE", "MEASURED", "UNMEASURED", "NUMERIC_FAILED", "COMPILE_FAILED",
    "Judgement", "Rejection", "TargetLimits", "target_limits", "judge_all", "judge_mla",
    "judge_attention_residual", "attention_residual_resources", "summarise_filter",
    "Budget", "CaseBudget", "P0_BUDGET", "CacheIdentity", "CandidateResult", "CandidateRun",
    "DeviceTimer", "DeviceTimerUnavailable", "HarnessError", "LAUNCH_CONTRACT_VIOLATED",
    "MEASUREMENT_SOURCE_DEVICE", "MEASUREMENT_SOURCE_SIMULATED", "OK", "PRUNED_STATIC",
    "Ranking", "SearchRound", "SimulatedClock", "Stats", "compute_stats",
    "confirmation_required", "expand_neighbourhood", "plan_digest", "pruned_result", "rank",
    "run_candidate", "search_case", "source_digest", "summarise_rounds", "toolchain_identity",
    "top_k",
]
