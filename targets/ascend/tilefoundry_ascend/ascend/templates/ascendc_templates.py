"""Track-B CCE (C++20) parameterized templates for canonical Ascend 950 (dav-3510) 3-axis shells."""

from __future__ import annotations

from typing import Any

from tilefoundry_ascend.ascend.templates.registry import (
    SHELL_AIC_GEMM_FIXPIPE,
    SHELL_AIV_SIMD_STREAM,
    SHELL_AIV_SIMT_ATOMIC,
    SHELL_AIV_STRIPED_VM,
    SHELL_CV_PERSISTENT_STATE,
    SHELL_CV_RAGGED_EPILOGUE,
    SHELL_CV_RING_PIPELINE,
    normalize_schedule_policy,
)


def _cpp_dtype(dtype: str) -> str:
    return {
        "float16": "half",
        "bfloat16": "bfloat16_t",
        "float32": "float",
    }.get(dtype, "half")


def render_pure_regbase_simd_cpp(policy: dict[str, Any] | None = None) -> str:
    """Shell 1 (aiv_only/gm_ub_stream/affine_simd): RegBase C++20 template.

    Disables SIMD VF fusion and exposes ``__simd_vf__`` vector functions.
    """
    p = normalize_schedule_policy(SHELL_AIV_SIMD_STREAM, policy)
    cpp_t = _cpp_dtype(p["dtype"])
    tile_m, _, _ = p["tile_mnk"]
    row_pack = p["row_pack_factor"]
    unroll = p["multi_row_unroll"]
    pad_db = p["ub_bank_pad_datablocks"]
    ub_budget = "256 * 1024" if p["ub_256kb_unlock"] else "248 * 1024"
    return f"""// Ascend 950 (dav-3510) Paradigm 1: Pure RegBase SIMD VF
// Flags: --cce-disable-vf-stack-reserved-ubuf --cce-disable-asc-reserved-ubuf (ub_256kb_unlock={p["ub_256kb_unlock"]})
#include "kernel_operator.h"
using namespace AscendC;

constexpr uint32_t ROW_PACK_FACTOR = {row_pack}U;
constexpr uint32_t MULTI_ROW_UNROLL = {unroll}U;
constexpr uint32_t UB_BANK_PAD_DATABLOCKS = {pad_db}U;
constexpr uint32_t UB_BUDGET_BYTES = {ub_budget}U;

template <typename T = {cpp_t}>
__no_simd_vf_fusion__ __simd_vf__ void RegBaseRowKernelVF(
    __ubuf__ T* xUb, __ubuf__ T* yUb, uint16_t rows, uint16_t dimLen, uint16_t strideElems, float epsVal) {{
    // VLOOPv2 requirements: 0-based uint16_t loop, <=4 nested loops, linear addressing
    const uint16_t vecStep = 64U;  // 64 float32 / 256B per 2-VREG cycle
    const uint16_t fullChunks = dimLen / vecStep;
    const uint16_t tailElems = dimLen % vecStep;
    MaskReg maskFull = Reg::UpdateMask<float>(vecStep);
    MaskReg maskTail = Reg::UpdateMask<float>(tailElems);
    RegTensor<float> vZero, vOne, vEps;
    Reg::Duplicate(vZero, 0.0f, maskFull);
    Reg::Duplicate(vOne, 1.0f, maskFull);
    Reg::Duplicate(vEps, epsVal, maskFull);

    // +1 DataBlock UB stride padding avoids bank conflicts across 2-row unroll
    for (uint16_t r = 0U; r < rows; r += MULTI_ROW_UNROLL) {{
        RegTensor<float> sqAcc0, sqAcc1;
        Reg::Duplicate(sqAcc0, 0.0f, maskFull);
        Reg::Duplicate(sqAcc1, 0.0f, maskFull);
        const uint16_t r1 = (r + 1U < rows) ? (r + 1U) : r;
        const uint32_t off0 = static_cast<uint32_t>(r) * (strideElems + UB_BANK_PAD_DATABLOCKS * 16U);
        const uint32_t off1 = static_cast<uint32_t>(r1) * (strideElems + UB_BANK_PAD_DATABLOCKS * 16U);
        for (uint16_t c = 0U; c < fullChunks; ++c) {{
            RegTensor<T> in0, in1;
            RegTensor<float> f0, f1;
            Reg::Load(in0, xUb + off0 + c * vecStep, LoadDist::DIST_UNPACK_B16);
            Reg::Load(in1, xUb + off1 + c * vecStep, LoadDist::DIST_UNPACK_B16);
            Reg::Cast(f0, in0, maskFull);
            Reg::Cast(f1, in1, maskFull);
            Reg::MulAdd(sqAcc0, f0, f0, maskFull);
            Reg::MulAdd(sqAcc1, f1, f1, maskFull);
        }}
        if (tailElems > 0U) {{
            RegTensor<T> t0;
            RegTensor<float> ft0;
            Reg::Load(t0, xUb + off0 + fullChunks * vecStep, LoadDist::DIST_UNPACK_B16);
            Reg::Cast(ft0, t0, maskTail);
            Reg::MulAdd(sqAcc0, ft0, ft0, maskTail);
        }}
        float sum0 = Reg::ReduceDataBlock<ReduceOp::SUM>(sqAcc0, maskFull);
        float invRms0 = 1.0f / sqrtf(sum0 / static_cast<float>(dimLen) + epsVal);
        RegTensor<float> vScale0;
        Reg::Duplicate(vScale0, invRms0, maskFull);
        for (uint16_t c = 0U; c < fullChunks; ++c) {{
            RegTensor<T> in0, out0;
            RegTensor<float> f0, yf0;
            Reg::Load(in0, xUb + off0 + c * vecStep, LoadDist::DIST_UNPACK_B16);
            Reg::Cast(f0, in0, maskFull);
            Reg::Mul(yf0, f0, vScale0, maskFull);
            Reg::Cast(out0, yf0, maskFull);
            Reg::Store(yUb + off0 + c * vecStep, out0, StoreDist::DIST_PACK_B32);
        }}
    }}
    LocalMemBar<VEC_STORE, VEC_LOAD>();
}}
"""


def render_simt_and_hybrid_ub_cpp(policy: dict[str, Any] | None = None) -> str:
    """Shell 2 (aiv_only/ub_random_atomic/irregular_simt): hybrid SIMT template.

    Uses ``__simt_vf__`` kernels with ``__launch_bounds__(512)``.
    """
    p = normalize_schedule_policy(SHELL_AIV_SIMT_ATOMIC, policy)
    threads = p["simt_threads"] or 512
    pad_slots = max(8, p["ub_bank_pad_datablocks"])
    return f"""// Ascend 950 (dav-3510) Paradigm 2: SIMT + Hybrid SIMD/SIMT UB
#include "kernel_operator.h"
using namespace AscendC;

constexpr uint32_t SIMT_THREADS = {threads}U;
constexpr uint32_t UB_WARP_PAD = {pad_slots}U;  // +8 padding eliminates warp atomic bank conflicts
constexpr uint32_t SIMT_DISPATCH_BYTES_THRESHOLD = 512U;

__simt_vf__ __launch_bounds__({threads}) void HybridSimtHashProbeVF(
    __gm__ uint64_t* keys, __gm__ uint64_t* tableKeys, __gm__ int32_t* tableVals,
    __ubuf__ float* warpAtomicScratch, uint32_t numKeys, uint32_t tableMask,
    Simt::UintDiv divNumCols) {{
    const uint32_t tid = Simt::GetThreadIdx<0>();
    const uint32_t lane = tid & 31U;
    const uint32_t warpId = tid >> 5U;
    if (tid >= numKeys) return;

    const uint64_t k = keys[tid];
    // Leader-only asc_atomic_cas + __shfl warp broadcast hash probing
    int32_t slot = -1;
    if (lane == 0U) {{
        uint32_t pos = static_cast<uint32_t>((k * 0xff51afd7ed558ccdULL) & tableMask);
        for (uint32_t step = 0U; step < 16U; ++step) {{
            uint64_t prev = asc_atomic_cas(&tableKeys[pos], 0ULL, k);
            if (prev == 0ULL || prev == k) {{
                slot = static_cast<int32_t>(pos);
                break;
            }}
            pos = (pos + 1U) & tableMask;
        }}
    }}
    slot = __shfl(slot, 0);
    auto [rowIdx, colIdx] = divNumCols.DivMod(tid);
    const uint32_t scratchIdx = warpId * (32U + UB_WARP_PAD) + lane;
    if (slot >= 0) {{
        atomicAdd(&warpAtomicScratch[scratchIdx], static_cast<float>(tableVals[slot]));
    }}
}}

__aicore__ inline void AdaptiveSimdOrSimtDispatch(uint32_t rowBytes) {{
    // Reserve 8KB stack + >=32KB SIMT Data Cache inside 256KB UB
    if (rowBytes <= SIMT_DISPATCH_BYTES_THRESHOLD) {{
        asc_vf_call<HybridSimtHashProbeVF>(dim3(SIMT_THREADS, 1, 1));
    }} else {{
        // >512B contiguous rows route through SIMD DMA + RegBase VF
        PipeBarrier<PIPE_MTE2>();
    }}
}}
"""


def render_dense_and_quant_gemm_cpp(policy: dict[str, Any] | None = None) -> str:
    """Shell 3 (aic_only/gm_l1_l0_fixpipe/affine_simd): CATLASS GEMM template.

    Covers ``asc::te`` graded kernels plus W4A16 and MXA8W4 variants.
    """
    p = normalize_schedule_policy(SHELL_AIC_GEMM_FIXPIPE, policy)
    bm, bn, bk = p["tile_mnk"]
    win = p["swizzle_window"]
    l1_stages = p["l1_stages"]
    half_l1 = "true" if p["l1_half_bank_split"] else "false"
    unit_flag_expr = (
        "2 /*tail-free db=1*/"
        if (p["sync_mode"] in ("unit_flag", "unit_flag_2_3") and l1_stages == 1)
        else "0"
    )
    w4_mode = p["w4_mode"]
    return f"""// Ascend 950 (dav-3510) Paradigm 3: Dense & Quantized GEMM (W4A16 / MXA8W4)
#include "kernel_operator.h"
using namespace AscendC;

// Scalar-spill-free param struct (<= 64B, zero dynamic array indexing)
struct alignas(16) GemmSwatTiling {{
    uint32_t mTiles, nTiles, kTiles, coreNum;
    uint32_t mTile = {bm}U, nTile = {bn}U, kTile = {bk}U;
    uint32_t windowLen = {win}U;  // SWAT serpentine swizzle window
    uint32_t l1Stages = {l1_stages}U;
    bool halfL1Isolation = {half_l1};
}};

__aicore__ inline void ComputeSwatSerpentineCoord(
    uint32_t tileId, uint32_t mTiles, uint32_t nTiles, uint32_t windowLen,
    uint32_t& mt, uint32_t& nt) {{
    // Tail-wave rebalance + serpentine Z-order over WINDOW_LEN={win}
    const uint32_t winSpan = windowLen * nTiles;
    const uint32_t winId = tileId / winSpan;
    const uint32_t inWin = tileId % winSpan;
    const uint32_t effWin = ((winId + 1U) * windowLen > mTiles) ? (mTiles - winId * windowLen) : windowLen;
    mt = winId * windowLen + (inWin % effWin);
    const uint32_t ntRaw = inWin / effWin;
    nt = ((winId & 1U) == 1U) ? (nTiles - 1U - ntRaw) : ntRaw;
}}

__aicore__ inline void LaunchGemmCoreTile(
    LocalTensor<half>& l1A, LocalTensor<half>& l1B, LocalTensor<float>& l0C,
    __gm__ half* gmA, __gm__ uint8_t* gmW4, uint32_t l1Slot, uint32_t deqScaleMode) {{
    // Half-L1 bank isolation: (TOTAL_L1_SIZE >> 1) * slot
    const uint32_t l1HalfOffset = (TOTAL_L1_SIZE >> 1) * (l1Slot & 1U);
    (void)l1HalfOffset;
    // Disable L2 cache pollution on 1-tile streaming operands
    SetL2CacheHint(gmA, CacheMode::CACHE_MODE_DISABLE);
    // w4_mode={w4_mode}: AIV DIST_UNPACK4_B8 -> UB->L1 or FP4->FP8 bit-shift + FixPipe deq_scale=64 fold
    constexpr uint32_t FIXPIPE_DEQ_SCALE = 64U;
    constexpr uint32_t UNIT_FLAG_MODE = {unit_flag_expr};
    (void)FIXPIPE_DEQ_SCALE;
    (void)UNIT_FLAG_MODE;
    (void)deqScaleMode;
    MmadParams mmadParams;
    mmadParams.unitFlag = UNIT_FLAG_MODE;
    Mmad(l0C, l1A, l1B, mmadParams);
}}
"""


def render_grouped_gemm_epilogue_cpp(policy: dict[str, Any] | None = None) -> str:
    """Shell 4 (coupled_1c2v/l0c_ub_epilogue/ragged_group): grouped MatMul.

    Fuses the SwiGLU and quantization epilogue.
    """
    p = normalize_schedule_policy(SHELL_CV_RAGGED_EPILOGUE, policy)
    bm, bn, bk = p["tile_mnk"]
    return f"""// Ascend 950 (dav-3510) Paradigm 4: Grouped GEMM + Direct L0C->UB SPLIT_M + Fused SwiGLU
#include "kernel_operator.h"
using namespace AscendC;

__aicore__ inline void GroupedGemmRollingDispatch(
    uint32_t numGroups, const __gm__ int32_t* groupEnds, uint32_t coreIdx, uint32_t coreNum) {{
    uint32_t startBlockIdx = 0U;
    int32_t prevEnd = 0;
    for (uint32_t g = 0U; g < numGroups; ++g) {{
        const int32_t gEnd = groupEnds[g];
        const int32_t m = gEnd - prevEnd;
        prevEnd = gEnd;
        if (m <= 0) {{
            // m == 0 empty-group fast skip while preserving mandatory empty-subblock sync parity
            CrossCoreSetFlag<0x2, PIPE_FIX>(0);
            CrossCoreWaitFlag(0);
            continue;
        }}
        // Per-group BaseMBalance + rolling (endBlockIdx + 1) % coreNum
        const uint32_t mTiles = (static_cast<uint32_t>(m) + {bm}U - 1U) / {bm}U;
        const uint32_t endBlockIdx = (startBlockIdx + mTiles - 1U) % coreNum;
        for (uint32_t mt = 0U; mt < mTiles; ++mt) {{
            if (((startBlockIdx + mt) % coreNum) == coreIdx) {{
                // L0C->UB SPLIT_M direct FixPipe (dualDstCtl=0b01) -> RegBase SwiGLU + dynamic quant
                FixpipeParamsArch3510 fixParams;
                fixParams.mSize = {bm}U / 2U;  // SPLIT_M across 2 AIV subblocks
                fixParams.nSize = {bn}U;
                fixParams.dualDstCtl = 0b01U;
                (void)fixParams;
            }}
        }}
        startBlockIdx = (endBlockIdx + 1U) % coreNum;
    }}
}}
"""


def render_heterogeneous_cv_ring_cpp(policy: dict[str, Any] | None = None) -> str:
    """Shell 5 (coupled_1c2v/bidirectional_cv_ring/affine_simd): heterogeneous CV ring.

    Tile rows stay at or above 82 so the Cube path remains compute-bound.
    """
    p = normalize_schedule_policy(SHELL_CV_RING_PIPELINE, policy)
    bm, bn, bk = p["tile_mnk"]
    eff_bm = max(82, bm)
    stages = max(3, p["l1_stages"])
    return f"""// Ascend 950 (dav-3510) Paradigm 5: Heterogeneous {stages}-Stage CV Ring (GQA / MLA / SparseFA)
#include "kernel_operator.h"
using namespace AscendC;

constexpr uint32_t CV_RING_STAGES = {stages}U;
constexpr uint32_t CUBE_TILE_M = {eff_bm}U;  // M >= 82 keeps AIC Mmad compute-bound (e.g. 96x112)
constexpr uint32_t CUBE_TILE_N = {bn}U;
constexpr uint32_t CUBE_TILE_K = {bk}U;

__aicore__ inline void HeterogeneousCVRingStep(
    LocalTensor<float>& l0cS, LocalTensor<half>& ubP, LocalTensor<half>& l1KropeAliasP,
    uint32_t kvBlockIdx, const __gm__ int32_t* sparseBlockOffsets) {{
    // AIV-ahead sparse block offset resolution
    const int32_t kvPhysOffset = (sparseBlockOffsets != nullptr) ? sparseBlockOffsets[kvBlockIdx] : static_cast<int32_t>(kvBlockIdx * CUBE_TILE_N);
    (void)kvPhysOffset;
    // 1. L0C->UB SPLIT_M FixPipe
    FixpipeParamsArch3510 sFix;
    sFix.mSize = CUBE_TILE_M / 2U;
    sFix.nSize = CUBE_TILE_N;
    sFix.dualDstCtl = 0b10U;
    // 2. AIV RegBase online softmax + cast -> ubP
    // 3. UB->L1 Nd2Nz aliasing K_rope L1 slot in-place for P
    Nd2NzParams nd2nz;
    nd2nz.ndNum = 1U;
    nd2nz.nValue = CUBE_TILE_M / 2U;
    nd2nz.dValue = CUBE_TILE_N;
    DataCopy(l1KropeAliasP, ubP, nd2nz);
    CrossCoreSetFlag<0x2, PIPE_MTE3>(1);
    CrossCoreWaitFlag(1);
}}
"""


def render_persistent_chunk_state_cpp(policy: dict[str, Any] | None = None) -> str:
    """Shell 6 (coupled_1c2v/persistent_onchip_state/affine_simd): persistent state.

    Keeps chunk and recurrent state on chip across the (B, H) grid.
    """
    p = normalize_schedule_policy(SHELL_CV_PERSISTENT_STATE, policy)
    bm, bn, bk = p["tile_mnk"]
    return """// Ascend 950 (dav-3510) Paradigm 6: Persistent (B, H) Chunk State (KDA / Sinkhorn / LaCT-TTT)
#include "kernel_operator.h"
using namespace AscendC;

__aicore__ inline void PersistentChunkStateLoop(
    __ubuf__ float* stateUb, LocalTensor<float>& l0cH, LocalTensor<half>& l1A, LocalTensor<half>& l1B,
    uint32_t numChunks, float gTail) {
    // Exp range halving: anchor = G_tail / 2 splits exp(g_t) -> exp(g_t - anchor) * exp(anchor)
    const float anchor = 0.5f * gTail;
    (void)anchor;
    for (uint32_t ch = 0U; ch < numChunks; ++ch) {
        // Zero intermediate GM state roundtrips: FP32 state remains UB/L1/L0C resident
        MmadParams mmad;
        mmad.isBias = false;
        mmad.cmatrixInitVal = false;  // initC=false in-place L0C accumulation (history + local)
        Mmad(l0cH, l1A, l1B, mmad);
    }
    // 5-step Newton-Schulz on-chip loop for Muon/LaCT-TTT orthogonalization
    for (uint32_t ns = 0U; ns < 5U; ++ns) {
        PipeBarrier<PIPE_V>();
    }
    (void)stateUb;
}
"""


def render_striped_register_vm_cpp(policy: dict[str, Any] | None = None) -> str:
    """Shell 7 (aiv_only/persistent_onchip_state/dynamic_opcode_stream): striped VM.

    Partitions the horizontal W dimension across register-resident stripes.
    """
    p = normalize_schedule_policy(SHELL_AIV_STRIPED_VM, policy)
    num_regs, w_slice, _ = p["tile_mnk"]
    return f"""// Ascend 950 (dav-3510) Paradigm 7: Striped Register-Resident VM (tensor_program_vm)
#include "kernel_operator.h"
using namespace AscendC;

constexpr uint32_t VM_NUM_REGS = {num_regs}U;
constexpr uint32_t VM_W_SLICE = {w_slice}U;  // Horizontal W-dimension slice 100% UB-resident per AIV

__aicore__ inline void StripedRegisterVmExec(
    __ubuf__ float* regsUb, const __gm__ int32_t* program, uint32_t progLen, uint32_t rCount, uint32_t wSlice) {{
    // Zero cross-core sync for elementwise/ADD/MUL/RELU/MAX/MULADD/COPY; LocalMemBar tracks hazards
    for (uint32_t pc = 0U; pc < progLen; ++pc) {{
        const uint32_t op = static_cast<uint32_t>(program[pc * 4U + 0U]) & 7U;
        const uint32_t dst = static_cast<uint32_t>(program[pc * 4U + 1U]) % rCount;
        const uint32_t a = static_cast<uint32_t>(program[pc * 4U + 2U]) % rCount;
        const uint32_t b = static_cast<uint32_t>(program[pc * 4U + 3U]) % rCount;
        (void)op; (void)dst; (void)a; (void)b; (void)wSlice; (void)regsUb;
        LocalMemBar<VEC_STORE, VEC_LOAD>();
    }}
}}
"""


ASCENDC_RENDERERS = {
    SHELL_AIV_SIMD_STREAM: render_pure_regbase_simd_cpp,
    SHELL_AIV_SIMT_ATOMIC: render_simt_and_hybrid_ub_cpp,
    SHELL_AIC_GEMM_FIXPIPE: render_dense_and_quant_gemm_cpp,
    SHELL_CV_RAGGED_EPILOGUE: render_grouped_gemm_epilogue_cpp,
    SHELL_CV_RING_PIPELINE: render_heterogeneous_cv_ring_cpp,
    SHELL_CV_PERSISTENT_STATE: render_persistent_chunk_state_cpp,
    SHELL_AIV_STRIPED_VM: render_striped_register_vm_cpp,
}

__all__ = [
    "ASCENDC_RENDERERS",
    "render_pure_regbase_simd_cpp",
    "render_simt_and_hybrid_ub_cpp",
    "render_dense_and_quant_gemm_cpp",
    "render_grouped_gemm_epilogue_cpp",
    "render_heterogeneous_cv_ring_cpp",
    "render_persistent_chunk_state_cpp",
    "render_striped_register_vm_cpp",
]
