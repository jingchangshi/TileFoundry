# 03-19 — native VCADD / VCVT / VMULA microtest DESIGN (WBS 03-19)

**STATUS: UNSOURCED / UNVALIDATED. DESIGN ONLY. NO DEVICE / NPU WAS USED IN THIS ROUND.**
These files are DELIVERED EXECUTABLE for the approved V09 device window of WBS 03-20;
they were **not executed** here. The only CPU work performed this round was
(a) reading the selected build sources read-only, (b) generating CPU MODEL predictions,
(c) syntax-parsing the scripts (no import of tilelang / torch_npu, no compile, no launch).
A CPU model prediction is NOT a native contract, NOT a device observation, and NOT a
validation. Architect 03-20 must rule on observed bits before any native credit.

## 1. Selected build (verified read-only in this session)

| field | value | how obtained |
|---|---|---|
| tilelang VERSION | \`0.1.15\` | \`<user>/workspace/tilelang/VERSION\` |
| selected build id | \`0.1.15+cuda.git8e2d3c82\` | HANDOFF §5.7 (the build identity under test) |
| git HEAD | \`8e2d3c823b0c3a01d2ce68d47dabfe98e35cc22b\` | \`git rev-parse HEAD\` in that tree |
| \`simd_inst.h\` sha256 | \`979f4ff76eee82bf092addcc069b238e3084af890f11660498f00640a00f20a1\` | sha256sum, matches HANDOFF |
| \`simd.py\` sha256 | \`89483579ddde5e36a352a853720fc02ba2b06badeab9eca74b31bcde6a04b57f\` | sha256sum (extra binding) |

Source anchors (literal read, not paraphrase):

| object | line | what the SOURCE says |
|---|---|---|
| \`vcadd\` | \`simd_inst.h:688-702\` | \`asc_reduce_sum(dst, src, mask)\`; the merge overload returns \`merge_masked(dst, result, reduction_result_mask<DstVec>(PAT_VL1))\` (line 702) |
| \`vcpadd\` | \`simd_inst.h:681-684\` | \`asc_pair_reduce_sum(dst, src, mask)\` |
| \`vmula\` | \`simd_inst.h:404-406\` | wrapper forwards to \`::vmula(*dst, src_0, src_1, mask, mode)\` |
| \`vcvt\` | \`simd_inst.h:291-294\` | wrapper forwards to \`::vcvt(dst, src, mask, controls...)\` |
| mask granularity | \`simd.py:1270-1274\` | default mask width \`min(_mask_bits(src.dtype), _mask_bits(target_dtype))\` ⇒ **16-bit** for bf16<->f32 |
| \`vcvt\` API | \`simd.py:1209-1215\` | \`round="ROUND_R"\`, \`part=0\`, \`mode="MODE_ZEROING"\` declared |
| \`vmula\` API | \`simd.py:635-646\` | docstring "Fused multiply-add: dst = dst + src0 * src1"; \`dst\` must be a READ\|WRITE buffer access |
| \`vcadd\` API | \`simd.py:957-966\` | docstring "Pairwise add reduction" — a **docstring**, not a device contract |
| \`vdupv\` | \`simd.py:933-939\` | \`POS_LOWEST\` broadcast, explicitly separate |

**SOURCED**: the wrapper names, call arguments and mask-granularity RULE in those files.
**UNSOURCED**: the native reduction tree, the output-lane layout (PAT_VL1 semantics in
hardware), lane positions, inactive-lane policy, rounding/FTZ/DAZ, FMA fusion, part
encoding, and what \`asc_reduce_sum\` actually computes. Source wrappers do not fix any of
those; only a device observation on THIS build can. HANDOFF §5.7 keeps six \`vcadd\` calls
and an explicit \`vdupv(POS_LOWEST)\`; a broadcast is never baked into \`vcadd\`.

## 2. Files

| file | role |
|---|---|
| \`vectors.py\` | CPU-only predictor; 14 cases, full 64-lane input bits; \`--emit expected_bits.json\` was run ONCE on CPU this round (design artifact, labelled prediction) |
| \`expected_bits.json\` | full 64-lane input bits / dst prefill / mask + predicted outputs for each HYPOTHESIS; sha256 \`e3427b129b85b56e4ec2c6a365b5556628be8246a17b1f7524ac60581c371a76\` (pinned by \`test_case_data.py::test_expected_bits_sha_matches_readme\`) |
| \`device_guard.py\` | root / auth / device0 / single flock / npu-smi before+after / bounded timeout / selected-build hash verification |
| \`probe_base.py\` | 3-input + out kernel ABI, full-lane readback, model comparison, \`--plan\` vs \`--execute\` |
| \`vcadd_probe.py\`, \`vcvt_probe.py\`, \`vmula_probe.py\` | ONE instruction under test per kernel (transport vld/vsts/copy excluded); vcadd also has \`--six-calls\` (six \`vcadd\` + explicit \`vdupv(POS_LOWEST)\`) |
| this README | design, preconditions, methodology, 03-20 command and verdict semantics |

## 3. Constructive discrimination (why random search is not enough)

\`vectors.py\` places exact f32 cancellation values at chosen lanes. \`[2^24, 1, -2^24, 0]\`
at lanes 0–3 makes \`2^24+1\` round back to \`2^24\` (ties-to-even), so an **adjacent-first**
tree loses the 1 and yields \`0x00000000\`, while a **strided / 2-chain** tree cancels
\`2^24\` against \`-2^24\` first and keeps \`1.0\` (\`0x3F800000\`). Random inputs almost
never hit this boundary. Predicted signatures (CPU MODELS, not native facts):

| case (lanes) | seq | adjacent | strided | rr2 | rr4 |
|---|---|---|---|---|---|
| cancel lanes 0,1,2,3 | \`0x00000000\` | \`0x00000000\` | \`0x3F800000\` | \`0x3F800000\` | \`0x00000000\` |
| cancel lanes 0,1,32,33 (lane position) | \`0x3F800000\` | \`0x3F800000\` | \`0x00000000\` | \`0x00000000\` | \`0x00000000\` |
| rounding \`1,2^24,1\` at 0,1,32 | \`0x4B800000\` | \`0x4B800000\` | \`0x4B800001\` | \`0x4B800001\` | \`0x4B800001\` |
| cancel at lanes 60–63 (lane position) | \`0x00000000\` | \`0x00000000\` | \`0x3F800000\` | \`0x3F800000\` | \`0x00000000\` |

Other vectors: a distinct finite sentinel in EVERY dst lane reveals **output layout**
(PAT_VL1 / broadcast / group) and **inactive-lane preservation** (MODE_MERGING keeps the
prefill, MODE_ZEROING writes zeros); subnormal f32 lanes probe **FTZ/DAZ**; the
\`[1,1,2^24]\` placement probes **rounding order**. \`zero\` patterns are controls, not
evidence of correctness.

**FMA** (vmula, one-bit decision): \`a=0x3F7FFFF8\`, \`b=0x3F800004\` ⇒
\`a*b = 1-2^-42\`, rounded product \`0x3F800000\`; \`acc=0x34400000\` (= 1.5 ulp).
Unfused (mul then add) hits the exact tie → ties-to-even \`0x3F800002\`; fused single
rounding is just BELOW the tie → \`0x3F800001\`. A constant-0 accumulator, a removed
mask, or a baked-in broadcast are the MUTATIONS that must kill these tests.

**FTZ**: \`0x00000001\` / \`0x007FFFFF\` / \`0x00800000\` subnormal lanes; IEEE keeps them
(model prediction \`0x00FFFFFF\` for the used vector), a flush-to-zero device differs.
**FMA / FTZ / inactive / part / mask-granularity** are all separate named cases.

Methodology (HANDOFF §6): no stale results, no empty-set satisfaction, every
"cannot fail" test must be killed by a mutation, negatives prove they CAN fail first,
random search is not a non-existence proof, no ritual artifacts. These probes RECORD
observed bits; they do not declare a native "PASS". \`NO_MODEL_MATCH\` is an honest outcome
and keeps the UNSOURCED label.

## 4. Device preconditions (enforced in code, not just prose)

1. **ROOT** (\`os.geteuid()==0\`; device ops require root here).
2. \`TF_DEVICE_AUTH=1\` for this invocation only; absence is a refusal, not a default.
3. \`ASCEND_RT_VISIBLE_DEVICES=0\`, \`torch.npu.set_device(0)\`.
4. **ONE team measurement job**: non-blocking \`flock\` on \`/home/.tf_device0.lock\`; <!-- no-machine-path: allow -->
   a second job is refused (\`DEVICE_LOCK_HELD\`). The parent holds the lock, children
   must NOT re-acquire (V09 note). No NPU reset by the probe.
5. \`npu-smi info\` snapshot **BEFORE** and **AFTER**, both saved raw
   (\`npu-smi.before.txt\` / \`npu-smi.after.txt\`).
6. **BOUNDED TIMEOUT** on every device subprocess (default 180 s, \`--timeout\`);
   a timeout is \`DEVICE_TIMEOUT\`, not a silent hang.
7. \`verify_build\` pins VERSION / git HEAD / \`simd_inst.h\` + \`simd.py\` sha256; any
   mismatch is \`BUILD_SHA_MISMATCH\` (exit 2) BEFORE a launch.
8. CPU and device ffi environments stay separate; no driver/CANN install, no sudo.

Exit codes: 0 observation captured + raw bits saved (NOT a native pass); 1 guard/
compile/launch failure; 2 build/ABI refusal; 75-style \`GuardError\` messages for
root/auth/lock failures. Verdict is written by architect 03-20, not by the probe.

## 5. 03-20 execution (NOT RUN IN THIS ROUND)

\`\`\`
# Lead holds the ONE approved V09 window; the probe itself takes the host lock once.
TF_DEVICE_AUTH=1 ASCEND_RT_VISIBLE_DEVICES=0 PYTHONDONTWRITEBYTECODE=1 \
  <device-env-python> tools/p3/native_probe/vcadd_probe.py \
  --case vcadd_cancel_adjacent --out <RUN>/vcadd/cancel_adjacent --execute --timeout 180
# repeat for each case / each probe:
#   vcvt_probe.py  --case vcvt_bf16_to_f32_part0 | vcvt_f32_to_bf16_rne_ftz | vcvt_inactive_lane_preservation
#   vmula_probe.py --case vmula_fma_fusion | vmula_inactive_lanes | vmula_loop_carry_ftz
#   vcadd_probe.py --six-calls --case vcadd_six_calls ...
\`\`\`
CPU-only \`--plan\` prints the preconditions, command and expected-bit file without
touching a device (also NOT run this round). Required deliverables per case:
\`observed_bits.json\` (all 64 lanes), \`kernel_source.txt\` + sha256,
\`device_window.json\`, before/after \`npu-smi.*\`, compile/launch stdout/stderr/exit,
and the selected-build CCE / binary hash. To become SOURCED, each claim needs:
selected-build source lines + wrapper + C++/CCE compile command + binary hash + device
output bits + an architect 03-20 ruling. A CPU model match alone never suffices.

## 6. What is explicitly NOT claimed

No native \`vcadd\` tree/layout, no \`vcvt\` carrier/part encoding, no FMA/FTZ/pack
guarantee, no "six calls = 64→1" proof, no 16-bit SimdVF mode, no cast fused into
\`vld\`, no relaxed checks, no dominance shortcut (same region instance, lexically
before). These stay **UNSOURCED / UNVALIDATED** until 03-20.
