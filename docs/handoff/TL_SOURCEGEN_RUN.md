# TL-SOURCEGEN-RUN（task-37）— 源码级 Ascend C、arch 串、逐 pass IR、两条事件权威

owner flex-4-g3。HEAD f341e47+工作树。未 launch kernel。命令前缀：
PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=src:targets/ascend envs/tfnpu-v2/bin/python
另见 TL_COMPILE_SITE.md（二进制层）、TL_DTYPE_SUPPORT.md（16-bit/lane/身份字节/伪造自查）、TL_EVENT_SURFACE.md（事件发射面/EXIT 义务）。

## 1) arch 串：本 build 不校验，950PR→arch 记 UNKNOWN

- Target({kind:ascend}) exit 0；attrs 含 max_vector_bits:64 / supports_vector_predicate:true /
  launch_grid_only:true / supports_kernel_status_return:false；arch=None；chip/vendor 传入被拒。
- bisheng.get_target_npu_arch(...) → dav-3510，仅是 contrib/bisheng.py:93 的默认值
  npu_arch or os.environ.get("ASCEND_NPU_ARCH", "dav-3510")。
- 否定证据：arch="not-an-arch" 同样被接受 → 本 build 不校验 arch 串。950PR↔arch = UNKNOWN（待设备侧 task-36），不写成已确认。

## 2) 源码级真实 Ascend C：跑通，不需要 bisheng、不需要设备

lower_with_context(enable_host_codegen=False, enable_device_compile=False) + kernel_source：
independent_ub 1397B / 0.101s；minimal_aiv 1424B / 0.004s；同一 parse 重跑逐字节相同。文本含
asc_copy_gm2ub_align / simd_inst::vlds_norm<float> / asc_create_mask_b32(PAT_ALL) /
simd_inst::vadd(...MODE_ZEROING) / simd_inst::vsts_norm。已作为 NativeArtifact 第二层，三层身份
（ascend_identity / TL 源文本 / parsed PrimFunc）分记、互不冒充。

## 3) kernel frame 是必需的（实测，不是风格偏好）

无 T.Kernel 时 lower 直接失败（exit≠0，原文）："InternalError: Check failed: undefined.size()
== 0 (1 vs. 0) : In PrimFunc vadd variables (buf_dyn_shmem,) are used, but are not passed in as
API arguments"（make_packed_api.cc:1064）。已实现为支持域要求 TL_MISSING_MESH_SCOPE /
TL_UNSUPPORTED_MESH_LEVEL / TL_MULTIPLE_MESH_SCOPES，不让裸 InternalError 逃逸。

## 4) 更正：tilelang.compile 不与 OR-Tools 冲突；触发条件是导入顺序（我原说法作废）

lead 质疑后重测（逐次取真实 rc，不用 tail 掩盖）：

| 环境 | solve→npu | npu→solve | TF/ortools 先→compile→solve | 仅 import tilelang |
|---|---|---|---|---|
| bare（无 CANN env） | rc=1（无 torch_npu 可导） | rc=1 | rc=0 DONE | rc=0 |
| sourced（set_env.sh） | rc=0 DONE | rc=139 | rc=139 | rc=0 |

触发三元组：sourced 环境 AND torch_npu 先于 from ortools.sat.python import cp_model → rc=139，
栈在 google::protobuf::AssignDescriptorsImpl / operations_research::sat::SatSolver。反向顺序
rc=0；含完整 tilelang.compile 的路径也 rc=0 → 编译本身不改变进程状态。因此撤回"必须子进程"
这个架构前提，降级为一条可判定注意事项：同一进程内先导入 TF/ortools，再导入 tilelang。测试仍
走子进程，是因为要验证真实产物落盘与缓存隔离，不是被段错误逼的；子进程为可选规避，不是约束。

## 5) 逐 pass IR 观察点（可离线核对 refinement）

只设 tl.enable_dump_ir → 0 文件（jit/kernel.py:299 的 instruments 只在 compile 路径挂）。显式
PassContext(instruments=[tvm.ir.instrument.DumpIR(dump_dir, refresh=True)]) → 74 份逐 pass IR，
不需 bisheng、不 import torch_npu、不开设备节点。实际改变本 kernel IR 的 pass：BindTarget、
MaterializeKernelLaunch、AscendLayoutInference、AscendLowerTileOp、LowerAccessPtr、
LowerOpaqueBlock、ConvertSSA、AnnotateEntryFunc、AnnotateDeviceRegions、AnnotateReadOnlyParams、
InferBufferAliases、MakePackedAPI、LowerDeviceKernelLaunch、Filter、LowerIntrin、Simplify。
语义计数 vld=2 / pset=1 / vadd=1 / vsts=1 从第一份到最后一份不变（UB 行 2→3→9 只是分配合并与
intrinsic 展开）→ 数据指令没被后端改写；asc_create_flag / set_flag / wait_flag 在 74 份里全 0。

## 6) 已闭合：「源码 flag 计数 0 但 aibin 有同步段」的矛盾不存在

支持域只有数据指令，程序里根本没有事件；且 auto_schedule=False 时 InsertSync 不跑。所以源码计数
0 与 aibin 里没有 .CrossCoreSync 是同一件事的两侧，不是矛盾，更不是"后端替我们补了同步"。
.CrossCoreSync 有无只作旁证；事件证明以 site/channel 对账为准，task-40 落地前 reader 必须停在
INCOMPLETE，不许为变绿写成 VERIFIED。

## 7) 事件发射的两条权威事实（勿删；实测细节在 TL_EVENT_SURFACE.md）

- hard-event 串的唯一权威是部署 build 的两处 C++：GetResourcePipeName (ascend_pipe.h:75-91，
  命名 MTE1/MTE2/MTE3/M/V/FIX/S) 与 GetHardEvent (insert_sync.cc:751-765，拼接后施加四条别名
  MTE2_M→MTE2_MTE1、M_MTE2→MTE1_MTE2、MTE3_M→MTE3_MTE1、M_MTE3→MTE1_MTE3)。
- **Python 侧不校验这个串**：sync.py:44-45 只检查 isinstance(hard_event, str)，所以拼错没有任何
  一层会替我们拦住，错误直达 pto.set_flag 实参 (codegen_pto.cc:4847-4853)。故 emit 侧照抄 C++ 表
  并注明来源。

## 8) 判定

可行且已实测：源码级真实 Ascend C；逐 pass IR；真实 aibin+executable.so+params.json+manifest.json
（含 ABI arg-size 核对与 manifest 字节校验）。
仍 UNKNOWN（不降格放行）：950PR↔arch；S0/S1 内 16-bit **尚无作者程序能走到认证**——正确语法的
bf16 能 FINALIZE 并**确实到达**发射支持域，被 TL_BUFFER_SCOPE_MISMATCH（carrier scope 未承认）
拒，故 G1/G2 未闭合；16-bit lane/extent 打包无交叉核对依据（target 只声明
(64,32)），reader 已实现为 gap，上限 INCOMPLETE；JITKernel 资源数字；事件与 flag 槽位的真实
lowering 需 task-40/46 与设备证据。
