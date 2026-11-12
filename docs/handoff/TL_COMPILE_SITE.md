# TL-COMPILE-SITE（task-41）— 显式编译站点与二进制静态检视

owner flex-4-g3。四条路径全部实测：编译成功 / BLOCKED_ENV / 未知键 / 翻键。
未 launch kernel；计算节点是硬拒边界。命令前缀同 TL_SOURCEGEN_RUN.md。


## 1) 工具链解析与真实编译 (四条路径均实测)

bisheng 解析权只属于部署后端 (BISHENG_HOME/bin 或 PATH)。踩过的真坑: 我自己去搜安装目录,
于是"发现了"bisheng 而后端找不到 → 把环境缺陷报成内部缺陷。已改为 find_bisheng_path()
为唯一权威, 安装目录只在拒绝文案里说明装在哪、怎么加 PATH。

source /data/pri/cann-9.2.0/ascend-toolkit/set_env.sh → exit 0, bisheng 进 PATH,
ASCEND_HOME_PATH 被设, BISHENG_HOME 不设。只加 PATH 不设 ASCEND_HOME_PATH 时bisheng
能解析但跑不动: LLVM ERROR: PlugIn Err: can not find ASCEND_HOME_PATH. → 具名
TL_TOOLCHAIN_ENVIRONMENT_INCOMPLETE (保留原文)。

编译产物 (隔离 TILELANG_CACHE_DIR 冷编译, 真实数字):
- kernel.aibin: ELF64 EXEC, machine 0x1029 (CCE), 11 sections (.text, .ascend.meta,
  .comment, __CCE_KernelArgSize, __CCE_KernelMetaData, .ascend.meta.vsum_kernel,
  .debug_frame, .symtab, .shstrtab, .strtab); __CCE_KernelArgSize = 16 = 2 参数 x 8
  (readelf -x 与我方纯 Python ELF 头解析双核一致; 曾有两处 ELF 头偏移错误已修:
  e_shentsize/e_shnum/e_shstrndx 在 0x3A 不在 0x30; e_shstrndx 选的是 section header,
  名字表要取该 section 自己的 sh_offset)。
- readelf -sW 符号: .Kernel_Type_vsum_kernel, .AIV_Kernel_Type_vsum_kernel,
  __CCE_Mix_Kernel_Policy; 无 .CrossCoreSync 段。
- 两键真的进二进制: auto_schedule False→asc 1424B / True→2156B, aibin sha 与
  semantic_identity 都变; 同配置换 cache 重编译 → semantic 与 aibin sha 不变,
  只有 build_identity (本次时间戳 + cache 目录) 变。旧构建身份不可复用。
- manifest.json 逐文件 sha 与磁盘字节核对 (不信表)。
- JITKernel 的 n_regs/n_spills/resource_usage 实测 None/None/{} → 我方
  CompiledArtifactFacts 不含任何资源字段 (测试断言 resource_fields == [])。
- 产物按内容配对, 不按 mtime: kernel 目录由 device_kernel.asc 的 sha 必须等于本次
  返回的 Ascend C 文本来认定; aibin 用部署的 AscendBinaryCache.make_key 对同一
  输入重算归属, 再退回"本次新出现且唯一"; 多义则 TL_COMPILE_KERNEL_AMBIGUOUS /
  TL_COMPILE_BINARY_AMBIGUOUS 具名拒。REQUIRED_PRODUCT_FILES 拆 KERNEL_PRODUCTS 与
  BINARY_PRODUCTS, 因为 kernel.aibin 根本不在 kernel 目录, 旧写法必然误报 INCOMPLETE。


## 2) 设备节点: 控制节点=记录, 计算节点=拒绝 (已写成测试)

取证方法 (可复跑): /tmp/probe_fdtrack.py 用 os.readlink(/proc/self/fd/N) 在每个阶段做
集合差。bare 环境在 emit / 源码级 codegen / 完整 compile 各阶段新增 /dev 节点为空
(只有 /dev/null)。sourced 环境下 plain "import tilelang" 就打开了 /dev/davinci_manager
与 /dev/hisi_hdc —— 那时我们一行代码都还没跑。
所以规则是: COMPUTE_NODE = ^/dev/davinci[0-9]+$ 命中 → TL_COMPILE_OPENED_DEVICE 硬拒
(谁打开都拒); 控制节点只记入 CompiledArtifactFacts.control_nodes。边界由
test_device_node_rule_passive_control_versus_active_compute 注入假 _device_state 直接判,
不靠注释。validator 自己主动使用设备仍须 flock, 不归本切片。


## 3) 身份字节不接受散文 (lead 的自查要求)

- _target_identity 改为只覆盖机器值 (profile_id, core_domains, cross_core_mechanism,
  每 pool 的 capacity/banks/scope/pipe_pair/mode/status); _without_prose 按键名剔除
  source/note/description/reason/proof_source。
- 双向消融测试 test_rewording_target_prose_cannot_drift_completion_identity:
  改措辞 → completion identity 不变; 改 capacity → target_identity 必须变。
- codegen/ascend_binary 的 semantic_identity 只含 sha/machine/argument_size/文件表,
  build_identity 只含 toolchain/时间戳/cache; refusal detail 与 gap 文案不进任何
  identity (grep 核过)。
- reader 侧 TlFacts.unsupported 原本是 "attr x" 这类拼句且进入 digest, 已改成机器
  元组 ("attr", name): 措辞可改而 parsed_identity 不动。

