# TL-LOWERING-SPEC：S0 作者样例的 TileLang 发射下限与 read_actual_tl 读取字段（task-19）

- 对象 build：tfnpu-v2 site-packages/tilelang 0.1.15+cpu.git0e687c17（bundled tvm 源自 tilelang-main-0e687c17-clone，traceback 可证）。file:line 均为该包内路径。方法：只读源码 + /tmp 只写源码再 parse 的实验（未 compile、未触设备，临时文件已删）。
- S0 基线：targets/ascend/tests_native_mainline/fixtures/authors/minimal_aiv.py —— For(tf.pipelined(4, max_stage=2, max_buffer_versions=2))、SimdVF 内 Vld/Vld/Pset/Vadd/Vsts、域外 GmToUb×2 + UbToGm×1。

## 1. 最小 TL 源码形态（已 parse 实测通过，骨架如下）

    import tilelang
    import tilelang.ascend.language as T
    LANES = 64
    @T.prim_func
    def s0_minimal_aiv(x: T.Tensor((LANES,), "float32"), out: T.Tensor((LANES,), "float32")):
        base = T.alloc_shared((LANES,), "float32")      # UB 载体（scope shared.dyn）
        T.copy(x, base)                                  # GmToUb
        for k in T.Pipelined(4, num_stages=2):           # TL 只有 num_stages，见“无 API”项
            tile = T.alloc_shared((LANES,), "float32")
            T.copy(x, tile)
            with T.SimdVF():
                left = T.simd.vld(base[0])               # 必须带索引访问
                right = T.simd.vld(tile[0])
                pred = T.simd.pset(32)
                added = T.simd.vadd(left, right, pred)
                T.simd.vsts(base[0], added)              # addr 在前、src 在后
        T.copy(base, out)                                # UbToGm

- 逐 op 映射（全部真实签名）：GM→UB/UB→GM = `T.copy(src, dst)`（example_vecadd.py:44 基线）；UB = `T.alloc_shared`（TL/ascend/language/allocate.py:15，默认 scope="shared.dyn"；**无 T.alloc_ub**）；向量区 = `T.SimdVF()`（frame.py）；vld = `T.simd.vld(addr, dist="NORM", post_inc=None)`（simd.py:324，**addr 必须是 BufferLoad**（如 base[0]），自动包 access_ptr(r, extent=64)；整 Buffer 会 AttributeError——实测）；pset = `T.simd.pset(elem_width, dist="PAT_ALL")`（simd.py:111，返回 boolx256）；vadd = `T.simd.vadd(src0, src1, mask=None, mode="MODE_ZEROING")`（simd.py:554）；vsts = `T.simd.vsts(addr, src, mask=None, dist="NORM_B32")`（simd.py:457）。
- 本 build **没有对应 API** 的 S0 概念：①tf.pipelined 的 typed `max_stage`/`max_buffer_versions`——TL 只有 T.Pipelined(num_stages=)；显式版本注解 `T.annotate_buffer_versions` 在函数顶层与 SimdVF 区内调用均抛 ValueError（sblock_attr 必须"top of a T.sblock()"，实测两次）→ **显式版本声明今天没有可用 TL 面**，版本仍属 tilefoundry plan 侧；②显式 vreg 分配（寄存器由 vld/vadd 返回，无 alloc API；simd.alloc_var/alloc_local 是指针/local 变体）；③S0 的 ascend_gm_to_ub/ascend_vld/… 命名不是 TL 源面 API，它们是 AscendLowerTileOp 之后的内建名。

## 2. 发射器最小输出契约

- **source 必须是磁盘文件**：@T.prim_func 依赖 inspect.getsourcelines（eager/utils.py:61；builder.py:1791 re-raise）→ 发射器先写临时 .py 再 import/parse；stdin/exec/heredoc 直接失败（实测）。
- parse 本身**不需要 PassContext/target**（纯 Python+AST 构造）；target="ascend" 与 pass_configs 只在 tilelang.compile（jit/__init__.py `compile(func, out_idx, execution_backend, target, target_host, verbose, pass_configs, compile_flags)`）时消费：normalize（transform/pass_config.py:286-296）→ PassContext(opt_level=3, config=…)（jit/kernel.py:310）。双键合同 `{tl.enable_auto_schedule: False, tl.disable_shared_memory_reuse: True}` 落在 compile 侧；read_actual_tl 的 parse 观察点不受双键影响。
- artifact 至少携带：TL 源文本 + 其 sha256 + parse 得到的 PrimFunc identity（本次实测 parse 是确定性的、无随机面）。

## 3. read_actual_tl 必须读回并核对的字段清单（全部由 §5 实验输出证明）

**读得到**（parse 级，字段名→本 build 真实来源）：
1. `params`：Var 名（x_handle/out_handle）+ `buffer_map` 条目（name=x、dtype=float32、shape=(64,)、scope=global）。
2. `attrs`：{"global_symbol": 函数名}。
3. GM/UB 拷贝：`Call op=tl.tileop.ascend_copy`，args=[`tl.region(x[0], 1, 64)`, `tl.region(base[0], 2, 64)`]——region=(访问表达式, rw, extent)，1=read/2=write，extent 即拷贝元素数。
4. For：`loop_var=k, min=0, extent=4, kind=0`（T.Pipelined(4) → kind=0 普通 For；bounds/step 可读）。
5. 向量区：`SBlock name="SIMDVF" annotations={'tl.vf_source_index': 0}` + `AttrStmt tl.simdvf_scope="simdvf" value=1`——区 kind 可读。
6. UB buffer 声明：tirx **Bind** 结点内 `T.Buffer((64,), scope="shared.dyn")`——名字/shape/dtype/scope 可读。
7. 指令：`Call op=tl.simd.vld args=[tl.access_ptr(base[0], 64, 1), "NORM"]`（access_ptr=(expr, extent, rw)）；`tl.simd.pset args=[IntImm(32), "PAT_ALL"]`（**elem_width 可读**）；`tl.simd.vadd args=[Var left, Var right, Var pred, "MODE_ZEROING"]`（**mask 操作数与 mode 可读**）；`tl.simd.vsts args=[tl.access_ptr(base[0], 64, 2), Var added, Var v, "NORM_B32"]`。
8. 向量 dtype：Bind/let 上 float32x64（=64 lanes，由 dtype 按 simd.py:96 _vreg_lanes 推导）、boolx256。
9. 常量：IntImm（loop extent 4、elem_width 32、access extent 64）。

**读不到 / 今天做不到**（如实）：T.Pipelined 的 `num_stages` 不持久化在 parsed IR（For 无 AttrStmt 包装，root block 无 annotations）→ 从 parse 不可恢复，max_stage/max_buffer_versions 更是 plan 侧概念；显式 buffer 版本注解无可用面（§1）；vreg lanes 数无独立字段（由 dtype 推导）；copy 的语义方向（gm→ub vs ub→gm）在 parse 级只能由两端 buffer scope 推断，无显式方向字段。

## 4. 双键关闭 AutoSchedule 后仍会动这段 kernel 的 pass 与观察点

- parse 得到的是**发射源 IR（pre-pipeline）**。compile 时按 ascend/pipeline.py 顺序，会触碰/可能触碰本 kernel 的 pass：RewriteDualCopy（AIV 双拷贝展开——GM→UB 是它的目标）、LegalizeSimdMerging（本 kernel MODE_ZEROING 预期 no-op 但 pass 会跑）、AscendLayoutInference/NormalizeAscendFractalStorage（可能加布局注解）、InsertNd2Nz 与 AscendInsertOOBPadding（仅 UB→L1/GM→L1，本 kernel 预期 no-op）、AscendSimdVFLowerParallel + AscendLowerTileOp（把 ascend_copy/tileop 降为 ascend_gm_to_ub 等内建——**op 名会变**）、NormalizeBufferVersion/RewriteAscendBufferVersionLayout、FlattenBuffer（**Buffer 结构不再保持**）、MakePackedAPI（**参数签名会变**）、RewriteFlagToBuf/AscendThreadSync（无 sync 预期 no-op）。双键 enable_auto_schedule=False 时 AutoSchedule 大分支（含 InsertSync）整段跳过、改跑 InferBufferAliases（pipeline.py:182-184）。
- **观察点裁定**：read_actual_tl 在 **parse-of-emitted-source**（§5 位置）读结构——这是唯一无需 compile 的观察点；pipeline 后的 IR 只能经 compile 时 tl.enable_dump_ir（jit/kernel.py:299-301，会落盘）观察，超出无编译合同 → 该对比能力标 UNKNOWN/边界。

## 5. 离线最小实验（只写源码 + parse，未 compile）

- 命令：把 §1 骨架写入 /tmp/s0_tl_probe2.py，然后 `PYTHONDONTWRITEBYTECODE=1 TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONPATH=/tmp tfnpu-v2/bin/python -c "from s0_tl_probe2 import s0_minimal_aiv as pf; print(type(pf)); print(str(pf))"` 加 §3 的 tirx 树 walk（贴于下）。
- 实测输出（节选，完整结构见 §3 各条）：

    === PARSED OK: PrimFunc ===
    --- buffer_map ---
      param x_handle -> name x | dtype float32 | shape ('64',) | scope global
      param out_handle -> name out | dtype float32 | shape ('64',) | scope global
    SBlockRealize -> block name=root
      SeqStmt of 3
          Call op=tl.tileop.ascend_copy
             arg Call op=tl.region args=['x[0]', '1', '64']
             arg Call op=tl.region args=['base[0]', '2', '64']
        For loop_var=k min=0 extent=4 kind=0
          SeqStmt of 2
              Call op=tl.tileop.ascend_copy
                 arg Call op=tl.region args=['x[0]', '1', '64']
                 arg Call op=tl.region args=['tile[0]', '2', '64']
            SBlock name=SIMD_VF
              annotations: {'tl.vf_source_index': T.int64(0)}
              AttrStmt key=tl.simdvf_scope node="simdvf" value=1
                SeqStmt of 6
                  [Bind] base = T.Buffer((64,), scope="shared.dyn")
                  Call op=tl.simd.vld
                     arg Call op=tl.access_ptr args=['base[0]', '64', '1']
                     arg StringImm "NORM"
                  Call op=tl.simd.pset
                     arg IntImm 32
                     arg StringImm "PAT_ALL"
                  Call op=tl.simd.vadd
                     arg Var left / arg Var right / arg Var pred / arg StringImm "MODE_ZEROING"
                  [Bind] v: T.boolx256 = T.tl.simd.pset(32, "PAT_ALL")   ← vsts 未传 mask 时自动补默认 pset
                  Call op=tl.simd.vsts
                     arg Call op=tl.access_ptr args=['base[0]', '64', '2']
                     arg Var added / arg Var v / arg StringImm "NORM_B32"
        Call op=tl.tileop.ascend_copy
           arg Call op=tl.region args=['base[0]', '1', '64']
           arg Call op=tl.region args=['out[0]', '2', '64']

- 附加实验（负例，证明“读不到”是实测而非假设）：T.annotate_buffer_versions({base:2}) 在函数顶层与 SimdVF 区内均抛 `ValueError: T.sblock_attr must be called at the top of a T.sblock()`（annotations.py:104 → tvm/tirx/script/builder/ir.py:470）。
- 结论：S0 最小 TL 骨架在本 build **可 parse、字段可读**；发射器须写盘、用索引访问、显式传 mask（避免 parse 自动补默认 pset 指令）、plan 侧版本/阶段留在 tilefoundry，不写进 TL 源。
