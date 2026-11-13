# 原始测试与输入合同

## 权威顺序

本轮用户明确以两个归档的 `golden.py`、测试和实际输入为语义依据。`fixtures/` 保留这些原文件，以及原 desc、proto 和许可证。候选实现、优化日志和历史性能不定义最优目标，也不决定编译器架构。

`case_matrix.json` 是 40 例的机器可读索引。它保留正式 case ID、实际 shape、dtype、attrs、输入范围、原 note、label、seed 和源文件 SHA256。原 note 是背景描述；若与 shape 冲突，读取实际输入 shape。不得根据备注跳过或缩小用例。

## 固定入口

```python
attention_residual(v_stack, x, w, gamma, alpha, epsilon=1e-6)
mla(q_nope, q_rope, k_nope, k_rope, v,
    numKVHeads=1, scaleValue=-1.0,
    inputLayout="BSND", is_causal=False)
```

候选返回一个连续的、原 dtype 的普通 Tensor。shape 与原算子合同相同。原测试入口为 `main(operator_api)`；直接运行 test 文件而没有调用 main 不构成测试。

MLA 的 `get_input` 将 V 规整为 K_nope 的副本。共享的是数值语义，不是指针地址。候选可选择复用已读入的 K_nope，但不能要求两个入参相同地址。该规整发生在测试输入准备中，不计入候选计算 kernel；候选 host 不能额外启动比较或 clone kernel 来证明值相等。

## 使用随包启动器

下列候选文件路径由本地团队替换为实际产物。候选文件可以导入工程中已经实现的 TileLang 代码；它不能调用包内 CPU 参考模型来冒充设备实现。

```bash
python3 "$WORK_PACKAGE/tools/run_contract.py" \
  --case attention_residual \
  --operator-file /实际路径/attention_residual_operator.py \
  --report /实际路径/results/attention_residual_numeric.json

python3 "$WORK_PACKAGE/tools/run_contract.py" \
  --case mla \
  --operator-file /实际路径/mla_operator.py \
  --report /实际路径/results/mla_numeric.json
```

`--preflight-only` 只检查依赖/输入/设备，不导入候选，也不执行 main 或 kernel。它的成功不能替代 20 例通过。

启动器完整采集 stdout/stderr，核对 20 个唯一 label，拒绝缺失、重复、未知 label 或非明确 passed 记录，最后同步 NPU。数值总结果以 JSON 中的完整集合和状态为准，不能只看子进程退出码。

## 不得替换的比较语义

原 helper 以 `accuracy_mode='cann-bench'` 调用 `npu_compare`。附件未提供 npu_compare 实现，proto 中的 thresholds 为空。本地 P0 应定位生效比较器及阈值来源，记录版本和文件身份。缺失时正式验收 BLOCKED；不能自行猜测 allclose 阈值并称通过原测试。

helper 用 double 输入运行 oracle。attention golden 内部仍强制 FP32 中间计算；MLA golden 保持 double 计算。两者不能用同一句“fp64 oracle”等同其实际中间精度。包内 Python float 和 NumPy 模型只用于小型算法诊断。

默认关闭历史 cache，防止误用旧输入或旧 oracle。若本地因资源需要启用 cache，应核对 golden、case、seed、dtype 及比较器身份，并保留命中记录。不能将 OOM/timeout/依赖缺失变成 SKIP_PASS。

## 数值通过之外

`run_contract.py` 本身不证明单 kernel、计划一致性或性能。团队另行采集候选调用范围内的 launch、实际 TileLang 配置/后端 IR、资源与计时。所有记录用同一个候选源码/plan/binary 身份关联。

正式 40 例不可拆成两个各自不同、不完整的“全量通过”报告；最终分派实现应覆盖全部 case。可以分进程逐批运行，以控制 oracle 内存，但每个原 case 的输入和比较规则保持不变，最终按准确 label 合并且无缺失/重复。
