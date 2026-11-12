# 恢复实施提示词

继续执行本工作包 v3。先读取本任务 progress.json、当前 git 状态和最近一份集成记录。保留已完成且能从正常入口复现的工作，不重跑无关研究，也不按旧 gate 字符串跳过尚未兑现的能力。

核对当前 commit 是否仍对应已有产物。若修改过 source、plan、layout、completion 或 toolchain，旧相应结果只作历史参考。使用 next_tasks.py 找依赖已满足的任务，分给当前可用 owner；公共 schema/主入口仍由 Lead 集成。

恢复前先确认没有遗留本任务 NPUjob；不杀其他任务，不 reset。继续沿唯一设备 executor 队列执行。用户已授权实现和测试，常规步骤不重复询问。

目标仍是两个 native 单逻辑 kernel、全部性能族、真实预算搜索、最终原 40 例和独立复现。若只有模型、文本 twin 或直接 TL 旁路通过，应回到对应框架任务完成接线。
