# RQ3：如何改进 Agent 的规划机制以提升复杂任务执行能力？

**状态：研究设计，尚未实现。**

## 问题与假设

当前仓库尚未实现端到端 Agent 环境执行。RQ3 研究显式状态和 verifier 驱动重规划，能否比一次性计划或纯 ReAct 循环提高复杂任务成功率并控制成本。

假设是先规划能减少无目标调用；只在可观测失败后重规划，可避免每步都调用模型并降低计划漂移。

## 最小闭环

```text
retrieve -> plan -> select skill -> act -> observe -> verify
                                     |              |
                                     +---- fail -----+ -> replan
```

状态记录任务 ID、候选 Skill ID、计划版本、当前子任务、动作参数、观察、verifier 结果、失败类别、步骤/token/时间预算。重规划必须保留旧计划和触发事件，不得覆盖 trace。

## 对照实验

1. **Direct ReAct**：依据当前观察直接选择 Skill 和动作。
2. **One-shot Plan-and-Execute**：开始时输出计划，失败不改计划。
3. **Verifier-gated Replan**：仅在 verifier、工具错误、前置条件违例或预算风险时修订未完成子任务。

三组共享模型、任务、候选、工具、步骤和 token 上限，并在固定检索条件下比较规划器。

## 判据与实施

主指标为 outcome-based verifier 成功率；辅助指标为首次计划成功率、重规划次数、成功和全部任务的步骤/token/时间、预算耗尽和计划执行偏离率。失败按缺 Skill、选错 Skill、错误参数、环境、计划依赖、验证、预算和解析错误分类。

先完成 SkillsBench 环境接口和稳定 trace schema，再实现两个基线与 verifier-gated replan，最后在固定任务子集上做配对评测并扩大规模。

## 与其他 RQ 的关系

RQ3 不改变 RQ1 的检索池与检索结果。文字反思与经验记忆由 RQ4 负责；RQ3 只处理基于执行状态的计划和重规划。
