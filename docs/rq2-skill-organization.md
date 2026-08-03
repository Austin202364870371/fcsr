# RQ2：召回的 Skills 应如何组织，才能提升 Agent 使用效率与准确性？

**状态：已实现受控规划实验，尚未完成带 verifier 的端到端比较。**

## 问题与假设

同一批召回 Skill 的组织会影响注意力分配、选择顺序和计划质量。RQ2 比较 Flat、Hierarchy 和 Evidence Graph，不重新检索更多 Skill；唯一自变量是固定候选集合的呈现结构。

假设是 Hierarchy 可降低长列表浏览负担，Evidence Graph 可在有正文证据时呈现依赖或互补关系。是否优于 Flat 必须在任务、候选、模型和预算均相同的条件下判断。

## 当前资产

Hard-15 固定 15 个 SkillsBench v1.1 任务和 FCSR Hard pool Top-20，并对模型匿名化 Skill 与组名。

- **Flat**：相关度排序的 Skill 卡片。
- **Hierarchy**：按 Skill ID 命名空间内部归组，模型只见 `C01` 等匿名组名。
- **Evidence Graph**：只建立 `explicit_reference` 与 `same_namespace` 两类可审计边。
- **Planner**：DeepSeek JSON Plan-and-Execute，Pydantic 校验计划、输入指纹和 checkpoint。

当前只记录有效计划率、GT 覆盖、token 和计划结构，不能声称 SkillsBench 任务成功率。文本相似度不作为图边，图也不向 Top-20 外扩展。

## 实验设计与判据

三种条件共享任务集、Top-20、Skill 原文、模型、提示、步骤/token 预算、温度、重试与 verifier。规划层报告计划 JSON 有效性、GT 覆盖、选择数量、计划深度、token 与延迟；执行层报告 verifier reward / 成功率、首次成功率、步骤、工具调用、总 token、时间与失败类别。

如果某种结构只提高 GT 覆盖、没有提高 verifier reward，应表述为选择代理指标改善，而非任务完成改善。

## 下一步

1. 接入完整 SkillsBench `environment/` 和 outcome-based `verifier/`。
2. 实现共享的 `plan -> act -> observe -> verify` 执行器。
3. 导出每任务 trace，记录呈现内容、选择、调用、观察、verifier 和成本。
4. 固定 RQ1 检索器后做配对多次运行，报告任务级差异和置信区间。

## 入口

`scripts/run_hard15_experiment.py`、`scripts/sync_hard15_tasks.py`、`src/agent/hard15_organizations.py`、`src/agent/hard15_planning.py`、`src/agent/hard15_experiment.py`、`HARD15_RUN.md`。
