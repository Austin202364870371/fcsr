# Hard-15 本地规划实验

这条管线把 FCSR 在 hard Skill pool 上得到的最终 Top-20，接到固定 15 个
SkillsBench v1.1 任务上，比较 Flat、Hierarchy、Graph 三种 Skill 组织方式。
当前只测“Skill 选择与计划质量”，不执行任务，也不把指标称为任务成功率。

## 架构边界

```text
原始 75 题 + FCSR Top-20 + hard Skill pool
                  |
           固定 Hard-15 / S01..S20 匿名化
                  |
        Flat | Hierarchy | Evidence Graph
                  |
          DeepSeek JSON Plan-and-Execute planner
                  |
      有效计划率、GT 覆盖、token、结构统计

以后接入：工具执行循环 -> SkillsBench 沙箱 -> 原始 verifier -> reward
```

Hierarchy 使用 Skill ID 命名空间做内部归组，但对模型只显示 `C01` 等匿名组名。
Graph 只建立两类可审计边：正文明确提到另一个候选时的
`explicit_reference`，以及共享匿名命名空间的 `same_namespace`。它不把文本
相似度伪装成依赖关系，也不会从 Top-20 以外扩图。

## 安装

在项目根目录执行：

```powershell
conda activate agent-learn
pip install -r requirements-hard15.txt
```

现有 `.env` 可以直接使用：

```dotenv
DEEPSEEK_API_KEY=你的密钥
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

## 直接运行

先同步约 27 MB 的公开任务上下文并做无 API 预检：

```powershell
$env:PYTHONPATH="src"
python scripts/run_hard15_experiment.py --sync --dry-run
```

预检成功后执行 45 次调用（15 题乘 3 种结构）：

```powershell
$env:PYTHONPATH="src"
python scripts/run_hard15_experiment.py
```

每完成一道题就原子写入 `reports/agent/hard15/plans/`。同配置重跑会跳过
已成功项并重试失败项；模型、预算或源版本变化时会拒绝混用旧 checkpoint。
最终查看 `reports/agent/hard15/summary.json`。`tasks.jsonl` 和
`presentations/` 是模型可见数据，`evaluation.jsonl` 含私有 GT 映射，不能放入
提示词。

## verifier 能力边界

SkillsBench 每个任务的 verifier 在隔离任务环境中检查最终文件和状态，例如
文件是否存在、JSON/表格 schema、数值容差、调度可行性、跨文件一致性以及
领域约束；测试通过比例或布尔结果写成 reward。它是 outcome-based，不要求
Agent 采用 oracle 的步骤。

早期本地玩具 verifier（`exact`、`contains_keys`）和单工具 runtime 已移除，避免
它们被误当作对复杂任务完成情况的验证。因此本阶段能真实跑通 45 个规划请求和
规划指标，但不能跑出可信的 SkillsBench 任务成功率。下一阶段必须下载完整
environment/verifier，并接入
沙箱执行循环后再报告 verifier reward。

## 实现依据

- SkillsBench v1.1：固定任务由 `task.md`、`environment/`、`oracle/`、
  `verifier/` 组成，任务采用沙箱和 outcome-based verifier：
  <https://github.com/benchflow-ai/skillsbench>
- ReAct：执行阶段应交替进行推理、动作和环境反馈，而不是一次选一个工具：
  <https://arxiv.org/abs/2210.03629>
- Plan-and-Solve：先分解为子任务再执行，支持本阶段的显式 JSON 计划 schema：
  <https://aclanthology.org/2023.acl-long.147/>
- DeepSeek JSON Output：使用 OpenAI-compatible API 和
  `response_format={"type":"json_object"}` 后仍进行本地 Pydantic 校验：
  <https://api-docs.deepseek.com/guides/json_mode/>

这里采用的是“Plan-and-Execute 的 planner 半段”。等 verifier 执行环境接通后，
再加入 `plan -> act -> observe -> verify -> replan` 状态循环，届时 Flat、Hierarchy、
Graph 的主结论必须以同一 verifier 下的端到端成功率为准。
