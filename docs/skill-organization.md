# Hard-15 Skill 组织实验

## 1. 目标与研究问题

本实验在 SkillsBench 上回答：当检索结果、可见 Skill、Agent、模型和执行环境全部固定时，检索到的原始 Skill 采用 Flat、Hierarchy 或 Evidence Graph 呈现，是否会改变 Agent 的任务成功率与执行效率。

主比较包括：

1. `flat_top8 - no_skill`：FCSR 检索出的原始 Skill 是否提供增益；
2. `hierarchy_top8 - flat_top8`：层次索引是否优于线性平铺；
3. `graph_top8 - flat_top8`：证据图索引是否优于线性平铺。

Hard-15 是一次 60 条 trajectory 的试运行。流程验收后，实验扩展到 FCSR 与 SkillsBench 重合的 63 个任务；Hard-15 结果不替代 63 任务的正式统计结果。

## 2. 明确边界

### 2.1 本实验包含

- 冻结 FCSR 检索结果；
- 每任务固定 FCSR Top-8 原始 Skill；
- `no_skill`、`flat_top8`、`hierarchy_top8`、`graph_top8` 四个条件；
- OpenHands Agent；
- `deepseek/deepseek-v4-flash`；
- Daytona sandbox；
- SkillsBench 原始 task、environment、oracle 和 verifier；
- 离线组织元数据生成、确定性验证和任务盲人工复核。

### 2.2 本实验不包含

- 不重新训练、比较或调用其他检索器；
- 不使用其他 report 覆盖冻结预测；
- 不补做 114 份 Contract，也不把 Contract 注入 Agent；
- 不使用现有约 8,000 份 Contract 代替原始 Skill；
- 不让组织器读取 task prompt、GT、oracle、verifier 或历史运行结果；
- 不根据 Agent 成败反向修改 hierarchy 或 graph；
- 不做运行时图修复、重规划或 GraSP；
- 不把 Skill 别名出现次数当作可靠的 Skill 使用证据。

## 3. 不可变输入

### 3.1 唯一检索来源

唯一合法的检索结果目录为：

```text
D:\zsk\Junior\HUST-Summercamp\fcsr\reports\reranker\hard\fcsr-multiskill3x-rrf
```

项目内相对路径为：

```text
reports/reranker/hard/fcsr-multiskill3x-rrf/
```

生成器只允许读取该目录的 `predictions.json`。`details.jsonl`、`records.jsonl` 和 `summary.json` 可用于审计，但不得触发重新检索或重新排序。正式 manifest 必须记录四个文件的 SHA-256；运行期间任一指纹变化均立即停止。

每个任务的可见集合定义为：

```text
V8(task) = predictions.json[task][0:8]
```

不得在 hierarchy 或 graph 阶段从 Top-20 重新挑选节点，不得补充 Top-8 外 Skill。

### 3.2 Hard-15 任务

任务清单固定读取：

```text
data/agent/hard15/task_ids.txt
data/agent/hard15/task_catalog.json
```

固定的 15 个任务为：

```text
jax-computing-basics
dialogue-parser
econ-detrending-correlation
citation-check
enterprise-information-search
flood-risk-analysis
syzkaller-ppdev-syzlang
manufacturing-fjsp-optimization
threejs-to-obj
threejs-structure-parser
setup-fuzzing-py
suricata-custom-exfil
powerlifting-coef-calc
xlsx-recover-data
parallel-tfidf-search
```

本地核验结果为：15/15 位于冻结预测中，15/15 具有从官方 `benchflow/skillsbench` 数据集下载的 `task.md` 和 `environment/Dockerfile`，15/15 对应 Hugging Face revision `be2a6ce2cb1f4ff67ce937307cade0c5a0477a13`。

端到端执行必须使用服务器上的完整 SkillsBench 包，不使用本地裁剪后的 `data/agent/hard15/packages/`。实验版本固定为：

```text
SkillsBench version: v1.1
GitHub commit: b63b7b2850226b6aa4fb5929a8c1ac7bc4d9a6af
Hugging Face revision: be2a6ce2cb1f4ff67ce937307cade0c5a0477a13
```

服务器预检必须确认 15 个 `tasks/<task-id>/` 均存在，并至少具有：

```text
task.md
environment/Dockerfile
oracle/
verifier/
```

任何任务缺失、别名不一致或不在冻结预测中时，整个批次失败；不得静默跳过或缩小分母。

### 3.3 权威 Skill 载荷来源

FCSR Hard pool 的权威载荷文件为：

```text
data/raw/skills_hard.jsonl.gz
```

其本地 SHA-256 为：

```text
492BD8E7958434DEEAE97C91FBD6921AECEFB19EA16D4605F100B645BEC5AF31
```

该文件来自 SkillRouter 公开的 Hard tier。每条标准化记录包含：

```text
skill_id, name, description, body, source
```

它不是每个 Skill 的原始安装目录：`name` 和 `description` 已从 Skill 元数据中拆出，`body` 保存 Markdown 主体，记录中没有稳定的上游 repo URL、commit、原始路径或附件清单。因此本实验不能把“恢复 114 个原始 GitHub Skill 包”设为前置条件。

Hard-15 Top-8 共 120 个任务内 Skill 实例、114 个唯一 Skill ID。本地流式核验结果为 114/114 均能在 `skills_hard.jsonl.gz` 中精确找到，其中：

```text
source=pool        77
source=gt          24
source=distractor  13
```

`pool` 记录源自上游开放 Skill Registry；`gt` 记录由 SkillsBench Skill 派生；`distractor` 是 benchmark 构造的干扰 Skill。特别是 `gt` 和 `distractor` 并不保证存在可安装的上游原目录。当前上游 Registry 还会持续更新，直接下载其最新目录可能与冻结 FCSR 检索时的文本不一致。

因此本实验把 JSONL 中冻结的 `name + description + body` 定义为“基准权威原始载荷”。`skill_id` 和 `source` 只用于离线精确连接和审计，不展示给组织器或 Agent。每个载荷记录：

- Skill ID 与 FCSR rank；
- `name`、`description`、`body`；
- canonical record SHA-256；
- 三个文本字段的字符数与估算 token 数；
- 记录是否唯一。

缺失或同一 ID 对应多个不同记录时立即失败。Contract 数据不参与这一步。

## 4. 四个实验条件

| 条件 | 可见原始 Skill | 组织索引 |
|---|---:|---|
| `no_skill` | 0 | 无 |
| `flat_top8` | 固定 `V8(task)` | FCSR rank 顺序 |
| `hierarchy_top8` | 固定 `V8(task)` | 语义层次目录 |
| `graph_top8` | 固定 `V8(task)` | 有类型、有原文证据的关系图和阅读顺序 |

三个有 Skill 条件必须满足：

- S01--S08 ID 完全一致；
- S01--S08 与 FCSR rank 一一对应；
- 原始 Skill 正文完全一致；
- 正文边界和截断规则完全一致；
- 每个原子载荷的 SHA-256 完全一致；
- 唯一差异是顶部的组织索引。

## 5. 原子 Skill 载荷

### 5.1 载荷原则

Agent 接收 SkillRouter Hard JSONL 中冻结的原始语义字段，而不是 Contract、摘要、当前上游仓库的新版本或组织器改写文本。构建器为每任务建立固定映射：

```text
S01 = FCSR rank 1 原始 Skill
...
S08 = FCSR rank 8 原始 Skill
```

每条 JSONL 记录先按固定序列化规则重建为 Markdown，再以明确边界嵌入复合 Skill：

```markdown
### S01

Retrieval rank: 1

<original_skill>
Name: <JSONL name 原文>

Description:
<JSONL description 原文>

Body:
<JSONL body 原文>
</original_skill>
```

边界标记、字段标题和 `Retrieval rank` 属于统一包装，不属于组织器输出，并在三个条件中保持一致。真实 Skill ID 和 `source` 只保存在 Agent 不可见的 manifest 中；不得把可能含有 `gt/`、`distractor/` 或其他数据集来源信息的 ID 展示给组织器或 Agent。`name`、`description` 和 `body` 的值逐字保留。

canonical record hash 通过字段名固定、UTF-8、无额外空白的 canonical JSON 对 `name + description + body` 计算；渲染后的 payload 另算 SHA-256。主实验中的“原始 Skill”明确指 FCSR 建库和检索实际使用的这三个冻结字段。JSONL 未发布的脚本、模板和附件无法可靠恢复，也不进入本次提示词组织实验；若后续研究完整安装包可用性，必须作为独立变量重新预注册。

### 5.2 上下文预算

构建前先统计每任务 Top-8 的总 token。若模型与 Agent 能完整承载，则保留原文全文。若必须截断，规则须满足：

当前按 `name + description + body` 字符数除以 4 粗估，Hard-15 每任务 Top-8 约为 7.8k--35.6k tokens，平均约 20.3k；最终仍须使用实际模型 tokenizer 和完整 Agent prompt 复核。该规模优先采用全文注入，不预先截断。

- 在查看 Agent 结果前冻结；
- 对三种组织方式相同；
- 每个 Skill 使用相同的单文档上限；
- 不采用 rank 靠前者先耗尽共享预算的策略；
- 优先保留 `name`、`description`、正文标题、使用说明、步骤、约束与错误处理；
- 截断位置和截断后 payload hash 写入 manifest。

不得分别为 Flat、Hierarchy 和 Graph 调整正文长度。

## 6. 离线混合式组织器

### 6.1 输入隔离

每个任务生成一次组织元数据。组织器只读取：

- 不含任务语义的匿名 task key；
- S01--S08 alias；
- FCSR rank；
- 八条冻结的 `name + description + body` 载荷。

组织器不得读取：

- task ID 和 task prompt；
- 真实 Skill ID、Skill 数据集来源目录和 `gt/`、`distractor/` 等标签；
- GT Skill；
- oracle 与 verifier；
- SkillsBench 环境中的答案文件；
- Contract；
- Agent trajectory、reward 或错误日志。

组织器使用固定模型、固定 system prompt、temperature 0、严格 JSON Schema。记录模型标识、endpoint 类型、prompt hash、请求参数、原始响应和响应 hash。组织元数据在任何 Agent run 前生成、验证、复核并冻结。

若组织器与执行 Agent 都使用 DeepSeek-V4-Flash，必须在论文中披露潜在的同模型表达偏好；确定性验证和任务盲复核用于限制该偏差。组织器绝不生成或改写原子 Skill 正文。

### 6.2 Hierarchy Schema

Hierarchy 采用最多三层：

```text
operation family -> artifact/input family -> Skill alias
```

组织器输出 `hierarchy.json`，至少包含：

```json
{
  "schema_version": "skill-hierarchy-v1",
  "roots": [
    {
      "label": "Data extraction and recovery",
      "children": [
        {
          "label": "Spreadsheet files",
          "skills": ["S01", "S04"]
        }
      ]
    }
  ]
}
```

验证规则：

- S01--S08 恰好各出现一次；
- 不得增加、删除或重新排名 Skill；
- 最大深度为 3；
- 标签必须是简短能力或对象名词短语；
- 标签不得包含任务答案、具体输出值或隐藏文件内容；
- 无法可靠分类时允许使用 `Other`。

### 6.3 Evidence Graph Schema

Graph 的节点固定为 S01--S08，只允许以下边：

- `produces_requires`：源 Skill 的输出可作为目标 Skill 的输入或依赖；
- `setup_execute`：源 Skill 完成配置或准备，目标 Skill 执行主要操作；
- `execute_verify`：源 Skill 产生结果，目标 Skill 检查、验证或审计结果；
- `format_conversion`：源 Skill 输出格式可供目标 Skill 使用；
- `explicit_reference`：源 Skill 原文明示目标 Skill 对应工具或过程。

组织器输出 `graph.json`，每条边必须包含源、目标、类型，以及来自两条冻结 Skill 载荷的精确证据片段。验证规则：

- 节点集合恰好为 S01--S08；
- 禁止自环、未知端点和未知边类型；
- 证据必须分别是源、目标原文的精确子串；
- 禁止 `same_namespace`；
- 禁止仅凭文本相似度创建关系；
- 证据不足时不创建边；允许孤立节点和零边图；
- 不得依据图中心性重新选择或重排 FCSR Top-8。

证据必须是对应 JSONL `name`、`description` 或 `body` 字段中的精确子串。证据全文只保存在审计 manifest 中，不重复注入 Agent。Agent 只看到类型化边表与确定性阅读顺序。阅读顺序通过强连通分量压缩后的拓扑排序生成；无依赖或并列节点按 FCSR rank 打破平局。

### 6.4 人工复核

Hard-15 的全部 15 份 hierarchy 和 15 份 graph 在运行前复核。复核者只能查看冻结的 Skill 文本载荷、组织元数据和证据，不得查看 task prompt、GT、oracle、verifier 或运行结果。

人工只能：

- 删除无证据边；
- 修正错误边类型或方向；
- 修正含义明显错误的层次标签；
- 将无法可靠分类的 Skill 移入 `Other`。

人工不得改写冻结的 `name`、`description` 或 `body`，不得增加 Skill、依据任务答案优化结构。所有修改写入 review log，包含修改前后内容和原因。复核完成后计算最终组织文件 SHA-256 并冻结。

## 7. Skill 包生成和注入

每个 `(task_id, method)` 生成一个复合虚拟 Skill：

```text
generated/skill-organization/<run_id>/<task_id>/<method>/
  skills/
    retrieved-skills/
      SKILL.md
  context_manifest.json
```

只有 `skills/` 目录通过 BenchFlow 挂载；`context_manifest.json` 不传给 Agent。

`SKILL.md` 结构固定为：

```markdown
---
name: retrieved-skills
description: Retrieved procedural skills organized for the current task.
---

# Retrieved procedural guidance

统一、任务无关的使用说明。

## Organization

Flat、Hierarchy 或 Graph 索引。

## Atomic skill payloads

S01--S08 的统一包装和冻结 `name + description + body`。
```

组织索引上限为 512 tokens。Flat 使用排名索引；Hierarchy 使用层次目录；Graph 使用边表和阅读顺序。三个条件的组织 token 数允许不同，因为组织开销是 treatment 的一部分，但必须记录绝对值及其占原始 Skill token 的比例。

注入使用 BenchFlow 官方入口：

```text
--skill-mode with-skill --skills-dir <method>/skills
```

`no_skill` 使用：

```text
--skill-mode no-skill
```

不得修改 SkillsBench `task.md`，不得修改 OpenHands/BenchFlow site-packages 来注入上下文。

## 8. 确定性验证闸门

生成完成后、调用任何 Agent 前，验证器必须检查：

1. 冻结 report 的 SHA-256 未变化；
2. 15 个 task ID 唯一，且全部存在于预测和服务器 SkillsBench；
3. 每任务恰好有 8 个冻结 Skill；
4. 每个 Skill ID 在 Hard JSONL 中唯一可定位且 canonical record 指纹匹配；
5. 三种有 Skill 条件的 Skill ID、rank、正文和 payload hash 相同；
6. Hierarchy 覆盖 S01--S08 且无重复；
7. Graph 节点为 S01--S08，全部边通过类型和原文证据验证；
8. 组织器没有接收禁用输入；
9. `SKILL.md` frontmatter 和目录布局通过 Skill/BenchFlow 检查；
10. manifest、组织文件、渲染文件之间的 hash 引用一致。

任一检查失败均停止整个批次，不生成部分实验结果。

## 9. 端到端执行流程

### Phase A：环境与任务预检

1. 激活服务器 `skillsbench` 环境；
2. 记录 Python、BenchFlow、OpenHands、Daytona 和依赖版本；
3. 验证 DeepSeek 凭据，但不记录密钥值；
4. 验证 SkillsBench commit；
5. 对 15 个任务执行结构检查；
6. 用 oracle 验证 task environment 和 verifier。

Oracle 失败属于任务/环境阻塞，Hard-15 pilot 必须暂停并修复；固定的 15 个任务不得静默排除、替换或在看到 Agent 结果后缩小。

### Phase B：冻结上下文

1. 从唯一 report 读取 Hard-15 Top-8；
2. 从 Hard JSONL 唯一提取并指纹化冻结 Skill 记录；
3. 统计上下文长度并应用统一预算规则；
4. 离线生成 hierarchy 和 graph 元数据；
5. 自动验证并任务盲人工复核；
6. 冻结 `experiment_manifest.json`；
7. 渲染 45 个有 Skill 的 `retrieved-skills/SKILL.md`；
8. 运行结构公平性验证器。

### Phase C：两任务 smoke test

选择一个 Top-8 完整覆盖任务和一个不完整覆盖任务，各运行四个条件，共 8 条 trajectory。检查：

- `no_skill` prompt 中没有生成 Skill；
- 三个 Skill 条件确实读取 `retrieved-skills`；
- `prompts.json` 或 trajectory 中可验证注入成功；
- 三个条件的原子 payload hash 相同；
- Agent、模型、推理模式、timeout 和 sandbox 配置相同；
- verifier 输出可解析；
- Daytona 清理警告不被误判为任务失败。

Smoke test 失败时修复管线并使用新的 run ID 重跑；不得把失败 smoke 数据合并进 pilot。

### Phase D：Hard-15 pilot

运行：

```text
15 tasks x 4 conditions x 1 repeat = 60 trajectories
```

Pilot 采用新的 Daytona sandbox。四条件执行顺序应按任务轮换或预先随机化并写入 manifest，避免始终把某个条件放在模型服务较早或较晚阶段。不得因任务失败自动改变提示或 Skill 包；基础设施级可重试必须使用预注册的统一规则。

### Phase E：结果汇总与判定

输出 15 x 4 的任务级矩阵，并分别统计：

- 固定分母 pass rate；
- verifier reward；
- 有效运行 pass rate；
- `0 -> 1`、`1 -> 0`、`1 -> 1`、`0 -> 0` 状态转移；
- input/output/total tokens；
- agent execution time 与 wall time；
- tool calls 与 trajectory steps；
- cost；
- timeout、agent error、verifier error 和 infrastructure error；
- Top-8 完整覆盖与不完整覆盖分层；
- 单 Skill 与多 Skill 任务分层。

Pilot 只报告描述性结果和任务级差异，不进行显著性检验。若 `no_skill` 成功达到 12/15 或以上，标记潜在 ceiling；若只有 0--2/15，标记潜在 floor。两者均不自动否定实验，但会影响 63 任务正式实验的任务难度或模型消融设计。

### Phase F：扩展到 63 个重合任务

只有在 Hard-15 的注入、验证、错误分类和结果汇总全部通过后扩展。正式实验建议：

```text
63 tasks x 4 conditions x 3 repeats = 756 trajectories
```

以任务为配对单位，对 `flat-no_skill`、`hierarchy-flat`、`graph-flat` 的差值做 10,000 次 paired bootstrap，报告绝对百分点、平均 reward 差值和 95% CI。固定分母结果与排除基础设施错误的结果必须同时报告。

## 10. 运行记录

每条 trajectory 至少记录：

```text
run_id, task_id, method, repeat_id,
skillsbench_commit, benchflow_version, agent, model, reasoning_mode, sandbox,
retrieval_report_hash, top8_skill_ids, atomic_payload_hashes,
hierarchy_hash, graph_hash, rendered_context_hash,
organization_tokens, raw_skill_tokens, total_injected_tokens,
status, reward, passed, verifier_errored,
input_tokens, output_tokens, total_tokens, cost_usd,
environment_setup_time_s, agent_execution_time_s, verifier_time_s, wall_time_s,
tool_calls, trajectory_steps, timeout, failure_type,
top8_gt_complete, single_or_multi
```

`top8_gt_complete` 和 `single_or_multi` 仅在评估汇总阶段关联，不进入 Agent prompt 或组织器输入。

## 11. 产物布局

```text
reports/agent/skill-organization/<run_id>/
  experiment_manifest.json
  preprocessing/
    frozen_skill_inventory.jsonl
    organizer_requests/
    organizer_responses/
    hierarchy/
    graph/
    review_log.jsonl
    validation_report.json
  generated/
    <task_id>/<method>/skills/retrieved-skills/SKILL.md
    <task_id>/<method>/context_manifest.json
  jobs/
    <condition>/...
  results/
    trajectories.jsonl
    task_matrix.csv
    aggregate.json
    failures.jsonl
```

密钥、认证文件和完整环境变量不得写入任何产物。

## 12. Hard-15 验收标准

进入 63 任务正式实验前必须满足：

- 15/15 任务存在于固定 SkillsBench v1.1；
- 15/15 oracle 预检通过；
- 冻结 report、Hard JSONL 和 114 条权威 Skill 记录均有稳定指纹；
- 15/15 hierarchy 与 graph 完成自动验证和任务盲人工复核；
- 三个有 Skill 条件的 120 个任务内 Skill 实例逐一通过 payload hash 等价检查；
- 8 条 smoke trajectory 全部产生可解释状态；
- 60 条 pilot trajectory 均有结果或明确错误分类；
- 注入成功能够从 prompt/trajectory 证据确认；
- 结果能够由 run ID、manifest 和冻结输入复放；
- 不存在 Contract、GT、oracle、verifier 或历史结果泄漏。

## 13. 已知限制

- 若离线组织器也使用 DeepSeek-V4-Flash，可能存在同模型表达偏好；需披露并在后续用不同组织器做小规模敏感性分析；
- 三种结构的组织 token 不完全相等，输入开销属于 treatment 的一部分，必须单独报告；
- Graph 允许零边，因此某些任务上的 Graph 可能退化为带阅读顺序的 Flat；
- Hard-15 样本量不足以作稳定统计推断，其作用是验证流程、检查 ceiling/floor 并发现失败模式；
- Top-8 未完整覆盖 GT 的任务无法仅靠组织方式恢复缺失知识，应与完整覆盖任务分层解释。
- SkillRouter JSONL 没有保存全部上游 repo、commit、路径和附件，因而本实验评估的是冻结文本 Skill 的组织效果，而不是完整可安装 Skill 包的组织效果。
