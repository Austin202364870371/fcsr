# RQ1：如何提升大规模 Agent Skill 库的检索准确率？

**状态：已实现数据与训练管线，尚未形成正式对比结论。**

## 问题与假设

在约 8 万条 Skill 的候选池中，关键词或单次向量检索可能遗漏任务所需能力。RQ1 研究：以原文证据约束的 Skill Contract 能否生成可靠训练数据，并通过 Bi-Encoder + Reranker 两阶段检索提升 SkillRouter 兼容评测的检索质量。

假设是 Contract 的 `operations`、`inputs`、`outputs`、`constraints` 和 `quality_criteria` 能限制合成 query 的语义范围；训练时使用这些信息、部署时仍只索引原始 Skill 的 `name + description + body`，可避免运行时依赖全库 Contract。

## 当前资产

- 新一轮数据从 Easy pool 以类别和语言分层抽样 32,000 条，并排除评测 GT / relevance Skill；旧版 8k 数据继续保留用于对照。
- Contract V2 要求每个字段指向原文 evidence，本地校验证据偏移、`source_hash` 与 schema。
- 已得到 7,995 条有效 Contract 和 7,342 条单 Skill query；版本见 `data/contracts/manifest.json`、`data/synthetic/single_v1/manifest.json`。
- 已实现本地/语义负例、Qwen + LoRA Bi-Encoder、Listwise Reranker，以及 Easy / Hard pool 计分。

这些是工程和数据资产，并不等同于性能结论。结论须来自冻结配置下的对照实验。

## 多 Skill 扩展

`data/synthetic/multiskill_v1/` 已保存 497 个有向 pair 与 51 个 triple。候选只在有效 Contract、有效单 Skill query、非 benchmark、artifact 输出到必需输入交接和操作互补同时成立时保留；22,635 个拒绝项保留原因。LLM 生成器只可把已验证候选写成任务、子任务和依赖，并以 JSON、Skill ID 顺序、source hash 与 DAG 校验拒绝越界输出。新的 Contract 和查询生成统一使用关闭思考模式的 `deepseek-v4-flash`；Contract、单 Skill query 和多 Skill candidate 均保持一条输入一个请求，并发上限为 16。请求池持续补位，失败只重试当前条目。

历史本地 Qwen3-8B 运行中，548 个候选有 541 条通过严格结构校验，7 条失败，448 条进入复核队列。新版训练改用 `data/training/multiskill_weighted/`，不再复制多 Skill query。每条组合任务在 Reranker 中只出现一次；在 Bi-Encoder 中按每个不同正例 Skill 各展开一次，并保留完整 `positive_skill_ids`，所有正例均从负例中排除。默认多 Skill 损失权重为 Bi-Encoder `1.5`、Reranker `3.0`，每个 epoch 对全部类型整体打乱。

## 数据比例与训练策略

旧版实际数据是 7,342 条单 Skill query 和 541 条多 Skill query，即 query 层约 93.1% : 6.9%。541 条组合任务按不同正例自然展开为 1,133 条 Bi-Encoder 记录，不复制时 Bi-Encoder 的原始多 Skill 占比为 13.4%，Reranker 为 6.9%。使用默认权重后，其有效加权梯度占比分别约为 18.8% 和 18.1%。如果 32k 数据保持相近产出率，可先按约 30k–31.5k 单 Skill query 与 2.2k–2.4k 多 Skill query 规划；这只是容量估算，正式比例必须由新 manifest 计算。

现有文献没有给出适用于 FCSR 的固定“最佳单/多 Skill 比例”。更常见的做法是把任务比例视作训练策略：[Dynamic Sampling Strategies for Multi-Task Reading Comprehension](https://aclanthology.org/2020.acl-main.86/) 根据当前任务表现动态调整采样，并报告交错任务实例有助于缓解遗忘；[Learning Task Sampling Policy for Multitask Learning](https://aclanthology.org/2021.findings-emnlp.375/) 学习任务采样策略而非固定均匀采样。[Multi-Task Retrieval for Knowledge-Intensive Tasks](https://aclanthology.org/2021.acl-long.89/) 支持共享检索器的多任务训练，但 [Improving Multitask Retrieval by Promoting Task Specialization](https://aclanthology.org/2023.tacl-1.68/) 表明朴素多任务模型可能落后于任务专用模型，需要 task prompt 或自适应学习。基于这些结果，当前方案选择可审计、易消融的静态类型权重，而不是磁盘复制；`1.5/3.0` 是本项目根据两阶段原始占比做的工程推断，不是论文直接给出的常数。

## 对照与判据

比较 BM25、未微调 Dense、固定 RRF Hybrid、FCSR Bi-Encoder、FCSR Bi-Encoder + Reranker，以及加入审计多 Skill 数据后的同预算模型。固定任务、Skill pool、截断、Top-K、底座模型和种子。

报告 `Recall@K`、`nDCG@K`、MRR、FullCoverage，以及 overall / single-skill / multi-skill 分组。多 Skill 主判据为 Hard pool 的 multi-skill Recall@20 与 FullCoverage 是否提升，并同时检查单 Skill 指标。

## 下一步

1. 完成 prompt006 的 32k Contract、单 Skill query 与多 Skill query 正式生成，并以 manifest 固化实际比例。
2. 用 `scripts/build_multiskill_training_data.py` 构建单遍 weighted 数据；不生成任何 repeat/replica 记录。
3. 在相同 epoch、随机种子、底座模型和候选池下比较单 Skill、`1.0/1.0`、`1.5/3.0` 与更强权重设置。
4. 对复核队列进行抽样审计，报告结构通过率之外的组合语义质量。
5. 将任务级结果和冻结配置写入 `reports/` 后再给出结论。

## 入口

`scripts/build_single_skill_data.py`、`scripts/build_multiskill_candidates.py`、`scripts/generate_multiskill_queries.py`、`scripts/build_multiskill_training_data.py`、`scripts/train_biencoder.py`、`scripts/train_reranker.py`、`scripts/evaluate.py`、`scripts/render_evaluation_tables.py`、`configs/base.yaml`。
