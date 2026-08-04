# RQ1：如何提升大规模 Agent Skill 库的检索准确率？

**状态：已实现数据与训练管线，尚未形成正式对比结论。**

## 问题与假设

在约 8 万条 Skill 的候选池中，关键词或单次向量检索可能遗漏任务所需能力。RQ1 研究：以原文证据约束的 Skill Contract 能否生成可靠训练数据，并通过 Bi-Encoder + Reranker 两阶段检索提升 SkillRouter 兼容评测的检索质量。

假设是 Contract 的 `operations`、`inputs`、`outputs`、`constraints` 和 `quality_criteria` 能限制合成 query 的语义范围；训练时使用这些信息、部署时仍只索引原始 Skill 的 `name + description + body`，可避免运行时依赖全库 Contract。

## 当前资产

- 从 Easy pool 以类别和语言分层抽样 8,000 条，并排除评测 GT / relevance Skill。
- Contract V2 要求每个字段指向原文 evidence，本地校验证据偏移、`source_hash` 与 schema。
- 已得到 7,995 条有效 Contract 和 7,342 条单 Skill query；版本见 `data/contracts/manifest.json`、`data/synthetic/single_v1/manifest.json`。
- 已实现本地/语义负例、Qwen + LoRA Bi-Encoder、Listwise Reranker，以及 Easy / Hard pool 计分。

这些是工程和数据资产，并不等同于性能结论。结论须来自冻结配置下的对照实验。

## 多 Skill 扩展

`data/synthetic/compositional_v1/` 已保存 497 个有向 pair 与 51 个 triple。候选只在有效 Contract、有效单 Skill query、非 benchmark、artifact 输出到必需输入交接和操作互补同时成立时保留；22,635 个拒绝项保留原因。本地 Qwen 生成器只可把已验证候选写成任务、子任务和依赖，并以 JSON、Skill ID 顺序、source hash 与 DAG 校验拒绝越界输出。

已在服务器以本地 Qwen3-8B 完成一次全量生成：548 个候选中 541 条通过严格校验，7 条失败，448 条进入复核队列。训练阶段采用 `data/training/rq1-mixed-3x/`：保留 7,342 条单 Skill 数据，并按原始组合任务组做 3 倍确定性采样。每条组合任务会分别展开到其每个正例 Skill 的 Bi-Encoder 样本，同时在同一条记录中保留完整 `positive_skill_ids`；所有正例均从负例候选中排除。Reranker 保持一条组合任务一个多标签 group。

## 对照与判据

比较 BM25、未微调 Dense、固定 RRF Hybrid、FCSR Bi-Encoder、FCSR Bi-Encoder + Reranker，以及加入审计多 Skill 数据后的同预算模型。固定任务、Skill pool、截断、Top-K、底座模型和种子。

报告 `Recall@K`、`nDCG@K`、MRR、FullCoverage，以及 overall / single-skill / multi-skill 分组。多 Skill 主判据为 Hard pool 的 multi-skill Recall@20 与 FullCoverage 是否提升，并同时检查单 Skill 指标。

## 下一步

1. 用 `scripts/build_rq1_mixed_training.py` 构建可训练的混合数据；541 条组合任务在 3 倍采样下产生 3,399 条多 Skill Bi-Encoder 样本与 1,623 个多标签 Reranker group。
2. 以相同训练预算完成单 Skill 基线与混合训练消融，并冻结随机种子、底座模型和候选池。
3. 对 448 条复核队列进行抽样审计，报告结构通过率之外的组合语义质量。
4. 将任务级结果和冻结配置写入 `reports/` 后再给出结论。

## 入口

`scripts/preprocess.py`、`scripts/build_compositional_candidates.py`、`scripts/generate_compositional_queries.py`、`scripts/build_rq1_mixed_training.py`、`scripts/train_biencoder.py`、`scripts/train_reranker.py`、`scripts/evaluate.py`、`configs/base.yaml`。
