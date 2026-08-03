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

`data/synthetic/compositional_v1/` 已保存 497 个有向 pair 与 51 个 triple。候选只在有效 Contract、有效单 Skill query、非 benchmark、artifact 输出到必需输入交接和操作互补同时成立时保留；22,635 个拒绝项保留原因。本地 Qwen 生成器已实现：它只可把已验证候选写成任务、子任务和依赖，并以 JSON、Skill ID 顺序、source hash 与 DAG 校验拒绝越界输出。数据尚未在服务器生成。

## 对照与判据

比较 BM25、未微调 Dense、固定 RRF Hybrid、FCSR Bi-Encoder、FCSR Bi-Encoder + Reranker，以及加入审计多 Skill 数据后的同预算模型。固定任务、Skill pool、截断、Top-K、底座模型和种子。

报告 `Recall@K`、`nDCG@K`、MRR、FullCoverage，以及 overall / single-skill / multi-skill 分组。多 Skill 主判据为 Hard pool 的 multi-skill Recall@20 与 FullCoverage 是否提升，并同时检查单 Skill 指标。

## 下一步

1. 从候选生成严格校验的 `compositional_queries.jsonl.gz`。
2. 按 query 目标数采样组合，不以修复所有 Contract 失败项为前提。
3. 以相同预算进行单 Skill 与混合训练消融。
4. 将任务级结果和冻结配置写入 `reports/` 后再给出结论。

## 入口

`scripts/preprocess.py`、`scripts/build_compositional_candidates.py`、`scripts/train_biencoder.py`、`scripts/train_reranker.py`、`scripts/evaluate.py`、`configs/base.yaml`。
