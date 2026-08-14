# RQ1：如何提升大规模 Agent Skill 库的检索准确率？

**状态：数据与训练管线已实现；当前 Contract 已完成，query、负例和训练数据待重新生成，尚无正式对比结论。**

## 问题与假设

在约 8 万条 Skill 的候选池中，关键词或单次向量检索可能遗漏任务所需能力。RQ1 研究：以原文证据约束的 Skill Contract 能否生成可靠训练数据，并通过 Bi-Encoder + Reranker 两阶段检索提升 SkillRouter 兼容评测的检索质量。

假设是 Contract 的 `operations`、`inputs`、`outputs`、`constraints` 和 `quality_criteria` 能限制合成 query 的语义范围；训练时使用这些信息、部署时仍只索引原始 Skill 的 `name + description + body`，可避免运行时依赖全库 Contract。

## 当前资产

- 从 Easy pool 按类别和语言分层抽样 32,000 条，排除了评测 GT / relevance Skill。抽样文件与 manifest 直接位于 `data/samples/`，该目录不再套子目录。
- Contract V2 每个字段都指向原文 evidence，本地校验 schema、证据偏移和 `source_hash`。
- `deepseek-v4-flash` 正式作业生成并通过本地复验的 Contract 为 31,977 条；23 条唯一 Skill 失败。重试作业未恢复这些条目，因此当前接受该缺口，详见 `data/contracts/manifest.json`。
- 旧版 8k 大文件已从服务器删除，本地备份仍由用户保留。历史统计（7,995 条 Contract、7,342 条单 Skill query）仅用于比例参考，不再是当前训练输入。
- 当前单 Skill query、多 Skill query、负例和 mixed 训练数据尚未重新生成；对应 manifest 明确标记为 awaiting 状态。

这些是工程和数据资产，并不等同于性能结论。结论须来自冻结配置下的对照实验。

## 单 Skill 负例与 FN 过滤

单 Skill query 由 DeepSeek 以一条输入一个请求生成，固定使用 `deepseek-v4-flash`、关闭思考、启用 JSON Output，并保持 16 路并发。

局部负例合并 BM25、同类别和随机候选，先排除相同 Skill ID、规范化名称相同、正文完全相同或正文 trigram Jaccard `>= 0.85` 的潜在假负例。语义阶段使用本地 Qwen3 Embedding 检索 Top-50，随后删除与正例 Skill 余弦相似度 `>= 0.95` 的候选。删除项单独写入 `data/synthetic/single_skill/semantic_fn_review.jsonl.gz`，过滤后的数据用于 Bi-Encoder 和 Reranker。

## 多 Skill 扩展

候选仅在有效 Contract、有效单 Skill query、非 benchmark、artifact 输出到必需输入交接和操作互补同时成立时保留。LLM 只能把已验证候选改写为任务、子任务和依赖；生成器按 JSON、Skill ID 顺序、source hash、子任务覆盖与 DAG 拒绝越界输出。

旧版 8k 运行曾得到 497 个有向 pair、51 个 triple，并有 541/548 条 query 通过严格结构校验；这些数字只作为历史产出率参考。当前 `data/synthetic/multi_skill/` 尚未生成候选和 query，应在当前单 Skill query 完成后重新构建。

mixed 数据不复制多 Skill query。每条组合任务在 Reranker 中只出现一次；在 Bi-Encoder 中按每个不同正例 Skill 各展开一次，并保留完整 `positive_skill_ids`。所有正例都从负例中排除。多 Skill 语义挖掘使用本地 Qwen Top-64，候选若与任一正例的余弦相似度 `>= 0.95` 即被过滤，并写入 `data/training/mixed/semantic_fn_review.jsonl.gz`。默认多 Skill损失权重为 Bi-Encoder `1.5`、Reranker `3.0`，每个 epoch 对全部类型整体打乱。

## 数据比例与训练策略

旧版历史数据为 7,342 条单 Skill query 和 541 条多 Skill query，即 query 层约 93.1% : 6.9%。541 条组合任务按不同正例自然展开为 1,133 条 Bi-Encoder 记录；不复制时，Bi-Encoder 的原始多 Skill 占比为 13.4%，Reranker 为 6.9%。默认权重下，有效加权梯度占比分别约为 18.8% 和 18.1%。这些只用于容量和消融规划，当前正式比例必须由新 manifest 计算。

现有文献没有给出适用于 FCSR 的固定“最佳单/多 Skill 比例”。更常见的做法是把任务比例视作训练策略：[Dynamic Sampling Strategies for Multi-Task Reading Comprehension](https://aclanthology.org/2020.acl-main.86/) 根据当前任务表现动态调整采样，并报告交错任务实例有助于缓解遗忘；[Learning Task Sampling Policy for Multitask Learning](https://aclanthology.org/2021.findings-emnlp.375/) 学习任务采样策略而非固定均匀采样。[Multi-Task Retrieval for Knowledge-Intensive Tasks](https://aclanthology.org/2021.acl-long.89/) 支持共享检索器的多任务训练，但 [Improving Multitask Retrieval by Promoting Task Specialization](https://aclanthology.org/2023.tacl-1.68/) 表明朴素多任务模型可能落后于任务专用模型，需要 task prompt 或自适应学习。当前方案因此采用可审计、易消融的静态类型权重，而不是磁盘复制；`1.5/3.0` 是工程起点，不是论文常数。

## 对照与判据

比较 BM25、未微调 Dense、固定 RRF Hybrid、FCSR Bi-Encoder、FCSR Bi-Encoder + Reranker，以及加入审计多 Skill 数据后的同预算模型。固定任务、Skill pool、截断、Top-K、底座模型和种子。

报告 `Recall@K`、`nDCG@K`、MRR、FullCoverage，以及 overall / single-skill / multi-skill 分组。多 Skill 主判据为 Hard pool 的 multi-skill Recall@20 与 FullCoverage 是否提升，并同时检查单 Skill 指标。

## 下一步

1. 用 31,977 条有效 Contract 生成当前单 Skill query，并更新 `data/synthetic/single_skill/manifest.json`。
2. 挖掘局部与语义负例，执行两层 FN 过滤，审计 `semantic_fn_review.jsonl.gz`，再构建单 Skill Reranker group。
3. 基于当前 Contract 与单 Skill query 重新构建多 Skill pair/triple 候选。
4. 先运行 50 条多 Skill DeepSeek pilot，审计通过后再以 16 路并发执行全量生成。
5. 挖掘多 Skill负例、对全部正例做 FN 过滤，并构建 `data/training/mixed/`。
6. 在相同 epoch、随机种子、底座模型和候选池下比较单 Skill、`1.0/1.0`、`1.5/3.0` 与更强权重设置。
7. 将任务级结果和冻结配置写入 `reports/` 后再给出结论。

## 入口

`scripts/build_single_skill_data.py`、`scripts/build_multiskill_candidates.py`、`scripts/generate_multiskill_queries.py`、`scripts/build_multiskill_training_data.py`、`scripts/train_biencoder.py`、`scripts/train_reranker.py`、`scripts/evaluate.py`、`scripts/render_evaluation_tables.py`、`configs/base.yaml`。
