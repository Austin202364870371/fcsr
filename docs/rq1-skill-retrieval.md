# RQ1：如何提升大规模 Agent Skill 库的检索准确率？

**状态：数据构建、训练与冻结 Hard-pool 评测均已完成；FCSR 已确定为正式主模型。**

## 问题与假设

在约 8 万条 Skill 的候选池中，关键词或单次向量检索可能遗漏任务所需能力。RQ1 研究：以原文证据约束的 Skill Contract 能否生成可靠训练数据，并通过 Bi-Encoder + Reranker 两阶段检索提升 SkillRouter 兼容评测的检索质量。

假设是 Contract 的 `operations`、`inputs`、`outputs`、`constraints` 和 `quality_criteria` 能限制合成 query 的语义范围；训练时使用这些信息、部署时仍只索引原始 Skill 的 `name + description + body`，可避免运行时依赖全库 Contract。

## 当前资产

- 从 Easy pool 按类别和语言分层抽样 32,000 条，排除了评测 GT / relevance Skill。抽样文件与 manifest 直接位于 `data/samples/`，该目录不再套子目录。
- Contract V2 每个字段都指向原文 evidence，本地校验 schema、证据偏移和 `source_hash`。
- `deepseek-v4-flash` 正式作业生成并通过本地复验的 Contract 为 31,977 条；23 条唯一 Skill 失败。重试作业未恢复这些条目，因此当前接受该缺口，详见 `data/contracts/manifest.json`。
- 旧版 8k 训练大文件已从服务器删除，本地备份仍由用户保留；对应 checkpoint 作为数据规模消融模型 FCSR-Small 保留。
- 当前正式数据包含 31,902 条单 Skill query 和 2,481 条多 Skill query；混合训练集包含 37,327 条 Bi-Encoder 记录和 34,383 个 Reranker group。
- Retriever 与 Reranker 均基于 Qwen3 0.6B 进行 LoRA 训练；正式组件位于 `checkpoints/fcsr/{retriever,reranker}`，小数据规模消融位于 `checkpoints/fcsr-small/{retriever,reranker}`。
- 所有正式 Hard-pool 评测覆盖 75 条任务和 79,141 条候选 Skill，零缺失预测、零无 GT 任务。

这些是工程和数据资产，并不等同于性能结论。结论须来自冻结配置下的对照实验。

## 单 Skill 负例与 FN 过滤

单 Skill query 由 DeepSeek 以一条输入一个请求生成，固定使用 `deepseek-v4-flash`、关闭思考、启用 JSON Output，并保持 16 路并发。

局部负例合并 BM25、同类别和随机候选，先排除相同 Skill ID、规范化名称相同、正文完全相同或正文 trigram Jaccard `>= 0.85` 的潜在假负例。语义阶段使用本地 Qwen3 Embedding 检索 Top-50，随后删除与正例 Skill 余弦相似度 `>= 0.95` 的候选。删除项单独写入 `data/synthetic/single_skill/semantic_fn_review.jsonl.gz`，过滤后的数据用于 Bi-Encoder 和 Reranker。

## 多 Skill 扩展

候选仅在有效 Contract、有效单 Skill query、非 benchmark、artifact 输出到必需输入交接和操作互补同时成立时保留。LLM 只能把已验证候选改写为任务、子任务和依赖；生成器按 JSON、Skill ID 顺序、source hash、子任务覆盖与 DAG 拒绝越界输出。

正式候选生成得到 2,488 个组合候选，其中 2,481 条 query 通过校验，7 条失败按既定容忍策略忽略。多 Skill 数据包含 2,025 个 pair 和 463 个 triple 候选，并保留完整失败与审核记录。

mixed 数据不复制多 Skill query。每条组合任务在 Reranker 中只出现一次；在 Bi-Encoder 中按每个不同正例 Skill 各展开一次，并保留完整 `positive_skill_ids`。所有正例都从负例中排除。多 Skill 语义挖掘使用本地 Qwen Top-64；语义、BM25、同类别和随机四类候选在入选前都逐一与全部正例比较，若与任一正例的余弦相似度 `>= 0.95` 即被过滤，并写入 `data/training/semantic_fn_review.jsonl.gz`。过滤后继续扫描同一来源以尽量补满 `4/3/2/1` 配额。默认多 Skill 损失权重为 Bi-Encoder `1.5`、Reranker `3.0`，每个 epoch 对全部类型整体打乱。

## 数据比例与训练策略

正式数据中，多 Skill 占 Bi-Encoder 原始记录的 14.5%，在 `1.5` 损失权重下占有效加权质量约 20.3%；多 Skill 占 Reranker group 的 7.2%，在 `3.0` 权重下占有效加权质量约 18.9%。数据不做磁盘复制，每个 epoch 对全部类型整体打乱。

现有文献没有给出适用于 FCSR 的固定“最佳单/多 Skill 比例”。更常见的做法是把任务比例视作训练策略：[Dynamic Sampling Strategies for Multi-Task Reading Comprehension](https://aclanthology.org/2020.acl-main.86/) 根据当前任务表现动态调整采样，并报告交错任务实例有助于缓解遗忘；[Learning Task Sampling Policy for Multitask Learning](https://aclanthology.org/2021.findings-emnlp.375/) 学习任务采样策略而非固定均匀采样。[Multi-Task Retrieval for Knowledge-Intensive Tasks](https://aclanthology.org/2021.acl-long.89/) 支持共享检索器的多任务训练，但 [Improving Multitask Retrieval by Promoting Task Specialization](https://aclanthology.org/2023.tacl-1.68/) 表明朴素多任务模型可能落后于任务专用模型，需要 task prompt 或自适应学习。当前方案因此采用可审计、易消融的静态类型权重，而不是磁盘复制；`1.5/3.0` 是工程起点，不是论文常数。

## 对照与判据

比较 BM25、未微调 Dense、固定 RRF Hybrid、FCSR Bi-Encoder、FCSR Bi-Encoder + Reranker，以及加入审计多 Skill 数据后的同预算模型。固定任务、Skill pool、截断、Top-K、底座模型和种子。

报告 `Recall@K`、`nDCG@K`、MRR、FullCoverage，以及 overall / single-skill / multi-skill 分组。多 Skill 主判据为 Hard pool 的 multi-skill Recall@20 与 FullCoverage 是否提升，并同时检查单 Skill 指标。

## 正式 Hard-pool 结果

评测只使用冻结的 Hard pool，不再维护 Easy-pool report。协议固定为 Dense/RRF
Top-50、RRF fusion depth 100、`rrf_k=60`，对第一阶段 Top-20 候选执行
Reranker，并将最终 Top-10 Skills 提供给任务。

正式系统命名如下：

- `FCSR`：完整训练语料得到的主系统，使用
  `checkpoints/fcsr/{retriever,reranker}`。
- `FCSR-Small`：早期小数据规模消融，使用
  `checkpoints/fcsr-small/{retriever,reranker}`。

两套系统都采用 `BM25 + FCSR Retriever -> RRF -> FCSR Reranker` 架构。

| 系统 | Hit@1 | MRR@10 | nDCG@10 | Recall@10 | Recall@20 | FullCoverage@10 | Multi FullCoverage@10 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ours: FCSR-Small | 0.6267 | 0.7189 | 0.5970 | 0.6642 | 0.6930 | **0.4933** | **0.3529** |
| Ours: FCSR | **0.6667** | **0.7414** | **0.6127** | **0.6741** | **0.7168** | 0.4533 | 0.2941 |

FCSR 在 Hit@1、MRR@10、nDCG@10、Recall@10 和 Recall@20 上全面优于
FCSR-Small，因此作为正式主模型。FCSR-Small 保留用于数据规模消融和 Coverage
对照。基础 BM25、Dense、RRF、Base Reranker 与 SkillRouter 报告保留在
`reports/baselines/hard/`；两个自有系统的规范报告位于
`reports/systems/{fcsr,fcsr-small}/hard/`。标准表格为
`reports/tables/hard-retrieval.md`、`hard-final.md` 和
`hard-two-stage.md`。

## 结论与维护范围

当前 RQ1 模型阶段已收敛，不再继续追加训练实验。维护范围只包括 FCSR、
FCSR-Small、Hard-pool baseline 以及相应报告；历史 single-skill checkpoint、
所有 Easy report、旧交叉搭配报告和实验性命名均已移除。

## 入口

`scripts/build_single_skill_data.py`、`scripts/build_multiskill_candidates.py`、`scripts/generate_multiskill_queries.py`、`scripts/build_multiskill_training_data.py`、`scripts/train_biencoder.py`、`scripts/train_reranker.py`、`scripts/evaluate.py`、`scripts/render_evaluation_tables.py`、`configs/fcsr.yaml`。
