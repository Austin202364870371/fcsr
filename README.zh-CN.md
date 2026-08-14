# FCSR

[English README](README.md)

FCSR（Function-aware Coverage Skill Retriever）是面向大规模 Agent Skill 检索的可复现实验管线。它从原始 Skill 中抽取带证据的 Contract，构建单 Skill 与多 Skill 合成训练数据，使用 Qwen 微调 Bi-Encoder 和 Reranker，并以兼容 SkillRouter 的协议评测单阶段或两阶段系统。

Contract 只用于构造可审计训练数据。线上检索仍只使用原始 Skill 的 `name + description + body`。

## 包含内容

| 模块 | 作用 |
|---|---|
| Contract 抽取 | 从 benchmark-safe 样本中抽取带原文证据的 Contract。 |
| 单 Skill 数据 | 生成查询、局部负例和语义负例。 |
| 多 Skill 数据 | 先用 Contract 校验 Skill 对/三元组，再让 LLM 只改写已验证组合。 |
| 训练 | 用 LoRA 微调 Qwen3 Embedding 与 Qwen3 Reranker。 |
| 评测 | 支持 Dense、BM25、RRF、重排、打分和标准化结果表。 |

## 目录结构

```text
configs/                         通用模型和数据默认配置
data/
  samples/                       扁平存放抽样文件及其 manifest
  contracts/                     Contract、失败记录及 manifest
  raw/                           Skill 池和公开评测任务
  pilots/                        与正式数据隔离的 pilot 产物
  synthetic/
    single_skill/                单 Skill 查询、负例及 manifest
    multi_skill/                 候选、查询及 manifest
  training/
    mixed/                       单遍、按类型加权的混合训练数据
docs/                            研究问题说明与参考文献
jobs/                            Slurm 任务模板
scripts/
  build_single_skill_data.py     Contract 与单 Skill 数据管线
  build_multiskill_candidates.py Contract 约束的 pair/triple 构建
  generate_multiskill_queries.py LLM 查询生成与严格校验
  build_multiskill_training_data.py
  train_biencoder.py
  train_reranker.py
  evaluate.py
  render_evaluation_tables.py
src/                             可复用实现模块
tests/                           离线回归测试
```

`data/samples/` 保持扁平：`sample_skills.jsonl.gz` 和 `manifest.json` 直接放在
该目录下，不再增加一级子目录。Git 跟踪所有 `data/**/manifest.json`，大数据、模型、
checkpoint、日志、缓存和生成报告则继续忽略。

## 环境

需要 Python 3.10 或更新版本。以下命令在仓库根目录运行。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item .env.example .env
# 编辑 .env 并填写 DEEPSEEK_API_KEY。
$env:PYTHONPATH = "src"
python -B -m unittest discover -s tests -v
```

在 CUDA 机器上，必要时先根据 [PyTorch 安装说明](https://pytorch.org/get-started/locally/) 安装与宿主环境匹配的 PyTorch，再安装 `requirements.txt`。

Contract 抽取和 LLM 查询生成使用 `deepseek-v4-flash`，关闭思考模式并启用
JSON Output。复制 `.env.example` 为 `.env`，填写 `DEEPSEEK_API_KEY`，不要提交
该文件。Contract 抽取、单 Skill query 和多 Skill query 都严格保持“一条输入对应
一个 API 请求”，同时最多运行 16 个独立请求。请求池会持续补位；失败只重试当前
条目，不会重跑已经成功的条目。

## 数据构建

### 1. 单 Skill 数据

```powershell
# 构建 benchmark-safe 的 32k Contract 样本，再抽取带证据的 Contract。
python -B scripts/build_single_skill_data.py sample `
  --sample-size 32000 --output-dir data/samples --overwrite
python -B scripts/build_single_skill_data.py contracts `
  --model deepseek-v4-flash --concurrency 16 `
  --sample data/samples/sample_skills.jsonl.gz `
  --output data/contracts/contracts.jsonl.gz `
  --failures data/contracts/failures.jsonl.gz

# 生成查询、局部负例，再在 GPU 上挖掘语义负例。
python -B scripts/build_single_skill_data.py queries
python -B scripts/build_single_skill_data.py local-negatives --overwrite
python -B scripts/build_single_skill_data.py semantic-negatives `
  --model models/Qwen3-Embedding-0.6B --device cuda --overwrite
```

输出位于 `data/synthetic/single_skill/`。局部负例合并 BM25、同类别与随机候选，
先按 Skill ID、规范化名称、正文完全相同和正文 trigram Jaccard `>= 0.85` 过滤
潜在假负例；语义阶段使用本地 Qwen3 Embedding 检索 Top-50，再删除与正例 Skill
余弦相似度 `>= 0.95` 的候选。被删除项写入 `semantic_fn_review.jsonl.gz` 供审计，
过滤后的记录再转换为可直接训练的 Bi-Encoder 与 Reranker 数据。

### 2. 多 Skill 数据

候选构建阶段不调用 LLM。只有当有效 Contract、非 benchmark Skill、artifact 交接和操作互补共同成立时，才保留有序 pair 或 triple。

```powershell
python -B scripts/build_multiskill_candidates.py

# DeepSeek 只为已验证候选写自然语言任务。
python -B scripts/generate_multiskill_queries.py `
  --model deepseek-v4-flash --concurrency 16 --max-attempts 3 --progress
```

生成器会校验 JSON 结构、正例 Skill ID 及顺序、source hash、子任务覆盖和依赖 DAG。结果写入 `data/synthetic/multi_skill/` 下的 `queries.jsonl.gz`、`query_failures.jsonl.gz` 与 `review_queue.jsonl.gz`。

### 3. 构建混合训练集

`mixed` 中每个原始 query/group 只保存一次，不再做三倍复制。
pair/triple 在 Bi-Encoder 中仍按不同正例 Skill 各展开一条，这是完整正例监督，
不是重复采样；Reranker 仍是一条 query 对应一个多标签 group。负例只从 Easy pool
挖掘，并排除全部正例 Skill ID。语义 Top-64 候选还会逐一与每个正例 Skill 比较，
按 `0.95` 阈值过滤，并把删除项写入
`data/training/mixed/semantic_fn_review.jsonl.gz`。每个 epoch 会把单、多 Skill
样本整体打乱交错训练。

默认对多 Skill Bi-Encoder 样本使用 `1.5` 损失权重，对多 Skill Reranker group
使用 `3.0` 权重。按预计原始比例，两阶段的有效多 Skill 梯度占比约为 18%–21%。
这两个权重是首轮工程设定，应进行消融，并不是文献给出的固定常数。

```bash
python -B scripts/build_multiskill_training_data.py \
  --negative-model models/Qwen3-Embedding-0.6B \
  --biencoder-multi-loss-weight 1.5 \
  --reranker-multi-loss-weight 3.0 \
  --output-dir data/training/mixed
```

## 模型训练

以下命令使用本地 Hugging Face 模型目录，并用清晰名称保存 LoRA adapter。

```bash
python -B scripts/train_biencoder.py \
  --train-data data/training/mixed/biencoder.jsonl.gz \
  --skills data/raw/skills_easy.jsonl.gz \
  --model models/Qwen3-Embedding-0.6B \
  --output-dir checkpoints/fcsr-emb-0.6b-multiskill-weighted

python -B scripts/train_reranker.py train \
  --groups data/training/mixed/reranker.jsonl.gz \
  --skills data/raw/skills_easy.jsonl.gz \
  --model models/Qwen3-Reranker-0.6B \
  --output-dir checkpoints/fcsr-rank-0.6b-multiskill-weighted
```

单 Skill 基线不需要构建混合集，但需要显式传入
`data/synthetic/single_skill/train_biencoder.jsonl.gz` 或
`data/synthetic/single_skill/train_reranker.jsonl.gz`，checkpoint 建议命名为
`fcsr-emb-0.6b-single` 和 `fcsr-rank-0.6b-single`。

## 评测

`evaluate.py` 将每个阶段拆开，保证每个实验都有独立、可追溯的产物。

```bash
# 第一阶段：用 RRF 获得高召回候选。
python -B scripts/evaluate.py hybrid \
  --queries data/raw/evaluation_queries.jsonl.gz \
  --skills data/raw/skills_hard.jsonl.gz \
  --model checkpoints/fcsr-emb-0.6b \
  --top-k 50 --fusion-depth 100 --rrf-k 60 \
  --output-predictions reports/retrieval/hard/hybrid/predictions.json \
  --output-records reports/retrieval/hard/hybrid/records.jsonl

# 第二阶段：重排 Top-50 候选，取 Top-10 打分。
python -B scripts/evaluate.py rerank \
  --retrieval-records reports/retrieval/hard/hybrid/records.jsonl \
  --skills data/raw/skills_hard.jsonl.gz \
  --model checkpoints/fcsr-rank-0.6b-multiskill-weighted \
  --top-k 10 \
  --output-predictions reports/reranker/hard/rrf-base-emb-multiskill-weighted/predictions.json \
  --output-records reports/reranker/hard/rrf-base-emb-multiskill-weighted/records.jsonl

python -B scripts/evaluate.py score \
  --tasks data/raw/evaluation_queries.jsonl.gz \
  --skills data/raw/skills_hard.jsonl.gz \
  --predictions reports/reranker/hard/rrf-base-emb-multiskill-weighted/predictions.json \
  --stage reranker --tier hard \
  --output-dir reports/reranker/hard/rrf-base-emb-multiskill-weighted

# 输出最终系统表和两阶段消融表。
python -B scripts/render_evaluation_tables.py
```

生成的表格：

- `reports/tables/hard-baselines.md`：每个系统只保留最终输出。
- `reports/tables/hard-two-stage-ablation.md`：仅比较两阶段系统的 retrieval 与 rerank。

最终表中的加粗仅表示数值最大，不表示统计显著性。

## 32k 数据比例与文献依据

旧版 8k 数据包含 7,342 条单 Skill query 和 541 条多 Skill query，原始比例约为
93.1% : 6.9%。如果新一轮候选产出率接近，32k 规模可先按约 30k–31.5k 条单
Skill query、2.2k–2.4k 条多 Skill query 规划；最终必须以新 manifest 的实际数量
为准。由于 pair/triple 会在 Bi-Encoder 侧按正例展开，Bi-Encoder 的原始多 Skill
记录占比预计约 13%–15%，不能给两个模型机械地使用同一个采样倍数。

多任务文献通常把任务比例作为采样或优化策略：动态/学习式任务采样可以优于均匀
采样，交错任务有助于缓解遗忘；检索研究也指出朴素多任务混合不一定优于任务专门化
模型。因此本项目采用“不复制数据、按 epoch 整体打乱、分模型设置类型损失权重”的
可审计方案。参考：
[Dynamic Sampling Strategies](https://aclanthology.org/2020.acl-main.86/)、
[Learning Task Sampling Policy](https://aclanthology.org/2021.findings-emnlp.375/)、
[Multi-Task Retrieval](https://aclanthology.org/2021.acl-long.89/) 和
[Promoting Task Specialization](https://aclanthology.org/2023.tacl-1.68/)。

## Slurm API 生成

API 生成是纯 CPU 任务，但仍须通过 Slurm 运行。Pilot 作业会在需要时创建确定性的
32k 分层样本，但只抽取前 32 条并写入独立目录：

```bash
sbatch jobs/extract_contracts_deepseek_pilot.sbatch
```

审计 pilot 后，再提交独立的 32k 正式作业：

```bash
sbatch jobs/extract_contracts_deepseek.sbatch
```

每个请求只包含一条 Skill；`CONCURRENCY=16` 只控制同时在途的独立请求数。
每条完成后会单独校验并追加保存。正式输出按 `(skill_id, source_hash)` 跳过
已完成记录，因此中断后可以安全续跑。Prompt 007 最多读取 20,000 个 body 字符，
要求按重要性排序，并限制
operations/constraints 最多 12 条、outputs 最多 10 条、其余集合最多 8 条；程序端
应用相同上限，并删除引用相同证据的 constraint/exclusion 重复项。全部集合项总数
最多 32 条；程序还会删除仅由 Skill 触发语句支持的 input/precondition，并且只保留
证据本身含明确否定范围措辞或直接位于排除标题下的 exclusion。可配置 feature flag
不再作为 exclusion；用户请求和 Skill 自己执行的工作流步骤不再作为 precondition；
近重复的 constraint/quality criterion 会被合并。Contract 最大输出为 6,144 tokens；最终
失败记录会在可用时保存最后一次原始响应和 API `finish_reason`，以便区分长度截断
与 JSON 语法错误。输出位于
`data/contracts/contracts.jsonl.gz`。

后续流程严格按依赖顺序提交：

```bash
# 1. DeepSeek 生成单 Skill query（16 路并发）。
sbatch jobs/generate_single_skill_deepseek.sbatch

# 2. 挖掘局部负例并做局部 FN 过滤。
sbatch jobs/mine_single_skill_local_negatives.sbatch

# 3. 用本地 Qwen 挖掘语义负例并做语义 FN 过滤。
sbatch jobs/mine_single_skill_semantic_negatives.sbatch

# 4. 将过滤后的单 Skill 记录转换为 Reranker group。
sbatch jobs/prepare_single_skill_reranker.sbatch

# 5. 构建 Contract 校验的多 Skill 候选。
sbatch jobs/build_multi_skill_candidates.sbatch

# 6. 先运行 50 条多 Skill DeepSeek pilot 并审计。
sbatch jobs/generate_multiskill_deepseek.sbatch

# 7. pilot 通过后生成全部多 Skill query。
sbatch --export=ALL,FULL_RUN=1 jobs/generate_multiskill_deepseek.sbatch

# 8. 挖掘多 Skill 负例、过滤 FN，并构建 mixed 训练数据。
sbatch jobs/build_mixed_training_data.sbatch
```

两个 DeepSeek query 作业均固定使用 `deepseek-v4-flash`、关闭思考、启用 JSON
Output、每次请求一个条目并保持 16 路并发。负例挖掘和训练数据构建只使用本地
Qwen 模型。连续提交时应添加 Slurm `afterok` 依赖；上游 manifest 和产物未审计前，
不要启动下游阶段。

## 文档

- [RQ1：Skill 检索](docs/rq1-skill-retrieval.md)
- [研究问题地图](docs/README.md)
- [English README](README.md)
