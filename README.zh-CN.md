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
  contracts/                     抽样 Skill 与 Contract
  raw/                           Skill 池和公开评测任务
  synthetic/
    single_skill_v1/             单 Skill 查询和训练记录
    multiskill_v1/               已验证候选、LLM 查询与 manifest
  training/
    multiskill3x/                混合训练集（Git 忽略）
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

大数据、模型、checkpoint、日志、缓存和生成的报告均被 Git 忽略。每个受跟踪的 `manifest.json` 记录数据版本、输入、参数和产物数量。

## 环境

需要 Python 3.10 或更新版本。以下命令在仓库根目录运行。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
python -B -m unittest discover -s tests -v
```

在 CUDA 机器上，必要时先根据 [PyTorch 安装说明](https://pytorch.org/get-started/locally/) 安装与宿主环境匹配的 PyTorch，再安装 `requirements.txt`。

Contract 抽取和单 Skill 查询生成默认使用本地的 `models/Qwen3-8B`，不需要 API
密钥，也不会访问网络。如需显式使用旧的 DeepSeek 路径，再配置不提交的 `.env`
并传入 `--provider deepseek --model deepseek-v4-flash`：

```dotenv
DEEPSEEK_API_KEY=your-key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
```

## 数据构建

### 1. 单 Skill 数据

```powershell
# 先构建 benchmark-safe 的 Contract 样本，再抽取带证据的 Contract。
python -B scripts/build_single_skill_data.py sample --overwrite
python -B scripts/build_single_skill_data.py contracts `
  --provider local --model models/Qwen3-8B --device cuda

# 生成查询、局部负例，再在 GPU 上挖掘语义负例。
python -B scripts/build_single_skill_data.py queries
python -B scripts/build_single_skill_data.py local-negatives --overwrite
python -B scripts/build_single_skill_data.py semantic-negatives `
  --model models/Qwen3-Embedding-0.6B --device cuda --overwrite
```

输出位于 `data/synthetic/single_skill_v1/`。

### 2. 多 Skill 数据

候选构建阶段不调用 LLM。只有当有效 Contract、非 benchmark Skill、artifact 交接和操作互补共同成立时，才保留有序 pair 或 triple。

```powershell
python -B scripts/build_multiskill_candidates.py

# Qwen3-8B 或其他本地指令模型只为已验证候选写自然语言任务。
python -B scripts/generate_multiskill_queries.py `
  --model models/Qwen3-8B --max-attempts 3 --progress
```

生成器会校验 JSON 结构、正例 Skill ID 及顺序、source hash、子任务覆盖和依赖 DAG。结果写入 `data/synthetic/multiskill_v1/` 下的 `queries.jsonl.gz`、`failures.jsonl.gz` 与 `review_queue.jsonl.gz`。

### 3. 构建混合训练集

`multiskill3x` 保留全部单 Skill 样本，并将每个原始多 Skill 任务组确定性重复三次。每个正例 Skill 展开为一条 Bi-Encoder 样本；Reranker 保留同一任务内全部正例的多标签 group。负例只从 Easy pool 挖掘，并排除全部正例 Skill ID。

```bash
python -B scripts/build_multiskill_training_data.py \
  --negative-model models/Qwen3-Embedding-0.6B \
  --multiplier 3 \
  --output-dir data/training/multiskill3x
```

## 模型训练

以下命令使用本地 Hugging Face 模型目录，并用清晰名称保存 LoRA adapter。

```bash
python -B scripts/train_biencoder.py \
  --train-data data/training/multiskill3x/biencoder.jsonl.gz \
  --skills data/raw/skills_easy.jsonl.gz \
  --model models/Qwen3-Embedding-0.6B \
  --output-dir checkpoints/fcsr-emb-0.6b-multiskill3x

python -B scripts/train_reranker.py train \
  --groups data/training/multiskill3x/reranker.jsonl.gz \
  --skills data/raw/skills_easy.jsonl.gz \
  --model models/Qwen3-Reranker-0.6B \
  --output-dir checkpoints/fcsr-rank-0.6b-multiskill3x
```

单 Skill 基线不需要构建混合集，使用默认训练输入和 `checkpoints/fcsr-emb-0.6b`、`checkpoints/fcsr-rank-0.6b` 即可。

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
  --model checkpoints/fcsr-rank-0.6b-multiskill3x \
  --top-k 10 \
  --output-predictions reports/reranker/hard/rrf-base-emb-multiskill3x/predictions.json \
  --output-records reports/reranker/hard/rrf-base-emb-multiskill3x/records.jsonl

python -B scripts/evaluate.py score \
  --tasks data/raw/evaluation_queries.jsonl.gz \
  --skills data/raw/skills_hard.jsonl.gz \
  --predictions reports/reranker/hard/rrf-base-emb-multiskill3x/predictions.json \
  --stage reranker --tier hard \
  --output-dir reports/reranker/hard/rrf-base-emb-multiskill3x

# 输出最终系统表和两阶段消融表。
python -B scripts/render_evaluation_tables.py
```

生成的表格：

- `reports/tables/hard-baselines.md`：每个系统只保留最终输出。
- `reports/tables/hard-two-stage-ablation.md`：仅比较两阶段系统的 retrieval 与 rerank。

最终表中的加粗仅表示数值最大，不表示统计显著性。

## Slurm 查询生成

离线 Contract 抽取 pilot 模板位于
[jobs/extract_contracts_qwen3_8b.sbatch](jobs/extract_contracts_qwen3_8b.sbatch)。它只申请
一张计算节点 GPU，启用 Hugging Face 离线模式，默认抽取 32 条并写入独立的
`data/contracts_local_qwen3_8b/`，便于正式扩容前检查质量和吞吐：

```bash
sbatch jobs/extract_contracts_qwen3_8b.sbatch
```

确认 pilot 后再把 `LIMIT` 设为空运行完整样本。同一输出路径会按照
`(skill_id, source_hash)` 自动跳过已完成记录，任务中断后可以安全续跑。

Qwen3-8B pilot 模板位于 [jobs/generate_multiskill_qwen3_8b.sbatch](jobs/generate_multiskill_qwen3_8b.sbatch)。根据实验室规则确认项目路径、QoS 和 GPU 后提交：

```bash
sbatch jobs/generate_multiskill_qwen3_8b.sbatch
```

## 迁移已有本地或服务器数据

此次重构仅改变命名，不改变数据 schema。`git pull` 后新目录中已经有受跟踪的 manifest，因此只合并被 Git 忽略的数据文件：

```bash
rsync -a --exclude manifest.json data/synthetic/single_v1/ \
  data/synthetic/single_skill_v1/
rsync -a --exclude manifest.json data/synthetic/compositional_v1/ \
  data/synthetic/multiskill_v1/
mv data/synthetic/multiskill_v1/compositional_queries.jsonl.gz \
  data/synthetic/multiskill_v1/queries.jsonl.gz
mv data/training/rq1-mixed-3x data/training/multiskill3x
mv data/processed/contract_fn_review.jsonl.gz \
  data/processed/semantic_negative_review.jsonl.gz
mv data/processed/synthetic_top20.json \
  data/processed/single_skill_top20_predictions.json
mv data/processed/synthetic_top20.jsonl.gz \
  data/processed/single_skill_top20_records.jsonl.gz
```

已有 reports 中，将同一路径的 `fcsr-base` 改为 `fcsr-single`、`base-rerank` 改为 `dense-base-reranker`。确认文件已复制后，再删除两个旧 synthetic 目录。重新构建数据后会更新对应 manifest；不要手工修改 JSONL 记录内容。

## 文档

- [RQ1：Skill 检索](docs/rq1-skill-retrieval.md)
- [研究问题地图](docs/README.md)
- [English README](README.md)
