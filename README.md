# FCSR：基于 Contract 的 Skill 检索与多 Skill 任务构造

**简体中文**

FCSR（Function-aware Coverage Skill Retriever）是一个面向 Agent Skill 检索的研究框架。它以约 8 万条 Skill 为检索池，复现 SkillRouter 风格的 Bi-Encoder + Reranker 两阶段检索，并增加可审计的 **Skill Contract**：Contract 只作用于训练数据构造，部署时仍检索原始 Skill 的 `name + description + body`。

当前仓库包含三条边界清晰的工作流：

1. 单 Skill 检索训练与 SR 兼容评测。
2. Contract 引导的多 Skill 候选构造，尚未调用 LLM 编写多 Skill 自然语言任务。
3. 独立的 Hard-15 Agent 规划实验，比较 Flat、Hierarchy 与 Evidence Graph 组织方式。

## 状态与边界

| 能力 | 状态 | 说明 |
| --- | --- | --- |
| Contract V2 抽取 | 已实现 | 证据偏移、来源哈希和 schema 均在本地校验。 |
| 单 Skill 合成查询 | 已生成 | `single_v1` 中有 7,342 条 query。 |
| 多 Skill 候选 | 已生成 | 497 个 pair、51 个 triple；仅有 Contract 规则证据。 |
| 多 Skill LLM 任务编写 | 已实现，待运行 | 本地 Qwen 生成器只消费已验证候选，并执行 JSON、顺序和 DAG 校验。 |
| Bi-Encoder / Reranker 训练 | 已实现 | Qwen + LoRA，支持 dry-run。 |
| Hard-15 端到端任务执行 | 未实现 | 当前评估的是规划质量，不报告任务成功率。 |

## 研究问题地图

项目按五个研究问题组织，文档会明确区分已实现的资产、待验证的实验和可报告的结论：

1. [RQ1：大规模 Skill 检索](docs/rq1-skill-retrieval.md)
2. [检索后 Skill 的组织实验](docs/skill-organization.md)
3. [RQ3：Agent 规划](docs/rq3-agent-planning.md)
4. [RQ4：Agent Reflection](docs/rq4-agent-reflection.md)
5. [RQ5：Agent 评估](docs/rq5-agent-evaluation.md)
## 快速开始

要求 Python 3.10 或更高版本。以下命令从仓库根目录执行。

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
python -B -m unittest discover -s tests -v
```

在 CUDA 主机上，优先按 [PyTorch 官方说明](https://pytorch.org/get-started/locally/) 安装与 CUDA 匹配的 PyTorch，再安装这份 `requirements.txt`。

## 依赖

项目只保留一份 `requirements.txt`，覆盖预处理、Qwen 训练、检索评测和 Hard-15 规划实验。

## 数据布局

大型逐行记录统一使用 `.jsonl.gz`；项目的 `data_io` 与所有脚本可直接读取，无需手动解压。

```text
data/raw/
  evaluation_queries.jsonl.gz        # 75 个公开评测任务
  skills_easy.jsonl.gz               # 78,361 条 Easy Skill
  skills_hard.jsonl.gz               # 79,141 条 Hard Skill

data/contracts/
  sample_skills.jsonl.gz             # benchmark-safe 的 8,000 条抽样
  contracts.jsonl.gz                 # 7,995 条 Contract
  failures.jsonl.gz

data/synthetic/
  single_v1/                          # 7,342 条单 Skill 训练记录
    manifest.json
    queries.jsonl.gz
    local_negatives.jsonl.gz
    train_biencoder.jsonl.gz
    train_reranker.jsonl.gz
  compositional_v1/                   # 多 Skill 构造阶段
    manifest.json
    candidates.jsonl.gz               # 497 pair + 51 triple
    candidate_rejections.jsonl.gz
```

`manifest.json` 是版本索引与可复现记录：它保存输入路径、候选构造参数、各产物数量和当前阶段状态。大数据文件默认不提交 Git，版本清单会提交。

## 配置

- [configs/base.yaml](configs/base.yaml)：数据路径、Contract 与检索超参数。
- [configs/model_qwen3_0_6b.yaml](configs/model_qwen3_0_6b.yaml)：Qwen 0.6B 模型和 LoRA 训练默认值。
- [configs/paths_autodl.yaml](configs/paths_autodl.yaml)：AutoDL 路径参考。

本地调用 DeepSeek 前，创建未提交的 `.env`：

```dotenv
DEEPSEEK_API_KEY=your-key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-v4-flash
```

## 单 Skill 数据管线

### 1. 抽样

```powershell
python -B scripts/preprocess.py sample --overwrite
```

抽样会排除 benchmark 的 GT / relevance Skill、去除正文完全重复项，并按类别和语言做确定性分层。

### 2. 抽取 Contract

先用小批量验证 API、schema 和证据对齐：

```powershell
python -B scripts/preprocess.py contracts --limit 3
python -B scripts/preprocess.py contracts
```

### 3. 生成单 Skill query 与本地负例

```powershell
python -B scripts/preprocess.py queries --limit 3
python -B scripts/preprocess.py queries
python -B scripts/preprocess.py local-negatives --overwrite
```

### 4. GPU 语义负例与 Bi-Encoder 训练

```bash
python -B scripts/preprocess.py semantic-negatives \
  --model Qwen/Qwen3-Embedding-0.6B --device cuda --overwrite
python -B scripts/train_biencoder.py --dry-run
python -B scripts/train_biencoder.py \
  --model Qwen/Qwen3-Embedding-0.6B \
  --output-dir checkpoints/fcsr-emb-0.6b
```

### 5. 构建 reranker 组并训练

```bash
python -B scripts/evaluate.py retrieve \
  --queries data/synthetic/single_v1/queries.jsonl.gz \
  --skills data/raw/skills_easy.jsonl.gz \
  --model checkpoints/fcsr-emb-0.6b \
  --output-predictions data/processed/synthetic_top20.json \
  --output-records data/processed/synthetic_top20.jsonl.gz

python -B scripts/train_reranker.py prepare --overwrite
python -B scripts/train_reranker.py train --dry-run
python -B scripts/train_reranker.py train \
  --model Qwen/Qwen3-Reranker-0.6B \
  --output-dir checkpoints/fcsr-rank-0.6b
```

reranker 数据只保存候选元数据；训练时会从原始 Skill 重建 prompt，避免重复存储十余万份长文本。

## 多 Skill 候选构造

这一步不调用 LLM。候选生成器只保留：

- `validated` Contract；
- 与单 Skill query `source_hash` 一致的 Skill；
- 不属于 benchmark 的 Skill；
- 上游 `outputs` 到下游必需 `inputs` 的完整或高重叠 artifact 交接；
- 满足操作互补的有向 pair，以及由两条有向边组成的 triple。

```powershell
python -B scripts/build_compositional_candidates.py
```

若需按新阈值重建，显式覆盖已有文件：

```powershell
python -B scripts/build_compositional_candidates.py `
  --max-artifact-frequency 5 `
  --max-pairs-per-source 16 `
  --overwrite
```

拒绝样本不会丢弃，而是写入 `candidate_rejections.jsonl.gz`，包含 `missing_or_stale_single_skill_query`、`weak_artifact_handoff`、`missing_complementary_operation` 等原因。生成器只消费这些候选，严格校验 JSON、Skill ID 顺序和依赖 DAG，不能自行发明 Skill 组合。

本地 Qwen 生成前先做无模型预检，再用一张 GPU 跑 50 条 pilot：

```powershell
python -B scripts/generate_compositional_queries.py --dry-run --limit 50
```

服务器上使用 [jobs/generate_compositional_qwen3_8b.sbatch](jobs/generate_compositional_qwen3_8b.sbatch) 提交 pilot。成功记录、失败记录和复核队列分别写入 `compositional_queries.jsonl.gz`、`failures.jsonl.gz` 与 `review_queue.jsonl.gz`。

## 检索与评测

以下示例运行 Easy pool；替换为 `skills_hard.jsonl.gz` 并使用 `--tier hard` 即可评测 Hard pool。

```bash
python -B scripts/evaluate.py retrieve \
  --queries data/raw/evaluation_queries.jsonl.gz \
  --skills data/raw/skills_easy.jsonl.gz \
  --model checkpoints/fcsr-emb-0.6b --top-k 50 \
  --output-predictions reports/retrieval_easy.json \
  --output-records reports/retrieval_easy.jsonl

python -B scripts/evaluate.py rerank \
  --retrieval-records reports/retrieval_easy.jsonl \
  --skills data/raw/skills_easy.jsonl.gz \
  --model checkpoints/fcsr-rank-0.6b --top-k 20 \
  --output-predictions reports/reranker_easy.json \
  --output-records reports/reranker_easy.jsonl

python -B scripts/evaluate.py score \
  --tasks data/raw/evaluation_queries.jsonl.gz \
  --skills data/raw/skills_easy.jsonl.gz \
  --predictions reports/reranker_easy.json \
  --stage reranker --tier easy
```

评测遵循 SkillRouter 兼容协议：过滤 `generic_only`，将 GT/relevance 与当前 Skill pool 取交集，报告整体、single-skill、multi-skill、FullCoverage 和分级 relevance nDCG。

## Hard-15 Skill 组织实验

新的端到端实验固定 `fcsr-multiskill3x-rrf` 的每任务 Top-8，向 SkillsBench Agent 注入相同的 SkillRouter Hard JSONL 文本载荷，并比较 No Skill、Flat、Hierarchy 和 Evidence Graph。旧的 planning-only 管线已经移除；实验约束、输入指纹、泄漏控制、Skill 打包方式和验收标准见 [Hard-15 Skill 组织实验](docs/skill-organization.md)。

## 常用命令

```powershell
# 列出所有入口的参数
python -B scripts/preprocess.py --help
python -B scripts/build_compositional_candidates.py --help
python -B scripts/train_biencoder.py --help
python -B scripts/train_reranker.py --help
python -B scripts/evaluate.py --help

# 运行完整离线回归
$env:PYTHONPATH = "src"
python -B -m unittest discover -s tests -v
```

## 研究关系

FCSR 使用 SkillRouter 公开的数据格式、池化方式、检索/重排指标和两阶段训练约定；Contract、负例构造与多 Skill 候选构造是本项目新增的可审计数据工程层。SkillRouter 本身不提供本仓库的预处理和训练数据构造实现。
