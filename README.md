# FCSR：面向功能与覆盖度的 Skill 检索器

[English](README_EN.md) | **简体中文**

FCSR（Function-aware Coverage Skill Retriever）是一个面向 Agent Skill 检索的轻量训练与评测框架，目标技能库规模约为 8 万条。项目沿用 SkillRouter 公开的检索数据格式和 Bi-Encoder/Reranker 两阶段架构，并引入基于原文证据的 **Skill Contract**，用于指导合成查询生成。

Contract 只作为抽样训练正例的旁路结构化信息。部署时，编码器仍然索引原始 Skill 的 `name + description + body`，因此不需要对完整的 8 万条 Skill 全部调用 Contract 抽取 API。

## 整体流程

```text
Easy 自然技能池
  -> 避开评测答案的类别 × 语言分层抽样（8,000 条）
  -> 使用 DeepSeek 抽取 Contract V2
  -> 基于 Contract 生成合成查询
  -> 挖掘 BM25 + 同类别 + 随机负样本
  -> 使用 Qwen 挖掘语义负样本并过滤假负例
  -> 使用 InfoNCE 训练 Bi-Encoder
  -> 使用训练后的 Bi-Encoder 构造 Top-20 候选组
  -> 使用 Listwise Loss 训练 Reranker
  -> 在 Easy/Hard 技能池上完成检索、重排和 SR 兼容评测
```

Hard 技能池只用于评测。它包含 Easy 自然技能池以及额外的基准干扰项。

## 项目结构

```text
configs/              路径、阈值及适配 RTX 4090 的默认配置
scripts/              数据预处理、训练和评测入口
src/                  扁平化的可复用模块，不再嵌套 Python 包
tests/                使用伪 API 客户端的离线单元测试
data/raw/             75 条评测查询以及 Easy/Hard 技能池
data/contracts/       8K 抽样结果、Contract、失败记录和清单
data/processed/       中间检索结果和待复核样本
data/synthetic/       合成查询及训练数据
checkpoints/           LoRA 或全量微调检查点
reports/               SR 兼容的指标汇总和逐任务结果
```

磁盘上的目录名全部使用小写，例如 `tests/` 和 `data/`。上面的缩进仅用于说明层级。

## 1. 本地环境准备

在本地计算机的 `fcsr` 根目录执行：

```powershell
python -m pip install -r requirements.txt
$env:PYTHONPATH = "src"
python -B -m unittest discover -s tests -v
```

如果本地没有安装 PyTorch，4 个训练数值测试会自动跳过，其余测试仍会正常执行。安装 `requirements-train.txt` 后，这些测试会自动恢复。

项目需要以下原始数据：

```text
data/raw/evaluation_queries.jsonl   整理后的 75 条计分任务
data/raw/skills_easy.jsonl          包含 78,361 条 Skill 的自然技能池
data/raw/skills_hard.jsonl          包含 79,141 条 Skill 的困难技能池
```

## 2. 在本地分层抽样 8,000 条 Skill

```powershell
python -B scripts/preprocess.py sample `
  --skills data/raw/skills_easy.jsonl `
  --tasks data/raw/evaluation_queries.jsonl `
  --sample-size 8000 --seed 42 `
  --output-dir data/contracts --overwrite
```

输出文件：

```text
data/contracts/sample_skills.jsonl
data/contracts/manifest.json
```

抽样器会排除评测集中所有 GT/relevance Skill，删除内容完全相同的重复项，并按照类别与语言分层执行确定性的平方根配额分配。

## 3. 使用 DeepSeek V4 Flash 抽取 Contract V2

在项目根目录的 `.env` 文件中填写 DeepSeek API Key：

```dotenv
DEEPSEEK_API_KEY=你的真实 API Key
```

程序会自动读取该文件，但不会覆盖已经在终端中设置的同名环境变量。`.env` 已加入 `.gitignore`，不得提交或上传到服务器；README 提供的 `scp` 命令也不会包含它。

默认模型为 [`deepseek-v4-flash`](https://api-docs.deepseek.com/news/news260424/)，API 地址仍为 `https://api.deepseek.com`。建议先测试 3 条数据：

```powershell
python -B scripts/preprocess.py contracts --limit 3
```

检查 `data/contracts/contracts.jsonl` 和 `data/contracts/failures.jsonl`。确认无误后，去掉 `--limit` 继续处理完整样本：

```powershell
python -B scripts/preprocess.py contracts
```

已完成的 `(skill_id, source_hash)` 会自动跳过。每个语义字段都必须引用原始 Skill 中的精确文本；证据偏移量和哈希由代码计算，并通过 `src/contract_schema.py` 校验。

## 4. 生成基于 Contract 的查询

同样先测试 3 条，再继续处理全部有效 Contract：

```powershell
python -B scripts/preprocess.py queries --limit 3
python -B scripts/preprocess.py queries
```

运行时会显示 `Queries` 进度条以及 `ok`、`skip`、`fail` 计数；在日志任务中可添加 `--no-progress` 关闭动态输出。

输出为 `data/synthetic/queries.jsonl`。Query Prompt V5 要求每条查询为 80--180 个英文词，并把 Contract 中的 operations、outputs、constraints 和 quality criteria 作为交付动作白名单；周边业务只能作为已经存在的场景背景。词数不合格、泄露 Skill 名称、JSON 格式无效或者缺少当前有效 Contract 的查询会被拒绝或重试，并在适用时写入失败记录。

## 5. 挖掘本地负样本

```powershell
python -B scripts/preprocess.py local-negatives `
  --queries data/synthetic/queries.jsonl `
  --skills data/raw/skills_easy.jsonl `
  --output data/synthetic/local_negatives.jsonl `
  --seed 42 --overwrite
```

终端会依次显示读取 Skill、构建 BM25 索引和逐 Query 挖掘三个阶段；进入挖掘阶段后显示处理数量、速度和 ETA。日志任务可添加 `--no-progress` 关闭动态输出。

在经过 Skill 身份、标准化名称、完全相同正文和字符三元组假负例过滤后，每条数据最多获得 `3 个 BM25 + 2 个同类别 + 1 个随机` 候选负样本。

## 6. 上传到 AutoDL

上传代码框架和本地生成的数据。服务器可以继续使用已经存在的原始数据集：

```powershell
scp -P 42112 -r configs scripts src requirements.txt requirements-train.txt README.md README_EN.md `
  root@connect.westb.seetacloud.com:/root/autodl-tmp/fcsr/
scp -P 42112 -r data/contracts data/synthetic `
  root@connect.westb.seetacloud.com:/root/autodl-tmp/fcsr/data/
```

## 7. 配置 AutoDL 并下载模型

在 AutoDL 服务器执行：

```bash
cd /root/autodl-tmp/fcsr
pip install -r requirements-train.txt
mkdir -p /root/autodl-tmp/models
hf download Qwen/Qwen3-Embedding-0.6B --local-dir /root/autodl-tmp/models/Qwen3-Embedding-0.6B
hf download Qwen/Qwen3-Reranker-0.6B --local-dir /root/autodl-tmp/models/Qwen3-Reranker-0.6B
```

这是项目唯一需要下载模型的步骤，不要在只负责数据预处理的本地计算机上执行。

## 8. 在 AutoDL 上挖掘语义负样本

```bash
python -B scripts/preprocess.py semantic-negatives \
  --local data/synthetic/local_negatives.jsonl \
  --skills data/raw/skills_easy.jsonl \
  --model /root/autodl-tmp/models/Qwen3-Embedding-0.6B \
  --output data/synthetic/train_biencoder.jsonl \
  --review data/processed/contract_fn_review.jsonl \
  --top-k 50 --threshold 0.95 --batch-size 8 --overwrite
```

最终每条训练数据最多包含 `4 个语义负例 + 3 个 BM25 负例 + 2 个同类别负例 + 1 个随机负例`。与正例嵌入相似度过高的候选会被过滤，并导出供人工复核。

## 9. 训练 Bi-Encoder

先在不加载模型的情况下校验训练数据：

```bash
python -B scripts/train_biencoder.py --dry-run
```

使用默认的 RTX 4090 安全 LoRA 配置开始训练：

```bash
python -B scripts/train_biencoder.py \
  --model /root/autodl-tmp/models/Qwen3-Embedding-0.6B \
  --output-dir checkpoints/fcsr-emb-0.6b
```

默认配置为：训练 1 个 epoch、micro-batch 为 1、梯度累积 16 步、BF16、梯度检查点，以及 0.05 的 InfoNCE 温度。正式训练会显示当前 epoch、已完成 batch、实时 loss、速度和 ETA。若希望更接近 SkillRouter 的全量微调方式，可以使用 `--method full`，但显存和训练成本会更高。

## 10. 构造 Top-20 候选组并训练 Reranker

使用训练后的 Bi-Encoder 检索合成查询：

```bash
python -B scripts/evaluate.py retrieve \
  --queries data/synthetic/queries.jsonl \
  --skills data/raw/skills_easy.jsonl \
  --model checkpoints/fcsr-emb-0.6b \
  --top-k 20 --batch-size 8 \
  --output-predictions data/processed/synthetic_top20.json \
  --output-records data/processed/synthetic_top20.jsonl
```

构造有序候选组并校验：

```bash
python -B scripts/train_reranker.py prepare \
  --retrieval data/processed/synthetic_top20.jsonl \
  --skills data/raw/skills_easy.jsonl \
  --output data/synthetic/train_reranker.jsonl --top-k 20 --overwrite
python -B scripts/train_reranker.py train --dry-run
```

开始训练：

```bash
python -B scripts/train_reranker.py train \
  --model /root/autodl-tmp/models/Qwen3-Reranker-0.6B \
  --output-dir checkpoints/fcsr-rank-0.6b
```

Listwise Loss 会将概率质量分配给候选组中的全部有效正例，并拒绝不包含任何正例的候选组。

## 11. 在 Easy 和 Hard 技能池上评测

每个技能池都需要先导出检索 Top-50，再重排前 20 个候选，最后计算指标。以下为 Easy 技能池示例：

```bash
python -B scripts/evaluate.py retrieve \
  --queries data/raw/evaluation_queries.jsonl \
  --skills data/raw/skills_easy.jsonl \
  --model checkpoints/fcsr-emb-0.6b --top-k 50 \
  --output-predictions reports/retrieval_easy.json \
  --output-records reports/retrieval_easy.jsonl
python -B scripts/evaluate.py rerank \
  --retrieval-records reports/retrieval_easy.jsonl \
  --skills data/raw/skills_easy.jsonl \
  --model checkpoints/fcsr-rank-0.6b --top-k 20 \
  --output-predictions reports/reranker_easy.json \
  --output-records reports/reranker_easy.jsonl
python -B scripts/evaluate.py score \
  --tasks data/raw/evaluation_queries.jsonl \
  --skills data/raw/skills_easy.jsonl \
  --predictions reports/retrieval_easy.json \
  --stage retrieval --tier easy
python -B scripts/evaluate.py score \
  --tasks data/raw/evaluation_queries.jsonl \
  --skills data/raw/skills_easy.jsonl \
  --predictions reports/reranker_easy.json \
  --stage reranker --tier easy
```

评测 Hard 技能池时，将技能文件替换为 `skills_hard.jsonl`，将 `--tier` 改为 `hard`，并将输出文件名中的 `*_easy.*` 改为 `*_hard.*`。

计分流程与 SkillRouter 保持一致：排除 `generic_only` 任务，将 GT/relevance 与当前技能池取交集，使用分级 relevance 计算 nDCG，并分别报告整体、single-skill、multi-skill 和 FullCoverage 指标。

## 方法范围

公开的 [SkillRouter 仓库](https://github.com/zhengyanzhao1997/SkillRouter) 以 MIT 许可证发布了基准数据以及推理和评测代码，但没有发布训练数据预处理脚本。FCSR 保留了其公开的数据格式、池化方法、评测指标、Top-20 候选组和两阶段训练约定；本项目中的负样本挖掘与训练代码则依据论文描述重新实现。

考虑项目预算，FCSR 与论文方案存在以下差异：

1. 分层抽样 8,000 个正例，而不是生成 37,979 个合成训练对。
2. 默认使用 LoRA，同时保留成本更高的全量微调选项。
3. Contract 只对抽样正例进行抽取；可疑假负例会导出供人工复核，不再额外调用付费 LLM 自动判断。
