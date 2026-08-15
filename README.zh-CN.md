# FCSR

FCSR（Full-Coverage Skill Retrieval）面向大规模 Agent Skill 库，使用可审计的合成数据训练两阶段检索系统：

```text
Task Query
  -> BM25 + FCSR Retriever
  -> Reciprocal Rank Fusion (RRF)
  -> FCSR Reranker
  -> Top-10 Skills
```

项目只维护冻结的 Hard-pool 评测。完整方法、数据统计和实验结论见 [RQ1 文档](docs/rq1-skill-retrieval.md)，英文说明见 [README.md](README.md)。

## 最终资产

保留两套同构系统：

- `FCSR`：完整训练集得到的正式模型，位于 `checkpoints/fcsr/`。
- `FCSR-Small`：早期小数据规模消融模型，位于 `checkpoints/fcsr-small/`。

每套模型均包含：

```text
retriever/   Qwen3-Embedding-0.6B LoRA
reranker/    Qwen3-Reranker-0.6B LoRA
```

评测结果位于：

- `reports/baselines/hard/`：BM25、Dense、RRF、Base Reranker 和 SkillRouter。
- `reports/systems/{fcsr,fcsr-small}/hard/`：自有系统逐阶段输出。
- `reports/tables/hard-retrieval.md`：第一阶段检索对比。
- `reports/tables/hard-final.md`：最终系统对比。
- `reports/tables/hard-two-stage.md`：两阶段消融。

## 目录

```text
configs/fcsr.yaml       唯一训练配置
data/                   原始、合成和混合训练数据
jobs/                   Slurm 正式作业
scripts/                数据、训练和评测入口
src/                    核心实现
tests/                  单元测试
checkpoints/            FCSR 与 FCSR-Small
reports/                Hard-pool 结果与表格
```

`data/raw/skills_easy.jsonl.gz` 是训练和负例挖掘候选池，不参与最终评测；最终评测只使用 `data/raw/skills_hard.jsonl.gz`。

## 环境

所有正式计算通过 Slurm 执行。项目使用本地 Conda 环境，不修改集群公共环境：

```bash
conda create -p ./env-qgen python=3.10 -y
conda activate ./env-qgen
pip install -r requirements.txt
```

DeepSeek 凭据保存在未跟踪的 `.env`：

```text
DEEPSEEK_API_KEY=...
```

生成模型固定为 `deepseek-v4-flash`，默认 16 路并发。

## 数据构建

以下是唯一保留的复现顺序。每一步完成并检查日志后再提交下一步。

```bash
sbatch jobs/extract_contracts_deepseek.sbatch
sbatch jobs/generate_single_skill_deepseek.sbatch
sbatch jobs/mine_single_skill_local_negatives.sbatch
sbatch jobs/mine_single_skill_semantic_negatives.sbatch
sbatch jobs/prepare_single_skill_reranker.sbatch
sbatch jobs/build_multi_skill_candidates.sbatch
sbatch jobs/generate_multiskill_deepseek.sbatch
sbatch jobs/build_mixed_training_data.sbatch
```

多 Skill 生成默认写入正式目录。如需小规模验证，只限制处理数量，不创建 Pilot 目录：

```bash
sbatch --export=ALL,LIMIT=50 jobs/generate_multiskill_deepseek.sbatch
```

负例流程包括 BM25、同类别、随机和语义候选，并对全部正例执行 FN 过滤；被移除的候选保存在对应的 `semantic_fn_review.jsonl.gz` 中。

## 训练

正式超参数只在 `configs/fcsr.yaml` 中维护，训练方式固定为 LoRA：

```bash
sbatch jobs/train_fcsr_retriever.sbatch
sbatch jobs/train_fcsr_reranker.sbatch
```

默认输出目录已包含最终 checkpoint。复现实验必须通过 `OUTPUT_DIR` 指定新目录，训练 job 会拒绝覆盖非空目录。

## Hard-pool 评测

先运行第一阶段，再运行对应 Reranker：

```bash
sbatch --export=ALL,FCSR_SYSTEM=fcsr,RETRIEVAL_MODE=dense jobs/evaluate_fcsr_retrieval.sbatch
sbatch --export=ALL,FCSR_SYSTEM=fcsr,RETRIEVAL_MODE=rrf jobs/evaluate_fcsr_retrieval.sbatch
sbatch --export=ALL,FCSR_SYSTEM=fcsr,RETRIEVAL_MODE=dense jobs/evaluate_fcsr_reranker.sbatch
sbatch --export=ALL,FCSR_SYSTEM=fcsr,RETRIEVAL_MODE=rrf jobs/evaluate_fcsr_reranker.sbatch
```

将 `FCSR_SYSTEM` 改为 `fcsr-small` 可复现规模消融。评测协议固定为第一阶段 Top-50、RRF depth 100、`rrf_k=60`、Reranker Top-20，最终向任务提供 Top-10 Skills。

重新渲染标准表格：

```bash
sbatch jobs/render_reports.sbatch
```

## 验证

```bash
bash -n jobs/*.sbatch
PYTHONPATH=.:src ./env-qgen/bin/python -m unittest discover -s tests -v
```

大型数据、checkpoint、日志和报告不由 Git 跟踪；所有小型 manifest 保留在 Git 中，用于记录数据来源和校验信息。
