# Intrebid cluster constraints

- Run every formal compute, inference, preprocessing, and training job through Slurm.
- Use GPU0 only for login, submission, file inspection, monitoring, and brief debugging. Do not run sustained workloads there.
- GPU1 is also a login/management node. Prefer compute nodes GPU2/GPU3 (8x L40 each) or GPU4 (8x H200).
- Use `fast` only for short pilots (maximum 6 hours), `normal` for routine jobs (maximum 3 days), `cpu` for CPU-only work, and `long` only when a justified job needs more than 3 days.
- Test large jobs on a small sample first. Request only the GPU, CPU, memory, and wall time the job actually needs.
- Do not bypass Slurm, keep idle allocations, flood the queue, or submit many near-duplicate experiments.
- Do not modify the shared base environment, CUDA, drivers, or system libraries. Use a project-local Conda environment.
- Keep Hugging Face caches, environments, logs, derived data, and checkpoints under the project directory. Do not duplicate large model weights.
- Never use `sudo`, alter shell startup files, write to system directories, or expose long-running services.
- Human-review generated commands before execution. Preserve public datasets and other users' files.
