# Model Training & Experimentation

##
---

### 13) Distributed Training

* **Trigger**

  * Automatic: new `golden_train/val/test.manifest` from Workflow #11, or new auto-mined slice from #12.
  * Manual: engineer launches a training run from the UI/CLI, selecting dataset spec + model recipe.
  * Scheduled: nightly/weekly rebuild on rolling data window.

* **Inputs**

  * Curated dataset manifests + `slices.yaml` (from #11), plus DVC tag/version.
  * Model recipe (backbone, heads, loss weights, augmentations) and training config (`train.yaml`).
  * Pretrained weights (optional) for warm-start or continual learning.
  * Code container image (ECR) and commit SHA; W\&B project/entity; environment secrets.
  * Hardware profile (e.g., `p4d.24xlarge` × N nodes) and storage profile (FSx for Lustre).

* **Core Steps**

  * Packaging & Staging

    * Build/push Docker image with PyTorch, NCCL, CUDA, AMP, torchvision/OpenMMLab, and your repo.
    * Stage dataset to **FSx for Lustre** (for I/O throughput) using DVC or AWS DataSync; keep bulk in S3.
    * Pre-generate sharded WebDataset/TFRecord (optional) for faster streaming.
  * Orchestration

    * Launch **SageMaker distributed training** job (preferred) or **EKS** Job with `torchrun` (DDP).
    * Configure NCCL and networking (env: `NCCL_DEBUG=INFO`, `NCCL_SOCKET_IFNAME=eth0`, `NCCL_ASYNC_ERROR_HANDLING=1`).
  * Training Runtime

    * Initialize **DDP** with `backend=nccl`, set `torch.set_float32_matmul_precision("high")` where supported.
    * **Data loading**: multi-worker `DataLoader` + persistent workers, pinned memory, async prefetch.
    * **Mixed precision (AMP)** + gradient scaling; gradient accumulation for large effective batch sizes.
    * **Task balancing** for multi-head models (e.g., HydraNet-style): dynamic or curriculum weighting.
    * **Checkpointing**: local → FSx (fast) every N steps; periodic sync to S3 (`s3://…/checkpoints/…`).
    * **Fault tolerance**: resume from last global step on spot interruption; save RNG states/optimizers/scalers.
  * Evaluation & Gating

    * After each epoch: run full **offline eval** on `val` + key **slices**; compute mAP/mIoU, per-class AP, trajectory ADE/FDE, lane F1, etc.
    * Produce **slice dashboards** (per weather/time/road class) and **regression checks** against last prod model.
    * Latency/throughput microbenchmarks: TorchScript export + dummy inference to report p50/p95 latency.
  * Logging & Metadata

    * Log all metrics/artifacts to **Weights & Biases**: run config, gradients, losses, images/videos, PR/ROC curves.
    * Register artifacts: dataset version (DVC hash), code SHA, Docker digest, checkpoints, exported models.
  * Testing inside the run (hard gates)

    * **Sanity checks**: one forward+backward batch before training; fail fast on NaNs or exploding loss.
    * **Determinism smoke** (seeded) on small shard; tolerance bands on metrics.
    * **Data drift guard**: compare batch feature stats vs. training reference (Evidently profile) and warn/abort on severe drift.
  * Post-Run Actions

    * Generate **model card** (data provenance, metrics, caveats) and attach to W\&B run & S3.
    * If gates pass, push **exported model** (TorchScript/ONNX) to artifact store and **Model Registry** (e.g., SageMaker Model Registry or W\&B Artifacts promoted alias).
    * Emit event to orchestration (Airflow/Step Functions) to trigger **#14 HPO** (if configured) or **#15 Eval/Sim** (next workflow).

* **AWS/Tooling**

  * **SageMaker Training** (DDP), **ECR**, **S3**, **FSx for Lustre**, **CloudWatch Logs**, **IAM**, optional **EKS**.
  * **PyTorch** (DDP/torchrun), **NCCL**, **torch.cuda.amp**, **OpenMMLab/MMDetection** (optional), **Albumentations**.
  * **W\&B** for tracking/artifacts; **DVC** for dataset pinning; **Evidently** for drift profiles.

* **Outputs**

  * Best checkpoint (`.pt`), plus **TorchScript/ONNX** exports; quantization-ready or TensorRT plan (optional).
  * `eval_report.json` (overall + per-slice metrics, latency), confusion matrices, PR curves.
  * **W\&B run** (summary, artifacts), **model card** (`model_card.md`), **training logs**.
  * Stored in **S3 (Gold)** under `/models/<project>/<semver or run_id>/…`, and linked in **W\&B Artifacts** and registry.

---

### 14) Hyper-Parameter Optimization / Sweeps

* **Trigger**

  * Auto: training workflow marks model as “candidate” but below target on one or more slices.
  * Scheduled: weekly sweeps on prioritized tasks (e.g., night/pedestrian detector).
  * Manual: engineer launches targeted sweep (e.g., loss weights for lane vs. detection).

* **Inputs**

  * Same dataset manifests as #13 (or focused **slice packs** for the weak area).
  * Baseline config (`train.yaml`) with **search space**: LR/WD, warmup, aug policy, loss weights, NMS/score thresholds, backbone/neck options, EMA on/off, AMP level, batch size/accum steps.
  * **Sweep strategy**: Bayesian, Hyperband/ASHA, Random, or **Population-Based Training** (PBT) for long runs.
  * Resource budget: max trials, parallelism, GPU hours, early-stop policy, cost cap.

* **Core Steps**

  * Orchestration & Budgeting

    * Create **W\&B Sweep** config (YAML) with objective metric (e.g., `val/mAP_weighted` or **multi-objective** with constraints: maximize `mAP_vehicle` subject to `latency_p95 < X ms` and `regression_Δslice < Y`).
    * Choose executor:

      * **Kubernetes Jobs** on EKS with the **W\&B agent** (elastic parallelism), **or**
      * **SageMaker** multiple training jobs tagged to the sweep (agent inside container).
    * Enforce **cost guardrails**: kill/truncate low-performers via **ASHA**; set CloudWatch alarms on spend.
  * Trial Runtime

    * Each trial inherits **DDP** setup from #13; parameters injected via env/CLI override.
    * Log full telemetry to W\&B: metrics, hyperparams, system stats (GPU util, memory), eval artifacts.
    * **Early stopping** on plateau or rule-based pruning (e.g., after 3 epochs if `val/mAP` < quantile Q).
  * Validation & Gating (per trial)

    * Same eval battery as #13, including **slice metrics** and **latency microbenchmarks**.
    * **Fairness/coverage checks**: ensure no >K% regression on protected or safety-critical slices.
    * **Stability check** (optional): re-run best few trials with a different seed on a small shard; require consistent ranking.
  * Selection & Promotion

    * W\&B **sweep dashboard** computes leaderboard; export **Pareto front** for multi-objective cases.
    * Auto-package **Top-K** models; re-evaluate on **holdout test** and simulation smoke set.
    * Generate **HPO report**: importance analysis (SHAP/Iceberg plots of hyperparams), budget used, recommended defaults.
    * Promote winner to **Model Registry** with tag `candidate/<task>/<date>`; attach sweep summary and config freeze.
  * Learnings Back-Prop

    * Update default training config with **new best hyperparams** (per task/slice).
    * Persist tuned **augmentation policies** and **loss weights** into recipe library.
    * If all trials fail gating on a specific slice, emit **data request** back to #8/#12 to mine more examples.

* **AWS/Tooling**

  * **EKS** (Jobs + W\&B agent) or **SageMaker** (parallel training jobs), **S3**, **FSx**, **CloudWatch**, **EventBridge** for triggers.
  * **W\&B Sweeps** (Bayes/Random/Hyperband/PBT), **PyTorch DDP**, **Ray Tune** (optional backend for advanced schedulers).
  * **Evidently** (sanity drift checks during trials), **Numba/TVM/TensorRT** (optional latency constraint evaluators).

* **Outputs**

  * `sweep_report.md` + `sweep_summary.json` (leaderboard, Pareto set, importance).
  * Top-K model artifacts + exports; W\&B artifacts with pinned configs and datasets.
  * Registry update (winner promoted) + **promotion event** to downstream eval/simulation pipeline.

---
