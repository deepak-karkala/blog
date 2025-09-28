# Packaging, Evaluation & Promotion Workflows

##

### 15) Packaging and Export

* **When it runs**

  * Automatically after a training run in #13 or a sweep winner in #14 is marked “candidate”.
  * On demand when an engineer requests a build for a specific target (cloud GPU, vehicle ECU, edge gateway).
  * Nightly to refresh performance-optimized builds with the latest compiler/runtime stacks.

* **Inputs**

  * Best checkpoint from #13/#14 plus its W\&B run, dataset DVC tag, and Git SHA.
  * Export recipe: desired backends and precisions per target, for example:

    * TorchScript or ONNX (opset 17) for CPU/GPU
    * TensorRT engines (FP32, FP16, INT8) for NVIDIA targets
    * Triton ensemble configuration if pre/post-processing is composed as a pipeline
  * Calibration shard for INT8 (balanced by slice, e.g., night rain pedestrians).
  * Inference contract template: input shapes, dtypes, normalization, output schema, confidences.

* **Steps**

  * **Repo staging**

    * Pin environment: Docker image digest, CUDA/cuDNN, PyTorch/NCCL versions.
    * Fetch artifacts from W\&B and S3; verify hashes; freeze the exact `requirements.lock`.
    * Sanity smoke: load checkpoint, single-batch forward, no NaNs/infs.
  * **Graph export**

    * TorchScript trace or script path with dynamic axes if needed; or export to ONNX with opset/IR version constraints.
    * Operator coverage report; fail fast if unsupported ops creep in.
  * **Runtime optimization**

    * Build **TensorRT** engines for target (T4, A10, A100 in cloud; Orin/Drive for edge) with per-device tactic replay.
    * Mixed precision plan selection; per-layer precision fallback where numerically sensitive.
    * **INT8**: create calibration cache with the curated shard; verify max absolute deviation vs FP32 on a validation micro-suite.
    * Optional **quantization-aware training** reuse: if available, prefer QAT checkpoints for INT8.
  * **Model repository assembly**

    * Create **Triton** model dir structure: `config.pbtxt`, versioned subfolders, pre/post processing as Python or TensorRT backends, optional **ensemble** to fuse steps.
    * Generate **inference\_config.json** describing IO schema, thresholds, NMS settings, class map, and expected augmentations disabled at inference.
  * **Security and compliance**

    * Generate SBOM with **Syft**; scan image and artifacts with **Trivy**.
    * License scan for third-party code; attach report to model card.
    * Sign artifacts and/or container with **cosign** or **AWS Signer**; store signatures in S3 and publish digest in release notes.
  * **Equivalence & performance checks**

    * **Numerical equivalence**: FP32 PyTorch vs exported engine on 1k randomized inputs per head; require Δ within tolerances (e.g., bbox IoU drift < 0.5% absolute on sample set; logits Δ < 1e-3).
    * **Latency/throughput microbench**: run on the target instance type; collect p50/p95 latency, GPU util, memory footprint.
    * **Contract smoke**: load model in a minimal Triton/TorchServe container; POST a known request; verify schema and ranges.
  * **Artifact packaging**

    * Produce: `model.ts` or `model.onnx`, `model.plan` (per device), `inference_config.json`, `calibration.cache`, `config.pbtxt`, SBOM, `export_report.json`.
    * Build and push serving container to **ECR** tagged with semver and Git SHA (e.g., `adas-detector:1.8.0-abcdef0`).
    * Attach everything to a **W\&B Artifact** and store in **S3 Gold** under `/models/<task>/<semver or run_id>/…`.

* **AWS/Tooling**

  * **ECR, S3, CodeBuild or GitHub Actions, KMS/Signer, Triton Inference Server, TensorRT, ONNX, TorchScript, W\&B Artifacts**.

* **Outputs**

  * Versioned, signed, performance-graded model packs per target.
  * `export_report.json` with compile flags, precisions, operator sets, and microbenchmarks.
  * Updated W\&B artifact lineage linking back to dataset and code.

---

### 16) Evaluation and Robustness

* **When it runs**

  * Immediately after packaging (#15) for each target precision.
  * Nightly regression across the full test library.
  * On request when a slice shows drift or new edge cases arrive from #12.

* **Inputs**

  * Exported models and serving containers from #15.
  * `golden_train/val/test.manifest`, `slices.yaml`, and extra **challenge suites** (rare weather, construction, tunnels).
  * Baseline “current production” metrics for A/B comparison.
  * Corruption/perturbation suite definitions and OOD probe sets.

* **Steps**

  * **Dataset integrity & leakage guards**

    * Verify manifests conform to schema; run **Great Expectations** on key fields.
    * Ensure no overlap of scene IDs across splits; enforce temporal and geographic separation policies.
  * **Primary evaluation**

    * Compute task-specific metrics:

      * 2D detection: COCO mAP, AP50/75, small/medium/large splits; per-class PR curves.
      * 3D detection: nuScenes metrics or KITTI AP on BEV and 3D boxes.
      * Segmentation/lanes: mIoU, F1, boundary IoU.
      * Prediction: ADE/FDE, miss rate at K.
    * **Slice evaluation** for weather, time of day, geography, road type; compute Δ vs previous release.
    * **Calibration**: ECE, Brier score, reliability diagrams; tune decision thresholds if needed.
  * **Robustness & stress testing**

    * **Image/point cloud corruptions**: blur, noise, JPEG compression, fog/rain/snow shaders, brightness/contrast; LiDAR dropouts; test at increasing severities; measure mAP/mIoU decay slopes.
    * **Temporal stress**: dropped frames, timestamp jitter, out-of-order batches; check tracker continuity and stability.
    * **Sensor faults**: zero out a camera or LiDAR for segments; confirm graceful degradation rules.
    * **Quantization sensitivity**: compare FP32 vs FP16 vs INT8 across slices.
  * **OOD & uncertainty**

    * OOD probes using max softmax probability or energy scores; compute AUROC/AUPR for OOD vs in-dist.
    * Uncertainty quality: NLL, coverage vs confidence; verify abstention policies are triggered sensibly.
  * **Latency and footprint**

    * Measure p50/p95 latency and throughput on target hardware using the packaged engine; cap memory and verify no OOM at peak batch/stream settings.
  * **Regression gates**

    * Define win conditions, e.g., `mAP_weighted +1.5` overall and **no** critical slice regression > 2%; latency p95 within budget; calibration ECE not worse.
    * If a gate fails, emit a **blocking report** and route back to #14 or #8/#12 to mine data for failing slices.
  * **Reporting**

    * Create `eval_report.json`, `slice_metrics.parquet`, `robustness_report.json`, latency summaries, confusion matrices, and reliability plots; log all to **W\&B**.
    * Generate a human-readable `evaluation_summary.md` with a “What improved / what regressed / next actions” section.

* **AWS/Tooling**

  * **EKS or SageMaker Processing**, **Athena/Glue** for audit queries, **W\&B**, **Evidently** for reference vs candidate drift checks, **Triton Perf Analyzer** or custom profilers.

* **Outputs**

  * Machine-readable reports and plots; green/red promotion signal with rationale.
  * Pinned W\&B run linking evaluation to the packaged artifact.

---

### 17) Drive Replay and Simulation

* **When it runs**

  * After #16 indicates the candidate is promising but needs **closed-loop** validation.
  * As a mandatory gate for any major change touching perception->planning interfaces.
  * Periodically to re-validate regressions and expand the scenario library.

* **Inputs**

  * Exported model container and configs from #15.
  * **Log replay** bundles: synchronized multi-sensor recordings with ground truth labels.
  * **Scenario library**: OpenSCENARIO files and procedurally generated scenes based on real-world events (disengagements, near misses).
  * Vehicle dynamics and controller configs for realistic closed-loop behavior.

* **Steps**

  * **Open-loop replay**

    * Reproduce sensor timing, distortions, and calibration; feed logs through the candidate model.
    * Compute frame/segment-level perception metrics against ground truth; analyze time-to-first-detection, track continuity, and ghosting.
    * Flag segments where the candidate diverges materially from production; surface them for targeted review.
  * **Scenario extraction**

    * Convert flagged real-world intervals to **OpenSCENARIO** with actors, trajectories, traffic rules, and weather.
    * Parameterize scenarios (vehicle speed, gap times, actor types) for robust sweeps.
  * **Closed-loop simulation**

    * Run in **CARLA** or **NVIDIA Omniverse/Drive** with high-fidelity sensors and physics.
    * Connect inference to the autonomy stack’s planning/control (or a proxy controller) so the model’s outputs drive the ego vehicle.
    * Randomize across seeds: weather, lighting, textures, spawn densities; run many permutations per scenario.
    * Collect safety metrics: collisions per 1k km, off-road incidents, traffic rule violations, TTC minima, and comfort metrics (jerk/acc).
  * **Batch orchestration**

    * Distribute thousands of runs on **EKS** or **AWS Batch** with GPU nodes; mount scene assets via S3/FSx.
    * Cache compiled simulation assets to avoid rebuilds; checkpoint long sweeps.
  * **Review and gating**

    * Aggregate results; compare to production baselines and to #16 offline metrics.
    * Define pass criteria per scenario category, e.g., **zero** collisions in NCAP-style scenes; no increase in near-misses for vulnerable road users; bounded comfort regressions.
    * Produce clips of failures for quick triage; create issues mapped back to #8/#12 for data requests if needed.

* **AWS/Tooling**

  * **EKS/ECS or Batch** with GPU, **S3/FSx**, **CloudWatch Logs**, **Omniverse/Drive Sim or CARLA**, **ROS/rosbag** replayers when needed, **OpenSCENARIO** toolchain, dashboards in **QuickSight**.

* **Outputs**

  * `sim_summary.parquet`, per-scenario CSV/JSON, video snippets of failures, heatmaps of violation types.
  * A “Sim Gate” verdict attached to the model’s W\&B artifact and promotion checklist.

---

### 18) Registry and Promotion

* **When it runs**

  * After #16 and #17 return green.
  * On product manager approval and change-control window availability.
  * On rollback events (reverse promotion).

* **Inputs**

  * Candidate model pack(s) and serving container(s) from #15.
  * Evaluation and simulation reports from #16/#17.
  * Model card, SBOM, vulnerability/license scans, and signatures.
  * Release notes, migration notes, and serving configs.

* **Steps**

  * **Registry entry**

    * Create/update entry in the **Model Registry** with immutable pointers:

      * W\&B Artifact digest, S3 artifact URIs, ECR image digest, commit SHA, dataset DVC tag.
      * Performance summary, slice table, latency budgets, supported targets, calibration cache version.
    * Apply **semantic versioning** and attach stage: `staging`, `candidate`, `production`.
  * **Governance checks**

    * Validate approvals: technical, safety, security, and product.
    * Verify signatures and SBOM status; ensure vulnerability gates pass or are waived with justification.
    * Lock down IAM policies for read-only production consumption.
  * **Promotion plan**

    * Choose rollout strategy: **shadow** (mirror traffic), **canary** (1%→5%→25%→100%), or **A/B** with customer cohorts.
    * Pre-deploy to **staging** Triton/TorchServe; run **API contract** smoke, performance soak (e.g., 30 min).
    * Define rollback SLOs: if p95 latency, error rate, or safety proxy metrics breach thresholds for N minutes, auto-rollback.
  * **Production push**

    * Deploy canary to EKS/ECS or **SageMaker Endpoints** with the new container and model; wire **CloudWatch** alarms and **Auto Scaling**.
    * Gradually shift traffic; keep shadow for behavioral diffing (store diff summaries to S3).
    * Validate real-world **proxy KPIs** (e.g., false emergency braking rate, perception-planning disagreement rates).
  * **Finalize & broadcast**

    * Promote registry stage to **production**; tag previous model as **rollback**.
    * Publish release notes, link to model card, evaluation, simulation, and SBOM.
    * Notify stakeholders; update dashboards.
  * **Post-promotion hooks**

    * Kick **Offline Mining** (#12) with fresh error clusters seeded from shadow/canary telemetry.
    * Schedule the next **weekly evaluation** on the full library to guard against late regressions.
    * Archive heavy intermediate artifacts per retention policy; maintain cost hygiene.

* **AWS/Tooling**

  * **SageMaker Model Registry** or **W\&B Artifacts** as registry-of-truth with a lightweight **DynamoDB** “promotion table” for active aliases.
  * **EKS/ECS or SageMaker Endpoints** for serving; **CloudWatch**, **Auto Scaling**, **EventBridge** for rollouts; **KMS/Signer/cosign** for integrity.
  * **Athena/QuickSight** for canary KPIs and shadow diffs.

* **Outputs**

  * A versioned, auditable **production** model entry with all lineage and approvals.
  * Canary/rollout timelines, SLO dashboards, and an automated rollback path.
  * Triggers fired to feed the next loop of the data engine.

---

