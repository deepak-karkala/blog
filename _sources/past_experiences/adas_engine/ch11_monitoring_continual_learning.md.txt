# Monitoring & Continual Learning

##

### 25) Drift Detection (Data, Prediction, Concept, Performance)

* **When it runs**

  * Stream: near-real-time sliding windows (e.g., 15-minute / 1-hour) over online inference topics for early warning.
  * Batch: scheduled **Airflow** jobs (hourly/daily/weekly) to produce canonical drift reports and trend lines.
  * On-demand: after a deployment, during canary, or if SLOs breach (p95 latency, error rate) or OOD counters spike.

* **Inputs**

  * **Online telemetry**: request schemas, feature snapshots (PII-scrubbed), model outputs (scores/boxes/masks/uncertainty), per-request timing + resource metrics.
  * **Reference baselines**: per-slice statistics frozen at promotion time from #16 (evaluation) and “healthy” historical windows (seasonal references).
  * **Label trickle**: a small, delayed stream of ground truth from #10 (human QA) and #9 (auto-label confirmations) to estimate concept drift where possible.

* **Steps**

  * **Collection & privacy**

    * Tap inference logs via **OpenTelemetry** exporters; route to **Kinesis** → **S3 (Parquet)**; strip or hash identifiers; mask PII fields at the edge.
    * Maintain a **feature dictionary** (name, type, valid range, unit) in Glue Data Catalog; enforce with validators.
  * **Drift computations**

    * **Schema drift**: required fields present; types/ranges; missingness change (Great Expectations).
    * **Covariate drift** (inputs):

      * Numeric: PSI, KS/AD tests, distributional distance; maintain rolling means/variances and quantiles.
      * Categorical: population stability, χ² tests; top-k category churn.
      * Temporal: autocorrelation changes; seasonality break detection (CUSUM on aggregates).
    * **Prediction drift**:

      * Score histograms per class; calibration shift (ECE, Brier); acceptance/abstention rate drift.
      * Spatial: IoU of drivable-area segmentation vs. last known stable; box geometry sanity (aspect ratio, area) distributions.
      * Temporal: track fragmentation, ID-switch rates, latency correlation with confidence.
    * **Concept drift** (where labels available): prequential error rates, sliding-window AUC/mAP/mIoU; DDM/ADWIN style changepoint detectors.
    * **OOD/uncertainty sentinels**: ensemble disagreement, Mahalanobis distance in penultimate embeddings; maintain per-slice OOD counters.
  * **Attribution & slicing**

    * Always compute by **critical slices** (weather, time, geography, road type, sensor subset) and cohorts (fleet/customer).
    * Root-cause heuristics: feature importance on drift indicators (e.g., Shapley on “drift vs no-drift” classification).
  * **Thresholding & governance**

    * Severity bands: **Green** (noise), **Yellow** (monitor), **Red** (action). Calibrate thresholds per slice to avoid alert fatigue.
    * **Dedup & cool-down**: suppress duplicate alerts within a lookback window; escalate if duration exceeds T.
  * **Reporting & alerts**

    * Emit `drift_report.json` and `drift_metrics.parquet`; publish Grafana tiles; send PagerDuty alerts with the top 3 implicated features/slices.
    * File an “Active Drift” ticket with a playbook link (triage, rollback rules, and data-mining recipe).

* **Core AWS / Tooling**

  * **Airflow**, **Kinesis**, **S3/Glue/Athena**, **Evidently**, **Great Expectations**, **OpenTelemetry**, **Prometheus/AMP**, **Grafana**, **QuickSight**.

* **Outputs & Storage**

  * Canonical drift artifacts in **S3** (`/monitoring/drift/YYYY/MM/DD/…`), Glue tables for Athena, alert records in incident tracker.
  * Event to **#26 Continual Learning Trigger** (with a compact spec of what and where drifted).

---

### 26) Continual Learning Trigger (Triage → Decide → Specify)

* **When it runs**

  * Fired by a **Red/Yellow** drift from #25, by performance regressions in canary/production, or by business/product requests (e.g., “construction zones increased; improve precision”).
  * Nightly “gap analysis” against strategic coverage goals (ensuring rare slices are kept in check).

* **Inputs**

  * Drift alert payloads: implicated features/slices, magnitude, duration, example URIs.
  * Error cohorts from #12 (offline mining) and canary/shadow diffs from #19.
  * Capacity & budget constraints (GPU hours, labeling budget), plus SLA windows.

* **Steps**

  * **Automated triage**

    * Validate alert (guard against noise): re-compute stats on a fresh window; check seasonality/holiday effects.
    * Safety assessment: does this slice intersect safety predicates (#28)? If yes, bump severity and enforce stricter timelines.
  * **Decisioning**

    * Choose the path: **data curation only** (expand training set), **threshold/logic change** (config flip via #20), **model fine-tune**, or **full retrain**.
    * Estimate **expected lift** vs. cost: consult historical learning curves per slice.
  * **Spec authoring**

    * Produce a structured `continual_learning_trigger.yaml` including:

      * Query definition for **Scenario Mining** (#8) and **Vector Index** (#7) pulls.
      * Target label types (which heads, which ontology versions), required volume per slice.
      * Auto-labeler confidence thresholds and human QA sampling rates (#9/#10).
      * Training strategy knob (fine-tune vs. from-scratch, loss weights, data sampling ratios).
      * Gating metrics & minimal win conditions for #16 (eval) and #17 (sim).
  * **Stakeholder review**

    * Async approvers (product/safety/platform); deadline based on severity and SLAs.
  * **Kickoff**

    * Emit events to #8/#9/#10 pipeline orchestrators; create a W\&B **Project** run group to track the cycle; open budget in labeling platform.

* **Core AWS / Tooling**

  * **AppSync/GraphQL** for dataset queries, **OpenSearch** + **FAISS** for similarity pulls, **DVC** for dataset manifests, **Labelbox/Ground Truth** for QA setup, **W\&B** for lifecycle tracking.

* **Outputs & Storage**

  * Versioned `continual_learning_trigger.yaml`, mined candidate lists, label job IDs, DVC dataset tags for the forthcoming training data.
  * Event emitted to #27 (Automated Retraining) once the new data crosses minimum viable volume/quality.

---

### 27) Automated Retraining (Data → Train → Gate → Package)

* **When it runs**

  * Upon readiness signals from #26 (data volume/quality met) or on a scheduled cadence (e.g., weekly retrains with incremental data).
  * On emergency hot-fix retrains (e.g., severe false positives for emergency vehicles).

* **Inputs**

  * Curated/labeled datasets from #11 (Golden/Slice Builder) updated per trigger spec.
  * Previous best model (for fine-tuning) and training configs from #13/#14, plus any new loss weights or augmentations.
  * Compute plan (nodes, GPUs, instance types), training budget, and time window.

* **Steps**

  * **Data assembly**

    * Pull manifests via **DVC** with exact git/DVC tag; integrity check counts, class balance, per-slice minimums; run Great Expectations on schema/valid ranges.
    * Apply **sampling strategy** from trigger: overweight drifted slices but maintain global distribution constraints; snapshot as `dataset_manifest.version.json`.
  * **Training job orchestration**

    * Launch distributed training (**PyTorch DDP**) on **SageMaker** or **EKS**; enable **AMP** (BF16/FP16), gradient accumulation, and gradient checkpointing as needed.
    * **Curriculum/fine-tune** options:

      * Warm start from best checkpoint; freeze low-level backbone for a stage if compute-bound.
      * Increase loss weights for target heads/slices; introduce targeted augmentations (fog/rain, motion blur).
    * Online **W\&B** logging for metrics, LR schedules, and confusion matrices per slice; checkpoint best-of-N by main metric.
  * **Auto HPO (optional)**

    * Fire a **W\&B Sweep** for a narrow grid/Bayesian search on a few sensitive hyperparams (LR, augment strength, NMS thresholds) with ASHA early-stop.
  * **Gating & export**

    * Evaluate on **held-out** and **safety** slices (#16); block if performance regresses by more than allowed deltas on guarded slices.
    * If green, run **Packaging/Export** (#15) for the winning checkpoint to produce TensorRT/ONNX/TorchScript packs.
  * **Artifact hygiene**

    * Write `training_report.json`, checkpoints, W\&B run refs; push model packs to **S3 + ECR**; update **Model Card** (data footprint deltas, known limitations).
  * **Handoff**

    * Notify #19 to begin canary/shadow; attach drift remediation rationale to the promotion ticket (#18).

* **Core AWS / Tooling**

  * **SageMaker Training** / **EKS**, **FSx for Lustre** (optional staging), **W\&B** (runs & sweeps), **DVC**, **Great Expectations**, **TensorRT/ONNX**, **Triton**.

* **Outputs & Storage**

  * Versioned trained models, export packs, and full lineage (dataset tags → code SHA → run ID).
  * Evaluation artifacts ready for #16/#17; promotion request stub pre-filled.

---

### 28) Testing in Production (Safety Predicates & Runtime Guards)

* **When it runs**

  * Always-on in **shadow** and **canary** paths (hard gates), and in full production (soft/strict gates depending on predicate).
  * Updated whenever the predicate library or operating boundaries change.

* **Inputs**

  * Live **model outputs** (boxes, masks, tracks, trajectories, confidences), ego and CAN/IMU telemetry, environmental context (weather/map tags), and historical baseline stats from #16/#17 for bounds.
  * **Predicate library**: a versioned set of rules/constraints derived from safety analysis, simulation studies, and regulatory requirements.

* **Steps**

  * **Predicate design & encoding**

    * Express predicates as declarative policies (e.g., **OPA/Rego** or a domain-specific ruleset) with thresholds configurable per slice:

      * **Geometric sanity**: boxes within FOV, plausible aspect ratios/areas, non-negative depths.
      * **Temporal consistency**: max per-frame change in drivable area; track acceleration and jerk within physical bounds; ID switch caps.
      * **Cross-sensor agreement**: camera vs LiDAR consensus; veto if strong disagreement persists N frames.
      * **Planner consistency proxies**: large divergence between perception-based risk map and planner’s dynamic constraints flags a violation.
      * **Uncertainty guard**: abstain or degrade gracefully when confidence below a calibrated floor.
      * **Performance guard**: if p99 latency > budget, reduce batch size/enable fallback model.
  * **Deployment**

    * Run the predicate engine as a **sidecar** or in-process filter before responses are accepted by downstream consumers.
    * For **edge**, keep a tiny, deterministic rules engine with bounded memory/CPU; for **cloud**, use **OPA** sidecar with hot-reloadable policies.
  * **Action modes**

    * **Shadow**: purely record violations; do not affect caller output; route samples to S3 for audit/mining.
    * **Canary/Prod**:

      * **Soft-gate**: log + emit warning headers; allow response.
      * **Hard-gate**: replace response with **safe fallback** (prior model, rule-based heuristic, or abstention code) and flag the event.
      * **Kill-switch**: automatic rollback trigger (revert traffic weights or switch to previous production model) if violation rate exceeds M occurrences in T minutes in any protected slice.
  * **Calibration & audits**

    * Periodically validate predicate hit rates with ground truth from #10; adjust thresholds to minimize false alarms without missing true hazards.
    * Run “predicate regression tests” from the simulation library (#17) as part of the promotion pipeline.
  * **Forensics & feedback**

    * Each violation stores a **forensic bundle**: request, outputs, predicate IDs hit, policy version, traces; redact PII; store in **S3** under `/safety/violations/YYYY/MM/DD`.
    * Generate safety dashboards (violation types over time/slices) and weekly audit packs; file tickets for systemic issues.
    * Emit **mining specs** to #12 for targeted data collection (e.g., “night-rain ped crossings where cross-sensor agreement < τ”).

* **Core AWS / Tooling**

  * **EKS** sidecars (OPA), **CloudWatch Logs/Alarms**, **Athena** + **QuickSight** for safety analytics, **EventBridge** to trigger rollbacks, **App Mesh/Istio** for circuit-breakers/fallback routing.

* **Outputs & Storage**

  * Predicate decision logs, violation bundles, policy versions, and automated rollback events.
  * Feedback artifacts (mining specs) feeding the next **data → training** loop.

---

