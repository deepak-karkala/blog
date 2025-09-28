# Reliability, Capacity, Maps

##

### 33) Incident RCA (Root Cause Analysis) — serving & pipeline reliability

* **Trigger**

  * Any of: production SLO/SLA breach (latency, error rate), safety predicate trip, anomaly alert from monitoring/drift (#25), simulation regression from pre-prod, canary rollback, repeated pipeline failures, or on-call/PagerDuty page.
  * Scheduled post-incident review within 72 hours for any SEV-1/SEV-2.

* **Inputs**

  * **Telemetry & traces:** CloudWatch metrics/logs, OpenSearch logs, Prometheus/Grafana dashboards, AWS X-Ray/Jaeger traces, NVML/DCGM GPU telemetry, feature-store freshness metrics.
  * **Change context:** Deployment events (Git SHA, container digest, config/flag deltas), model registry history (candidate → staging → prod), feature definitions, safety predicate versions.
  * **Data signals:** Request samples, mispredictions flagged by online validators, user-reported issues, shadow-mode diffs, recent drift reports.
  * **Artifacts:** Last successful build/run logs, canary analysis reports, A/B analysis, W\&B run metadata.

* **Steps (with testing/validation)**

  * **Immediate triage (T+0 to T+30min)**

    * Declare incident, assign IC (incident commander) and scribe; set severity; start timeline.
    * Freeze risky changes (deployment lock) and **engage runbooks** (safe rollback primitives).
    * Capture **context snapshot** automatically: last N deploys, feature-flag changes, top error signatures, p99/p999 latency delta, affected tenants/regions.
  * **Stabilize**

    * Execute **rollback or traffic shift** to last-good model/service (canary controller); validate health via smoke tests and golden synthetic checks (known-request replay must pass).
    * If feature-store or data pipeline is culprit: fail over to **degraded mode** (fallback features, cached responses, or heuristic policy).
  * **Data capture for forensics**

    * Quarantine a redacted sample of failing requests, feature vectors, and predictions (S3 `governance/incidents/<id>/samples/`); include traces, safety decisions, and model confidences.
    * Preserve relevant logs via export (CloudWatch → S3), pin dashboards.
  * **Hypothesis-driven analysis**

    * **Change correlation:** identify first bad time; align with **any change** (code, model, config, data). Use change-point detection on KPIs to narrow window.
    * **Reproduction:** re-run failing requests in a **deterministic container** with the exact model+flags; compare to last-good; run side-by-side diff.
    * **Dependency check:** upstream (feature freshness, schema drift), downstream (clients, map service).
    * **Model-centric probes:** calibration curves on failing slice, confusion matrix deltas, SHAP drift vs. baseline, feature importance changes.
  * **Root cause determination**

    * Classify: **Code defect**, **Model regression**, **Config/flag error**, **Data/feature drift**, **Infra capacity** (noisy neighbor, GPU ECC, throttling), **Map/trigger policy**.
    * Quantify blast radius (requests, segments, geos), cost impact, safety impact.
  * **Corrective & Preventive Actions (CAPA)**

    * Immediate fix (patch, hotfix, config revert), plus **long-term guardrail** (test, monitor, lint rule, rollout constraint).
    * Create **issue tickets** with owners & due dates. Integrate with CI gates (e.g., block deploy if schema version mismatch).
  * **Validation**

    * Post-fix **replay**: reproduce pre-incident failing cases → verify pass; run targeted load to confirm capacity headroom.
    * Add **new regression tests** (golden scenario) to simulation and eval suites; require pass before future promotions.
  * **Postmortem**

    * Write blameless RCA using template (5 Whys, fishbone); include timeline, contributing factors, detection gaps, MTTR/MTTD.
    * Review in weekly reliability review; track action items to closure.

* **Core Tooling/Services**

  * PagerDuty/Incident.io, CloudWatch/Logs/Alarms, OpenSearch, Prometheus/Grafana, AWS X-Ray/Jaeger, AWS CodeDeploy events, Feature-store metrics, W\&B, Athena/QuickSight for KPI drilldowns, Jupyter for ad-hoc analysis.

* **Outputs & Storage**

  * `s3://…/governance/incidents/<incident_id>/` (samples, dashboards, reports), **RCA document** (Markdown/PDF), Jira tickets, updated runbooks & tests, promotion gate updates.

---

### 34) Experiment GC (Garbage Collection) — artifacts, indices, and datasets hygiene

* **Trigger**

  * Weekly scheduled GC; **low free space** alert; **budget threshold** exceeded for storage/egress; repo archival; project sunset tag.
  * Manual **quarantine → purge** for compromised or incorrect datasets.

* **Inputs**

  * **Inventory sources:** S3 Inventory (per-bucket), Glue/Athena tables, Iceberg snapshots, DVC remotes & tags, W\&B runs/artifacts, ECR images/tags, OpenSearch indices, EMR logs, FSx/Lustre volumes.
  * **Usage signals:** Access logs (S3/Athena), last-read timestamps from index services, registry **in-use** pointers (current prod/staging models, golden datasets).
  * **Policies:** `retention_policy.yaml` (per class: Bronze/Silver/Gold), legal holds, exception lists, minimal-keep (e.g., N best runs per model).

* **Steps (with testing/validation)**

  * **Discovery & reachability**

    * Build **lineage graph**: artifact → consumers (models, datasets, docs). Anything “unreached” and older than policy horizon becomes a **candidate**.
    * Join with **usage stats** (no access in ≥N days) and **cost** (size × storage class).
  * **Protection rules**

    * Always protect: models **referenced by registry channels** (prod, canary), **golden datasets**, signed model cards/datasheets, compliance snapshots, incident forensics.
    * Legal holds override GC; DSR erasure queues take precedence.
  * **Action plan**

    * **S3**: batch delete candidates; transition to Glacier for keep-but-cold.
    * **Iceberg**: expire snapshots ≥ horizon; **rewrite manifests**; **vacuum** orphan files.
    * **W\&B**: delete old runs/artifacts except top-K per sweep by metric; export summary CSV first.
    * **ECR**: apply lifecycle policy (keep last M per repo & any tagged `stable`, delete dangling layers); scan for large bases to dedupe.
    * **OpenSearch**: apply ISM (Index State Mgmt) to roll over & delete old indices, or **shrink** & **forcemerge** if kept.
    * **Logs**: compress EMR/YARN logs; purge older than horizon.
  * **Safety checks**

    * **Dry-run** report (bytes to free, candidates count) → human approve for destructive steps.
    * Referential **integrity check**: no model or dataset manifest points at an about-to-delete URI.
    * **Restore drill**: pick 1% random deleted-to-Glacier objects and ensure restore works within SLA.
  * **Execution**

    * Orchestrate via Airflow/Step Functions with idempotent tasks and checkpointing; track failures & retries.
  * **Validation**

    * Post-GC audit: **Athena** reconciliation (sum sizes by class), check that dashboards & registry remain healthy.
    * Alert on “unexpected reference” errors if any job fails due to a missing artifact.

* **Core Tooling/Services**

  * S3 Inventory/Batch Ops/Glacier, Glue/Athena, EMR Spark for Iceberg maintenance, W\&B API, ECR lifecycle, OpenSearch ISM/Curator, Airflow/Step Functions, Jira for approval workflow.

* **Outputs & Storage**

  * `s3://…/governance/gc/reports/<date>.json`, deletion manifests, restored-object test logs, storage savings dashboard; policy & exception registry in Git.

---

### 35) GPU Capacity & Queues — scheduling, reservations, autoscaling, fairshare

* **Trigger**

  * Continuous: new training/HPO workloads submitted (Airflow/W\&B Sweeps); scheduled **capacity planning** (weekly); **queue-depth** or **wait-time** SLO breach; quota changes; new model roadmap requiring different accelerators.

* **Inputs**

  * Historical job metadata (GPU type/count, wall-clock, throughput), cluster utilization (DCGM, Prometheus), job queue stats (length, age), **SageMaker** job history, instance pricing (on-demand/spot), Savings Plans/RIs, project priorities, SLAs (e.g., “critical sweep completes in ≤48h”), dataset size and required I/O.
  * Node pool definitions (A100/H100 vs T4/L4; CPU-only for preprocessing); storage bandwidth (FSx/Lustre, S3).

* **Steps (with testing/validation)**

  * **Demand forecasting**

    * Time-series model predicts GPU-hours by queue for next 1–4 weeks; identify **peak weeks** and confidence intervals.
    * Scenario overlay: planned HPO waves, retrain cadence tied to drift alarms.
  * **Capacity planning**

    * Map demand to **node pools** (labels/taints): `gpu=A100`, `gpu=L4`, `cpu-prep`.
    * Purchase/adjust **Savings Plans**; set **SageMaker managed spot** %; reserve high-risk windows (e.g., releases).
    * Validate **data-path throughput**: if bottlenecked on I/O (S3→FSx), increase FSx/Lustre capacity, pre-stage datasets.
  * **Queueing & policy**

    * **Kueue/Volcano/Slurm** or SageMaker priorities: project quotas (GPU-hour budgets), **fairshare weights**, **preemption** rules for interruptible sweeps.
    * Admission controller enforces **budget tags & approvals** for large jobs; refuses jobs that exceed per-run budget unless label `SpendOverride`.
  * **Autoscaling & bin-packing**

    * **Karpenter/Cluster Autoscaler** spins node groups on demand; prefer bin-packing (anti-fragmentation), GPU topology aware (NVLink domains).
    * For spot, enable **checkpointing** every N minutes to S3/FSx; preemption handler requeues gracefully.
  * **Placement & topology**

    * Enable **NCCL topology hints**; rack-aware placement; multi-nic configs for multi-node DDP.
    * Node-affinity to place data-heavy jobs close to **FSx/Lustre**; avoid overloading S3 (throttle prefetchers).
  * **SLO management**

    * SLOs: **median wait time**, **p95 wait time**, **train throughput (img/s)**, **GPU util %**, **cost/GPU-hour**. Alert on breach.
    * If queue depth persists: auto-scale upper bound (if budget allows) or **shed load** (pause low-priority sweeps).
  * **Validation**

    * **Load simulation**: synthetic job submissions to validate queue fairness & SLOs.
    * **Failover drill**: zone outage simulation; ensure jobs reschedule; checkpoints recover.
    * **Throughput tests**: for each node pool, run standard training microbenchmarks; track regression over time.
  * **Governance**

    * Monthly **capacity review**: showback/chargeback per team; renovate underused images; deprecate old node pools.

* **Core Tooling/Services**

  * EKS (Kubernetes), Karpenter/Cluster Autoscaler, NVIDIA DCGM exporter + Prometheus/Grafana, Kueue/Volcano/Slurm or SageMaker managed training, FSx for Lustre, S3, Ray for distributed sweeps, EventBridge/Lambda for policy actions, Cost Explorer API.

* **Outputs & Storage**

  * Capacity plan (`capacity_plan_<month>.md`), queue configs (ConfigMaps/Slurm partitions), utilization dashboards, Savings Plan decisions, checkpoint manifests, SLA reports; all versioned in Git and logs in S3 governance.

---

### 36) Map/Trigger Policy Update — updating HD map layers & fleet trigger definitions

* **Trigger**

  * Periodic map refresh (e.g., weekly); **policy change** from safety team; evidence from scenario mining (#8/#12) showing gaps; external roadway updates (work zones, new speed limits); spike in false positives/negatives for specific trigger definitions.

* **Inputs**

  * **Map data deltas** (internal mapping pipeline outputs, vendor feeds, or OSM diffs), lane topology changes, speed limit updates, construction zones, geofences.
  * **Trigger performance**: alert rates, precision/recall of triggers (e.g., hard-brake, disengagement proximity), geographic breakdowns.
  * **Scenario feedback**: mined error clusters, audit results from Human QA (#10), simulation outcomes from drive replay/closed-loop (#17).
  * **Constraints**: ODD boundaries, regulatory requirements, privacy constraints.

* **Steps (with testing/validation)**

  * **Propose & author changes**

    * Author **policy YAML** (semver): includes **map layer updates** (road attributes, closures) and **trigger definitions** (thresholds, state machines, OOD/uncertainty bounds, geofenced overrides).
    * Attach rationale, expected impact (alerts/day, coverage gain), and risk assessment.
  * **Offline evaluation**

    * **Backtest** on last N weeks of logs: compute precision/recall, alert volume, **regional heatmaps**; confirm reduced false alarms or improved recall on target scenarios.
    * **Counterfactuals** in sim: vary thresholds, verify **safety balance** (miss rate vs. nuisance rate).
    * Verify **lat/long** accuracy (map matching via OSRM/Valhalla); ensure no regressions at map tile boundaries.
  * **Schema & consistency checks**

    * Validate policy schema (JSON Schema); enforce allowed ranges; check for **conflicting overrides** across geofences.
    * Ensure **version compatibility** with edge agent and cloud detectors (backwards/forwards).
  * **Security & signing**

    * Sign policy bundle with KMS; attach **in-toto** attestation; generate SBOM for any included logic plugins.
  * **Staged rollout**

    * Publish to **S3 policy bucket** (immutable path per version); create **IoT Jobs**/Greengrass deployment targeting a **canary cohort** (small % of fleet or specific region).
    * Enable **feature flags**: `trigger_policy.version`, `map_layer.version`, with kill-switch.
  * **Canary monitoring**

    * Watch alert rates, map match errors, CPU/mem impact on edge, OTA download success rates, and any safety predicate changes; compare to control cohort.
    * Roll forward if within guardrails; roll back on anomalies (auto if thresholds breached).
  * **Full rollout & enforcement**

    * Gradually increase cohort; record final adoption; ensure backend **parsers** accept new tags/fields; update catalog ETL if schema changed.
  * **Validation**

    * Weekly **policy audit**: recompute metrics; ensure no drift between **edge** and **cloud** policy versions; verify **replay** on key scenarios passes.
    * **Documentation**: update policy change log, trigger explanations for labelers/engineers.

* **Core Tooling/Services**

  * Geo stack: OSRM/Valhalla, GeoPandas/Shapely; data lake (Athena/OpenSearch) for backtests; AWS Location Service (optional); IoT Jobs/Greengrass for OTA; S3 static policy hosting; KMS for signing; Feature-flag service; QuickSight/Mapbox for heatmaps.

* **Outputs & Storage**

  * `s3://…/policy/map/<semver>/bundle.tar.gz`, `trigger_policy/<semver>/policy.yaml` (signed), **impact report** (before/after metrics, maps), rollout dashboard, change-log; links recorded in registry & internal portal.

---