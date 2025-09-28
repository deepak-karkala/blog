# ML Problem Framing

##

### 1) Understanding the Business Objective

**Objective:** deliver safer, more reliable driver-assistance for cars and trucks by increasing perception robustness and reducing intervention events—without exploding cloud costs or iteration time.

**Stakeholder alignment:**

* **Product:** safety KPIs first, low-latency UX (real-time alerts, <100–150 ms p95 from sensor ingest to decision).
* **Operations/Program:** predictable rollouts, gated by safety thresholds and audits.
* **Data/ML:** fast iteration loops, reproducible experiments, clear failure mining for edge cases.
* **MLOps/Infra:** scalable pipelines, cost controls, observability, and security-by-default.
* **Compliance/Legal:** privacy, traceability, explainability commensurate with safety review.

**Explainability baseline:** agree that perception models must support *post-hoc* explanations for incident review (e.g., SHAP for tabular/CAN features, saliency/Grad-CAM for vision). Define **reference baselines** (e.g., median-frame conditions) to anchor explanations and audits.

---

### 2) Is Machine Learning the Right Approach? (Use-Case Evaluation)

**Why ML here:** perception across cameras/lidar/radar is a high-dimensional pattern recognition problem; rules alone are brittle. Failure modes evolve (weather, construction, vehicle styles), requiring **continual learning**.

**When ML wins (our case):**

* Complex, non-linear patterns (multi-sensor fusion).
* Scale (millions of frames; long operational horizon).
* Evolving environment (drift) → retraining required.

**Guardrails & baselines:**

* Start with robust non-ML heuristics for sanity checks (e.g., speed gates, sensor drop detection).
* Gate ML predictions with safety predicates and **shadow mode** before activation.

**Data Engine requirement:** design a closed loop to **collect → curate → label → train → deploy → monitor → retrain**, with fleet triggers and targeted data mining to shorten time-to-improvement.

---

### 3) Defining the ML Problem

**Product ideal outcome:** fewer risky situations and fewer driver interventions at the same or better comfort level.

**Model goals (decomposed):**

* **Perception (primary):** multi-task detection & segmentation (vehicles, pedestrians, cyclists, lanes, drivable area), depth/BEV occupancy, and tracking.
* **Event understanding (secondary):** cut-ins, harsh braking ahead, lane closures, stationary hazards.
* **Confidence & uncertainty:** calibrated scores feeding planners/alerts.

**Inputs (multi-modal):**

* **Camera** (multi-view video), **Radar/Lidar** (where available), **IMU/GNSS/CAN**.
* **Context features:** weather/time, road class, speed limits (if available).

**Outputs & task types:**

* **Object detection/instance segmentation** (multilabel, multi-class).
* **BEV occupancy/semantic map** (dense prediction).
* **Time-to-collision / proximity risk** (regression), **event flags** (classification).

**Key issues to address:**

* **Long-tail & class imbalance:** rare but high-impact scenarios (night rain, occlusions, construction workers directing traffic).
* **Domain shift/drift:** seasonal/weather/geographic variation.
* **Label scarcity:** use auto-labeling, weak supervision, and targeted human QA.
* **Throughput & latency:** train at scale; serve under tight budgets and on constrained edge targets where applicable.

---

### 4) Assessing Feasibility & Risks

**Data readiness**

* Sufficient raw sensor coverage; rare events under-represented → plan **trigger-based mining** and similarity search.
* Labeling cost high → **hybrid strategy** (auto-label + human verification on hard slices).

**Technical constraints**

* **Latency:** real-time perception budget (<30–60 ms per frame for core heads; overall pipeline <100–150 ms p95).
* **Memory/compute:** fit models to deployment targets (quantization/distillation where needed).
* **Robustness:** enforce sensor health checks; degrade gracefully to fewer modalities if a sensor drops.

**Operational risk**

* Safety gating (shadow, A/B with small percentages, rollback).
* Strict experiment traceability (datasets, code, hyperparams) with **Weights & Biases (WandB)** for runs/artifacts, and Git/DVC (or equivalent) for data/model lineage.

**Cost & ROI**

* Control training/inference cost with spot/managed schedules, data pruning, tiered storage, and on-demand auto-labeling only for “high-value” clips.

**Ethics & compliance**

* PII handling (faces/plates blurring where required), audit trails, incident review packs with explanations.

---

### 5) Defining Success Metrics (Business, Model, Operational)

| Type                   | Metric                       | Definition                                              | How to Measure                                                                  | Target (initial)                 |
| ---------------------- | ---------------------------- | ------------------------------------------------------- | ------------------------------------------------------------------------------- | -------------------------------- |
| **Business**           | **Intervention rate ↓**      | Driver takeovers per 1000 km in assisted modes          | Fleet telemetry and event logs, normalized by km and conditions                 | **−15–20%** vs. baseline cohort  |
| **Business**           | **Safety-critical events ↓** | Near-miss / harsh-brake alerts per 1000 km              | On-vehicle triggers (brake pressure, decel, proximity) with post-hoc validation | **−15–22%**                      |
| **Business**           | **Feature reliability ↑**    | % sessions without faulted ADAS disengagement           | Session analytics, error codes, watchdog                                        | **+10–15%**                      |
| **Model (Perception)** | **Primary mAP / mIoU ↑**     | Detection mAP\@IoU, seg mIoU on curated “hard” sets     | Benchmarks by scenario slices (night/rain/occlusion/construction)               | **+5–8 pts** on hard slices      |
| **Model (Risk)**       | **Calibration (ECE) ↓**      | Expected Calibration Error for confidence outputs       | Reliability diagrams on offline eval & shadow data                              | **< 3–5%** ECE                   |
| **Model (Long-tail)**  | **Slice recall ↑**           | Recall on rare scenarios (e.g., night-rain pedestrians) | Per-slice eval sets maintained in W\&B Artifacts                                | **+10–15 pts**                   |
| **Operational**        | **TTMU ↓**                   | Time from failure discovery → safe model in prod        | Track via ticket timestamps and deploy tags                                     | **8–10 w → 2–3 w**               |
| **Operational**        | **Pipeline SLA**             | Ingestion→curation→label turnaround time per drive      | Orchestrator metrics, queue latencies                                           | **< 24 h**                       |
| **Operational**        | **Serving latency**          | End-to-end p95 inference latency                        | Traces (Cloud + edge), p50/p95/p99                                              | **< 100–150 ms p95**             |
| **Operational**        | **Cost efficiency**          | \$/1000 km processed; \$/training experiment            | Cost dashboards; W\&B sweep cost per win                                        | **−20–30%** vs. baseline quarter |
| **Operational**        | **Observability coverage**   | % models/pipelines with alerts, dashboards, SLOs        | Runbooks + monitoring inventory                                                 | **> 85%** coverage               |

**Notes on measurement**

* **Business metrics** are evaluated on held-out test routes and staggered rollouts to control for route/weather mix.
* **Model metrics** are tracked in **WandB** (projects for perception/risk; Artifacts for datasets, models, and eval sets).
* **Operational metrics** come from pipeline orchestration, tracing, and cost dashboards; tie each notable change to a W\&B run/deployment tag for auditability.

---

**Decision summary:** ML is necessary and appropriate for the perception and event-understanding layers, provided it is embedded in a **closed-loop data engine** with strong safety gating, targeted long-tail mining, reproducible experimentation (**WandB**), and rigorous operational controls.

