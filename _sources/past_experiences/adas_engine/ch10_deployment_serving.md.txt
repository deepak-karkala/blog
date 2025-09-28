# Deployment & Serving

##

### 19) Canary / Shadow Deployment

* **When it runs**

  * Immediately after **Registry & Promotion** (#18) marks a candidate as “ready-to-deploy”.
  * Also on demand for hotfixes and security-patch rebuilds of serving containers.

* **Inputs**

  * Versioned, signed model packs and serving container(s) from **Packaging** (#15)—TorchScript/ONNX/TensorRT with `config.pbtxt` and `inference_config.json`.
  * Model Registry record (artifact digests, semver, dataset/hash lineage).
  * Rollout policy (canary steps, shadow sampling %, abort thresholds).

* **Steps**

  * **Environment prep**

    * Provision/confirm **multi-AZ** EKS or SageMaker Endpoints; ensure VPC-only networking, VPC endpoints for S3, and TLS everywhere.
    * Warm capacity for both **shadow** and **canary** paths (separate HPA targets to isolate load).
  * **Shadow mode (read-only)**

    * Mirror a configurable % of real production traffic to the **shadow** model while keeping responses dark (not used by callers).
    * Log shadow outputs, latencies, and **behavioral diffs** vs. production to S3; compute summary KPIs (class-level recall/precision deltas, NMS stability, trajectory ADE/FDE deltas).
    * **Validation gates:** cap diff rates (e.g., abs(Δ recall pedestrian\_night) ≤ 1.5%); watch p95 latency. Auto-stop shadow if anomalies exceed pre-set budgets.
  * **Canary (serve a fraction of live traffic)**

    * Route a small cohort (e.g., 1%) to the candidate via ingress or SageMaker variant routing weights.
    * Enable **gray** logging: store complete requests + responses for the canary cohort, with PII redaction.
    * **Health & SLO checks:** request success rate, p95/p99 latency vs. SLO, GPU memory headroom, error budgets.
    * Increase traffic in steps (1% → 5% → 25% → 50% → 100%) only after each step maintains green KPIs for N minutes/hours.
  * **Abort / rollback path**

    * Instant rollback to previous production image via **blue/green** or revert variant weights.
    * Preserve failure bundle (requests, traces, metrics) to S3 for **Offline Mining** (#12).
  * **Documentation & sign-off**

    * Append canary/shadow results to the model card and Registry entry.
    * Notify stakeholders with a concise status page (live KPI tiles and roll-forward/rollback decision log).

* **Core AWS / Tooling**

  * **EKS** (Triton/TorchServe pods), **ALB/NLB** Ingress, **SageMaker Endpoints** (variant weights), **App Mesh/Istio** for traffic shaping, **CloudWatch** alarms, **EventBridge** for step promotions.
  * **OpenTelemetry** for traces, **Prometheus/AMP** + **Grafana** for SLOs, **S3** for shadow logs and diffs, **W\&B** for deployment run metadata.

* **Outputs & Storage**

  * Canary/shadow KPI reports, diff summaries, traces; stored in **S3** and linked in Registry.
  * Updated Registry stage (`candidate → production`) once canary completes.

---

### 20) A/B Testing & Feature Flags

* **When it runs**

  * After canary when we want outcome-level proof (business or safety proxy KPIs).
  * During experiments that tune thresholds, ensemble weights, or post-processing steps without retraining.

* **Inputs**

  * Deployed production and candidate models (or the same model with different **post-processing/threshold configs**).
  * **Experiment Plan**: primary metric(s), success criteria, sample size/power calculation, guardrails (safety, latency).

* **Steps**

  * **Flag & cohort design**

    * Define **treatment arms** (e.g., Threshold\_A vs Threshold\_B; Model\_v1.8 vs v1.7).
    * Cohort users/vehicles by geography, time window, or fleet slice to minimize interference.
    * Implement with a **config/flag service** (DynamoDB or LaunchDarkly) read at request start. Cache locally with short TTL to avoid flag server coupling.
  * **Routing & consistency**

    * Sticky assignment per device/vehicle to avoid cross-over contamination.
    * Keep **feature parity** across arms except for the variable under test.
  * **Metrics capture**

    * Online KPIs (success rate, false-positive interventions, latency p95) plus **safety proxies** (e.g., disagreement with planner, emergency brake proxy rates).
    * Aggregate with exact timestamps and cohort tags; anonymize IDs at the logger.
  * **Statistical analysis**

    * Sequential testing or fixed-horizon with correction for multiple looks; pre-register the test to avoid p-hacking.
    * Guardrail checks: if any safety guardrail breaches, auto-terminate the test and revert flags.
  * **Decision & rollout**

    * Promote winning config/model by flipping flags globally or per-slice; persist final config to **inference\_config.json** next release cycle.
    * Archive experiment results (effect size, confidence intervals, power achieved) in the Registry.

* **Core AWS / Tooling**

  * **DynamoDB** (flag store) or **LaunchDarkly**, **AppConfig**, **EventBridge** for change broadcasts.
  * **Athena/Glue** + **QuickSight** for analysis; **W\&B** to attach experiment metadata to model version.

* **Outputs & Storage**

  * `ab_summary.json`, dashboards, and final flag state in **DynamoDB/AppConfig**; linked to Registry and model card.

---

### 21) Edge Build & OTA Packaging (Vehicle/Device)

* **When it runs**

  * After cloud serving passes canary and we’re ready to produce **edge-optimized** builds.
  * On periodic runtime refreshes (driver version change, security patches) or new hardware SKU support.

* **Inputs**

  * Model engine(s) per target (TensorRT FP16/INT8) from #15, with calibration cache.
  * Edge runtime constraints: memory/compute budgets, power/thermal envelopes, allowable latency.
  * Device fleet manifest: hardware SKU mapping, minimum supported driver/SDK versions.

* **Steps**

  * **Cross-compile & optimize**

    * Build per-SKU **TensorRT** plans with tactic replay and builder flags aligned to target (e.g., Orin/Drive).
    * Fuse pre/post operations into CUDA plugins where beneficial; ensure zero-copy tensors across stages.
    * Run **quantization sanity** on-device emulation (QAT-aware if available).
  * **Runtime container/component**

    * Package as **Greengrass** component or OCI image with minimal base; pin CUDA/TensorRT versions; bundle `config.pbtxt` and `inference_config.json`.
    * Include a watchdog and **health endpoints**; implement local batcher and thermal-aware throttling hooks.
  * **Hardware-in-the-loop tests**

    * On a bench rig with target SoC, run **smoke suite**: contract tests, p95 latency, memory ceiling, and thermal soak.
    * **Determinism checks** at fixed seeds; performance variance bounds under thermal throttling scenarios.
  * **Security & compliance**

    * Code sign artifacts (**AWS Signer** or **cosign**) and produce a per-device **update manifest** with checksums.
    * SBOM attached; license and IP provenance validated.
  * **Release assembly**

    * Generate OTA bundle per cohort: artifact URIs, rollout policy, preconditions (battery level, vehicle parked, firmware min version), recovery strategy.
    * Publish metadata to the **OTA job catalog** (IoT Jobs/FleetWise campaign).

* **Core AWS / Tooling**

  * **AWS IoT Greengrass** components, **AWS IoT FleetWise** or **IoT Device Management** for campaigns, **S3** artifact buckets, **Signer/KMS** for signatures.
  * Bench automation: **EKS** runner or on-prem CI hardware with **GitHub Actions/CodeBuild**.

* **Outputs & Storage**

  * Signed edge bundles per SKU, update manifests, and bench reports; stored in **S3** and indexed in a **campaign DB** (DynamoDB/Registry).

---

### 22) OTA Delivery (Fleet Campaigns)

* **When it runs**

  * After edge bundles are ready and approved by safety/security leads.
  * Coordinated with operations windows (time-of-day, depot/garage schedules).

* **Inputs**

  * OTA bundles + manifests from #21.
  * Fleet segmentation (VIN/Device IDs by geography, customer, regulatory domain).
  * Rollout strategy: staged waves, max concurrent updates, stop conditions.

* **Steps**

  * **Campaign creation**

    * Define cohorts and scheduling: wave sizes, blackout periods, and retries.
    * Preconditions: device online, battery ≥ X%, connected to Wi-Fi or certain carriers, parked/ignition state.
  * **Secure distribution**

    * Ship via **IoT Jobs** with signed URIs; devices verify signature and checksum before install.
    * **Bandwidth shaping**: CDN/S3 transfer acceleration; per-region throttles to avoid network saturation.
  * **Install & verify**

    * Atomic swap: install to **A/B partition** or container tag; upon success, flip active pointer.
    * **Health probes** post-install: run a local inference self-test; send success beacon with version and basic KPIs.
    * On failure, auto-rollback to previous slot and report error codes.
  * **Monitoring & control**

    * Live campaign dashboard: started/succeeded/failed, per-region rates, error categories.
    * Pause/resume and wave-size adjustments in real time; stop campaign on thresholded failure rates.
  * **Post-deploy soak**

    * Collect **in-field telemetry**: latency/thermals, crash reports, edge-level OOD counters, and light-weight quality proxies (e.g., detection density by condition).
    * Feed anomalies to **Offline Mining** (#12).

* **Core AWS / Tooling**

  * **AWS IoT Jobs / FleetWise**, **IoT Core**, **CloudWatch**, **Athena** for campaign analytics, **QuickSight** dashboards.
  * **KMS** for artifact encryption at rest; **Private CA** for device certificates if needed.

* **Outputs & Storage**

  * Campaign status logs, per-device install receipts, post-install health beacons; all in **S3/DynamoDB**, surfaced in dashboards and linked to Registry.

---

### 23) Online Service Operations (Cloud Inference)

* **When it runs**

  * Always-on for cloud inference endpoints (batch and/or online).
  * Scales elastically with traffic; responds to deployments and load events.

* **Inputs**

  * Production model image(s), **inference\_config.json**, and feature/metadata services endpoints.
  * SLOs/SLCs: availability, p95/p99 latency, error budgets, cost-per-1k inferences.

* **Steps**

  * **Service layout**

    * **Ingress** → **Request validator** (schema, auth) → **Preprocessing** → **Model** → **Post-processing** → **Response**.
    * Optional **Feature Online Store** (Feast with DynamoDB/Redis) for feature joins; aggressive caching + TTLs.
  * **Resilience & scaling**

    * **HPA/KEDA** on GPU/CPU utilization, QPS, and queue depth; min pods to absorb cold starts.
    * Connection pools, timeouts, **circuit breakers** (Envoy/App Mesh) for downstream calls; backpressure via bounded queues.
    * **Multi-AZ**, pod disruption budgets, surge capacity for rollouts.
  * **Performance engineering**

    * Pin NUMA/GPU affinity; TensorRT/Triton dynamic batching with careful max delay.
    * Pre-allocate memory pools; enable CUDA graph capture where applicable.
    * Async I/O; zero-copy tensors; avoid per-request allocations.
  * **Security**

    * mTLS in-mesh; OIDC/JWT at edge; fine-grained IAM for S3/feature store access.
    * WAF rules for ingress, request size caps, schema enforcement, and PII redaction at loggers.
  * **Cost controls**

    * Right-size instance types, spot for batch, on-demand for online; autoscaling floors/ceilings.
    * Periodic **throughput/latency bin-packing** reviews and **mixed precision** tuning to reduce GPU ms/inference.
  * **Operational playbooks**

    * Runbooks for incident classes (latency spike, elevated 5xx, GPU OOM, feature store timeouts).
    * Synthetic probes and golden queries; regular failover/fire-drill practices.

* **Core AWS / Tooling**

  * **EKS** with **Triton/TorchServe**, **ALB/NLB**, **App Mesh/Istio**, **Feast** (DynamoDB/ElastiCache Redis), **CloudWatch**, **SQS/Kinesis** for async/batch, **SageMaker Endpoints** where managed is preferred.

* **Outputs & Storage**

  * Live responses (API), **structured logs**, **metrics**, **traces**, and **inference audit records** (S3 with lifecycle policies).

---

### 24) Observability (Telemetry, Drift, Explainability)

* **When it runs**

  * Continuously, from the moment traffic reaches shadow/canary through long-term production.
  * On scheduled jobs for deeper drift/quality analysis.

* **Inputs**

  * Request/response telemetry, model outputs, confidence histograms, selective ground truth (from human QA or auto-label confirmations), and reference statistics from #16.

* **Steps**

  * **Metrics**

    * **System**: QPS, p50/p95/p99 latency, GPU/CPU/memory utilization, queue depth, error rates (4xx/5xx).
    * **Model**: per-class score distributions, calibration ECE, acceptance/abstention rates, novelty counters (OOD flags).
    * **Data**: feature value histograms, missingness, input schema drift.
  * **Logs**

    * Structured, PII-redacted request/response logs; correlation IDs to join across services.
    * **Failure bundles**: auto-capture payload + model state for 5xx or large diffs; store to S3 with strict retention.
  * **Traces**

    * **OpenTelemetry** spans from ingress through model to downstream stores; trace sampling biased toward tail latency and errors.
  * **Dashboards & alerts**

    * Grafana/QuickSight boards by **SLO tiers**; CloudWatch alerts on SLO/SLA breaches, drift thresholds, and OOD spikes.
    * PagerDuty/Slack routes with severity mapping; include runbooks and **auto-remediation** hooks (e.g., scale-up, switch to previous model, or temporary rule override).
  * **Drift & quality analytics**

    * Daily/weekly jobs (**Airflow**) that run **Evidently** against rolling windows: covariate drift, concept drift (where labels available), PSI/KS tests per feature and per-slice.
    * **Canary sentinels**: raise alerts early for slices historically fragile (night + rain + pedestrian).
  * **Explainability**

    * Lightweight **SHAP-on-sample** or gradient-based saliency for a small percentage of requests in staging; store as artifacts for model debugging.
    * Maintain **model card** live sections: data slices slipping, observed biases, mitigations taken.
  * **Feedback loops**

    * Emit curated failure/novelty cohorts to **Offline Mining** (#12) with descriptors and query templates.
    * Track **time-to-mitigation** and **defect escape rate** as MLOps KPIs.

* **Core AWS / Tooling**

  * **OpenTelemetry Collector**, **AMP/Prometheus**, **Grafana**, **CloudWatch** (metrics/logs), **Athena/Glue** for large-scale log queries, **Evidently** for drift, **W\&B** for attaching production metrics to model versions.

* **Outputs & Storage**

  * Time-series metrics, traces, and logs in AMP/CloudWatch + **S3** data lake; drift reports; incident tickets; curated error cohorts for the next training loop.

---

