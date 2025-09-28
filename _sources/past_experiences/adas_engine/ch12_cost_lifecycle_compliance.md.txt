# Cost, Lifecycle, Compliance

##

### 29) Cost Telemetry (unit economics, showback/chargeback, carbon)

* **Trigger**

  * Hourly CloudWatch metrics ingestion for online services; per-job hooks for training/batch; nightly CUR (Cost & Usage Report) refresh; weekly finance roll-up.
  * PR merge to main for tag compliance checks (prevents deploying untagged resources).

* **Inputs**

  * **AWS CUR** in S3 (hourly/daily granularity, resource IDs with cost allocation tags).
  * **Tags** (mandatory): `Project=ADAS`, `Env`, `WorkflowId` (e.g., `w13_training`), `DatasetTag` (DVC tag), `ModelVersion` (registry version), `Team`, `CostCenter`.
  * **Runtime counters** emitted by jobs/services: GPU hours, instance type, container image SHA, data scanned (Athena/EMR), requests QPS, p50/p90/p99 latencies.
  * **Training metadata**: W\&B runs (epochs, wall-clock, GPU type/num), SageMaker job descriptions.
  * **Carbon factors** (optional): region-level kgCO₂/kWh and GPU/CPU power draw (from `nvidia-smi` logs or instance specs).

* **Steps (with testing/validation)**

  * **(Tag enforcement pre-deploy)**

    * IaC policy as code (OPA/Conftest) validates Terraform/CloudFormation require cost tags.
    * GitHub Action fails PR if mandatory tags missing; unit test stubs check Terraform plan for tags.
  * **(Ingestion)**

    * Kinesis/Firehose → S3 for online ops metrics; EMR/Spark job normalizes to `ops_cost_metrics.parquet` (schema: `ts, resource_id, workflow_id, qps, p50, p99, gpu_util, cpu_util, mem_gb, bytes_out`).
    * CUR loader (Athena CTAS) builds **materialized views** by tags: `cur_by_workflow`, `cur_by_model`, `cur_by_dataset`.
    * Training hooks post to an SQS queue per job start/stop with `{job_id, model_version, dataset_tag, nodes, gpus, start_ts, end_ts}`; aggregator joins with CUR line items by resource ARN.
  * **(Compute unit economics)**

    * **Per-model build sheet**: `$ / epoch`, `$ / mAP point`, `$ / 1M inferences`, `$ / GB scanned`, `$ / GPU-hour`.
    * **Per-scenario unit cost**: cost to acquire + label one example for key slices (rain/night, workzone, pedestrian).
  * **(Carbon telemetry)**

    * Estimate energy = Σ (GPU power × utilization × time + instance overhead); apply region carbon factor → kgCO₂ per job; attach to W\&B run summary.
  * **(Anomaly detection & guardrails)**

    * Cost Explorer/Anomaly Detection thresholds: alert if **daily burn** for `w13_training` deviates > 2σ or **S3 retrieval** spikes (Glacier restores).
    * Policy guard: block training jobs > configured **budget per run** unless label `SpendOverride=true`.
  * **(Validation)**

    * Great Expectations checks on `cur_by_*` views: non-null tags; sum by tag == account total (±1% to allow amortized fees).
    * Reconciliation test: random sample of SageMaker jobs must appear in CUR within 48h.
    * Dashboard snapshot diffs (golden numbers for last week) to catch regressions.

* **Tooling/Services**

  * **AWS**: CUR on S3 + Athena, Cost Explorer API, Cost Anomaly Detection, CloudWatch, EventBridge, QuickSight dashboards, SageMaker APIs, S3 Inventory.
  * **Data**: EMR/Spark or Glue ETL; Athena CTAS; Parquet/ICEBERG tables.
  * **CI**: OPA/Conftest for tag rules; GitHub Actions; Checkov/tfsec for IaC.
  * **Experiment tracking**: W\&B (attach cost & carbon to runs).

* **Outputs & Storage**

  * `s3://…/governance/cost/cur_by_workflow.parquet`, `cur_by_model.parquet`, `ops_cost_metrics.parquet`.
  * QuickSight dashboards: **Model Unit Economics**, **Ops Cost & Latency Heatmap**, **Carbon per Run**.
  * Alerts in SNS/Slack: cost anomalies, budget breaches.
  * W\&B run summaries updated with `train_cost_usd`, `gpu_hours`, `kgCO2`.

---

### 30) Data Lifecycle & Tiering (retention, tiering, compaction, right-to-erasure)

* **Trigger**

  * Daily lifecycle sweep; weekly compaction/OPTIMIZE; event-driven on **access pattern** change (S3 Storage Class Analysis).

* **Inputs**

  * **Data classes**: Bronze (raw logs), Silver (synced/converted), Gold (curated/labels), Hot feature tables, Cold archives.
  * S3 Access Logs / Storage Class Analysis (object age, last-access time).
  * **Lineage graph** (Neptune/Atlas): object → derived tables/manifests (enables erasure propagation).
  * Legal requests: DSR (data subject request) manifests: `{drive_id, vehicle_id, ts_range}`.

* **Steps (with testing/validation)**

  * **(Policy definition as code)**

    * Lifecycle YAML: for each **data class**, define retention, tier transitions, encryption, replication, PII status.

      * *Example*: Bronze camera: **30 days in Standard**, then **Intelligent-Tiering**, **archive to Glacier Deep Archive** at 180 days; retain 5 years if linked to unresolved safety incident.
    * Glue Iceberg table properties: `write.target-file-size-bytes`, `commit.manifest.min-count-to-merge`, snapshot retention (e.g., keep 14 days of snapshots).
  * **(Automated actions)**

    * Create/maintain S3 Lifecycle rules + Retrieval policies; **Intelligent-Tiering** for Silver, **One-Zone-IA** for low-risk derived frames, **Glacier** for Bronze archives.
    * **Compaction/OPTIMIZE**: EMR Spark job rewrites small Parquet/ICEBERG files into large (512MB–1GB) partitions; ZSTD compression; sort by `(date, vehicle_id, sensor)`.
    * **Partition evolution**: Validate partition strategy (e.g., `dt=YYYY-MM-DD/vehicle_id=`) and update Athena/Glue.
  * **(Right-to-erasure / legal hold)**

    * DSR processor traverses lineage to locate **all derivatives** (frames, embeddings, labels); issues S3 Batch Operations delete; tombstones rows in Iceberg; updates OpenSearch documents; re-compacts affected partitions.
    * Legal hold marks objects **Non-current Version Retention** via S3 Object Lock (compliance mode) where required.
  * **(Access & cost optimization)**

    * Storage Class Analysis reports → move infrequently accessed Gold labels ≥90 days to IA; auto-restore on demand with caching.
    * **Athena workgroup budgets** and **per-query bytes scanned limits** to curb runaway costs.
  * **(Validation & safety checks)**

    * Preflight: simulate lifecycle on a **canary bucket**; ensure no Gold/Registry artifacts are expired.
    * DVC/Git pointers audit: for each `dataset_spec.yaml`, verify referenced URIs exist after compaction/moves.
    * Random restore test from Glacier weekly; measure retrieval SLA; alert on failures.
    * GDPR audit trail: every erasure creates `erasure_receipt.json` with object list, versions, timestamps.

* **Tooling/Services**

  * **AWS**: S3 (Lifecycle, Object Lock, Inventory), Glacier, Intelligent-Tiering, S3 Batch Operations, Glue/Athena, EMR Spark, Lake Formation (permissions), Macie (PII discovery), CloudTrail (audit).
  * **Catalog/lineage**: Glue Data Catalog + (optional) Atlas/Neptune for graph lineage; OpenSearch index updates.

* **Outputs & Storage**

  * `s3://…/governance/lifecycle/policy.yaml`, `compaction_reports/…`, `erasure_receipts/…json`.
  * Glue/Athena metadata reflecting latest partitions; Lake Formation grants updated.
  * Ops dashboard: **Storage by Class**, **Hot/Cold by Dataset/Model**, **Restore SLA**.

---

### 31) Security Scans (code, containers, IaC, runtime, secrets)

* **Trigger**

  * On every PR and nightly; on container build; pre-deploy gate in CD; quarterly full DAST; after critical CVE advisories.

* **Inputs**

  * Source code (Python, infra), Dockerfiles, Terraform/IaC, Helm charts/K8s manifests.
  * SBOMs; dependency lockfiles; container images in ECR.
  * Staging endpoints for API DAST.

* **Steps (with testing/validation)**

  * **(SAST & dependency audit)**

    * **Semgrep/CodeQL**: rulepacks for Python (FastAPI, boto3 misuse, deserialization), Rego, Terraform.
    * **Bandit** for Python; **pip-audit**/**Safety** for Python dependencies; block on critical issues.
  * **(Containers & SBOM)**

    * **Trivy/Grype** image scan; fail on CRITICAL CVEs (non-ignored); enforce non-root user, read-only filesystem, drop CAPs.
    * **Syft** SBOM (CycloneDX SPDX) published as artifact; attach to model package in registry.
  * **(IaC & policy)**

    * **Checkov/tfsec** for Terraform; **cfn-nag** if CFN present; **Conftest (OPA)** enforces:

      * Encryption at rest (S3, EBS, RDS), TLS 1.2+, private subnets for GPU nodes, SG least privilege.
      * Mandatory cost tags; disallow `0.0.0.0/0` ingress to control planes; deny public S3 ACLs.
  * **(Secrets hygiene)**

    * **Gitleaks**/git-secrets on diffs; pre-commit hooks strip secrets.
    * CI verifies secrets only from OIDC-assumed roles; no long-lived keys in repo or container layers.
  * **(DAST & API posture)**

    * **OWASP ZAP** active scan against staging APIs (rate-limited); **k6** smoke load to ensure auth flows work under scan.
    * TLS checker (sslyze) validates cipher suites; HTTP security headers lint.
  * **(Runtime hardening)**

    * EKS/ECS task defs: seccomp `RuntimeDefault`, AppArmor (if supported), read-only root, tmpfs for `/tmp`, resource limits set.
    * **AWS Inspector** on instances/containers; **GuardDuty** and **Security Hub** aggregation.
  * **(Compliance pack & exceptions)**

    * Findings triage to Jira; risk acceptance workflow with expiry; exception registry in codeowners file.
    * Weekly roll-up: open vs. closed issues, MTTR, trend.

* **Tooling/Services**

  * **CI/CD**: GitHub Actions; CodeQL; Semgrep; Gitleaks; Trivy/Grype; Syft; Bandit; pip-audit; Checkov/tfsec; Conftest/OPA; OWASP ZAP; k6.
  * **AWS**: ECR scan, Inspector, GuardDuty, Security Hub, IAM Access Analyzer.

* **Outputs & Storage**

  * CI artifacts: `sast_report.sarif`, `dependency_vulns.json`, `sbom.cyclonedx.json`, `container_scan.json`, `iac_scan.json`, `zap_report.html`.
  * Security Hub/Inspector findings; Jira tickets; exception register `security/risk_acceptances.yaml`.

---

### 32) Datasheets & Model Cards (governance, transparency, sign-off)

* **Trigger**

  * After **Eval & Robustness** (#16) completes and a candidate is marked **ready-for-promotion**; on **Promotion** (#18) a final signed snapshot is minted; regenerate if training/config changes.

* **Inputs**

  * W\&B run metadata (hyperparams, metrics, artifacts), evaluation reports (per-slice metrics, calibration, robustness), training config YAML, dataset `slices.yaml` + datasheet, drift & bias audits (Evidently/GE), safety predicate versions, cost/carbon summary from #29, compliance attestations (security scans, PII checks), lineage (code SHA, container digest, data versions).

* **Steps (with testing/validation)**

  * **(Template render)**

    * Jinja2 templates for **Datasheet for Datasets** and **Model Card**; sections:

      * **Intended Use & Limitations** (operational domain, weather/time/sensor assumptions).
      * **Training Data** (provenance, size, class/condition balance, label sources: auto vs. human, QA rates).
      * **Evaluation**: overall & **slice metrics** (night/rain/workzone), error taxonomies, calibration plots, failure exemplars.
      * **Robustness**: perturbation tests (jpeg, blur, occlusion), drift sensitivity.
      * **Fairness/Compliance**: bias tests relevant to domain; privacy notes; applicable standards (e.g., cybersecurity controls).
      * **Operational**: latency/throughput envelopes, memory/compute footprint, dependency SBOM hash.
      * **Cost/Carbon**: \$\$ per epoch/run, kgCO₂ per training.
      * **Safety Predicates**: policy IDs enforced, thresholds, fallback behavior.
      * **Change Log**: deltas vs. prior version; migration notes.
  * **(Artifact gathering)**

    * Pull plots from W\&B; export as PNG; embed metrics tables from Athena queries; attach scan reports’ summaries (CVEs=0 critical).
    * Link to dataset datasheet: includes **collection methods**, **preprocessing**, **labeling guidelines**, **known gaps** (e.g., low snow coverage), **retention policy** (from #30).
  * **(Automated checks)**

    * **Completeness linter**: every required section present; numeric fields non-null; footnotes linkable.
    * **Consistency**: model hash in card matches registry entry & container digest; dataset tag matches DVC tag; metrics match evaluation report (checksum).
  * **(Approvals & signing)**

    * Codeowners-based reviewers: ML lead, Safety lead, Security, Product. GitHub PR “model-card-vX.Y.md”.
    * On approval: CI stamps **attestation** (in-toto SLSA provenance), signs with KMS; stores signed PDF/HTML.
  * **(Distribution & discoverability)**

    * Publish HTML to internal portal (S3 static hosting behind IAM/ALB); attach to **Model Registry** entry; persist link in W\&B run.
    * API endpoint `GET /model/{version}/card` returns signed snapshot; hash logged in promotion record.

* **Tooling/Services**

  * **Content**: Jinja2, Pandas, Matplotlib/Plotly for visuals.
  * **Tracking**: W\&B Artifacts; DVC; Git for versioning.
  * **Signing**: in-toto attestation, AWS KMS; SLSA provenance (optional).
  * **Registry**: Your model registry (SageMaker/MLflow-compatible) enriched with `model_card_uri`, `datasheet_uri`.

* **Outputs & Storage**

  * `s3://…/governance/model_cards/model_vX.Y/model_card.md|html|pdf` (signed), `datasheets/dataset_<tag>.md`.
  * Links recorded in Model Registry + W\&B; checksums in `promotion_record.json`.

---

