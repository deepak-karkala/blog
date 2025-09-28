# Scene Understanding & Data Mining

##

Perfect—here’s a clear, implementation-ready expansion of **Workflows 6–12: Scene Understanding & Data Mining**. I’ve kept each workflow consistent in structure so your team can lift this straight into design docs and tickets.

---

### 6) Scene Detection & Triggers

**Trigger**

* New drive finishes Foundations (Bronze → Silver), or a **telemetry trigger** arrives (e.g., ABS, disengagement, harsh brake).

**Inputs**

* Time-synced Silver assets: camera keyframes/clips, LiDAR sweeps/packets, CAN/IMU time series.
* Precomputed detections from Foundations (e.g., coarse lanes/objects).
* Map tiles/metadata (road class/intersection).
* Shadow vs prod model outputs (if available).

**Step-by-step (with guardrails)**

1. **Segment** the drive into candidate scenes

   * Adaptive segmentation on change-points: speed/accel deltas, heading/yaw rate, road topology transitions, stop→go, intersection proximity.
   * Merge/split logic: enforce min scene length (e.g., 3–60s), bridge micro-gaps (<500 ms).
2. **Detect & enrich**

   * Lightweight TensorRT/TorchScript models for 2D objects, lane edges, traffic-light state; ego-event heuristics from CAN/IMU (cut-in, hard brake, tailgating).
   * Optional OOD/uncertainty probes (entropy, ODIN) and **prod vs shadow disagreement** hooks.
3. **Score interestingness**

   * `score = α·rarity + β·uncertainty + γ·disagreement + δ·diversity_margin`.
   * Rarity from rolling histograms per slice (weather/time/geo).
4. **Validate**

   * **Great Expectations**: schema, timestamp monotonicity, frame-rate bounds, sensor coverage per scene.
   * Sanity plots sampled to S3; spot-check top-N scenes by score.
5. **Emit triggers**

   * Push `trigger_flags` (e.g., `disengagement=true`, `ood=true`) to EventBridge/SQS for downstream miners.

**Outputs**

* `scene_segments.parquet` fields:

  ```
  {scene_id, drive_id, vehicle_id, start_ts, end_ts, clip_uri[], tags[], trigger_flags[], score}
  ```
* `scene_events.parquet` (per-event rows with attributes, e.g., {type, ts, value, confidence}).

**Storage & Indexing**

* **S3 Silver** (Parquet, partitioned by dt/vehicle\_id).
* **Glue/Athena** external tables for analytics.
* **DynamoDB**: per-scene manifest (fast lookups).
* **OpenSearch**: scene docs for keyword/facet search.

**Core tooling**

* **Airflow** DAG → **EMR Spark** or **AWS Batch** containers; **Great Expectations**; **OpenSearch**; **DynamoDB**; **Weights & Biases** (runs + artifacts).

---

### 7) Vector Index (Similarity Search)

**Trigger**

* Batch completion of #6 (new scenes available).

**Inputs**

* Scene keyframes (images), clip thumbnails, LiDAR BEV features, scene tags/notes.

**Step-by-step (with guardrails)**

1. **Embed**

   * Images: CLIP ViT-B/32 (or ResNet-50) embeddings per keyframe; optionally average over clip.
   * LiDAR: BEV encoder or pooled PointNet++ features.
   * Text: sentence embeddings (tags/notes).
   * Concatenate or keep **modality-specific** indices (recommended).
2. **Normalize & reduce**

   * L2-normalize; optional PCA→256D; whitening for dense IVF/PQ.
3. **Index build**

   * **FAISS**: IVF-PQ (nlist, m, nbits tuned to latency), or
   * **OpenSearch k-NN**: HNSW (M, ef\_construction), per-scene document with vector field.
4. **Validate**

   * Retrieval smoke tests (query-by-example must return near-duplicates).
   * Offline **mAP\@K** and duplicate recall; log to **W\&B**.
   * **Evidently** drift on embedding stats (mean/var, cov trace); alert on large shift.

**Outputs**

* `embeddings.parquet` (uri, scene\_id, modality, vector\[]).
* FAISS artifacts (`embeddings.faiss`, `pca.npy`) **or** OpenSearch k-NN index.

**Storage & Indexing**

* **S3 Silver** (vectors + FAISS); **Glue** for vectors; **OpenSearch** for real-time k-NN search.

**Core tooling**

* **SageMaker Processing**/**EMR** for batch compute; **FAISS**/**OpenSearch k-NN**; **Evidently**; **W\&B**.

---

### 8) Scenario Mining (Programmatic / Query UI)

**Trigger**

* On-demand engineer queries, scheduled gap-analysis, or post-deployment error mining.

**Inputs**

* Scene catalog (#6), vector index (#7), telemetry triggers, map/weather joins.

**Step-by-step (with guardrails)**

1. **Unified Query** (GraphQL via **AWS AppSync**)

   * Combine: structured filters (time range, weather, road type), **OpenSearch** facets/text, and vector similarity (k-NN).
   * Example filter: `weather in [rain,snow] ∧ time_of_day=dusk ∧ tags.contains('cyclist') ∧ kNN(image_vec, q) < τ`.
2. **Long-tail mining**

   * Rarity scoring vs fleet distribution; enforce **diversity** (min pairwise distance); slice coverage constraints (ensure each critical slice ≥ target).
   * Budgeted selection under storage/labeling limits.
3. **Validate**

   * Deduplicate via perceptual hash & embedding distance.
   * Balance report by slice; human **UI spot QA** (sampled thumbnails).
4. **Materialize**

   * Emit **dataset spec** and clip lists; version with **DVC** and **W\&B Artifacts**.

**Outputs**

* `dataset_spec.yaml` (filters, slices, version pins), curated clip URI lists for `train/val/test`.

**Storage & Indexing**

* **S3 Gold** `curation/...`; **DVC** tags/locks; saved queries in **DynamoDB**; discoverability in **OpenSearch**.

**Core tooling**

* **AppSync (GraphQL)**, **OpenSearch**, **Athena**, **DVC**, internal React curation UI.

---

### 9) Auto-Labeling (Offboard)

**Trigger**

* New curated set from #8 (or nightly bulk run).

**Inputs**

* Curated clips; camera/LiDAR calibration; map layers (speed limits, lanes); optional prior labels.

**Step-by-step (with guardrails)**

1. **Run offboard labelers (GPU)**

   * 2D detection/segmentation (e.g., YOLOX / Mask R-CNN), 3D detection (CenterPoint/Pillar), multi-camera fusion (BEVFusion-style), tracking (ByteTrack/DeepSORT), lane topology (LaneATT or lane graph extractor).
   * Temporal smoothing (Kalman/IMM); identity stitching across cameras.
2. **Confidence gating**

   * Keep high-confidence pseudo-labels; route uncertain/rare classes to #10.
   * Calibrate thresholds per slice (e.g., night/rain).
3. **Self-consistency checks**

   * Cross-view re-projection residuals, track continuity, kinematics plausibility (speed vs displacement), lane adherence.
4. **Validate**

   * Label schema conformance; IoU and AP deltas against a **held-out human-labeled subset**; per-class coverage/imbalance report; **ECE** for calibration.
   * Gate deployment of labels on quality thresholds; log to **W\&B**.

**Outputs**

* `labels_auto/` (COCO/Waymo-style JSON), tracks, lane vectors/graphs, scene graphs; `quality_report.json`.

**Storage & Indexing**

* **S3 Gold**; summary tables in **Glue/Athena**; **W\&B Artifacts** (dataset + model provenance).

**Core tooling**

* **EKS/Batch** GPU jobs; PyTorch/TensorRT; multi-view fusion; **W\&B** for metrics/artifacts.

---

### 10) Human QA (HITL)

**Trigger**

* Low-confidence/uncertain slices from #9; periodic audit sampling.

**Inputs**

* Auto-labels + media; labeling ontology (versioned); guidelines & golden tasks.

**Step-by-step (with guardrails)**

1. **Priority queue**

   * Order by uncertainty, rarity, business priority; enforce per-slice quotas.
2. **Annotate/verify**

   * Labelers in **Labelbox** or **SageMaker Ground Truth**; consensus labeling for critical classes; adjudication by senior reviewers.
3. **Quality control**

   * Blind overlap to compute **IAA (κ/α)**; golden tasks; geometric linting (box aspect, mask holes).
4. **Validate**

   * Promote only if `precision@accept ≥ target` and **IAA ≥ threshold**; otherwise route feedback to guidelines or auto-labeler thresholds.

**Outputs**

* `labels_human/` (final truths), **diffs vs auto-labels**, QA reports, updated ontology version.

**Storage & Indexing**

* **S3 Gold** human labels; tool-native label DB for audit; **DVC** dataset tags.

**Core tooling**

* **Labelbox**/**Ground Truth**, reviewer dashboard, webhooks → S3; **W\&B** lineage links.

---

### 11) Golden / Slice Builder

**Trigger**

* After #9–#10 converge; prior to training cycles or benchmark refresh.

**Inputs**

* Labeled pools (auto + human), scenario specs from #8, slice definitions (weather/time/geo/object).

**Step-by-step (with guardrails)**

1. **Assemble & balance**

   * Stratified sampling to meet per-slice minima; handle class imbalance (reweighing or oversample rare).
   * Respect temporal/geographic boundaries to avoid leakage.
2. **Freeze & version**

   * Emit `*.manifest` with absolute URIs + checksums; produce **Datasheet for Datasets** and populate the **data section of the Model Card**; register in **W\&B Artifacts** + tag in **DVC**.
3. **Validate**

   * **Leakage checks**: no overlapping `scene_id/drive_id` across splits; near-duplicate screening via embedding distance.
   * Baseline eval on Golden validation set; per-slice metrics recorded; gate if regressions vs last baseline.

**Outputs**

* `golden_train/val/test.manifest` (with hashes), `slices.yaml`, `datasheet.md`, `model_card.md` (data section).

**Storage & Indexing**

* **S3 Gold**; **DVC** & semantic tags; **Glue/Athena** tables for audits; **W\&B** artifact registry.

**Core tooling**

* **DVC**, **Athena**, **Great Expectations** (row-level rules), **W\&B Artifacts**.

---

### 12) Offline Mining (Continuous Error/Drift Discovery)

**Trigger**

* Nightly/weekly schedule; after evals; whenever new prod telemetry/shadow logs land.

**Inputs**

* Production predictions & telemetry, shadow logs, monitoring slices, last Golden manifest.

**Step-by-step (with guardrails)**

1. **Aggregate errors**

   * Join predictions with ground truth (where available) or proxy outcomes; compute FP/FN by slice; maintain error leaderboards.
2. **Drift & OOD**

   * **Evidently**: PSI/KS on feature/score distributions vs reference; OOD scores (Mahalanobis/energy).
3. **Mine candidates**

   * Seed with top error exemplars → nearest neighbors via vector index (#7) → cluster (HDBSCAN/DBSCAN) to discover themes.
   * Exclude items present in last training set (check against manifests).
4. **Validate**

   * **Novelty** (embedding distance vs training), **utility** (expected error coverage gain); deduplicate; spot-check sample.
5. **Emit next round specs**

   * Produce `next_specs.yaml` + candidate lists for #8; log run in **W\&B**.

**Outputs**

* `error_buckets/` (clustered examples with tags), `mined_candidates.parquet`, `next_specs.yaml`.

**Storage & Indexing**

* **S3 Silver/Gold**; notes to **OpenSearch**; error dashboards in **Athena/QuickSight**.

**Core tooling**

* **Airflow** schedule; **Evidently**; **OpenSearch/FAISS**; **Athena/QuickSight**; **W\&B**.

---

### Output Schemas

* **scene\_segments.parquet**

  ```
  scene_id:str, drive_id:str, vehicle_id:str, start_ts:ts, end_ts:ts,
  clip_uri:array<str>, tags:array<str>, trigger_flags:array<str>, score:float
  ```

* **embeddings.parquet**

  ```
  uri:str, scene_id:str, modality:str, vector:array<float32>, dt:date
  ```

* **dataset\_spec.yaml** (excerpt)

  ```yaml
  name: cyclists_dusk_rain_v3
  slices:
    - name: dusk_rain_cyclists
      filters:
        weather: [rain]
        time_of_day: [dusk]
        tags: ["cyclist"]
      knn_seed_uri: "s3://.../seed.jpg"
      target_count: 5000
  excludes:
    manifests: ["s3://.../golden_train.manifest"]
  ```

* **labels\_auto/** (COCO-style excerpt)

  ```json
  { "images":[{"id":123,"file_name":"...","scene_id":"S1", "ts":"..."}],
    "annotations":[{"image_id":123,"category_id":1,"bbox":[x,y,w,h],"score":0.92}],
    "categories":[{"id":1,"name":"pedestrian"}] }
  ```

* **golden\_train.manifest**

  ```
  uri:str, checksum:str, scene_id:str, slice:str, label_uri:str
  ```

* **quality\_report.json** (auto-labels)

  ```json
  {"map@50":0.61,"ece":0.07,"iou_median":0.73,
   "by_class":{"pedestrian":{"ap50":0.58,"n":12400},"cyclist":{"ap50":0.55,"n":4200}}}
  ```

---

<!--
### Notes & Rationale (grounded by references)

* **Automated scene detection and event triggers** are a best practice to mine “needles in the haystack” efficiently; pipelines enrich scenes with tags like left-turns, hard-brakes, cut-ins, and index them for search (structured in DynamoDB, semantic in OpenSearch) to power downstream curation. &#x20;
* AWS’s reference workflows for **scene detection** show containerized extraction + Spark/EMR batches, writing scene metadata to S3, DynamoDB, and enabling Athena analytics—exactly the shape of #6.&#x20;
* **Similarity search / vector indexing** is central for “example-based” discovery and dataset curation; industry data engines expose built-in similarity and metadata search to curate datasets quickly. &#x20;
* The closed-loop **auto-labeling → HITL QA** pattern (heavy offboard models generate pseudo-labels, humans verify the uncertain tail) is widely adopted to scale labeling throughput while maintaining quality. &#x20;
* Overall, this stack operationalizes the **“data engine” flywheel** (triggers → mine → label → curate) that accelerates iterations and focuses human effort on the highest-value slices. &#x20;

---
-->

### Validation Strategy Embedded in These Workflows

* **Schema & quality gates** (Great Expectations) run at scene generation and dataset assembly to prevent corrupt data from propagating.
* **Statistical drift checks** (Evidently) on embedding distributions and slice composition before building the Golden set.
* **Retrieval sanity** on vector indices (near-duplicate recall, mAP\@K) ensures the mining UX returns useful neighbors.
* **Label quality** via IAA, golden tasks, and spot checks on auto-labels maintain training-data integrity.
* **Leakage checks** (scene overlap across splits) in #11 safeguard evaluation integrity.

---

