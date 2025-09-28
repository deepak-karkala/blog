# Data Ingestion Workflows

##

### 1) Ingestion (telemetry streaming + bulk sensor offload)

**Trigger**

* Telemetry: always-on collection schemes deployed to vehicles; cloud rules updated centrally.
* Bulk sensor logs: a new drive/SSD arrives at a copy station; or Snowball/DataSync task completes.

**Inputs**

* Telemetry (edge → cloud): CAN/GNSS/IMU summaries, event flags, light ML detections via **AWS IoT FleetWise / IoT Core**.
* Bulk logs: camera MP4/H.264/H.265, LiDAR PCAP/ROS bag, radar binaries, CAN MDF4. Moved via **AWS DataSync**, **Snowball Edge**, and/or **AWS Direct Connect**.

**Steps (with guardrails)**

1. **Edge publish (telemetry):** FleetWise edge agent packages signals and sends to cloud; FleetWise standardizes signals. Alerts on missing schema.
2. **Land telemetry:** IoT Core → Kinesis Data Firehose → **S3 Bronze** (JSON/Parquet). Partition by `dt/vehicle_id`. (AWS ADDF reference patterns).
3. **Bulk transfer:**

   * Copy station mounts SSD → **DataSync** task to S3 Bronze (checksum/throughput reported), or **Snowball Edge** jobs for multi-TB batches; sites with fixed links use **Direct Connect**.
4. **Registration:** S3 event → Lambda → write a **manifest (drive\_id, sensors, sizes, etags)** to **DynamoDB** and emit an **EventBridge** message to kick off downstream DAGs (Convert/Sync).

**Validation / testing hooks**

* **Checksums/ETags** match; **Great Expectations** “landed set completeness” (all declared sensors present).
* **S3 object ACL/KMS** policy tests; bucket policy unit tests (Terratest).

**Output & storage**

* Telemetry in `s3://lake/bronze/telemetry/...` (JSON/Parquet).
* Raw sensor logs in `s3://lake/bronze/drives/<drive_id>/...`.
* Drive manifest rows in **DynamoDB** (hot index).
* Event to **EventBridge** → Airflow DAG run id.

---

### 2) Integrity & PII (quality gates + anonymization)

**Trigger**

* Post-ingestion event (per drive) or micro-batch (per N files).

**Inputs**

* Newly landed Bronze sensor files; telemetry for context.

**Steps (with guardrails)**

1. **Structural integrity:** probe ROS/MDF4/MP4 headers, durations, monotonic timestamps; fail list to SQS dead-letter.
2. **PII redaction:**

   * **Faces:** **Amazon Rekognition** detect faces (image or video APIs) → **OpenCV** blur/mask; batched via Step Functions/Lambda/ECS.
   * **Plates:** detect as **text** (Rekognition DetectText) or a **Custom Labels** model for plates; blur with OpenCV.
3. **Quality gates:** **Great Expectations** suites (missing packets %, frame rate bounds, GPS bounds, CAN ranges).
4. **Security posture:** Tag outputs with “pii=redacted\:true”.

**Validation / testing hooks**

* **Before/after PII scan:** ensure no unblurred faces/plates remain on a sampled set; store evidence frames.
* GE failure = **pipeline hard-fail** with alert; metrics to CloudWatch.

**Output & storage**

* **S3 Silver** copies of redacted media (`.../silver/video/`, `.../silver/images/`), plus JSON sidecars noting redaction boxes.
* **Audit log** in DynamoDB (who/when/what redacted).

---

### 3) Sync & Convert (time alignment, transcoding, columnarization)

**Trigger**

* After Integrity & PII passes for a drive.

**Inputs**

* Bronze raw (rosbag/MDF4/PCAP/MP4), telemetry; calibration blobs if available.

**Steps (with guardrails)**

1. **Topic extraction:** containerized jobs on **AWS Batch/ECS** read rosbags and MDF4, extract topics to intermediate artifacts (PNG keyframes, JSON, **Parquet**).
2. **Standardize timebase:** resample/synchronize multi-rate streams (e.g., 10–30–100 Hz) to a uniform grid (e.g., 100 ms).
3. **Transcode:**

   * Video → keyframes/thumbnails; extract per-frame timestamps.
   * LiDAR PCAP → Parquet/Zarr packets (intensity, ring).
   * CAN MDF4 → Parquet tables.
   * Write **partitioned Parquet** (Hive style) to **S3 Silver**; **Glue Crawlers** register schemas; **Athena** queries enabled.
4. **Performance hygiene:** sort/order and choose sensible Parquet row-group sizes for typical filters; predicate pushdown.
5. **Orchestration:** Airflow (MWAA) coordinates Batch/EMR steps for conversion & sync as in AWS solutions.

**Validation / testing hooks**

* **Temporal alignment unit tests** (skew bounds across sensors).
* **Idempotency**: re-running conversion produces identical **content hashes**.
* **Athena smoke**: SELECT counts & schema match; Glue Data Catalog entries present.

**Output & storage**

* **S3 Silver**: `.../silver/parquet/<sensor>/dt=.../vehicle=...` and `.../silver/keyframes/`.
* **Glue Data Catalog** tables; **Athena** ready.

---

### 4) Metadata & Indexing (make drives searchable)

**Trigger**

* On completion of Sync & Convert for a drive (plus periodic enrichment backfills).

**Inputs**

* Silver Parquet/Zarr tables + keyframes; telemetry; drive manifest.

**Steps (with guardrails)**

1. **Low-level metadata:** derive coverage windows, sample rates, counts, resolutions; write **Iceberg/Parquet** metadata tables registered in **Glue Data Catalog**.
2. **Scene/event tags:** lightweight detectors & heuristics (e.g., harsh brake, cut-in, stop-n-go) run on Silver tables; produce **scene windows**.
3. **Search indices:**

   * **OpenSearch** documents for per-scene/per-clip tags, geo bounding boxes, weather keys; support free-text and structured search.
   * Optional **vector embeddings** (clip/scene embeddings) stored in OpenSearch **k-NN** fields for similarity (“find more like this”).
4. **Lineage:** link every index doc to S3 URIs + dataset/version (DVC hash); store in W\&B artifact metadata for cross-traceability.

**Validation / testing hooks**

* **Round-trip** validation: sample index doc → fetch S3 assets and confirm existence & timestamp alignment.
* **OpenSearch mapping tests** and ingestion success; shard/replica health alarms.

**Output & storage**

* **Glue Catalog**: structured tables for analytics (Athena/Redshift).
* **OpenSearch**: `scenes-*`, `clips-*` indices with tag + vector fields.

---

### 5) Map & Weather Enrichment (context joins)

**Trigger**

* After Metadata & Indexing, or nightly backfill (new map/weather drops).

**Inputs**

* Silver time-series + scene windows; map & weather sources.

**Steps (with guardrails)**

1. **Map layers:** ingest **OpenStreetMap Daylight** (or vendor tiles) and road attributes into S3/Glue; enable **Athena** joins on road class, intersection type, speed limits where available.
2. **Weather join:** pull historical weather and join by geo/time window to attach precipitation, visibility, wind.
3. **Geospatial ops:** EMR/Spark or Athena SQL to spatially snap GPS traces to road segments; cache per-segment summaries in **DynamoDB** for fast lookups.
4. **Persist enrichment:** write **Gold** tables: `scenes_enriched` with map class, intersection semantics, speed-limit context, weather tags.
5. **Index update:** push enriched tags to **OpenSearch** (e.g., `weather=rain`, `road_class=urban_primary`) to power scenario search.

**Validation / testing hooks**

* **Temporal tolerance tests** for weather joins (±X minutes).
* **Map snap QA**: % points snapped, max offset thresholds; sampled visual verification.
* **Athena smoke** on Gold tables.

**Output & storage**

* **S3 Gold**: `.../gold/scenes_enriched/` (Parquet/ Iceberg), Glue tables.
* **OpenSearch** indices updated with enrichment facets.
* **DynamoDB** cache for hot segment/weather joins.

---

#### Tooling & Services

* **ADDF** reference architecture for ingest → validate → extract → detect scenes → catalog/index.
* **AWS IoT FleetWise/IoT Core/Kinesis/Firehose** for telemetry ingest.
* **DataSync / Snowball Edge / Direct Connect** for bulk movement.
* **S3 + Glue Data Catalog + Athena** for lakehouse tables/SQL.
* **Batch/ECS/EMR** for conversion & Spark jobs (rosbag→Parquet, scene detection).
* **OpenSearch (k-NN optional)** for scenario & similarity search.
* **Rekognition + OpenCV** for privacy blurring.

