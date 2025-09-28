# Real-Time & Streaming Data Pipelines: Challenges, Solutions


**Preamble: The Unseen Engine of Real-Time AI**

While sophisticated models and algorithms often take the spotlight, the true enabler of impactful, real-time Machine Learning is the underlying **real-time and streaming data pipeline**. These pipelines are the unsung heroes, responsible for ingesting, transforming, and serving fresh, high-quality data at speed and scale. For an MLOps Lead, mastering the design, deployment, and operation of these pipelines is paramount. This compendium distills insights from industry practitioners at Tecton, Netflix (Keystone, Recommendation Architecture), LinkedIn, DoorDash, Databricks, Facebook (Meta DSI), and Uber (Real-time Infra, Michelangelo), providing a robust thinking framework for navigating the complexities of real-time data for AI.

---

**Chapter 1: The "Why": The Criticality and Inherent Challenges of Real-Time Data for ML**

1.  **The Business Imperative for Real-Time:**
    *   **Use Cases Driving the Need:** Fraud detection (Tecton), in-session recommendations (LinkedIn, Databricks Travel Reco), dynamic pricing (Uber), real-time bidding, anomaly detection, personalization (Netflix), estimating delivery/prep times (Facebook DSI for DLRMs, UberEATS ETD).
    *   **Value Proposition:** Increased accuracy, immediate adaptability to changing environments (LinkedIn notes ROC-AUC degradation with feature delay), improved user experience, direct impact on conversion and retention.
    *   **From Batch to Online:** Many organizations realize batch predictions are insufficient for dynamic user interactions or rapidly changing data patterns (e.g., new/unseen users in e-commerce, product viewing history for recommendations). (Tecton, Databricks)
        *   Netflix's recommendation architecture (pre-2013) already distinguished between offline, nearline (responsive to user events), and online computation, highlighting the long-standing need for fresher data.

2.  **The Gauntlet: Core Challenges in Real-Time Data Pipelines**
    *   **Challenge 1: Building *Reliable* Streaming & Real-Time ML Data Pipelines is Hard.**
        *   **Streaming Compute Internals:** Deep understanding of state management (Netflix Keystone: jobs with 10s of TB state), watermarks, checkpointing intervals, and internal state stores (RocksDB) is crucial. (Tecton - "Challenges", Netflix Keystone, Facebook DSI)
        *   **State Management Hell:** Large aggregations, group-bys (Netflix Keystone: complex sessionization), data skew (hot keys), and application memory leaks can lead to OutOfMemoryExceptions (OOMs) and pipeline crashes. (Tecton - "Challenges", Netflix Keystone on Job State/Traffic)
            *   *Solutions:* Salting, partial aggregations, careful watermark/checkpoint tuning, robust state backends. DoorDash Riviera simplified complex state management for "Delivery ASAP time" using Flink SQL interval joins.
        *   **Spiky Throughput:** Variable event volumes (e.g., e-commerce weekends, Netflix show releases) require elastic pipelines that can scale without OOMs. (Tecton - "Challenges", Netflix Keystone on Elasticity)
        *   **Operational Burden:** Managing numerous, individually tuned streaming jobs (Netflix Keystone: thousands) with alerting for edge cases is complex. (Tecton - "Challenges", Netflix Keystone on Operation Overhead)
    *   **Challenge 2: Reliably *Retrieving* Fresh Features at *Low Latency* is Hard.**
        *   **Decoupling Storage and Compute:** Essential to prevent high ingestion throughputs from impacting read performance for feature serving. (Tecton - "Challenges")
        *   **Low-Latency Storage Operations:**
            *   *Choice of Store:* In-memory (Redis) for speed vs. disk-based (Cassandra, DynamoDB, Pinot) for cost/scale. Trade-offs on data types, serialization, TTLs, caching. (Tecton - "Challenges",   on Feature Stores, LinkedIn uses Pinot, Uber Michelangelo uses Cassandra).
            *   *Operations:* Point lookups, batch retrievals (e.g., for recommendation candidates), small-range scans. (Tecton - "Challenges")
        *   **High Throughput Online Serving:** Handling spiky traffic and large batch retrievals (hundreds of candidates) while maintaining low tail latencies. (Tecton - "Challenges")
    *   **Challenge 3: Ensuring Service Levels (SLAs) at Scale is Hard.**
        *   **Key SLAs:** Feature Freshness (event to serving time: LinkedIn <15s, Tecton <1s), Serving Latency (get_features call: LinkedIn <100ms, Tecton <25ms), Uptime/Availability (99.95%+). (Tecton - "Challenges", Netflix Keystone on Resiliency)
        *   **Operational Burden & Multi-tenancy:** Continuous monitoring, alerting, and individual tuning of numerous pipelines. Netflix Keystone supports thousands of jobs with runtime & operational isolation.
    *   **Challenge 4: Mitigating Training/Serving Skew is Hard.**
        *   **Definition:** Model performs differently online than expected from offline training. (Tecton - "Challenges")
        *   **Common Causes:**
            *   **Time Travel / Late Arriving Data:** Training data leaks future information or uses data not yet available at the "effective timestamp" of serving. (Tecton - "Challenges")
            *   **Inconsistent Transformation Logic:** Different code/logic for feature engineering in batch training pipelines vs. real-time serving pipelines (e.g., Spark vs. Flink, or different SQL dialects). Databricks addresses this with MLflow `pyfunc` wrappers for on-demand features. Hopsworks taxonomy highlights the need for consistency. (Tecton - "Challenges", Hopsworks Taxonomy, Databricks)
            *   **Data Drift:** Production data distributions diverge from training data distributions. (Tecton - "Challenges")
        *   **Debugging:** Painful and time-consuming. Requires meticulous comparison of feature distributions and pipeline logic.
    *   **Challenge 5: Backfilling Features is Difficult.**
        *   Streaming sources often have limited retention (e.g., Kinesis 2 weeks).
        *   Historical backfills for new features or long window aggregations require combining batch sources with forward-fill from streams, which is complex to make consistent. Netflix Keystone offers dynamic source switching and rewind from checkpoint. (Tecton - "Aggregations Part 1",   on stateful training, Netflix Keystone)
    *   **Challenge 6: Complexity of Real-Time Aggregations**
        *   Memory constraints for long-running windows (Facebook DSI online preprocessing, Netflix Keystone).
        *   Achieving ultra-high freshness (<1s) with sliding windows vs. per-event updates.
        *   Generating point-in-time correct training datasets historically.

---

**Chapter 2: Architectural Patterns & Solutions for Real-Time Data Pipelines**

1.  **The Evolutionary Path to Real-Time ML**
    *   **Stage 1: Batch Prediction:**
        *   All predictions precomputed. Limitations: No personalization for new users, stale recommendations.
        *   *Not a prerequisite for online if building new.*
    *   **Stage 2: Online Prediction with Batch Features:**
        *   Predictions generated on request. Uses real-time user activity to look up precomputed *batch* features/embeddings for in-session adaptation.
        *   *Requires:* Streaming transport (Kafka, Kinesis), basic stream computation (sessionization), low-latency KV store for embeddings.
        *   *Challenges:* Inference latency, setting up streaming infra, embedding quality.
        *   *LinkedIn's early approach* for some recommendations might fit here, where batch-computed features are pushed to online stores (Venice).
    *   **Stage 3: Online Prediction with Online Features (Real-Time & Near Real-Time):**
        *   **Real-Time (RT) Features:** Computed *at prediction time*. Databricks calls these "on-demand features" computed via MLflow `pyfunc` preprocessor using request-time context.
        *   **Near Real-Time (NRT) Features (Streaming Features):** Precomputed asynchronously by a streaming engine. This is the focus of Tecton, DoorDash Riviera, LinkedIn's Samza/Pinot pipeline, Uber's Samza/Cassandra pipeline (Michelangelo), and Netflix Keystone.
        *   **Databricks Feature Computation Architectures:** Clearly defines batch, streaming, and on-demand, emphasizing using the right one based on freshness.

    <img src="../../_static/mlops/ch5_data_pipelines/streaming_pipelines/databricks/1.png" width="80%"/>

    <img src="../../_static/mlops/ch5_data_pipelines/streaming_pipelines/databricks/2.png" width="80%"/>

    [Best Practices for Realtime Feature Computation on Databricks](https://www.databricks.com/blog/best-practices-realtime-feature-computation-databricks)


2.  **The Core Pipeline: Ingestion, Transformation, Serving**
    *   **Stream Ingestion & Transport:**
        *   **Technology:** Apache Kafka, AWS Kinesis
        *   **Key Aspects:** Reliable event capture. Netflix Keystone's Data Pipeline service manages routing and messaging. DoorDash uses an enhanced Kafka REST Proxy. Uber has uReplicator and Chaperone for Kafka. Facebook uses Scribe and LogDevice.
        *   **Schema Management:** Crucial. Avro with Schema Registry, Protobuf
    *   **Stream Processing & Feature Computation:**
        *   **Technology:** Apache Flink, Spark Streaming, Apache Samza.
        *   **Logic/DSL:** SQL is popular for accessibility
        *   **State Management:** Key challenge. Netflix Keystone handles jobs with 10s of TB state.
    *   **Low-Latency Feature Serving (Online Store / KV Store):**
        *   **Technology:** Redis, Apache Pinot, Cassandra, AWS RDS
        *   **Function:** Store precomputed batch and streaming features for fast lookups during inference. LinkedIn's Pinot stores recent actions (96hr retention) for on-demand query-time feature computation.
    *   **Offline Store (for Training Data & Backfills):**
        *   **Technology:** Data Lakes (S3/GCS + table formats like Delta Lake). Data Warehouses (Snowflake - Databricks example).
        *   **Function:** Store historical raw events and computed features for model training and feature backfilling. LinkedIn logs computed near real-time features to HDFS. Facebook DSI stores DLRM training data in Hive/Tectonic.

3.  **Advanced Pattern: Tiled Time Window Aggregations**
    *   **Problem:** Efficiently serving long-window, high-freshness aggregations.
    *   **Solution Overview:**
        *   Break full window into smaller, pre-compacted "tiles" (e.g., 5-min, 1-hour sums).
        *   Store tiles and recent raw projected events in online/offline stores.
        *   At request time, combine aggregations from relevant tiles and raw events from head/tail of the window for the final feature value.
    *   **Ingestion Paths:**
        *   **Streaming Ingestion (Online Store):** Writes recent projected raw events.
        *   **Batch Ingestion (Online & Offline Store):** Backfills historical data (raw and compacted tiles) from an offline mirror (e.g., Hive table of streamed events).
        *   **Compaction Jobs (Spark Batch):** Periodically read raw events from offline mirror and create compacted tiles, writing to both online and offline stores.
    *   **Serving Paths:**
        *   **Online Serving:** Feature Server fetches necessary tiles and raw events from online store, computes final aggregation.
        *   **Offline Serving (Training Data):** Spark job fetches from offline store, aggregates as of different historical timestamps (point-in-time correctness).
    *   **Advantages:** Solves backfilling, supports long windows & ultra-freshness, compute/memory efficient (compacts, reuses tiles), fast retrieval.
    *   **Advanced (Sawtooth Windows - Airbnb):** Slide head of window, hop tail tile-by-tile to reduce raw event storage for very long windows, introducing slight (often tolerable) window size variation.


<img src="../../_static/mlops/ch5_data_pipelines/streaming_pipelines/patterns.svg"/>


4.  **The Role of Feature Stores/Platforms**
    *   **Core Function:** Centralize management of features for both training and serving, aiming to solve training-serving skew.
    *   **Key Capabilities (Idealized):**
        *   **Feature Definition & Transformation:** DSL or code-based definitions for batch, streaming, and on-demand features.
        *   **Orchestration:** Manages computation of features.
        *   **Storage:** Online (low-latency KV) and Offline (analytical/historical) stores.
        *   **Serving:** APIs to retrieve features for online inference and batch training data generation.
        *   **Monitoring:** Data quality, freshness, drift.
        *   **Governance:** Discovery, versioning, lineage, access control.
    *   **Addressing Real-Time Challenges:**
        *   Provides consistent transformation logic for online/offline paths.
        *   Manages complexities of streaming aggregations (e.g., Tecton's tiled approach).
        *   Offers optimized online/offline serving layers.
    *   **Databricks Feature Store:** Integrates with MLflow, supports batch, streaming (`streaming=True` API), and on-demand (via `pyfunc`) feature computation. Automatic lookup from online stores during model serving.
    *   **LinkedIn's approach (pre-Feature Store era, but illustrative):** Central Pinot store for recent actions, queried on-demand by various recommenders. This shares some philosophy with a feature store's consumption layer.
    *   **Uber Michelangelo:** Shared Feature Store with offline (HDFS) and online (Cassandra) components, and a DSL for feature definition and transformation, ensuring training-serving consistency.

5.  **Declarative Frameworks for Real-Time Feature Engineering**
    *   **DoorDash Riviera:** Users define feature pipelines via YAML (Flink SQL based). Aims to abstract Flink complexities, improve accessibility, reusability, and isolation.
        *   Custom Flink runtime and library for Protobuf support and connector abstractions.
        *   Generified Flink application JAR instantiated with different YAML configs.
    *   **Netflix Keystone SPaaS & Routing Service:** Declarative reconciliation architecture. Users declare "goal state" via UI/API; platform orchestrates. Aims to enable self-service.

---

**Chapter 3: The Data Transformation Taxonomy in Real-Time Systems (Hopsworks)**

Understanding where and how transformations occur is critical for reusable, maintainable, and skew-free real-time pipelines.

1.  **Model-Independent Transformations:**
    *   **Definition:** Produce reusable features, not specific to any single model. Applied once in feature pipelines (batch or streaming).
    *   **Examples:** Grouped aggregations (avg/min/max), windowed counts (clicks per day), RFM features, binning for categorization, stream windowed features (Flink/Spark), joins, time-series trend extraction.
    *   **Storage:** Output stored in a Feature Store (both online and offline portions).
    *   **Hopsworks Context:** Performed in "feature pipelines."
    *   *Examples:* LinkedIn's Samza jobs joining attributes to action events before storing in Pinot. DoorDash Riviera's Flink SQL for aggregations. Netflix Keystone for filtering/projection. Databricks batch/streaming feature computations. Uber Michelangelo's offline Spark/Hive pipelines. Facebook DSI's offline ETL.


2.  **Model-Dependent Transformations:**
    *   **Definition:** Produce features specific to one model, often parameterized by the model's training dataset (sample-dependent) or specific to a model architecture (e.g., LLM tokenization).
    *   **Examples:** Normalization/scaling (using training set mean/std), one-hot encoding based on training set vocabulary, text tokenization for a specific LLM.
    *   **Execution:** Applied *consistently* in both training pipelines (to create training data) and inference pipelines (before prediction).
    *   **Hopsworks Context:** Can be part of Scikit-Learn pipelines packaged with the model, TensorFlow preprocessing layers, PyTorch transforms, or defined in Hopsworks Feature Views.
    *   **Key to Avoiding Skew:** The *exact same* transformation logic and parameters (e.g., mean/std from the *original* training set) must be used.
    *   *Examples:* Normalization/scaling specific to a DLRM training run at Facebook. Tokenization for a specific NLP model at Netflix. Databricks example of `pyfunc` applying consistent preproc.


3.  **On-Demand Transformations:**
    *   **Definition:** Require input data only available at prediction request time (for online inference) or can be applied to historical data for backfilling features.
    *   **Execution (Online Inference):** Implemented as functions within the online inference pipeline. Takes request-time parameters and potentially precomputed features from the feature store.
    *   **Execution (Backfilling):** The same transformation function applied to historical event data in a feature pipeline to populate the feature store.
    *   **Model-Independence (for Feature Store Ingestion):** If backfilled into a feature store for reuse, the on-demand transformation itself should produce model-independent output. Model-specifics are chained *after* in the inference/training pipeline.
    *   **Hopsworks Context:** Implemented as Python/Pandas UDFs. Can be registered with feature groups (for backfilling via feature pipelines) and feature views (for consistent application in inference and training dataset generation).
    *   **Challenge:** Ensuring consistency between the online execution and the backfill execution if implemented in different systems/codebases. Feature stores aim to solve this by allowing definition once, execution in multiple contexts.
    *   *Examples:* LinkedIn: Querying Pinot for recent job applications (e.g., last N hours) and summarizing embeddings *at recommendation time*. Databricks: Computing user-destination distance using request-time user location and destination data from Feature Store.
    *   *Backfilling On-Demand Logic:* LinkedIn's approach of logging computed NRT features implies that the "on-demand" logic (Pinot query + summarization) can be replayed over historical logs for training data consistency. Databricks `pyfunc` wrapper ensures same logic for training (on historical/batch data) and serving (on request-time data). Hopsworks' distinction of on-demand for FS ingestion (backfill) vs. online inference is critical.

**Table: Data Transformation Taxonomy & Pipeline Placement**

| Transformation Type        | Reusable? | Sample Dependent? | Request-Time Data Needed (Online)? | Typical Execution Pipeline      | Output Stored in Feature Store? | Key Goal for MLOps Lead                               |
| :------------------------- | :-------- | :---------------- | :----------------------------------- | :------------------------------ | :------------------------------ | :---------------------------------------------------- |
| **Model-Independent**      | Yes       | No                | No (uses historical/stream data)     | Feature Pipeline (Batch/Stream) | Yes                             | Build robust, reusable features.                      |
| **Model-Dependent**        | No        | Yes (often)       | No (uses features from FS/TD)        | Training & Inference Pipelines  | No (applied on-the-fly)         | Ensure training-serving consistency; manage parameters. |
| **On-Demand (for FS)**     | Yes       | No                | Yes (for backfill, uses historical)  | Feature Pipeline (Backfill)     | Yes                             | Ensure consistency with online path; enable backfills.  |
| **On-Demand (Online)**     | No (direct output is model input) | No                | Yes                                  | Online Inference Pipeline       | No (ephemeral for request)      | Low latency; consistency with backfill path.          |


---

**Chapter 4: Evolving Towards Continual Learning with Real-Time Data**


1.  **Stage 1: Manual, Stateless Retraining:**
    *   Ad-hoc, infrequent model updates. Models trained from scratch. Painful and prone to errors.

2.  **Stage 2: Automated (Stateless) Retraining:**
    *   Scripts automate the retraining process (e.g., daily Spark jobs).
    *   *Requires:* Model store (MLFlow, SageMaker MR) for versioning, scheduler (Airflow, Argo).
    *   *"Log and Wait" Pattern:* Reusing features extracted during prediction for subsequent training to improve consistency.
    *   LinkedIn's conventional batch feature pipeline feeding recommenders. Uber Michelangelo's scheduled Spark jobs for offline features. Facebook DSI's periodic offline dataset generation.


3.  **Stage 3: Automated, Stateful Training (Fine-tuning):**
    *   Model continues training on new data instead of from scratch.
    *   *Benefits:* Less data needed for updates, reduced training cost (Grubhub 45x cost reduction).
    *   *Requires:* Better model store with lineage (model A fine-tuned from model B), reproducible streaming features (time-travel capability).
    *   Facebook's continuous training of DLRMs on fresh samples. While the paper doesn't explicitly state "stateful fine-tuning" in the deep learning sense for all models, the continuous nature and massive evolving datasets imply something beyond stateless retraining from scratch every single time for all parameters.


4.  **Stage 4: Continual Learning (The Holy Grail):**
    *   Models updated whenever data distributions shift or performance degrades, not just on a fixed schedule.
    *   *Requires:*
        *   **Trigger Mechanisms:** Time-based, performance-based, drift-based. Sophisticated monitoring to identify *meaningful* drift.
        *   **Continuous Evaluation:** Beyond static test sets. Backtesting, progressive evaluation, shadow deployments, A/B testing, bandits.
        *   **Orchestrator:** To spin up update/evaluation instances without interrupting serving.
    *   The industry is moving towards this, but full, widespread adoption is still nascent. Triggering retraining based on model performance degradation or concept drift is the goal.
    *   Meta's DLRM release process (exploratory, combo, RC) is a structured way to iterate and adapt, driven by model quality metrics, which is a step towards continual improvement, if not fully automated continual learning for all parameters.


**Online Prediction for Bandits:**
*   Bandits (multi-armed, contextual) offer more data-efficient model evaluation and exploration than A/B testing.
*   *Requires:* Online prediction, short feedback loops (to update arm/model performance), stateful mechanism to track performance and route traffic. Harder to implement but potentially huge gains.

---

**Chapter 5: Operationalizing Real-Time Pipelines - A Lead's Checklist**

1.  **SLAs - The Non-Negotiables:**
    *   **Feature Freshness:** Define and monitor. What's acceptable for the use case?
    *   **Serving Latency:** P95, P99. Critical for user-facing applications.
    *   **Uptime/Availability:** Design for fault tolerance. Netflix Keystone emphasizes failure as a first-class citizen.
    *   **Data Quality & Consistency:** How are these measured and ensured in a streaming context?
    *   *Netflix Keystone* explicitly mentions diverse user requirements for latency, duplicates, ordering, and delivery semantics, which translate to SLAs.

2.  **Technology Choices & Trade-offs (Recap):**
    *   **Streaming Engine (Flink vs. Spark Streaming vs. Kafka Streams):** State management capabilities, ecosystem, SQL support, community, operational complexity. (Netflix chose Flink for Keystone SPaaS).
    *   **Online Store (Redis vs. Cassandra vs. DynamoDB vs. other KV):** Latency, throughput, cost, data model flexibility, managed vs. self-hosted.
    *   **Feature Store/Platform (Build vs. Buy - Tecton, Hopsworks, Feast):** Consider the taxonomy support, integration, operational overhead.
    *   Industry Case Studies
        *   **Streaming Engine:** LinkedIn (Samza), Netflix/DoorDash/Uber (Flink), Databricks/Facebook (Spark). Rationale often includes ecosystem, state management, SQL support, and operational simplicity (managed services).
        *   **Online Store:** LinkedIn (Pinot), DoorDash/Tecton (Redis), Uber (Cassandra), Netflix (EVCache, Cassandra, MySQL for various rec components).
        *   **Feature Store/Platform:** Databricks Feature Store. Commercial (Tecton, Hopsworks). In-house (Uber Michelangelo).

3.  **Monitoring & Alerting:**
    *   **Pipeline Health:** Lag, throughput, error rates, resource utilization of streaming jobs.
    *   **Data Quality in Streams:** Schema violations, value distributions, null rates.
    *   **Feature Freshness & Latency:** End-to-end monitoring.
    *   **Cost Monitoring:** Especially for cloud-based managed services.
    *   *Netflix Keystone* emphasizes personalized monitor/alert dashboards for each job.

4.  **State Management Strategy:**
    *   In-memory vs. externalized state (e.g., RocksDB).
    *   Checkpointing frequency and recovery time.
    *   Handling large state and data skew.
    *   *Netflix Keystone* deals with job states up to 10s of TBs.
    *   *DoorDash Riviera* leverages Flink's state for interval joins.

5.  **Schema Management:**
    *   Schema Registry (Confluent, AWS Glue SR) for Kafka.
    *   Handling schema evolution in streaming sources and its impact downstream.
    *   *Netflix Keystone/Data Mesh* (Avro), *DoorDash Riviera* (Protobuf with Schema Registry).
    *   *LinkedIn's standard action schema* is a form of schema management for interoperability.

6.  **Backfilling & Reprocessing:**
    *   Strategy for new features on historical data or reprocessing due to bugs.
    *   Kappa architecture variants. Netflix Keystone's dynamic source switching.
    *   *Netflix Keystone* supports dynamic source switching and rewind from checkpoint.
    *   *Databricks* on-demand features using `pyfunc` can be applied to historical data for backfill.

7.  **Testing Real-Time Pipelines:**
    *   Unit tests for transformation logic.
    *   Integration tests with mocked sources/sinks.
    *   End-to-end tests in staging environments.
    *   Performance/load testing.
    *   Data validation against golden datasets or known properties.
    *   *Netflix Keystone* emphasizes unit tests, integration tests, operational canary, data parity canary.

8.  **Deployment & CI/CD:**
    *   Automated deployment of streaming jobs and feature definitions.
    *   Canary releases or blue/green for pipeline updates.
    *   Rollback strategies.
    *   *Netflix Keystone* uses Spinnaker and Titus (container runtime) for deployment orchestration.
    *   *DoorDash Riviera* Flink jobs deployed on Kubernetes via Helm charts, managed by Terraform.

---

**Chapter 6: The MLOps Lead's Real-Time Data Pipeline - Thinking Framework (Mind Map)**

<img src="../../_static/mlops/ch5_data_pipelines/streaming_pipelines/mindmap.svg"/>

___

