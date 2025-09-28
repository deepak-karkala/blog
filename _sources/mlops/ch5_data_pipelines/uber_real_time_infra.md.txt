# Uber: Real-time Data Infrastructure

**1. Introduction**

*   **Core Problem:** Uber faces massive (PBs/day), rapidly growing real-time data streams from diverse sources (apps, services, logs, DB CDC).
*   **Need:** Seconds-latency processing for critical use cases (pricing, fraud, ML) and diverse users (engineers to ops personnel).
*   **Challenge Triad:** Scaling *data volume* (exponential growth, multi-region), *use cases* (competing needs like freshness vs. consistency), and *users* (varying technical skill, managing large client base).
*   **Strategy:** Build a unified platform leveraging heavily customized *open-source* technologies.

**2. Requirements**

*   **Critical Needs:** High Consistency (zero data loss for finance), High Availability (99.99%), Low Freshness (seconds), Low Query Latency (sub-second p99 for interactive), Scalability (seamless growth), Cost Efficiency, and Flexibility (SQL & programmatic APIs, push & pull models).
*   **Inherent Trade-offs:** Cannot optimize all requirements simultaneously (e.g., CAP theorem forces choices like prioritizing freshness over consistency for Surge Pricing).

**3. Abstractions**

*   **Layered Architecture (Bottom-Up):**
    *   **Storage:** Long-term blob/o bject store (HDFS).
    *   **Stream:** Low-latency pub/sub (Kafka).
    *   **Compute:** Stream/batch processing logic (Flink).
    *   **OLAP:** Fast, limited SQL analytics on fresh/recent data (Pinot).
    *   **SQL:** Full SQL interface (Presto), federating or compiling to lower layers.
    *   **API:** Programmatic access for complex logic.
    *   **Metadata:** Schema definition, versioning, compatibility.
*   **Key Idea:** Provides distinct functional layers, allowing technology choices and evolution within each layer.

**4. System Overview**

*   **4.1 Apache Kafka (Streaming Storage):**
    *   Foundation for pub/sub. Uber runs one of the largest deployments.
    *   **Uber Enhancements:**
        *   *Cluster Federation:* Abstracts physical clusters for scalability and availability.
        *   *Dead Letter Queue (DLQ):* Isolates problematic messages without blocking pipelines.
        *   *Consumer Proxy:* Decouples clients via gRPC, centralizes complex logic (retries, batching), enables *push-based* dispatch for higher parallelism.
        *   *uReplicator:* Robust cross-datacenter replication.
        *   *Chaperone:* End-to-end auditing for data loss detection.
*   **4.2 Apache Flink (Stream Processing):**
    *   Chosen for robust state management and backpressure handling.
    *   **Uber Enhancements:**
        *   *FlinkSQL Contribution:* Enables SQL users to define streaming jobs easily.
        *   *Operational Automation:* Resource estimation, auto-scaling, automatic job monitoring/recovery.
        *   *Unified Deployment Architecture:* Manages lifecycle for both SQL and API-based jobs, abstracting underlying clusters (YARN, Peloton) and storage (HDFS, S3, GCS).
*   **4.3 Apache Pinot (OLAP):**
    *   Chosen for low-latency, high-throughput queries, lower footprint vs. alternatives (Elasticsearch, Druid). Uses lambda architecture internally.
    *   **Uber Enhancements:**
        *   *Scalable Upsert:* Unique exactly-once update capability via primary-key partitioning/routing.
        *   *Presto Integration:* Full SQL via connector with predicate/aggregation pushdown.
        *   *Ecosystem Integration:* Schema inference, Flink sink, workflow integration.
        *   *Peer-to-Peer Segment Recovery:* Improves availability/freshness by removing dependency on external "deep store" for recovery.
*   **4.4 HDFS (Archival Store):**
    *   Long-term, source-of-truth storage (Parquet format). Used for backfills and state persistence (Flink checkpoints, historical Pinot segments).
*   **4.5 Presto (Interactive Query):**
    *   Provides fast, interactive SQL across diverse sources via its connector API.
    *   **Uber Enhancement:** Deep Pinot connector integration with advanced pushdowns (projection, aggregation, limit) for low-latency queries on real-time data.

**5. Use Cases Analysis**

*   **Illustrates Trade-offs:**
    *   *Surge Pricing:* Flink pipeline optimized for **freshness/availability** over consistency (late data dropped).
    *   *Eats Restaurant Manager:* Pinot dashboard optimized for **low query latency** via Flink pre-aggregation, sacrificing some flexibility.
    *   *ML Monitoring:* Leverages Flink+Pinot for **scalability** handling high-cardinality metrics.
    *   *Eats Ops Automation:* Shows path from **ad-hoc exploration** (Presto+Pinot) to productionized alerting rules.
*   **Key Point:** Different use cases leverage the same components but tune them differently based on specific requirements.

**6. All-Active Strategy**

*   **Goal:** High availability and disaster recovery across multiple regions.
*   **Foundation:** Multi-region Kafka replication (uReplicator).
*   **Patterns:**
    *   *Active-Active (e.g., Surge):* Redundant computation (Flink state rebuilt from Kafka). Higher compute cost, faster failover.
    *   *Active-Passive (e.g., Payments):* Single active consumer instance. Requires complex *offset synchronization* service for lossless, consistent failover.

**7. Backfill**

*   **Problem:** Reprocessing historical data from archival storage (HDFS) using streaming logic (bugs, new features). Standard Lambda/Kappa insufficient at Uber's scale/retention.
*   **Uber's Kappa+ Solution:**
    *   *SQL:* Run same FlinkSQL on Kafka (real-time) and Hive (batch).
    *   *API:* Reuse Flink DataStream code directly on batch sources (Hive), handling bounded input, throttling, and out-of-order data challenges.

**8. Related Work**

*   Positions Uber's stack against industry alternatives in messaging, stream processing, OLAP, and SQL.
*   Highlights Uber's choice of *customized open source* and *loosely coupled systems* for flexibility, contrasting with more integrated (e.g., HTAP) or proprietary approaches.

**9. Lessons Learned**

*   **Open Source:** Powerful accelerator but requires significant investment in customization, integration, and operational tooling for large-scale, diverse needs.
*   **Rapid Evolution:** Requires strong abstractions (interfaces, thin clients), client management strategies (e.g., Monorepo), language consolidation, and robust CI/CD.
*   **Operations:** Automation (deployment, scaling, recovery) and observability (monitoring, alerting, chargeback) are crucial for managing scale efficiently.
*   **User Experience:** Self-serve capabilities (data discovery via metadata, easy onboarding UI), and robust debugging tools (auditing) are vital for scaling the user base.

**10. Conclusion**

*   Uber successfully built a flexible, scalable real-time infrastructure handling petabytes daily.
*   Key contributions (Kafka federation/proxy, FlinkSQL/automation, Pinot upsert/P2P recovery, Presto integration) directly addressed the scaling challenges of data, use cases, and users by extending open-source capabilities.

**11. Future Work**

*   **Unification:** Single programming model for stream and batch processing.
*   **Multi-Region/Zone:** Optimize for cost, reliability, and disaster tolerance.
*   **Cloud Agnosticism:** Enable portability between on-prem and cloud.
*   **Tiered Storage:** Implement cheaper storage for older data in Kafka and Pinot for cost efficiency.