# Meta - Understanding Data Storage and Ingestion for Large-Scale Deep Recommendation Model Training

**1. Introduction**

*   **Problem:** Datacenter-scale AI clusters (thousands of DSAs) train complex Deep Learning Recommendation Models (DLRMs). The Data Storage and Ingestion (DSI) pipeline—storing Exabytes, serving tens of TB/s—is becoming the *dominant* performance and power bottleneck over DSA compute.
*   **Focus:** Characterizes Meta's production end-to-end DSI pipeline (central data warehouse + Data PreProcessing Service - DPP) for DLRMs.
*   **Goal:** Identify hardware/system bottlenecks and motivate urgent research into DSI optimization.

**2. Recommendation Model Background**

*   **Workload:** DLRMs are massive (trillion+ parameters, ZetaFLOPs/train), trained via mini-batch SGD on specialized hardware (Meta's ZionEX nodes: 8xA100 GPUs, NVLink, RoCE).
*   **Parallelism:** Uses Data Parallelism and Model Parallelism (embedding sharding) across many nodes.
*   **DSI Role:** The pipeline is responsible for generating samples, storing them in datasets, and performing online preprocessing to convert samples into tensors loaded into GPU memory (HBMs) for each training step.

**3. Disaggregated Data Storage, Ingestion, and Training Pipeline**

*   **Storage System:**
    *   Raw logs -> ETL -> Hive tables (for schema consistency across services like Spark/Presto).
    *   Stored as DWRF (custom columnar ORC fork enabling feature filtering at storage layer) files.
    *   Persisted on Tectonic (Exabyte-scale distributed append-only filesystem using RocksDB, running on HDDs).
*   **Online Preprocessing System (DPP - Data PreProcessing Service):**
    *   *Motivation:* Needed to prevent data stalls on powerful GPU trainers.
    *   *Architecture:* Disaggregated service. Control Plane (DPP Master) manages job specs, work distribution (splits), fault tolerance (checkpointing), and auto-scaling. Data Plane (DPP Workers) are stateless, pull transforms & work splits, execute ETL using C++ binaries, buffer tensors. DPP Clients run on trainer nodes, fetch tensors via RPC for PyTorch.

**4. Coordinated Training at Scale**

*   **Industrial Reality:** Training isn't isolated runs but a *collaborative release process* (exploratory -> combo -> release candidate jobs).
*   **Resource Peaks:** Leads to periodic, concurrent *combo job peaks* requiring massive, co-located storage, preprocessing, and trainer capacity across many models simultaneously.
*   **Global Scale:** Requires dataset replication and intelligent scheduling across geo-distributed datacenters.
*   **Dataset Dynamism:** Features constantly evolve (added/deprecated), demanding adaptable storage and hindering static optimizations.

**5. Understanding Data Storage and Reading**

*   **Read Pattern:** Jobs read datasets >> local storage (needs central warehouse). Reads are < 1 epoch.
*   **Heavy Filtering:** Jobs select subsets of partitions (rows) and features (columns, ~10% used), leading to only ~20-40% of bytes being read per model.
*   **Small I/Os:** Columnar format (DWRF) + feature filtering results in *very small* (median ~1KB) read I/Os, severely stressing HDD IOPS.
*   **Inter-Job Reuse:** Significant data reuse *across* jobs for a given model; ~40% of hot bytes serve ~80% throughput, indicating caching potential (but reuse varies between models).

**6. Understanding Online Preprocessing**

*   **Bottleneck Origin:** Preprocessing directly on trainer host CPUs causes significant GPU stalls (Table 7).
*   **GPU Demand:** GPUs require high, variable tensor ingestion rates (up to 16.5 GB/s/node, 6x variation across models), set to increase 3.5x.
*   **Trainer Loading Cost:** Loading tensors via DPP Client consumes significant trainer host CPU/Memory BW (Fig 8) due to network stack, deserialization ("datacenter tax").
*   **DPP Worker Demand:** Requires many (9-55) workers per trainer node. Bottlenecks vary (Network ingress, CPU, Memory BW - Fig 9), with *Memory BW* projected as the future limit.
*   **DLRM Transforms:** Unique operations (Table 11), distinct from vision tasks. *Feature generation* dominates compute (75% cycles).

**7. Key Insights and Research Opportunities**

*   **Hardware Bottlenecks:** Storage (HDD IOPS/capacity gap), Ingestion (Memory BW), Trainer Hosts (CPU/Mem BW for data loading).
*   **Heterogeneous Hardware:** Need solutions balancing IOPS/capacity (e.g., SSDs/NVM caches for hot data) and accelerating transforms/network tax (e.g., GPUs, FPGAs, SmartNICs).
*   **Datacenter Planning/Scheduling:** Must be DSI-aware (capacity, co-location). Global scheduling needed to optimize replication/placement.
*   **Representative Benchmarks:** Urgent need for benchmarks reflecting industrial DSI reality (dynamic PB-scale data, <1 epoch filtered reads, complex transforms).
*   **System Co-design:** Essential. Example: DWRF feature flattening + coalesced reads + feature reordering optimized HDD access *and* reduced DPP CPU/memory needs, improving overall power efficiency (Table 12).

**8. Related Work**

*   Positions Meta's DSI pipeline (Hive/Tectonic/DWRF storage, DPP preprocessing) against traditional ETL, feature stores, other ML preprocessing (tf.data), caching systems, and training studies.
*   Highlights the novelty of characterizing the *full end-to-end DSI pipeline and workloads* for production DLRMs at scale.

**9. Conclusion**

*   **Main Takeaway:** DSI infrastructure is a critical, growing bottleneck for large-scale training capacity and power consumption.
*   **Contribution:** Presented and characterized Meta's production DSI pipeline, revealing unique workload characteristics.
*   **Call to Action:** Provided insights and research directions (benchmarks, co-design, heterogeneous hardware) to drive DSI optimization efforts.