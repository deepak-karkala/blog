# Coveo: MLOPs at reasonable scale

**1. Introduction**

*   **Core Problem:** Cutting-edge research in e-commerce recommender systems is largely concentrated in a few big players, creating a high barrier to entry for mid-to-large shops ("reasonable scale").
*   **Challenges for "Reasonable Scale":** Lack of open/representative datasets, non-relevant benchmarks, expensive computational resources, and lack of best practices/toolchains for productionizing models.
*   **Paper's Goal:**
    *   Define constraints and opportunities at "reasonable scale" (dozens of ML engineers, $10-500M revenue/year, TBs of behavioral data/year).
    *   Showcase an end-to-end, mostly open-source, serverless stack for productionizing recommenders with minimal infrastructure work.
    *   Provide actionable insights for practitioners with limited resources to make adoption choices in the evolving MLOps landscape.

**2. Principles for Models at Reasonable Scale**

These principles guide strategic decisions for ML in resource-constrained environments:

*   **(1) Data is King:** Accessible, clean, standardized data provides the biggest marginal gain, more so than small modeling adjustments. Proprietary data flows are strategically important as modeling becomes commoditized.
*   **(2) ELT is Better than ETL:** Clear separation of data ingestion (landing raw, immutable records) and processing (transformations) leads to reliable, reproducible pipelines.
*   **(3) PaaS/FaaS is Better than IaaS:** Maintaining infrastructure is costly and unnecessary at reasonable scale. Use fully-managed services for computation, auto-scaling, replication, etc. Focus scarce, high-quality engineering resources on ML, not infrastructure.
*   **(4) Distributed Computing is the Root of All Evil (at this scale):** Systems like Spark, even managed, are slow, hard to debug, and force unfamiliar programming patterns. At "reasonable scale," there are better tools (e.g., modern data warehouses that abstract distributed nature) for heavy lifting, freeing teams from distributed computing overhead.
*   **Key Takeaway for "Reasonable Scale":** The scale makes many powerful tools affordable and can streamline complexities, empowering small teams.

**3. Desiderata for In-Session Recommendations**

Functional requirements for an in-session recommendation system, from data ingestion to serving:

*   **Raw Data Ingestion:** Scalable collection and safe storage of shopper data, ensuring re-playability from raw events. (Fig 1: events from browser (1) -> raw data (2) -> raw table (3)).
*   **Data Preparation:** Includes data visualizations, BI dashboards, data quality checks, data wrangling, and feature preparation. (Fig 1: raw table (3) -> transformed table (4)).
*   **Model Training:** Includes model training, hyperparameter search, and behavioral checklists. (Fig 1: transformed table (4) -> model training (5)).
*   **Model Serving:** Serving predictions at scale. (Fig 1: model (5) -> recommendations (6)).
*   **Orchestration:** Monitoring UI, automated retries, and notification system.

**4. An End-to-End Stack (Fig. 2)**

A modern, mostly serverless and open-source data stack for "reasonable scale":

*   **Raw Data Ingestion:**
    *   **Technology:** AWS Lambda (PaaS, auto-scaling).
    *   **Mechanism:** Receives shopper events (e.g., from JS SDK).
*   **Storage:**
    *   **Technology:** Snowflake (PaaS-like data warehouse).
    *   **Mechanism:** Stores raw data in an append-only fashion.
*   **Data Preparation (for Visualization and QA):**
    *   **Technology:** dbt (builds SQL-based DAG of transformations).
    *   **Mechanism:** Prepares normalized tables for BI and quality checks.
*   **Model Training:**
    *   **Technology:** Metaflow (defines ML tasks as a DAG, abstracts cloud execution including GPU provisioning via decorators).
    *   **Mechanism:** Reads prepared data (often from Snowflake after dbt transformations) for training.
*   **Model Serving:**
    *   **Technology:** AWS SageMaker (hosted, auto-scaling, various hardware options).
    *   **Mechanism:** Deploys models versioned and artifacted by Metaflow. (Note: Metaflow's artifacting makes deployment options flexible).
*   **Orchestration:**
    *   **Technology:** Prefect (offers a hosted version for job monitoring and admin).
    *   **Mechanism:** Manages the execution of dbt, Metaflow, and other pipeline jobs.
*   **Crucial Observations:**
    *   **No Direct Infrastructure Maintenance by ML Engineers:** All tools are managed and scaled automatically by cloud providers or managed services.
    *   **Distributed Computing Abstracted:** Snowflake handles the distributed nature of "reasonable size" data via plain SQL. Downstream tasks (e.g., Metaflow training) can often run "locally" (on single, appropriately sized instances, including GPUs managed by Metaflow/SageMaker).
    *   **Mostly Open Source or Substitutable:** Warehouse aside (Snowflake), most tools are open source (dbt, Metaflow, Prefect) or have open-source alternatives that could fit into a similar serverless paradigm.

**5. Conclusion**

*   **Main Argument:** Infrastructure and architectural barriers in ML can be overcome by embracing a serverless paradigm, especially at "reasonable scale."
*   **Proven Capability:** The proposed (or similar) stack can process terabytes of data (raw events to GPU-powered recommendations) with minimal to no DevOps work, relying heavily on open-source solutions.
*   **Focus Shift:** Frees up ML teams to focus on data and model work, rather than infrastructure ("you do not need a bigger boat").

**Appendices**

*   **Appendix A (Research Distribution):** Highlights that most e-commerce ML research comes from a few large, public, B2C companies, underscoring the democratization problem the paper addresses.
*   **Appendix B (Bio):** Provides context on the author's experience in building and shipping ML models at scale.