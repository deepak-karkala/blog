# ML Expt tracking, Data Lineage, Model Registry

### I. ML Experiment Tracking: The Foundation of Iterative Development

(Sources: Neptune "ML Experiment Tracking", Neptune "ML Experiment Management", Google MLOps Guide Fig 4)

1.  **What is ML Experiment Tracking?**
    *   **Definition:** The systematic process of saving all relevant information (metadata) associated with each machine learning experiment run.
    *   **Goal:** To enable reproducibility, comparison, debugging, collaboration, and informed decision-making throughout the model development lifecycle.

2.  **Why Does It Matter? The MLOps Lead's Perspective:**
    *   **Organization & Discoverability:** Centralizes scattered experiment results, regardless of where they were run (local, cloud, notebooks). Prevents "lost" work and tribal knowledge.
    *   **Reproducibility:** Enables re-running experiments by capturing code, data, environment, and parameters. Critical for debugging and validation.
    *   **Efficient Comparison & Analysis:** Allows side-by-side comparison of metrics, parameters, learning curves, visualizations, and artifacts. Speeds up identification of what works and what doesn't.
    *   **Collaboration & Knowledge Sharing:** Provides a single source of truth for the team. Facilitates easy sharing of results and progress with stakeholders via persistent links or dashboards.
    *   **Live Monitoring & Resource Management:** Allows real-time tracking of running experiments, early stopping of unpromising runs, and monitoring hardware consumption for efficiency.
    *   **Debugging:** Helps pinpoint issues by comparing a failed run to a successful one, looking at code diffs, environment changes, or data shifts.

3.  **What to Track: The MLOps Lead's Checklist** (Neptune "ML Experiment Tracking", Chip Huyen Ch. 6)

    *   **Core Essentials (Must-Haves):**
        *   **Code Versions:** Git commit hashes, script snapshots (especially for uncommitted changes or notebooks). Tools like `nbdime`, `jupytext`.
        *   **Data Versions:** Hashes of datasets/data pointers (e.g., MD5, DVC tracked files). Crucial to link model performance to the exact data used. (See Section II: Data Lineage & Provenance).
        *   **Hyperparameters:** All parameters influencing the experiment (learning rate, batch size, architecture details, feature engineering steps). Log explicitly, avoid "magic numbers". Config files (YAML, Hydra) are good practice.
        *   **Environment:** Dependencies (e.g., `requirements.txt`, `conda.yml`, Dockerfile). Ensures consistent runtime.
        *   **Evaluation Metrics:** Key performance indicators on training, validation, and (sparingly) test sets. Log multiple relevant metrics.

    *   **Highly Recommended:**
        *   **Model Artifacts:** Serialized model weights/checkpoints (e.g., .h5, .pth, .pkl), especially the best performing ones.
        *   **Learning Curves:** Metrics over epochs/steps for both training and validation sets.
        *   **Performance Visualizations:** Confusion matrices, ROC/PR curves, prediction distributions.
        *   **Run Logs:** Standard output/error streams.
        *   **Hardware Consumption:** CPU/GPU/memory usage during training.
        *   **Experiment Notes/Tags:** Qualitative observations, hypotheses being tested.

    *   **Advanced/Context-Specific:**
        *   **Feature Importance/Explanations:** SHAP values, LIME outputs, attention maps.
        *   **Sample Predictions:** Examples of good/bad predictions, especially for vision or NLP tasks.
        *   **Gradient Norms/Weight Distributions:** For deep learning debugging.
        *   **For LLMs:** Prompts, chain configurations, specific metrics (ROUGE, BLEU), inference time.

4.  **Setting Up Experiment Tracking: Build vs. Buy vs. Self-Host** (Neptune "ML Experiment Tracking")

    | Approach                          | Pros                                                                     | Cons                                                                                                    | MLOps Lead Considerations                                                                                                                               |
    | :-------------------------------- | :----------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------ |
    | **Spreadsheets/Naming Conventions** | Simple to start.                                                         | Error-prone, not scalable, hard to collaborate, no live tracking, poor for complex metadata.              | **Strongly discourage** for any serious project. Only for very small, solo, short-term explorations.                                                    |
    | **Git for Metadata Files**        | Leverages existing VCS skills.                                           | Not designed for ML artifacts, poor comparison for >2 runs, difficult organization for many experiments. | Better than spreadsheets but quickly hits limitations for ML-specific needs.                                                                              |
    | **Build Your Own Tracker**        | Full control, tailored to specific needs.                                | High development & maintenance effort, risk of reinventing the wheel, requires diverse engineering skills. | Only if existing tools are truly insufficient AND significant engineering resources are available. Often a distraction from core ML work.                   |
    | **Self-Host Open Source Tool**    | No vendor lock-in, data stays on-premise, customizable.                  | Maintenance overhead (infra, updates, security), may lack dedicated support.                              | Suitable if strict data residency is a must or high customization is needed, and team has infra/ops capabilities. Assess community support.               |
    | **SaaS Experiment Tracker**       | Fully managed, scalable, expert support, rich features, rapid iteration. | Vendor dependency, data on third-party cloud (usually with strong security/compliance).                   | Often the most efficient for teams wanting to focus on ML. Evaluate based on features, integrations, security, pricing, and support. Examples: Neptune.ai. |

    *   **Key for MLOps Lead:** Champion the adoption of a dedicated experiment tracking tool. The productivity gains and risk reduction far outweigh the effort of manual methods.

---

### II. Data Lineage & Provenance: Understanding the "Story Behind the Data"

(Source: Neptune "Data Lineage in ML")

1.  **Definitions:**
    *   **Data Lineage:** Tracks data's journey from origin to consumption, including transformations and processes it underwent. Focuses on **metadata (data about the data)**.
    *   **Data Provenance:** Broader than lineage. Includes lineage but also tracks systems and processes that *influence* the data.

2.  **Why It's Crucial for MLOps:**
    *   **Reproducibility:** Essential for reproducing models and debugging. If input data or its processing changes, the model outcome will change.
    *   **Impact Analysis:** Understand how changes in upstream data sources or processing steps affect downstream models and business outcomes.
    *   **Debugging:** Trace back data issues (e.g., data quality degradation, schema changes) to their source.
    *   **Governance & Compliance:** Provides audit trails for data usage, transformations, and model training, critical for regulated industries.
    *   **Data Quality & Trust:** Helps ensure data integrity by understanding its origins and transformations.
    *   **Efficiency:** Prevents re-computation or re-engineering of data pipelines if lineage is clear.

3.  **Methods of Data Lineage Tracing:**
    *   **Data Tagging:** Relies on transformation tools consistently tagging data. Best for closed systems.
    *   **Self-Contained Lineage:** Lineage within a controlled data environment (e.g., data lake, data warehouse).
    *   **Parsing:** Analyzing code (SQL, Python) and transformation logic to infer lineage. Can be complex and language-dependent.
    *   **Pattern-Based Lineage:** Infers lineage by observing data patterns. Technology-agnostic but can miss code-driven transformations.

4.  **Data Lineage Across the ML Pipeline:**
    *   **Data Gathering:** Track source systems, ingestion methods, initial validation.
    *   **Data Processing:** Log all transformations, filters, feature engineering steps, versions of scripts.
    *   **Data Storing & Access:** Track storage locations, access permissions, data versions.
    *   **Data Querying:** For training data generation, log the queries and data snapshots used.

5.  **Best Practices for Data Lineage (from an MLOps perspective):**
    *   **Automation:** Manual lineage tracking is not scalable. Leverage tools that automatically capture lineage or integrate with lineage systems.
    *   **Granularity:** Track lineage at a level that is useful for debugging and reproducibility (e.g., dataset version, feature transformation script version).
    *   **Integration with Experiment Tracking:** Link experiment runs to specific versions of datasets and preprocessing code.
    *   **Integration with Feature Stores:** Feature stores inherently manage lineage for features.
    *   **Metadata Validation:** Ensure captured lineage information is accurate and complete.

6.  **Tools for Data Lineage (often overlapping with data cataloging or broader data management):**
    *   Talend Data Catalog, IBM DataStage, Datameer
    *   Open-source: Apache Atlas, OpenLineage, Marquez
    *   **Experiment trackers (like Neptune.ai)** can capture crucial parts of data lineage by versioning data inputs (hashes, paths) and code.
    *   **DVC (Data Version Control):** While primarily for data versioning, DVC pipelines (`dvc.yaml`) implicitly define data lineage for stages.

    ```mermaid
    graph LR
        subgraph "Data Sources"
            DS1[Source DB]
            DS2[API Feed]
            DS3[File Uploads]
        end
        subgraph "Ingestion & Staging"
            I[Ingest Process]
            S[Staging Area/Lake]
        end
        subgraph "Transformation & Feature Engineering"
            T1[Preprocessing Script V1]
            T2[Feature Engineering Script V2.1]
            FS[Feature Store]
        end
        subgraph "Model Training & Experimentation"
            D_Train[Training Dataset V3.2]
            M_Exp[Experiment Run ID: exp_abc]
            M_Art[Model Artifact: model_v1.2.pkl]
        end
        subgraph "Deployment & Serving"
            Dep[Deployed Model V1.2]
            Pred[Predictions]
        end

        DS1 -- Ingested_by --> I
        DS2 -- Ingested_by --> I
        DS3 -- Ingested_by --> I
        I -- Loads_to --> S
        S -- Input_for --> T1
        T1 -- Output_to --> S_Processed[Processed Data V1]
        S_Processed -- Input_for --> T2
        T2 -- Populates --> FS
        FS -- Source_for --> D_Train
        D_Train -- Used_in --> M_Exp
        M_Exp -- Produces --> M_Art
        M_Art -- Registered_and_Deployed_as --> Dep
        Dep -- Generates --> Pred

        classDef data fill:#lightblue,stroke:#333,stroke-width:2px;
        classDef process fill:#lightgreen,stroke:#333,stroke-width:2px;
        classDef artifact fill:#lightyellow,stroke:#333,stroke-width:2px;

        class DS1,DS2,DS3,S,S_Processed,D_Train,FS data;
        class I,T1,T2,M_Exp,Dep process;
        class M_Art,Pred artifact;
    ```

---

### III. ML Model Registry: Centralized Governance and Lifecycle Management

(Sources: Neptune "ML Model Registry", Google MLOps Guide Fig 4, Practitioners Guide to MLOps)

1.  **What is a Model Registry?**
    *   **Definition:** A centralized system for storing, versioning, managing, and governing trained machine learning models and their associated metadata throughout their lifecycle (from development to production and retirement).
    *   **Distinction from Model Repository/Store:** A repository might just store model files. A registry adds lifecycle management, versioning, metadata, and governance. A model store is a broader concept, potentially including a registry.

2.  **Why a Model Registry is Essential for MLOps:**
    *   **Centralized Storage & Discoverability:** Single source of truth for all trained models, making them easy to find, audit, and reuse.
    *   **Version Control for Models:** Tracks different versions of a model, allowing rollback and comparison. Essential as models are retrained or improved.
    *   **Standardized Hand-off:** Bridges the gap between data science (experimentation) and MLOps/engineering (deployment). Provides a clear point for promoting models.
    *   **Governance & Compliance:** Facilitates review, approval, and auditing of models before deployment. Stores documentation (e.g., model cards) and evidence of validation.
    *   **Automation & CI/CD/CT Integration:** Enables automated pipelines to register new model versions, trigger deployment workflows, and manage model stages (e.g., staging, production, archived).
    *   **Improved Security:** Can manage access controls for models, especially those trained on sensitive data.

3.  **Key Features and Functionalities of a Model Registry:** (Practitioners Guide to MLOps, Neptune "ML Model Registry")
    *   **Model Registration:** Ability to "publish" a trained model from an experiment tracking system or training pipeline.
    *   **Model Versioning:** Automatically assigns and tracks versions for each registered model.
    *   **Metadata Storage:** Stores comprehensive metadata:
        *   Link to the experiment run that produced it (lineage to code, data, params).
        *   Evaluation metrics (offline and online).
        *   Model artifacts (weights, serialized model file).
        *   Runtime dependencies (e.g., library versions).
        *   Model documentation (model cards, intended use, limitations).
        *   Owner, creation date, stage.
    *   **Model Staging & Transitions:** Defines and manages model lifecycle stages (e.g., "Development", "Staging", "Production", "Archived"). Supports workflows for promoting/demoting models.
    *   **API Access:** Programmatic interface for CI/CD systems, monitoring tools, and serving platforms to interact with the registry.
    *   **UI for Management:** A web interface for browsing, searching, comparing, and managing models and their versions.
    *   **Annotation & Tagging:** Ability to add custom tags and descriptions.
    *   **Access Control:** Manages permissions for who can register, approve, or deploy models.

4.  **Model Registry in the MLOps Workflow:** (Google MLOps Levels, Practitioners Guide to MLOps Fig 3, 15)

    ```mermaid
    graph TD
        A[Experimentation & Training Pipeline] -->|Trained Model & Metadata| B(Model Registry);
        B -- Stage: Staging --> C{Validation & QA};
        C -- Approved --> D[CI/CD for Deployment];
        D -- Deploy --> E(Production Serving Environment);
        E -- Feedback/Metrics --> F(Model Monitoring);
        F -- Performance Degradation --> A;
        B -- Discover/Fetch Model --> E;

        subgraph "Model Lifecycle Stages within Registry"
            direction LR
            Dev[Development Models]
            Staging[Staging Models]
            Prod[Production Models]
            Arch[Archived Models]
            Dev --> Staging;
            Staging --> Prod;
            Prod --> Arch;
        end
    ```
    *   **MLOps Level 0 (Manual):** Data scientist manually registers the model. Ops team manually pulls for deployment.
    *   **MLOps Level 1 (ML Pipeline Automation):** Automated training pipeline registers validated models. Deployment might still be manual or semi-automated.
    *   **MLOps Level 2 (CI/CD Pipeline Automation):** Fully automated CI/CD pipeline interacts with the registry to manage model promotion and deployment based on triggers and approvals.

5.  **Build vs. Maintain vs. Buy for Model Registry:** (Similar considerations as Experiment Tracking)
    *   **Building:** Very complex due to diverse needs (storage, API, UI, versioning, workflow). Rarely justifiable.
    *   **Maintaining Open Source (e.g., MLflow Model Registry):** Offers good features. Requires infra setup, maintenance, and expertise. Good for teams wanting control and having ops capabilities.
    *   **Buying/SaaS (e.g., Verta.ai, Neptune.ai, cloud provider registries like Vertex AI Model Registry, SageMaker Model Registry):**
        *   **Pros:** Fully managed, feature-rich, vendor support, faster to get started.
        *   **Cons:** Potential vendor lock-in, cost, data residency concerns for some.
    *   **MLOps Lead's Role:** Evaluate based on team size, MLOps maturity, existing stack, budget, and specific governance/compliance needs. Integration with existing experiment tracking and deployment tools is key.

---

### IV. Connecting the Dots: The MLOps Lead's Unified View

Experiment tracking, data lineage, and model registries are not isolated components but interconnected pillars of a mature MLOps ecosystem.

*   **Experiment Tracking** feeds into the **Model Registry**: Successful experiments yield candidate models that are registered. The metadata logged during tracking (parameters, data versions, code versions, metrics) becomes crucial for the registry entry, providing lineage and context.
*   **Data Lineage** underpins both: Knowing what data (and its transformations) went into an experiment run (tracked) and thus into a registered model is fundamental for reproducibility, debugging, and governance.
*   **Model Registry** enables **Deployment and Monitoring**: It provides a stable, versioned source for deployment systems. Monitoring systems feedback performance metrics to the registry, informing decisions about retraining or rollback.

**MLOps Lead's Strategic Decisions Framework:**

1.  **Define the "What" and "Why" for Tracking:**
    *   What metadata is *essential* for your team to reproduce, debug, and compare experiments effectively? (Start with the core list, expand as needed).
    *   Why is this specific piece of metadata important for your project's goals (e.g., compliance, debugging speed, performance improvement)?

2.  **Establish Data Handling Protocols:**
    *   How will data versions be managed and linked to experiments? (DVC, S3 versioning + hashes, feature store).
    *   How will data lineage be captured or inferred for key datasets used in model training?

3.  **Design the Model Lifecycle Flow:**
    *   What are the stages a model goes through from experiment to production (and potentially archive)?
    *   Who is responsible for approvals at each stage? What are the criteria?
    *   How will models be promoted through these stages (manual, semi-automated, fully automated via CI/CD)?

4.  **Tooling Selection - Holistic View:**
    *   Does your chosen experiment tracker integrate well or offer model registry capabilities?
    *   Does your model registry integrate with your deployment and monitoring tools?
    *   Does your data infrastructure support adequate lineage tracking?
    *   Consider the overall MLOps stack and aim for seamless integration rather than siloed tools.

    ```mermaid
    graph LR
        subgraph "Development & Experimentation Phase"
            A[Ideation/Hypothesis] --> B(Code Versioning - Git);
            B --> C(Data Preparation & Versioning - DVC/Feature Store);
            C --> D(Hyperparameter Configuration - YAML/Hydra);
            D --> E(Environment Setup - Docker/Conda);
            E --> F[ML Experiment Run];
            F --> G[Experiment Tracking System - Neptune/MLflow];
            G -- Log --> H(Code Hash);
            G -- Log --> I(Data Hash/Path);
            G -- Log --> J(Parameters);
            G -- Log --> K(Environment Config);
            G -- Log --> L(Metrics);
            G -- Log --> M(Model Artifacts/Checkpoints);
            G -- Log --> N(Visualizations);
        end

        subgraph "Model Governance & Lifecycle Management"
            O[Model Registry - Neptune/MLflow/Vertex AI];
            M -->|Register Model| O;
            L -- Link to Model Version --> O;
            I -- Link to Model Version --> O;
            H -- Link to Model Version --> O;
            J -- Link to Model Version --> O;
            O -- Model Stages --> P{Staging};
            P -- Validation/QA --> Q{Production};
            Q -- Trigger Retraining/Rollback --> F;
            Q -- Serve for Inference --> R[Deployment Platform];
        end

        subgraph "Data Lineage"
            S[Source Data Systems] --> T(ETL/Data Pipelines);
            T -- Transformation Logic --> U(Processed Data for Training);
            U --> C;
            classDef MLOpsTool fill:#f9f,stroke:#333,stroke-width:2px;
            class G,O MLOpsTool;
        end

        R --> V(Application / End Users);
        V -- Feedback / New Data --> S;
    ```