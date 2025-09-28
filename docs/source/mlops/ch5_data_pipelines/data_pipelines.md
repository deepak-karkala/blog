# Data Engineering for Reliable ML Pipelines

##

**Chapter 5: Mise en Place – Data Engineering for Reliable ML Pipelines**

*(Progress Label: 📍Stage 5: The Prep Station Standardization)*

### 🧑‍🍳 Introduction: The Art of Preparation in the ML Kitchen

In any Michelin-starred kitchen, the phase of "mise en place" – literally "everything in its place" – is paramount. It's the disciplined, meticulous preparation of all ingredients before the first flame is lit for service. Vegetables are perfectly chopped, sauces are prepped, proteins are portioned. This groundwork ensures consistency, efficiency, and quality in the final dishes. Without excellent mise en place, even the most talented chef would struggle during the rush of service.

Similarly, in our MLOps kitchen, **Data Engineering for Reliable ML Pipelines** is our mise en place. It's the process of transforming raw data – sourced and understood in Chapter 3 – into clean, validated, well-structured, and feature-rich inputs ready for model training and, eventually, inference. This isn't just about one-off data cleaning; it's about building *automated, reproducible, and reliable pipelines* that consistently deliver high-quality data. As noted in "Fundamentals of Data Engineering," while data preparation is an intermediate phase, it's often the most expensive in terms of resources and time, and crucial for avoiding error propagation. 

This chapter will delve into designing robust data processing workflows, techniques for automated data cleaning, transformation, labeling, and splitting. We will emphasize the critical role of in-pipeline data validation, versioning, and lineage tracking to ensure our "ingredients" are always of the highest standard. Finally, we'll discuss building and orchestrating these pipelines using tools like Airflow, ensuring our MLOps kitchen runs like a well-oiled machine.

---

### Section 5.1: Designing Robust Data Processing Workflows (The Master Prep List & Station Setup)

Before writing a single line of pipeline code, it's essential to design the overall data processing workflow. This involves understanding the sequence of operations, dependencies, and the desired output format for your ML tasks.

*   **5.1.1 From Raw Data to ML-Ready Data: The Goal**
    *   Recap of data sources (from Chapter 3) and their initial state.
    *   Defining the target state: What should the data look like just before it's fed into a model training process? (e.g., cleaned text, numerical features, consistent labels).
*   **5.1.2 Key Stages in a Data Engineering Pipeline**
    *   **Data Engineering Lifecycle Stages for ML Pipelines**

        <img src="../../_static/mlops/ch5_data_pipelines/ml_engineering.jpg"/>

        - [Source: mlops.org: An Overview of the End-to-End Machine Learning Workflow](https://ml-ops.org/content/end-to-end-ml-workflow)

    *   Briefly map these to pipeline stages:
        *   Data Ingestion (covered in Ch3, but pipelines consume its output)
        *   Data Cleaning & Wrangling
        *   Data Transformation & Standardization
        *   Feature Engineering (High-level, deep dive in Ch5)
        *   Data Labeling (if not already done)
        *   Data Splitting
        *   Data Validation
*   **5.1.3 ETL vs. ELT in ML Pipelines (When to Transform)**
    *   **ETL (Extract, Transform, Load):** Transform data *before* loading it into the final data store used by ML. Common when complex transformations are needed or if the ML training environment prefers pre-transformed data.
    *   **ELT (Extract, Load, Transform):** Load raw or lightly processed data into a data lake/warehouse, and transformations are applied *as part of the ML training pipeline* or by query engines. Favored by modern cloud data warehouses.
    *   *Choice for "Trending Now":* We'll lean towards ELT – raw/cleaned data in S3, transformations (like TF-IDF for plots) happen within the training pipeline steps.
*   **5.1.5 Principles for Designing Data Workflows**
    *   **Modularity:** Break down complex processing into smaller, manageable, and testable steps/tasks.
    *   **Reusability:** Design components that can be reused across different pipelines or for different datasets if applicable.
    *   **Idempotency:** Ensure that running a pipeline step multiple times with the same input produces the same output, preventing errors from retries.
    *   **Testability:** Design each step to be easily testable in isolation.
    *   **Parameterization:** Make pipelines configurable (e.g., input data paths, processing parameters) rather than hardcoding values.

---

### Section 5.2: Data Cleaning and Wrangling in Pipelines (Washing, Peeling, and Chopping)

This is where raw ingredients are meticulously cleaned and prepared. In a pipeline context, these steps must be automated and robust.

*   **5.2.1 Automated Handling of Missing Values**
    *   *Detection:* Script checks for nulls/NaNs based on column expectations.
    *   *Strategies for Pipelines:*
        *   **Imputation:**
            *   Mean/Median/Mode: Calculate these stats *from the training set portion of the current pipeline run* and apply. Store these stats as pipeline artifacts.
            *   Constant Value: e.g., "Unknown", -1 (use with caution).
            *   Model-based Imputation (Advanced): Using another model (e.g., k-NN imputer, regression) to predict missing values. Computationally more expensive.
        *   **Deletion:**
            *   Column Deletion: If >X% missing in the current batch and column isn't critical.
            *   Row Deletion: If critical features are missing (use sparingly).
    *   *Considerations:* Impact on data distribution, potential bias introduction. Log imputation strategies used.
*   **5.2.2 Systematic Outlier Detection & Treatment**
    *   *Detection in Pipelines:*
        *   Statistical Methods: Z-score, IQR applied to incoming data batches. Thresholds defined from training data or domain knowledge.
        *   Clipping: Define min/max allowable values for features.
    *   *Automated Treatment:*
        *   Capping/Flooring: Replace outliers with a max/min threshold.
        *   Transformation: Log transforms can reduce outlier impact.
        *   Removal (Cautious): Remove rows with extreme outliers if justified.
    *   *Importance:* Outliers can skew model training and evaluation significantly.
*   **5.2.3 Scripting Data Formatting & Restructuring**
    *   *Data Type Correction:* Ensuring numerical features are float/int, categoricals are strings, dates are datetime objects. Automated checks and conversions.
    *   *Text Cleaning:* Lowercasing, removing special characters, HTML stripping, normalizing whitespace – all as scriptable functions.
    *   *Standardizing Units:* e.g., converting all monetary values to USD, all weights to kilograms.
    *   *Reshaping Data:* Pivoting, unpivoting, joining datasets if needed *within the pipeline* (e.g., merging movie metadata with review data for a specific run).

---

### Section 5.3: Data Transformation & Standardization for Pipelines (Seasoning and Standard Cuts)

Ensuring all ingredients are in a consistent format and scale for the the ML model.

*   **5.3.1 Implementing Scaling (Normalization, Standardization)**
    *   **Fit on Training Data ONLY:** Crucial to prevent data leakage. The scaler (e.g., `MinMaxScaler`, `StandardScaler` from scikit-learn) is fit *only* on the training split of the current pipeline run.
    *   **Transform All Splits:** The *same fitted scaler* is then used to transform the training, validation, and test splits (and later, inference data).
    *   **Persisting Scalers:** The fitted scaler object itself becomes a pipeline artifact, versioned and stored for later use in inference to ensure consistency.
*   **5.3.2 Automated Handling of Skewness**
    *   Apply transformations like Log, Square Root, or Box-Cox as part of a preprocessing step in the pipeline if skewness is detected (e.g., via statistical tests or profiling of the current batch).
*   **5.3.3 Encoding Categorical Features within Pipelines**
    *   **One-Hot Encoding:**
        *   Fit encoder on training data to learn categories.
        *   Handle unseen categories in validation/test/inference data (e.g., by ignoring, or having an "unknown" category if the encoder supports it).
        *   Persist fitted encoder.
    *   **Label Encoding:** Similar fitting/persisting strategy. Be mindful of ordinal vs. nominal data.
    *   **Target/Impact Encoding (Advanced):** Requires careful handling to prevent leakage, often involving fitting on folds within the training set.
    *   **Hashing Trick:** Useful for high-cardinality features where a full vocabulary isn't feasible to maintain or new categories appear frequently. Implemented directly in the pipeline.

---

### Section 5.5: Data Labeling at Scale & Programmatic Labeling (Adding Taste Profiles)

While some data comes with labels, ML pipelines often need to deal with generating or refining labels at scale.

*   **5.5.1 Integrating Human Labeling Workflows**
    *   *Pipeline Output to Labeling Tools:* Design pipeline steps that can export data needing labels to platforms (Label Studio, Scale AI, SageMaker Ground Truth).
    *   *Pipeline Input from Labeling Tools:* Design steps to ingest and validate labels returned from these platforms.
    *   *Active Learning Loops:* A pipeline can select uncertain samples, send them for labeling, and then trigger retraining once new labels are available.
*   **5.5.2 Leveraging Weak Supervision & Snorkel-like Systems**
    *   Define Labeling Functions (LFs) as code.
    *   A pipeline stage can apply LFs to unlabeled data to generate probabilistic labels.
    *   Another stage can run the label model (Snorkel) to denoise and combine these weak labels.
    *   Output: Programmatically labeled training data.
*   **5.5.3 Building Feedback Loops for Natural Label Generation**
    *   Capture user interactions (clicks, purchases, ratings) from production systems (e.g., via Kafka stream or logs).
    *   A pipeline step processes these interactions to infer labels (e.g., click = positive, no click after X time = negative for a recommendation).
    *   Handle delays and ambiguity in feedback.
*   **5.5.5 Data Augmentation as a Pipeline Step**
    *   Implement augmentation techniques (e.g., for text: back-translation, synonym replacement; for images: rotations, flips) as part of the data loading or preprocessing tasks within the training pipeline.
    *   Apply augmentations on-the-fly to training batches to avoid storing massively inflated datasets.

---

### Section 5.5: Data Splitting and Sampling in Automated Workflows (Portioning for Testing)

Reliable model evaluation depends on correct and consistent data splitting and sampling, automated within the pipeline.

*   **5.5.1 Ensuring Stratified and Time-Aware Splits**
    *   **Time-Series Data:** Always split chronologically to prevent future data leaking into training/validation. The pipeline must handle date/timestamp columns to enforce this.
    *   **Stratified Sampling:** For classification, ensure class proportions are maintained across splits, especially for imbalanced datasets. Implement as a scriptable step.
    *   **Grouped Splits:** If data has inherent groupings (e.g., multiple samples from the same user), ensure all samples from a group are in the same split to avoid leakage.
*   **5.5.2 Implementing Resampling Techniques (Over/Under Sampling)**
    *   **Apply *after* splitting:** Resample only the training set. Validation and test sets should reflect the true data distribution.
    *   Integrate libraries like `imbalanced-learn` as a pipeline step.
    *   Document and version the sampling strategy.

---

### Section 5.6: Data Validation as a Pipeline Stage (The Sous-Chef's Quality Check)

Before training, it's crucial to validate that the prepared data meets expectations. This is a key gate in a reliable ML pipeline.

*   **5.6.1 Automated Schema Validation**
    *   Define an expected schema (column names, data types, order).
    *   A pipeline step validates incoming data batches against this schema.
    *   Detects schema skew (e.g., new columns, missing columns, type changes).
*   **5.6.2 Statistical Property & Distribution Drift Checks**
    *   Compare statistics (mean, median, null %, unique values) of the current data batch against a baseline (e.g., statistics from the initial training dataset or a "golden" dataset).
    *   Detects data drift.
    *   **Common Data Validation Checks in a Pipeline**
        | Check Type         | Description                                                                | Action if Failed                  |
        | :----------------- | :------------------------------------------------------------------------- | :-------------------------------- |
        | Schema Adherence   | Data matches expected columns, types, order.                               | Halt pipeline, Alert, Investigate |
        | Null Percentage    | Percentage of nulls in a column within acceptable threshold.               | Alert, Investigate (Impute/Drop)  |
        | Cardinality        | Number of unique values in a categorical column within expected range.     | Alert, Investigate                |
        | Value Range        | Numerical values within min/max bounds.                                    | Alert, Investigate (Clip/Filter)  |
        | Distribution Drift | Statistical distance (e.g., KS-test, PSI) from baseline below threshold. | Alert, Investigate, Trigger Retrain |
*   **5.6.3 Tools for Data Validation in Pipelines**
    *   **Great Expectations:** Define expectations in JSON, integrate into Python/Spark pipelines. Generates validation reports.
    *   **TensorFlow Data Validation (TFDV):** Infers schema, computes stats, detects anomalies. Integrates with TFX.
    *   **Deequ (Apache Spark):** For data quality monitoring on large datasets in Spark.
*   **5.6.5 Actions on Validation Failure**
    *   Halt the pipeline to prevent training on bad data.
    *   Send alerts to the relevant team (Data Engineering, MLOps).
    *   Quarantine problematic data for investigation.
    *   Log validation results comprehensively.

---

### Section 5.7: Data Versioning & Lineage in Practice (for Pipelines) (Tracking Every Ingredient's Source and Prep)

Ensuring traceability and reproducibility for every dataset consumed and produced by your pipelines.

*   **5.7.1 Tools & Techniques**
    *   **DVC:** Our chosen tool. Use `dvc add` for datasets, `dvc repro` to run DVC-defined stages. Committing `.dvc` files to Git versions the data pointers.
    *   **Git LFS:** For versioning larger individual files directly in Git (less ideal for very large or frequently changing datasets).
    *   **Delta Lake / LakeFS / Nessie:** Provide Git-like operations (branching, merging, time-travel) directly on data lakes. More advanced, powerful for complex data ecosystems.
*   **5.7.2 Capturing Full Data Lineage** 
    *   **Input Data Version:** Which version of raw data was used for a pipeline run?
    *   **Processing Code Version:** Which Git commit of the processing scripts was used?
    *   **Output Data Version:** Which version of processed data/features was generated?
    *   **Linking to Model Training:** Which version of processed data was used to train which model version?
    *   This is often facilitated by integrating the data pipeline with the ML Metadata Store and Experiment Tracking tools.

---

### Section 5.8: Building and Orchestrating Data Pipelines (The Kitchen's Choreography)

Bringing all the data engineering steps together into automated, scheduled, and monitored workflows.

*   **5.8.1 Choosing Orchestration Tools (Revisited for Data Pipelines)**
    *   **Apache Airflow (Our Choice):**
        *   *Pros:* Mature, large community, extensive operators for various systems, Python-based DAG definition.
        *   *Cons:* Can be complex to set up/manage, scheduler as a single point of failure (pre-Airflow 2.0), lacks data-native constructs.
    *   *Alternatives:* Prefect, Dagster (more data-aware), Kubeflow Pipelines (Kubernetes-native), Cloud-native (AWS Step Functions, Azure Data Factory, Google Cloud Composer/Dataflow).
*   **5.8.2 Best Practices for Reusable and Testable Data Pipeline Components (Airflow Tasks/Operators)**
    *   **Parameterization:** Design tasks to accept parameters (input paths, dates, processing options) for flexibility. Use Airflow Variables and Connections.
    *   **Containerization of Tasks:** Package complex task logic with its dependencies in Docker containers. Airflow's `KubernetesPodOperator` or `DockerOperator` can run these. This ensures environment isolation and reproducibility.
    *   **Unit Testing for Tasks:** Test individual Python functions within Airflow operators.
    *   **Integration Testing for DAGs:** Test DAG structure, task dependencies, and small end-to-end runs with sample data.
    *   **Idempotency:** Design tasks so they can be safely retried.
    *   **Error Handling & Retries:** Configure Airflow's retry mechanisms, SLAs, and alerting on task failure.
    *   **Custom Operators/Hooks:** Develop reusable custom components for common data operations specific to your organization.

---

### Project: "Trending Now" – Building the Data Ingestion & Preparation Pipeline

This section will guide the reader through implementing the core data pipeline for our "Trending Now" application, applying the concepts discussed.

*   **5.P.1 Designing the Data Ingestion and Preprocessing Steps for Movie Plots and Reviews**
    *   Detailed Python functions for scraping (e.g., a simplified TMDb fetcher, a generic review site scraper outline).
    *   Pandas-based cleaning: HTML removal, text normalization, handling missing plots/reviews.
    *   Structuring the output as Parquet files with defined schemas.
*   **5.P.2 Strategy for Labeling Genres**
    *   Primary strategy: Use genre tags directly available from the metadata source (e.g., TMDb).
    *   Discussion: If source genres are missing/unreliable, how could weak supervision (e.g., keyword spotting in plots: "space" -> Sci-Fi, "love story" -> Romance) be prototyped as a fallback or enrichment? (Conceptual, no full Snorkel implementation).
*   **5.P.3 Implementing Data Validation for the Ingestion Pipeline**
    *   Using Great Expectations:
        *   Define expectations: `title` must exist, `plot_summary` should be string, `release_date` should be valid date.
        *   Create a validation step in the Airflow DAG.
*   **5.P.4 Setting up Data Versioning for Ingested and Processed Data**
    *   Show DVC commands: `dvc add data/raw/movies.parquet`, `dvc add data/processed/cleaned_reviews.parquet`.
    *   Commit `.dvc` files and `dvc.lock` to Git.
    *   Push data to S3 remote: `dvc push`.
*   **5.P.5 Orchestrating the Daily/Weekly Data Ingestion Pipeline with Airflow**
    *   Provide a sample `data_ingestion_dag.py` file for Airflow.
    *   Tasks:
        1.  `scrape_new_releases_task` (PythonOperator calling scraping script)
        2.  `scrape_reviews_task` (PythonOperator calling review scraping script)
        3.  `preprocess_data_task` (PythonOperator calling cleaning/transformation script)
        5.  `validate_data_task` (PythonOperator calling Great Expectations script/checks)
        5.  `version_data_task` (BashOperator running `dvc add` and `dvc push`)
    *   Set a daily or weekly schedule.
    *   Discuss basic error handling and alerting via Airflow.

---

### 🧑‍🍳 Conclusion: Mise en Place Complete – Ready for the Main Course!

A perfectly executed "mise en place" is the unsung hero of any great kitchen. It's the disciplined, systematic preparation that transforms raw ingredients into components perfectly prepped for the chef's creative touch. In this chapter, we've laid out the "mise en place" for our MLOps kitchen: Data Engineering for Reliable ML Pipelines.

We've explored how to design robust data processing workflows, automate crucial steps like cleaning, transformation, and validation, and manage the complexities of data labeling and splitting within a pipeline context. We underscored the importance of data versioning with DVC and data lineage to ensure our data "ingredients" are always traceable and reproducible. Finally, we've seen how to orchestrate these intricate preparations using tools like Airflow, ensuring our data is consistently prepped to the highest standard, ready for the next stage.

With our data diligently cleaned, transformed, validated, and versioned, our ingredients are now perfectly prepped. The kitchen stations are organized, and the prep list is complete. We are now ready to move to the truly creative part of our culinary journey: "Perfecting Flavor Profiles – Feature Engineering and Feature Stores."

---
