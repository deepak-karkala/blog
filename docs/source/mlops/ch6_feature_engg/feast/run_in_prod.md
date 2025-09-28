# Running Feast in Production

##

Transitioning Feast from a local sandbox to a robust production deployment involves several key considerations across infrastructure, data management, and model integration. This guide outlines the best practices and common patterns.

**Core Production Architecture Philosophy:** Feast is modular. You can pick and choose components and workflows that fit your specific needs. Not all systems will require every Feast capability (e.g., stream ingestion or online serving).

### 1. Automating Feature Definition Deployment

This focuses on managing your feature definitions (`FeatureView`, `Entity`, etc.) as code and ensuring your infrastructure reflects these definitions.

*   **1.1. Feature Repository (Git):**
    *   **Action:** Store all Feast feature definitions (`.py` files) in a Git repository.
    *   **Benefit:** Version control, collaboration, and auditable changes to your feature landscape.
*   **1.2. Database-Backed Registry:**
    *   **Action:** Switch from the default file-based registry to a more scalable SQL-based registry (e.g., backed by PostgreSQL, MySQL).
    *   **Benefit:** Improved scalability and concurrent access for production environments.
    *   **Note:** Primarily works with the Python Feature Server; Java Feature Server has limitations with SQL registries.
*   **1.3. CI/CD for Registry Updates:**
    *   **Action:** Set up a CI/CD pipeline (e.g., GitHub Actions, Jenkins) to automatically run `feast plan` (to preview changes) on pull requests and `feast apply` (to update the registry) on merges to the main branch.
    *   **Benefit:** Ensures the registry is always in sync with the version-controlled feature definitions.
*   **1.4. Multiple Environments (Staging/Production):**
    *   **Action:** Create separate Feast environments (e.g., `staging`, `production`) with their own offline stores, online stores, and registries.
    *   **Benefit:** Allows testing changes to feature definitions and infrastructure in `staging` before promoting to `production`, minimizing risk.
    *   Refer to specific how-to guides for structuring repositories for multiple environments.

### 2. Loading and Updating Data in the Online Store

Ensuring fresh features are available for low-latency serving is crucial. This process is called **materialization**.

*   **2.1. Scalable Materialization:**
    *   **Problem:** The default in-process materialization engine loads all data into memory, which doesn't scale for large datasets.
    *   **Solutions:**
        *   Use specialized, scalable materialization engines:
            *   **Snowflake Materialization Engine:** If offline/online stores are in Snowflake.
            *   **Bytewax Materialization Engine:** Good Kubernetes-native option if not using Snowflake.
        *   Write a **custom materialization engine** (e.g., using Spark, Ray, Flink) to leverage existing distributed processing infrastructure.
*   **2.2. Scheduled Materialization (e.g., with Airflow):**
    *   **Action:** Orchestrate and schedule materialization jobs.
        *   Simple cases: `cron` might suffice for infrequent jobs.
        *   Complex cases: Use job orchestrators like Airflow or Kubernetes Jobs for parallel execution, dependency management, and retries.
    *   **Airflow Integration:**
        *   Use `PythonOperator` to call Feast SDK methods like `store.materialize_incremental(datetime.datetime.now())` or `store.materialize(data_interval_start, data_interval_end)`.
        *   The Airflow worker needs appropriate permissions to the Feast registry (to read config and update materialization history) and data stores.
*   **2.3. Stream Feature Ingestion:**
    *   **Action:** Use Feast's Push API (`store.push()`) to ingest features from streaming sources (e.g., Kafka, Kinesis via Spark/Flink/Beam jobs) directly into the online and/or offline stores.
    *   **Benefit:** Provides near real-time feature updates.
*   **2.4. Scheduled Batch Transformations (e.g., Airflow + dbt):**
    *   **Action:** Feast does not orchestrate upstream batch transformation DAGs. Use tools like Airflow to schedule dbt (or Spark, etc.) jobs that prepare data in your data warehouse before Feast ingests or reads it.

### 3. Using Feast for Model Training

Leveraging Feast to create consistent and point-in-time correct training datasets.

*   **3.1. Generating Training Data:**
    1.  **Instantiate `FeatureStore`:**
        *   In your training pipeline, create a `FeatureStore` object. Ensure the pipeline has access to `feature_store.yaml` (or configure programmatically) which points to the production registry.
        ```python
        from feast import FeatureStore
        fs = FeatureStore(repo_path="production_repo/") # or use programmatic config
        ```
    2.  **Generate Entity DataFrame:**
        *   Provide a Pandas DataFrame containing entity IDs and `event_timestamp` columns.
        *   Alternatively, use a SQL query string to dynamically generate this entity list from your data sources.
    3.  **Retrieve Historical Features:**
        *   Use `fs.get_historical_features()` with the entity DataFrame/SQL and a `FeatureService` (recommended for versioning).
        ```python
        training_retrieval_job = fs.get_historical_features(
            entity_df=entity_df_or_sql_string,
            features=fs.get_feature_service("model_name_v1"),
        )
        # Option 1: In-memory training
        training_df = training_retrieval_job.to_df()
        model = ml_library.fit(training_df)
        # Option 2: Offload to storage for distributed training
        training_retrieval_job.to_remote_storage("s3://path/to/training_data/")
        ```
*   **3.2. Versioning Features for Models:**
    *   **Action:** Establish a convention linking `FeatureService` names to model versions. For example, if your model is `my_model` version `1`, the corresponding `FeatureService` could be `my_model_v1`.
    *   **Benefit:** Ensures that the exact set of features used for training a specific model version is used during its inference.
    *   **Example (with MLflow):**
        ```python
        # During inference
        model_name = "my_model"
        model_version = 1
        feature_service_name = f"{model_name}_v{model_version}"
        feature_vector = fs.get_online_features(
            features=fs.get_feature_service(feature_service_name),
            entity_rows=[{"driver_id": 1001}]
        ).to_dict()
        prediction = model.predict(feature_vector)
        ```
    *   **Access Control:** Training pipelines and model serving services typically only need *read access* to the Feast registry and data stores.

### 4. Retrieving Online Features for Prediction

Getting the latest feature values for real-time inference.

*   **4.1. Python SDK within an Existing Python Service:**
    *   **Action:** If your model serving application is in Python, directly use the Feast Python SDK (`fs.get_online_features(...)`).
    *   **Benefit:** Simplest approach, avoids deploying extra services. The SDK connects directly to the online store.
    *   **Drawback:** Service must be in Python.
*   **4.2. Deploy Feast Feature Servers on Kubernetes:**
    *   **Action:** For non-Python services or for a more microservice-oriented architecture, deploy the Feast Feature Server (Python or Java) on Kubernetes. Use the **Feast Operator** for managing these deployments.
    *   **Feast Operator:** A Kubernetes operator that simplifies deploying and managing `FeatureStore` custom resources (CRs), which in turn manage Feature Server deployments, `feast apply` jobs, etc.
    *   **Steps:**
        1.  Install `kubectl`.
        2.  Install the Feast Operator (`kubectl apply -f <operator_install_yaml_url>`).
        3.  Define and deploy a `FeatureStore` Custom Resource (CR) manifest. This CR specifies project name, registry, online/offline stores, and server configurations.
    *   **Scalability:** Scale the Feature Server deployment based on load, ensuring the backend online store can handle the increased traffic.

### 5. Using Environment Variables in `feature_store.yaml`

*   **Action:** Use `${ENV_VAR}` syntax within your `feature_store.yaml` to dynamically set configuration values (e.g., connection strings, API keys, server endpoints for different environments).
    ```yaml
    online_store:
        type: redis
        connection_string: ${REDIS_CONNECTION_STRING_PROD} # Injected at runtime
    ```
*   **Benefit:** Allows for flexible configuration across environments and secure injection of secrets without hardcoding them in Git.

### Summary of a Production Setup:

1.  **CI/CD:** Git repository for feature definitions -> CI (e.g., GitHub Actions) runs `feast apply` -> Updates database-backed Feast Registry.
2.  **Data Ingestion:**
    *   **Batch:** Airflow (or similar) orchestrates upstream transformations (e.g., dbt, Spark) AND `feast materialize` jobs (using a scalable materialization engine like Snowflake or Bytewax/custom) to load data from DWH to Online Store.
    *   **Stream:** Existing streaming pipelines (Spark, Flink, Beam) use Feast's Push API to send fresh feature values to both Online and Offline Stores.
3.  **Model Training:** Training pipelines use Feast Python SDK to call `get_historical_features` against the Offline Store, generating training data using `FeatureService` definitions.
4.  **Online Serving:**
    *   Model serving applications (Python-based) directly use Feast Python SDK to call `get_online_features`.
    *   OR, model serving applications (any language) call a deployed Feast Feature Server (on Kubernetes, managed by Feast Operator) via HTTP/gRPC to get online features.
