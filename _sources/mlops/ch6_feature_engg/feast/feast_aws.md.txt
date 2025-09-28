# Running Feast with AWS

##

This guide outlines the typical workflow for installing, configuring, deploying, and using Feast in cloud environments like Snowflake, GCP, or AWS.

### 1. Install Feast

*   **Base Installation:**
    ```bash
    pip install feast
    ```
*   **Cloud-Specific Extras:** Install additional dependencies based on your chosen cloud provider and online/offline stores.
    *   **Snowflake:**
        ```bash
        pip install 'feast[snowflake]'
        ```
    *   **GCP (BigQuery, Datastore/Firestore):**
        ```bash
        pip install 'feast[gcp]'
        ```
    *   **AWS (Redshift, DynamoDB):**
        ```bash
        pip install 'feast[aws]'
        ```
    *   **Redis (e.g., AWS ElastiCache or standalone):**
        ```bash
        pip install 'feast[redis]'
        ```
*   **Purpose:** Ensures you have the necessary client libraries to interact with your chosen backend services.

### 2. Create a Feature Repository

A feature repository is a directory containing your Feast configurations and feature definitions.

*   **Action:** Use the `feast init` command to bootstrap a new repository.
    *   **Basic (local defaults):**
        ```bash
        feast init
        # Creates a directory like "my_feature_repo"
        ```
    *   **Cloud-Specific Templates:** Use the `-t` flag for templates pre-configured for specific providers. These will prompt for necessary credentials and connection details.
        ```bash
        feast init -t snowflake
        feast init -t gcp
        feast init -t aws
        ```
*   **Output:** `feast init` creates:
    *   A project directory (e.g., `my_feature_repo/`).
    *   `feature_store.yaml`: Configuration file for your feature store (provider, registry, online/offline stores).
    *   `example.py` (or similar): Python file with example feature definitions (`Entity`, `DataSource`, `FeatureView`).
    *   `data/` directory: Often contains sample data (e.g., a Parquet file) for the examples.
*   **Next Steps:**
    *   `cd my_feature_repo`
    *   Initialize a Git repository (`git init`) and commit these files. This is crucial for version control.
    *   Modify `example.py` with your actual feature definitions.
    *   Adjust `feature_store.yaml` to point to your production/staging cloud resources.

### 3. Deploy a Feature Store

This step registers your feature definitions with Feast and sets up any necessary infrastructure in your cloud environment based on your `feature_store.yaml` and definitions.

*   **Action:** Run `feast apply` from within your feature repository directory.
    ```bash
    feast apply
    ```
*   **Effect:**
    *   Parses your feature definition files (e.g., `example.py`).
    *   Updates the Feast registry (e.g., creates/updates `registry.pb` file in GCS/S3, or writes to a SQL registry).
    *   May create tables or other resources in your configured online/offline stores if they don't exist (behavior depends on the provider and store types).
*   **Important:** `feast apply` **does not** load data into the online store. It only sets up the definitions and metadata.
*   **Cleaning Up (Caution!):**
    *   `feast teardown` will attempt to remove infrastructure created by `feast apply`. **This is irreversible and will delete data/tables.** Use with extreme caution, especially in production.

### 4. Build a Training Dataset (Historical Feature Retrieval)

Feast enables the creation of point-in-time correct datasets for model training.

*   **Prerequisites:**
    *   Feature views must be defined and registered (`feast apply`).
    *   Historical feature data must exist in your offline store (e.g., tables in BigQuery, Snowflake, or files in S3).
*   **Steps:**
    1.  **Define Feature References or Use a Feature Service:**
        *   Specify which features you need, typically as a list of strings (`"feature_view_name:feature_name"`) or by referencing a pre-defined `FeatureService`.
        ```python
        # Using feature references
        feature_refs = [
            "driver_hourly_stats:conv_rate",
            "driver_hourly_stats:acc_rate"
        ]
        # Or, using a FeatureService (recommended for production)
        # feature_service = fs.get_feature_service("my_model_v1_features")
        ```
    2.  **Create an Entity DataFrame:**
        *   This DataFrame tells Feast *which entities* and *at what points in time* you need features for.
        *   It **must** contain:
            *   An `event_timestamp` column (Pandas Timestamps or datetime objects).
            *   Columns for all join keys of the entities involved in the selected features (e.g., `driver_id`).
        *   **Options:**
            *   **Pandas DataFrame:** Create it in your Python script. May require uploading to the offline store for some providers, which can be slow.
              ```python
              import pandas as pd
              from datetime import datetime
              entity_df = pd.DataFrame({
                  "event_timestamp": [pd.Timestamp(datetime.now(), tz="UTC")],
                  "driver_id": [1001]
              })
              ```
            *   **SQL Query (String):** Provide a SQL query that returns the entity keys and timestamps. This is often more efficient as the data stays within the data warehouse. Only works if all feature views are in the same offline store (e.g., all in BigQuery).
              ```python
              entity_df_sql = "SELECT event_timestamp, driver_id FROM my_project.my_labels_table WHERE ..."
              ```
    3.  **Launch Historical Retrieval:**
        *   Instantiate `FeatureStore` and call `get_historical_features()`.
        ```python
        from feast import FeatureStore
        fs = FeatureStore(repo_path="path/to/your/feature_repo/") # Or programmatic config

        training_job = fs.get_historical_features(
            features=feature_refs, # or features=feature_service
            entity_df=entity_df # or entity_df=entity_df_sql
        )
        training_df = training_job.to_df() # Convert to Pandas DataFrame
        # training_job.to_remote_storage() # Or save directly to cloud storage
        ```
*   **Outcome:** `training_df` will contain the original columns from your `entity_df` plus the joined feature values, all point-in-time correct.

### 5. Load Data into the Online Store (Materialization)

To serve features at low latency for online predictions, you need to load them from your offline store (batch sources) into an online store.

*   **Prerequisites:**
    *   Feature views must be defined and registered (`feast apply`).
    *   An online store must be configured in `feature_store.yaml`.
*   **Materialization Commands (CLI):**
    1.  **`feast materialize <start_date> <end_date>`:**
        *   Loads the latest feature values within the specified historical time range from batch sources into the online store.
        *   Example: `feast materialize 2021-04-07T00:00:00 2021-04-08T00:00:00`
        *   Can specify specific views: `--views driver_hourly_stats`
        *   This command is **stateless**. It's best used with an external scheduler (like Airflow) that manages the time ranges for each run.
    2.  **`feast materialize-incremental <end_date>` (Alternative):**
        *   Loads only *new* data that has arrived in batch sources up to the specified `end_date`.
        *   Example: `feast materialize-incremental 2021-04-08T00:00:00`
        *   This command is **stateful**. Feast tracks the last materialization timestamp for each feature view in the registry.
        *   On the first run, it materializes from the oldest timestamp in the source up to `end_date`. Subsequent runs use the previous `end_date` as the new start time.
*   **Programmatic Materialization (SDK):**
    *   `store.materialize(start_date, end_date, feature_views=["my_fv"])`
    *   `store.materialize_incremental(end_date, feature_views=["my_fv"])`
*   **Scheduling:** Materialization is typically run on a schedule (e.g., daily, hourly) using orchestrators like Airflow.

### 6. Read Features from the Online Store (Online Serving)

Once data is in the online store, your models can retrieve the latest feature values for real-time predictions.

*   **Prerequisites:**
    *   Features must be materialized into the online store.
*   **Steps (Python SDK):**
    1.  **Define Feature References or Use a Feature Service:**
        *   Specify the features needed for prediction. This list often comes from the model training phase and should be packaged with the deployed model.
        ```python
        features_for_prediction = [
            "driver_hourly_stats:conv_rate",
            "driver_hourly_stats:acc_rate"
        ]
        # Or use a FeatureService:
        # feature_service = fs.get_feature_service("my_model_v1_features_online")
        ```
    2.  **Provide Entity Rows:**
        *   A list of dictionaries, where each dictionary represents an entity (or set of entities for composite keys) for which you need features.
        *   **No `event_timestamp` is needed** because you're fetching the *latest* values.
        ```python
        entity_rows = [
            {"driver_id": 1001}, # For driver 1001
            {"driver_id": 1002}  # For driver 1002
        ]
        ```
    3.  **Read Online Features:**
        *   Instantiate `FeatureStore` and call `get_online_features()`.
        ```python
        from feast import FeatureStore
        fs = FeatureStore(repo_path="path/to/feature/repo/") # Or programmatic config

        online_features_response = fs.get_online_features(
            features=features_for_prediction, # or features=feature_service
            entity_rows=entity_rows
        )
        online_features_dict = online_features_response.to_dict()
        # online_features_dict will contain features and their values
        ```
*   **Alternative:** Use a deployed Feast Feature Server (REST API) for language-agnostic online feature retrieval.

### 7. Scaling Feast

As your feature store grows, certain default components might become bottlenecks.

*   **Scaling Feast Registry:**
    *   **Problem:** The default file-based registry can struggle with concurrent writes (e.g., multiple `feast apply` or materialization jobs) and becomes inefficient as it rewrites the whole file for any change.
    *   **Solution:** Switch to a **SQL-based registry** (e.g., backed by PostgreSQL, MySQL). This allows for concurrent, transactional, and fine-grained updates.
*   **Scaling Materialization:**
    *   **Problem:** The default in-memory materialization process (local engine) doesn't scale for large datasets as it runs on a single process.
    *   **Solutions (Pluggable Materialization Engines):**
        *   **Lambda-based engine (AWS):** Offloads materialization tasks to AWS Lambda.
        *   **Bytewax-based engine:** A Kubernetes-native streaming dataflow framework that can be used for scalable materialization.
        *   **Snowflake Materialization Engine:** If both offline and online are Snowflake.
        *   **Custom Engine:** Build your own engine using Spark, Ray, Flink, etc., to fit your existing infrastructure.

### 8. Structuring Feature Repos for Multiple Environments

Managing feature definitions across development, staging, and production environments.

*   **Goal:** Test changes in a non-production (staging) environment before promoting to production.
*   **Common Approaches:**
    1.  **Different Git Branches:**
        *   Maintain long-lived branches (e.g., `staging`, `main`/`production`).
        *   Changes are made to `staging`, tested, and then manually merged/copied to `main`.
        *   CI/CD applies changes from the respective branches to their corresponding environments.
    2.  **Separate `feature_store.yaml` and Separate Feast Object Definitions:**
        *   Directory structure:
            ```
            ├── staging/
            │   ├── driver_features.py
            │   └── feature_store.yaml  # Configured for staging resources
            ├── production/
            │   ├── driver_features.py  # Potentially a copy from staging
            │   └── feature_store.yaml  # Configured for production resources
            └── .github/workflows/      # CI/CD pipelines for staging & prod
            ```
        *   Changes are developed in `staging/`, tested, then copied to `production/`.
        *   CI/CD uses the `feature_store.yaml` from the respective directory.
        *   Can organize features into Python packages within each environment directory for better structure (e.g., `production/common/entities.py`, `production/ranking_model/views.py`).
    3.  **Shared Feast Object Definitions with Separate `feature_store.yaml` Files:**
        *   Directory structure:
            ```
            ├── staging/
            │   └── feature_store.yaml      # Configured for staging
            ├── production/
            │   └── feature_store.yaml      # Configured for production
            ├── driver_features.py          # Shared feature definitions
            └── .github/workflows/
            ```
        *   CI/CD specifies which `feature_store.yaml` to use with the `-f` (or `--config`) flag:
            ```bash
            feast -f staging/feature_store.yaml apply
            feast -f production/feature_store.yaml apply
            ```
        *   **Advantage:** Avoids code duplication for feature definitions, reducing copy-paste errors.
*   **CI/CD Role:** CI systems (e.g., GitHub Actions) are crucial. They run `feast apply` using the appropriate environment's configuration, updating the shared registry (e.g., a `registry.db` file on GCS/S3 or a SQL database) and configuring infrastructure. The CI system needs write access to production infrastructure, while client SDKs (for training/serving) typically only need read access.

By following these steps and considering the scaling and structuring strategies, teams can effectively deploy and manage Feast in production environments using Snowflake, GCP, AWS, or other cloud platforms.