# Feast Concepts
##
Understanding these fundamental concepts is key to effectively using Feast for managing the lifecycle of your machine learning features.

### 1. Project: Your Isolated Feature Universe

*   **Definition:** The top-level namespace in Feast. A project provides complete isolation for a feature store at the infrastructure level. This is achieved by namespacing resources, such as prefixing table names in the online/offline store with the project name.
*   **Scope:** Each project is a distinct universe of entities, features, and data sources. You cannot retrieve features from multiple projects in a single request.
*   **Best Practice:** It's recommended to have a single Feast project per environment (e.g., `dev`, `staging`, `prod`).
*   **Benefits:**
    *   **Logical Grouping:** Organizes related features, views, and services.
    *   **Isolation:** Prevents interference between different environments or large-scale initiatives.
    *   **Collaboration:** Defines clear boundaries for teams.
    *   **Access Control:** Can be a basis for permissioning (though Feast's `Permission` objects offer more granularity).

### 2. Data Source & Data Ingestion: Connecting to Your Raw Data

Feast doesn't manage the raw underlying data itself; instead, it defines how to connect to and interpret this data.

*   **Data Source:**
    *   **Definition:** Represents the raw data systems where your feature data originates (e.g., a table in BigQuery, files in S3, a Kafka topic).
    *   **Data Model:** Feast uses a time-series data model, expecting feature data to have timestamps indicating when the feature value was observed or generated.
    *   **Types:**
        1.  **Batch Data Sources:** Typically data warehouses (BigQuery, Snowflake, Redshift) or data lakes (S3, GCS). Feast ingests from these for online serving and queries them for historical retrieval.
        2.  **Stream Data Sources:**
            *   **Push Sources:** Allow users to directly push feature values into Feast (both offline and online stores). This is a common pattern for real-time updates.
            *   **[Alpha] Stream Sources:** Allow registration of metadata for Kafka or Kinesis topics. Users are responsible for the ingestion pipeline, though Feast provides some helpers.
        3.  **(Experimental) Request Data Sources:** Data that is only available at the moment of a prediction request (e.g., user input from an HTTP request). Primarily used as input for On-Demand Feature Views.

*   **Data Ingestion:**
    *   **Offline Use (Training/Batch Scoring from Batch Sources):** Feast often doesn't *ingest* data in the traditional sense. It queries your existing batch data sources directly, leveraging the compute engine of the offline store (e.g., BigQuery's query engine).
    *   **Online Use (Real-time Serving):**
        *   **Materialization (from Batch Sources):** The process of loading feature values from batch sources into the online store. The `materialize_incremental` command fetches the *latest* values for entities and ingests them. This is typically scheduled (e.g., via Airflow).
            *   For On-Demand Feature Views with `write_to_online_store=True`, the `transform_on_write` parameter controls if transformations are applied during this materialization (set to `False` to materialize pre-transformed features).
        *   **Pushing (from Stream Sources):** Streaming data can be pushed into the online store (and optionally the offline store) via Push Sources or custom stream processing jobs (e.g., using the contrib Spark processor for Kafka/Kinesis).
    *   **Schema Inference:** If a schema isn't explicitly defined for a batch data source, Feast attempts to infer it during `feast apply` by inspecting the source table or running a `LIMIT` query.

### 3. Entity: The "Subject" of Your Features

*   **Definition:** An entity represents a core business object or concept to which features are related (e.g., `customer`, `driver`, `product`). It's defined with a unique `name` and one or more `join_keys` (the primary keys used to link feature values).
    ```python
    driver = Entity(name='driver', join_keys=['driver_id'])
    ```
*   **Usage:**
    1.  **Defining & Storing Features:**
        *   Feature Views (see below) are associated with zero or more entities. This collection of entities for a feature view is its **entity key**.
        *   Examples:
            *   Zero entities: `num_daily_global_transactions` (a global feature).
            *   One entity: `user_age` (associated with a `user` entity).
            *   Multiple entities (composite key): `num_user_purchases_in_merchant_category` (associated with `user` and `merchant_category` entities).
        *   Reusing entity definitions across feature views is crucial for discoverability and consistency.
    2.  **Retrieving Features:**
        *   **Training Time:** Users provide a list of _entity keys + timestamps_ to fetch point-in-time correct features.
        *   **Serving Time:** Users provide _entity key(s)_ to fetch the latest feature values.
*   **Retrieving All Entities:**
    *   Feast supports generating features for a SQL-backed list of entities for *batch scoring*.
    *   For *real-time retrieval*, fetching all entities is not an out-of-the-box feature to prevent slow and expensive scan operations on data sources.

### 4. Feature View: Organizing and Defining Groups of Features

A Feature View is a central concept for declaring and managing features.

*   **Definition:** A logical collection of features, typically sourced from a single data source and often associated with one or more entities. It defines how Feast should interpret and access these features.
    *   **Online:** A stateful collection read via `get_online_features`.
    *   **Offline:** A stateless collection created via `get_historical_features`.
*   **Key Components:**
    *   `name`: Unique identifier within the project.
    *   `entities`: A list of `Entity` objects this view is associated with (can be empty for global features).
    *   `schema`: A list of `Field` objects defining the features in this view (name and data type). Optional, but highly recommended; if omitted, Feast infers it.
    *   `source`: The `DataSource` (batch, stream, or request) from which these features originate.
    *   `ttl` (Time-To-Live): Optional; limits how far back Feast looks for feature values during historical retrieval and can influence online store retention.
    *   `tags`: Optional metadata (e.g., `{'owner': 'fraud_team'}`).
*   **Important Note:** Feature views require timestamped data. A workaround for non-timestamped data is to insert dummy timestamps.
*   **Usage:**
    *   Generating training datasets.
    *   Defining the schema for loading features into the online store.
    *   Providing schema for retrieving features from the online store.
*   **Feature Inferencing:** If `schema` is not provided, Feast infers features from the data source columns (excluding entity join keys and timestamp columns).
*   **Entity Aliasing:** Allows joining an `entity_dataframe` (used in `get_historical_features`) to a Feature View when the column names in the `entity_dataframe` don't match the Feature View's entity `join_keys`. This is done dynamically using `.with_name("new_fv_name").with_join_key_map({"feature_view_join_key": "entity_df_column_name"})`.
    *   Useful when you don't control source column names or have multiple specialized entities that are subtypes of a general entity (e.g., "origin_location" and "destination_location" both aliasing a "location" entity).

*   **Field (Feature):**
    *   **Definition:** An individual, measurable property or characteristic, typically observed on an entity. Defined with a `name` and `dtype` (e.g., `Float32`, `Int64`).
    *   Fields are defined within a Feature View's `schema`.
    *   Feature names must be unique within a Feature View.
    *   Can have `tags` for additional metadata.

*   **Types of Feature Views:**
    1.  **Standard Feature View:** The most common type, typically backed by a batch data source.
    2.  **[Alpha] On-Demand Feature View (`on_demand_feature_view`):**
        *   Allows defining new features by applying Python transformations to:
            *   Existing features from other Feature Views.
            *   Request-time data (via `RequestSource`).
        *   Transformations are executed as Python code (often Pandas DataFrames) during both historical and online retrieval.
        *   **Scalability:** Fine for online serving (small data). For historical retrieval on large datasets, local Python execution might not scale well.
        *   **Use Case:** Rapid iteration by data scientists, light-weight transformations, combining diverse data sources at request time.
    3.  **[Alpha] Stream Feature View (`stream_feature_view`):**
        *   Extends a normal Feature View by having both a stream source (e.g., Kafka, Kinesis) and a batch source (for backfills/historical data).
        *   Designed for features that need to be updated with very low latency from streaming events.
        *   Can include transformations (e.g., Spark transformations if `mode="spark"`).

### 5. Feature Retrieval: Accessing Your Features

Feast provides APIs to get feature values for different ML lifecycle stages.

*   **Core APIs:**
    1.  `feature_store.get_historical_features(...)`: For training data generation and offline batch scoring. Performs point-in-time correct joins.
    2.  `feature_store.get_online_features(...)`: For real-time model predictions from the online store.
    3.  Feature Server Endpoints (e.g., `POST /get-online-features`): For language-agnostic online feature retrieval.

*   **Key Inputs for Retrieval:**
    *   **Feature Specification:**
        *   **Feature Service (Recommended for production):** A logical group of features (potentially from multiple Feature Views) required by a specific model or model version. You define it once and reference it by name.
            ```python
            driver_stats_fs = FeatureService(
                name="driver_activity_v1",
                features=[driver_stats_fv, driver_ratings_fv[["lifetime_rating"]]]
            )
            features = store.get_online_features(features=driver_stats_fs, ...)
            ```
        *   **Feature References (Good for experimentation):** A list of strings in the format `<feature_view_name>:<feature_name>`.
            ```python
            features = store.get_online_features(features=["driver_hourly_stats:conv_rate"], ...)
            ```
    *   **Entity Specification:**
        *   For `get_historical_features`: An "entity dataframe" (Pandas DataFrame or SQL query) containing entity join key values and **event timestamps** for point-in-time correctness.
        *   For `get_online_features`: A list of `entity_rows` (dictionaries of entity join key values). **No timestamps needed** as it fetches the latest values.

*   **Event Timestamp:** The timestamp recorded in your data source indicating when a feature event occurred. Crucial for point-in-time joins.

*   **Dataset (in Retrieval Context):** The output of `get_historical_features`. It's a table (e.g., Pandas DataFrame) containing the requested features joined onto the input entity dataframe.

### 6. Point-in-Time Joins: Ensuring Temporal Correctness

This is a critical capability of Feast for generating historically accurate training data, preventing data leakage.

*   **How it Works:** When you call `get_historical_features`, Feast uses the `event_timestamp` column in your entity dataframe. For each row in this dataframe, it looks up feature values from the specified Feature Views that were valid *at or before* that row's `event_timestamp`, but not after.
*   **TTL (Time-To-Live) Role:** The `ttl` defined on a Feature View limits how far back in time Feast will search for a feature value from the given `event_timestamp`. If a feature value is older than the `ttl` relative to the `event_timestamp`, it won't be joined.
*   **Example:** If your entity dataframe has an event at `2023-01-15 10:00:00` and a Feature View has a `ttl` of `2 hours`, Feast will look for feature values for that entity between `2023-01-15 08:00:00` and `2023-01-15 10:00:00`.

### 7. [Alpha] Saved Dataset: Persisting Feature Sets

*   **Purpose:** Allows you to save the output of `get_historical_features` (a feature dataset) for later use, such as model training, analysis, or data quality monitoring.
*   **Storage:**
    *   Metadata about the Saved Dataset is stored in the Feast registry.
    *   The actual raw data (features, entities, timestamps) is stored in your configured offline store (e.g., a new table in BigQuery).
*   **Creation:**
    1.  Call `store.get_historical_features(...)` to get a retrieval job.
    2.  Pass this job to `store.create_saved_dataset(from_=historical_job, name="my_dataset", storage=...)`. This triggers the job execution and persists the data.
*   **Planned Creation Methods:** Logging request/response data during online serving or features during writes to the online store.
*   **Retrieval:** `dataset = store.get_saved_dataset('my_dataset_name')`, then `dataset.to_df()`.

### 8. Permission: Securing Your Feature Store

Feast provides a model for configuring granular access policies to its resources.

*   **Scope:** Permissions are defined and stored in the Feast registry.
*   **Enforcement:** Performed by Feast servers (online feature server, offline feature server, registry server) when requests are made through them. *No enforcement when using a local provider directly with the SDK.*
*   **Core Components:**
    *   **`Resource`:** The Feast object being secured (e.g., `FeatureView`, `DataSource`, `Project`). Assumed to have a `name` and optional `tags`.
    *   **`Action`:** The operation being performed (e.g., `CREATE`, `DESCRIBE`, `UPDATE`, `DELETE`, `READ_ONLINE`, `WRITE_OFFLINE`). Aliases like `READ`, `WRITE`, `CRUD` simplify definitions.
    *   **`Policy`:** The rule for authorization (e.g., `RoleBasedPolicy` which checks user roles).
*   **`Permission` Object:** Defines a single permission rule with attributes:
    *   `name`: Name of the permission.
    *   `types`: List of resource types this permission applies to (e.g., `[FeatureView, FeatureService]`). Aliases like `ALL_RESOURCE_TYPES`.
    *   `name_patterns`: List of regex patterns to match resource names.
    *   `required_tags`: Dictionary of tags that must match the resource's tags.
    *   `actions`: List of actions authorized by this permission.
    *   `policy`: The policy object to apply.
*   **Important:** Resources not matching any configured `Permission` are *not secured* and are accessible by any user.
*   **Configuration:** Defined in the `auth` section of `feature_store.yaml`. Feast supports OIDC and Kubernetes RBAC. If `auth` is unspecified, it defaults to `no_auth` (no enforcement).

### 9. Tags: Adding Metadata to Feast Objects

While a specific `tags.md` document wasn't provided, tags are key-value pairs used throughout Feast to add arbitrary metadata to various objects.

*   **Usage:**
    *   **`Field`:** Each feature (field) can have tags.
    *   **`FeatureView`:** Can have tags for organizational purposes.
    *   **`Permission`:** Tags on resources can be used as a condition (`required_tags`) for applying a permission policy.
    *   **`SavedDataset`:** Can have tags.
*   **Purpose:**
    *   **Organization:** Grouping or categorizing resources (e.g., by team, sensitivity level, status).
    *   **Discovery:** Helping users find relevant features or resources.
    *   **Policy Enforcement:** As seen in `Permission`, tags can drive access control decisions.
