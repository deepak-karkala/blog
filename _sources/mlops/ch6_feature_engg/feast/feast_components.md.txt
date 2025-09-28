# Feast Components
##
Feast is a modular system composed of several key components that work together to provide a comprehensive feature store solution.

### 1. Feast Registry: The Central Catalog of Feature Definitions

*   **Definition:** A central catalog that stores all applied Feast object definitions (e.g., `FeatureView`, `Entity`, `DataSource`, `FeatureService`, `Permission`) and their related metadata.
*   **Purpose:**
    *   Enables discovery, search, and collaboration on features.
    *   Serves as the "source of truth" for what features exist and how they are defined.
    *   Accessed by various Feast components (SDK, Feature Server) to understand the feature landscape.
*   **Implementations:**
    *   **File-based (default):** Stores the registry as a serialized Protocol Buffer file on a local filesystem or cloud object store (GCS, S3).
    *   **SQL-based:** Uses a SQL database as the backend for the registry.
*   **Updating the Registry:**
    *   Typically done via `feast apply` command, often integrated into CI/CD pipelines.
    *   Best practice is to store feature definitions in a version-controlled repository (e.g., Git) and sync changes to the registry automatically.
    *   Multiple registries can exist, often corresponding to different environments (dev, staging, prod), with stricter write access for production registries.
*   **Accessing the Registry:**
    *   **Programmatically:** Define `RepoConfig` with `RegistryConfig` in Python code when instantiating `FeatureStore`.
        ```python
        from feast import FeatureStore, RepoConfig
        from feast.repo_config import RegistryConfig

        repo_config = RepoConfig(
            registry=RegistryConfig(path="gs://your-bucket/registry.pb"),
            project="my_project",
            provider="gcp",
            # ... other store configs
        )
        store = FeatureStore(config=repo_config)
        ```
    *   **Via `feature_store.yaml`:** Specify the `registry` path in the `feature_store.yaml` file.
        ```yaml
        project: my_project
        provider: gcp
        registry: s3://your-bucket/registry.pb
        # ... other store configs
        ```
        Then: `store = FeatureStore(repo_path=".")`

### 2. Offline Store: The Home for Historical Feature Data

*   **Definition:** An interface (`OfflineStore`) for Feast to interact with historical, time-series feature data. This data typically resides in your existing data warehouses or data lakes.
*   **Purpose:**
    1.  **Building Training Datasets:** Feast queries the offline store to generate point-in-time correct datasets for model training using `get_historical_features()`.
    2.  **Materializing Features:** Serves as the source from which features are loaded into the Online Store.
*   **How it Works:** Feast doesn't *own* the storage for the offline data but defines how to *query* it. It leverages the compute engine of the underlying system (e.g., BigQuery's query engine, Spark).
*   **Implementations:** `BigQueryOfflineStore`, `SnowflakeOfflineStore`, `RedshiftOfflineStore`, `FileOfflineStore` (for local Parquet/CSV files), etc.
*   **Configuration:** Defined in `feature_store.yaml`. Only one offline store can be active at a time.
*   **Compatibility:** Not all offline stores are compatible with all data source types (e.g., `BigQueryOfflineStore` cannot directly query local files defined as a `FileSource`).
*   **Writing to Offline Store:** Feast can be configured to log served features or directly push features (via Push Sources) to the offline store, enabling a feedback loop or archiving.

### 3. Online Store: Low-Latency Access to Fresh Features

*   **Definition:** A database optimized for low-latency reads, storing only the *latest* feature values for each entity key.
*   **Purpose:** To serve features quickly for real-time model inference.
*   **Data Population:**
    1.  **Materialization:** The `feast materialize` or `feast materialize_incremental` commands load data from the Offline Store into the Online Store.
    2.  **Push Sources:** Features can be written directly to the online store from streaming sources or other real-time processes using the `push` API.
*   **Schema:** The storage schema in the online store mirrors the feature definitions but only holds the most recent value per entity. No historical values are kept.
*   **Implementations:** Redis, DynamoDB, Datastore, SQLite (for local testing), etc.
*   **Configuration:** Defined in `feature_store.yaml`.

### 4. Feature Server: The API Gateway for Online Features

*   **Definition:** A REST API server (built with FastAPI) that provides low-latency access to features from the Online Store and handles feature pushes.
*   **Motivation:**
    *   Simplifies real-time feature retrieval for client applications (e.g., model serving systems).
    *   Provides endpoints for pushing data, ensuring freshness.
    *   Standardizes interaction via HTTP/JSON.
    *   Designed for scalability and secure communication (TLS).
*   **Architecture:** A stateless service that interacts with the Online Store (for data) and the Registry (for metadata/definitions).
*   **Key Endpoints:**
    *   `/get-online-features`: Retrieves latest feature values for specified entities.
    *   `/push`: Pushes feature data to the online and/or offline store.
    *   `/materialize`, `/materialize-incremental`: Triggers materialization jobs.
    *   `/retrieve-online-documents`: [Alpha] Supports vector similarity search for RAG.
    *   `/docs`: OpenAPI documentation for the server.
*   **Deployment:** Can be run locally (`feast serve`), via Docker, or on Kubernetes (Helm charts provided).

### 5. Batch Materialization Engine: Moving Data from Offline to Online

*   **Definition:** The component responsible for executing the process of loading data from the Offline Store into the Online Store (materialization).
*   **Purpose:** Abstracts the underlying technology used for materialization.
*   **Implementations:**
    *   `LocalMaterializationEngine` (default): Runs materialization as a local, serialized process within the Feast client's environment. Suitable for smaller datasets or simpler setups.
    *   `LambdaMaterializationEngine` (AWS): Delegates materialization tasks to AWS Lambda functions for better scalability.
    *   Custom Engines: Users can create their own engines (e.g., using Spark, Flink, or other distributed processing frameworks) if built-in ones aren't sufficient.
*   **Configuration:** Specified in `feature_store.yaml`.

### 6. Provider: Bundling Components for Specific Environments

*   **Definition:** An implementation of a feature store that orchestrates a specific set of components (offline store, online store, materialization engine, etc.) tailored for a particular environment (e.g., GCP, AWS, local).
*   **Purpose:** Simplifies setup by providing sensible default configurations for common cloud stacks while allowing overrides.
*   **Built-in Providers:**
    *   `local`: Uses local file system for offline store, SQLite for online store, and local materialization.
    *   `gcp`: Defaults to BigQuery (offline) and Datastore (online).
    *   `aws`: Defaults might involve S3/Redshift (offline) and DynamoDB (online).
*   **Customization:** Users can select a provider (e.g., `gcp`) but then override specific components (e.g., use Redis as the online store with the `gcp` provider).
*   **Custom Providers:** Users can create fully custom providers to integrate with bespoke infrastructure.
*   **Configuration:** Specified in `feature_store.yaml` via the `provider` key.

### 7. Authorization Manager (`AuthManager`): Enforcing Permissions

*   **Definition:** A pluggable component within Feast servers (Feature Server, Registry Server) responsible for handling authorization.
*   **Functionality:**
    1.  Extracts user identity/credentials (e.g., a token) from incoming client requests.
    2.  Validates these credentials against an authentication server (Feast itself **does not perform authentication**; it relies on external systems).
    3.  Retrieves user details (like roles) from the validated token or auth server.
    4.  Injects these user details into Feast's [Permission](#8-permission-securing-your-feature-store) framework to enforce access control policies.
*   **Supported Implementations (Out-of-the-box):**
    *   **OIDC (OpenID Connect):**
        *   Client retrieves a JWT token from an OIDC server and sends it as a Bearer Token.
        *   Feast server validates the token with the OIDC server and extracts roles (case-sensitive match with `Permission` roles) and `preferred_username` from the token.
        *   Requires OIDC server to be configured to expose roles in the access token and verify signature/expiry.
    *   **Kubernetes RBAC:**
        *   Client uses its Kubernetes service account token as the Bearer Token.
        *   Feast server (running in K8s) queries Kubernetes RBAC resources (Roles, RoleBindings) to determine the client's associated roles.
        *   Requires specific K8s RBAC setup: Feast `Permission` roles must match K8s `Role` names in the Feast service's namespace.
*   **Configuration:** Defined in the `auth` section of `feature_store.yaml`.
    *   `type: no_auth` (default if `auth` section is missing): No authorization is applied.
    *   `type: oidc`: Requires `client_id`, `auth_discovery_url`, and for clients, `username`, `password`, `client_secret`.
    *   `type: kubernetes`: No extra parameters usually needed if running within K8s.
*   **Consistency:** Client and server configurations must be consistent for token injection and validation to work.

### 8. OpenTelemetry Integration: Observability for Your Feature Store

*   **Definition:** Provides comprehensive monitoring and observability for Feast deployments by integrating with the OpenTelemetry standard.
*   **Motivation:** Critical for production ML systems to track performance, gain operational insights, troubleshoot issues, and optimize resources.
*   **Architecture:**
    *   **OpenTelemetry Collector:** Deployed alongside Feast (often in Kubernetes) to receive, process, and export telemetry data (metrics, traces, logs).
    *   **Instrumentation:** Feast (Python) can be auto-instrumented by OpenTelemetry to collect data.
    *   **Exporters:** Components within the Collector that send data to monitoring backends (e.g., Prometheus, Jaeger, etc.).
*   **Key Features:**
    *   Automated Python instrumentation.
    *   Collection of key metrics: CPU/memory usage, request latencies, feature retrieval stats (e.g., `feast_feature_server_memory_usage`, `feast_feature_server_cpu_usage`).
    *   Flexible configuration for data collection and export.
    *   Integration with Prometheus for metrics visualization.
*   **Setup (Typical Kubernetes):**
    1.  Deploy Prometheus Operator.
    2.  Deploy OpenTelemetry Operator (after installing `cert-manager`).
    3.  Configure an OpenTelemetry Collector instance (often via a `OpenTelemetryCollector` CRD in K8s).
    4.  Add OpenTelemetry instrumentation annotations to Feast server deployments (e.g., `instrumentation.opentelemetry.io/inject-python: "true"`).
    5.  Configure environment variables for the Feast server to point to the collector (e.g., `OTEL_EXPORTER_OTLP_ENDPOINT`).
    6.  Enable metrics in Feast Helm chart (`metrics.enabled=true`) and provide collector endpoint.
    7.  Deploy necessary manifests like `Instrumentation`, `ServiceMonitors`, and RBAC rules for OpenTelemetry components.
