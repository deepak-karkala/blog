# Adding or Reusing Tests in Feast
##

This guide explains the structure of Feast's test suite and how to leverage or extend it, whether you're adding new Feast functionality or testing a custom component like a new offline/online store.

### 1. Test Suite Overview

*   **Location:**
    *   Unit Tests: `sdk/python/tests/unit`
    *   Integration Tests: `sdk/python/tests/integration`
*   **Key Directories within `sdk/python/tests/integration`:**
    *   `e2e/`: End-to-end tests covering major user flows and server interactions.
    *   `feature_repos/`: Contains setup files, configurations, and data source/online store creators for constructing test environments. The `universal/` subdirectory is crucial for tests that run against multiple store backends.
    *   `offline_store/`: Tests specific to offline store functionalities like historical retrieval and push APIs.
    *   `online_store/`: Tests specific to online store functionalities like online retrieval and push APIs.
    *   `registration/`: Tests for `feast apply`, registry operations, CLI interactions, and type inference.
*   **`conftest.py` (in parent integration test directory):** Defines common [pytest fixtures](https://docs.pytest.org/en/6.2.x/fixture.html) used across many tests. These fixtures often abstract away specific store implementations, allowing tests to be "universal."
*   **Parametrization:** Tests are heavily parametrized to run against various combinations of offline stores, online stores, and providers without code duplication.

```inline-grid min-w-full grid-cols-[auto_1fr] p-2 [count-reset:line] print:whitespace-pre-wrap
$ tree
.
├── e2e
│   ├── test_go_feature_server.py
│   ├── test_python_feature_server.py
│   ├── test_universal_e2e.py
│   └── test_validation.py
├── feature_repos
│   ├── integration_test_repo_config.py
│   ├── repo_configuration.py
│   └── universal
│       ├── catalog
│       ├── data_source_creator.py
│       ├── data_sources
│       │   ├── __init__.py
│       │   ├── bigquery.py
│       │   ├── file.py
│       │   ├── redshift.py
│       │   └── snowflake.py
│       ├── entities.py
│       ├── feature_views.py
│       ├── online_store
│       │   ├── __init__.py
│       │   ├── datastore.py
│       │   ├── dynamodb.py
│       │   ├── hbase.py
│       │   └── redis.py
│       └── online_store_creator.py
├── materialization
│   └── test_lambda.py
├── offline_store
│   ├── test_feature_logging.py
│   ├── test_offline_write.py
│   ├── test_push_features_to_offline_store.py
│   ├── test_s3_custom_endpoint.py
│   └── test_universal_historical_retrieval.py
├── online_store
│   ├── test_push_features_to_online_store.py
│   └── test_universal_online.py
└── registration
    ├── test_feature_store.py
    ├── test_inference.py
    ├── test_registry.py
    ├── test_universal_cli.py
    ├── test_universal_odfv_feature_inference.py
    └── test_universal_types.py

```


### 2. Structure of the Test Suite

*   **Universal Feature Repo:**
    *   A core concept in Feast testing. It's a set of fixtures (e.g., `environment`, `universal_data_sources` defined in `conftest.py` and `feature_repos/`) that can be parameterized to create test scenarios covering different offline stores (File, Redshift, BigQuery, Snowflake), online stores (SQLite, Redis, Datastore, DynamoDB), and providers (local, gcp, aws).
    *   This allows writing a single test case that runs across many backend combinations.
*   **Integration vs. Unit Tests:**
    *   **Integration Tests:** Require external resources (e.g., actual cloud services like BigQuery, DynamoDB, or even Dockerized services if they mimic external dependencies). They test Feast's behavior with these external systems and typically cover more complex functionalities.
    *   **Unit Tests:** Can run purely locally (Docker resources are sometimes considered part of local for unit testing if they are lightweight and self-contained). They test local Feast behavior, simple functionalities, or mocked interactions.
*   **Main Types of Tests:**
    *   **Integration Tests:**
        *   **E2E tests:** Verify end-to-end flows (`init`, `apply`, `materialize`, `get_historical_features`, `get_online_features`).
            *   `test_universal_e2e.py`: Basic offline store e2e.
            *   `test_go_feature_server.py`, `test_python_feature_server.py`: Server-specific e2e.
            *   `test_validation.py`: Data quality monitoring (DQM) e2e.
        *   **Offline/Online Store Tests:** Focus on retrieval and push APIs.
            *   Historical retrieval: `test_universal_historical_retrieval.py`.
            *   Online retrieval: `test_universal_online.py`.
            *   Push APIs: `test_push_features_to_offline_store.py`, `test_push_features_to_online_store.py`.
        *   **Registration Tests:** Cover `feast apply`, `feast materialize` CLI commands (tested against universal setups), registry logic, and type inference.
    *   **Unit Tests:**
        *   Registry Diff Tests: Logic for determining infrastructure/registry changes.
        *   Local CLI/Feast Tests: CLI commands against local file offline store.
        *   Infrastructure Unit Tests: Mocked tests for specific components (e.g., DynamoDB), repo config, schema inference, key serialization.
        *   Feature Store Validation Tests: Class-level validation (hashing, protobuf/class serialization, error handling) for `DataSource`, `FeatureView`, `FeatureService`, etc.
    *   **Docstring Tests:** Smoke tests to ensure code examples in docstrings run without import or basic setup errors.

### 3. Understanding the Test Suite with an Example

The example test `test_historical_features` demonstrates key patterns:

```python
@pytest.mark.integration  # Marks as an integration test
@pytest.mark.universal_offline_stores # Parametrizes across all universal offline stores
@pytest.mark.parametrize("full_feature_names", [True, False]) # Another parameter
def test_historical_features(environment, universal_data_sources, full_feature_names):
    store = environment.feature_store # 'environment' fixture provides a configured FeatureStore
    (entities, datasets, data_sources) = universal_data_sources # 'universal_data_sources' provides test data

    # ... setup feature views, feature services ...
    # store.apply([...])

    job_from_df = store.get_historical_features(
        entity_df=entity_df_with_request_data,
        features=[...],
        full_feature_names=full_feature_names,
    )
    actual_df = job_from_df.to_df()
    # ... assertions (e.g., validate_dataframes(expected_df, actual_df, ...)) ...
```

*   **Key Fixtures:**
    *   `environment`: Sets up a `FeatureStore` instance, parameterized by provider and online/offline store. It abstracts away the setup complexity of these stores. Each parameterization of `environment` creates an `IntegrationTestRepoConfig`.
    *   `universal_data_sources`: Provides standard entities, datasets (e.g., driver, customer data), and data sources for tests.
*   **Markers:**
    *   `@pytest.mark.integration`: Designates it for `make test-python-integration`.
    *   `@pytest.mark.universal_offline_stores`: Runs the test for each offline store defined in the universal setup (File, Redshift, BigQuery, Snowflake).
    *   `@pytest.mark.parametrize`: Standard pytest mechanism for running a test multiple times with different arguments.

### 4. Writing a New Test or Reusing Existing Tests

*   **Adding a New Test to an Existing File:**
    *   Reuse existing fixture signatures (e.g., `environment`, `universal_data_sources`) to leverage the setup.
    *   Prefer expanding existing tests over creating new ones if the logic is similar, to reduce the overhead of store spin-up/tear-down.
    *   Use markers like `@pytest.mark.universal_offline_stores` and `@pytest.mark.universal_online_stores` for broad coverage. Can restrict to specific stores: `@pytest.mark.universal_online_stores(only=["redis"])`.
*   **Testing a New Offline/Online Store (from a Plugin Repo):**
    *   Install Feast in editable mode: `pip install -e .` (from Feast root).
    *   The core tests are parameterized by `FULL_REPO_CONFIGS` (defined in `feature_repos/repo_configuration.py`).
    *   To test your plugin:
        1.  Create your own Python file (e.g., `my_plugin_configs.py`) defining a new `FULL_REPO_CONFIGS` list that includes `IntegrationTestRepoConfig` instances for your custom store.
        2.  Set the environment variable `FULL_REPO_CONFIGS_MODULE=my_plugin_configs` (module path).
        3.  Run core tests: `make test-python-universal`.
    *   Refer to `feast-custom-offline-store-demo` and `feast-custom-online-store-demo` GitHub repos for examples.
*   **Important Considerations for New Stores:**
    *   **Type Mapping/Inference:**
        *   Update `feast.infra.offline_stores.offline_utils.inference` (or similar for online stores) so Feast can infer schemas from your data source.
        *   Update `feast.infra.utils.type_map` so Feast can convert your store's native data types to/from `feast.types` (e.g., `ValueType.INT64`).
    *   **Historical and Online Retrieval:** These are the most critical functionalities. Ensuring they work correctly implicitly tests many underlying read/write operations.
*   **Including a New Offline/Online Store in the Main Feast Repo:**
    *   **Offline Store:**
        *   Extend `feature_repos/universal/data_source_creator.py`.
        *   Add new `IntegrationTestRepoConfig`(s) to `feature_repos/repo_configuration.py` (usually test against SQLite online store; Redis/DynamoDB if production online store interactions are critical).
    *   **Online Store:**
        *   Add new configuration serialization logic if needed in `repo_configuration.py`.
        *   Add new `IntegrationTestRepoConfig`(s) for the online store in `repo_configuration.py`.
    *   Run full integration suite: `make test-python-integration`.
*   **Including a New Store from External Plugins (Community Maintained):**
    *   Code goes into `feast/infra/offline_stores/contrib/` or `feast/infra/online_stores/contrib/`.
    *   Extend `contrib_data_source_creator.py` (if offline).
    *   Add `IntegrationTestRepoConfig` to `contrib_repo_configuration.py`.
    *   Run contrib tests: `make test-python-contrib-universal`.
*   **Using Custom Data in a New Test:**
    *   See `test_universal_types.py` for an example.
    *   Essentially, create a DataFrame, then use `environment.data_source_creator.create_data_source(df, ...)` to load it into the currently parameterized offline store for the test.
    ```python
    @pytest.mark.integration
    def your_test(environment: Environment): # Environment fixture
        my_df = pd.DataFrame(...)
        # Use the environment's data_source_creator to make it compatible with the current offline store
        data_source = environment.data_source_creator.create_data_source(
            my_df,
            destination_name=environment.feature_store.project + "_my_custom_table"
        )
        # Define FeatureView using this data_source
        my_fv = FeatureView(name="my_custom_fv", source=data_source, ...)
        fs = environment.feature_store
        fs.apply([my_fv, ...])
        # ... rest of your test logic ...
    ```
*   **Running Your Own Redis Cluster for Testing:**
    *   Install Redis locally (e.g., `brew install redis`).
    *   Use scripts in `infra/scripts/redis-cluster.sh` (`start`, `create`, `stop`, `clean`) to manage a local Redis cluster for testing. This allows Redis-specific integration tests to pass locally.

This testing framework is essential for maintaining Feast's quality and ensuring compatibility across its supported (and pluggable) infrastructure components.