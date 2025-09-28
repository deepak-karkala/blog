# Validating Historical Features with Great Expectations

##

This tutorial demonstrates how to integrate Great Expectations (GE) with Feast to validate the statistical properties of historical feature datasets. This is crucial for detecting data drift or unexpected changes in feature distributions over time, which can impact model training and performance. The example uses a public dataset of Chicago taxi trips.

**Overall Goal:** To define a "reference profile" of a dataset from one time period and then validate new datasets (from different time periods or sources) against this reference profile.

### 0. Setup

*   **Installation:** Install Feast with Great Expectations support.
    ```bash
    pip install 'feast[ge]'
    # Potentially: pip install google-cloud-bigquery if pulling data from BQ
    ```

### 1. Dataset Preparation (Optional)

This step outlines how to pull and preprocess the raw Chicago taxi trip data from BigQuery into local Parquet files. **If you don't have a GCP account, you can use pre-supplied Parquet files that come with the tutorial's source code.**

*   **Process:**
    1.  Connect to BigQuery.
    2.  Run a SQL query to aggregate raw trip data by `taxi_id` and `day`, calculating features like `total_miles_travelled`, `total_trip_seconds`, `total_earned`, and `trip_count` for a specific period (e.g., 2019-01-01 to 2020-12-31).
    3.  Save the aggregated data to a Parquet file (e.g., `trips_stats.parquet`).
    4.  Run another query to get distinct `taxi_id`s for a specific year (e.g., 2019) to serve as the entity list.
    5.  Save these entity IDs to a Parquet file (e.g., `entities.parquet`).

### 2. Declaring Features in Feast

Define Feast objects (Entities, DataSources, FeatureViews, OnDemandFeatureViews) based on the prepared data.

*   **`FileSource`:** Define a `FileSource` pointing to the `trips_stats.parquet` file, specifying the `timestamp_field` ("day").
*   **`Entity`:** Define the `taxi_entity` with `taxi_id` as the join key.
*   **`BatchFeatureView`:**
    *   Create `trips_stats_fv` using the `taxi_entity` and `batch_source`.
    *   Define its schema with fields like `total_miles_travelled`, `total_trip_seconds`, etc.
*   **`on_demand_feature_view`:**
    *   Create `on_demand_stats` that calculates new features (e.g., `avg_fare`, `avg_speed`) from the `trips_stats_fv` using a Pandas DataFrame transformation.
    *   The input to the transformation function `inp` will be a Pandas DataFrame containing features from the sources (here, `trips_stats_fv`).
*   **Apply Definitions:**
    *   Instantiate `FeatureStore`.
    *   `store.apply([taxi_entity, trips_stats_fv, on_demand_stats])` to register these definitions in the Feast registry (defaulting to `feature_store.yaml` in the current directory).

### 3. Generating a Training (Reference) Dataset

Create a historical feature dataset that will serve as the "golden" or reference dataset for generating expectations.

*   **Load Entities:** Read `entities.parquet` into a Pandas DataFrame (`taxi_ids`).
*   **Create Timestamps:** Generate a DataFrame (`timestamps`) with a range of daily timestamps (e.g., "2019-06-01" to "2019-07-01").
*   **Create Entity DataFrame:** Perform a cross merge (Cartesian product) of `taxi_ids` and `timestamps` to create an `entity_df`. This `entity_df` will have rows for each `taxi_id` at each `event_timestamp`.
*   **Retrieve Historical Features:**
    *   Use `store.get_historical_features()` with this `entity_df` and a list of desired features from both the `BatchFeatureView` and the `OnDemandFeatureView`.
    ```python
    job = store.get_historical_features(
        entity_df=entity_df,
        features=[
            "trip_stats:total_miles_travelled", # ... other batch features
            "on_demand_stats:avg_fare",         # ... other on-demand features
        ]
    )
    ```
*   **Save as a `SavedDataset`:**
    *   Use `store.create_saved_dataset()` to persist the result of the historical retrieval job. This saves the feature data to a specified storage (e.g., a Parquet file) and registers the dataset's metadata in Feast.
    ```python
    store.create_saved_dataset(
        from_=job,
        name='my_training_ds', # Name for the saved dataset
        storage=SavedDatasetFileStorage(path='my_training_ds.parquet')
    )
    ```

### 4. Developing a Dataset Profiler (Great Expectations)

A "profiler" is a function that takes a dataset and generates a set of expectations (a Great Expectations `ExpectationSuite`) based on its characteristics. This suite becomes the reference profile.

*   **Load Reference `SavedDataset`:** Retrieve the saved dataset: `ds = store.get_saved_dataset('my_training_ds')`.
*   **Define Profiler Function:**
    *   Decorate the function with `@ge_profiler`.
    *   The function accepts a `great_expectations.dataset.PandasDataset` object (`ds_ge`).
    *   Use GE's expectation methods (e.g., `ds_ge.expect_column_values_to_be_between(...)`, `ds_ge.expect_column_mean_to_be_between(...)`, `ds_ge.expect_column_quantile_values_to_be_between(...)`) to define checks.
    *   These expectations can be based on domain knowledge (e.g., `avg_speed` between 0 and 60) or derived from the statistics of the reference dataset itself (e.g., mean of `trip_count` should be within +/- 10% of the observed mean in the reference data).
    *   The function must return `ds_ge.get_expectation_suite()`.
    ```python
    from feast.dqm.profilers.ge_profiler import ge_profiler
    from great_expectations.core.expectation_suite import ExpectationSuite
    from great_expectations.dataset import PandasDataset

    DELTA = 0.1 # Allowed window for mean checks

    @ge_profiler
    def stats_profiler(ds_ge: PandasDataset) -> ExpectationSuite:
        ds_ge.expect_column_values_to_be_between("avg_speed", min_value=0, max_value=60, mostly=0.99)
        observed_mean = ds_ge.trip_count.mean() # Access underlying Pandas Series
        ds_ge.expect_column_mean_to_be_between("trip_count",
                                            min_value=observed_mean * (1 - DELTA),
                                            max_value=observed_mean * (1 + DELTA))
        # ... more expectations
        return ds_ge.get_expectation_suite()
    ```
*   **Test Profiler:** Call `ds.get_profile(profiler=stats_profiler)` on the `SavedDataset` object. This applies the profiler to the reference data and prints the generated `ExpectationSuite`. **Verify all defined expectations are present; missing ones indicate they failed on the reference dataset itself (GE default behavior can be silent failure here).**
*   **Create Validation Reference:** Convert the `SavedDataset` and the `profiler` into a `ValidationReference` object. This `ValidationReference` encapsulates the learned profile.
    ```python
    validation_reference = ds.as_reference(name="validation_reference_dataset", profiler=stats_profiler)
    ```
*   **Self-Test:** Validate the original historical retrieval job (`job`) against this `validation_reference`. No exceptions should be raised if the profiler is well-defined for the reference data.
    ```python
    _ = job.to_df(validation_reference=validation_reference)
    ```

### 5. Validating New Historical Retrieval

Now, generate a new historical feature dataset (e.g., for a different time period, like Dec 2020) and validate it against the `validation_reference` created in step 4.

*   **Create New Entity DataFrame:** Generate a new `entity_df` for the new time period (e.g., "2020-12-01" to "2020-12-07").
*   **Get Historical Features for New Period:**
    ```python
    new_job = store.get_historical_features(entity_df=new_entity_df, features=...)
    ```
*   **Execute Retrieval with Validation:** When converting the new job to a DataFrame, pass the `validation_reference`.
    ```python
    from feast.dqm.errors import ValidationFailed
    try:
        df = new_job.to_df(validation_reference=validation_reference)
    except ValidationFailed as exc:
        print("Validation Failed! Report:")
        print(exc.validation_report) # This contains details of failed expectations
    ```
*   **Interpret Results:** If `ValidationFailed` is raised, `exc.validation_report` will show which expectations failed and the observed values versus expected ranges. In the tutorial example, the Dec 2020 data shows:
    *   Lower mean `trip_count`.
    *   Higher mean `earned_per_hour`.
    *   Higher `avg_fare` quantiles.
    These failures are expected due to changes in taxi usage patterns (e.g., COVID-19 impact, fare changes) between June 2019 (reference) and Dec 2020 (tested).

**Key Takeaway:** This process allows MLOps teams to:
1.  Define what "good" data looks like based on a reference period.
2.  Automatically check if new batches of training data conform to these expectations.
3.  Get alerted to significant data drift or quality issues before they silently degrade model performance.
4.  The `SavedDataset` concept in Feast is central to this, allowing feature sets to be persisted and profiled.