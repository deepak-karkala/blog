# Feast Architecture: A Technical Deep Dive for MLOps

##
Feast is an open-source feature store designed to bridge the gap between feature engineering and model serving, ensuring consistency and enabling low-latency access to features for both training and real-time inference. Its architecture is flexible, scalable, and built with MLOps best practices in mind.

<img src="../../../_static/mlops/ch6_feature_engg/feast/arch/1.png" width="80%"/>


Here's a breakdown of its key architectural components and philosophies:

### 1. Core Architectural Overview

Feast's architecture revolves around enabling efficient feature management and serving. Key characteristics include:

*   **Push Model:** Feast primarily uses a push model for data ingestion into the online store. This means data producers actively send feature data to Feast, optimizing for low-latency reads during inference.
*   **Feature Transformation:** Supports transformations for On-Demand and Streaming data sources. Batch transformations are planned for the future. For existing Batch and Streaming sources, Feast often relies on an external Feature Transformation Engine (typically your offline store like Snowflake, BigQuery, or Spark).
*   **Python-centric Serving:** Recommends Python for the feature store microservice, leveraging the rich ML ecosystem and prioritizing precomputation to make Python's overhead tolerable for serving.
*   **Data Ingestion & Write Patterns:** Provides flexibility in how data is written to the online store, catering to different consistency and latency requirements.
*   **Role-Based Access Control (RBAC):** Implements RBAC to secure resources and manage access across teams.

### 2. The Push vs. Pull Model: Optimizing for Speed

Feast champions a **Push Model** for feature ingestion.

*   **How it Works:** Data Producers (services or processes generating feature data) actively "push" feature values to Feast, which then stores them in the online store.
*   **Why Push?**
    *   **Low Latency Retrieval:** This makes Feast read-optimized. At inference time, features are readily available in the online store, minimizing lookup times.
    *   **Avoids Request-Time Bottlenecks:** A Pull Model (where Feast would fetch data from producers on demand) would introduce network latency for each producer call, with overall latency dictated by the slowest call.
*   **Trade-off:**
    *   **Eventual Consistency:** Strong consistency isn't guaranteed out-of-the-box. It requires careful design in orchestrating updates to Feast and how clients consume the data.
*   **Implication:** The "how" and "when" of pushing data become critical, leading to specific write patterns.

### 3. Write Patterns: Managing Data Ingestion

Building on the Push Model, Feast offers various patterns for how Data Producers send data and how feature values are written to the online store.

**A. Communication Patterns (Client to Feast):**

1.  **Synchronous:**
    *   The client (Data Producer) makes an API call (e.g., `push`, `write_to_online_store` methods, or Feature Server's `/push` endpoint) and waits for confirmation.
    *   Suitable for small numbers of entities or single entities where immediate confirmation is needed.
2.  **Asynchronous:**
    *   The client makes an API call (same methods/endpoints) but doesn't wait for completion.
    *   Can also be a "batch job" for large entity volumes using a batch materialization engine.
    *   A common pattern is "micro-batching" API calls to reduce write load on the online store.

**B. Feature Value Write Patterns (How features are computed & stored):**

1.  **Precomputing Transformations:**
    *   Feature transformations are done *before* writing to Feast.
    *   This can happen externally (batch jobs, streaming apps) or within the Feast feature server if raw data is pushed and Feast is configured to transform it upon write (via `push` or `write_to_online_store`).
2.  **Computing Transformations On Demand:**
    *   Transformations occur *inside* Feast, either:
        *   At the time of the client's feature request (read-time).
        *   When the data producer writes raw data to the online store (write-time), controlled by the `transform_on_write` parameter. This allows skipping transformations for already processed data while enabling them for API calls.
3.  **Hybrid (Precomputed + On Demand):**
    *   A mix: some features are precomputed, others (e.g., "Time Since Last Event") are computed on demand at request time.

**C. Tradeoffs (Critical for MLOps decisions):**

| Write Type    | Feature Computation            | Scenario                                                        | Recommendation                                                                          |
| :------------ | :----------------------------- | :-------------------------------------------------------------- | :-------------------------------------------------------------------------------------- |
| Asynchronous  | On Demand                      | Data-intensive, staleness-tolerant                              | Asynchronous writes with on-demand computation for load balancing.                      |
| Asynchronous  | Precomputed                    | High volume, non-critical data                                  | Asynchronous batch jobs with precomputed transformations for efficiency/scalability.    |
| Synchronous   | On Demand                      | High-stakes decisions (e.g., finance)                           | Synchronous writes with on-demand computation for freshness and correctness.            |
| Synchronous   | Precomputed                    | User-facing apps needing quick feedback                         | Synchronous writes with precomputed features for low latency.                           |
| Synchronous   | Hybrid (Precomputed + On Demand) | High-stakes, latency-optimized under constraints                | Synchronous writes, precompute most, on-demand for a few to balance latency/freshness. |

*   **Data Consistency:** Asynchronous writes risk stale data. Synchronous writes ensure freshness but can block producers.
*   **Correctness:** Critical for applications like lending; favors synchronous writes.
*   **Service Coupling:** Synchronous writes create tighter coupling. Failures in Feast writes can cascade to producers.
*   **Application Latency:** Asynchronous writes reduce perceived latency for the producer.

### 4. Feature Transformation: Where Logic Resides

A feature transformation is a function applied to raw or derived data. In Feast, these transformations can be executed by:

1.  **The Feast Feature Server:** For On-Demand transformations or transformations during synchronous writes.
2.  **An Offline Store:** (e.g., Snowflake, BigQuery, DuckDB, Spark) Typically for batch precomputation.
3.  **A Compute Engine:** A more generalized concept for external transformation execution.

**Key Implication:** Different transformation engines might necessitate different transformation code. Aligning the engine choice with the data producer, feature usage, and overall product needs is crucial. This is tightly coupled with the [Write Patterns](#3-write-patterns-managing-data-ingestion).

### 5. Language: Why Python for Feature Serving?

Feast strongly advocates for using **Python** for the feature serving microservice, even if other parts of your stack use different languages (Java/Go clients are available for retrieval).

1.  **Python is the Language of ML:** Meets ML practitioners where they are, with a rich ecosystem (TensorFlow, PyTorch, scikit-learn).
2.  **Precomputation is The Way:** The ideal pattern is to precompute features, reducing serving to a lightweight database lookup. The marginal overhead of Python for this lookup is often tolerable.
3.  **Serving in Another Language Risks Skew:** Re-implementing Python-developed feature logic in Java/Go/C++ for production serving is a primary source of training-serving skew, leading to degraded model performance. Minor exceptions (like "Time Since Last Event") exist but shouldn't be the rule.
4.  **Reimplementation is Excessive:** Rewriting is resource-intensive, error-prone, and has a high opportunity cost. The performance gains rarely justify this if features are largely precomputed.
5.  **Leverage Python Optimizations:**
    *   **Step 1: Quantify Bottlenecks:** Use tools like `cProfile` to identify inefficiencies (e.g., Pandas type conversion overhead).
    *   **Step 2: Optimize Calculations:** Prefer precomputation. For synchronous writes requiring fast computation, use vectorized operations (NumPy), JIT compilers (Numba), and caching (`lru_cache`). Feast itself is continuously optimized.

### 6. Feature Serving and Model Inference Strategies

Feast supports various patterns for how models consume features during inference:

<img src="../../../_static/mlops/ch6_feature_engg/feast/arch/3.png"/>

[Feature Serving and Model Inference](https://docs.feast.dev/getting-started/architecture/model-inference)


1.  **Online Model Inference with Online Features:**
    *   **How:** Client application fetches online features from Feast (`store.get_online_features(...)`) and then passes them to a model server (e.g., KServe) for prediction.
    *   **Use Case:** Applications needing request-time data for inference.

```python
features = store.get_online_features(
    feature_refs=[
        "user_data:click_through_rate",
        "user_data:number_of_clicks",
        "user_data:average_page_duration",
    ],
    entity_rows=[{"user_id": 1}],
)
model_predictions = model_server.predict(features)
```

2.  **Offline Model Inference without Online Features (Precomputed Predictions):**
    *   **How:** Model predictions are precomputed in a batch process and materialized into Feast's online store as just another feature. The client fetches these "predictions" using `store.get_online_features(...)`.
    *   **Use Case:** Simpler to implement, useful for quick impact.
    *   **Drawbacks:** Predictions can be stale; only available for entities present during batch computation.

```python
model_predictions = store.get_online_features(
    feature_refs=[
        "user_data:model_predictions",
    ],
    entity_rows=[{"user_id": 1}],
)
```

3.  **Online Model Inference with Online Features and Cached Predictions:**
    *   **How:** Most sophisticated. Predictions are cached. Inference runs when data producers write new/updated features to the online store. Client reads features and cached predictions. If prediction is missing/stale, it can trigger live inference and write back the new prediction.
    *   **Client Reads:** Fetch features + `model_predictions`. If `model_predictions` is `None`, run inference and write prediction back.
    *   **Client Writes (from Data Producer):** When new `user_data` arrives, run inference and write predictions to the online store.
    *   **Use Case:** Low-latency critical apps, features from multiple sources, computationally expensive models.

```python
# Client Reads
features = store.get_online_features(
    feature_refs=[
        "user_data:click_through_rate",
        "user_data:number_of_clicks",
        "user_data:average_page_duration",
        "user_data:model_predictions",
    ],
    entity_rows=[{"user_id": 1}],
)
if features.to_dict().get('user_data:model_predictions') is None:
    model_predictions = model_server.predict(features)
    store.write_to_online_store(feature_view_name="user_data", df=pd.DataFrame(model_predictions))
```

Note that in this case a seperate call to write_to_online_store is required when the underlying data changes and
predictions change along with it.

```python
# Client Writes from the Data Producer
user_data = request.POST.get('user_data')
model_predictions = model_server.predict(user_data) # assume this includes `user_data` in the Data Frame
store.write_to_online_store(feature_view_name="user_data", df=pd.DataFrame(model_predictions))
```

While this requires additional writes for every data producer, this approach will result in the lowest latency for
model inference.


4.  **Online Model Inference without Features:**
    *   **How:** Model server serves predictions directly without any features from Feast.
    *   **Use Case:** Some LLMs or models not requiring input features.
    *   **Note:** Retrieval Augmented Generation (RAG) *does* use features (document embeddings), fitting pattern #1.

**Client Orchestration:**
The examples often show a Feast-centric pattern (client -> Feast -> Model). An alternative is Inference-centric (client -> Inference Service, which then calls Feast).

### 7. Role-Based Access Control (RBAC)

RBAC in Feast ensures secure and controlled access to feature store resources.

*   **Purpose:** Restrict access based on user roles, maintaining data security and operational integrity.
*   **Functional Requirements:**
    *   Administrators assign permissions (operations, resources) to users/groups.
    *   Seamless integration with existing business code.
    *   Backward compatibility (non-authorized models as default).
*   **Business Goals:**
    *   Enable feature sharing across multiple teams with controlled access.
    *   Prevent unauthorized access to team-specific resources.
*   **Reference Architecture:**
    *   Feast operates as connected services, each enforcing authorization.
    *   **Service Endpoints:** Enforce permissions.
    *   **Client Integration:** Clients attach an authorization token to each request.
    *   **Service-to-Service Communication:** Always granted.
*   **Permission Model:**
    *   **Resource:** A securable object in Feast (e.g., feature view, project).
    *   **Action:** Logical operation on a resource (Create, Read, Update, Delete, etc.).
    *   **Policy:** Rules enforcing authorization (default is role-based).
*   **Authorization Architecture Components:**
    *   **Token Extractor:** Gets token from request header.
    *   **Token Parser:** Retrieves user details from token.
    *   **Policy Enforcer:** Validates request against user details and policies.
    *   **Token Injector:** (For service-to-service if needed) Adds token to outgoing secured requests.

- [Feast Architecture](https://docs.feast.dev/getting-started/architecture)