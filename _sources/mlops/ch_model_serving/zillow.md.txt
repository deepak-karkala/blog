# Serving Machine Learning Models Efficiently at Scale

**Introduction (Implicit)**

*   **Zillow's Core ML Usage:** ML powers many Zillow features (Zestimate, home recommendations, textual insights, floor plans, semantic search, Premier Agent connections).
*   **Need for Platform:** Critical to invest in platform solutions for economies of scale, easy onboarding, seamless integration with internal systems (data, experimentation, monitoring), and enabling rapid iteration for ML product teams.

**Machine Learning Development Lifecycle**

*   **Iterative Loops:**
    *   **Inner Loop (Experimentation):** Business objective -> ML problem framing -> data collection -> EDA -> feature generation -> model exploration -> evaluation -> iteration. Goal: Fail fast.
    *   **Middle Loop (Refinement):** Iterate on promising models to find the best version.
    *   **Outer Loop (Productionization):** Testing -> evaluation -> CICD for batch/online serving -> production code -> performance optimization -> monitoring -> operational pipeline. This loop is the most engineering-effort intensive.
*   **Friction:** Significant friction exists within and between these loops, prolonging development.

**Challenges with deploying Machine Learning Models and What Needs to Be Solved**

*   **Custom Business Logic:** Serving requires substantial custom logic: request preprocessing, feature extraction (on-the-fly or cached), post-processing, and dynamic scoring orchestration (e.g., conditional model chaining).
*   **Need for Customizability:** This necessitates a *customizable serving container solution* over pre-packed model servers which offer limited flexibility and restricted runtime environments.
*   **Cross-Cutting Concerns:** Consistency (dev/prod), high-quality/performant code (low latency), CICD, monitoring, integration with data/experimentation platforms.
*   **"Triple Friction" Problem:**
    1.  **ML Practitioners:** Don't want to dig into deep engineering; prefer to focus on modeling. Still need involvement to ensure model behavior consistency.
    2.  **Engineers:** Spend time re-writing code for production, learning model behavior, and avoiding pitfalls.
    3.  **Product Teams:** Experience long, risky timelines due to engineering ramp-up or model behavior changes.
*   **Proposed Solution (Two-Part Centralized Platform):**
    1.  **Model Server Creation & Deployments (User Layer):** Self-serve platform for ML practitioners to deploy models with business logic efficiently, without deep web service expertise.
    2.  **Serving ML Model Efficiently at Scale (Backend):** Operationally performant backend system built by engineers, addressing ML serving pitfalls.
*   **Benefits of Centralization:** Economies of scale, standardized onboarding, CICD, seamless integrations, predictable deployment timelines.

**Industry Landscape**

*   **OSS & Vendor Solutions:** Many exist, focusing on different lifecycle aspects (production pipelines vs. experimentation).
*   **Zillow's Approach:** Actively integrates some OSS/vendor solutions, but invests heavily in making them operational, compatible, and accessible internally.
*   **Gaps in Existing Serving Solutions:**
    *   **Ease of Use for ML Practitioners:** Often require significant software engineering expertise, expose low-level concepts (e.g., environment variable serialization).
    *   **Restrictions:** Model/serving logic may need special packaging.
    *   **Custom Code:** Pre-built servers don't fully solve the need for extensive custom code.
    *   **Performance:** Latency (especially P90/P95) can be compromised due to ML serving's unique characteristics.

**Our integration with OSS**

*   **Vision:** Cohesive end-to-end ML platform leveraging OSS where appropriate, integrated with Zillow's ecosystem.
*   **Key OSS Integrations:**
    *   **Kubeflow:** Powerful toolkit for ML on Kubernetes.
    *   **Knative & KServe:** Adopted as the model serving backend.
    *   **Metaflow:** Liked for its Pythonic syntax, steps/flows concept, and ease of decorators. Adopted for batch workflows with an internal orchestration layer ([zillow-metaflow](https://github.com/zillow/metaflow)).
*   **Effort Required:** Extensive engineering to make OSS operational, adapt to Zillow infra, meet SLAs, and create a "paved path" for users.

**ML Model as a Service**

*   **Definition:** ML model(s) deployed as a web service.
*   **Interaction Modes:**
    *   Real-time predictions for client apps (WWW API entrypoint).
    *   Step in a streaming pipeline (e.g., Kafka input/output).
*   **Core Components (Fig 2):** Model serving infrastructure, data/feature/model/metadata stores, A/B testing, monitoring/alerting.
*   **Abstraction Goal:** Hide "peripheral" components so ML practitioners focus on the model itself. Achieved via the two-part platform solution.

**User Layer – Model Server Creation and Deployments**

*   **"Service as Online Flow" Concept:**
    *   Inspired by the "flow" (DAG of steps) paradigm common in batch ML/data engineering (Kubeflow, Metaflow, Airflow, MLflow).
    *   Applies naturally to online serving: request processing -> preprocessing -> prediction -> postprocessing (Fig 3).
    *   Allows extensible, non-linear (e.g., concurrent preprocessing) DAGs.
*   **Pythonic Flow Syntax:**
    *   Chosen because Python is preferred by ML practitioners.
    *   Similar syntax to their existing `zillow-metaflow` batch flows, minimizing learning curve.
    *   Enables users to write online service code in pure Python without deep web service knowledge.
*   **Service Code Example (Fig 4):**
    1.  **`OnlineFlowSpec`:** Implements the core DAG with `@step` decorators and `self.next`.
    2.  **Flow Class Decorators (`@online_service`, `@online_resources`, `@autoscaling`):** Control engineering aspects (endpoint URL, resource limits, autoscaling) declaratively within the flow file.
    3.  **Flow Parameters (`Parameter`):** Pythonic abstraction for runtime configurations (vs. raw environment variables).
    4.  **`load()` function:** Executes warmup tasks (downloading/loading models, custom business logic) before serving.
    5.  **Input/Output Abstraction (`self.input`, `self.output`):** Hides low-level HTTP formatting; provides convenient Pandas DataFrame transformations.
*   **Serving SDK (`aip_serving_sdk`):** Python module providing these abstractions and wrapping a performant web server.
*   **Automatic Service Deployments:**
    *   Offline (batch) flow handles service deployment using an `online_publish` function.
    *   Artifact/metadata from the offline deployment flow is available in the online flow's `load` function (via `base_run` pointer), simplifying model/artifact loading.
    *   Enables Automatic Model Refreshes: models retrained in offline flows can be immediately redeployed.
    *   Leverages GitLab and GitLab CICD.
*   **Impact:** Saved ML practitioners at least 60% of time previously spent on infrastructure work.

**Backend – Serving ML Model Efficiently at Scale**

*   **Goal:** Provide an operationally excellent system, abstracted from ML practitioners.
*   **ML Serving Characteristics (vs. Regular Web Services):**
    1.  **CPU-bound (vs. IO-bound):**
        *   ML serving is mostly CPU (sometimes GPU)-bound, while regular web services are usually IO-bound. This means:
            *   Under the same resource boundary, each ML server replica can’t have as many parallel worker processes as that of regular web services. The benefit of asynchronous I/O can also be less significant for ML serving.
            *   The system level overhead, such as context switching among processes and CPU throttling for guaranteed resource allocation, matters more for ML serving and can affect latency performance more drastically than that of regular web services.
    2.  **Varying Request Workloads:** Light lookups vs. full CPU/GPU inference. Makes CPU-utilization autoscaling ineffective.
    3.  **Heavy Request Effect:** Long-running requests (near timeout) can cause:
        *   False-positive health probe alarms if replica is too busy.
        *   Excessive requests timing out in busy replica's backlog.
    4.  **Serverless Potential:** Scale-to-zero desirable for cost savings with off-peak traffic.
*   **Addressing Characteristics:**
    *   **Request/Event-based Autoscaling:** (Knative) Solves varying workloads and serverless.
    *   **Need for Smart Load Balancing:** To handle CPU-bound nature and heavy requests, a middle layer is needed to buffer surplus requests and dispatch only to replicas with capacity, and proxy probes.
*   **Technical Stack (Fig 5):**
    *   **Knative Serving:**
        1.  **Autoscaler component:** Request-based autoscaling (concurrency/RPS), serverless.
        2.  **Activator component:** Smart load balancing (buffers surplus requests, dispatches to replicas with capacity).
        3.  **Queue-proxy sidecar:** Attaches to each model server container. Proxies probes (can use TCP probe to avoid interrupting app layer), acts as an additional request buffer.
        4.  General traffic management (splitting, gradual rollout).
    *   **KServe:**
        1.  Performant base custom model server (optimized by Zillow).
        2.  Abstraction on top of Knative.
        3.  ML-specific features (request batching, separated transformer for pre/post-processing).
*   **Performance Gains:**
    *   20%-40% improvement in P50 and long-tail latencies compared to alternative vendor solutions.
    *   20%-80% less cost to serve same traffic (combined with other internal optimizations).
*   **User Abstraction:** Zillow's AIP Serving SDK hides Knative/Kubernetes complexity, exposing only the "online flow" user layer.
*   **Integrations:** Serving SDK integrates with Zillow tools (Datadog, Splunk) for monitoring/observability.

**Conclusion and What’s Next?**

*   **Success:** Achieved a self-service serving platform by combining an easy-to-use user layer with a scalable/performant backend, significantly reducing ML lifecycle friction.
*   **Future Plans:**
    *   More abstractions for data layer access, feature extraction.
    *   Enhanced observability in model performance metrics.
    *   Continued OSS engagement and contributions.
    *   Continuously improve user experience and computational efficiency.