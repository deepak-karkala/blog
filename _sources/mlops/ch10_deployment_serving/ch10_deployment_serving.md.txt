# Chapter 10: Deployment & Serving

**Chapter 10: Grand Opening – Model Deployment Strategies & Serving Infrastructure**

*(Progress Label: 📍Stage 10: Efficient and Elegant Service to Diners)*

### 🧑‍🍳 Introduction: From Approved Recipe to Diner's Table

The culmination of our MLOps kitchen's efforts—from problem framing and data engineering to model development and rigorous offline validation—is this moment: the "Grand Opening." This chapter is dedicated to the critical processes of **Model Deployment and Serving**, where our approved "signature dishes" (trained and validated ML models) are made accessible and operational, ready to delight our "diners" (end-users and applications) with valuable predictions.

This isn't merely about pushing a model file to a server. As an MLOps Lead, you understand that deploying and serving ML models reliably, scalably, and efficiently is a sophisticated engineering discipline. It involves strategic choices about how models are packaged, which deployment strategies to adopt across the serving spectrum (batch, online, streaming, edge), the architecture of the serving infrastructure, optimizing for inference performance and cost, and implementing robust CI/CD and progressive delivery mechanisms for safe and rapid updates. [guide\_deployment\_serving.md (Core Philosophy)]

We will explore the nuances of packaging models for portability, selecting the right deployment strategy based on business needs and technical constraints, architecting scalable serving patterns (from serverless functions to Kubernetes clusters), and diving deep into inference optimization techniques. We will also detail how CI/CD pipelines facilitate automated, reliable deployments and how progressive delivery strategies ensure that new model versions are rolled out safely. For our "Trending Now" project, this means taking our validated genre classification model (and our LLM-based enrichment logic) and making it a live, functioning service.

---

### Section 10.1: Packaging Models for Deployment (Preparing the Dish for Consistent Plating)

Before a model can be served, it must be packaged with all its necessary components to ensure it runs consistently across different environments.

*   **10.1.1 Model Serialization Formats: Capturing the Essence**
    *   **Purpose:** Saving the trained model (architecture and learned parameters/weights) in a portable format.
    *   **Common Formats:**
        *   **Pickle/Joblib (Python-specific):** Common for Scikit-learn, XGBoost. Simple but can have versioning/security issues.
        *   **ONNX (Open Neural Network Exchange):** Aims for framework interoperability (e.g., PyTorch to TensorFlow Lite). Good for portability but ensure full operator coverage for your model.
        *   **TensorFlow SavedModel:** Standard for TensorFlow models, includes graph definition and weights.
        *   **PyTorch `state_dict` + TorchScript:** `state_dict` for weights, TorchScript for a serializable and optimizable model graph.
        *   **H5 (HDF5):** Often used by Keras.
        *   **PMML (Predictive Model Markup Language):** XML-based standard, less common for deep learning.
    *   **MLOps Consideration:** Choose a format that is supported by your target serving runtime and facilitates versioning. The Model Registry (Chapter 7) should store these serialized artifacts. [guide\_deployment\_serving.md (III.A.1)]
*   **10.1.2 Containerization (Docker) for Serving: Ensuring a Consistent Kitchen Environment**
    *   **Why Docker?** Packages the model, inference code, and all dependencies (libraries, OS-level packages) into a portable image. Ensures consistency between development, staging, and production serving environments.
    *   **Dockerfile for Serving:**
        *   Start from a relevant base image (e.g., Python slim, specific framework image like `tensorflow/serving`, `pytorch/pytorch:serve`).
        *   `COPY` model artifact(s) and inference/API code into the image.
        *   Install dependencies from `requirements.txt`.
        *   Define `ENTRYPOINT` or `CMD` to start the model server/API application (e.g., run `uvicorn main:app` for FastAPI).
    *   **Best Practices for Serving Images:** Keep images small, use official/secure base images, install only necessary dependencies, run as non-root user.
    *   **ML-Specific Docker Wrappers:** Cog, BentoML, Truss can simplify creating serving containers by abstracting Dockerfile creation.

---

### Section 10.2: Choosing a Deployment Strategy: The Serving Spectrum (Dine-in, Takeaway, or Home Delivery?)

ML models can deliver predictions through various mechanisms, catering to different application needs.

*   **10.2.1 Batch Prediction (Asynchronous Inference): Pre-cooking Popular Dishes**
    *   **Concept:** Predictions are computed periodically (e.g., daily/hourly) for a large set of inputs and stored for later retrieval.
    *   **Use Cases:** Lead scoring, daily recommendations, risk profiling, when real-time predictions aren't critical.
    *   **Architecture:** Workflow orchestrator (Airflow) schedules a job (Spark, Python script) that loads data, loads model from registry, generates predictions, and stores them in a DB/DWH/Data Lake.
    *   **Tooling:** Airflow, Spark, Dask; SageMaker Batch Transform, Vertex AI Batch Predictions.
    *   **Pros:** Cost-effective for large volumes, high throughput, allows inspection before use.
    *   **Cons:** Stale predictions, not for dynamic inputs, delayed error detection.
*   **10.2.2 Online/Real-time Prediction (Synchronous Inference): Made-to-Order Dishes**
    *   **Concept:** Predictions are generated on-demand in response to individual requests, typically via a network API.
    *   **Use Cases:** Live fraud detection, interactive recommendations, dynamic pricing, search ranking.
    *   **Architecture:** Model exposed via API (REST/gRPC), often behind a load balancer, running on scalable compute (VMs, containers, serverless).
    *   **Tooling:** FastAPI/Flask, TensorFlow Serving, TorchServe, Triton, KServe, Seldon, Cloud Endpoints (SageMaker, Vertex AI).
    *   **Pros:** Fresh predictions, supports dynamic inputs.
    *   **Cons:** Infrastructure complexity, latency sensitive, requires careful online feature engineering if features are dynamic.
*   **10.2.3 Streaming Prediction: Continuously Seasoning Dishes with Live Feedback**
    *   **Concept:** Online prediction that leverages features computed in real-time from data streams (e.g., user clicks, sensor data). A specialized form of online prediction.
    *   **Use Cases:** Real-time anomaly detection in IoT, adaptive personalization based on in-session behavior.
    *   **Architecture:** Involves stream processing engines (Flink, Kafka Streams, Spark Streaming) for feature computation, which then feed into an online model server.
    *   **Tooling:** Kafka/Kinesis, Flink/Spark Streaming, Online Feature Stores.
    *   **Pros:** Highly adaptive to immediate changes.
    *   **Cons:** Highest complexity (streaming feature pipelines, state management).
*   **10.2.4 Edge Deployment (On-Device Inference): The Chef at Your Table**
    *   **Concept:** Model inference runs directly on the user's device (mobile, browser, IoT sensor, car).
    *   **Use Cases:** Low/no internet scenarios, ultra-low latency needs (robotics, AR), data privacy (on-device processing).
    *   **Architecture:** Optimized/compiled model deployed to edge device. May involve cloud for model updates (OTA) and telemetry.
    *   **Frameworks:** TensorFlow Lite, PyTorch Mobile/Edge (ExecuTorch), CoreML, ONNX Runtime, Apache TVM.
    *   **Pros:** Minimal latency, offline capability, enhanced privacy.
    *   **Cons:** Resource constraints (compute, memory, power), model update complexity, hardware heterogeneity.
*   **(Decision Framework Diagram)** Title: Choosing Your Deployment Strategy
    <img src="../../_static/mlops/ch10_deployment_serving/deployment_strategy_decision_framework.svg" width="100%" style="background-color: #FCF1EF;"/>


---

### Section 10.3: Prediction Serving Patterns and Architectures (The Kitchen's Service Design)

How the model serving logic is structured and integrated into the broader system.

*   **10.3.1 Model-as-Service (Networked Endpoints)**
    *   **API Styles: REST vs. gRPC**
        *   *REST:* HTTP-based, JSON payloads. Pros: ubiquitous, simple. Cons: higher overhead/latency.
        *   *gRPC:* HTTP/2, Protocol Buffers. Pros: high performance, efficient binary serialization, streaming. Cons: more complex client setup.
        *   *MLOps Lead Decision:* REST for broad compatibility/public APIs, gRPC for internal high-performance microservices.
    *   **Model Serving Runtimes:** Specialized servers optimized for ML inference.
        *   *TensorFlow Serving:* For TF SavedModels.
        *   *TorchServe:* For PyTorch models (`.mar` archives).
        *   *NVIDIA Triton Inference Server:* Multi-framework (TF, PyTorch, ONNX, TensorRT, Python backend), dynamic batching, concurrent model execution, ensemble scheduler. Highly performant. [guide\_deployment\_serving.md (V.E)]
        *   *BentoML:* Python-first framework for packaging models and creating high-performance prediction services.
*   **10.3.2 Serverless Functions for Model Inference (The Pop-Up Kitchen Stand)**
    *   **Concept:** Deploy model inference code as a function (e.g., AWS Lambda, Google Cloud Functions). Scales automatically, pay-per-use.
    *   **Pros:** Reduced ops overhead, cost-effective for sporadic traffic.
    *   **Cons:** Cold starts, package size limits, execution time limits, statelessness challenges.
    *   **Best Fit:** Lightweight models, intermittent traffic.
*   **10.3.3 Kubernetes for Scalable and Resilient Model Hosting (The Large, Orchestrated Restaurant Chain)**
    *   **Role:** Manages deployment, scaling (HPA), health, and networking of containerized model servers.
    *   **ML-Specific Platforms on Kubernetes:**
        *   *KServe (formerly KFServing):* Serverless inference on K8s, inference graphs, explainability.
        *   *Seldon Core:* Advanced deployments, inference graphs, A/B testing, MABs, explainers.
    *   **Benefits:** High scalability, resilience, portability, rich ecosystem.
    *   **Challenges:** K8s complexity. Managed K8s (EKS, GKE, AKS) or higher-level platforms are preferred.
*   **10.3.4 Comparison of High-Level Architectures (Monolithic, Microservices, Embedded)** [guide\_deployment\_serving.md (V.D)]
    *   **(Table)** Summary of Pros/Cons for Monolithic, Microservice, and Embedded approaches.

---

### Section 10.4: Inference Optimization for Performance and Cost (Streamlining Service for Speed and Efficiency)

Techniques to make predictions faster and cheaper without (significantly) sacrificing accuracy.

*   **10.4.1 Hardware Acceleration: Choosing the Right "Stove"**
    *   CPUs, GPUs (NVIDIA for inference: T4, A10, A100), TPUs (Google Edge TPUs), Custom AI Accelerators (AWS Inferentia).
    *   Trade-offs: Cost, performance per watt, framework support.
*   **10.4.2 Model Compression Techniques (Making the Recipe More Concise)**
    *   **Quantization:** Reducing numerical precision (FP32 -> FP16/BF16/INT8).
    *   **Pruning:** Removing less important weights/structures.
    *   **Knowledge Distillation:** Training a smaller student model to mimic a larger teacher.
    *   **Low-Rank Factorization & Compact Architectures:** Designing inherently efficient models (e.g., MobileNets).
*   **10.4.3 Compiler Optimizations: The Expert Prep Chef**
    *   Tools: Apache TVM, MLIR, XLA (for TensorFlow), TensorRT (NVIDIA).
    *   Function: Convert framework models to optimized code for specific hardware targets via Intermediate Representations (IRs). Perform graph optimizations like operator fusion.
*   **10.4.4 Server-Side Inference Optimizations (Efficient Kitchen Workflow)**
    *   **Adaptive/Dynamic Batching:** Grouping requests server-side (Triton, TorchServe). [FSDL - Lecture 5]
    *   **Concurrency:** Multiple model instances/threads per server.
    *   **Caching:** Storing results for frequent identical requests.
    *   **GPU Sharing/Multi-Model Endpoints:** Hosting multiple models on a single GPU to improve utilization (SageMaker MME, Triton).
    *   **Model Warmup:** Pre-loading models to avoid cold start latency.

---

### Section 10.5: CI/CD for Model Serving: Automating Model Deployments (Automating the Kitchen's Opening & Closing Procedures)

Automating the build, test, and deployment of the *model serving application* and the *models* it serves.

*   **10.5.1 Building and Testing Serving Components**
    *   **CI for Serving Application:** Unit tests for API logic, pre/post-processing code. Build Docker image for the serving application.
    *   **Model Compatibility Tests (Staging):** Ensure the model artifact loads correctly with the current serving application version and dependencies.
    *   **API Contract & Integration Tests (Staging):** Validate request/response schemas, interactions with Feature Store or other services.
    *   **Performance & Load Tests (Staging):** Verify SLAs are met before production.
*   **10.5.2 Integrating with Model Registry for Model Promotion & Deployment**
    *   CD pipeline triggered by a new "approved-for-production" model version in the registry.
    *   Pipeline fetches the specific model artifact and deploys it to the serving environment (e.g., updates the model file in S3 for a SageMaker Endpoint, or triggers a new K8s deployment with the new model version).
    *   Uber's Dynamic Model Loading: Service instances poll registry for model updates and load/retire models dynamically.

---

### Section 10.6: Progressive Delivery & Rollout Strategies for Safe Updates (Taste-Testing with Diners Before Full Menu Launch)

Minimizing risk when deploying new or updated models to production.

*   **10.6.1 Shadow Deployment (Silent Testing)**
    *   New model receives copy of live traffic, predictions logged but not served.
    *   Compares challenger vs. champion on real data without user impact.
*   **10.6.2 Canary Releases (Phased Rollout)**
    *   Gradually route small percentage of live traffic to new model. Monitor closely. Increase traffic if stable.
*   **10.6.3 Blue/Green Deployments (Full Switchover)**
    *   Two identical production environments. Deploy new model to "Green," test. Switch all traffic. "Blue" becomes standby.
*   **10.6.4 Implementing and Managing Rollbacks**
    *   Automated or one-click rollback to previous stable version if issues detected. Requires robust versioning of models and serving configurations.
    *   Monitoring is key to trigger rollbacks.

---

### Project: "Trending Now" – Deploying the Genre Classification Model & LLM Inference

Applying deployment concepts to our project.

*   **10.P.1 Packaging the Trained XGBoost/BERT Model for Serving (Revisiting)**
    *   **XGBoost:** Serialize using `joblib` or `pickle`. Discuss potential conversion to ONNX for broader compatibility if it were a more complex deployment.
    *   **BERT (Fine-tuned PyTorch model):**
        *   Option 1: Save `state_dict` and model class definition.
        *   Option 2: Convert to **TorchScript (`.pt`)** for a more self-contained package.
        *   Option 3: Export to **ONNX** for wider runtime compatibility (e.g., if targeting ONNX Runtime or TensorRT later).
        *   Decision for the guide: Focus on TorchScript or ONNX for BERT to demonstrate deployable formats.
    *   **Containerization:** The FastAPI application, along with the chosen serialized model and inference script (`predict.py`), will be packaged into a Docker image.
*   **10.P.2 (Conceptual) Applying Compression to the Educational BERT Model**
    *   *(This would be a "what-if" exploration or a future iteration, as initial deployment might not require it for the educational model).*
    *   **Scenario:** If the fine-tuned BERT model's latency on AWS App Runner (using CPU) is too high for an acceptable user experience.
    *   **Potential Steps:**
        1.  **Baseline:** Measure latency and accuracy of the uncompressed fine-tuned BERT model.
        2.  **Quantization (PTQ):**
            *   Use PyTorch's dynamic quantization or static quantization (with a small calibration set from our processed data) to convert the BERT model to INT8.
            *   Package this quantized model and re-evaluate latency and accuracy.
        3.  **Pruning (Conceptual):** Discuss how magnitude pruning could be applied to the BERT model using `torch.nn.utils.prune` followed by fine-tuning, and what the expected impact on size and potential for speedup would be.
        4.  **Knowledge Distillation (Conceptual):** If we had a much larger "teacher" BERT model, discuss how we could distill it into our current, smaller BERT architecture.
    *   **MLOps Consideration:** Any compression step would become part of the automated training/validation pipeline (Chapter 7), producing a compressed model artifact for registration and deployment. The CI/CD pipeline for serving (Chapter 10) would then deploy this compressed model.
*   **Compiler Considerations**
    *   **10.P.X.2 Conceptual Application of Compilers:**
        *   **Scenario 1: BERT Model (PyTorch) on CPU in App Runner.**
            *   *Option A (Simpler):* Rely on PyTorch JIT (TorchScript) optimizations if the model is scripted.
            *   *Option B (More Optimized):* Convert the TorchScript/ONNX BERT model to an **OpenVINO IR** using the Model Optimizer. The FastAPI application in the Docker container would then use the OpenVINO Inference Engine for execution. This would be expected to yield better CPU performance.
            *   *Option C (Alternative for ONNX):* Use **ONNX Runtime** directly in the FastAPI application with its CPU execution provider. ONNX Runtime applies its own graph optimizations.
        *   **Scenario 2: BERT Model on an NVIDIA GPU (if App Runner supported GPUs or we used EC2/ECS).**
            *   Export BERT to ONNX.
            *   Use **NVIDIA TensorRT** to build an optimized `.engine` file.
            *   The FastAPI application (or a Triton server if we scaled up) would load and run this TensorRT engine.
            *   Consider INT8 PTQ with TensorRT calibration for further speedup.
        *   **XGBoost Model:**
            *   Often served directly using its native library or via ONNX Runtime if converted. Specialized compilation is less common for traditional tree-based models compared to neural networks, though runtimes like ONNX Runtime can provide some level of optimization.
    *   **MLOps Pipeline Implication:**
        *   If compilation is used, the compilation step (e.g., `trtexec` for TensorRT, OpenVINO `mo` command) would become a part of the *model building/CI* process *after* training and serialization, producing the final deployable artifact (the `.engine` or IR files). This compiled artifact would then be packaged into the serving Docker image.
*   **Hardware Accelerator Considerations**
    *   **LLM Inference:**
        *   This is handled by the **LLM provider's hardware infrastructure** (likely powerful GPUs/TPUs). Our responsibility is managing API calls efficiently, not the underlying hardware.
    *   **Educational XGBoost/BERT Model Serving (FastAPI on AWS App Runner):**
        *   **App Runner primarily uses CPUs.**
        *   **XGBoost:** Typically runs efficiently on CPUs. No special accelerator needed for its scale in this project.
        *   **BERT (Educational Path):**
            *   *CPU Inference:* For a small BERT model (e.g., DistilBERT or a small `bert-base` fine-tuned) on App Runner (CPU), performance will be modest. This is acceptable for the educational path. Optimization would involve ONNX Runtime with CPU execution provider or OpenVINO if deployed to an Intel CPU environment.
            *   *If Latency Became Critical (Hypothetical GPU on App Runner or EC2/ECS):* If this educational BERT model needed very low latency, we would:
                1.  Choose a GPU instance (e.g., AWS G4dn with NVIDIA T4).
                2.  Export BERT to ONNX.
                3.  Compile to a TensorRT engine (potentially with INT8 quantization).
                4.  Deploy the FastAPI service with the TensorRT engine, likely using Triton Inference Server as the backend if managing multiple models or needing advanced features.
    *   **Data Ingestion & Processing (Airflow on EC2/MWAA):**
        *   These tasks (scraping, Pandas transformations, DVC, Redshift loading) are CPU-bound and do not typically require specialized ML accelerators. Focus is on sufficient CPU and memory for Airflow workers.
    **Conclusion for "Trending Now":** Specialized hardware accelerators are less of a direct concern for our primary LLM path due to API abstraction. For the educational model, CPU inference is the baseline, with a clear conceptual path to GPU acceleration (via ONNX & TensorRT) if performance requirements were to become more stringent.
*   **Runtime Engine Considerations**
    *   **LLM Inference Path (Primary):**
        *   The "runtime engine" is managed by the **LLM API provider** (e.g., Google's infrastructure for Gemini). We don't directly interact with it beyond making API calls. Our FastAPI application is the client to this managed runtime.
    *   **Educational XGBoost/BERT Model Path (FastAPI on App Runner):**
        *   **XGBoost (if served via ONNX Runtime):**
            *   *Compiler (Conceptual):* No explicit "compilation" step like for NNs, but ONNX conversion is a form of graph translation. ONNX Runtime itself performs graph optimizations when loading the model.
            *   *Runtime Engine:* **ONNX Runtime** (CPU Execution Provider by default on App Runner). The FastAPI app would use the `onnxruntime.InferenceSession` API.
        *   **BERT (PyTorch, then to ONNX or TorchScript):**
            *   *If TorchScript (`.pt`):* The **PyTorch JIT Runtime** (part of LibTorch, which would be a dependency in our Docker container) would load and execute the scripted model. FastAPI calls this.
            *   *If ONNX:* Similar to XGBoost, **ONNX Runtime** would be the runtime engine used within the FastAPI application.
            *   *If targeting TensorRT (hypothetical GPU deployment):*
                *   *Compiler:* TensorRT Builder (offline step).
                *   *Runtime Engine:* TensorRT Runtime (used by the inference code, potentially wrapped by Triton if we used Triton as the server).
    *   **Key Takeaway for the Project:** For the educational path, the runtime (ONNX Runtime or PyTorch JIT) will be a library integrated *within* our FastAPI application's Docker container, running on the App Runner CPU instances. We are not setting up a separate, standalone runtime engine process in the same way a dedicated inference server might manage multiple distinct runtime backends.
*   **Inference Server and Architecture Choices**
    *   **10.P.X.Y Educational XGBoost/BERT Path:**
        *   **Serving Logic:** Implemented within our **FastAPI** application.
        *   **Runtime Engine(s):**
            *   If XGBoost is served via ONNX: **ONNX Runtime** (CPU execution provider).
            *   If BERT is TorchScript: **PyTorch JIT Runtime**.
            *   If BERT is ONNX: **ONNX Runtime**.
        *   **Inference Server:** Our FastAPI application itself acts as a lightweight inference server for this path. It handles HTTP requests, loads the model via the chosen runtime, and returns predictions.
        *   **Deployment:** The FastAPI app (with model and runtime packaged in Docker) is deployed to AWS App Runner. App Runner handles scaling and load balancing.
    *   **10.P.X.Z LLM Enrichment Path:**
        *   **Serving Logic:** Implemented within the same **FastAPI** application.
        *   **Runtime Engine:** N/A (The runtime is managed by the LLM API provider, e.g., Google).
        *   **Inference Server:** Our FastAPI application acts as a client to the external LLM API and an orchestrator for the enrichment tasks.
        *   **Deployment:** Same FastAPI app on AWS App Runner.
    *   **Why Not a Full-Blown Inference Server (Triton, TF Serving) for This Project?**
        *   **Simplicity for Educational Focus:** For the scope of this guide's project, setting up and configuring a dedicated inference server like Triton would add significant complexity that might detract from other MLOps learning objectives.
        *   **FastAPI Sufficiency:** FastAPI is capable of handling the moderate load expected for this educational application and directly demonstrates API creation, model loading (via libraries), and containerization.
        *   **LLM Abstraction:** The LLM path is an API call, not direct model hosting.
        *   **MLOps Lead Note:** In a real-world, high-QPS production scenario with multiple complex local models (especially on GPUs), migrating the XGBoost/BERT path to be served by Triton (within Kubernetes or a GPU-enabled SageMaker endpoint) would be a strong consideration for performance, utilization, and advanced features like dynamic batching. The FastAPI app might then call Triton.
    *   **10.P.2 Designing the Inference Service (FastAPI)**
        *   **Path 1 (Educational): Serving XGBoost/BERT via REST API:**
            *   FastAPI endpoint that loads the serialized educational model.
            *   Endpoint accepts plot/review text, performs necessary preprocessing (consistent with training), makes prediction, returns genre.
        *   **Path 2 (Production-Realistic): API for LLM-Powered Enrichment:**
            *   FastAPI endpoints for:
                *   `POST /enrich-item`: Takes movie/show details (plot, reviews). Calls Gemini API for genre, summary, score, tags. Stores/returns results.
                *   `GET /items/{item_id}`: Retrieves enriched data.
                *   `GET /trending`: Implements logic to query and return trending items based on LLM scores/tags (details for ranking logic in later chapters or simplified for now).
            *   Securely manage LLM API key (using strategy from Chapter 3).
    *   **10.P.3 Choosing Serving Infrastructure & Containerization**
        *   Create a `Dockerfile` for the FastAPI application.
            *   Include Python base image, copy FastAPI app code, inference scripts, model artifacts (for educational path).
            *   Install dependencies from `requirements.txt`.
            *   Expose port and define CMD to run `uvicorn`.
        *   Chosen platform: **AWS App Runner** (Serverless Containers). Justification: Ease of deployment from container image, auto-scaling, managed HTTPS.
*   **10.P.4 Implementing CI/CD for the Inference Service (FastAPI on AWS App Runner)**
    *   **CI Workflow (`.github/workflows/ci_fastapi.yml` - triggered on PR to `dev`/`main`):**
        1.  Checkout code.
        2.  Setup Python, install dependencies (from `backend/requirements.txt`).
        3.  Run linters (`flake8`, `black`) and static type checks (`mypy`) on `backend/` code.
        4.  Run unit tests for FastAPI: `pytest backend/tests/unit/`.
        5.  (If educational model is bundled) Run any specific model loading unit tests within the FastAPI context.
        6.  Build Docker image for FastAPI app: `docker build -t trending-now-backend:$GITHUB_SHA -f backend/Dockerfile .`
        7.  Push Docker image to AWS ECR: `aws ecr get-login-password ... | docker login ...` then `docker push <ecr_repo_uri>:$GITHUB_SHA`.
        8.  (Optional) Security scan on the pushed ECR image.
    *   **CD Workflow to Staging (`.github/workflows/cd_staging_fastapi.yml` - triggered on merge to `dev`):**
        1.  Checkout code.
        2.  Setup AWS credentials (for Terraform and App Runner).
        3.  Setup Terraform.
        4.  `terraform apply` for Staging environment (`infrastructure/environments/staging/`) - This deploys/updates the App Runner service pointing to the new ECR image tag (`$GITHUB_SHA`).
        5.  Run API Integration Tests: `pytest backend/tests/integration/ --staging-url=<app_runner_staging_url>`.
        6.  Run Load Tests (Locust): `locust -f backend/tests/load/locustfile.py --host=<app_runner_staging_url> --headless ...`.
        7.  (If ephemeral staging for the FastAPI service was desired and separate from Airflow/Redshift staging, add `terraform destroy` here. Usually, App Runner service would be persistent and just updated).
    *   **CD Workflow to Production (`.github/workflows/cd_prod_fastapi.yml` - triggered on merge to `main` or manual approval after staging):**
        1.  (GitHub Environment with Manual Approval Gate).
        2.  Checkout code.
        3.  Setup AWS credentials.
        4.  Setup Terraform.
        5.  `terraform apply` for Production environment (`infrastructure/environments/production/`) - Updates Production App Runner service to new ECR image tag.
            *   *(Within Terraform/App Runner config, explore gradual deployment options if available, e.g., percentage-based rollout).*
        6.  Run smoke tests against production URL.
        7.  Notify team of deployment completion.
*   **10.P.5 Conceptual Progressive Delivery for the "Trending Now" Model/Service**
    *   **Scenario 1: Updating the Educational XGBoost/BERT Model Version**
        1.  A new, validated XGBoost/BERT model (e.g., `v1.2`) is registered in W&B and approved for production.
        2.  The CI/CD pipeline for the FastAPI service is triggered (or a dedicated model promotion pipeline).
        3.  A new version of the FastAPI Docker image is built, now packaging `model_v1.2`.
        4.  **Staging Deployment:** The new image is deployed to the Staging App Runner service. All Staging tests (API, load) pass.
        5.  **Production Rollout (using AWS App Runner's capabilities):**
            *   App Runner supports percentage-based traffic shifting for new deployments.
            *   Configure the production App Runner service to deploy the new image, initially routing, say, 10% of traffic to instances running the new model version.
            *   **Monitoring:** Closely monitor operational metrics (latency, errors on App Runner) and any available model-specific metrics (e.g., if we log prediction distributions for the 10% traffic) for the new version via CloudWatch/Grafana.
            *   **Incremental Increase:** If stable, gradually increase traffic to the new version (e.g., 50%, then 100%) over a defined period.
            *   **Rollback:** If issues are detected, App Runner allows for quick rollback to the previous stable deployment.
    *   **Scenario 2: Updating LLM Prompt or Logic for Review Summarization**
        1.  A new prompt or summarization logic is developed and tested offline.
        2.  Changes are made to the FastAPI service code.
        3.  CI runs, builds new Docker image.
        4.  **Staging Deployment:** New image deployed to Staging App Runner. Test the `/enrich-item` endpoint with sample movie data to ensure new summaries are generated correctly.
        5.  **Production Rollout (A/B Test or Canary for LLM logic):**
            *   Since App Runner's traffic splitting might be coarse for just an API logic change without distinct model files, we might implement application-level A/B testing if we want to compare two prompts simultaneously:
                *   The FastAPI endpoint, when called, randomly (or based on a user hash) decides to use `prompt_A` or `prompt_B` for the LLM call.
                *   Log which prompt was used along with user feedback (if we had a rating system for summaries) or downstream engagement.
            *   Alternatively, a simpler canary approach: deploy the new FastAPI version (with `prompt_B`) using App Runner's percentage-based rollout. Monitor the quality of summaries generated (e.g., through sampling and manual review initially) and operational metrics.
    *   **Shadow Deployment (Conceptual for LLM Path):**
        *   Modify the FastAPI `/enrich-item` endpoint:
            *   When a request comes, call the *current production* LLM prompt/logic to generate the summary served to the user.
            *   Asynchronously (or in parallel if performance allows), also call the *new candidate* LLM prompt/logic with the same input.
            *   Log both summaries (production and shadow candidate) for offline comparison.
            *   This allows evaluating a new prompt on live data without affecting users.

---

### 🧑‍🍳 Conclusion: The Doors are Open, Service Begins!

The "Grand Opening" is a milestone, signifying that our ML models, born from data and refined through rigorous experimentation, are now live and delivering predictions. This chapter has navigated the complex terrain of model deployment and serving, from packaging models for consistency with Docker to choosing appropriate deployment strategies like batch, online, or edge. We've explored diverse serving architectures, including API-driven Model-as-a-Service, serverless functions, and Kubernetes-orchestrated platforms, understanding their respective trade-offs.

Crucially, we delved into inference optimization – the art of making our models fast and cost-effective through compression, hardware acceleration, and clever server-side techniques. We've also established how CI/CD pipelines automate the deployment of our serving infrastructure and model updates, and how progressive delivery strategies like canary releases and shadow deployments ensure these updates are rolled out safely and reliably.

For our "Trending Now" project, we've containerized our FastAPI application, which serves both our educational genre model and integrates with LLMs for advanced content enrichment, and planned its deployment to a scalable serverless platform. With our models now actively serving predictions, the next critical phase is to continuously "listen to our diners" – through robust monitoring and observability – to ensure our ML kitchen maintains its Michelin standards and adapts to evolving tastes. This will be the focus of Chapter 10.