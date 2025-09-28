# MLOps End to End Planning

### ML/AI capabilities

<img src="../../_static/mlops/problem_framing/capabilities.jpg"/>

- [The AI Organization by David Carmona](https://learning.oreilly.com/library/view/the-ai-organization/9781492057369/)

### ML Applications

- Recommendation, which identifies the relevant product in a large collection based on the product description or user’s previous interactions.
- Top-K Items Selection, which organizes a set of items in a particular order that is suitable for user (e.g. search result).
- Classification, which assigns the input examples to one of the previously defined classes (e.g “spam”/”not spam”).
- Prediction, which assigns some most probable value to an entity of interest, such as stock value.
- Content Generation, to produce new content by learning from existing examples, such as finishing a Bach chorale cantata by learning from his former compositions.
- Question Answering, which answers an explicit question for example: “Does this text describe this image?”
- Automation, which can be a set of user steps performed automatically, such as stock trading
- Fraud and Anomaly Detection, to identify an action or transaction being a fraud or suspicious
- Information Extraction and Annotation, to identify important information in a text, such as people’s names, job descriptions, companies, and locations.





**MLOps Study Guide: Inside the Michelin-Grade ML Kitchen**

**Final Detailed Chapter & Section Outline**

---

**Chapter 1: Crafting the Vision – The Art of ML Problem Framing**
*(Progress Label: 📍Stage 1: The Restaurant Concept & Vision)*
*   **Section 1.1: Understanding the Business Objective (Why Build This 'Dish'? )**
    *   1.1.1 Identifying the Core Business Problem/Goal
    *   1.1.2 Aligning with Stakeholders (ML Engineers, PMs, Sales, Ops, Business)
    *   1.1.3 Defining Quantifiable Business Success Metrics (KPIs)
    *   1.1.4 Connecting ML Potential to Business Value
*   **Section 1.2: Is Machine Learning the Right Ingredient? (Initial Feasibility)**
    *   1.2.1 When to Consider Using ML (Complexity, Data, Prediction, Scale, Repetition, Change)
    *   1.2.2 When *Not* to Use ML (Simple Solutions, Insufficient Data, High Error Cost, Ethics)
    *   1.2.3 Assessing Organizational Readiness (Data Maturity, Talent, Infrastructure)
*   **Section 1.3: Defining the ML Problem (Translating Vision to Recipe)**
    *   1.3.1 Defining the Ideal Outcome vs. Model's Goal
    *   1.3.2 Identifying the Model's Output Type (Classification, Regression, Generation)
    *   1.3.3 The Importance of Framing (e.g., Classification vs. Regression)
    *   1.3.4 The Label Challenge: Identifying Proxy Labels
*   **Section 1.4: Deeper Feasibility & Risk Assessment (Can We Execute This Vision?)**
    *   1.4.1 Data Availability & Quality Deep Dive (Quantity, Features at Serving, Regulations)
    *   1.4.2 Problem Difficulty Assessment (Prior Art, Human Benchmarks, Adversaries)
    *   1.4.3 Required Prediction Quality & Cost of Errors
    *   1.4.4 Technical Requirements (Latency, Throughput, Resources, Platform, Interpretability, Retraining)
    *   1.4.5 Cost & ROI Estimation (Human, Machine, Inference Costs)
    *   1.4.6 Ethical Considerations & Fairness Review (Initial Check)
*   **Section 1.5: Defining Success Metrics (What Does a 'Michelin Star' Look Like?)**
    *   1.5.1 Business Success Metrics (Targets and Timeframes)
    *   1.5.2 Model Evaluation Metrics (Primary vs. Acceptability Goals)
    *   1.5.3 The Disconnect: Why Model Metrics ≠ Business Success
    *   1.5.4 Planning for Validation (A/B Testing, Gradual Rollout)
    *   1.5.5 Handling Multiple Objectives (Decoupling Strategy)
*   **Section 1.6: Initial Project Planning (The Experimental Mindset)**
    *   1.6.1 Why ML Projects are Uncertain & Non-Linear
    *   1.6.2 Adopting an Experimental Approach (Time-boxing, Scoping Down, Fail Fast)
    *   1.6.3 Starting Simple: The Importance of Baselines
*   **Project: "Trending Now" – Applying ML Problem Framing**
    *   Defining Business Goals for the "Trending Now" App
    *   Is ML the Right Approach for Genre Classification?
    *   Framing the Genre Classification Task (Inputs, Outputs, Model Type)
    *   Initial Feasibility for "Trending Now" (Data, Technical, Ethical)
    *   Success Metrics for the "Trending Now" Genre Model and App

---

**Chapter 2: Designing the Michelin-Grade Kitchen – The MLOps Blueprint & Operational Strategy**
*(Progress Label: 📍Stage 2: The Kitchen Blueprint & Operational Plan)*
*   **Section 2.1: What is MLOps? (Defining Our Culinary Operations)**
    *   2.1.1 The Evolution from DevOps to MLOps
    *   2.1.2 Unique Challenges of ML Systems vs. Traditional Software
    *   2.1.3 Core Goals of MLOps: Speed, Reliability, Scalability, Collaboration
*   **Section 2.2: The MLOps Lifecycle: An End-to-End Workflow (Mapping the Full Production Line)**
    *   2.2.1 Overview of Key MLOps Processes (ML Development, Training Operationalization, Continuous Training, Model Deployment, Prediction Serving, Continuous Monitoring, Data & Model Management - *aligned with Google Whitepaper*)
    *   2.2.2 The Iterative and Interconnected Nature of MLOps
*   **Section 2.3: Core MLOps Design Principles (Our Kitchen's Guiding Philosophies)**
    *   2.3.1 Automation (CI/CD/CT)
    *   2.3.2 Reproducibility (Versioning: Code, Data, Model, Parameters, Environment)
    *   2.3.3 Continuous X (Integration, Delivery, Training, Monitoring)
    *   2.3.4 Comprehensive Testing
    *   2.3.5 Monitoring and Observability
    *   2.3.6 Modularity & Reusability (Loosely Coupled Architecture)
    *   2.3.7 Scalability
    *   2.3.8 Collaboration & Defined Roles
    *   2.3.9 Governance, Security, and Compliance by Design
*   **Section 2.4: The MLOps Stack Canvas: Architecting Your System (The Kitchen Layout Plan)**
    *   2.4.1 Introduction to the MLOps Stack Canvas Framework
    *   2.4.2 Block 1: Value Proposition (Revisiting the "Why" for this system)
    *   2.4.3 Block 2: Data Sources and Data Versioning
        *   _Capability Deep Dive:_ Data Ingestion, Data Storage Solutions, Data Versioning Tools & Strategies.
    *   2.4.4 Block 3: Data Analysis and Experiment Management
        *   _Capability Deep Dive:_ Experimentation Environments (Notebooks, IDEs), Experiment Tracking Tools.
    *   2.4.5 Block 4: Feature Store and Workflows
        *   _Capability Deep Dive:_ Feature Engineering Tools, Feature Store Concepts & Platforms.
    *   2.4.6 Block 5: Foundations (DevOps & Code Management)
        *   _Capability Deep Dive:_ Source Control (Git), Code Quality (Linting, Formatting).
    *   2.4.7 Block 6: Continuous Integration, Training, and Deployment (ML Pipeline Orchestration)
        *   _Capability Deep Dive:_ CI/CD Tools (GitHub Actions, Jenkins), ML Pipeline Orchestrators.
    *   2.4.8 Block 7: Model Registry and Model Versioning
        *   _Capability Deep Dive:_ Model Registry Platforms, Model Versioning Strategies.
    *   2.4.9 Block 8: Model Deployment & Prediction Serving
        *   _Capability Deep Dive:_ Serving Infrastructure Options, API Management.
    *   2.4.10 Block 9: ML Model, Data, and System Monitoring
        *   _Capability Deep Dive:_ Monitoring Tools, Alerting Mechanisms.
    *   2.4.11 Block 10: Metadata Store
        *   _Capability Deep Dive:_ ML Metadata Management, Artifact Repositories.
    *   2.4.12 Overarching Considerations: Build vs. Buy, Platform Choices, Skills
*   **Section 2.5: MLOps Maturity Levels (Phasing the Kitchen Construction)**
    *   2.5.1 Level 0: Manual Processes (The Home Cook Experimenting)
    *   2.5.2 Level 1: ML Pipeline Automation (The Professional Prep Cook)
    *   2.5.3 Level 2: CI/CD Pipeline Automation (The Fully Automated Kitchen Line)
    *   2.5.4 Aligning with AI Readiness (Tactical, Strategic, Transformational)
*   **Section 2.6: Documenting MLOps Architecture (Architectural Decision Records - ADRs)**
*   **Section 2.7: Roles and Responsibilities in MLOps (Staffing the Kitchen)**
    *   2.7.1 Data Scientists, ML Engineers, MLOps Engineers, Data Engineers, Platform Engineers, Domain Experts, Product Managers, IT/Security
    *   2.7.2 Collaboration Models (Embedded vs. Centralized vs. Hybrid)
    *   2.7.3 Structuring ML Teams for Success
*   **Project: "Trending Now" – Blueprinting MLOps Strategy**
    *   Applying MLOps Principles to the "Trending Now" App
    *   Using the MLOps Stack Canvas to Plan "Trending Now" Infrastructure (Identifying necessary capabilities like Data Versioning, Experiment Tracking, Model Registry for the project)
    *   Determining the Initial MLOps Maturity Level for the Project
    *   Defining Roles for the "Trending Now" Project Team
---


**Content for New Chapter 3: Setting Up the "Trending Now" Kitchen – Project Planning & Design**

*   **Section 3.1: Project Overview & Requirements Recap**
    *   Presenting the finalized PRD.
    *   Presenting the App/User Flow diagrams.
*   **Section 3.2: Finalizing the Tech Stack**
    *   Presenting the chosen Tech Stack document (Backend, Frontend, Viz, MLOps Tools, Cloud Services, LLM API).
    *   Justification for key choices.
*   **Section 3.3: Pipeline Design for "Trending Now"**
    *   High-level definition of the 3 core pipelines: Data Ingestion, Model Training (XGBoost/BERT), Inference (LLM).
    *   Inputs, Outputs, Key Steps, and Triggers for each pipeline.
    *   Discussion of necessary scripts (conceptual level).
*   **Section 3.4: Environment Strategy (Dev, Staging, Prod)**
    *   Purpose and configuration goals for each environment for the "Trending Now" project.
    *   Data access strategy and permissions across environments.
*   **Section 3.5: CI/CD Strategy and Branching Model**
    *   Chosen branching strategy (e.g., Gitflow, GitHub Flow).
    *   Outline of CI pipeline steps (Static testing, Unit tests).
    *   Outline of CD pipeline steps (Deployment to Staging/Prod, Integration tests).
*   **Section 3.6: Project Directory Structure**
    *   Proposed layout for Frontend, Backend (FastAPI), MLOps (Pipelines, Scripts, Configs), Tests.
*   **Section 3.7: Detailed Implementation Plan**
    *   Presenting the Step-by-Step Plan mapping project tasks to Chapters 4-12.





**Chapter 3: The Market Run – Data Sourcing, Discovery & Understanding**
*(Progress Label: 📍Stage 3: Sourcing the Finest Ingredients)*
*   **Section 3.1: Identifying Data Requirements Based on ML Problem**
*   **Section 3.2: Exploring Data Sources**
    *   3.2.1 User Input Data
    *   3.2.2 System-Generated Data (Logs, Metrics)
    *   3.2.3 Internal Databases
    *   3.2.4 Third-Party Data (Earned, Paid, Public)
*   **Section 3.3: Initial Data Collection & Ingestion Strategies**
    *   3.3.1 Batch vs. Streaming Ingestion for Discovery
    *   3.3.2 Tools & Technologies for Initial Collection (ETL/ELT, Data Lakes, Warehouses)
*   **Section 3.4: Exploratory Data Analysis (EDA): The First Taste**
    *   3.4.1 Data Profiling (Understanding Schema, Statistics, Distributions)
    *   3.4.2 Visualization Techniques for Data Understanding
    *   3.4.3 Initial Data Quality Checks (Missing Values, Outliers, Inconsistencies)
*   **Section 3.5: Documenting Data: Data Cards and Cataloging Principles**
    *   3.5.1 Importance of Metadata Management
    *   3.5.2 Creating Data Dictionaries & Business Glossaries
    *   3.5.3 Utilizing Data Cards & Datasheets for Datasets
*   **Section 3.6: Early Considerations for Data Governance (Privacy, Security, and Permissions)**
    *   3.6.1 Identifying Sensitive Data (PII)
    *   3.6.2 Initial thoughts on Anonymization & Obfuscation
    *   3.6.3 Defining Access Control & Permissions
*   **Project: "Trending Now" – Data Sourcing and Understanding**
    *   Identifying Data Sources for Movies/TV Shows (APIs like TMDb, review sites – considerations for scraping)
    *   Initial Data Collection Strategy for "Trending Now"
    *   Exploratory Data Analysis on Sample Movie/Review Data
    *   Assessing Data Quality for Plot Summaries and Reviews
    *   Documenting Data Sources and Initial Governance Plan

---

**Chapter 5: Mise en Place – Data Engineering for Reliable ML Pipelines**
*(Progress Label: 📍Stage 4: The Prep Station Standardization)*
*   **Section 5.1: Designing Robust Data Processing Workflows**
*   **Section 5.2: Data Cleaning and Wrangling in Pipelines**
    *   5.2.1 Automated Handling of Missing Values
    *   5.2.2 Systematic Outlier Detection & Treatment
    *   5.2.3 Scripting Data Formatting & Restructuring
*   **Section 5.3: Data Transformation & Standardization for Pipelines**
    *   5.3.1 Implementing Scaling (Normalization, Standardization)
    *   5.3.2 Automated Handling of Skewness
    *   5.3.3 Encoding Categorical Features within Pipelines
*   **Section 5.4: Data Labeling at Scale & Programmatic Labeling**
    *   5.5.1 Integrating Human Labeling Workflows
    *   5.5.2 Leveraging Weak Supervision & Snorkel-like Systems
    *   5.5.3 Building Feedback Loops for Natural Label Generation
    *   5.5.4 Data Augmentation as a Pipeline Step
*   **Section 5.5: Data Splitting and Sampling in Automated Workflows**
    *   5.5.1 Ensuring Stratified and Time-Aware Splits
    *   5.5.2 Implementing Resampling Techniques (Over/Under Sampling)
*   **Section 5.6: Data Validation as a Pipeline Stage**
    *   5.6.1 Automated Schema Validation
    *   5.6.2 Statistical Property & Distribution Drift Checks
    *   5.6.3 Tools for Data Validation (e.g., TFDV, Great Expectations)
*   **Section 5.7: Data Versioning & Lineage in Practice**
    *   5.7.1 Tools & Techniques (DVC, Git LFS, Delta Lake, LakeFS)
    *   5.7.2 Capturing Full Data Lineage
*   **Section 5.8: Building and Orchestrating Data Pipelines**
    *   5.8.1 Choosing Orchestration Tools (Airflow, Prefect, Dagster, Kubeflow, SageMaker Pipelines)
    *   5.8.2 Best Practices for Reusable and Testable Data Pipeline Components
*   **Project: "Trending Now" – Building the Data Ingestion & Preparation Pipeline**
    *   Designing the Data Cleaning and Preprocessing Steps for Movie Plots and Reviews
    *   Strategy for Labeling Genres (e.g., using existing metadata, weak supervision if needed)
    *   Data Splitting for the Genre Classification Model
    *   Implementing Data Validation for the Ingestion Pipeline
    *   Setting up Data Versioning for Ingested and Processed Data
    *   Orchestrating the Daily/Weekly Data Ingestion Pipeline

---

**Chapter 5: Perfecting Flavor Profiles – Feature Engineering and Feature Stores**
*(Progress Label: 📍Stage 5: The Flavor Lab and Central Spice Rack)*
*   **Section 5.1: The Role of Feature Engineering in MLOps**
    *   5.1.1 Learned vs. Engineered Features
    *   5.1.2 Importance of Domain Knowledge in Feature Creation
*   **Section 5.2: Advanced Feature Creation and Transformation Techniques**
    *   5.2.1 Discretization & Binning Strategies
    *   5.2.2 Effective Feature Crossing for Non-Linearity
    *   5.2.3 Generating and Using Embeddings
    *   5.2.4 Handling High Cardinality Features with Hashing
*   **Section 5.3: Feature Selection and Importance in Production Contexts**
    *   5.3.1 Automating Feature Importance Calculation
    *   5.3.2 Strategies for Dimensionality Reduction
    *   5.3.3 Managing Feature Lifecycle (Creation, Deprecation)
*   **Section 5.4: Feature Stores: Architecture and Implementation**
    *   5.4.1 Core Concepts: Registry, Online Store, Offline Store, Serving APIs
    *   5.4.2 Feature Computation: Batch vs. Streaming Feature Pipelines
    *   5.4.3 Ensuring Training-Serving Consistency with Feature Stores
    *   5.4.4 Integrating Feature Stores into the Broader MLOps Ecosystem
    *   5.4.5 Evaluating Feature Store Solutions (Feast, Tecton, Vertex AI Feature Store, SageMaker Feature Store)
*   **Section 5.5: Feature Governance: Quality, Lineage, and Discovery**
*   **Project: "Trending Now" – Feature Engineering for Genre Classification**
    *   Extracting Features from Plot Summaries (e.g., TF-IDF, Embeddings from BERT)
    *   Extracting Features from Reviews (e.g., Sentiment, Keywords)
    *   Feature Selection for XGBoost vs. BERT's implicit feature learning
    *   Designing a Simple Feature Store (Conceptual) for "Trending Now" (if applicable, or discussing when it would be)
    *   Ensuring Feature Consistency for Training and (future) Inference

---

**Chapter 6: The Experimental Kitchen – Model Development & Iteration**
*(Progress Label: 📍Stage 6: The Chef's Test Counter)*
*   **Section 6.1: Setting Up the Productive Experimentation Environment**
    *   6.1.1 Leveraging Notebooks and IDEs Effectively
    *   6.1.2 Integrating with Version Control and Experiment Tracking
    *   6.1.3 Managing Dependencies and Reproducible Environments
*   **Section 6.2: Rapid Prototyping and Establishing Strong Baselines**
    *   6.2.1 Non-ML Baselines
    *   6.2.2 Simple ML Baselines (Logistic Regression, Tree-based models)
*   **Section 6.3: Iterative Model Selection and Architecture Design**
    *   6.3.1 Comparing Different Model Families
    *   6.3.2 Understanding Model Assumptions and Trade-offs
*   **Section 6.4: Deep Dive into Experiment Tracking and Versioning Tools**
    *   6.4.1 What to Log (Parameters, Code, Data, Metrics, Artifacts)
    *   6.4.2 Best Practices for Using MLflow, W&B, SageMaker Experiments etc.
*   **Section 6.5: Advanced Hyperparameter Optimization Strategies**
    *   6.5.1 Beyond Grid/Random Search: Bayesian Optimization, HyperBand
    *   6.5.2 Tools for Distributed HPO
*   **Section 6.6: Exploring AutoML for Efficient Experimentation**
*   **Section 6.7: Debugging Models: A Practical Guide During Development**
    *   6.7.1 Common Bugs and How to Spot Them
    *   6.7.2 Techniques for Diagnosing Training Issues
*   **Project: "Trending Now" – Developing the Genre Classification Model**
    *   Setting up the Experimentation Environment for "Trending Now"
    *   Building Baseline Models (e.g., simple keyword-based, logistic regression on TF-IDF)
    *   Experimenting with XGBoost and Fine-tuning a Pre-trained BERT Model
    *   Tracking Experiments for XGBoost and BERT using a chosen tool
    *   Hyperparameter Tuning for the selected models
    *   Debugging common issues encountered during training

---

**Chapter 7: Standardizing the Signature Dish – Building Scalable Training Pipelines**
*(Progress Label: 📍Stage 7: Codifying the Master Recipe)*
*   **Section 7.1: From Notebooks to Production-Grade Training Code**
    *   7.1.1 Refactoring for Modularity, Testability, and Reusability
    *   7.1.2 Configuration Management for Training
*   **Section 7.2: Designing and Implementing ML Training Pipelines**
    *   7.2.1 Common Pipeline Components (Data Ingestion, Preprocessing, Training, Evaluation, Registration)
    *   7.2.2 Building Custom Pipeline Components
*   **Section 7.3: CI/CD for Training Pipelines: Automating Build, Test, and Deployment**
    *   7.3.1 Unit and Integration Testing for Pipeline Components
    *   7.3.2 Building and Versioning Container Images for Training Steps
    *   7.3.3 Deploying Pipeline Definitions to Orchestration Platforms
*   **Section 7.4: Distributed Training Strategies for Production Scale**
    *   7.4.1 Data Parallelism
    *   7.4.2 Model Parallelism (Sharded, Pipelined, Tensor)
    *   7.4.3 Choosing the Right Strategy and Framework
*   **Section 7.5: Managing Training Infrastructure and Resources Effectively**
    *   7.5.1 Compute Instance Selection (CPU, GPU, Specialized HW)
    *   7.5.2 Spot Instances and Cost Optimization
*   **Section 7.6: Training Orchestration, Scheduling, and Triggering**
*   **Section 7.7: Comprehensive Model and Training Metadata Logging for Auditability**
*   **Project: "Trending Now" – Operationalizing Model Training**
    *   Refactoring Experimental Code into a Production-Ready Training Script
    *   Designing the Automated Training Pipeline (Data Loading, Preprocessing, Training, Evaluation, Registration)
    *   Implementing CI/CD for the Training Pipeline (Building containers, testing pipeline steps)
    *   (Conceptual) Distributed Training for BERT if dataset were very large
    *   Orchestrating the Training Pipeline (e.g., with manual trigger initially)
    *   Logging Training Metadata and Artifacts

---

**Chapter 8: The Head Chef's Approval – Rigorous Offline Model Evaluation & Validation**
*(Progress Label: 📍Stage 8: Stringent Quality Checks Before Service)*
*   **Section 8.1: Comprehensive Evaluation Metrics for Different Tasks**
*   **Section 8.2: Slice-Based Evaluation for Fairness and Identifying Hidden Biases**
*   **Section 8.3: Model Calibration: Ensuring Probabilities are Trustworthy**
*   **Section 8.4: Robustness Testing: Perturbation and Invariance Checks**
*   **Section 8.5: Creating and Utilizing Model Cards for Transparent Documentation**
*   **Section 8.6: The Model Validation Process in Automated Pipelines**
    *   8.6.1 Defining Validation Criteria and Thresholds
    *   8.6.2 Comparing Against Production Models and Baselines
    *   8.6.3 Automated Sign-off vs. Human-in-the-Loop for Promotion
*   **Section 8.7: Model Registry: Versioning and Managing Validated Models**
*   **Project: "Trending Now" – Evaluating the Genre Classification Model**
    *   Defining Key Evaluation Metrics for Genre Classification (Precision, Recall, F1 per genre, Macro/Micro F1)
    *   Slice-Based Evaluation (e.g., by source of data, movie vs. TV show)
    *   Assessing Model Calibration (especially if outputting probabilities)
    *   Creating a Model Card for the Trained XGBoost/BERT Model
    *   The Validation Process for Promoting the "Trending Now" Model to the Registry


**Chapter 8: The Crucible – Comprehensive Testing in ML Systems**
*(Progress Label: 📍Stage 8: Ensuring Every Component is Battle-Ready)*

**Section 8.1: The Imperative of Testing in MLOps (Beyond Model Accuracy)**
    *   8.1.1 Why Testing ML Systems is Different & More Complex than Traditional Software
    *   8.1.2 Goals of a Comprehensive ML Testing Strategy: Reliability, Robustness, Fairness, Compliance, Performance.
    *   8.1.3 The Cost of Insufficient Testing: From Silent Failures to Production Outages.
    *   8.1.4 Overview of Testing Categories in the ML Lifecycle (Data, Features, Models, Pipelines, Infrastructure).

**Section 8.2: Testing the Foundation – Data Validation & Feature Validation**
    *   8.2.1 **Data Validation in Ingestion Pipelines** *(This addresses your first point)*
        *   Purpose: Ensuring raw and processed data meets quality and schema expectations.
        *   Types of Checks:
            *   Schema Validation (data types, column presence, format).
            *   Statistical Property Checks (nulls, cardinality, ranges, distributions).
            *   Freshness and Volume Checks.
        *   Tools & Techniques: Great Expectations, TFDV, Deequ, custom scripts.
        *   Integration: As automated steps within data ingestion/ETL pipelines (e.g., Airflow tasks).
        *   Alerting: Raising alerts on critical validation failures.
    *   8.2.2 **Feature Validation** *(This addresses your second point)*
        *   Purpose: Ensuring features are correct, consistent, and suitable for model consumption.
        *   Location: Within feature engineering pipelines, before writing to a Feature Store, or on data retrieved from a Feature Store.
        *   Types of Checks:
            *   Similar to data validation but on engineered features.
            *   Checks for training-serving skew in feature distributions.
            *   Validation of feature logic (unit tests for feature transformation code).
        *   Tools: Similar to data validation, plus Feature Store specific validation capabilities.

**Section 8.3: Testing the Core – Offline Model Evaluation & Validation** *(This will incorporate the current planned content for Chapter 8)*
    *   8.3.1 **Comprehensive Evaluation Metrics for Different Tasks** (Beyond accuracy: Precision, Recall, F1, AUC-ROC, PR-AUC, business-aligned metrics).
    *   8.3.2 **Establishing Strong Baselines for Comparison** (Random, Heuristic, Zero-Rule, Human, Existing System).
    *   8.3.3 **Slice-Based Evaluation:** Identifying hidden biases and performance disparities across data segments. Techniques for defining and evaluating slices.
    *   8.3.4 **Model Calibration:** Ensuring probability outputs are reliable and trustworthy. Calibration curves and techniques.
    *   8.3.5 **Model Robustness Testing:**
        *   Perturbation Tests (sensitivity to noise).
        *   Invariance Tests (fairness, insensitivity to protected attributes).
        *   Directional Expectation Tests.
    *   8.3.6 **Model Interpretability and Explainability Checks** (SHAP, LIME – ensuring explanations make sense).
    *   8.3.7 **Creating and Utilizing Model Cards** for transparent documentation of evaluation results.
    *   8.3.8 **The Automated Model Validation Process in Training Pipelines:**
        *   Defining validation criteria, thresholds, and promotion gates.
        *   Comparing new model candidates against production models/baselines.
        *   Automated sign-off vs. Human-in-the-Loop for model promotion to registry.
    *   8.3.9 **Model Registry for Versioning and Managing Validated Models.**

**Section 8.4: Testing the Arteries – ML Pipeline Integration & End-to-End Testing** *(This addresses your fourth point)*
    *   8.4.1 **Why Pipeline Testing is Critical:** Ensuring components work together, data flows correctly, artifacts are produced as expected.
    *   8.4.2 **Unit Testing for Pipeline Components/Tasks:** Testing individual scripts and operators.
    *   8.4.3 **Integration Testing for Pipelines:**
        *   Data Ingestion Pipelines: Testing data flow from source to validated, processed storage (e.g., S3/Redshift).
        *   Feature Engineering Pipelines: Testing transformation logic and Feature Store interactions.
        *   Model Training Pipelines: Testing end-to-end run from data loading to model registration.
        *   Inference Pipelines (Offline/Batch part): Testing data loading, feature retrieval, prediction generation, and output storage.
    *   8.4.4 **Techniques:** Using sample data, mocked services (e.g., LLM APIs for staging tests), testing on staging environments.
    *   8.4.5 **Tools:** Pytest for custom test scripts, Airflow DAG testing utilities, CI/CD integration.

**Section 8.5: Testing the Frontline – Model Serving Infrastructure & Performance Testing** *(This addresses your fifth point)*
    *   8.5.1 **Deployment Validation:** Ensuring the model (and its container/package) can be successfully deployed and served by the chosen infrastructure (e.g., FastAPI on App Runner).
    *   8.5.2 **API Contract Testing:** Validating request/response schemas for the serving endpoint.
    *   8.5.3 **Consistency Checks:** Verifying that the deployed model in a staging/test serving environment gives the same prediction for a given input vector as it did during offline training/evaluation (catches serialization or environment bugs).
    *   8.5.4 **Load Testing (e.g., using Locust):**
        *   Measuring QPS (Queries Per Second), latency (p50, p90, p99).
        *   Understanding resource utilization (CPU, RAM, GPU) under load.
        *   Determining scaling behavior and bottlenecks.
    *   8.5.5 **Stress Testing:** Pushing the system beyond expected load to find breaking points.
    *   8.5.6 **End-to-End (E2E) Tests for Serving:** Simulating user requests through the entire serving stack.
    *   8.5.7 **Acceptance Testing (User/Business):** Validating that the deployed system meets user needs and business requirements in a staging/UAT environment.
    *   8.5.8 **Measuring the Delta Between Models (Operational):** Comparing latency, throughput, and resource usage of new vs. old models in a staging serving environment.

**Section 8.6: A Framework for Testing in MLOps: Where, When, and What** *(This directly addresses your request for a summary view)*
    *   **8.6.1 Testing in the Development Environment:**
        *   Focus: Code correctness, local validation of logic.
        *   Activities: Unit tests, local data validation on samples, IDE debugging.
    *   **8.6.2 Testing in CI (Continuous Integration) Pipelines:**
        *   Focus: Early detection of bugs, code quality, basic component integrity.
        *   Activities (on feature branch push/PR): Linting, static analysis, unit tests (code, data logic, feature logic), IaC validation.
    *   **8.6.3 Testing in CD (Continuous Delivery) to Staging Environment:**
        *   Focus: Integration, end-to-end functionality, operational readiness.
        *   Activities:
            *   *For Pipeline Deployments:* Full data/training/inference pipeline runs on sample/staging data, integration tests, component interaction checks.
            *   *For Model Serving Deployments:* Deployment validation, API contract tests, consistency checks, load/performance tests, E2E tests, acceptance tests.
    *   **8.6.4 Testing in CD to Production Environment (Post-Approval):**
        *   Focus: Smoke tests, ensuring safe rollout.
        *   Activities: Basic health checks post-deployment. *Full online testing (A/B, Canary) is detailed in Chapter 12.*
    *   **(Table)** Title: MLOps Testing Matrix: Environment vs. Pipeline vs. Test Type
        | Stage/Environment     | Key Pipeline(s) Involved     | Data Validation | Feature Validation | Model Offline Eval | Pipeline E2E/Integration | Serving Perf/Load Tests |
        | :-------------------- | :--------------------------- | :-------------- | :----------------- | :----------------- | :----------------------- | :---------------------- |
        | Dev (Local/IDE)       | Manual Scripts               | Partial/Manual  | Partial/Manual     | Initial/Manual     | Manual                   | N/A                     |
        | CI (GitHub Action)    | Code Build/Test              | Unit Tests (Logic) | Unit Tests (Logic) | Unit Tests (Model Code) | N/A                      | N/A                     |
        | Staging (CD)          | Data Ingestion, Training, Serving (Candidate) | Full (on Staging Data) | Full (on Staging Data) | Full (Candidate Model) | Yes                      | Yes (Serving Candidate) |
        | Production (Training) | Training Pipeline            | Full (on Prod Data) | Full (on Prod Data) | Full (New Model)   | (Monitored)              | N/A                     |
        | Production (Serving)  | Serving (Live Model)         | (Monitoring)    | (Monitoring)       | (Online Monitoring)  | (Monitoring)             | (Monitoring)            |

**Section 8.7: Tools and Frameworks for ML Testing**
    *   Recap of tools mentioned: Great Expectations, TFDV, Deequ, Pytest, Terratest (conceptual for IaC), Locust.
    *   Specific ML testing libraries or features within larger frameworks (e.g., `asserts` in TFX, evaluation capabilities in MLflow/W&B).

**Section 8.8: The ML Test Score Rubric: A Framework for Production Readiness**
    *   Introduction to the Google paper "The ML Test Score: A Rubric for ML Production Readiness and Technical Debt Reduction." [ML\_Test\_Score.pdf]
    *   Overview of its categories (Tests for Features & Data, Model Development, ML Infrastructure, Monitoring).
    *   How to use it as an aspirational guide or self-assessment tool.

**Section 8.9: Online Testing / Testing in Production (Preview)**
    *   Briefly define A/B testing, Canary releases, Shadow deployments.
    *   Explain that these are primarily for validating *live effectiveness and business impact*.
    *   Clearly state that this topic will be covered in-depth in **Chapter 12: Refining the Menu – Continual Learning & Production Testing for Model Evolution**.

**Project: "Trending Now" – Implementing a Comprehensive Testing Strategy**
*   **8.P.1 Data Validation for Ingestion Pipeline**
    *   Implement Great Expectations checks (or Pandas-based checks) for the output of `preprocess_data_task` and `get_llm_predictions_task`.
    *   Integrate this validation into the Airflow DAG.
*   **8.P.2 Feature Validation (Conceptual)**
    *   Discuss what feature validation checks would be relevant if we had a more complex feature engineering pipeline or a feature store (e.g., checking distribution of TF-IDF values, or embeddings).
*   **8.P.3 Rigorous Offline Model Evaluation (for Educational XGBoost/BERT Model)**
    *   Implement slice-based evaluation (e.g., accuracy for "Movie" vs. "TV Show" content types, or by source OTT if available).
    *   (Conceptual) Discuss how robustness testing (e.g., adding noise to plot summaries) would be done.
    *   Generate a Model Card.
*   **8.P.4 Pipeline Integration Testing (Staging)**
    *   Outline `pytest` integration tests for the Data Ingestion Airflow DAG to be run in the ephemeral staging environment.
    *   Focus on: triggering the DAG, ensuring it runs to completion with sample data, verifying data lands in staging S3 and (conceptually) staging Redshift.
*   **8.P.5 Model Performance & Serving Infrastructure Testing (Staging)**
    *   Outline `pytest` tests for the staging FastAPI endpoint (both LLM and educational model paths).
    *   Set up a simple Locust load test file to hit the staging FastAPI endpoint.
*   **8.P.6 Integrating Tests into GitHub Actions CI/CD**
    *   Add steps to `ci.yml` to run data/feature validation logic unit tests.
    *   Add steps to `cd_staging.yml` to execute pipeline integration tests and serving load tests against the ephemeral staging environment.

---

**Chapter 9: Grand Opening – Model Deployment Strategies & Serving Infrastructure**
*(Progress Label: 📍Stage 9: Efficient and Elegant Service to Diners)*
*   **Section 9.1: Packaging Models for Deployment**
    *   9.1.1 Model Serialization Formats (ONNX, PMML, Pickle, SavedModel, TorchScript)
    *   9.1.2 Containerization (Docker) for Serving
*   **Section 9.2: Choosing a Deployment Strategy: The Serving Spectrum**
    *   9.2.1 Batch Prediction: Use Cases, Architecture, Tooling
    *   9.2.2 Online/Real-time Prediction: Use Cases, Architecture, Tooling
    *   9.2.3 Streaming Prediction: Leveraging Real-time Features
    *   9.2.4 Edge Deployment: Requirements and Frameworks
*   **Section 9.3: Prediction Serving Patterns and Architectures**
    *   9.3.1 Model-as-Service (REST APIs, gRPC Endpoints)
    *   9.3.2 Serverless Functions for Model Inference
    *   9.3.3 Kubernetes for Scalable and Resilient Model Hosting
*   **Section 9.4: Inference Optimization for Performance and Cost**
    *   9.4.1 Hardware Acceleration (GPU, TPU, Custom Silicon)
    *   9.4.2 Model Compression Techniques (Quantization, Pruning, Distillation)
    *   9.4.3 Compiler Optimizations (TensorRT, OpenVINO, TVM)
    *   9.4.4 Batching and Caching in Serving
*   **Section 9.5: CI/CD for Model Serving: Automating Model Deployments**
    *   9.5.1 Building and Testing Serving Components
    *   9.5.2 Integrating with Model Registry for Model Promotion
*   **Section 9.6: Progressive Delivery & Rollout Strategies for Safe Updates**
    *   9.6.1 Shadow Deployment for Non-disruptive Testing
    *   9.6.2 Canary Releases for Gradual Rollout
    *   9.6.3 Blue/Green Deployments for Instant Switchover/Rollback
    *   9.6.4 Implementing and Managing Rollbacks
*   **Project: "Trending Now" – Deploying the Genre Classification Model & LLM Inference**
    *   Packaging the Trained XGBoost/BERT Model for Serving
    *   Designing the Inference Pipeline:
        *   Path 1 (Educational): Serving the XGBoost/BERT model via a REST API.
        *   Path 2 (Production-Realistic): Designing an API to an LLM for genre classification (prompt engineering, API key management).
    *   Choosing Serving Infrastructure (e.g., Serverless for both, or Kubernetes for BERT if self-hosted)
    *   Implementing CI/CD for the Inference Service (for the XGBoost/BERT path)
    *   Conceptual Progressive Delivery for the "Trending Now" model

---

**Chapter 10: Listening to the Diners – Production Monitoring & Observability for ML Systems**
*(Progress Label: 📍Stage 10: Continuous Feedback & Kitchen Awareness)*
*   **Section 10.1: Key Metrics for Production Model Monitoring**
    *   10.1.1 Operational Metrics (Latency, Throughput, Error Rates, Resource Usage)
    *   10.1.2 Model Performance/Accuracy-Related Metrics (Business KPIs, Ground Truth Proxy)
    *   10.1.3 Monitoring Prediction Outputs (Distribution Shifts, Confidence Scores)
    *   10.1.4 Monitoring Input Features (Data Validation, Distribution Shifts)
*   **Section 10.2: Detecting Data and Concept Drift in Production**
    *   10.2.1 Statistical Methods and Challenges
    *   10.2.2 Defining Appropriate Time Scale Windows
    *   10.2.3 Handling High-Dimensional Data in Drift Detection
*   **Section 10.3: Implementing an Effective Monitoring Toolbox**
    *   10.3.1 Structured Logging for ML Systems
    *   10.3.2 Dashboards and Visualization for Insights
    *   10.3.3 Alerting Strategies and Avoiding Alert Fatigue
*   **Section 10.4: Tools and Platforms for ML Monitoring**
    *   10.4.1 Open Source vs. Commercial Solutions
    *   10.4.2 Integrating Monitoring with the MLOps Stack
*   **Section 10.5: Observability: Going Beyond Monitoring for Deeper Insights**
    *   10.5.1 Understanding System Internals from External Outputs
    *   10.5.2 Enabling Root Cause Analysis for Model Failures
*   **Project: "Trending Now" – Monitoring the Genre Classification Service**
    *   Defining Monitoring Metrics for the XGBoost/BERT Service (Operational: latency, error rate; Model: input data drift for plots/reviews, prediction distribution drift for genres).
    *   Monitoring the LLM Service (API error rates, latency, cost, output quality/drift).
    *   Setting up Dashboards and Alerts for "Trending Now."
    *   Implementing Logging for Root Cause Analysis.

---

**Chapter 11: Refining the Menu – Continual Learning & Production Testing for Model Evolution**
*(Progress Label: 📍Stage 11: Evolving Dishes Based on Popularity & New Ingredients)*
*   **Section 11.1: The Imperative of Continual Learning in MLOps**
    *   11.1.1 Why Models Degrade (Data/Concept Drift, Evolving User Behavior)
    *   11.1.2 Benefits of Keeping Models Fresh
*   **Section 11.2: Strategies for Model Retraining and Updating**
    *   11.2.1 Triggers for Retraining (Scheduled, Performance-based, Drift-based, Volume-based)
    *   11.2.2 Data Curation and Selection for Retraining
    *   11.2.3 Stateful (Incremental) vs. Stateless (Full) Retraining
    *   11.2.4 Online Learning vs. Online Adaptation of Policies
*   **Section 11.3: Testing in Production: Validating Model Updates Safely**
    *   11.3.1 Limitations of Offline Evaluation for Evolving Systems
    *   11.3.2 A/B Testing for Comparing Model Versions
    *   11.3.3 Advanced Online Experimentation: Interleaving and Bandit Algorithms
*   **Section 11.4: Building Robust Feedback Loops for Continuous Improvement**
*   **Section 11.5: Automating the Continual Learning Cycle: From Monitoring to Redeployment**
*   **Project: "Trending Now" – Implementing Continual Learning**
    *   Defining Retraining Triggers for the Genre Model (e.g., new data from ingestion pipeline, performance degradation).
    *   Data Curation for Retraining (using newly scraped data).
    *   Deciding on Stateful vs. Stateless Retraining for the XGBoost/BERT model.
    *   Conceptual A/B Testing a new version of the genre model (or a new LLM prompt) in production.
    *   Automating the Retraining and Redeployment Cycle.
*   **Project: "Trending Now" – Implementing Continual Learning**
    *   Defining Retraining Triggers for the Genre Model (e.g., new data from ingestion pipeline, performance degradation).
    *   Data Curation for Retraining (using newly scraped data).
    *   Deciding on Stateful vs. Stateless Retraining for the XGBoost/BERT model.
    *   Conceptual A/B Testing a new version of the genre model (or a new LLM prompt) in production.
    *   Automating the Retraining and Redeployment Cycle.

---

**Chapter 12: Running a World-Class Establishment – Governance, Ethics & The Human Element in MLOps**
*(Progress Label: 📍Finale: The Restaurant's Enduring Legacy & Responsibility)*
*   **Section 12.1: Comprehensive Model Governance in MLOps**
    *   12.1.1 Framework for Model Governance (Reproducibility, Validation, Observation, Control, Security, Auditability)
    *   12.1.2 Regulatory Compliance and Documentation (e.g., EU AI Act)
    *   12.1.3 Model Service Catalogs and Internal Discoverability
*   **Section 12.2: Principles and Practices of Responsible AI in MLOps**
    *   12.2.1 Fairness: Identifying and Mitigating Bias Across the Lifecycle
    *   12.2.2 Explainability and Interpretability: Tools and Techniques (SHAP, LIME, Model Cards)
    *   12.2.3 Transparency: Making ML Systems Understandable
    *   12.2.4 Privacy-Preserving ML Techniques
    *   12.2.5 Security: Protecting Against Adversarial Attacks and Data Poisoning
*   **Section 12.3: Holistic Testing for ML Systems: The ML Test Score**
    *   12.3.1 Testing Features and Data
    *   12.3.2 Tests for Reliable Model Development
    *   12.3.3 ML Infrastructure Tests
    *   12.3.4 Integrating into CI/CD
*   **Section 12.4: Structuring and Managing High-Performing ML Teams**
    *   12.4.1 Essential MLOps Roles and Skills
    *   12.4.2 Effective Organizational Structures for MLOps
    *   12.4.3 Project Management for Iterative and Uncertain ML Projects
*   **Section 12.5: Designing User-Centric and Trustworthy ML Products**
    *   12.5.1 Bridging User Expectations and Model Reality
    *   12.5.2 Designing for Smooth Failure and User Feedback
*   **Section 12.6: The Future of MLOps: Trends, Challenges, and Opportunities**
*   **Project: "Trending Now" – Ensuring Governance and Responsibility**
    *   Applying Model Governance to "Trending Now" (Audit trails for data, training, and predictions).
    *   Responsible AI Considerations for Genre Classification (Potential biases in source data for plots/reviews, fairness in genre representation).
    *   Testing the "Trending Now" system against ethical guidelines.
    *   Reflecting on Team Structure and User Experience Design for the app.
    *   Final thoughts on scaling and evolving the "Trending Now" MLOps project.

---








### Typical ML workflow

<img src="../../_static/mlops/problem_framing/workflow1.jpg"/>

- [ml-ops.org: Typical ML workflow](https://ml-ops.org/content/end-to-end-ml-workflow)



















### References

- [ml-ops.org: Why you Might Want to use Machine Learning](https://ml-ops.org/content/motivation)