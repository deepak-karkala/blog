# Project: "Trending Now" 
##


### Dataset Links

- [HF: vishnupriyavr/wiki-movie-plots-with-summaries](https://huggingface.co/datasets/vishnupriyavr/wiki-movie-plots-with-summaries)
- [CMU Movie Summary Corpus](https://www.cs.cmu.edu/~ark/personas/)
- [Kaggle: MPST: Movie Plot Synopses with Tags](https://www.kaggle.com/datasets/cryptexcode/mpst-movie-plot-synopses-with-tags)

### Business Goals and ML Problem Framing

*   **1.P.1 Defining Business Goals for the "Trending Now" App**
    *   *Primary Goal:* To provide users with accurate and engaging genre classifications for newly released movies and TV shows, leading to increased user satisfaction and app usage.
    *   *Secondary Goals:* To build a showcase MLOps project demonstrating best practices; to understand the effort involved in maintaining such a system.
    *   *Stakeholders:* End-users (want accurate genres), developers (us, building the guide), potential employers/community (learning from the guide).
    *   *Business KPIs (Conceptual):* Daily Active Users (DAU), Session Duration, User Feedback Score on Genre Accuracy, Number of movies/shows processed per day.
*   **1.P.2 Is ML the Right Approach for Genre Classification?**
    *   *Complexity:* Yes, genre is nuanced and can be inferred from plots/reviews which involve complex patterns.
    *   *Data:* Assumed available via scraping (plots, reviews, existing genre tags for training).
    *   *Predictive Nature:* Yes, we are predicting a category (genre).
    *   *Scale:* Can scale to many movies/shows.
    *   *Changing Patterns:* New movies/shows constantly released; review language evolves.
    *   *Simpler Solutions:* A rule-based system (e.g., if "space" in plot, then "Sci-Fi") could be a baseline but likely insufficient for nuanced classification.
*   **1.P.3 Framing the Genre Classification Task**
    *   *Ideal Outcome:* Users quickly find movies/shows of genres they are interested in.
    *   *Model Goal (XGBoost/BERT training model):* Predict the genre(s) of a movie/TV show based on its plot summary and/or aggregated user reviews.
    *   *Model Goal (LLM for "production" inference):* Generate the genre(s) of a movie/TV show based on its plot summary and/or reviews, leveraging its broad knowledge.
    *   *ML Task Type:* Multilabel classification (a movie can belong to multiple genres like "Action" and "Comedy").
    *   *Proxy Labels:* For initial training, we'll rely on existing genre tags from scraped sources (e.g., TMDb). Need to be aware of potential inconsistencies or inaccuracies in these source labels.
*   **1.P.4 Initial Feasibility for "Trending Now"**
    *   *Data Availability:* Moderate. Scraping movie data and reviews is feasible. Quantity and quality of genre labels from sources need assessment.
    *   *Problem Difficulty:* Moderate. Genre classification is a known problem. Using BERT embeddings for text is established.
    *   *Prediction Quality:* For the educational XGBoost/BERT model, "good enough" for demonstration. For the LLM inference, aim for high perceived accuracy by users.
    *   *Technical Requirements:*
        *   Data Ingestion Pipeline: Needs to be robust to website changes.
        *   Model Training Pipeline (XGBoost/BERT): Manageable compute for a small/medium dataset.
        *   Inference Pipeline (LLM): Depends on API latency, cost, and rate limits.
    *   *Cost/ROI:* Primarily educational ROI. For a real app, ROI would depend on user engagement. LLM API costs are a factor.
    *   *Ethical Considerations:* Potential bias in training data genres (e.g., certain types of films from certain regions might be underrepresented or miscategorized in source data). Ensure diverse genres are handled.
*   **1.P.5 Success Metrics for the "Trending Now" Genre Model and App**
    *   *Business Success (App):* Increase in conceptual DAU (e.g., measured by guide readers engaging with project sections), positive feedback on project clarity.
    *   *Model Success (XGBoost/BERT - Offline):*
        *   Primary Metric: Macro F1-score (to handle genre imbalance).
        *   Acceptability: Precision/Recall per genre > 70% (example).
    *   *Model Success (LLM - Online/Conceptual):*
        *   Primary Metric: User-perceived accuracy of genre (qualitative feedback, or A/B test if comparing LLM prompts).
        *   Acceptability: Latency for LLM API response within acceptable limits for user experience.

---


### Blueprinting MLOps Strategy

Let's define the initial MLOps strategy for our "Trending Now" application.

*   **2.P.1 Applying MLOps Principles to "Trending Now"**
    *   *Automation:* Target automation for data ingestion, LLM inference calls, and potentially model retraining (if XGBoost/BERT path is further developed). CI/CD for backend API.
    *   *Reproducibility:* Version control for code (FastAPI, frontend, scraping scripts), data (using DVC for scraped data/labels), model artifacts (XGBoost/BERT in registry), prompts (versioned config/code), and environment (Docker).
    *   *Continuous X:* Implement CI/CD for backend/frontend. Design for potential CT for the XGBoost/BERT model based on new data or performance monitoring. Implement CM for LLM API calls (cost, latency, errors) and model outputs.
    *   *Testing:* Unit tests for backend logic, data validation checks for scraped data, testing LLM prompt effectiveness (offline), integration tests for API.
    *   *Monitoring:* Focus on LLM cost, latency, API errors, output quality (drift in scores/tags), operational metrics of FastAPI service.
    *   *Modularity:* Separate FastAPI backend, frontend, data ingestion pipeline, inference pipeline.
*   **2.P.2 Using the MLOps Stack Canvas to Plan "Trending Now" Infrastructure**
    *   *(Walk through key canvas blocks)*
    *   *Data Sources/Versioning:* Web scraping (APIs/HTML), store raw/processed data (e.g., S3/local Parquet), use DVC for versioning.
    *   *Experiment Management:* Minimal for LLM (prompt variations tracked in Git). Required if developing XGBoost/BERT (MLflow/W&B).
    *   *Feature Store:* Not strictly necessary initially; features (plot/review text) are directly passed to LLM. Could be considered later if complex features are derived.
    *   *CI/CD/CT Orchestration:* GitHub Actions for backend CI/CD. Simple scheduler (cron) or manual trigger for ingestion. Orchestrator less critical initially, could use basic scripts or serverless workflows (AWS Step Functions/Lambda).
    *   *Model Registry:* Minimal registry needed for LLM (track prompt versions/model endpoints). Essential for XGBoost/BERT (SageMaker/MLflow).
    *   *Deployment/Serving:* FastAPI backend (e.g., on AWS App Runner, Google Cloud Run, or simple EC2/VM), LLM via API.
    *   *Monitoring:* CloudWatch/Datadog for basic infra/API monitoring; custom logging/analysis for LLM cost/output drift.
    *   *Metadata Store:* Simple logging initially; potentially MLflow for tracking related artifacts if XGBoost/BERT is developed.
    *   *Build vs Buy:* Buy LLM API. Build backend/frontend/ingestion scripts. Use open-source/managed services for MLOps components where feasible (e.g., Git, DVC, potentially MLflow).
*   **2.P.3 Determining the Initial MLOps Maturity Level for the Project**
    *   Likely starts between Level 0 and Level 1. Manual experimentation (prompt engineering), potentially automated data ingestion pipeline, manual deployment of backend service. Aim to move towards Level 1/2 for the components we build (e.g., CI/CD for backend).
*   **2.P.4 Defining Roles for the "Trending Now" Project Team (Conceptual)**
    *   *ML Engineer/Backend Dev:* Builds FastAPI, integrates LLM, sets up ingestion.
    *   *Frontend Dev:* Builds HTML/CSS/JS and D3.js visualization.
    *   *(Implicit MLOps Role):* Responsible for deployment, monitoring setup, automation scripts. In a small team, this might overlap with the ML/Backend role initially.

---

### Executing the Data Sourcing & Initial Understanding Phase

This section details the practical first steps for our project, applying the concepts from this chapter.

*   **4.P.1 Finalizing Data Sources & Acquisition Strategy**
    *   Identify 2-4 primary API sources for movie/TV metadata (e.g., TMDb). Document API key acquisition and usage limits.
    *   Identify 4-5 diverse websites/blogs for scraping user reviews. Analyze their structure and `robots.txt`.
    *   Decision: For initial phase, focus on batch scraping/API calls.
*   **4.P.2 Implementing Initial Data Collection Scripts (Conceptual - to be built in Ch5)**
    *   Outline Python functions/classes for:
        *   Fetching data from TMDb API for new releases in the last week.
        *   A generic scraper structure for review websites (handle with care regarding terms of service).
*   **4.P.3 Setting up S3 and DVC for Raw Data**
    *   Create S3 buckets (e.g., `trending-now-raw-data`, `trending-now-processed-data`).
    *   Initialize DVC in the project, configure S3 as remote storage.
    *   Perform a sample scrape, add data to DVC, and push to S3.
*   **4.P.4 Exploratory Data Analysis (EDA) on Sample Data**
    *   Load sample scraped movie metadata (plots) and reviews into a Pandas DataFrame.
    *   Perform profiling:
        *   Plot length distributions for plots and reviews.
        *   Identify missing values for key fields (plot, review text, source-provided genres).
        *   Frequency counts for existing genre tags.
    *   Document findings in a Jupyter Notebook (e.g., `01-data-exploration.ipynb`).
*   **4.P.5 Initial Data Documentation for "Trending Now"**
    *   Create a `data_sources.md` file listing chosen APIs and websites, their URLs, purpose, and any known limitations or terms.
    *   Start a simple data dictionary (e.g., in a `data_dictionary.md` or as part of the notebook) describing fields like `title`, `plot_summary`, `review_text`, `source_genre_tags`.
*   **4.P.6 Basic Governance Setup**
    *   Define IAM roles (conceptual) for:
        *   `data-ingestion-pipeline-role` (read from APIs/web, write to raw S3).
        *   `ml-training-pipeline-role` (read from processed S3).
    *   Ensure scraping scripts respect `robots.txt`.

---


### Building the Data Ingestion Pipeline

