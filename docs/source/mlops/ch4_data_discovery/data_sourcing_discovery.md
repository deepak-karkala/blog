# Data Sourcing, Discovery & Understanding

##

### Chapter 4: The Market Run – Data Sourcing, Discovery & Understanding

*(Progress Label: 📍Stage 4: Sourcing the Finest Ingredients)*

### 🧑‍🍳 Introduction: The Quest for Perfect Ingredients

Every Michelin-starred dish begins long before the heat hits the pan – it starts with an uncompromising quest for the finest, freshest, and most suitable ingredients. A chef's ability to source, understand, and select the right raw materials is as crucial as their culinary skill. An exquisite recipe can be ruined by subpar produce, just as a sophisticated ML model can falter without high-quality, relevant data.

In our MLOps kitchen, this initial "market run" translates to **Data Sourcing, Discovery, and Understanding**. It's the foundational phase where we identify, acquire, explore, and validate the data that will fuel our machine learning models. As enterprises become increasingly data-driven, the challenge isn't just the volume of data, but its variety, velocity, and the ability to navigate this "deluge" to find what's trustworthy and relevant. 

This chapter will guide you, the MLOps Lead, through the process of establishing a robust data foundation. We'll look at identifying critical data needs, exploring diverse data sources, strategies for acquisition and ingestion, the pivotal role of Exploratory Data Analysis (EDA), and the importance of data documentation and governance from the very outset. We'll draw lessons from how leading tech companies like Netflix, LinkedIn, and Uber tackle these challenges, ensuring our "Trending Now" kitchen is stocked with ingredients we can trust.

---

### Section 4.1: Identifying Data Requirements (The Chef's Shopping List)

Before heading to the market, a chef meticulously plans their menu and lists the necessary ingredients. Similarly, before diving into data collection, we must clearly define what data is needed to address the ML problem framed in Chapter 1.

*   **4.1.1 Linking ML Problem to Data Needs**
    *   Revisit the ML problem definition: What are the input features hypothesized to predict the target variable?
    *   Example ("Trending Now" genre classification): We need movie/TV show plot summaries, user reviews, and existing genre labels (for training the educational model). We also need metadata like titles, release dates, and OTT platform information.
*   **4.1.2 Brainstorming Potential Data Sources (Where to Find the Ingredients?)**
    *   Consider internal systems, public datasets, third-party vendors, and data that might need to be generated (e.g., via user interaction if applicable).
    *   Initial assessment: Are these sources accessible? What's the likely quality and format?
*   **4.1.4 Defining Data Granularity, Volume, and Freshness Requirements**
    *   *Granularity:* What level of detail is needed? (e.g., individual reviews vs. aggregated review sentiment).
    *   *Volume:* How much data is likely needed for robust model training? (Rough estimates at this stage).
    *   *Freshness:* How up-to-date does the data need to be for training and inference? (e.g., daily updates for "Trending Now"). This impacts ingestion strategy.
*   **4.1.4 Early Consideration of Potential Biases in Data Sources**
    *   Could certain sources systematically over or under-represent specific groups or perspectives? (e.g., review sites dominated by a particular demographic).

---

### Section 4.2: Exploring the Market – Data Sources in the Wild

Once we know what we're looking for, we need to understand the "market" – the diverse landscape of data sources available. Each source has its own characteristics, accessibility, and potential challenges.

*   **4.2.1 User-Provided/Input Data**
    *   *Description:* Data explicitly entered by users (e.g., search queries, form submissions, uploaded content).
    *   *Challenges:* Prone to errors, malformation, varying quality. Requires robust validation.
    *   *Relevance to "Trending Now":* User search queries for "vibe search" could be a future input.
*   **4.2.2 System-Generated Data (The Kitchen's Own Stock)**
    *   *Description:* Logs from applications, services, model predictions, user interactions (clicks, views).
    *   *Characteristics:* Often high volume, can be structured or semi-structured. Less prone to malformation than user input but can still have quality issues.
    *   *Relevance to "Trending Now":* If the app had user accounts, their interaction logs (clicks on movies) would be system-generated data, invaluable for personalization (though not our V1 focus).
*   **4.2.4 Internal Databases & Data Warehouses (The Well-Organized Pantry)**
    *   *Description:* Structured data managed by various business systems (CRM, inventory, sales).
    *   *Characteristics:* Often high quality within their domain, but access and integration can be complex.
    *   *Relevance to "Trending Now":* Less direct for scraping external data, but in an enterprise, this would be a primary source.
*   **4.2.4 Public Datasets & APIs (Open Markets & Specialty Suppliers)**
    *   *Description:* Data made available by governments, research institutions, or companies via APIs (e.g., TMDb, OMDb for movie data).
    *   *Characteristics:* Accessibility varies (free, freemium, paid). Quality and documentation can be inconsistent. API rate limits and terms of service are crucial.
    *   *Relevance to "Trending Now":* Primary source for movie/TV show metadata (plots, release dates, existing genre tags, posters).
*   **4.2.5 Third-Party Data Vendors (Commercial Ingredient Purveyors)**
    *   *Description:* Companies that collect, aggregate, clean, and sell data.
    *   *Characteristics:* Can provide highly specific or difficult-to-obtain data, but comes at a cost. Quality and lineage need careful vetting.
    *   *Relevance to "Trending Now":* Unlikely for our project, but common in industry for enriched datasets.
*   **4.2.6 Web Scraping (Foraging for Unique Ingredients)**
    *   *Description:* Extracting data directly from websites.
    *   *Characteristics:* Can access data not available via APIs. Fragile (breaks with website structure changes), ethically and legally sensitive (respect `robots.txt`, terms of service, copyrights).
    *   *Relevance to "Trending Now":* Primary method for obtaining user reviews from various sites if APIs are unavailable or insufficient.

---

### Section 4.4: The Haul – Data Collection & Ingestion Strategies

Bringing the ingredients into the kitchen requires a plan for collection and initial storage. This involves choosing methods and tools that align with data freshness, volume, and variety.

*   **4.4.1 Batch vs. Streaming Ingestion (Scheduled Deliveries vs. Just-in-Time)**
    *   **Batch Ingestion:** Data collected and processed periodically (e.g., daily, hourly). Suitable for data that doesn't change rapidly or where near real-time processing isn't critical.
        *   *Tools:* ETL/ELT scripts, workflow orchestrators (Airflow).
        *   *"Trending Now" Application:* Daily/weekly scraping of new releases and reviews.
    *   **Streaming Ingestion:** Data collected and processed continuously as it arrives. For high-velocity data or use cases requiring near real-time updates.
        *   *Tools:* Message queues (Kafka, Kinesis), stream processing engines (Flink, Spark Streaming).
        *   *"Trending Now" Application:* Not a primary focus for V1, but could be used if we wanted to process reviews in real-time.
*   **4.4.2 Data Storage Solutions (Pantry Organization)**
    *   **Data Lakes (e.g., AWS S3, Google Cloud Storage):** Store raw data in its native format. Flexible, scalable, cost-effective for large volumes of diverse data types. Ideal for initial landing zone.
        *   *"Trending Now" Application:* Raw scraped HTML/JSON, and processed Parquet files will reside in S3.
    *   **Data Warehouses (e.g., Snowflake, Redshift, BigQuery):** Store structured, processed data optimized for analytics and BI. Less relevant for our project's direct data path but common in enterprises.
    *   **File Formats:**
        *   *Parquet/ORC:* Columnar formats, excellent for analytical queries, compression, and schema evolution. Good for processed data.
        *   *JSON/CSV:* Text-based, human-readable. Good for initial ingestion or smaller datasets.
        *   *"Trending Now" Application:* Store processed data in Parquet on S3 for efficiency.
*   **4.4.4 Data Versioning (Tracking Ingredient Batches)**
    *   Crucial for reproducibility and debugging.
    *   *Tools:* DVC is our chosen tool. Git LFS for smaller artifacts. Some data lake platforms offer versioning (e.g., Delta Lake).
    *   *"Trending Now" Application:* DVC will version datasets in S3 (raw and processed).

---

### Section 4.4: First Impressions – Exploratory Data Analysis (EDA) (The Chef's Initial Taste Test)

Once data is collected, EDA is the process of inspecting, visualizing, and understanding its main characteristics *before* formal modeling. It's about getting a "feel" for the ingredients.

*   **4.4.1 Key Goals of EDA for MLOps**
    *   Understand data structure, content, and quality.
    *   Identify potential data cleaning needs.
    *   Formulate hypotheses for feature engineering.
    *   Detect initial signs of bias or anomalies.
    *   Assess if the data is suitable for the intended ML task.
*   **4.4.2 Data Profiling: Understanding the Ingredients**
    *   *Techniques:* Calculating summary statistics (mean, median, min/max, counts, missing values), distributions (histograms, density plots), value frequencies, correlations.
    *   *Tools:* Pandas `describe()`, `info()`, visualization libraries (Matplotlib, Seaborn, Plotly). Automated EDA tools (e.g., Pandas Profiling, Sweetviz).
    *   *"Trending Now" Application:* Profile scraped plot summaries (length, common words), review text (length, sentiment indicators), existing genre tags (frequency, co-occurrence).
*   **4.4.3 Visualization Techniques for Data Understanding**
    *   Histograms and bar charts for distributions.
    *   Scatter plots for relationships between numerical features.
    *   Box plots for comparing distributions across categories.
    *   Word clouds for text data.
*   **4.4.4 Initial Data Quality Assessment**
    *   Identifying missing values, outliers, inconsistencies, unexpected values.
    *   Checking data types and formats.
    *   This stage informs the Data Cleaning steps in the next chapter.
*   **4.4.5 Documenting EDA Findings**
    *   Crucial to record observations, insights, and potential issues. Jupyter notebooks are excellent for this.

---

### Section 4.5: Curating the Pantry – Data Documentation, Catalogs & Discovery Platforms (Labeling Every Jar)

As the number of datasets and their complexity grows, simply storing them isn't enough. You need systems to document, organize, and make data discoverable – essentially, creating a well-labeled and indexed pantry that everyone in the kitchen can use. This is where data discovery platforms and metadata management become vital. [Data Discovery Platforms- Industry Case Studies.md, MLOps Stack Canvas.md]

*   **4.5.1 The Challenge of Data Discovery in Enterprises**
    *   *Motivations:* Overcoming data silos, reducing time spent searching, building trust, enabling governance. (Synthesized from "Why" section of Data Discovery Platforms- Industry Case Studies.md)
    *   *Lessons from Industry:* Netflix, LinkedIn, Airbnb all built platforms to tackle this.
*   **4.5.2 Key Features of a Data Discovery Solution**
    *   **(Table)** Title: Core Features of Data Discovery Platforms
        Source: Adapted from Table in Section 2 of `Data Discovery Platforms- Industry Case Studies.md`
        *(Include key features like Unified Search, Rich Metadata, Lineage, Profiling, Collaboration, Curation, APIs, Abstraction, Personalization, UI).*
*   **4.5.3 Metadata Management: The Foundation of Discovery (The Labels on the Jars)**
    *   *Technical Metadata:* Schema, data types, source system, location, update frequency.
    *   *Business Metadata:* Descriptions, ownership, business domain, tags, glossary terms, quality scores.
    *   *Operational Metadata:* Lineage info, job run statistics, access patterns.
    *   *MLOps-Specific Metadata:* Link to features, models trained on this data, experiment IDs.
*   **4.5.4 Data Catalogs: Your Team's Shared Inventory List**
    *   *Purpose:* Centralized inventory of data assets.
    *   *Functionality:* Search, browse, understand data context.
    *   *Tools:* Open-source (Amundsen, DataHub, Apache Atlas), Commercial (Alation, Collibra), Cloud-native (AWS Glue Data Catalog, Azure Purview).
*   **4.5.5 Data Lineage: Tracing the Ingredient's Journey**
    *   Visualizing upstream sources and downstream consumers.
    *   Critical for impact analysis, debugging data issues, and building trust.
*   **4.5.6 Architectures for Data Discovery Platforms (Generations 1-4)**
    *   Brief overview of the evolution: Monolithic pull-based -> Service API -> Event-Sourced (Stream-first).
    *   Emphasis on **push-based, event-driven metadata ingestion** for freshness and scalability, as advocated by DataHub.
*   **4.5.7 "Trending Now" Approach to Data Documentation & Discovery (Initial Steps)**
    *   *Simple Manifest/Catalog:* For V1, maintain a simple manifest file (e.g., a versioned CSV or JSON in Git/DVC) listing scraped sources, S3 paths for raw/processed data, DVC versions, and basic descriptions.
    *   *Data Cards (Conceptual):* Create conceptual Data Cards for key datasets (e.g., "Processed Movie Plots," "Aggregated Reviews").
    *   *Metadata in Notebooks:* Thoroughly document EDA findings and data characteristics in Jupyter notebooks.

---

### Section 4.6: Early Governance – Data Security, Privacy, and Compliance (Kitchen Access & Food Safety)

Even at the sourcing and discovery stage, data governance is not an afterthought. Establishing basic security, privacy, and compliance practices early is crucial, especially if dealing with user-generated content or sensitive information.

*   **4.6.1 Identifying Sensitive Data (PII) in Sourced Content**
    *   Even movie plots or reviews could inadvertently contain PII.
    *   Plan for initial checks and potential redaction/anonymization needs in Chapter 5.
*   **4.6.2 Access Control and Permissions Management (Who Gets Keys to the Pantry?)**
    *   *Using AWS IAM:* Define roles and policies for accessing S3 buckets and other AWS resources used for data storage and processing.
    *   Principle of Least Privilege: Grant only necessary permissions.
*   **4.6.3 Respecting Data Source Terms of Service & robots.txt**
    *   Ethical and legal considerations for web scraping and API usage.
    *   Documenting compliance with these terms.
*   **4.6.4 Data Retention and Deletion Policies (Conceptual)**
    *   While not a primary focus for our project's raw data, think about how long scraped data should be kept, especially if it contains user reviews that might later be subject to deletion requests.
*   **4.6.5 Initial Thoughts on Data Quality for Governance**
    *   How will we define and ensure the "trustworthiness" of our sourced data for MLOps, particularly genre labels?

---

### Project: "Trending Now" – Executing the Data Sourcing & Initial Understanding Phase

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

### 🧑‍🍳 Conclusion: Pantry Stocked, Ingredients Understood

The success of our Michelin-Grade ML Kitchen, and indeed our "Trending Now" application, hinges on the quality and understanding of our foundational ingredients – the data. In this chapter, we've navigated the bustling "market" of data sources, from structured APIs to the wild web of user reviews. We've established strategies for collecting and ingesting this data into our S3 "pantry," using tools like DVC to keep track of our "ingredient batches."

Through Exploratory Data Analysis, we've taken our first "taste test," profiling our initial data haul to understand its characteristics, quality, and potential quirks. We've also laid the groundwork for a well-documented and discoverable data ecosystem, recognizing the importance of metadata and learning from how industry giants manage their vast data estates. Crucially, we've embedded early considerations for data governance, ensuring our sourcing practices are secure and responsible from day one.

With our initial ingredients sourced, understood, and documented, and our project-specific data collection planned, our "Trending Now" kitchen is beginning to take shape. The pantry is no longer empty; it's stocked with the raw materials we'll refine in the next chapter: "Mise en Place – Data Engineering for Reliable ML Pipelines."

---