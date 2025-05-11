# ML Problem framing

```{toctree}
:hidden:

```

<!--![](../../_static/mlops/problem_framing/1.png)-->

<!--
<img src="../../_static/mlops/problem_framing/1.png" width="400" height="400" />
<img src="../../_static/mlops/problem_framing/2.png" width="400" height="400" />
<img src="../../_static/mlops/problem_framing/3.png" width="400" height="400" />
<img src="../../_static/mlops/problem_framing/4.png" width="400" height="400" />
<img src="../../_static/mlops/problem_framing/5.png" width="400" height="400" />
<img src="../../_static/mlops/problem_framing/6.png" width="400" height="400" />
<img src="../../_static/mlops/problem_framing/7.png" width="400" height="400" />
<img src="../../_static/mlops/problem_framing/8.png" width="400" height="400" />
<img src="../../_static/mlops/problem_framing/9.png" width="400" height="400" />
<img src="../../_static/mlops/problem_framing/10.png" width="400" height="400" />

<img src="../../_static/mlops/problem_framing/mlops_probframing.png" width="400" height="400" />
<br/>
Click <a href="../../_static/mlops/problem_framing/mlops_probframing.png">here</a> for high resolution image.
-->

---

## **Chapter 1: Crafting the Vision – The Art of ML Problem Framing**

**(Progress Label: 📍Stage 1: The Restaurant Concept & Vision)**

### 🧑‍🍳 Introduction: From Vision to Michelin Star

Before a single ingredient is sourced, before the first pan heats up, a Michelin-starred restaurant begins with a *vision*. What kind of experience will it offer? Who are the diners it aims to delight? What defines its unique culinary identity? This foundational concept guides every subsequent decision, from menu design to kitchen layout.

Similarly, building a successful Machine Learning powered application doesn't start with complex algorithms or vast datasets. It begins with **ML Problem Framing**: the critical process of translating a business need into a well-defined, feasible, and measurable machine learning task. Getting this stage right is paramount; a flawed framing can lead an ML project astray, wasting resources and failing to deliver real value, no matter how sophisticated the eventual model.

This chapter lays the groundwork for our "Michelin-Grade ML Kitchen." We'll explore how to dissect a business problem, determine if ML is the appropriate "cuisine," define precisely what our ML "dish" should achieve, assess if we have the "kitchen capabilities" (feasibility), and establish how we'll measure "Michelin-star" success.

**Overall Process Flow**
Let's visualize the ML Problem Framing process, linking it to our restaurant concept:

<img src="../../_static/mlops/problem_framing/steps.svg" width="400" height="400" />

---

### 1. Understanding the Business Objective (Why Build This 'Dish'?)

Every ML project must originate from a genuine business need or opportunity. Data scientists and ML engineers can get excited by technical challenges and state-of-the-art models, but if the project doesn't ultimately contribute to the business's goals, it's unlikely to succeed or even survive.

*   **Identify the Core Business Problem/Goal:** What specific pain point are you trying to alleviate or what opportunity are you trying to capture? Are you aiming to reduce costs, increase revenue, improve customer satisfaction, mitigate risk, or achieve something else? Be specific.
*   **Align with Stakeholders:** ML projects involve diverse stakeholders (ML engineers, product managers, salespeople, infrastructure teams, domain experts, managers, legal/compliance) who often have conflicting priorities. Engage them early to understand their perspectives and agree on the project's primary business objectives. For example:
    *   *ML Engineers* might focus on model accuracy and interesting technical challenges.
    *   *Sales* might prioritize features that drive immediate revenue (e.g., recommending expensive items).
    *   *Product Managers* might worry about latency, user experience, and avoiding errors.
    *   *Infrastructure/MLOps Engineers* might focus on scalability, maintainability, and platform stability.
    *   *Managers* need to justify the ROI and ensure alignment with broader business strategy.
*   **Define Quantifiable Business Metrics:** How will you measure the *business* impact? Vague goals like "improve user experience" are insufficient. Define concrete, measurable Key Performance Indicators (KPIs). Examples:
    *   Increase ad click-through rate (CTR) by X%.
    *   Reduce customer churn rate by Y%.
    *   Decrease manual effort in process Z by H hours per week.
    *   Increase average revenue per user (ARPU) by $A.
    *   Reduce fraudulent transaction value by $B per quarter.
*   **Connect ML Potential to Business Value:** Crucially, hypothesize *how* an ML solution could realistically influence these business metrics. Don't assume a better model metric automatically translates to better business outcomes. Many projects fail because this link is weak or unproven. We need evidence (often from experiments later) that improving the model's predictions directly impacts the target KPI.
*   **Discuss and agree on the level of model explainability**
    *   Discuss and agree with the business stakeholders on the acceptable level of model explainability required for the use case. Use the agreed level as a metric for evaluations and tradeoff analysis across the ML lifecycle. Explainability can help with understanding the cause of a prediction, auditing, and meeting regulatory requirements. It can be useful for building trust ensuring that the model is working as expected.
    *   Choose good baselines – Shapley values determine the contribution that each feature made to model prediction. SHAP Baselines for Explainability are crucial to building fair and explainable ML models. Choose the baseline carefully since model explanations are based on deviations from the baseline (the baseline, in the ML context, is a hypothetical instance). You can choose a baseline with a ‘low information content’ (e.g., by constructing an average instance from the training dataset by taking either the median or average for numerical features and the mode for categorical features) or a baseline with ‘high information content’ (e.g., an instance which represents a particular class of instances that you are interested in). 


---

### 2. Is Machine Learning the Right Ingredient? (The Initial Feasibility Check)

ML is powerful but complex and resource-intensive. It introduces technical debt and operational overhead. Before committing, critically assess if ML is truly necessary and appropriate for your business problem.

**When to Consider ML:**

1.  **Complex Patterns:** The problem involves patterns too complex for humans to define explicitly with rules (e.g., image recognition, natural language understanding). ML can be seen as "Software 2.0" where rules are learned, not coded.
2.  **Existing Data:** Sufficient, relevant data is available or can be collected. ML learns from data. (Though fake-it-til-you-make-it or zero-shot learning exist, they have caveats).
3.  **Predictive Nature:** The core task involves making predictions (forecasting, classification, estimating values). ML excels at making many cheap, approximate predictions.
4.  **Scale:** The task needs to be performed many times, making the upfront investment in ML worthwhile. Scale also generates more data.
5.  **Repetitive Task:** Patterns repeat often, making it easier for models to learn.
6.  **Changing Patterns:** The underlying patterns evolve over time, making hardcoded rules brittle. ML systems can adapt through retraining/continual learning.
7.  **Tolerance for Error:** The cost of occasional wrong predictions is acceptable or manageable. (e.g., movie recommendation vs. medical diagnosis).

**When NOT to Use ML (or Use with Caution):**

1.  **Simple Solutions Work:** If simple heuristics, rules-based systems, or traditional statistical methods can solve the problem adequately, use them! *Always try simple baselines first.*
2.  **Insufficient/Poor Data:** Not enough data, poor quality data, or inability to collect necessary data.
3.  **No Clear Patterns:** The process might be inherently random or too chaotic for patterns to be learned reliably.
4.  **"Needle in a Haystack":** Extremely rare events can be very hard to predict without massive, specialized datasets and techniques.
5.  **High Cost of Error:** Mistakes have severe consequences (financial, safety, ethical), requiring extremely high reliability often difficult for standard ML.
6.  **Interpretability is Paramount & Models are Black Boxes:** If explaining *exactly* why a prediction was made is legally or ethically essential, complex models might be unsuitable unless paired with robust interpretability techniques.
7.  **Not Cost-Effective:** The cost of developing and maintaining the ML system outweighs the potential benefits.
8.  **Unethical Application:** The intended use could lead to unfair discrimination, privacy violations, or other harms.

**Assess Organizational Readiness:**

*   **Data Maturity:** Is data being collected reliably? Are there existing data pipelines? Is data quality understood?
*   **ML Talent:** Do you have personnel with the right skills (Data Science, ML Engineering, MLOps)? Or a plan to acquire/train them?
*   **Consumer vs Enterprise ML**
    *   Enterprise applications might have stricter accuracy requirements but be more forgiving with latency requirements. 
    *   Consumer apps ML tend to be easy to distribute but difficult to monetise. Whereas enterprise ML problems are hard to figure out unless one is extremely familiar with the system. 
*   **Infrastructure:** Do you have the necessary compute resources and platform capabilities?


**ML Product Archetypes (Not All ML Projects Should Be Planned The Same Way)**

*   **Software 2.0 use cases**:  Taking something that software or a product does in an automated fashion today and augmenting its automation with machine learning.  Example: Code completion in IDE
    *   Is the new model truly much better?
    *   How can the performance continue to increase across iterations?
*   **Human-in-the-loop systems**: Helping humans do their jobs better by complementing them with ML-based tools. Example: turning sketches into slides, 
    *   consider more the context of the human user and what their needs might be. 
    *   How good does the system actually have to be to improve the life of a human reviewing its output?
*   **Autonomous systems**: Systems that apply machine learning to augment existing or implement new processes without human input. Example full self-driving
    *   focus heavily on the failure rate and its consequences. 
    *   leveraging humans in the loop can make development more feasible by adding guardrails and reducing the potential impact of failures.


**Data Flywheels**

* Strongly consider the concept of the data flywheel. For certain ML projects, as you improve your model, your product will get better and more users will engage with the product, thereby generating more data for the model to get even better. It's a classic virtuous cycle and truly the gold standard for ML projects.

*   **Do you have a data loop?**
    *   To build a data flywheel, you crucially need to be able to get labeled data from users in a scalable fashion. This helps increase access to high-quality data and define a data loop.
*   **Can you turn more data into a better model?**
    *   Make sure you can actually translate data scale into better model performance.
*   **Does better model performance lead to better product use?**
    *   You need to verify that improvements with models are actually tied to users enjoying the product more and benefiting from it!


**Simple Decision Flow - Should We Use ML?**

<img style="background-color:#FCF1EF" src="../../_static/mlops/problem_framing/ml_feasibility.svg"/>


---

### 3. Defining the ML Problem (Translating Vision to Recipe)

Once you've confirmed ML is a potential fit, you need to translate the broad business goal into a specific, technical ML problem formulation. This involves defining the inputs, the desired outputs, and the objective function that guides learning.

*   **Define the Ideal Outcome vs. Model's Goal:**
    *   **Ideal Outcome:** What is the ultimate task the *product* should perform for the user/business (stated earlier)? Example: "Recommend useful videos."
    *   **Model's Goal:** What specific prediction should the *ML model* make to help achieve that outcome? Example: "Predict whether a user will click 'like' on a video."
*   **Identify the Model's Output Type:** This determines the fundamental ML task.
    *   **Classification:** Predicts a category/class.
        *   *Binary:* Two classes (e.g., spam/not spam, fraud/not fraud).
        *   *Multiclass:* More than two classes, one label per input (e.g., classifying news articles into 'Sports', 'Politics', 'Tech').
        *   *Multilabel:* More than two classes, potentially multiple labels per input (e.g., tagging an image with 'dog', 'park', 'frisbee'). *Often challenging due to label annotation and varying output structure.*
        *   *High Cardinality:* Multiclass/Multilabel with a very large number of possible classes (e.g., product categorization with thousands of categories).
    *   **Regression:** Predicts a continuous numerical value (e.g., house price, temperature, delivery time).
    *   **Generation:** Creates new data (text, images, audio). Usually involves customizing pre-trained models via prompt engineering, fine-tuning, or distillation.
*   **Framing Matters:** The *same* business problem can often be framed as different ML tasks.
    *   *Example:* Predicting the next app a user will open.
        *   *Bad Framing (Multiclass):* Input: User/Context Features -> Output: Probability distribution over *all* apps. (Hard to scale with new apps).
        *   *Better Framing (Regression):* Input: User/Context Features + *App Features* -> Output: Single probability (0-1) of opening *that specific app*. (Easier to add new apps).
    *   *Consider thresholds:* A regression output (e.g., probability score) can be converted to classification by applying a threshold. Sometimes, framing directly as classification is better if the business logic relies heavily on fixed thresholds.
*   **The Label Challenge - Proxy Labels:** Often, the ideal outcome isn't directly measurable or available as a label in your data (e.g., "is this video *useful*?").
    *   You must choose a **proxy label**: a measurable feature in your dataset that *approximates* the ideal outcome.
    *   *Examples for "useful video":* Clicked 'like'? Shared? Watched > X%? Clicked play? Rewatched?
    *   *Pitfalls:* Proxy labels are imperfect. Each choice can introduce bias or unintended consequences (e.g., predicting clicks might optimize for clickbait, not usefulness). Choose the proxy that best aligns with the ideal outcome and has the fewest negative side effects for *your* specific context.

**Problem Framing Examples**

| Business Problem                       | Ideal Outcome                | Model Goal (Prediction)                | ML Task Type     | Potential Proxy Label Issues                                |
| :------------------------------------- | :--------------------------- | :------------------------------------- | :--------------- | :---------------------------------------------------------- |
| Reduce customer support costs          | Route tickets efficiently    | Predict correct support dept (A,B,C)   | Multiclass Class | N/A (assuming departments are clear)                      |
| Increase e-commerce sales              | Recommend relevant products  | Predict probability user buys item X | Regression       | Clicks (clickbait?), Add-to-cart (abandonment?)           |
| Detect fraudulent credit card usage    | Block fraudulent transactions | Predict if transaction is fraud      | Binary Class     | N/A (fraud labels usually available, though maybe sparse) |
| Personalize news feed                | Show engaging content        | Predict probability user clicks article | Regression       | Click (favors controversial?), Time Spent (favors long?)   |
| Automatically moderate forum comments | Remove toxic comments        | Predict if comment is toxic            | Binary Class     | Defining "toxic" objectively, edge cases, sarcasm         |

---

### 4. Assessing Feasibility & Risks (Can We Execute This Vision?)

Now, dig deeper into whether the framed ML problem is realistically achievable given your constraints and potential hurdles.

*   **Data Availability & Quality:**
    *   *Quantity:* Do you have *enough* labeled examples, especially for classification (often 100s-1000s per class)? Are rare classes represented?
    *   *Quality:* Is the data clean, relevant, and relatively unbiased? How expensive/difficult is labeling?
    *   *Feature Availability:* Will all features used for training *actually be available* at prediction time (serving)? This is a common pitfall!
        * For example, suppose a model predicts whether a customer will click a URL, and one of the features used in training include user_age. However, when the model serves a prediction, user_age isn't available, perhaps because the user hasn't created an account yet.
    *   *Regulations/Privacy:* Are there legal or privacy constraints (GDPR, CCPA) on using the data? How will you handle user consent and data security?
*   **Problem Difficulty & ML Limitations:**
    *   *Solved Before?:* Has your org or the wider community (Kaggle, papers) tackled similar problems? Leverage existing work if possible.
    *   *Human Benchmark:* How well can humans perform this task? This gives a baseline for expected difficulty.
    *   *Known Hard ML Problems:* Be wary of tasks requiring high reliability (autonomous systems, critical healthcare), complex structured outputs, long-term planning/reasoning, or strong generalization to out-of-distribution data.
    *   *Adversaries:* Will actors actively try to exploit or poison your model (e.g., spam filters, fraud detection)? This requires ongoing vigilance.
*   **Required Prediction Quality:**
    *   *Cost of Errors:* What are the consequences (financial, user trust, safety, legal, ethical) of wrong predictions?
    *   *Required Performance:* Does the model need near-perfect accuracy, or is "better than random" sufficient to provide value? Higher accuracy requirements exponentially increase cost and effort. Recognize diminishing returns.
    *   *Generative AI Specifics:* Assess required factual accuracy (mitigating confabulations/hallucinations) and tolerance for biased/toxic/plagiarized output.
*   **Technical Requirements:**
    *   *Latency/Throughput:* How fast must predictions be (milliseconds for real-time UI vs. hours for batch reports)? How many predictions per second (QPS)? (Remember latency distribution - p50, p90, p99 are often more meaningful than average).
    *   *Compute/Memory Resources (RAM):* How much compute is needed for training and serving? Can the model fit on the target platform (cloud, web browser, mobile/edge device)?
    *   *Interpretability:* Is explaining predictions a requirement (debugging, user trust, regulations)?
    *   *Retraining Frequency:* How often does the data change? How frequently will the model need retraining? This impacts cost and infrastructure.
*   **Cost & ROI:**
    *   *Human Costs:* Team size, expertise needed (DS, ML Eng, MLOps, Labelers, SMEs).
    *   *Machine Costs:* Compute (CPU/GPU/TPU for training/inference), storage, data licensing/labeling fees, serving costs.
        *   Perform pricing model analysis-Analyze each component of the workload. Determine if the component and resources will be running for extended periods and eligible for commitment discounts.
        *   Use managed services to reduce total cost of ownership (TCO)
    *   *Is it Worth It?:* Does the estimated business value justify the projected costs and risks? Define the overall ROI and opportunity cost. Develop a cost-benefit model.
*   **Define the overall environmental impact or benefit**
    *   How does this workload support our overall sustainability mission?
    *   How much data will we have to store and process?
    *   What is the impact of training the model?
    *   How often will we have to re-train?
    *   What are the impacts resulting from customer use of this workload?
    *   What will be the productive output compared with this total impact? 
*   **Ethical Considerations & Fairness Review:**
    *   Actively look for potential biases in data or model outcomes. Who might be negatively impacted?
    *   Is the intended use fair and responsible? Consider compliance requirements. Use tools like SageMaker Clarify to help detect bias early.

**Feasibility Checklist**

| Category             | Checkpoint                                        | Assessment (Low/Med/High Risk or Green/Yellow/Red) | Notes                                                  |
| :------------------- | :------------------------------------------------ | :------------------------------------------------- | :----------------------------------------------------- |
| **Data**             | Sufficient quantity & quality available?          | Med                                                | Need more data for rare classes                        |
|                      | Labeling feasible/affordable?                     | Green                                              | Existing labeling pipeline                             |
|                      | Features available at serving time?               | Green                                              | Verified                                               |
|                      | Privacy/Regulatory compliant?                     | Yellow                                             | Need legal review for GDPR implications                |
| **Problem Difficulty**    | Similar problems solved successfully?             | Green                                              | Internal team did similar project                      |
|                      | High reliability required?                        | Med                                                | Wrong recommendations impact revenue                   |
|                      | Adversarial attacks likely?                       | Low                                                | Not expected for this use case                         |
| **Prediction Quality**    | Cost of errors acceptable?                        | Med                                                | Bad recommendations -> churn                           |
|                      | Can achieve necessary quality?                    | Yellow                                             | Initial baseline shows promise, needs improvement      |
| **Technical Requirements**        | Latency target achievable? (<100ms p99)           | Yellow                                             | Simple models meet, complex ones don't                 |
|                      | Fits target platform/resources?                   | Green                                              | Cloud deployment, flexible resources                 |
|                      | Interpretability needed?                          | Low                                                | Not a strict requirement                               |
| **Cost/ROI**         | Estimated cost within budget?                     | Green                                              | Approved budget                                        |
|                      | Positive ROI projected?                           | Green                                              | Cost-benefit model looks positive                    |
| **Ethics/Fairness**  | Potential bias/fairness issues identified?        | Yellow                                             | Potential bias against new users, needs monitoring |

---

### 5. Defining Success Metrics (What Does a 'Michelin Star' Look Like?)

Clearly define how you will measure success *before* you start building. Distinguish between business goals and model evaluation metrics.

*   **Business Success Metrics (Revisited):** These are the ultimate measures of impact, tied to the original business objective (e.g., "Reduce churn by 5% within 6 months," "Increase CTR by 10% in Q3"). Define ambitious but specific targets and the timeframe for measurement (e.g., 6 days, 6 weeks, 6 months post-launch).
*   **Model Evaluation Metrics:** These are technical metrics used during development to assess model performance on specific ML tasks (e.g., Accuracy, Precision, Recall, F1-Score for classification; RMSE, MAE for regression; BLEU, ROUGE for text generation).
    *   **Choose a Single Primary Metric:** Select *one* key model metric to optimize during experimentation (e.g., prioritize Recall in fraud detection). This simplifies model comparison.

        *   **Handling Multiple Objectives: The Case for Decoupling**

            Often, an ML project needs to satisfy multiple, sometimes conflicting, goals. For instance, a newsfeed might aim to maximize user engagement (clicks) while also ensuring content quality and minimizing misinformation. Simply optimizing for engagement might unintentionally promote low-quality or extreme content.

            Two main strategies exist for handling such multi-objective problems:

            1.  **Combined Loss Function:** Define individual loss functions for each objective (e.g., `quality_loss`, `engagement_loss`) and train a *single model* to minimize a weighted combination (e.g., `loss = α * quality_loss + β * engagement_loss`).
                *   *Drawback:* Adjusting the balance between objectives (tuning the weights α and β) requires **retraining the entire model**, which can be slow and costly.

            2.  **Decoupled Models (Recommended):** Train *separate models*, each optimized for a single objective (e.g., one model predicts quality, another predicts engagement). Then, combine the *outputs (scores)* from these models using weights (e.g., `final_rank_score = α * quality_score + β * engagement_score`) to make the final decision (like ranking).
                *   *Advantages:*
                    *   Allows **tuning the trade-off weights (α, β) easily and quickly without retraining** the underlying models.
                    *   Facilitates **independent maintenance and retraining schedules** for each model, which is beneficial as different objectives (like spam detection vs. quality assessment) might evolve at different speeds.

            **In summary:** When faced with multiple objectives, decoupling them by training separate models and combining their outputs is often the preferred approach. It offers greater flexibility in tuning the system's behavior and simplifies long-term maintenance.


    *   **Define Acceptability Goals (Satisficing Metrics):** Set minimum acceptable thresholds for other important metrics (e.g., "Precision must be >= 80%," "Latency must be < 150ms p95").
    *   **Custom Metrics:** Consider developing custom metrics that directly reflect business objectives if standard metrics are insufficient (e.g., weighting errors based on their financial impact).
    *   **Model Complexity vs Impact:**
        *   For many tasks, a small improvement in performance can result in a huge boost in revenue or cost savings. Example: clickthrough rate in e-commerce website. 
        *   For many tasks, a small improvement might not be noticeable for users. If a simple model can do a reasonable job, complex models must perform significantly better to justify the complexity.

*   **The Disconnect & Validation:** *Remember: Excellent model metrics DO NOT guarantee business success.* A model with 99% accuracy might still fail to improve the business KPI if its predictions aren't actionable or if it optimizes for the wrong thing (e.g., predicting clicks instead of purchases).
    *   For instance, let's say your team develops a model to increase revenue by predicting customer churn. In theory, if you can predict whether or not a customer is likely to leave the platform, you can encourage them to stay. Your team creates a model with 95% prediction quality and tests it on a small sample of users. However, revenue doesn't increase. Customer churn actually increases. Here are some possible explanations:

        - Predictions don't occur early enough to be actionable. The model can only predict customer churn within a seven-day timeframe, which isn't soon enough to offer incentives to keep them on the platform.

        - Incomplete features. Maybe other factors contribute to customer churn that weren't in the training dataset.

        - Threshold isn't high enough. The model might need to have a prediction quality of 97% or higher for it to be useful.

    *   Solution: Plan for **A/B testing** or **gradual rollouts** (e.g., deploy to 1% of users) to measure the *actual* impact of the deployed model on the target *business* metrics. This validates the connection between your model and business success.
*   **Evaluate Progress Towards Success:** Use the business metrics gathered from real-world testing to determine if the project is on track.
    *   *Is the model moving the needle?* Even if evaluation metrics are mediocre, is it positively impacting business KPIs? -> Worth improving.
    *   *Is the model plateauing?* Great evaluation metrics, but no further improvement in business KPIs? -> Maybe good enough, or the link is broken.
    *   *Is it failing?* Poor evaluation metrics AND negative/no impact on business KPIs? -> Re-evaluate framing or abandon.

*   **Monitor model compliance to business requirements**
    *   Machine learning models degrade over time due to changes in the real world, such as data drift and concept drift. If not monitored, these changes could lead to models becoming inaccurate or even obsolete over time. It’s important to have a periodic monitoring process in place to make sure that your ML models continue to comply to your business requirements, and that deviations are captured and acted upon promptly.
        *   Agree on the metrics to monitor - Clearly establish the metrics that you want to capture from your model monitoring process. These metrics should be tied to your business requirements and should cover your dataset-related statistics and model inference metrics.
        *   Have an action plan on a drift–If anunacceptable drift is detected in a dataset or the model output, have an action plan to mitigate it based on the type of drift and the metrics associated. This mitigation could include kicking off a retraining pipeline, updating the model, augmenting your dataset with more instances, or enriching your feature engineering process.

**Business vs. Model Metrics**

| ML Use Case             | Business Success Metric(s)        | Primary Model Metric (Example) | Acceptability Goal(s) (Example)         |
| :---------------------- | :-------------------------------- | :----------------------------- | :-------------------------------------- |
| Spam Email Filter       | Reduce user reports of spam by 30% | Precision                      | Recall >= 99.5%                         |
| Product Recommendation  | Increase conversion rate by 5%    | Purchase Prediction Accuracy   | Latency < 200ms p99, Diversity Score > X |
| Churn Prediction        | Reduce monthly churn by 1%        | Recall (of likely churners)    | Precision >= 70%                        |
| Predictive Maintenance  | Reduce machine downtime by 15%    | F1-Score (predicting failure)  | False Positive Rate < 2%                |
| Medical Image Diagnosis | Improve diagnostic accuracy       | AUC                            | Specificity >= 98% (few false positives) |

---

### 6. Planning the ML Project (The Initial Kitchen Setup)

ML development is inherently experimental and uncertain. Unlike traditional software where the path is often clearer, ML involves trial-and-error. Estimating timelines is hard, and success isn't guaranteed. Adopt an experimental mindset from the start.

*   **Iterative Approach:** Plan for iteration. You'll likely revisit earlier steps (data collection, feature engineering, model selection) as you learn more. The process is cyclical, not linear.
*   **Start Simple & Fail Fast:** Begin with simple models and baselines (even non-ML ones). Attempt approaches with lower cost/effort first. If they fail, you haven't wasted significant resources.
*   **Time-Boxing:** Set specific timeframes for experiments or tasks (e.g., "Spend 2 weeks evaluating data feasibility," "Allocate 1 sprint to build a baseline model"). Re-assess progress and feasibility at the end of each timebox.
*   **Scope Management:** Be prepared to scope down requirements if initial approaches prove too difficult or costly. Incremental improvements over time often lead to impactful solutions.
*   **Documentation:** From the outset, document assumptions, decisions, data sources, and metrics. This is crucial for reproducibility and collaboration. (This links to lineage tracking later).


**AWS Well-Architected ML lifecycle**

<img src="../../_static/mlops/problem_framing/aws_well_architected.png" width="80%"/>

**AWS Well-Architected ML lifecycle phases**

<img src="../../_static/mlops/problem_framing/aws_ml_lifecycle_phases.png"/>

- [Source: AWS Well-Architected Framework: Machine Learning Lens](https://docs.aws.amazon.com/wellarchitected/latest/machine-learning-lens/machine-learning-lens.html)


**ML development phases**

<img src="../../_static/mlops/problem_framing/dev_phases.png"/>

- [Google Foundational Course: Managing ML Projects: ML development phases](https://developers.google.com/machine-learning/managing-ml-projects/phases)

---

### 7. Real-World Examples (Case Studies from Famous Kitchens)

Let's apply this framing process to a couple of scenarios:

*   **Case Study 1: E-commerce Product Recommendation (like Amazon/Netflix)**
    *   *Business Objective:* Increase user engagement (time on site) and purchase conversion rate. Stakeholders: Product, Sales, Eng. Business Metrics: Avg. Session Duration, Conversion Rate, Revenue per Visit.
    *   *Is ML Needed?:* Yes, predicting individual user preferences at scale is complex. Simple rules (e.g., show bestsellers) are insufficient for personalization. Data (user history, item metadata) is available.
    *   *ML Problem Framing:* Ideal Outcome: Show users items they will like and buy. Model Goal: Predict the probability a user *clicks* or *purchases* item X given context Y. Task: Regression (predict probability) or Classification (predict click/no-click, buy/no-buy). Proxy Labels: Click data is easy but prone to clickbait issues. Purchase data is better but sparser. Watch time/Like/Add-to-cart are other options. Need to choose carefully based on business goal.
    *   *Feasibility:* Data volume is high. Latency needs to be low (<200ms). Scalability is crucial. Interpretability might be useful ("Why was this recommended?"). Cost of errors (bad recommendations) is relatively low. ROI potential is high.
    *   *Success Metrics:* Business: Increase Conversion Rate by X%. Model: Optimize for AUC on purchase prediction (primary), maintain Latency < 200ms p99 (acceptability). A/B test against non-ML baseline (e.g., popular items).
*   **Case Study 2: Fraud Detection (Financial Services)**
    *   *Business Objective:* Reduce financial losses due to fraudulent transactions. Stakeholders: Risk Mgmt, Compliance, Eng. Business Metrics: $ Value of Fraud Prevented, False Positive Rate (blocking legitimate transactions).
    *   *Is ML Needed?:* Yes, fraud patterns are complex, constantly evolving, and hard to capture with static rules. Transaction data is available.
    *   *ML Problem Framing:* Ideal Outcome: Block fraudulent transactions without inconveniencing legitimate users. Model Goal: Predict if a transaction is fraudulent. Task: Binary Classification (Fraud/Not Fraud). Labels: Historical fraud data exists, but is highly imbalanced (fraud is rare).
    *   *Feasibility:* Data imbalance is a major challenge. High accuracy *and* low false positives needed (cost of errors is high for both false negatives - fraud loss, and false positives - customer friction). Latency must be very low for real-time blocking. Adversarial attacks are likely. Interpretability may be needed for disputes/audits.
    *   *Success Metrics:* Business: Reduce Fraud Loss by Y% while keeping False Positive Rate below Z%. Model: Optimize for Recall on Fraud class (primary), maintain Precision >= X% (acceptability), Latency < 50ms p99.

---

### 🧑‍🍳 Conclusion: The Foundation for Culinary Excellence

Just as a clear restaurant concept, a target clientele, and a vision for the signature dish provide the essential blueprint before any cooking begins, ML Problem Framing lays the critical foundation for building impactful AI-powered applications.

We've seen that this involves deeply understanding the business need, rigorously evaluating if ML is the right approach, carefully translating the goal into a specific ML task with appropriate outputs and labels (even proxies), assessing the feasibility across data, technical, cost, and ethical dimensions, and defining clear metrics for both model performance and ultimate business success.

Neglecting this stage is like a chef trying to create a masterpiece without knowing the cuisine, the diners, or the ingredients available. By investing time in thoughtful problem framing—in crafting our vision—we set our "ML Kitchen" on the path towards creating truly valuable and successful "dishes." The next step in our journey? Sourcing the finest ingredients – Data Discovery and Acquisition.

---


### Project: "Trending Now" – Applying ML Problem Framing

Now, let's apply the principles from this chapter to our "Trending Now Movies/TV Shows Genre Classification" MLOps project.

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

### References
- [Designing Machine Learning Systems by Chip Huyen](https://www.oreilly.com/library/view/designing-machine-learning/9781098107956/)
- [Full Stack Deep Learning - 2022 Course](https://fullstackdeeplearning.com/course/2022/)
- [AWS Well-Architected Framework: Machine Learning Lens](https://docs.aws.amazon.com/wellarchitected/latest/machine-learning-lens/machine-learning-lens.html)
- [Google Foundational Course: Managing ML Projects: Feasibility](https://developers.google.com/machine-learning/managing-ml-projects/feasibility)
