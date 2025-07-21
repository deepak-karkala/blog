# A/B Testing

This document synthesizes insights from Eppo, Netflix, and Microsoft, along with expert perspectives, covering the lifecycle of experimentation, advanced testing methodologies, feature management, and best practices for deploying and evaluating machine learning models in production.

### Part 1: Foundations of Digital Experimentation

**1.1. What is Digital Experimentation?**
Digital experimentation is the systematic process of testing and iterating on digital products, features, or marketing strategies by making deliberate changes and measuring their impact on user behavior and Key Performance Indicators (KPIs). Its purpose is to empower businesses to make data-driven decisions, understand user behavior (pain points, preferences, drop-offs), drive conversions (e.g., purchases, sign-ups), and improve overall performance (engagement, satisfaction).

**1.2. Core Experimentation Types:**

*   **A/B Testing (Split Testing):**
    *   **Definition:** Compares two or more variations (A/B/n) of a webpage, app feature, email, or ad by randomly splitting an audience. Typically, only one element is changed at a time between versions A (control/status quo) and B (variation) to isolate its impact.
    *   **"A/B testing vs. Split testing":** For 95% of people, these terms are synonymous. A subtle distinction sometimes made is that "split testing" can refer to testing entirely different versions of a page/app (potentially without a status quo control, e.g., testing two new email designs), while "A/B testing" usually implies a control. However, the core concept of splitting traffic to compare variations is the same. Splitting traffic via random assignment is crucial for creating comparable groups and ensuring a controlled experiment.
    *   **A/B/n Testing:** An extension of A/B testing where "n" represents any number of variations beyond two. Useful for exploring a wider range of ideas simultaneously.
*   **Multivariate Testing (MVT):** Tests multiple variables and their combinations simultaneously (e.g., headline + image + CTA button color) to understand how different elements interact and influence each other. It helps uncover synergies that A/B testing might miss.
*   **Multi-Armed Bandit (MAB) Testing:** A dynamic approach that automatically adjusts traffic allocation to different variations based on their real-time performance. It learns which variation is performing best and directs more traffic to it, useful for quickly identifying and capitalizing on top performers when testing multiple variations.
*   **A/A Testing:**
    *   **Definition:** Two *identical* versions (A and A) are tested against each other. Users are randomly split, and metrics are compared.
    *   **Purpose:** Verifies the accuracy and reliability of the testing setup (tools, randomization, data pipeline). If significant differences are found, it indicates problems. It also serves as a teaching tool about statistical noise, p-values, and error rates.
    *   **Frequency:** Essential when implementing a new testing tool, making major setup changes, encountering data discrepancies, or trying new experiment types (e.g., browser redirects where an A/A redirect can uncover latency issues).
    *   **Interpretation:** Ideally, no statistically significant difference is observed. If p-values are consistently below the threshold (e.g., 0.05), it signals an issue. Advanced A/A testing involves running hundreds/thousands of (simulated) A/A tests and plotting p-value distributions to ensure uniformity, guarding against statistical biases.

**1.3. Benefits of A/B Testing:**
(As highlighted by Eppo and Netflix product philosophy)
*   **Data-Driven Decisions:** Moves beyond assumptions, relying on evidence.
*   **Revenue Growth & Improved Margins:** Identifies what resonates to maximize revenue and profitability.
*   **Enhanced User Experience:** Continuously refines products based on user behavior.
*   **Reduced Risk:** Tests ideas on a smaller scale before full implementation, minimizing costly mistakes.
*   **Innovation & Competitive Edge:** Encourages bold experimentation, uncovers unique insights, and helps outpace competitors. Companies like Amazon, Netflix, Google, and Microsoft build their success on extensive experimentation.
*   **Better Content Engagement & Marketing ROI:** Fine-tunes content and identifies effective marketing strategies.
*   **Deeper Audience Insights:** Provides valuable understanding of user behavior and preferences.

**1.4. The Experiment Plan (Based on Eppo's Guide by Kristi Angel):**
An experiment plan is a blueprint and contract, guiding execution, ensuring rigor, and fostering transparency.
*   **Core Components:**
    1.  **Problem Statement:** Concise description of the business problem, background, proposed solution, and desired outcomes. Links to BRDs/PRDs.
    2.  **Hypothesis:** A distilled, statistical (though often framed from a product POV) statement, e.g., "By introducing X, we expect an INCREASE/DECREASE in Y (primary metric)."
    3.  **User Experience:** Screenshots, wireframes, or descriptions of control and treatment.
    4.  **Target Audience & Triggering:** Who is included (e.g., "all users," "users entering checkout on iOS in NYC") and where experiment allocation occurs.
    5.  **Sampling Strategy:** E.g., "100% of the population, 50% Control, 50% Treatment," or more advanced strategies like switchbacks.
    6.  **Metrics (Primary, Secondary, Guardrail):**
        *   **Primary:** Decision-making metric; deterioration would prevent rollout.
        *   **Secondary:** Supportive, hypothesis-generating; helps tell a robust story.
        *   **Guardrail:** Not expected to move, but movement indicates unintended effects (e.g., page load times, unsubscribes).
    7.  **Method of Analysis:** Statistical methods (t-test, DiD, CUPED++, sequential methods).
    8.  **Baseline Measure & Variance:** For the primary metric, ensuring it reflects the experiment's specific context (population, metric definition).
    9.  **Minimum Detectable Effect (MDE):** The smallest effect size considered meaningful and practically detectable. Iteratively chosen based on plausible lift and acceptable runtime.
    10. **Power and Significance Levels:** Typically 80% power (20% false negative rate) and 5% significance level (alpha).
    11. **Power Calculation (Sample Size Calculation):** Determines samples needed per variant.
    12. **Estimated Runtime:** Based on sample size and traffic, usually minimum two weeks.
    13. **Decision Matrix:** Pre-defined actions for possible outcomes (positive, negative, inconclusive) for decision metrics. Essential for objectivity and expediting post-experiment decisions.

### Part 2: Trustworthy Experimentation (Microsoft's Framework)

Microsoft emphasizes a three-stage approach to ensure trustworthy A/B testing.

**2.1. Pre-Experiment Stage:**
Ensuring a solid foundation before the experiment starts.

*   **Forming a Hypothesis and Selecting Users:**
    *   **Formulate Hypothesis & Success Metrics:** Clear, simple, falsifiable/provable hypothesis. Metrics should include user satisfaction, guardrail, feature/engagement, and data quality. Account for statistical power and the proportion of affected traffic.
    *   **Choose Appropriate Unit of Randomization:**
        *   *Stability of ID:* Cookie-based IDs are unstable; suitable for short-term tests. Monitor cookie churn.
        *   *Network Effects:* If SUTVA (Stable Unit Treatment Value Assumption) is violated (user actions affect others), use cluster-level randomization.
        *   *Enterprise Constraints:* Business limitations might necessitate enterprise-level randomization.
    *   **Check and Account for Pre-Experiment Bias:**
        *   *Retrospective-AA Analysis:* Compare metrics for treatment/control groups in the pre-experiment period.
        *   *Seedfinder:* Generate metrics for many randomization seeds and pick one with least pre-experiment bias.
        *   *Variance Reduction (e.g., CUPED/CUPED++):* Use pre-experiment data for the same metric to reduce bias and variance in the final analysis, increasing sensitivity.

*   **Pre-Experiment Engineering Design Plan:**
    *   **Set Up Counterfactual Logging:** For changes affecting a subset of users (e.g., a specific search answer). Log in control if the user *would have* seen the treatment. Enables apples-to-apples comparison by identifying the "intended-to-treat" population in both groups. Critical for "zoomed-in" or triggered analysis. Ensure correctness via SRM checks on the triggered population.
    *   **Have Custom Control and Standard Control:**
        *   *Custom Control:* Specific to an experiment (e.g., with counterfactual logging).
        *   *Standard Control:* Users not exposed to any experiment code. Comparing custom control to standard control can detect if the experiment setup itself (not the intended treatment) is causing issues.
    *   **Review Engineering Design Choices to Avoid Bias:** Prevent treatment effect leakage through shared infrastructure (e.g., shared ML model components, caches). Ensure resource availability is equal unless part of the treatment.

*   **Pre-Validation by Progressing Through Populations (Safe Rollout):**
    *   **Gradual Rollout Across Different User Populations:** Test on internal/dogfood/beta users before general users to catch egregious issues. Results may not generalize perfectly.
    *   **Gradual Rollout within a User Population:** Start with a small percentage (e.g., 1%) and ramp up (5%, 10%), monitoring metrics at each stage to detect regressions.

**2.2. During-Experiment Stage:**
Monitoring and analyzing the experiment as it runs.

*   **Measure Holistically and Frequently:**
    *   **Complete Metric Set (STEDI: Sensitive, Trustworthy, Efficient, Debuggable, Interpretable):**
        *   *Data Quality Metrics:* Telemetry consistency (Error Rates, Data Loss, Join Rates, SRM).
        *   *Overall Evaluation Criteria (OEC):* User satisfaction/loyalty (Session Count, Success Rate, Utility Rate).
        *   *Local Feature/Diagnostic Metrics:* Usage of specific features (Feature Coverage, CTR, Messages Received).
        *   *Guardrail Metrics:* Aspects not to degrade (Page Load Time, Crash Rate, Abandonment Rate).
    *   **Measure Early and Often:** Compute metrics at regular intervals (even near-real-time for critical ones). Use appropriate statistical methods for multiple hypothesis testing/early peeking (e.g., sequential testing, stronger significance for early reads).

*   **Monitor Metrics to Intervene when Needed:**
    *   **Set Up Alerts:** For major movements in business-critical metrics (Guardrails, OECs), especially SRMs. Alerts can trigger on large movements or p-value thresholds.
    *   **Auto-Shutdown Egregious A/B Tests:** For tests significantly degrading user experience or product (e.g., bugs leading to 404s).

*   **Slice the Analysis with Appropriate Segments:**
    *   **Use Stable Segments:** "Static" segments (Market, Country, Browser, App version, pre-experiment activity) where users remain in the same segment value. Ensure segments are balanced and SRM-free. Avoid dynamic segments (which can be affected by treatment) for direct slicing, or use weighted metrics.
    *   **Segment by Date:** Detect novelty effects (rapid decline of effect), learning effects, or interactions with external events (holidays, sports). A changing treatment effect over time warrants investigation.

**2.3. Post-Experiment Stage:**
Finalizing analysis, making decisions, and institutionalizing learnings.

*   **Verify Treatment Effects Do Not Invalidate Metrics:**
    *   **Ensure Metric Movements Align with Test Set-Up:** Check user characteristics against target population, traffic allocation. Do coverage metrics reflect the change?
    *   **Check for Telemetry-Breaking Changes:** If the treatment unintentionally changes how logs are collected (e.g., improved call logging reliability only in treatment), it biases metrics. Use data quality metrics to detect this.
    *   **Check for Imbalance in Metric Observation Units:**
        *   *Rate Metrics:* If the denominator (itself a metric) changes significantly, it can cause misleading movements in the rate metric. Report the denominator metric separately. Use the most fine-grained denominator that doesn't show stat-sig movement.
        *   *Metric SRM:* Mismatch in the number of a metric's observation units (e.g., more homepage loads in treatment due to engagement, skewing average Page Load Time if not accounted for). Check SRM on observation units.

*   **Estimate the Final Impact of the Treatment:**
    *   **Segment by Triggered/Non-Triggered Users (Triggered Analysis):** Zoom in on the population directly affected using counterfactual logging. Verify trigger correctness (no SRM in triggered scorecard, flat triggered-complement analysis).
    *   **Dilute Gains to Reflect the Trigger Rate:** The delta in triggered analysis isn't the overall impact. Dilute count metric deltas by multiplying by the user trigger rate. For ratio metrics, use more complex methods (e.g., Deng and Hu's variance reduction approach).
    *   **Tradeoff Observed Metric Movements:** If OEC improves but a Guardrail regresses (e.g., richer content vs. slower load time), a pre-defined weighting system or OEC framework reflecting business strategy is needed.

*   **Close the Loop on Experiment Results:**
    *   **When in Doubt, Reproduce Results:** If results are surprising or counter-intuitive (Twyman's Law), re-run the experiment with re-randomized users and potentially higher traffic.
    *   **Regularly Share Experiment Results:** Foster a learning culture via reviews, "ship" emails with A/B results, and company-wide talks (e.g., Microsoft's "Best Experiments Talks").
    *   **Archive Hypothesis, Tests, and Metric Movements:** Create an institutional memory/Experiment Corpus for meta-analysis, understanding metric behavior, and preventing cycles of shipping/unshipping similar changes.

### Part 3: Advanced Experimentation Techniques

**3.1. Clustered Experiments (Eppo):**
*   **Problem:** Traditional A/B testing randomizes individual users. This isn't sufficient when interference effects exist (one user's treatment affects another's experience or outcome) or when analysis units are naturally grouped.
*   **Definition:** Randomizes groups (clusters) of analysis units (e.g., companies, geographical regions, user segments) rather than individual units.
*   **When to Use:**
    1.  **Session-level metrics in user-randomized experiments:** Users act as clusters of sessions. Randomizing by user and analyzing session metrics can violate independence, increasing false positives.
    2.  **Organizational-Level Interference (B2B):** Users in the same organization influence each other (e.g., messaging software).
    3.  **Market-level tests:** E.g., testing a new pricing strategy in select cities.
*   **Statistical Challenge:** Lack of independence among observations within the same cluster. Naive application of traditional stats underestimates variances.
*   **Solution (Clustered Analysis):**
    *   Apply the delta method to the ratio of cluster-aggregated metrics and cluster size.
    *   Mathematically equivalent to Cluster Robust Standard Errors (CRSE) but more scalable.
    *   Accurately estimates variance, accounts for intra-cluster correlations.
    *   Flexible for various metrics (per-user, per-order, timeboxed, filtered).
    *   Allows advanced stats like CUPED++, Sequential Hybrid, Bayesian methods.
*   **Real-World Applications (Eppo Examples):**
    1.  *Measuring Average Order Value (AOV) with User-Level Randomization:* Randomize at user level (cluster of orders), analyze AOV at order level.
    2.  *User-Level Conversion Rate in Company-Randomized Experiments:* Randomize at company level (cluster of users), analyze conversion at user level.
*   **Implementation (Eppo SDK Example):** Pass cluster ID as primary ID, analysis unit as an attribute.
    ```javascript
    const variation = eppoClient.getBooleanAssignment(
      'enable-my-new-feature',
      companyId, // Cluster ID
      { userId: 123 /* Analysis unit attribute */ },
      false
    );
    ```

**3.2. Interleaving for Ranking Algorithms (Netflix):**
*   **Problem:** Discerning wins in core metrics (retention, streaming hours) for ranking algorithms via A/B testing requires large sample sizes and long durations.
*   **Solution:** A two-stage online experimentation process:
    1.  **Stage 1 (Pruning):** Interleaving to quickly identify promising ranking algorithms.
    2.  **Stage 2 (Validation):** Traditional A/B test on the pruned set for longer-term impact.
*   **Interleaving Definition:** Instead of splitting users into groups A and B, a single group is exposed to an *interleaved ranking* generated by blending results from algorithms A and B. Member preference is measured by attributing engagement (e.g., viewed hours) to the algorithm that recommended the item.
*   **Advantages:**
    *   Highly sensitive to ranking quality; detects preferences with much smaller sample sizes (>100x fewer users than A/B testing for Netflix).
    *   Predictive of A/B test success; interleaving metrics correlate strongly with A/B metrics.
    *   Uses a repeated measures design, removing uncertainty from population-level consumption habits and reducing impact of heavy user imbalance.
*   **Addressing Position Bias:** Essential that at any position, an item is equally likely to come from algorithm A or B.
    *   **Team Draft Interleaving:** Mimics sports team selection. Captains (algorithms A and B) toss a coin for first pick, then alternate, selecting their highest-ranked available "player" (video).
*   **Limitations:**
    *   Engineering complexity for implementation and consistency checks.
    *   Relative measurement of user preference; doesn't directly measure absolute metrics like retention (hence the second A/B test stage).

**3.3. Automated Canary Analysis (Netflix's Kayenta & Spinnaker):**
*   **Canary Release:** Reduces deployment risk by rolling out a new version (canary) to a small subset of users alongside the stable version. Traffic is split, and key metrics are compared. If degradation occurs, canary is aborted.
*   **Netflix's Augmented Canary Process:**
    *   **Production Cluster:** Unchanged, current version.
    *   **Baseline Cluster:** Same code/config as production (typically 3 new instances to avoid long-running process effects).
    *   **Canary Cluster:** Proposed changes (typically 3 new instances).
    *   Production gets most traffic; baseline and canary get small, equal amounts.
*   **Kayenta (Automated Canary Analysis Platform):** Integrated with Spinnaker (CD platform).
    *   **Metric Retrieval:** Gets key metrics (HTTP status, response times, exceptions, load avg) from sources like Prometheus, Stackdriver, Datadog, Atlas. Queries based on config file, scope (cluster, time range).
    *   **Judgment:** Compares baseline and canary metrics.
        1.  *Data Validation:* Ensures data exists for both. Marks "NODATA" if not.
        2.  *Data Cleaning:* Handles missing values (NaNs replaced with zeros for errors, removed for others).
        3.  *Metric Comparison:* Classifies each metric as "Pass," "High," or "Low." Uses confidence intervals (Mann-Whitney U test) to detect significant differences.
        4.  *Score Computation:* Ratio of "Pass" metrics to total metrics (e.g., 9/10 Pass = 90% score). Used by Spinnaker to continue or rollback.
    *   **Reporting:** UI in Spinnaker shows canary score, detailed breakdown by metric, input data.
    *   **Additional Features:** Archiving for re-running analysis with new algorithms, pluggable metric sources/judges/storage, REST API.
*   **Success:** More flexible, less hand-tuning than legacy systems. Focuses on semantic meaning of metrics.

### Part 4: Key A/B Testing Metrics to Track (Eppo)

Choosing the right metrics is crucial. Align with business goals, focus on primary vs. secondary metrics, consider the user journey.

1.  **Revenue:** Total money generated. (Total Revenue = Price * Units/Subscriptions Sold). Eppo's warehouse-native approach ensures accurate tracking from source of truth. Improve by increasing conversion, AOV, targeting high-value users.
2.  **Conversion Rate:** % of users completing a desired action (signup, download, purchase). (Conversion Rate = (Conversions / Total Visitors) * 100). Improve with clear CTAs, value proposition, minimizing friction, building trust.
3.  **Profit Margins:** % of revenue kept as profit after costs. (Gross Profit Margin = (Revenue - COGS) / Revenue * 100). A/B test for impact on margins, not just conversions. Understand product/SKU mix.
4.  **Customer Lifetime Value (LTV):** Predicted total revenue per customer. (LTV = Avg Order Value * Purchase Frequency * Customer Lifespan). Improve with excellent CX, nurturing relationships, providing long-term value.
5.  **User Retention Rate:** % of users continuing to engage over time. (Retention Rate = [(Users at Period End - New Users Acquired) / Users at Period Start] * 100). Improve with good onboarding, support, community, loyalty rewards.
6.  **Customer Satisfaction Score (CSAT):** Measures immediate satisfaction via surveys (e.g., 1-5 scale). (CSAT = (Positive Responses / Total Responses) * 100). Improve by easing pain points, timely feedback, closing the loop. (Note: CSAT can be subjective and require larger sample sizes).
7.  **Cart Abandonment Rate (E-commerce):** % of shoppers adding to cart but not purchasing. (Cart Abandonment Rate = (Abandoned Carts / Shopping Sessions Initiated) * 100). Improve by simplifying checkout, transparent shipping costs, multiple payment options, retargeting.
8.  **Click-Through Rate (CTR):** % of users clicking links/ads/CTAs after seeing them. (CTR = (Clicks / Impressions) * 100). Improve with compelling CTAs, eye-catching design, relevance.
9.  **Bounce Rate:** % of users leaving after viewing one page/screen without further action. (Bounce Rate = (Single-interaction Sessions / Total Sessions) * 100). Improve by matching expectations, visual appeal, internal linking, error handling.
10. **Average Session Duration:** Time user actively engages. (Avg Session Duration = Total Time Spent / Total Sessions). Improve with content depth/value, storytelling, interaction, content series.

### Part 5: Feature Flag-Driven Development (Eppo)

A software development technique where new features are wrapped in conditional code blocks (feature flags), allowing them to be turned on/off without redeployment.

**5.1. Core Elements:**
*   **Feature Flags:** Boolean (true/false) toggles in code. Off = feature hidden; On = feature active.
*   **Flag Management Systems (e.g., Eppo):** Dashboards for creating, organizing, tracking flags; targeting rules (location, subscription, device); monitoring impact.
*   **Deployment Pipeline Integration (CI/CD):**
    *   *CI:* Automated testing occurs even for features behind inactive flags.
    *   *CD:* Safe to push updates to production; new code is inactive until flag is turned on.
    *   *Controlled Release:* Activate features via flag management system, no redeployment needed.

**5.2. Benefits:**
*   **Incremental Rollouts (Progressive Delivery):** Release to small user percentages, monitor, gather feedback, then expand. Finds issues early, measures real-world impact.
*   **Risk Mitigation:** Flags act as "kill switches." Disable problematic features instantly without code rollback. Enables confident, frequent code releases.
*   **A/B Testing:** Easily test variations of features with specific user segments by associating flag states with experiment variants.

**5.3. Implementation Steps:**
1.  **Setting Up Feature Flags:** Choose a system (e.g., Eppo), wrap features in if-else logic checking flag status. Start simple.
2.  **Integrating with CI/CD Pipelines:** Automate flag creation, define deployment strategies (e.g., auto-remove old flags), establish review/audit processes.
3.  **Managing Feature Flag Lifecycles:** Document flags (purpose, creator, usage), plan for flag retirement to avoid technical debt.

**5.4. Best Practices:**
1.  **Clear Naming Conventions:** E.g., `feature_name_status_date` (e.g., `new_search_testing_20240509`).
2.  **Monitor Metrics to Measure Impact:** Track performance, user behavior, A/B test results for flagged features.
3.  **Keep Stakeholders Informed:** Developers, PMs, Marketing/Sales.

**5.5. Common Mistakes:**
1.  **Technical Debt from Outdated Flags:** Makes codebase harder to maintain, can cause unexpected interactions. Regularly audit and remove old flags.
2.  **Complexity of Management:** Testing various flag combinations, tracking purpose/ownership. Use centralized systems, document thoroughly, consider a global "kill switch."

### Part 6: Product Usage - Metrics, Analysis, and Strategies (Eppo)

Understanding how customers interact with a product.

**6.1. Key Metrics (some overlap with A/B testing metrics, viewed through product usage lens):**
*   **Daily Active Users (DAU):** Unique users interacting daily.
*   **Monthly Active Users (MAU):** Unique users interacting monthly.
*   **Session Duration:** Average time per session.
*   **Feature Usage:** How often specific features are used.
*   **Retention Rate:** % of users continuing product use over time.
*   **Churn Rate:** % of users stopping product use over time (opposite of retention).
*   **Engagement Rate:** How actively users interact (likes, comments, shares, tasks completed).

**6.2. Methods for Analyzing Product Usage:**
*   **User Analytics Tools (Google Analytics, Mixpanel, Amplitude):** Track actions, patterns, trends.
*   **Heatmaps (Hotjar):** Visualize clicks, scrolls, hovers.
*   **User Surveys and Feedback:** Gather qualitative data.
*   **A/B Testing (Eppo):** Compare versions of features/pages.

**6.3. Examples of Product Usage Analysis:**
*   **E-commerce:** Track user journey to identify friction in purchase funnel (e.g., high cart abandonment at shipping step).
*   **Mobile App:** Monitor feature usage (popular vs. underused) and session duration to improve flow.
*   **Content Platform:** Track content engagement (views, likes, shares) to tailor content strategy.

**6.4. Strategies to Grow Product Usage:**
*   **Focus on Onboarding Experience:** Guide users to "aha moment" with interactive walkthroughs, checklists.
*   **Prioritize User Experience (UX):** Intuitive navigation, clear instructions, appealing design, fast load times.
*   **Personalize User Experience:** Segment users, deliver custom recommendations/content.
*   **Continuous Feature Updates & Improvements:** Based on usage data and feedback.

### Part 7: Machine Learning - Continual Learning and Test in Production (OCR'd Text "Designing Machine Learning Systems")

**7.1. Continual Learning:**
*   **Definition:** Adapting ML models to data distribution shifts by continually updating them. Not necessarily with every sample, but often in micro-batches (e.g., every 512/1024 examples). The updated model (challenger) is evaluated against the existing one (champion) before deployment.
*   **Stateless Retraining vs. Stateful Training:**
    *   *Stateless:* Model trained from scratch each time. Requires more data, compute.
    *   *Stateful (Fine-tuning/Incremental Learning):* Model continues training on new data from a previous checkpoint. Requires less data, faster convergence (Grubhub: 45x less compute, 20% purchase-through rate increase). Can potentially avoid storing old data, aiding privacy.
    *   Stateful training is mostly for *data iteration* (refreshing model with new data, same architecture). *Model iteration* (new features/architecture) often still requires training from scratch, though research explores knowledge transfer/model surgery.
*   **Why Continual Learning?**
    *   Combat sudden data distribution shifts (e.g., ride-sharing price model during an unexpected event).
    *   Adapt to rare events (e.g., Black Friday, Singles Day).
    *   Overcome continuous cold start problem (new users, existing users switching devices, infrequent users with outdated historical data). TikTok adapts to users within minutes.
*   **Challenges:**
    1.  **Fresh Data Access:** Pulling new data quickly. Data warehouses can be slow. Real-time transports (Kafka, Kinesis) are faster. Labeling speed is often the bottleneck; natural labels with short feedback loops are best (dynamic pricing, CTR prediction). Stream processing for label computation is faster than batch. Programmatic labeling (Snorkel) or crowdsourcing can help.
    2.  **Evaluation Challenge:** Ensuring updates are good enough. Catastrophic failures amplify with frequent updates. Models are more susceptible to adversarial attacks. Thorough offline (Chapter 6 of source) and online (test in production) evaluation is crucial. Evaluation itself takes time.
    3.  **Algorithm Challenge:** Affects matrix-based (e.g., collaborative filtering) and tree-based models more for very fast updates. Neural networks adapt more easily. Feature extraction for partial datasets needs online computation of statistics (min, max, variance).

*   **Four Stages of Continual Learning Adoption:**
    1.  **Manual, Stateless Retraining:** Ad-hoc updates when performance degrades significantly. Common for new ML teams.
    2.  **Automated Retraining:** Scripts automate retraining (pull data, featurize, train, eval, deploy) periodically (e.g., daily via Spark). Requires schedulers (Airflow, Argo), accessible data, model store (S3, SageMaker, MLflow). "Log and wait" for feature reuse.
    3.  **Automated, Stateful Training:** Reconfigure scripts to load previous checkpoint and continue training. Requires mindset shift and robust data/model lineage tracking.
    4.  **Continual Learning (Trigger-based):** Updates triggered by time, performance drops, data volume, or drift detection, not fixed schedules. Requires solid monitoring and evaluation pipelines. Holy grail: combine with edge deployment.

**7.2. How Often to Update Your Models?**
*   Depends on the value of data freshness.
*   Experiment: Train models on data from different past time windows and evaluate on current data. If a model trained a quarter ago is much worse than one trained a month ago, update more frequently. Facebook found daily retraining for ad CTR reduced loss by 1% over weekly.

**7.3. Test in Production:**
Offline evaluation (test splits, backtests on recent data) isn't enough due to shifting distributions.
*   **Shadow Deployment:**
    1.  Deploy candidate model in parallel with existing model.
    2.  Route incoming requests to both for predictions.
    3.  Serve only existing model's prediction to the user.
    4.  Log new model's predictions for analysis.
    5.  Replace if satisfactory. Safe, but doubles inference cost.
*   **A/B Testing for Models:**
    1.  Deploy candidate alongside existing model.
    2.  Route a percentage of traffic randomly to the new model, rest to existing.
    3.  Monitor and analyze predictions/user feedback to determine statistically significant performance differences.
    *   Requires true randomization and sufficient sample size.
    *   Statistical significance isn't foolproof (p-value interpretation).
*   **Canary Release for Models:**
    1.  Deploy candidate (canary) alongside existing model.
    2.  Route a small portion of traffic to canary.
    3.  If performance satisfactory, increase traffic; otherwise, abort and revert.
    4.  Stop when canary serves all traffic or is aborted.
    *   Can be done without A/B testing (e.g., roll out to less critical market first).
    *   Netflix/Google use automated canary analysis (see Kayenta).
*   **Interleaving Experiments (for Recommender Systems):** (Covered in Netflix section, relevant here too)
    *   Expose users to recommendations from both models A and B in a blended list.
    *   Measure which model's recommendations users click on more.
    *   Netflix found it "reliably identifies best algorithms with considerably smaller sample size."
    *   Team-draft interleaving ensures fairness against position bias.
*   **Bandits:**
    *   Balance exploitation (choosing best-performing model so far) and exploration (trying other models).
    *   Stateful: Requires online predictions, short feedback loops, mechanism to track performance and route traffic.
    *   More data-efficient than A/B testing.
    *   Algorithms: ε-greedy, Thompson Sampling, Upper Confidence Bound (UCB).
    *   *Contextual Bandits:* For determining payout of each *action* (e.g., which item to recommend). Partial feedback problem. Balances showing liked items vs. items needing feedback.
*   **Evaluation Pipeline:** Good evaluation is not just *what* tests to run, but *who* runs them and *how*. Data scientists developing models often evaluate ad-hoc, introducing bias. Clear, automated pipelines with defined thresholds are needed.

### Part 8: Statistical Considerations and Pitfalls in A/B Testing (Mode Analytics - Julia Glick)

**8.1. The Purpose of A/B Testing:**
To learn about the *treatment effect* – the impact of an intervention on key metrics compared to a control/baseline. The primary job is to make *predictions* about future outcomes based on decisions.

**8.2. The Problem with Typical A/B Tests (Null Hypothesis Significance Testing - NHST):**
*   NHST estimates the difference and its uncertainty, yielding a test statistic (e.g., t-statistic). If it falls in a low-probability region under the null hypothesis (H0: no difference), H0 is rejected.
*   *Statistical Power:* Probability of correctly rejecting H0 when an effect exists. Depends on true effect size and uncertainty (sample size). Underpowered tests are common.
*   **The Statistical Significance Filter:**
    1.  Ask "is it real?" using NHST (e.g., p < 0.05).
    2.  If not "real," effect is considered zero for decisions.
    3.  If "real," use the observed effect size to guide decisions.
    *   **Problem:** We're now looking at an effect size *conditional* on statistical significance, which is no longer an unbiased estimator.
    *   **Result:** Effect size estimates become inflated (too extreme, further from zero). More pronounced in underpowered tests. This can lead to overly optimistic forecasts or chasing illusory lifts.

**8.3. How to Get Less Inflated Results:**
*   **Regularization:** Pulls estimates toward a central point (usually zero), introducing bias but reducing variance (bias-variance tradeoff). Addresses overfitting caused by the significance filter.
*   **A Bayesian Solution (Partial Pooling):**
    *   Assume each A/B test's true effect size is drawn from a common distribution (e.g., Gaussian or fatter-tailed like t-distribution, centered at 0). This is the *prior*.
    *   The observed test result is the true effect plus measurement error.
    *   Combine prior information (from past related tests) with current test data to get a *posterior* estimate.
    *   **Effect:**
        *   Highly powered tests (low standard error): Data overwhelms the prior; results shrink little.
        *   Badly underpowered tests (high standard error): Regularized heavily toward zero; "big wins" (often noise) diminish.
    *   Example: A test with 0.015 lift and 0.005 SE (t=3) might be regularized to 0.0087 (t=1.7) if past similar tests showed smaller effects.
    *   Requires buy-in, as it makes lifts look smaller.

**8.4. If Not Yet Open to Bayesian Models:**
1.  **Avoid Violating NHST/OLS Assumptions:** Pay as much attention to non-significant results as significant ones to avoid the filter. Difficult in practice. Looking only at effect sizes/CIs without significance testing can help, but checking if CI excludes zero is backdoor significance testing.
2.  **Design and Run High-Power Experiments:**
    *   Use past tests to inform plausible effect sizes for power analysis.
    *   High power (e.g., 90%) means the true distribution area where H0 isn't rejected is small, so conditional estimates are less inflated.
    *   Often hard to achieve due to costs, duration, or small true effects.
    *   Powering tests for *wished-for* lifts instead of *probable* lifts is a recipe for being led astray.
3.  **Do Informal Regularization (Mental Regularization):**
    *   Use domain knowledge and understanding of causal mechanisms.
    *   If a result is implausibly large or unexpected (e.g., retention lift without engagement lift for a UI change), mentally shrink it or investigate further.
    *   Be cautious; this injects human judgment, but we do it anyway. Formal regularization can be seen as an application of this.

**Conclusion:** A/B tests are the beginning, not the end, of solving a problem. Reflecting on why past decisions were made, whether using unbiased OLS or Bayesian pooling, improves learning and business outcomes.

### Part 9: Eppo's Role in the Ecosystem

Throughout these diverse topics, Eppo positions itself as a comprehensive, warehouse-native experimentation and feature management platform designed for rigor, precision, and trustworthiness.

*   **Warehouse-Native:** Integrates directly with data warehouses (Snowflake, BigQuery, Databricks, Redshift), ensuring data is from the internal source of truth, leading to accurate metrics (especially revenue, profit margins) and reducing setup for metric calculation.
*   **End-to-End Platform:** Covers experiment planning (hypotheses, sample size, metric alignment), design (feature flagging, advanced allocations, layers, holdouts), execution (SDKs for serving variations, diagnostics, real-time metrics), and analysis (robust statistical engine, CUPED++, sequential testing, Bayesian options, segment exploration, result sharing).
*   **Statistical Rigor:** Emphasizes trustworthy results, helping avoid pitfalls like the statistical significance filter by providing robust analysis and options like Bayesian methods or advanced variance reduction (CUPED++ for faster, more precise results).
*   **Advanced Methodologies Support:** Facilitates complex designs like Clustered Experiments and A/B/n testing.
*   **Feature Flagging:** Integrated for controlled rollouts, A/B testing of features, and risk mitigation.
*   **Democratization & Collaboration:** User-friendly interface for both technical and non-technical users, fostering a data-driven culture.

By combining these diverse perspectives, organizations can build a mature, reliable, and impactful experimentation and product development culture. The emphasis across all sources is on data-driven decisions, rigorous methodology, understanding statistical nuances, and leveraging technology to manage complexity and accelerate learning.