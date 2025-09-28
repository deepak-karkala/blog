# Business Challenge and Goals

##

### Business Challenge

Developing **Advanced Driver-Assistance Systems (ADAS)** for trucks and cars requires not just accurate models, but a **production-grade Data Engine** capable of continuously ingesting, curating, and learning from massive multi-modal sensor data.

* **Scale vs. Resources**: Each vehicle could generate **20–40 TB of data per day**, creating petabyte-scale challenges—but the team had to solve this with a small engineering staff and startup-level budgets.
* **Safety-Critical Domain**: Unlike e-commerce or IoT analytics, even a **single misclassification** in ADAS could result in real-world accidents. This demanded **99.9%+ reliability** across diverse conditions.
* **Long-Tail Edge Cases**: The majority of raw driving logs contained uninteresting data, but **<1% of scenarios** (e.g., emergency lane changes, night-time cut-ins, occluded pedestrians) were critical for safety and generalization.
* **Operationalization Gap**: Models could not remain research artifacts. They had to be productionized with **CI/CD, monitoring, retraining, and governance** in line with MLOps best practices.

The company needed a **data-centric MLOps solution** that could close the loop:
**Collect → Curate → Label → Train → Deploy → Monitor → Retrain.**

---

### Goals

The project’s overarching goals were to:

1. **Architect a Production-Grade ADAS Data Engine** on AWS for cars and trucks, enabling scalable ingestion, curation, labeling, training, and deployment.
2. **Enable Continuous Improvement** of perception and inference models via a closed-loop system inspired by Tesla’s “Operation Vacation” data engine.
3. **Operationalize MLOps Best Practices** for a small, cross-functional startup team (Product Manager, Data Engineer, ML/MLOps Engineer).
4. **Balance Cost, Latency, and Reliability** — optimizing AWS cloud pipelines for performance while staying within realistic startup cost constraints.

---

### Primary Business KPIs

These metrics directly measured **business value and safety outcomes**:

| KPI                                        | Description                                                                                         | Target Outcome                                                                           |
| ------------------------------------------ | --------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- |
| **Reduction in False Positives/Negatives** | % reduction in critical perception model errors (e.g., misclassified vehicles, missed pedestrians). | **20–25% reduction** after full pipeline deployment.                                     |
| **ADAS Feature Reliability**               | Frequency of disengagements or system overrides in assisted driving.                                | **15–20% fewer disengagements** in fleet tests.                                          |
| **Time-to-Model-Update (TTMU)**            | Time from discovering a new failure mode to deploying an updated model.                             | Reduced from **8–10 weeks → 2–3 weeks**.                                                 |
| **Fleet Safety Improvement**               | Incidents avoided due to perception/ADAS alerts.                                                    | Internal validation: **\~22% reduction in safety-critical failures** across test drives. |

---

### Secondary Engagement KPIs

These tracked **engineering efficiency and organizational maturity**:

| KPI                                  | Description                                                                   | Target Outcome                               |
| ------------------------------------ | ----------------------------------------------------------------------------- | -------------------------------------------- |
| **Data Pipeline Latency**            | Time from raw ingestion → curated dataset availability.                       | Under **24 hours per drive log**.            |
| **Model Training Throughput**        | Number of experiments completed per week.                                     | Increase from **\~2/week → \~8–10/week**.    |
| **CI/CD Automation Coverage**        | % of workflows (data, model, infra) automated via GitHub Actions + Terraform. | **>85% automated**.              |
| **Data Governance Compliance**       | Traceability of dataset → model → deployment (ISO 26262 readiness).           | Full lineage tracked in **MLflow + DVC**.    |
| **Cross-Functional Iteration Speed** | Average cycle time between ML, data engineering, and product validation.      | Reduced by **40%** through shared pipelines. |

---

