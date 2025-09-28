# ADAS: Data Engine

```{toctree}
:hidden:

ch0_business_challenge
ch1_ml_problem_framing
ch2_operational_strategy
ch3_pipelines_workflows
ch4_testing_strategy
ch6_data_ingestion_workflows
ch7_scene_understanding_data_mining
ch8_model_training
ch9_packaging_promotion
ch10_deployment_serving
ch11_monitoring_continual_learning
ch12_cost_lifecycle_compliance
ch13_reliability_capacity_maps
```

##
####
#####
###### [Business Challenge and Goals](ch0_business_challenge.md)
###### [ML Problem Framing](ch1_ml_problem_framing.md)
###### [Project Planning, Operational Strategy](ch2_operational_strategy.md)
###### [Workflows, Team, Roles](ch3_pipelines_workflows.md)
###### [End to End MLOPS Testing Strategy](ch4_testing_strategy.md)
<!--###### [Data Characteristics](ch5_data_characteristics.md)-->
###### [Data Ingestion Workflows](ch6_data_ingestion_workflows.md)
###### [Scene Understanding & Data Mining](ch7_scene_understanding_data_mining.md)
###### [Model Training & Experimentation](ch8_model_training.md)
###### [Packaging, Evaluation & Promotion](ch9_packaging_promotion.md)
###### [Deployment & Serving](ch10_deployment_serving.md)
###### [Monitoring & Continual Learning](ch11_monitoring_continual_learning.md)
###### [Cost, Lifecycle, Compliance](ch12_cost_lifecycle_compliance.md)
###### [Reliability, Capacity, Maps](ch13_reliability_capacity_maps.md)


* Inference
	- [PyTorch Quantization](https://pytorch.org/docs/stable/quantization.html)
	- https://hamel.dev/notes/serving/	ML Serving
	- **https://openai.com/index/triton/ : Introducing Triton: Open-source GPU programming for neural networks**

## TODO

### Scene Understanding & Data Mining
- OpenSearch Indexing, Semantic Search
	- How OpenSearch and GraphQL API works together
- Embedding models for images, text and LiDAR
- Vector DB indexing/search Algorithms: HNSW, IVF-PQ
- Details of Scenes/Triggers/Scenarios
	- Full list of cases
	- List of 3-5 challenging cases (multi-turn curate -> train -> deply -> monitor -> improve -> curate)

### Training
- **Use Ray for Distributed Training ?**
- Training
	- Task balancing for multi-head models (e.g., HydraNet-style): dynamic or curriculum weighting.
- Model Testing
	- Full list of test cases 
	- Slices
	- Regressions
- HPO
	- Baseline config (train.yaml) with search space: LR/WD, warmup, aug policy, loss weights, NMS/score thresholds, backbone/neck options, EMA on/off, AMP level, batch size/accum steps.
	- Sweep strategy: Bayesian, Hyperband/ASHA, Random, or Population-Based Training (PBT) for long runs.



