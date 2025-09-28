# DeepSpeed

### Features
- DeepSpeed rolls out high-performance inference support for large Transformer-based models with billions of parameters, like those at the scale of Turing-NLG 17B and Open AI GPT-3 175B. Our new technologies for optimizing inference cost and latency include:

	- <b>Inference-adapted parallelism</b> allows users to efficiently serve large models by adapting to the best parallelism strategies for multi-GPU inference, accounting for both inference latency and cost.

	- <b>Inference-optimized CUDA kernels</b> boost per-GPU efficiency by fully utilizing the GPU resources through deep fusion and novel kernel scheduling.

	- <b>Effective quantize-aware training</b> allows users to easily quantize models that can efficiently execute with low-precision, such as 8-bit integer (INT8) instead of 32-bit floating point (FP32), leading to both memory savings and latency reduction without hurting accuracy.

- Together, DeepSpeed Inference shows 1.9–4.4x latency speedups and 3.4–6.2x throughput gain and cost reduction when compared with existing work.





### References
- [May 2021: DeepSpeed: Accelerating large-scale model inference and training via system optimizations and compression](https://www.microsoft.com/en-us/research/blog/deepspeed-accelerating-large-scale-model-inference-and-training-via-system-optimizations-and-compression/)


