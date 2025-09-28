# Deploying LLMs

```{toctree}
:hidden:

```

### TODO
- PagedAttention
- FlashInfer
- inflight batching
- paged KV caching
- Continuous batching
- Fast model execution with CUDA/HIP graph
- Quantization
	- SmoothQuant
	- FP8
- Chunked prefill
- Prefix caching
- Multi-lora
- Token streaming using Server-Sent Events
- Safetensors weight loading
- Watermarking with A Watermark for Large Language Models
- Stop sequences
- speculative decoding algorithms
	- Medusa
	- SpecExec


### Scaling 


### Optimized inference engines
- Options
	- vLLM
	- TensorRT-LLM,
	- text-generation-inference and
	- lmdeploy


###### vLLM
- Easy, fast, and cheap LLM serving for everyone
- A high-throughput and memory-efficient inference and serving engine for LLMs

- vLLM is a fast and easy-to-use library for LLM inference and serving.

- vLLM is fast with:
 	- State-of-the-art serving throughput
	- Efficient management of attention key and value memory with PagedAttention
	- Continuous batching of incoming requests
	- Fast model execution with CUDA/HIP graph
	- Quantizations: GPTQ, AWQ, INT4, INT8, and FP8.
	- Optimized CUDA kernels, including integration with FlashAttention and FlashInfer.
	- Speculative decoding
	- Chunked prefill

- vLLM is flexible and easy to use with:
	- seamless integration with popular Hugging Face models
	- High-throughput serving with various decoding algorithms, including parallel sampling, beam search, and more
	- Tensor parallelism and pipeline parallelism support for distributed inference
	- Streaming outputs
	- OpenAI-compatible API server
	- Support NVIDIA GPUs, AMD CPUs and GPUs, Intel CPUs and GPUs, PowerPC CPUs, TPU, and AWS Neuron.
	- Prefix caching support
	- Multi-lora support

- vLLM seamlessly supports most popular open-source models on HuggingFace, including:
	- Transformer-like LLMs (e.g., Llama)
	- Mixture-of-Expert LLMs (e.g., Mixtral)
	- Embedding Models (e.g. E5-Mistral)
	- Multi-modal LLMs (e.g., LLaVA)


###### Text Generation Inference
- A Rust, Python and gRPC server for text generation inference. Used in production at Hugging Face to power Hugging Chat, the Inference API and Inference Endpoint.

- Text Generation Inference (TGI) is a toolkit for deploying and serving Large Language Models (LLMs). TGI enables high-performance text generation for the most popular open-source LLMs, including Llama, Falcon, StarCoder, BLOOM, GPT-NeoX, and more. 

- TGI implements many features, such as:
	- Simple launcher to serve most popular LLMs
	- Production ready (distributed tracing with Open Telemetry, Prometheus metrics)
	- Tensor Parallelism for faster inference on multiple GPUs
	- Token streaming using Server-Sent Events (SSE)
	- Continuous batching of incoming requests for increased total throughput
	- Messages API compatible with Open AI Chat Completion API
	- Optimized transformers code for inference using Flash Attention and Paged Attention on the most popular architectures
	- Quantization with :
		- bitsandbytes
		- GPT-Q
		- EETQ
		- AWQ
		- Marlin
		- fp8
	- Safetensors weight loading
	- Watermarking with A Watermark for Large Language Models
	- Logits warper (temperature scaling, top-p, top-k, repetition penalty)
	- Stop sequences
	- Log probabilities
	- Speculation \~2x latency
	- Guidance/JSON. Specify output format to speed up inference and make sure the output is valid according to some specs.
	- Custom Prompt Generation: Easily generate text by providing custom prompts to guide the model's output
	- Fine-tuning Support: Utilize fine-tuned models for specific tasks to achieve higher accuracy and performance

<div class="container py-4 py-md-5 px-4 px-md-3 text-body-secondary">
    <div class="row" >
      <div class="col-lg-4 mb-4">
        <img src="../../_static/genai/deploying_llms/tgi_internals_basic.png"></img>
      </div>
    </div>
</div>

- [HF: Text Generation Inference Architecture](https://github.com/huggingface/text-generation-inference)


###### TensorRT-LLM 
- TensorRT-LLM is a library for optimizing Large Language Model (LLM) inference. It provides state-of-the-art optimziations, including custom attention kernels, inflight batching, paged KV caching, quantization (FP8, INT4 AWQ, INT8 SmoothQuant, ++) and much more, to perform inference efficiently on NVIDIA GPUs

- TensorRT-LLM provides a Python API to build LLMs into optimized TensorRT engines. It contains runtimes in Python (bindings) and C++ to execute those TensorRT engines. It also includes a backend for integration with the NVIDIA Triton Inference Server. Models built with TensorRT-LLM can be executed on a wide range of configurations from a single GPU to multiple nodes with multiple GPUs (using Tensor Parallelism and/or Pipeline Parallelism).

- TensorRT-LLM comes with several popular models pre-defined. They can easily be modified and extended to fit custom needs via a PyTorch-like Python API. Refer to the Support Matrix for a list of supported models.

- TensorRT-LLM is built on top of the TensorRT Deep Learning Inference library. It leverages much of TensorRT's deep learning optimizations and adds LLM-specific optimizations on top, as described above. TensorRT is an ahead-of-time compiler; it builds "Engines" which are optimized representations of the compiled model containing the entire execution graph. These engines are optimized for a specific GPU architecture, and can be validated, benchmarked, and serialized for later deployment in a production environment.

- Tensor RT
	- NVIDIA® TensorRT™ is an SDK for optimizing trained deep-learning models to enable high-performance inference. TensorRT contains a deep learning inference optimizer and a runtime for execution.
	
	- NVIDIA® TensorRT™ is an ecosystem of APIs for high-performance deep learning inference. TensorRT includes an inference runtime and model optimizations that deliver low latency and high throughput for production applications. The TensorRT ecosystem includes TensorRT, TensorRT-LLM, TensorRT Model Optimizer, and TensorRT Cloud



### [Performance Benchmark](https://buildkite.com/vllm/performance-benchmark/builds/4068)
- We benchmark vllm, tensorrt-llm, lmdeploy and tgi using the following workload:
	- Input length: randomly sample 500 prompts from ShareGPT dataset (with fixed random seed).
	- Output length: the corresponding output length of these 500 prompts.
	- Models: llama-3 8B, llama-3 70B, mixtral 8x7B.
	- Average QPS (query per second): 4 for the small model (llama-3 8B) and 2 for other two models. For each QPS, the arrival time of each query is determined using a random Poisson process (with fixed random seed).
	- Evaluation metrics: Throughput (higher the better), TTFT (time to the first token, lower the better), ITL (inter-token latency, lower the better).


<div class="container py-4 py-md-5 px-4 px-md-3 text-body-secondary">
    <div class="row" >
      <div class="col-lg-4 mb-4">
        <img src="../../_static/genai/deploying_llms/benchmark.png"></img>
      </div>
    </div>
</div>

- [Inference engines Performance Benchmark](https://buildkite.com/vllm/performance-benchmark/builds/4068)



### Edge deployment

###### MLC LLM
- Universal LLM Deployment Engine with ML Compilation
- MLC LLM is a machine learning compiler and high-performance deployment engine for large language models. The mission of this project is to enable everyone to develop, optimize, and deploy AI models natively on everyone's platforms. 

- MLC LLM compiles and runs code on MLCEngine -- a unified high-performance LLM inference engine across the above platforms. MLCEngine provides OpenAI-compatible API available through REST server, python, javascript, iOS, Android, all backed by the same engine and compiler that we keep improving with the community.



### SkyPilot

- SkyPilot is a framework for running AI and batch workloads on any infra, offering unified execution, high cost savings, and high GPU availability.- 
- SkyPilot abstracts away infra burdens:- 

	- Launch dev clusters, jobs, and serving on any infra- 

	- Easy job management: queue, run, and auto-recover many jobs- 

- SkyPilot supports multiple clusters, clouds, and hardware (the Sky):- 

	- Bring your reserved GPUs, Kubernetes clusters, or 12+ clouds- 

	- Flexible provisioning of GPUs, TPUs, CPUs, with auto-retry- 

- SkyPilot cuts your cloud costs & maximizes GPU availability:- 

	- Autostop: automatic cleanup of idle resources- 

	- Managed Spot: 3-6x cost savings using spot instances, with preemption auto-recovery- 

	- Optimizer: 2x cost savings by auto-picking the cheapest & most available infra- 

- SkyPilot supports your existing GPU, TPU, and CPU workloads, with no code changes.- 

- Current supported infra (Kubernetes; AWS, GCP, Azure, OCI, Lambda Cloud, Fluidstack, RunPod, Cudo, Paperspace, Cloudflare, Samsung, IBM, VMware vSphere)


