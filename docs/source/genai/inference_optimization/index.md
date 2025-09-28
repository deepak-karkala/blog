# Inference Optimization

```{toctree}
:hidden:

lower_precision
inference_optimization
flash_attention
kv_cache
speculative_decoding
deepspeed
```

- Two of the main challenges with inference include latency and cost. 
	- Large-scale models are extremely computationally expensive and often too slow to respond in many practical scenarios.
	- Moreover, these models with tens or hundreds of billions of parameters, trained with aggregated memory from multiple GPUs, simply become too large to fit on a single GPU’s device memory for inference.
		- For example, a single NVIDIA V100 Tensor Core GPU with 32 GB of memory can only fit up to a 10-billion-parameter model for inference, and the latency is limited by single GPU performance





### General techniques for Optimizing LLM inference

###### [Quantization](lower_precision.md)
- Activations and weights are compressed to use a smaller number of bits.
- bitsandbytes is a quantization library that includes support for 4-bit and 8-bit quantization. Quantization reduces your model size compared to its native full precision version, making it easier to fit large models onto GPUs with limited memory.

- Inference on Google Colab’s free tier GPUs
	- [T5: 11B](https://colab.research.google.com/drive/1YORPWx4okIHXnjW7MSAidXN29mPVNT7F?usp=sharing)
	- [Bloom: 3B](https://colab.research.google.com/drive/1qOjXfQIAULfKvZqwCen8-MoWKGdSatZ4?usp=sharing)


- Tips
	 - quantization should be implemented with caution. Naive quantization techniques can lead to a substantial degradation in model quality.

	- When experimenting with techniques like quantization, we recommend using an LLM quality benchmark to evaluate the quality of the inference system, not just the quality of the model in isolation.

	- Additionally, it's important to explore deeper systems optimizations. In particular, quantization can make KV caches much more efficient.

	- token generation with LLMs at low batch sizes is a GPU memory bandwidth-bound problem, i.e. the speed of generation depends on how quickly model parameters can be moved from the GPU memory to on-chip caches. Converting model weights from FP16 (2 bytes) to INT8 (1 byte) or INT4 (0.5 byte) requires moving less data and thus speeds up token generation. However, quantization may negatively impact the model generation quality.




###### [Flash Attention](flash_attention.md)
- FlashAttention-2 is a faster and more efficient implementation of the standard attention mechanism that can significantly speedup inference by:
	- additionally parallelizing the attention computation over sequence length
	- partitioning the work between GPU threads to reduce communication and shared memory reads/writes between them

- <i>FlashAttention is more memory efficient, meaning you can train on much larger sequence lengths without running into out-of-memory issues. You can potentially reduce memory usage up to 20x for larger sequence lengths.</i> 


###### [KV Cache, Multi-Query-Attention, Grouped-Query-Attention](kv_cache.md)
- The Attention mechanism in decoder-only Transformer-based models is computationally inefficient. Each token attends to all previously seen tokens, and thus recomputes many of the same values as each new token is generated.

- For example, while generating the Nth token, the (N-1)th token attends to (N-2)th, (N-3)th … 1st tokens. Similarly, while generating (N+1)th token, attention for the Nth token again needs to look at the (N-1)th, (N-2)th, (N-3)th, … 1st tokens. KV caching, i.e., saving of intermediate keys/values for the attention layers, is used to preserve those results for later reuse, avoiding repeated computation.

- The size of the KV cache varies based on the number of sequences processed at a time and the length of these sequences. 
	- <code>KV cache size  = 
batch_size * seqlen * (d_model/n_heads) * n_layers * 2 (K and V) * 2 (bytes per Float16) * n_kv_heads</code>

- <b>KV Cache and Quantization</b>

<div class="container py-4 py-md-5 px-4 px-md-3 text-body-secondary">
  <div class="row" >
    <div class="col-lg-4 mb-4">
      <img src="../../_static/genai/inference_optimization/kvcache_quantization.png"></img>
    </div>
  </div>
</div>

Source: - [Databricks: Oct 2023: LLM Inference Performance Engineering: Best Practices](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices)

- The figure above shows GQA KV cache size calculated at different batch sizes at a sequence length of 1024 tokens. The parameter size for Llama2 models, in comparison, is 140 GB (Float16) for the 70B model. Quantization of KV cache is another technique (in addition to GQA/MQA) to reduce the size of KV cache



###### Compression:
- Sparsity or Distillation

###### Parallelization
- Tensor parallelism across multiple devices or pipeline parallelism for larger models.

###### PyTorch scaled dot product attention
- PyTorch’s torch.nn.functional.scaled_dot_product_attention (SDPA) can also call FlashAttention and memory-efficient attention kernels under the hood. SDPA support is currently being added natively in Transformers and is used by default for torch>=2.1.1 when an implementation is available.
	- You may also set attn_implementation="sdpa" in from_pretrained() to explicitly request SDPA to be used.

###### BetterTransformer
- BetterTransformer accelerates inference with its fastpath (native PyTorch specialized implementation of Transformer functions) execution. The two optimizations in the fastpath execution are:
	- fusion, which combines multiple sequential operations into a single “kernel” to reduce the number of computation steps
	- skipping the inherent sparsity of padding tokens to avoid unnecessary computation with nested tensors

- BetterTransformer also converts all attention operations to use the more memory-efficient scaled dot product attention (SDPA), and it calls optimized kernels like FlashAttention under the hood.

- <code>model = model.to_bettertransformer()</code>




### Important Metrics for LLM Serving
- <b>Time To First Token (TTFT)</b>:
	- How quickly users start seeing the model's output after entering their query.
	- Low waiting times for a response are essential in real-time interactions, but less important in offline workloads.
	- This metric is driven by the time required to process the prompt and then generate the first output token.

- <b>Time Per Output Token (TPOT)</b>
	- Time to generate an output token for each user that is querying our system.
	- This metric corresponds with how each user will perceive the "speed" of the model.
	- For example, a TPOT of 100 milliseconds/tok would be 10 tokens per second per user, or \~450 words per minute, which is faster than a typical person can read.

- <b>Latency</b>
	- The overall time it takes for the model to generate the full response for a user. Overall response latency can be calculated using the previous two metrics: latency = (TTFT) + (TPOT) * (the number of tokens to be generated)

- <b>Throughput</b>
	- The number of output tokens per second an inference server can generate across all users and requests.


- there is a tradeoff between throughput and time per output token: if we process 16 user queries concurrently, we'll have higher throughput compared to running the queries sequentially, but we'll take longer to generate output tokens for each user.



### Inference latency targets: heuristics for evaluating models
- <b>Output length dominates overall response latency</b>:
	- For average latency, you can usually just take your expected/max output token length and multiply it by an overall average time per output token for the model.

- <b>Input length is not significant for performance but important for hardware requirements</b>:
	- The addition of 512 input tokens increases latency less than the production of 8 additional output tokens in the MPT models. However, the need to support long inputs can make models harder to serve. For example, we recommend using the A100-80GB (or newer) to serve MPT-7B with its maximum context length of 2048 tokens.

- <b>Overall latency scales sub-linearly with model size</b>:
	- On the same hardware, larger models are slower, but the speed ratio won't necessarily match the parameter count ratio. MPT-30B latency is ~2.5x that of MPT-7B latency. Llama2-70B latency is ~2x that of Llama2-13B latency.

> Takeaway: before you anchor yourself to specific latency targets ("we need less than 20 ms per token"), you should spend some time characterizing your expected input and desired output lengths.



### Challenges in LLM Inference

###### Memory Bandwidth is Key
- Computations in LLMs are mainly dominated by matrix-matrix multiplication operations; these operations with small dimensions are typically memory-bandwidth-bound on most hardware.
	- When generating tokens in an autoregressive manner, one of the activation matrix dimensions (defined by batch size and number of tokens in the sequence) is small at small batch sizes. Therefore, the speed is dependent on how quickly we can load model parameters from GPU memory to local caches/registers, rather than how quickly we can compute on loaded data.
	- <i>Available and achieved memory bandwidth in inference hardware is a better predictor of speed of token generation than their peak compute performance.</i>

- Inference hardware utilization is very important in terms of serving costs. GPUs are expensive and we need them to do as much work as possible. 
	- <i>Shared inference services promise to keep costs low by combining workloads from many users, filling in individual gaps and batching together overlapping requests.</i>
	- <i>For large models like Llama2-70B, we only achieve good cost/performance at large batch sizes. Having an inference serving system that can operate at large batch sizes is critical for cost efficiency.</i>
	- <i>However, a large batch means larger KV cache size, and that in turn increases the number of GPUs required to serve the model.</i>
	- There's a tug-of-war here and shared service operators need to make some cost trade-offs and implement systems optimizations.


###### Model Bandwidth Utilization (MBU)
- Memory bandwidth dictates how quickly the data movement happens. To measure the underlying hardware's utilization, we introduce a new metric called Model Bandwidth Utilization (MBU).
	- MBU is defined as (achieved memory bandwidth) / (peak memory bandwidth)
		- where achieved memory bandwidth is ((total model parameter size + KV cache size) / TPOT).

- For example, if a 7B parameter running with 16-bit precision has TPOT equal to 14ms, then it's moving 14GB of parameters in 14ms translating to 1TB/sec bandwidth usage. If the peak bandwidth of the machine is 2TB/sec, we are running at an MBU of 50%.
	- For simplicity, this example ignores KV cache size, which is small for smaller batch sizes and shorter sequence lengths.
	- MBU values close to 100% imply that the inference system is effectively utilizing the available memory bandwidth.
	- <i>MBU is also useful to compare different inference systems (hardware + software) in a normalized manner. MBU is complementary to the Model Flops Utilization (MFU; introduced in the [PaLM paper](https://arxiv.org/abs/2204.02311)) metric which is important in compute-bound settings</i>


- <b>Memory vs Compute Bound</b>
	- In reality for low batch sizes (white dot), the observed performance is lower than maximum – how much lower is a measure of the MBU. For large batch sizes (yellow region), the system is compute bound, and the achieved throughput as a fraction of the peak possible throughput is measured as the Model Flops Utilization (MFU).

<div class="container py-4 py-md-5 px-4 px-md-3 text-body-secondary">
  <div class="row" >
    <div class="col-lg-4 mb-4">
      <img src="../../_static/genai/inference_optimization/memory_compute_bound.png"></img>
    </div>
  </div>
</div>


- <b>MBU and MFU determine how much more room is available to push the inference speed further on a given hardware setup</b>
	- measured MBU for different degrees of tensor parallelism with our TensorRT-LLM-based inference server.
	- Peak memory bandwidth utilization is attained when transferring large contiguous memory chunks. When smaller models like MPT-7B are distributed across multiple GPUs, we observe lower MBU as we are moving smaller memory chunks in each GPU.


<div class="container py-4 py-md-5 px-4 px-md-3 text-body-secondary">
  <div class="row" >
    <div class="col-lg-4 mb-4">
      <img src="../../_static/genai/inference_optimization/mbu_tensor_parallelism.png"></img>
    </div>
  </div>
</div>

- <b>MBU decreases as batch size increases.</b>
	- empirically observed MBU for different degrees of tensor parallelism and batch sizes on the NVIDIA H100 GPUs.
	- MBU decreases as batch size increases. However, as we scale GPUs, the relative decrease in MBU is less significant.
	- <i>It is also worthy to note that picking hardware with greater memory bandwidth can boost performance with fewer GPUs</i>. At batch size 1, we can achieve a higher MBU of 60% on 2xH100-80GBs as compared to 55% on 4xA100-40GB GPUs.

<div class="container py-4 py-md-5 px-4 px-md-3 text-body-secondary">
  <div class="row" >
    <div class="col-lg-4 mb-4">
      <img src="../../_static/genai/inference_optimization/mbu_batch_size.png"></img>
    </div>
  </div>
</div>


### Benchmarking Results

###### Latency
- <b>Sclaing to more GPUs: Impact of Tensor Parallelism on latency</b>
	- <i>As input prompts lengthen, time to generate the first token starts to consume a substantial portion of total latency. Tensor parallelizing across multiple GPUs helps reduce this latency.</i>
	
	- Unlike model training, scaling to more GPUs offers significant diminishing returns for inference latency. Eg. for Llama2-70B going from 4x to 8x GPUs only decreases latency by 0.7x at small batch sizes. One reason for this is that higher parallelism has lower MBU (as discussed earlier). Another reason is that tensor parallelism introduces communication overhead across a GPU node.

- <b>Batch size and latency</b>
	- At larger batch sizes, higher tensor parallelism leads to a more significant relative decrease in token latency. This goes in line with our earlier observation that the relative decrease in MBU is smaller at higher degrees of tensor parallelism for batch size 16 as compared to batch size 1.


- <b>Hardware</b>
	- We also compare GPU scaling across two different hardware. Because H100-80GB has 2.15x GPU memory bandwidth as compared to A100-40GB, we can see that latency is 36% lower at batch size 1 and 52% lower at batch size 16 for 4x systems.


###### Throughput
- We can trade off throughput and time per token by batching requests together. Grouping queries during GPU evaluation increases throughput compared to processing queries sequentially, but each query will take longer to complete (ignoring queueing effects).

- There are a few common techniques for batching inference requests:
	- <b>Static batching</b>: Client packs multiple prompts into requests and a response is returned after all sequences in the batch have been completed. Our inference servers support this but do not require it.
	
	- <b>Dynamic batching</b>: Prompts are batched together on the fly inside the server. Typically, this method performs worse than static batching but can get close to optimal if responses are short or of uniform length. Does not work well when requests have different parameters.
	
	- <b>Continuous batching</b>: The idea of batching requests together as they arrive was introduced in this excellent paper and is currently the SOTA method. Instead of waiting for all sequences in a batch to finish, it groups sequences together at the iteration level. It can achieve 10x-20x better throughput than dynamic batching.
		- [ORCA: A Distributed Serving System for Transformer-Based Generative Models](https://www.usenix.org/system/files/osdi22-yu.pdf)

<div class="container py-4 py-md-5 px-4 px-md-3 text-body-secondary">
  <div class="row" >
    <div class="col-lg-4 mb-4">
      <img src="../../_static/genai/inference_optimization/throughput_batching.png"></img>
    </div>
  </div>
</div>

Source: - [Databricks: Oct 2023: LLM Inference Performance Engineering: Best Practices](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices)


- Continuous batching is usually the best approach for shared services, but there are situations where the other two might be better. In low-QPS environments, dynamic batching can outperform continuous batching. It is sometimes easier to implement low-level GPU optimizations in a simpler batching framework. For offline batch inference workloads, static batching can avoid significant overhead and achieve better throughput.


###### Batch Size
- How well batching works is highly dependent on the request stream. But we can get an upper bound on its performance by benchmarking static batching with uniform requests.

<div class="container py-4 py-md-5 px-4 px-md-3 text-body-secondary">
  <div class="row" >
    <div class="col-lg-4 mb-4">
      <img src="../../_static/genai/inference_optimization/throughput_batch_size.png"></img>
    </div>
  </div>
</div>

Source: - [Databricks: Oct 2023: LLM Inference Performance Engineering: Best Practices](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices)


###### Latency Trade-Off
- Request latency increases with batch size. 
	- With one NVIDIA A100 GPU, for example, if we maximize throughput with a batch size of 64, latency increases by 4x while throughput increases by 14x.

	- Shared inference services typically pick a balanced batch size. Users hosting their own models should decide the appropriate latency/throughput trade-off for their applications.
		- In some applications, like chatbots, low latency for fast responses is the top priority.
		- In other applications, like batched processing of unstructured PDFs, we might want to sacrifice the latency to process an individual document to process all of them fast in parallel.

<div class="container py-4 py-md-5 px-4 px-md-3 text-body-secondary">
  <div class="row" >
    <div class="col-lg-4 mb-4">
      <img src="../../_static/genai/inference_optimization/throughput_latency.png"></img>
    </div>
  </div>
</div>

Source: - [Databricks: Oct 2023: LLM Inference Performance Engineering: Best Practices](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices)


> The figure above shows the throughput vs latency curve for the MPT-7B model. Each line on this curve is obtained by increasing the batch size from 1 to 256. This is useful in determining how large we can make the batch size, subject to different latency constraints. Recalling our roofline plot above, we find that these measurements are consistent with what we would expect. After a certain batch size, i.e., when we cross to the compute bound regime, every doubling of batch size just increases the latency without increasing throughput.

- <b>Understand low-level hardware details</b> 
	- When using parallelism, it's important to understand low-level hardware details. For instance, not all 8xA100 instances are the same across different clouds. Some servers have high bandwidth connections between all GPUs, others pair GPUs and have lower bandwidth connections between pairs. This could introduce bottlenecks, causing real-world performance to deviate significantly from the curves above.



### Inference Optimization: Key Recommendations

- <b>Identify your optimization target</b>: Do you care about interactive performance? Maximizing throughput? Minimizing cost? There are predictable trade-offs here.

- <b>Pay attention to the components of latency</b>: For interactive applications time-to-first-token drives how responsive your service will feel and time-per-output-token determines how fast it will feel.

- <b>Memory bandwidth is key</b>: Generating the first token is typically compute-bound, while subsequent decoding is memory-bound operation. Because LLM inference often operates in memory-bound settings, MBU is a useful metric to optimize for and can be used to compare the efficiency of inference systems.

- <b>Batching is critical</b>: Processing multiple requests concurrently is critical for achieving high throughput and for effectively utilizing expensive GPUs. For shared online services continuous batching is indispensable, whereas offline batch inference workloads can achieve high throughput with simpler batching techniques.

- <b>In depth optimizations</b>: Standard inference optimization techniques are important (eg. operator fusion, weight quantization) for LLMs but it's important to explore deeper systems optimizations, especially those which improve memory utilization. One example is KV cache quantization.

- <b>Hardware configurations</b>: The model type and expected workload should be used to decide deployment hardware. For instance, when scaling to multiple GPUs MBU falls much more rapidly for smaller models, such as MPT-7B, than it does for larger models, such as Llama2-70B. Performance also tends to scale sub-linearly with higher degrees of tensor parallelism. That said, a high degree of tensor parallelism might still make sense for smaller models if traffic is high or if users are willing to pay a premium for extra low latency.

- <b>Data Driven Decisions</b>: Understanding the theory is important, but we recommend always measuring end-to-end server performance. There are many reasons an inference deployment can perform worse than expected. MBU could be unexpectedly low because of software inefficiencies. Or differences in hardware between cloud providers could lead to surprises (we have observed a 2x latency difference between 8xA100 servers from two cloud providers).





### Heuristics
- <b> Token count</b>
	- Llama 2 tokenization is 19% longer than ChatGPT tokenization (but still has a much lower overall cost [Anyscale](https://www.anyscale.com/blog/llama-2-is-about-as-factually-accurate-as-gpt-4-for-summaries-and-is-30x-cheaper)
	-  Llama 2 required \~20% more tokens to train over the same amount of text as GPT-4. [HF](https://twitter.com/Thom_Wolf/status/1701206627859206450?s=20)






### References
- [HF Blog: Optimizing LLMs for Speed and Memory](https://huggingface.co/docs/transformers/main/en/llm_tutorial_optimization#optimizing-llms-for-speed-and-memory)
- [Databricks: Oct 2023: LLM Inference Performance Engineering: Best Practices](https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices)
- [HF Blog: GPU inference](https://huggingface.co/docs/transformers/main/en/perf_infer_gpu_one#pytorch-scaled-dot-product-attention)

