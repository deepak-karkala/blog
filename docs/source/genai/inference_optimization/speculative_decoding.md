# Speculative decoding

### Why is text generation so slow?
- why are forward passes slow? Forward passes are typically dominated by matrix multiplications
- bottleneck in the forward pass comes from loading the model layer weights into the computation cores of your device, not from performing the computations themselves.

### 3 methods to optimize model forward pass
- <b>hardware-specific model optimizations</b>
	- Flash Attention
	- Quantization

- <b>Batching</b>
	- when you know you’ll get concurrent text generation requests, you can batch the inputs and massively increase the throughput with a small latency penalty.
	- The model layer weights loaded into the device are now used on several input rows in parallel, which means that you’ll get more tokens out for approximately the same memory bandwidth burden.
	- The catch with batching is that you need additional device memory (or to offload the memory somewhere)

- <b>Tensor Parallelism</b>
	- you can distribute the workload using Tensor Parallelism and obtain lower latency.
	- With Tensor Parallelism, you split the memory bandwidth burden across multiple devices, but you now have to consider inter-device communication bottlenecks in addition to the monetary cost of running multiple devices.
	- The benefits depend largely on the model size:
		- models that easily fit on a single consumer device see very limited benefits.
		- Taking the results from this [DeepSpeed blog post](https://www.microsoft.com/en-us/research/blog/deepspeed-accelerating-large-scale-model-inference-and-training-via-system-optimizations-and-compression/), you see that you can spread a 17B parameter model across 4 GPUs to reduce the latency by 1.5x


### Intuition behind Speculative decoding
- <b>Language decoder forward pass, revisited</b>
	- During text generation, the typical iteration consists in the model receiving as input the latest generated token, plus cached internal computations for all other previous inputs, returning the next token logits. Caching is used to avoid redundant computations, resulting in faster forward passes, but it’s not mandatory (and can be used partially).

	- <i>When caching is disabled, the input contains the entire sequence of tokens generated so far and the output contains the logits corresponding to the next token for all positions in the sequence!</i> The logits at position N correspond to the distribution for the next token if the input consisted of the first N tokens, ignoring all subsequent tokens in the sequence. <i>In the particular case of greedy decoding, if you pass the generated sequence as input and apply the argmax operator to the resulting logits, you will obtain the generated sequence back.</i>

> This means that you can use a model forward pass for a different purpose: in addition to feeding some tokens to predict the next one, you can also pass a sequence to the model and double-check whether the model would generate that same sequence (or part of it).


- Let’s consider for a second that you have access to a magical latency-free oracle model that generates the same sequence as your model, for any given input. For argument’s sake, it can’t be used directly, it’s limited to being an assistant to your generation procedure. Using the property described above, you could use this assistant model to get candidate output tokens followed by a forward pass with your model to confirm that they are indeed correct. In this utopian scenario, the latency of text generation would be reduced from O(n) to O(1), with n being the number of generated tokens. For long generations, we're talking about several orders of magnitude.

- Walking a step towards reality, let's assume the assistant model has lost its oracle properties. Now it’s a latency-free model that gets some of the candidate tokens wrong, according to your model. Due to the autoregressive nature of the task, as soon as the assistant gets a token wrong, all subsequent candidates must be invalidated. However, that does not prevent you from querying the assistant again, after correcting the wrong token with your model, and repeating this process iteratively. Even if the assistant fails a few tokens, text generation would have an order of magnitude less latency than in its original form.

- Obviously, there are no latency-free assistant models. Nevertheless, it is relatively easy to find a model that approximates some other model’s text generation outputs – smaller versions of the same architecture trained similarly often fit this property. Moreover, when the difference in model sizes becomes significant, the cost of using the smaller model as an assistant becomes an afterthought after factoring in the benefits of skipping a few forward passes.


### Greedy decoding with assisted generation
- Use greedy decoding to generate a certain number of candidate tokens with the assistant model, producing candidates. The number of produced candidate tokens is initialized to 5 the first time assisted generation is called.

- Using our model, do a forward pass with candidates, obtaining logits.

- Use the token selection method (.argmax() for greedy search or .multinomial() for sampling) to get the next_tokens from logits.

- Compare next_tokens to candidates and get the number of matching tokens. Remember that this comparison has to be done with left-to-right causality: after the first mismatch, all candidates are invalidated.

- Use the number of matches to slice things up and discard variables related to unconfirmed candidate tokens. In essence, in next_tokens, keep the matching tokens plus the first divergent token (which our model generates from a valid candidate subsequence).

- Adjust the number of candidate tokens to be produced in the next iteration — our original heuristic increases it by 2 if ALL tokens match and decreases it by 1 otherwise.


<div class="container py-4 py-md-5 px-4 px-md-3 text-body-secondary">
  <div class="row" >
    <div class="col-lg-4 mb-4">
      <video width="640" height="480" controls autoplay>
      	<source src="../../_static/genai/inference_optimization/assisted_generation.mp4" type="video/mp4">
      </video>
    </div>
  </div>
</div>


### Code
- pass the assistant model under the new assistant_model keyword argument and reap the latency gains
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

prompt = "Alice and Bob"
checkpoint = "EleutherAI/pythia-1.4b-deduped"
assistant_checkpoint = "EleutherAI/pythia-160m-deduped"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(checkpoint)
inputs = tokenizer(prompt, return_tensors="pt").to(device)

model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
assistant_model = AutoModelForCausalLM.from_pretrained(assistant_checkpoint).to(device)
outputs = model.generate(**inputs, assistant_model=assistant_model)
print(tokenizer.batch_decode(outputs, skip_special_tokens=True))
# ['Alice and Bob are sitting in a bar. Alice is drinking a beer and Bob is drinking a']
```


### Speed ups and takeaways

- Requires access to an assistant model that is at least an order of magnitude smaller than your model (the bigger the difference, the better);
- Gets up to 3x speedups in the presence of INT8 and up to 2x otherwise, when the model fits in the GPU memory;
- If you’re playing with models that do not fit in your GPU and are relying on memory offloading, you can see up to 10x speedups;
- Shines in input-grounded tasks, like automatic speech recognition or summarization.




### References
- [HF May 2023: Assisted Generation: a new direction toward low-latency text generation](https://huggingface.co/blog/assisted-generation)