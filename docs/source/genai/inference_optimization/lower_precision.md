# Quantization

- Reducing the precision of model weights and activations during inference can dramatically reduce hardware requirements. 
	- For instance, switching from 16-bit weights to 8-bit weights can halve the number of required GPUs in memory constrained environments (eg. Llama2-70B on A100s). Dropping down to 4-bit weights makes it possible to run inference on consumer hardware (eg. Llama2-70B on Macbooks).


### How Quantization works ?
- Quantization schemes aim at reducing the precision of weights while trying to keep the model’s inference results as accurate as possible (a.k.a as close as possible to bfloat16).

- Note that quantization works especially well for text generation since all we care about is choosing the set of most likely next tokens and don’t really care about the exact values of the next token logit distribution. All that matters is that the next token logit distribution stays roughly the same so that an argmax or topk operation gives the same results.

- There are various quantization techniques, in general, all quantization techniques work as follows:
	- Quantize all weights to the target precision
	- Load the quantized weights, and pass the input sequence of vectors in bfloat16 precision
	- Dynamically dequantize weights to bfloat16 to perform the computation with their input vectors in bfloat16 precision

- In a nutshell, <i>Dequantization and re-quantization is performed sequentially for all weight matrices as the inputs run through the network graph. Therefore, inference time is often not reduced when using quantized weights, but rather increases.</i>




### FP16, BF16
- Memory requirements of LLMs can be best understood by seeing the LLM as a set of weight matrices and vectors and the text inputs as a sequence of vectors. 

- Nowadays, models are however rarely trained in full float32 precision, but usually in bfloat16 precision or less frequently in float16 precision. 

- For shorter text inputs (less than 1024 tokens), the memory requirement for inference is very much dominated by the memory requirement to load the weights. Therefore, for now, let’s assume that the memory requirement for inference is equal to the memory requirement to load the model into the GPU VRAM.
	- GPT3 requires 2 * 175 GB = 350 GB VRAM
	- Bloom requires 2 * 176 GB = 352 GB VRAM
	- Llama-2-70b requires 2 * 70 GB = 140 GB VRAM
	- Falcon-40b requires 2 * 40 GB = 80 GB VRAM
	- MPT-30b requires 2 * 30 GB = 60 GB VRAM
	- bigcode/starcoder requires 2 * 15.5 = 31 GB VRAM


> 
>- Loading the weights of a model having X billion parameters requires roughly 4 X GB of VRAM in float32 precision. 
>- Loading the weights of a model having X billion parameters requires roughly 2 X GB of VRAM in bfloat16/float16 precision.


- largest GPU chip on the market is the A100 & H100 offering 80GB of VRAM. Most of the models listed before require more than 80GB just to be loaded and therefore necessarily require tensor parallelism and/or pipeline parallelism.


> Almost all models are trained in bfloat16 nowadays, there is no reason to run the model in full float32 precision if your GPU supports bfloat16. Float32 won’t give better inference results than the precision that was used to train the model.


### 8-bits, 4-bits
- It has been found that model weights can be quantized to 8-bit or 4-bits without a significant loss in performance [Dettmers et al.](https://arxiv.org/abs/2208.07339). Model can be quantized to even 3 or 2 bits with an acceptable loss in performance as shown in the recent [GPTQ](https://arxiv.org/abs/2210.17323) paper.

- AutoGPTQ
	- To see how one can quantize models to require even less GPU VRAM memory than 4-bit

### Example
- bigcode/octocoder: 31 GB VRAM in BF16

```python
!pip install bitsandbytes
model = AutoModelForCausalLM.from_pretrained("bigcode/octocoder", load_in_8bit=True, pad_token_id=0)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

result = pipe(prompt, max_new_tokens=60)[0]["generated_text"][len(prompt):]
result

def bytes_to_giga_bytes(bytes):
  return bytes / 1024 / 1024 / 1024
bytes_to_giga_bytes(torch.cuda.max_memory_allocated()) 
```

> OctoCoder in 8-bit precision reduced the required GPU VRAM from 32G GPU VRAM to only 15GB and running the model in 4-bit precision further reduces the required GPU VRAM to just a bit over 9GB. Inference slows down a bit compared to BF16 format, due to the aggressive quantization method.
> 
>4-bit quantization allows the model to be run on GPUs such as RTX3090, V100, and T4 which are quite accessible for most people.


> it is important to remember that model quantization trades improved memory efficiency against accuracy and in some cases inference time























