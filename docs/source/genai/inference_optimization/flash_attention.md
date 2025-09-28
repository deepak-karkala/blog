# Flash Attention

### References
- [HF Blog: Optimizing LLMs for Speed and Memory](https://huggingface.co/docs/transformers/main/en/llm_tutorial_optimization#optimizing-llms-for-speed-and-memory)


### Why Flash Attention ? Answer: Attention layer is expensive for longer context length
- Self-attention layers are central to Large Language Models (LLMs) in that they enable the model to understand the contextual relationships between input tokens. However, the peak GPU memory consumption for self-attention layers grows quadratically both in compute and memory complexity with number of input tokens (also called sequence length) that we denote in the following by N . 

- While this is not really noticeable for shorter input sequences (of up to 1000 input tokens), it becomes a serious problem for longer input sequences (at around 16000 input tokens).

<div class="container py-4 py-md-5 px-4 px-md-3 text-body-secondary">
  <div class="row" >
    <div class="col-lg-4 mb-4">
      <img src="../../_static/genai/inference_optimization/self_attention_memory.png"></img>
    </div>
  </div>
</div>

Source: [HF Blog: Optimizing LLMs for Speed and Memory](https://huggingface.co/docs/transformers/main/en/llm_tutorial_optimization#optimizing-llms-for-speed-and-memory)

- Long story short, the default self-attention algorithm quickly becomes prohibitively memory-expensive for large input contexts.
	- As LLMs improve in text comprehension and generation, they are applied to increasingly complex tasks. While models once handled the translation or summarization of a few sentences, they now manage entire pages, demanding the capability to process extensive input lengths.


### How does it work ?
- How can we get rid of the exorbitant memory requirements for large input lengths? We need a new way to compute the self-attention mechanism that gets rid of the QK<sup>T</sup> matrix.
	- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)

> By keeping track of softmax normalization statistics and by using some smart mathematics, Flash Attention gives numerical identical outputs compared to the default self-attention layer at a memory cost that only increases linearly with N.

<div class="container py-4 py-md-5 px-4 px-md-3 text-body-secondary">
  <div class="row" >
    <div class="col-lg-4 mb-4">
      <img src="../../_static/genai/inference_optimization/flash_attention.png"></img>
    </div>
  </div>
</div>

Source: [HF Blog: Optimizing LLMs for Speed and Memory](https://huggingface.co/docs/transformers/main/en/llm_tutorial_optimization#optimizing-llms-for-speed-and-memory)


- Looking at the formula, one would intuitively say that Flash Attention must be much slower compared to the default self-attention formula as more computation needs to be done. Indeed Flash Attention requires more FLOPs compared to normal attention as the softmax normalization statistics have to constantly be recomputed.

> However, Flash Attention is much faster in inference compared to default attention which comes from its ability to significantly reduce the demands on the slower, high-bandwidth memory of the GPU (VRAM), focusing instead on the faster on-chip memory (SRAM).
> Essentially, Flash Attention makes sure that all intermediate write and read operations can be done using the fast on-chip SRAM memory instead of having to access the slower VRAM memory to compute the output vector O.

- <i>In practice, there is currently absolutely no reason to not use Flash Attention if available. The algorithm gives mathematically the same outputs, and is both faster and more memory-efficient.</i>


```python
model = AutoModelForCausalLM.from_pretrained("bigcode/octocoder", torch_dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("bigcode/octocoder")

# enable Flash Attention. To do so, we convert the model to BetterTransformer and by doing so enabling PyTorch’s SDPA self-attention which in turn is able to use Flash Attention.
model.to_bettertransformer()

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
```


```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM

model_id = "tiiuae/falcon-7b"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# load in 8bit
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_8bit=True,
    attn_implementation="flash_attention_2",
)

# load in 4bit
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    load_in_4bit=True,
    attn_implementation="flash_attention_2",
)
```

### [Highlights from Paper: FlashAttention: Fast and Memory-Efficient Exact Attention
with IO-Awareness](https://arxiv.org/pdf/2205.14135)

<div class="container py-4 py-md-5 px-4 px-md-3 text-body-secondary">
  <div class="row" >
    <div class="col-lg-4 mb-4">
      <img src="../../_static/genai/inference_optimization/flash_attention_paper1.png"></img>
    </div>
  </div>
</div>

<div class="container py-4 py-md-5 px-4 px-md-3 text-body-secondary">
  <div class="row" >
    <div class="col-lg-4 mb-4">
      <img src="../../_static/genai/inference_optimization/flash_attention_paper2.png"></img>
    </div>
  </div>
</div>

<div class="container py-4 py-md-5 px-4 px-md-3 text-body-secondary">
  <div class="row" >
    <div class="col-lg-4 mb-4">
      <img src="../../_static/genai/inference_optimization/flash_attention_paper3.png"></img>
    </div>
  </div>
</div>
Source: [FlashAttention: Fast and Memory-Efficient Exact Attention
with IO-Awareness](https://arxiv.org/pdf/2205.14135)

