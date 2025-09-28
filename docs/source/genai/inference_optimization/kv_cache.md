# KV cache, MQA, GQA

### KV cache

- Auto-regressive text generation with LLMs works by iteratively putting in an input sequence, sampling the next token, appending the next token to the input sequence, and continuing to do so until the LLM produces a token that signifies that the generation has finished.

- Let’s run a quick code snippet to show how auto-regressive works in practice. We will simply take the most likely next token via torch.argmax

```python
input_ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda")

for _ in range(5):
  next_logits = model(input_ids)["logits"][:, -1:]
  next_token_id = torch.argmax(next_logits,dim=-1)

  input_ids = torch.cat([input_ids, next_token_id], dim=-1)
  print("shape of input_ids", input_ids.shape)

generated_text = tokenizer.batch_decode(input_ids[:, -5:])
generated_text
```

```code
shape of input_ids torch.Size([1, 21])
shape of input_ids torch.Size([1, 22])
shape of input_ids torch.Size([1, 23])
shape of input_ids torch.Size([1, 24])
shape of input_ids torch.Size([1, 25])
[' Here is a Python function']
```

- As we can see every time we increase the text input tokens by the just sampled token. With very few exceptions, LLMs are trained using the causal language modeling objective and therefore mask the upper triangle matrix of the attention score.

- <i>In order to reduce unnecessary computation, one can therefore cache each layer’s key-value vectors for all previous timesteps.</i>

- In the following, we will tell the LLM to make use of the key-value cache by retrieving and forwarding it for each forward pass. In Transformers, we can retrieve the key-value cache by passing the use_cache flag to the forward call and can then pass it with the current token.

```python
past_key_values = None # past_key_values is the key-value cache
generated_tokens = []
next_token_id = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda")

for _ in range(5):
  next_logits, past_key_values = model(next_token_id, past_key_values=past_key_values, use_cache=True).to_tuple()
  next_logits = next_logits[:, -1:]
  next_token_id = torch.argmax(next_logits, dim=-1)

  print("shape of input_ids", next_token_id.shape)
  print("length of key-value cache", len(past_key_values[0][0]))  # past_key_values are of shape [num_layers, 0 for k, 1 for v, batch_size, length, hidden_dim]
  generated_tokens.append(next_token_id.item())

generated_text = tokenizer.batch_decode(generated_tokens)
generated_text
```

```code
shape of input_ids torch.Size([1, 1])
length of key-value cache 20
shape of input_ids torch.Size([1, 1])
length of key-value cache 21
shape of input_ids torch.Size([1, 1])
length of key-value cache 22
shape of input_ids torch.Size([1, 1])
length of key-value cache 23
shape of input_ids torch.Size([1, 1])
length of key-value cache 24
[' Here', ' is', ' a', ' Python', ' function']
```

- <i>As one can see, when using the key-value cache the text input tokens are not increased in length, but remain a single input vector. The length of the key-value cache on the other hand is increased by one at every decoding step.</i>

> Making use of the key-value cache means that the QK<sup>T</sup> is essentially reduced to q<sub>c</sub>K<sup>T</sup> q<sub>c</sub> being the query projection of the currently passed input token which is always just a single vector.


- Using the key-value cache has two advantages:
	- Significant increase in computational efficiency as less computations are performed compared to computing the full QK<sup>T</sup> matrix. This leads to an increase in inference speed. 
	- The maximum required memory is not increased quadratically with the number of generated tokens, but only increases linearly.


> One should always make use of the key-value cache as it leads to identical results and a significant speed-up for longer input sequences. Transformers has the key-value cache enabled by default when making use of the text pipeline or the generate method.


### Multi-round conversation
- The key-value cache is especially useful for applications such as chat where multiple passes of auto-regressive decoding are required.
	- Keeping all the context is crucial for LLMs deployed in chat so that the LLM understands all the previous context of the conversation. 

	- The key-value cache is extremely useful for chat as it allows us to continuously grow the encoded chat history instead of having to re-encode the chat history again from scratch (as e.g. would be the case when using an encoder-decoder architecture).

- Great, no additional time is spent recomputing the same key and values for the attention layer! There is however one catch.
	- While the required peak memory for the QK<sup>T</sup> matrix is significantly reduced, holding the key-value cache in memory can become very memory expensive for long input sequences or multi-turn chat. Remember that the key-value cache needs to store the key-value vectors for all previous input vectors, for all self-attention layers and for all attention heads.

> Example: Let’s compute the number of float values that need to be stored in the key-value cache for the LLM bigcode/octocoder.
> For input sequence length of 16000,
> 2 * 16_000 * config.n_layer * config.n_head * config.n_embd // config.n_head
> 7864320000
> Roughly 8 billion float values! Storing 8 billion float values in float16 precision requires around 15 GB of RAM which is circa half as much as the model weights themselves!

- Researchers have proposed two methods that allow to significantly reduce the memory cost of storing the key-value cache.
	- Multi-Query-Attention (MQA)
	- Grouped-Query-Attention (GQA)


### Multi-Query-Attention (MQA)
- [MQA Paper: Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150)

> Instead of using n_head key-value projections weights, one can use a single head-value projection weight pair that is shared across all attention heads without that the model’s performance significantly degrades.

- As most LLMs use between 20 and 100 attention heads, MQA significantly reduces the memory consumption of the key-value cache.
	- For the LLM used in this notebook we could therefore reduce the required memory consumption from 15 GB to less than 400 MB at an input sequence length of 16000.

- In addition to memory savings, MQA also leads to improved computational efficiency as explained in the following.
	- In auto-regressive decoding, large key-value vectors need to be reloaded, concatenated with the current key-value vector pair to be then fed into the q<sub>c</sub>K<sup>T</sup> computation at every step.
	- For auto-regressive decoding, the required memory bandwidth for the constant reloading can become a serious time bottleneck.
	- By reducing the size of the key-value vectors less memory needs to be accessed, thus reducing the memory bandwidth bottleneck.

- MQA has seen wide adoption by the community and is now used by many of the most popular LLMs:
	- Falcon
	- PaLM
	- MPT
	- BLOOM


### Grouped-Query-Attention (GQA)
- [Paper: Grouped-Query-Attention](https://arxiv.org/abs/2305.13245)

- found that using MQA can often lead to quality degradation compared to using vanilla multi-key-value head projections. 
	- The paper argues that more model performance can be kept by less drastically reducing the number of query head projection weights.
	- Instead of using just a single key-value projection weight, n < n_head key-value projection weights should be used.
	- By choosing n to a significantly smaller value than n_head, such as 2,4 or 8 almost all of the memory and speed gains from MQA can be kept while sacrificing less model capacity and thus arguably less performance.

- Moreover, the authors of GQA found out that existing model checkpoints can be uptrained to have a GQA architecture with as little as 5% of the original pre-training compute.
	- While 5% of the original pre-training compute can still be a massive amount, GQA uptraining allows existing checkpoints to be useful for longer input sequences.


### Takeaways

> it is strongly recommended to make use of either GQA or MQA if the LLM is deployed with auto-regressive decoding and is required to handle large input sequences as is the case for example for chat.

> The reason massive LLMs such as GPT3/4, Llama-2-70b, Claude, PaLM can run so quickly in chat-interfaces such as Hugging Face Chat or ChatGPT is to a big part thanks to the above-mentioned improvements in precision, algorithms, and architecture. Going forward, accelerators such as GPUs, TPUs, etc… will only get faster and allow for more memory, but one should nevertheless always make sure to use the best available algorithms and architectures to get the most bang for your buck.













