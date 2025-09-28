# Latency Optimization

For agents to be viable, they must be both performant and cost-effective. High latency ruins the user experience, while uncontrolled costs make an agent financially unsustainable.

#### Where Latency Comes From

- **LLM Inference Time**: Large models, especially if hosted on the cloud or on limited hardware, can be slow to produce outputs (e.g., hundreds of milliseconds to several seconds per request, depending on model size and prompt length).

- **Token Generation Length**: If the agent’s responses or chain-of-thought are long, generating many tokens serially adds time. The model typically generates tokens one by one.

- **Tool Call Delays**: Any external API or tool the agent calls has its own response time (database queries, web requests, etc.). Some might be quick (<100ms), others (like web search or complex computations) can take seconds.

- **Sequential Steps**: Agents might do multiple reasoning cycles. If an agent needs 5 steps and each step involves an LLM call of 1s, that’s ~5s already not counting other overhead. 

- **Network Overhead**: If using external APIs (like OpenAI’s), network latency to the API and back matters. Also, transferring large prompts or retrieved documents over the network can be slow.

- **Orchestration overhead**: If your orchestrator is not optimized (maybe using a slower language or waiting unnecessarily), it adds up.


#### Latency Reduction Techniques

1. **Model Size and Type**: Use the smallest, fastest model that achieves acceptable quality for the task. Larger models (like GPT-4 class) are usually slower than smaller ones (GPT-3.5 class or distilled models). A great approach is a model cascade: try a fast model first, and only if it’s not confident, defer to a bigger model. Or use the big model for critical steps only. Adam Silverman (Agency AI) noted using different LLMs for different tasks to reduce cost – this also often reduces latency. For example, use a quick model to parse user intent, and only call the heavy model if deep reasoning is needed.

2. **Prompt Optimization**: Longer prompts (with lots of instructions or history) mean more tokens for the model to process, increasing time. You can optimize by removing unnecessary verbosity in instructions, or summarizing long histories. Using few-shot examples in prompts can improve reliability but also adds tokens – consider if you can achieve the same via fine-tuning (which front-loads that knowledge into weights rather than paying a cost each request).

3. **Parallelization of Subtasks**: If an agent has independent subtasks, execute them in parallel. For instance, if the agent needs to call two different APIs and the order doesn’t matter, fire them off concurrently rather than sequentially. OpenAI’s guidance on latency mentions parallelization as a powerful technique for multi-step LLM workflows. Some orchestrators support async tool calls. However, ensure tasks truly don’t depend on each other; otherwise, parallel execution could cause issues.

4. **Caching**: Many agent queries repeat or have overlapping parts. Implement caching at multiple levels:
	5. **Result caching**: If the agent tool call “search X” was done recently, cache the result for a short time. Or if the agent frequently summarizes the same document, cache that summary.

	6. **LLM response caching (prompt-response pairs)**: It’s tricky because the same prompt exactly might not repeat often. But certain sub-prompts might repeat (like if you have a planning prompt for similar tasks).

	7. **Embedding caching**: If using vector search, cache embeddings of common queries or documents to avoid recomputation. According to one source, caching can reduce costs (and implicitly latency) by a huge factor for repetitive prompts. For instance, caching the prompt embeddings or using a prefix cache if many queries share a prefix (some frameworks do “memoization” of transformer computation for identical prefix tokens across requests).

8. **Streaming Responses**: If using an LLM that supports streaming output (token by token), you can start displaying or using partial results before the full completion is done. This doesn’t change total latency for final completion, but improves perceived latency for a user because they start seeing an answer quickly. It’s especially useful in chat interfaces (the user sees the answer being typed out).

9. **Reduce Steps with Smarter Prompts**: Sometimes you can collapse steps. For instance, rather than having the agent do 3 separate tool calls and reasoning in between, maybe one prompt can ask the model to output a plan and a final answer in one go (if latency is more important than absolute correctness, this might be acceptable). Or use techniques like Plan-and-Execute where planning is separate but execution of simple parts might be batched.

10. **Hardware and Hosting Optimizations**: If you host models yourself, use GPUs or optimized inference engines (like ONNX runtime, INT8 quantization, etc.) to speed up LLM inference. If using an API, consider the region and tier (some providers have faster infrastructure for higher price tiers, or specific model versions optimized for speed).

11. **Asynchronous Design**: Don’t block the entire system while the agent works if possible. For instance, if the agent needs to do something that takes 10 seconds (maybe gathering a lot of info), design the interface such that the user can be given an acknowledgment and maybe a progress indicator, rather than freezing. This is more of a UX consideration but important. In a multi-agent system, agents might operate asynchronously and then sync up.

12. **Circuit Breakers for Slow Tools**: If an external API is slow, have a timeout or fallback. For example, if a web search is taking too long, perhaps notify the user “I’ll get back to you later on that” or present partial info. Or use an alternate source. This prevents one hung step from hanging the whole agent.

13. **Profiling and Narrowing Bottlenecks**: Use the observability tools to identify where time is spent. Perhaps 80% of time is in one specific API call – focus optimization efforts there (maybe caching its result or optimizing that service). Or if the LLM is the bottleneck, maybe prompt simplification or a smaller model will give biggest win. A targeted approach ensures you’re not trying to optimize things that don’t impact much.


#### Considerations

**Trade-off with Quality**: Latency optimizations often come with trade-offs. A smaller model might be faster but less accurate. Fewer reasoning steps might be quicker but lead to more errors. It’s important to evaluate the latency vs. quality curve. Define acceptable latency for your use case (e.g., <2 seconds for a chatbot reply perhaps). Then try to maximize quality within that budget. You might use different strategies in different modes – e.g., a “fast mode” vs “accurate mode” for the agent depending on user preference or context. Some tasks where a slight delay is fine (like generating a detailed report) might use the slower thorough chain-of-thought, whereas quick Q&A might use a faster direct approach.


**Example – Speeding up a Multi-Agent System**: Suppose you have an AI scheduling assistant that checks multiple calendars and suggests meeting times. Initially, it sequentially checked each person’s calendar (taking ~1s each) and then compiled results – taking maybe 5-6 seconds to respond. You optimize by parallelizing calendar checks (all calendars queried concurrently, completing in ~1s total since they are independent). You also found the LLM was summarizing availability in a verbose way – you shorten the prompt and use a simpler logic to finalize the time suggestion. Now the agent responds in ~2 seconds total. Users find it much more snappy. You did notice a slight increase in cases where the agent picks a suboptimal time (maybe due to less reasoning), but it’s within tolerable range, and the faster response seems to make users happier on balance.

**Another example: a customer support agent** might have initially always used a powerful but slow reasoning model for every query. By implementing a “quick classifier” model first – to identify if a query is simple or complex – if simple, it uses a lightweight FAQ responder which is very fast; if complex, only then call the heavy model. This kind of routing can dramatically cut the average latency for easy questions (which might be the majority).

