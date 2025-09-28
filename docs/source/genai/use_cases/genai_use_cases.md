# GenAI Use Cases

### References
- [Llamaindex: High-Level Concepts](https://docs.llamaindex.ai/en/stable/getting_started/concepts/)


### LLM Applications Categories
- <b>Structured Data Extraction</b>
	- Pydantic extractors allow you to specify a precise data structure to extract from your data and use LLMs to fill in the missing pieces in a type-safe way. This is useful for extracting structured data from unstructured sources like PDFs, websites, and more, and is key to automating workflows.

- <b>Query Engines</b>:
	- A query engine is an end-to-end flow that allows you to ask questions over your data. It takes in a natural language query, and returns a response, along with reference context retrieved and passed to the LLM.

- <b>Chat Engines</b>:
	- A chat engine is an end-to-end flow for having a conversation with your data (multiple back-and-forth instead of a single question-and-answer).

- <b>Agents</b>:
	- An agent is an automated decision-maker powered by an LLM that interacts with the world via a set of tools. Agents can take an arbitrary number of steps to complete a given task, dynamically deciding on the best course of action rather than following pre-determined steps. This gives it additional flexibility to tackle more complex tasks.


### Use Cases

- <b>Amazon Rufus</b>
	- Rufus is a generative AI-powered expert shopping assistant trained on Amazon’s extensive product catalog, customer reviews, community Q&As, and information from across the web to answer customer questions on a variety of shopping needs and products, provide comparisons, and make recommendations based on conversational context.

	- From broad research at the start of a shopping journey such as “what to consider when buying running shoes?” to comparisons such as “what are the differences between trail and road running shoes?” to more specific questions such as “are these durable?”, Rufus meaningfully improves how easy it is for customers to find and discover the best products to meet their needs, integrated seamlessly into the same Amazon shopping experience they use regularly.

	- Features
		- Learn what to look for while shopping product categories:
			- what to consider when buying headphones?
		- Shop by occasion or purpose
			- I want to start an indoor garden
		- Get help comparing product categories
			- compare drip to pour-over coffee makers
		- Find the best recommendations
			- best dinosaur toys for a 5-year-old
		- Getting the latest product updates
			- What are denim trends for women?
		- Accessing current and past orders
			- When are my dog treats arriving?
			- When was the last time I ordered sunscreen?
		- Answering questions not obviously related to shopping
			- What do I need for a summer party?
		- Ask questions about a specific product while on a product detail page
			- is this jacket machine washable?
			- Rufus will generate answers based on listing details, customer reviews, and community Q&As.










