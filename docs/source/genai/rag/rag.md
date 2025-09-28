# RAG

### References
- [BentoML: May 2024: Building RAG with Open-Source and Custom AI Models](https://www.bentoml.com/blog/building-rag-with-open-source-and-custom-ai-models)
- [Cameron Wolfe: A Practitioners Guide to Retrieval Augmented Generation (RAG)](https://cameronrwolfe.substack.com/p/a-practitioners-guide-to-retrieval)
- [Langchain Nov 2023: Deconstructing RAG](https://blog.langchain.dev/deconstructing-rag/)
- [Langchain Nov 2023: Applying OpenAI's RAG Strategies](https://blog.langchain.dev/applying-openai-rag/)


### What is RAG ?

- LLMs are trained on enormous bodies of data but they aren't trained on your data. Retrieval-Augmented Generation (RAG) solves this problem by adding your data to the data LLMs already have access to. 

- In RAG, your data is loaded and prepared for queries or "indexed". User queries act on the index, which filters your data down to the most relevant context. This context and your query then go to the LLM along with a prompt, and the LLM provides a response.

- Even if what you're building is a chatbot or an agent, you'll want to know RAG techniques for getting data into your application.

<div class="container py-4 py-md-5 px-4 px-md-3 text-body-secondary">
    <div class="row" >
      <div class="col-lg-4 mb-4">
        <img src="../../_static/genai/rag/basic_rag.png"></img>
      </div>
    </div>
</div>

- [Llamaindex: High-Level Concepts](https://docs.llamaindex.ai/en/stable/getting_started/concepts/)


### RAG Stages

- <b>Loading</b>: this refers to getting your data from where it lives -- whether it's text files, PDFs, another website, a database, or an API -- into your workflow.
	- Documents
	- Connectors

- <b>Indexing</b>: this means creating a data structure that allows for querying the data. For LLMs this nearly always means creating vector embeddings, numerical representations of the meaning of your data, as well as numerous other metadata strategies to make it easy to accurately find contextually relevant data.
	- Indexes
	- Embeddings

- <b>Storing</b>: once your data is indexed you will almost always want to store your index, as well as other metadata, to avoid having to re-index it.

- <b>Querying</b>: for any given indexing strategy there are many ways you can utilize LLMs and data structures to query, including sub-queries, multi-step queries and hybrid strategies.

- <b>Evaluation</b>: a critical step in any flow is checking how effective it is relative to other strategies, or when you make changes. Evaluation provides objective measures of how accurate, faithful and fast your responses to queries are.


### Structure of a RAG Pipeline
- <b>Cleaning and chunking</b>
	- clean the data and extract the raw textual information from these heterogenous data sources.
	- split it into sets of shorter sequences that typically contain around 100-500 tokens
	- The goal of chunking is to split the data into units of retrieval

- <b>Searching over chunks</b>
	- build a dense retrieval system by 
		- i) using an embedding model2 to produce a corresponding vector representation for each of our chunks and
		- ii) indexing all of these vector representations within a vector database.
	- Then, we can embed the input query using the same embedding model and perform an efficient vector search to retrieve semantically-related chunks.

	- Many RAG applications use pure vector search to find relevant textual chunks, but we can create a much better retrieval pipeline by re-purposing existing approaches from AI-powered search. 
		- Namely, we can augment <b>dense retrieval with a lexical (or keyword-based) retrieval component</b>, forming a hybrid search algorithm.
		- Then, we can add a <b>fine-grained re-ranking step—either with a cross-encoder</b> or a less expensive component—to sort candidate chunks based on relevance

- <b>Data wrangling</b>
	- we can pass a compressed version of the textual information into the LLM’s prompt instead of the full document, thus saving costs.
	- Langchain: add an extra processing step after retrieval that passes textual chunks through an LLM for summarization or reformatting prior to feeding them to the final LLM.

- <b>Do we always search for chunks?</b>
	- Within RAG, we usually use search algorithms to match input queries to relevant textual chunks. However, there are several different algorithms and tools that can be used to power RAG.
		- connecting LLMs to graph databases, forming a RAG system that can search for relevant information via queries to a graph database
		- directly connected LLMs to search APIs like Google or Serper for accessing up-to-date information.

- <b>Generating the output</b>
	- Once we have retrieved relevant textual chunks, the final step of RAG is to insert these chunks into a language model’s prompt and generate an output.


### RAG Advantages
- Reducing hallucinations
- Access to up-to-date information
- Data security
- Ease of implementation
	- Compared to other knowledge injection techniques—finetuning (or continued pretraining) is the primary alternative—RAG is both simpler to implement and computationally cheaper. As we will see, RAG also produces much better results compared to continued pretraining.


### Challenges in production RAG

- <b>Retrieval performance</b>
	- Recall: Not all chunks that are relevant to the user query are retrieved.
	- Precision: Not all chunks retrieved are relevant to the user query.
	- Data ingestion: Complex documents, semi-structured and unstructured data.

- <b>Response synthesis</b>
	- Safeguarding: Is the user query toxic or offensive and how to handle it.
	- Tool use: Use tools such as browsers or search engines to assist the response generation.
	- Context accuracy: Retrieved chunks lacking necessary context or containing misaligned context.

- <b>Response evaluation</b>
	- Synthetic dataset for evaluation: LLMs can be used to create evaluation datasets for measuring the RAG system’s responses.
	- LLMs as evaluators: LLMs also serve as evaluators themselves.



### Improving RAG pipeline with custom AI models
- <b>Embedding models</b>
	- Fine-tuning an embedding model on a domain-specific dataset often enhances the retrieval accuracy. This is due to the improvement of embedding representations for the specific context during the fine-tuning process.
	- Results from OpenAI RAG Experiments
		- did not report a considerable boost in performance from embedding fine-tuning, favorable results have been reported. OpenAI notes that this is probably not advised as "low-hanging-fruit".
		- [Langchain guide to fine tuning open source LLMs](https://blog.langchain.dev/using-langsmith-to-support-fine-tuning-of-open-source-llms/)
		- [Hugging Face: Training Sentence Transformers](https://huggingface.co/blog/how-to-train-sentence-transformers?ref=blog.langchain.dev)


- <b>Which LLM to use ?</b>
	- The right model should align with your data policies, budget plan, and the specific demands of your RAG application.
		- Security and privacy
		- Latency requirement
			- Time to first token
			- Time per output token
			- Is it serving real-time chat applications or offline data processing jobs?
		- Reliability
			- response time and generation quality.
		- Capabilities
			- For simple tasks, can it be replaced by a smaller specialized models?
		- Domain knowledge.


- <b>Context-aware chunking</b>
	- Most simple RAG systems rely on fixed-size chunking, dividing documents into equal segments with some overlap to ensure continuity. This method, while straightforward, can sometimes strip away the rich context embedded in the data.

	- By contrast, context-aware chunking breaks down text data into more meaningful pieces, considering the actual content and its structure. Instead of splitting text at fixed intervals (like word count), it identifies logical breaks in the text using NLP techniques. These breaks can occur at the end of sentences, paragraphs, or when topics shift.


- <b>Parsing complex documents</b>
	- The real world throws complex documents at us - product reviews, emails, recipes, and websites that not only contain textual content but are also enriched with structure, images, charts, and tables. Consider integrating the following models and tools into your RAG systems:

	- <b>Layout analysis</b>:
		- [LayoutLMv3](https://arxiv.org/abs/2204.08387): integrates text and layout with image processing without relying on conventional CNNs, streamlining the architecture and leveraging masked language and image modeling, making it highly effective in understanding both text-centric and image-centric tasks.

	- <b>Table detection and extraction</b>:
		- [Table Transformer (TATR)](https://github.com/microsoft/table-transformer) is specifically designed for detecting, extracting, and recognizing the structure of tables within documents. It operates similarly to object detection models, using a DETR-like architecture to achieve high precision in both table detection and functional analysis of table contents.

	- <b>Document question-answering systems</b>: Building a Document Visual Question Answering (DocVQA) system often requires multiple models, such as models for layout analysis, OCR, entity extraction, and finally, models trained to answer queries based on the document's content and structure.
		- Tools like [Donut](https://arxiv.org/abs/2111.15664) and the latest versions of LayoutLMv3 can be helpful in developing robust DocVQA systems.

	- <b>Fine-tuning</b>: Existing open-source models are great places to start but with additional fine-tuning on your specific documents, handling its unique content or structure, can often lead to greater performance.


- <b>Metadata filtering</b>
	- Incorporating these models into your RAG systems, especially when combined with NLP techniques, allows for the extraction of rich metadata from documents. This includes elements like the sentiment expressed in text, the structure or summarization of a document, or the data encapsulated in a table.
	- <i>Most modern vector databases supports storing metadata alongside text embeddings, as well as using metadata filtering during retrieval, which can significantly enhance the retrieval accuracy.</i>


- <b>Reranking models</b>
	- <b>Initial retrieval</b>: An embedding model acts as a first filter, scanning the entire database and identifying a pool of potentially relevant documents. This initial retrieval is fast and efficient.

	- <b>Reranking</b>: The reranking model then takes over, examining the shortlisted documents from the first stage. It analyzes each document's content in more detail, considering its specific relevance to the user's query. Based on this analysis, the reranking model reorders the documents, placing the most relevant ones at the top (sometimes at both ends of the context window for maximum relevance).

	- Many may think this can increase latency. However, reranking also means you don’t need to send all retrieved chunks to the LLM, leading to faster generation time.

- <b>Cross-modal retrieval</b>
	- Cross-modal retrieval transcends traditional text-based limitations, supporting interplay between different types of data, such as audio and visual content. 
	- when a RAG system incorporates models like [BLIP](https://arxiv.org/abs/2201.12086) for visual reasoning, it’s able to understand the context within images, improving the textual data pipeline with visual insights.
	- [ImageBind: One Embedding Space To Bind Them All](https://www.bentoml.com/blog/building-rag-with-open-source-and-custom-ai-models)


- As we improve our RAG system for production, the complexity increases accordingly. Ultimately, we may find ourselves orchestrating a group of AI models, each playing its part in the workflow of data processing and response generation.


<div class="container py-4 py-md-5 px-4 px-md-3 text-body-secondary">
    <div class="row" >
      <div class="col-lg-4 mb-4">
        <img src="../../_static/genai/rag/rag_components.png"></img>
      </div>
    </div>
</div>

- [BentoML: May 2024: Building RAG with Open-Source and Custom AI Models](https://www.bentoml.com/blog/building-rag-with-open-source-and-custom-ai-models)



### Scaling RAG services with multiple custom AI models

###### Serving embedding models
- Asynchronous non-blocking invocation
- Shared model replica across multiple API workers
- Adaptive batching:
	- Within a BentoML Service, there is a dispatcher that manages how batches should be optimized by dynamically adjusting batch sizes and wait time to suit the current load. This mechanism is called adaptive batching in BentoML. In the context of text embedding models, we often see performance improvements up to 3x in latency and 2x in throughput comparing to non-batching implementations.

###### Self-hosting LLMs
- There are a variety of open-source tools available for self-hosting LLMs.
	- vLLM,
	- OpenLLM,
	- mlc-llm, and
	- TensorRT-LLM 

- Consider the following when choosing such tools:
	- <b>Inference best practices</b>
		- Continuous batching,
		- Paged Attention,
		- Flash Attention, and
		- Automatic prefix caching
	- <b>Customizations</b>
		- advanced stop conditioning (when a model should cease generating further content)
		- specific output formats (ensuring the results adhere to a specific structure or standard), or
		- input validation (using a classification model to detect)

- [BentoML + vLLM](https://github.com/bentoml/BentoVLLM)

- In addition to the LLM inference server, the infrastructure required for scaling LLM workloads also comes with unique challenges.
	
	- <b>GPU Scaling</b>
		- <i>Unlike traditional workloads, GPU utilization metrics can be deceptive for LLMs. Even if the metrics suggest full capacity, there might still be room for more requests and more throughput. This is why solutions like BentoCloud offers concurrency-based autoscaling. Such an approach learns the semantic meanings of different requests, using dynamic batching and wise resource management strategies to scale effectively.</i>
	
	- <b>Cold start and fast scaling with large container image and model files</b>
		- <i>Downloading large images and models from remote storage and loading models into GPU memory is a time-consuming process, breaking most existing cloud infrastructure’s assumptions about the workload. Specialized infrastructure, like BentoCloud, helps accelerate this process via lazy image pulling, streaming model loading and in-cluster caching.</i>


### Model composition
- Model composition is a strategy that combines multiple models to solve a complex problem that cannot be easily addressed by a single model. Two typical scenarios used in RAG systems.

###### Document processing pipeline
- The models used in this process might have different resource requirements, some requiring GPUs for model inference and others, more lightweight, running efficiently on CPUs. Such a setup naturally fits into a distributed system of micro-services, each service serving a different AI model or function. This architectural choice can drastically improve resource utilization and reduce cost.

- BentoML facilitates this process by allowing users to easily implement a distributed inference graph, where each stage can be a separate BentoML Service wrapping the capability of the corresponding model. In production, they can be deployed and scaled separately.


###### Using small language models
- In some cases, "small" models can be an ideal choice for their efficiency, particularly for simpler, more direct tasks like summarization, classification, and translation. 

- <b>Rapid response</b>: 
	- For example, when a user query is submitted, a small model like BERT can swiftly determine if the request is inappropriate or toxic. If so, it can reject the query directly, conserving resources by avoiding the follow-up steps.

- <b>Routing</b>:
	- These nimble models can act as request routers. Fine-tuned BERT-like model needs no more than 10 milliseconds to identify which tools or data sources are needed for a given request. By contrast, an LLM may need a few seconds or more to complete.


###### BentoML: Uniting separate RAG components
- Running a RAG system with a large number of custom AI models on a single GPU is highly inefficient, if not impossible. Although each model could be deployed and hosted separately, this approach makes it challenging to iterate and enhance the system as a whole.

- BentoML is optimized for building such serving systems, streamlining both the workflow from development to deployment and the serving architecture itself. 
	- Define the entire RAG components within one Python file
	- Compile them to one versioned unit for evaluation and deployment
	- Adopt baked-in model serving and inference best practices like adaptive batching
	- Assign each model inference components to different GPU shapes and scale them independently for maximum resource efficiency
	- Monitor production performance in BentoCloud, which provides comprehensive observability like tracing and logging


### History of RAG
- RAG was first proposed in [1]—in 2021, 
	- when LLMs were less explored and Seq2Seq models were extremely popular—to help with solving knowledge-intensive tasks, or tasks that humans cannot solve without access to an external knowledge source.
	- the RAG strategy proposed in [1] is not simply an inference-time technique for improving factuality. Rather, it is a general-purpose finetuning recipe that allows us to connect pretrained language models with external information sources.
	- When training the model in [1], we first embed the input query using the query encoder of DPR and perform a nearest neighbor search within the document index to return the K most similar textual chunks. From here, we can concatenate a textual chunk with the input query and pass this concatenated input to BART to generate an output.

- <b>RAG with LLMs: Main differences</b>
	- Finetuning is optional and oftentimes not used. Instead, we rely upon the in context learning abilities of the LLM to leverage the retrieved data.
	- Due to the large context windows present in most LLMs, we can pass several documents into the model’s input at once when generating a response.


### Evaluation

###### RAGAS
- difficult to evaluate, as there are many dimensions of “performance” that characterize an effective RAG pipeline:
	- The ability to identify relevant documents.
	- Properly exploiting data in the documents via in context learning.
	- Generating a high-quality, grounded output.

- Retrieval Augmented Generation Assessment (RAGAS), for evaluating these complex RAG pipelines without any human-annotated datasets or reference answers. In particular, three classes of metrics are used for evaluation:
	- Faithfulness: the answer is grounded in the given context.
	- Answer relevance: the answer addresses the provided question.
	- Context relevance: the retrieved context is focused and contains as little irrelevant information as possible.

- How RAGAS works ?
	- we can evaluate each of these metrics in an automated fashion by prompting powerful foundation models like ChatGPT or GPT-4. 
		- For example, faithfulness is evaluated by prompting an LLM to extract a set of factual statements from the generated answer, then prompting an LLM again to determine if each of these statements can be inferred from the provided context;
		- Answer and context relevance are evaluated similarly (potentially with some added tricks based on embedding similarity.


### Practical Tips for RAG Applications

###### RAG is a Search Engine
- the same retrieval and ranking techniques that have been used by search engines for years can be applied by RAG to find more relevant textual chunks

- <b>Don’t just use vector search</b>
	- Many RAG systems purely leverage dense retrieval (embeddings based vector search) for finding relevant textual chunks. However, semantic search has a tendency to yield false positives and may have noisy results.
	- To solve this, we should perform hybrid retrieval using a combination of vector and lexical search—just like a normal (AI-powered) search engine! The approach to vector search does not change, but we can perform a parallel lexical search by:
		- Extracting keywords from the input prompt.
		- Performing a lexical search with these keywords.
		- Taking a weighted combination of results from lexical/vector search

	- By performing hybrid search, we make our RAG pipeline more robust and reduce the frequency of irrelevant chunks in the model’s context. 
		- Plus, adopting keyword-based search allows us to perform clever tricks like promoting documents with important keywords, excluding documents with negative keywords, or even augmenting documents with synthetically-generated data for better matching

- <b>Optimizing the RAG pipeline</b>:
	- To improve our retrieval system, we need to collect metrics that allow us to evaluate its results similarly to any normal search engine.
	- we can evaluate the results of our retrieval system using traditional search metrics (e.g., DGC or nDCG), test changes to the system via AB tests, and iteratively improve our results.


- <b>Improving over time</b>
	- Adding ranking to the retrieval pipeline, either using a cross-encoder or a hybrid model that performs both retrieval and ranking: [ColBERT](https://arxiv.org/abs/2004.12832).
		- ColBERTv2 retriever (a free server hosting a Wikipedia 2017 "abstracts" search index containing the first paragraph of each article from 2017 dump)

	- Finetuning the embedding model for dense retrieval over human-collected relevance data (i.e., pairs of input prompts with relevant/irrelevant passages).

	- Finetuning the LLM generator over examples of high-quality outputs so that it learns to better follow instructions and leverage useful context.

	- Using LLMs to augment either the input prompt or the textual chunks with extra synthetic data to improve retrieval.

- <b>How to evaluate ?</b>
	- To successfully apply RAG in practice, it is important that we evaluate all parts of the end-to-end RAG system—including both <b>retrieval and generation</b>—so that we can reliably benchmark improvements that are made to each component.
	- <i>For each of these changes, we can measure their impact over historical data in an offline manner. To truly understand whether they positively impact the RAG system, however, we should rely upon online AB tests that compare metrics from the new and improved system to the prior system in real-time tests with humans.</i>



###### Optimizing the Context Window
- Successfully applying RAG is not just a matter of retrieving the correct context—prompt engineering plays a massive role. Once we have the relevant data, we must craft a prompt that 
	- i) includes this context and
	- ii) formats it in a way that elicits a grounded output from the LLM. 

- Within this section, we will investigate a few strategies for crafting effective prompts with RAG to gain a better understanding of how to properly include context within a model’s prompt.

- <b>RAG needs a larger context window</b>

- <b>Maximizing diversity</b>
	- Although the textual chunks to be included are selected by our retrieval pipeline, we can optimize our prompting strategy by adding a specialized selection component that sub-selects the results of retrieval.

	- Selection does not change the retrieval process of RAG. Rather, selection is added to the end of the retrieval pipeline—after relevant chunks of text have already been identified and ranked—to determine how documents can best be sub-selected and ordered within the resulting prompt.

	- Diversity ranker
		- Use the retrieval pipeline to generate a large set of documents that could be included in the model’s prompt.
		- Select the document that is most similar to the input (or query), as determined by embedding cosine similarity.
		- For each remaining document, select the document that is least similar to the documents that are already selected.

- <b>Optimizing context layout</b>	
	- Despite increases in context lengths, recent research indicates that LLMs struggle to capture information in the middle of a large context window. Information at the beginning and end of the context window is captured most accurately, causing certain data to be “lost in the middle”.

	- we can take the relevant textual chunks from our retrieval pipeline and iteratively place the most relevant chunks at the beginning and end of the context window.


###### Data Cleaning and Formatting
- In most RAG applications, our model will be retrieving textual information from many different sources.
	- For example, an assistant that is built to discuss the details of a codebase with a programmer may pull information from the code itself, documentation pages, blog posts, user discussion threads, and more. In this case, the data being used for RAG has a variety of different formats that could lead to artifacts (e.g., logos, icons, special symbols, and code blocks) within the text that have the potential to confuse the LLM when generating output.

	- <i>In order for the application to function properly, we must extract, clean, and format the text from each of these heterogenous sources. Put simply, there’s a lot more to preprocessing data for RAG than just splitting textual data into chunks</i>


- <b>Performance impact</b>
	- [](https://www.databricks.com/blog/announcing-mlflow-28-llm-judge-metrics-and-best-practices-llm-evaluation-rag-applications-part)
	- investing into proper data preprocessing for RAG has several benefits:
		- 20% boost in the correctness of LLM-generated answers.
		- 64% reduction in the number of tokens passed into the model.
		- Noticeable improvement in overall LLM behavior

- <b>Data cleaning pipeline</b>
	- To craft a functioning data pipeline, we should
		- i) observe large amounts of data within our knowledge base,
		- ii) visually inspect whether unwanted artifacts are present, and
		- iii) amend issues that we find by adding changes to the data cleaning pipeline. 



### OpenAI's RAG Strategies
- Open AI reported a series of RAG experiments for a customer that they worked with. While evaluation metics will depend on your specific application, it’s interesting to see what worked and what didn't for them. 
	- there is no "one-size-fits-all" solution because different problems require different retrieval techniques.

<div class="container py-4 py-md-5 px-4 px-md-3 text-body-secondary">
    <div class="row" >
      <div class="col-lg-4 mb-4">
        <img src="../../_static/genai/rag/openai_rag_strategies.png"></img>
      </div>
    </div>
</div>

<div class="container py-4 py-md-5 px-4 px-md-3 text-body-secondary">
    <div class="row" >
      <div class="col-lg-4 mb-4">
        <img src="../../_static/genai/rag/openai_rag_results.png"></img>
      </div>
    </div>
</div>

Source: [Applying OpenAI's RAG Strategies](https://blog.langchain.dev/applying-openai-rag/)

###### Baseline
- Distance-based vector database retrieval embeds (represents) queries in high-dimensional space and finds similar embedded documents based on "distance".
	- The base-case retrieval method used in the OpenAI study mentioned cosine similarity. 
	- various distance metrics
		- [Weaviate](https://weaviate.io/blog/distance-metrics-in-vector-search?ref=blog.langchain.dev)
		-	[Pinecone](https://www.pinecone.io/learn/vector-similarity/?ref=blog.langchain.dev)


### Major RAG Themes

###### Query Transformations

- how can we make retrieval robust to variability in user input?
	- For example, user questions may be poorly worded for the challenging task of retrieval. 
- [Langchain Blog on Query Transformations](https://blog.langchain.dev/query-transformations/)

- <b>Query expansion</b>
	- Consider the question "Who won a championship more recently, the Red Sox or the Patriots?" Answering this can benefit from asking two specific sub-questions:
		- "When was the last time the Red Sox won a championship?"
		- "When was the last time the Patriots won a championship?"
	- Query expansion decomposes the input into sub-questions, each of which is a more narrow retrieval challenge.

	- [Langchain: Multi-query retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/MultiQueryRetriever?ref=blog.langchain.dev) performs sub-question generation, retrieval, and returns the unique union of the retrieved docs. 

	- [Langchain: RAG fusion](https://github.com/langchain-ai/langchain/blob/master/cookbook/rag_fusion.ipynb?ref=blog.langchain.dev) builds on by ranking of the returned docs from each of the sub-questions.

	- <b>Step back Prompting</b>
		- offers a third approach in this vein, generating a step-back question to ground an answer synthesis in higher-level concepts or principles. 
		- For example, a question about physics can be abstracted into a question and answer about the physical principles behind the user query. The final answer can be derived from the input question as well as the step-back answer. 
		- [Langchain: Step-back prompting](https://github.com/langchain-ai/langchain/blob/master/cookbook/stepback-qa.ipynb?ref=blog.langchain.dev) 

- <b>HyDE</b> 
	- LangChain’s HyDE (Hypothetical Document Embeddings) retriever generates hypothetical documents for an incoming query, embeds them, and uses them in retrieval. The idea is that these simulated documents may have more similarity to the desired source documents than the question. 
	- [HyDE Paper](https://arxiv.org/abs/2212.10496?ref=blog.langchain.dev)
	- [Langchain HyDE](https://python.langchain.com/docs/templates/hyde?ref=blog.langchain.dev)



- <b>Query re-writing</b>
	- To address poorly framed or worded user inputs, [Rewrite-Retrieve-Read](https://arxiv.org/pdf/2305.14283.pdf?ref=blog.langchain.dev) is an approach re-writes user questions in order to improve retrieval.
	- [Langchain: Rewrite](https://github.com/langchain-ai/langchain/blob/master/cookbook/rewrite.ipynb?ref=blog.langchain.dev)

- <b>Query compression</b>
	- In some RAG applications, a user question follows a broader chat conversation. In order to properly answer the question, the full conversational context may be required.
	- [Langchain: WebLang](https://smith.langchain.com/hub/langchain-ai/weblangchain-search-query?ref=blog.langchain.dev&organizationId=1fa8b1f4-fcb9-4072-9aa9-983e35ad61b8) to compress chat history into a final question for retrieval.
	- [Langchain Rewrite implementation](https://github.com/langchain-ai/langchain/blob/master/cookbook/rewrite.ipynb?ref=blog.langchain.dev)


###### Routing
- A second question to ask when thinking about RAG: where does the data live?
	- In many RAG demos, data lives in a single vectorstore but this is often not the case in production settings. When operating across a set of various datastores, incoming queries need to be routed. LLMs can be used to support dynamic query routing effectively
	- [Langchain: Routing](https://python.langchain.com/docs/expression_language/how_to/routing?ref=blog.langchain.dev)



###### Query Construction
- A third question to ask when thinking about RAG: what syntax is needed to query the data?
	- While routed questions are in natural language, data is stored in sources such as relational or graph databases that require specific syntax to retrieve. And even vectorstores utilize structured metadata for filtering. In all cases, natural language from the query needs to be converted into a query syntax for retrieval.

- [Langchain Blog on Query Construction](https://blog.langchain.dev/query-construction/)

<div class="container py-4 py-md-5 px-4 px-md-3 text-body-secondary">
    <div class="row" >
      <div class="col-lg-4 mb-4">
        <img src="../../_static/genai/rag/structured_query.jpeg"></img>
      </div>
    </div>
</div>

- [Langchain Sep 2023: Deconstructing RAG](https://blog.langchain.dev/deconstructing-rag/)

- <b>Text-to-SQL</b>
	- Text-to-SQL can be done easily by providing an LLM the natural language question along with relevant table information; open source LLMs have proven effective at this task, enabling data privacy.
		- [Langchain: SQL](https://python.langchain.com/docs/expression_language/cookbook/sql_db?ref=blog.langchain.dev)
		- [Langchain: SQL Ollama](https://github.com/langchain-ai/langchain/tree/master/templates/sql-ollama?ref=blog.langchain.dev)
		- [Langchain: SQL Llama2](https://github.com/langchain-ai/langchain/tree/master/templates/sql-llama2?ref=blog.langchain.dev)

	- Mixed type (structured and unstructured) data storage in relational databases is increasingly common.
		- An embedded document column can be included using the [open-source pgvector](https://github.com/pgvector/pgvector) extension for PostgreSQL. 
		- It's also possible to interact with this semi-structured data using natural language, marrying the expressiveness of SQL with semantic search
		- [Langchain Cookbook](https://github.com/langchain-ai/langchain/blob/master/cookbook/retrieval_in_sql.ipynb?ref=blog.langchain.dev)
		- [Langchain Template](https://github.com/langchain-ai/langchain/tree/master/templates/sql-pgvector?ref=blog.langchain.dev)


- <b>Text-to-Cypher</b>
	- Knowledge Graphs vs SQL vs Vector Stores
		- While vector stores readily handle unstructured data, they don't understand the relationships between vectors.
		- While SQL databases can model relationships, schema changes can be disruptive and costly. 
		- Knowledge graphs can address these challenges by modeling the relationships between data and extending the types of relationships without a major overhaul. They are desirable for data that has many-to-many relationships or hierarchies that are difficult to represent in tabular form.

	- Example implementation: [Langchain Neo4j: Using a Knowledge Graph to implement a DevOps RAG application](https://blog.langchain.dev/using-a-knowledge-graph-to-implement-a-devops-rag-application/)

	- [Langchain Neo4j Cypher Template](https://github.com/langchain-ai/langchain/tree/master/templates/neo4j-cypher?ref=blog.langchain.dev)
	- [Langchain Neo4j Advanced Template](https://github.com/langchain-ai/langchain/tree/master/templates/neo4j-advanced-rag?ref=blog.langchain.dev)


- <b>Text-to-metadata filters</b>
	- Vectorstores equipped with metadata filtering enable structured queries to filter embedded unstructured documents.
	- [Langchain: Self-query retriever](https://python.langchain.com/docs/modules/data_connection/retrievers/self_query/?ref=blog.langchain.dev#constructing-from-scratch-with-lcel)
	- [Langchain: Self-query Template](https://github.com/langchain-ai/langchain/tree/master/templates/rag-self-query?ref=blog.langchain.dev)


###### Indexing
- how to design my index? For vectorstores, there is considerable opportunity to tune parameters like the chunk size and / or the document embedding strategy to support variable data types.

- <b>Chunk Size</b>
	- OpenAI: notable boost in performance that they saw simply from experimenting with the chunk size during document embedding
	- it's worth examining where the document is split using various split sizes or strategies and whether semantically related content is unnaturally split.


- <b>Document embedding strategy</b>
	- [Langchain Links to Multimodal RAG Implementations](https://blog.langchain.dev/semi-structured-multi-modal-rag/)

	- One of the simplest and most useful ideas in index design is to decouple what you embed (for retrieval) from what you pass to the LLM (for answer synthesis).
		- For example, consider a large passage of text with lots of redundant detail. We can embed a few different representations of this to improve retrieval, such as a summary or small chunks to narrow the scope of information that is embedded. In either case, we can then retrieve the full text to pass to the LLM.
		- These can be implemented using multi-vector and [parent-document](https://python.langchain.com/docs/modules/data_connection/retrievers/parent_document_retriever?ref=blog.langchain.dev) retriever, respectively.

	- The multi-vector retriever also works well for semi-structured documents that contain a mix of text and tables
		- In these cases, it's possible to extract each table, produce a summary of the table that is well suited for retrieval, but return the raw table to the LLM for answer synthesis.

		- <i>We can take this one step further: with the advent of multi-modal LLMs, it's possible to use generate and embed image summaries as one means of image retrieval for documents that contain text and images</i>

		- This may be appropriate for cases where multi-modal embeddings are not expected to reliably retrieve the images, as may be the case with complex figures or table.
			- [Langchain Multimodal RAG Cookbook](https://github.com/langchain-ai/langchain/blob/master/cookbook/Multi_modal_RAG.ipynb?ref=blog.langchain.dev)

		- Alternate approach: using open source (OpenCLIP) multi-modal embeddings for retrieval of images based on more straightforward visual concepts.
			- [Langchain OpenCLIP MultiModal RAG Cookbook](https://github.com/langchain-ai/langchain/blob/master/cookbook/multi_modal_RAG_chroma.ipynb?ref=blog.langchain.dev)


<div class="container py-4 py-md-5 px-4 px-md-3 text-body-secondary">
    <div class="row" >
      <div class="col-lg-4 mb-4">
        <img src="../../_static/genai/rag/multi_modal_options.jpeg"></img>
      </div>
    </div>
</div>



###### Post-Processing
- how to combine the documents that I have retrieved?
	- This is important, because the context window has limited size and redundant documents (e.g., from different sources) will utilize tokens without providing unique information to the LLM. 
	- A number of approaches for document post-processing (e.g., to improve diversity or filter for recency) have emerged.

- <b>Re-ranking</b>
	
	- <b>Cohere ReRank</b> endpoint is one approach, which can be used for document compression (reduce redundancy) in cases where we are retrieving a large number of documents.
		- First stage keyword search followed by second stage semantic Top K retrieval
		- [Langchain Cohere ReRank](https://python.langchain.com/docs/integrations/retrievers/cohere-reranker?ref=blog.langchain.dev)
	
	- <b>RAG-fusion</b> 
		- uses reciprocal rank fusion to ReRank documents returned from a retriever.
		- [RAG Fusion](https://github.com/Raudaschl/rag-fusion)
		- [RAG Fusion Blog](https://towardsdatascience.com/forget-rag-the-future-is-rag-fusion-1147298d8ad1?ref=blog.langchain.dev)
		- [Langchain RAG Fusion Implementation](https://github.com/langchain-ai/langchain/blob/master/cookbook/rag_fusion.ipynb?ref=blog.langchain.dev)


	- <b>MMR</b>:
		- To balance between relevance and diversity, many vectorstores offer max-marginal-relevance search.
		- [Pinecone MMR](https://python.langchain.com/docs/integrations/vectorstores/pinecone?ref=blog.langchain.dev#maximal-marginal-relevance-searches)


- <b>Classification</b>
	- OpenAI classified each retrieved document based upon its content and then chose a different prompt depending on that classification. This marries tagging of text for classification with [logical routing](https://python.langchain.com/docs/expression_language/how_to/routing?ref=blog.langchain.dev) (in this case, for the prompt) based on a tag.


- <b>Clustering</b>
	- Some approaches have used clustering of embedded documents with sampling, which may be helpful in cases where we are consolidating documents across a wide range sources.
	- [Langchain Merger Retrievers](https://python.langchain.com/docs/integrations/retrievers/merger_retriever?ref=blog.langchain.dev)


<div class="container py-4 py-md-5 px-4 px-md-3 text-body-secondary">
    <div class="row" >
      <div class="col-lg-4 mb-4">
        <img src="../../_static/genai/rag/mmr_clustering.jpeg"></img>
      </div>
    </div>
</div>



### Advanced techniques

- [LlamaIndex RAG Cheatshet](https://www.llamaindex.ai/blog/a-cheat-sheet-and-some-recipes-for-building-advanced-rag-803a9d94c41b)

<div class="container py-4 py-md-5 px-4 px-md-3 text-body-secondary">
    <div class="row" >
      <div class="col-lg-4 mb-4">
        <img src="../../_static/genai/rag/basic_rag_cheatsheet.png"></img>
      </div>
    </div>
</div>

<div class="container py-4 py-md-5 px-4 px-md-3 text-body-secondary">
    <div class="row" >
      <div class="col-lg-4 mb-4">
        <img src="../../_static/genai/rag/advanced_rag_cheatsheat.png"></img>
      </div>
    </div>
</div>


###### Generation must be able to make good use of the retrieved documents
- Information Compression
- Result Re-Rank
- Generator Fine Tuning
	- Fine tune to help ensure retrieved docs are aligned to LLM
	- SELF-RAG
- Adapter Methods
	- Attach external adapters to align relevant docs to LLM

###### Simultaneously addressing Retrieval and Generation success requirements
- Generator-Enhanced Retrieval: These techniques make use of the LLM’s inherent reasoning abilities to refine the user query before retrieval is performed so as to better indicate what exactly it requires to provide a useful response.
	- [LlamaIndex Generator-Enhanced Retrieval Recipe](https://docs.llamaindex.ai/en/stable/examples/query_engine/flare_query_engine.html)
- Iterative Retrieval-Generator RAG: For some complex cases, multi-step reasoning may be required to provide a useful and relevant answer to the user query.
	- ITRG, ITER-RETGEN
	- [LlamaIndex Iterative Retrieval-Generator Recipe](https://docs.llamaindex.ai/en/stable/examples/evaluation/RetryQuery.html#retry-query-engine)


### MultiModel RAG: Multi-Vector Retriever for RAG on tables, text, and images
- Langchain Cookbooks for Multimodal RAG
	- [Cookbook for multi-modal (text + tables + images) RAG](https://github.com/langchain-ai/langchain/blob/master/cookbook/Semi_structured_and_multi_modal_RAG.ipynb?ref=blog.langchain.dev)
	- [Cookbook for private multi-modal (text + tables + images) RAG](https://github.com/langchain-ai/langchain/blob/master/cookbook/Semi_structured_multi_modal_RAG_LLaMA2.ipynb?ref=blog.langchain.dev)


###### Multi-Vector Retriever
- It uses a simple, powerful idea for RAG: decouple documents, which we want to use for answer synthesis, from a reference, which we want to use for retriever.

- <b>Document Loading</b>
	- <b>Unstructured</b> is a great ELT tool well-suited for this because it can extract elements (tables, images, text) from numerous file types.
		- For example, Unstructured will partition PDF files by first removing all embedded image blocks.
		- Then it will use a layout model (YOLOX) to get bounding boxes (for tables) as well as titles, which are candidate sub-sections of the document (e.g., Introduction, etc).
		- It will then perform post processing to aggregate text that falls under each title and perform further chunking into text blocks for downstream processing based on user-specific flags (e.g., min chunk size, etc)

- <b>Semi-Structured Data</b>
	- The combination of Unstructured file parsing and multi-vector retriever can support RAG on semi-structured data, which is a challenge for naive chunking strategies that may spit tables. We generate summaries of table elements, which is better suited to natural language retrieval. If a table summary is retrieved via semantic similarity to a user question, the raw table is passed to the LLM for answer synthesis as described above.

	- [Langchain Cookbook for semi-structured (tables + text) RAG](https://github.com/langchain-ai/langchain/blob/master/cookbook/Semi_Structured_RAG.ipynb?ref=blog.langchain.dev)

- <b>Multi-Modal Data</b>
	- We can take this one step further and consider images, which is quickly becoming enabled by the release of <i>multi-modal LLMs such as GPT4-V and open source models such as LLaVA and Fuyu-8b</i>. There are at least three ways to approach the problem, which utilize the multi-vector retriever framework as discussed above.

	- <b>Option 1</b>: Use multimodal embeddings (such as CLIP) to embed images and text together. Retrieve either using similarity search, but simply link to images in a docstore. Pass raw images and text chunks to a multimodal LLM for synthesis.

	- <b>Option 2:</b> Use a multimodal LLM (such as GPT4-V, LLaVA, or FUYU-8b) to produce text summaries from images. Embed and retrieve text summaries using a text embedding model. And, again, reference raw text chunks or tables from a docstore for answer synthesis by a LLM; in this case, we exclude images from the docstore (e.g., because can't feasibility use a multi-modal LLM for synthesis).
		-  <i>[7b parameter LLaVA](https://huggingface.co/mys/ggml_llava-v1.5-7b/tree/main) model ([weights](https://huggingface.co/mys/ggml_llava-v1.5-7b/tree/main)) to generate image image summaries. LLaVA recently added to llama.cpp, which allows it run on consumer laptops (Mac M2 max, 32gb \~45 token / sec) and produces reasonable image summaries.</i>

	- <b>Option 3</b>: Use a multimodal LLM (such as GPT4-V, LLaVA, or FUYU-8b) to produce text summaries from images. Embed and retrieve image summaries with a reference to the raw image, as we did above in option 1. And, again, pass raw images and text chunks to a multimodal LLM for answer synthesis. This option is sensible if we don't want to use multimodal embeddings.

<div class="container py-4 py-md-5 px-4 px-md-3 text-body-secondary">
    <div class="row" >
      <div class="col-lg-4 mb-4">
        <img src="../../_static/genai/rag/multi_modal_options.jpeg"></img>
      </div>
    </div>
</div>


- <b>Multi-Modal Models</b>
	- <b>Open Source options</b>
		- If data privacy is a concern, this RAG pipeline can be run locally using open source components on a consumer laptop.
		- [LLaVA](https://github.com/haotian-liu/LLaVA/?ref=blog.langchain.dev) 7b for image summarization,
		- [Chroma](https://www.trychroma.com/?ref=blog.langchain.dev) vectorstore
		- open source embeddings (Nomic’s [GPT4All](https://python.langchain.com/docs/integrations/text_embedding/gpt4all?ref=blog.langchain.dev)), 
		- the multi-vector retriever, and
		- [LLaMA2-13b-chat](https://python.langchain.com/docs/integrations/chat/ollama?ref=blog.langchain.dev) via [Ollama.ai](http://ollama.ai/?ref=blog.langchain.dev) for answer generation.
		- [Langchain Cookbook for private multi-modal (text + tables + images) RAG](https://github.com/langchain-ai/langchain/blob/master/cookbook/Semi_structured_multi_modal_RAG_LLaMA2.ipynb?ref=blog.langchain.dev)










