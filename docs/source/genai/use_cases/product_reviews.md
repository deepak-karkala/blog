# Product Reviews: Summarization


### What is AI summarization?
- Put simply, AI summarization is the use of AI technologies to distill text, documents, or content into a short and easily digestible format. For example, AI summarization can use natural language processing or understanding to condense a long PDF and restate its most important takeaways in just a few sentences.

- There are two primary types of AI summarization: extractive and abstractive.
	- <b>Extractive summarization</b> leverages statistical methods to identify sentences that are most likely to be important.
	- <b>Abstractive summarization</b> generates new sentences that summarize the main points of the original text.


### Why product reviews summarization ?
- The benefits of AI summarization
	- range from cost savings to improved accessibility to information. AI summarization can help businesses and organizations save time and money when producing research, business intelligence, or insights. AI-powered summarization can extract key information from news articles, research, legal and financial documents, technical literature, and even customer feedback. Summartization, then, means more time acting on information instead of sifting through it.

- Review summarization – Analyze sizeable quantities of reviews from both internal and external sources by identifying and condensing pertinent information into concise summaries.
	- allow customers to more easily surface reviews that mention certain product attributes. For example, a customer looking to understand whether a product is easy to use can easily surface reviews mentioning “ease of use” by tapping on that product attribute under the review highlights.


- Sentiment analysis – Assess whether the reviews have a positive, negative, or neutral tone, and assign confidence scores for the given sentiment.

- Action item extraction – Automatically extract a list of action items that suggest possible product improvements based on trends and recurring themes in the reviews.

- Visualization – Generate business intelligence (BI) dashboards that display key metrics and graphs.

- Business value
	- Improve product and service quality
	- Improve the customer experience
	- Scale and speed
	- <b>Monitoring marketplace seller performance</b>
		- <i>By using automated sentiment analysis of marketplace reviews to classify customer reviews as positive, negative, and neutral, marketplaces can systematically monitor sellers’ performance and rapidly detect problems.</i>


### System Architecture

<div class="container py-4 py-md-5 px-4 px-md-3 text-body-secondary">
    <div class="row" >
      <div class="col-lg-4 mb-4">
        <img src="../../_static/genai/use_cases/product_reviews/summarization_architecture.jpeg"></img>
      </div>
    </div>
</div>

Source: [AWS: Analyze customer reviews using Amazon Bedrock](https://aws.amazon.com/blogs/machine-learning/analyze-customer-reviews-using-amazon-bedrock/)

###### Workflow
- Customer reviews can be imported into an Amazon Simple Storage Service (Amazon S3) bucket as JSON objects. This bucket will have event notifications enabled to invoke an AWS Lambda function to process the objects created or updated.
- The Lambda function runs the business logic to process the customer reviews within the input JSON file. These reviews are then included as context in the predefined prompt template used as input to the FM. The prompt has detailed instructions to be followed by the FM to generate a JSON output with summary, sentiment, and action items from the reviews. The function then invokes an FM of choice on Amazon Bedrock.
- Amazon Bedrock invokes the FM and responds with the generated output based on the input prompt.
- The Lambda function parses the output from Amazon Bedrock and persists the necessary data (summary of reviews, overall sentiment, and action items) in Amazon DynamoDB. The review summary stored in DynamoDB can optionally be displayed on the website to help customers make purchase decisions, without needing to navigate through a long list of customer reviews.
- Amazon EventBridge Scheduler invokes a Lambda function one time a day that generates a report of the products whose summary and sentiment were updated in DynamoDB in the past 24 hours.
- The Lambda function generates a CSV file with the changes (product, review_summary, sentiment_score, and action_item), and persists the CSV to Amazon S3.
- The Amazon S3 event notification invokes Amazon Simple Notification Service (Amazon SNS) as soon as the CSV report is uploaded.
- Amazon SNS sends an email to merchandizing and other relevant teams, who can then review the report and resolve any action items.
- Optionally, data stored on DynamoDB can be used to build business dashboards to monitor the customer sentiment about products or services over time. The reference architecture uses the AWS BI service Amazon QuickSight to visualize the data insights from DynamoDB.



### Data Pre-processing
- <b>Sampling</b>
	- <b>need to pay attention to both good and bad reviews, and prepare our data so that a good amount of both sides make it through</b>
		- chances are most products receive more positive reviews than negative (hinting at an imbalanced dataset), so if we straight up implement a generic sampler, we might loose a good proportion of the negative reviews received by the products.
	- <b>context lengths </b>
		- Most of the LLMs' performance begin to degrade with longer context lengths both from an quality perspective (how good is the summary?) and performance perspective (how fast can it run?). So, we need to slice and dice our reviews in a respective way to create sensible batches of reviews.

- <b>Filtering</b>
	- <b>Time Filters: No need to use all reviews</b>
		- Choose an appropriate duration of reviews you might want to consider for generating summary and sentiment (for example, excluding customer reviews older than X years, and so on).
		- not all product reviews need to be selected to generate the result. Instead, what can be done is some data analysis beforehand on how helpful people consider each review, to only select the reviews that customers find beneficial. This way, the focus is on reviews that are proven to resonate with the end users of the products.

	- <b>Bot reviews</b>
		- same reviewer reviewed the same product multiple times
		- One liners

	- <b>Verified vs Non-Verified Purchases</b>
		- The difference is - if a review is verified, it means that the review came from someone who actually purchased the book. Non-verified reviews may include false reviews or inputs from bots who try to spam reviews for multiple reasons. Ideally, we want to keep our data as high quality as possible.

- <b>Transformations</b>
	- Given the UNIX time, we want to extract things like date and time, year month, and year week so we can slice and dice our data properly with different time ranges.

- <b>Cleaning</b>
	- Some of the reviews and the headlines contain unexpected characters like HTML code or links. We want to get rid of those to make sure only relevant information stays on in the text

- <b>Token count</b>
	- A simple math to go from word count to token count is: <code>n_tokens = n_words * 1.2</code>
	- Calculate total number of tokens in reviews dataset using tiktoken library to get estimate of cost
	- Check if there are any text which the tiktoken encoder couldn't deal with.


- <b>Review length</b>
	- Plot histogram
	- some reviews which are quite short (less than 3-4 words long.)
	- most of the reviews are less than 200 tokens long
	- <b>what to do with very long reviews </b>
		- Filter out reviews longer than x amount of tokens
	 	- Cut the reviews short, meaning that take only the first x amount of tokens.
	- <b>Star Rating vs Token Length: see if there is a relationship between token length and star rating</b>
		- Intuition: a disappointed reviewer might leave a longer review/complaint.
		- It does look like reviews get shorter by their avg and median token count, and standard deviation gets smaller as we move up from 1 to 5 ratings.
	- <i>We can also tell that the average review is shorter than 70 tokens for all rating groups and the standard deviation is at 136, which means that 70 + 136 = 206 token length will be long enough to capture most of the reviews.</i>
		- Verify what % of reviews have token length longer than this? 
			- 5% of the reviews are too long for each rating.

- <b>Star Rating Distribution</b>
	- there is a big imbalance towards 5s & 4s. Combined, they make up for 85% of the reviews. We have to be careful about this imbalance while sampling
	
	- If we just sample and take a bunch of reviews, and then try to understand how a product can be improved, we might run into some problems because our sample dataset will be overwhelmed with positive reviews. Therefore, what we can do is create two classes - high_score and low_score.

	- This can assume that a customer who rates a product with 4 or 5 stars is giving it a high score, meaning that they are most probably satisfied, and a customer that gives anything lower than that is giving it a low score, meaning that they are not satisfied with their purchase.

	- Then, we can use the high scores to understand what the customers are happy with, and the low scores to understand what can be improved, or what the customers disliked.


- <b>Review Length Shortening</b>
	- shorten the reviews rather than filtering them out, and set the max_n_tokens to be 200.


- <b> Popular Products</b>
	- Some of the products or services in your catalog might have a large volume of customer reviews whose overall size can be much higher than the context window size of the model you chose for inference. Apply alternate techniques to analyze such reviews:
	- For example, split the customer reviews of the product or service into multiple groups, analyze each group separately in the first iteration, then use the results of the first iteration as input context and generate the final output (that is, the final output review summary will be a summary of all review summaries from the first iteration). It might need multiple iterations depending on the volume of reviews.

- <b>Privacy</b>
	- If the customer review feed files have any customer details, classify the S3 bucket used for storage accordingly and apply the necessary security guardrails to limit access to this dataset. Also, make sure you don’t include any customer information in the prompt to the FM. 




### Prompting for Summarization

###### Models
- Google Gemini
- [Vertex AI PaLM API](https://codelabs.developers.google.com/text-summ-large-docs-stuffing)
- [Bedrock: Anthropic Claude 3 Sonnet](https://docs.aws.amazon.com/bedrock/latest/userguide/model-parameters-claude.html)
- Falcon-7B-Instruct
- Llama-2-7B-Chat
- Stable-Beluga-7B
- MPT-7B-Instruct Model
- Documents
	- [Google Cloud's Document AI](https://cloud.google.com/document-ai)


###### Choosing a model
- The following suggestions should be considered when choosing an FM:
	- Review the end-user license agreements and request model access for the FMs you want to work with.
	- Experiment with different text generation models supported by Amazon Bedrock.
	- Use the Amazon Bedrock model evaluation feature to evaluate the supported models. You can use Amazon SageMaker Ground Truth to label the sample dataset that you want to use for model evaluation on Amazon Bedrock.
	- Review the model pricing by provider and Amazon Bedrock service quotas.
		- <i>large Claude V2 model can be switched for a smaller, more focused model that is more aware of the context in which it acts. This way, cost is reduced while performance is improved.</i>

	- Consider the model throughput and context window size limits to scale the solution to meet your data volume and frequency needs.

- Cost
	- Third party embedding model vs Open Source (Llama 2) model

- Throughput
	- Number of reviews processed per time
	- Example: Databricks: 760,000 reviews per hour


- Managed Models vs Self Hosted Models
	- <b>Amazon Bedrock</b>
		- Amazon Bedrock is a fully managed service that offers a choice of high-performing FMs from leading AI companies like AI21 Labs, Anthropic, Cohere, Meta, Mistral AI, Stability AI, and Amazon through a single API, along with a broad set of capabilities to build generative AI applications with security, privacy, and responsible AI.

		- automatically summarizes key information, recognizes the customer sentiment, and generates actionable insights from customer reviews. This method shows significant promise in saving human analysts time while producing high-quality results.

	- <b>Inference optimizations</b>


###### Tokenizer
- EleutherAI/gpt-neox-20b


###### Parameters
- <b>Temperature</b> – The amount of randomness injected into the response. Defaults to 1. Ranges from 0-1.
- <b>Top P</b> – Use nucleus sampling. In nucleus sampling, Anthropic Claude computes the cumulative distribution over all the options for each subsequent token in decreasing probability and cuts it off after it reaches a particular probability specified by top P. You should alter either temperature or top P, but not both.
- <b>Top K</b> – Only sample from the top K options for each subsequent token. Use top K to remove long tail low probability responses.
- <b>Maximum Length</b> – The maximum number of tokens to generate before stopping.


###### Prompting
- Identify the insights you want to derive from the customer reviews and refine the model prompts and parameters to suit your needs.
- Optimize the prompt template and apply suitable prompt engineering techniques to generate the model output and required format based on your business needs.

- We provide a list of reviews as context and create a prompt to generate an output with a concise summary, overall sentiment, confidence score of the sentiment, and action items from the input reviews. Our example prompt requests the FM to generate the response in JSON format.

<div class="container py-4 py-md-5 px-4 px-md-3 text-body-secondary">
    <div class="row" >
      <div class="col-lg-4 mb-4">
        <img src="../../_static/genai/use_cases/product_reviews/summarization_prompt.png"></img>
      </div>
    </div>
</div>

Source: [AWS: Analyze customer reviews using Amazon Bedrock](https://aws.amazon.com/blogs/machine-learning/analyze-customer-reviews-using-amazon-bedrock/)

The output generated in response to this prompt is a JSON string that includes the following attributes:
- reviews_summary – The summary generated from the input customer reviews of a product.
- overall_sentiment – Overall sentiment based on the input customer reviews.
- sentiment_confidence – Confidence score of the overall_sentiment on the scale of 0–1 (as indicated in the prompt).
- reviews_positive, reviews_negative, and reviews_neutral – Percentage of positive, negative, and neutral reviews, respectively.
- action_items – List of action items identified from the input reviews.


```json
{
  "reviews_summary": " The reviews indicate that Hanes sweatpants are generally comfortable, well-made, and offer good value for the price. However, sizing inconsistencies seem to be a major issue, with many customers finding the pants either too large or too small. The lack of pockets and fading issues were also mentioned. Overall, the sentiment leans positive, but improvements in sizing accuracy and product features could enhance customer satisfaction.",
  "overall_sentiment": "positive",
  "sentiment_confidence": 0.8,
  "reviews_positive": 60,
  "reviews_neutral": 20,
  "reviews_negative": 20,
  "action_items": [
    "Provide a detailed size chart for better sizing accuracy",
    "Consider adding pockets to the sweatpants design",
    "Investigate and address fading issues with the fabric"
  ]
}
```

- Other questions to include
	- What are the top three points of negative feedback found across these reviews?
	- What features do our customers like best about this product?
	- Do customers feel they are receiving sufficient value from the product relative to what they are being asked to pay?
	- Are there any reviews that are especially negative or are using inappropriate language?
	- Recognizing that positively rated products are likely to highlight the strengths of these books while lower rated products are likely to focus on their weaknesses, they separate these reviews based on user-provided ratings and task an LLM to extract different sets of information from each high-level category of reviews.


###### Prompt Engineering Experiments


- <b>Positive Prompt Variations</b>
```python

# Prompt Variations
positive_prompt_1 = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Provide three bullet-point summary capturing what customers liked about this book using the reviews below.

Reviews: {review}

### Response:
"""

positive_prompt_2 = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Identify three aspects that readers liked about the book and provide a summary for each from the reviews below.

Reviews: {review}

### Response:
"""

positive_prompt_3 = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Distill and provide three bullet points capturing what customers most appreciated about the book from the reviews below.

Reviews: {review}

### Response:
"""

positive_prompt_4 = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Identify three distinct and specific aspects that readers enjoyed about the book from the reviews below, and provide a bullet point summary for each.

Reviews: {review}

### Response:
"""

positive_prompt_5 = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Analyze the book reviews below and identify three distinct aspects that readers enjoyed about the book. Return the result as three succinct bullet points.

Reviews: {review}

### Response:
"""

positive_prompt_6 = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Analyze the book reviews below and identify three distinct aspects that readers enjoyed about the book. Be sure to include any character dynamics, plot elements, or emotional responses mentioned by the reviewers. Return the result as three succinct bullet points.

Reviews: {review}

### Response:
"""

positive_prompt_7 = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Analyze the provided book reviews and identify distinct aspects in three bullet points that readers enjoyed about the book. For each aspect, provide a brief explanation using the specific details mentioned in the reviews, focusing on character dynamics, plot elements, or emotional responses elicited.

Reviews: {review}

### Response:
"""
```


- <b>Negative Prompt Variations</b>
```python
negative_prompt_1 = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Provide a three bullet-point summary capturing what customers disliked about this book using the reviews below.

Reviews: {review}

### Response:
"""

negative_prompt_2 = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Identify three aspects that readers disliked about the book and provide a summary for each from the reviews below.

Reviews: {review}

### Response:
"""

negative_prompt_3 = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Distill and provide three bullet points capturing what customers most criticized about the book from the reviews below.

Reviews: {review}

### Response:
"""

negative_prompt_4 = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Identify three distinct and specific aspects that readers did not enjoy about the book from the reviews below, and provide a bullet point summary for each.

Reviews: {review}

### Response:
"""

negative_prompt_5 = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Analyze the book reviews below and identify three distinct aspects that readers disliked about the book. Return the result as three succinct bullet points.

Reviews: {review}

### Response:
"""

negative_prompt_6 = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Analyze the provided book reviews and identify three distinct aspects that readers disliked about the book. Be sure to include any character dynamics, plot elements, or emotional responses mentioned by the reviewers that led to negative experiences. Return the answer as three succinct bullet points.

Reviews: {review}

### Response:
"""

negative_prompt_7 = """Below is an instruction that describes a task. Write a response that appropriately completes the request.
### Instruction:
Analyze the provided book reviews and identify distinct aspects in three bullet points that readers disliked about the book. For each aspect, provide a brief explanation using the specific details mentioned in the reviews, focusing on character dynamics, plot elements, or emotional responses elicited.

Reviews: {review}

### Response:
"""
```


### [Evaluating Summarization](../evaluation/summarization.md)



### Inference Optimisations
- <b>Compare benchmarks for various libraries</b>
	- CTranslate2
	- vLLM
	- TGI
	- TensorRT-LLM
- Quantization at various bits vs Performance
- Find the optimal batch size parameter based on GPU memory



### Deployment
###### Serverless (AWS Lambda) vs Batch (AWS Batch / EMR)
- Analyze products in batches to limit the number of concurrent Lambda invocations if your product or service volumes are higher. You might need an event scheduler to invoke the Lambda functions instead of the current Amazon S3 event notifications, which invoke one Lambda function per product JSON. Review Lambda quotas and function timeout to create batches. You can also consider alternate services such as AWS Step Functions or AWS Batch.






### Post Deployment

###### When to update summaries for reviews ?
- Choose between analyzing all reviews of a product or just the new reviews (that is, use new reviews and the existing review summary from DynamoDB) each time there’s an update to reviews of that product.

- Analyze the customer reviews of a product or service only when there are new reviews added for the day:
	- Import the customer review JSON files to an S3 bucket only when there are new reviews for the product.
	- Each time customer reviews of a product are analyzed, maintain metadata in DynamoDB to identify any incremental reviews in the latest feed.

###### Business Process / KPI:
- Define a business process to review the sentiment scores and action items of products and services that have recurring negative sentiments in reviews, take actions to resolve your customer concerns, and improve your products and services.
- Define a mechanism to measure the sentiment for products and services for which the FM recommended action items were resolved.



### Langchain Implementation
- A central question for building a summarizer is how to pass your documents into the LLM's context window. Two common approaches for this are:

	- <b>Stuff</b>: Simply "stuff" all your documents into a single prompt. This is the simplest approach (see here for more on the create_stuff_documents_chain constructor, which is used for this method).
		- We can use chain_type="stuff", especially if using larger context window models such as:
			- 16k token OpenAI gpt-3.5-turbo-1106
			- 100k token Anthropic Claude-2

	- <b>Map-reduce</b>: Summarize each document on it's own in a "map" step and then "reduce" the summaries into a final summary (see here for more on the MapReduceDocumentsChain, which is used for this method).



<div class="container py-4 py-md-5 px-4 px-md-3 text-body-secondary">
    <div class="row" >
      <div class="col-lg-4 mb-4">
        <img src="../../_static/genai/use_cases/product_reviews/stuff_mapreduce.png"></img>
      </div>
    </div>
</div>

Source: [Langchain: Summarization Implementation](https://python.langchain.com/v0.1/docs/use_cases/summarization/)


- either pipeline can be wrapped in a single object: load_summarize_chain
```python
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI

loader = WebBaseLoader("https://lilianweng.github.io/posts/2023-06-23-agent/")
docs = loader.load()

llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-1106")
chain = load_summarize_chain(llm, chain_type="stuff")

chain.run(docs)
```

- <b>Option 1. Stuff</b>
	- When we use load_summarize_chain with chain_type="stuff", we will use the StuffDocumentsChain.
	- The chain will take a list of documents, inserts them all into a prompt, and passes that prompt to an LLM:

```python
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from langchain_core.prompts import PromptTemplate

# Define prompt
prompt_template = """Write a concise summary of the following:
"{text}"
CONCISE SUMMARY:"""
prompt = PromptTemplate.from_template(prompt_template)

# Define LLM chain
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Define StuffDocumentsChain
stuff_chain = StuffDocumentsChain(llm_chain=llm_chain, document_variable_name="text")

docs = loader.load()
print(stuff_chain.run(docs))
```


- <b>Option 2. Map-Reduce</b>
	- we'll first map each document to an individual summary using an LLMChain. Then we'll use a ReduceDocumentsChain to combine those summaries into a single global summary.

	- First, we specify the LLMChain to use for mapping each document to an individual summary:

```python
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain_text_splitters import CharacterTextSplitter

llm = ChatOpenAI(temperature=0)

# Map
map_template = """The following is a set of documents
{docs}
Based on this list of docs, please identify the main themes 
Helpful Answer:"""
map_prompt = PromptTemplate.from_template(map_template)
map_chain = LLMChain(llm=llm, prompt=map_prompt)
```

- The ReduceDocumentsChain handles taking the document mapping results and reducing them into a single output. It wraps a generic CombineDocumentsChain (like StuffDocumentsChain) but adds the ability to collapse documents before passing it to the CombineDocumentsChain if their cumulative size exceeds token_max. In this example, we can actually re-use our chain for combining our docs to also collapse our docs.

	- So if the cumulative number of tokens in our mapped documents exceeds 4000 tokens, then we'll recursively pass in the documents in batches of < 4000 tokens to our StuffDocumentsChain to create batched summaries. And once those batched summaries are cumulatively less than 4000 tokens, we'll pass them all one last time to the StuffDocumentsChain to create the final summary.


```python
# Run chain
reduce_chain = LLMChain(llm=llm, prompt=reduce_prompt)

# Takes a list of documents, combines them into a single string, and passes this to an LLMChain
combine_documents_chain = StuffDocumentsChain(
    llm_chain=reduce_chain, document_variable_name="docs"
)

# Combines and iteratively reduces the mapped documents
reduce_documents_chain = ReduceDocumentsChain(
    # This is final chain that is called.
    combine_documents_chain=combine_documents_chain,
    # If documents exceed context for `StuffDocumentsChain`
    collapse_documents_chain=combine_documents_chain,
    # The maximum number of tokens to group documents into.
    token_max=4000,
)

# Combining our map and reduce chains into one:

# Combining documents by mapping a chain over them, then combining results
map_reduce_chain = MapReduceDocumentsChain(
    # Map chain
    llm_chain=map_chain,
    # Reduce chain
    reduce_documents_chain=reduce_documents_chain,
    # The variable name in the llm_chain to put the documents in
    document_variable_name="docs",
    # Return the results of the map steps in the output
    return_intermediate_steps=False,
)

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=1000, chunk_overlap=0
)
split_docs = text_splitter.split_documents(docs)
```

- <b>Option 3. Refine</b>
	- RefineDocumentsChain is similar to map-reduce:

	- The refine documents chain constructs a response by looping over the input documents and iteratively updating its answer. For each document, it passes all non-document inputs, the current document, and the latest intermediate answer to an LLM chain to get a new answer.

```python
prompt_template = """Write a concise summary of the following:
{text}
CONCISE SUMMARY:"""
prompt = PromptTemplate.from_template(prompt_template)

refine_template = (
    "Your job is to produce a final summary\n"
    "We have provided an existing summary up to a certain point: {existing_answer}\n"
    "We have the opportunity to refine the existing summary"
    "(only if needed) with some more context below.\n"
    "------------\n"
    "{text}\n"
    "------------\n"
    "Given the new context, refine the original summary in Italian"
    "If the context isn't useful, return the original summary."
)
refine_prompt = PromptTemplate.from_template(refine_template)
chain = load_summarize_chain(
    llm=llm,
    chain_type="refine",
    question_prompt=prompt,
    refine_prompt=refine_prompt,
    return_intermediate_steps=True,
    input_key="input_documents",
    output_key="output_text",
)
result = chain({"input_documents": split_docs}, return_only_outputs=True)
```












### References 
- [AWS Bedrock Aug 2024: Analyze customer reviews using Amazon Bedrock](https://aws.amazon.com/blogs/machine-learning/analyze-customer-reviews-using-amazon-bedrock/)
- [Databricks Aug 2023: Automated Analysis of Product Reviews Using Large Language Models (LLMs)](https://www.databricks.com/blog/automated-analysis-product-reviews-using-large-language-models-llms)


###### Implementations
- [AWS Bedrock](https://github.com/aws-samples/analyze-customer-reviews-through-amazon-bedrock)
- [Databricks](https://github.com/databricks-industry-solutions/review-summarisation)
- [Langchain: Summarization Implementation](https://python.langchain.com/v0.1/docs/use_cases/summarization/)


###### Use Cases
- [Amazon: How Amazon continues to improve the customer reviews experience with generative AI](https://www.aboutamazon.com/news/amazon-ai/amazon-improves-customer-reviews-with-generative-ai)

