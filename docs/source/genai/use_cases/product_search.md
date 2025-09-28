# Product Search

### How LLM can help ?
- Without the precise terms to locate a specific widget or an article on a narrowly defined topic, consumers find themselves frustrated scrolling through lists of items that just aren’t quite right.

- Using LLMs, we can task a model with poring over product descriptions, written content or the transcripts associated with audio recordings and responding to user searches with suggestions for things relevant to their prompts. Users don’t need the precise terms to find what they are looking for, just a general description with which the LLM can orient itself to their needs. The end result is a powerful new experience that leaves users feeling as if they have received personalized, expert guidance as they engage the site.



### Should the embedding model be fine tuned ?
- to be effective in searching a specific body of documents, the model doesn’t even need to be trained specifically on it. If trained on even a relatively small volume of content, these models can perform content summarization and generation tasks with impressive acumen.

- <i>But with fine-tuning, we can tweak the orientation of the model to the specific content against which it is intended to be engaged. By taking a pre-trained model and engaging it in additional rounds of training on the product descriptions, product reviews, written posts, transcripts, etc. that make up a specific site, the ability of model to respond to user prompts in a manner more consistent with this content is improved, making it a worthwhile step for many organizations to perform.</i>


###### how does one label search results for fine-tuning ?
- All you need is a set of queries and the results returned for them. 
- This data set doesn’t need to be super large for it to be effective though the more search results available the better. 
- <i>A human then must assign a numerical score to each search result to indicate its relevance to the search phrase. While this can be made complicated, you will likely find very good results by simply assigning relevant search results a value of 1.0, irrelevant search results a value of 0.0, and partially relevant results a value somewhere in between.</i>


### Reference Implementation
- [Enhancing Product Search with Large Language Models (LLMs)](https://www.databricks.com/blog/enhancing-product-search-large-language-models-llms.html)
	- Dataset
		-  Wayfair Annotation Dataset (WANDS). This dataset provides descriptive text for 42,000+ products on the Wayfair website and 233K labeled results generated from 480 searches.
	- Model
		- open source model from Hugging Face
		- Using an open source model from Hugging Face, we first assemble an out-of-the box search with no fine-tuning and are able to deliver surprisingly good results. We then fine-tune the model using our labeled search results, boosting search performance considerably.
	- Deployment
		- These models are then packaged for deployment as a microservice hosted with Databricks model serving.
	- [Notebooks](https://d1r5llqwmkrl74.cloudfront.net/notebooks/RCG/product-search/index.html#product-search_1.html)









