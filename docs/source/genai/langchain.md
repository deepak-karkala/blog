# Langchain

```bash
! pip install langchain_community tiktoken langchain-openai langchainhub chromadb langchain
```

```python
import os
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = <your-api-key>

os.environ['OPENAI_API_KEY'] = <your-api-key>
```

### Prompt Templates
- 

```python
from langchain.llms import OpenAI

# initialize the models
openai = OpenAI(
    model_name="text-davinci-003",
    openai_api_key="YOUR_API_KEY"
)

from langchain import PromptTemplate

template = """Answer the question based on the context below. If the
question cannot be answered using the information provided answer
with "I don't know".

Context: loreum ipsum ...
Question: {query}

Answer: """

prompt_template = PromptTemplate(
    input_variables=["query"],
    template=template
)
```

###### Few Shot Prompt Template
```python
# Few Shot Prompt Templates
from langchain import FewShotPromptTemplate

# create our examples
examples = [
    {
        "query": "How are you?",
        "answer": "I can't complain but sometimes I still do."
    }, {
        "query": "What time is it?",
        "answer": "It's time to get a watch."
    }
]

# create a example template
example_template = """
User: {query}
AI: {answer}
"""

# create a prompt example from above template
example_prompt = PromptTemplate(
    input_variables=["query", "answer"],
    template=example_template
)

# now break our previous prompt into a prefix and suffix
# the prefix is our instructions
prefix = """The following are exerpts from conversations with an AI
assistant. The assistant is typically sarcastic and witty, producing
creative  and funny responses to the users questions. Here are some
examples: 
"""
# and the suffix our user input and output indicator
suffix = """
User: {query}
AI: """

# now create the few shot prompt template
few_shot_prompt_template = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_prompt,
    prefix=prefix,
    suffix=suffix,
    input_variables=["query"],
    example_separator="\n\n"
)

```

###### Counting number of tokens
```python
from langchain.callbacks import get_openai_callback
def count_tokens(chain, query):
    with get_openai_callback() as cb:
        result = chain.run(query)
        print(f'Spent a total of {cb.total_tokens} tokens')

    return result
```


### Chains
-  chain is basically a pipeline that processes an input by using a specific combination of primitives. Intuitively, it can be thought of as a 'step' that performs a certain set of operations on an input and returns the result. They can be anything from a prompt-based pass through a LLM to applying a Python function to an text.

- Chains are divided in three types: Utility chains, Generic chains and Combine Documents chains.
	- Utility Chains: chains that are usually used to extract a specific answer from a llm with a very narrow purpose and are ready to be used out of the box.
	- Generic Chains: chains that are used as building blocks for other chains but cannot be used out of the box on their own.
	- Combine Documents chains


- <b>Utility Chains</b>
	- Utility chains usually follow the same basic structure: there is a prompt for constraining the llm to return a very specific type of response from a given query. We can ask the llm to create SQL queries, API calls and even create Bash commands on the fly.

- Example: LLMMathChain gives llms the ability to do math

```python
llm_math = LLMMathChain(llm=llm, verbose=True)
count_tokens(llm_math, "What is 13 raised to the .3432 power?")
```

<code>
> Entering new LLMMathChain chain...
What is 13 raised to the .3432 power?
```python
import math
print(math.pow(13, .3432))
```

Answer: 2.4116004626599237

> Finished chain.
Spent a total of 272 tokens
'Answer: 2.4116004626599237\n'
</code>

- The chain recieved a question in natural language and sent it to the llm. The llm returned a Python code which the chain compiled to give us an answer. A few questions arise.. How did the llm know that we wanted it to return Python code?

- The question we send as input to the chain is not the only input that the llm recieves. The input is inserted into a wider context, which gives precise instructions on how to interpret the input we send. This is called a prompt. Let's see what this chain's prompt is!

```python
print(llm_math.prompt.template)
```

<code>
You are GPT-3, and you can't do math.

You can do basic math, and your memorization abilities are impressive, but you can't do any complex calculations that a human could not do in their head. You also have an annoying tendency to just make up highly specific, but wrong, answers.

So we hooked you up to a Python 3 kernel, and now you can execute code. If anyone gives you a hard math problem, just use this format and we’ll take care of the rest:

Question: ${{Question with hard calculation.}}
```python
${{Code that prints what you need to know}}
```
```output
${{Output of your code}}
```
Answer: ${{Answer}}

Otherwise, use this simpler format:

Question: ${{Question without hard calculation}}
Answer: ${{Answer}}

Begin.

Question: What is 37593 * 67?

```python
print(37593 * 67)
```
```output
2518731
```
Answer: 2518731
</code>

- Another interesting point about this chain is that it not only runs an input through the llm but it later compiles Python code. Let's see exactly how this works.

```python
print(inspect.getsource(llm_math._call))
```

<code>
	def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        llm_executor = LLMChain(prompt=self.prompt, llm=self.llm)
        python_executor = PythonREPL()
        self.callback_manager.on_text(inputs[self.input_key], verbose=self.verbose)
        t = llm_executor.predict(question=inputs[self.input_key], stop=["```output"])
        self.callback_manager.on_text(t, color="green", verbose=self.verbose)
        t = t.strip()
        if t.startswith("```python"):
            code = t[9:-4]
            output = python_executor.run(code)
            self.callback_manager.on_text("\nAnswer: ", verbose=self.verbose)
            self.callback_manager.on_text(output, color="yellow", verbose=self.verbose)
            answer = "Answer: " + output
        elif t.startswith("Answer:"):
            answer = t
        else:
            raise ValueError(f"unknown format from LLM: {t}")
        return {self.output_key: answer}
</code>

- We now have the full picture of the chain: either the llm returns an answer (for simple math problems) or it returns Python code which we compile for an exact answer to harder problems.

- <b>Generic Chains</b>
	- Example: Formatting input text (regex) and paraphrasing to different style (LLM)

```python
def transform_func(inputs: dict) -> dict:
    text = inputs["text"]
    
    # replace multiple new lines and multiple spaces with a single one
    text = re.sub(r'(\r\n|\r|\n){2,}', r'\n', text)
    text = re.sub(r'[ \t]+', ' ', text)

    return {"output_text": text}

# TransformChain to format text
clean_extra_spaces_chain = TransformChain(input_variables=["text"], output_variables=["output_text"], transform=transform_func)

# paraphrasing to different style
template = """Paraphrase this text:
{output_text}
In the style of a {style}.
Paraphrase: """
prompt = PromptTemplate(input_variables=["style", "output_text"], template=template)
style_paraphrase_chain = LLMChain(llm=llm, prompt=prompt, output_key='final_output')

# combine them both to work as one integrated chain. For that we will use SequentialChain which is our third generic chain building block.
sequential_chain = SequentialChain(chains=[clean_extra_spaces_chain, style_paraphrase_chain], input_variables=['text', 'style'], output_variables=['final_output'])
```



```python
```


### Chat Models
- Chat Models are newer forms of language models that take messages in and output a message.


### Conversational Memory for LLMs
- The memory allows a Large Language Model (LLM) to remember previous interactions with the user. By default, LLMs are stateless — meaning each incoming query is processed independently of other interactions. The only thing that exists for a stateless agent is the current input, nothing else.
	- There are many applications where remembering previous interactions is very important, such as chatbots. Conversational memory allows us to do that.

```python
from langchain.chains import ConversationChain
# now initialize the conversation chain
conversation = ConversationChain(llm=llm)
print(conversation.prompt.template)
```

<code>
The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Current conversation:
{history}
Human: {input}
AI:
</code>

###### Forms of Conversational Memory

- <b>ConversationBufferMemory</b>
	- The ConversationBufferMemory is the most straightforward conversational memory in LangChain. The raw input of the past conversation between the human and AI is passed — in its raw form — to the {history} parameter. 

- Buffer saves every interaction in the chat history directly
```python
from langchain.chains.conversation.memory import ConversationBufferMemory
conversation_buf = ConversationChain(
    llm=llm,
    memory=ConversationBufferMemory()
)
print(conversation_buf.memory.buffer)
```
<code>
Human: Good morning AI!
AI:  Good morning! It's a beautiful day today, isn't it? How can I help you?
Human: My interest here is to explore the potential of integrating Large Language Models with external knowledge
AI:  Interesting! Large Language Models are a type of artificial intelligence that can process natural language and generate text. They can be used to generate text from a given context, or to answer questions about a given context. Integrating them with external knowledge can help them to better understand the context and generate more accurate results. Is there anything else I can help you with?
Human: I just want to analyze the different possibilities. What can you think of?
</code>

- Using ConversationBufferMemory, we very quickly use a lot of tokens and even exceed the context window limit of even the most advanced LLMs available today. Let’s take a look at other options that help remedy this.

**ConversationSummaryMemory**
- To avoid excessive token usage, we can use **ConversationSummaryMemory**. As the name would suggest, this form of memory summarizes the conversation history before it is passed to the {history} parameter.

```python
from langchain.chains.conversation.memory import ConversationSummaryMemory

conversation = ConversationChain(
	llm=llm,
	memory=ConversationSummaryMemory(llm=llm)
)
```

- <i>the summary memory initially uses far more tokens. However, as the conversation progresses, the summarization approach grows more slowly. In contrast, the buffer memory continues to grow linearly with the number of tokens in the chat.</i>

**ConversationBufferWindowMemory**
- The ConversationBufferWindowMemory acts in the same way as our earlier “buffer memory” but adds a window to the memory. Meaning that we only keep a given number of past interactions before “forgetting” them

- Although this method isn’t suitable for remembering distant interactions, it is good at limiting the number of tokens being used — a number that we can increase/decrease depending on our needs. For the longer conversation used in our earlier comparison, we can set k=6 and reach \~1.5K tokens per interaction after 27 total interactions:

```python
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
conversation = ConversationChain(
	llm=llm,
	memory=ConversationBufferWindowMemory(k=1)
)
```

**ConversationSummaryBufferMemory**
- The ConversationSummaryBufferMemory is a mix of the ConversationSummaryMemory and the ConversationBufferWindowMemory. It summarizes the earliest interactions in a conversation while maintaining the max_token_limit most recent tokens in their conversation

```python
conversation_sum_bufw = ConversationChain(
    llm=llm, memory=ConversationSummaryBufferMemory(
        llm=llm,
        max_token_limit=650
)
```

> Although requiring more tweaking on what to summarize and what to maintain within the buffer window, the ConversationSummaryBufferMemory does give us plenty of flexibility and is the only one of our memory types (so far) that allows us to remember distant interactions and store the most recent interactions in their raw — and most information-rich — form.

- other options
	- **ConversationKnowledgeGraphMemory** and 
	- **ConversationEntityMemory**

<code>
</code>

<code>
</code>

<code>
</code>

<code>
</code>

<code>
</code>






### Document Loaders

- Unstructured:
	- supports loading of text files, powerpoints, html, pdfs, images
	- all documents are split using specific knowledge about each document format to partition the document into semantic units (document elements). Chunking produces a sequence of CompositeElement, Table, or TableChunk elements. Each “chunk” is an instance of one of these three types.

- Webpages
	- Web: Uses urllib and BeautifulSoup to load and parse HTML web pages

- PDFs
	- PyPDF: Uses `pypdf` to load and parse PDFs

- Cloud Providers
	- AWS S3 Directory: Load documents from an AWS S3 directory
	- AWS S3 File: Load documents from an AWS S3 file

- Social Platforms
	- Twitter TwitterTweetLoader
	- Reddit RedditPostsLoader

- Messaging Services
	- Telegram	TelegramChatFileLoader
	- WhatsApp	WhatsAppChatLoader
	- Discord	DiscordChatLoader
	- Facebook Chat	FacebookChatLoader

- Productivity tools
	- Figma	FigmaFileLoader
	- Notion	NotionDirectoryLoader
	- Slack	SlackDirectoryLoader
	- Trello	TrelloLoader

- Common File Types
	- CSVLoader
	- JSONLoader
	- DirectoryLoader


```python
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()
```

#### How to load PDFs

###### 3 options
- (pypdf): Pass simple string representation of text that is embedded in a PDF to LLM
- (Unstructured): Layout analysis and extraction of text from images
- (Multimodal LLMs): Casting a PDF page to an image and passing it to LLM (with Multimodel support) directly


###### Option 1: Simple and fast text extraction
- If you are looking for a simple string representation of text that is embedded in a PDF, the method below is appropriate. It will return a list of Document objects-- one per page-- containing a single string of the page's text in the Document's page_content attribute. It will not parse text in images or scanned PDF pages. 
- <i>Under the hood it uses the pypydf Python library.</i>


```python
%pip install -qU pypdf

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(file_path)
pages = []
async for page in loader.alazy_load():
    pages.append(page)

# Vector search over PDFs
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

vector_store = InMemoryVectorStore.from_documents(pages, OpenAIEmbeddings())
docs = vector_store.similarity_search("What is LayoutParser?", k=2)
for doc in docs:
    print(f'Page {doc.metadata["page"]}: {doc.page_content[:300]}\n')
```


###### Option 2: Layout analysis and extraction of text from images
- If you require a more granular segmentation of text (e.g., into distinct paragraphs, titles, tables, or other structures) or require extraction of text from images, the method below is appropriate.
- It will return a list of Document objects, where each object represents a structure on the page.
- The Document's metadata stores the page number and other information related to the object (e.g., it might store table rows and columns in the case of a table object).

- Unstructured supports multiple parameters for PDF parsing:
	- strategy (e.g., "fast" or "hi-res")
	- API or local processing. You will need an API key to use the API.
	- The hi-res strategy provides support for document layout analysis and OCR

```python
from langchain_unstructured import UnstructuredLoader

loader = UnstructuredLoader(
    file_path=file_path,
    strategy="hi_res",
    partition_via_api=True,
    coordinates=True,
)
docs = []
for doc in loader.lazy_load():
    docs.append(doc)

# We can use the document metadata to recover content from a single page
first_page_docs = [doc for doc in docs if doc.metadata.get("page_number") == 1]

```

- <b>Extracting tables and other structures</b>
	- Each Document we load represents a structure, like a title, paragraph, or table.
	- Some structures may be of special interest for indexing or question-answering tasks. These structures may be:
		- Classified for easy identification;
		- Parsed into a more structured representation. 
	- Note that although the table text is collapsed into a single string in the document's content, the metadata contains a representation of its rows and columns:

```python
def render_page(doc_list: list, page_number: int, print_text=True) -> None:
    pdf_page = fitz.open(file_path).load_page(page_number - 1)
    page_docs = [
        doc for doc in doc_list if doc.metadata.get("page_number") == page_number
    ]
    segments = [doc.metadata for doc in page_docs]
    plot_pdf_with_boxes(pdf_page, segments)
    if print_text:
        for doc in page_docs:
            print(f"{doc.page_content}\n")

# Render Page 5
render_page(docs, 5)

# Extract Table in Page 5
segments = [
    doc.metadata
    for doc in docs
    if doc.metadata.get("page_number") == 5 and doc.metadata.get("category") == "Table"
]
```

- <b>Extracting text from specific sections</b>
	- Structures may have parent-child relationships -- for example, a paragraph might belong to a section with a title. If a section is of particular interest (e.g., for indexing) we can isolate the corresponding Document objects.

```python
# extract all text associated with the document's "Conclusion" section
conclusion_docs = []
parent_id = -1
for doc in docs:
    if doc.metadata["category"] == "Title" and "Conclusion" in doc.page_content:
        parent_id = doc.metadata["element_id"]
    if doc.metadata.get("parent_id") == parent_id:
        conclusion_docs.append(doc)

for doc in conclusion_docs:
    print(doc.page_content)
```

- <b>Extracting text from images</b>
	- OCR is run on images, enabling the extraction of text therein
	- text from the figure is extracted and incorporated into the content of the Document.



###### Option 3: Use of multimodal models
 - Many modern LLMs support inference over multimodal inputs (e.g., images). In some applications -- such as question-answering over PDFs with complex layouts, diagrams, or scans -- it may be advantageous to skip the PDF parsing, instead casting a PDF page to an image and passing it to a model directly.
 - This allows a model to reason over the two dimensional content on the page, instead of a "one-dimensional" string representation.

```python
%pip install -qU PyMuPDF pillow langchain-openai

import base64
import io

import fitz
from PIL import Image


def pdf_page_to_base64(pdf_path: str, page_number: int):
    pdf_document = fitz.open(pdf_path)
    page = pdf_document.load_page(page_number - 1)  # input is one-indexed
    pix = page.get_pixmap()
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

    buffer = io.BytesIO()
    img.save(buffer, format="PNG")

    return base64.b64encode(buffer.getvalue()).decode("utf-8")
base64_image = pdf_page_to_base64(file_path, 11)

from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4o-mini")


from langchain_core.messages import HumanMessage
query = "What is the name of the first step in the pipeline?"

message = HumanMessage(
    content=[
        {"type": "text", "text": query},
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
        },
    ],
)
response = llm.invoke([message])
print(response.content)
```





### Text Splitters
- Once you've loaded documents, you'll often want to transform them to better suit your application. The simplest example is you may want to split a long document into smaller chunks that can fit into your model's context window
	- When you want to deal with long pieces of text, it is necessary to split up that text into chunks. As simple as this sounds, there is a lot of potential complexity here. Ideally, you want to keep the semantically related pieces of text together. What "semantically related" means could depend on the type of text.

- Options
	- Recursive
		- RecursiveCharacterTextSplitter, RecursiveJsonSplitter
		- Splits on: A list of user defined characters
		- Recursively splits text. This splitting is trying to keep related pieces of text next to each other. This is the recommended way to start splitting text.
	- HTML
		- HTMLHeaderTextSplitter, HTMLSectionSplitter
		- Splits on: HTML specific characters
		- Splits text based on HTML-specific characters. Notably, this adds in relevant information about where that chunk came from (based on the HTML)
	- Markdown
		- MarkdownHeaderTextSplitter
		- Splits text based on Markdown-specific characters.
	- Code
		- Splits text based on characters specific to coding languages. 
	- Token
		- Splits text on tokens. There exist a few different ways to measure tokens.
	- Character
		- CharacterTextSplitter
		- Splits text based on a user defined character.
	- Semantic Chunker
		- SemanticChunker
		- First splits on sentences. Then combines ones next to each other if they are semantically similar enough.
	- AI21 Semantic Text Splitter
		- AI21SemanticTextSplitter
		- Identifies distinct topics that form coherent pieces of text and splits along those.


- Tip
	- Text splitter is the recommended one for generic text. 
		- It is parameterized by a list of characters. It tries to split on them in order until the chunks are small enough. The default list is ["\n\n", "\n", " ", ""].
		- This has the effect of trying to keep all paragraphs (and then sentences, and then words) together as long as possible, as those would generically seem to be the strongest semantically related pieces of text.

```python
# Split
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=300, 
    chunk_overlap=50)

# Make splits
splits = text_splitter.split_documents(docs)
```


### Embedding models
- Embedding models create a vector representation of a piece of text. By representing the text in this way, you can perform mathematical operations that allow you to do things like search for other pieces of text that are most similar in meaning. These natural language search capabilities underpin many types of context retrieval, where we provide an LLM with the relevant data it needs to effectively respond to a query.

- The Embeddings class is a class designed for interfacing with text embedding models. There are many different embedding model providers (OpenAI, Cohere, Hugging Face, etc) and local models, and this class is designed to provide a standard interface for all of them.

- The base Embeddings class in LangChain provides two methods: one for embedding documents and one for embedding a query. The former takes as input multiple texts, while the latter takes a single text. 

```python
from langchain_huggingface import HuggingFaceEmbeddings
embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

embeddings = embeddings_model.embed_documents(
    [
        "Hi there!",
        "Oh, hello!",
        "What's your name?",
        "My friends call me World",
        "Hello World!"
    ]
)

embedded_query = embeddings_model.embed_query("What was the name mentioned in the conversation?")
```


### Vector stores
- One of the most common ways to store and search over unstructured data is to embed it and store the resulting embedding vectors, and then at query time to embed the unstructured query and retrieve the embedding vectors that are 'most similar' to the embedded query. A vector store takes care of storing embedded data and performing vector search for you.

- Most vector stores can also store metadata about embedded vectors and support filtering on that metadata before similarity search, allowing you more control over returned documents.

- Free, open-source Options
	- Chroma, FAISS, Lance

```python
# Load the document, split it into chunks, embed each chunk and load it into the vector store.
raw_documents = TextLoader('sample.txt').load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
documents = text_splitter.split_documents(raw_documents)

from langchain_community.vectorstores import FAISS
db = FAISS.from_documents(documents, OpenAIEmbeddings())

# Similarity search:
# This will take incoming documents, create an embedding of them, and then find all documents with the most similar embedding
query = "What did the ..."
docs = db.similarity_search(query)

# Similarity search by vector
# search for documents similar to a given embedding vector using similarity_search_by_vector which accepts an embedding vector as a parameter instead of a string.
embedding_vector = OpenAIEmbeddings().embed_query(query)
docs = db.similarity_search_by_vector(embedding_vector)
```

- <b>Async Operations</b>
	- Vector stores are usually run as a separate service that requires some IO operations, and therefore they might be called asynchronously. That gives performance benefits as you don't waste time waiting for responses from external services. That might also be important if you work with an asynchronous framework, such as FastAPI

```python
# LangChain supports async operation on vector stores. 
docs = await db.asimilarity_search(query)
```


### Retrievers
- A retriever is an interface that returns documents given an unstructured query. It is more general than a vector store. A retriever does not need to be able to store documents, only to return (or retrieve) them. Retrievers can be created from vector stores, but are also broad enough to include Wikipedia search and Amazon Kendra.

- Retrievers accept a string query as input and return a list of Document's as output.





### Generator Models
- Multimodal Support
	- ChatAnthropic
	- ChatOpenAI
	- ChatVertexAI
	- ChatGoogleGenerativeAI
- Other Models
	- ChatMistralAI
	- ChatFireworks
	- ChatTogether
	- ChatGroq
	- ChatCohere
	- ChatBedrock
	- ChatHuggingFace
	- ChatNVIDIA
	- ChatOllama
	- ChatLlamaCpp
	- ChatAI21
	- ChatDatabricks



### Retrieval Augmentation
- We have two primary types of knowledge for LLMs. The parametric knowledge refers to everything the LLM learned during training and acts as a frozen snapshot of the world for the LLM.

- The second type of knowledge is source knowledge. This knowledge covers any information fed into the LLM via the input prompt. When we talk about retrieval augmentation, we’re talking about giving the LLM valuable source knowledge.


```python
from datasets import load_dataset
data = load_dataset("wikipedia", "20220301.simple", split='train[:10000]')

# Tokenizer
import tiktoken  # !pip install tiktoken
tokenizer = tiktoken.get_encoding('p50k_base')
# create the length function
def tiktoken_len(text):
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)
```

###### Creating Chunks
- Limiting input improves the LLM’s ability to follow instructions, reduces generation costs, and helps us get faster responses
- Provide users with more precise information sources as we can narrow down the information source to a smaller chunk of text.

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=400,
    chunk_overlap=20,
    length_function=tiktoken_len,
    separators=["\n\n", "\n", " ", ""]
)
```

###### Creating Embeddings
The vector embeddings are vital to retrieving relevant context for our LLM. We take the chunks of text we’d like to store in our knowledge base and encode each chunk into a vector embedding.


```python
from langchain.embeddings.openai import OpenAIEmbeddings

model_name = 'text-embedding-ada-002'
embed = OpenAIEmbeddings(
    document_model_name=model_name,
    query_model_name=model_name,
    openai_api_key=OPENAI_API_KEY
)

texts = [
    'this is the first chunk of text',
    'then another second chunk of text is here'
]
res = embed.embed_documents(texts)
```

###### Vector Database
- A vector database is a type of knowledge base that allows us to scale the search of similar embeddings to billions of records, manage our knowledge base by adding, updating, or removing records, and even do things like filtering.
```python
import pinecone

index_name = 'langchain-retrieval-augmentation'
pinecone.init(
        api_key="YOUR_API_KEY",  # find api key in console at app.pinecone.io
        environment="YOUR_ENV"  # find next to api key in console
)
# we create a new index
pinecone.create_index(
        name=index_name,
        metric='dotproduct',
        dimension=len(res[0]) # 1536 dim of text-embedding-ada-002
)
index = pinecone.GRPCIndex(index_name)
```

- The indexing process consists of us iterating through the data we’d like to add to our knowledge base, creating IDs, embeddings, and metadata — then adding these to the index.

```python
from tqdm.auto import tqdm
from uuid import uuid4

batch_limit = 100

texts = []
metadatas = []

for i, record in enumerate(tqdm(data)):
    # first get metadata fields for this record
    metadata = {
        'wiki-id': str(record['id']),
        'source': record['url'],
        'title': record['title']
    }
    # now we create chunks from the record text
    record_texts = text_splitter.split_text(record['text'])
    # create individual metadata dicts for each chunk
    record_metadatas = [{
        "chunk": j, "text": text, **metadata
    } for j, text in enumerate(record_texts)]
    # append these to current batches
    texts.extend(record_texts)
    metadatas.extend(record_metadatas)
    # if we have reached the batch_limit we can add texts
    if len(texts) >= batch_limit:
        ids = [str(uuid4()) for _ in range(len(texts))]
        embeds = embed.embed_documents(texts)
        index.upsert(vectors=zip(ids, embeds, metadatas))
        texts = []
        metadatas = []
```

###### Vector Store and Querying
```python
from langchain.vectorstores import Pinecone

text_field = "text"
# switch back to normal index for langchain
index = pinecone.Index(index_name)
vectorstore = Pinecone(
    index, embed.embed_query, text_field
)

query = "who was Benito Mussolini?"
vectorstore.similarity_search(
    query,  # our search query
    k=3  # return 3 most relevant docs
)
```

###### Generative Question Answering

```python
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# completion llm
llm = ChatOpenAI(
    openai_api_key=OPENAI_API_KEY,
    model_name='gpt-3.5-turbo',
    temperature=0.0
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)
qa.run(query)
```


###### Retrieval Agents (Adding conversational ability to RAG)
- Conversational agents can struggle with data freshness, knowledge about specific domains, or accessing internal documentation. By coupling agents with retrieval augmentation tools we no longer have these problems. One the other side, using "naive" retrieval augmentation without the use of an agent means we will retrieve contexts with every query. Again, this isn't always ideal as not every query requires access to external knowledge. Merging these methods gives us the best of both worlds.

```python
from langchain.agents import Tool

tools = [
    Tool(
        name='Knowledge Base',
        func=qa.run,
        description=(
            'use this tool when answering general knowledge queries to get '
            'more information about the topic'
        )
    )
]

from langchain.agents import initialize_agent
agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3,
    early_stopping_method='generate',
    memory=conversational_memory
)
```

- adding citations to the response, allowing a user to see where the information is coming from. We can do this using a slightly different version of the RetrievalQA chain called RetrievalQAWithSourcesChain.

```python
from langchain.chains import RetrievalQAWithSourcesChain
qa_with_sources = RetrievalQAWithSourcesChain.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)
```


### Agents
- What are Agents?: We can think of agents as enabling “tools” for LLMs. Like how a human would use a calculator for maths or perform a Google search for information — agents allow an LLM to do the same thing. Using agents, an LLM can write and execute Python code. It can search for information and even query a SQL database.

- Agents and Tools
	- To use agents, we require three things:
		- A base LLM,
		- A tool that we will be interacting with,
		- An agent to control the interaction.

- create a new calculator tool from the existing llm_math chain
```python
from langchain.chains import LLMMathChain
from langchain.agents import Tool

llm_math = LLMMathChain(llm=llm)

# initialize the math tool
math_tool = Tool(
    name='Calculator',
    func=llm_math.run,
    description='Useful for when you need to answer questions about math.'
)
# when giving tools to LLM, we must pass as list of tools
tools = [math_tool]


# a prebuilt llm_math tool does the same thing, we can only follow this second approach if a prebuilt tool for our use case exists.
from langchain.agents import load_tools
tools = load_tools(
    ['llm-math'],
    llm=llm
)
```

- To initialize a simple agent, we can do the following. 
```python
from langchain.agents import initialize_agent

# The agent used here is a "zero-shot-react-description" agent. Zero-shot means the agent functions on the current action only — it has no memory. It uses the ReAct framework to decide which tool to use, based solely on the tool’s description
zero_shot_agent = initialize_agent(
    agent="zero-shot-react-description",
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3
)
zero_shot_agent("what is (4.5*2.1)^2.2?")

# Let’s add a plain and simple LLM tool
prompt = PromptTemplate(
    input_variables=["query"],
    template="{query}"
)
llm_chain = LLMChain(llm=llm, prompt=prompt)
# initialize the LLM tool
llm_tool = Tool(
    name='Language Model',
    func=llm_chain.run,
    description='use this tool for general purpose queries and logic'
)
tools.append(llm_tool)
```


###### Agent Types

**Zero Shot ReAct**
- we use this agent to perform “zero-shot” tasks on some input. That means the agent considers one single interaction with the agent — it will have no memory.
- ReAct framework: LLM could cycle through Reasoning and Action steps. Enabling a multi-step process for identifying answers.

```python
tools = load_tools(
    ["llm-math"], 
    llm=llm
)
# add our custom SQL db tool
tools.append(sql_tool)

zero_shot_agent = initialize_agent(
    agent="zero-shot-react-description", 
    tools=tools, 
    llm=llm,
    verbose=True,
    max_iterations=3,
)

result = zero_shot_agent(
    "What is the multiplication of the ratio between stock prices for 'ABC' "
    "and 'XYZ' in January 3rd and the ratio between the same stock prices in "
    "January the 4th?"
)
```

> At each step, there is a Thought that results in a chosen Action and Action Input. If the Action were to use a tool, then an Observation (the output from the tool) is passed back to the agent.

- If we look at the prompt being used by the agent, we can see how the LLM decides which tool to use.

```python
print(zero_shot_agent.agent.llm_chain.prompt.template)
```

<code>
Answer the following questions as best you can. You have access to the following tools:

Calculator: Useful for when you need to answer questions about math.
Stock DB: Useful for when you need to answer questions about stocks and their prices.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [Calculator, Stock DB]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
</code>

> These tools and the thought process separate agents from chains in LangChain. Whereas a chain defines an immediate input/output process, the logic of agents allows a step-by-step thought process. The advantage of this step-by-step process is that the LLM can work through multiple reasoning steps or tools to produce a better answer.

> The agent_scratchpad is where we add every thought or action the agent has already performed. All thoughts and actions (within the current agent executor chain) can then be accessed by the next thought-action-observation loop, enabling continuity in agent actions.


**Conversational ReAct**
- The zero-shot agent works well but lacks conversational memory. This lack of memory can be problematic for chatbot-type use cases that need to remember previous interactions in a conversation. Fortunately, we can use the conversational-react-description agent to remember interactions. We can think of this agent as the same as our previous Zero Shot ReAct agent, but with conversational memory.

- unlike our zero-shot agent, we can now ask follow-up questions. 
```python
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(memory_key="chat_history")

conversational_agent = initialize_agent(
    agent='conversational-react-description', 
    tools=tools, 
    llm=llm,
    verbose=True,
    max_iterations=3,
    memory=memory,
)
```

**ReAct Docstore**
- Another common agent is the react-docstore agent. As before, it uses the ReAct methodology, but now it is explicitly built for information search and lookup using a LangChain docstore.

- LangChain docstores allow us to store and retrieve information using traditional retrieval methods. One of these docstores is Wikipedia, which gives us access to the information on the site.

- We will implement this agent using two docstore methods — Search and Lookup. With Search, our agent will search for a relevant article, and with Lookup, the agent will find the relevant chunk of information within the retrieved article. To initialize these two tools, we do:

```python
from langchain import Wikipedia
from langchain.agents.react.base import DocstoreExplorer

docstore=DocstoreExplorer(Wikipedia())
tools = [
    Tool(
        name="Search",
        func=docstore.search,
        description='search wikipedia'
    ),
    Tool(
        name="Lookup",
        func=docstore.lookup,
        description='lookup a term in wikipedia'
    )
]
docstore_agent = initialize_agent(
    tools, 
    llm, 
    agent="react-docstore", 
    verbose=True,
    max_iterations=3
)
docstore_agent("What were Archimedes' last words?")
```


**Self-Ask With Search**
- This agent is the first you should consider when connecting an LLM with a search engine. The agent will perform searches and ask follow-up questions as often as required to get a final answer. We initialize the agent like so:

```python
from langchain import SerpAPIWrapper

# initialize the search chain
search = SerpAPIWrapper(serpapi_api_key='serp_api_key')

# create a search tool
tools = [
    Tool(
        name="Intermediate Answer",
        func=search.run,
        description='google search'
    )
]

# initialize the search enabled agent
self_ask_with_search = initialize_agent(
    tools,
    llm,
    agent="self-ask-with-search",
    verbose=True
)
self_ask_with_search(
    "who lived longer; Plato, Socrates, or Aristotle?"
)
```


### Custom tools for agents
> At their core, tools are objects that consume some input, typically in the format of a string (text), and output some helpful information as a string. In reality, they are little more than a simple function that we’d find in any code. The only difference is that tools take input from an LLM and feed their output to an LLM.


###### Simple Calculator Tool
- The tool is a simple calculator that calculates a circle’s circumference based on the circle’s radius.

```python
from langchain.tools import BaseTool
from math import pi
from typing import Union
  
# The description is a natural language description of the tool the LLM uses to decide whether it needs to use it. Tool descriptions should be very explicit on what they do, when to use them, and when not to use them.

# Following this, we have two methods, _run and _arun. When a tool is used, the _run method is called by default. The _arun method is called when a tool is to be used asynchronously.

class CircumferenceTool(BaseTool):
      name = "Circumference calculator"
      description = "use this tool when you need to calculate a circumference using the radius of a circle"

    def _run(self, radius: Union[int, float]):
        return float(radius)*2.0*pi

    def _arun(self, radius: int):
        raise NotImplementedError("This tool does not support async")

tools = [CircumferenceTool()]

# initialize agent with tools
agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3,
    early_stopping_method='generate',
    memory=conversational_memory
)
agent("can you calculate the circumference of a circle that has a radius of 7.81mm")
```

- LLMs are generally bad at math, but that doesn’t stop them from trying to do math. The problem is due to the LLM’s overconfidence in its mathematical ability. To fix this, we must tell the model that it cannot do math. We will add a single sentence that tells the model that it is “terrible at math” and should never attempt to do it.

- With this added to the original prompt text, we create a new prompt using agent.agent.create_prompt — this will create the correct prompt structure for our agent, including tool descriptions. Then, we update agent.agent.llm_chain.prompt.

```python
sys_msg = """
....

Unfortunately, the Assistant is terrible at maths. When provided with math questions, no matter how simple, assistant always refers to its trusty tools and absolutely does NOT try to answer math questions by itself

....
"""
new_prompt = agent.agent.create_prompt(
    system_message=sys_msg,
    tools=tools
)
agent.agent.llm_chain.prompt = new_prompt
```

###### Advanced Tool Usage

- Example: **ImageCaptionTool**
	- Taking inspiration from the HuggingGPT paper [1], we will take an existing open-source expert model that has been trained for a specific task that our LLM cannot do.

	- That model will be the Salesforce/blip-image-captioning-large model hosted on Hugging Face. This model takes an image and describes it, something that we cannot do with our LLM.

```python
# !pip install transformers
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration


# specify model to be used
hf_model = "Salesforce/blip-image-captioning-large"
# use GPU if it's available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# preprocessor will prepare images for the model
processor = BlipProcessor.from_pretrained(hf_model)
# then we initialize the model itself
model = BlipForConditionalGeneration.from_pretrained(hf_model).to(device)

desc = (
    "use this tool when given the URL of an image that you'd like to be "
    "described. It will return a simple caption describing the image."

)
class ImageCaptionTool(BaseTool):
    name = "Image captioner"
    description = desc
    
    def _run(self, url: str):
        # download the image and convert to PIL object
        image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')
        # preprocess the image
        inputs = processor(image, return_tensors="pt").to(device)
        # generate the caption
        out = model.generate(**inputs, max_new_tokens=20)
        # get the caption
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption
    
    def _arun(self, query: str):
        raise NotImplementedError("This tool does not support async")

tools = [ImageCaptionTool()]
agent(f"What does this image show?\n{img_url}")
```

### Streaming

###### LLM Streaming to Stdout
- The simplest form of streaming is to simply "print" the tokens as they're generated. To set this up we need to initialize an LLM (one that supports streaming, not all do) with two specific parameters:
    - streaming=True, to enable streaming
    - callbacks=[SomeCallBackHere()], where we pass a LangChain callback class (or list containing multiple).

- The streaming parameter is self-explanatory. The callbacks parameter and callback classes less so — essentially they act as little bits of code that do something as each token from our LLM is generated. As mentioned, the simplest form of streaming is to print the tokens as they're being generated, like with the StreamingStdOutCallbackHandler

```python
llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0.0,
    model_name="gpt-3.5-turbo",
    streaming=True,  # ! important
    callbacks=[StreamingStdOutCallbackHandler()]  # ! important
)
```

- things begin to get much more complicated as soon as we begin using agents: **custom callback handler**
```python
import sys
class CallbackHandler(StreamingStdOutCallbackHandler):
    def __init__(self):
        self.content: str = ""
        self.final_answer: bool = False

    def on_llm_new_token(self, token: str, **kwargs: any) -> None:
        self.content += token
        if "Final Answer" in self.content:
            # now we're in the final answer section, but don't print yet
            self.final_answer = True
            self.content = ""
        if self.final_answer:
            if '"action_input": "' in self.content:
                if token not in ["}"]:
                    sys.stdout.write(token)  # equal to `print(token, end="")`
                    sys.stdout.flush()

agent.agent.llm_chain.llm.callbacks = [CallbackHandler()]
```

###### Using FastAPI with Agents
- In most cases we'll be placing our LLMs, Agents, etc behind something like an API. Let's add that into the mix and see how we can implement streaming for agents with FastAPI.

- Unlike with our StdOut streaming, we now need to send our tokens to a generator function that feeds those tokens to FastAPI via a StreamingResponse object. To handle this we need to use async code, otherwise our generator will not begin emitting anything until after generation is already complete.

- The Queue is accessed by our callback handler, as as each token is generated, it puts the token into the queue. Our generator function asyncronously checks for new tokens being added to the queue. As soon as the generator sees a token has been added, it gets the token and yields it to our StreamingResponse.

- To see it in action, we'll define a stream requests function called get_stream

```python
def get_stream(query: str):
    s = requests.Session()
    with s.get(
        "http://localhost:8000/chat",
        stream=True,
        json={"text": query}
    ) as r:
        for line in r.iter_content():
            print(line.decode("utf-8"), end="")

get_stream("hi there!")
```

### Multi-Query for RAG
- Distance-based vector database retrieval embeds (represents) queries in high-dimensional space and finds similar embedded documents based on a distance metric. But, retrieval may produce different results with subtle changes in query wording, or if the embeddings do not capture the semantics of the data well. Prompt engineering / tuning is sometimes done to manually address these problems, but can be tedious.

- The **MultiQueryRetriever** automates the process of prompt tuning by using an LLM to generate multiple queries from different perspectives for a given user input query. For each query, it retrieves a set of relevant documents and takes the unique union across all queries to get a larger set of potentially relevant documents. By generating multiple perspectives on the same question, the MultiQueryRetriever can mitigate some of the limitations of the distance-based retrieval and get a richer set of results.

```python
from langchain.retrievers.multi_query import MultiQueryRetriever
retriever = MultiQueryRetriever.from_llm(
    retriever=vectorstore.as_retriever(), llm=llm
)

# We set logging so that we can see the queries as they're generated by our LLM
# Set logging for the queries
import logging
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

# To query with our multi-query retriever we call the get_relevant_documents method.
question = "tell me about llama 2?"
docs = retriever.get_relevant_documents(query=question)
len(docs)

# From this we get a variety of docs retrieved by each of our queries independently. By default the retriever is returning 3 docs for each query — totalling 9 documents — however, as there is some overlap we actually return 6 unique docs.

# Adding the Generation in RAG
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

QA_PROMPT = PromptTemplate(
    input_variables=["query", "contexts"],
    template="""You are a helpful assistant who answers user queries using the
    contexts provided. If the question cannot be answered using the information
    provided say "I don't know".

    Contexts:
    {contexts}

    Question: {query}""",
)

# Chain
qa_chain = LLMChain(llm=llm, prompt=QA_PROMPT)
out = qa_chain(
    inputs={
        "query": question,
        "contexts": "\n---\n".join([d.page_content for d in docs])
    }
)
out["text"]
```

**Chaining Everything with a SequentialChain**

- We can pull together the logic above into a function or set of methods, whatever is prefered — however if we'd like to use LangChain's approach to this we must "chain" together multiple chains. The first retrieval component is (1) not a chain per se, and (2) requires processing of the output. To do that, and fit with LangChain's "chaining chains" approach, we setup the retrieval component within a TransformChain:

```python
from langchain.chains import TransformChain

def retrieval_transform(inputs: dict) -> dict:
    docs = retriever.get_relevant_documents(query=inputs["question"])
    docs = [d.page_content for d in docs]
    docs_dict = {
        "query": inputs["question"],
        "contexts": "\n---\n".join(docs)
    }
    return docs_dict

retrieval_chain = TransformChain(
    input_variables=["question"],
    output_variables=["query", "contexts"],
    transform=retrieval_transform
)

# Now we chain this with our generation step using the SequentialChain:
from langchain.chains import SequentialChain

rag_chain = SequentialChain(
    chains=[retrieval_chain, qa_chain],
    input_variables=["question"],  # we need to name differently to output "query"
    output_variables=["query", "contexts", "text"]
)

# Then we perform the full RAG pipeline:
out = rag_chain({"question": question})
out["text"]
```

###### Custom Multiquery
- We'll try this with custom prompt, to encourage more variety in search queries.

```python
from typing import List
from langchain.chains import LLMChain
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser


# Output parser will split the LLM result into a list of queries
class LineList(BaseModel):
    # "lines" is the key (attribute name) of the parsed output
    lines: List[str] = Field(description="Lines of text")


class LineListOutputParser(PydanticOutputParser):
    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> LineList:
        lines = text.strip().split("\n")
        return LineList(lines=lines)


output_parser = LineListOutputParser()

template = """
Your task is to generate 3 different search queries that aim to
answer the user question from multiple perspectives. The user questions
are focused on Large Language Models, Machine Learning, and related
disciplines.
Each query MUST tackle the question from a different viewpoint, we
want to get a variety of RELEVANT search results.
Provide these alternative questions separated by newlines.
Original question: {question}
"""

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template=template,
)
llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
# Chain
llm_chain = LLMChain(llm=llm, prompt=QUERY_PROMPT, output_parser=output_parser)

# Run
retriever = MultiQueryRetriever(
    retriever=vectorstore.as_retriever(), llm_chain=llm_chain, parser_key="lines"
)  # "lines" is the key (attribute name) of the parsed output

# Results
docs = retriever.get_relevant_documents(
    query=question
)
len(docs)

retrieval_chain = TransformChain(
    input_variables=["question"],
    output_variables=["query", "contexts"],
    transform=retrieval_transform
)

rag_chain = SequentialChain(
    chains=[retrieval_chain, qa_chain],
    input_variables=["question"],  # we need to name differently to output "query"
    output_variables=["query", "contexts", "text"]
)
```

### LangChain Expression Language (LCEL)

- The LangChain Expression Language (LCEL) is an abstraction of some interesting Python concepts into a format that enables a "minimalist" code layer for building chains of LangChain components.

- LCEL comes with strong support for:
    - Superfast development of chains.
    - Advanced features such as streaming, async, parallel execution, and more.
    - Easy integration with LangSmith and LangServe. 

```python
# To understand LCEL syntax let's first build a simple chain using the traditional LangChain syntax
from langchain.chains import LLMChain
chain = LLMChain(
    prompt=prompt,
    llm=model,
    output_parser=output_parser
)
# and run
out = chain.run(topic="Artificial Intelligence")

# With LCEL we create our chain differently using pipe operators (|) rather than Chains
lcel_chain = prompt | model | output_parser

# and run
out = lcel_chain.invoke({"topic": "Artificial Intelligence"})
print(out)

# The syntax here is not typical for Python but uses nothing but native Python. Our | operator simply takes output from the left and feeds it into the function on the right.
```

**How the Pipe Operator Works**
- To understand what is happening with LCEL and the pipe operator we create our own pipe-compatible functions.
- When the Python interpreter sees the | operator between two objects (like a | b) it attempts to feed object a into the __or__ method of object b. That means these patterns are equivalent.

```python
# object approach
chain = a.__or__(b)
chain("some input")

# pipe approach
chain = a | b
chain("some input")
```

- With that in mind, we can build a Runnable class that consumes a function and turns it into a function that can be chained with other functions using the pipe operator |.
```python
class Runnable:
    def __init__(self, func):
        self.func = func

    def __or__(self, other):
        def chained_func(*args, **kwargs):
            # the other func consumes the result of this func
            return other(self.func(*args, **kwargs))
        return Runnable(chained_func)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

def add_five(x):
    return x + 5

def multiply_by_two(x):
    return x * 2

# wrap the functions with Runnable
add_five = Runnable(add_five)
multiply_by_two = Runnable(multiply_by_two)

# run them using the object approach
chain = add_five.__or__(multiply_by_two)

# let's try using the pipe operator | to chain them together:
# chain the runnable functions together
chain = add_five | multiply_by_two

# invoke the chain
chain(3) # we should return 16
```
- at its core, this is the pipe logic that LCEL uses when chaining together components.

###### Runnables
- When working with LCEL we may find that we need to modify the flow of values, or the values themselves as they are passed between components — for this, we can use runnables.

```python
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough
)

retriever_a = vecstore_a.as_retriever()
retriever_b = vecstore_b.as_retriever()

prompt_str = """Answer the question below using the context:

Context: {context}

Question: {question}

Answer: """
prompt = ChatPromptTemplate.from_template(prompt_str)

# We use two new objects here, RunnableParallel and RunnablePassthrough. The RunnableParallel object allows us to define multiple values and operations, and run them all in parallel. Here we call retriever_a using the input to our chain (below), and then pass the results from retriever_a to the next component in the chain via the "context" parameter.

# The RunnablePassthrough object is used as a "passthrough" take takes any input to the current component (retrieval) and allows us to provide it in the component output via the "question" key.

retrieval = RunnableParallel(
    {
        "context_a": retriever_a, "context_b": retriever_b,
        "question": RunnablePassthrough()
    }
)

chain = retrieval | prompt | model | output_parser
```


###### Runnable Lambdas
- The RunnableLambda is a LangChain abstraction that allows us to turn Python functions into pipe-compatible functions, similar to the Runnable class we created 

```python
from langchain_core.runnables import RunnableLambda

def add_five(x):
    return x + 5

def multiply_by_two(x):
    return x * 2

# wrap the functions with RunnableLambda
add_five = RunnableLambda(add_five)
multiply_by_two = RunnableLambda(multiply_by_two)

# As with our earlier Runnable abstraction, we can use | operators to chain together our RunnableLambda abstractions.
chain = add_five | multiply_by_two

# Unlike our Runnable abstraction, we cannot run the RunnableLambda chain by calling it directly, instead we must call chain.invoke
chain.invoke(3)
```

- Naturally, we can feed custom functions into our chains using this approach. Let's try a short chain and see where we might want to insert a custom function

```python
prompt_str = "Tell me an short fact about {topic}"
prompt = ChatPromptTemplate.from_template(prompt_str)

chain = prompt | model | output_parser

chain.invoke({"topic": "Artificial Intelligence"})

# The returned text always includes the initial "Here's a short fact about ...\n\n" — let's add a function to split on double newlines "\n\n" and only return the fact itself.
def extract_fact(x):
    if "\n\n" in x:
        return "\n".join(x.split("\n\n")[1:])
    else:
        return x
    
get_fact = RunnableLambda(extract_fact)
chain = prompt | model | output_parser | get_fact
```








### References
- [Learn RAG From Scratch – Python AI Tutorial from a LangChain Engineer](https://www.youtube.com/watch?v=sVcwVQRHIc8&t=37s)