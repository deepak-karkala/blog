# Agents

### References
- [Pinecone: Superpower LLMs with Conversational Agents](https://www.pinecone.io/learn/series/langchain/langchain-agents/)
- [LLM Powered Autonomous Agents](https://lilianweng.github.io/posts/2023-06-23-agent/)


### What
- We can think of agents as enabling “tools” for LLMs. Like how a human would use a calculator for maths or perform a Google search for information — agents allow an LLM to do the same thing.
- The potentiality of LLM extends beyond generating well-written copies, stories, essays and programs; it can be framed as a powerful general problem solver.
- Examples
	- [AutoGPT](https://github.com/Significant-Gravitas/Auto-GPT),
	- [GPT-Engineer](https://github.com/AntonOsika/gpt-engineer) and
	- [BabyAGI](https://github.com/yoheinakajima/babyagi)



### Agent System Overview
- Planning
	- Subgoal and decomposition: The agent breaks down large tasks into smaller, manageable subgoals, enabling efficient handling of complex tasks.
	- Reflection and refinement: The agent can do self-criticism and self-reflection over past actions, learn from mistakes and refine them for future steps, thereby improving the quality of final results.

- Memory
	- Short-term memory: I would consider all the in-context learning (See Prompt Engineering) as utilizing short-term memory of the model to learn.
	- Long-term memory: This provides the agent with the capability to retain and recall (infinite) information over extended periods, often by leveraging an external vector store and fast retrieval.

- Tool use
	- The agent learns to call external APIs for extra information that is missing from the model weights (often hard to change after pre-training), including current information, code execution capability, access to proprietary information sources and more.


### Planning
- A complicated task usually involves many steps. An agent needs to know what they are and plan ahead.

###### Task Decomposition
- [Chain of thought  CoT](https://arxiv.org/abs/2201.11903)
	- The model is instructed to “think step by step” to utilize more test-time computation to decompose hard tasks into smaller and simpler steps. CoT transforms big tasks into multiple manageable tasks and shed lights into an interpretation of the model’s thinking process.

	- generates a sequence of short sentences to describe reasoning logics step by step, known as reasoning chains or rationales, to eventually lead to the final answer. The benefit of CoT is more pronounced for complicated reasoning tasks, while using large models (e.g. with more than 50B parameters). Simple tasks only benefit slightly from CoT prompting.

	- Types of CoT prompts
		- <b>Few-shot CoT</b>. It is to prompt the model with a few demonstrations, each containing manually written (or model-generated) high-quality reasoning chains.
		- <b>Zero-shot CoT</b>. Use natural language statement like Let's think step by step to explicitly encourage the model to first generate reasoning chains and then to prompt with 
			- [Therefore, the answer is](https://arxiv.org/abs/2205.11916) to produce answers or
			- [Let's work this out it a step by step to be sure we have the right answer](https://arxiv.org/abs/2211.01910)

- [Tree of Thoughts](https://arxiv.org/abs/2305.10601)
	- extends CoT by exploring multiple reasoning possibilities at each step. It first decomposes the problem into multiple thought steps and generates multiple thoughts per step, creating a tree structure. The search process can be BFS (breadth-first search) or DFS (depth-first search) with each state evaluated by a classifier (via a prompt) or majority vote.

	- Task decomposition can be done
		- (1) by LLM with simple prompting like "Steps for XYZ.\n1.", "What are the subgoals for achieving XYZ?",
		- (2) by using task-specific instructions; e.g. "Write a story outline." for writing a novel, or
		- (3) with human inputs.


###### Self-Reflection
- Self-reflection is a vital aspect that allows autonomous agents to improve iteratively by refining past action decisions and correcting previous mistakes. It plays a crucial role in real-world tasks where trial and error are inevitable.

- [ReAct](https://arxiv.org/abs/2210.03629)
	- integrates reasoning and acting within LLM by extending the action space to be a combination of task-specific discrete actions and the language space. The former enables LLM to interact with the environment (e.g. use Wikipedia search API), while the latter prompting LLM to generate reasoning traces in natural language.

	- In both experiments on knowledge-intensive tasks and decision-making tasks, ReAct works better than the Act-only baseline where Thought: … step is removed.

	- The ReAct prompt template incorporates explicit steps for LLM to think, roughly formatted as:

	```python
	Thought: ...
	Action: ...
	Observation: ...
	... (Repeated many times)
	```






### How to use
- To use agents, we require three things:
	- A base LLM,
	- A tool that we will be interacting with,
	- An agent to control the interaction.

```python
# Base LLM
from langchain import OpenAI
llm = OpenAI(
    openai_api_key="OPENAI_API_KEY",
    temperature=0,
    model_name="text-davinci-003"
)

# initialize the calculator tool.
# When initializing tools, we either create a custom tool or load a prebuilt tool.
# In either case, the “tool” is a utility chain given a tool name and description
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


# Or use a prebuilt tool
from langchain.agents import load_tools
tools = load_tools(
    ['llm-math'],
    llm=llm
)

# initialize a simple agent
# The agent used here is a "zero-shot-react-description" agent. Zero-shot means the agent functions on the current action only — it has no memory. It uses the ReAct framework to decide which tool to use, based solely on the tool’s description
from langchain.agents import initialize_agent
zero_shot_agent = initialize_agent(
    agent="zero-shot-react-description",
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3
)
```

### Agent Types

###### Zero Shot ReAct
- we use this agent to perform “zero-shot” tasks on some input. That means the agent considers one single interaction with the agent — it will have no memory.

	- At each step, there is a Thought that results in a chosen Action and Action Input. If the Action were to use a tool, then an Observation (the output from the tool) is passed back to the agent. 
	
	- We first tell the LLM the tools it can use (Calculator and Stock DB). Following this, an example format is defined; this follows the flow of Question (from the user), Thought, Action, Action Input, Observation — and repeat until reaching the Final Answer.

	- These tools and the thought process separate agents from chains in LangChain.

	- Whereas a chain defines an immediate input/output process, the logic of agents allows a step-by-step thought process. The advantage of this step-by-step process is that the LLM can work through multiple reasoning steps or tools to produce a better answer.

- The <b>agent_scratchpad</b> is where we add every thought or action the agent has already performed. All thoughts and actions (within the current agent executor chain) can then be accessed by the next thought-action-observation loop, enabling continuity in agent actions.


###### Conversational ReAct
- The zero-shot agent works well but lacks conversational memory. This lack of memory can be problematic for chatbot-type use cases that need to remember previous interactions in a conversation.

- we can use the conversational-react-description agent to remember interactions. We can think of this agent as the same as our previous Zero Shot ReAct agent, but with conversational memory.

```python
# To initialize the agent, we first need to initialize the memory we’d like to use
from langchain.memory import ConversationBufferMemory
memory = ConversationBufferMemory(memory_key="chat_history")

# We pass this to the memory parameter when initializing our agent
conversational_agent = initialize_agent(
    agent='conversational-react-description', 
    tools=tools, 
    llm=llm,
    verbose=True,
    max_iterations=3,
    memory=memory,
)
```

- unlike zero-shot agent, we can ask follow-up questions to Conversational ReAct agent. It’s worth noting that the conversational ReAct agent is designed for conversation and struggles more than the zero-shot agent when combining multiple complex steps.



###### ReAct Docstore
- Another common agent is the react-docstore agent. As before, it uses the ReAct methodology, but now it is explicitly built for information search and lookup using a LangChain docstore.

- LangChain docstores allow us to store and retrieve information using traditional retrieval methods. One of these docstores is Wikipedia, which gives us access to the information on the site.

- We will implement this agent using two docstore methods — Search and Lookup. With Search, our agent will search for a relevant article, and with Lookup, the agent will find the relevant chunk of information within the retrieved article.

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
```


###### Self-Ask With Search
- This agent is the first you should consider when connecting an LLM with a search engine.
- The agent will perform searches and ask follow-up questions as often as required to get a final answer.

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
```











