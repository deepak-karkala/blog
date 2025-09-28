# Evaluation

```{toctree}
:hidden:

summarization
langsmith
rag_eval
```

### Overview

<div class="container py-4 py-md-5 px-4 px-md-3 text-body-secondary">
    <div class="row" >
      <div class="col-lg-4 mb-4">
        <img src="../../_static/genai/evaluation/eval_cycle.png"></img>
      </div>
    </div>
</div>

Source: [Hamel Husain: Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/)


### The Types Of Evaluation

- three levels of evaluation to consider:
	- Level 1: Unit Tests
	- Level 2: Model & Human Eval (this includes debugging)
	- Level 3: A/B testing

- <b>When (and how often) to run these levels ?</b>
	- The cost of Level 3 > Level 2 > Level 1. This dictates the cadence and manner you execute them. For example, I often run Level 1 evals on every code change, Level 2 on a set cadence and Level 3 only after significant product changes. It’s also helpful to conquer a good portion of your Level 1 tests before you move into model-based tests, as they require more work and time to execute.

	- There isn’t a strict formula as to when to introduce each level of testing. You want to balance getting user feedback quickly, managing user perception, and the goals of your AI product.

- <b>Example GenAI product</b>
	- Rechat is a SaaS application that allows real estate professionals to perform various tasks, such as managing contracts, searching for listings, building creative assets, managing appointments.
	- Rechat’s AI assistant, Lucy, is a canonical AI product: a conversational interface that obviates the need to click, type, and navigate the software. 



#### Level 1: Unit Tests
- Unit tests for LLMs are assertions (like you would write in pytest)
- The important part is that these assertions should run fast and cheaply as you develop your application so that you can run them every time your code changes.
- <i>If you have trouble thinking of assertions, you should critically examine your traces and failure modes. </i>

###### Step 1: Write Scoped Tests</b>
- The most effective way to think about unit tests is to break down the scope of your LLM into features and scenarios. 

- <b>Example 1: Feature: Listing Finder</b>
	- This feature to be tested is a function call that responds to a user request to find a real estate listing. For example, <i>“Please find listings with more than 3 bedrooms less than $2M in San Jose, CA”</i>
	- Scenario --- Assertions
		- Only one listing matches user query	--- len(listing_array) == 1
		- Multiple listings match user query	--- len(listing_array) > 1
		- No listings match user query ---	len(listing_array) == 0

- <b>Example 2: test to ensures that sensitive/private information (Eg:UUID) is not mentioned in the LLM output</b>
	- Retrieval results returned to the LLM contain fields that shouldn’t be surfaced to the user; such as the UUID associated with an entry. Our LLM prompt tells the LLM to not include UUIDs. We use a simple regex to assert that the LLM response doesn’t include UUIDs.

- Evaluation should include hundreds of these unit tests.
	- We continuously update them based on new failures we observe in the data as users challenge the AI or the product evolves. These unit tests are crucial to getting feedback quickly when iterating on your AI system (prompt engineering, improving RAG, etc.).
	- Many people eventually outgrow their unit tests and move on to other levels of evaluation as their product matures, but it is essential not to skip this step!


###### Step 2: Create Test Cases
- To test these assertions, you must generate test cases or inputs that will trigger all scenarios you wish to test. I often utilize an LLM to generate these inputs synthetically.

- For each of these test cases, we execute the first user input to create the contact. We then execute the second query to fetch that contact. If the CRM doesn’t return exactly 1 result then we know there was a problem either creating or fetching the contact.

```code
Write 50 different instructions that a real estate agent can give to his assistant to create contacts on his CRM. The contact details can include name, phone, email, partner name, birthday, tags, company, address and job.

For each of the instructions, you need to generate a second instruction which can be used to look up the created contact.

The results should be a JSON code block with only one string as the instruction like the following:

[
  ["Create a contact for John (johndoe@apple.com)", 
  "What's the email address of John Smith?"]
]
```

Using the above prompt, we generate test cases like below:

```code
[ 
    [
        'Create a contact for John Smith (johndoe@apple.com) with phone number 123-456-7890 and address 123 Apple St.', 
        'What\'s the email address of John Smith?'
    ],
    [
        'Add Emily Johnson with phone 987-654-3210, email emilyj@email.com, and company ABC Inc.', 
        'What\'s the phone number for Emily Johnson?'
    ],
    [
        'Create a contact for Tom Williams with birthday 10/20/1985, company XYZ Ltd, and job title Manager.', 
        'What\'s Tom Williams\' job title?'
    ],
    [
        'Add a contact for Susan Brown with partner name James Brown, and email susanb@email.com.', 
    'What\'s the partner name of Susan Brown?'
    ],
…
]
```


>- You must constantly update these tests as you observe data through human evaluation and debugging. The key is to make these as challenging as possible while representing users’ interactions with the system.	
>
>- You don’t need to wait for production data to test your system. You can make educated guesses about how users will use your product and generate synthetic data. You can also let a small set of users use your product and let their usage refine your synthetic data generation strategy. One signal you are writing good tests and assertions is when the model struggles to pass them - these failure modes become problems you can solve with techniques like fine-tuning later on.	
>
>- On a related note, unlike traditional unit tests, you don’t necessarily need a 100% pass rate. Your pass rate is a product decision, depending on the failures you are willing to tolerate.


###### Step 3: Run & Track Your Tests Regularly
- There are many ways to orchestrate Level 1 tests. Rechat has been leveraging CI infrastructure (e.g., GitHub Actions, GitLab Pipelines, etc.) to execute these tests. 

- In addition to tracking tests, you need to track the results of your tests over time so you can see if you are making progress. If you use CI, you should collect metrics along with versions of your tests/prompts outside your CI system for easy analysis and tracking.

- I recommend starting simple and leveraging your existing analytics system to visualize your test results. For example, Rechat uses [Metabase](https://www.metabase.com/) to track their LLM test results over time.


#### Level 2: Human & Model Eval

- After you have built a solid foundation of Level 1 tests, you can move on to other forms of validation that cannot be tested by assertions alone. A prerequisite to performing human and model-based eval is to log your traces.

###### Logging Traces
- A trace is a concept that has been around for a while in software engineering and is a log of a sequence of events such as user sessions or a request flow through a distributed system. In other words, tracing is a logical grouping of logs. 

- In the context of LLMs, traces often refer to conversations you have with a LLM. For example, a user message, followed by an AI response, followed by another user message, would be an example of a trace.
	- Rechat was using LangChain which automatically logs trace events to LangSmith for you

> Tools: arize, human loop, openllmetry and honeyhive

###### Looking At Your Traces
- This means rendering your traces in domain-specific ways. I’ve often found that it’s better to build my own data viewing & labeling tool so I can gather all the information I need onto one screen.
	- What tool (feature) & scenario was being evaluated.
	- Whether the trace resulted from a synthetic input or a real user input.
	- Filters to navigate between different tools and scenario combinations.
	- Links to the CRM and trace logging system for the current record.

- These tools can be built with lightweight front-end frameworks like Gradio, Streamlit, Panel, or Shiny in less than a day.

- <i>I often start by labeling examples as good or bad. I’ve found that assigning scores or more granular ratings is more onerous to manage than binary ratings. There are advanced techniques you can use to make human evaluation more efficient or accurate (e.g., active learning, consensus voting, etc.), but I recommend starting with something simple.</i>

- As discussed later, these labeled examples measure the quality of your system, validate automated evaluation, and curate high-quality synthetic data for fine-tuning.

- <b>How much data should you look at?</b>
	- When starting, you should examine as much data as possible. I usually read traces generated from ALL test cases and user-generated traces at a minimum. You can never stop looking at data—no free lunch exists. However, you can sample your data more over time, lessening the burden.


###### Automated Evaluation w/ LLMs
- You should track the correlation between model-based and human evaluation to decide how much you can rely on automatic evaluation. Furthermore, by collecting critiques from labelers explaining why they are making a decision, you can iterate on the evaluator model to align it with humans through prompt engineering or fine-tuning. However, I tend to favor prompt engineering for evaluator model alignment.

- I love using low-tech solutions like Excel to iterate on aligning model-based eval with humans. This spreadsheet would contain the following information:
	- model response: this is the prediction made by the LLM.
	- model critique: this is a critique written by a (usually more powerful) LLM about your original LLM’s prediction.
	- model outcome: this is a binary label the critique model assigns to the model response as being “good” or “bad.”
	
- Human then fills out his version of the same information - meaning his critique, outcome, and desired response for 25-50 examples at a time.
	- This information allowed me to iterate on the prompt of the critique model to make it sufficiently aligned with human over time.

- General tips on model-based eval:
	- Use the most powerful model you can afford. It often takes advanced reasoning capabilities to critique something well. You can often get away with a slower, more powerful model for critiquing outputs relative to what you use in production.
	- Model-based evaluation is a meta-problem within your larger problem. You must maintain a mini-evaluation system to track its quality. I have sometimes fine-tuned a model at this stage (but I try not to).
	- After bringing the model-based evaluator in line with the human, you must continue doing periodic exercises to monitor the model and human agreement.
	
- <i>My favorite aspect about creating a good evaluator model is that its critiques can be used to curate high-quality synthetic data, which I will touch upon later.</i>


#### Level 3: A/B Testing

- Finally, it is always good to perform A/B tests to ensure your AI product is driving user behaviors or outcomes you desire. A/B testing for LLMs compared to other types of products isn’t too different.

- It’s okay to put this stage off until you are sufficiently ready and convinced that your AI product is suitable for showing to real users. This level of evaluation is usually only appropriate for more mature products.


#### Evaluating RAG
- Aside from evaluating your system as a whole, you can evaluate sub-components of your AI, like RAG.


#### Eval Systems Unlock Superpowers For Free

###### Fine-Tuning
- Rechat resolved many failure modes through fine-tuning that were not possible with prompt engineering alone. Fine-tuning is best for learning syntax, style, and rules, whereas techniques like RAG supply the model with context or up-to-date facts.

- 99% of the labor involved with fine-tuning is assembling high-quality data that covers your AI product’s surface area. However, if you have a solid evaluation system like Rechat’s, you already have a robust data generation and curation engine.

- <b>Data Synthesis & Curation</b>
	- To illustrate why data curation and synthesis come nearly for free once you have an evaluation system, consider the case where you want to create additional fine-tuning data for the listing finder mentioned earlier. First, you can use LLMs to generate synthetic data with a prompt like following:

	- This is almost identical to the exercise for producing test cases! You can then use your Level 1 & Level 2 tests to filter out undesirable data that fails assertions or that the critique model thinks are wrong. You can also use your existing human evaluation tools to look at traces to curate traces for a fine-tuning dataset.

```code
Imagine if Zillow was able to parse natural language. Come up with 50 different ways users would be able to search listings there. Use real names for cities and neighborhoods.

You can use the following parameters:

<ommitted for confidentiality>

Output should be a JSON code block array. Example:

[
"Homes under $500k in New York"
]
```



###### Debugging
- When you get a complaint or see an error related to your AI product, you should be able to debug this quickly. If you have a robust evaluation system, you already have:
	- A database of traces that you can search and filter.
	- A set of mechanisms (assertions, tests, etc) that can help you flag errors and bad behaviors.
	- Log searching & navigation tools that can help you find the root cause of the error. For example, the error could be RAG, a bug in the code, or a model performing poorly.
	- The ability to make changes in response to the error and quickly test its efficacy.




### Evaluation: General tips
- <i>Evaluation systems create a flywheel that allows you to iterate very quickly. It’s almost always where people get stuck when building AI products.</i>  Some key takeaways to keep in mind:

	- Remove ALL friction from looking at data.
	- Keep it simple. Don’t buy fancy LLM tools. Use what you have first.
	- You are doing it wrong if you aren’t looking at lots of data.
	- Don’t rely on generic evaluation frameworks to measure the quality of your AI. Instead, create an evaluation system specific to your problem.
	- Write lots of tests and frequently update them.
	- LLMs can be used to unblock the creation of an eval system. Examples include using a LLM to:
		- Generate test cases and write assertions
		- Generate synthetic data
		- Critique and label data etc.
	- Re-use your eval infrastructure for debugging and fine-tuning.



### References
- [Langsmith: Evaluate an LLM Application](https://docs.smith.langchain.com/how_to_guides/evaluation/evaluate_llm_application#use-a-summary-evaluator)

- [Hamel Husain: Your AI Product Needs Evals](https://hamel.dev/blog/posts/evals/)

- [Eugene Yan: Task-Specific LLM Evals that Do & Don't Work](https://eugeneyan.com/writing/evals/)