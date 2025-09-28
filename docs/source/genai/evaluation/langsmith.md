# Evaluation using LangSmith

- Core components of LangSmith evaluation:
	- Dataset: These are the inputs to your application used for conducting evaluations.
	- Evaluator: An evaluator is a function responsible for scoring your AI application based on the provided dataset.



### Datasets
- They consist of examples that provide inputs and, optionally, expected reference outputs for assessing your AI application. Each example in a dataset is a data point with an inputs dictionary, an optional outputs dictionary, and an optional metadata dictionary. 

###### Creating datasets
- Manually curated examples
	- <i>You probably want to cover a few different common edge cases or situations you can imagine. Even 10-20 high-quality, manually-curated examples can go a long way.</i>

- Historical logs
	- If possible - try to collect end user feedback. You can then see which datapoints got negative feedback. That is super valuable! These are spots where your application did not perform well. You should add these to your dataset to test against in the future.

- Synthetic data
	- Once you have a few examples, you can try to artificially generate examples. It's generally advised to have a few good hand-craft examples before this, as this synthetic data will often resemble them in some way.

###### Types of datasets
- kv (Key-Value) Dataset:
	- ideal for evaluating chains and agents that require multiple inputs or generate multiple outputs.
- llm (Language Model) Dataset:
	- The llm dataset is designed for evaluating "completion" style language models.
- chat Dataset:
	- The chat dataset is designed for evaluating LLM structured "chat" messages as inputs and outputs.
	- This dataset type is useful for evaluating conversational AI systems or chatbots.


###### Partitioning datasets
- Save evaluation time/cost:
	- When setting up your evaluation, you may want to partition your dataset into different splits. For example, you might use a smaller split for many rapid iterations and a larger split for your final evaluation.

- Based on scenarios/use-cases
	- In addition, splits can be important for the interpretability of your experiments.
		- For example, if you have a RAG application, you may want your dataset splits to focus on different types of questions (e.g., factual, opinion, etc) and to evaluate your application on each split separately.







### Evaluators

- Evaluators are functions in LangSmith that score how well your application performs on a particular example. Evaluators receive these inputs:
	- Example: The example from your Dataset.
	- Root_run: The output and intermediate steps from running the inputs through the application.

- The evaluator returns an EvaluationResult (or a similarly structured dictionary), which consists of:
	- Key: The name of the metric being evaluated.
	- Score: The value of the metric for this example.
	- Comment: The reasoning or additional string information justifying the score.

- There are a few approaches and types of scoring functions that can be used in LangSmith evaluation.

###### Human

###### Heuristic
- Heuristic evaluators are hard-coded functions that perform computations to determine a score. To use them, you typically will need a set of rules that can be easily encoded into a function.
	- They can be reference-free (e.g., check the output for empty string or valid JSON). Or they can compare task output to a reference (e.g., check if the output matches the reference exactly).

###### LLM-as-judge
- LLM-as-judge evaluators use LLMs to score system output. To use them, you typically encode the grading rules / criteria in the LLM prompt. 
	- They can be reference-free (e.g., check if system output contains offensive content or adheres to specific criteria). Or,
	- they can compare task output to a reference (e.g., check if the output is factually accurate relative to the reference).

- <i>With LLM-as-judge evaluators, it is important to carefully review the resulting scores and tune the grader prompt if needed. Often a process of trial-and-error is required to get LLM-as-judge evaluator prompts to produce reliable scores.</i>


###### Pairwise
- Pairwise evaluators pick the better of two task outputs based upon some criteria. This can use either a heuristic ("which response is longer"), an LLM (with a specific pairwise prompt), or human (asking them to manually annotate examples).

- <b>When should you use pairwise evaluation?</b> Pairwise evaluation is helpful when it is difficult to directly score an LLM output, but easier to compare two outputs. This can be the case for tasks like summarization - it may be hard to give a summary a perfect score on a scale of 1-10, but easier to tell if it's better than a baseline.


















