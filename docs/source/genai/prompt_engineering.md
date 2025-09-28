**Concept: Prompt Engineering**

1.  **Core Idea (Mental Model/Mind Map Sketch):**
    *   **Analogy & Definition:** Prompt engineering is the process of crafting instructions (prompts) to guide a language model to generate a desired outcome. It's a form of "human-to-AI communication," akin to being a skilled "AI whisperer" or a "director for a language model." Unlike fine-tuning, it guides model behavior without altering underlying weights.
    *   **Core Principle:** LLMs are fundamentally prediction engines; they predict the next sequence of words most likely to follow the input prompt based on their vast training data. Your prompt "configures the model weights" temporarily for a specific task. Effective prompting activates the desired "program" or capability within the LLM.
    *   **Discipline & Rigor:** While often involving "fiddling," effective prompt engineering is an iterative process demanding systematic experimentation, evaluation, and rigor, much like any other machine learning experiment. It's a crucial skill, though production-ready AI also requires broader ML knowledge.

2.  **Key Factors & Considerations:**
    *   **Model Choice & Capability:**
        *   Different models (e.g., GPT-series, Gemini, Llama, Claude) possess varying strengths in instruction following, reasoning, speed, cost, and context window size. Prompts require optimization for specific models and their versions.
        *   **Instruction Following & Robustness:** A model's ability to follow instructions accurately and its robustness to slight prompt perturbations (e.g., "5" vs. "five," capitalization) dictates the amount of engineering effort. More powerful models tend to be more robust.
    *   **Clarity, Specificity & Explicitness:**
        *   Prompts must be clear, concise, and unambiguous. Avoid jargon or overly complex language. Explain precisely what you want the model to do.
        *   If undesirable model behaviors are observed, explicitly adjust the prompt to correct them (e.g., "Output only integer scores, not fractional ones").
    *   **Anatomy of a Prompt (Multiple Perspectives):**
        *   **General Structure:** Often includes:
            1.  **Task Description:** What the model should do, its role, and the desired output format.
            2.  **Example(s) (Shots):** Demonstrations of how to perform the task.
            3.  **The Task (Input Data):** The concrete input for the model to process (e.g., text to summarize, question to answer).
        *   **Azure Components:** Instructions, Primary Content, Examples, Cue (to jumpstart output), Supporting Content (contextual info).
        *   **System vs. User Prompts (Common in APIs):**
            *   **System Prompt:** High-level instructions from the developer, defining the model's persona, overall task, or strict rules. Many providers emphasize well-crafted system prompts for better performance.
            *   **User Prompt:** Contains the specific user query, task input, or dynamic contextual data.
            *   **Assistant Prompt:** Represents the model's previous responses; used for multi-turn conversations and for providing few-shot examples in chat formats.
            *   **Model-Specific Chat Templates:** Models internally combine system, user, and assistant messages using specific templates (e.g., Llama 2 & 3 have distinct formats with special tokens like `<s>[INST]`, `<<SYS>>`, `<|begin_of_text|>`). Using the wrong template can severely degrade performance or lead to unexpected behavior. Always verify and use the correct template.
    *   **In-Context Learning (ICL):** The ability of LLMs to learn a task from examples provided directly within the prompt, without requiring weight updates. This facilitates continuous learning by incorporating new information via the prompt.
    *   **Context Provisioning (Grounding / RAG):**
        *   Providing the model with relevant documents or data ("context" or "haystack") to base its answers on. Crucial for accuracy, up-to-date information, and reducing hallucinations.
        *   **Terminology:** "Prompt" refers to the entire input to the model. "Context" is the information provided *within* the prompt for the model to use.
    *   **Output Structure & Format Specification:**
        *   Clearly define the desired output: conciseness, specific formats (JSON, Markdown, bullet points), absence of preambles.
        *   Use markers (e.g., `JSON Response:`) to signal where structured output should begin.
    *   **LLM Output Configuration Parameters:**
        *   **Max Output Tokens:** Limits the length of the generated response.
        *   **Temperature:** Controls randomness (0 for deterministic, higher for creative).
        *   **Top-K:** Samples from the K most probable next tokens.
        *   **Top-P (Nucleus Sampling):** Samples from the smallest token set whose cumulative probability exceeds P.
        *   *General advice: Adjust Temperature OR Top-P, not both simultaneously.*
    *   **Context Length & Efficiency:**
        *   Models have maximum input token limits (context windows), which have been rapidly expanding (e.g., Gemini 1.5 Pro up to 2M tokens).
        *   **"Needle in a Haystack" (NIAH) Effect:** Information at the beginning or end of a long context is often processed more effectively by models than information in the middle.
    *   **Tokenization & Space Efficiency:** Understand that words are broken into tokens. Efficient phrasing and avoiding unnecessary whitespace can save tokens. The format of examples can also impact token count.
    *   **Prompt Caching:** Reusing static parts of prompts can save cost and latency if the API supports it.

3.  **Prompting Techniques & Strategies:**

    *   **Zero-Shot Learning:** Instructing the model without any examples. Relies on the model's pre-trained knowledge.
    *   **One-Shot & Few-Shot Learning (ICL):** Providing one (one-shot) or multiple (few-shot) examples of input-output pairs. Highly effective for guiding model behavior, specifying formats, and teaching new tasks within the prompt.
        *   *Number of Shots:* Depends on model capability and task complexity. Powerful models might need fewer shots for general tasks.
        *   *Best Practices for Shots:* Use diverse, high-quality, well-written examples. Mix class labels in classification examples. Ensure token-efficient formatting.
    *   **Role Prompting (Persona Adoption):** Instruct the model to adopt a specific persona (e.g., "You are an expert travel guide"). This influences tone, style, and the perspective of the response.
    *   **Instruction Placement & Repetition:**
        *   Generally, place critical instructions/task descriptions at the beginning. Some models (e.g., Llama 3) may prefer them at the end. Experiment.
        *   Repeating instructions (e.g., before and after user input) can reinforce them, especially for safety.
    *   **Priming the Output & Using Markers:** Start the model's response with a prefix or use delimiters to guide formatting or signal the start of a specific section.
    *   **Clear Syntax & Formatting (Markdown, XML):** Utilize Markdown (headers, lists) or XML tags to structure the prompt, clearly separate sections (e.g., instructions, examples, context), and improve readability for both humans and the model.
    *   **Breaking Down Complex Tasks (Prompt Chaining/Decomposition):**
        *   Divide a complex task into a sequence of simpler subtasks, each handled by a dedicated, simpler prompt.
        *   *Benefits:* Better performance, easier debugging, ability to monitor intermediate outputs, potential for parallelization, simpler prompt creation.
        *   *Trade-offs:* Can increase perceived latency by users; may involve more API calls, though simpler models can be used for some subtasks.
    *   **Giving the Model "Time to Think":**
        *   **Chain of Thought (CoT) Prompting:** Instruct the model to output its reasoning process step-by-step before arriving at the final answer. Significantly improves performance on tasks requiring reasoning.
            *   *Implementation:* Add phrases like "Think step by step," "Explain your rationale," or provide specific steps/examples of reasoning.
            *   *Best Practices for CoT:* Set temperature to 0. Ensure the final answer is clearly distinguishable from the reasoning.
        *   **Self-Critique:** Prompt the model to review and critique its own outputs or reasoning.
    *   **Advanced Reasoning Techniques:**
        *   **Self-Consistency:** Generate multiple CoT reasoning paths (using a non-zero temperature) and select the most frequent answer (majority vote). Improves accuracy over single CoT.
        *   **Tree of Thoughts (ToT):** Allows the model to explore multiple reasoning paths in a tree-like structure, evaluating intermediate "thoughts." Suited for complex problems requiring exploration.
        *   **ReAct (Reason and Act):** An agentic approach where the model interleaves reasoning steps with action steps (e.g., using external tools like search engines, code interpreters, or APIs) and observes the results to inform further reasoning.
            *Code Example (Illustrative):*
            ```python
            # Illustrative; specific libraries (e.g., LangChain) and setup required
            # from langchain.agents import initialize_agent, load_tools
            # from langchain_community.llms import VertexAI # Example LLM
            # llm = VertexAI(temperature=0.1)
            # tools = load_tools(["serpapi"], llm=llm) # Requires SERPAPI_API_KEY
            # agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
            # result = agent.run("How many children do the band members of Metallica have?")
            ```
            *   **Step-Back Prompting:** Prompt the model to first consider a more general question or abstract concept related to the specific query, then use that general understanding to address the specific query.
    *   **Automated Prompt Engineering (APE) & Optimization Tools:**
        *   Using LLMs to generate, critique, or refine prompts.
        *   Specialized tools (e.g., OpenPrompt, DSPy, Promptbreeder, TextGrad) aim to automatically find optimal prompts or prompt chains based on evaluation data.
        *   Tools for enforcing structured outputs (e.g., Guidance, Outlines, Instructor).
        *   *Caution:* Automated tools can generate many hidden API calls, use incorrect model templates, or change unexpectedly. It's wise to first manually craft prompts to understand requirements, and always inspect/validate tool-generated prompts.
    *   **Using Affordances (Tool Use):** Design prompts to make the model call external tools (e.g., search, calculators) for information it lacks or to perform actions, rather than relying solely on its parametric knowledge.
    *   **Meta Prompts:** High-level instructions prepended to the main prompt to guide overall behavior (e.g., "You must be kind"). Can be vulnerable to injection.
    *   **Restricting Knowledge to Provided Context:** Challenging. Techniques include explicit instructions ("Answer using only the provided context"), examples of questions *not* to answer from external knowledge, and requesting citations from the context. No method is entirely foolproof.
    *   **Code-Specific Prompting:** Techniques for generating, explaining, translating, and debugging code.
    *   **GPT-4.1 Specifics:** Reminders for persistence/tool use/planning in agentic workflows, using dedicated tool fields, optimal long context strategies (delimiters, instruction placement).

4.  **Trade-offs:**
    *   **Creativity vs. Factuality:** Higher temperature/Top-P/Top-K yields more diverse outputs but can reduce accuracy. Lower values are for deterministic tasks.
    *   **Prompt Detail/Length vs. Cost/Latency:** Complex prompts (many shots, CoT) use more tokens, increasing cost and potentially latency.
    *   **Robustness vs. Engineering Effort:** Less robust models require more "fiddling" and careful prompt construction.
    *   **Number of Examples vs. Inference Cost & Context Limit:** More examples improve ICL but increase costs and consume context window.
    *   **Decomposition Benefits vs. Latency/Cost:** Prompt chaining improves performance but may increase perceived latency and total queries.
    *   **Automation Benefits vs. Control/Cost:** Tools can accelerate PE but may obscure processes, incur hidden costs, and reduce direct control.

5.  **Common Challenges, Pitfalls, and Defensive Prompting:**
    *   **Sensitivity to Phrasing:** Small wording changes can yield vastly different outputs.
    *   **Model-Specific Behavior & Templates:** Prompts are not always portable across models; precise adherence to chat templates is critical.
    *   **Recency Bias:** Information at the end of a long prompt may have undue influence. (But also see NIAH "ends are better than middle").
    *   **Repetition Loops:** Models getting stuck generating repetitive text.
    *   **Hallucinations/Fabrication:** Generating plausible-sounding but incorrect information.
    *   **JSON Repair Needs:** Truncated or malformed JSON output due to token limits. Libraries like `json-repair` can help.
    *   **Understanding and Defending Against Prompt Attacks:** A critical area for production systems.
        *   **Types of Attacks:**
            1.  **Prompt Extraction:** Deducing the system prompt to replicate or exploit the application.
            2.  **Jailbreaking & Prompt Injection:** Subverting safety filters or injecting malicious instructions.
            3.  **Information Extraction:** Forcing the model to reveal sensitive training data or context.
        *   **Risks from Attacks:** Unauthorized code/tool execution, data leaks, social harms (e.g., generating instructions for illegal activities), misinformation, service interruption/subversion, brand damage.
        *   **Jailbreaking/Injection Methods:**
            *   *Direct Manual Hacking:* Obfuscation (misspellings, unusual characters, mixed languages), output formatting manipulation (requesting a poem about a harmful topic), role-playing (DAN - "Do Anything Now", grandma exploit).
            *   *Automated Attacks:* Algorithmic generation/mutation of attack prompts (e.g., PAIR using an attacker AI).
            *   *Indirect Prompt Injection:* Malicious instructions embedded in data retrieved by the model from external sources (web pages, documents, emails, database entries) that the model then processes as trusted input.
        *   **Information Extraction Methods:**
            *   *Factual Probing (LAMA-style):* Fill-in-the-blank prompts to extract relational knowledge.
            *   *Memorization Exploitation:* Prompts designed to trigger regurgitation of specific training data (e.g., PII, copyrighted content). Divergence attacks (e.g., "repeat 'poem' forever") can cause models to output raw training data. Larger models may memorize more. Diffusion models can also leak training images.
        *   **Defenses:** A multi-layered approach is necessary.
            1.  **Model-Level Defense:** Fine-tuning models to better differentiate and prioritize system instructions over user/tool inputs (e.g., OpenAI's Instruction Hierarchy). Training models to recognize malicious prompts and provide safe responses to borderline requests.
            2.  **Prompt-Level Defense:** Explicit negative constraints ("Do not return sensitive information..."). Repeating key instructions. Warning the model about potential malicious user tactics within the prompt itself. Carefully inspecting and hardening default prompts used by tools.
            3.  **System-Level Defense:** Input/output filtering (keyword blocking, PII detection, toxicity checks). Anomaly detection for unusual prompt patterns. Isolating execution of model-generated code in sandboxed environments. Requiring human approval for high-impact actions (e.g., database modifications). Defining and enforcing out-of-scope topics for the application.

6.  **Best Practices & Production Insights:**
    *   **Iterative Development:** Prompt engineering is an experimental cycle of crafting, testing, analyzing, documenting, and refining.
    *   **Start Simple:** Begin with basic prompts and add complexity (e.g., few-shot, CoT) as needed.
    *   **Clarity & Specificity:** Be explicit and unambiguous. Use instructions over constraints where possible, but use constraints for safety/formatting.
    *   **Provide High-Quality Examples (Few-shot):** One of the most effective techniques. Ensure diversity and relevance.
    *   **Specify Output Format Clearly:** Essential for consistency and downstream processing.
    *   **Provide Sufficient Context:** Ground the model to improve accuracy and reduce hallucinations.
    *   **Decompose Complex Tasks:** Break problems into smaller, manageable sub-prompts.
    *   **Give the Model "Time to Think" (CoT, etc.):** For complex reasoning.
    *   **Systematic Evaluation & Versioning:**
        *   Test changes systematically. Version prompts like code. Use experiment tracking.
        *   Pin production applications to specific model snapshots for consistency.
        *   Build automated evaluations (evals) to measure prompt performance.
    *   **Prompt Organization & Management:**
        *   Separate prompts from application code (e.g., into `.py` files, dedicated `.prompt` files, or a prompt catalog). Improves reusability, testability, readability, and collaboration.
        *   Use metadata for prompts (model, date, purpose, parameters, schemas) for better organization.
        *   Consider a versioned prompt catalog for managing prompts across multiple applications.
    *   **Defensive Mindset:** Assume prompts will be public and design for security. Write your system prompt assuming it will one day be leaked.
    *   **Cautious Use of Tools:** Start by writing your own prompts to gain understanding. If using PE tools, inspect their generated prompts, understand API call volume, and check for template correctness.
    *   **Stay Updated:** The field (models, techniques, attacks, defenses) evolves rapidly.

7.  **Key Questions for a Lead GenAI/ML Engineer:**
    *   Which model (and specific version/snapshot) is optimal for our task considering performance, cost, context length, and robustness?
    *   What is our standardized, iterative process for prompt design, testing, versioning, and deployment? How do we track experiments?
    *   How are we ensuring prompts correctly use model-specific chat templates, especially when integrating with different APIs or tools?
    *   What comprehensive evaluation metrics and frameworks (including security and safety) are in place to objectively measure prompt effectiveness and business RoI?
    *   What is our strategy for managing prompts as assets (e.g., prompt catalogs, version control integration)? How do we handle updates when shared prompts change?
    *   How robust are our prompts to input perturbations, and how are we mitigating risks from prompt injection, jailbreaking, and information extraction (both direct and indirect attacks)?
    *   What is our multi-layered defense strategy (model-level, prompt-level, system-level), and how do we balance security measures (e.g., violation rate) with user experience (e.g., false refusal rate)?
    *   Are we leveraging techniques like CoT, RAG, and task decomposition effectively for complex tasks, and what is their impact on latency and cost?
    *   How do we manage the trade-off between detailed, specific instructions and the risk of prompt brittleness or overfitting?
    *   What is our approach to using (or not using) automated prompt engineering tools, considering control, cost, transparency, and maintenance?
    *   How does our team stay current with the latest prompting techniques, model capabilities, attack vectors, and defensive best practices?
    *   Are we considering instruction hierarchies (if applicable) when designing interactions between system prompts and potentially untrusted data sources (e.g., RAG documents, tool outputs)?
