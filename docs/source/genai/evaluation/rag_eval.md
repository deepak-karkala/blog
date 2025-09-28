# RAG Evaluation

###

##### RAGAS Metrics


<p>
\[ \text{Faithfulness score} = {|\text{Number of claims in the generated answer that can be inferred from given context}| \over |\text{Total number of claims in the generated answer}|} \]
</p>

<p>
	\[ \text{Answer relevancy} = \frac{1}{N} \sum_{i=1}^{N} \frac{E_{g_i} \cdot E_o}{\|E_{g_i}\|\|E_o\|} \]
	\( E_{g_{i}}, E_{o} \): Embeddings of the generated and original questions and \( N \) is the number of generated questions, which is 3 default.
</p>

<p>
	\[ \text{Context Precision@K} = \frac{\sum_{k=1}^{K} \left( \text{Precision@k} \times v_k \right)}{\text{Total number of relevant items in the top } K \text{ results}} \]
	\[ \text{Precision@k} = {\text{true positives@k} \over  (\text{true positives@k} + \text{false positives@k})} \]
	\( K \)  is the total number of chunks in contexts and  \( v_{k} \in \{0, 1\} \) is the relevance indicator at rank \(k\).
</p>


<p>
	\[ \text{context recall} = {|\text{GT claims that can be attributed to context}| \over |\text{Number of claims in GT}|} \]
</p>


<p>
	\[ \text{noise sensitivity (relevant)} = {|\text{Number of incorrect claims in answer}| \over |\text{Number of claims in the Answer}|} \]
</p>


##### RAGAS Summarization Score
<p>
	\[ \text{QA score} = \frac{|\text{correctly answered questions}|}{|\text{total questions}|} \]
</p>


<p>
	\[ \text{conciseness score} = 1 - \frac{\min(\text{length of summary}, \text{length of context})}{\text{length of context} + \text{1e-10}} \]
</p>


<p>
	\[ \begin{split}\text{Summarization Score} = \text{QA score}*\text{coeff} + \\
\text{conciseness score}*\text{(1-coeff)}\end{split} \]
</p>

<script id="MathJax-script" type="text/javascript" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>