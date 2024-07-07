### Word2Vec

One of the fundamental problems in building natural language-learning systems is the question of how to  represent words ? This is vital to build practical natural language applications such as Machine translation, Question answering , information retrieval, Summarization , analysis of text, speech to text, text to speech etc.


One of the simplest ways to represent words is by using independent vectors for each word.

<div class="container py-4 py-md-5 px-4 px-md-3 text-body-secondary">
    <div class="row" >
      <div class="col-lg-3 mb-3">
        <img width="300px" src="../_static/nlp/word2vec/word2vec-matrix-rep.png"></img>
      </div>
    </div>
</div>



One of the drawbacks of this representation is that each column vector’s length is equal to the number of words in the vocabulary which is a huge number. But unfortunately we do not encode any notion of similarity or any other relationship between words. Eg: In our word representation space, we would want the words ‘banking and finance’ to be closer to each other while at the same time both being significantly far from the word ‘watermelon’. But in our current representation, they are all equally close or equally far from each other. In order to overcome this, we need an alternate representations of words, one which encodes the notion of closeness amongst similar words. Word2vec is one such model used to generate word representations.

#### Word2Vec Model

Word2vec model is based on the hypothesis that the meaning of a word can be derived from the distribution of contexts in which it appears. This seemingly simple idea is one of the most influential and successful ideas in all of modern NLP. The word2vec model represents each word in a fixed vocabulary as a low-dimensional (much smaller than vocabulary size) vector. 


##### Notation
<p>
Let \(|V|\) be the total number of words in the vocabulary and \(d\) be the length of vector representations of each word in the vocabulary. Let U and V be matrices storing the vector representations of the outside and center words respectively in the Vocabulary.
</p>

<p>\[ U \in R^{d \; \text{x} \; |V|}, V \in R^{d \; \text{x} \; |V|} \]</p>




<div class="container py-4 py-md-5 px-4 px-md-3 text-body-secondary">
    <div class="row" >
      <div class="col-lg-3 mb-3">
        <img width="700px" src="../_static/nlp/word2vec/word2vec-matrices.png"></img>
      </div>
    </div>
</div>

##### Model

Given this setup, there are two models to learn the word representations,
<ul>
<li>Skip gram: Given the centre word, compute how likely the outside word will occur in the context window of this centre word</li>
<li>Continuous Bag Of Words (CBOW): Given the context (surrounding words), compute how likely the centre word will occur</li>
</ul>

The rest of the article will use skip gram model while the CBOW model can be implemented in a similar manner.


<p>
Word2vec is a probabilistic model specified as follows, where \(u_w\) refers to the column of \(U\) corresponding to word \(w\) (and likewise for \(V\)). Given the centre word \(c\), this computes the probability of outside word \(o\) occurring in the context window of centre word.
</p>

<div class="container py-4 py-md-5 px-4 px-md-3 text-body-secondary">
    <div class="row" >
      <div class="col-lg-3 mb-3">
        <img width="600px" src="../_static/nlp/word2vec/word2vec-model.png"></img>
      </div>
    </div>
</div>

<p>
  Where
  <ul>
    <li>\(v_c\): Center word representation of a specific word \(c\). Stored as one column in the matrix \(V\).</li>
    <li>\(u_o\): Outside word representation of a specific word \(o\). Stored as one column in the matrix \(U\).</li>
  </ul>
</p>


<br>

##### Intuition

We want the model to assign high probabilities (ideally 1) for outside words which are more likely to occur in the context window of centre word. At the same time, we want the model to assign low probabilities (ideally 0) to words which are least likely to occur in the context window of centre word. 

<p>
  How do we express this mathematically ? For a given centre word and outside word pair, Let us have two vectors, \(y\) and \(\hat{y}\) of length Vocab, 
  <ul>
    <li>\(y\): Expected probability of outside word for a given centre word: Will be one for the actual outside word, zero for all other words</li>
    <li>\(\hat{y}\): Probabilities of outside words to occur in context of this centre word as predicted by the model</li>
  </ul>
</p>

<div class="container py-4 py-md-5 px-4 px-md-3 text-body-secondary">
    <div class="row" >
      <div class="col-lg-3 mb-3">
        <img width="400px" src="../_static/nlp/word2vec/word2vec-y-yhat.png"></img>
      </div>
    </div>
</div>


<p>
One way to quantify how good or bad the model’s predictions are is by using the cross entropy between the true distribution \(y\) and the predicted distribution \(\hat{y}\), for a particular center word c and a particular outside word \(o\). Accordingly the Cross Entropy loss objective is given by,
</p>

<p>
  \[ L_{naive-softmax}(v_c, o, U) =  - \sum_{w \in \mathcal{V}} y_w log(\hat{y_w})   \]
</p>

<p>
Since \(y\) is 0 for all but one word, this reduces to. 
\[ L_{naive-softmax}(v_c, o, U) = − log (\hat{y_o}) \]
\[ L_{naive-softmax}(v_c, o, U) = − log P(O = o|C = c) \]
</p>

{{< callout info >}}
<b>Intuition</b>: We now have a method to quantify the loss based on the how likely the center and outside words are likely to occur within a context window,
  <ul>
    <li>If \(\hat{y_o}\) is 1, loss is 0, model's prediction is perfect</li>
    <li>Closer \(\hat{y_o}\) is to 1, closer the loss is to 0</li>
    <li>and further away it is from 1, larger the loss is.</li>
  </ul>
{{< /callout >}}

<p>
This is the loss function, the objective is to minimise this loss over the entire corpus of windows and all the documents in the training data. 
</p>


<p>
  Let \(D\) be a set of documents \({d}\) where each document is a sequence of words \( w_{1}^d,....w_{m}^d, \), with all \( w ∈ V\). Let k be a positive-integer window size. Center word takes on the value of each word \(w_{i}\) in each document, and for each such \(w_{i}\), the corresponding outside words are {\( w_{i−k}, . . . , w_{i−1}, w_{i+1}, . . . , w_{i+k} \)}.
</p>

<p>
  \[ L(U, V) =  \sum_{d \in D} \sum_{i=1}^m \sum_{j=1}^k -log \; p_{U,V}(w_{i-j}^d | w_{i}^d)   \]
</p>

It can be observed that we are taking the sum over,
1. All documents in the corpus
2. All words in each document
3. All words occuring in the window of the likelihood of the outside word given the center word.

<div class="container py-4 py-md-5 px-4 px-md-3 text-body-secondary">
    <div class="row">
      <div class="col-lg-3 mb-3">
        <img width="700px" src="../_static/nlp/word2vec/word2vec-corpus.png"></img>
      </div>
    </div>
</div>


##### Code structuring
<div class="container py-4 py-md-5 px-4 px-md-3 text-body-secondary">
    <div class="row">
      <div class="col-lg-3 mb-3">
        <img width="700px" src="../_static/nlp/word2vec/word2vec-code.png"></img>
      </div>
    </div>
</div>



##### Learning Algorithm: Stochastic Gradient Descent
Now that we have defined the loss function, the next objective is to minimise the loss function. In order to do this, we need gradients of loss function with respect to the word vectors. We can then use Stochastic Gradient Descent algorithm to guide the learning process and minimise the loss function.

<p>
  \[  x_{n+1} = x_{n} - learningRate * \nabla_{x} Loss \]
</p>

<p>
In this section we will derive the gradients of loss function \(L(U,V)\) with respect to centre word \(v_c\), outside word matrix \(U\). We will begin with the naive softmax first and then derive the same for the Negative Sampling variant.
</p>

##### Naive-Softmax: Gradient wrt Center word vector
<p>
\[ L_{naive-softmax}(v_c, o, U) = − log P(O = o|C = c) \]
\[ L_{naive-softmax}(v_c, o, U) = − log (\hat{y_o}) \]
\[ L_{naive-softmax}(v_c, o, U) = − log  \dfrac {exp(u_o^T v_c)}  {\sum_{w \in \mathcal{V}} exp(u_w^T v_c)} \]
</p>

<p>
  Let us first compute the gradients of the loss with respect to the center word \(v_c\)
</p>

<p>
<!--\[  \nabla_{v_c} L_{naive-softmax}(v_c, o, U)  = - \dfrac{\partial}{\partial v_c} log \dfrac {exp(u_o^T v_c)}  {\sum_{w \in Vocab} exp(u_w^T v_c)} \]-->
\[  \nabla_{v_c} L_{naive-softmax}(v_c, o, U)  = - \nabla_{v_c} log \dfrac {exp(u_o^T v_c)}  {\sum_{w \in Vocab} exp(u_w^T v_c)} \]
\[ = - \nabla_{v_c} ( u_o^T v_c - log \sum_{w} exp(u_w^T v_c)   ) \]
\[ = -u_o + \dfrac{1}{\sum_{w \in \mathcal{V}} exp(u_w^T v_c)} \sum_{x \in \mathcal{V}} \nabla_{v_c} exp(u_x^T v_c)  \]
\[ = -u_o + \dfrac{1}{\sum_{w \in \mathcal{V}} exp(u_w^T v_c)} \sum_{x \in \mathcal{V}} exp(u_x^T v_c) u_x  \]
\[ = -u_o + \sum_{x \in \mathcal{V}} \dfrac{exp(u_x^T v_c)}{\sum_{w \in \mathcal{V}} exp(u_w^T v_c)}  u_x  \]
\[ = -u_o + \sum_{x \in \mathcal{V}} \mathcal{P}(O = x | C = c) u_x  \]
\[ = -u_o + \sum_{x \in \mathcal{V}} \hat{y_x} u_x  \]
</p>

<p>
  The SGD update step for the center word vector \(v_c\) will now be,
  \[ v_c^{n+1} = v_c^{n} - learningRate * \nabla_{v_c} L_{naive-softmax}(v_c, o, U) \]
  \[ v_c^{n+1} = v_c^{n} + learningRate * (u_o - \sum_{x \in \mathcal{V}} \hat{y_x} u_x )   \]
</p>

{{< callout info >}}
<p>
<b>Intuition</b>: So what does this equation tell us,
we have the vector for the word actually observed: \(u_o\). We subtract from that the vector, intuitively, that the model expected—in the sense that it’s the sum over all the vocabulary of the probability the model assigned to that word, multiplied by the vector that was assigned to that word. So, the \(v_c\) vector is updated to be “more like” the word vector that was actually observed than the word vector it expected. 
</p>
<p>
In short, we are making \(v_c\) to be more like \(u_o\), and this is the essence of word2vec model, for a given pair of (center, outside) words vector representations should be similar to each other whereas for any two random words selected, the word representations should be very different.

<div class="container py-4 py-md-5 px-4 px-md-3 text-body-secondary">
    <div class="row">
      <div class="col-lg-3 mb-3">
        <img width="700px" src="../_static/nlp/word2vec/word2vec-vc-intuition.png"></img>
      </div>
    </div>
</div>

</p>
{{< /callout >}}

<p>
  It is also interesting to note that, all the words in the vocabulary \({ u_1, u_2,....u_{|V|}   }\) are involved in order to update the center word vector \(v_c\). So let us rewrite the above equation update in terms of the outside word vector matrix \(U\). This will enable us to write vectorised code for gradient computation and thereby improve the training time.
</p>

<p>
\[  \nabla_{v_c} L_{naive-softmax}(v_c, o, U)  = - (u_o - \sum_{x \in \mathcal{V}} \hat{y_x} u_x)  \]
\[  \nabla_{v_c} L_{naive-softmax}(v_c, o, U)  = - U (y - \hat{y})  \]
</p>


##### Naive-Softmax: Gradient wrt outside word vectors
<p>
\[ L_{naive-softmax}(v_c, o, U) = − log P(O = o|C = c) \]
\[ L_{naive-softmax}(v_c, o, U) = − log (\hat{y_o}) \]
\[ L_{naive-softmax}(v_c, o, U) = − log  \dfrac {exp(u_o^T v_c)}  {\sum_{w \in \mathcal{V}} exp(u_w^T v_c)} \]
</p>

<p>
There are two cases here, <br>
Case 1: \(w=o\), So \(u_w\) is an outside word for this center word <br>
</p>

<p>
  \[ \nabla_{u_o} L_{naive-softmax}(v_c, o, U) = - \nabla_{u_o} log  \dfrac {exp(u_o^T v_c)}  {\sum_{w \in \mathcal{V}} exp(u_w^T v_c)} \]
  \[ = - (\nabla_{u_o} u_o^T v_c - \nabla_{u_o} log \sum_{x} exp(u_x^T v_c)) \]
  \[ = -v_c + \nabla_{u_o} log \sum_{x \in \mathcal{V}} exp(u_x^T v_c)   \]
  \[ = -v_c + \nabla_{u_o} log ( exp(u_o^T v_c) + \sum_{x \in \mathcal{V}, l\neq o} exp(u_x^T v_c) )    \]
  \[ = -v_c + \dfrac{1}{exp(u_o^T v_c) + \sum_{x \in \mathcal{V}, l\neq o} exp(u_x^T v_c)} (\nabla_{u_o} exp(u_o^T v_c) + \nabla_{u_o} \sum_{x \in \mathcal{V}, l\neq o} exp(u_x^T v_c) )     \]
</p>
<p>
  \[ = -v_c + \dfrac{1}{exp(u_o^T v_c) + \sum_{x \in \mathcal{V}, l\neq o} exp(u_x^T v_c)} ( exp(u_o^T v_c) v_c + 0 ) \]
  \[ = -v_c + \dfrac{exp(u_o^T v_c)}{\sum_{x \in \mathcal{V}} exp(u_x^T v_c)} v_c \]
  \[ = -v_c + \mathcal{P}(W=o|C=c)) v_c   \]
  \[ = -(1 - \hat{y_o})v_c  \]
</p> 


<p>
Case 2: \(w\neq o\), So  \(u_w\) is not an outside word for this center word
</p>


<p>
\[ \nabla_{u_w} L_{naive-softmax}(v_c, o, U) = - \nabla_{u_w} log  \dfrac {exp(u_o^T v_c)}  {\sum_{w \in \mathcal{V}} exp(u_w^T v_c)} \]
  \[ = - (\nabla_{u_w} u_o^T v_c - \nabla_{u_w} log \sum_{x} exp(u_x^T v_c)) \]
  \[ = - (0 - \dfrac{exp(u_w^T v_c)}{\sum_{x \in \mathcal{V}} exp(u_x^T v_c)} v_c  )  \]
  \[ = \dfrac{exp(u_w^T v_c)}{\sum_{x \in \mathcal{V}} exp(u_x^T v_c)} v_c \]
  \[ = \mathcal{P}(W \neq o | C = c) v_c \]
  \[ = \hat{y_w} v_c \]
</p>

<p>
  Combining cases 1 and 2, the gradient of loss function wrt the outside word matrix \(U\) is given by,
</p>

<p>
  \[ \nabla_{U} L_{naive-softmax}(v_c, o, U) = v_c [ \hat{y_1} \; \hat{y_2} ......-(1-\hat{y_o}).....\hat{y_{\mathcal{V}}}  ] \]
  \[ \nabla_{U} L_{naive-softmax}(v_c, o, U) = - v_c(y - \hat{y}) \]
</p>

<p>
  The SGD update step for the outside word vectors \(U\) will now be,
  \[ U^{n+1} = U^{n} - learningRate * \nabla_{U} L_{naive-softmax}(v_c, o, U) \]
  \[ U^{n+1} = U^{n} + learningRate * v_c(y - \hat{y})   \]
</p>







{{< callout info >}}
<p>
<b>Intuition</b>: So what does this equation tell us, <br>
<ol>
  <li>
    One thing to be noted is the fact that all the words in the vocabulary will be updated for each (center, outside) words pair in the training data. This is computationally very expensive and the Negative-Sampling technique described in the next section is used to address this drawback.
  </li>
  <li>
    For a given (center, outside) word pair, 
      <ol>
        <li>
          we want \(\hat{y}\) to be close to the one for the outside word \(u_o\). The center word vector \(v_c\) is scaled by the difference \((1-\hat{y_o})\) and added to the word vector \(u_o\). So larger the difference between estimated and target value, \(v_c\) scaled by larger factor will be added to \(u_w\). <b>In essence we are making the word vector representation of the outside word \(u_o\) similar to that of the center word \(v_c\).</b>
        </li>
        <li>
          we want \(\hat{y}\) to be close to 0 for all the other words in the vocabulary. The center word vector \(v_c\) is scaled by the factor \(-\hat{y_{w \neq o}}\) and added to the word vectors \(u_{w \neq o}\). <b>In essence we are making the word vector representation of non-outside words \(u_{w \neq o}\) dissimilar to that of the center word \(v_c\).</b>
        </li>
      </ol>
  </li>
</ol>

<div class="container py-4 py-md-5 px-4 px-md-3 text-body-secondary">
    <div class="row">
      <div class="col-lg-3 mb-3">
        <img width="700px" src="../_static/nlp/word2vec/word2vec-uw-intuition.png"></img>
      </div>
    </div>
</div>

</p>
{{< /callout >}}



#### Negative Sampling

The drawback of word2vec model is that SGD update step involves using all the words in the vocabulary. This is prohibitively expensive since the total number of words in the vocabulary is massive. In order to avoid this, skip-gram with negative sampling was introduced. To see how this improves over the naive skip-gram, let us revisit the word2vec probabilistic model,

<div class="container py-4 py-md-5 px-4 px-md-3 text-body-secondary">
    <div class="row" >
      <div class="col-lg-3 mb-3">
        <img width="600px" src="../_static/nlp/word2vec/word2vec-model.png"></img>
      </div>
    </div>
</div>

From this, we can observe that instead of updating all the words for each context window, we can choose to update only a small number of words which do not occur in this context window. Choosing such a small set of random words, is referred to as negative sampling.

<p>
  Let us sample \(K\) negative words \( {w_1, w_2,.... w_K} \) from the vocabulary. Their outside word vectors are respectively given by \( {u_{w_1}, u_{w_2},.... u_{w_K} } \).  For a center word \(c\) and an outside word \(o\), the negative sampling loss function is given by
</p>

<p>
\[  L_{negative-sampling}(v_c, o, U) = -log(\sigma(u_o^T v_c)) - \sum_{s=1}^K log(\sigma(-u_{w_s}^T v_c))  \]
</p>

<p>
  where \( \sigma() \) is the sigmoid function, <br>
  \( \sigma(x) = \dfrac{1}{1 + exp(-x)} \) and <br>
  \( \dfrac{\partial}{\partial x} \sigma(x) = \sigma(x) (1 - \sigma(x)) \)
</p>

With this change to the model, the gradient calculations are modified as follows,


##### Negative-Sampling: Gradient wrt Center word vector
<p>
\[  \nabla_{v_c} L_{negative-sampling}(v_c, o, U) = - \nabla_{v_c} log(\sigma(u_o^T v_c)) - \nabla{v_c} \sum_{s=1}^K log(\sigma(-u_{w_s}^T v_c))  \]  
\[  = - \dfrac{1}{\sigma(u_o^T v_c)} \sigma(u_o^T v_c) (1 - \sigma(u_o^T v_c)) u_o - \sum_{s=1}^K \dfrac{1}{(1 - \sigma(u_{w_s}^T v_c))} (-\sigma(u_{w_s}^T v_c)) (1 - \sigma(u_{w_s}^T v_c)) u_{w_s} \]
\[  = -(1 - \sigma(u_o^T v_c))u_o  - \sum_{s=1}^K (1 - \sigma(-u_{w_s}^T v_c)) (-u_{w_s})  \]
</p>


##### Negative-Sampling: Gradient wrt Outside word Matrix
<p> There are two cases here,<br>
    Case 1:When the word is an outside word \(u_{w=o}\) for this particular center word \(v_c\),
</p>

<p>
  \[ \nabla_{u_o} L_{negative-sampling}(v_c, o, U) = - \nabla_{u_o} log(\sigma(u_o^T v_c)) - \nabla{u_o} \sum_{s=1}^K log(\sigma(-u_{w_s}^T v_c))  \]
  \[ = - \dfrac{1}{\sigma(u_o^T v_c)} \sigma(u_o^T v_c) (1 - \sigma(u_o^T v_c))v_c - 0  \]
  \[ = -(1 - \sigma(u_o^T v_c)) v_c \]
</p>

<p> 
    Case 2: When the word is negatively sampled word \(u_{w \in Vocab{|w=w_s}}\) for this particular center word \(v_c\),
</p>

<p>
  \[ \nabla_{u_{w_s}} L_{negative-sampling}(v_c, o, U) = - \nabla_{u_{w_s}} log(\sigma(u_o^T v_c)) - \nabla{u_{w_s}} \sum_{s=1}^K log(\sigma(-u_{w_s}^T v_c))  \]
  \[ = 0 - \sum_{s=1}^K \dfrac{1}{(1 - \sigma(u_{w_s}^T v_c))} (-\sigma(u_{w_s}^T v_c)) (1 - \sigma(u_{w_s}^T v_c)) v_c  \]
  \[ = (1 - \sigma(-u_{w_s}^T v_c)) v_c \]
</p>


#### Skip-gram: Accumulated Gradients over an entire context window

<p>
The above sections derived gradients of loss function for a given pair of center and outside words. Let us now use those to compute the gradients over one particular context window, where we will have one corresponding center vector and \(m\) outside words where \(m\) is the context window size. For a given context window, let \(c = w_t\) be the center word and let \([w_{t-m},....w_{t-1}, w_t, w_{t+1},... w_{t+m}] \) be the outside words.
</p>

<p>
  The total loss for the context window is then given by,
</p>

<p>
  \[ L_{skip-gram}(v_c, w_{t−m}, . . . w_{t+m}, U) = \sum_{-m \leq j \leq m, j \neq 0} L(v_c, w_{t+j}, U)  \]
</p>

<p>
  where, \( L(v_c, w_{t+j} , U)\) represents an arbitrary loss term for the center word \( c = w_t\) and outside word
\( w_{t+j}\). \( L(v_c, w_{t+j} , U)\) could be \( L_{naive-softmax}(v_c, w_{t+j} , U)\) or \(L_{neg-sample}(v_c, w_{t+j} , U)\).
</p>

<p>
  The total loss over a context window, is simply the sum of individual loss of each (center, outside) word pair. As a result the gradients over the entire context window will also be sum of individual gradients of loss wrt each pair of (center, outside) words,
</p>


<p>
  The loss gradients over context window wrt the outside word vectors \(U\) is given by,
  \[ \dfrac{\partial}{\partial U} L_{skip-gram}(v_c, w_{t−m}, . . . w_{t+m}, U) = \sum_{-m \leq j \leq m, j \neq 0} \dfrac{\partial}{\partial U} L(v_c, w_{t+j}, U) \]
</p>

<p>
  The loss gradients over context window wrt the center word vector \(v_c\) is given by,
  \[ \dfrac{\partial}{\partial v_c} L_{skip-gram}(v_c, w_{t−m}, . . . w_{t+m}, U) = \sum_{-m \leq j \leq m, j \neq 0} \dfrac{\partial}{\partial v_c} L(v_c, w_{t+j}, U) \]
</p>

<p>
  The loss gradients over context window wrt the other center word vectors \(v_{w|w \neq c}\) is given by,
  \[ \dfrac{\partial}{\partial v_w} L_{skip-gram}(v_c, w_{t−m}, . . . w_{t+m}, U) = 0 \]
</p>

<p>
  Now that we have computed the gradients of the Loss function \( L(v_c, w_{t+j} , U)\) with respect to all the
model parameters \(U\) and \(V\), we can now implement these in code and train the word2vec Skip-gram model.
</p>


#### Code: Word2vec using Numpy 
The code is based on the template of [Word2Vec Assignment from Stanford's NLP Course CS224n](https://web.stanford.edu/class/cs224n/assignments/a2.pdf) . It uses the [Stanford Sentiment Treebank Dataset](https://paperswithcode.com/dataset/sst) to train the Word2Vec Skip-gram model.

We start with importing the libraries,
<div class="row">
  <div class="col-lg-10 col-md-12 col-sm-12 col-12 mx-auto" style="overflow: scroll;">
    <script src="https://gist.github.com/deepak-karkala/b21bef6503b8a4f214cbec2526f6474a.js"></script>
  </div>
</div>

The sigmoid and softmax function definitions are given by,
<div class="row">
  <div class="col-lg-10 col-md-12 col-sm-12 col-12 mx-auto" style="height: 60vh; overflow: scroll;">
    <script src="https://gist.github.com/deepak-karkala/4a1e0da56cc83117ce8fa56c1dc8e932.js"></script>
  </div>
</div>

At the top level, we define a function to train the Word2Vec model over multiple iterations of training data,
<div class="row">
  <div class="col-lg-10 col-md-12 col-sm-12 col-12 mx-auto" style="height: 60vh; overflow: scroll;">
    <script src="https://gist.github.com/deepak-karkala/74b8c0563a0faf5972a987c58ef0c1db.js"></script>
  </div>
</div>

For each iteration (or epoch), the following function will then accumulate the losses and gradients over several context windows,
<div class="row">
  <div class="col-lg-10 col-md-12 col-sm-12 col-12 mx-auto" style="height: 60vh; overflow: scroll;">
    <script src="https://gist.github.com/deepak-karkala/f43491550daaaaeb9da020ea28559d51.js"></script>
  </div>
</div>

For a given context window, the Skip-gram Model will accumulate the losses and the gradients over all pairs of (center, outside) word pairs,
<div class="row">
  <div class="col-lg-10 col-md-12 col-sm-12 col-12 mx-auto" style="height: 60vh; overflow: scroll;">
    <script src="https://gist.github.com/deepak-karkala/d02a7914bc93b2d5e7bdb23f35de87bc.js"></script>
  </div>
</div>


The Naive-Softmax Model is where we compute the loss and the gradient for one given pair of (center, outside) words. This function uses the gradient formulae derived in the earlier section,
<div class="row">
  <div class="col-lg-10 col-md-12 col-sm-12 col-12 mx-auto" style="height: 60vh; overflow: scroll;">
    <script src="https://gist.github.com/deepak-karkala/9e225fd138de61408f862350685c52c0.js"></script>
  </div>
</div>


The Word2Vec model can now be trained using the above functions. The learning rate, number of training iterations, dimensions of word vector representations, context window size are all parameters which can be tuned,
<div class="row">
  <div class="col-lg-10 col-md-12 col-sm-12 col-12 mx-auto" style="height: 60vh; overflow: scroll;">
    <script src="https://gist.github.com/deepak-karkala/e88666fca3f0c909315d4e15d2cae128.js"></script>
  </div>
</div>



#### Results: Visualisation
As desired, the loss function reduces as we train the model. 

<div class="container py-4 py-md-5 px-4 px-md-3 text-body-secondary">
    <div class="row" >
      <div class="col-lg-3 mb-3">
        <img width="600px" src="../_static/nlp/word2vec/word2vec-loss.png"></img>
      </div>
    </div>
</div>

This implies that the vector representations of words similar to each other (the ones which occur together within a context window) are pushed close to each other whereas that of random word pairs will be pushed further away from each other.

<!--
In order to get an intuition of what the learned representations look like, word vectors were reduced to 2 dimensions using PCA and plotted,
-->

#### Summary

Word2vec model is based on the hypothesis that the meaning of a word can be derived from the distribution of contexts in which it appears. This seemingly simple idea is one of the most influential and successful ideas in all of modern NLP and continues to be so even for training modern LLMs.

Implementing word2vec from scratch using Numpy, without any high level ML frameworks, gives us better insights into probabilistic models, computation of gradients for SGD and the end to end training process of a Machine Learning Model. These vector representations of words can now be used in the downstream applications such as Sentiment Analysis, Summarisation, Question-Answering, Information Retrieval. 


#### References

- [Stanford CS224n Lecture Notes](https://web.stanford.edu/class/cs224n/readings/cs224n_winter2023_lecture1_notes_draft.pdf)
- [Stanford CS224n Assignment](https://web.stanford.edu/class/cs224n/assignments/a2.pdf)


<script id="MathJax-script" type="text/javascript" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>

