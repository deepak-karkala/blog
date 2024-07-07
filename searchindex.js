Search.setIndex({"docnames": ["index", "nlp/index", "nlp/rnn", "nlp/word2vec", "past_experiences/adas", "past_experiences/ecom", "past_experiences/index", "past_experiences/iot", "vision/image_segmentation", "vision/index"], "filenames": ["index.md", "nlp/index.md", "nlp/rnn.md", "nlp/word2vec.md", "past_experiences/adas.md", "past_experiences/ecom.md", "past_experiences/index.md", "past_experiences/iot.md", "vision/image_segmentation.rst", "vision/index.md"], "titles": ["Home", "Natural Language Processing", "Recurrent Neural Networks", "Word2Vec", "ADAS", "Ecom", "Past Experiences", "Predictive Analytics in IoT", "Image Segmentation", "Computer Vision"], "terms": {"A": 0, "clean": 0, "customis": 0, "sphinx": 0, "document": [0, 3], "theme": 0, "stanford": [0, 3], "cs224n": [0, 3], "lectur": [0, 3], "note": [0, 3], "involv": [2, 3], "follow": [2, 3], "compon": 2, "In": [2, 3], "thi": [2, 3, 7], "articl": [2, 3], "we": [2, 3], "go": 2, "implement": [2, 3], "each": [2, 3], "from": [2, 3], "scratch": [2, 3], "us": 2, "numpi": 2, "class": 2, "object": [2, 3], "def": 2, "__init__": 2, "self": 2, "y": [2, 3], "y_pred": 2, "rais": 2, "notimpl": 2, "gradient": 2, "accuraci": 2, "return": 2, "0": [2, 3], "crossentropyloss": 2, "p": [2, 3], "np": 2, "clip": 2, "1e": 2, "15": 2, "1": [2, 3], "log": [2, 3], "y_true": 2, "argmax": 2, "axi": 2, "sum": [2, 3], "len": 2, "updat": [2, 3], "w": [2, 3], "grad_w": 2, "sgd": [2, 3], "learning_r": 2, "descent": 2, "neuralnetwork": 2, "loss_funct": 2, "metric": 2, "valid": 2, "add": [2, 7], "method": [2, 3], "which": [2, 3, 7], "set": [2, 3, 7], "input": 2, "shape": 2, "set_input_shap": 2, "output_shap": 2, "attach": 2, "an": 2, "ha": 2, "weight": 2, "hasattr": 2, "initi": 2, "current": [2, 3], "append": 2, "x": [2, 3], "n_epoch": 2, "batch_siz": 2, "x_val": 2, "none": 2, "y_val": 2, "print_once_every_epoch": 2, "fix": [2, 3], "number": [2, 3], "epoch": [2, 3], "rang": 2, "For": [2, 3], "run": 2, "over": 2, "all": [2, 3, 7], "minibatch": 2, "x_batch": 2, "y_batch": 2, "batch_iter": 2, "acc": 2, "train_on_batch": 2, "mean": [2, 3], "At": [2, 3], "end": [2, 3], "get": 2, "val": 2, "data": [2, 3], "i": [2, 3, 7], "val_loss": 2, "val_acc": 2, "test_on_batch": 2, "singl": 2, "one": [2, 3], "sampl": 2, "_forward_pass": 2, "true": [2, 3], "comput": [2, 3], "wrt": 2, "grad_loss": 2, "back": 2, "propag": 2, "through": 2, "entir": 2, "_backward_pass": 2, "fals": 2, "calcul": [2, 3], "output": 2, "nn": 2, "layer_output": 2, "forward_pass": 2, "grad": 2, "revers": 2, "backward_pass": 2, "pic": 2, "page4": 2, "2": [2, 3], "unit": 2, "equat": [2, 3], "backpropag": 2, "within": [2, 3], "paramet": [2, 3], "dim_input": 2, "int": 2, "dimens": [2, 3], "dim_hidden": 2, "hidden": 2, "state": 2, "relu": 2, "bptt_trunc": 2, "time": [2, 3], "step": [2, 3], "input_shap": 2, "5": 2, "w_hh": 2, "matrix": 2, "w_xh": 2, "w_ho": 2, "matric": [2, 3], "limit": 2, "math": 2, "sqrt": 2, "random": [2, 3], "uniform": 2, "w_xh_opt": 2, "copi": 2, "w_hh_opt": 2, "w_ho_opt": 2, "num_paramet": 2, "prod": 2, "gif": 2, "page6": 2, "across": 2, "num_timestep": 2, "layer_input": 2, "save": 2, "backprop": 2, "intermedi": 2, "state_act_input": 2, "zero": [2, 3], "state_act_output": 2, "iter": [2, 3], "t": [2, 3], "matmul": 2, "page7": 2, "accumul": 2, "multipl": [2, 3], "dim_output": 2, "variabl": 2, "appropri": 2, "size": [2, 3], "grad_inp": 2, "zeros_lik": 2, "grad_wxh": 2, "grad_whh": 2, "grad_who": 2, "timestep": 2, "grad_state_act_output": 2, "grad_state_act_input": 2, "bptt": 2, "num_timestamp": 2, "onli": [2, 3], "tt": 2, "arang": 2, "max": 2, "previou": 2, "__call__": 2, "exp": [2, 3], "e_x": 2, "keepdim": 2, "definit": [2, 3], "clf": 2, "adam": 2, "10": 2, "61": 2, "training_loss_acc": 2, "validation_loss_acc": 2, "x_train": 2, "y_train": 2, "300": 2, "512": 2, "x_test": 2, "y_test": 2, "label": 2, "y_test1": 2, "accuracy_scor": 2, "plot": 2, "train_loss": 2, "plt": 2, "titl": 2, "ylabel": 2, "xlabel": 2, "legend": 2, "show": 2, "without": [2, 3], "ani": [2, 3], "high": [2, 3], "level": [2, 3], "framework": [2, 3], "give": [2, 3], "u": [2, 3], "better": [2, 3], "insight": [2, 3], "probabilist": [2, 3], "process": [2, 3], "machin": [2, 3], "learn": 2, "One": 3, "fundament": 3, "problem": 3, "build": 3, "natur": 3, "languag": 3, "system": 3, "question": 3, "how": 3, "repres": 3, "vital": 3, "practic": 3, "applic": 3, "translat": 3, "answer": 3, "inform": 3, "retriev": 3, "summar": 3, "analysi": 3, "text": 3, "speech": 3, "etc": 3, "simplest": 3, "wai": 3, "independ": 3, "drawback": 3, "represent": 3, "column": 3, "": 3, "length": 3, "equal": 3, "vocabulari": 3, "huge": 3, "But": 3, "unfortun": 3, "do": 3, "encod": 3, "notion": 3, "similar": 3, "other": 3, "relationship": 3, "between": 3, "eg": 3, "our": 3, "space": 3, "would": 3, "want": 3, "bank": 3, "financ": 3, "closer": 3, "while": 3, "same": 3, "both": 3, "being": 3, "significantli": 3, "far": 3, "watermelon": 3, "thei": 3, "ar": 3, "close": 3, "order": 3, "overcom": 3, "need": 3, "altern": 3, "amongst": 3, "gener": 3, "base": 3, "hypothesi": 3, "can": 3, "deriv": 3, "distribut": 3, "appear": 3, "seemingli": 3, "simpl": 3, "idea": 3, "most": 3, "influenti": 3, "success": 3, "modern": 3, "nlp": 3, "The": [3, 7], "low": 3, "dimension": 3, "much": 3, "smaller": 3, "than": 3, "let": 3, "v": 3, "total": 3, "d": 3, "store": 3, "respect": 3, "r": 3, "given": 3, "setup": 3, "two": 3, "centr": 3, "like": 3, "occur": 3, "continu": 3, "bag": 3, "Of": 3, "cbow": 3, "surround": 3, "rest": 3, "manner": 3, "specifi": 3, "where": 3, "u_w": 3, "correspond": 3, "likewis": 3, "c": 3, "probabl": 3, "o": 3, "v_c": 3, "specif": 3, "u_o": 3, "assign": 3, "ideal": 3, "more": 3, "least": 3, "express": 3, "mathemat": 3, "pair": 3, "have": 3, "hat": 3, "vocab": 3, "expect": 3, "Will": 3, "actual": 3, "predict": 3, "quantifi": 3, "good": 3, "bad": 3, "cross": 3, "entropi": 3, "particular": 3, "accordingli": 3, "loss": 3, "l_": 3, "sum_": 3, "mathcal": 3, "y_w": 3, "sinc": 3, "reduc": 3, "y_o": 3, "callout": 3, "info": 3, "now": 3, "If": 3, "perfect": 3, "further": 3, "awai": 3, "larger": 3, "function": 3, "minimis": 3, "corpu": 3, "train": 3, "sequenc": 3, "w_": 3, "m": 3, "k": 3, "posit": 3, "integ": 3, "take": 3, "valu": [3, 7], "l": 3, "j": 3, "p_": 3, "It": 3, "observ": 3, "likelihood": 3, "defin": 3, "next": 3, "guid": 3, "x_": 3, "n": 3, "learningr": 3, "nabla_": 3, "section": 3, "begin": 3, "first": 3, "variant": 3, "dfrac": 3, "u_x": 3, "y_x": 3, "so": 3, "what": 3, "doe": 3, "tell": 3, "subtract": 3, "sens": 3, "multipli": 3, "wa": 3, "short": 3, "make": [3, 7], "essenc": 3, "should": 3, "wherea": 3, "select": 3, "veri": 3, "differ": 3, "also": 3, "interest": 3, "u_1": 3, "u_2": 3, "u_": 3, "rewrit": 3, "abov": 3, "term": 3, "enabl": 3, "write": 3, "vectoris": 3, "therebi": 3, "improv": 3, "There": 3, "case": 3, "here": 3, "neq": 3, "combin": 3, "y_1": 3, "y_2": 3, "y_": 3, "thing": 3, "fact": 3, "computation": 3, "expens": 3, "techniqu": 3, "describ": 3, "address": 3, "scale": 3, "ad": [3, 7], "estim": 3, "target": 3, "factor": 3, "non": 3, "dissimilar": 3, "prohibit": 3, "massiv": 3, "avoid": 3, "introduc": 3, "To": 3, "see": 3, "revisit": 3, "instead": 3, "choos": 3, "small": 3, "w_1": 3, "w_2": 3, "w_k": 3, "Their": 3, "sigma": 3, "sigmoid": 3, "partial": 3, "With": 3, "chang": 3, "modifi": 3, "nabla": 3, "when": 3, "those": 3, "w_t": 3, "leq": 3, "arbitrari": 3, "could": 3, "simpli": 3, "individu": 3, "As": 3, "v_": 3, "v_w": 3, "templat": 3, "cours": 3, "sentiment": 3, "treebank": 3, "dataset": 3, "start": 3, "import": [3, 7], "librari": 3, "top": [3, 7], "sever": 3, "formula": 3, "earlier": 3, "rate": 3, "tune": 3, "desir": 3, "impli": 3, "ones": 3, "togeth": 3, "push": 3, "even": 3, "llm": 3, "ml": 3, "These": 3, "downstream": 3, "summaris": 3, "new": 7, "version": 7, "2020": 7, "12": 7, "28": 7, "beta22": 7, "furo": 7, "fairli": 7, "straightforward": 7, "site": 7, "wide": 7, "announc": 7, "aka": 7, "banner": 7, "page": 7, "websit": 7, "done": 7, "kei": 7, "html_theme_opt": 7, "your": 7, "conf": 7, "py": 7, "file": 7, "html": 7, "includ": 7, "em": 7}, "objects": {}, "objtypes": {}, "objnames": {}, "titleterms": {"home": 0, "past": [0, 6], "experi": [0, 6], "natur": 1, "languag": 1, "process": 1, "recurr": 2, "neural": 2, "network": 2, "task": 2, "loss": 2, "function": 2, "cross": 2, "entropi": 2, "optim": 2, "ml": 2, "model": [2, 3], "initialis": 2, "ad": 2, "layer": 2, "fit": 2, "train": 2, "batch": 2, "test": 2, "forward": 2, "pass": 2, "backward": 2, "predict": [2, 7], "rnn": 2, "activ": 2, "sigmoid": 2, "softmax": [2, 3], "dataset": 2, "build": 2, "evalu": 2, "summari": [2, 3], "refer": [2, 3], "word2vec": 3, "notat": 3, "intuit": 3, "code": 3, "structur": 3, "learn": 3, "algorithm": 3, "stochast": 3, "gradient": 3, "descent": 3, "naiv": 3, "wrt": 3, "center": 3, "word": 3, "vector": 3, "outsid": 3, "neg": 3, "sampl": 3, "matrix": 3, "skip": 3, "gram": 3, "accumul": 3, "over": 3, "an": 3, "entir": 3, "context": 3, "window": 3, "us": 3, "numpi": 3, "result": 3, "visualis": 3, "ada": 4, "ecom": 5, "analyt": 7, "iot": 7, "imag": 8, "segment": 8, "comput": 9, "vision": 9}, "envversion": {"sphinx.domains.c": 3, "sphinx.domains.changeset": 1, "sphinx.domains.citation": 1, "sphinx.domains.cpp": 9, "sphinx.domains.index": 1, "sphinx.domains.javascript": 3, "sphinx.domains.math": 2, "sphinx.domains.python": 4, "sphinx.domains.rst": 2, "sphinx.domains.std": 2, "sphinx": 58}, "alltitles": {"Home": [[0, "home"]], "Past Experiences": [[0, "past-experiences"], [6, "past-experiences"]], "Natural Language Processing": [[1, "natural-language-processing"]], "Recurrent Neural Networks": [[2, "recurrent-neural-networks"]], "Task": [[2, "task"]], "Loss Function: Cross Entropy Loss": [[2, "loss-function-cross-entropy-loss"]], "Optimizers": [[2, "optimizers"]], "ML Model": [[2, "ml-model"]], "Model: Neural Network": [[2, "model-neural-network"]], "Initialise": [[2, "initialise"], [2, "id1"]], "Adding layers": [[2, "adding-layers"]], "Fit the model": [[2, "fit-the-model"]], "Train on a batch": [[2, "train-on-a-batch"]], "Test on a batch": [[2, "test-on-a-batch"]], "Forward Pass": [[2, "forward-pass"], [2, "id2"]], "Backward Pass": [[2, "backward-pass"], [2, "id3"]], "Predict": [[2, "predict"]], "Layers: RNN": [[2, "layers-rnn"]], "Activation: Sigmoid": [[2, "activation-sigmoid"]], "Activation: Softmax": [[2, "activation-softmax"]], "Dataset": [[2, "dataset"]], "Build and fit the Model": [[2, "build-and-fit-the-model"]], "Evaluation": [[2, "evaluation"]], "Summary": [[2, "summary"], [3, "summary"]], "References": [[2, "references"], [3, "references"]], "Word2Vec": [[3, "word2vec"]], "Word2Vec Model": [[3, "word2vec-model"]], "Notation": [[3, "notation"]], "Model": [[3, "model"]], "Intuition": [[3, "intuition"]], "Code structuring": [[3, "code-structuring"]], "Learning Algorithm: Stochastic Gradient Descent": [[3, "learning-algorithm-stochastic-gradient-descent"]], "Naive-Softmax: Gradient wrt Center word vector": [[3, "naive-softmax-gradient-wrt-center-word-vector"]], "Naive-Softmax: Gradient wrt outside word vectors": [[3, "naive-softmax-gradient-wrt-outside-word-vectors"]], "Negative Sampling": [[3, "negative-sampling"]], "Negative-Sampling: Gradient wrt Center word vector": [[3, "negative-sampling-gradient-wrt-center-word-vector"]], "Negative-Sampling: Gradient wrt Outside word Matrix": [[3, "negative-sampling-gradient-wrt-outside-word-matrix"]], "Skip-gram: Accumulated Gradients over an entire context window": [[3, "skip-gram-accumulated-gradients-over-an-entire-context-window"]], "Code: Word2vec using Numpy": [[3, "code-word2vec-using-numpy"]], "Results: Visualisation": [[3, "results-visualisation"]], "ADAS": [[4, "adas"]], "Ecom": [[5, "ecom"]], "Predictive Analytics in IoT": [[7, "predictive-analytics-in-iot"]], "Image Segmentation": [[8, "image-segmentation"]], "Computer Vision": [[9, "computer-vision"]]}, "indexentries": {}})