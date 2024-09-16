# General

- <b>PyTorch LogSoftmax vs Softmax for CrossEntropyLoss</b>

	- [StackOverFlow](https://stackoverflow.com/questions/65192475/pytorch-logsoftmax-vs-softmax-for-crossentropyloss)

		- Yes, NLLLoss takes log-probabilities (log(softmax(x))) as input. Why?. Because if you add a nn.LogSoftmax (or F.log_softmax) as the final layer of your model's output, you can easily get the probabilities using torch.exp(output), and in order to get cross-entropy loss, you can directly use nn.NLLLoss. Of course, log-softmax is more stable as you said.

		- And, there is only one log (it's in nn.LogSoftmax). There is no log in nn.NLLLoss.

		- nn.CrossEntropyLoss() combines nn.LogSoftmax() (that is, log(softmax(x))) and nn.NLLLoss() in one single class. Therefore, the output from the network that is passed into nn.CrossEntropyLoss needs to be the raw output of the network (called logits), not the output of the softmax function.

	- [PyTorch NLLLoss](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html)
		- The input given through a forward call is expected to contain log-probabilities of each class.
		- Obtaining log-probabilities in a neural network is easily achieved by adding a LogSoftmax layer in the last layer of your network. You may use CrossEntropyLoss instead, if you prefer not to add an extra layer.