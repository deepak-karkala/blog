---
layout: docs
title: Recurrent Neural Networks from scratch
description: Implementing RNN from scratch using Numpy.
group: getting-started
toc: true
---

### Explained: RNN

Building a ML model involves the following components,

<div class="container py-4 py-md-5 px-4 px-md-3 text-body-secondary">
    <div class="row" >
      <div class="col-lg-3 mb-3">
        <img width="500px" src="../_static/projects/rnn/rnn-ml-model-components.png"></img>
      </div>
    </div>
</div>

In this article, we will implement each of these components from scratch using Numpy and train a RNN model.



#### Task

#### Loss Function: Cross Entropy Loss
```python
class Loss(object):
    def __init__(self): pass

    def loss(self, y, y_pred):
        raise NotImplemented()
    
    def gradient(self, y, y_pred):
        raise NotImplemented()
    
    def accuracy(self, y, y_pred):
        return 0

class CrossEntropyLoss(Loss):
    def __init__(self): pass

    def loss(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - y * np.log(p) - (1 - y) * np.log(1 - p)

    def gradient(self, y, p):
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return -y/p + (1 - y)/(1 - p)
    
    def accuracy(self, y, p):
        y_true = np.argmax(y, axis=1)
        y_pred = np.argmax(p, axis=1)
        return np.sum(y_true == y_pred, axis=0) / len(y_true)

```

#### Optimizers
```python
class Optimizer(object):
    def __init__(self): pass

    def update(self, w, grad_w): pass


class SGD(Optimizer):
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate
    
    def update(self, w, grad_w):
        # Gradient descent
        return w - self.learning_rate * grad_w
```

#### ML Model
#### Model: Neural Network

###### Initialise
```python
class NeuralNetwork():
    def __init__(self, optimizer, loss):
        self.optimizer = optimizer
        self.loss_function = loss
        self.layers = []
        self.metrics = {"training":{"loss":[], "accuracy":[]}, "validation":{"loss":[], "accuracy":[]}}
```
###### Adding layers
```python
def add(self, layer):
    """ Method which adds a layer to the neural network """
    # Set input shape of layer
    if self.layers:
        layer.set_input_shape(shape = self.layers[-1].output_shape())

    # Attach an optimizer if this layer has weights
    if hasattr(layer, 'initialize'):
        layer.initialize(self.optimizer)

    # Add current layer to network
    self.layers.append(layer)
```
###### Fit the model
```python
def fit(self, X, y, n_epochs, batch_size, X_val=None, y_val=None, print_once_every_epochs=None):
""" Trains the model for a fixed number of epochs """
    for epoch in range(n_epochs):
        training = {"loss":[], "accuracy":[]}

        # For each epoch, run over all minibatches
        for X_batch, y_batch in batch_iterator(X, y, batch_size):
            loss, acc = self.train_on_batch(X_batch, y_batch)
            training["loss"].append(loss)
            training["accuracy"].append(acc)
        self.metrics["training"]["loss"].append(np.mean(training["loss"]))
        self.metrics["training"]["accuracy"].append(np.mean(training["accuracy"]))

        # At the end of each epoch, get loss,acc on train and val data
        if X_val is not None and y_val is not None:
            val_loss, val_acc = self.test_on_batch(X_val, y_val)
            self.metrics["validation"]["loss"].append(val_loss)
            self.metrics["validation"]["accuracy"].append(val_acc)

    return self.metrics["training"], self.metrics["validation"]
```
###### Train on a batch
```python
def train_on_batch(self, X, y):
    """ Single gradient update over one batch of samples """
    # Forward pass
    y_pred = self._forward_pass(X, training=True)
    # Compute loss function and accuracy on train data
    loss = np.mean(self.loss_function.loss(y, y_pred))
    acc = self.loss_function.acc(y, y_pred)
    # Gradient wrt input of loss function
    grad_loss = self.loss_function.gradient(y, y_pred)
    # Backward pass (back propagate gradients through entire network)
    self._backward_pass(grad_loss)
    return loss, acc
```

###### Test on a batch
```python
def test_on_batch(self, X, y):
    """ Evaluates the model over a single batch of samples """
    y_pred = self._forward_pass(X, training=False)
    loss = np.mean(self.loss_function.loss(y, y_pred))
    acc = np.mean(self.loss_function.acc(y, y_pred))
    return loss, acc
```
###### Forward Pass
```python
def _forward_pass(self, X, training=True):
    """ Calculate the output of the NN """
    layer_output = X
    for layer in self.layers:
        layer_output = layer.forward_pass(layer_output, training)
    return layer_output
```
###### Backward Pass
```python
def _backward_pass(self, grad):
    """ Back propagation and update the weights in each layer """
    for layer in reversed(self.layers):
        grad = layer.backward_pass(grad)
    return grad
```
###### Predict
```python
def predict(self, X):
    """ Model prediction on samples """
    return self._forward_pass(X, training=False)
```


#### Layers: RNN

Pic: (Page4) 2 units of a RNN layer
<div class="container py-4 py-md-5 px-4 px-md-3 text-body-secondary">
    <div class="row" >
      <div class="col-lg-3 mb-3">
        <img width="800px" src="../_static/projects/rnn/rnn-layer-dim.png"></img>
      </div>
    </div>
</div>

Equation: RNN Layer Forward Pass

Pic: (Page4) Backpropagation of gradient within a single unit of RNN

Equation: RNN Layer Backward Pass


###### Initialise
```python
class RNN(Layer):
    """Recurrent Neural Network layer

    Parameters:
    ----------
    dim_input: int
        Dimension of input
    dim_hidden: int
        Dimension of hidden state
    activation: Activation function
        ReLU() or Softmax()
    bptt_trunc: int
        Number of time steps of gradient back-propagation 
    """

    def __init__(self, input_shape, dim_hidden, activation, bptt_trunc=5):
        self.input_shape = input_shape
        self.dim_input = input_shape[-1]
        self.dim_hidden = dim_hidden
        self.activation = activation
        self.bptt_trunc = bptt_trunc
        self.W_hh = None    # Hidden to hidden weight matrix
        self.W_xh = None    # Input to hidden weight matrix
        self.W_ho = None    # Hidden to output weight matrix
    

    def initialize(self, optimizer):
        # Initialise the weight matrices
        limit = 1 / math.sqrt(self.dim_input)
        self.W_xh = np.random.uniform(-limit, limit, (self.dim_input, self.dim_hidden))
        limit = 1 / math.sqrt(self.dim_hidden)
        self.W_hh = np.random.uniform(-limit, limit, (self.dim_hidden, self.dim_hidden))
        self.W_ho = np.random.uniform(-limit, limit, (self.dim_hidden, self.dim_input))
        # Optimizers
        self.W_xh_opt = copy.copy(optimizer)
        self.W_hh_opt = copy.copy(optimizer)
        self.W_ho_opt = copy.copy(optimizer)

    def num_parameters(self):
        return np.prod(self.W_xh.shape) + np.prod(self.W_hh.shape) + np.prod(self.W_ho.shape)

    def output_shape(self):
        return self.input_shape
```
###### Forward Pass
GIF: (Page6) Forward propagation across time steps

```python
def forward_pass(self, X, training=True):
    batch_size, num_timesteps, dim_input = X.shape
    self.layer_input = X        # Save layer input for backprop

    # Save intermediate states to be used in backprop
    self.state_act_input = np.zeros((batch_size, num_timesteps, self.dim_hidden))
    self.state_act_output = np.zeros((batch_size, num_timesteps + 1, self.dim_hidden))
    self.output = np.zeros((batch_size, num_timesteps, self.dim_input))

    self.state_act_output[:,-1,:] = np.zeros((batch_size, self.dim_hidden))
    # Iterate through time steps
    for t in range(num_timesteps):
        self.state_act_input[:,t,:] = np.matmul(X[:,t,:], self.W_xh) + np.matmul(self.state_act_output[:,t-1,:], self.W_hh)
        self.state_act_output[:,t,:] = self.activation(self.state_act_input[:,t,:])
        self.output[:,t,:] = np.matmul(self.state_act_output[:,t,:], self.W_ho)

    return self.output
```
###### Backward Pass
GIF: (Page6) Backward propagation across time steps for one output time step
GIF: (Page7) Gradient accumulation: Backward propagation over multiple output time steps

```python
 def backward_pass(self, grad):
    batch_size, num_timesteps, dim_output = grad.shape

    # Initialise gradient variables to appropriate size
    grad_inp = np.zeros_like(grad)
    grad_wxh = np.zeros_like(self.W_xh)
    grad_whh = np.zeros_like(self.W_hh)
    grad_who = np.zeros_like(self.W_ho)

    # Backpropagate for t timesteps
    for t in reversed(range(num_timesteps)):
        # Gradient wrt Output matrix (Accumulate over time steps)
        grad_who += np.matmul(self.state_act_output[:,t,:].T, grad[:,t,:])

        # Gradient wrt State at Activation Output
        grad_state_act_output = np.matmul(grad[:,t,:], self.W_ho.T)
        # Gradient wrt State at Activation Input
        grad_state_act_input = grad_state_act_output * self.activation.gradient(self.state_act_input[:,t,:])

        # Gradient wrt Input
        grad_inp[:,t,:] = np.matmul(grad_state_act_input, self.W_xh.T)

        # Backpropagate through time (for each output, t-self.bptt )
        #   For num_timestamps > self.bptt, propagate gradients back only self.bptt steps
        for tt in reversed(np.arange(max(0, t-self.bptt_trunc), t+1)):
            # Gradient wrt Input matrix (Accumulate over time steps)
            grad_wxh += np.matmul(self.layer_input[:,tt,:].T, grad_state_act_input)
            # Gradient wrt State matrix (Accumulate over time steps)
            grad_whh += np.matmul(self.state_act_output[:,tt-1,:].T, grad_state_act_input)

            # Gradient wrt previous state
            grad_state_act_output = np.matmul(grad_state_act_input, self.W_hh.T)
            grad_state_act_input = grad_state_act_output * self.activation.gradient(self.state_act_input[:,tt-1,:])
    
    # Update weights using optimizer
    self.W_xh = self.W_xh_opt.update(self.W_xh, grad_wxh)
    self.W_hh = self.W_hh_opt.update(self.W_hh, grad_whh)
    self.W_ho = self.W_ho_opt.update(self.W_ho, grad_who)

    # Return gradients wrt layer input for num_timesteps timesteps
    return grad_inp
```


#### Activation: Sigmoid
```python
class Sigmoid():
  def __init__(self):
    return

  def __call__(self, x):
    return 1 / (1 + np.exp(-x))

  def gradient(self, x):
    p = self.__call__(x)
    return p * (1 - p)
```

#### Activation: Softmax
```python
class Softmax():
  def __call__(self, x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)

  def gradient(self, x):
    p = self.__call__(x)
    return p * (1 - p)
```

#### Dataset

#### Build and fit the Model
```python
# Model definition
clf = NeuralNetwork(optimizer=Adam(),
                    loss=CrossEntropyLoss())
clf.add(RNN(input_shape=(10, 61), dim_hidden=10, activation=Sigmoid(), bptt_trunc=5))
clf.add(Activation(Softmax()))

# Fit the model to training data
training_loss_acc, validation_loss_acc = clf.fit(X_train, y_train, n_epochs=300, batch_size=512, X_val=X_test, y_val=y_test, print_once_every_epochs=10)

```

#### Evaluation
```python
# Predict labels of the test data
y_pred = np.argmax(clf.predict(X_test), axis=2)
y_test1 = np.argmax(y_test, axis=2)
accuracy = np.mean(accuracy_score(y_test1, y_pred))


# Plot training and validation loss
train_loss = training_loss_acc["loss"]
val_loss = validation_loss_acc["loss"]

plt.plot(range(len(train_loss)), train_loss, label="Training Loss")
plt.plot(range(len(val_loss)), val_loss, label="Validation Loss")
plt.title("Loss Function")
plt.ylabel('Training Loss')
plt.xlabel('Iterations')
plt.legend(['Training Loss', 'Validation Loss'])
plt.show()
```

#### Summary
Implementing RNN from scratch using Numpy, without any high level ML frameworks, gives us better insights into probabilistic models, computation of gradients for SGD and the end to end training process of a Machine Learning Model.


#### References

- [Machine Learning From Scratch](https://github.com/eriklindernoren/ML-From-Scratch)


<script id="MathJax-script" type="text/javascript" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>

