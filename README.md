# whalegrad 

'Whalegrad' - A Lightweight Deep Learning Framework

Welcome to whalegrad, my personal deep learning framework project. This is learning project for me. and yeah I recreate the wheel. (readme file written by me and modified by chatgpt :) 

![](https://github.com/saurabhaloneai/whalegrad/blob/main/images/whalegrad.png)


Features:

* Lightweight Framework: Whalegrad is a compact deep learning framework built entirely in Python, spanning just under 1000 lines of code. It's designed to be easy to understand and modify.

* Inspired by the Greats: Andrej Karpathy's(GOD) micrograd and tinygrad by geohot(hack) – Their work inspired me to create something of my own.

* Multi-Dimensional Array Support: just like tensor it supports the whalor. it is nothing but just the fancy name 

* Automatic Gradient Computation: yeah it supoorts the autograd 

* Functionality: is it have lots of activation fucntions, loss fucntions, optims and new embeddings.


## Get start with autograd:

### Install 

```
pip install whalegrad

```
```
#import
from whalegrad.engine.whalor import Whalor

a = Whalor([5], requires_grad =True,)
b = Whalor([4], requires_grad = True)
c = a * b
g = c * b 
f = g - a
k = f * a + b

# print(a.shape)

k.backward([1])

# print(a)

print(a.grad)
print(b.grad) 

```

## MLP 

```
#import 

from whalegrad.nn.layers.activations import ReLU, sigmoid, tanh
from whalegrad.nn.layers.linear import Linear
from whalegrad.nn.loss import  SoftmaxCE, BinaryCrossEntropy
from whalegrad.nn.layers.Module import Module
from whalegrad.nn.optim import Adam, Momentum, RMSProp, SGD
from whalegrad.engine.whalor import Whalor
from whalegrad.nn.layers.containers import Sequential
from whalegrad.engine.toolbox import grad_check
from whalegrad.nn.layers.essential import get_batches

```


```
#build the MLP model 

class MLP(Model):

  def __init__(self):
    self.stack = Sequential(
      Linear(2,100),
      ReLU(),
      Linear(100,1),
      sigmoid()
    )
  
  def forward(self, inputs):
    return self.stack(inputs)

```


```
#train the model

num_iter = 100

def train(optim, Model=Model, num_iter=num_iter, loss_list=None, print_freq=1, print_vals=False):
  for i in range(num_iter):
    optim.zero_grad()
    outputs = Model(X_train)
    loss = loss_fn(outputs, y_train)
    if loss_list is not None:
      loss_list.append(loss.data)
    loss.backward()
    optim.step()
    if i%print_freq==0 and print_vals:
      print(f"iter {i+1}/{num_iter}\nloss: {loss}\n")

train(optim, print_vals=True)

```
## Credits 

1. micrograd 
2. tinygrad 


## Supported 

### Activation functions 

| Activation      | Description              |
|-----------------|--------------------------|
| ReLU            | Rectified Linear Unit    |
| Sigmoid         | Sigmoid activation       |
| Tanh            | Hyperbolic Tangent       |
| Softmax         | Softmax activation       |
| LeakyReLU       | Leaky Rectified Linear Unit |
| Swish           | Swish activation         |
| SwiGLU          | SwiGLU activation        |


### Loss function 

| Loss Function           | Description                               |
|--------------------------|-------------------------------------------|
| MeanSquaredError (MSE)   | Mean Squared Error                        |
| BinaryCrossEntropy (BCE) | Binary Cross Entropy                      |
| CrossEntropy (CE)        | Cross Entropy                             |
| SoftmaxCE                | Softmax Cross Entropy                     |

### optim 

| Optimizer         | Description                                           |
|-------------------|-------------------------------------------------------|
| SGD               | Stochastic Gradient Descent                           |
| Momentum          | Stochastic Gradient Descent with Momentum             |
| RMSProp           | Root Mean Square Propagation                          |
| Adam              | Adaptive Moment Estimation                            |

