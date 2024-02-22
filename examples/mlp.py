# import ( we don't need to import all of them, but it's just for the sake of the example)

from whalegrad.nn.layers.activations import ReLU, sigmoid, tanh
from whalegrad.nn.layers.linear import Linear
from whalegrad.nn.loss import  SoftmaxCE, BinaryCrossEntropy
from whalegrad.nn.layers.model import Model
from whalegrad.nn.optim import Adam, Momentum, RMSProp, SGD
from whalegrad.engine.whalor import Whalor
from whalegrad.nn.layers.containers import Sequential
from whalegrad.engine.toolbox import grad_check
from whalegrad.nn.layers.essential import get_batches
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from sklearn.datasets import make_moons
from whalegrad  import nn 
## Load the dataset

# make moon form sklearn

X, y = make_moons(n_samples=1000, noise=0.1)
X_train,X_test ,y_train ,y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# convert into whalor (tensor)

X_train, X_test = Whalor(X_train), Whalor(X_test)
y_train, y_test = Whalor(y_train.reshape(800,1)), Whalor(y_test.reshape(200,1))

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# define the Module

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

# define the loss function
# configure the Module
Module = MLP()
loss_fn = BinaryCrossEntropy()
optim = Adam(Module.parameters(), 0.05)

# train the Module loop
num_iter = 100
# training loop
def train(optim, Module=Module, num_iter=num_iter, loss_list=None, print_freq=1, print_vals=False):
  for i in range(num_iter):
    optim.zero_grad()
    outputs = Module(X_train)
    loss = loss_fn(outputs, y_train)
    if loss_list is not None:
      loss_list.append(loss.data)
    loss.backward()
    optim.step()
    if i%print_freq==0 and print_vals:
      print(f"iter {i+1}/{num_iter}\nloss: {loss}\n")

# train the Module 
train(optim, print_vals=True)


#prediction test 

# evaluate the Module
with Module.eval():
  test_outputs = Module(X_test)
  preds = np.where(test_outputs.data>=0.5, 1, 0)
  
#print the result 

print(classification_report(y_test.data.astype(int).flatten(), preds.flatten()))

#result accuracy

print(accuracy_score(y_test.data.astype(int).flatten(), preds.flatten()))