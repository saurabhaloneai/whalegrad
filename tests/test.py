## make the classifier for binary clssification

# TODO :
# 1. linear : ✅
# 2. optim => adam , sgd ✅
# 3. loss => mse, Binary Cross Entropy ✅
# 4. acc => ✅
# 5. activation => sigmoid, relu and tanh ✅
# 6. no grad caln for inference like pytoch ✅
# 7. save and load model ✅
# 8. add more activation function ✅


from whalegrad.nn.layers.activations import ReLU, Sigmoid, Tanh
from whalegrad.nn.layers.linear import Linear
from whalegrad.nn.loss import BCE
from whalegrad.nn.layers.model import Model
from whalegrad.nn.optim import Adam, Momentum, RMSProp, GD
from whalegrad.engine.whalor import Whalor
from whalegrad.nn.layers.containers import Sequential
from whalegrad.engine.toolbox import grad_check
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

import numpy as np

X, y = make_moons(n_samples=1000, noise=0.05, random_state=100)
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X,y)

num_train, num_test = 750, 250 # number of train and test examples
num_iter = 50 # number of iterations

# data as tensors
X_train, X_test = Whalor(X_train_orig[:num_train,:]), Whalor(X_test_orig[:num_test,:])
y_train, y_test = Whalor(y_train_orig[:num_train].reshape(num_train,1)), Whalor(y_test_orig[:num_test].reshape(num_test,1))


# define the model
class NN(Model):
  def __init__(self):
    self.stack = Sequential(
      Linear(2,100),
      ReLU(),
      Linear(100,1),
      Sigmoid()
    )
  
  def forward(self, inputs):
    return self.stack(inputs)

# configure the model
model = NN()
loss_fn = BCE()
optim = Adam(model.parameters(), 0.05) 

# training loop
def train(optim, model=model, num_iter=num_iter, loss_list=None, print_freq=1, print_vals=False):
  for i in range(num_iter):
    optim.zero_grad()
    outputs = model(X_train)
    loss = loss_fn(outputs, y_train)
    if loss_list is not None:
      loss_list.append(loss.data)
    loss.backward()
    optim.step()
    if i%print_freq==0 and print_vals:
      print(f"iter {i+1}/{num_iter}\nloss: {loss}\n")
      
train(optim, print_vals=True)

adam_losses = []
momentum_losses = []
rms_losses = []
gd_losses = []

losses = [
  adam_losses,
  momentum_losses,
  rms_losses,
  gd_losses
]

optims = [
  Adam,
  Momentum,
  RMSProp,
  GD
]

num_iter = 100
for optim,loss_list in zip(optims, losses):
  model = NN() # Resetting the model, to remove the previously learnt weights
  train(optim(model.parameters(), 0.005), model=model, num_iter=num_iter, loss_list=loss_list)
plt.plot(np.arange(1, num_iter+1, 1), gd_losses, label='Gradient Descent')
plt.plot(np.arange(1, num_iter+1, 1), rms_losses, label='RMSProp')
plt.plot(np.arange(1, num_iter+1, 1), momentum_losses, label='Momentum')
plt.plot(np.arange(1, num_iter+1, 1), adam_losses, label='Adam')
plt.legend()
plt.show()
# evaluate the model
with model.eval():
  test_outputs = model(X_test)
  preds = np.where(test_outputs.data>=0.5, 1, 0)

print(classification_report(y_test.data.astype(int).flatten(), preds.flatten()))

print(accuracy_score(y_test.data.astype(int).flatten(), preds.flatten()))

grad_check(model, X_train, y_train, loss_fn)
