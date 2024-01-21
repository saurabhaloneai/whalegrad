
from whalegrad.nn.layers.activations import ReLU, Sigmoid, Tanh
from whalegrad.nn.layers.base import Linear
from whalegrad.nn.loss import BCE
from whalegrad.nn.layers.model import Model
from whalegrad.nn.optim import Adam, Momentum, RMSProp, GD
from whalegrad.engine.whalor import Whalor
from whalegrad.nn.layers.base import Sequential
from sklearn.datasets import make_moons, make_blobs
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import numpy as np


X, y = make_moons(n_samples=100, noise=0.1)

y = y*2 - 1 # make y be -1 or 1
# visualize in 2D
plt.figure(figsize=(5,5))
plt.scatter(X[:,0], X[:,1], c=y, s=20, cmap='jet')

X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X,y)
num_train, num_test = 750, 250 # number of train and test examples
num_iter = 50 # number of iterations

X_train, X_test = Whalor(X_train_orig[:num_train,:]), Whalor(X_test_orig[:num_test,:])
y_train, y_test = Whalor(y_train_orig[:num_train].reshape(num_train,1)), Whalor(y_test_orig[:num_test].reshape(num_test,1))

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

# model = NN()
# loss_fn = BCE()
# optim = Adam(model.parameters(), 0.05) 

# model = MLP(2, [16, 16, 1]) # 2-layer neural network
# print(model)
# print("number of parameters", len(model.parameters()))

# # optimization
# for k in range(100):
    
#     # forward
#     total_loss, acc = loss()
    
#     # backward
#     model.zero_grad()
#     total_loss.backward()
    
#     # update (sgd)
#     learning_rate = 1.0 - 0.9*k/100
#     for p in model.parameters():
#         p.data -= learning_rate * p.grad
    
#     if k % 1 == 0:
#         print(f"step {k} loss {total_loss.data}, accuracy {acc*100}%")
        
        
# visualize decision boundary

model = NN()
loss_fn = BCE()
optim = Adam(model.parameters(), 0.05) 

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


h = 0.25
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Xmesh = np.c_[xx.ravel(), yy.ravel()]
inputs = [list(map(Value, xrow)) for xrow in Xmesh]
scores = list(map(model, inputs))
Z = np.array([s.data > 0 for s in scores])
Z = Z.reshape(xx.shape)

fig = plt.figure()
plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()        