## make the classifier for binary clssification

# TODO :
# 1. linear : ✅
# 2. optim => adam , sgd
# 3. loss => mse, Binary Cross Entropy
# 4. acc => ✅
# 5. activation => sigmoid, relu and tanh
# 6. no grad caln for inference like pytoch
# 7. save and load model 
# 8. add more activation function



from whalegrad.nn.layers.activations import ReLU, sigmoid, tanh
from whalegrad.nn.layers.linear import Linear
from whalegrad.nn.loss import BinaryCrossEntropy
from whalegrad.nn.layers.base import Model
from whalegrad.nn.optim import Adam
from whalegrad.engine.whalor import Whalor
from whalegrad.nn.layers.containers import Sequential

from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# load data
X, y = make_circles(n_samples=1000, noise=0.05, random_state=100)
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X,y)


num_train, num_test = 750, 250 # number of train and test examples
num_iter = 50 # number of iterations


X_train, X_test = Whalor(X_train_orig[:num_train,:]), Whalor(X_test_orig[:num_test,:])
y_train, y_test = Whalor(y_train_orig[:num_train].reshape(num_train,1)), Whalor(y_test_orig[:num_test].reshape(num_test,1))


class NN(Model):
  def __init__(self):
    
    super(NN, self).__init__()
      
    
    self.layer1 =  self.Linear(2,100)
    self.ReLU = ReLU()  
    self.layer2 = self.Linear(100,1)
    self.layer3 = self.sigmoid()
    
  
  def forward(self, inputs):
    return self.layer3(self.layer2(self.layer1(inputs)))
  
  
model = NN()
loss_fn =BinaryCrossEntropy()
optim = Adam(model.parameters(), 0.05)