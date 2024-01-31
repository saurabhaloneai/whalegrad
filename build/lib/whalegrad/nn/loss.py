import numpy as np
from whalegrad.engine.functions import sum as _sum, log
from whalegrad.engine.functions import Action
from whalegrad.nn.layers.activations import softmax


class Loss:
  
  def __call__(self, outputs, targets):
    
    return self.forward(outputs, targets)
  
  def get_num_examples(self, outputs_shape):
    
    if len(outputs_shape) in (0, 1):
      return 1
    else:
      return outputs_shape[0] 


# <------------MEANSQUAREDERROR------------>
class MeanSquaredError(Loss):
  
  def forward(self, outputs, targets):
    
    num_examples = self.get_num_examples(outputs.shape)
    cost = (1/(2*num_examples))*_sum((outputs-targets)**2)
    return cost
  
  def __repr__(self):
    return f'MSE()'
  
  def __str__(self):
    return 'MeanSquaredError'


# <------------BINARYCROSSENTROPY------------>
class BinaryCrossEntropy(Loss):
  
  def forward(self, outputs, targets, epsilon=1e-9):
    
    num_examples = self.get_num_examples(outputs.shape)
    entropy = _sum((outputs*log(targets+epsilon)) + ((1-outputs)*(log(1-targets+epsilon))))
    cost = (-1/num_examples)*entropy
    return cost
  
  def __repr__(self):
    return f'BCE()'
  
  def __str__(self):
    return 'BinaryCrossEntropy'


# <------------CROSSENTROPY------------>
class CrossEntropy(Loss):
  
  def forward(self, outputs, targets, epsilon=1e-9):
    
    num_examples = self.get_num_examples(outputs.shape)
    entropy = _sum(targets*log(outputs+epsilon))
    cost = (-1/num_examples)*entropy
    return cost
  
  def __repr__(self):
    return 'CE()'
  
  def __str__(self):
    return 'CrossEntropy'


# <------------SOFTMAXCROSSENTROPY------------>
class SoftmaxCE(Action, Loss):
  
  def __init__(self, axis):
    self.axis = axis

  def forward(self, outputs, targets, epsilon=1e-9):
    
    num_examples = self.get_num_examples(outputs.shape)
    probs = softmax.calc_softmax(outputs.data, axis=self.axis)
    entropy = np.sum(targets.data*np.log(probs+epsilon))
    cost = (-1/num_examples)*entropy
    return self.get_result_whalor(cost, outputs, targets)
  
  def backward(self, outputs, targets):
    
    def sce_backward(ug):
      num_examples = self.get_num_examples(outputs.shape)
      probs = softmax.calc_softmax(outputs.data, axis=self.axis)
      return (ug/num_examples)*(probs-targets.data) # ug is a scalar(1 by default), because loss calculated in forward is a scalar
    outputs.set_grad_fn(sce_backward)
    assert targets.requires_grad is False, 'Targets Tensor should have requires_grad=False'