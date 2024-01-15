from base import Module
from whalegrad.engine.functions import dot
from base import Param
import numpy as np

class Linear(Module):
  
  def __init__(self, in_features, out_features):
    self.in_features = in_features
    self.out_features = out_features
    self.weights = Param(np.random.randn(in_features, out_features), requires_grad=True)
    self.bias = Param(np.zeros((1, out_features)), requires_grad=True)
  
  def forward(self, inputs):
   
    return dot(inputs, self.weights) + self.bias
  
  def __repr__(self):
    return f'Linear({self.in_features}, {self.out_features})'
  
  def __str__(self):
    return f'Linear in:{self.in_features} out:{self.out_features}'