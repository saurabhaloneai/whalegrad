# lest build the dropout Module
from copy import deepcopy
from whalegrad.engine.whalor import Whalor
import numpy as np
from whalegrad.engine.functions import dot
from whalegrad.engine.functions import Action
from .base import Core



class Dropout(Core, Action):
  
  def __init__(self, prob):
    Core.__init__(self)
    assert prob>0 and prob<=1
    self.prob = prob
  
  def forward(self, inputs):
    
    if self.eval:
      filter = np.ones(inputs.shape) 
      filter = np.where(np.random.random(inputs.shape)<self.prob, 1, 0)
    inputs, filter = self.get_Whalors(inputs, filter)
    if not(self.eval): 
      result = (inputs.data*filter.data)/self.prob
    else:
      result = inputs.data
    return self.get_result_whalor(inputs.data, inputs, filter)
  
  def backward(self, inputs, filter):
    
    if not(self.eval):
      inputs.set_grad_fn(lambda ug:(ug*filter.data)/self.prob)
    inputs.set_grad_fn(lambda ug:ug)
  
  def __repr__(self):
    return f'Dropout(prob={self.prob})'
  
  def __str__(self):
    return f'Dropout(prob={self.prob})'