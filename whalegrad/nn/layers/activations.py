from .base import Core
from whalegrad.engine.functions import Action
import numpy as np


# <------------RELU------------>
class ReLU(Core, Action):
  
  def forward(self, inputs):
    
    inputs = self.get_whalors(inputs)
    return self.get_result_whalor(np.maximum(0, inputs.data), inputs)
  
  def backward(self, inputs):
    
    inputs.set_grad_fn(lambda ug:np.where(inputs.data>=0, 1, 0)*ug)

  def __repr__(self):
    return 'ReLU()'
  
  def __str__(self):
    return 'ReLU'


# <------------SIGMOID------------>
class sigmoid(Core, Action):
  
  def forward(self, inputs):
    
    inputs = self.get_whalors(inputs)
    return self.get_result_whalor(1/(1+np.exp(-inputs.data)), inputs)
  
  def backward(self, inputs):
    
    result = 1/(1+np.exp(-inputs.data))
    inputs.set_grad_fn(lambda ug:(result*(1-result))*ug)

  def __repr__(self):
    return 'Sigmoid()'
  
  def __str__(self):
    return 'Sigmoid'


# <------------TANH------------>
class tanh(Core, Action):
  
  def forward(self, inputs):
    
    inputs = self.get_whalors(inputs)
    return self.get_result_whalor(np.tanh(inputs.data), inputs)
  
  def backward(self, inputs):
    
    result = np.tanh(inputs.data)
    inputs.set_grad_fn(lambda ug:(1-np.power(result,2))*ug)
  
  def __repr__(self):
    return 'Tanh()'
  
  def __str__(self):
    return 'Tanh'


# <------------SOFTMAX------------>
class softmax(Core, Action):
  
  def __init__(self, axis):
    
    self.axis = axis

  def forward(self, inputs):
    
    inputs = self.get_whalors(inputs)
    result = self.calc_softmax(inputs.data, axis=self.axis)
    return self.get_result_whalor(result, inputs)
  
  def backward(self, inputs):
    
    def softmax_grad(arr, ug_slices): 
      local_grad = -np.broadcast_to(arr, (arr.size, arr.size))
      np.fill_diagonal(local_grad, 1+np.diagonal(local_grad))
      local_grad = local_grad*arr.reshape(arr.size, 1)
      result = np.dot(local_grad, ug_slices.pop(0))
      return result
    
    def get_ug_slices(arr, ug_slices):
      ug_slices.append(arr)

    def grad_backward(ug):
      result = np.apply_along_axis(self.calc_softmax, self.axis, inputs.data)
      ug_slices = []
      np.apply_along_axis(get_ug_slices, self.axis, ug, ug_slices)
      grads = np.apply_along_axis(softmax_grad, self.axis, result, ug_slices)
      return grads

    inputs.set_grad_fn(grad_backward)
  
  @staticmethod
  def calc_softmax(arr, axis=None):
    
    exponentiated = np.exp(arr-np.max(arr, axis=axis, keepdims=True))
    sum_val = np.sum(exponentiated, axis=axis, keepdims=True)
    return exponentiated/sum_val
  
  def __repr__(self):
    return f'Softmax(axis={self.axis})'
  
  def __str__(self):
    return 'Softmax'


# <------------LEAKYRELU------------>
class LeakyReLU(Core, Action):
  
  def __init__(self, leak=0.01):
    
    self.leak = leak

  def forward(self, inputs):
    
    inputs = self.get_whalors(inputs)
    arr = inputs.data
    return self.get_result_whalor(np.where(arr>=0, arr, self.leak*arr), inputs)
  
  def backward(self, inputs):
    
    inputs.set_grad_fn(lambda ug: np.where(inputs.data>=0, 1, self.leak)*ug)

  def __repr__(self):
    return f'LeakyReLU(leak={self.leak})'
  
  def __str__(self):
    return 'LeakyReLU'
  
#---------Swish----------#

class Swish(Core, Action):

    def __init__(self, beta=1.0):
        self.beta = beta

    def forward(self, inputs):
        inputs = self.get_whalors(inputs)
        result = inputs.data / (1 + np.exp(-self.beta * inputs.data))
        return self.get_result_whalor(result, inputs)

    def backward(self, inputs):
        exp_beta_x = np.exp(-self.beta * inputs.data)
        sigmoid_x = 1 / (1 + exp_beta_x)
        swish_grad = (self.beta * exp_beta_x * inputs.data + sigmoid_x) / (1 + exp_beta_x)**2
        inputs.set_grad_fn(lambda ug: swish_grad * ug)

    def __repr__(self):
        return f'Swish(beta={self.beta})'

    def __str__(self):
        return 'Swish'
      
#--------SwiGLU--------#

class SwiGLU(Core, Action):

    def __init__(self, beta=1.0):
        self.beta = beta

    def forward(self, inputs):
        inputs = self.get_whalors(inputs)
        gate = 1 / (1 + np.exp(-self.beta * inputs.data))
        result = inputs.data * gate
        return self.get_result_whalor(result, inputs)

    def backward(self, inputs):
        exp_beta_x = np.exp(-self.beta * inputs.data)
        sigmoid_x = 1 / (1 + exp_beta_x)
        swiglu_gate_grad = self.beta * exp_beta_x * sigmoid_x / (1 + exp_beta_x)**2
        inputs.set_grad_fn(lambda ug: swiglu_gate_grad * ug)

    def __repr__(self):
        return f'SwiGLU(beta={self.beta})'

    def __str__(self):
        return 'SwiGLU'