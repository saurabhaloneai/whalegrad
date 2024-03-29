import numpy as np
from whalegrad.engine.toolbox import current_graph


class Optimizer:
  
  def __init__(self, params, lr):
    self.params = params
    self.lr = lr

  def zero_grad(self, all_members=False):
    
    if all_members:
      graph = current_graph()
      graph.zero_grad()
    for param in self.params: # This is done for redundancy, if all_members=True on a graph that's been reset
        param.zero_grad()


class SGD(Optimizer):
  
  def __init__(self, params, lr):
    super().__init__(params, lr)
  
  def step(self):
    '''Updates the params
    '''
    for param in self.params:
      if param.requires_grad:
        param.data -= (self.lr*param.grad)
  
  def __repr__(self):
    return f'GD(params={self.params}, lr={self.lr})'
  
  def __str__(self):
    return f'GD(params={self.params}, lr={self.lr})'


class Momentum(Optimizer):
  
  def __init__(self, params, lr, beta=0.9):
    super().__init__(params, lr)
    self.beta = beta
    self.init_momentum_grads()

  def step(self):
    
    self.update_momentum_grads()
    for param in self.params:
      if param.requires_grad:
        param.data -= (self.lr*param.momentum_grad)
  
  def init_momentum_grads(self):
    
    for param in self.params:
      if param.requires_grad:
        param.momentum_grad = 0
  
  def update_momentum_grads(self):
    
    for param in self.params:
      if param.requires_grad:
        param.momentum_grad = (self.beta*param.momentum_grad) + ((1-self.beta)*param.grad)
  
  def __repr__(self):
    return f'Momentum(params={self.params}, lr={self.lr}, beta={self.beta})'
  
  def __str__(self):
    return f'Momentum(params={self.params}, lr={self.lr}, beta={self.beta})'


class RMSProp(Optimizer):
  
  def __init__(self, params, lr, beta=0.9, epsilon=1e-8):
    super().__init__(params, lr)
    self.beta = beta
    self.epsilon = epsilon
    self.init_rms_grads()

  def step(self):
    
    self.update_rms_grads()
    for param in self.params:
      if param.requires_grad:
        param.data -= (self.lr*(param.grad/(np.sqrt(param.rms_grad) + self.epsilon)))
  
  def init_rms_grads(self):
    
    for param in self.params:
      if param.requires_grad:
        param.rms_grad = 0
  
  def update_rms_grads(self):
    
    for param in self.params:
      if param.requires_grad:
        param.rms_grad = (self.beta*param.rms_grad) + ((1-self.beta)*np.square(param.grad))
  
  def __repr__(self):
    return f'RMSProp(params={self.params}, lr={self.lr}, beta={self.beta}, epsilon={self.epsilon})'
  
  def __str__(self):
    return f'RMSProp(params={self.params}, lr={self.lr}, beta={self.beta}, epsilon={self.epsilon})'


class Adam(Optimizer):
  
  def __init__(self, params, lr, beta1=0.9, beta2=0.999, epsilon=1e-8):
    super().__init__(params, lr)
    self.iter = 0
    self.beta1, self.beta2 = beta1, beta2
    self.epsilon = epsilon
    self.init_adam_grads()
  
  def step(self):
    
    self.iter+=1
    self.update_adam_grads()
    for param in self.params:
      if param.requires_grad:
        bias_corrected_momentum_grad = param.momentum_grad/(1-(self.beta1**self.iter))
        bias_corrected_rms_grad = param.rms_grad/(1-(self.beta2**self.iter))
        param.data -= (self.lr*(bias_corrected_momentum_grad/(np.sqrt(bias_corrected_rms_grad)+self.epsilon)))
  
  def init_adam_grads(self):
    
    for param in self.params:
      if param.requires_grad:
        param.momentum_grad = 0
        param.rms_grad = 0
  
  def update_adam_grads(self):
    
    for param in self.params:
      if param.requires_grad:
        param.momentum_grad = (self.beta1*param.momentum_grad) + ((1-self.beta1)*param.grad)
        param.rms_grad = (self.beta2*param.rms_grad) + ((1-self.beta2)*np.square(param.grad))

  def reset_iter(self):
    
    self.iter = 0
  
  def __repr__(self):
    return f'Adam(params={self.params}, lr={self.lr}, beta1={self.beta1}, beta2={self.beta2}, epsilon={self.epsilon})'
  
  def __str__(self):
    return f'Adam(params={self.params}, lr={self.lr}, beta1={self.beta1}, beta2={self.beta2}, epsilon={self.epsilon})'
    
  