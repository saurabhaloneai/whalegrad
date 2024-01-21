from copy import deepcopy
from whalegrad.engine.whalor import Whalor
import numpy as np
from whalegrad.engine.functions import dot
from whalegrad.engine.functions import Action


class Container:
  
  def __init__(self):
    self.eval = False
    self.layers = None

  def __call__(self, inputs):
    
    return self.forward(inputs)

  def parameters(self, as_dict=False):
    
    params = []
    for layer in self.layers:
      if as_dict:
        params.append(layer.parameters(as_dict))
      else:
        params+=layer.parameters(as_dict)
    return params
  
  def set_eval(self, eval):
    
    self.eval = eval
    for layer in self.layers:
      layer.set_eval(eval)
  
  def set_params(self, container_params):
    
    for layer_param, layer in zip(container_params, self.layers):
      layer.set_params(layer_param)
  
  def freeze(self):
    
    for layer in self.layers:
      layer.freeze()
  
  def unfreeze(self):
    
    for layer in self.layers:
      layer.unfreeze()
  
  def __repr__(self):
    layers = []
    for layer in self.layers:
      layers.append(f'{layer.__str__()}')
    layers_repr = ', '.join(layers)
    return layers_repr
  
  def __str__(self):
    layers = []
    for layer in self.layers:
      layers.append(f'{layer.__repr__()}')
    layers_str = ', '.join(layers)
    return layers_str


class Layer:
  
  def __init__(self):
    self.eval = False

  def __call__(self, inputs):
    
    return self.forward(inputs)

  def parameters(self, as_dict=False):
    
    params = {}
    for attr, val in self.__dict__.items():
      if isinstance(val, Param):
        params[attr] = val.data if as_dict else val
    return params if as_dict else list(params.values())
  
  def set_eval(self, eval):
    
    self.eval = eval
  
  def set_params(self, layer_params):
    
    for attr, param_data in layer_params.items():
      param = self.__getattribute__(attr)
      param.data = param_data
  
  def freeze(self):
    
    for param in self.parameters(as_dict=False):
      param.freeze()
  
  def unfreeze(self):
    
    for param in self.parameters(as_dict=False):
      param.unfreeze()
  
  def __getstate__(self):
    
    state = deepcopy(self.__dict__)
    for param_attr in self.parameters(as_dict=True).keys():
      state[param_attr].data = 0 # Wanted to set it to None, but it isnt supported by Whalor, so set it to the next best 0
    return state
  
  def __setattr__(self, attr, val):
    
    if (isinstance(val, Param)) and (attr in self.__dict__):
      raise AttributeError(f"Attribute {attr} has already been defined, it cannot be defined again for a Param")
    object.__setattr__(self, attr, val)


class Param(Whalor):
  

  def __init__(self, data, requires_grad=False, requires_broadcasting=True):
    super().__init__(data, requires_grad, requires_broadcasting)
    self.__frozen = False
  
  def freeze(self):
    
    self.requires_grad = False
    self.__frozen = True
  
  def unfreeze(self):
    
    if self.__frozen:
      self.requires_grad = True
    self.__frozen = False
  
  def __str__(self):
    return f'Param({super().__str__()})'
  
  def __repr__(self):
    return f'Param({super().__repr__()})'

#-------------------------------------------------------------------------------------------------------------#




class Sequential(Container):
  
  
  def __init__(self, *args):
    self.layers = args
  
  def forward(self, inputs):
    
    for layer in self.layers:
      output = layer(inputs)
      inputs = output
    return output
  
  def __str__(self):
    return f'Sequential(\n{super().__str__()}\n)'
  
  def __repr__(self):
    return f'Sequential(\n{super().__repr__()}\n)'


class Linear(Layer):
  
  def __init__(self, num_in, num_out):
    self.num_in = num_in
    self.num_out = num_out
    self.weights = Param(np.random.randn(num_in, num_out), requires_grad=True)
    self.bias = Param(np.zeros((1, num_out)), requires_grad=True)
  
  def forward(self, inputs):
    
    return dot(inputs, self.weights) + self.bias
  
  def __repr__(self):
    return f'Linear({self.num_in}, {self.num_out})'
  
  def __str__(self):
    return f'Linear in:{self.num_in} out:{self.num_out}'


class Dropout(Layer, Action):
  
  def __init__(self, prob):
    Layer.__init__(self)
    assert prob>0 and prob<=1, 'Probability should be between 0 and 1'
    self.prob = prob
  
  def forward(self, inputs):
    
    if self.eval:
      filter = np.ones(inputs.shape) # dont discard anything, just dummy because if eval or not, backward needs to have filter arg
    else:
      filter = np.where(np.random.random(inputs.shape)<self.prob, 1, 0)
    inputs, filter = self.get_Whalors(inputs, filter)
    if not(self.eval): # Deliberately repeated condition check for eval, because if in eval, it shouldnt be scaled by prob
      result = (inputs.data*filter.data)/self.prob
    else:
      result = inputs.data
    return self.get_result_Whalor(inputs.data, inputs, filter)
  
  def backward(self, inputs, filter):
    
    if not(self.eval):
      inputs.set_grad_fn(lambda ug:(ug*filter.data)/self.prob)
    inputs.set_grad_fn(lambda ug:ug)
  
  def __repr__(self):
    return f'Dropout(prob={self.prob})'
  
  def __str__(self):
    return f'Dropout(prob={self.prob})'


# these imports should be done after defining Layer, Container, Param to avoid circular import
# from .misc import Sequential, Linear, Dropout
# from .conv import Conv2D, Conv3D, MaxPool2D, MaxPool3D