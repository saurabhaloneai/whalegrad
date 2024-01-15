from whalegrad.engine.whalor import Whalor
from copy import deepcopy


class Container:
  
  def __init__(self):
    self.eval = False
    self.Modules = None

  def __call__(self, inputs):
   
    return self.forward(inputs)

  def parameters(self, as_dict=False):
   
    params = []
    for Module in self.Modules:
      if as_dict:
        params.append(Module.parameters(as_dict))
      else:
        params+=Module.parameters(as_dict)
    return params
  
  def set_eval(self, eval):
    
    self.eval = eval
    for Module in self.Modules:
      Module.set_eval(eval)
  
  def set_params(self, container_params):
    
    for Module_param, Module in zip(container_params, self.Modules):
      Module.set_params(Module_param)
  
  def freeze(self):
    
    for Module in self.Modules:
      Module.freeze()
  
  def unfreeze(self):
    
    for Module in self.Modules:
      Module.unfreeze()
  
  def __repr__(self):
    Modules = []
    for Module in self.Modules:
      Modules.append(f'{Module.__str__()}')
    Modules_repr = ', '.join(Modules)
    return Modules_repr
  
  def __str__(self):
    Modules = []
    for Module in self.Modules:
      Modules.append(f'{Module.__repr__()}')
    Modules_str = ', '.join(Modules)
    return Modules_str


class Module:
  
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
  
  def set_params(self, Module_params):
    
    for attr, param_data in Module_params.items():
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
      state[param_attr].data = 0 # Wanted to set it to None, but it isnt supported by  Whalor, so set it to the next best 0
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