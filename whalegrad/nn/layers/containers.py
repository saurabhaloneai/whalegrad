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
