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