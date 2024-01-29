import dill
from itertools import chain as list_flattener
from .containers import Container
from .base import Core
from whalegrad.engine.toolbox import current_graph


class Model:
  def __call__(self, inputs):
    
    return self.forward(inputs)
  
  def eval(self, no_track=True):
    
    return EvalMode(self, no_track)
  
  def get_layers(self):
    
    layers = {}
    for attr, val in self.__dict__.items():
      if isinstance(val, (Container, Core)):
        layers[attr] = val
    return layers
  
  def parameters(self, as_dict=False):
    
    params = {}
    for attr, layer in self.get_layers().items():
      params[attr] = layer.parameters(as_dict)
    return params if as_dict else list(list_flattener(*params.values()))
  
  def set_eval(self, eval):
    
    for layer in self.get_layers().values():
      layer.set_eval(eval)
  
  def save(self, fpath):
    
    params = self.parameters(as_dict=True)
    with open(fpath, 'wb') as fp:
      dill.dump(params, fp)
    print(f"\nPARAMS SAVED at {fpath}\n")

  def load(self, fpath):
    
    with open(fpath, 'rb') as fp:
      params = dill.load(fp)
    for attr, param in params.items():
      layer = self.__getattribute__(attr)
      layer.set_params(param)
    print(f"\nPARAMS LOADED from {fpath}\n")
  
  def __setattr__(self, attr, val):
    
    if isinstance(val, (Container, Core)) and (attr in self.__dict__):
      raise AttributeError(f"Attribute {attr} has already been defined, it cannot be defined again for a Container/Core")
    object.__setattr__(self, attr, val)
  
  def __repr__(self):
    return f'Model( {[str(layer) for layer in self.get_layers().values()]} )'
  
  def __str__(self):
    return f'Model( {[str(layer) for layer in self.get_layers().values()]} )'


class EvalMode:
  
  def __init__(self, model, no_track):
    self.model = model
    self.no_track = no_track
    self.graph = current_graph()

  def __enter__(self):
    
    if self.no_track:
      self.graph.track = False
    self.model.set_eval(True)
  
  def __exit__(self, exc_type, exc_value, exc_traceback):
    
    if self.no_track:
      self.graph.track = True
    self.model.set_eval(False)