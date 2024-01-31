import numpy as np
from .base.graph import Graph
from itertools import zip_longest


def check_data(data):
  
  supported_types = (int, float, list, np.ndarray)
  if type(data) in supported_types:
    if not isinstance(data, np.ndarray):
      data = np.array(data)
    try:
      data = data.astype(float)
    except ValueError:
      raise TypeError("Whoa! Elements of data should be float or at least look like they can be turned into float. What's goin' on?")
  else:
    raise TypeError(f"Oops! Expected data of types {supported_types}, but got {type(data)}. You're throwin' me a curveball here!")
  return data

def unbroadcast_data(data, orig_data_shape, broadcasted_shape):

  def determine_sum_axes(orig_data_shape, broadcasted_shape):
      
      axes_to_be_summed = []
      zipped = list(zip_longest(tuple(reversed(broadcasted_shape)), tuple(reversed(orig_data_shape)), fillvalue=None))
      for dim, (dim_broadcasted, dim_orig) in enumerate(reversed(zipped)):
          if dim_broadcasted != dim_orig:
              axes_to_be_summed.append(dim)
      return tuple(axes_to_be_summed)

  if broadcasted_shape is not None:
      axes_to_be_summed = determine_sum_axes(orig_data_shape, broadcasted_shape)
      unbroadcasted_data = np.sum(data, axis=axes_to_be_summed)
  else:
      unbroadcasted_data = data
  return unbroadcasted_data

def current_graph():
  
  if Graph.graph is None:
    from  .config import GRAPH_GB
    graph = GRAPH_GB
  else:
    graph = Graph.graph
  return graph


class new_graph:
  
  def __enter__(self):
    Graph.graph = Graph()
  
  def __exit__(self, exc_type, exc_value, exc_traceback):
    Graph.graph = None


class no_track:
  
  def __init__(self):
    self.graph = current_graph()

  def __enter__(self):
    self.graph.track = False
  
  def __exit__(self, exc_type, exc_value, exc_traceback):
    self.graph.track = True


def validate_gradient(analytical_grads, calculated_grads, epsilon, print_vals):
  
  dist = np.linalg.norm(analytical_grads-calculated_grads)/(np.linalg.norm(analytical_grads) + np.linalg.norm(calculated_grads))
  if print_vals:
    print("Gradient Check Distance:", dist)
    if dist<epsilon:
      print("Gradient Vibe PASSED")
    else:
      print("Gradient Vibe FAILED")
  return dist


def tweak_parameters(analytical_grads, calculated_grads, params, get_loss, epsilon):
  
  for param in params:
    if param.requires_grad:
      if not(isinstance(param.grad, np.ndarray)):
        param.grad = np.array(param.grad)
      for idx in np.ndindex(param.shape):
        with no_track():
          param.data[idx]+=epsilon 
          loss1 = get_loss()
          param.data[idx]-=(2*epsilon) 
          loss2 = get_loss()
          param.data[idx]+=epsilon 
        calculated_grads.append(param.grad[idx])
        analytical_grads.append((loss1.data-loss2.data)/(2*epsilon))
    param.zero_grad() 

def grad_check(model, inputs, targets, loss_fn, epsilon=1e-7, print_vals=True):
  
  params = model.parameters()
  analytical_grads = []
  calculated_grads = []

  for param in params:
    param.zero_grad()

  def get_loss():
    outputs = model(inputs)
    loss = loss_fn(outputs, targets)
    return loss

  with new_graph():
    loss = get_loss()
    loss.backward()
    tweak_parameters(analytical_grads, calculated_grads, params, get_loss, epsilon)

  analytical_grads = np.array(analytical_grads)
  calculated_grads = np.array(calculated_grads)
  return validate_gradient(analytical_grads, calculated_grads, epsilon, print_vals)


def validate_function_gradient(fn, inputs, params, targets=None, loss_fn=None, epsilon=1e-7, print_vals=True, **kwargs):
  
  if loss_fn is None:
    from ..nn.loss import MSE
    loss_fn = MSE()
  analytical_grads = []
  calculated_grads = []

  for param in params:
    param.zero_grad()

  def get_loss(targets=targets):
    outputs = fn(*inputs, **kwargs)
    if targets is None:
      from .whalor import Whalor as Whalor
      targets = Whalor(np.ones(outputs.shape))
    loss = loss_fn(outputs, targets)
    return loss
  
  with new_graph():
    loss = get_loss()
    loss.backward()
    tweak_parameters(analytical_grads, calculated_grads, params, get_loss, epsilon)

  analytical_grads = np.array(analytical_grads)
  calculated_grads = np.array(calculated_grads)
  return validate_gradient(analytical_grads, calculated_grads, epsilon, print_vals)
