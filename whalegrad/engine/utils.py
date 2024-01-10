import numpy as np
from graph import Graph
from itertools import zip_longest

def preprocess_numeric_data(data):
    
    supported_types = (int, float, list, np.ndarray)

    if not isinstance(data, supported_types):
        raise TypeError(f"Expected data of types {supported_types} instead got {type(data)}")

    if not isinstance(data, np.ndarray):
        data = np.array(data)

    try:
        data = data.astype(float)
    except ValueError:
        raise TypeError("Elements of data should be of type float or be typecastable to float")

    return data


def unbroadcast_data(data, orig_data_shape, broadcasted_shape):
    

    def get_axes_to_be_summed(orig_data_shape, broadcasted_shape):
    
        return tuple(dim for dim, (dim_broadcasted, dim_orig) in enumerate(zip_longest(reversed(broadcasted_shape), reversed(orig_data_shape), fillvalue=None)) if dim_broadcasted != dim_orig)

    axes_to_be_summed = get_axes_to_be_summed(orig_data_shape, broadcasted_shape)
    unbroadcasted_data = np.sum(data, axis=axes_to_be_summed) if broadcasted_shape is not None else data

    return unbroadcasted_data


def current_graph():
    '''Returns the graph that is in use

    If Graph.graph is None, the global graph GRAPH_GB is used

    Returns:
      Graph object that is currently used
    '''
    from .. import GRAPH_GB
    return Graph.graph if Graph.graph is not None else GRAPH_GB

class new_graph:
  
  def __enter__(self):
    Graph.graph = Graph()
  
  def __exit__(self, exc_type, exc_value, exc_traceback):
    Graph.graph = None

class NoTrack:
    
    def __enter__(self):
      current_graph().track = False

    def __exit__(self, exc_type, exc_value, exc_traceback):
        current_graph().track = True

def compare_gradients(analytical_grads, calculated_grads, epsilon, print_vals=True):
  
    norm_analytical = np.linalg.norm(analytical_grads)
    norm_calculated = np.linalg.norm(calculated_grads)
    
    dist = np.linalg.norm(analytical_grads - calculated_grads) / (norm_analytical + norm_calculated)
    
    if print_vals:
        print("Gradient Check Distance:", dist)
        print("Gradient Check", "PASSED" if dist < epsilon else "FAILED")
    
    return dist


def calculate_numerical_gradients(analytical_grads, calculated_grads, params, get_loss, epsilon):
    '''Calculates numerical gradients by wiggling parameters and compares with analytical gradients.

    For each parameter in params, the value is wiggled by epsilon, and the loss is calculated.
    Similarly, the value is wiggled by -2*epsilon to get another loss. Using these two losses,
    the analytical gradient is calculated and appended to analytical_grads. The gradient in the
    parameter is appended to calculated_grads.

    Args:
        analytical_grads (list of int or float): Gradients that are calculated analytically
          by wiggling the parameters
        calculated_grads (list of int or float): Gradients that are calculated through
          backpropagation
        params (list of Tensor): All params that need to be wiggled
        get_loss: Function that is used to calculate the loss
        epsilon (float): The amount by which params need to be wiggled
    '''
    for param in params:
        if param.requires_grad:
            if not isinstance(param.grad, np.ndarray):
                param.grad = np.array(param.grad)
            
            for idx in np.ndindex(param.shape):
                with NoTrack():
                    param.data[idx] += epsilon  # PLUS
                    loss1 = get_loss()
                    
                    param.data[idx] -= 2 * epsilon  # MINUS
                    loss2 = get_loss()
                    
                    param.data[idx] += epsilon  # ORIGINAL
                
                calculated_grads.append(param.grad[idx])
                analytical_grads.append((loss1.data - loss2.data) / (2 * epsilon))
                
            param.zero_grad()  # to prevent any side effects

def grade_check(model, inputs, targets, loss_fn, epsilon=1e-7,print_vals=True):
    
    params = model.parameters()
    analytical_grads = []
    calculated_grads = []

    def get_loss():
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
        return loss

    with new_graph():
        loss = get_loss()
        loss.backward()
        calculate_numerical_gradients(analytical_grads, calculated_grads, params, get_loss, epsilon)

    analytical_grads = np.array(analytical_grads)
    calculated_grads = np.array(calculated_grads)
    return compare_gradients(analytical_grads, calculated_grads, epsilon, print_vals)


def function_gradients_checker(fn, inputs=None, params=None, targets=None, loss_fn=None, epsilon=1e-7, print_vals=True, **kwargs):
    
    if loss_fn is None:
     from ..nn.loss import MSE
     loss_fn = MSE()
    analytical_grads = []
    calculated_grads = []

    for param in params:
     
     param.zero_grad()
 
    # def get_loss(targets=targets):
    #   outputs = fn(*inputs, **kwargs)
    #   if targets is None:
    #     from .value import Tensor as tensor
    #     targets = tensor(np.ones(outputs.shape))
    #   loss = loss_fn(outputs, targets)
    #   return loss
   
    # with new_graph():
    #   loss = get_loss()
    #   loss.backward()
    #   calculate_numerical_gradients(analytical_grads, calculated_grads, params, get_loss, epsilon)
  
    # analytical_grads = np.array(analytical_grads)
    # calculated_grads = np.array(calculated_grads)
    # return compare_gradients(analytical_grads, calculated_grads, epsilon, print_vals)
 