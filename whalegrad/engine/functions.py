import numpy as np
from whalegrad.engine.base.node import Node


class Action:
  
  
  def process_inputs(self, inputs):
    
    from .whalor import Whalor
    inputs = list(inputs)
    for i,operand in enumerate(inputs):
      if not isinstance(operand, Whalor):
        inputs[i] = Whalor(operand)
    return tuple(inputs)
  
  def get_Whalors(self, *inputs):
    
    Whalors = self.process_inputs(inputs)
    if len(Whalors)==0:
      return None
    elif len(Whalors)==1:
      return Whalors[0]
    else:
      return Whalors
  
  def get_broadcast_shape(self, *Whalors):
    
    for whals in Whalors:
      if not(whals.requires_broadcasting):
        return None
    try:
      return np.broadcast_shapes(*(whals.data.shape for whals in Whalors))
    except ValueError:
      return None
  
  def result_requires_grad(self, Whalors):
    
    for whals in Whalors:
      if whals.requires_grad:
        return True
    return False
  
  def get_result_whalor(self, result, *Whalors):
    
    from .whalor import Whalor
    from .toolbox import current_graph
    graph = current_graph()
    result = result.astype(np.ndarray)
    result_Whalor = Whalor(result, self.result_requires_grad(Whalors))
    if graph.track:
      result_node = Node(result_Whalor)
      result_node.backward_fn = self.backward
      result_node.parent_broadcast_shape = self.get_broadcast_shape(*Whalors)
      graph.create_edge(result_node, Whalors)
    return result_Whalor
  
  def backward(self, *args):
    
    raise NotImplementedError(f"Backward method not implemented for Action {self}")
  
'''-----------------------------------------------------------------------------------------------------------------------------'''

class Add(Action):
    
  
  def forward(self, whals1, whals2):
      
      whals1, whals2 = self.get_Whalors(whals1, whals2)
      return self.get_result_whalor(whals1.data+whals2.data, whals1, whals2)
  
  def backward(self, whals1, whals2):
     
      whals1.set_grad_fn(lambda ug:ug)
      whals2.set_grad_fn(lambda ug:ug)
  
def add(whals1, whals2):
    
    return Add().forward(whals1, whals2)
  
  
# <------------SUB------------>
class Sub(Action):
  
  def forward(self, whals1, whals2):
    
    whals1, whals2 = self.get_Whalors(whals1, whals2)
    return self.get_result_whalor(whals1.data-whals2.data, whals1, whals2)
  
  def backward(self, whals1, whals2):
   
    whals1.set_grad_fn(lambda ug:ug)
    whals2.set_grad_fn(lambda ug:-ug)

def sub(whals1, whals2):
  
  return Sub().forward(whals1, whals2)


# <------------MUL------------>
class Mul(Action):
  
  def forward(self, whals1, whals2):
    
    whals1, whals2 = self.get_Whalors(whals1, whals2)
    return self.get_result_whalor(whals1.data*whals2.data, whals1, whals2)
  
  def backward(self, whals1, whals2):
    
    whals1.set_grad_fn(lambda ug:whals2.data*ug)
    whals2.set_grad_fn(lambda ug:whals1.data*ug)

def mul(whals1, whals2):
  
  return Mul().forward(whals1, whals2)


# <------------DIV------------>
class Div(Action):
  
  def forward(self, whals1, whals2):
    
    whals1, whals2 = self.get_Whalors(whals1, whals2)
    return self.get_result_whalor(whals1.data/whals2.data, whals1, whals2)
  
  def backward(self, whals1, whals2):
    
    whals1.set_grad_fn(lambda ug:(1/whals2.data)*ug)
    whals2.set_grad_fn(lambda ug:((-1*whals1.data)/np.power(whals2.data, 2))*ug)

def div(whals1, whals2):
  
  return Div().forward(whals1, whals2)


# <------------DOT------------>
class Dot(Action):
  
  def forward(self, whals1, whals2):
    
    whals1, whals2 = self.get_Whalors(whals1, whals2)
    return self.get_result_whalor(np.dot(whals1.data, whals2.data), whals1, whals2)
  
  def backward(self, whals1, whals2):
    
    whals1.set_grad_fn(lambda ug:np.dot(ug, whals2.data.T))
    whals2.set_grad_fn(lambda ug:np.dot(whals1.data.T, ug))
    return whals1.grad, whals2.grad

def dot(whals1, whals2):
  
  return Dot().forward(whals1, whals2)


# <------------EXP------------>
class Exp(Action):
  
  def forward(self, whals):
    
    whals = self.get_Whalors(whals)
    return self.get_result_whalor(np.exp(whals.data), whals)
  
  def backward(self, whals):
    
    whals.set_grad_fn(lambda ug:np.exp(whals.data)*ug)

def exp(whals):
  
  return Exp().forward(whals)


# <------------LOG------------>
class Log(Action):
  
  def forward(self, whals):
    
    whals = self.get_Whalors(whals)
    return self.get_result_whalor(np.log(whals.data), whals)
  
  def backward(self, whals):
    
    whals.set_grad_fn(lambda ug:(1/whals.data)*ug)

def log(whals):
  
  return Log().forward(whals)


# <------------POW------------>
class Pow(Action):
  
  def forward(self, whals1, whals2):
    
    whals1, whals2 = self.get_Whalors(whals1, whals2)
    return self.get_result_whalor(np.power(whals1.data, whals2.data), whals1, whals2)
  
  def backward(self, whals1, whals2):
    
    result = np.power(whals1.data, whals2.data)
    whals1.set_grad_fn(lambda ug:(np.power(whals1.data, whals2.data-1) * whals2.data)*ug)
    whals2.set_grad_fn(lambda ug:(result*np.log(whals1.data))*ug)

def pow(whals1, whals2):
  
  return Pow().forward(whals1, whals2)


# <------------SUM------------>
class Sum(Action):
  
  def __init__(self, axis=None):
    
    self.axis = axis
  
  def forward(self, whals):
    
    whals = self.get_Whalors(whals)
    return self.get_result_whalor(np.sum(whals.data, axis=self.axis), whals)
  
  def backward(self, whals):
    
    def sum_backward(ug):
      if self.axis is not None:
        ug = np.expand_dims(ug, axis=self.axis)
      grads = np.ones(whals.shape)*ug
      return grads
    whals.set_grad_fn(sum_backward)

def sum(whals, axis=None):
  
  return Sum(axis).forward(whals)


# <------------TRANSPOSE------------>
class Transpose(Action):
  
  def forward(self, whals):
    
    whals = self.get_Whalors(whals)
    return self.get_result_whalor(whals.data.T, whals)

  def backward(self, whals):
    
    whals.set_grad_fn(lambda ug:ug.T)

def transpose(whals):
  
  return Transpose().forward(whals)


# <------------FLATTEN------------>
class Flatten(Action):
  
  def forward(self, whals):
    
    whals = self.get_Whalors(whals)
    flattened = whals.data.flatten()
    return self.get_result_whalor(flattened.reshape(flattened.shape[0],1), whals)
  
  def backward(self, whals):
    
    whals.set_grad_fn(lambda ug:ug.reshape(whals.shape))

def flatten(whals):
  
  return Flatten().forward(whals)


# <------------RESHAPE------------>
class Reshape(Action):
  
  def forward(self, whals, new_shape):
    
    whals = self.get_Whalors(whals)
    return self.get_result_whalor(whals.data.reshape(new_shape), whals)
  
  def backward(self, whals):
   
    whals.set_grad_fn(lambda ug:ug.reshape(whals.shape))

def reshape(whals, new_shape):
  
  return Reshape().forward(whals, new_shape)