import numpy as np
from node import Node


class Operation:
 
  def process_operands(self, operands):
    
    from .value import Tensor
    operands = list(operands)
    for i,operand in enumerate(operands):
      if not isinstance(operand, Tensor):
        operands[i] = Tensor(operand)
    return tuple(operands)
  
#   for i,operand in enumerate(operands):
#  operands = list(operands) 
  def get_tensors(self, *operands):
    
    tensors = self.process_operands(operands)
    if len(tensors)==0:
      return None
    elif len(tensors)==1:
      return tensors[0]
    else:
      return tensors
  
  def get_broadcast_shape(self, *tensors):
   
    for tens in tensors:
      if not(tens.requires_broadcasting):
        return None
    try:
      return np.broadcast_shapes(*(tens.data.shape for tens in tensors))
    except ValueError:
      return None
  
  def result_requires_grad(self, tensors):
  
    for tens in tensors:
      if tens.requires_grad:
        return True
    return False
  
  def get_result_tensor(self, result, *tensors):
    
    from .value import Tensor
    from .utils import get_graph
    graph = get_graph()
    result = result.astype(np.ndarray)
    result_tensor = Tensor(result, self.result_requires_grad(tensors))
    if graph.track:
      result_node = Node(result_tensor)
      result_node.backward_fn = self.backward
      result_node.parent_broadcast_shape = self.get_broadcast_shape(*tensors)
      graph.add_edge(result_node, tensors)
    return result_tensor
  
  def backward(self, *args):
  
    raise NotImplementedError(f"Backward method not implemented for Operation {self}")

# <------------ADD------------>
class Add(Operation):
  '''Element wise addition between two Tensors or Tensor-like
  '''
  def forward(self, tens1, tens2):
  
    tens1, tens2 = self.get_tensors(tens1, tens2)
    return self.get_result_tensor(tens1.data+tens2.data, tens1, tens2)

  def backward(self, tens1, tens2):
   
    tens1.set_grad_fn(lambda ug:ug)
    tens2.set_grad_fn(lambda ug:ug)

def add(tens1, tens2):
 
  return Add().forward(tens1, tens2)


# <------------SUB------------>
class Sub(Operation):
  
  def forward(self, tens1, tens2):
   
    tens1, tens2 = self.get_tensors(tens1, tens2)
    return self.get_result_tensor(tens1.data-tens2.data, tens1, tens2)
  
  def backward(self, tens1, tens2):
   
    tens1.set_grad_fn(lambda ug:ug)
    tens2.set_grad_fn(lambda ug:-ug)

def sub(tens1, tens2):
  
  return Sub().forward(tens1, tens2)


# <------------MUL------------>
class Mul(Operation):
 
  def forward(self, tens1, tens2):
   
    tens1, tens2 = self.get_tensors(tens1, tens2)
    return self.get_result_tensor(tens1.data*tens2.data, tens1, tens2)
  
  def backward(self, tens1, tens2):
   
    tens1.set_grad_fn(lambda ug:tens2.data*ug)
    tens2.set_grad_fn(lambda ug:tens1.data*ug)

def mul(tens1, tens2):
  
  return Mul().forward(tens1, tens2)


# <------------DIV------------>
class Div(Operation):
 
  def forward(self, tens1, tens2):
    
    tens1, tens2 = self.get_tensors(tens1, tens2)
    return self.get_result_tensor(tens1.data/tens2.data, tens1, tens2)
  
  def backward(self, tens1, tens2):
    
    tens1.set_grad_fn(lambda ug:(1/tens2.data)*ug)
    tens2.set_grad_fn(lambda ug:((-1*tens1.data)/np.power(tens2.data, 2))*ug)

def div(tens1, tens2):
  
  return Div().forward(tens1, tens2)


# <------------DOT------------>
class Dot(Operation):
  
  def forward(self, tens1, tens2):
    
    tens1, tens2 = self.get_tensors(tens1, tens2)
    return self.get_result_tensor(np.dot(tens1.data, tens2.data), tens1, tens2)
  
  def backward(self, tens1, tens2):
    
    tens1.set_grad_fn(lambda ug:np.dot(ug, tens2.data.T))
    tens2.set_grad_fn(lambda ug:np.dot(tens1.data.T, ug))

def dot(tens1, tens2):
  
  return Dot().forward(tens1, tens2)


# <------------EXP------------>
class Exp(Operation):
  
  def forward(self, tens):
    
    tens = self.get_tensors(tens)
    return self.get_result_tensor(np.exp(tens.data), tens)
  
  def backward(self, tens):
   
    tens.set_grad_fn(lambda ug:np.exp(tens.data)*ug)

def exp(tens):
  
  return Exp().forward(tens)


# <------------LOG------------>
class Log(Operation):
  
  def forward(self, tens):
    
    tens = self.get_tensors(tens)
    return self.get_result_tensor(np.log(tens.data), tens)
  
  def backward(self, tens):
   
    tens.set_grad_fn(lambda ug:(1/tens.data)*ug)

def log(tens):
  
  return Log().forward(tens)


# <------------POW------------>
class Pow(Operation):
  
  def forward(self, tens1, tens2):
    
    tens1, tens2 = self.get_tensors(tens1, tens2)
    return self.get_result_tensor(np.power(tens1.data, tens2.data), tens1, tens2)
  
  def backward(self, tens1, tens2):
    
    result = np.power(tens1.data, tens2.data)
    tens1.set_grad_fn(lambda ug:(np.power(tens1.data, tens2.data-1) * tens2.data)*ug)
    tens2.set_grad_fn(lambda ug:(result*np.log(tens1.data))*ug)

def pow(tens1, tens2):
 
  return Pow().forward(tens1, tens2)


# <------------SUM------------>
class Sum(Operation):
  
  def __init__(self, axis=None):
   
    self.axis = axis
  
  def forward(self, tens):
    
    tens = self.get_tensors(tens)
    return self.get_result_tensor(np.sum(tens.data, axis=self.axis), tens)
  
  def backward(self, tens):
    
    def sum_backward(ug):
      if self.axis is not None:
        ug = np.expand_dims(ug, axis=self.axis)
      grads = np.ones(tens.shape)*ug
      return grads
    tens.set_grad_fn(sum_backward)

def sum(tens, axis=None):
 
  return Sum(axis).forward(tens)


# <------------TRANSPOSE------------>
class Transpose(Operation):
 
  def forward(self, tens):
    
    tens = self.get_tensors(tens)
    return self.get_result_tensor(tens.data.T, tens)

  def backward(self, tens):
    
    tens.set_grad_fn(lambda ug:ug.T)

def transpose(tens):
  
  return Transpose().forward(tens)


# <------------FLATTEN------------>
class Flatten(Operation):
  
  def forward(self, tens):
    
    tens = self.get_tensors(tens)
    flattened = tens.data.flatten()
    return self.get_result_tensor(flattened.reshape(flattened.shape[0],1), tens)
  
  def backward(self, tens):
    
    tens.set_grad_fn(lambda ug:ug.reshape(tens.shape))

def flatten(tens):
  
  return Flatten().forward(tens)


# <------------RESHAPE------------>
class Reshape(Operation):
  
  def forward(self, tens, new_shape):
    
    tens = self.get_tensors(tens)
    return self.get_result_tensor(tens.data.reshape(new_shape), tens)
  
  def backward(self, tens):
    
    tens.set_grad_fn(lambda ug:ug.reshape(tens.shape))

def reshape(tens, new_shape):
 
  return Reshape().forward(tens, new_shape)
