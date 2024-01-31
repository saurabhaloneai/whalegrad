from .toolbox import check_data, unbroadcast_data
from .functions import add, sub, mul, div, pow as _pow, transpose, sum as _sum, exp, dot, flatten, reshape

#Whalor == Whalor , whals == whals , whalors ==  Whalors. 

class Whalor:
  def __init__(self, data, requires_grad=False, requires_broadcasting=True):
    
    self.data = data
    self.requires_grad = requires_grad
    self.requires_broadcasting = requires_broadcasting
    self.grad = 0. if requires_grad else None
    self.grad_fn = None
  
  def zero_grad(self):
    
    self.grad = 0. if self.requires_grad else None
  
  def backward(self, upper_grad=1., preserve_graph=False):
    
    if not(self.requires_grad):
      raise ValueError("Only Whalors who requires_grad can call backward")
    from .toolbox import current_graph
    graph = current_graph()
    upper_grad = check_data(upper_grad)
    if self.shape!=upper_grad.shape:
      raise ValueError("Shapes of grad and Whalor data must match!")
    self.accumulate_grad(upper_grad) # Setting the grad of the current Whalor by adding the upper_grad
    node = graph.get_node(self)
    node.backward(preserve_graph)
    if not(preserve_graph):
      graph.reset_graph() # Whalors are auto-removed, this is just for redundancy / safety
  
  def _backward(self, node, preserve_graph, calculate_grads=True):
    
    from .toolbox import current_graph
    graph = current_graph()
    for child in node.children:
      if self.requires_grad and calculate_grads:
        child.backward_fn(*[node.whals for node in child.parents])
        upper_grad = child.whals.grad
        grad = self.grad_fn(upper_grad)
        grad = unbroadcast_data(grad, self.shape, child.parent_broadcast_shape)
        grad = grad.reshape(self.shape)
        self.accumulate_grad(grad)
      if not(preserve_graph) and child.are_parents_visited():
        graph.remove_Whalor(child.whals)
    if not(preserve_graph) and node.are_parents_visited():
      graph.remove_Whalor(node.whals)
  
  def set_grad_fn(self, grad_fn):
    
    self.grad_fn = grad_fn if self.requires_grad else None

  def __add__(self, other):
    
    return add(self, other)
  
  def __radd__(self, other):
    
    return add(other, self)
  
  def __sub__(self, other):
    
    return sub(self, other)
  
  def __rsub__(self, other):
    
    return sub(other, self)
  
  def __mul__(self, other):
    
    return mul(self, other)
  
  def __rmul__(self, other):
    
    return mul(other, self)
  
  def __truediv__(self, other):
    
    return div(self, other)
  
  def __rtruediv__(self, other):
    
    return div(other, self)
  
  def __pow__(self, other):
    
    return _pow(self, other)
  
  def __rpow__(self, other):
    
    return _pow(other, self)
  
  def __pos__(self):
    
    return (1*self)
  
  def __neg__(self):
    
    return (-1*self)
  
  def dot(self, other):
    
    return dot(self, other)
  
  def sum(self, axis=None):
    
    return _sum(self, axis)
  
  def exp(self):
    
    return exp(self)
  
  def flatten(self):
    
    return flatten(self)
  
  def reshape(self, new_shape):
    
    return reshape(self, new_shape)
  
  def accumulate_grad(self, grad):
    
    self.grad+=grad
  
  @property
  def data(self):
    
    return self._data
  
  @data.setter
  def data(self, data):
    
    self._data = check_data(data)

  @property
  def shape(self):
    
    return self.data.shape
  
  @property
  def T(self):
    
    return transpose(self)
  
  def __getitem__(self, *indices):
    
    supported_types = (int, slice)
    for index in indices:
      if type(index) not in supported_types:
        raise TypeError(f"Expected index of {supported_types} instead got {type(index)}")
    return Whalor(self.data[indices], requires_grad=self.requires_grad)
  
  def __repr__(self):
    return f'Whalor({self.data}, requires_grad={self.requires_grad})'
  
  def __str__(self):
    return f'Whalor( {self.data},\n requires_grad={self.requires_grad},\n grad_fn={self.grad_fn},\n shape={self.shape} )\n'