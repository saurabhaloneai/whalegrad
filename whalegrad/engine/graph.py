from node import Node


class Graph:
  

  graph = None
  
  def __init__(self):
   
    self.nodes_dict = {}
    self.track = True
  
  def create_edge(self, result_node, operands):
    
    self.add_node(result_node)
    for operand in operands:
      if self.get_node(operand) is None:
        self.add_tensor(operand)
      operand_node = self.get_node(operand)
      result_node.append_parent(operand_node)
      operand_node.append_child(result_node)
  
  def add_node(self, node):
    
    self.nodes_dict[node.tens] = node

  def get_node(self, tens):
    
    return self.nodes_dict.get(tens)
  
  def add_tensor(self, tens):
    
    self.nodes_dict[tens] = Node(tens)
  
  def remove_tensor(self, tens):
    
    self.nodes_dict.pop(tens)
  
  def reset_visited(self):
    
    for node in self.nodes_dict.values():
      node.visited = False
  
  def reset_graph(self):
    
    self.nodes_dict = {}
  
  def zero_grad(self):
    
    for tensor in self.nodes_dict.keys():
      tensor.zero_grad()
  
  def __repr__(self):
    return 'Graph()'
  
  def __str__(self):
    return f'Graph( {self.nodes_dict} )'  