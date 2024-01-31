from .node import Node

class Graph:
  

  graph = None
  
  def __init__(self):
   
    self.nodes_dict = {}
    self.track = True
  
  def create_edge(self, result_node, inputs):
    
    self.add_node(result_node)
    for operand in inputs:
      if self.get_node(operand) is None:
        self.add_Whalor(operand)
      operand_node = self.get_node(operand)
      result_node.append_parent(operand_node)
      operand_node.append_child(result_node)
  
  def add_node(self, node):
    
    self.nodes_dict[node.whals] = node

  def get_node(self, whals):
    
    return self.nodes_dict.get(whals)
  
  def add_Whalor(self, whals):
    
    self.nodes_dict[whals] = Node(whals)
  
  def remove_Whalor(self, whals):
    
    self.nodes_dict.pop(whals)
  
  def reset_visited(self):
    
    for node in self.nodes_dict.values():
      node.visited = False
  
  def reset_graph(self):
    
    self.nodes_dict = {}
  
  def zero_grad(self):
    
    for Whalor in self.nodes_dict.keys():
      Whalor.zero_grad()
  
  def __repr__(self):
    return 'Graph()'
  
  def __str__(self):
    return f'Graph( {self.nodes_dict} )'  