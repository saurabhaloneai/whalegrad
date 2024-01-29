class Node:
  

  def __init__(self, whals):
    
    self.whals = whals
    self.children = []
    self.parents = []
    self.parent_broadcast_shape = None
    self.backward_fn = None
    self.visited = False
  
  def topological_sort(self):
    
    sorted_Whalors = []
    if self.are_children_visited():
      self.visited = True
      sorted_Whalors.append(self.whals)
      for parent in self.parents:
        if not(parent.visited):
          sorted_Whalors+=parent.topological_sort()
    else:
      for child in self.children:
        if not(child.visited):
          sorted_Whalors+=child.topological_sort()
    return sorted_Whalors
  
  def backward(self, retain_graph):
    
    from whalegrad.engine.toolbox import current_graph
    graph = current_graph()
    graph.reset_visited()
    self.visit_all_children()
    sorted_Whalors = self.topological_sort()
    graph.reset_visited()

    sorted_Whalors.pop(0)
    self.visited = True
    self.whals._backward(self, retain_graph, calculate_grads=False)

    for whals in sorted_Whalors:
      node = graph.get_node(whals)
      node.visited = True
      whals._backward(node, retain_graph)

  def visit_all_children(self):
    
    for child in self.children:
      child.visited = True

  def are_children_visited(self):
    
    for child in self.children:
      if not(child.visited):
        return False
    return True
  
  def are_parents_visited(self):
    
    for parent in self.parents:
      if not(parent.visited):
        return False
    return True
  
  def append_child(self, other):
    
    self.children.append(other)
  
  def append_parent(self, other):
    
    self.parents.append(other)
  
  def __repr__(self):
    return f'Node({self.whals})'
  
  def __str__(self):
    return f'Node( \n{self.whals}\nbackward_fn: {self.backward_fn}\nvisited: {self.visited}\n )'



## Another approach to topological sort 
# class Node:
 

#   def __init__(self, whals):
   
#     self.whals = whals
#     self.children = []
#     self.parents = []
#     self.parent_broadcast_shape = None
#     self.backward_fn = None
#     self.visited = False
  
#   def topological_sort(self):
     
#     sorted_Whalors = []
    
#     if not self.are_children_visited():
#         for child in self.children:
#             if not child.visited:
#                 sorted_Whalors += child.topological_sort()

#     self.visited = True
#     sorted_Whalors.append(self.whals)

#     for parent in self.parents:
#         if not parent.visited:
#             sorted_Whalors += parent.topological_sort()

#     return sorted_Whalors
         
#   def backward(self, preserve_graph):
#     from toolbox import current_graph
#     graph = current_graph()
    
#     print(f"Current Node: {self}")
    
#     graph.reset_visited()
#     self.visit_all_children()
#     sorted_Whalors = self.topological_sort()
#     graph.reset_visited()

#     sorted_Whalors.pop(0)  # Remove the Whalor corresponding to the current node
#     self.visited = True
    
#     print(f"Calling _backward on {self.whals}")
    
#     self.whals._backward(self, preserve_graph, calculate_grads=False)

#     for whals in sorted_Whalors:
#         node = graph.get_node(whals)
#         node.visited = True
        
#         print(f"Calling _backward on {whals}")
        
#         whals._backward(node, preserve_graph)
       
  
#   # def backward(self, preserve_graph):
  
    
#   #   from toolbox import current_graph
#   #   graph = current_graph()
#   #   graph.reset_visited()
#   #   self.visit_all_children() # this allows for gradient calculation from any intermediate node in the graph
#   #   sorted_Whalors = self.topological_sort()
#   #   graph.reset_visited()

#   #   sorted_Whalors.pop(0) # Remove the Whalor corresponding to the current node
#   #   self.visited = True
#   #   self.whals._backward(self, preserve_graph, calculate_grads=False)

#   #   for whals in sorted_Whalors:
#   #     node = graph.get_node(whals)
#   #     node.visited = True
#   #     whals._backward(node, preserve_graph)

#   def visit_all_children(self):
    
#     for child in self.children:
#       child.visited = True

#   def are_children_visited(self):
    
#     for child in self.children:
#       if not(child.visited):
#         return False
#     return True
  
#   def are_parents_visited(self):
   
#     for parent in self.parents:
#       if not(parent.visited):
#         return False
#     return True
  
#   def append_child(self, other):
    
#     self.children.append(other)
  
#   def append_parent(self, other):
   
#     self.parents.append(other)
  
#   def __repr__(self):
#     return f'Node({self.whals})'
  
#   def __str__(self):
#     return f'Node( \n{self.whals}\nbackward_fn: {self.backward_fn}\nvisited: {self.visited}\n )'
  

  