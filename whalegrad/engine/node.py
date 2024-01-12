class Node:
 

  def __init__(self, tens):
   
    self.tens = tens
    self.children = []
    self.parents = []
    self.parent_broadcast_shape = None
    self.backward_fn = None
    self.visited = False
  
  def topological_sort(self):
     
    sorted_tensors = []
    
    if not self.are_children_visited():
        for child in self.children:
            if not child.visited:
                sorted_tensors += child.topological_sort()

    self.visited = True
    sorted_tensors.append(self.tens)

    for parent in self.parents:
        if not parent.visited:
            sorted_tensors += parent.topological_sort()

    return sorted_tensors
         
        
  
  def backward(self, preserve_graph):
    
    from toolbox import current_graph
    graph = current_graph()
    graph.reset_visited()
    self.visit_all_children() # this allows for gradient calculation from any intermediate node in the graph
    sorted_tensors = self.topological_sort()
    graph.reset_visited()

    sorted_tensors.pop(0) # Remove the Tensor corresponding to the current node
    self.visited = True
    self.tens._backward(self, preserve_graph, calculate_grads=False)

    for tens in sorted_tensors:
      node = graph.get_node(tens)
      node.visited = True
      tens._backward(node, preserve_graph)

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
    return f'Node({self.tens})'
  
  def __str__(self):
    return f'Node( \n{self.tens}\nbackward_fn: {self.backward_fn}\nvisited: {self.visited}\n )'
  

  