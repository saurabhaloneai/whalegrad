#import
from utils import current_graph 

#base code 

class Node:
    
 
 def __init__(self, vals):
  
  self.vals = vals 
  self.childrens= []
  self.parents = []
  self.bordacast_parent_shape = None
  self.backward_function = None
  self.visited = False
 
 def topological_sort(self):
  
  sorted_value = []
    
  if not self.check_children_visited():
      for child in self.children:
          if not child.visited:
              sorted_vals += child.topological_sort()

  self.visited = True
  sorted_value.append(self.vals)

  for parent in self.parents:
      if not parent.visited:
          sorted_value += parent.topological_sort()

  return sorted_value
  
  def backward(self,preserve_graph):
     
     #imported the current graph from graph
     graph = current_graph()
     graph.default_vis()
     self.mark_visit_all_children() # by usig this you can calculate gradient of any intermediate Node
     sorted_value = self.topological_sort()
     graph.default_vis()

     sorted_value.pop(0) # remove the value accordin to the current node
     self.visited = True
     self.vals._backward(self, preserve_graph, calculate_grads=False)

     for vals in sorted_value:
        node = graph.current_node(vals)
        node.visited = True
        vals._backward(node, preserve_graph)

  def mark_visited_all_children(self):
     
     for child in self.children:
        child.visited = True

  def check_children_visited(self):
     
     return all(child.visited for child in self.children)
 
  def 