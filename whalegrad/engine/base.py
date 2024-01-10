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
    
  if not self.are_children_visited():
      for child in self.children:
          if not child.visited:
              sorted_vals += child.topological_sort()

  self.visited = True
  sorted_value.append(self.vals)

  for parent in self.parents:
      if not parent.visited:
          sorted_value += parent.topological_sort()

  return sorted_value
  