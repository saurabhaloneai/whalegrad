from node import Node

class Graph:
    
    graph = None # if not provided grpah it will call the global graph

    def __init__(self):
       
       self.nodes_dict = {}
       self.trace = True

    def create_edge(self, final_node, input_nodes):
       
       self.create_node(final_node)
       for i in input_nodes: #i stands for input_node
           if self.get_node(i) is None:
              self.create_value(i)   
           input_node = self.get_node(i)
           final_node.create_parent(input_node)
           input_node.create_child(final_node)
          

          
    def create_node(self, node):
       
       self.nodes_dict[node.vals] = node


    def get_node(self, vals):
       
       return self.nodes_dict.get(vals)
    
    def create_value(self, vals):
       
       self.nodes_dict[vals] = Node(vals)

    def remove_value(self, vals):
       
       self.nodes_dict.pop(vals)

    def default_vis(self):
       for node in self.nodes_dict.values():
          node.visited = False
       

    def default_graph(self):
       
       self.nodes_dict = {}
    
    def zero_grad(self):
       
       for value in self.nodes_dict.Keys():
          value.zero_grad()
    
    def __repr__(self):
       return 'Graph()'
    
    def __str__(self):
       return f'Graph({self.nodes_dict})'
    