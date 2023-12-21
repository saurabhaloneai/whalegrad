from .node import Node

class Graph:
    '''
    '''
    graph = None 

    def __init__(self):
        '''
        '''

        self.nodes_dict = {}
        self.track = True

    def add_edge(self,result_node, operands):
        '''
        '''

        self.add_node(result_node)
        for operand in operands:
            if self.get_node(operands) is None :
                self.add_tensor(operand)
            operand_node = self.get_nude(operand)
            result_node.add_parent(operand_node)
            operand_node.add_child(result_node)

    def add_node(self,node):
        '''
        '''
        self.nodes_dict[node.tensor] = node 

    def get_node(self,tensor):
        '''
        '''

        return self.nodes_dict.get(tensor)

    def add_tensor(self, tensor):
        '''
        ''' 
        self.nodes_dict.pop(tensor)  

    def remove_tensor(self, tensor):
        '''
        '''
        self.nodes_dict.pop(tensor)

    def reset_visited(self):
        '''
        '''
        for node in self.nodes_dict.values():
            node.visited =False

    def zero_grad(self):
        '''
        '''

        for tensor in self.nodes_dict.keys():
            tensor.zero_grad()


    def __repr__(self):
        return 'Graph()'

    def __str__(self):
        return f'Graph({self.nodes_dict})'        




            
        
                        

