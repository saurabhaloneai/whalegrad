class Node:
    '''
   for computation graph

    '''
    def __init__(self,ten):
        '''

        '''

        self.ten = ten
        self.children = []
        self.paresnts = []
        self.parenst_brodcat_shape = None
        self.backward_fn = None
        self.visited = False 

    def topl_sort(self):
        '''

        '''

        sorted_tensors = []

        if self.are_children_visited():
            self.visited =True
            sorted_tensors.append(self.ten)
            for parent in self.parents:
                if not(parent.visited):
                    sorted_tensors += parent.topl_sort()

        else :
            for child in self.children:
                if not(child.visited):
                    sorted_tensors += child.topl_sort()
            
        return sorted_tensors
        

    def backward(self,retain_graph):
        '''
        '''

        from .utils import get_graph
        graph = get_graph()
        graph.reset_visited()
        self.visit_all_children()
        sorted_tensors = self.top_sort()
        graph.reset_visited()

        sorted_tensors.pop(0)

        self.tens._backward(self,retain_graph, calculate_grads=False)

        for ten in sorted_tensors:
            node = graph.get_node(ten)
            node.visited = True 
            ten._backward(node, retain_graph)

    def visit_all_children(self):
        '''

        '''

        for child in self.children :
            child.visited = True

    def are_children_visited(self):
        '''


        '''

        for child in self.children :  
            if not(child.visited):
                return False
            return True

    def are_parents_visited(self):
        
        '''

        '''

        for parent in self.parents:
            if not(parent.visited):
                return False 
        return True

    def add_child(self, other):
        
        '''


        '''
        self.children.append(other)

    def add_parent(self,other):

        '''

        '''

        self.parents.append(other)

    def __repr__(self):
        
        return f"Node(\n{self.ten}\nbackward_fn: {self.backward_fn}\nvisited : {self.visited}\n)"



        


                

        
        


    
