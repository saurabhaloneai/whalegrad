class Node:
    """
A computational graph node for tensor operations, maintaining parent-child relationships. 
Parameters include lists of children and parents, parent broadcast shape, backward function (grad_fn), 
and a visitation status flag.
"""

    def __init__(self, tensor):
        """
        Initializes a Node with a given tensor.

        Parameters:
        - tensor (Tensor): The tensor corresponding to the Node.
        """
        self.tensor = tensor
        self.children = []
        self.parents = []
        self.parent_broadcast_shape = None
        self.backward_fn = None
        self.visited = False

    def topological_sort(self):
        """
        Performs topological sort of all Nodes starting from the current Node. badically for recording gradient of all nodes.

        Sorts the graph topologically to perform the backward pass efficiently,
        ensuring that all the children are calculated before the current node's gradient.

        Returns:
        - List of tensors in topologically sorted order.
        """
        sorted_tensors = []

        if self.are_children_visited():
            self.visited = True
            sorted_tensors.append(self.tensor)
            for parent in self.parents:
                if not parent.visited:
                    sorted_tensors += parent.topological_sort()
        else:
            for child in self.children:
                if not child.visited:
                    sorted_tensors += child.topological_sort()

        return sorted_tensors

    def backward(self, retain_graph):
        """
        Initiates backward pass from the current Node, excluding children.This helps in calculating the grdients  
        Pops the corresponding Tensor from sorted_tensors for _backward call with calculate_grads=False. 
        Topologically sorts Tensors from the current Node, marks Nodes as visited, and initiates Tensor backward pass. 
        Includes retain_graph parameter for graph retention decision.
        """
        from .utils import get_graph
        graph = get_graph()
        graph.reset_visited()
        self.visit_all_children()
        sorted_tensors = self.topological_sort()
        graph.reset_visited()

        sorted_tensors.pop(0)  # Remove the Tensor corresponding to the current node
        self.visited = True
        self.tensor._backward(self, retain_graph, calculate_grads=False)

        for tensor in sorted_tensors:
            node = graph.get_node(tensor)
            node.visited = True
            tensor._backward(node, retain_graph)

    def visit_all_children(self):
        """
        Marks all children as visited.
        """
        for child in self.children:
            child.visited = True

    def are_children_visited(self):
        """
        Checks if all children are visited.

        Returns:
        - True if all children are visited, else False.
        """
        for child in self.children:
            if not child.visited:
                return False
        return True

    def are_parents_visited(self):
        """
        Checks if all parents are visited.

        Returns:
        - True if all parents are visited, else False.
        """
        for parent in self.parents:
            if not parent.visited:
                return False
        return True

    def add_child(self, other):
        """
        Adds a child to the Node.

        Parameters:
        - other (Node): The child Node.
        """
        self.children.append(other)

    def add_parent(self, other):
        """
        Adds a parent to the Node.

        Parameters:
        - other (Node): The parent Node.
        """
        self.parents.append(other)

    def __repr__(self):
        return f'Node({self.tensor})'

    def __str__(self):
        return f'Node( \n{self.tensor}\nbackward_fn: {self.backward_fn}\nvisited: {self.visited}\n )'



class Graph:
    """
    Represents a computation graph for automatic differentiation.

    The graph is constructed during the forward pass and utilized by the backward
    pass to compute gradients. This class keeps track of nodes and tensors.

    Attributes:
        graph (Graph or None): Graph object currently in use. If None, the global
            _NG_GRAPH is used; otherwise, a specific graph object is utilized.
        nodes_dict (dict): Stores key-value pairs of tensors and their corresponding
            nodes in the graph.
        track (bool): Determines whether the graph tracks tensor operations. If True,
            during any operation resulting in a new tensor, the operands are added as
            parents to the result tensor, and the result tensor is added as a child to
            the operands. If False, these tracking operations are skipped. Defaults to True.
    """

    graph = None

    def __init__(self):
        """
        Initializes the Graph instance with an empty nodes_dict and track set to True.
        """
        self.nodes_dict = {}
        self.track = True

    def add_edge(self, result_node, operands):
        """
        Creates edges between nodes in the graph.

        Adds edges between the result_node (created during an operation) and its operands.
        The result_node is added as a child of each operand, and the result_node adds all
        operands as its parents.

        Args:
            result_node (Node): Node created in Operation.get_result_tensor.
            operands (list of Tensor): All the operands for an Operation.
        """
        self.add_node(result_node)
        for operand in operands:
            if self.get_node(operand) is None:
                self.add_tensor(operand)
            operand_node = self.get_node(operand)
            result_node.add_parent(operand_node)
            operand_node.add_child(result_node)

    def add_node(self, node):
        """
        Adds a Node to the graph.

        Creates a key-value pair in nodes_dict with the specified node as the value
        and its tensor attribute as the key.

        Args:
            node (Node): Node to be added to the graph.
        """
        self.nodes_dict[node.tensor] = node

    def get_node(self, tensor):
        """
        Retrieves a Node from the graph based on the given tensor.

        Args:
            tensor (Tensor): Tensor whose node is to be fetched.

        Returns:
            Node if found, else None.
        """
        return self.nodes_dict.get(tensor)

    def add_tensor(self, tensor):
        """
        Adds a Tensor to the graph.

        A new node is created for the tensor, and the corresponding entry is made
        in nodes_dict.

        Args:
            tensor (Tensor): Tensor to be added.
        """
        self.nodes_dict[tensor] = Node(tensor)

    def remove_tensor(self, tensor):
        """
        Removes a Tensor from the graph.

        Pops the tensor from nodes_dict.

        Args:
            tensor (Tensor): Tensor to be removed.
        """
        self.nodes_dict.pop(tensor)

    def reset_visited(self):
        """
        Sets the visited attribute to False for each Node in the graph.
        """
        for node in self.nodes_dict.values():
            node.visited = False

    def reset_graph(self):
        """
        Resets the entire graph by emptying nodes_dict.
        """
        self.nodes_dict = {}

    def zero_grad(self):
        """
        Performs zero_grad on all tensors in the graph.

        Iterates through nodes_dict and calls zero_grad on the tensors.
        """
        for tensor in self.nodes_dict.keys():
            tensor.zero_grad()

    def __repr__(self):
        return 'Graph()'

    def __str__(self):
        return f'Graph({self.nodes_dict})'
