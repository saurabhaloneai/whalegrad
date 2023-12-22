from .node import Node

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
