class Node:
    """
    Represents a node in a computation graph.

    Used as an abstraction to connect tensors together and hold relationships.
    Each tensor is assigned a Node, and this Node monitors all incoming edges (parents)
    and outgoing edges (children).

    Parameters:
    - children (list of Node): List of all Nodes which use the current Node
      as an operand in an Operation.
    - parents (list of Node): List of all Nodes (operands) that have resulted in the creation
      of the current Node.
    - parent_broadcast_shape (tuple or None): If the parent needs to be broadcasted from one shape to
      another, then the final broadcasted shape of the parent is stored here.
      If they cannot be broadcasted, then it is None.
    - backward_fn (Operation.backward): Sets the grad_fn of Tensor(operand) involved in the Operation.
    - visited (bool): If Node is visited or not.
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
        Performs topological sort of all Nodes starting from the current Node.

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
        Initiates backward pass starting from the current Node.

        This first visits all the children to ensure that they aren't included in
        sorted_tensors as they aren't required, as the backward pass is initiated from the current node.

        Then it pops its corresponding Tensor from sorted_tensors (it is the first tensor) so that
        _backward can be called on it with calculate_grads=False, so that grads aren't calculated for
        it, but allows flushing of all Tensors.

        Next, it topologically sorts all Tensors starting from the current Node, and then the Node
        corresponding to the Tensor is retrieved, which is marked as visited, and the Tensor's
        backward pass is initiated.

        Parameters:
        - retain_graph (bool): If the graph should be retained after backward pass or flushed
          after backward calculation.
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

