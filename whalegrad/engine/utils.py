

from .base import Graph
from .base import GRAPH_GB

def current_graph():
    """
    Get the global graph instance.

    Returns:
    - Graph
        Global graph instance.
    """
    return Graph.graph or GRAPH_GB
