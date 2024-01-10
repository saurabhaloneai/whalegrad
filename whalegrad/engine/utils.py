

from .base import Graph
from . import GRAPH_GB

# def current_graph():
#     """
#     Get the global graph instance.

#     Returns:
#     - Graph
#         Global graph instance.
#     """
#     return Graph.graph or GRAPH_GB


def current_graph():
  '''Returns graph that is in use and present in Graph.graph

  If Graph.graph is None, then the global graph _NG_GRAPH is used

  Returns:
    Graph object that is currently used
  '''
  if Graph.graph is None:
    from .. import GRAPH_GB
    graph = GRAPH_GB
  else:
    graph = Graph.graph
  return graph