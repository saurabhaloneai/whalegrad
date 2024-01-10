
from graph import Graph


def current_graph():
    '''Returns the graph that is in use

    If Graph.graph is None, the global graph GRAPH_GB is used

    Returns:
      Graph object that is currently used
    '''
    from .. import GRAPH_GB
    return Graph.graph if Graph.graph is not None else GRAPH_GB