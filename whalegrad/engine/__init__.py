from .tensor import Tensor as tensor
from .functions import add, sub, mul, div, pow, exp, log, dot, sum, transpose, flatten, reshape
from .utils import new_graph, no_track

## global variable for graph
from base import Graph

global _WH_GRAPH
_WH_GRAPH = Graph()
'''
  _WH_GRAPH is the global graph object used to construct backprop graphs
'''