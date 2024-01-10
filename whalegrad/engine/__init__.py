# from .tensor import Tensor as tensor
# from .functions import add, sub, mul, div, pow, exp, log, dot, sum, transpose, flatten, reshape
from utils import new_graph, no_track
from utils import current_graph
## global variable for graph
from base import Graph

# global GRAPH_GB
# GRAPH_GB = Graph()
# '''
#   _WH_GRAPH is the global graph object used to construct backprop graphs
# '''