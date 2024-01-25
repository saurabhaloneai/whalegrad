import numpy as np
from .base import Core
from .base import Param 

class Embedding(Core):
    """Implements a simple lookup table that maps each input integer to a
    high-dimensional vector.

    Typically used to embed discrete tokens for processing by neural networks.

    Args:
        num_embeddings (int): How many possible discrete tokens can we embed.
                              Usually called the vocabulary size.
        dims (int): The dimensionality of the embeddings.
    """

    def __init__(self, num_embeddings: int, dims: int):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.dims = dims
        scale = np.sqrt(1 / dims)
        self.weight = Param(np.random.normal(0, scale, (num_embeddings, dims)))

    def _extra_repr(self):
        return f"{self.num_embeddings}, {self.dims}"

    def __call__(self, x):
        return self.weight[x]
