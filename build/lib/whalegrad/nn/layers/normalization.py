import numpy as np  
from .base import Core

class LayerNorm(Core):
    def __init__(self, dims):
        self.gamma = np.ones((1, dims))
        self.beta = np.zeros((1, dims))

    def __call__(self, x):
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        return self.gamma * (x - mean) / (std + 1e-8) + self.beta