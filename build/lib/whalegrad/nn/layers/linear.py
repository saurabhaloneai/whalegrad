import numpy as np
from whalegrad.engine.functions import dot
from .base import Core
from .base import Param

class Linear(Core):
    def __init__(self, num_in, num_out, bias=True):
        self.num_in = num_in
        self.num_out = num_out
        self.weights = Param(np.random.randn(num_in, num_out), requires_grad=True)
        self.bias = Param(np.zeros((1, num_out)), requires_grad=True) if bias else None

    def forward(self, inputs):
        output = dot(inputs, self.weights)
        if self.bias is not None:
            output = output + self.bias
        return output

    def __repr__(self):
        return f'Linear({self.num_in}, {self.num_out}, bias={self.bias is not None})'

    def __str__(self):
        return f'Linear in:{self.num_in} out:{self.num_out}{" with bias" if self.bias is not None else ""}'
