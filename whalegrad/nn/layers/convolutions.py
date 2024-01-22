import numpy as np
from typing import Union

class Conv1d:
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        scale = np.sqrt(1 / (in_channels * kernel_size))
        self.weight = np.random.uniform(
            low=-scale,
            high=scale,
            size=(out_channels, kernel_size, in_channels),
        )
        if bias:
            self.bias = np.zeros((out_channels,))

        self.padding = padding
        self.stride = stride

    def _extra_repr(self):
        return (
            f"{self.weight.shape[-1]}, {self.weight.shape[0]}, "
            f"kernel_size={self.weight.shape[1]}, stride={self.stride}, "
            f"padding={self.padding}, bias={'bias' in self.__dict__}"
        )

    def __call__(self, x):
        y = np.conv1d(x, self.weight, stride=self.stride, padding=self.padding)
        if "bias" in self.__dict__:
            y = y + self.bias
        return y

class Conv2d:
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple],
        stride: Union[int, tuple] = 1,
        padding: Union[int, tuple] = 0,
        bias: bool = True,
    ):
        kernel_size, stride, padding = map(
            lambda x: (x, x) if isinstance(x, int) else x,
            (kernel_size, stride, padding),
        )
        scale = np.sqrt(1 / (in_channels * kernel_size[0] * kernel_size[1]))
        self.weight = np.random.uniform(
            low=-scale,
            high=scale,
            size=(out_channels, *kernel_size, in_channels),
        )
        if bias:
            self.bias = np.zeros((out_channels,))

        self.padding = padding
        self.stride = stride

    def _extra_repr(self):
        return (
            f"{self.weight.shape[-1]}, {self.weight.shape[0]}, "
            f"kernel_size={self.weight.shape[1:3]}, stride={self.stride}, "
            f"padding={self.padding}, bias={'bias' in self.__dict__}"
        )

    def __call__(self, x):
        y = np.conv2d(x, self.weight, stride=self.stride, padding=self.padding)
        if "bias" in self.__dict__:
            y = y + self.bias
        return y
