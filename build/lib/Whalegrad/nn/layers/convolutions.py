import numpy as np
from typing import Union
from .base import Core  
from .base import Param 

import math
import numpy as np
from whalegrad.engine.functions import Action
from .essential import conv2d, conv3d, maxpool2d, maxpool3d


import numpy as np



class Conv2D(Core):
  
  def __init__(self, kernel_shape, padding=0, stride=1):
    
    self.padding = padding
    self.stride = stride
    if len(kernel_shape)!=2:
      raise ValueError("Kernel shape can only have 2 dims")
    self.weights = Param(np.random.randn(*kernel_shape), requires_grad=True, requires_broadcasting=False)
    self.bias = Param(0, requires_grad=True, requires_broadcasting=False)
  
  def forward(self, inputs):
    
    return conv2d(inputs, self.weights, self.bias, self.padding, self.stride)
  
  def __repr__(self):
    return f'Conv2D(kernel_shape={self.weights.shape}, padding={self.padding}, stride={self.stride})'
  
  def __str__(self):
    return f'Conv2D(kernel_shape={self.weights.shape}, padding={self.padding}, stride={self.stride})'


class Conv3D(Core):
  
  def __init__(self, in_channels, out_channels, kernel_shape, padding=0, stride=1):
    
    self.padding = padding
    self.stride = stride
    if len(kernel_shape)!=2:
      raise ValueError("Kernel shape can only have 2 dims")
    self.weights = Param(np.random.randn(out_channels, in_channels, *kernel_shape), requires_grad=True, requires_broadcasting=False)
    self.bias = Param(np.zeros(out_channels), requires_grad=True, requires_broadcasting=False)
  
  def forward(self, inputs):
    
    return conv3d(inputs, self.weights, self.bias, self.padding, self.stride)
  
  def __repr__(self):
    kernel_shape = self.weights.shape
    return f'Conv3D(out_channels={kernel_shape[0]}, in_channels={kernel_shape[1]}, kernel_shape={kernel_shape[2:]}, padding={self.padding}, stride={self.stride})'
  
  def __str__(self):
    kernel_shape = self.weights.shape
    return f'Conv3D(out_channels={kernel_shape[0]}, in_channels={kernel_shape[1]}, kernel_shape={kernel_shape[2:]}, padding={self.padding}, stride={self.stride})'


class MaxPool2D(Core):
  
  def __init__(self, kernel_shape, padding=0, stride=1):
    self.padding = padding
    self.stride = stride
    if len(kernel_shape)!=2:
      raise ValueError("Kernel shape can only have 2 dims")
    self.kernel_shape = kernel_shape
  
  def forward(self, inputs):
    
    return maxpool2d(inputs, self.kernel_shape, self.padding, self.stride)
  
  def __repr__(self):
    return f'MaxPool2D(kernel_shape={self.kernel_shape}, padding={self.padding}, stride={self.stride})'
  
  def __str__(self):
    return f'MaxPool2D(kernel_shape={self.kernel_shape}, padding={self.padding}, stride={self.stride})'


class MaxPool3D(Core):
  
  def __init__(self, kernel_shape, padding=0, stride=1):
    self.padding = padding
    self.stride = stride
    if len(kernel_shape)!=2:
      raise ValueError("Kernel shape can only have 2 dims")
    self.kernel_shape = kernel_shape
  
  def forward(self, inputs):
    
    return maxpool3d(inputs, self.kernel_shape, self.padding, self.stride)
  
  def __repr__(self):
    return f'MaxPool3D(kernel_shape={self.kernel_shape}, padding={self.padding}, stride={self.stride})'
  
  def __str__(self):
    return f'MaxPool3D(kernel_shape={self.kernel_shape}, padding={self.padding}, stride={self.stride})'