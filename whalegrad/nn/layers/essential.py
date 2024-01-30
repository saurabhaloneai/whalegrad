import dill
from .model import Model
import numpy as np
from typing import Union
from .base import Core  
from .base import Param 

import math
import numpy as np
from whalegrad.engine.functions import Action



def save_model(fpath, model):
  
  if not isinstance(model, Model):
    raise TypeError(f'Expected Model object, instead got {type(model)}')
  with open(fpath,'wb') as fp:
    dill.dump(model, fp)
    print(f'MODEL SAVED at {fpath}')


def load_model(fpath):
  
  with open(fpath,'rb') as fp:
    model = dill.load(fp)
    print(f'MODEL LOADED from {fpath}')
  return model


def get_batches(inputs, targets=None, batch_size=None):
  
  if targets is not None:
    assert inputs.shape[0]==targets.shape[0], '0th dim should be number of examples and should match'

  num_examples = inputs.shape[0]

  if batch_size is not None:
    if batch_size > num_examples:
      raise ValueError("Batch size cannot be greater than the number of examples")
    elif batch_size<0:
      raise ValueError("Batch size cannot be negative")
    elif batch_size==0:
      raise ValueError("Batch size cannot be zero")
  else:
    batch_size = num_examples

  start = 0
  while start<num_examples:
    end = start+batch_size if start+batch_size<num_examples else num_examples
    if targets is not None:
      yield inputs[start:end], targets[start:end]
    else:
      yield inputs[start:end]
    start = end
    
#---------------------#
import math
import numpy as np



class Conv:
  
  def generate_fragments(self, padded_data, kernel_shape):
    
    assert self.stride>=1, 'Stride must be greater than or equal to 1'
    assert self.padding>=0, 'Padding must be greater than or equal to zero'
    padded_data_slice = ()
    padded_data_slice += ((slice(None)),)*(len(padded_data.shape)-2)
    inputs_x_dim, inputs_y_dim = padded_data.shape[-2:]
    kernel_x_dim, kernel_y_dim = kernel_shape[-2:]
    j = 0
    while(j+kernel_y_dim<=inputs_y_dim):
      i = 0
      while(i+kernel_x_dim<=inputs_x_dim):
        row_slice = slice(i, i+kernel_x_dim)
        col_slice = slice(j, j+kernel_y_dim)
        yield padded_data[padded_data_slice+(row_slice, col_slice)], row_slice, col_slice
        i+=self.stride
      j+=self.stride
  
  def get_result_shape(self, inputs_shape, kernel_shape):
    
    inputs_x_dim, inputs_y_dim = inputs_shape[-2:]
    kernel_x_dim, kernel_y_dim = kernel_shape[-2:]
    def result_dim(inputs_dim, kernel_dim):
      return math.floor(((inputs_dim + (2*self.padding) - kernel_dim)/self.stride) + 1)
    result_x_dim = result_dim(inputs_x_dim, kernel_x_dim)
    result_y_dim = result_dim(inputs_y_dim, kernel_y_dim)
    return result_x_dim, result_y_dim
  
  def pad(self, data):
    
    padding = () 
    padding+=((0,0),)*(len(data.shape)-2)
    padding+=((self.padding,self.padding),)*2
    return np.pad(data, padding)

  def unpad(self, padded_data):
    
    padded_x_dim, padded_y_dim = padded_data.shape[-2:]
    extractor_slice = ()
    extractor_slice+=((slice(None)),)*(len(padded_data.shape)-2)
    extractor_slice+=(slice(self.padding,padded_x_dim-self.padding), slice(self.padding,padded_y_dim-self.padding))
    return padded_data[extractor_slice]
  
  def fragment_iterator(self, padded_inputs, kernel_shape, *args):
    
    return zip(self.generate_fragments(padded_inputs, kernel_shape), *args)


# <------------CONV2D------------>
class Conv2D(Action, Conv):
  
  def __init__(self, padding, stride):
    self.padding = padding
    self.stride = stride
  
  def forward(self, inputs, kernel, bias):
    
    inputs, kernel, bias = self.get_Whalors(inputs, kernel, bias)
    self.validate_inputs(inputs)
    outputs = np.empty((inputs.shape[0], *self.get_result_shape(inputs.shape, kernel.shape)))
    padded_inputs = self.pad(inputs.data)
    for (fragment, _, _), idx in self.fragment_iterator(padded_inputs, kernel.shape, np.ndindex(outputs.shape[-2:])):
      output = np.sum((fragment*kernel.data), axis=(1,2)) + bias.data
      outputs[:,idx[0],idx[1]] = output
    return self.get_result_whalor(outputs, inputs, kernel, bias)
  
  def backward(self, inputs, kernel, bias):
    
    from whalegrad.engine.toolbox import unbroadcast_data
    padded_inputs = self.pad(inputs.data)

    def inputs_backward(ug):
      inputs_grads = np.zeros(padded_inputs.shape)
      for (fragment, row_slice, col_slice), idx in self.fragment_iterator(padded_inputs, kernel.shape, np.ndindex(ug.shape[-2:])):
        sliced_ug = ug[:,idx[0],idx[1]]
        sum_grad = np.ones(fragment.shape)*sliced_ug.reshape(sliced_ug.size,1,1)
        fragment_grad = kernel.data*sum_grad
        inputs_grads[:, row_slice, col_slice]+=fragment_grad
      unpadded_inputs_grads = self.unpad(inputs_grads)
      return unpadded_inputs_grads

    def kernel_backward(ug):
      kernel_grads = np.zeros(kernel.shape)
      for (fragment, _, _), idx in self.fragment_iterator(padded_inputs, kernel.shape, np.ndindex(ug.shape[-2:])):
        sliced_ug = ug[:,idx[0],idx[1]]
        sum_grad = np.ones(fragment.shape)*sliced_ug.reshape(sliced_ug.size,1,1)
        kernel_grad = unbroadcast_data(fragment*sum_grad, kernel.shape, fragment.shape)
        kernel_grads+=kernel_grad
      return kernel_grads

    def bias_backward(ug):
      return np.sum(ug)
      
    inputs.set_grad_fn(inputs_backward)
    kernel.set_grad_fn(kernel_backward)
    bias.set_grad_fn(bias_backward)
  
  def validate_inputs(self, inputs):
    
    if len(inputs.shape)!=3: # The first dimension should be number of examples
      raise ValueError("Only 3D inputs, with 0th dim as number of examples are supported!")

def conv2d(inputs, kernel, bias, padding, stride):
  
  return Conv2D(padding, stride).forward(inputs, kernel, bias)


# <------------CONV3D------------>
class Conv3D(Action, Conv):
  
  def __init__(self, padding, stride):
    self.padding = padding
    self.stride = stride
  
  def forward(self, inputs, kernel, bias):
    
    inputs, kernel, bias = self.get_Whalors(inputs, kernel, bias)
    self.validate_inputs(inputs)
    outputs = np.empty((inputs.shape[0], kernel.shape[0], *self.get_result_shape(inputs.shape, kernel.shape)))
    padded_inputs = self.pad(inputs.data)
    for (fragment,_,_), idx in self.fragment_iterator(padded_inputs, kernel.shape, np.ndindex(outputs.shape[-2:])):
      expanded_fragment = np.expand_dims(fragment, axis=1)
      output = expanded_fragment*kernel.data
      output = np.sum(output, axis=(2,3,4)) + bias.data
      outputs[:,:,idx[0],idx[1]] = output
    return self.get_result_tensor(outputs, inputs, kernel, bias)
  
  def backward(self, inputs, kernel, bias):
    
    from whalegrad.engine.toolbox import unbroadcast_data

    padded_inputs = self.pad(inputs.data)

    def inputs_backward(ug):
      inputs_grads = np.zeros(padded_inputs.shape)
      for (fragment, row_slice, col_slice), idx in self.fragment_iterator(padded_inputs, kernel.shape, np.ndindex(ug.shape[-2:])):
        expanded_fragment = np.expand_dims(fragment, axis=1)
        sliced_ug = ug[:,:,idx[0],idx[1]]
        sliced_ug = sliced_ug.reshape(*sliced_ug.shape,1,1,1)
        sum_grad = np.ones(expanded_fragment.shape)*sliced_ug
        fragment_grad = np.sum(sum_grad*kernel.data, axis=1)
        inputs_grads[:,:,row_slice,col_slice]+=fragment_grad
      unpadded_inputs_grads = self.unpad(inputs_grads)
      return unpadded_inputs_grads
    
    def kernel_backward(ug):
      kernel_grads = np.zeros(kernel.shape)
      for (fragment,_,_), idx in self.fragment_iterator(padded_inputs, kernel.shape, np.ndindex(ug.shape[-2:])):
        expanded_fragment = np.expand_dims(fragment,1)
        sliced_ug = ug[:,:,idx[0],idx[1]]
        sliced_ug = sliced_ug.reshape(*sliced_ug.shape,1,1,1)
        sum_grad = np.ones(expanded_fragment.shape)*sliced_ug
        kernel_grad = sum_grad*expanded_fragment
        kernel_grad = unbroadcast_data(kernel_grad, kernel.shape, kernel_grad.shape)
        kernel_grads+=kernel_grad
      return kernel_grads
    
    def bias_backward(ug):
      grad = np.sum(ug, axis=0)
      grad = np.sum(grad, axis=2, keepdims=True)
      grad = np.sum(grad, axis=1, keepdims=True)
      return grad
    
    inputs.set_grad_fn(inputs_backward)
    kernel.set_grad_fn(kernel_backward)
    bias.set_grad_fn(bias_backward)
  
  def validate_inputs(self, inputs):
    
    if len(inputs.shape)!=4:
      raise ValueError("Only 4D inputs, with 0th dim as number of examples, 1st dim as number of channels are supported!")

def conv3d(inputs, kernel, bias, padding, stride):
  
  return Conv3D(padding, stride).forward(inputs, kernel, bias)


# <------------MAXPOOL2D------------>
class MaxPool2D(Action, Conv):
  
  def __init__(self, kernel_shape, padding, stride):
    
    if len(kernel_shape)==2:
      self.kernel_shape = kernel_shape
    else:
      raise ValueError("Only 2D kernels are allowed!")
    self.padding = padding
    self.stride = stride
  
  def forward(self, inputs):
    
    inputs = self.get_Whalors(inputs)
    self.validate_inputs(inputs)
    outputs = np.empty((inputs.shape[0], *self.get_result_shape(inputs.shape, self.kernel_shape)))
    padded_inputs = self.pad(inputs.data)
    for (fragment,_,_), idx in self.fragment_iterator(padded_inputs, self.kernel_shape, np.ndindex(outputs.shape[-2:])):
      outputs[:,idx[0],idx[1]] = np.max(fragment, axis=(1,2))
    return self.get_result_tensor(outputs, inputs)
  
  def backward(self, inputs):
    
    padded_inputs = self.pad(inputs.data)

    def inputs_backward(ug):
      inputs_grad = np.empty(padded_inputs.shape)
      for (fragment,row_slice,col_slice),idx in self.fragment_iterator(padded_inputs, self.kernel_shape, np.ndindex(ug.shape[-2:])):
        sliced_ug = ug[:,idx[0],idx[1]]
        fragment_shape = fragment.shape
        flattened_fragment = fragment.reshape(fragment_shape[0], fragment_shape[1]*fragment_shape[2])
        args = np.argmax(flattened_fragment, axis=-1)
        fragment_grad = np.eye(flattened_fragment.shape[-1])[args] # one hot encoding of args
        fragment_grad = fragment_grad.reshape(fragment_shape)
        inputs_grad[:,row_slice,col_slice] = fragment_grad*np.expand_dims(sliced_ug,axis=(1,2))
      unpadded_inputs_grads = self.unpad(inputs_grad)
      return unpadded_inputs_grads

    inputs.set_grad_fn(inputs_backward)
  
  def validate_inputs(self, inputs):
    
    if len(inputs.shape)!=3: # The first dimension should be number of examples
      raise ValueError("Only 3D inputs, with 0th dim as number of examples are supported!")

def maxpool2d(inputs, kernel_shape, padding, stride):
  
  return MaxPool2D(kernel_shape, padding, stride).forward(inputs)


# <------------MAXPOOL3D------------>
class MaxPool3D(Action, Conv):
  
  def __init__(self, kernel_shape, padding, stride):
    
    if len(kernel_shape)==2:
      self.kernel_shape = kernel_shape
    else:
      raise ValueError("Only 2D kernels are allowed!")
    self.padding = padding
    self.stride = stride
  
  def forward(self, inputs):
    
    inputs = self.get_Whalors(inputs)
    self.validate_inputs(inputs)
    outputs = np.empty((inputs.shape[0], inputs.shape[1], *self.get_result_shape(inputs.shape, self.kernel_shape)))
    padded_inputs = self.pad(inputs.data)
    for (fragment,_,_), idx in self.fragment_iterator(padded_inputs, self.kernel_shape, np.ndindex(outputs.shape[-2:])):
      outputs[:,:,idx[0],idx[1]] = np.max(fragment, axis=(2,3))
    return self.get_result_tensor(outputs, inputs)
  
  def backward(self, inputs):
    
    padded_inputs = self.pad(inputs.data)

    def inputs_backward(ug):
      inputs_grad = np.empty(padded_inputs.shape)
      for (fragment,row_slice,col_slice),idx in self.fragment_iterator(padded_inputs, self.kernel_shape, np.ndindex(ug.shape[-2:])):
        sliced_ug = ug[:,:,idx[0],idx[1]]
        fragment_shape = fragment.shape
        flattened_fragment = fragment.reshape(fragment_shape[0]*fragment_shape[1], fragment_shape[2]*fragment_shape[3])
        args = np.argmax(flattened_fragment, axis=-1)
        fragment_grad = np.eye(flattened_fragment.shape[-1])[args] # one hot encoding of args
        fragment_grad = fragment_grad.reshape(fragment_shape)
        inputs_grad[:,:,row_slice,col_slice] = fragment_grad*np.expand_dims(sliced_ug,axis=(2,3))
      unpadded_inputs_grads = self.unpad(inputs_grad)
      return unpadded_inputs_grads
    
    inputs.set_grad_fn(inputs_backward)
  
  def validate_inputs(self, inputs):
    
    if len(inputs.shape)!=4:
      raise ValueError("Only 4D inputs, with 0th dim as number of examples, 1st dim as number of channels are supported!")

def maxpool3d(inputs, kernel_shape, padding, stride):
  
  return MaxPool3D(kernel_shape, padding, stride).forward(inputs)