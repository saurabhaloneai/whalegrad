# import numpy as np
# import pytest
# from whalegrad.nn.loss import MeanSquaredError, BinaryCrossEntropy, CrossEntropy, SoftmaxCE

# # Test MeanSquaredError
# # Test MeanSquaredError

# inputs = np.array([1.0, 2.0, 3.0])

# def self_attention(inputs):
#     return inputs   # Dummy function


# def test_mean_squared_error():
#     mse = MeanSquaredError()
#     targets = np.array([1.0, 2.0, 3.0])
#     output = mse.forward(inputs, targets)
#     expected_output = np.mean(np.square(inputs - targets))
#     assert output == expected_output
    


dic = { "a" : 1, "b" : 2, "c" : 3}


diff = 3

if diff in dic:
    
  a = dic[diff]
  
  
  print(a)