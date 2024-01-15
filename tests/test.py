## make the classifier for binary clssification

# TODO :
# 1. linear : ✅
# 2. optim => adam , sgd
# 3. loss => mse, Binary Cross Entropy
# 4. acc => manual
# 5. activation => sigmoid, relu and tanh
# 6. no grad caln for inference like pytoch
# 7. save and load model 
# 8. add more activation function


# y = xw + b
import numpy as np
from whalegrad.nn.layers.base import Module
from whalegrad.nn.layers.linear import Linear

# class model(Module):
    
#     def__init__(self,in_features.)


# x# Assuming you have already defined your Module and Param classes
# ...

# Create an instance of the Linear class
linear_layer = Linear(in_features=3, out_features=2)

# Generate some random input data (replace this with your actual input data)
input_data = np.random.randn(1, 3)

# Perform forward pass
output = linear_layer.forward(input_data)

# Print the results
print("Linear Layer Representation:", linear_layer)
print("Input Data:\n", input_data)
print("Output:\n", output)
  