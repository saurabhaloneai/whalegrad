# An activation function in a neural network defines 
# how the weighted sum of the input is transformed into an
# output from a node or nodes in a layer of the network.

# Sometimes the activation function is called a “transfer function.” 
# If the output range of the activation function is limited, 
# then it may be called a “squashing function.” 
# Many activation functions are nonlinear and 
# may be referred to as the “nonlinearity” in the layer or the network design.



import numpy as np


def sigmoid(x):
    
    return 1/(1+np.exp(-x))





