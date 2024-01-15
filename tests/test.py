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
# import numpy as np
# from whalegrad.nn.layers.base import Module

from whalegrad.nn.acc import Accuracy

# linear_layer = Linear(in_features=3, out_features=2)

# # Generate some random input data (replace this with your actual input data)
# input_data = np.random.randn(1, 3)

# # Perform forward pass
# output = linear_layer.forward(input_data)

# # Print the results
# print("Linear Layer Representation:", linear_layer)
# print("Input Data:\n", input_data)
# print("Output:\n", output)

import numpy as np

# y_preds = np.array([0.9, 0.2, 0.1, 0.5, 0.8 ])
# y_true = np.array([0.9, 0.2, 0.1, 0.5, 0.7 ])

# accc = acc()

# print(accc.forward(y_preds=y_preds, y_true=y_true))
# Create an instance of the Accuracy class
accuracy_calculator = Accuracy(y_preds=np.array([1, 20, 90, 89, 50]), y_true=np.array([1, 2, 3, 4, 0]))

# Calculate the accuracy using the forward method
accuracy = accuracy_calculator.forward()

# Print the result
print("Accuracy:", accuracy)
