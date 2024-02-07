import numpy as np
import pytest
from whalegrad.nn.layers.convolutions import Conv2D, Conv3D, MaxPool2D, MaxPool3D

# Replace 'your_module_name' with the actual module name where the Conv2D, Conv3D, MaxPool2D, MaxPool3D classes are defined.

@pytest.fixture
def sample_input():
    # You may need to modify this depending on the input requirements of your functions.
    return np.random.randn(1, 3, 32, 32)  # Assuming 3-channel image of size 32x32

def test_conv2d_forward(sample_input):
    conv_layer = Conv2D(kernel_shape=(3, 3), padding=1, stride=1)
    output = conv_layer.forward(sample_input)
    assert output.shape == (1, 3, 32, 32)  # Adjust the expected shape accordingly

def test_conv3d_forward(sample_input):
    conv_layer = Conv3D(in_channels=3, out_channels=16, kernel_shape=(3, 3, 3), padding=1, stride=1)
    output = conv_layer.forward(sample_input)
    assert output.shape == (1, 16, 32, 32)  # Adjust the expected shape accordingly

def test_maxpool2d_forward(sample_input):
    maxpool_layer = MaxPool2D(kernel_shape=(2, 2), padding=0, stride=2)
    output = maxpool_layer.forward(sample_input)
    assert output.shape == (1, 3, 16, 16)  # Adjust the expected shape accordingly

def test_maxpool3d_forward(sample_input):
    maxpool_layer = MaxPool3D(kernel_shape=(2, 2, 2), padding=0, stride=2)
    output = maxpool_layer.forward(sample_input)
    assert output.shape == (1, 3, 16, 16)  # Adjust the expected shape accordingly

# Add more test functions as needed based on your implementation details.

# To run the tests, execute the following command in your terminal:
# pytest test_conv_layers.py
