import unittest
import numpy as np
from whalegrad.nn.layers.activations import ReLU, sigmoid, tanh, softmax, LeakyReLU, Swish, SwiGLU
import pytest

def test_relu():
    relu = ReLU()
    input_data = np.array([[-1, 0, 1], [2, -2, 3]])
    output = relu.forward(input_data)
    expected_output = np.array([[0, 0, 1], [2, 0, 3]])
    np.testing.assert_array_equal(output.data, expected_output)

def test_sigmoid():
    Sigmoid = sigmoid()
    input_data = np.array([[-1, 0, 1], [2, -2, 3]])
    output = Sigmoid.forward(input_data)
    expected_output = 1 / (1 + np.exp(-input_data))
    np.testing.assert_allclose(output.data, expected_output)

def test_tanh():
    Tanh = tanh()
    input_data = np.array([[-1, 0, 1], [2, -2, 3]])
    output = Tanh.forward(input_data)
    expected_output = np.tanh(input_data)
    np.testing.assert_allclose(output.data, expected_output)

def test_softmax():
    Softmax = softmax(axis=1)  # Assuming axis=1 for simplicity
    input_data = np.array([[1, 2, 3], [4, 5, 6]])
    output = Softmax.forward(input_data)
    expected_output = np.apply_along_axis(Softmax.calc_softmax, 1, input_data)
    np.testing.assert_allclose(output.data, expected_output)

def test_leaky_relu():
    leaky_relu = LeakyReLU(leak=0.01)
    input_data = np.array([[-1, 0, 1], [2, -2, 3]])
    output = leaky_relu.forward(input_data)
    expected_output = np.where(input_data >= 0, input_data, 0.01 * input_data)
    np.testing.assert_allclose(output.data, expected_output)


def test_swish():
    swish = Swish(beta=2.0)
    input_data = np.array([[-1, 0, 1], [2, -2, 3]])
    output = swish.forward(input_data)
    expected_output = input_data / (1 + np.exp(-2 * input_data))
    np.testing.assert_allclose(output.data, expected_output)    

def test_swiglu():
    swiglu = SwiGLU(beta=2.0)
    input_data = np.array([[-1, 0, 1], [2, -2, 3]])
    output = swiglu.forward(input_data)
    gate = 1 / (1 + np.exp(-2 * input_data))
    expected_output = input_data * gate
    np.testing.assert_allclose(output.data, expected_output)


