import numpy as np

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    return x * (1 - x)

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)

def relu_derivative(x: np.ndarray) -> np.ndarray:
    return np.where(x <= 0, 0, 1)

def softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def softmax_derivative(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    softmax = e_x / e_x.sum(axis=0)

    s = softmax.reshape(-1, 1)
    jacobian = np.diagflat(s) - np.dot(s, s.T)

    return jacobian

def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def tanh_derivative(x: np.ndarray) -> np.ndarray:
    return 1 - np.tanh(x) ** 2

def linear(x: np.ndarray) -> np.ndarray:
    return x

def linear_derivative(x: np.ndarray) -> np.ndarray:
    return np.ones_like(x)

activations = {
    "sigmoid": {
        "function": sigmoid,
        "derivative": sigmoid_derivative
    },
    "softmax": {
        "function": softmax,
        "derivative": softmax_derivative
    },
    "relu": {
        "function": relu,
        "derivative": relu_derivative
    },
    "tanh": {
        "function": tanh,
        "derivative": tanh_derivative
    },
    "linear": {
        "function": linear,
        "derivative": linear_derivative
    }
}