import numpy as np

def he_initializer(shape):
    return np.random.randn(*shape) * np.sqrt(2 / shape[0])

def he_uniform_initializer(shape):
    return np.random.uniform(-np.sqrt(6 / shape[0]), np.sqrt(6 / shape[0]), shape)

def xavier_initializer(shape):
    return np.random.randn(*shape) * np.sqrt(1 / shape[0])

def random_initializer(shape):
    return np.random.randn(*shape)

def zeros_initializer(shape):
    return np.zeros(shape)

def ones_initializer(shape):
    return np.ones(shape)

initializers = {
    "he": he_initializer,
    "he_uniform": he_uniform_initializer,
    "xavier": xavier_initializer,
    "random": random_initializer,
    "zeros": zeros_initializer,
    "ones": ones_initializer
}