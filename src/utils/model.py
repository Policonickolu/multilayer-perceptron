from joblib import dump, load

def save_model(filename, weights, biases, layers, activations):
    dump((weights, biases, layers, activations), filename)

def load_model(filename):
    weights, biases, layers, activations = load(filename)
    return weights, biases, layers, activations