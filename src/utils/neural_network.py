import numpy as np

from utils.activations import activations, softmax

from utils.evaluations import categorical_crossentropy
from utils.hyperparameters import Hyperparameters
from utils.initializations import initializers , zeros_initializer

class NeuralNetwork:
    def __init__(self, params: Hyperparameters):
        print("Initializing Neural Network with layer sizes", list(zip(params.layers, [None] + params.activations + ['softmax'])))
        self.layers = params.layers

        self.activations = params.activations
        act = [activations[activation] for activation in params.activations]
        self.activations_function = [activation['function'] for activation in act]
        self.activations_derivatives = [activation['derivative'] for activation in act]

        self.num_classes = params.num_classes

        self.optimizer = params.optimizer
        self.initializer = params.initializer

        self.shuffle = params.shuffle

        self.epochs = params.epochs
        self.learning_rate = params.learning_rate

        self.iteration_type = params.iteration_type
        self.batch_size = params.batch_size

        self.momentum = params.momentum
        self.patience = params.patience
        self.decay = params.decay

        self.weights = self.initialize_weights(initializers[params.initializer])
        self.biases = self.initialize_biases()

        self.epsilon = 1e-8

        if self.optimizer in ['momentum', 'nesterov', 'adam', 'adamax', 'nadam', 'radam']:
            self.m_w, self.m_b = self.initialize_cache()
        if self.optimizer in ['adagrad', 'rmsprop', 'adam', 'adamax', 'nadam', 'adadelta', 'radam']:
            self.beta1 = self.decay[0]
            self.beta2 = self.decay[1]
            self.cache_w, self.cache_b = self.initialize_cache()
        if self.optimizer == 'adadelta':
            self.delta_w, self.delta_b = self.initialize_cache()
        if self.optimizer == 'ftrl':
            self.alpha = self.decay[0]
            self.beta = self.decay[1]
            self.z_w, self.z_b, self.n_w, self.n_b = self.initialize_ftrl()

    def initialize_weights(self, initializer):
        weights = {}
        for i in range(1, len(self.layers)):
            weights[i] = initializer((self.layers[i-1], self.layers[i]))
        return weights

    def initialize_biases(self):
        biases = {}
        for i in range(1, len(self.layers)):
            biases[i] = zeros_initializer((self.layers[i]))
        return biases

    def initialize_cache(self):
        cache_w = {}
        cache_b = {}
        for i in range(1, len(self.layers)):
            cache_w[i] = zeros_initializer((self.layers[i-1], self.layers[i]))
            cache_b[i] = zeros_initializer((self.layers[i],))
        return cache_w, cache_b

    def initialize_ftrl(self):
        z_w = {}
        z_b = {}
        n_w = {}
        n_b = {}
        for i in range(1, len(self.layers)):
            z_w[i] = zeros_initializer((self.layers[i-1], self.layers[i]))
            z_b[i] = zeros_initializer((self.layers[i],))
            n_w[i] = zeros_initializer((self.layers[i-1], self.layers[i]))
            n_b[i] = zeros_initializer((self.layers[i],))
        return z_w, z_b, n_w, n_b

    def forward_propagation(self, X):
        A = X
        cache = {0: A}
        for i in range(1, len(self.layers) - 1):
            Z = A.dot(self.weights[i]) + self.biases[i]
            A = self.activations_function[i-1](Z)
            cache[i] = A

        Z = A.dot(self.weights[len(self.layers) - 1]) + self.biases[len(self.layers) - 1]
        A = softmax(Z)
        cache[len(self.layers) - 1] = A

        return A, cache

    def backpropagation(self, X, Y, cache):
        m = X.shape[1]
        gradients = {}
        A = cache[len(self.layers) - 1]
        dZ = A - Y
        gradients['dw' + str(len(self.layers) - 1)] = (1 / m) * cache[len(self.layers) - 2].T.dot(dZ)
        gradients['db' + str(len(self.layers) - 1)] = (1 / m) * np.sum(dZ, axis=0)
        dA = dZ.dot(self.weights[len(self.layers) - 1].T)
        for i in reversed(range(1, len(self.layers) - 1)):
            dZ = dA * self.activations_derivatives[i-1](cache[i])
            gradients['dw' + str(i)] = (1 / m) * cache[i-1].T.dot(dZ)
            gradients['db' + str(i)] = (1 / m) * np.sum(dZ, axis=0)
            dA = dZ.dot(self.weights[i].T)
        return gradients

    def compute_cost(self, A, Y):
        return categorical_crossentropy(Y, A)

    def compute_metrics(self, A, Y):
        predictions = np.argmax(A, axis=1)
        labels = np.argmax(Y, axis=1)

        accuracy = np.mean(predictions == labels)

        TP = np.sum([np.sum((predictions == i) & (labels == i)) for i in range(self.num_classes)])
        FP = np.sum([np.sum((predictions == i) & (labels != i)) for i in range(self.num_classes)])
        FN = np.sum([np.sum((predictions != i) & (labels == i)) for i in range(self.num_classes)])

        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=int)
        for i in range(len(labels)):
            confusion_matrix[labels[i]][predictions[i]] += 1

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'confusion_matrix': confusion_matrix
        }

        return metrics

    def predict(self, X):
        A, _ = self.forward_propagation(X)
        return A

    def train(self, X, Y, iterations, learning_rate, X_val=None, Y_val=None):
        print('Training...')
        print(f'Epochs:\t\t {self.epochs}')
        print(f'Learning rate:\t {self.learning_rate}')
        print(f'Optimizer:\t {self.optimizer}')
        print(f'Initializer:\t {self.initializer}')
        print(f'Patience:\t {self.patience}')
        print(f'Iteration type:\t {self.iteration_type}')
        print(f'Batch size:\t {self.batch_size}')
        print(f'Shuffle:\t {self.shuffle}')

        self.train_metrics = []
        self.val_metrics = []

        if self.iteration_type == 'minibatch':
            batch_size = self.batch_size
        elif self.iteration_type == 'stochastic':
            batch_size = 1
        else:
            batch_size = X.shape[0]

        best_cost = float('inf')
        patience_counter = 0

        for i in range(iterations):
            if self.shuffle:
                permutation = np.random.permutation(X.shape[0])
                X = X[permutation]
                Y = Y[permutation]

            for j in range(0, X.shape[0], batch_size):
                X_mini = X[j:j + self.batch_size]
                Y_mini = Y[j:j + self.batch_size]
                A, cache = self.forward_propagation(X_mini)
                gradients = self.backpropagation(X_mini, Y_mini, cache)
                self.update_parameters(gradients, learning_rate, i + j + 1)

            A = self.forward_propagation(X)[0]
            cost = self.compute_cost(A, Y)
            A_val = self.forward_propagation(X_val)[0]
            cost_val = self.compute_cost(A_val, Y_val)

            metrics = self.compute_metrics(A, Y)
            metrics = {'epoch': i + 1, 'loss': cost, **metrics}
            self.train_metrics.append(metrics)
            metrics_val = self.compute_metrics(A_val, Y_val)
            metrics_val = {'epoch': i + 1, 'loss': cost_val, **metrics_val}
            self.val_metrics.append(metrics_val)

            compared_cost = cost_val
            print(f'epoch {i+1}/{iterations}, loss: {cost:.4f} - acc: {metrics["accuracy"]:.4f} - f1: {metrics["f1_score"]:.4f} - val_loss: {cost_val:.4f} - val_acc: {metrics_val["accuracy"]:.4f} - val_f1: {metrics_val["f1_score"]:.4f}')

            # Early stopping
            if compared_cost < best_cost:
                best_cost = compared_cost
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter > self.patience:
                print(f'Early stopping on epoch {i+1}')
                break

    def update_parameters(self, gradients, learning_rate, t):
        if self.optimizer == 'momentum':
            for i in range(1, len(self.layers)):
                self.m_w[i] = self.momentum * self.m_w[i] + learning_rate * gradients['dw' + str(i)]
                self.m_b[i] = self.momentum * self.m_b[i] + learning_rate * gradients['db' + str(i)]
                self.weights[i] -= self.m_w[i]
                self.biases[i] -= self.m_b[i]
        elif self.optimizer == 'nesterov':
            for i in range(1, len(self.layers)):
                v_prev_w = self.m_w[i]
                v_prev_b = self.m_b[i]
                self.m_w[i] = self.momentum * self.m_w[i] + learning_rate * gradients['dw' + str(i)]
                self.m_b[i] = self.momentum * self.m_b[i] + learning_rate * gradients['db' + str(i)]
                self.weights[i] -= (1 + self.momentum) * self.m_w[i] - self.momentum * v_prev_w
                self.biases[i] -= (1 + self.momentum) * self.m_b[i] - self.momentum * v_prev_b
        elif self.optimizer == 'adagrad':
            for i in range(1, len(self.layers)):
                self.cache_w[i] += gradients['dw' + str(i)] ** 2
                self.cache_b[i] += gradients['db' + str(i)] ** 2
                self.weights[i] -= learning_rate * gradients['dw' + str(i)] / (np.sqrt(self.cache_w[i]) + self.epsilon)
                self.biases[i] -= learning_rate * gradients['db' + str(i)] / (np.sqrt(self.cache_b[i]) + self.epsilon)
        elif self.optimizer == 'rmsprop':
            for i in range(1, len(self.layers)):
                self.cache_w[i] = self.decay[0] * self.cache_w[i] + (1 - self.decay[0]) * gradients['dw' + str(i)]**2
                self.cache_b[i] = self.decay[0] * self.cache_b[i] + (1 - self.decay[0]) * gradients['db' + str(i)]**2
                self.weights[i] -= learning_rate * gradients['dw' + str(i)] / (np.sqrt(self.cache_w[i]) + self.epsilon)
                self.biases[i] -= learning_rate * gradients['db' + str(i)] / (np.sqrt(self.cache_b[i]) + self.epsilon)
        elif self.optimizer == 'adam':
            for i in range(1, len(self.layers)):
                self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * gradients['dw' + str(i)]
                self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * gradients['db' + str(i)]
                self.cache_w[i] = self.beta2 * self.cache_w[i] + (1 - self.beta2) * gradients['dw' + str(i)]**2
                self.cache_b[i] = self.beta2 * self.cache_b[i] + (1 - self.beta2) * gradients['db' + str(i)]**2
                m_w_hat = self.m_w[i] / (1 - self.beta1**t)
                m_b_hat = self.m_b[i] / (1 - self.beta1**t)
                v_w_hat = self.cache_w[i] / (1 - self.beta2**t)
                v_b_hat = self.cache_b[i] / (1 - self.beta2**t)
                self.weights[i] -= learning_rate * m_w_hat / (np.sqrt(v_w_hat) + self.epsilon)
                self.biases[i] -= learning_rate * m_b_hat / (np.sqrt(v_b_hat) + self.epsilon)
        elif self.optimizer == 'adamax':
            for i in range(1, len(self.layers)):
                self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * gradients['dw' + str(i)]
                self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * gradients['db' + str(i)]
                self.cache_w[i] = np.maximum(self.beta2 * self.cache_w[i], np.abs(gradients['dw' + str(i)]))
                self.cache_b[i] = np.maximum(self.beta2 * self.cache_b[i], np.abs(gradients['db' + str(i)]))
                m_w_hat = self.m_w[i] / (1 - self.beta1**t)
                m_b_hat = self.m_b[i] / (1 - self.beta1**t)
                self.weights[i] -= learning_rate * m_w_hat / (self.cache_w[i] + self.epsilon)
                self.biases[i] -= learning_rate * m_b_hat / (self.cache_b[i] + self.epsilon)
        elif self.optimizer == 'nadam':
            for i in range(1, len(self.layers)):
                self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * gradients['dw' + str(i)]
                self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * gradients['db' + str(i)]
                m_w_hat = self.m_w[i] / (1 - self.beta1**t)
                m_b_hat = self.m_b[i] / (1 - self.beta1**t)
                self.cache_w[i] = self.beta2 * self.cache_w[i] + (1 - self.beta2) * gradients['dw' + str(i)]**2
                self.cache_b[i] = self.beta2 * self.cache_b[i] + (1 - self.beta2) * gradients['db' + str(i)]**2
                v_w_hat = self.cache_w[i] / (1 - self.beta2**t)
                v_b_hat = self.cache_b[i] / (1 - self.beta2**t)
                self.weights[i] -= learning_rate * (self.beta1 * m_w_hat + (1 - self.beta1) * gradients['dw' + str(i)]) / (np.sqrt(v_w_hat) + self.epsilon)
                self.biases[i] -= learning_rate * (self.beta1 * m_b_hat + (1 - self.beta1) * gradients['db' + str(i)]) / (np.sqrt(v_b_hat) + self.epsilon)
        elif self.optimizer == 'adadelta':
            for i in range(1, len(self.layers)):
                self.cache_w[i] = self.decay[0] * self.cache_w[i] + (1 - self.decay[0]) * gradients['dw' + str(i)]**2
                self.cache_b[i] = self.decay[0] * self.cache_b[i] + (1 - self.decay[0]) * gradients['db' + str(i)]**2
                delta_w = np.sqrt((self.delta_w[i] + self.epsilon) / (self.cache_w[i] + self.epsilon)) * gradients['dw' + str(i)]
                delta_b = np.sqrt((self.delta_b[i] + self.epsilon) / (self.cache_b[i] + self.epsilon)) * gradients['db' + str(i)]
                self.weights[i] -= delta_w
                self.biases[i] -= delta_b
                self.delta_w[i] = self.decay[0] * self.delta_w[i] + (1 - self.decay[0]) * delta_w**2
                self.delta_b[i] = self.decay[0] * self.delta_b[i] + (1 - self.decay[0]) * delta_b**2
        elif self.optimizer == 'radam':
            for i in range(1, len(self.layers)):
                self.m_w[i] = self.beta1 * self.m_w[i] + (1 - self.beta1) * gradients['dw' + str(i)]
                self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * gradients['db' + str(i)]
                self.cache_w[i] = self.beta2 * self.cache_w[i] + (1 - self.beta2) * gradients['dw' + str(i)]**2
                self.cache_b[i] = self.beta2 * self.cache_b[i] + (1 - self.beta2) * gradients['db' + str(i)]**2
                m_w_hat = self.m_w[i] / (1 - self.beta1**t)
                m_b_hat = self.m_b[i] / (1 - self.beta1**t)
                v_w_hat = self.cache_w[i] / (1 - self.beta2**t)
                v_b_hat = self.cache_b[i] / (1 - self.beta2**t)
                r_w = np.where(v_w_hat >= self.epsilon, np.sqrt((v_w_hat - self.epsilon) / (1 - self.beta2**t)), 0)
                r_b = np.where(v_b_hat >= self.epsilon, np.sqrt((v_b_hat - self.epsilon) / (1 - self.beta2**t)), 0)
                self.weights[i] -= learning_rate * r_w * m_w_hat
                self.biases[i] -= learning_rate * r_b * m_b_hat
        elif self.optimizer == 'ftrl':
            for i in range(1, len(self.layers)):
                g_w = gradients['dw' + str(i)]
                g_b = gradients['db' + str(i)]
                sigma_w = (np.sqrt(self.n_w[i] + g_w**2) - np.sqrt(self.n_w[i])) / self.alpha
                sigma_b = (np.sqrt(self.n_b[i] + g_b**2) - np.sqrt(self.n_b[i])) / self.alpha
                self.z_w[i] += g_w - sigma_w * self.weights[i]
                self.z_b[i] += g_b - sigma_b * self.biases[i]
                self.n_w[i] += g_w**2
                self.n_b[i] += g_b**2
                self.weights[i] = - ((self.beta + np.sqrt(self.n_w[i])) / self.alpha + self.epsilon) / self.z_w[i]
                self.biases[i] = - ((self.beta + np.sqrt(self.n_b[i])) / self.alpha + self.epsilon) / self.z_b[i]
        else:
            for i in range(1, len(self.layers)):
                self.weights[i] -= learning_rate * gradients['dw' + str(i)]
                self.biases[i] -= learning_rate * gradients['db' + str(i)]