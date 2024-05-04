from dataclasses import dataclass
import os
import utils.data as data
from typing import Dict, List
from utils.hyperparameters import Hyperparameters
from utils.metrics import save_metrics
from utils.model import save_model
from utils.neural_network import NeuralNetwork
from utils.plot import plot_loss_and_accuracy
from utils.train_args import get_args

def main():
    args = get_args()

    X_train, y_train, _ = data.load_and_clean_data(args.train_data_path, args.label_column, args.headers, args.disabled_columns, args.outliers)
    X_val, y_val, _ = data.load_and_clean_data(args.val_data_path, args.label_column, args.headers, args.disabled_columns, args.outliers)

    # Check train and val have thame shape (but not the same number of examples)
    if X_train.shape[1] != X_val.shape[1]:
        print(f"Train and validation data have different number of features: {X_train.shape[1]} != {X_val.shape[1]}")
        exit(1)
    if y_train.shape[1] != y_val.shape[1]:
        print(f"Train and validation data have different number of unique labels: {y_train.shape[1]} != {y_val.shape[1]}")
        exit(1)
    if X_train.shape[1] < 1:
        print(f"Train and validation data have no features.")
        exit(1)
    if y_train.shape[1] < 1:
        print(f"Train and validation data have no labels.")
        exit(1)

    print(f'x_train shape: {X_train.shape}')
    print(f'x_valid shape: {X_val.shape}')

    hyperparameters = Hyperparameters(
        layers=args.layers,
        activations=args.activations,
        num_classes=y_train.shape[1],
        shuffle=not args.no_shuffle,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        iteration_type=args.iteration_type,
        batch_size=args.batch_size,
        optimizer=args.optimizer,
        initializer=args.initializer,
        momentum=args.momentum,
        patience=args.patience,
        decay=args.decay
    )

    # Size of input layer = number of features
    hyperparameters.layers.insert(0, X_train.shape[1])
    # Size of output layer = number of unique values in y_train
    hyperparameters.layers.append(y_train.shape[1])

    nn = NeuralNetwork(hyperparameters)


    # weights, biases, losses, accuracies, val_losses, val_accuracies =
    nn.train(
        X_train, y_train,
        iterations=hyperparameters.epochs,
        learning_rate=0.002,
        X_val=X_val, Y_val=y_val
    )

    # Save the weights and biases
    print(f"> saving model '{os.path.relpath(args.model_path)}' to disk...")
    save_model(args.model_path, nn.weights, nn.biases, nn.layers, nn.activations)

    train_losses = [metrics['loss'] for metrics in nn.train_metrics]
    train_accuracies = [metrics['accuracy'] for metrics in nn.train_metrics]
    val_losses = [metrics['loss'] for metrics in nn.val_metrics]
    val_accuracies = [metrics['accuracy'] for metrics in nn.val_metrics]

    if args.plot:
        plot_loss_and_accuracy(train_losses, train_accuracies, val_losses, val_accuracies)

    save_metrics('metrics/train_metrics.csv', nn.train_metrics)
    save_metrics('metrics/val_metrics.csv', nn.val_metrics)

    print('Training complete')

if __name__ == '__main__':
    main()