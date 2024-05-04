import argparse
import os
import sys
import numpy as np
import pandas as pd
from utils.data import load_and_clean_data
from utils.evaluations import binary_crossentropy, categorical_crossentropy
from utils.model import load_model
from utils.neural_network import NeuralNetwork
from utils.hyperparameters import Hyperparameters

def get_args():
    parser = argparse.ArgumentParser(description="Make predictions with a trained neural network model.")

    script_dir = os.path.dirname(os.path.abspath(sys.modules['__main__'].__file__))
    data_path = os.path.abspath(os.path.join(script_dir, '../data/val.csv'))
    model_path = os.path.abspath(os.path.join(script_dir, '../models/model.npz'))

    parser.add_argument(
        '--model', type=str, default=model_path,
        help="Path to the saved model. (default: models/model.npz)"
    )
    parser.add_argument(
        '--data_path', type=str, default=data_path,
        help="Path to the test data. (default: data/val.csv)"
    )
    parser.add_argument(
        '--label_column', type=int, default=1,
        help='The index of the column containing the labels. (default: %(default)s)'
    )
    parser.add_argument(
        '--disabled_columns', type=int, nargs='*', default=[0],
        help='The indexes of the columns to disable. (default: %(default)s)'
    )
    parser.add_argument(
        '--headers', action='store_true',
        help='The dataset has a headers row to ignore. (default: %(default)s)'
    )
    parser.add_argument(
        '--outliers', choices=['keep', 'clamp', 'remove'], default='keep',
        help='How to handle outliers in the dataset preprocessing. (default: %(default)s)'
    )

    return parser.parse_args()

def main():
    args = get_args()

    # Load the test data
    X, y, label_index_to_classes = load_and_clean_data(args.data_path, args.label_column, args.headers, args.disabled_columns, args.outliers)

    # Load the model
    weights, biases, layers, activations = load_model(args.model)

    hyperparameters = Hyperparameters(
        layers=layers,
        activations=activations,
        num_classes=layers[-1],
    )

    nn = NeuralNetwork(hyperparameters)
    nn.weights = weights
    nn.biases = biases

    print('Predicting...\n')
    y_pred = nn.predict(X)
    y_pred_hard = np.where(y_pred == np.max(y_pred, axis=1)[:, None], 1, 0)

    if nn.num_classes > 2:
        # Categorical classification
        loss = categorical_crossentropy(y, y_pred)
        loss_hard = categorical_crossentropy(y, y_pred_hard)
    else:
        # Binary classification by keeping the first column only
        loss = binary_crossentropy(y[:, 0], y_pred[:, 0])
        loss_hard = binary_crossentropy(y[:, 0], y_pred_hard[:, 0])


    print('Results:\n')

    classes_pred = np.argmax(y_pred, axis=1)
    classes = np.argmax(y, axis=1)
    classes_count = {label_index_to_classes[label]: np.sum(classes_pred == label) for label in np.unique(classes_pred)}
    print(f"Predicted classes repartition: {classes_count}")
    classes_count = {label_index_to_classes[label]: np.sum(classes == label) for label in np.unique(classes)}
    print(f"True classes repartition: {classes_count}")

    metrics = nn.compute_metrics(y, y_pred)

    print(f"\nSoft loss:\t{loss:.4f} \t\t(inf -> 0)")
    print(f"Hard loss:\t{loss_hard:.4f} \t\t(inf -> 0)")
    print(f"Accuracy: \t{metrics['accuracy']:.4f} \t\t(0 -> 1)")
    print(f"Precision:\t{metrics['precision']:.4f} \t\t(0 -> 1)")
    print(f"Recall:   \t{metrics['recall']:.4f} \t\t(0 -> 1)")
    print(f"F1 Score: \t{metrics['f1_score']:.4f} \t\t(0 -> 1)")
    print(f"\nConfusion matrix:")
    print("         \t" + '\t'.join(["Pred. " + label_index_to_classes[label] for label in range(nn.num_classes)]))
    for i in range(metrics['confusion_matrix'].shape[0]):
        print("Actual " + label_index_to_classes[i] + "\t" + '\t'.join([str(np.sum((classes_pred == i) & (classes == j))) for j in range(nn.num_classes)]))

if __name__ == '__main__':
    main()