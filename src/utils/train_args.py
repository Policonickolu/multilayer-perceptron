import argparse
import os
import sys

def get_args() -> argparse.Namespace:

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train a neural network model.')


    # Network Parameters

    parser.add_argument(
        '--layers', type=int, nargs='+', default=[24, 24],
        help='The number of nodes in each hidden layer of the network. (default: %(default)s)'
    )

    parser.add_argument(
        '--activations', choices=['sigmoid', 'relu', 'tanh', 'linear'], nargs='+', default=['sigmoid'],
        help='The activation function for each hidden layer (output will always be softmax).  (default: %(default)s)'
    )


    # Training Parameters

    parser.add_argument(
        '--epochs', type=int, default=100,
        help='The number of epochs to train the model for. (default: %(default)s)'
    )
    parser.add_argument(
        '--learning_rate', type=float, default=0.01,
        help='The learning rate for the optimizer. (default: %(default)s)'
    )
    parser.add_argument(
        '--no_shuffle', action='store_true',
        help='If the dataset should not be shuffled between epochs. (default: %(default)s)'
    )
    parser.add_argument(
        '--iteration_type',
        choices=['batch', 'stochastic', 'minibatch'],
        default='batch',
        help='The types of dataset iteration. (default: %(default)s)'
    )
    parser.add_argument(
        '--batch_size', type=int, default=8,
        help='The number of samples in each mini-batch. (default: %(default)s)'
    )
    parser.add_argument(
        '--optimizer',
        choices=[
            'gradient_descent', 'momentum', 'nesterov',
            'adagrad', 'rmsprop', 'adadelta', 'adam',
            'adamax', 'nadam', 'radam', 'ftrl'
        ],
        default='gradient_descent',
        help='The optimization function. (default: %(default)s)'
    )
    parser.add_argument(
        '--initializer',
        choices=[
            'he', 'he_uniform', 'xavier', 'random', 'zeros', 'ones'
        ],
        default='he',
        help='The initialization function. (default: %(default)s)'
    )
    parser.add_argument(
        '--momentum', type=float, default=0.9,
        help='The momentum used in some optimization functions. (default: %(default)s)'
    )
    parser.add_argument(
        '--patience', type=int, default=10,
        help='Number of epochs without improvement before early stopping. (default: %(default)s)'
    )
    parser.add_argument(
        '--decay', type=float, nargs='+', default=[0.9, 0.99],
        help='The decay rates used in some optimization functions. (default: %(default)s)'
    )

    # Data

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

    # Features

    parser.add_argument(
        '--plot', action='store_true',
        help='Plot the loss and accuracy after training. (default: %(default)s)'
    )


    # File Paths

    script_dir = os.path.dirname(os.path.abspath(sys.modules['__main__'].__file__))
    train_data_path = os.path.abspath(os.path.join(script_dir, '../data/train.csv'))
    val_data_path = os.path.abspath(os.path.join(script_dir, '../data/val.csv'))
    model_path = os.path.abspath(os.path.join(script_dir, '../models/model.npz'))

    parser.add_argument(
        '--train_data_path', type=str, default=train_data_path,
        help='The path to the training data file. (default: data/train.csv)'
    )
    parser.add_argument(
        '--val_data_path', type=str, default=val_data_path,
        help='The path to the validation data file. (default: data/val.csv)'
    )
    parser.add_argument(
        '--model_path', type=str, default=model_path,
        help='The path to save the trained model in. (default: models/model.npz)'
    )

    args = parser.parse_args()

    # Minimum number of layers is 1
    if len(args.layers) < 1:
        parser.error("The number of hidden layers must be at least 1.")

    # Minimum size in each layer is 1
    if min(args.layers) < 1:
        parser.error("The number of nodes in each layer must be at least 1.")

    # If missing activations functions
    if len(args.activations) < len(args.layers):
        # If activations contains only one value, repeat it for the remaining layers
        if len(args.activations) == 1:
            args.activations *= (len(args.layers))
        # If activations contains more than one value, print an error
        else:
            parser.error("The number of activations functions must match the number of layers when.")

    if len(args.activations) > len(args.layers):
        parser.error("The number of activations functions must match the number of layers.")

    if len(args.decay) < 2 and args.optimizer in ['adam', 'adamax', 'nadam', 'radam', 'ftrl']:
        parser.error("The number of decay rates must be at least 2 for the selected optimizer.")

    # if args.train_data_path is not None, check if args.train_data_path exists and is a file
    if not os.path.isfile(args.train_data_path):
        parser.error(f"File not found: {args.train_data_path}")
    # idem with args.val_data_path but only if args.val_data_path is not None
    if not os.path.isfile(args.val_data_path):
        parser.error(f"File not found: {args.val_data_path}")
    # check if args.model_path exists and is a directory, if true, add model.npz to the path
    if os.path.isdir(args.model_path):
        args.model_path = os.path.join(args.model_path, 'model.npz')
    # check if args.model_path exists or can be written
    if not os.access(os.path.dirname(args.model_path), os.W_OK):
        parser.error(f"Cannot write to the model path: {args.model_path}")


    return args