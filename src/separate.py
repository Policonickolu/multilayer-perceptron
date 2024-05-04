import argparse
import numpy as np
import pandas as pd
import os
import sys
import utils.data

def split_data(data: pd.DataFrame, train_ratio=0.8, rng=None):

    # Check if the dataset has more than one row
    if len(data) <= 1:
        print("Error: The dataset must have more than one row.")
        sys.exit(1)

    # Shuffle the dataset
    data = data.sample(frac=1, random_state=rng).reset_index(drop=True)

    # Index where to split
    train_index = int(len(data) * train_ratio)

    # Split the dataset
    train_data = data[:train_index]
    val_data = data[train_index:]

    return train_data, val_data

def get_args() -> argparse.Namespace:
    # Parse command-line arguments
    script_dir = os.path.dirname(os.path.realpath(__file__))
    parser = argparse.ArgumentParser(description='Split a dataset into a training set and a validation set.')
    parser.add_argument('--data_path', type=str, default=os.path.join(script_dir, '../data/data.csv'), help='The path to the dataset file. (default: ../data/data.csv)')
    parser.add_argument('--output_path', type=str, default=os.path.join(script_dir, '../data'), help='The path to the output folder. (default: ../data)')
    parser.add_argument('--val_size', type=float, default=0.2, help='The size of the validation set as a proportion of the total data. (default: %(default)s)')
    parser.add_argument('--seed', type=int, default=None, help='The random seed for shuffling the data.')
    return parser.parse_args()

def main():
    # Get command-line arguments
    args = get_args()

    # Random number generator for shuffling
    rng = None
    if args.seed is not None:
        rng = np.random.default_rng(args.seed)
    else:
        rng = np.random.default_rng()

    # Load the data
    data = utils.data.load(args.data_path)
    if data is None:
        return

    # Split the dataset
    train_ratio = 1 - args.val_size
    train_data, val_data = split_data(data, train_ratio, rng)

    # Save the training and validation data
    train_data.to_csv(os.path.join(args.output_path, 'train.csv'), index=False, header=False)
    val_data.to_csv(os.path.join(args.output_path, 'val.csv'), index=False, header=False)

if __name__ == '__main__':
    main()