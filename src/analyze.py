import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():

    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Plot dataset info.')
    script_dir = os.path.dirname(os.path.realpath(__file__))
    parser.add_argument('--file', type=str, default=os.path.join(script_dir, '../data/data.csv'), help='Path to the CSV file (default: ../data/data.csv)')
    parser.add_argument('--headers', action='store_true', help='The dataset has headers. (default: %(default)s)')
    parser.add_argument('--no_plot', action='store_true', help='Should plot graphics (default: %(default)s)')
    args = parser.parse_args()

    # Load the dataset
    df = pd.read_csv(args.file, header="infer" if args.headers else None)

    # Display the first 5 rows of the dataset
    print(df.head())

    # Display the summary statistics of the dataset
    print(df.describe())

    # Plot a histogram for each numerical attribute
    df.hist(bins=50, figsize=(15,15))
    plt.savefig('plots/histogram.png', bbox_inches='tight')
    if args.no_plot is False:
        plt.tight_layout()
        plt.show()

    # Plot a correlation matrix
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_columns:
        df[col] = pd.factorize(df[col])[0]
    corr_matrix = df.corr()
    plt.figure(figsize=(15,15))
    sns.set_theme(font_scale=0.8)
    hm = sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.savefig('plots/corr_matrix.png', bbox_inches='tight')
    if args.no_plot is False:
        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    main()