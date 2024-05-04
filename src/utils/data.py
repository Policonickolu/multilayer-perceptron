import sys
import pandas as pd
import numpy as np

def load_and_clean_data(filename: str, label_column: int, headers: bool, disabled_columns: list[int], outliers: str) -> tuple[np.ndarray, np.ndarray]:
    data = load(filename, headers)

    # Remove duplicate rows before removing columns
    data.drop_duplicates(inplace=True)

    # Remove and extract labels
    labels = data.iloc[:, [label_column]]
    remaining_indices = set(range(data.shape[1])) - set([label_column]) - set(disabled_columns)
    data = data.iloc[:, list(remaining_indices)]

    # Data Cleaning: Handle missing values, outliers
    data = clean(data, outliers)
    labels = clean(labels, outliers)

    # Data Encoding: Convert categorical data to numerical data.
    data = pd.get_dummies(data).astype(int)
    label_index_to_classes = {index: label for index, label in enumerate(labels.iloc[:,0].unique())}
    label = pd.get_dummies(labels).astype(int)

    # Feature Scaling: Standardize with Z-score normalization (best for gaussian distribution)
    data = scale_features(data)

    # Convert boolean values to integers (True/False to 1/0)
    boolean_cols = data.select_dtypes(include=['bool']).columns
    data[boolean_cols] = data[boolean_cols].astype(int)

    return data.values, label.values, label_index_to_classes

def load(filename: str, headers=False) -> pd.DataFrame:
    try:
        if headers:
            data = pd.read_csv(filename)
        else:
            data = pd.read_csv(filename, header=None)
        return data
    except FileNotFoundError:
        print(f"File not found: {filename}")
        sys.exit(1)
    except pd.errors.ParserError:
        print(f"Invalid file format: {filename}")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

def clean(data: pd.DataFrame, outliers: str) -> pd.DataFrame:

    # Select columns : numerical and non-numerical
    non_numerical_cols = data.select_dtypes(include=['object', 'category', 'bool']).columns
    numerical_cols = data.select_dtypes(include=[np.number]).columns

    # Handle outliers values
    if outliers in ["remove", "clamp"]:
        for col in numerical_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            if outliers == "remove":
                data = data[~((data[col] < lower_bound) | (data[col] > upper_bound))]
            elif outliers == "clamp":
                data[col] = np.where(data[col] < lower_bound, lower_bound, data[col])
                data[col] = np.where(data[col] > upper_bound, upper_bound, data[col])

    # For non-numerical columns, fill NaNs with the mode (most frequent value)
    data[non_numerical_cols] = data[non_numerical_cols].apply(lambda x: x.fillna(x.mode()[0]))
    # For numerical columns, fill NaNs with the mean
    data[numerical_cols] = data[numerical_cols].apply(lambda x: x.fillna(x.mean()))

    # For other types, delete rows with NaNs
    data.dropna(inplace=True)

    return data

def scale_features(data: pd.DataFrame) -> pd.DataFrame:
    # Standardize with Z-score normalization (best for gaussian distribution)
    numerical_cols = data.select_dtypes(include=[np.number]).columns
    means = data[numerical_cols].mean()
    stds = data[numerical_cols].std()
    stds.replace(0, 1, inplace=True)
    data[numerical_cols] = (data[numerical_cols] - means) / stds
    return data