import pandas as pd

def save_metrics(filename, metrics):
    df = pd.DataFrame(metrics)
    df = df.round(4)
    df['confusion_matrix'] = df['confusion_matrix'].apply(lambda arr: str(arr.tolist()).replace(' ', ''))
    df.to_csv(filename, index=False)