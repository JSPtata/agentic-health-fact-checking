import pandas as pd

def load_dataset(file_path):
    """
    Loads a CSV dataset.
    """
    data = pd.read_csv(file_path)

    print("Dataset loaded successfully")
    print("Shape:", data.shape)
    print("Columns:", data.columns.tolist())

    return data


if __name__ == "__main__":
    file_path = "data/sample.csv"

    df = load_dataset(file_path)

    print(df.head())
