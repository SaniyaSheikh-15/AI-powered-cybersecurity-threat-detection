import pandas as pd

def save_predictions(data, path="outputs/predictions.csv"):
    pd.DataFrame(data).to_csv(path, index=False)
