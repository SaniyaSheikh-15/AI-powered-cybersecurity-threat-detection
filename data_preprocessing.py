import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(path):
    return pd.read_csv(path)

def preprocess_data(df):
    df = df.copy()
    df = df.drop(columns=[col for col in ['Timestamp'] if col in df.columns], errors='ignore')

    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    return df, label_encoders

def split_data(df):
    X = df.drop('Label', axis=1)
    y = df['Label']
    return X, y
