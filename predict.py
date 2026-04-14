import joblib
import pandas as pd

def load_model():
    return joblib.load("models/threat_model.pkl")

def predict(input_data):
    model = load_model()
    df = pd.DataFrame([input_data])
    pred = model.predict(df)[0]
    prob = model.predict_proba(df).max()
    return pred, prob
