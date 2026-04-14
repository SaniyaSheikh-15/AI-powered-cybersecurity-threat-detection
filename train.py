import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from src.data_preprocessing import load_data, preprocess_data, split_data
from src.model import build_model

def train_model(data_path):
    df = load_data(data_path)
    df, _ = preprocess_data(df)
    X, y = split_data(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_model()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))

    joblib.dump(model, "models/threat_model.pkl")
    return model
