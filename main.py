from src.train import train_model
from src.predict import predict
from src.utils import save_predictions

if __name__ == "__main__":
    model = train_model("data/sample_data.csv")

    sample_input = {
        "Feature1": 10,
        "Feature2": 5,
        "Feature3": 2
    }

    pred, prob = predict(sample_input)

    result = {"prediction": pred, "confidence": prob}
    print(result)

    save_predictions([result])
