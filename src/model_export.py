
import os
import joblib


def save_model(model, file_path="artifacts/best_model.pkl"):

    folder = os.path.dirname(file_path)

    if folder:
        os.makedirs(folder, exist_ok=True)

    joblib.dump(model, file_path)

    return file_path


def load_model(file_path="artifacts/best_model.pkl"):

    if not os.path.exists(file_path):
        raise FileNotFoundError("Model file does not exist")

    model = joblib.load(file_path)

    return model
