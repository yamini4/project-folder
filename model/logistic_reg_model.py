import joblib

def load_model():
    model = joblib.load("saved_models/logistic_regression.pkl")
    return model
