import joblib

def load_model():
    model = joblib.load("saved_models/xgboost.pkl")
    return model
