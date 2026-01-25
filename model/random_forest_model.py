import joblib

def load_model():
    model = joblib.load("saved_models/random_forest.pkl")
    return model
