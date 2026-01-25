import joblib

def load_model():
    model = joblib.load("saved_models/knn.pkl")
    return model
