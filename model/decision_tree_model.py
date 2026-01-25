import joblib

def load_model():
    model = joblib.load("saved_models/decision_tree.pkl")
    return model
