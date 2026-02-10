import pickle
import os

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ------------------------------------------------------------------
# 1. Create model directory if not exists
# ------------------------------------------------------------------
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------------------------------------------------------
# 2. Load / prepare your dataset
# ------------------------------------------------------------------
# Example placeholders — REPLACE with your actual data
# X: shape (n_samples, n_features)
# y: shape (n_samples,)
from sklearn.datasets import load_iris
X, y = load_iris(return_X_y=True)

# ------------------------------------------------------------------
# 3. Initialize models
# ------------------------------------------------------------------
models = {
    "logistic_reg_model": LogisticRegression(max_iter=1000),
    "decision_tree_model": DecisionTreeClassifier(),
    "KNN_model": KNeighborsClassifier(n_neighbors=5),
    "naive_bayes_model": GaussianNB(),
    "random_forest_model": RandomForestClassifier(n_estimators=100),
    "XGBoost_model": XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss"
    )
}

# ------------------------------------------------------------------
# 4. Train and save models
# ------------------------------------------------------------------
for filename, model in models.items():
    model.fit(X, y)

    file_path = os.path.join(MODEL_DIR, filename)

    with open(file_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Saved: {file_path}")

print("All models trained and saved successfully.")
