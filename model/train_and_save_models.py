import pickle
import os
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
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
# 2. Load Dataset
# ------------------------------------------------------------------
# Save your data as adult.csv and place in same folder
df = pd.read_csv("adult_train.csv")

# ------------------------------------------------------------------
# 3. Preprocessing
# ------------------------------------------------------------------

# Remove spaces from column names
df.columns = df.columns.str.strip()

# Encode target variable (income)
df["income"] = df["income"].str.strip()
df["income"] = LabelEncoder().fit_transform(df["income"])

# Encode all categorical columns
categorical_cols = df.select_dtypes(include="object").columns

for col in categorical_cols:
    df[col] = df[col].str.strip()
    df[col] = LabelEncoder().fit_transform(df[col])

# Split features and target
X = df.drop("income", axis=1)
y = df["income"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------------------------------------------
# 4. Initialize models
# ------------------------------------------------------------------
models = {
    "logistic_reg_model.pkl": LogisticRegression(max_iter=1000),
    "decision_tree_model.pkl": DecisionTreeClassifier(),
    "KNN_model.pkl": KNeighborsClassifier(n_neighbors=5),
    "naive_bayes_model.pkl": GaussianNB(),
    "random_forest_model.pkl": RandomForestClassifier(n_estimators=100),
    "XGBoost_model.pkl": XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss"
    )
}

# ------------------------------------------------------------------
# 5. Train and save models
# ------------------------------------------------------------------
for filename, model in models.items():
    model.fit(X_train, y_train)

    file_path = os.path.join(MODEL_DIR, filename)

    with open(file_path, "wb") as f:
        pickle.dump(model, f)

    print(f"Saved: {file_path}")

print("All models trained and saved successfully.")
