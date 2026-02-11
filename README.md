a. Problem statement:
Ans: The dataset contains demographic and employment parameters that predict whether an individual's annual income is above or below $50K.
     Based on the set of features provided, we have to predict the income level (<=50K or >50K) of new individuals when input features are given.

b. Dataset description
Ans: Input features: 14
     Features: ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country']
     Output feature: income (Binary classification: <=50K or >50K)


c. Models used: 6 Classification Models – Comparison Table
Ans:
    ## 📊 Model Performance Comparison (Adult Income Dataset)

    | ML Model | Accuracy | ROC AUC | Precision | Recall | F1 Score | MCC |
    |---------|---------|---------|-----------|--------|---------|------|
    | Logistic Regression | 0.8054 | 0.8141 | 0.6526 | 0.3770 | 0.4779 | 0.3897 |
    | Decision Tree | 0.8099 | 0.7401 | 0.5957 | 0.6079 | 0.6017 | 0.4769 |
    | KNN | 0.7741 | 0.6555 | 0.5381 | 0.3086 | 0.3923 | 0.2813 |
    | Naive Bayes | 0.7953 | 0.8279 | 0.6401 | 0.3053 | 0.4134 | 0.3388 |
    | Random Forest (Ensemble) | 0.9725 | 0.9897 | 0.9565 | 0.9279 | 0.9420 | 0.9241 |
    | **XGBoost (Ensemble)** | **0.9010** | **0.9558** | **0.8378** | **0.7305** | **0.7805** | **0.7198** |


- Observations on the performance of each model on the Adult Income dataset:

    | **ML Model Name**            | **Observation about Model Performance**                                          |
    | ---------------------------- | -------------------------------------------------------------------------------- |
    | **Logistic Regression**      | Performs well on linear data but gives lower accuracy for complex income patterns. |
    | **Decision Tree**            | Fits the training data well but may overfit on income prediction tasks.           |
    | **kNN**                      | Works well for classification but is sensitive to feature scaling.                |
    | **Naive Bayes**              | Fast and simple but less accurate when demographic features are correlated.       |
    | **Random Forest (Ensemble)** | Provides higher accuracy and better generalization for income prediction.         |
    | **XGBoost (Ensemble)**       | Delivers the best performance for binary income classification (>50K vs <=50K).   |
