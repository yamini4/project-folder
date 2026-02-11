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
    | Logistic Regression | 0.8338 | 0.9174 | 0.8207 | 0.8338 | 0.8249 | 0.5123 |
    | Decision Tree | 0.9366 | 0.9559 | 0.9353 | 0.9366 | 0.9352 | 0.8226 |
    | KNN | 0.8731 | 0.9062 | 0.8661 | 0.8731 | 0.8649 | 0.6253 |
    | Naive Bayes | 0.7825 | 0.9004 | 0.8514 | 0.7825 | 0.8005 | 0.5480 |
    | Random Forest (Ensemble) | 0.9456 | 0.9855 | 0.9445 | 0.9456 | 0.9446 | 0.8483 |
    | **XGBoost (Ensemble)** | **0.9637** | **0.9884** | **0.9637** | **0.9637** | **0.9636** | **0.9002** |


- Observations on the performance of each model on the Adult Income dataset:

    | **ML Model Name**            | **Observation about Model Performance**                                          |
    | ---------------------------- | -------------------------------------------------------------------------------- |
    | **Logistic Regression**      | Performs well on linear data but gives lower accuracy for complex income patterns. |
    | **Decision Tree**            | Fits the training data well but may overfit on income prediction tasks.           |
    | **kNN**                      | Works well for classification but is sensitive to feature scaling.                |
    | **Naive Bayes**              | Fast and simple but less accurate when demographic features are correlated.       |
    | **Random Forest (Ensemble)** | Provides higher accuracy and better generalization for income prediction.         |
    | **XGBoost (Ensemble)**       | Delivers the best performance for binary income classification (>50K vs <=50K).   |
