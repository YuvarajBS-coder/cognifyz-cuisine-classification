# Project Report: Cuisine Classification

## 1. Introduction
Cuisine classification is a multiclass machine-learning problem in which a model estimates the primary cuisine associated with a restaurant from available restaurant attributes. This project was prepared for Cognifyz Technologies Machine Learning Task 3.

## 2. Objective
To preprocess a restaurant dataset, train a classification algorithm, evaluate it on unseen test data, and provide a simple interface for testing predictions.

## 3. Dataset and target
The supplied restaurant dataset contains fields related to restaurant identity, location, cost, booking/delivery options, ratings, votes and cuisine. Rows with missing cuisine labels are removed. As some restaurants list several cuisines, the first cuisine in the comma-separated field is treated as the primary label. To keep the task manageable, the 15 most frequent primary labels are retained.

## 4. Preprocessing
- Remove rows with missing target labels.
- Split cuisine strings and select the first label.
- Retain the 15 most frequent target classes.
- Use median imputation and standard scaling for numerical features.
- Use most-frequent imputation and one-hot encoding for categorical features.
- Use a stratified 80/20 training/testing split.

## 5. Algorithm
A Random Forest classifier with 180 trees is used for multiclass classification. Preprocessing and the classifier are combined in a scikit-learn Pipeline so that the same transformations are applied during training and prediction.

## 6. Evaluation
The training script calculates accuracy, precision, recall and F1-score and saves them in `metrics.json`. A confusion matrix is saved to `graphs/confusion_matrix.png`. The exact score should be taken from the generated metrics file after running `python train_model.py`.

## 7. Application
The Streamlit interface accepts city, currency, cost, price range, rating, votes and booking/delivery information. It returns a predicted primary cuisine and the model's top five class probability estimates.

## 8. Limitations and future work
The input features do not directly describe ingredients or menu items, so cuisine predictions may be uncertain. The dataset has class imbalance and some missing or inconsistent values. Future work could include menu text, more representative features, cross-validation, hyperparameter tuning and comparison with Random Forest or gradient boosting.

## 9. Conclusion
This project demonstrates the end-to-end workflow for a multiclass classification task: preprocessing, model training, evaluation, serialization and a small interactive application. Its performance should be interpreted in light of the available features and dataset limitations.

## 10. References
- Cognifyz Technologies, Machine Learning Task 3: Cuisine Classification (task brief supplied by the student).
- scikit-learn documentation: https://scikit-learn.org/
- Streamlit documentation: https://docs.streamlit.io/
- pandas documentation: https://pandas.pydata.org/docs/
