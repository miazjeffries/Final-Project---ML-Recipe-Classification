""" This script fita a Random Forest Model as a baseline """

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


''' CREATE TRAIN/TEST SPLIT '''
# Load processed dataset
data = pd.read_csv('../data/processed_data.csv')

# Check data and class balances
print(data.head())
print(data['Recipe Category'].value_counts())

# Combine ingredients and instructions into a single feature
data['Recipe'] = data['Ingredients'] + " " + data['Instructions']
data = data.drop(axis=1, columns=['Ingredients', 'Instructions'])

# Take care of potential NaNs
data['Recipe'] = data['Recipe'].fillna("").astype(str)
data = data[data['Recipe'].str.strip() != ""]

# Set independent and dependent categories
X = data['Recipe']
y = data['Recipe Category']

# Create the train/test data split
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y
)

# Convert text to TF-IDF vectors for ML processing
print("\nVectorizing data...")

vectorizer = TfidfVectorizer(
    max_features=5000, 
    ngram_range=(1, 2),
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


''' FIT RANDOM FOREST AS A BASELINE MODEL '''
# Random forest
random_forest = RandomForestClassifier(
    random_state=42,
    n_estimators=150,
    max_depth=None
)

print("\nFitting Random Forest...")
random_forest.fit(X_train_tfidf, y_train)

# Evaluate random forest
y_pred = random_forest.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred) # Check overall performance
print('Accuracy: ', accuracy)

print("\nClassification Report: \n")
print(classification_report(y_test, y_pred)) # Check per class performance metrics

# Save results to CSV files for future reference
accuracy_copy = pd.DataFrame({
    "metric": ["accuracy"],
    "value": [accuracy]
})
accuracy_copy.to_csv("../results/baseline_RF_accuracy.csv", index=False)

report_dict = classification_report(y_test, y_pred, output_dict=True)
report_copy = pd.DataFrame(report_dict).transpose()
report_copy.to_csv("../results/baseline_RF_classification_report.csv")

results_copy = pd.DataFrame({
    "true_label": y_test.values,
    "predicted_label": y_pred
})
results_copy.to_csv("../results/baseline_RF_test_predictions.csv", index=False)

print("\nRandom Forest metrics saved successfully.")
