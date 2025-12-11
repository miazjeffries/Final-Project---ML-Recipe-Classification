""" This script a DistilBERT model to classify recipes """

import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from datasets import Dataset
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
os.makedirs("Results", exist_ok=True)

''' LOAD AND PREP PROCESSED DATA '''
# Load processed dataset
data = pd.read_csv('../data/processed_data.csv')

# Combine ingredients and instructions into a single feature
data['Recipe'] = data['Ingredients'] + " " + data['Instructions']
data = data.drop(axis=1, columns=['Ingredients', 'Instructions'])

# Take care of potential NaNs
data['Recipe'] = data['Recipe'].fillna("").astype(str)
data = data[data['Recipe'].str.strip() != ""]

# Check data and class balances
print(data.head())
print(data['Recipe Category'].value_counts())

# Merge 'Soups' with 'Main Dishes' to adjust for underrepresentation
data['Recipe Category'] = data['Recipe Category'].replace({"Soups": "Main Dish"})

# Merge 'Other' with 'Snacks' (mostly consists of small dishes, dips, etc.) to adjust training classes
data["Recipe Category"] = data["Recipe Category"].replace({"Other": "Snacks"})


''' PREP FOR DISTILBERT '''
# LABEL ENCODING
# str --> numerical ids
print("\nEncoding labels...")

label_encoder = LabelEncoder()
data['Labels'] = label_encoder.fit_transform(data['Recipe Category'])
num_labels = len(label_encoder.classes_)
print("\nNumber of classes: ", num_labels)

# Save label mapping
label_map = pd.DataFrame({
    "label_id": range(num_labels),
    "label_name": label_encoder.classes_
})
label_map.to_csv("../results/distilbert_label_map.csv", index=False)


# train / val / test splits- 70% train, 10% val, 20% test

# first split- train_full (80%), test (20%)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    data['Recipe'],
    data['Labels'],
    test_size=0.2,
    random_state=42,
    stratify=data['Labels']
)

# second split- train (70%), val (10%) from 80% train_full
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full,
    y_train_full,
    test_size=0.125,
    random_state=42,
    stratify=y_train_full
)

# df for datasets
train = pd.DataFrame({"text": X_train, "label": y_train})
val   = pd.DataFrame({"text": X_val, "label": y_val})
test  = pd.DataFrame({"text": X_test, "label": y_test})

train_dataset = Dataset.from_pandas(train)
val_dataset   = Dataset.from_pandas(val)
test_dataset  = Dataset.from_pandas(test)


# Tokenization
print("\nTokenizing with DistilBERT...")
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(batch):
    return tokenizer(batch['text'], padding='max_length',truncation=True, max_length=256)


train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset   = val_dataset.map(tokenize, batched=True)
test_dataset  = test_dataset.map(tokenize, batched=True)

train_dataset.set_format('torch', columns=["input_ids", "attention_mask", "label"])
val_dataset.set_format('torch',   columns=["input_ids", "attention_mask", "label"])
test_dataset.set_format('torch',  columns=["input_ids", "attention_mask", "label"])


''' RUN DISTILBERT MODEL '''
# Model setup
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)

training_args = TrainingArguments(
    output_dir='../results/distilbert_results',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01
)

# Metrics output
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {"accuracy": accuracy_score(labels, preds)}

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics
)

# Train model

print("\nTraining model...")
trainer.train()

# Evaluate on val
print("\nEvaluating on validation set...")
val_predictions = trainer.predict(val_dataset)
val_pred = np.argmax(val_predictions.predictions, axis=1)
val_accuracy = accuracy_score(y_val, val_pred)
val_report = classification_report(
    y_val,
    val_pred,
    target_names=label_encoder.classes_,
    output_dict=True
)
print("Validation accuracy:", val_accuracy)
pd.DataFrame({"val_accuracy": [val_accuracy]}).to_csv(
    "../results/distilbert_val_accuracy.csv",
    index=False
)
pd.DataFrame(val_report).transpose().to_csv(
    "../results/distilbert_val_classification_report.csv"
)

# Evaluate on test set
print("\nEvaluating on test set...")
predictions = trainer.predict(test_dataset)
y_pred = np.argmax(predictions.predictions, axis=1)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(
    y_test,
    y_pred,
    target_names=label_encoder.classes_,
    output_dict=True
)

# Display and save final accuracy score
print("\nAccuracy: ", accuracy)
pd.DataFrame({"accuracy": [accuracy]}).to_csv(
    "../results/distilbert_accuracy.csv",
    index=False
)

# Display and save final classification report
print("\nClassification Report:  ", report)
pd.DataFrame(report).transpose().to_csv(
    "../results/distilbert_classification_report.csv"
)

# Display and save predictions
pred_data = pd.DataFrame({
    "true_label": y_test.values,
    "predicted_label": y_pred
})

pred_data["true_label_name"] = label_encoder.inverse_transform(pred_data["true_label"])
pred_data["predicted_label_name"] = label_encoder.inverse_transform(pred_data["predicted_label"])

pred_data.to_csv("../results/distilbert_test_predictions.csv", index=False)

print(pred_data.head())

# Save trained model
model.save_pretrained("distilbert_recipe_model")
tokenizer.save_pretrained("distilbert_recipe_model")

print("\nReports and model successfully saved.")