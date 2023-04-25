import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import Pipeline
from nltk.tokenize import word_tokenize
from nltk import download as nltk_download
from pyarrow import feather

nltk_download('punkt')

# Load your dataset here
df = feather.read_feather("data/training_data.feather")

# Define a custom tokenizer function using NLTK word_tokenize
def tokenize(text):
    return word_tokenize(text)

# Split the data into training and test sets
train_df = df[df['is_train'] == True]
test_df = df[df['is_train'] == False]

print(f"Training set size: {len(train_df)}")
print(f"Test set size: {len(test_df)}")

# Create a pipeline with the TfidfVectorizer and the MultinomialNB classifier
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
    ('classifier', MultinomialNB())
])

print("Training model...")
# Train the model
pipeline.fit(train_df['title'], train_df['value'])

print("Evaluating model...")
# Predict on the test set
predictions = pipeline.predict(test_df['title'])

# Evaluate the model
print("Accuracy:", accuracy_score(test_df['value'], predictions))
print("F1 Score:", f1_score(test_df['value'], predictions))
print("Confusion Matrix:\n", confusion_matrix(test_df['value'], predictions))
print("\nClassification Report:\n", classification_report(test_df['value'], predictions))