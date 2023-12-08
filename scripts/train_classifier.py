import tensorflow_hub as hub
from sklearn.svm import SVC
import numpy as np
import pandas as pd
# Load Universal Sentence Encoder
embed = hub.load("/Users/rupampatir/Desktop/Conv AI Project/ubchatbot/llm/UniversalSentenceEncoder")

# Example training data (intents and sentences)
df = pd.read_csv('training_data.csv', header=None)
df.columns = ['text', 'label']

# Split data into features and labels
training_sentences = df['text']
intents = df['label']

# Convert sentences to embeddings
training_embeddings = embed(training_sentences)

# Train a classifier
clf = SVC().fit(training_embeddings, intents)

# Example user input
user_input = "Can you tell about Hongxin Hu?"
user_embedding = embed([user_input])

# Predict intent
predicted_intent = clf.predict(user_embedding)[0]
print(predicted_intent)






# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import SVC
# from sklearn.pipeline import make_pipeline
# from sklearn.metrics import classification_report
# import spacy

# # Load the spaCy model for text preprocessing
# nlp = spacy.load('en_core_web_md')

# # Read CSV data
# df = pd.read_csv('path_to_your_csv_file.csv', header=None)
# df.columns = ['text', 'label']

# # Split data into features and labels
# X = df['text']
# y = df['label']

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Create a text classification pipeline
# pipeline = make_pipeline(TfidfVectorizer(), SVC())
# pipeline.fit(X_train, y_train)

# # Evaluate the model
# predictions = pipeline.predict(X_test)
# print(classification_report(y_test, predictions))

# import joblib

# # Save the model
# joblib.dump(pipeline, 'text_classifier_model.pkl')

# # To load and use the model
# model = joblib.load('text_classifier_model.pkl')
# new_question = "What are the prerequisites for the Advanced Mathematics course?"
# print(model.predict([new_question]))