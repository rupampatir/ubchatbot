import os
import tensorflow_hub as hub

# Set the TFHUB_CACHE_DIR
os.environ['TFHUB_CACHE_DIR'] = '/Users/rupampatir/Desktop/Conv AI Project/ubchatbot/data/'

# Now, when you load a model, it will be downloaded to the specified directory
model = hub.load("https://www.kaggle.com/models/google/universal-sentence-encoder/frameworks/TensorFlow2/variations/universal-sentence-encoder/versions/2")

# https://colab.research.google.com/drive/168w53YDlPdqM9unchSSzQYOYQbtGdO-d?authuser=0#scrollTo=ocZY3UUvLXX1