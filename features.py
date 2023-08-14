from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
from preprocess import preprocess_data

def get_features(text):

    # initialize the vectorizer with ngram
    vectorizer = CountVectorizer(ngram_range=(1,2)) 

    # fit the vectorizer on the text
    features = vectorizer.fit_transform(text)
    # get the vocabulary. This also includes phrases of upto 2 words thanks to ngram_range
    vocabulary = vectorizer.get_feature_names_out()
    # Apply normalization
    normalized_features = normalize(features.toarray())
    return normalized_features, vocabulary

def normalize(X):
    # Z-Score normalization
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = (X - mean) / std
    return X

# basically this creates a vocabulary of all the words in the text and then counts the number of times each word appears in each text
# the output is a matrix of size (number of texts, number of words in vocabulary) and [i,j] entry is the number of times word j appears in text i
# Note: The features matrix must be reshaped to a tensor of size (1, batch_size, vocab_size) in order to be used for further processing
