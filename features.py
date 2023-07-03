from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
from preprocess import preprocess_data

def get_features(text):
    vectorizer = CountVectorizer(ngram_range=(1,2))
    features = vectorizer.fit_transform(text)
    return features, vectorizer.get_feature_names_out()

# basically this creates a vocabulary of all the words in the text and then counts the number of times each word appears in each text
# the output is a matrix of size (number of texts, number of words in vocabulary) and [i,j] entry is the number of times word j appears in text i


path = 'Dataset/dataset.csv'
df = pd.read_csv(path)
df = df[:10]

text, labels = preprocess_data(df)

X, feature_names = get_features(text)

print(X.shape)
print("features: ", feature_names)
