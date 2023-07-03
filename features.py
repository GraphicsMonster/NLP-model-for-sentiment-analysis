from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import pandas as pd
from preprocess import preprocess_data

def get_features(text):
    vectorizer = CountVectorizer()
    features = vectorizer.fit_transform(text)
    return features, vectorizer.get_feature_names_out()


path = 'Dataset/dataset.csv'
df = pd.read_csv(path)
df = df[:10]

text, labels = preprocess_data(df)

X, feature_names = get_features(text)

print(X.shape)
print(feature_names.shape)
