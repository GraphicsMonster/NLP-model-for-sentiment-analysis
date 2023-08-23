import pandas as pd
import nltk as nl
from nltk.tokenize import TweetTokenizer

def preprocess_text(text):

    # Initialize tokenizer
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)

    # Tokenize the text and turn into lowercase
    tokens = tokenizer.tokenize(text)
    tokens = [token.lower() for token in tokens]

    # Remove stopwords
    stopwords = set(nl.corpus.stopwords.words('english'))
    tokens = [token for token in tokens if token not in stopwords]

    # Remove punctuation and non-alphabetic characters
    tokens = [token for token in tokens if token.isalpha()]

    # Lemmatization
    lemmatizer = nl.stem.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Remove links
    tokens = [token for token in tokens if not token.startswith('http')]

    # Remove tokens with hashtags
    tokens = [token for token in tokens if not token.startswith('#')]

    # Join tokens into a string
    processed_text = ' '.join(tokens)

    return processed_text

def preprocess_labels(labels):

    #convert labels to integral values
    label_map = {
        'positive': 1,
        'negative': 2,
        'uncertainty': 0,
        'litigious': 3
    }

    # mapping labels to the label map
    processed_labels = [label_map[label] for label in labels]

    return processed_labels


def preprocess_data(df):

    text = df[df['Language'] == 'en']['Text'].tolist()
    labels = df[df['Language'] == 'en']['Label'].tolist()

    # Preprocess text
    text = [preprocess_text(t) for t in text]
    labels = preprocess_labels(labels)
    
    return text, labels







