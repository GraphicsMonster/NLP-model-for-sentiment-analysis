import pandas as pd
import nltk as nl

def preprocess_text(text):

    # Tokenize the text and turn into lowercase
    tokens = nl.word_tokenize(text)
    tokens = [token.lower() for token in tokens]

    # Remove stopwords
    stopwords = nl.corpus.stopwords.words('english')
    tokens = [token for token in tokens if token not in stopwords]

    # Remove punctuation
    tokens = [token for token in tokens if token.isalpha()]

    # Stemming
    lemmatizer = nl.stem.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]

    # Remove hashtags
    tokens = [token for token in tokens if token.startswith('#') == False]

    # Remove mentions
    tokens = [token for token in tokens if token.startswith('@') == False]

    # Remove links
    tokens = [token for token in tokens if token.startswith('http://') == False]

    # Remove empty tokens
    tokens = [token for token in tokens if token != '']

    return tokens

def preprocess_data(df):

    text = df['Text'].tolist()
    labels = df['Label'].tolist()

    # Preprocess text
    text = [preprocess_text(t) for t in text]
    
    return text, labels

path = './Dataset/dataset.csv'
df = pd.read_csv(path)
text, labels = preprocess_data(df)
print(text[0:5])
print(labels[0:5])






