import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from preprocess import preprocess_data
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from features import get_features
from torch.utils.tensorboard import SummaryWriter
from sklearn.utils import shuffle

class SentimentAnalysisModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=hidden_dim, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc = nn.Linear(hidden_dim * 23, output_dim)
        self.writer = SummaryWriter()

    def forward(self, text):
        # text = torch.tensor(text, dtype=torch.long)
        embedded = self.embedding(text)
        embedded = embedded.permute(0, 2, 1)
        conv1_out = self.conv1(embedded)
        conv2_out = self.conv2(conv1_out)
        pooled = self.pool(conv2_out)
        pooled = pooled.permute(0, 2, 1)
        pooled = pooled.reshape(pooled.shape[0], -1)
        dense = self.fc(pooled)
        probs = nn.Softmax(dim=1)(dense)
        return probs
    
    def train(self, X, labels, num_epochs, batch_size, learning_rate=0.001):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        labels = torch.tensor(labels, dtype=torch.long)
        for epoch in range(num_epochs):
            for batch in range(0, len(X), batch_size):
                optimizer.zero_grad()
                inputs = torch.tensor(X[batch:batch+batch_size], dtype=torch.long)
                targets = labels[batch:batch+batch_size]
                outputs = self.predict(inputs)
                outputs = torch.tensor(outputs, dtype=torch.float64)
                targets = torch.tensor(targets, dtype=torch.float64, requires_grad=True)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                if(batch % 10 == 0):
                    print("epoch: ", epoch, " loss: ", loss.item(), " batch: ", batch, " out of: ", len(X), " batches")

                # log loss to tensorboard
                global_step = epoch * len(X) + batch
                self.writer.add_scalar('Loss/train', loss.item(), global_step)

    def predict(self, X):
        outputs = self.forward(X)
        new_outputs = []
        for i in range(outputs.shape[0]):
            new_outputs.append(torch.argmax(outputs[i]))
        return new_outputs
    
    def evaluate(self, X, labels):
        predicted = self.predict(X)
        predicted = np.array(predicted)
        print("predicted: ", predicted)
        print("labels: ", labels)
        
        # Accuracy calculation logic
        correct = 0
        for i in range(len(predicted)):
            if predicted[i] == labels[i]:
                correct += 1

        accuracy = correct / len(labels)
        return accuracy

    def save(self, path):
        torch.save(self.state_dict(), path)

df = pd.read_csv("Dataset/dataset.csv")
df = df[:15000]

text = df[df['Language'] == 'en']['Text'].tolist()
labels = df[df['Language'] == 'en']['Label'].tolist()

# preprocessed data
text_test, labels_test = preprocess_data(df)


# 'text' is the list of preprocessed text and 'vocab_size' is the size of your vocabulary
tokenizer = Tokenizer()
tokenizer.fit_on_texts(text)
sequences = tokenizer.texts_to_sequences(text)
padded_sequences = pad_sequences(sequences, maxlen=50, padding='post', truncating='post')
print("padded_sequences.shape: ", padded_sequences.shape)

# Convert labels to class indices
label_mapping = {label: index for index, label in enumerate(sorted(set(labels)))}
labels = [label_mapping[label] for label in labels]

# Shuffle the data
padded_sequences, labels = shuffle(padded_sequences, labels)

# convert labels to one-hot encoding
num_classes = 4
one_hot_labels = torch.eye(num_classes)[labels].type(torch.LongTensor)

# hyperparameters
vocab_size = len(tokenizer.word_index) + 1
embedding_dim = 128
hidden_dim = 128
output_dim = num_classes

# initialize the model
model = SentimentAnalysisModel(vocab_size, embedding_dim, hidden_dim, output_dim)

# train the model
model.train(padded_sequences, labels, num_epochs=100, batch_size=64, learning_rate=1e-3)

# evaluate the model
X_test = pd.read_csv("Dataset/dataset.csv")
X_test = X_test[25000 : 30000]
X_test, labels_test = preprocess_data(X_test)
# labels_test = torch.eye(num_classes)[labels_test].type(torch.LongTensor)
sequences = tokenizer.texts_to_sequences(X_test)
padded_sequences = pad_sequences(sequences, maxlen=50, padding='post', truncating='post')
padded_sequences = torch.tensor(padded_sequences, dtype=torch.long)
labels_test = torch.tensor(labels_test, dtype=torch.long)
accuracy = model.evaluate(padded_sequences, labels_test)
print("Accuracy: ", accuracy)