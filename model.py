import numpy as np
import pandas as pd
from preprocess import preprocess_data
from features import get_features
from convolution import Conv1DLayer
from pooling import PoolingLayer
from fullyconnectedlayer import FullyConnectedLayer
from classification import ClassificationLayer

class SentimentAnalysisModel:

    def __init__(self, num_filters, filter_size, pool_size, input_size, output_size, hidden_units, num_classes, learning_rate):

        self.num_filters = num_filters
        self.filter_size = filter_size
        self.pool_size = pool_size
        self.hidden_units = hidden_units
        self.num_classes = num_classes
        self.learning_rate = learning_rate

        # Initialize the layers
        self.conv_layer = Conv1DLayer(num_filters, filter_size)
        self.pool_layer = PoolingLayer(pool_size)
        self.fc_layer = FullyConnectedLayer(input_size, output_size, hidden_units)
        self.classification_layer = ClassificationLayer(input_size=output_size, num_classes=num_classes) #The number of input neurons in the classification layer are equal to the number of output neurons of the FC layer.

    def train(self, X, labels, num_epochs, batch_size):

        num_batches = X.shape[0] // batch_size
        total_loss = 0.0
        
        labels = self.one_hot_encode(labels, self.num_classes)

    # Training the model
        for epoch in range(1, num_epochs+1):
            total_loss = 0.0
            num_batches = X.shape[0] // batch_size

            for batch in range(num_batches):
                # Get the batch data
                batch_start = batch * batch_size
                batch_end = batch_start + batch_size
                inputs = X[batch_start:batch_end]
                print("inputs shape", inputs.shape)
                targets = labels[batch_start:batch_end]

                # Forward pass
                conv_output = self.conv_layer.forward(inputs)
                pool_output = self.pool_layer.forward(conv_output)
                fc_output = self.fc_layer.forward(pool_output)
                probs = self.classification_layer.forward(fc_output)
                
                print("final probs: ", probs)

                # Compute loss for each sample
                loss = self.classification_layer.loss(probs, targets)
                total_loss += loss
                print("total loss at epoch {}, batch {} : {}".format(epoch, batch, loss))

                # Backward pass
                grad_probs = self.classification_layer.backward(probs, targets, self.learning_rate)
                grad_fc = self.fc_layer.backward(grad_probs, self.learning_rate)
                grad_fc = grad_fc.reshape((pool_output.shape)) # reshape the gradient to match the input shape of the pooling layer
                grad_pool = self.pool_layer.backward(grad_fc)
                grad_conv = self.conv_layer.backward(grad_pool, self.learning_rate)

            # Print the loss every 10 epochs
            if epoch % 10 == 0:
                print("Epoch {}: loss = {}".format(epoch, total_loss))


    def predict(self, X):

        conv_output = self.conv_layer.forward(X)
        pool_output = self.pool_layer.forward(conv_output)
        fc_output = self.fc_layer.forward(pool_output)
        probs = self.classification_layer.forward(fc_output)
        # return the class with the highest probability
        return np.argmax(probs, axis=1)
    
    def one_hot_encode(self, labels, num_classes):
        
        one_hot_labels = np.zeros((len(labels), num_classes))

        for i in range(len(labels)):
            one_hot_labels[i, labels[i]] = 1

        return one_hot_labels

# Testing the model with a small dataset

path = './Dataset/dataset.csv'
df = pd.read_csv(path)
df = df[:200]

# Preprocess the data
X, labels = preprocess_data(df)

# Get the features
X, vocab = get_features(X)
print("vocab length: ", len(vocab))

# Convert labels to one-hot encoding
num_classes = 4
one_hot_labels = np.eye(num_classes)[labels].astype(float)

# Initialize the model
model = SentimentAnalysisModel(num_filters=10, filter_size=3, pool_size=2, input_size=10, output_size=4, hidden_units=32, num_classes=num_classes, learning_rate=0.1)

# Train the model
model.train(X, labels, num_epochs=50, batch_size=32)

# Test the model
df = pd.read_csv(path)
df = df[300: 350]
X, labels = preprocess_data(df)
X, vocab = get_features(X)

preds = model.predict(X)
print(preds.shape)
print(len(labels))
print("Accuracy = {}".format(np.mean(preds == np.argmax(one_hot_labels, axis=1))))