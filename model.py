import numpy as np
from convolution import Conv1DLayer
from pooling import PoolingLayer
from fullyconnectedlayer import FullyConnectedLayer
from classification import ClassificationLayer

class SentimentAnalysisModel:

    def __init__(self, num_inputs, num_outputs, num_filters, filter_size, pool_size, num_hidden):
        self.conv_layer = Conv1DLayer(num_filters, filter_size)
        self.pool_layer = PoolingLayer(pool_size)
        self.fc_layer = FullyConnectedLayer(num_filters, num_hidden)
        self.classification_layer = ClassificationLayer(num_hidden, num_outputs)

    def forward(self, inputs):
        conv_output = self.conv_layer.forward(inputs)
        pool_output = self.pool_layer.forward(conv_output)
        fc_output = self.fc_layer.forward(pool_output)
        output = self.classification_layer.forward(fc_output)
        return output
    
    def backward(self, grad_outputs, learning_rate):
        grad_outputs = self.classification_layer.backward(grad_outputs)
        grad_outputs = self.fc_layer.backward(grad_outputs, learning_rate)
        grad_outputs = self.pool_layer.backward(grad_outputs)
        grad_outputs = self.conv_layer.backward(grad_outputs, learning_rate)
        return grad_outputs
    