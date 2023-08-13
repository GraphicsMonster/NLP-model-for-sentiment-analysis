import numpy as np

class FullyConnectedLayer:

    def __init__(self, input_size, output_size, hidden_units=None):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_units = hidden_units
        self.weights = None
        self.biases = None
        self.hidden_weights = None
        self.hidden_biases = None
        self.inputs = None

    def initialize_parameters(self, inputs):
        
        self.hidden_weights = np.random.randn(inputs.shape[1], self.hidden_units)
        self.hidden_biases = np.random.randn(self.hidden_units)

    def forward(self, inputs): 
        
        inputs = inputs.reshape(inputs.shape[0], -1)  # Flatten input
        self.inputs = inputs
        # print("input shape during FC forward: ", inputs.shape)

        if self.hidden_weights is None and self.hidden_biases is None:
            self.initialize_parameters(inputs)

        hidden_outputs = np.dot(inputs, self.hidden_weights) + self.hidden_biases
        hidden_outputs[hidden_outputs <= 0] = 0  # ReLU forward for the hidden layer

        # print("FC layer outputs shape: ", hidden_outputs.shape)
        return hidden_outputs

    def backward(self, grad_outputs, learning_rate):

        batch_size = grad_outputs.shape[0]

        # Compute gradients of the hidden weights and biases
        grad_hidden_weights = np.dot(self.inputs.T, grad_outputs) / batch_size
        grad_hidden_biases = np.sum(grad_outputs, axis=0) / batch_size

        # Compute gradients of the inputs
        grad_inputs = np.dot(grad_outputs, self.hidden_weights.T)

        # Update hidden weights and biases
        self.hidden_weights -= learning_rate * grad_hidden_weights
        self.hidden_biases -= learning_rate * grad_hidden_biases

        # print("output shape during FC layer's backpass: ", grad_inputs.shape)
        return grad_inputs

