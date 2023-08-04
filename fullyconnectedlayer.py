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

    def initialize_parameters(self):
        self.weights = np.random.randn(self.input_size, self.output_size)
        self.biases = np.zeros(self.output_size)

        if self.hidden_units is not None:
            self.hidden_weights = np.random.randn(self.input_size, self.hidden_units)
            self.hidden_biases = np.zeros(self.input_size)

    def forward(self, inputs):
        if self.weights is None or self.biases is None:
            self.initialize_parameters()

        batch_size, input_sequence_length, num_filters = inputs.shape
        inputs = inputs.reshape(batch_size, num_filters * input_sequence_length)  # Flatten the inputs

        if self.hidden_units is not None:
            hidden_outputs_dot = np.dot(inputs.T, self.hidden_weights.T)
            print("hidden weight shape: ", self.hidden_weights.shape)
            print("hidden outputs dot product: ", hidden_outputs_dot.shape)
            hidden_outputs = hidden_outputs_dot + self.hidden_biases
            hidden_outputs = np.maximum(0, hidden_outputs)  # ReLU activation for the hidden layer
            outputs = np.dot(hidden_outputs, self.weights) + self.biases
        else:
            outputs = np.dot(inputs.T, self.weights.T) + self.biases

        print("outputs shape during FC forward pass: ", outputs.shape)
        return outputs

    def backward(self, grad_outputs, learning_rate):
        batch_size, output_size = grad_outputs.shape
        grad_inputs = np.dot(grad_outputs, self.weights.T)
        grad_inputs = grad_inputs.reshape(batch_size, -1, self.input_size)  # Reshape back to original input shape

        if self.hidden_units is not None:
            hidden_outputs = np.dot(grad_outputs.reshape(batch_size, -1), self.hidden_weights) + self.hidden_biases
            hidden_outputs[hidden_outputs <= 0] = 0  # ReLU backward for the hidden layer
            grad_hidden = np.dot(grad_outputs, self.weights.T)
            grad_hidden[hidden_outputs <= 0] = 0  # ReLU backward for the hidden layer

            grad_hidden_weights = np.dot(grad_outputs.reshape(batch_size, -1).T, grad_hidden)
            grad_hidden_biases = np.sum(grad_hidden, axis=0)

        grad_weights = np.dot(grad_outputs.T, grad_outputs.reshape(batch_size, -1)).T
        grad_biases = np.sum(grad_outputs, axis=0)

        # Update parameters using gradients
        self.weights -= learning_rate * grad_weights / batch_size
        self.biases -= learning_rate * grad_biases / batch_size

        if self.hidden_units is not None:
            self.hidden_weights -= learning_rate * grad_hidden_weights / batch_size
            self.hidden_biases -= learning_rate * grad_hidden_biases / batch_size

        return grad_inputs
