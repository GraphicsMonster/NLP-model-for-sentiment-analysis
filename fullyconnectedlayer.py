import numpy as np

class FullyConnectedLayer:

    
    def __init__(self, input_size, output_size):
        
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.random.randn(input_size)

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.dot(inputs, self.weights.T) + self.biases
        return self.outputs
    
    def backward(self, grad_outputs, learning_rate):
        self.grad_inputs = np.dot(grad_outputs, self.weights)
        self.grad_weights = np.dot(grad_outputs.T, self.inputs)
        self.grad_biases = np.sum(grad_outputs, axis=0)
        self.weights -= learning_rate * self.grad_weights
        return self.grad_inputs