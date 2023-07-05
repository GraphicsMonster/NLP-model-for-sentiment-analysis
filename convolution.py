import numpy as np

class Conv1DLayer:

    def __init__(self, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.conv_filter = np.random.randn(filter_size, num_filters)

    def loss(self, pred, target):
        # compute loss function
        return np.mean((pred - target) ** 2)

    def forward(self, inputs):
        self.inputs = inputs
        num_inputs = inputs.shape[0]
        output_length = num_inputs - self.filter_size + 1
        self.output = np.zeros((output_length, self.num_filters))
        # Convolution
        # input dim is basically the size of the vocabulary

        for i in range(output_length):
            self.output[i] = np.dot(inputs[i:i+self.filter_size], self.conv_filter.T)

        return self.output
    
    def backward(self, grad_outputs, learning_rate):
        grad_input = np.zeros(grad_outputs.shape)
        grad_filter = np.zeros(self.conv_filter.shape)

        for i in range(grad_outputs.shape[0]):
            for j in range(self.num_filters):
               receptive_field = self.inputs[i:i+self.filter_size]
               grad_input[i:i+self.filter_size] += self.conv_filter[:, j] * grad_outputs[i, j]
               grad_filter[:, j] += receptive_field * grad_outputs[i, j]

            # Update the weights
            self.conv_filter -= learning_rate * grad_filter

        return grad_input