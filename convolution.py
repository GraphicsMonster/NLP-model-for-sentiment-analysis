import numpy as np

class Conv1DLayer:

    def __init__(self, num_filters, filter_size):
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.weights = None
        self.biases = None

    def initialize_params(self, inputs_shape):
        input_channels = inputs_shape[1]
        self.weights = np.random.randn(self.num_filters, input_channels, self.filter_size)
        self.biases = np.zeros(self.num_filters)

    def forward(self, inputs):
        if self.weights is None or self.biases is None:
            self.initialize_params(inputs.shape)

        inputs = inputs.toarray()
        self.inputs = inputs[:, :, np.newaxis]
        batch_size, vocab_size = inputs.shape
        output_sequence_length = vocab_size - self.filter_size + 1
        outputs = np.zeros((batch_size, output_sequence_length, self.num_filters))

        for batch in range(batch_size):
            for output_sequence in range(output_sequence_length):
                for filter in range(self.num_filters):
                    #Applying convolution operation
                    outputs[batch, output_sequence, filter] = np.sum(
                        self.inputs[batch, output_sequence:output_sequence + self.filter_size].T * self.weights[filter]).T + self.biases[filter]
        
        # print("outputs shape during convolution forward pass: ", outputs.shape)
        return outputs

    def backward(self, grad_outputs, learning_rate):
        batch_size, output_sequence_length, num_filters = grad_outputs.shape
        grad_inputs = np.zeros(self.inputs.shape)  # Initialize with the shape of the input
        grad_filters = np.zeros_like(self.weights)  # Initialize with the shape of the weights

        for batch in range(batch_size):
            for filter_idx in range(self.num_filters):
                for i in range(output_sequence_length):
                    start = i
                    end = i + self.filter_size
                    grad_inputs[batch, start:end] += np.sum(grad_outputs[batch, i, filter_idx] * self.weights[filter_idx])
                    grad_filters[filter_idx] += np.outer(grad_outputs[batch, i, filter_idx], self.inputs[batch, start:end])

        # Update filter weights and biases using gradients
        self.weights -= learning_rate * grad_filters
        self.biases -= learning_rate * np.sum(grad_outputs, axis=(0, 1))  # Update biases using sum of gradients

        return grad_inputs




