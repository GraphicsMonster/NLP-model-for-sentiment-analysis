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
        
        print("outputs shape during convolution forward pass: ", outputs.shape)
        return outputs
    
    def backward(self, grad_outputs, learning_rate):
        batch_size, _, input_sequence_length = self.inputs.shape
        grad_inputs = np.zeros_like(self.inputs)
        grad_weights = np.zeros_like(self.weights)
        grad_biases = np.zeros_like(self.biases)
        filter_size = self.filter_size

        # Reshape grad_outputs to (batch_size, num_filters, input_sequence_length, 1)
        grad_outputs_reshaped = grad_outputs[:, :, :, np.newaxis]

        # Compute gradients for inputs and weights using broadcasting
        grad_inputs = np.sum(grad_outputs_reshaped * self.weights[:, :, :, np.newaxis], axis=1)
        grad_weights = np.sum(grad_outputs_reshaped * self.inputs[:, np.newaxis, :, :], axis=0)

        # Accumulate gradients across the batch
        grad_biases = np.sum(grad_outputs, axis=(0, 2))

        # Take average of gradients across the batch (if using mini-batch gradient descent)
        grad_weights /= batch_size
        grad_biases /= batch_size

        # Update parameters using gradients
        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases

        return grad_inputs


