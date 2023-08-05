import numpy as np

class PoolingLayer:

    def __init__(self, pool_size):
        self.pool_size = pool_size

    def forward(self, inputs):
        batch_size, input_sequence_length, num_filters = inputs.shape
        output_sequence_length = input_sequence_length // self.pool_size
        self.output = np.zeros((batch_size, output_sequence_length, num_filters))

        for batch in range(batch_size):
            for filter in range(num_filters):
                for i in range(output_sequence_length):
                    #Performing max pooling
                    start = i * self.pool_size
                    end = start + self.pool_size
                    self.output[batch, i, filter] = np.max(inputs[batch, start:end, filter])

        print("output shape during pooling forward pass: ", self.output.shape)
        return self.output
    
    def backward(self, grad_outputs):
        batch_size, output_sequence_length, num_filters = grad_outputs.shape
        input_sequence_length = output_sequence_length * self.pool_size
        grad_inputs = np.zeros((batch_size, input_sequence_length, num_filters))

        for batch in range(batch_size):
            for filter in range(num_filters):
                for i in range(output_sequence_length):
                    start = i * self.pool_size
                    end = start + self.pool_size
                    grad_inputs[batch, start:end, filter] = grad_outputs[batch, i, filter]

        print("shape of the output of backpass of pooling layer: ", grad_inputs.shape)
        return grad_inputs