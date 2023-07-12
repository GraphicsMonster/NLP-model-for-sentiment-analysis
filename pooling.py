import numpy as np

class PoolingLayer:

    def __init__(self, pool_size):
        self.pool_size = pool_size

    def forward(self, inputs):
        self.inputs = inputs
        batch_size, input_size = inputs.shape
        self.output = np.zeros((batch_size, input_size // self.pool_size))
        
        for i in range(batch_size):
            for j in range(0, input_size, self.pool_size):
                self.output[i, j // self.pool_size] = np.max(inputs[i, j:j+self.pool_size])

        return self.output
    
    def backward(self, grad_outputs):
        grad_inputs = np.zeros(self.inputs.shape)
        batch_size, input_size = grad_outputs.shape

        for i in range(batch_size):
            for j in range(input_size):
                grad_inputs[i, j*self.pool_size:(j+1)*self.pool_size] = grad_outputs[i, j]

        return grad_inputs
