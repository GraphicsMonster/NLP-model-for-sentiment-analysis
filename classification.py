import numpy as np

class ClassificationLayer:

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.random.randn(output_size)

    def forward(self, inputs):
        self.inputs = inputs
        self.logits = np.dot(inputs, self.weights) + self.biases
        self.logits -= np.max(self.logits, axis=1, keepdims=True)  # Stability trick to prevent overflow
        self.probs = np.exp(self.logits) / np.sum(np.exp(self.logits), axis=1, keepdims=True)

        return self.probs

    def loss(self, pred_probs, targets):
        num_samples = len(targets)
        loss = -np.sum(targets * np.log(pred_probs + 1e-10)) / num_samples
        return loss

    def backward(self, grad_probs, targets):
        num_samples = len(targets)
        self.grad_inputs = (grad_probs - targets) / num_samples
        self.grad_weights = np.dot(self.inputs.T, self.grad_inputs.reshape(-1, 1))
        self.grad_biases = np.sum(self.grad_inputs, axis=0)
        return self.grad_inputs

