import numpy as np

class ClassificationLayer:

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size)
        self.biases = np.random.randn(output_size)

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases
        self.output = self.output - np.max(self.output, axis=1, keepdims=True)  # To prevent overflow
        self.output_probs = np.exp(self.output) / np.sum(np.exp(self.output), axis=1, keepdims=True)

        return self.output_probs

    def loss(self, pred_probs, targets):
        num_samples = len(targets)
        targets = np.array(targets)
        targets -= 1
        correct_probs = pred_probs[np.arange(num_samples), targets]
        loss = -np.log(correct_probs)
        total_loss = np.sum(loss) / num_samples
        return total_loss
        # this is the cross entropy loss function

    def backward(self, grad_outputs):
        self.grad_inputs = np.dot(grad_outputs, self.weights.T)
        self.grad_weights = np.dot(self.inputs.T, grad_outputs)
        self.grad_biases = np.sum(grad_outputs, axis=0)
        return self.grad_inputs
