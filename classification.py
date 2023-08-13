import numpy as np

class ClassificationLayer:
    def __init__(self, input_size, num_classes):
        self.input_size = input_size
        self.num_classes = num_classes
        self.weights = None
        self.biases = None

    def initialize_parameters(self, inputs):
        self.weights = np.random.randn(inputs.shape[1], self.num_classes)
        self.biases = np.random.randn(self.num_classes)

    def forward(self, inputs):
        if self.weights is None or self.biases is None:
            self.initialize_parameters(inputs)

        outputs = np.dot(inputs, self.weights) + self.biases
        outputs = self.softmax(outputs)
        # print("output shape during classification layer's forward pass: ", outputs.shape)
        return outputs
    
    def softmax(self, logits):
        max_logits = np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits - max_logits)
        return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    def loss(self, probs, targets):

        loss = np.maximum(0, 1 - targets * probs) # hinge loss function

        return loss.sum()
    
    def backward(self, probs, targets, learning_rate):
        num_samples = len(targets)
        grad_logits = probs - targets
        grad_inputs = np.dot(grad_logits, self.weights.T)

        grad_weights = np.dot(grad_inputs.T, grad_logits) / num_samples
        grad_biases = np.sum(grad_logits, axis=0) / num_samples

        # Update parameters using gradients
        self.weights -= learning_rate * grad_weights
        self.biases -= learning_rate * grad_biases

        # print("output shape during classification layer's backpass: ", grad_inputs.shape)
        return grad_inputs

