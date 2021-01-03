"""62136."""

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def derivative(x):
    return x * (1 - x)

class Perceptron:
    def __init__(self, input_dim: int, output_dim: int):
        self.synaptic_weights = 2 * np.random.random((input_dim, output_dim)) - 1

    def train(
        self,
        training_inputs: np.array,
        training_outputs: np.array,
        training_iterations: int
    ):
        for _ in range(training_iterations):
            output = self.activate(training_inputs)
            error = training_outputs - output
            adjustments = np.dot(training_inputs.T, error * derivative(output))
            self.synaptic_weights += adjustments

    def activate(self, inputs: np.array):
        inputs = inputs.astype(float)
        result = sigmoid(np.dot(inputs, self.synaptic_weights))
        return result