import numpy as np 


class NeuralNet:
    """
    A basic feed-forward neural network
    """
    def __init__(self, sizes):
        """
        Initializes a new neural network. Sizes is a list of numbers indicating the sizes of 
        the layers in this neural network.
        """
        self.sizes = sizes
        self.layers = len(sizes)
        self.weights = [np.random.normal(size=(x, y)) for x, y in zip(sizes[1:], sizes[:-1])]
        self.biases = [np.random.normal(size=x) for x in sizes[1:]]

    def feed_forward(self, input):
        """
        Returns the output of this neural_net based on the input
        """
        for x in range(self.layers - 1):
            input = sigmoid(self.weights @ input + self.biases[x])

        return input

    def backprop_gradient(self, input, desired_output):
        activations = [input]
        z = []
        for layer in range(self.layers):
            z.append(np.dot(self.weights[layer], activations[layer]) + self.biases[layer])
            activations.append(sigmoid(z[layer]))


def sigmoid(x):
    """
    The compression function used for this neural network
    """
    return 1.0 / (1.0 + np.exp(-x))

def deriv_sigmoid(x):
    """
    derivative of the sigmoid function
    """
    return np.exp(x) / ((1 + np.exp(x)) ** 2)

def deriv_cost(expected, actual):
    """
    The derivative of mean squared error loss.

    expected: the desired activation of an output neuron
    actual: the actual activation of the same output neuron
    """
    return (actual - expected)