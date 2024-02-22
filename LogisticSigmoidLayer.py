from layers.Layer import Layer
import numpy as np

class LogisticSigmoidLayer(Layer):
    """
    Logistic sigmoid layer of the neural network.

    This class represents a logistic sigmoid layer of a neural network, which applies
    the logistic sigmoid activation function to input data.

    Methods:
        __init__: Initialize the logistic sigmoid layer.
        forward: Perform forward pass and set previous input and output.
        gradient: Compute gradients for the logistic sigmoid layer.
        backward: Perform backward pass for the logistic sigmoid layer.
    """

    def __init__(self):
        """
        Initialize the logistic sigmoid layer.

        """
        super().__init__()

    def forward(self, dataIn):
        """
        Perform forward pass and set previous input and output.

        :param dataIn: Input data as an NxD matrix.
        :type dataIn: np.ndarray

        :return: Output data after applying logistic sigmoid function.
        :rtype: np.ndarray
        """

        self.setPrevIn(dataIn)

        # Apply logistic sigmoid function
        y = 1 / (1 + np.exp(-dataIn))

        self.setPrevOut(y)
        return y

    def gradient(self):
        """
        Compute gradients for the logistic sigmoid layer.

        :return: Gradients for the logistic sigmoid layer.
        :rtype: np.ndarray
        """

        # Compute gradient using output of the layer
        y_hat = self.getPrevOut()
        return y_hat * (1 - y_hat)

    def backward(self, gradIn):
        """
        Perform backward pass for the logistic sigmoid layer.

        :param gradIn: Gradient of the loss with respect to the output of this layer.
        :type gradIn: np.ndarray

        :return: Gradient of the loss with respect to the input of this layer.
        :rtype: np.ndarray
        """

        # Compute gradient of the loss with respect to the input using chain rule
        return self.gradient() * gradIn
