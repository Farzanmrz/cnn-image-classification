# Imports
from layers.Layer import Layer
import numpy as np

class LinearLayer(Layer):
    """
    Linear layer of the neural network.

    This class represents a linear layer of a neural network, which simply passes input data
    through without any transformation.

    Methods:
        __init__: Initialize the linear layer.
        forward: Perform forward pass and set previous input and output.
        gradient: Compute gradients for the linear layer.
        backward: Perform backward pass for the linear layer.
    """

    def __init__(self):
        """
        Initialize the linear layer.

        """

        super().__init__()

    def forward(self, dataIn):
        """
        Perform forward pass and set previous input and output.

        :param dataIn: Input data as an NxD matrix.
        :type dataIn: np.ndarray

        :return: Input data as an NxD matrix.
        :rtype: np.ndarray
        """

        self.setPrevIn(dataIn)

        # Simply pass input data through
        y = dataIn

        self.setPrevOut(y)
        return y

    def gradient(self):
        """
        Compute gradients for the linear layer.

        :return: Gradients for the linear layer.
        :rtype: np.ndarray
        """

        # Gradients are ones of the same shape as previous output
        return np.ones_like(self.getPrevOut())

    def backward(self, gradIn):
        """
        Perform backward pass for the linear layer.

        :param gradIn: Gradient of the loss with respect to the output of this layer.
        :type gradIn: np.ndarray

        :return: Gradient of the loss with respect to the input of this layer.
        :rtype: np.ndarray
        """

        # Return the gradient unchanged
        return gradIn
