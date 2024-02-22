# Imports
from layers.Layer import Layer
import numpy as np

class tanhLayer(Layer):
    """
    Hyperbolic Tangent (tanh) Activation Layer.

    This class represents the tanh activation layer used in neural networks.

    Methods:
        forward: Perform forward pass, compute the tanh activation, and set previous input and output.
        gradient: Compute the gradient of the tanh activation.
        backward: Perform backward pass, compute the gradient of the loss with respect to the input.
    """

    def __init__(self):
        """Initialize the tanh activation layer."""
        super().__init__()

    def forward(self, dataIn):
        """
        Perform forward pass, compute the tanh activation, and set previous input and output.

        :param dataIn: Input data.
        :type dataIn: np.ndarray
        :return: Output after applying tanh activation.
        :rtype: np.ndarray
        """
        # Set previous input
        self.setPrevIn(dataIn)

        # Compute tanh activation
        y = (np.exp(dataIn) - np.exp(-dataIn)) / (np.exp(dataIn) + np.exp(-dataIn))

        # Set previous output
        self.setPrevOut(y)
        return y

    def gradient(self):
        """
        Compute the gradient of the tanh activation.

        :return: Gradient of the tanh activation.
        :rtype: np.ndarray
        """
        # Compute gradient of tanh activation
        return 1 - (self.getPrevOut() ** 2)

    def backward(self, gradIn):
        """
        Perform backward pass, compute the gradient of the loss with respect to the input.

        :param gradIn: Gradient of the loss with respect to the output of this layer.
        :type gradIn: np.ndarray
        :return: Gradient of the loss with respect to the input.
        :rtype: np.ndarray
        """
        # Compute gradient of loss with respect to the input
        return gradIn * self.gradient()
