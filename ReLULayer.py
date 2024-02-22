# Imports
from layers.Layer import Layer
import numpy as np

class ReLULayer(Layer):
    """
    ReLU Activation Layer.

    This class represents the Rectified Linear Unit (ReLU) activation layer used in neural networks.

    Methods:
        __init__: Initialize the ReLU layer.
        forward: Perform the forward pass through the ReLU layer.
        gradient: Compute the gradient of the ReLU activation function.
        backward: Perform the backward pass through the ReLU layer.
    """

    def __init__(self):
        """Initialize the ReLU layer."""
        super().__init__()

    def forward(self, dataIn):
        """
        Perform the forward pass through the ReLU layer.

        :param dataIn: Input data as an NxD matrix.
        :type dataIn: np.ndarray
        :return: Output data after ReLU activation.
        :rtype: np.ndarray
        """
        # Set previous input
        self.setPrevIn(dataIn)

        # Apply ReLU activation function
        y = np.maximum(0, dataIn)

        # Set previous output
        self.setPrevOut(y)

        # Return forward propogated value
        return y

    def gradient(self):
        """
        Compute the gradient of the ReLU activation function.

        :return: Gradient of the ReLU activation function.
        :rtype: np.ndarray
        """
        # Compute gradient of ReLU activation
        return np.where(self.getPrevIn() > 0, 1, 0)

    def backward(self, gradIn):
        """
        Perform the backward pass through the ReLU layer.

        :param gradIn: Gradient of the loss with respect to the output of this layer.
        :type gradIn: np.ndarray
        :return: Gradient of the loss with respect to the input of this layer.
        :rtype: np.ndarray
        """
        # Compute gradient of loss with respect to input
        return gradIn * self.gradient()
