import numpy as np
from Layer import Layer

class FlatteningLayer(Layer):
    def __init__(self):
        """
        Initializes a flattening layer, which is used to convert a multi-dimensional input
        into a one-dimensional vector. This is often used when transitioning from convolutional
        layers to fully connected layers within a neural network.
        """
        super().__init__()

    def forward(self, dataIn):
        """
        Perform the forward pass by flattening the input data into a one-dimensional array.

        :param dataIn: Incoming multi-dimensional feature map.
        :type dataIn: np.ndarray

        :return: Flattened one-dimensional feature vector.
        :rtype: np.ndarray
        """
        self.setPrevIn(dataIn)  # Store the input data for use in the backward pass

        # Flatten the input data into a one-dimensional vector in column-major order
        y = dataIn.flatten('F').reshape(1, -1)
        self.setPrevOut(y)  # Store the flattened data for use in subsequent layers
        return y

    def gradient(self, gradIn):
        """
        Compute the gradient of the loss function with respect to the input of this layer.
        Since this layer is a simple flattening layer, the gradient is simply reshaped back
        to the original input shape in column-major order.

        :param gradIn: Gradient of the loss with respect to the output of this layer.
        :type gradIn: np.ndarray

        :return: Gradient reshaped to the shape of the original input.
        :rtype: np.ndarray
        """
        return gradIn.reshape(self.getPrevIn().shape, order='F')

    def backward(self, gradIn):
        """
        Perform the backward pass by reshaping the gradient of the loss function with respect
        to the output of the layer back to the shape of the original input.

        :param gradIn: Gradient of the loss with respect to the output of this layer.
        :type gradIn: np.ndarray

        :return: Gradient reshaped to the shape of the original input.
        :rtype: np.ndarray
        """
        # Reshape gradient to match the input shape
        return self.gradient(gradIn)
