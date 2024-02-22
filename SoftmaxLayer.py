from layers.Layer import Layer
import numpy as np
import pandas as pd
import scipy

class SoftmaxLayer(Layer):
    """
    A layer that applies the softmax function to its input.

    The softmax function is applied in the forward pass, converting input values
    into probabilities that sum to 1 across each row. In the backward pass, it computes
    the gradient of the loss with respect to the input of the softmax function.
    """

    def __init__(self):
        """
        Initializes the SoftmaxLayer.
        """
        super().__init__()

    def forward(self, dataIn):
        """
        Performs the forward pass using the softmax function.

        :param dataIn: Input data to the softmax layer, expected to be an NxD matrix
                       where N is the number of samples and D is the dimensionality.
        :type dataIn: np.ndarray or pd.DataFrame
        :return: The output of the softmax function applied to `dataIn`.
        :rtype: np.ndarray
        """
        # Convert input to numpy array if it's a pandas DataFrame
        if isinstance(dataIn, pd.DataFrame):
            dataIn = dataIn.values

        self.setPrevIn(dataIn)
        # Apply the softmax function
        numer = np.exp(dataIn - np.max(dataIn, axis=1, keepdims=True))
        denom = np.sum(numer, axis=1, keepdims=True)
        y = numer / denom
        self.setPrevOut(y)
        return y

    def gradient(self):
        """
        Computes the gradient of the softmax output with respect to the input.

        This method computes the Jacobian matrix for each output of the softmax layer.

        :return: An array of Jacobian matrices for each sample in the previous output.
        :rtype: np.ndarray
        """
        return np.array([np.diagflat(y) - np.outer(y, y) for y in self.getPrevOut()])

    def backward(self, gradIn):
        """
        Performs the backward pass, computing the gradient of the loss function with respect to the input.

        :param gradIn: The gradient of the loss with respect to the output of the softmax layer.
        :type gradIn: np.ndarray or scipy.sparse matrix
        :return: The gradient of the loss with respect to the input of the softmax layer.
        :rtype: np.ndarray
        """
        selfGrad = self.gradient()

        # Convert sparse matrix gradients to dense, if necessary
        if scipy.sparse.issparse(gradIn):
            gradIn = gradIn.toarray()
        if scipy.sparse.issparse(selfGrad):
            selfGrad = selfGrad.toarray()

        # Compute the dot product of the gradient with the softmax gradient
        return np.einsum('...i,...ij->...j', gradIn, selfGrad)
