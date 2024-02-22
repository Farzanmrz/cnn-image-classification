# Imports
import numpy as np

class LogLoss:
    """
    Logarithmic Loss function.

    This class represents the logarithmic loss function used for evaluating classification models.

    Methods:
        eval: Calculate the logarithmic loss.
        gradient: Calculate the gradient of the logarithmic loss.
    """

    def eval(self, Y, Yhat, epsilon=1e-7):
        """
        Calculate the logarithmic loss.

        :param Y: True labels.
        :type Y: np.ndarray
        :param Yhat: Predicted probabilities.
        :type Yhat: np.ndarray
        :param epsilon: Smoothing parameter to avoid division by zero, defaults to 1e-7.
        :type epsilon: float, optional
        :return: Logarithmic loss value.
        :rtype: float
        """
        # Calculate the logarithmic loss
        return -np.mean(Y * np.log(Yhat + epsilon) + (1 - Y) * np.log(1 - Yhat + epsilon))

    def gradient(self, Y, Yhat, epsilon=1e-7):
        """
        Calculate the gradient of the logarithmic loss.

        :param Y: True labels.
        :type Y: np.ndarray
        :param Yhat: Predicted probabilities.
        :type Yhat: np.ndarray
        :param epsilon: Smoothing parameter to avoid division by zero, defaults to 1e-7.
        :type epsilon: float, optional
        :return: Gradient of the logarithmic loss.
        :rtype: np.ndarray
        """
        # Calculate the gradient of the logarithmic loss
        return -((Y - Yhat) / (Yhat * (1 - Yhat) + epsilon))
