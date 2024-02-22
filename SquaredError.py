# Imports
import numpy as np

class SquaredError:
    """
    Squared Error Loss function.

    This class represents the squared error loss function used for evaluating regression models.

    Methods:
        eval: Calculate the squared error loss.
        gradient: Calculate the gradient of the squared error loss.
    """

    def eval(self, Y, Yhat):
        """
        Calculate the squared error loss.

        :param Y: True labels.
        :type Y: np.ndarray
        :param Yhat: Predicted values.
        :type Yhat: np.ndarray
        :return: Squared error loss value.
        :rtype: float
        """
        # Calculate the squared error loss
        return np.mean((Y - Yhat) * (Y - Yhat))

    def gradient(self, Y, Yhat):
        """
        Calculate the gradient of the squared error loss.

        :param Y: True labels.
        :type Y: np.ndarray
        :param Yhat: Predicted values.
        :type Yhat: np.ndarray
        :return: Gradient of the squared error loss.
        :rtype: np.ndarray
        """
        # Calculate the gradient of the squared error loss
        return -2 * (Y - Yhat)
