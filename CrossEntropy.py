# Imports
import numpy as np
import scipy


class CrossEntropy:
	"""
	Cross Entropy Loss function.

	This class represents the cross entropy loss function used for evaluating classification models.

	Methods:
		eval: Calculate the cross entropy loss.
		gradient: Calculate the gradient of the cross entropy loss.
	"""

	def eval( self, Y, Yhat, epsilon = 1e-7 ):
		"""
		Calculate the cross entropy loss.

		:param Y: True labels.
		:type Y: np.ndarray
		:param Yhat: Predicted probabilities.
		:type Yhat: np.ndarray
		:param epsilon: Smoothing parameter to avoid division by zero, defaults to 1e-7.
		:type epsilon: float, optional
		:return: Cross entropy loss value.
		:rtype: float
		"""
		if scipy.sparse.issparse(Y):
			Y = Y.toarray()
		if scipy.sparse.issparse(Yhat):
			Yhat = Yhat.toarray()
		return -np.mean(np.sum(Y * np.log(Yhat + epsilon), axis=1))


	def gradient( self, Y, Yhat, epsilon = 1e-7 ):
		"""
		Calculate the gradient of the cross entropy loss.

		:param Y: True labels.
		:type Y: np.ndarray
		:param Yhat: Predicted probabilities.
		:type Yhat: np.ndarray
		:param epsilon: Smoothing parameter to avoid division by zero, defaults to 1e-7.
		:type epsilon: float, optional
		:return: Gradient of the cross entropy loss.
		:rtype: np.ndarray
		"""

		# Calculate the gradient of the cross entropy loss
		return -((Y) / (Yhat + epsilon))
