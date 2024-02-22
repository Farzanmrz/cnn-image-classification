from Layer import Layer
import numpy as np
import pandas as pd


class FullyConnectedLayer2(Layer):
	"""
	A fully connected neural network layer implementing Xavier weight initialization and ADAM optimization.

	:param sizeIn: The number of features for input data.
	:param sizeOut: The number of features for output data.
	"""

	def __init__( self, sizeIn, sizeOut ):
		"""
		Initializes the FullyConnectedLayer2 with Xavier weight initialization and sets up ADAM optimization parameters.

		:param sizeIn: The number of input features.
		:param sizeOut: The number of output features.
		"""
		self.sizeIn = sizeIn
		self.sizeOut = sizeOut

		# Xavier (Glorot) weight initialization
		xav_weight = (6 / (sizeIn + sizeOut)) ** 0.5
		self.weights = np.random.uniform(-xav_weight, xav_weight, (sizeIn, sizeOut))
		self.bias = np.random.uniform(-xav_weight, xav_weight, (1, sizeOut))

		# Initialize ADAM optimization variables
		self.s = 0  # First moment vector
		self.r = 0  # Second moment vector
		self.p1 = 0.9  # Decay rate for the first moment estimates
		self.p2 = 0.999  # Decay rate for the second moment estimates
		self.delta = 1e-8  # Small constant for numerical stability

	def getWeights( self ):
		"""
		Returns the current weights of the layer.

		:return: The weight matrix of the layer.
		:rtype: np.ndarray
		"""
		return self.weights

	def setWeights( self, weights ):
		"""
		Sets the weights of the layer to the provided weights.

		:param weights: A new weight matrix to be used for the layer.
		:type weights: np.ndarray
		"""
		self.weights = weights

	def getBiases( self ):
		"""
		Returns the current biases of the layer.

		:return: The bias vector of the layer.
		:rtype: np.ndarray
		"""
		return self.bias

	def setBiases( self, biases ):
		"""
		Sets the biases of the layer to the provided biases.

		:param biases: A new bias vector to be used for the layer.
		:type biases: np.ndarray
		"""
		self.bias = biases

	def forward( self, dataIn ):
		"""
		Performs the forward pass through the layer using the input data.

		:param dataIn: Input data to the layer.
		:type dataIn: np.ndarray or pd.DataFrame
		:return: The output of the layer after applying weights and biases.
		:rtype: np.ndarray
		"""
		if isinstance(dataIn, pd.DataFrame):
			dataIn = dataIn.values

		self.setPrevIn(dataIn)
		y = np.dot(dataIn, self.getWeights()) + self.getBiases()
		self.setPrevOut(y)
		return y

	def gradient( self ):
		"""
		Returns the transpose of the weights matrix, used for backpropagation.

		:return: Transpose of the weights matrix.
		:rtype: np.ndarray
		"""
		return self.weights.T

	def backward( self, gradIn ):
		"""
		Performs the backward pass through the layer.

		:param gradIn: Gradient of the loss function with respect to the output of the layer.
		:type gradIn: np.ndarray
		:return: Gradient of the loss function with respect to the input of the layer.
		:rtype: np.ndarray
		"""
		return gradIn @ self.gradient()

	def updateWeights( self, gradIn, t, eta = 0.0001 ):
		"""
		Updates the weights and biases of the layer using the ADAM optimization algorithm.

		:param gradIn: Gradient of the loss function with respect to the output of the layer.
		:param t: Current iteration number (epoch).
		:param eta: Learning rate.
		:type gradIn: np.ndarray
		:type t: int
		:type eta: float
		"""
		dJdb = np.sum(gradIn, axis = 0) / gradIn.shape[ 0 ]
		dJdW = (np.array(self.getPrevIn()).T @ gradIn) / gradIn.shape[ 0 ]

		# First moment update
		self.s = (self.p1 * self.s) + ((1 - self.p1) * dJdW)

		# Second moment update
		self.r = (self.p2 * self.r) + ((1 - self.p2) * (dJdW ** 2))

		# Correct moments for bias correction
		s_corrected = self.s / (1 - self.p1 ** (t + 1))
		r_corrected = self.r / (1 - self.p2 ** (t + 1))

		# Compute update term
		update_term = s_corrected / (np.sqrt(r_corrected) + self.delta)

		# Update weights and biases
		self.weights -= eta * update_term
		self.bias -= eta * dJdb
