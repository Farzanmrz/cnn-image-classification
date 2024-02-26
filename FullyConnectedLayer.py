# Imports
from Layer import Layer
import numpy as np
import pandas as pd


class FullyConnectedLayer(Layer):
	"""
	Fully connected layer of the neural network.

	This class represents a fully connected layer in a neural network, which connects every input neuron
	to every output neuron.

	Attributes:
		sizeIn (int): Number of input features.
		sizeOut (int): Number of output features.
		weights (numpy.ndarray): Weight matrix connecting input and output neurons.
		bias (numpy.ndarray): Bias vector.

	Methods:
		__init__: Initialize the fully connected layer with input and output sizes, and randomly initialize weights and biases.
		getWeights: Get the weight matrix of the layer.
		setWeights: Set the weight matrix of the layer.
		getBiases: Get the bias vector of the layer.
		setBiases: Set the bias vector of the layer.
		forward: Perform forward pass through the layer.
		gradient: Compute gradients for the fully connected layer.
		backward: Perform backward pass for the fully connected layer.
		updateWeights: Update weights and biases of the layer using gradient descent.
	"""

	def __init__( self, sizeIn, sizeOut ):
		"""
		Initialize the fully connected layer with input and output sizes,
		and randomly initialize weights and biases.

		:param sizeIn: Number of input features.
		:type sizeIn: int
		:param sizeOut: Number of output features.
		:type sizeOut: int
		"""
		self.sizeIn = sizeIn
		self.sizeOut = sizeOut

		# Xavier weight initialization
		xav_weight = np.sqrt(6 / (sizeIn + sizeOut))

		# Randomly initialize weights
		self.weights = np.random.uniform(-xav_weight, xav_weight, (sizeIn, sizeOut))

		# Initialize variables for Adam optimization algorithm
		self.s = 0  # First moment vector
		self.r = 0  # Second moment vector
		self.p1 = 0.9  # Decay rate for the first moment estimates
		self.p2 = 0.999  # Decay rate for the second moment estimates
		self.delta = 1e-8  # Small constant for numerical stability


	def getWeights( self ):
		"""
		Get the weight matrix of the layer.

		:return: Weight matrix.
		:rtype: numpy.ndarray
		"""
		return self.weights

	def setWeights( self, weights ):
		"""
		Set the weight matrix of the layer.

		:param weights: New weight matrix.
		:type weights: numpy.ndarray
		"""
		self.weights = weights

	def getBiases( self ):
		"""
		Get the bias vector of the layer.

		:return: Bias vector.
		:rtype: numpy.ndarray
		"""
		return self.bias

	def setBiases( self, biases ):
		"""
		Set the bias vector of the layer.

		:param biases: New bias vector.
		:type biases: numpy.ndarray
		"""
		self.bias = biases

	def forward( self, dataIn ):
		"""
		Perform forward pass through the layer.

		:param dataIn: Input data as an NxD matrix.
		:type dataIn: numpy.ndarray

		:return: Output data as an NxK matrix.
		:rtype: numpy.ndarray
		"""
		# Check if dataIn is a DataFrame and convert it to ndarray if needed
		if isinstance(dataIn, pd.DataFrame):
			dataIn = dataIn.values

		# Set previous input
		self.setPrevIn(dataIn)

		# Calculate output using dot product of input and weights, plus bias
		y = np.dot(dataIn, self.getWeights()) #+ self.getBiases()

		# Set previous output
		self.setPrevOut(y)
		return y

	def gradient( self ):
		"""
		Compute gradients for the fully connected layer.

		:rtype: numpy.ndarray
		"""
		return self.getWeights().T

	def backward( self, gradIn: np.ndarray ):
		"""
		Performs the backward pass of the layer.

		:param gradIn: The gradient of the loss with respect to the output of this layer.
		:return: The gradient of the loss with respect to the input of this layer.
		"""
		return gradIn @ self.gradient()


	def updateWeights( self, gradIn,t, eta = 0.01 ):
		"""
		Updates the weights and biases of the layer using gradient descent and the Adam optimization algorithm.

		:param gradIn: The gradient of the loss with respect to the output of this layer.
		:param t: The current iteration (time step) of the optimization.
		:param eta: The learning rate.
		"""

		# Compute gradients of weights
		dJdW = self.getPrevIn().T @ gradIn

		# First moment update
		self.s = (self.p1 * self.s) + ((1 - self.p1) * dJdW)

		# Second moment update
		self.r = (self.p2 * self.r) + ((1 - self.p2) * (dJdW ** 2))

		# Final update term
		update_term = (self.s / (1 - (self.p1 ** (t + 1)))) / (np.sqrt((self.r) / (1 - (self.p2 ** (t + 1)))) + self.delta)

		# Update weights and biases using gradient descent
		self.setWeights(self.getWeights() - (eta * update_term))

