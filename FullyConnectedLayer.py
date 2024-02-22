# Imports
from layers.Layer import Layer
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

		# Randomly initialize weights and biases
		self.weights = np.random.uniform(-1e-4, 1e-4, (sizeIn, sizeOut))
		self.bias = np.random.uniform(-1e-4, 1e-4, (1, sizeOut))

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
		y = np.dot(dataIn, self.getWeights()) + self.getBiases()

		# Set previous output
		self.setPrevOut(y)
		return y

	def gradient( self ):
		"""
		Compute gradients for the fully connected layer.

		:rtype: numpy.ndarray
		"""
		return self.weights.T

	def backward( self, gradIn ):
		"""
		Perform backward pass for the fully connected layer.

		:param gradIn: Gradient of the loss with respect to the output of this layer.
		:type gradIn: numpy.ndarray

		:rtype: numpy.ndarray
		"""
		return gradIn @ self.gradient()

	def updateWeights( self, gradIn, eta = 0.0001 ):
		"""
		Update weights and biases of the layer using gradient descent.

		:param gradIn: Gradient of the loss with respect to the output of this layer.
		:type gradIn: numpy.ndarray
		:param eta: Learning rate (default is 0.0001).
		:type eta: float
		"""
		# Compute gradients of weights and biases
		dJdb = np.sum(gradIn, axis = 0) / gradIn.shape[ 0 ]
		dJdW = (self.getPrevIn().T @ gradIn) / gradIn.shape[ 0 ]

		# Update weights and biases using gradient descent
		self.setWeights(self.getWeights() - (eta * dJdW))
		self.setBiases(self.getBiases() - (eta * dJdb))
