import numpy as np
import pandas as pd

from Layer import Layer


class ConvolutionalLayer(Layer):
	"""
	A Convolutional Layer

	:param kw: width of the kernel.
	:param kh: height of the kernel.
	"""

	def __init__( self, kh, kw ):
		"""
		Initializes the Convolutional Layer

		:param kw: kernel width.
		:param kh: kernel height.
		"""
		self.kw = kw
		self.kh = kh

		# Initialize weights which is the kernel
		self.weights = np.random.uniform(-1e-4, 1e-4, (kh, kw))

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

	def forward( self, dataIn ):
		"""
		Performs the forward pass through the layer using the input data.

		:param dataIn: Input data to the layer.
		:type dataIn: np.ndarray or pd.DataFrame
		:return: The output of the layer after applying weights and biases.
		:rtype: np.ndarray
		"""
		self.setPrevIn(dataIn)
		y = self.crossCorrelate2D(dataIn)
		self.setPrevOut(y)
		return y

	def gradient( self ):
		"""
		Returns the transpose of the weights matrix, used for backpropagation.

		:return: Transpose of the weights matrix.
		:rtype: np.ndarray
		"""
		pass

	def backward( self, gradIn ):
		"""
		Performs the backward pass through the layer.

		:param gradIn: Gradient of the loss function with respect to the output of the layer.
		:type gradIn: np.ndarray
		:return: Gradient of the loss function with respect to the input of the layer.
		:rtype: np.ndarray
		"""
		pass

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
		dJdk = gradIn * self.getPrevIn()

		self.weights -= eta * dJdk

	def crossCorrelate2D( self, dataIn ):
		dim1 = dataIn.shape[ 0 ] - self.kh + 1
		dim2 = dataIn.shape[ 1 ] - self.kw + 1
		fmap = np.zeros((dim1, dim2))
		for i in range(dim1):
			for j in range(dim2):
				fmap[ i, j ] = np.sum(self.getWeights() * dataIn[ i: i + self.kh, j: j + self.kw ])

		return fmap


x = np.array([ [ 1, 2, 3, 4 ], [ 2, 2, 3, 2 ], [ 1, 3, 3, 3 ], [ 4, 4, 4, 4 ], [ 4, 8, 4, 7 ] ])
kern = np.array([ [ 1, 2 ], [ 2, 3 ] ])

lay = ConvolutionalLayer(3, 3)

print(lay.getWeights())

y = lay.forward(x)
print(y)
