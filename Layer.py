# Imports
from abc import ABC, abstractmethod


class Layer(ABC):
	"""
	Abstract base class for neural network layers.

	This class defines a common interface for different types of layers in a neural network.

	Attributes:
		__prevIn (list): Private variable to store the input from the previous layer.
		__prevOut (list): Private variable to store the output to pass to the next layer.
	"""

	def __init__( self ):
		"""
		Initialize the layer.

		Initializes the private variables _prevIn and _prevOut.
		"""
		self._prevIn = [ ]
		self._prevOut = [ ]

	def setPrevIn( self, dataIn ):
		"""
		Set the input received from the previous layer.

		:param dataIn: Data to be set as input of layer.
		:type dataIn: list
		"""
		self._prevIn = dataIn

	def setPrevOut( self, out ):
		"""
		Set the output to pass to the next layer.

		:param out: Data to be set as output of layer.
		:type out: list
		"""
		self._prevOut = out

	def getPrevIn( self ):
		"""
		Get the input received from the previous layer.

		:return: Input data of the layer.
		:rtype: list
		"""
		return self._prevIn

	def getPrevOut( self ):
		"""
		Get the output to pass to the next layer.

		:return: Output data of the layer.
		:rtype: list
		"""
		return self._prevOut

	@abstractmethod
	def forward( self, dataIn ):
		"""
		Perform the forward pass through the layer.

		:param dataIn: Input data to the layer.
		:type dataIn: list

		:return: Output data of the layer.
		:rtype: list
		"""
		pass

	@abstractmethod
	def gradient( self ):
		"""
		Compute the gradient of the layer (not yet implemented).

		:rtype: None
		"""
		pass

	@abstractmethod
	def backward( self, gradIn ):
		"""
		Perform the backward pass through the layer (not yet implemented).

		:param gradIn: Gradient of the loss with respect to the output of this layer.
		:type gradIn: list

		:rtype: None
		"""
		pass
