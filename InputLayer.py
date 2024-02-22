# Imports
from layers.Layer import Layer
import numpy as np


class InputLayer(Layer):
	"""
	Input layer of the neural network.

	This class represents the input layer of a neural network, responsible for normalizing input data
	by computing mean and standard deviation.

	Attributes:
		meanX (float): Mean of the input data.
		stdX (float): Standard deviation of the input data.

	Methods:
		__init__: Initialize the input layer with input data and compute mean and standard deviation.
		forward: Perform forward pass, normalize input data, and set previous input and output.
		gradient: Placeholder method for computing gradients (not yet implemented).
		backward: Placeholder method for performing backward pass (not yet implemented).
	"""

	def __init__( self, dataIn ):
		"""
		Initialize the input layer with input data and compute mean and standard deviation.

		:param dataIn: Input data as an NxD matrix.
		:type dataIn: np.ndarray
		"""

		# Compute mean and standard deviation of the input data
		self.meanX = dataIn.mean()
		self.stdX = dataIn.std()

		# Prevent division by zero by replacing zero standard deviation with 1
		self.stdX = np.where(self.stdX == 0, 1, self.stdX)

	def forward( self, dataIn ):
		"""
		Perform forward pass, normalize input data, and set previous input and output.

		:param dataIn: Input data as an NxD matrix.
		:type dataIn: np.ndarray

		:return: Normalized input data as an NxD matrix.
		:rtype: np.ndarray
		"""

		# Set previous input
		self.setPrevIn(dataIn)

		# Normalize input data using computed mean and standard deviation
		y = (dataIn - self.meanX) / self.stdX

		# Set previous output
		self.setPrevOut(y)
		return y

	def gradient( self ):
		"""
		Compute gradients for the input layer.

		Since the input layer typically does not have parameters to learn, this function would
		generally not be implemented. If the design requires the input layer to have learnable parameters,
		this method should be overridden in the subclass.

		Raises:
			NotImplementedError: This exception is raised to indicate that gradient computation
								 is not applicable for the input layer within the current design.

		:rtype: None
		"""
		raise NotImplementedError("Gradient calculation is not implemented for InputLayer.")

	def backward( self, gradIn ):
		"""
		Perform the backward pass for the input layer.

		The input layer's backward pass would typically not be implemented because it is the first layer
		in the network and does not have any previous layers to send gradients to. However, if required,
		it should propagate the gradient of the loss function with respect to its inputs to any previous
		network components, which is not typical for an input layer.

		:param gradIn: Gradient of the loss with respect to the output of this layer.
					   However, since this layer does not have any weights, this would normally not be used.
		:type gradIn: np.ndarray

		Raises:
			NotImplementedError: This exception is raised to indicate that the backward pass is not
								 implemented for the input layer as it does not have preceding layers.

		:rtype: None
		"""
		raise NotImplementedError("Backward pass is not implemented for InputLayer.")
