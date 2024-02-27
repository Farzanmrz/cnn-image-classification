import numpy as np
from Layer import Layer


class ConvolutionalLayer(Layer):
    """
    A Convolutional Layer class for use in Convolutional Neural Networks (CNNs).
    This layer applies a set of learnable filters (kernels) to the input data to create feature maps,
    which are useful for image recognition tasks.

    :param kh: Kernel height.
    :param kw: Kernel width.
    """

    def __init__( self, kh, kw ):
        """
        Initializes the Convolutional Layer with specified kernel size.

        :param kh: Kernel height.
        :param kw: Kernel width.
        """
        super().__init__()

        # Set seed for reproducibility
        np.random.seed(42)

        # Initialize kernel height and width
        self.kh = kh
        self.kw = kw

        # Initialize variables for Adam optimization algorithm
        self.s = 0  # First moment vector
        self.r = 0  # Second moment vector
        self.p1 = 0.9  # Decay rate for the first moment estimates
        self.p2 = 0.999  # Decay rate for the second moment estimates
        self.delta = 1e-8 # Small constant for numerical stability
        # Initialize kernel weights with small random values
        self.weights = np.random.uniform(-1e-4, 1e-4, (kh, kw))

        # Initialize variables for Adam optimization algorithm
        self.s = 0  # First moment vector
        self.r = 0  # Second moment vector
        self.p1 = 0.9  # Decay rate for the first moment estimates
        self.p2 = 0.999  # Decay rate for the second moment estimates
        self.delta = 1e-8 # Small constant for numerical stability

    def getWeights(self):
        """
        Returns the current kernel (weights) of the convolutional layer.

        :return: The kernel matrix of the layer.
        :rtype: np.ndarray
        """
        return self.weights

    def setWeights(self, weights):
        """
        Sets the kernel (weights) of the convolutional layer to the provided matrix.

        :param weights: A new kernel matrix for the layer.
        :type weights: np.ndarray
        """
        self.weights = weights

    def forward( self, dataIn ):
        """
        Applies the convolution operation to the input data using the current kernel weights,
        producing a feature map.

        :param dataIn: The input data to the convolutional layer.
        :return: The resulting feature map after applying the convolution operation.
        """
        # Store the input data for use in the backward pass
        self.setPrevIn(dataIn)

        # Compute the feature map using the cross-correlation function
        feature_map = self.crossCorrelate2D(dataIn, self.getWeights())

        # Store the output feature map for the backward pass
        self.setPrevOut(feature_map)

        # Return the feature map
        return feature_map

    def gradient( self ):
        """
        Computes the gradient of the loss function with respect to the kernel weights.
        This method is a placeholder and should be implemented as needed.

        :return: None
        """
        pass

    def backward( self, gradIn ):
        """
        Computes the gradient of the loss function with respect to the input data.
        This method is a placeholder and should be implemented as needed.

        :param gradIn: The gradient of the loss function with respect to the output of this layer.
        :type gradIn: np.ndarray
        :return: None
        """
        pass

    def updateWeights(self, gradIn,t, eta = 0.01):
        """
        Updates the kernel weights of the convolutional layer using the Adam optimization algorithm.

        :param gradIn: Gradient of the loss function with respect to the output of this layer.
        :param t: The current iteration or time step in the optimization process.
        :param eta: The learning rate.
        :type gradIn: np.ndarray
        :type t: int
        :type eta: float
        """
        # Apply the input gradient to the previous input data to produce the gradient of the loss function with respect to the weights
        dJdW = np.mean(self.crossCorrelate2D(self.getPrevIn(), gradIn), axis = 0)

        # First moment update
        self.s = (self.p1 * self.s) + ((1 - self.p1) * dJdW)

        # Second moment update
        self.r = (self.p2 * self.r) + ((1 - self.p2) * (dJdW ** 2))

        # Final update term
        update_term = (self.s / (1 - (self.p1 ** (t + 1)))) / (np.sqrt((self.r) / (1 - (self.p2 ** (t + 1)))) + self.delta)

        # Update weights
        self.setWeights(self.getWeights() - (eta * update_term))

    def crossCorrelate2D( self, dataIn, kernel):
        """
        Performs cross-correlation between the input data and the kernel (2D or 3D), producing a feature map.

        :param dataIn: The input data to the layer.
        :param kernel: The kernel (weights) to apply to the input data.
        :type dataIn: np.ndarray
        :type kernel: np.ndarray
        :return: The resulting feature map from the cross-correlation operation.
        :rtype: np.ndarray
        """
        # Determine the kernel dimensions
        kernel_height, kernel_width = kernel.shape[ -2 ], kernel.shape[ -1 ]

        # Prepare an output tensor with correctly calculated dimensions
        feature_maps = np.zeros((dataIn.shape[ 0 ], dataIn.shape[ 1 ] - kernel_height + 1, dataIn.shape[ 2 ] - kernel_width + 1))

        # Loop over each image in the tensor
        for b in range(feature_maps.shape[ 0 ]):

            # Loop over the rows of the image
            for i in range(feature_maps.shape[ 1 ]):

                # Loop over the columns of the image
                for j in range(feature_maps.shape[ 2 ]):

                    # Use the kernel for the corresponding image if 3D else use the kernel 2D matrix
                    current_kernel = kernel[ b ] if kernel.ndim == 3 else kernel
                    feature_maps[ b, i, j ] = np.sum(current_kernel * dataIn[ b, i:i + kernel_height, j:j + kernel_width ])

        return feature_maps



