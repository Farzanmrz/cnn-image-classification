import numpy as np
from Layer import Layer

class ConvolutionalLayer(Layer):
    """
    A Convolutional Layer used in Convolutional Neural Networks (CNNs) for processing
    image-like input data. This layer applies a set of learnable filters to the input
    to create feature maps.

    :param kh: Height of the kernel (filter).
    :param kw: Width of the kernel (filter).
    """

    def __init__(self, kh, kw):
        """
        Initializes the Convolutional Layer with specified kernel size.

        :param kh: Kernel height.
        :param kw: Kernel width.
        """
        super().__init__()

        # Initialize weights with small random values. The weights represent the kernel.
        self.weights = np.random.uniform(-1e-4, 1e-4, (kh, kw))

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

    def forward(self, dataIn):

        self.setPrevIn(dataIn)

        # Get the feature map from the input data and weights matrix
        feature_map = self.crossCorrelate2D(dataIn, self.getWeights())


        self.setPrevOut(feature_map)

        return feature_map


    def gradient(self):
        """
        This method should compute the gradient of the loss function with respect to
        the kernel weights. It is left unimplemented here.

        :return: None
        """
        pass

    def backward(self, gradIn):
        """
        Performs the backward pass through the convolutional layer, computing the gradient
        of the loss function with respect to the input data. It is left unimplemented here.

        :param gradIn: Gradient of the loss function with respect to the output of the layer.
        :type gradIn: np.ndarray
        :return: None
        """
        pass

    def updateWeights(self, gradIn, eta = 0.01):

        # Apply the gradIn as kernel to the previous input data to produce djdw
        djdw = self.crossCorrelate2D(self.getPrevIn(), gradIn)

        # Update weights
        self.setWeights(self.getWeights() - (eta * np.mean(djdw,axis = 0)))

    def crossCorrelate2D( self, dataIn, kernel):

        # Get kernel height and width for 2D and 3D kernels
        if kernel.ndim == 2:
            kernel_height, kernel_width = kernel.shape
        elif kernel.ndim == 3:
            _, kernel_height, kernel_width = kernel.shape

        # Throw error if kernel is not 2D or 3D
        else:
            raise ValueError("Kernel must be either 2D or 3D.")

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



