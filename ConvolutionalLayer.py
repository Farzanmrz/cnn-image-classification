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
        self.kh = kh  # Height of the kernel
        self.kw = kw  # Width of the kernel

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
        """
        Performs the forward pass of the convolutional layer using the input data.
        This involves applying the kernel to the input data to produce a feature map.

        :param dataIn: Input data to the convolutional layer.
        :type dataIn: np.ndarray
        :return: The feature map output of the layer after applying the kernel.
        :rtype: np.ndarray
        """
        self.setPrevIn(dataIn)

        # Declare array to store feature map of each image
        feature_maps = []

        # Loop through each image in tensor
        for i in range(dataIn.shape[ 0 ]):

            # Apply the kernel to the input data to produce a feature map
            feature_map = self.crossCorrelate2D(dataIn[ i ], self.getWeights())

            # Append the feature map to the list of feature maps
            feature_maps.append(feature_map)

        self.setPrevOut(np.array(feature_maps))

        return np.array(feature_maps)


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
        """
        Updates the kernel weights of the convolutional layer using the gradient of the
        loss function with respect to the output of the layer. This method assumes the use
        of the ADAM optimization algorithm but is not fully implemented.

        :param gradIn: Gradient of the loss function with respect to the output of the layer.
        :param t: Current iteration number (epoch).
        :param eta: Learning rate.
        :type gradIn: np.ndarray
        :type t: int
        :type eta: float
        """
        # Declare array to store feature map of each image
        feature_maps = []

        prev_input = self.getPrevIn()

        # Loop through each image in tensor
        for i in range(gradIn.shape[ 0 ]):

            # Apply the kernel to the input data to produce a feature map
            feature_map = self.crossCorrelate2D(prev_input[i], gradIn[i])

            # Append the feature map to the list of feature maps
            feature_maps.append(feature_map)

        feature_maps = np.array(feature_maps)
        # Normalize the accumulated gradients by the batch size

        feature_maps = feature_maps / feature_maps.shape[0]

        self.setWeights(self.getWeights() - (eta * np.mean(feature_maps,axis = 0)))

    def crossCorrelate2D( self, dataIn, kernel):

        dim1 = dataIn.shape[ 0 ] - kernel.shape[ 0 ] + 1
        dim2 = dataIn.shape[ 1 ] - kernel.shape[ 1 ] + 1
        feature_map = np.zeros((dim1, dim2))
        for i in range(dim1):
            for j in range(dim2):
                feature_map[ i, j ] = np.sum(kernel * dataIn[ i:i + kernel.shape[ 0 ], j:j + kernel.shape[ 1 ] ])

        return feature_map



