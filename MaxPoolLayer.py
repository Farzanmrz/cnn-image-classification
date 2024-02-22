import numpy as np

from Layer import Layer


class MaxPoolLayer(Layer):
    def __init__(self, width, stride):
        """
        Initialize the max pooling layer.

        """
        super().__init__()

        # Intialize the window width and stride
        self.width = width
        self.stride = stride

    def forward(self, dataIn):
        """
        Perform forward pass and set previous input and output.

        :param dataIn: Incoming feature map.
        :type dataIn: np.ndarray

        :return: Output data after applying logistic sigmoid function.
        :rtype: np.ndarray
        """

        self.setPrevIn(dataIn)

        # Calculate dimensions of the output
        dim1 = int(np.floor(((dataIn.shape[0] - self.width) / self.stride) + 1))
        dim2 = int(np.floor(((dataIn.shape[1] - self.width) / self.stride) + 1))

        # Initialize the output
        y = np.zeros((dim1, dim2))
        for i in range(dim1):
            for j in range(dim2):
                y[i, j] = np.max(
                    dataIn[
                        i * self.stride : i * self.stride + self.width,
                        j * self.stride : j * self.stride + self.width,
                    ]
                )

        self.setPrevOut(y)
        return y

    def gradient(self):
        """
        Compute gradients for the logistic sigmoid layer.

        :return: Gradients for the logistic sigmoid layer.
        :rtype: np.ndarray
        """

        # Compute gradient using output of the layer
        y_hat = self.getPrevOut()
        return y_hat * (1 - y_hat)

    def backward(self, gradIn):
        """
        Perform backward pass for the logistic sigmoid layer.

        :param gradIn: Gradient of the loss with respect to the output of this layer.
        :type gradIn: np.ndarray

        :return: Gradient of the loss with respect to the input of this layer.
        :rtype: np.ndarray
        """

        # Compute gradient of the loss with respect to the input using chain rule
        return self.gradient() * gradIn


x = np.array(
    [
        [4, 7, 1, 7, 2, 3],
        [6, 3, 5, 6, 4, 2],
        [6, 5, 6, 4, 3, 7],
        [4, 2, 5, 2, 5, 0],
        [5, 6, 6, 2, 5, 3],
        [2, 1, 2, 3, 2, 3]
    ]
)


lay = MaxPoolLayer(3, 3)

y = lay.forward(x)
print(y)
