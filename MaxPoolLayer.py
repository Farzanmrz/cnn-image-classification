import numpy as np
from Layer import Layer

class MaxPoolLayer(Layer):
    def __init__(self, width, stride):
        """
        Initialize the max pooling layer with specified width and stride.
        The max pooling operation slides a window across the input data taking the maximum value
        in the window at each step.

        :param width: Width (and height) of the pooling window.
        :param stride: Distance between successive pooling windows.
        """
        super().__init__()
        self.width = width  # Width of the pooling window
        self.stride = stride  # Stride between successive pooling windows

    def forward(self, dataIn):

        self.setPrevIn(dataIn)  # Store the input data for use in the backward pass

        # Calculate dimensions of the output feature map
        dim0 = dataIn.shape[ 0 ]
        dim1 = int(np.floor(((dataIn.shape[1] - self.width) / self.stride) + 1))
        dim2 = int(np.floor(((dataIn.shape[2] - self.width) / self.stride) + 1))

        # Initialize the output tensor with zeros
        y = np.zeros((dim0, dim1, dim2))

        # Loop through each tensor
        for h in range(dim0):
            for i in range(dim1):
                for j in range(dim2):
                    # Perform max pooling operation on the window of data
                    y[h, i, j] = np.max(
                        dataIn[h,
                            i * self.stride: i * self.stride + self.width,
                            j * self.stride: j * self.stride + self.width,
                        ]
                    )

        self.setPrevOut(y)  # Store the output data for use in subsequent layers
        return y

    def gradient(self, gradIn):

        fmap = self.getPrevIn()  # Retrieve the input feature map from the forward pass

        # Initialize the gradient array with zeros, same size as the input feature map
        grad = np.zeros_like(fmap)

        for h in range(gradIn.shape[0]):
            for i in range(gradIn.shape[1]):
                for j in range(gradIn.shape[2]):

                    # Extract the window of data that was used in the forward pass
                    window = fmap[h,
                        i * self.stride: i * self.stride + self.width,
                        j * self.stride: j * self.stride + self.width,
                    ]

                    # Find the index of the maximum value in the window
                    max_val_index = np.unravel_index(np.argmax(window, axis=None), window.shape)

                    # Assign the gradient to the position of the maximum value
                    grad[h,
                        i * self.stride + max_val_index[0],
                        j * self.stride + max_val_index[1],
                    ] = gradIn[h, i, j]

        return grad

    def backward(self, gradIn):

        return self.gradient(gradIn)
