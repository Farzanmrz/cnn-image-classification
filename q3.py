# Imports
import ConvolutionalLayer, MaxPoolLayer, FlatteningLayer, FullyConnectedLayer, LogisticSigmoidLayer, LogLoss
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

# Define layers
layers = [ConvolutionalLayer.ConvolutionalLayer(9, 9), MaxPoolLayer.MaxPoolLayer(4, 4),
          FlatteningLayer.FlatteningLayer(), FullyConnectedLayer.FullyConnectedLayer(64, 64),
          LogisticSigmoidLayer.LogisticSigmoidLayer(), LogLoss.LogLoss()]

# Store initial kernel
init_kernel = layers[ 0 ].getWeights()

# Create synthetic data images for vertical and horizontal images
vimg = np.zeros((40, 40))
himg = np.zeros((40, 40))
vimg[ :, 4 ] = 1
himg[ 32, : ] = 1

# Visualize the synthetic data images
# fig, ax = plt.subplots(1, 2, figsize = (10, 5))
# ax[ 0 ].imshow(vimg, cmap = 'gray')
# ax[ 1 ].imshow(himg, cmap = 'gray')
# ax[ 0 ].axis('off')
# ax[ 1 ].axis('off')
# plt.show()

# Function to run the CNN
def runCNN(img, layers):
    ############ Forward pass through the convolutional layer ############

    # Convolutional layer
    fmap = layers[ 0 ].forward(img)

    # Max pooling layer
    z = layers[ 1 ].forward(fmap)

    # Flattening layer
    h1 = layers[ 2 ].forward(z)

    # Fully connected layer
    h2 = layers[ 3 ].forward(h1).T

    # Logistic sigmoid layer
    yhat = layers[ 4 ].forward(h2)

    # Log loss layer


    return yhat

print(runCNN(vimg, layers))
