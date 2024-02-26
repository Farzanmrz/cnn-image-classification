# Imports
import ConvolutionalLayer, MaxPoolLayer, FlatteningLayer, FullyConnectedLayer, LogisticSigmoidLayer, LogLoss
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

# Define layers
layers = [ConvolutionalLayer.ConvolutionalLayer(9, 9), MaxPoolLayer.MaxPoolLayer(4, 4),
          FlatteningLayer.FlatteningLayer(), FullyConnectedLayer.FullyConnectedLayer(64, 1),
          LogisticSigmoidLayer.LogisticSigmoidLayer(), LogLoss.LogLoss()]

# Store initial kernel
init_kernel = layers[ 0 ].getWeights()

# Create synthetic data images for vertical and horizontal images
vimg = np.zeros((40, 40))
himg = np.zeros((40, 40))
vimg[ :, 4 ] = 1
himg[ 32, : ] = 1

# Create the input tensor
img =np.array([vimg, himg])

# Create target vector y
y = np.array([[0],[1]])


# Function to run the CNN
def runCNN(layers, img, y):

    fin_l = []

    prev = 0

    for epoch in range(1000):
    ############ Forward pass through the convolutional layer ############


        h = img
        # Convolutional layer
        fmap = layers[ 0 ].forward(h)

        # Max pooling layer
        z = layers[ 1 ].forward(fmap)

        # Flattening layer
        h1 = layers[ 2 ].forward(z)

        # Fully connected layer
        h2 = layers[ 3 ].forward(h1)

        # Logistic sigmoid layer
        yhat = layers[ 4 ].forward(h2)

        # Log loss layer
        loss = layers[ 5 ].eval(y, yhat)
        fin_l.append(loss)

        ############ Backward pass through the convolutional layer ############

        # Log loss layer
        ll_grad = layers[ 5 ].gradient(y, yhat)

        # Logistic sigmoid layer
        ls_grad = layers[ 4 ].backward(ll_grad)


        # Fully connected layer
        fc_grad = layers[ 3 ].backward(ls_grad)


        layers[3].updateWeights(ls_grad)

        # Flattening layer
        fl_grad = layers[ 2 ].backward(fc_grad)

        # Max pooling layer
        grad = layers[ 1 ].backward(fl_grad)

        # Convolutional layer update weights
        layers[ 0 ].updateWeights(grad)

        if epoch % 100 == 0:
            print("Epoch: ", epoch + 1)
            print("Loss: ", loss)

        if epoch > 1000 and np.abs(loss - prev) < 1e-10:
            print("Converged at epoch: ", epoch + 1)
            print("Loss: ", loss)
            break

        prev = loss


    return fin_l, layers[ 0 ].getWeights()




loss, res = runCNN(layers, img, y)



fig, ax = plt.subplots(2, 2, figsize = (10, 5))
ax[1][0].imshow(init_kernel, cmap = 'gray')
ax[1][1].imshow(res, cmap = 'gray')
ax[ 0 ][0].imshow(vimg, cmap = 'gray')
ax[ 0 ][1].imshow(himg, cmap = 'gray')
ax[ 0 ][0].axis('off')
ax[ 1 ][0].axis('off')
ax[ 0 ][1].axis('off')
ax[ 1 ][1].axis('off')
plt.show()

