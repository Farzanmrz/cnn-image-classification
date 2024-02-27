# Imports
import ConvolutionalLayer, MaxPoolLayer, FlatteningLayer, FullyConnectedLayer, LogisticSigmoidLayer, LogLoss
import numpy as np
import matplotlib.pyplot as plt

# Create synthetic data images for vertical and horizontal images
vimg = np.zeros((40, 40))
himg = np.zeros((40, 40))
vimg[ :, 4 ] = 1
himg[ 32, : ] = 1

# Create the input tensor
img = np.array([ vimg, himg ])

# Create target vector y
y = np.array([ [ 0 ], [ 1 ] ])

# Define layers
layers = [ ConvolutionalLayer.ConvolutionalLayer(9, 9),
           MaxPoolLayer.MaxPoolLayer(4, 4),
           FlatteningLayer.FlatteningLayer(),
           FullyConnectedLayer.FullyConnectedLayer(64, 1),
           LogisticSigmoidLayer.LogisticSigmoidLayer(),
           LogLoss.LogLoss() ]

# Store initial kernel
init_kernel = layers[ 0 ].getWeights()


def fprop( layers, x, y ):
	"""
	Performs forward propagation through the network, excluding the last layer,
	and calculates the loss using the last layer.

	This function propagates an input through all layers of the network except
	the last one, then uses the last layer to compute the loss based on the
	output of the preceding layer and the true labels.

	:param layers: List of layers in the CNN, configured for forward propagation.
	:param x: Input data to the network, numpy array of shape (n_samples, height, width).
	:param y: True labels for the input data, numpy array of shape (n_samples, 1).

	:return: A tuple containing the output from the last but one layer and the loss
			 calculated using the last layer.
	"""

	# Initialize activation with input x
	activation = x

	# Propagate through all layers except the last
	for layer in layers[ :-1 ]:
		activation = layer.forward(activation)

	# Calculate loss using the last layer
	loss = layers[ -1 ].eval(y, activation)

	# Return the result of last layer and loss
	return activation, loss


def bProp( layers, Y, h, t = 0 ):
	"""
	Performs backward propagation through the network for gradient calculation and weight updates.

	This function computes the gradient of the loss with respect to the weights of each layer in the network
	by propagating the error backwards. It updates the weights of the layers based on the computed gradients.
	The process begins from the last layer and proceeds through each layer up to the first, updating weights
	as necessary.

	:param layers: A list of layers in the network, including convolutional, fully connected, and loss layers.
	:param Y: The true labels for the input data, numpy array of shape (n_samples, 1).
	:param h: The output from the last layer during forward propagation, numpy array of shape (n_samples, n_classes).
	:param t: The current epoch or iteration number, used for updating weights with time-dependent learning rates.

	Note: This function updates the weights of the network in-place and does not return any value.
	"""

	# Calculate initial gradient based on the loss
	grad = layers[ -1 ].gradient(Y, h)

	# Iterate backwards through layers, skipping the first layer
	for i in range(len(layers) - 2, 0, -1):

		# Calculate new gradient for the current layer
		newgrad = layers[ i ].backward(grad)

		# Update weights if the layer is a FullyConnectedLayer
		if isinstance(layers[ i ], FullyConnectedLayer.FullyConnectedLayer):
			layers[ i ].updateWeights(grad, t)

		# Update gradient
		grad = newgrad

	# Update gradient for final Convolutional layer
	layers[ 0 ].updateWeights(grad, t)


def runCNN( layers, img, y ):
	"""
	Trains a Convolutional Neural Network on a given dataset and returns the loss history and learned kernel weights.

	The function performs forward propagation to calculate the loss, then backward propagation to update the weights
	of the layers. It prints the loss at specified intervals and checks for convergence by comparing the change in loss
	across epochs. Training stops when the model converges or when the maximum number of epochs is reached.

	:param layers: A list of layers that constitute the CNN. Each layer should support forward and backward operations.
	:type layers: List[Layer]
	:param img: The input data (images) on which the CNN will be trained. It should be a numpy array of shape (n, height, width),
				where n is the number of images.
	:type img: numpy.ndarray
	:param y: The target outputs (labels) for the input data. It should be a numpy array of shape (n, 1), where n is the number of images.
	:type y: numpy.ndarray

	:return: A tuple containing two elements:
			 - The first element is a list of loss values recorded at each epoch during training.
			 - The second element is the numpy array representing the weights of the first convolutional layer after training.
	:rtype: (list, numpy.ndarray)
	"""
	fin_l = [ ]
	prev = 0

	# Loop through epochs
	for epoch in range(5000):

		# Forward propagation
		yhat, loss = fprop(layers, img, y)
		fin_l.append(loss)

		# Backward propagation
		bProp(layers, y, yhat, epoch)

		# Print loss every 500 epochs
		if epoch % 500 == 0:
			print("Epoch: ", epoch + 1)
			print("Loss: ", loss)

		# Terminating condition
		if np.abs(loss - prev) < 1e-7:
			print("Converged at epoch: ", epoch + 1)
			print("Loss: ", loss)
			break

		prev = loss

	return fin_l, layers[ 0 ].getWeights()


# Train the CNN
loss, fin_kern = runCNN(layers, img, y)

# Visualize the images
fig, ax = plt.subplots(1, 2, figsize = (10, 5))
ax[ 0 ].imshow(vimg, cmap = 'gray')
ax[ 1 ].imshow(himg, cmap = 'gray')
ax[ 0 ].axis('off')
ax[ 1 ].axis('off')
ax[0].set_title('Vertical Image')
ax[1].set_title('Horizontal Image')
plt.savefig("2a.png")

# Plot initial and final kernels
fig, ax = plt.subplots(1, 2, figsize = (10, 5))
ax[ 0 ].imshow(init_kernel, cmap = 'gray')
ax[ 1 ].imshow(fin_kern, cmap = 'gray')
ax[ 0 ].axis('off')
ax[ 1 ].axis('off')
ax[0].set_title('Initial Kernel')
ax[1].set_title('Final Kernel')
plt.savefig("2b.png")
plt.show()

# Plot the log loss against epochs
epochs = list(range(1, len(loss) + 1))
plt.figure(figsize = (10, 5))
plt.plot(range(1, len(loss) + 1), loss, linestyle='-', color='b')
plt.scatter([1, len(loss)], [loss[0], loss[-1]], color='red', zorder=5)  # Ensure points are on top
plt.text(1 + 250, loss[0] - 0.02, f'({1}, {loss[0]})', color='red', fontsize=14, ha='right')
plt.text(len(loss) - 300, loss[-1] + 0.05, f'({len(loss)}, {loss[-1]})', color='red', fontsize=14, ha='left')
plt.title('Log Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Log Loss')
plt.savefig("2c.png")
plt.show()
