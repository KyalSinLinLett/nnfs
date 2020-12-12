import numpy as np

# the sigmoid function
def sigmoid(x):
	return 1 / (1 + np.exp(-x))

# the derivative of the sigmoid
def sigmoid_derivative(x):
	return x * (1 - x)

# some simple training inputs
training_inputs = np.array([[0,0,1], 
							[1,1,1], 
							[1,0,1], 
							[0,1,1]])

# outputs of those training inputs
training_outputs = np.array([[0,1,1,0]]).T

np.random.seed(1)

# setting weights of the synapses
synaptic_weights = 2 * np.random.random((3,1)) - 1

print("Random starting synaptic weights: ")
print(synaptic_weights)

for i in range(10000):
	input_layer = training_inputs

	# produce an output from the preceptron
	outputs = sigmoid(np.dot(input_layer, synaptic_weights))
	# calculate the error
	error = training_outputs - outputs
	# adjustments needed to apply to our synaptic weights
	adjustments = error * sigmoid_derivative(outputs)
	# adjust our weights
	synaptic_weights += np.dot(input_layer.T, adjustments)

print("Synaptic weights after training: ")
print(synaptic_weights)

print("Outputs after training: ")
print(outputs)