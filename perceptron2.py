import numpy as np

class Perceptron():

	'''
	A simple perceptron(neuron) that does simple forward and backwards propagation for learning
	'''

	def __init__(self):
		np.random.seed(1)
		self.synaptic_weights = 2 * np.random.random((3,1)) - 1

	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def sigmoid_derivative(self, x):
		return x * (1 - x)

	def train(self, training_inputs, training_outputs, iterations):

		for i in range(iterations):
			output = self.predict(training_inputs)
			error = training_outputs - output
			adjustments = np.dot(training_inputs.T, error * self. sigmoid_derivative(output))
			self.synaptic_weights += adjustments

	def predict(self, inputs):
		inputs = inputs.astype(float)
		output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
		return output

if __name__ == "__main__":

	# some simple training inputs
	training_inputs = np.array([[0,0,1], 
								[1,1,1], 
								[1,0,1], 
								[0,1,1]])

	# outputs of those training inputs
	training_outputs = np.array([[0,1,1,0]]).T

	p1 = Perceptron()

	print("Random synaptic weights:")
	print(p1.synaptic_weights)

	p1.train(training_inputs, training_outputs, 50000)

	print("Synaptic weights after training: ")
	print(p1.synaptic_weights)

	A = str(input("Input 1: "))
	B = str(input("Input 2: "))
	C = str(input("Input 3: "))

	print("New situation: input data = ", A, B, C)
	print("Output data: ")
	print(p1.predict(np.array([[A, B, C]])))