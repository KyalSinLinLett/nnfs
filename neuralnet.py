import numpy as np

def sigmoid(x):
	'''
	sigmoid activation function that will be applied t the weighted sum 'x' to produce an output
	'''
	return 1.0 / (1.0 + np.exp(-x))

def compute_weighted_sum(inputs, weights, bias):
	'''
	calculates the dot product of the inputs and the weights plus the bias value
	'''
	return np.sum(inputs * weights) + bias

def initialize_network(num_inputs, num_hidden_layers, num_nodes_hidden, num_nodes_output):
	'''
	Function to initialize a neural network structure with random weights and biases
	num_inputs -> (int) -> number of inputs
	num_hidden_layers -> (int) -> number of hidden layers
	num_nodes_hidden -> (list) -> number of nodes in each hidden layer
	num_nodes_output -> (int) -> number of nodes in the output layer

	will return a logical neural net structure as a python dict
	'''
	
	num_nodes_previous = num_inputs # number of nodes in the previous layer
	
	network = {}

	# loop thru each layer and randomly initialize the weights and biases
	for layer in range(num_hidden_layers + 1): # will iterate until the last layer(output layer/node)

		if layer == num_hidden_layers:
			layer_name = "output" # this condition is true only for the last layer in the network
			num_nodes = num_nodes_hidden[layer] # number of nodes in this specific layer

		else:
			layer_name = f"layer_{layer+1}"
			num_nodes = num_nodes_hidden[layer] # number of nodes in this specific layer

		# at this point a layer is defined, the num of nodes in that layer is also defined

		# Now, initialize the weights and biases of each node at the hidden layer
		network[layer_name] = {} # adding the hidden layer to the network structure as a dict which will contain the weights and biases of each node
		for node in range(num_nodes): # iter thru each node in the hidden layer
			
			node_name = f"node_{node+1}"
			
			network[layer_name][node_name] = {
				'weights': np.around(np.random.uniform(size=num_nodes_previous), decimals=2), # the size param is to determine how many synaptic weight/synapses that the node has
				'bias': np.around(np.random.uniform(size=1), decimals=3)
			}

		num_nodes_previous = num_nodes

	return network # return the network structure

def forward_propagate(network, inputs):
	'''
	propagates from the inputs to the output layer and outputs a prediction for each node
	in the output layer
	network -> (dict) -> the network structure
	inputs -> (np.array()) -> the inputs length must match the 'num_inputs' when the network was initialized.
	'''

	layer_inputs = list(inputs) # set the inputs of the nodes for the layers

	for layer in network: # iterate through the layers in the network

		layer_data = network[layer] # will set the layer data as the dict that contains the nodes, weights, biases of each of the nodes
		
		layer_outputs = [] # list to store all the outputs from each nodes in the layer

		for layer_node in layer_data: # iterate through each node in the layer

			node_data = layer_data[layer_node] # set the node data (weights and bias)

			# calculate the output of a node in this layer
			node_output = sigmoid(compute_weighted_sum(layer_inputs, node_data['weights'], node_data['bias']))

			layer_outputs.append(np.around(node_output[0], decimals=4))

		if layer != 'output':
			print(f"The outputs of the nodes in hidden layer number {layer.split('_')[1]} is {layer_outputs}.")

	# once the output layer is reached, the layer_outputs list will be assigned with the output of the output layer
	network_predictions = layer_outputs
	
	return network_predictions


