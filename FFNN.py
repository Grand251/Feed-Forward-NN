import numpy as np

class FFNN:
	def predict(self, X):
	
		weights = self.model['weights']
		bias = self.model['bias']

		num_layers = len(weights)

		#0 Will account for input layer
		outs = list(range(num_layers + 1))
		nets = list(range(num_layers + 1))
		outs[0] = X
		nets[0] = X
		layer_input = X
	
		for i in range(1, num_layers + 1):
			nets[i] = layer_input.dot(weights[i - 1]) + bias[i - 1]
			outs[i] = np.tanh(nets[i])
			layer_input = outs[i]
	
		return layer_input


	def build_model(self, X, Y, layer_info, num_passes):
		#np.random.seed(0)

		#Not counting hidden layer
		num_layers = len(layer_info) - 1
		weights = list(range(num_layers))
		bias = list(range(num_layers))

		for i in range(num_layers):
			weights[i] = (np.random.rand(layer_info[i], layer_info[i + 1]) * 2) - 1 
			bias[i] = (np.random.rand(layer_info[i + 1]) * 2) - 1 
	
		#0 Will account for input layer
		outs = list(range(num_layers + 1))
		nets = list(range(num_layers + 1))
		outs[0] = X
		nets[0] = X
	
		for it in range(num_passes):
			layer_input = X
			for i in range(1, num_layers + 1):
				nets[i] = layer_input.dot(weights[i - 1]) + bias[i - 1]
				outs[i] = np.tanh(nets[i])
				layer_input = outs[i]
	
			#Backpropagation
			learning_rate = 0.1
			error = layer_input - Y
	
			dWeights = list(range(num_layers))
			dBias = list(range(num_layers))
			delta = list(range(num_layers))
	
			delta[num_layers - 1] = error
	
			for i in list(range(num_layers))[::-1]:
				dWeights[i] = (outs[i].T).dot(delta[i])
				dBias[i] = np.sum(delta[i], axis=0)
				if i > 0:
					delta[i -1] = delta[i].dot(weights[i].T) * (1 - np.power(outs[i], 2))
			for i in range(num_layers):
				weights[i] -= learning_rate * dWeights[i]
				bias[i] -= learning_rate * dBias[i]
		
		#Save weights and bias to the object
		self.model = {'weights': weights, 'bias': bias}
		
		#Find the total squared error with respect to the whole training set
		predictions = self.predict(X)
		prediction_error = np.sum(np.square(predictions - Y))
		
		return prediction_error


