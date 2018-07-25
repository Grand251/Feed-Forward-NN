# Feed-Forward-NN

Usage:

ffnn = FFNN()

ffnn.build_model(X, Y, layer_info, num_passes)

ffnn.predict(test)


X: A Numpy array of training instances

Y: A Numpy array of training labels

layer_info: A list describing the number of neurons in each layer starting with the input layer. The size of the input layer must match the number of features in the training/test instances
  ex: [5, 10, 10, 2] Describes a network with 5 input neurons, two hidden layers each with 10 neurons and an output layer of 2 neurons.

num_passes: Number of times to run through the training data during training

test: A Numpy array of test instances.


build_model() returns the squared error of the entire training set after training is complete.

predict() returns a Numpy array of predicted labels of the same length of the number of tested instances
