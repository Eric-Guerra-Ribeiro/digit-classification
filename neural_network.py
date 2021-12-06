import numpy as np
import confusion_matrix

def sigmoid(x):
    """
    Sigmoid function.

    :param x: input to the function.
    :type x: float or numpy matrix.
    :return: output of the sigmoid function evaluated at x.
    :rtype x: float or numpy matrix.
    """
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_derivative(x):
    """
    Sigmoid function derivative.

    :param x: input to the function.
    :type x: float or numpy matrix.
    :return: derivative of the sigmoid function evaluated at x.
    :rtype: float or numpy matrix.
    """
    return sigmoid(x) * (1.0 - sigmoid(x))


class NeuralNetwork:
    """
    Represents a two-layers Neural Network (NN) for multi-class classification.
    The sigmoid activation function is used for all neurons.
    """
    def __init__(self, train_input, train_labels, class_list, num_epochs, num_hiddens, alpha=1.0,
                 weights_lay1=None, weights_lay2=None, bias_lay1=None, bias_lay2=None):
        """
        Constructs a three-layers Neural Network.

        :param num_inputs: number of inputs of the NN.
        :type num_inputs: int.
        :param num_hiddens: number of neurons in the hidden layer.
        :type num_hiddens: int.
        :param k_classes: number of outputs of the NN.
        :type k_classes: int.
        :param alpha: learning rate.
        :type alpha: float.
        """
        self.train_input = train_input
        self.num_inputs = np.shape(train_input)[0]
        self.num_hiddens = num_hiddens
        self.class_key = {}
        self.class_list = class_list
        self.k_classes = len(class_list)
        for i in range(self.k_classes):
            self.class_key[class_list[i]]  = i
        self.train_labels = np.zeros(len(train_labels))
        for i in range(len(train_labels)):
            self.train_labels[i] = self.class_key[train_labels[i]]
        self.train_labels = self.train_labels.astype(int)
        self.expected_outputs = np.ones((self.k_classes, len(train_labels)))
        for i in range(self.k_classes):
            for j in range(len(train_labels)):
                if i != self.train_labels[j]:
                    self.expected_outputs[i][j] = -1.
        self.num_epochs = num_epochs
        self.alpha = alpha
        self.weights = [None] * 3
        self.biases = [None] * 3
        if weights_lay1 is None:
            self.weights[1] = 0.001 * np.random.randn(num_hiddens, self.num_inputs)
        else:
            self.weights[1] = weights_lay1
        if weights_lay2 is None:
            self.weights[2] = 0.001 * np.random.randn(self.k_classes, num_hiddens)
        else:
            self.weights[2] = weights_lay2
        if bias_lay1 is None:
            self.biases[1] = np.zeros((num_hiddens, 1))
        else:
            self.biases[1] = bias_lay1
        if bias_lay2 is None:
            self.biases[2] = np.zeros((self.k_classes, 1))
        else:
            self.biases[2] = bias_lay2
        self.costs = np.zeros(self.num_epochs)
        self.test_labels = None
        self.test_prediction = None
        self.confusion_matrix = None


    def forward_propagation(self, inputs):
        """
        Executes forward propagation.
        Notice that the z and a of the first layer (l = 0) are equal to the NN's input.

        :param inputs: inputs to the network.
        :type inputs: (num_inputs, num_samples) numpy array.
        :return z: values computed by applying weights and biases at each layer of the NN.
        :rtype z: 3-dimensional list of (num_neurons[l], num_samples) numpy matrices.
        :return a: activations computed by applying the activation function to z at each layer.
        :rtype a: 3-dimensional list of (num_neurons[l], num_samples) numpy matrices.
        """
        z = [None] * 3
        a = [None] * 3
        z[0] = inputs
        a[0] = inputs
        for i in range(1,3):
            z[i] = self.weights[i] @ a[i-1] + self.biases[i]
            a[i] = sigmoid(z[i])
        return z, a


    def compute_cost(self, inputs, expected_outputs):
        """
        Computes the logistic regression cost of this network.

        :param inputs: inputs to the network.
        :type inputs: (num_inputs, num_samples) numpy array.
        :param expected_outputs: expected outputs of the network.
        :type expected_outputs: list of numpy matrices.
        :return: logistic regression cost.
        :rtype: float.
        """
        z, a = self.forward_propagation(inputs)
        y = expected_outputs
        y_hat = a[-1]
        cost = np.mean(-(y * np.log(y_hat) + (1.0 - y) * np.log(1.0 - y_hat)))
        return cost


    def compute_gradient_back_propagation(self, inputs, expected_outputs):
        """
        Computes the gradient with respect to the NN's parameters using back propagation.

        :param inputs: inputs to the network.
        :type inputs: (num_inputs, num_samples) numpy array.
        :param expected_outputs: expected outputs of the network.
        :type expected_outputs: (k_classes, num_samples) numpy array.
        :return weights_gradient: gradients of the weights at each layer.
        :rtype weights_gradient: 3-dimensional list of numpy arrays.
        :return biases_gradient: gradients of the biases at each layer.
        :rtype biases_gradient: 3-dimensional list of numpy arrays.
        """
        weights_gradient = [None] * 3
        biases_gradient = [None] * 3
        deltas = [None] * 3
        z, a = self.forward_propagation(inputs)
        y_hat = a[-1]
        y = expected_outputs
        num_samples = np.shape(y_hat)[1]
        deltas[2] = y_hat - y
        deltas[1] = (np.transpose(self.weights[2])@deltas[2])*sigmoid_derivative(z[1])
        for i in range(2,0,-1):
            biases_gradient[i] = np.mean(deltas[i], axis = 1, keepdims = True)
            weights_gradient[i] = deltas[i] @ np.transpose(a[i-1])/num_samples
        return weights_gradient, biases_gradient


    def back_propagation(self, inputs, expected_outputs):
        """
        Executes the back propagation algorithm to update the NN's parameters.

        :param inputs: inputs to the network.
        :type inputs: (num_inputs, num_samples) numpy array.
        :param expected_outputs: expected outputs of the network.
        :type expected_outputs: (k_classes, num_samples) numpy array.
        """
        weights_gradient, biases_gradient = self.compute_gradient_back_propagation(inputs, expected_outputs)
        for i in range(2, 0, -1):
            self.weights[i] -= self.alpha*weights_gradient[i]
            self.biases[i] -= self.alpha*biases_gradient[i]


    def train(self):
        """
        """
        for i in range(self.num_epochs):
            self.back_propagation(self.train_input, self.expected_outputs)
            self.costs[i] = self.compute_cost(self.train_input, self.expected_outputs)
            print("Epoch: {:4d}, Cost: {}".format(i+1, self.costs[i]))


    def test(self, test_input, test_label):
        """
        """
        self.test_labels = np.zeros(len(test_label))
        for i in range(len(test_label)):
            self.test_labels[i] = self.class_key[test_label[i]]
        self.test_labels = self.test_labels.astype(int)
        _, test_output = self.forward_propagation(test_input)
        test_output = test_output[2]
        self.test_prediction = np.argmax(test_output, axis=0)
        self.test_prediction = self.test_prediction.astype(int)
        self.confusion_matrix = None


    def construct_confusion_matrix(self):
        """
        """
        if self.test_labels is None:
            print("No test data found.")
            return
        self.confusion_matrix = confusion_matrix.ConfusionMatrix(self.test_labels, self.test_prediction, self.k_classes)


    def show_confusion_matrix(self, tag="", title="Confusion Matrix"):
        """
        """
        if self.confusion_matrix is None:
            self.construct_confusion_matrix()
            if self.confusion_matrix is None:
                return
        self.confusion_matrix.plot_confusion_matrix(self.class_list, tag=tag, title=title)


    def export_weights_bias(self, path):
        """
        """
        np.savetxt(path+"weights_lay1.gz", self.weights[1])
        np.savetxt(path+"weights_lay2.gz", self.weights[2])
        np.savetxt(path+"bias_lay1.gz", self.biases[1])
        np.savetxt(path+"bias_lay2.gz", self.biases[2])
