import numpy as np
import confusion_matrix

class LeastSquaresMultiClassClassifier:
    """
    """
    def __init__(self, train_input, train_label, class_list, weights = None):
        """
        """
        self.train_input = train_input
        self.class_key = {}
        self.class_list = class_list
        self.k_classes = len(class_list)
        for i in range(self.k_classes):
            self.class_key[class_list[i]] = i
        self.trains_labels = np.zeros(len(train_label))
        for i in range(len(train_label)):
            self.trains_labels[i] = self.class_key[train_label[i]]
        self.trains_labels = self.trains_labels.astype(int)
        self.train_right_output = np.ones((len(train_label), self.k_classes))
        for i in range(len(train_label)):
            for j in range(self.k_classes):
                if j != self.trains_labels[i]:
                    self.train_right_output[i][j] = -1.
        if weights is None:
            self.weights = np.linalg.pinv(self.train_input)@self.train_right_output
        else:
            self.weights = weights
        self.test_labels = None
        self.test_prediction = None
        self.confusion_matrix = None


    def test(self, test_input, test_label):
        """
        """
        self.test_labels = np.zeros(len(test_label))
        for i in range(len(test_label)):
            self.test_labels[i] = self.class_key[test_label[i]]
        self.test_labels = self.test_labels.astype(int)
        test_output = test_input@self.weights
        self.test_prediction = np.argmax(test_output, axis=1)
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


    def export_weights(self, path):
        """
        """
        np.savetxt(path, self.weights)
