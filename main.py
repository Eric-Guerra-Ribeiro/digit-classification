import numpy as np
from least_squares import LeastSquaresMultiClassClassifier

if __name__ == "__main__":
    train_input = np.loadtxt("MNIST\\train_input.gz", dtype=float)
    train_input_bias = np.loadtxt("MNIST\\train_input_bias.gz", dtype=float)
    train_label = np.loadtxt("MNIST\\train_label.gz", dtype=str)
    test_input = np.loadtxt("MNIST\\test_input.gz" , dtype=float) 
    test_input_bias = np.loadtxt("MNIST\\test_input_bias.gz" , dtype=float) 
    test_label = np.loadtxt("MNIST\\test_label.gz" , dtype=str)
    class_list = np.loadtxt("MNIST\\class_list.gz" , dtype=str)
    weights = np.loadtxt("MNIST\\weights.gz", dtype=float)
    weights_bias = np.loadtxt("MNIST\\weights_bias.gz", dtype=float)

    least_squares_classifier = LeastSquaresMultiClassClassifier(train_input, train_label, class_list, weights=weights)
    least_squares_classifier.test(test_input, test_label)
    least_squares_classifier.show_confusion_matrix(tag="teste", title="Least Squares Confusion Matrix Test Set without Bias")
    least_squares_classifier.print_error_rate()
    least_squares_classifier.test(train_input, train_label)
    least_squares_classifier.show_confusion_matrix(tag="train", title="Least Squares Confusion Matrix Train Set without Bias")
    least_squares_classifier.print_error_rate()

    least_squares_classifier_bias = LeastSquaresMultiClassClassifier(train_input_bias, train_label, class_list, weights=weights_bias)
    least_squares_classifier_bias.test(test_input_bias, test_label)
    least_squares_classifier_bias.show_confusion_matrix(tag="teste_bias", title="Least Squares Confusion Matrix Test Set with Bias")
    least_squares_classifier_bias.print_error_rate()
    least_squares_classifier_bias.test(train_input_bias, train_label)
    least_squares_classifier_bias.show_confusion_matrix(tag="train_bias", title="Least Squares Confusion Matrix Train Set with Bias")
    least_squares_classifier_bias.print_error_rate()
