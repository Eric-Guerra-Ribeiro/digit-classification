import numpy as np
import least_squares

if __name__ == "__main__":
    train_input = np.loadtxt("MNIST\\train_input.gz", dtype=float)
    train_label = np.loadtxt("MNIST\\train_label.gz", dtype=str)
    test_input = np.loadtxt("MNIST\\test_input.gz" , dtype=float) 
    test_label = np.loadtxt("MNIST\\test_label.gz" , dtype=str)
    class_list = np.loadtxt("MNIST\\class_list.gz" , dtype=str)
    least_squares_classifier = least_squares.LeastSquaresMultiClassClassifier(train_input, train_label, class_list)
    least_squares_classifier.test(test_input, test_label)
    least_squares_classifier.show_confusion_matrix(tag="teste")
    # least_squares_classifier.test(train_input, train_label)
    # least_squares_classifier.show_confusion_matrix(tag="train")


