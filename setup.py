import numpy as np
import dataset
import least_squares

if __name__ == "__main__":
    train_input = dataset.get_input(dataset.TRAIN_PATH)
    train_label = dataset.get_label(dataset.TRAIN_PATH)
    class_list = dataset.get_classes()
    np.savetxt("MNIST\\train_input.gz", train_input)
    np.savetxt("MNIST\\train_label.gz", train_label, fmt="%d")
    np.savetxt("MNIST\\test_input.gz", dataset.get_input(dataset.TEST_PATH))
    np.savetxt("MNIST\\test_label.gz", dataset.get_label(dataset.TEST_PATH), fmt="%d")
    np.savetxt("MNIST\\class_list.gz", class_list, fmt="%d")
    least_squares_classifier = least_squares.LeastSquaresMultiClassClassifier(train_input, train_label, class_list)
    least_squares_classifier.export_weights()
