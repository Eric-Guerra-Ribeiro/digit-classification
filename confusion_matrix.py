import numpy as np
import matplotlib.pyplot as plt

class ConfusionMatrix:
    """
    """
    def __init__(self, labels, predictions, k_classes, class_list):
        """
        """
        self.confusion_matrix = np.zeros((k_classes, k_classes))
        for i in range(len(labels)):
            self.confusion_matrix[labels[i]][predictions[i]] += 1
        self.confusion_matrix = self.confusion_matrix.astype(int)
        self.class_list = class_list


    def plot_confusion_matrix(self, tag="", title="Confusion Matrix"):
        """
        """
        fig, ax = plt.subplots()
        im = ax.imshow(self.confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(self.confusion_matrix.shape[1]),
               yticks=np.arange(self.confusion_matrix.shape[0]),
               xticklabels=self.class_list, yticklabels=self.class_list,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')
        thresh = self.confusion_matrix.max() / 2.
        for i in range(self.confusion_matrix.shape[0]):
            for j in range(self.confusion_matrix.shape[1]):
                ax.text(j, i, format(self.confusion_matrix[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if self.confusion_matrix[i, j] > thresh else "black")
        fig.tight_layout()
        plt.savefig("Confusion Matrixes\\confusionmatrix{}.png".format(tag), facecolor='w', edgecolor='w',
                    orientation='portrait', transparent=True, pad_inches=0.1)
        plt.show()


    def print_error_rate(self):
        cases_by_class = np.sum(self.confusion_matrix, axis=1)
        errors_by_class = np.sum(self.confusion_matrix - np.diag(np.diag(self.confusion_matrix)), axis=1)
        total_errors = np.sum(errors_by_class)
        total_cases = np.sum(cases_by_class)
        print("Error Rate: {:.2f}%".format(100*total_errors/total_cases))
        for i in range(len(self.class_list)):
            print("Error Rate for {}: {:.2f}%".format(self.class_list[i], 100*errors_by_class[i]/cases_by_class[i]))
            print("Misidentified as:", end=" ")
            for j in range(len(self.class_list)):
                if i != j and self.confusion_matrix[i][j] != 0:
                    print("{} {:.2f}%,".format(self.class_list[j], 100*self.confusion_matrix[i][j]/cases_by_class[i]), end=" ")
            print()
        print()
