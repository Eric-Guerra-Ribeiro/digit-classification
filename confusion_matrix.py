import numpy as np
import matplotlib.pyplot as plt

class ConfusionMatrix:
    """
    """
    def __init__(self, labels, predictions, k_classes):
        """
        """
        self.confusion_matrix = np.zeros((k_classes, k_classes))
        for i in range(len(labels)):
            self.confusion_matrix[labels[i]][predictions[i]] += 1
        self.confusion_matrix = self.confusion_matrix.astype(int)


    def plot_confusion_matrix(self, class_list, tag="", title="Confusion Matrix"):
        """
        """
        fig, ax = plt.subplots()
        im = ax.imshow(self.confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(self.confusion_matrix.shape[1]),
               yticks=np.arange(self.confusion_matrix.shape[0]),
               xticklabels=class_list, yticklabels=class_list,
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
