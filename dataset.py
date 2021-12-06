import numpy as np
from PIL import Image

IMAGE_WIDTH = 28
IMAGE_HEIGHT = 28
TEST_PATH = "MNIST\Test"
TRAIN_PATH = "MNIST\Train"
N_TEST_IMAGES = 200
N_TRAIN_IMAGES = 2400

def get_input(path):
    """
    """
    if path == TRAIN_PATH:
        number_images = N_TRAIN_IMAGES
    elif path == TEST_PATH:
        number_images = N_TEST_IMAGES
    input = np.ones([number_images, IMAGE_HEIGHT*IMAGE_WIDTH + 1 ])
    for i in range(number_images):
        newpath = path + "\{}.jpg".format(i+1)
        img = Image.open(newpath)
        image = np.asarray(img).flatten()
        image = image/255
        image = np.append(image, [1])
        input[i] =  image
    return input

def get_label(path):
    """
    """
    if path == TRAIN_PATH:
        number_images = N_TRAIN_IMAGES
    elif path == TEST_PATH:
        number_images = N_TEST_IMAGES
    label = np.zeros(number_images)
    for i in range(10):
        for j in  range(number_images//10):
            label[number_images*i//10+j] = i
    return label.astype(int)


def get_classes():
    """
    """
    return np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

if __name__ == "__main__":
    np.savetxt("MNIST\\train_input.gz", get_input(TRAIN_PATH))
    np.savetxt("MNIST\\train_label.gz", get_label(TRAIN_PATH), fmt="%d")
    np.savetxt("MNIST\\test_input.gz", get_input(TEST_PATH))
    np.savetxt("MNIST\\test_label.gz", get_label(TEST_PATH), fmt="%d")
    np.savetxt("MNIST\\class_list.gz", get_classes(), fmt="%d")