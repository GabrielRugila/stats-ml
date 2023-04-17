import numpy as np
import pathlib


def get_mnist():
    with np.load(f"{pathlib.Path(__file__).parent.absolute()}/data/mnist.npz") as f:
        images, labels = f["x_train"], f["y_train"]
    images = images.astype("float32") / 255
    images = np.reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))
    labels = np.eye(10)[labels]
    return images, labels

def split_data(images, labels, split=0.8):
    split_index = int(images.shape[0] * split)
    train_images = images[:split_index]
    train_labels = labels[:split_index]
    test_images = images[split_index:]
    test_labels = labels[split_index:]
    return train_images, train_labels, test_images, test_labels