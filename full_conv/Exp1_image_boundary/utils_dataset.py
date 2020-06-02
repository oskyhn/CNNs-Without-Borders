import numpy as np
import cv2
np.random.seed(1988)  # for reproducibility

def four_q_gen(dataset, set_type=0):

    """
    * Generates training, validation and test set of imagenet-location classification task.
    * There are two classes:
        1. Upper-left
        2. Bottom-right

    * Input arguments:
        dataset -- pytorch dataset object
        set_type -- output dataset type (train, val, test)

    * Output arguments:
        full_x_dataset -- numpy array with size of generated input training images [dataset_size, channel, width, height]
        full_y_dataset -- numpy array with size of generated input training labels [data_size]
    * Usage
        x_train, y_train = four_q_gen(dataset, set_type=0)
        x_val, y_val = four_q_gen(dataset, set_type=1)
        x_test, y_test = four_q_gen(dataset, set_type=2)
    """
    full_x_dataset = np.zeros([2000, 120, 120, 3])
    full_y_dataset = np.zeros([2000, 1])
    for idx, i in enumerate(range(0, len(dataset), 50), 0):
        patch = cv2.resize(np.moveaxis(dataset[i][0].numpy(), 0, -1), (56, 56))
        full_x_dataset[idx, :56, :56, :] = patch
        full_y_dataset[idx] = 0
        full_x_dataset[1000+idx, 64:, 64:, :] = patch
        full_y_dataset[1000+idx] = 1

    full_x_dataset = np.moveaxis(full_x_dataset, -1, 1)
    return full_x_dataset, full_y_dataset