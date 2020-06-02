#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
utils functions for RED-GREEN experiment
"""

import numpy as np

def gen_box_data(trainset, y_train, length=1000, image_size=32, box_size=4,
                 offset1=7, offset2=23, shiftdiv=6):

    """
    * Generates training set which has colored 2 boxes (green and red) in each image.
    * There are two classes:
        1. Red-to-the-left-of-Green
        2. Green-to-the-left-of-Red
    
    * Input arguments:   
        trainset -- empty numpy array with size of your input training images [dataset_size, channel, width, height]
        y_train -- empty numpy array with size of your input training labels [data_size]
        length -- number of images in the dataset [data_size]
        image_size -- spatial size of image 
        box_size -- spatial size of box (square)
        offset1 -- offset for class 1
        offset2 --  offset for class 2
        shiftdiv -- division of shift to fluctuate the location around the offset on the y-axis

    * Output arguments:
        trainset -- numpy array with size of generated input training images [dataset_size, channel, width, height]
        y_train -- numpy array with size of generated input training labels [data_size]
    """
    np.random.seed(1988)
    img = np.zeros([3, image_size, image_size], dtype=float)
    patch = np.ones([box_size, box_size], dtype=float)
    off_size = image_size - box_size

    for i in range(length):

        if i % 2 == 0:
            im = img.copy()
            offsetx = np.random.randint(off_size/2 - box_size)
            offsety = np.random.randint(-off_size/shiftdiv, off_size/shiftdiv) + offset1
            # obj 1
            im[0, offsety:offsety+box_size, offsetx:offsetx+box_size] = patch
            # obj 2
            offsetx2 = np.random.randint(offsetx, off_size)
            offsety2 = np.random.randint(-off_size/shiftdiv, off_size/shiftdiv) + offset1

            while ((abs(offsetx-offsetx2) < box_size+1)):
                offsetx2 = np.random.randint(offsetx, off_size)
                offsety2 = np.random.randint(-off_size/shiftdiv, off_size/shiftdiv) + offset1

            im[1, offsety2:offsety2+box_size, offsetx2:offsetx2+box_size] = patch

            trainset[i] = im
            y_train[i] = 0

        elif i % 2 == 1:
            im = img.copy()
            offsetx = np.random.randint(off_size/2 - box_size)
            offsety = np.random.randint(-off_size/shiftdiv, off_size/shiftdiv) + offset2
            # obj 1
            im[1, offsety:offsety+box_size, offsetx:offsetx+box_size] = patch
            # obj 2
            offsetx2 = np.random.randint(offsetx, off_size)
            offsety2 = np.random.randint(-off_size/shiftdiv, off_size/shiftdiv) + offset2

            while ((abs(offsetx-offsetx2) < box_size+1)):
                offsetx2 = np.random.randint(offsetx, off_size)
                offsety2 = np.random.randint(-off_size/shiftdiv, off_size/shiftdiv) + offset2

            im[0, offsety2:offsety2+box_size, offsetx2:offsetx2+box_size] = patch

            trainset[i] = im
            y_train[i] = 1
   
    return trainset, y_train

def gen_box_data_test(testset1, y_test1, testset2, y_test2, length=1000,
                      image_size=32, box_size=4, offset1=7, offset2=23,
                      shiftdiv=6):

    """
    * Generates training set which has colored 2 boxes (green and red) in each image.
    * There are two classes:
        1. Red-to-the-left-of-Green
        2. Green-to-the-left-of-Red

    * Input arguments:   
        trainset -- empty numpy array with size of your input training images [dataset_size, channel, width, height]
        y_train -- empty numpy array with size of your input training labels [data_size]
        length -- number of images in the dataset [data_size]
        image_size -- spatial size of image 
        box_size -- spatial size of box (square)
        offset1 -- offset for class 1
        offset2 --  offset for class 2
        shiftdiv -- division of shift to fluctuate the location around the offset on the y-axis

    * Output arguments:
        testset1 -- numpy array with size of generated input test images similar to training set [dataset_size, channel, width, height]
        y_test1-- numpy array with size of generated input test labels [data_size]
        testset2 -- numpy array with size of generated input test images dissimilar to training set [dataset_size, channel, width, height]
		(swapping the classes) 
        y_test2-- numpy array with size of generated input test labels [data_size]
    """
    np.random.seed(1988)
    img = np.zeros([3, image_size, image_size], dtype=float)
    patch = np.ones([box_size, box_size], dtype=float)
    off_size = image_size - box_size

    for i in range(length):

        if i % 2 == 0:
            im = img.copy()
            img2 = img.copy()
            offsetx = np.random.randint(off_size/2 - box_size)
            offsety = np.random.randint(-off_size/shiftdiv, off_size/shiftdiv) + offset1
            # obj 1
            im[0, offsety:offsety+box_size, offsetx:offsetx+box_size] = patch
            offsety += 16
            img2[0, offsety:offsety+box_size, offsetx:offsetx+box_size] = patch
            # obj 2
            offsetx2 = np.random.randint(offsetx, off_size)
            offsety2 = np.random.randint(-off_size/shiftdiv, off_size/shiftdiv) + offset1

            while ((abs(offsetx-offsetx2) < box_size+1)):
                offsetx2 = np.random.randint(offsetx, off_size)
                offsety2 = np.random.randint(-off_size/shiftdiv, off_size/shiftdiv) + offset1
    
            im[1, offsety2:offsety2+box_size, offsetx2:offsetx2+box_size] = patch
            offsety2 += 16
            img2[1, offsety2:offsety2+box_size, offsetx2:offsetx2+box_size] = patch

            testset1[i] = im
            y_test1[i] = 0
            testset2[i] = img2
            y_test2[i] = 0

        elif i % 2 == 1:
            im = img.copy()
            img2 = img.copy()
            offsetx = np.random.randint(off_size/2 - box_size)
            offsety = np.random.randint(-off_size/shiftdiv, off_size/shiftdiv) + offset2
            # obj 1
            im[1, offsety:offsety+box_size, offsetx:offsetx+box_size] = patch
            offsety -= 16
            img2[1, offsety:offsety+box_size, offsetx:offsetx+box_size] = patch
            # obj 2
            offsetx2 = np.random.randint(offsetx, off_size)
            offsety2 = np.random.randint(-off_size/shiftdiv, off_size/shiftdiv) + offset2

            while ((abs(offsetx-offsetx2) < box_size+1)):
                offsetx2 = np.random.randint(offsetx, off_size)
                offsety2 = np.random.randint(-off_size/shiftdiv, off_size/shiftdiv) + offset2

            im[0, offsety2:offsety2+box_size, offsetx2:offsetx2+box_size] = patch
            offsety2 -= 16
            img2[0, offsety2:offsety2+box_size, offsetx2:offsetx2+box_size] = patch

            testset1[i] = im
            y_test1[i] = 1
            testset2[i] = img2
            y_test2[i] = 1
     
    return testset1, y_test1, testset2, y_test2