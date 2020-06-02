"""
Generating Imagenet-50 training set from full Imagenet
"""

import os
import shutil
import sys
import time
from tqdm import tqdm
import pickle

def listdir_fullpath(d):
    return [os.path.join(d, f) for f in os.listdir(d)]

if __name__ == '__main__':

    main_dir_path = sys.argv[1]  # directory of train set "/home/user/imagenet/raw-data"
    output_dir = sys.argv[2]  # output location "/home/user/datasets/"
    main_traindir_path = main_dir_path + '/' + 'train'

    train_dir = output_dir + 'imagenet_50/train/'
    if not os.path.exists(train_dir):
        print("path doesn't exist. trying to make")
        os.makedirs(train_dir)

    t1 = time.time()

    dirnames = listdir_fullpath(main_traindir_path)

    pickle_in = open("train.pickle", "rb")
    train = pickle.load(pickle_in)
    pickle_in.close()

    for image_dir in tqdm(dirnames):
        train_dir_name = train_dir + os.path.basename(image_dir)

        if os.path.exists(train_dir_name):
            shutil.rmtree(train_dir_name)
            pass
        os.mkdir(train_dir_name)

        for train_file in train:
            if image_dir == main_traindir_path + '/' + train_file[0]:
                train_file_name = train_dir_name + "/" + os.path.basename(train_file[1])
                shutil.copy2(main_traindir_path+ '/'+train_file[0] + '/' + train_file[1],
                             train_file_name)
            else:
                continue

    t2 = time.time()

    print("Data generation time : ", (t2-t1) / 60.0)
