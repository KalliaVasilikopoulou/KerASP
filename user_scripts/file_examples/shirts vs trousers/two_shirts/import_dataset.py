import numpy as np
import matplotlib.pyplot as plt

import io
from PIL import Image

import random

from tqdm import tqdm

from user_scripts.training_configurations import training_configurations

mnist_dataset_portion = training_configurations['mnist_dataset_portion']

### Prepare the training dataset ###

from keras.datasets import fashion_mnist

(mnist_images, mnist_labels) = fashion_mnist.load_data()[0]

### keep only 1 (trouser) and 6 data (shirt) ###

keep_classes = [6, 1]

mnist_images_small = []
mnist_labels_small = []

for ind,label in enumerate(mnist_labels):
    if label in keep_classes:
        mnist_images_small.append(mnist_images[ind])
        mnist_labels_small.append(keep_classes.index(label))        # we save 6 label as 0 and 1 label as 1

mnist_images_small = np.array(mnist_images_small)
mnist_labels_small = np.array(mnist_labels_small)


### split to train and validation sets ###

mnist_dataset_split_point = int(mnist_images_small.shape[0] * mnist_dataset_portion)
mnist_images_small = mnist_images_small[:mnist_dataset_split_point]
mnist_labels_small = mnist_labels_small[:mnist_dataset_split_point]


### pre-process train images ###

mnist_images_small = np.expand_dims(mnist_images_small.astype('float32') / 255, axis=-1)

### shuffle dataset ###

indices = np.arange(mnist_images_small.shape[0])
np.random.shuffle(indices)

mnist_images_small = mnist_images_small[indices]
mnist_labels_small = mnist_labels_small[indices]


###########################################

##input_data = mnist_images_small
##input_labels = mnist_labels_small

data_dict = {'1':
                 {'input_data': mnist_images_small,
                  'input_labels': mnist_labels_small}}





    


