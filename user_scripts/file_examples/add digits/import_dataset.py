import numpy as np
import matplotlib.pyplot as plt

import io
from PIL import Image

import random

from tqdm import tqdm

from user_scripts.training_configurations import training_configurations

mnist_dataset_portion = training_configurations['mnist_dataset_portion']

### Prepare the training dataset ###

from keras.datasets import mnist

(mnist_images, mnist_labels) = mnist.load_data()[0]


### split to train and validation sets ###

mnist_dataset_split_point = int(mnist_images.shape[0] * mnist_dataset_portion)
mnist_images = mnist_images[:mnist_dataset_split_point]
mnist_labels = mnist_labels[:mnist_dataset_split_point]


### pre-process train images ###

mnist_images = np.expand_dims(mnist_images.astype('float32') / 255, axis=-1)

### shuffle dataset ###

indices = np.arange(mnist_images.shape[0])
np.random.shuffle(indices)

mnist_images = mnist_images[indices]
mnist_labels = mnist_labels[indices]


###########################################

##input_data = mnist_images
##input_labels = mnist_labels

data_dict = {'1':
                 {'input_data': mnist_images,
                  'input_labels': mnist_labels}}





    


