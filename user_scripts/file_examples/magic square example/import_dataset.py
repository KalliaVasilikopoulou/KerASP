import numpy as np
import matplotlib.pyplot as plt

import io
from PIL import Image

import random

from tqdm import tqdm

from user_scripts.training_configurations import training_configurations

mnist_dataset_portion = training_configurations['mnist_dataset_portion']
add_typed_digits_to_dataset = training_configurations['add_typed_digits_to_dataset']

N = training_configurations['N']

cell_1_index = training_configurations['cell_1_index']
cell_2_index = training_configurations['cell_2_index']
cell_3_index = training_configurations['cell_3_index']
cell_indices = [cell_1_index, cell_2_index, cell_3_index]

cell_ukn_index = training_configurations['cell_ukn_index']

### Prepare the training dataset ###
from keras.datasets import mnist

(mnist_images, mnist_labels) = mnist.load_data()[0]

mnist_dataset_split_point = int(mnist_images.shape[0] * mnist_dataset_portion)
mnist_images = mnist_images[:mnist_dataset_split_point]
mnist_labels = mnist_labels[:mnist_dataset_split_point]

if add_typed_digits_to_dataset:
    ### Injection of typed MNIST digit images and labels and shuffling ###

    ### Import Dataset if it exists ###
    try:
        typed_digits_images = np.load('mnist_magic_square_array/typed_digits_images.npy')
        typed_digits_labels = np.load('mnist_magic_square_array/typed_digits_labels.npy')

        print('Loaded typed digits images and labels from corresponding files')

    except FileNotFoundError:
        ### Create Magic Squares ###
        from create_dataset.create_typed_digits_dataset import typed_digits_images, typed_digits_labels
        
        ### Save Dataset ###
        np.save('mnist_magic_square_array/typed_digits_images.npy', typed_digits_images)
        np.save('mnist_magic_square_array/typed_digits_labels.npy', typed_digits_labels)

        print('Saved typed digits images and labels to corresponding files')


    mnist_images = np.concatenate([mnist_images, typed_digits_images], axis=0)
    mnist_labels = np.concatenate([mnist_labels, typed_digits_labels], axis=0)

mnist_images = np.expand_dims(mnist_images.astype('float32') / 255, axis=-1)

### shuffle dataset ###

indices = np.arange(mnist_images.shape[0])
np.random.shuffle(indices)

mnist_images = mnist_images[indices]
mnist_labels = mnist_labels[indices]


###########################################

input_data = mnist_images
input_labels = mnist_labels

data_dict = {'1':
                 {'input_data': mnist_images,
                  'input_labels': input_labels},
             '2':
                  {'input_data': mnist_images,
                   'input_labels': input_labels}}





