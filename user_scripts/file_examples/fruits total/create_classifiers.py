from keras.applications import VGG16
from keras.applications.vgg16 import decode_predictions
import numpy as np

classifier = VGG16(include_top=True,
                   weights='imagenet',
                   input_tensor=None,
                   input_shape=None,
                   pooling=None,
                   classes=1000)
