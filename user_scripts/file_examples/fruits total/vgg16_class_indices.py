from keras.applications import VGG16
from keras.applications.vgg16 import decode_predictions
import numpy as np

vgg16 = VGG16(include_top=True,
              weights='imagenet',
              input_tensor=None,
              input_shape=None,
              pooling=None,
              classes=1000)

vgg16_class_indices = decode_predictions(np.expand_dims(np.arange(1000), 0), top=1000)[0]
vgg16_class_indices = {tuple_item[1]: tuple_item[2] for tuple_item in vgg16_class_indices}
