import numpy as np
from keras.utils import to_categorical


def custom_to_categorical(x, classes_list):
    
    for c_ind, c in enumerate(classes_list):
        x[x==c] = c_ind
        
    x = to_categorical(x, num_classes=len(classes_list))

    return x
