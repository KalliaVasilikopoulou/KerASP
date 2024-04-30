import keras
from keras import layers, models

from training_configurations import training_configurations

classifier_input = keras.Input(shape=(28,28,1))
x = layers.Conv2D(32, (3,3), activation='relu')(classifier_input)
x = layers.MaxPooling2D((2,2))(x)
x = layers.Conv2D(64, (3,3), activation='relu')(x)
x = layers.MaxPooling2D((2,2))(x)
x = layers.Conv2D(64, (3,3), activation='relu')(x)
x = layers.Flatten()(x)
x = layers.Dense(64, activation='relu')(x)

classifier_output = layers.Dense(9, activation = 'softmax', name='cell_output_probs')(x)

classifier = models.Model(classifier_input, classifier_output, name='classifier')
