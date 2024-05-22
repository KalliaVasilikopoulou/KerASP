import keras
from keras import layers, models

classifier_input = keras.Input(shape=(28,28,1))
x = layers.Conv2D(32, (3,3), activation='relu')(classifier_input)
x = layers.MaxPooling2D((2,2))(x)
x = layers.Conv2D(64, (3,3), activation='relu')(x)
x = layers.MaxPooling2D((2,2))(x)
x = layers.Conv2D(64, (3,3), activation='relu')(x)
x = layers.Flatten()(x)
x = layers.Dense(64, activation='relu')(x)

classifier_output = layers.Dense(2, activation = 'softmax')(x)

classifier = models.Model(classifier_input, classifier_output, name='digit_classifier')

#classifiers = {'1': classifier}
