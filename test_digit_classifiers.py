import numpy as np
import matplotlib.pyplot as plt

from user_scripts.training_configurations import training_configurations

### Import Test Images ###

sample_start = 50
sample_size = 10

from keras.datasets import mnist

mnist_images_test, mnist_labels_test = mnist.load_data()[1]

mnist_images = []
mnist_labels = []

search_ind = sample_start
samples_found = 0
while samples_found < sample_size:
    if mnist_labels_test[search_ind] == 0 or mnist_labels_test[search_ind] == 1:
        mnist_images.append(mnist_images_test[search_ind])
        mnist_labels.append(mnist_labels_test[search_ind])
        samples_found +=1
    search_ind +=1

mnist_images = np.array(mnist_images)
mnist_labels = np.array(mnist_labels)

mnist_images = mnist_images.reshape((sample_size, 28, 28, 1))
mnist_images = mnist_images.astype('float32') / 255

for mnist_image in mnist_images:
    plt.figure()
    plt.title('mnist image')
    plt.imshow(mnist_image, aspect='auto', cmap='gray')


from keras.models import load_model

classifier = load_model('saved_models/graph_execution/classifier/classifier_1_model.h5')

for mnist_image, mnist_label in zip(mnist_images, mnist_labels):
    mnist_image = mnist_image.reshape((1,28,28,1))
    classifier_output = np.argmax(classifier.predict(mnist_image)[0])
    print('\tclassifier result:', classifier_output)
    print('\tactual digit value:', mnist_label)
    print('\n')


plt.show()




