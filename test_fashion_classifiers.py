import numpy as np
import matplotlib.pyplot as plt

from user_scripts.training_configurations import training_configurations

### Import Test Images ###

sample_start = 50
sample_size = 10

from keras.datasets import fashion_mnist

mnist_images_test, mnist_labels_test = fashion_mnist.load_data()[1]
mnist_images_test = mnist_images_test[sample_start:]
mnist_labels_test = mnist_labels_test[sample_start:]

mnist_images = []
mnist_labels = []

### keep only 1 (trouser) and 6 data (shirt) ###

keep_classes = [6, 1]

samples_found = 0
for ind,label in enumerate(mnist_labels_test):
    if label in keep_classes:
        mnist_images.append(mnist_images_test[ind])
        mnist_labels.append(keep_classes.index(label))        # we save 6 label as 0 and 1 label as 1
        samples_found +=1
        if samples_found == sample_size: break

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

classes_names = ["shirt", "trouser"]

for mnist_image, mnist_label in zip(mnist_images, mnist_labels):
    mnist_image = mnist_image.reshape((1,28,28,1))
    classifier_result = classes_names[np.argmax(classifier.predict(mnist_image)[0])]
    actual_value = classes_names[mnist_label]
    print('\tclassifier result:', classifier_result)
    print('\tactual digit value:', actual_value)
    print('\n')


plt.show()




