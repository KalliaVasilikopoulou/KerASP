import numpy as np
from tqdm import tqdm

import tensorflow as tf

import keras
from keras import models, optimizers
from keras import backend as K
from keras.utils import to_categorical

import time
import gc


import numpy as np

from project_utils.convert_to_categorical import custom_to_categorical as to_categorical


### Set Training Parameters ###

from user_scripts.training_configurations import training_configurations

learning_rate = training_configurations['learning_rate']
decay = training_configurations['decay']

epochs = training_configurations['epochs']
batch_size = training_configurations['batch_size']
validation_split = training_configurations['validation_split']


program_specs = training_configurations['program_specs']

num_of_objects = program_specs['num_of_objects']
list_of_object_classes = program_specs['list_of_object_classes']
classes_per_object = len(list_of_object_classes)

object_type = program_specs['object_type']
output_type = program_specs['output_type']
classes_type = program_specs['classes_type']


try: tokenize_data = training_configurations['tokenize_data']
except KeyError: tokenize_data = False

### Import Dataset If It Exists ###

try:
    import os
    classifier_inputs = 0
    for _, _, files in os.walk('train_data/'):
        for file in files:    
            if file.startswith('train_data') and file.endswith('.npy'):
                classifier_inputs += 1
    
    train_data = []
    for data_ind in range(classifier_inputs):
        train_data.append(np.load('train_data/train_data_'+str(data_ind)+'.npy'))
    train_obj_classes_labels = np.load('train_data/train_obj_classes_labels.npy')
    train_labels = np.load('train_data/train_labels.npy')

    if tokenize_data:
        from tokenization.handle_tokenizers import load_tokenizers
        tokenizers = [str(i) for i in range(classifier_inputs)]
        load_tokenizers(tokenizers)

    print('Loaded train data and labels from corresponding files')
    
### If Dataset Does Not Exist, Create New Dataset ###
except FileNotFoundError:

    from create_dataset.create_dataset import train_data, train_obj_classes_labels, train_labels 

    for data_ind, data in enumerate(train_data):
        np.save('train_data/train_data_'+str(data_ind)+'.npy', data)
    np.save('train_data/train_obj_classes_labels.npy', train_obj_classes_labels)
    np.save('train_data/train_labels.npy', train_labels)

    if tokenize_data:
        from tokenization.handle_tokenizers import save_tokenizers
        from create_dataset.create_dataset import train_tokenizers_dict
        save_tokenizers(train_tokenizers_dict)

    print('Saved train data and labels to corresponding files')

### Pre-process Dataset Data And Labels ###

print('Imported train data and labels:')
for data_ind, data in enumerate(train_data): 
    print('train_data data', data_ind, 'shape:', data.shape)            # every array has shape: (samples, num_of_objects, datapoint shape)
print('train_obj_classes_labels shape:', train_obj_classes_labels.shape)      # (samples, num_of_objects)
print('train_labels shape:', train_labels.shape)                        # (samples,)

for i in range(len(train_data)):    # i is index of class input (input for different data types), while j is index of object data
    train_data[i] = np.array_split(train_data[i], train_data[i].shape[1], axis=1)
    train_data[i] = [np.squeeze(j, axis=1) for j in train_data[i]]      # list conctaining classifier_inputs elements and each element is a list containing num_of_objects elements and each element has shape (samples, datapoint shape)


from find_interpretations.find_object_classes_for_output_class import find_all_obj_classes_for_known_output_classes
_, output_classes_list = find_all_obj_classes_for_known_output_classes(return_output_classes_list=True)

train_labels = to_categorical(train_labels, classes_list=output_classes_list)

print('train data list length:', len(train_data))                # classifier_inputs
print('train data_k list length:', len(train_data[0]))           # num_of_objects
print('train data_k object_n shape:', train_data[0][0].shape)    # (samples, datapoint shape)
print('train labels shape:', train_labels.shape)                 # (samples, solver classes)

### Split Point ###

train_split = 1 - validation_split
split_point = int(train_data[0][0].shape[0] * train_split)

train_data_draft = []
validation_data_draft = []
for data in train_data:
    validation_data_draft.append([object_data[split_point:] for object_data in data])
    train_data_draft.append([object_data[:split_point] for object_data in data])
train_data = train_data_draft
validation_data = validation_data_draft

validation_labels = train_labels[split_point:]
train_labels = train_labels[:split_point]


### Instatiate an optimizer ###
optimizer = optimizers.Adam(learning_rate=learning_rate, decay=decay)


# Prepare the metrics.
train_acc_metric = keras.metrics.CategoricalAccuracy()
val_acc_metric = keras.metrics.CategoricalAccuracy()



### Import Model ###

from user_scripts.create_classifier import classifier

classifier = models.clone_model(classifier)
classifier.summary()



### Train Model ###

def separate_probs(classes_probs):    # (None,classes_per_object)

    return [classes_probs[:,class_i] for class_i in range(classes_per_object)]      # [(None,),(None,),...,(None,)], length of list=classes_per_object


from find_interpretations.find_object_classes_for_output_class import find_all_obj_classes_for_known_output_classes
objects_combinations_of_all_output_classes, output_classes_list = find_all_obj_classes_for_known_output_classes(return_output_classes_list=True)


### Calculate Gradients Functions ###

classifier_inputs = 0
classifier_input_shapes = []

for layer in classifier.layers:
    if layer.name.startswith('input'):
        classifier_inputs +=1
        classifier_input_shapes.append(tuple([batch_size]) + layer.output.shape[1:])

x_tensor_list = [[tf.Variable(np.zeros(shape=classifier_input_shape, dtype='float32')) for _ in range(num_of_objects)] for classifier_input_shape in classifier_input_shapes]

### Training Step ###
def exec_train_step(x_array_list, y):
    K.clear_session()
    gc.collect()
    
    with tf.GradientTape(persistent=True) as tape:
        try:
            for inp in range(classifier_inputs):
                for obj in range(num_of_objects):
                    x_tensor_list[inp][obj].assign(x_array_list[inp][obj])
        except ValueError:  # last batch of dataset has less samples than batch_size
            padding_dim_0 = batch_size - x_array_list[0][0].shape[0]
            for inp in range(classifier_inputs):
                for obj in range(num_of_objects):
                    x_array_list[inp][obj] = K.concatenate([x_array_list[inp][obj], tf.zeros(shape=tuple([padding_dim_0])+x_array_list[inp][obj].shape[1:])], axis=0)
                    x_tensor_list[inp][obj].assign(x_array_list[inp][obj])
            y = K.concatenate([y,tf.zeros(shape=tuple([padding_dim_0])+y.shape[1:])], axis=0)

        classes_probs = [classifier([x_tensor_list[inp][obj] for inp in range(classifier_inputs)], training=True) for obj in range(num_of_objects)]

        probs = [separate_probs(object_probs) for object_probs in classes_probs]    # a list containing num_of_objects lists, where each one of these num_of_objects lists contains classes_per_object (None,) tensors 

        combinations_probs_of_all_output_classes = [[K.prod(K.concatenate([K.expand_dims(probs[i][list_of_object_classes.index(j)], axis=-1) for i,j in enumerate(comb)], axis=-1), axis=-1) for comb in objects_combinations_of_output_class_k] for objects_combinations_of_output_class_k in objects_combinations_of_all_output_classes]   # for each sum: [(None,), (None,),...,(None,)], length of list = num of interpretations

        probs_of_all_output_classes = [K.sum(K.concatenate([K.expand_dims(comb_probs, axis=-1) for comb_probs in combinations_probs_of_output_class_k], axis=-1), axis=-1) for combinations_probs_of_output_class_k in combinations_probs_of_all_output_classes]       # [(None,),...,], length = all possible sums = satisfiable values

        probs_of_all_output_classes = K.concatenate([K.expand_dims(probs_of_output_class_k, axis=-1) for probs_of_output_class_k in probs_of_all_output_classes], axis=-1)   # (None, satisfiable values)

        loss = K.categorical_crossentropy(y, probs_of_all_output_classes)

    grads = tape.gradient(loss, classifier.trainable_weights)
    optimizer.apply_gradients(zip(grads, classifier.trainable_weights))
    train_acc_metric.update_state(y, probs_of_all_output_classes)
    
    return loss


### Validation Step ###
def exec_val_step(x, y):
    
    classes_probs = [classifier([x[inp][obj] for inp in range(classifier_inputs)], training=False) for obj in range(num_of_objects)]

    probs = [separate_probs(object_probs) for object_probs in classes_probs]    # a list containing num_of_objects lists, where each one of these num_of_objects lists contains classes_per_object (None,) tensors 

    combinations_probs_of_all_output_classes = [[K.prod(K.concatenate([K.expand_dims(probs[i][list_of_object_classes.index(j)], axis=-1) for i,j in enumerate(comb)], axis=-1), axis=-1) for comb in objects_combinations_of_output_class_k] for objects_combinations_of_output_class_k in objects_combinations_of_all_output_classes]   # for each sum: [(None,), (None,),...,(None,)], length of list = num of interpretations

    probs_of_all_output_classes = [K.sum(K.concatenate([K.expand_dims(comb_probs, axis=-1) for comb_probs in combinations_probs_of_output_class_k], axis=-1), axis=-1) for combinations_probs_of_output_class_k in combinations_probs_of_all_output_classes]       # [(None,),...,], length = all possible sums = satisfiable values

    probs_of_all_output_classes = K.concatenate([K.expand_dims(probs_of_output_class_k, axis=-1) for probs_of_output_class_k in probs_of_all_output_classes], axis=-1)   # (None, satisfiable values)

    val_acc_metric.update_state(y, probs_of_all_output_classes)



### Training Loop ###
    
steps = train_data[0][0].shape[0] // batch_size + (train_data[0][0].shape[0] % batch_size > 1)
val_steps = validation_data[0][0].shape[0] // batch_size + (validation_data[0][0].shape[0] % batch_size > 1)

for epoch in range(epochs):
    start_time = time.time()

    # Shuffle train dataset before the start of each epoch
    indices = np.arange(train_data[0][0].shape[0])
    np.random.shuffle(indices)
    for inp in range(classifier_inputs):
        for obj in range(num_of_objects):
            train_data[inp][obj] = train_data[inp][obj][indices]
    train_labels = train_labels[indices]

    # Iterate over the batches of the dataset.
    for step in tqdm(range(steps), total = steps, ncols= 100, desc ='Epoch '+str(epoch+1)+'/'+str(epochs), position=0, leave=True):
        x_batch_train = []
        for inp in range(classifier_inputs):
            x_batch_of_input = []
            for obj in range(num_of_objects):
                if  (step+1)*batch_size < train_data[0][0].shape[0] - 1:
                    x_batch_of_input.append(train_data[inp][obj][step*batch_size : (step+1)*batch_size])
                else:
                    x_batch_of_input.append(train_data[inp][obj][step*batch_size:])
            x_batch_train.append(x_batch_of_input)
        y_batch_train = train_labels[step*batch_size : (step+1)*batch_size] if (step+1)*batch_size < train_data[0][0].shape[0] - 1 else train_labels[step*batch_size:]
        exec_train_step(x_batch_train, y_batch_train)

    # Display metrics at the end of each epoch.
    train_acc = train_acc_metric.result()
    print("Training acc over epoch: %.4f" % (float(train_acc),))

    # Reset training metrics at the end of each epoch
    train_acc_metric.reset_states()

    # Run a validation loop at the end of each epoch.
    for val_step in range(val_steps):
        x_batch_val = []
        for inp in range(classifier_inputs):
            x_batch_of_input_val = []
            for obj in range(num_of_objects):
                if  (val_step+1)*batch_size < validation_data[0][0].shape[0] - 1:
                    x_batch_of_input_val.append(validation_data[inp][obj][val_step*batch_size : (val_step+1)*batch_size])
                else:
                    x_batch_of_input_val.append(validation_data[inp][obj][val_step*batch_size:])
            x_batch_val.append(x_batch_of_input_val)
        y_batch_val = validation_labels[val_step*batch_size : (val_step+1)*batch_size] if (val_step+1)*batch_size < validation_data[0][0].shape[0] - 1 else validation_labels[val_step*batch_size:]
        exec_val_step(x_batch_val, y_batch_val)

    val_acc = val_acc_metric.result()
    val_acc_metric.reset_states()
    print("Validation acc: %.4f" % (float(val_acc),))
    print("Time taken: %.2fs" % (time.time() - start_time))

    

### save model ###

# save classifier model
classifier.save('saved_models/eager_execution/classifier_model.h5')
print('saved classifier model')


### save model weights ###

# save classifier weights
classifier.save_weights('saved_models/eager_execution/classifier_weights.h5')
print('saved classifier weights')















