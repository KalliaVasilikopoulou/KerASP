import numpy as np

from project_utils.convert_to_categorical import custom_to_categorical as to_categorical


### Set Training Parameters ###

from user_scripts.training_configurations import training_configurations

epochs = training_configurations['epochs']
batch_size = training_configurations['batch_size']
validation_split = training_configurations['validation_split']

neurasp_conf = training_configurations['neurasp_conf']
classifiers_conf = neurasp_conf['classifiers_conf']

try: tokenize_data = training_configurations['tokenize_data']
except KeyError: tokenize_data = False

### Import Dataset If It Exists ###

try:
    import os
    classifier_inputs = {i: 0 for i in classifiers_conf}
    for _, _, files in os.walk('train_data/'):
        for file in files:    
            if file.startswith('train_data') and file.endswith('.npy'):
                ind_classifier = file.split('_classifier_', 1)[-1].split('.npy', 1)[0]
                classifier_inputs[ind_classifier] += 1

    train_data = {}
    train_obj_classes_labels = {}
    for ind_classifier in classifiers_conf:
        data_of_classifier = []
        for data_ind in range(classifier_inputs[ind_classifier]):
            data_of_classifier.append(np.load('train_data/train_data_'+str(data_ind)+'_classifier_'+str(ind_classifier)+'.npy'))
        train_data[ind_classifier] = data_of_classifier
        train_obj_classes_labels[ind_classifier] = np.load('train_data/train_obj_classes_labels_classifier_'+str(ind_classifier)+'.npy')
    train_labels = np.load('train_data/train_labels.npy')

    if tokenize_data:
        from tokenization.handle_tokenizers import load_tokenizers
        tokenizers = [str(i)+'_classifier_'+str(ind_classifier) for ind_classifier in classifiers_conf for i in range(classifier_inputs[ind_classifier])]
        load_tokenizers(tokenizers)

    print('Loaded train data and labels from corresponding files')
    
### If Dataset Does Not Exist, Create New Dataset ###
except FileNotFoundError:

    from create_dataset.create_dataset import train_data, train_obj_classes_labels, train_labels

    for ind_classifier in classifiers_conf:
        for data_ind, data in enumerate(train_data[ind_classifier]):
            np.save('train_data/train_data_'+str(data_ind)+'_classifier_'+str(ind_classifier)+'.npy', data)
        np.save('train_data/train_obj_classes_labels_classifier_'+str(ind_classifier)+'.npy', train_obj_classes_labels[ind_classifier])
    np.save('train_data/train_labels.npy', train_labels)

    if tokenize_data:
        from tokenization.handle_tokenizers import save_tokenizers
        from create_dataset.create_dataset import train_tokenizers_dict
        save_tokenizers(train_tokenizers_dict)

    print('Saved train data and labels to corresponding files')

### Pre-process Dataset Data And Labels ###

print('Imported train data and labels:')
for ind_classifier in classifiers_conf:
    for data_ind, data in enumerate(train_data[ind_classifier]):
        print('train_data classifier', ind_classifier, 'data', data_ind, 'shape:', data.shape)
    print('train_obj_classes_labels classifier', ind_classifier, 'labels shape:', train_obj_classes_labels[ind_classifier].shape)
print('train_labels labels shape:', train_labels.shape)

for ind_classifier in classifiers_conf:
    for i in range(len(train_data[ind_classifier])):    # i is index of class input (input for different data types), while j is index of object data
        data = train_data[ind_classifier][i]
        data = np.array_split(data, data.shape[1], axis=1)
        train_data[ind_classifier][i] = [np.squeeze(j, axis=1) for j in data]      # list conctaining classifier_inputs elements and each element is a list containing num_of_objects elements and each element has shape (samples, datapoint shape)

### Convert train data dict to list ###
train_data = list(train_data.values())

from find_interpretations.find_output_class_for_object_classes import find_all_output_classes_for_known_obj_classes
_, output_classes_list = find_all_output_classes_for_known_obj_classes(return_output_classes_list=True)

train_labels = to_categorical(train_labels, classes_list=output_classes_list)

print('train data classifiers:', len(train_data))                                # num of individual classifiers
print('train data classifier 1 inputs (list length):', len(train_data[0]))            # classifier_inputs
print('train data_k list length:', len(train_data[0][0]))                             # num_of_objects that go into classifier '1'
print('train data_k object_n shape:', train_data[0][0][0].shape)                      # (samples, datapoint shape)
print('train labels shape:', train_labels.shape)                                        # (samples, solver classes)

##train_data_draft = []
##for obj in all_objects:
    

### Import Model ###

from build_model.build_solver import Solver

solver_inst = Solver(compile_model=True)
solver_inst.model_obj_summary()
##solver_inst.visualize_model_obj()
solver_inst.plot_model_obj()

solver = solver_inst.model_obj
classifiers = solver_inst.model_obj_classifiers

### Train Model ###

import keras
stop_early = keras.callbacks.EarlyStopping(monitor='val_acc', patience=3)

# before executing this python file, go to an anaconda prompt, activate the environment with the tensorboard module, go to current path and type:
# tensorboard --logdir=tensorboard_logs

# tensorboard charts can be found at:
# localhost:6006
tensorboard = keras.callbacks.TensorBoard(log_dir='tensorboard_logs', histogram_freq=1, embeddings_freq=1)

### clean tensorboard logs from previous executions ###
from project_utils.clean_tensorboard_logs import clean_tensorboard_logs
clean_tensorboard_logs()

solver.fit(x=train_data, y=train_labels,
           epochs=epochs,
           batch_size=batch_size,
           verbose=1,
           validation_split=validation_split,
           callbacks=[stop_early,tensorboard])


### save model ###

# save solver model
solver.save('saved_models/graph_execution/solver/solver_model.h5')
print('saved solver model')

# save classifier model
for ind_classifier in classifiers_conf:
    classifiers[ind_classifier].save('saved_models/graph_execution/classifier/classifier_'+str(ind_classifier)+'_model.h5')
print('saved classifier model')


### save model weights ###

# save solver weights
solver.save_weights('saved_models/graph_execution/solver/solver_weights.h5')
print('saved solver weights')

# save classifier weights
for ind_classifier in classifiers_conf:
    classifiers[ind_classifier].save_weights('saved_models/graph_execution/classifier/classifier_'+str(ind_classifier)+'_weights.h5')
print('saved classifier weights')















