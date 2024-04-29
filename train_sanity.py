import numpy as np

from keras.utils import to_categorical


### Set Training Parameters ###

from training_configurations import training_configurations

epochs = training_configurations['epochs']
batch_size = training_configurations['batch_size']
validation_split = training_configurations['validation_split']

try: tokenize_data = training_configurations['tokenize_data']
except KeyError: tokenize_data = False


program_specs = training_configurations['program_specs']
list_of_object_classes = program_specs['list_of_object_classes']
classes_per_object = len(list_of_object_classes)


### Import Dataset If It Exists ###

try:
    import os
    classifier_inputs = 0
    if tokenize_data: tokenizers = 0
    
    for _, _, files in os.walk("train_data/"):
        for file in files:    
            if file.startswith('train_data') and file.endswith('.npy'):
                classifier_inputs += 1
        if tokenize_data:
            for file in files:
                if file.startswith('train_tokenizer') and file.endswith('.pickle'):
                    tokenizers += 1
    
    train_data = []
    for data_ind in range(classifier_inputs):
        train_data.append(np.load('train_data/train_data_'+str(data_ind)+'.npy'))
    train_obj_classes_labels = np.load('train_data/train_obj_classes_labels.npy')
    train_labels = np.load('train_data/train_labels.npy')

    if tokenize_data:
        from tokenization.handle_tokenizers import load_tokenizers
        tokenizers = [str(i) for i in range(tokenizers)]
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
        from tokenization.tokenize_data import train_tokenizers_dict
        save_tokenizers(train_tokenizers_dict)

    print('Saved train data and labels to corresponding files')

### Pre-process Dataset Images And Labels ###

print('Imported train data and labels:')
for data_ind, data in enumerate(train_data): 
    print('train_data data', data_ind, 'shape:', data.shape)            # every array has shape: (samples, num_of_objects, datapoint shape)
print('train_obj_classes_labels shape:', train_obj_classes_labels.shape)      # (samples, num_of_objects)
print('train_labels shape:', train_labels.shape)                        # (samples,)

for i in range(len(train_data)):    # i is index of class input (input for different data types), while j is index of object data
    train_data[i] = np.array_split(train_data[i], train_data[i].shape[1], axis=1)
    train_data[i] = [np.squeeze(j, axis=1) for j in train_data[i]]      # list conctaining classifier_inputs elements and each element is a list containing num_of_objects elements and each element has shape (samples, datapoint shape)


from find_interpretations.find_output_class_for_object_classes import find_all_output_classes_for_known_obj_classes
_, output_classes_list = find_all_output_classes_for_known_obj_classes(return_output_classes_list=True)

train_labels = to_categorical(train_labels, classes_list=output_classes_list)

print('train data list length:', len(train_data))                # classifier_inputs
print('train data_k list length:', len(train_data[0]))           # num_of_objects
print('train data_k object_n shape:', train_data[0][0].shape)    # (samples, datapoint shape)
print('train labels shape:', train_labels.shape)                 # (samples, solver classes)

### Import Model ###

from build_model.build_solver import Solver

solver_inst = Solver(compile_model=True)
solver_inst.model_obj_summary()
##solver_inst.visualize_model_obj()
solver_inst.plot_model_obj()

solver = solver_inst.model_obj
classifier = solver_inst.model_obj_classifier

### Train Model ###
import keras
stop_early = keras.callbacks.EarlyStopping(monitor='val_acc', patience=3)

# before executing this python file, go to an anaconda prompt, activate the environment with the tensorboard module, go to current path and type:
# tensorboard --logdir=tensorboard_logs

# tensorboard charts can be found at:
# localhost:6006
tensorboard = keras.callbacks.TensorBoard(log_dir='tensorboard_logs', histogram_freq=1, embeddings_freq=1)

train_obj_classes_labels = to_categorical(train_obj_classes_labels, num_classes=classes_per_object)

classifier.fit(x=train_data, y=train_obj_classes_labels,
           epochs=epochs,
           batch_size=batch_size,
           verbose=1,
           validation_split=validation_split,
           callbacks=[stop_early,tensorboard])

















