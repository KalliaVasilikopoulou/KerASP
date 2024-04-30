import numpy as np

from user_scripts.import_dataset import input_data, input_labels

### check if input data has correct structure ###
if isinstance(input_data, list):
    pass
elif isinstance(input_data, np.ndarray):
    input_data = [input_data]
else:
    raise Exception('Input data must be a numpy array or a list of numpy arrays')

classifier_inputs = len(input_data)


### OUTPUT CLASSES DATASET CREATION INITIATION ###

from training_configurations import training_configurations

program_specs = training_configurations['program_specs']

num_of_objects = program_specs['num_of_objects']
list_of_object_classes = program_specs['list_of_object_classes']
classes_per_object = len(list_of_object_classes)

try: tokenize_data = training_configurations['tokenize_data']
except KeyError: tokenize_data = False


data_organized_by_object_class = {obj_class:[] for obj_class in list_of_object_classes}

for i,label in enumerate(input_labels):
    if label in list_of_object_classes:
        data_organized_by_object_class[int(label)].append([data[i] for data in input_data])

print('\nTotal data (samples) per object class')
for obj_class in data_organized_by_object_class:
    print('Total data (samples) for object class', str(obj_class)+':', len(data_organized_by_object_class[obj_class]))

data_organized_by_object_class_pointer = {obj_class:0 for obj_class in list_of_object_classes}


from find_interpretations.find_output_class_for_object_classes import find_all_output_classes_for_known_obj_classes
output_classes, output_classes_list = find_all_output_classes_for_known_obj_classes(print_details=True, return_output_classes_list=True)

from project_utils.convert_number import number_to_base

train_data = [[] for _ in range(classifier_inputs)]
train_obj_classes_labels = []
train_labels = []

import itertools
all_object_classes_combs = [list(comb) for comb in itertools.product(list_of_object_classes, repeat=num_of_objects)]

import sys
samples = 0
print('\nDataset creation...')

while True:
    try:
        for ind, output_class_for_object_classes in enumerate(output_classes):
            if output_class_for_object_classes == -1: continue # if unsatisfiable, continue
            
            object_classes_comb = all_object_classes_combs[ind]

            sample_data = [[] for _ in range(classifier_inputs)]
            sample_obj_classes_labels = []
            
            for object_class in object_classes_comb:
                sample_data_for_obj_class = data_organized_by_object_class[object_class][data_organized_by_object_class_pointer[object_class]]
                data_organized_by_object_class_pointer[object_class] = data_organized_by_object_class_pointer[object_class]+1
                                                                      # if data_organized_by_object_class_pointer[object_class]+1 < len(data_organized_by_object_class_pointer[object_class]) else 0
                for i in range(classifier_inputs):
                    sample_data[i].append(sample_data_for_obj_class[i])
                sample_obj_classes_labels.append(object_class)

            for j in range(classifier_inputs):
                train_data[j].append(sample_data[j])
            train_obj_classes_labels.append(sample_obj_classes_labels)
            train_labels.append(output_class_for_object_classes)

            samples +=1
            sys.stdout.write("\rSamples created: %i" % samples)
            sys.stdout.flush()

    except IndexError:
        print('\n')
        break

train_data = [np.array(data) for data in train_data]            # every array has shape: (samples, num_of_objects, datapoint shape)
train_obj_classes_labels = np.array(train_obj_classes_labels)   # (samples, num_of_objects)
train_labels = np.array(train_labels)                           # (samples,)

### shuffle again ###

indices = np.arange(train_data[0].shape[0])
np.random.shuffle(indices)

train_data = [data[indices] for data in train_data]
train_obj_classes_labels = train_obj_classes_labels[indices]
train_labels = train_labels[indices]

for data_ind, data in enumerate(train_data):
    print('train_data data', data_ind, 'shape:', data.shape)
print('train_obj_classes_labels data shape:', train_obj_classes_labels.shape)
print('train_labels dataset shape:', train_labels.shape)


if tokenize_data:
    from tokenization.tokenize_data import tokenize_data
    train_data, train_tokenizers_dict = tokenize_data(train_data)



