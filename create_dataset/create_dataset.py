import numpy as np

try: from user_scripts.import_dataset import data_dict
except ImportError:
    from user_scripts.import_dataset import input_data, input_labels
    data_dict = {'1':{'input_data': input_data, 'input_labels': input_labels}}


### check if input data has correct structure ###

correct_structure = '''
                    {
                        1: {'input_data': np.ar, 'input_labels': np.ar},
                        2: {'input_data': [np.ar, np.ar, np.ar, ...], 'input_labels': np.ar}
                        3: ...
                    }
                    '''

if isinstance(data_dict, dict):
    for ind_classifier in data_dict:
        if isinstance(data_dict[ind_classifier], dict):
            if isinstance(data_dict[ind_classifier]['input_data'], list):
                if isinstance(data_dict[ind_classifier]['input_data'][0], np.ndarray):
                    pass
                else:
                    raise Exception('Please submit your data according to this structure:\n'+correct_structure)
            elif isinstance(data_dict[ind_classifier]['input_data'], np.ndarray):
                data_dict[ind_classifier]['input_data'] = [data_dict[ind_classifier]['input_data']]
            else:
                raise Exception('Please submit your data according to this structure:\n'+correct_structure)
            if isinstance(data_dict[ind_classifier]['input_labels'], np.ndarray):
                pass
            else:
                raise Exception('Please submit your data according to this structure:\n'+correct_structure)
            if isinstance(data_dict[ind_classifier]['input_data'][0], np.ndarray):
                pass
        else:
            raise Exception('Please submit your data according to this structure:\n'+correct_structure)
else:
    raise Exception('Please submit your data according to this structure:\n'+correct_structure)


### OUTPUT CLASSES DATASET CREATION INITIATION ###

from training_configurations import training_configurations

neurasp_conf = training_configurations['neurasp_conf']
classifiers_conf = neurasp_conf['classifiers_conf']
# sort classifiers_conf dictionary
classifiers_conf = dict(sorted(classifiers_conf.items()))

objects_of_classifier = {i: sorted(classifiers_conf[i]['objects']) for i in classifiers_conf}
classes_of_classifier = {i: sorted(classifiers_conf[i]['list_of_object_classes']) for i in classifiers_conf}

all_objects = [obj for i in classifiers_conf for obj in objects_of_classifier[i]]       # we must not have duplicates
all_objects.sort()

classifier_of_object = {}
classes_of_object = {}
for obj in all_objects:
    for i in classifiers_conf:
        if obj in objects_of_classifier[i]:
            classifier_of_object[obj] = i
            classes_of_object[obj] = classes_of_classifier[i]

try: tokenize_data = training_configurations['tokenize_data']
except KeyError: tokenize_data = False

classifier_inputs = {ind_classifier: len(data_dict[ind_classifier]['input_data']) for ind_classifier in classifiers_conf}


#{1: [[],[]], 2: [[],[]]}
data_organized_by_class = {i: {classifier_class:[] for classifier_class in classes_of_classifier[i]} for i in classifiers_conf}

for ind_classifier in classifiers_conf:
    for i,label in enumerate(data_dict[ind_classifier]['input_labels']):
        if label in classes_of_classifier[ind_classifier]:
            data_organized_by_class[ind_classifier][int(label)].append([data[i] for data in data_dict[ind_classifier]['input_data']])

for ind_classifier in classifiers_conf:
    print('\nTotal data (samples) per class of classifier', ind_classifier)
    for classifier_class in data_organized_by_class[ind_classifier]:
        print('\tTotal data (samples) for class', str(classifier_class)+':', len(data_organized_by_class[ind_classifier][classifier_class]))

data_organized_by_class_pointer = {i: {obj_class:0 for obj_class in classes_of_classifier[i]} for i in classifiers_conf}


from find_interpretations.find_output_class_for_object_classes import find_all_output_classes_for_known_obj_classes
output_classes, output_classes_list = find_all_output_classes_for_known_obj_classes(print_details=True, return_output_classes_list=True)

from project_utils.convert_number import number_to_base

train_data = {i: [[] for _ in range(classifier_inputs[i])] for i in classifiers_conf}
train_obj_classes_labels = {i: [] for i in classifiers_conf}
train_labels = []

import itertools
all_object_classes_combs = list(itertools.product(*[classes_of_object[obj] for obj in all_objects]))

import sys
samples = 0
print('\nDataset creation...')

while True:
    try:
        for ind, output_class_for_object_classes in enumerate(output_classes):
            if output_class_for_object_classes == -1: continue # if unsatisfiable, continue
            
            object_classes_comb = all_object_classes_combs[ind]

            sample_data = {i: [[] for _ in range(classifier_inputs[i])] for i in classifiers_conf}
            sample_obj_classes_labels = {i: [] for i in classifiers_conf}
            
            for obj, class_of_obj in zip(all_objects, object_classes_comb):
                ind_classifier = classifier_of_object[obj]
                
                sample_data_for_obj_class = data_organized_by_class[ind_classifier][class_of_obj][data_organized_by_class_pointer[ind_classifier][class_of_obj]]
                data_organized_by_class_pointer[ind_classifier][class_of_obj] = data_organized_by_class_pointer[ind_classifier][class_of_obj]+1

                for i in range(classifier_inputs[ind_classifier]):
                    sample_data[ind_classifier][i].append(sample_data_for_obj_class[i])
                sample_obj_classes_labels[ind_classifier].append(class_of_obj)

            for ind_classifier in classifiers_conf:
                for j in range(classifier_inputs[ind_classifier]):
                    train_data[ind_classifier][j].append(sample_data[ind_classifier][j])
                train_obj_classes_labels[ind_classifier].append(sample_obj_classes_labels[ind_classifier])
            train_labels.append(output_class_for_object_classes)

            samples +=1
            sys.stdout.write("\rSamples created: %i" % samples)
            sys.stdout.flush()

    except IndexError:
        print('\n')
        break

train_data = {ind_classifier: [np.array(data) for data in train_data[ind_classifier]] for ind_classifier in classifiers_conf}    # every array has shape: (samples, num_of_objects, datapoint shape)
train_obj_classes_labels = {ind_classifier: np.array(train_obj_classes_labels[ind_classifier]) for ind_classifier in classifiers_conf}   # (samples, num_of_objects)
train_labels = np.array(train_labels)                           # (samples,)

### shuffle again ###

ind_classifier = '1'
ind_data = 0
indices = np.arange(train_data[ind_classifier][ind_data].shape[0])
np.random.shuffle(indices)

train_data = {ind_classifier: [data[indices] for data in train_data[ind_classifier]] for ind_classifier in classifiers_conf}
train_obj_classes_labels = {ind_classifier: train_obj_classes_labels[ind_classifier][indices] for ind_classifier in classifiers_conf}
train_labels = train_labels[indices]

for ind_classifier in classifiers_conf:
    for data_ind, data in enumerate(train_data[ind_classifier]):
        print('train_data classifier', ind_classifier, 'data', data_ind, 'shape:', data.shape)
    print('train_obj_classes_labels classifier', ind_classifier, 'labels shape:', train_obj_classes_labels[ind_classifier].shape)
print('train_labels labels shape:', train_labels.shape)


if tokenize_data:
    from tokenization.tokenize_data import tokenize_data
    train_data, train_tokenizers_dict = tokenize_data(train_data)





