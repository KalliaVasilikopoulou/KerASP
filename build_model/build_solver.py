import numpy as np

import keras
from keras import layers, models, optimizers
from keras import backend as K

from keras.utils import plot_model
from keras_visualizer import visualizer

from training_configurations import training_configurations

dropout_rate = training_configurations['dropout_rate']
learning_rate = training_configurations['learning_rate']
decay = training_configurations['decay']


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

output_type = neurasp_conf['output_type']

try: from user_scripts.create_classifiers import classifiers
except ImportError:
    from user_scripts.create_classifiers import classifier
    classifiers = {'1': classifier}

from find_interpretations.find_object_classes_for_output_class import find_all_obj_classes_for_known_output_classes

class Solver():
    def __init__(self, compile_model = False, compile_model_classifiers = False):

        self.model_obj_classifiers = classifiers
        self.model_obj = self.build_model_obj()
        
        if compile_model:
            self.compile_model_obj()
        if compile_model_classifiers:
            self.compile_model_obj_classifiers()

    def build_model_obj(self):      # classifier is a keras model

        print('Building Keras model...')

        #solver_input = [[] for _ in range(len(classifiers_conf.items()))]
        solver_input = {i: [] for i in classifiers_conf}
        classes_probs = []

        for obj in all_objects:

            ind_classifier = classifier_of_object[obj]
            classifier_x = self.model_obj_classifiers[ind_classifier]

            classifier_inputs = 0
            classifier_input_shapes = []
            
            for layer in classifier_x.layers:
                if layer.name.startswith('input'):
                    classifier_inputs +=1
                    classifier_input_shapes.append(layer.output.shape[1:])

            if classifier_inputs > 1:
                obj_input = [keras.Input(shape=classifier_input_shape, name='classifier_'+str(ind_classifier)+'_input_'+str(classifier_input)+'_of_'+classifiers_conf[ind_classifier]['object_type']+'_'+str(obj))
                                         for classifier_input, classifier_input_shape in zip(range(classifier_inputs), classifier_input_shapes)]
            else:
                obj_input = keras.Input(shape=classifier_input_shapes[0], name='classifier_'+str(ind_classifier)+'_input_of_'+classifiers_conf[ind_classifier]['object_type']+'_'+str(obj))

            solver_input[ind_classifier].append(obj_input)
            
            obj_probs = classifier_x(obj_input)
            classes_probs.append(obj_probs)      # [(None,classes_per_object),(None,classes_per_object),(None,classes_per_object)]

        def separate_probs(classes_probs):    # (None,classes_per_object)

            return [classes_probs[:,class_i] for class_i in range(K.int_shape(classes_probs)[-1])]      # [(None,),(None,),...,(None,)], length of list=classes_per_object

        probs = [layers.Lambda(separate_probs, name='separate_'+classifiers_conf[classifier_of_object[object_x]]['object_type']+'_object_'+str(object_x)+'_'+classifiers_conf[classifier_of_object[object_x]]['classes_type']+'_probs')(object_probs) for object_x, object_probs in zip(all_objects, classes_probs)]    # a list containing 3 lists, where each one of these 3 lists contains classes_per_object number of (None,) tensors


        objects_combinations_of_all_output_classes, output_classes_list = find_all_obj_classes_for_known_output_classes(return_output_classes_list=True)
        
        combinations_probs_of_all_output_classes = [[layers.Multiply(name=output_type+'_'+str(output_class_k)+'_comb_no_'+str(comb_ind+1)+'_prob')([probs[obj][classes_of_object[obj].index(class_x)] for obj,class_x in enumerate(comb)])  # obj = object, class_x = class of object
                                                 for comb_ind,comb in enumerate(objects_combinations_of_output_class_k)]
                                                for output_class_k, objects_combinations_of_output_class_k in zip(output_classes_list, objects_combinations_of_all_output_classes)]   # [(None,), (None,),...,(None,)], length of list = num of interpretations

        probs_of_all_output_classes = [layers.Add(name=output_type+'_'+str(output_class_k)+'_prob')(combinations_probs_of_output_class_k) for output_class_k, combinations_probs_of_output_class_k in zip(output_classes_list, combinations_probs_of_all_output_classes)]       # [(None,),...,], length = all possible values = satisfiable values

        reshape_layer = layers.Reshape((1,), name=output_type+'_k_prob_reshaped')
        probs_of_all_output_classes = [reshape_layer(probs_of_output_class_k) for probs_of_output_class_k in probs_of_all_output_classes]   # (None, satisfiable values)
        
        solver_output = layers.Concatenate(axis=-1)(probs_of_all_output_classes)   # (None, satisfiable values)
        
        solver = models.Model(solver_input, solver_output, name='solver')

        print('Keras model building is completed.')

        return solver

    def model_obj_summary(self, include_classifiers=True):
        if include_classifiers:
            for ind_classifier in classifiers_conf:
                self.model_obj_classifiers[ind_classifier].summary()
        self.model_obj.summary()

        return None

    def plot_model_obj(self, include_classifiers=True):
        if include_classifiers:
            for ind_classifier in classifiers_conf:
                # save to .png file
                plot_model(self.model_obj_classifiers[ind_classifier], show_shapes=True, to_file='build_model/classifier_'+str(ind_classifier)+'_model.png')
                # save to .pdf file
                plot_model(self.model_obj_classifiers[ind_classifier], show_shapes=True, to_file='build_model/classifier_'+str(ind_classifier)+'_model.pdf')
        # save to .png file
        plot_model(self.model_obj, show_shapes=True, to_file='build_model/solver_model.png')
        # save to .pdf file
        plot_model(self.model_obj, show_shapes=True, rankdir="LR", dpi=700, to_file='build_model/solver_model.pdf')

        return None

    def visualize_model_obj(self, include_classifiers=True):
        if include_classifiers:
            for ind_classifier in classifiers_conf:
                visualizer(self.model_obj_classifiers[ind_classifier], file_name='build_model/classifier_'+str(ind_classifier)+'_model_visualization', file_format='png', view=True)
        visualizer(self.model_obj, file_name='build_model/solver_model_visualization', file_format='png', view=True)

    
    def compile_model_obj(self):
        # compiler params are fixed - should not be easy-to-change parameters
        optimizer = optimizers.Adam(learning_rate=learning_rate, decay=decay)
        self.model_obj.compile(optimizer=optimizer,
                               loss='categorical_crossentropy',
                               metrics=['acc'])

    def compile_model_obj_classifiers(self):
        # compiler params are fixed - should not be easy-to-change parameters
        optimizer = optimizers.Adam(learning_rate=learning_rate, decay=decay)
        for ind_classifier in classifiers_conf:
            self.model_obj_classifiers[ind_classifier].compile(optimizer=optimizer,
                                                               loss='categorical_crossentropy',
                                                               metrics=['acc'])
        


