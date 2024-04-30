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

program_specs = training_configurations['program_specs']

num_of_objects = program_specs['num_of_objects']
list_of_object_classes = program_specs['list_of_object_classes']
classes_per_object = len(list_of_object_classes)

object_type = program_specs['object_type']
output_type = program_specs['output_type']
classes_type = program_specs['classes_type']

from user_scripts.create_classifier import classifier

from find_interpretations.find_object_classes_for_output_class import find_all_obj_classes_for_known_output_classes

class Solver():
    def __init__(self, compile_model = False, compile_model_classifier = False):
        
        self.model_obj_classifier = models.clone_model(classifier)
        self.model_obj = self.build_model_obj()
        
        if compile_model:
            self.compile_model_obj()
        if compile_model_classifier:
            self.compile_model_obj_classifier()

    def build_model_obj(self):      # classifier is a keras model

        print('Building Keras model...')

        classifier_inputs = 0
        classifier_input_shapes = []
        
        for layer in self.model_obj_classifier.layers:
            if layer.name.startswith('input'):
                classifier_inputs +=1
                classifier_input_shapes.append(layer.output.shape[1:])

        if classifier_inputs > 1:
            solver_input = [[keras.Input(shape=classifier_input_shape, name='classifier_input_'+str(classifier_input)+'_of_'+object_type+'_'+str(object_i))
                             for classifier_input, classifier_input_shape in zip(range(classifier_inputs), classifier_input_shapes)]
                            for object_i in range(num_of_objects)]
        else:
            solver_input = [keras.Input(shape=classifier_input_shapes[0], name='input_of_'+object_type+'_'+str(object_i)) for object_i in range(num_of_objects)]
        
        classes_probs = [self.model_obj_classifier(i) for i in solver_input]      # [(None,classes_per_object),(None,classes_per_object),(None,classes_per_object)]


        def separate_probs(classes_probs):    # (None,classes_per_object)

            return [classes_probs[:,class_i] for class_i in range(classes_per_object)]      # [(None,),(None,),...,(None,)], length of list=classes_per_object

        probs = [layers.Lambda(separate_probs, name='separate_'+object_type+'_'+str(object_index)+'_probs')(object_probs) for object_index, object_probs in enumerate(classes_probs)]    # a list containing 3 lists, where each one of these 3 lists contains classes_per_object number of (None,) tensors


        objects_combinations_of_all_output_classes, output_classes_list = find_all_obj_classes_for_known_output_classes(return_output_classes_list=True)
        
        combinations_probs_of_all_output_classes = [[layers.Multiply(name=output_type+'_'+str(output_class_k)+'_'+object_type+'_comb_no_'+str(comb_ind+1)+'_prob')([probs[i][list_of_object_classes.index(j)] for i,j in enumerate(comb)])  # i = object, j = class of object
                                                 for comb_ind,comb in enumerate(objects_combinations_of_output_class_k)]
                                                for output_class_k, objects_combinations_of_output_class_k in zip(output_classes_list, objects_combinations_of_all_output_classes)]   # [(None,), (None,),...,(None,)], length of list = num of interpretations

        probs_of_all_output_classes = [layers.Add(name=output_type+'_'+str(output_class_k)+'_prob')(combinations_probs_of_output_class_k) for output_class_k, combinations_probs_of_output_class_k in zip(output_classes_list, combinations_probs_of_all_output_classes)]       # [(None,),...,], length = all possible values = satisfiable values

        reshape_layer = layers.Reshape((1,), name=output_type+'_k_prob_reshaped')
        probs_of_all_output_classes = [reshape_layer(probs_of_output_class_k) for probs_of_output_class_k in probs_of_all_output_classes]   # (None, satisfiable values)
        
        solver_output = layers.Concatenate(axis=-1)(probs_of_all_output_classes)   # (None, satisfiable values)
        
        solver = models.Model(solver_input, solver_output, name='solver')

        print('Keras model building is completed.')

        return solver

    def model_obj_summary(self, include_classifier=True):
        if include_classifier:
            self.model_obj_classifier.summary()
        self.model_obj.summary()

        return None

    def plot_model_obj(self, include_classifier=True):
        if include_classifier:
            # save to .png file
            plot_model(self.model_obj_classifier, show_shapes=True, to_file='build_model/classifier_model.png')
            # save to .pdf file
            plot_model(self.model_obj_classifier, show_shapes=True, to_file='build_model/classifier_model.pdf')
        # save to .png file
        plot_model(self.model_obj, show_shapes=True, to_file='build_model/solver_model.png')
        # save to .pdf file
        plot_model(self.model_obj, show_shapes=True, rankdir="LR", dpi=700, to_file='build_model/solver_model.pdf')

        return None

    def visualize_model_obj(self, include_classifier=True):
        if include_classifier:
            visualizer(self.model_obj_classifier, file_name='build_model/classifier_model_visualization', file_format='png', view=True)
        visualizer(self.model_obj, file_name='build_model/solver_model_visualization', file_format='png', view=True)

    
    def compile_model_obj(self):
        # compiler params are fixed - should not be easy-to-change parameters
        optimizer = optimizers.Adam(learning_rate=learning_rate, decay=decay)
        self.model_obj.compile(optimizer=optimizer,
                               loss='categorical_crossentropy',
                               metrics=['acc'])

    def compile_model_obj_classifier(self):
        # compiler params are fixed - should not be easy-to-change parameters
        optimizer = optimizers.Adam(learning_rate=learning_rate, decay=decay)
        self.model_obj_classifier.compile(optimizer=optimizer,
                                          loss='categorical_crossentropy',
                                          metrics=['acc'])
        


